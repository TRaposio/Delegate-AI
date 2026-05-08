"""Indexing entry point: chunks.jsonl → embeddings.npy + chunk_ids.json + sidecar.

Run: `python -m wca_rag.index`

Reads `data/chunks.jsonl`, embeds each chunk's `text_for_embedding`, writes:
- `data/embeddings.npy`        : (n_chunks, embedding_dim) float32, L2-normalized
- `data/chunk_ids.json`        : list of regulation_ids, parallel to embeddings rows
- `data/embeddings.meta.json`  : sidecar with model + corpus fingerprint

The fingerprint (`chunks_text_hash`) is a sha1 over the sorted per-chunk
`text_hash` values from chunks.jsonl. If the parser is re-run and any chunk
text changes, the fingerprint changes — that's the signal that embeddings
are stale and need rebuilding.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from wca_rag.embedder import Embedder, SentenceTransformerEmbedder


DEFAULT_CHUNKS_PATH = Path("data/chunks.jsonl")
DEFAULT_OUT_DIR = Path("data")
EMBEDDER_VERSION = "1.0"


def load_chunks(path: Path) -> list[dict]:
    """Read chunks.jsonl into a list of dicts, preserving file order."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the parser first: python -m wca_rag.parser"
        )
    with path.open("r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    if not chunks:
        raise ValueError(f"{path} is empty.")
    return chunks


def compute_corpus_fingerprint(chunks: list[dict]) -> str:
    """Sha1 over sorted per-chunk text_hash values.

    Order-independent (sorted) so reordering chunks doesn't trigger a false
    'stale' signal. Hashes the *content* fingerprints, not the embeddings or
    the embedding text — the parser already decided what counts as a content
    change.
    """
    hashes = sorted(c["text_hash"] for c in chunks)
    return hashlib.sha1("\n".join(hashes).encode("utf-8")).hexdigest()


def build_index(
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    out_dir: Path = DEFAULT_OUT_DIR,
    embedder: Embedder | None = None,
) -> None:
    chunks = load_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    if embedder is None:
        print("Loading embedder (first run downloads ~130MB to ~/.cache/huggingface/)")
        embedder = SentenceTransformerEmbedder()

    texts = [c["text_for_embedding"] for c in chunks]
    chunk_ids = [c["regulation_id"] for c in chunks]

    print(f"Embedding {len(texts)} chunks with {embedder.model_name}...")
    embeddings = embedder.encode_documents(texts)

    # Sanity checks before writing — cheaper to fail here than silently produce
    # a broken index.
    assert embeddings.shape == (len(chunks), embedder.embedding_dim), (
        f"shape mismatch: {embeddings.shape} vs ({len(chunks)}, {embedder.embedding_dim})"
    )
    assert embeddings.dtype == np.float32, f"dtype: {embeddings.dtype}"
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"embeddings not L2-normalized: norms range {norms.min():.4f}..{norms.max():.4f}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = out_dir / "embeddings.npy"
    chunk_ids_path = out_dir / "chunk_ids.json"
    meta_path = out_dir / "embeddings.meta.json"

    np.save(embeddings_path, embeddings)
    with chunk_ids_path.open("w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, indent=2)

    meta = {
        "model_name": embedder.model_name,
        "embedding_dim": embedder.embedding_dim,
        "normalized": True,
        "n_chunks": len(chunks),
        "chunks_text_hash": compute_corpus_fingerprint(chunks),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "embedder_version": EMBEDDER_VERSION,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {embeddings_path} ({embeddings.shape}, {embeddings.dtype})")
    print(f"Wrote {chunk_ids_path} ({len(chunk_ids)} ids)")
    print(f"Wrote {meta_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks", type=Path, default=DEFAULT_CHUNKS_PATH,
        help=f"Path to chunks.jsonl (default: {DEFAULT_CHUNKS_PATH})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--model", type=str, default="BAAI/bge-small-en-v1.5",
        help="sentence-transformers model name",
    )
    args = parser.parse_args(argv)

    embedder = SentenceTransformerEmbedder(model_name=args.model)
    build_index(chunks_path=args.chunks, out_dir=args.out_dir, embedder=embedder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
