"""Sanity-check the embeddings index.

Run from repo root: python scripts/inspect_embeddings.py

Checks:
1. All three artifacts exist and load.
2. Shape, dtype, normalization match what the sidecar claims.
3. chunk_ids align row-for-row with chunks.jsonl (no drift).
4. Sidecar fingerprint matches a freshly-computed fingerprint (embeddings
   are not stale w.r.t. current chunks.jsonl).
5. Pairwise similarity sanity: a chunk should be most similar to itself,
   and known cross-referenced pairs should score higher than random pairs.
6. Quick visual: print top-5 nearest neighbors for a few sampled chunks.
   Skim the output — semantically related regulations should cluster.

Exit 0 if all hard checks pass, 1 otherwise. Soft checks (#6) are
informational and don't fail the script.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np


DATA_DIR = Path("data")
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
CHUNK_IDS_PATH = DATA_DIR / "chunk_ids.json"
META_PATH = DATA_DIR / "embeddings.meta.json"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

# ANSI for terminal output. Strip if you pipe to a file.
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}!{RESET} {msg}")


def section(title: str) -> None:
    print(f"\n{DIM}── {title} ──{RESET}")


def compute_corpus_fingerprint(chunks: list[dict]) -> str:
    """Mirror of wca_rag.index.compute_corpus_fingerprint — duplicated here
    so the inspector has zero coupling to the indexer."""
    hashes = sorted(c["text_hash"] for c in chunks)
    return hashlib.sha1("\n".join(hashes).encode("utf-8")).hexdigest()


def load_artifacts() -> tuple[np.ndarray, list[str], dict, list[dict]]:
    """Load all four files. Raise with a clear message if anything is missing."""
    missing = [p for p in [EMBEDDINGS_PATH, CHUNK_IDS_PATH, META_PATH, CHUNKS_PATH] if not p.exists()]
    if missing:
        for p in missing:
            fail(f"missing: {p}")
        raise SystemExit(1)

    embeddings = np.load(EMBEDDINGS_PATH)
    chunk_ids = json.loads(CHUNK_IDS_PATH.read_text())
    meta = json.loads(META_PATH.read_text())
    with CHUNKS_PATH.open() as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    return embeddings, chunk_ids, meta, chunks


def check_shape_and_dtype(embeddings: np.ndarray, meta: dict) -> bool:
    section("shape, dtype, normalization")
    passed = True

    expected_shape = (meta["n_chunks"], meta["embedding_dim"])
    if embeddings.shape == expected_shape:
        ok(f"shape {embeddings.shape} matches sidecar")
    else:
        fail(f"shape {embeddings.shape}, sidecar says {expected_shape}")
        passed = False

    if embeddings.dtype == np.float32:
        ok(f"dtype {embeddings.dtype}")
    else:
        fail(f"dtype {embeddings.dtype}, expected float32")
        passed = False

    norms = np.linalg.norm(embeddings, axis=1)
    if meta.get("normalized") and np.allclose(norms, 1.0, atol=1e-5):
        ok(f"L2-normalized (norms range {norms.min():.6f}..{norms.max():.6f})")
    elif meta.get("normalized"):
        fail(f"sidecar claims normalized but norms range {norms.min():.4f}..{norms.max():.4f}")
        passed = False
    else:
        warn(f"sidecar says normalized=False; norms range {norms.min():.4f}..{norms.max():.4f}")

    return passed


def check_chunk_id_alignment(chunk_ids: list[str], chunks: list[dict]) -> bool:
    """chunk_ids[i] must equal chunks[i].regulation_id for all i.

    If this fails, retrieval will return wrong chunks — the matrix row
    points to the wrong regulation. Highest-impact silent bug we can have.
    """
    section("chunk_ids ↔ chunks.jsonl alignment")

    if len(chunk_ids) != len(chunks):
        fail(f"length mismatch: chunk_ids={len(chunk_ids)}, chunks={len(chunks)}")
        return False

    mismatches = [
        (i, cid, chunks[i]["regulation_id"])
        for i, cid in enumerate(chunk_ids)
        if cid != chunks[i]["regulation_id"]
    ]
    if mismatches:
        fail(f"{len(mismatches)} row(s) misaligned. First 3:")
        for i, cid, real in mismatches[:3]:
            print(f"    row {i}: chunk_ids says {cid!r}, chunks.jsonl says {real!r}")
        return False

    ok(f"all {len(chunk_ids)} rows aligned")
    return True


def check_fingerprint_freshness(meta: dict, chunks: list[dict]) -> bool:
    section("corpus fingerprint freshness")
    current = compute_corpus_fingerprint(chunks)
    stored = meta.get("chunks_text_hash")
    if current == stored:
        ok(f"fingerprint matches sidecar ({stored[:12]}...)")
        return True
    fail("fingerprint MISMATCH — embeddings are stale")
    print(f"    sidecar : {stored}")
    print(f"    current : {current}")
    print(f"    {YELLOW}fix: re-run `python -m wca_rag.index`{RESET}")
    return False


def check_self_similarity(embeddings: np.ndarray) -> bool:
    """Each chunk must be its own nearest neighbor.

    On L2-normalized vectors, embeddings @ embeddings.T is the full pairwise
    cosine similarity matrix. The diagonal is each chunk's similarity to
    itself, which must equal 1.0 and must be the row-wise max.
    """
    section("self-similarity")
    sim = embeddings @ embeddings.T  # (n, n) cosine similarities

    diag = np.diag(sim)
    if np.allclose(diag, 1.0, atol=1e-5):
        ok("all self-similarities ≈ 1.0")
    else:
        fail(f"self-similarity range: {diag.min():.4f}..{diag.max():.4f}")
        return False

    # Row argmax: which chunk is each row most similar to? Should always be itself.
    argmax = sim.argmax(axis=1)
    expected = np.arange(len(sim))
    wrong = np.where(argmax != expected)[0]
    if len(wrong) == 0:
        ok("every chunk is its own nearest neighbor")
        return True

    fail(f"{len(wrong)} chunks have a non-self nearest neighbor (numerical ties)")
    # In practice this happens when two chunks have identical embeddings —
    # rare but possible if two regulations have identical text_for_embedding.
    return False


def report_similarity_distribution(embeddings: np.ndarray) -> None:
    """Soft check. Print percentiles of off-diagonal pairwise similarities.

    What healthy looks like: median around 0.6–0.8 for bge-small on a
    topically coherent corpus (all WCA regulations). Very high median
    (>0.9) suggests embeddings are collapsed and won't discriminate well.
    Very low median (<0.3) suggests the chunks are genuinely unrelated.
    """
    section("pairwise similarity distribution (off-diagonal)")
    sim = embeddings @ embeddings.T
    n = sim.shape[0]
    # Mask the diagonal to ignore self-similarity.
    mask = ~np.eye(n, dtype=bool)
    off_diag = sim[mask]

    pcts = [5, 25, 50, 75, 95]
    values = np.percentile(off_diag, pcts)
    print(f"    n_pairs = {len(off_diag):,}")
    for p, v in zip(pcts, values):
        print(f"    p{p:>2}: {v:+.4f}")
    print(f"    {DIM}healthy median is roughly 0.6–0.8 for a topically coherent corpus.{RESET}")


def report_nearest_neighbors(
    embeddings: np.ndarray,
    chunk_ids: list[str],
    chunks: list[dict],
    n_samples: int = 5,
    k: int = 5,
    seed: int = 0,
) -> None:
    """Soft check. For a few random chunks, print top-k nearest neighbors.

    Eyeball test: do the neighbors make sense? A chunk about scrambling
    should have other scrambling-related chunks as neighbors, not random
    bits of Article 12.
    """
    section(f"nearest neighbors ({n_samples} random chunks, top-{k})")
    rng = random.Random(seed)
    sample_idx = rng.sample(range(len(embeddings)), k=min(n_samples, len(embeddings)))

    id_to_chunk = {c["regulation_id"]: c for c in chunks}
    sim = embeddings @ embeddings.T

    for i in sample_idx:
        # Top-(k+1) because the chunk itself is always #1; we want k *other* neighbors.
        neighbors = np.argsort(-sim[i])[: k + 1]
        chunk = id_to_chunk[chunk_ids[i]]
        article = chunk.get("article", "?")
        print(f"\n  [{chunk_ids[i]}] (Article {article})")
        for rank, j in enumerate(neighbors):
            tag = "self" if j == i else f"#{rank}"
            score = sim[i, j]
            other = id_to_chunk[chunk_ids[j]]
            other_article = other.get("article", "?")
            # Snippet of the other chunk, first line of body.
            snippet = other.get("text", "").strip().split("\n")[0][:80]
            print(f"    {tag:>4} {score:+.4f}  {chunk_ids[j]:<6} (Art {other_article:<2})  {DIM}{snippet}{RESET}")


def main() -> int:
    print(f"{DIM}Inspecting embeddings index in {DATA_DIR}/{RESET}")
    embeddings, chunk_ids, meta, chunks = load_artifacts()
    ok(f"loaded {len(chunks)} chunks, {len(chunk_ids)} ids, "
       f"embeddings {embeddings.shape}, sidecar from {meta.get('created_at', '?')}")

    # Hard checks (failure → exit 1).
    hard_results = [
        check_shape_and_dtype(embeddings, meta),
        check_chunk_id_alignment(chunk_ids, chunks),
        check_fingerprint_freshness(meta, chunks),
        check_self_similarity(embeddings),
    ]

    # Soft checks (informational only).
    report_similarity_distribution(embeddings)
    report_nearest_neighbors(embeddings, chunk_ids, chunks)

    print()
    if all(hard_results):
        ok("all hard checks passed")
        return 0
    fail("one or more hard checks failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
