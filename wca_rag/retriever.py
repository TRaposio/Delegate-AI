"""Retriever: query → top-k chunks.

Loads the persisted index (embeddings.npy + chunk_ids.json + chunks.jsonl)
and answers `retrieve(query, k)` calls. The actual retrieval math is one
matrix multiply: `embeddings @ query_vec` produces 108 cosine similarity
scores; we argsort and slice.

This is the bi-encoder consumer side. The embedder produced
L2-normalized vectors at index time; we expect normalized vectors here
too, so dot product == cosine similarity directly.

For the WCA corpus (108 chunks) we do brute-force search over the full
matrix. At this scale there is nothing faster than a single matrix
multiply. ChromaDB will replace this when (a) we want metadata filtering
or (b) the corpus grows past ~10k chunks. Until then, dependency-free
NumPy is the right tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wca_rag.embedder import Embedder, SentenceTransformerEmbedder


DEFAULT_DATA_DIR = Path("data")
DEFAULT_K = 5


@dataclass
class RetrievalHit:
    """One retrieved chunk with its similarity score.

    The score is cosine similarity in [-1, 1]. On normalized embeddings
    with bge-small, expect typical hits in roughly [0.4, 0.85] — see
    scripts/inspect_embeddings.py for distribution stats.
    """
    rank: int
    score: float
    regulation_id: str
    chunk: dict  # full chunk dict from chunks.jsonl

    def __repr__(self) -> str:
        return f"RetrievalHit(rank={self.rank}, score={self.score:.4f}, id={self.regulation_id!r})"


class Retriever:
    """In-memory brute-force retriever over a persisted embedding index.

    Loads three artifacts at construction time:
    - embeddings.npy   : (n, dim) float32, L2-normalized
    - chunk_ids.json   : list[str], parallel to embeddings rows
    - chunks.jsonl     : full chunk dicts, indexed by regulation_id

    Then `retrieve(query, k)` embeds the query and returns top-k hits.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        chunk_ids: list[str],
        chunks_by_id: dict[str, dict],
        embedder: Embedder,
    ) -> None:
        # Defensive checks: failing here is much more debuggable than
        # silently retrieving the wrong chunk.
        if embeddings.shape[0] != len(chunk_ids):
            raise ValueError(
                f"row count mismatch: embeddings has {embeddings.shape[0]} rows, "
                f"chunk_ids has {len(chunk_ids)}"
            )
        missing = [cid for cid in chunk_ids if cid not in chunks_by_id]
        if missing:
            raise ValueError(
                f"{len(missing)} chunk_id(s) not found in chunks.jsonl: {missing[:5]}..."
            )
        if embeddings.shape[1] != embedder.embedding_dim:
            raise ValueError(
                f"dim mismatch: embeddings dim={embeddings.shape[1]}, "
                f"embedder dim={embedder.embedding_dim} — index built with a different model?"
            )

        self.embeddings = embeddings
        self.chunk_ids = chunk_ids
        self.chunks_by_id = chunks_by_id
        self.embedder = embedder

    @classmethod
    def from_disk(
        cls,
        data_dir: Path = DEFAULT_DATA_DIR,
        embedder: Embedder | None = None,
    ) -> "Retriever":
        """Load all artifacts from disk and build a ready-to-use retriever.

        If `embedder` is None, instantiate the default SentenceTransformerEmbedder.
        Pass an embedder explicitly when you already have one loaded (saves
        the ~1s model load) or when injecting a fake for tests.
        """
        data_dir = Path(data_dir)
        embeddings_path = data_dir / "embeddings.npy"
        chunk_ids_path = data_dir / "chunk_ids.json"
        chunks_path = data_dir / "chunks.jsonl"

        for p in (embeddings_path, chunk_ids_path, chunks_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"{p} not found. Run `python -m wca_rag.index` first."
                )

        embeddings = np.load(embeddings_path)
        chunk_ids = json.loads(chunk_ids_path.read_text(encoding="utf-8"))

        chunks_by_id: dict[str, dict] = {}
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                chunks_by_id[chunk["regulation_id"]] = chunk

        if embedder is None:
            embedder = SentenceTransformerEmbedder()

        return cls(
            embeddings=embeddings,
            chunk_ids=chunk_ids,
            chunks_by_id=chunks_by_id,
            embedder=embedder,
        )

    def retrieve(self, query: str, k: int = DEFAULT_K) -> list[RetrievalHit]:
        """Embed query, score against all chunks, return top-k.

        On normalized embeddings, `embeddings @ query_vec` IS cosine similarity.
        argsort gives indices in ascending order; we take the last k and
        reverse for descending.
        """
        if not query or not query.strip():
            raise ValueError("query is empty")
        k = min(k, len(self.chunk_ids))
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        query_vec = self.embedder.encode_query(query)            # (dim,)
        scores = self.embeddings @ query_vec                     # (n,)
        # argsort ascending → take the last k → reverse for descending.
        # argpartition would be faster for large n, but for n=108 the
        # difference is microseconds and full argsort is more obvious.
        top_k_idx = np.argsort(scores)[-k:][::-1]

        return [
            RetrievalHit(
                rank=rank,
                score=float(scores[i]),
                regulation_id=self.chunk_ids[i],
                chunk=self.chunks_by_id[self.chunk_ids[i]],
            )
            for rank, i in enumerate(top_k_idx, start=1)
        ]
