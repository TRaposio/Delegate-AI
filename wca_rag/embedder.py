"""Embedder interface and SentenceTransformer-based implementation.

The Embedder turns text into dense vectors for retrieval. We split the
interface into `encode_documents` and `encode_query` to make the
query/document asymmetry (e.g. bge's query prefix) a contract, not a
caller-side responsibility. See docs/CONCEPTS.md (bi-encoder entry, when
written) for background.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# bge-small-en-v1.5 query prefix, per the model card. Documents get no prefix.
# v1.5 made this optional but still helpful for short queries — keep it.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(ABC):
    """Abstract base for embedding models.

    Two methods rather than one to force every implementation to declare how
    it handles query vs document asymmetry. Models like bge use a query
    prefix; OpenAI's API is symmetric; Voyage/Cohere take an input_type
    parameter. A single `encode()` would hide this and produce silent bugs
    on model swap.

    Embeddings are L2-normalized so dot product == cosine similarity. The
    retriever can use a plain matrix multiply without re-normalizing.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier, written to the embeddings sidecar."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output vector dimensionality."""

    @abstractmethod
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Embed corpus chunks. Returns shape (n, embedding_dim), float32, L2-normalized."""

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        """Embed a single user query. Returns shape (embedding_dim,), float32, L2-normalized."""


class SentenceTransformerEmbedder(Embedder):
    """sentence-transformers wrapper. Default: BAAI/bge-small-en-v1.5.

    Local, free, 384-dim, ~130MB download cached at ~/.cache/huggingface/.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 32,
        query_prefix: str = BGE_QUERY_PREFIX,
        device: str | None = None,
    ) -> None:
        # Local import: sentence-transformers is a heavy dep. Importing at
        # module level would slow down `import wca_rag.embedder` for callers
        # that only need the ABC (e.g. tests with a fake embedder).
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._model.get_embedding_dimension()

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,  # L2-normalize → dot product == cosine
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 50,
        )
        return vectors.astype(np.float32, copy=False)

    def encode_query(self, text: str) -> np.ndarray:
        prefixed = f"{self._query_prefix}{text}"
        vector = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vector.astype(np.float32, copy=False)
