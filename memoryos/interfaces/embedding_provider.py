"""Embedding provider contracts for MemoryOS.

This module defines the public interface that every embedding backend should
follow. Keep this file dependency-light so users can plug in sentence
transformers, OpenAI-compatible embeddings, local models, or custom providers
without changing the rest of MemoryOS.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


Embedding = Sequence[float]
EmbeddingBatch = Sequence[Embedding]


class EmbeddingProvider(ABC):
    """Abstract base class for embedding backends.

    Implementations should accept a list/sequence of texts and return one
    embedding per text in the same order.
    """

    model_name: str | None = None
    dimension: int | None = None

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> Any:
        """Return embeddings for a batch of texts.

        Concrete providers may return a list of lists or a numpy array. The
        calling layer should normalize the result only where needed.
        """
        raise NotImplementedError

    def embed_one(self, text: str) -> Embedding:
        """Return a single embedding for one text."""
        embeddings = self.embed([text])
        return embeddings[0]

    @abstractmethod
    def similarity(self, text1: str, text2: str) -> float:
        """Return semantic similarity between two text values."""
        raise NotImplementedError


# Backward-compatible alias for naming flexibility.
BaseEmbeddingProvider = EmbeddingProvider


__all__ = [
    "Embedding",
    "EmbeddingBatch",
    "EmbeddingProvider",
    "BaseEmbeddingProvider",
]
