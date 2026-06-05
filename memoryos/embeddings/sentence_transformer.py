"""Sentence Transformer embedding implementation with a safe dev fallback.

MemoryOS should work as a library even before optional embedding dependencies are
installed. When ``sentence-transformers`` is available, this class uses it. When
it is missing, it falls back to a deterministic hash embedding so tests and local
smoke checks can still run. Install ``sentence-transformers`` for real semantic
quality.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Iterable, List, Optional

import numpy as np

from memoryos.exceptions import DependencyNotInstalledError, EmbeddingError
from memoryos.embeddings.base import BaseEmbedding

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(BaseEmbedding):
    """Embed text using sentence-transformers, with optional hash fallback."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        dimension: int = 384,
        allow_fallback: bool = True,
    ):
        self.model_name = model_name
        self.dimension = int(dimension)
        self.allow_fallback = allow_fallback
        self.model: Optional[object] = None
        self.using_fallback = False

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(model_name)
        except Exception as exc:
            if not allow_fallback:
                raise DependencyNotInstalledError(
                    "sentence-transformers is not installed. Install it with: pip install sentence-transformers",
                    details={"package": "sentence-transformers", "model_name": model_name},
                ) from exc
            self.using_fallback = True
            logger.warning(
                "sentence-transformers is unavailable; using deterministic hash embeddings for development/testing."
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        texts = [str(text or "") for text in texts]
        if self.model is not None:
            try:
                embeddings = self.model.encode(texts)  # type: ignore[attr-defined]
                return np.asarray(embeddings, dtype=np.float32)
            except Exception as exc:  # pragma: no cover - model/runtime dependent
                raise EmbeddingError("Failed to generate sentence-transformer embeddings.") from exc

        return np.vstack([self._hash_embed(text) for text in texts]).astype(np.float32)

    def similarity(self, text1: str, text2: str) -> float:
        embedding1 = self.embed([text1])[0]
        embedding2 = self.embed([text2])[0]
        return self.cosine_similarity(embedding1, embedding2)

    def _hash_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = self._tokenize(text)
        if not tokens:
            return vector

        # Unigrams + simple bigrams improve recall for short tests without any ML dependency.
        features: List[str] = tokens[:]
        features.extend(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))

        for token in features:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 if "_" not in token else 0.65
            vector[index] += sign * weight

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", str(text or "").lower())

    @staticmethod
    def cosine_similarity(vec1: Iterable[float], vec2: Iterable[float]) -> float:
        a = np.asarray(list(vec1), dtype=np.float32)
        b = np.asarray(list(vec2), dtype=np.float32)
        if a.size == 0 or b.size == 0:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
