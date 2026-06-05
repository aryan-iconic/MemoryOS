from __future__ import annotations

import builtins
from typing import Any

import numpy as np
import pytest

from memoryos.embeddings.sentence_transformer import SentenceTransformerEmbedding
from memoryos.exceptions import DependencyNotInstalledError


def test_sentence_transformer_embedding_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("blocked for deterministic coverage")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    embedder = SentenceTransformerEmbedding(dimension=8, allow_fallback=True)
    assert embedder.using_fallback is True
    assert embedder.model is None

    embeddings = embedder.embed(["dark UI memory", ""])
    assert embeddings.shape == (2, 8)
    assert embeddings.dtype == np.float32
    assert np.linalg.norm(embeddings[0]) > 0
    assert np.allclose(embeddings[1], np.zeros(8, dtype=np.float32))
    assert embedder._tokenize("Hello, AI-2!") == ["hello", "ai", "2"]

    with pytest.raises(DependencyNotInstalledError):
        SentenceTransformerEmbedding(dimension=8, allow_fallback=False)
