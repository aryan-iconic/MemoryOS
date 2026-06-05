"""Optional FAISS vector index implementation for MemoryOS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from memoryos.exceptions import DependencyNotInstalledError, IndexBackendError
from memoryos.storage.index import BaseVectorIndex, VectorSearchResult


class FAISSVectorIndex(BaseVectorIndex):
    """Cosine-similarity FAISS index using normalized inner product search.

    FAISS is optional. Install ``faiss-cpu`` before using this backend.
    """

    def __init__(self, dim: int, persist_path: str = "memoryos.faiss"):
        self.dim = dim
        self.persist_path = persist_path
        self.metadata_path = f"{persist_path}.meta.json"
        self._faiss = self._import_faiss()
        self.index = self._faiss.IndexFlatIP(dim)
        self.ids: List[str] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._vectors: Dict[str, List[float]] = {}

    def add(self, record_id: str, vector: Sequence[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        clean = self._normalize_vector(vector)

        # IndexFlatIP does not support deleting/replacing arbitrary IDs directly.
        # For correctness, rebuild when replacing an existing ID.
        self._vectors[record_id] = clean.tolist()
        self.metadata[record_id] = metadata or {}
        self._rebuild_index()

    def search(self, query_vector: Sequence[float], top_k: int = 5, min_score: Optional[float] = None) -> List[VectorSearchResult]:
        if top_k <= 0 or len(self.ids) == 0:
            return []

        query = self._normalize_vector(query_vector).reshape(1, -1)
        limit = min(top_k, len(self.ids))
        scores, indexes = self.index.search(query, limit)

        results: List[VectorSearchResult] = []
        threshold = -1.0 if min_score is None else min_score
        for score, idx in zip(scores[0], indexes[0]):
            if idx < 0:
                continue
            record_id = self.ids[int(idx)]
            score_float = float(score)
            if score_float >= threshold:
                results.append(
                    VectorSearchResult(
                        id=record_id,
                        score=score_float,
                        metadata=dict(self.metadata.get(record_id, {})),
                    )
                )

        return results

    def delete(self, record_id: str) -> None:
        if record_id not in self._vectors:
            return
        self._vectors.pop(record_id, None)
        self.metadata.pop(record_id, None)
        self._rebuild_index()

    def clear(self) -> None:
        self.ids.clear()
        self.metadata.clear()
        self._vectors.clear()
        self.index = self._faiss.IndexFlatIP(self.dim)

    def save(self, path: Optional[str] = None) -> None:
        target = Path(path or self.persist_path)
        if target.parent and str(target.parent) != ".":
            target.parent.mkdir(parents=True, exist_ok=True)

        self._faiss.write_index(self.index, str(target))
        payload = {
            "dim": self.dim,
            "ids": self.ids,
            "metadata": self.metadata,
            "vectors": self._vectors,
        }
        Path(f"{target}.meta.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Optional[str] = None) -> None:
        target = Path(path or self.persist_path)
        meta_target = Path(f"{target}.meta.json")
        if not target.exists() or not meta_target.exists():
            raise IndexBackendError("FAISS index or metadata file is missing.", details={"path": str(target)})

        self.index = self._faiss.read_index(str(target))
        payload = json.loads(meta_target.read_text(encoding="utf-8"))
        self.dim = int(payload.get("dim", self.dim))
        self.ids = list(payload.get("ids", []))
        self.metadata = dict(payload.get("metadata", {}))
        self._vectors = dict(payload.get("vectors", {}))

    def _normalize_vector(self, vector: Sequence[float]) -> np.ndarray:
        if hasattr(vector, "tolist"):
            vector = vector.tolist()  # type: ignore[assignment]
        arr = np.asarray([float(value) for value in vector], dtype=np.float32)
        if arr.size != self.dim:
            raise IndexBackendError("Vector dimension mismatch.", details={"expected_dim": self.dim, "actual_dim": int(arr.size)})
        norm = np.linalg.norm(arr)
        if norm == 0:
            raise IndexBackendError("Cannot index a zero vector.")
        return arr / norm

    def _rebuild_index(self) -> None:
        self.index = self._faiss.IndexFlatIP(self.dim)
        self.ids = list(self._vectors.keys())
        if not self.ids:
            return
        matrix = np.vstack([self._normalize_vector(self._vectors[record_id]) for record_id in self.ids]).astype(np.float32)
        self.index.add(matrix)

    @staticmethod
    def _import_faiss() -> Any:
        try:
            import faiss  # type: ignore
            return faiss
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise DependencyNotInstalledError(
                "FAISS is not installed. Install it with: pip install faiss-cpu",
                details={"package": "faiss-cpu"},
            ) from exc

    def __len__(self) -> int:
        return len(self.ids)


FaissIndex = FAISSVectorIndex
VectorIndex = FAISSVectorIndex
