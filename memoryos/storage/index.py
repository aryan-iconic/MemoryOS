"""Vector index abstractions and a dependency-free in-memory implementation."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from memoryos.exceptions import IndexBackendError


@dataclass
class VectorRecord:
    """One vector entry stored inside a vector index."""

    id: str
    vector: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """Search result returned by vector indexes."""

    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseVectorIndex(ABC):
    """Contract for vector index backends such as FAISS or pgvector."""

    @abstractmethod
    def add(
        self,
        record_id: str,
        vector: Sequence[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or replace a vector."""

    def add_many(self, records: Iterable[VectorRecord]) -> None:
        for record in records:
            self.add(record.id, record.vector, record.metadata)

    @abstractmethod
    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        """Search nearest vectors."""

    @abstractmethod
    def delete(self, record_id: str) -> None:
        """Delete a vector by ID."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all vectors."""

    @abstractmethod
    def save(self, path: Optional[str] = None) -> None:
        """Persist the index."""

    @abstractmethod
    def load(self, path: Optional[str] = None) -> None:
        """Load the index."""

    @abstractmethod
    def __len__(self) -> int:
        """Return vector count."""


class InMemoryVectorIndex(BaseVectorIndex):
    """Simple cosine-similarity vector index with no external dependency.

    It is not meant for huge production datasets, but it is perfect for tests,
    demos, and small local MemoryOS use cases.
    """

    def __init__(self, dim: Optional[int] = None, persist_path: Optional[str] = None):
        self.dim = dim
        self.persist_path = persist_path
        self._records: Dict[str, VectorRecord] = {}

    def add(
        self,
        record_id: str,
        vector: Sequence[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        clean_vector = _to_float_list(vector)
        self._validate_vector(clean_vector)
        self._records[record_id] = VectorRecord(record_id, clean_vector, metadata or {})

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[VectorSearchResult]:
        if top_k <= 0:
            return []

        query = _to_float_list(query_vector)
        self._validate_vector(query)

        results: List[VectorSearchResult] = []
        threshold = -1.0 if min_score is None else min_score

        for record in self._records.values():
            score = cosine_similarity(query, record.vector)
            if score >= threshold:
                results.append(VectorSearchResult(record.id, score, dict(record.metadata)))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def delete(self, record_id: str) -> None:
        self._records.pop(record_id, None)

    def clear(self) -> None:
        self._records.clear()  # pragma: no cover

    def save(self, path: Optional[str] = None) -> None:
        target = Path(path or self.persist_path or "memoryos_index.json")
        if target.parent and str(target.parent) != ".":
            target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "dim": self.dim,
            "records": [record.__dict__ for record in self._records.values()],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Optional[str] = None) -> None:
        target = Path(path or self.persist_path or "memoryos_index.json")
        if not target.exists():
            raise IndexBackendError("Vector index file does not exist.", details={"path": str(target)})

        payload = json.loads(target.read_text(encoding="utf-8"))
        self.dim = payload.get("dim", self.dim)
        self._records.clear()
        for raw in payload.get("records", []):
            record = VectorRecord(
                id=raw["id"],
                vector=_to_float_list(raw["vector"]),
                metadata=raw.get("metadata", {}),
            )
            self._validate_vector(record.vector)
            self._records[record.id] = record

    def _validate_vector(self, vector: Sequence[float]) -> None:
        if not vector:
            raise IndexBackendError("Vector cannot be empty.")

        if self.dim is None:
            self.dim = len(vector)  # pragma: no cover

        if len(vector) != self.dim:
            raise IndexBackendError(
                "Vector dimension mismatch.",
                details={"expected_dim": self.dim, "actual_dim": len(vector)},
            )

    def __len__(self) -> int:
        return len(self._records)


def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
    if len(vec1) != len(vec2) or not vec1:
        return 0.0  # pragma: no cover

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(dot / (norm1 * norm2))


def _to_float_list(vector: Sequence[float]) -> List[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()  # type: ignore[assignment]  # pragma: no cover
    return [float(value) for value in vector]


VectorIndex = InMemoryVectorIndex
