"""Memory retrieval across working, episodic, and semantic memory layers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from memoryos.exceptions import RetrievalError
from memoryos.models import MemorySearchResult
from memoryos.retrieval.ranker import MemoryRanker


class MemoryRetriever:
    """Fetch, normalize, deduplicate, and rank memory results."""

    def __init__(
        self,
        *,
        working_memory: Any = None,
        semantic_memory: Any = None,
        episodic_memory: Any = None,
        ranker: Optional[MemoryRanker] = None,
    ):
        self.working_memory = working_memory
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.ranker = ranker or MemoryRanker()

    def retrieve(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        top_k: int = 8,
        semantic_top_k: Optional[int] = None,
        episodic_top_k: Optional[int] = None,
        include_working: bool = True,
        include_semantic: bool = True,
        include_episodic: bool = True,
        min_score: Optional[float] = None,
        fact_type: Optional[str] = None,
    ) -> List[MemorySearchResult]:
        if not (query or "").strip():
            return []

        try:
            results: List[MemorySearchResult] = []

            if include_working and self.working_memory is not None:
                results.extend(self._retrieve_working(query))

            if include_semantic and self.semantic_memory is not None:
                results.extend(
                    self._retrieve_semantic(
                        query,
                        top_k=semantic_top_k or top_k,
                        session_id=session_id,
                        min_score=min_score,
                        fact_type=fact_type,
                    )
                )

            if include_episodic and self.episodic_memory is not None:
                results.extend(
                    self._retrieve_episodic(
                        query,
                        top_k=episodic_top_k or max(1, top_k // 2),
                        session_id=session_id,
                        min_score=min_score,
                    )
                )

            deduped = self.deduplicate(results)
            return self.ranker.rank(deduped, top_k=top_k)
        except Exception as exc:  # pragma: no cover - defensive
            if isinstance(exc, RetrievalError):
                raise
            raise RetrievalError("Failed to retrieve memory results.", details={"query": query}) from exc

    def _retrieve_working(self, query: str) -> List[MemorySearchResult]:
        if hasattr(self.working_memory, "search"):
            return list(self.working_memory.search(query))
        return []

    def _retrieve_semantic(
        self,
        query: str,
        *,
        top_k: int,
        session_id: Optional[str],
        min_score: Optional[float],
        fact_type: Optional[str],
    ) -> List[MemorySearchResult]:
        if not hasattr(self.semantic_memory, "search"):
            return []
        raw_results = self.semantic_memory.search(
            query=query,
            top_k=top_k,
            fact_type=fact_type,
            session_id=session_id,
            min_score=min_score,
        )
        return [self._normalize_result(item, source="semantic") for item in raw_results]

    def _retrieve_episodic(
        self,
        query: str,
        *,
        top_k: int,
        session_id: Optional[str],
        min_score: Optional[float],
    ) -> List[MemorySearchResult]:
        if session_id is None or not hasattr(self.episodic_memory, "search"):
            return []
        raw_results = self.episodic_memory.search(
            query=query,
            session_id=session_id,
            limit=top_k,
            min_score=min_score,
        )
        return [self._normalize_result(item, source="episodic") for item in raw_results]

    def deduplicate(self, results: Iterable[MemorySearchResult]) -> List[MemorySearchResult]:
        seen: Dict[str, MemorySearchResult] = {}
        for result in results:
            key = result.id or self._normalize_text(result.content)
            existing = seen.get(key)
            if existing is None or result.score > existing.score:
                seen[key] = result
        return list(seen.values())

    def _normalize_result(self, item: Any, *, source: str) -> MemorySearchResult:
        if isinstance(item, MemorySearchResult):
            return item

        if isinstance(item, tuple) and len(item) == 2:
            payload, score = item
            if source == "episodic" and isinstance(payload, dict):
                return MemorySearchResult(
                    id=payload.get("id"),
                    content=payload.get("summary", ""),
                    source="episodic",
                    score=float(score),
                    timestamp=payload.get("end_timestamp") or payload.get("created_at"),
                    metadata={
                        "session_id": payload.get("session_id"),
                        "turn_count": payload.get("turn_count"),
                        "match_type": "episodic_semantic",
                        **dict(payload.get("metadata", {}) or {}),
                    },
                )

            # Supports older semantic style: (Fact, score)
            if hasattr(payload, "content"):
                return MemorySearchResult(
                    id=getattr(payload, "id", None),
                    content=getattr(payload, "content", ""),
                    source="semantic",
                    score=float(score),
                    type=getattr(payload, "type", None),
                    confidence=getattr(payload, "confidence", None),
                    timestamp=getattr(payload, "timestamp", None),
                    metadata={
                        "session_id": getattr(payload, "session_id", None),
                        "match_type": "semantic",
                    },
                )

        if isinstance(item, dict):
            content = item.get("content") or item.get("summary") or item.get("text") or ""
            return MemorySearchResult(
                id=item.get("id"),
                content=content,
                source=source,  # type: ignore[arg-type]
                score=float(item.get("score", 0.0)),
                type=item.get("type") or item.get("fact_type"),
                confidence=item.get("confidence"),
                timestamp=item.get("timestamp") or item.get("created_at"),
                metadata=dict(item.get("metadata", {}) or {}),
            )

        raise RetrievalError("Unsupported retrieval result format.", details={"type": type(item).__name__})

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join((text or "").lower().strip().split())


Retriever = MemoryRetriever
