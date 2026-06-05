"""Ranking utilities for MemoryOS retrieval results."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Iterable, List, Optional

from memoryos.exceptions import RankingError
from memoryos.models import MemorySearchResult


class MemoryRanker:
    """Ranks memory results using similarity, confidence, recency, and source.

    The input score remains the strongest signal, but the final score gets small
    boosts for reliable facts, recent memories, and source quality.
    """

    SOURCE_WEIGHTS = {
        "semantic": 1.00,
        "episodic": 0.92,
        "working": 0.88,
    }

    TYPE_WEIGHTS = {
        "identity": 1.05,
        "decision": 1.03,
        "goal": 1.02,
        "preference": 1.01,
        "context": 1.00,
    }

    def __init__(
        self,
        *,
        similarity_weight: float = 0.70,
        confidence_weight: float = 0.15,
        recency_weight: float = 0.10,
        source_weight: float = 0.05,
        recency_half_life_days: float = 30.0,
    ):
        self.similarity_weight = similarity_weight
        self.confidence_weight = confidence_weight
        self.recency_weight = recency_weight
        self.source_weight = source_weight
        self.recency_half_life_days = recency_half_life_days

    def rank(
        self,
        results: Iterable[MemorySearchResult],
        *,
        top_k: Optional[int] = None,
        now: Optional[float] = None,
    ) -> List[MemorySearchResult]:
        try:
            now_ts = now or time.time()
            scored = []
            for result in results:
                final_score = self.score(result, now=now_ts)
                try:
                    ranked = replace(result, score=final_score)
                except TypeError:
                    result.score = final_score
                    ranked = result
                ranked.metadata = dict(ranked.metadata or {})
                ranked.metadata["ranked_score"] = final_score
                ranked.metadata.setdefault("raw_score", result.score)
                scored.append(ranked)

            scored.sort(key=lambda item: item.score, reverse=True)
            if top_k is not None:
                return scored[:top_k]
            return scored
        except Exception as exc:  # pragma: no cover - defensive
            if isinstance(exc, RankingError):
                raise
            raise RankingError("Failed to rank memory results.") from exc

    def score(self, result: MemorySearchResult, *, now: Optional[float] = None) -> float:
        now_ts = now or time.time()
        raw_similarity = self._clamp(float(result.score or 0.0))
        confidence = self._extract_confidence(result)
        recency = self._recency_score(result.timestamp, now_ts)
        source_boost = self.SOURCE_WEIGHTS.get(str(result.source), 1.0)
        type_boost = self.TYPE_WEIGHTS.get(str(result.type or result.metadata.get("fact_type", "context")), 1.0)

        combined = (
            raw_similarity * self.similarity_weight
            + confidence * self.confidence_weight
            + recency * self.recency_weight
            + source_boost * self.source_weight
        )
        return round(self._clamp(combined * type_boost), 6)

    def _extract_confidence(self, result: MemorySearchResult) -> float:
        if result.confidence is not None:
            return self._clamp(float(result.confidence))
        metadata = result.metadata or {}
        for key in ("confidence", "original_confidence"):
            if key in metadata and metadata[key] is not None:
                return self._clamp(float(metadata[key]))
        return 0.70

    def _recency_score(self, timestamp: Optional[float], now: float) -> float:
        if timestamp is None:
            return 0.50
        age_seconds = max(0.0, now - float(timestamp))
        half_life_seconds = max(1.0, self.recency_half_life_days * 24 * 60 * 60)
        return self._clamp(0.5 ** (age_seconds / half_life_seconds))

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))


Ranker = MemoryRanker
DefaultRanker = MemoryRanker
