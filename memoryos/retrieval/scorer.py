"""Scoring logic for retrieval ranking."""

from __future__ import annotations

from typing import Optional

from memoryos.extraction.confidence import ConfidenceScorer


class RetrievalScorer:
    """Small helper that combines retrieval score with confidence scoring."""

    def __init__(self, confidence_scorer: Optional[ConfidenceScorer] = None):
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()

    def score_fact(self, fact_content: str, base_score: float) -> float:
        return self.confidence_scorer.calculate(
            text=fact_content,
            base_score=base_score,
        )


__all__ = ["RetrievalScorer"]
