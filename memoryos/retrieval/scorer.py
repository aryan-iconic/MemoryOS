"""Scoring logic for retrieval ranking."""
from ..extraction.confidence import ConfidenceScorer
from types import List

class RetrievalScorer:
    def __init__(
        self,
        confidence_scorer: ConfidenceScorer = None,
    ):
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()

    def score_fact(
        self,
        fact_content: str,
        base_score: float,
    ) -> float:
        return self.confidence_scorer.calculate(
            text=fact_content,
            base_score=base_score,
        )