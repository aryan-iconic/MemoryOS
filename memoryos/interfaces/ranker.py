"""Ranking contracts for MemoryOS retrieval.

Rankers take candidate memories from working, episodic, and semantic memory and
return the best items for the current query/context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from ..models import MemorySearchResult


class RankerInterface(ABC):
    """Abstract base class for retrieval rankers."""

    @abstractmethod
    def score(self, query: str, result: MemorySearchResult) -> float:
        """Return a final ranking score for a single memory result."""
        raise NotImplementedError

    def rank(
        self,
        query: str,
        results: Sequence[MemorySearchResult],
        limit: Optional[int] = None,
    ) -> List[MemorySearchResult]:
        """Rank memory results from highest to lowest score.

        Concrete rankers can override this method for advanced reranking.
        """
        ranked = sorted(
            results,
            key=lambda item: self.score(query, item),
            reverse=True,
        )

        if limit is not None:
            return ranked[:limit]

        return ranked


# Naming aliases for easier imports.
BaseRanker = RankerInterface
MemoryRanker = RankerInterface


__all__ = ["RankerInterface", "BaseRanker", "MemoryRanker"]
