"""Storage backend contract for MemoryOS.

Any persistent backend should implement this interface: SQLite, Postgres,
MongoDB, local JSON, cloud storage, or encrypted storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

from ..models import Fact, Turn

FactInput = Union[Fact, Dict[str, Any]]
TurnInput = Union[Turn, Dict[str, Any]]
Episode = Dict[str, Any]


class StorageBackend(ABC):
    """Abstract base class for MemoryOS storage providers."""

    @abstractmethod
    def save_fact(self, fact: FactInput) -> None:
        """Persist or update one fact."""
        raise NotImplementedError

    def save_facts(self, facts: Sequence[FactInput]) -> None:
        """Persist many facts.

        Backends can override this for faster bulk inserts.
        """
        for fact in facts:
            self.save_fact(fact)

    @abstractmethod
    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Return one fact by id, or None when missing."""
        raise NotImplementedError

    @abstractmethod
    def get_all_facts(self, limit: Optional[int] = None) -> List[Fact]:
        """Return facts ordered by backend preference, usually newest first."""
        raise NotImplementedError

    @abstractmethod
    def get_facts_by_session(self, session_id: str) -> List[Fact]:
        """Return all facts belonging to a session."""
        raise NotImplementedError

    @abstractmethod
    def get_facts_by_type(self, fact_type: str) -> List[Fact]:
        """Return all facts matching a MemoryOS fact type."""
        raise NotImplementedError

    @abstractmethod
    def search_facts_keyword(self, keyword: str) -> List[Fact]:
        """Return facts matching a simple keyword query."""
        raise NotImplementedError

    @abstractmethod
    def update_fact_access_count(self, fact_id: str) -> None:
        """Increment the access count for a fact after retrieval/use."""
        raise NotImplementedError

    @abstractmethod
    def delete_fact(self, fact_id: str) -> None:
        """Delete a fact by id."""
        raise NotImplementedError

    @abstractmethod
    def save_turn(self, turn: TurnInput) -> None:
        """Persist or update one conversation turn."""
        raise NotImplementedError

    @abstractmethod
    def get_turns_by_session(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return turns for a session, usually chronological oldest to newest."""
        raise NotImplementedError

    @abstractmethod
    def save_episode(self, episode: Episode) -> None:
        """Persist or update one compressed episode summary."""
        raise NotImplementedError

    @abstractmethod
    def get_episodes_by_session(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Episode]:
        """Return compressed episodes for a session."""
        raise NotImplementedError

    @abstractmethod
    def get_all_episodes(self, limit: Optional[int] = None) -> List[Episode]:
        """Return compressed episodes across all sessions."""
        raise NotImplementedError

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        """Delete facts, turns, and episodes for one session."""
        raise NotImplementedError

    @abstractmethod
    def clear_all(self) -> None:
        """Delete all MemoryOS data from this backend."""
        raise NotImplementedError


__all__ = [
    "Episode",
    "FactInput",
    "TurnInput",
    "StorageBackend",
]
