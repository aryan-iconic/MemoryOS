"""Abstract storage contract for MemoryOS backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from memoryos.models import Fact, Turn


class StorageBackend(ABC):
    """Base contract every MemoryOS storage backend should implement."""

    @abstractmethod
    def save_fact(self, fact: Union[Fact, Dict[str, Any]]) -> Optional[Fact]:
        """Insert or update a fact and return the saved fact when possible."""

    @abstractmethod
    def save_facts(self, facts: List[Union[Fact, Dict[str, Any]]]) -> List[Fact]:
        """Save multiple facts."""

    @abstractmethod
    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Return one fact by ID."""

    @abstractmethod
    def get_all_facts(self, limit: Optional[int] = None) -> List[Fact]:
        """Return all facts, newest first."""

    @abstractmethod
    def get_facts_by_session(self, session_id: str, limit: Optional[int] = None) -> List[Fact]:
        """Return facts for a session."""

    @abstractmethod
    def get_facts_by_type(self, fact_type: str, limit: Optional[int] = None) -> List[Fact]:
        """Return facts of a specific type."""

    @abstractmethod
    def search_facts_keyword(self, keyword: str, limit: Optional[int] = None) -> List[Fact]:
        """Search facts by plain keyword."""

    @abstractmethod
    def update_fact_access_count(self, fact_id: str) -> None:
        """Increment a fact's access counter."""

    @abstractmethod
    def delete_fact(self, fact_id: str) -> None:
        """Delete a fact by ID."""

    @abstractmethod
    def save_turn(self, turn: Union[Turn, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Insert or update a conversation turn."""

    @abstractmethod
    def get_turns_by_session(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return turns for a session in chronological order."""

    @abstractmethod
    def save_episode(self, episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Insert or update an episodic memory summary."""

    @abstractmethod
    def get_episodes_by_session(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return episodes for a session."""

    @abstractmethod
    def get_all_episodes(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return all episodes."""

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        """Delete facts, turns, and episodes for one session."""

    @abstractmethod
    def clear_all(self) -> None:
        """Delete all MemoryOS records from this backend."""

    def close(self) -> None:
        """Close backend resources when needed."""
        return None  # pragma: no cover


BaseStorage = StorageBackend
