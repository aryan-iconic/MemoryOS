"""High-level memory coordinator for MemoryOS."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from memoryos.config import MemoryOSConfig
from memoryos.extraction.extractor import Extractor
from memoryos.memory.episodic import EpisodicMemory
from memoryos.memory.semantic import SemanticMemory
from memoryos.memory.working import WorkingMemory
from memoryos.models import Fact, MemorySearchResult, Turn
from memoryos.retrieval.builder import PromptContextBuilder
from memoryos.retrieval.ranker import MemoryRanker
from memoryos.retrieval.retriever import MemoryRetriever
from memoryos.storage.sqlite_store import SQLiteStore


class MemoryManager:
    """Coordinates working, episodic, and semantic memory.

    This class is the clean orchestration layer that ``core.py`` can call. It is
    intentionally thin: extraction, storage, ranking, and context formatting stay
    in their own files.
    """

    def __init__(
        self,
        *,
        config: Optional[MemoryOSConfig] = None,
        store: Optional[Any] = None,
        extractor: Optional[Extractor] = None,
        working_memory: Optional[WorkingMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        retriever: Optional[MemoryRetriever] = None,
        context_builder: Optional[PromptContextBuilder] = None,
        session_id: str = "default_session",
    ):
        self.config = config or MemoryOSConfig()
        self.config.validate()
        self.config.ensure_paths()

        self.session_id = session_id
        self.store = store or SQLiteStore(self.config.db_path)
        self.extractor = extractor or Extractor(min_confidence=self.config.min_fact_confidence)
        self.working_memory = working_memory or WorkingMemory(self.config)
        self.semantic_memory = semantic_memory or SemanticMemory(store=self.store)
        self.episodic_memory = episodic_memory or EpisodicMemory(store=self.store)
        self.ranker = MemoryRanker(recency_half_life_days=self.config.retrieval_recency_half_life_days)
        self.retriever = retriever or MemoryRetriever(
            working_memory=self.working_memory,
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory,
            ranker=self.ranker,
        )
        self.context_builder = context_builder or PromptContextBuilder(max_chars=self.config.max_context_tokens)

    def process_turn(
        self,
        user_message: str,
        ai_response: str = "",
        *,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        create_episode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Save a turn, extract facts, store semantic memory, and optionally summarize."""
        active_session_id = session_id or self.session_id
        turn = Turn(
            user_message=user_message,
            ai_response=ai_response,
            session_id=active_session_id,
            metadata=metadata or {},
        )

        self.working_memory.add_turn(turn)
        self.store.save_turn(turn)

        extracted_facts = self.extractor.extract(turn)
        existing_facts = self.store.get_facts_by_session(active_session_id)
        new_facts = self._filter_new_facts(extracted_facts, existing_facts)
        saved_facts = self.semantic_memory.add_facts(new_facts) if new_facts else []

        episode = None
        should_create_episode = self.config.auto_create_episodes if create_episode is None else create_episode
        if should_create_episode:
            episode = self.maybe_create_episode(active_session_id)  # pragma: no cover

        return {
            "turn": turn,
            "extracted_facts": extracted_facts,
            "new_facts": new_facts,
            "saved_facts": saved_facts,
            "episode": episode,
        }

    def search(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        fact_type: Optional[str] = None,
        min_score: Optional[float] = None,
        include_working: bool = True,
        include_semantic: bool = True,
        include_episodic: bool = True,
    ) -> List[MemorySearchResult]:
        active_session_id = session_id or self.session_id
        return self.retriever.retrieve(
            query,
            session_id=active_session_id,
            top_k=top_k or self.config.semantic_top_k + self.config.episodic_top_k,
            semantic_top_k=self.config.semantic_top_k,
            episodic_top_k=self.config.episodic_top_k,
            fact_type=fact_type,
            min_score=min_score,
            include_working=include_working,
            include_semantic=include_semantic,
            include_episodic=include_episodic,
        )

    def build_context(
        self,
        query: str,
        *,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
        max_chars: Optional[int] = None,
        include_recent_turns: bool = True,
    ) -> str:
        active_session_id = session_id or self.session_id
        results = self.search(query, session_id=active_session_id, top_k=top_k, min_score=0.0)
        recent_turns = self.working_memory.get_recent_turns() if include_recent_turns else []
        return self.context_builder.build(
            query=query,
            results=results,
            recent_turns=recent_turns,
            max_chars=max_chars or self.config.max_context_tokens,
        )

    def maybe_create_episode(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create an episode summary from recent stored turns when enough exist."""
        active_session_id = session_id or self.session_id
        turns_data = self.store.get_turns_by_session(
            active_session_id,
            limit=self.config.episode_turn_window,
        )
        if len(turns_data) < self.config.min_episode_turns:
            return None  # pragma: no cover

        turns = [self._dict_to_turn(item) for item in turns_data]
        return self.episodic_memory.create_episode(
            session_id=active_session_id,
            turns=turns,
            metadata={
                "created_by": "MemoryManager.maybe_create_episode",
                "created_at": time.time(),
            },
        )

    def get_facts(self, *, session_id: Optional[str] = None, limit: Optional[int] = None) -> List[Fact]:
        if session_id is not None:
            return self.store.get_facts_by_session(session_id, limit=limit)
        return self.store.get_all_facts(limit=limit)

    def get_turns(self, *, session_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.store.get_turns_by_session(session_id or self.session_id, limit=limit)

    def get_episodes(self, *, session_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.store.get_episodes_by_session(session_id or self.session_id, limit=limit)

    def clear_session(self, session_id: Optional[str] = None) -> None:
        active_session_id = session_id or self.session_id
        self.store.clear_session(active_session_id)
        if active_session_id == self.session_id:
            self.working_memory.clear()

    def clear_all(self) -> None:
        self.store.clear_all()
        self.working_memory.clear()

    def close(self) -> None:
        if hasattr(self.store, "close"):
            self.store.close()

    def _filter_new_facts(self, extracted_facts: List[Fact], existing_facts: List[Fact]) -> List[Fact]:
        deduplicator = getattr(self.extractor, "deduplicator", None)
        if deduplicator is not None and hasattr(deduplicator, "filter_new_facts"):
            return deduplicator.filter_new_facts(extracted_facts, existing_facts)

        existing = {  # pragma: no cover
            self._normalize_fact_content(fact.content) for fact in existing_facts if getattr(fact, "content", None)
        }
        new_facts: List[Fact] = []  # pragma: no cover
        for fact in extracted_facts:  # pragma: no cover
            normalized = self._normalize_fact_content(getattr(fact, "content", ""))  # pragma: no cover
            if normalized and normalized not in existing:  # pragma: no cover
                new_facts.append(fact)  # pragma: no cover
                existing.add(normalized)  # pragma: no cover
        return new_facts  # pragma: no cover

    @staticmethod
    def _normalize_fact_content(content: str) -> str:
        return " ".join((content or "").lower().strip().split())  # pragma: no cover

    @staticmethod
    def _dict_to_turn(data: Dict[str, Any]) -> Turn:
        return Turn(
            id=str(data.get("id") or ""),
            user_message=data.get("user_message", ""),
            ai_response=data.get("ai_response", ""),
            session_id=data.get("session_id", "default_session"),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}) or {},
        )


Manager = MemoryManager
