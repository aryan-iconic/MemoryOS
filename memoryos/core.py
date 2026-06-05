"""Public MemoryOS API.

This module intentionally stays thin. The heavy work is delegated to
``memoryos.memory.manager.MemoryManager`` so storage, retrieval, extraction,
compression, and ranking remain replaceable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from memoryos.config import MemoryOSConfig
from memoryos.exceptions import ConfigError, MemoryOSError, ValidationError
from memoryos.memory.manager import MemoryManager
from memoryos.models import Fact, MemorySearchResult, Turn


class MemoryOS:
    """Main user-facing class for MemoryOS.

    Basic usage::

        from memoryos import MemoryOS

        memory = MemoryOS(db_path="memoryos.db", session_id="default")
        memory.process_turn("My name is Aryan. I prefer dark UI.")
        context = memory.build_context("What UI does the user prefer?")

    The constructor accepts either a full ``MemoryOSConfig`` object or quick
    keyword overrides such as ``db_path`` and ``session_id``.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        session_id: str = "default_session",
        config: Optional[Union[MemoryOSConfig, Dict[str, Any]]] = None,
        store: Optional[Any] = None,
        extractor: Optional[Any] = None,
        working_memory: Optional[Any] = None,
        semantic_memory: Optional[Any] = None,
        episodic_memory: Optional[Any] = None,
        retriever: Optional[Any] = None,
        context_builder: Optional[Any] = None,
        **config_overrides: Any,
    ):
        self.config = self._build_config(
            config=config,
            db_path=db_path,
            overrides=config_overrides,
        )
        self.session_id = session_id
        self.manager = MemoryManager(
            config=self.config,
            store=store,
            extractor=extractor,
            working_memory=working_memory,
            semantic_memory=semantic_memory,
            episodic_memory=episodic_memory,
            retriever=retriever,
            context_builder=context_builder,
            session_id=session_id,
        )

        # Compatibility aliases for older code that accessed internals directly.
        self.store = self.manager.store
        self.extractor = self.manager.extractor
        self.working_memory = self.manager.working_memory
        self.semantic_memory = self.manager.semantic_memory
        self.episodic_memory = self.manager.episodic_memory
        self.retriever = self.manager.retriever
        self.context_builder = self.manager.context_builder

    @classmethod
    def from_config(
        cls,
        config: Union[MemoryOSConfig, Dict[str, Any]],
        *,
        session_id: str = "default_session",
        **kwargs: Any,
    ) -> "MemoryOS":
        """Create MemoryOS from a config object or config dictionary."""
        return cls(config=config, session_id=session_id, **kwargs)

    @classmethod
    def from_env(
        cls,
        *,
        prefix: str = "MEMORYOS_",
        session_id: str = "default_session",
        **kwargs: Any,
    ) -> "MemoryOS":
        """Create MemoryOS from environment variables.

        Example: ``MEMORYOS_DB_PATH=./data/memoryos.db``.
        """
        return cls(config=MemoryOSConfig.from_env(prefix=prefix), session_id=session_id, **kwargs)

    def process_turn(
        self,
        user_message: str,
        ai_response: str = "",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        create_episode: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Save one conversation turn and extract/store new durable facts."""
        user_message = self._validate_text(user_message, field_name="user_message")
        ai_response = "" if ai_response is None else str(ai_response)
        return self.manager.process_turn(
            user_message=user_message,
            ai_response=ai_response,
            session_id=session_id,
            metadata=metadata,
            create_episode=create_episode,
        )

    def add_memory(
        self,
        content: str,
        *,
        fact_type: str = "context",
        confidence: float = 0.95,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Fact:
        """Manually add a long-term fact without running extraction."""
        content = self._validate_text(content, field_name="content")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValidationError("confidence must be between 0 and 1.", details={"confidence": confidence})

        fact = Fact(
            content=content,
            type=fact_type,  # type: ignore[arg-type]
            confidence=float(confidence),
            session_id=session_id or self.session_id,
            source="manual",
            metadata=metadata or {},
        )
        saved = self.semantic_memory.add_fact(fact)
        return saved or fact

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        fact_type: Optional[str] = None,
        session_id: Optional[str] = None,
        min_score: Optional[float] = 0.0,
        include_working: bool = True,
        include_semantic: bool = True,
        include_episodic: bool = True,
    ) -> List[MemorySearchResult]:
        """Search all enabled memory layers and return ranked results."""
        query = self._validate_text(query, field_name="query")
        return self.manager.search(
            query,
            session_id=session_id,
            top_k=top_k,
            fact_type=fact_type,
            min_score=min_score,
            include_working=include_working,
            include_semantic=include_semantic,
            include_episodic=include_episodic,
        )

    # Short alias used by some integrations.
    search = search_memory

    def build_context(
        self,
        query: str,
        limit: Optional[int] = None,
        session_id: Optional[str] = None,
        max_chars: Optional[int] = None,
        include_recent_turns: bool = True,
        **_: Any,
    ) -> str:
        """Build the final compact context block for an LLM prompt."""
        query = self._validate_text(query, field_name="query")
        return self.manager.build_context(
            query,
            session_id=session_id,
            top_k=limit,
            max_chars=max_chars,
            include_recent_turns=include_recent_turns,
        )

    def build_prompt_context(
        self,
        query: str,
        memory_limit: int = 5,
        turn_limit: int = 6,
        max_chars: int = 3000,
    ) -> str:
        """Backward-compatible prompt context method.

        ``turn_limit`` is respected for recent working memory. Retrieval is
        handled through the manager so semantic, episodic, and working memory are
        combined consistently.
        """
        query = self._validate_text(query, field_name="query")
        results = self.manager.search(query, top_k=memory_limit, min_score=0.0)
        recent_turns = self.working_memory.get_recent_turns(turn_limit)
        return self.context_builder.build(
            query=query,
            results=results,
            recent_turns=recent_turns,
            max_chars=max_chars,
        )

    def maybe_create_episode(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Create an episodic summary from recent stored turns when enough exist."""
        return self.manager.maybe_create_episode(session_id=session_id)

    def get_all_facts(self, limit: Optional[int] = None) -> List[Fact]:
        return self.manager.get_facts(limit=limit)

    def get_session_facts(self, session_id: Optional[str] = None, limit: Optional[int] = None) -> List[Fact]:
        return self.manager.get_facts(session_id=session_id or self.session_id, limit=limit)

    def get_turns(self, session_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.manager.get_turns(session_id=session_id or self.session_id, limit=limit)

    def get_episodes(self, session_id: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        return self.manager.get_episodes(session_id=session_id or self.session_id, limit=limit)

    def clear_session(self, session_id: Optional[str] = None) -> None:
        self.manager.clear_session(session_id=session_id or self.session_id)

    def clear_all(self) -> None:
        self.manager.clear_all()

    def close(self) -> None:
        self.manager.close()

    def __enter__(self) -> "MemoryOS":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @staticmethod
    def _build_config(
        *,
        config: Optional[Union[MemoryOSConfig, Dict[str, Any]]],
        db_path: Optional[str],
        overrides: Dict[str, Any],
    ) -> MemoryOSConfig:
        if config is None:
            config_obj = MemoryOSConfig()
        elif isinstance(config, MemoryOSConfig):
            config_obj = config
        elif isinstance(config, dict):
            config_obj = MemoryOSConfig.from_dict(config)
        else:
            raise ConfigError("config must be a MemoryOSConfig, dictionary, or None.")

        merged = config_obj.to_dict() if hasattr(config_obj, "to_dict") else dict(config_obj.__dict__)
        if db_path is not None:
            merged["db_path"] = str(Path(db_path))
        merged.update({key: value for key, value in overrides.items() if value is not None})
        final_config = MemoryOSConfig.from_dict(merged)
        final_config.validate()
        final_config.ensure_paths()
        return final_config

    @staticmethod
    def _validate_text(value: Any, *, field_name: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValidationError(f"{field_name} cannot be empty.")
        return text


__all__ = ["MemoryOS"]
