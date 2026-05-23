from __future__ import annotations

from typing import Any, Dict, List, Optional

from .extraction.extractor import Extractor
from .memory.semantic import SemanticMemory
from .models import Fact, MemorySearchResult, Turn
from .storage.sqlite_store import SQLiteStore
from .memory.working import WorkingMemory

class MemoryOS:
    def __init__(
        self,
        db_path: str = "memoryos.db",
        session_id: str = "default_session",
        similarity_threshold: float = 0.35,
    ):
        self.session_id = session_id
        self.store = SQLiteStore(db_path)
        self.extractor = Extractor()
        self.working_memory = WorkingMemory()

        self.semantic_memory = SemanticMemory(
            store=self.store,
            similarity_threshold=similarity_threshold,
        )

    def process_turn(
        self,
        user_message: str,
        ai_response: str = "",
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        active_session_id = session_id or self.session_id

        turn = Turn(
            session_id=active_session_id,
            user_message=user_message,
            ai_response=ai_response,
        )

        self.store.save_turn(turn)

        extracted_facts = self.extractor.extract(turn)
        self.working_memory.add_turn(turn)
        self.store.save_turn(turn)
        existing_facts = self.store.get_facts_by_session(active_session_id)

        new_facts = self._filter_new_facts(
            extracted_facts=extracted_facts,
            existing_facts=existing_facts,
        )

        saved_facts = self.semantic_memory.add_facts(new_facts)

        return {
            "turn": turn,
            "extracted_facts": extracted_facts,
            "new_facts": new_facts,
            "saved_facts": saved_facts,
        }

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        fact_type: Optional[str] = None,
        session_id: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[MemorySearchResult]:

        active_session_id = session_id or self.session_id

        return self.semantic_memory.search(
            query=query,
            top_k=top_k,
            fact_type=fact_type,
            session_id=active_session_id,
            min_score=min_score,
        )

    def build_context(
        self,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None,
        max_chars: int = 1500,
        min_score: Optional[float] = 0.20,
    ) -> str:

        results = self.search_memory(
            query=query,
            top_k=limit,
            session_id=session_id,
            min_score=min_score,
        )

        if not results:
            return ""

        lines = ["Relevant user memory:"]

        for result in results:
            metadata = result.metadata or {}

            fact_type = metadata.get("fact_type", "unknown")
            confidence = metadata.get("original_confidence", 0.0)

            lines.append(
                f"- {result.content} "
                f"(type={fact_type}, confidence={confidence:.2f}, score={result.score:.3f})"
            )

        context = "\n".join(lines)

        if len(context) > max_chars:
            context = context[:max_chars].rstrip() + "..."

        return context
    def build_prompt_context(
        self,
        query: str,
        memory_limit: int = 5,
        turn_limit: int = 6,
        max_chars: int = 3000,
    ) -> str:
        memory_context = self.build_context(
            query=query,
            limit=memory_limit,
            max_chars=max_chars // 2,
            min_score=0.20,
        )

        working_context = self.working_memory.build_context(
            limit=turn_limit,
            max_chars=max_chars // 2,
        )

        parts = []

        if memory_context:
            parts.append(memory_context)

        if working_context:
            parts.append(working_context)

        final_context = "\n\n".join(parts)

        if len(final_context) > max_chars:
            final_context = final_context[:max_chars].rstrip() + "..."

        return final_context
        
    def get_all_facts(
        self,
        limit: Optional[int] = None,
    ) -> List[Fact]:
        return self.store.get_all_facts(limit=limit)

    def get_session_facts(
        self,
        session_id: Optional[str] = None,
    ) -> List[Fact]:
        active_session_id = session_id or self.session_id
        return self.store.get_facts_by_session(active_session_id)

    def get_turns(
        self,
        session_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        active_session_id = session_id or self.session_id

        return self.store.get_turns_by_session(
            session_id=active_session_id,
            limit=limit,
        )

    def clear_session(
        self,
        session_id: Optional[str] = None,
    ) -> None:
        active_session_id = session_id or self.session_id
        self.store.clear_session(active_session_id)

    def clear_all(self) -> None:
        self.store.clear_all()

    def _filter_new_facts(
        self,
        extracted_facts: List[Fact],
        existing_facts: List[Fact],
    ) -> List[Fact]:

        deduplicator = getattr(self.extractor, "deduplicator", None)

        if deduplicator is not None and hasattr(deduplicator, "filter_new_facts"):
            return deduplicator.filter_new_facts(
                new_facts=extracted_facts,
                existing_facts=existing_facts,
            )

        if deduplicator is not None and hasattr(deduplicator, "deduplicate_facts"):
            combined_facts = existing_facts + extracted_facts
            unique_facts = deduplicator.deduplicate_facts(combined_facts)

            existing_contents = {
                self._normalize_fact_content(fact.content)
                for fact in existing_facts
                if getattr(fact, "content", None)
            }

            return [
                fact
                for fact in unique_facts
                if self._normalize_fact_content(fact.content) not in existing_contents
            ]

        existing_contents = {
            self._normalize_fact_content(fact.content)
            for fact in existing_facts
            if getattr(fact, "content", None)
        }

        new_facts = []

        for fact in extracted_facts:
            content = getattr(fact, "content", "")
            normalized_content = self._normalize_fact_content(content)

            if normalized_content and normalized_content not in existing_contents:
                new_facts.append(fact)
                existing_contents.add(normalized_content)

        return new_facts