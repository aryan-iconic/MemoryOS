"""Build final prompt context from retrieved memories."""

from __future__ import annotations

from typing import Iterable, List, Optional

from memoryos.models import MemorySearchResult, Turn


class PromptContextBuilder:
    """Formats MemoryOS memory results into a clean prompt context block."""

    def __init__(
        self,
        *,
        max_chars: int = 4000,
        include_scores: bool = True,
    ):
        self.max_chars = max_chars
        self.include_scores = include_scores

    def build(
        self,
        *,
        query: str = "",
        results: Optional[Iterable[MemorySearchResult]] = None,
        recent_turns: Optional[Iterable[Turn]] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        budget = max_chars or self.max_chars
        sections: List[str] = []
        results_list = list(results or [])

        semantic = [item for item in results_list if item.source == "semantic"]
        episodic = [item for item in results_list if item.source == "episodic"]
        working = [item for item in results_list if item.source == "working"]

        if semantic:
            sections.append(self._format_results("Relevant user facts", semantic))

        if episodic:
            sections.append(self._format_results("Relevant past conversation summaries", episodic))

        if working:
            sections.append(self._format_results("Matching recent conversation", working))  # pragma: no cover

        if recent_turns:
            recent_text = self._format_turns(list(recent_turns))
            if recent_text:
                sections.append(recent_text)

        context = "\n\n".join(section for section in sections if section.strip())
        return self._trim(context, budget)

    def build_from_layers(
        self,
        *,
        memory_context: str = "",
        episodic_context: str = "",
        working_context: str = "",
        max_chars: Optional[int] = None,
    ) -> str:
        budget = max_chars or self.max_chars  # pragma: no cover
        sections = [  # pragma: no cover
            memory_context.strip(),
            episodic_context.strip(),
            working_context.strip(),
        ]
        return self._trim("\n\n".join(section for section in sections if section), budget)  # pragma: no cover

    def _format_results(self, title: str, results: List[MemorySearchResult]) -> str:
        lines = [f"{title}:"]
        for item in results:
            content = " ".join((item.content or "").split())
            if not content:
                continue  # pragma: no cover

            detail_parts = []
            fact_type = item.type or (item.metadata or {}).get("fact_type")
            if fact_type:
                detail_parts.append(f"type={fact_type}")
            if self.include_scores:
                detail_parts.append(f"score={item.score:.3f}")
            confidence = item.confidence or (item.metadata or {}).get("original_confidence")
            if confidence is not None:
                detail_parts.append(f"confidence={float(confidence):.2f}")

            suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""
            lines.append(f"- {content}{suffix}")
        return "\n".join(lines)

    def _format_turns(self, turns: List[Turn]) -> str:
        if not turns:
            return ""  # pragma: no cover
        lines = ["Recent conversation:"]
        for turn in turns:
            if hasattr(turn, "as_text"):
                lines.append(turn.as_text())
            elif isinstance(turn, dict):  # pragma: no cover
                lines.append(  # pragma: no cover
                    f"User: {turn.get('user_message', '')}\nAI: {turn.get('ai_response', '')}"
                )
        return "\n".join(lines)

    @staticmethod
    def _trim(text: str, max_chars: int) -> str:
        if not text or len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "..."


ContextBuilder = PromptContextBuilder
MemoryContextBuilder = PromptContextBuilder
