"""Short-term working memory for recent conversation turns."""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from memoryos.models import MemoryOSConfig, MemorySearchResult, Turn


class WorkingMemory:
    def __init__(self, config: Optional[MemoryOSConfig] = None):
        self.config = config or MemoryOSConfig()
        self.turns: Deque[Turn] = deque(maxlen=self.config.working_memory_size)

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def get_recent_turns(self, n: Optional[int] = None) -> List[Turn]:
        turns = list(self.turns)
        if n is None:
            return turns
        return turns[-n:]

    def get_context_text(self, n: Optional[int] = None) -> str:
        recent_turns = self.get_recent_turns(n)
        return "\n".join(turn.as_text() for turn in recent_turns)

    def build_context(self, limit: Optional[int] = None, max_chars: int = 2000) -> str:
        context = self.get_context_text(limit)
        if not context:
            return ""
        context = "Recent conversation:\n" + context
        if len(context) > max_chars:
            context = "..." + context[-max_chars:].lstrip()
        return context

    def search(self, query: str) -> List[MemorySearchResult]:
        normalized_query = query.lower().strip()
        if not normalized_query:
            return []

        results: List[MemorySearchResult] = []
        for turn in self.turns:
            text = turn.as_text()
            if normalized_query in text.lower():
                results.append(
                    MemorySearchResult(
                        content=text,
                        source="working",
                        score=1.0,
                        id=turn.id,
                        timestamp=turn.timestamp,
                        metadata={
                            "session_id": turn.session_id,
                            "match_type": "keyword",
                        },
                    )
                )
        return results

    def clear(self) -> None:
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)
