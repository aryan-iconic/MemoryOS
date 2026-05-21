from memoryos.models import Turn, MemorySearchResult, MemoryOSConfig
from typing import List, Dict, Any, Optional
from collections import deque

class WorkingMemory:
    def __init__(self, config: Optional[MemoryOSConfig] = None):
        self.config = config or MemoryOSConfig()
        self.turns = deque(maxlen=self.config.working_memory_size)

    def add_turn(self, turn: Turn):
        self.turns.append(turn)


    def get_recent_turns(self, n: Optional[int] = None) -> List[Turn]:
        turns = list(self.turns)
        if n is None:
            return turns
        return self.turns[-n:]

    def get_context_text(self, n: Optional[int] = None) -> str:
        recent_turns = self.get_recent_turns(n)
        return "\n".join(turn.as_text() for turn in recent_turns)

    def search(self, query: str) -> List[MemorySearchResult]:
        query = query.lower()
        results: List[MemorySearchResult] = []
        for turn in self.turns:
            text_lower = turn.as_text().lower()
            if query in text_lower:
                results.append(
                    MemorySearchResult(
                        content=turn.as_text(),
                        source="working",
                        score=1.0,
                        id=turn.id,
                        timestamp=turn.timestamp,
                        metadata={
                            "session_id": turn.session_id,
                            "match_type": "keyword"
                        }
                    )
                )
        return results

    def clear(self):
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)