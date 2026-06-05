"""MemoryOS smoke/debug runner.

Run from the project root:

    python debug.py

It creates a temporary SQLite database, processes a few turns, searches memory,
creates an episode, and prints a compact health report.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

from memoryos import MemoryOS, MemoryOSConfig


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="memoryos_debug_") as tmp:
        db_path = str(Path(tmp) / "memoryos_debug.db")
        config = MemoryOSConfig(
            db_path=db_path,
            working_memory_size=8,
            min_episode_turns=2,
            episode_turn_window=4,
            semantic_top_k=5,
            episodic_top_k=3,
        )
        memory = MemoryOS(config=config, session_id="debug_session")
        memory.clear_all()

        turns = [
            ("My name is Aryan. I prefer dark UI. I want to build MemoryOS.", "Great, I will remember that."),
            ("I am working on an AI memory system. I decided to keep it model agnostic.", "That is a strong direction."),
            ("Remember that MemoryOS should support SQLite and vector search.", "Saved as durable context."),
        ]

        processed = []
        for user_message, ai_response in turns:
            result = memory.process_turn(user_message=user_message, ai_response=ai_response)
            processed.append(
                {
                    "new_facts": [fact.content for fact in result["new_facts"]],
                    "saved_facts": [fact.content for fact in result["saved_facts"]],
                }
            )

        manual_fact = memory.add_memory(
            "User wants MemoryOS to be a reusable memory layer for AI applications.",
            fact_type="goal",
            confidence=0.95,
        )
        episode = memory.maybe_create_episode()
        search_results = memory.search_memory("What UI preference and project should be remembered?", top_k=5)
        context = memory.build_prompt_context("What should the assistant know about this user?", memory_limit=5)

        report: Dict[str, Any] = {
            "database": db_path,
            "turn_count": len(memory.get_turns()),
            "fact_count": len(memory.get_all_facts()),
            "episode_created": episode is not None,
            "manual_fact": manual_fact.content,
            "search_results": [item.content for item in search_results],
            "context_preview": context[:800],
            "processed": processed,
        }

        print(json.dumps(report, indent=2, ensure_ascii=False))
        memory.close()


if __name__ == "__main__":
    main()
