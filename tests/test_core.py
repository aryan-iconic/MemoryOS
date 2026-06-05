"""Core API smoke tests."""

from __future__ import annotations

from pathlib import Path

from memoryos import MemoryOS, MemoryOSConfig


def test_core_api_smoke(tmp_path: Path) -> None:
    memory = MemoryOS(
        config=MemoryOSConfig(db_path=str(tmp_path / "core.db"), min_episode_turns=2),
        session_id="core_session",
    )
    memory.clear_all()
    memory.process_turn("My name is Aryan. I prefer dark UI.", "Saved.")
    memory.process_turn("I want to build MemoryOS.", "Great.")

    assert memory.get_all_facts()
    assert memory.search_memory("dark UI", min_score=0.0)
    assert "Recent conversation" in memory.build_context("What should you know?", limit=5)
    memory.close()
