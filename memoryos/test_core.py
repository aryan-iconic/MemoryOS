"""Manual package-level smoke test.

Prefer running the real pytest suite with:

    python -m pytest
"""

from memoryos import MemoryOS

if __name__ == "__main__":
    memory = MemoryOS(db_path="memoryos_test.db", session_id="test_session")
    memory.clear_all()
    memory.process_turn("My name is Aryan. I prefer dark UI. I want to build MemoryOS.", "Nice.")
    print(memory.build_context("What does the user prefer?", limit=5))
