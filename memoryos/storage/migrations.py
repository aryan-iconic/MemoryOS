"""Database migration helpers for MemoryOS storage."""

from __future__ import annotations

from memoryos.storage.sqlite_store import SQLiteStore


def migrate_database(old_db_path: str, new_db_path: str) -> dict[str, int]:
    """Copy facts, turns, and episodes between two SQLite MemoryOS databases."""
    old_store = SQLiteStore(old_db_path)
    new_store = SQLiteStore(new_db_path)
    counts = {"facts": 0, "turns": 0, "episodes": 0}
    try:
        for fact in old_store.get_all_facts():
            new_store.save_fact(fact)
            counts["facts"] += 1

        session_ids = {fact.session_id for fact in old_store.get_all_facts()}
        for episode in old_store.get_all_episodes():
            session_ids.add(str(episode.get("session_id", "")))
            new_store.save_episode(dict(episode))
            counts["episodes"] += 1

        for session_id in sorted(session for session in session_ids if session):
            for turn in old_store.get_turns_by_session(session_id):
                new_store.save_turn(dict(turn))
                counts["turns"] += 1
    finally:
        old_store.close()
        new_store.close()
    return counts


__all__ = ["migrate_database"]
