"""Database migrations for storage."""
from .sqlite_store import SQLiteStore

def migrate_database(old_db_path: str, new_db_path: str) -> None:
    old_store = SQLiteStore(old_db_path)
    new_store = SQLiteStore(new_db_path)
    old_store.connect()
    new_store.connect()
    old_store.create_tables()
    new_store.create_tables()
    # Migrate facts
    old_facts = old_store.get_all_facts()
    for fact in old_facts:
        new_store.add_fact(
            content=fact.content,
            type=fact.type,
            confidence=fact.confidence,
            session_id=fact.session_id,
            source=fact.source,
            timestamp=fact.timestamp,
            access_count=fact.access_count,
            embedding=fact.embedding,
            metadata=fact.metadata,
        )
    # Migrate turns
    old_turns = old_store.get_all_turns()
    for turn in old_turns:
        new_store.add_turn(
            user_message=turn.user_message,
            ai_response=turn.ai_response,
            session_id=turn.session_id,
            timestamp=turn.timestamp,
            metadata=turn.metadata,
        )
    old_store.close()
    new_store.close()