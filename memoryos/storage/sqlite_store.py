"""SQLite storage backend implementation."""
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import Fact, Turn

class SQLiteStore:
    def __init__(self, db_path: str = "memoryos.db"):
        self.db_path = db_path
        self._ensure_parent_dir()
        self._initialize_db()

    def _ensure_parent_dir(self) -> None:
        path = Path(self.db_path)

        if path.parent and str(path.parent) != ".":
            path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _initialize_db(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(                """
                CREATE TABLE IF NOT EXISTS facts (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    embedding TEXT,
                    metadata TEXT
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    user_message TEXT NOT NULL,
                    ai_response TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    metadata TEXT
                )
                """
            )
            conn.commit()

    def save_fact(self, fact: Dict[str, Any]):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO facts (id, content, type, confidence, session_id, source, timestamp, access_count, embedding, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)   
            """, (
                fact["id"],
                fact["content"],
                fact["type"],
                fact["confidence"],
                fact["session_id"],
                fact.get("source", "conversation"),
                fact.get("timestamp", time.time()),
                fact.get("access_count", 0),
                sqlite3.Binary(pickle.dumps(fact.get("embedding"))) if fact.get("embedding") else None,
                json.dumps(fact.get("metadata", {}))
            ))
            conn.commit()
    
    def get_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_fact(row).to_dict()
        return None

    def get_facts_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM facts WHERE session_id = ?", (session_id,))
            rows = cursor.fetchall()
            return [self._row_to_fact(row).to_dict() for row in rows]
    
    def update_fact_access_count(self, fact_id: str):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE facts SET access_count = access_count + 1 WHERE id = ?", (access_count, fact_id))
            conn.commit()

    def delete_fact(self, fact_id: str):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            conn.commit()

        def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        embedding = json.loads(row["embedding"]) if row["embedding"] else None
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return Fact(
            id=row["id"],
            content=row["content"],
            type=row["type"],
            confidence=row["confidence"],
            session_id=row["session_id"],
            source=row["source"],
            timestamp=row["timestamp"],
            access_count=row["access_count"],
            embedding=embedding,
            metadata=metadata,
        )

    def save_turn(self, turn: Dict[str, Any]):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO turns (id, user_message, ai_response, session_id, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                turn["id"],
                turn["user_message"],
                turn["ai_response"],
                turn["session_id"],
                turn.get("timestamp", time.time()),
                json.dumps(turn.get("metadata", {}))
            ))
            conn.commit()

    def get_turns_by_session(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        
            query = """
            SELECT * FROM turns
            WHERE session_id = ?
            ORDER BY timestamp DESC
        """
            params = [session_id]
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
        with self._connect() as conn:            
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            turns = [self._row_to_turn(row) for row in rows]
            return list(reversed(turns))

    def _row_to_turn(self, row: sqlite3.Row) -> Dict[str, Any]:
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        return {
            "id": row["id"],
            "user_message": row["user_message"],
            "ai_response": row["ai_response"],
            "session_id": row["session_id"],
            "timestamp": row["timestamp"],
            "metadata": metadata,
        }       

    def clear_session(self, session_id: str) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM facts WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))

            conn.commit()