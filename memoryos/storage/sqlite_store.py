from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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

    def _initialize_db(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
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

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_facts_session
                ON facts(session_id)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_facts_type
                ON facts(type)
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_turns_session
                ON turns(session_id)
                """
            )

            conn.commit()

    def save_fact(self, fact: Union[Fact, Dict[str, Any]]) -> None:
        fact_data = self._fact_to_dict(fact)

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO facts (
                    id,
                    content,
                    type,
                    confidence,
                    session_id,
                    source,
                    timestamp,
                    access_count,
                    embedding,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact_data["id"],
                    fact_data["content"],
                    fact_data["type"],
                    fact_data["confidence"],
                    fact_data["session_id"],
                    fact_data.get("source", "conversation"),
                    fact_data.get("timestamp", time.time()),
                    fact_data.get("access_count", 0),
                    self._serialize_embedding(fact_data.get("embedding")),
                    json.dumps(fact_data.get("metadata", {}), ensure_ascii=False),
                ),
            )

            conn.commit()

    def save_facts(self, facts: List[Union[Fact, Dict[str, Any]]]) -> None:
        for fact in facts:
            self.save_fact(fact)

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))
            row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_fact(row)

    def get_all_facts(self, limit: Optional[int] = None) -> List[Fact]:
        query = """
            SELECT *
            FROM facts
            ORDER BY timestamp DESC
        """

        params: List[Any] = []

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_fact(row) for row in rows]

    def get_facts_by_session(self, session_id: str) -> List[Fact]:
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT *
                FROM facts
                WHERE session_id = ?
                ORDER BY timestamp DESC
                """,
                (session_id,),
            )

            rows = cursor.fetchall()

        return [self._row_to_fact(row) for row in rows]

    def get_facts_by_type(self, fact_type: str) -> List[Fact]:
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT *
                FROM facts
                WHERE type = ?
                ORDER BY timestamp DESC
                """,
                (fact_type,),
            )

            rows = cursor.fetchall()

        return [self._row_to_fact(row) for row in rows]

    def search_facts_keyword(self, keyword: str) -> List[Fact]:
        pattern = f"%{keyword}%"

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT *
                FROM facts
                WHERE content LIKE ?
                ORDER BY confidence DESC, timestamp DESC
                """,
                (pattern,),
            )

            rows = cursor.fetchall()

        return [self._row_to_fact(row) for row in rows]

    def update_fact_access_count(self, fact_id: str) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE facts
                SET access_count = access_count + 1
                WHERE id = ?
                """,
                (fact_id,),
            )

            conn.commit()

    def delete_fact(self, fact_id: str) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            conn.commit()

    def save_turn(self, turn: Union[Turn, Dict[str, Any]]) -> None:
        turn_data = self._turn_to_dict(turn)

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO turns (
                    id,
                    user_message,
                    ai_response,
                    session_id,
                    timestamp,
                    metadata
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_data["id"],
                    turn_data["user_message"],
                    turn_data.get("ai_response", ""),
                    turn_data["session_id"],
                    turn_data.get("timestamp", time.time()),
                    json.dumps(turn_data.get("metadata", {}), ensure_ascii=False),
                ),
            )

            conn.commit()

    def get_turns_by_session(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT *
            FROM turns
            WHERE session_id = ?
            ORDER BY timestamp DESC
        """

        params: List[Any] = [session_id]

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

        turns = [self._row_to_turn(row) for row in rows]

        return list(reversed(turns))

    def clear_session(self, session_id: str) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM facts WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))

            conn.commit()

    def clear_all(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM facts")
            cursor.execute("DELETE FROM turns")

            conn.commit()

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        metadata = self._safe_json_loads(row["metadata"], default={})
        embedding = self._deserialize_embedding(row["embedding"])

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

    def _row_to_turn(self, row: sqlite3.Row) -> Dict[str, Any]:
        metadata = self._safe_json_loads(row["metadata"], default={})

        return {
            "id": row["id"],
            "user_message": row["user_message"],
            "ai_response": row["ai_response"],
            "session_id": row["session_id"],
            "timestamp": row["timestamp"],
            "metadata": metadata,
        }

    def _fact_to_dict(self, fact: Union[Fact, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(fact, dict):
            return fact

        if hasattr(fact, "to_dict"):
            return fact.to_dict()

        return {
            "id": fact.id,
            "content": fact.content,
            "type": fact.type,
            "confidence": fact.confidence,
            "session_id": fact.session_id,
            "source": fact.source,
            "timestamp": fact.timestamp,
            "access_count": fact.access_count,
            "embedding": fact.embedding,
            "metadata": fact.metadata,
        }

    def _turn_to_dict(self, turn: Union[Turn, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(turn, dict):
            return turn

        if hasattr(turn, "to_dict"):
            return turn.to_dict()

        return {
            "id": turn.id,
            "user_message": turn.user_message,
            "ai_response": getattr(turn, "ai_response", ""),
            "session_id": turn.session_id,
            "timestamp": turn.timestamp,
            "metadata": getattr(turn, "metadata", {}),
        }

    def _serialize_embedding(self, embedding: Any) -> Optional[str]:
        if embedding is None:
            return None

        if isinstance(embedding, str):
            return embedding

        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()

        return json.dumps(embedding)

    def _deserialize_embedding(self, embedding: Optional[str]) -> Any:
        if embedding is None:
            return None

        return self._safe_json_loads(embedding, default=embedding)

    def _safe_json_loads(self, value: Optional[str], default: Any) -> Any:
        if not value:
            return default

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default