"""SQLite storage backend for MemoryOS."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional, Union

from memoryos.exceptions import SerializationError
from memoryos.models import Fact, Turn
from memoryos.storage.base import StorageBackend
from memoryos.storage.db import SQLiteDatabase


class SQLiteStore(StorageBackend):
    """SQLite-backed implementation of the MemoryOS storage contract."""

    def __init__(self, db_path: str = "memoryos.db"):
        self.db_path = db_path
        self.db = SQLiteDatabase(db_path)

    def _connect(self) -> sqlite3.Connection:
        return self.db.connect()

    def _initialize_db(self) -> None:
        self.db.initialize()

    def save_fact(self, fact: Union[Fact, Dict[str, Any]]) -> Fact:
        fact_data = self._fact_to_dict(fact)
        now = time.time()
        fact_data.setdefault("created_at", now)
        fact_data.setdefault("timestamp", now)
        fact_data.setdefault("source", "conversation")
        fact_data.setdefault("access_count", 0)
        fact_data.setdefault("metadata", {})

        with self.db.session() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO facts (
                    id, content, type, confidence, session_id, source,
                    timestamp, access_count, embedding, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fact_data["id"],
                    fact_data["content"],
                    fact_data["type"],
                    float(fact_data["confidence"]),
                    fact_data["session_id"],
                    fact_data.get("source", "conversation"),
                    float(fact_data.get("timestamp", now)),
                    int(fact_data.get("access_count", 0)),
                    self._serialize_embedding(fact_data.get("embedding")),
                    self._json_dumps(fact_data.get("metadata", {})),
                    float(fact_data.get("created_at", now)),
                ),
            )

        if isinstance(fact, Fact):
            return fact
        return Fact.from_dict(fact_data)

    def save_facts(self, facts: List[Union[Fact, Dict[str, Any]]]) -> List[Fact]:
        return [self.save_fact(fact) for fact in facts]

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        with self.db.session() as conn:
            row = conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return self._row_to_fact(row) if row is not None else None

    def get_all_facts(self, limit: Optional[int] = None) -> List[Fact]:
        query = "SELECT * FROM facts ORDER BY timestamp DESC"
        params: List[Any] = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.db.session() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_facts_by_session(self, session_id: str, limit: Optional[int] = None) -> List[Fact]:
        query = "SELECT * FROM facts WHERE session_id = ? ORDER BY timestamp DESC"
        params: List[Any] = [session_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.db.session() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_facts_by_type(self, fact_type: str, limit: Optional[int] = None) -> List[Fact]:
        query = "SELECT * FROM facts WHERE type = ? ORDER BY timestamp DESC"
        params: List[Any] = [fact_type]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.db.session() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def search_facts_keyword(self, keyword: str, limit: Optional[int] = None) -> List[Fact]:
        query = "SELECT * FROM facts WHERE content LIKE ? ORDER BY confidence DESC, timestamp DESC"
        params: List[Any] = [f"%{keyword}%"]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.db.session() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def update_fact_access_count(self, fact_id: str) -> None:
        with self.db.session() as conn:
            conn.execute("UPDATE facts SET access_count = access_count + 1 WHERE id = ?", (fact_id,))

    def delete_fact(self, fact_id: str) -> None:
        with self.db.session() as conn:
            conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))

    def save_turn(self, turn: Union[Turn, Dict[str, Any]]) -> Dict[str, Any]:
        turn_data = self._turn_to_dict(turn)
        now = time.time()
        turn_data.setdefault("timestamp", now)
        turn_data.setdefault("created_at", now)
        turn_data.setdefault("metadata", {})
        turn_data.setdefault("ai_response", "")

        with self.db.session() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO turns (
                    id, user_message, ai_response, session_id,
                    timestamp, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_data["id"],
                    turn_data.get("user_message", ""),
                    turn_data.get("ai_response", ""),
                    turn_data["session_id"],
                    float(turn_data.get("timestamp", now)),
                    self._json_dumps(turn_data.get("metadata", {})),
                    float(turn_data.get("created_at", now)),
                ),
            )
        return turn_data

    def get_turns_by_session(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM turns WHERE session_id = ? ORDER BY timestamp DESC"
        params: List[Any] = [session_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.db.session() as conn:
            rows = conn.execute(query, params).fetchall()
        turns = [self._row_to_turn(row) for row in rows]
        return list(reversed(turns))

    def save_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        now = time.time()
        episode.setdefault("created_at", now)
        episode.setdefault("metadata", {})
        episode.setdefault("turn_count", 0)

        with self.db.session() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes (
                    id, session_id, summary, start_timestamp, end_timestamp,
                    turn_count, embedding, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode["id"],
                    episode["session_id"],
                    episode["summary"],
                    float(episode["start_timestamp"]),
                    float(episode["end_timestamp"]),
                    int(episode.get("turn_count", 0)),
                    self._serialize_embedding(episode.get("embedding")),
                    self._json_dumps(episode.get("metadata", {})),
                    float(episode.get("created_at", now)),
                ),
            )
        return episode

    def get_episodes_by_session(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM episodes WHERE session_id = ? ORDER BY created_at DESC"
        params: List[Any] = [session_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.db.session() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def get_all_episodes(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM episodes ORDER BY end_timestamp DESC"
        params: List[Any] = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.db.session() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def clear_session(self, session_id: str) -> None:
        with self.db.session() as conn:
            conn.execute("DELETE FROM facts WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM episodes WHERE session_id = ?", (session_id,))

    def clear_all(self) -> None:
        with self.db.session() as conn:
            conn.execute("DELETE FROM facts")
            conn.execute("DELETE FROM turns")
            conn.execute("DELETE FROM episodes")

    def close(self) -> None:
        self.db.close()

    def _row_to_fact(self, row: sqlite3.Row) -> Fact:
        return Fact(
            id=row["id"],
            content=row["content"],
            type=row["type"],
            confidence=float(row["confidence"]),
            session_id=row["session_id"],
            source=row["source"],
            timestamp=float(row["timestamp"]),
            access_count=int(row["access_count"] or 0),
            embedding=self._deserialize_embedding(row["embedding"]),
            metadata=self._json_loads(row["metadata"], default={}),
        )

    def _row_to_turn(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "user_message": row["user_message"],
            "ai_response": row["ai_response"],
            "session_id": row["session_id"],
            "timestamp": row["timestamp"],
            "metadata": self._json_loads(row["metadata"], default={}),
            "created_at": row["created_at"],
        }

    def _row_to_episode(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "summary": row["summary"],
            "start_timestamp": row["start_timestamp"],
            "end_timestamp": row["end_timestamp"],
            "turn_count": row["turn_count"],
            "embedding": self._deserialize_embedding(row["embedding"]),
            "metadata": self._json_loads(row["metadata"], default={}),
            "created_at": row["created_at"],
        }

    def _fact_to_dict(self, fact: Union[Fact, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(fact, dict):
            return dict(fact)
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
            return dict(turn)
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
        return self._json_dumps(embedding)

    def _deserialize_embedding(self, embedding: Optional[str]) -> Any:
        if embedding is None:
            return None
        return self._json_loads(embedding, default=embedding)

    @staticmethod
    def _json_dumps(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError as exc:
            raise SerializationError("Failed to serialize value to JSON.", details={"value_type": type(value).__name__}) from exc

    @staticmethod
    def _json_loads(value: Optional[str], default: Any) -> Any:
        if value is None or value == "":
            return default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
