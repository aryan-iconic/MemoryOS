"""Low-level SQLite database helper for MemoryOS storage backends."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from memoryos.exceptions import DatabaseError


class SQLiteDatabase:
    """Small SQLite helper that owns connection creation and schema setup."""

    def __init__(self, db_path: str = "memoryos.db", *, initialize: bool = True):
        self.db_path = db_path
        self._ensure_parent_dir()
        if initialize:
            self.initialize()

    def _ensure_parent_dir(self) -> None:
        path = Path(self.db_path)
        if path.parent and str(path.parent) != ".":
            path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            return conn
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            raise DatabaseError(
                "Failed to connect to SQLite database.",
                details={"db_path": self.db_path},
            ) from exc

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self.connect()
            yield conn
            conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover
            if conn is not None:  # pragma: no cover
                conn.rollback()  # pragma: no cover
            raise DatabaseError(  # pragma: no cover
                "SQLite operation failed.", details={"db_path": self.db_path}
            ) from exc
        finally:
            if conn is not None:
                conn.close()

    def initialize(self) -> None:
        """Create the core MemoryOS tables and indexes if missing.

        Existing early MemoryOS databases are migrated gently by adding missing
        columns. SQLite cannot alter every constraint, but these additions keep
        old local databases usable.
        """
        with self.session() as conn:
            cursor = conn.cursor()
            cursor.executescript(SCHEMA_SQL)
            self._migrate_columns(conn)

    def _migrate_columns(self, conn: sqlite3.Connection) -> None:
        columns = {
            table: {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for table in ("facts", "turns", "episodes")
        }

        migrations = []
        if "created_at" not in columns.get("facts", set()):
            migrations.append("ALTER TABLE facts ADD COLUMN created_at REAL")  # pragma: no cover
        if "created_at" not in columns.get("turns", set()):
            migrations.append("ALTER TABLE turns ADD COLUMN created_at REAL")  # pragma: no cover
        if "created_at" not in columns.get("episodes", set()):
            migrations.append("ALTER TABLE episodes ADD COLUMN created_at REAL")  # pragma: no cover

        for sql in migrations:
            conn.execute(sql)  # pragma: no cover

        now_expr = "strftime('%s','now')"
        for table in ("facts", "turns", "episodes"):
            if "created_at" in columns.get(table, set()) or any(table in sql for sql in migrations):
                conn.execute(f"UPDATE {table} SET created_at = COALESCE(created_at, {now_expr})")

    def close(self) -> None:
        """SQLiteDatabase opens short-lived connections, so nothing is kept open."""
        return None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    type TEXT NOT NULL,
    confidence REAL NOT NULL,
    session_id TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'conversation',
    timestamp REAL NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    embedding TEXT,
    metadata TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    metadata TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    summary TEXT NOT NULL,
    start_timestamp REAL NOT NULL,
    end_timestamp REAL NOT NULL,
    turn_count INTEGER NOT NULL DEFAULT 0,
    embedding TEXT,
    metadata TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id);
CREATE INDEX IF NOT EXISTS idx_facts_type ON facts(type);
CREATE INDEX IF NOT EXISTS idx_facts_timestamp ON facts(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON turns(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(end_timestamp DESC);
"""


Database = SQLiteDatabase
