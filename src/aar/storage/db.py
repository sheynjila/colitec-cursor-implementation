"""Shared SQLite connection for application stores (not LangGraph checkpoints)."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class Database:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.isolation_level = None
        self.lock = threading.RLock()
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                yield self.conn
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                request_json TEXT NOT NULL,
                research_round INTEGER NOT NULL DEFAULT 0,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                spent_usd REAL NOT NULL DEFAULT 0,
                reserved_usd REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                error_code TEXT,
                error_message TEXT,
                result_json TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                node TEXT,
                sub_question_id TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                node TEXT,
                sub_question_id TEXT,
                alias TEXT,
                provider TEXT,
                tier TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                estimated_cost REAL,
                latency_ms REAL,
                tool_outcome TEXT,
                validation_decision TEXT,
                qa_result TEXT,
                error_code TEXT,
                extra_json TEXT
            );

            CREATE TABLE IF NOT EXISTS learning (
                alias TEXT PRIMARY KEY,
                pass_ema REAL NOT NULL DEFAULT 0,
                cost_ema REAL NOT NULL DEFAULT 0,
                latency_ema REAL NOT NULL DEFAULT 0,
                samples INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS provenance (
                run_id TEXT NOT NULL,
                ref TEXT NOT NULL,
                title TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                text TEXT NOT NULL,
                document_id TEXT,
                chunk_id TEXT,
                retrieved_url TEXT,
                labeled_fixture INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (run_id, ref)
            );

            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                citation TEXT NOT NULL,
                PRIMARY KEY (doc_id, chunk_id)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                doc_id, chunk_id, title, text, citation
            );

            CREATE TABLE IF NOT EXISTS feedback (
                run_id TEXT PRIMARY KEY,
                rating INTEGER NOT NULL,
                note TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

    def close(self) -> None:
        self.conn.close()
