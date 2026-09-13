from __future__ import annotations

import json
from datetime import datetime

from aar.schemas import ProgressEvent
from aar.storage.db import Database


class EventStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def append(self, event: ProgressEvent) -> ProgressEvent:
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (run_id, event_type, timestamp, node, sub_question_id, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    event.event_type,
                    event.timestamp.isoformat(),
                    event.node,
                    event.sub_question_id,
                    json.dumps(event.payload),
                ),
            )
            event.id = int(cursor.lastrowid or 0)
        return event

    def list_for_run(self, run_id: str, after_id: int = 0) -> list[ProgressEvent]:
        with self.db.lock:
            rows = self.db.conn.execute(
            """
            SELECT * FROM events
            WHERE run_id = ? AND id > ?
            ORDER BY id ASC
            """,
            (run_id, after_id),
        ).fetchall()
        return [
            ProgressEvent(
                id=row["id"],
                run_id=row["run_id"],
                event_type=row["event_type"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                node=row["node"],
                sub_question_id=row["sub_question_id"],
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        ]
