from __future__ import annotations

from aar.schemas import ProvenanceRecord
from aar.storage.db import Database


class ProvenanceLedger:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, run_id: str, record: ProvenanceRecord) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO provenance (
                    run_id, ref, title, source_kind, text, document_id, chunk_id,
                    retrieved_url, labeled_fixture
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, ref) DO UPDATE SET
                    title = excluded.title,
                    text = excluded.text
                """,
                (
                    run_id,
                    record.ref,
                    record.title,
                    record.source_kind,
                    record.text,
                    record.document_id,
                    record.chunk_id,
                    record.retrieved_url,
                    1 if record.labeled_fixture else 0,
                ),
            )

    def get(self, run_id: str, ref: str) -> ProvenanceRecord | None:
        row = self.db.conn.execute(
            "SELECT * FROM provenance WHERE run_id = ? AND ref = ?",
            (run_id, ref),
        ).fetchone()
        if row is None:
            return None
        return ProvenanceRecord(
            ref=row["ref"],
            title=row["title"],
            source_kind=row["source_kind"],
            text=row["text"],
            document_id=row["document_id"],
            chunk_id=row["chunk_id"],
            retrieved_url=row["retrieved_url"],
            labeled_fixture=bool(row["labeled_fixture"]),
        )

    def list_for_run(self, run_id: str) -> list[ProvenanceRecord]:
        rows = self.db.conn.execute(
            "SELECT * FROM provenance WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [
            ProvenanceRecord(
                ref=row["ref"],
                title=row["title"],
                source_kind=row["source_kind"],
                text=row["text"],
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
                retrieved_url=row["retrieved_url"],
                labeled_fixture=bool(row["labeled_fixture"]),
            )
            for row in rows
        ]
