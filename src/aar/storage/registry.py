from __future__ import annotations

import json
from datetime import UTC, datetime

from aar.errors import NotFoundError
from aar.schemas import ResearchReport, ResearchRequest, RunRecord, RunStatus
from aar.storage.db import Database


def _now() -> str:
    return datetime.now(UTC).isoformat()


class RunRegistry:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, request: ResearchRequest, run_id: str) -> RunRecord:
        now = _now()
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, status, request_json, research_round, warnings_json,
                    spent_usd, reserved_usd, created_at, updated_at
                ) VALUES (?, ?, ?, 0, '[]', 0, 0, ?, ?)
                """,
                (run_id, RunStatus.queued.value, request.model_dump_json(), now, now),
            )
        return self.get(run_id)

    def get(self, run_id: str) -> RunRecord:
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Run {run_id} was not found.")
        return RunRecord(
            run_id=row["run_id"],
            status=RunStatus(row["status"]),
            request=ResearchRequest.model_validate_json(row["request_json"]),
            research_round=row["research_round"],
            warnings=json.loads(row["warnings_json"]),
            spent_usd=row["spent_usd"],
            reserved_usd=row["reserved_usd"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
        )

    def update(
        self,
        run_id: str,
        *,
        status: RunStatus | None = None,
        research_round: int | None = None,
        warnings: list[str] | None = None,
        spent_usd: float | None = None,
        reserved_usd: float | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        result: dict | None = None,
    ) -> None:
        current = self.get(run_id)
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE runs SET
                    status = ?,
                    research_round = ?,
                    warnings_json = ?,
                    spent_usd = ?,
                    reserved_usd = ?,
                    error_code = ?,
                    error_message = ?,
                    result_json = COALESCE(?, result_json),
                    updated_at = ?
                WHERE run_id = ?
                """,
                (
                    (status or current.status).value,
                    current.research_round if research_round is None else research_round,
                    json.dumps(warnings if warnings is not None else current.warnings),
                    current.spent_usd if spent_usd is None else spent_usd,
                    current.reserved_usd if reserved_usd is None else reserved_usd,
                    error_code if error_code is not None else current.error_code,
                    error_message if error_message is not None else current.error_message,
                    json.dumps(result) if result is not None else None,
                    _now(),
                    run_id,
                ),
            )

    def get_result(self, run_id: str) -> dict | None:
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT result_json FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError(f"Run {run_id} was not found.")
        if not row["result_json"]:
            return None
        return json.loads(row["result_json"])

    def get_report(self, run_id: str) -> ResearchReport | None:
        result = self.get_result(run_id)
        if not result or not result.get("report"):
            return None
        return ResearchReport.model_validate(result["report"])
