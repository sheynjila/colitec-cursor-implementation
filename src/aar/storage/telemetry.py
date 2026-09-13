from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

from aar.storage.db import Database

SECRET_KEYS = re.compile(
    r"(api[_-]?key|secret|token|password|authorization|credential)",
    re.IGNORECASE,
)
SECRET_VALUE = re.compile(r"(sk-|AIza|ghp_|xox[baprs]-)[A-Za-z0-9_\-]{8,}")


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[redacted]" if SECRET_KEYS.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return SECRET_VALUE.sub("[redacted]", value)
    return value


class TelemetryStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def write(
        self,
        *,
        run_id: str,
        node: str | None = None,
        sub_question_id: str | None = None,
        alias: str | None = None,
        provider: str | None = None,
        tier: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        estimated_cost: float | None = None,
        latency_ms: float | None = None,
        tool_outcome: str | None = None,
        validation_decision: str | None = None,
        qa_result: str | None = None,
        error_code: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        safe_extra = redact(extra or {})
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO telemetry (
                    run_id, timestamp, node, sub_question_id, alias, provider, tier,
                    input_tokens, output_tokens, estimated_cost, latency_ms,
                    tool_outcome, validation_decision, qa_result, error_code, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    datetime.now(UTC).isoformat(),
                    node,
                    sub_question_id,
                    alias,
                    provider,
                    tier,
                    input_tokens,
                    output_tokens,
                    estimated_cost,
                    latency_ms,
                    tool_outcome,
                    validation_decision,
                    qa_result,
                    error_code,
                    json.dumps(safe_extra),
                ),
            )

    def rows_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self.db.lock:
            rows = self.db.conn.execute(
                "SELECT * FROM telemetry WHERE run_id = ? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        with self.db.lock:
            return self._summary()

    def _summary(self) -> dict[str, Any]:
        run_row = self.db.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(spent_usd), 0) AS cost FROM runs"
        ).fetchone()
        tier_rows = self.db.conn.execute(
            """
            SELECT COALESCE(tier, 'unknown') AS tier, COUNT(*) AS n
            FROM telemetry
            WHERE alias IS NOT NULL
            GROUP BY tier
            """
        ).fetchall()
        val = self.db.conn.execute(
            """
            SELECT
                SUM(CASE WHEN validation_decision = 'accepted' THEN 1 ELSE 0 END) AS passed,
                SUM(CASE WHEN validation_decision IS NOT NULL THEN 1 ELSE 0 END) AS total
            FROM telemetry
            """
        ).fetchone()
        premium = self.db.conn.execute(
            """
            SELECT
                SUM(CASE WHEN extra_json LIKE '%"premium": true%' THEN 1 ELSE 0 END) AS premium_calls,
                SUM(CASE WHEN alias IS NOT NULL THEN 1 ELSE 0 END) AS model_calls
            FROM telemetry
            """
        ).fetchone()
        latency = self.db.conn.execute(
            "SELECT AVG(latency_ms) AS avg_ms FROM telemetry WHERE latency_ms IS NOT NULL"
        ).fetchone()
        passed = val["passed"] or 0
        total = val["total"] or 0
        model_calls = premium["model_calls"] or 0
        premium_calls = premium["premium_calls"] or 0
        return {
            "runs": run_row["n"] or 0,
            "total_cost_usd": float(run_row["cost"] or 0),
            "calls_by_tier": {row["tier"]: row["n"] for row in tier_rows},
            "validation_pass_rate": (passed / total) if total else 0.0,
            "premium_avoidance_rate": (
                1.0 - (premium_calls / model_calls) if model_calls else 1.0
            ),
            "average_latency_ms": float(latency["avg_ms"] or 0.0),
        }
