from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from aar.storage.db import Database

ALPHA = 0.3


class LearningStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, alias: str) -> dict[str, Any]:
        with self.db.lock:
            row = self.db.conn.execute(
                "SELECT * FROM learning WHERE alias = ?", (alias,)
            ).fetchone()
        if row is None:
            return {
                "alias": alias,
                "pass_ema": 0.0,
                "cost_ema": 0.0,
                "latency_ema": 0.0,
                "samples": 0,
                "preference_score": 0.0,
            }
        return {
            "alias": alias,
            "pass_ema": row["pass_ema"],
            "cost_ema": row["cost_ema"],
            "latency_ema": row["latency_ema"],
            "samples": row["samples"],
            "preference_score": preference_score(
                row["pass_ema"], row["cost_ema"], row["latency_ema"]
            ),
        }

    def all_scores(self) -> dict[str, dict[str, Any]]:
        with self.db.lock:
            rows = self.db.conn.execute("SELECT alias FROM learning").fetchall()
        return {row["alias"]: self.get(row["alias"]) for row in rows}

    def update(
        self,
        alias: str,
        *,
        passed: bool,
        cost_usd: float,
        latency_ms: float,
    ) -> dict[str, float]:
        current = self.get(alias)
        samples = int(current["samples"]) + 1
        if samples == 1 and current["samples"] == 0:
            pass_ema = 1.0 if passed else 0.0
            cost_ema = cost_usd
            latency_ema = latency_ms
        else:
            pass_ema = ALPHA * (1.0 if passed else 0.0) + (1 - ALPHA) * current["pass_ema"]
            cost_ema = ALPHA * cost_usd + (1 - ALPHA) * current["cost_ema"]
            latency_ema = ALPHA * latency_ms + (1 - ALPHA) * current["latency_ema"]
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO learning (alias, pass_ema, cost_ema, latency_ema, samples, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(alias) DO UPDATE SET
                    pass_ema = excluded.pass_ema,
                    cost_ema = excluded.cost_ema,
                    latency_ema = excluded.latency_ema,
                    samples = excluded.samples,
                    updated_at = excluded.updated_at
                """,
                (
                    alias,
                    pass_ema,
                    cost_ema,
                    latency_ema,
                    samples,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return self.get(alias)


def preference_score(pass_ema: float, cost_ema: float, latency_ema: float) -> float:
    return (
        0.5 * pass_ema
        + 0.3 / (1.0 + max(cost_ema, 0.0))
        + 0.2 / (1.0 + max(latency_ema, 0.0) / 1000.0)
    )
