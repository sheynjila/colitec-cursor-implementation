from __future__ import annotations

from collections import defaultdict
from typing import Any

from langgraph.types import RunnableConfig

from aar.graph.runtime import get_runtime
from aar.nodes.common import emit
from aar.schemas import Fact, ModelAttempt, QAStatus, ResearchReport


def learning_update(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    runtime = get_runtime(config)
    run_id = state["run_id"]
    report = ResearchReport.model_validate(state["report"]) if state.get("report") else None
    facts = [Fact.model_validate(item) for item in state.get("facts") or []]
    attempts = [ModelAttempt.model_validate(item) for item in state.get("model_attempts") or []]
    qa_pass = bool(report and report.qa_status == QAStatus.approved)
    validated = sum(1 for fact in facts if fact.validated)
    validation_rate = (validated / len(facts)) if facts else (1.0 if qa_pass else 0.0)

    by_alias: dict[str, list[ModelAttempt]] = defaultdict(list)
    for attempt in attempts:
        by_alias[attempt.alias].append(attempt)

    warning = None
    try:
        for alias, group in by_alias.items():
            cost = sum(item.actual_cost_usd for item in group) / len(group)
            latency = sum(item.latency_ms for item in group) / len(group)
            runtime.learning.update(
                alias,
                passed=qa_pass and validation_rate >= 0.5,
                cost_usd=cost,
                latency_ms=latency,
            )
    except Exception:  # noqa: BLE001
        warning = "learning_store_failed"

    event = emit(
        runtime,
        run_id,
        "run_completed",
        "learning_update",
        qa_status=report.qa_status.value if report else "missing",
        learning_failed=bool(warning),
    )
    updates: dict[str, Any] = {"events": [event]}
    if warning:
        updates["warnings"] = [warning]
    return updates
