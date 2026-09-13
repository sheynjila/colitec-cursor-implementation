from __future__ import annotations

import threading
import uuid
from typing import Any

from aar.errors import safe_error_code, safe_error_message
from aar.graph.builder import compile_graph, get_compiled_graph
from aar.graph.runtime import AppRuntime
from aar.graph.state import initial_state
from aar.nodes.common import emit
from aar.schemas import (
    ProgressEvent,
    ResearchRequest,
    RunStatus,
)
from aar.storage.telemetry import redact


class RunService:
    def __init__(self, runtime: AppRuntime, graph: Any | None = None) -> None:
        self.runtime = runtime
        self.graph = graph or get_compiled_graph(str(runtime.settings.checkpoint_db_path))

    def create_run(self, request: ResearchRequest) -> str:
        run_id = str(uuid.uuid4())
        self.runtime.registry.create(request, run_id)
        thread = threading.Thread(target=self.execute, args=(run_id, request), daemon=True)
        thread.start()
        return run_id

    def execute(self, run_id: str, request: ResearchRequest) -> dict[str, Any]:
        config = {
            "configurable": {
                "thread_id": run_id,
                "app_runtime": self.runtime,
            }
        }
        try:
            emit(self.runtime, run_id, "run_started", "run_service", status="queued")
            result = self.graph.invoke(initial_state(run_id, request), config=config)
            spent, reserved = self.runtime.budget.snapshot(run_id)
            report = result.get("report")
            self.runtime.registry.update(
                run_id,
                status=RunStatus.succeeded,
                research_round=result.get("research_round") or 0,
                warnings=result.get("warnings") or [],
                spent_usd=spent,
                reserved_usd=reserved,
                result={
                    "report": report,
                    "facts": result.get("facts") or [],
                    "limitations": result.get("limitations") or [],
                    "qa": result.get("qa"),
                    "planner_fallback": result.get("planner_fallback"),
                    "writer_fallback": result.get("writer_fallback"),
                    "model_attempts": result.get("model_attempts") or [],
                    "routes": result.get("routes") or [],
                },
            )
            return result
        except Exception as exc:  # noqa: BLE001
            spent, reserved = self.runtime.budget.snapshot(run_id)
            self.runtime.registry.update(
                run_id,
                status=RunStatus.failed,
                spent_usd=spent,
                reserved_usd=reserved,
                error_code=safe_error_code(exc),
                error_message=safe_error_message(exc),
            )
            self.runtime.events.append(
                ProgressEvent(
                    run_id=run_id,
                    event_type="run_failed",
                    node="run_service",
                    payload=redact({"code": safe_error_code(exc)}),
                )
            )
            self.runtime.telemetry.write(
                run_id=run_id,
                node="run_service",
                error_code=safe_error_code(exc),
            )
            raise

    def store_feedback(self, run_id: str, rating: int, note: str) -> None:
        self.runtime.registry.get(run_id)
        with self.runtime.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO feedback (run_id, rating, note, created_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(run_id) DO UPDATE SET rating = excluded.rating, note = excluded.note
                """,
                (run_id, rating, note),
            )
        self.runtime.telemetry.write(
            run_id=run_id,
            node="feedback",
            extra={"rating": rating, "has_note": bool(note)},
        )


def compile_once(runtime: AppRuntime, graph: Any | None = None) -> Any:
    if graph is not None:
        return graph
    return compile_graph()
