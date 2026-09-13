from __future__ import annotations

from typing import Any

from langgraph.types import RunnableConfig

from aar.graph.runtime import get_runtime
from aar.nodes.common import emit, request_from_state
from aar.schemas import RunStatus


def request_intake(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    runtime = get_runtime(config)
    request = request_from_state(state)
    run_id = state["run_id"]
    if not request.question or request.max_cost_usd < 0:
        raise ValueError("Request intake received incomplete state.")
    runtime.registry.update(run_id, status=RunStatus.running)
    event = emit(
        runtime,
        run_id,
        "intake",
        "request_intake",
        allowed_sources=request.allowed_sources.value,
        desired_depth=request.desired_depth.value,
        risk_level=request.risk_level.value,
    )
    runtime.telemetry.write(run_id=run_id, node="request_intake", extra={"status": "running"})
    return {"status": RunStatus.running.value, "events": [event]}
