from __future__ import annotations

from typing import Any

from langgraph.types import RunnableConfig

from aar.graph.runtime import get_runtime
from aar.llm.routing import RoutingContext, route_one
from aar.llm.scoring import score_sub_question
from aar.nodes.common import emit, request_from_state
from aar.schemas import SubQuestion


def complexity_router(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    runtime = get_runtime(config)
    request = request_from_state(state)
    run_id = state["run_id"]
    remaining = runtime.budget.remaining(run_id, request.max_cost_usd)
    ctx = RoutingContext(
        settings=runtime.settings,
        learning=runtime.learning,
        remaining_budget=remaining,
    )
    updated: list[dict[str, Any]] = []
    routes: list[dict[str, Any]] = list(state.get("routes") or [])
    events: list[dict[str, Any]] = []
    for raw in state.get("sub_questions") or []:
        sub = SubQuestion.model_validate(raw)
        sub.score = score_sub_question(sub, request)
        sub.route = route_one(sub, request, ctx)
        updated.append(sub.model_dump())
        routes.append(sub.route.model_dump())
        events.append(
            emit(
                runtime,
                run_id,
                "route_selected",
                "complexity_router",
                sub_question_id=sub.id,
                alias=sub.route.selected_alias,
                tier=sub.score.tier.value,
                total=sub.score.total,
                rejection_reasons=sub.route.rejection_reasons[-5:],
            )
        )
        runtime.telemetry.write(
            run_id=run_id,
            node="complexity_router",
            sub_question_id=sub.id,
            alias=sub.route.selected_alias,
            provider=sub.route.selected_provider,
            tier=sub.score.tier.value,
            extra={"rejection_reasons": sub.route.rejection_reasons[-8:]},
        )
    return {"sub_questions": updated, "routes": routes, "events": events}
