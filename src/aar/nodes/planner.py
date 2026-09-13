from __future__ import annotations

import uuid
from typing import Any

from langgraph.types import RunnableConfig

from aar.graph.runtime import get_runtime
from aar.llm.routing import RoutingContext, route_planning
from aar.nodes.common import attempt_from_result, emit, generate_paid, request_from_state
from aar.schemas import AllowedSources, SourceHint, SubQuestion
from aar.textutil import extract_json_object


def planner(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    runtime = get_runtime(config)
    request = request_from_state(state)
    run_id = state["run_id"]
    remaining = runtime.budget.remaining(run_id, request.max_cost_usd)
    ctx = RoutingContext(
        settings=runtime.settings,
        learning=runtime.learning,
        remaining_budget=remaining,
    )
    route = route_planning(request, ctx)
    warnings: list[str] = []
    attempts: list[dict[str, Any]] = []
    parsed: list[SubQuestion] | None = None
    fallback = False

    if route.selected_alias:
        prompt = _planner_prompt(request)
        result = generate_paid(
            runtime,
            run_id=run_id,
            alias=route.selected_alias,
            prompt=prompt,
            node="planner",
            purpose="planning",
            ceiling=request.max_cost_usd,
        )
        if result and result.alias == route.selected_alias:
            attempts.append(attempt_from_result(result, purpose="planning", node="planner"))
            parsed = _parse_plan(result.text, request.allowed_sources)
        else:
            warnings.append("planner_model_failed")
            fallback = True
    else:
        warnings.append("planner_route_unavailable")
        fallback = True

    if parsed is None:
        parsed = _fallback_plan(request)
        fallback = True
        warnings.append("planner_used_deterministic_fallback")

    event = emit(
        runtime,
        run_id,
        "plan_created",
        "planner",
        count=len(parsed),
        fallback=fallback,
        alias=route.selected_alias,
    )
    return {
        "sub_questions": [item.model_dump() for item in parsed],
        "routes": [route.model_dump()],
        "planner_fallback": fallback,
        "warnings": warnings,
        "events": [event],
        "model_attempts": attempts,
    }


def _planner_prompt(request: Any) -> str:
    return (
        "Plan the research as JSON with a sub_questions array of 3 to 7 items. "
        "Each item needs text, source_hint (web|documents|both), and a measurable coverage_goal. "
        f"Allowed sources: {request.allowed_sources.value}. "
        f"Question: {request.question}"
    )


def _parse_plan(text: str, allowed: AllowedSources) -> list[SubQuestion] | None:
    payload = extract_json_object(text)
    if not payload:
        return None
    raw_items = payload.get("sub_questions") or payload.get("subquestions") or []
    if not isinstance(raw_items, list):
        return None
    questions: list[SubQuestion] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text_value = str(item.get("text") or "").strip()
        if len(text_value) < 8 or text_value.casefold() in seen:
            continue
        hint = _coerce_hint(str(item.get("source_hint") or "web"), allowed)
        goal = str(item.get("coverage_goal") or f"at least one validated fact about {text_value[:80]}")
        questions.append(
            SubQuestion(
                id=f"sq-{uuid.uuid4().hex[:8]}",
                text=text_value,
                source_hint=hint,
                coverage_goal=goal,
            )
        )
        seen.add(text_value.casefold())
    if len(questions) < 3 or len(questions) > 7:
        return None
    return questions


def _coerce_hint(raw: str, allowed: AllowedSources) -> SourceHint:
    hint = raw if raw in {"web", "documents", "both"} else "web"
    if allowed == AllowedSources.web:
        return SourceHint.web
    if allowed == AllowedSources.documents:
        return SourceHint.documents
    return SourceHint(hint)


def _fallback_plan(request: Any) -> list[SubQuestion]:
    if request.allowed_sources == AllowedSources.web:
        hint = SourceHint.web
    elif request.allowed_sources == AllowedSources.documents:
        hint = SourceHint.documents
    else:
        hint = SourceHint.both
    question = request.question
    templates = [
        (
            f"What is the core scope of this topic: {question}",
            "at least one validated fact that defines the topic scope",
        ),
        (
            f"What evidence exists about this topic: {question}",
            "at least one validated fact with retrieved evidence",
        ),
        (
            f"What limits or open questions remain for: {question}",
            "at least one validated fact about limits or remaining questions",
        ),
    ]
    return [
        SubQuestion(
            id=f"sq-fallback-{index}",
            text=text,
            source_hint=hint,
            coverage_goal=goal,
        )
        for index, (text, goal) in enumerate(templates, start=1)
    ]
