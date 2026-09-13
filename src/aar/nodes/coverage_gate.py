from __future__ import annotations

from typing import Any

from langgraph.types import RunnableConfig

from aar.graph.runtime import get_runtime
from aar.llm.catalog import get_spec
from aar.nodes.common import emit, request_from_state
from aar.schemas import Fact, SubQuestion
from aar.textutil import content_words


def is_covered(sub: SubQuestion, facts: list[Fact]) -> bool:
    for fact in facts:
        if not fact.validated or fact.sub_question_id != sub.id:
            continue
        target = f"{sub.text} {sub.coverage_goal}"
        if len(content_words(fact.claim) & content_words(target)) >= 2:
            return True
    return False


def coverage_gate(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    runtime = get_runtime(config)
    request = request_from_state(state)
    run_id = state["run_id"]
    facts = [Fact.model_validate(item) for item in state.get("facts") or []]
    updated: list[SubQuestion] = []
    uncovered: list[str] = []
    for raw in state.get("sub_questions") or []:
        sub = SubQuestion.model_validate(raw)
        sub.covered = is_covered(sub, facts)
        updated.append(sub)
        if not sub.covered:
            uncovered.append(sub.id)

    research_round = int(state.get("research_round") or 1)
    remaining = runtime.budget.remaining(run_id, request.max_cost_usd)
    can_retry = (
        bool(uncovered)
        and research_round < runtime.settings.max_research_rounds
        and _budget_allows_retry(updated, remaining)
    )
    limitations: list[str] = []
    retry_ids: list[str] = []
    next_round = research_round
    if can_retry:
        retry_ids = uncovered
        next_round = research_round + 1
    else:
        if uncovered:
            if research_round >= runtime.settings.max_research_rounds:
                limitations.append("coverage_limited_by_rounds")
            if remaining <= 0:
                limitations.append("coverage_limited_by_budget")
            if any(
                sub.route and sub.route.selected_alias is None
                for sub in updated
                if sub.id in uncovered
            ):
                limitations.append("coverage_limited_by_route")
            if request.allowed_sources.value != "both":
                limitations.append("coverage_limited_by_sources")
            limitations.append("insufficient_evidence_for_some_subquestions")

    event = emit(
        runtime,
        run_id,
        "coverage_decision",
        "coverage_gate",
        uncovered=uncovered,
        retry=bool(retry_ids),
        research_round=next_round,
        limitations=limitations,
    )
    runtime.registry.update(run_id, research_round=next_round)
    return {
        "sub_questions": [item.model_dump() for item in updated],
        "retry_sub_question_ids": retry_ids,
        "research_round": next_round,
        "limitations": limitations,
        "coverage_complete": not uncovered,
        "events": [event],
    }


def route_after_coverage(state: dict[str, Any]) -> str:
    if state.get("retry_sub_question_ids"):
        return "research_supervisor"
    return "writer"


def _budget_allows_retry(subs: list[SubQuestion], remaining: float) -> bool:
    if remaining <= 0:
        needed_paid = False
        for sub in subs:
            if sub.covered or not sub.route or not sub.route.selected_alias:
                continue
            spec = get_spec(sub.route.selected_alias)
            if spec and spec.paid:
                needed_paid = True
        return not needed_paid
    return True
