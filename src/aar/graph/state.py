from __future__ import annotations

from typing import Annotated, Any, TypedDict

from aar.schemas import Fact, ProgressEvent, ProvenanceRecord, ResearchRequest


def merge_by_id(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in left + right:
        key = item.get("id") or item.get("fact_id")
        if not key:
            continue
        merged[key] = {**merged.get(key, {}), **item}
    return list(merged.values())


def merge_provenance(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {item["ref"]: item for item in left}
    for item in right:
        ref = item.get("ref")
        if not ref:
            continue
        merged[ref] = {**merged.get(ref, {}), **item}
    return list(merged.values())


def extend_unique(left: list[Any], right: list[Any]) -> list[Any]:
    out = list(left)
    for item in right:
        if item not in out:
            out.append(item)
    return out


def replace(left: Any, right: Any) -> Any:
    return right if right is not None else left


class GraphState(TypedDict, total=False):
    run_id: str
    request: dict[str, Any]
    status: str
    research_round: int
    sub_questions: list[dict[str, Any]]
    facts: Annotated[list[dict[str, Any]], merge_by_id]
    validation_results: Annotated[list[dict[str, Any]], merge_by_id]
    provenance: Annotated[list[dict[str, Any]], merge_provenance]
    warnings: Annotated[list[str], extend_unique]
    events: Annotated[list[dict[str, Any]], extend_unique]
    routes: list[dict[str, Any]]
    model_attempts: Annotated[list[dict[str, Any]], extend_unique]
    limitations: Annotated[list[str], extend_unique]
    report: dict[str, Any] | None
    qa: dict[str, Any] | None
    planner_fallback: bool
    writer_fallback: bool
    retry_sub_question_ids: list[str]
    coverage_complete: bool


def initial_state(run_id: str, request: ResearchRequest) -> GraphState:
    return {
        "run_id": run_id,
        "request": request.model_dump(),
        "status": "queued",
        "research_round": 0,
        "sub_questions": [],
        "facts": [],
        "validation_results": [],
        "provenance": [],
        "warnings": [],
        "events": [],
        "routes": [],
        "model_attempts": [],
        "limitations": [],
        "report": None,
        "qa": None,
        "planner_fallback": False,
        "writer_fallback": False,
        "retry_sub_question_ids": [],
        "coverage_complete": False,
    }


def facts_from_state(state: GraphState) -> list[Fact]:
    return [Fact.model_validate(item) for item in state.get("facts") or []]


def provenance_from_state(state: GraphState) -> list[ProvenanceRecord]:
    return [ProvenanceRecord.model_validate(item) for item in state.get("provenance") or []]


def event_payload(event: ProgressEvent) -> dict[str, Any]:
    return event.model_dump(mode="json")
