from __future__ import annotations

from typing import Any

from langgraph.types import RunnableConfig, Send

from aar.graph.runtime import get_runtime
from aar.nodes.common import emit, request_from_state
from aar.schemas import AllowedSources, SourceHint, SubQuestion


def research_supervisor(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    runtime = get_runtime(config)
    run_id = state["run_id"]
    current_round = int(state.get("research_round") or 0)
    if current_round == 0:
        current_round = 1
    retry_ids = set(state.get("retry_sub_question_ids") or [])
    events = [
        emit(
            runtime,
            run_id,
            "worker_started",
            "research_supervisor",
            research_round=current_round,
            retry_ids=sorted(retry_ids),
        )
    ]
    runtime.registry.update(run_id, research_round=current_round)
    return {"research_round": current_round, "events": events}


def dispatch_workers(state: dict[str, Any]) -> list[Send] | str:
    request = request_from_state(state)
    retry_ids = set(state.get("retry_sub_question_ids") or [])
    sends: list[Send] = []
    for raw in state.get("sub_questions") or []:
        sub = SubQuestion.model_validate(raw)
        if not sub.active:
            continue
        if retry_ids:
            if sub.id not in retry_ids:
                continue
        elif sub.covered:
            continue
        hints = _hints_for(sub.source_hint, request.allowed_sources)
        packet = {
            "run_id": state["run_id"],
            "request": state["request"],
            "sub_question": sub.model_dump(),
            "research_round": state.get("research_round") or 1,
            "selected_alias": (sub.route.selected_alias if sub.route else None),
        }
        if "web" in hints:
            sends.append(Send("web_search_worker", {**packet, "worker": "web"}))
        if "documents" in hints:
            sends.append(Send("document_search_worker", {**packet, "worker": "documents"}))
    return sends or "research_join"


def _hints_for(hint: SourceHint, allowed: AllowedSources) -> list[str]:
    if allowed == AllowedSources.web:
        return ["web"] if hint in {SourceHint.web, SourceHint.both} else []
    if allowed == AllowedSources.documents:
        return ["documents"] if hint in {SourceHint.documents, SourceHint.both} else []
    if hint == SourceHint.web:
        return ["web"]
    if hint == SourceHint.documents:
        return ["documents"]
    return ["web", "documents"]
