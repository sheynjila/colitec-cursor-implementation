from __future__ import annotations

from pathlib import Path

from aar.api.service import RunService
from aar.config import Settings
from aar.graph.runtime import build_runtime
from aar.nodes.coverage_gate import coverage_gate, is_covered
from aar.nodes.research_supervisor import dispatch_workers
from aar.schemas import (
    AllowedSources,
    DesiredDepth,
    Fact,
    ResearchRequest,
    RiskLevel,
    RouteDecision,
    SourceHint,
    SubQuestion,
)


def _request(question: str = "What evidence-grounded constraints should a research assistant enforce?") -> ResearchRequest:
    return ResearchRequest(
        question=question,
        allowed_sources=AllowedSources.both,
        desired_depth=DesiredDepth.standard,
        risk_level=RiskLevel.low,
        max_cost_usd=0.0,
    )


def test_compiled_run_alias_equals_invoked(service: RunService) -> None:
    run_id = "alias-match"
    service.runtime.registry.create(_request(), run_id)
    result = service.execute(run_id, _request())
    selected = {
        route["selected_alias"]
        for route in result.get("routes") or []
        if route.get("selected_alias")
    }
    invoked = {attempt["alias"] for attempt in result.get("model_attempts") or []}
    invoked.update(call["alias"] for call in service.runtime.models.call_log)
    assert selected
    assert invoked <= selected or invoked == selected
    for attempt in result.get("model_attempts") or []:
        assert attempt["alias"] in selected


def test_zero_budget_has_zero_paid_calls(service: RunService) -> None:
    request = _request()
    run_id = "zero-budget"
    service.runtime.registry.create(request, run_id)
    result = service.execute(run_id, request)
    paid = [attempt for attempt in result.get("model_attempts") or [] if attempt.get("paid")]
    assert paid == []
    assert service.runtime.budget.paid_calls(run_id) == 0
    assert service.runtime.budget.snapshot(run_id)[0] == 0
    rows = service.runtime.telemetry.rows_for_run(run_id)
    assert rows
    assert result.get("report")


def test_parallel_both_dispatch_and_join() -> None:
    both = SubQuestion(
        id="sq-both",
        text="What retry and coverage limits keep research bounded?",
        source_hint=SourceHint.both,
        coverage_goal="at least one validated fact about coverage retries",
        route=RouteDecision(purpose="subquestion", selected_alias="fake-standard"),
    )
    state = {
        "run_id": "dispatch",
        "request": _request().model_dump(),
        "sub_questions": [both.model_dump()],
        "retry_sub_question_ids": [],
        "research_round": 1,
    }
    sends = dispatch_workers(state)
    assert isinstance(sends, list)
    nodes = sorted(send.node for send in sends)
    assert nodes == ["document_search_worker", "web_search_worker"]


def test_supervisor_skips_inactive_and_covered() -> None:
    inactive = SubQuestion(
        id="sq-off",
        text="Inactive question about routing budget rules",
        source_hint=SourceHint.web,
        coverage_goal="at least one validated fact about routing",
        active=False,
    )
    covered = SubQuestion(
        id="sq-done",
        text="Already covered evidence constraints for research assistants",
        source_hint=SourceHint.documents,
        coverage_goal="at least one validated fact about evidence",
        covered=True,
    )
    state = {
        "run_id": "skip",
        "request": _request().model_dump(),
        "sub_questions": [inactive.model_dump(), covered.model_dump()],
        "retry_sub_question_ids": [],
        "research_round": 1,
    }
    assert dispatch_workers(state) == "research_join"


def test_selective_second_round_skips_covered(tmp_path: Path) -> None:
    covered = SubQuestion(
        id="sq-covered",
        text="What evidence constraints apply to research assistants?",
        source_hint=SourceHint.documents,
        coverage_goal="at least one validated fact about evidence constraints",
        covered=False,
    )
    uncovered = SubQuestion(
        id="sq-open",
        text="What is the tax treatment of unicorn pizza franchises?",
        source_hint=SourceHint.web,
        coverage_goal="at least one validated fact about unicorn pizza tax",
        covered=False,
    )
    facts = [
        Fact(
            id="f1",
            claim="Evidence constraints apply to research assistants because quotes must match retrieved evidence",
            evidence_quote="validate every candidate fact against retrieved evidence",
            source_ref="doc:x",
            source_title="x",
            sub_question_id="sq-covered",
            validated=True,
        )
    ]
    assert is_covered(covered, facts)
    assert not is_covered(uncovered, facts)
    state = {
        "run_id": "retry",
        "request": _request().model_dump(),
        "sub_questions": [covered.model_dump(), uncovered.model_dump()],
        "facts": [facts[0].model_dump()],
        "research_round": 1,
        "retry_sub_question_ids": [],
        "warnings": [],
        "events": [],
        "limitations": [],
    }

    real = build_runtime(Settings(env="test", data_dir=tmp_path, enable_fake=True, web_mode="fixture"))
    real.registry.create(_request(), "retry")
    updated = coverage_gate(state | {"run_id": "retry"}, {"configurable": {"app_runtime": real}})
    assert updated["retry_sub_question_ids"] == ["sq-open"]
    retry_state = {
        **state,
        "run_id": "retry",
        "retry_sub_question_ids": updated["retry_sub_question_ids"],
        "sub_questions": updated["sub_questions"],
    }
    sends = dispatch_workers(retry_state)
    assert isinstance(sends, list)
    assert all(send.arg["sub_question"]["id"] == "sq-open" for send in sends)


def test_learning_failure_keeps_report(service: RunService, monkeypatch) -> None:
    def boom(*_args, **_kwargs):
        raise RuntimeError("learning write failed")

    monkeypatch.setattr(service.runtime.learning, "update", boom)
    run_id = "learning-fail"
    request = _request()
    service.runtime.registry.create(request, run_id)
    result = service.execute(run_id, request)
    assert result.get("report")
    assert result["report"]["markdown"]
    assert "learning_store_failed" in (result.get("warnings") or [])


def test_telemetry_comes_from_compiled_run(service: RunService) -> None:
    run_id = "telemetry-run"
    request = _request()
    service.runtime.registry.create(request, run_id)
    service.execute(run_id, request)
    rows = service.runtime.telemetry.rows_for_run(run_id)
    nodes = {row["node"] for row in rows}
    assert "validator" in nodes
    assert "final_qa" in nodes
    assert any(row["validation_decision"] for row in rows)
