from __future__ import annotations

import re
from typing import Any

from langgraph.types import RunnableConfig

from aar.graph.runtime import get_runtime
from aar.nodes.common import emit, request_from_state
from aar.nodes.writer import template_report
from aar.schemas import Fact, QAStatus, ResearchReport, SourceItem, SubQuestion


def run_qa(
    report: ResearchReport,
    *,
    question: str,
    subs: list[SubQuestion],
    facts: list[Fact],
    limitations: list[str],
) -> list[str]:
    warnings: list[str] = []
    markdown = report.markdown
    if "## Executive Summary" not in markdown:
        warnings.append("missing_executive_summary")
    for sub in subs:
        if f"## {sub.text}" not in markdown:
            warnings.append(f"missing_section:{sub.id}")
    if "## Sources" not in markdown:
        warnings.append("missing_sources")

    cited = {int(num) for num in re.findall(r"\[(\d+)\]", markdown)}
    source_numbers = {item.number for item in report.sources}
    if cited - source_numbers:
        warnings.append("citation_without_source")
    if source_numbers - cited and facts:
        warnings.append("source_not_cited")
    if len(report.sources) != len({item.ref for item in report.sources}):
        warnings.append("duplicate_sources")

    approved_refs = {fact.source_ref for fact in facts}
    for item in report.sources:
        if item.ref not in approved_refs:
            warnings.append("unknown_link")
            break

    for fact in facts:
        if not fact.validated and fact.claim and fact.claim in markdown:
            warnings.append("rejected_claim_leakage")
            break

    if limitations and "## Limitations" not in markdown and "insufficient" not in markdown.lower():
        warnings.append("missing_limitations")
    if not facts and "insufficient" not in markdown.lower():
        warnings.append("missing_insufficient_disclosure")
    return warnings


def final_qa(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    runtime = get_runtime(config)
    request = request_from_state(state)
    run_id = state["run_id"]
    facts = [Fact.model_validate(item) for item in state.get("facts") or [] if item.get("validated")]
    rejected = [Fact.model_validate(item) for item in state.get("facts") or [] if not item.get("validated")]
    subs = [SubQuestion.model_validate(item) for item in state.get("sub_questions") or []]
    limitations = list(state.get("limitations") or [])
    report = ResearchReport.model_validate(state["report"]) if state.get("report") else template_report(
        request.question, subs, facts, _sources(facts), limitations
    )
    warnings = run_qa(report, question=request.question, subs=subs, facts=facts, limitations=limitations)
    for fact in rejected:
        if fact.claim and fact.claim in report.markdown:
            warnings.append("rejected_claim_leakage")
    fallback = bool(state.get("writer_fallback"))
    if warnings:
        report = template_report(request.question, subs, facts, _sources(facts), limitations)
        fallback = True
        warnings = run_qa(report, question=request.question, subs=subs, facts=facts, limitations=limitations)
        status = QAStatus.approved if not warnings else QAStatus.not_approved
        report.qa_status = status
        report.qa_warnings = warnings
        report.writer_fallback = True
    else:
        report.qa_status = QAStatus.approved
        report.qa_warnings = []
        report.writer_fallback = fallback

    event = emit(
        runtime,
        run_id,
        "qa_completed",
        "final_qa",
        status=report.qa_status.value,
        warnings=report.qa_warnings,
    )
    runtime.telemetry.write(
        run_id=run_id,
        node="final_qa",
        qa_result=report.qa_status.value,
        extra={"warnings": report.qa_warnings},
    )
    return {
        "report": report.model_dump(),
        "qa": {"status": report.qa_status.value, "warnings": report.qa_warnings},
        "writer_fallback": report.writer_fallback,
        "events": [event],
    }


def _sources(facts: list[Fact]) -> list[SourceItem]:
    items: list[SourceItem] = []
    seen: set[str] = set()
    for fact in facts:
        if fact.source_ref in seen:
            continue
        seen.add(fact.source_ref)
        items.append(SourceItem(number=len(items) + 1, ref=fact.source_ref, title=fact.source_title))
    return items
