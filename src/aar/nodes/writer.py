from __future__ import annotations

import re
from typing import Any

from langgraph.types import RunnableConfig

from aar.graph.runtime import get_runtime
from aar.llm.routing import RoutingContext, route_writing
from aar.nodes.common import attempt_from_result, emit, generate_paid, request_from_state
from aar.schemas import Fact, QAStatus, ResearchReport, SourceItem, SubQuestion


def writer(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    runtime = get_runtime(config)
    request = request_from_state(state)
    run_id = state["run_id"]
    facts = [
        Fact.model_validate(item)
        for item in state.get("facts") or []
        if item.get("validated")
    ]
    subs = [SubQuestion.model_validate(item) for item in state.get("sub_questions") or []]
    limitations = list(state.get("limitations") or [])
    sources = _sources(facts)
    remaining = runtime.budget.remaining(run_id, request.max_cost_usd)
    ctx = RoutingContext(
        settings=runtime.settings,
        learning=runtime.learning,
        remaining_budget=remaining,
    )
    route = route_writing(request, ctx, len(facts))
    warnings: list[str] = []
    attempts: list[dict[str, Any]] = []
    fallback = True
    report = template_report(request.question, subs, facts, sources, limitations)

    if route.selected_alias and facts:
        prompt = _writer_prompt(request.question, subs, facts, sources, limitations)
        result = generate_paid(
            runtime,
            run_id=run_id,
            alias=route.selected_alias,
            prompt=prompt,
            node="writer",
            purpose="writing",
            ceiling=request.max_cost_usd,
        )
        if result and result.alias == route.selected_alias:
            attempts.append(attempt_from_result(result, purpose="writing", node="writer"))
            draft = _parse_llm_report(result.text, subs, facts, sources, limitations)
            if draft is not None:
                report = draft
                fallback = False
            else:
                warnings.append("writer_parse_failed")
        else:
            warnings.append("writer_model_failed")
    elif route.selected_alias is None:
        warnings.append("writer_route_unavailable")

    if not facts:
        limitations = list(dict.fromkeys(limitations + ["insufficient_evidence"]))
        report = template_report(request.question, subs, facts, sources, limitations)
        fallback = True

    report.writer_fallback = fallback
    event = emit(
        runtime,
        run_id,
        "report_drafted",
        "writer",
        fallback=fallback,
        alias=route.selected_alias,
        fact_count=len(facts),
    )
    return {
        "report": report.model_dump(),
        "writer_fallback": fallback,
        "limitations": limitations,
        "warnings": warnings,
        "events": [event],
        "model_attempts": attempts,
        "routes": list(state.get("routes") or []) + [route.model_dump()],
    }


def template_report(
    question: str,
    subs: list[SubQuestion],
    facts: list[Fact],
    sources: list[SourceItem],
    limitations: list[str],
) -> ResearchReport:
    source_by_ref = {item.ref: item.number for item in sources}
    if facts:
        summary = (
            f"This report answers: {question} "
            f"It uses {len(facts)} validated fact(s) from {len(sources)} approved source(s). "
            "Claims without retrieved evidence are omitted."
        )
    else:
        summary = (
            f"There is insufficient retrieved evidence to answer: {question} "
            "No validated facts were accepted, so this report does not invent claims or citations. "
            "Treat the result as a coverage failure, not a completed literature review."
        )
    sections: list[dict[str, Any]] = []
    lines = ["## Executive Summary", summary, ""]
    for sub in subs:
        related = [fact for fact in facts if fact.sub_question_id == sub.id]
        paragraphs: list[str] = []
        if related:
            for fact in related:
                number = source_by_ref.get(fact.source_ref)
                citation = f" [{number}]" if number else ""
                paragraphs.append(f"{fact.claim}{citation}")
        else:
            paragraphs.append("No validated evidence was accepted for this sub-question.")
        body = " ".join(paragraphs)
        sections.append({"sub_question_id": sub.id, "title": sub.text, "body": body})
        lines.extend([f"## {sub.text}", body, ""])
    lines.append("## Sources")
    if sources:
        for item in sources:
            lines.append(f"[{item.number}] {item.title} — {item.ref}")
    else:
        lines.append("No approved sources.")
    if limitations:
        lines.extend(["", "## Limitations", "; ".join(limitations)])
    return ResearchReport(
        executive_summary=summary,
        sections=sections,
        markdown="\n".join(lines).strip(),
        sources=sources,
        limitations=limitations,
        qa_status=QAStatus.pending,
        writer_fallback=True,
    )


def _sources(facts: list[Fact]) -> list[SourceItem]:
    items: list[SourceItem] = []
    seen: set[str] = set()
    for fact in facts:
        if fact.source_ref in seen:
            continue
        seen.add(fact.source_ref)
        items.append(SourceItem(number=len(items) + 1, ref=fact.source_ref, title=fact.source_title))
    return items


def _writer_prompt(
    question: str,
    subs: list[SubQuestion],
    facts: list[Fact],
    sources: list[SourceItem],
    limitations: list[str],
) -> str:
    fact_lines = []
    source_by_ref = {item.ref: item.number for item in sources}
    for fact in facts:
        fact_lines.append(
            f"- [{source_by_ref.get(fact.source_ref)}] {fact.claim} quote={fact.evidence_quote!r} ref={fact.source_ref}"
        )
    return (
        "Write a cited research report. Use only the validated facts and approved sources. "
        "Include a two- or three-sentence executive summary, one section per sub-question, "
        "inline numbered citations, a Sources list, and limitations.\n"
        f"Question: {question}\n"
        f"Sub-questions: {[sub.text for sub in subs]}\n"
        f"Facts:\n{chr(10).join(fact_lines)}\n"
        f"Sources: {[item.model_dump() for item in sources]}\n"
        f"Limitations: {limitations}\n"
    )


def _parse_llm_report(
    text: str,
    subs: list[SubQuestion],
    facts: list[Fact],
    sources: list[SourceItem],
    limitations: list[str],
) -> ResearchReport | None:
    if "## Executive Summary" not in text or "## Sources" not in text:
        return None
    summary_match = re.search(
        r"## Executive Summary\s+(.+?)(?:\n## |\Z)", text, flags=re.S
    )
    if not summary_match:
        return None
    summary = summary_match.group(1).strip()
    sentences = [part for part in re.split(r"(?<=[.!?])\s+", summary) if part.strip()]
    if not 2 <= len(sentences) <= 4:
        return None
    sections = []
    for sub in subs:
        if sub.text not in text:
            return None
        sections.append({"sub_question_id": sub.id, "title": sub.text, "body": ""})
    return ResearchReport(
        executive_summary=summary,
        sections=sections,
        markdown=text.strip(),
        sources=sources,
        limitations=limitations,
        qa_status=QAStatus.pending,
        writer_fallback=False,
    )
