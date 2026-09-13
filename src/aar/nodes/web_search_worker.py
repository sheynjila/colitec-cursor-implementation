from __future__ import annotations

import hashlib
from typing import Any

from langgraph.types import RunnableConfig

from aar.errors import AARError, BudgetRejected
from aar.graph.runtime import get_runtime
from aar.nodes.common import emit, request_from_state
from aar.nodes.summarizer import summarize_passage
from aar.schemas import Fact, ProvenanceRecord, SubQuestion
from aar.tools.ssrf import UrlBlocked


def web_search_worker(packet: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    runtime = get_runtime(config)
    request = request_from_state(packet)
    run_id = packet["run_id"]
    sub = SubQuestion.model_validate(packet["sub_question"])
    warnings: list[str] = []
    facts: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

    events.append(
        emit(
            runtime,
            run_id,
            "worker_started",
            "web_search_worker",
            sub_question_id=sub.id,
            worker="web",
        )
    )

    if not runtime.tool_uses.consume(run_id, sub.id):
        warnings.append(f"tool_limit:{sub.id}:web_search")
        return {"warnings": warnings, "events": events}

    reservation: str | None = None
    try:
        if runtime.settings.tavily_cost_usd and runtime.settings.web_mode != "fixture":
            try:
                reservation = runtime.budget.reserve(
                    run_id, request.max_cost_usd, runtime.settings.tavily_cost_usd
                )
            except BudgetRejected:
                warnings.append("web_search_budget_rejected")
                return {"warnings": warnings, "events": events}
        hits = runtime.web.search(sub.text)
        if reservation:
            runtime.budget.settle(reservation, runtime.settings.tavily_cost_usd or 0.0)
            reservation = None
        runtime.telemetry.write(
            run_id=run_id,
            node="web_search_worker",
            sub_question_id=sub.id,
            tool_outcome="search_ok",
        )
    except AARError as exc:
        if reservation:
            runtime.budget.release(reservation)
        warnings.append(f"{exc.code}:{exc.message}")
        runtime.telemetry.write(
            run_id=run_id,
            node="web_search_worker",
            sub_question_id=sub.id,
            tool_outcome="search_failed",
            error_code=exc.code,
        )
        return {"warnings": warnings, "events": events}

    for hit in hits[:2]:
        if not runtime.tool_uses.consume(run_id, sub.id):
            warnings.append(f"tool_limit:{sub.id}:web_fetch")
            break
        try:
            final_url, text, labeled = runtime.web.fetch(hit.url)
        except (UrlBlocked, AARError, Exception) as exc:  # noqa: BLE001
            warnings.append(f"web_fetch_failed:{hit.url}")
            runtime.telemetry.write(
                run_id=run_id,
                node="web_search_worker",
                sub_question_id=sub.id,
                tool_outcome="fetch_blocked",
                error_code=getattr(exc, "code", "web_fetch_failed"),
            )
            continue
        if not text.strip():
            warnings.append(f"empty_page:{final_url}")
            continue
        record = ProvenanceRecord(
            ref=final_url,
            title=hit.title,
            source_kind="web",
            text=text,
            retrieved_url=final_url,
            labeled_fixture=labeled or hit.labeled_fixture,
        )
        runtime.provenance.add(run_id, record)
        provenance.append(record.model_dump())
        quote = _first_sentence(text)
        if not quote:
            continue
        summary = None
        if len(text) > 500 and runtime.tool_uses.remaining(run_id, sub.id) > 0:
            if runtime.tool_uses.consume(run_id, sub.id):
                summary = summarize_passage(
                    runtime,
                    run_id=run_id,
                    alias=packet.get("selected_alias"),
                    ceiling=request.max_cost_usd,
                    record=record,
                    sub_question_id=sub.id,
                )
        claim = summary or f"{sub.text.rstrip('?')} is addressed by retrieved evidence."
        fact = Fact(
            id=_fact_id(run_id, sub.id, final_url, quote),
            claim=claim[:400],
            evidence_quote=quote,
            source_ref=final_url,
            source_title=hit.title,
            sub_question_id=sub.id,
        )
        facts.append(fact.model_dump())
        events.append(
            emit(
                runtime,
                run_id,
                "fact_candidate",
                "web_search_worker",
                sub_question_id=sub.id,
                fact_id=fact.id,
                source_ref=final_url,
            )
        )

    if not facts:
        warnings.append(f"uncovered_web:{sub.id}")
    return {
        "facts": facts,
        "provenance": provenance,
        "warnings": warnings,
        "events": events,
    }


def _first_sentence(text: str) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    for separator in ".!?":
        index = cleaned.find(separator)
        if 40 <= index <= 280:
            return cleaned[: index + 1]
    return cleaned[:240]


def _fact_id(run_id: str, sub_id: str, ref: str, quote: str) -> str:
    digest = hashlib.sha256(f"{run_id}|{sub_id}|{ref}|{quote}".encode()).hexdigest()[:16]
    return f"fact-{digest}"
