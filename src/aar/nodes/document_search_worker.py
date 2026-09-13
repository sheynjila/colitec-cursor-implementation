from __future__ import annotations

import hashlib
from typing import Any

from langgraph.types import RunnableConfig

from aar.graph.runtime import get_runtime
from aar.nodes.common import emit, request_from_state
from aar.nodes.summarizer import summarize_passage
from aar.schemas import Fact, ProvenanceRecord, SubQuestion


def document_search_worker(packet: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    runtime = get_runtime(config)
    request = request_from_state(packet)
    run_id = packet["run_id"]
    sub = SubQuestion.model_validate(packet["sub_question"])
    warnings: list[str] = []
    facts: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = [
        emit(
            runtime,
            run_id,
            "worker_started",
            "document_search_worker",
            sub_question_id=sub.id,
            worker="documents",
        )
    ]

    if not runtime.tool_uses.consume(run_id, sub.id):
        warnings.append(f"tool_limit:{sub.id}:document_search")
        return {"warnings": warnings, "events": events}

    rows = runtime.documents.search(f"{sub.text} {sub.coverage_goal}")
    runtime.telemetry.write(
        run_id=run_id,
        node="document_search_worker",
        sub_question_id=sub.id,
        tool_outcome="search_ok" if rows else "empty_index",
    )
    if not rows:
        warnings.append("empty_document_index" if runtime.documents.count() == 0 else f"uncovered_documents:{sub.id}")
        return {"warnings": warnings, "events": events}

    for row in rows[:3]:
        record = ProvenanceRecord(
            ref=row["citation"],
            title=row["title"],
            source_kind="document",
            text=row["text"],
            document_id=row["doc_id"],
            chunk_id=row["chunk_id"],
        )
        runtime.provenance.add(run_id, record)
        provenance.append(record.model_dump())
        quote = " ".join(row["text"].split())[:240]
        if not quote:
            warnings.append(f"absent_passage:{row['citation']}")
            continue
        summary = None
        if len(row["text"]) > 500 and runtime.tool_uses.remaining(run_id, sub.id) > 0:
            if runtime.tool_uses.consume(run_id, sub.id):
                summary = summarize_passage(
                    runtime,
                    run_id=run_id,
                    alias=packet.get("selected_alias"),
                    ceiling=request.max_cost_usd,
                    record=record,
                    sub_question_id=sub.id,
                )
        claim = summary or _claim_from(sub, row["text"])
        fact = Fact(
            id=_fact_id(run_id, sub.id, row["citation"], quote),
            claim=claim[:400],
            evidence_quote=quote,
            source_ref=row["citation"],
            source_title=row["title"],
            sub_question_id=sub.id,
        )
        facts.append(fact.model_dump())
        events.append(
            emit(
                runtime,
                run_id,
                "fact_candidate",
                "document_search_worker",
                sub_question_id=sub.id,
                fact_id=fact.id,
                source_ref=row["citation"],
            )
        )

    return {
        "facts": facts,
        "provenance": provenance,
        "warnings": warnings,
        "events": events,
    }


def _claim_from(sub: SubQuestion, passage: str) -> str:
    snippet = " ".join(passage.split())[:180]
    return f"{sub.text.rstrip('?')} is supported by local evidence: {snippet}"


def _fact_id(run_id: str, sub_id: str, ref: str, quote: str) -> str:
    digest = hashlib.sha256(f"{run_id}|{sub_id}|{ref}|{quote}".encode()).hexdigest()[:16]
    return f"fact-{digest}"
