"""Deterministic validator. Calls no LLM and does not decide budget."""

from __future__ import annotations

from typing import Any

from langgraph.types import RunnableConfig

from aar.graph.runtime import get_runtime
from aar.nodes.common import emit
from aar.schemas import Fact, SubQuestion, ValidationResult
from aar.textutil import content_words, normalize_text

UNKNOWN_SOURCE = "unknown_source"
MISSING_QUOTE = "missing_quote"
QUOTE_ABSENT = "quote_absent_from_source"
IRRELEVANT = "irrelevant_claim"
DUPLICATE = "duplicate_claim"


def validate_fact(
    fact: Fact,
    *,
    provenance_text: str | None,
    sub: SubQuestion | None,
    accepted_claims: set[str],
) -> tuple[Fact, str | None]:
    if provenance_text is None:
        return fact.model_copy(update={"validated": False, "rejection_reason": UNKNOWN_SOURCE}), UNKNOWN_SOURCE
    if not fact.evidence_quote.strip():
        return fact.model_copy(update={"validated": False, "rejection_reason": MISSING_QUOTE}), MISSING_QUOTE
    if normalize_text(fact.evidence_quote) not in normalize_text(provenance_text):
        return fact.model_copy(update={"validated": False, "rejection_reason": QUOTE_ABSENT}), QUOTE_ABSENT
    target = " ".join(
        [
            sub.text if sub else "",
            sub.coverage_goal if sub else "",
        ]
    )
    overlap = content_words(fact.claim) & content_words(target)
    if len(overlap) < 2:
        return fact.model_copy(update={"validated": False, "rejection_reason": IRRELEVANT}), IRRELEVANT
    normalized_claim = normalize_text(fact.claim)
    if normalized_claim in accepted_claims:
        return fact.model_copy(update={"validated": False, "rejection_reason": DUPLICATE}), DUPLICATE
    return fact.model_copy(update={"validated": True, "rejection_reason": None}), None


def validator(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    runtime = get_runtime(config)
    run_id = state["run_id"]
    subs = {item["id"]: SubQuestion.model_validate(item) for item in state.get("sub_questions") or []}
    provenance_by_ref = {item["ref"]: item.get("text", "") for item in state.get("provenance") or []}
    for record in runtime.provenance.list_for_run(run_id):
        provenance_by_ref.setdefault(record.ref, record.text)

    accepted_claims: set[str] = set()
    updated: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for raw in state.get("facts") or []:
        fact = Fact.model_validate(raw)
        updated_fact, reason = validate_fact(
            fact,
            provenance_text=provenance_by_ref.get(fact.source_ref),
            sub=subs.get(fact.sub_question_id),
            accepted_claims=accepted_claims,
        )
        if updated_fact.validated:
            accepted_claims.add(normalize_text(updated_fact.claim))
        updated.append(updated_fact.model_dump())
        result = ValidationResult(
            fact_id=fact.id,
            accepted=updated_fact.validated,
            reason=reason,
        )
        results.append(result.model_dump())
        events.append(
            emit(
                runtime,
                run_id,
                "fact_validated",
                "validator",
                sub_question_id=fact.sub_question_id,
                fact_id=fact.id,
                accepted=updated_fact.validated,
                reason=reason,
            )
        )
        runtime.telemetry.write(
            run_id=run_id,
            node="validator",
            sub_question_id=fact.sub_question_id,
            validation_decision="accepted" if updated_fact.validated else "rejected",
            extra={"reason": reason},
        )
    return {"facts": updated, "validation_results": results, "events": events}
