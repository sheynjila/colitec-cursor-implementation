from __future__ import annotations

from aar.nodes.final_qa import run_qa
from aar.nodes.validator import (
    DUPLICATE,
    IRRELEVANT,
    MISSING_QUOTE,
    QUOTE_ABSENT,
    UNKNOWN_SOURCE,
    validate_fact,
)
from aar.nodes.writer import template_report
from aar.schemas import Fact, SourceHint, SubQuestion


def _sub() -> SubQuestion:
    return SubQuestion(
        id="sq-1",
        text="What evidence constraints apply to research assistants?",
        source_hint=SourceHint.documents,
        coverage_goal="at least one validated fact about evidence constraints",
    )


def _fact(**overrides: object) -> Fact:
    payload = {
        "id": "fact-1",
        "claim": "Evidence constraints require retrieved quotes for research assistants",
        "evidence_quote": "validate every candidate fact against retrieved evidence",
        "source_ref": "doc:evidence_constraints#chunk:c001",
        "source_title": "Evidence constraints",
        "sub_question_id": "sq-1",
    }
    payload.update(overrides)
    return Fact.model_validate(payload)


def test_validation_result_schema_reasons() -> None:
    from aar.schemas import ValidationResult

    result = ValidationResult(fact_id="fact-1", accepted=False, reason="unknown_source")
    assert result.reason == "unknown_source"


def test_unknown_source_and_missing_quote() -> None:
    unknown, reason = validate_fact(_fact(), provenance_text=None, sub=_sub(), accepted_claims=set())
    assert reason == UNKNOWN_SOURCE
    assert not unknown.validated

    missing, reason = validate_fact(
        _fact(evidence_quote=""),
        provenance_text="validate every candidate fact against retrieved evidence",
        sub=_sub(),
        accepted_claims=set(),
    )
    assert reason == MISSING_QUOTE
    assert not missing.validated


def test_quote_absent_irrelevant_and_duplicate() -> None:
    absent, reason = validate_fact(
        _fact(),
        provenance_text="this page never mentions the required quote",
        sub=_sub(),
        accepted_claims=set(),
    )
    assert reason == QUOTE_ABSENT

    irrelevant, reason = validate_fact(
        _fact(claim="Unrelated astronomy claim about distant galaxies"),
        provenance_text="validate every candidate fact against retrieved evidence",
        sub=_sub(),
        accepted_claims=set(),
    )
    assert reason == IRRELEVANT

    first, reason = validate_fact(
        _fact(),
        provenance_text="Assistants must validate every candidate fact against retrieved evidence.",
        sub=_sub(),
        accepted_claims=set(),
    )
    assert reason is None and first.validated

    duplicate, reason = validate_fact(
        _fact(id="fact-2"),
        provenance_text="Assistants must validate every candidate fact against retrieved evidence.",
        sub=_sub(),
        accepted_claims={first.claim.casefold()},
    )
    assert reason == DUPLICATE


def test_approved_report_has_no_invented_citation() -> None:
    fact = _fact(validated=True)
    sub = _sub()
    report = template_report(
        "What evidence constraints apply to research assistants?",
        [sub],
        [fact],
        [],
        [],
    )
    # rebuild with sources from writer helper by using sources list
    from aar.nodes.writer import _sources

    sources = _sources([fact])
    report = template_report(
        "What evidence constraints apply to research assistants?",
        [sub],
        [fact],
        sources,
        ["insufficient_evidence_for_some_subquestions"],
    )
    warnings = run_qa(
        report,
        question="What evidence constraints apply to research assistants?",
        subs=[sub],
        facts=[fact],
        limitations=["insufficient_evidence_for_some_subquestions"],
    )
    assert "unknown_link" not in warnings
    assert all(item.ref == fact.source_ref for item in report.sources)
    assert "https://invented.example" not in report.markdown
