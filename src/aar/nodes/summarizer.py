"""Shared summarizer utility. Not a graph node and never dispatched independently."""

from __future__ import annotations

from aar.graph.runtime import AppRuntime
from aar.nodes.common import generate_paid
from aar.schemas import ProvenanceRecord


def summarize_passage(
    runtime: AppRuntime,
    *,
    run_id: str,
    alias: str | None,
    ceiling: float,
    record: ProvenanceRecord,
    sub_question_id: str,
) -> str:
    """Condense a passage while keeping the source identity and an exact quote."""
    quote = record.text[:240].strip()
    if not alias or len(record.text) < 500:
        return f"Source {record.ref} ({record.title}) exact quote: \"{quote}\""
    prompt = (
        "Summarize the TEXT while retaining the source identity and one exact quotation.\n"
        f"SOURCE: {record.ref}\nTITLE: {record.title}\nTEXT:\n{record.text[:3000]}"
    )
    result = generate_paid(
        runtime,
        run_id=run_id,
        alias=alias,
        prompt=prompt,
        node="summarizer",
        purpose="summarize",
        ceiling=ceiling,
        sub_question_id=sub_question_id,
    )
    if result is None:
        return f"Source {record.ref} ({record.title}) exact quote: \"{quote}\""
    summary = result.text.strip()
    if record.ref not in summary:
        summary = f"{record.ref}: {summary}"
    if quote[:40] not in summary:
        summary = f"{summary}\nExact quote: \"{quote}\""
    return summary
