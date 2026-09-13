from __future__ import annotations

from typing import Any

from langgraph.types import RunnableConfig

from aar.graph.runtime import get_runtime
from aar.graph.state import merge_by_id, merge_provenance
from aar.nodes.common import emit
from aar.textutil import normalize_text


def research_join(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
    runtime = get_runtime(config)
    run_id = state["run_id"]
    facts = merge_by_id([], state.get("facts") or [])
    provenance = merge_provenance([], state.get("provenance") or [])
    seen_quotes: set[tuple[str, str]] = set()
    unique_facts: list[dict[str, Any]] = []
    for fact in facts:
        key = (normalize_text(fact.get("source_ref", "")), normalize_text(fact.get("evidence_quote", "")))
        if key in seen_quotes:
            continue
        seen_quotes.add(key)
        unique_facts.append(fact)
    event = emit(
        runtime,
        run_id,
        "fact_candidate",
        "research_join",
        merged_facts=len(unique_facts),
        provenance=len(provenance),
    )
    return {"facts": unique_facts, "provenance": provenance, "events": [event]}
