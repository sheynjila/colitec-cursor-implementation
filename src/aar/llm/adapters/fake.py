from __future__ import annotations

import json
from collections.abc import Callable

from aar.errors import ProviderError
from aar.llm.catalog import ModelSpec
from aar.llm.client import GenerationResult, estimate_tokens


class FakeAdapter:
    provider = "fake"

    def __init__(
        self,
        responder: Callable[[str, str], str] | None = None,
        fail_aliases: set[str] | None = None,
        extra_output_tokens: int = 0,
    ) -> None:
        self.responder = responder or default_fake_responder
        self.fail_aliases = fail_aliases or set()
        self.calls: list[dict[str, str]] = []
        self.extra_output_tokens = extra_output_tokens

    def generate(
        self,
        spec: ModelSpec,
        prompt: str,
        *,
        max_output_tokens: int,
        system: str | None = None,
    ) -> GenerationResult:
        self.calls.append({"alias": spec.alias, "prompt": prompt[:80]})
        if spec.alias in self.fail_aliases:
            raise ProviderError(f"Forced failure for {spec.alias}")
        text = self.responder(spec.alias, prompt)
        output_tokens = estimate_tokens(text) + self.extra_output_tokens
        output_tokens = min(output_tokens, max_output_tokens)
        return GenerationResult(
            text=text,
            alias=spec.alias,
            provider="fake",
            input_tokens=estimate_tokens((system or "") + prompt),
            output_tokens=output_tokens,
            latency_ms=1.0,
            finish_reason="stop",
            estimated_cost_usd=0.0,
            actual_cost_usd=spec.projected_cost(
                estimate_tokens((system or "") + prompt), output_tokens
            ),
        )


def default_fake_responder(alias: str, prompt: str) -> str:
    lower = prompt.lower()
    if "sub-questions" in lower or "subquestions" in lower or "plan the research" in lower:
        return json.dumps(
            {
                "sub_questions": [
                    {
                        "text": "What constraints define evidence-grounded research assistants?",
                        "source_hint": "documents",
                        "coverage_goal": "at least one validated fact about evidence constraints",
                    },
                    {
                        "text": "How should model routing and cost limits be enforced?",
                        "source_hint": "documents",
                        "coverage_goal": "at least one validated fact about routing or budget",
                    },
                    {
                        "text": "What retry and coverage limits keep research bounded?",
                        "source_hint": "both",
                        "coverage_goal": "at least one validated fact about coverage retries",
                    },
                ]
            }
        )
    if "write a cited research report" in lower or "executive summary" in lower:
        return (
            "## Executive Summary\n"
            "The assistant must ground claims in retrieved evidence, route models by score, "
            "and stop when coverage or budget limits are reached.\n\n"
            "## Evidence constraints\n"
            "Validated facts require an exact quote from a retrieved source [1].\n\n"
            "## Sources\n"
            "[1] fixture\n"
        )
    if "summarize" in lower:
        return prompt.split("TEXT:", 1)[-1][:400]
    return f"{alias} completed the request without adding unsupported claims."
