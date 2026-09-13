from __future__ import annotations

from typing import Any

from aar.errors import BudgetRejected, ProviderError, safe_error_code
from aar.graph.runtime import AppRuntime
from aar.llm.catalog import get_spec
from aar.llm.client import GenerationResult, estimate_tokens
from aar.llm.routing import projected_call_cost
from aar.schemas import ModelAttempt, ProgressEvent, ResearchRequest
from aar.storage.telemetry import redact


def request_from_state(state: dict[str, Any]) -> ResearchRequest:
    return ResearchRequest.model_validate(state["request"])


def emit(runtime: AppRuntime, run_id: str, event_type: str, node: str, **payload: Any) -> dict[str, Any]:
    event = runtime.events.append(
        ProgressEvent(
            run_id=run_id,
            event_type=event_type,
            node=node,
            sub_question_id=payload.pop("sub_question_id", None),
            payload=redact(payload),
        )
    )
    return event.model_dump(mode="json")


def generate_paid(
    runtime: AppRuntime,
    *,
    run_id: str,
    alias: str,
    prompt: str,
    node: str,
    purpose: str,
    ceiling: float,
    sub_question_id: str | None = None,
    system: str | None = None,
) -> GenerationResult | None:
    spec = get_spec(alias)
    if spec is None:
        return None
    worst = projected_call_cost(
        spec, estimate_tokens((system or "") + prompt), runtime.settings.max_output_tokens
    )
    try:
        reservation = runtime.budget.reserve(run_id, ceiling, worst, paid=spec.paid)
    except BudgetRejected:
        runtime.telemetry.write(
            run_id=run_id,
            node=node,
            sub_question_id=sub_question_id,
            alias=alias,
            provider=spec.provider,
            tier=spec.tier.value,
            error_code="budget_rejected",
        )
        return None
    try:
        result = runtime.models.generate(
            alias,
            prompt,
            max_output_tokens=runtime.settings.max_output_tokens,
            system=system,
        )
        runtime.budget.settle(reservation, result.actual_cost_usd if spec.paid else 0.0)
        runtime.telemetry.write(
            run_id=run_id,
            node=node,
            sub_question_id=sub_question_id,
            alias=result.alias,
            provider=result.provider,
            tier=spec.tier.value,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            estimated_cost=result.estimated_cost_usd,
            latency_ms=result.latency_ms,
            extra={"premium": spec.premium, "usage_delta": result.usage_delta},
        )
        return result
    except ProviderError as exc:
        runtime.budget.release(reservation)
        runtime.telemetry.write(
            run_id=run_id,
            node=node,
            sub_question_id=sub_question_id,
            alias=alias,
            provider=spec.provider,
            tier=spec.tier.value,
            error_code=safe_error_code(exc),
        )
        return None


def attempt_from_result(
    result: GenerationResult,
    *,
    purpose: str,
    node: str,
    sub_question_id: str | None = None,
) -> dict[str, Any]:
    spec = get_spec(result.alias)
    return ModelAttempt(
        alias=result.alias,
        provider=result.provider,
        purpose=purpose,
        sub_question_id=sub_question_id,
        node=node,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        estimated_cost_usd=result.estimated_cost_usd,
        actual_cost_usd=result.actual_cost_usd,
        latency_ms=result.latency_ms,
        finish_reason=result.finish_reason,
        paid=bool(spec.paid) if spec else False,
    ).model_dump()
