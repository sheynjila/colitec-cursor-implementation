from __future__ import annotations

from aar.config import Settings
from aar.llm.catalog import get_spec
from aar.llm.routing import (
    REJECTION_BUDGET,
    REJECTION_CAPABILITY,
    REJECTION_CONTEXT,
    REJECTION_DISABLED,
    REJECTION_PREMIUM,
    REJECTION_PROVIDER,
    REJECTION_TIER,
    REJECTION_UNAVAILABLE,
    RoutingContext,
    reject_reason,
    route_one,
    select_alias,
)
from aar.llm.scoring import score_sub_question
from aar.schemas import ModelTier, ResearchRequest, SourceHint, SubQuestion


def _request(**overrides: object) -> ResearchRequest:
    payload = {
        "question": "What constraints should a research assistant enforce?",
        "allowed_sources": "documents",
        "desired_depth": "brief",
        "risk_level": "low",
        "max_cost_usd": 1.0,
    }
    payload.update(overrides)
    return ResearchRequest.model_validate(payload)


def _sub(text: str, hint: SourceHint = SourceHint.web) -> SubQuestion:
    return SubQuestion(id="sq-1", text=text, source_hint=hint, coverage_goal="coverage goal")


def test_scoring_boundaries_4_5_10_11() -> None:
    cheap = score_sub_question(_sub("Define topic scope now"), _request())
    assert cheap.total == 4
    assert cheap.tier == ModelTier.cheap

    standard_low = score_sub_question(_sub("Define latest topic scope now"), _request())
    assert standard_low.total == 5
    assert standard_low.tier == ModelTier.standard

    standard_high = score_sub_question(
        _sub("Define topic scope now"),
        _request(desired_depth="deep", risk_level="medium"),
    )
    assert standard_high.total == 10
    assert standard_high.tier == ModelTier.standard

    strong = score_sub_question(
        _sub("Define latest topic scope now"),
        _request(desired_depth="deep", risk_level="medium"),
    )
    assert strong.total == 11
    assert strong.tier == ModelTier.strong


def _ctx(**overrides: object) -> RoutingContext:
    settings = Settings(
        enable_fake=True,
        allow_premium=False,
        max_output_tokens=256,
        gemini_api_key="",
        anthropic_api_key="",
        openrouter_api_key="",
        enable_bedrock=False,
    )
    values = {
        "settings": settings,
        "learning": None,
        "remaining_budget": 1.0,
        "estimated_input_tokens": 800,
    }
    values.update(overrides)
    return RoutingContext(**values)  # type: ignore[arg-type]


def test_each_alias_rejection_reason() -> None:
    disabled = reject_reason(get_spec("fake-cheap"), _ctx(settings=Settings(enable_fake=False)), ModelTier.cheap)
    assert disabled == REJECTION_DISABLED

    provider = reject_reason(get_spec("gemini-flash-lite"), _ctx(), ModelTier.cheap)
    assert provider == REJECTION_PROVIDER

    capability = reject_reason(
        get_spec("fake-cheap"),
        _ctx(required_capability="tools"),
        ModelTier.cheap,
    )
    assert capability == REJECTION_CAPABILITY

    context = reject_reason(
        get_spec("fake-cheap"),
        _ctx(estimated_input_tokens=10_000_000),
        ModelTier.cheap,
    )
    assert context == REJECTION_CONTEXT

    tier = reject_reason(get_spec("fake-cheap"), _ctx(), ModelTier.strong)
    assert tier == REJECTION_TIER

    premium = reject_reason(get_spec("fake-premium"), _ctx(remaining_budget=100), ModelTier.strong)
    assert premium == REJECTION_PREMIUM

    budget = reject_reason(get_spec("fake-premium"), _ctx(allow_premium=True, remaining_budget=0), ModelTier.strong)
    assert budget == REJECTION_BUDGET

    spec, reasons = select_alias(ModelTier.strong, _ctx(settings=Settings(enable_fake=False), remaining_budget=0))
    assert spec is None
    assert REJECTION_UNAVAILABLE in reasons


def test_learning_is_only_a_same_tier_tie_break(runtime) -> None:
    runtime.learning.update("fake-cheap", passed=True, cost_usd=0.0, latency_ms=1.0)
    runtime.learning.update("ollama-llama3.2", passed=False, cost_usd=0.0, latency_ms=900.0)
    ctx = RoutingContext(
        settings=runtime.settings,
        learning=runtime.learning,
        remaining_budget=1.0,
        estimated_input_tokens=200,
    )
    chosen, _ = select_alias(ModelTier.cheap, ctx)
    assert chosen is not None
    assert chosen.alias == "fake-cheap"


def test_route_one_records_score_and_does_not_change_tier() -> None:
    sub = _sub("Define latest topic scope now")
    decision = route_one(sub, _request(), _ctx())
    assert decision.required_tier == ModelTier.standard
    assert decision.selected_alias == "fake-standard"
    assert decision.score is not None
    assert decision.score.tier == ModelTier.standard
