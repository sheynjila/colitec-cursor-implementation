"""Alias selection. Filters are applied in a fixed order; tiers never change silently."""

from __future__ import annotations

from dataclasses import dataclass

from aar.config import Settings
from aar.llm.catalog import CATALOG, ModelSpec, enabled_for, get_spec
from aar.llm.client import estimate_tokens
from aar.llm.scoring import score_sub_question, writing_score
from aar.schemas import (
    ModelTier,
    ResearchRequest,
    RouteDecision,
    ScoreBreakdown,
    SubQuestion,
)
from aar.storage.learning import LearningStore

REJECTION_DISABLED = "disabled"
REJECTION_PROVIDER = "provider_unavailable"
REJECTION_CAPABILITY = "capability_mismatch"
REJECTION_CONTEXT = "context_too_small"
REJECTION_TIER = "tier_mismatch"
REJECTION_PREMIUM = "premium_blocked"
REJECTION_BUDGET = "over_budget"
REJECTION_UNAVAILABLE = "route_unavailable"

REQUIRED_CAPABILITY = "chat"
DEFAULT_INPUT_TOKENS = 1200


@dataclass
class RoutingContext:
    settings: Settings
    learning: LearningStore | None
    remaining_budget: float
    estimated_input_tokens: int = DEFAULT_INPUT_TOKENS
    disabled_aliases: set[str] | None = None
    required_capability: str = REQUIRED_CAPABILITY
    allow_premium: bool | None = None


def projected_call_cost(spec: ModelSpec, input_tokens: int, max_output_tokens: int) -> float:
    return spec.projected_cost(input_tokens, max_output_tokens)


def reject_reason(spec: ModelSpec, ctx: RoutingContext, required_tier: ModelTier) -> str | None:
    if not enabled_for(spec, ctx.settings, ctx.disabled_aliases):
        if spec.provider == "fake" and not ctx.settings.enable_fake:
            return REJECTION_DISABLED
        if spec.provider != "fake" and not enabled_for(spec, ctx.settings, None):
            return REJECTION_PROVIDER
        return REJECTION_DISABLED
    if ctx.required_capability not in spec.capabilities:
        return REJECTION_CAPABILITY
    if spec.context_window < ctx.estimated_input_tokens:
        return REJECTION_CONTEXT
    if spec.tier != required_tier:
        return REJECTION_TIER
    allow_premium = ctx.settings.allow_premium if ctx.allow_premium is None else ctx.allow_premium
    if spec.premium and not allow_premium:
        return REJECTION_PREMIUM
    worst_case = projected_call_cost(
        spec, ctx.estimated_input_tokens, ctx.settings.max_output_tokens
    )
    if spec.paid and worst_case > ctx.remaining_budget + 1e-12:
        return REJECTION_BUDGET
    return None


def _preference(alias: str, learning: LearningStore | None) -> float:
    if learning is None:
        return 0.0
    return float(learning.get(alias)["preference_score"])


def select_alias(required_tier: ModelTier, ctx: RoutingContext) -> tuple[ModelSpec | None, list[str]]:
    reasons: list[str] = []
    eligible: list[tuple[float, float, ModelSpec]] = []
    for spec in CATALOG:
        reason = reject_reason(spec, ctx, required_tier)
        if reason:
            reasons.append(f"{spec.alias}:{reason}")
            continue
        cost = projected_call_cost(spec, ctx.estimated_input_tokens, ctx.settings.max_output_tokens)
        eligible.append((cost, -_preference(spec.alias, ctx.learning), spec))
    if not eligible:
        return None, reasons + [REJECTION_UNAVAILABLE]
    eligible.sort(key=lambda item: (item[0], item[1], item[2].alias))
    return eligible[0][2], reasons


def _decision(
    purpose: str,
    required_tier: ModelTier,
    ctx: RoutingContext,
    score: ScoreBreakdown | None,
    sub_question_id: str | None,
) -> RouteDecision:
    spec, reasons = select_alias(required_tier, ctx)
    if spec is None:
        return RouteDecision(
            purpose=purpose,  # type: ignore[arg-type]
            sub_question_id=sub_question_id,
            score=score,
            required_tier=required_tier,
            selected_alias=None,
            selected_provider=None,
            projected_cost_usd=0.0,
            rejection_reasons=reasons,
            fallback="deterministic_template",
        )
    return RouteDecision(
        purpose=purpose,  # type: ignore[arg-type]
        sub_question_id=sub_question_id,
        score=score,
        required_tier=required_tier,
        selected_alias=spec.alias,
        selected_provider=spec.provider,
        projected_cost_usd=projected_call_cost(
            spec, ctx.estimated_input_tokens, ctx.settings.max_output_tokens
        ),
        rejection_reasons=reasons,
        fallback=None,
    )


def route_planning(request: ResearchRequest, ctx: RoutingContext) -> RouteDecision:
    prompt_tokens = estimate_tokens(request.question) + 400
    local = RoutingContext(
        settings=ctx.settings,
        learning=ctx.learning,
        remaining_budget=ctx.remaining_budget,
        estimated_input_tokens=max(ctx.estimated_input_tokens, prompt_tokens),
        disabled_aliases=ctx.disabled_aliases,
        allow_premium=ctx.allow_premium,
    )
    depth_tier = {
        "brief": ModelTier.cheap,
        "standard": ModelTier.standard,
        "deep": ModelTier.strong,
    }[request.desired_depth.value]
    return _decision("planning", depth_tier, local, None, None)


def route_one(sub: SubQuestion, request: ResearchRequest, ctx: RoutingContext) -> RouteDecision:
    score = sub.score or score_sub_question(sub, request)
    return _decision("subquestion", score.tier, ctx, score, sub.id)


def route_writing(
    request: ResearchRequest,
    ctx: RoutingContext,
    fact_count: int,
) -> RouteDecision:
    score = writing_score(request, fact_count)
    return _decision("writing", score.tier, ctx, score, None)


def require_spec(alias: str) -> ModelSpec:
    spec = get_spec(alias)
    if spec is None:
        raise KeyError(alias)
    return spec
