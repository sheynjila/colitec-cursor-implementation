"""Deterministic complexity scoring. Each dimension is 0-5.

Rules (applied in order, then clamped):
- reasoning_depth: +2 if why/how/compare/versus/vs/impact/cause/because/trade-off;
  +2 if desired_depth is deep; +1 if standard; +1 if sub-question has more than 18 words.
- context_size: +1 if under 12 words, +2 if under 24, else +3; +1 if source_hint is both;
  +1 if desired_depth is deep.
- tool_need: +1 if web or documents, +2 if both; +1 if latest/current/today/recent/2024/2025/2026;
  +1 if desired_depth is deep.
- risk: low=1, medium=2, high=4; +2 if medical/legal/safety/clinical/financial/weapon.
- accuracy_need: +1 base; +2 if risk_level is high; +1 if exact/cite/statistic/date/percent/number/quantify;
  +1 if desired_depth is deep.

Tier map: 0-4 cheap, 5-10 standard, 11-25 strong.
"""

from __future__ import annotations

from aar.schemas import (
    DesiredDepth,
    ModelTier,
    ResearchRequest,
    ScoreBreakdown,
    SourceHint,
    SubQuestion,
)
from aar.textutil import word_count

REASONING_MARKERS = (
    "why",
    "how",
    "compare",
    "versus",
    " vs ",
    "impact",
    "cause",
    "because",
    "trade-off",
    "tradeoff",
)
TOOL_MARKERS = ("latest", "current", "today", "recent", "2024", "2025", "2026")
RISK_MARKERS = ("medical", "legal", "safety", "clinical", "financial", "weapon")
ACCURACY_MARKERS = ("exact", "cite", "statistic", "date", "percent", "number", "quantify")


def _clamp(value: int) -> int:
    return max(0, min(5, value))


def _has_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = f" {text.casefold()} "
    return any(marker in lowered for marker in markers)


def score_sub_question(sub: SubQuestion, request: ResearchRequest) -> ScoreBreakdown:
    text = f"{sub.text} {sub.coverage_goal}"
    words = word_count(sub.text)

    reasoning = 0
    if _has_any(text, REASONING_MARKERS):
        reasoning += 2
    if request.desired_depth == DesiredDepth.deep:
        reasoning += 2
    elif request.desired_depth == DesiredDepth.standard:
        reasoning += 1
    if words > 18:
        reasoning += 1

    if words < 12:
        context = 1
    elif words < 24:
        context = 2
    else:
        context = 3
    if sub.source_hint.value == "both":
        context += 1
    if request.desired_depth == DesiredDepth.deep:
        context += 1

    tool = 2 if sub.source_hint.value == "both" else 1
    if _has_any(text, TOOL_MARKERS):
        tool += 1
    if request.desired_depth == DesiredDepth.deep:
        tool += 1

    risk = {"low": 1, "medium": 2, "high": 4}[request.risk_level.value]
    if _has_any(text, RISK_MARKERS):
        risk += 2

    accuracy = 1
    if request.risk_level.value == "high":
        accuracy += 2
    if _has_any(text, ACCURACY_MARKERS):
        accuracy += 1
    if request.desired_depth == DesiredDepth.deep:
        accuracy += 1

    breakdown = ScoreBreakdown(
        reasoning_depth=_clamp(reasoning),
        context_size=_clamp(context),
        tool_need=_clamp(tool),
        risk=_clamp(risk),
        accuracy_need=_clamp(accuracy),
        total=0,
        tier=ModelTier.cheap,
    )
    total = (
        breakdown.reasoning_depth
        + breakdown.context_size
        + breakdown.tool_need
        + breakdown.risk
        + breakdown.accuracy_need
    )
    if total <= 4:
        tier = ModelTier.cheap
    elif total <= 10:
        tier = ModelTier.standard
    else:
        tier = ModelTier.strong
    return breakdown.model_copy(update={"total": total, "tier": tier})


def writing_score(request: ResearchRequest, fact_count: int) -> ScoreBreakdown:
    synthetic = SubQuestion(
        id="writing",
        text=("Write a cited report from validated facts " + request.question)[:200],
        source_hint=SourceHint.documents,
        coverage_goal="complete cited report",
    )
    scored = score_sub_question(synthetic, request)
    if fact_count > 8:
        total = min(25, scored.total + 1)
        tier = ModelTier.cheap if total <= 4 else ModelTier.standard if total <= 10 else ModelTier.strong
        return scored.model_copy(update={"total": total, "tier": tier})
    return scored
