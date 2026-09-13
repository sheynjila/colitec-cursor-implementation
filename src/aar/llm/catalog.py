"""Reviewable model catalog. Reviewed against official docs on 2026-09-13.

Sources consulted:
- https://docs.anthropic.com/en/docs/about-claude/models
- https://docs.anthropic.com/en/docs/about-claude/pricing
- https://ai.google.dev/gemini-api/docs/models
- https://ai.google.dev/gemini-api/docs/pricing
- https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html
- https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-lite.html
- https://openrouter.ai/anthropic
Verify vendor IDs and prices before enabling a live paid model.
"""

from __future__ import annotations

from dataclasses import dataclass

from aar.config import Settings
from aar.schemas import ModelTier


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    provider: str
    vendor_id: str
    tier: ModelTier
    context_window: int
    max_output_tokens: int
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    capabilities: tuple[str, ...] = ("chat", "json")
    premium: bool = False
    paid: bool = True

    def projected_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_mtok + output_tokens * self.output_cost_per_mtok
        ) / 1_000_000.0


CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        alias="fake-cheap",
        provider="fake",
        vendor_id="fake-cheap",
        tier=ModelTier.cheap,
        context_window=32_000,
        max_output_tokens=2048,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
        paid=False,
    ),
    ModelSpec(
        alias="fake-standard",
        provider="fake",
        vendor_id="fake-standard",
        tier=ModelTier.standard,
        context_window=64_000,
        max_output_tokens=2048,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
        paid=False,
    ),
    ModelSpec(
        alias="fake-strong",
        provider="fake",
        vendor_id="fake-strong",
        tier=ModelTier.strong,
        context_window=128_000,
        max_output_tokens=4096,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
        paid=False,
    ),
    ModelSpec(
        alias="fake-premium",
        provider="fake",
        vendor_id="fake-premium",
        tier=ModelTier.strong,
        context_window=200_000,
        max_output_tokens=4096,
        input_cost_per_mtok=10.0,
        output_cost_per_mtok=50.0,
        premium=True,
        paid=True,
    ),
    ModelSpec(
        alias="ollama-llama3.2",
        provider="ollama",
        vendor_id="llama3.2",
        tier=ModelTier.cheap,
        context_window=32_000,
        max_output_tokens=2048,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
        paid=False,
    ),
    ModelSpec(
        alias="ollama-llama3.1",
        provider="ollama",
        vendor_id="llama3.1",
        tier=ModelTier.standard,
        context_window=32_000,
        max_output_tokens=2048,
        input_cost_per_mtok=0.0,
        output_cost_per_mtok=0.0,
        paid=False,
    ),
    ModelSpec(
        alias="gemini-flash-lite",
        provider="gemini",
        vendor_id="gemini-3.1-flash-lite",
        tier=ModelTier.cheap,
        context_window=1_000_000,
        max_output_tokens=8192,
        input_cost_per_mtok=0.25,
        output_cost_per_mtok=1.50,
    ),
    ModelSpec(
        alias="gemini-flash",
        provider="gemini",
        vendor_id="gemini-3.6-flash",
        tier=ModelTier.standard,
        context_window=1_000_000,
        max_output_tokens=8192,
        input_cost_per_mtok=0.75,
        output_cost_per_mtok=3.75,
    ),
    ModelSpec(
        alias="gemini-pro",
        provider="gemini",
        vendor_id="gemini-3.1-pro-preview",
        tier=ModelTier.strong,
        context_window=1_000_000,
        max_output_tokens=8192,
        input_cost_per_mtok=2.00,
        output_cost_per_mtok=12.00,
        premium=True,
    ),
    ModelSpec(
        alias="claude-haiku",
        provider="anthropic",
        vendor_id="claude-haiku-4-5-20251001",
        tier=ModelTier.standard,
        context_window=200_000,
        max_output_tokens=8192,
        input_cost_per_mtok=1.00,
        output_cost_per_mtok=5.00,
    ),
    ModelSpec(
        alias="claude-sonnet",
        provider="anthropic",
        vendor_id="claude-sonnet-5",
        tier=ModelTier.strong,
        context_window=1_000_000,
        max_output_tokens=8192,
        input_cost_per_mtok=2.00,
        output_cost_per_mtok=10.00,
    ),
    ModelSpec(
        alias="claude-opus",
        provider="anthropic",
        vendor_id="claude-opus-5",
        tier=ModelTier.strong,
        context_window=1_000_000,
        max_output_tokens=8192,
        input_cost_per_mtok=5.00,
        output_cost_per_mtok=25.00,
        premium=True,
    ),
    ModelSpec(
        alias="openrouter-llama",
        provider="openrouter",
        vendor_id="meta-llama/llama-3.2-3b-instruct",
        tier=ModelTier.cheap,
        context_window=32_000,
        max_output_tokens=2048,
        input_cost_per_mtok=0.03,
        output_cost_per_mtok=0.06,
    ),
    ModelSpec(
        alias="openrouter-qwen",
        provider="openrouter",
        vendor_id="qwen/qwen-2.5-7b-instruct",
        tier=ModelTier.standard,
        context_window=32_000,
        max_output_tokens=2048,
        input_cost_per_mtok=0.10,
        output_cost_per_mtok=0.20,
    ),
    ModelSpec(
        alias="bedrock-nova-micro",
        provider="bedrock",
        vendor_id="amazon.nova-micro-v1:0",
        tier=ModelTier.cheap,
        context_window=128_000,
        max_output_tokens=2048,
        input_cost_per_mtok=0.035,
        output_cost_per_mtok=0.14,
    ),
    ModelSpec(
        alias="bedrock-nova-lite",
        provider="bedrock",
        vendor_id="amazon.nova-lite-v1:0",
        tier=ModelTier.standard,
        context_window=300_000,
        max_output_tokens=2048,
        input_cost_per_mtok=0.06,
        output_cost_per_mtok=0.24,
    ),
    ModelSpec(
        alias="bedrock-nova-pro",
        provider="bedrock",
        vendor_id="amazon.nova-pro-v1:0",
        tier=ModelTier.strong,
        context_window=300_000,
        max_output_tokens=2048,
        input_cost_per_mtok=0.80,
        output_cost_per_mtok=3.20,
        premium=True,
    ),
)


def get_spec(alias: str) -> ModelSpec | None:
    for spec in CATALOG:
        if spec.alias == alias:
            return spec
    return None


def catalog_entries() -> list[ModelSpec]:
    return list(CATALOG)


def provider_available(provider: str, settings: Settings) -> bool:
    if provider == "fake":
        return settings.enable_fake
    if provider == "ollama":
        return True
    if provider == "gemini":
        return bool(settings.gemini_api_key)
    if provider == "anthropic":
        return bool(settings.anthropic_api_key)
    if provider == "openrouter":
        return bool(settings.openrouter_api_key)
    if provider == "bedrock":
        return bool(settings.enable_bedrock)
    return False


def enabled_for(spec: ModelSpec, settings: Settings, extra_disabled: set[str] | None = None) -> bool:
    if extra_disabled and spec.alias in extra_disabled:
        return False
    return provider_available(spec.provider, settings)
