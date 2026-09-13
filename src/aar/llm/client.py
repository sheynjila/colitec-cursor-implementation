from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from aar.errors import ProviderError
from aar.llm.catalog import get_spec


@dataclass
class GenerationResult:
    text: str
    alias: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str
    estimated_cost_usd: float
    actual_cost_usd: float
    usage_delta: float = 0.0
    error_code: str | None = None


@dataclass
class ModelClient:
    adapters: dict[str, Any]
    call_log: list[dict[str, Any]] = field(default_factory=list)

    def generate(
        self,
        alias: str,
        prompt: str,
        *,
        max_output_tokens: int,
        system: str | None = None,
    ) -> GenerationResult:
        spec = get_spec(alias)
        if spec is None:
            raise ProviderError(f"Unknown model alias: {alias}", code="unknown_alias")
        adapter = self.adapters.get(spec.provider)
        if adapter is None:
            raise ProviderError(
                f"No adapter registered for provider {spec.provider}.",
                code="adapter_missing",
            )
        started = time.perf_counter()
        try:
            result = adapter.generate(
                spec, prompt, max_output_tokens=max_output_tokens, system=system
            )
        except ProviderError as exc:
            self.call_log.append(
                {
                    "alias": alias,
                    "provider": spec.provider,
                    "error_code": exc.code,
                }
            )
            raise
        except Exception as exc:  # noqa: BLE001
            self.call_log.append(
                {
                    "alias": alias,
                    "provider": spec.provider,
                    "error_code": "provider_error",
                }
            )
            raise ProviderError(str(exc)[:180]) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        estimated = spec.projected_cost(result.input_tokens, result.output_tokens)
        result.alias = alias
        result.provider = spec.provider
        result.latency_ms = result.latency_ms or latency_ms
        result.estimated_cost_usd = estimated
        if result.actual_cost_usd == 0.0:
            result.actual_cost_usd = estimated
        result.usage_delta = result.actual_cost_usd - estimated
        self.call_log.append(
            {
                "alias": result.alias,
                "provider": result.provider,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "estimated_cost_usd": result.estimated_cost_usd,
                "actual_cost_usd": result.actual_cost_usd,
            }
        )
        return result


def estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)
