"""HTTP-backed provider adapters. Keys are read only at the adapter boundary."""

from __future__ import annotations

import time

import httpx

from aar.config import Settings
from aar.errors import ProviderError
from aar.llm.catalog import ModelSpec
from aar.llm.client import GenerationResult, estimate_tokens


def _usage_tokens(payload: dict, prompt: str, text: str) -> tuple[int, int]:
    usage = payload.get("usage") or payload.get("usageMetadata") or {}
    input_tokens = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("promptTokenCount")
        or estimate_tokens(prompt)
    )
    output_tokens = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("candidatesTokenCount")
        or estimate_tokens(text)
    )
    return input_tokens, output_tokens


class OllamaAdapter:
    provider = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.host = settings.ollama_host.rstrip("/")
        self.chat_model = settings.ollama_chat_model.strip()

    def generate(
        self,
        spec: ModelSpec,
        prompt: str,
        *,
        max_output_tokens: int,
        system: str | None = None,
    ) -> GenerationResult:
        body = {
            "model": self.chat_model or spec.vendor_id,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "options": {"num_predict": max_output_tokens},
        }
        try:
            response = httpx.post(f"{self.host}/api/generate", json=body, timeout=120.0)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Ollama request failed: {exc!s}"[:180]) from exc
        text = str(data.get("response") or "")
        return GenerationResult(
            text=text,
            alias=spec.alias,
            provider="ollama",
            input_tokens=int(data.get("prompt_eval_count") or estimate_tokens(prompt)),
            output_tokens=int(data.get("eval_count") or estimate_tokens(text)),
            latency_ms=float(data.get("total_duration") or 0) / 1_000_000.0,
            finish_reason=str(data.get("done_reason") or "stop"),
            estimated_cost_usd=0.0,
            actual_cost_usd=0.0,
        )


class GeminiAdapter:
    provider = "gemini"

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.gemini_api_key

    def generate(
        self,
        spec: ModelSpec,
        prompt: str,
        *,
        max_output_tokens: int,
        system: str | None = None,
    ) -> GenerationResult:
        if not self.api_key:
            raise ProviderError("Gemini key is not configured.", code="provider_unavailable")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{spec.vendor_id}:generateContent"
        )
        body: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_output_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        try:
            response = httpx.post(url, params={"key": self.api_key}, json=body, timeout=45.0)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Gemini request failed: {exc!s}"[:180]) from exc
        text = ""
        for candidate in data.get("candidates") or []:
            for part in ((candidate.get("content") or {}).get("parts") or []):
                text += str(part.get("text") or "")
        input_tokens, output_tokens = _usage_tokens(data, prompt, text)
        return GenerationResult(
            text=text,
            alias=spec.alias,
            provider="gemini",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=0.0,
            finish_reason="stop",
            estimated_cost_usd=0.0,
            actual_cost_usd=0.0,
        )


class AnthropicAdapter:
    provider = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.anthropic_api_key

    def generate(
        self,
        spec: ModelSpec,
        prompt: str,
        *,
        max_output_tokens: int,
        system: str | None = None,
    ) -> GenerationResult:
        if not self.api_key:
            raise ProviderError("Anthropic key is not configured.", code="provider_unavailable")
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": spec.vendor_id,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        try:
            response = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=body,
                timeout=45.0,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Anthropic request failed: {exc!s}"[:180]) from exc
        text = "".join(
            str(block.get("text") or "")
            for block in data.get("content") or []
            if block.get("type") == "text"
        )
        input_tokens, output_tokens = _usage_tokens(data, prompt, text)
        return GenerationResult(
            text=text,
            alias=spec.alias,
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=0.0,
            finish_reason=str(data.get("stop_reason") or "stop"),
            estimated_cost_usd=0.0,
            actual_cost_usd=0.0,
        )


class OpenRouterAdapter:
    provider = "openrouter"

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.openrouter_api_key

    def generate(
        self,
        spec: ModelSpec,
        prompt: str,
        *,
        max_output_tokens: int,
        system: str | None = None,
    ) -> GenerationResult:
        if not self.api_key:
            raise ProviderError("OpenRouter key is not configured.", code="provider_unavailable")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": spec.vendor_id,
                    "messages": messages,
                    "max_tokens": max_output_tokens,
                },
                timeout=45.0,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"OpenRouter request failed: {exc!s}"[:180]) from exc
        text = str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        input_tokens, output_tokens = _usage_tokens(data, prompt, text)
        return GenerationResult(
            text=text,
            alias=spec.alias,
            provider="openrouter",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=0.0,
            finish_reason=str((data.get("choices") or [{}])[0].get("finish_reason") or "stop"),
            estimated_cost_usd=0.0,
            actual_cost_usd=0.0,
        )


class BedrockAdapter:
    provider = "bedrock"

    def __init__(self, settings: Settings) -> None:
        self.region = settings.aws_region

    def generate(
        self,
        spec: ModelSpec,
        prompt: str,
        *,
        max_output_tokens: int,
        system: str | None = None,
    ) -> GenerationResult:
        try:
            import boto3
        except Exception as exc:  # noqa: BLE001
            raise ProviderError("boto3 is not installed.", code="provider_unavailable") from exc
        try:
            client = boto3.client("bedrock-runtime", region_name=self.region)
            body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt if not system else f"{system}\n\n{prompt}"}],
                    }
                ],
                "inferenceConfig": {"maxTokens": max_output_tokens},
            }
            started = time.perf_counter()
            response = client.converse(modelId=spec.vendor_id, **body)
            latency_ms = (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Bedrock request failed: {exc!s}"[:180]) from exc
        text = ""
        for item in ((response.get("output") or {}).get("message") or {}).get("content") or []:
            text += str(item.get("text") or "")
        usage = response.get("usage") or {}
        return GenerationResult(
            text=text,
            alias=spec.alias,
            provider="bedrock",
            input_tokens=int(usage.get("inputTokens") or estimate_tokens(prompt)),
            output_tokens=int(usage.get("outputTokens") or estimate_tokens(text)),
            latency_ms=latency_ms,
            finish_reason=str(response.get("stopReason") or "stop"),
            estimated_cost_usd=0.0,
            actual_cost_usd=0.0,
        )
