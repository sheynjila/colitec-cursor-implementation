"""Optional live smokes. Skipped unless a local or cloud backend is actually configured."""

from __future__ import annotations

import os

import httpx
import pytest

from aar.config import Settings
from aar.llm.adapters.http import (
    AnthropicAdapter,
    BedrockAdapter,
    GeminiAdapter,
    OllamaAdapter,
    OpenRouterAdapter,
)
from aar.llm.catalog import get_spec
from aar.tools.web import WebSearchTool


def _settings() -> Settings:
    return Settings()


def _ollama_reachable(host: str) -> bool:
    try:
        response = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=2.0)
        return response.status_code < 500
    except Exception:
        return False


@pytest.mark.live
def test_optional_live_model_smoke() -> None:
    settings = _settings()
    live_ollama = os.getenv("AAR_LIVE_OLLAMA", "").lower() in {"1", "true", "yes"}
    if live_ollama and _ollama_reachable(settings.ollama_host):
        spec = get_spec("ollama-llama3.2")
        assert spec
        result = OllamaAdapter(settings).generate(
            spec, "Reply with the word ok.", max_output_tokens=16
        )
        assert result.text
        return
    if settings.enable_bedrock:
        spec = get_spec("bedrock-nova-micro")
        assert spec
        result = BedrockAdapter(settings).generate(
            spec, "Reply with the word ok.", max_output_tokens=16
        )
        assert result.text
        return
    if settings.gemini_api_key:
        spec = get_spec("gemini-flash-lite")
        assert spec
        result = GeminiAdapter(settings).generate(
            spec, "Reply with the word ok.", max_output_tokens=16
        )
        assert result.text
        return
    if settings.anthropic_api_key:
        spec = get_spec("claude-haiku")
        assert spec
        result = AnthropicAdapter(settings).generate(
            spec, "Reply with the word ok.", max_output_tokens=16
        )
        assert result.text
        return
    if settings.openrouter_api_key:
        spec = get_spec("openrouter-llama")
        assert spec
        result = OpenRouterAdapter(settings).generate(
            spec, "Reply with the word ok.", max_output_tokens=16
        )
        assert result.text
        return
    pytest.skip(
        "No live model backend configured. Set AAR_LIVE_OLLAMA=true with Ollama running, "
        "AAR_ENABLE_BEDROCK=true with AWS credentials, or a Gemini/Anthropic/OpenRouter key."
    )


@pytest.mark.live
def test_optional_live_tavily_search() -> None:
    settings = _settings()
    if not settings.tavily_api_key:
        pytest.skip("AAR_TAVILY_API_KEY is unset. Tavily is web search, not a model provider.")
    settings.web_mode = "live"
    settings.demo_fixtures = False
    hits = WebSearchTool(settings).search("evidence grounded research assistant", limit=1)
    assert hits
    assert hits[0].url.startswith("http")
    assert not hits[0].labeled_fixture
