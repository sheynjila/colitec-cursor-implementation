from __future__ import annotations

from typing import Protocol

from aar.llm.catalog import ModelSpec
from aar.llm.client import GenerationResult


class ProviderAdapter(Protocol):
    provider: str

    def generate(
        self,
        spec: ModelSpec,
        prompt: str,
        *,
        max_output_tokens: int,
        system: str | None = None,
    ) -> GenerationResult: ...
