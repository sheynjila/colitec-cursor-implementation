from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aar.config import Settings
from aar.llm.adapters.fake import FakeAdapter
from aar.llm.adapters.http import (
    AnthropicAdapter,
    BedrockAdapter,
    GeminiAdapter,
    OllamaAdapter,
    OpenRouterAdapter,
)
from aar.llm.client import ModelClient
from aar.storage.budget import BudgetLedger
from aar.storage.db import Database
from aar.storage.documents import DocumentIndex
from aar.storage.events import EventStore
from aar.storage.learning import LearningStore
from aar.storage.provenance import ProvenanceLedger
from aar.storage.registry import RunRegistry
from aar.storage.telemetry import TelemetryStore
from aar.storage.tools_counter import ToolUseCounter
from aar.tools.web import WebSearchTool


@dataclass
class AppRuntime:
    settings: Settings
    db: Database
    registry: RunRegistry
    events: EventStore
    telemetry: TelemetryStore
    learning: LearningStore
    provenance: ProvenanceLedger
    documents: DocumentIndex
    budget: BudgetLedger
    tool_uses: ToolUseCounter
    models: ModelClient
    web: WebSearchTool
    fake: FakeAdapter | None = None


def build_runtime(
    settings: Settings,
    *,
    db: Database | None = None,
    fake: FakeAdapter | None = None,
) -> AppRuntime:
    database = db or Database(settings.app_db_path)
    documents = DocumentIndex(database)
    _seed_corpus(settings.corpus_dir)
    if documents.count() == 0:
        documents.index_directory(settings.corpus_dir)
    adapters: dict[str, Any] = {
        "ollama": OllamaAdapter(settings),
        "gemini": GeminiAdapter(settings),
        "anthropic": AnthropicAdapter(settings),
        "openrouter": OpenRouterAdapter(settings),
        "bedrock": BedrockAdapter(settings),
    }
    fake_adapter = fake or FakeAdapter()
    adapters["fake"] = fake_adapter
    return AppRuntime(
        settings=settings,
        db=database,
        registry=RunRegistry(database),
        events=EventStore(database),
        telemetry=TelemetryStore(database),
        learning=LearningStore(database),
        provenance=ProvenanceLedger(database),
        documents=documents,
        budget=BudgetLedger(),
        tool_uses=ToolUseCounter(settings.tool_uses_per_subquestion),
        models=ModelClient(adapters=adapters),
        web=WebSearchTool(settings),
        fake=fake_adapter,
    )


def _seed_corpus(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.glob("*.txt")):
        return
    packaged = Path(__file__).resolve().parents[1] / "corpus"
    if not packaged.exists():
        return
    for src in packaged.glob("*.txt"):
        shutil.copy(src, dest / src.name)


def get_runtime(config: dict[str, Any] | Any) -> AppRuntime:
    runtime = (config.get("configurable") or {}).get("app_runtime")
    if runtime is None:
        raise RuntimeError("app_runtime is missing from graph config")
    return runtime
