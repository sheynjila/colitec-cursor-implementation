from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aar.api.app import create_app
from aar.api.service import RunService
from aar.config import Settings
from aar.graph.builder import compile_graph
from aar.graph.runtime import AppRuntime, build_runtime
from aar.llm.adapters.fake import FakeAdapter
from aar.storage.db import Database


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        env="test",
        data_dir=tmp_path,
        enable_fake=True,
        web_mode="fixture",
        demo_fixtures=True,
        allow_premium=False,
        max_output_tokens=256,
    )


@pytest.fixture
def runtime(settings: Settings) -> AppRuntime:
    return build_runtime(settings, db=Database(settings.app_db_path), fake=FakeAdapter())


@pytest.fixture
def graph():
    return compile_graph()


@pytest.fixture
def service(runtime: AppRuntime, graph) -> RunService:
    return RunService(runtime, graph=graph)


@pytest.fixture
def client(settings: Settings, runtime: AppRuntime, graph) -> TestClient:
    return TestClient(create_app(settings=settings, runtime=runtime, graph=graph))
