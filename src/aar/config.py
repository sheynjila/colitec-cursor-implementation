"""Environment-backed settings. Credentials are never printed or persisted."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AAR_",
        env_file=".env",
        extra="ignore",
    )

    env: str = "local"
    data_dir: Path = Field(default_factory=lambda: _project_root() / "data")
    host: str = "127.0.0.1"
    port: int = 8000

    enable_fake: bool = False
    enable_bedrock: bool = False
    web_mode: str = "live"  # live | fixture
    demo_fixtures: bool = False
    allow_premium: bool = False
    max_output_tokens: int = 1024
    worker_timeout_seconds: float = 20.0
    fetch_max_bytes: int = 200_000
    tool_uses_per_subquestion: int = 4
    max_research_rounds: int = 2

    tavily_api_key: str = ""
    tavily_cost_usd: float | None = None

    @field_validator("tavily_cost_usd", mode="before")
    @classmethod
    def _empty_optional_float(cls, value: object) -> object:
        if value == "":
            return None
        return value

    ollama_host: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    openrouter_api_key: str = ""
    aws_region: str = "us-east-1"

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir

    @property
    def app_db_path(self) -> Path:
        return self.ensure_data_dir() / "app.db"

    @property
    def checkpoint_db_path(self) -> Path:
        return self.ensure_data_dir() / "checkpoints.sqlite"

    @property
    def corpus_dir(self) -> Path:
        path = self.ensure_data_dir() / "corpus"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def fixture_or_demo(self) -> bool:
        return self.web_mode == "fixture" or self.demo_fixtures or self.enable_fake

    def key_presence(self) -> dict[str, bool]:
        return {
            "tavily": bool(self.tavily_api_key),
            "gemini": bool(self.gemini_api_key),
            "anthropic": bool(self.anthropic_api_key),
            "openrouter": bool(self.openrouter_api_key),
            "ollama": True,
            "bedrock": bool(self.enable_bedrock),
        }


def load_settings() -> Settings:
    return Settings()
