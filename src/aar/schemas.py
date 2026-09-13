"""Typed request, graph-adjacent, and API response schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AllowedSources(StrEnum):
    web = "web"
    documents = "documents"
    both = "both"


class DesiredDepth(StrEnum):
    brief = "brief"
    standard = "standard"
    deep = "deep"


class RiskLevel(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"


class SourceHint(StrEnum):
    web = "web"
    documents = "documents"
    both = "both"


class ModelTier(StrEnum):
    cheap = "cheap"
    standard = "standard"
    strong = "strong"


class RunStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class QAStatus(StrEnum):
    approved = "approved"
    not_approved = "not_approved"
    pending = "pending"


class ResearchRequest(BaseModel):
    question: str = Field(min_length=8, max_length=4000)
    allowed_sources: AllowedSources
    desired_depth: DesiredDepth
    risk_level: RiskLevel
    max_cost_usd: float = Field(ge=0.0, le=100.0)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if len(cleaned) < 8:
            raise ValueError("question must contain at least 8 characters")
        return cleaned


class ScoreBreakdown(BaseModel):
    reasoning_depth: int = Field(ge=0, le=5)
    context_size: int = Field(ge=0, le=5)
    tool_need: int = Field(ge=0, le=5)
    risk: int = Field(ge=0, le=5)
    accuracy_need: int = Field(ge=0, le=5)
    total: int = Field(ge=0, le=25)
    tier: ModelTier


class RouteDecision(BaseModel):
    purpose: Literal["planning", "subquestion", "writing"]
    sub_question_id: str | None = None
    score: ScoreBreakdown | None = None
    required_tier: ModelTier | None = None
    selected_alias: str | None = None
    selected_provider: str | None = None
    projected_cost_usd: float = 0.0
    rejection_reasons: list[str] = Field(default_factory=list)
    fallback: str | None = None


class SubQuestion(BaseModel):
    id: str
    text: str
    source_hint: SourceHint
    coverage_goal: str
    covered: bool = False
    active: bool = True
    score: ScoreBreakdown | None = None
    route: RouteDecision | None = None


class Fact(BaseModel):
    id: str
    claim: str
    evidence_quote: str
    source_ref: str
    source_title: str
    sub_question_id: str
    validated: bool = False
    rejection_reason: str | None = None


class ProvenanceRecord(BaseModel):
    ref: str
    title: str
    source_kind: Literal["web", "document"]
    text: str
    document_id: str | None = None
    chunk_id: str | None = None
    retrieved_url: str | None = None
    labeled_fixture: bool = False


class ValidationResult(BaseModel):
    fact_id: str
    accepted: bool
    reason: str | None = None


class ModelAttempt(BaseModel):
    alias: str
    provider: str
    purpose: str
    sub_question_id: str | None = None
    node: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    actual_cost_usd: float = 0.0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    paid: bool = False
    error_code: str | None = None


class ProgressEvent(BaseModel):
    id: int | None = None
    run_id: str
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    node: str | None = None
    sub_question_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class SourceItem(BaseModel):
    number: int
    ref: str
    title: str


class ResearchReport(BaseModel):
    executive_summary: str
    sections: list[dict[str, Any]]
    markdown: str
    sources: list[SourceItem]
    limitations: list[str]
    qa_status: QAStatus = QAStatus.pending
    qa_warnings: list[str] = Field(default_factory=list)
    writer_fallback: bool = False


class RunRecord(BaseModel):
    run_id: str
    status: RunStatus
    request: ResearchRequest
    research_round: int = 0
    warnings: list[str] = Field(default_factory=list)
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    error_code: str | None = None
    error_message: str | None = None


class RunStatusResponse(BaseModel):
    run_id: str
    status: RunStatus
    research_round: int
    warnings: list[str]
    spent_usd: float
    reserved_usd: float
    error_code: str | None = None
    error_message: str | None = None


class ResearchAccepted(BaseModel):
    run_id: str
    status: RunStatus = RunStatus.queued


class ReportResponse(BaseModel):
    run_id: str
    status: RunStatus
    report: ResearchReport | None = None
    citations: list[SourceItem] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    qa_status: QAStatus | None = None
    spent_usd: float = 0.0
    planner_fallback: bool = False
    writer_fallback: bool = False
    error_code: str | None = None
    error_message: str | None = None


class FeedbackRequest(BaseModel):
    rating: int = Field(ge=1, le=5)
    note: str = Field(default="", max_length=2000)


class FeedbackResponse(BaseModel):
    run_id: str
    stored: bool = True


class CatalogEntryView(BaseModel):
    alias: str
    provider: str
    vendor_id: str
    tier: ModelTier
    enabled: bool
    premium: bool
    context_window: int
    capabilities: list[str]
    input_cost_per_mtok: float
    output_cost_per_mtok: float
    preference_score: float = 0.0
    pass_ema: float = 0.0
    cost_ema: float = 0.0
    latency_ema: float = 0.0


class MetricsSummary(BaseModel):
    runs: int
    total_cost_usd: float
    calls_by_tier: dict[str, int]
    validation_pass_rate: float
    premium_avoidance_rate: float
    average_latency_ms: float


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, bool]
    keys_present: dict[str, bool] = Field(default_factory=dict)
