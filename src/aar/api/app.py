from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError as PydanticValidationError

from aar.api.service import RunService
from aar.config import Settings, load_settings
from aar.errors import AARError, NotFoundError
from aar.graph.builder import compile_graph, sqlite_checkpointer
from aar.graph.runtime import AppRuntime, build_runtime
from aar.llm.catalog import catalog_entries, enabled_for
from aar.schemas import (
    CatalogEntryView,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsSummary,
    ReadyResponse,
    ReportResponse,
    ResearchAccepted,
    ResearchRequest,
    RunStatus,
    RunStatusResponse,
)


def create_app(
    settings: Settings | None = None,
    runtime: AppRuntime | None = None,
    graph: Any | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    runtime = runtime or build_runtime(settings)
    if graph is None:
        graph = compile_graph(sqlite_checkpointer(str(settings.checkpoint_db_path)))
    service = RunService(runtime, graph=graph)

    app = FastAPI(title="Colitech Multi-Agent Research Assistant", version="0.1.0")
    app.state.settings = settings
    app.state.runtime = runtime
    app.state.service = service
    ui_path = Path(__file__).resolve().parent.parent / "ui" / "index.html"

    @app.exception_handler(AARError)
    async def aar_error_handler(_request: Request, exc: AARError) -> JSONResponse:
        status_code = 404 if isinstance(exc, NotFoundError) else 400
        return JSONResponse(status_code=status_code, content=exc.as_dict())

    @app.exception_handler(Exception)
    async def hidden_error_handler(_request: Request, _exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"code": "internal_error", "message": "The request failed due to an internal error."},
        )

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(ui_path)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    async def ready() -> ReadyResponse:
        checks = {
            "sqlite": _sqlite_ready(runtime),
            "catalog": bool(catalog_entries()),
            "documents": runtime.documents.count() >= 0,
        }
        return ReadyResponse(
            ready=all(checks.values()),
            checks=checks,
            keys_present=settings.key_presence(),
        )

    @app.post("/api/research", response_model=ResearchAccepted, status_code=202)
    async def research(payload: ResearchRequest) -> ResearchAccepted:
        try:
            request = ResearchRequest.model_validate(payload.model_dump())
        except PydanticValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        if settings.env == "test" or settings.enable_fake and settings.web_mode == "fixture":
            # Keep the HTTP contract (202 + run_id) but execute inline in test/demo.
            run_id = _create_and_maybe_run(service, request, sync=settings.env == "test")
        else:
            run_id = service.create_run(request)
        return ResearchAccepted(run_id=run_id, status=RunStatus.queued)

    @app.get("/api/runs/{run_id}", response_model=RunStatusResponse)
    async def get_run(run_id: str) -> RunStatusResponse:
        record = runtime.registry.get(run_id)
        spent, reserved = runtime.budget.snapshot(run_id)
        return RunStatusResponse(
            run_id=record.run_id,
            status=record.status,
            research_round=record.research_round,
            warnings=record.warnings,
            spent_usd=spent if spent else record.spent_usd,
            reserved_usd=reserved if reserved else record.reserved_usd,
            error_code=record.error_code,
            error_message=record.error_message,
        )

    @app.get("/api/report/{run_id}", response_model=ReportResponse)
    async def get_report(run_id: str) -> ReportResponse:
        record = runtime.registry.get(run_id)
        result = runtime.registry.get_result(run_id)
        report_model = runtime.registry.get_report(run_id)
        return ReportResponse(
            run_id=run_id,
            status=record.status,
            report=report_model,
            citations=report_model.sources if report_model else [],
            limitations=(result or {}).get("limitations") or [],
            qa_status=report_model.qa_status if report_model else None,
            spent_usd=record.spent_usd,
            planner_fallback=bool((result or {}).get("planner_fallback")),
            writer_fallback=bool((result or {}).get("writer_fallback")),
            error_code=record.error_code,
            error_message=record.error_message,
        )

    @app.get("/api/stream/{run_id}")
    async def stream(run_id: str) -> StreamingResponse:
        runtime.registry.get(run_id)

        async def event_publisher():
            last_id = 0
            while True:
                events = runtime.events.list_for_run(run_id, after_id=last_id)
                for event in events:
                    last_id = event.id or last_id
                    payload = event.model_dump(mode="json")
                    yield f"event: {event.event_type}\ndata: {json.dumps(payload)}\n\n"
                    if event.event_type in {"run_completed", "run_failed"}:
                        return
                record = runtime.registry.get(run_id)
                if record.status in {RunStatus.succeeded, RunStatus.failed} and not events:
                    terminal = (
                        "run_completed" if record.status == RunStatus.succeeded else "run_failed"
                    )
                    yield (
                        f"event: {terminal}\n"
                        f"data: {json.dumps({'run_id': run_id, 'event_type': terminal})}\n\n"
                    )
                    return
                await asyncio.sleep(0.15)

        return StreamingResponse(
            event_publisher(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/models", response_model=list[CatalogEntryView])
    async def models() -> list[CatalogEntryView]:
        scores = runtime.learning.all_scores()
        views = []
        for spec in catalog_entries():
            learned = scores.get(spec.alias, {})
            views.append(
                CatalogEntryView(
                    alias=spec.alias,
                    provider=spec.provider,
                    vendor_id=spec.vendor_id,
                    tier=spec.tier,
                    enabled=enabled_for(spec, settings),
                    premium=spec.premium,
                    context_window=spec.context_window,
                    capabilities=list(spec.capabilities),
                    input_cost_per_mtok=spec.input_cost_per_mtok,
                    output_cost_per_mtok=spec.output_cost_per_mtok,
                    preference_score=float(learned.get("preference_score") or 0.0),
                    pass_ema=float(learned.get("pass_ema") or 0.0),
                    cost_ema=float(learned.get("cost_ema") or 0.0),
                    latency_ema=float(learned.get("latency_ema") or 0.0),
                )
            )
        return views

    @app.get("/api/metrics/summary", response_model=MetricsSummary)
    async def metrics() -> MetricsSummary:
        return MetricsSummary.model_validate(runtime.telemetry.summary())

    @app.post("/api/runs/{run_id}/feedback", response_model=FeedbackResponse)
    async def feedback(run_id: str, payload: FeedbackRequest) -> FeedbackResponse:
        service.store_feedback(run_id, payload.rating, payload.note)
        return FeedbackResponse(run_id=run_id)

    return app


def _create_and_maybe_run(service: RunService, request: ResearchRequest, *, sync: bool) -> str:
    if not sync:
        return service.create_run(request)
    run_id = str(__import__("uuid").uuid4())
    service.runtime.registry.create(request, run_id)
    try:
        service.execute(run_id, request)
    except Exception:
        pass
    return run_id


def _sqlite_ready(runtime: AppRuntime) -> bool:
    try:
        runtime.db.conn.execute("SELECT 1").fetchone()
        return True
    except Exception:
        return False


