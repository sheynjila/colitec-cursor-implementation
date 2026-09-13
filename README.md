# Colitech Multi-Agent Research Assistant

Local, evidence-grounded research assistant. A structured request is planned into sub-questions, routed to an eligible model alias, researched from permitted sources, validated against retrieved evidence, and written only from accepted facts.

This workspace started empty except for the implementation prompt. The Adaptive Multi-Agent Research Assistant v2 runbook and Colitech graph image were not in the repository, so the product was built from `Colitech_Agentic_AI_Final_Prompt.txt` under `src/aar`.

## Requirements

- Python 3.12 or newer (developed on 3.13)
- Optional: Tavily, Gemini, Anthropic, OpenRouter, Ollama, or Bedrock credentials

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
copy .env.example .env
```

The checked-in `.env.example` enables labeled fake models and fixture web pages so you can run without keys. That is demo/test mode, not live research.

## Sample local run

```powershell
python -m aar
```

Open http://127.0.0.1:8000 and submit:

> What evidence-grounded constraints should a multi-agent research assistant enforce?

Recommended first request: `allowed_sources=documents`, `desired_depth=brief`, `max_cost_usd=0`. The local keyword index includes short Colitech policy notes. The report must cite those passages or say evidence is insufficient.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/api/research` | 202 + `run_id` |
| GET | `/api/runs/{run_id}` | status, round, warnings, cost |
| GET | `/api/report/{run_id}` | report, citations, QA, cost |
| GET | `/api/stream/{run_id}` | SSE progress |
| GET | `/api/models` | catalog + learning scores |
| GET | `/api/metrics/summary` | aggregate telemetry |
| POST | `/api/runs/{run_id}/feedback` | post-run rating |
| GET | `/health`, `/ready` | liveness / readiness |

## Checks

```powershell
python -m pytest
python -m ruff check src tests
python -m mypy src/aar
```

Latest local run (2026-09-13):

- `python -m pytest -q` → **24 passed, 2 skipped** (optional live model and Tavily smokes)
- `python -m ruff check src tests` → **All checks passed**
- `python -m mypy src/aar` → **Success: no issues found in 50 source files**

Optional live smokes (`pytest -m live`) use the first configured model backend:

- local Ollama when `AAR_LIVE_OLLAMA=true` and Ollama is running (no Gemini/Anthropic key). Set `AAR_OLLAMA_CHAT_MODEL` to the exact `ollama list` name if it is not `llama3.2` / `llama3.1`.
- AWS Bedrock when `AAR_ENABLE_BEDROCK=true` and AWS credentials are present
- otherwise Gemini, Anthropic, or OpenRouter if that key is set

Tavily is web search only. `AAR_TAVILY_API_KEY` does not replace a model backend.

## Architecture

START → Request Intake → Planner → Complexity Router → Research Supervisor → Send(Web and/or Document workers) → Research Join → Validator → Coverage Gate → Writer → Final QA → Learning Update → END

Coverage Gate may loop only to Research Supervisor. Planner runs once. `source_hint=both` creates two Sends, not a third worker. The summarizer is a utility used by workers.

SQLite stores run records, telemetry, provenance, learning scores, and the document index. LangGraph checkpoints use a separate SQLite file. See `docs/graph.md`.

## Cost and tools

The request ceiling is enforced before each paid model call with an atomic reservation. Tavily search is billed by the vendor per request. If `AAR_TAVILY_COST_USD` is unset, that tool is **not** included in the hard all-in ceiling.

## Gap report

See the final status in the implementation notes after tests run. Live providers are exercised only when you add keys. Automated tests use the fake provider and labeled fixtures.
