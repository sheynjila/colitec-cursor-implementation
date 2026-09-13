from __future__ import annotations

REQUIRED_EVENTS = {
    "run_started",
    "plan_created",
    "route_selected",
    "worker_started",
    "coverage_decision",
    "report_drafted",
    "qa_completed",
}

SAMPLE = {
    "question": "What evidence-grounded constraints should a multi-agent research assistant enforce?",
    "allowed_sources": "documents",
    "desired_depth": "brief",
    "risk_level": "low",
    "max_cost_usd": 0,
}


def test_rejects_malformed_request(client) -> None:
    response = client.post("/api/research", json={**SAMPLE, "max_cost_usd": -1})
    assert response.status_code == 422


def test_complete_api_run_without_network(client) -> None:
    created = client.post("/api/research", json=SAMPLE)
    assert created.status_code == 202
    run_id = created.json()["run_id"]

    status = client.get(f"/api/runs/{run_id}")
    assert status.status_code == 200
    assert status.json()["status"] in {"queued", "running", "succeeded"}
    assert "spent_usd" in status.json()

    report = client.get(f"/api/report/{run_id}")
    assert report.status_code == 200
    body = report.json()
    assert body["report"]
    assert "citations" in body
    assert "limitations" in body
    assert "spent_usd" in body
    assert body["qa_status"] in {"approved", "not_approved"}
    assert "insufficient" in body["report"]["markdown"].lower() or body["report"]["sources"]
    assert "invented.example" not in body["report"]["markdown"]

    events = list(client.get(f"/api/stream/{run_id}").iter_lines())
    joined = "\n".join(line.decode() if isinstance(line, bytes) else line for line in events)
    assert "run_started" in joined or "run_completed" in joined
    assert "run_completed" in joined or "succeeded" in joined

    feedback = client.post(f"/api/runs/{run_id}/feedback", json={"rating": 5, "note": "clear limitations"})
    assert feedback.status_code == 200
    stored = client.app.state.runtime.db.conn.execute(
        "SELECT rating, note FROM feedback WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert stored["rating"] == 5
    result = client.app.state.runtime.registry.get_result(run_id)
    assert "rating" not in (result or {})

    models = client.get("/api/models")
    assert models.status_code == 200
    assert all("api_key" not in item for item in models.json())

    metrics = client.get("/api/metrics/summary")
    assert metrics.status_code == 200
    assert metrics.json()["runs"] >= 1

    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready").json()
    assert ready["ready"] is True
    assert set(ready["keys_present"]) == {
        "tavily",
        "gemini",
        "anthropic",
        "openrouter",
        "ollama",
        "bedrock",
    }
    assert all(isinstance(flag, bool) for flag in ready["keys_present"].values())
    assert "AIza" not in str(ready)
    assert "sk-" not in str(ready)


def test_sse_event_order(client) -> None:
    run_id = client.post("/api/research", json=SAMPLE).json()["run_id"]
    events = client.app.state.runtime.events.list_for_run(run_id)
    types = [event.event_type for event in events]
    assert REQUIRED_EVENTS.issubset(set(types))
    assert "intake" in types
    assert types[-1] in {"run_completed", "run_failed"}
    assert types.index("plan_created") < types.index("qa_completed")


def test_unknown_run_is_controlled_error(client) -> None:
    response = client.get("/api/runs/missing-run")
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "not_found"
    assert "Traceback" not in response.text
    assert "secret" not in response.text.lower()


def test_ui_has_progress_and_report_panels(client) -> None:
    page = client.get("/")
    assert page.status_code == 200
    html = page.text
    assert "Progress" in html
    assert "Report" in html
    assert 'id="question"' in html
    assert "/api/stream/" in html
    assert "Sources" in html
    assert "logo" not in html.lower() or "Colitech" in html
