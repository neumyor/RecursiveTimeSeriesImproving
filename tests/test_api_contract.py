from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx

from harnessing_ts.state.workspace_store import WorkspaceStore

def _get(app, path: str) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(path)
    return asyncio.run(request())


def _post(app, path: str, payload: dict) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.post(path, json=payload)
    return asyncio.run(request())


def test_bootstrap_contract_includes_full_static_payload(tmp_path, monkeypatch) -> None:
    from harnessing_ts.server import create_app

    WorkspaceStore(tmp_path).ensure_layout()
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(tmp_path))

    response = _get(create_app(), "/api/bootstrap")

    assert response.status_code == 200
    payload = response.json()
    assert {
        "variant",
        "state",
        "timeline",
        "mainParts",
        "nodes",
        "nodePartsById",
        "nodeSpecs",
        "fileTree",
        "llmConfig",
        "runtimeSettings",
        "knowledgeGraph",
        "knowledgeBaseSummary",
        "knowledgeGraphParts",
        "knowledgeGraphBuild",
        "knowledgeGraphLlmConfig",
        "referenceFeatureBuild",
        "referenceFeatureTool",
        "referenceFeatureParts",
        "chainSummary",
        "chainSummaryBuild",
        "chainSummaryParts",
        "dryRun",
        "debugEnabled",
        "runtime",
        "execution",
    }.issubset(payload)
    assert {"running", "nodeRunning", "knowledgeGraphRunning", "chainSummaryRunning", "workspaceUv"}.issubset(payload["runtime"])
    assert payload["variant"]["id"] == "NOD-KGR-KTL-CRV-SUB-ADA"
    assert payload["variant"]["features"] == ["NOD", "KGR", "KTL", "CRV", "SUB", "ADA"]


def test_realtime_sse_endpoint_is_registered(tmp_path, monkeypatch) -> None:
    from harnessing_ts.server import create_app

    WorkspaceStore(tmp_path).ensure_layout()
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(tmp_path))

    app = create_app()
    routes = {getattr(route, "path", "") for route in app.routes}

    assert "/api/events" in routes
    assert "/api/execution/pause" in routes
    assert "/api/execution/force-stop" in routes
    assert "/api/workflow/reset" in routes
    assert "/api/reference-features/status" in routes
    assert "/api/reference-features/tool" in routes
    assert "/api/reference-features/run" in routes


def test_execution_control_rejects_invalid_phase_and_unconfirmed_force_stop(tmp_path, monkeypatch) -> None:
    from harnessing_ts.server import create_app

    WorkspaceStore(tmp_path).ensure_layout()
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(tmp_path))
    app = create_app()

    assert _post(app, "/api/execution/pause", {"reason": "test"}).status_code == 409
    assert _post(app, "/api/execution/force-stop", {"confirm": False}).status_code == 400


def test_live_payload_exposes_the_sse_snapshot_field_set(tmp_path) -> None:
    from harnessing_ts.api.payloads import build_live_payload
    from harnessing_ts.orchestrator import HarnessOrchestrator

    orchestrator = HarnessOrchestrator(tmp_path, dry_run=True, mode="auto")
    orchestrator.initialize()

    payload = build_live_payload(orchestrator=orchestrator, task_running=lambda task: False)

    assert {
        "variant",
        "state",
        "timeline",
        "mainParts",
        "nodes",
        "nodePartsById",
        "runtimeSettings",
        "knowledgeGraph",
        "knowledgeBaseSummary",
        "knowledgeGraphParts",
        "knowledgeGraphBuild",
        "knowledgeGraphLlmConfig",
        "referenceFeatureBuild",
        "referenceFeatureTool",
        "referenceFeatureParts",
        "chainSummary",
        "chainSummaryBuild",
        "chainSummaryParts",
        "runtime",
        "execution",
    } == set(payload)
    assert {"running", "nodeRunning", "knowledgeGraphRunning", "chainSummaryRunning", "pipelineComplete", "workspaceUv"} == set(payload["runtime"])


def test_send_rejects_failed_execution_before_starting_background_task(tmp_path, monkeypatch) -> None:
    from harnessing_ts.server import create_app

    WorkspaceStore(tmp_path).ensure_layout()
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(tmp_path))
    app = create_app()
    orchestrator = app.state.orchestrator

    async def fail_execution() -> None:
        await orchestrator.execution.begin("main", "session-1")
        await orchestrator.execution.fail(RuntimeError("broken checkpoint"))

    asyncio.run(fail_execution())
    response = _post(app, "/api/send", {"text": "continue"})

    assert response.status_code == 409
    assert "execution is failed" in response.json()["detail"]
    assert app.state.run_task is None


def test_pause_background_failure_publishes_failed_snapshot(tmp_path) -> None:
    from harnessing_ts.orchestrator import HarnessOrchestrator
    from harnessing_ts.server import _run_pause_execution

    orchestrator = HarnessOrchestrator(tmp_path, dry_run=True, mode="auto")
    orchestrator.initialize()

    class FailingRunner:
        session_id = "session-1"
        is_running = True

        async def interrupt(self) -> None:
            raise RuntimeError("pause transport failed")

    orchestrator.main_runner = FailingRunner()  # type: ignore[assignment]
    published: list[tuple[str, dict]] = []
    realtime = SimpleNamespace(publish=lambda event, payload: published.append((event, payload)))

    async def scenario() -> None:
        await orchestrator.execution.begin("main", "session-1")
        await _run_pause_execution(orchestrator, realtime, "test")

    asyncio.run(scenario())

    assert orchestrator.execution.snapshot()["phase"] == "failed"
    assert published[-1][0] == "live_snapshot"
    assert published[-1][1]["snapshot"]["execution"]["phase"] == "failed"
