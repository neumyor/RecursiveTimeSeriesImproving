from __future__ import annotations

import asyncio
import json
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
    monkeypatch.setenv("TS_HARNESS_LEGACY", "true")

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


def test_llm_config_save_keeps_the_effective_api_key(tmp_path, monkeypatch) -> None:
    from harnessing_ts.server import create_app

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "config.llm.json").write_text(
        json.dumps({"authMode": "manual", "model": "repo-model", "apiKey": "sk-repo", "baseUrl": "https://repo.example"}),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(workspace))
    monkeypatch.chdir(repo)

    response = _post(
        create_app(), "/api/llm-config",
        {"authMode": "manual", "model": "edited-model", "baseUrl": "https://edited.example"},
    )

    assert response.status_code == 200
    written = json.loads((workspace / "config.llm.json").read_text(encoding="utf-8"))
    assert written["model"] == "edited-model"
    assert written["apiKey"] == "sk-repo"


def test_masked_llm_config_tolerates_a_manual_config_without_a_key(tmp_path) -> None:
    from harnessing_ts.api.payloads import masked_llm_config

    (tmp_path / "config.llm.json").write_text(
        json.dumps({"authMode": "manual", "model": "model-without-key"}), encoding="utf-8",
    )

    masked = masked_llm_config(tmp_path)

    assert masked["config"]["model"] == "model-without-key"
    assert masked["sdk"]["model"] == "model-without-key"


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
        "chainSummary",
        "chainSummaryBuild",
        "chainSummaryParts",
        "runtime",
        "execution",
    } == set(payload)
    assert {"running", "nodeRunning", "knowledgeGraphRunning", "chainSummaryRunning", "pipelineComplete", "workspaceUv"} == set(payload["runtime"])


def test_clear_logs_rejects_reset_while_node_task_is_active(tmp_path, monkeypatch) -> None:
    from harnessing_ts.server import create_app

    WorkspaceStore(tmp_path).ensure_layout()
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TS_HARNESS_DEBUG", "true")
    app = create_app()
    orchestrator = app.state.orchestrator

    async def request() -> httpx.Response:
        release = asyncio.Event()

        async def node_turn() -> None:
            await release.wait()

        orchestrator._active_node_task = asyncio.create_task(node_turn())
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.post("/api/debug/clear-logs", json={"scope": "chat", "confirmReset": True})
        finally:
            release.set()
            await orchestrator._active_node_task

    response = asyncio.run(request())

    assert response.status_code == 409
    assert "run is active" in response.json()["detail"]


def test_knowledge_graph_build_rejected_when_variant_disables_it(tmp_path, monkeypatch) -> None:
    from harnessing_ts.server import create_app

    WorkspaceStore(tmp_path).ensure_layout()
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("TS_HARNESS_VARIANT", "NOD-FQA-KTL-CXR-SUB-ADA")
    app = create_app()

    response = _post(app, "/api/knowledge-graph/build", {"trigger": "manual"})

    assert response.status_code == 400
    assert "disabled by ablation variant" in response.json()["detail"]
    assert app.state.knowledge_graph_task is None


def test_send_after_failed_execution_starts_fresh_attempt(tmp_path, monkeypatch) -> None:
    from harnessing_ts.server import create_app

    WorkspaceStore(tmp_path).ensure_layout()
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(tmp_path))
    app = create_app()
    orchestrator = app.state.orchestrator

    class Runner:
        session_id = "session-1"

        async def send_with_user_echo(self, _text: str, context_text: str | None = None) -> list[dict]:
            return []

    orchestrator.main_runner = Runner()  # type: ignore[assignment]
    orchestrator._main_runner_knowledge_graph_ready = False
    orchestrator._main_runner_reference_feature_ready = False

    async def fail_previous_turn() -> None:
        await orchestrator.execution.begin("main", "session-1")
        await orchestrator.execution.fail(RuntimeError("broken checkpoint"))

    asyncio.run(fail_previous_turn())
    assert orchestrator.execution.snapshot()["phase"] == "failed"

    async def send_and_settle():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/api/send", json={"text": "try again"})
            if app.state.run_task is not None:
                await app.state.run_task
            return response

    response = asyncio.run(send_and_settle())

    assert response.status_code == 200
    assert response.json()["accepted"] is True
    assert orchestrator.execution.snapshot()["phase"] == "ready"


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
