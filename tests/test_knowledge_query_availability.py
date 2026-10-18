from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from harnessing_ts.orchestrator import HarnessOrchestrator
from harnessing_ts.state.workspace_store import WorkspaceStore
from harnessing_ts.tools.compose_tools import build_main_allowed_tools


def _mark_graph_ready(store: WorkspaceStore) -> None:
    store.write_knowledge_graph_build_status({"status": "completed"})
    manifest = store.root / "knowledge_base" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "schemaVersion": 5,
        "validation": {"ok": True, "errors": [], "warnings": []},
    }), encoding="utf-8")


def test_knowledge_graph_ready_requires_completed_status_and_valid_manifest(tmp_path) -> None:
    store = WorkspaceStore(tmp_path)
    store.ensure_layout()

    assert store.is_knowledge_graph_ready() is False

    store.write_knowledge_graph_build_status({"status": "completed"})
    assert store.is_knowledge_graph_ready() is False

    _mark_graph_ready(store)
    assert store.is_knowledge_graph_ready() is True

    store.write_knowledge_graph_build_status({"status": "failed"})
    assert store.is_knowledge_graph_ready() is False


def test_main_allowlist_only_contains_query_knowledge_when_query_is_ready() -> None:
    unavailable = build_main_allowed_tools(knowledge_query_ready=False)
    available = build_main_allowed_tools(knowledge_query_ready=True)

    assert "mcp__ts_harness__query_knowledge" not in unavailable
    assert "mcp__ts_harness__query_knowledge" in available
    assert "mcp__ts_harness__enter_node" in unavailable


def test_main_runner_tools_exclude_query_before_graph_build(tmp_path) -> None:
    orchestrator = HarnessOrchestrator(tmp_path, dry_run=False)
    orchestrator.initialize()
    fake_runner = MagicMock()

    with patch("harnessing_ts.orchestrator.build_main_runner", return_value=fake_runner) as build:
        orchestrator._ensure_main_runner()

    assert build.call_args.kwargs["knowledge_query_ready"] is False
    assert orchestrator._main_runner_knowledge_graph_ready is False


def test_main_runner_tools_include_query_after_successful_graph_build(tmp_path) -> None:
    orchestrator = HarnessOrchestrator(tmp_path, dry_run=False)
    orchestrator.initialize()
    _mark_graph_ready(orchestrator.store)
    fake_runner = MagicMock()

    with patch("harnessing_ts.orchestrator.build_main_runner", return_value=fake_runner) as build:
        orchestrator._ensure_main_runner()

    assert build.call_args.kwargs["knowledge_query_ready"] is True
    assert orchestrator._main_runner_knowledge_graph_ready is True


def test_rqa_main_runner_tools_include_query_without_graph(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TS_HARNESS_VARIANT", "NOD-RQA-KTL-CRV-SUB-ADA")
    orchestrator = HarnessOrchestrator(tmp_path, dry_run=False)
    orchestrator.initialize()
    fake_runner = MagicMock()

    with patch("harnessing_ts.orchestrator.build_main_runner", return_value=fake_runner) as build:
        orchestrator._ensure_main_runner()

    assert build.call_args.kwargs["knowledge_query_ready"] is True
    assert orchestrator._main_runner_knowledge_graph_ready is True


def test_fqa_node_runner_tools_exclude_query_before_source_ready(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TS_HARNESS_VARIANT", "NOD-FQA-KTL-CXR-SUB-ADA")
    orchestrator = HarnessOrchestrator(tmp_path, dry_run=False)
    orchestrator.initialize()
    node = orchestrator.store.create_node_session("knowledge-to-tools")

    with patch("harnessing_ts.orchestrator.build_node_runner", return_value=MagicMock()) as build:
        orchestrator._spawn_node_runner(node)

    assert build.call_args.kwargs["knowledge_query_ready"] is False


def test_fqa_node_runner_tools_include_query_after_source_ready(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TS_HARNESS_VARIANT", "NOD-FQA-KTL-CXR-SUB-ADA")
    orchestrator = HarnessOrchestrator(tmp_path, dry_run=False)
    orchestrator.initialize()
    node = orchestrator.store.create_node_session("iterative-solving")
    orchestrator.store.write_reference_feature_status({"status": "completed"})

    with patch.object(orchestrator.store, "is_reference_feature_extractor_ready", return_value=True):
        with patch("harnessing_ts.orchestrator.build_node_runner", return_value=MagicMock()) as build:
            orchestrator._spawn_node_runner(node)

    assert build.call_args.kwargs["knowledge_query_ready"] is True


def test_node_runner_resumes_dsh_session_after_restart(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TS_HARNESS_VARIANT", "NOD-FQA-KTL-CXR-SUB-ADA")
    orchestrator = HarnessOrchestrator(tmp_path, dry_run=False)
    orchestrator.initialize()
    node = orchestrator.store.create_node_session("iterative-solving")
    node["sdkSessionId"] = "previous-sdk-session"

    with patch("harnessing_ts.orchestrator.build_node_runner", return_value=MagicMock()) as build:
        orchestrator._spawn_node_runner(node)

    # The local DSH JSONL persistence/checkpoint composition restores the
    # durable session id after a runtime restart.
    assert build.call_args.kwargs["resume_session_id"] == "previous-sdk-session"


def test_query_endpoint_rejects_unbuilt_graph_before_starting_reasoner(tmp_path) -> None:
    orchestrator = HarnessOrchestrator(tmp_path)
    orchestrator.initialize()

    with pytest.raises(RuntimeError, match="Knowledge graph is not ready"):
        asyncio.run(orchestrator.query_knowledge("What patterns matter?"))


def test_existing_main_runner_is_recreated_when_graph_availability_changes(tmp_path) -> None:
    orchestrator = HarnessOrchestrator(tmp_path)
    orchestrator.initialize()
    runner = MagicMock()
    runner.is_running = False
    runner.close = AsyncMock()
    orchestrator.main_runner = runner
    orchestrator._main_runner_knowledge_graph_ready = False
    _mark_graph_ready(orchestrator.store)

    asyncio.run(orchestrator._refresh_main_runner_for_dynamic_tools())

    assert orchestrator.main_runner is None
    assert orchestrator._main_runner_knowledge_graph_ready is None
    runner.close.assert_awaited_once()
    assert orchestrator.store.read_timeline()[-1]["type"] == "main_runner_closed"


def test_knowledge_graph_llm_config_inherits_max_tokens_from_main(tmp_path) -> None:
    orchestrator = HarnessOrchestrator(tmp_path, dry_run=True)
    orchestrator.initialize()
    orchestrator.store.write_main_llm_config({"model": "main-model", "maxTokens": 8000})

    assert orchestrator._knowledge_graph_llm_config().maxTokens == 8000
    assert orchestrator._knowledge_graph_llm_config().model == "main-model"

    orchestrator.store.write_knowledge_graph_llm_config({"model": "graph-model", "maxTokens": 2000})
    graph = orchestrator._knowledge_graph_llm_config()
    assert graph.maxTokens == 2000
    assert graph.model == "graph-model"

    orchestrator.store.write_knowledge_graph_llm_config({"maxTokens": None})
    reverted = orchestrator._knowledge_graph_llm_config()
    assert reverted.maxTokens == 8000
    assert reverted.model == "graph-model"
