from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from harnessing_ts.orchestrator import HarnessOrchestrator
from harnessing_ts.tools.compose_tools import build_node_native_tools
from harnessing_ts.variants import get_variant


async def _leased_capability(orchestrator, *, role="main", node_id=None, tools=()):
    actor = f"node:{node_id}" if node_id else "main"
    await orchestrator.execution.begin(actor, f"session-{actor}")
    return orchestrator._issue_mcp_capability(
        role=role,
        node_session_id=node_id,
        allowed_tools=list(tools),
    )["DSH_MCP_CAPABILITY"]


def test_dsh_mcp_dispatch_uses_live_orchestrator_state(tmp_path) -> None:
    async def scenario() -> None:
        orchestrator = HarnessOrchestrator(tmp_path, dry_run=True, mode="auto")
        orchestrator.initialize()
        orchestrator.configure_mcp_control("http://127.0.0.1:4327")
        capability = await _leased_capability(
            orchestrator,
            tools=["mcp__ts_harness__enter_node"],
        )
        result = await orchestrator.dispatch_dsh_mcp_call(
            capability=capability,
            workspace=str(tmp_path),
            role="main",
            node_session_id=None,
            tool="enter_node",
            args={"nodeType": "problem-contract", "rationale": "MCP control test"},
        )
        assert result["status"] == "started"
        assert orchestrator.get_state()["activeNodeSessionId"] == result["nodeSessionId"]
    asyncio.run(scenario())


def test_dsh_mcp_capability_is_bound_to_role_tool_epoch_and_lease(tmp_path) -> None:
    async def scenario() -> None:
        orchestrator = HarnessOrchestrator(tmp_path, dry_run=True, mode="auto")
        orchestrator.initialize()
        orchestrator.configure_mcp_control("http://127.0.0.1:4327")
        capability = await _leased_capability(
            orchestrator,
            tools=["mcp__ts_harness__enter_node"],
        )
        with pytest.raises(PermissionError, match="different session"):
            await orchestrator.dispatch_dsh_mcp_call(
                capability=capability, workspace=str(tmp_path), role="node",
                node_session_id="node-1", tool="enter_node", args={},
            )
        with pytest.raises(PermissionError, match="not allowed"):
            await orchestrator.dispatch_dsh_mcp_call(
                capability=capability, workspace=str(tmp_path), role="main",
                node_session_id=None, tool="query_knowledge", args={},
            )
        await orchestrator.execution.turn_finished(checkpoint_seq=4)
        with pytest.raises(PermissionError, match="lease is no longer active"):
            await orchestrator.dispatch_dsh_mcp_call(
                capability=capability, workspace=str(tmp_path), role="main",
                node_session_id=None, tool="enter_node", args={},
            )
    asyncio.run(scenario())


def test_pausing_rejects_new_mcp_calls_without_http_finalization(tmp_path) -> None:
    class Runtime:
        session_id = "session-main"

        def __init__(self) -> None:
            self.release = asyncio.Event()

        async def interrupt(self) -> None:
            await self.release.wait()

        async def force_stop(self) -> None:
            self.release.set()

    async def scenario() -> None:
        orchestrator = HarnessOrchestrator(tmp_path, dry_run=True, mode="auto")
        orchestrator.initialize()
        orchestrator.configure_mcp_control("http://127.0.0.1:4327")
        capability = await _leased_capability(
            orchestrator,
            tools=["mcp__ts_harness__enter_node"],
        )
        runtime = Runtime()
        pause = asyncio.create_task(orchestrator.execution.pause(runtime, "pause"))
        await asyncio.sleep(0)
        assert orchestrator.execution.snapshot()["phase"] == "pausing"
        with pytest.raises(PermissionError, match="lease is no longer active"):
            await orchestrator.dispatch_dsh_mcp_call(
                capability=capability, workspace=str(tmp_path), role="main",
                node_session_id=None, tool="enter_node", args={},
            )
        runtime.release.set()
        assert (await pause)["phase"] == "paused"
    asyncio.run(scenario())


def test_main_async_mcp_result_is_rejected_after_lease_retires(tmp_path) -> None:
    async def scenario() -> None:
        orchestrator = HarnessOrchestrator(tmp_path, dry_run=True, mode="auto")
        orchestrator.initialize()
        orchestrator.configure_mcp_control("http://127.0.0.1:4327")
        capability = await _leased_capability(
            orchestrator,
            tools=["mcp__ts_harness__query_knowledge"],
        )
        release = asyncio.Event()

        async def delayed_query(**kwargs):
            await release.wait()
            kwargs["result_guard"]()
            return {"answer": "late"}

        orchestrator.query_knowledge = delayed_query  # type: ignore[method-assign]
        orchestrator.store.is_knowledge_graph_ready = lambda: True  # type: ignore[method-assign]
        call = asyncio.create_task(orchestrator.dispatch_dsh_mcp_call(
            capability=capability,
            workspace=str(tmp_path),
            role="main",
            node_session_id=None,
            tool="query_knowledge",
            args={"question": "test"},
        ))
        await asyncio.sleep(0)
        await orchestrator.execution.turn_finished(checkpoint_seq=4)
        release.set()
        with pytest.raises(RuntimeError, match="lease is no longer active"):
            await call

    asyncio.run(scenario())


def test_send_without_control_url_fails_before_starting_a_runner(tmp_path) -> None:
    async def scenario() -> None:
        orchestrator = HarnessOrchestrator(tmp_path, dry_run=False, mode="auto")
        orchestrator.initialize()

        class Runner:
            session_id = "session-1"
            config = object()

        orchestrator.main_runner = Runner()  # type: ignore[assignment]
        orchestrator._main_runner_knowledge_graph_ready = False
        orchestrator._main_runner_reference_feature_ready = False

        with pytest.raises(RuntimeError, match="without an MCP control URL"):
            await orchestrator.send_main_user_message("hello")

        assert orchestrator.execution.snapshot()["phase"] == "ready"

    asyncio.run(scenario())


def test_cxr_record_run_dispatch_rejects_reference_feature_artifact_paths(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        monkeypatch.setenv("TS_HARNESS_VARIANT", "NOD-FQA-KTL-CXR-SUB-ADA")
        orchestrator = HarnessOrchestrator(tmp_path, dry_run=True, mode="auto")
        orchestrator.initialize()
        orchestrator.configure_mcp_control("http://127.0.0.1:4327")
        node = orchestrator.store.create_node_session("iterative-solving", "test", None)
        node["status"] = "running"
        orchestrator.store.write_node_session(node)
        state = orchestrator.get_state()
        state["activeNode"] = "iterative-solving"
        state["activeNodeSessionId"] = node["id"]
        orchestrator.store.write_state(state)
        await orchestrator.execution.begin(f"node:{node['id']}", f"session-{node['id']}")
        capability = orchestrator._issue_mcp_capability(
            role="node",
            node_session_id=node["id"],
            allowed_tools=["mcp__ts_harness__record_run"],
        )["DSH_MCP_CAPABILITY"]

        with pytest.raises(RuntimeError, match="forbids executing or recording reference feature"):
            await orchestrator.dispatch_dsh_mcp_call(
                capability=capability,
                workspace=str(tmp_path),
                role="node",
                node_session_id=node["id"],
                tool="record_run",
                args={
                    "runId": "run-001",
                    "status": "completed",
                    "startedAt": "2026-10-18T00:00:00Z",
                    "artifactPaths": ["runs/iterations/001/reference-feature-usage.json"],
                },
            )

    asyncio.run(scenario())


def test_mcp_bridge_credentials_use_dsh_scrubbed_namespace(tmp_path) -> None:
    orchestrator = HarnessOrchestrator(tmp_path, dry_run=True, mode="auto")
    orchestrator.initialize()
    orchestrator.configure_mcp_control("http://127.0.0.1:4327")
    bridge_env = orchestrator._issue_mcp_capability(
        role="main", allowed_tools=["mcp__ts_harness__enter_node"],
    )
    assert set(bridge_env) == {"DSH_MCP_CONTROL_URL", "DSH_MCP_CAPABILITY"}
    cordis_path = Path(__file__).parents[1] / "backend" / "harnessing_ts" / "config" / "deepseek-harness" / "cordis.yml"
    cordis = cordis_path.read_text(encoding="utf-8")
    assert "DSH_MCP_CONTROL_URL: !!js process.env.DSH_MCP_CONTROL_URL" in cordis
    assert "DSH_MCP_CAPABILITY: !!js process.env.DSH_MCP_CAPABILITY" in cordis


def test_subagent_and_web_tool_visibility_follow_variant_and_deployment(monkeypatch) -> None:
    full = get_variant("NOD-KGR-KTL-CRV-SUB-ADA")
    without_subagents = replace(full, independent_subagents=False)
    monkeypatch.delenv("TS_HARNESS_ENABLE_WEB_FETCH", raising=False)
    monkeypatch.delenv("TS_HARNESS_ENABLE_WEB_SEARCH", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    full_tools = build_node_native_tools("iterative-solving", variant=full)
    no_subagent_tools = build_node_native_tools("iterative-solving", variant=without_subagents)
    assert "subagent" in full_tools and "subagent_fork" in full_tools
    assert "subagent" not in no_subagent_tools
    assert "web_fetch" not in full_tools and "web_search" not in full_tools
    monkeypatch.setenv("TS_HARNESS_ENABLE_WEB_FETCH", "true")
    monkeypatch.setenv("TS_HARNESS_ENABLE_WEB_SEARCH", "true")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    assert {"web_fetch", "web_search"}.issubset(build_node_native_tools("iterative-solving", variant=full))
