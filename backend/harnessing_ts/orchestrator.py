from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from harnessing_ts.agent.sdk_runner import SdkRunner
from harnessing_ts.agent.session_factory import build_main_runner, build_node_runner
from harnessing_ts.agent.translate import public_display_part, system_text_part, user_text_part
from harnessing_ts.chain_summary import build_chain_summary
from harnessing_ts.execution import ExecutionController
from harnessing_ts.knowledge_graph import (
    answer_extractor_source_query,
    answer_knowledge_query,
    answer_reference_query,
    build_knowledge_graph,
    get_neighbors,
    get_supporting_evidence,
    search_evidence_notes,
    search_graph,
    search_knowledge_notes,
    suggest_next_checks,
)
from harnessing_ts.node_state import NodeStateMachine
from harnessing_ts.reference_feature_extractor import (
    execute_reference_feature_extractor,
    inspect_reference_feature_extractor,
    validate_reference_feature_extractor,
)
from harnessing_ts.prompts.compose import PromptContext, build_main_attachment
from harnessing_ts.schema import (
    NODE_SPECS,
    ControlRequest,
    NodeSession,
    NodeType,
    Part,
    RunRecord,
    WorkspaceState,
)
from harnessing_ts.settings.llm import LlmConfig, mask_llm_config, read_effective_llm_config
from harnessing_ts.state.jsonl import clear_file
from harnessing_ts.state.workspace_store import WorkspaceStore, now_iso
from harnessing_ts.tools.compose_tools import build_node_native_tools
from harnessing_ts.variants import resolve_variant


def _main_node_snapshot(node: NodeSession | None) -> dict[str, Any] | None:
    if not node:
        return None
    return {
        "id": node.get("id"),
        "nodeType": node.get("nodeType"),
        "status": node.get("status"),
        "success": node.get("success"),
        "goalMet": node.get("goalMet"),
        "nextNode": node.get("nextNode"),
        "nextNodeSpecified": bool(node.get("nextNodeSpecified", False)),
        "loopDecision": node.get("loopDecision"),
        "summary": node.get("summary"),
        "outputPaths": list(node.get("outputPaths", [])),
    }


def _main_control_snapshot(control: ControlRequest | None) -> dict[str, Any] | None:
    if not control:
        return None
    return {
        "id": control.get("id"),
        "kind": control.get("kind"),
        "status": control.get("status"),
        "nodeType": control.get("nodeType"),
        "nodeSessionId": control.get("nodeSessionId"),
    }


class HarnessOrchestrator:
    def __init__(self, workspace_path: Path, locale: str = "zh", mode: str = "manual", dry_run: bool = False) -> None:
        self.workspace_path = workspace_path
        self.locale = locale
        self.mode = mode
        self.dry_run = dry_run
        self.variant = resolve_variant()
        self.store = WorkspaceStore(workspace_path)
        self.execution = ExecutionController(self.store)
        self.node_state = NodeStateMachine(workspace_path, self.variant)
        self.state: WorkspaceState | None = None
        self.main_runner: SdkRunner | None = None
        self._main_runner_knowledge_graph_ready: bool | None = None
        self._main_runner_reference_feature_ready: bool | None = None
        self.active_node_runner: SdkRunner | None = None
        self.active_node_session: NodeSession | None = None
        self._active_node_task: asyncio.Task[None] | None = None
        self._realtime_event_sink: Callable[[str, dict[str, Any]], None] | None = None
        self._runtime_snapshot_provider: Callable[[], dict[str, Any]] | None = None
        # A part is re-emitted on SSE only when its public payload differs
        # from the last emission for the same part id, e.g. when a merged
        # tool result updates an existing tool card.
        self._last_emitted_parts: dict[str, Part] = {}
        self._mcp_control_url: str | None = None
        self._mcp_capabilities: dict[str, dict[str, Any]] = {}

    def initialize(self, reporter: Callable[[str], None] | None = None) -> WorkspaceState:
        self.state = self.store.initialize(self.mode, reporter=reporter)
        self._stamp_variant_state()
        return self.state

    def set_realtime_event_sink(self, sink: Callable[[str, dict[str, Any]], None] | None) -> None:
        self._realtime_event_sink = sink

    def set_runtime_snapshot_provider(self, provider: Callable[[], dict[str, Any]] | None) -> None:
        self._runtime_snapshot_provider = provider

    def is_active_node_task_running(self) -> bool:
        task = self._active_node_task
        return task is not None and not task.done()

    def active_node_task(self) -> asyncio.Task[None] | None:
        return self._active_node_task

    def _emit_runtime_snapshot(self) -> None:
        if self._runtime_snapshot_provider:
            self._emit_realtime("live_snapshot", {"snapshot": self._runtime_snapshot_provider()})

    def recover_interrupted_execution(self) -> bool:
        """Fence a turn abandoned by a dead server; DSH repairs it on resume."""
        self._ensure_initialized()
        assert self.state
        changed = self.execution.recover_interrupted()
        node_id = self.state.get("activeNodeSessionId")
        node = self.store.read_node_session(node_id) if node_id else None
        if self.store.read_execution() is None and node and node.get("status") == "running":
            raise RuntimeError(
                "Workspace has a running node but no state/execution.json; its pause "
                "boundary cannot be reconstructed. Reset the workflow before starting again."
            )
        if changed and node and node.get("status") == "running":
            node.update({"status": "paused", "pausedAt": now_iso(), "summary": "Paused after server restart; DSH will repair and resume the same session."})
            self.store.write_node_session(node)
            self.active_node_session = node
        return changed

    async def complete_interrupted_recovery(self) -> None:
        snapshot = self.execution.snapshot()
        if snapshot["phase"] != "recovering":
            return
        actor_id = str(snapshot.get("actorId") or "")
        try:
            if actor_id == "main":
                self._ensure_main_runner()
                runner = self.main_runner
            elif actor_id.startswith("node:"):
                node_id = actor_id.split(":", 1)[1]
                node = self.store.read_node_session(node_id)
                if not node:
                    raise RuntimeError(f"Cannot recover missing node actor {actor_id}")
                self.active_node_session = node
                self._spawn_node_runner(node)
                runner = self.active_node_runner
            else:
                raise RuntimeError(f"Unknown recovery actor {actor_id}")
            if runner is None:
                raise RuntimeError(f"No runner available for recovery actor {actor_id}")
            session_id = snapshot.get("sessionId")
            if not isinstance(session_id, str) or not session_id:
                raise RuntimeError(f"Recovery actor {actor_id} has no durable session identity")
            runner.adopt_recovery_session(session_id)
            await runner.recover()
            await self.execution.recovery_finished(checkpoint_seq=runner.checkpoint_seq)
        except Exception as exc:
            await self.execution.fail(exc)
        self._emit_runtime_snapshot()

    def configure_mcp_control(self, control_url: str) -> None:
        """Route DSH stdio MCP calls back to this live orchestrator instance.

        The MCP server lives in a DSH-owned child process.  Giving it a scoped,
        unguessable capability avoids reconstructing another orchestrator (and
        therefore losing this process's state and realtime event sink).  The
        bridge values use DSH_MCP_* names so DSH's subprocess seam scrubs them
        from model-executed shell commands; Cordis forwards them explicitly to
        the MCP process only.
        """
        self._mcp_control_url = control_url.rstrip("/")

    def _require_mcp_control_url(self) -> None:
        if not self._mcp_control_url:
            raise RuntimeError(
                "Cannot start a DSH runner without an MCP control URL: its MCP tool calls "
                "would have no route back to the orchestrator. Run the harness through "
                "ts-harness-server, or use --dry-run for state-only operation."
            )

    def _issue_mcp_capability(
        self,
        *,
        role: str,
        allowed_tools: list[str],
        node_session_id: str | None = None,
    ) -> dict[str, str]:
        self._require_mcp_control_url()
        token = secrets.token_urlsafe(32)
        self._mcp_capabilities[token] = {
            "workspace": str(self.workspace_path.resolve()),
            "role": role,
            "nodeSessionId": node_session_id or "",
            "allowedTools": frozenset(allowed_tools),
            "epoch": self.execution.snapshot().get("epoch"),
            "leaseId": self.execution.snapshot().get("leaseId"),
        }
        return {
            "DSH_MCP_CONTROL_URL": self._mcp_control_url,
            "DSH_MCP_CAPABILITY": token,
        }

    def _rotate_runner_mcp_capability(self, runner: SdkRunner) -> None:
        self._revoke_runner_mcp_capability(runner)
        bridge_env = self._issue_mcp_capability(
            role=runner.config.session_role,
            allowed_tools=runner.config.allowed_tools,
            node_session_id=runner.config.node_session_id,
        )
        runner.config.env = {**(runner.config.env or {}), **bridge_env}

    def _revoke_runner_mcp_capability(self, runner: SdkRunner) -> None:
        config = getattr(runner, "config", None)
        token = (getattr(config, "env", None) or {}).get("DSH_MCP_CAPABILITY")
        if token:
            self._mcp_capabilities.pop(token, None)

    def _mcp_call_is_current(self, capability: str, node_session_id: str | None = None) -> bool:
        """Return whether an MCP call still belongs to its leased runner.

        A capability is revoked as soon as its runner is closed.  Checking it
        again after an awaited operation prevents an old MCP request from
        publishing a late result into a paused node or a newer runner for the
        same node session.
        """
        record = self._mcp_capabilities.get(capability)
        if not record or record.get("nodeSessionId") != (node_session_id or ""):
            return False
        execution = self.execution.snapshot()
        leased = bool(
            execution.get("phase") == "running"
            and record.get("epoch") == execution.get("epoch")
            and record.get("leaseId") == execution.get("leaseId")
        )
        if not leased or not node_session_id:
            return leased
        state = self.store.read_state() or self.state
        node = self.store.read_node_session(node_session_id)
        return bool(
            state
            and state.get("activeNodeSessionId") == node_session_id
            and node
            and node.get("status") == "running"
        )

    def _require_current_mcp_call(self, capability: str, node_session_id: str | None = None) -> None:
        if not self._mcp_call_is_current(capability, node_session_id):
            raise RuntimeError("This MCP call was cancelled because its runner lease is no longer active.")

    async def dispatch_dsh_mcp_call(
        self,
        *,
        capability: str,
        workspace: str,
        role: str,
        node_session_id: str | None,
        tool: str,
        args: dict[str, Any],
    ) -> Any:
        """Execute a bridged MCP call on the live state owner.

        This is intentionally not a public API: ``server.py`` binds it to
        loopback only and validates the per-runner capability before dispatch.
        """
        record = self._mcp_capabilities.get(capability)
        if not record or str(record["workspace"]) != str(Path(workspace).resolve()):
            raise PermissionError("Invalid DeepSeek Harness MCP capability.")
        if record["role"] != role or record["nodeSessionId"] != (node_session_id or ""):
            raise PermissionError("DeepSeek Harness MCP capability is bound to a different session.")
        if not self._mcp_call_is_current(capability, node_session_id):
            raise PermissionError("DeepSeek Harness MCP capability lease is no longer active.")
        qualified_tool = f"mcp__ts_harness__{tool}"
        if qualified_tool not in record["allowedTools"]:
            raise PermissionError(f"DeepSeek Harness MCP tool is not allowed: {tool}")
        if tool == "enter_node":
            return await self.request_enter_node(args)
        if tool == "query_knowledge":
            return await self.request_query_knowledge(
                args,
                node_session_id=node_session_id,
                capability=capability,
            )
        if tool == "extract_reference_features":
            return self.request_extract_reference_features(args)
        if tool == "inspect_reference_feature_extractor":
            return self.request_inspect_reference_feature_extractor(args)
        if tool == "validate_reference_feature_extractor":
            return await self.request_validate_reference_feature_extractor(args)
        if tool == "finish_node":
            return await self.request_finish_node(args)
        if tool in {"record_artifact", "record_run"}:
            node = self.store.read_node_session(node_session_id)
            if not node:
                raise RuntimeError("Unknown active node session.")
            if tool == "record_artifact":
                return self.record_artifact_for_node(args, node_session_id, node["nodeType"])
            return self.record_run_for_node(args, node_session_id, node["nodeType"])
        if tool == "get_runtime_settings":
            return self.get_runtime_settings()
        raise RuntimeError(f"Unknown DeepSeek Harness MCP tool: {tool}")

    def _emit_realtime(self, event_type: str, payload: dict[str, Any]) -> None:
        if self._realtime_event_sink:
            self._realtime_event_sink(event_type, payload)

    def _emit_main_parts(self, _part: Part) -> None:
        self._emit_part_if_changed("main_parts", "mainPart", public_display_part(_part))

    def _emit_knowledge_graph_parts(self, _part: Part) -> None:
        self._emit_part_if_changed("knowledge_graph_parts", "knowledgeGraphPart", public_display_part(_part))

    def _emit_knowledge_graph_snapshot(self) -> None:
        self._emit_realtime("knowledge_graph_snapshot", {
            "knowledgeGraph": self.get_knowledge_graph(),
            "knowledgeBaseSummary": self.get_knowledge_base_summary(),
            "knowledgeGraphBuild": self.get_knowledge_graph_build_status(),
        })

    def _emit_node_parts(self, node_id: str, _part: Part) -> None:
        part = public_display_part(_part)
        cache_key = f"node_parts:{node_id}:{part.get('id')}"
        if self._last_emitted_parts.get(cache_key) == part:
            return
        self._last_emitted_parts[cache_key] = part
        self._emit_realtime("node_parts", {
            "nodePartById": {node_id: part},
            "nodes": self.get_node_sessions(),
            "state": self.get_state(),
            "timeline": self.get_timeline(),
        })

    def _emit_chain_summary_parts(self, _part: Part) -> None:
        self._emit_part_if_changed("chain_summary_parts", "chainSummaryPart", public_display_part(_part))

    def _emit_part_if_changed(self, event_type: str, payload_key: str, part: Part) -> None:
        cache_key = f"{event_type}:{part.get('id')}"
        if self._last_emitted_parts.get(cache_key) == part:
            return
        self._last_emitted_parts[cache_key] = part
        self._emit_realtime(event_type, {payload_key: part})

    async def send_main_user_message(self, text: str) -> list[Part]:
        self._ensure_initialized()
        assert self.state
        if self.state["activeNode"]:
            node = self._restore_active_node_session()
            if node and node.get("status") == "paused":
                return await self._resume_active_node(text, node)
            raise RuntimeError(f"Main session is locked while node {self.state['activeNode']} is active.")
        self.store.append_timeline({"type": "user_message", "timestamp": now_iso(), "message": text[:500]})
        if self.dry_run:
            user_part = user_text_part(text)
            self.store.append_main_part(user_part)
            dry_run_lines = ["Dry run mode: main session was not sent to the DeepSeek Harness runtime."]
            if self.variant.node_chain:
                dry_run_lines.append("Use `start-node <nodeType>` to exercise harness state transitions without SDK.")
            else:
                dry_run_lines.append(f"{self.variant.id} direct tool use requires a live SDK session.")
            part = system_text_part("\n".join(dry_run_lines))
            self.store.append_main_part(part)
            return [user_part, part]
        await self._refresh_main_runner_for_dynamic_tools()
        self._ensure_main_runner()
        assert self.main_runner
        main_attachment = build_main_attachment(
            PromptContext(str(self.workspace_path), self.locale),
            self._main_progress_snapshot(),
            self.variant,
        )
        parts = await self._send_execution_turn(
            self.main_runner,
            "main",
            text,
            context_text=main_attachment,
        )
        return parts

    async def _send_execution_turn(
        self,
        runner: SdkRunner,
        actor_id: str,
        text: str,
        *,
        context_text: str | None = None,
    ) -> list[Part]:
        session_id = getattr(runner, "session_id", None) or f"ts-harness-{uuid4().hex}"
        setattr(runner, "session_id", session_id)
        if getattr(runner, "config", None) is not None:
            self._require_mcp_control_url()
        await self.execution.begin(actor_id, session_id)
        if getattr(runner, "config", None) is not None:
            self._rotate_runner_mcp_capability(runner)
        self._emit_runtime_snapshot()
        try:
            parts = await runner.send_with_user_echo(text, context_text=context_text)
        except BaseException as exc:
            phase = self.execution.snapshot()["phase"]
            if phase not in {"pausing", "paused", "recovering"}:
                await self.execution.fail(exc)
            if getattr(runner, "config", None) is not None:
                self._revoke_runner_mcp_capability(runner)
            self._emit_runtime_snapshot()
            raise
        await self.execution.turn_finished(checkpoint_seq=getattr(runner, "checkpoint_seq", None))
        if getattr(runner, "config", None) is not None:
            self._revoke_runner_mcp_capability(runner)
        self._emit_runtime_snapshot()
        return parts

    async def request_enter_node(self, args: dict[str, Any]) -> dict[str, Any]:
        self._ensure_initialized()
        assert self.state
        if not self.variant.node_chain:
            raise RuntimeError(f"{self.variant.id} disables the HarnessingTS node chain; solve directly in the main session.")
        self._validate_failed_node_retry_request(args)
        if self._control_mode() == "manual":
            return self._park_control_request("enter_node", args)
        active_id = self.state.get("activeNodeSessionId")
        if active_id:
            active = self.store.read_node_session(active_id)
            if active and active.get("nodeType") == args.get("nodeType") and active.get("status") in {"running", "paused", "waiting_approval"}:
                return {
                    "status": "already_started",
                    "nodeSessionId": active["id"],
                    "nodeType": active["nodeType"],
                    "nodeStatus": active.get("status"),
                }
        node = await self._start_node(args, background=True)
        return {"status": "started", "nodeSessionId": node["id"], "nodeType": node["nodeType"]}

    async def request_finish_node(self, args: dict[str, Any]) -> dict[str, Any]:
        self._ensure_initialized()
        assert self.state
        node = self.active_node_session or self._restore_active_node_session()
        if self._control_mode() == "manual":
            result = self._park_control_request("finish_node", args, node)
            if node:
                node["status"] = "waiting_approval"
                node["summary"] = "Waiting for human approval to finish this node."
                self.store.write_node_session(node)
            return result
        return await self.finish_node(args)

    async def enter_node(self, args: dict[str, Any], *, background: bool = False) -> NodeSession:
        self._ensure_initialized()
        assert self.state
        if not self.variant.node_chain:
            raise RuntimeError(f"{self.variant.id} disables node entry.")
        self._validate_failed_node_retry_request(args)
        return await self._start_node(args, background=background)

    async def _start_node(self, args: dict[str, Any], *, background: bool = False) -> NodeSession:
        assert self.state
        node_type = args["nodeType"]
        if node_type == "knowledge-to-tools" and not self.variant.knowledge_to_tools:
            raise RuntimeError(
                f"{self.variant.id} disables the knowledge-to-tools node; the reference feature "
                f"extractor is not built. Continue with iterative-solving directly."
            )
        if self.state["activeNode"]:
            raise RuntimeError(f"Cannot enter {node_type}; active node {self.state['activeNode']} is still running.")
        # Defensive: once the pipeline is fully complete (final-summary
        # finished, no next node), refuse to re-enter any node. A stale
        # enter_node from the original task would otherwise respawn
        # a runner for an already-completed node and have the agent
        # re-do work in the chat (e.g. "now writing problem-contract
        # artifacts" after final-summary is done). The user must
        # reset the workspace before re-running.
        if self._is_pipeline_complete():
            raise RuntimeError(
                f"Cannot enter {node_type}: pipeline is already complete "
                f"(completedNodes={self.state['completedNodes']}). "
                f"Use the Reset Workspace action to clear state before re-running."
            )
        if node_type == "final-summary" and self._read_iteration_state_recommend_exit() is False:
            raise RuntimeError("Cannot enter final-summary while user/iteration-state.md has recommend_exit: false.")
        node = self.store.create_node_session(node_type, args.get("rationale"), args.get("inputSummary"))
        node["status"] = "running"
        self.store.write_node_session(node)
        self.state["activeNode"] = node_type
        self.state["activeNodeSessionId"] = node["id"]
        self.store.write_state(self.state)
        self.store.append_timeline({
            "type": "node_entered",
            "timestamp": now_iso(),
            "nodeSessionId": node["id"],
            "nodeType": node_type,
            "message": args.get("rationale"),
            "payload": {"inputSummary": args.get("inputSummary")},
        })
        self.active_node_session = node
        if self.dry_run:
            self.store.append_node_part(node["id"], system_text_part(f"Dry run entered node {node_type}."))
            return node
        self._spawn_node_runner(node)
        assert self.active_node_runner
        prompt = f"请执行 {node_type} node。"
        if args.get("inputSummary"):
            prompt += f"\n\n用户/主会话补充上下文：\n{args['inputSummary']}"
        if background:
            self._start_node_turn(node, prompt)
            return node
        try:
            node_parts = await self._send_execution_turn(self.active_node_runner, f"node:{node['id']}", prompt)
        except BaseException as exc:
            # The SDK call crashed before the node agent could reach
            # `finish_node`. Release the active-node lock, mark the
            # session as failed with the SDK error in the summary, and
            # re-raise so the HTTP caller still observes the failure.
            await self._handle_node_runner_return(node, sdk_error=exc)
            raise
        await self._handle_node_runner_return(node)
        return node

    def _start_node_turn(self, node: NodeSession, prompt: str) -> None:
        if self.is_active_node_task_running():
            raise RuntimeError("Cannot start a node runner while another node task is still active.")
        self._active_node_task = asyncio.create_task(self._run_node_turn_in_background(node, prompt))
        self._emit_runtime_snapshot()

    async def _run_node_turn_in_background(self, node: NodeSession, prompt: str) -> None:
        """Run a node outside the MCP control response's lifetime."""
        runner = self.active_node_runner
        should_auto_advance = False
        try:
            if runner is None:
                raise RuntimeError("Node runner was not initialized.")
            actor_id = f"node:{node['id']}"
            while True:
                execution = self.execution.snapshot()
                if execution["phase"] in {"paused", "failed"} and execution.get("actorId") != actor_id:
                    return
                if execution["phase"] not in {"running", "pausing", "recovering"}:
                    break
                if execution.get("actorId") == actor_id:
                    break
                await asyncio.sleep(0.01)
            await self._send_execution_turn(runner, f"node:{node['id']}", prompt)
        except BaseException as exc:
            await self._handle_node_runner_return(node, sdk_error=exc, auto_advance=False)
        else:
            await self._handle_node_runner_return(node, auto_advance=False)
            latest = self.store.read_node_session(node["id"]) or node
            should_auto_advance = latest.get("status") == "completed" and latest.get("success") is True
        finally:
            # interrupt_current() owns shutdown when it pauses a node.  Do
            # not close the captured runner a second time after its delayed
            # transport error has been reconciled.
            if runner is not None and self.active_node_runner is runner:
                await self._close_active_node_runner("node_turn_finished")
            current = asyncio.current_task()
            if self._active_node_task is current:
                self._active_node_task = None
        if should_auto_advance:
            latest = self.store.read_node_session(node["id"]) or node
            await self._maybe_auto_enter_next_node(latest)
        self._emit_runtime_snapshot()

    async def finish_node(self, args: dict[str, Any]) -> dict[str, Any]:
        self._ensure_initialized()
        assert self.state
        if not self.active_node_session:
            self.active_node_session = self._restore_active_node_session()
        if not self.active_node_session:
            raise RuntimeError("No active node session to finish.")
        node = self.active_node_session
        success = args.get("success", True)
        goal_met = args.get("goalMet")
        next_node_specified = self._is_next_node_specified(args)
        next_node = self._normalize_next_node(args.get("nextNode"))
        loop_decision = self._normalize_loop_decision(args.get("loopDecision"))
        output_paths = list(args.get("outputPaths") or [])
        self._validate_finish_control(
            node["nodeType"],
            success,
            next_node,
            loop_decision,
            output_paths,
            goal_met,
            next_node_specified,
        )

        node["status"] = "completed" if success else "failed"
        node["completedAt"] = now_iso()
        node["summary"] = args.get("summary", "")
        node["success"] = success
        node["goalMet"] = goal_met
        node["nextNode"] = next_node
        node["nextNodeSpecified"] = next_node_specified
        node["loopDecision"] = loop_decision
        node["outputPaths"] = output_paths
        self.store.write_node_session(node)

        if node["success"] and node["nodeType"] not in self.state["completedNodes"]:
            self.state["completedNodes"].append(node["nodeType"])
        self.state["activeNode"] = None
        self.state["activeNodeSessionId"] = None
        if node["nodeType"] == "problem-contract" and node["success"]:
            self.state["contractReady"] = True
        self.store.write_state(self.state)
        self.store.append_timeline({
            "type": "node_finished",
            "timestamp": node["completedAt"],
            "nodeSessionId": node["id"],
            "nodeType": node["nodeType"],
            "message": node.get("summary"),
            "payload": {
                "success": node["success"],
                "goalMet": args.get("goalMet"),
                "nextNode": node.get("nextNode"),
                "loopDecision": node.get("loopDecision"),
                "outputPaths": node["outputPaths"],
            },
        })
        # Keep the runner reference until its DSH turn reaches checkpoint so a
        # concurrent pause can still address the correct actor. The workflow
        # session itself is terminal and must no longer receive node-scoped
        # application writes.
        self.active_node_session = None
        next_node = self._next_node_after_completion(node) if node["success"] else None
        return {"nodeSessionId": node["id"], "nextNode": next_node}

    def record_artifact_for_node(self, args: dict[str, Any], node_id: str, node_type: NodeType) -> dict[str, bool]:
        self._ensure_initialized()
        self.node_state.reject_reference_feature_paths(node_type, [args.get("path")])
        self.store.record_artifact(args["path"], node_id, node_type, args.get("summary"))
        return {"ok": True}

    def record_run_for_node(self, args: RunRecord, node_id: str, node_type: NodeType) -> dict[str, bool]:
        self._ensure_initialized()
        self.node_state.reject_reference_feature_paths(node_type, [args.get("runId"), *(args.get("artifactPaths") or [])])
        record = dict(args)
        record["nodeSessionId"] = node_id
        record["nodeType"] = node_type
        self.store.record_run(record)
        return {"ok": True}

    async def request_query_knowledge(
        self,
        args: dict[str, Any],
        *,
        node_session_id: str | None = None,
        capability: str | None = None,
    ) -> dict[str, Any]:
        result_guard: Callable[[], None] | None = None
        if capability:
            result_guard = lambda: self._require_current_mcp_call(capability, node_session_id)
        elif node_session_id:
            raise PermissionError("Node knowledge queries require an active MCP capability.")
        return await self.query_knowledge(
            question=str(args.get("question", "")),
            domain=args.get("domain"),
            context=args.get("context") if isinstance(args.get("context"), dict) else None,
            observations=args.get("observations") if isinstance(args.get("observations"), list) else None,
            include_evidence=bool(args.get("includeEvidence", False)),
            result_guard=result_guard,
        )

    def request_extract_reference_features(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self.variant.reference_feature_extractor:
            raise RuntimeError(f"Reference feature extraction is disabled by ablation variant {self.variant.id}.")
        if not self.store.is_reference_feature_extractor_ready():
            raise RuntimeError("Reference feature extractor is not ready. Build and validate it first.")
        output = execute_reference_feature_extractor(self.workspace_path, args.get("input"))
        self.store.append_timeline({
            "type": "reference_features_extracted",
            "timestamp": now_iso(),
            "message": f"features={len(output.get('features', []))}",
        })
        return output

    def request_inspect_reference_feature_extractor(self, _args: dict[str, Any]) -> dict[str, Any]:
        return inspect_reference_feature_extractor(self.workspace_path)

    async def request_validate_reference_feature_extractor(self, args: dict[str, Any]) -> dict[str, Any]:
        """MCP tool used by the main session during the
        `knowledge-to-tools` node to have the backend perform the
        deterministic validation pass and publish a ready extractor
        for downstream nodes.
        """
        allow_execution = self.variant.reference_feature_execution
        run_tests = bool(args.get("runTests", True)) and allow_execution
        status = self.store.validate_and_store_reference_feature(
            run_tests=run_tests,
            allow_execution=allow_execution,
        )
        await_refresh = self.main_runner is not None and not self.state.get("activeNode")
        # The new tool may have just been validated; refresh the main
        # runner when this is called from the main session. During a node
        # MCP call, closing the main runner can cancel the SDK control
        # request that is still returning the tool result. The next main
        # turn refreshes dynamic tools before sending anyway.
        if await_refresh:
            await self._refresh_main_runner_for_dynamic_tools()
        return status

    def _preserve_active_node_lock(self, node: NodeSession) -> None:
        """Persist the lock that makes a paused node resumable after restart."""
        state = self.store.read_state() or self.state
        if state is None:
            raise RuntimeError("Workspace state is not initialized.")
        state["activeNode"] = node["nodeType"]
        state["activeNodeSessionId"] = node["id"]
        self.store.write_state(state)
        self.state = state

    async def interrupt_current(
        self,
        reason: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        assert self.state
        current = self.execution.snapshot()
        if current["phase"] == "paused":
            return {"target": current["actorId"], "execution": current, "alreadyPaused": True}
        if current["phase"] != "running":
            raise RuntimeError(f"execution is {current['phase']}; there is no running turn to pause")
        actor_id = str(current.get("actorId") or "")
        runner = self.active_node_runner if actor_id.startswith("node:") else self.main_runner
        if runner is None:
            raise RuntimeError(f"No DSH runtime is available for active actor {actor_id or 'unknown'}.")
        target = "node" if actor_id.startswith("node:") else "main"
        snapshot = await self.execution.pause(runner, reason or "Paused by user.")
        node_id = actor_id.split(":", 1)[1] if actor_id.startswith("node:") else None
        state = self.store.read_state() or self.state
        if node_id and state.get("activeNodeSessionId") == node_id:
            node = self.store.read_node_session(node_id)
            if node and node.get("status") not in {"completed", "failed", "exited"}:
                node.update({"status": "paused", "pausedAt": now_iso(), "summary": "Paused by user."})
                self.store.write_node_session(node)
                self.active_node_session = node
        self._emit_runtime_snapshot()
        return {"target": target, "execution": snapshot, "state": self.get_state()}

    async def force_stop_current(self) -> dict[str, Any]:
        current = self.execution.snapshot()
        actor_id = str(current.get("actorId") or "")
        runner = self.active_node_runner if actor_id.startswith("node:") else self.main_runner
        if runner is None:
            raise RuntimeError(f"No runtime is available to force-stop actor {actor_id or 'unknown'}.")
        snapshot = await self.execution.force_stop(runner)
        self._emit_runtime_snapshot()
        return {"execution": snapshot, "state": self.get_state()}

    async def _resume_active_node(self, text: str, node: NodeSession) -> list[Part]:
        assert self.state
        node["status"] = "running"
        node.pop("pausedAt", None)
        node["summary"] = "Resumed with user guidance."
        self.store.write_node_session(node)
        self.active_node_session = node
        self.store.append_timeline({
            "type": "node_resumed",
            "timestamp": now_iso(),
            "nodeSessionId": node["id"],
            "nodeType": node["nodeType"],
            "message": text[:500],
        })
        if self.dry_run:
            part = system_text_part("Dry run mode: paused node received resume guidance.")
            self.store.append_node_part(node["id"], part)
            return [part]
        self._spawn_node_runner(node)
        assert self.active_node_runner
        prompt = "\n".join([
            f"请继续执行 {node['nodeType']} node。此前节点被用户暂时中断，现在用户补充了以下说明：",
            text,
            "",
            "请结合已有 workspace 文件、节点日志和这条补充说明继续推进当前 node。不要重新开始整个 pipeline，除非用户明确要求。",
        ])
        self._start_node_turn(node, prompt)
        return []

    async def approve_pending_control(self) -> dict[str, Any]:
        self._ensure_initialized()
        assert self.state
        request = self.store.clear_pending_control()
        self.state = self.store.read_state()
        if not request:
            return {"approved": False, "message": "No pending control request.", "state": self.get_state()}
        self.store.append_timeline({
            "type": "control_approved",
            "timestamp": now_iso(),
            "nodeSessionId": request.get("nodeSessionId"),
            "nodeType": request.get("nodeType"),
            "message": request.get("message") or request["kind"],
            "payload": request,
        })
        if request["kind"] == "enter_node":
            result = await self.enter_node(request["args"], background=True)
            return {"approved": True, "result": result, "state": self.get_state()}
        if request["kind"] == "finish_node":
            result = await self.finish_node(request["args"])
            if result.get("nextNode"):
                self._park_control_request("enter_node", {
                    "nodeType": result["nextNode"],
                    "rationale": f"Previous node finished and pipeline is ready for {result['nextNode']}.",
                    "inputSummary": "Approve this control request to continue the node chain.",
                })
            return {"approved": True, "result": result, "state": self.get_state()}
        raise RuntimeError(f"Unknown control request kind: {request['kind']}")

    def reject_pending_control(self, reason: str | None = None) -> dict[str, Any]:
        self._ensure_initialized()
        assert self.state
        request = self.store.clear_pending_control()
        self.state = self.store.read_state()
        if not request:
            return {"rejected": False, "message": "No pending control request.", "state": self.get_state()}
        message = reason or "Rejected by human."
        self.store.append_timeline({
            "type": "control_rejected",
            "timestamp": now_iso(),
            "nodeSessionId": request.get("nodeSessionId"),
            "nodeType": request.get("nodeType"),
            "message": message,
            "payload": request,
        })
        if request["kind"] == "finish_node" and request.get("nodeSessionId"):
            node = self.store.read_node_session(request["nodeSessionId"])
            if node and node.get("status") == "waiting_approval":
                node["status"] = "paused"
                node["summary"] = f"Finish request rejected: {message}"
                self.store.write_node_session(node)
                self.store.append_node_part(node["id"], system_text_part(f"Harness control rejected finish_node: {message}"))
        return {"rejected": True, "state": self.get_state()}

    def get_state(self) -> WorkspaceState:
        self._ensure_initialized()
        disk_state = self.store.read_state()
        if disk_state:
            self.state = disk_state
        assert self.state
        return self.state

    def get_timeline(self, limit: int | None = 500) -> list[dict[str, Any]]:
        self._ensure_initialized()
        return self.store.read_timeline(limit=limit)

    def get_main_parts(self, limit: int | None = 200) -> list[Part]:
        self._ensure_initialized()
        return self.store.read_main_parts(limit=limit)

    def get_node_sessions(self) -> list[NodeSession]:
        self._ensure_initialized()
        return self.store.list_node_sessions()

    def get_node_parts(self, node_session_id: str, limit: int | None = 200) -> list[Part]:
        self._ensure_initialized()
        return self.store.read_node_parts(node_session_id, limit=limit)

    def get_node_parts_by_id(self, limit_per_node: int = 200) -> dict[str, list[Part]]:
        self._ensure_initialized()
        out: dict[str, list[Part]] = {}
        for node in self.get_node_sessions():
            out[node["id"]] = self.store.read_node_parts(node["id"], limit=limit_per_node)
        return out

    def get_file_tree(self) -> dict[str, Any]:
        self._ensure_initialized()
        return self.store.list_file_tree()

    def get_runtime_status(self) -> dict[str, Any] | None:
        self._ensure_initialized()
        return self.store.read_runtime_status()

    def get_variant(self) -> dict[str, Any]:
        return self.variant.public_payload()

    def get_runtime_settings(self) -> dict[str, Any]:
        self._ensure_initialized()
        return self.store.read_runtime_settings()

    def update_runtime_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        self._ensure_initialized()
        updated = self.store.write_runtime_settings(settings)
        self.state = self.store.read_state()
        return updated

    def get_knowledge_graph(self) -> dict[str, Any]:
        self._ensure_initialized()
        if not self.variant.knowledge_graph:
            return {}
        return self.store.read_knowledge_graph()

    def get_knowledge_base_summary(self) -> dict[str, Any]:
        self._ensure_initialized()
        if not self.variant.knowledge_graph:
            return {}
        return self.store.read_knowledge_base_summary()

    def get_knowledge_base_cards(self, kind: str, limit: int = 200) -> dict[str, Any]:
        self._ensure_initialized()
        self._require_knowledge_variant()
        return self.store.read_knowledge_base_cards(kind, limit)

    def get_knowledge_graph_build_status(self) -> dict[str, Any]:
        self._ensure_initialized()
        return self.store.read_knowledge_graph_build_status()

    def get_knowledge_graph_parts(self, limit: int | None = 200) -> list[Part]:
        self._ensure_initialized()
        if not self.variant.knowledge_graph:
            return []
        return self.store.read_knowledge_graph_parts(limit=limit)

    def get_chain_summary(self) -> dict[str, Any]:
        self._ensure_initialized()
        return self.store.read_chain_summary()

    def get_chain_summary_status(self) -> dict[str, Any]:
        self._ensure_initialized()
        return self.store.read_chain_summary_status()

    def get_chain_summary_parts(self, limit: int | None = 200) -> list[Part]:
        self._ensure_initialized()
        return self.store.read_chain_summary_parts(limit=limit)

    def get_reference_feature_status(self) -> dict[str, Any]:
        self._ensure_initialized()
        if not self.variant.reference_feature_source:
            return {
                "status": "disabled",
                "ready": False,
                "message": f"Reference feature extractor is disabled by ablation variant {self.variant.id}.",
            }
        status = self.store.read_reference_feature_status()
        if status.get("status") == "completed":
            try:
                status = {
                    **status,
                    **validate_reference_feature_extractor(
                        self.workspace_path,
                        allow_execution=self.variant.reference_feature_execution,
                    ),
                }
            except (OSError, RuntimeError, ValueError) as exc:
                status = {**status, "ready": False, "validationError": str(exc)}
        return status

    def get_reference_feature_parts(self) -> list[Part]:
        self._ensure_initialized()
        if not self.variant.reference_feature_source:
            return []
        return self.store.read_reference_feature_parts()

    def get_reference_feature_tool(self) -> dict[str, Any]:
        self._ensure_initialized()
        if not self.variant.reference_feature_extractor:
            return {}
        if not self.store.is_reference_feature_extractor_ready():
            return {}
        return inspect_reference_feature_extractor(self.workspace_path)

    def get_knowledge_graph_llm_config(self) -> dict[str, Any]:
        self._ensure_initialized()
        cfg = self._knowledge_graph_llm_config()
        return {"config": mask_llm_config(cfg), "sdk": {}}

    async def query_knowledge(
        self,
        question: str,
        domain: str | None = None,
        context: dict[str, Any] | None = None,
        observations: list[str] | None = None,
        include_evidence: bool = False,
        result_guard: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        self._ensure_initialized()
        if not self.variant.knowledge_query:
            raise RuntimeError(f"Knowledge queries are disabled by ablation variant {self.variant.id}.")
        if self.variant.knowledge_query_source == "graph" and not self.store.is_knowledge_graph_ready():
            raise RuntimeError("Knowledge graph is not ready. Build it successfully before querying knowledge.")
        if self.variant.knowledge_query_source == "references":
            result = await answer_reference_query(
                workspace_path=self.workspace_path,
                store=self.store,
                llm_config=self._knowledge_graph_llm_config(),
                question=question,
                domain=domain,
                context=context,
                observations=observations,
                include_evidence=include_evidence,
            )
        elif self.variant.knowledge_query_source == "extractor_source":
            if not self.store.is_reference_feature_extractor_ready(allow_execution=False):
                raise RuntimeError("Reference feature extractor source is not ready. Build and statically validate it first.")
            result = await answer_extractor_source_query(
                workspace_path=self.workspace_path,
                store=self.store,
                llm_config=self._knowledge_graph_llm_config(),
                question=question,
                domain=domain,
                context=context,
                observations=observations,
                include_evidence=include_evidence,
            )
        else:
            result = await answer_knowledge_query(
                workspace_path=self.workspace_path,
                store=self.store,
                llm_config=self._knowledge_graph_llm_config(),
                question=question,
                domain=domain,
                context=context,
                observations=observations,
                include_evidence=include_evidence,
            )
        if result_guard:
            result_guard()
        self.store.append_timeline({
            "type": "knowledge_query_answered",
            "timestamp": now_iso(),
            "message": question[:240],
            "payload": {
                "domain": domain,
                "source": self.variant.knowledge_query_source,
                "supportingKnowledge": result.get("supporting_knowledge", []),
                "supportingEvidence": result.get("supporting_evidence", []) if include_evidence else [],
            },
        })
        return result

    def search_knowledge_notes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self._ensure_initialized()
        self._require_knowledge_variant()
        return search_knowledge_notes(self.workspace_path, query, top_k)

    def search_evidence_notes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self._ensure_initialized()
        self._require_knowledge_variant()
        return search_evidence_notes(self.workspace_path, query, top_k)

    def search_knowledge_graph(self, query: str, relation_type: str | None = None, top_k: int = 10) -> list[dict[str, Any]]:
        self._ensure_initialized()
        self._require_knowledge_variant()
        return search_graph(self.workspace_path, query, relation_type, top_k)

    def get_knowledge_neighbors(self, concept: str, depth: int = 1) -> dict[str, Any]:
        self._ensure_initialized()
        self._require_knowledge_variant()
        return get_neighbors(self.workspace_path, concept, depth)

    def get_knowledge_supporting_evidence(self, note_or_edge_id: str, top_k: int = 8) -> list[dict[str, Any]]:
        self._ensure_initialized()
        self._require_knowledge_variant()
        return get_supporting_evidence(self.workspace_path, note_or_edge_id, top_k)

    def suggest_knowledge_next_checks(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self._ensure_initialized()
        self._require_knowledge_variant()
        return suggest_next_checks(self.workspace_path, query, top_k)

    def update_knowledge_graph_llm_config(self, values: dict[str, Any]) -> dict[str, Any]:
        self._ensure_initialized()
        self.store.write_knowledge_graph_llm_config(values)
        cfg = self._knowledge_graph_llm_config()
        return {"config": mask_llm_config(cfg), "sdk": {}}

    def update_main_llm_config(self, values: dict[str, Any]) -> dict[str, Any]:
        self._ensure_initialized()
        self.store.write_main_llm_config(values)
        cfg = read_effective_llm_config(self.workspace_path)
        return {"config": mask_llm_config(cfg), "sdk": {}}

    def _record_build_outcome(
        self,
        write_status: Callable[[dict[str, Any]], Any],
        event_prefix: str,
        trigger: str,
        *,
        status: str,
        message: str,
        timeline_message: str | None = None,
    ) -> None:
        """Persist a finished builder status and its timeline event."""
        finished = now_iso()
        write_status({
            "running": False,
            "status": status,
            "finishedAt": finished,
            "message": message,
        })
        self.store.append_timeline({
            "type": f"{event_prefix}_{status}",
            "timestamp": finished,
            "message": timeline_message if timeline_message is not None else message,
            "payload": {"trigger": trigger},
        })

    async def build_knowledge_graph(self, trigger: str = "manual", uploaded_paths: list[str] | None = None) -> dict[str, Any]:
        self._ensure_initialized()
        setattr(self, "_knowledge_graph_pause_requested", False)
        try:
            if not self.variant.knowledge_graph:
                raise RuntimeError(f"Knowledge graph building is disabled by ablation variant {self.variant.id}.")
            if trigger == "rebuild":
                self.store.clear_knowledge_graph()
            status = self.store.write_knowledge_graph_build_status({
                "running": True,
                "status": "running",
                "startedAt": now_iso(),
                "finishedAt": None,
                "trigger": trigger,
                "message": "Knowledge graph build is running.",
                "uploadedPaths": uploaded_paths or [],
            })
            self.store.append_timeline({
                "type": "knowledge_graph_build_started",
                "timestamp": status["startedAt"],
                "message": trigger,
                "payload": {"uploadedPaths": uploaded_paths or []},
            })
            self._emit_knowledge_graph_snapshot()
            await build_knowledge_graph(
                workspace_path=self.workspace_path,
                store=self.store,
                llm_config=self._knowledge_graph_llm_config(),
                trigger=trigger,
                uploaded_paths=uploaded_paths,
                on_part=self._emit_knowledge_graph_parts,
                on_runner=lambda runner: setattr(self, "_knowledge_graph_runner", runner),
                on_change=self._emit_knowledge_graph_snapshot,
            )
        except Exception as exc:
            if getattr(self, "_knowledge_graph_pause_requested", False):
                self._record_build_outcome(
                    self.store.write_knowledge_graph_build_status, "knowledge_graph_build", trigger,
                    status="paused", message="Knowledge graph builder paused.", timeline_message="paused",
                )
                self._emit_knowledge_graph_snapshot()
                return {"ok": False, "paused": True, "knowledgeGraph": self.get_knowledge_graph(), "status": self.get_knowledge_graph_build_status()}
            self._record_build_outcome(
                self.store.write_knowledge_graph_build_status, "knowledge_graph_build", trigger,
                status="failed", message=str(exc),
            )
            self._emit_knowledge_graph_snapshot()
            raise
        if getattr(self, "_knowledge_graph_pause_requested", False):
            self._record_build_outcome(
                self.store.write_knowledge_graph_build_status, "knowledge_graph_build", trigger,
                status="paused", message="Knowledge graph builder paused.", timeline_message="paused",
            )
            self._emit_knowledge_graph_snapshot()
            return {"ok": False, "paused": True, "knowledgeGraph": self.get_knowledge_graph(), "status": self.get_knowledge_graph_build_status()}
        self._record_build_outcome(
            self.store.write_knowledge_graph_build_status, "knowledge_graph_build", trigger,
            status="completed", message="Knowledge graph updated.", timeline_message="knowledge_base",
        )
        self._emit_knowledge_graph_snapshot()
        return {"ok": True, "knowledgeGraph": self.get_knowledge_graph(), "status": self.get_knowledge_graph_build_status()}

    async def pause_knowledge_graph_build(self, reason: str | None = None) -> dict[str, Any]:
        message = reason or "Paused from knowledge graph UI."
        setattr(self, "_knowledge_graph_pause_requested", True)
        runner = getattr(self, "_knowledge_graph_runner", None)
        if runner is not None:
            await runner.interrupt()
            status = self.store.write_knowledge_graph_build_status({
                "status": "pausing",
                "message": message,
            })
            self.store.append_timeline({
                "type": "knowledge_graph_build_pausing",
                "timestamp": now_iso(),
                "message": message,
            })
            self._emit_knowledge_graph_snapshot()
            return {"paused": True, "status": status}
        status = self.store.write_knowledge_graph_build_status({
            "running": False,
            "status": "paused",
            "finishedAt": now_iso(),
            "message": message,
        })
        self.store.append_timeline({
            "type": "knowledge_graph_build_paused",
            "timestamp": status["finishedAt"],
            "message": message,
        })
        self._emit_knowledge_graph_snapshot()
        return {"paused": True, "status": status}

    async def build_chain_summary(self, trigger: str = "manual") -> dict[str, Any]:
        self._ensure_initialized()
        setattr(self, "_chain_summary_pause_requested", False)
        try:
            clear_file(self.store.chain_summary_log_path)
            status = self.store.write_chain_summary_status({
                "running": True,
                "status": "running",
                "startedAt": now_iso(),
                "finishedAt": None,
                "trigger": trigger,
                "message": "Chain builder is running.",
            })
            self.store.append_timeline({
                "type": "chain_summary_build_started",
                "timestamp": status["startedAt"],
                "message": trigger,
            })
            summary = await build_chain_summary(
                workspace_path=self.workspace_path,
                store=self.store,
                llm_config=self._knowledge_graph_llm_config(),
                on_part=self._emit_chain_summary_parts,
                on_runner=lambda runner: setattr(self, "_chain_summary_runner", runner),
            )
        except Exception as exc:
            if getattr(self, "_chain_summary_pause_requested", False):
                self._record_build_outcome(
                    self.store.write_chain_summary_status, "chain_summary_build", trigger,
                    status="paused", message="Chain builder paused.", timeline_message="paused",
                )
                return {"ok": False, "paused": True, "chainSummary": self.get_chain_summary(), "status": self.get_chain_summary_status()}
            self._record_build_outcome(
                self.store.write_chain_summary_status, "chain_summary_build", trigger,
                status="failed", message=str(exc),
            )
            raise
        if getattr(self, "_chain_summary_pause_requested", False):
            self._record_build_outcome(
                self.store.write_chain_summary_status, "chain_summary_build", trigger,
                status="paused", message="Chain builder paused.", timeline_message="paused",
            )
            return {"ok": False, "paused": True, "chainSummary": self.get_chain_summary(), "status": self.get_chain_summary_status()}
        self._record_build_outcome(
            self.store.write_chain_summary_status, "chain_summary_build", trigger,
            status="completed", message="Chain summary updated.", timeline_message="artifacts/chain-summary.json",
        )
        return {"ok": True, "chainSummary": summary, "status": self.get_chain_summary_status()}

    async def pause_chain_summary_build(self, reason: str | None = None) -> dict[str, Any]:
        message = reason or "Paused from chain summary UI."
        setattr(self, "_chain_summary_pause_requested", True)
        runner = getattr(self, "_chain_summary_runner", None)
        if runner is not None:
            await runner.interrupt()
            status = self.store.write_chain_summary_status({
                "status": "pausing",
                "message": message,
            })
            self.store.append_timeline({
                "type": "chain_summary_build_pausing",
                "timestamp": now_iso(),
                "message": message,
            })
            return {"paused": True, "status": status}
        status = self.store.write_chain_summary_status({
            "running": False,
            "status": "paused",
            "finishedAt": now_iso(),
            "message": message,
        })
        self.store.append_timeline({
            "type": "chain_summary_build_paused",
            "timestamp": status["finishedAt"],
            "message": message,
        })
        return {"paused": True, "status": status}

    def read_workspace_file(self, path: str) -> dict[str, Any]:
        self._ensure_initialized()
        return self.store.read_text_file(path)

    def upload_reference_file(self, filename: str, content: bytes) -> str:
        self._ensure_initialized()
        path = self.store.write_reference_file(filename, content)
        self._emit_realtime("workspace_files", {
            "fileTree": self.get_file_tree(),
            "change": {"kind": "reference_uploaded", "path": path},
        })
        return path

    def delete_reference_file(self, path: str) -> str:
        self._ensure_initialized()
        deleted = self.store.delete_reference_file(path)
        self._emit_realtime("workspace_files", {
            "fileTree": self.get_file_tree(),
            "change": {"kind": "reference_deleted", "path": deleted},
        })
        return deleted

    def upload_raw_data_zip(self, filename: str, content: bytes) -> dict[str, Any]:
        self._ensure_initialized()
        result = self.store.extract_raw_data_zip(filename, content)
        self._emit_realtime("workspace_files", {
            "fileTree": self.get_file_tree(),
            "change": {
                "kind": "raw_data_uploaded",
                "archive": result.get("archive"),
                "targetDir": result.get("targetDir"),
                "extractedCount": len(result.get("extracted") or []),
            },
        })
        return result

    async def clear_debug_logs(self, scope: str = "main") -> None:
        self._ensure_initialized()
        reset_reason = "chat_reset" if scope == "chat" else "workspace_reset" if scope == "all" else "main_log_cleared"
        await self._close_main_runner(
            reason=reset_reason,
            message=(
                "Main runner closed for workspace reset; the next user message will start a fresh SDK session."
                if scope == "all"
                else "Main runner closed for chat reset; the next user message will start a fresh SDK session."
                if scope == "chat"
                else "Main runner closed after clearing main logs; the next user message will start a fresh SDK session."
            ),
        )
        if scope in {"all", "chat"}:
            await self._close_active_node_runner("workspace reset" if scope == "all" else "chat reset")
        self.store.clear_debug_logs(scope if scope in {"all", "chat"} else "main")
        self._last_emitted_parts.clear()
        if scope in {"all", "chat"}:
            self.state = self.store.read_state()
            self._stamp_variant_state()
            self.active_node_session = None

    async def reset_workflow_preserving_assets(self) -> WorkspaceState:
        """Reset orchestration state while retaining user-provided assets."""
        self._ensure_initialized()
        if self.is_active_node_task_running() or (self.main_runner and self.main_runner.is_running):
            raise RuntimeError("Cannot reset workflow while a harness task is active.")
        await self._close_main_runner(
            reason="workflow_reset_preserving_assets",
            message="Main runner closed for workflow reset while preserving uploaded assets.",
        )
        await self._close_active_node_runner("workflow_reset_preserving_assets")
        self.state = self.store.reset_workflow_preserving_assets()
        self._stamp_variant_state()
        self.active_node_session = None
        self._active_node_task = None
        self._last_emitted_parts.clear()
        self._emit_runtime_snapshot()
        return self.state

    def get_node_specs(self) -> list[dict[str, Any]]:
        if not self.variant.node_chain:
            return []
        return [
            {
                "type": spec.type,
                "phase": spec.phase,
                "purpose": self.variant.node_purpose(spec.type, spec.purpose),
                "requires": self.variant.node_requires(spec.type, spec.requires),
                "produces": self.variant.node_produces(spec.type, spec.produces),
                "next": self.variant.node_next(spec.type, spec.next),
                "nativeTools": build_node_native_tools(spec.type, variant=self.variant),
            }
            for spec in NODE_SPECS
            if self.variant.node_enabled(spec.type)
        ]

    async def close(self) -> None:
        if self.main_runner:
            await self.main_runner.close()
        if self.active_node_runner:
            await self.active_node_runner.close()

    def _control_mode(self) -> str:
        self._ensure_initialized()
        assert self.state
        return "auto" if self.state.get("controlMode") == "auto" else "manual"

    def _park_control_request(
        self,
        kind: str,
        args: dict[str, Any],
        node: NodeSession | None = None,
    ) -> dict[str, Any]:
        node_type = str(args.get("nodeType") or (node["nodeType"] if node else ""))
        request: ControlRequest = {
            "id": str(uuid4()),
            "kind": kind,
            "status": "pending",
            "createdAt": now_iso(),
            "nodeType": node_type,
            "args": args,
            "message": self._control_request_message(kind, node_type, args),
        }
        if node:
            request["nodeSessionId"] = node["id"]
        self.store.set_pending_control(request)
        self.state = self.store.read_state()
        return {
            "status": "pending_human_decision",
            "controlRequestId": request["id"],
            "kind": kind,
            "nodeType": node_type,
            "message": request["message"],
        }

    def _control_request_message(self, kind: str, node_type: str, args: dict[str, Any]) -> str:
        if kind == "enter_node":
            return f"Agent requested entering node {node_type}: {args.get('rationale', '')}"
        if kind == "finish_node":
            return f"Agent requested finishing node {node_type}: {args.get('summary', '')}"
        return kind

    # Maximum number of reminder turns the harness will send to a node
    # session that returned without calling finish_node. After this many
    # reminders the harness gives up and marks the node as failed so
    # the active-node lock is released and the orchestrator can recover
    # on the next main turn. A single reminder is enough in practice;
    # the bound exists only to keep the protocol bounded.
    _NODE_PROTOCOL_MAX_REMINDERS = 1

    async def _handle_node_runner_return(
        self,
        node: NodeSession,
        *,
        sdk_error: BaseException | None = None,
        auto_advance: bool = True,
    ) -> None:
        """Reconcile node session state after a node runner turn ends.

        Three paths:

        1. ``sdk_error`` is set: the SDK call itself crashed. Mark the
           node as ``failed`` with the SDK error in the summary, release
           the active-node lock so the next main turn isn't blocked,
           and let the original exception bubble up to the HTTP caller.

        2. The node already reached a terminal status
           (``completed`` / ``paused`` / ``failed`` / ``exited`` /
           ``waiting_approval``): honor the existing auto-next or
           no-op behavior. The ``completed`` branch additionally closes
           the long-lived main runner once the pipeline is complete so
           stale multi-hour-delayed tool_results from the main session
           are dropped.

        3. The node returned without calling ``finish_node`` and the
           SDK did not throw: append a system reminder to the node log
           and re-prompt the same node session once. If the agent
           still does not call ``finish_node`` after the bounded
           reminder turn, give up and mark the node as ``failed`` with
           a clear summary, releasing the lock.
        """
        self.state = self.store.read_state()
        latest = self.store.read_node_session(node["id"]) or node
        if sdk_error is not None:
            if self.execution.snapshot()["phase"] in {"pausing", "paused", "recovering"}:
                if latest.get("status") in {"completed", "failed", "exited"}:
                    # A terminal workflow transition committed before the DSH
                    # pause boundary. Pause may retire the runtime, but must not
                    # roll application state back to a resumable node.
                    return
                latest.update({"status": "paused", "pausedAt": now_iso(), "summary": "Paused by user."})
                self.store.write_node_session(latest)
                self._preserve_active_node_lock(latest)
                self.active_node_session = latest
                return
            # If the user already resolved the node session (e.g. by
            # clicking Interrupt, which sets status=paused), the SDK
            # error is just a delayed side effect of that intervention
            # and we must not overwrite the user's decision. Still
            # tear down the dead runner. For paused nodes, keep the
            # active-node lock so the next user message can resume the
            # same node instead of starting a new main turn.
            if latest.get("status") == "paused":
                if self.active_node_runner is not None:
                    await self._close_active_node_runner("node_sdk_error_after_user_pause")
                self.active_node_runner = None
                self.active_node_session = latest
                return
            if latest.get("status") in {"waiting_approval", "failed", "exited"}:
                await self._release_runner_and_lock("node_sdk_error_after_user_resolution")
                return
            await self._fail_node_after_sdk_crash(node, latest, sdk_error)
            return
        if latest.get("status") == "waiting_approval":
            return
        if latest.get("status") == "completed":
            if auto_advance and self._control_mode() == "auto":
                await self._maybe_auto_enter_next_node(latest)
            # Once the pipeline is fully complete (no more auto-next
            # node), close the long-lived DSH runtime. This releases
            # the session-owned child processes and prevents a stale
            # main turn from acting on a completed pipeline.
            if self._is_pipeline_complete():
                await self._close_main_runner()
            return
        if latest.get("status") in {"paused", "failed", "exited"}:
            return
        await self._remind_or_fail_unfinished_node(node, latest, auto_advance=auto_advance)

    async def _fail_node_after_sdk_crash(
        self,
        node: NodeSession,
        latest: NodeSession,
        sdk_error: BaseException,
    ) -> None:
        """Mark a node session as failed because the underlying SDK
        runner crashed (control-request timeout, subprocess error,
        network drop, etc.). Release the active-node lock so the
        orchestrator can recover on the next main turn. The original
        exception is re-raised by the caller."""
        # The SDK shutdown can race with a user pause between the caller's
        # initial status read and this failure path.  Re-read durable state
        # immediately before emitting failure artifacts so a deliberate pause
        # always wins the race.
        current = self.store.read_node_session(node["id"])
        if current and current.get("status") == "paused":
            self._preserve_active_node_lock(current)
            self.active_node_session = current
            return
        reason = f"Node SDK runner crashed: {type(sdk_error).__name__}: {sdk_error}"
        self.store.append_timeline({
            "type": "node_runner_crash",
            "timestamp": now_iso(),
            "nodeSessionId": latest.get("id"),
            "nodeType": latest.get("nodeType"),
            "message": reason,
        })
        self.store.append_node_part(latest.get("id"), system_text_part(f"Harness error: {reason}"))
        self.store.append_timeline({
            "type": "node_protocol_error",
            "timestamp": now_iso(),
            "nodeSessionId": latest.get("id"),
            "nodeType": latest.get("nodeType"),
            "message": reason,
        })
        await self.finish_node({
            "success": False,
            "summary": reason,
            "goalMet": False,
            "outputPaths": [],
        })
        # Make sure no runner reference is leaked even if finish_node
        # took an unusual path.
        await self._release_runner_and_lock("node_runner_crash")

    async def _release_runner_and_lock(self, reason: str) -> None:
        """Best-effort cleanup of the active node runner and the
        active-node lock. Used after a node session reaches a terminal
        status through a non-standard path (SDK error, protocol
        abandonment) so the next main turn is not blocked."""
        if self.active_node_runner is not None:
            await self._close_active_node_runner(reason)
        self.active_node_runner = None
        state = self.store.read_state() or self.state
        paused = (
            self.store.read_node_session(state["activeNodeSessionId"])
            if state and state.get("activeNodeSessionId")
            else None
        )
        if paused and paused.get("status") == "paused":
            self._preserve_active_node_lock(paused)
            self.active_node_session = paused
            return
        self.active_node_session = None
        if self.state is not None:
            self.state["activeNode"] = None
            self.state["activeNodeSessionId"] = None
            self.store.write_state(self.state)

    async def _remind_or_fail_unfinished_node(
        self,
        node: NodeSession,
        latest: NodeSession,
        *,
        auto_advance: bool = True,
    ) -> None:
        """Handle a node that returned without calling ``finish_node``
        and without an SDK crash. Re-prompt the same node session once
        with a reminder, then mark as ``failed`` if the reminder turn
        also ends without a terminal status.

        ``auto_advance`` is inherited from the caller: inside the
        background node task it is False, because entering the next node
        here would replace ``active_node_runner`` and fail against the
        still-running task. The task's own post-turn flow advances
        instead. A foreground (non-background) entry keeps True."""
        reminders_used = int(latest.get("protocolReminders", 0) or 0)
        if reminders_used >= self._NODE_PROTOCOL_MAX_REMINDERS:
            summary = (
                "Node runner returned without calling finish_node MCP after "
                f"{reminders_used} reminder turn(s); harness is giving up."
            )
            self.store.append_timeline({
                "type": "node_protocol_error",
                "timestamp": now_iso(),
                "nodeSessionId": latest.get("id"),
                "nodeType": latest.get("nodeType"),
                "message": summary,
            })
            self.store.append_node_part(latest.get("id"), system_text_part(f"Harness error: {summary}"))
            await self.finish_node({
                "success": False,
                "summary": summary,
                "goalMet": False,
                "outputPaths": [],
            })
            await self._release_runner_and_lock("node_protocol_abandoned")
            return
        latest["protocolReminders"] = reminders_used + 1
        latest["protocolReminderReason"] = "Node runner returned without calling finish_node MCP."
        self.store.write_node_session(latest)
        self.store.append_timeline({
            "type": "node_protocol_reminder",
            "timestamp": now_iso(),
            "nodeSessionId": latest.get("id"),
            "nodeType": latest.get("nodeType"),
            "message": (
                f"Node runner returned without calling finish_node MCP; "
                f"sending reminder turn {latest['protocolReminders']} of "
                f"{self._NODE_PROTOCOL_MAX_REMINDERS}."
            ),
            "payload": {"remindersUsed": latest["protocolReminders"]},
        })
        self.store.append_node_part(
            latest.get("id"),
            system_text_part(
                "Harness reminder: this turn ended without calling "
                "mcp__ts_harness__finish_node. Review the workspace and node log, "
                "complete the required artifacts, and call finish_node with a "
                "summary, outputPaths, and (for iterative-solving) explicit "
                "loopDecision/nextNode. If you cannot proceed, call finish_node "
                "with success=false so the orchestrator can decide what to do."
            ),
        )
        self.active_node_session = latest
        if self.dry_run or self.active_node_runner is None:
            # No live runner to re-prompt (dry-run mode or the runner
            # was already torn down). Treat the reminder as the final
            # turn and mark as failed so the lock is released.
            await self._handle_node_runner_return(latest, auto_advance=auto_advance)
            return
        try:
            await self._send_execution_turn(
                self.active_node_runner,
                f"node:{latest['id']}",
                "[Harness reminder] This turn must end with mcp__ts_harness__finish_node. "
                "Inspect the current workspace and node log, then call finish_node."
            )
        except BaseException as exc:
            await self._handle_node_runner_return(latest, sdk_error=exc)
            return
        await self._handle_node_runner_return(latest, auto_advance=auto_advance)

    async def _close_runner(self, runner: SdkRunner, *, event_type: str, message: str, payload: dict[str, Any]) -> None:
        """Interrupt and close a DSH runner, then record the outcome in the
        timeline.  Interrupt and close failures are recorded as
        runner_close_failed timeline events; a failed interrupt still
        attempts close."""
        failures: list[str] = []
        if runner.is_running:
            try:
                await runner.interrupt()
            except Exception as exc:
                failures.append(f"interrupt: {type(exc).__name__}: {exc}")
        try:
            await runner.close()
        except Exception as exc:
            failures.append(f"close: {type(exc).__name__}: {exc}")
        for failure in failures:
            self.store.append_timeline({
                "type": "runner_close_failed",
                "timestamp": now_iso(),
                "message": failure,
                "payload": {**payload, "runnerEvent": event_type},
            })
        self._revoke_runner_mcp_capability(runner)
        self.store.append_timeline({
            "type": event_type,
            "timestamp": now_iso(),
            "message": message,
            "payload": payload,
        })

    async def _close_main_runner(
        self,
        reason: str = "pipeline_complete",
        message: str = "Main runner closed (pipeline complete); late tool_results from the main session will be discarded.",
    ) -> None:
        """Terminate the long-lived DeepSeek Harness runtime subprocess.

        This releases its JSON-RPC session and MCP child process.  A later
        resume can restore its durable DSH session/checkpoint when requested.

        Safe to call multiple times; subsequent calls are no-ops."""
        runner = self.main_runner
        if runner is None:
            return
        self.main_runner = None
        self._main_runner_knowledge_graph_ready = None
        self._main_runner_reference_feature_ready = None
        await self._close_runner(
            runner,
            event_type="main_runner_closed",
            message=message,
            payload={"reason": reason},
        )

    async def _close_active_node_runner(self, reason: str) -> None:
        runner = self.active_node_runner
        if runner is None:
            return
        self.active_node_runner = None
        await self._close_runner(
            runner,
            event_type="active_node_runner_closed",
            message=f"Active node runner closed for {reason}.",
            payload={"reason": reason},
        )

    async def _maybe_auto_enter_next_node(self, finished_node: NodeSession) -> None:
        self._ensure_initialized()
        assert self.state
        if self._control_mode() != "auto":
            return
        latest = self.store.read_node_session(finished_node["id"]) or finished_node
        if self.state.get("activeNode") or latest.get("status") != "completed" or latest.get("success") is not True:
            return
        next_node = self._next_node_after_completion(latest)
        if not next_node:
            return
        self.store.append_timeline({
            "type": "auto_next",
            "timestamp": now_iso(),
            "nodeSessionId": latest["id"],
            "nodeType": latest["nodeType"],
            "message": f"Auto-entering {next_node}.",
            "payload": {"nextNode": next_node},
        })
        await self.enter_node({
            "nodeType": next_node,
            "rationale": f"Previous node {latest['nodeType']} completed successfully; continuing node chain.",
            "inputSummary": f"Upstream node {latest['nodeType']} completed. Summary: {latest.get('summary', '')}. Output paths: {', '.join(latest.get('outputPaths', []))}",
        }, background=True)

    def _is_pipeline_complete(self) -> bool:
        """Whether the persisted node chain has fully completed."""
        return self.node_state.is_pipeline_complete(self.store.read_state())

    def _next_node_after_completion(self, latest: NodeSession) -> NodeType | None:
        return self.node_state.next_node_after_completion(latest)

    def _validate_failed_node_retry_request(self, args: dict[str, Any]) -> None:
        """Block accidental node entry after a protocol or runner failure.

        The main prompt tells the agent to retry a failed node only
        after an explicit user request, but stale context or delayed
        tool calls can still invoke enter_node. Make the routing rule
        enforceable at the backend boundary: the next entry after a
        failed/exited node must be an explicit retry of that same node.
        """
        latest = self._latest_node_session()
        if not latest or latest.get("status") not in {"failed", "exited"}:
            return
        node_type = args.get("nodeType")
        retry = args.get("retryFailedNode") is True
        if retry and node_type == latest.get("nodeType"):
            return
        raise RuntimeError(
            f"Cannot enter {node_type}: latest node {latest.get('nodeType')} ended with "
            f"status {latest.get('status')}. Ask the user before retrying, then call "
            "enter_node with retryFailedNode=true for that same node."
        )

    def _latest_node_session(self) -> NodeSession | None:
        sessions = self.store.list_node_sessions()
        return sessions[-1] if sessions else None

    def _is_next_node_specified(self, args: dict[str, Any]) -> bool:
        if "nextNode" not in args:
            return False
        value = args.get("nextNode")
        if value is None:
            return False
        return True

    def _normalize_next_node(self, value: Any) -> NodeType | None:
        return self.node_state.normalize_next_node(value)

    def _normalize_loop_decision(self, value: Any) -> str | None:
        return self.node_state.normalize_loop_decision(value)

    def _validate_finish_control(
        self,
        node_type: NodeType,
        success: bool,
        next_node: NodeType | None,
        loop_decision: str | None,
        output_paths: list[str] | None = None,
        goal_met: bool | None = None,
        next_node_specified: bool = False,
    ) -> None:
        self.node_state.validate_finish_control(
            node_type,
            success,
            next_node,
            loop_decision,
            output_paths,
            goal_met,
            next_node_specified,
        )

    def _read_iteration_state_recommend_exit(self) -> bool | None:
        return self.node_state.read_iteration_state_recommend_exit()

    def _ensure_initialized(self) -> None:
        if not self.state:
            self.initialize()

    def _stamp_variant_state(self) -> None:
        if not self.state:
            return
        if (
            self.state.get("variantId") == self.variant.id
            and self.state.get("variantName") == self.variant.name
        ):
            return
        previous = self.state.get("variantId")
        self.state["variantId"] = self.variant.id
        self.state["variantName"] = self.variant.name
        self.store.write_state(self.state)
        self.store.append_timeline({
            "type": "variant_selected",
            "timestamp": now_iso(),
            "message": f"{self.variant.id} · {self.variant.name}",
            "payload": {"variant": self.variant.public_payload(), "previousVariantId": previous},
        })

    def _require_knowledge_variant(self) -> None:
        if not self.variant.knowledge_graph:
            raise RuntimeError(f"Knowledge graph access is disabled by ablation variant {self.variant.id}.")

    def _knowledge_query_ready(self) -> bool:
        if not self.variant.knowledge_query:
            return False
        if self.variant.knowledge_query_source == "graph":
            return self.store.is_knowledge_graph_ready()
        if self.variant.knowledge_query_source == "references":
            return True
        if self.variant.knowledge_query_source == "extractor_source":
            return self.store.is_reference_feature_extractor_ready(allow_execution=False)
        return False

    def _knowledge_graph_llm_config(self) -> LlmConfig:
        main = read_effective_llm_config(self.workspace_path)
        if not self.store.knowledge_graph_llm_path.exists():
            return main
        graph = self.store.read_knowledge_graph_llm_config()
        return LlmConfig(
            authMode=graph.get("authMode") if graph.get("authMode") in {"manual", "sdk-default"} else main.authMode,
            model=graph.get("model") or main.model,
            apiKey=graph.get("apiKey") or main.apiKey,
            baseUrl=graph.get("baseUrl") or main.baseUrl,
            protocol=graph.get("protocol") or main.protocol,
            maxTokens=graph.get("maxTokens") or main.maxTokens,
        )

    def _restore_active_node_session(self) -> NodeSession | None:
        if not self.state or not self.state.get("activeNodeSessionId"):
            return None
        return self.store.read_node_session(self.state["activeNodeSessionId"])

    def _main_progress_snapshot(self) -> dict[str, Any]:
        """Build the authoritative, read-only routing context for the main agent."""
        self._ensure_initialized()
        state = self.store.read_state() or self.state
        assert state
        sessions = self.store.list_node_sessions()
        latest = sessions[-1] if sessions else None
        active = self.store.read_node_session(state["activeNodeSessionId"]) if state.get("activeNodeSessionId") else None

        recommended_action = "enter_node"
        recommended_node: NodeType | None = "problem-contract"
        reason = "No node has completed; start by establishing the problem contract."
        if not self.variant.node_chain:
            recommended_action = "direct_tool_use"
            recommended_node = None
            reason = f"{self.variant.id} uses one main coding-agent session and disables the node chain."
        elif active:
            recommended_action = "active_node_in_progress"
            recommended_node = None
            reason = f"Node {active['nodeType']} is currently {active.get('status', 'active')}; do not start another node."
        elif "final-summary" in state.get("completedNodes", []):
            recommended_action = "pipeline_complete"
            recommended_node = None
            reason = "The final-summary node has completed."
        elif state.get("pendingControl"):
            recommended_action = "await_control_approval"
            recommended_node = None
            reason = f"A {state['pendingControl'].get('kind', 'control')} request is waiting for human approval."
        elif latest and latest.get("status") == "completed" and latest.get("success") is True:
            recommended_node = self._next_node_after_completion(latest)
            if recommended_node:
                recommended_action = "enter_node"
                reason = f"The latest completed node routes to {recommended_node}."
            else:
                recommended_action = "pipeline_stopped"
                reason = "The latest completed node explicitly stopped the pipeline or has no successor."
        elif latest and latest.get("status") in {"failed", "exited"}:
            recommended_action = "retry_failed_node"
            recommended_node = latest["nodeType"]
            reason = f"The latest {latest['nodeType']} node ended with status {latest.get('status')}. Retry only on explicit user request."
        elif "problem-contract" in state.get("completedNodes", []):
            recommended_node = "iterative-solving" if not self.variant.knowledge_to_tools else "knowledge-to-tools"
            reason = (
                "The problem contract is complete; build the deterministic reference feature extractor next."
                if self.variant.knowledge_to_tools
                else "The problem contract is complete and no later completed node determines a different route."
            )

        anchor_paths = (
            "user/problem-contract.md",
            "user/data-spec.md",
            "user/iteration-state.md",
            "reports/final-summary.md",
            "user/final-solution.md",
        )
        return {
            "variant": self.variant.public_payload(),
            "workspaceId": state.get("workspaceId"),
            "controlMode": state.get("controlMode"),
            "pendingControl": _main_control_snapshot(state.get("pendingControl")),
            "activeNode": state.get("activeNode"),
            "activeNodeSession": _main_node_snapshot(active),
            "completedNodes": list(state.get("completedNodes", [])),
            "pipelineComplete": self.node_state.is_pipeline_complete(state),
            "latestNodeSession": _main_node_snapshot(latest),
            "anchorArtifacts": {path: (self.workspace_path / path).exists() for path in anchor_paths},
            "knowledgeGraphReady": self.variant.knowledge_graph and self.store.is_knowledge_graph_ready(),
            "knowledgeQueryReady": self._knowledge_query_ready(),
            "knowledgeQuerySource": self.variant.knowledge_query_source if self.variant.knowledge_query else "none",
            "referenceFeatureExtractorReady": self.variant.reference_feature_extractor and self.store.is_reference_feature_extractor_ready(),
            "referenceFeatureSourceReady": self.variant.reference_feature_source and self.store.is_reference_feature_extractor_ready(allow_execution=False),
            "recommendedAction": recommended_action,
            "recommendedNode": recommended_node,
            "routingReason": reason,
        }

    def _ensure_main_runner(self) -> None:
        if self.main_runner:
            return
        current_state = self.store.read_state() or self.state
        resume_session_id = current_state.get("mainSdkSessionId") if current_state else None

        def on_session_id(sdk_session_id: str) -> None:
            latest_state = self.store.read_state() or self.state
            if not latest_state:
                return
            latest_state["mainSdkSessionId"] = sdk_session_id
            self.store.write_state(latest_state)
            self.state = latest_state

        knowledge_query_ready = self._knowledge_query_ready()
        reference_feature_ready = self.variant.reference_feature_extractor and self.store.is_reference_feature_extractor_ready()
        self.main_runner = build_main_runner(
            workspace_path=self.workspace_path,
            locale=self.locale,
            log_path=self.store.main_log_path,
            knowledge_query_ready=knowledge_query_ready,
            reference_feature_extractor_ready=reference_feature_ready,
            variant=self.variant,
            on_part=self._emit_main_parts,
            on_session_id=on_session_id,
            resume_session_id=resume_session_id,
        )
        self._main_runner_knowledge_graph_ready = knowledge_query_ready
        self._main_runner_reference_feature_ready = reference_feature_ready

    async def _refresh_main_runner_for_dynamic_tools(self) -> None:
        if not self.main_runner:
            return
        graph_ready = self._knowledge_query_ready()
        feature_ready = self.variant.reference_feature_extractor and self.store.is_reference_feature_extractor_ready()
        if self._main_runner_knowledge_graph_ready == graph_ready and self._main_runner_reference_feature_ready == feature_ready:
            return
        await self._close_main_runner(
            reason="dynamic_tool_availability_changed",
            message="Main runner closed because reference tool availability changed; the next turn will use a fresh tool set.",
        )

    def _spawn_node_runner(self, node: NodeSession) -> None:
        node_id = node["id"]
        resume_session_id = node.get("sdkSessionId")
        run_epoch = int(node.get("runEpoch", 0) or 0) + 1
        node["runEpoch"] = run_epoch
        self.store.write_node_session(node)

        def on_session_id(sdk_session_id: str) -> None:
            latest = self.store.read_node_session(node_id)
            state = self.store.read_state() or self.state
            if not (
                latest
                and latest.get("status") in {"running", "paused"}
                and latest.get("runEpoch") == run_epoch
                and state
                and state.get("activeNodeSessionId") == node_id
            ):
                return
            latest["sdkSessionId"] = sdk_session_id
            self.store.write_node_session(latest)

        self.active_node_runner = build_node_runner(
            workspace_path=self.workspace_path,
            locale=self.locale,
            node=node,
            log_path=self.store.node_log_path(node_id),
            knowledge_query_ready=self._knowledge_query_ready(),
            reference_feature_extractor_ready=(
                self.variant.reference_feature_extractor
                and self.store.is_reference_feature_extractor_ready()
            ),
            on_session_id=on_session_id,
            on_part=lambda part: self._emit_node_parts(node_id, part),
            variant=self.variant,
            resume_session_id=resume_session_id,
        )
