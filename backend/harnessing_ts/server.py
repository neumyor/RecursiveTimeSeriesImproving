from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import mimetypes
import os
from pathlib import Path
import socket
import sys
import threading
from typing import Any
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from harnessing_ts.api.payloads import (
    build_bootstrap_payload,
    build_experiments_bootstrap_payload,
    build_live_payload,
    masked_llm_config,
)
from harnessing_ts.api.realtime import RealtimeEvent, RealtimeEventBroker
from harnessing_ts.experiments_api import router as experiments_router
from harnessing_ts.orchestrator import HarnessOrchestrator
from harnessing_ts.paths import control_mode_from_env, default_workspace_path, frontend_root
from harnessing_ts.settings.llm import mask_llm_config, read_effective_llm_config
from harnessing_ts.state.workspace_store import WorkspaceStore
from harnessing_ts.variants import resolve_variant, variant_help_text


class SendRequest(BaseModel):
    text: str


class ClearLogsRequest(BaseModel):
    scope: str = "main"
    confirmReset: bool = False


class WorkflowResetRequest(BaseModel):
    preserveAssets: bool = True


class PauseRequest(BaseModel):
    reason: str | None = None


class ForceStopRequest(BaseModel):
    confirm: bool = False


class ControlDecisionRequest(BaseModel):
    reason: str | None = None


class RuntimeSettingsRequest(BaseModel):
    iterativeCandidateCount: int | None = None
    knowledgeGraphExtractionDepth: int | None = None


class KnowledgeGraphBuildRequest(BaseModel):
    trigger: str | None = "manual"


class DeleteReferenceRequest(BaseModel):
    path: str


class KnowledgeGraphLlmConfigRequest(BaseModel):
    authMode: str | None = None
    protocol: str | None = None
    model: str | None = None
    apiKey: str | None = None
    baseUrl: str | None = None
    maxTokens: int | None = None


class ReferenceFeatureRunRequest(BaseModel):
    input: Any


class MainLlmConfigRequest(BaseModel):
    authMode: str | None = None
    protocol: str | None = None
    model: str | None = None
    apiKey: str | None = None
    baseUrl: str | None = None
    maxTokens: int | None = None


class KnowledgeQueryRequest(BaseModel):
    domain: str | None = None
    question: str
    context: dict[str, Any] | None = None
    observations: list[str] | None = None
    includeEvidence: bool = False


class DshMcpControlRequest(BaseModel):
    workspace: str
    role: str
    nodeSessionId: str | None = None
    tool: str
    args: dict[str, Any] = {}


def _legacy_enabled() -> bool:
    return os.getenv("TS_HARNESS_LEGACY") == "true"


class _LazyOrchestrator:
    """Construct the legacy harness on first use, not when the server starts.

    The experiments console never calls a legacy route.  Eager construction
    would run the workspace uv sync and the old session recovery for a system
    the console does not use; TS_HARNESS_LEGACY=true keeps the old eager
    behavior for deployments that still use the interactive harness.
    """

    _OWN_ATTRS = frozenset({"_workspace_path", "_dry_run", "_locale", "_mode", "_configure", "_instance", "_lock"})

    def __init__(self, workspace_path: Path, *, dry_run: bool, mode: str, configure: Any) -> None:
        object.__setattr__(self, "_workspace_path", workspace_path)
        object.__setattr__(self, "_dry_run", dry_run)
        object.__setattr__(self, "_locale", "zh")
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_configure", configure)
        object.__setattr__(self, "_instance", None)
        object.__setattr__(self, "_lock", threading.Lock())

    @property
    def initialized(self) -> bool:
        return self._instance is not None

    def ensure_initialized(self) -> Any:
        return self._get()

    def _get(self) -> Any:
        with self._lock:
            if self._instance is None:
                instance = HarnessOrchestrator(
                    self._workspace_path, dry_run=self._dry_run, locale=self._locale, mode=self._mode
                )
                instance.initialize(reporter=_startup_report)
                self._configure(instance)
                instance.recover_interrupted_execution()
                object.__setattr__(self, "_instance", instance)
            return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._OWN_ATTRS:
            object.__setattr__(self, name, value)
        else:
            setattr(self._get(), name, value)


def _configure_legacy(orchestrator: HarnessOrchestrator, realtime: RealtimeEventBroker, control_url: str) -> None:
    orchestrator.set_realtime_event_sink(realtime.publish)
    orchestrator.set_runtime_snapshot_provider(lambda: _live(orchestrator))
    orchestrator.configure_mcp_control(control_url)


def create_app() -> FastAPI:
    workspace_path = default_workspace_path()
    dry_run = os.getenv("TS_HARNESS_DRY_RUN") == "true"
    debug_enabled = os.getenv("TS_HARNESS_DEBUG") == "true"
    legacy_enabled = _legacy_enabled()
    control_mode = control_mode_from_env() if legacy_enabled else "auto"
    frontend = frontend_root()
    web_root = frontend / "dist" if (frontend / "dist").exists() else frontend
    realtime = RealtimeEventBroker()
    control_url = os.getenv("DSH_MCP_CONTROL_URL", "").strip()
    if not control_url:
        # DSH's MCP process is a local child of this server.  It must never
        # discover the public listening address or inherit a user-facing URL.
        control_url = f"http://127.0.0.1:{_startup_port()}"
    orchestrator = _LazyOrchestrator(
        workspace_path,
        dry_run=dry_run,
        mode=control_mode,
        configure=lambda instance: _configure_legacy(instance, realtime, control_url),
    )
    if legacy_enabled:
        orchestrator.ensure_initialized()
    _print_ready_urls()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            if orchestrator.initialized:
                await orchestrator.complete_interrupted_recovery()
            yield
        finally:
            # asyncio tasks and MCP capabilities are process-local. Persist
            # an explicit paused state before Uvicorn tears the process down.
            if orchestrator.initialized:
                try:
                    # Server shutdown follows the same native DSH checkpoint
                    # boundary as an explicit user pause.
                    if orchestrator.execution.snapshot()["phase"] == "running":
                        await orchestrator.interrupt_current("Harness server is shutting down.")
                finally:
                    await orchestrator.close()

    app = FastAPI(title="HarnessingTS", lifespan=lifespan)
    app.include_router(experiments_router)
    app.state.orchestrator = orchestrator
    app.state.realtime = realtime
    app.state.workspace_path = workspace_path
    app.state.dry_run = dry_run
    app.state.debug_enabled = debug_enabled
    app.state.run_task = None
    app.state.knowledge_graph_task = None
    app.state.chain_summary_task = None
    app.state.pause_task = None

    @app.middleware("http")
    async def no_store_cache(request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception(_request: Any, exc: Exception) -> JSONResponse:
        return JSONResponse({"error": _format_exception(exc)}, status_code=500)

    @app.get("/api/bootstrap")
    async def bootstrap() -> dict[str, Any]:
        return _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)

    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        async def stream():
            queue = realtime.subscribe()
            try:
                initial = RealtimeEvent(0, "bootstrap_snapshot", {
                    "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled),
                })
                yield initial.as_sse()
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield event.as_sse()
            finally:
                realtime.unsubscribe(queue)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/state")
    async def state() -> dict[str, Any]:
        return orchestrator.get_state()

    @app.get("/api/timeline")
    async def timeline() -> list[dict[str, Any]]:
        return orchestrator.get_timeline()

    @app.get("/api/main-log")
    async def main_log() -> list[dict[str, Any]]:
        return orchestrator.get_main_parts()

    @app.get("/api/nodes")
    async def nodes() -> list[dict[str, Any]]:
        return orchestrator.get_node_sessions()

    @app.get("/api/nodes/{node_session_id}/log")
    async def node_log(node_session_id: str) -> list[dict[str, Any]]:
        return orchestrator.get_node_parts(node_session_id)

    @app.get("/api/node-specs")
    async def node_specs() -> list[dict[str, Any]]:
        return orchestrator.get_node_specs()

    @app.get("/api/llm-config")
    async def llm_config() -> dict[str, Any]:
        return _masked_llm_config(workspace_path)

    @app.get("/api/runtime-settings")
    async def runtime_settings() -> dict[str, Any]:
        return orchestrator.get_runtime_settings()

    @app.post("/internal/dsh/mcp")
    async def dsh_mcp_control(
        request: Request,
        payload: DshMcpControlRequest,
    ) -> Any:
        client_host = request.client.host if request.client else ""
        if client_host not in {"127.0.0.1", "::1"}:
            raise HTTPException(status_code=403, detail="DSH MCP control endpoint is loopback-only.")
        capability = request.headers.get("x-ts-harness-capability", "")
        if not capability:
            raise HTTPException(status_code=401, detail="Missing DSH MCP capability.")
        try:
            result = await orchestrator.dispatch_dsh_mcp_call(
                capability=capability,
                workspace=payload.workspace,
                role=payload.role,
                node_session_id=payload.nodeSessionId,
                tool=payload.tool,
                args=payload.args,
            )
            return result
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from None
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.post("/api/runtime-settings")
    async def update_runtime_settings(request: RuntimeSettingsRequest) -> dict[str, Any]:
        settings = orchestrator.update_runtime_settings(request.model_dump(exclude_none=True))
        return {"settings": settings, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.get("/api/knowledge-graph")
    async def knowledge_graph() -> dict[str, Any]:
        return orchestrator.get_knowledge_graph()

    @app.get("/api/knowledge-base/summary")
    async def knowledge_base_summary() -> dict[str, Any]:
        return orchestrator.get_knowledge_base_summary()

    @app.get("/api/knowledge-base/cards")
    async def knowledge_base_cards(kind: str, limit: int = 200) -> dict[str, Any]:
        try:
            return orchestrator.get_knowledge_base_cards(kind, limit)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.post("/api/knowledge/query")
    async def query_knowledge(request: KnowledgeQueryRequest) -> dict[str, Any]:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="question required")
        return await orchestrator.query_knowledge(
            request.question,
            request.domain,
            request.context,
            request.observations,
            include_evidence=request.includeEvidence,
        )

    @app.get("/api/knowledge/search/notes")
    async def search_knowledge_notes(q: str, top_k: int = 5) -> list[dict[str, Any]]:
        return orchestrator.search_knowledge_notes(q, top_k)

    @app.get("/api/knowledge/search/evidence")
    async def search_evidence(q: str, top_k: int = 5) -> list[dict[str, Any]]:
        return orchestrator.search_evidence_notes(q, top_k)

    @app.get("/api/knowledge/search/graph")
    async def search_graph(q: str, relation_type: str | None = None, top_k: int = 10) -> list[dict[str, Any]]:
        return orchestrator.search_knowledge_graph(q, relation_type, top_k)

    @app.get("/api/knowledge/graph/neighbors")
    async def graph_neighbors(concept: str, depth: int = 1) -> dict[str, Any]:
        return orchestrator.get_knowledge_neighbors(concept, depth)

    @app.get("/api/knowledge/supporting-evidence")
    async def supporting_evidence(id: str, top_k: int = 8) -> list[dict[str, Any]]:
        return orchestrator.get_knowledge_supporting_evidence(id, top_k)

    @app.get("/api/knowledge/next-checks")
    async def next_checks(q: str, top_k: int = 5) -> list[dict[str, Any]]:
        return orchestrator.suggest_knowledge_next_checks(q, top_k)

    @app.post("/api/knowledge-graph/build")
    async def trigger_knowledge_graph_build(request: KnowledgeGraphBuildRequest) -> dict[str, Any]:
        accepted = _start_knowledge_graph_build(app, orchestrator, realtime, request.trigger or "manual", [])
        return {"accepted": accepted, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.post("/api/knowledge-graph/pause")
    async def pause_knowledge_graph_build(request: PauseRequest) -> dict[str, Any]:
        await orchestrator.pause_knowledge_graph_build(request.reason)
        realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})
        return {"bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.post("/api/knowledge-graph/continue")
    async def continue_knowledge_graph_build() -> dict[str, Any]:
        accepted = _start_knowledge_graph_build(app, orchestrator, realtime, "continue", [])
        return {"accepted": accepted, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.get("/api/knowledge-graph/status")
    async def knowledge_graph_status() -> dict[str, Any]:
        return orchestrator.get_knowledge_graph_build_status()

    @app.get("/api/knowledge-graph/log")
    async def knowledge_graph_log() -> list[dict[str, Any]]:
        return orchestrator.get_knowledge_graph_parts()

    @app.get("/api/knowledge-graph/llm-config")
    async def knowledge_graph_llm_config() -> dict[str, Any]:
        return orchestrator.get_knowledge_graph_llm_config()

    @app.post("/api/knowledge-graph/llm-config")
    async def update_knowledge_graph_llm_config(request: KnowledgeGraphLlmConfigRequest) -> dict[str, Any]:
        config = orchestrator.update_knowledge_graph_llm_config(request.model_dump(exclude_none=True))
        return {"config": config, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.get("/api/chain-summary")
    async def chain_summary() -> dict[str, Any]:
        return orchestrator.get_chain_summary()

    @app.get("/api/chain-summary/status")
    async def chain_summary_status() -> dict[str, Any]:
        return orchestrator.get_chain_summary_status()

    @app.get("/api/chain-summary/log")
    async def chain_summary_log() -> list[dict[str, Any]]:
        return orchestrator.get_chain_summary_parts()

    @app.post("/api/chain-summary/build")
    async def trigger_chain_summary_build() -> dict[str, Any]:
        accepted = _start_chain_summary_build(app, orchestrator, realtime, "manual")
        return {"accepted": accepted, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.post("/api/chain-summary/pause")
    async def pause_chain_summary_build(request: PauseRequest) -> dict[str, Any]:
        await orchestrator.pause_chain_summary_build(request.reason)
        realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})
        return {"bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.post("/api/llm-config")
    async def update_main_llm_config(request: MainLlmConfigRequest) -> dict[str, Any]:
        # Saving settings must not construct the legacy harness.  The editor
        # leaves the key blank to keep the current one, so carry the effective
        # key into the workspace file instead of writing an unusable manual
        # config that would shadow the source the key came from.
        values = request.model_dump(exclude_none=True)
        effective = read_effective_llm_config(workspace_path)
        if not values.get("apiKey") and effective.apiKey:
            values["apiKey"] = effective.apiKey
        if values.get("maxTokens") is None and effective.maxTokens is not None:
            values["maxTokens"] = effective.maxTokens
        store = WorkspaceStore(workspace_path)
        store.ensure_layout()
        store.write_main_llm_config(values)
        config = read_effective_llm_config(workspace_path)
        return {
            "config": {"config": mask_llm_config(config), "sdk": {}},
            "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled),
        }

    @app.get("/api/reference-features/status")
    async def reference_feature_status() -> dict[str, Any]:
        return orchestrator.get_reference_feature_status()

    @app.get("/api/reference-features/tool")
    async def reference_feature_tool() -> dict[str, Any]:
        return orchestrator.get_reference_feature_tool()

    @app.post("/api/reference-features/run")
    async def run_reference_feature_tool(request: ReferenceFeatureRunRequest) -> dict[str, Any]:
        return orchestrator.request_extract_reference_features({"input": request.input})

    @app.get("/api/files/tree")
    async def file_tree() -> dict[str, Any]:
        return orchestrator.get_file_tree()

    @app.get("/api/files/content")
    async def file_content(path: str) -> dict[str, Any]:
        try:
            return orchestrator.read_workspace_file(unquote(path))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="file not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from None

    @app.get("/api/files/preview")
    async def file_preview(path: str) -> FileResponse:
        rel_path = unquote(path).strip().lstrip("/")
        target = (workspace_path / rel_path).resolve()
        workspace_root = workspace_path.resolve()
        if target != workspace_root and workspace_root not in target.parents:
            raise HTTPException(status_code=403, detail="preview is limited to workspace files")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if not (media_type.startswith("image/") or media_type == "application/pdf"):
            raise HTTPException(status_code=415, detail="preview only supports images and PDFs")
        response = FileResponse(target, media_type=media_type, filename=target.name)
        response.headers["Content-Disposition"] = f'inline; filename="{target.name}"'
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/api/references/preview")
    async def reference_preview(path: str) -> FileResponse:
        rel_path = unquote(path).strip().lstrip("/")
        target = (workspace_path / rel_path).resolve()
        references_root = (workspace_path / "references").resolve()
        if target != references_root and references_root not in target.parents:
            raise HTTPException(status_code=403, detail="reference preview is limited to references/")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="reference file not found")
        media_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        response = FileResponse(target, media_type=media_type, filename=target.name)
        response.headers["Content-Disposition"] = f'inline; filename="{target.name}"'
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.post("/api/references/upload")
    async def upload_references(files: list[UploadFile] = File(...)) -> dict[str, Any]:
        uploaded: list[dict[str, Any]] = []
        for file in files:
            content = await file.read()
            path = orchestrator.upload_reference_file(file.filename or "reference", content)
            uploaded.append({"path": path, "size": len(content)})
        return {"uploaded": uploaded, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.delete("/api/references")
    async def delete_reference(request: DeleteReferenceRequest) -> dict[str, Any]:
        try:
            path = orchestrator.delete_reference_file(request.path)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="reference file not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from None
        return {"deleted": {"path": path}, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.post("/api/data/raw/upload-zip")
    async def upload_raw_data_zip(file: UploadFile = File(...)) -> dict[str, Any]:
        content = await file.read()
        try:
            result = orchestrator.upload_raw_data_zip(file.filename or "raw-data.zip", content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return {"uploaded": result, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.post("/api/send")
    async def send(request: SendRequest) -> dict[str, Any]:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="text required")
        current = app.state.run_task
        if current is not None and not current.done():
            if orchestrator.execution.snapshot()["phase"] != "paused":
                raise HTTPException(status_code=409, detail="a harness run is already active")
            try:
                await asyncio.wait_for(asyncio.shield(current), timeout=2.0)
            except TimeoutError:
                raise HTTPException(status_code=409, detail="paused execution is still settling; retry shortly") from None
        node_task = orchestrator.active_node_task()
        if node_task is not None and not node_task.done():
            if orchestrator.execution.snapshot()["phase"] != "paused":
                raise HTTPException(status_code=409, detail="a harness run is already active")
            try:
                await asyncio.wait_for(asyncio.shield(node_task), timeout=2.0)
            except TimeoutError:
                raise HTTPException(status_code=409, detail="paused execution is still settling; retry shortly") from None
        phase = orchestrator.execution.snapshot()["phase"]
        # A failed attempt holds no lease; a new prompt starts a fresh attempt.
        if phase not in {"ready", "paused", "failed"}:
            raise HTTPException(status_code=409, detail=f"execution is {phase}; it cannot accept a prompt")
        app.state.run_task = asyncio.create_task(_run_main_turn(orchestrator, text, realtime))
        setattr(orchestrator, "_server_run_task", app.state.run_task)
        realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})
        return {"accepted": True, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.post("/api/execution/pause", status_code=202)
    async def pause_execution(request: PauseRequest, response: Response) -> dict[str, Any]:
        phase = orchestrator.execution.snapshot()["phase"]
        if phase == "paused":
            response.status_code = 200
            return {"accepted": True, "execution": orchestrator.execution.snapshot()}
        if phase == "pausing":
            return {"accepted": True, "execution": orchestrator.execution.snapshot()}
        if phase != "running":
            raise HTTPException(status_code=409, detail=f"execution is {phase}")
        task = app.state.pause_task
        if task is None or task.done():
            task = asyncio.create_task(_run_pause_execution(orchestrator, realtime, request.reason))
            app.state.pause_task = task
        await asyncio.sleep(0)
        realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})
        return {"accepted": True, "execution": orchestrator.execution.snapshot()}

    @app.post("/api/execution/force-stop", status_code=202)
    async def force_stop_execution(request: ForceStopRequest) -> dict[str, Any]:
        if not request.confirm:
            raise HTTPException(status_code=400, detail="confirm=true is required")
        try:
            result = await orchestrator.force_stop_current()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})
        return {"accepted": True, **result}

    @app.post("/api/workflow/reset")
    async def reset_workflow(request: WorkflowResetRequest) -> dict[str, Any]:
        if not request.preserveAssets:
            raise HTTPException(status_code=400, detail="This endpoint only supports preserveAssets=true.")
        if _task_running(app.state.run_task) or orchestrator.is_active_node_task_running():
            raise HTTPException(status_code=409, detail="cannot reset workflow while a harness task is active")
        try:
            state = await orchestrator.reset_workflow_preserving_assets()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})
        return {"ok": True, "state": state, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.post("/api/control/approve")
    async def approve_control() -> dict[str, Any]:
        current = app.state.run_task
        if current is not None and not current.done():
            raise HTTPException(status_code=409, detail="a harness run is already active")
        app.state.run_task = asyncio.create_task(_run_control_approval(orchestrator, realtime))
        setattr(orchestrator, "_server_run_task", app.state.run_task)
        realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})
        return {"accepted": True, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.post("/api/control/reject")
    async def reject_control(request: ControlDecisionRequest) -> dict[str, Any]:
        result = orchestrator.reject_pending_control(request.reason)
        return {"result": result, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    @app.post("/api/debug/clear-logs")
    async def clear_logs(request: ClearLogsRequest) -> dict[str, Any]:
        if not debug_enabled:
            raise HTTPException(status_code=403, detail="debug actions are disabled")
        if request.scope not in {"main", "chat", "all"}:
            raise HTTPException(status_code=400, detail="invalid clear scope")
        if request.scope in {"chat", "all"}:
            if not request.confirmReset:
                raise HTTPException(status_code=400, detail="reset confirmation required")
            if (
                _task_running(app.state.run_task)
                or _task_running(app.state.knowledge_graph_task)
                or _task_running(app.state.chain_summary_task)
                or orchestrator.is_active_node_task_running()
            ):
                raise HTTPException(status_code=409, detail="cannot reset workspace while a run is active")
        await orchestrator.clear_debug_logs(request.scope)
        return {"ok": True, "bootstrap": _bootstrap(orchestrator, workspace_path, dry_run, debug_enabled)}

    if web_root.exists():
        assets_root = web_root / "assets"
        app.mount("/assets", StaticFiles(directory=str(assets_root if assets_root.exists() else web_root)), name="assets")

        @app.get("/{full_path:path}")
        async def static_files(full_path: str) -> FileResponse:
            return FileResponse(_static_file_target(web_root, full_path))

    return app


def _static_file_target(web_root: Path, full_path: str) -> Path:
    """Resolve a request path under web_root; sibling-prefix escapes are rejected."""
    root = web_root.resolve()
    target = (web_root / (full_path or "index.html")).resolve()
    if not target.is_relative_to(root):
        raise HTTPException(status_code=403, detail="Forbidden")
    if target.is_dir():
        target = target / "index.html"
    if not target.exists():
        target = root / "index.html"
    return target


def _bootstrap(orchestrator: Any, workspace_path: Path, dry_run: bool, debug_enabled: bool) -> dict[str, Any]:
    if not getattr(orchestrator, "initialized", True):
        return build_experiments_bootstrap_payload(
            workspace_path=workspace_path, dry_run=dry_run, debug_enabled=debug_enabled,
        )
    return build_bootstrap_payload(
        orchestrator=orchestrator,
        workspace_path=workspace_path,
        dry_run=dry_run,
        debug_enabled=debug_enabled,
        task_running=_task_running,
    )


def _live(orchestrator: HarnessOrchestrator) -> dict[str, Any]:
    return build_live_payload(orchestrator=orchestrator, task_running=_task_running)


async def _run_main_turn(orchestrator: HarnessOrchestrator, text: str, realtime: RealtimeEventBroker) -> None:
    setattr(orchestrator, "_server_run_task", asyncio.current_task())
    try:
        await orchestrator.send_main_user_message(text)
    except RuntimeError as exc:
        if "Paused by user" not in str(exc):
            orchestrator.store.append_timeline({"type": "error", "timestamp": now_iso_for_server(), "message": str(exc)})
            orchestrator.store.append_main_part(system_error_part_for_server(str(exc)))
    except Exception as exc:
        orchestrator.store.append_timeline({"type": "error", "timestamp": now_iso_for_server(), "message": str(exc)})
        orchestrator.store.append_main_part(system_error_part_for_server(str(exc)))
    finally:
        setattr(orchestrator, "_server_run_task", None)
        realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})


async def _run_pause_execution(
    orchestrator: HarnessOrchestrator,
    realtime: RealtimeEventBroker,
    reason: str,
) -> None:
    """Own pause-task completion so failures always reach SSE and are observed."""
    try:
        await orchestrator.interrupt_current(reason)
    except Exception as exc:
        orchestrator.store.append_timeline({
            "type": "execution_pause_failed",
            "timestamp": now_iso_for_server(),
            "message": _format_exception(exc),
        })
    finally:
        realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})


async def _run_control_approval(orchestrator: HarnessOrchestrator, realtime: RealtimeEventBroker) -> None:
    setattr(orchestrator, "_server_run_task", asyncio.current_task())
    try:
        await orchestrator.approve_pending_control()
    except Exception as exc:
        orchestrator.store.append_timeline({"type": "error", "timestamp": now_iso_for_server(), "message": str(exc)})
        orchestrator.store.append_main_part(system_error_part_for_server(str(exc)))
    finally:
        setattr(orchestrator, "_server_run_task", None)
        realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})


def _start_knowledge_graph_build(
    app: FastAPI,
    orchestrator: HarnessOrchestrator,
    realtime: RealtimeEventBroker,
    trigger: str,
    uploaded_paths: list[str],
) -> bool:
    if not orchestrator.variant.knowledge_graph:
        # Raise before the background task exists: the response must report
        # the error instead of accepting a build that can never run.
        raise HTTPException(
            status_code=400,
            detail=f"Knowledge graph building is disabled by ablation variant {orchestrator.variant.id}.",
        )
    current = getattr(app.state, "knowledge_graph_task", None)
    if current is not None and not current.done():
        orchestrator.store.append_timeline({
            "type": "knowledge_graph_build_skipped",
            "timestamp": now_iso_for_server(),
            "message": "Knowledge graph build is already running.",
            "payload": {"trigger": trigger, "uploadedPaths": uploaded_paths},
        })
        return False
    app.state.knowledge_graph_task = asyncio.create_task(
        _run_knowledge_graph_build(orchestrator, realtime, trigger, uploaded_paths)
    )
    setattr(orchestrator, "_server_knowledge_graph_task", app.state.knowledge_graph_task)
    realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})
    return True


async def _run_knowledge_graph_build(
    orchestrator: HarnessOrchestrator,
    realtime: RealtimeEventBroker,
    trigger: str,
    uploaded_paths: list[str],
) -> None:
    setattr(orchestrator, "_server_knowledge_graph_task", asyncio.current_task())
    try:
        await orchestrator.build_knowledge_graph(trigger, uploaded_paths)
    except Exception:
        # The orchestrator records the failure status and timeline event
        # before re-raising; the response was already accepted.
        pass
    finally:
        setattr(orchestrator, "_server_knowledge_graph_task", None)
        realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})


def _start_chain_summary_build(
    app: FastAPI,
    orchestrator: HarnessOrchestrator,
    realtime: RealtimeEventBroker,
    trigger: str,
) -> bool:
    current = getattr(app.state, "chain_summary_task", None)
    if current is not None and not current.done():
        orchestrator.store.append_timeline({
            "type": "chain_summary_build_skipped",
            "timestamp": now_iso_for_server(),
            "message": "Chain builder is already running.",
            "payload": {"trigger": trigger},
        })
        return False
    app.state.chain_summary_task = asyncio.create_task(_run_chain_summary_build(orchestrator, realtime, trigger))
    setattr(orchestrator, "_server_chain_summary_task", app.state.chain_summary_task)
    realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})
    return True


async def _run_chain_summary_build(
    orchestrator: HarnessOrchestrator,
    realtime: RealtimeEventBroker,
    trigger: str,
) -> None:
    setattr(orchestrator, "_server_chain_summary_task", asyncio.current_task())
    try:
        await orchestrator.build_chain_summary(trigger)
    except Exception:
        # The orchestrator records the failure status and timeline event
        # before re-raising; the response was already accepted.
        pass
    finally:
        setattr(orchestrator, "_server_chain_summary_task", None)
        realtime.publish("live_snapshot", {"snapshot": _live(orchestrator)})


def _task_running(task: Any) -> bool:
    return task is not None and not task.done()


def now_iso_for_server() -> str:
    from harnessing_ts.state.workspace_store import now_iso

    return now_iso()


def system_error_part_for_server(message: str) -> dict[str, Any]:
    from harnessing_ts.agent.translate import system_text_part

    part = system_text_part(f"Harness error: {message}")
    part["type"] = "result"
    part["status"] = "failed"
    part["resultText"] = message
    part["raw"] = {"is_error": True, "result": message}
    return part


def _masked_llm_config(workspace_path: Path) -> dict[str, Any]:
    return masked_llm_config(workspace_path)


def _format_exception(exc: Exception) -> str:
    if isinstance(exc, BaseExceptionGroup):
        messages = [str(item) for item in exc.exceptions if str(item)]
        return "; ".join(messages) or str(exc)
    return str(exc) or exc.__class__.__name__


def main() -> None:
    legacy_enabled = _legacy_enabled()
    try:
        host = _startup_host()
        port = _startup_port()
        control_mode = control_mode_from_env() if legacy_enabled else None
        variant = resolve_variant() if legacy_enabled else None
    except RuntimeError as exc:
        _print_startup_error(str(exc), include_variant_help=True)
        raise SystemExit(2) from None
    except ValueError as exc:
        _print_startup_error(str(exc))
        raise SystemExit(2) from None
    print(f"Workspace: {default_workspace_path()}")
    if legacy_enabled:
        print(f"Control mode: {control_mode}")
        print(f"Ablation variant: {variant.id} · {variant.name}")
    else:
        print("Legacy interactive harness: disabled (set TS_HARNESS_LEGACY=true to enable).")
    if os.getenv("TS_HARNESS_DRY_RUN") == "true":
        print("Dry-run mode enabled")
    if os.getenv("TS_HARNESS_DEBUG") == "true":
        print("Debug actions enabled")
    print("Starting the experiments server. The web UI URL will be printed when ready.")
    uvicorn.run("harnessing_ts.server:create_app", host=host, port=port, factory=True)


def _startup_host() -> str:
    host = os.getenv("HOST", "127.0.0.1").strip()
    if not host:
        raise ValueError("Invalid HOST: empty value. Use HOST=127.0.0.1 for local access or HOST=0.0.0.0 for LAN/server access.")
    return host


def _startup_port() -> int:
    raw = os.getenv("PORT", "4327").strip()
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid PORT={raw!r}; expected an integer from 1 to 65535, for example PORT=4327.") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid PORT={raw!r}; expected a TCP port from 1 to 65535, for example PORT=4327.")
    return port


def _print_startup_error(message: str, *, include_variant_help: bool = False) -> None:
    print("HarnessingTS startup configuration error", file=sys.stderr)
    print(message, file=sys.stderr)
    if include_variant_help:
        print("", file=sys.stderr)
        print("Valid variant configuration:", file=sys.stderr)
        print(variant_help_text(), file=sys.stderr)


def _startup_report(message: str) -> None:
    print(message, flush=True)


def _print_ready_urls() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "4327"))
    print(f"HarnessingTS web UI ready: http://{host}:{port}", flush=True)
    if host in {"0.0.0.0", "::"}:
        lan_host = _best_effort_lan_host()
        if lan_host:
            print(f"LAN access URL: http://{lan_host}:{port}", flush=True)


def _best_effort_lan_host() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            host = sock.getsockname()[0]
            return host if host and not host.startswith("127.") else None
    except OSError:
        try:
            host = socket.gethostbyname(socket.gethostname())
            return host if host and not host.startswith("127.") else None
        except OSError:
            return None


if __name__ == "__main__":
    main()
