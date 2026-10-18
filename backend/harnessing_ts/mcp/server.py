"""MCP bridge for DeepSeek Harness.

DSH starts this module over stdio through its official MCP client
plugin.  Stateful calls carry a per-runner capability back to the live
server-owned orchestrator over the loopback control URL.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from harnessing_ts.schema import NODE_TYPES


def text_result(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _without_intend(args: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in args.items() if key != "intend"}


def _workspace_path() -> Path:
    value = os.getenv("TS_HARNESS_WORKSPACE")
    if not value:
        raise RuntimeError("TS_HARNESS_WORKSPACE is required for the DeepSeek Harness MCP bridge.")
    return Path(value).resolve()


def _control_url() -> str | None:
    value = os.getenv("DSH_MCP_CONTROL_URL", "").strip()
    return value.rstrip("/") if value else None


def _control_capability() -> str | None:
    value = os.getenv("DSH_MCP_CAPABILITY", "").strip()
    return value or None


async def _call_control(tool: str, args: dict[str, Any]) -> Any:
    """Forward a tool call to the live server-owned orchestrator."""
    control_url = _control_url()
    capability = _control_capability()
    if not control_url or not capability:
        raise RuntimeError(
            "DSH_MCP_CONTROL_URL and DSH_MCP_CAPABILITY are required for the "
            "DeepSeek Harness MCP bridge; they are provided by the orchestrator."
        )
    payload = {
        "workspace": str(_workspace_path()),
        "role": _role(),
        "nodeSessionId": os.getenv("TS_HARNESS_DSH_NODE_SESSION_ID") or None,
        "tool": tool,
        "args": args,
    }
    return await asyncio.to_thread(_post_control, control_url, capability, payload)


def _post_control(control_url: str, capability: str, payload: dict[str, Any]) -> Any:
    request = Request(
        f"{control_url}/internal/dsh/mcp",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-TS-Harness-Capability": capability,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=305) as response:  # nosec B310: loopback URL is capability-gated
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("detail") or body
        except json.JSONDecodeError:
            detail = body
        raise RuntimeError(f"HarnessingTS control request failed ({exc.code}): {detail}") from None
    except URLError as exc:
        raise RuntimeError(f"HarnessingTS control endpoint is unavailable: {exc.reason}") from None


def _role() -> str:
    return os.getenv("TS_HARNESS_DSH_SESSION_ROLE", "main")


def create_dsh_mcp_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError(
            "The 'mcp' package is required for the DeepSeek Harness MCP bridge "
            f"(interpreter: {sys.executable})."
        ) from exc

    server = FastMCP("ts_harness")
    role = _role()

    if role == "knowledge_builder":
        _register_knowledge_base_tools(server)
        return server

    _register_role_tools(server, role)
    return server


def _register_role_tools(server: Any, role: str) -> None:
    """Register the harness bridge tools the role may call."""

    async def enter_node(nodeType: str, rationale: str, intend: str, inputSummary: str | None = None, retryFailedNode: bool | None = None) -> str:
        if nodeType not in NODE_TYPES:
            raise ValueError(f"Unknown node type: {nodeType}")
        return text_result(await _call_control("enter_node", _without_intend({
            "nodeType": nodeType, "rationale": rationale, "inputSummary": inputSummary, "retryFailedNode": retryFailedNode, "intend": intend,
        })))

    async def query_knowledge(question: str, intend: str, domain: str | None = None, context: dict[str, Any] | None = None, observations: list[str] | None = None, includeEvidence: bool = False) -> str:
        return text_result(await _call_control("query_knowledge", _without_intend({
            "question": question, "domain": domain, "context": context, "observations": observations, "includeEvidence": includeEvidence, "intend": intend,
        })))

    async def extract_reference_features(input: Any, intend: str) -> str:
        return text_result(await _call_control("extract_reference_features", {"input": input}))

    async def inspect_reference_feature_extractor(intend: str) -> str:
        return text_result(await _call_control("inspect_reference_feature_extractor", {}))

    async def validate_reference_feature_extractor(intend: str, runTests: bool = True) -> str:
        return text_result(await _call_control("validate_reference_feature_extractor", {"runTests": runTests}))

    async def finish_node(success: bool, summary: str, intend: str, goalMet: bool | None = None, nextNode: str | None = None, loopDecision: str | None = None, outputPaths: list[str] | None = None) -> str:
        return text_result(await _call_control("finish_node", _without_intend({
            "success": success, "summary": summary, "goalMet": goalMet, "nextNode": nextNode,
            "loopDecision": loopDecision, "outputPaths": outputPaths or [], "intend": intend,
        })))

    async def record_artifact(path: str, intend: str, summary: str | None = None) -> str:
        return text_result(await _call_control("record_artifact", {"path": path, "summary": summary or ""}))

    async def record_run(runId: str, status: str, startedAt: str, artifactPaths: list[str], intend: str, finishedAt: str | None = None, summary: str | None = None, metrics: dict[str, Any] | None = None) -> str:
        return text_result(await _call_control("record_run", {
            "runId": runId, "status": status, "startedAt": startedAt, "artifactPaths": artifactPaths,
            "finishedAt": finishedAt, "summary": summary or "", "metrics": metrics or {},
        }))

    async def get_runtime_settings(intend: str) -> str:
        return text_result(await _call_control("get_runtime_settings", {}))

    role_tools = {
        "enter_node": ("Request entry into a time-series harness node.", ("main",), enter_node),
        "query_knowledge": (
            "Ask the configured independent knowledge reasoning agent a domain question.",
            ("main", "node"),
            query_knowledge,
        ),
        "extract_reference_features": (
            "Run the validated deterministic reference feature extractor.",
            ("main", "node"),
            extract_reference_features,
        ),
        "inspect_reference_feature_extractor": (
            "Inspect the generated deterministic reference feature extractor.",
            ("main", "node"),
            inspect_reference_feature_extractor,
        ),
        "validate_reference_feature_extractor": (
            "Validate the generated deterministic reference feature extractor.",
            ("main", "node"),
            validate_reference_feature_extractor,
        ),
        "finish_node": ("Request completion of the active harness node.", ("node",), finish_node),
        "record_artifact": ("Record a produced artifact path in the harness timeline.", ("node",), record_artifact),
        "record_run": ("Record a tool-guided solving run.", ("node",), record_run),
        "get_runtime_settings": ("Read live HarnessingTS runtime parameters.", ("node",), get_runtime_settings),
    }
    for name, (description, roles, fn) in role_tools.items():
        if role in roles:
            server.tool(name=name, description=description)(fn)


def _register_knowledge_base_tools(server: Any) -> None:
    from harnessing_ts.knowledge_graph import build_knowledge_base_tool_callbacks

    callbacks = build_knowledge_base_tool_callbacks(_workspace_path())

    def callback(name: str, args: dict[str, Any]) -> str:
        value = callbacks[name](_without_intend(args))
        if inspect.isawaitable(value):
            raise RuntimeError(f"Knowledge-base callback {name} must be synchronous in the stdio bridge.")
        return text_result(value)

    @server.tool(name="scan_references")
    async def scan_references(intend: str, include_processed: bool = False) -> str:
        return callback("scan_references", locals())

    @server.tool(name="extract_reference_text")
    async def extract_reference_text(intend: str, reference_id: str | None = None, path: str | None = None, pages: str | None = None, max_chars_per_page: int | None = None) -> str:
        return callback("extract_reference_text", locals())

    @server.tool(name="update_reference_brief")
    async def update_reference_brief(reference_id: str, brief: str, intend: str, title: str | None = None) -> str:
        return callback("update_reference_brief", locals())

    @server.tool(name="add_evidence")
    async def add_evidence(reference_file: str, quoted_fragments: list[str], intend: str, page: str | None = None, section: str | None = None, notes: str | None = None) -> str:
        return callback("add_evidence", locals())

    @server.tool(name="add_knowledge")
    async def add_knowledge(topic: str, description: str, evidence_ids: list[str], intend: str, summary: str | None = None, notes: str | None = None) -> str:
        return callback("add_knowledge", locals())

    @server.tool(name="list_pending_knowledge")
    async def list_pending_knowledge(intend: str, limit: int = 10) -> str:
        return callback("list_pending_knowledge", locals())

    @server.tool(name="get_knowledge")
    async def get_knowledge(knowledge_id: str, intend: str, detail: str = "compact") -> str:
        return callback("get_knowledge", locals())

    @server.tool(name="search_classes")
    async def search_classes(query: str, intend: str, top_k: int = 10) -> str:
        return callback("search_classes", locals())

    @server.tool(name="upsert_class")
    async def upsert_class(label: str, concept_level: int, concept_type: str, description_addition: str, source_knowledge_ids: list[str], intend: str, aliases: list[str] | None = None, evidence_ids: list[str] | None = None) -> str:
        return callback("upsert_class", locals())

    @server.tool(name="search_relations")
    async def search_relations(intend: str, source_class_id: str | None = None, target_class_id: str | None = None, relation_type: str | None = None) -> str:
        return callback("search_relations", locals())

    @server.tool(name="upsert_relation")
    async def upsert_relation(source_class_id: str, relation_type: str, target_class_id: str, description_addition: str, source_knowledge_ids: list[str], intend: str, relation_depth: int | None = None, evidence_ids: list[str] | None = None) -> str:
        return callback("upsert_relation", locals())

    @server.tool(name="validate_knowledge_base")
    async def validate_knowledge_base(intend: str) -> str:
        return callback("validate_knowledge_base", locals())

    @server.tool(name="finalize_knowledge_base")
    async def finalize_knowledge_base(intend: str) -> str:
        return callback("finalize_knowledge_base", locals())


def main() -> None:
    create_dsh_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
