from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from harnessing_ts.orchestrator import HarnessOrchestrator
from harnessing_ts.settings.llm import (
    SdkInvocationConfig,
    build_sdk_invocation_config,
    mask_llm_config,
    mask_sdk_invocation_config,
    read_effective_llm_config,
)
from harnessing_ts.state.jsonl import read_json


TaskRunning = Callable[[Any], bool]


def build_experiments_bootstrap_payload(
    *, workspace_path: Path, dry_run: bool, debug_enabled: bool,
) -> dict[str, Any]:
    """The experiments console's bootstrap, without constructing the legacy harness.

    The console only reads the workspace state, the masked LLM config and the
    (optional) legacy runtime status.  A missing runtime record means the
    legacy workspace environment was never set up, which is the normal case
    for an experiments-only server.
    """
    workspace_uv = read_json(workspace_path / "state" / "runtime.json")
    if not isinstance(workspace_uv, dict):
        workspace_uv = {}
    workspace_uv.setdefault("workspace", str(workspace_path))
    return {
        "variant": None,
        "state": read_json(workspace_path / "state" / "workspace.json") or {},
        "llmConfig": masked_llm_config(workspace_path),
        "dryRun": dry_run,
        "debugEnabled": debug_enabled,
        "runtime": {
            "running": False,
            "nodeRunning": False,
            "knowledgeGraphRunning": False,
            "chainSummaryRunning": False,
            "pipelineComplete": False,
            "workspaceUv": workspace_uv,
        },
    }


def build_bootstrap_payload(
    *,
    orchestrator: HarnessOrchestrator,
    workspace_path: Path,
    dry_run: bool,
    debug_enabled: bool,
    task_running: TaskRunning,
) -> dict[str, Any]:
    return {
        "variant": orchestrator.get_variant(),
        "state": orchestrator.get_state(),
        "timeline": orchestrator.get_timeline(),
        "mainParts": orchestrator.get_main_parts(),
        "nodes": orchestrator.get_node_sessions(),
        "nodePartsById": orchestrator.get_node_parts_by_id(),
        "nodeSpecs": orchestrator.get_node_specs(),
        "fileTree": orchestrator.get_file_tree(),
        "llmConfig": masked_llm_config(workspace_path),
        "runtimeSettings": orchestrator.get_runtime_settings(),
        "knowledgeGraph": orchestrator.get_knowledge_graph(),
        "knowledgeBaseSummary": orchestrator.get_knowledge_base_summary(),
        "knowledgeGraphParts": orchestrator.get_knowledge_graph_parts(),
        "knowledgeGraphBuild": orchestrator.get_knowledge_graph_build_status(),
        "knowledgeGraphLlmConfig": orchestrator.get_knowledge_graph_llm_config(),
        "referenceFeatureBuild": orchestrator.get_reference_feature_status(),
        "referenceFeatureTool": orchestrator.get_reference_feature_tool(),
        "chainSummary": orchestrator.get_chain_summary(),
        "chainSummaryBuild": orchestrator.get_chain_summary_status(),
        "chainSummaryParts": orchestrator.get_chain_summary_parts(),
        "dryRun": dry_run,
        "debugEnabled": debug_enabled,
        "runtime": _runtime_payload(orchestrator, task_running),
        "execution": orchestrator.execution.snapshot(),
    }


def build_live_payload(
    *,
    orchestrator: HarnessOrchestrator,
    task_running: TaskRunning,
) -> dict[str, Any]:
    return {
        "variant": orchestrator.get_variant(),
        "state": orchestrator.get_state(),
        "timeline": orchestrator.get_timeline(),
        "mainParts": orchestrator.get_main_parts(),
        "nodes": orchestrator.get_node_sessions(),
        "nodePartsById": orchestrator.get_node_parts_by_id(),
        "runtimeSettings": orchestrator.get_runtime_settings(),
        "knowledgeGraph": orchestrator.get_knowledge_graph(),
        "knowledgeBaseSummary": orchestrator.get_knowledge_base_summary(),
        "knowledgeGraphParts": orchestrator.get_knowledge_graph_parts(),
        "knowledgeGraphBuild": orchestrator.get_knowledge_graph_build_status(),
        "knowledgeGraphLlmConfig": orchestrator.get_knowledge_graph_llm_config(),
        "referenceFeatureBuild": orchestrator.get_reference_feature_status(),
        "referenceFeatureTool": orchestrator.get_reference_feature_tool(),
        "chainSummary": orchestrator.get_chain_summary(),
        "chainSummaryBuild": orchestrator.get_chain_summary_status(),
        "chainSummaryParts": orchestrator.get_chain_summary_parts(),
        "runtime": _runtime_payload(orchestrator, task_running),
        "execution": orchestrator.execution.snapshot(),
    }


def masked_llm_config(workspace_path: Path) -> dict[str, Any]:
    cfg = read_effective_llm_config(workspace_path)
    try:
        sdk = build_sdk_invocation_config(cfg)
    except RuntimeError:
        # A manual config without a key is still displayable; the settings
        # page must render so the user can supply the missing key.
        sdk = SdkInvocationConfig(model=cfg.model.strip() or None)
    return {"config": mask_llm_config(cfg), "sdk": mask_sdk_invocation_config(sdk)}


def _runtime_payload(orchestrator: HarnessOrchestrator, task_running: TaskRunning) -> dict[str, Any]:
    pipeline_complete = orchestrator.node_state.is_pipeline_complete(orchestrator.get_state())
    node_running = orchestrator.is_active_node_task_running()
    execution = orchestrator.execution.snapshot()
    return {
        "running": False if pipeline_complete else execution["phase"] in {"running", "pausing", "recovering"},
        "nodeRunning": node_running,
        "knowledgeGraphRunning": task_running(getattr(orchestrator, "_server_knowledge_graph_task", None)),
        "chainSummaryRunning": task_running(getattr(orchestrator, "_server_chain_summary_task", None)),
        "pipelineComplete": pipeline_complete,
        "workspaceUv": orchestrator.get_runtime_status(),
    }
