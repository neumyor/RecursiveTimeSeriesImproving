from __future__ import annotations

from pathlib import Path
from typing import Callable

from harnessing_ts.agent.sdk_runner import SdkRunner, SdkRunnerConfig
from harnessing_ts.prompts.compose import (
    PromptContext,
    build_main_system_prompt,
    build_node_attachment,
    build_node_system_prompt,
)
from harnessing_ts.schema import NodeSession, NodeType, Part
from harnessing_ts.settings.llm import read_effective_llm_config, build_sdk_invocation_config
from harnessing_ts.state.message_log import MessageLog
from harnessing_ts.tools.compose_tools import build_main_allowed_tools, build_node_allowed_tools
from harnessing_ts.variants import DEFAULT_VARIANT_ID, AblationVariant, get_variant


def build_main_runner(
    *,
    workspace_path: Path,
    locale: str,
    log_path: Path,
    knowledge_query_ready: bool,
    reference_feature_extractor_ready: bool,
    variant: AblationVariant | None = None,
    on_part: Callable[[Part], None] | None = None,
    on_session_id: Callable[[str], None] | None = None,
    resume_session_id: str | None = None,
) -> SdkRunner:
    variant = variant or get_variant(DEFAULT_VARIANT_ID)
    ctx = PromptContext(str(workspace_path), locale)
    sdk_config = build_sdk_invocation_config(read_effective_llm_config(workspace_path))
    return SdkRunner(SdkRunnerConfig(
        cwd=workspace_path,
        system_prompt=build_main_system_prompt(ctx, variant),
        attachment_text=None,
        allowed_tools=build_main_allowed_tools(
            knowledge_query_ready=knowledge_query_ready,
            reference_feature_extractor_ready=reference_feature_extractor_ready,
            variant=variant,
        ),
        model=sdk_config.model,
        provider=sdk_config.provider,
        max_tokens=sdk_config.max_tokens,
        env=sdk_config.env,
        session_role="main",
        log=MessageLog(log_path),
        on_part=on_part,
        on_session_id=on_session_id,
        resume_session_id=resume_session_id,
    ))


def build_node_runner(
    *,
    workspace_path: Path,
    locale: str,
    node: NodeSession,
    log_path: Path,
    knowledge_query_ready: bool,
    reference_feature_extractor_ready: bool,
    on_session_id: Callable[[str], None],
    on_part: Callable[[Part], None] | None = None,
    variant: AblationVariant | None = None,
    resume_session_id: str | None = None,
) -> SdkRunner:
    variant = variant or get_variant(DEFAULT_VARIANT_ID)
    ctx = PromptContext(str(workspace_path), locale)
    sdk_config = build_sdk_invocation_config(read_effective_llm_config(workspace_path))
    node_type: NodeType = node["nodeType"]
    return SdkRunner(SdkRunnerConfig(
        cwd=workspace_path,
        system_prompt=build_node_system_prompt(node_type, ctx, variant),
        attachment_text=build_node_attachment(node_type, node.get("inputSummary")),
        allowed_tools=build_node_allowed_tools(
            node_type,
            knowledge_query_ready=knowledge_query_ready,
            reference_feature_extractor_ready=reference_feature_extractor_ready,
            variant=variant,
        ),
        model=sdk_config.model,
        provider=sdk_config.provider,
        max_tokens=sdk_config.max_tokens,
        env=sdk_config.env,
        session_role="node",
        node_session_id=node["id"],
        log=MessageLog(log_path),
        on_session_id=on_session_id,
        on_part=on_part,
        resume_session_id=resume_session_id,
    ))
