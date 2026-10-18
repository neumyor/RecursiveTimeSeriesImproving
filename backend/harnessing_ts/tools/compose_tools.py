from __future__ import annotations

import os

from harnessing_ts.config.markdown import node_document
from harnessing_ts.schema import NodeType
from harnessing_ts.variants import DEFAULT_VARIANT_ID, AblationVariant, get_variant

READ_ONLY = ["read", "glob", "grep"]
FILE_MUTATION = ["write", "edit"]
SHELL = ["bash"]
SUBAGENT_TOOLS = ["subagent", "subagent_fork", "send_message", "interrupt_agent", "list_agents"]


def _web_tools() -> list[str]:
    tools: list[str] = []
    if os.getenv("TS_HARNESS_ENABLE_WEB_FETCH", "").strip().lower() == "true":
        tools.append("web_fetch")
    if (
        os.getenv("TS_HARNESS_ENABLE_WEB_SEARCH", "").strip().lower() == "true"
        and os.getenv("DEEPSEEK_API_KEY", "").strip()
    ):
        tools.append("web_search")
    return tools


def _native_tools(*, include_subagents: bool) -> list[str]:
    tools = [*READ_ONLY, *FILE_MUTATION, *SHELL, *_web_tools()]
    if include_subagents:
        tools.extend(SUBAGENT_TOOLS)
    return tools


DIRECT_TOOL_USE = _native_tools(include_subagents=True)
MAIN_POOL = ["mcp__ts_harness__enter_node"]
VALIDATE_REFERENCE_FEATURE_TOOL = "mcp__ts_harness__validate_reference_feature_extractor"
REFERENCE_FEATURE_TOOLS = ["mcp__ts_harness__extract_reference_features", "mcp__ts_harness__inspect_reference_feature_extractor"]
NODE_POOL = ["mcp__ts_harness__finish_node", "mcp__ts_harness__get_runtime_settings", "mcp__ts_harness__record_artifact", "mcp__ts_harness__record_run", "mcp__ts_harness__query_knowledge", *REFERENCE_FEATURE_TOOLS]


def build_main_allowed_tools(
    *,
    knowledge_query_ready: bool = False,
    reference_feature_extractor_ready: bool = False,
    variant: AblationVariant | None = None,
) -> list[str]:
    variant = variant or get_variant(DEFAULT_VARIANT_ID)
    if variant.knowledge_query_source == "references":
        knowledge_query_ready = True
    if variant.direct_main_tool_use:
        return sorted(set(_native_tools(include_subagents=variant.independent_subagents)))
    tools = [*READ_ONLY, *MAIN_POOL]
    if knowledge_query_ready and variant.knowledge_query:
        tools.append("mcp__ts_harness__query_knowledge")
    if variant.knowledge_to_tools:
        tools.append(VALIDATE_REFERENCE_FEATURE_TOOL)
    if reference_feature_extractor_ready and variant.reference_feature_extractor:
        tools.extend(REFERENCE_FEATURE_TOOLS)
    return sorted(set(tools))


def build_node_allowed_tools(
    node_type: NodeType,
    *,
    knowledge_query_ready: bool | None = None,
    reference_feature_extractor_ready: bool = False,
    variant: AblationVariant | None = None,
) -> list[str]:
    variant = variant or get_variant(DEFAULT_VARIANT_ID)
    if knowledge_query_ready is None:
        knowledge_query_ready = variant.knowledge_query
    pool = list(NODE_POOL)
    if not variant.knowledge_query or not knowledge_query_ready:
        pool.remove("mcp__ts_harness__query_knowledge")
    if not variant.reference_feature_extractor or not reference_feature_extractor_ready:
        pool = [tool for tool in pool if tool not in REFERENCE_FEATURE_TOOLS]
    if variant.knowledge_to_tools and node_type == "knowledge-to-tools":
        pool.append(VALIDATE_REFERENCE_FEATURE_TOOL)
    return sorted(set(build_node_native_tools(node_type, variant=variant) + pool))


def build_node_native_tools(
    node_type: NodeType,
    *,
    variant: AblationVariant | None = None,
) -> list[str]:
    variant = variant or get_variant(DEFAULT_VARIANT_ID)
    tools = list(node_document(node_type)["native_tools"])
    if not variant.independent_subagents:
        tools = [tool for tool in tools if tool not in SUBAGENT_TOOLS]
    # Web tools are deployment opt-ins.  Do not expose a schema that Cordis
    # intentionally omitted because the provider is not configured.
    enabled_web = set(_web_tools())
    return [tool for tool in tools if tool not in {"web_fetch", "web_search"} or tool in enabled_web]
