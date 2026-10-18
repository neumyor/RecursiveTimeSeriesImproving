from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from harnessing_ts.node_state import NodeStateMachine
from harnessing_ts.prompts.compose import PromptContext, build_main_system_prompt, build_node_system_prompt
from harnessing_ts.tools.compose_tools import build_main_allowed_tools, build_node_allowed_tools
from harnessing_ts.variants import DEFAULT_VARIANT_ID, canonical_variant_id, get_variant, resolve_variant


FULL = DEFAULT_VARIANT_ID
RQA_FULL = "NOD-RQA-KTL-CRV-SUB-ADA"
RQA_NO_CASE = "NOD-RQA-KTL-SUB-ADA"
RQA_NO_TOOLS = "NOD-RQA-CRV-SUB-ADA"
RQA_NO_CASE_NO_TOOLS = "NOD-RQA-SUB-ADA"
FQA_NO_EXEC_CASE = "NOD-FQA-KTL-CXR-SUB-ADA"
DIRECT = "DIR"


def test_variant_registry_exposes_feature_code_profiles(monkeypatch) -> None:
    monkeypatch.delenv("TS_HARNESS_VARIANT", raising=False)
    assert resolve_variant().id == FULL
    assert get_variant("RQA-NOD-KTL-SUB-ADA").id == RQA_NO_CASE
    assert canonical_variant_id("ADA-SUB-KTL-RQA-NOD") == RQA_NO_CASE
    assert get_variant("rqa-nod-ktl-sub-ada").id == RQA_NO_CASE
    with pytest.raises(RuntimeError, match="unknown feature code"):
        get_variant("NOD-XYZ")
    with pytest.raises(RuntimeError, match="DIR cannot be combined"):
        get_variant("DIR-NOD")
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        get_variant("NOD-KGR-RQA")
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        get_variant("NOD-FQA-RQA-KTL")
    with pytest.raises(RuntimeError, match="FQA requires KTL"):
        get_variant("NOD-FQA-CXR-SUB-ADA")
    with pytest.raises(RuntimeError, match="syntactically valid but not a registered"):
        get_variant("NOD-RQA")


def test_rebuilt_variant_capabilities_match_ablation_table() -> None:
    full = get_variant(FULL)
    rqa_full = get_variant(RQA_FULL)
    rqa_no_case = get_variant(RQA_NO_CASE)
    rqa_no_tools = get_variant(RQA_NO_TOOLS)
    rqa_no_case_no_tools = get_variant(RQA_NO_CASE_NO_TOOLS)
    direct = get_variant(DIRECT)
    fqa_no_exec_case = get_variant(FQA_NO_EXEC_CASE)

    assert full.knowledge_graph is True
    assert full.knowledge_query_source == "graph"
    assert full.knowledge_to_tools is True
    assert full.case_review is True

    assert rqa_full.knowledge_graph is False
    assert rqa_full.knowledge_query is True
    assert rqa_full.knowledge_query_source == "references"
    assert rqa_full.knowledge_to_tools is True
    assert rqa_full.case_review is True

    assert rqa_no_case.knowledge_query_source == "references"
    assert rqa_no_case.knowledge_to_tools is True
    assert rqa_no_case.case_review is False

    assert rqa_no_tools.knowledge_query_source == "references"
    assert rqa_no_tools.knowledge_to_tools is False
    assert rqa_no_tools.reference_feature_execution is False
    assert rqa_no_tools.case_review is True

    assert rqa_no_case_no_tools.knowledge_query_source == "references"
    assert rqa_no_case_no_tools.knowledge_to_tools is False
    assert rqa_no_case_no_tools.reference_feature_execution is False
    assert rqa_no_case_no_tools.case_review is False

    assert direct.node_chain is False
    assert direct.knowledge_query is False
    assert direct.reference_feature_execution is False
    assert direct.direct_main_tool_use is True

    assert fqa_no_exec_case.knowledge_graph is False
    assert fqa_no_exec_case.knowledge_query is True
    assert fqa_no_exec_case.knowledge_query_source == "extractor_source"
    assert fqa_no_exec_case.knowledge_to_tools is True
    assert fqa_no_exec_case.reference_feature_source is True
    assert fqa_no_exec_case.reference_feature_extractor is False
    assert fqa_no_exec_case.reference_feature_execution is False
    assert fqa_no_exec_case.case_review is True
    assert fqa_no_exec_case.case_review_reference_features is False


def test_variant_capabilities_change_actual_tool_permissions() -> None:
    assert "mcp__ts_harness__query_knowledge" in build_node_allowed_tools("iterative-solving", variant=get_variant(RQA_FULL))
    assert "mcp__ts_harness__query_knowledge" in build_node_allowed_tools("iterative-solving", variant=get_variant(RQA_NO_TOOLS))
    assert "mcp__ts_harness__query_knowledge" not in build_node_allowed_tools("iterative-solving", variant=get_variant(DIRECT))

    assert "mcp__ts_harness__query_knowledge" in build_main_allowed_tools(variant=get_variant(RQA_FULL))
    assert "mcp__ts_harness__validate_reference_feature_extractor" in build_main_allowed_tools(variant=get_variant(RQA_FULL))
    assert "mcp__ts_harness__validate_reference_feature_extractor" not in build_main_allowed_tools(variant=get_variant(RQA_NO_TOOLS))
    assert "mcp__ts_harness__enter_node" not in build_main_allowed_tools(variant=get_variant(DIRECT))
    assert "bash" in build_main_allowed_tools(variant=get_variant(DIRECT))
    assert "mcp__ts_harness__validate_reference_feature_extractor" in build_main_allowed_tools(variant=get_variant(FQA_NO_EXEC_CASE))
    assert "mcp__ts_harness__extract_reference_features" not in build_main_allowed_tools(
        reference_feature_extractor_ready=True,
        variant=get_variant(FQA_NO_EXEC_CASE),
    )
    assert "mcp__ts_harness__inspect_reference_feature_extractor" not in build_node_allowed_tools(
        "iterative-solving",
        reference_feature_extractor_ready=True,
        variant=get_variant(FQA_NO_EXEC_CASE),
    )


def test_variant_prompt_overlays_are_composed_without_changing_full_profile() -> None:
    context = PromptContext("/tmp/workspace")
    full_main = build_main_system_prompt(context, get_variant(FULL))
    rqa_full_node = build_node_system_prompt("iterative-solving", context, get_variant(RQA_FULL))
    rqa_no_case_node = build_node_system_prompt("iterative-solving", context, get_variant(RQA_NO_CASE))
    rqa_no_tools_node = build_node_system_prompt("iterative-solving", context, get_variant(RQA_NO_TOOLS))
    direct_main = build_main_system_prompt(context, get_variant(DIRECT))
    fqa_no_exec_case_node = build_node_system_prompt("iterative-solving", context, get_variant(FQA_NO_EXEC_CASE))

    assert "## Node Chain" in full_main
    assert "direct reference QA" in rqa_full_node
    assert "No Knowledge Graph + No Case Review" in rqa_no_case_node
    assert "No Knowledge Graph + No Knowledge Tools" in rqa_no_tools_node
    assert "## Node Chain" not in direct_main
    assert "Single-Agent Tool Use" in direct_main
    assert "Extractor Source QA + No Extractor Execution" in fqa_no_exec_case_node


def test_no_case_review_variants_remove_only_case_review_finish_artifact(tmp_path) -> None:
    output_paths = [
        "reports/iterations/001-candidate-review.md",
        "reports/iterations/001-summary.md",
        "user/iteration-state.md",
    ]
    full_output_paths = [
        "reports/iterations/001-candidate-review.md",
        "reports/iterations/001-case-review.md",
        "reports/iterations/001-summary.md",
        "user/iteration-state.md",
    ]
    NodeStateMachine(tmp_path, get_variant(RQA_NO_CASE)).validate_iterative_output_paths(output_paths)
    NodeStateMachine(tmp_path, get_variant(RQA_NO_CASE_NO_TOOLS)).validate_iterative_output_paths(output_paths)

    with pytest.raises(RuntimeError, match="disables case review"):
        NodeStateMachine(tmp_path, get_variant(RQA_NO_CASE)).validate_iterative_output_paths(full_output_paths)
    with pytest.raises(RuntimeError, match="disables case review"):
        NodeStateMachine(tmp_path, get_variant(RQA_NO_CASE_NO_TOOLS)).validate_iterative_output_paths(full_output_paths)

    with pytest.raises(RuntimeError, match="case review"):
        NodeStateMachine(tmp_path, get_variant(RQA_FULL)).validate_iterative_output_paths(output_paths)


def test_single_agent_variant_disables_nodes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TS_HARNESS_VARIANT", DIRECT)
    from harnessing_ts.orchestrator import HarnessOrchestrator

    orchestrator = HarnessOrchestrator(tmp_path, dry_run=True)
    state = orchestrator.initialize()

    assert orchestrator.get_variant()["id"] == DIRECT
    assert state["variantId"] == DIRECT
    assert orchestrator.get_node_specs() == []
    progress = orchestrator._main_progress_snapshot()
    assert progress["recommendedAction"] == "direct_tool_use"
    assert progress["recommendedNode"] is None
    with pytest.raises(RuntimeError, match="disables the HarnessingTS node chain"):
        asyncio.run(orchestrator.request_enter_node({"nodeType": "problem-contract"}))


def test_rqa_variant_disables_graph_builder_but_allows_reference_query_agent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TS_HARNESS_VARIANT", RQA_FULL)
    from harnessing_ts.orchestrator import HarnessOrchestrator

    orchestrator = HarnessOrchestrator(tmp_path, dry_run=False)
    orchestrator.initialize()

    assert orchestrator.get_knowledge_graph() == {}
    assert orchestrator._main_progress_snapshot()["knowledgeQuerySource"] == "references"
    assert orchestrator._main_progress_snapshot()["knowledgeQueryReady"] is True
    knowledge_to_tools_spec = next(spec for spec in orchestrator.get_node_specs() if spec["type"] == "knowledge-to-tools")
    assert knowledge_to_tools_spec["requires"] == [
        "user/problem-contract.md",
        "user/data-spec.md",
        "references/**",
    ]
    with pytest.raises(RuntimeError, match=f"disabled by ablation variant {RQA_FULL}"):
        asyncio.run(orchestrator.build_knowledge_graph())

    with patch("harnessing_ts.orchestrator.answer_reference_query", new=AsyncMock(return_value={"answer": "from refs"})) as query:
        assert asyncio.run(orchestrator.query_knowledge("What matters?"))["answer"] == "from refs"
    query.assert_awaited_once()


def test_knowledge_tools_variants_skip_knowledge_to_tools_node_and_disable_validate_tool(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TS_HARNESS_VARIANT", RQA_NO_TOOLS)
    from harnessing_ts.orchestrator import HarnessOrchestrator

    orchestrator = HarnessOrchestrator(tmp_path, dry_run=True)
    orchestrator.initialize()

    capabilities = orchestrator.get_variant()["capabilities"]
    assert capabilities["knowledgeToTools"] is False
    assert capabilities["referenceFeatureExtractor"] is False
    assert [spec["type"] for spec in orchestrator.get_node_specs()] == [
        "problem-contract",
        "iterative-solving",
        "final-summary",
    ]
    assert orchestrator.get_node_specs()[0]["next"] == "iterative-solving"

    with pytest.raises(RuntimeError, match="disables the knowledge-to-tools node"):
        asyncio.run(orchestrator.enter_node({"nodeType": "knowledge-to-tools", "rationale": "should be blocked"}))

    state = NodeStateMachine(tmp_path, get_variant(RQA_NO_TOOLS))
    node: dict = {
        "id": "pc-1",
        "nodeType": "problem-contract",
        "status": "completed",
        "success": True,
        "nextNode": None,
        "nextNodeSpecified": False,
        "loopDecision": None,
        "goalMet": False,
    }
    assert state.next_node_after_completion(node) == "iterative-solving"


def test_full_variant_routes_problem_contract_to_knowledge_to_tools(tmp_path) -> None:
    state = NodeStateMachine(tmp_path, get_variant(FULL))
    node: dict = {
        "id": "pc-1",
        "nodeType": "problem-contract",
        "status": "completed",
        "success": True,
        "nextNode": None,
        "nextNodeSpecified": False,
        "loopDecision": None,
        "goalMet": False,
    }
    assert state.next_node_after_completion(node) == "knowledge-to-tools"


def test_fqa_variant_uses_extractor_source_query_and_static_validation(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TS_HARNESS_VARIANT", FQA_NO_EXEC_CASE)
    from harnessing_ts.orchestrator import HarnessOrchestrator

    orchestrator = HarnessOrchestrator(tmp_path, dry_run=False)
    orchestrator.initialize()

    assert orchestrator.get_variant()["capabilities"]["referenceFeatureExecution"] is False
    assert orchestrator.get_variant()["capabilities"]["referenceFeatureExtractor"] is False
    assert orchestrator.get_variant()["capabilities"]["referenceFeatureSource"] is True
    assert orchestrator._main_progress_snapshot()["knowledgeQuerySource"] == "extractor_source"
    assert orchestrator._main_progress_snapshot()["knowledgeQueryReady"] is False

    with pytest.raises(RuntimeError, match="source is not ready"):
        asyncio.run(orchestrator.query_knowledge("What rules exist?"))

    with patch(
        "harnessing_ts.orchestrator.answer_extractor_source_query",
        new=AsyncMock(return_value={"answer": "from extractor source"}),
    ) as query:
        orchestrator.store.write_reference_feature_status({"status": "completed"})
        with patch.object(orchestrator.store, "is_reference_feature_extractor_ready", return_value=True):
            assert asyncio.run(orchestrator.query_knowledge("What rules exist?"))["answer"] == "from extractor source"
    query.assert_awaited_once()


def test_fqa_variant_rejects_executable_reference_feature_tools(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TS_HARNESS_VARIANT", FQA_NO_EXEC_CASE)
    from harnessing_ts.orchestrator import HarnessOrchestrator

    orchestrator = HarnessOrchestrator(tmp_path, dry_run=False)
    orchestrator.initialize()

    with pytest.raises(RuntimeError, match=f"disabled by ablation variant {FQA_NO_EXEC_CASE}"):
        orchestrator.request_extract_reference_features({"input": {"case": "x"}})
    assert orchestrator.get_reference_feature_tool() == {}


def test_fqa_case_review_keeps_case_review_but_rejects_reference_feature_usage(tmp_path) -> None:
    state = NodeStateMachine(tmp_path, get_variant(FQA_NO_EXEC_CASE))
    valid_paths = [
        "reports/iterations/001-candidate-review.md",
        "reports/iterations/001-case-review.md",
        "reports/iterations/001-summary.md",
        "user/iteration-state.md",
    ]
    state.validate_iterative_output_paths(valid_paths)
    with pytest.raises(RuntimeError, match="forbids executing or recording reference feature"):
        state.validate_iterative_output_paths([*valid_paths, "runs/iterations/001/reference-feature-usage.json"])


def test_no_case_review_variant_does_not_treat_reference_feature_usage_as_cxr(tmp_path) -> None:
    state = NodeStateMachine(tmp_path, get_variant(RQA_NO_CASE))
    state.validate_iterative_output_paths([
        "reports/iterations/001-candidate-review.md",
        "reports/iterations/001-summary.md",
        "runs/iterations/001/reference-feature-usage.json",
        "user/iteration-state.md",
    ])
