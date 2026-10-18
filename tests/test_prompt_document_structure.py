from __future__ import annotations

from harnessing_ts.config.markdown import node_document


def test_node_specs_match_mandatory_output_contracts() -> None:
    problem_outputs = node_document("problem-contract")["produces"]
    iterative_outputs = node_document("iterative-solving")["produces"]
    knowledge_outputs = node_document("knowledge-to-tools")["produces"]

    assert problem_outputs == ["user/problem-contract.md", "user/data-spec.md"]
    assert "reports/iterations/<iteration-id>-candidate-review.md" in iterative_outputs
    assert knowledge_outputs
    assert "tools/reference-feature-extractor/**" in knowledge_outputs
    assert "tools/reference-feature-extractor/evidence-map.json" in knowledge_outputs
    assert "tools/reference-feature-extractor/feature-plan.json" in knowledge_outputs
    assert "tools/reference-feature-extractor/evaluation-report.json" in knowledge_outputs
    assert "state/reference-feature-build.json" in knowledge_outputs


def test_node_chain_orders_problem_to_knowledge_to_tools_to_solving_to_summary() -> None:
    from harnessing_ts.config.markdown import node_documents

    chain = [(item["type"], item["next"]) for item in node_documents()]
    assert chain == [
        ("problem-contract", "knowledge-to-tools"),
        ("knowledge-to-tools", "iterative-solving"),
        ("iterative-solving", "final-summary"),
        ("final-summary", None),
    ]
