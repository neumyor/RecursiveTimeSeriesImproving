from __future__ import annotations

import asyncio

import pytest

from harnessing_ts.orchestrator import HarnessOrchestrator


def test_knowledge_graph_snapshot_contains_live_graph_summary_and_build_status(tmp_path):
    orchestrator = HarnessOrchestrator(tmp_path, dry_run=True)
    orchestrator.initialize()
    events: list[tuple[str, dict]] = []
    orchestrator.set_realtime_event_sink(lambda event_type, payload: events.append((event_type, payload)))

    orchestrator._emit_knowledge_graph_snapshot()

    event_type, payload = events[-1]
    assert event_type == "knowledge_graph_snapshot"
    assert "knowledgeGraph" in payload
    assert "knowledgeBaseSummary" in payload
    assert "knowledgeGraphBuild" in payload


def test_build_knowledge_graph_records_failed_status_for_prebuild_errors(tmp_path, make_orchestrator) -> None:
    """A failure raised before the builder runs (e.g. clearing the graph
    for a rebuild) must land in the build status, not vanish."""
    orchestrator = make_orchestrator(tmp_path, mode="auto")

    def failing_clear() -> None:
        raise OSError("disk full while clearing the knowledge graph")

    orchestrator.store.clear_knowledge_graph = failing_clear  # type: ignore[method-assign]

    with pytest.raises(OSError, match="disk full"):
        asyncio.run(orchestrator.build_knowledge_graph("rebuild"))

    status = orchestrator.get_knowledge_graph_build_status()
    assert status["status"] == "failed"
    assert status["running"] is False
    assert "disk full" in status["message"]
    assert any(
        event["type"] == "knowledge_graph_build_failed"
        for event in orchestrator.store.read_timeline()
    )
