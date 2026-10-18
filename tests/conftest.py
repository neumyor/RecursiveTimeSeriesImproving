from __future__ import annotations

import pytest

from harnessing_ts.orchestrator import HarnessOrchestrator
from harnessing_ts.state.workspace_store import WorkspaceStore


@pytest.fixture(autouse=True)
def skip_real_workspace_uv_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests must not materialize a full uv environment for every tmp workspace."""
    monkeypatch.setenv("TS_HARNESS_SKIP_WORKSPACE_UV", "true")


@pytest.fixture
def make_orchestrator():
    """Shared factory for an initialized orchestrator on a tmp workspace."""

    def _make(tmp_path, *, mode: str = "auto", dry_run: bool = True) -> HarnessOrchestrator:
        WorkspaceStore(tmp_path).ensure_layout()
        orchestrator = HarnessOrchestrator(tmp_path, dry_run=dry_run, locale="zh", mode=mode)
        orchestrator.initialize()
        return orchestrator

    return _make
