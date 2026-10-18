from __future__ import annotations

import asyncio

import pytest

from harnessing_ts.execution import ExecutionController
from harnessing_ts.state.workspace_store import WorkspaceStore


class Runtime:
    session_id = "session-1"

    def __init__(self) -> None:
        self.pause_calls = 0
        self.force_calls = 0
        self.release = asyncio.Event()
        self.pause_retired = False
        self.checkpoint_seq = -1

    async def interrupt(self) -> None:
        self.pause_calls += 1
        try:
            await self.release.wait()
        finally:
            self.checkpoint_seq = 12
            self.pause_retired = True

    async def force_stop(self) -> None:
        self.force_calls += 1
        self.release.set()

    async def recover(self) -> None:
        assert self.pause_retired


def controller(tmp_path) -> ExecutionController:
    store = WorkspaceStore(tmp_path)
    store.ensure_layout()
    return ExecutionController(store)


def test_turn_lifecycle_keeps_stable_identity_and_checkpoint(tmp_path) -> None:
    async def run() -> None:
        control = controller(tmp_path)
        started = await control.begin("main", "session-1")
        assert started["phase"] == "running"
        assert started["leaseId"]
        finished = await control.turn_finished(checkpoint_seq=12)
        assert finished["phase"] == "ready"
        assert finished["sessionId"] == "session-1"
        assert finished["checkpointSeq"] == 12
        resumed = await control.begin("main", "session-1")
        assert resumed["epoch"] == 2
    asyncio.run(run())


def test_pause_is_idempotent_and_has_no_timeout(tmp_path) -> None:
    async def run() -> None:
        control = controller(tmp_path)
        runtime = Runtime()
        await control.begin("main", runtime.session_id)
        first = asyncio.create_task(control.pause(runtime, "pause"))
        await asyncio.sleep(0)
        second = asyncio.create_task(control.pause(runtime, "pause again"))
        await asyncio.sleep(0)
        assert control.snapshot()["phase"] == "pausing"
        assert runtime.pause_calls == 1
        runtime.release.set()
        one, two = await asyncio.gather(first, second)
        assert one["phase"] == two["phase"] == "paused"
        assert one["checkpointSeq"] == two["checkpointSeq"] == 12
        assert runtime.pause_calls == 1
        timeline = control.store.read_timeline()
        assert timeline[-1]["payload"]["checkpointSeq"] == 12
    asyncio.run(run())


def test_force_stop_only_exists_while_pausing(tmp_path) -> None:
    async def run() -> None:
        control = controller(tmp_path)
        runtime = Runtime()
        await control.begin("main", runtime.session_id)
        with pytest.raises(RuntimeError, match="only available"):
            await control.force_stop(runtime)
        pause = asyncio.create_task(control.pause(runtime, "pause"))
        await asyncio.sleep(0)
        stopped = await control.force_stop(runtime)
        assert stopped["phase"] == "paused"
        assert runtime.force_calls == 1
        await pause
    asyncio.run(run())


def test_begin_after_failure_starts_a_fresh_attempt(tmp_path) -> None:
    async def run() -> None:
        control = controller(tmp_path)
        first = await control.begin("main", "session-1")
        failed = await control.fail(RuntimeError("transport died mid-turn"))
        assert failed["phase"] == "failed"
        assert failed["leaseId"] is None
        assert failed["operationId"] is None

        # A new prompt on the failed execution starts a fresh attempt:
        # new epoch, new lease, and the failure cleared from view.
        second = await control.begin("main", "session-1")
        assert second["phase"] == "running"
        assert second["leaseId"]
        assert second["leaseId"] != first["leaseId"]
        assert second["epoch"] == first["epoch"] + 1
        assert second["lastError"] is None

        finished = await control.turn_finished(checkpoint_seq=5)
        assert finished["phase"] == "ready"
    asyncio.run(run())


def test_restart_fences_abandoned_lease_without_changing_session(tmp_path) -> None:
    async def start() -> None:
        control = controller(tmp_path)
        await control.begin("node:n1", "session-1")
    asyncio.run(start())
    recovered = controller(tmp_path)
    assert recovered.recover_interrupted() is True
    snapshot = recovered.snapshot()
    assert snapshot["phase"] == "recovering"
    assert snapshot["sessionId"] == "session-1"
    assert snapshot["leaseId"] is None
    asyncio.run(recovered.recovery_finished(checkpoint_seq=9))
    assert recovered.snapshot()["phase"] == "paused"
