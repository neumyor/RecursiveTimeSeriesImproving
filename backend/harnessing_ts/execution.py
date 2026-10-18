from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import uuid4

from harnessing_ts.state.workspace_store import WorkspaceStore, now_iso

ExecutionPhase = Literal["ready", "running", "pausing", "paused", "recovering", "failed"]


class PausableRuntime(Protocol):
    session_id: str | None
    checkpoint_seq: int

    async def interrupt(self) -> None: ...
    async def force_stop(self) -> None: ...
    async def recover(self) -> None: ...


def new_execution(actor_id: str, session_id: str | None = None) -> dict[str, Any]:
    return {
        "actorId": actor_id,
        "sessionId": session_id,
        "phase": "ready",
        "epoch": 0,
        "leaseId": None,
        "operationId": None,
        "checkpointSeq": -1,
        "pauseRequestedAt": None,
        "pausedAt": None,
        "recovery": None,
        "lastError": None,
    }


@dataclass
class ExecutionController:
    store: WorkspaceStore

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()
        self._pause_task: asyncio.Task[dict[str, Any]] | None = None

    def snapshot(self) -> dict[str, Any]:
        state = self._state()
        phase = state["phase"]
        return {
            **state,
            "canPause": phase == "running",
            "canForceStop": phase == "pausing",
            "canContinue": phase == "paused",
        }

    def _state(self) -> dict[str, Any]:
        return self.store.read_execution() or new_execution("main")

    def ensure_actor(self, actor_id: str, session_id: str | None) -> dict[str, Any]:
        state = self.store.read_execution()
        if not state or state.get("actorId") != actor_id:
            state = new_execution(actor_id, session_id)
            self.store.write_execution(state)
        elif session_id and not state.get("sessionId"):
            state["sessionId"] = session_id
            self.store.write_execution(state)
        return state

    async def begin(self, actor_id: str, session_id: str) -> dict[str, Any]:
        async with self._lock:
            state = self.ensure_actor(actor_id, session_id)
            # A failed attempt holds no lease (fail() cleared it), so a new
            # prompt starts a fresh attempt in a new epoch instead of locking
            # the session out forever.
            if state["phase"] not in {"ready", "paused", "failed"}:
                raise RuntimeError(f"execution is {state['phase']}; it cannot accept a prompt")
            state.update({
                "sessionId": session_id,
                "phase": "running",
                "epoch": int(state.get("epoch", 0)) + 1,
                "leaseId": uuid4().hex,
                "operationId": uuid4().hex,
                "pauseRequestedAt": None,
                "pausedAt": None,
                "recovery": None,
                "lastError": None,
            })
            self.store.write_execution(state)
            return state

    async def turn_finished(self, *, checkpoint_seq: int | None = None) -> dict[str, Any]:
        async with self._lock:
            state = self._state()
            if state["phase"] == "running":
                state.update({
                    "phase": "ready",
                    "leaseId": None,
                    "operationId": None,
                })
                if checkpoint_seq is not None:
                    state["checkpointSeq"] = checkpoint_seq
                self.store.write_execution(state)
            return state

    async def pause(self, runner: PausableRuntime, reason: str) -> dict[str, Any]:
        async with self._lock:
            state = self._state()
            if state["phase"] == "paused":
                return state
            if state["phase"] == "pausing" and self._pause_task is not None:
                task = self._pause_task
            else:
                if state["phase"] != "running":
                    raise RuntimeError(f"execution is {state['phase']}; there is no running turn to pause")
                operation_id = uuid4().hex
                state.update({
                    "phase": "pausing",
                    "operationId": operation_id,
                    "pauseRequestedAt": now_iso(),
                })
                self.store.write_execution(state)
                task = asyncio.create_task(self._finish_pause(runner, reason, operation_id))
                self._pause_task = task
        return await task

    async def _finish_pause(self, runner: PausableRuntime, reason: str, operation_id: str) -> dict[str, Any]:
        try:
            await runner.interrupt()
            async with self._lock:
                state = self._state()
                if state.get("operationId") == operation_id and state["phase"] == "pausing":
                    checkpoint_seq = getattr(runner, "checkpoint_seq", None)
                    state.update({
                        "phase": "paused",
                        "leaseId": None,
                        "operationId": None,
                        "pausedAt": now_iso(),
                        "lastError": None,
                    })
                    if checkpoint_seq is not None:
                        state["checkpointSeq"] = checkpoint_seq
                    self.store.write_execution(state)
                    self.store.append_timeline({
                        "type": "execution_paused",
                        "timestamp": state["pausedAt"],
                        "message": reason,
                        "payload": {
                            "actorId": state["actorId"],
                            "sessionId": state["sessionId"],
                            "checkpointSeq": state["checkpointSeq"],
                        },
                    })
                return state
        except Exception as exc:
            if self._state().get("phase") == "recovering":
                return self._state()
            await self.fail(exc)
            raise
        finally:
            self._pause_task = None

    async def force_stop(self, runner: PausableRuntime) -> dict[str, Any]:
        async with self._lock:
            state = self._state()
            if state["phase"] != "pausing":
                raise RuntimeError("force stop is only available while an execution is pausing")
            state.update({"phase": "recovering", "recovery": {"kind": "forced-stop", "startedAt": now_iso()}})
            self.store.write_execution(state)
            pause_task = self._pause_task
        try:
            await runner.force_stop()
            # The killed transport releases session/pause. Its cleanup still
            # owns the old runtime, so recovery must not publish a successor
            # until that task has completely retired.
            if pause_task is not None:
                await pause_task
            await runner.recover()
        except Exception as exc:
            await self.fail(exc)
            raise
        async with self._lock:
            state = self._state()
            state.update({"phase": "paused", "leaseId": None, "operationId": None, "pausedAt": now_iso()})
            self.store.write_execution(state)
            return state

    async def fail(self, error: BaseException) -> dict[str, Any]:
        async with self._lock:
            state = self._state()
            state.update({
                "phase": "failed",
                "leaseId": None,
                "operationId": None,
                "lastError": str(error),
            })
            self.store.write_execution(state)
            return state

    def recover_interrupted(self) -> bool:
        state = self.store.read_execution()
        if not state:
            return False
        if state.get("phase") not in {"running", "pausing", "recovering"}:
            return False
        state.update({
            "phase": "recovering",
            "leaseId": None,
            "operationId": None,
            "pausedAt": None,
            "recovery": {"kind": "process-restart", "requiresDshRepairOnContinue": True},
        })
        self.store.write_execution(state)
        return True

    async def recovery_finished(self, *, checkpoint_seq: int | None = None) -> dict[str, Any]:
        async with self._lock:
            state = self._state()
            if state["phase"] != "recovering":
                return state
            state.update({"phase": "paused", "pausedAt": now_iso(), "lastError": None})
            if checkpoint_seq is not None:
                state["checkpointSeq"] = checkpoint_seq
            self.store.write_execution(state)
            return state
