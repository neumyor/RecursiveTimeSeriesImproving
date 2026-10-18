"""Node-protocol recovery tests.

The harness guards against the case where a node runner returns
without calling `mcp__ts_harness__finish_node`. There are two
subcases:

1. The DeepSeek Harness runtime call itself crashes (control-request timeout,
   subprocess error, network drop, etc.). The harness must mark the
   node as failed with the SDK error in the summary, release the
   active-node lock so the next main turn isn't blocked, and re-raise
   the original exception to the HTTP caller. If the user already
   paused the node, the delayed runtime error must preserve the paused
   active node so the user can resume it.

2. The SDK call returns normally but the agent never called
   `finish_node`. The harness must give the agent one bounded reminder
   turn inside the same node session. If the agent still doesn't
   finish, the harness gives up and marks the node as failed so the
   lock is released.

These tests exercise both paths without invoking a live runtime; they
inject a fake SdkRunner into the orchestrator and drive the
`_handle_node_runner_return` flow.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from harnessing_ts.agent.translate import user_text_part
from harnessing_ts.orchestrator import HarnessOrchestrator


def _create_running_node(orchestrator: HarnessOrchestrator, node_type: str = "problem-contract") -> dict:
    node = orchestrator.store.create_node_session(node_type, "test", None)
    node["status"] = "running"
    orchestrator.store.write_node_session(node)
    state = orchestrator.store.read_state()
    state["activeNode"] = node_type
    state["activeNodeSessionId"] = node["id"]
    orchestrator.store.write_state(state)
    orchestrator.state = orchestrator.store.read_state()
    orchestrator.active_node_session = node
    return node


class _FakeNodeRunner:
    """A controllable stand-in for SdkRunner used in the reminder
    path. Records every user message sent to it and lets the test
    decide whether the next turn raises or returns normally."""

    def __init__(self) -> None:
        self.is_running = True
        self.session_id = "test-sdk-session"
        self.user_messages: list[str] = []
        self.raise_on: dict[int, BaseException] = {}
        self._turn = 0

    async def send_with_user_echo(self, text: str, context_text: str | None = None) -> list[dict[str, Any]]:
        self.user_messages.append(text)
        if self._turn in self.raise_on:
            exc = self.raise_on[self._turn]
            self._turn += 1
            raise exc
        self._turn += 1
        # Drop the user part so the caller can append reminder prompts
        # without doubling up on already-logged entries.
        return [user_text_part(text)]

    async def interrupt(self) -> None:
        self.is_running = False

    async def close(self) -> None:
        self.is_running = False


class _InterruptingNodeRunner(_FakeNodeRunner):
    """Raises the transport error only after the pause path shuts it down."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def send_with_user_echo(self, text: str, context_text: str | None = None) -> list[dict[str, Any]]:
        self.user_messages.append(text)
        self.started.set()
        await self.release.wait()
        raise RuntimeError("DeepSeek Harness runtime stdout closed")

    async def interrupt(self) -> None:
        self.is_running = False
        self.release.set()


def test_handle_node_runner_return_reminds_then_gives_up_when_agent_still_does_not_finish(tmp_path, make_orchestrator) -> None:
    orchestrator = make_orchestrator(tmp_path, dry_run=False)
    node = _create_running_node(orchestrator, "problem-contract")
    fake_runner = _FakeNodeRunner()
    orchestrator.active_node_runner = fake_runner  # type: ignore[assignment]

    # First call: no SDK error, status is `running`. The handler
    # should issue a reminder turn inside the same node session and
    # then re-evaluate. Because the fake runner never writes a
    # terminal status, the second pass falls into the give-up branch
    # and marks the node as failed.
    asyncio.run(orchestrator._handle_node_runner_return(node))

    latest = orchestrator.store.read_node_session(node["id"])
    assert latest is not None
    assert latest.get("status") == "failed"
    assert latest.get("success") is False
    assert "reminder" in (latest.get("summary") or "").lower()
    state = orchestrator.get_state()
    assert state["activeNode"] is None
    assert state["activeNodeSessionId"] is None
    assert orchestrator.active_node_runner is None
    # One reminder turn was issued; the second handler pass
    # short-circuited the second reminder and went straight to fail.
    assert len(fake_runner.user_messages) == 1
    assert "finish_node" in fake_runner.user_messages[0].lower()
    timeline = orchestrator.store.read_timeline()
    assert any(event["type"] == "node_protocol_reminder" for event in timeline)
    assert any(
        event["type"] == "node_protocol_error" and "reminder" in event["message"]
        for event in timeline
    )


def test_handle_node_runner_return_fails_after_bounded_reminders(tmp_path, make_orchestrator) -> None:
    orchestrator = make_orchestrator(tmp_path, dry_run=False)
    node = _create_running_node(orchestrator, "problem-contract")
    fake_runner = _FakeNodeRunner()
    orchestrator.active_node_runner = fake_runner  # type: ignore[assignment]
    # Pretend the previous round already used up the reminder budget.
    node["protocolReminders"] = orchestrator._NODE_PROTOCOL_MAX_REMINDERS
    orchestrator.store.write_node_session(node)

    asyncio.run(orchestrator._handle_node_runner_return(node))

    latest = orchestrator.store.read_node_session(node["id"])
    assert latest is not None
    assert latest.get("status") == "failed"
    assert "finish_node" in (latest.get("summary") or "")
    assert latest.get("success") is False
    state = orchestrator.get_state()
    assert state["activeNode"] is None
    assert state["activeNodeSessionId"] is None
    assert orchestrator.active_node_runner is None
    assert orchestrator.active_node_session is None
    timeline = orchestrator.store.read_timeline()
    assert any(
        event["type"] == "node_protocol_error" and "reminder" in event["message"]
        for event in timeline
    )
    # No additional reminder turn was issued.
    assert fake_runner.user_messages == []


def test_handle_node_runner_return_reminder_then_success_clears_protocol_reminders(tmp_path, make_orchestrator) -> None:
    orchestrator = make_orchestrator(tmp_path, mode="manual", dry_run=False)
    node = _create_running_node(orchestrator, "problem-contract")
    fake_runner = _FakeNodeRunner()
    orchestrator.active_node_runner = fake_runner  # type: ignore[assignment]

    # First handler call: status is running → issue reminder.
    asyncio.run(orchestrator._handle_node_runner_return(node))
    # Simulate the agent finally calling finish_node on the reminder
    # turn by writing a terminal status to disk before the second
    # handler call.
    latest = orchestrator.store.read_node_session(node["id"])
    assert latest is not None
    latest["status"] = "completed"
    latest["success"] = True
    latest["summary"] = "reminder worked"
    orchestrator.store.write_node_session(latest)
    state = orchestrator.store.read_state()
    state["activeNode"] = None
    state["activeNodeSessionId"] = None
    orchestrator.store.write_state(state)
    orchestrator.state = orchestrator.store.read_state()

    # Second handler call: status is completed → no further reminder.
    asyncio.run(orchestrator._handle_node_runner_return(node))

    assert len(fake_runner.user_messages) == 1


def test_handle_node_runner_return_sdk_error_marks_failed_and_releases_lock(tmp_path, make_orchestrator) -> None:
    orchestrator = make_orchestrator(tmp_path, dry_run=False)
    node = _create_running_node(orchestrator, "iterative-solving")
    fake_runner = _FakeNodeRunner()
    orchestrator.active_node_runner = fake_runner  # type: ignore[assignment]

    sdk_error = RuntimeError("control request timeout: initialize")
    asyncio.run(orchestrator._handle_node_runner_return(node, sdk_error=sdk_error))

    latest = orchestrator.store.read_node_session(node["id"])
    assert latest is not None
    assert latest.get("status") == "failed"
    assert latest.get("success") is False
    assert "control request timeout" in (latest.get("summary") or "")
    state = orchestrator.get_state()
    assert state["activeNode"] is None
    assert state["activeNodeSessionId"] is None
    assert orchestrator.active_node_runner is None
    assert orchestrator.active_node_session is None
    timeline = orchestrator.store.read_timeline()
    assert any(event["type"] == "node_runner_crash" for event in timeline)
    assert any(event["type"] == "node_protocol_error" for event in timeline)


def test_enter_node_swallows_sdk_exception_after_marking_failure(tmp_path, make_orchestrator) -> None:
    """When the SDK call inside `enter_node` raises, the harness must
    mark the node as failed, release the lock, and re-raise the
    original exception so the HTTP caller still observes it."""

    orchestrator = make_orchestrator(tmp_path, dry_run=False)
    fake_runner = _FakeNodeRunner()
    fake_runner.raise_on[0] = RuntimeError("SDK subprocess crashed")
    # Replace the spawn step so it installs our fake without
    # instantiating a real SdkRunner.
    orchestrator._spawn_node_runner = lambda node: setattr(orchestrator, "active_node_runner", fake_runner)  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="SDK subprocess crashed"):
        asyncio.run(orchestrator.enter_node({
            "nodeType": "problem-contract",
            "rationale": "start workflow",
            "inputSummary": "user request",
        }))

    state = orchestrator.get_state()
    assert state["activeNode"] is None
    sessions = orchestrator.store.list_node_sessions()
    assert len(sessions) == 1
    assert sessions[0]["status"] == "failed"
    assert "SDK subprocess crashed" in (sessions[0].get("summary") or "")
    timeline = orchestrator.store.read_timeline()
    assert any(event["type"] == "node_runner_crash" for event in timeline)


def test_resume_active_node_swallows_sdk_exception_after_marking_failure(tmp_path, make_orchestrator) -> None:
    orchestrator = make_orchestrator(tmp_path, dry_run=False)
    node = _create_running_node(orchestrator, "problem-contract")
    node["status"] = "paused"
    node["summary"] = "Paused by user: needs guidance"
    orchestrator.store.write_node_session(node)

    fake_runner = _FakeNodeRunner()
    fake_runner.raise_on[0] = RuntimeError("SDK subprocess crashed")
    orchestrator._spawn_node_runner = lambda node: setattr(orchestrator, "active_node_runner", fake_runner)  # type: ignore[assignment]

    async def resume_and_wait() -> None:
        result = await orchestrator._resume_active_node("continue with more detail", node)
        assert result == []
        for _ in range(20):
            if not orchestrator.is_active_node_task_running():
                break
            await asyncio.sleep(0.01)

    asyncio.run(resume_and_wait())

    state = orchestrator.get_state()
    assert state["activeNode"] is None
    latest = orchestrator.store.read_node_session(node["id"])
    assert latest is not None
    assert latest.get("status") == "failed"
    assert "SDK subprocess crashed" in (latest.get("summary") or "")


def test_resume_active_node_clears_pause_timestamp(tmp_path, make_orchestrator) -> None:
    orchestrator = make_orchestrator(tmp_path, dry_run=False)
    orchestrator.dry_run = True
    node = _create_running_node(orchestrator, "problem-contract")
    node.update({
        "status": "paused",
        "pausedAt": "2026-08-22T14:50:57Z",
        "summary": "Paused by user.",
    })
    orchestrator.store.write_node_session(node)

    result = asyncio.run(orchestrator._resume_active_node("continue", node))

    assert result
    resumed = orchestrator.store.read_node_session(node["id"])
    assert resumed is not None
    assert resumed["status"] == "running"
    assert "pausedAt" not in resumed
    assert resumed["summary"] == "Resumed with user guidance."


def test_send_main_user_message_unblocked_after_protocol_failure(tmp_path, make_orchestrator) -> None:
    """After a node SDK crash, the next main user message must not be
    rejected with `Main session is locked while node X is active`."""

    orchestrator = make_orchestrator(tmp_path, dry_run=False)
    # Pre-populate the active node lock to simulate the state just
    # before a fresh node entry, then trigger the SDK crash path on
    # the handler.
    node = _create_running_node(orchestrator, "knowledge-to-tools")
    fake_runner = _FakeNodeRunner()
    orchestrator.active_node_runner = fake_runner  # type: ignore[assignment]
    sdk_error = RuntimeError("SDK subprocess crashed")

    asyncio.run(orchestrator._handle_node_runner_return(node, sdk_error=sdk_error))

    state = orchestrator.get_state()
    assert state["activeNode"] is None
    assert state["activeNodeSessionId"] is None


def test_handle_node_runner_return_sdk_error_preserves_user_pause(tmp_path, make_orchestrator) -> None:
    """If the user already paused the node before the SDK exception
    reaches the handler, the harness must not overwrite the pause
    with a ``failed`` status; the user explicitly chose to pause."""

    orchestrator = make_orchestrator(tmp_path, dry_run=False)
    node = _create_running_node(orchestrator, "problem-contract")
    node["status"] = "paused"
    node["summary"] = "Paused by user: needs guidance"
    orchestrator.store.write_node_session(node)
    fake_runner = _FakeNodeRunner()
    orchestrator.active_node_runner = fake_runner  # type: ignore[assignment]

    sdk_error = RuntimeError("Interrupted by user.")
    asyncio.run(orchestrator._handle_node_runner_return(node, sdk_error=sdk_error))

    latest = orchestrator.store.read_node_session(node["id"])
    assert latest is not None
    assert latest.get("status") == "paused"
    assert latest.get("summary") == "Paused by user: needs guidance"
    state = orchestrator.get_state()
    assert state["activeNode"] == "problem-contract"
    assert state["activeNodeSessionId"] == node["id"]
    assert orchestrator.active_node_runner is None
    # The user pause was preserved; the SDK error did not turn it
    # into a failure.
    timeline = orchestrator.store.read_timeline()
    assert not any(event["type"] == "node_runner_crash" for event in timeline)
    assert not any(event["type"] == "node_protocol_error" for event in timeline)


def test_terminal_node_state_wins_over_concurrent_pause_cleanup(tmp_path, make_orchestrator) -> None:
    orchestrator = make_orchestrator(tmp_path, dry_run=False)
    node = _create_running_node(orchestrator, "problem-contract")
    node.update({"status": "completed", "success": True, "summary": "done"})
    orchestrator.store.write_node_session(node)
    state = orchestrator.store.read_state()
    state["activeNode"] = None
    state["activeNodeSessionId"] = None
    orchestrator.store.write_state(state)
    execution = orchestrator.execution.ensure_actor(f"node:{node['id']}", "session-1")
    execution["phase"] = "paused"
    orchestrator.store.write_execution(execution)

    asyncio.run(orchestrator._handle_node_runner_return(
        node,
        sdk_error=RuntimeError("Paused by user."),
        auto_advance=False,
    ))

    latest = orchestrator.store.read_node_session(node["id"])
    assert latest["status"] == "completed"
    assert latest["summary"] == "done"
    assert orchestrator.get_state()["activeNodeSessionId"] is None




def test_protocol_recovery_does_not_corrupt_timeline_jsonl(tmp_path, make_orchestrator) -> None:
    orchestrator = make_orchestrator(tmp_path, dry_run=False)
    node = _create_running_node(orchestrator, "problem-contract")
    fake_runner = _FakeNodeRunner()
    fake_runner.raise_on[0] = RuntimeError("crashed mid turn")
    orchestrator._spawn_node_runner = lambda node: setattr(orchestrator, "active_node_runner", fake_runner)  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        asyncio.run(orchestrator.enter_node({
            "nodeType": "problem-contract",
            "rationale": "start",
            "inputSummary": "go",
        }))

    timeline_path = tmp_path / "logs" / "timeline.jsonl"
    raw = timeline_path.read_text(encoding="utf-8")
    # Every line must be valid JSON; the SDK recovery must not leave
    # partial writes that break the audit log.
    for line in raw.splitlines():
        if not line.strip():
            continue
        json.loads(line)


def test_recover_interrupted_execution_rejects_running_node_without_execution_state(tmp_path, make_orchestrator) -> None:
    """A crash between node entry and the first execution.json write leaves
    a running node with no pause boundary; recovery must refuse to guess."""
    orchestrator = make_orchestrator(tmp_path, dry_run=False)
    _create_running_node(orchestrator, "problem-contract")
    (tmp_path / "state" / "execution.json").unlink(missing_ok=True)

    with pytest.raises(RuntimeError, match="no state/execution.json"):
        orchestrator.recover_interrupted_execution()

    assert orchestrator.store.read_node_session(
        orchestrator.get_state()["activeNodeSessionId"]
    )["status"] == "running"
