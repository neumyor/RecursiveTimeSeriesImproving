from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from harnessing_ts.agent.translate import dsh_event_to_parts, user_text_part
from harnessing_ts.paths import REQUIRED_DSH_RUNTIME_FILES
from harnessing_ts.schema import Part
from harnessing_ts.state.message_log import MessageLog


@dataclass
class SdkRunnerConfig:
    cwd: Path
    system_prompt: str
    allowed_tools: list[str]
    log: MessageLog
    attachment_text: str | None = None
    model: str | None = None
    provider: str = "deepseek-official"
    max_tokens: int | None = None
    env: dict[str, str] | None = None
    session_role: str = "main"
    node_session_id: str | None = None
    on_part: Callable[[Part], None] | None = None
    on_session_id: Callable[[str], None] | None = None
    resume_session_id: str | None = None


class SdkRunner:
    """Run one harness session through DeepSeek Harness.

    The official DSH Python SDK is synchronous and owns a reusable JSON-RPC
    runtime subprocess.  This adapter keeps the application-facing async API,
    runs blocking SDK work in a worker thread, and records durable DSH session
    events in the existing workspace transcript format.
    """

    def __init__(self, config: SdkRunnerConfig) -> None:
        self.config = config
        self._pending_attachment = config.attachment_text
        self.session_id: str | None = config.resume_session_id or f"ts-harness-{uuid4().hex}"
        self._harness: Any | None = None
        self._runtime_epoch = 0
        self._checkpointed_epoch = -1
        self._settlement_lock = asyncio.Lock()
        self._running = False
        self._interrupted = False
        self._session_persisted = config.resume_session_id is not None
        self._pause_started = threading.Event()
        self._pause_finished = threading.Event()
        self._pending_user_echo: Part | None = None
        self._accepted_user_echo: Part | None = None
        self.checkpoint_seq = -1

    async def send_with_user_echo(self, text: str, context_text: str | None = None) -> list[Part]:
        user_part = user_text_part(text)
        self._pending_user_echo = user_part
        self._accepted_user_echo = None
        query_text = f"{context_text}\n\n## Current User Message\n{text}" if context_text else text
        try:
            assistant_parts = await self.send(query_text)
        finally:
            # session/open or prompt rejection must not leave a local user turn
            # that DSH never accepted into its append-only inbox.
            self._pending_user_echo = None
        return [self._accepted_user_echo or user_part, *assistant_parts]

    async def send(self, text: str) -> list[Part]:
        content = f"{self._pending_attachment}\n\n{text}" if self._pending_attachment else text
        self._pending_attachment = None
        if self._harness is None:
            self._harness = self._new_harness()
            self._runtime_epoch += 1
        harness = self._harness
        runtime_epoch = self._runtime_epoch
        self._running = True
        self._interrupted = False
        self._pause_started.clear()
        self._pause_finished.clear()
        try:
            await asyncio.to_thread(self._open_session, harness)
            self._commit_session_identity()
        except BaseException:
            self._running = False
            await self._close_harness(harness)
            raise
        loop = asyncio.get_running_loop()
        streamed_events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        emitted_event_keys: set[str] = set()
        parts: list[Part] = []

        def on_notification(notification: Any) -> None:
            """Move DSH notifications from its worker thread onto this loop."""
            event = _dsh_session_event(notification, self.session_id)
            if event is None:
                return
            try:
                loop.call_soon_threadsafe(streamed_events.put_nowait, event)
            except RuntimeError:
                # The owning event loop is shutting down.  The final SDK
                # result still supplies the persisted events where possible.
                return

        def append_event(event: dict[str, Any]) -> None:
            key = _dsh_event_key(event)
            if key in emitted_event_keys:
                return
            if event.get("type") == "agent/inbox/spliced" and self._pending_user_echo is not None:
                self._accepted_user_echo = self._append_part(self._pending_user_echo)
                self._pending_user_echo = None
                emitted_event_keys.add(key)
                return
            sequence = event.get("seq")
            stable_id = (
                f"dsh:{self.session_id}:{sequence}"
                if isinstance(sequence, (str, int))
                else None
            )
            event_parts = dsh_event_to_parts(event, part_id=stable_id)
            if not event_parts:
                return
            emitted_event_keys.add(key)
            for part in event_parts:
                persisted_part = self._append_part(part)
                _replace_or_append_part(parts, persisted_part)

        def drain_streamed_events() -> None:
            while True:
                try:
                    append_event(streamed_events.get_nowait())
                except asyncio.QueueEmpty:
                    return

        run_task = asyncio.create_task(asyncio.to_thread(
            harness.run,
            content,
            session_id=self.session_id,
            on_notification=on_notification,
        ))
        try:
            # DeepSeekHarness.run is synchronous, but it invokes
            # on_notification for every JSON-RPC notification while the turn
            # is still active.  Consume those notifications immediately so
            # main, node, knowledge-graph, and chain-summary UIs all receive
            # tool and assistant progress during a long-running turn.
            while not run_task.done():
                event_task = asyncio.create_task(streamed_events.get())
                done, _pending = await asyncio.wait(
                    {run_task, event_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if event_task in done:
                    append_event(event_task.result())
                else:
                    event_task.cancel()
                    await asyncio.gather(event_task, return_exceptions=True)
            result = await run_task
            # Let callbacks queued by the worker immediately before its
            # completion run, then drain them before falling back to the
            # complete result event list.
            await asyncio.sleep(0)
            drain_streamed_events()
        except BaseException:
            await asyncio.sleep(0)
            drain_streamed_events()
            if self._pause_started.is_set():
                await asyncio.to_thread(self._pause_finished.wait)
            # The DSH client owns the child process.  Do not reuse a runtime
            # whose transport or turn has failed; the next send starts cleanly.
            await self._close_harness(harness)
            self._running = False
            raise

        try:
            if self._pause_started.is_set():
                await asyncio.to_thread(self._pause_finished.wait)
            elif not self._interrupted:
                wait_for_pause = False
                async with self._settlement_lock:
                    # interrupt() publishes the intent before waiting for this
                    # lock. Recheck it here so checkpoint and session/pause can
                    # never race on the same JSON-RPC transport.
                    if self._pause_started.is_set():
                        wait_for_pause = True
                    else:
                        await asyncio.to_thread(self._checkpoint_session, harness)
                        self._checkpointed_epoch = runtime_epoch
                        self._session_persisted = True
                if wait_for_pause:
                    await asyncio.to_thread(self._pause_finished.wait)
        finally:
            await self._close_harness(harness)
            self._running = False
        if self._interrupted:
            raise RuntimeError("Paused by user.")

        # The result is authoritative and also covers SDK/runtime versions
        # that do not deliver every event through on_notification.
        for event in getattr(result, "events", []):
            if isinstance(event, dict):
                append_event(event)
        return parts

    @property
    def is_running(self) -> bool:
        return self._running

    def adopt_recovery_session(self, session_id: str) -> None:
        """Bind a freshly reconstructed runner to the interrupted identity."""
        if self._running or self._harness is not None:
            raise RuntimeError("cannot replace the DSH session identity of an active runtime")
        self.session_id = session_id
        self._session_persisted = True

    async def interrupt(self) -> None:
        """Durably pause the active DSH turn at its native cancellation boundary."""
        self._interrupted = True
        self._pause_started.set()
        harness = self._harness
        runtime_epoch = self._runtime_epoch
        try:
            async with self._settlement_lock:
                if self._checkpointed_epoch == runtime_epoch:
                    # The turn reached a normal durable checkpoint immediately
                    # before the pause request acquired the settlement lock.
                    pass
                elif harness is None or self.session_id is None:
                    raise RuntimeError("cannot pause without an active or checkpointed DSH runtime")
                elif self._harness is not harness:
                    raise RuntimeError("DSH runtime ownership changed while pausing")
                else:
                    await asyncio.to_thread(self._pause_session, harness)
                    self._checkpointed_epoch = runtime_epoch
                    self._session_persisted = True
        finally:
            await self._close_harness(harness)
            self._pause_finished.set()

    async def force_stop(self) -> None:
        """Terminate an unresponsive runtime; the next open must use DSH resume."""
        self._interrupted = True
        harness = self._harness
        if harness is None:
            return
        proc = getattr(getattr(harness, "client", None), "_proc", None)
        if proc is not None and proc.poll() is None:
            process_group = _owned_process_group(proc)
            if process_group is not None:
                os.killpg(process_group, signal.SIGTERM)
            else:
                proc.terminate()
            try:
                await asyncio.to_thread(proc.wait, 2.0)
            except subprocess.TimeoutExpired:
                if process_group is not None:
                    os.killpg(process_group, signal.SIGKILL)
                else:
                    proc.kill()
                await asyncio.to_thread(proc.wait)
        # Killing the process unblocks session/pause. Wait until interrupt()
        # has released the old runtime before recovery creates a successor.
        if self._pause_started.is_set():
            await asyncio.to_thread(self._pause_finished.wait)
        await self._close_harness(harness)

    async def recover(self) -> None:
        """Resume the same durable identity so DSH can repair a torn boundary."""
        if self.session_id is None:
            raise RuntimeError("cannot recover without a DSH session identity")
        async with self._settlement_lock:
            self._session_persisted = True
            harness = self._new_harness()
            self._harness = harness
            self._runtime_epoch += 1
            runtime_epoch = self._runtime_epoch
            try:
                await asyncio.to_thread(self._open_session, harness, recovery=True)
                self._commit_session_identity()
                await asyncio.to_thread(self._checkpoint_session, harness)
                self._checkpointed_epoch = runtime_epoch
            finally:
                await self._close_harness(harness)

    def _new_harness(self) -> Any:
        try:
            from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig
        except Exception as exc:
            raise RuntimeError(
                "DeepSeek Harness Python SDK is not installed. Run `uv sync` first. "
                f"Original error: {exc}"
            ) from exc
        return DeepSeekHarness(DeepSeekHarnessConfig(
            provider=self.config.provider,
            model=self.config.model or "deepseek-v4-flash",
            max_tokens=self.config.max_tokens,
            cwd=str(self.config.cwd),
            runtime_cwd=str(_dsh_runtime_root()),
            runtime_bin=str(_dsh_runtime_bin()),
            session_root=str(self.config.cwd / ".dsh-sessions"),
            cordis=str(_cordis_config_path()),
            env=self._runtime_env(),
            shutdown_timeout_seconds=30.0,
            launch_args_override=_runtime_launch_args(),
        ))

    async def close(self) -> None:
        await self._close_harness()

    async def _close_harness(self, expected: Any | None = None) -> None:
        harness = self._harness if expected is None else expected
        if expected is not None and self._harness is not expected:
            return
        self._harness = None
        if harness is not None:
            try:
                await asyncio.to_thread(harness.close)
            except Exception:
                pass

    def _open_session(self, harness: Any, *, recovery: bool = False) -> None:
        if self.session_id is None:
            raise RuntimeError("DSH runtime or session identity is unavailable")
        harness.start()
        mode = "recover" if recovery else "resume" if self._session_persisted else "create"
        response = harness.client.request(
            "session/open",
            {
                "sessionId": self.session_id,
                "mode": mode,
            },
            response_model=_SessionOpenResponse,
        )
        expected = {"created", "resumed"} if recovery else {"resumed" if self._session_persisted else "created"}
        if response.source not in expected:
            raise RuntimeError(f"DSH session/open returned {response.source}; expected one of {sorted(expected)}")
        self._session_persisted = True

    def _commit_session_identity(self) -> None:
        if self.session_id is not None and self.config.on_session_id:
            self.config.on_session_id(self.session_id)

    def _checkpoint_session(self, harness: Any) -> None:
        if self.session_id is None:
            raise RuntimeError("cannot checkpoint a closed DSH runtime")
        response = harness.client.request(
            "session/checkpoint",
            {"sessionId": self.session_id},
            response_model=_SessionCheckpointResponse,
        )
        self.checkpoint_seq = response.checkpointSeq

    def _pause_session(self, harness: Any) -> None:
        if self.session_id is None:
            raise RuntimeError("cannot pause without a DSH session identity")
        response = harness.client.request(
            "session/pause",
            {"sessionId": self.session_id},
            response_model=_SessionCheckpointResponse,
        )
        self.checkpoint_seq = response.checkpointSeq

    def _append_part(self, part: Part) -> Part:
        persisted_part = self.config.log.append(part)
        if self.config.on_part:
            self.config.on_part(persisted_part)
        return persisted_part

    def _runtime_env(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.config.env:
            env.update(self.config.env)
        env.update({
            "DSH_SYSTEM_PROMPT": self.config.system_prompt,
            "TS_HARNESS_DSH_SESSION_ROLE": self.config.session_role,
            "TS_HARNESS_DSH_ALLOWED_TOOLS": ",".join(self.config.allowed_tools),
            "TS_HARNESS_DSH_PYTHON": os.sys.executable,
            "TS_HARNESS_WORKSPACE": str(self.config.cwd),
        })
        if self.config.node_session_id:
            env["TS_HARNESS_DSH_NODE_SESSION_ID"] = self.config.node_session_id
        venv = self.config.cwd / ".venv"
        bin_dir = venv / "bin"
        if venv.exists():
            env["VIRTUAL_ENV"] = str(venv)
            if bin_dir.exists():
                current_path = env.get("PATH", "")
                env["PATH"] = os.pathsep.join(dict.fromkeys([str(bin_dir), *filter(None, current_path.split(os.pathsep))]))
        else:
            env.pop("VIRTUAL_ENV", None)
        env["UV_PROJECT"] = str(self.config.cwd)
        return env


def _cordis_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "deepseek-harness" / "cordis.yml"


def _dsh_runtime_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "dsh_runtime"
    required = [root / rel for rel in REQUIRED_DSH_RUNTIME_FILES]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "DeepSeek Harness runtime extension is incomplete "
            f"({', '.join(missing)}). Run `ts-harness setup-server` or "
            "`npm ci --prefix backend/dsh_runtime` from the repository root."
        )
    return root


def _dsh_runtime_bin() -> Path:
    root = _dsh_runtime_root()
    entry = root / "node_modules" / "@deepseek-ai" / "dsh-sdk-jsonrpc-demo" / "lib" / "bin.js"
    if not entry.is_file():  # guarded by _dsh_runtime_root; this check only narrows the type
        raise RuntimeError("DeepSeek Harness JSON-RPC runtime entrypoint is missing.")
    return entry


def _runtime_launch_args() -> tuple[str, ...] | None:
    setsid = shutil.which("setsid")
    if setsid is None:
        return None
    return (setsid, str(_dsh_runtime_bin()))


def _owned_process_group(proc: subprocess.Popen[Any]) -> int | None:
    try:
        group = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        return None
    return group if group == proc.pid else None


def _dsh_session_event(notification: Any, session_id: str | None) -> dict[str, Any] | None:
    """Extract a root-session DSH event from a streaming notification."""
    if session_id is None or getattr(notification, "method", None) != "session.event":
        return None
    payload = getattr(notification, "payload", None)
    if not isinstance(payload, dict) or payload.get("sessionId") != session_id:
        return None
    event = payload.get("event")
    return event if isinstance(event, dict) else None


def _dsh_event_key(event: dict[str, Any]) -> str:
    """Return a stable per-turn de-duplication key for DSH events."""
    sequence = event.get("seq")
    if isinstance(sequence, (str, int)):
        return f"seq:{sequence}"
    # Runtime events normally carry seq.  For the small number that do not,
    # object identity is a stable key because the SDK hands the notification
    # callback and the final result list the same event object.
    return f"object:{id(event)}"


def _replace_or_append_part(parts: list[Part], part: Part) -> None:
    """Keep a tool result as an update to its original tool-call card."""
    part_id = part.get("id")
    if isinstance(part_id, str):
        for index, existing in enumerate(parts):
            if existing.get("id") == part_id:
                parts[index] = part
                return
    parts.append(part)


class _SessionOpenResponse(BaseModel):
    sessionId: str
    source: str
    lastSeq: int


class _SessionCheckpointResponse(BaseModel):
    sessionId: str
    checkpointSeq: int
    turnEnd: dict[str, Any] | None = None
