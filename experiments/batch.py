from __future__ import annotations

"""Parallel driver for the P2 main experiment matrix (final plan §3.6, §7).

One cell = one (task, arm, seed); its run dir is <runs-root>/<task>-<arm>-<seed>.
The driver enumerates the spec's matrix, skips cells that are already finished,
and runs the rest through a fixed pool of loop.py subprocesses.  Failures are
contained: a nonzero exit is recorded in batch-status.jsonl and the batch
moves on, so one bad cell never costs the night's run; every failed cell gets
exactly one retry at the end for transient (LLM tunnel, network) causes.

Resume rules
- done.json in the run dir marks a cell finished and it is never re-invoked.
  loop.py actually writes done.json into the heldout dir whose path
  run_dir/heldout.json records, so that location counts too.
- A cell with rounds.jsonl but no done.json is simply re-invoked: loop.py
  resumes mid-run at the next round (its own per-round checkpointing).
- Killing this batch is safe: children run in their own process groups and are
  killed with it, and the next batch start skips only what finished.

Environment: children inherit the parent environment unchanged (loop.py
reads TS_HARNESS_LLM_* / DEEPSEEK_API_KEY via settings/llm.py and discovers
config.llm.json from the workspace, its cwd, or the project root; the child's
cwd is the repo root).  No byte this driver writes ever carries a secret:
values of environment keys containing KEY/TOKEN/SECRET are masked in the
status log and scrubbed from the per-cell logs after each attempt.

Status log (<runs-root>/batch-status.jsonl), one JSON object per attempt,
appended as attempts finish:
    {"event": "ok"|"fail", "cell": "<task>-<arm>-<seed>", "exitCode": 0,
     "attempt": 1|2, "seconds": 12.3, "at": "<iso8601 UTC>",
     "detail": "<redacted log tail, failures only>"}

Usage:
    uv run --extra research python experiments/batch.py \
        --spec experiments/specs/main-v2.json [--workers 8] [--only task=...,arm=...]
"""

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from harnessing_ts.headroom.config import ARM_VALUES

DEFAULT_LOOP_CMD = "uv run --extra research python -u experiments/loop.py"
DEFAULT_RUNS_ROOT = "experiments/runs/p2"
_DETAIL_TAIL_BYTES = 800
_KILL_GRACE_SECONDS = 5.0
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SECRET_KEY_RE = re.compile(r"KEY|TOKEN|SECRET", re.IGNORECASE)


class BatchError(Exception):
    """Batch-fatal: the spec or the invocation is wrong, so no cell can run."""


# ------------------------------------------------------------------ spec


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str  # "synthetic" | "real"
    csv: Path | None  # absolute path, real tasks only


@dataclass(frozen=True)
class Spec:
    tasks: tuple[TaskSpec, ...]
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    budget: int


def load_spec(path: Path) -> Spec:
    """Parse and validate the frozen matrix spec (any problem is batch-fatal).

    Unknown top-level keys (sizes, ...) are accepted and ignored:
    the loop's CLI takes no sizes flag, so those fields are registry
    bookkeeping, not driver inputs.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BatchError(f"cannot read spec {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BatchError(f"spec {path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise BatchError(f"spec {path} must be a JSON object")
    missing = [key for key in ("tasks", "arms", "seeds", "budget") if key not in raw]
    if missing:
        raise BatchError(f"spec {path} is missing required key(s): {', '.join(missing)}")

    tasks: list[TaskSpec] = []
    names: set[str] = set()
    rows = raw["tasks"]
    if not isinstance(rows, list) or not rows:
        raise BatchError(f"spec {path}: tasks must be a non-empty list")
    for index, row in enumerate(rows):
        where = f"spec {path} tasks[{index}]"
        if not isinstance(row, dict):
            raise BatchError(f"{where} must be an object")
        name = row.get("name")
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise BatchError(f"{where}: name must match {_NAME_RE.pattern}")
        if name in names:
            raise BatchError(f"{where}: duplicate task name {name!r} (run dirs would collide)")
        names.add(name)
        kind = row.get("kind")
        if kind not in ("synthetic", "real"):
            raise BatchError(f"{where}: kind must be 'synthetic' or 'real'")
        csv_path: Path | None = None
        if kind == "real":
            csv_field = row.get("csv")
            if not isinstance(csv_field, str) or not csv_field:
                raise BatchError(f"{where}: real tasks need a non-empty 'csv' path")
            csv_path = Path(csv_field)
            if not csv_path.is_absolute():
                csv_path = _REPO_ROOT / csv_path  # spec paths are repo-root-relative
            if not csv_path.is_file():
                raise BatchError(f"{where}: real csv {csv_path} does not exist")
        tasks.append(TaskSpec(name=name, kind=kind, csv=csv_path))

    arms = raw["arms"]
    if not isinstance(arms, list) or not arms or any(arm not in ARM_VALUES for arm in arms):
        raise BatchError(f"spec {path}: arms must be a non-empty subset of {list(ARM_VALUES)}")
    seeds = raw["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
    ):
        raise BatchError(f"spec {path}: seeds must be a non-empty list of non-negative ints")
    budget = raw["budget"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise BatchError(f"spec {path}: budget must be a positive int")
    if len(set(arms)) != len(arms) or len(set(seeds)) != len(seeds):
        raise BatchError(f"spec {path}: duplicate arms or seeds would collide in run dirs")
    if raw.get("schemaVersion", 1) >= 2:
        # Spec v2 froze a precondition: its G0 gates must be measured with
        # the deep incumbent under the fixed-level stop rule before any v2
        # run.  Refuse to launch until both frozen artifacts exist.
        missing = [
            str(artifact)
            for artifact in (
                _REPO_ROOT / "experiments" / "specs" / "g0-final-report-v3.json",
                _REPO_ROOT / "experiments" / "specs" / "g0-final-report-v3-deep.json",
            )
            if not artifact.is_file()
        ]
        if missing:
            raise BatchError(
                "spec v2 requires the G0 artifacts frozen under the fixed-level stop rule "
                f"first: missing {', '.join(missing)} (run experiments/stage0.py and "
                "experiments/stage0.py --deep)"
            )
    return Spec(tasks=tuple(tasks), arms=tuple(arms), seeds=tuple(seeds), budget=budget)


# ------------------------------------------------------------------ cells


@dataclass(frozen=True)
class Cell:
    task: TaskSpec
    arm: str
    seed: int

    @property
    def cell_id(self) -> str:
        return f"{self.task.name}-{self.arm}-{self.seed}"

    def run_dir(self, runs_root: Path) -> Path:
        return runs_root / self.cell_id


def enumerate_cells(spec: Spec) -> list[Cell]:
    """The full (task, arm, seed) matrix, task-major."""
    return [
        Cell(task=task, arm=arm, seed=seed)
        for task in spec.tasks
        for arm in spec.arms
        for seed in spec.seeds
    ]


def cell_done(cell: Cell, runs_root: Path) -> bool:
    """True when the cell never needs to run again.

    done.json in the run dir is the plan's stated contract; loop.py also
    writes done.json into the heldout dir recorded in run_dir/heldout.json,
    so both locations count.  Anything unreadable counts as unfinished: the
    re-invocation is loop.py's own cheap no-op resume.
    """
    run_dir = cell.run_dir(runs_root)
    if (run_dir / "done.json").is_file():
        return True
    marker = run_dir / "heldout.json"
    if not marker.is_file():
        return False
    try:
        heldout = Path(json.loads(marker.read_text(encoding="utf-8"))["path"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return (heldout / "done.json").is_file()


def parse_only(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def cell_matches(cell: Cell, tokens: list[str]) -> bool:
    """OR semantics, so rerun lists work: `--only a-fixed-3,a-self-built-5` reruns
    exactly those two cells.  A token is a bare substring of the cell id, or
    `key=value` with key in {task, arm, seed, cell} matched as a substring of
    that field (`task=ett,arm=fixed` = every ett cell plus every fixed cell).
    """
    if not tokens:
        return True
    fields = {"task": cell.task.name, "arm": cell.arm, "seed": str(cell.seed), "cell": cell.cell_id}
    for token in tokens:
        if "=" in token:
            key, _, wanted = token.partition("=")
            field = fields.get(key.strip().lower())
            if field is not None and wanted.strip() in field:
                return True
        elif token in cell.cell_id:
            return True
    return False


# ------------------------------------------------- secrets and cell logs


def redact(text: str) -> str:
    """Mask the VALUES of secret-bearing environment keys wherever they occur.

    Children inherit the full parent environment (the loop needs the LLM
    credentials), so a crashing child can echo a key back in a traceback.
    This file's contract is that nothing it writes carries such a value.
    """
    for name, value in os.environ.items():
        if value and _SECRET_KEY_RE.search(name):
            text = text.replace(value, "***")
    return text


def _scrub_log(path: Path) -> None:
    """Rewrite a finished cell log with secret values masked.

    Only touched when a secret value actually occurs, so the common case
    keeps the child's output byte-for-byte.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return
    scrubbed = redact(text)
    if scrubbed != text:
        path.write_text(scrubbed, encoding="utf-8")


def _log_tail(path: Path) -> str:
    """The last chunk of a cell log, for the status record's detail field."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _DETAIL_TAIL_BYTES))
            return redact(handle.read().decode("utf-8", errors="replace"))
    except OSError:
        return ""


# ------------------------------------------------------- cell subprocess


_LIVE_PROCS: list[subprocess.Popen[bytes]] = []
_LIVE_LOCK = threading.Lock()


def kill_live_cells() -> None:
    """SIGTERM every live cell's process group, SIGKILL the stragglers.

    Each child starts its own session, so `uv run` and its python die
    together; an orphaned loop would keep spending LLM budget otherwise.
    """
    with _LIVE_LOCK:
        procs = list(_LIVE_PROCS)
    for proc in procs:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
    deadline = time.monotonic() + _KILL_GRACE_SECONDS
    for proc in procs:
        try:
            proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def spawn_cell(cmd: list[str], run_dir: Path, log_path: Path, attempt: int) -> tuple[int | None, str]:
    """Run one cell's loop process to completion.

    Returns (exit_code, log_tail); exit_code None means the process could not
    even be started, which is a contained cell failure like any other.  The
    child inherits os.environ unchanged and runs with the repo root as cwd
    (so the default relative `experiments/loop.py` and the project-root
    config.llm.json discovery both work).  Its combined output is appended
    to log_path across attempts.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = redact(
        f"\n===== batch attempt {attempt} at "
        f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} | {' '.join(cmd)} =====\n"
    )
    with log_path.open("ab") as log:
        log.write(header.encode("utf-8"))
        log.flush()
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=_REPO_ROOT,
                start_new_session=True,
            )
        except OSError as exc:
            return None, redact(f"could not start the loop process: {exc}")
        with _LIVE_LOCK:
            _LIVE_PROCS.append(proc)
        try:
            code = proc.wait()
        finally:
            with _LIVE_LOCK:
                _LIVE_PROCS.remove(proc)
    _scrub_log(log_path)
    return code, _log_tail(log_path)


# ------------------------------------------------------------------ driver


class SystemicFailure(RuntimeError):
    """Consecutive cell failures that indicate a batch-wide fault."""


@dataclass(frozen=True)
class CellOutcome:
    cell: Cell
    attempt: int
    exit_code: int | None
    seconds: float
    detail: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class BatchDriver:
    """Runs cells through a fixed pool of subprocesses.

    A thread pool is the subprocess pool: worker threads only block on their
    child's exit, so the GIL is irrelevant and exit codes need no pickling.
    All recording happens on the main thread as futures complete, so the
    status log needs no lock.
    """

    def __init__(self, spec: Spec, runs_root: Path, loop_cmd: str, workers: int) -> None:
        self.spec = spec
        self.runs_root = runs_root
        self.loop_cmd = shlex.split(loop_cmd)
        if not self.loop_cmd:
            raise BatchError("--loop-cmd is empty after shell splitting")
        self.workers = workers
        self.status_path = runs_root / "batch-status.jsonl"
        self.log_dir = runs_root / "logs"

    def _attempt(self, cell: Cell, attempt: int) -> CellOutcome:
        """One cell attempt; any exception becomes a contained cell failure."""
        started = time.monotonic()
        try:
            cmd = self.loop_cmd + [
                "--run-dir", str(cell.run_dir(self.runs_root)),
                "--task", cell.task.name,
                "--arm", cell.arm,
                "--seed", str(cell.seed),
                "--budget", str(self.spec.budget),
            ]
            if cell.task.kind == "real":
                assert cell.task.csv is not None
                cmd += ["--real-csv", str(cell.task.csv)]
            code, detail = spawn_cell(
                cmd, cell.run_dir(self.runs_root), self.log_dir / f"{cell.cell_id}.log", attempt
            )
        except Exception as exc:  # a bug in this driver still only loses one cell
            code, detail = None, redact(f"batch worker error: {exc}")
        return CellOutcome(cell, attempt, code, time.monotonic() - started, detail)

    def run_pass(self, cells: list[Cell], attempt: int) -> list[CellOutcome]:
        """One attempt over a set of cells at --workers concurrency."""
        outcomes: list[CellOutcome] = []
        pool = ThreadPoolExecutor(max_workers=self.workers)
        try:
            futures = [pool.submit(self._attempt, cell, attempt) for cell in cells]
            consecutive_failures = 0
            for future in as_completed(futures):
                outcome = future.result()
                outcomes.append(outcome)
                self._report(outcome, done=len(outcomes), total=len(cells))
                consecutive_failures = 0 if outcome.ok else consecutive_failures + 1
                if consecutive_failures >= max(6, self.workers):
                    # A billing/auth failure kills every cell identically;
                    # running the rest of the matrix would only burn time.
                    kill_live_cells()
                    # Cancel the queued cells too: without this they would all
                    # run and fail the same way after the abort.  The finally
                    # below repeats the shutdown, which is a no-op by then.
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise SystemicFailure(
                        f"{consecutive_failures} consecutive cell failures "
                        "(suspect billing/auth); aborting the pass"
                    )
        except KeyboardInterrupt:
            kill_live_cells()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        finally:
            pool.shutdown(wait=True)
        return outcomes

    def _report(self, outcome: CellOutcome, done: int, total: int) -> None:
        record: dict[str, Any] = {
            "event": "ok" if outcome.ok else "fail",
            "cell": outcome.cell.cell_id,
            "exitCode": outcome.exit_code,
            "attempt": outcome.attempt,
            "seconds": round(outcome.seconds, 1),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if not outcome.ok:
            record["detail"] = outcome.detail
        with self.status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        state = "ok" if outcome.ok else "FAIL"
        print(
            f"[batch] {state} {outcome.cell.cell_id} "
            f"(exit {outcome.exit_code}, {outcome.seconds:.1f}s) [{done}/{total}]",
            flush=True,
        )


# -------------------------------------------------------------------- CLI


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec", type=Path, required=True, help="frozen experiment spec JSON")
    parser.add_argument("--workers", type=int, default=8, help="concurrent cells (default 8)")
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated cell substrings, or task=/arm=/seed=/cell= substring "
        "filters; a cell runs if ANY token matches",
    )
    parser.add_argument(
        "--runs-root", type=Path, default=Path(DEFAULT_RUNS_ROOT),
        help=f"run-dir root (default {DEFAULT_RUNS_ROOT})",
    )
    parser.add_argument(
        "--loop-cmd", default=DEFAULT_LOOP_CMD,
        help="command prefix that runs one loop cell (tests substitute a fake)",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    # Paths in this CLI are repo-root-relative regardless of the caller's cwd.
    if not args.spec.is_absolute():
        args.spec = _REPO_ROOT / args.spec
    if not args.runs_root.is_absolute():
        args.runs_root = _REPO_ROOT / args.runs_root
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns 0 (all cells ok), 1 (cell failures remain after
    the retry), 2 (batch-fatal), 130 (interrupted)."""
    try:
        return _main(argv)
    except BatchError as exc:
        print(f"[batch] fatal: {exc}", file=sys.stderr, flush=True)
        return 2
    except SystemicFailure as exc:
        print(f"[batch] aborted: {exc}", file=sys.stderr, flush=True)
        return 3
    except KeyboardInterrupt:
        print(
            "[batch] interrupted; unfinished cells resume on the next start "
            "(cells with done.json are skipped)",
            file=sys.stderr,
            flush=True,
        )
        return 130


def _main(argv: list[str] | None) -> int:
    started = time.monotonic()
    args = _parse_args(argv)
    spec = load_spec(args.spec)
    driver = BatchDriver(spec, args.runs_root, args.loop_cmd, args.workers)
    driver.runs_root.mkdir(parents=True, exist_ok=True)
    driver.log_dir.mkdir(parents=True, exist_ok=True)

    cells = enumerate_cells(spec)
    tokens = parse_only(args.only)
    selected = [cell for cell in cells if cell_matches(cell, tokens)]
    runnable: list[Cell] = []
    done_count = 0
    for cell in selected:
        if cell_done(cell, args.runs_root):
            done_count += 1
        else:
            runnable.append(cell)
    print(
        f"[batch] spec {args.spec.name}: {len(cells)} cells, {len(selected)} selected, "
        f"{done_count} already done, {len(runnable)} to run, workers={args.workers}",
        flush=True,
    )

    failures: list[CellOutcome] = []
    if runnable:
        outcomes = driver.run_pass(runnable, attempt=1)
        # done.json is the truth: a nonzero exit after writing it needs no retry.
        retry = [o.cell for o in outcomes if not o.ok and not cell_done(o.cell, args.runs_root)]
        if retry:
            print(f"[batch] retrying {len(retry)} failed cell(s) once", flush=True)
            failures = [o for o in driver.run_pass(retry, attempt=2) if not o.ok]
    if failures:
        print(f"[batch] {len(failures)} cell(s) failed after the retry:", flush=True)
        for outcome in failures:
            log = driver.log_dir / f"{outcome.cell.cell_id}.log"
            print(
                f"[batch]   {outcome.cell.cell_id} (exit {outcome.exit_code}; see {log})",
                flush=True,
            )
    print(
        f"[batch] finished in {time.monotonic() - started:.1f}s: "
        f"{len(runnable) - len(failures)} ok, {len(failures)} failed, {done_count} skipped",
        flush=True,
    )
    return 1 if failures else 0


def _sigterm_to_interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt


if __name__ == "__main__":
    # `kill <batch>` must behave like Ctrl-C: children live in their own
    # process groups, so a plain SIGTERM would otherwise orphan them mid-round.
    signal.signal(signal.SIGTERM, _sigterm_to_interrupt)
    sys.exit(main())
