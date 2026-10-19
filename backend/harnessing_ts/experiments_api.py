"""Experiment-run API: start/stop experiments/batch.py and read its run directory.

The contract is docs/experiments-api.md.  Everything here is plain file reads
plus two kinds of background child process, each in its own session: batch.py
(pid in <run>/batch-process.json) and posthoc.py (pid in
<run>/analysis-process.json).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from harnessing_ts.agent.translate import collapse_tool_parts, filter_display_parts
from harnessing_ts.state.jsonl import read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "experiments" / "runs"
SPECS_DIR = REPO_ROOT / "experiments" / "specs"

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PROCESS_FILE = "batch-process.json"
_BATCH_LOG = "batch-run.log"
_ANALYSIS_PROCESS_FILE = "analysis-process.json"
_ANALYSIS_LOG = "analysis-run.log"
_ANALYSIS_FILE = "analysis.json"
_STATUS_FILE = "batch-status.jsonl"
_LOG_TAIL_LINES = 200
_LOG_TAIL_BYTES = 256 * 1024
_EVENT_TAIL = 50
_STOP_WAIT_SECONDS = 8.0  # batch's own kill grace is 5 s
_TOKEN_KEYS = ("inputTokens", "outputTokens", "cacheReadTokens", "reasoningTokens")

router = APIRouter(prefix="/api/experiments")

# Popen objects of child processes started by this server, by pid.  Polling
# them reaps the exit status, so a finished child never lingers as a zombie
# that os.kill(pid, 0) would still report as alive.
_children: dict[int, subprocess.Popen[bytes]] = {}
_lock = threading.Lock()


# ------------------------------------------------------------------ helpers


def _valid_name(value: Any) -> bool:
    return isinstance(value, str) and _NAME_RE.fullmatch(value) is not None


def _check_name(value: str, what: str) -> str:
    if not _valid_name(value):
        raise HTTPException(status_code=400, detail=f"invalid {what} name: {value!r}")
    return value


def _run_dir(name: str) -> Path:
    path = RUNS_ROOT / _check_name(name, "run")
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"run not found: {name}")
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Object lines of a jsonl file; unparseable lines (a torn last line) are skipped."""
    try:
        return [row for row in read_jsonl(path) if isinstance(row, dict)]
    except (OSError, ValueError):
        return []


def _first_row(path: Path) -> dict[str, Any] | None:
    """The first line of a jsonl file as an object, None when absent or unparseable."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            row = json.loads(handle.readline())
    except (OSError, ValueError):
        return None
    return row if isinstance(row, dict) else None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _tail(path: Path, lines: int = _LOG_TAIL_LINES) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - _LOG_TAIL_BYTES))
            data = handle.read()
    except OSError:
        return ""
    return "\n".join(data.decode("utf-8", errors="replace").splitlines()[-lines:])


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


def _heldout_dir(cell_dir: Path) -> Path | None:
    pointer = _read_json(cell_dir / "heldout.json")
    if isinstance(pointer, dict) and isinstance(pointer.get("path"), str):
        return Path(pointer["path"])
    return None


def _mask(text: str, heldout: Path | None) -> str:
    return text.replace(str(heldout), "<heldout>") if heldout is not None else text


# -------------------------------------------------------------------- specs


def _spec_summary(file: Path, raw: dict[str, Any]) -> dict[str, Any]:
    tasks = [t["name"] for t in raw["tasks"] if isinstance(t, dict) and isinstance(t.get("name"), str)] \
        if isinstance(raw.get("tasks"), list) else []
    arms = [a for a in raw.get("arms", []) if isinstance(a, str)] if isinstance(raw.get("arms"), list) else []
    seeds = [s for s in raw.get("seeds", []) if isinstance(s, int)] if isinstance(raw.get("seeds"), list) else []
    budget = raw.get("budget")
    return {
        "name": raw.get("name") if isinstance(raw.get("name"), str) else file.stem,
        "file": file.name,
        "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
        "cellCount": len(tasks) * len(arms) * len(seeds),
        "tasks": tasks,
        "arms": arms,
        "seeds": seeds,
        "budget": budget if isinstance(budget, int) else 0,
    }


def _load_spec_file(file: Path) -> dict[str, Any] | None:
    """The spec summary for a spec file, None when it is not a matrix spec."""
    raw = _read_json(file)
    if isinstance(raw, dict) and "tasks" in raw:
        return _spec_summary(file, raw)
    return None


@router.get("/specs")
def list_specs() -> list[dict[str, Any]]:
    specs = []
    for file in sorted(SPECS_DIR.glob("*.json")):
        summary = _load_spec_file(file)
        if summary is not None:
            specs.append(summary)
    return specs


# ------------------------------------------------------------------ process


def _pid_alive(pid: int) -> bool:
    """A child of this server is alive until it exits; any other pid must be a
    process-group leader (children start in their own session, so pid == pgid).
    The leader check rules out most reused pids but not one that happens to
    lead its own group."""
    child = _children.get(pid)
    if child is not None:
        return child.poll() is None
    try:
        os.kill(pid, 0)
        return os.getpgid(pid) == pid
    except (ProcessLookupError, PermissionError):
        return False


def _process_record(run_dir: Path, file: str = _PROCESS_FILE) -> dict[str, Any] | None:
    record = _read_json(run_dir / file)
    if isinstance(record, dict) and isinstance(record.get("pid"), int) and record["pid"] > 1:
        return record
    return None


def _process_status(record: dict[str, Any] | None) -> dict[str, Any]:
    if record is None:
        return {"state": "none", "pid": None, "specFile": None, "startedAt": None}
    spec_path = record.get("specPath")
    return {
        "state": "running" if _pid_alive(record["pid"]) else "stopped",
        "pid": record["pid"],
        "specFile": Path(spec_path).name if isinstance(spec_path, str) else None,
        "startedAt": record.get("startedAt"),
    }


def _launch(
    run_dir: Path, command: list[str], log_file: str, log_mode: str, record_file: str, extra: dict[str, Any],
) -> dict[str, Any]:
    """Start command in its own session, output to run_dir/log_file; remember its pid."""
    with (run_dir / log_file).open(log_mode) as log:
        try:
            proc = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                cwd=REPO_ROOT, start_new_session=True,
            )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"could not start process: {exc}") from None
    _children[proc.pid] = proc
    record = {"pid": proc.pid, "startedAt": _iso(time.time()), **extra}
    temp = run_dir / f"{record_file}.tmp"
    temp.write_text(json.dumps(record), encoding="utf-8")
    temp.replace(run_dir / record_file)
    return record


def _batch_command(spec_path: Path, runs_root: Path, workers: int, only: str) -> list[str]:
    command = [
        sys.executable, str(REPO_ROOT / "experiments" / "batch.py"),
        "--spec", str(spec_path), "--runs-root", str(runs_root), "--workers", str(workers),
    ]
    if only:
        command.append(f"--only={only}")
    return command


class StartRequest(BaseModel):
    spec: str = ""
    workers: int = 4
    only: str = ""


@router.post("/runs/{name}/start")
def start_run(name: str, body: StartRequest) -> dict[str, Any]:
    _check_name(name, "run")
    _check_name(body.spec, "spec")
    if not 1 <= body.workers <= 16:
        raise HTTPException(status_code=400, detail="workers must be between 1 and 16")
    spec_path = SPECS_DIR / body.spec
    if spec_path.suffix != ".json" or not spec_path.is_file():
        raise HTTPException(status_code=404, detail=f"spec not found: {body.spec}")
    if _load_spec_file(spec_path) is None:
        raise HTTPException(status_code=400, detail=f"spec has no tasks field: {body.spec}")
    with _lock:
        run_dir = RUNS_ROOT / name
        record = _process_record(run_dir)
        if record is not None and _pid_alive(record["pid"]):
            raise HTTPException(status_code=409, detail=f"run {name} already has a running process")
        previous = _process_status(record)["specFile"]
        if previous is not None and previous != body.spec:
            raise HTTPException(
                status_code=409,
                detail=f"run {name} belongs to spec {previous}; use a new run name for {body.spec}",
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        record = _launch(
            run_dir, _batch_command(spec_path, run_dir, body.workers, body.only), _BATCH_LOG, "ab", _PROCESS_FILE,
            {"specPath": str(spec_path), "workers": body.workers, "only": body.only},
        )
    return {"process": _process_status(record)}


@router.post("/runs/{name}/stop")
def stop_run(name: str) -> dict[str, Any]:
    run_dir = _run_dir(name)
    with _lock:
        record = _process_record(run_dir)
        if record is None or not _pid_alive(record["pid"]):
            raise HTTPException(status_code=409, detail=f"run {name} has no running process")
        try:
            os.killpg(record["pid"], signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        deadline = time.monotonic() + _STOP_WAIT_SECONDS
        while _pid_alive(record["pid"]) and time.monotonic() < deadline:
            time.sleep(0.05)
    return {"process": _process_status(record)}


# ----------------------------------------------------------------- analysis


def _real_csvs(run_dir: Path) -> dict[str, Path]:
    """{task: csv path} for the real tasks of the spec this run was started with."""
    record = _process_record(run_dir)
    spec_path = record.get("specPath") if record else None
    raw = _read_json(SPECS_DIR / Path(spec_path).name) if isinstance(spec_path, str) else None
    tasks = raw.get("tasks") if isinstance(raw, dict) else None
    csvs: dict[str, Path] = {}
    for task in tasks if isinstance(tasks, list) else []:
        if isinstance(task, dict) and task.get("kind") == "real" and isinstance(task.get("csv"), str):
            csvs[task["name"]] = REPO_ROOT / task["csv"]
    return csvs


def _analysis_command(run_dir: Path, real_csvs: dict[str, Path]) -> list[str]:
    command = [
        "uv", "run", "--extra", "research", "python",
        str(REPO_ROOT / "experiments" / "posthoc.py"),
        "--runs", str(run_dir), "--out", str(run_dir / _ANALYSIS_FILE),
    ]
    for task, csv in real_csvs.items():
        command += ["--real-csv", f"{task}={csv}"]
    return command


@router.post("/runs/{name}/analysis")
def start_analysis(name: str) -> dict[str, Any]:
    run_dir = _run_dir(name)
    with _lock:
        record = _process_record(run_dir, _ANALYSIS_PROCESS_FILE)
        if record is not None and _pid_alive(record["pid"]):
            raise HTTPException(status_code=409, detail=f"run {name} already has a running analysis")
        (run_dir / _ANALYSIS_FILE).unlink(missing_ok=True)
        record = _launch(
            run_dir, _analysis_command(run_dir, _real_csvs(run_dir)), _ANALYSIS_LOG, "wb",
            _ANALYSIS_PROCESS_FILE, {},
        )
    return {"process": _process_status(record)}


@router.get("/runs/{name}/analysis")
def get_analysis(name: str) -> dict[str, Any]:
    run_dir = _run_dir(name)
    analysis = _read_json(run_dir / _ANALYSIS_FILE)
    return {
        "process": _process_status(_process_record(run_dir, _ANALYSIS_PROCESS_FILE)),
        "analysis": analysis if isinstance(analysis, dict) else None,
        "logTail": _tail(run_dir / _ANALYSIS_LOG),
    }


# -------------------------------------------------------------------- cells


def _spec_for(record: dict[str, Any] | None) -> dict[str, Any] | None:
    spec_path = record.get("specPath") if record else None
    return _load_spec_file(SPECS_DIR / Path(spec_path).name) if isinstance(spec_path, str) else None


def _cell_parts(cell: str, rows: list[dict[str, Any]]) -> tuple[str, str, int] | None:
    """(task, arm, seed) of a cell found on disk without a spec: <task>-<arm>-<seed>."""
    head, _, seed = cell.rpartition("-")
    arm = rows[0].get("arm") if rows else None
    if not (seed.isascii() and seed.isdigit()) or not isinstance(arm, str) or not head.endswith(f"-{arm}"):
        return None
    return head[: -len(arm) - 1], arm, int(seed)


def _cell_list(run_dir: Path, spec: dict[str, Any] | None) -> list[tuple[str, str, str, int]]:
    """(cell, task, arm, seed): the spec matrix, then any other dir that has rounds."""
    cells: list[tuple[str, str, str, int]] = []
    if spec is not None:
        cells = [
            (f"{task}-{arm}-{seed}", task, arm, seed)
            for task in spec["tasks"] for arm in spec["arms"] for seed in spec["seeds"]
        ]
    known = {cell[0] for cell in cells}
    for child in sorted(run_dir.iterdir()):
        if child.name in known or not child.is_dir() or not (child / "rounds.jsonl").is_file():
            continue
        first = _first_row(child / "rounds.jsonl")
        parts = _cell_parts(child.name, [first] if first else [])
        if parts is not None:
            cells.append((child.name, *parts))
    return cells


def _is_done(cell_dir: Path) -> bool:
    """Same criterion as batch.cell_done: done.json in the heldout dir that heldout.json points to."""
    heldout = _heldout_dir(cell_dir)
    return heldout is not None and (heldout / "done.json").is_file()


def _timestamp(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def _last_events(run_dir: Path) -> dict[str, tuple[str | None, float | None]]:
    """cell -> (event, event time) of the cell's last batch-status line."""
    return {
        row["cell"]: (row.get("event"), _timestamp(row.get("at")))
        for row in _read_jsonl(run_dir / _STATUS_FILE) if "cell" in row
    }


def _cell_state(
    run_dir: Path, cell: str, last_event: tuple[str | None, float | None] | None, started_at: float | None,
) -> str:
    """started_at is the batch process's start time when it is alive, else None."""
    if _is_done(run_dir / cell):
        return "done"
    event, event_at = last_event if last_event is not None else (None, None)
    if started_at is not None:
        try:
            modified = (run_dir / "logs" / f"{cell}.log").stat().st_mtime
        except OSError:
            modified = None
        # Event times have whole-second precision, so compare the log's whole second.
        if modified is not None and modified >= started_at and (event_at is None or int(modified) > event_at):
            return "running"
    return "failed" if event == "fail" else "pending"


def _certifications(cell_dir: Path) -> list[dict[str, Any]]:
    heldout = _heldout_dir(cell_dir)
    rows = _read_jsonl(heldout / "certification.jsonl") if heldout is not None else []
    return [
        {
            "round": row.get("round"),
            "incumbentId": row.get("incumbentId"),
            "saturated": bool(row.get("saturated")),
            "level": row.get("level"),
            "failures": [
                {"label": f.get("label"), "ucb": f.get("ucb"), "threshold": f.get("threshold")}
                for f in row.get("failures") or [] if isinstance(f, dict)
            ],
        }
        for row in rows
    ]


def _incumbent_curve(cell_dir: Path, rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The incumbent's diag MSE after each round, round 0 being m0 (None when m0.json is absent)."""
    heldout = _heldout_dir(cell_dir)
    m0 = _read_json(heldout / "posthoc" / "m0.json") if heldout is not None else None
    mse = m0.get("diagMse") if isinstance(m0, dict) and _number(m0.get("diagMse")) else None
    curve = [{"round": 0, "mse": mse, "accepted": False}]
    for index, row in enumerate(rounds):
        accepted = row.get("accepted") is True
        if accepted and _number(row.get("diagMse")):
            mse = row["diagMse"]
        number = row["round"] if isinstance(row.get("round"), int) else index + 1
        curve.append({"round": number, "mse": mse, "accepted": accepted})
    return curve


def _cell_summary(
    run_dir: Path, cell: tuple[str, str, str, int], budget: int | None,
    last_event: tuple[str | None, float | None] | None, started_at: float | None,
) -> dict[str, Any]:
    cell_id, task, arm, seed = cell
    cell_dir = run_dir / cell_id
    rounds = _read_jsonl(cell_dir / "rounds.jsonl")
    accepted = [row for row in rounds if row.get("accepted") is True]
    diag = [row["diagMse"] for row in rounds if _number(row.get("diagMse"))]
    certs = _certifications(cell_dir)
    return {
        "cell": cell_id, "task": task, "arm": arm, "seed": seed,
        "roundsDone": len(rounds),
        "budget": budget,
        "accepted": len(accepted),
        "incumbent": accepted[-1].get("candidateId", "m0") if accepted else "m0",
        "saturated": certs[-1]["saturated"] if certs else None,
        "state": _cell_state(run_dir, cell_id, last_event, started_at),
        "lastDiagMse": diag[-1] if diag else None,
        "incumbentMse": _incumbent_curve(cell_dir, rounds)[-1]["mse"],
    }


# --------------------------------------------------------------------- runs


@router.get("/runs")
def list_runs() -> list[dict[str, Any]]:
    if not RUNS_ROOT.is_dir():
        return []
    runs = []
    for run_dir in sorted(RUNS_ROOT.iterdir()):
        if not run_dir.is_dir() or not _valid_name(run_dir.name):
            continue
        record = _process_record(run_dir)
        spec = _spec_for(record)
        cells = _cell_list(run_dir, spec)
        stamps = [
            path.stat().st_mtime
            for path in [run_dir / _STATUS_FILE, run_dir / _BATCH_LOG]
            + [run_dir / cell[0] / "rounds.jsonl" for cell in cells]
            if path.is_file()
        ]
        runs.append({
            "name": run_dir.name,
            "process": _process_status(record),
            "cellsDone": sum(1 for cell in cells if _is_done(run_dir / cell[0])),
            "cellsTotal": spec["cellCount"] if spec is not None else None,
            "lastActivityAt": _iso(max(stamps)) if stamps else None,
        })
    return runs


@router.get("/runs/{name}")
def get_run(name: str) -> dict[str, Any]:
    run_dir = _run_dir(name)
    record = _process_record(run_dir)
    process = _process_status(record)
    started_at = _started_at(process)
    spec = _spec_for(record)
    budget = spec["budget"] if spec is not None else None
    last_events = _last_events(run_dir)
    cells = [
        _cell_summary(run_dir, cell, budget, last_events.get(cell[0]), started_at)
        for cell in _cell_list(run_dir, spec)
    ]
    events = []
    for row in _read_jsonl(run_dir / _STATUS_FILE)[-_EVENT_TAIL:]:
        cell = row.get("cell") if isinstance(row.get("cell"), str) else ""
        detail = row.get("detail")
        if isinstance(detail, str) and _valid_name(cell):
            detail = _mask(detail, _heldout_dir(run_dir / cell))
        events.append({
            "event": row.get("event"), "cell": cell, "exitCode": row.get("exitCode"),
            "attempt": row.get("attempt"), "seconds": row.get("seconds"), "at": row.get("at"),
            "detail": detail,
        })
    return {
        "name": name,
        "process": process,
        "spec": spec,
        "cells": cells,
        "batchLogTail": _tail(run_dir / _BATCH_LOG),
        "batchEvents": events,
    }


def _started_at(process: dict[str, Any]) -> float | None:
    """The process start time while it is running, else None."""
    return _timestamp(process["startedAt"]) if process["state"] == "running" else None


def _cell_dir(run_dir: Path, cell: str) -> Path:
    return run_dir / _check_name(cell, "cell")


def _round_row(row: dict[str, Any]) -> dict[str, Any]:
    proposal = row.get("proposal") if isinstance(row.get("proposal"), dict) else {}
    tokens = row.get("tokens") if isinstance(row.get("tokens"), dict) else {}
    return {
        "round": row.get("round"),
        "candidateId": row.get("candidateId"),
        "parent": row.get("parent"),
        "status": row.get("status"),
        "diagMse": row.get("diagMse"),
        "dmP": row.get("dmP"),
        "accepted": row.get("accepted") is True,
        "error": row.get("error"),
        "evalNote": row.get("evalNote"),
        "targetAspect": proposal.get("targetAspect"),
        "inspirationParent": row.get("inspirationParent"),
        "judgeCause": row.get("judgeCause"),
        "judgeFailed": row.get("judgeFailed") is True,
        "tokens": {key: tokens.get(key, 0) for key in _TOKEN_KEYS},
        "judgeTokens": row.get("judgeTokens"),
        "seconds": row.get("seconds", 0),
    }


@router.get("/runs/{name}/cells/{cell}")
def get_cell(name: str, cell: str) -> dict[str, Any]:
    run_dir = _run_dir(name)
    cell_dir = _cell_dir(run_dir, cell)
    record = _process_record(run_dir)
    spec = _spec_for(record)
    entry = next((c for c in _cell_list(run_dir, spec) if c[0] == cell), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"cell not found: {cell}")
    started_at = _started_at(_process_status(record))
    summary = _cell_summary(run_dir, entry, spec["budget"] if spec is not None else None,
                            _last_events(run_dir).get(cell), started_at)
    rounds = _read_jsonl(cell_dir / "rounds.jsonl")
    return {
        "cell": cell, "task": entry[1], "arm": entry[2], "seed": entry[3],
        "state": summary["state"],
        "budget": summary["budget"],
        "rounds": [_round_row(row) for row in rounds],
        "incumbentCurve": _incumbent_curve(cell_dir, rounds),
        "certifications": _certifications(cell_dir),
        "logTail": _mask(_tail(run_dir / "logs" / f"{cell}.log"), _heldout_dir(cell_dir)),
    }


def _posthoc(cell_dir: Path, candidate_id: Any) -> list[dict[str, Any]] | None:
    heldout = _heldout_dir(cell_dir)
    if heldout is None or not isinstance(candidate_id, str) or not _valid_name(candidate_id):
        return None
    stored = _read_json(heldout / "posthoc" / f"{candidate_id}.json")
    card = stored.get("card") if isinstance(stored, dict) else None
    if not isinstance(card, list):
        return None
    return [
        {key: item.get(key) for key in ("label", "aspect", "delta", "lcb", "ucb")}
        for item in card if isinstance(item, dict)
    ]


@router.get("/runs/{name}/cells/{cell}/rounds/{n}")
def get_round(name: str, cell: str, n: str) -> dict[str, Any]:
    run_dir = _run_dir(name)
    cell_dir = _cell_dir(run_dir, cell)
    if not (n.isascii() and n.isdigit()):
        raise HTTPException(status_code=400, detail=f"invalid round number: {n!r}")
    number = int(n)
    row = next((r for r in _read_jsonl(cell_dir / "rounds.jsonl") if r.get("round") == number), None)
    if row is None:
        raise HTTPException(status_code=404, detail=f"round not found: {n}")
    log = _read_jsonl(cell_dir / "ws" / "logs" / f"round-{number}.jsonl")
    return {
        "round": number,
        "proposal": row.get("proposal") if isinstance(row.get("proposal"), dict) else {},
        "posthoc": _posthoc(cell_dir, row.get("candidateId")),
        "parts": filter_display_parts(collapse_tool_parts(log)),
    }
