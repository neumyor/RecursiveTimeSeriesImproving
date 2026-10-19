"""Tests for the experiment-run API (backend/harnessing_ts/experiments_api.py).

A fake runs root and specs dir live under tmp_path.  Most tests replace the
batch command with a tiny python script that prints a line and sleeps; the
lifecycle tests run the real experiments/batch.py with a fake loop script, so
stop, resume and --only exercise the real process tree without any LLM.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from harnessing_ts import experiments_api as api

FAKE_BATCH = [sys.executable, "-c", "import time; print('fake batch up', flush=True); time.sleep(60)"]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _jsonl(path: Path, rows: list[dict]) -> None:
    _write(path, "".join(json.dumps(row) + "\n" for row in rows))


def _round(
    n: int, cid: str, *, accepted: bool, diag: float | None = 1.0, aspect: str | None = "lag", arm: str = "fixed",
) -> dict:
    return {
        "round": n, "candidateId": cid, "parent": "m0", "arm": arm, "status": "evaluated",
        "diagMse": diag, "dmP": 0.5, "accepted": accepted, "error": None, "evalNote": None,
        "proposal": {"candidateId": cid, "targetAspect": aspect, "mechanism": "m"},
        "tokens": {"inputTokens": 10, "outputTokens": 20, "cacheReadTokens": 1, "reasoningTokens": 0},
        "inspirationParent": None, "judgeCause": None, "judgeFailed": False, "judgeTokens": None,
        "seconds": 1.5,
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    specs = tmp_path / "specs"
    heldout = tmp_path / "secret-heldout" / "abc"
    monkeypatch.setattr(api, "RUNS_ROOT", runs)
    monkeypatch.setattr(api, "SPECS_DIR", specs)
    monkeypatch.setattr(api, "_batch_command", lambda spec, root, workers, only: FAKE_BATCH)

    _write(specs / "tiny.json", json.dumps({
        "name": "tiny-spec", "registeredAt": "2026-10-19", "description": "two tasks",
        "tasks": [{"name": "t1", "kind": "synthetic"}, {"name": "t2", "kind": "synthetic"}],
        "arms": ["fixed", "self-built"], "seeds": [0], "budget": 3,
    }))
    _write(specs / "report.json", json.dumps({"gate": "g0"}))

    run = runs / "r1"
    done = run / "t1-fixed-0"
    _jsonl(done / "rounds.jsonl", [
        _round(1, "c1", accepted=True, diag=1.0),
        _round(2, "c2", accepted=False, diag=0.8),
        _round(3, "c3", accepted=True, diag=0.5),
    ])
    _write(done / "heldout.json", json.dumps({"path": str(heldout)}))
    _jsonl(heldout / "certification.jsonl", [
        {"round": 1, "incumbentId": "m0", "saturated": False, "level": 0.95,
         "failures": [{"label": "total", "ucb": 0.04, "threshold": 0.02}]},
        {"round": 3, "incumbentId": "c3", "saturated": True, "level": 0.95, "failures": []},
    ])
    _write(heldout / "done.json", json.dumps({"rounds": 3}))
    _write(heldout / "posthoc" / "m0.json", json.dumps({"round": 0, "proposal": {}, "diagMse": 1.2, "card": []}))
    _write(heldout / "posthoc" / "c1.json", json.dumps({
        "round": 1, "proposal": {}, "diagMse": 1.0, "dmP": 0.5,
        "card": [{"label": "total", "aspect": "other", "delta": 0.2, "lcb": -0.1, "ucb": 0.5, "mu2": 1e-4,
                  "by_horizon": [0.1, 0.2]}],
    }))
    _jsonl(done / "ws" / "logs" / "round-1.jsonl", [
        {"id": "p1", "timestamp": "t", "role": "assistant", "type": "text", "text": "hello",
         "raw": {"type": "assistant/message"}},
        {"id": "p2", "timestamp": "t", "role": "assistant", "type": "tool_call", "name": "read",
         "input": {"file_path": "a.py"}, "intend": "read", "toolUseId": "x", "status": "completed",
         "text": "read", "resultText": "file body", "raw": {"type": "tool/call"}, "resultRaw": {"big": 1}},
    ])
    _write(run / "logs" / "t1-fixed-0.log", f"line one\nheldout at {heldout}\n")

    partial = run / "t1-self-built-0"
    _write(partial / "rounds.jsonl",
           json.dumps(_round(1, "c1", accepted=False, diag=None, arm="self-built")) + "\n" + '{"round": 2, "cand')
    _write(partial / "heldout.json", json.dumps({"path": str(heldout) + "-partial"}))
    _jsonl(run / "batch-status.jsonl", [
        {"event": "ok", "cell": "t1-fixed-0", "exitCode": 0, "attempt": 1, "seconds": 3.0, "at": "a"},
        {"event": "fail", "cell": "t1-self-built-0", "exitCode": 1, "attempt": 1, "seconds": 1.0,
         "at": "b", "detail": f"crashed reading {heldout}-partial"},
    ])
    _write(run / "batch-run.log", "".join(f"batch line {i}\n" for i in range(300)))

    app = FastAPI()
    app.include_router(api.router)
    client = TestClient(app)
    yield client, run, heldout
    _kill_children()


def _kill_children() -> None:
    for pid in list(api._children):
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        api._children[pid].wait()
    api._children.clear()


def _finished_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _stopped_record(run: Path) -> None:
    _write(run / "batch-process.json", json.dumps({
        "pid": _finished_pid(), "specPath": str(api.SPECS_DIR / "tiny.json"),
        "startedAt": "2026-10-19T00:00:00+00:00", "workers": 2, "only": "",
    }))


def test_specs_lists_only_matrix_specs(env) -> None:
    client, _, _ = env
    response = client.get("/api/experiments/specs")
    assert response.status_code == 200
    assert response.json() == [{
        "name": "tiny-spec", "file": "tiny.json", "registeredAt": "2026-10-19",
        "description": "two tasks", "cellCount": 4,
        "tasks": ["t1", "t2"], "arms": ["fixed", "self-built"], "seeds": [0], "budget": 3,
    }]


def test_runs_list_summarises_each_run(env) -> None:
    client, run, _ = env
    (api.RUNS_ROOT / "empty").mkdir()
    (api.RUNS_ROOT / "stray.txt").write_text("x")
    _stopped_record(run)
    runs = {r["name"]: r for r in client.get("/api/experiments/runs").json()}
    assert set(runs) == {"r1", "empty"}
    assert runs["r1"]["process"]["state"] == "stopped"
    assert runs["r1"]["cellsDone"] == 1
    assert runs["r1"]["cellsTotal"] == 4
    assert runs["r1"]["lastActivityAt"]
    assert runs["empty"]["process"] == {"state": "none", "pid": None, "specFile": None, "startedAt": None}
    assert runs["empty"]["cellsTotal"] is None
    assert runs["empty"]["lastActivityAt"] is None


def test_run_snapshot_cell_states_and_summaries(env) -> None:
    client, run, heldout = env
    _stopped_record(run)
    body = client.get("/api/experiments/runs/r1").json()
    assert body["process"]["state"] == "stopped"
    assert body["process"]["specFile"] == "tiny.json"
    assert body["spec"]["file"] == "tiny.json"
    cells = {c["cell"]: c for c in body["cells"]}
    assert list(cells) == ["t1-fixed-0", "t1-self-built-0", "t2-fixed-0", "t2-self-built-0"]
    assert cells["t1-fixed-0"]["state"] == "done"
    assert cells["t1-self-built-0"]["state"] == "failed"
    assert cells["t2-fixed-0"]["state"] == "pending"
    done = cells["t1-fixed-0"]
    assert (done["roundsDone"], done["accepted"], done["incumbent"], done["budget"]) == (3, 2, "c3", 3)
    assert done["saturated"] is True
    assert (done["lastDiagMse"], done["incumbentMse"]) == (0.5, 0.5)
    partial = cells["t1-self-built-0"]
    assert partial["roundsDone"] == 1  # torn last line skipped
    assert (partial["incumbent"], partial["saturated"], partial["lastDiagMse"]) == ("m0", None, None)
    assert partial["incumbentMse"] is None  # no m0.json behind the partial cell
    assert cells["t2-fixed-0"]["roundsDone"] == 0
    assert body["batchLogTail"].splitlines() == [f"batch line {i}" for i in range(100, 300)]
    assert [e["event"] for e in body["batchEvents"]] == ["ok", "fail"]
    assert body["batchEvents"][0]["detail"] is None
    assert str(heldout) not in json.dumps(body)
    assert "<heldout>" in body["batchEvents"][1]["detail"]


def test_run_without_process_record_still_lists_cells_on_disk(env) -> None:
    client, _, _ = env
    body = client.get("/api/experiments/runs/r1").json()
    assert body["process"]["state"] == "none"
    assert body["spec"] is None
    cells = {c["cell"]: c for c in body["cells"]}
    assert set(cells) == {"t1-fixed-0", "t1-self-built-0"}
    assert cells["t1-fixed-0"]["budget"] is None
    assert (cells["t1-fixed-0"]["task"], cells["t1-fixed-0"]["arm"], cells["t1-fixed-0"]["seed"]) == ("t1", "fixed", 0)


def test_cell_detail(env) -> None:
    client, _, heldout = env
    body = client.get("/api/experiments/runs/r1/cells/t1-fixed-0").json()
    assert (body["cell"], body["task"], body["arm"], body["seed"], body["state"]) == (
        "t1-fixed-0", "t1", "fixed", 0, "done")
    assert [r["round"] for r in body["rounds"]] == [1, 2, 3]
    first = body["rounds"][0]
    assert first["targetAspect"] == "lag"
    assert first["accepted"] is True
    assert first["tokens"] == {"inputTokens": 10, "outputTokens": 20, "cacheReadTokens": 1, "reasoningTokens": 0}
    assert body["certifications"][0] == {
        "round": 1, "incumbentId": "m0", "saturated": False, "level": 0.95,
        "failures": [{"label": "total", "ucb": 0.04, "threshold": 0.02}],
    }
    assert body["certifications"][1]["saturated"] is True
    assert "line one" in body["logTail"]
    assert str(heldout) not in json.dumps(body)


def test_cell_detail_defaults_missing_fields(env) -> None:
    client, run, _ = env
    _jsonl(run / "t1-self-built-0" / "rounds.jsonl", [{"round": 1, "arm": "self-built", "candidateId": "c1", "accepted": False}])
    row = client.get("/api/experiments/runs/r1/cells/t1-self-built-0").json()["rounds"][0]
    assert row["targetAspect"] is None
    assert row["tokens"] == {"inputTokens": 0, "outputTokens": 0, "cacheReadTokens": 0, "reasoningTokens": 0}
    assert row["judgeFailed"] is False


def test_round_detail_has_public_parts_and_posthoc(env) -> None:
    client, _, heldout = env
    body = client.get("/api/experiments/runs/r1/cells/t1-fixed-0/rounds/1").json()
    assert body["round"] == 1
    assert body["proposal"]["candidateId"] == "c1"
    assert body["posthoc"] == [{"label": "total", "aspect": "other", "delta": 0.2, "lcb": -0.1, "ucb": 0.5}]
    assert [p["type"] for p in body["parts"]] == ["text", "tool_call"]
    for part in body["parts"]:
        assert "raw" not in part and "resultRaw" not in part
    assert body["parts"][1]["resultText"] == "file body"
    assert str(heldout) not in json.dumps(body)


def test_round_detail_without_posthoc_or_log(env) -> None:
    client, _, _ = env
    body = client.get("/api/experiments/runs/r1/cells/t1-fixed-0/rounds/2").json()
    assert body["posthoc"] is None
    assert body["parts"] == []


def test_round_detail_without_heldout_pointer(env) -> None:
    client, run, _ = env
    (run / "t1-fixed-0" / "heldout.json").unlink()
    cell = client.get("/api/experiments/runs/r1/cells/t1-fixed-0").json()
    assert cell["state"] == "pending"
    assert cell["certifications"] == []
    assert client.get("/api/experiments/runs/r1/cells/t1-fixed-0/rounds/1").json()["posthoc"] is None


@pytest.mark.parametrize("path", [
    "/api/experiments/runs/bad%20name",
    "/api/experiments/runs/r1%0A",
    "/api/experiments/runs/r1/cells/bad%20cell",
    "/api/experiments/runs/r1/cells/t1-fixed-0/rounds/abc",
])
def test_invalid_names_are_400(env, path: str) -> None:
    client, _, _ = env
    response = client.get(path)
    assert response.status_code == 400
    assert isinstance(response.json()["detail"], str)


def test_dot_names_are_rejected() -> None:
    for value in (".", ".."):
        with pytest.raises(api.HTTPException) as caught:
            api._check_name(value, "run")
        assert caught.value.status_code == 400


def test_missing_things_are_404(env) -> None:
    client, _, _ = env
    for path in [
        "/api/experiments/runs/nope",
        "/api/experiments/runs/r1/cells/t9-fixed-0",
        "/api/experiments/runs/r1/cells/t1-fixed-0/rounds/99",
        "/api/experiments/runs/r1/cells/t1-fixed-0/rounds/0",
    ]:
        assert client.get(path).status_code == 404, path
    assert client.post("/api/experiments/runs/nope/stop").status_code == 404


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def _exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _live_process() -> subprocess.Popen:
    """A sleeping process in its own session, registered like a started batch."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    api._children[proc.pid] = proc
    return proc


def test_start_validation(env) -> None:
    client, _, _ = env
    url = "/api/experiments/runs/new/start"
    assert client.post(url, json={"spec": "../tiny.json"}).status_code == 400
    assert client.post(url, json={}).status_code == 400
    assert client.post(url, json={"spec": "tiny.json", "workers": 0}).status_code == 400
    assert client.post(url, json={"spec": "tiny.json", "workers": 17}).status_code == 400
    assert client.post(url, json={"spec": "absent.json"}).status_code == 404
    assert client.post(url, json={"spec": "report.json"}).status_code == 400
    assert client.post("/api/experiments/runs/bad%20name/start", json={"spec": "tiny.json"}).status_code == 400
    assert not (api.RUNS_ROOT / "new").exists()


def test_trailing_newline_name_is_rejected_without_creating_a_directory(env) -> None:
    client, _, _ = env
    assert client.post("/api/experiments/runs/r1%0A/start", json={"spec": "tiny.json"}).status_code == 400
    assert client.post("/api/experiments/runs/fresh%0A/start", json={"spec": "tiny.json"}).status_code == 400
    assert client.post("/api/experiments/runs/fresh/start", json={"spec": "tiny.json%0A"}).status_code == 400
    assert client.get("/api/experiments/runs/r1/cells/t1-fixed-0%0A").status_code == 400
    assert sorted(path.name for path in api.RUNS_ROOT.iterdir()) == ["r1"]
    for bad in ("-lead", ".hidden", "a\n"):
        with pytest.raises(api.HTTPException):
            api._check_name(bad, "run")


# --------------------------------------------------- real batch.py lifecycle

FAKE_LOOP = '''
import json, os, sys, time
from pathlib import Path

args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
cell = f"{args['--task']}-{args['--arm']}-{args['--seed']}"
work = Path(WORK)
with (work / "calls.txt").open("a") as calls:
    calls.write(cell + "\\n")
run_dir = Path(args["--run-dir"])
heldout = work / "heldout" / cell
heldout.mkdir(parents=True, exist_ok=True)
run_dir.mkdir(parents=True, exist_ok=True)
rows = [
    {"round": n, "candidateId": f"c{n}", "parent": "m0", "arm": args["--arm"], "status": "evaluated",
     "diagMse": 1.0 / n, "dmP": 0.5, "accepted": n % 2 == 1, "proposal": {}, "seconds": 0.1}
    for n in (1, 2, 3)
]
(run_dir / "rounds.jsonl").write_text("".join(json.dumps(row) + "\\n" for row in rows))
(run_dir / "heldout.json").write_text(json.dumps({"path": str(heldout)}))
if cell == SLOW and (work / "slow.flag").exists():
    (work / "slow.pid").write_text(str(os.getpid()))
    time.sleep(120)
(heldout / "done.json").write_text(json.dumps({"rounds": 3}))
'''

SLOW_CELL = "t1-self-built-0"


@pytest.fixture
def real_batch(env, tmp_path, monkeypatch):
    """The real experiments/batch.py driving a fake loop: one cell sleeps while slow.flag exists."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "slow.flag").write_text("")
    loop = work / "fake_loop.py"
    loop.write_text(f"WORK = {str(work)!r}\nSLOW = {SLOW_CELL!r}\n" + FAKE_LOOP, encoding="utf-8")

    def command(spec: Path, runs_root: Path, workers: int, only: str) -> list[str]:
        cmd = [
            sys.executable, str(api.REPO_ROOT / "experiments" / "batch.py"), "--spec", str(spec),
            "--workers", str(workers), "--runs-root", str(runs_root),
            "--loop-cmd", f"{sys.executable} {loop}",
        ]
        return cmd + [f"--only={only}"] if only else cmd

    monkeypatch.setattr(api, "_batch_command", command)
    yield env[0], work
    slow_pid = work / "slow.pid"
    if slow_pid.exists() and _exists(int(slow_pid.read_text())):
        os.kill(int(slow_pid.read_text()), signal.SIGKILL)


def _calls(work: Path) -> list[str]:
    path = work / "calls.txt"
    return path.read_text().split() if path.exists() else []


def _states(client: TestClient, base: str) -> dict[str, str]:
    return {c["cell"]: c["state"] for c in client.get(base).json()["cells"]}


def _wait_stopped(client: TestClient, base: str, timeout: float = 30.0) -> None:
    _wait_for(lambda: client.get(base).json()["process"]["state"] == "stopped", timeout)


def test_stop_and_resume_run_the_real_batch(real_batch) -> None:
    client, work = real_batch
    base = "/api/experiments/runs/life"
    started = client.post(f"{base}/start", json={"spec": "tiny.json", "workers": 1})
    assert started.status_code == 200
    process = started.json()["process"]
    assert process["state"] == "running" and process["specFile"] == "tiny.json"
    assert client.post(f"{base}/start", json={"spec": "tiny.json"}).status_code == 409

    _wait_for(lambda: (work / "slow.pid").exists(), 30)
    slow_pid = int((work / "slow.pid").read_text())
    states = _states(client, base)
    assert states == {
        "t1-fixed-0": "done", SLOW_CELL: "running", "t2-fixed-0": "pending", "t2-self-built-0": "pending",
    }
    assert _calls(work) == ["t1-fixed-0", SLOW_CELL]

    stopped = client.post(f"{base}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["process"]["state"] == "stopped"
    # batch's SIGTERM handler took the sleeping loop down with it
    _wait_for(lambda: not _exists(slow_pid), 10)
    assert not _exists(process["pid"])
    assert _states(client, base)[SLOW_CELL] == "pending"  # rounds on disk, nothing running, no fail event
    assert client.post(f"{base}/stop").status_code == 409

    (work / "slow.flag").unlink()
    assert client.post(f"{base}/start", json={"spec": "tiny.json", "workers": 2}).status_code == 200
    _wait_stopped(client, base)
    assert set(_states(client, base).values()) == {"done"}
    # Completed cells were not invoked again; the interrupted one was resumed once.  (The restart runs
    # two workers, so the order in which the remaining cells start is not asserted.)
    calls = _calls(work)
    assert {cell: calls.count(cell) for cell in set(calls)} == {
        "t1-fixed-0": 1, SLOW_CELL: 2, "t2-fixed-0": 1, "t2-self-built-0": 1,
    }
    events = client.get(base).json()["batchEvents"]
    assert [e["event"] for e in events if e["cell"] == "t1-fixed-0"] == ["ok"]


def test_only_runs_just_the_named_cell(real_batch) -> None:
    client, work = real_batch
    base = "/api/experiments/runs/narrow"
    response = client.post(f"{base}/start", json={"spec": "tiny.json", "workers": 2, "only": "cell=t2-fixed-0"})
    assert response.status_code == 200
    _wait_stopped(client, base)
    assert _calls(work) == ["t2-fixed-0"]
    assert _states(client, base) == {
        "t1-fixed-0": "pending", "t1-self-built-0": "pending", "t2-fixed-0": "done", "t2-self-built-0": "pending",
    }


# ------------------------------------------------------------ process rules


def test_cell_state_rule_while_the_batch_is_alive(env) -> None:
    client, run, _ = env
    proc = _live_process()
    started = 1_800_000_000
    iso = lambda offset: time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(started + offset))  # noqa: E731
    _write(run / "batch-process.json", json.dumps({
        "pid": proc.pid, "specPath": str(api.SPECS_DIR / "tiny.json"), "startedAt": iso(0), "workers": 1, "only": "",
    }))
    cell = "t1-self-built-0"
    log = run / "logs" / f"{cell}.log"
    _write(log, "output")

    def state() -> str:
        return _states(client, "/api/experiments/runs/r1")[cell]

    def event(offset: int) -> None:
        _jsonl(run / "batch-status.jsonl", [
            {"event": "fail", "cell": cell, "exitCode": 1, "attempt": 1, "seconds": 1.0, "at": iso(offset)}])

    event(-100)                                  # failed before this process started
    os.utime(log, (started - 50, started - 50))  # log older than the process
    assert state() == "failed"
    os.utime(log, (started + 50, started + 50))  # written since the start, after the failure
    assert state() == "running"
    event(100)                                   # failure recorded after the last log write
    assert state() == "failed"
    log.unlink()
    assert state() == "failed"
    (run / "batch-status.jsonl").unlink()
    assert state() == "pending"                  # never reached by this process
    assert _states(client, "/api/experiments/runs/r1")["t1-fixed-0"] == "done"


def test_corrupt_process_record_is_treated_as_none(env) -> None:
    client, run, _ = env
    for text in ("{not json", json.dumps({"pid": "12"}), json.dumps({"pid": 1}), json.dumps([1])):
        _write(run / "batch-process.json", text)
        assert client.get("/api/experiments/runs/r1").json()["process"]["state"] == "none"
    assert client.post("/api/experiments/runs/r1/stop").status_code == 409
    assert client.post("/api/experiments/runs/r1/start", json={"spec": "tiny.json"}).status_code == 200


def test_concurrent_starts_yield_one_process(env) -> None:
    _, _, _ = env
    barrier = threading.Barrier(2)
    codes: list[int] = []

    def post() -> None:
        app = FastAPI()
        app.include_router(api.router)
        client = TestClient(app)
        barrier.wait()
        codes.append(client.post("/api/experiments/runs/race/start", json={"spec": "tiny.json"}).status_code)

    threads = [threading.Thread(target=post) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(codes) == [200, 409]
    assert len(api._children) == 1


def test_state_is_recovered_after_a_server_restart(env) -> None:
    client, _, _ = env
    base = "/api/experiments/runs/revive"
    pid = client.post(f"{base}/start", json={"spec": "tiny.json"}).json()["process"]["pid"]
    proc = api._children.pop(pid)  # the new server has no Popen for the old child
    try:
        assert not api._children
        assert client.get(base).json()["process"]["state"] == "running"
        assert client.post(f"{base}/start", json={"spec": "tiny.json"}).status_code == 409
        assert client.post(f"{base}/stop").json()["process"]["state"] == "stopped"
    finally:
        proc.kill()
        proc.wait()


def test_a_reused_pid_that_is_not_a_group_leader_is_not_ours(env) -> None:
    client, run, _ = env
    stranger = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])  # same session as pytest
    try:
        _write(run / "batch-process.json", json.dumps({
            "pid": stranger.pid, "specPath": str(api.SPECS_DIR / "tiny.json"), "startedAt": "2026-10-19T00:00:00+00:00",
        }))
        assert client.get("/api/experiments/runs/r1").json()["process"]["state"] == "stopped"
        assert client.post("/api/experiments/runs/r1/stop").status_code == 409
        assert stranger.poll() is None  # stop never signalled it
    finally:
        stranger.kill()
        stranger.wait()


def test_start_with_another_spec_is_a_conflict(env) -> None:
    client, run, _ = env
    _stopped_record(run)  # r1 belongs to tiny.json
    _write(api.SPECS_DIR / "other.json", json.dumps({"tasks": [{"name": "t1"}], "arms": ["fixed"], "seeds": [0], "budget": 1}))
    response = client.post("/api/experiments/runs/r1/start", json={"spec": "other.json"})
    assert response.status_code == 409
    assert "tiny.json" in response.json()["detail"]
    assert client.post("/api/experiments/runs/r1/start", json={"spec": "tiny.json"}).status_code == 200


def test_start_creates_missing_run_dir(env) -> None:
    client, _, _ = env
    response = client.post("/api/experiments/runs/fresh/start", json={"spec": "tiny.json"})
    assert response.status_code == 200
    assert (api.RUNS_ROOT / "fresh" / "batch-process.json").is_file()
    assert client.post("/api/experiments/runs/fresh/stop").json()["process"]["state"] == "stopped"


def test_batch_command_uses_run_dir_as_runs_root(tmp_path) -> None:
    command = api._batch_command(Path("/s/x.json"), tmp_path / "r", 3, "arm=fixed")
    assert command[1].endswith("experiments/batch.py")
    assert command[command.index("--runs-root") + 1] == str(tmp_path / "r")
    assert command[command.index("--workers") + 1] == "3"
    assert "--only=arm=fixed" in command
    assert not any(arg.startswith("--only") for arg in api._batch_command(Path("/s/x.json"), tmp_path, 1, ""))


# ----------------------------------------------------------------- analysis


def _fake_analysis(monkeypatch, body: str, seen: list | None = None) -> None:
    def command(run_dir: Path, real_csvs: dict) -> list[str]:
        if seen is not None:
            seen.append((run_dir, real_csvs))
        return [sys.executable, "-c", body, str(run_dir / "analysis.json")]

    monkeypatch.setattr(api, "_analysis_command", command)


WRITE_ANALYSIS = (
    "import json, sys; "
    "open(sys.argv[1], 'w').write(json.dumps({'endpointTable': [1], 'contrasts': [], 'runs': [], "
    "'incomplete': [], 'audit': {}})); print('analysis ok', flush=True)"
)


def test_analysis_before_any_start(env) -> None:
    client, _, _ = env
    body = client.get("/api/experiments/runs/r1/analysis").json()
    assert body == {"process": {"state": "none", "pid": None, "specFile": None, "startedAt": None},
                    "analysis": None, "logTail": ""}
    assert client.get("/api/experiments/runs/nope/analysis").status_code == 404
    assert client.post("/api/experiments/runs/nope/analysis").status_code == 404
    assert client.post("/api/experiments/runs/bad%20name/analysis").status_code == 400


def test_analysis_runs_in_the_background_and_returns_the_payload(env, monkeypatch) -> None:
    client, run, _ = env
    seen: list = []
    _fake_analysis(monkeypatch, WRITE_ANALYSIS, seen)
    base = "/api/experiments/runs/r1/analysis"
    response = client.post(base)
    assert response.status_code == 200
    assert response.json()["process"]["pid"] == json.loads((run / "analysis-process.json").read_text())["pid"]
    assert seen == [(api.RUNS_ROOT / "r1", {})]
    _wait_for(lambda: client.get(base).json()["process"]["state"] == "stopped")
    body = client.get(base).json()
    assert body["analysis"] == {"endpointTable": [1], "contrasts": [], "runs": [], "incomplete": [], "audit": {}}
    assert body["logTail"] == "analysis ok"
    assert client.post(base).status_code == 200  # a finished analysis can be rerun


def test_only_one_analysis_runs_at_a_time(env, monkeypatch) -> None:
    client, _, _ = env
    _fake_analysis(monkeypatch, "import time; time.sleep(60)")
    base = "/api/experiments/runs/r1/analysis"
    assert client.post(base).status_code == 200
    assert client.get(base).json()["process"]["state"] == "running"
    assert client.post(base).status_code == 409


def test_corrupt_or_partial_analysis_file_is_null(env) -> None:
    client, run, _ = env
    for text in ('{"endpointTable": [1, ', "", "[1, 2]", "null"):
        _write(run / "analysis.json", text)
        response = client.get("/api/experiments/runs/r1/analysis")
        assert response.status_code == 200
        assert response.json()["analysis"] is None


def test_analysis_passes_real_task_csvs_from_the_runs_spec(env, monkeypatch) -> None:
    client, run, _ = env
    _write(api.SPECS_DIR / "mixed.json", json.dumps({
        "tasks": [
            {"name": "syn", "kind": "synthetic"},
            {"name": "real-ett", "kind": "real", "csv": "experiments/data/ETTh1.csv"},
            {"name": "bad name", "kind": "real", "csv": "x.csv"},
        ],
        "arms": ["fixed"], "seeds": [0], "budget": 1,
    }))
    _write(run / "batch-process.json", json.dumps({
        "pid": 99999999, "specPath": str(api.SPECS_DIR / "mixed.json"), "startedAt": "2026-10-19T00:00:00+00:00"}))
    seen: list = []
    _fake_analysis(monkeypatch, "pass", seen)
    assert client.post("/api/experiments/runs/r1/analysis").status_code == 200
    assert seen[0][1] == {
        "real-ett": api.REPO_ROOT / "experiments/data/ETTh1.csv",
        "bad name": api.REPO_ROOT / "x.csv",
    }


def test_analysis_command_runs_posthoc_with_the_run_as_input_and_output(tmp_path) -> None:
    command = api._analysis_command(tmp_path / "r", {"real-ett": Path("/d/ETTh1.csv")})
    assert command == [
        "uv", "run", "--extra", "research", "python",
        str(api.REPO_ROOT / "experiments" / "posthoc.py"),
        "--runs", str(tmp_path / "r"), "--out", str(tmp_path / "r" / "analysis.json"),
        "--real-csv", "real-ett=/d/ETTh1.csv",
    ]
    assert "--real-csv" not in api._analysis_command(tmp_path / "r", {})


# --------------------------------------------------------- incumbent curve


def test_incumbent_curve_carries_the_accepted_candidate_forward(env) -> None:
    client, _, _ = env
    body = client.get("/api/experiments/runs/r1/cells/t1-fixed-0").json()
    assert body["incumbentCurve"] == [
        {"round": 0, "mse": 1.2, "accepted": False},
        {"round": 1, "mse": 1.0, "accepted": True},
        {"round": 2, "mse": 1.0, "accepted": False},  # rejected: the incumbent stays
        {"round": 3, "mse": 0.5, "accepted": True},
    ]
    assert [r["diagMse"] for r in body["rounds"]] == [1.0, 0.8, 0.5]


def test_incumbent_curve_without_m0_starts_at_the_first_accepted_round(env) -> None:
    client, run, _ = env
    _jsonl(run / "t1-self-built-0" / "rounds.jsonl", [
        _round(1, "c1", accepted=False, diag=1.5, arm="self-built"),
        _round(2, "c2", accepted=True, diag=0.9, arm="self-built"),
        _round(3, "c3", accepted=False, diag=None, arm="self-built"),
    ])
    body = client.get("/api/experiments/runs/r1/cells/t1-self-built-0").json()
    assert body["incumbentCurve"] == [
        {"round": 0, "mse": None, "accepted": False},
        {"round": 1, "mse": None, "accepted": False},
        {"round": 2, "mse": 0.9, "accepted": True},
        {"round": 3, "mse": 0.9, "accepted": False},
    ]
    summary = {c["cell"]: c for c in client.get("/api/experiments/runs/r1").json()["cells"]}["t1-self-built-0"]
    assert summary["incumbentMse"] == 0.9
    assert summary["lastDiagMse"] == 0.9  # the last round has no diagMse


def test_incumbent_mse_is_m0_before_any_acceptance(env) -> None:
    client, run, _ = env
    _jsonl(run / "t1-fixed-0" / "rounds.jsonl", [_round(1, "c1", accepted=False, diag=2.0)])
    summary = {c["cell"]: c for c in client.get("/api/experiments/runs/r1").json()["cells"]}["t1-fixed-0"]
    assert (summary["incumbentMse"], summary["lastDiagMse"]) == (1.2, 2.0)


def test_round_detail_merges_tool_results(env) -> None:
    client, run, _ = env
    _jsonl(run / "t1-fixed-0" / "ws" / "logs" / "round-2.jsonl", [
        {"id": "a", "timestamp": "t", "role": "assistant", "type": "tool_call", "name": "read", "input": {},
         "intend": "read it", "toolUseId": "u1", "status": "running", "text": "read", "raw": {"x": 1}},
        {"id": "b", "timestamp": "t", "role": "tool", "type": "tool_result", "toolUseId": "u1",
         "status": "completed", "resultText": "body", "text": "body", "raw": {"y": 2}},
    ])
    parts = client.get("/api/experiments/runs/r1/cells/t1-fixed-0/rounds/2").json()["parts"]
    assert [(p["type"], p["status"], p["resultText"]) for p in parts] == [("tool_call", "completed", "body")]
    assert "raw" not in parts[0] and "resultRaw" not in parts[0]


def test_create_app_serves_the_experiment_routes(env, tmp_path, monkeypatch) -> None:
    from harnessing_ts.server import create_app
    from harnessing_ts.state.workspace_store import WorkspaceStore

    workspace = tmp_path / "workspace"
    WorkspaceStore(workspace).ensure_layout()
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(workspace))
    response = TestClient(create_app()).get("/api/experiments/specs")
    assert response.status_code == 200
    assert response.json()[0]["file"] == "tiny.json"


def test_experiment_children_receive_the_server_llm_config_path(tmp_path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(workspace))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output = tmp_path / "env.txt"
    command = [
        sys.executable, "-c",
        f"import os, pathlib; pathlib.Path({str(output)!r}).write_text("
        "os.environ.get('TS_HARNESS_LLM_CONFIG_FILE', ''))",
    ]

    record = api._launch(
        run_dir, command, "process.log", "wb", "process.json", {},
        extra_env=api._llm_config_env(),
    )
    api._children.pop(record["pid"]).wait(timeout=10)

    assert output.read_text() == str(workspace / "config.llm.json")
