from __future__ import annotations

"""Tests for the experiment-run API (backend/harnessing_ts/experiments_api.py).

A fake runs root and specs dir live under tmp_path.  The batch command is
replaced by a tiny python script that prints a line and sleeps, so the
start/stop lifecycle runs real process groups without batch.py or any LLM.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
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
        "name": "tiny-spec", "description": "two tasks",
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
    for pid in list(api._children):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
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
        "name": "tiny-spec", "file": "tiny.json", "description": "two tasks", "cellCount": 4,
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
    assert (done["lastDiagMse"], done["bestDiagMse"]) == (0.5, 0.5)
    partial = cells["t1-self-built-0"]
    assert partial["roundsDone"] == 1  # torn last line skipped
    assert (partial["incumbent"], partial["saturated"], partial["lastDiagMse"]) == ("m0", None, None)
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


def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached")


def test_start_stop_lifecycle(env) -> None:
    client, run, _ = env
    base = "/api/experiments/runs/r1"
    assert client.post(f"{base}/stop").status_code == 409  # nothing running

    started = client.post(f"{base}/start", json={"spec": "tiny.json", "workers": 2, "only": "t1"})
    assert started.status_code == 200
    process = started.json()["process"]
    assert process["state"] == "running" and process["specFile"] == "tiny.json"
    record = json.loads((run / "batch-process.json").read_text())
    assert record["pid"] == process["pid"] and record["workers"] == 2 and record["only"] == "t1"
    _wait_for(lambda: "fake batch up" in (run / "batch-run.log").read_text())

    snapshot = client.get(base).json()
    assert snapshot["process"]["state"] == "running"
    cells = {c["cell"]: c["state"] for c in snapshot["cells"]}
    assert cells["t1-fixed-0"] == "done"
    assert cells["t1-self-built-0"] == "running"  # has rounds, not done, process alive
    assert cells["t2-fixed-0"] == "pending"
    assert snapshot["batchLogTail"].splitlines()[-1] == "fake batch up"

    assert client.post(f"{base}/start", json={"spec": "tiny.json"}).status_code == 409

    stopped = client.post(f"{base}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["process"]["state"] == "stopped"
    assert client.get(base).json()["process"]["state"] == "stopped"
    assert client.get(base).json()["cells"][1]["state"] == "failed"  # last event is fail, no process
    assert client.post(f"{base}/stop").status_code == 409

    again = client.post(f"{base}/start", json={"spec": "tiny.json"})
    assert again.status_code == 200
    assert again.json()["process"]["pid"] != process["pid"]
    assert again.json()["process"]["state"] == "running"
    assert client.post(f"{base}/stop").json()["process"]["state"] == "stopped"


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


def test_create_app_serves_the_experiment_routes(env, tmp_path, monkeypatch) -> None:
    from harnessing_ts.server import create_app
    from harnessing_ts.state.workspace_store import WorkspaceStore

    workspace = tmp_path / "workspace"
    WorkspaceStore(workspace).ensure_layout()
    monkeypatch.setenv("TS_HARNESS_WORKSPACE", str(workspace))
    response = TestClient(create_app()).get("/api/experiments/specs")
    assert response.status_code == 200
    assert response.json()[0]["file"] == "tiny.json"
