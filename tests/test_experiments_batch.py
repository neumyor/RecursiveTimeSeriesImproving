from __future__ import annotations

"""Tests for the P2 parallel batch driver (experiments/batch.py).

Everything runs against a FAKE loop command (the --loop-cmd seam): a tiny
script in tmp_path that mirrors loop.py's CLI surface, records every
invocation so tests can count attempts, creates the resume sentinels
(rounds.jsonl, done.json inside the heldout dir that heldout.json points
to, exactly where loop.py writes it) and exits 0 -- with seed-parameterized
failure modes so crash containment, the single retry, and secret scrubbing
are exercised without the real loop, any LLM, or any task simulation.
"""

import importlib.util
import json
from pathlib import Path
import shlex
import sys
import time

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

FAKE_SECRET = "sk-fake-batch-test-SECRET-value"

# Fails for seeds in BATCH_TEST_FAIL_ONCE_SEEDS the first time (after writing
# rounds.jsonl, i.e. exactly a crash mid-run) and in BATCH_TEST_FAIL_ALWAYS_SEEDS
# every time.  It also prints its API key like a chatty crash dump would, so
# the driver's scrubbing has something real to scrub.
FAKE_LOOP_SOURCE = """\
import argparse, json, os, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", required=True)
parser.add_argument("--task")
parser.add_argument("--arm")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--budget", type=int, default=25)
parser.add_argument("--real-csv")
args = parser.parse_args()

run_dir = Path(args.run_dir)
run_dir.mkdir(parents=True, exist_ok=True)
with (run_dir / "invocations").open("a", encoding="utf-8") as handle:
    handle.write(f"{args.task}|{args.arm}|{args.seed}|{args.budget}|{args.real_csv}\\n")
print(f"TS_HARNESS_LLM_API_KEY={os.environ.get('TS_HARNESS_LLM_API_KEY', '<missing>')}")
fail_once = {int(s) for s in os.environ.get("BATCH_TEST_FAIL_ONCE_SEEDS", "").split(",") if s}
fail_always = {int(s) for s in os.environ.get("BATCH_TEST_FAIL_ALWAYS_SEEDS", "").split(",") if s}
if args.seed in fail_once and not (run_dir / "rounds.jsonl").exists():
    (run_dir / "rounds.jsonl").write_text("{}\\n", encoding="utf-8")  # crashed mid-run
    sys.exit(1)
if args.seed in fail_always:
    sys.exit(2)
(run_dir / "rounds.jsonl").write_text("{}\\n", encoding="utf-8")
heldout = run_dir / "heldout"
heldout.mkdir(parents=True, exist_ok=True)
(heldout / "done.json").write_text("{}\\n", encoding="utf-8")
(run_dir / "heldout.json").write_text(json.dumps({"path": str(heldout)}) + "\\n", encoding="utf-8")
"""

_batch_module = None


def _load_batch():
    # experiments/ is deliberately not on the repo pythonpath (only backend/
    # is), so the driver is imported by file path, exactly like loop.py
    # imports realdata.py.
    global _batch_module
    if _batch_module is None:
        spec = importlib.util.spec_from_file_location(
            "rhg_batch", REPO_ROOT / "experiments" / "batch.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["rhg_batch"] = module  # register before exec: dataclasses need it
        spec.loader.exec_module(module)
        _batch_module = module
    _batch_module._STOPPING.clear()  # a stop in an earlier test must not carry over
    return _batch_module


@pytest.fixture
def fake_loop(tmp_path: Path) -> Path:
    script = tmp_path / "fake_loop.py"
    script.write_text(FAKE_LOOP_SOURCE, encoding="utf-8")
    return script


def _write_spec(path: Path, *, tasks: list[dict], arms: list[str], seeds: list[int]) -> Path:
    path.write_text(
        json.dumps(
            {
                "tasks": tasks,
                "arms": arms,
                "seeds": seeds,
                "budget": 3,
                "sizes": {"train": 100},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path) -> Path:
    path.write_text("date,ch0\n2020-01-01T00:00:00,1\n2020-01-01T01:00:00,2\n", encoding="utf-8")
    return path


def _loop_cmd(fake: Path) -> str:
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(fake))}"


def _main(batch, fake: Path, spec: Path, runs: Path, *extra: str) -> int:
    return batch.main(
        [
            "--spec", str(spec),
            "--runs-root", str(runs),
            "--loop-cmd", _loop_cmd(fake),
            "--workers", "4",
            *extra,
        ]
    )


def _invocation_counts(runs: Path) -> dict[str, int]:
    return {
        path.name: len((path / "invocations").read_text(encoding="utf-8").splitlines())
        for path in runs.iterdir()
        if path.is_dir() and (path / "invocations").is_file()
    }


def test_matrix_enumeration(tmp_path: Path) -> None:
    batch = _load_batch()
    spec = _write_spec(
        tmp_path / "spec.json",
        tasks=[
            {"name": "hmm", "kind": "synthetic"},
            {"name": "real-ett", "kind": "real", "csv": str(_write_csv(tmp_path / "ett.csv"))},
        ],
        arms=["scalar", "fixed"],
        seeds=[0, 1, 2, 3],
    )
    cells = batch.enumerate_cells(batch.load_spec(spec))
    assert len(cells) == 16
    assert {cell.cell_id for cell in cells} == {
        f"{task}-{arm}-{seed}"
        for task in ("hmm", "real-ett")
        for arm in ("scalar", "fixed")
        for seed in (0, 1, 2, 3)
    }


def test_batch_runs_matrix_with_skip_retry_records_and_redaction(
    tmp_path: Path, fake_loop: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    batch = _load_batch()
    monkeypatch.setenv("TS_HARNESS_LLM_API_KEY", FAKE_SECRET)
    monkeypatch.setenv("BATCH_TEST_FAIL_ONCE_SEEDS", "1")  # transient: fails, then succeeds
    monkeypatch.setenv("BATCH_TEST_FAIL_ALWAYS_SEEDS", "3")  # permanent: fails the retry too
    runs = tmp_path / "runs"
    spec = _write_spec(
        tmp_path / "spec.json",
        tasks=[
            {"name": "hmm", "kind": "synthetic"},
            {"name": "real-ett", "kind": "real", "csv": str(_write_csv(tmp_path / "ett.csv"))},
        ],
        arms=["scalar", "fixed"],
        seeds=[0, 1, 2, 3],
    )
    # One cell is already finished: the loop command must never run for it.
    pre_done_heldout = runs / "hmm-scalar-0" / "heldout"
    pre_done_heldout.mkdir(parents=True)
    (pre_done_heldout / "done.json").write_text("{}\n", encoding="utf-8")
    (runs / "hmm-scalar-0" / "heldout.json").write_text(
        json.dumps({"path": str(pre_done_heldout)}), encoding="utf-8"
    )

    code = _main(batch, fake_loop, spec, runs)

    assert code == 1  # the four seed-3 cells fail even after the retry
    assert not (runs / "hmm-scalar-0" / "invocations").exists()  # done cell was skipped

    expected = {
        f"{task}-{arm}-{seed}"
        for task in ("hmm", "real-ett")
        for arm in ("scalar", "fixed")
        for seed in (0, 1, 2, 3)
    }
    for cell_id in expected:
        if cell_id == "hmm-scalar-0":
            continue
        first = (runs / cell_id / "invocations").read_text(encoding="utf-8").splitlines()[0]
        task, arm, seed, budget, csv = first.split("|")
        assert cell_id == f"{task}-{arm}-{seed}"  # --task/--arm/--seed were passed through
        assert int(budget) == 3
        # The command carried the real csv exactly for real-task cells.
        assert (csv == str(tmp_path / "ett.csv")) == (task == "real-ett")
        done = (runs / cell_id / "heldout" / "done.json").is_file()
        assert done == (not cell_id.endswith("-3"))

    counts = _invocation_counts(runs)
    assert counts["hmm-fixed-0"] == 1  # succeeded first try
    assert counts["hmm-fixed-1"] == 2  # failed once, retried once, succeeded
    assert counts["hmm-fixed-3"] == 2  # retried ONCE even though it failed again
    assert counts["real-ett-scalar-1"] == 2

    records = [
        json.loads(line)
        for line in (runs / "batch-status.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    fails = [r for r in records if r["event"] == "fail"]
    oks = [r for r in records if r["event"] == "ok"]
    assert len(oks) == 11  # 15 run cells - 4 permanent failures
    assert [r["attempt"] for r in fails if r["cell"] == "hmm-fixed-1"] == [1]
    assert [r["attempt"] for r in fails if r["cell"] == "hmm-fixed-3"] == [1, 2]
    assert all(r["exitCode"] in (1, 2) for r in fails)
    assert all("at" in r and "seconds" in r for r in records)
    # Final failures are reported in the summary.
    out = capsys.readouterr().out
    assert "hmm-fixed-3" in out and "failed after the retry" in out

    # No secret anywhere the batch wrote, even though the fake printed it.
    assert FAKE_SECRET not in out + capsys.readouterr().err
    for path in runs.rglob("*"):
        if path.is_file():
            assert FAKE_SECRET not in path.read_text(encoding="utf-8", errors="replace"), path
    # The chatty fake DID hit the log; the scrub is what removed it.
    assert "TS_HARNESS_LLM_API_KEY=***" in (runs / "logs" / "hmm-fixed-0.log").read_text(
        encoding="utf-8"
    )


def test_only_filter_subsets_the_matrix(tmp_path: Path, fake_loop: Path) -> None:
    batch = _load_batch()
    spec = _write_spec(
        tmp_path / "spec.json",
        tasks=[{"name": "hmm", "kind": "synthetic"}, {"name": "real-ett", "kind": "real",
                                                      "csv": str(_write_csv(tmp_path / "ett.csv"))}],
        arms=["scalar", "fixed"],
        seeds=[0, 1],
    )
    runs = tmp_path / "only-runs"
    assert _main(batch, fake_loop, spec, runs, "--only", "hmm-fixed") == 0
    assert set(_invocation_counts(runs)) == {"hmm-fixed-0", "hmm-fixed-1"}

    runs2 = tmp_path / "only-runs2"
    assert _main(batch, fake_loop, spec, runs2, "--only", "seed=1") == 0
    assert set(_invocation_counts(runs2)) == {
        "hmm-scalar-1", "hmm-fixed-1", "real-ett-scalar-1", "real-ett-fixed-1"
    }


def test_unstartable_loop_command_is_a_contained_cell_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    batch = _load_batch()
    spec = _write_spec(
        tmp_path / "spec.json",
        tasks=[{"name": "hmm", "kind": "synthetic"}],
        arms=["fixed"],
        seeds=[0, 1],
    )
    runs = tmp_path / "runs"
    code = batch.main(
        [
            "--spec", str(spec),
            "--runs-root", str(runs),
            "--loop-cmd", "definitely-not-a-command-batchtest",
        ]
    )
    assert code == 1  # a cell failure, reported -- not a batch-fatal crash
    records = [
        json.loads(line)
        for line in (runs / "batch-status.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 4  # 2 cells x (fail + retry-fail)
    assert all(r["event"] == "fail" and r["exitCode"] is None for r in records)
    assert all("could not start the loop process" in r["detail"] for r in records)


def test_bad_specs_and_args_are_batch_fatal(
    tmp_path: Path, fake_loop: Path, capsys: pytest.CaptureFixture
) -> None:
    batch = _load_batch()
    good = _write_spec(
        tmp_path / "good.json",
        tasks=[{"name": "hmm", "kind": "synthetic"}],
        arms=["fixed"],
        seeds=[0],
    )
    runs = tmp_path / "runs"
    assert _main(batch, fake_loop, good, runs) == 0

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")
    missing_key = tmp_path / "missing_key.json"
    missing_key.write_text('{"tasks": [{"name": "hmm", "kind": "synthetic"}]}', encoding="utf-8")
    bad_arm = tmp_path / "bad_arm.json"
    bad_arm.write_text(
        json.dumps({"tasks": [{"name": "hmm", "kind": "synthetic"}], "arms": ["carsd"],
                    "seeds": [0], "budget": 3}),
        encoding="utf-8",
    )
    missing_csv = tmp_path / "missing_csv.json"
    missing_csv.write_text(
        json.dumps({"tasks": [{"name": "ett", "kind": "real", "csv": "experiments/data/nope.csv"}],
                    "arms": ["fixed"], "seeds": [0], "budget": 3}),
        encoding="utf-8",
    )
    untouched = tmp_path / "untouched-runs"
    for spec_path in (malformed, missing_key, bad_arm, missing_csv):
        assert _main(batch, fake_loop, spec_path, untouched) == 2
        assert "fatal" in capsys.readouterr().err
    # Batch-fatal means no cell was ever started: not even the runs root exists.
    assert not untouched.exists()

    with pytest.raises(SystemExit):
        batch.main(["--spec", str(good), "--runs-root", str(runs), "--workers", "0"])


def test_systemic_failure_cancels_queued_cells(tmp_path: Path) -> None:
    """A billing/auth-style failure streak aborts the pass AND cancels the
    queued cells: only the cells up to (or just past) the six-failure abort
    threshold ever start, and everything still queued is never invoked."""
    batch = _load_batch()
    spec = batch.load_spec(
        _write_spec(
            tmp_path / "spec.json",
            tasks=[{"name": "hmm", "kind": "synthetic"}],
            arms=["fixed", "scalar"],
            seeds=[0, 1, 2, 3],
        )
    )
    driver = batch.BatchDriver(spec, tmp_path / "runs", "python -c pass", workers=1)
    driver.runs_root.mkdir(parents=True, exist_ok=True)
    driver.log_dir.mkdir(parents=True, exist_ok=True)
    started: list[str] = []

    def fake_attempt(cell: "batch.Cell", attempt: int) -> "batch.CellOutcome":
        started.append(cell.cell_id)
        time.sleep(0.2)  # long enough that the abort lands while a cell runs
        return batch.CellOutcome(cell, attempt, 2, 0.0, "boom")

    driver._attempt = fake_attempt  # type: ignore[method-assign]
    cells = batch.enumerate_cells(spec)
    with pytest.raises(batch.SystemicFailure):
        driver.run_pass(cells, attempt=1)
    # the streak hit the max(6, workers) threshold and aborted mid-matrix:
    # at most the first seven cells started (the one in flight when the
    # sixth failure landed), and the rest were cancelled while queued
    assert 6 <= len(started) <= 7
    assert "hmm-scalar-3" not in started


def test_main_v3_spec_loads_and_enumerates_the_same_matrix() -> None:
    """The v3 spec passes load_spec (G0 artifact gate included) and freezes the
    same 48-cell first-sweep matrix as v2."""
    batch = _load_batch()
    v2 = batch.load_spec(REPO_ROOT / "experiments" / "specs" / "main-v2.json")
    v3 = batch.load_spec(REPO_ROOT / "experiments" / "specs" / "main-v3.json")
    cells = batch.enumerate_cells(v3)
    assert len(cells) == 48
    assert [cell.cell_id for cell in cells] == [
        cell.cell_id for cell in batch.enumerate_cells(v2)
    ]
