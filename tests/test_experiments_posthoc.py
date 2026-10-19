from __future__ import annotations

"""Posthoc analysis tests: real loop artifacts, tmp_path only.

The fixtures generate genuine run directories with HeadroomLoop + a scripted
runner (the same pattern as tests/test_experiments_loop.py), so posthoc.py
consumes the real artifact schema — rounds.jsonl, heldout reference, per-
candidate posthoc cards, certification ledger — rather than a hand-built
mock.  The heldout directory is kept inside tmp_path by pointing
loop.HELDOUT_ROOT at a tmp dir before the runs are created.
"""

import asyncio
from datetime import datetime, timedelta
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, source: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


loop = _load_module("rhg_loop_posthoc_tests", _REPO_ROOT / "experiments" / "loop.py")
posthoc = _load_module("rhg_posthoc_tests", _REPO_ROOT / "experiments" / "posthoc.py")

SMALL_SIZES = {"train": 1500, "diag": 1100, "cert": 1300, "test": 1800}


def _small_cfg(budget: int = 3) -> Any:
    return loop.HeadroomConfig(
        input_length=48,
        horizon=8,
        lag_segments=((0, 7), (8, 15), (16, 31)),
        fir_len_cap=48,
        beyond_l_lags=2,
        band_period_range=(4, 24),
        budget_candidates=budget,
        candidate_timeout_seconds=60.0,
        ridge_alphas=(1.0, 100.0),
        evolution=False,  # tests never call the (real) diversity judge
    )


class ScriptedRunner:
    """A fake LLM session whose side effects come from a fixed script."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def send(self, prompt: str) -> list[Any]:
        self.calls.append(prompt)
        return []

    async def close(self) -> None:
        return None


# A self-contained VAR(7) with iterated multi-step rollout, appended to the
# incumbent source: lgssm_xchannel IS a coupled VAR at lags 3/5/7, so this
# candidate closes most of m0's diag gap and is accepted (DM p ~ 1e-42 at the
# small sizes); the heavily-shrunk follow-up variants stay evaluated-but-
# rejected — exactly the accepted/rejected mix the posthoc diagnostics need.
_VAR_SOURCE = '''


class VarRolloutAdapter(M0Adapter):
    """Joint least-squares VAR(P) with an iterated h-step rollout."""

    _REG = {reg!r}
    _P = 7

    def __init__(self) -> None:
        M0Adapter.__init__(self)
        self.model_id = "{cid}"
        self._phi = None

    def fit(self, train, meta):
        H = int(meta["H"])
        C = int(meta["C"])
        T = train.shape[0]
        rows = np.arange(self._P, T)
        design = np.stack(
            [train[rows - k] for k in range(1, self._P + 1)], axis=2
        ).reshape(rows.size, -1)
        eye = np.eye(design.shape[1])
        self._phi = np.linalg.solve(
            design.T @ design + self._REG * eye, design.T @ train[rows]
        )
        self._H = H
        self._C = C

    def predict_batch(self, X):
        X = np.asarray(X, dtype=np.float64)
        B = X.shape[0]
        out = np.empty((B, self._H, self._C))
        buf = X[:, -self._P :, :].copy()
        for h in range(self._H):
            design = np.stack(
                [buf[:, self._P - k] for k in range(1, self._P + 1)], axis=2
            ).reshape(B, -1)
            nxt = design @ self._phi
            out[:, h, :] = nxt
            buf = np.concatenate([buf[:, 1:, :], nxt[:, None, :]], axis=1)
        return out


def create_adapter() -> M0Adapter:
    return VarRolloutAdapter()
'''


def _var_candidate(reg: float) -> Any:
    """Build one round's candidate writer: append a VarRolloutAdapter with
    regularization ``reg`` to whatever the incumbent currently is."""

    def make(ws: Path, index: int) -> tuple[str, dict[str, Any]]:
        incumbent = (ws / "models" / "incumbent" / "model.py").read_text(encoding="utf-8")
        candidate_id = f"c-var-{index}"
        source = incumbent + _VAR_SOURCE.format(reg=reg, cid=candidate_id)
        proposal = {
            "candidateId": candidate_id,
            "targetAspect": "cross-channel",
            "aspectDetail": "joint lag structure across channels",
            "mechanism": "least-squares VAR(7) with iterated rollout",
            "expectation": "diag MSE drops",
        }
        return source, proposal

    return make


def _make_factory(script: list[Any]) -> Any:
    # The script position is shared across runners: repair attempts create a
    # fresh runner each, and the script advances per SEND, not per runner.
    state = {"pos": 0}

    def factory(ws: Path, system_prompt: str, round_index: int) -> Any:
        class _Runner(ScriptedRunner):
            async def send(self, prompt: str) -> list[Any]:
                self.calls.append(prompt)
                pos = state["pos"]
                state["pos"] += 1
                if pos >= len(script):
                    return []
                step = script[pos]
                if step is None:
                    return []
                model_source, proposal = step(ws, pos)
                candidate_dir = ws / "models" / "candidates" / proposal["candidateId"]
                candidate_dir.mkdir(parents=True, exist_ok=True)
                (candidate_dir / "model.py").write_text(model_source, encoding="utf-8")
                (candidate_dir / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
                return []

        return _Runner()

    return factory


def _candidate_script(budget: int) -> list[Any]:
    """Round 0 writes the true-structure VAR fix (accepted on diag); later
    rounds write heavily-shrunk variants of it (evaluated, rejected)."""
    return [_var_candidate(reg) for reg in [1e-8] + [1e8] * max(budget - 1, 0)]


def _run_loop(
    run_dir: Path,
    heldout_root: Path,
    *,
    arm: str,
    seed: int = 0,
    budget: int = 3,
    task: str = "lgssm_xchannel",
    real_csv: Path | None = None,
) -> None:
    loop.HELDOUT_ROOT = heldout_root  # keep the heldout side inside tmp_path
    loop_obj = loop.HeadroomLoop(
        run_dir=run_dir,
        task_name=task,
        arm=arm,
        seed=seed,
        cfg=_small_cfg(budget),
        sizes=None if real_csv is not None else SMALL_SIZES,
        real_csv=real_csv,
        runner_factory=_make_factory(_candidate_script(budget)),
        m0_template="m0.py",
    )
    asyncio.run(loop_obj.run())


def _rounds(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _write_cfg(tmp: Path) -> Path:
    """The analysis config file: the same small cfg the loops ran under."""
    path = tmp / "small-config.json"
    _small_cfg().to_json(path)
    return path


@pytest.fixture(scope="module")
def synthetic_analysis(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Two finished runs (card + scalar arms, same task and seed) plus the
    posthoc report produced by the real CLI over their directory."""
    tmp = tmp_path_factory.mktemp("synthetic")
    runs_dir = tmp / "p2"
    _run_loop(runs_dir / "fixed-s0", tmp / "heldout-root", arm="fixed", seed=0, budget=3)
    _run_loop(runs_dir / "scalar-s0", tmp / "heldout-root", arm="scalar", seed=0, budget=3)
    out = tmp / "analysis.json"
    code = posthoc.main(
        ["--m0-template", "m0.py", "--runs", str(runs_dir), "--out", str(out), "--config", str(_write_cfg(tmp))]
    )
    assert code == 0
    raw = out.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw  # strict JSON, nulls everywhere
    return {
        "report": json.loads(out.read_text(encoding="utf-8")),
        "runs_dir": runs_dir,
        "config": _write_cfg(tmp),
    }


def test_incumbent_chain_replay_and_endpoints(synthetic_analysis: dict[str, Any]) -> None:
    report = synthetic_analysis["report"]
    assert len(report["runs"]) == 2
    for rec in report["runs"]:
        assert rec["error"] is None, rec["error"]
        rounds = _rounds(synthetic_analysis["runs_dir"] / rec["run"])
        expected = ["m0"] + [row["candidateId"] for row in rounds if row["accepted"]]
        assert rec["incumbentChain"] == expected
        # the scripted VAR(7) fix is accepted at round 1 and never beaten after
        assert rec["acceptances"] == 1
        assert len(rec["incumbents"]) == 2
        assert rec["incumbents"][1]["id"] == expected[1]
        # m0 is not the oracle: the raw gap is positive and the m0 ngap is 1.0
        assert rec["incumbents"][0]["gap"] > 0.0
        assert rec["incumbents"][0]["mseOracle"] > 0.0
        assert rec["incumbents"][0]["normalizedGap"] == pytest.approx(1.0)
        for inc in rec["incumbents"]:
            assert inc["normalizedGap"] is not None
            assert np.isfinite(inc["normalizedGap"])
            assert inc["normalizedGap"] <= 1.5  # accepted chain never degrades test badly
        # the accepted incumbent genuinely closes most of the test gap
        assert rec["incumbents"][1]["normalizedGap"] < 1.0
        assert rec["e1"] == pytest.approx(rec["incumbents"][-1]["normalizedGap"])
        assert rec["e1"] is not None and np.isfinite(rec["e1"]) and rec["e1"] <= 1.5
        # E2 averages the incumbent's value after each round index; the only
        # acceptance happened at round 1, so every round's value equals E1
        assert rec["e2"] == pytest.approx(rec["e1"])
        # diagnostics
        assert rec["acceptances"] == sum(1 for row in rounds if row["accepted"])
        assert rec["evaluated"] == len(rounds)
        assert rec["failureRate"] == 0.0
        assert sum(rec["candidateTypes"].values()) == len(rounds)
        assert rec["candidateTypes"]["torch"] == 0  # the scripted candidates are numpy-only
        for key in ("inputTokens", "outputTokens"):
            assert key in rec["tokens"]
        assert rec["seconds"] >= 0.0
        # directed/off-target rates are fractions over comparable proposals
        for key in ("directedRate", "offTargetRate"):
            value = rec[key]
            assert value is None or 0.0 <= value <= 1.0
        assert rec["directedDenominator"] == len(rounds)
        assert rec["offTargetDenominator"] == rec["acceptances"]
        # the recomputed cert LCB exists for accepted incumbents only (spec)
        assert "certTotalLcb" not in rec["incumbents"][0]
        assert rec["incumbents"][1]["certTotalLcb"] is not None
        # 1 accepted incumbent < the 3 the LCB-gap correlation needs
        assert rec["lcbGapCorrelation"] is None
        # the runs are clean: no out-of-ws paths, series matches the reference
        assert rec["audit"]["flagged"] is False
        assert rec["audit"]["violations"] == []
        assert rec["audit"]["seriesMatchesTask"] is True
        # the stop point is the first SATURATED certification entry; on a task
        # with genuine headroom none may exist, in which case both fields null
        heldout = Path(
            json.loads(
                (synthetic_analysis["runs_dir"] / rec["run"] / "heldout.json").read_text(encoding="utf-8")
            )["path"]
        )
        certs = [
            json.loads(line)
            for line in (heldout / "certification.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if any(entry["saturated"] for entry in certs):
            assert rec["stopPoint"] is not None
            assert rec["stopPoint"]["incumbentId"] in rec["incumbentChain"]
            assert rec["postStopGain"] is not None
        else:
            assert rec["stopPoint"] is None
            assert rec["ngapAtStop"] is None
            assert rec["postStopGain"] is None


def test_endpoint_table_and_contrasts(synthetic_analysis: dict[str, Any]) -> None:
    report = synthetic_analysis["report"]
    table = {(row["task"], row["arm"]): row for row in report["endpointTable"]}
    assert ("lgssm_xchannel", "fixed") in table
    assert ("lgssm_xchannel", "scalar") in table
    for row in report["endpointTable"]:
        assert row["nRuns"] >= 1
        assert row["e1"]["median"] is not None
        assert row["e1"]["iqr"] >= 0.0
        assert row["e2"]["median"] is not None
    contrasts = {c["contrast"]: c for c in report["contrasts"]}
    assert set(contrasts) == {
        "self-built-vs-fixed",
        "fixed-vs-scalar",
        "self-built-vs-scalar",
        "augmented-vs-self-built",
        "augmented-vs-fixed",
    }
    card_scalar = contrasts["fixed-vs-scalar"]
    assert card_scalar["nBlocks"] == 1  # one (task, seed) block shared by both arms
    assert card_scalar["permutations"] == "exact"
    for contrast in report["contrasts"]:
        if contrast["nBlocks"]:
            assert 0.0 < contrast["pValue"] <= 1.0
            assert contrast["permLow"] <= contrast["permHigh"]
        else:  # arms without paired runs report nulls, never NaN
            assert contrast["statistic"] is None
            assert contrast["pValue"] is None


def test_permutation_helper_branches() -> None:
    # 2^5 = 32 patterns: exact enumeration
    small = [0.10, -0.20, 0.30, 0.05, -0.10]
    exact = posthoc._perm_contrast(small)
    assert exact["permutations"] == "exact"
    assert 0.0 < exact["pValue"] <= 1.0
    # 2^15 > 8192: the Monte Carlo branch, reproducible via the fixed seed
    big = [0.1 * ((-1.0) ** i) for i in range(15)]
    mc = posthoc._perm_contrast(big)
    assert mc["permutations"] == "monte-carlo"
    assert 0.0 < mc["pValue"] <= 1.0
    assert posthoc._perm_contrast(big) == mc
    # a null contrast (identical arms) must give p = 1, not a division error
    assert posthoc._perm_contrast([0.0, 0.0, 0.0])["pValue"] == 1.0


def test_audit_flags_planted_out_of_ws_paths(
    synthetic_analysis: dict[str, Any], tmp_path: Path
) -> None:
    """A copied run tree with planted tool-call leaks must be flagged, while
    in-workspace, backend, and template reads stay clean."""
    src = synthetic_analysis["runs_dir"] / "fixed-s0"
    runs_dir = tmp_path / "planted"
    dst = runs_dir / "fixed-s0"
    shutil.copytree(src, dst)  # heldout.json still points at the shared tmp heldout
    ws = dst / "ws"
    heldout_dir = Path(json.loads((dst / "heldout.json").read_text(encoding="utf-8"))["path"])
    logs = ws / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    parts = [
        # in-workspace read: allowed
        {"type": "tool_call", "name": "read", "input": {"path": str(ws / "data" / "series.npy")}},
        # repo backend read: explicitly allowed by the audit contract
        {
            "type": "tool_call",
            "name": "read",
            "input": {"path": str(_REPO_ROOT / "backend" / "harnessing_ts" / "headroom" / "config.py")},
        },
        # a text part mentioning a path is NOT a tool call: never flagged
        {"type": "text", "role": "user", "text": "someone wrote /etc/passwd in prose"},
        # planted violations: an out-of-ws read and a heldout-path glob
        {"type": "tool_call", "name": "grep", "input": {"path": "/etc/passwd"}},
        {"type": "tool_call", "name": "glob", "input": {"pattern": str(heldout_dir / "**")}},
        # a workspace-relative path whose slash separates directories is not absolute
        {"type": "tool_call", "name": "read", "input": {"path": "models/candidates/x/model.py"}},
        # nor is the slash after a glob wildcard
        {"type": "tool_call", "name": "glob", "input": {"pattern": "models/*/model.py"}},
    ]
    (logs / "round-1.jsonl").write_text(
        "\n".join(json.dumps(part) for part in parts) + "\n", encoding="utf-8"
    )
    # a DSH session log carries the same leak in its tool/call record
    zstandard = pytest.importorskip("zstandard")
    session_dir = ws / ".dsh-sessions" / "s1"
    session_dir.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            {
                "type": "tool/call",
                "data": {"name": "read", "arguments": json.dumps({"path": "/etc/hosts"})},
            }
        )
        + "\n"
    )
    (session_dir / "session.jsonl.zstd").write_bytes(
        zstandard.ZstdCompressor().compress(payload.encode("utf-8"))
    )

    report = posthoc.analyze_runs(
        runs_dir, config=synthetic_analysis["config"], real_csvs={}
    )
    rec = report["runs"][0]
    assert rec["error"] is None
    assert rec["audit"]["flagged"] is True
    violations = rec["audit"]["violations"]
    assert "/etc/passwd" in violations
    assert "/etc/hosts" in violations
    assert any(str(heldout_dir) in item for item in violations)
    assert not any("series.npy" in item for item in violations)
    assert not any("backend" in item for item in violations)
    assert not any(item.startswith("/candidates") for item in violations)
    assert report["audit"]["pathViolationRuns"] == [rec["run"]]


def _tiny_real_csv(path: Path, rows: int = 6000, channels: int = 2) -> Path:
    """An ETT-shaped hourly CSV long enough for load_real_task's 100-origin
    minimum on every split at the small cfg (train 3600 / diag 1200 / cert 600)."""
    rng = np.random.default_rng(7)
    start = datetime(2020, 1, 1)
    lines = ["date," + ",".join(f"ch_{c}" for c in range(channels))]
    t = np.arange(rows, dtype=np.float64)
    for row in range(rows):
        values = [
            0.9 ** t[row] * 10.0 + 3.0 * np.sin(2.0 * np.pi * t[row] / 24.0 + 0.5 * c) + 0.5 * rng.normal()
            for c in range(channels)
        ]
        stamp = start + timedelta(hours=row)
        lines.append(stamp.isoformat() + "," + ",".join(f"{v:.6f}" for v in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_real_run_reports_null_gaps_and_ratio(tmp_path: Path) -> None:
    """A real (oracle-free) run: gaps null, E1 is the MSE ratio, and a missing
    CSV mapping degrades to an error record instead of a crash."""
    csv = _tiny_real_csv(tmp_path / "real.csv")
    runs_dir = tmp_path / "realruns"
    _run_loop(
        runs_dir / "real-s0",
        tmp_path / "heldout-root",
        arm="fixed",
        seed=0,
        budget=2,
        task="real-fake",
        real_csv=csv,
    )
    config = _write_cfg(tmp_path)
    report = posthoc.analyze_runs(runs_dir, config=config, real_csvs={"real-fake": csv})
    rec = report["runs"][0]
    assert rec["error"] is None, rec["error"]
    assert rec["task"] == "real-fake"
    assert rec["real"] is True
    rounds = _rounds(runs_dir / "real-s0")
    expected = ["m0"] + [row["candidateId"] for row in rounds if row["accepted"]]
    assert rec["incumbentChain"] == expected
    for inc in rec["incumbents"]:
        assert inc["gap"] is None  # no oracle on real tasks
        assert inc["mseOracle"] is None
        assert inc["normalizedGap"] is None
        assert inc["mseRatio"] is not None and inc["mseRatio"] > 0.0
    assert rec["e1Kind"] == "mseRatio"
    assert rec["e1"] == pytest.approx(rec["incumbents"][-1]["mseRatio"])
    assert rec["lcbGapCorrelation"] is None  # no gap to correlate against

    # Without the TASK=CSV mapping the run errors out but the report survives.
    missing = posthoc.analyze_runs(runs_dir, config=config, real_csvs={})
    assert missing["runs"][0]["error"] is not None
    assert "real-fake" in missing["runs"][0]["error"]
    assert missing["endpointTable"] == []


def test_probe_accounting_reports_the_probe_output_fraction(tmp_path: Path) -> None:
    """roundsWithProbeOutput is the FRACTION of rounds whose prompt carried
    at least one probe output (2 of 4 rounds -> 0.5), null for no rounds."""
    probes_dir = tmp_path / "ws" / "probes"
    probes_dir.mkdir(parents=True)
    (probes_dir / "validation.jsonl").write_text(
        json.dumps({"ok": True}) + "\n" + json.dumps({"ok": False}) + "\n", encoding="utf-8"
    )
    rounds = [
        {"probeOutputsRendered": 2, "probesSeconds": 1.0},
        {"probeOutputsRendered": 0, "probesSeconds": 0.5},
        {"probeOutputsRendered": 5, "probesSeconds": 2.0},
        {"probeOutputsRendered": 0, "probesSeconds": 0.5},
    ]
    accounting = posthoc._probe_accounting(tmp_path, rounds)
    assert accounting is not None
    assert accounting["written"] == 2
    assert accounting["validated"] == 1
    assert accounting["roundsWithProbeOutput"] == pytest.approx(0.5)
    assert accounting["probesSeconds"] == pytest.approx(4.0)
    empty = posthoc._probe_accounting(tmp_path, [])
    assert empty is not None
    assert empty["roundsWithProbeOutput"] is None


def test_judge_accounting_counts_calls_failures_and_tokens() -> None:
    rounds = [
        {
            "judgeCause": "acceptance",
            "judgeFailed": False,
            "judgeTokens": {"inputTokens": 10, "outputTokens": 2, "cacheReadTokens": 1, "reasoningTokens": 0},
        },
        {"judgeCause": None, "judgeFailed": False, "judgeTokens": None},
        {
            "judgeCause": "rejected-streak",
            "judgeFailed": True,
            "judgeTokens": {"inputTokens": 5, "outputTokens": 1, "cacheReadTokens": 0, "reasoningTokens": 3},
        },
    ]
    assert posthoc._judge_accounting(rounds) == {
        "calls": 2,
        "failed": 1,
        "tokens": {"inputTokens": 15, "outputTokens": 3, "cacheReadTokens": 1, "reasoningTokens": 3},
    }
    assert posthoc._judge_accounting([]) == {
        "calls": 0,
        "failed": 0,
        "tokens": {"inputTokens": 0, "outputTokens": 0, "cacheReadTokens": 0, "reasoningTokens": 0},
    }


def _card(*aspects: tuple[str, float]) -> list[dict[str, Any]]:
    return [
        {"label": f"row-{aspect}", "aspect": aspect, "delta": 1.0, "lcb": lcb, "ucb": lcb + 1.0}
        for aspect, lcb in aspects
    ]


def test_run_tokens_include_the_judge_tokens(
    synthetic_analysis: dict[str, Any], tmp_path: Path
) -> None:
    """The run's token totals cover both the candidate rounds and the judge
    calls: a run's judgeTokens are part of its LLM cost."""
    report = synthetic_analysis["report"]
    src = synthetic_analysis["runs_dir"] / report["runs"][0]["run"]
    dst = tmp_path / src.name
    shutil.copytree(src, dst)
    rounds = _rounds(dst)
    judge = {"inputTokens": 7, "outputTokens": 1, "cacheReadTokens": 0, "reasoningTokens": 0}
    for row in rounds:
        row["judgeTokens"] = judge
    (dst / "rounds.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rounds), encoding="utf-8"
    )
    patched = posthoc.analyze_runs(tmp_path, config=synthetic_analysis["config"])
    assert [rec["run"] for rec in patched["runs"]] == [src.name]
    rec = patched["runs"][0]
    assert rec["error"] is None, rec["error"]
    for key, expected_rounds in (
        ("inputTokens", 7 * len(rounds)),
        ("outputTokens", 1 * len(rounds)),
        ("cacheReadTokens", 0),
        ("reasoningTokens", 0),
    ):
        base = sum(int((row.get("tokens") or {}).get(key, 0)) for row in rounds)
        assert rec["tokens"][key] == base + expected_rounds, key


def test_evolution_accounting_post_switch_diversity() -> None:
    """postSwitchDiversity is the fraction of switches whose first acceptance
    closed a disjoint aspect family; null when no acceptance follows."""
    cards = {
        "m0": _card(("lag-structure", 10.0), ("periodicity", 8.0)),
        "c1": _card(("lag-structure", 4.0), ("periodicity", 8.0)),
        "c2": _card(("lag-structure", 4.0), ("periodicity", 8.0)),
        "c3": _card(("lag-structure", 3.0), ("periodicity", 8.0)),
        "c4": _card(("lag-structure", 3.0), ("periodicity", 8.0)),
        "c5": _card(("lag-structure", 3.0), ("periodicity", 2.0)),
    }
    rounds = [
        {"round": 1, "candidateId": "c1", "parent": "m0", "accepted": True},
        {"round": 2, "candidateId": "c2", "parent": "c1", "accepted": False,
         "inspirationParent": "x1"},
        {"round": 3, "candidateId": "c3", "parent": "c2", "accepted": True},
        {"round": 4, "candidateId": "c4", "parent": "c3", "accepted": False,
         "inspirationParent": "x2"},
        {"round": 5, "candidateId": "c5", "parent": "c4", "accepted": True},
    ]
    result = posthoc._evolution_accounting(rounds, lambda cid: cards.get(cid))
    assert result["switchCount"] == 2
    # switch 1: the first acceptance after it repeats lag-structure -> not disjoint;
    # switch 2: the first acceptance after it closes periodicity -> disjoint
    assert result["switches"][0]["disjoint"] is False
    assert result["switches"][1]["disjoint"] is True
    assert result["postSwitchDiversity"] == pytest.approx(0.5)

    # a switch with no acceptance after it is unmeasured: excluded from the
    # fraction, and with no measured switch at all the value is null
    tail = {"round": 6, "candidateId": "c6", "parent": "c5", "accepted": False,
            "inspirationParent": "x3"}
    only_unmeasured = posthoc._evolution_accounting(
        rounds + [tail], lambda cid: cards.get(cid)
    )
    assert only_unmeasured["switchCount"] == 3
    assert only_unmeasured["switches"][2]["disjoint"] is None
    assert only_unmeasured["postSwitchDiversity"] == pytest.approx(0.5)
    no_switches = posthoc._evolution_accounting(
        [row for row in rounds if not row.get("inspirationParent")],
        lambda cid: cards.get(cid),
    )
    assert no_switches["postSwitchDiversity"] is None


def test_m0_template_precedence_flag_over_recorded(synthetic_analysis: dict[str, Any]) -> None:
    """An explicit --m0-template wins over the run's recorded template; without
    the flag the recorded name decides what the replay loads for m0."""
    cfg = loop.HeadroomConfig.from_json(synthetic_analysis["config"])
    run_dir = synthetic_analysis["runs_dir"] / "fixed-s0"
    # the runs were made with the linear template, so that is what they record
    recorded = posthoc._read_json(
        Path(posthoc._read_json(run_dir / "heldout.json")["path"]) / "incumbent-reference.json"
    )["template"]
    assert recorded == "m0.py"

    replay = posthoc._Replay(run_dir, cfg, {}, None)  # no explicit flag
    assert "Initial parent model (m0)" in replay.candidate_state("m0")["source"]

    explicit = posthoc._Replay(
        run_dir, cfg, {}, posthoc._REPO_ROOT / "experiments" / "templates" / "m0_deep.py"
    )
    assert "Initial incumbent (m0-patchtst)" in explicit.candidate_state("m0")["source"]


def test_incomplete_runs_are_listed_not_averaged(synthetic_analysis: dict[str, Any]) -> None:
    """A run with a rounds ledger but no done.json is interrupted: excluded
    from the runs/endpoint table, and reported by name under "incomplete"."""
    # the fixture's own runs all carry done.json on their heldout side
    clean = posthoc.analyze_runs(
        synthetic_analysis["runs_dir"], config=synthetic_analysis["config"], real_csvs={}
    )
    assert clean["incomplete"] == []
    assert len(clean["runs"]) == 2

    src = synthetic_analysis["runs_dir"] / "fixed-s0"
    heldout_src = Path(posthoc._read_json(src / "heldout.json")["path"])
    scratch = synthetic_analysis["runs_dir"].parent / "interrupted"
    runs_dir = scratch / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    dst = runs_dir / "interrupted-s0"
    shutil.copytree(src, dst)
    heldout_dst = scratch / "heldout-interrupted"
    shutil.copytree(heldout_src, heldout_dst)
    (heldout_dst / "done.json").unlink()  # the interruption marker
    (dst / "heldout.json").write_text(json.dumps({"path": str(heldout_dst)}), encoding="utf-8")

    report = posthoc.analyze_runs(runs_dir, config=synthetic_analysis["config"], real_csvs={})
    assert report["runs"] == []
    assert report["incomplete"] == ["interrupted-s0"]
    assert report["endpointTable"] == []
