from __future__ import annotations

import asyncio
from pathlib import Path
import json
import sys
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_loop():
    source = _REPO_ROOT / "experiments" / "loop.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("rhg_loop", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rhg_loop"] = module
    spec.loader.exec_module(module)
    return module


def _small_cfg(loop_module: Any):
    return loop_module.HeadroomConfig(
        arm="fixed",
        input_length=48,
        horizon=8,
        lag_segments=((0, 7), (8, 15), (16, 31)),
        fir_len_cap=48,
        beyond_l_lags=2,
        band_period_range=(4, 24),
        budget_candidates=3,
        candidate_timeout_seconds=60.0,
        ridge_alphas=(1.0, 100.0),
    )


SMALL_SIZES = {"train": 1500, "diag": 1100, "cert": 1300, "test": 1800}


class ScriptedRunner:
    """A fake LLM session whose side effects come from a fixed script."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def send(self, prompt: str) -> list[Any]:
        self.calls.append(prompt)
        return []

    async def close(self) -> None:
        return None


def _better_candidate(ws: Path, index: int) -> tuple[str, dict[str, Any]]:
    """A self-contained low-regularization variant of the incumbent.

    Candidates must be single self-contained files, so the incumbent source
    is copied and textually patched: less ridge shrinkage plus a model_id
    that matches the candidate directory.
    """
    incumbent = (ws / "models" / "incumbent" / "model.py").read_text(encoding="utf-8")
    candidate_id = f"c-lowreg-{index}"
    source = incumbent.replace("_RIDGE_REG = 1000.0", "_RIDGE_REG = 10.0")
    patch = (
        "\n\nclass PatchedAdapter(M0Adapter):\n"
        "    def __init__(self) -> None:\n"
        "        M0Adapter.__init__(self)\n"
        f"        self.model_id = \"{candidate_id}\"\n"
        "\n"
        "\n"
        "def create_adapter() -> M0Adapter:\n"
        "    return PatchedAdapter()\n"
    )
    source = source + patch
    proposal = {
        "candidateId": candidate_id,
        "targetAspect": "lag-structure",
        "aspectDetail": "recent lags",
        "mechanism": "less ridge shrinkage",
        "expectation": "diag MSE drops",
    }
    return source, proposal


def _make_factory(script: list[Any]):
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


def _rounds(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "rounds.jsonl").read_text().splitlines()]


def test_loop_runs_three_rounds_and_writes_ledger(tmp_path: Path) -> None:
    loop_module = _load_loop()
    factory = _make_factory([_better_candidate, _better_candidate, _better_candidate])
    run_dir = tmp_path / "run1"
    loop = loop_module.HeadroomLoop(
        run_dir=run_dir,
        task_name="lgssm_xchannel",
        arm="fixed",
        seed=0,
        cfg=_small_cfg(loop_module),
        m0_template="m0.py",
        sizes=SMALL_SIZES,
        runner_factory=factory,
    )
    asyncio.run(loop.run())

    rounds = _rounds(run_dir)
    assert len(rounds) == 3
    assert all(row["status"] == "evaluated" for row in rounds)
    assert all(np.isfinite(row["diagMse"]) for row in rounds)
    assert len({row["candidateId"] for row in rounds}) == 3


def test_loop_resume_continues_from_ledger(tmp_path: Path) -> None:
    loop_module = _load_loop()
    factory = _make_factory([_better_candidate, _better_candidate, _better_candidate])
    run_dir = tmp_path / "run2"
    common = dict(
        run_dir=run_dir,
        task_name="lgssm_xchannel",
        arm="fixed",
        seed=0,
        cfg=_small_cfg(loop_module),
        m0_template="m0.py",
        sizes=SMALL_SIZES,
    )
    asyncio.run(loop_module.HeadroomLoop(runner_factory=factory, **common).run())
    assert len(_rounds(run_dir)) == 3

    # Resume with a fresh instance: budget spent, no new rounds.
    factory2 = _make_factory([])
    asyncio.run(loop_module.HeadroomLoop(runner_factory=factory2, **common).run())
    assert len(_rounds(run_dir)) == 3


def test_loop_repair_round_recovers_within_the_round(tmp_path: Path) -> None:
    """A round whose first send writes nothing recovers via a repair send.

    Repairs happen INSIDE the round (up to MAX_REPAIRS extra sends), so the
    ledger still records one evaluated round.
    """
    loop_module = _load_loop()
    script: list[Any] = [None, _better_candidate, _better_candidate, _better_candidate]
    factory = _make_factory(script)
    run_dir = tmp_path / "run3"
    loop = loop_module.HeadroomLoop(
        run_dir=run_dir,
        task_name="lgssm_xchannel",
        arm="scalar",
        seed=0,
        cfg=_small_cfg(loop_module),
        m0_template="m0.py",
        sizes=SMALL_SIZES,
        runner_factory=factory,
    )
    asyncio.run(loop.run())
    rounds = _rounds(run_dir)
    assert len(rounds) == 3
    assert all(row["status"] == "evaluated" for row in rounds)


def test_loop_round_fails_after_exhausted_repairs(tmp_path: Path) -> None:
    loop_module = _load_loop()
    factory = _make_factory([None, None, None])
    run_dir = tmp_path / "run3b"
    loop = loop_module.HeadroomLoop(
        run_dir=run_dir,
        task_name="lgssm_xchannel",
        arm="scalar",
        seed=0,
        cfg=_small_cfg(loop_module),
        m0_template="m0.py",
        sizes=SMALL_SIZES,
        runner_factory=factory,
    )
    asyncio.run(loop.run())
    rounds = _rounds(run_dir)
    assert len(rounds) == 3
    assert rounds[0]["status"] == "failed"
    assert rounds[0]["candidateId"] == ""
    # Every round consumed exactly 1 + MAX_REPAIRS sends of nothing.
    assert all(row["status"] == "failed" for row in rounds)


def test_loop_heldout_records_and_isolation(tmp_path: Path) -> None:
    loop_module = _load_loop()
    factory = _make_factory([_better_candidate] * 3)
    run_dir = tmp_path / "run4"
    loop = loop_module.HeadroomLoop(
        run_dir=run_dir,
        task_name="lgssm_xchannel",
        arm="fixed",
        seed=0,
        cfg=_small_cfg(loop_module),
        m0_template="m0.py",
        sizes=SMALL_SIZES,
        runner_factory=factory,
    )
    asyncio.run(loop.run())
    rounds = _rounds(run_dir)
    n_accepted = sum(1 for row in rounds if row["accepted"])
    heldout = loop.heldout
    cert_path = heldout / "certification.jsonl"
    # Round-0 certification of m0 must exist even when nothing is accepted —
    # otherwise a zero-acceptance run would have no stop point at all.
    assert cert_path.is_file()
    cert_lines = [json.loads(line) for line in cert_path.read_text().splitlines()]
    assert len(cert_lines) == n_accepted + 1
    assert cert_lines[0]["incumbentId"] == "m0"
    for row in cert_lines:
        assert "saturated" in row and "level" in row
    posthoc_files = list((heldout / "posthoc").glob("*.json"))
    assert len(posthoc_files) >= sum(1 for row in rounds if row["status"] == "evaluated")
    # The heldout directory lives outside the agent-visible workspace.
    assert heldout.resolve().is_relative_to(Path.home())
    assert not heldout.resolve().is_relative_to(run_dir.resolve())


def test_empty_dsh_turn_is_fatal_not_silent(tmp_path: Path) -> None:
    """A turn that produces no parts (billing/auth rejection) must raise
    after the retries, not burn the budget as silent no-op rounds."""
    loop_module = _load_loop()

    class _EmptySdkRunner:
        """A duck-typed SdkRunner whose turns produce nothing (the observed
        billing-death signature: the endpoint rejects every request)."""

        async def send(self, prompt: str) -> list[Any]:
            return []

        async def close(self) -> None:
            return None

    def factory(ws: Path, system_prompt: str, round_index: int) -> Any:
        # Route through the real DSH wrapper: the emptiness check lives there.
        return loop_module._SdkRoundRunner(_EmptySdkRunner())

    loop = loop_module.HeadroomLoop(
        run_dir=tmp_path / "run5",
        task_name="lgssm_xchannel",
        arm="fixed",
        seed=0,
        cfg=_small_cfg(loop_module),
        m0_template="m0.py",
        sizes=SMALL_SIZES,
        runner_factory=factory,
    )
    with pytest.raises(RuntimeError, match="DSH runtime failed") as excinfo:
        asyncio.run(loop.run())
    # The root cause must name the empty-turn billing signature.
    assert "no parts" in str(excinfo.value.__cause__)


# ------------------------------------------------------------------ probes

_AC24_PROBE = '''\
import numpy as np


class Ac24Probe:
    def __init__(self, probe_id: str) -> None:
        self.probe_id = probe_id

    def run(self, artifacts: dict) -> dict:
        e = np.asarray(artifacts["residuals"]["e"], dtype=np.float64)
        e = e[~np.isnan(e).any(axis=1)]
        x = e[:, 0] - e[:, 0].mean()
        denom = float(np.mean(x * x))
        if denom <= 0.0 or x.shape[0] <= 24:
            return {"ac24": 0.0}
        return {"ac24": float(np.mean(x[24:] * x[:-24]) / denom)}


def create_probe() -> Ac24Probe:
    return Ac24Probe("__PID__")
'''


def _probe_proposal(pid: str, direction: str) -> dict[str, Any]:
    return {
        "probeId": pid,
        "kind": "residual",
        "measures": "lag-24 autocorrelation of the 1-step residual",
        "signalKey": "ac24",
        "expectedDirection": direction,
        "rationale": "leftover period-24 structure means the model missed a cycle",
    }


def test_self_built_arm_validates_and_runs_probes(tmp_path: Path) -> None:
    """The self-built round trip: write probes -> fixture validation (good one
    kept, wrong-direction one rejected) -> verdicts + probe output in the next
    round's prompt, and none of the fixed card's content anywhere."""
    loop_module = _load_loop()
    script_pos = {"pos": 0}
    prompts: list[str] = []

    def factory(ws: Path, system_prompt: str, round_index: int) -> Any:
        class _Runner(ScriptedRunner):
            async def send(self, prompt: str) -> list[Any]:
                prompts.append(prompt)
                pos = script_pos["pos"]
                script_pos["pos"] += 1
                model_source, proposal = _better_candidate(ws, pos)
                candidate_dir = ws / "models" / "candidates" / proposal["candidateId"]
                candidate_dir.mkdir(parents=True, exist_ok=True)
                (candidate_dir / "model.py").write_text(model_source, encoding="utf-8")
                (candidate_dir / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
                if pos == 0:
                    for pid, direction in (("p-ac24", "higher"), ("p-ac24-bad", "lower")):
                        pdir = ws / "probes" / pid
                        pdir.mkdir(parents=True, exist_ok=True)
                        (pdir / "probe.py").write_text(
                            _AC24_PROBE.replace("__PID__", pid), encoding="utf-8"
                        )
                        (pdir / "probe_proposal.json").write_text(
                            json.dumps(_probe_proposal(pid, direction)), encoding="utf-8"
                        )
                return []

        return _Runner()

    run_dir = tmp_path / "run-probes"
    loop = loop_module.HeadroomLoop(
        run_dir=run_dir,
        task_name="lgssm_xchannel",
        arm="self-built",
        seed=0,
        cfg=_small_cfg(loop_module),
        sizes=SMALL_SIZES,
        runner_factory=factory,
        m0_template="m0.py",
    )
    asyncio.run(loop.run())

    # Round 1 saw no probes yet: the guide and the bootstrap message, no
    # validation section, and none of the fixed card's content.
    assert "Building your own diagnostic probes" in prompts[0]
    assert "No validated probes yet" in prompts[0]
    assert "## Probe validation results" not in prompts[0]
    assert "Remaining improvement by coordinate" not in prompts[0]

    # Both probes were validated after the first turn; only the honest one
    # entered the library.
    library = json.loads((run_dir / "ws" / "probes" / "library.json").read_text(encoding="utf-8"))
    assert [row["probeId"] for row in library] == ["p-ac24"]
    validations = [
        json.loads(line)
        for line in (run_dir / "ws" / "probes" / "validation.jsonl").read_text().splitlines()
    ]
    assert {row["probeId"]: row["ok"] for row in validations} == {"p-ac24": True, "p-ac24-bad": False}
    assert validations[1]["detail"]  # the rejection carries a reason

    # Round 2's prompt shows both verdicts and the kept probe's live output.
    assert "| p-ac24 | kept |" in prompts[1]
    assert "| p-ac24-bad | rejected |" in prompts[1]
    assert "| p-ac24 | lag-24 autocorrelation of the 1-step residual |" in prompts[1]
    assert '"ac24"' in prompts[1]

    rounds = _rounds(run_dir)
    assert rounds[0]["probesValidated"] == 2
    assert rounds[1]["probesValidated"] == 0
    assert rounds[2]["probesValidated"] == 0
    # The candidate side of the round is untouched by probe writing.
    assert all(row["status"] == "evaluated" for row in rounds)


def test_self_built_samples_activations_with_alignment(tmp_path: Path) -> None:
    """The activation artifact: [B, P, D] per layer, aligned to residual rows,
    and absent (not an error) for incumbents without the extension."""
    loop_module = _load_loop()
    loop = loop_module.HeadroomLoop(
        run_dir=tmp_path / "run-act",
        task_name="lgssm_xchannel",
        arm="self-built",
        seed=0,
        cfg=_small_cfg(loop_module),
        sizes=SMALL_SIZES,
        m0_template="m0.py",
    )
    loop.prepare()
    loop.load_state()
    horizon = loop.cfg.horizon
    channels = loop.seg_diag.values.shape[1]

    class _Stub:
        model_id = "stub"

        def predict_batch(self, x: np.ndarray, return_layers: bool = False) -> Any:
            assert return_layers
            b = x.shape[0]
            return (
                np.zeros((b, horizon, channels)),
                {"layers": [("b0", np.ones((b, 3, 4), dtype=np.float32))]},
            )

    loop._sample_incumbent_activations(_Stub())
    assert loop._activations is not None and set(loop._activations) == {"b0"}
    assert loop._activations["b0"].shape == (256, 3, 4)
    positions = loop._activation_origins
    assert positions.shape == (256,)
    # alignment: activations[i] belongs to residual row positions[i]
    np.testing.assert_array_equal(
        loop.seg_diag.origins[positions][:5], loop.seg_diag.origins[positions][:5]
    )
    assert positions[0] == 0 and positions[-1] == loop.seg_diag.origins.size - 1

    artifacts = loop._probe_artifacts()
    assert set(artifacts["activations"]) == {"b0"}
    assert artifacts["meta"]["activationOrigins"] == positions.tolist()
    assert artifacts["residuals"]["r"].shape[0] == loop.seg_diag.r.shape[0]
    assert artifacts["meta"]["task"] == "synthetic"

    class _Plain:
        model_id = "plain"

        def predict_batch(self, x: np.ndarray, return_layers: bool = False) -> np.ndarray:
            # Accepts the extension keyword but returns no layer dump: the
            # loop must treat the incumbent as activation-less, not crash.
            return np.zeros((x.shape[0], horizon, channels))

    loop._sample_incumbent_activations(_Plain())
    assert loop._activations is None
    artifacts = loop._probe_artifacts()
    assert artifacts["activations"] == {}
    assert artifacts["meta"]["activationOrigins"] == []


# ---------------------------------------------------------------- evolution

def _write_card(run_heldout: Path, candidate_id: str, open_aspects: dict[str, float]) -> None:
    """A heldout posthoc card whose aspects carry the given lcb values."""
    rows = [
        {
            "label": f"l{i}",
            "aspect": aspect,
            "delta": lcb,
            "lcb": lcb,
            "ucb": lcb + 0.01,
            "mu2": 0.0,
            "by_horizon": [],
        }
        for i, (aspect, lcb) in enumerate(open_aspects.items())
    ]
    (run_heldout / "posthoc").mkdir(parents=True, exist_ok=True)
    (run_heldout / "posthoc" / f"{candidate_id}.json").write_text(
        json.dumps({"round": 1, "proposal": {}, "diagMse": 0.5, "dmP": None, "card": rows}),
        encoding="utf-8",
    )


def _evolution_loop(loop_module: Any, tmp_path: Path, arm: str = "fixed"):
    loop = loop_module.HeadroomLoop(
        run_dir=tmp_path / "run-evo",
        task_name="lgssm_xchannel",
        arm=arm,
        seed=0,
        cfg=_small_cfg(loop_module),
        sizes=SMALL_SIZES,
        m0_template="m0.py",
    )
    loop.prepare()
    loop.load_state()  # segments are needed by _render_round; ledger stays read-only
    return loop


def _ledger_row(round_index: int, cid: str, parent: str, accepted: bool, diag_mse: float) -> str:
    return json.dumps(
        {
            "round": round_index,
            "candidateId": cid,
            "parent": parent,
            "arm": "fixed",
            "status": "evaluated",
            "diagMse": diag_mse,
            "dmP": 0.01,
            "accepted": accepted,
            "error": None,
            "proposal": {},
            "tokens": {},
            "feedbackChars": 100,
            "attempts": 1,
            "candidateSeconds": {"fit": 1.0, "predict": 0.1},
            "inspirationParent": None,
            "probesValidated": 0,
            "seconds": 2.0,
        }
    )


def test_evolution_switch_fires_on_concentration(tmp_path: Path) -> None:
    loop_module = _load_loop()
    loop = _evolution_loop(loop_module, tmp_path)
    heldout = loop.heldout
    # m0 leaves periodicity and lag-structure open.
    _write_card(heldout, "m0", {"periodicity": 0.05, "lag-structure": 0.04})
    # Three consecutive acceptances each chip at lag-structure (the largest
    # lcb drop) without closing anything: the concentration trigger.
    _write_card(heldout, "c1", {"periodicity": 0.05, "lag-structure": 0.02})
    _write_card(heldout, "c2", {"periodicity": 0.05, "lag-structure": 0.012})
    _write_card(heldout, "c3", {"periodicity": 0.05, "lag-structure": 0.006})
    # An unaccepted branch that closed periodicity: the different signature.
    _write_card(heldout, "c-other", {"lag-structure": 0.04})
    _write_card(heldout, "c-lag", {"lag-structure": 0.045})
    rows = [
        _ledger_row(1, "c1", "m0", True, 0.49),
        _ledger_row(2, "c2", "c1", True, 0.48),
        _ledger_row(3, "c3", "c2", True, 0.47),
        _ledger_row(4, "c-other", "m0", False, 0.50),
        _ledger_row(5, "c-lag", "m0", False, 0.55),
    ]
    loop.rounds_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    state = loop_module.LoopState(round_index=6, incumbent_id="c3", history=[])

    rec = loop._evolution_recommendation(state)
    assert rec == ("c-other", "concentrated")
    block = loop._parent_block(state, rec)
    assert "models/candidates/c-other/model.py" in block
    assert "0.5000" in block
    # Neutral wording: the aspect names stay heldout.
    assert "lag-structure" not in block and "periodicity" not in block
    # And the rendered round carries the switch while the no-switch render
    # does not.
    with_switch = loop._render_round(state, "fb", "", rec)
    assert "Diversity switch" in with_switch
    without = loop._render_round(state, "fb", "", None)
    assert "Diversity switch" not in without


def test_evolution_switch_fires_on_cert_saturation(tmp_path: Path) -> None:
    loop_module = _load_loop()
    loop = _evolution_loop(loop_module, tmp_path)
    heldout = loop.heldout
    _write_card(heldout, "m0", {"periodicity": 0.05, "lag-structure": 0.04})
    _write_card(heldout, "c-other", {"lag-structure": 0.04})
    loop.rounds_path.write_text(
        _ledger_row(1, "c-other", "m0", False, 0.50) + "\n", encoding="utf-8"
    )
    (heldout / "certification.jsonl").write_text(
        json.dumps({"round": 0, "incumbentId": "m0", "saturated": True, "level": 0.05, "failures": []})
        + "\n",
        encoding="utf-8",
    )
    state = loop_module.LoopState(round_index=2, incumbent_id="m0", history=[])
    rec = loop._evolution_recommendation(state)
    assert rec == ("c-other", "saturated")
    block = loop._parent_block(state, rec)
    assert "below the actionable threshold" in block


def test_evolution_stays_silent_without_a_trigger(tmp_path: Path) -> None:
    loop_module = _load_loop()
    loop = _evolution_loop(loop_module, tmp_path)
    heldout = loop.heldout
    _write_card(heldout, "m0", {"periodicity": 0.05, "lag-structure": 0.04})
    _write_card(heldout, "c1", {"lag-structure": 0.04})
    _write_card(heldout, "c2", {"periodicity": 0.05})
    # Only two acceptances (need three) and no saturation.
    loop.rounds_path.write_text(
        _ledger_row(1, "c1", "m0", True, 0.49) + "\n" + _ledger_row(2, "c2", "c1", True, 0.48) + "\n",
        encoding="utf-8",
    )
    state = loop_module.LoopState(round_index=3, incumbent_id="c2", history=[])
    assert loop._evolution_recommendation(state) is None
    assert loop._parent_block(state, None) == ""


def test_evolution_disabled_by_config(tmp_path: Path) -> None:
    loop_module = _load_loop()
    cfg = loop_module.HeadroomConfig(
        arm="fixed",
        input_length=48,
        horizon=8,
        lag_segments=((0, 7), (8, 15), (16, 31)),
        fir_len_cap=48,
        beyond_l_lags=2,
        band_period_range=(4, 24),
        budget_candidates=3,
        candidate_timeout_seconds=60.0,
        ridge_alphas=(1.0, 100.0),
        evolution=False,
    )
    loop = loop_module.HeadroomLoop(
        run_dir=tmp_path / "run-evo-off",
        task_name="lgssm_xchannel",
        arm="fixed",
        seed=0,
        cfg=cfg,
        sizes=SMALL_SIZES,
        m0_template="m0.py",
    )
    loop.prepare()
    (loop.heldout / "certification.jsonl").write_text(
        json.dumps({"round": 0, "incumbentId": "m0", "saturated": True, "level": 0.05, "failures": []})
        + "\n",
        encoding="utf-8",
    )
    state = loop_module.LoopState(round_index=1, incumbent_id="m0", history=[])
    assert loop._evolution_recommendation(state) is None
