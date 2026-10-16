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


def _small_cfg(loop_module: Any, evolution: bool = False):
    return loop_module.HeadroomConfig(
        input_length=48,
        horizon=8,
        lag_segments=((0, 7), (8, 15), (16, 31)),
        fir_len_cap=48,
        beyond_l_lags=2,
        band_period_range=(4, 24),
        budget_candidates=3,
        candidate_timeout_seconds=60.0,
        ridge_alphas=(1.0, 100.0),
        evolution=evolution,
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
    assert rounds[0]["probesSubmitted"] == 2
    assert rounds[0]["probesKept"] == 1
    assert rounds[1]["probesSubmitted"] == 0
    assert rounds[2]["probesSubmitted"] == 0
    # The library renders from round 2 on; validation cost wall time in round 1.
    assert rounds[0]["probeOutputsRendered"] == 0
    assert rounds[1]["probeOutputsRendered"] == 1
    assert rounds[0]["probesSeconds"] > 0
    # The candidate side of the round is untouched by probe writing.
    assert all(row["status"] == "evaluated" for row in rounds)


def test_validated_probe_runs_from_the_heldout_frozen_copy(tmp_path: Path) -> None:
    """The library runs the source that passed validation, not whatever the
    agent's write tools later left in ws/probes/<id>/probe.py."""
    loop_module = _load_loop()
    script_pos = {"pos": 0}

    def factory(ws: Path, system_prompt: str, round_index: int) -> Any:
        class _Runner(ScriptedRunner):
            async def send(self, prompt: str) -> list[Any]:
                pos = script_pos["pos"]
                script_pos["pos"] += 1
                model_source, proposal = _better_candidate(ws, pos)
                candidate_dir = ws / "models" / "candidates" / proposal["candidateId"]
                candidate_dir.mkdir(parents=True, exist_ok=True)
                (candidate_dir / "model.py").write_text(model_source, encoding="utf-8")
                (candidate_dir / "proposal.json").write_text(json.dumps(proposal), encoding="utf-8")
                if pos == 0:
                    pdir = ws / "probes" / "p-ac24"
                    pdir.mkdir(parents=True, exist_ok=True)
                    (pdir / "probe.py").write_text(
                        _AC24_PROBE.replace("__PID__", "p-ac24"), encoding="utf-8"
                    )
                    (pdir / "probe_proposal.json").write_text(
                        json.dumps(_probe_proposal("p-ac24", "higher")), encoding="utf-8"
                    )
                return []

        return _Runner()

    run_dir = tmp_path / "run-freeze"
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

    ws_source = run_dir / "ws" / "probes" / "p-ac24" / "probe.py"
    frozen = Path(
        json.loads((run_dir / "heldout.json").read_text(encoding="utf-8"))["path"]
    ) / "probe-sources" / "p-ac24" / "probe.py"
    assert frozen.is_file()
    # Tamper with the workspace copy after validation.
    ws_source.write_text(
        (run_dir / "ws" / "probes" / "p-ac24" / "probe.py").read_text(encoding="utf-8").replace(
            "ac24", "tampered"
        ),
        encoding="utf-8",
    )
    artifacts = loop._probe_artifacts()
    output = loop._probe_infra().run_probe(frozen, artifacts)
    assert "ac24" in output  # the frozen source, not the tampered ws copy


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
    # Evenly spaced over the segment's residual rows (the alignment claim:
    # activations[i] belongs to residual row positions[i]).
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


# ----------------------------------------------------------- diversity judge

_SWITCH_VERDICT = json.dumps(
    {
        "switch": True,
        "parent": "c-other",
        "reason": "The recent chain keeps chipping at the same kind of structure; "
        "try the cross-channel direction.",
    }
)
_NO_SWITCH_VERDICT = json.dumps(
    {"switch": False, "parent": None, "reason": "continue from the incumbent."}
)


class _ScriptedJudge:
    """Mock judge session: scripted parts (or an exception) per send."""

    def __init__(self, script: list[Any], calls: list[str]) -> None:
        self._script = script
        self._calls = calls

    async def send(self, prompt: str) -> list[Any]:
        self._calls.append(prompt)
        if not self._script:
            return [_NO_SWITCH_VERDICT]
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        return [step]

    async def close(self) -> None:
        return None


def _judge_factory_for(script: list[Any], calls: list[str]) -> Any:
    def factory(judge_ws: Path, system_prompt: str) -> Any:
        return _ScriptedJudge(script, calls)

    return factory


def _judge_loop(
    loop_module: Any,
    tmp_path: Path,
    ledger_rows: list[dict[str, Any]],
    *,
    cert_rows: list[dict[str, Any]] | None = None,
    script: list[Any] | None = None,
    evolution: bool = True,
) -> tuple[Any, Any, list[str]]:
    """A prepared loop with a crafted ledger and a scripted judge."""
    calls: list[str] = []
    loop = loop_module.HeadroomLoop(
        run_dir=tmp_path / "run-judge",
        task_name="lgssm_xchannel",
        arm="fixed",
        seed=0,
        cfg=_small_cfg(loop_module, evolution=evolution),
        sizes=SMALL_SIZES,
        judge_factory=_judge_factory_for(script or [], calls),
        m0_template="m0.py",
    )
    loop.prepare()
    loop.load_state()
    loop.rounds_path.write_text(
        "\n".join(json.dumps(row) for row in ledger_rows) + "\n", encoding="utf-8"
    )
    if cert_rows is not None:
        (loop.heldout / "certification.jsonl").write_text(
            "\n".join(json.dumps(row) for row in cert_rows) + "\n", encoding="utf-8"
        )
    state = loop_module.LoopState(
        round_index=len(ledger_rows) + 1,
        incumbent_id=str(ledger_rows[-1]["candidateId"]) if ledger_rows[-1].get("accepted") else "m0",
        history=[],
    )
    return loop, state, calls


def _row(
    round_index: int, cid: str, parent: str, accepted: bool, diag_mse: float = 0.5
) -> dict[str, Any]:
    return {
        "round": round_index,
        "candidateId": cid,
        "parent": parent,
        "arm": "fixed",
        "status": "evaluated",
        "diagMse": diag_mse,
        "dmP": 0.01,
        "accepted": accepted,
        "error": None,
        "proposal": {
            "targetAspect": "lag-structure",
            "aspectDetail": "recent lags",
            "mechanism": "less ridge shrinkage",
        },
        "tokens": {},
        "inspirationParent": None,
        "judgeCause": None,
        "probesSubmitted": 0,
        "seconds": 2.0,
    }


def test_judge_triggers_fire_from_pure_ledger_rules(tmp_path: Path) -> None:
    loop_module = _load_loop()
    cert = [{"round": 0, "incumbentId": "m0", "saturated": True, "level": 0.95, "failures": []}]

    def rows(n: int, accepted_last: bool = False) -> list[dict[str, Any]]:
        out = [_row(i + 1, f"c{i + 1}", "m0", False) for i in range(n - 1)]
        out.append(_row(n, f"c{n}", "m0", accepted_last))
        return out

    cases = {
        "acceptance": (rows(2, accepted_last=True), None),
        "rejections": (rows(3, accepted_last=False), None),
        # fewer than three rejections, but the cert is saturated and no
        # judge call happened at or after that certification round
        "saturation": (rows(1, accepted_last=False), cert),
        "none": (rows(1, accepted_last=False), []),
    }
    for cause, (ledger, certs) in cases.items():
        loop, state, calls = _judge_loop(loop_module, tmp_path / cause, ledger, cert_rows=certs)
        verdict = asyncio.run(loop._judge_recommendation(state))
        if cause == "none":
            assert not calls and verdict is None
            continue
        assert calls, f"{cause}: the judge must be called"
        assert loop._last_judge["cause"] == cause
    # A standing saturated state does not re-ask after a call was recorded.
    ledger = rows(1)
    ledger[0]["judgeCause"] = "saturation"
    loop, state, calls = _judge_loop(loop_module, tmp_path / "standing", ledger, cert_rows=cert)
    asyncio.run(loop._judge_recommendation(state))
    assert not calls


def test_judge_valid_switch_renders_in_the_round_prompt(tmp_path: Path) -> None:
    loop_module = _load_loop()
    ledger = [
        _row(1, "c1", "m0", False),
        _row(2, "c-other", "m0", False, 0.42),
        _row(3, "c2", "m0", True),
    ]
    loop, state, calls = _judge_loop(
        loop_module, tmp_path, ledger, script=[_SWITCH_VERDICT]
    )
    verdict = asyncio.run(loop._judge_recommendation(state))
    assert verdict is not None
    parent, reason, cause = verdict
    assert (parent, cause) == ("c-other", "acceptance")
    assert "cross-channel direction" in reason
    # The digest carried the ledger and the measured card for the judge.
    assert "c-other" in calls[0] and "diagMse" in calls[0]
    assert loop._last_judge["failed"] is False
    prompt = loop._render_round(state, "feedback", "", verdict)
    assert "Diversity switch" in prompt
    assert "models/candidates/c-other/model.py" in prompt
    assert "0.4200" in prompt  # the MSE comes from the ledger, not the judge
    assert reason in prompt


def test_judge_invalid_parent_falls_back_to_greedy(tmp_path: Path) -> None:
    loop_module = _load_loop()
    bogus = json.dumps(
        {"switch": True, "parent": "c-does-not-exist", "reason": "try a different direction"}
    )
    ledger = [
        _row(1, "c1", "m0", False),
        _row(2, "c-other", "m0", False, 0.42),
        _row(3, "c2", "m0", True),
    ]
    loop, state, calls = _judge_loop(
        loop_module, tmp_path, ledger, script=[bogus, bogus]
    )
    verdict = asyncio.run(loop._judge_recommendation(state))
    assert verdict is None
    assert len(calls) == 2  # one retry, then the guardrail gives up
    assert loop._last_judge["failed"] is True
    assert loop._parent_block(state, None) == ""


def test_judge_transient_failure_retries_then_greedy(tmp_path: Path) -> None:
    loop_module = _load_loop()
    ledger = [
        _row(1, "c1", "m0", False),
        _row(2, "c-other", "m0", False, 0.42),
        _row(3, "c2", "m0", True),
    ]
    loop, state, calls = _judge_loop(
        loop_module, tmp_path, ledger, script=[RuntimeError("endpoint down"), _SWITCH_VERDICT]
    )
    verdict = asyncio.run(loop._judge_recommendation(state))
    assert verdict is not None  # the retry succeeded
    assert len(calls) == 2
    assert loop._last_judge["failed"] is False
    loop, state, calls = _judge_loop(
        loop_module,
        tmp_path / "both-fail",
        ledger,
        script=[RuntimeError("endpoint down"), TimeoutError("slow")],
    )
    assert asyncio.run(loop._judge_recommendation(state)) is None
    assert loop._last_judge["failed"] is True


def test_judge_reason_leak_filter_rejects_numbers_and_status_words(tmp_path: Path) -> None:
    loop_module = _load_loop()
    leaky_digits = json.dumps(
        {"switch": True, "parent": "c-other", "reason": "the last candidate improved by 12 percent"}
    )
    leaky_status = json.dumps(
        {"switch": True, "parent": "c-other", "reason": "the incumbent line is now saturated"}
    )
    ledger = [
        _row(1, "c1", "m0", False),
        _row(2, "c-other", "m0", False, 0.42),
        _row(3, "c2", "m0", True),
    ]
    loop, state, calls = _judge_loop(
        tmp_path=tmp_path / "digits",
        loop_module=loop_module,
        ledger_rows=ledger,
        script=[leaky_digits, _SWITCH_VERDICT],
    )
    assert asyncio.run(loop._judge_recommendation(state)) is not None
    assert len(calls) == 2  # the leaky verdict was discarded, the retry won
    loop, state, calls = _judge_loop(
        tmp_path=tmp_path / "status",
        loop_module=loop_module,
        ledger_rows=ledger,
        script=[leaky_status, leaky_status],
    )
    assert asyncio.run(loop._judge_recommendation(state)) is None
    assert loop._last_judge["failed"] is True


def test_judge_reads_heldout_cards_in_the_digest(tmp_path: Path) -> None:
    loop_module = _load_loop()
    ledger = [
        _row(1, "c1", "m0", False),
        _row(2, "c-other", "m0", False, 0.42),
        _row(3, "c2", "m0", True),
    ]
    loop, state, calls = _judge_loop(loop_module, tmp_path, ledger, script=[_SWITCH_VERDICT])
    card = {
        "round": 2,
        "proposal": {},
        "diagMse": 0.42,
        "dmP": None,
        "card": [
            {"label": "lag", "aspect": "lag-structure", "delta": 0.02, "lcb": 0.011,
             "ucb": 0.03, "mu2": 0.0, "by_horizon": []}
        ],
    }
    (loop.heldout / "posthoc").mkdir(parents=True, exist_ok=True)
    (loop.heldout / "posthoc" / "c2.json").write_text(json.dumps(card), encoding="utf-8")
    digest = loop._evolution_digest(state)
    assert "remaining improvement by aspect: lag-structure: 0.0110" in digest
    assert "claimed: lag-structure; recent lags; less ridge shrinkage" in digest
    # The judge workspace lives inside heldout, never under ws/.
    assert str(loop._judge_ws()).startswith(str(loop.heldout))


def test_judge_disabled_by_config_never_calls(tmp_path: Path) -> None:
    loop_module = _load_loop()
    ledger = [
        _row(1, "c1", "m0", False),
        _row(2, "c-other", "m0", False, 0.42),
        _row(3, "c2", "m0", True),
    ]
    loop, state, calls = _judge_loop(
        loop_module, tmp_path, ledger, script=[_SWITCH_VERDICT], evolution=False
    )
    assert asyncio.run(loop._judge_recommendation(state)) is None
    assert not calls  # cfg.evolution=False: not even one session is opened


_MEAN_CANDIDATE = """\
import numpy as np


class MeanPredictor:
    def __init__(self) -> None:
        self.model_id = "__CID__"
        self._mean = None
        self._H = 0

    def fit(self, train, meta):
        self._mean = np.asarray(train, dtype=np.float64).mean(axis=0)
        self._H = int(meta["H"])

    def predict_batch(self, x):
        b = x.shape[0]
        out = np.broadcast_to(self._mean[None, None, :], (b, self._H, self._mean.shape[0]))
        return np.array(out, dtype=np.float64)


def create_adapter() -> MeanPredictor:
    return MeanPredictor()
"""


def test_deep_incumbent_round_trip(tmp_path: Path) -> None:
    """The v2 default incumbent (PatchTST-lite) must traverse the real loop
    path: in-process fit in load_state, round feedback over its residuals,
    the round-0 deep posthoc card, and candidate evaluation against it.
    Every torch candidate in P2 v2 derives from this template."""
    loop_module = _load_loop()
    script_pos = {"pos": 0}

    def factory(ws: Path, system_prompt: str, round_index: int) -> Any:
        class _Runner(ScriptedRunner):
            async def send(self, prompt: str) -> list[Any]:
                self.calls.append(prompt)
                pos = script_pos["pos"]
                script_pos["pos"] += 1
                candidate_id = f"c-mean-{pos}"
                candidate_dir = ws / "models" / "candidates" / candidate_id
                candidate_dir.mkdir(parents=True, exist_ok=True)
                (candidate_dir / "model.py").write_text(
                    _MEAN_CANDIDATE.replace("__CID__", candidate_id), encoding="utf-8"
                )
                (candidate_dir / "proposal.json").write_text(
                    json.dumps(
                        {
                            "candidateId": candidate_id,
                            "targetAspect": "level-bias",
                            "aspectDetail": "constant mean",
                            "mechanism": "predict the channel mean",
                            "expectation": "diag MSE changes",
                        }
                    ),
                    encoding="utf-8",
                )
                return []

        return _Runner()

    run_dir = tmp_path / "run-deep"
    loop = loop_module.HeadroomLoop(
        run_dir=run_dir,
        task_name="lgssm_xchannel",
        arm="fixed",
        seed=0,
        cfg=_small_cfg(loop_module),
        sizes=SMALL_SIZES,
        runner_factory=factory,
        m0_template="m0_deep.py",
    )
    asyncio.run(loop.run())
    rounds = _rounds(run_dir)
    assert len(rounds) == 3
    assert all(row["status"] == "evaluated" for row in rounds)
    # The round-0 deep incumbent card exists, so evolution signatures and the
    # chain analysis always have a parent card to diff against.
    assert (loop.heldout / "posthoc" / "m0.json").is_file()
    assert any(
        p.name == "c-mean-0.json" for p in (loop.heldout / "posthoc").glob("*.json")
    )
    # The in-process fit used the deep template's own model id.
    assert loop._incumbent_model_id in ("m0-patchtst", "c-mean-0")
