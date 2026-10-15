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
        arm="card",
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
        arm="card",
        seed=0,
        cfg=_small_cfg(loop_module),
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
        arm="card",
        seed=0,
        cfg=_small_cfg(loop_module),
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
        arm="card",
        seed=0,
        cfg=_small_cfg(loop_module),
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
        arm="card",
        seed=0,
        cfg=_small_cfg(loop_module),
        sizes=SMALL_SIZES,
        runner_factory=factory,
    )
    with pytest.raises(RuntimeError, match="DSH runtime failed") as excinfo:
        asyncio.run(loop.run())
    # The root cause must name the empty-turn billing signature.
    assert "no parts" in str(excinfo.value.__cause__)
