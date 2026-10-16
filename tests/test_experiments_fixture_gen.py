"""The fixture-family generation pipeline: DSL schema, gates, frozen registry.

experiments/fixture_gen.py is the development-time trust mechanism behind
experiments/probes.py's fixture families: an LLM flow proposes a family as a
constrained DSL record and four deterministic gates measure it.  These tests
pin (1) the DSL schema (bad injections rejected before any gate runs),
(2) the frozen registry reproduces the pre-pipeline fixtures BITWISE — the
migration of the four hand-designed families is provenance, not new numbers,
(3) each gate does its own job (a nondeterministic witness, a
mean-shifting injection and a periodicity design that the band-24 probe
completes are all rejected, with numbers), and (4) a good proposal is
appended to the registry through the CLI's admit().

The zero-noise side of the trust rule (a lag-13 noise probe completing no
family, generated families included) lives in test_experiments_probes.py's
test_negative_controls_kill_noise_probes, which validates over the whole
registry.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _REPO_ROOT / "experiments" / "specs" / "probe-fixture-registry.json"
_PROPOSALS_DIR = _REPO_ROOT / "experiments" / "specs" / "proposals"


def _load_fixture_gen() -> Any:
    """Load experiments/fixture_gen.py by file path (it loads probes itself)."""
    source = _REPO_ROOT / "experiments" / "fixture_gen.py"
    spec = importlib.util.spec_from_file_location("rhg_fixture_gen", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rhg_fixture_gen"] = module
    spec.loader.exec_module(module)
    return module


G = _load_fixture_gen()
P = G.load_probes_module()
REGISTRY = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))


def family_entry(family_id: str) -> dict[str, Any]:
    entry = next((e for e in REGISTRY["families"] if e["familyId"] == family_id), None)
    assert entry is not None, f"no {family_id!r} family in the registry"
    return entry


# Digests of the fixtures the PRE-registry implementation produced (residual:
# seeds 0-2; activation: seed 0), captured before the rewiring and
# independently re-verified against git-HEAD probes.py.  The residual lists
# carry 10 entries in registry order: the first SIX are the historical seed
# families (the migration must reproduce them bitwise — provenance, not new
# numbers); the last four are the two pipeline-admitted families at their
# appended seed slots.


# ---------------------------------------------------------------------------
# the frozen registry satisfies the DSL
# ---------------------------------------------------------------------------


def test_frozen_registry_satisfies_the_dsl_schema() -> None:
    assert G.registry_problems(REGISTRY, P) == []
    ids = [entry["familyId"] for entry in REGISTRY["families"]]
    assert ids == ["period24", "cross-lag3", "nonlinear", "activation", "horizon-range", "level-bias"]
    origins = {entry["familyId"]: entry["origin"] for entry in REGISTRY["families"]}
    assert origins["horizon-range"] == "generated"
    assert origins["level-bias"] == "generated"
    assert all(origins[f] == "seed" for f in ("period24", "cross-lag3", "nonlinear", "activation"))


def test_seed_families_keep_their_historical_seed_slots() -> None:
    """The migration must not move a child-seed slot: the legacy layout was
    family-major pairs at 2*f+p and controls at 6 + position*2 + c."""
    expected_pairs = {"period24": [0, 1], "cross-lag3": [2, 3], "nonlinear": [4, 5]}
    for family_id, pair_seeds in expected_pairs.items():
        layout = family_entry(family_id)["seedLayout"]
        assert layout["pairSeeds"] == pair_seeds
        assert layout["controlSeeds"] == [
            [6 + 2 * position, 7 + 2 * position] for position in pair_seeds
        ]
    assert family_entry("activation")["seedLayout"] == {
        "netSeed": 0,
        "directionSeed": 1,
        "pairDataSeeds": [2, 3],
        "controlDataSeeds": [[4, 5], [6, 7]],
    }


def test_checked_in_proposals_match_the_generated_registry_entries() -> None:
    """The registry is the audit trail of the CLI admissions: the two checked-in
    proposal files must be exactly what was appended (DSL, witness, rationale)."""
    for family_id in ("horizon-range", "level-bias"):
        proposal = json.loads((_PROPOSALS_DIR / f"{family_id}.json").read_text(encoding="utf-8"))
        entry = family_entry(family_id)
        assert entry["generator"] == proposal["generator"]
        assert entry["witnessProbe"] == proposal["witnessProbe"]
        assert entry["rationale"] == proposal["rationale"]
        evidence = entry["gateEvidence"]
        assert evidence["witness"]["deterministic"] is True
        assert all(seed_data["complete"] for seed_data in evidence["witness"]["perSeed"].values())
        # Gate 2 with numbers: no registered witness completed the family.
        assert all(
            probe["completedAtSeeds"] == []
            for probe in evidence["specificity"]["probes"].values()
        )
        assert evidence["amplitude"]["maxStdChange"] < G.STD_CHANGE_BOUND
        assert evidence["amplitude"]["maxMeanShiftStd"] < G.MEAN_SHIFT_STD_BOUND
        assert evidence["reproducibility"]["bitwiseIdentical"] is True


# ---------------------------------------------------------------------------
# bitwise reproduction
# ---------------------------------------------------------------------------


def test_generated_families_are_bitwise_reproducible() -> None:
    for family_id in ("horizon-range", "level-bias"):
        result = G.gate_reproducibility(P, family_entry(family_id), seeds=(0, 1))
        assert result["passed"], result["report"]


# ---------------------------------------------------------------------------
# the DSL schema rejects bad injections before any gate runs
# ---------------------------------------------------------------------------


def horizon_proposal(**overrides: Any) -> dict[str, Any]:
    proposal = json.loads((_PROPOSALS_DIR / "horizon-range.json").read_text(encoding="utf-8"))
    for key, value in overrides.items():
        proposal[key] = value
    return proposal


def _first_problem(proposal: dict[str, Any]) -> str:
    return "; ".join(G.proposal_problems(proposal, REGISTRY, P))


def test_zero_amplitude_is_rejected() -> None:
    proposal = horizon_proposal()
    proposal["generator"]["injection"]["amplitudeFraction"] = 0.0
    assert "amplitudeFraction must be a number in (0, 3]" in _first_problem(proposal)


def test_stationarity_breaking_clean_process_is_rejected() -> None:
    proposal = horizon_proposal()
    proposal["generator"]["clean"]["channels"][0]["phi"] = 1.0
    assert "phi must be in (-1, 1) for stationarity" in _first_problem(proposal)


def test_unknown_injection_type_is_rejected() -> None:
    proposal = horizon_proposal()
    proposal["generator"]["injection"] = {
        "type": "arbitrary_wave", "channel": 0, "amplitudeFraction": 0.5
    }
    assert "injection type must be one of" in _first_problem(proposal)


def test_seed_handling_is_not_the_proposals_to_control() -> None:
    # The pipeline assigns child-seed slots; a proposal carrying seed keys
    # (or a seedLayout of its own) falls outside the closed DSL.
    proposal = horizon_proposal()
    proposal["generator"]["seed"] = 12345
    assert "unknown keys ['seed']" in _first_problem(proposal)
    proposal = horizon_proposal()
    proposal["seedLayout"] = {"pairSeeds": [0, 1], "controlSeeds": [[2, 3], [4, 5]]}
    assert "unknown keys ['seedLayout']" in _first_problem(proposal)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda p: p.update(familyId="Bad Id!"), "familyId must match"),
        (lambda p: p.update(familyId="period24"), "already in the registry"),
        (lambda p: p["witnessProbe"]["proposal"].pop("signalKey"), "missing key 'signalKey'"),
        (
            lambda p: p["witnessProbe"]["proposal"].update(expectedDirection="sideways"),
            "expectedDirection must be 'higher' or 'lower'",
        ),
    ],
)
def test_more_schema_rejections(mutation: Any, expected: str) -> None:
    proposal = horizon_proposal()
    mutation(proposal)
    assert expected in _first_problem(proposal)


# ---------------------------------------------------------------------------
# the gates, one job each
# ---------------------------------------------------------------------------


def test_amplitude_gate_rejects_a_mean_shifting_injection() -> None:
    """cycles=0.05 is a near-constant drift: the whole offset lands in the
    marginal mean and gate 3 must reject it with numbers (this is the
    level-bias design's failed first iteration)."""
    entry = copy.deepcopy(family_entry("level-bias"))
    entry["familyId"] = "level-bias-mean-shift"
    entry["generator"]["injection"]["cycles"] = 0.05
    result = G.gate_amplitude(P, entry, seeds=(0,))
    assert not result["passed"]
    assert "max mean shift" in result["report"]
    assert result["evidence"]["maxMeanShiftStd"] >= G.MEAN_SHIFT_STD_BOUND


def test_witness_gate_rejects_a_nondeterministic_witness() -> None:
    """The witness completes its family but sprinkles unseeded randomness into
    the signalKey: gate 1's determinism rerun must catch it."""
    entry = copy.deepcopy(family_entry("horizon-range"))
    entry["familyId"] = "horizon-range-nondeterministic"
    jittered = entry["witnessProbe"]["source"].replace(
        "return {\"horizonRatio\": num / den}",
        "return {\"horizonRatio\": (num / den) * (1.0 + 1e-9 * np.random.random())}",
    )
    assert jittered != entry["witnessProbe"]["source"]  # the mutation took
    entry["witnessProbe"]["source"] = jittered
    result = G.gate_witness(P, entry, seeds=(0,))
    assert not result["passed"]
    assert "deterministic" in result["report"]
    # the jitter is tiny: the family itself still completes — the failure is
    # the determinism rerun, not the pair rule
    assert result["evidence"]["perSeed"][0]["complete"] is True


def test_specificity_gate_rejects_a_design_the_band_probe_sees() -> None:
    """A period-24 sinusoid injection is NOT a new family: the registered
    band-24 witness completes it at every seed, so gate 2 rejects the proposal
    (the horizon-range design's failed first iteration)."""
    entry = copy.deepcopy(family_entry("horizon-range"))
    entry["familyId"] = "period24-clone"
    entry["generator"]["injection"] = {
        "type": "sinusoid", "channel": 0, "period": 24, "amplitudeFraction": 0.75,
    }
    entry["seedLayout"] = G.assign_seed_slots(REGISTRY, "residual")
    seed_registry = {
        "version": 1,
        "families": [e for e in REGISTRY["families"] if e["origin"] == "seed" and e["kind"] == "residual"],
    }
    result = G.gate_specificity(P, entry, seed_registry, seeds=(0,))
    assert not result["passed"]
    assert result["evidence"]["probes"]["band24-energy"]["completedAtSeeds"] == [0]


def test_admit_rejects_a_bad_proposal_without_touching_the_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(REGISTRY, indent=2), encoding="utf-8")
    proposal = horizon_proposal()
    proposal["generator"]["injection"]["amplitudeFraction"] = 0.0
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    assert G.admit(proposal_path, registry_path, seeds=(0,)) == 1
    assert json.loads(registry_path.read_text(encoding="utf-8")) == REGISTRY  # untouched


def test_admit_rejects_a_duplicate_design_then_appends_a_new_one(tmp_path: Path) -> None:
    """The full CLI pipeline on the checked-in horizon-range proposal, against
    a registry copy.  Re-proposing the horizon design under a fresh id is
    REJECTED at gate 2 — the registered horizon-ratio witness completes the
    clone, because a family IS its witness signature and the registry must
    not hold two of the same.  A registry that has no residual families yet
    admits it: all four gates, then the append with the gate evidence."""
    proposal = horizon_proposal()
    proposal["familyId"] = "horizon-range-admission-test"
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")

    # (a) duplicate design vs the full registry: gate 2 rejects, no append
    full = tmp_path / "registry-full.json"
    full.write_text(json.dumps(REGISTRY, indent=2), encoding="utf-8")
    assert G.admit(proposal_path, full, seeds=(0,)) == 1
    assert json.loads(full.read_text(encoding="utf-8")) == REGISTRY  # untouched

    # (b) the same proposal against a registry with no residual families:
    # gates 1, 3, 4 all pass (specificity is vacuous) and the family is
    # appended with its assigned slots and evidence
    fresh = tmp_path / "registry-fresh.json"
    fresh.write_text(
        json.dumps({"version": 1, "families": [family_entry("activation")]}, indent=2),
        encoding="utf-8",
    )
    assert G.admit(proposal_path, fresh, seeds=(0,)) == 0
    appended = json.loads(fresh.read_text(encoding="utf-8"))
    assert len(appended["families"]) == 2
    entry = appended["families"][-1]
    assert entry["familyId"] == "horizon-range-admission-test"
    assert entry["origin"] == "generated"
    assert entry["seedLayout"] == {"pairSeeds": [0, 1], "controlSeeds": [[2, 3], [4, 5]]}
    assert entry["gateEvidence"]["witness"]["perSeed"]["0"]["complete"] is True
    assert entry["gateEvidence"]["witness"]["deterministic"] is True
    assert entry["gateEvidence"]["reproducibility"]["bitwiseIdentical"] is True
    # and the REAL registry was not touched by the test admission
    assert len(REGISTRY["families"]) == 6
