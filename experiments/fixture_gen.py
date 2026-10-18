"""Fixture-family generation pipeline: an LLM proposes, deterministic code gates.

This module is the development-time authoring pipeline for the
probe-validation fixture families: an offline LLM flow PROPOSES a family as
a constrained JSON DSL record — data, never free code — and this module
MEASURES the proposal against four deterministic admission gates, each
reported with numbers.  Nothing here runs during P2: the loop only ever
sees the frozen registry (experiments/specs/probe-fixture-registry.json)
this CLI appends to.

DSL (a proposal file)
---------------------
    {"familyId": str,                # [a-z0-9][a-z0-9-]{0,31}, registry-unique
     "kind": "residual" | "activation",
     "generator": {
       "type": "series_pair",
       "clean": {"length": int, "channels": [<channel>, ...]},
       "injection": <injection>} |
      {"type": "activation_mlp_rank1", "params": {...}},
     "witnessProbe": {"source": <probe.py source>,
                      "proposal": {"probeId", "measures", "signalKey",
                                   "expectedDirection", "rationale"}},
     "rationale": str}

    channel := {"type": "ar1", "phi": -1<phi<1, "innovationSd": >0}
             | {"type": "ar1_plus_white", ..., "whiteSd": >0}
             | {"type": "white", "sd": >0}
             | {"type": "sinusoid_plus_white", "period": >=2, "amplitude": >0,
                "whiteSd": >0}

    injection (residual) :=
      {"type": "sinusoid" | "horizon_scaled", "channel": int, "period": >=2,
       "amplitudeFraction": (0, 3]}    # of the clean channel's sample sd
     | {"type": "cross_lag", "channel", "sourceChannel" != channel,
        "lag": >=1, "amplitudeFraction"}
     | {"type": "threshold_bilinear", "targetChannel",
        "driverChannel" != targetChannel, "amplitudeFraction"}
     | {"type": "constant_offset", "channel", "cycles": [0, 1),
        "amplitudeFraction"}

Every key set is exact (unknown keys are rejected, seed keys doubly so: a
proposal never touches randomness — the pipeline assigns the child-seed
slots).  All builders draw seed-derived randomness only, in a frozen
spec-independent order (see probes._clean_series).

Two type names deserve honesty notes.  ``horizon_scaled`` is realized as a
narrowband component at the design period: a stationary series injection
cannot literally scale with h (h indexes read offsets of one fixed series),
so the horizon design is realized spectrally — a component at period
~4*(H-1) is invisible in 1-step residual increments and maximal at the full
H-step gap.  ``constant_offset`` is the level-bias design: a level drift
slower than one full cycle over the series (cycles < 1); a TRUE constant is
absorbed by the mean-predictor incumbent and dies at gate 1, so the
admissible family is the low-frequency limit the witness must argue for.
The gates measure the design's claim, not its label.

The four admission gates
------------------------
1. WITNESS — the witness probe passes probes.check_probe_source, is
   deterministic (a rerun of the seed-0 pair-0 planted fixture reproduces
   its output bitwise) and completes its own family under the CURRENT
   validation rule — declared direction + >5% relative margin vs the clean
   member and both controls + gap>spread (probes._pair_passes does the
   arithmetic) — at EVERY seed 0-4.
2. SPECIFICITY — the registered witness probes of the same kind's OTHER
   families (for a residual proposal today: the band-24, cross-lag-3 and
   bilinear probes) must complete the proposal's family at NO seed 0-4: a
   family a zero-noise/unrelated probe can complete is a false positive.
3. AMPLITUDE SANITY — the injection's perturbation of the clean process's
   marginal statistics stays bounded: per channel, |std(planted) -
   std(clean)| / std(clean) < 0.20 and |mean(planted) - mean(clean)| <
   0.25 * std(clean), verified numerically at seeds 0-4 on both pairs
   (activation kind: the residuals must be bitwise unchanged and the
   layer-2 injection beta within (0, 3]).
4. REPRODUCIBILITY — two builds of the family's pairs and controls from
   the same DSL record are bitwise identical at every seed 0-4.

CLI
---
    uv run --extra research python experiments/fixture_gen.py \
        --proposal <file.json> [--registry <path>]

On a full pass the family is appended to the registry (origin "generated",
with its DSL record, witness, rationale, the seedLayout the pipeline
assigned and the gate evidence with the measured numbers) and the exit code
is 0.  On any failure the failing gate prints its verdict with numbers and
the exit code is 1 — the registry is never touched.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = _REPO_ROOT / "experiments" / "specs" / "probe-fixture-registry.json"

# The admission seeds: a family must clear every gate at every one of these.
ADMISSION_SEEDS = (0, 1, 2, 3, 4)
FAMILY_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,31}\Z")
# Gate 3 bounds (activation kind: residuals unchanged + beta bound instead).
STD_CHANGE_BOUND = 0.20
MEAN_SHIFT_STD_BOUND = 0.25
ACT_BETA_BOUND = 3.0

_PROPOSAL_KEYS = {"familyId", "kind", "generator", "witnessProbe", "rationale"}
_CHANNEL_KEYS = {
    "ar1": {"phi", "innovationSd"},
    "ar1_plus_white": {"phi", "innovationSd", "whiteSd"},
    "white": {"sd"},
    "sinusoid_plus_white": {"period", "amplitude", "whiteSd"},
}
_INJECTION_KEYS = {
    "sinusoid": {"channel", "period", "amplitudeFraction"},
    "horizon_scaled": {"channel", "period", "amplitudeFraction"},
    "cross_lag": {"channel", "sourceChannel", "lag", "amplitudeFraction"},
    "threshold_bilinear": {"targetChannel", "driverChannel", "amplitudeFraction"},
    "constant_offset": {"channel", "cycles", "amplitudeFraction"},
}
_ACTIVATION_PARAM_KEYS = {
    "length", "nResiduals", "windowStride", "layer1Width", "layer2Width",
    "sinusoidAmplitude", "noiseSd", "period", "injectBeta",
}


def load_probes_module() -> Any:
    """experiments/probes.py by file path, like the loop loads its modules."""
    source = _REPO_ROOT / "experiments" / "probes.py"
    spec = importlib.util.spec_from_file_location("rhg_probes_gen", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rhg_probes_gen"] = module
    spec.loader.exec_module(module)
    return module


# --- proposal schema (the constrained DSL) -----------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_exact_keys(record: dict[str, Any], expected: set[str], where: str) -> list[str]:
    missing = sorted(expected - set(record))
    unknown = sorted(set(record) - expected)
    problems = [f"{where} is missing keys {missing}"] if missing else []
    if unknown:
        problems.append(f"{where} carries unknown keys {unknown} (the DSL is closed)")
    return problems


def _check_channel(channel: Any, where: str) -> list[str]:
    if not isinstance(channel, dict):
        return [f"{where} must be an object, got {type(channel).__name__}"]
    kind = channel.get("type")
    if kind not in _CHANNEL_KEYS:
        return [f"{where} type must be one of {sorted(_CHANNEL_KEYS)}, got {kind!r}"]
    problems = _check_exact_keys(channel, _CHANNEL_KEYS[kind] | {"type"}, where)
    if _CHANNEL_KEYS[kind] - set(channel):
        return problems  # missing keys: the range checks below would KeyError
    if kind in ("ar1", "ar1_plus_white"):
        if not (-1.0 < channel["phi"] < 1.0):
            problems.append(f"{where} phi must be in (-1, 1) for stationarity, got {channel['phi']!r}")
        if not (channel["innovationSd"] > 0):
            problems.append(f"{where} innovationSd must be > 0, got {channel['innovationSd']!r}")
    if kind in ("ar1_plus_white", "sinusoid_plus_white") and channel["whiteSd"] <= 0:
        problems.append(f"{where} whiteSd must be > 0, got {channel['whiteSd']!r}")
    if kind == "white" and channel["sd"] <= 0:
        problems.append(f"{where} sd must be > 0, got {channel['sd']!r}")
    if kind == "sinusoid_plus_white":
        if channel["period"] < 2:
            problems.append(f"{where} period must be >= 2, got {channel['period']!r}")
        if not (channel["amplitude"] > 0):
            problems.append(f"{where} amplitude must be > 0, got {channel['amplitude']!r}")
    return problems


def _check_injection(injection: Any, n_channels: int) -> list[str]:
    if not isinstance(injection, dict):
        return [f"injection must be an object, got {type(injection).__name__}"]
    kind = injection.get("type")
    if kind not in _INJECTION_KEYS:
        return [
            f"injection type must be one of {sorted(_INJECTION_KEYS)}, got {kind!r}"
        ]
    where = f"injection ({kind})"
    problems = _check_exact_keys(injection, _INJECTION_KEYS[kind] | {"type"}, where)
    if _INJECTION_KEYS[kind] - set(injection):
        return problems  # missing keys: the range checks below would misfire

    def in_range(name: str) -> bool:
        return isinstance(injection[name], int) and 0 <= injection[name] < n_channels

    fraction = injection["amplitudeFraction"]
    if not (_is_number(fraction) and 0.0 < fraction <= 3.0):
        problems.append(
            f"{where} amplitudeFraction must be a number in (0, 3], got {fraction!r} "
            "(zero amplitude plants nothing)"
        )
    if kind in ("sinusoid", "horizon_scaled", "cross_lag", "constant_offset"):
        if not in_range("channel"):
            problems.append(
                f"{where} channel must be an int index of the clean channels, "
                f"got {injection['channel']!r}"
            )
    if kind in ("sinusoid", "horizon_scaled"):
        if not (_is_number(injection["period"]) and injection["period"] >= 2):
            problems.append(f"{where} period must be a number >= 2, got {injection['period']!r}")
    if kind == "cross_lag":
        if not in_range("sourceChannel"):
            problems.append(
                f"{where} sourceChannel must be an int index of the clean channels, "
                f"got {injection['sourceChannel']!r}"
            )
        if injection["channel"] == injection["sourceChannel"]:
            problems.append(f"{where} sourceChannel must differ from channel")
        if not (isinstance(injection["lag"], int) and injection["lag"] >= 1):
            problems.append(f"{where} lag must be an int >= 1, got {injection['lag']!r}")
    if kind == "threshold_bilinear":
        for name in ("targetChannel", "driverChannel"):
            if not in_range(name):
                problems.append(
                    f"{where} {name} must be an int index of the clean channels, "
                    f"got {injection[name]!r}"
                )
        if injection["targetChannel"] == injection["driverChannel"]:
            problems.append(f"{where} driverChannel must differ from targetChannel")
    if kind == "constant_offset":
        cycles = injection["cycles"]
        if not (_is_number(cycles) and 0.0 <= cycles < 1.0):
            problems.append(
                f"{where} cycles must be a number in [0, 1) (slower than one full "
                f"cycle over the series), got {cycles!r}"
            )
    return problems


def _check_generator(generator: Any, kind: str) -> list[str]:
    if not isinstance(generator, dict):
        return [f"generator must be an object, got {type(generator).__name__}"]
    gtype = generator.get("type")
    if kind == "residual" and gtype != "series_pair":
        return [f"a residual proposal's generator type must be 'series_pair', got {gtype!r}"]
    if kind == "activation" and gtype != "activation_mlp_rank1":
        return [
            f"an activation proposal's generator type must be 'activation_mlp_rank1', got {gtype!r}"
        ]
    if gtype == "series_pair":
        problems = _check_exact_keys(generator, {"type", "clean", "injection"}, "generator")
        clean = generator.get("clean")
        if not isinstance(clean, dict):
            return problems + [f"generator.clean must be an object, got {type(clean).__name__}"]
        problems += _check_exact_keys(clean, {"length", "channels"}, "generator.clean")
        length = clean.get("length")
        if not (isinstance(length, int) and length >= 256):
            problems.append(f"generator.clean length must be an int >= 256, got {length!r}")
        channels = clean.get("channels")
        if not (isinstance(channels, list) and 1 <= len(channels) <= 8):
            problems.append(f"generator.clean channels must be a list of 1-8 specs, got {channels!r}")
            return problems
        for index, channel in enumerate(channels):
            problems += _check_channel(channel, f"generator.clean channel {index}")
        problems += _check_injection(generator.get("injection"), len(channels))
        return problems
    # activation_mlp_rank1: fixed-form constructor, exact parameter set
    problems = _check_exact_keys(generator, {"type", "params"}, "generator")
    params = generator.get("params")
    if not isinstance(params, dict):
        return problems + [f"generator.params must be an object, got {type(params).__name__}"]
    problems += _check_exact_keys(params, _ACTIVATION_PARAM_KEYS, "generator.params")
    for name in ("length", "nResiduals", "windowStride", "layer1Width", "layer2Width", "period"):
        value = params.get(name)
        if not (isinstance(value, int) and value > 0):
            problems.append(f"generator.params {name} must be a positive int, got {value!r}")
    for name in ("sinusoidAmplitude", "noiseSd"):
        value = params.get(name)
        if not (_is_number(value) and value > 0):
            problems.append(f"generator.params {name} must be a number > 0, got {value!r}")
    beta = params.get("injectBeta")
    if not (_is_number(beta) and 0.0 < beta <= ACT_BETA_BOUND):
        problems.append(f"generator.params injectBeta must be in (0, {ACT_BETA_BOUND}], got {beta!r}")
    return problems


def proposal_problems(proposal: Any, registry: dict[str, Any], probes: Any) -> list[str]:
    """Every way a proposal file can fall outside the DSL, loudly."""
    if not isinstance(proposal, dict):
        return [f"the proposal must be a JSON object, got {type(proposal).__name__}"]
    problems = _check_exact_keys(proposal, _PROPOSAL_KEYS, "proposal")
    if problems and "unknown keys" in problems[0]:
        return problems  # key errors first; the rest may not even type-check
    family_id = proposal.get("familyId")
    if not isinstance(family_id, str) or not FAMILY_ID_PATTERN.match(family_id):
        problems.append(f"familyId must match [a-z0-9][a-z0-9-]{{0,31}}, got {family_id!r}")
    kind = proposal.get("kind")
    if kind not in ("residual", "activation"):
        problems.append(f"kind must be 'residual' or 'activation', got {kind!r}")
    else:
        problems += _check_generator(proposal.get("generator"), kind)
    witness = proposal.get("witnessProbe")
    if not isinstance(witness, dict):
        problems.append(f"witnessProbe must be an object, got {type(witness).__name__}")
    else:
        problems += _check_exact_keys(witness, {"source", "proposal"}, "witnessProbe")
        if not isinstance(witness.get("source"), str) or not witness["source"].strip():
            problems.append("witnessProbe.source must be a non-empty string (probe.py source)")
        inner = witness.get("proposal")
        if isinstance(inner, dict):
            problems += [f"witnessProbe.proposal {p}" for p in probes._proposal_problems(inner)]
        else:
            problems.append("witnessProbe.proposal must be a JSON object")
    if not isinstance(proposal.get("rationale"), str) or not proposal["rationale"].strip():
        problems.append("rationale must be a non-empty string")
    if isinstance(family_id, str):
        known = [entry["familyId"] for entry in registry["families"]]
        if family_id in known:
            problems.append(f"familyId {family_id!r} is already in the registry {known}")
    return problems


def registry_problems(registry: dict[str, Any], probes: Any) -> list[str]:
    """Every way a registry entry can fall outside the DSL, loudly.

    The frozen registry is the pipeline's own output, so it must satisfy the
    same schema a fresh proposal does (plus the provenance keys the CLI
    adds).  Run by tests to pin the checked-in file.
    """
    if not isinstance(registry, dict) or not isinstance(registry.get("families"), list):
        return ["registry must be an object with a 'families' list"]
    problems: list[str] = []
    seen_ids: set[str] = set()
    expected = _PROPOSAL_KEYS | {"origin", "seedLayout", "gateEvidence"}
    for entry in registry["families"]:
        family_id = entry.get("familyId", "<unnamed>")
        where = f"family {family_id!r}"
        if not isinstance(entry, dict):
            problems.append(f"{where} must be an object, got {type(entry).__name__}")
            continue
        problems += [
            f"{where} {p}" for p in _check_exact_keys(entry, expected, where)
        ]
        if family_id in seen_ids:
            problems.append(f"{where} appears twice in the registry")
        seen_ids.add(family_id)
        if not isinstance(family_id, str) or not FAMILY_ID_PATTERN.match(family_id):
            problems.append(f"{where} familyId must match [a-z0-9][a-z0-9-]{{0,31}}")
        if entry.get("kind") not in ("residual", "activation"):
            problems.append(f"{where} kind must be 'residual' or 'activation', got {entry.get('kind')!r}")
        else:
            problems += [f"{where} {p}" for p in _check_generator(entry.get("generator"), entry["kind"])]
        if entry.get("origin") not in ("seed", "generated"):
            problems.append(f"{where} origin must be 'seed' or 'generated', got {entry.get('origin')!r}")
        witness = entry.get("witnessProbe")
        if isinstance(witness, dict) and isinstance(witness.get("proposal"), dict):
            problems += [
                f"{where} witnessProbe.proposal {p}"
                for p in probes._proposal_problems(witness["proposal"])
            ]
        else:
            problems.append(f"{where} witnessProbe must carry a proposal object")
        if not isinstance(entry.get("rationale"), str) or not entry["rationale"].strip():
            problems.append(f"{where} rationale must be a non-empty string")
        layout = entry.get("seedLayout")
        if not isinstance(layout, dict):
            problems.append(f"{where} seedLayout must be an object")
            continue
        slots_ok = True

        def _slots(values: Any, count: int, label: str) -> None:
            nonlocal slots_ok
            if not (
                isinstance(values, list)
                and len(values) == count
                and all(isinstance(v, int) and v >= 0 for v in values)
            ):
                problems.append(f"{where} seedLayout {label} must be {count} non-negative ints")
                slots_ok = False

        if entry.get("kind") == "residual":
            _slots(layout.get("pairSeeds"), 2, "pairSeeds")
            if isinstance(layout.get("controlSeeds"), list):
                for pair in layout["controlSeeds"]:
                    _slots(pair, 2, "controlSeeds entry")
            else:
                problems.append(f"{where} seedLayout controlSeeds must be a list of pairs")
                slots_ok = False
        else:
            for key in ("netSeed", "directionSeed"):
                if not isinstance(layout.get(key), int) or layout[key] < 0:
                    problems.append(f"{where} seedLayout {key} must be a non-negative int")
                    slots_ok = False
            _slots(layout.get("pairDataSeeds"), 2, "pairDataSeeds")
            if isinstance(layout.get("controlDataSeeds"), list):
                for pair in layout["controlDataSeeds"]:
                    _slots(pair, 2, "controlDataSeeds entry")
            else:
                problems.append(f"{where} seedLayout controlDataSeeds must be a list of pairs")
                slots_ok = False
        if slots_ok:
            flat = [
                seed
                for seed in (
                    layout.get("pairSeeds", [])
                    + [s for pair in layout.get("controlSeeds", []) for s in pair]
                )
                if isinstance(seed, int)
            ]
            if len(set(flat)) != len(flat):
                problems.append(f"{where} seedLayout reuses a child-seed slot within the family")
    return problems


# --- seed-slot assignment -----------------------------------------------------


def assign_seed_slots(registry: dict[str, Any], kind: str) -> dict[str, Any]:
    """The next free child-seed slots for a new family of one kind.

    Slots are positions in the kind's SeedSequence spawn pool (see
    probes._seed_pool); the pipeline assigns them, never the proposal.  A
    family takes 2 pair slots then 2x_N_CONTROLS control slots (activation:
    net, direction, 2 pair data, 2x_N_CONTROLS control data).
    """
    used: list[int] = []
    for entry in registry["families"]:
        if entry["kind"] != kind:
            continue
        layout = entry["seedLayout"]
        if kind == "residual":
            used.extend(layout["pairSeeds"])
            used.extend(seed for pair in layout["controlSeeds"] for seed in pair)
        else:
            used.extend([layout["netSeed"], layout["directionSeed"], *layout["pairDataSeeds"]])
            used.extend(seed for pair in layout["controlDataSeeds"] for seed in pair)
    base = (max(used) + 1) if used else 0
    if kind == "residual":
        return {
            "pairSeeds": [base, base + 1],
            "controlSeeds": [[base + 2, base + 3], [base + 4, base + 5]],
        }
    return {
        "netSeed": base,
        "directionSeed": base + 1,
        "pairDataSeeds": [base + 2, base + 3],
        "controlDataSeeds": [[base + 4, base + 5], [base + 6, base + 7]],
    }


# --- gate plumbing ------------------------------------------------------------


def _probe_values(
    probes: Any,
    source_path: Path,
    artifacts_list: list[dict[str, Any]],
    probe_id: str,
    signal_key: str,
) -> tuple[list[float] | None, list[dict[str, Any]] | None, str | None]:
    """Run one probe over fixtures; (values, outputs, error)."""
    values: list[float] = []
    outputs: list[dict[str, Any]] = []
    for artifacts in artifacts_list:
        try:
            wrapper = probes._run_probe_subprocess(
                source_path, artifacts, probes.DEFAULT_TIMEOUT_SECONDS
            )
            values.append(probes._signal_value(wrapper, probe_id, signal_key))
            outputs.append(wrapper["output"])
        except probes.ProbeError as exc:
            return None, None, f"{type(exc).__name__}: {exc}"
    return values, outputs, None


def _artifacts_equal(a: dict[str, Any], b: dict[str, Any], probes: Any) -> bool:
    """Bitwise fixture equality (e is NaN-padded by design)."""
    for key in ("e", "r"):
        if not np.array_equal(
            a["residuals"][key], b["residuals"][key], equal_nan=(key == "e")
        ):
            return False
    if not np.array_equal(a["residuals"]["origins"], b["residuals"]["origins"]):
        return False
    if set(a["activations"]) != set(b["activations"]):
        return False
    for name, values in a["activations"].items():
        if not np.array_equal(values, b["activations"][name]):
            return False
    return probes._canonical(a["meta"]) == probes._canonical(b["meta"])


def _with_witness_source(probes: Any, witness: dict[str, Any], run):
    """Write the witness source to a temp probe.py and run(probes, path)."""
    with tempfile.TemporaryDirectory(prefix="fixture-gen-") as tmp:
        source_path = Path(tmp) / "witness_probe.py"
        source_path.write_text(witness["source"], encoding="utf-8")
        return run(source_path)


# --- the four gates -------------------------------------------------------------


def gate_witness(
    probes: Any, entry: dict[str, Any], seeds: tuple[int, ...] = ADMISSION_SEEDS
) -> dict[str, Any]:
    """Gate 1: the witness completes its own family under the current rule."""
    witness = entry["witnessProbe"]
    proposal = witness["proposal"]
    probe_id = proposal["probeId"]
    signal_key = proposal["signalKey"]
    direction = proposal["expectedDirection"]
    result: dict[str, Any] = {"gate": "witness", "passed": False, "evidence": {}, "report": ""}
    try:
        probes.check_probe_source(witness["source"])
    except probes.cand.SourceRejected as exc:
        result["report"] = f"gate 1 WITNESS FAIL: witness source rejected: {exc}"
        return result

    def measure(source_path: Path) -> dict[str, Any]:
        evidence: dict[str, Any] = {"perSeed": {}, "deterministic": False}
        for seed in seeds:
            pairs = probes.build_family_pairs(entry, seed)
            controls = probes.build_family_controls(entry, seed)
            records, complete = [], True
            for (planted, clean), pair_controls in zip(pairs, controls):
                values, outputs, error = _probe_values(
                    probes, source_path, [planted, clean, *pair_controls], probe_id, signal_key
                )
                if error is not None:
                    return {
                        "passed": False,
                        "evidence": evidence,
                        "report": f"gate 1 WITNESS FAIL: probe error at seed {seed}: {error}",
                    }
                planted_value, clean_value, control_values = values[0], values[1], values[2:]
                passes, margin = probes._pair_passes(
                    direction, planted_value, [clean_value, *control_values]
                )
                records.append(
                    {
                        "planted": planted_value,
                        "clean": clean_value,
                        "controls": control_values,
                        "margin": margin,
                        "passes": passes,
                    }
                )
                complete = complete and passes
            evidence["perSeed"][seed] = {"pairs": records, "complete": complete}
        # Determinism: the seed-0 pair-0 planted fixture run twice must give
        # identical output (the same contract validate_probe enforces).
        pair0 = probes.build_family_pairs(entry, seeds[0])[0][0]
        _, first_outputs, error = _probe_values(
            probes, source_path, [pair0], probe_id, signal_key
        )
        if error is not None:
            return {
                "passed": False,
                "evidence": evidence,
                "report": f"gate 1 WITNESS FAIL: determinism rerun errored: {error}",
            }
        _, repeat_outputs, error = _probe_values(
            probes, source_path, [pair0], probe_id, signal_key
        )
        if error is not None:
            return {
                "passed": False,
                "evidence": evidence,
                "report": f"gate 1 WITNESS FAIL: determinism rerun errored: {error}",
            }
        evidence["deterministic"] = probes._canonical(first_outputs[0]) == probes._canonical(
            repeat_outputs[0]
        )
        failed = [seed for seed in seeds if not evidence["perSeed"][seed]["complete"]]
        if failed:
            report = [
                f"gate 1 WITNESS FAIL: the witness did not complete its own family at seeds {failed}"
            ]
            for seed in failed:
                for index, record in enumerate(evidence["perSeed"][seed]["pairs"]):
                    report.append(
                        f"  seed {seed} pair {index}: planted {record['planted']:.6g} vs clean "
                        f"{record['clean']:.6g} (controls "
                        f"{', '.join(f'{value:.6g}' for value in record['controls'])}), "
                        f"margin {record['margin']:.3g}, "
                        f"{'pass' if record['passes'] else 'fail'}"
                    )
            return {"passed": False, "evidence": evidence, "report": "\n".join(report)}
        if not evidence["deterministic"]:
            return {
                "passed": False,
                "evidence": evidence,
                "report": "gate 1 WITNESS FAIL: the witness is not deterministic "
                "(rerunning the seed-0 pair-0 planted fixture changed its output)",
            }
        report = ["gate 1 WITNESS PASS: family complete at every seed, output deterministic"]
        for seed in seeds:
            for index, record in enumerate(evidence["perSeed"][seed]["pairs"]):
                report.append(
                    f"  seed {seed} pair {index}: planted {record['planted']:.6g} vs clean "
                    f"{record['clean']:.6g} (controls "
                    f"{', '.join(f'{value:.6g}' for value in record['controls'])}), "
                    f"margin {record['margin']:.3g}, pass"
                )
        return {"passed": True, "evidence": evidence, "report": "\n".join(report)}

    result.update(_with_witness_source(probes, witness, measure))
    return result


def gate_specificity(
    probes: Any,
    entry: dict[str, Any],
    registry: dict[str, Any],
    seeds: tuple[int, ...] = ADMISSION_SEEDS,
) -> dict[str, Any]:
    """Gate 2: no registered same-kind witness completes the new family."""
    witnesses = [
        other["witnessProbe"]
        for other in registry["families"]
        if other["kind"] == entry["kind"] and other["familyId"] != entry["familyId"]
    ]
    evidence: dict[str, Any] = {"probes": {}, "seeds": list(seeds)}
    if not witnesses:
        evidence["probes"] = {}
        return {
            "gate": "specificity",
            "passed": True,
            "evidence": evidence,
            "report": "gate 2 SPECIFICITY PASS (vacuously): no other same-kind family exists",
        }
    completed_any = False
    report = []
    for witness in witnesses:
        proposal = witness["proposal"]
        probe_id = proposal["probeId"]
        signal_key = proposal["signalKey"]
        direction = proposal["expectedDirection"]
        completed_at: list[int] = []

        def measure(source_path: Path) -> None:
            for seed in seeds:
                pairs = probes.build_family_pairs(entry, seed)
                controls = probes.build_family_controls(entry, seed)
                pairs_passed = 0
                for (planted, clean), pair_controls in zip(pairs, controls):
                    values, _, error = _probe_values(
                        probes,
                        source_path,
                        [planted, clean, *pair_controls],
                        probe_id,
                        signal_key,
                    )
                    if error is not None:
                        continue  # a probe that cannot run cannot complete the family
                    passes, _margin = probes._pair_passes(direction, values[0], values[1:])
                    pairs_passed += int(passes)
                if pairs_passed == len(pairs):
                    completed_at.append(seed)

        _with_witness_source(probes, witness, measure)
        evidence["probes"][probe_id] = {"completedAtSeeds": completed_at}
        if completed_at:
            completed_any = True
            report.append(
                f"  {probe_id}: COMPLETED the family at seeds {completed_at} (false positive)"
            )
        else:
            report.append(f"  {probe_id}: completed the family at no seed")
    if completed_any:
        return {
            "gate": "specificity",
            "passed": False,
            "evidence": evidence,
            "report": "gate 2 SPECIFICITY FAIL: a registered witness completed the new family\n"
            + "\n".join(report),
        }
    return {
        "gate": "specificity",
        "passed": True,
        "evidence": evidence,
        "report": "gate 2 SPECIFICITY PASS: no registered witness completed the new family\n"
        + "\n".join(report),
    }


def gate_amplitude(
    probes: Any, entry: dict[str, Any], seeds: tuple[int, ...] = ADMISSION_SEEDS
) -> dict[str, Any]:
    """Gate 3: the injection's perturbation of the clean marginals is bounded."""
    evidence: dict[str, Any] = {
        "bound": {"maxStdChange": STD_CHANGE_BOUND, "maxMeanShiftStd": MEAN_SHIFT_STD_BOUND},
        "maxStdChange": 0.0,
        "maxMeanShiftStd": 0.0,
    }
    if entry["kind"] == "activation":
        # The activation contract: the rank-1 injection touches layer 2 only —
        # the residual channels must be bitwise unchanged — and beta bounded.
        beta = entry["generator"]["params"]["injectBeta"]
        residuals_identical = True
        for seed in seeds:
            for planted, clean in probes.build_family_pairs(entry, seed):
                residuals_identical = residuals_identical and all(
                    np.array_equal(
                        planted["residuals"][key],
                        clean["residuals"][key],
                        equal_nan=(key == "e"),  # e is NaN-padded by design
                    )
                    for key in ("e", "r", "origins")
                )
        evidence["residualsBitwiseUnchanged"] = residuals_identical
        evidence["injectBeta"] = beta
        ok = residuals_identical and 0.0 < beta <= ACT_BETA_BOUND
        report = (
            f"gate 3 AMPLITUDE {'PASS' if ok else 'FAIL'}: residuals bitwise unchanged "
            f"({residuals_identical}), injectBeta {beta} (bound (0, {ACT_BETA_BOUND}])"
        )
        return {"gate": "amplitude", "passed": ok, "evidence": evidence, "report": report}

    worst_std = 0.0
    worst_mean = 0.0
    for seed in seeds:
        children = probes._seed_pool("residual", seed, extra=entry)
        for index in entry["seedLayout"]["pairSeeds"]:
            rng = np.random.default_rng(probes._child_seed(children, index))
            planted, clean = probes._series_pair(rng, entry["generator"])
            for channel in range(clean.shape[1]):
                std_clean = float(clean[:, channel].std())
                worst_std = max(
                    worst_std,
                    abs(float(planted[:, channel].std()) - std_clean) / std_clean,
                )
                worst_mean = max(
                    worst_mean,
                    abs(float(planted[:, channel].mean()) - float(clean[:, channel].mean()))
                    / std_clean,
                )
    evidence["maxStdChange"] = worst_std
    evidence["maxMeanShiftStd"] = worst_mean
    ok = worst_std < STD_CHANGE_BOUND and worst_mean < MEAN_SHIFT_STD_BOUND
    report = (
        f"gate 3 AMPLITUDE {'PASS' if ok else 'FAIL'}: max per-channel std change "
        f"{worst_std:.4f} (bound < {STD_CHANGE_BOUND}), max mean shift {worst_mean:.4f} "
        f"clean-sd (bound < {MEAN_SHIFT_STD_BOUND}), measured on both pairs at seeds {list(seeds)}"
    )
    return {"gate": "amplitude", "passed": ok, "evidence": evidence, "report": report}


def gate_reproducibility(
    probes: Any, entry: dict[str, Any], seeds: tuple[int, ...] = ADMISSION_SEEDS
) -> dict[str, Any]:
    """Gate 4: two builds from the same DSL record are bitwise identical."""
    evidence: dict[str, Any] = {"seeds": list(seeds), "bitwiseIdentical": True}
    for seed in seeds:
        first_pairs = probes.build_family_pairs(entry, seed)
        second_pairs = probes.build_family_pairs(entry, seed)
        first_controls = probes.build_family_controls(entry, seed)
        second_controls = probes.build_family_controls(entry, seed)
        for a, b in [
            *[(p, q) for (p, _), (q, _) in zip(first_pairs, second_pairs)],
            *[(p, q) for (_, p), (_, q) in zip(first_pairs, second_pairs)],
            *[
                (a, b)
                for pair_a, pair_b in zip(first_controls, second_controls)
                for a, b in zip(pair_a, pair_b)
            ],
        ]:
            if not _artifacts_equal(a, b, probes):
                evidence["bitwiseIdentical"] = False
    ok = evidence["bitwiseIdentical"]
    report = (
        f"gate 4 REPRODUCIBILITY {'PASS' if ok else 'FAIL'}: double builds of pairs and "
        f"controls are {'bitwise identical' if ok else 'NOT identical'} at seeds {list(seeds)}"
    )
    return {"gate": "reproducibility", "passed": ok, "evidence": evidence, "report": report}


# --- the CLI -------------------------------------------------------------------


def admit(
    proposal_path: Path,
    registry_path: Path,
    seeds: tuple[int, ...] = ADMISSION_SEEDS,
) -> int:
    """Run the four gates on one proposal; append on full pass. Returns exit code."""
    probes = load_probes_module()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    family_id = proposal.get("familyId", "<unnamed>")
    kind = proposal.get("kind", "<no kind>")
    print(f"proposal: {family_id} (kind {kind})")
    problems = proposal_problems(proposal, registry, probes)
    if problems:
        print("schema FAIL: the proposal falls outside the DSL:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    layout = assign_seed_slots(registry, kind)
    print(f"schema OK; assigned seed slots: {json.dumps(layout)}")
    entry = {
        "familyId": family_id,
        "kind": kind,
        "origin": "generated",
        "generator": proposal["generator"],
        "seedLayout": layout,
        "witnessProbe": proposal["witnessProbe"],
        "rationale": proposal["rationale"],
        "gateEvidence": {},
    }
    gates = [
        ("witness", lambda: gate_witness(probes, entry, seeds)),
        (
            "specificity",
            lambda: gate_specificity(probes, entry, registry, seeds),
        ),
        ("amplitude", lambda: gate_amplitude(probes, entry, seeds)),
        ("reproducibility", lambda: gate_reproducibility(probes, entry, seeds)),
    ]
    evidence: dict[str, Any] = {}
    for name, gate in gates:
        result = gate()
        print(result["report"])
        evidence[name] = result["evidence"]
        if not result["passed"]:
            print(f"\nVERDICT: REJECTED at gate {name}; the registry was not touched.")
            return 1
    entry["gateEvidence"] = evidence
    registry["families"].append(entry)
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"\nVERDICT: ADMITTED; {family_id} appended to {registry_path} "
        f"(origin 'generated', {len(registry['families'])} families total)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Admit one fixture-family proposal through the four gates into the frozen registry."
    )
    parser.add_argument("--proposal", type=Path, required=True, help="proposal JSON file (the DSL)")
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"registry to append to (default: {DEFAULT_REGISTRY})",
    )
    args = parser.parse_args(argv)
    return admit(args.proposal, args.registry)


if __name__ == "__main__":
    sys.exit(main())
