"""Probe contract: AST source vetting, planted-defect validation, isolated execution.

A probe is a diagnostic script the treatment-arm agent writes against the
incumbent's evaluation artifacts; the agent-facing prompt text lives
elsewhere.  This module is the trust mechanism: it decides whether a probe's
declared signal is real (validation) and runs validated probes in isolation
(execution).  The isolation patterns mirror
backend/harnessing_ts/headroom/candidate.py deliberately — the probe runner
reuses cand's whitelists, environment builder, exit-code convention and error
hierarchy rather than growing a second one.

Probe contract
--------------
One probe = one directory ``probes/<probe-id>/probe.py`` whose id matches
``cand.CANDIDATE_ID_PATTERN`` ([a-z0-9][a-z0-9-]{0,63}) and whose file defines
a top-level ``create_probe()`` returning an object with:

- ``probe_id: str`` — must equal the directory name (``run_probe``) resp. the
  proposal's ``probeId`` (``validate_probe``);
- ``run(artifacts: dict) -> dict`` — a JSON-serializable summary dict whose
  TOTAL serialized size is at most ``OUTPUT_CAP_BYTES`` (4096) bytes; larger
  output is rejected as ``ProbeOutputTooLarge``.  Deterministic by contract:
  the same artifacts must produce the same output (bitwise not required, but
  validation reruns one fixture and requires identical output).

``artifacts`` is the incumbent's evaluation artifacts — residuals and
activations only, the two channels the paper gives probes.  Windows and
targets are deliberately NOT part of the contract: probe output is rendered
back into the agent's prompt, so raw input windows and ground-truth targets
must not flow through it; anything target-conditional a diagnostic needs is
already in ``r``, per (origin, h, c):

    residuals    {"e": [T_e, C] float64,      1-step residual series,
                                              NaN-padded at the head and
                                              tail (realized rows only),
                  "r": [n_r, H, C] float64,   H-step residuals at origins,
                  "origins": [n_r] int64}     time index of each r row (into e)
    activations  {layer_name: [n_sub, n_p, d] float32}  per-layer activations
                 of the subsampled windows ({} when the incumbent has none)
    meta         dict of plain JSON values: L, H, C, seed, incumbent=model_id,
                 task = the DE-IDENTIFIED string "synthetic" | "real-deidentified"
                 (never the registry/task name — probe output flows back into
                 the agent's prompt), and, when activations are present,
                 "activationOrigins": list[int] of length n_sub indexing rows
                 of r, so activations[i] aligns with r[activationOrigins[i]].

``probe_proposal.json`` sits beside the source and declares what the probe
claims to measure (extra keys are tolerated; the loop adds "kind"):

    {"probeId": str, "measures": str, "signalKey": str,
     "expectedDirection": "higher" | "lower", "rationale": str}

``signalKey`` names the output key that carries the validated signal; its
value must be a finite number.

Validation (``validate_probe``) — the trust rule
-----------------------------------------------
Deterministic fixture families (``make_fixtures``), each pair (planted,
clean), plus per-pair NEGATIVE CONTROLS (``make_controls``): _N_CONTROLS (2)
fresh clean-only fixtures of the same family — an empirical null for the
probe's signalKey.

- residual kind, five families of two pairs each, in registry order:
  period24, cross-lag3, nonlinear — the hand-designed "seed" families — and
  horizon-range, level-bias, admitted through the experiments/fixture_gen.py
  proposal pipeline.  A family is a frozen
  record in experiments/specs/probe-fixture-registry.json — its DSL
  parameters (constrained clean-process and injection constructors), its
  witness probe, its rationale and the gate evidence that admitted it; the
  pipeline, not this module, decides what gets in.
  * ``period24``: clean = per-channel AR(1) (phi=0.6, innovation sd 1.0)
    plus white noise (sd 0.5) on both channels, T=4096, C=2; planted =
    clean + a period-24 sinusoid on channel 0 at amplitude
    0.5 * sd(clean channel 0).
  * ``cross-lag3``: same clean; planted channel 0 += 0.5 *
    sd(clean ch0)/sd(clean ch1) times channel 1 three rows back.
  * ``nonlinear``: clean channel 0 = pure AR(1) (no white part — with one
    the planted/clean covariance ratio would differ between numerator and
    denominator of the correlations, breaking the exact linear invisibility
    below), channel 1 = AR(1)+white; planted channel 0 follows the same
    AR(1) but its innovation is scaled by (1 + gamma * tanh(channel-1
    centered level / its sd)), gamma = 0.5/sd(driver) — a threshold-type
    cross-channel coupling at 0.5 innovation-sd.  Every second moment of
    the planted channel equals the clean ones up to one common scale factor
    (the driver is symmetric, zero-mean and independent of channel 0, which
    kills every cross term), so linear probes see EXACTLY the clean
    process — measured: a 16-lag linear R^2 probe completes no nonlinear
    pair at seeds 0-2 — while the conditional variance carries the defect,
    which a bilinear probe (e0^2 on tanh(e1) lags) reads at R^2 ~ 0.13
    against a ~1e-3 clean floor.
  * ``horizon-range``: clean = white both channels (innovation sd 1.0 plus
    white sd 0.5, phi=0 — no autocorrelation); planted channel 0 += a
    period-12 (= 4*(H-1)) sinusoid at 0.75 * sd(clean ch0).  A stationary
    series injection cannot literally scale with h, so the horizon design
    is realized spectrally: the narrowband component contributes ~2% of
    the 1-step-gap residual increment power but ~27% of the full 3-step-gap
    power on channel 0 — invisible at short horizons, detectable only
    through a long-horizon contrast.  The witness is the ratio of 3-step to
    1-step residual increment power.
  * ``level-bias``: clean = AR(1)+white both channels; planted channel 0 +=
    a half-cosine level drift, 0.7 * sd(clean ch0) * cos(pi * t / T) — a
    level offset slow enough that the global-mean incumbent absorbs its DC
    part (a true constant dies here) but that no local window can track, so
    the residual carries a persistent block-level bias.  The witness is
    the mu^2 statistic: the mean over 8 equal blocks of e of the squared
    block means.
  All fixtures are built by ``artifacts_from_series`` with a mean-predictor
  incumbent, so a defect the incumbent cannot predict lands in the
  residuals at full strength.
- activation kind, one family of two pairs.  Both members share one series
  (channel 0 = a period-24 sinusoid of amplitude 1.0 plus white noise sd
  0.5, channel 1 = white noise) and one fixed seeded 3-layer torch-CPU MLP
  (64->32->16->8, ReLU, threads=1); they differ only in the net variant:
  planted adds a rank-1 signal to layer 2's output — a fixed seeded unit
  direction scaled by the KNOWN analytic period-24 component of the
  channel-0 residual at the sampled origins (so a probe reads it from
  ``r``, never from targets), at per-coordinate sd ``_ACT_INJECT_BETA``
  (2.0) times the clean layer-2 sd.  Layer 2 (and only layer 2; layer 1 is
  upstream) becomes linearly informative about the bandpassed residual.

A PAIR passes when the declared direction holds against the pair's clean
member AND every control, each with relative margin > 5%
(|a-b| / max(|a|, |b|, tiny)), AND the planted value clears the whole clean
cluster by more than the cluster's own width (gap > spread: planted -
max(clean side) > max(clean side) - min(clean side), mirrored for "lower").
Margins alone do not reject near-zero noise (planted 1.6e-5 vs clean 1.2e-5
is a 29% "margin"): the controls turn the clean side into an empirical null
and gap > spread is the significance floor such noise cannot systematically
clear — for exchangeable noise draws the pair rule holds only with ordering
luck (Monte Carlo: 6% per pair for light-tailed noise, ~15% for
heavy-tailed R^2-like noise, versus ~25% for the margin rule alone), and
the two-pair family rule squares it again (~0.4-2% per family).  A FAMILY
completes when every one of its pairs passes.
A probe PASSES validation when at least one family completes (a periodicity
probe completes period24, a lag probe cross-lag3, a nonlinear probe the
nonlinear family, a horizon-contrast probe horizon-range, a block-level
probe level-bias — specialization is the point), the output is
deterministic (one fixture run twice gives identical output) and within the
size cap.  Measured at seeds 0-2: the periodicity, lag-3 and bilinear probes
each complete exactly their own family and a lag-13 cross-channel noise
probe completes nothing; the horizon-range and level-bias witnesses
likewise complete exactly their own families (fixture_gen admission gates:
seeds 0-4).  Runtime: 41 subprocess runs per residual validation,
9 per activation (~3s / ~2s measured), against the 15s target.

Probe-side failures come back as ``{"ok": False, "reason": ...}`` rather
than raised, because a rejected probe is a budgeted outcome for the loop.

Execution (``run_probe``)
-------------------------
Same subprocess protocol as candidate.py — literally the same helper,
``cand.run_isolated``: the source passes ``check_probe_source`` first, then
runs under ``cand.build_env()`` in a fresh temp cwd, stdout merged into
stderr, ``start_new_session`` + SIGKILL of the whole process group on
timeout, results via a JSON file (the only channel out).  The worker is this module itself, spawned through ``python -c`` with
the module path as argv so the child's sys.path never sees experiments/
(whose loop.py/batch.py must not shadow anything); artifacts travel via a
temp npz (e/r/origins float64/int64, activations float32, meta as JSON).
Exit codes follow candidate_worker: 0 = result written, 1 = crash, 3 =
contract violation.

This is correctness tooling, not a security boundary — the same caveat as
candidate.py's docstring.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from harnessing_ts.headroom import candidate as cand

# Probe ids double as directory names under probes/, exactly like candidates.
PROBE_ID_PATTERN = cand.CANDIDATE_ID_PATTERN
# Hard cap on a probe's serialized output: a summary, not an exfiltration
# channel for the artifacts themselves.
OUTPUT_CAP_BYTES = 4096
DEFAULT_TIMEOUT_SECONDS = 120.0
# Validation rule: declared direction with relative margin above this floor.
RELATIVE_MARGIN = 0.05
# Per-pair negative controls: fresh clean-only fixtures of the same family.
# Two of them: the gap>spread significance floor needs a clean cluster wide
# enough to estimate its own spread; with a single control the cluster is
# two points and near-zero noise passes far more often (false-pass analysis
# in the module docstring).
_N_CONTROLS = 2
_TINY = 1e-12  # guards the margin denominator against two zeros
_PAIRS_PER_FAMILY = 2

# --- fixture design constants (documented in the module docstring) ----------
_INPUT_LENGTH = 32
_HORIZON = 4
_RESIDUAL_C = 2
_N_RESIDUALS = 256
_RESIDUAL_STRIDE = 7

# --- the frozen fixture-family registry --------------------------------------
#
# The fixture families live in experiments/specs/probe-fixture-registry.json
# (checked in): each entry is a DSL record — constrained clean-process and
# injection constructors, never free code — plus its witness probe, its
# rationale, the child-seed slots its pairs/controls draw from and the gate
# evidence that admitted it.  experiments/fixture_gen.py is the
# development-time pipeline that vets and appends entries; this module only
# ever READS the registry.  Families occupy fixed seed slots in registry
# order (SeedSequence children are keyed by spawn position), so appending a
# family never disturbs the draws of the families before it.
_REGISTRY_PATH = Path(__file__).resolve().parent / "specs" / "probe-fixture-registry.json"
_REGISTRY: dict[str, Any] | None = None


def load_registry() -> dict[str, Any]:
    """The frozen fixture-family registry (cached read of the checked-in JSON)."""
    global _REGISTRY
    if _REGISTRY is None:
        registry = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        families = registry.get("families")
        if not isinstance(families, list) or not families:
            raise ValueError(f"fixture registry {_REGISTRY_PATH} carries no families")
        _REGISTRY = registry
    return _REGISTRY


def _families_of_kind(kind: str) -> list[dict[str, Any]]:
    """The registry's family entries of one kind, in registry order."""
    entries = [entry for entry in load_registry()["families"] if entry["kind"] == kind]
    if not entries:
        raise ValueError(f"fixture registry has no {kind!r} families")
    return entries


def _family_ids(kind: str) -> tuple[str, ...]:
    """Family ids per kind, in the order make_fixtures emits their pairs."""
    return tuple(entry["familyId"] for entry in _families_of_kind(kind))


def _kinds() -> tuple[str, ...]:
    return tuple(sorted({entry["kind"] for entry in load_registry()["families"]}))


class ProbeError(cand.CandidateError):
    """Base for every probe failure; the subclass is the category.

    ProbeTimeout/ProbeCrash/ProbeContractViolation also subclass the matching
    cand category, so a caller that already handles candidate failures (the
    loop) classifies probe failures the same way.
    """


class ProbeTimeout(cand.CandidateTimeout, ProbeError):
    """The probe subprocess exceeded its budget and its process group was killed."""


class ProbeCrash(cand.CandidateCrash, ProbeError):
    """The worker exited nonzero (probe raised, OOM, ...) or produced invalid output."""


class ProbeContractViolation(cand.ContractViolation, ProbeError):
    """probe.py breaks the probe contract: missing create_probe/run/probe_id,
    wrong output type, non-JSON-serializable output, probe_id mismatch, or an
    output missing its declared signalKey."""


class ProbeOutputTooLarge(ProbeError):
    """The probe's serialized output exceeds OUTPUT_CAP_BYTES."""


# --- source vetting ---------------------------------------------------------

# cand.check_source requires a top-level create_adapter() — the one candidate-
# contract rule that does not transfer to probes (they define create_probe).
# Appending a clean stub keeps the ENTIRE whitelist walk (imports, forbidden
# names, forbidden attribute calls, line numbers, future-annotations handling)
# inside cand.check_source, so the probe and candidate whitelists can never
# drift apart; the probe factory requirement is then enforced separately on
# the unmodified source.  A probe that defines its own create_adapter is
# harmless: the stub only ever satisfies the check, and the real factory
# lookup below still goes through create_probe.
_ADAPTER_STUB = (
    '\n\n\ndef create_adapter():\n'
    '    raise RuntimeError("probes define create_probe(), not create_adapter()")\n'
)


def check_probe_source(source: str) -> None:
    """Reject probe sources outside the candidate whitelist; raises SourceRejected."""
    try:
        cand.check_source(source + _ADAPTER_STUB)
    except cand.SourceRejected as exc:
        # Reword the fixed cand header for the repair turn; if cand's wording
        # ever changes the message merely stays candidate-flavored.
        raise cand.SourceRejected(
            str(exc).replace("candidate source rejected:", "probe source rejected:", 1)
        ) from exc
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "create_probe" for node in ast.parse(source).body
    ):
        raise cand.SourceRejected("probe.py must define a top-level create_probe() (the probe factory)")


# --- the worker (runs inside the isolated subprocess) ----------------------

# Exit codes follow candidate_worker.py: 0 = result written, 1 = crash,
# 3 = probe-contract violation (missing callable, wrong output type, ...).
# A probe raising inside run() is a crash (exit 1), mirroring a candidate
# raising inside fit(); only structural breakage is a contract violation.
_WORKER_CONTRACT_EXIT = cand.WORKER_CONTRACT_EXIT


class _WorkerContractError(Exception):
    """Probe-contract failure inside the worker; maps to exit code 3."""


# The child is started as `python -c <bootstrap> <probes.py> <npz_in> <json_out>`.
# Going through -c (rather than executing this file) keeps experiments/ off the
# child's sys.path — its loop.py/batch.py must not shadow any import the probe
# or numpy/scipy/torch makes — and the child's cwd is a fresh empty temp dir,
# so sys.path[0] resolves to nothing.  The worker code itself stays in this
# module: there is no second file to keep in sync.
_WORKER_BOOTSTRAP = """\
import importlib.util, sys
spec = importlib.util.spec_from_file_location('rhg_probes_worker', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
sys.exit(module.worker_main(sys.argv[2:]))
"""

_PROBES_SOURCE = Path(__file__).resolve()


def _jsonify_numpy(value: Any) -> float | int:
    """json.dumps default handler: numpy scalars become plain numbers.

    np.float64 already subclasses float, but np.float32 / np.int32 do not;
    rejecting them would burn repair turns on a pure serialization detail.
    Anything else still fails the dumps (and becomes a contract violation).
    """
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def worker_main(argv: list[str]) -> int:
    """Subprocess entry, spawned via _WORKER_BOOTSTRAP; never run by hand."""
    if len(argv) != 2:
        print("usage: probes.py <npz_in> <json_out>", file=sys.stderr)
        return 1
    try:
        _worker_run(dict(np.load(Path(argv[0]), allow_pickle=False)), Path(argv[1]))
    except _WorkerContractError:
        traceback.print_exc()
        return _WORKER_CONTRACT_EXIT
    except Exception:
        traceback.print_exc()
        return 1
    return 0


def _worker_run(data: dict[str, np.ndarray], json_out: Path) -> None:
    """Reassemble the artifacts, run the probe, write the JSON result file."""
    # The npz flattens the nested artifacts dict: activations live under
    # "act/<layer_name>" keys (npz names may carry slashes), meta is a JSON
    # string.  Everything else round-trips as-is.
    artifacts: dict[str, Any] = {
        "residuals": {
            "e": np.asarray(data["e"], dtype=np.float64),
            "r": np.asarray(data["r"], dtype=np.float64),
            "origins": np.asarray(data["origins"], dtype=np.int64),
        },
        "activations": {
            key[4:]: np.asarray(data[key], dtype=np.float32)
            for key in data
            if key.startswith("act/")
        },
        "meta": json.loads(str(data["meta"].item())),
    }
    probe, probe_id = _load_probe(Path(str(data["source_path"].item())))
    out = probe.run(artifacts)
    if not isinstance(out, dict):
        raise _WorkerContractError(
            f"run() must return a dict, got {type(out).__name__} (the summary the loop records)"
        )
    try:
        json.dumps(out, default=_jsonify_numpy)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _WorkerContractError(
            f"run() output must be JSON-serializable, got {type(exc).__name__}: {exc}"
        ) from exc
    # The JSON file is the only channel out: probeId rides along so the parent
    # can bind the result to the expected id with a clean expected-vs-actual
    # message (the worker has no access to the proposal).
    json_out.write_text(
        json.dumps({"probeId": probe_id, "output": out}, default=_jsonify_numpy),
        encoding="utf-8",
    )


def _load_probe(source_path: Path) -> tuple[Any, str]:
    """Import the vetted probe module and return (probe, probe_id).

    Mirrors candidate_worker._load_adapter: the parent already AST-vetted the
    source, so this is the first and only place probe code executes.
    """
    spec = importlib.util.spec_from_file_location("_headroom_probe_worker", source_path)
    if spec is None or spec.loader is None:
        raise _WorkerContractError(f"cannot build an import spec for probe source {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise _WorkerContractError(
            f"importing probe source {source_path} raised {type(exc).__name__}: {exc}"
        ) from exc
    factory = getattr(module, "create_probe", None)
    if not callable(factory):
        raise _WorkerContractError(f"probe module must define a callable create_probe(), found {factory!r}")
    try:
        probe = factory()
    except Exception as exc:
        raise _WorkerContractError(f"create_probe() raised {type(exc).__name__}: {exc}") from exc
    probe_id = getattr(probe, "probe_id", None)
    if not isinstance(probe_id, str):
        raise _WorkerContractError(
            f"probe must carry a str probe_id attribute, found {type(probe_id).__name__}"
        )
    if not callable(getattr(probe, "run", None)):
        raise _WorkerContractError(
            f"probe must expose a callable run(artifacts), found {type(probe).__name__} without one"
        )
    return probe, probe_id


# --- isolated execution (parent side) ---------------------------------------


def _write_artifacts_npz(path: Path, artifacts: dict[str, Any], source_path: Path) -> None:
    """Flatten one artifacts dict into the worker's npz contract.

    Structural validation only (keys, shapes, dtypes) — loud ValueErrors for
    harness-side wiring bugs.  Windows/targets are not part of the contract
    and are rejected if present: a dict that carries them is mis-wired, and
    letting them slip through would reopen the information leak the contract
    exists to close.
    """
    if not isinstance(artifacts, dict):
        raise ValueError(f"artifacts must be a dict, got {type(artifacts).__name__}")
    required = ("residuals", "activations", "meta")
    missing = [key for key in required if key not in artifacts]
    if missing:
        raise ValueError(f"artifacts is missing keys {missing}")
    leaked = [key for key in ("windows", "targets") if key in artifacts]
    if leaked:
        raise ValueError(
            f"artifacts carries {leaked}, which are not part of the probe contract "
            f"(probes see residuals and activations only)"
        )
    residuals = artifacts["residuals"]
    if not (isinstance(residuals, dict) and {"e", "r", "origins"} <= set(residuals)):
        raise ValueError(
            f"artifacts['residuals'] must be a dict with keys 'e', 'r' and 'origins', got {residuals!r}"
        )
    e = np.asarray(residuals["e"], dtype=np.float64)
    r = np.asarray(residuals["r"], dtype=np.float64)
    origins = np.asarray(residuals["origins"], dtype=np.int64)
    if e.ndim != 2:
        raise ValueError(f"residuals['e'] must be [T_e, C], got shape {e.shape}")
    if r.ndim != 3:
        raise ValueError(f"residuals['r'] must be [n_r, H, C], got shape {r.shape}")
    if origins.ndim != 1 or origins.shape[0] != r.shape[0]:
        raise ValueError(
            f"residuals['origins'] must have one entry per r row ({r.shape[0]}), "
            f"got shape {origins.shape}"
        )
    activations = artifacts["activations"]
    if not isinstance(activations, dict):
        raise ValueError(f"activations must be a dict, got {type(activations).__name__}")
    arrays: dict[str, Any] = {
        "e": e,
        "r": r,
        "origins": origins,
        "meta": np.array(json.dumps(artifacts["meta"])),
        "source_path": np.array(str(Path(source_path).resolve())),
    }
    for name, values in activations.items():
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 3:
            raise ValueError(
                f"activations[{name!r}] must be [n_sub, n_p, d], got shape {values.shape}"
            )
        arrays[f"act/{name}"] = values
    np.savez(path, **arrays)


def _run_probe_subprocess(source_path: Path, artifacts: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Run one probe in the isolated worker; returns its {"probeId", "output"} wrapper.

    Failures raise exactly one ProbeError subclass (see the classes above);
    the display id in every message is the probe directory name, the only id
    the parent knows before the probe has run.
    """
    source_path = Path(source_path)
    if not source_path.is_file():
        raise ValueError(f"probe source {source_path} does not exist")
    display_id = source_path.parent.name
    with tempfile.TemporaryDirectory(prefix="headroom-probe-") as run_dir:
        npz_in, json_out = Path(run_dir) / "probe-in.npz", Path(run_dir) / "probe-out.json"
        _write_artifacts_npz(npz_in, artifacts, source_path)
        command = [sys.executable, "-c", _WORKER_BOOTSTRAP, str(_PROBES_SOURCE), str(npz_in), str(json_out)]
        code, tail = cand.run_isolated(
            command,
            Path(run_dir),
            timeout,
            lambda message: ProbeTimeout(f"probe {display_id!r} {message}"),
        )
        if code == _WORKER_CONTRACT_EXIT:
            raise ProbeContractViolation(
                f"probe {display_id!r} broke the probe contract inside the worker:\n{tail}"
            )
        if code != 0:
            raise ProbeCrash(
                f"probe {display_id!r} crashed in the worker (exit code {code}):\n{tail}"
            )
        try:
            wrapper = json.loads(json_out.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ProbeCrash(
                f"probe {display_id!r}: worker exited 0 but wrote no readable JSON result to "
                f"{json_out.name}: {type(exc).__name__}: {exc}"
            ) from exc
        if not (
            isinstance(wrapper, dict)
            and isinstance(wrapper.get("probeId"), str)
            and isinstance(wrapper.get("output"), dict)
        ):
            raise ProbeCrash(
                f"probe {display_id!r}: worker wrote a malformed result wrapper "
                f"(expected a JSON object with a str probeId and a dict output), got {wrapper!r}"
            )
        # Size cap, enforced on every run (validation and execution alike):
        # re-serializing with the same json.dumps defaults reproduces exactly
        # the bytes the worker wrote for the output part.
        size = len(json.dumps(wrapper["output"]).encode("utf-8"))
        if size > OUTPUT_CAP_BYTES:
            raise ProbeOutputTooLarge(
                f"probe {display_id!r} serialized output is {size} bytes, "
                f"expected at most {OUTPUT_CAP_BYTES} bytes"
            )
        return wrapper


def run_probe(source_path: Path, artifacts: dict[str, Any], timeout: float = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Run one (validated) probe on real evaluation artifacts; returns its output dict.

    The source is vetted before any subprocess spawns (cand.SourceRejected on
    violation).  The probe object's probe_id must equal the directory name —
    probes/<probe-id>/probe.py is the layout the loop reads, so a swapped file
    cannot silently run under another id.  Raises ProbeTimeout, ProbeCrash,
    ProbeContractViolation or ProbeOutputTooLarge; never returns partial data.
    """
    source_path = Path(source_path)
    check_probe_source(source_path.read_text(encoding="utf-8"))
    wrapper = _run_probe_subprocess(source_path, artifacts, timeout)
    expected = source_path.parent.name
    actual = wrapper["probeId"]
    if actual != expected:
        raise ProbeContractViolation(
            f"probe {expected!r}: probe_id must equal the probe directory name, "
            f"expected {expected!r}, got {actual!r}"
        )
    return wrapper["output"]


# --- validation: the trust mechanism -----------------------------------------


def _proposal_problems(proposal: Any) -> list[str]:
    """Expected-vs-actual problems with an agent-written probe_proposal.json."""
    if not isinstance(proposal, dict):
        return [f"must be a JSON object, got {type(proposal).__name__}"]
    problems: list[str] = []
    for key in ("probeId", "measures", "signalKey", "expectedDirection", "rationale"):
        if key not in proposal:
            problems.append(f"missing key {key!r}")
        elif not isinstance(proposal[key], str) or not proposal[key]:
            problems.append(f"{key!r} must be a non-empty string, got {proposal[key]!r}")
    if isinstance(proposal.get("probeId"), str) and not PROBE_ID_PATTERN.fullmatch(proposal["probeId"]):
        problems.append(
            f"probeId must match [a-z0-9][a-z0-9-]{{0,63}}, got {proposal['probeId']!r}"
        )
    if "expectedDirection" in proposal and proposal["expectedDirection"] not in ("higher", "lower"):
        problems.append(
            f"expectedDirection must be 'higher' or 'lower', got {proposal['expectedDirection']!r}"
        )
    return problems


def _signal_value(wrapper: dict[str, Any], probe_id: str, signal_key: str) -> float:
    """The declared signal of one run: probe_id bound, signalKey present and finite."""
    actual = wrapper["probeId"]
    if actual != probe_id:
        raise ProbeContractViolation(
            f"probe {probe_id!r}: probe_id must equal probe_proposal.json's probeId, "
            f"expected {probe_id!r}, got {actual!r}"
        )
    output = wrapper["output"]
    if signal_key not in output:
        raise ProbeContractViolation(
            f"probe {probe_id!r}: output must carry its declared signalKey {signal_key!r}, "
            f"found keys {sorted(output)}"
        )
    value = output[signal_key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ProbeContractViolation(
            f"probe {probe_id!r}: signalKey {signal_key!r} must carry a finite number, got {value!r}"
        )
    return float(value)


def _canonical(output: dict[str, Any]) -> str:
    """Canonical serialization for the determinism comparison.

    sort_keys makes key order irrelevant (the contract asks for stable output,
    not byte-identical key order), and NaN survives the comparison because
    both sides serialize it to the same "NaN" token — unlike ==, where
    NaN != NaN would flag every NaN-carrying probe as non-deterministic.
    """
    return json.dumps(output, sort_keys=True)


def _pair_passes(direction: str, planted: float, clean_side: list[float]) -> tuple[bool, float]:
    """The per-pair trust rule; returns (passes, worst relative margin).

    clean_side carries the pair's clean member plus every negative control —
    an empirical null for the signalKey.  The pair passes when the declared
    direction holds against EVERY null value with > RELATIVE_MARGIN relative
    margin (the prescribed rule) AND the planted value clears the whole null
    cluster by more than the cluster's own width (gap > spread — the
    nonparametric significance floor that kills near-zero noise passes, which
    the purely relative margin rule demonstrably let through).
    """
    worst = max(clean_side) if direction == "higher" else min(clean_side)
    gap = (planted - worst) if direction == "higher" else (worst - planted)
    spread = max(clean_side) - min(clean_side)
    margins = [abs(planted - value) / max(abs(planted), abs(value), _TINY) for value in clean_side]
    passes = gap > 0.0 and min(margins) > RELATIVE_MARGIN and gap > spread
    return passes, min(margins)


def validate_probe(
    source_path: Path,
    proposal: dict[str, Any],
    kind: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Validate one probe against deterministic planted-defect fixture families.

    Returns a JSON-plain record the loop stores as-is: {"ok": bool,
    "reason": str | None, "families": {family: {"pairsPassed": int,
    "pairsTotal": int, "complete": bool}}, "pairs_passed": int,
    "familyComplete": bool, "deterministic": bool, "measured": {pair index:
    {"family", "planted", "clean", "controls", "margin", "passes"}}}.
    ok is True iff familyComplete (every pair of at least one family passes
    the control rule) AND deterministic AND within the size cap.  Probe-side
    failures (timeout, crash, contract violation, oversized output) come back
    as ok=False with the classified message, not as raised errors: a rejected
    probe is a budgeted round outcome, not a harness bug.  A bad ``kind`` is
    a harness bug and raises ValueError.
    """
    if kind not in _kinds():
        raise ValueError(f"kind must be one of {sorted(_kinds())}, got {kind!r}")
    measured: dict[int, dict[str, Any]] = {}
    result: dict[str, Any] = {
        "ok": False,
        "reason": None,
        "families": {},
        "pairs_passed": 0,
        "familyComplete": False,
        "deterministic": False,
        "measured": measured,
    }
    problems = _proposal_problems(proposal)
    # The loop's proposals carry a sixth key, "kind" (which data the probe
    # reads); extra keys are tolerated, but a declared kind that disagrees
    # with the validation kind means the agent described one probe and the
    # loop is validating another.
    if "kind" in proposal and proposal["kind"] != kind:
        problems.append(
            f"proposal declares kind {proposal['kind']!r} but validation is running {kind!r}"
        )
    if problems:
        result["reason"] = "invalid probe_proposal.json: " + "; ".join(problems)
        return result
    source_path = Path(source_path)
    try:
        check_probe_source(source_path.read_text(encoding="utf-8"))
    except cand.SourceRejected as exc:
        result["reason"] = f"source rejected: {exc}"
        return result

    probe_id = proposal["probeId"]
    signal_key = proposal["signalKey"]
    direction = proposal["expectedDirection"]
    families = _family_ids(kind)
    fixtures = make_fixtures(kind, seed)
    controls = make_controls(kind, seed)
    try:
        first_output: dict[str, Any] | None = None
        for index, (planted, clean) in enumerate(fixtures):
            wrapper_planted = _run_probe_subprocess(source_path, planted, DEFAULT_TIMEOUT_SECONDS)
            if index == 0:
                first_output = wrapper_planted["output"]
            value_planted = _signal_value(wrapper_planted, probe_id, signal_key)
            value_clean = _signal_value(
                _run_probe_subprocess(source_path, clean, DEFAULT_TIMEOUT_SECONDS),
                probe_id,
                signal_key,
            )
            control_values = [
                _signal_value(
                    _run_probe_subprocess(source_path, control, DEFAULT_TIMEOUT_SECONDS),
                    probe_id,
                    signal_key,
                )
                for control in controls[index]
            ]
            passes, margin = _pair_passes(direction, value_planted, [value_clean] + control_values)
            measured[index] = {
                "family": families[index // _PAIRS_PER_FAMILY],
                "planted": value_planted,
                "clean": value_clean,
                "controls": control_values,
                "margin": margin,
                "passes": passes,
            }
        assert first_output is not None
        # Determinism: the same fixture (pair 0, planted) run a second time in
        # a fresh subprocess must produce identical output — this catches
        # clock-seeded and hash-order-dependent probes that an in-process
        # double call would not.
        repeat = _run_probe_subprocess(source_path, fixtures[0][0], DEFAULT_TIMEOUT_SECONDS)
        _signal_value(repeat, probe_id, signal_key)  # same binding checks on the rerun
    except ProbeError as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        _fill_family_summary(result, families, len(fixtures))
        return result

    _fill_family_summary(result, families, len(fixtures))
    deterministic = _canonical(first_output) == _canonical(repeat["output"])
    result["deterministic"] = deterministic
    family_complete = any(summary["complete"] for summary in result["families"].values())
    result["familyComplete"] = family_complete
    if not family_complete:
        by_family = ", ".join(
            f"{family} {summary['pairsPassed']}/{summary['pairsTotal']}"
            for family, summary in result["families"].items()
        )
        values = "; ".join(
            f"pair{index} [{entry['family']}]: {entry['planted']:.6g} vs clean {entry['clean']:.6g} "
            f"(controls {', '.join(f'{value:.6g}' for value in entry['controls'])}, "
            f"margin {entry['margin']:.3g}, {'pass' if entry['passes'] else 'fail'})"
            for index, entry in sorted(measured.items())
        )
        result["reason"] = (
            f"declared direction {direction!r} with the control rule completed no fixture family "
            f"(validation needs every pair of one family to pass); by family: {by_family}; "
            f"signalKey {signal_key!r} values, planted vs clean side: {values}"
        )
        return result
    if not deterministic:
        result["reason"] = (
            "probe output is not deterministic: rerunning the same pair-0 planted fixture produced "
            f"{_canonical(first_output)[:200]} then {_canonical(repeat['output'])[:200]}"
        )
        return result
    result["ok"] = True
    return result


def _fill_family_summary(result: dict[str, Any], families: tuple[str, ...], n_pairs: int) -> None:
    """Aggregate the per-pair measurements into the per-family record."""
    pairs_per_family = n_pairs // len(families)
    measured: dict[int, dict[str, Any]] = result["measured"]
    for family_index, family in enumerate(families):
        entries = [
            measured[pair]
            for pair in range(family_index * pairs_per_family, (family_index + 1) * pairs_per_family)
            if pair in measured
        ]
        passed = sum(1 for entry in entries if entry["passes"])
        result["families"][family] = {
            "pairsPassed": passed,
            "pairsTotal": pairs_per_family,
            "complete": passed == pairs_per_family,
        }
    result["pairs_passed"] = sum(1 for entry in measured.values() if entry["passes"])


# --- fixture generation ------------------------------------------------------


def artifacts_from_series(
    values: np.ndarray,
    *,
    input_length: int = _INPUT_LENGTH,
    horizon: int = _HORIZON,
    n_residuals: int = _N_RESIDUALS,
    residual_stride: int = _RESIDUAL_STRIDE,
    meta_update: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one internally-consistent artifacts dict from a [T, C] series.

    The incumbent is the mean predictor (the classic naive baseline): every
    structure in the series the incumbent cannot know lands in the residuals
    at full strength, which is exactly what the planted defects exploit, and
    unlike a persistence incumbent it does not difference the defect away.
    Fixtures are synthetic, so fitting the mean on the full series breaks no
    train/test boundary — what matters is that the same incumbent processes
    both members of a pair.  e is the series minus its channel means,
    NaN-padded at the head and tail so probes must handle the real
    artifacts' NaN layout; H-step residuals are sampled at residual_stride.
    No windows/targets by contract — the fixture builders keep the windows
    to themselves.  Used by make_fixtures and by tests that build artifacts
    from REGISTRY tasks.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"values must be [T, C], got shape {values.shape}")
    t_len, channels = values.shape
    last = t_len - 1 - horizon  # last origin whose H targets lie inside the series
    residual_origins = (input_length - 1) + residual_stride * np.arange(n_residuals)
    if residual_origins[-1] > last:
        raise ValueError(
            f"series of length {t_len} is too short for {n_residuals} residuals at stride "
            f"{residual_stride} with L={input_length}, H={horizon}; last usable origin is {last}"
        )
    mean = values.mean(axis=0)
    # r[o, h, c] = x[o+h+1, c] - mean_c: the mean-predictor residual at origin o.
    r = np.stack([values[o + 1 : o + 1 + horizon] - mean for o in residual_origins])
    e = np.full((t_len, channels), np.nan)
    # Realized through row T-H INCLUSIVE, mirroring segment_data's e_full
    # (residuals.py): the last origin with a fully realized H-step target is
    # T-1-H, whose 1-step residual lands on row T-H.
    e[1 : t_len - horizon + 1] = values[1 : t_len - horizon + 1] - mean
    meta: dict[str, Any] = {
        "L": input_length,
        "H": horizon,
        "C": channels,
        "T": t_len,
        "incumbent": "mean",
        "task": "synthetic",
    }
    if meta_update:
        meta.update(meta_update)
    return {
        "residuals": {"e": e, "r": r, "origins": residual_origins},
        "activations": {},
        "meta": meta,
    }


# --- the DSL builders (the registry's constrained constructors) ----------------
#
# These functions consume a family entry's "generator" record.  The DSL is
# small by design (see experiments/fixture_gen.py, which owns the schema and
# the admission gates): four parametric clean-channel constructors, five
# parametric residual injections, one fixed-form activation constructor.
# The draw order below is FROZEN and spec-independent, so no proposal can
# game the random stream: one joint innovation block (T, C) scaled per
# channel (every channel consumes its column whatever its type), the AR(1)
# recursion for every channel, deterministic sinusoids, then one white block
# (T, n_white) assigned to the white-carrying channels in channel order.


def _channel_params(channel: dict[str, Any]) -> dict[str, Any]:
    """Normalize one DSL clean-channel spec to phi/innovationSd/whiteSd/sinusoid."""
    kind = channel["type"]
    if kind == "ar1":
        return {
            "phi": float(channel["phi"]),
            "innovationSd": float(channel["innovationSd"]),
            "whiteSd": 0.0,
            "sinusoid": None,
        }
    if kind == "ar1_plus_white":
        return {
            "phi": float(channel["phi"]),
            "innovationSd": float(channel["innovationSd"]),
            "whiteSd": float(channel["whiteSd"]),
            "sinusoid": None,
        }
    if kind == "white":
        return {"phi": 0.0, "innovationSd": 0.0, "whiteSd": float(channel["sd"]), "sinusoid": None}
    if kind == "sinusoid_plus_white":
        return {
            "phi": 0.0,
            "innovationSd": 0.0,
            "whiteSd": float(channel["whiteSd"]),
            "sinusoid": (float(channel["period"]), float(channel["amplitude"])),
        }
    raise ValueError(f"unknown clean channel type {kind!r}")


def _clean_series(
    rng: np.random.Generator, clean_spec: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(clean, innovation, phi) [T, C] arrays from the DSL clean spec."""
    length = int(clean_spec["length"])
    channels = [_channel_params(channel) for channel in clean_spec["channels"]]
    innovation = rng.standard_normal((length, len(channels))) * np.array(
        [channel["innovationSd"] for channel in channels]
    )
    phi = np.array([channel["phi"] for channel in channels])
    ar = np.empty_like(innovation)
    ar[0] = innovation[0]
    for t in range(1, length):
        ar[t] = phi * ar[t - 1] + innovation[t]
    clean = ar.copy()
    time_axis = np.arange(length)
    for index, channel in enumerate(channels):
        if channel["sinusoid"] is not None:
            period, amplitude = channel["sinusoid"]
            clean[:, index] += amplitude * np.sin(2.0 * np.pi * time_axis / period)
    white_channels = [index for index, ch in enumerate(channels) if ch["whiteSd"] > 0.0]
    if white_channels:
        white = rng.standard_normal((length, len(white_channels)))
        for column, index in enumerate(white_channels):
            clean[:, index] += channels[index]["whiteSd"] * white[:, column]
    return clean, innovation, phi


def _inject_series(
    clean: np.ndarray,
    innovation: np.ndarray,
    phi: np.ndarray,
    injection: dict[str, Any],
) -> np.ndarray:
    """planted = clean + the DSL injection; see the module docstring for designs."""
    kind = injection["type"]
    time_axis = np.arange(clean.shape[0])
    planted = clean.copy()
    if kind in ("sinusoid", "horizon_scaled"):
        channel = int(injection["channel"])
        amplitude = injection["amplitudeFraction"] * float(clean[:, channel].std())
        planted[:, channel] += amplitude * np.sin(2.0 * np.pi * time_axis / injection["period"])
        return planted
    if kind == "cross_lag":
        channel = int(injection["channel"])
        source = int(injection["sourceChannel"])
        lag = int(injection["lag"])
        # The target channel reads the source channel `lag` rows back, scaled
        # so the injected component carries amplitudeFraction of its own sd.
        scale = (
            injection["amplitudeFraction"]
            * float(clean[:, channel].std())
            / float(clean[:, source].std())
        )
        planted[lag:, channel] += scale * clean[:-lag, source]
        return planted
    if kind == "threshold_bilinear":
        target = int(injection["targetChannel"])
        driver_channel = int(injection["driverChannel"])
        level = clean[:, driver_channel] - clean[:, driver_channel].mean()
        driver = np.tanh(level / level.std())
        gamma = injection["amplitudeFraction"] / driver.std()
        # Replay the target channel's AR(1) with its innovation scaled by the
        # threshold-coupled driver one row back.
        for t in range(1, clean.shape[0]):
            planted[t, target] = (
                phi[target] * planted[t - 1, target]
                + innovation[t, target] * (1.0 + gamma * driver[t - 1])
            )
        return planted
    if kind == "constant_offset":
        # The level-bias design: a level drift slower than one full cycle over
        # the series (cycles < 1 per the DSL), so the global-mean incumbent
        # absorbs only the DC part.
        channel = int(injection["channel"])
        amplitude = injection["amplitudeFraction"] * float(clean[:, channel].std())
        planted[:, channel] += amplitude * np.cos(
            2.0 * np.pi * injection["cycles"] * time_axis / clean.shape[0]
        )
        return planted
    raise ValueError(f"unknown injection type {kind!r}")


def _series_pair(rng: np.random.Generator, generator: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """(planted, clean) [T, C] series of one DSL series_pair family.

    Both members share the innovation draws, so the comparison isolates the
    planted defect exactly.
    """
    clean, innovation, phi = _clean_series(rng, generator["clean"])
    planted = _inject_series(clean, innovation, phi, generator["injection"])
    return planted, clean


def _residual_fixture_meta(
    child_seed: int, entry: dict[str, Any], control: bool
) -> dict[str, Any]:
    generator = entry["generator"]
    return {
        "kind": "residual",
        "family": entry["familyId"],
        "seed": child_seed,
        "control": control,
        "phi": generator["clean"]["channels"][0]["phi"],
        "injectionSdFraction": generator["injection"]["amplitudeFraction"],
    }


def _residual_pair(child_seed: int, entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One (planted, clean) residual-fixture pair; see the module docstring."""
    rng = np.random.default_rng(child_seed)
    series_planted, series_clean = _series_pair(rng, entry["generator"])
    meta = _residual_fixture_meta(child_seed, entry, control=False)
    return (
        artifacts_from_series(series_planted, meta_update=meta),
        artifacts_from_series(series_clean, meta_update=meta),
    )


def _residual_control(child_seed: int, entry: dict[str, Any]) -> dict[str, Any]:
    """One clean-only control fixture: a fresh draw of the family's clean process.

    The planted series is generated and discarded — the clean member never
    depends on it, and sharing one code path guarantees controls are exactly
    what a pair's clean member is: clean data.
    """
    rng = np.random.default_rng(child_seed)
    _, series_clean = _series_pair(rng, entry["generator"])
    meta = _residual_fixture_meta(child_seed, entry, control=True)
    return artifacts_from_series(series_clean, meta_update=meta)


def _activation_net(
    entry: dict[str, Any], children: list[np.random.SeedSequence]
) -> tuple[tuple[Any, ...], int]:
    """(weights, direction_seed): the family's fixed seeded net and direction.

    One fixed net and one fixed injection direction per validation seed; the
    two pairs differ only in the data draw.  Built once per build_family_*
    call and shared by the pairs (and, in build_family_controls, by the
    controls).
    """
    import torch

    # Fixture generation is tiny; pin threads like build_env does.
    torch.set_num_threads(1)
    params = entry["generator"]["params"]
    layout = entry["seedLayout"]
    net_rng = np.random.default_rng(children[layout["netSeed"]])
    fan_in = _INPUT_LENGTH * _RESIDUAL_C
    d1, d2 = int(params["layer1Width"]), int(params["layer2Width"])
    weights = (
        torch.from_numpy(net_rng.standard_normal((fan_in, d1)) / math.sqrt(fan_in)),
        torch.from_numpy(net_rng.standard_normal((d1, d2)) / math.sqrt(d1)),
        torch.from_numpy(net_rng.standard_normal((d2, _HORIZON * _RESIDUAL_C)) / math.sqrt(d2)),
    )
    return weights, _child_seed(children, layout["directionSeed"])


def _activation_pair(
    weights: tuple[Any, ...],
    data_seed: int,
    direction_seed: int,
    pair_index: int,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One (planted, clean) activation-fixture pair; see the module docstring."""
    import torch

    torch.set_num_threads(1)  # fixture generation is tiny; pin threads like build_env does
    data_rng = np.random.default_rng(data_seed)
    direction_rng = np.random.default_rng(direction_seed)
    time_axis = np.arange(params["length"])
    values = data_rng.standard_normal((params["length"], _RESIDUAL_C)) * params["noiseSd"]
    values[:, 0] += params["sinusoidAmplitude"] * np.sin(2.0 * np.pi * time_axis / params["period"])
    # r rows and the net's input windows share the same evenly spaced origins,
    # so meta["activationOrigins"] is arange — the alignment the loop promises.
    base = artifacts_from_series(
        values,
        n_residuals=params["nResiduals"],
        residual_stride=params["windowStride"],
        meta_update={"kind": "activation", "pair": pair_index},
    )
    origins = base["residuals"]["origins"]
    windows = np.stack([values[o - _INPUT_LENGTH + 1 : o + 1].reshape(-1) for o in origins])
    w1, w2, w3 = weights

    def forward(inject: Any | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = torch.from_numpy(windows)
        h1 = torch.relu(x @ w1)
        h2 = torch.relu(h1 @ w2)
        if inject is not None:
            h2 = h2 + torch.from_numpy(inject)
        return h1.numpy(), h2.numpy(), (h2 @ w3).numpy()

    # The injected scalar signal: the KNOWN analytic period-24 component of
    # the channel-0 mean-predictor residual at the sampled origins, averaged
    # over the H steps — exactly what a bandpass of r[:, :, 0].mean(axis=1)
    # recovers, so a probe reads the planted signal from r (its contract
    # channel), never from targets.
    signal = params["sinusoidAmplitude"] * np.array(
        [
            np.mean(np.sin(2.0 * np.pi * (o + np.arange(1, _HORIZON + 1)) / params["period"]))
            for o in origins
        ]
    )
    unit = direction_rng.standard_normal(params["layer2Width"])
    unit /= np.linalg.norm(unit)

    h1_clean, h2_clean, out_clean = forward(None)
    amplitude = params["injectBeta"] * float(h2_clean.std())
    inject = amplitude * (signal / signal.std())[:, None] * unit[None, :]
    h1_planted, h2_planted, out_planted = forward(inject)

    def pack(h1: np.ndarray, h2: np.ndarray, out: np.ndarray, variant: str) -> dict[str, Any]:
        pair = dict(base)
        pair["activations"] = {
            "layer1": h1[:, None, :].astype(np.float32),
            "layer2": h2[:, None, :].astype(np.float32),
            "layer3": out[:, None, :].astype(np.float32),
        }
        pair["meta"] = {
            **base["meta"],
            "variant": variant,
            "injectBeta": params["injectBeta"],
            "activationOrigins": list(range(params["nResiduals"])),
        }
        return pair

    return pack(h1_planted, h2_planted, out_planted, "planted"), pack(h1_clean, h2_clean, out_clean, "clean")


# --- family assembly: registry entries -> fixture pairs and controls ----------


def _seed_pool(
    kind: str, seed: int, extra: dict[str, Any] | None = None
) -> list[np.random.SeedSequence]:
    """The kind's child SeedSequences at one validation seed, positionally keyed.

    Every family entry declares its child-seed slots (seedLayout) as positions
    in this pool; ``extra`` adds one not-yet-registered entry (a proposal the
    fixture_gen pipeline is gating) to the pool.  SeedSequence children are
    keyed by spawn position, so the first k children are identical whatever
    the pool size: the seed families occupy 18 (residual) resp. 8
    (activation) slots and generated families append fresh ones without
    disturbing the seed families' fixtures.
    """
    entries = list(_families_of_kind(kind))
    if extra is not None and not any(
        entry["familyId"] == extra.get("familyId") for entry in entries
    ):
        entries.append(extra)
    slots: list[int] = []
    for entry in entries:
        layout = entry["seedLayout"]
        if kind == "residual":
            slots.extend(layout["pairSeeds"])
            slots.extend(seed for pair in layout["controlSeeds"] for seed in pair)
        else:
            slots.append(layout["netSeed"])
            slots.append(layout["directionSeed"])
            slots.extend(layout["pairDataSeeds"])
            slots.extend(seed for pair in layout["controlDataSeeds"] for seed in pair)
    if not slots or min(slots) < 0:
        raise ValueError(f"registry {kind!r} families declare no usable seed slots: {slots}")
    return list(np.random.SeedSequence(seed).spawn(max(slots) + 1))


def _child_seed(children: list[np.random.SeedSequence], index: int) -> int:
    """One child slot's integer seed (the only way builders consume the pool)."""
    return int(children[index].generate_state(1, dtype=np.uint32)[0])


def build_family_pairs(
    entry: dict[str, Any], seed: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """One family's (planted, clean) pairs at a validation seed.

    ``entry`` is a registry record (or an in-memory proposal entry carrying
    the same keys): familyId, kind, generator, seedLayout.  Deterministic:
    same entry + seed -> bitwise identical pairs.
    """
    layout = entry["seedLayout"]
    children = _seed_pool(entry["kind"], seed, extra=entry)
    if entry["kind"] == "residual":
        return [_residual_pair(_child_seed(children, index), entry) for index in layout["pairSeeds"]]
    params = entry["generator"]["params"]
    weights, direction_seed = _activation_net(entry, children)
    return [
        _activation_pair(
            weights, _child_seed(children, layout["pairDataSeeds"][index]), direction_seed, index, params
        )
        for index in range(_PAIRS_PER_FAMILY)
    ]


def build_family_controls(entry: dict[str, Any], seed: int) -> list[list[dict[str, Any]]]:
    """One family's per-pair negative controls at a validation seed.

    Aligned with build_family_pairs' order (controls[i] belongs to pair i):
    fresh draws of the same clean generating process — the empirical null the
    pair rule is judged against.  The activation kind shares the family's one
    net and injection direction: clean variant, new data.
    """
    layout = entry["seedLayout"]
    children = _seed_pool(entry["kind"], seed, extra=entry)
    if entry["kind"] == "residual":
        return [
            [_residual_control(_child_seed(children, index), entry) for index in pair_slots]
            for pair_slots in layout["controlSeeds"]
        ]
    params = entry["generator"]["params"]
    weights, direction_seed = _activation_net(entry, children)
    return [
        [
            # The clean member of a freshly seeded pair: same net, same
            # injection direction, new data, no injection applied.
            _activation_pair(
                weights,
                _child_seed(children, layout["controlDataSeeds"][pair][control]),
                direction_seed,
                pair,
                params,
            )[1]
            for control in range(_N_CONTROLS)
        ]
        for pair in range(_PAIRS_PER_FAMILY)
    ]


def make_fixtures(kind: str, seed: int = 0) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Deterministic (planted, clean) artifact pairs for validation; see the docstring.

    Pairs come family-major in REGISTRY order: pair i belongs to family
    _family_ids(kind)[i // _PAIRS_PER_FAMILY].  Same seed -> bitwise
    identical fixtures, so a validation record is reproducible and the loop
    can re-validate any probe against exactly the fixtures it was first
    judged on.  The families are frozen registry records
    (experiments/specs/probe-fixture-registry.json); experiments/fixture_gen.py
    is the development-time pipeline that admits new ones.  Generation stays
    well under 2s per pair (the activation net is a 64->32->16->8 CPU MLP on
    384 windows).
    """
    if kind not in _kinds():
        raise ValueError(f"kind must be one of {sorted(_kinds())}, got {kind!r}")
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for entry in _families_of_kind(kind):
        pairs.extend(build_family_pairs(entry, seed))
    return pairs


def make_controls(kind: str, seed: int = 0) -> list[list[dict[str, Any]]]:
    """Per-pair negative controls: _N_CONTROLS clean-only fixtures each, aligned
    with make_fixtures' pair order (controls[i] belongs to pair i).

    Controls are fresh draws of the same clean generating process (the
    activation kind: same fixed net and injection direction, clean variant,
    new data seeds) — the empirical null the pair rule is judged against.
    Same seed -> bitwise identical controls.
    """
    if kind not in _kinds():
        raise ValueError(f"kind must be one of {sorted(_kinds())}, got {kind!r}")
    controls: list[list[dict[str, Any]]] = []
    for entry in _families_of_kind(kind):
        controls.extend(build_family_controls(entry, seed))
    return controls



if __name__ == "__main__":  # manual debugging only; the loop uses the -c bootstrap
    sys.exit(worker_main(sys.argv[1:]))
