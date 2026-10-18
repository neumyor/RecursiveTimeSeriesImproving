"""Probe validation and execution: planted-defect families, controls, isolation.

These tests exercise experiments/probes.py exactly the way the loop will:
sources written by "the agent" into probes/<id>/probe.py, proposals as parsed
probe_proposal.json, every run inside the killpg-protected subprocess.  The
probe sources below are intentionally realistic agent output — numpy-only
diagnostics under the candidate whitelist — so check_probe_source, the worker
contract checks and the family/controls validation rule are all exercised
end to end.  The trust-rule tests pin the contract: honest probes complete
exactly their own fixture family, a near-zero noise probe completes nothing
(negative controls), and a linear probe cannot see the bilinear defect.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_probes() -> Any:
    """Load experiments/probes.py by file path, like the loop loads its modules."""
    source = _REPO_ROOT / "experiments" / "probes.py"
    spec = importlib.util.spec_from_file_location("rhg_probes", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rhg_probes"] = module
    spec.loader.exec_module(module)
    return module


P = _load_probes()

# A GOOD periodicity probe: period-24 band energy of the 1-step residual series.
GOOD_PROBE_SOURCE = '''import numpy as np


def create_probe():
    return Band24Probe()


class Band24Probe:
    def __init__(self):
        self.probe_id = "band24-energy"

    def run(self, artifacts):
        e = np.asarray(artifacts["residuals"]["e"], dtype=np.float64)
        e = e[np.isfinite(e).all(axis=1)]
        e = e - e.mean(axis=0)
        power = (np.abs(np.fft.rfft(e, axis=0)) ** 2).sum(axis=1)
        freqs = np.fft.rfftfreq(e.shape[0])
        periods = np.zeros_like(freqs)
        positive = freqs > 0.0
        periods[positive] = 1.0 / freqs[positive]
        band = (periods >= 21.6) & (periods <= 26.4)
        return {"band24": float(power[band].sum())}
'''

# A GOOD lag probe: R^2 of the channel-0 residual on channel 1 three rows back.
LAG3_PROBE_SOURCE = '''import numpy as np


def create_probe():
    return Xlag3Probe()


class Xlag3Probe:
    def __init__(self):
        self.probe_id = "xlag3-r2"

    def run(self, artifacts):
        e = np.asarray(artifacts["residuals"]["e"], dtype=np.float64)
        e = e[np.isfinite(e).all(axis=1)]
        y = e[3:, 0]
        design = np.hstack([e[:-3, 1:2], np.ones((y.shape[0], 1))])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        total = y - y.mean()
        return {"xlag3R2": float(1.0 - residual @ residual / (total @ total))}
'''

# A GOOD nonlinear probe: R^2 of the squared channel-0 residual on thresholded
# (tanh) channel-1 lags — the bilinear/conditional-variance signature.
BILINEAR_PROBE_SOURCE = '''import numpy as np


def create_probe():
    return BilinearProbe()


class BilinearProbe:
    def __init__(self):
        self.probe_id = "bilinear-r2"

    def run(self, artifacts):
        e = np.asarray(artifacts["residuals"]["e"], dtype=np.float64)
        e = e[np.isfinite(e).all(axis=1)]
        y = e[2:, 0] ** 2
        s = float(e[:, 1].std())
        design = np.hstack([
            np.tanh(e[1:-1, 1:2] / s),
            np.tanh(e[:-2, 1:2] / s),
            np.ones((y.shape[0], 1)),
        ])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        total = y - y.mean()
        return {"bilinearR2": float(1.0 - residual @ residual / (total @ total))}
'''

# A LINEAR-ONLY probe: 16-lag linear R^2 of both channels' residuals.  It can
# see the sinusoid and the lag-3 defect but must be blind to the bilinear one.
LINEAR_PROBE_SOURCE = '''import numpy as np


def create_probe():
    return LinearLagProbe()


class LinearLagProbe:
    def __init__(self):
        self.probe_id = "linear-lags"

    def run(self, artifacts):
        e = np.asarray(artifacts["residuals"]["e"], dtype=np.float64)
        e = e[np.isfinite(e).all(axis=1)]
        y = e[8:, 0]
        feats = [e[8 - k : e.shape[0] - k, c] for k in range(1, 9) for c in range(e.shape[1])]
        design = np.hstack([np.stack(feats, axis=1), np.ones((y.shape[0], 1))])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        total = y - y.mean()
        return {"linearR2": float(1.0 - residual @ residual / (total @ total))}
'''

# A NOISE probe: lag-13 cross-channel R^2 — nothing is planted at lag 13, so
# the signalKey is pure near-zero noise.
NOISE_PROBE_SOURCE = '''import numpy as np


def create_probe():
    return NoiseProbe()


class NoiseProbe:
    def __init__(self):
        self.probe_id = "noise-r2"

    def run(self, artifacts):
        e = np.asarray(artifacts["residuals"]["e"], dtype=np.float64)
        e = e[np.isfinite(e).all(axis=1)]
        y = e[13:, 0]
        design = np.hstack([e[:-13, 1:2], np.ones((y.shape[0], 1))])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ beta
        total = y - y.mean()
        return {"noiseR2": float(1.0 - residual @ residual / (total @ total))}
'''

# A GOOD activation probe: linear-probe R^2 of the bandpassed channel-0
# residual (aligned to the activations through meta["activationOrigins"]) from
# each layer's mean-pooled representation; the signal is the layer-2 minus
# layer-1 gap.  The sampled origins sit at stride 2, so period 24 in time is
# period 12 in sample units.
ACT_PROBE_SOURCE = '''import numpy as np


def create_probe():
    return LayerGapProbe()


class LayerGapProbe:
    def __init__(self):
        self.probe_id = "layer-gap"

    def run(self, artifacts):
        r = np.asarray(artifacts["residuals"]["r"], dtype=np.float64)
        rows = np.asarray(artifacts["meta"]["activationOrigins"], dtype=np.int64)
        y = r[rows, :, 0].mean(axis=1)
        y = y - y.mean()
        spec = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(y.shape[0])
        periods = np.zeros_like(freqs)
        positive = freqs > 0.0
        periods[positive] = 1.0 / freqs[positive]
        keep = (periods >= 10.8) & (periods <= 13.2)
        y_band = np.fft.irfft(spec * keep, n=y.shape[0])

        def r2(layer):
            feat = np.asarray(artifacts["activations"][layer], dtype=np.float64)
            feat = feat.mean(axis=1)
            design = np.hstack([feat, np.ones((feat.shape[0], 1))])
            beta, *_ = np.linalg.lstsq(design, y_band, rcond=None)
            residual = y_band - design @ beta
            total = y_band - y_band.mean()
            return 1.0 - float(residual @ residual) / float(total @ total)

        return {"layer2_gap": float(r2("layer2") - r2("layer1"))}
'''

# A GARBAGE probe: declares signalKey "band24" but never produces it.
GARBAGE_PROBE_SOURCE = '''import numpy as np


def create_probe():
    return GarbageProbe()


class GarbageProbe:
    def __init__(self):
        self.probe_id = "garbage-probe"

    def run(self, artifacts):
        return {"unrelated": float(1.0)}
'''

HANG_PROBE_SOURCE = '''import numpy as np


def create_probe():
    return HangProbe()


class HangProbe:
    def __init__(self):
        self.probe_id = "hang-probe"

    def run(self, artifacts):
        while True:
            pass
'''

BIG_PROBE_SOURCE = '''import numpy as np


def create_probe():
    return BigProbe()


class BigProbe:
    def __init__(self):
        self.probe_id = "big-output"

    def run(self, artifacts):
        return {"blob": "x" * 8192}
'''

OS_PROBE_SOURCE = '''import os

import numpy as np


def create_probe():
    return OsProbe()


class OsProbe:
    def __init__(self):
        self.probe_id = "os-probe"

    def run(self, artifacts):
        return {"env": float(len(os.environ))}
'''


def write_probe(tmp_path: Path, probe_id: str, source: str) -> Path:
    """Place a probe exactly where the loop looks for it: probes/<id>/probe.py."""
    directory = tmp_path / "probes" / probe_id
    directory.mkdir(parents=True)
    source_path = directory / "probe.py"
    source_path.write_text(source, encoding="utf-8")
    return source_path


def make_proposal(probe_id: str, signal_key: str, direction: str = "higher") -> dict[str, Any]:
    return {
        "probeId": probe_id,
        "measures": "period-24 band energy of the 1-step residual series",
        "signalKey": signal_key,
        "expectedDirection": direction,
        "rationale": (
            "a period-24 oscillation the incumbent cannot predict lands in e, "
            "so the band energy rises when the defect is planted"
        ),
    }


def registry_artifacts(task_name: str, seed: int, sizes: dict[str, int]) -> dict[str, Any]:
    """A small but real artifacts dict from a REGISTRY task's train_diag series.

    meta["task"] stays the de-identified "synthetic": the registry name must
    never ride along, because probe output flows back into the agent's prompt.
    """
    from harnessing_ts.headroom.synthetic import REGISTRY

    task = REGISTRY[task_name](seed=seed, sizes=sizes)
    return P.artifacts_from_series(task.train_diag, meta_update={"seed": seed})


# Family-major pair order, matching the registry order make_fixtures emits:
# three seed families + two pipeline-admitted families, two pairs each.
FAMILY_OF_PAIR = (
    "period24", "period24",
    "cross-lag3", "cross-lag3",
    "nonlinear", "nonlinear",
    "horizon-range", "horizon-range",
    "level-bias", "level-bias",
)


# ---------------------------------------------------------------------------
# error taxonomy
# ---------------------------------------------------------------------------


def test_error_taxonomy_subclasses_the_candidate_hierarchy() -> None:
    for category in (P.ProbeTimeout, P.ProbeCrash, P.ProbeContractViolation, P.ProbeOutputTooLarge):
        assert issubclass(category, P.ProbeError)
        assert issubclass(category, P.cand.CandidateError)
        assert issubclass(category, RuntimeError)  # headroom-contracts.md §1
    assert issubclass(P.ProbeError, RuntimeError)


# ---------------------------------------------------------------------------
# validation (the trust mechanism)
# ---------------------------------------------------------------------------


def test_good_band_probe_passes_validation(tmp_path: Path) -> None:
    source = write_probe(tmp_path, "band24-energy", GOOD_PROBE_SOURCE)
    # Fixtures and controls are deterministic: the same seed must reproduce
    # them exactly, or validation records would not be comparable.
    fixtures = P.make_fixtures("residual", seed=0)
    assert len(fixtures) == 10  # five families, two pairs each
    controls = P.make_controls("residual", seed=0)
    assert [len(entry) for entry in controls] == [P._N_CONTROLS] * 10
    for index, (planted, clean) in enumerate(fixtures):
        assert np.isnan(planted["residuals"]["e"][0]).all()  # NaN-padded head
        assert np.isnan(planted["residuals"]["e"][-1]).all()  # ... and tail
        assert planted["residuals"]["origins"].shape == planted["residuals"]["r"].shape[:1]
        assert planted["residuals"]["origins"].dtype == np.int64
        assert planted["meta"]["task"] == "synthetic"  # de-identified
        assert planted["meta"]["family"] == FAMILY_OF_PAIR[index]
        assert set(planted) == {"residuals", "activations", "meta"}  # no windows/targets
    for (planted, clean), pair_controls in zip(fixtures, controls):
        for control in pair_controls:
            assert control["meta"]["family"] == clean["meta"]["family"]
            assert control["meta"]["control"] is True
    for (p1, c1), (p2, c2) in zip(fixtures, P.make_fixtures("residual", seed=0)):
        # equal_nan: the e series is NaN-padded by design, and NaN != NaN under
        # the default comparison.
        assert np.array_equal(p1["residuals"]["e"], p2["residuals"]["e"], equal_nan=True)
        assert np.array_equal(c1["residuals"]["r"], c2["residuals"]["r"])

    # The loop's proposals carry a sixth key, "kind"; extra keys are tolerated.
    proposal = {**make_proposal("band24-energy", "band24"), "kind": "residual"}
    result = P.validate_probe(source, proposal, "residual")
    assert result["ok"] is True, result["reason"]
    assert result["reason"] is None
    assert result["deterministic"] is True
    assert result["familyComplete"] is True
    assert result["families"]["period24"] == {"pairsPassed": 2, "pairsTotal": 2, "complete": True}
    # Specificity: a periodicity probe completes only its own family.
    assert result["families"]["cross-lag3"]["complete"] is False
    assert result["families"]["nonlinear"]["complete"] is False
    assert set(result["measured"]) == set(range(10))
    for entry in result["measured"].values():
        assert len(entry["controls"]) == P._N_CONTROLS
    # The direction/margin claims hold on the probe's own family; the other
    # families' pairs may legitimately fail (that is the specialization rule).
    period24 = [e for e in result["measured"].values() if e["family"] == "period24"]
    for entry in period24:
        assert entry["planted"] > entry["clean"]  # direction holds
        assert entry["margin"] > P.RELATIVE_MARGIN
        assert entry["passes"] is True


def test_wrong_direction_fails_with_measured_values(tmp_path: Path) -> None:
    source = write_probe(tmp_path, "band24-energy", GOOD_PROBE_SOURCE)
    result = P.validate_probe(
        source, make_proposal("band24-energy", "band24", direction="lower"), "residual"
    )
    assert result["ok"] is False
    reason = result["reason"]
    assert "lower" in reason
    # The repair turn needs the numbers: every measured value must appear.
    for entry in result["measured"].values():
        assert f"{entry['planted']:.6g}" in reason
        assert f"{entry['clean']:.6g}" in reason


def test_garbage_probe_is_a_contract_violation(tmp_path: Path) -> None:
    source = write_probe(tmp_path, "garbage-probe", GARBAGE_PROBE_SOURCE)
    result = P.validate_probe(source, make_proposal("garbage-probe", "band24"), "residual")
    assert result["ok"] is False
    assert "ProbeContractViolation" in result["reason"]
    assert "band24" in result["reason"]  # the expected signal key is named
    assert result["measured"] == {}  # the very first pair already failed


def test_lag3_probe_completes_only_cross_lag3(tmp_path: Path) -> None:
    source = write_probe(tmp_path, "xlag3-r2", LAG3_PROBE_SOURCE)
    result = P.validate_probe(source, make_proposal("xlag3-r2", "xlag3R2"), "residual")
    assert result["ok"] is True, result["reason"]
    assert result["families"]["cross-lag3"] == {"pairsPassed": 2, "pairsTotal": 2, "complete": True}
    assert result["families"]["period24"]["complete"] is False
    assert result["families"]["nonlinear"]["complete"] is False


def test_nonlinear_probe_completes_only_the_nonlinear_family(tmp_path: Path) -> None:
    source = write_probe(tmp_path, "bilinear-r2", BILINEAR_PROBE_SOURCE)
    result = P.validate_probe(source, make_proposal("bilinear-r2", "bilinearR2"), "residual")
    assert result["ok"] is True, result["reason"]
    assert result["families"]["nonlinear"] == {"pairsPassed": 2, "pairsTotal": 2, "complete": True}
    assert result["families"]["period24"]["complete"] is False
    assert result["families"]["cross-lag3"]["complete"] is False
    # The measured signal is the bilinear R^2 gap: large on the planted
    # nonlinear pairs, at the clean noise floor everywhere else.
    nonlinear = [e for e in result["measured"].values() if e["family"] == "nonlinear"]
    for entry in nonlinear:
        assert entry["planted"] > 0.05
        assert abs(entry["clean"]) < 0.01


def test_linear_probe_fails_the_nonlinear_family(tmp_path: Path) -> None:
    """The bilinear defect is invisible to linear probes by construction: the
    planted channel's second moments equal the clean ones up to one common
    scale factor, so a linear R^2 sees pure sampling noise there.  The probe
    may honestly complete the linear families (sinusoid, lag-3) — the point
    is that it cannot see the bilinear defect."""
    source = write_probe(tmp_path, "linear-lags", LINEAR_PROBE_SOURCE)
    result = P.validate_probe(source, make_proposal("linear-lags", "linearR2"), "residual")
    assert result["families"]["nonlinear"]["complete"] is False
    assert result["families"]["nonlinear"]["pairsPassed"] < 2


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_negative_controls_kill_noise_probes(tmp_path: Path, seed: int) -> None:
    """A near-zero signalKey must complete no family: with the negative
    controls and the gap>spread rule, pure noise cannot clear the clean
    cluster's own width, at every seed."""
    source = write_probe(tmp_path, "noise-r2", NOISE_PROBE_SOURCE)
    result = P.validate_probe(source, make_proposal("noise-r2", "noiseR2"), "residual", seed=seed)
    assert result["ok"] is False
    assert result["familyComplete"] is False
    assert all(not summary["complete"] for summary in result["families"].values())
    # and the values themselves tell the noise story: near zero on both sides
    for entry in result["measured"].values():
        assert abs(entry["planted"]) < 0.01
        assert abs(entry["clean"]) < 0.01


def test_activation_probe_passes_validation(tmp_path: Path) -> None:
    fixtures = P.make_fixtures("activation", seed=0)
    controls = P.make_controls("activation", seed=0)
    assert len(fixtures) == 2
    assert [len(entry) for entry in controls] == [P._N_CONTROLS] * 2
    for (planted, clean), pair_controls in zip(fixtures, controls):
        # The injection happens at layer 2's output: layer 1 (upstream) must be
        # identical between the pair members, layer 2 (and 3) must differ.
        assert np.array_equal(planted["activations"]["layer1"], clean["activations"]["layer1"])
        assert not np.array_equal(planted["activations"]["layer2"], clean["activations"]["layer2"])
        assert planted["activations"]["layer2"].dtype == np.float32
        # The loop's alignment promise: activation i corresponds to r row
        # meta["activationOrigins"][i] (arange in the fixtures).
        n_sub = planted["activations"]["layer2"].shape[0]
        assert planted["meta"]["activationOrigins"] == list(range(n_sub))
        assert planted["residuals"]["origins"].shape[0] == n_sub
        for control in pair_controls:
            assert control["meta"]["variant"] == "clean"  # no injection applied
            assert not np.array_equal(control["activations"]["layer2"], planted["activations"]["layer2"])

    source = write_probe(tmp_path, "layer-gap", ACT_PROBE_SOURCE)
    result = P.validate_probe(
        source, make_proposal("layer-gap", "layer2_gap", direction="higher"), "activation"
    )
    assert result["ok"] is True, result["reason"]
    # Activation rule: BOTH pairs must pass (they share one net and one
    # injection direction, so a single lucky draw must not be enough).
    assert result["families"]["activation"] == {"pairsPassed": 2, "pairsTotal": 2, "complete": True}
    assert result["familyComplete"] is True
    for entry in result["measured"].values():
        assert entry["planted"] > entry["clean"]
        assert entry["margin"] > P.RELATIVE_MARGIN


def test_validate_probe_rejects_bad_proposals_and_kind(tmp_path: Path) -> None:
    source = write_probe(tmp_path, "band24-energy", GOOD_PROBE_SOURCE)
    result = P.validate_probe(source, {"probeId": "band24-energy"}, "residual")
    assert result["ok"] is False
    for key in ("measures", "signalKey", "expectedDirection", "rationale"):
        assert key in result["reason"]

    proposal = make_proposal("band24-energy", "band24", direction="sideways")
    result = P.validate_probe(source, proposal, "residual")
    assert result["ok"] is False
    assert "sideways" in result["reason"]

    # The loop's proposals carry a sixth key, "kind"; extra keys are tolerated,
    # but a declared kind that disagrees with the validation kind is rejected.
    proposal = {**make_proposal("band24-energy", "band24"), "kind": "activation"}
    result = P.validate_probe(source, proposal, "residual")
    assert result["ok"] is False
    assert "activation" in result["reason"] and "residual" in result["reason"]

    with pytest.raises(ValueError, match="kind"):
        P.validate_probe(source, make_proposal("band24-energy", "band24"), "spectral")


# ---------------------------------------------------------------------------
# execution (run_probe)
# ---------------------------------------------------------------------------


def test_run_probe_round_trip_on_registry_artifacts(tmp_path: Path) -> None:
    artifacts = registry_artifacts("hmm_ar_switch", seed=7, sizes={"train": 1200, "diag": 800, "cert": 60, "test": 60})
    source = write_probe(tmp_path, "band24-energy", GOOD_PROBE_SOURCE)
    out = P.run_probe(source, artifacts)
    assert set(out) == {"band24"}
    assert isinstance(out["band24"], float)
    assert math.isfinite(out["band24"])


def test_run_probe_timeout_kills_process_group(tmp_path: Path) -> None:
    artifacts = registry_artifacts("random_walk", seed=3, sizes={"train": 1500, "diag": 600, "cert": 40, "test": 40})
    source = write_probe(tmp_path, "hang-probe", HANG_PROBE_SOURCE)

    start = time.perf_counter()
    with pytest.raises(P.ProbeTimeout, match="5s"):
        P.run_probe(source, artifacts, timeout=5.0)
    elapsed = time.perf_counter() - start
    # 5s budget + 15s slack: if killpg failed, communicate() would block on the
    # busy-waiting child forever; finishing promptly is the observable proof
    # that the whole process group died.
    assert elapsed < 20.0


def test_run_probe_rejects_oversized_output(tmp_path: Path) -> None:
    artifacts = registry_artifacts("random_walk", seed=3, sizes={"train": 1500, "diag": 600, "cert": 40, "test": 40})
    source = write_probe(tmp_path, "big-output", BIG_PROBE_SOURCE)
    with pytest.raises(P.ProbeOutputTooLarge, match="4096"):
        P.run_probe(source, artifacts)


def test_run_probe_rejects_artifacts_with_windows_or_targets(tmp_path: Path) -> None:
    """Windows/targets are not part of the probe contract: probe output is
    rendered back into the agent's prompt, so raw windows and ground-truth
    targets must not even reach the subprocess."""
    artifacts = registry_artifacts("random_walk", seed=3, sizes={"train": 1500, "diag": 600, "cert": 40, "test": 40})
    source = write_probe(tmp_path, "band24-energy", GOOD_PROBE_SOURCE)
    leaked = {
        **artifacts,
        "windows": np.zeros((2, 32, 1)),
        "targets": np.zeros((2, 4, 1)),
    }
    with pytest.raises(ValueError, match="not part of the probe contract"):
        P.run_probe(source, leaked)


# ---------------------------------------------------------------------------
# source vetting before any subprocess
# ---------------------------------------------------------------------------


def test_check_source_rejects_import_before_any_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        pytest.fail("a subprocess spawned before the probe source was vetted")

    monkeypatch.setattr(P, "_run_probe_subprocess", boom)
    source = write_probe(tmp_path, "os-probe", OS_PROBE_SOURCE)

    result = P.validate_probe(source, make_proposal("os-probe", "band24"), "residual")
    assert result["ok"] is False
    assert "source rejected" in result["reason"]
    assert "import of 'os'" in result["reason"]


ECHO_SOURCE = '''import os


def create_probe():
    return EchoProbe()


class EchoProbe:
    def __init__(self):
        self.probe_id = "echo-path"

    def run(self, artifacts):
        return {"file": __file__, "dir": os.path.dirname(__file__)}
'''


def test_probes_run_from_a_temp_copy_so_no_heldout_path_escapes(tmp_path: Path) -> None:
    """An agent-written probe can echo its own __file__ (or raise it); the
    running path is a temp copy, so neither the output rendered into the
    agent's feedback nor a crash message can carry the heldout location."""
    frozen = tmp_path / "probe-sources" / "echo-path" / "probe.py"
    frozen.parent.mkdir(parents=True)
    frozen.write_text(ECHO_SOURCE, encoding="utf-8")
    artifacts = registry_artifacts(
        "random_walk", seed=3, sizes={"train": 1500, "diag": 600, "cert": 40, "test": 40}
    )

    wrapper = P.run_probe(frozen, artifacts)
    for value in wrapper.values():
        assert str(tmp_path) not in value
        assert "headroom-probe-" in value

    crash = tmp_path / "probe-sources" / "echo-crash" / "probe.py"
    crash.parent.mkdir(parents=True)
    crash.write_text(
        ECHO_SOURCE.replace('return {"file": __file__, "dir": os.path.dirname(__file__)}',
                             'raise RuntimeError("my dir is " + os.path.dirname(__file__))')
        .replace('"echo-path"', '"echo-crash"'),
        encoding="utf-8",
    )
    with pytest.raises(P.ProbeCrash) as excinfo:
        P.run_probe(crash, artifacts)
    assert str(tmp_path) not in str(excinfo.value)
    assert "headroom-probe-" in str(excinfo.value)


def test_run_probe_binds_probe_id_to_the_directory_name(tmp_path: Path) -> None:
    artifacts = registry_artifacts("random_walk", seed=3, sizes={"train": 1500, "diag": 600, "cert": 40, "test": 40})
    # The file declares one id but lives under another directory: the runner
    # must refuse to execute it under the directory's name.
    source = write_probe(tmp_path, "not-the-declared-id", GOOD_PROBE_SOURCE)
    with pytest.raises(P.ProbeContractViolation, match="probe_id"):
        P.run_probe(source, artifacts)
