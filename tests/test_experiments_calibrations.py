from __future__ import annotations

"""Network-free small-size tests for the pre-P2-freeze calibrations.

Everything runs at stage0-smoke sizes on synthetic data written to tmp_path:
CAL-2's bootstrap null + short-lookback planted acceptance plumbing on a small
synthetic task, CAL-3's real-task loader path on a tiny timestamp-free CSV,
and CAL-1's bandwidth-floor CI plumbing on one reduced-size seed.  No network,
and nothing outside tmp_path is written (the calibration functions return
dicts; only experiments/calibrations.py's main() writes artifacts).
"""

import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import _min_bandwidth
from harnessing_ts.headroom.synthetic import REGISTRY


def _load_calibrations():
    # experiments/ is deliberately not on the repo pythonpath, so the module
    # is imported by file path exactly like the loop imports realdata.
    name = "rhg_calibrations"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "experiments" / "calibrations.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cal = _load_calibrations()


def reduced_cfg() -> HeadroomConfig:
    """Stage0-smoke-style config: burn_in = 31 + 48 = 79, L = 48, H = 24."""
    return HeadroomConfig(
        input_length=48,
        horizon=24,
        lag_segments=((0, 7), (8, 15), (16, 31)),
        fir_len_cap=48,
        beyond_l_lags=2,
        band_period_range=(4, 24),
        budget_candidates=6,
        ridge_alphas=(1.0, 100.0),
    )


def _write_channel_csv(path: Path, values: np.ndarray) -> None:
    """Write a timestamp-free CSV: header ch_0..ch_{C-1}, then the raw rows."""
    values = np.asarray(values, dtype=np.float64)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"ch_{c}" for c in range(values.shape[1])])
        for row in values:
            writer.writerow([f"{v:.6f}" for v in row])


_ROWS = 4200  # loader-minimum: keeps >=100 usable test origins at embargo 96


def _multiperiod_csv(path: Path, seed: int = 0) -> None:
    """Write a small lgssm_multiperiod realization as the CAL-2 stand-in task.

    The process's latent level (observed only through channel mixing plus AR
    noise) makes m0-short-lookback a genuine planted defect next to the full
    lookback, which a hand-rolled toy series would not: a pure sinusoid's
    phase is already identified from two lags, so a longer window buys nothing.
    """
    task = REGISTRY["lgssm_multiperiod"](
        seed=seed, sizes={"train": _ROWS // 2, "diag": _ROWS // 2, "cert": 10, "test": 10}
    )
    _write_channel_csv(path, task.train_diag)


def _random_walk_grid(rows: int, channels: int, seed: int = 3) -> np.ndarray:
    """A near-random-walk grid: the Exchange-Rate negative control in miniature."""
    rng = np.random.default_rng(seed)
    x = 0.01 * rng.standard_normal((rows, channels)).cumsum(axis=0)
    x += 0.05 * rng.standard_normal((rows, channels))
    return x


def test_bootstrap_refit_varies_the_fit_deterministically() -> None:
    cfg = reduced_cfg()
    rng = np.random.default_rng(11)
    train = 0.5 * rng.standard_normal((400, 2)).cumsum(axis=0)
    meta = {"L": cfg.input_length, "H": cfg.horizon, "C": 2, "seed": 0}
    windows = np.lib.stride_tricks.sliding_window_view(train, (cfg.input_length, 2))
    x = windows[: -cfg.horizon, 0]  # [B, L, C]

    plain = cal.m0.variant("m0-linear")
    plain.fit(train, meta)
    boot_a = cal._BootstrapRefitM0(boot_seed=5)
    boot_a.fit(train, meta)
    boot_b = cal._BootstrapRefitM0(boot_seed=5)
    boot_b.fit(train, meta)
    boot_c = cal._BootstrapRefitM0(boot_seed=6)
    boot_c.fit(train, meta)

    pred_plain = plain.predict_batch(x)
    pred_a = boot_a.predict_batch(x)
    # Same boot seed -> bitwise identical fit; the resample actually moved the
    # fit off the plain one; a different seed moves it somewhere else.
    np.testing.assert_array_equal(pred_a, boot_b.predict_batch(x))
    assert not np.allclose(pred_a, pred_plain)
    assert not np.allclose(pred_a, boot_c.predict_batch(x))


def test_cal2_null_and_planted_rates(tmp_path: Path) -> None:
    cfg = reduced_cfg()
    csv_path = tmp_path / "small-seasonal.csv"
    _multiperiod_csv(csv_path)
    result = cal.cal2_etth1_acceptance(csv_path, cfg, n_null=3, n_planted=3)

    null, planted = result["null"], result["planted"]
    assert 0.0 <= null["acceptanceRate"] <= 1.0
    assert 0.0 <= planted["acceptanceRate"] <= 1.0
    assert null["n"] == 3 and planted["n"] == 3
    for row in null["draws"] + planted["draws"]:
        assert 0.0 <= row["dmP"] <= 1.0
        assert isinstance(row["accepted"], bool)
        assert np.isfinite(row["meanD"])
        assert row["bandwidth"] == _min_bandwidth(cfg)
    # The planted defect is a real improvement: the deterministic pair must
    # be accepted by the DM gate.
    assert planted["deterministic_pair"]["accepted"] is True

    r0 = result["round0_certification"]
    assert isinstance(r0["saturated"], bool)
    assert r0["totalThreshold"] > 0.0
    assert np.isfinite(r0["totalUcb"])
    assert isinstance(r0["failures"], list)
    # The stand-in CSV is timestamp-free, so the loader reports an unknown
    # frequency and the plain channel count of the written grid.
    assert result["freq"] == "unknown"
    assert result["channels"] == 4
    assert result["diag_usable_origins"] > 0
    # Verdict follows the pre-registered rule from the measured rates.
    expected = null["acceptanceRate"] <= 0.10 and planted["acceptanceRate"] >= 0.5
    assert result["verdict"]["pass"] is expected


def test_cal3_loader_path(tmp_path: Path) -> None:
    cfg = reduced_cfg()
    csv_path = tmp_path / "tiny-random-walk.csv"
    _write_channel_csv(csv_path, _random_walk_grid(_ROWS, channels=3))
    result = cal.cal3_exchange_rate(csv_path, cfg)

    # A timestamp-free all-numeric grid: the loader's auto-detection treats it
    # as pure channels with an unknown frequency (the exchange-rate layout).
    assert result["freq"] == "unknown"
    assert result["channels"] == 3

    r0 = result["round0"]
    assert isinstance(r0["saturated"], bool)
    assert np.isfinite(r0["totalLcb"])
    assert r0["totalThreshold"] > 0.0
    assert isinstance(r0["failures"], list)

    assert len(result["top_card_rows"]) >= 1
    for row in result["top_card_rows"]:
        assert {"label", "aspect", "delta", "lcb", "ucb"} <= set(row)
        assert np.isfinite(row["delta"]) and np.isfinite(row["lcb"])
    assert np.isfinite(result["diag_card_total"]["lcb"])
    assert isinstance(result["verdict"]["pass"], bool)
    assert result["verdict"]["pass"] == r0["saturated"]


def test_cal1_bandwidth_smoke() -> None:
    cfg = reduced_cfg()
    floors = (_min_bandwidth(cfg), 120)  # baseline must be the current floor
    sizes = {"train": 700, "diag": 500, "cert": 700, "test": 300}
    result = cal.cal1_bandwidth(cfg, sizes, seeds=range(1), floors=floors, reps=2, ci_size=300)

    assert result["floors"] == list(floors)
    assert result["seeds"] == [0] and result["reps_per_seed"] == 2
    seed_row = result["per_seed"][0]
    assert seed_row["n_cert_origins"] > 300
    for floor in floors:
        assert seed_row["widths"][floor] > 0.0
        assert seed_row["se"][floor] > 0.0
        assert 0.0 <= seed_row["coverage_lcb"][floor] <= 1.0
        assert 0.0 <= seed_row["coverage_ucb"][floor] <= 1.0
        assert seed_row["width_ratio"][floor] > 0.0
    assert seed_row["width_ratio"][floors[0]] == 1.0
    assert 0.0 <= result["pooled_coverage"][floors[1]]["lcb"] <= 1.0
    assert result["verdict"]["smallest_floor_all_seeds_ge_090"] in (floors[0], floors[1], None)
    assert isinstance(result["verdict"]["freeze_or_raise"], str)


def test_cal1_rejects_mismatched_baseline_floor() -> None:
    cfg = reduced_cfg()
    with pytest.raises(ValueError, match="must equal the current HAC floor"):
        cal.cal1_bandwidth(
            cfg,
            {"train": 700, "diag": 500, "cert": 700, "test": 300},
            seeds=range(1),
            floors=(120, 240),
            reps=1,
            ci_size=300,
        )
