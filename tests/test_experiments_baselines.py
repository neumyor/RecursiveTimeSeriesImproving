"""Tests for the non-LLM baselines, loaded by file path (experiments/ is not an installed package)."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.residuals import segment_data
from harnessing_ts.headroom.synthetic import REGISTRY
from harnessing_ts.headroom.timeidx import Splits, segment_origins, windows

REPO_ROOT = Path(__file__).resolve().parents[1]

# stage0's smoke configuration: burn_in (31 + 48 = 79) covers the lookback (48)
# and every feature depth these segments exercise (beyond-l 64, horizon 27).
SMOKE_CFG = HeadroomConfig(
    input_length=48,
    lag_segments=((0, 7), (8, 15), (16, 31)),
    fir_len_cap=48,
    beyond_l_lags=2,
    band_period_range=(4, 24),
    budget_candidates=6,
    ridge_alphas=(1.0, 100.0),
)
SMALL_SIZES = {"train": 1600, "diag": 1200, "cert": 1400, "test": 2400}


def _load(name: str, source: Path):
    # experiments/ is deliberately not on the repo pythonpath (only backend/ is),
    # so these files are loaded the same way the loop loads candidate model.py files.
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before execution: dataclasses resolves string annotations via
    # sys.modules[cls.__module__].
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


baselines = _load("experiments_baselines", REPO_ROOT / "experiments" / "baselines.py")
m0 = _load("m0_template_baselines", REPO_ROOT / "experiments" / "templates" / "m0.py")


def test_one_shot_m0g_on_lgssm_xchannel() -> None:
    task = REGISTRY["lgssm_xchannel"](seed=3, sizes=SMALL_SIZES)
    result = baselines.one_shot_m0g(task, m0.create_adapter(), SMOKE_CFG, SMALL_SIZES)
    assert set(result) == {"mse_m0", "mse_m0g", "gap", "delta", "remaining_gap_ratio"}
    assert all(isinstance(value, float) and math.isfinite(value) for value in result.values())
    # The greedy corrector must leave a bounded share of the paired gap and
    # must not be catastrophically worse than plain m0.
    assert 0.0 <= result["remaining_gap_ratio"] <= 2.0
    assert result["mse_m0g"] <= 1.5 * result["mse_m0"]
    # delta is the corrector's realized gain: mse_m0g = mse_m0 - delta exactly.
    assert result["mse_m0g"] == pytest.approx(result["mse_m0"] - result["delta"])


def _last_value_plus_bias(
    values: np.ndarray, seg_start: int, seg_stop: int, cfg: HeadroomConfig, bias: float
) -> np.ndarray:
    """Naive last-value forecasts with a planted constant bias, one row per
    realized origin of the segment (segment_data's pred layout)."""
    origins = np.arange(seg_start, seg_stop - cfg.horizon)
    pred = np.broadcast_to(
        values[origins][:, None, :], (origins.size, cfg.horizon, values.shape[1])
    ).copy()
    return pred + bias


def test_uniform_m0g_postprocess_shape_and_planted_reduction() -> None:
    # Planted defect: a constant level bias on a standardized AR(1) series —
    # the intercept-repairable case, so the corrector must strictly reduce MSE.
    rng = np.random.default_rng(12)
    T = 2000
    eps = rng.standard_normal(T)
    x = np.empty(T)
    x[0] = 0.0
    for t in range(1, T):
        x[t] = 0.9 * x[t - 1] + eps[t]
    values = ((x - x.mean()) / x.std())[:, None]
    cfg = SMOKE_CFG
    bias = 0.3

    fit_start, fit_stop = 200, 1200
    apply_start, apply_stop = 1300, 1900
    seg_fit = segment_data(
        values, fit_start, fit_stop, _last_value_plus_bias(values, fit_start, fit_stop, cfg, bias), cfg
    )
    pred_apply = _last_value_plus_bias(values, apply_start, apply_stop, cfg, bias)
    seg_apply = segment_data(values, apply_start, apply_stop, pred_apply, cfg)
    # No periodicity is planted, so no band peaks: the caller hands them over.
    pred_arg = pred_apply[seg_apply.origins - apply_start]
    corrected = baselines.uniform_m0g_postprocess(pred_arg, seg_fit, seg_apply, cfg, [])

    assert corrected.shape == pred_arg.shape == (
        seg_apply.origins.size,
        cfg.horizon,
        values.shape[1],
    )
    _, y = windows(values, seg_apply.origins, cfg)
    mse_before = float(np.mean(np.square(y - pred_arg)))
    mse_after = float(np.mean(np.square(y - corrected)))
    assert mse_after < mse_before, "the corrector must repair the planted level bias"


def test_rule_fix_variants_shapes_determinism_and_difference() -> None:
    task = REGISTRY["lgssm_xchannel"](seed=5, sizes=SMALL_SIZES)
    train = task.train_diag[: task.diag_start]
    meta = {"L": SMOKE_CFG.input_length, "H": SMOKE_CFG.horizon, "C": train.shape[1], "seed": 5}
    X, _ = windows(task.train_diag, np.array([600, 1000, 1400]), SMOKE_CFG)

    base = m0.create_adapter()
    base.fit(train, meta)
    base_pred = base.predict_batch(X)

    for rule_id in baselines.RULE_IDS:
        first = baselines.rule_fix_variant(rule_id)
        assert first.model_id == rule_id
        first.fit(train, meta)
        pred = first.predict_batch(X)
        assert pred.shape == (3, SMOKE_CFG.horizon, train.shape[1])
        assert np.all(np.isfinite(pred))
        # Closed-form solves, no RNG: two fits on the same data agree bitwise.
        second = baselines.rule_fix_variant(rule_id)
        second.fit(train, meta)
        np.testing.assert_array_equal(pred, second.predict_batch(X))
        # A rule fix must change the model, not alias m0.
        assert not np.array_equal(pred, base_pred), rule_id


def test_run_baselines_flat_finite_results() -> None:
    results = baselines.run_baselines("lgssm_xchannel", seed=7, cfg=SMOKE_CFG, sizes=SMALL_SIZES)
    expected = {"m0.mse", "m0.gap", "m0g.mse", "m0g.gap", "m0g.delta", "m0g.remaining_gap_ratio"}
    expected |= {f"{rule}.{key}" for rule in baselines.RULE_IDS for key in ("mse", "gap")}
    assert set(results) == expected
    assert all(isinstance(value, float) for value in results.values())
    assert all(math.isfinite(value) for value in results.values())
    # A rule fix shares m0's regularized ridge machinery, so a large blow-up
    # means a broken design, not a wrong recipe (this guard caught a scale bug
    # where the AR head was trained with the level mixed into its target).
    for rule in baselines.RULE_IDS:
        assert results[f"{rule}.mse"] <= 1.5 * results["m0.mse"], rule


def test_stats_forecast_baseline_arima_beats_zero_predictor_on_ar1() -> None:
    pytest.importorskip("statsforecast")
    rng = np.random.default_rng(11)
    burn, T = 500, 2000
    eps = rng.standard_normal(burn + T)
    x = np.empty(burn + T)
    x[0] = 0.0
    for t in range(1, burn + T):
        x[t] = 0.9 * x[t - 1] + eps[t]
    values = ((x[burn:] - x[burn:].mean()) / x[burn:].std())[:, None]
    cfg = replace(SMOKE_CFG, horizon=8)
    split = Splits(
        train=slice(0, 1200), diag=slice(1200, 1400), cert=slice(1400, 1600), test=slice(1600, 2000)
    )
    out = baselines.stats_forecast_baseline(values, cfg, split, "arima")

    # The zero-predictor reference on the exact subsampled origins the
    # baseline scored (stride comes back in the result, so they cannot drift).
    origins = segment_origins(split.test.start, split.test.stop, cfg)[:: out["origin_stride"]]
    _, y = windows(values, origins, cfg)
    zero_mse = float(np.mean(np.square(y)))
    assert out["n_origins"] == origins.size
    assert out["origin_stride"] == 4
    assert len(out["mse_by_channel"]) == 1
    # ARIMA(1,0,1) tracks the conditional mean of a strong AR(1), which the
    # zero predictor cannot: a wide margin, not a coin flip.
    assert out["mse"] < 0.9 * zero_mse


def test_error_cases() -> None:
    with pytest.raises(ValueError, match="unknown rule fix"):
        baselines.rule_fix_variant("rule-bogus")

    # The cross-channel fix is inapplicable on a univariate design.
    uni = np.random.default_rng(2).standard_normal((600, 1))
    meta = {"L": SMOKE_CFG.input_length, "H": SMOKE_CFG.horizon, "C": 1, "seed": 0}
    with pytest.raises(ValueError, match="at least 2 channels"):
        baselines.rule_fix_variant("rule-crosschannel").fit(uni, meta)

    # one_shot_m0g refuses a sizes dict that does not describe the task.
    task = REGISTRY["lgssm_xchannel"](seed=0, sizes=SMALL_SIZES)
    with pytest.raises(ValueError, match="does not describe the task"):
        baselines.one_shot_m0g(
            task, m0.create_adapter(), SMOKE_CFG, {"train": 99, "diag": 99, "cert": 1, "test": 1}
        )

    # kind is validated before statsforecast is ever imported.
    values = np.zeros((200, 1))
    split = Splits(
        train=slice(0, 100), diag=slice(100, 120), cert=slice(120, 140), test=slice(140, 200)
    )
    with pytest.raises(ValueError, match="kind"):
        baselines.stats_forecast_baseline(values, SMOKE_CFG, split, "bogus")
    with pytest.raises(ValueError, match="unknown task"):
        baselines.run_baselines("no-such-task", seed=0, cfg=SMOKE_CFG, sizes=SMALL_SIZES)
