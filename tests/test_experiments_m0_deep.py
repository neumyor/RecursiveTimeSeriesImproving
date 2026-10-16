"""Tests for the m0-patchtst template, loaded by file path (experiments/ is not an installed package)."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
_SMALL_META = {"L": 48, "H": 12, "C": 3, "seed": 0}
_BIG_META = {"L": 96, "H": 24, "C": 4, "seed": 0}
# (L - patch_len) // stride + 1 at L=48: what the layers probe must dump.
_N_PATCHES_SMALL = (48 - 16) // 8 + 1
# The calibration budget from the m0_deep spec: fit on train[:20000] x C=4
# must finish in 240s on this machine (the loop's own candidate budget is
# looser; this is the tighter guard on the incumbent's fit cost).
_FIT_BUDGET_SECONDS = 240.0


def _load_m0_deep():
    # experiments/ is deliberately not on the repo pythonpath (only backend/ is),
    # so the template is loaded the same way the loop loads candidate model.py files.
    spec = importlib.util.spec_from_file_location(
        "m0_deep_template", REPO_ROOT / "experiments" / "templates" / "m0_deep.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m0_deep = _load_m0_deep()


def _ar_series(T: int = 3000, C: int = 3, seed: int = 7) -> np.ndarray:
    """Seeded per-channel AR(1) series, standardized per channel (train values are standardized by contract)."""
    rng = np.random.default_rng(seed)
    burn = 200
    phi = np.array([0.92, 0.75, 0.55, 0.85])[:C]
    x = np.zeros((T + burn, C))
    eps = rng.standard_normal((T + burn, C))
    for t in range(1, T + burn):
        x[t] = phi * x[t - 1] + eps[t]
    x = x[burn:]
    return (x - x.mean(axis=0)) / x.std(axis=0)


def _seasonal_series(
    T: int = 4000, C: int = 3, period: int = 24, seed: int = 11, noise: float = 0.1
) -> np.ndarray:
    """Seeded per-channel sinusoid (period 24, so L=48 holds two full cycles) plus light noise, standardized."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    x = np.empty((T, C))
    for c in range(C):
        x[:, c] = np.sin(2 * np.pi * t / period + 2 * np.pi * c / C) + noise * rng.standard_normal(T)
    return (x - x.mean(axis=0)) / x.std(axis=0)


def _windows(x: np.ndarray, origins: np.ndarray, L: int) -> np.ndarray:
    return np.stack([x[o - L + 1 : o + 1] for o in origins])


def test_shapes_determinism_and_row_independence() -> None:
    x = _ar_series()
    a1 = m0_deep.create_adapter()
    a1.fit(x, _SMALL_META)
    a2 = m0_deep.create_adapter()
    a2.fit(x, _SMALL_META)
    assert a1.model_id == "m0-patchtst"

    X = _windows(x, np.arange(1500, 1508), _SMALL_META["L"])
    p1 = a1.predict_batch(X)
    p2 = a2.predict_batch(X)
    assert p1.shape == (8, _SMALL_META["H"], _SMALL_META["C"])
    assert p1.dtype == np.float64
    assert np.all(np.isfinite(p1))

    # Determinism gate (same process, same (train, meta)): the loop compares
    # two fits at rtol 1e-5; cross-process bitwise equality is not required.
    np.testing.assert_allclose(p1, p2, rtol=1e-5, atol=0.0)

    # Row independence, exactly the worker's cheat probe (candidate_worker):
    # re-predict single rows and compare with np.allclose(atol=1e-9), which
    # keeps numpy's default rtol=1e-5.  The float64 predict path holds
    # single-vs-batch BLAS rounding near 1e-15, far under that bar.
    for i in (0, 3, 7):
        single = a1.predict_batch(X[i : i + 1])
        assert np.allclose(single[0], p1[i], atol=1e-9), f"row {i} differs from its single-row prediction"


def test_return_layers_shapes_and_pred_equality() -> None:
    x = _ar_series()
    adapter = m0_deep.create_adapter()
    adapter.fit(x, _SMALL_META)
    X = _windows(x, np.arange(1600, 1604), _SMALL_META["L"])

    pred_plain = adapter.predict_batch(X)
    result = adapter.predict_batch(X, return_layers=True)
    assert isinstance(result, tuple) and len(result) == 2
    pred, info = result
    layers = info["layers"]

    # Patch embedding + one entry per transformer block (2 blocks).
    assert [name for name, _ in layers] == ["patch_embed", "block0", "block1"]
    for name, arr in layers:
        assert isinstance(arr, np.ndarray), f"{name} activation must be numpy"
        # Channel-independent copies averaged back to one tensor per window row.
        assert arr.shape == (4, _N_PATCHES_SMALL, 64), f"{name} has shape {arr.shape}"
        assert arr.dtype == np.float32
        assert np.all(np.isfinite(arr))

    # The probe only appends to a list inside the same forward, so the
    # predictions are bitwise identical to the default path.
    np.testing.assert_array_equal(pred, pred_plain)


def test_seasonal_beats_zero_predictor_on_train_windows() -> None:
    x = _seasonal_series()
    adapter = m0_deep.create_adapter()
    adapter.fit(x, _SMALL_META)
    L, H = _SMALL_META["L"], _SMALL_META["H"]
    origins = np.arange(L - 1, x.shape[0] - H, 4)[:500]
    X = _windows(x, origins, L)
    y = np.stack([x[o + 1 : o + 1 + H] for o in origins])
    pred = np.concatenate(
        [adapter.predict_batch(X[i : i + 256]) for i in range(0, origins.size, 256)]
    )
    mse = float(np.mean((pred - y) ** 2))
    var = float(y.var())
    # On standardized data the zero predictor scores var(y); a model that
    # learned anything about the period-24 structure lands clearly below.
    assert mse < 0.9 * var, f"train-window MSE {mse:.4f} not below 0.9*var(y) = {0.9 * var:.4f}"


def test_fit_meta_validation_and_predict_before_fit() -> None:
    x = _ar_series(T=600)
    adapter = m0_deep.create_adapter()
    X = np.zeros((2, _SMALL_META["L"], _SMALL_META["C"]))
    with pytest.raises(RuntimeError, match="before fit"):
        adapter.predict_batch(X)

    with pytest.raises(ValueError, match=r"missing required keys: \['seed'\]"):
        adapter.fit(x, {"L": 48, "H": 12, "C": 3})
    with pytest.raises(ValueError, match=r"missing required keys: \['L', 'seed'\]"):
        adapter.fit(x, {"H": 12, "C": 3})


def test_no_batchnorm_and_predict_forces_eval() -> None:
    x = _ar_series()
    adapter = m0_deep.create_adapter()
    adapter.fit(x, _SMALL_META)
    # The tests own this private handle: the adapter contract exposes nothing
    # else for walking modules.
    model = adapter._model
    assert model is not None

    # BatchNorm couples rows through batch statistics (the worker's
    # single-vs-batch probe would flag it as a cheat); dropout in train mode
    # would make predict nondeterministic.  The template has neither.
    assert not any(
        isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
        for m in model.modules()
    )
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())

    X = _windows(x, np.arange(1700, 1704), _SMALL_META["L"])
    expected = adapter.predict_batch(X)
    # Even if a caller leaves the model in train mode, predict_batch forces
    # eval() itself (the documented choice in m0_deep.py), so the output is
    # unchanged and the model is left in eval mode.
    model.train()
    forced = adapter.predict_batch(X)
    np.testing.assert_array_equal(forced, expected)
    assert not model.training


def test_fit_time_calibration_20000x4() -> None:
    """Last test on purpose: the file's one heavyweight fit.

    Guards the calibrated _EPOCHS choice against machine slowdowns and
    accidental architecture growth: fit on train[:20000] x C=4 must stay
    inside the 240s calibration budget (the measured value at the shipped
    config is recorded in m0_deep.py's docstring).
    """
    x = _ar_series(T=20000, C=4)
    adapter = m0_deep.create_adapter()
    start = time.perf_counter()
    adapter.fit(x, _BIG_META)
    seconds = time.perf_counter() - start
    X = _windows(x, np.arange(15000, 15004), _BIG_META["L"])
    assert adapter.predict_batch(X).shape == (4, _BIG_META["H"], _BIG_META["C"])
    assert seconds <= _FIT_BUDGET_SECONDS, (
        f"fit on 20000x4 (L=96, H=24) took {seconds:.1f}s, over the "
        f"{_FIT_BUDGET_SECONDS:.0f}s calibration budget - reduce m0_deep._EPOCHS or shrink the architecture"
    )
