"""Tests for the m0 template family, loaded by file path (experiments/ is not an installed package)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_META = {"L": 96, "H": 24, "C": 3, "seed": 0}


def _load_m0():
    # experiments/ is deliberately not on the repo pythonpath (only backend/ is),
    # so the template is loaded the same way the loop will load candidate model.py files.
    spec = importlib.util.spec_from_file_location(
        "m0_template", REPO_ROOT / "experiments" / "templates" / "m0.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m0 = _load_m0()


def _ar_series(T: int = 3000, C: int = 3, seed: int = 7) -> np.ndarray:
    """Seeded per-channel AR(1) series, standardized per channel (train values are standardized by contract)."""
    rng = np.random.default_rng(seed)
    burn = 200
    phi = np.array([0.92, 0.75, 0.55])[:C]
    x = np.zeros((T + burn, C))
    eps = rng.standard_normal((T + burn, C))
    for t in range(1, T + burn):
        x[t] = phi * x[t - 1] + eps[t]
    x = x[burn:]
    return (x - x.mean(axis=0)) / x.std(axis=0)


def _windows(x: np.ndarray, origins: np.ndarray, L: int) -> np.ndarray:
    return np.stack([x[o - L + 1 : o + 1] for o in origins])


def test_predict_batch_shape() -> None:
    x = _ar_series()
    adapter = m0.variant(m0.M0_LINEAR)
    adapter.fit(x, _META)
    X = _windows(x, np.arange(1500, 1508), _META["L"])
    pred = adapter.predict_batch(X)
    assert pred.shape == (8, _META["H"], _META["C"])
    assert np.all(np.isfinite(pred))


def test_fit_is_deterministic() -> None:
    x = _ar_series()
    a1 = m0.variant(m0.M0_LINEAR)
    a2 = m0.variant(m0.M0_LINEAR)
    a1.fit(x, _META)
    a2.fit(x, _META)
    X = _windows(x, np.array([1500, 2600]), _META["L"])
    # m0 has no RNG and no hidden state, so two fits on the same data agree bitwise.
    np.testing.assert_array_equal(a1.predict_batch(X), a2.predict_batch(X))


def test_short_lookback_invariance() -> None:
    x = _ar_series()
    X = _windows(x, np.arange(1200, 1206), _META["L"])
    X2 = X.copy()
    X2[:, : _META["L"] - 24, :] += 5.0  # perturb only rows older than the last 24

    short = m0.variant(m0.M0_SHORT)
    short.fit(x, _META)
    # m0-short reads only the last 24 rows, so untouched predictions are bitwise equal.
    np.testing.assert_array_equal(short.predict_batch(X), short.predict_batch(X2))

    full = m0.variant(m0.M0_LINEAR)
    full.fit(x, _META)
    assert not np.array_equal(
        full.predict_batch(X), full.predict_batch(X2)
    ), "perturbing rows within L must change m0-linear predictions"


def test_horizon_bias_matches_linear_plus_offset() -> None:
    x = _ar_series()
    X = _windows(x, np.arange(1800, 1805), _META["L"])
    lin = m0.variant(m0.M0_LINEAR)
    bias = m0.variant(m0.M0_BIAS)
    lin.fit(x, _META)
    bias.fit(x, _META)
    diff = bias.predict_batch(X) - lin.predict_batch(X)
    h = np.arange(1, _META["H"] + 1, dtype=np.float64)
    # The two variants share the same fit (bias is applied at predict time only), so the
    # difference is exactly 0.02*h; atol=1e-12 covers the float64 rounding of the shared
    # base predictions (values are O(1) after standardization, ulp ~1e-16). The expected
    # offset is broadcast to the full shape because assert_allclose demands equal shapes.
    expected = np.broadcast_to(0.02 * h[None, :, None], diff.shape)
    np.testing.assert_allclose(diff, expected, rtol=0.0, atol=1e-12)


def test_error_cases() -> None:
    with pytest.raises(ValueError, match="unknown m0 model id"):
        m0.variant("m0-bogus")

    X = np.zeros((2, _META["L"], _META["C"]))
    with pytest.raises(RuntimeError, match="before fit"):
        m0.variant(m0.M0_LINEAR).predict_batch(X)

    x = _ar_series()
    meta_missing_seed = {"L": 96, "H": 24, "C": 3}
    with pytest.raises(ValueError, match="seed"):
        m0.variant(m0.M0_LINEAR).fit(x, meta_missing_seed)


def test_factory_and_model_ids() -> None:
    assert m0.MODEL_IDS == ("m0-linear", "m0-short-lookback", "m0-horizon-bias")
    assert m0.create_adapter().model_id == m0.M0_LINEAR
    assert m0.variant(m0.M0_SHORT).model_id == m0.M0_SHORT
    assert m0.variant(m0.M0_BIAS).model_id == m0.M0_BIAS
