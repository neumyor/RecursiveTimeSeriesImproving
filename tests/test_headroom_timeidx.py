from __future__ import annotations

import numpy as np
import pytest

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.timeidx import ab_split, chrono_splits, segment_origins, windows


def test_chrono_splits_bounds_and_embargo() -> None:
    cfg = HeadroomConfig(train_ratio=0.5, diag_ratio=0.2, cert_ratio=0.1)
    splits = chrono_splits(10_000, cfg, embargo=10)
    assert splits.train.stop == 5_000
    assert splits.diag.start == 5_010
    assert splits.diag.stop == 7_010
    assert splits.cert.start == 7_020
    assert splits.test.stop == 10_000


def test_chrono_splits_rejects_too_short_series() -> None:
    cfg = HeadroomConfig()
    with pytest.raises(ValueError, match="test rows"):
        chrono_splits(50, cfg, embargo=5)


def test_segment_origins_respect_burn_in_and_horizon() -> None:
    cfg = HeadroomConfig()
    origins = segment_origins(100, 2_000, cfg)
    assert origins.min() == 100 + cfg.burn_in
    assert origins.max() == 2_000 - cfg.horizon - 1
    assert np.all(np.diff(origins) == 1)


def test_segment_origins_rejects_short_segment() -> None:
    cfg = HeadroomConfig()
    with pytest.raises(ValueError, match="no usable origins"):
        segment_origins(0, cfg.burn_in + cfg.horizon, cfg)


def test_windows_content_matches_manual_slicing() -> None:
    cfg = HeadroomConfig(input_length=5, horizon=3)
    values = np.arange(80, dtype=np.float64).reshape(-1, 2)
    origins = np.array([10, 20], dtype=np.int64)
    x, y = windows(values, origins, cfg)
    assert x.shape == (2, 5, 2)
    assert y.shape == (2, 3, 2)
    np.testing.assert_array_equal(x[0], values[6:11])
    np.testing.assert_array_equal(y[0], values[11:14])


def test_windows_rejects_out_of_range_origins() -> None:
    cfg = HeadroomConfig(input_length=4, horizon=2)
    values = np.zeros((10, 1))
    with pytest.raises(ValueError, match="before the start"):
        windows(values, np.array([2]), cfg)
    with pytest.raises(ValueError, match="beyond the end"):
        windows(values, np.array([8]), cfg)


def test_ab_split_keeps_gap_between_halves() -> None:
    origins = np.arange(100, dtype=np.int64)
    a, b = ab_split(origins, a_ratio=0.5, embargo=5)
    assert a.max() + 5 < b.min()
    assert a.size + b.size + 5 == 100


def test_ab_split_rejects_impossible_cuts() -> None:
    with pytest.raises(ValueError):
        ab_split(np.arange(6), a_ratio=0.5, embargo=4)
