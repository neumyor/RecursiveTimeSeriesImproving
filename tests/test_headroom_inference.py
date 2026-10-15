from __future__ import annotations

import numpy as np
from scipy import stats

from harnessing_ts.headroom.inference import (
    auto_bandwidth,
    block_bootstrap_ci,
    dm_one_sided_p,
    hac_variance,
    mean_ci,
)


def test_hac_variance_matches_iid_closed_form() -> None:
    rng = np.random.default_rng(0)
    d = rng.normal(0.0, 2.0, size=20_000)
    # Var(mean) for iid is sigma^2/n; bandwidth 0 keeps only the diagonal term.
    assert abs(hac_variance(d, 0) - 4.0 / 20_000) < 4.0 / 20_000 * 0.05


def test_hac_variance_matches_ma1_closed_form() -> None:
    rng = np.random.default_rng(1)
    n = 40_000
    eps = rng.normal(0.0, 1.0, size=n + 1)
    theta = 0.6
    d = eps[1:] + theta * eps[:-1]
    expected = (1.0 + theta**2 + 2.0 * theta) / n  # gamma0 + 2*gamma1 over n
    assert abs(hac_variance(d, 3) - expected) < 0.15 * expected


def test_mean_ci_coverage_on_iid_normal() -> None:
    rng = np.random.default_rng(2)
    covered = 0
    reps = 2_000
    for _ in range(reps):
        d = rng.normal(1.0, 2.0, size=200)
        _, lo, hi = mean_ci(d, bandwidth=5, level=0.95)
        covered += lo <= 1.0 <= hi
    coverage = covered / reps
    assert 0.93 <= coverage <= 0.97, coverage


def test_dm_pvalue_under_null_is_roughly_uniform() -> None:
    rng = np.random.default_rng(3)
    rejected = 0
    reps = 1_000
    for _ in range(reps):
        d = rng.normal(0.0, 1.0, size=300)
        rejected += dm_one_sided_p(d, bandwidth=4) < 0.05
    rate = rejected / reps
    # HAC + HLN is slightly conservative at this size; keep the band wide.
    assert 0.01 <= rate <= 0.10, rate


def test_dm_pvalue_detects_real_improvement() -> None:
    rng = np.random.default_rng(4)
    d = rng.normal(0.5, 1.0, size=300)
    assert dm_one_sided_p(d, bandwidth=4) < 1e-6


def test_block_bootstrap_ci_covers_true_mean() -> None:
    rng = np.random.default_rng(5)
    covered = 0
    reps = 300
    for _ in range(reps):
        eps = rng.normal(0.0, 1.0, size=260)
        d = eps[1:] + 0.5 * eps[:-1] + 1.0
        lo, hi = block_bootstrap_ci(d, block=8, runs=200, level=0.95, rng=rng)
        covered += lo <= 1.0 <= hi
    assert 0.90 <= covered / reps <= 0.99


def test_auto_bandwidth_respects_minimum() -> None:
    assert auto_bandwidth(np.zeros(50), min_bw=23) >= 23
    # The 4*(n/100)^(2/9) rule grows very slowly; it passes 23 around n~260k.
    assert auto_bandwidth(np.zeros(1_000_000), min_bw=23) > 23
