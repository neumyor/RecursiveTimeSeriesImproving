from __future__ import annotations

"""HAC inference on per-origin loss differences: DM tests, CIs, bootstrap.

All functions treat their input as the per-origin series
``d_o = mean_{h,c}(r^2 - (r - g)^2)`` (positive = improvement).  Overlapping
multi-step windows make ``d`` serially correlated, so every interval and
p-value uses Newey--West HAC variances with the Harvey--Leybourne--Newbold
small-sample correction; see docs/headroom-contracts.md §6.
"""

import numpy as np
from scipy import stats


def hac_variance(d: np.ndarray, bandwidth: int) -> float:
    """Newey--West (Bartlett) estimate of the variance of the mean of ``d``."""
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}")
    if bandwidth < 0 or bandwidth >= n:
        raise ValueError(f"bandwidth must be in [0, {n - 1}], got {bandwidth}")
    dm = d - d.mean()
    total = float(dm @ dm)
    for k in range(1, bandwidth + 1):
        weight = 1.0 - k / (bandwidth + 1.0)
        total += 2.0 * weight * float(dm[k:] @ dm[:-k])
    return total / (n * n)


def auto_bandwidth(d: np.ndarray, min_bw: int) -> int:
    """Rule-of-thumb bandwidth, never below ``min_bw`` (callers pass >= H-1)."""
    n = np.asarray(d).size
    if n < 2:
        raise ValueError(f"need at least 2 observations, got {n}")
    return max(int(min_bw), int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def _hln_factor(n: int, bandwidth: int) -> float:
    """Harvey--Leybourne--Newbold correction multiplying the t-statistic.

    The corrected statistic is t * sqrt((n + 1 - 2K + K(K-1)/n) / n), which
    is slightly below 1 for typical n >> K and makes the test conservative.
    """
    return float(np.sqrt((n + 1.0 - 2.0 * bandwidth + bandwidth * (bandwidth - 1.0) / n) / n))


def mean_ci(d: np.ndarray, bandwidth: int, level: float) -> tuple[float, float, float]:
    """(mean, lcb, ucb) with HAC standard error, t(n-1) critical value, HLN widening."""
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    mean = float(d.mean())
    se = float(np.sqrt(max(hac_variance(d, bandwidth), np.finfo(float).tiny)))
    crit = float(stats.t.ppf(0.5 + level / 2.0, df=n - 1))
    half = crit * se / _hln_factor(n, bandwidth)
    return mean, mean - half, mean + half


def dm_one_sided_p(d: np.ndarray, bandwidth: int) -> float:
    """One-sided p-value for H0: mean(d) <= 0 (no improvement).

    Positive means are evidence of improvement; the HLN-corrected statistic is
    compared against t(n-1).
    """
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    if n < 3:
        raise ValueError(f"need at least 3 observations for a DM test, got {n}")
    mean = float(d.mean())
    variance = hac_variance(d, bandwidth)
    if variance <= 0.0:
        # Degenerate constant series: an improvement is certain (or absent).
        return 0.0 if mean > 0 else 1.0
    se = float(np.sqrt(variance))
    t_stat = mean / se * _hln_factor(n, bandwidth)
    return float(stats.t.sf(t_stat, df=n - 1))
