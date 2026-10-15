from __future__ import annotations

"""Markdown feedback rendering for the scalar / bj / card / placebo arms.

All four arms share one byte-identical scaffold and differ only in the
middle block, so any measured behavioural difference between arms is
attributable to that block; see docs/headroom-contracts.md §9 and the
coding plan §2.5/§3.2 (arm-neutral wording: no dataset or domain
vocabulary, no status language such as "certified").
"""

from typing import Any

import numpy as np
from scipy import stats
from scipy.signal import find_peaks, periodogram

from harnessing_ts.headroom.config import ARM_VALUES, HeadroomConfig
from harnessing_ts.headroom.correctors import GainReport

# The bj diagnostics are fixed by the protocol (contract §9), not configurable:
# the whole point of the arm is a standard Box-Jenkins style diagnostic view.
_BJ_ACF_LAGS = 48
_BJ_LB_LAGS = (12, 24, 48)
_BJ_TOP_PEAKS = 5
_BJ_PER_CHANNEL_MAX = 4  # above this channel count the ACF table shows the mean
_HISTORY_ROWS = 10


def _fmt(value: Any) -> str:
    """Render one number at 4 decimals; non-finite values become 'n/a'."""
    number = float(value)
    if not np.isfinite(number):
        return "n/a"
    if number == 0.0:
        number = 0.0  # collapse -0.0 so the appendix never shows "-0.0000"
    return f"{number:.4f}"


def acf(e: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation per channel, rows = lags 1..max_lag, as [max_lag, C].

    Uses the standard biased estimator rho_k = c_k / c_0 with
    c_k = sum_t (x_t - x_bar)(x_{t-k} - x_bar): |rho| <= 1 by construction,
    which is what the Ljung-Box formula below expects.  A constant channel
    has no autocorrelation to report (NaN), not a zero one.
    """
    e = np.asarray(e, dtype=np.float64)
    if e.ndim != 2:
        raise ValueError(f"e must be [T, C], got shape {e.shape}")
    n, n_channels = e.shape
    out = np.full((max_lag, n_channels), np.nan)
    if n < 2:
        return out
    centered = e - e.mean(axis=0)
    c0 = (centered * centered).sum(axis=0)
    for k in range(1, min(max_lag, n - 1) + 1):
        c_k = (centered[k:] * centered[:-k]).sum(axis=0)
        np.divide(c_k, c0, out=out[k - 1], where=c0 > 0)
    return out


def ljung_box(rho: np.ndarray, n: int, lags: tuple[int, ...]) -> dict[int, tuple[float, float]]:
    """Ljung-Box Q and chi-square p at each requested lag for a 1-D ACF.

    Q(m) = n(n+2) sum_{k=1..m} rho_k^2 / (n-k) with p = chi2.sf(Q, m),
    implemented locally because the plain formula is the contract.
    """
    rho = np.asarray(rho, dtype=np.float64).ravel()
    result: dict[int, tuple[float, float]] = {}
    for m in lags:
        if m > rho.size or m >= n or not np.isfinite(rho[:m]).all():
            result[m] = (float("nan"), float("nan"))
            continue
        q = n * (n + 2.0) * float(np.sum(rho[:m] ** 2 / (n - np.arange(1, m + 1))))
        result[m] = (q, float(stats.chi2.sf(q, m)))
    return result


def min_p_ljung_box(
    rho: np.ndarray, n: int, lags: tuple[int, ...]
) -> dict[int, tuple[float, float]]:
    """Per-channel Ljung-Box, reduced at each lag to the smallest-p channel.

    The portmanteau is a single-series test, and the cross-channel mean ACF
    is no series' ACF: sign cancellation can push its Q toward zero while
    every channel is autocorrelated, which would hand the bj agent a weaker
    diagnostic than the card arm's per-coordinate view.  Reporting the
    strongest per-channel evidence at each lag names no channel and ranks
    nothing; at C = 1 it reduces to the textbook test.
    """
    rho = np.asarray(rho, dtype=np.float64)
    per_channel = [ljung_box(rho[:, channel], n, lags) for channel in range(rho.shape[1])]
    worst: dict[int, tuple[float, float]] = {}
    for m in lags:
        # A channel with a NaN ACF (constant) or too few rows has no vote.
        finite = [per_channel[c][m] for c in range(rho.shape[1]) if np.isfinite(per_channel[c][m][1])]
        worst[m] = min(finite, key=lambda pair: pair[1]) if finite else (float("nan"), float("nan"))
    return worst


def periodogram_peaks(e: np.ndarray, top: int) -> list[tuple[float, float]]:
    """Up to ``top`` spectral peaks of the channel-averaged periodogram.

    Returns (period, power) rows in ascending period order.  Selecting the
    largest peaks is inherent to "top", but they are deliberately not ranked
    by power for the agent: the bj arm must not order anything by importance.
    """
    e = np.asarray(e, dtype=np.float64)
    if e.shape[0] < 4:
        return []
    # Center each channel here so the DC bin is exactly zero and cannot
    # dominate the peak list; detrend is disabled to keep that explicit.
    freqs, power = periodogram(e - e.mean(axis=0), axis=0, fs=1.0, detrend=False)
    mean_power = power.mean(axis=1)
    peaks, _ = find_peaks(mean_power)
    selected = peaks[np.argsort(mean_power[peaks])[-top:]]
    rows = [(float(1.0 / freqs[k]), float(mean_power[k])) for k in selected]
    return sorted(rows, key=lambda row: row[0])


def render_feedback(
    arm: str,
    incumbent: dict[str, Any],
    gains: list[GainReport],
    history: list[dict[str, Any]],
    cfg: HeadroomConfig,
    placebo_perm: tuple[int, ...] | None = None,
) -> str:
    """Render one round of agent feedback as markdown (contract §9).

    The scaffold (title with round and remaining budget, incumbent diagnostic
    MSE, per-horizon-group MSE, candidate history) is byte-identical across
    arms; only the middle block differs.  ``scalar`` gets no middle block,
    ``bj`` gets residual diagnostics, ``card``/``placebo`` get the defect
    card (the card wording is "remaining improvement (lower bound)", never
    "headroom").

    Card ordering: coordinate rows are sorted by the ``lcb`` values the
    caller passed.  This function never computes or adjusts intervals; the
    caller must compute the card gains at the Bonferroni-adjusted level
    ``correctors.card_level(cfg, period_peaks, n_channels)`` (the A-M4
    winner's-curse adjustment) and pass the resulting reports here.  NaN lcb
    values sort last; ties keep the caller's order.  The ``total`` gain
    becomes the aggregate row and ``mu2`` the level row; hgroup gains are
    certification statistics and are excluded.

    Placebo: exactly the card layout with the coordinate rows permuted —
    each row keeps its own numbers, only the order changes.  With
    ``placebo_perm=None`` the permutation is
    ``np.random.default_rng(cfg.seed).permutation(len(rows))``, i.e. fixed
    per (seed, row count) and independent of global RNG state.
    """
    if arm not in ARM_VALUES:
        raise ValueError(f"arm must be one of {ARM_VALUES}, got {arm!r}")
    blocks = [_scaffold_head(incumbent), _scaffold_groups(incumbent)]
    if arm == "bj":
        blocks.append(_bj_block(incumbent))
    elif arm in ("card", "placebo"):
        rows, total = _split_gains(gains)
        ordered = _sorted_by_lcb(rows)
        if arm == "placebo":
            perm = (
                tuple(placebo_perm)
                if placebo_perm is not None
                else _default_perm(len(ordered), cfg)
            )
        else:
            perm = None
        blocks.append(_card_table(ordered, total, perm))
    blocks.append(_scaffold_history(history))
    return "\n\n".join(blocks) + "\n"


def _scaffold_head(incumbent: dict[str, Any]) -> str:
    """Title line (round + remaining budget) and the incumbent diagnostic MSE."""
    return (
        f"# Improvement round {int(incumbent['round'])} — "
        f"{int(incumbent['budgetLeft'])} candidates remaining\n\n"
        f"Incumbent diagnostic MSE: {_fmt(incumbent['diagMse'])}"
    )


def _scaffold_groups(incumbent: dict[str, Any]) -> str:
    """MSE per horizon group; dict order is the caller's display order."""
    lines = ["## MSE by horizon group", "", "| group | mse |", "| --- | --- |"]
    for group, mse in incumbent["mseByHGroup"].items():
        lines.append(f"| {group} | {_fmt(mse)} |")
    return "\n".join(lines)


def _scaffold_history(history: list[dict[str, Any]]) -> str:
    """Candidate history table, capped to the last 10 entries."""
    lines = [
        "## Candidate history",
        "",
        "| round | candidate | accepted | diagnostic mse |",
        "| --- | --- | --- | --- |",
    ]
    for row in history[-_HISTORY_ROWS:]:
        accepted = "yes" if row["accepted"] else "no"
        lines.append(
            f"| {int(row['round'])} | {row['candidateId']} | {accepted} | {_fmt(row['diagMse'])} |"
        )
    return "\n".join(lines)


def _bj_block(incumbent: dict[str, Any]) -> str:
    """Raw statistical diagnostics of the 1-step residual series.

    No gains, no importance ordering, no status statements: the bj arm must
    make the agent do the inference itself.
    """
    e = np.asarray(incumbent["eFull"], dtype=np.float64)
    if e.ndim != 2:
        raise ValueError(f"incumbent['eFull'] must be [T, C], got shape {e.shape}")
    # e_full is defined only at origins; dropping the NaN rows in between
    # leaves the contiguous realized 1-step residual series.
    e = e[~np.isnan(e).any(axis=1)]
    rho = acf(e, _BJ_ACF_LAGS)

    if e.shape[1] <= _BJ_PER_CHANNEL_MAX:
        headers = [f"ch{c}" for c in range(e.shape[1])]
        acf_rows = [
            "| lag | " + " | ".join(headers) + " |",
            "| --- |" + " --- |" * len(headers),
        ]
        acf_rows += [
            f"| {k} | " + " | ".join(_fmt(v) for v in rho[k - 1]) + " |"
            for k in range(1, _BJ_ACF_LAGS + 1)
        ]
    else:
        mean_rho = rho.mean(axis=1)
        acf_rows = ["| lag | mean |", "| --- | --- |"]
        acf_rows += [
            f"| {k} | {_fmt(mean_rho[k - 1])} |" for k in range(1, _BJ_ACF_LAGS + 1)
        ]

    lb = min_p_ljung_box(rho, e.shape[0], _BJ_LB_LAGS)
    lb_rows = [
        "| lag | Q (min p over channels) | p (min over channels) |",
        "| --- | --- | --- |",
    ]
    lb_rows += [f"| {m} | {_fmt(q)} | {_fmt(p)} |" for m, (q, p) in lb.items()]

    peak_rows = ["| period (steps) | power |", "| --- | --- |"]
    peak_rows += [
        f"| {_fmt(period)} | {_fmt(power)} |" for period, power in periodogram_peaks(e, _BJ_TOP_PEAKS)
    ]

    sections = [
        "## 1-step residual autocorrelation\n\n" + "\n".join(acf_rows),
        "## Ljung-Box portmanteau test\n\n" + "\n".join(lb_rows),
        "## Top periodogram peaks\n\n" + "\n".join(peak_rows),
    ]
    return "\n\n".join(sections)


def _split_gains(gains: list[GainReport]) -> tuple[list[GainReport], GainReport | None]:
    """Card rows (all coordinates) plus the total report.

    hgroup gains are excluded: the contract reserves them for certification,
    not for the coordinate table the card renders.
    """
    rows: list[GainReport] = []
    total: GainReport | None = None
    for gain in gains:
        kind = gain.coordinate.kind
        if kind == "total":
            total = gain
        elif kind != "hgroup":
            rows.append(gain)
    return rows, total


def _sorted_by_lcb(rows: list[GainReport]) -> list[GainReport]:
    """Descending by the caller's lcb (computed at correctors.card_level upstream)."""
    return sorted(
        rows,
        key=lambda gain: float(gain.lcb) if np.isfinite(gain.lcb) else -np.inf,
        reverse=True,
    )


def _default_perm(n: int, cfg: HeadroomConfig) -> tuple[int, ...]:
    """Deterministic permutation from cfg.seed alone; no global RNG state."""
    return tuple(int(i) for i in np.random.default_rng(cfg.seed).permutation(n))


def _card_table(
    ordered: list[GainReport], total: GainReport | None, perm: tuple[int, ...] | None
) -> str:
    """Defect card: coordinate rows, then the aggregate total, then mu2."""
    if perm is not None:
        if sorted(perm) != list(range(len(ordered))):
            raise ValueError(
                f"placebo_perm must be a permutation of 0..{len(ordered) - 1}, got {tuple(perm)}"
            )
        ordered = [ordered[i] for i in perm]
    lines = [
        "## Remaining improvement by coordinate",
        "",
        "| aspect | coordinate | delta | lower bound | strongest at h |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [_card_row(gain) for gain in ordered]
    if total is not None:
        # The total is not one defect candidate but the aggregate bound, so
        # it is reported after the coordinates rather than ranked among them.
        lines.append(_card_row(total))
        # mu2 = (mean residual)^2 is the realized gain of a constant level
        # fix, so it fills the delta column; no interval is available for it.
        lines.append(f"| level-bias | mu2 | {_fmt(total.mu2)} | n/a | n/a |")
    return "\n".join(lines)


def _card_row(gain: GainReport) -> str:
    window = _peak_range(gain.by_horizon)
    strongest = "n/a" if window is None else f"h in [{window[0]}, {window[1]}]"
    return (
        f"| {gain.coordinate.aspect} | {gain.coordinate.label} | "
        f"{_fmt(gain.delta)} | {_fmt(gain.lcb)} | {strongest} |"
    )


def _peak_range(by_horizon: np.ndarray) -> tuple[int, int] | None:
    """1-based inclusive horizon band around the argmax of ``by_horizon``.

    The band widens while the per-horizon gain stays above half its maximum
    (an FWHM-style window).  NaN entries act as barriers: an unknown value
    cannot be claimed as part of the band.  None means "nothing to say".
    """
    values = np.asarray(by_horizon, dtype=np.float64)
    if values.size == 0:
        return None
    masked = np.where(np.isfinite(values), values, -np.inf)
    top = int(np.argmax(masked))
    if not np.isfinite(masked[top]):
        return None
    threshold = 0.5 * masked[top]
    lo = hi = top
    while lo > 0 and masked[lo - 1] >= threshold:
        lo -= 1
    while hi + 1 < values.size and masked[hi + 1] >= threshold:
        hi += 1
    return lo + 1, hi + 1
