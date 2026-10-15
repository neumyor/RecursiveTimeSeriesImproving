from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from scipy.signal import lfilter

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.feedback import (
    acf,
    ljung_box,
    min_p_ljung_box,
    periodogram_peaks,
    render_feedback,
)

HORIZON = 24
ARMS = ("scalar", "bj", "card", "placebo")
# Contract/plan §3.2 forbidden vocabulary: dataset names, domain words, and
# status language that could prime or certify any arm.
FORBIDDEN_SUBSTRINGS = (
    "ecg",
    "ett",
    "electricity",
    "traffic",
    "weather",
    "exchange",
    "headroom",
    "certified",
    "certificate",
    "saturated",
)


@dataclass(frozen=True)
class _Coordinate:
    """Stand-in for correctors.Coordinate; feedback duck-types at runtime."""

    kind: str
    label: str
    aspect: str
    spec: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Gain:
    """Stand-in for correctors.GainReport; the renderer duck-types at runtime."""

    coordinate: _Coordinate
    delta: float
    lcb: float
    ucb: float
    by_horizon: np.ndarray
    mu2: float
    n_eval: int


def _make_gains() -> list[_Gain]:
    def by_horizon(delta: float, peak: int) -> np.ndarray:
        # Tent profile over horizons; only the peak position matters here.
        h = np.arange(1, HORIZON + 1, dtype=np.float64)
        return np.maximum(0.0, 1.0 - np.abs(h - peak) / 6.0) * delta

    specs = [
        ("band", "band(P=24)", "periodicity", 0.034, 0.026, 2),
        ("lag", "lag[0,23]", "lag-structure", 0.021, 0.015, 1),
        ("beyond_l", "beyond_l", "long-range-beyond-window", 0.015, 0.009, 20),
        ("channel", "channel->3", "cross-channel", 0.006, 0.002, 5),
        ("horizon", "horizon", "horizon-range", 0.004, 0.001, 12),
    ]
    gains = [
        _Gain(_Coordinate(kind, label, aspect), delta, lcb, lcb + 0.004, by_horizon(delta, peak), 0.0004, 500)
        for kind, label, aspect, delta, lcb, peak in specs
    ]
    gains.append(
        _Gain(_Coordinate("total", "total", "other"), 0.061, 0.047, 0.075, by_horizon(0.061, 4), 0.0004, 500)
    )
    # hgroup gains exist for certification; the card must not render them.
    gains.append(
        _Gain(
            _Coordinate("hgroup", "hgroup[1,12]", "horizon-range"),
            0.010,
            0.005,
            0.014,
            by_horizon(0.010, 6),
            0.0004,
            500,
        )
    )
    return gains


def _make_efull(rows: int = 600, channels: int = 2, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, size=(rows, channels))


def _make_incumbent(e_full: np.ndarray | None = None) -> dict[str, Any]:
    return {
        "diagMse": 0.4123,
        "mseByHGroup": {"hgroup[1,12]": 0.4012, "hgroup[13,24]": 0.4234},
        "round": 3,
        "budgetLeft": 17,
        "eFull": _make_efull() if e_full is None else e_full,
    }


def _make_history(entries: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "round": i + 1,
            "candidateId": f"c{i + 1:02d}",
            "accepted": i % 2 == 0,
            "diagMse": 0.4300 + 0.001 * i,
        }
        for i in range(entries)
    ]


def _card_rows(text: str) -> list[str]:
    """Data rows of the defect-card table, in rendered order."""
    section = text.split("## Remaining improvement by coordinate", 1)[1]
    section = section  # the shared scaffold no longer renders history (round.md owns it)
    return [
        line
        for line in section.splitlines()
        if line.startswith("| ") and not line.startswith("| ---") and not line.startswith("| aspect")
    ]


def test_scalar_is_exactly_the_scaffold() -> None:
    out = render_feedback(
        "scalar", _make_incumbent(_make_efull()), _make_gains(), _make_history(2), HeadroomConfig()
    )
    assert out.startswith("# Improvement round 3 — 17 candidates remaining\n\n")
    assert "Incumbent diagnostic MSE: 0.4123" in out
    assert "| group | mse |" in out
    assert "| hgroup[1,12] | 0.4012 |" in out
    assert "| hgroup[13,24] | 0.4234 |" in out
    # No middle block of any other arm leaks in.
    assert "autocorrelation" not in out
    assert "aspect" not in out
    assert "Remaining improvement" not in out


def test_all_arms_share_the_identical_scaffold() -> None:
    incumbent = _make_incumbent(_make_efull())
    history = _make_history(3)
    gains = _make_gains()
    scalar = render_feedback("scalar", incumbent, gains, history, HeadroomConfig())
    # The scaffold pieces, extracted from the scalar render: the header and
    # the horizon-group table must appear verbatim in every arm.
    head, sep, rest = scalar.partition("## MSE by horizon group")
    assert sep
    chunks = (head, "## MSE by horizon group" + rest)
    for arm in ARMS:
        out = render_feedback(arm, incumbent, gains, history, HeadroomConfig())
        for chunk in chunks:
            assert chunk in out, arm
        # scalar is the only arm whose output is the bare scaffold.
        assert (out == scalar) == (arm == "scalar")


def test_bj_reports_diagnostics_without_any_gain_numbers() -> None:
    incumbent = _make_incumbent(_make_efull(rows=2000, seed=21))
    # Magnitudes an ACF (bounded by 1) can never reproduce, so their absence
    # from the rendered text proves no gain numbers leaked.
    gains = [
        _Gain(_Coordinate("band", "band(P=24)", "periodicity"), 88.8888, 77.7777, 90.0, np.zeros(HORIZON), 66.6666, 500),
        _Gain(_Coordinate("total", "total", "other"), 99.9999, 88.8888, 101.0, np.zeros(HORIZON), 66.6666, 500),
    ]
    out = render_feedback("bj", incumbent, gains, _make_history(2), HeadroomConfig())
    for needle in (
        "88.8888",
        "77.7777",
        "99.9999",
        "66.6666",
        "band(P=24)",
        "periodicity",
        "delta",
        "lower bound",
        "aspect",
    ):
        assert needle not in out, needle
    for heading in (
        "## 1-step residual autocorrelation",
        "## Ljung-Box portmanteau test",
        "## Top periodogram peaks",
    ):
        assert heading in out


def test_bj_ljung_box_row_shows_min_p_across_channels() -> None:
    # One strongly autocorrelated channel, one white channel: the old mean-ACF
    # portmanteau diluted the signal (and can cancel it outright across
    # sign-opposed channels), weakening the bj arm relative to card. The
    # per-channel minimum p must show a tiny p at lag 24.
    n = 2000
    rng = np.random.default_rng(33)
    ar1 = lfilter([1.0], [1.0, -0.9], rng.normal(0.0, 1.0, size=n))
    e = np.column_stack([ar1, rng.normal(0.0, 1.0, size=n)])
    gains = [
        _Gain(
            _Coordinate("band", "band(P=24)", "periodicity"),
            88.8888,
            77.7777,
            90.0,
            np.zeros(HORIZON),
            66.6666,
            500,
        )
    ]
    out = render_feedback("bj", _make_incumbent(e), gains, _make_history(1), HeadroomConfig())
    assert "Q (min p over channels)" in out
    # The lag-24 row is the AR(1) channel's own (Q, p) — the minimum-p
    # channel — and its p is tiny (renders as 0.0000).
    rho = acf(e, 48)
    q24, p24 = ljung_box(rho[:, 0], n, (24,))[24]
    assert p24 < 1e-10
    assert f"| 24 | {q24:.4f} | {p24:.4f} |" in out
    # The bj arm still renders no gain numbers and no coordinate labels.
    assert "88.8888" not in out
    assert "band(P=24)" not in out


def test_card_orders_coordinates_by_lcb_descending() -> None:
    out = render_feedback("card", _make_incumbent(), _make_gains(), _make_history(1), HeadroomConfig())
    section = out.split("## Remaining improvement by coordinate", 1)[1]
    expected = ["band(P=24)", "lag[0,23]", "beyond_l", "channel->3", "horizon"]
    positions = [section.index(f"| {label} |") for label in expected]
    assert positions == sorted(positions)
    # The aggregate total and mu2 rows come after every coordinate row.
    assert max(positions) < section.index("| other | total | 0.0610 |")
    assert section.index("| other | total | 0.0610 |") < section.index(
        "| level-bias | mu2 | 0.0004 | n/a | n/a |"
    )
    # hgroup gains feed certification only: the scaffold MSE table is the
    # sole place the group label appears.
    assert out.count("hgroup[1,12]") == 1


def test_placebo_renders_the_same_rows_in_a_permuted_order() -> None:
    cfg = HeadroomConfig()
    incumbent = _make_incumbent()
    gains = _make_gains()
    history = _make_history(1)
    card_rows = _card_rows(render_feedback("card", incumbent, gains, history, cfg))
    placebo_rows = _card_rows(render_feedback("placebo", incumbent, gains, history, cfg))
    assert sorted(placebo_rows) == sorted(card_rows)  # same multiset of rows
    assert placebo_rows != card_rows  # seed-0 default permutation moves them
    assert placebo_rows[-2:] == card_rows[-2:]  # total and mu2 stay pinned at the end


def test_explicit_placebo_perm_is_honored() -> None:
    cfg = HeadroomConfig()
    incumbent = _make_incumbent()
    gains = _make_gains()
    history = _make_history(1)
    card_rows = _card_rows(render_feedback("card", incumbent, gains, history, cfg))
    perm = (4, 2, 0, 3, 1)
    placebo_rows = _card_rows(
        render_feedback("placebo", incumbent, gains, history, cfg, placebo_perm=perm)
    )
    assert placebo_rows[:-2] == [card_rows[i] for i in perm]
    assert placebo_rows[-2:] == card_rows[-2:]


def test_invalid_placebo_perm_is_rejected() -> None:
    incumbent = _make_incumbent()
    gains = _make_gains()
    history = _make_history(1)
    for bad in ((0, 0, 1, 2, 3), (0, 1, 2), (0, 1, 2, 3, 5)):
        with pytest.raises(ValueError, match="permutation"):
            render_feedback(
                "placebo", incumbent, gains, history, HeadroomConfig(), placebo_perm=bad
            )


def test_unknown_arm_is_rejected() -> None:
    with pytest.raises(ValueError, match="arm"):
        render_feedback("bogus", _make_incumbent(), [], _make_history(1), HeadroomConfig())


def test_no_forbidden_words_in_any_arm() -> None:
    incumbent = _make_incumbent(_make_efull())
    gains = _make_gains()
    history = _make_history(2)
    for arm in ARMS:
        lowered = render_feedback(arm, incumbent, gains, history, HeadroomConfig()).lower()
        for word in FORBIDDEN_SUBSTRINGS:
            assert word not in lowered, (arm, word)


def test_history_table_is_owned_by_round_md() -> None:
    """round.md renders the candidate history; the feedback scaffold no longer
    duplicates it (one table in the prompt, byte-identical across arms)."""
    out = render_feedback("card", _make_incumbent(), _make_gains(), _make_history(15), HeadroomConfig())
    assert "| 15 | c15 |" not in out
    assert "## Candidate history" not in out



def test_ljung_box_white_noise_keeps_p_above_threshold() -> None:
    rng = np.random.default_rng(11)
    e = rng.normal(0.0, 1.0, size=(5000, 1))
    rho = acf(e, 48)[:, 0]
    result = ljung_box(rho, 5000, (12, 24, 48))
    # Under whiteness each p is ~U(0,1); with this fixed seed most stay above 0.01.
    assert sum(1 for _, p in result.values() if p > 0.01) >= 2


def test_ljung_box_ar1_rejects_whiteness() -> None:
    n = 5000
    eps = np.random.default_rng(12).normal(0.0, 1.0, size=n)
    e = lfilter([1.0], [1.0, -0.9], eps)  # x_t = 0.9 x_{t-1} + eps_t
    rho = acf(e[:, None], 48)[:, 0]
    assert abs(rho[0] - 0.9) < 0.02  # ACF sanity: lag-1 autocorrelation is phi
    _, p = ljung_box(rho, n, (24,))[24]
    assert p < 1e-10


def test_min_p_ljung_box_reports_the_autocorrelated_channel() -> None:
    # Channel 0 is a strong AR(1), channel 1 is white noise. The cross-channel
    # mean ACF is no series' ACF — here it halves channel 0's signal, and with
    # opposite-signed channels it cancels outright — so the portmanteau must
    # run per channel; the minimum p then exposes the AR(1) at every lag.
    n = 2000
    rng = np.random.default_rng(31)
    ar1 = lfilter([1.0], [1.0, -0.9], rng.normal(0.0, 1.0, size=n))
    e = np.column_stack([ar1, rng.normal(0.0, 1.0, size=n)])
    rho = acf(e, 48)
    worst = min_p_ljung_box(rho, n, (12, 24, 48))
    for m, (q, p) in worst.items():
        q0, p0 = ljung_box(rho[:, 0], n, (m,))[m]
        assert (q, p) == (q0, p0)  # the AR(1) channel is the minimum-p channel
        assert p < 1e-10


def test_min_p_ljung_box_skips_channels_with_no_acf() -> None:
    # A constant channel has a NaN ACF and therefore no vote; the minimum
    # must fall to the remaining informative channel rather than become n/a.
    rng = np.random.default_rng(32)
    e = np.column_stack([np.full(2000, 3.0), rng.normal(0.0, 1.0, size=2000)])
    rho = acf(e, 48)
    q, p = min_p_ljung_box(rho, 2000, (24,))[24]
    q1, p1 = ljung_box(rho[:, 1], 2000, (24,))[24]
    assert (q, p) == (q1, p1)
    assert np.isfinite(p)


def test_periodogram_peaks_locate_a_known_period() -> None:
    rng = np.random.default_rng(13)
    t = np.arange(4000)
    e = np.sin(2.0 * np.pi * t / 24.0)[:, None] + 0.2 * rng.normal(size=(4000, 1))
    peaks = periodogram_peaks(e, top=5)
    periods = [period for period, _ in peaks]
    assert 1 <= len(peaks) <= 5
    assert periods == sorted(periods)  # period order, never ranked by power
    assert min(abs(period - 24.0) for period in periods) < 0.5  # within half a bin


def test_bj_drops_nan_rows_of_efull() -> None:
    series = np.random.default_rng(14).normal(0.0, 1.0, size=(1000, 1))
    e_full = np.full((2000, 1), np.nan)
    e_full[1::2] = series  # only every other index is an origin
    out = render_feedback("bj", _make_incumbent(e_full), [], _make_history(1), HeadroomConfig())
    assert f"| 1 | {acf(series, 48)[0, 0]:.4f} |" in out


def test_bj_mean_column_when_many_channels() -> None:
    e = np.random.default_rng(15).normal(0.0, 1.0, size=(600, 6))
    out = render_feedback("bj", _make_incumbent(e), [], _make_history(1), HeadroomConfig())
    assert "| lag | mean |" in out
    assert "ch0" not in out
