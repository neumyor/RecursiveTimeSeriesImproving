from __future__ import annotations

from dataclasses import dataclass

import pytest

from harnessing_ts.headroom.certify import CertReport, certify
from harnessing_ts.headroom.config import HeadroomConfig


@dataclass(frozen=True)
class _StubCoordinate:
    """Duck-typed stand-in for correctors.Coordinate; certify only reads `.label`."""

    label: str


@dataclass(frozen=True)
class _StubGain:
    """Duck-typed stand-in for correctors.GainReport; certify reads `.label`/`.ucb`/`.level`."""

    coordinate: _StubCoordinate
    ucb: float
    level: float


def _level(cfg: HeadroomConfig) -> float:
    # The fixed level certify demands of every gain, whatever the number of
    # statistics.
    return 1.0 - cfg.certify_alpha


def _gains(pairs: list[tuple[str, float]], level: float) -> list[_StubGain]:
    return [
        _StubGain(coordinate=_StubCoordinate(label), ucb=ucb, level=level)
        for label, ucb in pairs
    ]


def test_all_ucbs_below_thresholds_certifies_saturation() -> None:
    cfg = HeadroomConfig(certify_alpha=0.05, certify_epsilon=0.02)
    gains = _gains(
        [("total", 0.5), ("hgroup[1,12]", 0.1), ("hgroup[13,24]", 0.05)],
        level=_level(cfg),
    )
    mse_by_group = {"total": 100.0, "hgroup[1,12]": 80.0, "hgroup[13,24]": 40.0}
    report = certify(gains, mse_by_group, cfg)
    assert isinstance(report, CertReport)
    assert report.saturated is True
    assert report.failures == []
    # The fixed per-statistic level, whatever the number of statistics.
    assert report.level == pytest.approx(0.95)


def test_single_violation_is_reported_with_its_threshold() -> None:
    cfg = HeadroomConfig()
    # 0.02 * 80.0 = 1.6: the middle gain sits exactly at its threshold, which must
    # count as a violation because the stop rule demands a strict ucb < threshold.
    gains = _gains(
        [("total", 0.5), ("hgroup[1,12]", 1.6), ("hgroup[13,24]", 0.05)],
        level=_level(cfg),
    )
    mse_by_group = {"total": 100.0, "hgroup[1,12]": 80.0, "hgroup[13,24]": 40.0}
    report = certify(gains, mse_by_group, cfg)
    assert report.saturated is False
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure["label"] == "hgroup[1,12]"
    assert failure["ucb"] == pytest.approx(1.6)
    assert failure["threshold"] == pytest.approx(0.02 * 80.0)


def test_invalid_arguments_raise_value_error() -> None:
    cfg = HeadroomConfig()
    gains = _gains([("total", 0.5)], level=_level(cfg))
    mse_by_group = {"total": 100.0}
    with pytest.raises(ValueError, match="total"):
        certify(gains, {"hgroup[1,12]": 80.0}, cfg)  # gain label missing
    with pytest.raises(ValueError, match="empty"):
        certify([], mse_by_group, cfg)


def test_realistic_numbers_round_trip_against_epsilon_rule() -> None:
    cfg = HeadroomConfig(certify_epsilon=0.02)
    mse_by_group = {
        # ECG-scale cert-segment MSE per group; thresholds must be epsilon * these.
        "total": 4.21,
        "hgroup[1,12]": 5.83,
        "hgroup[13,24]": 2.47,
    }
    level = _level(cfg)
    near = _gains(
        [
            ("total", 0.02 * 4.21 * 0.999),
            ("hgroup[1,12]", 0.02 * 5.83 * 0.999),
            ("hgroup[13,24]", 0.02 * 2.47 * 0.999),
        ],
        level=level,
    )
    report = certify(near, mse_by_group, cfg)
    assert report.saturated is True
    assert report.failures == []
    # Crossing any single threshold by a hair flips the verdict and reports the
    # exact threshold, so the caller can render the remaining margin.
    crossed = _gains(
        [
            ("total", 0.02 * 4.21 * 1.001),
            ("hgroup[1,12]", 0.02 * 5.83 * 0.999),
            ("hgroup[13,24]", 0.02 * 2.47 * 0.999),
        ],
        level=level,
    )
    report = certify(crossed, mse_by_group, cfg)
    assert report.saturated is False
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure["label"] == "total"
    assert failure["ucb"] == pytest.approx(0.02 * 4.21 * 1.001)
    assert failure["threshold"] == pytest.approx(cfg.certify_epsilon * mse_by_group["total"])
