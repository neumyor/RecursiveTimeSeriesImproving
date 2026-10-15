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


def _level(cfg: HeadroomConfig, j_max: int, n_gains: int) -> float:
    # The Bonferroni level certify both recomputes and demands of every gain.
    return 1.0 - (cfg.certify_alpha / j_max) / n_gains


def _gains(pairs: list[tuple[str, float]], level: float) -> list[_StubGain]:
    return [
        _StubGain(coordinate=_StubCoordinate(label), ucb=ucb, level=level)
        for label, ucb in pairs
    ]


def test_all_ucbs_below_thresholds_certifies_saturation() -> None:
    cfg = HeadroomConfig(certify_alpha=0.05, certify_epsilon=0.02)
    gains = _gains(
        [("total", 0.5), ("hgroup[1,12]", 0.1), ("hgroup[13,24]", 0.05)],
        level=_level(cfg, j_max=10, n_gains=3),
    )
    mse_by_group = {"total": 100.0, "hgroup[1,12]": 80.0, "hgroup[13,24]": 40.0}
    report = certify(gains, mse_by_group, cfg, j=1, j_max=10)
    assert isinstance(report, CertReport)
    assert report.round_index == 1
    assert report.saturated is True
    assert report.failures == []
    # level = 1 - (alpha / j_max) / K with K = 3 (Bonferroni across the K statistics).
    assert report.level == pytest.approx(1.0 - (0.05 / 10.0) / 3.0)


def test_single_violation_is_reported_with_its_threshold() -> None:
    cfg = HeadroomConfig()
    # 0.02 * 80.0 = 1.6: the middle gain sits exactly at its threshold, which must
    # count as a violation because the stop rule demands a strict ucb < threshold.
    gains = _gains(
        [("total", 0.5), ("hgroup[1,12]", 1.6), ("hgroup[13,24]", 0.05)],
        level=_level(cfg, j_max=10, n_gains=3),
    )
    mse_by_group = {"total": 100.0, "hgroup[1,12]": 80.0, "hgroup[13,24]": 40.0}
    report = certify(gains, mse_by_group, cfg, j=3, j_max=10)
    assert report.round_index == 3
    assert report.saturated is False
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure["label"] == "hgroup[1,12]"
    assert failure["ucb"] == pytest.approx(1.6)
    assert failure["threshold"] == pytest.approx(0.02 * 80.0)


def test_invalid_arguments_raise_value_error() -> None:
    cfg = HeadroomConfig()
    gains = _gains([("total", 0.5)], level=_level(cfg, j_max=5, n_gains=1))
    mse_by_group = {"total": 100.0}
    with pytest.raises(ValueError, match="total"):
        certify(gains, {"hgroup[1,12]": 80.0}, cfg, j=1, j_max=5)  # gain label missing
    with pytest.raises(ValueError, match="j_max"):
        certify(gains, mse_by_group, cfg, j=6, j_max=5)
    with pytest.raises(ValueError, match="empty"):
        certify([], mse_by_group, cfg, j=1, j_max=5)
    with pytest.raises(ValueError, match="j >= 1"):
        certify(gains, mse_by_group, cfg, j=0, j_max=5)


def test_gain_computed_at_the_wrong_level_is_rejected() -> None:
    cfg = HeadroomConfig()
    expected = _level(cfg, j_max=5, n_gains=2)
    # Only the second gain is off-level: the guard is per gain, not per call.
    gains = [
        _StubGain(_StubCoordinate("total"), 0.5, expected),
        _StubGain(_StubCoordinate("hgroup[1,12]"), 0.1, 0.95),
    ]
    with pytest.raises(RuntimeError) as excinfo:
        certify(gains, {"total": 100.0, "hgroup[1,12]": 80.0}, cfg, j=1, j_max=5)
    message = str(excinfo.value)
    assert "hgroup[1,12]" in message  # the offending gain's label
    assert "0.95" in message  # ... and its level
    assert f"{expected}" in message  # ... and the level certify required
    # ... and the correct wiring to use instead.
    assert "all_gains" in message
    assert "certify_level" in message


def test_level_off_by_float_noise_is_accepted() -> None:
    cfg = HeadroomConfig()
    level = _level(cfg, j_max=5, n_gains=1)
    # Re-deriving the level through a different float path stays inside the
    # 1e-9 tolerance, so the guard must not fire on float noise.
    gains = [_StubGain(_StubCoordinate("total"), 0.5, level + 1e-12)]
    report = certify(gains, {"total": 100.0}, cfg, j=1, j_max=5)
    assert report.saturated is True


def test_larger_j_max_spends_alpha_more_thinly() -> None:
    cfg = HeadroomConfig(certify_alpha=0.05)
    mse_by_group = {"total": 100.0, "hgroup[1,12]": 80.0}
    tight = certify(
        _gains([("total", 0.5), ("hgroup[1,12]", 0.1)], level=_level(cfg, j_max=2, n_gains=2)),
        mse_by_group,
        cfg,
        j=1,
        j_max=2,
    )
    loose = certify(
        _gains([("total", 0.5), ("hgroup[1,12]", 0.1)], level=_level(cfg, j_max=20, n_gains=2)),
        mse_by_group,
        cfg,
        j=1,
        j_max=20,
    )
    # Uniform spending: each of the j_max planned certifications gets alpha / j_max,
    # so a larger j_max splits the budget thinner per check. The saved level is then
    # higher, so the CI the caller builds at that level is wider and the bound on
    # retained gain looser (harder to declare saturation). The verdict on already
    # computed UCBs is unchanged; only the level moves.
    assert tight.level == pytest.approx(1.0 - (0.05 / 2.0) / 2.0)
    assert loose.level == pytest.approx(1.0 - (0.05 / 20.0) / 2.0)
    assert loose.level > tight.level
    assert tight.saturated is True
    assert loose.saturated is True
    # Family-wise budget invariant: j_max * alpha_j = certify_alpha.
    for report, j_max in ((tight, 2), (loose, 20)):
        assert j_max * (1.0 - report.level) * 2 == pytest.approx(0.05)


def test_realistic_numbers_round_trip_against_epsilon_rule() -> None:
    cfg = HeadroomConfig(certify_epsilon=0.02)
    mse_by_group = {
        # ECG-scale cert-segment MSE per group; thresholds must be epsilon * these.
        "total": 4.21,
        "hgroup[1,12]": 5.83,
        "hgroup[13,24]": 2.47,
    }
    level = _level(cfg, j_max=8, n_gains=3)
    near = _gains(
        [
            ("total", 0.02 * 4.21 * 0.999),
            ("hgroup[1,12]", 0.02 * 5.83 * 0.999),
            ("hgroup[13,24]", 0.02 * 2.47 * 0.999),
        ],
        level=level,
    )
    report = certify(near, mse_by_group, cfg, j=2, j_max=8)
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
    report = certify(crossed, mse_by_group, cfg, j=2, j_max=8)
    assert report.saturated is False
    assert len(report.failures) == 1
    failure = report.failures[0]
    assert failure["label"] == "total"
    assert failure["ucb"] == pytest.approx(0.02 * 4.21 * 1.001)
    assert failure["threshold"] == pytest.approx(cfg.certify_epsilon * mse_by_group["total"])
