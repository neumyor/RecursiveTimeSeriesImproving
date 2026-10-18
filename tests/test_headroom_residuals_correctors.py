from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import (
    Coordinate,
    all_gains,
    build_features,
    coordinate_gain,
    enumerate_coordinates,
    select_period_peaks,
)
from harnessing_ts.headroom.residuals import SegmentData, segment_data
from harnessing_ts.headroom.timeidx import ab_split, segment_origins

T_LEN = 1600
SEG_START = 200
SEG_STOP = 1600
N_CHANNELS = 2


def small_cfg() -> HeadroomConfig:
    """Test-sized config: burn_in drops from 591 to 39 so T=1600 suffices."""
    return HeadroomConfig(input_length=48, horizon=8, lag_segments=((0, 23),), fir_len_cap=16)


def wide_cfg() -> HeadroomConfig:
    """Also exercises the step-2 deep-lag segment; burn_in = 151."""
    return HeadroomConfig(
        input_length=48,
        horizon=8,
        lag_segments=((0, 23), (24, 95), (96, 135)),
        fir_len_cap=16,
    )


def zero_model_segment(values: np.ndarray, cfg: HeadroomConfig) -> SegmentData:
    """Model == 0, so the residuals are the future values themselves."""
    n_rows = SEG_STOP - SEG_START - cfg.horizon
    pred = np.zeros((n_rows, cfg.horizon, values.shape[1]))
    return segment_data(values, SEG_START, SEG_STOP, pred, cfg)


def segment_from_residuals(
    values: np.ndarray, r_planted: np.ndarray, cfg: HeadroomConfig
) -> SegmentData:
    """pred[t] = target - r_planted[t], so segment_data recovers r_planted."""
    rows = np.arange(SEG_START, SEG_STOP - cfg.horizon)
    targets = values[rows[:, None] + np.arange(1, cfg.horizon + 1)]
    return segment_data(values, SEG_START, SEG_STOP, targets - r_planted, cfg)


def split_ab(seg: SegmentData, cfg: HeadroomConfig) -> tuple[SegmentData, SegmentData]:
    """A/B as separate SegmentData views over the same realized series.

    B's features legitimately read residuals realized during A — they are in
    B origins' past — which is why e_full/r_full are shared, not split.
    """
    a_origins, b_origins = ab_split(seg.origins, a_ratio=0.5, embargo=cfg.horizon)
    i_a = np.searchsorted(seg.origins, a_origins)
    i_b = np.searchsorted(seg.origins, b_origins)
    seg_a = SegmentData(seg.values, a_origins, seg.r[i_a], seg.e_full, seg.r_full)
    seg_b = SegmentData(seg.values, b_origins, seg.r[i_b], seg.e_full, seg.r_full)
    return seg_a, seg_b


def test_segment_data_nan_placement_and_shapes() -> None:
    cfg = small_cfg()
    rng = np.random.default_rng(2)
    values = rng.normal(size=(T_LEN, N_CHANNELS))
    n_rows = SEG_STOP - SEG_START - cfg.horizon
    pred = rng.normal(size=(n_rows, cfg.horizon, N_CHANNELS))
    seg = segment_data(values, SEG_START, SEG_STOP, pred, cfg)
    assert seg.e_full.shape == (T_LEN, N_CHANNELS)
    assert seg.r_full.shape == (T_LEN, cfg.horizon, N_CHANNELS)
    np.testing.assert_array_equal(seg.origins, segment_origins(SEG_START, SEG_STOP, cfg))
    assert seg.r.shape == (seg.origins.size, cfg.horizon, N_CHANNELS)
    realized = np.zeros(T_LEN, dtype=bool)
    realized[SEG_START : SEG_STOP - cfg.horizon] = True
    np.testing.assert_array_equal(~np.isnan(seg.r_full).any(axis=(1, 2)), realized)
    # e_full is the h=1 residual shifted one row: e_full[t] = r[origin t-1,
    # h=1], the latest 1-step residual realized BY time t (plan §2.3).
    e_rows = np.zeros(T_LEN, dtype=bool)
    e_rows[SEG_START + 1 : SEG_STOP - cfg.horizon + 1] = True
    np.testing.assert_array_equal(~np.isnan(seg.e_full).any(axis=1), e_rows)
    t = SEG_START + 37
    np.testing.assert_allclose(seg.e_full[t], values[t] - pred[t - 1 - SEG_START, 0])
    np.testing.assert_allclose(
        seg.r_full[t], values[t + 1 : t + 1 + cfg.horizon] - pred[t - SEG_START]
    )
    i = 5
    o = int(seg.origins[i])
    np.testing.assert_allclose(seg.r[i], seg.r_full[o])
    np.testing.assert_allclose(seg.e_full[o], seg.r_full[o - 1, 0])


def test_segment_data_rejects_misaligned_pred() -> None:
    cfg = small_cfg()
    values = np.zeros((T_LEN, N_CHANNELS))
    with pytest.raises(ValueError, match="one row per origin"):
        segment_data(values, SEG_START, SEG_STOP, np.zeros((3, cfg.horizon, N_CHANNELS)), cfg)
    with pytest.raises(ValueError, match="not longer than horizon"):
        segment_data(values, 100, 105, np.zeros((0, cfg.horizon, N_CHANNELS)), cfg)


def test_enumerate_coordinates_labels_aspects_and_channel_max() -> None:
    cfg = small_cfg()
    coords = enumerate_coordinates(cfg, [24], 5)
    assert [c.label for c in coords] == [
        "lag[0,23]",
        "band(P=24)",
        "beyond_l",
        "channel->0",
        "channel->1",
        "channel->2",
        "channel->3",
        "channel->4",
        "horizon",
    ]
    aspects = {c.kind: c.aspect for c in coords}
    assert aspects["lag"] == "lag-structure"
    assert aspects["band"] == "periodicity"
    assert aspects["beyond_l"] == "long-range-beyond-window"
    assert aspects["channel"] == "cross-channel"
    assert aspects["horizon"] == "horizon-range"
    assert [c.spec for c in coords[:2]] == [{"k1": 0, "k2": 23}, {"period": 24}]
    # channel_max caps how many channels get their own coordinate
    capped = enumerate_coordinates(replace(cfg, channel_max=2), [24], 5)
    assert [c.label for c in capped if c.kind == "channel"] == ["channel->0", "channel->1"]


def test_white_noise_total_gain_near_zero() -> None:
    cfg = small_cfg()
    rng = np.random.default_rng(1)
    sigma = 2.0
    values = rng.normal(0.0, sigma, size=(T_LEN, N_CHANNELS))
    seg = zero_model_segment(values, cfg)
    seg_a, seg_b = split_ab(seg, cfg)
    peaks = select_period_peaks(seg.e_full, seg_a.origins, cfg)
    reports = all_gains(seg_a, seg_b, cfg, peaks)
    labels = [rep.coordinate.label for rep in reports]
    assert labels[0] == "total"
    assert "lag[0,23]" in labels and "beyond_l" in labels and "horizon" in labels
    assert "channel->0" in labels and "channel->1" in labels
    assert any(label.startswith("band(P=") for label in labels)
    assert labels[-2] == "hgroup[1,4]" and labels[-1] == "hgroup[5,8]"
    total = reports[0]
    # Nothing is predictable, so a fitted corrector can only add variance;
    # alpha selection keeps that an order of magnitude below the noise.
    assert abs(total.delta) < 0.08 * sigma**2, total.delta
    assert total.lcb <= 0.02 * sigma**2, total.lcb
    assert total.ucb >= total.lcb
    assert total.n_eval == seg_b.origins.size
    assert total.by_horizon.shape == (cfg.horizon,)
    assert total.mu2 == pytest.approx(seg_b.r.mean() ** 2, abs=1e-12)
    assert np.all(np.isfinite(total.by_horizon))


def test_lag_coordinate_recovers_planted_ar1() -> None:
    cfg = small_cfg()
    n_steps, n_channels = cfg.horizon, N_CHANNELS
    rng = np.random.default_rng(3)
    values = rng.normal(size=(T_LEN, n_channels))
    n_rows = SEG_STOP - SEG_START - n_steps
    phi = 0.6
    # One independent AR(1) chain per channel; the e series (the shifted
    # h=1 residual) IS this chain, and every step's residual continues it:
    # r[o, h, c] = 0.6 * e_full[o, c] + fresh noise.  A persistent 1-step
    # error trailing into every horizon is exactly the defect the lag
    # coordinate repairs.  All (h, c) cells carry it so the shared ridge
    # alpha is not dominated by noise columns.
    eps = rng.normal(size=(n_rows + 1, n_channels))
    chain = np.empty((n_rows + 1, n_channels))
    chain[0] = eps[0] / np.sqrt(1.0 - phi**2)  # stationary start
    for i in range(1, n_rows + 1):
        chain[i] = phi * chain[i - 1] + eps[i]
    var_e = float(chain.var())
    r_planted = rng.normal(size=(n_rows, n_steps, n_channels))
    # h=1 continues the chain (so e_full[t] = chain[t - SEG_START]);
    # h>=2 repeats the latest realized e plus fresh noise.
    r_planted[:, 0, :] = chain[1:]
    r_planted[:, 1:, :] += phi * chain[:-1, None, :]
    seg = segment_from_residuals(values, r_planted, cfg)
    seg_a, seg_b = split_ab(seg, cfg)
    reports = {rep.coordinate.label: rep for rep in all_gains(seg_a, seg_b, cfg, [])}
    lag = reports["lag[0,23]"]
    # The best corrector explains phi^2 * Var(e) in every (h, c) cell, so
    # both by_horizon and the (h, c)-averaged delta equal it.  Per-cell
    # realized gains still fluctuate with the finite eval segment, so the
    # by_horizon band is wider than the delta band (observed spread ~16%).
    expected = phi**2 * var_e
    for h in range(n_steps):
        assert abs(lag.by_horizon[h] - expected) < 0.2 * expected, h
    assert abs(lag.delta - expected) < 0.15 * expected
    # Coordinates blind to channel c's own e history find nothing.
    # (The horizon coordinate is exempt: its per-h blocks are the realized
    # same-step residuals, which carry the same chain by construction.)
    for label in ("beyond_l", "channel->0", "channel->1"):
        assert abs(reports[label].delta) < 0.1 * expected, label


def test_band_coordinate_detects_planted_period() -> None:
    cfg = small_cfg()
    rng = np.random.default_rng(4)
    period = 24
    amplitude = 2.0
    values = rng.normal(0.0, 0.3, size=(T_LEN, N_CHANNELS))
    values[:, 0] += amplitude * np.sin(2.0 * np.pi * np.arange(T_LEN) / period)
    seg = zero_model_segment(values, cfg)
    seg_a, seg_b = split_ab(seg, cfg)
    peaks = select_period_peaks(seg.e_full, seg_a.origins, cfg)
    assert peaks and peaks[0] == period
    coords = enumerate_coordinates(cfg, peaks, N_CHANNELS)
    band = next(c for c in coords if c.kind == "band" and c.spec["period"] == period)
    report = coordinate_gain(band, seg_a, seg_b, cfg)
    # The sinusoid is an exact linear combination of the two filtered phase
    # lags (P//4 = a quarter period), so it is recovered at every step of
    # channel 0; the mean over h and channels leaves signal_var / C.
    signal_var = amplitude**2 / 2.0
    assert report.delta > 0.5 * signal_var / N_CHANNELS
    assert abs(report.by_horizon[0] - signal_var / N_CHANNELS) < (
        0.25 * signal_var / N_CHANNELS
    )


def test_horizon_coordinate_recovers_same_horizon_pattern() -> None:
    cfg = small_cfg()
    n_steps, n_channels = cfg.horizon, N_CHANNELS
    rng = np.random.default_rng(6)
    values = rng.normal(size=(T_LEN, n_channels))
    n_rows = SEG_STOP - SEG_START - n_steps
    r_planted = rng.normal(size=(n_rows, n_steps, n_channels))
    # Every channel's h=2 residual follows its own same-horizon chain:
    # r_full[t, 1, c] = 0.5 * r_full[t-2, 1, c] + eps.  Only the horizon
    # coordinate's h=2 block can see r_full[., 1, :]; the e series (h=1) and
    # the raw values are blind to it.  All channels carry the pattern so the
    # per-h ridge alpha is not dominated by noise columns.
    phi = 0.5
    eps = rng.normal(size=(n_rows, n_channels))
    for c in range(n_channels):
        for t in range(2, n_rows):
            r_planted[t, 1, c] = phi * r_planted[t - 2, 1, c] + eps[t, c]
    var_h2 = float(r_planted[:, 1, :].var())
    seg = segment_from_residuals(values, r_planted, cfg)
    seg_a, seg_b = split_ab(seg, cfg)
    reports = {rep.coordinate.label: rep for rep in all_gains(seg_a, seg_b, cfg, [])}
    horizon = reports["horizon"]
    # Only the h=2 cells are explainable, so the (h, c)-averaged delta is
    # phi^2 * Var(h=2 residual) scaled by one step out of H.
    expected = phi**2 * var_h2
    assert abs(horizon.by_horizon[1] - expected) < 0.2 * expected
    for h in (0, *range(2, n_steps)):
        assert abs(horizon.by_horizon[h]) < 0.06 * expected
    assert abs(horizon.delta - expected / n_steps) < 0.2 * expected / n_steps
    # A blind coordinate's held-out cost from fitting its ~50 features on
    # the A segment is a few percent of the signal (observed ~0.01 here).
    assert abs(reports["lag[0,23]"].delta) < 0.1 * expected


def test_truncation_invariance_of_features() -> None:
    cfg = wide_cfg()
    rng = np.random.default_rng(5)
    n_channels = 3
    t = np.arange(T_LEN)
    values = rng.normal(size=(T_LEN, n_channels))
    values[:, 0] += 1.5 * np.sin(2.0 * np.pi * t / 12)  # periodicity for band
    values[:, 1] += 0.5 * values[:, 0]  # cross-channel coupling
    values[:, 2] += 0.6 * np.roll(values[:, 2], 1) + rng.normal(size=T_LEN)  # slow AR
    seg = zero_model_segment(values, cfg)
    coords = [
        Coordinate("lag", "lag[0,23]", "lag-structure", {"k1": 0, "k2": 23}),
        Coordinate("lag", "lag[96,135]", "lag-structure", {"k1": 96, "k2": 135}),
        Coordinate("band", "band(P=12)", "periodicity", {"period": 12}),
        Coordinate("band", "band(P=24)", "periodicity", {"period": 24}),
        Coordinate("beyond_l", "beyond_l", "long-range-beyond-window", {}),
        Coordinate("channel", "channel->0", "cross-channel", {"target_channel": 0}),
        Coordinate("horizon", "horizon", "horizon-range", {}),
        # total/hgroup gains come from the greedy FittedCorrector, not
        # build_features (which now rejects those kinds); every feature block
        # the greedy corrector can use is covered by the kinds above.
    ]
    x_full = {c.label: build_features(seg, c, cfg)[0] for c in coords}
    rows = [0, len(seg.origins) // 3, len(seg.origins) - 1]
    for coord in coords:
        for i in rows:
            o = int(seg.origins[i])
            truncated = SegmentData(
                values=seg.values[: o + 1],
                origins=np.array([o]),
                r=seg.r[i : i + 1],
                e_full=seg.e_full[: o + 1],
                r_full=seg.r_full[: o + 1],
            )
            x_trunc, _ = build_features(truncated, coord, cfg)
            np.testing.assert_allclose(
                x_full[coord.label][i], x_trunc[0], rtol=1e-12, atol=1e-12, err_msg=coord.label
            )


def test_select_period_peaks_rejects_bad_inputs() -> None:
    cfg = small_cfg()
    with pytest.raises(ValueError, match="contiguous"):
        select_period_peaks(np.zeros((100, 2)), np.array([10, 12]), cfg)
    e = np.zeros((100, 1))
    e[:50] = np.nan
    with pytest.raises(ValueError, match="undefined"):
        select_period_peaks(e, np.arange(20), cfg)


def test_build_features_rejects_unknown_kind() -> None:
    cfg = small_cfg()
    seg = zero_model_segment(np.zeros((T_LEN, N_CHANNELS)), cfg)
    with pytest.raises(ValueError, match="unknown coordinate kind"):
        build_features(seg, Coordinate("nope", "nope", "other", {}), cfg)


def test_band_filter_passes_target_period_at_production_scale() -> None:
    """Regression for the missing firwin Nyquist factor: at the default
    fir_len_cap the band(P=24) filter must actually pass period 24."""
    from scipy.signal import firwin, freqz

    period = 24
    taps = min(256, 4 * period)
    if taps % 2 == 0:
        taps += 1
    fir = firwin(
        taps,
        cutoff=(2.0 / (1.2 * period), 2.0 / (0.8 * period)),
        window="hamming",
        pass_zero=False,
    )
    freqs, response = freqz(fir, worN=8000)
    target = 2.0 * np.pi / period
    gain = np.interp(target, freqs, np.abs(response))
    assert gain >= 0.9, f"band filter gain at period {period} is only {gain:.3f}"


def test_greedy_total_recovers_planted_ar1_and_not_noise() -> None:
    """The greedy aggregate (replacing the union ridge) must capture a planted
    AR(1) defect and stay near zero on white noise."""
    from harnessing_ts.headroom.correctors import corrector_gain, fit_greedy

    cfg = wide_cfg()
    # Planted AR(1) on the values themselves (m == 0, so the residuals are
    # the future values).  The chain must be generated recursively — the
    # vectorized one-liner would mix two independent white noises instead.
    rng = np.random.default_rng(11)
    phi = 0.6
    values = np.empty((T_LEN, N_CHANNELS))
    values[0] = rng.normal(size=N_CHANNELS)
    for t in range(1, T_LEN):
        values[t] = phi * values[t - 1] + rng.normal(size=N_CHANNELS)
    seg = zero_model_segment(values, cfg)
    seg_a, seg_b = split_ab(seg, cfg)
    peaks = select_period_peaks(seg.e_full, seg_a.origins, cfg)
    fitted = fit_greedy(seg_a, cfg, peaks)
    total = corrector_gain(fitted, seg_b, cfg)
    var_x = 1.0 / (1.0 - phi**2)
    # delta averages over H horizons with decaying phi^(2h) gains; the ideal
    # is ~0.11 and ridge shrinkage on 24 collinear lags keeps about half.
    # A broken greedy (empty selection or sign flip) lands near zero.
    assert total.delta > 0.3 * phi**2 * var_x / cfg.horizon, (total.delta, phi**2 * var_x)
    assert total.level == 0.95
    # The greedy corrector selected something (not the empty intercept fit).
    assert len(fitted.coordinates) >= 1

    # White noise: nothing to select, total must stay near zero.
    values_w = rng.normal(0.0, 2.0, size=(T_LEN, N_CHANNELS))
    seg_w = zero_model_segment(values_w, cfg)
    seg_wa, seg_wb = split_ab(seg_w, cfg)
    peaks_w = select_period_peaks(seg_w.e_full, seg_wa.origins, cfg)
    total_w = corrector_gain(fit_greedy(seg_wa, cfg, peaks_w), seg_wb, cfg)
    assert abs(total_w.delta) < 0.08 * 4.0, total_w.delta


def test_channel_coordinate_excluded_for_univariate_series() -> None:
    cfg = wide_cfg()
    coords = enumerate_coordinates(cfg, [12], 1)
    assert all(c.kind != "channel" for c in coords)
    assert all(c.kind != "channel" for c in enumerate_coordinates(cfg, [], 1))


def test_card_level_is_stricter_than_95pct() -> None:
    from harnessing_ts.headroom.correctors import card_level

    cfg = wide_cfg()
    peaks = [24, 168]
    card = card_level(cfg, peaks, N_CHANNELS)
    assert 0.95 < card < 1.0
    # (The certification level is the plain fixed 1 - certifyAlpha; only the
    # card keeps a Bonferroni level.)


def test_channel_block_spans_lags_0_to_23() -> None:
    """Contract §5: the channel coordinate reads k = 0..23 (24 lags), so with
    one other channel its e and dx blocks together have 2 * 24 columns."""
    cfg = small_cfg()
    rng = np.random.default_rng(9)
    seg = zero_model_segment(rng.normal(size=(T_LEN, N_CHANNELS)), cfg)
    coord = Coordinate("channel", "channel->0", "cross-channel", {"target_channel": 0})
    x, _ = build_features(seg, coord, cfg)
    assert x.shape[1] == 2 * 24


def test_greedy_selection_ignores_residual_level_bias() -> None:
    """A large residual mean must not stop greedy selection from picking the
    coordinate that explains the variation around that mean: candidates carry
    the same intercept-only term the empty-label baseline does."""
    from harnessing_ts.headroom.correctors import (
        FittedCorrector,
        corrector_gain,
        fit_greedy,
    )

    cfg = small_cfg()
    rng = np.random.default_rng(12)
    values = rng.normal(size=(T_LEN, N_CHANNELS))
    n_rows = SEG_STOP - SEG_START - cfg.horizon
    phi = 0.6
    level = 4.0
    # Same planted AR(1) defect as the lag-recovery test, but the chain (and
    # therefore every residual) sits at a large nonzero level.
    eps = rng.normal(size=(n_rows + 1, N_CHANNELS))
    chain = np.empty((n_rows + 1, N_CHANNELS))
    chain[0] = level + eps[0]
    for i in range(1, n_rows + 1):
        chain[i] = level + phi * (chain[i - 1] - level) + eps[i]
    var_e = float(chain.var())
    r_planted = rng.normal(size=(n_rows, cfg.horizon, N_CHANNELS))
    r_planted[:, 0, :] = chain[1:]
    r_planted[:, 1:, :] += level + phi * chain[:-1, None, :]
    seg = segment_from_residuals(values, r_planted, cfg)
    seg_a, seg_b = split_ab(seg, cfg)
    fitted = fit_greedy(seg_a, cfg, [])
    assert "lag[0,23]" in [coord.label for coord in fitted.coordinates]
    total = corrector_gain(fitted, seg_b, cfg)
    # The selection must beat an intercept-only corrector by the AR signal:
    # every (h, c) cell carries phi^2 * Var(e) on top of the level, which
    # both correctors' intercepts absorb equally.
    intercept_only = FittedCorrector(
        coordinates=(),
        means=np.zeros(0),
        stds=np.ones(0),
        coef=np.zeros((0, cfg.horizon * N_CHANNELS)),
        intercept=seg_a.r.mean(axis=0).reshape(-1),
    )
    base = corrector_gain(intercept_only, seg_b, cfg)
    assert total.delta > base.delta + 0.5 * phi**2 * var_e, (total.delta, base.delta)
