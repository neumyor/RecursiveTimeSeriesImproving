from __future__ import annotations

"""Coordinates over the realized information set, their ridge correctors, and held-out gains."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, lfilter, periodogram

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.inference import auto_bandwidth, mean_ci
from harnessing_ts.headroom.residuals import SegmentData

# Contract §5 fixes the channel coordinate's lag window at k = 0..23, matched
# to the first lag segment so the e history and value differences stay
# comparable in depth.
_CHANNEL_LAGS = 24

# Every coordinate names the defect it can repair with the same neutral
# aspect vocabulary the feedback arms share.
_COORDINATE_ASPECTS = {
    "total": "other",
    "lag": "lag-structure",
    "band": "periodicity",
    "beyond_l": "long-range-beyond-window",
    "channel": "cross-channel",
    "horizon": "horizon-range",
    "hgroup": "other",
}


@dataclass(frozen=True)
class Coordinate:
    """One named slice of the realized information set (contract §5)."""

    kind: str  # total | lag | band | beyond_l | channel | horizon | hgroup
    label: str  # e.g. "lag[0,23]", "band(P=24)", "channel->3", "hgroup[1,12]"
    aspect: str  # periodicity | lag-structure | long-range-beyond-window | ...
    spec: dict  # e.g. {"k1": 0, "k2": 23} / {"period": 24} / {"target_channel": 3}


@dataclass(frozen=True)
class GainReport:
    """Held-out realized gain of one coordinate's corrector (positive = better)."""

    coordinate: Coordinate
    delta: float  # mean per-origin loss reduction on the eval segment
    lcb: float  # HAC + HLN lower confidence bound at `level`
    ucb: float
    by_horizon: np.ndarray  # [H] per-step gains
    mu2: float  # (mean residual on the eval segment)^2, the level-bias scale
    n_eval: int


def select_period_peaks(
    e_full: np.ndarray,
    a_origins: np.ndarray,
    cfg: HeadroomConfig,
) -> list[int]:
    """Dominant integer periods of the e series at the contiguous A origins.

    Only the A segment is used so the eval segment cannot influence which
    periodicities the correctors are allowed to repair (A-M4).  Channels are
    averaged before peaks are ranked: a common oscillation across channels
    is the reproducible kind, a channel-private one is usually noise.
    """
    e_full = np.asarray(e_full, dtype=np.float64)
    if e_full.ndim != 2:
        raise ValueError(f"e_full must be [T, C], got shape {e_full.shape}")
    a_origins = np.asarray(a_origins, dtype=np.int64)
    if a_origins.ndim != 1 or a_origins.size < 2:
        raise ValueError(
            f"a_origins must be a 1-D array of at least 2 origins, got shape {a_origins.shape}"
        )
    if np.any(np.diff(a_origins) != 1):
        raise ValueError("a_origins must be contiguous: the e series is sampled at consecutive origins")
    if a_origins.min() < 0 or a_origins.max() >= e_full.shape[0]:
        raise ValueError(
            f"a_origins must index e_full of length {e_full.shape[0]}, got "
            f"[{a_origins.min()}, {a_origins.max()}]"
        )
    e = e_full[a_origins]
    if not np.all(np.isfinite(e)):
        raise ValueError("e_full is undefined (NaN) at some A origins; origins must lie in the realized range")
    freqs, power = periodogram(e[:, 0], detrend="constant")
    for channel in range(1, e.shape[1]):
        power = power + periodogram(e[:, channel], detrend="constant")[1]
    power /= e.shape[1]
    lo, hi = cfg.band_period_range
    # A bin qualifies when its integer period falls inside the configured
    # band; written multiplicatively to avoid dividing by the zero frequency.
    in_band = (freqs > 0.0) & (freqs * hi >= 1.0) & (freqs * lo <= 1.0)
    bins = np.flatnonzero(in_band)
    candidates = [
        int(bins[j])
        for j in range(1, bins.size - 1)
        if power[bins[j]] > power[bins[j - 1]] and power[bins[j]] >= power[bins[j + 1]]
    ]
    candidates.sort(key=lambda b: power[b], reverse=True)
    chosen: list[int] = []
    for b in candidates:
        # 峰间至少隔 2 个频率点: two accepted peaks always keep at least two
        # frequency bins between them, so one spectral lobe of a windowed
        # periodogram is never reported twice.
        if all(abs(b - k) > 2 for k in chosen):
            chosen.append(b)
        if len(chosen) == cfg.max_period_peaks:
            break
    return [int(round(1.0 / float(freqs[b]))) for b in chosen]


def enumerate_coordinates(
    cfg: HeadroomConfig,
    period_peaks: list[int],
    n_channels: int,
    horizon: int,
) -> list[Coordinate]:
    """The repairable coordinates; hgroups are added by all_gains, not here."""
    coordinates = [
        Coordinate("lag", f"lag[{k1},{k2}]", _COORDINATE_ASPECTS["lag"], {"k1": k1, "k2": k2})
        for k1, k2 in cfg.lag_segments
    ]
    coordinates += [
        Coordinate("band", f"band(P={period})", _COORDINATE_ASPECTS["band"], {"period": period})
        for period in period_peaks
    ]
    coordinates.append(Coordinate("beyond_l", "beyond_l", _COORDINATE_ASPECTS["beyond_l"], {}))
    coordinates += [
        Coordinate("channel", f"channel->{i}", _COORDINATE_ASPECTS["channel"], {"target_channel": i})
        for i in range(min(n_channels, cfg.channel_max))
    ]
    # The horizon coordinate needs no spec: its per-h blocks are implied by
    # r_full's shape; `horizon` stays in the signature per the contract.
    coordinates.append(Coordinate("horizon", "horizon", _COORDINATE_ASPECTS["horizon"], {}))
    return coordinates


def _lag_block(e_full: np.ndarray, origins: np.ndarray, k1: int, k2: int) -> np.ndarray:
    """e_full[o-k] for k in [k1, k2]; deep lags are sampled every second step."""
    # From k1 >= 96 (the default third segment) neighbouring lags are nearly
    # collinear in any series slow enough to matter at that depth, so
    # stepping by 2 costs little and keeps the feature count bounded.
    step = 2 if k1 >= 96 else 1
    ks = np.arange(k1, k2 + 1, step)
    idx = origins[:, None] - ks[None, :]
    return e_full[idx].reshape(origins.size, ks.size * e_full.shape[1])


def _band_block(e_full: np.ndarray, origins: np.ndarray, period: int, cfg: HeadroomConfig) -> np.ndarray:
    """Causal FIR band-pass of e around period P, read at two phase lags."""
    taps = min(cfg.fir_len_cap, 4 * period)
    if taps % 2 == 0:
        taps += 1  # odd length keeps the FIR phase-linear
    fir = firwin(
        taps,
        cutoff=(1.0 / (1.2 * period), 1.0 / (0.8 * period)),
        window="hamming",
        pass_zero=False,
    )
    # e_full is NaN off the realized rows, so the filter sees zeros there.
    # That cannot corrupt the rows we read: lfilter is causal, and burn_in =
    # max_lag + fir_len_cap keeps every tapped row inside the segment's
    # realized range for every usable origin.
    filled = np.nan_to_num(e_full, nan=0.0)
    filtered = lfilter(fir, [1.0], filled, axis=0)
    quarter = period // 4
    return np.concatenate([filtered[origins], filtered[origins - quarter]], axis=1)


def _beyond_block(values: np.ndarray, origins: np.ndarray, cfg: HeadroomConfig) -> np.ndarray:
    """values[o-k] - values[o] for k = L + 16*j: what a longer lookback saw.

    Raw values, not residuals: the point of this coordinate is precisely to
    measure structure the model's window could never see (A-M11), and
    differencing against values[o] keeps the feature level-free.
    """
    ks = cfg.input_length + 16 * np.arange(cfg.beyond_l_lags)
    past = values[origins[:, None] - ks[None, :]]
    current = values[origins][:, None, :]
    return (past - current).reshape(origins.size, ks.size * values.shape[1])


def _channel_blocks(
    seg: SegmentData,
    origins: np.ndarray,
    target: int | None,
) -> list[np.ndarray]:
    """Other channels' e history and value differences at k = 0..23.

    ``target=None`` (the total/hgroup union) keeps every channel: across all
    channel->i coordinates each channel appears as someone else's predictor.
    """
    ks = np.arange(_CHANNEL_LAGS + 1)
    idx = origins[:, None] - ks[None, :]
    blocks = []
    for channel in range(seg.values.shape[1]):
        if target is not None and channel == target:
            continue
        e = seg.e_full[idx, channel]
        dx = seg.values[idx, channel] - seg.values[origins, channel][:, None]
        blocks.append(np.concatenate([e, dx], axis=1))
    return blocks


def _horizon_blocks(r_full: np.ndarray, origins: np.ndarray, cfg: HeadroomConfig) -> list[np.ndarray]:
    """Per-h blocks r_full[o-h-k, h]: the trajectory of the h-step errors.

    One block per h (contract §5): coordinate_gain fits each step with its
    own ridge, because a step-h error follows its own same-horizon chain
    and a shared fit would smear the steps together (A-M3).
    """
    n_lags = cfg.horizon_realized_lags
    n_steps, n_channels = r_full.shape[1], r_full.shape[2]
    blocks = []
    for h in range(1, n_steps + 1):
        idx = origins[:, None] - (h + np.arange(n_lags))[None, :]
        blocks.append(r_full[idx, h - 1, :].reshape(origins.size, n_lags * n_channels))
    return blocks


def build_features(
    seg: SegmentData,
    coord: Coordinate,
    cfg: HeadroomConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """(X[B, F], Y[B, H, C]) at seg.origins, using only this coordinate's slice.

    Every feature is read at rows at or before its origin, so the arrays can
    be truncated at any origin without changing that origin's row.
    """
    origins = np.asarray(seg.origins, dtype=np.int64)
    if coord.kind == "lag":
        blocks = [_lag_block(seg.e_full, origins, coord.spec["k1"], coord.spec["k2"])]
    elif coord.kind == "band":
        blocks = [_band_block(seg.e_full, origins, coord.spec["period"], cfg)]
    elif coord.kind == "beyond_l":
        blocks = [_beyond_block(seg.values, origins, cfg)]
    elif coord.kind == "channel":
        blocks = _channel_blocks(seg, origins, coord.spec["target_channel"])
    elif coord.kind == "horizon":
        blocks = _horizon_blocks(seg.r_full, origins, cfg)
    elif coord.kind in ("total", "hgroup"):
        # hgroup reuses the total features and only re-scopes the evaluation
        # (contract §5), so both kinds build the identical union here.  The
        # band blocks come from spec["periods"], filled by all_gains: the
        # peaks are chosen on A and must be the same for every coordinate.
        blocks = [_lag_block(seg.e_full, origins, k1, k2) for k1, k2 in cfg.lag_segments]
        blocks += [_band_block(seg.e_full, origins, period, cfg) for period in coord.spec.get("periods", ())]
        blocks.append(_beyond_block(seg.values, origins, cfg))
        blocks += _channel_blocks(seg, origins, None)
        blocks += _horizon_blocks(seg.r_full, origins, cfg)
    else:
        raise ValueError(f"unknown coordinate kind {coord.kind!r}")
    x = np.concatenate(blocks, axis=1) if blocks else np.zeros((origins.size, 0))
    if coord.kind == "channel":
        # Only channel i's residuals are in play; the zeros are exact: a
        # zero target column yields a zero ridge column for every alpha, so
        # neither the alpha choice nor channel i's fit is affected.
        y = np.zeros_like(seg.r)
        y[:, :, coord.spec["target_channel"]] = seg.r[:, :, coord.spec["target_channel"]]
    else:
        y = seg.r
    return x, y


def _fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alphas: tuple[float, ...],
    embargo: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge with intercept; alpha picked on a time-ordered 80/20 split of x.

    The eval rows start `embargo` (= H) after the fit rows so overlapping
    multi-step targets never straddle the split.  The winning alpha is then
    refit on every row: it has been validated, so hiding rows from the final
    corrector would only discard information.
    """
    n = x.shape[0]
    n_fit = int(n * 0.8)
    eval_start = n_fit + embargo
    if n_fit < 2 or eval_start >= n:
        raise ValueError(
            f"need more than {eval_start} rows for the 80/20 alpha split with "
            f"embargo {embargo}, got {n}"
        )
    x_fit, y_fit = x[:n_fit], y[:n_fit]
    x_val, y_val = x[eval_start:], y[eval_start:]
    mean_x, mean_y = x_fit.mean(axis=0), y_fit.mean(axis=0)
    centered = x_fit - mean_x
    gram = centered.T @ centered
    cross = centered.T @ (y_fit - mean_y)
    eye = np.eye(x.shape[1])
    best_alpha = alphas[0]
    best_mse = np.inf
    for alpha in alphas:
        coef = np.linalg.solve(gram + alpha * eye, cross)
        resid = y_val - ((x_val - mean_x) @ coef + mean_y)
        mse = float(np.mean(resid * resid))
        if mse < best_mse:
            best_alpha, best_mse = alpha, mse
    # Closed form with intercept: center by the fit means, add them back.
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    centered = x - mean_x
    coef = np.linalg.solve(centered.T @ centered + best_alpha * eye, centered.T @ (y - mean_y))
    return coef, mean_y - mean_x @ coef


def coordinate_gain(
    coord: Coordinate,
    seg_a: SegmentData,
    seg_b: SegmentData,
    cfg: HeadroomConfig,
    level: float = 0.95,
) -> GainReport:
    """Fit on A, evaluate the realized gain per origin on B (contract §5)."""
    x_a, y_a = build_features(seg_a, coord, cfg)
    x_b, y_b = build_features(seg_b, coord, cfg)
    if coord.kind == "horizon":
        # One ridge per h on that h's own block of the feature matrix.
        width = cfg.horizon_realized_lags * seg_a.values.shape[1]
        y_hat = np.empty_like(y_b)
        for h in range(cfg.horizon):
            cols = slice(h * width, (h + 1) * width)
            coef, intercept = _fit_ridge(x_a[:, cols], y_a[:, h, :], cfg.ridge_alphas, cfg.horizon)
            y_hat[:, h, :] = x_b[:, cols] @ coef + intercept
    else:
        coef, intercept = _fit_ridge(
            x_a, y_a.reshape(x_a.shape[0], -1), cfg.ridge_alphas, cfg.horizon
        )
        y_hat = (x_b @ coef + intercept).reshape(y_b.shape)
    r = seg_b.r
    gains = r * r - (r - y_hat) ** 2  # [n_eval, H, C], positive = improvement
    channels = (
        [int(coord.spec["target_channel"])]
        if coord.kind == "channel"
        else list(range(r.shape[2]))
    )
    steps = (
        slice(int(coord.spec["h1"]) - 1, int(coord.spec["h2"]))
        if coord.kind == "hgroup"
        else slice(None)
    )
    d = gains[:, steps, channels].mean(axis=(1, 2))
    mean, lcb, ucb = mean_ci(d, bandwidth=auto_bandwidth(d, cfg.horizon), level=level)
    return GainReport(
        coordinate=coord,
        delta=mean,
        lcb=lcb,
        ucb=ucb,
        by_horizon=gains[:, :, channels].mean(axis=(0, 2)),
        mu2=float(r.mean() ** 2),
        n_eval=int(seg_b.origins.size),
    )


def all_gains(
    seg_a: SegmentData,
    seg_b: SegmentData,
    cfg: HeadroomConfig,
    period_peaks: list[int],
) -> list[GainReport]:
    """total + every enumerated coordinate + the two step-length groups.

    The total and hgroup coordinates need the A-selected periods to build
    their band blocks, so they carry them in spec["periods"] — build_features
    has no other way to receive them (contract signature).
    """
    periods = tuple(period_peaks)
    n_channels = seg_a.values.shape[1]
    horizon = cfg.horizon
    coordinates = [
        Coordinate("total", "total", _COORDINATE_ASPECTS["total"], {"periods": periods})
    ]
    coordinates += enumerate_coordinates(cfg, period_peaks, n_channels, horizon)
    half = horizon // 2
    coordinates += [
        Coordinate(
            "hgroup",
            f"hgroup[1,{half}]",
            _COORDINATE_ASPECTS["hgroup"],
            {"h1": 1, "h2": half, "periods": periods},
        ),
        Coordinate(
            "hgroup",
            f"hgroup[{half + 1},{horizon}]",
            _COORDINATE_ASPECTS["hgroup"],
            {"h1": half + 1, "h2": horizon, "periods": periods},
        ),
    ]
    return [coordinate_gain(coord, seg_a, seg_b, cfg) for coord in coordinates]
