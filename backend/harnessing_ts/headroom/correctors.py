"""Coordinates over the realized information set, their ridge correctors, and held-out gains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, lfilter, periodogram

from harnessing_ts.headroom.config import BEYOND_L_STRIDE, HeadroomConfig
from harnessing_ts.headroom.inference import auto_bandwidth, mean_ci
from harnessing_ts.headroom.residuals import SegmentData

# Contract §5 fixes the channel coordinate's lag window at k = 0..23, matched
# to the first lag segment so the e history and value differences stay
# comparable in depth.
_CHANNEL_LAGS = 24
_GREEDY_MAX_STEPS = 4

# From this lag depth on neighbouring lags are nearly collinear in any series
# slow enough to matter at that depth, so _lag_block samples them every second
# step: it costs little and keeps the feature count bounded.
_DEEP_LAG_START = 96

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
    level: float  # confidence level lcb/ucb were computed at
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
        # One spectral lobe must be reported once: require a relative
        # frequency separation, not a bin separation (adjacent bins of a
        # long-tailed lobe would otherwise all qualify at short periods).
        # The first and last in-band bins are excluded above (no in-band
        # neighbour on one side).
        period_b = 1.0 / float(freqs[b])
        if all(abs(period_b - 1.0 / float(freqs[k])) / (1.0 / float(freqs[k])) >= 0.15 for k in chosen):
            chosen.append(b)
        if len(chosen) == cfg.max_period_peaks:
            break
    periods: list[int] = []
    for b in chosen:
        period = int(round(1.0 / float(freqs[b])))
        # Adjacent bins can round to the same integer period at short
        # periods; duplicates would inflate the certification family.
        if period not in periods:
            periods.append(period)
    return periods


def enumerate_coordinates(
    cfg: HeadroomConfig,
    period_peaks: list[int],
    n_channels: int,
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
    # A channel->i coordinate needs at least one OTHER channel to feed on;
    # with C=1 it degenerates to an intercept-only fit and would sit on the
    # card of every univariate task.
    if n_channels >= 2:
        coordinates += [
            Coordinate("channel", f"channel->{i}", _COORDINATE_ASPECTS["channel"], {"target_channel": i})
            for i in range(min(n_channels, cfg.channel_max))
        ]
    # The horizon coordinate needs no spec: its per-h blocks are implied by
    # r_full's shape.
    coordinates.append(Coordinate("horizon", "horizon", _COORDINATE_ASPECTS["horizon"], {}))
    return coordinates


def _lag_block(e_full: np.ndarray, origins: np.ndarray, k1: int, k2: int) -> np.ndarray:
    """e_full[o-k] for k in [k1, k2]; deep lags are sampled every second step."""
    step = 2 if k1 >= _DEEP_LAG_START else 1
    ks = np.arange(k1, k2 + 1, step)
    idx = origins[:, None] - ks[None, :]
    return e_full[idx].reshape(origins.size, ks.size * e_full.shape[1])


def _band_block(e_full: np.ndarray, origins: np.ndarray, period: int, cfg: HeadroomConfig) -> np.ndarray:
    """Causal FIR band-pass of e around period P, read at two phase lags."""
    taps = min(cfg.fir_len_cap, 4 * period)
    if taps % 2 == 0:
        taps += 1  # odd length keeps the FIR phase-linear
    # firwin cutoff is a fraction of Nyquist (1.0 = pi rad/sample), so the
    # angular band [2*pi/(1.2P), 2*pi/(0.8P)] becomes [2/(1.2P), 2/(0.8P)].
    fir = firwin(
        taps,
        cutoff=(2.0 / (1.2 * period), 2.0 / (0.8 * period)),
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
    ks = cfg.input_length + BEYOND_L_STRIDE * np.arange(cfg.beyond_l_lags)
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
    ks = np.arange(_CHANNEL_LAGS)
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
        raise ValueError(
            "total/hgroup gains come from the greedy FittedCorrector "
            "(fit_greedy + corrector_gain), not from build_features"
        )
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


def _internal_split(n: int, embargo: int) -> tuple[int, int]:
    """The 80/20 fit/eval row split with an `embargo` gap between them."""
    n_fit = int(n * 0.8)
    eval_start = n_fit + embargo
    if n_fit < 2 or eval_start >= n:
        raise ValueError(
            f"need more than {eval_start} rows for the 80/20 split with embargo {embargo}, got {n}"
        )
    return n_fit, eval_start


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
    n_fit, eval_start = _internal_split(x.shape[0], embargo)
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
    d = gains[:, :, channels].mean(axis=(1, 2))
    mean, lcb, ucb = mean_ci(d, bandwidth=auto_bandwidth(d, _min_bandwidth(cfg)), level=level)
    return GainReport(
        coordinate=coord,
        delta=mean,
        lcb=lcb,
        ucb=ucb,
        level=level,
        by_horizon=gains[:, :, channels].mean(axis=(0, 2)),
        mu2=float(r.mean() ** 2),
        n_eval=int(seg_b.origins.size),
    )


@dataclass(frozen=True)
class FittedCorrector:
    """A fitted multi-coordinate linear corrector, built by greedy selection.

    The aggregate ("total") corrector is NOT the union of every coordinate's
    features: the union is over-parameterized on low-signal tasks and
    transfers negatively out of sample (measured on lgssm_xchannel, where
    single coordinates gain while the union loses).  Greedy forward
    selection over coordinates, validated on an A-internal time split, keeps
    the corrector inside the same linear class G while staying honest: the
    eval segment (B, cert, or test) never touches the selection.
    """

    coordinates: tuple[Coordinate, ...]
    means: np.ndarray  # [F] feature means from the fit rows
    stds: np.ndarray  # [F] feature stds from the fit rows
    coef: np.ndarray  # [F, H*C] in standardized space
    intercept: np.ndarray  # [H*C]


def _pool_coordinates(
    cfg: HeadroomConfig,
    period_peaks: list[int],
    n_channels: int,
) -> list[Coordinate]:
    """Coordinates eligible for greedy aggregation (horizon fits per-h)."""
    return [
        c for c in enumerate_coordinates(cfg, period_peaks, n_channels)
        if c.kind != "horizon"
    ]


def _min_bandwidth(cfg: HeadroomConfig) -> int:
    """HAC bandwidth floor: the gain series inherits the residual memory.

    The corrector reads lags up to max_lag and targets overlap over H steps,
    so d_o can be autocorrelated out to max_lag + H (a seasonal task with
    period 168 otherwise gets a bandwidth-24 CI that is far too narrow).
    """
    return cfg.max_lag + cfg.horizon


def fit_greedy(
    seg_a: SegmentData,
    cfg: HeadroomConfig,
    period_peaks: list[int],
) -> FittedCorrector:
    """Greedy forward selection of coordinates on the A segment alone."""
    pool = _pool_coordinates(cfg, period_peaks, seg_a.values.shape[1])
    y = seg_a.r.reshape(seg_a.origins.size, -1)
    n_fit, eval_start = _internal_split(y.shape[0], cfg.horizon)

    # Precompute each coordinate's standardized block once, split into the
    # A-internal fit and validation rows; pair-wise gram blocks make every
    # greedy step a cheap assembly + solve instead of a fresh matrix product.
    blocks: dict[str, np.ndarray] = {}
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for coord in pool:
        x_all, _ = build_features(seg_a, coord, cfg)
        mean = x_all[:n_fit].mean(axis=0)
        std = x_all[:n_fit].std(axis=0)
        std[std == 0.0] = 1.0
        blocks[coord.label] = (x_all - mean) / std
        means[coord.label] = mean
        stds[coord.label] = std
    y_fit, y_val = y[:n_fit], y[eval_start:]

    def _val_mse(labels: tuple[str, ...]) -> float:
        if not labels:
            # Intercept-only baseline: predict the fit-part mean.
            return float(np.mean(np.square(y_val - y_fit.mean(axis=0))))
        design_fit = np.concatenate([blocks[label][:n_fit] for label in labels], axis=1)
        design_val = np.concatenate([blocks[label][eval_start:] for label in labels], axis=1)
        gram = design_fit.T @ design_fit
        cross = design_fit.T @ y_fit
        eye = np.eye(design_fit.shape[1])
        best = np.inf
        for alpha in cfg.ridge_alphas:
            coef = np.linalg.solve(gram + alpha * eye, cross)
            # Candidates share the baseline's intercept — the fit-part mean
            # of y — so a nonzero residual mean inflates nobody's MSE.
            resid = y_val - design_val @ coef - y_fit.mean(axis=0)
            best = min(best, float(np.mean(np.square(resid))))
        return best

    baseline_mse = _val_mse(())
    selected: list[str] = []
    remaining = [coord.label for coord in pool]
    while remaining and len(selected) < _GREEDY_MAX_STEPS:
        best_label: str | None = None
        best_mse = baseline_mse
        for label in remaining:
            mse = _val_mse(tuple(selected + [label]))
            if mse < best_mse:
                best_mse, best_label = mse, label
        if best_label is None:
            break
        selected.append(best_label)
        remaining.remove(best_label)
        baseline_mse = best_mse

    # Refit the selected set on all of A with its own alpha choice.
    label_to_coord = {coord.label: coord for coord in pool}
    coords = tuple(label_to_coord[label] for label in selected)
    x_all = (
        np.concatenate([blocks[label] for label in selected], axis=1)
        if selected
        else np.zeros((y.shape[0], 0))
    )
    coef, intercept = _fit_ridge(x_all, y, cfg.ridge_alphas, cfg.horizon)
    feature_means = (
        np.concatenate([means[label] for label in selected])
        if selected
        else np.zeros(x_all.shape[1])
    )
    feature_stds = (
        np.concatenate([stds[label] for label in selected])
        if selected
        else np.ones(x_all.shape[1])
    )
    return FittedCorrector(
        coordinates=coords,
        means=feature_means,
        stds=feature_stds,
        coef=coef,
        intercept=intercept,
    )


def corrector_gain(
    fitted: FittedCorrector,
    seg_b: SegmentData,
    cfg: HeadroomConfig,
    level: float = 0.95,
    *,
    h_scope: tuple[int, int] | None = None,
    label: str | None = None,
) -> GainReport:
    """Held-out gain of a fitted corrector on seg_b (optionally h-scoped)."""
    if fitted.coordinates:
        x_b = np.concatenate(
            [build_features(seg_b, coord, cfg)[0] for coord in fitted.coordinates], axis=1
        )
        y_hat = ((x_b - fitted.means) / fitted.stds) @ fitted.coef + fitted.intercept
    else:
        shape = (seg_b.origins.size, *seg_b.r.shape[1:])
        y_hat = np.broadcast_to(fitted.intercept.reshape(1, *seg_b.r.shape[1:]), shape).copy()
    y_hat = y_hat.reshape(seg_b.r.shape)
    r = seg_b.r
    gains = r * r - (r - y_hat) ** 2
    steps = slice(h_scope[0] - 1, h_scope[1]) if h_scope else slice(None)
    d = gains[:, steps, :].mean(axis=(1, 2))
    mean, lcb, ucb = mean_ci(d, bandwidth=auto_bandwidth(d, _min_bandwidth(cfg)), level=level)
    return GainReport(
        coordinate=Coordinate(
            "hgroup" if h_scope else "total",
            label or "total",
            _COORDINATE_ASPECTS.get("total" if not h_scope else "hgroup", "other"),
            {"h1": h_scope[0], "h2": h_scope[1]} if h_scope else {},
        ),
        delta=mean,
        lcb=lcb,
        ucb=ucb,
        level=level,
        by_horizon=gains.mean(axis=(0, 2)),
        mu2=float(r.mean() ** 2),
        n_eval=int(seg_b.origins.size),
    )


def card_level(
    cfg: HeadroomConfig,
    period_peaks: list[int],
    n_channels: int,
) -> float:
    """Bonferroni-adjusted level for the card's coordinate rows (A-M4)."""
    card_size = len(enumerate_coordinates(cfg, period_peaks, n_channels))
    return 1.0 - cfg.acceptance_alpha / max(card_size, 1)


def all_gains(
    seg_a: SegmentData,
    seg_b: SegmentData,
    cfg: HeadroomConfig,
    period_peaks: list[int],
    level: float = 0.95,
) -> list[GainReport]:
    """total (greedy) + every enumerated coordinate + the two step-length groups.

    The total and both hgroups share ONE greedy corrector fitted on A; the
    hgroups only re-scope the evaluation to their horizon range, which is
    exactly the certification semantics (same corrector, per-scope statistic).
    """
    n_channels = seg_a.values.shape[1]
    fitted = fit_greedy(seg_a, cfg, period_peaks)
    total = corrector_gain(fitted, seg_b, cfg, level)
    coordinates = enumerate_coordinates(cfg, period_peaks, n_channels)
    half = cfg.horizon // 2
    hgroups = [
        corrector_gain(
            fitted, seg_b, cfg, level, h_scope=(1, half), label=f"hgroup[1,{half}]"
        ),
        corrector_gain(
            fitted, seg_b, cfg, level, h_scope=(half + 1, cfg.horizon), label=f"hgroup[{half + 1},{cfg.horizon}]"
        ),
    ]
    coord_gains = [coordinate_gain(coord, seg_a, seg_b, cfg, level=level) for coord in coordinates]
    return [total, *coord_gains, *hgroups]
