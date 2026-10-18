"""Shared evaluation helpers for headroom experiments (stage0, baselines, loop, posthoc, calibrations).

Bridges the m0/candidate Adapter contract to the headroom core: predictions
at target-realized origins, paired oracle gaps, segment assembly, the DM
acceptance gate, and certification gains.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
import sys
from typing import Any, Callable, Protocol

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from harnessing_ts.headroom.certify import CertReport, certify
from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import (
    FittedCorrector,
    GainReport,
    _min_bandwidth,
    all_gains,
    build_features,
    select_period_peaks,
)
from harnessing_ts.headroom.inference import auto_bandwidth, dm_one_sided_p
from harnessing_ts.headroom.residuals import SegmentData, segment_data
from harnessing_ts.headroom.timeidx import ab_split, windows


class Adapter(Protocol):
    """The candidate/m0 contract: fit on train only, predict in batches."""

    model_id: str

    def fit(self, train: np.ndarray, meta: dict[str, Any]) -> None: ...
    def predict_batch(self, x: np.ndarray) -> np.ndarray: ...


def fit_meta(cfg: HeadroomConfig, channels: int, seed: int) -> dict[str, Any]:
    return {"L": cfg.input_length, "H": cfg.horizon, "C": channels, "seed": seed}


def target_origins(seg_start: int, seg_stop: int, cfg: HeadroomConfig) -> np.ndarray:
    """All origins whose H targets lie fully inside [seg_start, seg_stop)."""
    return np.arange(seg_start, seg_stop - cfg.horizon, dtype=np.int64)


def predict_adapter(
    adapter: Adapter,
    values: np.ndarray,
    seg_start: int,
    seg_stop: int,
    cfg: HeadroomConfig,
    chunk: int | None = 4096,
) -> np.ndarray:
    """Adapter predictions [B, H, C] at target_origins(seg_start, seg_stop).

    Predictions are gathered in blocks of ``chunk`` origins (pass
    ``chunk=None`` for one predict_batch call).  Per-row results are identical
    up to BLAS-order rounding (the Adapter contract's row independence), and
    peak memory drops from one model-sized activation tensor over ALL origins
    to one over ``chunk`` of them — the deep incumbent's float64 forward at
    full test sizes otherwise allocates multi-GB activations in one call, so
    chunking is the DEFAULT; the loop, worker, posthoc, baselines and
    calibrations all inherit it.
    """
    origins = target_origins(seg_start, seg_stop, cfg)
    expected = (origins.size, cfg.horizon, values.shape[1])
    if chunk is None:
        x, _ = windows(values, origins, cfg)
        pred = np.asarray(adapter.predict_batch(x), dtype=np.float64)
    else:
        if isinstance(chunk, bool) or not isinstance(chunk, (int, np.integer)) or chunk < 1:
            raise ValueError(f"chunk must be a positive int, got {chunk!r}")
        pred = np.empty(expected, dtype=np.float64)
        for begin in range(0, origins.size, chunk):
            block = origins[begin : begin + chunk]
            x, _ = windows(values, block, cfg)
            pred[begin : begin + block.size] = adapter.predict_batch(x)
    if pred.shape != expected:
        raise RuntimeError(f"adapter {adapter.model_id} returned shape {pred.shape}, expected {expected}")
    return pred


def predict_oracle(
    task: Any, values: np.ndarray, seg_start: int, seg_stop: int, cfg: HeadroomConfig
) -> np.ndarray:
    """Oracle predictions at target_origins, filtering from the full array.

    The oracle sees the whole values array (full information), so for the diag
    segment it also uses the train prefix.
    """
    origins = target_origins(seg_start, seg_stop, cfg)
    pred = np.asarray(task.oracle.forecast(values, origins, cfg), dtype=np.float64)
    expected = (origins.size, cfg.horizon, values.shape[1])
    if pred.shape != expected:
        raise RuntimeError(f"oracle returned shape {pred.shape}, expected {expected}")
    return pred


def build_segment(
    values: np.ndarray, seg_start: int, seg_stop: int, pred: np.ndarray, cfg: HeadroomConfig
) -> SegmentData:
    """SegmentData from a pred aligned with target_origins(seg_start, seg_stop)."""
    return segment_data(values, seg_start, seg_stop, pred, cfg)


def predict_segments(
    predict: Callable[[np.ndarray, int, int], np.ndarray],
    segments: dict[str, tuple[np.ndarray, int, int]],
    cfg: HeadroomConfig,
    *,
    precomputed: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, SegmentData], dict[str, np.ndarray]]:
    """Predict and assemble every named segment.

    ``segments`` maps a name to (values, seg_start, seg_stop); ``predict``
    returns [B, H, C] predictions for the realized origins of one such triple.
    An entry in ``precomputed`` replaces the call for that segment (its rows
    must cover every realized origin, like predict_adapter's return).  Returns
    ({name: SegmentData}, {name: predictions at the segment's usable origins}).
    """
    precomputed = precomputed or {}
    built: dict[str, SegmentData] = {}
    usable: dict[str, np.ndarray] = {}
    for name, (values, start, stop) in segments.items():
        pred = precomputed.get(name)
        if pred is None:
            pred = predict(values, start, stop)
        built[name] = build_segment(values, start, stop, pred, cfg)
        usable[name] = pred[built[name].origins - start]
    return built, usable


def fit_predict_segments(
    adapter: Adapter,
    train: np.ndarray,
    cfg: HeadroomConfig,
    seed: int,
    segments: dict[str, tuple[np.ndarray, int, int]],
    *,
    chunk: int | None = 4096,
    precomputed: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, SegmentData], dict[str, np.ndarray]]:
    """Fit ``adapter`` on train (fit_meta at ``seed``), then predict_segments.

    ``precomputed`` carries predictions already computed for some segments by
    an adapter that is already fitted: the fit is skipped and those segments
    keep the given predictions.
    """
    if precomputed is None:
        adapter.fit(train, fit_meta(cfg, train.shape[1], seed))
    return predict_segments(
        partial(predict_adapter, adapter, cfg=cfg, chunk=chunk), segments, cfg, precomputed=precomputed
    )


def diag_a_peaks(seg: SegmentData, cfg: HeadroomConfig) -> list[int]:
    """Period peaks selected on the diag-A (fit) half of a diagnostic segment."""
    seg_a, _ = split_ab(seg, cfg)
    return select_period_peaks(seg_a.e_full, seg_a.origins, cfg)


def restrict(seg: SegmentData, origins_subset: np.ndarray) -> SegmentData:
    """Subset a segment's origins (and aligned residual rows) to a sorted subset."""
    origins_subset = np.asarray(origins_subset, dtype=np.int64)
    pos = np.searchsorted(seg.origins, origins_subset)
    if np.any(pos >= seg.origins.size) or not np.array_equal(seg.origins[pos], origins_subset):
        raise ValueError("restrict: origins_subset must be a sorted subset of the segment origins")
    return replace(seg, origins=origins_subset, r=seg.r[pos])


def split_ab(seg: SegmentData, cfg: HeadroomConfig) -> tuple[SegmentData, SegmentData]:
    """diag-A (fit) / diag-B (evaluate) halves with an H-step gap."""
    a_origins, b_origins = ab_split(seg.origins, cfg.diag_a_ratio, cfg.horizon)
    return restrict(seg, a_origins), restrict(seg, b_origins)


def paired_gap(
    values: np.ndarray,
    origins: np.ndarray,
    pred: np.ndarray,
    oracle_pred: np.ndarray,
    cfg: HeadroomConfig,
) -> float:
    """Paired optimality gap: MSE(pred) - MSE(oracle) on the same origins."""
    _, y = windows(values, origins, cfg)
    return float(np.mean(np.square(y - pred)) - np.mean(np.square(y - oracle_pred)))


def mse_by_group(seg: SegmentData, gains: list[GainReport]) -> dict[str, float]:
    """Per-label residual MSE on a segment, matching certify's statistic family.

    The denominator for each statistic is the MSE of the residual scope that
    coordinate can fix: hgroup over its horizon range, channel->i over channel
    i, and everything else (lag/band/beyond_l/horizon/total) over all cells.
    """
    out: dict[str, float] = {}
    for gain in gains:
        label = gain.coordinate.label
        if label.startswith("hgroup["):
            h1, h2 = (int(part) for part in label[len("hgroup[") : -1].split(","))
            out[label] = float(np.mean(np.square(seg.r[:, h1 - 1 : h2, :])))
        elif label.startswith("channel->"):
            channel = int(label[len("channel->") :])
            out[label] = float(np.mean(np.square(seg.r[:, :, channel])))
        else:
            out[label] = float(np.mean(np.square(seg.r)))
    return out


def dm_acceptance(
    seg_parent: SegmentData, seg_candidate: SegmentData, cfg: HeadroomConfig
) -> dict[str, Any]:
    """The loop's acceptance gate: one-sided DM (HAC + HLN) at acceptance_alpha.

    ``d`` is the paired per-origin loss-difference series
    mean(parent^2) - mean(candidate^2); the return carries the p-value, the
    accept decision, and d's diagnostics.
    """
    d = (np.square(seg_parent.r) - np.square(seg_candidate.r)).mean(axis=(1, 2))
    bandwidth = int(auto_bandwidth(d, _min_bandwidth(cfg)))
    dm_p = float(dm_one_sided_p(d, bandwidth))
    return {
        "dmP": dm_p,
        "accepted": bool(dm_p < cfg.acceptance_alpha),
        "meanD": float(d.mean()),
        "bandwidth": bandwidth,
        "nOrigins": int(d.size),
    }


def certification(
    seg_diag: SegmentData, seg_cert: SegmentData, cfg: HeadroomConfig, peaks: list[int]
) -> tuple[CertReport, list[GainReport], dict[str, float]]:
    """The certification statistic family: (certify report, gains, group MSEs).

    Gains are computed at the fixed level 1 - certify_alpha (a two-sided CI,
    so each UCB is one-sided conservative); certify() derives the same level
    from cfg.certify_alpha and records it in the report.
    """
    gains = all_gains(seg_diag, seg_cert, cfg, peaks, level=1.0 - cfg.certify_alpha)
    mse = mse_by_group(seg_cert, gains)
    return certify(gains, mse, cfg), gains, mse


def corrector_predictions(
    fitted: FittedCorrector, seg: SegmentData, cfg: HeadroomConfig
) -> np.ndarray:
    """The fitted corrector's residual prediction at seg.origins, [B, H, C].

    Mirrors corrector_gain's application exactly (same standardization and
    intercept, including the intercept-only empty selection), so gains read
    off these predictions match corrector_gain.
    """
    if fitted.coordinates:
        x = np.concatenate(
            [build_features(seg, coord, cfg)[0] for coord in fitted.coordinates], axis=1
        )
        y_hat = ((x - fitted.means) / fitted.stds) @ fitted.coef + fitted.intercept
    else:
        shape = (seg.origins.size, *seg.r.shape[1:])
        y_hat = np.broadcast_to(fitted.intercept.reshape(1, *seg.r.shape[1:]), shape)
    return np.asarray(y_hat, dtype=np.float64).reshape(seg.r.shape)


def gain_row(gain: GainReport) -> dict[str, Any]:
    """One gain as the JSON card row of a heldout posthoc card."""
    return {
        "label": gain.coordinate.label,
        "aspect": gain.coordinate.aspect,
        "delta": gain.delta,
        "lcb": gain.lcb,
        "ucb": gain.ucb,
        "mu2": gain.mu2,
        "by_horizon": gain.by_horizon,
    }


def aspect_lcb(card_rows: list[dict[str, Any]] | None) -> dict[str, float]:
    """Per aspect, the strongest remaining-improvement bound in a card."""
    strongest: dict[str, float] = {}
    if not card_rows:
        return strongest
    for row in card_rows:
        aspect = str(row["aspect"])
        lcb = row.get("lcb")
        value = -np.inf if lcb is None else float(lcb)
        strongest[aspect] = max(strongest.get(aspect, -np.inf), value)
    return strongest


def m0_fit_windows(
    train: np.ndarray, lookback: int, horizon: int, origins: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """m0's fit rows at ``origins``: (win [N, K, C] oldest-first,
    z = win - x[o], y = x[o+h] - x[o])."""
    win = train[origins[:, None] + np.arange(-lookback + 1, 1)[None, :]]
    last = train[origins][:, None, :]
    y = train[origins[:, None] + np.arange(1, horizon + 1)[None, :]] - last
    return win, win - last, y


def ridge_weights(design: np.ndarray, targets: np.ndarray, reg: float) -> np.ndarray:
    """Channel-independent closed-form ridge (m0's solver).

    ``design`` [N, F, C] and ``targets`` [N, H, C] give weights [F, H, C]:
    one solve per channel; channels never see each other.
    """
    eye = np.eye(design.shape[1])
    weights = np.empty((design.shape[1], targets.shape[1], design.shape[2]), dtype=np.float64)
    for c in range(design.shape[2]):
        d = design[:, :, c]
        weights[:, :, c] = np.linalg.solve(d.T @ d + reg * eye, d.T @ targets[:, :, c])
    return weights
