"""Shared evaluation helpers for headroom experiments (stage0, baselines, loop).

Bridges the m0/candidate Adapter contract to the headroom core: predictions
at target-realized origins, paired oracle gaps, and segment assembly.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any, Protocol

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import GainReport
from harnessing_ts.headroom.residuals import SegmentData, segment_data
from harnessing_ts.headroom.timeidx import ab_split, segment_origins, windows


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
