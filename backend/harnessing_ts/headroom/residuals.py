from __future__ import annotations

"""Segment residual tensors: per-origin residuals plus the full realized e/r series."""

from dataclasses import dataclass

import numpy as np

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.timeidx import segment_origins


@dataclass(frozen=True)
class SegmentData:
    """Realized information of one segment; see headroom-contracts.md §4.

    ``values`` is the segment's array with its prefix history, and every
    other member is indexed with the same (local) time axis.  ``r_full[t]``
    is the residual row of origin ``t`` (defined where all H targets of
    origin ``t`` lie inside the segment, NaN elsewhere), and ``e_full[t]``
    is the latest 1-step residual realized BY time ``t`` — the one made at
    origin ``t-1`` whose target is ``x[t]`` (plan §2.3: ``e_t = r[t-1, 1]``).
    The one-step shift is what keeps the k=0 lag feature inside the realized
    information set: row ``o`` of ``e_full`` never peeks at ``x[o+1]``.
    Features only ever read rows at or before the origin they describe, so
    truncating the arrays at any origin leaves that origin's features
    unchanged.
    """

    values: np.ndarray  # [T, C] the segment's full array, prefix included
    origins: np.ndarray  # [B] usable origins (local indices), ascending
    r: np.ndarray  # [B, H, C] residuals at origins, aligned with origins
    e_full: np.ndarray  # [T, C] realized e series, NaN outside the segment
    r_full: np.ndarray  # [T, H, C] realized residual tensor, NaN likewise


def segment_data(
    values: np.ndarray,
    seg_start: int,
    seg_stop: int,
    pred: np.ndarray,
    cfg: HeadroomConfig,
) -> SegmentData:
    """Build SegmentData from the model's predictions inside one segment.

    ``pred`` carries one row per origin ``t = seg_start .. seg_stop - H - 1``,
    i.e. every origin whose H targets lie inside the segment — not just the
    burn-in restricted usable origins.  The burn-in only limits which
    origins a corrector may be *fit* at; the e/r series must already exist
    from ``seg_start`` on, because the deepest lag at the first usable
    origin (``seg_start + burn_in = seg_start + max_lag + fir_len_cap``)
    reaches back into exactly that prefix, and ``burn_in`` is sized so those
    rows are realized (contract §1).
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"values must be [T, C], got shape {values.shape}")
    t_len, n_channels = values.shape
    if not 0 <= seg_start < seg_stop <= t_len:
        raise ValueError(f"need 0 <= seg_start < seg_stop <= {t_len}, got [{seg_start}, {seg_stop})")
    if seg_stop - seg_start <= cfg.horizon:
        raise ValueError(
            f"segment [{seg_start}, {seg_stop}) is not longer than horizon {cfg.horizon}, "
            "so no origin has all targets inside"
        )
    n_realized = seg_stop - seg_start - cfg.horizon
    pred = np.asarray(pred, dtype=np.float64)
    expected = (n_realized, cfg.horizon, n_channels)
    if pred.shape != expected:
        raise ValueError(
            f"pred must carry one row per origin {seg_start}..{seg_stop - cfg.horizon - 1} "
            f"(every origin whose {cfg.horizon} targets lie inside the segment, so the e/r "
            f"series is realized from seg_start on), expected shape {expected}, got {pred.shape}"
        )
    origins = segment_origins(seg_start, seg_stop, cfg)
    rows = np.arange(seg_start, seg_stop - cfg.horizon)
    targets = values[rows[:, None] + np.arange(1, cfg.horizon + 1)]
    realized = targets - pred  # [n_realized, H, C]
    e_full = np.full((t_len, n_channels), np.nan)
    r_full = np.full((t_len, cfg.horizon, n_channels), np.nan)
    r_full[seg_start : seg_stop - cfg.horizon] = realized
    # e_full[t] = r[origin t-1, h=1]: the residual whose target is x[t], i.e.
    # the latest 1-step residual realized by time t. Realized[i] belongs to
    # origin seg_start+i, so it lands in e_full one row later.
    e_full[seg_start + 1 : seg_stop - cfg.horizon + 1] = realized[:, 0, :]
    return SegmentData(
        values=values,
        origins=origins,
        r=realized[origins - seg_start],
        e_full=e_full,
        r_full=r_full,
    )
