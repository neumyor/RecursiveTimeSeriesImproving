"""Time-index conventions: origin semantics, chronological splits, windows.

An origin ``o`` is the index of the LAST observation of the input window.
The input is ``values[o-L+1 .. o]`` and the targets are ``values[o+1 .. o+H]``.
This is the convention every other headroom module relies on; see
docs/headroom-contracts.md §1 and §3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from harnessing_ts.headroom.config import HeadroomConfig


@dataclass(frozen=True)
class Splits:
    """Chronological index ranges into a single values array; test is last."""

    train: slice
    diag: slice
    cert: slice
    test: slice


def chrono_splits(values_len: int, cfg: HeadroomConfig, embargo: int) -> Splits:
    """Split ``values_len`` points into train/diag/cert/test plus embargo gaps.

    The embargo steps are unassigned buffer rows inserted before the diag,
    cert, and test segments so adjacent segments never share boundary rows.
    """
    if values_len <= 0:
        raise ValueError(f"values_len must be positive, got {values_len}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")
    n_train = int(values_len * cfg.train_ratio)
    n_diag = int(values_len * cfg.diag_ratio)
    n_cert = int(values_len * cfg.cert_ratio)
    n_test = values_len - n_train - n_diag - n_cert - 3 * embargo
    if n_test <= cfg.horizon:
        raise ValueError(
            f"values_len={values_len} leaves only {n_test} test rows after "
            f"train={n_train}, diag={n_diag}, cert={n_cert} and embargo={embargo}; "
            "need more data or smaller ratios"
        )
    cursor = 0
    train = slice(cursor, cursor + n_train)
    cursor += n_train + embargo
    diag = slice(cursor, cursor + n_diag)
    cursor += n_diag + embargo
    cert = slice(cursor, cursor + n_cert)
    cursor += n_cert + embargo
    test = slice(cursor, values_len)
    return Splits(train=train, diag=diag, cert=cert, test=test)


def segment_origins(seg_start: int, seg_stop: int, cfg: HeadroomConfig) -> np.ndarray:
    """Usable origins for the segment ``[seg_start, seg_stop)``.

    An origin is usable when every realized-residual feature at ``o`` can be
    built from inside the segment (``o >= seg_start + burn_in``) and all H
    targets lie inside the segment (``o + H <= seg_stop - 1``).
    """
    if seg_stop <= seg_start:
        raise ValueError(f"segment must be non-empty, got [{seg_start}, {seg_stop})")
    first = seg_start + cfg.burn_in
    last = seg_stop - cfg.horizon - 1
    if last < first:
        raise ValueError(
            f"segment [{seg_start}, {seg_stop}) has no usable origins: need at least "
            f"burn_in + horizon + 1 = {cfg.burn_in + cfg.horizon + 1} rows, got {seg_stop - seg_start}"
        )
    return np.arange(first, last + 1, dtype=np.int64)


def windows(values: np.ndarray, origins: np.ndarray, cfg: HeadroomConfig) -> tuple[np.ndarray, np.ndarray]:
    """Gather (X[B, L, C], Y[B, H, C]) at the given origins.

    Input windows may extend before ``seg_start`` into earlier rows of
    ``values`` (realized past); only the targets must stay inside the segment,
    which ``segment_origins`` already guarantees for its callers.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"values must be [T, C], got shape {values.shape}")
    origins = np.asarray(origins, dtype=np.int64)
    L, H = cfg.input_length, cfg.horizon
    if origins.size == 0:
        return np.empty((0, L, values.shape[1])), np.empty((0, H, values.shape[1]))
    if origins.min() < L - 1:
        raise ValueError(f"origin {origins.min()} has an input window before the start of values")
    if origins.max() + H > values.shape[0] - 1:
        raise ValueError(f"origin {origins.max()} has targets beyond the end of values")
    x_idx = origins[:, None] + np.arange(-(L - 1), 1, dtype=np.int64)[None, :]
    y_idx = origins[:, None] + np.arange(1, H + 1, dtype=np.int64)[None, :]
    return values[x_idx], values[y_idx]


def ab_split(origins: np.ndarray, a_ratio: float, embargo: int) -> tuple[np.ndarray, np.ndarray]:
    """Split sorted origins into A (fit) and B (evaluate) with a gap of ``embargo``."""
    origins = np.asarray(origins, dtype=np.int64)
    if origins.size < 4:
        raise ValueError(f"need at least 4 origins to split, got {origins.size}")
    if not 0.0 < a_ratio < 1.0:
        raise ValueError(f"a_ratio must be in (0, 1), got {a_ratio}")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}")
    n_a = int(origins.size * a_ratio)
    n_b = origins.size - n_a - embargo
    if n_b < 2 or n_a < 2:
        raise ValueError(
            f"cannot split {origins.size} origins with a_ratio={a_ratio} and embargo={embargo}"
        )
    return origins[:n_a].copy(), origins[n_a + embargo :].copy()
