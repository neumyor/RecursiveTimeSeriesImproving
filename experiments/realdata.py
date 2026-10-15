from __future__ import annotations

"""De-identified real-task loader for the residual-headroom experiments.

Loads an ETT-shaped hourly CSV (timestamp column first, then C numeric
channels), standardizes it with TRAIN-only statistics (final plan §2.1), and
cuts it into the same layout stage0 uses for SyntheticTask, so RealTask drops
into evaluate_cell exactly where SyntheticTask sits:

- ``train_diag`` is one contiguous array: train = [0, diag_start), diag =
  [diag_start, T), diag starts right after train.
- ``cert``/``test`` are self-contained arrays whose first ``burn_in +
  input_length`` rows are copied history (the tail rows immediately before the
  segment, embargo rows included).  Downstream applies stage0's standalone
  convention to them: first prediction origin at index ``input_length - 1``,
  usable origins from ``burn_in`` on.

``chrono_splits`` fixes the global train/diag/cert/test ranges and inserts
``embargo`` unassigned rows before diag, cert and test.  Only the diag|cert and
cert|test gaps matter for the statistics (correctors are fit on diag and
certified on cert; cert is in turn predictive history for test), and they are
fully preserved: no segment's eval rows ever overlap an adjacent segment's.
The train|diag embargo rows belong to no array -- ``train_diag`` concatenates
the two slices, which is what keeps ``diag_start == len(train)``; train is only
ever a fit set, so dropping its boundary buffer costs nothing but the seam it
avoids in the concatenated array.

De-identification is structural: the header is read and discarded, channels are
just columns 0..C-1 (``ch_i`` downstream), and nothing but the sampling
frequency survives the timestamp column.  Original column names never leave
``_parse_csv``, so no artifact handed to the agent can carry the dataset
identity (final plan §4; minipaper's real-task role: the LLM must not recognize
the dataset, and the paired optimality gap is undefined because real tasks have
no oracle).

Tests run without the network on a tiny synthetic CSV in tmp_path
(tests/test_experiments_realdata.py); the real file lives untracked in
experiments/data/ (see its .gitignore).
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.timeidx import chrono_splits, segment_origins

MIN_USABLE_ORIGINS = 100


@dataclass(frozen=True)
class RealTask:
    """One de-identified real task in the SyntheticTask layout, minus the oracle.

    ``train_diag`` is the standardized contiguous train+diag block; ``cert``
    and ``test`` are standardized arrays chronologically after it, each with
    ``burn_in + input_length`` copied prefix rows in front (the tail rows
    before the segment), so the standalone-segment convention applies to them
    verbatim.  ``oracle`` is None: real tasks have no full-information
    reference, hence no paired optimality gap.  ``structure`` carries only the
    channel count and the sampling frequency -- never a dataset name.
    """

    name: str
    train_diag: np.ndarray  # [T, C] standardized, contiguous train+diag
    diag_start: int  # == len(train); diag = [diag_start, T)
    cert: np.ndarray  # standardized, chronologically after train_diag
    test: np.ndarray  # standardized, chronologically after cert
    oracle: None
    structure: dict[str, Any]  # {"channels": C, "freq": "1h"} - no dataset name


def _is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _sampling_freq(first: str, second: str | None) -> str:
    """Coarse sampling interval ("1h", "15min", "30s") from the first two rows."""
    if second is None:
        return "unknown"
    delta = datetime.fromisoformat(second) - datetime.fromisoformat(first)
    seconds = int(round(delta.total_seconds()))
    if seconds <= 0:
        raise ValueError(f"timestamps must be strictly increasing, got {first!r} then {second!r}")
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}min"
    return f"{seconds}s"


def _parse_csv(csv_path: Path) -> tuple[np.ndarray, str]:
    """Read the CSV as a [T, C] float64 grid plus the sampling-frequency string.

    The first row is always the header; it is read and discarded immediately
    (the de-identification step -- nothing keys on column names, and none
    survive).  The first column is the timestamp column when it does not parse
    as a number (ETT convention); only the spacing of its first two values
    survives, as a frequency string.  Files without a timestamp column are
    treated as all-channel grids with an "unknown" frequency.
    """
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # header: read and discard
        first = next(reader, None)
        second = next(reader, None)
    if first is None:
        raise ValueError(f"{csv_path} is empty or has no data rows below the header")
    n_cols = len(first)
    if _is_float(first[0]):
        return np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2), "unknown"
    if n_cols < 2:
        raise ValueError(f"{csv_path} has a timestamp column but no channel columns")
    freq = _sampling_freq(first[0], second[0] if second is not None else None)
    values = np.loadtxt(
        csv_path, delimiter=",", skiprows=1, usecols=range(1, n_cols), ndmin=2, encoding="utf-8"
    )
    return values, freq


def _require_usable(name: str, seg_start: int, seg_stop: int, cfg: HeadroomConfig) -> None:
    """Fail early when an eval split cannot support cards/HAC/certification.

    Every downstream statistic (coordinate gains, DM acceptance, certification)
    consumes usable origins, so a split with fewer than MIN_USABLE_ORIGINS of
    them is a misconfigured task, not a runtime surprise later.
    """
    try:
        count = segment_origins(seg_start, seg_stop, cfg).size
    except ValueError as exc:
        raise ValueError(f"{name} split is too short: {exc}") from exc
    if count < MIN_USABLE_ORIGINS:
        raise ValueError(
            f"{name} split yields only {count} usable origins, need at least {MIN_USABLE_ORIGINS} "
            f"(burn_in={cfg.burn_in}, horizon={cfg.horizon}; for cert/test the copied prefix "
            "covers the burn-in)"
        )


def load_real_task(csv_path: Path, cfg: HeadroomConfig, *, embargo: int = 96) -> RealTask:
    """Load one de-identified real task: parse, train-standardize, split, validate.

    ``embargo`` is the buffer ``chrono_splits`` inserts between adjacent
    segments; see the module docstring for how the three gaps are used.
    Raises ValueError with the offending numbers whenever the series is too
    short to yield a usable task.
    """
    values, freq = _parse_csv(Path(csv_path))
    if not np.isfinite(values).all():
        raise ValueError("CSV contains non-finite values; the loader needs a complete numeric grid")
    n_rows, channels = values.shape

    splits = chrono_splits(n_rows, cfg, embargo)

    # Train-only standardization (final plan §2.1): later splits keep their
    # drift -- a mean-shifted test split is part of the real task's difficulty.
    train_rows = values[splits.train]
    mean = train_rows.mean(axis=0)
    std = train_rows.std(axis=0)
    if np.any(std == 0.0):
        raise ValueError(
            f"{int(np.count_nonzero(std == 0.0))} of {channels} channels are constant on the "
            "train split; per-channel standardization is impossible"
        )
    values = (values - mean) / std

    prefix = cfg.burn_in + cfg.input_length
    if splits.train.stop < cfg.input_length - 1:
        raise ValueError(
            f"train split is {splits.train.stop} rows; the first diag origin needs an input "
            f"window of input_length - 1 = {cfg.input_length - 1} history rows before diag"
        )
    if splits.cert.start < prefix or splits.test.start < prefix:
        raise ValueError(
            f"not enough history before cert/test for the standalone prefix of "
            f"burn_in + input_length = {prefix} rows (cert starts at row {splits.cert.start}, "
            f"test at row {splits.test.start})"
        )
    # Usable-origin counts on the actual arrays that downstream will see: diag
    # lives inside train_diag, cert/test are the standalone arrays.
    _require_usable("diag", splits.train.stop, splits.train.stop + (splits.diag.stop - splits.diag.start), cfg)
    _require_usable("cert", cfg.input_length - 1, prefix + (splits.cert.stop - splits.cert.start), cfg)
    _require_usable("test", cfg.input_length - 1, prefix + (splits.test.stop - splits.test.start), cfg)

    # train_diag concatenates the two slices (diag right after train,
    # diag_start == len(train)); the train|diag embargo rows belong to no
    # array.  cert/test copy the tail rows before them as prefix history.
    train_diag = np.concatenate([values[splits.train], values[splits.diag]])
    cert = values[splits.cert.start - prefix : splits.cert.stop]
    test = values[splits.test.start - prefix : splits.test.stop]
    return RealTask(
        name="real-deidentified",
        train_diag=train_diag,
        diag_start=splits.train.stop - splits.train.start,
        cert=cert,
        test=test,
        oracle=None,
        structure={"channels": int(channels), "freq": freq},
    )


def real_task_adapter_inputs(task: RealTask, cfg: HeadroomConfig) -> dict[str, Any]:
    """The pieces the loop's evaluation helpers need, in stage0's convention.

    ``train`` is the adapter's fit set (train rows only); ``diag_values`` is
    the array diag windows read from (its train prefix is realized past);
    ``standalone_start`` is the segment start for cert/test -- the same value
    stage0.evaluate_cell passes for SyntheticTask segments.
    """
    return {
        "train": task.train_diag[: task.diag_start],
        "diag_values": task.train_diag,
        "diag_start": task.diag_start,
        "diag_stop": task.train_diag.shape[0],
        "cert_values": task.cert,
        "test_values": task.test,
        "standalone_start": cfg.input_length - 1,
    }
