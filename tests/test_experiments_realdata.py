from __future__ import annotations

"""Network-free tests for the de-identified real-task loader.

A tiny synthetic hourly CSV (ETT shape: ISO timestamp + 7 channels) stands in
for the real file, so nothing here needs the download.  Every expectation is
reconstructed from chrono_splits on the exact grid that was written, which
pins the split/embargo/prefix layout independently of the loader's internals.
Channel 0 of the grid is 10000 + t, so standardized channel-0 values invert
back to global row indices -- that is what the embargo test checks arrays, not
formulas.
"""

import csv
from datetime import datetime, timedelta
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = REPO_ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.timeidx import chrono_splits, segment_origins

# The loader's default embargo; tests pin it by not passing it.
EMBARGO = 96

_ROWS = 6000
_N_CHANNELS = 7
# ETT-shaped header; the distinctive names let the de-identification test
# prove none of them survive the loader.
_HEADER = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def _load_realdata():
    # experiments/ is deliberately not on the repo pythonpath (only backend/ is),
    # so the loader is imported by file path exactly like the loop will import it.
    spec = importlib.util.spec_from_file_location(
        "rhg_realdata", REPO_ROOT / "experiments" / "realdata.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses resolve string annotations via sys.modules.
    sys.modules["rhg_realdata"] = module
    spec.loader.exec_module(module)
    return module


rd = _load_realdata()


def reduced_cfg() -> HeadroomConfig:
    """Stage0-smoke-style config: burn_in = 31 + 48 = 79, L = 48, H = 24."""
    return HeadroomConfig(
        input_length=48,
        horizon=24,
        lag_segments=((0, 7), (8, 15), (16, 31)),
        fir_len_cap=48,
        beyond_l_lags=2,
        band_period_range=(4, 24),
        budget_candidates=6,
        ridge_alphas=(1.0, 100.0),
    )


def _ett_grid(rows: int) -> np.ndarray:
    """The [rows, 7] grid that gets written, already as the file will hold it.

    Channel 0 is 10000 + t (strictly increasing -> row identity); channels
    1..6 add sines and a trend so train-only standardization has drift to
    preserve on the later splits.  Values are routed through the same
    "%.6f" formatting the writer uses, so the grid equals the file exactly
    and reconstructions can be compared tightly.
    """
    t = np.arange(rows, dtype=np.float64)
    raw = np.empty((rows, _N_CHANNELS))
    raw[:, 0] = 10_000.0 + t
    for c in range(1, _N_CHANNELS):
        raw[:, c] = 10_000.0 + 11.0 * c + 3.0 * np.sin(2.0 * np.pi * t / 24.0 + c) + 0.01 * t
    return np.array([[float(f"{v:.6f}") for v in row] for row in raw])


def _write_raw_csv(path: Path, raw: np.ndarray) -> None:
    """Write an ETT-shaped hourly CSV: ISO timestamp + the given numeric grid."""
    start = datetime(2020, 1, 1)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_HEADER[: raw.shape[1] + 1])
        for t, row in enumerate(raw):
            timestamp = (start + timedelta(hours=t)).isoformat(sep=" ")
            writer.writerow([timestamp] + [f"{v:.6f}" for v in row])


def _load(tmp_path: Path, rows: int = _ROWS) -> tuple:
    """Write the standard grid, load it with the default embargo, return pieces."""
    cfg = reduced_cfg()
    csv_path = tmp_path / "tiny.csv"
    raw = _ett_grid(rows)
    _write_raw_csv(csv_path, raw)
    task = rd.load_real_task(csv_path, cfg)  # default embargo == EMBARGO
    splits = chrono_splits(rows, cfg, EMBARGO)
    return task, cfg, raw, splits


def test_standardization_uses_train_stats_only(tmp_path: Path) -> None:
    task, cfg, raw, splits = _load(tmp_path)
    train = task.train_diag[: task.diag_start]
    np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(train.std(axis=0), 1.0, rtol=1e-9)
    # Exact reconstruction from the train-slice statistics only: the loader
    # must not have touched diag/cert/test rows when computing them.
    mean = raw[splits.train].mean(axis=0)
    std = raw[splits.train].std(axis=0)
    np.testing.assert_allclose(train, (raw[splits.train] - mean) / std, atol=1e-12)
    # Drift is preserved: the test split is several train-sigmas off zero.
    prefix = cfg.burn_in + cfg.input_length
    test_content = task.test[prefix:]
    assert np.abs(test_content.mean(axis=0)).min() > 1.0


def test_task_is_deidentified(tmp_path: Path) -> None:
    task, _, _, _ = _load(tmp_path)
    assert task.name == "real-deidentified"
    assert task.oracle is None
    assert task.structure == {"channels": _N_CHANNELS, "freq": "1h"}
    # The whole exposed surface the loop/agent could see, free of column names.
    surface = json.dumps(task.structure) + task.name
    for original in _HEADER[1:]:
        assert original not in surface


def test_split_layout_matches_stage0_convention(tmp_path: Path) -> None:
    task, cfg, raw, splits = _load(tmp_path)
    n_train = splits.train.stop - splits.train.start
    n_diag = splits.diag.stop - splits.diag.start
    n_cert = splits.cert.stop - splits.cert.start
    n_test = splits.test.stop - splits.test.start
    prefix = cfg.burn_in + cfg.input_length
    mean = raw[splits.train].mean(axis=0)
    std = raw[splits.train].std(axis=0)
    stdized = (raw - mean) / std

    # train_diag: train then diag, contiguous, diag_start == len(train).
    assert task.diag_start == n_train
    assert task.train_diag.shape == (n_train + n_diag, _N_CHANNELS)
    np.testing.assert_allclose(task.train_diag[:n_train], stdized[splits.train], atol=1e-12)
    np.testing.assert_allclose(task.train_diag[n_train:], stdized[splits.diag], atol=1e-12)

    # cert/test: burn_in + L copied prefix rows, then exactly the segment rows.
    assert task.cert.shape == (prefix + n_cert, _N_CHANNELS)
    assert task.test.shape == (prefix + n_test, _N_CHANNELS)
    np.testing.assert_allclose(task.cert[:prefix], stdized[splits.cert.start - prefix : splits.cert.start], atol=1e-12)
    np.testing.assert_allclose(task.cert[prefix:], stdized[splits.cert], atol=1e-12)
    np.testing.assert_allclose(task.test[:prefix], stdized[splits.test.start - prefix : splits.test.start], atol=1e-12)
    np.testing.assert_allclose(task.test[prefix:], stdized[splits.test], atol=1e-12)

    # Standalone convention: first prediction origin at input_length - 1,
    # usable origins from burn_in (the first one lands on the last prefix row,
    # an embargo row whose targets are already inside the segment), and every
    # split has enough usable origins for the statistics downstream.
    first_usable = cfg.input_length - 1 + cfg.burn_in
    assert first_usable == prefix - 1
    assert segment_origins(cfg.input_length - 1, task.cert.shape[0], cfg)[0] == first_usable
    assert segment_origins(task.diag_start, task.train_diag.shape[0], cfg).size >= 100
    assert segment_origins(cfg.input_length - 1, task.cert.shape[0], cfg).size >= 100
    assert segment_origins(cfg.input_length - 1, task.test.shape[0], cfg).size >= 100


def test_embargo_keeps_eval_rows_apart(tmp_path: Path) -> None:
    task, cfg, raw, splits = _load(tmp_path)
    prefix = cfg.burn_in + cfg.input_length
    mean0 = raw[splits.train, 0].mean()
    std0 = raw[splits.train, 0].std()

    def global_rows(std_column: np.ndarray) -> np.ndarray:
        # Channel 0 is 10000 + t, so standardized values invert to row indices.
        return np.rint(std_column * std0 + mean0 - 10_000.0).astype(np.int64)

    train_rows = global_rows(task.train_diag[: task.diag_start, 0])
    diag_rows = global_rows(task.train_diag[task.diag_start :, 0])
    cert_rows = global_rows(task.cert[prefix:, 0])
    test_rows = global_rows(task.test[prefix:, 0])
    assert np.array_equal(train_rows, np.arange(splits.train.start, splits.train.stop))
    assert np.array_equal(diag_rows, np.arange(splits.diag.start, splits.diag.stop))
    assert np.array_equal(cert_rows, np.arange(splits.cert.start, splits.cert.stop))
    assert np.array_equal(test_rows, np.arange(splits.test.start, splits.test.stop))

    # Eval rows are disjoint across adjacent segments with the embargo intact.
    assert cert_rows.min() - diag_rows.max() == EMBARGO + 1
    assert test_rows.min() - cert_rows.max() == EMBARGO + 1
    # The train|diag embargo rows belong to no array: train_diag concatenates
    # the two slices, so inside the array diag starts right after train while
    # the global gap (EMBARGO unshared rows) stays visible.
    assert diag_rows.min() - train_rows.max() == EMBARGO + 1


def test_loader_is_deterministic(tmp_path: Path) -> None:
    cfg = reduced_cfg()
    csv_path = tmp_path / "tiny.csv"
    _write_raw_csv(csv_path, _ett_grid(_ROWS))
    first = rd.load_real_task(csv_path, cfg)
    second = rd.load_real_task(csv_path, cfg)
    assert first.name == second.name
    assert first.diag_start == second.diag_start
    assert first.structure == second.structure
    assert first.oracle is None and second.oracle is None
    np.testing.assert_array_equal(first.train_diag, second.train_diag)
    np.testing.assert_array_equal(first.cert, second.cert)
    np.testing.assert_array_equal(first.test, second.test)


def test_real_task_adapter_inputs(tmp_path: Path) -> None:
    task, cfg, _, _ = _load(tmp_path)
    inputs = rd.real_task_adapter_inputs(task, cfg)
    assert set(inputs) == {
        "train",
        "diag_values",
        "diag_start",
        "diag_stop",
        "cert_values",
        "test_values",
        "standalone_start",
    }
    assert np.array_equal(inputs["train"], task.train_diag[: task.diag_start])
    assert np.array_equal(inputs["diag_values"], task.train_diag)
    assert inputs["diag_start"] == task.diag_start
    assert inputs["diag_stop"] == task.train_diag.shape[0]
    assert np.array_equal(inputs["cert_values"], task.cert)
    assert np.array_equal(inputs["test_values"], task.test)
    # The segment start the loop must pass for cert/test (stage0 convention).
    assert inputs["standalone_start"] == cfg.input_length - 1


def test_short_series_raises_informative_error(tmp_path: Path) -> None:
    cfg = reduced_cfg()
    csv_path = tmp_path / "tiny.csv"
    _write_raw_csv(csv_path, _ett_grid(1050))
    # 1050 rows, no embargo: diag passes (107 usable origins) but cert yields
    # only 105 - 24 + 1 = 82, so the loader must refuse with the numbers.
    with pytest.raises(ValueError, match="usable origins"):
        rd.load_real_task(csv_path, cfg, embargo=0)


def test_rejects_constant_channel_and_nonfinite_values(tmp_path: Path) -> None:
    cfg = reduced_cfg()
    csv_path = tmp_path / "tiny.csv"
    raw = _ett_grid(1600)
    constant = raw.copy()
    constant[:, 2] = 5.0
    _write_raw_csv(csv_path, constant)
    with pytest.raises(ValueError, match="constant on the train split"):
        rd.load_real_task(csv_path, cfg, embargo=0)

    holed = raw.copy()
    holed[10, 1] = np.nan
    _write_raw_csv(csv_path, holed)
    with pytest.raises(ValueError, match="non-finite"):
        rd.load_real_task(csv_path, cfg, embargo=0)
