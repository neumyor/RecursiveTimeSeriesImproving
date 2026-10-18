from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.signal import lfilter

from harnessing_ts.headroom.candidate import (
    CANDIDATE_ID_PATTERN,
    CandidateCrash,
    CandidateError,
    CandidateTimeout,
    CheatDetected,
    ContractViolation,
    SourceRejected,
    build_env,
    check_source,
    evaluate_candidate,
    smoke_test,
)
from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.timeidx import windows

L, H, C = 32, 4, 2
CFG = HeadroomConfig(input_length=L, horizon=H)
TRAIN_END = 800

RIDGE_SOURCE = """import numpy as np


def create_adapter():
    return RidgeAdapter()


class RidgeAdapter:
    def __init__(self):
        self.model_id = "ridge-e2e"

    def fit(self, train, meta):
        L, H = int(meta["L"]), int(meta["H"])
        origins = np.arange(L - 1, train.shape[0] - H)
        x = np.stack([train[o - L + 1 : o + 1].reshape(-1) for o in origins])
        y = np.stack([train[o + 1 : o + 1 + H].reshape(-1) for o in origins])
        self.weights = np.linalg.solve(x.T @ x + np.eye(x.shape[1]), x.T @ y)
        self.L, self.H, self.C = L, H, train.shape[1]

    def predict_batch(self, X):
        B = X.shape[0]
        return (X.reshape(B, -1) @ self.weights).reshape(B, self.H, self.C)
"""

CHEAT_SOURCE = """import numpy as np


def create_adapter():
    return CheatAdapter()


class CheatAdapter:
    def __init__(self):
        self.model_id = "cheat-stats"

    def fit(self, train, meta):
        self.H = int(meta["H"])

    def predict_batch(self, X):
        # Peeks at batch statistics: every row's forecast reads the other rows
        # in its batch — exactly what the single-row probe must catch.
        level = X.mean(axis=0)[-1]
        return np.tile(level, (X.shape[0], self.H, 1))
"""

SLOW_SOURCE = """import numpy as np


def create_adapter():
    return SlowAdapter()


class SlowAdapter:
    def __init__(self):
        self.model_id = "slow-timeout"

    def fit(self, train, meta):
        while True:  # time is not an importable module for candidates; burn CPU instead
            pass

    def predict_batch(self, X):
        return np.zeros((X.shape[0], 4, X.shape[2]))
"""

CRASH_SOURCE = """import numpy as np


def create_adapter():
    return CrashAdapter()


class CrashAdapter:
    def __init__(self):
        self.model_id = "crash-boom"

    def fit(self, train, meta):
        raise RuntimeError("boom: candidate fit exploded")

    def predict_batch(self, X):
        return np.zeros((X.shape[0], 4, X.shape[2]))
"""

BAD_SHAPE_SOURCE = """import numpy as np


def create_adapter():
    return FlatAdapter()


class FlatAdapter:
    def __init__(self):
        self.model_id = "bad-shape"

    def fit(self, train, meta):
        pass

    def predict_batch(self, X):
        return np.zeros((X.shape[0], 2))
"""


def ar_series(n: int, seed: int = 0) -> np.ndarray:
    """AR(2) with per-channel means: learnable, non-trivial, cheap to generate."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, 2)) * 0.3
    values = lfilter([1.0], [1.0, -0.9, 0.5], noise, axis=0)
    return values + np.array([0.5, -1.0])


def make_meta() -> dict[str, Any]:
    # Same keys the loop builds (experiments/evaluate.py::fit_meta).
    return {"L": L, "H": H, "C": C, "seed": 0}


def interleaved(n_batches: int, per_batch: int, first: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Diag rows in interleaved batch order; same-batch rows are `stride` apart.

    Row order alternates batches ((round, batch) major), and batch b's rows
    sit at first + stride*j + 13*b, so the intra-batch origin spacing is
    exactly `stride`.  With stride >= L + H no same-batch row's window can
    contain another's targets — the structural anti-peeking guarantee
    documented in candidate.py.
    """
    origins = np.asarray(
        [first + stride * j + 13 * b for j in range(per_batch) for b in range(n_batches)],
        dtype=np.int64,
    )
    batch_ids = np.asarray([b for _ in range(per_batch) for b in range(n_batches)], dtype=np.int64)
    return origins, batch_ids


def write_npz_input(
    path: Path,
    source_path: Path,
    series: np.ndarray,
    train_end: int,
    diag_origins: np.ndarray,
    diag_batch_ids: np.ndarray,
    single_check: np.ndarray,
) -> None:
    """Build npz_in exactly as the loop will (candidate.py docstring contract)."""
    x_diag, _ = windows(series, diag_origins, CFG)
    np.savez(
        path,
        train=series[:train_end],
        meta=np.array(json.dumps(make_meta())),
        x_diag=x_diag,
        batch_ids_diag=diag_batch_ids.astype(np.int32),
        single_check_idx=single_check.astype(np.int32),
        source_path=np.array(str(source_path)),
    )


def write_standard_inputs(source: Path, series: np.ndarray, source_text: str, tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    """e2e-shaped inputs: 60 diag rows in 3 interleaved batches, 10 single-check rows."""
    source.write_text(source_text, encoding="utf-8")
    diag_origins, batch_ids = interleaved(3, 20, TRAIN_END, L + H)
    single_check = np.arange(0, 60, 6, dtype=np.int64)
    npz_in, npz_out = tmp_path / "in.npz", tmp_path / "out.npz"
    write_npz_input(npz_in, source, series, TRAIN_END, diag_origins, batch_ids, single_check)
    return npz_in, npz_out, diag_origins, batch_ids


def write_small_inputs(source: Path, series: np.ndarray, source_text: str, tmp_path: Path) -> tuple[Path, Path]:
    """Minimal inputs for failure-path tests: 6 diag rows, 2 single-check rows."""
    source.write_text(source_text, encoding="utf-8")
    diag_origins, batch_ids = interleaved(2, 3, 50, L + H)
    single_check = np.asarray([0, 3], dtype=np.int64)
    npz_in, npz_out = tmp_path / "in.npz", tmp_path / "out.npz"
    write_npz_input(npz_in, source, series, 100, diag_origins, batch_ids, single_check)
    return npz_in, npz_out


# ---------------------------------------------------------------------------
# check_source
# ---------------------------------------------------------------------------


def test_check_source_accepts_whitelisted_model() -> None:
    check_source(RIDGE_SOURCE)  # must not raise
    check_source(
        "import math\nimport numpy as np\nimport torch\nimport torch.nn as nn\n"
        "from scipy.signal import lfilter\nfrom scipy import stats\n\n\n"
        "def create_adapter():\n    return 1\n"
    )


@pytest.mark.parametrize(
    "fragment,source",
    [
        ("import of 'os'", "import numpy as np\nimport os\n\n\ndef create_adapter():\n    return 1\n"),
        ("import of 'requests'", "import numpy as np\n\n\nimport requests\n\n\ndef create_adapter():\n    return 1\n"),
        ("use of 'open'", "import numpy as np\n\n\ndef create_adapter():\n    open('leak.txt')\n    return 1\n"),
        ("call to .save()", "import numpy as np\n\n\ndef create_adapter():\n    np.save('leak.npy', 1)\n    return 1\n"),
        ("use of 'exec'", "import numpy as np\n\n\ndef create_adapter():\n    exec('pass')\n    return 1\n"),
        ("use of '__import__'", "import numpy as np\n\n\ndef create_adapter():\n    __import__('os')\n    return 1\n"),
        ("must define a top-level create_adapter()", "import numpy as np\n\n\ndef not_the_factory():\n    return 1\n"),
        ("relative imports are not allowed", "import numpy as np\n\nfrom . import sibling\n\n\ndef create_adapter():\n    return 1\n"),
        ("syntax error", "def broken(:\n"),
    ],
)
def test_check_source_rejections(fragment: str, source: str) -> None:
    with pytest.raises(SourceRejected) as exc_info:
        check_source(source)
    assert fragment in str(exc_info.value)


def test_check_source_reports_line_numbers() -> None:
    with pytest.raises(SourceRejected) as exc_info:
        check_source("import numpy as np\n\n\ndef create_adapter():\n    np.loadtxt('leak.csv')\n    return 1\n")
    assert "line 5: call to .loadtxt()" in str(exc_info.value)


def test_exception_categories_are_candidate_errors() -> None:
    for category in (SourceRejected, CandidateTimeout, CandidateCrash, CheatDetected, ContractViolation):
        assert issubclass(category, CandidateError)
        assert issubclass(category, RuntimeError)  # headroom-contracts.md §1


def test_candidate_id_pattern() -> None:
    assert CANDIDATE_ID_PATTERN.fullmatch("c07")
    assert CANDIDATE_ID_PATTERN.fullmatch("m0")
    assert CANDIDATE_ID_PATTERN.fullmatch("a" * 64)
    for bad in ("", "-x", "Upper", "c_7", "c x", "a" * 65):
        assert CANDIDATE_ID_PATTERN.fullmatch(bad) is None


# ---------------------------------------------------------------------------
# smoke_test (isolated worker subprocess)
# ---------------------------------------------------------------------------


def test_smoke_test_reports_shape_and_no_metrics(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    source.write_text(RIDGE_SOURCE, encoding="utf-8")
    train = ar_series(400)
    result = smoke_test(source, "ridge-e2e", train, make_meta(), CFG)
    assert result["ok"] is True
    assert result["error"] is None
    assert result["predict_shape"] == (4, H, C)

    bad = smoke_test(source, "other-id", train, make_meta(), CFG)
    assert bad["ok"] is False
    assert bad["predict_shape"] is None
    assert "model_id" in str(bad["error"])


def test_smoke_reports_contract_breaks_as_repair_text(tmp_path: Path) -> None:
    train = ar_series(400)

    no_factory = tmp_path / "no_factory.py"
    no_factory.write_text("import numpy as np\n\n\nclass Adapter:\n    pass\n", encoding="utf-8")
    result = smoke_test(no_factory, "no-factory", train, make_meta(), CFG)
    assert result["ok"] is False
    assert "create_adapter" in str(result["error"])

    no_predict = tmp_path / "no_predict.py"
    no_predict.write_text(
        "import numpy as np\n\n\ndef create_adapter():\n    return A()\n\n\n"
        "class A:\n    def __init__(self):\n        self.model_id = 'no-predict'\n\n"
        "    def fit(self, train, meta):\n        pass\n",
        encoding="utf-8",
    )
    result = smoke_test(no_predict, "no-predict", train, make_meta(), CFG)
    assert result["ok"] is False
    assert "predict_batch" in str(result["error"])


def test_smoke_hang_is_bounded_by_the_subprocess(tmp_path: Path) -> None:
    """A candidate that hangs in fit must never hang the parent: the smoke
    runs under the worker's killpg timeout, so `while True: pass` costs the
    smoke budget, not the whole unattended run."""
    source = tmp_path / "model.py"
    source.write_text(SLOW_SOURCE, encoding="utf-8")
    train = ar_series(400)
    hung_cfg = HeadroomConfig(input_length=L, horizon=H, candidate_timeout_seconds=5.0)

    start = time.perf_counter()
    result = smoke_test(source, "slow-timeout", train, make_meta(), hung_cfg)
    elapsed = time.perf_counter() - start
    # If killpg failed, communicate() would block on the busy-waiting child
    # forever; finishing near the 5s smoke budget is the proof the whole
    # process group died and no leftover process is still spinning.
    assert elapsed < 8.0
    assert result["ok"] is False
    assert result["predict_shape"] is None
    # The hang is recorded as repair text, distinctly from a crash, so the
    # loop's smoke path can never raise into the run.
    assert "hung" in str(result["error"])


# ---------------------------------------------------------------------------
# evaluate_candidate (subprocess path)
# ---------------------------------------------------------------------------


def test_interleaved_batches_respect_spacing() -> None:
    origins, batch_ids = interleaved(3, 20, TRAIN_END, L + H)
    assert origins.size == 60 and batch_ids.size == 60
    for b in np.unique(batch_ids):
        same = np.sort(origins[batch_ids == b])
        assert int(np.min(np.diff(same))) >= L + H


def test_evaluate_end_to_end_ridge(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    series = ar_series(1600)
    npz_in, npz_out, diag_origins, _ = write_standard_inputs(source, series, RIDGE_SOURCE, tmp_path)

    evaluate_candidate(source, "ridge-e2e", npz_in, npz_out, CFG, timeout_seconds=120.0)

    with np.load(npz_out) as out:
        pred_diag = out["pred_diag"]
        assert pred_diag.shape == (60, H, C)
        assert pred_diag.dtype == np.float64
        assert bool(out["batch_invariant"]) is True
        assert float(out["fit_seconds"]) >= 0.0
        assert float(out["predict_seconds"]) >= 0.0
    # Sane predictions, checked the cheap way: the model fit on train must
    # beat the zero predictor on train windows, and the predictions that came
    # back through the subprocess must beat it on the diag windows too.
    import importlib.util
    spec = importlib.util.spec_from_file_location("ridge_e2e_test", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    adapter = module.create_adapter()
    adapter.fit(series[:TRAIN_END], make_meta())
    train_x, train_y = windows(series[:TRAIN_END], np.arange(700, 796, dtype=np.int64), CFG)
    train_pred = np.asarray(adapter.predict_batch(train_x))
    assert float(np.mean((train_pred - train_y) ** 2)) < float(np.mean(train_y**2))
    _, diag_y = windows(series, diag_origins, CFG)
    assert float(np.mean((pred_diag - diag_y) ** 2)) < float(np.mean(diag_y**2))


def test_timeout_kills_process_group_quickly(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    series = ar_series(200)
    npz_in, npz_out = write_small_inputs(source, series, SLOW_SOURCE, tmp_path)

    start = time.perf_counter()
    with pytest.raises(CandidateTimeout, match="exceeded the 3s budget"):
        evaluate_candidate(source, "slow-timeout", npz_in, npz_out, CFG, timeout_seconds=3.0)
    elapsed = time.perf_counter() - start
    # If killpg failed, communicate() would block until the busy-waiting child
    # finished on its own; returning promptly is the observable proof that the
    # whole process group died and no leftover process is still spinning.
    assert elapsed < 3.0 + 2.0
    assert not npz_out.exists()


def test_cheat_detection_flags_batch_statistics(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    series = ar_series(1600)
    npz_in, npz_out, _, _ = write_standard_inputs(source, series, CHEAT_SOURCE, tmp_path)

    with pytest.raises(CheatDetected, match="batch-invariant"):
        evaluate_candidate(source, "cheat-stats", npz_in, npz_out, CFG, timeout_seconds=120.0)
    # The worker finished normally; the cheat flag itself is in the file.
    with np.load(npz_out) as out:
        assert bool(out["batch_invariant"]) is False
        assert out["pred_diag"].shape == (60, H, C)


def test_crash_carries_stderr_tail(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    series = ar_series(200)
    npz_in, npz_out = write_small_inputs(source, series, CRASH_SOURCE, tmp_path)

    with pytest.raises(CandidateCrash) as exc_info:
        evaluate_candidate(source, "crash-boom", npz_in, npz_out, CFG, timeout_seconds=120.0)
    message = str(exc_info.value)
    assert "exit code 1" in message
    assert "RuntimeError: boom: candidate fit exploded" in message


def test_stderr_tail_carries_a_long_traceback_in_full() -> None:
    from harnessing_ts.headroom.candidate import _STDERR_TAIL_CHARS, _tail

    traceback = "Traceback (most recent call last):\n" + "\n".join(
        f'  File "model.py", line {i}, in fit' for i in range(300)
    ) + "\nRuntimeError: boom"
    assert len(traceback) > 1_200  # a shallow tail would cut the stack away
    assert _tail(traceback) == traceback
    assert _tail("") == "(no output captured)"



def test_contract_violation_inside_worker(tmp_path: Path) -> None:
    source = tmp_path / "model.py"
    series = ar_series(200)
    npz_in, npz_out = write_small_inputs(source, series, BAD_SHAPE_SOURCE, tmp_path)

    with pytest.raises(ContractViolation, match="predict_batch returned shape"):
        evaluate_candidate(source, "bad-shape", npz_in, npz_out, CFG, timeout_seconds=120.0)


# ---------------------------------------------------------------------------
# environment isolation
# ---------------------------------------------------------------------------


def test_build_env_is_a_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_TEST_SENTINEL", "must-not-leak")
    monkeypatch.setenv("HEADROOM_TEST_API_KEY", "sk-nope")
    home = Path("/tmp/headroom-run-xyz")
    env = build_env(home)
    assert "HEADROOM_TEST_SENTINEL" not in env
    assert "HEADROOM_TEST_API_KEY" not in env
    assert env["PATH"] == os.environ["PATH"]
    assert env["HOME"] == str(home)
    assert env["PYTHONPATH"] == str(Path(__file__).resolve().parents[1] / "backend")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert env[key] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_incumbent_templates_pass_the_source_check() -> None:
    """Every candidate derives from an m0 template, whose
    `from __future__ import annotations` line the whitelist must allow.
    Both templates are covered: the linear one (still used by the fast test
    suite and the model-agnostic G0 gates) and the PatchTST-lite deep one
    (the v2 default incumbent — its torch imports must stay whitelisted)."""
    from pathlib import Path

    templates = Path(__file__).resolve().parents[1] / "experiments" / "templates"
    check_source((templates / "m0.py").read_text(encoding="utf-8"))
    check_source((templates / "m0_deep.py").read_text(encoding="utf-8"))


def test_stderr_tail_marks_a_cut_head() -> None:
    """A stderr longer than the cap renders its last cap characters with a
    marker that the head was cut."""
    from harnessing_ts.headroom.candidate import _STDERR_TAIL_CHARS, _tail

    longer = "x" * (_STDERR_TAIL_CHARS + 10)
    tail = _tail(longer)
    assert tail.startswith("… (earlier output truncated)")
    assert tail.endswith("x" * 100)
    short = "Traceback (most recent call last): boom"
    assert _tail(short) == short

def test_whitelist_rejects_the_attribute_and_import_bypasses() -> None:
    """Each closed bypass, rejected by name: dynamic attribute builtins,
    dunder attribute access, the FFI backdoor and the file-reading
    submodules reachable from otherwise-allowed roots."""
    bypasses = [
        'import numpy as np\n\ndef create_adapter():\n    return None\n\nx = getattr(np, "lo" + "ad")\n',
        'import numpy as np\n\ndef create_adapter():\n    return None\n\nx = setattr(np.ndarray, "x", 1)\n',
        'import numpy as np\n\ndef create_adapter():\n    return None\n\nx = ().__class__.__base__\n',
        'import numpy as np\n\ndef create_adapter():\n    return None\n\nx = __builtins__["open"]\n',
        'import scipy.io\n\ndef create_adapter():\n    return None\n\nx = scipy.io.loadmat("/tmp/x")\n',
        'from scipy.io import loadmat\n\ndef create_adapter():\n    return None\n\nx = loadmat("/tmp/x")\n',
        'import numpy.ctypeslib\n\ndef create_adapter():\n    return None\n\nx = numpy.ctypeslib.ctypes.CDLL(None)\n',
        'import numpy as np\n\ndef create_adapter():\n    return None\n\nx = np.DataSource()\n',
        'import scipy.signal\n\ndef create_adapter():\n    return None\n\nx = scipy.io.netcdf_file("/tmp/x")\n',
    ]
    for source in bypasses:
        with pytest.raises(SourceRejected):
            check_source(source)


def test_whitelist_still_accepts_the_templates_and_init_calls() -> None:
    """The hardening rejects no legitimate code: both incumbent templates
    pass, and initializer calls through super()/parent classes survive."""
    templates = Path(__file__).resolve().parents[1] / "experiments" / "templates"
    for name in ("m0.py", "m0_deep.py"):
        check_source((templates / name).read_text(encoding="utf-8"))
    check_source(
        "class A:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "\n"
        "class B(A):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "\n"
        "\n"
        "def create_adapter():\n"
        "    return B()\n"
    )
