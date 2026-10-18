"""Candidate contract: AST source vetting, adapter loading, isolated evaluation.

A candidate is one file, ``models/candidates/<id>/model.py``, with an id
matching ``CANDIDATE_ID_PATTERN``, defining ``create_adapter() -> Adapter``.
The adapter carries ``model_id`` equal to the candidate id and two methods:
``fit(train, meta)`` (train only; early stopping must happen inside train) and
``predict_batch(x) -> y`` mapping ``[B, L, C]`` to ``[B, H, C]``.  ``meta`` is
built by the loop with keys ``L``, ``H``, ``C``, ``seed`` (see
``experiments/evaluate.py::fit_meta``) and never contains the dataset name.

The loop composes the pipeline in this order: ``check_source`` ->
``smoke_test`` (isolated subprocess on a short budget; feeds the agent shape
and error text only, never metrics) -> ``evaluate_candidate`` (the same
subprocess protocol on the full budget).  Both subprocess phases run
``candidate_worker`` under the same killpg timeout: a smoke that ran
in-process could not be bounded, so a candidate that hangs in ``fit`` would
stall the whole unattended run — the one failure mode no external watchdog
can heal.  Every failure lands in exactly one ``CandidateError`` category
with an expected-vs-actual message, so the loop can count the candidate
against the budget and hand the text back to the agent in a repair turn.

Subprocess npz contract — ``npz_in``, written by the loop:

    train             [T, C]     float64  training series (train segment only)
    meta              0-d str    JSON dict {"L", "H", "C", "seed", ...}
    x_diag            [B, L, C]  float64  diag input windows, interleaved order
    batch_ids_diag    [B] int32  interleaved batch id of each x_diag row
    single_check_idx  [S] int32  x_diag row indices also predicted one-by-one
    source_path       0-d str    absolute path of the candidate model.py
    candidate_id      0-d str    optional, smoke only: the adapter's model_id must equal it

No targets are ever written into npz_in; the loop keeps them and computes
every metric itself.  ``npz_out``, written by the worker:

    pred_diag [B, H, C] float64, batch_invariant (bool),
    fit_seconds and predict_seconds (float64).

Interleaved batching — the loop constructs it, the worker honors it: x_diag
rows are ordered so any two rows sharing a batch id have origins at least
L + H apart, so no row's input window contains another row's targets and
batch statistics cannot carry target information across rows.  The worker
makes one ``predict_batch`` call per batch id and reassembles predictions in
the original row order.  The single-check rows close the remaining hole: they
are re-predicted as batches of one and compared to their batch predictions
(atol 1e-9), so a ``predict_batch`` that reads batch composition (for
example by normalising over the batch axis) is flagged as a cheat.

This is correctness tooling, not a security boundary: the AST whitelist and
the environment whitelist stop obvious footguns in LLM-generated code, but a
determined author could get around them (getattr chains are a known hole).
What matters for the experiment is recovery: a timed-out candidate is
SIGKILLed as a whole process group, and one bad candidate never stalls the
run.
"""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.timeidx import windows

# Candidate ids double as directory names under models/candidates/ and as the
# adapter's model_id, so they are kebab-case and bounded (CANDIDATE_ID_PATTERN).
CANDIDATE_ID_PATTERN: re.Pattern[str] = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")

# The repo's backend/ directory: PYTHONPATH for the worker subprocess, so
# ``-m harnessing_ts.headroom.candidate_worker`` resolves no matter how the
# parent interpreter was launched (pytest, uv, plain python).
_BACKEND_DIR = Path(__file__).resolve().parents[2]

# Exit code the worker uses for adapter-contract failures (missing callable,
# wrong prediction shape); every other nonzero exit is a crash.  Kept in sync
# with CONTRACT_EXIT in candidate_worker.py — the two modules deliberately do
# not import each other, keeping the child's import surface numpy-only.
WORKER_CONTRACT_EXIT = 3

_ALLOWED_IMPORT_ROOTS = frozenset({"numpy", "math", "scipy", "torch"})
# Builtins that reach outside the computation or re-enter the interpreter.
_FORBIDDEN_NAMES = frozenset(
    {"open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "breakpoint", "exit", "quit"}
)
# Method-style calls for file, process, and network access.  Any owner may be
# harmful (np.load, os.system, urllib.request), so the attribute name alone is
# enough reason to reject; a candidate defining its own .save() just renames it.
_FORBIDDEN_ATTR_CALLS = frozenset(
    {"load", "save", "savez", "loadtxt", "savetxt", "genfromtxt", "fromfile", "tofile", "memmap", "system", "popen", "urlopen", "request"}
)

_SMOKE_TRAIN_CAP = 4000
_SMOKE_WINDOWS = 4
# The smoke budget is a fraction of the evaluation budget: a fit that cannot
# finish on four windows within a minute is a hang, not a slow model.
_SMOKE_TIMEOUT_CAP = 60.0
_STDERR_TAIL_CHARS = 16000
_RESULT_KEYS = ("pred_diag", "batch_invariant", "fit_seconds", "predict_seconds")



class CandidateError(RuntimeError):
    """Base for every candidate failure; the subclass is the category.

    Subclasses RuntimeError per headroom-contracts.md §1 (subprocess failures
    raise RuntimeError or a subclass), with messages that always name what was
    expected and what was found, so the loop can hand the text straight back
    to the agent in a repair turn.
    """


class SourceRejected(CandidateError):
    """The source violates the AST import/name whitelist or lacks create_adapter."""


class CandidateTimeout(CandidateError):
    """The evaluation subprocess exceeded its budget and was killed."""


class CandidateCrash(CandidateError):
    """The worker exited nonzero (candidate raised, OOM, ...) or produced invalid output."""


class CheatDetected(CandidateError):
    """predict_batch is not batch-invariant: it reads batch statistics."""


class ContractViolation(CandidateError):
    """model.py breaks the adapter contract: missing create_adapter/fit/predict_batch,
    model_id mismatch, or wrong prediction shapes."""


def check_source(source: str) -> None:
    """Reject candidate sources outside the whitelist; raises SourceRejected.

    An AST walk, not a regex: comments, docstrings and string contents never
    trigger false rejections, and nothing that parses can slip past on
    formatting.  All violations are reported at once (with line numbers) so a
    repair turn can fix the whole file, not one line per round.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SourceRejected(f"line {exc.lineno}: syntax error: {exc.msg}") from exc

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in _ALLOWED_IMPORT_ROOTS:
                    violations.append(
                        f"line {node.lineno}: import of {alias.name!r} is not allowed; "
                        f"candidates may import only numpy, math, scipy.* and torch.*"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                violations.append(f"line {node.lineno}: relative imports are not allowed in candidates")
            elif node.module == "__future__":
                # Annotation directives inherited from the incumbent template;
                # a no-op at runtime and required by every derived candidate.
                continue
            elif node.module is None or node.module.split(".")[0] not in _ALLOWED_IMPORT_ROOTS:
                violations.append(
                    f"line {node.lineno}: import from {node.module!r} is not allowed; "
                    f"candidates may import only numpy, math, scipy.* and torch.*"
                )
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in _FORBIDDEN_NAMES:
            violations.append(
                f"line {node.lineno}: use of {node.id!r} is not allowed in candidates "
                f"(no file/process/interpreter access)"
            )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _FORBIDDEN_ATTR_CALLS
        ):
            violations.append(
                f"line {node.lineno}: call to .{node.func.attr}() is not allowed in candidates "
                f"(no file/process/network access)"
            )
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "create_adapter" for node in tree.body
    ):
        violations.append("model.py must define a top-level create_adapter() (the adapter factory)")
    if violations:
        raise SourceRejected("candidate source rejected:\n" + "\n".join(violations))


def build_env() -> dict[str, str]:
    """Whitelisted environment for the worker subprocess; nothing else passes.

    The parent environment is deliberately not copied: API keys, harness
    state and workspace paths must not reach candidate code.  Only PATH,
    HOME and TMPDIR pass through (the child needs them for caches and temp
    files), PYTHONPATH is pinned to the repo backend so the worker module
    resolves, thread pins keep numpy/scipy/torch from spawning a thread storm
    on the same machine the loop runs on, and CUDA is hidden — candidates
    evaluate on CPU (macOS MPS has no hide flag; the same env is used for the
    posthoc re-runs, so reproducibility survives that gap).
    """
    env = {
        "PYTHONPATH": str(_BACKEND_DIR),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    for key in ("PATH", "HOME", "TMPDIR"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def smoke_test(
    source_path: Path,
    candidate_id: str,
    train: np.ndarray,
    meta: dict[str, Any],
    cfg: HeadroomConfig,
) -> dict[str, Any]:
    """Fit + predict on a train slice in the isolated worker; shape and error text only.

    This is the repair-turn instrument: the loop feeds ``error`` back to the
    agent verbatim, so it must carry no performance signal (no MSE, no
    residual statistics) — the agent must not be able to tune on it.  The
    inputs are minimal — the first ``_SMOKE_TRAIN_CAP`` train rows and
    ``_SMOKE_WINDOWS`` diag windows cut from TRAIN ONLY (diag data does not
    exist yet at smoke time, and the smoke must never carry a performance
    signal back) — and the budget is ``min(_SMOKE_TIMEOUT_CAP,
    cfg.candidate_timeout_seconds)``.  Any candidate-side failure is captured
    as text instead of raised, because a failing smoke run is an expected,
    budgeted outcome, not a harness bug.  That includes a hang: the smoke runs
    under the worker's killpg timeout, so ``while True: pass`` in fit costs
    the smoke budget and comes back as "hung in smoke" text — repair advice
    pointing at infinite loops, distinct from a crash's traceback — never as a
    parent-process stall.  Only a misconfigured harness raises (ValueError).
    """
    train = np.asarray(train, dtype=np.float64)
    if train.ndim != 2:
        raise ValueError(f"train must be [T, C], got shape {train.shape}")
    seg = train[: min(train.shape[0], _SMOKE_TRAIN_CAP)]
    first_origin = seg.shape[0] - cfg.horizon - _SMOKE_WINDOWS
    if first_origin < cfg.input_length - 1:
        # A too-short train is a harness misconfiguration, not a candidate
        # failure, so it stays a loud ValueError rather than repair text.
        raise ValueError(
            f"smoke_test needs at least input_length + horizon + {_SMOKE_WINDOWS - 1} = "
            f"{cfg.input_length + cfg.horizon + _SMOKE_WINDOWS - 1} train rows, got {seg.shape[0]}"
        )
    origins = np.arange(first_origin, seg.shape[0] - cfg.horizon, dtype=np.int64)
    x_diag, _ = windows(seg, origins, cfg)
    result: dict[str, Any] = {"ok": False, "error": None, "predict_shape": None}
    with tempfile.TemporaryDirectory(prefix="headroom-smoke-") as run_dir:
        npz_in, npz_out = Path(run_dir) / "smoke-in.npz", Path(run_dir) / "smoke-out.npz"
        np.savez(
            npz_in,
            train=seg,
            meta=np.array(json.dumps(meta)),
            x_diag=x_diag,
            batch_ids_diag=np.zeros(x_diag.shape[0], dtype=np.int32),
            single_check_idx=np.zeros(0, dtype=np.int32),
            source_path=np.array(str(Path(source_path).resolve())),
            # Smoke-only key: holds the adapter to the model_id contract
            # inside the worker (the evaluation npz has no id to check
            # against).
            candidate_id=np.array(candidate_id),
        )
        try:
            evaluate_candidate(
                source_path, candidate_id, npz_in, npz_out, cfg,
                timeout_seconds=min(_SMOKE_TIMEOUT_CAP, cfg.candidate_timeout_seconds),
            )
        except (CandidateCrash, ContractViolation) as exc:
            result["error"] = str(exc)
            return result
        except CandidateTimeout as exc:
            # A hang is qualitatively different from a crash — no exception, no
            # output, no progress — so the error string says so and points the
            # repair turn at infinite loops instead of raised errors.
            result["error"] = (
                f"hung in smoke: {exc} (no crash and no output; look for an "
                f"infinite loop in fit/predict_batch rather than a raised exception)"
            )
            return result
        with np.load(npz_out, allow_pickle=False) as out:
            # The worker's predictions are read for their shape only — the
            # smoke never surfaces numbers the agent could tune on.
            result["predict_shape"] = tuple(out["pred_diag"].shape)
    result["ok"] = True
    return result


def run_isolated(
    command: list[str],
    run_dir: Path,
    timeout_seconds: float,
    timeout_error: Callable[[str], CandidateError],
) -> tuple[int, str]:
    """Run ``command`` in a fresh cwd under ``build_env()``; return (exit code, output tail).

    The shared isolation pattern for every untrusted-subprocess run (candidate
    and probe workers alike): both child streams share one pipe (a chatty
    child cannot corrupt any channel — the result file is the only channel
    out), and ``start_new_session`` makes the child a session leader whose pid
    is the process group id, so one SIGKILL reaps the worker and anything it
    spawned.  SIGKILL, not SIGTERM — code wedged in native code must die, not
    negotiate.  On timeout, ``timeout_error`` receives the budget for the
    message and is raised.
    """
    process = subprocess.Popen(
        command,
        cwd=run_dir,
        env=build_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # already dead; the reap below still collects it
        process.communicate()
        raise timeout_error(f"exceeded the {timeout_seconds:g}s budget and its process group was killed") from None
    return process.returncode, _tail(output)


def evaluate_candidate(
    source_path: Path,
    candidate_id: str,
    npz_in: Path,
    npz_out: Path,
    cfg: HeadroomConfig,
    *,
    timeout_seconds: float,
) -> None:
    """Evaluate one candidate in an isolated subprocess, classifying every failure.

    Success returns None — the predictions live in npz_out and the loop owns
    the targets, so no metric is computed here.  Failures raise exactly one
    exception, and the category is the point:

    - ``CandidateTimeout`` — the budget ran out; the whole process group is
      SIGKILLed first, so nothing keeps burning CPU afterwards;
    - ``ContractViolation`` — worker exit code 3: the adapter broke the
      callable or shape contract inside the subprocess;
    - ``CandidateCrash`` — any other nonzero exit, or a missing/invalid
      result npz, with the stderr tail attached;
    - ``CheatDetected`` — the worker finished but its single-vs-batch probe
      flagged ``predict_batch`` as batch-composition-dependent.

    The environment is the ``build_env()`` whitelist and the cwd a fresh temp
    directory, so relative-path games in candidate code land nowhere.
    """
    _check_candidate_id(candidate_id)
    if not Path(source_path).is_file():
        raise ValueError(f"candidate source {source_path} does not exist")
    npz_in = Path(npz_in).resolve()
    npz_out = Path(npz_out).resolve()
    command = [
        sys.executable, "-m", "harnessing_ts.headroom.candidate_worker",
        str(npz_in), str(npz_out),
    ]
    with tempfile.TemporaryDirectory(prefix="headroom-candidate-") as run_dir:
        code, tail = run_isolated(
            command,
            Path(run_dir),
            timeout_seconds,
            lambda message: CandidateTimeout(f"candidate {candidate_id!r} {message}"),
        )
    if code == WORKER_CONTRACT_EXIT:
        raise ContractViolation(
            f"candidate {candidate_id!r} broke the adapter contract inside the worker:\n{tail}"
        )
    if code != 0:
        raise CandidateCrash(
            f"candidate {candidate_id!r} crashed in the worker (exit code {code}):\n{tail}"
        )
    _validate_result(npz_out, candidate_id)


def _check_candidate_id(candidate_id: str) -> None:
    if not CANDIDATE_ID_PATTERN.fullmatch(candidate_id):
        raise ValueError(f"candidate_id must match [a-z0-9][a-z0-9-]{{0,63}}, got {candidate_id!r}")


def _tail(output: str) -> str:
    """The last _STDERR_TAIL_CHARS characters, with a marker when the head is cut."""
    if not output:
        return "(no output captured)"
    if len(output) <= _STDERR_TAIL_CHARS:
        return output
    return "… (earlier output truncated)\n" + output[-_STDERR_TAIL_CHARS:]


def _validate_result(npz_out: Path, candidate_id: str) -> None:
    """Check the worker's result file before the loop trusts it.

    The worker enforces the [B, H, C] shape and float64 contract per
    predict_batch call; this pass checks the one thing the worker cannot
    itself guarantee — that the file it handed back is readable, complete
    (every result key present) and carries an honest batch-invariance flag.
    """
    try:
        result = np.load(npz_out, allow_pickle=False)
    except Exception as exc:
        raise CandidateCrash(
            f"candidate {candidate_id!r}: worker exited 0 but wrote no readable result to "
            f"{npz_out.name}: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        missing = [key for key in _RESULT_KEYS if key not in result.files]
        if missing:
            raise CandidateCrash(f"candidate {candidate_id!r}: result npz is missing keys {missing}")
        if not bool(result["batch_invariant"]):
            raise CheatDetected(
                f"candidate {candidate_id!r} is not batch-invariant: re-predicting single rows "
                f"disagrees with their batch predictions beyond atol=1e-9, so predict_batch reads "
                f"batch statistics (e.g. normalises over the batch axis); a forecast must depend "
                f"on its own window only"
            )
    except (CandidateCrash, CheatDetected):
        raise
    except Exception as exc:
        raise CandidateCrash(
            f"candidate {candidate_id!r}: result npz {npz_out.name} is invalid: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        result.close()
