"""Subprocess worker spawned by ``candidate.evaluate_candidate``; never run by hand.

Reads npz_in (train, meta, x_diag, x_cert, batch_ids_diag, single_check_idx,
source_path — the binding key contract lives in candidate.py's docstring),
fits the candidate, and writes pred_diag/pred_cert/batch_invariant/
fit_seconds/predict_seconds to npz_out.  The result file is the only channel
out: the parent merges the worker's stdout into stderr, so nothing a candidate
prints can corrupt the channel.

The parent already AST-vetted the source, so this module does not re-check:
the rejection message would only be duplicated, and the smoke phase
(``candidate.load_adapter`` + ``smoke_test``) has already executed the module
once in-process.

The worker stays numpy-only on purpose.  torch, if the candidate uses it, is
imported by the candidate itself, inside the pinned-thread environment the
parent set up — importing it here would force torch on every candidate.

Exit codes: 0 = result written; 1 = crash (traceback on stderr); 3 = adapter
contract violation (missing callable or wrong prediction shape).  The parent
maps 3 to ContractViolation and any other nonzero exit to CandidateCrash.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

CRASH_EXIT = 1
# == candidate.WORKER_CONTRACT_EXIT; duplicated rather than imported so the
# child's import surface stays numpy-only.
CONTRACT_EXIT = 3

# Agreement tolerance between single-row and batch predictions.  rtol keeps
# np.allclose's default: batched BLAS paths round differently from single-row
# ones, and an honest model's last-bit noise must not register as cheating,
# while batch-statistics peeking produces O(1) gaps.
_SINGLE_ATOL = 1e-9


class _ContractError(Exception):
    """Adapter-contract failure inside the worker; maps to exit code 3."""


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <npz_in> <npz_out>", file=sys.stderr)
        return CRASH_EXIT
    try:
        data = dict(np.load(Path(argv[0]), allow_pickle=False))
        _run(data, Path(argv[1]))
    except _ContractError:
        traceback.print_exc()
        return CONTRACT_EXIT
    except Exception:
        traceback.print_exc()
        return CRASH_EXIT
    return 0


def _run(data: dict[str, np.ndarray], npz_out: Path) -> None:
    meta = json.loads(data["meta"].item())
    train = np.asarray(data["train"], dtype=np.float64)
    x_diag = np.asarray(data["x_diag"], dtype=np.float64)
    x_cert = np.asarray(data["x_cert"], dtype=np.float64)
    batch_ids = np.asarray(data["batch_ids_diag"]).astype(np.int64).ravel()
    single_idx = np.asarray(data["single_check_idx"]).astype(np.int64).ravel()
    source_path = Path(data["source_path"].item())
    horizon = int(meta["H"])
    channels = int(train.shape[1])
    if batch_ids.shape[0] != x_diag.shape[0]:
        # A mismatch would leave rows of pred_diag as uninitialized memory —
        # fail loudly instead of scoring garbage.
        raise ValueError(
            f"batch_ids_diag has {batch_ids.shape[0]} entries but x_diag has "
            f"{x_diag.shape[0]} rows"
        )

    adapter = _load_adapter(source_path)
    start = time.perf_counter()
    adapter.fit(train, meta)
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    # One predict_batch call per interleaved batch, predictions reassembled in
    # the original row order.  Intra-batch spacing is the caller's structural
    # anti-peeking guarantee (candidate.py docstring); this loop just honors
    # the batch ids, whichever rows they select.
    pred_diag = np.empty((x_diag.shape[0], horizon, channels), dtype=np.float64)
    for batch_id in np.unique(batch_ids):
        rows = np.flatnonzero(batch_ids == batch_id)
        pred_diag[rows] = _predict(adapter, x_diag[rows], (rows.size, horizon, channels))
    # Batch-invariance probe: re-predict sampled rows as batches of one.  A
    # predict_batch that reads batch statistics cannot reproduce its own
    # batch answers, which is the cheating pattern this exists to catch.
    batch_invariant = True
    for i in single_idx:
        single = _predict(adapter, x_diag[i : i + 1], (1, horizon, channels))
        if not np.allclose(single[0], pred_diag[i], atol=_SINGLE_ATOL):
            batch_invariant = False
            break
    # cert has no batch ids by contract: one call, one batch.  Cheaters are
    # flagged on diag before the loop can trust their cert numbers.
    if x_cert.shape[0]:
        pred_cert = _predict(adapter, x_cert, (x_cert.shape[0], horizon, channels))
    else:
        pred_cert = np.empty((0, horizon, channels), dtype=np.float64)
    predict_seconds = time.perf_counter() - start

    np.savez(
        npz_out,
        pred_diag=pred_diag,
        pred_cert=pred_cert,
        batch_invariant=np.array(batch_invariant),
        fit_seconds=np.array(fit_seconds),
        predict_seconds=np.array(predict_seconds),
    )


def _load_adapter(source_path: Path) -> Any:
    """Import the vetted candidate module and instantiate its adapter."""
    spec = importlib.util.spec_from_file_location("_headroom_candidate_worker", source_path)
    if spec is None or spec.loader is None:
        raise _ContractError(f"cannot build an import spec for candidate source {source_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass/pickle-adjacent code in the candidate
    # can resolve the module by name.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise _ContractError(
            f"importing candidate source {source_path} raised {type(exc).__name__}: {exc}"
        ) from exc
    factory = getattr(module, "create_adapter", None)
    if not callable(factory):
        raise _ContractError(
            f"candidate module must define a callable create_adapter(), found {factory!r}"
        )
    try:
        adapter = factory()
    except Exception as exc:
        raise _ContractError(f"create_adapter() raised {type(exc).__name__}: {exc}") from exc
    for method in ("fit", "predict_batch"):
        if not callable(getattr(adapter, method, None)):
            raise _ContractError(
                f"adapter must expose a callable {method}(), found "
                f"{type(adapter).__name__} without one"
            )
    return adapter


def _predict(adapter: Any, x: np.ndarray, expected: tuple[int, int, int]) -> np.ndarray:
    """Call predict_batch and enforce the [B, H, C] output contract."""
    out = np.asarray(adapter.predict_batch(x))
    if out.shape != expected:
        raise _ContractError(
            f"predict_batch returned shape {tuple(out.shape)} for input shape {x.shape}, "
            f"expected {expected} ([B, L, C] -> [B, H, C])"
        )
    return out.astype(np.float64, copy=False)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
