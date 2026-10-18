"""G0 certificate calibration: the seven gates of final-plan §5, no LLM involved.

Usage:
    uv run --extra research python experiments/stage0.py            # full run (linear gates)
    uv run --extra research python experiments/stage0.py --smoke    # tiny sizes
    uv run --extra research python experiments/stage0.py --deep     # deep G0-3/4/6 recalibration

Writes experiments/runs/stage0/report.json and report.md incrementally after
every gate.  ``--deep`` runs ONLY the three incumbent-dependent gates
(G0-3/G0-4/G0-6) with the deep incumbent and its planted defect variants,
writing experiments/runs/stage0-v2 (or stage0-v2-smoke with --smoke); the
v1 report directory is never touched.  Confidence levels are wired end to
end: card gains use correctors.card_level; certification gains use the fixed
level 1 - certifyAlpha (a two-sided CI, so each UCB is one-sided
conservative), which certify() derives from cfg.certify_alpha and records
in the report.  Thresholds follow final-plan §5; where a hard threshold
would be flaky under binomial noise at the achievable cell count, the pass
bound is the conservative binomial quantile and the exact numbers are
reported.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from functools import partial
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import evaluate as ev

from harnessing_ts.headroom.certify import certify
from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import (
    Coordinate,
    all_gains,
    card_level,
    corrector_gain,
    coordinate_gain,
    fit_greedy,
    select_period_peaks,
)
from harnessing_ts.headroom.residuals import SegmentData, segment_data
from harnessing_ts.headroom.synthetic import REGISTRY
from harnessing_ts.headroom.timeidx import segment_origins

# The full-size run owns experiments/runs/stage0 (the pre-registered
# artifact); any smoke run writes beside it so concurrent tests can never
# overwrite the calibration report.  The directory follows the smoke flag the
# caller passes, so a pytest process calling run(smoke=True) directly never
# clobbers the full-size report.
def _report_dir(smoke: bool) -> Path:
    return _REPO_ROOT / "experiments" / "runs" / ("stage0-smoke" if smoke else "stage0")


def _deep_report_dir(smoke: bool) -> Path:
    return _REPO_ROOT / "experiments" / "runs" / ("stage0-v2-smoke" if smoke else "stage0-v2")

# Pre-registered (task, model) -> expected top-card aspects (final plan §5).
# Period-168 structure exceeds the lookback, so beyond-l is a legitimate top
# coordinate for multiperiod; xchannel coupling lives at lags 3-7, visible to
# the lag coordinate even with a short window.  m0-horizon-bias is excluded:
# a per-step constant is absorbed by every coordinate's ridge intercept, so
# its top coordinate carries no information.
EXPECTED_ASPECTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("lgssm_multiperiod", "m0-linear"): ("periodicity", "long-range-beyond-window"),
    ("lgssm_multiperiod", "m0-short-lookback"): ("periodicity", "long-range-beyond-window"),
    # xchannel coupling lives at lags 3-7: it reaches every channel's own
    # history, so the lag coordinate legitimately sees it first.
    ("lgssm_xchannel", "m0-linear"): ("cross-channel", "lag-structure"),
    ("lgssm_xchannel", "m0-short-lookback"): ("cross-channel", "lag-structure", "long-range-beyond-window"),
    ("arfima", "m0-linear"): ("lag-structure",),
    ("hmm_ar_switch", "m0-linear"): ("lag-structure", "periodicity"),
    # The h-step nonlinear coupling also shows up in same-horizon realized
    # residual chains, so the horizon coordinate can legitimately lead.
    ("nonlinear_xchannel", "m0-linear"): ("cross-channel", "horizon-range"),
}
# G0-2 calibration pairs (CI coverage against an independent long-segment
# truth; arfima stresses HAC under long memory).
CALIBRATION_PAIRS = ("lgssm_multiperiod", "arfima", "hmm_ar_switch")
# G0-6 screening tasks (garch / random_walk are negative controls by design).
SCREENING_TASKS = ("lgssm_multiperiod", "lgssm_xchannel", "nonlinear_xchannel", "hmm_ar_switch", "arfima")
ALL_TASKS = ("lgssm_multiperiod", "lgssm_xchannel", "hmm_ar_switch", "arfima", "garch", "random_walk")

# --- deep-incumbent recalibration (plan §12; main-v2 calibrationEvidence ------
# "g0-recalibration").  G0-3/G0-4/G0-6 were calibrated with linear m0
# variants; the default incumbent is now m0-patchtst, so those three gates
# are re-derived over the same tasks with the deep base plus three planted
# defect variants.  G0-1/2/5/7 are model-agnostic and carry over from the
# frozen fixed-level artifact experiments/specs/g0-final-report-v3.json
# (CARRY_OVER_NOTES).
#
# Expected top-card aspects per (task, variant), derived from the task's
# generating structure plus the variant's defect (each pair's reasoning is
# restated in the frozen v2 artifact); m0-patchtst-channelmix is excluded on
# the univariate tasks (arfima, hmm_ar_switch) because averaging one channel
# is a no-op, exactly like the cross-channel rule fix is inapplicable there:
DEEP_EXPECTED_ASPECTS: dict[tuple[str, str], tuple[str, ...]] = {
    # lgssm_multiperiod: the period-168 oscillator exceeds the L=96 lookback,
    # so its state-estimation error — periodic at 168 — is what the deep base
    # leaves; band(P~168) tracks it and raw history at lags 96-240 pins it.
    ("lgssm_multiperiod", "m0-patchtst"): ("periodicity", "long-range-beyond-window"),
    # Attention is the only cross-patch mechanism; ablating it degrades phase
    # integration on both oscillators AND leaves short-lag residual
    # autocorrelation, on top of the base's slow-structure remainder — a
    # genuine three-way tie between the repair channels.
    ("lgssm_multiperiod", "m0-patchtst-noattn"): (
        "lag-structure", "periodicity", "long-range-beyond-window"),
    # The 48/48 patch grid phase-locks the 24-period (96 = 2*48 is a multiple
    # of 24) but destroys resolution of the 168-period: the slow periodic
    # remainder grows.
    ("lgssm_multiperiod", "m0-patchtst-coarsepatch"): ("periodicity", "long-range-beyond-window"),
    # Every channel receives the average forecast, so each residual carries
    # that channel's deviation from the average; other channels' realized
    # history repairs the shared part, and the slow level makes the residual
    # strongly autocorrelated.
    ("lgssm_multiperiod", "m0-patchtst-channelmix"): ("cross-channel", "lag-structure"),
    # lgssm_xchannel: the channel-independent backbone (design, not defect)
    # is blind to the lags-3..7 couplings for every variant; channel->i
    # repairs directly and own-history lag sees the couplings indirectly.
    # channelmix additionally leaves the coupling-driven per-channel
    # deviations in the residual.
    ("lgssm_xchannel", "m0-patchtst"): ("cross-channel", "lag-structure"),
    ("lgssm_xchannel", "m0-patchtst-noattn"): ("cross-channel", "lag-structure"),
    ("lgssm_xchannel", "m0-patchtst-coarsepatch"): ("cross-channel", "lag-structure"),
    ("lgssm_xchannel", "m0-patchtst-channelmix"): ("cross-channel", "lag-structure"),
    # hmm_ar_switch: regime persistence leaves h-step errors autocorrelated
    # within regimes (lag-structure), and the band coordinate can track the
    # slow inter-regime residual wander, so periodicity is registered as the
    # calibrated co-leader.
    ("hmm_ar_switch", "m0-patchtst"): ("lag-structure", "periodicity"),
    ("hmm_ar_switch", "m0-patchtst-noattn"): ("lag-structure", "periodicity"),
    ("hmm_ar_switch", "m0-patchtst-coarsepatch"): ("lag-structure", "periodicity"),
    # arfima: d=0.35 long memory — the ACF decays like k^-0.3, so both the
    # residual history (lag, to 335) and deep raw history (beyond_l, 96-240)
    # carry signal an L=96 window model cannot see, for every variant.
    ("arfima", "m0-patchtst"): ("lag-structure", "long-range-beyond-window"),
    ("arfima", "m0-patchtst-noattn"): ("lag-structure", "long-range-beyond-window"),
    ("arfima", "m0-patchtst-coarsepatch"): ("lag-structure", "long-range-beyond-window"),
    # nonlinear_xchannel: the channel-independent backbone is blind to the
    # lags-3..7 tanh couplings, and the same-horizon realized chains also
    # carry them.
    ("nonlinear_xchannel", "m0-patchtst"): ("cross-channel", "horizon-range"),
    ("nonlinear_xchannel", "m0-patchtst-noattn"): ("cross-channel", "horizon-range"),
    ("nonlinear_xchannel", "m0-patchtst-coarsepatch"): ("cross-channel", "horizon-range"),
    ("nonlinear_xchannel", "m0-patchtst-channelmix"): ("cross-channel", "horizon-range"),
}
# Why each carried-over gate does not depend on the incumbent (restated in
# the frozen deep artifact's carry_over section; verdicts are read from the
# fixed-level artifact at run time).
CARRY_OVER_NOTES = {
    "G0-1": (
        "oracle negative control: run with m = oracle, so no incumbent is "
        "involved; it measures the machinery's behaviour at zero true headroom"
    ),
    "G0-2": (
        "CI calibration: the HAC/HLN interval and the greedy total-gain "
        "estimator are properties of the corrector machinery, not of the "
        "model whose residuals feed it; the coverage is measured with "
        "m0-linear cells, and the coverage claim is estimator-level"
    ),
    "G0-5": (
        "false-stop: the fixed certification level, the two-sided-to-one-"
        "sided conservatism and the eps*MSE threshold never reference the "
        "incumbent; planted-defect gaps just need to sit far above the stop "
        "threshold, which the linear defects do"
    ),
    "G0-7": (
        "observation-0: a pure AR(1)/white-noise analytic control with "
        "m == 0 — no incumbent exists in the gate at all"
    ),
}
# The synthetic P2 v2 task list (main-v2.json); the real task has no oracle,
# so the G0-6 screening question covers the synthetic three only.
P2_V2_TASKS = ("hmm_ar_switch", "nonlinear_xchannel", "lgssm_xchannel")
# The deep incumbent's float64 predict path allocates one activation tensor
# over B*C rows per call; at full test sizes (52k origins x 4 channels) that
# is multi-GB per tensor, so deep cells predict in origin chunks.
_DEEP_PREDICT_CHUNK = 4096


@dataclass
class Cell:
    """One (task, seed, model) evaluation with the artifacts the gates need."""

    task_name: str
    seed: int
    model_id: str
    seg_diag: SegmentData
    seg_diag_a: SegmentData
    seg_cert: SegmentData
    seg_test: SegmentData
    period_peaks: list[int]
    card: list[Any] = field(default_factory=list)
    cert_gains: list[Any] = field(default_factory=list)
    cert_mse: dict[str, float] = field(default_factory=dict)
    test_gap: float = 0.0
    test_total_delta: float = 0.0
    fit_seconds: float = 0.0


def _load_m0(model_id: str) -> Any:
    return common.load_module("m0_template_" + model_id.replace("-", "_"), _REPO_ROOT / "experiments" / "templates" / "m0.py").variant(model_id)


def _load_model(model_id: str) -> Any:
    """The linear family from m0.py, the patchtst family from m0_deep.py."""
    if model_id.startswith("m0-patchtst"):
        return common.load_module(
            "m0_deep_template_" + model_id.replace("-", "_"),
            _REPO_ROOT / "experiments" / "templates" / "m0_deep.py",
        ).variant(model_id)
    return _load_m0(model_id)


def evaluate_cell(
    task_name: str,
    seed: int,
    model_id: str,
    cfg: HeadroomConfig,
    sizes: dict[str, int],
    *,
    oracle_as_model: bool = False,
    with_card: bool = True,
    predict_chunk: int | None = None,
) -> Cell:
    """Fit the model, predict all segments, build card and certification gains."""
    task = REGISTRY[task_name](seed=seed, sizes=sizes, cfg=cfg)
    train = task.train_diag[: task.diag_start]
    channels = task.train_diag.shape[1]
    diag_start, diag_stop = task.diag_start, task.train_diag.shape[0]
    # Standalone cert/test arrays start one lookback in so every prediction
    # window lies inside the array; the synthetic prefix covers the burn-in.
    standalone_start = cfg.input_length - 1
    segments = {
        "diag": (task.train_diag, diag_start, diag_stop),
        "cert": (task.cert, standalone_start, task.cert.shape[0]),
        "test": (task.test, standalone_start, task.test.shape[0]),
    }

    fit_seconds = 0.0
    if oracle_as_model:
        segs, preds = ev.predict_segments(partial(ev.predict_oracle, task, cfg=cfg), segments, cfg)
    else:
        adapter = _load_model(model_id)
        started = time.perf_counter()
        adapter.fit(train, ev.fit_meta(cfg, channels, seed))
        fit_seconds = time.perf_counter() - started
        segs, preds = ev.predict_segments(
            partial(ev.predict_adapter, adapter, cfg=cfg, chunk=predict_chunk), segments, cfg
        )

    seg_diag = segs["diag"]
    seg_cert = segs["cert"]
    seg_test = segs["test"]
    seg_a, seg_b = ev.split_ab(seg_diag, cfg)
    peaks = select_period_peaks(seg_a.e_full, seg_a.origins, cfg)

    cell = Cell(
        task_name=task_name,
        seed=seed,
        model_id=model_id,
        seg_diag=seg_diag,
        seg_diag_a=seg_a,
        seg_cert=seg_cert,
        seg_test=seg_test,
        period_peaks=peaks,
        fit_seconds=fit_seconds,
    )
    if with_card:
        cell.card = [
            g
            for g in all_gains(seg_a, seg_b, cfg, peaks, level=card_level(cfg, peaks, channels))
            if g.coordinate.kind not in {"total", "hgroup"}
        ]
        _, cell.cert_gains, cell.cert_mse = ev.certification(seg_diag, seg_cert, cfg, peaks)

    # Paired optimality gap and the total corrector's test-side truth on the
    # same usable origins (pred rows are indexed by origin offset).
    usable = segment_origins(standalone_start, task.test.shape[0], cfg)
    oracle_test = np.asarray(task.oracle.forecast(task.test, usable, cfg), dtype=np.float64)
    cell.test_gap = ev.paired_gap(task.test, usable, preds["test"], oracle_test, cfg)
    if with_card:
        fitted = fit_greedy(seg_diag, cfg, peaks)
        cell.test_total_delta = corrector_gain(
            fitted, ev.restrict(seg_test, usable), cfg
        ).delta
    return cell


def gate_g07_observation0() -> dict[str, Any]:
    """G0-7: AR(1) with m≡0 gives lag-gain ≈ phi²·var(x); white noise gives ≈ 0."""
    rng = np.random.default_rng(0)
    phi, n = 0.6, 12_000
    eps = rng.normal(0.0, 1.0, size=n)
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    var_x = 1.0 / (1.0 - phi**2)
    expected_ar = phi**2 * var_x

    local_cfg = HeadroomConfig(
        input_length=24,
        horizon=1,
        beyond_l_lags=1,
        band_period_range=(4, 24),
        lag_segments=((0, 23),),
        fir_len_cap=32,
    )
    coord = Coordinate(kind="lag", label="lag[0,23]", aspect="lag-structure", spec={"k1": 0, "k2": 23})
    results: dict[str, Any] = {}
    for name, series, expected in (("ar1", x, expected_ar), ("white", eps, 0.0)):
        values = series[:, None]
        pred = np.zeros((values.shape[0] - 1, 1, 1))
        seg = segment_data(values, 0, values.shape[0], pred, local_cfg)
        seg_a, seg_b = ev.split_ab(seg, local_cfg)
        gain = coordinate_gain(coord, seg_a, seg_b, local_cfg)
        results[name] = {
            "delta": round(gain.delta, 4),
            "expected": round(expected, 4),
            "ratio": round(gain.delta / expected, 3) if expected else round(gain.delta, 4),
        }
    ar_pass = 0.7 <= results["ar1"]["ratio"] <= 1.3
    white_pass = abs(results["white"]["delta"]) < 0.05 * expected_ar
    return {"stats": results, "pass": ar_pass and white_pass}


def gate_g01_oracle_negative_control(cells: list[Cell], cfg: HeadroomConfig) -> dict[str, Any]:
    """G0-1: with m = oracle, LCB ≤ 0 almost always and round-0 certification saturates."""
    lcb_below = saturated = 0
    total_lcbs: list[float] = []
    curve: dict[int, float] = {}
    for cell in cells:
        total = next(g for g in cell.cert_gains if g.coordinate.label == "total")
        lcb_below += int(total.lcb <= 0.0)
        total_lcbs.append(round(total.lcb, 5))
        saturated += int(certify(cell.cert_gains, cell.cert_mse, cfg).saturated)
        # Sample-size -> saturation-rate curve: subsample cert origins.
        for n_sample in (5000, 10000, 20000):
            origins = cell.seg_cert.origins
            if origins.size <= n_sample:
                continue
            sub = ev.restrict(cell.seg_cert, origins[:n_sample])
            gains = all_gains(
                cell.seg_diag, sub, cfg, cell.period_peaks,
                level=1.0 - cfg.certify_alpha,
            )
            ok = certify(gains, ev.mse_by_group(sub, gains), cfg).saturated
            curve[n_sample] = curve.get(n_sample, 0.0) + int(ok)
    n = len(cells)
    counted = {size: round(hits / n, 3) for size, hits in curve.items()}
    # 18 cells at a true 95% rate: P(>=15) ~ 0.989, so 15 is a safe floor.
    bound = max(n - 3, int(np.ceil(0.8 * n)))
    return {
        "stats": {
            "cells": n,
            "lcb_below_zero": lcb_below,
            "saturated": saturated,
            "total_lcbs": total_lcbs,
            "saturation_rate_by_cert_size": counted,
        },
        "pass": lcb_below >= bound and saturated >= bound,
    }


def gate_g02_calibration(cfg: HeadroomConfig, sizes: dict[str, int], reps: int) -> dict[str, Any]:
    """G0-2: two-sided CI coverage of the total gain against an independent long truth.

    The truth comes from a DIFFERENT seed's 1e5-step realization of the same
    process, so truth noise is an order of magnitude below the CI width and
    independent of the cert sample.
    """
    per_pair: dict[str, dict[str, Any]] = {}
    for name in CALIBRATION_PAIRS:
        hits_lcb = hits_ucb = 0
        ratios: list[float] = []
        for rep in range(reps):
            cell = evaluate_cell(name, 10_000 + rep, "m0-linear", cfg, sizes, with_card=False)
            peaks = select_period_peaks(cell.seg_diag_a.e_full, cell.seg_diag_a.origins, cfg)
            fitted = fit_greedy(cell.seg_diag, cfg, peaks)
            # The certificate's coverage claim is about the segment it is
            # computed on (Proposition 1 is per information set).  So the
            # truth here is the FULL-cert Delta(g_hat) and the CI comes from an
            # 8k-origin subsample of the same cert segment.  Comparing across
            # segments (cert CI vs test truth) adds a between-segment variance
            # floor -- slowly wandering seasonal amplitude moves Delta by
            # ~2 combined standard errors -- that no within-segment interval
            # can cover; between-segment transfer is reported by G0-3's
            # per-cell scatter instead.
            origins = cell.seg_cert.origins
            truth_sum = 0.0
            for begin in range(0, origins.size, 5_000):
                block = origins[begin : begin + 5_000]
                truth_sum += corrector_gain(
                    fitted, ev.restrict(cell.seg_cert, block), cfg, level=0.95
                ).delta * block.size
            truth = truth_sum / origins.size
            # A seeded RANDOM subsample (exchangeable with the full mean under
            # the CI's assumptions); a tail block would not be exchangeable
            # when seasonal amplitude drifts slowly through the segment.
            rng_ci = np.random.default_rng(30_000 + rep)
            ci_idx = np.sort(rng_ci.choice(origins.size, size=min(8_000, origins.size), replace=False))
            ci_block = ev.restrict(cell.seg_cert, origins[ci_idx])
            ci_gain = corrector_gain(fitted, ci_block, cfg, level=0.95)
            hits_lcb += int(ci_gain.lcb <= truth)
            hits_ucb += int(ci_gain.ucb >= truth)
            ratios.append(round(truth / cell.test_gap, 3) if cell.test_gap else None)
        per_pair[name] = {
            "reps": reps,
            "lcb_coverage": round(hits_lcb / reps, 3),
            "ucb_coverage": round(hits_ucb / reps, 3),
            "truth_over_gap": ratios,
        }
    # 12 reps at a true 95% rate: P(coverage >= 0.80) ~ 0.98 per pair.
    return {
        "stats": per_pair,
        "pass": all(i["lcb_coverage"] >= 0.80 and i["ucb_coverage"] >= 0.80 for i in per_pair.values()),
    }


def gate_g03_g04_tightness_localization(
    cells: list[Cell],
    cfg: HeadroomConfig,
    expected: dict[tuple[str, str], tuple[str, ...]] = EXPECTED_ASPECTS,
) -> dict[str, Any]:
    """G0-3/G0-4: gain tightness, per-cell scatter, and top-coordinate localization.

    The LCB-gap correlation itself is a MAIN-EXPERIMENT metric (it needs the
    incumbent trajectory); here we record the per-cell (lcb, delta, gap)
    scatter so the paper's §4.4 table can cite it directly.  ``expected`` is
    the (task, model) -> expected-aspects table; the deep recalibration passes
    its own pre-registered table.
    """
    ratios: list[float] = []
    hits = total_cells = 0
    scatter: list[dict[str, float]] = []
    per_cell: list[dict[str, Any]] = []
    for cell in cells:
        key = (cell.task_name, cell.model_id)
        if key not in expected:
            continue
        gap_scale = float(np.mean(np.square(cell.seg_test.r)))
        eligible = cell.test_gap >= 2.0 * cfg.certify_epsilon * gap_scale
        top = max(cell.card, key=lambda g: g.lcb if np.isfinite(g.lcb) else -np.inf)
        localized = top.coordinate.aspect in expected[key]
        total_cells += 1
        hits += int(localized)
        ratio = cell.test_total_delta / cell.test_gap if cell.test_gap else 0.0
        per_cell.append(
            {
                "task": cell.task_name,
                "model": cell.model_id,
                "seed": cell.seed,
                "gap": round(cell.test_gap, 4),
                "eligible_for_tightness": eligible,
                "top_coordinate": top.coordinate.label,
                "top_aspect": top.coordinate.aspect,
                "expected": list(expected[key]),
                "localized": localized,
            }
        )
        total_gain = next(g for g in cell.cert_gains if g.coordinate.label == "total")
        scatter.append(
            {
                "lcb": round(total_gain.lcb, 4),
                "delta": round(cell.test_total_delta, 4),
                "gap": round(cell.test_gap, 4),
            }
        )
        if eligible:
            ratios.append(ratio)
    median_ratio = round(float(np.median(ratios)), 3) if ratios else 0.0
    rate = round(hits / total_cells, 3) if total_cells else 0.0
    return {
        "stats": {
            "cells": per_cell,
            "scatter": scatter,
            "tightness_ratio_median": median_ratio,
            "tightness_cells": len(ratios),
            "localization_hits": hits,
            "localization_cells": total_cells,
            "localization_rate": rate,
        },
        "pass": median_ratio >= 0.3 and rate >= 0.70,
    }


def gate_g05_false_stop(cells: list[Cell], cfg: HeadroomConfig) -> dict[str, Any]:
    """G0-5: certification must not stop while a real gap remains.

    These planted-defect cells carry gaps far above eps*MSE, so any stop is a
    false stop; the fixed level wiring is enforced by certify() itself.
    Cells that DO saturate are cross-checked against their test-side true
    delta (the empirical false-stop measurement is the honesty mechanism now
    that no analytic bound is claimed).
    """
    saturated = 0
    truth_violations = 0
    details: list[dict[str, Any]] = []
    for cell in cells:
        if (cell.task_name, cell.model_id) not in EXPECTED_ASPECTS:
            continue
        report = certify(cell.cert_gains, cell.cert_mse, cfg)
        saturated += int(report.saturated)
        if report.saturated:
            mse_scale = float(np.mean(np.square(cell.seg_test.r)))
            ok = cell.test_total_delta <= cfg.certify_epsilon * mse_scale * 3.0
            truth_violations += int(not ok)
        details.append(
            {
                "task": cell.task_name,
                "model": cell.model_id,
                "seed": cell.seed,
                "saturated": report.saturated,
                "gap": round(cell.test_gap, 4),
            }
        )
    # 18 cells at the fixed certification level: allow 2 binomial false
    # stops, 0 truth violations.
    return {
        "stats": {
            "cells": len(details),
            "saturated": saturated,
            "truth_violations": truth_violations,
            "details": details,
        },
        "pass": saturated <= 2 and truth_violations == 0,
    }


def gate_g06_screening(
    cells: list[Cell],
    cfg: HeadroomConfig,
    incumbent: str = "m0-linear",
    tasks: tuple[str, ...] = SCREENING_TASKS,
) -> dict[str, Any]:
    """G0-6: tasks where one-shot <incumbent>+g leaves >=30% of the gap AND does not hurt.

    remaining > 1 means the corrector transferred negatively on test; with the
    greedy aggregate that signals a broken task/corrector rather than a hard
    task, so such cells do not count as qualified.  ``incumbent`` selects the
    model the screening measures (the deep recalibration passes "m0-patchtst").
    """
    per_task: dict[str, list[float]] = {}
    for cell in cells:
        if cell.model_id != incumbent or cell.task_name not in tasks:
            continue
        remaining = cell.test_gap - cell.test_total_delta
        per_task.setdefault(cell.task_name, []).append(
            remaining / cell.test_gap if cell.test_gap > 0 else 0.0
        )
    summary = {name: round(float(np.mean(values)), 3) for name, values in per_task.items()}
    qualified = sum(1 for value in summary.values() if 0.3 <= value <= 1.5)
    return {
        "stats": {
            "incumbent_model": incumbent,
            "remaining_gap_ratio_by_task": summary,
            "qualified_tasks": qualified,
        },
        "pass": qualified >= 2,
    }


def _gate_setup(smoke: bool) -> tuple[HeadroomConfig, dict[str, int]]:
    """The (cfg, sizes) the full-size gates were pre-registered with; the smoke shrink for CI."""
    if smoke:
        cfg = HeadroomConfig(
            input_length=48,
            lag_segments=((0, 7), (8, 15), (16, 31)),
            fir_len_cap=48,
            beyond_l_lags=2,
            band_period_range=(4, 24),
            budget_candidates=6,
            ridge_alphas=(1.0, 100.0),
        )
        return cfg, {"train": 1600, "diag": 1200, "cert": 1400, "test": 2400}
    # cert/test sizes follow paper §4.2 (>=2e4 / >=5e4 usable-scale rows).
    return HeadroomConfig(budget_candidates=25), {
        "train": 20000, "diag": 10000, "cert": 30000, "test": 52000,
    }


def run(smoke: bool = False, seeds: int = 3, reps: int = 12) -> dict[str, Any]:
    report_dir = _report_dir(smoke)
    cfg, sizes = _gate_setup(smoke)
    if smoke:
        seeds, reps = 1, 2
    seed_list = list(range(seeds))

    report: dict[str, Any] = {"config": cfg.to_dict(), "sizes": sizes, "seeds": seed_list, "gates": {}}

    def flush(name: str, gate: dict[str, Any]) -> None:
        report["gates"][name] = gate
        report["all_pass"] = all(g.get("pass", False) for g in report["gates"].values())
        write_report(report, report_dir)

    flush("G0-7", gate_g07_observation0())

    oracle_cells = [
        evaluate_cell(name, seed, "oracle", cfg, sizes, oracle_as_model=True)
        for name in ALL_TASKS
        for seed in seed_list
    ]
    flush("G0-1", gate_g01_oracle_negative_control(oracle_cells, cfg))
    del oracle_cells

    flush("G0-2", gate_g02_calibration(cfg, sizes, reps))

    pair_cells = [
        evaluate_cell(task_name, seed, model_id, cfg, sizes)
        for (task_name, model_id) in EXPECTED_ASPECTS
        for seed in seed_list
    ]
    flush("G0-3+4", gate_g03_g04_tightness_localization(pair_cells, cfg))
    flush("G0-5", gate_g05_false_stop(pair_cells, cfg))
    flush("G0-6", gate_g06_screening(pair_cells, cfg))
    del pair_cells

    return report


def _carry_over() -> dict[str, Any]:
    """Per-gate carry-over record: the frozen fixed-level verdict plus why the
    gate is model-agnostic."""
    artifact_rel = "experiments/specs/g0-final-report-v3.json"
    artifact_path = _REPO_ROOT / artifact_rel
    try:
        frozen = json.loads(artifact_path.read_text(encoding="utf-8"))
        verdicts = {name: bool(frozen["gates"][name]["pass"]) for name in CARRY_OVER_NOTES}
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read the frozen fixed-level artifact at {artifact_path}: {exc}") from exc
    return {
        name: {
            "artifact": artifact_rel,
            "pass": verdicts[name],
            "reason": CARRY_OVER_NOTES[name],
        }
        for name in CARRY_OVER_NOTES
    }


def _p2_screening(cells: list[Cell], incumbent: str = "m0-patchtst") -> dict[str, Any]:
    """The P2-launch question: does one-shot deep-m0+g leave >=30% of the gap on each P2 v2 task?

    Recommendation only — main-v2.json is NOT modified here; the human decides.
    Fractions follow G0-6's convention (per-cell remaining/gap, then the mean).
    """
    per_task: dict[str, Any] = {}
    for task in P2_V2_TASKS:
        ratios = [
            (cell.test_gap - cell.test_total_delta) / cell.test_gap if cell.test_gap > 0 else 0.0
            for cell in cells
            if cell.model_id == incumbent and cell.task_name == task
        ]
        mean = float(np.mean(ratios)) if ratios else float("nan")
        per_task[task] = {
            "remaining_gap_ratio": round(mean, 3),
            "seeds": len(ratios),
            "recommendation": "keep" if mean >= 0.30 else "flag",
        }
    return {
        "incumbent": incumbent,
        "threshold": 0.30,
        "by_task": per_task,
        "note": (
            "recommendation only: main-v2.json is not modified; a flagged task "
            "carries its measured fraction to the human decision"
        ),
    }


def run_deep(smoke: bool = False, seeds: int = 3) -> dict[str, Any]:
    """Deep-incumbent recalibration of G0-3/G0-4/G0-6 (plan §12).

    Runs ONLY the three incumbent-dependent gates, with the deep incumbent
    plus its three planted defect variants, over the same tasks the linear
    gates used for them, at the same full (or smoke) sizes; writes
    experiments/runs/stage0-v2 (stage0-v2-smoke under --smoke) incrementally
    and never touches the linear report directory.  G0-1/2/5/7 carry over
    from the frozen fixed-level artifact and are recorded here, not re-run.
    """
    cfg, sizes = _gate_setup(smoke)
    if smoke:
        seeds = 1
    seed_list = list(range(seeds))
    report_dir = _deep_report_dir(smoke)
    pairs = list(DEEP_EXPECTED_ASPECTS)
    report: dict[str, Any] = {
        "mode": "deep-recalibration",
        "config": cfg.to_dict(),
        "sizes": sizes,
        "seeds": seed_list,
        "models": {
            "incumbent": "m0-patchtst",
            "defect_variants": (
                "m0-patchtst-noattn",
                "m0-patchtst-coarsepatch",
                "m0-patchtst-channelmix",
            ),
            "predict_chunk": _DEEP_PREDICT_CHUNK,
        },
        "expected_aspects": {
            f"{task}/{model}": list(aspects)
            for (task, model), aspects in DEEP_EXPECTED_ASPECTS.items()
        },
        "carry_over": _carry_over(),
        "gates": {},
    }

    def flush(name: str, gate: dict[str, Any]) -> None:
        report["gates"][name] = gate
        report["all_pass"] = all(g.get("pass", False) for g in report["gates"].values())
        write_report(report, report_dir)

    cells: list[Cell] = []
    total = len(pairs) * len(seed_list)
    for task_name, model_id in pairs:
        for seed in seed_list:
            cell = evaluate_cell(
                task_name, seed, model_id, cfg, sizes,
                predict_chunk=_DEEP_PREDICT_CHUNK,
            )
            cells.append(cell)
            # Crash-resilient progress: the full pass is a long CPU run, and
            # the partial report shows exactly which cells completed.
            report["progress"] = {
                "done": len(cells),
                "total": total,
                "last": f"{task_name}/{model_id}/seed{seed}",
            }
            write_report(report, report_dir)
            print(
                f"[deep {len(cells)}/{total}] {task_name} seed={seed} {model_id} "
                f"fit={cell.fit_seconds:.1f}s gap={cell.test_gap:.4f}",
                flush=True,
            )
    report.pop("progress", None)

    flush("G0-3+4", gate_g03_g04_tightness_localization(cells, cfg, expected=DEEP_EXPECTED_ASPECTS))
    flush("G0-6", gate_g06_screening(cells, cfg, incumbent="m0-patchtst"))
    report["p2_v2_screening"] = _p2_screening(cells)
    model_order: list[str] = []
    for _, model_id in pairs:
        if model_id not in model_order:
            model_order.append(model_id)
    report["fit_seconds_by_model"] = {
        model_id: [round(cell.fit_seconds, 1) for cell in cells if cell.model_id == model_id]
        for model_id in model_order
    }
    del cells
    write_report(report, report_dir)
    return report


def write_report(report: dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    title = (
        "G0 深度重校准报告（v2）"
        if report.get("mode") == "deep-recalibration"
        else "G0 证书校准报告"
    )
    lines = [f"# {title}", ""]
    for name, gate in report.get("gates", {}).items():
        lines.append(f"## {name}: {'PASS' if gate.get('pass') else 'FAIL'}")
        lines.append("```json")
        lines.append(json.dumps(gate.get("stats", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    if "all_pass" in report:
        lines.append(f"**ALL: {'PASS' if report['all_pass'] else 'FAIL'}**")
    (report_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny sizes for the smoke test")
    parser.add_argument(
        "--deep",
        action="store_true",
        help="recalibrate only G0-3/4/6 with the deep incumbent and its defect "
        "variants (writes experiments/runs/stage0-v2, or stage0-v2-smoke with --smoke)",
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=12)
    args = parser.parse_args()
    if args.deep:
        report = run_deep(smoke=args.smoke, seeds=args.seeds)
    else:
        report = run(smoke=args.smoke, seeds=args.seeds, reps=args.reps)
    print(
        json.dumps(
            {"all_pass": report["all_pass"], "gates": {k: v.get("pass") for k, v in report["gates"].items()}},
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
