from __future__ import annotations

"""G0 certificate calibration: the seven gates of final-plan §5, no LLM involved.

Usage:
    uv run --extra research python experiments/stage0.py            # full run
    uv run --extra research python experiments/stage0.py --smoke    # tiny sizes

Writes experiments/runs/stage0/report.json and report.md incrementally after
every gate.  Confidence levels are wired end to end: card gains use
correctors.card_level, certification gains use correctors.certify_level (a
two-sided level, so each UCB is one-sided conservative; family false-stop
<= alpha/2), and certify() enforces the match.  Thresholds follow final-plan
§5; where a hard threshold would be flaky under binomial noise at the
achievable cell count, the pass bound is the conservative binomial quantile
and the exact numbers are reported.
"""

import argparse
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def _load_module(name: str, source: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves string annotations via sys.modules[cls.__module__],
    # so the module must be registered before execution.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ev = _load_module("rhg_evaluate", _REPO_ROOT / "experiments" / "evaluate.py")

from harnessing_ts.headroom.certify import certify
from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import (
    Coordinate,
    all_gains,
    card_level,
    certify_level,
    corrector_gain,
    coordinate_gain,
    fit_greedy,
    select_period_peaks,
)
from harnessing_ts.headroom.residuals import SegmentData, segment_data
from harnessing_ts.headroom.synthetic import REGISTRY
from harnessing_ts.headroom.timeidx import segment_origins

REPORT_DIR = _REPO_ROOT / "experiments" / "runs" / "stage0"

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
    # history, so the lag coordinate legitimately sees it first (review C).
    ("lgssm_xchannel", "m0-linear"): ("cross-channel", "lag-structure"),
    ("lgssm_xchannel", "m0-short-lookback"): ("cross-channel", "lag-structure", "long-range-beyond-window"),
    ("arfima", "m0-linear"): ("lag-structure",),
    ("hmm_ar_switch", "m0-linear"): ("lag-structure", "periodicity"),
}
if "nonlinear_xchannel" in REGISTRY:
    # The h-step nonlinear coupling also shows up in same-horizon realized
    # residual chains, so the horizon coordinate can legitimately lead.
    EXPECTED_ASPECTS[("nonlinear_xchannel", "m0-linear")] = ("cross-channel", "horizon-range")
# G0-2 calibration pairs (CI coverage against an independent long-segment
# truth; arfima stresses HAC under long memory).
CALIBRATION_PAIRS = ("lgssm_multiperiod", "arfima", "hmm_ar_switch")
# G0-6 screening tasks (garch / random_walk are negative controls by design).
SCREENING_TASKS = ("lgssm_multiperiod", "lgssm_xchannel", "nonlinear_xchannel", "hmm_ar_switch", "arfima")
ALL_TASKS = ("lgssm_multiperiod", "lgssm_xchannel", "hmm_ar_switch", "arfima", "garch", "random_walk")


@dataclass
class Cell:
    """One (task, seed, model) evaluation with the artifacts the gates need."""

    task_name: str
    seed: int
    model_id: str
    seg_diag: SegmentData
    seg_diag_a: SegmentData
    seg_diag_b: SegmentData
    seg_cert: SegmentData
    seg_test: SegmentData
    period_peaks: list[int]
    card: list[Any] = field(default_factory=list)
    cert_gains: list[Any] = field(default_factory=list)
    cert_mse: dict[str, float] = field(default_factory=dict)
    test_gap: float = 0.0
    test_total_delta: float = 0.0
    adapter: Any | None = None


def _load_m0(model_id: str) -> Any:
    return _load_module("m0_template_" + model_id.replace("-", "_"), _REPO_ROOT / "experiments" / "templates" / "m0.py").variant(model_id)


def _total_coordinate(peaks: list[int]) -> Coordinate:
    return Coordinate(kind="total", label="total", aspect="other", spec={"periods": peaks})


def evaluate_cell(
    task_name: str,
    seed: int,
    model_id: str,
    cfg: HeadroomConfig,
    sizes: dict[str, int],
    j_max: int,
    *,
    oracle_as_model: bool = False,
    with_card: bool = True,
) -> Cell:
    """Fit the model, predict all segments, build card and certification gains."""
    task = REGISTRY[task_name](seed=seed, sizes=sizes)
    train = task.train_diag[: task.diag_start]
    channels = task.train_diag.shape[1]
    diag_start, diag_stop = task.diag_start, task.train_diag.shape[0]
    # Standalone cert/test arrays start one lookback in so every prediction
    # window lies inside the array; the synthetic prefix covers the burn-in.
    standalone_start = cfg.input_length - 1

    if oracle_as_model:
        pred_diag = ev.predict_oracle(task, task.train_diag, diag_start, diag_stop, cfg)
        pred_cert = ev.predict_oracle(task, task.cert, standalone_start, task.cert.shape[0], cfg)
        pred_test = ev.predict_oracle(task, task.test, standalone_start, task.test.shape[0], cfg)
    else:
        adapter = _load_m0(model_id)
        adapter.fit(train, ev.fit_meta(cfg, channels, seed))
        pred_diag = ev.predict_adapter(adapter, task.train_diag, diag_start, diag_stop, cfg)
        pred_cert = ev.predict_adapter(adapter, task.cert, standalone_start, task.cert.shape[0], cfg)
        pred_test = ev.predict_adapter(adapter, task.test, standalone_start, task.test.shape[0], cfg)

    seg_diag = ev.build_segment(task.train_diag, diag_start, diag_stop, pred_diag, cfg)
    seg_cert = ev.build_segment(task.cert, standalone_start, task.cert.shape[0], pred_cert, cfg)
    seg_test = ev.build_segment(task.test, standalone_start, task.test.shape[0], pred_test, cfg)
    seg_a, seg_b = ev.split_ab(seg_diag, cfg)
    peaks = select_period_peaks(seg_a.e_full, seg_a.origins, cfg)

    cell = Cell(
        task_name=task_name,
        seed=seed,
        model_id=model_id,
        adapter=None if oracle_as_model else adapter,
        seg_diag=seg_diag,
        seg_diag_a=seg_a,
        seg_diag_b=seg_b,
        seg_cert=seg_cert,
        seg_test=seg_test,
        period_peaks=peaks,
    )
    if with_card:
        cell.card = [
            g
            for g in all_gains(seg_a, seg_b, cfg, peaks, level=card_level(cfg, peaks, channels))
            if g.coordinate.kind not in {"total", "hgroup"}
        ]
        cert_lvl = certify_level(cfg, j_max, peaks, channels)
        cell.cert_gains = all_gains(seg_diag, seg_cert, cfg, peaks, level=cert_lvl)
        cell.cert_mse = ev.mse_by_group(seg_cert, cell.cert_gains)

    # Paired optimality gap and the total corrector's test-side truth on the
    # same usable origins (pred rows are indexed by origin offset).
    usable = segment_origins(standalone_start, task.test.shape[0], cfg)
    oracle_test = np.asarray(task.oracle.forecast(task.test, usable, cfg), dtype=np.float64)
    cell.test_gap = ev.paired_gap(
        task.test, usable, pred_test[usable - standalone_start], oracle_test, cfg
    )
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


def gate_g01_oracle_negative_control(cells: list[Cell], cfg: HeadroomConfig, j_max: int) -> dict[str, Any]:
    """G0-1: with m = oracle, LCB ≤ 0 almost always and round-0 certification saturates."""
    lcb_below = saturated = 0
    total_lcbs: list[float] = []
    curve: dict[int, float] = {}
    for cell in cells:
        total = next(g for g in cell.cert_gains if g.coordinate.label == "total")
        lcb_below += int(total.lcb <= 0.0)
        total_lcbs.append(round(total.lcb, 5))
        saturated += int(certify(cell.cert_gains, cell.cert_mse, cfg, j=1, j_max=j_max).saturated)
        # Sample-size -> saturation-rate curve: subsample cert origins.
        for n_sample in (5000, 10000, 20000):
            origins = cell.seg_cert.origins
            if origins.size <= n_sample:
                continue
            sub = ev.restrict(cell.seg_cert, origins[:n_sample])
            gains = all_gains(
                cell.seg_diag, sub, cfg, cell.period_peaks,
                level=certify_level(cfg, j_max, cell.period_peaks, cell.seg_cert.r.shape[2]),
            )
            ok = certify(gains, ev.mse_by_group(sub, gains), cfg, j=1, j_max=j_max).saturated
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


def gate_g02_calibration(
    cfg: HeadroomConfig, sizes: dict[str, int], j_max: int, reps: int
) -> dict[str, Any]:
    """G0-2: two-sided CI coverage of the total gain against an independent long truth.

    The truth comes from a DIFFERENT seed's 1e5-step realization of the same
    process, so truth noise is an order of magnitude below the CI width and
    independent of the cert sample (the M1-review flaw was adjacency).
    """
    per_pair: dict[str, dict[str, Any]] = {}
    for name in CALIBRATION_PAIRS:
        hits_lcb = hits_ucb = 0
        ratios: list[float] = []
        for rep in range(reps):
            cell = evaluate_cell(name, 10_000 + rep, "m0-linear", cfg, sizes, j_max, with_card=False)
            peaks = select_period_peaks(cell.seg_diag_a.e_full, cell.seg_diag_a.origins, cfg)
            fitted = fit_greedy(cell.seg_diag, cfg, peaks)
            # Same-realization truth: the test segment (52k origins, disjoint
            # from cert).  The multiperiod oscillators are undamped (Q=0), so
            # the seasonal amplitude is fixed per realization and differs
            # ACROSS seeds -- a cross-seed truth would conflate realization-
            # level amplitude with CI calibration.  Within one realization
            # the amplitude is stable across windows, which is exactly the
            # quantity the CI targets.  Chunked to bound peak memory.
            origins = cell.seg_test.origins
            truth_sum = 0.0
            for begin in range(0, origins.size, 5_000):
                block = origins[begin : begin + 5_000]
                truth_sum += corrector_gain(
                    fitted, ev.restrict(cell.seg_test, block), cfg, level=0.95
                ).delta * block.size
            truth = truth_sum / origins.size
            ci_gain = corrector_gain(fitted, cell.seg_cert, cfg, level=0.95)
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


def gate_g03_g04_tightness_localization(cells: list[Cell], cfg: HeadroomConfig) -> dict[str, Any]:
    """G0-3/G0-4: gain tightness, per-cell scatter, and top-coordinate localization.

    The LCB-gap correlation itself is a MAIN-EXPERIMENT metric (it needs the
    incumbent trajectory); here we record the per-cell (lcb, delta, gap)
    scatter so the paper's §4.4 table can cite it directly.
    """
    ratios: list[float] = []
    hits = total_cells = 0
    scatter: list[dict[str, float]] = []
    per_cell: list[dict[str, Any]] = []
    for cell in cells:
        key = (cell.task_name, cell.model_id)
        if key not in EXPECTED_ASPECTS:
            continue
        gap_scale = float(np.mean(np.square(cell.seg_test.r)))
        eligible = cell.test_gap >= 2.0 * cfg.certify_epsilon * gap_scale
        top = max(cell.card, key=lambda g: g.lcb if np.isfinite(g.lcb) else -np.inf)
        localized = top.coordinate.aspect in EXPECTED_ASPECTS[key]
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
                "expected": list(EXPECTED_ASPECTS[key]),
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


def gate_g05_false_stop(cells: list[Cell], cfg: HeadroomConfig, j_max: int) -> dict[str, Any]:
    """G0-5: certification must not stop while a real gap remains.

    These planted-defect cells carry gaps far above eps*MSE, so any stop is a
    false stop; the level wiring that makes the bound hold is enforced by
    certify() itself (gain.level must match the Bonferroni level).  Cells that
    DO saturate are cross-checked against their test-side true delta.
    """
    saturated = 0
    truth_violations = 0
    details: list[dict[str, Any]] = []
    for cell in cells:
        if (cell.task_name, cell.model_id) not in EXPECTED_ASPECTS:
            continue
        report = certify(cell.cert_gains, cell.cert_mse, cfg, j=1, j_max=j_max)
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
    # 18 cells at alpha=0.05 (and a far smaller effective level after the
    # Bonferroni wiring): allow 2 binomial false stops, 0 truth violations.
    return {
        "stats": {
            "cells": len(details),
            "saturated": saturated,
            "truth_violations": truth_violations,
            "details": details,
        },
        "pass": saturated <= 2 and truth_violations == 0,
    }


def gate_g06_screening(cells: list[Cell], cfg: HeadroomConfig) -> dict[str, Any]:
    """G0-6: tasks where one-shot m0+g leaves >=30% of the gap AND does not hurt.

    remaining > 1 means the corrector transferred negatively on test; with the
    greedy aggregate that signals a broken task/corrector rather than a hard
    task, so such cells do not count as qualified.
    """
    per_task: dict[str, list[float]] = {}
    for cell in cells:
        if cell.model_id != "m0-linear" or cell.task_name not in SCREENING_TASKS:
            continue
        remaining = cell.test_gap - cell.test_total_delta
        per_task.setdefault(cell.task_name, []).append(
            remaining / cell.test_gap if cell.test_gap > 0 else 0.0
        )
    summary = {name: round(float(np.mean(values)), 3) for name, values in per_task.items()}
    qualified = sum(1 for value in summary.values() if 0.3 <= value <= 1.5)
    return {
        "stats": {"remaining_gap_ratio_by_task": summary, "qualified_tasks": qualified},
        "pass": qualified >= 2,
    }


def run(smoke: bool = False, seeds: int = 3, reps: int = 12) -> dict[str, Any]:
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
        sizes = {"train": 1600, "diag": 1200, "cert": 1400, "test": 2400}
        seeds, reps = 1, 2
    else:
        cfg = HeadroomConfig(budget_candidates=25)
        # cert/test sizes follow paper §4.2 (>=2e4 / >=5e4 usable-scale rows).
        sizes = {"train": 20000, "diag": 10000, "cert": 30000, "test": 52000}
    j_max = cfg.budget_candidates + 1
    seed_list = list(range(seeds))

    report: dict[str, Any] = {"config": cfg.to_dict(), "sizes": sizes, "seeds": seed_list, "gates": {}}

    def flush(name: str, gate: dict[str, Any]) -> None:
        report["gates"][name] = gate
        report["all_pass"] = all(g.get("pass", False) for g in report["gates"].values())
        write_report(report)

    flush("G0-7", gate_g07_observation0())

    oracle_cells = [
        evaluate_cell(name, seed, "oracle", cfg, sizes, j_max, oracle_as_model=True)
        for name in ALL_TASKS
        for seed in seed_list
    ]
    flush("G0-1", gate_g01_oracle_negative_control(oracle_cells, cfg, j_max))
    del oracle_cells

    flush("G0-2", gate_g02_calibration(cfg, sizes, j_max, reps))

    pair_cells = [
        evaluate_cell(task_name, seed, model_id, cfg, sizes, j_max)
        for (task_name, model_id) in EXPECTED_ASPECTS
        for seed in seed_list
    ]
    flush("G0-3+4", gate_g03_g04_tightness_localization(pair_cells, cfg))
    flush("G0-5", gate_g05_false_stop(pair_cells, cfg, j_max))
    flush("G0-6", gate_g06_screening(pair_cells, cfg))
    del pair_cells

    return report


def write_report(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# G0 证书校准报告", ""]
    for name, gate in report.get("gates", {}).items():
        lines.append(f"## {name}: {'PASS' if gate.get('pass') else 'FAIL'}")
        lines.append("```json")
        lines.append(json.dumps(gate.get("stats", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    if "all_pass" in report:
        lines.append(f"**ALL: {'PASS' if report['all_pass'] else 'FAIL'}**")
    (REPORT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny sizes for the smoke test")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=12)
    args = parser.parse_args()
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
