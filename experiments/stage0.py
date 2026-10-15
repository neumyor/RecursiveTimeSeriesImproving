from __future__ import annotations

"""G0 certificate calibration: the seven gates of final-plan §5, no LLM involved.

Usage:
    uv run --extra research python experiments/stage0.py            # full run
    uv run --extra research python experiments/stage0.py --smoke    # tiny sizes

Writes experiments/runs/stage0/report.json and report.md.  Every gate reports
its raw statistics plus a pass/fail verdict; the run passes only if all gates
pass.  Thresholds follow final-plan §5; where a hard threshold would be flaky
under binomial noise at the achievable cell count, the pass bound is relaxed
to the conservative binomial quantile and the exact numbers are reported.
"""

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import evaluate as ev
from harnessing_ts.headroom.certify import certify
from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import (
    Coordinate,
    all_gains,
    coordinate_gain,
    select_period_peaks,
)
from harnessing_ts.headroom.residuals import SegmentData, segment_data
from harnessing_ts.headroom.synthetic import REGISTRY
from harnessing_ts.headroom.timeidx import segment_origins

REPORT_DIR = _REPO_ROOT / "experiments" / "runs" / "stage0"

# Pre-registered (task, model) -> expected top-card aspects (final plan §5).
# Period-168 structure exceeds the lookback, so its missing information lives
# beyond the window: beyond-l is a legitimate top coordinate for multiperiod.
EXPECTED_ASPECTS: dict[tuple[str, str], tuple[str, ...]] = {
    ("lgssm_multiperiod", "m0-linear"): ("periodicity", "long-range-beyond-window"),
    ("lgssm_multiperiod", "m0-short-lookback"): ("periodicity", "long-range-beyond-window"),
    ("lgssm_multiperiod", "m0-horizon-bias"): ("periodicity", "long-range-beyond-window", "horizon-range", "level-bias"),
    ("lgssm_xchannel", "m0-linear"): ("cross-channel",),
    ("lgssm_xchannel", "m0-short-lookback"): ("cross-channel", "long-range-beyond-window"),
    ("arfima", "m0-linear"): ("lag-structure",),
    ("hmm_ar_switch", "m0-linear"): ("lag-structure", "periodicity"),
}
# G0-2 calibration pairs (total-coordinate CI coverage; arfima stresses HAC under long memory).
CALIBRATION_PAIRS = ("lgssm_multiperiod", "arfima", "hmm_ar_switch")
# G0-6 screening tasks (garch / random_walk are negative controls by design).
SCREENING_TASKS = ("lgssm_multiperiod", "lgssm_xchannel", "hmm_ar_switch", "arfima")
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


def _load_m0(model_id: str) -> Any:
    import importlib.util

    source = _REPO_ROOT / "experiments" / "templates" / "m0.py"
    spec = importlib.util.spec_from_file_location("m0_template", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.variant(model_id)


def _total_coordinate(peaks: list[int]) -> Coordinate:
    return Coordinate(kind="total", label="total", aspect="other", spec={"periods": peaks})


def evaluate_cell(
    task_name: str,
    seed: int,
    model_id: str,
    cfg: HeadroomConfig,
    sizes: dict[str, int],
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
        seg_diag=seg_diag,
        seg_diag_a=seg_a,
        seg_diag_b=seg_b,
        seg_cert=seg_cert,
        seg_test=seg_test,
        period_peaks=peaks,
    )
    if with_card:
        cell.card = [
            g for g in all_gains(seg_a, seg_b, cfg, peaks) if g.coordinate.kind not in {"total", "hgroup"}
        ]
        cell.cert_gains = all_gains(seg_diag, seg_cert, cfg, peaks)
        cell.cert_mse = ev.mse_by_group(seg_cert, cell.cert_gains)

    # Paired optimality gap on usable test origins: rows of pred_test are
    # indexed by origin offset from standalone_start.
    usable = segment_origins(standalone_start, task.test.shape[0], cfg)
    oracle_test = np.asarray(task.oracle.forecast(task.test, usable, cfg), dtype=np.float64)
    cell.test_gap = ev.paired_gap(
        task.test, usable, pred_test[usable - standalone_start], oracle_test, cfg
    )
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

    local_cfg = HeadroomConfig(horizon=1, lag_segments=((0, 23),), fir_len_cap=32)
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
    for cell in cells:
        total = next(g for g in cell.cert_gains if g.coordinate.label == "total")
        lcb_below += int(total.lcb <= 0.0)
        total_lcbs.append(round(total.lcb, 5))
        saturated += int(certify(cell.cert_gains, cell.cert_mse, cfg, j=1, j_max=j_max).saturated)
    n = len(cells)
    # 18 cells at a true 95% rate: P(>=15) ~ 0.9998, so 15 is a safe floor.
    bound = max(n - 3, int(np.ceil(0.8 * n)))
    return {
        "stats": {
            "cells": n,
            "lcb_below_zero": lcb_below,
            "saturated": saturated,
            "total_lcbs": total_lcbs,
        },
        "pass": lcb_below >= bound and saturated >= bound,
    }


def gate_g02_calibration(cfg: HeadroomConfig, sizes: dict[str, int], reps: int) -> dict[str, Any]:
    """G0-2: two-sided CI coverage of the total gain against a large-sample truth."""
    per_pair: dict[str, dict[str, Any]] = {}
    for name in CALIBRATION_PAIRS:
        hits_lcb = hits_ucb = 0
        ratios: list[float] = []
        for rep in range(reps):
            cell = evaluate_cell(name, 10_000 + rep, "m0-linear", cfg, sizes, with_card=False)
            origins = cell.seg_test.origins
            n_truth = min(10_000, origins.size // 2)
            truth_block = ev.restrict(cell.seg_test, origins[:n_truth])
            ci_block = ev.restrict(cell.seg_test, origins[n_truth + cfg.horizon : n_truth + cfg.horizon + 5_000])
            total = _total_coordinate(cell.period_peaks)
            truth_gain = coordinate_gain(total, cell.seg_diag, truth_block, cfg)
            ci_gain = coordinate_gain(total, cell.seg_diag, ci_block, cfg)
            hits_lcb += int(ci_gain.lcb <= truth_gain.delta)
            hits_ucb += int(ci_gain.ucb >= truth_gain.delta)
            ratios.append(round(truth_gain.delta / cell.test_gap, 3) if cell.test_gap else None)
        per_pair[name] = {
            "reps": reps,
            "lcb_coverage": round(hits_lcb / reps, 3),
            "ucb_coverage": round(hits_ucb / reps, 3),
            "truth_over_gap": ratios,
        }
    # 12 reps x 3 pairs at a true 95% rate: P(coverage >= 0.80) ~ 0.92 per pair.
    return {
        "stats": per_pair,
        "pass": all(i["lcb_coverage"] >= 0.80 and i["ucb_coverage"] >= 0.80 for i in per_pair.values()),
    }


def gate_g03_g04_tightness_localization(cells: list[Cell], cfg: HeadroomConfig) -> dict[str, Any]:
    """G0-3/G0-4: gain tightness on test and top-coordinate localization on diag."""
    ratios: list[float] = []
    hits = total_cells = 0
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
        if eligible:
            test_gain = coordinate_gain(
                _total_coordinate(cell.period_peaks), cell.seg_diag, cell.seg_test, cfg
            )
            ratios.append(test_gain.delta / cell.test_gap)
    median_ratio = round(float(np.median(ratios)), 3) if ratios else 0.0
    rate = round(hits / total_cells, 3) if total_cells else 0.0
    return {
        "stats": {
            "cells": per_cell,
            "tightness_ratio_median": median_ratio,
            "tightness_cells": len(ratios),
            "localization_hits": hits,
            "localization_cells": total_cells,
            "localization_rate": rate,
        },
        "pass": median_ratio >= 0.3 and rate >= 0.70,
    }


def gate_g05_false_stop(cells: list[Cell], cfg: HeadroomConfig, j_max: int) -> dict[str, Any]:
    """G0-5: certification must not stop while a real gap remains."""
    saturated = 0
    details: list[dict[str, Any]] = []
    for cell in cells:
        if (cell.task_name, cell.model_id) not in EXPECTED_ASPECTS:
            continue
        report = certify(cell.cert_gains, cell.cert_mse, cfg, j=1, j_max=j_max)
        saturated += int(report.saturated)
        details.append(
            {
                "task": cell.task_name,
                "model": cell.model_id,
                "seed": cell.seed,
                "saturated": report.saturated,
                "gap": round(cell.test_gap, 4),
            }
        )
    # 21 cells at alpha=0.05: P(<=2 false stops) ~ 0.92; 0 expected in practice.
    return {"stats": {"cells": len(details), "saturated": saturated, "details": details}, "pass": saturated <= 2}


def gate_g06_screening(cells: list[Cell], cfg: HeadroomConfig) -> dict[str, Any]:
    """G0-6: after one-shot m0+g, at least 2 screening tasks keep >=30% of the gap."""
    per_task: dict[str, list[float]] = {}
    for cell in cells:
        if cell.model_id != "m0-linear" or cell.task_name not in SCREENING_TASKS:
            continue
        post_gain = coordinate_gain(_total_coordinate(cell.period_peaks), cell.seg_diag, cell.seg_test, cfg)
        remaining = cell.test_gap - post_gain.delta
        per_task.setdefault(cell.task_name, []).append(remaining / cell.test_gap if cell.test_gap > 0 else 0.0)
    summary = {name: round(float(np.mean(values)), 3) for name, values in per_task.items()}
    qualified = sum(1 for value in summary.values() if value >= 0.3)
    return {
        "stats": {"remaining_gap_ratio_by_task": summary, "qualified_tasks": qualified},
        "pass": qualified >= 2,
    }


def run(smoke: bool = False, seeds: int = 3, reps: int = 12) -> dict[str, Any]:
    if smoke:
        cfg = HeadroomConfig(
            input_length=48,
            lag_segments=((0, 7), (8, 15), (16, 31)),
            fir_len_cap=32,
            budget_candidates=6,
            ridge_alphas=(1.0, 100.0),
        )
        sizes = {"train": 1600, "diag": 1200, "cert": 1400, "test": 2400}
        seeds, reps = 1, 2
    else:
        cfg = HeadroomConfig(budget_candidates=25)
        sizes = {"train": 20000, "diag": 10000, "cert": 12000, "test": 26000}
    j_max = cfg.budget_candidates + 1
    seed_list = list(range(seeds))

    report: dict[str, Any] = {"config": cfg.to_dict(), "sizes": sizes, "seeds": seed_list, "gates": {}}
    report["gates"]["G0-7"] = gate_g07_observation0()

    oracle_cells = [
        evaluate_cell(name, seed, "oracle", cfg, sizes, oracle_as_model=True)
        for name in ALL_TASKS
        for seed in seed_list
    ]
    report["gates"]["G0-1"] = gate_g01_oracle_negative_control(oracle_cells, cfg, j_max)
    del oracle_cells

    report["gates"]["G0-2"] = gate_g02_calibration(cfg, sizes, reps)

    pair_cells = [
        evaluate_cell(task_name, seed, model_id, cfg, sizes)
        for (task_name, model_id) in EXPECTED_ASPECTS
        for seed in seed_list
    ]
    report["gates"]["G0-3+4"] = gate_g03_g04_tightness_localization(pair_cells, cfg)
    report["gates"]["G0-5"] = gate_g05_false_stop(pair_cells, cfg, j_max)
    report["gates"]["G0-6"] = gate_g06_screening(pair_cells, cfg)
    del pair_cells

    report["all_pass"] = all(gate.get("pass", False) for gate in report["gates"].values())
    return report


def write_report(report: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# G0 证书校准报告", ""]
    for name, gate in report["gates"].items():
        lines.append(f"## {name}: {'PASS' if gate.get('pass') else 'FAIL'}")
        lines.append("```json")
        lines.append(json.dumps(gate.get("stats", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    lines.append(f"**ALL: {'PASS' if report['all_pass'] else 'FAIL'}**")
    (REPORT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny sizes for the smoke test")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--reps", type=int, default=12)
    args = parser.parse_args()
    report = run(smoke=args.smoke, seeds=args.seeds, reps=args.reps)
    write_report(report)
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
