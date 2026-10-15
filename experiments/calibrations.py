from __future__ import annotations

"""Pre-P2-freeze calibrations for review C's risk register items #4/#5.

Three calibrations, one JSON artifact each under experiments/runs/calibrations/
(written by main() as each finishes; the functions themselves only return):

- CAL-1 (risk #4, HAC bandwidth): the lgssm_multiperiod damped oscillators
  (rho = 0.999, amplitude half-life ~700 steps) outlive the HAC bandwidth floor
  max_lag + H = 359, so Bartlett truncation may understate the variance of the
  total-gain series.  Full sizes, seeds 0..7: fit the greedy total on diag,
  take the per-origin gain series d on cert, and recompute the 95% CI at
  bandwidth floors {359, 700, 1200, 2400}: (a) the width ratio vs 359, (b) a
  stage0-G0-2-style within-cert calibration (random 8k-origin subsample CI vs
  the full-cert mean, 8 reps per seed).  Decision rule: the smallest floor
  whose lcb AND ucb coverage is >= 0.9 on every seed.

- CAL-2 (risk #5, DM acceptance on a short real diag): ETTh1 leaves only ~2869
  usable diag origins under the 359 floor, so the loop's DM acceptance may be
  mis-sized on the real task.  (a) Null: candidates are m0 refit on bootstrap
  resamples of its own training windows (the fit varies, the expected loss
  does not) -> false-accept rate at alpha = 0.05 over 20 seeds.  (b) Planted:
  parent m0-short-lookback vs m0-linear candidates (a real improvement the
  agent could make) -> acceptance rate over 8 seeds.  (c) Round-0 certification
  of m0 on the real cert segment.  PASS rule: null <= 0.10 and planted >= 0.5.

- CAL-3 (negative control): the Exchange-Rate task (8 currencies, near random
  walk) should certify round 0 as eps-saturated: no headroom beyond noise.
  The loader has no timestamp_column parameter; its auto-detection treats an
  all-numeric grid as pure channels, so the CSV is written with ONLY the eight
  ch_0..ch_7 columns (row order is the index) -- an explicit integer index
  column would be loaded as a ninth ramp channel.

Usage:
    uv run --extra research python experiments/calibrations.py --all
    uv run --extra research python experiments/calibrations.py --only cal2
"""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


def _load_module(name: str, source: Path) -> Any:
    """Load a repo file that is not an installed package (baselines' pattern).

    An already-registered module is reused, so loading this file after stage0
    or the tests (same canonical names) shares one module object.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# evaluate.py holds the shared Adapter contract and segment/gap helpers; the
# experiments/ files are deliberately not on the repo pythonpath.
ev = _load_module("rhg_evaluate", _REPO_ROOT / "experiments" / "evaluate.py")
realdata = _load_module("rhg_realdata", _REPO_ROOT / "experiments" / "realdata.py")
m0 = _load_module("rhg_m0_template", _REPO_ROOT / "experiments" / "templates" / "m0.py")

from harnessing_ts.headroom.certify import certify
from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import (
    _min_bandwidth,
    all_gains,
    build_features,
    card_level,
    certify_level,
    fit_greedy,
    select_period_peaks,
)
from harnessing_ts.headroom.inference import auto_bandwidth, dm_one_sided_p, hac_variance, mean_ci
from harnessing_ts.headroom.synthetic import REGISTRY

OUT_DIR = _REPO_ROOT / "experiments" / "runs" / "calibrations"
DATA_DIR = _REPO_ROOT / "experiments" / "data"
ETTH1_CSV = DATA_DIR / "ETTh1.csv"
EXCHANGE_CSV = DATA_DIR / "exchange-rate.csv"
# Full sizes follow the pre-registered stage0 full run (paper 4.2).  CAL-1
# never reads the test segment, but the task is constructed at the frozen
# sizes so the calibration is exactly the pre-registered configuration.
FULL_SIZES = {"train": 20000, "diag": 10000, "cert": 30000, "test": 52000}
# 359 == _min_bandwidth(HeadroomConfig()) == max_lag + H; 700 ~ the oscillator
# amplitude half-life (the review's concern), 1200/2400 bracket it generously.
BANDWIDTH_FLOORS = (359, 700, 1200, 2400)
FILENAMES = {
    "cal1": "cal1-hac-bandwidth.json",
    "cal2": "cal2-etth1-acceptance.json",
    "cal3": "cal3-exchange-rate.json",
}


def _r(value: Any, digits: int = 8) -> float:
    return round(float(value), digits)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_safe(value: Any) -> Any:
    """NaN -> None recursively; numpy scalars -> plain Python for strict JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if np.isnan(number) else number
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


# --- shared pieces -------------------------------------------------------------------


def _gain_series(fitted: Any, seg: Any, cfg: HeadroomConfig) -> np.ndarray:
    """corrector_gain's per-origin series d = mean_{h,c}(r^2 - (r - g)^2).

    The CI plumbing inside corrector_gain hard-wires the config's bandwidth
    floor, so the calibrations rebuild the identical series here and hand it
    to inference.mean_ci at controlled bandwidths.  Mathematically this is
    restrict()-invariant: features at an origin read only rows at or before it.
    """
    if fitted.coordinates:
        x = np.concatenate([build_features(seg, coord, cfg)[0] for coord in fitted.coordinates], axis=1)
        y_hat = ((x - fitted.means) / fitted.stds) @ fitted.coef + fitted.intercept
        y_hat = y_hat.reshape(seg.r.shape)
    else:
        y_hat = np.broadcast_to(
            fitted.intercept.reshape(1, *seg.r.shape[1:]), (seg.origins.size, *seg.r.shape[1:])
        ).copy()
    gains = seg.r * seg.r - (seg.r - y_hat) ** 2
    return gains.mean(axis=(1, 2))


def _real_pieces(
    csv_path: Path, cfg: HeadroomConfig, embargo: int
) -> tuple[Any, dict[str, Any]]:
    """Load one real task through the de-identified loader (loop's prepare())."""
    task = realdata.load_real_task(Path(csv_path), cfg, embargo=embargo)
    return task, realdata.real_task_adapter_inputs(task, cfg)


def _fit_predict_real(
    adapter: Any, pieces: dict[str, Any], cfg: HeadroomConfig, seed: int
) -> tuple[Any, Any]:
    """Fit on train, predict diag + cert; (seg_diag, seg_cert) in loop convention."""
    channels = pieces["train"].shape[1]
    adapter.fit(pieces["train"], ev.fit_meta(cfg, channels, seed))
    pred_diag = ev.predict_adapter(
        adapter, pieces["diag_values"], pieces["diag_start"], pieces["diag_stop"], cfg
    )
    pred_cert = ev.predict_adapter(
        adapter, pieces["cert_values"], pieces["standalone_start"], pieces["cert_values"].shape[0], cfg
    )
    seg_diag = ev.build_segment(
        pieces["diag_values"], pieces["diag_start"], pieces["diag_stop"], pred_diag, cfg
    )
    seg_cert = ev.build_segment(
        pieces["cert_values"],
        pieces["standalone_start"],
        pieces["cert_values"].shape[0],
        pred_cert,
        cfg,
    )
    return seg_diag, seg_cert


def _dm_acceptance(seg_parent: Any, seg_candidate: Any, cfg: HeadroomConfig) -> dict[str, Any]:
    """The loop's acceptance gate: one-sided DM (HAC + HLN) at acceptance_alpha."""
    d = (np.square(seg_parent.r) - np.square(seg_candidate.r)).mean(axis=(1, 2))
    bandwidth = int(auto_bandwidth(d, _min_bandwidth(cfg)))
    dm_p = float(dm_one_sided_p(d, bandwidth))
    return {
        "dmP": _r(dm_p, 12),
        "accepted": bool(dm_p < cfg.acceptance_alpha),
        "meanD": _r(d.mean()),
        "bandwidth": bandwidth,
        "nOrigins": int(d.size),
    }


def _round0_certification(
    seg_diag: Any, seg_cert: Any, cfg: HeadroomConfig, peaks: list[int], channels: int
) -> dict[str, Any]:
    """The loop's round-0 certification of the incumbent (j = 1 of j_max)."""
    j_max = cfg.budget_candidates + 1
    level = certify_level(cfg, j_max, peaks, channels)
    gains = all_gains(seg_diag, seg_cert, cfg, peaks, level=level)
    mse = ev.mse_by_group(seg_cert, gains)
    report = certify(gains, mse, cfg, j=1, j_max=j_max)
    total = next(g for g in gains if g.coordinate.label == "total")
    threshold = cfg.certify_epsilon * mse["total"]
    return {
        "gains": gains,
        "report": report,
        "j_max": j_max,
        "total": total,
        "threshold": threshold,
    }


# --- CAL-1: HAC bandwidth sensitivity -------------------------------------------------


def _synthetic_gain_series(
    task_name: str, seed: int, model_id: str, cfg: HeadroomConfig, sizes: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Greedy-total gain series on cert: fit m0 on train, correctors on full diag.

    This is the certificate's total statistic (stage0's all_gains wiring)
    stripped to the one series CAL-1 needs: peaks come from diag-A, the greedy
    corrector is fitted on the FULL diag segment, and d is evaluated on cert.
    The test segment and oracle are never touched.
    """
    task = REGISTRY[task_name](seed=seed, sizes=sizes)
    train = task.train_diag[: task.diag_start]
    channels = train.shape[1]
    adapter = m0.variant(model_id)
    adapter.fit(train, ev.fit_meta(cfg, channels, seed))
    diag_start, diag_stop = task.diag_start, task.train_diag.shape[0]
    standalone_start = cfg.input_length - 1
    pred_diag = ev.predict_adapter(adapter, task.train_diag, diag_start, diag_stop, cfg)
    pred_cert = ev.predict_adapter(
        adapter, task.cert, standalone_start, task.cert.shape[0], cfg
    )
    seg_diag = ev.build_segment(task.train_diag, diag_start, diag_stop, pred_diag, cfg)
    seg_cert = ev.build_segment(
        task.cert, standalone_start, task.cert.shape[0], pred_cert, cfg
    )
    seg_a, _ = ev.split_ab(seg_diag, cfg)
    peaks = select_period_peaks(seg_a.e_full, seg_a.origins, cfg)
    fitted = fit_greedy(seg_diag, cfg, peaks)
    d = _gain_series(fitted, seg_cert, cfg)
    return seg_cert.origins, d, [c.label for c in fitted.coordinates], peaks


def cal1_bandwidth(
    cfg: HeadroomConfig | None = None,
    sizes: dict[str, int] | None = None,
    *,
    task_name: str = "lgssm_multiperiod",
    model_id: str = "m0-linear",
    seeds: Iterable[int] = range(8),
    floors: Iterable[int] = BANDWIDTH_FLOORS,
    reps: int = 8,
    ci_size: int = 8000,
    level: float = 0.95,
) -> dict[str, Any]:
    """CAL-1: does the 359 bandwidth floor understate the total-gain variance?"""
    cfg = cfg or HeadroomConfig(budget_candidates=25)
    sizes = sizes or FULL_SIZES
    floors = tuple(int(f) for f in floors)
    baseline = floors[0]
    if baseline != _min_bandwidth(cfg):
        raise ValueError(
            f"floors[0] = {baseline} must equal the current HAC floor "
            f"_min_bandwidth(cfg) = {_min_bandwidth(cfg)} (the ratio baseline)"
        )
    if max(floors) >= ci_size:
        raise ValueError(f"ci_size = {ci_size} must exceed the largest floor {max(floors)}")
    per_seed: dict[int, Any] = {}
    width_ratios: dict[int, list[float]] = {f: [] for f in floors if f != baseline}
    se_ratios: dict[int, list[float]] = {f: [] for f in floors if f != baseline}
    pooled = {f: {"lcb": 0, "ucb": 0} for f in floors}
    per_seed_pass = {f: True for f in floors}
    total_pairs = 0
    for seed in seeds:
        origins, d, labels, peaks = _synthetic_gain_series(task_name, int(seed), model_id, cfg, sizes)
        truth = float(d.mean())
        widths: dict[int, float] = {}
        ses: dict[int, float] = {}
        for floor in floors:
            bandwidth = auto_bandwidth(d, floor)
            _, lcb, ucb = mean_ci(d, bandwidth=bandwidth, level=level)
            widths[floor] = ucb - lcb
            ses[floor] = float(np.sqrt(hac_variance(d, bandwidth)))
        cover = {f: {"lcb": 0, "ucb": 0} for f in floors}
        # Stage0-G0-2's pattern: a seeded RANDOM subsample (exchangeable with
        # the full mean under the CI's assumptions) against the full-cert truth.
        rng = np.random.default_rng(30_000 + int(seed))
        for _rep in range(reps):
            ci_idx = np.sort(
                rng.choice(origins.size, size=min(ci_size, origins.size), replace=False)
            )
            d_sub = d[ci_idx]
            for floor in floors:
                _, lcb, ucb = mean_ci(
                    d_sub, bandwidth=auto_bandwidth(d_sub, floor), level=level
                )
                cover[floor]["lcb"] += int(lcb <= truth)
                cover[floor]["ucb"] += int(ucb >= truth)
        for floor in floors:
            pooled[floor]["lcb"] += cover[floor]["lcb"]
            pooled[floor]["ucb"] += cover[floor]["ucb"]
            per_seed_pass[floor] = per_seed_pass[floor] and (
                cover[floor]["lcb"] >= 0.9 * reps and cover[floor]["ucb"] >= 0.9 * reps
            )
        total_pairs += reps
        per_seed[int(seed)] = {
            "n_cert_origins": int(origins.size),
            "delta": _r(truth),
            "selected_coordinates": labels,
            "period_peaks": [int(p) for p in peaks],
            "widths": {f: _r(widths[f]) for f in floors},
            "se": {f: _r(ses[f]) for f in floors},
            "width_ratio": {f: _r(widths[f] / widths[baseline]) for f in floors},
            "coverage_lcb": {f: cover[f]["lcb"] / reps for f in floors},
            "coverage_ucb": {f: cover[f]["ucb"] / reps for f in floors},
        }
        for floor in width_ratios:
            width_ratios[floor].append(widths[floor] / widths[baseline])
            se_ratios[floor].append(ses[floor] / ses[baseline])
    pooled_coverage = {
        f: {"lcb": _r(pooled[f]["lcb"] / total_pairs), "ucb": _r(pooled[f]["ucb"] / total_pairs)}
        for f in floors
    }
    smallest = next((f for f in floors if per_seed_pass[f]), None)
    if smallest == baseline:
        recommendation = f"freeze the HAC floor at {baseline}"
    elif smallest is None:
        recommendation = f"no floor up to {max(floors)} meets the decision rule"
    else:
        recommendation = f"raise the HAC floor from {baseline} to {smallest}"
    ratio_1200 = width_ratios.get(1200)
    return {
        "name": "cal1-hac-bandwidth",
        "generated_utc": _utc_now(),
        "task": task_name,
        "model": model_id,
        "config": cfg.to_dict(),
        "sizes": dict(sizes),
        "seeds": [int(s) for s in seeds],
        "floors": list(floors),
        "reps_per_seed": reps,
        "ci_size": ci_size,
        "level": level,
        "per_seed": per_seed,
        "width_ratio_mean": {f: _r(float(np.mean(v))) for f, v in width_ratios.items()},
        "se_ratio_mean": {f: _r(float(np.mean(v))) for f, v in se_ratios.items()},
        "pooled_coverage": pooled_coverage,
        "verdict": {
            "coverage_ge_090_every_floor": all(
                pooled_coverage[f]["lcb"] >= 0.9 and pooled_coverage[f]["ucb"] >= 0.9
                for f in floors
            ),
            "width_ratio_at_1200": _r(float(np.mean(ratio_1200))) if ratio_1200 else None,
            "decision_rule": (
                "smallest floor whose lcb AND ucb coverage is >= 0.9 on every seed "
                f"({reps} subsample reps per seed)"
            ),
            "smallest_floor_all_seeds_ge_090": smallest,
            "freeze_or_raise": recommendation,
        },
    }


# --- CAL-2: ETTh1 DM-acceptance calibration -------------------------------------------


class _BootstrapRefitM0(m0.M0Adapter):
    """m0 refit on a bootstrap resample of its own training windows (the null).

    Keeps the null "no true improvement" while varying the fit: same window
    construction, same closed-form ridge, but the fit origins are drawn with
    replacement from m0's own origin grid.  Each sampled window keeps its
    intact rows, so unlike a row-resampled array no window ever straddles
    unrelated data.  The expected loss equals the plain refit's up to an
    O(1/n_windows) bootstrap bias, and mean_d is reported per draw so the
    null's centring stays visible.  fit mirrors m0.M0Adapter.fit exactly
    apart from the resample (the template is frozen post-M2).
    """

    def __init__(self, *, boot_seed: int, lookback_len: int | None = None) -> None:
        super().__init__(lookback_len=lookback_len)
        self._boot_seed = int(boot_seed)

    def fit(self, train: np.ndarray, meta: dict) -> None:
        missing = [key for key in m0._META_KEYS if key not in meta]
        if missing:
            raise ValueError(f"fit meta is missing required keys: {missing}")
        train = np.asarray(train, dtype=np.float64)
        if train.ndim != 2:
            raise ValueError(f"train must be a [T, C] array, got shape {train.shape}")
        L = int(meta["L"])
        H = int(meta["H"])
        C = int(meta["C"])
        if min(L, H, C) < 1:
            raise ValueError(f"meta L, H, C must be positive, got L={L}, H={H}, C={C}")
        if train.shape[1] != C:
            raise ValueError(f"train has {train.shape[1]} channels, but meta C={C}")
        if self._lookback_len is not None and not 1 <= self._lookback_len <= L:
            raise ValueError(f"lookback_len={self._lookback_len} must be in [1, L={L}]")
        K = L if self._lookback_len is None else self._lookback_len
        T = train.shape[0]
        if T < K + H:
            raise ValueError(f"train has {T} rows, need at least K+H={K + H} to build one window")
        origins = np.arange(K - 1, T - H, m0._FIT_STRIDE)
        rng = np.random.default_rng(self._boot_seed)
        origins = origins[rng.integers(0, origins.size, origins.size)]
        z = train[origins[:, None] + np.arange(-K + 1, 1)[None, :]]  # [N, K, C]
        z -= train[origins][:, None, :]  # residualize by x[o]
        y = train[origins[:, None] + np.arange(1, H + 1)[None, :]]  # [N, H, C]
        y -= train[origins][:, None, :]
        eye = np.eye(K)
        self._weights = np.empty((K, H, C), dtype=np.float64)
        for c in range(C):
            zc = z[:, :, c]
            self._weights[:, :, c] = np.linalg.solve(
                zc.T @ zc + m0._RIDGE_REG * eye, zc.T @ y[:, :, c]
            )
        self._K = K
        self._H = H
        self._C = C


def cal2_etth1_acceptance(
    csv_path: Path | str | None = None,
    cfg: HeadroomConfig | None = None,
    *,
    embargo: int = 96,
    n_null: int = 20,
    n_planted: int = 8,
) -> dict[str, Any]:
    """CAL-2: false-accept rate, planted-detection rate, round-0 certificate."""
    csv_path = Path(csv_path) if csv_path is not None else ETTH1_CSV
    cfg = cfg or HeadroomConfig(budget_candidates=25)
    task, pieces = _real_pieces(csv_path, cfg, embargo)
    channels = int(pieces["train"].shape[1])

    # Incumbent = the loop's round-0 parent: plain m0 fitted on train.
    parent = m0.variant("m0-linear")
    seg_parent, seg_cert = _fit_predict_real(parent, pieces, cfg, seed=0)
    seg_a, _ = ev.split_ab(seg_parent, cfg)
    peaks = select_period_peaks(seg_a.e_full, seg_a.origins, cfg)
    parent_diag_mse = float(np.mean(np.square(seg_parent.r)))

    # (a) Null: bootstrap refits of the same model class.  The bootstrap makes
    # the candidate's fit vary (the one thing m0's determinism cannot), while
    # the expected diag loss stays the incumbent's.
    null_rows = []
    for s in range(n_null):
        boot_seed = 1_000 + s
        candidate = _BootstrapRefitM0(boot_seed=boot_seed)
        seg_cand, _ = _fit_predict_real(candidate, pieces, cfg, seed=boot_seed)
        row = _dm_acceptance(seg_parent, seg_cand, cfg)
        row["bootSeed"] = boot_seed
        row["diagMse"] = _r(np.mean(np.square(seg_cand.r)))
        null_rows.append(row)
    null_rate = float(np.mean([row["accepted"] for row in null_rows]))

    # (b) Planted: m0-short-lookback parent vs m0-linear candidates -- the
    # lookback extension is the real improvement an agent could propose, and
    # the bootstrap keeps the 8 seeds from being one deterministic comparison.
    short = m0.variant("m0-short-lookback")
    seg_short, _ = _fit_predict_real(short, pieces, cfg, seed=0)
    planted_rows = []
    for s in range(n_planted):
        boot_seed = 2_000 + s
        candidate = _BootstrapRefitM0(boot_seed=boot_seed)
        seg_cand, _ = _fit_predict_real(candidate, pieces, cfg, seed=boot_seed)
        row = _dm_acceptance(seg_short, seg_cand, cfg)
        row["bootSeed"] = boot_seed
        row["diagMse"] = _r(np.mean(np.square(seg_cand.r)))
        planted_rows.append(row)
    planted_rate = float(np.mean([row["accepted"] for row in planted_rows]))
    deterministic_pair = _dm_acceptance(seg_short, seg_parent, cfg)

    # (c) Round-0 certification of m0 on the real cert segment.
    cert = _round0_certification(seg_parent, seg_cert, cfg, peaks, channels)
    total_ucb = float(cert["total"].ucb)
    threshold = float(cert["threshold"])

    return {
        "name": "cal2-etth1-acceptance",
        "generated_utc": _utc_now(),
        "csv": csv_path.name,
        "config": cfg.to_dict(),
        "embargo": embargo,
        "freq": task.structure["freq"],
        "channels": channels,
        "diag_usable_origins": int(seg_parent.origins.size),
        "cert_usable_origins": int(seg_cert.origins.size),
        "diag_origins_over_bandwidth": _r(seg_parent.origins.size / _min_bandwidth(cfg)),
        "parent_diag_mse": _r(parent_diag_mse),
        "null": {
            "n": n_null,
            "alpha": cfg.acceptance_alpha,
            "design": "m0 refit on bootstrap-resampled training windows vs plain m0",
            "acceptanceRate": _r(null_rate),
            "accepts": int(sum(row["accepted"] for row in null_rows)),
            "meanD": _r(float(np.mean([row["meanD"] for row in null_rows]))),
            "draws": null_rows,
        },
        "planted": {
            "n": n_planted,
            "design": "m0-short-lookback parent vs bootstrap m0-linear candidates",
            "acceptanceRate": _r(planted_rate),
            "accepts": int(sum(row["accepted"] for row in planted_rows)),
            "deterministic_pair": deterministic_pair,
            "draws": planted_rows,
        },
        "round0_certification": {
            "saturated": bool(cert["report"].saturated),
            "level": _r(cert["report"].level, 12),
            "jMax": cert["j_max"],
            "totalUcb": _r(total_ucb),
            "totalThreshold": _r(threshold),
            "totalMargin": _r(threshold - total_ucb),
            "totalUcbOverThreshold": _r(total_ucb / threshold),
            "failures": [
                {"label": f["label"], "ucb": _r(f["ucb"]), "threshold": _r(f["threshold"])}
                for f in cert["report"].failures
            ],
        },
        "verdict": {
            "rule": "acceptance_rate_null <= 0.10 and acceptance_rate_planted >= 0.5",
            "acceptance_rate_null": _r(null_rate),
            "acceptance_rate_planted": _r(planted_rate),
            "pass": bool(null_rate <= 0.10 and planted_rate >= 0.5),
        },
    }


# --- CAL-3: Exchange-Rate negative control --------------------------------------------


def cal3_exchange_rate(
    csv_path: Path | str | None = None,
    cfg: HeadroomConfig | None = None,
    *,
    embargo: int = 96,
) -> dict[str, Any]:
    """CAL-3: the near-random-walk task must certify round 0 as eps-saturated."""
    csv_path = Path(csv_path) if csv_path is not None else EXCHANGE_CSV
    cfg = cfg or HeadroomConfig(budget_candidates=25)
    task, pieces = _real_pieces(csv_path, cfg, embargo)
    channels = int(pieces["train"].shape[1])

    adapter = m0.variant("m0-linear")
    seg_diag, seg_cert = _fit_predict_real(adapter, pieces, cfg, seed=0)
    seg_a, seg_b = ev.split_ab(seg_diag, cfg)
    peaks = select_period_peaks(seg_a.e_full, seg_a.origins, cfg)

    cert = _round0_certification(seg_diag, seg_cert, cfg, peaks, channels)
    total = cert["total"]

    # The loop's posthoc-card rows on diag (A fit, B evaluate, card level),
    # sorted by Bonferroni LCB; the top rows are what the card arm would see.
    card = all_gains(seg_a, seg_b, cfg, peaks, level=card_level(cfg, peaks, channels))
    rows = sorted(
        (g for g in card if g.coordinate.kind not in {"total", "hgroup"}),
        key=lambda g: g.lcb,
        reverse=True,
    )
    card_total = next(g for g in card if g.coordinate.label == "total")

    return {
        "name": "cal3-exchange-rate",
        "generated_utc": _utc_now(),
        "csv": csv_path.name,
        "config": cfg.to_dict(),
        "embargo": embargo,
        "freq": task.structure["freq"],
        "channels": channels,
        "diag_usable_origins": int(seg_diag.origins.size),
        "cert_usable_origins": int(seg_cert.origins.size),
        "period_peaks": [int(p) for p in peaks],
        "diag_mse": _r(np.mean(np.square(seg_diag.r))),
        "cert_mse": _r(np.mean(np.square(seg_cert.r))),
        "round0": {
            "saturated": bool(cert["report"].saturated),
            "level": _r(cert["report"].level, 12),
            "jMax": cert["j_max"],
            "totalDelta": _r(total.delta),
            "totalLcb": _r(total.lcb),
            "totalUcb": _r(total.ucb),
            "totalThreshold": _r(cert["threshold"]),
            "failures": [
                {"label": f["label"], "ucb": _r(f["ucb"]), "threshold": _r(f["threshold"])}
                for f in cert["report"].failures
            ],
        },
        "top_card_rows": [
            {
                "label": g.coordinate.label,
                "aspect": g.coordinate.aspect,
                "delta": _r(g.delta),
                "lcb": _r(g.lcb),
                "ucb": _r(g.ucb),
            }
            for g in rows[:5]
        ],
        "diag_card_total": {
            "delta": _r(card_total.delta),
            "lcb": _r(card_total.lcb),
            "ucb": _r(card_total.ucb),
        },
        "verdict": {
            "claim": "round-0 certificate reports no headroom beyond noise",
            "saturated": bool(cert["report"].saturated),
            "pass": bool(cert["report"].saturated),
        },
    }


# --- entry point ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="run all three calibrations")
    parser.add_argument("--only", choices=("cal1", "cal2", "cal3"), help="run one calibration")
    args = parser.parse_args()
    if not args.all and args.only is None:
        parser.error("pass --all or --only {cal1,cal2,cal3}")
    # cal2/cal3 land their results in minutes; cal1 runs ~15-25 min, so it goes
    # last and a partial --all still leaves both quick artifacts on disk.
    order = [name for name in ("cal2", "cal3", "cal1") if args.all or name == args.only]
    runners = {
        "cal1": cal1_bandwidth,
        "cal2": cal2_etth1_acceptance,
        "cal3": cal3_exchange_rate,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for name in order:
        started = time.time()
        print(f"[calibrations] {name} started", flush=True)
        try:
            result: dict[str, Any] = runners[name]()
        except Exception as exc:  # persist the failure, keep the rest running
            failures.append(name)
            result = {
                "name": name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"[calibrations] {name} FAILED: {exc}", flush=True)
        result["seconds"] = round(time.time() - started, 1)
        _write_json(OUT_DIR / FILENAMES[name], result)
        print(
            f"[calibrations] {name} written to {OUT_DIR / FILENAMES[name]} "
            f"({result['seconds']}s)",
            flush=True,
        )
    if failures:
        raise SystemExit(f"calibrations failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
