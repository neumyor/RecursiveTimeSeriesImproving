"""Non-LLM baselines for the main experiment (final-plan §4).

- one-shot m0+g: the greedy total linear corrector fitted once on the FULL
  diag segment and evaluated on test (A-M7) — `one_shot_m0g`.
- the uniform m+g fairness protocol shared with posthoc —
  `uniform_m0g_postprocess` (fit the corrector on the fit segment, apply it to
  any method's predictions on the apply segment).
- rule fixes: the four fixed repair recipes of §4 as M0Adapter variants — no
  diagnostics, no LLM — `rule_fix_variant` and `run_baselines`.
- classic per-channel statistical baselines for the REAL task —
  `stats_forecast_baseline` (statsforecast ARIMA/ETS).  Synthetic tasks
  already carry exact oracles, so they do not need it.

Everything reuses the headroom core (fit_greedy / FittedCorrector) and
evaluate.py's Adapter contract and segment assembly; nothing here reimplements
segment construction.
"""

from __future__ import annotations

from pathlib import Path
import sys
from dataclasses import replace
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND = str(_REPO_ROOT / "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.correctors import (
    FittedCorrector,
    build_features,
    fit_greedy,
    select_period_peaks,
)
from harnessing_ts.headroom.residuals import SegmentData
from harnessing_ts.headroom.synthetic import REGISTRY, SyntheticTask
from harnessing_ts.headroom.timeidx import Splits, segment_origins, windows


sys.path.insert(0, str(Path(__file__).resolve().parent))

import common

# evaluate.py holds the shared Adapter contract and segment/gap helpers.
ev = common.load_module("rhg_evaluate", _REPO_ROOT / "experiments" / "evaluate.py")
# The m0 template is the ridge core every rule fix extends; experiments/ is
# deliberately not on the repo pythonpath, so it is loaded by file path the
# same way tests/test_experiments_m0.py loads it.
m0 = common.load_module("rhg_m0_template", _REPO_ROOT / "experiments" / "templates" / "m0.py")

RULE_FOURIER = "rule-fourier"
RULE_LOWFREQ = "rule-lowfreq"
RULE_CROSSCHANNEL = "rule-crosschannel"
RULE_ARHEAD = "rule-arhead"
RULE_IDS = (RULE_FOURIER, RULE_LOWFREQ, RULE_CROSSCHANNEL, RULE_ARHEAD)

# Periods of the Fourier fix: the strong sub-daily/daily/weekly scales a fixed
# recipe may assume without diagnostics (the card arm earns its keep by picking
# the periods out of the residuals instead).
_FOURIER_PERIODS = (24, 48, 96, 168)
# Depth of the cross-channel and AR-head fixes' recent-lag blocks (lags 1..8).
_RECENT_LAGS = 8
# statsforecast choices, documented per the plan: fixed ARIMA(1,0,1) with the
# package's default CSS-ML estimation and no seasonal terms (the
# "auto_arima=False, order=(1,0,1)" speed choice), and the fixed
# additive/additive/no-seasonal ETS ("AAN", Holt's linear method).  Automatic
# order selection would multiply an already origin-by-origin cost for a
# baseline §4 only needs as a classic per-channel reference.
_ARIMA_ORDER = (1, 0, 1)
_ETS_MODEL = "AAN"
# Each origin x channel costs its own forward pass through the fitted model,
# so usable origins are subsampled to every _ORIGIN_STRIDE-th one; the reported
# MSE covers that subsample only (documented limitation).
_ORIGIN_STRIDE = 4


# --- shared assembly -----------------------------------------------------------------


def _fit_predict_segments(
    task: SyntheticTask, adapter: ev.Adapter, cfg: HeadroomConfig, seed: int
) -> tuple[np.ndarray, SegmentData, SegmentData]:
    """stage0's evaluate_cell assembly: fit on train, predict diag and test.

    Returns (test predictions at the usable origins, diag segment, test
    segment).  diag is predicted even when only test is scored so every caller
    shares one code path (the extra einsums cost milliseconds).
    """
    train = task.train_diag[: task.diag_start]
    adapter.fit(train, ev.fit_meta(cfg, train.shape[1], seed))
    diag_start, diag_stop = task.diag_start, task.train_diag.shape[0]
    # Standalone cert/test arrays start one lookback in so every prediction
    # window lies inside the array; the synthetic prefix covers the burn-in.
    standalone_start = cfg.input_length - 1
    pred_diag = ev.predict_adapter(adapter, task.train_diag, diag_start, diag_stop, cfg)
    pred_test = ev.predict_adapter(adapter, task.test, standalone_start, task.test.shape[0], cfg)
    seg_diag = ev.build_segment(task.train_diag, diag_start, diag_stop, pred_diag, cfg)
    seg_test = ev.build_segment(task.test, standalone_start, task.test.shape[0], pred_test, cfg)
    # pred rows are indexed over ALL realized origins; the segments keep only
    # the usable ones, so align before any per-origin pairing.
    pred_test_usable = pred_test[seg_test.origins - standalone_start]
    return pred_test_usable, seg_diag, seg_test


def _mse_and_gap(
    task: SyntheticTask, pred: np.ndarray, seg_test: SegmentData, cfg: HeadroomConfig
) -> tuple[float, float]:
    """(test MSE, paired oracle gap) at the test segment's usable origins."""
    oracle = np.asarray(task.oracle.forecast(task.test, seg_test.origins, cfg), dtype=np.float64)
    expected = (seg_test.origins.size, cfg.horizon, task.test.shape[1])
    if oracle.shape != expected:
        raise RuntimeError(f"oracle returned shape {oracle.shape}, expected {expected}")
    mse = float(np.mean(np.square(seg_test.r)))
    return mse, ev.paired_gap(task.test, seg_test.origins, pred, oracle, cfg)


def _test_mse_and_gap(
    task: SyntheticTask, adapter: ev.Adapter, cfg: HeadroomConfig, seed: int
) -> tuple[float, float]:
    """Fit an adapter on train; (test MSE, paired oracle gap) on one pass."""
    pred, _, seg_test = _fit_predict_segments(task, adapter, cfg, seed)
    return _mse_and_gap(task, pred, seg_test, cfg)


def _corrector_predictions(
    fitted: FittedCorrector, seg: SegmentData, cfg: HeadroomConfig
) -> np.ndarray:
    """The fitted corrector's residual prediction at seg.origins, [B, H, C].

    Mirrors corrector_gain's application exactly (same standardization and
    intercept, including the intercept-only empty selection), so gains read
    off these predictions match corrector_gain.
    """
    if fitted.coordinates:
        x = np.concatenate(
            [build_features(seg, coord, cfg)[0] for coord in fitted.coordinates], axis=1
        )
        y_hat = ((x - fitted.means) / fitted.stds) @ fitted.coef + fitted.intercept
    else:
        shape = (seg.origins.size, *seg.r.shape[1:])
        y_hat = np.broadcast_to(fitted.intercept.reshape(1, *seg.r.shape[1:]), shape)
    return np.asarray(y_hat, dtype=np.float64).reshape(seg.r.shape)


# --- one-shot m0+g and the uniform m+g protocol ---------------------------------------


def one_shot_m0g(
    task: SyntheticTask, adapter: ev.Adapter, cfg: HeadroomConfig, sizes: dict[str, int]
) -> dict[str, float]:
    """One-shot m0+g (final plan §4, A-M7): m0 fitted on train, the greedy
    total corrector fitted once on the FULL diag segment (SegmentData over the
    usable origins), both scored on the test usable origins against the
    paired oracle gap.

    Period peaks come from diag-A only (§2.3 A-M4), so the eval/test segment
    never touches the selection even though the fit rows span all of diag.
    The adapter's meta seed is cfg.seed: every baseline adapter here is
    deterministic, the seed only satisfies the candidate contract.
    """
    n_diag = task.train_diag.shape[0] - task.diag_start
    if sizes.get("train") != task.diag_start or sizes.get("diag") != n_diag:
        raise ValueError(
            f"sizes {sizes} does not describe the task: train_diag holds "
            f"train={task.diag_start}, diag={n_diag} rows"
        )
    pred_test, seg_diag, seg_test = _fit_predict_segments(task, adapter, cfg, cfg.seed)
    seg_a, _ = ev.split_ab(seg_diag, cfg)
    peaks = select_period_peaks(seg_a.e_full, seg_a.origins, cfg)
    fitted = fit_greedy(seg_diag, cfg, peaks)
    g_test = _corrector_predictions(fitted, seg_test, cfg)
    mse_m0, gap = _mse_and_gap(task, pred_test, seg_test, cfg)
    mse_m0g = float(np.mean(np.square(seg_test.r - g_test)))
    delta = mse_m0 - mse_m0g
    # remaining = 0 when there is no gap to close (stage0's G0-6 convention).
    remaining = (gap - delta) / gap if gap > 0.0 else 0.0
    return {
        "mse_m0": mse_m0,
        "mse_m0g": mse_m0g,
        "gap": gap,
        "delta": delta,
        "remaining_gap_ratio": remaining,
    }


def uniform_m0g_postprocess(
    pred: np.ndarray,
    seg_fit: SegmentData,
    seg_apply: SegmentData,
    cfg: HeadroomConfig,
    peaks: list[int],
) -> np.ndarray:
    """The fairness protocol: fit the greedy total corrector on ``seg_fit``
    (a run's full diag), apply the correction to ``pred`` on ``seg_apply``.

    ``pred`` carries one row per apply-segment usable origin ([B, H, C],
    aligned with seg_apply.origins); the return is the corrected predictions,
    same shape.
    """
    pred = np.asarray(pred, dtype=np.float64)
    expected = (seg_apply.origins.size, *seg_apply.r.shape[1:])
    if pred.shape != expected:
        raise ValueError(
            f"pred must align with seg_apply.origins, shape {expected}, got {pred.shape}"
        )
    fitted = fit_greedy(seg_fit, cfg, peaks)
    return pred + _corrector_predictions(fitted, seg_apply, cfg)


# --- rule fixes ----------------------------------------------------------------------


def _m0_fit_windows(
    train: np.ndarray, lookback: int, horizon: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """m0's fit windows: (win [N,K,C] oldest-first, z = win - x[o], y = x[o+h] - x[o]).

    Identical indexing to M0Adapter.fit (the ridge core being extended), kept
    here so the rule fixes' designs are built from exactly m0's rows.
    """
    T = train.shape[0]
    origins = np.arange(lookback - 1, T - horizon, stride)
    win = train[origins[:, None] + np.arange(-lookback + 1, 1)[None, :]]
    last = train[origins][:, None, :]
    y = train[origins[:, None] + np.arange(1, horizon + 1)[None, :]] - last
    return win, win - last, y


class _WindowExtraM0(m0.M0Adapter):
    """m0's ridge design plus per-channel extra columns, solved in ONE ridge.

    Extra columns are functions of the prediction window alone (I_o-measurable),
    so fit (on train windows) and predict share one code path.  The parent's
    fit runs first to reuse its meta/train validation; its plain-m0 weights are
    then replaced by the joint solve.
    """

    def _extra(self, win: np.ndarray) -> np.ndarray:
        """Window win [B, K, C] -> extra columns [B, F, C], level-free."""
        raise NotImplementedError

    def fit(self, train: np.ndarray, meta: dict) -> None:
        super().fit(train, meta)  # validation plus the plain-m0 weights as a byproduct
        win, z, y = _m0_fit_windows(train, self._K, self._H, m0._FIT_STRIDE)
        design = np.concatenate([z, self._extra(win)], axis=1)  # [N, K+F, C]
        eye = np.eye(design.shape[1])
        self._weights = np.empty((design.shape[1], self._H, self._C), dtype=np.float64)
        for c in range(self._C):
            d = design[:, :, c]
            self._weights[:, :, c] = np.linalg.solve(
                d.T @ d + m0._RIDGE_REG * eye, d.T @ y[:, :, c]
            )

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("predict_batch called before fit")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"X must be a [B, L, C] array, got shape {X.shape}")
        if X.shape[2] != self._C:
            raise ValueError(f"X has {X.shape[2]} channels, but fit with C={self._C}")
        if X.shape[1] < self._K:
            raise ValueError(f"X window has {X.shape[1]} rows, but lookback is K={self._K}")
        window = X[:, -self._K :, :]
        design = np.concatenate([window - window[:, -1:, :], self._extra(window)], axis=1)
        return np.einsum("bkc,khc->bhc", design, self._weights) + X[:, -1:, :]


class RuleFourier(_WindowExtraM0):
    """band -> Fourier fix: per-channel projections of the window onto sin/cos
    bases at fixed periods.

    The projections are linear in the window values, so they add no
    representational power over m0's K free lags; what they add is a
    low-dimensional periodic prior — under one shared ridge penalty K free
    lags blur a period-P oscillation while two basis columns carry it at full
    strength.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_id = RULE_FOURIER

    def _extra(self, win: np.ndarray) -> np.ndarray:
        K = win.shape[1]
        ks = np.arange(K)
        cols: list[np.ndarray] = []
        for period in _FOURIER_PERIODS:
            cols.append(np.sin(2.0 * np.pi * ks / period))
            cols.append(np.cos(2.0 * np.pi * ks / period))
        basis = np.stack(cols, axis=1)  # [K, F]
        z = win - win[:, -1:, :]
        # 1/sqrt(K): a white-noise projection has column scale ~sqrt(K/2)*sd,
        # so this keeps the extra columns comparable to the z columns under the
        # single shared ridge penalty.
        return np.einsum("bkc,kf->bfc", z, basis) / np.sqrt(K)


class RuleLowFreq(_WindowExtraM0):
    """beyond-L -> low-frequency phase fix: the window's cumulative level
    excess plus its two slowest Fourier bins stand in for the phase of an
    oscillation slower than the window (the candidate contract fixes L=96, so
    a longer lookback is not available to a rule fix either)."""

    def __init__(self) -> None:
        super().__init__()
        self.model_id = RULE_LOWFREQ

    def _extra(self, win: np.ndarray) -> np.ndarray:
        K = win.shape[1]
        ks = np.arange(K)
        basis = np.stack(
            [
                np.ones(K),  # cumulative sum: total level excess over the window
                np.cos(np.pi * ks / K),  # slowest in-window half cycle
                np.sin(np.pi * ks / K),
                np.cos(2.0 * np.pi * ks / K),
                np.sin(2.0 * np.pi * ks / K),
            ],
            axis=1,
        )
        z = win - win[:, -1:, :]
        return np.einsum("bkc,kf->bfc", z, basis) / np.sqrt(K)


class RuleCrossChannel(_WindowExtraM0):
    """channel -> cross-channel fix: each channel's design gains the other
    channels' recent values (lags 1..8, residualized by their own last value)."""

    def __init__(self) -> None:
        super().__init__()
        self.model_id = RULE_CROSSCHANNEL

    def _extra(self, win: np.ndarray) -> np.ndarray:
        B, K, C = win.shape
        if C < 2:
            # Raised from fit (the design builder), where the channel count
            # is first known for sure.
            raise ValueError(
                f"{RULE_CROSSCHANNEL} needs at least 2 channels, got C={C}"
            )
        if K < _RECENT_LAGS + 1:
            raise ValueError(
                f"{RULE_CROSSCHANNEL} reads lags 1..{_RECENT_LAGS} and needs "
                f"lookback >= {_RECENT_LAGS + 1}, got K={K}"
            )
        # Rows at lags 1.._RECENT_LAGS, differenced against the origin value of
        # the SAME channel (level-free, m0's z convention).
        dx = win[:, -_RECENT_LAGS - 1 : -1, :] - win[:, -1:, :]  # [B, 8, C]
        extra = np.empty((B, _RECENT_LAGS * (C - 1), C))
        for c in range(C):
            others = [j for j in range(C) if j != c]
            extra[:, :, c] = dx[:, :, others].reshape(B, -1)
        return extra


class RuleArHead(m0.M0Adapter):
    """lag -> AR(8) residual head: a second ridge fitted on the base model's
    train residuals.

    The head regresses the base's H-step forecast error on the 8 most recent
    within-window differences.  A literal AR(8) on realized residuals cannot
    run inside the Adapter contract — recomputing the base's residual at origin
    o-1 needs one row more than the L-row window carries — so the recent value
    differences stand in for the realized residual chain (documented deviation
    from the sketch; they are the part of e the window can always see).
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_id = RULE_ARHEAD
        self._head_coef: np.ndarray | None = None  # [8, H, C]
        self._head_intercept: np.ndarray | None = None  # [H, C]

    def fit(self, train: np.ndarray, meta: dict) -> None:
        super().fit(train, meta)  # the base m0 weights plus validation
        if self._K < _RECENT_LAGS + 1:
            raise ValueError(
                f"{RULE_ARHEAD} reads lags 1..{_RECENT_LAGS} and needs "
                f"lookback >= {_RECENT_LAGS + 1}, got K={self._K}"
            )
        _, z, y = _m0_fit_windows(train, self._K, self._H, m0._FIT_STRIDE)
        # Everything stays in m0's x[o]-differenced scale: y and the base's
        # forecast z @ W are both relative to x[o], so their difference is the
        # true forecast error (adding x[o] back here would corrupt the target
        # with the level and the head would learn to cancel it).
        base = np.einsum("bkc,khc->bhc", z, self._weights)
        resid = y - base  # [N, H, C]: the base's train forecast errors
        head_in = z[:, -_RECENT_LAGS - 1 : -1, :]  # lags 1..8, [N, 8, C]
        self._head_coef = np.empty((_RECENT_LAGS, self._H, self._C))
        self._head_intercept = np.empty((self._H, self._C))
        eye = np.eye(_RECENT_LAGS)
        for c in range(self._C):
            x = head_in[:, :, c]
            target = resid[:, :, c]
            mean_x, mean_y = x.mean(axis=0), target.mean(axis=0)
            centered = x - mean_x
            coef = np.linalg.solve(
                centered.T @ centered + m0._RIDGE_REG * eye,
                centered.T @ (target - mean_y),
            )
            self._head_coef[:, :, c] = coef
            self._head_intercept[:, c] = mean_y - mean_x @ coef

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        base = super().predict_batch(X)  # validation plus the plain-m0 forecast
        X = np.asarray(X, dtype=np.float64)
        z = X[:, -self._K :, :] - X[:, -1:, :]
        head_in = z[:, -_RECENT_LAGS - 1 : -1, :]  # [B, 8, C]
        correction = np.einsum("bkc,khc->bhc", head_in, self._head_coef) + self._head_intercept
        return base + correction


_VARIANTS: dict[str, type] = {
    RULE_FOURIER: RuleFourier,
    RULE_LOWFREQ: RuleLowFreq,
    RULE_CROSSCHANNEL: RuleCrossChannel,
    RULE_ARHEAD: RuleArHead,
}


def rule_fix_variant(model_id: str) -> ev.Adapter:
    """Build one rule-fix adapter (same fit/predict_batch contract as m0)."""
    try:
        cls = _VARIANTS[model_id]
    except KeyError:
        raise ValueError(
            f"unknown rule fix id {model_id!r}; expected one of {RULE_IDS}"
        ) from None
    return cls()


def run_baselines(
    task_name: str, seed: int, cfg: HeadroomConfig, sizes: dict[str, int]
) -> dict[str, float]:
    """m0, one-shot m0+g, and all four rule fixes on one task.

    Every entry is scored on the same test usable origins against the paired
    oracle gap; the result is one flat dict keyed "<model_id>.<metric>" for
    posthoc.  All four rule fixes run (picking the one the top card coordinate
    suggests would be over-engineering) so the report shows what each fixed
    recipe buys.  On a univariate task the cross-channel fix is inapplicable
    and its two entries are NaN.
    """
    if task_name not in REGISTRY:
        raise ValueError(f"unknown task {task_name!r}; expected one of {sorted(REGISTRY)}")
    cfg = replace(cfg, seed=seed)  # the adapter meta seed follows the run seed
    task = REGISTRY[task_name](seed=seed, sizes=sizes, cfg=cfg)
    results: dict[str, float] = {}

    m0g = one_shot_m0g(task, m0.create_adapter(), cfg, sizes)
    results["m0.mse"] = m0g["mse_m0"]
    results["m0.gap"] = m0g["gap"]
    results["m0g.mse"] = m0g["mse_m0g"]
    results["m0g.gap"] = m0g["gap"]
    results["m0g.delta"] = m0g["delta"]
    results["m0g.remaining_gap_ratio"] = m0g["remaining_gap_ratio"]

    for rule_id in RULE_IDS:
        if rule_id == RULE_CROSSCHANNEL and task.train_diag.shape[1] < 2:
            # Inapplicable, not zero: NaN keeps the key present for posthoc.
            results[f"{rule_id}.mse"] = float("nan")
            results[f"{rule_id}.gap"] = float("nan")
            continue
        mse, gap = _test_mse_and_gap(task, rule_fix_variant(rule_id), cfg, cfg.seed)
        results[f"{rule_id}.mse"] = mse
        results[f"{rule_id}.gap"] = gap
    return results


# --- classic statistical baselines (real task only) ----------------------------------


def stats_forecast_baseline(
    values: np.ndarray,
    cfg: HeadroomConfig,
    split: Splits,
    kind: str,
    *,
    origin_stride: int = _ORIGIN_STRIDE,
) -> dict[str, Any]:
    """Per-channel ARIMA/ETS baseline on a real task (plan §4).

    Each channel's model is fitted ONCE on the train slice; the H-step
    forecast at every usable test origin applies the fitted model to that
    origin's last ``input_length`` rows (statsforecast's ``forward`` rolls the
    state through the window).  Cost scales with origins x channels, so origins
    are subsampled to every ``origin_stride``-th (default
    ``_ORIGIN_STRIDE = 4``) usable one and the reported MSE covers that
    subsample only — a documented limitation of this baseline.

    Orders are fixed for speed: ARIMA(1,0,1) with statsforecast's default
    CSS-ML estimation and no seasonal terms, and the fixed
    additive/additive/no-seasonal ETS ("AAN", Holt's linear method).

    Raises RuntimeError when statsforecast is not installed.
    """
    if kind not in ("arima", "ets"):
        raise ValueError(f"kind must be 'arima' or 'ets', got {kind!r}")
    try:
        import statsforecast.models as sf
    except ImportError as exc:
        raise RuntimeError("statsforecast requires the research extras") from exc

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"values must be [T, C], got shape {values.shape}")
    if origin_stride < 1:
        raise ValueError(f"origin_stride must be >= 1, got {origin_stride}")
    train = np.ascontiguousarray(values[split.train])
    if train.shape[0] < 32:
        raise ValueError(
            f"train slice has {train.shape[0]} rows, too short for a per-channel ARIMA/ETS fit"
        )
    origins = segment_origins(split.test.start, split.test.stop, cfg)[::origin_stride]
    if origins.size == 0:
        raise ValueError("the test slice has no usable origins at this cfg")
    L, H, C = cfg.input_length, cfg.horizon, values.shape[1]
    _, y = windows(values, origins, cfg)  # [B, H, C] targets on the same origins
    pred = np.empty_like(y)
    window_idx = np.arange(-L + 1, 1)
    for c in range(C):
        series = np.ascontiguousarray(values[:, c])
        if kind == "arima":
            model = sf.ARIMA(order=_ARIMA_ORDER, season_length=1)
            model.fit(np.ascontiguousarray(train[:, c]))
            for i, o in enumerate(origins):
                window = np.ascontiguousarray(series[o + window_idx])
                pred[i, :, c] = model.forward(window, h=H)["mean"]
        else:
            fitted = sf.ets_f(np.ascontiguousarray(train[:, c]), m=1, model=_ETS_MODEL)
            for i, o in enumerate(origins):
                window = np.ascontiguousarray(series[o + window_idx])
                pred[i, :, c] = sf.forecast_ets(sf.forward_ets(fitted, window), H)["mean"]
    sq = np.square(y - pred)
    return {
        "mse": float(sq.mean()),
        "mse_by_channel": [float(sq[:, :, c].mean()) for c in range(C)],
        "n_origins": int(origins.size),
        "origin_stride": int(origin_stride),
    }
