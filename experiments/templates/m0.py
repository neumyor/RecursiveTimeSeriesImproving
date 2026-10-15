"""Initial parent model (m0): per-channel direct ridge forecaster on windows residualized by the last observed value."""

from __future__ import annotations

import numpy as np

# Named variants: the clean incumbent plus the two stage-0 planted defects
# (docs/residual-headroom-final-coding-plan.zh-CN.md §5).
M0_LINEAR = "m0-linear"
M0_SHORT = "m0-short-lookback"
M0_BIAS = "m0-horizon-bias"
MODEL_IDS = (M0_LINEAR, M0_SHORT, M0_BIAS)

_RIDGE_REG = 1000.0  # closed-form ridge regularization, frozen in the plan
_FIT_STRIDE = 4  # stride of training windows built inside train
_SHORT_LOOKBACK = 24  # the short-lookback defect reads only the last 24 rows
_BIAS_PER_STEP = 0.02  # the horizon-bias defect adds 0.02 * h to every h-step forecast
_META_KEYS = ("L", "H", "C", "seed")


class M0Adapter:
    """Direct multi-step ridge forecaster with channel-independent weights.

    Both features and targets are residualized by the last observed value:
    z_k = x[o-k] - x[o] (k = 0..K-1) predicts x[o+h] - x[o], so the forecast is
    x[o] + z @ W with one [K, H] weight matrix per channel. Lag k=0 is identically
    zero and ridge pins its weight to zero; keeping it makes "the last lookback_len
    rows" exact. There is no intercept and no RNG, so the same data always yields
    bitwise-identical predictions.
    """

    model_id: str

    def __init__(self, *, lookback_len: int | None = None, horizon_bias: float = 0.0) -> None:
        self._lookback_len = lookback_len
        self._horizon_bias = horizon_bias
        # The three named variants are exactly the three distinct constructor
        # argument combinations, so the id is derived rather than passed in.
        if horizon_bias != 0.0:
            self.model_id = M0_BIAS
        elif lookback_len is None:
            self.model_id = M0_LINEAR
        else:
            self.model_id = M0_SHORT
        self._weights: np.ndarray | None = None  # [K, H, C], set by fit

    def fit(self, train: np.ndarray, meta: dict) -> None:
        """Fit on train [T, C] (standardized); meta must carry L, H, C, seed."""
        missing = [key for key in _META_KEYS if key not in meta]
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
        # meta["seed"] is part of the candidate contract but unused: m0 is deterministic.

        # Origins o = K-1, K-1+stride, ... with o+H <= T-1, so every window and its
        # targets lie fully inside train (fit sees no data beyond train, A-S1/B-S3).
        origins = np.arange(K - 1, T - H, _FIT_STRIDE)
        z = train[origins[:, None] + np.arange(-K + 1, 1)[None, :]]  # [N, K, C] rows oldest first
        z -= train[origins][:, None, :]  # residualize by x[o]
        y = train[origins[:, None] + np.arange(1, H + 1)[None, :]]  # [N, H, C] targets
        y -= train[origins][:, None, :]  # x[o+h] - x[o]

        # One closed-form ridge solve per channel: channels never see each other
        # (this channel independence is what the `channel` coordinate is meant to find).
        eye = np.eye(K)
        self._weights = np.empty((K, H, C), dtype=np.float64)
        for c in range(C):
            zc = z[:, :, c]
            self._weights[:, :, c] = np.linalg.solve(zc.T @ zc + _RIDGE_REG * eye, zc.T @ y[:, :, c])
        self._K = K
        self._H = H
        self._C = C

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """X [B, L, C] -> [B, H, C]; only the last lookback_len rows of X are read."""
        if self._weights is None:
            raise RuntimeError("predict_batch called before fit")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"X must be a [B, L, C] array, got shape {X.shape}")
        if X.shape[2] != self._C:
            raise ValueError(f"X has {X.shape[2]} channels, but fit with C={self._C}")
        if X.shape[1] < self._K:
            raise ValueError(f"X window has {X.shape[1]} rows, but lookback is K={self._K}")
        z = X[:, -self._K :, :] - X[:, -1:, :]  # same residualization as in fit
        out = np.einsum("bkc,khc->bhc", z, self._weights) + X[:, -1:, :]
        # Added unconditionally (a zero bias is an exact no-op) so every variant
        # shares one code path; for m0-horizon-bias this is the planted defect.
        steps = np.arange(1, self._H + 1, dtype=np.float64)[None, :, None]
        return out + self._horizon_bias * steps


def create_adapter() -> M0Adapter:
    """Candidate-contract entry point: the m0 family starts at the clean linear variant."""
    return M0Adapter()


def variant(model_id: str) -> M0Adapter:
    """Build a named variant; unknown ids raise ValueError rather than silently defaulting."""
    if model_id == M0_LINEAR:
        return M0Adapter()
    if model_id == M0_SHORT:
        return M0Adapter(lookback_len=_SHORT_LOOKBACK)
    if model_id == M0_BIAS:
        return M0Adapter(horizon_bias=_BIAS_PER_STEP)
    raise ValueError(f"unknown m0 model id {model_id!r}; expected one of {MODEL_IDS}")
