"""Initial incumbent (m0-patchtst): a PatchTST-lite deep default candidate.

The paper direction wants the loop's starting incumbent to be a real deep
model, so hidden-layer analysis and structure exploration operate on
transformer activations rather than a linear map.  This file keeps the exact
Adapter contract of m0.py — ``create_adapter()`` returning an object with
``model_id`` ("m0-patchtst"), ``fit(train [T, C], meta {L, H, C, seed})`` and
``predict_batch(x) -> [B, H, C]`` — plus one probe extension:
``predict_batch(x, return_layers=True)`` also returns the patch-embedding and
per-block activations (see ``M0DeepAdapter.predict_batch``).

Architecture (deliberately small and CPU-fast):

- front end: per-window per-channel instance normalization (mean/std over the
  L rows of each window, eps 1e-5 inside the std, undone on the output) — the
  standard PatchTST/RevIN front end.  No train-set statistics are used at
  predict time, so each forecast is a function of its own window only, which
  is what the worker's single-vs-batch probe (atol 1e-9) demands.
- patching: patch_len 16, stride 8 -> n_patches = (L - 16) // 8 + 1 (11 at
  L=96, 5 at L=48); a trailing remainder shorter than patch_len is dropped.
- backbone: learnable positional embedding + 2 pre-norm transformer encoder
  blocks (d_model 64, 4 heads, FFN 4x with GELU).
- head: flatten patches -> Linear -> H outputs.
- channel independence (PatchTST default): every channel passes through the
  same weights with channels folded into the batch axis.
- no dropout, no BatchNorm anywhere: the eval path is deterministic and
  row-independent by construction (LayerNorm normalizes within one window).

Determinism: fit seeds torch and numpy from meta["seed"] before model init and
any sampling, trains with a single thread, and finishes in a fixed number of
epochs (no early stopping — fit sees only train), so two fits on the same
(train, meta) give identical weights.

Measured fit seconds at the calibrated config (torch 2.14.1 CPU, one thread,
Apple M4; train 20000 x C=4, L=96, H=24, 3 epochs, batch 128, AdamW lr 1e-3,
weight decay 1e-4): ~4.5s (repeat runs 4.5-5.1s) — ~50x headroom under the
240s calibration budget, so the loop's 600s candidate timeout is never the
binding constraint.  Small config (train 3000 x C=3, L=48, H=12, 3 epochs):
~1.4s including first-call torch warmup.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Single thread from the very first import: the loop runs many candidates on
# one machine, and a torch thread storm in a candidate is indistinguishable
# from a hang at the process-group level.  fit() repeats it defensively.
torch.set_num_threads(1)

MODEL_ID = "m0-patchtst"

_PATCH_LEN = 16
_PATCH_STRIDE = 8
_D_MODEL = 64
_N_HEADS = 4
_N_BLOCKS = 2
_FFN_MULT = 4
# Epochs calibrated on this M4 CPU: 3 epochs fit train[:20000] x C=4 in ~4.5s
# (target <= 240s; measured, see the module docstring), and clearly pass the
# seasonal sanity check.  Constant lr — no scheduler: nothing here needed
# annealing, and one fewer knob.
_EPOCHS = 3
_BATCH = 128
_LR = 1e-3
_WEIGHT_DECAY = 1e-4
_FIT_STRIDE = 4  # stride of training windows built inside train (same as m0)
_NORM_EPS = 1e-5  # inside the per-window std, so constant windows stay finite
_POS_INIT_STD = 0.02  # positional-embedding init, the usual transformer scale
_META_KEYS = ("L", "H", "C", "seed")


class _SelfAttention(nn.Module):
    """Multi-head self-attention over the patch axis, written out explicitly.

    No nn.MultiheadAttention and no F.scaled_dot_product_attention: both can
    dispatch to fused fast paths whose kernel choice depends on batch shape
    and dtype, and the probe path needs the exact same math in float64
    (predict) and float32 (train) with single-row vs batched inputs agreeing
    far below atol 1e-9.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B*, P, D] -> [B*, P, D]; each row attends to its own patches only."""
        b, p, _ = x.shape
        qkv = self.qkv(x).view(b, p, 3, self.n_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))  # [B*, H, P, hd]
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.head_dim))
        merged = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(b, p, -1)
        return self.out_proj(merged)


class _EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block (attention + FFN, each with residual)."""

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _SelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = ffn_mult * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the block OUTPUT: after attention+FFN and both residual adds.

        (Pre-norm: the captured activation is what the next block consumes.)
        """
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class _PatchTST(nn.Module):
    """Channel-independent backbone: [B*, 1, L] normalized -> [B*, H] normalized."""

    def __init__(self, n_patches: int, horizon: int) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(1, _D_MODEL, kernel_size=_PATCH_LEN, stride=_PATCH_STRIDE)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, _D_MODEL) * _POS_INIT_STD)
        self.blocks = nn.ModuleList(
            _EncoderBlock(_D_MODEL, _N_HEADS, _FFN_MULT) for _ in range(_N_BLOCKS)
        )
        self.head = nn.Linear(n_patches * _D_MODEL, horizon)

    def forward(
        self, x: torch.Tensor, layers: list[tuple[str, torch.Tensor]] | None = None
    ) -> torch.Tensor:
        """x [B*, 1, L] -> [B*, H]; probe activations appended to ``layers`` if given.

        Collection only appends — it changes no math, so predictions are
        bitwise identical with and without ``layers``.  Captured entries:
        "patch_embed" (after the positional add) then "block0", "block1"
        (each block's output), every tensor [B*, n_patches, d_model].
        """
        z = self.patch_embed(x).transpose(1, 2) + self.pos_embed
        if layers is not None:
            layers.append(("patch_embed", z))
        for i, block in enumerate(self.blocks):
            z = block(z)
            if layers is not None:
                layers.append((f"block{i}", z))
        return self.head(z.reshape(z.shape[0], -1))


class M0DeepAdapter:
    """PatchTST-lite forecaster behind the candidate Adapter contract.

    Training runs in float32 (CPU speed); after fit the weights are converted
    to float64 so predict_batch runs in float64: single-row and batched BLAS
    paths then differ by ~1e-15, orders of magnitude below the worker's
    single-vs-batch atol of 1e-9, instead of the ~1e-7 float32 rounding that
    would ride the tolerance edge for O(1) predictions.
    """

    model_id: str = MODEL_ID

    def __init__(self) -> None:
        self._model: _PatchTST | None = None
        self._L = 0
        self._H = 0
        self._C = 0
        self._n_patches = 0

    def fit(self, train: np.ndarray, meta: dict) -> None:
        """Fit on train [T, C] (standardized); meta must carry L, H, C, seed."""
        torch.set_num_threads(1)
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
        if L < _PATCH_LEN:
            raise ValueError(f"lookback L={L} cannot fit even one patch of length {_PATCH_LEN}")
        if train.shape[1] != C:
            raise ValueError(f"train has {train.shape[1]} channels, but meta C={C}")
        T = train.shape[0]
        if T < L + H:
            raise ValueError(f"train has {T} rows, need at least L+H={L + H} to build one window")
        seed = int(meta["seed"])

        # Window pairs from train only: origins o = L-1, L-1+stride, ... with
        # o+H <= T-1, so every input window and its targets lie fully inside
        # train (fit sees no data beyond train, exactly like m0).
        origins = np.arange(L - 1, T - H, _FIT_STRIDE)
        xw = train[origins[:, None] + np.arange(-L + 1, 1)[None, :]]  # [N, L, C]
        yw = train[origins[:, None] + np.arange(1, H + 1)[None, :]]  # [N, H, C]

        # Seed everything before model init and any sampling, so the same
        # (train, meta) yields the same weights.  numpy is seeded too even
        # though the loop below uses torch RNG only — any sampling added
        # later must stay deterministic without a second thought.
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Instance normalization per window per channel, computed from the
        # INPUT window alone: the model learns norm(x) -> norm(y), and
        # predict undoes the same stats (so loss and metric live in the same
        # normalized space, the standard PatchTST/RevIN setup).
        mean = xw.mean(axis=1)  # [N, C]
        std = np.sqrt(xw.var(axis=1) + _NORM_EPS)
        xn = (xw - mean[:, None, :]) / std[:, None, :]
        yn = (yw - mean[:, None, :]) / std[:, None, :]

        n_windows = origins.size
        n_patches = (L - _PATCH_LEN) // _PATCH_STRIDE + 1
        model = _PatchTST(n_patches, H)

        # Channel independence: [N, C, ...] layouts keep channel folding (a
        # plain reshape of [N, L, C] would interleave channels with time).
        x_t = torch.from_numpy(xn).float().permute(0, 2, 1).unsqueeze(2)  # [N, C, 1, L]
        y_t = torch.from_numpy(yn).float().permute(0, 2, 1)  # [N, C, H]
        optimizer = torch.optim.AdamW(model.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)
        model.train()
        for _ in range(_EPOCHS):
            perm = torch.randperm(n_windows)
            for start in range(0, n_windows, _BATCH):
                idx = perm[start : start + _BATCH]
                xb = x_t[idx].reshape(-1, 1, L)  # [B*C, 1, L]: channels in the batch axis
                yb = y_t[idx].reshape(-1, H)
                optimizer.zero_grad()
                loss = F.mse_loss(model(xb), yb)
                loss.backward()
                optimizer.step()
        model.eval()
        # One-time conversion: predict runs in float64 from here on.
        self._model = model.to(torch.float64)
        self._L = L
        self._H = H
        self._C = C
        self._n_patches = n_patches

    def predict_batch(
        self, X: np.ndarray, return_layers: bool = False
    ) -> np.ndarray | tuple[np.ndarray, dict[str, list[tuple[str, np.ndarray]]]]:
        """X [B, L, C] -> [B, H, C] float64; only the last L rows of X are read.

        Deterministic and row-independent: no RNG, eval mode, no_grad, and a
        float64 forward (see the class docstring for why float64).

        With ``return_layers=True`` returns ``(pred, info)`` where
        ``info["layers"]`` is ``[(name, ndarray), ...]`` with one entry per
        captured depth — "patch_embed", "block0", "block1" — each
        ``[B, n_patches, d_model]`` float32 numpy.  The backbone is
        channel-independent, so internally each depth is [B*C, P, D]; the
        dump averages the C per-channel copies to one tensor per window.
        The predictions are bitwise identical to the return_layers=False
        path (same forward; the probe only appends to a list).
        """
        if self._model is None:
            raise RuntimeError("predict_batch called before fit")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 3:
            raise ValueError(f"X must be a [B, L, C] array, got shape {X.shape}")
        if X.shape[2] != self._C:
            raise ValueError(f"X has {X.shape[2]} channels, but fit with C={self._C}")
        if X.shape[1] < self._L:
            raise ValueError(f"X window has {X.shape[1]} rows, but fit with L={self._L}")
        torch.set_num_threads(1)
        model = self._model
        # Defensive eval(): no dropout or BatchNorm exists in this model, but
        # forcing eval mode here makes the deterministic-eval-path contract
        # hold even if one is ever added (tests assert this).
        model.eval()
        xw = X[:, -self._L :, :]
        mean = xw.mean(axis=1)  # [B, C]
        std = np.sqrt(xw.var(axis=1) + _NORM_EPS)
        xn = (xw - mean[:, None, :]) / std[:, None, :]
        B = xw.shape[0]
        with torch.no_grad():
            layers: list[tuple[str, torch.Tensor]] | None = [] if return_layers else None
            x_t = torch.from_numpy(xn).permute(0, 2, 1).unsqueeze(2)  # [B, C, 1, L] float64
            out = model(x_t.reshape(-1, 1, self._L), layers)  # [B*C, H]
            pred = out.reshape(B, self._C, self._H).permute(0, 2, 1).numpy()  # [B, H, C]
        pred = pred * std[:, None, :] + mean[:, None, :]  # undo the instance norm
        if layers is None:
            return pred
        dump = [
            (name, act.reshape(B, self._C, self._n_patches, _D_MODEL).mean(axis=1).numpy().astype(np.float32))
            for name, act in layers
        ]
        return pred, {"layers": dump}


def create_adapter() -> M0DeepAdapter:
    """Candidate-contract entry point: the deep incumbent starts here."""
    return M0DeepAdapter()
