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

Variants (the stage-0 deep recalibration, plan §12; mirrors m0.py's
``variant()``): ``variant(model_id)`` builds the clean incumbent plus three
planted defect variants used to re-derive G0-3/G0-4/G0-6 around a deep
incumbent.  Each variant documents the residual signature the defect is meant
to leave (what the diagnostic card should localize):

- ``m0-patchtst`` — the clean default; this id is exactly ``create_adapter()``'s
  model (same class, same defaults, bitwise-identical fit).
- ``m0-patchtst-noattn`` — the attention branch removed from both encoder
  blocks; each patch is processed by the FFN alone, so the only cross-patch
  mixing left is the flattened-head Linear.  Intended signature: impaired
  within-window temporal integration — lag-structure (and, on periodic tasks,
  periodicity) tops the card.
- ``m0-patchtst-coarsepatch`` — patch_len/stride 48/48 (n_patches = 2 at
  L=96), collapsing within-window periodic resolution.  Intended signature:
  periodicity (and long-range-beyond-window) tops the card.
- ``m0-patchtst-channelmix`` — after the instance normalization every channel
  is replaced by the per-window cross-channel average, so the
  channel-independent backbone sees one averaged series and every channel's
  forecast is that average re-leveled by its own window statistics.
  Intended signature: cross-channel tops the card.  Needs C >= 2.

Every variant keeps the base contract: deterministic seeded fit,
row-independent fp64 predict, no batch statistics, and a ``model_id`` equal
to the variant id.
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

# Named variants: the clean incumbent plus the three stage-0 planted defects
# (plan §12; the deep recalibration of G0-3/4/6 runs on exactly these).
M0_NOATTN = "m0-patchtst-noattn"
M0_COARSEPATCH = "m0-patchtst-coarsepatch"
M0_CHANNELMIX = "m0-patchtst-channelmix"
MODEL_IDS = (MODEL_ID, M0_NOATTN, M0_COARSEPATCH, M0_CHANNELMIX)

_PATCH_LEN = 16
_PATCH_STRIDE = 8
# The coarsepatch defect: patches of 48 with stride 48 give n_patches = 2 at
# L=96 (and 1 at L=48), killing within-window periodic resolution.
_COARSE_PATCH_LEN = 48
_COARSE_PATCH_STRIDE = 48
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
    """Pre-norm transformer encoder block (attention + FFN, each with residual).

    ``attention=False`` is the noattn defect: the attention branch (and its
    norm) is not built at all, which is exactly a zero attention output — the
    residual add becomes a no-op — so each patch is processed by the FFN alone.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, *, attention: bool = True) -> None:
        super().__init__()
        self.attention = bool(attention)
        if self.attention:
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
        if self.attention:
            x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class _PatchTST(nn.Module):
    """Channel-independent backbone: [B*, 1, L] normalized -> [B*, H] normalized.

    ``patch_len``/``patch_stride`` carry the coarsepatch defect; ``attention``
    carries the noattn defect (see the module docstring).
    """

    def __init__(
        self,
        n_patches: int,
        horizon: int,
        *,
        attention: bool = True,
        patch_len: int = _PATCH_LEN,
        patch_stride: int = _PATCH_STRIDE,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv1d(1, _D_MODEL, kernel_size=patch_len, stride=patch_stride)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, _D_MODEL) * _POS_INIT_STD)
        self.blocks = nn.ModuleList(
            _EncoderBlock(_D_MODEL, _N_HEADS, _FFN_MULT, attention=attention)
            for _ in range(_N_BLOCKS)
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

    def __init__(
        self,
        *,
        attention: bool = True,
        patch_len: int = _PATCH_LEN,
        patch_stride: int = _PATCH_STRIDE,
        channel_mix: bool = False,
    ) -> None:
        """The clean incumbent by default; one defect knob per named variant.

        The named variants are exactly the four legal constructor
        configurations, so the id is derived rather than passed in (m0.py's
        pattern) and any other combination raises instead of masquerading as a
        named variant.
        """
        config = (bool(attention), (int(patch_len), int(patch_stride)), bool(channel_mix))
        legal: dict[tuple[bool, tuple[int, int], bool], str] = {
            (True, (_PATCH_LEN, _PATCH_STRIDE), False): MODEL_ID,
            (False, (_PATCH_LEN, _PATCH_STRIDE), False): M0_NOATTN,
            (True, (_COARSE_PATCH_LEN, _COARSE_PATCH_STRIDE), False): M0_COARSEPATCH,
            (True, (_PATCH_LEN, _PATCH_STRIDE), True): M0_CHANNELMIX,
        }
        if config not in legal:
            raise ValueError(
                f"unknown m0-patchtst variant configuration {config!r}; the named "
                f"variants are built by variant(), expected one of {MODEL_IDS}"
            )
        self.model_id = legal[config]
        self._attention = config[0]
        self._patch_len, self._patch_stride = config[1]
        self._channel_mix = config[2]
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
        if L < self._patch_len:
            raise ValueError(f"lookback L={L} cannot fit even one patch of length {self._patch_len}")
        if train.shape[1] != C:
            raise ValueError(f"train has {train.shape[1]} channels, but meta C={C}")
        if self._channel_mix and C < 2:
            raise ValueError(f"{M0_CHANNELMIX} needs at least 2 channels to confuse, got C={C}")
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
        # (train, meta) yields the same weights.  numpy is seeded alongside
        # torch so any sampling stays deterministic without a second thought.
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
        if self._channel_mix:
            # The channelmix defect: every channel slot receives the
            # cross-channel average of the normalized window.  All C slots
            # share one input while the (per-slot) targets still differ, so
            # the shared-weight backbone can only learn the channel-average
            # conditional forecast; per-channel and cross-channel structure
            # never reaches the model.
            xn = np.repeat(xn.mean(axis=2)[:, :, None], C, axis=2)

        n_windows = origins.size
        n_patches = (L - self._patch_len) // self._patch_stride + 1
        model = _PatchTST(
            n_patches, H, attention=self._attention,
            patch_len=self._patch_len, patch_stride=self._patch_stride,
        )

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
        # forcing eval mode here keeps the deterministic-eval-path contract
        # explicit (tests assert this).
        model.eval()
        xw = X[:, -self._L :, :]
        mean = xw.mean(axis=1)  # [B, C]
        std = np.sqrt(xw.var(axis=1) + _NORM_EPS)
        xn = (xw - mean[:, None, :]) / std[:, None, :]
        if self._channel_mix:
            # fit()'s defect, applied identically here: every channel slot
            # sees the averaged normalized window, so all C forecast rows of
            # one window are the same normalized average (the per-channel
            # mean/std below re-level it back onto each channel's scale).
            xn = np.repeat(xn.mean(axis=2)[:, :, None], self._C, axis=2)
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


def variant(model_id: str) -> M0DeepAdapter:
    """Build a named variant; unknown ids raise ValueError rather than silently defaulting.

    The defect and its intended residual signature (what the stage-0 card
    should localize) are documented per id in the module docstring; see
    MODEL_IDS for the full list.
    """
    if model_id == MODEL_ID:
        return M0DeepAdapter()
    if model_id == M0_NOATTN:
        return M0DeepAdapter(attention=False)
    if model_id == M0_COARSEPATCH:
        return M0DeepAdapter(patch_len=_COARSE_PATCH_LEN, patch_stride=_COARSE_PATCH_STRIDE)
    if model_id == M0_CHANNELMIX:
        return M0DeepAdapter(channel_mix=True)
    raise ValueError(f"unknown m0 model id {model_id!r}; expected one of {MODEL_IDS}")
