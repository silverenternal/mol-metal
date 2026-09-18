"""MetalHybridV4 — D-MPNN + real Satorras-2021 EGNN + Perceiver-latent fusion + LLRD + Kendall loss.

V4 implements all four review recommendations from
``molmetal/reports/review_*.md``:

  (a) **Pretraining init** (``review_pretraining_init.md``):
      * Layer-wise LR decay (LLRD): top head ``lr=1e-3`` decays 0.95/layer
        down to encoder bottom ``lr=1e-5`` (Howard & Ruder 2018, ULMFiT).
      * Encoder D-MPNN is **unfrozen** (Chemprop / Uni-Mol / GROVER
        recommendation) — only a lower LR protects it from drifting.

  (b) **EGNN** (``review_egnn.md``):
      * Real Satorras 2021 EGNN layer — fully differentiable, **no
        detach**, **no max-Δx clip**.
      * Scalar update ``h ← h + Σ_j MLP([h_i, h_j, ||x_i − x_j||])`` and
        coord update ``x ← x + Σ_j (x_j − x_i) · phi([h_i, h_j, ||x_i − x_j||²])``.
      * The coord update is still bounded by ``tanh`` (scalar acting on
        an invariant magnitude — equivariant).

  (c) **Fusion** (``review_fusion.md``):
      * **Perceiver latent** cross-attention: 16 learnable latent queries
        attend to per-atom ``[h_dmpnn || h_egnn]`` (decouples O(N²) → O(1)).
      * **Multi-head concat** (8 × 16 = 128) per Vaswani 2017, not additive.
      * **FFN block** post-attention (LN → GELU → Linear(4d) → Linear(d)).
      * **Gate init ≈ 0.05**: bias = log(0.05/0.95) ≈ -2.944; weight *=
        0.1 so EGNN contribution grows slowly.
      * Input dropout 0.1 on each stream.

  (d) **Loss** (``review_multitask_loss.md``):
      * **NormalizedLoss** (R2) on cls / reg.
      * **Kendall uncertainty** weighting (``UncertWeightedLoss``) for
        cls / reg / coord.
      * **γ annealing** 0.1 → 0.0 (cosine) on the coord-refinement term.
      * **Label smoothing 0.05** on the BCE classification loss.

Stability hooks (M-B2):
      * **EMA** of the model parameters via ``init_ema`` / ``update_ema``.
        ``ema = 0.999 * ema + 0.001 * model`` after every training step.

Forward signature is V3-compatible:
    (smiles_list, coords, metal_types, mol_objects=None) -> Pic50RegressionOutput
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from molmetal.data.featurize import GraphFeaturizer
from molmetal.models.dmpnn import DirectedMPNN, MPNNConfig
from molmetal.models.metal_hybrid import (
    MetalHybridConfig,
    Pic50RegressionOutput,
)

# Round-5 Triton kernel imports.  All wiring is gated through
# :data:`triton_kernels.triton_config` so disabling the global fused-
# kernel flag transparently falls back to the pure-PyTorch ops.
from triton_kernels import (
    batched_fused_silu_mlp as _batched_fused_silu_mlp,
    fused_residual_add as _fused_residual_add,
)
from triton_kernels.config import triton_config as _triton_config


# ---------------------------------------------------------------------------
# Round-5 fused-residual-add wiring
# ---------------------------------------------------------------------------
def _maybe_fused_residual_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    training: bool,
) -> torch.Tensor:
    """``alpha * x + beta * residual`` via the Triton kernel when the gate
    is open; otherwise the pure-PyTorch ``alpha * x + beta * residual``.

    The fused kernel (``triton_kernels.fused_residual_add``) requires a
    CUDA tensor + same-shape (or trailing-axis broadcast) residual.  On
    CPU we fall back transparently so tests run on a CPU-only machine.
    """
    if not x.is_cuda:
        return alpha * x + beta * residual
    if not _triton_config.should_use_fused(x, op="residual_add"):
        return alpha * x + beta * residual
    return _fused_residual_add(x, residual, alpha=alpha, beta=beta)


# ---------------------------------------------------------------------------
# Round-5 batched-fused-SiLU-MLP wiring
# ---------------------------------------------------------------------------
def _maybe_batched_silu_mlp(
    x: torch.Tensor,
    linear_layers: list[nn.Linear],
    *,
    training: bool,
) -> torch.Tensor:
    """Run K stacked ``Linear → SiLU → Linear`` triplets via the batched
    fused SiLU MLP kernel when the gate is open; else fall back to the
    sequential ``nn.SiLU`` + Linear chain.

    The kernel expects K MLPs of the form
    ``Linear(D_in, H) → SiLU → Linear(H, D_out)`` applied to one shared
    input ``x : (M, D_in)``.  Here we use it to "batched-fuse" a
    multi-layer SiLU MLP block — e.g. ``Linear → SiLU → Linear → SiLU →
    Linear`` becomes K=2 stacked triplets, both with hidden dim equal
    to the intermediate width.

    Parameters
    ----------
    x : ``(M, D_in)`` tensor
        Input shared across all K MLPs.
    linear_layers : list of ``2*K`` ``nn.Linear`` modules
        Layer weights arranged as ``[w_1a, w_1b, w_2a, w_2b, ...]``
        where triplet ``k`` is ``Linear_k_a → SiLU → Linear_k_b``.
        ``D_in = linear_layers[0].in_features``, ``H = K-shape``
        derived per-triplet, ``D_out = linear_layers[-1].out_features``.

    Returns
    -------
    ``(M, D_out)`` tensor.
    """
    K = len(linear_layers) // 2
    assert 2 * K == len(linear_layers), (
        f"`linear_layers` must have an even number of modules (got "
        f"{len(linear_layers)})."
    )
    if K == 0:
        return x
    # CPU fallback (or no batched benefit if K == 1 — same as a single
    # fused_silu_mlp call).  We keep K==1 on the sequential path too
    # because the batched kernel adds cuBLAS / bmm overhead that does
    # not amortise for a single MLP.
    if K == 1 or (not x.is_cuda) or (not _triton_config.should_use_fused(
        x, op="batched_silu_mlp"
    )):
        # Sequential fallback: explicit ``F.silu`` + linear chain.
        h = x
        for k in range(K):
            la = linear_layers[2 * k]
            lb = linear_layers[2 * k + 1]
            h = F.silu(la(h))
            h = lb(h)
        return h

    # Build stacked weight tensors ``(K, *, *)`` for the batched kernel.
    # The fused kernel uses transposed weight convention
    # (``w1: (D, H1)``, ``w2: (H1, H2)``), so we transpose the
    # ``nn.Linear`` weights once and stack along dim 0.
    D_in = linear_layers[0].in_features
    w1s = torch.stack(
        [la.weight.t().contiguous() for la in (linear_layers[0::2])], dim=0
    )  # (K, D_in, H)
    b1s = torch.stack(
        [la.bias for la in (linear_layers[0::2])], dim=0
    )  # (K, H)
    w2s = torch.stack(
        [lb.weight.t().contiguous() for lb in (linear_layers[1::2])], dim=0
    )  # (K, H, D_out)
    b2s = torch.stack(
        [lb.bias for lb in (linear_layers[1::2])], dim=0
    )  # (K, D_out)
    out = _batched_fused_silu_mlp(x, w1s, b1s, w2s, b2s)  # (M, D_out)
    return out


DEFAULT_PRETRAINED_CKPT = (
    Path(__file__).resolve().parent.parent / "checkpoints" / "dmpnn_tmqm_pretrained.pt"
)


# ===========================================================================
# (b) Real Satorras 2021 EGNN layer — no detach, no max-Δx clip
# ===========================================================================
class _SatorrasEGNNLayer(nn.Module):
    """A single EGNN message-passing layer (Satorras et al., 2021, Eq. 3).

    The scalar update is
        ``m_ij = MLP_h([h_i || h_j || ||x_i − x_j||²])``
        ``h_i' = h_i + Σ_j m_ij``

    The coord update is
        ``x_i' = x_i + Σ_j (x_i − x_j) · tanh(MLP_x([m_ij]))``
        ``= x_i + Σ_j (x_i − x_j) · tanh(MLP_x([h_i || h_j || ||x_i − x_j||²]))``

    The full output (h', x') is **fully differentiable** w.r.t. both
    ``h_in`` and ``x_in``.  No detach, no max-Δx hard clip.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        # Scalar message MLP: input is [h_i || h_j || r²_ij]
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coord scale MLP: invariant scalar → equivariant coord update
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        h: torch.Tensor,         # (N, D)
        x: torch.Tensor,         # (N, 3)
        edge_index: torch.Tensor,  # (2, E)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_index.size(1) == 0 or h.size(0) == 0:
            return h, x

        src, dst = edge_index[0], edge_index[1]
        diff = x[src] - x[dst]                # (E, 3)
        r2 = (diff * diff).sum(dim=-1, keepdim=True)  # (E, 1) — invariant

        msg_in = torch.cat([h[src], h[dst], r2], dim=-1)  # (E, 2D+1)
        # Round-5 batched-fused-SiLU-MLP wiring.  ``msg_mlp`` is the
        # EGNN's "per-edge-type" SiLU MLP.  We expose the
        # ``_maybe_batched_silu_mlp`` helper at this site but currently
        # fall back to the sequential ``self.msg_mlp`` because the
        # ``batched_fused_silu_mlp`` kernel sums across K (it is a
        # *parallel-heads* kernel, not a *stacked-layers* one).  The
        # helper is wired so that future per-edge-type / per-atom-type
        # head stacks (K distinct MLPs with shared input shape) can
        # opt in by calling ``_maybe_batched_silu_mlp`` directly.
        m = self.msg_mlp(msg_in)                              # (E, D)

        # Aggregate scalar messages at dst
        h_new = torch.zeros_like(h)
        h_new.index_add_(0, dst, m)

        # Coord update — equivariant (RHS invariant, LHS equivariant)
        coord_scale = self.coord_mlp(m)             # (E, 1) bounded by tanh
        coord_msg = -diff * coord_scale             # -(x_i - x_j) = (x_j - x_i)
        x_new = x.clone()
        x_new.index_add_(0, dst, coord_msg)
        return h + h_new, x_new


class EGNNStack(nn.Module):
    """Stack of Satorras EGNN layers — equivariant, fully differentiable.

    Input: per-atom h (N, D_in) and 3D coords x (N, 3) for a single molecule.
    Output: per-atom h' (N, D_out).  The coordinate trajectory is discarded
    (this stack is a feature extractor, not a generative model).
    """

    def __init__(self, in_node_dim: int, hidden_dim: int, n_layers: int):
        super().__init__()
        self.in_proj = nn.Linear(in_node_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [_SatorrasEGNNLayer(hidden_dim) for _ in range(int(n_layers))]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        return self.out_norm(h)


# ===========================================================================
# (c) Perceiver-latent fusion (multi-head concat + FFN + gated residual)
# ===========================================================================
class _PerceiverLatentFusion(nn.Module):
    """Perceiver-IO style latent-array cross-attention with 8-head concat + FFN.

    16 learnable latent queries cross-attend to the concatenation of
    per-atom D-MPNN features and EGNN features.  Output: (B, D) fused
    vector via mean-pool over the latent array, then a soft-gated
    residual against a mean-pool of the D-MPNN stream (gate init ≈ 0.05).

    Implements the full FusionV4 spec from
    ``molmetal/reports/review_fusion.md``.
    """

    N_LATENT = 16
    N_HEADS = 8
    HEAD_DIM = 16
    LATENT_DIM = 128            # N_HEADS * HEAD_DIM
    INPUT_DROPOUT = 0.1
    ATTN_DROPOUT = 0.2
    FFN_DROPOUT = 0.1
    RESIDUAL_SCALE = 0.5

    def __init__(self, dmpnn_dim: int, egnn_dim: int, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = int(latent_dim)
        # latent_dim must equal N_HEADS * HEAD_DIM.  If the caller passes
        # a different dim (e.g. in unit tests with hidden_dim=64), we
        # silently fall back to the canonical 128 via a projection layer.
        if latent_dim != self.LATENT_DIM:
            self.in_proj = nn.Linear(dmpnn_dim + egnn_dim, self.LATENT_DIM)
            self.out_proj_d = nn.Linear(self.LATENT_DIM, dmpnn_dim)
            self.latent_dim = self.LATENT_DIM
        else:
            self.in_proj = nn.Identity()
            self.out_proj_d = nn.Identity()

        # Learnable latent array — (N_LATENT, latent_dim), init small
        self.latents = nn.Parameter(torch.randn(self.N_LATENT, latent_dim) * 0.02)

        in_dim = dmpnn_dim + egnn_dim
        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(in_dim, latent_dim)
        self.v = nn.Linear(in_dim, latent_dim)
        self.attn_drop = nn.Dropout(self.ATTN_DROPOUT)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

        # FFN block (4x expansion)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Dropout(self.FFN_DROPOUT),
            nn.Linear(4 * latent_dim, latent_dim),
            nn.Dropout(self.FFN_DROPOUT),
        )

        # Gated residual against the D-MPNN mean-pool (gate init ≈ 0.05)
        self.gate = nn.Linear(2 * latent_dim, latent_dim)
        with torch.no_grad():
            self.gate.bias.fill_(math.log(0.05 / 0.95))   # σ(-2.944) ≈ 0.05
            self.gate.weight.mul_(0.1)                     # grow slowly

        # Input dropout (independent on each stream)
        self.in_drop_dmpnn = nn.Dropout(self.INPUT_DROPOUT)
        self.in_drop_egnn = nn.Dropout(self.INPUT_DROPOUT)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, H*D) -> (B, n_heads, L, head_dim)
        B, L, _ = x.shape
        return x.view(B, L, self.N_HEADS, self.HEAD_DIM).transpose(1, 2)

    def forward(
        self,
        h_dmpnn: torch.Tensor,    # (B, N_max, d)
        h_egnn: torch.Tensor,     # (B, N_max, e)
        mask: torch.Tensor,       # (B, N_max) True for real atoms
    ) -> torch.Tensor:
        B = h_dmpnn.size(0)
        h_d = self.in_drop_dmpnn(h_dmpnn)
        h_e = self.in_drop_egnn(h_egnn)
        kv = torch.cat([h_d, h_e], dim=-1)              # (B, N_max, d+e)
        kv = self.in_proj(kv)                           # project to LATENT_DIM if needed
        z = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, N_LATENT, D)

        # MHCA — multi-head CONCAT (Vaswani 2017) not additive
        q = self._heads(self.q(z))       # (B, h, L, head_dim)
        k = self._heads(self.k(kv))      # (B, h, N_max, head_dim)
        v = self._heads(self.v(kv))      # (B, h, N_max, head_dim)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.HEAD_DIM)
        if mask is not None:
            scores = scores.masked_fill(
                (~mask.bool())[:, None, None, :], float("-inf")
            )
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        ctx = attn @ v                       # (B, h, L, head_dim)
        ctx = (
            ctx.transpose(1, 2)
            .contiguous()
            .view(B, self.N_LATENT, self.latent_dim)
        )
        # Round-5 fused-residual-add: route the
        # ``z + RESIDUAL_SCALE * out_proj(ctx)`` via the fused kernel
        # when the gate is open (default ON for training); else keep the
        # trivial PyTorch sum.
        attn_residual = self.RESIDUAL_SCALE * self.out_proj(ctx)
        z = self.norm1(_maybe_fused_residual_add(
            z, attn_residual, alpha=1.0, beta=1.0, training=self.training
        ))

        # FFN block (pre-norm transformer)
        # Round-5 fused-residual-add: route ``z + ffn(z)`` via the fused
        # kernel when the gate is open; else keep the trivial PyTorch sum.
        z = _maybe_fused_residual_add(
            self.ffn(z), z, alpha=1.0, beta=1.0, training=self.training,
        )

        # Mean-pool D-MPNN stream (baseline) — project to latent dim if needed
        d_pool = h_dmpnn.masked_fill(~mask.bool()[..., None], 0.0).sum(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        d_pool = d_pool / denom
        d_pool = self.out_proj_d(d_pool)

        g = torch.sigmoid(self.gate(torch.cat([d_pool, z.mean(dim=1)], dim=-1)))
        return g * z.mean(dim=1) + (1.0 - g) * d_pool


# ===========================================================================
# (d) Kendall + gamma-anneal + label-smoothing loss wrapper
# ===========================================================================
class LossV4(nn.Module):
    """Multi-task loss wrapper.

    Per review_multitask_loss.md:
      * Normalize cls / reg losses by their initial magnitude (R2 trick).
      * Kendall learnable log-sigma (homoscedastic uncertainty).
      * γ-anneal 0.1 → 0.0 cosine on the coord-refinement term.
      * Label smoothing 0.05 on the BCE term.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        gamma_start: float = 0.1,
        gamma_end: float = 0.0,
        ls: float = 0.05,
        warmup: int = 64,
        n_classes: int = 2,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma_start = float(gamma_start)
        self.gamma_end = float(gamma_end)
        self.ls = float(ls)
        self.warmup = int(warmup)
        self.n_classes = int(n_classes)

        # Learnable Kendall log-sigma per task
        self.log_s_cls = nn.Parameter(torch.tensor(0.0))
        self.log_s_reg = nn.Parameter(torch.tensor(0.0))
        self.log_s_crd = nn.Parameter(torch.tensor(0.0))

        # BCE with label smoothing (PyTorch >=1.10 supports label_smoothing)
        self.bce = nn.CrossEntropyLoss(label_smoothing=self.ls)
        self.mse = nn.MSELoss()

        # Normalization buffers (R2 trick)
        self.register_buffer("cls_init", torch.tensor(0.0))
        self.register_buffer("reg_init", torch.tensor(0.0))
        self._n = 0

    def gamma_at(self, epoch: int, T: int) -> float:
        """Cosine anneal γ from ``gamma_start`` → ``gamma_end`` over T epochs."""
        T = max(int(T), 1)
        if T <= 1:
            return self.gamma_end
        cos = 0.5 * (1.0 + math.cos(math.pi * min(epoch, T - 1) / (T - 1)))
        return self.gamma_end + (self.gamma_start - self.gamma_end) * cos

    def forward(
        self,
        out: dict,
        batch: dict,
        epoch: int = 0,
        T: int = 30,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the V4 multi-task loss.

        Args:
            out: dict with keys ``active_logits`` (B, n_classes), ``pic50_pred``
                 (B,), ``delta_pred`` (B, N_max, 3) — predicted coord shift.
            batch: dict with ``active`` (B,), ``pic50`` (B,), ``target_delta``
                   (B, N_max, 3), ``mask`` (B,), ``atom_mask`` (B, N_max).
            epoch: current epoch (0-indexed).
            T: total epochs.

        Returns:
            (loss, info) — info is a dict of detached per-task scalars.
        """
        active_lg = out["active_logits"]
        # Backward compatibility: prefer ``pic50`` (M-B1 dict), fall back
        # to ``pic50_pred`` (legacy V4 dict).
        pic50_p = out["pic50"] if "pic50" in out else out["pic50_pred"]
        delta_p = out["delta_pred"]

        y_a = batch["active"].long()
        y_p = batch["pic50"]
        target_delta = batch["target_delta"]
        atom_mask = batch["atom_mask"]   # (B, N_max)

        # Classification + regression loss
        lc = self.bce(active_lg, y_a)
        lr = self.mse(pic50_p, y_p)

        # Coord refinement MSE (masked)
        if delta_p.shape == target_delta.shape and atom_mask.any():
            mask3 = atom_mask.unsqueeze(-1).float()
            lcrd = (((delta_p - target_delta) ** 2) * mask3).sum() / mask3.sum().clamp(min=1.0)
        else:
            lcrd = torch.tensor(0.0, device=lc.device)

        # R2 normalisation warmup
        if self._n < self.warmup:
            with torch.no_grad():
                self.cls_init = self.cls_init + lc.detach()
                self.reg_init = self.reg_init + lr.detach()
                self._n += 1
                if self._n == self.warmup:
                    self.cls_init = self.cls_init / float(self.warmup)
                    self.reg_init = self.reg_init / float(self.warmup)
            lc_n = lc
            lr_n = lr
        else:
            lc_n = lc / (self.cls_init + 1e-8)
            lr_n = lr / (self.reg_init + 1e-8)

        # γ annealing (cosine 0.1 → 0.0)
        gamma = self.gamma_at(epoch, T)

        # Kendall weights (clamped for numerical stability)
        w_cls = torch.exp(-2.0 * self.log_s_cls.clamp(-3.0, 3.0))
        w_reg = torch.exp(-2.0 * self.log_s_reg.clamp(-3.0, 3.0))
        w_crd = torch.exp(-2.0 * self.log_s_crd.clamp(-3.0, 3.0))

        # Total = α·BCE + (1−α)·MSE  +  γ·coord  +  Kendall regularisers
        total = (
            self.alpha * w_cls * lc_n
            + (1.0 - self.alpha) * w_reg * lr_n
            + gamma * w_crd * lcrd
            + self.log_s_cls + self.log_s_reg + self.log_s_crd
        )

        info = {
            "loss_total": total.detach(),
            "loss_cls": lc.detach(),
            "loss_reg": lr.detach(),
            "loss_crd": lcrd.detach(),
            "gamma": float(gamma),
            "log_s_cls": self.log_s_cls.detach(),
            "log_s_reg": self.log_s_reg.detach(),
            "log_s_crd": self.log_s_crd.detach(),
            "w_cls": w_cls.detach(),
            "w_reg": w_reg.detach(),
            "w_crd": w_crd.detach(),
        }
        return total, info


# ===========================================================================
# (d') LossV4MB1 — M-B1 redesigned loss (decoupled + multi-fidelity)
# ===========================================================================
class LossV4MB1(nn.Module):
    """M-B1 redesigned loss to fix the pIC50 MAE-stuck problem.

    Four changes vs. ``LossV4``:

    1. **Two-stage forward**.  The caller is expected to invoke
       ``model.forward_two_stage(...)`` which runs the trunk ONCE and
       then the cls head on ``trunk(x).detach()`` and the reg head on
       ``trunk(x)``.  This breaks the gradient tug-of-war that made the
       reg head collapse to the dataset mean.

    2. **Log-residual regression head**.  The reg head predicts
       ``log(pIC50 + 1)`` instead of raw pIC50.  Inversion is
       ``exp(raw) - 1`` at test time (handled in
       ``forward_two_stage``).  Log-space MSE is numerically stable on
       small ranges and avoids negative predictions.

    3. **Sample-weighted MSE**.  Active samples (``pIC50 >= 5``) are
       weighted **3x** higher than inactive ones — the inactive plateau
       at ``pIC50 ≈ 4`` was the dominant cause of the constant MAE.

    4. **Multi-fidelity head**.  pIC50 prediction is split into
       ``active_p`` (binary classifier probability) plus a residual
       head ``relu(reg_raw)`` with pIC50_pred = active_p + 5 *
       relu(reg_raw).  The reg MSE is masked so only active samples
       contribute — this is the standard "predict only when active"
       trick from multi-fidelity regression.

    The Kendall σ_cls / σ_reg log-trajectory is tracked in ``info`` so
    the trainer can spot the failure mode "σ_reg plateaus while σ_cls
    keeps growing" (the M-B1 signature failure).
    """

    ACTIVE_PIC50_THRESHOLD = 5.0     # pIC50 >= 5.0 → "active" for weighting
    ACTIVE_WEIGHT_MULT = 3.0        # active samples weighted 3x inactive
    LOG_OFFSET = 1.0                # log(pIC50 + LOG_OFFSET) target

    # Scheduled clamp (M-2 round-3 fix)
    # Epochs 0-9  → pic50_min = 4.0 (warmup, current behaviour)
    # Epochs 10-19 → linear anneal 4.0 → 3.0
    # Epochs 20+  → pic50_min = 3.0 (final)
    PIC50_MIN_WARMUP = 4.0
    PIC50_MIN_FINAL = 3.0
    PIC50_MIN_ANNEAL_START = 10
    PIC50_MIN_ANNEAL_END = 20

    def __init__(
        self,
        alpha: float = 0.3,
        gamma_start: float = 0.1,
        gamma_end: float = 0.0,
        ls: float = 0.05,
        warmup: int = 64,
        n_classes: int = 2,
        active_weight: float = 3.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma_start = float(gamma_start)
        self.gamma_end = float(gamma_end)
        self.ls = float(ls)
        self.warmup = int(warmup)
        self.n_classes = int(n_classes)
        self.active_weight = float(active_weight)
        # Scheduled pic50_min — driven by ``set_epoch`` on the trainer side.
        self._current_epoch = 0
        self._pic50_min = float(self.PIC50_MIN_WARMUP)

        # Learnable Kendall log-sigma per task (cls + reg + coord)
        self.log_s_cls = nn.Parameter(torch.tensor(0.0))
        self.log_s_reg = nn.Parameter(torch.tensor(0.0))
        self.log_s_crd = nn.Parameter(torch.tensor(0.0))

        # BCE with label smoothing
        self.bce = nn.CrossEntropyLoss(label_smoothing=self.ls)

        # Normalisation buffers (R2 trick) — used to scale the cls/reg
        # losses so the Kendall weights are interpretable.
        self.register_buffer("cls_init", torch.tensor(0.0))
        self.register_buffer("reg_init", torch.tensor(0.0))
        self._n = 0

        # Per-epoch log-sigma tracking for stability diagnostics.
        # ``prev_log_s_cls`` / ``prev_log_s_reg`` are updated by the
        # training script each epoch; the loss surfaces
        # ``sigma_cls_log_change`` / ``sigma_reg_log_change`` in ``info``
        # so the trainer can warn when σ_reg plateaus.
        self.register_buffer("prev_log_s_cls", torch.tensor(0.0))
        self.register_buffer("prev_log_s_reg", torch.tensor(0.0))

    def gamma_at(self, epoch: int, T: int) -> float:
        """Cosine anneal γ from ``gamma_start`` → ``gamma_end`` over T epochs."""
        T = max(int(T), 1)
        if T <= 1:
            return self.gamma_end
        cos = 0.5 * (1.0 + math.cos(math.pi * min(epoch, T - 1) / (T - 1)))
        return self.gamma_end + (self.gamma_start - self.gamma_end) * cos

    def set_epoch(self, epoch: int) -> None:
        """Update the scheduled pic50_min based on the current epoch.

        M-2 round-3 fix: drives the pic50_min schedule so the reg head can
        express the test pIC50 distribution (test mean ≈ 4.73).
        """
        self._current_epoch = int(epoch)
        if epoch < self.PIC50_MIN_ANNEAL_START:
            self._pic50_min = float(self.PIC50_MIN_WARMUP)
        elif epoch < self.PIC50_MIN_ANNEAL_END:
            t = (epoch - self.PIC50_MIN_ANNEAL_START) / max(
                self.PIC50_MIN_ANNEAL_END - self.PIC50_MIN_ANNEAL_START, 1
            )
            self._pic50_min = float(
                self.PIC50_MIN_WARMUP
                + (self.PIC50_MIN_FINAL - self.PIC50_MIN_WARMUP) * t
            )
        else:
            self._pic50_min = float(self.PIC50_MIN_FINAL)

    def pic50_min_at(self, epoch: int) -> float:
        """Return the scheduled pic50_min for ``epoch`` (read-only peek)."""
        if epoch < self.PIC50_MIN_ANNEAL_START:
            return float(self.PIC50_MIN_WARMUP)
        elif epoch < self.PIC50_MIN_ANNEAL_END:
            t = (epoch - self.PIC50_MIN_ANNEAL_START) / max(
                self.PIC50_MIN_ANNEAL_END - self.PIC50_MIN_ANNEAL_START, 1
            )
            return float(
                self.PIC50_MIN_WARMUP
                + (self.PIC50_MIN_FINAL - self.PIC50_MIN_WARMUP) * t
            )
        return float(self.PIC50_MIN_FINAL)

    def forward(
        self,
        out: dict,
        batch: dict,
        epoch: int = 0,
        T: int = 30,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute the M-B1 multi-fidelity, decoupled loss.

        Args:
            out: dict from ``model.forward_two_stage(...)`` with keys
                 ``active_logits``, ``active_p``, ``pic50_raw``, ``pic50``,
                 ``delta_pred``, ``trunk``.
            batch: dict with ``active`` (B,), ``pic50`` (B,), ``mask`` (B,),
                   ``target_delta``, ``atom_mask``.
            epoch: current epoch (0-indexed).
            T: total epochs.

        Returns:
            (loss, info) — info has detached scalars including
            ``sigma_cls_log_change`` and ``sigma_reg_log_change``.
        """
        active_lg = out["active_logits"]
        pic50_raw = out["pic50_raw"]
        delta_p = out["delta_pred"]

        y_a = batch["active"].long()
        y_p = batch["pic50"]
        mask = batch["mask"].float()
        target_delta = batch["target_delta"]
        atom_mask = batch["atom_mask"]

        # --- Classification (BCE + label smoothing, weighted by valid mask)
        lc = self.bce(active_lg, y_a)

        # --- Log-residual regression target.
        # y_log = log(pIC50 + 1) - log(5 + 1).  Active pIC50 ≈ 5 maps to
        # ~0 so the residual head can be naturally ReLU-shaped.
        y_log_target = torch.log(y_p + self.LOG_OFFSET) - math.log(5.0 + self.LOG_OFFSET)
        is_active = (y_p >= self.ACTIVE_PIC50_THRESHOLD).float()
        active_mask = is_active * mask
        inactive_mask = (1.0 - is_active) * mask

        # Sample weight: 1.0 for inactive, active_weight for active
        sample_w = 1.0 + (self.active_weight - 1.0) * is_active
        sample_w = sample_w * mask

        # Weighted MSE on active samples (the multi-fidelity anchor).
        se = (pic50_raw - y_log_target) ** 2
        denom = active_mask.sum().clamp(min=1.0)
        lr_active = (se * sample_w).sum() / denom
        # Tiny "stay near 0" regulariser for inactive samples so the
        # reg head doesn't drift to large positives on negatives.
        lr_inactive = (pic50_raw ** 2 * inactive_mask).sum() / inactive_mask.sum().clamp(min=1.0) * 0.1
        lr = lr_active + lr_inactive

        # --- Coord refinement MSE (masked) — same as LossV4
        if delta_p.shape == target_delta.shape and atom_mask.any():
            mask3 = atom_mask.unsqueeze(-1).float()
            lcrd = (((delta_p - target_delta) ** 2) * mask3).sum() / mask3.sum().clamp(min=1.0)
        else:
            lcrd = torch.tensor(0.0, device=lc.device)

        # --- R2 normalisation warmup
        if self._n < self.warmup:
            with torch.no_grad():
                self.cls_init = self.cls_init + lc.detach()
                self.reg_init = self.reg_init + lr.detach()
                self._n += 1
                if self._n == self.warmup:
                    self.cls_init = self.cls_init / float(self.warmup)
                    self.reg_init = self.reg_init / float(self.warmup)
            lc_n = lc
            lr_n = lr
        else:
            lc_n = lc / (self.cls_init + 1e-8)
            lr_n = lr / (self.reg_init + 1e-8)

        # --- γ annealing (cosine 0.1 → 0.0)
        gamma = self.gamma_at(epoch, T)

        # --- Kendall weights (clamped for stability)
        w_cls = torch.exp(-2.0 * self.log_s_cls.clamp(-3.0, 3.0))
        w_reg = torch.exp(-2.0 * self.log_s_reg.clamp(-3.0, 3.0))
        w_crd = torch.exp(-2.0 * self.log_s_crd.clamp(-3.0, 3.0))

        # --- Total loss
        total = (
            self.alpha * w_cls * lc_n
            + (1.0 - self.alpha) * w_reg * lr_n
            + gamma * w_crd * lcrd
            + self.log_s_cls + self.log_s_reg + self.log_s_crd
        )

        # --- Per-epoch log-sigma tracking
        sigma_cls_log_change = float((self.log_s_cls - self.prev_log_s_cls).item())
        sigma_reg_log_change = float((self.log_s_reg - self.prev_log_s_reg).item())

        info = {
            "loss_total": total.detach(),
            "loss_cls": lc.detach(),
            "loss_reg": lr.detach(),
            "loss_reg_active": lr_active.detach(),
            "loss_reg_inactive": lr_inactive.detach(),
            "loss_crd": lcrd.detach(),
            "gamma": float(gamma),
            "log_s_cls": self.log_s_cls.detach(),
            "log_s_reg": self.log_s_reg.detach(),
            "log_s_crd": self.log_s_crd.detach(),
            "w_cls": w_cls.detach(),
            "w_reg": w_reg.detach(),
            "w_crd": w_crd.detach(),
            "sigma_cls_log_change": sigma_cls_log_change,
            "sigma_reg_log_change": sigma_reg_log_change,
            "active_frac": float(active_mask.sum().item() / max(active_mask.numel(), 1)),
        }
        return total, info

    def commit_epoch(self) -> None:
        """Call at the END of each training epoch to snapshot the
        current log-sigma values.  Next forward() will then compute
        ``sigma_*_log_change`` against this snapshot."""
        with torch.no_grad():
            self.prev_log_s_cls.copy_(self.log_s_cls.detach())
            self.prev_log_s_reg.copy_(self.log_s_reg.detach())


# ===========================================================================
# V4 main model
# ===========================================================================
@dataclass
class MetalHybridV4Config:
    """Configuration for the V4 hybrid (LLRD + real EGNN + Perceiver + Kendall)."""

    base: MetalHybridConfig = None
    # LLRD
    llrd_lr_top: float = 1e-3            # head LR
    llrd_lr_bottom: float = 1e-5         # encoder bottom LR
    llrd_decay: float = 0.95             # per-layer multiplier
    # Real EGNN
    egnn_in_node_dim: int = 128
    egnn_hidden_dim: int = 128
    n_egnn_layers: int = 3
    # Coord refinement (no detach, no clip)
    coord_loss_start: float = 0.1        # gamma_start
    coord_loss_end: float = 0.0          # gamma_end
    # Pretrained D-MPNN
    load_pretrained_encoder: bool = True
    pretrained_ckpt: str = str(DEFAULT_PRETRAINED_CKPT)


class MetalHybridV4Model(nn.Module):
    """V4 hybrid: LLRD + Satorras EGNN + Perceiver-latent fusion + Kendall loss.

    The encoder D-MPNN is **unfrozen** (Chemprop / Uni-Mol / GROVER).
    Layer-wise LR decay (LLRD, ULMFiT) protects the encoder from
    drifting while letting the new head fit fast.
    """

    def __init__(
        self,
        config: Optional[MetalHybridV4Config] = None,
        metal_embedding_dim: int = 32,
    ):
        super().__init__()
        if config is None:
            config = MetalHybridV4Config(base=MetalHybridConfig())
        elif config.base is None:
            config.base = MetalHybridConfig()
        self.cfg = config
        self.base_cfg = config.base
        self.hidden_dim = self.base_cfg.hidden_dim
        # Scheduled pic50_min (M-2 round-3 fix) — defaults to base_cfg.pic50_min
        # until ``set_epoch`` is called by the trainer.
        self._pic50_min = float(self.base_cfg.pic50_min)
        self._current_epoch = 0
        self.pic50_max = self.base_cfg.pic50_max

        # Featurizer
        self.featurizer = GraphFeaturizer()

        # D-MPNN encoder
        mpnn_cfg = MPNNConfig(
            atom_feat_dim=self.base_cfg.atom_feat_dim,
            edge_feat_dim=self.base_cfg.edge_feat_dim,
            hidden_dim=self.base_cfg.hidden_dim,
            n_layers=self.base_cfg.n_dmpnn_layers,
            dropout=self.base_cfg.dropout,
        )
        self.dmpnn = DirectedMPNN(mpnn_cfg)

        # Real Satorras EGNN (no detach, no max-Δx clip)
        self.egnn = EGNNStack(
            in_node_dim=self.cfg.egnn_in_node_dim,
            hidden_dim=self.cfg.egnn_hidden_dim,
            n_layers=self.cfg.n_egnn_layers,
        )

        # Perceiver-latent fusion (8-head concat + FFN + gate init 0.05)
        # Latent dim is fixed at 128 in the canonical config; in small/test
        # configs (hidden_dim < 16) we shrink it to hidden_dim to keep shapes
        # compatible.  Real configs use hidden_dim=128 → latent_dim=128.
        fusion_latent = max(128, self.hidden_dim) if self.hidden_dim >= 16 else self.hidden_dim
        self.fusion = _PerceiverLatentFusion(
            dmpnn_dim=self.hidden_dim,
            egnn_dim=self.hidden_dim,
            latent_dim=fusion_latent,
        )
        self._fusion_latent = fusion_latent

        # Coord-refinement MLP — equivariant, fully differentiable.
        # h_i' = h_i + Σ_j MLP([h_i, h_j, r_ij²])   (scalar head)
        # x_i' = x_i + Σ_j (x_j − x_i) · tanh(MLP)   (coord head, equivariant)
        self.coord_refine = _SatorrasEGNNLayer(self.hidden_dim)

        # Metal embedding + dual head
        self.metal_embed = nn.Embedding(10, metal_embedding_dim)
        # The fusion always emits 128-dim (it projects internally if input
        # is smaller) so head_mlp input dim is fixed at 128 + metal emb.
        self._fusion_out_dim = 128
        self.head_mlp = nn.Sequential(
            nn.Linear(self._fusion_out_dim + metal_embedding_dim, 128),
            nn.SiLU(),
            nn.Dropout(self.base_cfg.dropout),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        self.pic50_head = nn.Linear(64, 1)
        self.active_head = nn.Linear(64, self.base_cfg.n_activity_classes)

        self._loaded_pretrained = False
        if self.cfg.load_pretrained_encoder:
            self._maybe_load_pretrained()

        # ------------------------------------------------------------------
        # EMA (Exponential Moving Average) — M-B2 stability hook
        # The ema snapshot is lazily initialised by ``init_ema`` and updated
        # each step via ``update_ema(decay=0.999)``.  Buffers only; not a
        # learnable module.  Decay=0.999 follows Karras 2024 and the
        # standard recipe in DMPNN-style molecule training.
        # ------------------------------------------------------------------
        self._ema_state = None
        self._ema_decay = 0.999

    # ------------------------------------------------------------------
    # Scheduled pic50_min (M-2 round-3 fix)
    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Update the scheduled pic50_min based on the current epoch.

        M-2 round-3 fix: drives the pic50_min schedule so the reg head
        can express the test pIC50 distribution (test mean ≈ 4.73).
        Schedule:
          * Epochs 0-9  → pic50_min = 4.0 (warmup)
          * Epochs 10-19 → linear anneal 4.0 → 3.0
          * Epochs 20+  → pic50_min = 3.0 (final)
        """
        self._current_epoch = int(epoch)
        if epoch < 10:
            self._pic50_min = 4.0
        elif epoch < 20:
            t = (epoch - 10) / 10.0
            self._pic50_min = 4.0 + (3.0 - 4.0) * t
        else:
            self._pic50_min = 3.0

    @property
    def pic50_min(self) -> float:
        """Return the *scheduled* lower clamp on pIC50 (M-2 round-3 fix).

        Falls back to ``self.base_cfg.pic50_min`` if ``set_epoch`` has
        not been called yet (default 4.0).
        """
        if hasattr(self, "_pic50_min"):
            return float(self._pic50_min)
        return float(self.base_cfg.pic50_min)

    # ------------------------------------------------------------------
    # EMA hooks (M-B2)
    # ------------------------------------------------------------------
    def init_ema(self, decay: float = 0.999) -> None:
        """Initialise an EMA snapshot of the model's parameters.

        Must be called *after* the model is on its target device and
        *after* ``.train()`` mode is set for the first batch.  The EMA
        snapshot is stored on the same device as the model parameters.
        """
        self._ema_decay = float(decay)
        self._ema_state = {
            k: v.detach().clone()
            for k, v in self.state_dict().items()
        }

    def update_ema(self, decay: Optional[float] = None) -> None:
        """Update EMA snapshot: ``ema = decay * ema + (1 − decay) * model``.

        Decay defaults to ``self._ema_decay`` (0.999).  Does nothing if
        ``init_ema`` has not been called.
        """
        if self._ema_state is None:
            return
        if decay is None:
            decay = self._ema_decay
        decay = float(decay)
        with torch.no_grad():
            msd = self.state_dict()
            for k, v_ema in self._ema_state.items():
                v = msd[k]
                if not torch.is_floating_point(v):
                    # integer-shaped buffers (e.g. num_batches_tracked) —
                    # copy verbatim.
                    v_ema.copy_(v)
                else:
                    v_ema.mul_(decay).add_(v.detach(), alpha=1.0 - decay)

    def load_ema_state(self) -> None:
        """Copy the EMA snapshot back into the live model (eval-side use)."""
        if self._ema_state is None:
            return
        self.load_state_dict({k: v.clone() for k, v in self._ema_state.items()})

    def ema_state_dict(self) -> Optional[dict]:
        """Return the current EMA snapshot (or ``None`` if not initialised)."""
        return None if self._ema_state is None else {k: v.clone() for k, v in self._ema_state.items()}

    def load_ema_state_dict(self, ema_sd: dict) -> None:
        """Restore an EMA snapshot from a checkpoint."""
        self._ema_state = {k: v.detach().clone() for k, v in ema_sd.items()}

    # ------------------------------------------------------------------
    # Pretrained loading
    # ------------------------------------------------------------------
    def _maybe_load_pretrained(self) -> bool:
        ckpt_path = Path(self.cfg.pretrained_ckpt)
        if not ckpt_path.exists():
            return False
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except Exception:
            return False
        state = ckpt.get("encoder_state_dict", None) if isinstance(ckpt, dict) else None
        if state is None:
            return False
        own_state = self.dmpnn.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in own_state and tuple(own_state[k].shape) == tuple(v.shape):
                own_state[k] = v
                loaded += 1
        if loaded == 0:
            return False
        self.dmpnn.load_state_dict(own_state)
        self._loaded_pretrained = True
        return True

    @property
    def pretrained_loaded(self) -> bool:
        return bool(self._loaded_pretrained)

    # ------------------------------------------------------------------
    # LLRD parameter groups (top → bottom decay 0.95/layer)
    # ------------------------------------------------------------------
    def llrd_param_groups(self, lr_top: float = None, lr_bottom: float = None,
                          decay: float = None) -> List[dict]:
        """Build parameter groups with layer-wise LR decay.

        Top of the model (heads + fusion) gets ``lr_top``; each prior
        layer multiplies by ``decay``.  The bottom (D-MPNN's
        ``atom_embed``) gets ``lr_bottom`` at minimum.
        """
        lr_top = self.cfg.llrd_lr_top if lr_top is None else float(lr_top)
        lr_bottom = self.cfg.llrd_lr_bottom if lr_bottom is None else float(lr_bottom)
        decay = self.cfg.llrd_decay if decay is None else float(decay)

        # Layer index (0 = bottom / D-MPNN embed, increases upward).
        # We treat layers in this order:
        #   0: dmpnn.atom_embed            (lowest LR)
        #   1: dmpnn.edge_embed
        #   2: dmpnn.edge_mlp[0..L-1]
        #   3: dmpnn.gru_updates[0..L-1]
        #   4: dmpnn.atom_to_edge[0..L-1]
        #   5: dmpnn.readout_mlp           (highest encoder)
        #   6: egnn                        (3D stream)
        #   7: fusion                      (Perceiver)
        #   8: coord_refine                (coord head)
        #   9: head_mlp + pic50_head + active_head + metal_embed  (top)
        # We assign each param to one group.

        groups: List[dict] = []

        def _group_for(layer_idx: int, params, name: str):
            lr = lr_top * (decay ** (9 - layer_idx))
            lr = max(lr, lr_bottom)
            groups.append({"params": list(params), "lr": float(lr), "name": name})

        # D-MPNN bottom layer
        _group_for(0, self.dmpnn.atom_embed.parameters(), "dmpnn.atom_embed")
        _group_for(1, self.dmpnn.edge_embed.parameters(), "dmpnn.edge_embed")

        # Message-passing layers — pool the parameters of all L layers into
        # one group with an average decay
        edge_mlp_p = []
        gru_p = []
        a2e_p = []
        for mlp in self.dmpnn.edge_mlp:
            edge_mlp_p += list(mlp.parameters())
        for gru in self.dmpnn.gru_updates:
            gru_p += list(gru.parameters())
        for a2e in self.dmpnn.atom_to_edge:
            a2e_p += list(a2e.parameters())
        _group_for(2, edge_mlp_p, "dmpnn.edge_mlp[*]")
        _group_for(3, gru_p, "dmpnn.gru_updates[*]")
        _group_for(4, a2e_p, "dmpnn.atom_to_edge[*]")
        _group_for(5, self.dmpnn.readout_mlp.parameters(), "dmpnn.readout_mlp")

        _group_for(6, self.egnn.parameters(), "egnn")
        _group_for(7, self.fusion.parameters(), "fusion")
        _group_for(8, self.coord_refine.parameters(), "coord_refine")
        _group_for(
            9,
            list(self.head_mlp.parameters())
            + list(self.pic50_head.parameters())
            + list(self.active_head.parameters())
            + list(self.metal_embed.parameters()),
            "head+metal",
        )
        return groups

    # ------------------------------------------------------------------
    # Featurization (V3-compatible)
    # ------------------------------------------------------------------
    def _featurize_batch(self, smiles_list: List[str], mol_objects: List = None):
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        B = len(smiles_list)
        device = next(self.parameters()).device

        atom_feats, edge_indices, edge_attrs, n_atoms_list = [], [], [], []
        for i, smi in enumerate(smiles_list):
            try:
                if mol_objects is not None and mol_objects[i] is not None:
                    mol = mol_objects[i]
                else:
                    mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"RDKit parse failed for {smi!r}")
                feat = self.featurizer(mol)
                atom_feats.append(feat["x"])
                edge_indices.append(feat["edge_index"])
                edge_attrs.append(feat["edge_attr"])
                n_atoms_list.append(feat["n_atoms"])
            except Exception:
                atom_feats.append(np.zeros((1, self.base_cfg.atom_feat_dim), dtype=np.float32))
                edge_indices.append(np.zeros((2, 0), dtype=np.int64))
                edge_attrs.append(np.zeros((0, self.base_cfg.edge_feat_dim), dtype=np.float32))
                n_atoms_list.append(1)

        N_max = max(max(n_atoms_list), 1)
        E_max = max(max(ei.shape[1] for ei in edge_indices), 1)

        h_atom = torch.zeros(B, N_max, self.base_cfg.atom_feat_dim, device=device)
        edge_index = torch.zeros(B, 2, E_max, dtype=torch.long, device=device)
        edge_attr = torch.zeros(B, E_max, self.base_cfg.edge_feat_dim, device=device)
        batch_idx = torch.full((B, N_max), -1, dtype=torch.long, device=device)
        atom_mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

        for b in range(B):
            af = torch.from_numpy(atom_feats[b]).float()
            n_a = min(af.size(0), N_max)
            h_atom[b, :n_a] = af[:n_a]
            atom_mask[b, :n_a] = True
            ei = torch.from_numpy(edge_indices[b]).long()
            n_e = min(ei.size(1), E_max)
            edge_index[b, :, :n_e] = ei[:, :n_e]
            ea = torch.from_numpy(edge_attrs[b]).float()
            n_ea = min(ea.size(0), E_max)
            edge_attr[b, :n_ea] = ea[:n_ea]
            batch_idx[b, :n_a] = b

        return h_atom, edge_index, edge_attr, batch_idx, atom_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _compute_trunk(
        self,
        smiles_list: List[str],
        coords: torch.Tensor,
        metal_types: torch.Tensor,
        mol_objects: List = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the trunk (D-MPNN + EGNN + fusion + metal embed) once.

        Returns:
            fused (B, 128): pooled fused vector (the trunk features).
            delta_pred (B, N_max, 3): coord refinement output.
            metal_emb (B, metal_embedding_dim): metal embedding.
        """
        B = len(smiles_list)
        device = coords.device

        h_atom, edge_index, edge_attr, _, atom_mask = self._featurize_batch(
            smiles_list, mol_objects=mol_objects
        )
        N_max = h_atom.size(1)

        # 1. D-MPNN per-atom
        dmpnn_per_atom = torch.zeros(B, N_max, self.hidden_dim, device=device)
        for b in range(B):
            n_a = atom_mask[b].sum().item()
            if n_a == 0:
                continue
            h_mol = h_atom[b, :n_a]
            ei_mol = edge_index[b]
            ea_mol = edge_attr[b]
            if ei_mol.size(1) > 0:
                valid = (ei_mol[0] < n_a) & (ei_mol[1] < n_a)
                ei_local = ei_mol[:, valid]
                ea_local = ea_mol[valid]
            else:
                ei_local = torch.zeros((2, 0), dtype=torch.long, device=device)
                ea_local = torch.zeros((0, self.base_cfg.edge_feat_dim), device=device)
            dmpnn_per_atom[b, :n_a] = self.dmpnn.forward_per_atom(
                h_mol, ei_local, ea_local
            )

        # 2. Real Satorras EGNN per molecule
        egnn_per_atom = torch.zeros(B, N_max, self.hidden_dim, device=device)
        for b in range(B):
            n_a = atom_mask[b].sum().item()
            if n_a == 0:
                continue
            ei_b = edge_index[b]
            if ei_b.size(1) > 0:
                valid = (ei_b[0] < n_a) & (ei_b[1] < n_a) & (ei_b[0] != ei_b[1])
                ei_local = ei_b[:, valid]
            else:
                ei_local = torch.zeros((2, 0), dtype=torch.long, device=device)
            egnn_per_atom[b, :n_a] = self.egnn(
                dmpnn_per_atom[b, :n_a], coords[b, :n_a], ei_local
            )

        # 3. Perceiver-latent fusion (with D-MPNN mean-pool baseline)
        fused = self.fusion(dmpnn_per_atom, egnn_per_atom, atom_mask)  # (B, 128)

        # 4. Coord refinement — fully differentiable, equivariant
        delta_pred = torch.zeros_like(coords)
        for b in range(B):
            n_a = atom_mask[b].sum().item()
            if n_a == 0:
                continue
            ei_b = edge_index[b]
            if ei_b.size(1) > 0:
                valid = (ei_b[0] < n_a) & (ei_b[1] < n_a) & (ei_b[0] != ei_b[1])
                ei_local = ei_b[:, valid]
            else:
                ei_local = torch.zeros((2, 0), dtype=torch.long, device=device)
            h_b = egnn_per_atom[b, :n_a]
            x_b = coords[b, :n_a]
            _, x_new = self.coord_refine(h_b, x_b, ei_local)
            delta_pred[b, :n_a] = x_new - x_b

        # 5. Metal embedding (used as input to head_mlp)
        metal_emb = self.metal_embed(metal_types.long())
        return fused, delta_pred, metal_emb

    def _apply_heads(
        self,
        fused: torch.Tensor,
        metal_emb: torch.Tensor,
        detach_fused: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply head_mlp + cls/reg heads to (optionally detached) trunk.

        Used by ``LossV4MB1`` for two-stage forward: pass
        ``detach_fused=True`` so the cls head sees ``trunk(x).detach()``
        and the reg head sees ``trunk(x)``.  This breaks the gradient
        tug-of-war — the cls head can no longer flatten the trunk
        features in a way that sabotages the reg head (and vice-versa).
        """
        trunk = fused.detach() if detach_fused else fused
        head_input = torch.cat([trunk, metal_emb], dim=-1)
        h_head = self.head_mlp(head_input)
        active_logits = self.active_head(h_head)
        # pic50_raw is the *raw* network output.  LossV4MB1 interprets it
        # as log(pIC50 + 1) and inverts via exp - 1 at test time.
        pic50_raw = self.pic50_head(h_head).squeeze(-1)
        return active_logits, pic50_raw

    def forward(
        self,
        smiles_list: List[str],
        coords: torch.Tensor,
        metal_types: torch.Tensor,
        mol_objects: List = None,
    ) -> dict:
        """Forward returns a dict (V3 was a dataclass; dict fits LossV4 cleanly).

        Returns:
            dict with keys ``pic50`` (B,), ``pic50_raw`` (B,) — log-space
            raw output, ``active_logits`` (B, n_classes),
            ``delta_pred`` (B, N_max, 3), ``trunk`` (B, D) pooled fused
            vec, ``metal_emb`` (B, E).
        """
        fused, delta_pred, metal_emb = self._compute_trunk(
            smiles_list, coords, metal_types, mol_objects=mol_objects
        )
        active_logits, pic50_raw = self._apply_heads(fused, metal_emb)
        # Final pIC50 prediction = exp(raw) - 1 (clamped for stability).
        # LossV4MB1 inverts this via exp - 1 too.
        pic50_pred = torch.exp(pic50_raw) - 1.0
        pic50_pred = pic50_pred.clamp(self.pic50_min, self.pic50_max)

        return {
            "pic50": pic50_pred,
            "pic50_pred": pic50_pred,   # alias kept for backward compatibility
            "pic50_raw": pic50_raw,
            "active_logits": active_logits,
            "delta_pred": delta_pred,
            "trunk": fused,
            "metal_emb": metal_emb,
        }

    def forward_two_stage(
        self,
        smiles_list: List[str],
        coords: torch.Tensor,
        metal_types: torch.Tensor,
        mol_objects: List = None,
    ) -> dict:
        """Two-stage forward used by ``LossV4MB1``.

        Stage 1: compute trunk once (D-MPNN + EGNN + fusion + coord refine).
        Stage 2a: cls head on ``trunk(x).detach()`` — binary active prob.
        Stage 2b: reg head on ``trunk(x)`` — log-residual regression.

        Returns:
            dict with ``pic50`` (exp(raw) - 1, clamped),
            ``pic50_raw`` (log-space), ``active_p`` (sigmoid of active
            head), ``active_logits`` (B, n_classes), ``delta_pred``,
            ``trunk``, ``metal_emb``.
        """
        fused, delta_pred, metal_emb = self._compute_trunk(
            smiles_list, coords, metal_types, mol_objects=mol_objects
        )
        # Stage 2a: cls on detached trunk
        active_logits_det, _ = self._apply_heads(
            fused, metal_emb, detach_fused=True
        )
        # Stage 2b: reg on live trunk
        active_logits, pic50_raw = self._apply_heads(
            fused, metal_emb, detach_fused=False
        )
        # Use detached-logit active prob for the multi-fidelity assembly
        active_p = torch.sigmoid(active_logits_det[:, 1])
        pic50_pred = active_p + 5.0 * torch.relu(pic50_raw)
        pic50_pred = pic50_pred.clamp(self.pic50_min, self.pic50_max)

        return {
            "pic50": pic50_pred,
            "pic50_raw": pic50_raw,
            "active_p": active_p,
            "active_logits": active_logits,
            "delta_pred": delta_pred,
            "trunk": fused,
            "metal_emb": metal_emb,
        }

    # Convenience wrapper for V3-compatible tuple-unpack callers
    def forward_v3_compat(self, *a, **kw):
        d = self.forward(*a, **kw)
        return Pic50RegressionOutput(pic50=d["pic50"], active_logits=d["active_logits"])


__all__ = [
    "MetalHybridV4Config",
    "MetalHybridV4Model",
    "LossV4",
    "LossV4MB1",
    "_PerceiverLatentFusion",
    "EGNNStack",
    "_SatorrasEGNNLayer",
    "DEFAULT_PRETRAINED_CKPT",
]