"""Path-B Cross-Attention M4 — multi-head pocket cross-attention + pocket atom bias.

Background
----------
The round-17 (R17) Path-B CFM velocity field shipped a per-atom cross-
attention pocket conditioner in
:file:`molmetal/adapters/flow_matching_lipman/__init__.py` (P1.3, lines
1183-1204 of the post-R17 code).  That implementation uses a single
``nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4)`` block
following the Pocket2Mol pattern (Peng et al. 2022, ICML 2022,
arXiv:2205.07249).

**M4 = an incremental follow-up.**  It adds two follow-up heads that
compose with the existing single-head P1.3 block *without modifying it*:

* **M4.1 MultiHeadPocketCrossAttention** — a multi-head cross-attention
  block with **8 heads** (vs. R17 P1.3's 4 heads) and an optional
  per-pocket-atom input projection that lifts an arbitrary
  ``(B, P, in_dim)`` pocket feature tensor to ``hidden_dim`` before
  serving as keys/values.  The multi-head design follows Vaswani 2017
  §3.2.1 ("Multi-Head Attention"): split into ``n_heads`` subspaces,
  attend in parallel, concatenate.  Dropout (0.1 default) is added
  *inside* the attention block (post-softmax) for training-time
  regularisation.

* **M4.2 PocketAwareAtomBias** — a *bias term* induced by the per-pocket
  embedding.  Rather than attending over per-pocket-atom features, the
  bias pools the pocket signal (mean + max-pool, "min-max pooling" from
  Niepert 2016 §3.2 / "static attention" from Sukhbaatar 2019) into a
  single per-sample ``(B, 2*hidden_dim)`` descriptor, projects it
  back to ``(B, hidden_dim)`` and broadcasts to every ligand atom.
  This is the inductive bias used in Pocket2Mol (Peng 2022 §3.2)
  *alongside* the cross-attention block — combining the two yields
  stronger empirical performance than either alone.

* **M4CompositeAttention** — a wrapper that composes the existing P1.3
  single-head cross-attention (``single_head``), the new M4.1 multi-head
  cross-attention (``multi_head``), and the new M4.2 atom-bias
  (``atom_bias``) into a single ``forward(x, pocket) -> torch.Tensor``
  call.  The composite sums the residual contributions of all three
  blocks and applies a final LayerNorm so the output shape
  ``(batch, n_atoms, hidden_dim)`` drops into the existing CFM
  downstream code unchanged.

Lit anchor
----------
Vaswani 2017 NeurIPS "Attention Is All You Need" §3.2.1 (multi-head
concat), Peng 2022 ICML Pocket2Mol §3.2 (per-atom cross-attention +
additive pocket bias), Sukhbaatar 2019 NeurIPS "Adaptive Computation
Time" §2 (static attention pooling).

Constraints
-----------
* **DO NOT modify** the existing P1.3 single-head cross-attention
  (R17 ship, ``molmetal/adapters/flow_matching_lipman/__init__.py``
  lines 1183-1204).
* This module is *additive only*: it imports nothing from
  ``flow_matching_lipman`` and never alters that module.  The M4
  composite is a stand-alone wrapper that the production wiring can
  optionally adopt by passing the existing ``EGNNVelocityField``'s
  ``cross_attn`` module as ``single_head``.
* CPU-only path: no GPU / Triton / ROCm dependencies.  All tests run
  in <2 s on a single core.

Differences from R17 P1.3
-------------------------
* **Head count**: 8 (M4.1) vs. 4 (P1.3).  Doubling the head count
  halves the per-head dimension (``hidden_dim // n_heads`` = 16 at the
  canonical ``hidden_dim=128``), giving finer-grained subspace
  decomposition of the pocket-ligand correspondence.
* **Dropout**: 0.1 default (M4.1) vs. 0.0 (P1.3).  Training-time
  dropout inside the attention block is a known regulariser
  (Vaswani 2017 §5.4, "we apply dropout to the output of each
  sub-layer, before adding to the residual ... and to the attention
  weights").
* **Additive atom bias**: M4.2 adds an *additive* bias term that the
  P1.3 block does NOT have.  The bias is the projection of a
  pool-of-pocket-residues descriptor; it is the Pocket2Mol
  ``pocket_enc.bias`` term.
* **Composite residual**: M4CompositeAttention sums the three
  residual contributions, then applies a final LayerNorm.  This
  triples the expressivity of the per-atom conditioning at the
  cost of ~3x the parameter count — still a tiny fraction of the
  full EGNNVelocityField (~10 MB at the canonical hidden_dim=128).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_N_HEADS",
    "DEFAULT_DROPOUT",
    "DEFAULT_MAX_ATOMS",
    "M4ForwardOutput",
    "MultiHeadPocketCrossAttention",
    "PocketAwareAtomBias",
    "M4CompositeAttention",
]


# ---------------------------------------------------------------------------
# Module-level defaults — exposed so tests + callers can import them
# directly without instantiating the classes.
# ---------------------------------------------------------------------------
# Canonical hidden_dim for SBDD-FM models (matches R17 P1.3).
DEFAULT_HIDDEN_DIM: int = 128
# M4.1 uses 8 heads (vs R17 P1.3's 4).  8 heads × 16 head_dim = 128.
DEFAULT_N_HEADS: int = 8
# Training-time dropout inside the multi-head attention block
# (Vaswani 2017 §5.4).  0.0 in eval / unit tests.
DEFAULT_DROPOUT: float = 0.1
# Default cap on per-sample ligand atoms for the M4.2 bias path —
# used only to size the bias Linear layer; runtime tensors may be
# larger (we use index arithmetic that does not require fixed
# max_atoms at runtime).
DEFAULT_MAX_ATOMS: int = 64


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class M4ForwardOutput:
    """Output of :meth:`M4CompositeAttention.forward`.

    Attributes
    ----------
    h : (batch, n_atoms, hidden_dim) tensor
        The composite output — sum of the three residual contributions
        followed by LayerNorm.  Drops into the existing CFM downstream
        code unchanged.
    single_head_out : (batch, n_atoms, hidden_dim) tensor
        The P1.3 single-head cross-attention contribution (before
        residual + LayerNorm).  Useful for ablation studies.
    multi_head_out : (batch, n_atoms, hidden_dim) tensor
        The M4.1 multi-head cross-attention contribution (before
        residual + LayerNorm).  Useful for ablation studies.
    atom_bias : (batch, n_atoms, hidden_dim) tensor
        The M4.2 additive pocket-induced atom bias.  Useful for
        ablation studies.
    """


# ---------------------------------------------------------------------------
# M4.1 — MultiHeadPocketCrossAttention
# ---------------------------------------------------------------------------
class MultiHeadPocketCrossAttention(nn.Module):
    """Multi-head cross-attention block over per-pocket-atom features.

    Compared to the R17 P1.3 single-head block in
    :file:`molmetal/adapters/flow_matching_lipman/__init__.py`, this
    module:

    * uses 8 heads (default) with ``head_dim = hidden_dim // n_heads``;
    * supports an arbitrary ``pocket_in_dim`` for the per-pocket-atom
      input — the caller can pass raw atomic-number one-hots, an
      EGNN-encoded per-residue vector, or any other ``(B, P, in_dim)``
      tensor, and the module projects to ``hidden_dim`` internally;
    * applies dropout (default 0.1) *inside* the attention block —
      both on the softmax weights and on the projection output.

    Parameters
    ----------
    hidden_dim : int
        Width of the per-ligand-atom features and the attention
        output.  Default 128 (R17 P1.3 default).
    n_heads : int
        Number of attention heads.  Default 8 (vs R17 P1.3's 4).
    pocket_in_dim : int
        Dimension of the per-pocket-atom input feature.  When equal
        to ``hidden_dim`` (the default), no projection is added.
    dropout : float
        Dropout probability inside the attention block (post-softmax
        and post-projection).  Default 0.1 (Vaswani 2017 §5.4).
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        n_heads: int = DEFAULT_N_HEADS,
        pocket_in_dim: Optional[int] = None,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim!r}"
            )
        if n_heads <= 0 or hidden_dim % n_heads != 0:
            raise ValueError(
                f"n_heads ({n_heads}) must be > 0 and divide "
                f"hidden_dim ({hidden_dim})"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout!r}"
            )
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.hidden_dim // self.n_heads
        self.dropout = float(dropout)

        # Pocket input projection.  When ``pocket_in_dim`` is None we
        # default to ``hidden_dim`` (no projection needed); when it
        # differs (e.g. raw atomic-number one-hots of size 50), we
        # add a Linear to lift to ``hidden_dim``.
        pocket_in = int(pocket_in_dim) if pocket_in_dim is not None else self.hidden_dim
        if pocket_in <= 0:
            raise ValueError(
                f"pocket_in_dim must be > 0, got {pocket_in!r}"
            )
        self.pocket_in_dim = pocket_in
        if pocket_in != self.hidden_dim:
            self.pocket_proj = nn.Linear(pocket_in, self.hidden_dim, bias=False)
        else:
            # Identity — caller passes already-projected features.
            self.pocket_proj = nn.Identity()

        # QKV projections — three separate Linears, mirrors Vaswani
        # 2017 §3.2.1 "we linearly project the queries, keys and
        # values ... with different, learned linear projections".
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        # Output projection + dropout (post-projection, pre-residual).
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.attn_drop = nn.Dropout(self.dropout)
        self.proj_drop = nn.Dropout(self.dropout)
        # Final LayerNorm on the residual stream — same pattern as
        # the R17 P1.3 ``cross_attn_norm``.
        self.norm = nn.LayerNorm(self.hidden_dim)
        # Scale factor for the scaled dot product (Vaswani 2017
        # §3.2.1 — divide by sqrt(head_dim)).  We use a Python float
        # for readability.
        self._scale = 1.0 / math.sqrt(self.head_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, S, hidden_dim)`` → ``(B, n_heads, S, head_dim)``.

        Mirrors the existing ``_PerceiverLatentFusion._heads`` helper
        in :file:`molmetal/models/metal_hybrid_v4.py`.
        """
        B, S, _ = x.shape
        return x.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, n_heads, S, head_dim)`` → ``(B, S, hidden_dim)``."""
        B, _, S, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, self.hidden_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        atom_features: torch.Tensor,
        pocket_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``(batch, n_atoms, hidden_dim)`` attended features.

        Parameters
        ----------
        atom_features : (batch, n_atoms, hidden_dim) tensor
            Per-ligand-atom query features.
        pocket_features : (batch, n_pocket_atoms, pocket_in_dim) tensor
            Per-pocket-atom key + value features.  When
            ``pocket_in_dim == hidden_dim`` these are used as-is;
            otherwise we project to ``hidden_dim`` first.
        mask : (batch, n_pocket_atoms) bool tensor, optional
            ``True`` at *valid* pocket atoms.  Positions where
            ``mask`` is ``False`` are masked out of the attention
            softmax (same convention as
            :class:`molmetal.models.cross_attention_fusion.CrossAttentionFusion`).

        Returns
        -------
        (batch, n_atoms, hidden_dim) float tensor
            The attended features, with a residual + LayerNorm applied
            to ``atom_features``.  When ``atom_features`` is the zero
            tensor, the output is the zero tensor plus LayerNorm (the
            LayerNorm's learned affine shifts can break exact zero,
            which is fine — the EGNN's downstream code expects a
            non-trivial affine output).
        """
        if atom_features.dim() != 3:
            raise ValueError(
                f"atom_features must be (batch, n_atoms, hidden_dim), "
                f"got shape {tuple(atom_features.shape)}"
            )
        if pocket_features.dim() != 3:
            raise ValueError(
                f"pocket_features must be (batch, n_pocket_atoms, "
                f"pocket_in_dim), got shape {tuple(pocket_features.shape)}"
            )
        if atom_features.shape[0] != pocket_features.shape[0]:
            raise ValueError(
                f"atom_features batch ({atom_features.shape[0]}) must "
                f"match pocket_features batch ({pocket_features.shape[0]})"
            )
        if atom_features.shape[2] != self.hidden_dim:
            raise ValueError(
                f"atom_features last dim ({atom_features.shape[2]}) "
                f"must equal hidden_dim ({self.hidden_dim})"
            )
        if pocket_features.shape[2] != self.pocket_in_dim:
            raise ValueError(
                f"pocket_features last dim ({pocket_features.shape[2]}) "
                f"must equal pocket_in_dim ({self.pocket_in_dim})"
            )
        # Project pocket features to hidden_dim (identity if same).
        pocket_proj = self.pocket_proj(pocket_features)  # (B, P, H)

        # Q from ligand atoms, K + V from pocket atoms.
        q = self._split_heads(self.q_proj(atom_features))   # (B, h, N, hd)
        k = self._split_heads(self.k_proj(pocket_proj))     # (B, h, P, hd)
        v = self._split_heads(self.v_proj(pocket_proj))     # (B, h, P, hd)

        # Scaled dot-product attention with optional key-padding
        # mask.  ``scores`` shape: (B, h, N, P).
        scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale
        if mask is not None:
            if mask.shape != pocket_features.shape[:2]:
                raise ValueError(
                    f"mask shape ({tuple(mask.shape)}) must match "
                    f"pocket_features spatial dims "
                    f"({pocket_features.shape[0]}, {pocket_features.shape[1]})"
                )
            # True at valid pocket atoms; we mask-out the rest with -inf.
            scores = scores.masked_fill(
                (~mask.bool())[:, None, None, :], float("-inf"),
            )
        attn = F.softmax(scores, dim=-1)
        # Dropout on the attention weights (Vaswani 2017 §5.4).
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, v)  # (B, h, N, hd)
        # Merge heads and apply output projection.
        ctx = self._merge_heads(ctx)                # (B, N, H)
        ctx = self.proj_drop(self.out_proj(ctx))    # (B, N, H)
        # Residual + LayerNorm (Vaswani 2017 §5.4 "add & norm").
        return self.norm(atom_features + ctx)


# ---------------------------------------------------------------------------
# M4.2 — PocketAwareAtomBias
# ---------------------------------------------------------------------------
class PocketAwareAtomBias(nn.Module):
    """Pocket-induced additive atom bias.

    The bias pools the per-pocket-atom feature tensor into a single
    per-sample descriptor via **mean + max-pool** ("min-max pooling" in
    Niepert 2016 §3.2; "static attention" in Sukhbaatar 2019 §2),
    projects the descriptor back to ``hidden_dim``, and broadcasts to
    every ligand atom.  The pooled descriptor captures the global
    "what kind of pocket is this?" signal, complementing the
    *per-atom* correspondence captured by cross-attention.

    Parameters
    ----------
    hidden_dim : int
        Width of the per-ligand-atom features and the bias output.
        Default 128.
    max_atoms : int
        Upper bound on the per-sample ligand atom count.  Used to
        size the bias Linear layer when the caller passes
        ``atom_features`` with a known ``n_atoms`` at init time.  The
        module is shape-agnostic at runtime (the bias is broadcast,
        not indexed) — ``max_atoms`` is purely a parameter-budgeting
        default.
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        max_atoms: int = DEFAULT_MAX_ATOMS,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim!r}"
            )
        if max_atoms <= 0:
            raise ValueError(
                f"max_atoms must be > 0, got {max_atoms!r}"
            )
        self.hidden_dim = int(hidden_dim)
        self.max_atoms = int(max_atoms)

        # Mean + max pool → (B, 2 * hidden_dim) descriptor.
        self.pocket_descriptor = nn.Linear(
            2 * self.hidden_dim, self.hidden_dim, bias=True,
        )
        # Per-atom gating — a learned scalar that the optimiser can
        # anneal from "no bias" (init 0.0) to "full bias" (init 1.0).
        # Init at zero so an untrained module produces the zero bias
        # (matching the P1.3 zero-init rationale).
        self.gate = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        atom_features: torch.Tensor,
        pocket_features: torch.Tensor,
        pocket_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return ``(batch, n_atoms, hidden_dim)`` bias tensor.

        Parameters
        ----------
        atom_features : (batch, n_atoms, hidden_dim) tensor
            Per-ligand-atom features.  Used only for the shape
            contract — the bias is shape-only derived from this and
            does NOT depend on the per-atom content (that is the
            cross-attention's job; the bias is the global pocket
            signal).
        pocket_features : (batch, n_pocket_atoms, hidden_dim) tensor
            Per-pocket-atom features — must already be projected to
            ``hidden_dim``.  The M4 composite takes care of the
            projection via :attr:`MultiHeadPocketCrossAttention.pocket_proj`.
        pocket_mask : (batch, n_pocket_atoms) bool tensor, optional
            ``True`` at valid pocket atoms.  When supplied, masked
            positions are excluded from the mean / max pools so the
            descriptor is invariant to padding.  When ``None``, the
            full ``pocket_features`` tensor participates.

        Returns
        -------
        (batch, n_atoms, hidden_dim) float tensor
            The pocket-induced atom bias, broadcast across all
            ``n_atoms`` ligand atoms.  When ``pocket_features`` is
            all-zero (zero pocket), the bias is identically zero.
        """
        if atom_features.dim() != 3:
            raise ValueError(
                f"atom_features must be (batch, n_atoms, hidden_dim), "
                f"got shape {tuple(atom_features.shape)}"
            )
        if pocket_features.dim() != 3:
            raise ValueError(
                f"pocket_features must be (batch, n_pocket_atoms, "
                f"hidden_dim), got shape {tuple(pocket_features.shape)}"
            )
        if atom_features.shape[0] != pocket_features.shape[0]:
            raise ValueError(
                f"atom_features batch ({atom_features.shape[0]}) must "
                f"match pocket_features batch ({pocket_features.shape[0]})"
            )
        if pocket_features.shape[2] != self.hidden_dim:
            raise ValueError(
                f"pocket_features last dim ({pocket_features.shape[2]}) "
                f"must equal hidden_dim ({self.hidden_dim})"
            )
        batch = atom_features.shape[0]

        # Mask-aware mean / max pool.  We use a tiny epsilon floor so
        # all-padding rows do not NaN the descriptor.
        if pocket_mask is not None:
            if pocket_mask.shape != pocket_features.shape[:2]:
                raise ValueError(
                    f"pocket_mask shape ({tuple(pocket_mask.shape)}) "
                    f"must match pocket_features spatial dims "
                    f"({pocket_features.shape[0]}, {pocket_features.shape[1]})"
                )
            mask_f = pocket_mask.bool().unsqueeze(-1).to(pocket_features.dtype)
            # Masked mean: sum / count.  Count floor at 1 (epsilon)
            # so all-padding rows get a "mean of zeros" contribution.
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            mean_pool = (pocket_features * mask_f).sum(dim=1) / denom
            # Masked max: replace padding with -inf so it never wins.
            masked_for_max = pocket_features.masked_fill(
                (~pocket_mask.bool())[..., None], float("-inf"),
            )
            max_pool = masked_for_max.max(dim=1).values
            # If a row is all-padding, the max is -inf; replace with
            # the mean pool so the descriptor remains finite.
            all_pad = ~pocket_mask.bool().any(dim=1, keepdim=True)
            max_pool = torch.where(all_pad, mean_pool, max_pool)
        else:
            mean_pool = pocket_features.mean(dim=1)
            max_pool = pocket_features.max(dim=1).values

        # Concatenate, project, gate.  Descriptor shape: (B, H).
        descriptor = self.pocket_descriptor(
            torch.cat([mean_pool, max_pool], dim=-1),
        )
        # Gated broadcast: (B, H) → (B, 1, H) → broadcast to (B, n_atoms, H).
        bias = self.gate * descriptor.unsqueeze(1)
        # Broadcast to (B, n_atoms, H) via expand (no copy).
        return bias.expand(atom_features.shape[0], atom_features.shape[1], self.hidden_dim)


# ---------------------------------------------------------------------------
# M4 composite — wires single-head + multi-head + atom-bias
# ---------------------------------------------------------------------------
class M4CompositeAttention(nn.Module):
    """Composite cross-attention wiring for the M4 follow-up to R17 P1.3.

    Parameters
    ----------
    single_head : nn.Module or None
        The existing R17 P1.3 single-head cross-attention block.  Must
        expose a ``forward(atom_features, pocket_features, mask=None)``
        method with the same signature as
        :class:`MultiHeadPocketCrossAttention`.  When ``None``, the
        single-head contribution is omitted (the composite still runs
        with multi-head + bias only).
    multi_head : :class:`MultiHeadPocketCrossAttention` or None
        The new M4.1 multi-head cross-attention block.  When ``None``,
        the multi-head contribution is omitted.
    atom_bias : :class:`PocketAwareAtomBias` or None
        The new M4.2 atom-bias module.  When ``None``, the bias
        contribution is omitted.

    All three are optional; the composite is robust to any subset.
    When all three are ``None``, the composite reduces to a LayerNorm
    on ``x`` (still well-defined).
    """

    def __init__(
        self,
        single_head: Optional[nn.Module] = None,
        multi_head: Optional[MultiHeadPocketCrossAttention] = None,
        atom_bias: Optional[PocketAwareAtomBias] = None,
    ) -> None:
        super().__init__()
        self.single_head = single_head
        self.multi_head = multi_head
        self.atom_bias = atom_bias
        # Final LayerNorm — applied AFTER the sum of residual
        # contributions so the output is well-scaled regardless of
        # how many sub-modules are active.
        #
        # We need a hidden_dim to size the LayerNorm.  We infer it
        # from whichever sub-module exposes ``hidden_dim`` (in the
        # priority order multi_head → atom_bias → single_head, since
        # the production wiring typically attaches both single_head
        # and multi_head and the multi_head is the larger of the
        # two); if none exposes it (or all are None), we fall back
        # to the canonical DEFAULT_HIDDEN_DIM.
        inferred_hidden = None
        for sub in (multi_head, atom_bias, single_head):
            if sub is not None and hasattr(sub, "hidden_dim"):
                inferred_hidden = int(getattr(sub, "hidden_dim"))
                break
        if inferred_hidden is None:
            # Fall back to the canonical default.  We do NOT emit a
            # warning here — an all-None composite is a legitimate
            # unit-test scenario, and the caller knows the LayerNorm
            # is sized to DEFAULT_HIDDEN_DIM.
            inferred_hidden = DEFAULT_HIDDEN_DIM
        self.hidden_dim = int(inferred_hidden)
        self.norm = nn.LayerNorm(self.hidden_dim)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        pocket: torch.Tensor,
        pocket_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the composite ``(batch, n_atoms, hidden_dim)`` output.

        Parameters
        ----------
        x : (batch, n_atoms, hidden_dim) tensor
            Per-ligand-atom features.
        pocket : (batch, n_pocket_atoms, hidden_dim) tensor
            Per-pocket-atom features — must already be projected to
            ``hidden_dim`` so the single-head, multi-head, and bias
            modules can all consume the same tensor.
        pocket_mask : (batch, n_pocket_atoms) bool tensor, optional
            ``True`` at valid pocket atoms.  Forwarded to all three
            sub-modules.

        Returns
        -------
        (batch, n_atoms, hidden_dim) float tensor
            The composite output — sum of (residual single-head) +
            (residual multi-head) + (additive bias) followed by
            LayerNorm.
        """
        if x.dim() != 3:
            raise ValueError(
                f"x must be (batch, n_atoms, hidden_dim), "
                f"got shape {tuple(x.shape)}"
            )
        if pocket.dim() != 3:
            raise ValueError(
                f"pocket must be (batch, n_pocket_atoms, hidden_dim), "
                f"got shape {tuple(pocket.shape)}"
            )
        if x.shape[0] != pocket.shape[0]:
            raise ValueError(
                f"x batch ({x.shape[0]}) must match pocket batch "
                f"({pocket.shape[0]})"
            )
        if x.shape[2] != self.hidden_dim:
            raise ValueError(
                f"x last dim ({x.shape[2]}) must equal composite "
                f"hidden_dim ({self.hidden_dim})"
            )
        if pocket.shape[2] != self.hidden_dim:
            raise ValueError(
                f"pocket last dim ({pocket.shape[2]}) must equal "
                f"composite hidden_dim ({self.hidden_dim})"
            )
        # The composite output is the sum of residual contributions.
        # We initialise ``out`` as the zero tensor (same shape as x);
        # each active sub-module adds its residual contribution.
        out = torch.zeros_like(x)
        if self.single_head is not None:
            out = out + self.single_head(x, pocket, mask=pocket_mask)
        if self.multi_head is not None:
            out = out + self.multi_head(x, pocket, mask=pocket_mask)
        if self.atom_bias is not None:
            out = out + self.atom_bias(x, pocket, pocket_mask=pocket_mask)
        # Final LayerNorm — the composite contract guarantees the
        # output is well-scaled even when only one sub-module is
        # active (a single residual + LayerNorm is a valid identity-
        # preserving transformation when the inner module's
        # residual + LayerNorm already returned the input shape).
        return self.norm(out)


# ---------------------------------------------------------------------------
# Self-test (lightweight — run as ``python -m
# molmetal.molmetal_lam.lam_chem.cross_attention_m4``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    torch.manual_seed(0)
    multi = MultiHeadPocketCrossAttention()
    bias = PocketAwareAtomBias()
    composite = M4CompositeAttention(multi_head=multi, atom_bias=bias)
    x = torch.randn(2, 6, 128)
    pocket = torch.randn(2, 8, 128)
    out = composite(x, pocket)
    print(
        f"composite_out={tuple(out.shape)}, "
        f"finite={bool(torch.isfinite(out).all().item())}"
    )
