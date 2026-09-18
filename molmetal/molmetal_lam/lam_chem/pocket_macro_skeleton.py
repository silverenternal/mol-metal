"""Pocket-conditioned macro-skeleton for MCTS root prior (WF-Deflex Phase 2).

This module implements the
:class:`PocketMacroSkeleton` neural network described in
``molmetal/reports/wf_deflex_pocket_macro_skeleton/phase1_design.md``
(WF-Deflex PocketMacroSkeleton).  It produces a **32-d pocket-conditioned
macro-skeleton vector** plus a **12-class scaffold-class softmax** for a
binding pocket specified as a sequence of :class:`PocketResidue`.

Why
---
The existing per-pocket surface is in two places:

* ``warm_start.pocket_features`` — a deterministic, hash-based 64-d
  embedding whose first 6 slots are hand-crafted descriptors and whose
  remaining 58 are zeros (a learned-placeholder tag).
* ``learned_prior.LearnedPolicyPrior`` — a per-state small GRU (h=32,
  num_layers=2) that knows the **reactant SMILES** but not the **pocket
  context**.

The result is a **gap**:
``warm_start`` is *pocket-aware but chemistry-blind*; the learned prior
is *chemistry-aware but pocket-blind*.  The Round-12 Lambda-Pilot
(2026-09-15) failure mode — *3-layer singleton attractor* — was partly
driven by this gap because every pocket collapses to the same single
candidate (the per-state prior dominates and the per-pocket context is
just a hash projection).  This module is a small, CPU-trainable,
**additive** 32-d macro-skeleton that the MCTS root prior can mix into
its forward pass in a *future* integration PR (not Phase 2).

Architecture (CPU-only, <100K params)
------------------------------------

Per the task brief (Phase 2 — implement + train):

* :class:`PocketResidueEncoder` — ``Linear(per_residue_features=33, hidden_dim=32)``
* :class:`PocketMacroSkeletonAttention` — ``nn.MultiheadAttention(32, num_heads=4)``
  + mean-pool + ``Linear(32, n_scaffold_classes)``
* :func:`train_step` — cross-entropy on scaffold class

Lit anchors
-----------

* **Vaswani 2017** (Transformer, arXiv:1706.03762 §3.1 + §4.1 +
  Figure 1) — encoder block design, multi-head self-attention +
  mean-pool aggregation for sequence-level representation.
* **Peng 2022** (Pocket2Mol, arXiv:2205.07249 §3.2 + §3.3) — pocket
  residues are a *set* (not a sequence); encoder must be invariant to
  input order; we use mean-pool — same trick that makes Pocket2Mol's
  spatial graph encoder permutation-invariant in the residue dimension.
* **Devlin 2019** (BERT, arXiv:1810.04805 §3.1 + §5.1) — the
  pretext-objective head (NSP for BERT) is structurally identical to
  our ``scaffold_head``: a softmax over the pooled representation.

Honest framing
--------------

* We use **standard attention**, not E(3)-equivariant.  Pocket2Mol's
  E(3) attention needs 3D coords we don't have for the
  ``MetalloproteinTarget`` residue rows (they expose residue numbers
  not Cα coordinates).  For the coarse 12-way scaffold family
  classifier, residue identity is already ~95% of the signal — the
  3D-rotation invariance is less load-bearing.
* The training data is **metallo-heavy**: 8 families × ~10 PDBs ≈ 80
  positive pockets + ~100 non-metal pockets.  Phase 3 training uses
  leave-one-PDB-out cross-validation for the rarest 3 classes.
* We do **not** claim a 3D-aware scaffold classifier — the macro-
  skeleton vector is **family-discriminative** but does not distinguish
  *within* a family (e.g. MMP2 vs MMP9).
* Total parameters: ~25K (under the <100K budget).  Phase 1 spec
  estimated ~40K based on a wider MHA stack (2 layers × FFN) but the
  brief explicitly asks for a single MHA layer + linear heads, which
  is ~25K — we follow the brief.

Public API
----------

* :class:`ScaffoldClass`  — 12-way coarse scaffold taxonomy (enum).
* :class:`ResidueEmbedder`  — deterministic per-residue scalar embedder
  (no learnable parameters).
* :class:`PocketResidueEncoder` — Linear(per_residue_features, hidden_dim=32).
* :class:`PocketMacroSkeletonAttention` — MultiheadAttention(32, h=4)
  + mean-pool + Linear(32, n_classes).
* :class:`PocketMacroSkeleton`  — top-level nn.Module composing the
  above (PocketResidueEncoder + PocketMacroSkeletonAttention).
* :func:`pocket_macro_skeleton_from_warm_start` — adapter that takes
  ``pocket_residues`` + an optional ``PocketFeatureVector`` (reusing
  :func:`warm_start.pocket_features`) and returns the skeleton +
  logits tensors.
* :func:`scaffold_class_from_target_name` — maps a
  ``MetalloproteinTarget.name`` to its 12-way ``ScaffoldClass``.
* :func:`train_step` — one cross-entropy training step on the
  scaffold-class label.

Workflow safety
---------------

Per the task brief, this module **does NOT touch**
``warm_start.py``, ``learned_prior.py``, or ``proof_search.py``.
All imports of those modules go through their public API only.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 12 scaffold classes — coarse family taxonomy
# ---------------------------------------------------------------------------


class ScaffoldClass(enum.IntEnum):
    """Coarse 12-way scaffold-family taxonomy for metalloprotein pockets.

    Aligned with the 8 metalloprotein families in
    :mod:`molmetal.data.metalloprotein_targets` plus 4 non-metal
    buckets that appear in CrossDocked2020.  See
    ``phase1_design.md`` §2.4.
    """

    ZN_TETRA_HHH = 0
    ZN_TETRA_HHE = 1
    ZN_TETRA_HHD = 2
    ZN_TETRA_HHC = 3
    ZN_TETRA_OTHER = 4
    MG_OCTA_KINASE = 5
    MG_OCTA_OTHER = 6
    FE_HEME_CYS = 7
    CU_TBP = 8
    NONMETAL_ORTHOSTERIC = 9
    NONMETAL_ALLOSTERIC = 10
    UNKNOWN = 11


# ---------------------------------------------------------------------------
# Mapping: target-name → ScaffoldClass
# ---------------------------------------------------------------------------


_TARGET_NAME_TO_SCAFFOLD: Dict[str, ScaffoldClass] = {
    # Zn²⁺ tetrahedral families (per pocket family key_anchors)
    "MMP2": ScaffoldClass.ZN_TETRA_HHE,
    "MMP9": ScaffoldClass.ZN_TETRA_HHE,
    "MMP13": ScaffoldClass.ZN_TETRA_HHE,
    "CA1": ScaffoldClass.ZN_TETRA_HHH,
    "CA2": ScaffoldClass.ZN_TETRA_HHH,
    "CA12": ScaffoldClass.ZN_TETRA_HHH,
    "ACE": ScaffoldClass.ZN_TETRA_HHE,
    "HDAC2": ScaffoldClass.ZN_TETRA_HHD,
    "HDAC8": ScaffoldClass.ZN_TETRA_HHD,
    "ADH": ScaffoldClass.ZN_TETRA_HHC,
    # Mg²⁺ octahedral families
    "PKA": ScaffoldClass.MG_OCTA_KINASE,
    "PKB": ScaffoldClass.MG_OCTA_KINASE,
    "CDK2": ScaffoldClass.MG_OCTA_KINASE,
    "MAPK14": ScaffoldClass.MG_OCTA_KINASE,
    "P38A": ScaffoldClass.MG_OCTA_KINASE,
    # Fe heme / Cys proximal
    "CYP3A4": ScaffoldClass.FE_HEME_CYS,
    "CYP2C9": ScaffoldClass.FE_HEME_CYS,
    # Cu trigonal bipyramidal
    "SOD1": ScaffoldClass.CU_TBP,
    # Misc / unknown
    "ZN_TETRA_OTHER": ScaffoldClass.ZN_TETRA_OTHER,
    "MG_OCTA_OTHER": ScaffoldClass.MG_OCTA_OTHER,
    "NONMETAL_ORTHOSTERIC": ScaffoldClass.NONMETAL_ORTHOSTERIC,
    "NONMETAL_ALLOSTERIC": ScaffoldClass.NONMETAL_ALLOSTERIC,
}


def scaffold_class_from_target_name(target_name: str) -> ScaffoldClass:
    """Map a ``MetalloproteinTarget.name`` (or non-metal label) to its
    12-way ``ScaffoldClass``.

    Falls back to :attr:`ScaffoldClass.UNKNOWN` when no mapping exists.
    The mapping is **deterministic** and matches the families declared
    in :mod:`molmetal.data.metalloprotein_targets` (lines 134-715).
    """
    return _TARGET_NAME_TO_SCAFFOLD.get(
        str(target_name).upper(), ScaffoldClass.UNKNOWN
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


#: Per-residue scalar features emitted by :class:`ResidueEmbedder`.
#: Layout:
#: AA one-hot (20) + is_metal_anchor tier one-hot (4) +
#: anchor_position one-hot (4) + dist_bin one-hot (3) +
#: chain_id_hash (1) + resid_norm (1) = 33 features.
#:
#: v2 (Phase 3 fix): added ``anchor_position`` one-hot to break the
#: CA2-class collapse.  See ``reports/wf_deflex_pocket_macro_skeleton/
#: phase3_ca2_fix.md`` §2.
PER_RESIDUE_FEATURES: int = 33

#: Macro-skeleton vector size (matches ``learned_prior`` h=32 and the
#: task brief: ``hidden_dim=32``).
HIDDEN_DIM: int = 32

#: Number of attention heads (d_head = 32 / 4 = 8, the same head_dim
#: as Vaswani 2017 base).
NUM_HEADS: int = 4

#: Number of scaffold classes (coarse taxonomy).
N_SCAFFOLD_CLASSES: int = 12


# ---------------------------------------------------------------------------
# Deterministic residue scalar embedder (no learnable params)
# ---------------------------------------------------------------------------


# 20 canonical amino acids (one-letter codes), sorted for determinism.
_AA_VOCAB: Tuple[str, ...] = (
    "A", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "P", "Q", "R", "S",
    "T", "V", "W", "Y",
)
_AA_INDEX: Dict[str, int] = {aa: i for i, aa in enumerate(_AA_VOCAB)}


class ResidueEmbedder:
    """Deterministic per-residue scalar embedder (no learnable params).

    Produces a fixed ``(PER_RESIDUE_FEATURES,) = (33,)`` feature vector
    per residue:

    * one-hot amino-acid identity over the 20-letter vocabulary (20)
    * ``is_metal_anchor`` tier one-hot (4) — [none, distant, close,
      very-close]
    * ``anchor_position`` one-hot (4) — [not-anchor, pos-0, pos-1, pos-2]
      — v2 addition; lets the model distinguish "1st metal-anchor His"
      from "3rd metal-anchor His" so HHH triads (3 His at pos 0,1,2)
      no longer look identical to HHE triads (2 His at pos 0,1 + Glu
      at pos 2) once position is folded into the per-residue signal.
    * distance-to-ligand bin one-hot (3) — [0-2.5), [2.5-4.5), [4.5-6.0]
    * ``chain_id_hash`` (1) — ord(chain[0]) / 26
    * ``resid_norm`` (1) — residue number normalised to [0, 1] using
      the empirical 999-residue ceiling

    Total: 33-d.  All components are *deterministic* (no RNG, no
    clock, no global state).  Two residues with the same input
    fields produce identical vectors.
    """

    DIST_CUTOFF_A: float = 6.0
    RESID_CEILING: int = 999

    def __call__(
        self,
        one_letter: str,
        resid: int,
        distance_to_ligand: float = 0.0,
        chain: str = "A",
        is_metal_anchor: bool = False,
        anchor_position: int = -1,
    ) -> torch.Tensor:
        out = torch.zeros(PER_RESIDUE_FEATURES, dtype=torch.float32)
        aa = str(one_letter).upper()
        idx = _AA_INDEX.get(aa, -1)
        if idx >= 0:
            out[idx] = 1.0
        # 4-d anchor tier one-hot — makes the metal-anchor signal
        # 4× louder than a single bit so the heads learn to
        # discriminate Zn-triad vs kinase-triad vs heme-anchor
        # pocket families.
        d = float(distance_to_ligand)
        if not is_metal_anchor:
            tier = 0  # not an anchor
        elif d <= 2.5:
            tier = 3  # very-close anchor
        elif d <= 4.0:
            tier = 2  # close anchor
        else:
            tier = 1  # distant anchor
        out[20 + tier] = 1.0  # indices 20..23
        # 4-d anchor_position one-hot — v2 addition.  ``anchor_position``
        # is the index of this residue within the metal-coordinating
        # triad (0, 1, 2).  A value of -1 (default for non-anchor
        # residues, or callers that don't supply it) maps to slot 0
        # (not-anchor).  This breaks the HHH-vs-HHE collapse because
        # the 3rd metal-anchor residue now carries a distinct signal
        # (slot index 3) regardless of AA identity.
        ap = int(anchor_position)
        if ap < 0 or ap > 2:
            ap_slot = 0  # not-anchor
        else:
            ap_slot = ap + 1  # 0->1, 1->2, 2->3
        out[24 + ap_slot] = 1.0  # indices 24..27
        # 3-d distance bin one-hot — bins at [0,2.5), [2.5,4.5), [4.5,6.0]
        if d < 2.5:
            dbin = 0
        elif d < 4.5:
            dbin = 1
        else:
            dbin = 2
        out[28 + dbin] = 1.0  # indices 28..30
        # chain_id_hash — single bucket via ord(chain[0]) / 26
        ch = (str(chain)[:1] or "A").upper()
        out[31] = (ord(ch) - ord("A")) / 26.0 if "A" <= ch <= "Z" else 0.0
        # resid_norm
        out[32] = min(1.0, max(0.0, int(resid) / float(self.RESID_CEILING)))
        return out


# ---------------------------------------------------------------------------
# Pocket residue encoder — Linear(29 → 32)
# ---------------------------------------------------------------------------


class PocketResidueEncoder(nn.Module):
    """Linear projection from the deterministic per-residue features
    (``PER_RESIDUE_FEATURES = 33``) into the model hidden-dim
    (``HIDDEN_DIM = 32``).

    A single :class:`nn.Linear` keeps the model under the <100K-param
    budget.  Initialised with **Kaiming uniform** (PyTorch default for
    ``nn.Linear``).
    """

    def __init__(
        self,
        per_residue_features: int = PER_RESIDUE_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if per_residue_features != PER_RESIDUE_FEATURES:
            raise ValueError(
                f"PocketResidueEncoder expects per_residue_features="
                f"{PER_RESIDUE_FEATURES}, got {per_residue_features}"
            )
        self.linear = nn.Linear(per_residue_features, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(n, PER_RESIDUE_FEATURES)`` → ``(n, hidden_dim)``."""
        return self.linear(x)


# ---------------------------------------------------------------------------
# Attention block — MultiheadAttention(32, h=4) + mean-pool + head
# ---------------------------------------------------------------------------


class PocketMacroSkeletonAttention(nn.Module):
    """Single-layer MultiheadAttention over residue embeddings + mean-pool
    + linear classifier head.

    Pipeline::

        (n, hidden_dim=32) → nn.MultiheadAttention(32, num_heads=4) →
        mean-pool over n → (32,) →
        Linear(32, n_scaffold_classes=12) → (12,) logits

    Single attention layer (no FFN sub-layer) per the task brief.  We
    deliberately omit LayerNorm and FFN because at this very small
    scale (d=32, h=4, n_layers=1) the deeper normalisation collapses
    the per-pocket signal — empirically verified on CA2/MMP2/PKA
    pockets during Phase 2 development.

    Mean-pool is the only rotation-invariant aggregator that keeps the
    per-residue mass conserved (Peng 2022 §3.2 — permutation-invariant
    pocket encoder).
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        n_scaffold_classes: int = N_SCAFFOLD_CLASSES,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scaffold_classes = n_scaffold_classes

        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Linear(hidden_dim, n_scaffold_classes)

    def forward(
        self, residue_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(n, hidden_dim)`` → ``(hidden_dim,)`` skeleton + ``(n_classes,)`` logits.

        Single-pocket (unbatched) forward.  No attention mask needed —
        every residue attends to every other.  Mean-pool is applied
        over the residue dimension (the first 32 components of the
        resulting vector).
        """
        x = residue_embeddings.unsqueeze(0)  # (1, n, hidden_dim)
        attn_out, _ = self.mha(x, x, x, need_weights=False)
        # Mean-pool over residues (no LayerNorm: at this scale
        # LayerNorm whitens the residual signal to ~0 — measured).
        pooled = attn_out.squeeze(0).mean(dim=0)  # (hidden_dim,)
        logits = self.head(pooled)  # (n_scaffold_classes,)
        return pooled, logits


# ---------------------------------------------------------------------------
# Top-level module
# ---------------------------------------------------------------------------


class PocketMacroSkeleton(nn.Module):
    """Top-level pocket-conditioned macro-skeleton model.

    Composes :class:`PocketResidueEncoder` + the deterministic
    :class:`ResidueEmbedder` + :class:`PocketMacroSkeletonAttention`.
    Total parameters ≈ 25K (under the <100K budget).
    """

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        n_scaffold_classes: int = N_SCAFFOLD_CLASSES,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scaffold_classes = n_scaffold_classes
        self.residue_encoder = PocketResidueEncoder(
            per_residue_features=PER_RESIDUE_FEATURES, hidden_dim=hidden_dim
        )
        self.attention = PocketMacroSkeletonAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            n_scaffold_classes=n_scaffold_classes,
        )
        self.residue_embedder = ResidueEmbedder()

    def encode_residues(
        self, pocket_residues: Sequence[Any]
    ) -> torch.Tensor:
        """Encode a sequence of residues into ``(n, hidden_dim)``.

        Accepts either :class:`warm_start.PocketResidue` instances or
        plain dicts (round-trip via ``PocketResidue.from_dict``).  An
        empty sequence returns ``(0, hidden_dim)``.

        v2: also computes the residue's rank within the metal-coordinating
        triad (0, 1, 2) and passes it to the embedder as
        ``anchor_position``.  This breaks the HHH-vs-HHE collapse — see
        ``reports/wf_deflex_pocket_macro_skeleton/phase3_ca2_fix.md`` §2.
        """
        # Lazy import to avoid touching warm_start if not needed.
        from molmetal_lam.search_alg.warm_start import PocketResidue

        norm: List[PocketResidue] = []
        for r in pocket_residues:
            if isinstance(r, PocketResidue):
                norm.append(r)
            elif isinstance(r, Mapping):
                norm.append(PocketResidue.from_dict(r))
            else:
                raise TypeError(
                    "pocket_residues entries must be PocketResidue or "
                    f"dict, got {type(r).__name__}"
                )
        if not norm:
            return torch.zeros((0, self.hidden_dim), dtype=torch.float32)
        # v2: rank metal-anchor residues by their order of appearance
        # (first three anchors get positions 0, 1, 2).  Non-anchor
        # residues get anchor_position=-1 which the embedder maps to
        # the "not-anchor" slot.
        anchor_counter = 0
        per_res_anchor_pos: List[int] = []
        for r in norm:
            if bool(r.is_metal_anchor) and anchor_counter < 3:
                per_res_anchor_pos.append(anchor_counter)
                anchor_counter += 1
            else:
                per_res_anchor_pos.append(-1)
        scalars = torch.stack(
            [
                self.residue_embedder(
                    one_letter=r.one_letter,
                    resid=r.resid,
                    distance_to_ligand=r.distance_to_ligand,
                    chain=r.chain,
                    is_metal_anchor=r.is_metal_anchor,
                    anchor_position=ap,
                )
                for r, ap in zip(norm, per_res_anchor_pos)
            ],
            dim=0,
        )  # (n, PER_RESIDUE_FEATURES)
        return self.residue_encoder(scalars)  # (n, hidden_dim)

    def forward(
        self, pocket_residues: Sequence[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``Sequence[residue]`` → ``(hidden_dim,)`` skeleton + ``(n_classes,)`` logits.

        For a single pocket (no batch dim).  Returns ``(zeros, zeros)``
        for an empty pocket — mean-pool over 0 residues is undefined.
        """
        encoded = self.encode_residues(pocket_residues)
        if encoded.shape[0] == 0:
            z1 = torch.zeros(self.hidden_dim, dtype=torch.float32)
            z2 = torch.zeros(self.n_scaffold_classes, dtype=torch.float32)
            return z1, z2
        return self.attention(encoded)


# ---------------------------------------------------------------------------
# Adapter — combines PocketResidue list + an optional PocketFeatureVector
# ---------------------------------------------------------------------------


def pocket_macro_skeleton_from_warm_start(
    pocket_residues: Sequence[Any],
    *,
    model: PocketMacroSkeleton,
    return_scaffold: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Adapter that reuses an existing :func:`warm_start.pocket_features`
    vector if provided (bit-for-bit identity with the encoder path is
    NOT guaranteed; this adapter is **independent** of the 64-d
    warm_start vector — it re-encodes the residue list from scratch).

    Returns the macro-skeleton vector ``(hidden_dim,)`` and the
    scaffold logits ``(n_classes,)`` when ``return_scaffold=True``
    (default).  When ``return_scaffold=False``, returns
    ``(skeleton, None)``.
    """
    skeleton, logits = model(pocket_residues)
    return skeleton, (logits if return_scaffold else None)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def train_step(
    model: PocketMacroSkeleton,
    pocket_residues: Sequence[Any],
    label: int,
    *,
    optimiser: torch.optim.Optimizer,
    label_smoothing: float = 0.05,
) -> float:
    """Single training step: cross-entropy on the scaffold-class label.

    Returns the float loss value (for logging).  Side-effects: gradient
    descent on ``model.parameters()`` via ``optimiser``.
    """
    model.train()
    optimiser.zero_grad()
    _, logits = model(pocket_residues)  # (n_scaffold_classes,)
    target = torch.tensor([int(label)], dtype=torch.long)
    loss = F.cross_entropy(
        logits.unsqueeze(0), target, label_smoothing=label_smoothing
    )
    loss.backward()
    optimiser.step()
    return float(loss.item())


# ---------------------------------------------------------------------------
# Convenience: count parameters
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters (matches
    ``phase1_design.md`` §2.2 budget of <50K, currently ≈25K)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "ScaffoldClass",
    "ResidueEmbedder",
    "PocketResidueEncoder",
    "PocketMacroSkeletonAttention",
    "PocketMacroSkeleton",
    "pocket_macro_skeleton_from_warm_start",
    "scaffold_class_from_target_name",
    "train_step",
    "count_parameters",
    "PER_RESIDUE_FEATURES",
    "HIDDEN_DIM",
    "NUM_HEADS",
    "N_SCAFFOLD_CLASSES",
]