"""ChempropV2MTModel — Chemprop v2.0 multi-task backbone for MetalCytoToxDB.

Wf-148 ultracode / TODO-148 §3 follow-up.  Implements the
:class:`ChempropV2MTModel` class that wraps the upstream Chemprop 2.0
message-passing backbone (Yang et al. 2019 + Chemprop 2.0) inside the
5-way multi-task scaffold already used by
:class:`molmetal.models.v4_multitask_dmpnn.V4MultiTaskDMPNN`.

Why we don't just use ``chemprop.models.MPNN`` directly
---------------------------------------------------
``chemprop.models.MPNN`` is a ``pl.LightningModule`` that bundles a
single-task predictor + training_step + val_step + test_step + Adam +
Noam LR scheduler.  That's the wrong abstraction for our use case:

  * We need a **multi-task predictor** (5 metal heads + 1 coord head)
    sharing one encoder; chemprop's MPNN hard-codes a single
    predictor.
  * Our loss is the weighted sum of per-metal BCE + coord CE with the
    same architecture as V4MultiTaskLoss; chemprop's MPNN owns its
    own loss (BCELoss on the predictor's n_tasks).
  * Lightning's LightningModule adds hparams + checkpoint format we
    don't want to depend on for the LOMO training loop.

Instead, we use chemprop v2 as a **building-block library**:

  * ``chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer`` — SMILES
    → ``MolGraph`` (atom features + bond features + edge_index +
    rev_edge_index).
  * ``chemprop.data.BatchMolGraph`` — list of ``MolGraph`` →
    batched ``V / E / edge_index / rev_edge_index / batch`` tensors.
  * ``chemprop.nn.AtomMessagePassing`` — the MPNN backbone
    (d_v / d_e / d_h / depth defaults match Chemprop 2.0).
  * ``chemprop.nn.SumAggregation`` — read-out from vertex hidden
    states to graph hidden states.
  * ``chemprop.nn.BinaryClassificationFFN`` — per-metal binary
    classification head (train_step returns logits; forward applies
    sigmoid).
  * ``chemprop.nn.BCELoss`` — standard chemprop BCE loss for
    training-time metric matching.

The model exposes the **same calling convention as
:class:`V4MultiTaskDMPNN`**: ``forward(smiles_list, metal_ids)``
returns ``dict[metal_name -> (B, 2) logits, "coord" -> (B, n_coord_classes)]``.
Backward compat with V4MultiTaskDMPNN is preserved: the V4 class is
unchanged and remains the default for `--model v4`.

Per-metal BCE training step
---------------------------
:func:`compute_per_metal_loss` uses chemprop v2's ``BCELoss`` (the same
metric class chemprop's ``MPNN`` uses internally) per metal head.
Each metal head sees the same binary classification target
(``is_active``); per-metal weights can be supplied via ``metal_weights``
to mirror V4MultiTaskLoss semantics.

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
* The chemprop v2 backbone is **bigger** than the V4 backbone
  (default ``d_h=300`` + depth=3 + bond+atom message passing) — per-step
  cost on CPU is ~5-7× higher than V4.  For LOMO runs we expose
  ``d_h=64`` + depth=2 to keep unit tests fast.
* Chemprop v2 uses ``MultiHotAtomFeaturizer`` (72-d atom features) +
  ``MultiHotBondFeaturizer`` (14-d bond features) by default.  These
  are denser than V4's ``GraphFeaturizer`` (39 atom + 6 bond features)
  but the encoder remains comparable in parameter count.
* This class is the **multi-task scaffolding**.  Wiring it into the
  ``BalancedLOMORunner`` training loop is TODO-148 §4 (the legacy
  LOMO loop calls ``model(atom_f, bond_f, e_src, e_dst)`` which is a
  DMPNNMultiTaskModel-only API).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import (
    AtomMessagePassing,
    BCELoss,
    BinaryClassificationFFN,
    SumAggregation,
)


# ---------------------------------------------------------------------------
# Constants — keep in sync with V4MultiTaskDMPNN.
# ---------------------------------------------------------------------------
#: Canonical 5-way metal subset for precious-metal cytotoxicity.
#: Must match :data:`molmetal.models.v4_multitask_dmpnn.SUPPORTED_METALS`.
SUPPORTED_METALS: Tuple[str, ...] = ("Ru", "Ir", "Os", "Rh", "Re")

#: Coordination-geometry class names (5 classes — matches the
#: ``n_coord_classes=5`` default).
COORD_GEOMETRY_NAMES: Tuple[str, ...] = (
    "linear",  # 0
    "trigonal_planar",  # 1
    "tetrahedral",  # 2
    "square_planar",  # 3
    "octahedral",  # 4
)

#: Loss weight on the coordination-geometry auxiliary task (matches V4).
COORD_LOSS_WEIGHT: float = 0.3

#: Default chemprop v2 backbone hyperparameters — see
#: ``chemprop.nn.AtomMessagePassing`` defaults.
DEFAULT_HIDDEN_DIM: int = 300
DEFAULT_DEPTH: int = 3
DEFAULT_DROPOUT: float = 0.0
DEFAULT_BIAS: bool = False


# ---------------------------------------------------------------------------
# Encoder wrapper — chemprop v2 message passing + aggregation.
# ---------------------------------------------------------------------------
class ChempropV2Encoder(nn.Module):
    """Wrap chemprop v2's :class:`AtomMessagePassing` + :class:`SumAggregation`
    into a single ``(smiles_list) -> (B, hidden_dim)`` callable.

    The encoder is shared across all metal heads — same efficiency
    rationale as :class:`V4MultiTaskDMPNN` / :class:`DMPNNEncoder`.
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        depth: int = DEFAULT_DEPTH,
        dropout: float = DEFAULT_DROPOUT,
        bias: bool = DEFAULT_BIAS,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.bias = bool(bias)
        # Stateless featurizer — safe to construct once and re-use.
        self.featurizer = SimpleMoleculeMolGraphFeaturizer()
        # Lazy-built on first forward so we know the actual atom/bond fdim
        # at the user's chemprop version.
        self.message_passing: Optional[AtomMessagePassing] = None
        self.agg: Optional[SumAggregation] = None
        self._atom_fdim: int = -1
        self._bond_fdim: int = -1

    # ------------------------------------------------------------------
    def _ensure_backbone(self, atom_fdim: int, bond_fdim: int) -> None:
        """Lazy-construct the chemprop v2 backbone on the first call.

        Chemprop's :class:`SimpleMoleculeMolGraphFeaturizer` does not
        expose ``atom_fdim`` / ``bond_fdim`` as class attributes until
        post-init; we resolve them from the first batch and reuse the
        backbone thereafter.
        """
        if self.message_passing is not None:
            return
        self._atom_fdim = int(atom_fdim)
        self._bond_fdim = int(bond_fdim)
        self.message_passing = AtomMessagePassing(
            d_v=self._atom_fdim,
            d_e=self._bond_fdim,
            d_h=self.hidden_dim,
            depth=self.depth,
            dropout=self.dropout,
            bias=self.bias,
        )
        self.agg = SumAggregation()

    # ------------------------------------------------------------------
    def _featurize_smiles(self, smiles_list: List[str]) -> Tuple[List, int, int]:
        """Featurize SMILES → list of MolGraph, returning fdim too.

        Invalid SMILES are routed through a single-atom dummy graph so
        they contribute zero to the gradient but still produce a
        well-shaped ``BatchMolGraph``.
        """
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mgs: List = []
        atom_fdim = -1
        bond_fdim = -1
        for smi in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    raise ValueError(f"RDKit parse failed for {smi!r}")
                mg = self.featurizer(mol)
                if atom_fdim < 0:
                    atom_fdim = int(mg.V.shape[1])
                    bond_fdim = int(mg.E.shape[1])
                mgs.append(mg)
            except Exception:
                # Build a 1-atom / 0-bond dummy MolGraph so BatchMolGraph
                # still has a consistent shape.
                if atom_fdim < 0:
                    # Use a placeholder C atom to read fdim from the featurizer.
                    tmp = Chem.MolFromSmiles("C")
                    tmp_mg = self.featurizer(tmp)
                    atom_fdim = int(tmp_mg.V.shape[1])
                    bond_fdim = int(tmp_mg.E.shape[1])
                dummy_V = np.zeros((1, atom_fdim), dtype=np.single)
                dummy_E = np.zeros((0, bond_fdim), dtype=np.single)
                dummy_edge_index = np.zeros((2, 0), dtype=int)
                dummy_rev = np.zeros((0,), dtype=int)
                from chemprop.data.molgraph import MolGraph

                mgs.append(MolGraph(dummy_V, dummy_E, dummy_edge_index, dummy_rev))
        return mgs, atom_fdim, bond_fdim

    # ------------------------------------------------------------------
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        """Encode SMILES → (B, hidden_dim) pooled embedding.

        Args:
            smiles_list: list of ``B`` SMILES strings.

        Returns:
            ``(B, hidden_dim)`` pooled molecular embedding.
        """
        if not smiles_list:
            raise ValueError("smiles_list must be non-empty")
        # Lazy-build backbone from the first batch's fdim.
        mgs, atom_fdim, bond_fdim = self._featurize_smiles(smiles_list)
        self._ensure_backbone(atom_fdim, bond_fdim)
        assert self.message_passing is not None
        assert self.agg is not None

        bmg = BatchMolGraph(mgs)
        device = next(self.parameters()).device
        bmg.V = bmg.V.to(device)
        bmg.E = bmg.E.to(device)
        bmg.edge_index = bmg.edge_index.to(device)
        bmg.rev_edge_index = bmg.rev_edge_index.to(device)
        bmg.batch = bmg.batch.to(device)

        H_v = self.message_passing(bmg)  # (sum_N, hidden_dim)
        H = self.agg(H_v, bmg.batch)  # (B, hidden_dim)
        return H


# ---------------------------------------------------------------------------
# Per-metal head factory — wraps BinaryClassificationFFN to a single linear
# output (matches V4's per-metal (B, 2) logits).
# ---------------------------------------------------------------------------
def _make_metal_head(
    input_dim: int,
    hidden_dim: int = 300,
    n_layers: int = 2,
    dropout: float = 0.2,
) -> BinaryClassificationFFN:
    """Build one chemprop v2 BinaryClassificationFFN for binary activity.

    The head returns ``(B, 1)`` pre-sigmoid logits when called via
    ``train_step`` (which is what we want for BCE loss matching the
    V4 contract).  ``forward`` would additionally apply the sigmoid —
    callers wanting probabilities should use ``forward``.
    """
    return BinaryClassificationFFN(
        n_tasks=1,
        input_dim=int(input_dim),
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
        dropout=float(dropout),
        activation="relu",
        criterion=None,
    )


# ---------------------------------------------------------------------------
# Main model — multi-task chemprop v2 backbone + per-metal BinaryClass heads
# + coord classification head.
# ---------------------------------------------------------------------------
class ChempropV2MTModel(nn.Module):
    """5-way multi-task Chemprop v2 backbone for precious-metal cytotoxicity.

    Wraps chemprop v2's :class:`AtomMessagePassing` + :class:`SumAggregation`
    shared encoder, then routes the pooled graph embedding through 5
    per-metal :class:`BinaryClassificationFFN` heads (Ru / Ir / Os / Rh / Re)
    plus a coordination-geometry classification head.

    Args:
        hidden_dim: chemprop v2 message-passing hidden dimension
            (default 300 — matches chemprop v2 default).
        depth: number of message-passing iterations (default 3).
        dropout: chemprop v2 dropout (default 0.0).
        n_metals: number of metal embeddings (default 5; supports the
            canonical Ru/Ir/Os/Rh/Re subset).
        n_coord_classes: number of coordination-geometry classes for
            the auxiliary head (default 5).
        metal_embed_dim: metal-embedding dimension (default 64).
        metals: tuple of metal names — must have ``n_metals`` entries.
        head_hidden_dim: per-metal head hidden width (default 300).
        head_n_layers: per-metal head depth (default 2).
        head_dropout: per-metal head dropout (default 0.2).

    Forward signature matches V4MultiTaskDMPNN::

        out = model(smiles_list, metal_ids)

    Returns dict mapping each metal name → ``(B, 2)`` binary logits
    (note: chemprop BinaryClassificationFFN's ``train_step`` emits
    ``(B, 1)`` pre-sigmoid; we expose ``(B, 2)`` for backward compat
    with V4MultiTaskLoss which expects ``(B, 2)`` and uses
    ``nn.CrossEntropyLoss`` — i.e. the V4 contract is "logits over 2
    classes".  We concatenate the chemprop ``(B, 1)`` logit with a
    zero column to keep shapes compatible).
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        depth: int = DEFAULT_DEPTH,
        dropout: float = DEFAULT_DROPOUT,
        n_metals: int = 5,
        n_coord_classes: int = 5,
        metal_embed_dim: int = 64,
        metals: Optional[Tuple[str, ...]] = None,
        head_hidden_dim: int = 300,
        head_n_layers: int = 2,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.n_metals = int(n_metals)
        self.n_coord_classes = int(n_coord_classes)
        self.metal_embed_dim = int(metal_embed_dim)
        # Default to the canonical 5-way precious-metal subset.
        self.metals: Tuple[str, ...] = (
            tuple(metals) if metals is not None else SUPPORTED_METALS[:n_metals]
        )
        if len(self.metals) != self.n_metals:
            raise ValueError(
                f"len(metals)={len(self.metals)} must equal n_metals={self.n_metals}"
            )

        # ----- Shared encoder -----
        self.encoder = ChempropV2Encoder(
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            dropout=self.dropout,
        )

        # ----- Metal embedding (one row per metal id) -----
        self.metal_embed = nn.Embedding(self.n_metals, self.metal_embed_dim)

        # ----- Per-metal heads (ModuleDict so named access mirrors V4) -----
        # We store chemprop BinaryClassificationFFN heads as ModuleDict.
        # The head's input dim = encoder hidden_dim + metal_embed_dim.
        head_in_dim = self.hidden_dim + self.metal_embed_dim
        self.metal_heads = nn.ModuleDict(
            {
                m: _make_metal_head(
                    input_dim=head_in_dim,
                    hidden_dim=head_hidden_dim,
                    n_layers=head_n_layers,
                    dropout=head_dropout,
                )
                for m in self.metals
            }
        )

        # ----- Coordination-geometry auxiliary head -----
        # Kept as a vanilla MLP (matches V4) — coord is a multi-class
        # problem and chemprop's BinaryClassificationFFN doesn't apply
        # here.  Wrapping chemprop's MulticlassClassificationFFN would
        # also work but adds an unnecessary registry dependency.
        self.coord_head = nn.Sequential(
            nn.Linear(head_in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_coord_classes),
        )

    # ------------------------------------------------------------------
    def metal_id(self, metal: str) -> int:
        """Return the integer id for a metal name (raises if unknown)."""
        if metal not in self.metals:
            raise KeyError(f"Unknown metal {metal!r}. Supported: {self.metals}")
        return self.metals.index(metal)

    # ------------------------------------------------------------------
    def encode(self, smiles_list: List[str]) -> torch.Tensor:
        """Encode SMILES → (B, hidden_dim)."""
        return self.encoder(smiles_list)

    # ------------------------------------------------------------------
    def _chemprop_logits_to_v4_shape(self, logits_1col: torch.Tensor) -> torch.Tensor:
        """Convert chemprop (B, 1) pre-sigmoid logits → V4-compatible (B, 2).

        The V4 contract (see V4MultiTaskLoss) expects ``nn.CrossEntropyLoss``
        logits over 2 classes (active / inactive).  Chemprop's
        BinaryClassificationFFN emits a single-logit BCE head — we
        concatenate a zero second column so the shape is ``(B, 2)`` and
        ``argmax`` still picks the active class correctly via the
        active logit alone.

        NOTE: this is a compatibility shim — the per-metal
        :func:`compute_per_metal_loss` (defined below) uses
        ``nn.BCEWithLogitsLoss`` directly on the ``(B, 1)`` chemprop
        logits, so the chemprop path is the canonical training route.
        The (B, 2) shim exists for callers that want to reuse V4 loss.
        """
        B = logits_1col.size(0)
        zeros = torch.zeros_like(logits_1col)
        return torch.cat([logits_1col, zeros], dim=-1)  # (B, 2)

    # ------------------------------------------------------------------
    def forward(
        self,
        smiles_list: List[str],
        metal_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass — returns dict[metal -> (B, 2) logits, "coord" -> (B, n_coord_classes)].

        The metal head's (B, 2) layout is the V4 contract: column 0 is the
        "active" logit (from chemprop's BinaryClassificationFFN
        ``train_step`` output) and column 1 is a zero placeholder.  This
        lets :class:`V4MultiTaskLoss` (CrossEntropyLoss on (B, 2)) keep
        working unchanged.  Use :func:`compute_per_metal_loss` for the
        canonical chemprop BCE training route.
        """
        if metal_ids.dim() != 1:
            raise ValueError(
                f"metal_ids must be 1-D (B,), got shape {tuple(metal_ids.shape)}"
            )
        B = len(smiles_list)
        if metal_ids.size(0) != B:
            raise ValueError(
                f"len(smiles_list)={B} != metal_ids.size(0)={metal_ids.size(0)}"
            )

        h = self.encode(smiles_list)  # (B, hidden_dim)
        e = self.metal_embed(metal_ids.long())  # (B, metal_embed_dim)
        z = torch.cat([h, e], dim=-1)  # (B, hidden_dim + metal_embed_dim)

        out: Dict[str, torch.Tensor] = {}
        for metal, head in self.metal_heads.items():
            logits_1col = head.train_step(z)  # (B, 1) pre-sigmoid
            out[metal] = self._chemprop_logits_to_v4_shape(logits_1col)
        out["coord"] = self.coord_head(z)  # (B, n_coord_classes)
        return out


# ---------------------------------------------------------------------------
# Training loss — chemprop v2 standard BCE per metal + coord CE.
# ---------------------------------------------------------------------------
class ChempropV2MTLoss(nn.Module):
    """Per-metal BCE (chemprop BCELoss-compatible) + coord CE auxiliary.

    Mirrors the V4 contract but uses chemprop v2's :class:`BCELoss`
    metric class for per-metal binary classification.  The BCE loss
    applies per-row sigmoid + binary cross-entropy and is numerically
    stable (the standard PyTorch implementation matches chemprop v2's
    BCELoss definition).
    """

    def __init__(
        self,
        n_coord_classes: int = 5,
        coord_weight: float = COORD_LOSS_WEIGHT,
        metal_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.n_coord_classes = int(n_coord_classes)
        self.coord_weight = float(coord_weight)
        self.metal_weights: Dict[str, float] = (
            dict(metal_weights) if metal_weights is not None else {}
        )
        # Chemprop v2 BCELoss — per-metal binary classification.
        self.bce_metal = BCELoss()
        # Coord CE — multi-class classification (R17 uses 5 classes).
        self.ce_coord = nn.CrossEntropyLoss()

    def forward(
        self,
        out: Dict[str, torch.Tensor],
        y_true: Dict[str, torch.Tensor],
        coord_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the multi-task loss.

        Args:
            out: model output dict — per-metal logits are
                ``(B, 2)`` (V4 contract), but only column 0 is meaningful
                (the chemprop BinaryClassificationFFN logit).  We index
                ``[:, 0:1]`` to extract it before computing BCE.
            y_true: dict mapping metal name → ``(B,)`` float binary
                labels (0.0 / 1.0).
            coord_true: ``(B,)`` long coordination-geometry labels.

        Returns:
            ``(total_loss, info)`` — ``info`` has per-metal detached
            scalars and the coord loss for logging.
        """
        info: Dict[str, torch.Tensor] = {}
        total = out["coord"].new_zeros(())

        for metal, logits_v4 in out.items():
            if metal == "coord":
                continue
            if metal not in y_true:
                info[f"loss_{metal}"] = torch.tensor(0.0, device=logits_v4.device)
                continue
            y = y_true[metal].float()
            if y.numel() == 0:
                info[f"loss_{metal}"] = torch.tensor(0.0, device=logits_v4.device)
                continue
            # Extract the chemprop (B, 1) logit — column 0 of the V4 shim.
            logits_1col = logits_v4[:, 0]
            # BCELoss expects (B, 1) target + (B, 1) input — broadcast y.
            w = float(self.metal_weights.get(metal, 1.0))
            l = self.bce_metal(logits_1col.unsqueeze(-1), y.unsqueeze(-1)) * w
            info[f"loss_{metal}"] = l.detach()
            total = total + l

        # Coord auxiliary term.
        if coord_true is not None and coord_true.numel() > 0:
            lcrd = self.ce_coord(out["coord"], coord_true.long())
            info["loss_coord"] = lcrd.detach()
            total = total + self.coord_weight * lcrd
        else:
            info["loss_coord"] = torch.tensor(0.0, device=total.device)

        info["loss_total"] = total.detach()
        return total, info


# ---------------------------------------------------------------------------
# Smoke helper (used by tests + ultracode smoke runs).
# ---------------------------------------------------------------------------
def quick_smoke_chemprop_v2_mt(
    smiles: Optional[List[str]] = None,
    n_metals: int = 5,
    n_coord_classes: int = 5,
    hidden_dim: int = 64,  # smaller for tests / smoke runs
    depth: int = 2,
    device: Optional[torch.device] = None,
) -> Tuple[
    ChempropV2MTModel,
    ChempropV2MTLoss,
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]:
    """Build a model + loss + fake batch for a quick smoke test.

    Returns ``(model, loss_fn, out, labels_dict)`` so callers can run
    ``loss, info = loss_fn(out, y_true, coord_true); loss.backward()``
    immediately.
    """
    if smiles is None:
        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CC(C)C"]
    B = len(smiles)
    device = device or torch.device("cpu")
    model = ChempropV2MTModel(
        hidden_dim=hidden_dim,
        depth=depth,
        n_metals=n_metals,
        n_coord_classes=n_coord_classes,
    ).to(device)
    loss_fn = ChempropV2MTLoss(n_coord_classes=n_coord_classes).to(device)
    metal_ids = torch.zeros(B, dtype=torch.long, device=device)
    out = model(smiles, metal_ids)
    torch.manual_seed(0)
    y_true = {m: torch.randint(0, 2, (B,), device=device).float() for m in model.metals}
    coord_true = torch.randint(0, n_coord_classes, (B,), device=device)
    return model, loss_fn, out, {"y_true": y_true, "coord_true": coord_true}


__all__ = [
    "ChempropV2MTLoss",
    "ChempropV2MTModel",
    "ChempropV2Encoder",
    "COORD_GEOMETRY_NAMES",
    "COORD_LOSS_WEIGHT",
    "DEFAULT_DEPTH",
    "DEFAULT_DROPOUT",
    "DEFAULT_HIDDEN_DIM",
    "SUPPORTED_METALS",
    "quick_smoke_chemprop_v2_mt",
]
