"""3-cell-line multi-task D-MPNN for cisplatin resistance — Phase A / Stage 2.3.

Extends the metal-cytox multi-task pattern
(:class:`molmetal.models.v4_multitask_dmpnn.V4MultiTaskDMPNN`) to
**cell-line conditioned** resistance prediction. The classifier predicts,
for each of three breast / ovarian / lung cancer cell lines, whether a
Pt(II) complex will be active (IC50 below a cohort threshold) given:

    - the molecule's 2D structure  -> D-MPNN graph embedding h
    - a chemoinformatic resistance fingerprint -> r  (n_resist_feat -> 64)
    - the target cell line ID         -> c  (n_cells -> 32)

These three streams are concatenated to form the head input
z = [h ; r ; c], and one 2-class head per cell line produces logits.

Class imbalance is handled with focal loss (Lin et al. 2017). The shared
encoder + chemoinformatic side-features reflect the Phase A / Stage 2.3
hypothesis: cell-line resistance phenotypes are driven by **both**
molecular structure and QSAR descriptors (logP, thiol count, donor
count, cis/trans geometry, reduction potential, MW, TPSA, HBA, HBD).

Cell lines (default)
--------------------
``MCF-7`` — breast adenocarcinoma, cisplatin-sensitive reference
``A549``  — non-small-cell lung carcinoma
``A2780cis`` — cisplatin-resistant ovarian carcinoma (the canonical
              resistant line; cross-resistance benchmark)

Forward signature
-----------------
::

    out = model(smiles_list, cell_ids, resist_feat)
    # out = {"MCF-7": (B,2), "A549": (B,2), "A2780cis": (B,2)}

Loss (per-cell, summed)
-----------------------
::

    loss = sum_c focal_loss(out[c], y_true[c])
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from molmetal.models.v4_multitask_dmpnn import DMPNNEncoder
from molmetal_lam.sbdd_env.protein_metal_binding import focal_loss


# ---------------------------------------------------------------------------
# Constants — keep in sync with the Phase A plan
# ---------------------------------------------------------------------------
#: Default cell-line roster (3 cell lines for Phase A / Stage 2.3).
DEFAULT_CELL_LINES: Tuple[str, ...] = ("MCF-7", "A549", "A2780cis")

#: Default resistance-feature dimension.  Matches the 9-element
#: :data:`molmetal_lam.lam_chem.cisplatin_resistance.RESISTANCE_FEATURE_KEYS`
#: tuple, padded with one sparsity indicator so the model consumes a
#: fixed 10-dim input regardless of how the chemoinformatic pipeline
#: evolves (e.g. adding GSH-trap features later).
DEFAULT_N_RESIST_FEAT: int = 10

#: Dropout inside each per-cell head.
HEAD_DROPOUT: float = 0.3

#: Cell-embedding dimension.
CELL_EMBED_DIM: int = 32

#: Resistance-projection output dimension.
RESIST_PROJ_DIM: int = 64


# ---------------------------------------------------------------------------
# Per-cell head factory
# ---------------------------------------------------------------------------
def _make_cell_head(
    in_dim: int,
    hidden_dim: int = 128,
    n_classes: int = 2,
    dropout: float = HEAD_DROPOUT,
) -> nn.Sequential:
    """Build one cell-line head: Linear -> ReLU -> Dropout -> Linear.

    Mirrors the head shape used by
    :class:`molmetal.models.v4_multitask_dmpnn.V4MultiTaskDMPNN` but with
    the Phase A spec's dropout rate (0.3 vs 0.2 — reflects the smaller
    per-cell training set per head and the heavier class imbalance in
    the cisplatin-resistant cohort).
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, n_classes),
    )


# ---------------------------------------------------------------------------
# Multi-task model
# ---------------------------------------------------------------------------
class CisplatinResistanceDMPNN(nn.Module):
    """3-cell-line multi-task D-MPNN with chemoinformatic side-features.

    Architecture (per Phase A / Stage 2.3 spec)::

        smiles  --> DMPNNEncoder --> h                  (B, hidden_dim)
        res_feat --> Linear(n_resist_feat, 64) --> r    (B, 64)
        cell_id --> Embedding(n_cells, 32) --> c        (B, 32)
        z = cat([h, r, c], dim=-1)                      (B, hidden+64+32)

        per-cell head (3x):
            Linear(z -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 2)

    Args:
        hidden_dim: D-MPNN hidden width (default 300 per spec).
        n_layers:   D-MPNN message-passing layers (default 3 per spec).
        n_cells:    number of cell lines (default 3).
        n_resist_feat: dimensionality of the chemoinformatic
            resistance feature vector (default 10).
        cell_lines: ordered tuple of cell-line labels — must have
            ``n_cells`` entries.  Used as keys for the per-cell
            :class:`nn.ModuleDict` and as the public label set.
        head_hidden_dim: per-cell head hidden width (default 128).
        head_dropout: per-cell head dropout (default 0.3 per spec).
    """

    def __init__(
        self,
        hidden_dim: int = 300,
        n_layers: int = 3,
        n_cells: int = 3,
        n_resist_feat: int = DEFAULT_N_RESIST_FEAT,
        cell_lines: Optional[Tuple[str, ...]] = None,
        head_hidden_dim: int = 128,
        head_dropout: float = HEAD_DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_cells = int(n_cells)
        self.n_resist_feat = int(n_resist_feat)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_dropout = float(head_dropout)

        # ----- cell-line label set -----
        self.cell_lines: Tuple[str, ...] = (
            tuple(cell_lines) if cell_lines is not None else DEFAULT_CELL_LINES[:n_cells]
        )
        if len(self.cell_lines) != self.n_cells:
            raise ValueError(
                f"len(cell_lines)={len(self.cell_lines)} must equal "
                f"n_cells={self.n_cells}"
            )

        # ----- shared encoder (re-uses the V4 backbone so the test
        #       surface and featuriser stay consistent across the family) -----
        self.encoder = DMPNNEncoder(
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
        )

        # ----- resistance side-features projection -----
        self.resist_proj = nn.Linear(self.n_resist_feat, RESIST_PROJ_DIM)

        # ----- cell-line embedding (one row per cell id) -----
        self.cell_embed = nn.Embedding(self.n_cells, CELL_EMBED_DIM)

        # ----- per-cell heads (ModuleDict so named access mirrors spec) -----
        head_in = self.hidden_dim + RESIST_PROJ_DIM + CELL_EMBED_DIM
        self.cell_heads = nn.ModuleDict(
            {
                cl: _make_cell_head(
                    in_dim=head_in,
                    hidden_dim=self.head_hidden_dim,
                    n_classes=2,
                    dropout=self.head_dropout,
                )
                for cl in self.cell_lines
            }
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        smiles_list: List[str],
        cell_ids: torch.Tensor,
        resist_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run the multi-task forward pass.

        Args:
            smiles_list: list of B SMILES strings (one per row).
            cell_ids:    (B,) long tensor in ``[0, n_cells)``.
            resist_feat: (B, n_resist_feat) float tensor — chemoinformatic
                resistance fingerprint.  Compute via
                :func:`molmetal_lam.lam_chem.cisplatin_resistance.compute_resistance_features`
                and pad to ``n_resist_feat`` (the 9 chemoinformatic keys +
                one padding indicator slot).

        Returns:
            ``dict[cell_line -> (B, 2) logits]``.  Keys exactly match
            :attr:`cell_lines`.
        """
        if cell_ids.dim() != 1:
            raise ValueError(
                f"cell_ids must be 1-D, got shape {tuple(cell_ids.shape)}"
            )
        if cell_ids.size(0) != len(smiles_list):
            raise ValueError(
                f"cell_ids.size(0)={cell_ids.size(0)} must equal "
                f"len(smiles_list)={len(smiles_list)}"
            )
        if resist_feat.dim() != 2 or resist_feat.size(0) != len(smiles_list):
            raise ValueError(
                f"resist_feat must be (B, n_resist_feat); got "
                f"{tuple(resist_feat.shape)} for B={len(smiles_list)}"
            )
        if resist_feat.size(1) != self.n_resist_feat:
            raise ValueError(
                f"resist_feat.size(1)={resist_feat.size(1)} must equal "
                f"n_resist_feat={self.n_resist_feat}"
            )

        h = self.encoder(smiles_list)            # (B, hidden_dim)
        r = self.resist_proj(resist_feat)        # (B, 64)
        c = self.cell_embed(cell_ids)            # (B, 32)
        z = torch.cat([h, r, c], dim=-1)         # (B, hidden+64+32)
        return {cl: head(z) for cl, head in self.cell_heads.items()}


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------
def cisplatin_resistance_loss(
    logits: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    gamma: float = 2.0,
    alpha: float = -1.0,
) -> torch.Tensor:
    """Sum of focal losses across the cell-line heads.

    Args:
        logits:  ``{cell_line: (B, 2) logits}`` from
                 :meth:`CisplatinResistanceDMPNN.forward`.
        targets: ``{cell_line: (B,) long in {0, 1}}``.
        gamma:   focal-loss focusing parameter (default 2.0 per spec).
        alpha:   ``< 0`` -> auto-balance from the mini-batch prior
                 (recommended for the cisplatin-resistant cohort where
                 the positive class is rare; mirrors the V4 multi-task
                 pattern).

    Returns:
        Scalar tensor — *sum* over cell lines (NOT mean), so the loss
        magnitude is comparable across configs with different numbers
        of cell lines.
    """
    losses = []
    for cl, lg in logits.items():
        if cl not in targets:
            raise KeyError(f"missing target for cell line {cl!r}")
        losses.append(focal_loss(lg, targets[cl], gamma=gamma, alpha=alpha))
    return torch.stack(losses).sum()


__all__ = [
    "CisplatinResistanceDMPNN",
    "DEFAULT_CELL_LINES",
    "DEFAULT_N_RESIST_FEAT",
    "HEAD_DROPOUT",
    "CELL_EMBED_DIM",
    "RESIST_PROJ_DIM",
    "cisplatin_resistance_loss",
]
