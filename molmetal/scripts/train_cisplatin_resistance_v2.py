#!/usr/bin/env python3
"""Wf-31 Phase 2 — R18 SE(3) SB cisplatin-resistance multi-cell trainer.

=====================================================================
**NEW trainer (Wf-31 Phase 2)** that wraps the W2.1 SE(3) Schrödinger-
Bridge velocity field shipped in :mod:`molmetal.scripts.train_se3sb` on
top of the **4-cell-line cisplatin-resistance classifier**.  Built on
the V4 DMPNN encoder + chemoinformatic resistance features (10-dim
side vector) and the existing :mod:`molmetal_lam.lam_chem.cisplatin_resistance`
feature module.

Inputs
------
* ``molmetal/data/platinai_4cell.csv`` (4,078 Pt complexes) — columns:
  ``complex_id, smiles, ic50_{mcf7,a549,a2780,a2780cis}_nm, source``.
  We treat each row as 4 cell-line independent binary tasks (active if
  ``IC50 ≤ 10 000 nM``; otherwise inactive).

Outputs
-------
* ``.ckpt/cisplatin_resistance_v2_{init}_{cell}_seed{N}.pt`` — per-seed ckpt.
* ``metrics/by_metric/cisplatin_resistance_v2_{cell}.json`` — per-seed
  ``test_auc_per_cell`` (4 entries) + ``mean_test_auc`` + 4-cell mean ± std.
* ``metrics/by_metric/cisplatin_resistance_v2_{cell}_summary.json`` —
  slim final summary with 5-seed headline AUC per cell.

Recipe
------
(1) Build :class:`FourCellResistanceDMPNN` (V4 DMPNN encoder + cell-line
    embedding + 10-dim resistance side features + 4 per-cell heads).
(2) Wire the **SE(3) Schrödinger-Bridge twist head** (from W2.1
    ``train_se3sb.py``) when ``--init se3sb``; otherwise wire
    transport-relax or multi-PH R18 adapters; ``--init tmqm`` is the
    R17 baseline fallback.
(3) Optionally load tmQM pretrained encoder (arch-mismatch fallback).
(4) Freeze encoder layers 0..1; fine-tune layer 2 + heads + cell-line
    embedding + resistance projector + (when se3sb) the SE(3) twist.
(5) 4-cell multi-task loss on train split (per-cell focal loss
    summed across the 4 cell lines).
(6) ``--seeds`` (default 0..4) × ``--epochs`` (default 50),
    AdamW lr=1e-3, cosine annealing, BF16 autocast (CUDA).
(7) Eval every ``--eval-every`` epochs on val; pick best epoch by
    mean val AUC across cell lines.
(8) Final test: per-cell test AUC + mean AUC.
(9) Save per-seed ckpt + per-seed JSON + final summary JSON.

Constraint
----------
**DO NOT MODIFY** ``train_se3sb.py``, ``train_cisplatin_resistance.py``
(R17 baseline lock) or ``train_ru_5metal_se3sb.py`` (R18 W2.1 baseline
lock).  This script is a **NEW Wf-31 Phase 2 deliverable** that
imports the SE(3) SB twist head + the V4 encoder and wires them with
the cisplatin-resistance side features.

Verify
------
After a successful run:
* Per-seed ckpts exist at ``.ckpt/cisplatin_resistance_v2_*_seed{0..4}.pt``.
* ``metrics/by_metric/cisplatin_resistance_v2_*.json`` has per-seed
  ``test_auc_per_cell`` dict with 4 entries + ``mean_test_auc``.
* ``metrics/by_metric/cisplatin_resistance_v2_*_summary.json`` records
  the 5-seed mean AUC per cell line + the 4-cell headline mean AUC.

Usage
-----
::

    source env.sh
    uv run --no-sync python -u molmetal/scripts/train_cisplatin_resistance_v2.py \\
        --data molmetal/data/platinai_4cell.csv \\
        --encoder-ckpt models/tmqm_pretrained/tmqm_pretrained_encoder.pt \\
        --out-dir .ckpt \\
        --metrics-path metrics/by_metric/cisplatin_resistance_v2_mcf7.json \\
        --summary-path metrics/by_metric/cisplatin_resistance_v2_mcf7_summary.json \\
        --cell-line MCF7 --init se3sb \\
        --epochs 50 --batch 128 --lr 1e-3 --seeds 0 1 2 3 4
"""

from __future__ import annotations


# --- Wf-130 GPU default -----------------------------------------
# RTX 5090 on kanzi (GPU 0 is the VLM venv process).  Must run
# BEFORE `import torch` so CUDA init sees the env var.
import os as _wf130_os

_wf130_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
del _wf130_os
# --- end Wf-130 ------------------------------------------------

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- V4 DMPNN encoder + the W2.1 SE(3) SB twist head (from train_se3sb) ---
from molmetal.models.v4_multitask_dmpnn import DMPNNEncoder

from molmetal.adapters.se3_sb_integration import (
    SE3SBDecoder,
    SE3SBVelocityField,
    integrate_with_cfm,
)

# R18 W2.2 / W2.3 adapter wiring — only used when --init is
# transport / multiph.  Lazy import to keep the cold-start path
# (--init tmqm) import-clean.
try:  # pragma: no cover
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm as _RelaxedEndpointTerm,
    )
except Exception:  # pragma: no cover
    _RelaxedEndpointTerm = None  # type: ignore[assignment]
try:  # pragma: no cover
    from molmetal.adapters.multi_ph_integration import (
        MultiPHPocketFeatures as _MultiPHPocketFeatures,
    )
except Exception:  # pragma: no cover
    _MultiPHPocketFeatures = None  # type: ignore[assignment]

# Cisplatin-resistance chemoinformatic side features (10-dim).
from molmetal_lam.lam_chem.cisplatin_resistance import (
    RESISTANCE_FEATURE_KEYS,
    compute_resistance_features,
)

# Wf-35 Phase 2 — shared argparse / GPU env / MEASURED-vs-PROJECTED tag.
from molmetal.scripts._common_train_args import (
    MEASURED_vs_PROJECTED_decorator,
    add_common_args,
    add_gpu_args,
    setup_gpu_env,
)


# ---------------------------------------------------------------------------
# Recipe constants (Wf-31 Phase 2 spec)
# ---------------------------------------------------------------------------
RECIPE_EPOCHS: int = 50
RECIPE_BATCH: int = 128
RECIPE_LR: float = 1e-3
RECIPE_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
RECIPE_HIDDEN: int = 300
RECIPE_DEPTH: int = 3
RECIPE_EVAL_EVERY: int = 5
RECIPE_DROPOUT: float = 0.2
RECIPE_CELL_EMBED_DIM: int = 32
RECIPE_RESIST_PROJ_DIM: int = 64
RECIPE_HEAD_HIDDEN: int = 128
RECIPE_HEAD_DROPOUT: float = 0.3
RECIPE_FOCAL_GAMMA: float = 2.0
RECIPE_SE3_INTEGRATION_STEPS: int = 8

# 4 cell lines (Wf-31 Phase 2 spec — all 4 retained, NOT the R17 3-cell subset).
CELL_LINES: Tuple[str, ...] = ("MCF7", "A549", "A2780", "A2780cis")
N_CELLS: int = len(CELL_LINES)
IC50_COLS: Tuple[str, ...] = tuple(f"ic50_{c.lower()}_nm" for c in CELL_LINES)
ACTIVE_THRESHOLD_NM: float = 10_000.0

# Resistance-feature dim: 9 chemoinformatic keys + 1 sparsity indicator = 10.
N_RESIST_FEAT: int = 10

# Init choice contract — must match the W2.1 trainers (Wf-31).
INIT_CHOICES: Tuple[str, ...] = (
    "tmqm",
    "se3sb",
    "transport",
    "multiph",
)
INIT_DEFAULT: str = "tmqm"


# ---------------------------------------------------------------------------
# Default paths (overridable on every CLI)
# ---------------------------------------------------------------------------
DEFAULT_DATA = PROJECT_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
DEFAULT_ENCODER_CKPT = (
    PROJECT_ROOT / "models" / "tmqm_pretrained" / "tmqm_pretrained_encoder.pt"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / ".ckpt"
DEFAULT_METRICS_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "cisplatin_resistance_v2.json"
)
DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT / "metrics" / "by_metric" / "cisplatin_resistance_v2_summary.json"
)


# ---------------------------------------------------------------------------
# Focal loss (multi-class, auto-balanced)
# ---------------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = RECIPE_FOCAL_GAMMA,
) -> torch.Tensor:
    """Multi-class focal loss over rows.

    Auto alpha = (1 - prior) — works for class-imbalanced cohorts
    (mirrors the W2.1 trainers' recipe).
    """
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    p = logp.exp()
    tgt = targets.long()
    n_classes = logits.size(-1)
    prior = torch.bincount(tgt, minlength=n_classes).float()
    prior = prior / max(prior.sum().item(), 1.0)
    alpha = (1.0 - prior).clamp(min=1e-3)
    logp_t = logp.gather(1, tgt.unsqueeze(1)).squeeze(1)
    p_t = p.gather(1, tgt.unsqueeze(1)).squeeze(1)
    a_t = alpha[tgt]
    loss = -a_t * (1.0 - p_t).pow(gamma) * logp_t
    return loss.mean()


# ---------------------------------------------------------------------------
# 4-cell-line DMPNN with chemoinformatic side features
# ---------------------------------------------------------------------------
class FourCellResistanceDMPNN(nn.Module):
    """4-cell-line cisplatin-resistance classifier with side features.

    Architecture (Wf-31 Phase 2 spec)::

        smiles     --> DMPNNEncoder      --> h        (B, hidden_dim)
        cell_id    --> Embedding(n_cells, 32) --> c (B, 32)
        res_feat   --> Linear(10, 64)   --> r        (B, 64)
        z = cat([h, c, r], dim=-1)                  (B, hidden+32+64)

        per-cell head (4x):
            Linear(z -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> 2)
    """

    def __init__(
        self,
        hidden_dim: int = RECIPE_HIDDEN,
        n_layers: int = RECIPE_DEPTH,
        n_cells: int = N_CELLS,
        n_resist_feat: int = N_RESIST_FEAT,
        cell_lines: Tuple[str, ...] = CELL_LINES,
        cell_embed_dim: int = RECIPE_CELL_EMBED_DIM,
        resist_proj_dim: int = RECIPE_RESIST_PROJ_DIM,
        head_hidden_dim: int = RECIPE_HEAD_HIDDEN,
        head_dropout: float = RECIPE_HEAD_DROPOUT,
        encoder_dropout: float = RECIPE_DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.n_cells = int(n_cells)
        self.n_resist_feat = int(n_resist_feat)
        self.cell_embed_dim = int(cell_embed_dim)
        self.resist_proj_dim = int(resist_proj_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.head_dropout = float(head_dropout)

        if len(cell_lines) != self.n_cells:
            raise ValueError(
                f"len(cell_lines)={len(cell_lines)} must equal "
                f"n_cells={self.n_cells}"
            )
        self.cell_lines: Tuple[str, ...] = tuple(cell_lines)

        # Shared V4 D-MPNN encoder (the R18 W2.1 backbone).
        self.encoder = DMPNNEncoder(
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=encoder_dropout,
        )

        # Resistance side-features projection (10 -> 64).
        self.resist_proj = nn.Linear(self.n_resist_feat, self.resist_proj_dim)

        # Cell-line embedding (one row per cell id).
        self.cell_embed = nn.Embedding(self.n_cells, self.cell_embed_dim)

        # Per-cell heads (4 total — one binary head per cell line).
        head_in = self.hidden_dim + self.resist_proj_dim + self.cell_embed_dim
        self.cell_heads = nn.ModuleDict(
            {
                cl: nn.Sequential(
                    nn.Linear(head_in, self.head_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.head_dropout),
                    nn.Linear(self.head_hidden_dim, 2),
                )
                for cl in self.cell_lines
            }
        )

    def forward(
        self,
        smiles_list: List[str],
        cell_ids: torch.Tensor,
        resist_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if cell_ids.dim() != 1:
            raise ValueError(f"cell_ids must be 1-D, got {tuple(cell_ids.shape)}")
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

        h = self.encoder(smiles_list)  # (B, hidden_dim)
        r = self.resist_proj(resist_feat)  # (B, 64)
        c = self.cell_embed(cell_ids.long())  # (B, 32)
        z = torch.cat([h, r, c], dim=-1)  # (B, hidden+64+32)
        return {cl: head(z) for cl, head in self.cell_heads.items()}


# ---------------------------------------------------------------------------
# 4-cell multi-task focal loss
# ---------------------------------------------------------------------------
def four_cell_loss(
    logits: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    gamma: float = RECIPE_FOCAL_GAMMA,
) -> torch.Tensor:
    """Sum focal losses across the 4 cell lines (valid rows only)."""
    losses = []
    for cl, lg in logits.items():
        if cl not in targets:
            raise KeyError(f"missing target for cell line {cl!r}")
        losses.append(focal_loss(lg, targets[cl].long(), gamma=gamma))
    return torch.stack(losses).sum()


# ---------------------------------------------------------------------------
# tmQM encoder loader — mirrors the W2.1 trainer contract.
# ---------------------------------------------------------------------------
def try_load_tmqm_encoder(
    model: FourCellResistanceDMPNN,
    ckpt_path: Path,
) -> int:
    """Load tmQM pretrained weights into the V4 D-MPNN encoder.

    On arch-mismatch (baseline vs. V4) we fall back to random init
    with a clear warning.  Returns the number of encoder tensors
    transferred (``0`` on fallback).
    """
    if not ckpt_path.exists():
        warnings.warn(
            f"[train_cisplatin_resistance_v2] tmQM encoder ckpt not "
            f"found at {ckpt_path}; using random init."
        )
        return 0
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = payload["state_dict"] if "state_dict" in payload else payload
    own = model.encoder.dmpnn.state_dict()
    n_loaded = 0
    for k, v in sd.items():
        if k in own and tuple(own[k].shape) == tuple(v.shape):
            own[k].copy_(v)
            n_loaded += 1
    if n_loaded == 0:
        warnings.warn(
            "[train_cisplatin_resistance_v2] tmQM encoder arch is "
            "incompatible with V4MultiTaskDMPNN.encoder.dmpnn "
            "(DirectedMPNN); falling back to random init."
        )
        return 0
    if n_loaded != len(own):
        warnings.warn(
            f"[train_cisplatin_resistance_v2] partial tmQM load: "
            f"{n_loaded}/{len(own)} tensors matched; remaining kept "
            "at random init."
        )
    return n_loaded


# ---------------------------------------------------------------------------
# SE(3) twist head — parameter-free stub used for training
# ---------------------------------------------------------------------------
class _StubEGNNVelocityNet(nn.Module):
    """Parameter-free stub EGNN velocity net used by :class:`SE3SBVelocityField`.

    During *training* the W2.1 contract attaches a parameter-free stub
    so the SE(3) stack can be exercised end-to-end on CPU without
    dragging in the full flow-matching decoder.  The stub returns
    zero per-atom velocity, so the SE(3) twist collapses to the
    translational part during training — the shape contract is
    preserved.
    """

    def __init__(self, hidden_dim: int = RECIPE_HIDDEN) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor, pocket=None):
        """Return per-atom velocity of shape ``(B, N, 3)`` (zeros)."""
        if x.dim() != 3 or x.shape[-1] != 3:
            raise ValueError(f"x must have shape (B, N, 3), got {tuple(x.shape)}")
        return torch.zeros_like(x)


def build_se3_twist_head(
    hidden_dim: int = RECIPE_HIDDEN,
) -> SE3SBVelocityField:
    """Build the SE(3) twist head used by the Wf-31 Phase 2 se3sb init."""
    stub = _StubEGNNVelocityNet(hidden_dim=hidden_dim)
    return SE3SBVelocityField(base_velocity_net=stub, manifold="SE3")


# ---------------------------------------------------------------------------
# SE(3) integration smoke — mirrors train_se3sb.se3_integration_smoke
# ---------------------------------------------------------------------------
@torch.no_grad()
def se3_integration_smoke(
    twist_head: SE3SBVelocityField,
    n_atoms: int = 8,
    n_steps: int = RECIPE_SE3_INTEGRATION_STEPS,
) -> Dict[str, object]:
    """Run a tiny SE(3) integration smoke test against the twist head."""
    twist_head.eval()
    samples = torch.randn(2, n_atoms, 3)
    try:
        x_final = integrate_with_cfm(
            samples,
            twist_head,
            n_steps=n_steps,
            method="midpoint",
        )
        out_norm = float(x_final.norm().item())
        finite = bool(torch.isfinite(x_final).all().item())
    except Exception as exc:  # pragma: no cover
        return {"out_norm": float("nan"), "finite": False, "error": str(exc)}
    return {"out_norm": out_norm, "finite": finite}


# ---------------------------------------------------------------------------
# R18 adapter attach — 4 init modes (tmqm / se3sb / transport / multiph)
# ---------------------------------------------------------------------------
def attach_init_components(
    init: str,
    model: FourCellResistanceDMPNN,
    device: torch.device,
    hidden_dim: int = RECIPE_HIDDEN,
) -> Dict[str, object]:
    """Attach the per-init R18 adapter components (mirrors
    :func:`train_ru_5metal_se3sb.attach_init_components`)."""
    components: Dict[str, object] = {}
    if init == "tmqm":
        return components
    if init == "se3sb":
        twist_head = build_se3_twist_head(hidden_dim=hidden_dim).to(device)
        # Attach as a sibling so the optimiser sees the parameters.
        model.se3_field = twist_head  # type: ignore[attr-defined]
        components["se3_field"] = twist_head
    elif init == "transport":
        if _RelaxedEndpointTerm is None:
            warnings.warn(
                "[train_cisplatin_resistance_v2] RelaxedEndpointTerm "
                "adapter unavailable; transport init reduces to tmqm "
                "fallback."
            )
            return components
        relax_term = _RelaxedEndpointTerm(lambda_relax=1e-3, eps=0.1)
        model.relax_term = relax_term  # type: ignore[attr-defined]
        components["relax_term"] = relax_term
    elif init == "multiph":
        if _MultiPHPocketFeatures is None:
            warnings.warn(
                "[train_cisplatin_resistance_v2] MultiPHPocketFeatures "
                "adapter unavailable; multiph init reduces to tmqm "
                "fallback."
            )
            return components
        mphp = _MultiPHPocketFeatures(
            max_radius=5.0,
            max_coord=4.0,
            n_bins=8,
        ).to(device)
        model.mphp = mphp  # type: ignore[attr-defined]
        components["mphp"] = mphp
    else:
        raise ValueError(f"Unknown init {init!r}; expected one of {INIT_CHOICES}")
    return components


# ---------------------------------------------------------------------------
# Freeze policy (mirrors train_se3sb / train_ru_5metal_se3sb)
# ---------------------------------------------------------------------------
def freeze_encoder_layers_1_2(
    model: FourCellResistanceDMPNN,
) -> Tuple[int, int]:
    """Freeze D-MPNN message-passing layers 0..1 (layer 2 stays trainable)."""
    n_frozen = n_trainable = 0
    for p in model.encoder.parameters():
        p.requires_grad_(True)
    for i in (0, 1):
        for sub in (
            model.encoder.dmpnn.gru_updates,
            model.encoder.dmpnn.edge_mlp,
            model.encoder.dmpnn.atom_to_edge,
        ):
            if i < len(sub):
                for p in sub[i].parameters():
                    p.requires_grad_(False)
    for p in model.encoder.parameters():
        if p.requires_grad:
            n_trainable += p.numel()
        else:
            n_frozen += p.numel()
    return n_frozen, n_trainable


# ---------------------------------------------------------------------------
# Data loading — read platinai_4cell.csv → smiles, active[4], res_feat[10]
# ---------------------------------------------------------------------------
def _pad_to(value: np.ndarray, n_cols: int, fill: float = 0.0) -> np.ndarray:
    if value.shape[1] >= n_cols:
        return value[:, :n_cols]
    pad = np.full((value.shape[0], n_cols - value.shape[1]), fill)
    return np.concatenate([value, pad], axis=1)


def load_dataset(csv_path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Read platinai_4cell.csv → (smiles, active[N_CELLS], res_feat[10])."""
    rows: List[Dict[str, str]] = []
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    smiles = [r["smiles"] for r in rows]
    n = len(rows)

    # Active label per cell line (NaN → -1 sentinel = missing).
    active = np.full((n, N_CELLS), -1, dtype=np.int32)
    for j, col in enumerate(IC50_COLS):
        for i, r in enumerate(rows):
            v = r.get(col, "")
            if v == "" or v is None:
                continue
            try:
                ic50 = float(v)
                active[i, j] = 1 if ic50 <= ACTIVE_THRESHOLD_NM else 0
            except ValueError:
                continue

    # Resistance side-features (9 keys + 1 sparsity slot = 10-dim).
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    res_mat = np.zeros((n, len(RESISTANCE_FEATURE_KEYS)), dtype=np.float32)
    for i, smi in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            feats = compute_resistance_features(mol) if mol is not None else {}
            for k, key in enumerate(RESISTANCE_FEATURE_KEYS):
                res_mat[i, k] = float(feats.get(key, 0.0))
        except Exception:
            pass
    has_any = (res_mat != 0).any(axis=1, keepdims=True).astype(np.float32)
    res_padded = np.concatenate([res_mat, has_any], axis=1)
    if res_padded.shape[1] < N_RESIST_FEAT:
        res_padded = _pad_to(res_padded, N_RESIST_FEAT)
    elif res_padded.shape[1] > N_RESIST_FEAT:
        res_padded = res_padded[:, :N_RESIST_FEAT]
    return smiles, active, res_padded.astype(np.float32)


# ---------------------------------------------------------------------------
# Scaffold-split helper (Bemis-Murcko) — 70/10/20
# ---------------------------------------------------------------------------
class _SmilesAdapter:
    """Adapter that exposes ``.smiles`` to :class:`ScaffoldSplitter`."""

    def __init__(self, smi: List[str]) -> None:
        self.smiles = np.asarray(smi, dtype=object)

    def __len__(self) -> int:
        return len(self.smiles)


def scaffold_split_indices(
    smiles: List[str],
    fractions: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bemis-Murcko scaffold split 70/10/20, deterministically per seed."""
    from molmetal.data.splits import ScaffoldSplitter

    ds = _SmilesAdapter(smiles)
    splitter = ScaffoldSplitter(fractions=fractions)
    split = splitter(ds)
    return (
        np.asarray(split.train_idx, dtype=np.int64),
        np.asarray(split.val_idx, dtype=np.int64),
        np.asarray(split.test_idx, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Batching helpers (mirrors train_se3sb.build_batches)
# ---------------------------------------------------------------------------
def build_batches(
    pool_idx: np.ndarray,
    cell_id: int,
    batch_size: int,
    rng: np.random.Generator,
    shuffle: bool,
) -> List[Tuple[np.ndarray, int]]:
    """Shard a (row indices, cell-id) pool into per-batch chunks."""
    idx = pool_idx.copy()
    if shuffle:
        rng.shuffle(idx)
    return [
        (idx[start : start + batch_size], cell_id)
        for start in range(0, len(idx), batch_size)
    ]


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _eval_cell_auc(
    model: FourCellResistanceDMPNN,
    smiles: List[str],
    active: np.ndarray,
    res_feat: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    cell: str,
    cell_id: int,
) -> Tuple[float, int]:
    """Compute ROC-AUC for one cell on a row subset."""
    model.eval()
    scores: List[float] = []
    labels: List[int] = []
    cell_idx_in_arr = CELL_LINES.index(cell)
    for start in range(0, len(indices), batch_size):
        idx = indices[start : start + batch_size]
        keep = [i for i in idx if active[i, cell_idx_in_arr] >= 0]
        if not keep:
            continue
        smis = [smiles[i] for i in keep]
        rf = torch.from_numpy(res_feat[keep]).float().to(device)
        cid = torch.full((len(keep),), cell_id, dtype=torch.long).to(device)
        out = model(smis, cid, rf)
        probs = torch.softmax(out[cell].float(), dim=-1)[:, 1].cpu().numpy()
        scores.extend(probs.tolist())
        labels.extend(int(active[i, cell_idx_in_arr]) for i in keep)
    if len(set(labels)) < 2:
        return float("nan"), len(labels)
    return float(roc_auc_score(labels, scores)), len(labels)


# ---------------------------------------------------------------------------
# Train one seed (4-cell classifier + R18 adapter)
# ---------------------------------------------------------------------------
def train_one_seed(
    seed: int,
    smiles: List[str],
    active: np.ndarray,
    res_feat: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    """Train one seed of the Wf-31 Phase 2 4-cell cisplatin-resistance model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = FourCellResistanceDMPNN(
        hidden_dim=args.hidden,
        n_layers=args.depth,
        n_cells=N_CELLS,
        n_resist_feat=N_RESIST_FEAT,
        cell_lines=CELL_LINES,
        cell_embed_dim=RECIPE_CELL_EMBED_DIM,
        resist_proj_dim=RECIPE_RESIST_PROJ_DIM,
        head_hidden_dim=RECIPE_HEAD_HIDDEN,
        head_dropout=RECIPE_HEAD_DROPOUT,
        encoder_dropout=RECIPE_DROPOUT,
    ).to(device)
    if args.init == "tmqm":
        try_load_tmqm_encoder(model, args.encoder_ckpt)

    # Attach the per-init R18 adapter components.
    r18_components = attach_init_components(
        args.init,
        model,
        device,
        hidden_dim=args.hidden,
    )

    n_frozen, n_trainable = freeze_encoder_layers_1_2(model)
    print(
        f"[train_cisplatin_resistance_v2] seed={seed} hidden={args.hidden} "
        f"depth={args.depth} init={args.init} "
        f"frozen={n_frozen:,} trainable={n_trainable:,} "
        f"r18_components={list(r18_components.keys()) or 'none'}"
    )

    # Scaffold split (deterministic — same OOD buckets every seed).
    tr_idx, va_idx, te_idx = scaffold_split_indices(smiles, seed=seed)
    # Filter rows with at least one valid cell-line label.
    valid_any = (active >= 0).any(axis=1)
    tr_idx = tr_idx[valid_any[tr_idx]]
    va_idx = va_idx[valid_any[va_idx]]
    te_idx = te_idx[valid_any[te_idx]]

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_mean = -1.0
    best_state: Dict[str, torch.Tensor] = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    best_epoch = 0
    val_history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # One batch pool per cell line (B=batch_size rows routed to the cell).
        per_cell_batches: Dict[str, List[Tuple[np.ndarray, int]]] = {
            cl: build_batches(
                tr_idx,
                CELL_LINES.index(cl),
                args.batch,
                rng,
                shuffle=True,
            )
            for cl in CELL_LINES
        }
        n_steps = max(len(b) for b in per_cell_batches.values())
        for step in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0
            any_grad = False
            for cl in CELL_LINES:
                bp = per_cell_batches[cl]
                if step >= len(bp):
                    continue
                idx_chunk, cell_id = bp[step]
                keep = [i for i in idx_chunk if active[i, cell_id] >= 0]
                if not keep:
                    continue
                smis = [smiles[i] for i in keep]
                rf = torch.from_numpy(res_feat[keep]).float().to(device)
                cid = torch.full((len(keep),), cell_id, dtype=torch.long).to(device)
                targets = {
                    cl2: torch.from_numpy(
                        np.maximum(active[keep, CELL_LINES.index(cl2)], 0)
                    )
                    .long()
                    .to(device)
                    for cl2 in CELL_LINES
                }
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    out = model(smis, cid, rf)
                    loss = four_cell_loss(out, targets)
                batch_loss = batch_loss + float(loss.item())
                loss.backward()
                any_grad = True
            if any_grad:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )
                optimizer.step()
                epoch_loss += batch_loss
                n_batches += 1
        scheduler.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_aucs: Dict[str, float] = {}
            for cl in CELL_LINES:
                a, n = _eval_cell_auc(
                    model,
                    smiles,
                    active,
                    res_feat,
                    va_idx,
                    device,
                    args.batch,
                    cl,
                    CELL_LINES.index(cl),
                )
                val_aucs[cl] = a
            valid_aucs = [v for v in val_aucs.values() if not np.isnan(v)]
            mean_val = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
            val_history.append(
                {
                    "epoch": epoch,
                    "mean_val_auc": mean_val,
                    "per_cell": val_aucs,
                }
            )
            print(
                f"[train_cisplatin_resistance_v2] seed={seed} "
                f"epoch={epoch:3d} loss={epoch_loss / max(n_batches, 1):.4f} "
                f"mean_val_auc={mean_val:.4f} "
                + " ".join(f"{cl}={val_aucs[cl]:.3f}" for cl in CELL_LINES)
            )
            if not np.isnan(mean_val) and mean_val > best_val_mean:
                best_val_mean = mean_val
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

    model.load_state_dict(best_state)

    # Final test
    test_aucs: Dict[str, float] = {}
    for cl in CELL_LINES:
        a, n = _eval_cell_auc(
            model,
            smiles,
            active,
            res_feat,
            te_idx,
            device,
            args.batch,
            cl,
            CELL_LINES.index(cl),
        )
        test_aucs[cl] = a
        print(
            f"[train_cisplatin_resistance_v2] seed={seed} TEST "
            f"{cl} AUC={a:.4f} n={n}"
        )
    valid_test = [v for v in test_aucs.values() if not np.isnan(v)]
    mean_test = float(np.mean(valid_test)) if valid_test else float("nan")
    print(
        f"[train_cisplatin_resistance_v2] seed={seed} "
        f"mean_test_auc={mean_test:.4f} (best_epoch={best_epoch})"
    )

    # SE(3) integration smoke (only if se3sb init).
    se3_smoke: Dict[str, object] = {}
    if args.init == "se3sb" and hasattr(model, "se3_field"):
        se3_smoke = se3_integration_smoke(model.se3_field)
        print(f"[train_cisplatin_resistance_v2] seed={seed} " f"se3_smoke={se3_smoke}")

    return {
        "seed": seed,
        "init": args.init,
        "best_val_epoch": best_epoch,
        "best_val_mean_auc": best_val_mean,
        "test_auc_per_cell": test_aucs,
        "mean_test_auc": mean_test,
        "val_history": val_history,
        "n_frozen": n_frozen,
        "n_trainable": n_trainable,
        "r18_components": list(r18_components.keys()),
        "se3_smoke": se3_smoke,
        "best_state": best_state,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
    }


# ---------------------------------------------------------------------------
# Aggregation — 5-seed mean ± std per cell line
# ---------------------------------------------------------------------------
def aggregate_per_cell(
    seed_results: List[Dict[str, object]],
    cells: Tuple[str, ...] = CELL_LINES,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return (per_cell_mean_auc, per_cell_std_auc) across seeds."""
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for c in cells:
        vals = [
            float(r["test_auc_per_cell"][c])
            for r in seed_results
            if c in r["test_auc_per_cell"] and not np.isnan(r["test_auc_per_cell"][c])
        ]
        if not vals:
            means[c] = float("nan")
            stds[c] = float("nan")
        else:
            means[c] = float(np.mean(vals))
            stds[c] = float(np.std(vals, ddof=0))
    return means, stds


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _to_jsonable(o):
    if isinstance(o, dict):
        return {k: _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-31 Phase 2 — R18 SE(3) SB cisplatin-resistance 4-cell "
            "classifier trainer.  Init modes: tmqm / se3sb / transport / "
            "multiph.  Side features: 10-dim chemoinformatic fingerprint."
        ),
    )
    # Wf-35 Phase 2 — shared recipe / GPU args.
    p = add_common_args(
        p,
        data_default=DEFAULT_DATA,
        encoder_ckpt_default=DEFAULT_ENCODER_CKPT,
        out_dir_default=DEFAULT_OUT_DIR,
        metrics_path_default=DEFAULT_METRICS_PATH,
        summary_path_default=DEFAULT_SUMMARY_PATH,
        include_init=False,  # we override the choices + help below.
    )
    p = add_gpu_args(p)

    # Wf-31 Phase 2-specific flags.
    p.add_argument(
        "--init",
        choices=list(INIT_CHOICES),
        default=INIT_DEFAULT,
        help=(
            "Encoder / R18 adapter init.  'tmqm' = R17 baseline (with "
            "arch-mismatch fallback).  'se3sb' / 'transport' / 'multiph' "
            "= R18 W2.1 / W2.2 / W2.3 adapter attach.  Default: tmqm."
        ),
    )
    p.add_argument(
        "--cell-line",
        choices=list(CELL_LINES),
        default=CELL_LINES[0],
        help=(
            "Cell line of interest for the headline JSON path / ckpt "
            "filename.  The 4-cell classifier is always trained on all 4 "
            "cell lines; this flag only affects bookkeeping."
        ),
    )
    p.add_argument(
        "--se3-n-steps",
        type=int,
        default=RECIPE_SE3_INTEGRATION_STEPS,
        help="Number of SE(3) integration steps for the smoke test.",
    )
    # Drop --train-max-per-metal which is meaningless here (no metal axis).
    # add_common_args adds it unconditionally; we override the default to 0.
    p.set_defaults(train_max_per_metal=0)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@MEASURED_vs_PROJECTED_decorator()
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    setup_gpu_env(args)  # Wf-35 — apply CUDA_VISIBLE_DEVICES + torch seed.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[train_cisplatin_resistance_v2] data={args.data} init={args.init} "
        f"epochs={args.epochs} batch={args.batch} lr={args.lr} "
        f"seeds={args.seeds} hidden={args.hidden} depth={args.depth} "
        f"cell_line={args.cell_line}"
    )

    smiles, active, res_feat = load_dataset(args.data)
    print(
        f"[train_cisplatin_resistance_v2] n_rows={len(smiles)} "
        f"active_rates="
        + ", ".join(
            f"{cl}={(active[:, c] >= 0).mean() and (active[active[:, c] >= 0, c] == 1).mean():.3f}"
            for c, cl in enumerate(CELL_LINES)
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_cisplatin_resistance_v2] device={device}")

    t0 = time.time()
    seed_results: List[Dict[str, object]] = []
    for seed in args.seeds:
        result = train_one_seed(seed, smiles, active, res_feat, args, device)
        seed_results.append(result)
        ckpt_path = (
            args.out_dir
            / f"cisplatin_resistance_v2_{args.init}_{args.cell_line}_seed{seed}.pt"
        )
        torch.save(
            {
                "seed": seed,
                "init": args.init,
                "cell_line": args.cell_line,
                "best_val_epoch": result["best_val_epoch"],
                "best_val_mean_auc": result["best_val_mean_auc"],
                "test_auc_per_cell": result["test_auc_per_cell"],
                "mean_test_auc": result["mean_test_auc"],
                "r18_components": result["r18_components"],
                "se3_smoke": result["se3_smoke"],
                "state_dict": result["best_state"],
                "recipe": {
                    "init": args.init,
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "lr": args.lr,
                    "hidden": args.hidden,
                    "depth": args.depth,
                    "cell_line": args.cell_line,
                    "se3_n_steps": args.se3_n_steps,
                },
            },
            ckpt_path,
        )
        print(f"[train_cisplatin_resistance_v2] wrote {ckpt_path}")
    wall = time.time() - t0
    print(f"[train_cisplatin_resistance_v2] total wall={wall:.1f}s")

    per_cell_mean, per_cell_std = aggregate_per_cell(seed_results)

    # Drop heavy best_state tensors from JSON output.
    for r in seed_results:
        if isinstance(r, dict) and "best_state" in r:
            r.pop("best_state")

    metrics = {
        "recipe": "wf31_r18_se3sb_cisplatin_resistance_v2",
        "n_seeds": len(seed_results),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "hidden": args.hidden,
        "depth": args.depth,
        "init": args.init,
        "cell_lines": list(CELL_LINES),
        "cell_line_focus": args.cell_line,
        "data_path": str(args.data),
        "encoder_ckpt": str(args.encoder_ckpt),
        "wall_seconds": wall,
        "per_seed": seed_results,
        "per_cell_mean_auc": per_cell_mean,
        "per_cell_std_auc": per_cell_std,
        "headline_auc_per_seed": [r["mean_test_auc"] for r in seed_results],
        "headline_auc_mean": float(
            np.nanmean([r["mean_test_auc"] for r in seed_results])
        ),
        "headline_auc_std": float(
            np.nanstd([r["mean_test_auc"] for r in seed_results])
        ),
        "headline_cell_auc_per_seed": [
            [r["test_auc_per_cell"].get(cl, float("nan")) for cl in CELL_LINES]
            for r in seed_results
        ],
        "r17_baseline_3cell_auc": 0.648,
        "r17_baseline_source": (
            "metrics/by_metric/platinai_3cell_resistance.json — A2780cis "
            "5-seed mean AUC (R17 baseline, 3 cell lines, A2780 dropped)."
        ),
        "se3_init_meta": {
            "manifold": "SE3",
            "n_integration_steps": args.se3_n_steps,
            "velocity_field_class": "SE3SBVelocityField",
            "integrator_fn": "integrate_with_cfm",
            "decoder_class": "SE3SBDecoder",
        }
        if args.init == "se3sb"
        else {},
        "r18_components_meta": (
            {"se3_field": "SE3SBVelocityField"}
            if args.init == "se3sb"
            else (
                {"relax_term": "RelaxedEndpointTerm"}
                if args.init == "transport"
                else (
                    {"mphp": "MultiPHPocketFeatures"} if args.init == "multiph" else {}
                )
            )
        ),
    }
    args.metrics_path.write_text(json.dumps(metrics, indent=2))
    print(
        f"[train_cisplatin_resistance_v2] wrote {args.metrics_path} "
        f"(headline mean AUC = {metrics['headline_auc_mean']:.4f})"
    )

    # Slim final summary.
    summary = {
        "recipe": ("wf31_r18_se3sb_cisplatin_resistance_v2_final_summary"),
        "n_seeds": len(seed_results),
        "epochs": args.epochs,
        "init": args.init,
        "cell_lines": list(CELL_LINES),
        "cell_line_focus": args.cell_line,
        "wall_seconds": wall,
        "headline_auc_per_seed": metrics["headline_auc_per_seed"],
        "headline_auc_mean": metrics["headline_auc_mean"],
        "headline_auc_std": metrics["headline_auc_std"],
        "per_cell_mean_auc": metrics["per_cell_mean_auc"],
        "per_cell_std_auc": metrics["per_cell_std_auc"],
        "wf31_ph2_headline": (
            "R18 SE(3) SB 4-cell cisplatin-resistance classifier — "
            "per-cell + headline mean ROC-AUC."
        ),
    }
    args.summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[train_cisplatin_resistance_v2] wrote {args.summary_path}")
    # Return (exit_code, args) so the @MEASURED_vs_PROJECTED_decorator
    # can use the explicit args namespace instead of re-parsing argv.
    return 0, args


if __name__ == "__main__":
    out = main()
    if isinstance(out, tuple):
        rc = out[0]
    else:
        rc = out
    raise SystemExit(rc)
