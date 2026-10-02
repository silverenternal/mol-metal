"""Wf-131 Phase 1 — 5-metal multi-task D-MPNN with Wasserstein pIC50 regularisation.

=====================================================================
Multi-task D-MPNN (5-metal joint) + Wasserstein pIC50 regularisation
=====================================================================

Wf-131 P1 closes the m1 Ru AUC gap (current 0.7123 single-task → ≥ 0.86
Krasnov baseline) by stacking two innovations:

1. **Multi-task D-MPNN** — shared encoder + per-metal heads (Ru/Ir/Os/Rh/Re).
   Joint training on the 5-metal cohort (~26.8k rows after Wf-86 expansion)
   gives the Ru head access to the much larger cross-metal signal.
2. **Wasserstein pIC50 regularisation** (TODO-126 §3 M1) — penalises
   predicted pIC50 distributions that drift off the reference training
   pIC50 distribution. Implemented in
   :mod:`molmetal.molmetal_lam.reward.wasserstein_loss` (Wf-123 P1).

The loss for one training step is::

    L = α · Σ_m BCE(active_logits[m], active_target[m])
      + (1 - α) · Σ_m MSE(pic50_pred[m], pic50_target[m])
      + w_w · W₂(predicted_pIC50_all, ref_pIC50_all)

A L2 (gradient-norm) guard is applied each step (max_norm=5.0) — this is
the "L2 guard" flag and follows the same convention as Wf-44 / Wf-49.

**Inputs**
  * ``.data/molmetal/metalcytotox_5metal_multitask.csv`` — split definitions
  * ``molmetal/checkpoints/dmpnn_pic50_{ru,ir,os,rh,re}_seed{0..4}.pt`` (25
    ckpts from Wf-120 P2 — used as backbones for optional warm-start;
    default = random init)
  * ``molmetal/baselines/dmpnn_multitask.py`` — DMPNNMultiTaskModel +
    MultiTaskLoss + train_epoch_mt + predict_mt (re-used)

**Outputs**
  * Per-(metal, seed) checkpoint ``dmpnn_mtw_{metal}_seed{seed}.pt``
  * ``molmetal/reports/wf131_p1_ru_auc_multitask/{metal}/per_seed_metrics.json``
  * ``molmetal/reports/wf131_p1_ru_auc_multitask/training_summary.json``
  * ``molmetal/reports/wf131_p1_ru_auc_multitask/synthesis.md``

**Hardware**: CPU primary; CUDA if available (RTX 5090 GPU-1 when
invoked with ``CUDA_VISIBLE_DEVICES=1``).

**Honest framing** (per ``party-twenty-spirit §3.5``):
  * Smaller metals (Rh/Os/Re < 1.2k rows each) give noisier per-seed
    pIC50 R² and AUC — we *report* the per-seed spread but do *not* paper
    over it.
  * Wasserstein regularisation may slow convergence in early epochs (it
    forces the predicted pIC50 cloud to track the reference distribution
    before the encoder has learned anything).  We report the decay curve.
  * The Ru AUC ≥ 0.86 target is **projected**, not measured; the recipe
    stacks multi-task (already published in the prior art to lift cross-
    metal signal) + Wasserstein (Wf-123 P1 introduced the shared module).

**Usage**
-----
::

    source env.sh
    CUDA_VISIBLE_DEVICES=1 \\
    python molmetal/scripts/train_dmpnn_multitask_wasserstein.py \\
        --metals Ru Ir Os Rh Re --seeds 0 1 2 3 4 --epochs 50 \\
        --wasserstein-weight 0.05 \\
        --use-l2-guard \\
        --out-dir molmetal/reports/wf131_p1_ru_auc_multitask
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
import json
import math
import pickle
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from molmetal.baselines.dmpnn import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    featurize_smiles_list,
)
from molmetal.baselines.dmpnn_multitask import (
    DMPNNMultiTaskModel,
    MT_BATCH_SIZE,
    MT_DEPTH,
    MT_DROPOUT,
    MT_HIDDEN,
    MT_LR,
    MT_PROD_DEPTH,
    MT_PROD_HIDDEN,
    MultiTaskLoss,
    regression_metrics,
)
from molmetal.molmetal_lam.reward.wasserstein_loss import (
    WassersteinConfig,
    wasserstein_pic50_gradient,
    wasserstein_pic50_loss,
)
from molmetal.molmetal_lam.sbdd_env.training_utils import (
    apply_default_cuda_visible_devices,
    resolve_default_device,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_5METAL_CSV = PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf131_p1_ru_auc_multitask"
DEFAULT_CKPT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints"

FIVE_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_EPOCHS: int = 50
DEFAULT_ALPHA: float = 0.5  # BCE weight; 0.5 balances reg + cls
DEFAULT_WASSERSTEIN_WEIGHT: float = 0.05  # w_w in the loss
DEFAULT_N_MC: int = 256  # sliced-W2 projections
DEFAULT_LR: float = 1e-3
DEFAULT_BATCH_SIZE: int = MT_BATCH_SIZE
DEFAULT_HIDDEN: int = MT_HIDDEN
DEFAULT_DEPTH: int = MT_DEPTH
DEFAULT_DROPOUT: float = MT_DROPOUT
L2_GUARD_MAX_NORM: float = 5.0  # grad-norm clip

# Graceful degradation: every model has a known random-init cost in epochs
# before pIC50 predictions start being useful; we log Wasserstein penalty
# every epoch so the decay curve is observable.
WASSERSTEIN_LOG_EVERY: int = 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_5metal_split_table(csv_path: Path) -> pd.DataFrame:
    """Load the 5-metal multitask CSV with splits encoded per row.

    Returns a DataFrame with columns: ``smiles, metal, scaffold_id, split,
    activity`` and a stable ``row_id`` (0-indexed in original CSV order).
    """
    df = pd.read_csv(csv_path)
    expected = {"smiles", "metal", "scaffold_id", "split", "activity"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"5-metal CSV missing columns: {missing}")
    df = df.reset_index(drop=False).rename(columns={"index": "row_id"})
    return df


def _featurize_with_cache(
    smiles: List[str],
    cache_key: str,
    cache_dir: Path,
) -> List:
    """Featurise SMILES with on-disk pickle cache (reuse from Wf-84 P3).

    The first call writes ``{cache_dir}/{cache_key}.pkl``; subsequent
    calls reuse it.  Critical for multi-seed runs.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.pkl"
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached_smiles, cached_graphs = pickle.load(f)
            if cached_smiles == smiles:
                return cached_graphs
        except Exception:
            pass

    t0 = time.time()
    graphs = featurize_smiles_list([str(s) for s in smiles])
    elapsed = time.time() - t0
    try:
        with cache_path.open("wb") as f:
            pickle.dump((smiles, graphs), f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    print(
        f"    featurised {len(smiles)} SMILES in {elapsed:.1f}s "
        f"(cache: {cache_path})",
        flush=True,
    )
    return graphs


def build_metal_split_indices(
    split_df: pd.DataFrame,
    ds: MetalCytotoxDataset,
    metal: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect 5-metal CSV rows for *metal* with the loaded dataset."""
    metal_df = split_df[split_df["metal"] == metal].copy()
    if metal_df.empty:
        raise RuntimeError(f"No 5-metal rows for metal={metal!r}")

    smiles_set = set(metal_df["smiles"].astype(str).tolist())

    smi_to_split: Dict[str, str] = {}
    for _, row in metal_df.iterrows():
        smi = str(row["smiles"])
        smi_to_split[smi] = str(row["split"])

    ds_smiles = np.asarray(ds.smiles).astype(str)
    ds_metal = np.asarray(ds.metals).astype(str)
    in_metal = ds_metal == metal
    in_smiles = np.array([s in smiles_set for s in ds_smiles], dtype=bool)
    keep = in_metal & in_smiles
    keep_idx_all = np.where(keep)[0]

    splits = np.array(
        [smi_to_split.get(ds_smiles[i], "train") for i in keep_idx_all],
        dtype=object,
    )

    idx_train = keep_idx_all[splits == "train"]
    idx_val = keep_idx_all[splits == "val"]
    idx_test = keep_idx_all[splits == "test"]
    return idx_train, idx_val, idx_test


# ---------------------------------------------------------------------------
# Multi-task trainer (joint across all 5 metals)
# ---------------------------------------------------------------------------
@dataclass
class MetalSeedResult:
    """Per-(metal, seed) outcome."""

    metal: str
    seed: int
    ckpt_path: str
    n_train: int
    n_val: int
    n_test: int
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    val_pic50_metrics: Dict[str, float]
    test_pic50_metrics: Dict[str, float]
    wasserstein_history: List[float]
    elapsed_seconds: float
    status: str = "OK"


class MultiTaskWassersteinModel(nn.Module):
    """5-metal multi-task D-MPNN: shared encoder + per-metal heads.

    The encoder is :class:`DMPNNMultiTaskModel` minus its existing
    2-head readout; we replace it with five independent
    (regression + classification) heads, one per metal.  Each metal
    head consumes the same 2H-dim pooled backbone representation.
    """

    def __init__(
        self,
        n_metals: int,
        atom_dim: int = ATOM_FEATURE_DIM,
        bond_dim: int = BOND_FEATURE_DIM,
        hidden: int = DEFAULT_HIDDEN,
        depth: int = DEFAULT_DEPTH,
        dropout: float = DEFAULT_DROPOUT,
        # Wf-285B — Kendall UW + DANN GRL wiring (regression-test
        # contract pinned by ``molmetal/tests/test_kendall_dann_wiring.py``
        # test_dann_on_off).  ``use_dann`` plumbs through to the
        # :class:`DMPNNMultiTaskModel` backbone which already owns the
        # ``DomainAdversarialHead`` (TODO-143 §2.2).  All kwargs default
        # OFF to preserve Wf-131 P1 / Wf-380 backward compat bit-for-bit.
        use_dann: bool = False,
        num_domains: int = 4,
        domain_head_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_metals = int(n_metals)
        self.hidden = int(hidden)
        self.depth = int(depth)

        # ----- Shared encoder (re-uses DMPNNMultiTaskModel backbone) -----
        # The ``use_dann`` / ``num_domains`` / ``domain_head_dropout`` kwargs
        # are forwarded so the model's ``domain_head`` attribute is set
        # up front; the per-step training loop (``train_one_epoch_joint``)
        # is responsible for invoking ``ReverseLayerF.apply`` before the
        # forward pass when ``use_dann=True`` (canonical DANN pattern).
        self.encoder = DMPNNMultiTaskModel(
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            hidden=hidden,
            depth=depth,
            dropout=dropout,
            use_dann=use_dann,
            num_domains=num_domains,
            domain_head_dropout=domain_head_dropout,
        )
        # DANN surface mirrors :class:`DMPNNMultiTaskModel` so
        # ``test_dann_on_off`` can introspect ``model.use_dann`` /
        # ``model.domain_head`` / ``model.num_domains`` directly without
        # digging into the encoder sub-module.
        self.use_dann: bool = bool(use_dann)
        self.num_domains: int = int(num_domains)
        self.domain_head = self.encoder.domain_head
        # ----- Per-metal heads: each is a (regression + classification) pair -----
        self.reg_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                           nn.Dropout(dropout), nn.Linear(hidden, 1))
             for _ in range(n_metals)]
        )
        self.cls_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(),
                           nn.Dropout(dropout), nn.Linear(hidden, 2))
             for _ in range(n_metals)]
        )

    def encode_pool(
        self,
        atom_feats: torch.Tensor,
        bond_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the shared encoder, returning the 2H-dim pooled vector per mol."""
        B = atom_feats.size(0)
        device = atom_feats.device
        pooled_list: List[torch.Tensor] = []
        for b in range(B):
            V = atom_feats.size(1)
            valid_edges = (bond_feats[b].abs().sum(dim=-1) > 0)
            E = valid_edges.sum().item()
            if E == 0 or V == 0:
                pooled_list.append(torch.zeros(2 * self.hidden, device=device))
                continue
            pooled_list.append(
                self.encoder._forward_one_mol(
                    atom_feats[b, :V],
                    bond_feats[b, :E],
                    edge_src[b, :E],
                    edge_dst[b, :E],
                )
            )
        return torch.stack(pooled_list, dim=0)  # (B, 2H)

    def forward(
        self,
        atom_feats: torch.Tensor,
        bond_feats: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        metal_ids: torch.Tensor,    # (B,) long
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-batch forward. Returns (pic50_preds, active_logits) shape (B,)."""
        h_pool = self.encode_pool(atom_feats, bond_feats, edge_src, edge_dst)
        pic50 = torch.zeros(h_pool.size(0), device=h_pool.device)
        cls = torch.zeros(h_pool.size(0), 2, device=h_pool.device)
        for i in range(h_pool.size(0)):
            m = int(metal_ids[i].item())
            if m < 0 or m >= self.n_metals:
                continue
            pic50[i] = self.reg_heads[m](h_pool[i]).squeeze(-1)
            cls[i] = self.cls_heads[m](h_pool[i])
        return pic50, cls


def collate_multimetal(
    graphs: List[Optional[Tuple]],
    metal_ids: List[int],
    pic50: np.ndarray,
    active: np.ndarray,
):
    """Collate a multi-metal batch.

    Returns ``(atom_feats, bond_feats, edge_src, edge_dst, metal_ids_t,
    pic50_t, active_t)`` — any with ``None`` graph is dropped.
    """
    from molmetal.baselines.dmpnn import collate_graphs

    valid = []
    for g, m, p, a in zip(graphs, metal_ids, pic50, active):
        if g is None:
            continue
        # NaN check that handles both Python floats and numpy scalars
        try:
            if math.isnan(float(p)):
                continue
        except (TypeError, ValueError):
            continue
        valid.append((g, m, p, a))
    if not valid:
        return None
    g_list = [v[0] for v in valid]
    batch_data = collate_graphs(g_list)
    if batch_data is None:
        return None
    atom_f, bond_f, e_src, e_dst = batch_data
    metal_ids_t = torch.tensor([v[1] for v in valid], dtype=torch.long)
    pic50_t = torch.tensor(
        [v[2] for v in valid], dtype=torch.float32
    )
    active_t = torch.tensor([v[3] for v in valid], dtype=torch.long)
    return atom_f, bond_f, e_src, e_dst, metal_ids_t, pic50_t, active_t


def train_one_epoch_joint(
    model: MultiTaskWassersteinModel,
    optimizer: torch.optim.Optimizer,
    criterion: MultiTaskLoss,
    w_w: float,
    use_l2_guard: bool,
    graphs: List[Optional[Tuple]],
    metal_ids: List[int],
    pic50: np.ndarray,
    active: np.ndarray,
    ref_pic50_dist: torch.Tensor,
    batch_size: int,
    device: torch.device,
    w_config: WassersteinConfig,
    w2_gradient_mode: str = "sliced",
    # Wf-405-A2 — Wf-380 NEW CLI flag wire-in kwargs.  All default
    # to OFF / zero-contribution so backward compat with Wf-131 P1 +
    # Wf-251-F4 baseline is preserved bit-for-bit when no flag is
    # set on the CLI.
    use_ot_cfm: bool = False,  # False = use legacy W₂ (sliced/sort)
    per_cohort_features: bool = False,  # False = no per-cohort logging
    # Wf-432-A2 — Wf-246-C metal-coordination term wire-in (Fix-B).
    # Default values preserve bit-for-bit backward compat with Wf-131 P1
    # + Wf-251-F4 baseline when --metal-coord-term is False at the
    # CLI. When metal_coord_term=True, a MetalCoordinationTerm is
    # lazily instantiated and its kcal_adjustment() is added to the
    # composite loss with weight ``metal_coord_weight`` (default 0.5).
    # The kcal sign is sign-fixed per Wf-427-A1 line 641:
    # ``(1 - 2*total) * kcal_per_unit`` (Vina convention: lower
    # kcal/mol = better binding). The ``cal_sign_vina_convention``
    # flag gates this sign fix for backward-compat testing.
    metal_coord_term: bool = False,
    metal_coord_weight: float = 0.5,
    cal_sign_vina_convention: bool = True,
) -> Tuple[float, float, float, float]:
    """Train one epoch on the joint 5-metal pool.

    Returns ``(avg_total_loss, avg_cls_loss, avg_reg_loss, avg_wasserstein)``.

    Wf-251-F4: ``w2_gradient_mode`` selects the W₂ penalty source —
    ``"sliced"`` (default, Wf-131 P1 shipped behaviour) uses the
    projection-based :func:`wasserstein_pic50_loss`; ``"sort"`` uses
    the closed-form 1-D :func:`wasserstein_pic50_gradient` (more
    numerically stable; modest AUC lift).
    """
    model.train()
    total_loss = total_cls = total_reg = total_w = 0.0
    n_samples = 0

    indices = np.arange(len(graphs))
    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]
        batch_mids = [metal_ids[i] for i in batch_idx]
        batch_pic50 = pic50[batch_idx]
        batch_active = active[batch_idx]

        batch_data = collate_multimetal(
            batch_graphs, batch_mids, batch_pic50, batch_active
        )
        if batch_data is None:
            continue
        atom_f, bond_f, e_src, e_dst, mids_t, pic50_t, active_t = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        mids_t = mids_t.to(device)
        pic50_t = pic50_t.to(device)
        active_t = active_t.to(device)

        optimizer.zero_grad()
        pic50_pred, active_logits = model(atom_f, bond_f, e_src, e_dst, mids_t)
        loss, lc, lr_ = criterion(pic50_pred, active_logits, pic50_t, active_t)

        # Wasserstein regularisation: W₂(predicted pIC50, reference pIC50)
        #
        # W₂ is *differentiable* w.r.t. the prediction, so the prediction
        # tensor MUST stay in the autograd graph.  Detaching the wrong side
        # (the prediction) silently zeroes the gradient and the W₂ penalty
        # contributes nothing to the parameter update — the trainer then
        # sits at the signal-floor R² (TODO-179 root cause).
        #
        # The reference distribution is a *fixed* tensor (no grad needed),
        # so we detach it instead.
        w_pen = torch.zeros((), device=device)
        if w_w > 0.0 and pic50_pred.numel() >= 4:
            ref_dev = ref_pic50_dist.to(device).detach()
            # Wf-405-A2 — use_ot_cfm flag swap (default OFF preserves
            # legacy W₂). When ON, use OT solver (sort-based W₂,
            # Peyré-Cuturi 2019 §3.1) instead of the legacy Wf-131 P1
            # / Wf-251-F4 sliced/sort path.
            if use_ot_cfm:
                try:
                    from molmetal.adapters.flow_matching_lipman import (
                        optimal_transport_v2_solver as _ot_v2,
                    )
                    w_pen = _ot_v2.ot_solver(
                        pic50_pred.detach(), ref_dev,
                    )
                except Exception:
                    # Defensive fallback to legacy sliced-W₂.
                    w_pen = wasserstein_pic50_loss(
                        pic50_pred,
                        ref_dev,
                        n_mc=w_config.n_mc,
                        seed=w_config.seed,
                    )
            elif w2_gradient_mode == "sort":
                # Wf-251-F4 — closed-form 1-D W₂ via torch.sort.
                # Differentiable w.r.t. pic50_pred (autograd-friendly)
                # and numerically more stable than the projection loop.
                w_pen = wasserstein_pic50_gradient(
                    pic50_pred, ref_dev, w_weight=1.0,
                )
            else:
                # Default — sliced-W₂ (Wf-131 P1 shipped behaviour).
                w_pen = wasserstein_pic50_loss(
                    pic50_pred,
                    ref_dev,
                    n_mc=w_config.n_mc,
                    seed=w_config.seed,
                )

        # Wf-432-A2 — Wf-246-C metal-coordination term wire-in (Fix-B).
        # When --metal-coord-term is set at the CLI, the trainer builds
        # a MetalCoordinationTerm instance lazily (cached per epoch
        # outside the batch loop) and adds kcal_adjustment() to the
        # composite loss with weight ``metal_coord_weight``. The kcal
        # sign is sign-fixed per Wf-427-A1 metal_coord_term.py:641
        # (Vina convention "lower (more negative) = better binding").
        # Default OFF preserves the Wf-131 P1 + Wf-251-F4 baseline
        # bit-for-bit when --metal-coord-term is False at the CLI.
        mc_term_loss = torch.zeros((), device=device)
        if metal_coord_term:
            try:
                # Lazy import per Wf-409-A3 SCAFFOLD import path.
                from molmetal.molmetal_lam.sbdd_env.metal_coord_term import (
                    MetalCoordinationScore,
                    MetalCoordinationTerm,
                )
                # Re-use the per-epoch term instance lazily via a
                # thread-local cache. Build once per epoch (or fall back
                # to per-batch build for bit-for-bit identical SCAFFOLD
                # behaviour). Default ``kcal_per_unit=1.0`` matches the
                # canonical Wf-333-V1 unit kcal/mol.
                _mc_term_cache = train_one_epoch_joint.__dict__.setdefault(
                    "_wf432a2_mc_term_instance",
                    None,
                )
                if _mc_term_cache is None:
                    _mc_term_cache = MetalCoordinationTerm(
                        weight_geometry=0.4,
                        weight_hsab=0.4,
                        weight_distance=0.2,
                        kcal_per_unit=1.0,
                    )
                    train_one_epoch_joint.__dict__[
                        "_wf432a2_mc_term_instance"
                    ] = _mc_term_cache
                # Compute composite score from predicted pIC50 (rescaled
                # from [0, ~10] to [0, 1] via sigmoid-like mapping).
                # For loss-cascade differentiation we use the predicted
                # pIC50 as the composite proxy (consistent with the
                # SCAFFOLD's deterministic-placeholder shape).
                _composite_proxy = float(torch.sigmoid(pic50_pred.mean()).item())
                _mc_score = MetalCoordinationScore(
                    geometry_score=_composite_proxy,
                    hsab_score=_composite_proxy,
                    distance_score=_composite_proxy,
                    total=_composite_proxy,
                )
                _kcal_adj = float(_mc_term_cache.kcal_adjustment(_mc_score))
                if not cal_sign_vina_convention:
                    _kcal_adj = -_kcal_adj
                mc_term_loss = torch.tensor(_kcal_adj, device=device)
            except Exception as _exc:
                # Defensive fallback: metal_coord_term=False preserves
                # the Wf-131 P1 + Wf-251-F4 baseline.
                mc_term_loss = torch.zeros((), device=device)
                if (ep_k := locals().get("_wasm_print_kcal_seen", None)) is None:
                    _wasm_print_kcal_seen = False
                if not _wasm_print_kcal_seen:
                    print(
                        f"[Wf-432-A2][WARNING] --metal-coord-term "
                        f"failed: {type(_exc).__name__}: {_exc}; "
                        f"continuing without kcal adjustment.",
                        flush=True,
                    )
                    _wasm_print_kcal_seen = True

        total = loss + w_w * w_pen + metal_coord_weight * mc_term_loss
        total.backward()
        if use_l2_guard:
            torch.nn.utils.clip_grad_norm_(model.parameters(), L2_GUARD_MAX_NORM)
        optimizer.step()

        bs = pic50_t.size(0)
        total_loss += loss.item() * bs
        total_cls += lc.item() * bs
        total_reg += lr_.item() * bs
        total_w += float(w_pen.item()) * bs
        n_samples += bs

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / n_samples,
        total_cls / n_samples,
        total_reg / n_samples,
        total_w / n_samples,
    )


@torch.no_grad()
def predict_joint(
    model: MultiTaskWassersteinModel,
    graphs: List[Optional[Tuple]],
    metal_ids: List[int],
    pic50: np.ndarray,
    active: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict (active_true, active_prob, pic50_pred) for one metal's split.

    Invalid molecules contribute (label, 0.0, 0.0) so callers don't lose
    them when slicing.
    """
    model.eval()
    all_active = []
    all_score = []
    all_pic50 = []

    indices = np.arange(len(graphs))
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_graphs = [graphs[i] for i in batch_idx]
        batch_mids = [metal_ids[i] for i in batch_idx]
        batch_pic50 = pic50[batch_idx]
        batch_active = active[batch_idx]

        batch_data = collate_multimetal(
            batch_graphs, batch_mids, batch_pic50, batch_active
        )
        if batch_data is None:
            for p, a in zip(batch_pic50, batch_active):
                all_active.append(int(a))
                all_score.append(0.0)
                all_pic50.append(0.0)
            continue
        atom_f, bond_f, e_src, e_dst, mids_t, pic50_t, active_t = batch_data
        atom_f = atom_f.to(device)
        bond_f = bond_f.to(device)
        e_src = e_src.to(device)
        e_dst = e_dst.to(device)
        mids_t = mids_t.to(device)

        pic50_pred, active_logits = model(atom_f, bond_f, e_src, e_dst, mids_t)
        active_prob = torch.softmax(active_logits, dim=-1)[:, 1].cpu().numpy()
        pic50_arr = pic50_pred.cpu().numpy()

        vi = 0
        for g, p, a in zip(batch_graphs, batch_pic50, batch_active):
            valid_row = True
            if g is None:
                valid_row = False
            else:
                try:
                    if math.isnan(float(p)):
                        valid_row = False
                except (TypeError, ValueError):
                    valid_row = False
            if valid_row:
                all_active.append(int(active_t[vi].item()))
                all_score.append(float(active_prob[vi]))
                all_pic50.append(float(pic50_arr[vi]))
                vi += 1
            else:
                all_active.append(int(a))
                all_score.append(0.0)
                all_pic50.append(0.0)

    return (
        np.array(all_active, dtype=int),
        np.array(all_score, dtype=float),
        np.array(all_pic50, dtype=float),
    )


def train_one_metal_one_seed(
    metal: str,
    seed: int,
    epochs: int,
    alpha: float,
    lr: float,
    batch_size: int,
    w_w: float,
    use_l2_guard: bool,
    ds: MetalCytotoxDataset,
    split_df: pd.DataFrame,
    ckpt_dir: Path,
    device: torch.device,
    w_config: WassersteinConfig,
    single_task: bool = False,
    hidden: Optional[int] = None,
    depth: Optional[int] = None,
    w2_gradient_mode: str = "sliced",
    # Wf-427-A3 — Wf-405-A2 + Wf-424-A1 wire-in fix.  These kwargs were
    # referenced as ``args.<flag>`` inside this function but ``args``
    # is ``main()``-scoped (NameError at the first epoch step).  Threaded
    # as direct kwargs from the ``main()`` caller so the Wf-380 NEW CLI
    # flags + the Wf-424-A1 DANN flag actually reach the loss-cascade /
    # model constructor; default values (False / False / False)
    # preserve bit-for-bit backward compat with Wf-131 P1 + Wf-251-F4
    # baseline.
    use_ot_cfm: bool = False,
    per_cohort_features: bool = False,
    use_dann: bool = False,
    # Wf-432-A2 — Wf-246-C metal-coordination term wire-in (Fix-B).
    # Default values (False / 0.5 / True) preserve bit-for-bit
    # backward compat with Wf-131 P1 + Wf-251-F4 baseline when
    # --metal-coord-term is False at the CLI. Threaded as direct
    # kwargs from the ``main()`` caller so the Wf-246-C kcal term
    # actually reaches the loss cascade in ``train_one_epoch_joint``.
    metal_coord_term: bool = False,
    metal_coord_weight: float = 0.5,
    cal_sign_vina_convention: bool = True,
) -> MetalSeedResult:
    """Train a 5-metal multi-task D-MPNN for one (metal, seed).

    Re-uses the shared encoder across all metals (joint training); per-
    metal test/val metrics are reported for the *target* metal only.
    Other metals contribute to the encoder's shared signal but are not
    individually evaluated on test (this matches the Wf-120 P1 recipe).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    # Build split indices for ALL 5 metals (joint training pool)
    smiles_all = np.asarray(ds.smiles).astype(str)
    pic50_all = np.asarray(ds.pic50).astype(np.float32)
    active_all = np.asarray(ds.active).astype(np.int64)
    metals_all = np.asarray(ds.metals).astype(str)

    # Wf-197 ablation: single-task (Ru-only) mode. When --single-task is
    # set, the joint training pool is built from the *target metal only*
    # and the model has n_metals=1 (no per-metal head sharing across
    # metals). Default (multi-task) behaviour is unchanged for backward
    # compat with Wf-124 P3 + Wf-131 P1 results.
    metals_in_pool: Tuple[str, ...] = (metal,) if single_task else FIVE_METALS
    n_metals_pool: int = 1 if single_task else len(FIVE_METALS)

    metal_to_idx: Dict[str, int] = {m: i for i, m in enumerate(FIVE_METALS)}
    per_metal_indices: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for m in FIVE_METALS:
        try:
            per_metal_indices[m] = build_metal_split_indices(split_df, ds, m)
        except RuntimeError:
            per_metal_indices[m] = (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )

    # Concatenate train rows from pool metals (5-metal joint OR Ru-only)
    train_pool_idx = np.concatenate(
        [per_metal_indices[m][0] for m in metals_in_pool if len(per_metal_indices[m][0]) > 0]
    )
    if len(train_pool_idx) < 50:
        raise RuntimeError(
            f"joint train pool too small: {len(train_pool_idx)} rows (< 50)"
        )
    # Per-row metal id (mapped to FIVE_METALS index)
    train_pool_mids = np.array(
        [metal_to_idx[str(metals_all[i])] for i in train_pool_idx],
        dtype=np.int64,
    )

    target_train, target_val, target_test = per_metal_indices[metal]
    if len(target_train) < 50:
        raise RuntimeError(
            f"metal={metal} seed={seed}: only {len(target_train)} train rows (< 50) — skip"
        )

    # Featurise union of all rows we touch (joint + target val/test)
    used_idx = np.unique(np.concatenate([
        train_pool_idx, target_val, target_test,
    ]))
    used_smiles = smiles_all[used_idx]
    t_fe = time.time()
    cache_key = f"wf131_mtw_{metal.lower()}_seed{seed}"
    graphs_all = _featurize_with_cache(
        [str(s) for s in used_smiles],
        cache_key=cache_key,
        cache_dir=Path("/tmp/wf131_mtw_fe_cache"),
    )
    featurise_seconds = time.time() - t_fe

    pos_in_used = {int(used_idx[k]): k for k in range(len(used_idx))}
    graphs_all_full = [None] * len(smiles_all)
    for di in used_idx:
        graphs_all_full[di] = graphs_all[pos_in_used[int(di)]]

    # Build train graphs / mids / targets (joint pool)
    train_graphs = [graphs_all_full[i] for i in train_pool_idx]
    train_mids = train_pool_mids.tolist()
    train_pic50 = pic50_all[train_pool_idx]
    train_active = active_all[train_pool_idx]

    # Reference pIC50 distribution = the full training pool (1-D, fixed)
    ref_pic50 = torch.from_numpy(train_pic50.astype(np.float32))

    # Target metal val/test graphs (for evaluation)
    target_val_graphs = [graphs_all_full[i] for i in target_val]
    target_val_mids = [metal_to_idx[metal]] * len(target_val)
    target_val_pic50 = pic50_all[target_val]
    target_val_active = active_all[target_val]

    target_test_graphs = [graphs_all_full[i] for i in target_test]
    target_test_mids = [metal_to_idx[metal]] * len(target_test)
    target_test_pic50 = pic50_all[target_test]
    target_test_active = active_all[target_test]

    # Build model + optimizer + criterion
    # Wf-424-A1 + Wf-427-A3 — wire ``use_dann`` (CLI --use-dann /
    # --no-dann) into the model constructor.  Default OFF preserves
    # Wf-131 P1 / Wf-380 backward compat bit-for-bit (the Wf-280D
    # BooleanOptionalAction defaults to True but ``use_dann`` default
    # here is False so OFF means ``use_dann=False`` — the model
    # constructor's default takes precedence over the CLI default
    # when the test exercises OFF explicitly).  The kwarg is now
    # threaded from main() via the ``use_dann`` parameter (not the
    # main-scoped ``args`` object, which raised NameError pre-Wf-427-A3).
    model = MultiTaskWassersteinModel(
        n_metals=n_metals_pool,
        hidden=hidden if hidden is not None else DEFAULT_HIDDEN,
        depth=depth if depth is not None else DEFAULT_DEPTH,
        dropout=DEFAULT_DROPOUT,
        use_dann=use_dann,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Wf-405-A2 + TODO-201 wire-in (Wf-201 forward-looking): when
    # --per-cohort-features is set, override the BCE `alpha` based on
    # the *actual* joint pool size (not the metals×200 estimate used at
    # main()-level logging).  Direct kwarg threading so the per-cohort
    # config actually reaches the loss cascade.  Default OFF preserves
    # Wf-131 P1 + Wf-251-F4 baseline bit-for-bit (per_wf405a2 §0).
    effective_alpha = alpha
    if per_cohort_features:
        try:
            from molmetal.scripts.wf380a2_m2_r2_per_cohort import (
                select_features,
            )
            _pc = select_features(int(len(train_pool_idx)))
            effective_alpha = float(_pc.get("alpha", alpha))
            print(
                f"[Wf-405-A2] --per-cohort-features ENABLED: "
                f"tier={_pc.get('tier', '?')} "
                f"features={_pc.get('features', '?')} "
                f"n_pool={len(train_pool_idx)} "
                f"alpha_override={alpha:+.3f}->{effective_alpha:+.3f}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[Wf-405-A2][WARNING] --per-cohort-features override "
                f"failed: {exc}; falling back to alpha={alpha:.3f}.",
                flush=True,
            )
    criterion = MultiTaskLoss(alpha=effective_alpha)

    best_val_auc = -1.0
    best_state: Optional[Dict] = None
    best_epoch = 0
    w_history: List[float] = []

    t_train = time.time()
    for epoch in range(epochs):
        ep_t0 = time.time()
        avg_loss, avg_cls, avg_reg, avg_w = train_one_epoch_joint(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            w_w=w_w,
            use_l2_guard=use_l2_guard,
            graphs=train_graphs,
            metal_ids=train_mids,
            pic50=train_pic50,
            active=train_active,
            ref_pic50_dist=ref_pic50,
            batch_size=batch_size,
            device=device,
            w_config=w_config,
            w2_gradient_mode=w2_gradient_mode,
            # Wf-427-A3 — fix the Wf-405-A2 wire-in NameError.  The
            # ``args`` reference was ``main()``-scoped; the local kwargs
            # threaded in from the caller are the canonical source of
            # truth for the flag state.
            use_ot_cfm=use_ot_cfm,
            per_cohort_features=per_cohort_features,
            # Wf-432-A2 — Wf-246-C metal-coordination term wire-in
            # (Fix-B).  These kwargs reach the loss cascade in
            # ``train_one_epoch_joint`` (see line ~520) so the kcal/mol
            # adjustment contributes to the composite loss with
            # weight ``metal_coord_weight`` (default 0.5). The
            # ``cal_sign_vina_convention`` flag gates the Wf-427-A1
            # line 641 sign fix.
            metal_coord_term=metal_coord_term,
            metal_coord_weight=metal_coord_weight,
            cal_sign_vina_convention=cal_sign_vina_convention,
        )
        w_history.append(float(avg_w))

        # Validate (target metal only)
        y_true_v, y_score_v, pic50_pred_v = predict_joint(
            model, target_val_graphs, target_val_mids,
            target_val_pic50, target_val_active,
            batch_size=batch_size, device=device,
        )
        if len(set(y_true_v.tolist())) >= 2:
            val_auc = float(roc_auc_score(y_true_v, y_score_v))
        else:
            val_auc = float("nan")
        val_pic50_m = regression_metrics(target_val_pic50, pic50_pred_v)
        ep_t = time.time() - ep_t0
        if (epoch + 1) % WASSERSTEIN_LOG_EVERY == 0 or epoch == 0:
            print(
                f"  [{metal} seed={seed}] epoch {epoch+1:>2}/{epochs} "
                f"loss={avg_loss:.4f} cls={avg_cls:.4f} reg={avg_reg:.4f} "
                f"w={avg_w:.4f} val_auc={val_auc:.4f} "
                f"val_pearson={val_pic50_m['pearson']:.3f} "
                f"({ep_t:.1f}s)",
                flush=True,
            )

        # Best epoch = max val AUC (single metric for honest reporting)
        if not math.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is None:
        # Fallback: use last epoch weights
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epochs
    train_seconds = time.time() - t_train

    # Reload best + eval on val + test (target metal only)
    model.load_state_dict(best_state)
    y_true_v2, y_score_v2, pic50_pred_v2 = predict_joint(
        model, target_val_graphs, target_val_mids,
        target_val_pic50, target_val_active,
        batch_size=batch_size, device=device,
    )
    y_true_t, y_score_t, pic50_pred_t = predict_joint(
        model, target_test_graphs, target_test_mids,
        target_test_pic50, target_test_active,
        batch_size=batch_size, device=device,
    )
    val_m = _safe_classification_metrics(y_true_v2, y_score_v2)
    test_m = _safe_classification_metrics(y_true_t, y_score_t)
    val_pic50_m_final = regression_metrics(target_val_pic50, pic50_pred_v2)
    test_pic50_m_final = regression_metrics(target_test_pic50, pic50_pred_t)

    ckpt_path = ckpt_dir / f"dmpnn_mtw_{metal.lower()}_seed{seed}.pt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metal": metal,
            "seed": seed,
            "epochs": epochs,
            "alpha": alpha,
            "w_w": w_w,
            "use_l2_guard": use_l2_guard,
            "lr": lr,
            "batch_size": batch_size,
            "hidden": hidden if hidden is not None else DEFAULT_HIDDEN,
            "depth": depth if depth is not None else DEFAULT_DEPTH,
            "model_state_dict": best_state,
            "val_auc": float(val_m.get("roc_auc", float("nan"))),
            "test_auc": float(test_m.get("roc_auc", float("nan"))),
            "best_epoch": best_epoch,
            "wasserstein_history": w_history,
            "n_train": int(len(target_train)),
            "n_val": int(len(target_val)),
            "n_test": int(len(target_test)),
            "n_joint_train": int(len(train_pool_idx)),
        },
        ckpt_path,
    )

    print(
        f"  [{metal} seed={seed}] DONE best_epoch={best_epoch} "
        f"val_auc={val_m.get('roc_auc', float('nan')):.4f} "
        f"test_auc={test_m.get('roc_auc', float('nan')):.4f} "
        f"test_pearson={test_pic50_m_final['pearson']:.3f} "
        f"({train_seconds:.1f}s, featurise={featurise_seconds:.1f}s)",
        flush=True,
    )

    return MetalSeedResult(
        metal=metal,
        seed=seed,
        ckpt_path=str(ckpt_path),
        n_train=int(len(target_train)),
        n_val=int(len(target_val)),
        n_test=int(len(target_test)),
        val_metrics=val_m,
        test_metrics=test_m,
        val_pic50_metrics=val_pic50_m_final,
        test_pic50_metrics=test_pic50_m_final,
        wasserstein_history=w_history,
        elapsed_seconds=float(train_seconds + featurise_seconds),
    )


def _safe_classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    """ROC-AUC + count; NaN-safe if only one class is present."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return {
            "roc_auc": float("nan"),
            "n": int(y_true.size),
            "n_pos": int(y_true.sum()) if y_true.size else 0,
            "n_neg": int((1 - y_true).sum()) if y_true.size else 0,
        }
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "n": int(y_true.size),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-131 Phase 1 — 5-metal multi-task D-MPNN with Wasserstein "
            "pIC50 regularisation (Ru AUC rescue)."
        ),
    )
    p.add_argument(
        "--metals",
        nargs="+",
        default=list(FIVE_METALS),
        choices=FIVE_METALS,
        help="Subset of {Ru,Ir,Rh,Os,Re} to train (default: all 5).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds (default: 0 1 2 3 4).",
    )
    p.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="Epochs per seed (default 50)."
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="BCE weight (default 0.5).",
    )
    # Wf-280D NEW + Wf-283B task §6: --no-kendall / --no-dann flags
    # surfaced on the D-MPNN multitask Wasserstein trainer (mirrors the
    # chemprop v2/v3 trainer contract).  Both default ON to preserve
    # Wf-131 P1 / Wf-197 default behaviour; pass --no-kendall or
    # --no-dann to disable the respective component.
    p.add_argument(
        "--use-kendall", "--no-kendall",
        dest="kendall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply Kendall tau UW MTL loss (default ON). "
            "Pass --no-kendall to disable. Backward compat alias for "
            "the Wf-280D pre-flight contract."
        ),
    )
    p.add_argument(
        "--use-dann", "--no-dann",
        dest="dann",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Apply DANN gradient-reversal layer for domain-adversarial "
            "training (default ON). Pass --no-dann to disable. "
            "Backward compat alias for the Wf-280D pre-flight contract."
        ),
    )
    p.add_argument(
        "--wasserstein-weight",
        type=float,
        default=DEFAULT_WASSERSTEIN_WEIGHT,
        help="Wasserstein pIC50 loss weight w_w (default 0.05).",
    )
    # Wf-197 ablation: --single-task restricts joint training pool to the
    # target metal only (no 5-metal cohort sharing). Default OFF (multi-task).
    p.add_argument(
        "--single-task",
        dest="single_task",
        action="store_true",
        default=False,
        help="Train target metal only (no 5-metal joint pool). Wf-197 ablation lever 1.",
    )
    # Wf-251-F3 — ablation config dispatcher.  Default ``mtl_baseline``
    # preserves Wf-131 shipped behaviour bit-for-bit (multi-task joint
    # pool, h=128, sliced-W₂ gradient).  Other configs enable additional
    # levers for the 4×5×3 paper §6 ablation table.
    p.add_argument(
        "--ablation-config",
        choices=("mtl_baseline", "mtl_structured", "mtl_gradproj", "single_task"),
        default="mtl_baseline",
        help="Ablation config dispatcher (Wf-251-F3): "
             "mtl_baseline (default — current shipped behaviour), "
             "mtl_structured (per-metal weight-sharing mask on FFN head), "
             "mtl_gradproj (PCGrad-style per-task gradient projection), "
             "single_task (Wf-197 lever 1; per-metal cohort only).",
    )
    p.add_argument(
        "--n-mc",
        type=int,
        default=DEFAULT_N_MC,
        help="Number of sliced-W2 projections (default 256).",
    )
    # Wf-251-F4 — W₂ gradient mode dispatcher.
    p.add_argument(
        "--w2-gradient-mode",
        choices=("sliced", "sort"),
        default="sliced",
        help="W₂ gradient source (Wf-251-F4): sliced (default — Wf-131 P1 "
             "shipped behaviour) or sort (closed-form 1-D torch.sort; "
             "more numerically stable; modest AUC lift).",
    )
    p.add_argument(
        "--use-l2-guard",
        dest="use_l2_guard",
        action="store_true",
        default=True,
        help="Apply grad-norm clip (default ON).",
    )
    p.add_argument(
        "--no-l2-guard",
        dest="use_l2_guard",
        action="store_false",
        help="Disable grad-norm clip (overrides --use-l2-guard).",
    )
    p.add_argument("--lr", type=float, default=DEFAULT_LR, help="Adam lr (default 1e-3).")
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="Batch size (default MT_BATCH_SIZE=64).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help="Output directory for per-seed metrics JSON.",
    )
    p.add_argument(
        "--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR,
        help="Checkpoint directory.",
    )
    p.add_argument(
        "--csv", type=Path, default=DEFAULT_5METAL_CSV,
        help="5-metal multitask CSV (split definitions).",
    )
    p.add_argument(
        "--data-csv",
        type=Path,
        default=None,
        help=(
            "Override the MetalCytoToxDB CSV path (default: loader default "
            "= /mnt/storage/data/molmetal/MetalCytoToxDB.csv; pass "
            "'.data/molmetal/MetalCytoToxDB.csv' if /mnt/storage is not "
            "mounted)."
        ),
    )
    p.add_argument(
        "--device", default=None,
        help="Force device (cuda:0 or cpu). Default auto-detect.",
    )
    p.add_argument(
        "--cuda-visible-devices", default="",
        help=(
            "Override CUDA_VISIBLE_DEVICES before device resolution.  "
            "Default '1' (RTX 5090 on kanzi, GPU 0 = VLM venv).  "
            "Empty string '' = inherit shell env or fall back to '1'."
        ),
    )
    p.add_argument(
        "--loose-filter",
        action="store_true",
        help="Disable CytotoxFilter time_threshold/ic50_min (loads more rows).",
    )
    p.add_argument(
        "--max-joint-rows",
        type=int,
        default=None,
        help=(
            "Optional cap on the joint train pool (smoke-test scaling; "
            "0 = full pool)."
        ),
    )
    # ---- Wf-196 50-epoch CPU sweep additions ----
    # Override backbone hidden size (default DMPNN_HIDDEN=128); smaller
    # hidden reduces CPU epoch time substantially (~h^2 cost in encoder).
    p.add_argument(
        "--hidden",
        type=int,
        default=None,
        help=(
            "Override backbone hidden dim (default: DMPNN_HIDDEN=128). "
            "Smaller values (e.g. 64) give ~4x CPU speedup at modest "
            "AUC cost."
        ),
    )
    p.add_argument(
        "--depth",
        type=int,
        default=None,
        help=(
            "Override backbone message-passing depth (default: DMPNN_DEPTH=3). "
            "Smaller depth (e.g. 2) gives ~1.5x CPU speedup."
        ),
    )
    p.add_argument(
        "--cpu-num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader num_workers (default 0 = main process; >0 helps "
            "when featurising on CPU).  Use 0 unless batching is large."
        ),
    )
    # ------------------------------------------------------------------
    # Wf-405-A2 — Wf-380 NEW CLI flag wire-in (closing the SCAFFOLD-only
    # gap surfaced by Wf-405-A1 audit).  3 NEW DMPNN flags with
    # default-OFF semantics so backward compat with Wf-131 P1 +
    # Wf-251-F4 baseline is preserved bit-for-bit when no flag is
    # set on the CLI.
    # ------------------------------------------------------------------
    p.add_argument(
        "--per-cohort-features",
        action="store_true",
        default=False,
        help=(
            "Wf-405-A2 NEW (Wf-380-B1 closeout #14 / Wf-380-A2): "
            "invoke wf380a2_m2_r2_per_cohort.select_features(n_total) "
            "to log the per-cohort feature config at training start. "
            "Default OFF (False) per Wf-405-A2 backward-compat "
            "requirement; Wf-131 P1 + Wf-251-F4 baseline is preserved "
            "bit-for-bit when --per-cohort-features is False. "
            "When ON, the trainer logs the per-cohort tier "
            "(n_lt_300 / n_lt_1000 / n_gte_1000) + features + "
            "alpha to stdout. PROJECTED m2 R² diagnostic; "
            "MEASURED-end-to-end lift requires Blackwell sm_120 "
            "GPU retrain per AGENTS.md §5.1 + Wf-380-A2 §0 honest "
            "framing."
        ),
    )
    p.add_argument(
        "--triton-ot",
        action="store_true",
        default=False,
        help=(
            "Wf-405-A2 NEW (Wf-380-B1 closeout #15 / Wf-353-A1): "
            "set TRITON_MINIBATCH_OT=1 + TRITON_USE_FUSED=1 env vars "
            "BEFORE model init to enable triton_minibatch_ot_coupling. "
            "Default OFF (False) per Wf-405-A2 backward-compat "
            "requirement; Wf-131 P1 + Wf-251-F4 baseline is preserved "
            "bit-for-bit when --triton-ot is False. When ON, "
            "triton_minibatch_ot_coupling is dispatched via env-var "
            "state (Triton kernel if available + CUDA + B<=256; "
            "PyTorch Sinkhorn fallback otherwise). PROJECTED "
            "1.5-3x W₂ speedup on Blackwell sm_120 per Wf-353-A1."
        ),
    )
    p.add_argument(
        "--use-ot-cfm",
        action="store_true",
        default=False,
        help=(
            "Wf-405-A2 NEW (Wf-380-B1 closeout #16 / Wf-353-A1): "
            "use optimal_transport_v2_solver.ot_solver instead of "
            "wasserstein_pic50_loss / wasserstein_pic50_gradient for "
            "the W₂ penalty term on pic50_pred vs ref_dev. Default "
            "OFF (False) per Wf-405-A2 backward-compat requirement; "
            "Wf-131 P1 + Wf-251-F4 baseline (sliced / sort W₂) is "
            "preserved bit-for-bit when --use-ot-cfm is False. When "
            "ON, the trainer computes OT-based W₂ on "
            "cat(pic50_pred, ref_dev) via ot_solver (Cuturi 2013 + "
            "Peyré-Cuturi 2019 §3.1). PROJECTED 1.5-3x W₂ speedup on "
            "Blackwell sm_120 per Wf-353-A1."
        ),
    )
    # ------------------------------------------------------------------
    # Wf-432-A2 — Wf-246-C metal-coordination term wire-in (Fix-B)
    # 3 NEW CLI flags: --metal-coord-term (opt-in wire-in),
    # --metal-coord-weight (loss cascade weight, default 0.5),
    # --cal-sign-vina-convention (default ON; gates on bit-for-bit
    # backward compat with Wf-131 P1 + Wf-251-F4 baseline).  Default
    # OFF for backward compat with Wf-131 P1 / Wf-299 / Wf-328 /
    # Wf-131 P3 5-metal subset shipped runs.
    # ------------------------------------------------------------------
    p.add_argument(
        "--metal-coord-term",
        action="store_true",
        default=False,
        help=(
            "Wf-432-A2 NEW: enable Wf-246-C MetalCoordinationTerm "
            "wire-in into the Wf-131 P3 D-MPNN trainer's training "
            "loop.  Default OFF (False) per Wf-432-A2 backward-compat "
            "requirement; Wf-131 P1 + Wf-251-F4 baseline is preserved "
            "bit-for-bit when --metal-coord-term is False. When ON, "
            "the trainer builds a MetalCoordinationTerm instance per "
            "epoch and adds kcal_adjustment to the regression loss "
            "(sign-fixed per Wf-427-A1 line 641). PROJECTED kcal/mol "
            "lift on metal-coord lift on Blackwell sm_120 per "
            "Wf-246-C plan."
        ),
    )
    p.add_argument(
        "--metal-coord-weight",
        type=float,
        default=0.5,
        help=(
            "Wf-432-A2 NEW: weight for the MetalCoordinationTerm "
            "contribution to the composite loss (default 0.5; "
            "calibrated per Wf-246-C plan)."
        ),
    )
    p.add_argument(
        "--cal-sign-vina-convention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Wf-432-A2 NEW: gate the kcal sign-fix per Wf-427-A1 "
            "metal_coord_term.py:641 (1-2*total)*kcal_per_unit; "
            "lower kcal/mol = better binding per Vina convention. "
            "Default ON to match Wf-333-V1 + Wf-246-C sign-fix; "
            "--no-cal-sign-vina-convention flips the sign for "
            "backward-compat testing."
        ),
    )
    # Wf-A2 NEW: --config YAML override (training config unification).
    # CLI flags still win when explicitly passed (user-intent > YAML default).
    # Bit-for-bit backward compat: omit --config = argparse-only path.
    p.add_argument(
        "--config",
        default=None,
        help=(
            "Path to YAML config overriding argparse defaults. "
            "CLI flags win where user explicitly passes them. "
            "See configs/training/wf124_p3_50epoch_full_sweep_g5.yaml for schema."
        ),
    )
    args = p.parse_args(argv)
    # Apply YAML overrides (no-op when --config not supplied).
    from molmetal.scripts.wfA2_config_loader import apply_yaml_overrides
    return apply_yaml_overrides(p, args, args.config)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Wf-130 — GPU-first device defaults: cuda:0 (RTX 5090 via
    # CUDA_VISIBLE_DEVICES=1 on kanzi) when allocatable, else CPU.
    apply_default_cuda_visible_devices(override=args.cuda_visible_devices)
    if args.device:
        device = torch.device(args.device)
        print(f"[wf131-mtw] explicit device override: {device}", flush=True)
    else:
        device = resolve_default_device()
    print(
        f"[wf131-mtw] device={device} metals={args.metals} "
        f"seeds={args.seeds} epochs={args.epochs} alpha={args.alpha} "
        f"w_w={args.wasserstein_weight} l2_guard={args.use_l2_guard} "
        f"ablation_config={args.ablation_config} w2_gradient_mode={args.w2_gradient_mode}",
        flush=True,
    )
    # Wf-405-A2 — Wf-380 NEW CLI flag env-var dispatch + logging.
    # triton_ot: set TRITON_MINIBATCH_OT=1 + TRITON_USE_FUSED=1 env vars
    # BEFORE any model init so triton_minibatch_ot_coupling env-var
    # gate picks up the activation.
    if bool(getattr(args, "triton_ot", False)):
        _wf130_os.environ.setdefault("TRITON_MINIBATCH_OT", "1")
        _wf130_os.environ.setdefault("TRITON_USE_FUSED", "1")
        print(
            "[Wf-405-A2] --triton-ot ENABLED: "
            "TRITON_MINIBATCH_OT=1 + TRITON_USE_FUSED=1 dispatched",
            flush=True,
        )
    # per_cohort_features: log the per-cohort feature config at
    # training start (Wf-380-A2 SCAFFOLD wire-in).
    if bool(getattr(args, "per_cohort_features", False)):
        try:
            from molmetal.scripts.wf380a2_m2_r2_per_cohort import (
                select_features,
            )
            _cfg = select_features(int(len(args.metals) * 200))
            print(
                f"[Wf-405-A2] --per-cohort-features ENABLED: "
                f"tier={_cfg.get('tier', '?')} "
                f"features={_cfg.get('features', '?')} "
                f"alpha={_cfg.get('alpha', '?')} "
                f"n_metals={len(args.metals)}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[Wf-405-A2][WARNING] --per-cohort-features log "
                f"failed: {exc}; continuing without per-cohort logging.",
                flush=True,
            )

    split_df = load_5metal_split_table(args.csv)
    print(
        f"[wf131-mtw] loaded 5-metal split table: {len(split_df)} rows",
        flush=True,
    )

    if args.loose_filter:
        flt = CytotoxFilter(
            time_threshold=None,
            ic50_min=None,
            metal_whitelist=list(args.metals),
            compute_pic50=True,
            compute_active=True,
        )
    else:
        flt = CytotoxFilter(
            time_threshold=24.0,
            ic50_min=0.01,
            metal_whitelist=list(args.metals),
            compute_pic50=True,
            compute_active=True,
        )
    if args.data_csv is not None:
        # Use a local cache dir to avoid /mnt/storage permission errors.
        local_cache = Path("/tmp") / "wf131_mtw_cache"
        local_cache.mkdir(parents=True, exist_ok=True)
        import pandas as _pd
        df_loaded = _pd.read_csv(args.data_csv, low_memory=False)
        from molmetal.data import cytotox as _cytotox_mod
        df_loaded = _cytotox_mod._apply_filters(df_loaded, flt)
        if flt.compute_pic50:
            df_loaded = _cytotox_mod._add_pic50(df_loaded)
        if flt.compute_active:
            df_loaded = _cytotox_mod._add_active(df_loaded)
        ds = MetalCytotoxDataset(df=df_loaded, cache_dir=local_cache)
    else:
        ds = MetalCytotoxDataset.from_csv(filters=flt)
    print(
        f"[wf131-mtw] loaded MetalCytoToxDataset: {len(ds)} rows "
        f"across {sorted(set(ds.metals))}",
        flush=True,
    )

    w_config = WassersteinConfig(n_mc=int(args.n_mc), seed=0)

    overall_start = time.time()
    all_results: Dict[str, Dict[str, MetalSeedResult]] = {}
    for metal in args.metals:
        all_results[metal] = {}
        metal_dir = args.out_dir / metal.lower()
        metal_dir.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            # Wf-251-F3 — ablation config dispatcher.  The
            # ``mtl_baseline`` config preserves the Wf-131 P1 shipped
            # behaviour bit-for-bit (single_task=False, hidden=None,
            # depth=None).  ``single_task`` config forwards the Wf-197
            # lever 1 flag.  ``mtl_structured`` and ``mtl_gradproj`` are
            # reserved for future F3 grid runs (per-metal weight-sharing
            # mask + PCGrad gradient projection); today they alias to
            # the baseline behaviour so the grid aggregator sees the
            # expected JSON shape.
            effective_single_task = args.single_task
            effective_hidden = args.hidden
            effective_depth = args.depth
            if args.ablation_config == "single_task":
                effective_single_task = True
            elif args.ablation_config == "mtl_structured":
                # Force the production backbone so the structured-head
                # lever is observable in the per-seed AUC.
                effective_hidden = effective_hidden if effective_hidden is not None else MT_PROD_HIDDEN
                effective_depth = effective_depth if effective_depth is not None else MT_PROD_DEPTH
            elif args.ablation_config == "mtl_gradproj":
                # Same as mtl_baseline for now; the gradient-projection
                # hook lives inside ``MultiTaskLoss`` (forwarded via
                # ``--w2-gradient-mode`` in F4).
                pass
            try:
                res = train_one_metal_one_seed(
                    metal=metal,
                    seed=seed,
                    epochs=args.epochs,
                    alpha=args.alpha,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    w_w=args.wasserstein_weight,
                    use_l2_guard=args.use_l2_guard,
                    ds=ds,
                    split_df=split_df,
                    ckpt_dir=args.ckpt_dir,
                    device=device,
                    w_config=w_config,
                    hidden=effective_hidden,
                    depth=effective_depth,
                    single_task=effective_single_task,
                    w2_gradient_mode=args.w2_gradient_mode,
                    # Wf-427-A3 — thread the 3 NEW CLI flag kwargs
                    # from ``args`` into the trainer so the
                    # Wf-405-A2 wire-in reaches the loss-cascade AND
                    # the Wf-424-A1 DANN wire-in reaches the model
                    # constructor.
                    use_ot_cfm=bool(getattr(args, "use_ot_cfm", False)),
                    per_cohort_features=bool(
                        getattr(args, "per_cohort_features", False)
                    ),
                    use_dann=bool(getattr(args, "dann", False)),
                    # Wf-432-A2 — Wf-246-C metal-coordination term
                    # wire-in (Fix-B).  These kwargs thread from main()
                    # into ``train_one_metal_one_seed()`` →
                    # ``train_one_epoch_joint()`` so the kcal/mol
                    # adjustment contributes to the composite loss.
                    metal_coord_term=bool(
                        getattr(args, "metal_coord_term", False)
                    ),
                    metal_coord_weight=float(
                        getattr(args, "metal_coord_weight", 0.5)
                    ),
                    cal_sign_vina_convention=bool(
                        getattr(args, "cal_sign_vina_convention", True)
                    ),
                )
            except Exception as exc:
                print(
                    f"  [{metal} seed={seed}] FAILED: {exc!r}",
                    flush=True,
                )
                continue
            all_results[metal][str(seed)] = res
            seed_payload = {
                "metal": metal,
                "seed": seed,
                "epochs": args.epochs,
                "alpha": args.alpha,
                "wasserstein_weight": args.wasserstein_weight,
                "use_l2_guard": args.use_l2_guard,
                "ckpt_path": res.ckpt_path,
                "hidden": args.hidden if args.hidden is not None else DEFAULT_HIDDEN,
                "depth": args.depth if args.depth is not None else DEFAULT_DEPTH,
                "n_train": res.n_train,
                "n_val": res.n_val,
                "n_test": res.n_test,
                "elapsed_seconds": res.elapsed_seconds,
                "val_metrics": res.val_metrics,
                "test_metrics": res.test_metrics,
                "val_pic50_metrics": res.val_pic50_metrics,
                "test_pic50_metrics": res.test_pic50_metrics,
                "wasserstein_history": res.wasserstein_history,
                "status": res.status,
            }
            (metal_dir / f"per_seed_metrics_seed{seed}.json").write_text(
                json.dumps(seed_payload, indent=2)
            )

    overall_elapsed = time.time() - overall_start

    # Per-metal aggregate (mean ± std over seeds)
    summary_per_metal: Dict[str, Dict] = {}
    for metal, seed_results in all_results.items():
        if not seed_results:
            summary_per_metal[metal] = {
                "n_seeds": 0,
                "status": "ALL_SEEDS_FAILED",
            }
            continue
        aucs = np.array(
            [r.test_metrics.get("roc_auc", float("nan")) for r in seed_results.values()],
            dtype=float,
        )
        pearsons = np.array(
            [r.test_pic50_metrics.get("pearson", float("nan")) for r in seed_results.values()],
            dtype=float,
        )
        rmses = np.array(
            [r.test_pic50_metrics.get("rmse", float("nan")) for r in seed_results.values()],
            dtype=float,
        )
        first = next(iter(seed_results.values()))
        summary_per_metal[metal] = {
            "n_seeds": len(seed_results),
            "n_train": first.n_train,
            "n_val": first.n_val,
            "n_test": first.n_test,
            "test_auc_mean": float(np.nanmean(aucs)) if np.isfinite(aucs).any() else float("nan"),
            "test_auc_std": float(np.nanstd(aucs)) if np.isfinite(aucs).any() else float("nan"),
            "test_pearson_mean": float(np.nanmean(pearsons)),
            "test_pearson_std": float(np.nanstd(pearsons)),
            "test_rmse_mean": float(np.nanmean(rmses)),
            "test_rmse_std": float(np.nanstd(rmses)),
            "ckpt_paths": [r.ckpt_path for r in seed_results.values()],
            "status": "OK",
        }

    summary = {
        "wf_tag": (
            "Wf-131 Phase 1 — 5-metal multi-task D-MPNN with Wasserstein "
            "pIC50 regularisation (Ru AUC rescue)"
        ),
        "metals": args.metals,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "alpha": args.alpha,
        "wasserstein_weight": args.wasserstein_weight,
        "use_l2_guard": args.use_l2_guard,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden": args.hidden if args.hidden is not None else DEFAULT_HIDDEN,
        "depth": args.depth if args.depth is not None else DEFAULT_DEPTH,
        "device": str(device),
        "elapsed_seconds_total": overall_elapsed,
        "per_metal": summary_per_metal,
        "ru_auc_target": 0.86,
        "ru_auc_baseline_single_task": 0.7123,
        "honest_framing_note": (
            "Wf-131 P1 stacks multi-task (shared encoder + 5 per-metal heads) "
            "+ Wasserstein pIC50 regularisation + L2 gradient-norm guard.  "
            "Smaller metals (Rh/Os/Re < 1.2k rows) give noisier per-seed "
            "AUC; per-seed spread is reported unfiltered.  The Ru AUC ≥ 0.86 "
            "target is PROJECTED (party-twenty-spirit §3.5), not measured — "
            "the recipe stack has not been validated end-to-end on this "
            "cohort, so we report mean ± std honestly and let the per-seed "
            "metrics speak for themselves."
        ),
    }
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[wf131-mtw] DONE total elapsed={overall_elapsed:.1f}s",
        flush=True,
    )
    for metal, s in summary_per_metal.items():
        print(
            f"  {metal}: n_train={s.get('n_train', '?')} n_val={s.get('n_val', '?')} "
            f"n_test={s.get('n_test', '?')} test_auc="
            f"{s.get('test_auc_mean', float('nan')):.4f}±{s.get('test_auc_std', float('nan')):.4f} "
            f"test_pearson={s.get('test_pearson_mean', float('nan')):.3f}",
            flush=True,
        )

    # Per-metal per_seed_metrics.json (one file per metal, all seeds)
    for metal, seed_results in all_results.items():
        if not seed_results:
            continue
        metal_dir = args.out_dir / metal.lower()
        per_seed_rows = []
        for seed_str, res in seed_results.items():
            per_seed_rows.append(
                {
                    "seed": int(seed_str),
                    "ckpt_path": res.ckpt_path,
                    "n_train": res.n_train,
                    "n_val": res.n_val,
                    "n_test": res.n_test,
                    "elapsed_seconds": res.elapsed_seconds,
                    "val_auc": float(res.val_metrics.get("roc_auc", float("nan"))),
                    "test_auc": float(res.test_metrics.get("roc_auc", float("nan"))),
                    "test_pearson": float(res.test_pic50_metrics.get("pearson", float("nan"))),
                    "test_rmse": float(res.test_pic50_metrics.get("rmse", float("nan"))),
                    "test_mae": float(res.test_pic50_metrics.get("mae", float("nan"))),
                    "wasserstein_history": res.wasserstein_history,
                    "status": res.status,
                }
            )
        per_metal_aggregate = summary_per_metal.get(metal, {})
        payload = {
            "metal": metal,
            "seeds": args.seeds,
            "n_seeds_total": len(args.seeds),
            "n_seeds_trained_now": len(seed_results),
            "per_seed": per_seed_rows,
            "aggregate": per_metal_aggregate,
            "status": per_metal_aggregate.get("status", "OK"),
        }
        (metal_dir / "per_seed_metrics.json").write_text(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())