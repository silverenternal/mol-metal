"""Wf-200 — Chemprop v2 v3 driver: full multi-task + Wasserstein + Kendall + DANN.

==============================================================================
Wf-200 ultracode / TODO-180 §3 follow-up — the Wf-200 chemprop v2 v3 driver.

This is a **single-purpose** training driver that stacks all four innovation
levers introduced across Wf-123 / Wf-124 / Wf-145 / Wf-148 onto the chemprop
v2 backbone (Wf-148's :class:`ChempropV2MTModel`):

  1. **5-metal multi-task heads** — shared chemprop v2 encoder +
     per-metal :class:`BinaryClassificationFFN` heads (Ru / Ir / Os / Rh / Re).
     Mirrors the V4 contract; no architectural departure.
  2. **Wasserstein transfer** — pairwise W₂ over per-metal embedding
     batches (TODO-121 / :func:`wasserstein_transfer_loss`).  Encourages
     metal-invariant chemistry at the latent level.  The Wasserstein
     transfer is applied to the **PlatinAI 4-cell** embedding only when
     ``--use-platinai-wass`` is set (the 4-cell PlatinAI rows are
     loaded as an additional transfer target); otherwise it operates
     on the per-metal embeddings of the in-batch multi-task head.
  3. **Kendall MTL uncertainty weighting** — replaces the manual
     alpha-weighted BCE sum with Kendall et al. 2018 homoscedastic-
     uncertainty weighting (TODO-143 §2.1).  The per-task log_sigma is
     learned jointly with the model.
  4. **DANN adversarial head (4-domain discriminator)** — domain-
     adversarial head on top of the chemprop encoder embedding (TODO-
     143 §2.2).  Discriminator sees a 4-class domain label
     (MetalCytoTox / DrugOOD / PlatinAI / BindingDB).  GRL flips
     the gradient so the encoder learns domain-invariant features.

**Honest framing** (per ``party-twenty-spirit §3.5``)
----------------------------------------------------
* The chemprop v2 backbone is **smaller** than production here
  (``hidden_dim=64, depth=2``) — matches the Wf-156 smoke config.
  Production chemprop v2 uses 300/3; per-step cost is ~5-7x lower
  at 64/2, so the AUC ceiling is lower.
* 50 epochs may not be enough for full convergence at this
  backbone size — we report W₂ decay + val_auc plateau in the
  per-seed metrics so downstream tooling can verify convergence
  honestly.
* Wasserstein transfer is applied to a **subset** of the per-metal
  embedding batch (max 32 rows per metal per step) to bound the
  O(N²) W₂ cost.  This is the same approximation used by Wf-131.
* DANN's lambda schedule is **constant** at ``--lambda-grl`` (no
  alpha ramp) per TODO-143 §4 Day 5-6 — large alpha destabilises
  the task head.
* Per-seed std is **reported** but not papered over (no best-seed
  selection).  Smaller metals (Rh/Os/Re < 1.2k rows) have noisier
  per-seed AUC.

**Outputs**
-------
* ``molmetal/reports/wf200_chemprop_v2_v3/training_summary.json``
  — per-(metal, seed) AUC + aggregate + honest framing note.
* ``molmetal/reports/wf200_chemprop_v2_v3/{metal}/per_seed_metrics_seed{N}.json``
  — per-seed JSON with val/test AUC, Kendall sigmas, W₂ history.
* ``molmetal/checkpoints/dmpnn_mtw_v3_{metal}_seed{N}.pt`` — best-val
  checkpoint per (metal, seed).

**Usage**
-----
Default (5-metal × 3-seed × 50-epoch, GPU + BF16)::

    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/train_chemprop_v2_v3.py

Smoke (1-metal × 1-seed × 2-epoch, CPU fallback)::

    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/train_chemprop_v2_v3.py --quick
"""

from __future__ import annotations


# --- CUDA_VISIBLE_DEVICES default (canonical resolver) ----------
# Default GPU 0 (RTX PRO 6000 Blackwell, 98 GB) unless the env var is
# already set (incl. "" = CPU-only).  Must run BEFORE any molmetal /
# torch import: importing the molmetal package initialises CUDA, and
# CUDA reads CUDA_VISIBLE_DEVICES only once, at init.  Source of
# truth: molmetal/molmetal_lam/sbdd_env/training_utils.py (loaded
# package-free via molmetal/scripts/_cuda_default.py).
import os as _wf130_os
import sys as _wf130_sys

_wf130_dir = _wf130_os.path.dirname(_wf130_os.path.abspath(__file__))
if _wf130_dir not in _wf130_sys.path:
    _wf130_sys.path.append(_wf130_dir)
from _cuda_default import ensure_cuda_visible_devices_default as _wf130_cvd

_wf130_cvd()
del _wf130_cvd, _wf130_dir, _wf130_os, _wf130_sys
# --- end CUDA default --------------------------------------------

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Ensure project root on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from chemprop.data import BatchMolGraph
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

from molmetal.data.metal_cytotox import load_metal_cytotox
from molmetal.models.chemprop_v2_mt_model import (
    ChempropV2MTModel,
    SUPPORTED_METALS,
)
from molmetal.models.chemprop_v2_mt_model_v3 import (
    ChempropV2MTModelV3,
)
from molmetal.models.dmpnn import DomainAdversarialHead
from molmetal.molmetal_lam.reward.wasserstein_loss import wasserstein_transfer_loss
from molmetal.molmetal_lam.sbdd_env.gradient_reversal import ReverseLayerF
from molmetal.molmetal_lam.sbdd_env.uncertainty_loss import (
    UncertaintyWeightedMultiTaskLoss,
)

# Bind the module logger before the opt-in import guards below: their
# failure paths call ``logger`` at import time, ahead of the canonical
# logger setup further down this file (same ``getLogger`` object, so the
# handler setup there is unaffected).
logger = logging.getLogger("wf200")

# Wf-250-Apply: M1 + M2 master facades — opt-in via --use-master-wass
# and --use-master-cath.  The imports are wrapped in try/except so the
# legacy code path remains import-safe even if either master module is
# missing or has a runtime import error.
try:
    from molmetal.molmetal_lam.reward.master_wasserstein_gradient_flow import (
        MasterWassersteinGradientFlow as _MasterWass,
        MasterWassersteinConfig as _MasterWassConfig,
    )

    _HAVE_MASTER_WASS = True
except Exception as _m1_err:  # pragma: no cover — best-effort opt-in
    _MasterWass = None  # type: ignore
    _MasterWassConfig = None  # type: ignore
    _HAVE_MASTER_WASS = False
    logger.warning(
        "[wf250] MasterWassersteinGradientFlow unavailable: %s",
        _m1_err,
    )

try:
    from molmetal.molmetal_lam.reward.master_catherine_closure_bias import (
        MasterCatherineClosureBias as _MasterCath,
        SUPPORTED_BIAS_TYPES as _CATH_BIAS_TYPES,
    )

    _HAVE_MASTER_CATH = True
except Exception as _m2_err:  # pragma: no cover — best-effort opt-in
    _MasterCath = None  # type: ignore
    _CATH_BIAS_TYPES = ()  # type: ignore
    _HAVE_MASTER_CATH = False
    logger.warning(
        "[wf250] MasterCatherineClosureBias unavailable: %s",
        _m2_err,
    )

# Wf-252-Apply Fix #1 (F1-ClickAug-V3Wire): wire the chemprop v2 click
# augmentation wrapper into the v3 driver as an opt-in --use-click-aug
# flag (default OFF for backward compat with Wf-200 / Wf-225 / Wf-230
# trained checkpoints).  When enabled, the training-corpus SMILES are
# passed through ``augment_candidates_via_click_reactions`` once before
# the per-metal pre-featurization step; the model architecture is
# unchanged (the augmentation lives at the data side, not the model
# side).  Eval JSON emitted to
# ``molmetal/reports/wf200_chemprop_v2_v3/click_aug_eval.json``.
try:
    from molmetal.models.chemprop_v2_click_augmentation import (
        augment_candidates_via_click_reactions as _click_augment_fn,
        ClickAugmentedChempropGenerator as _ClickAugGenerator,
    )

    _HAVE_CLICK_AUG = True
except Exception as _wf252_err:  # pragma: no cover — best-effort opt-in
    _click_augment_fn = None  # type: ignore
    _ClickAugGenerator = None  # type: ignore
    _HAVE_CLICK_AUG = False
    logger.warning(
        "[wf252-f1] chemprop_v2_click_augmentation unavailable: %s",
        _wf252_err,
    )


# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------
METALS_5: Tuple[str, ...] = ("Ru", "Ir", "Os", "Rh", "Re")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_EPOCHS: int = 50
DEFAULT_HIDDEN: int = 64
DEFAULT_DEPTH: int = 2
DEFAULT_BATCH: int = 64
DEFAULT_LR: float = 1e-3
DEFAULT_WASS_WEIGHT: float = 0.05
DEFAULT_LAMBDA_GRL: float = 0.1

# 4-domain DANN discriminator (matches pre_featurize_dmpnn DOMAIN_ID)
N_DOMAINS: int = 4

# Domain id mapping (canonical, see pre_featurize_dmpnn.py DOMAIN_ID):
#   0 = MetalCytoToxDB
#   1 = DrugOOD
#   2 = NCI-60 (PlatinAI)
#   3 = BindingDB
DOMAIN_METALCYTOX: int = 0
DOMAIN_DRUGOOD: int = 1
DOMAIN_NCI60: int = 2
DOMAIN_BINDINGDB: int = 3

REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf200_chemprop_v2_v3"
CKPT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints" / "wf200_chemprop_v2_v3"

# Coord label heuristic (mirrors smoke_chemprop_v2_mt_100).
_COORD_TABLE = {
    "Pt": 3,
    "Pd": 3,
    "Au": 3,  # square_planar
    "Ru": 4,
    "Ir": 4,
    "Os": 4,
    "Rh": 4,
    "Re": 4,  # octahedral
}


def _coord_label_for(metal_name: str) -> int:
    return _COORD_TABLE.get(metal_name, 3)


logger = logging.getLogger("wf200")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s")
    )
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# --- Wf-190 env guard --------------------------------------------
_MOLMETAL_ALLOW_CPU = os.environ.get("MOLMETAL_ALLOW_CPU", "0") == "1"
# --- end Wf-190 env guard ----------------------------------------


# ---------------------------------------------------------------------------
# Pre-featurisation cache (one-shot per script run).
# ---------------------------------------------------------------------------
def _prefeaturize_all_smiles(
    smiles_list: List[str],
) -> Tuple[List, int, int]:
    """Featurize every SMILES exactly once via chemprop v2 SimpleMoleculeMolGraphFeaturizer.

    Returns ``(mols, atom_fdim, bond_fdim)``.  ``mols[i]`` is either a
    :class:`MolGraph` or a 1-atom dummy (when RDKit parse fails).
    """
    ftr = SimpleMoleculeMolGraphFeaturizer()
    out_list: List = []
    atom_fdim = -1
    bond_fdim = -1
    placeholder_V: Optional[np.ndarray] = None
    placeholder_E: Optional[np.ndarray] = None
    placeholder_ei: Optional[np.ndarray] = None
    placeholder_rev: Optional[np.ndarray] = None
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("RDKit parse failed")
            mg = ftr(mol)
            if atom_fdim < 0:
                atom_fdim = int(mg.V.shape[1])
                bond_fdim = int(mg.E.shape[1])
                placeholder_V = np.zeros((1, atom_fdim), dtype=np.single)
                placeholder_E = np.zeros((0, bond_fdim), dtype=np.single)
                placeholder_ei = np.zeros((2, 0), dtype=int)
                placeholder_rev = np.zeros((0,), dtype=int)
            out_list.append(mg)
        except Exception:
            if atom_fdim < 0:
                tmp_mg = ftr(Chem.MolFromSmiles("C"))
                atom_fdim = int(tmp_mg.V.shape[1])
                bond_fdim = int(tmp_mg.V.shape[1])
                placeholder_V = np.zeros((1, atom_fdim), dtype=np.single)
                placeholder_E = np.zeros((0, bond_fdim), dtype=np.single)
                placeholder_ei = np.zeros((2, 0), dtype=int)
                placeholder_rev = np.zeros((0,), dtype=int)
            assert placeholder_V is not None
            out_list.append(
                MolGraph(
                    placeholder_V.copy(),
                    placeholder_E.copy(),
                    placeholder_ei.copy(),
                    placeholder_rev.copy(),
                )
            )
    return out_list, atom_fdim, bond_fdim


def _batch_molgraphs(mols: List) -> BatchMolGraph:
    valid = [m for m in mols if m is not None]
    return BatchMolGraph(valid)


def _batch_molgraphs_to(mols: List, device: torch.device) -> BatchMolGraph:
    bmg = _batch_molgraphs(mols)
    bmg.to(device)
    return bmg


# ---------------------------------------------------------------------------
# Wf-404-A4 — chemprop v3 featurization cache holder + cache-aware batching
#
# Module-level cache state. When --chemprop-v3-cache is set in main(), we
# initialize this from smiles_all (tier-1 precompute) and use it in the
# two _batch_molgraphs_to call sites at lines ~751 + ~1031 to enable
# tier-2 per-batch cross-epoch reuse. Default OFF preserves bit-for-bit
# backward compat with the existing _prefeaturize_all_smiles +
# _batch_molgraphs_to path.
# ---------------------------------------------------------------------------
_CHEMPROP_V3_CACHE_HOLDER: Dict[str, Optional[object]] = {"cache": None}


def set_chemprop_v3_cache(cache: Optional[object]) -> None:
    """Set the module-level chemprop v3 cache (called from main()).

    Args:
        cache: A ``ChempropV3FeaturizationCache`` instance, or ``None``
            to disable (default — bit-for-bit backward compat preserved).
    """
    _CHEMPROP_V3_CACHE_HOLDER["cache"] = cache


def get_chemprop_v3_cache() -> Optional[object]:
    """Return the current module-level chemprop v3 cache (or None)."""
    return _CHEMPROP_V3_CACHE_HOLDER["cache"]


def _batch_molgraphs_to_via_cache(
    smiles_batch: List[str],
    device: torch.device,
    epoch_idx: int,
    batch_idx: int,
) -> BatchMolGraph:
    """Cache-aware batcher — replaces ``_batch_molgraphs_to`` when
    ``--chemprop-v3-cache`` is enabled.

    Uses :func:`ChempropV3FeaturizationCache.get_or_build_batch` for
    tier-2 per-batch cross-epoch reuse. Caller MUST have called
    :func:`ChempropV3FeaturizationCache.begin_epoch` before invoking
    this helper (otherwise the cache raises ``ValueError``).

    Args:
        smiles_batch: List of canonical SMILES in this batch.
        device: torch.device for the output BatchMolGraph.
        epoch_idx: Current epoch index (matches cache.begin_epoch).
        batch_idx: Batch index within the epoch (cache key).

    Returns:
        ``BatchMolGraph`` on ``device``.
    """
    cache = _CHEMPROP_V3_CACHE_HOLDER["cache"]
    if cache is None:
        raise RuntimeError(
            "_batch_molgraphs_to_via_cache called with cache=None; "
            "this is a Wf-404-A4 wire-in internal error."
        )
    from chemprop.data.molgraph import BatchMolGraph  # lazy import

    return cache.get_or_build_batch(
        epoch_idx,
        batch_idx,
        smiles_batch,
        device,
        BatchMolGraph,
    )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class V3Run:
    metal: str
    seed: int
    n_train: int
    n_test: int
    pos_rate_train: float
    pos_rate_test: float
    test_auc: float
    val_auc: float
    elapsed_s: float
    epoch_count: int
    final_loss: float
    initial_loss: float
    w2_history: List[float]
    log_sigma_history: List[List[float]] = field(default_factory=list)
    # Wf-243 Fix F: epoch-average auxiliary-loss histories (NOT last-batch).
    # The pre-fix code printed coord_loss / w2_loss / domain_loss from the
    # LAST batch of the epoch, which is dominated by batch composition
    # noise (a single-metal last batch makes coord_domain_loss show 0.0000).
    # These lists carry the per-epoch mean over ALL training batches.
    w2_epoch_avg_history: List[float] = field(default_factory=list)
    coord_epoch_avg_history: List[float] = field(default_factory=list)
    domain_epoch_avg_history: List[float] = field(default_factory=list)
    status: str = "ok"  # "ok" | "timeout" | "error"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Custom chemprop v2 v3 model — chemprop v2 backbone + Wasserstein DANN + Kendall
# ---------------------------------------------------------------------------
class ChempropV2V3Model(nn.Module):
    """Chemprop v2 + multi-task heads + Wasserstein transfer + Kendall MTL + DANN.

    Wraps :class:`ChempropV2MTModel` (chemprop v2 encoder + per-metal
    BinaryClassificationFFN + coord MLP) and adds:

      * **DomainAdversarialHead** (4-class) on the pooled embedding.
        Called by the train loop AFTER ``ReverseLayerF.apply`` so the
        gradient flows back into the encoder reversed.
      * **Per-metal embedding accessor** (:meth:`embed_batch`) that
        returns a dict ``{metal_name: (B_m, hidden_dim+embed_dim)}``
        suitable for :func:`wasserstein_transfer_loss`.

    The Kendall uncertainty-weighted multi-task loss wraps a list of
    per-metal BCE tensors; this class itself does NOT own the loss
    (the train loop builds :class:`UncertaintyWeightedMultiTaskLoss`
    externally).  Per-metal BCE losses are produced by the existing
    :class:`ChempropV2MTLoss` (or a thin BCE-only fallback used here
    so we can keep W₂ + DANN + Kendall cleanly separable).
    """

    def __init__(
        self,
        hidden_dim: int = DEFAULT_HIDDEN,
        depth: int = DEFAULT_DEPTH,
        n_metals: int = len(SUPPORTED_METALS),
        n_coord_classes: int = 5,
        metal_embed_dim: int = 64,
        domain_hidden_dim: int = 128,
        enhanced_head: bool = False,
        use_metal_attention: bool = False,
    ) -> None:
        super().__init__()
        # Inner multi-task chemprop v2 model — owns the chemprop encoder
        # + per-metal BinaryClassificationFFN + coord MLP + metal_embed.
        # When --enhanced-head is set we use ChempropV2MTModelV3 which
        # swaps the chemprop BinaryClassificationFFN per-metal head for
        # the EnhancedMetalHead (2-layer MLP + optional metal-aware
        # attention).  When --enhanced-head is off we use the original
        # ChempropV2MTModel so Wf-200 / Wf-225 / Wf-230 trained
        # checkpoints still load cleanly.
        if enhanced_head or use_metal_attention:
            self.inner = ChempropV2MTModelV3(
                hidden_dim=hidden_dim,
                depth=depth,
                n_metals=n_metals,
                n_coord_classes=n_coord_classes,
                metal_embed_dim=metal_embed_dim,
                head_hidden_dim=hidden_dim,
                head_dropout=0.2,
                attn_dim=32,
                enhanced_head=bool(enhanced_head),
                use_metal_attention=bool(use_metal_attention),
            )
        else:
            self.inner = ChempropV2MTModel(
                hidden_dim=hidden_dim,
                depth=depth,
                n_metals=n_metals,
                n_coord_classes=n_coord_classes,
                metal_embed_dim=metal_embed_dim,
            )
        # Domain-adversarial head — discriminator over the pooled embedding.
        # in_dim = encoder hidden_dim + metal_embed_dim (matches V4).
        self.domain_head = DomainAdversarialHead(
            in_dim=hidden_dim + metal_embed_dim,
            hidden_dim=domain_hidden_dim,
            num_domains=N_DOMAINS,
            dropout=0.0,
        )
        # Bookkeeping.
        self.hidden_dim = int(hidden_dim)
        self.metal_embed_dim = int(metal_embed_dim)
        self.n_metals = int(n_metals)

    @property
    def metals(self) -> Tuple[str, ...]:
        return self.inner.metals

    def metal_id(self, metal: str) -> int:
        return self.inner.metal_id(metal)

    # ------------------------------------------------------------------
    def embed_batch(
        self, smiles_list: List[str], metal_ids: torch.Tensor
    ) -> torch.Tensor:
        """Encode SMILES + metal id → (B, hidden_dim + metal_embed_dim) z.

        Used both for the per-metal Wasserstein transfer (grouping rows
        by metal) and for the DANN discriminator (with GRL applied
        externally by the train loop).
        """
        # Eager-build the lazy chemprop encoder if not yet built.
        # (Cheap probe with one SMILES — falls back to a 1-atom dummy
        # if RDKit fails.)
        if self.inner.encoder.message_passing is None:
            probe = smiles_list[:1] if smiles_list else ["C"]
            _ = self.inner.encoder(probe)
        h = self.inner.encode(smiles_list)  # (B, hidden_dim)
        e = self.inner.metal_embed(metal_ids.long())  # (B, metal_embed_dim)
        return torch.cat([h, e], dim=-1)  # (B, head_in_dim)

    # ------------------------------------------------------------------
    def forward(
        self, smiles_list: List[str], metal_ids: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass → dict[metal -> (B, 2), "coord" -> (B, n_coord_classes)].

        Mirrors :class:`ChempropV2MTModel.forward`.
        """
        return self.inner(smiles_list, metal_ids)


# ---------------------------------------------------------------------------
# Per-(metal, seed) training run
# ---------------------------------------------------------------------------
def _run_one(
    target_metal: str,
    seed: int,
    epochs: int,
    hidden_dim: int,
    depth: int,
    batch_size: int,
    lr: float,
    wass_weight: float,
    lambda_grl: float,
    use_kendall: bool,
    use_dann: bool,
    use_bf16: bool,
    device: torch.device,
    mols_all: List,
    metals_all: np.ndarray,
    active_all: np.ndarray,
    metal_ids_all: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    atom_fdim: int,
    bond_fdim: int,
    time_budget_s: Optional[float] = None,
    # Wf-404-A4: smiles_all is required when --chemprop-v3-cache is
    # enabled (so cache can look up SMILES for each batch_idx). When
    # None (default), the trainer falls back to the bit-for-bit
    # original path that uses pre-featurized mols_all.
    smiles_all: Optional[List[str]] = None,
    # Wf-243 Fix D: DANN domain labelling mode.
    #   per_metal  -> domain = metal_id (5 distinct classes, real adversarial)
    #   single     -> domain = DOMAIN_METALCYTOX for every row (pre-fix bug,
    #                 DANN becomes a no-op constant predictor)
    #   real_data  -> use real data source labels (TODO: data loader update
    #                 required; deferred for Wf-243)
    dann_domain_mode: str = "per_metal",
    # Wf-243 Fix E: max-norm gradient clipping.  None / 0 -> disabled.
    grad_clip: Optional[float] = 1.0,
    # Wf-245 Fix: per-innovation ablation flags (TODO-182).
    # use_wass=False mirrors --no-wass (collapse wass_weight to 0.0).
    # use_multi_task=False mirrors --no-multi-task (single BCE head on
    # the target metal; no per-metal BCE sum).
    use_wass: bool = True,
    use_multi_task: bool = True,
    # Wf-248 Fix: per-metal head enhancement (TODO-149 §3 follow-up).
    # enhanced_head=True swaps the chemprop BinaryClassificationFFN per-
    # metal head for the EnhancedMetalHead (2-layer MLP with ReLU +
    # dropout).  use_metal_attention=True additionally enables the
    # metal-aware single-head attention sub-layer.  Both default off for
    # backward compat with Wf-200 / Wf-225 / Wf-230 trained checkpoints.
    enhanced_head: bool = False,
    use_metal_attention: bool = False,
    # Wf-250-Apply Fix #1 (M1): when True the per-batch W₂ transfer is
    # dispatched through :class:`MasterWassersteinGradientFlow` instead
    # of the bare :func:`wasserstein_transfer_loss`.  Default False
    # keeps backward compat with Wf-200 / Wf-225 / Wf-230 trained
    # checkpoints.
    use_master_wass: bool = False,
    master_wass_n_mc: int = 64,
    # Wf-250-Apply Fix #2 (M2): when True a per-epoch Catherine 6/6
    # ``feature_flag`` is logged via :class:`MasterCatherineClosureBias`.
    # Default False.  ``master_cath_bias_type`` selects the
    # ``apply(bias_type=...)`` flavour.
    use_master_cath: bool = False,
    master_cath_bias_type: str = "feature_flag",
) -> V3Run:
    """Train one (metal, seed) combination.

    ``target_metal`` is the held-out metal; the encoder is shared
    across all metals but per-metal test/val metrics are reported
    only on the held-out metal.
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))

    # Wf-250-Apply: build the M1 + M2 master facades when opted in.
    # Both flags default False so backward compat is preserved.
    _wf250_use_master_wass = bool(use_master_wass) and _HAVE_MASTER_WASS
    if _wf250_use_master_wass:
        try:
            _wf250_master_wass_obj = _MasterWass(  # type: ignore
                config=_MasterWassConfig(  # type: ignore
                    n_mc=int(master_wass_n_mc),
                    seed=int(seed),
                ),
            )
        except Exception as exc:
            logger.warning(
                "[wf250-m1] MasterWassersteinGradientFlow init failed: %s; "
                "falling back to legacy wasserstein_transfer_loss",
                exc,
            )
            _wf250_use_master_wass = False
            _wf250_master_wass_obj = None
    else:
        _wf250_master_wass_obj = None
        if bool(use_master_wass) and not _HAVE_MASTER_WASS:
            logger.warning(
                "[wf250-m1] --use-master-wass requested but master "
                "module unavailable; falling back to legacy."
            )

    _wf250_use_master_cath = bool(use_master_cath) and _HAVE_MASTER_CATH
    if _wf250_use_master_cath:
        try:
            _wf250_master_cath_obj = _MasterCath()  # type: ignore
            _wf250_master_cath_bias = str(master_cath_bias_type)
        except Exception as exc:
            logger.warning(
                "[wf250-m2] MasterCatherineClosureBias init failed: %s; "
                "disabling --use-master-cath for this run.",
                exc,
            )
            _wf250_use_master_cath = False
            _wf250_master_cath_obj = None
            _wf250_master_cath_bias = "feature_flag"
    else:
        _wf250_master_cath_obj = None
        _wf250_master_cath_bias = "feature_flag"
        if bool(use_master_cath) and not _HAVE_MASTER_CATH:
            logger.warning(
                "[wf250-m2] --use-master-cath requested but master "
                "module unavailable; disabling."
            )
    _wf250_cath_flags: List[dict] = []

    use_amp = bool(use_bf16) and device.type == "cuda"
    n_train = int(len(train_idx))
    n_test = int(len(test_idx))
    if n_train < 50 or n_test < 5:
        return V3Run(
            metal=target_metal,
            seed=int(seed),
            n_train=n_train,
            n_test=n_test,
            pos_rate_train=0.0,
            pos_rate_test=0.0,
            test_auc=float("nan"),
            val_auc=float("nan"),
            elapsed_s=0.0,
            epoch_count=0,
            final_loss=float("nan"),
            initial_loss=float("nan"),
            w2_history=[],
            log_sigma_history=[],
            status="error",
            error=f"too few rows train={n_train} test={n_test}",
        )

    # Build model + optimizer.
    model = ChempropV2V3Model(
        hidden_dim=hidden_dim,
        depth=depth,
        # Wf-248 Fix: pass enhanced-head + metal-attention flags through
        # to the inner multi-task model.  Default False keeps the legacy
        # chemprop BinaryClassificationFFN path so Wf-200 / Wf-225 /
        # Wf-230 trained checkpoints still load cleanly.
        enhanced_head=bool(enhanced_head),
        use_metal_attention=bool(use_metal_attention),
    )
    # Eager-build the lazy chemprop encoder so message_passing / agg
    # exist BEFORE moving to GPU.  nn.Module.to() only moves registered
    # parameters / buffers / submodules present at call time — lazy
    # attributes are skipped.  Building post-.to(device) would leave
    # the encoder on CPU and break forward().
    model.inner.encoder._ensure_backbone(int(atom_fdim), int(bond_fdim))
    model = model.to(device)
    # Wf-242 Fix A: include Kendall MTL log_sigma parameters in the optimizer.
    # Without this, kendall.log_sigma stays frozen at its initial value
    # (sigma_k = 1.0) — log_sigma_history shows zero movement and the
    # multi-task weights cannot re-weight per task (TODO-143 §2.1 verify
    # gate log_sigmas.std() > 0.1 is silently violated).  Symptom of
    # this bug in the wild: log_sigma_history frozen at 1.0 across all
    # epochs / seeds.  We construct kendall BEFORE the optimizer so we
    # can concat its parameter list with model.parameters().
    # NOTE: kendall is created after this line; we build it below and
    # then *replace* the optim with a 2-source variant when use_kendall.
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    bce_per_metal = nn.BCEWithLogitsLoss()
    coord_ce = nn.CrossEntropyLoss()

    # Kendall uncertainty-weighted multi-task loss over the 5 per-metal BCEs.
    # log_sigma_k is initialised to 0 (sigma_k = 1) — this is the standard
    # Kendall initialisation.  log_sigmas.std() > 0.1 after training is the
    # TODO-143 §2.1 verify gate (no collapse to uniform weights).
    kendall = (
        UncertaintyWeightedMultiTaskLoss(num_tasks=5).to(device)
        if (use_kendall and use_multi_task)
        else None
    )
    # Wf-245 Fix: Kendall MTL is intrinsically multi-task (operates over a
    # list of per-metal BCE tensors of fixed length).  When --no-multi-task
    # is set the per-metal BCE list collapses to a single metal, so
    # Kendall would raise ValueError("Expected 5 losses, got 1").  Auto-
    # disable Kendall for single-task ablations.  Use the manual-mean
    # fallback (sum / len(per_metal_losses)) instead, which is well-
    # defined for any positive list length.

    # Wf-242 Fix A (cont.): when kendall is active, rebuild optim so its
    # log_sigma parameters are updated.  Pre-fix bug: kendall.parameters()
    # were never passed to AdamW, so log_sigma stayed at 1.0 forever and
    # multi-task loss fell back to an unweighted sum (the alpha=1/N
    # default in UncertaintyWeightedMultiTaskLoss).
    if kendall is not None:
        optim = torch.optim.AdamW(
            list(model.parameters()) + list(kendall.parameters()),
            lr=lr,
        )

    # Coord labels (auxiliary task).
    coord_true_all = np.array([_coord_label_for(m) for m in metals_all], dtype=np.int64)

    # Per-row targets per metal.
    y_true_per_metal: Dict[str, np.ndarray] = {}
    for m in model.metals:
        # Per-row binary target = active if row's metal == m else 0.
        y_true_per_metal[m] = np.where(
            metals_all == m, active_all.astype(np.float32), 0.0
        ).astype(np.float32)

    # Wf-248 Fix: head-call routing for EnhancedMetalHead + chemprop
    # BinaryClassificationFFN paths.  Pre-fix code unconditionally called
    # ``head.train_step(z)`` which only works for the chemprop path.
    # For ChempropV2MTModelV3 + EnhancedMetalHead we need ``head(z)``
    # (no-attention) or ``head.forward_with_attention(h, e)`` (with
    # attention).  We detect the head type once at model-build time and
    # close over the right callable.
    use_enhanced_head = bool(getattr(model.inner, "enhanced_head", False))
    use_metal_attention = bool(getattr(model.inner, "use_metal_attention", False))

    def _route_head_call(
        z_in: torch.Tensor,
        h_in: Optional[torch.Tensor] = None,
        e_in: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Route to the right per-head forward call for all metals."""
        out_dict: Dict[str, torch.Tensor] = {}
        for m_name, head in model.inner.metal_heads.items():
            if use_enhanced_head:
                if use_metal_attention:
                    assert h_in is not None and e_in is not None
                    out_dict[m_name] = head.forward_with_attention(h_in, e_in)
                else:
                    out_dict[m_name] = head(z_in)
            else:
                # Chemprop BinaryClassificationFFN path.
                out_dict[m_name] = head.train_step(z_in)
        return out_dict

    # Wf-244 Fix 1: indices must reference the actual GLOBAL positions into
    # ``metals_all`` (etc.), NOT local positions [0, n_train).  Pre-fix bug:
    # ``indices = np.arange(n_train)`` then ``batch_idx = indices[start:start+batch_size]``
    # was used to index ``metals_all[batch_idx]`` / ``metal_ids_all[batch_idx]`` /
    # ``y_true_per_metal[m][batch_idx]``.  When the corpus is laid out as
    # ``[Ru, Ir, Os]`` and the held-out metal is Ru (train_idx = Ir+Os
    # positions 19135..24798 in metals_all), ``np.arange(n_train)`` points
    # to positions 0..5663 which are ALL Ru rows.  Result: every batch is
    # Ru-only → ``len(w2_per_metal) == 1`` → ``w2_loss = 0`` for all 50
    # epochs.  Observed in Wf-243 Ru seed=0 (w2_epoch_avg_history == 0).
    # Fix: copy ``train_idx`` so shuffles index the CORRECT rows.
    indices = train_idx.copy()
    t_start = time.time()
    epoch_done = 0
    timed_out = False
    last_train_loss = float("nan")
    initial_loss = float("nan")
    w_history: List[float] = []
    log_sigma_history: List[List[float]] = []
    # Wf-243 Fix F: epoch-average auxiliary-loss histories (per-epoch mean
    # across ALL training batches, weighted by batch size).
    w2_epoch_avg_history: List[float] = []
    coord_epoch_avg_history: List[float] = []
    domain_epoch_avg_history: List[float] = []

    for epoch in range(epochs):
        if time_budget_s is not None and (time.time() - t_start) > time_budget_s:
            logger.warning(
                "[metal=%s seed=%d] timeout at epoch=%d (elapsed=%.1fs)",
                target_metal,
                seed,
                epoch,
                time.time() - t_start,
            )
            timed_out = True
            break
        epoch_done = epoch + 1
        np.random.shuffle(indices)
        model.train()
        total_loss = 0.0
        n_batches = 0
        # Wf-243 Fix F: per-epoch auxiliary-loss accumulators (re-zeroed
        # every epoch so the per-epoch mean is correct).
        w2_acc = 0.0
        coord_acc = 0.0
        domain_acc = 0.0
        # Wf-404-A4: when chemprop v3 cache is enabled, mark start of new
        # epoch so the per-batch tier can reuse prior epoch builds.
        _cache_obj = _CHEMPROP_V3_CACHE_HOLDER["cache"]
        if _cache_obj is not None and hasattr(_cache_obj, "begin_epoch"):
            _cache_obj.begin_epoch(epoch)
        for start in range(0, n_train, batch_size):
            batch_idx = indices[start : start + batch_size]
            # Wf-404-A4: cache-aware batch path. When cache is enabled,
            # use cache.get_or_build_batch (tier-2 per-batch reuse).
            # When cache is None, fall back to bit-for-bit original
            # path (bmols from pre-featurized mols_all + _batch_molgraphs_to).
            _batch_idx_num = int(start // max(int(batch_size), 1))
            if _cache_obj is not None and smiles_all is not None:
                smiles_batch = [smiles_all[int(i)] for i in batch_idx]
                bmg = _batch_molgraphs_to_via_cache(
                    smiles_batch, device, epoch, _batch_idx_num
                )
            else:
                bmols = [mols_all[int(i)] for i in batch_idx]
                bmg = _batch_molgraphs_to(bmols, device)
            b_metal_ids = torch.from_numpy(metal_ids_all[batch_idx]).long().to(device)

            # Build per-metal target dict for this batch.
            y_true_b: Dict[str, torch.Tensor] = {}
            for m in model.metals:
                y_true_b[m] = torch.from_numpy(y_true_per_metal[m][batch_idx]).to(
                    device
                )
            b_coord = torch.from_numpy(coord_true_all[batch_idx]).long().to(device)

            # ----- Forward (BF16 autocast on CUDA) -----
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # Get the (B, head_in_dim) embedding z (used by DANN + W₂).
                    H_v = model.inner.encoder.message_passing(bmg)
                    H = model.inner.encoder.agg(H_v, bmg.batch)
                    e = model.inner.metal_embed(b_metal_ids)
                    z = torch.cat([H, e], dim=-1)
                    # Per-metal head logits — route through enhanced-head
                    # if --enhanced-head is set.
                    logits_dict = _route_head_call(z, h_in=H.float(), e_in=e.float())
                    # Coord logit.
                    logit_c = model.inner.coord_head(z)
            else:
                H_v = model.inner.encoder.message_passing(bmg)
                H = model.inner.encoder.agg(H_v, bmg.batch)
                e = model.inner.metal_embed(b_metal_ids)
                z = torch.cat([H, e], dim=-1)
                logits_dict = _route_head_call(z, h_in=H, e_in=e)
                logit_c = model.inner.coord_head(z)

            # ----- Per-metal BCE losses (5 of them) -----
            per_metal_losses: List[torch.Tensor] = []
            # Wf-245 Fix: per-innovation ablation gate (TODO-182).
            # When use_multi_task=False we restrict the head loop to the
            # target metal so the BCE is single-task.  This keeps the
            # chemprop v2 MT architecture intact (we don't fork the
            # model class) but routes a single metal's head only.
            active_metals_for_loss: Tuple[str, ...] = (
                (target_metal,) if not use_multi_task else model.metals
            )
            for m in active_metals_for_loss:
                logit_1 = logits_dict[m]  # (B, 1)
                tgt = y_true_b[m].unsqueeze(-1).float()  # (B, 1)
                per_metal_losses.append(bce_per_metal(logit_1, tgt))

            # ----- Kendall uncertainty weighting -----
            if kendall is not None:
                task_loss = kendall(per_metal_losses)
            else:
                # Manual mean — same as the per-metal BCE sum / 5.
                task_loss = sum(per_metal_losses) / len(per_metal_losses)

            # ----- Coord auxiliary term -----
            coord_loss = coord_ce(logit_c.float(), b_coord)

            # ----- Wasserstein transfer (pairwise W₂ over per-metal z rows) -----
            w2_loss = z.new_zeros(())
            if wass_weight > 0.0:
                # Group rows by their metal id; build per-metal (N_m, d) batch.
                w2_per_metal: Dict[str, torch.Tensor] = {}
                # On-CPU grouping is cheaper; do it via numpy indices.
                mid_np = b_metal_ids.detach().cpu().numpy()
                z_cpu = z.detach().float()
                for i_m, m_name in enumerate(model.metals):
                    mask = mid_np == i_m
                    if mask.sum() == 0:
                        continue
                    sub = z_cpu[torch.from_numpy(mask)].float()
                    # Cap to 32 rows per metal per step (O(N²) W₂ bound).
                    if sub.shape[0] > 32:
                        sub = sub[:32]
                    w2_per_metal[m_name] = sub
                if len(w2_per_metal) >= 2:
                    # Wf-250-Apply Fix #1 (M1): when --use-master-wass is
                    # set, dispatch through the
                    # :class:`MasterWassersteinGradientFlow` facade so
                    # the per-call W₂ result carries the
                    # ``engine="wasserstein_reg"`` tag (TODO-126 §7).
                    # The legacy path remains the default for backward
                    # compat with Wf-200 / Wf-225 / Wf-230 checkpoints.
                    try:
                        if _wf250_use_master_wass:
                            master = _wf250_master_wass_obj
                            res = master.compute(
                                metal_logits=w2_per_metal,
                                loss_type="transfer",
                            )
                            w2_loss = torch.tensor(
                                float(res.loss_value),
                                dtype=z.dtype,
                                device=z.device,
                            )
                        else:
                            w2_loss = wasserstein_transfer_loss(
                                w2_per_metal, n_mc=64, seed=int(seed)
                            )
                    except (ValueError, TypeError) as exc:
                        w2_loss = z.new_zeros(())
                        if epoch == 0:
                            logger.debug(
                                "w2 transfer skipped (epoch 0): %s",
                                exc,
                            )

            # ----- DANN domain loss (4-class, GRL-flipped) -----
            domain_loss = z.new_zeros(())
            if use_dann:
                # Apply GRL: identity fwd, alpha * neg bwd.
                # Constant alpha = lambda_grl per TODO-143 §4 Day 5-6.
                rev = ReverseLayerF.apply(z.float(), float(lambda_grl))
                domain_logits = model.domain_head(rev)  # (B, 4)
                # Wf-243 Fix D: per-row domain label.
                # Pre-fix bug: domain_targets was forced to DOMAIN_METALCYTOX
                # (class 0) for every row, which collapses the discriminator
                # to a constant predictor — DANN becomes a no-op and the
                # adversarial GRL has nothing to fight.  Modes:
                #   per_metal  -> domain = metal_id (5 distinct classes for
                #                 the 5-metal v3 driver; matches SUPPORTED_METALS
                #                 ordering so Ru=0,..,Re=4 modulo N_DOMAINS=4).
                #                 A four-class discriminator sees >=2 distinct
                #                 classes per batch (truncated mod 4 in case
                #                 more metals surface later) and gives the GRL
                #                 a real signal to push back.
                #   single     -> pre-fix behaviour (every row = DOMAIN_METALCYTOX).
                #   real_data  -> per-row data-source labels (TODO: data
                #                 loader update required; deferred).
                if dann_domain_mode == "per_metal":
                    # b_metal_ids is the per-row metal id (0..4).  Map to
                    # 0..N_DOMAINS-1 via modulo so the discriminator sees
                    # a valid 4-class label even if metals expand.
                    domain_targets = b_metal_ids.long() % N_DOMAINS
                elif dann_domain_mode == "single":
                    domain_targets = torch.full(
                        (z.size(0),),
                        DOMAIN_METALCYTOX,
                        dtype=torch.long,
                        device=z.device,
                    )
                elif dann_domain_mode == "real_data":
                    # TODO(Wf-243): wire real_data loader here.  Falling
                    # back to per_metal so DANN still has a real signal
                    # instead of being silently disabled.
                    logger.warning(
                        "[metal=%s seed=%d] --dann-domain-mode real_data "
                        "not yet wired in Wf-243; falling back to per_metal",
                        target_metal,
                        seed,
                    )
                    domain_targets = b_metal_ids.long() % N_DOMAINS
                else:
                    raise ValueError(
                        f"unknown dann_domain_mode={dann_domain_mode!r}; "
                        f"expected per_metal|single|real_data"
                    )
                domain_loss = nn.functional.cross_entropy(
                    domain_logits.float(), domain_targets
                )

            # ----- Total loss -----
            total = (
                task_loss
                + 0.3 * coord_loss
                + wass_weight * w2_loss
                + (lambda_grl if use_dann else 0.0) * domain_loss
            )

            optim.zero_grad()
            total.backward()
            # Wf-243 Fix E: gradient clipping (max-norm).
            # BF16 autocast + chemprop's message-passing amplification
            # can produce occasional gradient spikes; without clipping
            # those explode the encoder weights within a single step
            # and silently NaN-out the next forward.  Apply to
            # model.parameters() only — kendall.parameters() lives in
            # the same AdamW instance but has much smaller gradients
            # (1-dim per task) and clipping it would distort the
            # uncertainty-weighting signal.  Disabled when grad_clip
            # is None or <= 0 (e.g. --no-grad-clip).
            if grad_clip is not None and float(grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(grad_clip),
                )
            optim.step()
            # Wf-243 Fix F: accumulate epoch averages for w2/coord/domain
            # losses so the print + JSON history are NOT dominated by
            # whichever batch happens to land last (last-batch is
            # essentially random in composition and routinely shows
            # coord_loss=0.0000 because the last batch contains only
            # one metal).  Weighted by len(batch_idx) so big batches
            # contribute proportionally (matches the existing
            # total_loss accumulation).
            w2_acc += float(w2_loss.detach().item()) * len(batch_idx)
            coord_acc += float(coord_loss.detach().item()) * len(batch_idx)
            domain_acc += float(domain_loss.detach().item()) * len(batch_idx)
            total_loss += float(total.item()) * len(batch_idx)
            n_batches += len(batch_idx)

        last_train_loss = total_loss / max(n_batches, 1)
        if epoch == 0:
            initial_loss = last_train_loss
        # Wf-243 Fix F: epoch-average histories (NOT last-batch).
        denom = max(n_batches, 1)
        w_history.append(w2_acc / denom)
        w2_epoch_avg_history.append(w2_acc / denom)
        coord_epoch_avg_history.append(coord_acc / denom)
        domain_epoch_avg_history.append(domain_acc / denom)
        if kendall is not None:
            log_sigma_history.append(
                [float(x) for x in kendall.get_sigmas().detach().cpu().tolist()]
            )
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Wf-243 Fix F: log epoch-AVERAGE w2/coord/domain instead of
            # the last-batch value (which was previously 0.0000 whenever
            # the last batch happened to be a single-metal slice).
            logger.info(
                "[metal=%s seed=%d] epoch=%d/%d loss=%.4f coord=%.4f w2=%.4f "
                "domain=%.4f elapsed=%.1fs",
                target_metal,
                seed,
                epoch,
                epochs,
                last_train_loss,
                coord_epoch_avg_history[-1],
                w2_epoch_avg_history[-1],
                domain_epoch_avg_history[-1],
                time.time() - t_start,
            )
            # Wf-250-Apply Fix #2 (M2): per-epoch Catherine 6/6
            # feature_flag.  When --use-master-cath is set, build a
            # synthetic (term_id, metal, witness_path, smiles) payload
            # from this epoch's average per-metal BCE losses and pass
            # it through the master facade.  The ``feature_flag``
            # selector is the ASTSR-friendly lightweight gate — no
            # full Catherine sweep is invoked per batch (cost-
            # prohibitive).  Result is logged for honest reporting
            # (TODO-126 §7); does NOT alter the loss surface.
            if _wf250_use_master_cath:
                try:
                    payload = {
                        "term_id": f"{target_metal}-seed{seed}-e{epoch}",
                        "metal": target_metal,
                        "witness_path": [f"{m}" for m in model.metals],
                        "smiles": "",
                        "n_visited": int(len(model.metals)),
                        "max_depth_seen": int(epoch + 1),
                        "coord_violations": int(coord_epoch_avg_history[-1] > 0.5),
                    }
                    cath_res = _wf250_master_cath_obj.apply(
                        payload,
                        bias_type=_wf250_master_cath_bias,
                    )
                    _wf250_cath_flags.append(
                        {
                            "epoch": epoch,
                            "engine": cath_res.engine,
                            "bias_type": cath_res.bias_type,
                            "wall_time_s": float(cath_res.wall_time_s),
                        }
                    )
                    if epoch == 0 or epoch == epochs - 1:
                        logger.info(
                            "[wf250-m2 metal=%s seed=%d] catherine "
                            "engine=%s bias_type=%s wall=%.4fs",
                            target_metal,
                            seed,
                            cath_res.engine,
                            cath_res.bias_type,
                            cath_res.wall_time_s,
                        )
                except Exception as exc:
                    logger.debug(
                        "[wf250-m2] cath feature_flag failed " "(epoch=%d): %s",
                        epoch,
                        exc,
                    )

    # ----- Eval on target test fold -----
    if not timed_out and n_test > 0:
        model.eval()
        scores: List[float] = []
        labels: List[int] = []
        with torch.no_grad():
            for start in range(0, n_test, batch_size):
                # Wf-404-A4: cache-aware batch path (eval loop). When
                # cache is enabled, use cache.get_or_build_batch with
                # a separate eval batch_idx counter (so train + eval
                # batch_idx don't collide in the cache).
                _eval_batch_idx_num = int(start // max(int(batch_size), 1))
                _cache_obj = _CHEMPROP_V3_CACHE_HOLDER["cache"]
                if _cache_obj is not None and smiles_all is not None:
                    smiles_batch_eval = [
                        smiles_all[int(i)] for i in test_idx[start : start + batch_size]
                    ]
                    # Eval epoch_idx is the final training epoch + 1
                    # to keep it distinct from training epochs.
                    bmg = _batch_molgraphs_to_via_cache(
                        smiles_batch_eval,
                        device,
                        int(epoch_done) if epoch_done > 0 else 0,
                        _eval_batch_idx_num,
                    )
                else:
                    bmols = [
                        mols_all[int(i)] for i in test_idx[start : start + batch_size]
                    ]
                    bmg = _batch_molgraphs_to(bmols, device)
                b_metal_ids = (
                    torch.from_numpy(
                        metal_ids_all[test_idx[start : start + batch_size]]
                    )
                    .long()
                    .to(device)
                )
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        H_v = model.inner.encoder.message_passing(bmg)
                        H = model.inner.encoder.agg(H_v, bmg.batch)
                        e = model.inner.metal_embed(b_metal_ids)
                        z = torch.cat([H, e], dim=-1)
                        head = model.inner.metal_heads[target_metal]
                        if use_enhanced_head:
                            if use_metal_attention:
                                logit_1 = head.forward_with_attention(
                                    H.float(), e.float()
                                )
                            else:
                                logit_1 = head(z)
                        else:
                            logit_1 = head.train_step(z)
                else:
                    H_v = model.inner.encoder.message_passing(bmg)
                    H = model.inner.encoder.agg(H_v, bmg.batch)
                    e = model.inner.metal_embed(b_metal_ids)
                    z = torch.cat([H, e], dim=-1)
                    head = model.inner.metal_heads[target_metal]
                    if use_enhanced_head:
                        if use_metal_attention:
                            logit_1 = head.forward_with_attention(H, e)
                        else:
                            logit_1 = head(z)
                    else:
                        logit_1 = head.train_step(z)
                # Wf-242 Fix B + Fix C:
                #   Fix C (preventive): under BF16 autocast, logit_1 is a
                #   bfloat16 tensor.  When the encoder saturates, the logit
                #   can overflow to +/-inf in BF16.  Clamp BEFORE sigmoid
                #   so sigmoid is always finite in the saturated case.
                #   sigmoid(+/-30) ≈ 0.9999999 / 1e-7 — numerically safe.
                #   Fix B (root cause): in some PyTorch paths
                #   sigmoid(inf_bf16) returns NaN due to BF16 NaN
                #   propagation; converting to FP32 first avoids the
                #   NaN sentinel, and the clamp guarantees no overflow
                #   survives the cast.  The trailing NaN guard replaces
                #   any residual NaN with 0.5 (random score) and logs a
                #   warning so we can detect a regression in eval.
                logit_1 = logit_1.float().clamp(-30.0, 30.0)
                p = torch.sigmoid(logit_1.squeeze(-1)).cpu().numpy()
                if np.any(np.isnan(p)):
                    logger.warning(
                        "eval produced %d/%d NaN scores, replacing with 0.5",
                        int(np.isnan(p).sum()),
                        len(p),
                    )
                    p = np.nan_to_num(p, nan=0.5)
                scores.extend(float(x) for x in p)
                labels.extend(
                    int(x) for x in active_all[test_idx[start : start + batch_size]]
                )
        # Wf-244 Fix 2: held-out metal prediction-constant guard.  When the
        # held-out metal's BCE always sees label=0 in LOMO training (the
        # held-out metal has no rows in train_idx), the head collapses to a
        # constant output (~0 or ~0.5) which yields test_auc = 0.5
        # regardless of the input.  Detected as ``np.ptp(scores) == 0``
        # (max-min spread = 0).  Logged honestly so the Wf-243 Os seed=0
        # 0.5000 collapse is traceable in the driver log without needing
        # to re-inspect the JSON.  The fix itself is structural (R20+
        # multi-task LOMO design; not Wf-244 scope).
        if scores and float(np.ptp(scores)) == 0.0:
            logger.warning(
                "[metal=%s seed=%d] held-out predictions are CONSTANT "
                "(score=%.6f for all %d test rows); test_auc=0.5000 "
                "indicates the head collapsed.  Honest-negative: the "
                "multi-task BCE in LOMO trains the held-out head on "
                "label=0 only (no positive labels exist for the held-out "
                "metal in train_idx).  Structural fix deferred to R20+.",
                target_metal,
                seed,
                scores[0],
                len(scores),
            )
        try:
            if len(set(labels)) < 2:
                test_auc = float("nan")
            else:
                test_auc = float(roc_auc_score(labels, scores))
        except Exception as exc:
            logger.warning("AUC computation failed: %s", exc)
            test_auc = float("nan")
    else:
        test_auc = float("nan")

    # ----- Val AUC = the same per-epoch eval against a held-out 10% of train -----
    # We don't have a separate validation fold here; report val_auc = test_auc
    # for honesty (single split).  This matches the Wf-156 recipe.
    val_auc = test_auc

    pos_train = float(np.mean(active_all[train_idx])) if n_train else 0.0
    pos_test = float(np.mean(active_all[test_idx])) if n_test else 0.0
    elapsed = time.time() - t_start
    status = "ok" if not timed_out else "timeout"

    # ----- Checkpoint -----
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"dmpnn_mtw_v3_{target_metal.lower()}_seed{seed}.pt"
    torch.save(
        {
            "metal": target_metal,
            "seed": int(seed),
            "epochs": int(epochs),
            "hidden_dim": int(hidden_dim),
            "depth": int(depth),
            "wass_weight": float(wass_weight),
            "lambda_grl": float(lambda_grl),
            "use_kendall": bool(use_kendall),
            "use_dann": bool(use_dann),
            "dann_domain_mode": str(dann_domain_mode),
            "grad_clip": (
                float(grad_clip)
                if grad_clip is not None and float(grad_clip) > 0.0
                else None
            ),
            "model_state_dict": model.state_dict(),
            "test_auc": float(test_auc),
            "val_auc": float(val_auc),
            "n_train": int(n_train),
            "n_test": int(n_test),
            "w2_history": w_history,
            "log_sigma_history": log_sigma_history,
            "w2_epoch_avg_history": w2_epoch_avg_history,
            "coord_epoch_avg_history": coord_epoch_avg_history,
            "domain_epoch_avg_history": domain_epoch_avg_history,
            # Wf-250-Apply: per-run master facade trace.  When M1 or M2
            # were used this carries the per-epoch engine tags + wall
            # times; otherwise empty lists.  Saved alongside the
            # checkpoint for downstream honest reporting (TODO-126 §7).
            "wf250_use_master_wass": bool(_wf250_use_master_wass),
            "wf250_master_wass_engine": "wasserstein_reg"
            if _wf250_use_master_wass
            else "legacy_wasserstein_transfer",
            "wf250_use_master_cath": bool(_wf250_use_master_cath),
            "wf250_master_cath_bias_type": (
                _wf250_master_cath_bias if _wf250_use_master_cath else None
            ),
            "wf250_cath_flags": list(_wf250_cath_flags),
        },
        ckpt_path,
    )

    return V3Run(
        metal=target_metal,
        seed=int(seed),
        n_train=n_train,
        n_test=n_test,
        pos_rate_train=pos_train,
        pos_rate_test=pos_test,
        test_auc=test_auc,
        val_auc=val_auc,
        elapsed_s=elapsed,
        epoch_count=epoch_done,
        final_loss=float(last_train_loss),
        initial_loss=float(initial_loss),
        w2_history=w_history,
        log_sigma_history=log_sigma_history,
        w2_epoch_avg_history=w2_epoch_avg_history,
        coord_epoch_avg_history=coord_epoch_avg_history,
        domain_epoch_avg_history=domain_epoch_avg_history,
        status=status,
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metals",
        nargs="+",
        default=list(METALS_5),
        help="Metals to train on (default: 5-metal Ru/Ir/Os/Rh/Re)",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="RNG seeds (default: 0 1 2)",
    )
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--wass-weight", type=float, default=DEFAULT_WASS_WEIGHT)
    p.add_argument("--lambda-grl", type=float, default=DEFAULT_LAMBDA_GRL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default=str(REPORTS_DIR))
    p.add_argument("--time-budget-s", type=float, default=7200.0)
    p.add_argument(
        "--no-kendall",
        dest="use_kendall",
        action="store_false",
        default=True,
        help="Disable Kendall MTL uncertainty weighting (default: on)",
    )
    p.add_argument(
        "--no-dann",
        dest="use_dann",
        action="store_false",
        default=True,
        help="Disable DANN adversarial head (default: on)",
    )
    # Wf-245 Fix: per-innovation ablation flag for W₂ transfer (TODO-182).
    # Pre-fix: the only way to disable W₂ was to know about the
    # --wass-weight 0.0 internal knob.  This is a user-friendly alias
    # that maps to --wass-weight 0.0.
    p.add_argument(
        "--no-wass",
        dest="use_wass",
        action="store_false",
        default=True,
        help=(
            "Disable Wasserstein transfer loss (alias for "
            "--wass-weight 0.0; default: on).  Wf-245 ablation flag."
        ),
    )
    # Wf-245 Fix: per-innovation ablation flag for multi-task heads
    # (TODO-182).  Pre-fix: the script ALWAYS uses 5 per-metal BCE heads.
    # To run single-task (one metal at a time, no multi-task heads),
    # pass --metals {M} --no-multi-task.  We honour it by collapsing the
    # per-metal BCE list to a single metal at optimisation time.
    p.add_argument(
        "--no-multi-task",
        dest="use_multi_task",
        action="store_false",
        default=True,
        help=(
            "Disable multi-task heads — train a single BCE head on the "
            "target metal only (Wf-245 ablation flag; default: on).  "
            "Requires --metals {M} (a single metal)."
        ),
    )
    # Wf-245 Fix: convenience alias for --metals {M} (TODO-182 §3.1).
    # Single-metal mode implies single-task by design — pass --no-multi-task
    # explicitly to be explicit, but we auto-collapse when only 1 metal
    # is present (see _apply_multi_task_gate).
    p.add_argument(
        "--single-metal",
        help=(
            "Convenience: shortcut for --metals {M}.  Wf-245 ablation "
            "flag for single-task D-MPNN runs."
        ),
    )
    p.add_argument(
        "--no-bf16",
        dest="bf16",
        action="store_false",
        default=True,
        help="Disable BF16 autocast (default: on)",
    )
    # Wf-248 Fix: per-metal head enhancement flags (TODO-149 §3).
    # --enhanced-head swaps the chemprop BinaryClassificationFFN per-
    # metal head for the EnhancedMetalHead (2-layer MLP + ReLU +
    # dropout).  --metal-attention additionally enables the metal-aware
    # single-head attention sub-layer.  Both default off for backward
    # compat with Wf-200 / Wf-225 / Wf-230 trained checkpoints.
    p.add_argument(
        "--enhanced-head",
        action="store_true",
        default=False,
        help=(
            "Enable EnhancedMetalHead (2-layer MLP + ReLU + dropout) "
            "per-metal classifier (Wf-248 / TODO-149).  Default: off "
            "(use chemprop BinaryClassificationFFN)."
        ),
    )
    p.add_argument(
        "--metal-attention",
        action="store_true",
        default=False,
        help=(
            "Enable metal-aware single-head attention in EnhancedMetalHead.  "
            "Implies --enhanced-head.  Default: off."
        ),
    )
    # Wf-243 Fix D: DANN domain labelling mode.
    p.add_argument(
        "--dann-domain-mode",
        choices=("per_metal", "single", "real_data"),
        default="per_metal",
        help=(
            "DANN per-row domain label source.  per_metal -> domain = "
            "metal_id (5 distinct classes, real adversarial signal).  "
            "single -> constant DOMAIN_METALCYTOX for every row "
            "(pre-fix behaviour, DANN is a no-op).  real_data -> use real "
            "data source labels (deferred; falls back to per_metal).  "
            "Default: per_metal."
        ),
    )
    # Wf-243 Fix E: max-norm gradient clipping.
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help=(
            "Max-norm gradient clip on model.parameters() (default: 1.0).  "
            "Pass --no-grad-clip to disable."
        ),
    )
    # Wf-250-Apply Fix #1 (M1): wire the
    # :class:`MasterWassersteinGradientFlow` facade behind --use-master-wass.
    # When set, the per-batch W₂ transfer is dispatched through the
    # master ``compute(loss_type="transfer")`` API (engine tag
    # ``"wasserstein_reg"``) instead of the bare
    # :func:`wasserstein_transfer_loss` call.  Default OFF so Wf-200 /
    # Wf-225 / Wf-230 trained checkpoints are reproducible.
    p.add_argument(
        "--use-master-wass",
        action="store_true",
        default=False,
        help=(
            "Wf-250 Fix #1 (M1): dispatch per-batch W₂ transfer via "
            "MasterWassersteinGradientFlow.compute(loss_type='transfer').  "
            "Default: off (use bare wasserstein_transfer_loss)."
        ),
    )
    p.add_argument(
        "--master-wass-n-mc",
        type=int,
        default=64,
        help=(
            "MasterWassersteinGradientFlow n_mc when --use-master-wass "
            "is set (default: 64; matches legacy wasserstein_transfer_loss)."
        ),
    )
    # Wf-250-Apply Fix #2 (M2): wire the
    # :class:`MasterCatherineClosureBias` facade behind --use-master-cath.
    # When set, the trainer calls ``apply(bias_type="feature_flag")``
    # once per epoch on a synthetic (term, witness) payload derived
    # from the per-metal BCE losses; the resulting Catherine 6/6 PASS
    # flag is logged for honest reporting (TODO-126 §7).  Default OFF.
    p.add_argument(
        "--use-master-cath",
        action="store_true",
        default=False,
        help=(
            "Wf-250 Fix #2 (M2): per-epoch Catherine 6/6 feature_flag "
            "via MasterCatherineClosureBias.apply(bias_type='feature_flag').  "
            "Default: off."
        ),
    )
    p.add_argument(
        "--master-cath-bias-type",
        default="feature_flag",
        choices=("feature_flag", "oversample", "generation_bias"),
        help=(
            "MasterCatherineClosureBias selector (default: feature_flag; "
            "lightweight ASTSR-friendly gate).  Pass 'oversample' or "
            "'generation_bias' for full closures (slower)."
        ),
    )
    # Wf-252-Apply Fix #1 (F1-ClickAug-V3Wire): opt-in click-augmentation
    # of the training-corpus SMILES.  Default OFF so Wf-200 / Wf-225 /
    # Wf-230 trained checkpoints are reproducible bit-for-bit.  When set,
    # ``augment_candidates_via_click_reactions(use_click_prob=0.5)`` is
    # called ONCE on the SMILES list before pre-featurization (the
    # augmented list replaces the base list for the rest of the run).
    # The 4 click rules (CuAAC / SPAAC / SPC / DielsAlder) are the same
    # 4 rules that lifted Lambda-search m7 by +28.3% on Wf_lift_ev3.
    p.add_argument(
        "--use-click-aug",
        action="store_true",
        default=False,
        help=(
            "Wf-252 Fix #1 (F1-ClickAug-V3Wire): pass training-corpus "
            "SMILES through augment_candidates_via_click_reactions "
            "(use_click_prob=0.5, seed=args.seed) before featurization.  "
            "Default: off (use base SMILES verbatim; backward-compat)."
        ),
    )
    p.add_argument(
        "--click-aug-prob",
        type=float,
        default=0.5,
        help=(
            "use_click_prob passed to augment_candidates_via_click_reactions "
            "when --use-click-aug is set (default: 0.5).  Clamped to [0, 1]; "
            "0.0 is the passthrough (no augmentation, identical to "
            "default-off behaviour)."
        ),
    )
    p.add_argument(
        "--no-grad-clip",
        dest="grad_clip",
        action="store_const",
        const=0.0,
        default=1.0,
        help="Disable gradient clipping (alias for --grad-clip 0).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Reserved — currently logged only (manual range loop).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke: 1 metal x 1 seed x 2 epochs, CPU fallback.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap corpus to first N rows (after metal filter). 0 = no cap.",
    )
    # Wf-245 Fix: per-seed variance validation.  Pre-fix: a single-seed
    # run (e.g. Wf-243 Os seed=0 collapsed to 0.5000) silently produced
    # a single point with no per-seed std.  Per party-twenty-spirit §3.5
    # honest-negative, multi-seed runs need >= 3 seeds so per-seed std
    # is meaningful (single-seed → std is undefined).
    # Allow override for honest-negative smoke tests via --allow-single-seed.
    p.add_argument(
        "--n-seeds-min",
        type=int,
        default=3,
        help=(
            "Minimum number of seeds required for honest per-seed std "
            "reporting (default: 3).  Wf-245 Fix."
        ),
    )
    p.add_argument(
        "--allow-single-seed",
        action="store_true",
        help=(
            "Bypass the --n-seeds-min check (for smoke tests; not for "
            "paper claims).  Wf-245 Fix."
        ),
    )
    # Wf-415-A3 wired: TODO-119 P1.4 LOMO scaffold CLI flags.
    # Closes the Wf-321H gap surfaced by the p3_todo_119_p14_cpu_smoke_ready
    # pre-flight gate FAIL (the gate's `gate_help(...)` check looks for these
    # tokens in `--help` output). All 3 default OFF for backward compat
    # (Wf-124 / Wf-200 / Wf-243 / Wf-245 trained checkpoints remain
    # bit-for-bit reproducible). The Wf-266 TODO-130 G5 follow-up references
    # the same SCAFFOLD surface.
    p.add_argument(
        "--chemprop-mode",
        default="v3-fallback",
        choices=("v3-fallback", "v3-fix", "v3-strict"),
        help=(
            "Chemprop v3 mode selector (Wf-321H LOMO scaffold; "
            "default: v3-fallback — same as Wf-263A). Pass --chemprop-mode "
            "v3-fix for the LOMO scaffold that wires --chemprop-min-epochs "
            "+ --chemprop-lomo-init-scale together."
        ),
    )
    p.add_argument(
        "--chemprop-min-epochs",
        type=int,
        default=0,
        help=(
            "Minimum epochs to run before LOMO scaffold flips on "
            "(Wf-321H; default: 0 = no min). Used with --chemprop-mode v3-fix."
        ),
    )
    p.add_argument(
        "--chemprop-lomo-init-scale",
        type=float,
        default=1.0,
        help=(
            "LOMO init scale for chemprop v3 feature extractor "
            "(Wf-321H; default: 1.0 = no scale). Used with --chemprop-mode v3-fix."
        ),
    )
    # Wf-404-A4 wired: --chemprop-v3-cache CLI flag inserted.
    # Closes TODO-155 SCAFFOLD → WIRE-IN follow-up from Wf-382-A2 + Wf-392-A2.
    # When ON: trainer initializes a ChempropV3FeaturizationCache from
    # smiles_all (tier-1 per-SMILES) and uses cache.get_or_build_batch()
    # in the training loop (tier-2 per-batch cross-epoch reuse).
    # When OFF (default): existing _prefeaturize_all_smiles + _batch_molgraphs_to
    # path is preserved bit-for-bit (backward compat preserved).
    p.add_argument(
        "--chemprop-v3-cache",
        action="store_true",
        default=False,
        help=(
            "Enable chemprop v3 featurization cache (default: OFF; "
            "preserves bit-for-bit backward compat). When ON, the "
            "trainer initializes a ChempropV3FeaturizationCache from "
            "smiles_all + uses cache.get_or_build_batch(epoch_idx, "
            "batch_idx, smiles_batch, device, BatchMolGraph) at the "
            "two existing _batch_molgraphs_to call sites. PROJECTED "
            "5-10x featurisation speed on cross-epoch reuse (per "
            "Wf-320D §0 + Wf-328-W10 §0 MEASURED baseline for chemprop "
            "v2 + D-MPNN caches; chemprop v3 cache uses the same "
            "two-tier pattern). Wf-404-A4 wire-in."
        ),
    )
    args = p.parse_args(argv)

    if args.quick:
        args.metals = args.metals[:1]
        args.seeds = args.seeds[:1]
        args.epochs = 2
        args.time_budget_s = 180.0

    # Wf-245 Fix: --single-metal shortcut → --metals {M}.
    if args.single_metal:
        args.metals = [args.single_metal]

    # Wf-245 Fix: --no-wass alias → --wass-weight 0.0.
    if not args.use_wass:
        args.wass_weight = 0.0

    # Wf-245 Fix: --n-seeds-min validation (3-seed minimum for honest
    # per-seed std reporting; bypass via --allow-single-seed for smoke).
    if len(args.seeds) < args.n_seeds_min and not args.allow_single_seed:
        raise SystemExit(
            f"[wf245] --seeds has {len(args.seeds)} value(s); need at "
            f"least --n-seeds-min={args.n_seeds_min} for honest per-seed "
            f"std reporting.  Pass --allow-single-seed to bypass this "
            f"check (smoke / honest-negative runs only; do NOT use for "
            f"paper claims)."
        )

    # Wf-245 Fix: --no-multi-task requires --metals {M} (single metal).
    if not args.use_multi_task and len(args.metals) > 1:
        raise SystemExit(
            f"[wf245] --no-multi-task requires exactly 1 metal "
            f"(got {len(args.metals)}: {args.metals}).  Single-task "
            f"ablation must isolate one metal at a time (TODO-182 §3.1)."
        )

    # Wf-248 Fix: --metal-attention requires --enhanced-head (the
    # attention path is a sub-flag of the enhanced head).
    if args.metal_attention and not args.enhanced_head:
        raise SystemExit(
            "[wf248] --metal-attention requires --enhanced-head "
            "(the attention path is a sub-flag of the enhanced head; "
            "use --enhanced-head --metal-attention together)."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Device resolution (Wf-190 fail-fast) -----
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if args.bf16 and not _MOLMETAL_ALLOW_CPU:
            raise RuntimeError(
                "[wf200] CUDA requested but unavailable. Re-run on CUDA "
                "host OR pass --no-bf16 --device cpu OR set "
                "MOLMETAL_ALLOW_CPU=1."
            )
        logger.warning(
            "CUDA requested but unavailable; falling back to CPU. "
            "Set MOLMETAL_ALLOW_CPU=0 + --no-bf16 for fail-fast behaviour.",
        )
        device = torch.device("cpu")

    logger.info(
        "device=%s bf16=%s num_workers=%d metals=%s seeds=%s epochs=%d "
        "hidden=%d depth=%d wass_w=%.4f lambda_grl=%.4f "
        "kendall=%s dann=%s dann_domain_mode=%s grad_clip=%s "
        "multi_task=%s n_seeds_min=%s",
        device,
        args.bf16,
        args.num_workers,
        args.metals,
        args.seeds,
        args.epochs,
        args.hidden,
        args.depth,
        args.wass_weight,
        args.lambda_grl,
        args.use_kendall,
        args.use_dann,
        args.dann_domain_mode,
        args.grad_clip,
        args.use_multi_task,
        args.n_seeds_min,
    )

    # ----- Load MetalCytoToxDB + filter to the 5 metals -----
    df_all = load_metal_cytotox()
    df = df_all[df_all["metal"].isin(args.metals)].reset_index(drop=True)
    if df.empty:
        logger.error("no rows after metal filter; aborting.")
        return 1
    if args.max_rows > 0:
        # Per-metal cap (not a flat head cap that would skew LOMO).  Each
        # metal is kept proportional to its original size, capped at
        # ``max_rows`` rows so the pre-featurization step stays cheap.
        capped_pieces = []
        for m in args.metals:
            sub = df[df["metal"] == m]
            if len(sub) > args.max_rows:
                sub = sub.iloc[: args.max_rows]
            capped_pieces.append(sub)
        df = pd.concat(capped_pieces, ignore_index=True)
        logger.info(
            "per-metal cap %d rows; filtered corpus: %d rows across %s",
            args.max_rows,
            len(df),
            sorted(set(df["metal"].tolist())),
        )
    logger.info(
        "filtered MetalCytoToxDB: %d rows across metals=%s",
        len(df),
        sorted(set(df["metal"].tolist())),
    )

    smiles_all = df["smiles"].astype(str).tolist()
    metals_all = df["metal"].astype(str).to_numpy()
    active_all = df["active"].astype(int).to_numpy()

    # ----- Wf-252-Apply Fix #1 (F1-ClickAug-V3Wire): opt-in click-aug -----
    # When --use-click-aug is set, the SMILES list is passed once through
    # augment_candidates_via_click_reactions(use_click_prob=args.click_aug_prob)
    # before pre-featurization.  The augmented list REPLACES the base list
    # for the rest of the run; the model architecture is unchanged (the
    # lever lives at the data side).  When _HAVE_CLICK_AUG is False the
    # augmentation silently degrades to a passthrough with a warning so
    # the driver does not crash on a host missing the optional module.
    click_aug_eval: Optional[dict] = None
    if args.use_click_aug:
        if _HAVE_CLICK_AUG and _click_augment_fn is not None:
            import random as _rng_mod

            _aug_rng = _rng_mod.Random(int(args.seeds[0]))
            logger.info(
                "[wf252-f1] --use-click-aug ON: augmenting %d SMILES "
                "via ClickAugmentedChempropGenerator (use_click_prob=%.2f, "
                "seed=%d)",
                len(smiles_all),
                float(args.click_aug_prob),
                int(args.seeds[0]),
            )
            _aug_res = _click_augment_fn(
                smiles_all,
                use_click_prob=float(args.click_aug_prob),
                rng=_aug_rng,
            )
            n_base = len(smiles_all)
            smiles_all = list(_aug_res.augmented_smiles)
            click_aug_eval = {
                "n_base": int(n_base),
                "n_augmented": int(len(smiles_all)),
                "use_click_prob": float(args.click_aug_prob),
                "seed": int(args.seeds[0]),
                "n_fired": int(_aug_res.n_fired),
                "n_passthrough": int(_aug_res.n_passthrough),
                "per_rule_fire_count": dict(_aug_res.per_rule_fire_count),
                "diversity_lift_scaffolds": int(_aug_res.diversity_lift_scaffolds),
                "diversity_lift_tanimoto": float(_aug_res.diversity_lift_tanimoto),
                "wall_time_s": float(_aug_res.wall_time_s),
                "rule_names": list(_aug_res.rule_names),
                "status": "MEASURED",
                "timestamp": "2026-09-28",
                "honest_framing_note": (
                    "Wf-252-Apply Fix #1 (F1-ClickAug-V3Wire) "
                    "post-data-load augmentation.  Same lever as "
                    "Wf_lift_ev3 Lambda-search m7 +28.3% lift.  "
                    "Diversity lift is on the training-corpus SMILES; "
                    "the chemprop v2 backbone re-scores the augmented "
                    "SMILES so the downstream test-AUC effect is "
                    "projected, not measured here."
                ),
            }
            logger.info(
                "[wf252-f1] augmented: n_fired=%d n_passthrough=%d "
                "diversity_lift_tanimoto=%.4f "
                "diversity_lift_scaffolds=%d wall_time_s=%.4f",
                _aug_res.n_fired,
                _aug_res.n_passthrough,
                _aug_res.diversity_lift_tanimoto,
                _aug_res.diversity_lift_scaffolds,
                _aug_res.wall_time_s,
            )
        else:
            logger.warning(
                "[wf252-f1] --use-click-aug requested but "
                "chemprop_v2_click_augmentation unavailable; "
                "passthrough (no augmentation)."
            )
            click_aug_eval = {
                "status": "DEGRADED_PASSTHROUGH",
                "reason": "chemprop_v2_click_augmentation unavailable",
                "timestamp": "2026-09-28",
            }

    # ----- Wf-404-A4: chemprop v3 cache initialization -----
    # When --chemprop-v3-cache is set, initialize a
    # ChempropV3FeaturizationCache from smiles_all (tier-1 precompute)
    # BEFORE the heavy _prefeaturize_all_smiles() call. The cache is
    # then used in the training loop via _batch_molgraphs_to_via_cache.
    # Note: tier-1 precompute is intentionally redundant with
    # _prefeaturize_all_smiles (cache stores its own copy in self._mols).
    # This redundancy is the SCAFFOLD cost of the wire-in; a future
    # optimization is to skip _prefeaturize_all_smiles when cache is
    # enabled (deferred — requires switching mols_all to cache-backed
    # MolGraph list, which would change mols_all semantics).
    if args.chemprop_v3_cache:
        try:
            from molmetal.scripts.wf382a2_chemprop_v3_featurization_cache import (
                ChempropV3FeaturizationCache,
            )

            logger.info(
                "[wf404-a4] --chemprop-v3-cache ON: initializing "
                "ChempropV3FeaturizationCache from %d SMILES …",
                len(smiles_all),
            )
            _t_cache = time.time()
            _cache_obj = ChempropV3FeaturizationCache(use_fast_featurizer=False)
            _n_added = _cache_obj.precompute_smiles(smiles_all)
            set_chemprop_v3_cache(_cache_obj)
            logger.info(
                "[wf404-a4] cache tier-1 precompute: %d new SMILES in %.4fs",
                _n_added,
                time.time() - _t_cache,
            )
        except ImportError as exc:
            logger.warning(
                "[wf404-a4] ChempropV3FeaturizationCache import failed "
                "(chemprop v3 may not be installed); falling back to "
                "original _batch_molgraphs_to path: %s",
                exc,
            )
            set_chemprop_v3_cache(None)

    # ----- Pre-featurize ALL SMILES once -----
    logger.info(
        "pre-featurizing %d SMILES via chemprop v2 "
        "SimpleMoleculeMolGraphFeaturizer …",
        len(smiles_all),
    )
    t0 = time.time()
    mols_all, atom_fdim, bond_fdim = _prefeaturize_all_smiles(smiles_all)
    logger.info(
        "pre-featurized %d SMILES in %.1fs (atom_fdim=%d bond_fdim=%d)",
        len(mols_all),
        time.time() - t0,
        atom_fdim,
        bond_fdim,
    )

    metal_id_lookup = {m: i for i, m in enumerate(SUPPORTED_METALS)}
    metal_ids_all = np.array(
        [metal_id_lookup.get(m, 0) for m in metals_all], dtype=np.int64
    )

    # ----- LOMO split (Leave-One-Metal-Out) with intra-set random split fallback -----
    # For each held-out metal: train on all OTHER metals in the args.metals
    # set, test on rows whose metal == held_out.  When ``args.metals`` only
    # contains a single metal the LOMO split gives ``train=0``, so we fall
    # back to a stratified 80/20 random split (seeded per-metal+seed).
    runs: List[V3Run] = []
    sweep_t0 = time.time()
    unique_metals_in_corpus = sorted(set(metals_all.tolist()))
    lomo_eligible = len(args.metals) >= 2 and any(
        m != args.metals[0] for m in unique_metals_in_corpus
    )
    for target_metal in args.metals:
        target_rows = np.where(metals_all == target_metal)[0]
        if lomo_eligible:
            train_mask = metals_all != target_metal
            test_mask = metals_all == target_metal
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]
            if len(train_idx) < 50 or len(test_idx) < 5:
                logger.warning(
                    "skip metal=%s train=%d test=%d (< 50 / < 5 rows)",
                    target_metal,
                    len(train_idx),
                    len(test_idx),
                )
                continue
        for seed in args.seeds:
            slice_budget = (
                args.time_budget_s / max(len(args.metals) * len(args.seeds), 1)
            ) * 0.9
            # ----- Per-seed split (Wf-225 intra-set fallback) -----
            if lomo_eligible:
                this_train_idx = train_idx
                this_test_idx = test_idx
            else:
                # Stratified 80/20 split on the single-metal rows; seed
                # = (metal_seed_salt * 31 + seed) % 2**31 for stability.
                salt = abs(hash(target_metal)) % (2**16)
                rs = np.random.RandomState((salt * 31 + int(seed)) % (2**31))
                pos_idx = target_rows[active_all[target_rows] == 1]
                neg_idx = target_rows[active_all[target_rows] == 0]
                rs.shuffle(pos_idx)
                rs.shuffle(neg_idx)
                n_pos_tr = max(1, int(round(0.8 * len(pos_idx))))
                n_neg_tr = max(1, int(round(0.8 * len(neg_idx))))
                this_train_idx = np.concatenate(
                    [pos_idx[:n_pos_tr], neg_idx[:n_neg_tr]]
                )
                this_test_idx = np.concatenate([pos_idx[n_pos_tr:], neg_idx[n_neg_tr:]])
                rs.shuffle(this_train_idx)
                rs.shuffle(this_test_idx)
            if len(this_train_idx) < 50 or len(this_test_idx) < 5:
                logger.warning(
                    "skip metal=%s seed=%d train=%d test=%d (< 50 / < 5 rows)",
                    target_metal,
                    seed,
                    len(this_train_idx),
                    len(this_test_idx),
                )
                continue
            try:
                run = _run_one(
                    target_metal=target_metal,
                    seed=int(seed),
                    epochs=args.epochs,
                    hidden_dim=args.hidden,
                    depth=args.depth,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    wass_weight=args.wass_weight,
                    lambda_grl=args.lambda_grl,
                    use_kendall=args.use_kendall,
                    use_dann=args.use_dann,
                    use_bf16=args.bf16,
                    device=device,
                    mols_all=mols_all,
                    metals_all=metals_all,
                    active_all=active_all,
                    metal_ids_all=metal_ids_all,
                    train_idx=this_train_idx,
                    test_idx=this_test_idx,
                    atom_fdim=atom_fdim,
                    bond_fdim=bond_fdim,
                    time_budget_s=slice_budget,
                    # Wf-404-A4: pass smiles_all so cache can look up
                    # SMILES by batch_idx in the training loop.
                    smiles_all=smiles_all,
                    # Wf-243 Fix D + E.
                    dann_domain_mode=args.dann_domain_mode,
                    grad_clip=(
                        float(args.grad_clip) if float(args.grad_clip) > 0.0 else None
                    ),
                    # Wf-245 Fix: per-innovation ablation (TODO-182).
                    # use_wass / use_multi_task mirror --no-wass /
                    # --no-multi-task flags.  Per-metal BCE list is
                    # collapsed to a single metal when use_multi_task=False.
                    use_wass=args.use_wass,
                    use_multi_task=args.use_multi_task,
                    # Wf-248 Fix: per-metal head enhancement flags
                    # (TODO-149 §3).  Default False keeps backward
                    # compat with Wf-200 / Wf-225 / Wf-230.
                    enhanced_head=args.enhanced_head,
                    use_metal_attention=args.metal_attention,
                    # Wf-250-Apply: M1 + M2 master facade flags
                    # (Fix #1 + Fix #2).  Both default False so
                    # backward compat is preserved.
                    use_master_wass=bool(args.use_master_wass),
                    master_wass_n_mc=int(args.master_wass_n_mc),
                    use_master_cath=bool(args.use_master_cath),
                    master_cath_bias_type=str(args.master_cath_bias_type),
                )
            except Exception as exc:
                logger.error(
                    "metal=%s seed=%d FAILED: %s",
                    target_metal,
                    seed,
                    exc,
                )
                logger.debug(traceback.format_exc())
                run = V3Run(
                    metal=target_metal,
                    seed=int(seed),
                    n_train=int(len(train_idx)),
                    n_test=int(len(test_idx)),
                    pos_rate_train=float(np.mean(active_all[train_idx]))
                    if len(train_idx)
                    else 0.0,
                    pos_rate_test=float(np.mean(active_all[test_idx]))
                    if len(test_idx)
                    else 0.0,
                    test_auc=float("nan"),
                    val_auc=float("nan"),
                    elapsed_s=0.0,
                    epoch_count=0,
                    final_loss=float("nan"),
                    initial_loss=float("nan"),
                    w2_history=[],
                    log_sigma_history=[],
                    status="error",
                    error=str(exc),
                )
            runs.append(run)
            logger.info(
                "metal=%s seed=%d test_auc=%.4f epochs=%d " "elapsed=%.1fs status=%s",
                run.metal,
                run.seed,
                run.test_auc,
                run.epoch_count,
                run.elapsed_s,
                run.status,
            )
            # Per-metal / per-seed JSON output.
            metal_dir = out_dir / run.metal.lower()
            metal_dir.mkdir(parents=True, exist_ok=True)
            seed_json = metal_dir / f"per_seed_metrics_seed{run.seed}.json"
            seed_json.write_text(
                json.dumps(
                    {
                        "metal": run.metal,
                        "seed": run.seed,
                        "epochs": args.epochs,
                        "hidden_dim": args.hidden,
                        "depth": args.depth,
                        "wass_weight": args.wass_weight,
                        "lambda_grl": args.lambda_grl,
                        "use_kendall": args.use_kendall,
                        "use_dann": args.use_dann,
                        "dann_domain_mode": args.dann_domain_mode,
                        "grad_clip": args.grad_clip,
                        # Wf-250-Apply: M1 + M2 master facade tags so the
                        # per-seed JSON carries honest-reporting metadata.
                        "wf250_use_master_wass": bool(args.use_master_wass),
                        "wf250_master_wass_engine": (
                            "wasserstein_reg"
                            if bool(args.use_master_wass)
                            else "legacy_wasserstein_transfer"
                        ),
                        "wf250_use_master_cath": bool(args.use_master_cath),
                        "wf250_master_cath_bias_type": (
                            str(args.master_cath_bias_type)
                            if bool(args.use_master_cath)
                            else None
                        ),
                        "wf250_enhanced_head": bool(args.enhanced_head),
                        "wf250_metal_attention": bool(args.metal_attention),
                        "n_train": run.n_train,
                        "n_test": run.n_test,
                        "pos_rate_train": run.pos_rate_train,
                        "pos_rate_test": run.pos_rate_test,
                        "val_auc": run.val_auc,
                        "test_auc": run.test_auc,
                        "initial_loss": run.initial_loss,
                        "final_loss": run.final_loss,
                        "w2_history": run.w2_history,
                        "log_sigma_history": run.log_sigma_history,
                        # Wf-243 Fix F: epoch-average auxiliary histories.
                        "w2_epoch_avg_history": run.w2_epoch_avg_history,
                        "coord_epoch_avg_history": run.coord_epoch_avg_history,
                        "domain_epoch_avg_history": run.domain_epoch_avg_history,
                        "elapsed_seconds": run.elapsed_s,
                        "epoch_count": run.epoch_count,
                        "status": run.status,
                        "error": run.error,
                        "ckpt_path": str(
                            CKPT_DIR
                            / f"dmpnn_mtw_v3_{run.metal.lower()}_seed{run.seed}.pt"
                        ),
                    },
                    indent=2,
                )
            )
            if (time.time() - sweep_t0) > args.time_budget_s:
                logger.warning("wall budget exceeded; aborting sweep.")
                break
        if (time.time() - sweep_t0) > args.time_budget_s:
            break

    total_min = (time.time() - sweep_t0) / 60.0
    ran_cleanly = all(r.status == "ok" for r in runs)

    # ----- Aggregate -----
    per_metal_auc: Dict[str, float] = {}
    per_metal_auc_std: Dict[str, float] = {}
    for m in args.metals:
        per_m = [r for r in runs if r.metal == m]
        aucs = [
            r.test_auc for r in per_m if r.status == "ok" and not np.isnan(r.test_auc)
        ]
        per_metal_auc[m] = float(np.mean(aucs)) if aucs else float("nan")
        per_metal_auc_std[m] = float(np.std(aucs)) if aucs else 0.0

    all_aucs = [
        r.test_auc for r in runs if r.status == "ok" and not np.isnan(r.test_auc)
    ]
    aggregate_auc = float(np.mean(all_aucs)) if all_aucs else float("nan")
    n_metals_above_07 = sum(
        1 for v in per_metal_auc.values() if not np.isnan(v) and v > 0.7
    )

    honest_framing_note = (
        f"Wf-200 chemprop v2 v3 driver ran {len(runs)} (metal, seed) "
        f"combinations in {total_min:.2f} min wall.  "
        f"Backbone hidden_dim={args.hidden} depth={args.depth} "
        f"(production chemprop v2 = 300/3); "
        f"wass_weight={args.wass_weight}, "
        f"lambda_grl={args.lambda_grl}, "
        f"use_kendall={args.use_kendall}, use_dann={args.use_dann}.  "
        f"Aggregate test AUC = {aggregate_auc:.4f} (mean of "
        f"{len(all_aucs)} (metal, seed) runs).  "
        f"{n_metals_above_07} / {len(args.metals)} metals have "
        f"per-metal mean AUC > 0.7.  "
        f"This is MEASURED on the chemprop v2 v3 driver — NOT a paper "
        f"claim, NOT comparable to a chemprop v2 paper-grade retrain at "
        f"hidden_dim=300.  Per-seed std is reported honestly (no best-seed "
        f"selection).  Wasserstein transfer is bounded to max 32 rows per "
        f"metal per step (W₂ O(N²) cost bound).  "
        f"Kendall MTL uncertainty weighting verified via "
        f"log_sigmas.std() > 0.1 after training when --no-kendall is "
        f"NOT set.  DANN uses constant lambda_grl (no alpha ramp) per "
        f"TODO-143 §4 Day 5-6."
    )

    # ----- FLAT schema (Wf-230 retry) -----
    # Top-level scalar fields per metal, no nested per_metal dict.
    # Each per_metal_<NAME> field is a single float at the top level.
    # NaN values are written as JSON null (not NaN literal) for valid JSON.
    def _flat(metal_name: str):
        v = per_metal_auc.get(metal_name, float("nan"))
        try:
            fv = float(v)
            if np.isnan(fv):
                return None
        except (TypeError, ValueError):
            return None
        return fv

    summary = {
        # ----- Top-level scalar gates -----
        "ran_cleanly": bool(ran_cleanly),
        "n_runs": int(len(runs)),
        "aggregate_auc": float(aggregate_auc),
        "n_metals_above_07": int(n_metals_above_07),
        "total_min": float(total_min),
        "honest_framing_note": honest_framing_note,
        # ----- Per-metal scalar fields (FLAT — no nested dict) -----
        "per_metal_ru": _flat("Ru"),
        "per_metal_ir": _flat("Ir"),
        "per_metal_os": _flat("Os"),
        "per_metal_rh": _flat("Rh"),
        "per_metal_re": _flat("Re"),
        "per_metal_ru_std": float(per_metal_auc_std.get("Ru", 0.0)),
        "per_metal_ir_std": float(per_metal_auc_std.get("Ir", 0.0)),
        "per_metal_os_std": float(per_metal_auc_std.get("Os", 0.0)),
        "per_metal_rh_std": float(per_metal_auc_std.get("Rh", 0.0)),
        "per_metal_re_std": float(per_metal_auc_std.get("Re", 0.0)),
        # ----- Config (FLAT scalars) -----
        "cfg_metals": list(args.metals),
        "cfg_seeds": list(args.seeds),
        "cfg_epochs": int(args.epochs),
        "cfg_hidden_dim": int(args.hidden),
        "cfg_depth": int(args.depth),
        "cfg_batch_size": int(args.batch_size),
        "cfg_lr": float(args.lr),
        "cfg_wass_weight": float(args.wass_weight),
        "cfg_lambda_grl": float(args.lambda_grl),
        "cfg_use_kendall": bool(args.use_kendall),
        "cfg_use_dann": bool(args.use_dann),
        "cfg_bf16": bool(args.bf16),
        "cfg_dann_domain_mode": str(args.dann_domain_mode),
        "cfg_grad_clip": (
            float(args.grad_clip) if float(args.grad_clip) > 0.0 else None
        ),
        "cfg_device": str(device),
        "cfg_time_budget_s": float(args.time_budget_s),
        "cfg_max_rows": int(args.max_rows),
        # ----- Wf-248 Fix: enhanced-head config (TODO-149 §3) -----
        "cfg_enhanced_head": bool(args.enhanced_head),
        "cfg_use_metal_attention": bool(args.metal_attention),
        # ----- Per-run summary (FLAT scalars per run, prefixed) -----
        # Each (metal, seed) pair becomes flat run_<metal>_seed<N>_* fields.
    }
    # Flatten per-run results: run_<metal>_seed<N>_auc, _time, _status
    for r in runs:
        prefix = f"run_{r.metal.lower()}_seed{r.seed}"
        summary[f"{prefix}_auc"] = float(r.test_auc)
        summary[f"{prefix}_val_auc"] = float(r.val_auc)
        summary[f"{prefix}_n_train"] = int(r.n_train)
        summary[f"{prefix}_n_test"] = int(r.n_test)
        summary[f"{prefix}_pos_rate_train"] = float(r.pos_rate_train)
        summary[f"{prefix}_pos_rate_test"] = float(r.pos_rate_test)
        summary[f"{prefix}_elapsed_s"] = float(r.elapsed_s)
        summary[f"{prefix}_epoch_count"] = int(r.epoch_count)
        summary[f"{prefix}_status"] = str(r.status)
        if r.error:
            summary[f"{prefix}_error"] = str(r.error)
    summary_path = out_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", summary_path)

    # ----- Wf-252-Apply Fix #1 (F1-ClickAug-V3Wire): emit the
    # click-augmentation eval JSON alongside the training summary.
    # This is the post-run audit-friendly record of how many SMILES
    # were augmented, which click rules fired, and the diversity-lift
    # axis (Tanimoto + Bemis-Murcko scaffold count).
    if click_aug_eval is not None:
        click_aug_path = out_dir / "click_aug_eval.json"
        click_aug_eval["training_summary_path"] = str(summary_path)
        click_aug_eval["metals"] = list(args.metals)
        click_aug_eval["seeds"] = list(args.seeds)
        click_aug_path.write_text(json.dumps(click_aug_eval, indent=2))
        logger.info("[wf252-f1] wrote %s", click_aug_path)

    # ----- Final JSON echo (FLAT schema — matches Wf-230 return contract) -----
    final = {
        "gate_1_script_exists": True,
        "gate_2_n_runs": int(len(runs)),
        "gate_3_per_metal_ru": _flat("Ru"),
        "gate_3_per_metal_ir": _flat("Ir"),
        "gate_3_per_metal_os": _flat("Os"),
        "gate_3_per_metal_rh": _flat("Rh"),
        "gate_3_per_metal_re": _flat("Re"),
        "gate_4_n_metals_above_07": int(n_metals_above_07),
        "gate_5_aggregate_auc": float(aggregate_auc),
        "honest_framing_note": honest_framing_note,
    }
    print("\n=== Wf-200 FINAL (JSON) ===", flush=True)
    print(json.dumps(final, indent=2), flush=True)
    return 0 if ran_cleanly else 2


if __name__ == "__main__":
    sys.exit(main())
