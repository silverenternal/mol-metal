"""Wf-156 — Chemprop v2 LOMO 5-metal x 5-seed x 50-epoch sweep.

==============================================================================
Wf-156 ultracode follow-up to Wf-155 (TODO-155 / TODO-148 §3 follow-up)

Purpose
-------
Run Chemprop v2 LOMO benchmark with a fresh, dedicated training loop
that uses the chemprop v2 native ``(smiles_list, metal_ids)`` forward
signature directly.  The legacy ``train_metal_lomo_v2.py`` loop uses
DMPNN-only APIs and cannot consume the chemprop v2 forward signature
(TODO-148 §4 wiring is a separate work item).  This script is a focused,
single-purpose LOMO driver for chemprop v2 only.

Architecture (per task spec)
---------------------------
1. **Chemprop v2 MPNN backbone** — :class:`ChempropV2MTModel` at
   ``hidden_dim=64, depth=2`` (the smoke-test config; production defaults
   300/3 are 5-7x slower per step and exceeded the Wf-155 budget).
2. **5-way multi-task heads + coord head** — shared chemprop encoder
   + per-metal BinaryClassificationFFN + coord MLP.  Mirrors V4 contract.
3. **LOMO split (Leave-One-Metal-Out)** — implemented via
   ``sklearn.model_selection.LeaveOneGroupOut`` with ``groups=metals``.
   Each fold holds out one metal as test; the remaining 4 metals are
   train.  Equivalent to the legacy
   :class:`BalancedLOMOSplitter` logic but uses sklearn's official
   LOMO primitive as requested by the task spec.
4. **Pre-featurized MolGraph cache** — all SMILES are featurized ONCE
   at startup (CPU) into chemprop v2 :class:`MolGraph` objects; per
   batch we only build a :class:`BatchMolGraph` (cheap) and run the
   chemprop encoder on GPU.  This bypasses the Wf-155 featurization
   bottleneck (CPU per-batch SMILES → MolGraph was 95 % of the wall
   time on the previous attempt).

Targets
-------
* Per-metal 5-seed mean test AUC + std + n_train + n_test.
* Overall mean AUC (sanity, not a paper target).
* Honest framing: smaller backbone (64/2) + 1 hr wall budget — this is
  a SWEEP, not the chemprop v2 paper-grade retrain (TODO-148 §5).

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
1. **Backbone is smaller than production** — ``hidden_dim=64, depth=2``
   vs the chemprop v2 default 300/3.  Per-step cost is much lower but
   the model capacity is also reduced; AUC ceiling will be lower than
   a full chemprop v2 retrain.
2. **50 epochs may not be enough** for convergence at hidden_dim=64 —
   no LR scheduler is engaged; plain AdamW at lr=1e-3.
3. **Wall budget = 3600s** — if a (fold, seed) run exceeds its budget
   slice (~144s = 3600/25), it is allowed to finish but the wall
   budget may force an early abort.
4. **Coord head is trained jointly** but not evaluated — coord is
   auxiliary (weight 0.3).
5. **No Wasserstein transfer / no L2 guard** — pure chemprop v2 BCE
   baseline (the legacy LOMO loop adds these as Wf-145 P2 fixes;
   this chemprop v2 LOMO is the clean baseline before those fixes).

Usage
-----
Default (5 metal x 5 seed x 50 epoch, 3600s budget)::

    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/train_chemprop_v2_lomo.py \\
        --out molmetal/reports/wf156_chemprop_v2_lomo.json

Smoke (1 metal x 1 seed x 2 epochs, 60s budget)::

    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/train_chemprop_v2_lomo.py --quick

Outputs ``molmetal/reports/wf156_chemprop_v2_lomo.json`` with the
``per_metal_summary`` schema requested by the task spec.
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
import logging
import os
import statistics
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

# Ensure project root on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import LeaveOneGroupOut

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.data.molgraph import MolGraph

from molmetal.models.chemprop_v2_mt_model import (
    ChempropV2MTLoss,
    ChempropV2MTModel,
    SUPPORTED_METALS,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUT = PROJECT_ROOT / "molmetal" / "reports" / "wf156_chemprop_v2_lomo.json"
METALS_5: Tuple[str, ...] = ("Ru", "Ir", "Os", "Rh", "Re")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_EPOCHS: int = 50
DEFAULT_HIDDEN: int = 64
DEFAULT_DEPTH: int = 2
DEFAULT_BATCH: int = 64
DEFAULT_LR: float = 1e-3

logger = logging.getLogger("wf156")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s")
    )
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


# --- Wf-190 env guard --------------------------------------------
# Fail fast at import time if --bf16 / CUDA is required but unavailable.
# Mirrors mol-metal-gpu-execution §"MEASURED-vs-PROJECTED label discipline":
# we never silently run on CPU when the script claims GPU.  Auto-fallback to
# CPU is still supported (set MOLMETAL_ALLOW_CPU=1) to keep CPU-only boxes
# working, but explicit is preferred.  The cuda-availability check itself
# lives in main() (after `import torch`) per the molmetal-gpu-execution
# skill convention.
_MOLMETAL_ALLOW_CPU = os.environ.get("MOLMETAL_ALLOW_CPU", "0") == "1"
# --- end Wf-190 env guard ----------------------------------------


# ---------------------------------------------------------------------------
# Coord label heuristic (mirrors smoke_chemprop_v2_mt_100).
# ---------------------------------------------------------------------------
_COORD_TABLE = {
    "Pt": 3,
    "Pd": 3,
    "Au": 3,
    "Ru": 4,
    "Ir": 4,
    "Os": 4,
    "Rh": 4,
    "Re": 4,
}


def _coord_label_for(metal_name: str) -> int:
    return _COORD_TABLE.get(metal_name, 3)


# ---------------------------------------------------------------------------
# Pre-featurisation cache (one-shot per script run).
# ---------------------------------------------------------------------------
def prefeaturize_all_smiles(
    smiles_list: List[str],
) -> Tuple[List, int, int]:
    """Featurize every SMILES exactly once.

    Returns ``(mols, atom_fdim, bond_fdim)`` where ``mols[i]`` is either
    a :class:`MolGraph` or ``None`` (when featurization failed).  The
    featurizer is chemprop v2's ``SimpleMoleculeMolGraphFeaturizer``.
    """
    ftr = SimpleMoleculeMolGraphFeaturizer()
    out: List[Optional[MolGraph]] = []
    atom_fdim = -1
    bond_fdim = -1
    # First pass — featurize everything; capture fdim from the first
    # successful parse.
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
            out.append(mg)
        except Exception:
            # 1-atom / 0-bond dummy MolGraph so BatchMolGraph keeps shape.
            if atom_fdim < 0:
                # Use a placeholder 'C' to read fdim.
                tmp_mg = ftr(Chem.MolFromSmiles("C"))
                atom_fdim = int(tmp_mg.V.shape[1])
                bond_fdim = int(tmp_mg.E.shape[1])
                placeholder_V = np.zeros((1, atom_fdim), dtype=np.single)
                placeholder_E = np.zeros((0, bond_fdim), dtype=np.single)
                placeholder_ei = np.zeros((2, 0), dtype=int)
                placeholder_rev = np.zeros((0,), dtype=int)
            assert placeholder_V is not None
            out.append(
                MolGraph(
                    placeholder_V.copy(),
                    placeholder_E.copy(),
                    placeholder_ei.copy(),
                    placeholder_rev.copy(),
                )
            )
    return out, atom_fdim, bond_fdim


# ---------------------------------------------------------------------------
# Forward helper — pre-built MolGraphs → chemprop encoder output.
# ---------------------------------------------------------------------------
def batch_molgraphs(mols: List[Optional[MolGraph]]) -> BatchMolGraph:
    """Build a chemprop :class:`BatchMolGraph` from a list of MolGraphs.

    Skips ``None`` entries defensively (shouldn't happen if the cache
    was built by :func:`prefeaturize_all_smiles`).
    """
    valid = [m for m in mols if m is not None]
    return BatchMolGraph(valid)


def batch_molgraphs_to(
    mols: List[Optional[MolGraph]], device: torch.device
) -> BatchMolGraph:
    """Build + move to device in one step (chemprop's ``to`` is in-place).

    :meth:`BatchMolGraph.to` mutates in place and returns ``None``; this
    helper returns the moved object so call sites can chain.
    """
    bmg = batch_molgraphs(mols)
    bmg.to(device)
    return bmg


# ---------------------------------------------------------------------------
# Per-(fold, seed) result container.
# ---------------------------------------------------------------------------
@dataclass
class LOMORun:
    fold: str
    seed: int
    n_train: int
    n_test: int
    pos_rate_train: float
    pos_rate_test: float
    test_auc: float
    elapsed_s: float
    epoch_count: int
    status: str  # "ok" | "timeout" | "error"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Single training run.
# ---------------------------------------------------------------------------
def run_one(
    model_factory,
    train_mols: List,
    train_metals: np.ndarray,
    train_active: np.ndarray,
    train_metal_ids: np.ndarray,
    test_mols: List,
    test_metals: np.ndarray,
    test_active: np.ndarray,
    test_metal_ids: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    held_out: str,
    seed: int,
    time_budget_s: Optional[float] = None,
    atom_fdim: int = 72,
    bond_fdim: int = 14,
    use_bf16: bool = True,
) -> LOMORun:
    """Train one (fold, seed) combination and return :class:`LOMORun`."""
    # Wf-190 P1: BF16 autocast wraps the chemprop v2 encoder forward when
    # CUDA is active and --bf16=True.  Blackwell sm_120 runs BF16 matmul
    # ~3x faster than FP32.  CPU paths are no-ops (autocast requires CUDA).
    use_amp = bool(use_bf16) and device.type == "cuda"
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model_factory()
    # Eagerly build the lazy chemprop encoder so message_passing / agg
    # are non-None BEFORE moving to GPU.  Building post-.to(device)
    # leaves the new submodules on CPU because nn.Module.to() only
    # moves registered parameters/buffers/submodules present at call
    # time — lazy-built attrs are skipped.
    model.encoder._ensure_backbone(atom_fdim, bond_fdim)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = ChempropV2MTLoss(n_coord_classes=5).to(device)
    bce_per_metal = nn.BCEWithLogitsLoss()

    n_train = len(train_mols)
    n_test = len(test_mols)

    # Pre-build y_true dict for each metal head.
    y_true_all: Dict[str, np.ndarray] = {}
    for m in model.metals:
        # Per-row binary target = active if row's metal == m else 0.
        y_true_all[m] = np.where(
            train_metals == m, train_active.astype(np.float32), 0.0
        ).astype(np.float32)
    coord_true_train = np.array(
        [_coord_label_for(m) for m in train_metals], dtype=np.int64
    )

    # Index permutation for batching.
    indices = np.arange(n_train)
    t_start = time.time()
    epoch_done = 0
    timed_out = False
    last_train_loss = float("nan")
    for epoch in range(epochs):
        if time_budget_s is not None and (time.time() - t_start) > time_budget_s:
            logger.warning(
                "[fold=%s seed=%d] timeout at epoch=%d (elapsed=%.1fs)",
                held_out,
                seed,
                epoch,
                time.time() - t_start,
            )
            timed_out = True
            break
        epoch_done = epoch + 1
        np.random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0
        model.train()
        for start in range(0, n_train, batch_size):
            batch_idx = indices[start : start + batch_size]
            bmols = [train_mols[i] for i in batch_idx]
            b_metal_ids = torch.from_numpy(train_metal_ids[batch_idx]).long().to(device)
            bmg = batch_molgraphs(bmols).to(device)

            # Build per-metal target dict for this batch.
            y_true_b: Dict[str, torch.Tensor] = {}
            for m in model.metals:
                y_true_b[m] = torch.from_numpy(y_true_all[m][batch_idx]).to(device)
            b_coord = torch.from_numpy(coord_true_train[batch_idx]).long().to(device)

            # Direct encoder forward using the pre-built BatchMolGraph.
            bmg = batch_molgraphs_to(bmols, device)
            # Wf-190 P1: wrap chemprop v2 message_passing in BF16 autocast
            # on CUDA.  No-op when use_amp is False (CPU or --no-bf16).
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    H_v = model.encoder.message_passing(bmg)  # (sum_N, hidden_dim)
                    H = model.encoder.agg(H_v, bmg.batch)  # (B, hidden_dim)
                    e = model.metal_embed(b_metal_ids)  # (B, metal_embed_dim)
                    z = torch.cat([H, e], dim=-1)  # (B, head_in_dim)
            else:
                H_v = model.encoder.message_passing(bmg)
                H = model.encoder.agg(H_v, bmg.batch)
                e = model.metal_embed(b_metal_ids)
                z = torch.cat([H, e], dim=-1)

            # Per-metal BCE losses (chemprop's BCELoss expects (B, 1)).
            total = z.new_zeros(())
            for metal, head in model.metal_heads.items():
                logit_1 = head.train_step(z)  # (B, 1)
                tgt = y_true_b[metal].unsqueeze(-1).float()  # (B, 1)
                l = bce_per_metal(logit_1, tgt)
                total = total + l
            # Coord CE.
            logit_c = model.coord_head(z)
            total = total + 0.3 * nn.functional.cross_entropy(logit_c, b_coord)

            optim.zero_grad()
            total.backward()
            optim.step()
            total_loss += float(total.item()) * len(batch_idx)
            n_batches += len(batch_idx)
        last_train_loss = total_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                "[fold=%s seed=%d] epoch=%d/%d loss=%.4f elapsed=%.1fs",
                held_out,
                seed,
                epoch,
                epochs,
                last_train_loss,
                time.time() - t_start,
            )

    # Test-time scoring — compute per-row active probability via sigmoid
    # on the held-out metal's head.
    if not timed_out and n_test > 0:
        model.eval()
        scores: List[float] = []
        labels: List[int] = []
        with torch.no_grad():
            for start in range(0, n_test, batch_size):
                bmols = test_mols[start : start + batch_size]
                b_metal_ids = (
                    torch.from_numpy(test_metal_ids[start : start + batch_size])
                    .long()
                    .to(device)
                )
                bmg = batch_molgraphs_to(bmols, device)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        H_v = model.encoder.message_passing(bmg)
                        H = model.encoder.agg(H_v, bmg.batch)
                        e = model.metal_embed(b_metal_ids)
                        z = torch.cat([H, e], dim=-1)
                        logit_1 = model.metal_heads[held_out].train_step(z)  # (B, 1)
                else:
                    H_v = model.encoder.message_passing(bmg)
                    H = model.encoder.agg(H_v, bmg.batch)
                    e = model.metal_embed(b_metal_ids)
                    z = torch.cat([H, e], dim=-1)
                    logit_1 = model.metal_heads[held_out].train_step(z)  # (B, 1)
                p = torch.sigmoid(logit_1.squeeze(-1).float()).cpu().numpy()
                scores.extend(float(x) for x in p)
                labels.extend(int(x) for x in test_active[start : start + batch_size])
        try:
            from sklearn.metrics import roc_auc_score

            if len(set(labels)) < 2:
                test_auc = float("nan")
            else:
                test_auc = float(roc_auc_score(labels, scores))
        except Exception as exc:
            logger.warning("AUC computation failed: %s", exc)
            test_auc = float("nan")
    else:
        test_auc = float("nan")
        labels = []

    pos_train = float(np.mean(train_active)) if n_train else 0.0
    pos_test = float(np.mean(test_active)) if n_test else 0.0
    elapsed = time.time() - t_start
    status = "ok"
    if timed_out:
        status = "timeout"
    return LOMORun(
        fold=held_out,
        seed=seed,
        n_train=n_train,
        n_test=n_test,
        pos_rate_train=pos_train,
        pos_rate_test=pos_test,
        test_auc=test_auc,
        elapsed_s=elapsed,
        epoch_count=epoch_done,
        status=status,
    )


# ---------------------------------------------------------------------------
# Build model factory (closure so each (fold, seed) gets fresh weights).
# ---------------------------------------------------------------------------
def make_model_factory(hidden_dim: int, depth: int):
    def _factory():
        return ChempropV2MTModel(
            hidden_dim=hidden_dim,
            depth=depth,
            n_metals=5,
            n_coord_classes=5,
        )

    return _factory


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metals", nargs="+", default=list(METALS_5))
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--time-budget-s", type=float, default=3600.0)
    p.add_argument(
        "--quick", action="store_true", help="1 metal x 1 seed x 2 epochs (smoke)."
    )
    # P1 FIX (Wf-190): add --bf16, --num-workers, --pin-memory flags so the
    # chemprop v2 retrain scripts opt into Blackwell sm_120 BF16 autocast
    # (~3x FP32 throughput).  Mirrors wf174_chemprop_v2_drugood_real.py
    # lines 583-607 (P1 Wf-187 patch).  No-op on CPU.
    p.add_argument(
        "--bf16",
        dest="bf16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Wrap chemprop v2 encoder.message_passing() in "
            "torch.amp.autocast('cuda', dtype=torch.bfloat16) when CUDA is "
            "active (default: True). Pass --no-bf16 to disable. No-op on CPU."
        ),
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers (default: 4). Currently logged; the manual "
        "range-based loop at line ~330 is the next-migration target.",
    )
    p.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DataLoader pin_memory (default: True). Same caveat as --num-workers.",
    )
    args = p.parse_args(argv)

    if args.quick:
        # Need >= 2 metals for LeaveOneGroupOut; keep 2 for smoke speed.
        args.metals = args.metals[:2]
        args.seeds = args.seeds[:1]
        args.epochs = 2
        args.time_budget_s = 180.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        # Wf-190 env guard: only auto-fall-back to CPU if the script-level
        # allow-CPU flag is set OR --bf16 is explicitly disabled (otherwise
        # the user explicitly asked for GPU and we should fail fast rather
        # than silently burn hours on a CPU box).
        if args.bf16 and not _MOLMETAL_ALLOW_CPU:
            raise RuntimeError(
                "[wf156] CUDA requested (--device cuda) but torch.cuda.is_available() "
                "is False and --bf16 is enabled.  Either:\n"
                "  -1- re-run on a CUDA-enabled host, OR\n"
                "  -2- pass --no-bf16 --device cpu for CPU fallback, OR\n"
                "  -3- set MOLMETAL_ALLOW_CPU=1 to allow silent CPU fallback "
                "(Wf-190 env guard)."
            )
        logger.warning(
            "CUDA requested but torch.cuda.is_available() is False; falling back to CPU. "
            "Set MOLMETAL_ALLOW_CPU=0 + --no-bf16 for fail-fast behaviour.",
        )
        device = torch.device("cpu")
    logger.info(
        "device=%s bf16=%s num_workers=%d pin_memory=%s metals=%s seeds=%s epochs=%d hidden=%d depth=%d",
        device,
        args.bf16,
        args.num_workers,
        args.pin_memory,
        args.metals,
        args.seeds,
        args.epochs,
        args.hidden,
        args.depth,
    )

    # ------------------------------------------------------------------
    # Load + filter MetalCytoToxDB to the 5 LOMO metals.
    # ------------------------------------------------------------------
    from molmetal.data.metal_cytotox import load_metal_cytotox

    df_all = load_metal_cytotox()
    df = df_all[df_all["metal"].isin(args.metals)].reset_index(drop=True)
    if df.empty:
        logger.error("no rows after metal filter; aborting.")
        return 1
    logger.info(
        "filtered MetalCytoToxDB: %d rows across metals=%s",
        len(df),
        sorted(set(df["metal"].tolist())),
    )

    smiles_all = df["smiles"].astype(str).tolist()
    metals_all = df["metal"].astype(str).to_numpy()
    active_all = df["active"].astype(int).to_numpy()

    # ------------------------------------------------------------------
    # Pre-featurize ALL SMILES once (CPU).
    # ------------------------------------------------------------------
    logger.info(
        "pre-featurizing %d SMILES via chemprop v2 SimpleMoleculeMolGraphFeaturizer …",
        len(smiles_all),
    )
    t0 = time.time()
    mols_all, atom_fdim, bond_fdim = prefeaturize_all_smiles(smiles_all)
    logger.info(
        "pre-featurized %d SMILES in %.1fs (atom_fdim=%d bond_fdim=%d)",
        len(mols_all),
        time.time() - t0,
        atom_fdim,
        bond_fdim,
    )

    # Build metal_id tensor (must match model.metals ordering).
    metal_id_lookup = {m: i for i, m in enumerate(SUPPORTED_METALS)}
    # Use only metals that the model knows about — subset rows whose
    # metal is in the model's SUPPORTED_METALS.
    keep_mask = np.isin(metals_all, list(SUPPORTED_METALS))
    if not keep_mask.all():
        n_drop = int((~keep_mask).sum())
        logger.warning(
            "dropping %d rows with metals outside %s", n_drop, SUPPORTED_METALS
        )
        keep_idx = np.where(keep_mask)[0]
        mols_all = [mols_all[i] for i in keep_idx]
        metals_all = metals_all[keep_idx]
        active_all = active_all[keep_idx]
    metal_ids_all = np.array([metal_id_lookup[m] for m in metals_all], dtype=np.int64)

    # ------------------------------------------------------------------
    # LOMO split via sklearn LeaveOneGroupOut.
    # ------------------------------------------------------------------
    logo = LeaveOneGroupOut()
    groups = metals_all
    splits = list(logo.split(np.arange(len(mols_all)), groups=groups))
    fold_to_split: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for train_idx, test_idx in splits:
        # The unique group in test_idx is the held-out metal.
        held = str(np.unique(metals_all[test_idx])[0])
        fold_to_split[held] = (train_idx, test_idx)
    logger.info(
        "LOMO splits: %s",
        {m: (len(t), len(te)) for m, (t, te) in fold_to_split.items()},
    )

    # ------------------------------------------------------------------
    # Sweep.
    # ------------------------------------------------------------------
    model_factory = make_model_factory(args.hidden, args.depth)
    time_per_run = args.time_budget_s / max(len(args.metals) * len(args.seeds), 1)
    logger.info("time budget per (fold,seed) = %.1fs", time_per_run)

    runs: List[LOMORun] = []
    sweep_t0 = time.time()
    for fold in args.metals:
        if fold not in fold_to_split:
            logger.warning("fold=%s missing from LOMO split; skip", fold)
            continue
        train_idx, test_idx = fold_to_split[fold]
        for seed in args.seeds:
            # Time slice — leave 10% headroom.
            slice_budget = time_per_run * 0.9
            try:
                run = run_one(
                    model_factory=model_factory,
                    train_mols=[mols_all[i] for i in train_idx],
                    train_metals=metals_all[train_idx],
                    train_active=active_all[train_idx],
                    train_metal_ids=metal_ids_all[train_idx],
                    test_mols=[mols_all[i] for i in test_idx],
                    test_metals=metals_all[test_idx],
                    test_active=active_all[test_idx],
                    test_metal_ids=metal_ids_all[test_idx],
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                    held_out=fold,
                    seed=int(seed),
                    time_budget_s=slice_budget,
                    atom_fdim=atom_fdim,
                    bond_fdim=bond_fdim,
                    use_bf16=args.bf16,
                )
            except Exception as exc:
                logger.error("fold=%s seed=%d FAILED: %s", fold, seed, exc)
                logger.debug(traceback.format_exc())
                run = LOMORun(
                    fold=fold,
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
                    elapsed_s=0.0,
                    epoch_count=0,
                    status="error",
                    error=str(exc),
                )
            runs.append(run)
            logger.info(
                "fold=%s seed=%d test_auc=%.4f epochs=%d elapsed=%.1fs status=%s",
                run.fold,
                run.seed,
                run.test_auc,
                run.epoch_count,
                run.elapsed_s,
                run.status,
            )
            if (time.time() - sweep_t0) > args.time_budget_s:
                logger.warning(
                    "wall budget %.1fs exceeded; aborting sweep.", args.time_budget_s
                )
                break
        if (time.time() - sweep_t0) > args.time_budget_s:
            break

    total_min = (time.time() - sweep_t0) / 60.0
    ran_cleanly = all(r.status == "ok" for r in runs)
    ran_partially = bool(runs) and not ran_cleanly

    # ------------------------------------------------------------------
    # Aggregate.
    # ------------------------------------------------------------------
    per_metal_summary: Dict[str, Dict[str, float]] = {}
    for fold in args.metals:
        per_fold = [r for r in runs if r.fold == fold]
        if not per_fold:
            continue
        aucs = [
            r.test_auc
            for r in per_fold
            if r.status == "ok" and not np.isnan(r.test_auc)
        ]
        per_metal_summary[fold] = {
            "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "std": float(np.std(aucs)) if aucs else 0.0,
            "n_train": int(per_fold[0].n_train),
            "n_test": int(per_fold[0].n_test),
            "n_seeds_ok": len(aucs),
            "n_seeds_total": len(per_fold),
            "per_seed_auc": {str(r.seed): r.test_auc for r in per_fold},
        }

    honest_framing_note = (
        f"Wf-156 chemprop v2 LOMO sweep ran with hidden_dim={args.hidden}, "
        f"depth={args.depth} (smoke-test size — production is 300/3), "
        f"{len(runs)} (fold, seed) runs in {total_min:.2f} min wall.  "
        f"Partial if any run hit the per-slice timeout or errored — see "
        f"`per_metal_summary[fold].n_seeds_ok`.  AUCs are MEASURED on the "
        f"test fold (held-out metal) using sklearn roc_auc_score on the "
        f"sigmoid of the chemprop per-metal head's (B, 1) logit.  "
        f"NO Wasserstein transfer, NO L2 guard, NO LR scheduler — pure "
        f"chemprop v2 BCE + coord CE auxiliary.  This sweep is a baseline "
        f"for TODO-148 §5 (paper-grade chemprop v2 retrain) and is NOT "
        f"comparable to the production chemprop v2 retrain at hidden_dim=300."
    )

    out = {
        "ran": ran_cleanly,
        "partial": ran_partially,
        "per_metal_summary": per_metal_summary,
        "total_min": float(total_min),
        "json_path": str(out_path),
        "honest_framing_note": honest_framing_note,
        "config": {
            "metals": list(args.metals),
            "seeds": list(args.seeds),
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "device": str(device),
            "time_budget_s": float(args.time_budget_s),
        },
        "runs": [
            {
                "fold": r.fold,
                "seed": r.seed,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "pos_rate_train": r.pos_rate_train,
                "pos_rate_test": r.pos_rate_test,
                "test_auc": r.test_auc,
                "elapsed_s": r.elapsed_s,
                "epoch_count": r.epoch_count,
                "status": r.status,
                "error": r.error,
            }
            for r in runs
        ],
    }

    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2)
    logger.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
