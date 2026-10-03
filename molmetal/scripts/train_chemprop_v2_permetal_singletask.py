"""Per-metal single-task chemprop v2 training (no multi-task, no shared heads).

Wf-170 / TODO-170 honest re-run: train one chemprop v2 BinaryClassificationFFN
per (metal, seed) on **only that metal's data**, hold out 20% stratified for
in-domain validation. Use chemprop v2 default backbone (hidden=300, depth=3)
to match the Wf-124 P3 head-to-head in-domain protocol (TODO-154 §2:
Ru 0.909, Ir 0.885, Os 0.889, Rh 0.863, Re 0.882).

Outputs (per (metal, seed)):
    molmetal/reports/wf170_permetal_singletask/{metal}_seed{N}.json
        {metal, seed, n_train, n_val, n_test, pos_rate_train, pos_rate_test,
         in_domain_test_auc, elapsed_s, epoch_count, status, ckpt_path}

Outputs (sweep aggregate):
    molmetal/reports/wf170_permetal_singletask/sweep_summary.json
        per_metal_seed_auc / per_metal_summary / mean_auc_across_metals

Usage (default 5-metal x 5-seed x 50-epoch, 7200s wall budget):
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/train_chemprop_v2_permetal_singletask.py

Honest framing per party-twenty-spirit §3.5:
    1. Backbone uses chemprop v2 defaults (hidden=300, depth=3) — matches the
       Wf-124 P3 in-domain protocol exactly (TODO-154 §1, 14 paired runs).
    2. Per-metal single-task (no shared heads, no multi-task BCE, no coord
       auxiliary) — collapses to one BinaryClassificationFFN per metal.
    3. Train/val split 80/20 stratified by active; val/test = same 20% fold.
       No LOMO, no cross-metal transfer. This is the in-domain benchmark.
    4. Wall budget 7200s = 2 hr; 25 runs total.
    5. NO Wasserstein / NO L2 guard / NO shared encoder / NO coord head.
       This is the cleanest apples-to-apples single-task per-metal baseline.
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
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from chemprop.data import BatchMolGraph
from chemprop.data.molgraph import MolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.nn import AtomMessagePassing, BinaryClassificationFFN, SumAggregation
from torch.optim.lr_scheduler import LambdaLR

from molmetal.data.metal_cytotox import load_metal_cytotox


# ---------------------------------------------------------------------------
# Defaults — chemprop v2 backbone @ defaults (hidden=300, depth=3)
# ---------------------------------------------------------------------------
METALS_5: Tuple[str, ...] = ("Ru", "Ir", "Os", "Rh", "Re")
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_EPOCHS: int = 50
DEFAULT_HIDDEN: int = 300  # chemprop v2 default
DEFAULT_DEPTH: int = 3  # chemprop v2 default
DEFAULT_BATCH: int = 64
DEFAULT_LR: float = 1e-3

DEFAULT_OUT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf170_permetal_singletask"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("wf170_permetal")
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
# Pre-featurise ALL SMILES once (CPU).
# ---------------------------------------------------------------------------
def prefeaturize_all_smiles(smiles_list: List[str]) -> Tuple[List, int, int]:
    """Featurise every SMILES once. Returns (mols, atom_fdim, bond_fdim).

    Each entry in mols is a chemprop v2 MolGraph or a 1-atom placeholder when
    RDKit parse failed (keeps BatchMolGraph shape stable).
    """
    ftr = SimpleMoleculeMolGraphFeaturizer()
    out: List[Optional[MolGraph]] = []
    placeholder_V = placeholder_E = placeholder_ei = placeholder_rev = None
    atom_fdim = -1
    bond_fdim = -1
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
            if atom_fdim < 0:
                tmp = ftr(Chem.MolFromSmiles("C"))
                atom_fdim = int(tmp.V.shape[1])
                bond_fdim = int(tmp.E.shape[1])
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
# Single (metal, seed) run.
# ---------------------------------------------------------------------------
@dataclass
class PerMetalRun:
    metal: str
    seed: int
    n_train: int
    n_val: int
    pos_rate_train: float
    pos_rate_test: float
    in_domain_test_auc: float
    elapsed_s: float
    epoch_count: int
    status: str
    ckpt_path: Optional[str] = None
    error: Optional[str] = None


def build_backbone(atom_fdim: int, bond_fdim: int, hidden_dim: int, depth: int):
    """Build a fresh chemprop v2 message passing + aggregation backbone."""
    mp = AtomMessagePassing(d_v=atom_fdim, d_e=bond_fdim, d_h=hidden_dim, depth=depth)
    agg = SumAggregation()
    return mp, agg


def run_one(
    smiles_subset: List[str],
    active_subset: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    atom_fdim: int,
    bond_fdim: int,
    hidden_dim: int,
    depth: int,
    ckpt_out: Optional[Path] = None,
    use_bf16: bool = True,
) -> PerMetalRun:
    """Train one (metal, seed) single-task chemprop v2 classifier."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 80/20 stratified split.
    idx = np.arange(len(smiles_subset))
    idx_train, idx_val = train_test_split(
        idx,
        test_size=0.20,
        random_state=seed,
        stratify=active_subset.astype(int),
    )
    n_train = len(idx_train)
    n_val = len(idx_val)
    pos_rate_train = float(np.mean(active_subset[idx_train]))
    pos_rate_test = float(np.mean(active_subset[idx_val]))

    # Pre-featurise subset (small — subset is per-metal only).
    mols_subset, atom_fdim_v, bond_fdim_v = prefeaturize_all_smiles(smiles_subset)
    assert atom_fdim_v == atom_fdim
    assert bond_fdim_v == bond_fdim

    # Build backbone + classifier head (single-task, n_tasks=1).
    mp, agg = build_backbone(atom_fdim, bond_fdim, hidden_dim, depth)
    head = BinaryClassificationFFN(
        n_tasks=1,
        input_dim=mp.output_dim,
        hidden_dim=hidden_dim,
        n_layers=2,
        dropout=0.0,
    )
    mp = mp.to(device)
    agg = agg.to(device)
    head = head.to(device)
    # chemprop v2 MPNN default: Adam + Noam-like scheduler
    # (warmup_epochs=2, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4).
    init_lr = 1e-4
    max_lr = 1e-3
    final_lr = 1e-4
    warmup_epochs = 2
    n_train_batches = (n_train + batch_size - 1) // batch_size
    warmup_steps = warmup_epochs * n_train_batches
    total_steps = epochs * n_train_batches
    cooldown_steps = max(total_steps - warmup_steps, 1)
    optim = torch.optim.Adam(
        list(mp.parameters()) + list(agg.parameters()) + list(head.parameters()),
        lr=init_lr,
    )

    def lr_lambda(step: int):
        if step < warmup_steps:
            warmup_factor = (max_lr - init_lr) / max(warmup_steps, 1)
            return step * warmup_factor / init_lr + 1.0
        elif step < warmup_steps + cooldown_steps:
            cooldown_factor = (final_lr / max_lr) ** (1.0 / cooldown_steps)
            return (max_lr * (cooldown_factor ** (step - warmup_steps))) / init_lr
        else:
            return final_lr / init_lr

    scheduler = LambdaLR(optim, lr_lambda)
    bce = nn.BCEWithLogitsLoss()

    train_active_t = torch.from_numpy(active_subset[idx_train]).float().to(device)

    # Wf-190 P1: BF16 autocast wraps the chemprop v2 message_passing forward
    # pass when CUDA is active.  Blackwell sm_120 BF16 matmul is ~3x FP32.
    # No-op on CPU or when --no-bf16 is set.
    use_amp = bool(use_bf16) and device.type == "cuda"

    t0 = time.time()
    epoch_done = 0
    for epoch in range(epochs):
        epoch_done = epoch + 1
        order = np.random.permutation(n_train)
        mp.train()
        agg.train()
        head.train()
        total_loss = 0.0
        nb = 0
        for s in range(0, n_train, batch_size):
            bi = order[s : s + batch_size]
            bmols = [mols_subset[int(i)] for i in bi]
            bmg = BatchMolGraph(bmols)
            bmg.to(device)
            tgt = train_active_t[torch.from_numpy(bi).long().to(device)].unsqueeze(-1)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    H_v = mp(bmg)
                    H = agg(H_v, bmg.batch)
                    logit = head.train_step(H)  # (B, 1)
                    loss = bce(logit, tgt)
            else:
                H_v = mp(bmg)
                H = agg(H_v, bmg.batch)
                logit = head.train_step(H)  # (B, 1)
                loss = bce(logit, tgt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step()
            total_loss += float(loss.item()) * len(bi)
            nb += len(bi)
        last_loss = total_loss / max(nb, 1)
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                "[seed=%d] epoch=%d/%d loss=%.4f elapsed=%.1fs",
                seed,
                epoch,
                epochs,
                last_loss,
                time.time() - t0,
            )

    # Test-time scoring on the val holdout.
    mp.eval()
    agg.eval()
    head.eval()
    scores: List[float] = []
    labels: List[int] = []
    with torch.no_grad():
        for s in range(0, n_val, batch_size):
            bi = idx_val[s : s + batch_size]
            bmols = [mols_subset[int(i)] for i in bi]
            bmg = BatchMolGraph(bmols)
            bmg.to(device)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    H_v = mp(bmg)
                    H = agg(H_v, bmg.batch)
                    logit = head.train_step(H)
                    p = torch.sigmoid(logit.squeeze(-1).float()).cpu().numpy()
            else:
                H_v = mp(bmg)
                H = agg(H_v, bmg.batch)
                logit = head.train_step(H)
                p = torch.sigmoid(logit.squeeze(-1)).cpu().numpy()
            scores.extend(float(x) for x in p)
            labels.extend(int(x) for x in active_subset[bi])
    if len(set(labels)) < 2:
        test_auc = float("nan")
    else:
        test_auc = float(roc_auc_score(labels, scores))

    elapsed = time.time() - t0

    # Save checkpoint.
    if ckpt_out is not None:
        ckpt_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mp_state": mp.state_dict(),
                "agg_state": agg.state_dict(),
                "head_state": head.state_dict(),
                "metal": "single_task",  # the metal is implicit from filename
                "seed": seed,
                "atom_fdim": atom_fdim,
                "bond_fdim": bond_fdim,
                "hidden_dim": hidden_dim,
                "depth": depth,
                "epochs": epochs,
                "in_domain_test_auc": test_auc,
            },
            str(ckpt_out),
        )

    return PerMetalRun(
        metal="?",  # caller fills in
        seed=seed,
        n_train=n_train,
        n_val=n_val,
        pos_rate_train=pos_rate_train,
        pos_rate_test=pos_rate_test,
        in_domain_test_auc=test_auc,
        elapsed_s=elapsed,
        epoch_count=epoch_done,
        status="ok",
        ckpt_path=str(ckpt_out) if ckpt_out else None,
    )


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
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--time-budget-s", type=float, default=7200.0)
    p.add_argument(
        "--quick", action="store_true", help="2 metal x 1 seed x 2 epochs (smoke)."
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
            "Wrap chemprop v2 message_passing() in "
            "torch.amp.autocast('cuda', dtype=torch.bfloat16) when CUDA is "
            "active (default: True). Pass --no-bf16 to disable. No-op on CPU."
        ),
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader num_workers (default: 4). Currently logged; the manual "
        "range-based loop is the next-migration target.",
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
        args.metals = args.metals[:2]
        args.seeds = args.seeds[:1]
        args.epochs = 2
        args.time_budget_s = 180.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        # Wf-190 env guard: only auto-fall-back to CPU if the script-level
        # allow-CPU flag is set OR --bf16 is explicitly disabled (otherwise
        # the user explicitly asked for GPU and we should fail fast rather
        # than silently burn hours on a CPU box).
        if args.bf16 and not _MOLMETAL_ALLOW_CPU:
            raise RuntimeError(
                "[wf170 per-metal] CUDA requested (--device cuda) but "
                "torch.cuda.is_available() is False and --bf16 is enabled.  Either:\n"
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

    # Load MetalCytoToxDB and filter to 5 metals.
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

    # Per-metal subset cache.
    per_metal_smiles: Dict[str, List[str]] = {}
    per_metal_active: Dict[str, np.ndarray] = {}
    for m in args.metals:
        sub = df[df["metal"] == m].reset_index(drop=True)
        per_metal_smiles[m] = sub["smiles"].astype(str).tolist()
        per_metal_active[m] = sub["active"].astype(int).to_numpy()
        logger.info(
            "metal=%s n_rows=%d pos_rate=%.4f",
            m,
            len(sub),
            float(per_metal_active[m].mean()),
        )

    # Pre-featurise a single tiny SMILES once to capture fdim.
    sample_smi = per_metal_smiles[args.metals[0]][:1]
    _, atom_fdim, bond_fdim = prefeaturize_all_smiles(sample_smi)
    logger.info(
        "atom_fdim=%d bond_fdim=%d (chemprop v2 SimpleMoleculeMolGraphFeaturizer)",
        atom_fdim,
        bond_fdim,
    )

    n_runs = len(args.metals) * len(args.seeds)
    time_per_run = args.time_budget_s / max(n_runs, 1)
    logger.info(
        "time budget per (metal,seed) = %.1fs for %d runs", time_per_run, n_runs
    )

    sweep_t0 = time.time()
    sweep_results: Dict[str, Dict[str, dict]] = {m: {} for m in args.metals}
    summary_records: List[PerMetalRun] = []

    for metal in args.metals:
        for seed in args.seeds:
            json_out = out_dir / f"{metal}_seed{seed}.json"
            ckpt_out = (
                PROJECT_ROOT
                / "molmetal"
                / "checkpoints"
                / "wf170_permetal_singletask"
                / f"{metal}_seed{seed}.pt"
            )
            # Time slice — leave 10% headroom.
            slice_budget = time_per_run * 0.9
            t_run = time.time()
            try:
                run = run_one(
                    smiles_subset=per_metal_smiles[metal],
                    active_subset=per_metal_active[metal],
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                    seed=seed,
                    atom_fdim=atom_fdim,
                    bond_fdim=bond_fdim,
                    hidden_dim=args.hidden,
                    depth=args.depth,
                    ckpt_out=ckpt_out,
                    use_bf16=args.bf16,
                )
                run.metal = metal
                if (time.time() - t_run) > slice_budget:
                    run.status = "timeout"
                logger.info(
                    "metal=%s seed=%d in_domain_test_auc=%.4f epochs=%d elapsed=%.1fs status=%s",
                    metal,
                    seed,
                    run.in_domain_test_auc,
                    run.epoch_count,
                    run.elapsed_s,
                    run.status,
                )
                rec = {
                    "metal": metal,
                    "seed": seed,
                    "n_train": run.n_train,
                    "n_val": run.n_val,
                    "pos_rate_train": run.pos_rate_train,
                    "pos_rate_test": run.pos_rate_test,
                    "in_domain_test_auc": run.in_domain_test_auc,
                    "elapsed_s": run.elapsed_s,
                    "epoch_count": run.epoch_count,
                    "status": run.status,
                    "ckpt_path": run.ckpt_path,
                }
                json_out.write_text(json.dumps(rec, indent=2))
                sweep_results[metal][str(seed)] = {
                    "auc": run.in_domain_test_auc,
                    "time_sec": run.elapsed_s,
                }
                summary_records.append(run)
            except Exception as exc:
                logger.error(
                    "metal=%s seed=%d failed: %s\n%s",
                    metal,
                    seed,
                    exc,
                    traceback.format_exc(),
                )
                rec = {
                    "metal": metal,
                    "seed": seed,
                    "status": "error",
                    "error": str(exc),
                }
                json_out.write_text(json.dumps(rec, indent=2))
                sweep_results[metal][str(seed)] = {
                    "auc": float("nan"),
                    "time_sec": float("nan"),
                }

    total_elapsed = time.time() - sweep_t0
    logger.info("sweep done in %.1fs", total_elapsed)

    # Compute per-metal summary.
    per_metal_summary: Dict[str, dict] = {}
    aucs_across_metals: List[float] = []
    for metal in args.metals:
        vals = [
            v["auc"]
            for v in sweep_results[metal].values()
            if v["auc"] == v["auc"]  # not NaN
        ]
        if vals:
            mean_auc = float(statistics.mean(vals))
            std_auc = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
            per_metal_summary[metal] = {
                "mean_auc": mean_auc,
                "std": std_auc,
                "n_seeds": len(vals),
            }
            aucs_across_metals.append(mean_auc)
        else:
            per_metal_summary[metal] = {
                "mean_auc": float("nan"),
                "std": float("nan"),
                "n_seeds": 0,
            }

    mean_auc_across_metals = (
        float(statistics.mean(aucs_across_metals))
        if aucs_across_metals
        else float("nan")
    )

    out_json = {
        "ran": True,
        "total_wall_min": total_elapsed / 60.0,
        "per_metal_seed_auc": sweep_results,
        "per_metal_summary": per_metal_summary,
        "mean_auc_across_metals": mean_auc_across_metals,
        "matches_in_domain_protocol": True,  # chemprop defaults hidden=300/depth=3 + 50 epochs + 80/20 stratified
        "honest_framing_note": (
            "Per-metal single-task chemprop v2 backbone at defaults (hidden=300, "
            "depth=3, 50 epochs), 80/20 stratified holdout per metal. NO multi-task, "
            "NO shared encoder across metals, NO Wasserstein reg, NO coord head. "
            "Apples-to-apples with the Wf-124 P3 in-domain protocol (TODO-154 §2: "
            "Ru 0.909, Ir 0.885, Os 0.889, Rh 0.863, Re 0.882 mean 0.886)."
        ),
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(out_json, indent=2))
    logger.info("wrote sweep_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
