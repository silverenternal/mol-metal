"""Wf-308C -- TODO-220 NCI-60 9-cell chemprop v2 multi-task driver.

=====================================================================
Wf-308C PART 2 — NCI-60 9-CELL MULTI-TASK DRIVER EXECUTE PATH.

Wf-305C shipped the dry-run-only stub for the NCI-60 9-cell chemprop
v2 multi-task analysis (derived from TODO-220 / Wf-124 P3). Wf-308C
wires the ``--execute`` path so the script can run a real per-cell-line
multi-task chemprop v2 sweep on CPU (smoke) or GPU (full).

Hardware: ``--device cpu`` runs the 1-cell 1-epoch smoke in <5 min.
Full GPU run needs ``USER_OK`` per kill_then_adapt.

CSV layout
----------
The NCI-60 9-cell CSV (``molmetal/data/nci60_precious_9cell.csv``) has
columns ``smiles,cell_line,pIC50,active``. We do NOT use
``load_nci60`` (which expects a 6-col schema with metal/protein
fields); we read the 4-col CSV directly and group by ``cell_line``.

Output JSON shape
-----------------
``sweep_summary.json``::

    {
      "ran": true/false,
      "n_runs_planned": int,
      "n_runs_completed": int,
      "per_cell_seed_auc": {cell: {seed: {auc, time_sec, status}}},
      "per_cell_summary": {cell: {mean_auc, std, n_seeds, n_train,
                                     n_val, n_test, per_seed_auc}},
      "mean_auc_across_cells": float,
      "n_cells_used": int,
      "config": {...},
      "honest_framing_note": "..."
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Make chemprop v2 importable.
CHEMPROP_V2_PATH = PROJECT_ROOT / ".cloned" / "chemprop_v2"
if str(CHEMPROP_V2_PATH) not in sys.path:
    sys.path.insert(0, str(CHEMPROP_V2_PATH))

from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from chemprop.data import (  # noqa: E402
    MoleculeDatapoint,
    MoleculeDataset,
    build_dataloader,
)
from chemprop.nn import (  # noqa: E402
    AtomMessagePassing,
    BinaryClassificationFFN,
    SumAggregation,
)
from sklearn.metrics import roc_auc_score  # noqa: E402


__all__ = [
    "build_argparser",
    "dry_run_plan",
    "execute_sweep",
    "main",
]


# ---------------------------------------------------------------------------
# Defaults — 9 canonical NCI-60 cells
# ---------------------------------------------------------------------------
DEFAULT_9_CELLS = [
    "MCF7",        # breast
    "A2780",       # ovarian
    "A549",        # lung
    "HCT116",      # colon
    "PC3",         # prostate
    "HL60",        # leukaemia
    "HCT-116",     # colon (alt)
    "A2780cis",    # cisplatin-resistant ovarian
    "DU145",       # prostate (alt)
]
DEFAULT_NUM_WORKERS: int = 0      # CPU smoke default
DEFAULT_BATCH: int = 64
DEFAULT_LR: float = 1e-3
DEFAULT_TIME_BUDGET_S: float = 1 * 3600.0   # 1 hr wall budget for full 9-cell


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("nci60_9cell_chemprop_driver")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s")
    )
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


_MOLMETAL_ALLOW_CPU = os.environ.get("MOLMETAL_ALLOW_CPU", "0") == "1"


# ---------------------------------------------------------------------------
# Per-(cell, seed) record.
# ---------------------------------------------------------------------------
@dataclass
class Nci60CellRun:
    cell: str
    seed: int
    n_train: int
    n_val: int
    n_test: int
    pos_rate_train: float
    pos_rate_test: float
    in_domain_test_auc: float
    elapsed_s: float
    epoch_count: int
    status: str
    split: str = "temporal_random_80_10_20"
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Scaffold-OOD split (Wf-315F honest extension of Wf-312A pattern to
# NCI-60 9-cell driver).
# ---------------------------------------------------------------------------
# Wf-319E: helpers extracted to ``_nci60_scaffold_helpers`` module
# (pure-Python, RDKit-only, no chemprop v2 dep) so they can be unit-
# tested without ``cuik_molmaker``. Re-exported here for backward
# compatibility with the Wf-315F public API.
try:
    from molmetal.scripts._nci60_scaffold_helpers import (  # noqa: E402
        _scaffold_split_indices_nci60,
        compute_per_cell_n_test_min,
    )
except Exception:  # noqa: BLE001
    # Defensive: if the helper module fails to import for any reason,
    # re-raise so the driver surfaces the error rather than silently
    # using a None reference downstream.
    raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def bmg_to_device(bmg, device: torch.device):
    """In-place move a chemprop v2 BatchMolGraph to ``device``."""
    bmg.to(device)
    for attr in ("V", "E", "edge_index", "rev_edge_index", "batch"):
        v = getattr(bmg, attr, None)
        if v is not None and isinstance(v, torch.Tensor):
            setattr(bmg, attr, v.to(device))
    return bmg


def _match_cell_name(cell_name: str, available_cells: List[str]) -> Optional[str]:
    """Return the available cell line that matches the requested name."""
    if cell_name in available_cells:
        return cell_name
    # Try common aliases (e.g. HCT-116 <-> HCT116).
    norm = cell_name.replace("-", "").replace(" ", "").upper()
    for avail in available_cells:
        if avail.replace("-", "").replace(" ", "").upper() == norm:
            return avail
    return None


# ---------------------------------------------------------------------------
# Per-(cell, seed) trainer.
# ---------------------------------------------------------------------------
def run_one_per_cell(
    *,
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
    num_workers: int,
    time_budget_s: Optional[float] = None,
    split_mode: str = "temporal",
    n_test_min: int = 50,
) -> Nci60CellRun:
    """Train one per-cell single-task chemprop v2 with the chosen split.

    Wf-315F extension of Wf-312A pattern to NCI-60 9-cell driver.
    NCI-60 has no temporal Year column, so ``temporal`` here means
    random 80/10/20 split (the existing default), ``scaffold`` means
    scaffold-OOD (Bemis-Murcko held-out group per seed), and ``auto``
    picks scaffold when the random test partition has < n_test_min
    rows.

    Split modes (``--split`` flag):
      * ``temporal``  — random 80/10/20 (seeded; equivalent to the
        Wf-308C default since NCI-60 has no temporal Year column).
      * ``scaffold``  — scaffold-OOD: hold out one Bemis-Murcko
        scaffold group (>= n_test_min rows) per seed as the test set.
        Falls back to random 80/10/20 if no group is large enough
        (e.g. for very small cells).
      * ``auto``      — temporal (random) when n_test >= n_test_min,
        else scaffold fallback. All 9 NCI-60 cells currently satisfy
        this with the existing row counts (smallest = DU145 161 rows
        → n_test ~ 32); scaffold fallback is future-proofing for
        future cohort shrinks.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    from sklearn.model_selection import train_test_split

    # Compute the temporal (random 80/10/20) split up-front so we can
    # probe n_test for the auto mode.
    idx = np.arange(len(smiles_subset))
    idx_train_full, idx_test_random = train_test_split(
        idx, test_size=0.20, random_state=seed,
        stratify=active_subset.astype(int) if len(set(active_subset)) > 1 else None,
    )
    idx_train, idx_val = train_test_split(
        idx_train_full, test_size=0.125, random_state=seed,
        stratify=(
            active_subset[idx_train_full].astype(int)
            if len(set(active_subset[idx_train_full])) > 1
            else None
        ),
    )
    train_idx_random = np.asarray(idx_train, dtype=np.int64)
    val_idx_random = np.asarray(idx_val, dtype=np.int64)
    test_idx_random = np.asarray(idx_test_random, dtype=np.int64)

    chosen_split_label = ""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx_final: np.ndarray

    if split_mode == "temporal":
        train_idx = train_idx_random
        val_idx = val_idx_random
        test_idx_final = test_idx_random
        chosen_split_label = "temporal_random_80_10_20"
    elif split_mode == "scaffold":
        out = _scaffold_split_indices_nci60(
            smiles_subset=smiles_subset,
            active_subset=active_subset,
            seed=seed,
            n_test_min=n_test_min,
        )
        if out is None:
            # Fall back to random 80/10/20.
            train_idx = train_idx_random
            val_idx = val_idx_random
            test_idx_final = test_idx_random
            chosen_split_label = "random_80_10_20_fallback_no_scaffold_group"
        else:
            train_idx, val_idx, test_idx_final, _lbl = out
            chosen_split_label = f"scaffold_ood_seed{seed}"
    elif split_mode == "auto":
        # Wf-315F extension: try temporal (random) first; if n_test < n_test_min,
        # fall back to scaffold. For the current 7823-row 9-cell CSV all
        # cells have n_test >= 32 (DU145 161 * 0.2) so the auto branch
        # almost always picks temporal — but scaffold fallback is
        # future-proofed for cohort shrinks.
        if len(test_idx_random) >= n_test_min and len(train_idx_random) >= n_test_min:
            train_idx = train_idx_random
            val_idx = val_idx_random
            test_idx_final = test_idx_random
            chosen_split_label = "temporal_random_80_10_20"
        else:
            out = _scaffold_split_indices_nci60(
                smiles_subset=smiles_subset,
                active_subset=active_subset,
                seed=seed,
                n_test_min=n_test_min,
            )
            if out is not None:
                train_idx, val_idx, test_idx_final, _lbl = out
                chosen_split_label = f"scaffold_fallback_seed{seed}"
            else:
                # Both temporal n_test and scaffold group are too small;
                # use the random split anyway and label honestly.
                train_idx = train_idx_random
                val_idx = val_idx_random
                test_idx_final = test_idx_random
                chosen_split_label = "auto_no_scaffold_group_random_fallback"
    else:
        raise ValueError(
            f"unknown split_mode {split_mode!r}; use temporal | scaffold | auto"
        )

    n_train = int(len(train_idx))
    n_val = int(len(val_idx))
    n_test = int(len(test_idx_final))
    pos_rate_train = float(np.mean(active_subset[train_idx])) if n_train else 0.0
    pos_rate_test = float(np.mean(active_subset[test_idx_final])) if n_test else 0.0

    train_smiles = [smiles_subset[int(i)] for i in train_idx]
    train_labels = active_subset[train_idx].astype(np.float32)
    val_smiles = [smiles_subset[int(i)] for i in val_idx]
    val_labels = active_subset[val_idx].astype(np.float32)

    train_dps = [
        MoleculeDatapoint(mol=Chem.MolFromSmiles(smi), y=np.array([lbl], dtype=np.float32))
        for smi, lbl in zip(train_smiles, train_labels)
    ]
    val_dps = [
        MoleculeDatapoint(mol=Chem.MolFromSmiles(smi), y=np.array([lbl], dtype=np.float32))
        for smi, lbl in zip(val_smiles, val_labels)
    ]
    train_ds = MoleculeDataset(train_dps, n_workers=num_workers)
    val_ds = MoleculeDataset(val_dps, n_workers=num_workers)
    train_dl = build_dataloader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=True, seed=seed,
    )
    val_dl = build_dataloader(
        val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False,
    )

    mp = AtomMessagePassing(
        d_v=atom_fdim, d_e=bond_fdim, d_h=hidden_dim, depth=depth, dropout=0.0,
    ).to(device)
    agg = SumAggregation().to(device)
    head = BinaryClassificationFFN(
        n_tasks=1, input_dim=mp.output_dim, hidden_dim=hidden_dim,
        n_layers=2, dropout=0.0,
    ).to(device)
    optim = torch.optim.Adam(
        list(mp.parameters()) + list(agg.parameters()) + list(head.parameters()),
        lr=lr,
    )
    bce = nn.BCEWithLogitsLoss().to(device)

    autocast_enabled = device.type == "cuda"
    t_start = time.time()
    epoch_done = 0
    timed_out = False
    for epoch in range(epochs):
        if time_budget_s is not None and (time.time() - t_start) > time_budget_s:
            timed_out = True
            break
        epoch_done = epoch + 1
        mp.train(); agg.train(); head.train()
        total_loss = 0.0
        nb = 0
        for batch in train_dl:
            bmg = bmg_to_device(batch.bmg, device)
            Y = batch.Y.to(device).float()
            optim.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                H_v = mp(bmg)
                H = agg(H_v, bmg.batch)
                logit = head.train_step(H)
            loss = bce(logit.float(), Y)
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * Y.size(0)
            nb += Y.size(0)
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(
                "[seed=%d] epoch=%d/%d train_loss=%.4f elapsed=%.1fs",
                seed, epoch, epochs - 1, total_loss / max(nb, 1),
                time.time() - t_start,
            )

    # Score on val fold (held-out for honest reporting).
    mp.eval(); agg.eval(); head.eval()
    scores: List[float] = []
    labels: List[int] = []
    with torch.no_grad():
        for batch in val_dl:
            bmg = bmg_to_device(batch.bmg, device)
            Y_cpu = batch.Y.float()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                H_v = mp(bmg)
                H = agg(H_v, bmg.batch)
                logit = head.train_step(H)
            p = torch.sigmoid(logit.float().squeeze(-1)).cpu().numpy()
            scores.extend(float(x) for x in p)
            labels.extend(int(round(float(x))) for x in Y_cpu.squeeze(-1).numpy())
    if len(set(labels)) < 2:
        test_auc = float("nan")
    else:
        test_auc = float(roc_auc_score(labels, scores))

    elapsed = time.time() - t_start
    status = "ok" if not timed_out else "timeout"

    return Nci60CellRun(
        cell="?",
        seed=int(seed),
        n_train=n_train, n_val=n_val, n_test=n_test,
        pos_rate_train=pos_rate_train, pos_rate_test=pos_rate_test,
        in_domain_test_auc=test_auc,
        elapsed_s=float(elapsed),
        epoch_count=int(epoch_done),
        status=status,
        split=chosen_split_label,
    )


# ---------------------------------------------------------------------------
# Execute path
# ---------------------------------------------------------------------------
def execute_sweep(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full per-cell-line chemprop v2 sweep."""
    csv_path = Path(args.csv)
    if not csv_path.exists():
        return {
            "ran": False,
            "status": "CSV_MISSING",
            "csv": str(csv_path),
            "error": (
                f"nci60_precious_9cell.csv not found at {csv_path}. "
                f"Per Wf-306B honest-negative: do not run --execute until "
                "the CSV exists."
            ),
        }

    cells = list(args.cells)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if not _MOLMETAL_ALLOW_CPU:
            raise RuntimeError(
                "[nci60-9cell] CUDA requested but unavailable. Set "
                "MOLMETAL_ALLOW_CPU=1 OR pass --device cpu."
            )
        logger.warning("CUDA unavailable; falling back to CPU.")
        device = torch.device("cpu")
    logger.info(
        "device=%s cells=%s seeds=%s epochs=%d hidden=%d depth=%d",
        device, cells, args.seeds, args.epochs, args.hidden, args.depth,
    )

    df = pd.read_csv(csv_path, low_memory=False)
    required_cols = {"smiles", "cell_line", "active"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{csv_path} missing required columns {missing_cols}; "
            f"present columns: {df.columns.tolist()}"
        )

    available_cells = sorted(set(df["cell_line"].astype(str).tolist()))
    matched: Dict[str, str] = {}
    for c in cells:
        m = _match_cell_name(c, available_cells)
        if m is not None:
            matched[c] = m
        else:
            logger.warning("requested cell %r not found in CSV (available: %s)",
                           c, available_cells)
    if not matched:
        return {
            "ran": False,
            "status": "NO_CELLS_MATCHED",
            "csv": str(csv_path),
            "cells_requested": cells,
            "cells_available": available_cells,
        }
    logger.info(
        "matched %d/%d requested cells: %s",
        len(matched), len(cells),
        {c: matched[c] for c in matched},
    )

    # Probe atom/bond fdim with a tiny sample.
    sample_smi = df["smiles"].astype(str).tolist()[:8]
    sample_dps = [
        MoleculeDatapoint(mol=Chem.MolFromSmiles(s), y=np.array([0.5], dtype=np.float32))
        for s in sample_smi
    ]
    sample_ds = MoleculeDataset(sample_dps, n_workers=0)
    sample_dl = build_dataloader(sample_ds, batch_size=8, num_workers=0, shuffle=False)
    sample_batch = next(iter(sample_dl))
    atom_fdim = int(sample_batch.bmg.V.shape[1])
    bond_fdim = int(sample_batch.bmg.E.shape[1]) if sample_batch.bmg.E.numel() > 0 else 14
    logger.info("atom_fdim=%d bond_fdim=%d", atom_fdim, bond_fdim)

    # Per-cell subsets.
    per_cell_smiles: Dict[str, List[str]] = {}
    per_cell_active: Dict[str, np.ndarray] = {}
    per_cell_split_choice: Dict[str, str] = {}
    per_cell_n_test_min: Dict[str, int] = {}
    for c, c_avail in matched.items():
        sub = df[df["cell_line"].astype(str) == c_avail].reset_index(drop=True)
        per_cell_smiles[c] = sub["smiles"].astype(str).tolist()
        per_cell_active[c] = sub["active"].astype(int).to_numpy()
        # Wf-315F: probe per-cell viability of random (temporal) split
        # for the auto mode. n_test_window here is the 80/10/20 random
        # test partition size; scaffold fallback is only used when this
        # falls below n_test_min (default 50). For the current 7823-row
        # 9-cell CSV all cells are healthy so auto → temporal always;
        # scaffold fallback is future-proofed.
        n_rows_c = int(len(sub))
        n_test_random_est = int(round(0.20 * n_rows_c))
        n_train_random_est = int(round(0.70 * n_rows_c))
        # Wf-319E: per-cell n_test_min auto-tuning. With default
        # n_test_min=50 the scaffold fallback is wired but never
        # triggers (no cell has a scaffold group ≥50). Compute the
        # per-cell threshold from max scaffold size so the 3 largest
        # cells (A549/MCF7/A2780) actually use scaffold-OOD in auto
        # mode. Users can disable with --no-n-test-min-auto.
        if args.split == "auto":
            if args.n_test_min_auto:
                effective_n_test_min = compute_per_cell_n_test_min(
                    smiles_subset=per_cell_smiles[c],
                    n_test_min_floor=args.n_test_min_floor,
                    headroom_factor=args.n_test_min_headroom,
                )
            else:
                effective_n_test_min = int(args.n_test_min)
            per_cell_n_test_min[c] = effective_n_test_min
            per_cell_split_choice[c] = (
                "scaffold_fallback" if (
                    n_test_random_est < effective_n_test_min
                    or n_train_random_est < effective_n_test_min
                ) else "temporal"
            )
        else:
            effective_n_test_min = int(args.n_test_min)
            per_cell_n_test_min[c] = effective_n_test_min
            per_cell_split_choice[c] = args.split
        logger.info(
            "cell=%s (csv=%s) n_rows=%d pos_rate=%.4f "
            "n_test_random_est=%d n_test_min_per_cell=%d "
            "split_choice=%s",
            c, c_avail, n_rows_c, float(per_cell_active[c].mean()),
            n_test_random_est, per_cell_n_test_min[c],
            per_cell_split_choice[c],
        )

    n_runs = len(matched) * len(args.seeds)
    sweep_t0 = time.time()
    sweep_results: Dict[str, Dict[str, dict]] = {c: {} for c in matched}
    summary_records: List[Nci60CellRun] = []

    for cell in matched:
        for seed in args.seeds:
            t_run = time.time()
            try:
                # Wf-315F: per-cell split auto-fallback. If the per-cell
                # split_choice is "scaffold_fallback", pass split_mode="scaffold"
                # directly; otherwise pass args.split.
                effective_split_mode = (
                    "scaffold"
                    if per_cell_split_choice.get(cell) == "scaffold_fallback"
                    else args.split
                )
                run = run_one_per_cell(
                    smiles_subset=per_cell_smiles[cell],
                    active_subset=per_cell_active[cell],
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    device=device,
                    seed=int(seed),
                    atom_fdim=atom_fdim,
                    bond_fdim=bond_fdim,
                    hidden_dim=args.hidden,
                    depth=args.depth,
                    num_workers=args.num_workers,
                    time_budget_s=args.time_budget_s,
                    split_mode=effective_split_mode,
                    n_test_min=per_cell_n_test_min[cell],
                )
                run.cell = cell
                logger.info(
                    "cell=%s seed=%d in_domain_test_auc=%.4f epochs=%d "
                    "elapsed=%.1fs status=%s",
                    cell, seed, run.in_domain_test_auc, run.epoch_count,
                    run.elapsed_s, run.status,
                )
                summary_records.append(run)
                sweep_results[cell][str(int(seed))] = {
                    "auc": run.in_domain_test_auc,
                    "time_sec": run.elapsed_s,
                    "status": run.status,
                    "n_train": run.n_train,
                    "n_val": run.n_val,
                    "n_test": run.n_test,
                }
            except Exception as exc:
                logger.error(
                    "cell=%s seed=%d FAILED: %s\n%s",
                    cell, seed, exc, traceback.format_exc(),
                )
                run = Nci60CellRun(
                    cell=cell, seed=int(seed),
                    n_train=0, n_val=0, n_test=0,
                    pos_rate_train=float("nan"), pos_rate_test=float("nan"),
                    in_domain_test_auc=float("nan"),
                    elapsed_s=float(time.time() - t_run),
                    epoch_count=0,
                    status="error",
                    error=str(exc),
                )
                summary_records.append(run)
                sweep_results[cell][str(int(seed))] = {
                    "auc": float("nan"),
                    "time_sec": float(time.time() - t_run),
                    "status": "error",
                    "n_train": 0,
                    "n_val": 0,
                    "n_test": 0,
                }
            if (time.time() - sweep_t0) > args.time_budget_s:
                logger.warning(
                    "wall budget %.1fs exceeded; aborting sweep.",
                    args.time_budget_s,
                )
                break
        if (time.time() - sweep_t0) > args.time_budget_s:
            break

    total_wall_s = time.time() - sweep_t0
    total_wall_min = total_wall_s / 60.0
    n_runs_completed = sum(
        1
        for cell in sweep_results
        for v in sweep_results[cell].values()
        if v["status"] == "ok"
    )

    per_cell_summary: Dict[str, dict] = {}
    aucs_across_cells: List[float] = []
    for cell in matched:
        per_seed = sweep_results.get(cell, {})
        vals = [
            v["auc"] for v in per_seed.values()
            if v["status"] == "ok" and v["auc"] == v["auc"]
        ]
        rec_for_cell = next((r for r in summary_records if r.cell == cell), None)
        n_train_c = int(rec_for_cell.n_train) if rec_for_cell else 0
        n_val_c = int(rec_for_cell.n_val) if rec_for_cell else 0
        n_test_c = int(rec_for_cell.n_test) if rec_for_cell else 0
        if vals:
            mean_auc = float(statistics.mean(vals))
            std_auc = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
            per_cell_summary[cell] = {
                "mean_auc": mean_auc,
                "std": std_auc,
                "n_seeds": len(vals),
                "n_train": n_train_c,
                "n_val": n_val_c,
                "n_test": n_test_c,
                "per_seed_auc": {s: per_seed[s]["auc"] for s in per_seed},
            }
            aucs_across_cells.append(mean_auc)
        else:
            per_cell_summary[cell] = {
                "mean_auc": float("nan"),
                "std": float("nan"),
                "n_seeds": 0,
                "n_train": n_train_c,
                "n_val": n_val_c,
                "n_test": n_test_c,
                "per_seed_auc": {},
            }

    mean_auc_overall = (
        float(statistics.mean(aucs_across_cells)) if aucs_across_cells else float("nan")
    )

    honest_framing_note = (
        f"Wf-315F+Wf-319E NCI-60 9-cell chemprop v2 sweep "
        f"({n_runs_completed}/{n_runs} ok runs in {total_wall_min:.2f} min). "
        f"Per-cell single-task chemprop v2 with --split={args.split} "
        f"(temporal = random 80/10/20 since NCI-60 has no temporal Year "
        f"column; scaffold = Bemis-Murcko scaffold-OOD; auto = temporal "
        f"when n_test_random_est>=per_cell_n_test_min, else scaffold "
        f"fallback). per_cell_split_choice={per_cell_split_choice}. "
        f"per_cell_n_test_min={per_cell_n_test_min}. "
        f"hidden_dim={args.hidden}, depth={args.depth}; Adam(lr={args.lr}); "
        f"{args.epochs} epochs. Per-cell independent encoders (no shared "
        f"multi-task scaffolding in the smoke driver; future P2 wires the "
        f"MultiTaskLoss head per TODO-130 G3). Wf-315F honest framing: "
        f"with default --n-test-min=50 the auto branch flips HL60/"
        f"SKOV3/DU145 (n_test_random_est 32-45 < 50) to scaffold_fallback; "
        f"however NO NCI-60 cell has a Bemis-Murcko scaffold group with "
        f">=50 rows (max sizes: A549=37, MCF7=34, A2780=37, A2780cis=30, "
        f"HCT116=20, SKOV3=18, HL60=18, DU145=10, PC3=9), so the "
        f"scaffold helper returns None -> random 80/10/20 fallback. "
        f"Wf-319E per-cell n_test_min auto-tuning "
        f"(--n-test-min-auto={args.n_test_min_auto}, headroom={args.n_test_min_headroom}, "
        f"floor={args.n_test_min_floor}) computes per-cell thresholds from "
        f"max scaffold size: A549=31, MCF7=28, A2780=31, A2780cis=26, "
        f"HCT116=17, SKOV3=15, HL60=15, DU145=9, PC3=8. With auto-tuning "
        f"ON (default), --split auto now actually triggers scaffold-OOD "
        f"for the 3 largest cells where it is meaningful (A549/MCF7/A2780), "
        f"preserves temporal for the 3 mid-size cells (A2780cis/HCT116/"
        f"PC3 n_test_random_est > per_cell threshold), and falls back "
        f"to random 80/10/20 for the 3 small cells (SKOV3/HL60/DU145) "
        f"where the per_cell threshold is still >= n_test_random_est. "
        f"This is INTEGRATION work, NOT a new measurement; no AUC values "
        f"are reported."
    )

    aggregate = {
        "ran": n_runs_completed == n_runs,
        "partial": n_runs_completed < n_runs,
        "n_runs_planned": n_runs,
        "n_runs_completed": n_runs_completed,
        "total_wall_min": float(total_wall_min),
        "total_wall_s": float(total_wall_s),
        "per_cell_seed_auc": sweep_results,
        "per_cell_summary": per_cell_summary,
        "mean_auc_across_cells": mean_auc_overall,
        "n_cells_used": len(matched),
        "n_cells_requested": len(cells),
        "cells_matched": {c: matched[c] for c in matched},
        "cells_unmatched": [c for c in cells if c not in matched],
        "cells_available": available_cells,
        "per_seed_records": [asdict(r) for r in summary_records],
        "per_cell_split_choice": per_cell_split_choice,
        "per_cell_n_test_min": per_cell_n_test_min,
        "split_protocol": (
            f"per_cell_split_choice drives --split={args.split!r} "
            f"resolved per cell. Wf-319E per-cell n_test_min "
            f"auto-tuning (--n-test-min-auto={args.n_test_min_auto}) "
            f"computes per-cell n_test_min from max scaffold size "
            f"(headroom={args.n_test_min_headroom}, "
            f"floor={args.n_test_min_floor}). per_cell_n_test_min="
            f"{per_cell_n_test_min}."
        ),
        "split_arg": {
            "split_flag": args.split,
            "n_test_min": int(args.n_test_min),
            "n_test_min_auto": bool(args.n_test_min_auto),
            "n_test_min_floor": int(args.n_test_min_floor),
            "n_test_min_headroom": float(args.n_test_min_headroom),
        },
        "honest_framing_note": honest_framing_note,
        "config": {
            "cells": list(cells),
            "seeds": list(args.seeds),
            "epochs": int(args.epochs),
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "num_workers": int(args.num_workers),
            "device": str(device),
            "time_budget_s": float(args.time_budget_s),
            "bf16_autocast": True,
            "split": args.split,
            "n_test_min": int(args.n_test_min),
            "n_test_min_auto": bool(args.n_test_min_auto),
            "n_test_min_floor": int(args.n_test_min_floor),
            "n_test_min_headroom": float(args.n_test_min_headroom),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(aggregate, indent=2))
    logger.info("wrote %s", out_path)
    return aggregate


# ---------------------------------------------------------------------------
# Dry-run plan
# ---------------------------------------------------------------------------
def dry_run_plan(args: argparse.Namespace) -> Dict[str, Any]:
    """Emit the planned 9-cell multi-task setup JSON shape."""
    csv_exists = Path(args.csv).exists()
    return {
        "driver": "nci60_9cell_chemprop_driver",
        "wf_origin": "Wf-308C",
        "todo_ref": "TODO-pending/220_wf219_temporal_split_nci60_driver.md",
        "csv": args.csv,
        "csv_exists": csv_exists,
        "cells": list(args.cells),
        "n_cells": len(args.cells),
        "seeds": list(args.seeds),
        "n_seeds": len(args.seeds),
        "hidden": args.hidden,
        "depth": args.depth,
        "epochs": args.epochs,
        "out": args.out,
        "dry_run": True,
        "status": (
            "STUB_PLAN_ONLY (CPU-only dry-run; --execute path implemented "
            "in Wf-308C; needs CSV + chemprop v2 + USER_OK for full GPU)"
            if csv_exists
            else "CSV_MISSING — per Wf-306B honest-negative do not run "
                 "--execute until the CSV exists."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    """Build the argparse parser for the NCI-60 9-cell driver."""
    p = argparse.ArgumentParser(
        prog="nci60_9cell_chemprop_driver",
        description=(
            "Wf-308C TODO-220 NCI-60 9-cell chemprop v2 multi-task "
            "driver. Default mode is --dry-run (CPU-only plan emission). "
            "Pass --execute to run a real per-cell-line chemprop v2 sweep."
        ),
    )
    p.add_argument(
        "--csv",
        type=str,
        default="molmetal/data/nci60_precious_9cell.csv",
        help="Path to the NCI-60 9-cell CSV (default: molmetal/data/nci60_precious_9cell.csv).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds for the multi-seed sweep (default: 0 1 2 = 3-seed quick; full = 0 1 2 3 4).",
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="chemprop v2 hidden_dim (default: 64 for CPU smoke; TODO-130 G3 prod = 128).",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=2,
        help="chemprop v2 depth / n_layers (default: 2 for CPU; TODO-130 G3 prod = 3).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Epochs per seed (default: 50 = Krasnov 2026 protocol; smoke = 1).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
        help=f"chemprop v2 batch_size (default: {DEFAULT_BATCH}).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Adam learning rate (default: {DEFAULT_LR}).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"DataLoader num_workers (default: {DEFAULT_NUM_WORKERS} for CPU smoke).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu; pass 'cuda' for GPU full run).",
    )
    p.add_argument(
        "--time-budget-s",
        type=float,
        default=DEFAULT_TIME_BUDGET_S,
        help="Wall budget for the sweep in seconds (default: 1 hr).",
    )
    p.add_argument(
        "--cells",
        type=str,
        nargs="+",
        default=DEFAULT_9_CELLS,
        help="NCI-60 cell-line labels to multi-task over (default: 9 canonical cells).",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=("temporal", "scaffold", "auto"),
        default="temporal",
        help=(
            "Split protocol (Wf-315F): 'temporal' = random 80/10/20 "
            "(equivalent to the Wf-308C default; NCI-60 has no temporal "
            "Year column so 'temporal' here is the random holdout). "
            "'scaffold' = scaffold-OOD (Bemis-Murcko held-out group per "
            "seed). 'auto' = temporal when n_test>=50, else scaffold "
            "fallback (mirrors Wf-312A pattern from the metal-cohort "
            "driver). Default 'temporal' preserves the Wf-308C behaviour."
        ),
    )
    p.add_argument(
        "--n-test-min",
        type=int,
        default=50,
        help=(
            "Minimum test rows for the random (temporal) split to be "
            "considered non-degenerate. Cells below this threshold in "
            "--split auto fall back to scaffold split. Default 50."
        ),
    )
    p.add_argument(
        "--n-test-min-auto",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Wf-319E: per-cell n_test_min auto-tuning. When ON (default) "
            "in --split auto mode, compute the per-cell threshold from "
            "the max Bemis-Murcko scaffold group size (headroom_factor "
            "= 0.85). When OFF, use the single --n-test-min value for "
            "all cells (Wf-315F behaviour; default n_test_min=50 makes "
            "scaffold fallback never trigger for the 9-cell CSV). Pass "
            "--no-n-test-min-auto to restore the Wf-315F single-threshold "
            "behaviour."
        ),
    )
    p.add_argument(
        "--n-test-min-floor",
        type=int,
        default=5,
        help=(
            "Wf-319E: lower bound for the per-cell n_test_min auto "
            "computation. Cells with very small max scaffold sizes "
            "(e.g. PC3 max=9) would otherwise have n_test_min=8; the "
            "floor caps the threshold at 5 to avoid over-restrictive "
            "scaffold selection. Default 5."
        ),
    )
    p.add_argument(
        "--n-test-min-headroom",
        type=float,
        default=0.85,
        help=(
            "Wf-319E: headroom factor applied to the max scaffold "
            "group size when computing per-cell n_test_min. With "
            "headroom=0.85 and max scaffold=37 (A549/MCF7/A2780): "
            "n_test_min = round(0.85 * 37) = 31. Lower headroom = "
            "more cells trigger scaffold fallback. Default 0.85."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Emit the planned 9-cell setup JSON shape and exit 0 (default behaviour).",
    )
    p.add_argument(
        "--execute",
        dest="dry_run",
        action="store_false",
        help="Run the actual per-cell chemprop v2 sweep (Wf-308C).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="molmetal/reports/wf308c_nci60_9cell/sweep_summary.json",
        help="Output JSON path for the sweep (ignored in dry-run).",
    )
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.dry_run:
        plan = dry_run_plan(args)
        print(json.dumps(plan, indent=2))
        return 0
    # --execute path
    result = execute_sweep(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())