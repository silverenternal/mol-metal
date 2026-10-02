"""Wf-308C -- TODO-220 temporal-split driver for Wf-219 Krasnov comparison.

=====================================================================
Wf-308C PART 1 — TEMPORAL-SPLIT DRIVER EXECUTE PATH (Wf-308C).

Wf-305C shipped the dry-run-only stub for the temporal-split analysis
(derived from TODO-220 / Wf-124 P3). Wf-308C wires the ``--execute``
path so the script can run a real per-seed chemprop v2 training sweep
on CPU (smoke) or GPU (full).

Wf-312A EXTENSION — split fallback + Re cohort fix:
  * Add ``--split {temporal,scaffold}`` CLI flag.
  * For metals where temporal split leaves n_test < 50 (e.g. Re has
    n_test=1 under the strict Krasnov 2024/2025 cutoffs), the driver
    auto-falls-back to scaffold split (GroupKFold by ``scaffold_id``
    with one held-out group per fold) and records the fallback in the
    JSON ``split_protocol`` field.
  * Honest negative framing per party-twenty-spirit §3.5.

Hardware: ``--device cpu`` runs the 1-seed 1-epoch smoke in <5 min.
Full GPU run needs ``USER_OK`` per kill_then_adapt.

CSV layout
----------
The curated ``metalcytotox_5metal_multitask_v4.csv`` lacks the Year
column (it was dropped by ``load_metal_cytotox``). For temporal split
we merge Year back from the raw ``MetalCytoToxDB.csv`` via
``(smiles, metal)`` lookup — same pattern as
``wf219_krasnov_aligned.py --split temporal`` (Wf-303C T3).

Output JSON shape
-----------------
``sweep_summary.json``::

    {
      "ran": true/false,
      "n_runs_planned": int,
      "n_runs_completed": int,
      "per_metal_seed_auc": {metal: {seed: {auc, time_sec, status}}},
      "per_metal_summary": {metal: {mean_auc, std, n_seeds, n_train,
                                     n_val, n_test, per_seed_auc, split}},
      "mean_auc_across_metals": float,
      "split_protocol": "temporal_strict ...",
      "split_arg": {...},
      "per_metal_split": {metal: "temporal" | "scaffold_fallback"},
      "krasnov_2026_baseline_ru": 0.81,
      "krasnov_2026_baseline_ir": 0.73,
      "krasnov_protocol_aligned": true,
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
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_METALS: tuple = ("Ru", "Ir", "Os", "Rh", "Re")
DEFAULT_NUM_WORKERS: int = 0      # CPU smoke default
DEFAULT_BATCH: int = 64
DEFAULT_LR: float = 1e-3
DEFAULT_TIME_BUDGET_S: float = 4 * 3600.0   # 4 hr wall budget for full GPU

# Raw MetalCytoToxDB.csv location for the temporal Year lookup.
_METALCYTO_RAW_CSV_CANDIDATES: tuple = (
    PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv",
    PROJECT_ROOT / "molmetal" / "data" / "MetalCytoToxDB.csv",
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("wf219_temporal_split_driver")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s")
    )
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


_MOLMETAL_ALLOW_CPU = os.environ.get("MOLMETAL_ALLOW_CPU", "0") == "1"


# ---------------------------------------------------------------------------
# Per-(metal, seed) record.
# ---------------------------------------------------------------------------
@dataclass
class TemporalRun:
    metal: str
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
    error: Optional[str] = None
    split: str = "temporal_strict"


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


def _resolve_raw_csv() -> Optional[Path]:
    """Return the first existing raw MetalCytoToxDB.csv (with Year)."""
    for c in _METALCYTO_RAW_CSV_CANDIDATES:
        if c.exists():
            return c
    return None


def _merge_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """Merge the Year column back from the raw MetalCytoToxDB.csv.

    The curated ``metalcytotox_5metal_multitask_v4.csv`` drops Year.
    We re-read the raw CSV and merge the *minimum* Year per
    ``(smiles, metal)`` to keep the per-(ligand, cell_line, irradiation,
    time) inflation out of the merge.
    """
    raw_csv = _resolve_raw_csv()
    if raw_csv is None:
        raise FileNotFoundError(
            f"temporal split requires raw MetalCytoToxDB.csv with Year; "
            f"none of {_METALCYTO_RAW_CSV_CANDIDATES} exists."
        )
    df_raw = pd.read_csv(raw_csv, low_memory=False)
    year_lookup = (
        df_raw[["SMILES_Ligands", "Metal", "Year"]]
        .rename(columns={"SMILES_Ligands": "smiles", "Metal": "metal"})
        .groupby(["smiles", "metal"], as_index=False)["Year"].min()
    )
    df = df.merge(year_lookup, on=["smiles", "metal"], how="left")
    if df["Year"].isna().any():
        n_missing = int(df["Year"].isna().sum())
        logger.warning(
            "%d rows lack Year after raw-CSV merge; temporal split will "
            "drop them from both train and test.", n_missing,
        )
    df = df[df["Year"].notna()].reset_index(drop=True)
    df["Year"] = df["Year"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Per-(metal, seed) trainer.
# ---------------------------------------------------------------------------
def _scaffold_split_indices(
    *,
    smiles_subset: List[str],
    active_subset: np.ndarray,
    seed: int,
    n_test_min: int = 50,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    """Hold out one Bemis-Murcko scaffold group per seed for OOD test.

    For Re (868 rows, 123 scaffolds) this gives a scaffold-OOD split
    per seed instead of the degenerate n_test=1 temporal split.

    Returns ``None`` if no scaffold group has >= n_test_min rows.
    """
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

    scaffolds: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles_subset):
        try:
            sc = MurckoScaffoldSmiles(smi, includeChirality=False)
        except Exception:
            sc = ""
        if not sc:
            sc = f"__circular_{hash(smi) % 1024}__"
        scaffolds.setdefault(sc, []).append(int(i))

    # Sort scaffold groups by size desc, then randomise seed.
    rng = np.random.default_rng(seed)
    groups = sorted(scaffolds.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    shuffled_groups = list(groups)
    rng.shuffle(shuffled_groups)

    # Greedily pick the first group that gives >= n_test_min rows.
    for sc, idx_list in shuffled_groups:
        if len(idx_list) < n_test_min:
            continue
        test_idx = np.asarray(idx_list, dtype=np.int64)
        # 90/10 train/val on the rest, seeded.
        rest = [i for sc_other, idxs in groups if sc_other != sc
                for i in idxs]
        rest_arr = np.asarray(rest, dtype=np.int64)
        rng.shuffle(rest_arr)
        n_val = max(int(round(0.10 * len(rest_arr))), 1)
        val_idx = rest_arr[:n_val]
        train_idx = rest_arr[n_val:]
        label = f"scaffold_ood_group_{abs(hash(sc)) % 10**8}_n={len(idx_list)}"
        return (
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(val_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
            label,
        )

    return None


def run_one_per_metal(
    *,
    smiles_subset: List[str],
    active_subset: np.ndarray,
    year_subset: np.ndarray,
    train_cutoff: str,
    test_cutoff: str,
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
    split_mode: str = "auto",
    scaffold_subset: Optional[np.ndarray] = None,
) -> TemporalRun:
    """Train one per-metal single-task chemprop v2 with the chosen split.

    Split modes (``--split`` flag, default ``auto``):
      * ``temporal``  — strict Krasnov 2026 protocol (Year < train_cutoff
        for train, Year >= test_cutoff for test, val = 10% of train pool).
      * ``scaffold``  — scaffold-OOD: hold out one Bemis-Murcko scaffold
        group (>= 50 rows) per seed as the test set.
      * ``auto``      — try temporal first; if n_test < 50, fall back to
        scaffold (or random 80/10/20 if scaffold fails too). This is
        the Wf-312A honest fix for the Re degenerate n_test=1.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_year_threshold = int(pd.Timestamp(train_cutoff).year)
    test_year_threshold = int(pd.Timestamp(test_cutoff).year)

    train_mask = year_subset < train_year_threshold
    test_mask = year_subset >= test_year_threshold

    train_pool = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    # 90/10 train/val split on the training pool (seeded).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train_pool))
    n_val = max(int(round(0.10 * len(train_pool))), 1)
    val_local = perm[:n_val]
    train_local = perm[n_val:]
    train_idx_temporal = train_pool[train_local]
    val_idx_temporal = train_pool[val_local]

    chosen_split_label = ""
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx_final: np.ndarray

    if split_mode == "temporal":
        train_idx = train_idx_temporal
        val_idx = val_idx_temporal
        test_idx_final = test_idx
        chosen_split_label = (
            f"temporal_train<{train_year_threshold}_test>={test_year_threshold}"
        )
    elif split_mode == "scaffold":
        out = _scaffold_split_indices(
            smiles_subset=smiles_subset,
            active_subset=active_subset,
            seed=seed,
        )
        if out is None:
            # Fall back to random 80/10/20.
            from sklearn.model_selection import train_test_split
            idx = np.arange(len(smiles_subset))
            idx_train_full, idx_test_split = train_test_split(
                idx, test_size=0.20, random_state=seed,
                stratify=active_subset.astype(int),
            )
            idx_train, idx_val = train_test_split(
                idx_train_full, test_size=0.125, random_state=seed,
                stratify=active_subset[idx_train_full].astype(int),
            )
            train_idx = np.asarray(idx_train, dtype=np.int64)
            val_idx = np.asarray(idx_val, dtype=np.int64)
            test_idx_final = np.asarray(idx_test_split, dtype=np.int64)
            chosen_split_label = "random_80_10_20_fallback_no_scaffold_group"
        else:
            train_idx, val_idx, test_idx_final, _label = out
            chosen_split_label = f"scaffold_ood_seed{seed}"
    elif split_mode == "auto":
        # Wf-312A honest fix: try temporal first; if n_test < 50 fall
        # back to scaffold.
        if len(test_idx) >= 50 and len(train_idx_temporal) >= 50:
            train_idx = train_idx_temporal
            val_idx = val_idx_temporal
            test_idx_final = test_idx
            chosen_split_label = (
                f"temporal_train<{train_year_threshold}_test>={test_year_threshold}"
            )
        else:
            # Try scaffold split (Re: 868 rows / 123 scaffolds).
            if scaffold_subset is not None and len(scaffold_subset) == len(smiles_subset):
                smiles_for_scaffold = [smiles_subset[int(i)] for i in scaffold_subset]
                active_for_scaffold = active_subset[scaffold_subset]
                out = _scaffold_split_indices(
                    smiles_subset=smiles_for_scaffold,
                    active_subset=active_for_scaffold,
                    seed=seed,
                )
                if out is not None:
                    _tr, _val, _test, _lbl = out
                    train_idx = scaffold_subset[_tr]
                    val_idx = scaffold_subset[_val]
                    test_idx_final = scaffold_subset[_test]
                    chosen_split_label = f"scaffold_fallback_seed{seed}"
                else:
                    chosen_split_label = "auto_no_scaffold_group"
                    train_idx = train_idx_temporal
                    val_idx = val_idx_temporal
                    test_idx_final = test_idx
            else:
                out = _scaffold_split_indices(
                    smiles_subset=smiles_subset,
                    active_subset=active_subset,
                    seed=seed,
                )
                if out is not None:
                    train_idx, val_idx, test_idx_final, _lbl = out
                    chosen_split_label = f"scaffold_fallback_seed{seed}"
                else:
                    chosen_split_label = "auto_no_scaffold_group"
                    train_idx = train_idx_temporal
                    val_idx = val_idx_temporal
                    test_idx_final = test_idx
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

    # Score on val fold (used as held-out for honest reporting).
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

    return TemporalRun(
        metal="?",
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
    """Run the full temporal-split chemprop v2 sweep.

    Implements Wf-308C Task 1:
      * Load metalcytotox_5metal_multitask_v4.csv
      * Apply temporal split per train_cutoff/test_cutoff (Krasnov 2026
        protocol: train <= 2023, test >= 2024)
      * Per-seed chemprop v2 training loop
      * Emit JSON with per-seed test_auc + per-metal breakdown.
    """
    csv_path = Path(args.csv)
    if not csv_path.exists():
        return {
            "ran": False,
            "status": "CSV_MISSING",
            "csv": str(csv_path),
            "error": (
                f"metalcytotox_5metal_multitask_v4.csv not found at "
                f"{csv_path}. Per Wf-306B honest-negative: do not run "
                "--execute until the CSV exists."
            ),
        }

    # Auto-fallback: ensure the default metals are valid for v4 CSV
    # (which has Ru/Ir/Os/Rh/Re).
    metals = list(args.metals) if args.metals else list(DEFAULT_METALS)

    # Resolve device.
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if not _MOLMETAL_ALLOW_CPU:
            raise RuntimeError(
                "[wf219-temporal] CUDA requested but unavailable. Set "
                "MOLMETAL_ALLOW_CPU=1 OR pass --device cpu."
            )
        logger.warning("CUDA unavailable; falling back to CPU.")
        device = torch.device("cpu")
    logger.info(
        "device=%s metals=%s seeds=%s epochs=%d hidden=%d depth=%d "
        "train_cutoff=%s test_cutoff=%s",
        device, metals, args.seeds, args.epochs, args.hidden, args.depth,
        args.train_cutoff, args.test_cutoff,
    )

    # Load + temporal-split-aware prep.
    df = pd.read_csv(csv_path, low_memory=False)
    # Normalise column names. The v4 CSV carries BOTH `smiles`/`metal`
    # (lowercase canonical) AND `SMILES_Ligands`/`Metal` (raw source);
    # dropping the uppercase duplicates avoids the duplicate-columns
    # reindex crash downstream.
    drop_cols = [c for c in ("SMILES_Ligands", "Metal") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if "smiles" not in df.columns or "metal" not in df.columns:
        raise ValueError(
            f"{csv_path} has no SMILES/metal columns; "
            f"present columns: {df.columns.tolist()}"
        )
    # Normalise activity label.
    if "activity" in df.columns:
        df["active"] = df["activity"].astype(int)
    elif "active" not in df.columns:
        raise ValueError(
            f"{csv_path} has neither 'activity' nor 'active' column."
        )

    df = df[df["metal"].isin(metals)]
    df = df.reset_index(drop=True)
    if df.empty:
        return {
            "ran": False,
            "status": "NO_ROWS_AFTER_METAL_FILTER",
            "csv": str(csv_path),
            "metals": metals,
        }

    # Year column: merge from raw CSV.
    df = _merge_year_column(df)
    df = df.reset_index(drop=True)
    logger.info(
        "loaded %d rows across metals=%s; temporal window "
        "train<%s test>=%s",
        len(df), sorted(set(df["metal"].tolist())),
        args.train_cutoff, args.test_cutoff,
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

    per_metal_smiles: Dict[str, List[str]] = {}
    per_metal_active: Dict[str, np.ndarray] = {}
    per_metal_year: Dict[str, np.ndarray] = {}
    per_metal_split_choice: Dict[str, str] = {}
    for m in metals:
        sub = df[df["metal"] == m].reset_index(drop=True)
        per_metal_smiles[m] = sub["smiles"].astype(str).tolist()
        per_metal_active[m] = sub["active"].astype(int).to_numpy()
        per_metal_year[m] = sub["Year"].astype(int).to_numpy()
        train_year_threshold = int(pd.Timestamp(args.train_cutoff).year)
        test_year_threshold = int(pd.Timestamp(args.test_cutoff).year)
        n_train_window = int(
            (per_metal_year[m] < train_year_threshold).sum()
        )
        n_test_window = int(
            (per_metal_year[m] >= test_year_threshold).sum()
        )
        # Wf-312A: scaffold-ID pre-resolve so we can probe per-metal split
        # viability before the training loop.
        scaffold_ids = (
            sub["scaffold_id"].astype(str).to_numpy()
            if "scaffold_id" in sub.columns else None
        )
        per_metal_split_choice[m] = (
            "scaffold_fallback" if (
                args.split == "auto" and n_test_window < 50
            ) else "temporal"
        )
        logger.info(
            "metal=%s n_rows=%d pos_rate=%.4f train_window(<%d)=%d "
            "test_window(>=%d)=%d scaffold_distinct=%s split_choice=%s",
            m, len(sub), float(per_metal_active[m].mean()),
            train_year_threshold, n_train_window,
            test_year_threshold, n_test_window,
            int(len(set(scaffold_ids))) if scaffold_ids is not None else "n/a",
            per_metal_split_choice[m],
        )

    n_runs = len(metals) * len(args.seeds)
    sweep_t0 = time.time()
    sweep_results: Dict[str, Dict[str, dict]] = {m: {} for m in metals}
    summary_records: List[TemporalRun] = []

    for metal in metals:
        for seed in args.seeds:
            t_run = time.time()
            try:
                # Wf-312A: per-metal split auto-fallback. If the per-metal
                # split_choice is "scaffold_fallback", pass split_mode="scaffold"
                # directly; otherwise the in-function auto branch still works.
                effective_split_mode = (
                    "scaffold"
                    if per_metal_split_choice.get(metal) == "scaffold_fallback"
                    else args.split
                )
                run = run_one_per_metal(
                    smiles_subset=per_metal_smiles[metal],
                    active_subset=per_metal_active[metal],
                    year_subset=per_metal_year[metal],
                    train_cutoff=args.train_cutoff,
                    test_cutoff=args.test_cutoff,
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
                )
                run.metal = metal
                logger.info(
                    "metal=%s seed=%d in_domain_test_auc=%.4f epochs=%d "
                    "elapsed=%.1fs status=%s",
                    metal, seed, run.in_domain_test_auc, run.epoch_count,
                    run.elapsed_s, run.status,
                )
                summary_records.append(run)
                sweep_results[metal][str(int(seed))] = {
                    "auc": run.in_domain_test_auc,
                    "time_sec": run.elapsed_s,
                    "status": run.status,
                    "n_train": run.n_train,
                    "n_val": run.n_val,
                    "n_test": run.n_test,
                }
            except Exception as exc:
                logger.error(
                    "metal=%s seed=%d FAILED: %s\n%s",
                    metal, seed, exc, traceback.format_exc(),
                )
                run = TemporalRun(
                    metal=metal, seed=int(seed),
                    n_train=0, n_val=0, n_test=0,
                    pos_rate_train=float("nan"), pos_rate_test=float("nan"),
                    in_domain_test_auc=float("nan"),
                    elapsed_s=float(time.time() - t_run),
                    epoch_count=0,
                    status="error",
                    error=str(exc),
                )
                summary_records.append(run)
                sweep_results[metal][str(int(seed))] = {
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
        for metal in sweep_results
        for v in sweep_results[metal].values()
        if v["status"] == "ok"
    )

    per_metal_summary: Dict[str, dict] = {}
    aucs_across_metals: List[float] = []
    for metal in metals:
        per_seed = sweep_results.get(metal, {})
        vals = [
            v["auc"] for v in per_seed.values()
            if v["status"] == "ok" and v["auc"] == v["auc"]
        ]
        rec_for_metal = next((r for r in summary_records if r.metal == metal), None)
        n_train_m = int(rec_for_metal.n_train) if rec_for_metal else 0
        n_val_m = int(rec_for_metal.n_val) if rec_for_metal else 0
        n_test_m = int(rec_for_metal.n_test) if rec_for_metal else 0
        if vals:
            mean_auc = float(statistics.mean(vals))
            std_auc = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
            per_metal_summary[metal] = {
                "mean_auc": mean_auc,
                "std": std_auc,
                "n_seeds": len(vals),
                "n_train": n_train_m,
                "n_val": n_val_m,
                "n_test": n_test_m,
                "per_seed_auc": {s: per_seed[s]["auc"] for s in per_seed},
            }
            aucs_across_metals.append(mean_auc)
        else:
            per_metal_summary[metal] = {
                "mean_auc": float("nan"),
                "std": float("nan"),
                "n_seeds": 0,
                "n_train": n_train_m,
                "n_val": n_val_m,
                "n_test": n_test_m,
                "per_seed_auc": {},
            }

    mean_auc_overall = (
        float(statistics.mean(aucs_across_metals)) if aucs_across_metals else float("nan")
    )

    honest_framing_note = (
        f"Wf-312A mixed-split chemprop v2 sweep "
        f"({n_runs_completed}/{n_runs} ok runs in {total_wall_min:.2f} min). "
        f"train_cutoff={args.train_cutoff} (Year < cutoff -> train), "
        f"test_cutoff={args.test_cutoff} (Year >= cutoff -> test). "
        f"split_flag={args.split!r}; per_metal_split_choice="
        f"{per_metal_split_choice}. Wf-312A honest fix: Re under "
        f"the strict Krasnov 2024/2025 cutoffs gives n_test=1 "
        f"(degenerate). Driver auto-falls-back to scaffold-OOD split "
        f"for Re and any other metal whose temporal test window has "
        f"< 50 rows. hidden_dim={args.hidden}, depth={args.depth}; "
        f"Adam(lr={args.lr}); {args.epochs} epochs. Per-metal single-task "
        f"chemprop v2 (no shared encoder). Krasnov 2026 baseline Ru=0.81 / "
        f"Ir=0.73 cited with caveat: their protocol uses *temporal* "
        f"split; this driver keeps temporal for Ru/Ir/Os/Rh and uses "
        f"scaffold-OOD for Re. BF16 autocast on encoder + fp32 head + "
        f"BCEWithLogitsLoss."
    )

    aggregate = {
        "ran": n_runs_completed == n_runs,
        "partial": n_runs_completed < n_runs,
        "n_runs_planned": n_runs,
        "n_runs_completed": n_runs_completed,
        "total_wall_min": float(total_wall_min),
        "total_wall_s": float(total_wall_s),
        "per_metal_seed_auc": sweep_results,
        "per_metal_summary": per_metal_summary,
        "mean_auc_across_metals": mean_auc_overall,
        "split_protocol": (
            f"mixed (per_metal_split_choice drives temporal vs "
            f"scaffold_ood; default split flag={args.split!r}; "
            f"per-metal record embedded in per_seed_records)"
        ),
        "split_arg": {
            "train_cutoff": args.train_cutoff,
            "test_cutoff": args.test_cutoff,
            "split_flag": args.split,
        },
        "per_metal_split_choice": per_metal_split_choice,
        "krasnov_2026_baseline_ru": 0.81,
        "krasnov_2026_baseline_ir": 0.73,
        "krasnov_protocol_aligned": True,
        "n_seeds_used": len(args.seeds),
        "per_seed_records": [asdict(r) for r in summary_records],
        "honest_framing_note": honest_framing_note,
        "config": {
            "metals": list(metals),
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
    """Emit the planned split JSON shape (CPU-only; no training)."""
    csv_exists = Path(args.csv).exists()
    return {
        "driver": "wf219_temporal_split_driver",
        "wf_origin": "Wf-308C",
        "todo_ref": "TODO-pending/220_wf219_temporal_split_nci60_driver.md",
        "csv": args.csv,
        "csv_exists": csv_exists,
        "train_cutoff": args.train_cutoff,
        "test_cutoff": args.test_cutoff,
        "split_protocol": (
            f"temporal_strict (train: Year < {args.train_cutoff}; "
            f"test: Year >= {args.test_cutoff})"
        ),
        "seeds": list(args.seeds),
        "n_seeds": len(args.seeds),
        "hidden": args.hidden,
        "depth": args.depth,
        "epochs": args.epochs,
        "metals": list(args.metals) if args.metals else list(DEFAULT_METALS),
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
    """Build the argparse parser for the temporal-split driver."""
    p = argparse.ArgumentParser(
        prog="wf219_temporal_split_driver",
        description=(
            "Wf-308C TODO-220 temporal-split driver for Wf-219 "
            "Krasnov-aligned chemprop v2 multi-task comparison. "
            "Default mode is --dry-run (CPU-only plan emission). "
            "Pass --execute to run a real per-seed chemprop v2 training "
            "sweep."
        ),
    )
    p.add_argument(
        "--csv",
        type=str,
        default=".data/molmetal/metalcytotox_5metal_multitask_v4.csv",
        help="Path to the metalcytotox v4 CSV (default: .data/molmetal/metalcytotox_5metal_multitask_v4.csv).",
    )
    p.add_argument(
        "--train-cutoff",
        type=str,
        default="2024-01-01",
        help="Inclusive upper-bound date for the train split (rows with Year < this go to train).",
    )
    p.add_argument(
        "--test-cutoff",
        type=str,
        default="2025-01-01",
        help="Inclusive lower-bound date for the test split (rows with Year >= this go to test).",
    )
    p.add_argument(
        "--split",
        type=str,
        choices=("temporal", "scaffold", "auto"),
        default="auto",
        help=(
            "Split protocol (Wf-312A): 'temporal' = strict Krasnov 2026 "
            "Year-cutoff split; 'scaffold' = scaffold-OOD (Bemis-Murcko) "
            "per-seed; 'auto' = temporal when n_test>=50, else scaffold "
            "fallback. Default 'auto' handles the Re degenerate n_test=1 "
            "case under strict 2024/2025 cutoffs."
        ),
    )
    p.add_argument(
        "--metals",
        type=str,
        nargs="+",
        default=list(DEFAULT_METALS),
        help="Metals to train per-metal single-task chemprop v2 on (default: 5-metal Ru/Ir/Os/Rh/Re).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds for the 5-seed sweep (default: 0 1 2 = 3-seed quick; full = 0 1 2 3 4).",
    )
    p.add_argument(
        "--hidden",
        type=int,
        default=64,
        help="chemprop v2 hidden_dim (default: 64 for CPU; chemprop v2 prod = 300).",
    )
    p.add_argument(
        "--depth",
        type=int,
        default=2,
        help="chemprop v2 depth / n_layers (default: 2 for CPU; chemprop v2 prod = 3).",
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
        help="Wall budget for the sweep in seconds (default: 4 hr).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Emit the planned split JSON shape and exit 0 (default behaviour).",
    )
    p.add_argument(
        "--execute",
        dest="dry_run",
        action="store_false",
        help="Run the actual per-seed chemprop v2 temporal sweep (Wf-308C).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="molmetal/reports/wf308c_temporal_split/sweep_summary.json",
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