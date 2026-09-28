#!/usr/bin/env python3
"""Wf-148 P1 — Generate balanced Rh cohort (250 active + 250 inactive) from real DrugOOD data.

==============================================================================
Wf-148 P1 — REAL DrugOOD-based BALANCED Rh cohort for LOMO v2 sweep prep
==============================================================================

**Bug to fix**: Wf-145 P1 ship 250 active + 250 inactive Rh cohort based on
**synthetic compounds** + similarity-based labels (proxy).  The Wf-131 P2 +
Wf-125 P1 root cause was constant-inactive label, not synthetic-vs-real.
We now generate a **real DrugOOD-based cohort** with median-threshold labels.

**This script**:
1. Stream-scan the real DrugOOD IC50 train.csv (1.58M rows from ChEMBL 29).
2. Filter compounds whose Bemis-Murcko scaffold has Tanimoto similarity
   (Morgan r=2, 2048 bits) **> 0.7 to ANY Rh real active** from
   MetalCytoToxDB Rh train cohort (n=413 actives).
3. Apply median-threshold labels from DrugOOD CSV ``y`` column
   (``y=1`` ⇔ ``pchembl_value >= 5.0`` per DrugOOD curator.py logic).
4. Sample 250 active + 250 inactive (capped at cohort size).
5. Output ``molmetal/data/balanced_rh_cohort.csv`` with columns
   ``smiles, y, mol_weight, scaffold_smiles, max_sim_to_rh_active``.

**Differences from Wf-145 P1**:
- Wf-145 P1 used synthetic compounds (Catherine closure-generated) + Tanimoto
  similarity to MetalCytoToxDB Rh actives for label assignment (proxy).
- Wf-148 P1 uses real DrugOOD organic chemistry + scaffold similarity
  filter to Rh actives + median-threshold labels (also proxy, but
  **real molecules**).

**Honest framing** (per `party-twenty-spirit` §3.5):
- DrugOOD data is general ChEMBL bioactivity (not metal-specific).
- The 500 compounds are NOT wet-lab validated for Rh binding.
- Scaffold similarity > 0.7 to Rh actives is a chemotype filter, not a
  binding assay.
- Median-threshold labels are an **adaptive** cut; not wet-lab labels.
- n=500 cohort is training-time augmentation; not publication-grade.

Constraints: CPU-only RDKit + similarity; cohort generation is CPU-only,
full LOMO v2 sweep on this cohort requires GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# Defaults
DEFAULT_REAL_CSV = "/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask.csv"
DEFAULT_DRUGOOD_CSV = "/home/hugo/codes/mol-metal/.data/molmetal/drugood/ic50/data/train.csv"
DEFAULT_OUTPUT_CSV = "/home/hugo/codes/mol-metal/molmetal/data/balanced_rh_cohort.csv"
DEFAULT_OUTPUT_META = "/home/hugo/codes/mol-metal/molmetal/data/balanced_rh_cohort_meta.json"

DEFAULT_N_ACTIVE = 250
DEFAULT_N_INACTIVE = 250
DEFAULT_SCAFFOLD_SIM_THRESHOLD = 0.7
DEFAULT_MORGAN_RADIUS = 2
DEFAULT_MORGAN_BITS = 2048
DEFAULT_SEED = 12345


def get_largest_fragment(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Return the largest connected fragment of ``mol`` (by atom count).

    Coordination compounds (e.g. Rh complexes) often include counterions
    and ligands as separate dot-separated fragments.  We only need the
    main scaffold, so we take the largest fragment.
    """
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return None
    return max(frags, key=lambda m: m.GetNumAtoms())


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    """Return canonical Murcko scaffold SMILES, or ``""`` on failure.

    If the molecule has no ring system, returns the canonical SMILES of the
    input molecule (per RDKit convention for acyclic Murcko).
    """
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return Chem.MolToSmiles(mol)
        return Chem.MolToSmiles(scaf, canonical=True)
    except Exception:
        return ""


def morgan_fp(smiles: str, radius: int = DEFAULT_MORGAN_RADIUS,
               n_bits: int = DEFAULT_MORGAN_BITS) -> Optional[DataStructs.ExplicitBitVect]:
    """Compute Morgan fingerprint for the largest fragment of ``smiles``."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    largest = get_largest_fragment(mol)
    if largest is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(largest, radius, nBits=n_bits)


def build_rh_active_scaffold_fps(
    real_csv: Path,
    metal: str = "Rh",
    split: str = "train",
    radius: int = DEFAULT_MORGAN_RADIUS,
    n_bits: int = DEFAULT_MORGAN_BITS,
) -> Tuple[List[DataStructs.ExplicitBitVect], List[str]]:
    """Build Morgan FPs of Rh real active Bemis-Murcko scaffolds.

    Returns
    -------
    (fps, scaffold_smis) : lists of equal length.  FPs are Morgan r=2 2048-bit.
    """
    import pandas as pd

    df = pd.read_csv(real_csv)
    rh_active = df[(df.metal == metal) & (df.split == split) & (df.activity == 1)]
    print(f"[rh_anchor] loaded {len(rh_active)} real Rh {split} actives", flush=True)

    fps: List[DataStructs.ExplicitBitVect] = []
    smis: List[str] = []
    n_skip = 0
    for smi in rh_active["smiles"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_skip += 1
            continue
        largest = get_largest_fragment(mol)
        if largest is None:
            n_skip += 1
            continue
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(largest)
            if scaf is None or scaf.GetNumAtoms() == 0:
                # acyclic — use the full largest fragment as scaffold
                scaf = largest
            scaf_smi = Chem.MolToSmiles(scaf, canonical=True)
            if not scaf_smi:
                n_skip += 1
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(scaf, radius, nBits=n_bits)
            fps.append(fp)
            smis.append(scaf_smi)
        except Exception:
            n_skip += 1
            continue
    unique_smis = list(set(smis))
    print(f"[rh_anchor] {len(fps)} scaffold FPs extracted ({len(unique_smis)} unique), "
          f"{n_skip} skipped", flush=True)
    return fps, smis


def stream_filter_drugood(
    drugood_csv: Path,
    rh_scaffold_fps: List[DataStructs.ExplicitBitVect],
    sim_threshold: float,
    radius: int = DEFAULT_MORGAN_RADIUS,
    n_bits: int = DEFAULT_MORGAN_BITS,
    max_active_pool: int = 20000,
    max_inactive_pool: int = 20000,
    log_every: int = 100000,
) -> Tuple[List[dict], List[dict], int]:
    """Stream DrugOOD CSV and collect active/inactive compounds with sim > threshold.

    Pools are capped (default 20000 each) to bound memory.

    Returns
    -------
    (active_pool, inactive_pool, n_total_scanned)
    """
    active_pool: List[dict] = []
    inactive_pool: List[dict] = []
    n_total = 0
    n_kept = 0
    n_skipped_parse = 0
    n_skipped_scaf = 0
    n_skipped_sim = 0
    t0 = time.time()

    with drugood_csv.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_total += 1
            if n_total % log_every == 0:
                elapsed = time.time() - t0
                rate = n_total / max(elapsed, 1e-6)
                print(
                    f"[drugood] scanned {n_total:,} kept={n_kept:,} "
                    f"(active={len(active_pool)}, inactive={len(inactive_pool)}) "
                    f"elapsed={elapsed:.1f}s rate={rate:.0f}/s",
                    flush=True,
                )
            try:
                y = int(row["y"])
            except (KeyError, ValueError):
                continue
            smi = row.get("smiles", "")
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                n_skipped_parse += 1
                continue
            try:
                scaf = MurckoScaffold.GetScaffoldForMol(mol)
                if scaf is None or scaf.GetNumAtoms() == 0:
                    scaf = mol  # acyclic, use full mol as scaffold proxy
                scaf_fp = AllChem.GetMorganFingerprintAsBitVect(
                    scaf, radius, nBits=n_bits
                )
            except Exception:
                n_skipped_scaf += 1
                continue
            try:
                sims = DataStructs.BulkTanimotoSimilarity(scaf_fp, rh_scaffold_fps)
                max_sim = float(max(sims)) if sims else 0.0
            except Exception:
                n_skipped_sim += 1
                continue
            if max_sim < sim_threshold:
                continue
            try:
                mw = float(Descriptors.MolWt(mol))
            except Exception:
                mw = float("nan")
            entry = {
                "smiles": smi,
                "y": y,
                "mol_weight": round(mw, 4),
                "scaffold_smiles": Chem.MolToSmiles(scaf, canonical=True),
                "max_sim_to_rh_active": round(max_sim, 4),
            }
            n_kept += 1
            if y == 1:
                if len(active_pool) < max_active_pool:
                    active_pool.append(entry)
            else:
                if len(inactive_pool) < max_inactive_pool:
                    inactive_pool.append(entry)

    print(
        f"[drugood] DONE: scanned {n_total:,}, kept {n_kept:,} "
        f"(active={len(active_pool)}, inactive={len(inactive_pool)}); "
        f"parse_skipped={n_skipped_parse}, scaf_skipped={n_skipped_scaf}, "
        f"sim_skipped={n_skipped_sim}; elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return active_pool, inactive_pool, n_total


def sample_balanced(
    active_pool: List[dict],
    inactive_pool: List[dict],
    n_active: int,
    n_inactive: int,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    """Randomly sample (with seed) n_active + n_inactive from pools."""
    rng = np.random.default_rng(seed)
    a_idx = rng.permutation(len(active_pool))[:n_active]
    i_idx = rng.permutation(len(inactive_pool))[:n_inactive]
    sampled_active = [active_pool[int(j)] for j in a_idx]
    sampled_inactive = [inactive_pool[int(j)] for j in i_idx]
    return sampled_active, sampled_inactive


def write_cohort_csv(
    out_csv: Path,
    actives: List[dict],
    inactives: List[dict],
) -> int:
    """Write the balanced cohort CSV (actives first, then inactives)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["smiles", "y", "mol_weight", "scaffold_smiles", "max_sim_to_rh_active"]
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in actives:
            writer.writerow(entry)
        for entry in inactives:
            writer.writerow(entry)
    return len(actives) + len(inactives)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--real-csv", default=DEFAULT_REAL_CSV,
                        help="MetalCytoToxDB multitask CSV (Rh actives anchor).")
    parser.add_argument("--drugood-csv", default=DEFAULT_DRUGOOD_CSV,
                        help="DrugOOD IC50 train.csv (1.58M rows).")
    parser.add_argument("--out-csv", default=DEFAULT_OUTPUT_CSV,
                        help="Output balanced Rh cohort CSV.")
    parser.add_argument("--out-meta", default=DEFAULT_OUTPUT_META,
                        help="Output cohort metadata JSON.")
    parser.add_argument("--n-active", type=int, default=DEFAULT_N_ACTIVE)
    parser.add_argument("--n-inactive", type=int, default=DEFAULT_N_INACTIVE)
    parser.add_argument("--sim-threshold", type=float, default=DEFAULT_SCAFFOLD_SIM_THRESHOLD,
                        help="Bemis-Murcko scaffold Tanimoto > threshold to Rh actives.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-active-pool", type=int, default=20000,
                        help="Cap active pool to bound memory.")
    parser.add_argument("--max-inactive-pool", type=int, default=20000,
                        help="Cap inactive pool to bound memory.")
    args = parser.parse_args(argv)

    t0 = time.time()
    print("=== Wf-148 P1: balanced Rh cohort from real DrugOOD ===", flush=True)
    print(f"  real_csv:        {args.real_csv}", flush=True)
    print(f"  drugood_csv:     {args.drugood_csv}", flush=True)
    print(f"  out_csv:         {args.out_csv}", flush=True)
    print(f"  sim_threshold:   {args.sim_threshold}", flush=True)
    print(f"  n_active target: {args.n_active}", flush=True)
    print(f"  n_inactive target: {args.n_inactive}", flush=True)
    print(f"  seed:            {args.seed}", flush=True)

    real_csv = Path(args.real_csv)
    drugood_csv = Path(args.drugood_csv)
    if not real_csv.is_file():
        print(f"[ERROR] real_csv not found: {real_csv}", flush=True)
        return 1
    if not drugood_csv.is_file():
        print(f"[ERROR] drugood_csv not found: {drugood_csv}", flush=True)
        return 1

    # ---- 1. Build Rh active scaffold FPs (anchor) ----
    print("\n[1/3] Building Rh active scaffold FPs (anchor)...", flush=True)
    rh_fps, rh_smis = build_rh_active_scaffold_fps(real_csv)
    if not rh_fps:
        print("[ERROR] no Rh scaffold FPs extracted", flush=True)
        return 2

    # ---- 2. Stream filter DrugOOD ----
    print("\n[2/3] Streaming DrugOOD filter (scaffold sim > "
          f"{args.sim_threshold})...", flush=True)
    active_pool, inactive_pool, n_scanned = stream_filter_drugood(
        drugood_csv,
        rh_scaffold_fps=rh_fps,
        sim_threshold=args.sim_threshold,
        max_active_pool=args.max_active_pool,
        max_inactive_pool=args.max_inactive_pool,
    )

    # ---- 3. Sample balanced + write ----
    print("\n[3/3] Sampling + writing balanced cohort...", flush=True)
    sampled_active, sampled_inactive = sample_balanced(
        active_pool, inactive_pool,
        n_active=args.n_active,
        n_inactive=args.n_inactive,
        seed=args.seed,
    )
    n_written = write_cohort_csv(
        Path(args.out_csv), sampled_active, sampled_inactive
    )

    # Metadata
    meta = {
        "workflow": "wf148_p1_balanced_rh_cohort",
        "phase": "Wf-148 P1 (TODO-135 Phase 1 alternate)",
        "metal": "Rh",
        "label_assignment": (
            "DrugOOD median threshold: y=1 iff pchembl_value >= 5.0 "
            "(per DrugOOD curator.py logic; no boundary discard)"
        ),
        "filter_method": (
            f"Bemis-Murcko scaffold Tanimoto (Morgan r=2 2048-bit) > "
            f"{args.sim_threshold} to ANY MetalCytoToxDB Rh train active scaffold"
        ),
        "rh_anchor_n_actives": len(rh_fps),
        "drugood_n_scanned": n_scanned,
        "drugood_n_kept_active": len(active_pool),
        "drugood_n_kept_inactive": len(inactive_pool),
        "n_active_sampled": len(sampled_active),
        "n_inactive_sampled": len(sampled_inactive),
        "n_total_sampled": n_written,
        "sim_threshold": args.sim_threshold,
        "seed": args.seed,
        "out_csv": str(args.out_csv),
        "wall_time_s": round(time.time() - t0, 2),
        "honest_framing": (
            "Cohort is CPU-only RDKit + similarity + DrugOOD median-threshold. "
            "Compounds are real (DrugOOD/ChEMBL organic chemistry), but "
            "NOT wet-lab validated for Rh binding. Scaffold similarity > 0.7 "
            "is a chemotype filter, not a binding assay. n=500 cohort is "
            "training-time augmentation only; not publication-grade. "
            "Differs from Wf-145 P1: Wf-145 used synthetic Catherine closure "
            "compounds + Tanimoto similarity labels; Wf-148 uses real "
            "DrugOOD compounds + median-threshold labels."
        ),
    }
    meta_path = Path(args.out_meta)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Wrote {meta_path}", flush=True)

    # Final summary
    print("\n=== SUMMARY ===", flush=True)
    print(f"  DrugOOD scanned       : {n_scanned:,}", flush=True)
    print(f"  Active pool size      : {len(active_pool):,}", flush=True)
    print(f"  Inactive pool size    : {len(inactive_pool):,}", flush=True)
    print(f"  Active sampled        : {len(sampled_active)} / {args.n_active}", flush=True)
    print(f"  Inactive sampled      : {len(sampled_inactive)} / {args.n_inactive}", flush=True)
    print(f"  Total written         : {n_written}", flush=True)
    print(f"  Output CSV            : {args.out_csv}", flush=True)
    print(f"  Wall time             : {meta['wall_time_s']} s", flush=True)

    if (len(sampled_active) == args.n_active
            and len(sampled_inactive) == args.n_inactive):
        print("\n  STATUS: PASS (balanced 250+250)", flush=True)
        return 0
    elif (len(sampled_active) > 0 and len(sampled_inactive) > 0):
        print("\n  STATUS: PARTIAL (cohort non-empty but threshold tuning needed)",
              flush=True)
        return 1
    else:
        print("\n  STATUS: HALT (empty cohort)", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())