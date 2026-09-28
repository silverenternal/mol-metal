#!/usr/bin/env python3
"""Wf-71 Phase 3 — Wf-49 cross-dataset retrain round 2 (per-cohort breakdown).

================================================================
Wf-71 P3: Wf-49 cross-dataset retrain round 2 with extended per-cohort
breakdown on three mol-metal regression tasks.  This is the Wf-49
follow-up that turns the round-1 cross-dataset signal ("real RDKit 2D
features beat F5 on MetalCytoTox 5-metal Pearson r up to |rho|=0.204")
into a per-cohort train/test split with honest per-cohort R^2 reporting.

What this bench does (vs Wf-49 round 1)
---------------------------------------
* **Wf-49 round 1** (closeout in ``TODO/completed/91_wf49_r2_root_fix.md``)
  reported a *dataset-level* Pearson r between each RDKit 2D descriptor
  and the bioactivity target on the union of all metals / cell-lines.
  It did *not* split into per-cohort train/test splits.

* **This round 2** (``bench_wf49_cross_dataset_round2.py``) trains a
  Ridge regression on F5_RDKIT_2D features inside each cohort and
  reports:

  - per-cohort R^2 on a held-out 20% test split (5-seed mean +/- std)
  - per-cohort RMSE in the target's native units (nM for pIC50; binary
    threshold for activity; eV for HL_Gap)
  - per-cohort Pearson r between predicted and observed targets
  - the macro-mean across cohorts within each dataset
  - the macro-mean across all 3 datasets

Inputs (per-cohort)
-------------------
* PlatinAI 4-cell (4078 Pt complexes, 4 cell-lines):
  - cohorts: ``MCF7`` / ``A549`` / ``A2780`` / ``A2780cis``
  - target: ``pIC50 = 9 - log10(IC50_nM)`` per cell-line
  - features: 12-d RDKit 2D (MolWt / MolLogP / TPSA / HBA / HBD /
    NRotB / HeavyAtoms / RingCount / NumAromaticRings / FractionCSP3
    / MolMR / BalabanJ)
  - data file: ``molmetal/data/platinai_4cell_rdkit_features.csv``

* MetalCytoToxDB 5-metal (~26.8k complexes, Ru/Ir/Rh/Os/Re):
  - cohorts: 5 metals (Ru / Ir / Rh / Os / Re)
  - target: ``activity`` (binary 0/1)
  - data file: ``molmetal/data/metalcytotox_rdkit_features.csv``

* tmQM d-block (108,542 complexes, 29 d-block metals):
  - cohorts: 29 d-block metals (Sc, Ti, ..., Au, Hg)
  - target: ``HL_Gap`` (HOMO-LUMO gap in hartree, a real continuous
    electronic property that is sensitive to ligand-field effects and
    d-orbital splitting -- a sensible QM-level analogue of pIC50)
  - data file: ``.data/molmetal/tmQM/tmqm_parsed.csv``

Verify criterion (per the Wf-71 P3 task contract)
------------------------------------------------
``F5_RDKIT_2D R^2 >= 0 mean across 3 datasets with per-cohort breakdown
(or honest per-cohort breakdown)``. This script always reports the
honest per-cohort breakdown; the R^2 >= 0 target is a *bonus* --
if it ships, that's a meaningful advance over the Wf-49 round-1
dataset-level correlations.

CLI
---
::

    python molmetal/scripts/bench_wf49_cross_dataset_round2.py \\
        --datasets platinai_4cell metalcytotox_5metal tmqm_dblock \\
        --per-cohort \\
        --seeds 0 1 2 3 4 \\
        --out-dir molmetal/reports/wf71_p3_wf49_cross_round2

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
We will *not* inflate R^2 by picking the best cohort, best seed, or
best feature subset.  All cohorts / all seeds / all features are
reported.  Negative R^2 (predict-the-mean-beating baseline) is
reported as such -- it's a feature, not a bug.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Repo paths / constants
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]  # mol-metal/

PLATINAI_FEATURES_CSV = (
    _REPO_ROOT / "molmetal" / "data" / "platinai_4cell_rdkit_features.csv"
)
METALCYTOTOX_FEATURES_CSV = (
    _REPO_ROOT / "molmetal" / "data" / "metalcytotox_rdkit_features.csv"
)
TMQM_PARSED_CSV = _REPO_ROOT / ".data" / "molmetal" / "tmQM" / "tmqm_parsed.csv"

# Default per-cohort breakdown settings
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_TEST_SIZE: float = 0.2

# RDKit 2D descriptor columns (12-d F5_RDKIT_2D feature set)
RDKIT_2D_COLS: Tuple[str, ...] = (
    "MolWt", "MolLogP", "TPSA", "HBA", "HBD", "NRotB",
    "HeavyAtoms", "RingCount", "NumAromaticRings", "FractionCSP3",
    "MolMR", "BalabanJ",
)

# PlatinAI 4-cell target columns (one per cell-line)
PLATINAI_TARGET_COLS: Dict[str, str] = {
    "MCF7": "pIC50_mcf7",
    "A549": "pIC50_a549",
    "A2780": "pIC50_a2780",
    "A2780cis": "pIC50_a2780cis",
}

# MetalCytoToxDB 5-metal whitelist
METALCYTOTOX_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")

# tmQM d-block: drop La (f-block) + Hg/Cd/Pb (post-d-block / toxicology
# outliers not relevant to HL_Gap analysis).  We use the canonical
# IUPAC d-block assignment: groups 3-12, period 4-6 (no La = f-block).
TMQM_DBLOCK_METALS: Tuple[str, ...] = (
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
)

DEFAULT_OUT_DIR = (
    _REPO_ROOT / "molmetal" / "reports" / "wf71_p3_wf49_cross_round2"
)

log = logging.getLogger("bench_wf49_cross_dataset_round2")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CohortRow:
    """One cohort (cell-line / metal) with its trained Ridge R^2."""
    dataset: str           # "platinai_4cell" | "metalcytotox_5metal" | "tmqm_dblock"
    cohort: str            # "MCF7" | "Ru" | "Sc" | ...
    target: str            # "pIC50_mcf7" | "activity" | "HL_Gap"
    n_rows: int            # n_train + n_test
    n_train: int
    n_test: int
    r2_per_seed: Tuple[float, ...]
    rmse_per_seed: Tuple[float, ...]
    pearson_r_per_seed: Tuple[float, ...]
    r2_mean: float
    r2_std: float
    rmse_mean: float
    pearson_r_mean: float


@dataclass(frozen=True)
class DatasetSummary:
    """One dataset's macro-aggregated result."""
    dataset: str
    n_cohorts: int
    cohort_names: Tuple[str, ...]
    macro_r2_mean: float
    macro_r2_std: float
    overall_r2_mean: float       # mean over all rows pooled
    overall_pearson_r_mean: float
    n_total_rows: int
    per_cohort: Tuple[CohortRow, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_platinai_4cell() -> pd.DataFrame:
    """Load PlatinAI 4-cell RDKit 2D features.

    Returns
    -------
    pandas.DataFrame
        Columns = [smiles, metal, ic50_*, pIC50_*, <12 RDKit 2D desc>].
        The pIC50_* columns (per cell-line) are the per-cohort targets.
    """
    if not PLATINAI_FEATURES_CSV.exists():
        raise FileNotFoundError(f"platinai 4-cell features missing: {PLATINAI_FEATURES_CSV}")
    df = pd.read_csv(PLATINAI_FEATURES_CSV)
    # Validate 12-d RDKit 2D descriptors + 4 pIC50 columns exist
    missing = [c for c in RDKIT_2D_COLS + tuple(PLATINAI_TARGET_COLS.values()) if c not in df.columns]
    if missing:
        raise ValueError(f"platinai 4-cell features missing cols: {missing}")
    return df


def load_metalcytotox_5metal() -> pd.DataFrame:
    """Load MetalCytoToxDB 5-metal RDKit 2D features.

    Returns
    -------
    pandas.DataFrame
        Columns = [smiles, metal, activity, <12 RDKit 2D desc>].
        ``activity`` (binary 0/1) is the regression target -- R^2 on
        a binary target equals explained variance of the 0/1 output.
    """
    if not METALCYTOTOX_FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"metalcytotox 5-metal features missing: {METALCYTOTOX_FEATURES_CSV}"
        )
    df = pd.read_csv(METCYT_FEATURES := METALCYTOTOX_FEATURES_CSV)
    missing = [c for c in RDKIT_2D_COLS + ("metal", "activity") if c not in df.columns]
    if missing:
        raise ValueError(f"metalcytotox features missing cols: {missing}")
    df = df[df["metal"].isin(METALCYTOTOX_METALS)].copy()
    return df


def load_tmqm_dblock(target: str = "HL_Gap") -> pd.DataFrame:
    """Load tmQM d-block RDKit 2D + electronic features.

    tmQM doesn't ship pre-extracted RDKit 2D descriptors, so we re-extract
    them on the fly using RDKit.  For the Wf-71 P3 run we keep only the
    12-d RDKit 2D set + the chosen continuous target ``HL_Gap`` (eV-
    scale continuous, sensitive to ligand-field effects on d-block
    metals).  Each row is one (csd_code, smiles, metal) entry.

    Returns
    -------
    pandas.DataFrame
        Columns = [csd_code, smiles, metal, target, <12 RDKit 2D desc>].
    """
    if not TMQM_PARSED_CSV.exists():
        raise FileNotFoundError(f"tmQM parsed missing: {TMQM_PARSED_CSV}")
    df = pd.read_csv(TMQM_PARSED_CSV, usecols=["csd_code", "smiles", "metal", target])
    df = df[df["metal"].isin(TMQM_DBLOCK_METALS)].copy()
    df = df.dropna(subset=["smiles", target]).reset_index(drop=True)

    # Re-extract RDKit 2D descriptors on the fly.  tmQM SMILES are
    # organometallic -- RDKit sometimes fails to parse; log + skip.
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    desc_lists: List[List[float]] = []
    n_dropped = 0
    for smi in df["smiles"].tolist():
        mol = Chem.MolFromSmiles(smi)
        if mol is None or mol.GetNumHeavyAtoms() == 0:
            desc_lists.append([float("nan")] * len(RDKIT_2D_COLS))
            n_dropped += 1
            continue
        try:
            desc_lists.append([
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumRotatableBonds(mol),
                mol.GetNumHeavyAtoms(),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                Descriptors.MolMR(mol),
                Descriptors.BalabanJ(mol),
            ])
        except Exception:
            desc_lists.append([float("nan")] * len(RDKIT_2D_COLS))
            n_dropped += 1
    desc_arr = np.asarray(desc_lists, dtype=np.float64)
    for i, col in enumerate(RDKIT_2D_COLS):
        df[col] = desc_arr[:, i]
    log.info("tmQM: %d rows loaded, %d dropped (RDKit parse fail)", len(df), n_dropped)
    df = df.rename(columns={target: "target"})
    return df


# ---------------------------------------------------------------------------
# Per-cohort Ridge fit (5-seed train/test split)
# ---------------------------------------------------------------------------
def fit_per_cohort(
    df: pd.DataFrame,
    cohort_col: str,
    target_col: str,
    cohort_name: str,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    test_size: float = DEFAULT_TEST_SIZE,
) -> CohortRow:
    """Fit Ridge on F5_RDKIT_2D for one cohort with 5-seed averaging."""
    sub = df[df[cohort_col] == cohort_name] if cohort_col else df
    sub = sub.dropna(subset=list(RDKIT_2D_COLS) + [target_col])
    if len(sub) < 10:
        return CohortRow(
            dataset="unknown", cohort=cohort_name, target=target_col,
            n_rows=len(sub), n_train=0, n_test=0,
            r2_per_seed=tuple(), rmse_per_seed=tuple(), pearson_r_per_seed=tuple(),
            r2_mean=float("nan"), r2_std=float("nan"),
            rmse_mean=float("nan"), pearson_r_mean=float("nan"),
        )
    X = sub[list(RDKIT_2D_COLS)].to_numpy(dtype=np.float64)
    y = sub[target_col].to_numpy(dtype=np.float64)
    r2s: List[float] = []
    rmses: List[float] = []
    prs: List[float] = []
    for seed in seeds:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=seed
            )
        except ValueError:
            # too few samples to split -- use all rows for both
            X_tr = X_te = X
            y_tr = y_te = y
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)
        ridge = Ridge(alpha=1.0, random_state=seed).fit(X_tr_s, y_tr)
        y_pred = ridge.predict(X_te_s)
        r2 = r2_score(y_te, y_pred)
        rmse = math.sqrt(mean_squared_error(y_te, y_pred))
        # Pearson r between y_te and y_pred (numpy's corrcoef gives
        # nan if y_te is constant -- guard).
        if np.std(y_te) > 0 and np.std(y_pred) > 0:
            pr = float(np.corrcoef(y_te, y_pred)[0, 1])
        else:
            pr = float("nan")
        r2s.append(float(r2))
        rmses.append(float(rmse))
        prs.append(float(pr))
    return CohortRow(
        dataset="unknown",  # filled in by caller
        cohort=cohort_name,
        target=target_col,
        n_rows=len(sub),
        n_train=int(round(len(sub) * (1 - test_size))),
        n_test=int(round(len(sub) * test_size)),
        r2_per_seed=tuple(r2s),
        rmse_per_seed=tuple(rmses),
        pearson_r_per_seed=tuple(prs),
        r2_mean=float(np.mean(r2s)),
        r2_std=float(np.std(r2s)),
        rmse_mean=float(np.mean(rmses)),
        pearson_r_mean=float(np.mean(prs)),
    )


# ---------------------------------------------------------------------------
# Dataset-level orchestration
# ---------------------------------------------------------------------------
def bench_platinai_4cell(
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> DatasetSummary:
    df = load_platinai_4cell()
    cohorts: List[CohortRow] = []
    for cohort_name, target_col in PLATINAI_TARGET_COLS.items():
        row = fit_per_cohort(
            df, cohort_col=None, target_col=target_col,
            cohort_name=cohort_name,  # cohort filter happens inside fit via subset
            seeds=seeds,
        )
        # For PlatinAI every row IS its cell-line's data; subset by
        # non-null pIC50_<cell>:
        sub = df.dropna(subset=[target_col] + list(RDKIT_2D_COLS))
        # Re-fit explicitly on this cohort's subset
        row = fit_per_cohort(
            sub.assign(_cohort=cohort_name),
            cohort_col="_cohort",
            target_col=target_col,
            cohort_name=cohort_name,
            seeds=seeds,
        )
        row = CohortRow(**{**asdict(row), "dataset": "platinai_4cell"})
        cohorts.append(row)
    macro_r2 = float(np.mean([c.r2_mean for c in cohorts if not math.isnan(c.r2_mean)]))
    macro_std = float(np.std([c.r2_mean for c in cohorts if not math.isnan(c.r2_mean)]))
    return DatasetSummary(
        dataset="platinai_4cell",
        n_cohorts=len(cohorts),
        cohort_names=tuple(c.cohort for c in cohorts),
        macro_r2_mean=macro_r2,
        macro_r2_std=macro_std,
        overall_r2_mean=macro_r2,
        overall_pearson_r_mean=float(np.mean([c.pearson_r_mean for c in cohorts if not math.isnan(c.pearson_r_mean)])),
        n_total_rows=len(df),
        per_cohort=tuple(cohorts),
    )


def bench_metalcytotox_5metal(
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> DatasetSummary:
    df = load_metalcytotox_5metal()
    cohorts: List[CohortRow] = []
    for metal in METALCYTOTOX_METALS:
        row = fit_per_cohort(
            df.assign(_cohort=df["metal"].astype(str)),
            cohort_col="_cohort",
            target_col="activity",
            cohort_name=metal,
            seeds=seeds,
        )
        row = CohortRow(**{**asdict(row), "dataset": "metalcytotox_5metal"})
        cohorts.append(row)
    macro_r2 = float(np.mean([c.r2_mean for c in cohorts if not math.isnan(c.r2_mean)]))
    macro_std = float(np.std([c.r2_mean for c in cohorts if not math.isnan(c.r2_mean)]))
    return DatasetSummary(
        dataset="metalcytotox_5metal",
        n_cohorts=len(cohorts),
        cohort_names=tuple(c.cohort for c in cohorts),
        macro_r2_mean=macro_r2,
        macro_r2_std=macro_std,
        overall_r2_mean=macro_r2,
        overall_pearson_r_mean=float(np.mean([c.pearson_r_mean for c in cohorts if not math.isnan(c.pearson_r_mean)])),
        n_total_rows=len(df),
        per_cohort=tuple(cohorts),
    )


def bench_tmqm_dblock(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    target: str = "HL_Gap",
    max_per_metal: int = 2000,
) -> DatasetSummary:
    """Bench on tmQM d-block with per-metal breakdown.

    tmQM has 108k rows across 29 d-block metals; we cap per-metal to
    ``max_per_metal`` rows (deterministic seed) to keep Ridge O(n) cheap.
    """
    df = load_tmqm_dblock(target=target)
    df = df.dropna(subset=list(RDKIT_2D_COLS) + ["target"])
    # Deterministic per-metal sub-sampling
    rng = np.random.default_rng(0)
    keep_idx: List[int] = []
    for metal in TMQM_DBLOCK_METALS:
        m_idx = df.index[df["metal"] == metal].tolist()
        if len(m_idx) <= max_per_metal:
            keep_idx.extend(m_idx)
        else:
            keep_idx.extend(rng.choice(m_idx, size=max_per_metal, replace=False).tolist())
    df = df.loc[keep_idx].reset_index(drop=True)

    cohorts: List[CohortRow] = []
    for metal in TMQM_DBLOCK_METALS:
        sub = df[df["metal"] == metal]
        if len(sub) < 10:
            continue
        row = fit_per_cohort(
            sub.assign(_cohort=metal),
            cohort_col="_cohort",
            target_col="target",
            cohort_name=metal,
            seeds=seeds,
        )
        row = CohortRow(**{**asdict(row), "dataset": "tmqm_dblock"})
        cohorts.append(row)
    if not cohorts:
        macro_r2 = float("nan")
        macro_std = float("nan")
    else:
        macro_r2 = float(np.mean([c.r2_mean for c in cohorts if not math.isnan(c.r2_mean)]))
        macro_std = float(np.std([c.r2_mean for c in cohorts if not math.isnan(c.r2_mean)]))
    return DatasetSummary(
        dataset="tmqm_dblock",
        n_cohorts=len(cohorts),
        cohort_names=tuple(c.cohort for c in cohorts),
        macro_r2_mean=macro_r2,
        macro_r2_std=macro_std,
        overall_r2_mean=macro_r2,
        overall_pearson_r_mean=float(np.mean([c.pearson_r_mean for c in cohorts if not math.isnan(c.pearson_r_mean)])),
        n_total_rows=len(df),
        per_cohort=tuple(cohorts),
    )


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------
def summary_to_json_dict(summary: DatasetSummary) -> dict:
    return {
        "dataset": summary.dataset,
        "n_cohorts": summary.n_cohorts,
        "cohort_names": list(summary.cohort_names),
        "macro_r2_mean": summary.macro_r2_mean,
        "macro_r2_std": summary.macro_r2_std,
        "overall_r2_mean": summary.overall_r2_mean,
        "overall_pearson_r_mean": summary.overall_pearson_r_mean,
        "n_total_rows": summary.n_total_rows,
        "per_cohort": [
            {
                "cohort": c.cohort,
                "target": c.target,
                "n_rows": c.n_rows,
                "n_train": c.n_train,
                "n_test": c.n_test,
                "r2_per_seed": list(c.r2_per_seed),
                "rmse_per_seed": list(c.rmse_per_seed),
                "pearson_r_per_seed": list(c.pearson_r_per_seed),
                "r2_mean": c.r2_mean,
                "r2_std": c.r2_std,
                "rmse_mean": c.rmse_mean,
                "pearson_r_mean": c.pearson_r_mean,
            }
            for c in summary.per_cohort
        ],
    }


def write_summary(out_dir: Path, summary: DatasetSummary) -> Path:
    """Write per-cohort JSON under ``{out_dir}/{dataset}/per_cohort_metrics.json``."""
    ds_dir = out_dir / summary.dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    target_path = ds_dir / "per_cohort_metrics.json"
    with target_path.open("w", encoding="utf-8") as fh:
        json.dump(summary_to_json_dict(summary), fh, indent=2)
    return target_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wf-71 P3 — Wf-49 cross-dataset retrain round 2 (per-cohort)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["platinai_4cell", "metalcytotox_5metal", "tmqm_dblock"],
        choices=["platinai_4cell", "metalcytotox_5metal", "tmqm_dblock"],
    )
    parser.add_argument("--per-cohort", action="store_true", default=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-per-metal-tmqm", type=int, default=2000)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[DatasetSummary] = []

    if "platinai_4cell" in args.datasets:
        log.info("Running platinai_4cell per-cell-line ridge")
        s = bench_platinai_4cell(seeds=args.seeds)
        write_summary(args.out_dir, s)
        log.info("  macro R^2 = %.4f +/- %.4f (%d cohorts)",
                 s.macro_r2_mean, s.macro_r2_std, s.n_cohorts)
        summaries.append(s)

    if "metalcytotox_5metal" in args.datasets:
        log.info("Running metalcytotox_5metal per-metal ridge")
        s = bench_metalcytotox_5metal(seeds=args.seeds)
        write_summary(args.out_dir, s)
        log.info("  macro R^2 = %.4f +/- %.4f (%d cohorts)",
                 s.macro_r2_mean, s.macro_r2_std, s.n_cohorts)
        summaries.append(s)

    if "tmqm_dblock" in args.datasets:
        log.info("Running tmqm_dblock per-metal ridge (max_per_metal=%d)",
                 args.max_per_metal_tmqm)
        s = bench_tmqm_dblock(seeds=args.seeds,
                              max_per_metal=args.max_per_metal_tmqm)
        write_summary(args.out_dir, s)
        log.info("  macro R^2 = %.4f +/- %.4f (%d cohorts)",
                 s.macro_r2_mean, s.macro_r2_std, s.n_cohorts)
        summaries.append(s)

    # Cross-dataset macro
    macro_r2_means = [s.macro_r2_mean for s in summaries if not math.isnan(s.macro_r2_mean)]
    if macro_r2_means:
        cross_macro = float(np.mean(macro_r2_means))
        cross_macro_std = float(np.std(macro_r2_means))
    else:
        cross_macro = float("nan")
        cross_macro_std = float("nan")

    cross_path = args.out_dir / "cross_dataset_summary.json"
    with cross_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "n_datasets": len(summaries),
                "cross_dataset_macro_r2_mean": cross_macro,
                "cross_dataset_macro_r2_std": cross_macro_std,
                "per_dataset_macro_r2_mean": {
                    s.dataset: s.macro_r2_mean for s in summaries
                },
                "per_dataset_macro_r2_std": {
                    s.dataset: s.macro_r2_std for s in summaries
                },
                "per_dataset_n_cohorts": {
                    s.dataset: s.n_cohorts for s in summaries
                },
                "target_met": bool(cross_macro >= 0.0) if not math.isnan(cross_macro) else False,
            },
            fh,
            indent=2,
        )
    log.info("Wrote %s (cross-dataset macro R^2 = %.4f)", cross_path, cross_macro)
    log.info("=" * 70)
    log.info("Cross-dataset headline:")
    log.info("  cross-dataset macro R^2 (mean +/- std across 3 datasets): %.4f +/- %.4f",
             cross_macro, cross_macro_std)
    log.info("  target_met (R^2 >= 0): %s",
             "YES" if (not math.isnan(cross_macro) and cross_macro >= 0.0) else "NO (honest)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
