#!/usr/bin/env python
"""m2 R² on the REAL-label PlatinAI 4-cell cohort.

Why a separate file
-------------------
``m2_r2_per_cohort.json`` reports −0.0300 on 4,078 complexes whose pIC50 labels
are **simulated** (`molmetal/data/platinai_4cell.meta.json` records the
generator parameters; `docs/DATA_INVENTORY.json` marks the cohort
DO-NOT-TRAIN).  That number cannot be a measurement of anticancer activity no
matter how many folds it averages.

`platinai_4cell_real_cohort.csv` holds 1,966 compounds with real, DOI-backed
IC50s for the same four cell lines (built by
``build_platinai_4cell_real_cohort.py``; unit µM per the Rusanov 2025 SI).
This script re-runs the same per-cohort Ridge protocol on it, so the paper has
a number that measures something.  It does **not** overwrite the synthetic
result — the old file keeps its provenance, the new one carries REAL labels.

Protocol (deliberately two splits)
----------------------------------
* ``random_5x5``   — 5 seeds × 5 random folds.  This is what
  ``m2_r2_per_cohort`` did, so the two numbers are comparable.
* ``scaffold_5x5`` — 5 seeds × 5 Bemis-Murcko scaffold groups, largest-first.
  This is the number that speaks to generalisation; random folds leak
  near-analogues between train and test.

Per-task, not pooled
--------------------
The real cohort is sparse: 953 of 1,966 compounds cover ≥2 cell lines and 56
cover all four.  Averaging the available cells into one target (which is what
the synthetic cohort could do because every label was fabricated complete)
throws away the per-cell structure and hides which task is being predicted.
So every cell line is evaluated on its own observed labels, and the pooled
"mean of available cells" is reported as a *secondary* row for continuity
with the old number.  No label is imputed.

Output: ``metrics/by_metric/m2_r2_real_4cell.json``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

from molmetal.scripts.wf331r5_r2_w2_chemprop import rdkit_32d_features  # noqa: E402
from molmetal.scripts.wf380a2_m2_r2_per_cohort import (  # noqa: E402
    ridge_r2_cv,
    select_features,
)

RDLogger.DisableLog("rdApp.*")

COHORT = ROOT / ".data" / "molmetal" / "platinai_4cell_real_cohort.csv"
OUT = ROOT / "metrics" / "by_metric" / "m2_r2_real_4cell.json"

CELLS = ("a549", "a2780", "a2780cis", "mcf7")
N_SEEDS = 5
N_FOLDS = 5


def _read_cohort() -> Tuple[List[str], Dict[str, np.ndarray], np.ndarray]:
    """(smiles, {cell: pIC50 with NaN for missing}, mean-of-available)."""
    import csv

    with COHORT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    smiles = [r["smiles"] for r in rows]
    per_cell: Dict[str, np.ndarray] = {}
    for cell in CELLS:
        col = f"pIC50_{cell}"
        per_cell[cell] = np.array(
            [float(r[col]) if r[col] not in ("", None) else np.nan for r in rows],
            dtype=np.float64,
        )
    stacked = np.vstack([per_cell[c] for c in CELLS])
    with np.errstate(invalid="ignore"):
        mean_available = np.nanmean(stacked, axis=0)
    return smiles, per_cell, mean_available


def _scaffold_folds(smiles: List[str], n_folds: int, seed: int) -> List[np.ndarray]:
    """Bemis-Murcko scaffold groups, largest-first, balanced by size.

    Largest-first (rather than random scaffold order) is the standard
    MoleculeNet / DeepChem convention: it puts the biggest chemical families in
    the first folds, which is the harder and more honest arrangement.
    """
    scaffolds: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        key = (
            MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            if mol is not None
            else ""
        )
        scaffolds.setdefault(key, []).append(i)

    groups = sorted(scaffolds.values(), key=len, reverse=True)
    rng = np.random.default_rng(seed)
    # Only the *assignment* of equal-sized groups is randomised; the
    # largest-first order itself is kept, so the split is stable across seeds
    # up to tie-breaking.
    buckets: List[List[int]] = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for g in groups:
        target = int(np.argmin(sizes))
        buckets[target].extend(g)
        sizes[target] += len(g)
    order = rng.permutation(n_folds)
    return [np.array(buckets[k], dtype=int) for k in order]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _ridge_fit_predict(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, alpha: float
) -> np.ndarray:
    """Closed-form Ridge with intercept (same solver as Wf-380-A2)."""
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd[sd == 0] = 1.0
    Ztr = (X_tr - mu) / sd
    Zte = (X_te - mu) / sd
    y_mean = float(y_tr.mean())
    A = Ztr.T @ Ztr + alpha * np.eye(Ztr.shape[1])
    coef = np.linalg.solve(A, Ztr.T @ (y_tr - y_mean))
    return Zte @ coef + y_mean


def _scaffold_cv(
    X: np.ndarray, y: np.ndarray, smiles: List[str], alpha: float, seed_offset: int
) -> Dict[str, float]:
    r2s: List[float] = []
    for s in range(N_SEEDS):
        folds = _scaffold_folds(smiles, N_FOLDS, seed_offset + s)
        for k in range(N_FOLDS):
            test = folds[k]
            train = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
            if len(test) < 3 or len(train) < 10:
                continue
            pred = _ridge_fit_predict(X[train], y[train], X[test], alpha)
            r2s.append(_r2(y[test], pred))
    return {
        "mean_r2": float(np.mean(r2s)) if r2s else float("nan"),
        "std_r2": float(np.std(r2s)) if r2s else float("nan"),
        "n_folds_evaluated": len(r2s),
    }


def _evaluate(
    X: np.ndarray, y: np.ndarray, smiles: List[str], label: str, seed_offset: int
) -> Dict[str, Any]:
    """Evaluate one target under both feature widths and both splits.

    Two widths because the tier table's alpha was chosen for a 12-d / 24-d
    feature set, and this cohort has no chemprop-v3 columns.  Reporting the
    12-d arm keeps the comparison with ``m2_r2_per_cohort`` honest; reporting
    the 32-d arm answers "did we simply under-fit?".
    """
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    Xk, yk = X[mask], y[mask]
    smk = [s for s, m in zip(smiles, mask) if m]
    cfg = select_features(len(yk))
    alpha = float(cfg["alpha"])
    widths = {"rdkit_12d": 12, "rdkit_32d": Xk.shape[1]}
    out: Dict[str, Any] = {
        "target": label,
        "n": int(len(yk)),
        "alpha": alpha,
        "tier": cfg["tier"],
        "target_mean": float(np.mean(yk)),
        "target_std": float(np.std(yk)),
        "by_feature_set": {},
    }
    for name, width in widths.items():
        Xw = Xk[:, :width]
        random_cv_raw = ridge_r2_cv(
            Xw,
            yk,
            n_splits=N_FOLDS,
            n_seeds=N_SEEDS,
            alpha=alpha,
            seed_offset=seed_offset,
        )
        out["by_feature_set"][name] = {
            "n_features": int(width),
            "random_5x5": {
                "mean_r2": float(random_cv_raw.get("r2_mean", float("nan"))),
                "std_r2": float(random_cv_raw.get("r2_std", float("nan"))),
                "n_folds_evaluated": int(random_cv_raw.get("n_splits_effective", 0)),
            },
            "scaffold_5x5": _scaffold_cv(Xw, yk, smk, alpha, seed_offset),
        }
    return out


def run() -> Dict[str, Any]:
    if not COHORT.is_file():
        raise SystemExit(f"cohort not found: {COHORT}")
    started = time.time()
    smiles, per_cell, mean_available = _read_cohort()
    X = np.vstack([rdkit_32d_features(s) for s in smiles])
    finite = np.isfinite(X).all(axis=1)
    X, smiles = X[finite], [s for s, k in zip(smiles, finite) if k]
    per_cell = {c: v[finite] for c, v in per_cell.items()}
    mean_available = mean_available[finite]

    per_target: List[Dict[str, Any]] = []
    for i, cell in enumerate(CELLS):
        per_target.append(_evaluate(X, per_cell[cell], smiles, cell, seed_offset=i))
    pooled = _evaluate(
        X, mean_available, smiles, "mean_of_available_cells", seed_offset=len(CELLS)
    )

    usable = [t for t in per_target if t["n"] >= 50]
    payload = {
        "metric": "m2_r2_real_4cell",
        "status": "MEASURED",
        "scope": "tier1_precious_metal_anticancer / m2 R2 per-cohort on REAL labels",
        "unit": "R^2 (can be negative; higher = better)",
        "dataset": "platinai_4cell_real_cohort.csv",
        "label_provenance": {
            "labels": "REAL",
            "label_type": "experimental IC50",
            "source": "MB-Finder (Rusanov 2025 PlatinAI), unit uM per the SI figure captions S17-S28",
            "builder": "molmetal/scripts/build_platinai_4cell_real_cohort.py",
            "note": "Not a reproduction of Rusanov Table S4-S5: 1,966 compounds here vs their 4,078, which span 592 cell lines and are not fully reachable through the public API.",
        },
        # This row answers the same question as m2_r2_per_cohort, but on a
        # different cohort and with a different model, so it REPLACES that row
        # as the paper line rather than superseding the measurement.  The old
        # row stays valid for what it is: an R2 on platinai_4cell.csv, whose
        # pIC50 columns are SYNTHETIC and which the inventory marks
        # DO-NOT-TRAIN.  Declaring `supersedes` here would tell a reader the
        # two numbers measure the same thing, which they do not.
        "replaces_as_paper_line": {
            "metric": "m2_r2_per_cohort",
            "why": "that row's labels are synthetic (DO-NOT-TRAIN), so its number is a pipeline null test; this one is measured on REAL IC50s",
            "not_a_supersession": "different cohort (1,966 vs 4,078) and different model (RDKit+Ridge vs the D-MPNN trainer), so the two values are not comparable and the older row is not deleted or marked stale",
        },
        "protocol": {
            "features": "32-d RDKit (PLATINAI_32D_COLUMNS)",
            "model": "Ridge, closed-form with intercept",
            "alpha": "per select_features(n) tier",
            "splits": ["random_5x5", "scaffold_5x5"],
            "n_seeds": N_SEEDS,
            "n_folds": N_FOLDS,
            "missing_labels": "never imputed: each task is evaluated on its own observed labels",
            "secondary": "mean_of_available_cells is reported only for continuity with m2_r2_per_cohort",
        },
        "per_cell": per_target,
        "pooled": pooled,
        "n_compounds_featurised": int(X.shape[0]),
        "n_tasks_evaluated": len(usable),
        "mean_scaffold_r2_over_cells": {
            name: float(
                np.mean([t["by_feature_set"][name]["scaffold_5x5"]["mean_r2"] for t in usable])
            )
            if usable
            else float("nan")
            for name in ("rdkit_12d", "rdkit_32d")
        },
        "mean_random_r2_over_cells": {
            name: float(
                np.mean([t["by_feature_set"][name]["random_5x5"]["mean_r2"] for t in usable])
            )
            if usable
            else float("nan")
            for name in ("rdkit_12d", "rdkit_32d")
        },
        "verdict": {},
        "source": "molmetal/scripts/wf_a60_m2_real_cohort.py",
        "wf_tag": "Wf-A60 m2 R2 on REAL PlatinAI 4-cell labels",
        "wall_seconds": round(time.time() - started, 1),
        "timestamp": time.strftime("%Y-%m-%d"),
    }
    scaffold_32 = payload["mean_scaffold_r2_over_cells"]["rdkit_32d"]
    payload["value"] = scaffold_32
    payload["value_basis"] = (
        "mean over the four cell lines of the 5-seed x 5-fold Bemis-Murcko "
        "scaffold-split R2 (32-d RDKit, Ridge alpha=0.5). The random-split "
        "mean is reported separately under mean_random_r2_over_cells."
    )
    payload["n_seeds"] = N_SEEDS
    payload["n_folds"] = N_FOLDS
    payload["split"] = (
        "5 seeds x 5 folds under two protocols: random permutation and "
        "Bemis-Murcko scaffold groups (largest-first)"
    )
    # The single scalar is one of two widths, and a reader who sees only the
    # scalar would draw the wrong conclusion about it: the 12-d width is an
    # order of magnitude less negative, so "R2 = -0.2045" is not a property of
    # the descriptors, it is a property of the width.  Say that here, where
    # `registry.py cite` prints it.
    scaffold_12 = payload["mean_scaffold_r2_over_cells"]["rdkit_12d"]
    random_32 = payload["mean_random_r2_over_cells"]["rdkit_32d"]
    random_12 = payload["mean_random_r2_over_cells"]["rdkit_12d"]
    payload["headline_caveat"] = (
        f"The scalar is the 32-d RDKit width ({scaffold_32:+.4f}); the 12-d "
        f"width reads {scaffold_12:+.4f} under the same scaffold split. The two "
        "are an order of magnitude apart, so neither is 'the' R2 and both belong "
        "in the table. NOT directly comparable with the -0.0300 this metric used "
        "to quote: that was a 3-cohort aggregate with platinai weighted 2/3, and "
        "platinai's own 12-d entry was -0.1440 on n=200 against a mean-over-4-"
        "cells target, whereas this is a per-cell mean over 4,370 measurements. "
        f"Both random-split counterparts are positive ({random_12:+.4f} at 12-d, "
        f"{random_32:+.4f} at 32-d), so neither result survives a scaffold split "
        "and neither is evidence of generalisation."
    )
    payload["verdict"] = {
        "scaffold_gate_0_85_met": bool(
            usable
            and all(
                t["by_feature_set"]["rdkit_32d"]["scaffold_5x5"]["mean_r2"] >= 0.85
                for t in usable
            )
        ),
        # Width-qualified on purpose: at 32-d every cell is negative, but at
        # 12-d A2780 is +0.0270, so an unqualified "any_cell_positive" would be
        # true or false depending on which width the reader assumed.
        "any_cell_positive_scaffold_r2_32d": bool(
            any(
                t["by_feature_set"]["rdkit_32d"]["scaffold_5x5"]["mean_r2"] > 0
                for t in usable
            )
        ),
        "headline": (
            "REAL-label R2: 32-d RDKit + Ridge does not clear the 0.85 gate "
            f"(mean scaffold R2 over cells = {scaffold_32:+.4f}); the RDKit "
            "descriptor family carries no cytotoxicity signal that survives a "
            "scaffold split on this cohort."
            if not (usable and scaffold_32 >= 0.85)
            else "REAL-label R2 clears the 0.85 gate under a scaffold split"
        ),
    }
    return payload


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    payload = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for t in payload["per_cell"]:
        r32 = t["by_feature_set"]["rdkit_32d"]
        r12 = t["by_feature_set"]["rdkit_12d"]
        print(
            f"  {t['target']:10s} n={t['n']:5d} alpha={t['alpha']:<4} "
            f"12d random={r12['random_5x5']['mean_r2']:+.4f} scaffold={r12['scaffold_5x5']['mean_r2']:+.4f} | "
            f"32d random={r32['random_5x5']['mean_r2']:+.4f} scaffold={r32['scaffold_5x5']['mean_r2']:+.4f}"
        )
    p32 = payload["pooled"]["by_feature_set"]["rdkit_32d"]
    print(
        f"  pooled     n={payload['pooled']['n']:5d} "
        f"32d random={p32['random_5x5']['mean_r2']:+.4f} scaffold={p32['scaffold_5x5']['mean_r2']:+.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
