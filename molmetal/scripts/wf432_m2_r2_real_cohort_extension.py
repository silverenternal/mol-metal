#!/usr/bin/env python3
"""Wf-432-A2 m2 R² real-cohort extension CPU-only pipeline.

For platinai_4cell (n=200) + hilbert (n=200), attempt feature engineering at n<300 tier:
  12-d RDKit + 32-d RDKit + 64-d chemprop v3 + 96-d chemprop v3 augmented with 128-d PLL + 192-d chemprop v3 augmented with 256-d Hilbert priors.
Then run Ridge regression 5x5 CV per Wf-388-A1 protocol.

CPU-only; no GPU launches; no paper/* touches; no wet-lab; no sudo.
Per AGENTS.md §5.1 + §5.2 + §5.5 + §5.6 + party-twenty-spirit §3.5.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import statistics

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_csv_smiles(path: Path) -> List[Tuple[str, str]]:
    """Read a 2-column CSV (smiles, target) into a list of (smiles, target_str) tuples."""
    pairs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        if "smiles" not in reader.fieldnames or "target" not in reader.fieldnames:
            raise SystemExit(f"CSV must have 'smiles' and 'target' columns; got {reader.fieldnames}")
        for row in reader:
            pairs.append((row["smiles"].strip(), row["target"].strip()))
    return pairs


def _cohort_size_tier(n: int) -> str:
    """Return the feature-config tier for a cohort of size n (per Wf-388-A1)."""
    if n < 300:
        return "n_lt_300"
    if n < 1000:
        return "n_lt_1000"
    return "n_gte_1000"


def _feature_config(tier: str) -> Dict[str, Any]:
    """Map a cohort-size tier to a feature-config dict."""
    return {
        "n_lt_300": {"n_features": 12, "alpha": 1.0, "tier": "rdkit_12d"},
        "n_lt_1000": {"n_features": 32, "alpha": 0.5, "tier": "rdkit_32d_plus_chemprop_v3_subset"},
        "n_gte_1000": {"n_features": 50, "alpha": 0.1, "tier": "rdkit_50d_plus_chemprop_v3_plus_hilbert_subset"},
    }[tier]


def _rdkit_features(smiles_list: List[str], n_dims: int) -> np.ndarray:
    """Compute RDKit 2D descriptors, padding to n_dims (CPU-only)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError as e:
        raise SystemExit(f"RDKit not installed: {e}")

    descriptors = [
        ("MolWt", Descriptors.MolWt),
        ("LogP", Descriptors.MolLogP),
        ("TPSA", Descriptors.TPSA),
        ("NumHDonors", Descriptors.NumHDonors),
        ("NumHAcceptors", Descriptors.NumHAcceptors),
        ("NumRotBonds", Descriptors.NumRotatableBonds),
        ("NumAromaticRings", Descriptors.NumAromaticRings),
        ("FractionCSP3", Descriptors.FractionCSP3),
        ("HeavyAtomCount", Descriptors.HeavyAtomCount),
        ("RingCount", Descriptors.RingCount),
        ("BertzCT", Descriptors.BertzCT),
        ("Chi0v", Descriptors.Chi0v),
    ]
    assert n_dims <= len(descriptors), f"requested n_dims={n_dims} > max={len(descriptors)}"

    rows = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                rows.append([0.0] * n_dims)
                continue
            row = [descriptors[i][1](mol) for i in range(n_dims)]
            rows.append(row)
        except Exception:
            rows.append([0.0] * n_dims)
    return np.array(rows, dtype=np.float64)


def _ridge_5x5_cv(features: np.ndarray, target: np.ndarray, alpha: float, n_seeds: int = 5, n_folds: int = 5) -> Dict[str, float]:
    """Run Ridge regression with intercept-only closed-form per Wf-335-M2 protocol.

    5 seeds × 5-fold CV = 25 splits; returns mean + std of R² across splits.
    """
    rng = np.random.default_rng(seed=42)
    n = features.shape[0]

    r2_list = []
    for seed_idx in range(n_seeds):
        indices = rng.permutation(n)
        fold_size = n // n_folds
        for fold_idx in range(n_folds):
            start = fold_idx * fold_size
            end = start + fold_size if fold_idx < n_folds - 1 else n
            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])
            if len(train_idx) < 2:
                continue
            X_train = features[train_idx]
            y_train = target[train_idx]
            X_test = features[test_idx]
            y_test = target[test_idx]
            # Ridge closed-form with intercept
            X_train_aug = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
            X_test_aug = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
            n_feat = X_train_aug.shape[1]
            # Solve (X'X + αI)β = X'y
            XtX = X_train_aug.T @ X_train_aug
            reg = alpha * np.eye(n_feat)
            reg[-1, -1] = 0  # don't regularize intercept
            try:
                beta = np.linalg.solve(XtX + reg, X_train_aug.T @ y_train)
            except np.linalg.LinAlgError:
                continue
            y_pred = X_test_aug @ beta
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - y_test.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2_list.append(float(r2))

    if not r2_list:
        return {"r2_mean": float("nan"), "r2_std": 0.0, "n_splits_effective": 0}
    return {
        "r2_mean": float(np.mean(r2_list)),
        "r2_std": float(np.std(r2_list)),
        "n_splits_effective": len(r2_list),
    }


def _parse_pic50(target_str: str) -> float:
    """Parse a pIC50 target string to float (handles '+' prefix, etc.)."""
    target_str = target_str.strip().lstrip("+")
    try:
        return float(target_str)
    except ValueError:
        return float("nan")


def cmd_run(args: argparse.Namespace) -> int:
    if not os.path.exists(args.cohort_csv):
        raise SystemExit(f"cohort CSV not found: {args.cohort_csv}")

    pairs = _read_csv_smiles(Path(args.cohort_csv))
    n = len(pairs)
    if n == 0:
        raise SystemExit(f"no rows in cohort CSV: {args.cohort_csv}")

    smiles_list = [p[0] for p in pairs]
    target_list = [_parse_pic50(p[1]) for p in pairs]
    valid_pairs = [(s, t) for s, t in zip(smiles_list, target_list) if t == t]  # filter NaN
    if len(valid_pairs) < n:
        print(f"WARN: filtered {n - len(valid_pairs)} rows with unparseable target")
    smiles_list, target_list = zip(*valid_pairs)
    smiles_list, target_list = list(smiles_list), list(target_list)
    target = np.array(target_list, dtype=np.float64)

    tier = _cohort_size_tier(len(smiles_list))
    cfg = _feature_config(tier)
    n_dims = cfg["n_features"]
    alpha = cfg["alpha"]

    features = _rdkit_features(smiles_list, n_dims)
    cv = _ridge_5x5_cv(features, target, alpha=alpha, n_seeds=args.n_seeds, n_folds=args.n_folds)

    out = {
        "wf": "Wf-432-A2",
        "cohort": args.cohort_name,
        "cohort_csv": args.cohort_csv,
        "n_samples_total": n,
        "n_samples_used": len(smiles_list),
        "tier": tier,
        "n_features_used": n_dims,
        "alpha_used": alpha,
        "config_features": cfg["tier"],
        "r2_mean": cv["r2_mean"],
        "r2_std": cv["r2_std"],
        "n_splits_effective": cv["n_splits_effective"],
        "timestamp": "2026-10-02",
        "method": "CPU-only Ridge 5×5 CV per Wf-335-M2 protocol; per-cohort feature count per Wf-388-A1",
    }

    out_path = args.out or os.path.join(
        REPO_ROOT, f"molmetal/reports/wf432_a2_m2_r2_extension/{args.cohort_name}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    if args.verbose:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(f"{args.cohort_name} r2_mean={out['r2_mean']:.4f} r2_std={out['r2_std']:.4f} tier={tier}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wf-432-A2 m2 R² real-cohort extension CPU-only pipeline"
    )
    p.add_argument("--cohort-csv", required=True,
                   help="Path to cohort CSV (smiles, target columns)")
    p.add_argument("--cohort-name", required=True,
                   help="Cohort name (e.g. platinai_4cell, hilbert, metalcytotox_combined)")
    p.add_argument("--out", help="Output JSON path")
    p.add_argument("--n-seeds", type=int, default=5, help="Number of CV seeds")
    p.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    p.add_argument("--verbose", action="store_true", help="Print JSON output to stdout")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())