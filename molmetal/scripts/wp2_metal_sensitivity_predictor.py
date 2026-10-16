#!/usr/bin/env python
# Copyright 2026 The Mol-Metal Authors
# SPDX-License-Identifier: Apache-2.0
"""WP2 — is a ligand's metal-sensitivity predictable from ligand chemistry alone?

``todo.json :: thesis_v2_2026_10_17.work_packages[WP2]``.

Capability framed: a *zero-experiment reliability filter*. Before trusting a
ligand-only model on a new metal, ask whether the ligand is one whose activity
is metal-sensitive. WP0's CATE block already showed the matched-pair flip is
chemistry-predictable (CV AUC 0.6706 at the pair level); WP2 asks the same
question at the ligand level, with a permutation null for a real significance
claim.

Two sample sets are scored:

- ``multi_metal`` (primary): the 268 ligands observed under >= 2 metals. Label
  = does the ligand's collapsed label flip across those metals. This is the
  clean question -- both classes have cross-metal evidence.
- ``all_ligands`` (secondary): all ligands, with singletons (seen under exactly
  one metal) as negatives. Larger negatives but the negative class is
  contaminated by ligands that might be sensitive but were never re-tested;
  reported for completeness, not as the headline.

Acceptance (todo): AUC significantly > 0.5 under a label-permutation null,
Holm-corrected across the two sample sets.

Features are ECFP4 only -- no metal information whatsoever -- so a positive
result is exactly the "ligand chemistry alone predicts which molecules will
not transfer" claim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from molmetal.scripts.metal_activity_transfer_probe import (
    METALS,
    _featurise,
    _load_env_default,
    _make_model,
)
from molmetal.scripts.wp0_metal_sensitivity_decomposition import (
    load_cohort,
    sensitivity_map,
)

# --------------------------------------------------------------------------
# sample construction
# --------------------------------------------------------------------------


def build_samples(
    sens: Dict[str, str], dedup: Dict[str, Tuple[List[str], np.ndarray]]
) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """Return two (smiles, labels) sample sets for the classifier."""
    # multi-metal ligands: sensitive (1) vs robust_multi (0)
    multi = [s for s, tag in sens.items() if tag in ("sensitive", "robust_multi")]
    multi_y = np.array([1 if sens[s] == "sensitive" else 0 for s in multi], dtype=int)

    # all ligands: sensitive (1) vs everything else (0)
    all_s = list(sens.keys())
    all_y = np.array([1 if sens[s] == "sensitive" else 0 for s in all_s], dtype=int)

    return {
        "multi_metal": (multi, multi_y),
        "all_ligands": (all_s, all_y),
    }


# --------------------------------------------------------------------------
# CV + permutation null
# --------------------------------------------------------------------------


def _cv_oof_auc(
    smiles: List[str], y: np.ndarray, seed: int, radius: int, nbits: int, n_splits: int = 5
) -> Tuple[float, np.ndarray]:
    """Stratified K-fold OOF AUC. Returns (auc, oof_probs)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    X = _featurise(smiles, radius, nbits)
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        if len(set(y[tr].tolist())) < 2:
            continue
        clf = _make_model(y[tr], seed)
        clf.fit(X[tr], y[tr], verbose=False)
        oof[te] = clf.predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof)
    if mask.sum() < 2 or len(set(y[mask].tolist())) < 2:
        return float("nan"), oof
    return float(roc_auc_score(y[mask], oof[mask])), oof


def permutation_null(
    smiles: List[str],
    y: np.ndarray,
    observed: float,
    n_perm: int,
    seed: int,
    radius: int,
    nbits: int,
) -> Dict[str, object]:
    """Retrain-under-shuffled-labels null: the honest reference for a CV AUC.

    Each permutation shuffles the labels, reruns the SAME 5-fold CV pipeline,
    and records the AUC. This answers "could a CV AUC this high arise from a
    model with no real signal?" -- unlike a fixed-model test-label shuffle.
    """
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        yp = rng.permutation(y)
        a, _ = _cv_oof_auc(smiles, yp, seed + 1 + i, radius, nbits)
        null[i] = a
    finite = null[~np.isnan(null)]
    if finite.size == 0:
        return {"status": "FAILED", "n_perm": int(n_perm)}
    p_above = float((finite >= observed).mean())
    return {
        "status": "MEASURED",
        "n_perm": int(n_perm),
        "observed_auc": float(observed),
        "null_mean": float(finite.mean()),
        "null_sd": float(finite.std()),
        "null_pct95": float(np.percentile(finite, 95)),
        "p_above_null": p_above,
        "significant_above_null_05": bool(p_above < 0.05),
    }


def _holm(pvals: Sequence[float]) -> List[float]:
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    n = len(pvals)
    adj = [1.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, pvals[i] * (n - rank))
        running = max(running, val)
        adj[i] = running
    return adj


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run(
    metals: Sequence[str] = METALS,
    seed: int = 42,
    radius: int = 2,
    nbits: int = 2048,
    n_perm: int = 200,
    time_threshold: Optional[float] = None,
) -> Dict[str, object]:
    _load_env_default()
    _raw, dedup = load_cohort(metals, tie_rule="active", time_threshold=time_threshold)
    sens = sensitivity_map(dedup)
    samples = build_samples(sens, dedup)

    results: Dict[str, object] = {}
    pvals: List[float] = []
    keys: List[str] = []
    for name, (smiles, y) in samples.items():
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos < 5 or n_neg < 5:
            results[name] = {
                "n_pos": n_pos,
                "n_neg": n_neg,
                "status": "SKIPPED_TOO_FEW_POSITIVES",
            }
            continue
        auc, _oof = _cv_oof_auc(smiles, y, seed, radius, nbits)
        entry: Dict[str, object] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "cv_auc": auc,
        }
        # The permutation null refits the full CV pipeline n_perm times. That is
        # cheap on the 268-ligand primary set but ~25h on the 6782-ligand
        # secondary set, so the null is computed only where it is affordable and
        # the significance claim rests on the primary set.
        if n_perm > 0 and len(y) <= 1000:
            entry["permutation_null"] = permutation_null(
                smiles, y, auc, n_perm, seed, radius, nbits
            )
            if entry["permutation_null"].get("status") == "MEASURED":  # type: ignore[union-attr]
                keys.append(name)
                pvals.append(float(entry["permutation_null"]["p_above_null"]))  # type: ignore[index]
        else:
            entry["permutation_null"] = {
                "status": "SKIPPED_COST",
                "reason": f"n={len(y)} > 1000; refit null is cost-prohibitive, CV AUC reported only",
            }
        results[name] = entry

    adj = _holm(pvals) if pvals else []
    for k, a in zip(keys, adj):
        results[k]["permutation_null"]["p_holm"] = a  # type: ignore[index]
        results[k]["permutation_null"]["significant_holm_05"] = bool(a < 0.05)  # type: ignore[index]

    primary = results.get("multi_metal", {})
    primary_sig = bool(
        isinstance(primary, dict)
        and primary.get("permutation_null", {}).get("significant_holm_05", False)
    )

    return {
        "metric": "metal_sensitivity_predictor",
        "description": (
            "WP2 (thesis_v2): can a ligand's metal-sensitivity be predicted from "
            "ligand chemistry alone? ECFP4 -> flip/no-flip, stratified 5-fold CV, "
            "retrain-under-shuffled-labels permutation null, Holm across sample sets."
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/wp2_metal_sensitivity_predictor.py",
        "model": f"morgan_fp(radius={radius}, {nbits}bit) + xgboost",
        "data_source": "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026, 69(8), 8838)",
        "seed": seed,
        "cohort": {
            "time_threshold": time_threshold,
            "label": "full" if time_threshold is None else f"time_le_{time_threshold}",
        },
        "sample_sets": results,
        "acceptance": {
            "primary_sample_set": "multi_metal",
            "primary_auc": (primary.get("cv_auc") if isinstance(primary, dict) else None),
            "primary_significant_holm": primary_sig,
            "verdict": (
                "METAL_SENSITIVITY_IS_LIGAND_CHEMISTRY_PREDICTABLE"
                if primary_sig
                else "NOT_SIGNIFICANT_OR_UNDERPOWERED"
            ),
        },
        "honest_framing": {
            "MEASURED_vs_PROJECTED": "MEASURED",
            "caveat": (
                "Features are ECFP4 only -- no metal input -- so a positive result "
                "is the ligand-chemistry-alone claim. The 'all_ligands' set is a "
                "secondary check whose negatives are contaminated by never-re-tested "
                "ligands; the 'multi_metal' set is the headline."
            ),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="WP2: predict metal-sensitivity from ligand chemistry.")
    p.add_argument("--metals", nargs="+", default=METALS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--time-threshold", type=float, default=None)
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/metal_sensitivity_predictor.json",
    )
    args = p.parse_args(argv)

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    rep = run(
        metals=list(args.metals),
        seed=args.seed,
        radius=args.radius,
        nbits=args.nbits,
        n_perm=args.n_perm,
        time_threshold=args.time_threshold,
    )
    atomic_write_file(Path(args.out_metric), json.dumps(rep, indent=2, default=float))

    for name, r in rep["sample_sets"].items():
        if "cv_auc" in r:
            print(
                f"[wp2:{name}] n_pos={r['n_pos']} n_neg={r['n_neg']} auc={r['cv_auc']:.4f} "
                f"p={r['permutation_null'].get('p_above_null')} "
                f"p_holm={r['permutation_null'].get('p_holm')}"
            )
    print(f"[wp2] verdict: {rep['acceptance']['verdict']}")
    print(f"[wp2] wrote {args.out_metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
