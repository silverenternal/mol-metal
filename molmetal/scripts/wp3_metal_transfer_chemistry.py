#!/usr/bin/env python
# Copyright 2026 The Mol-Metal Authors
# SPDX-License-Identifier: Apache-2.0
"""WP3 — does metal chemistry drive transfer? (zero-experiment metal-swap)

``todo.json :: thesis_v2_2026_10_17.work_packages[WP3]``.

Capability framed: zero-experiment metal-swap prediction + metal recommendation.
"Without synthesising anything, can we say whether swapping the metal is likely
to work -- and which metal to swap to?"

Three pieces:

1. **Metal-metal distance matrix** from transferable descriptors (rich 25-d +
   chemistry 3-d), normalised, Euclidean. Computable for all 10 metal pairs.

2. **Distance vs measured transfer.** Correlate the descriptor distance with the
   measured matched-pair transfer (label-flip rate / mean |Delta pIC50|). HONEST
   LIMITATION: only 3 metal pairs share matched ligands in this cohort (Ru-Rh,
   Ru-Os, Ir-Rh), so this correlation has n=3 and is reported as a data-limited
   observation, not a finding. Also correlated against the per-metal LOMO AUC
   (n=5), which is a weaker but better-populated proxy.

3. **CATE with metal descriptors.** Extend WP0's ligand-only CATE model
   (ECFP4 -> flip, CV AUC 0.6706) with the metal-descriptor difference. If metal
   chemistry adds predictive power beyond ligand chemistry, the metal-swap
   effect is not purely a ligand property -- it is a ligand x metal property,
   which is what makes a *recommendation* possible.

Acceptance (todo): descriptor distance vs measured transfer significantly
correlated (reported with the n=3 caveat); CATE prediction AUC significantly
> 0.5.
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
from molmetal.scripts.wp0_metal_sensitivity_decomposition import load_cohort

# --------------------------------------------------------------------------
# metal descriptors
# --------------------------------------------------------------------------


def metal_descriptors(metals: Sequence[str]) -> Dict[str, np.ndarray]:
    """rich (25-d) + chemistry (3-d) transferable descriptor per metal."""
    from molmetal.data.metal_graph import chem_metal_block, rich_metal_block

    out: Dict[str, np.ndarray] = {}
    for m in metals:
        out[m] = np.concatenate([rich_metal_block(m), chem_metal_block(m)])
    return out


def distance_matrix(desc: Dict[str, np.ndarray], metals: Sequence[str]) -> Dict[str, object]:
    """Z-normalise each descriptor column, then Euclidean distance."""
    M = np.vstack([desc[m] for m in metals])
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (M - mu) / sd
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=-1))
    pairs = {}
    for i, m1 in enumerate(metals):
        for j in range(i + 1, len(metals)):
            m2 = metals[j]
            pairs[f"{m1}-{m2}"] = float(D[i, j])
    return {
        "normalisation": "per-column z-score across the 5 metals, then Euclidean",
        "n_features": int(Z.shape[1]),
        "pairwise_distance": pairs,
    }


# --------------------------------------------------------------------------
# measured transfer
# --------------------------------------------------------------------------


def measured_transfer(
    raw_rows: Dict[str, List[Tuple[str, str, int, float]]], metals: Sequence[str]
) -> Dict[str, object]:
    """Per metal pair, the matched-pair transfer from raw rows."""
    pair_eff: Dict[Tuple[str, str], Dict[str, Tuple[int, float]]] = defaultdict(dict)
    for m in metals:
        for smi, cell, y, p in raw_rows[m]:
            pair_eff[(smi, cell)][m] = (y, p)

    per_pair: Dict[str, object] = {}
    for i, m1 in enumerate(metals):
        for m2 in metals[i + 1 :]:
            flips: List[int] = []
            adps: List[float] = []
            for _k, pm in pair_eff.items():
                if m1 not in pm or m2 not in pm:
                    continue
                y1, p1 = pm[m1]
                y2, p2 = pm[m2]
                flips.append(int(y1 != y2))
                if not (math.isnan(p1) or math.isnan(p2)):
                    adps.append(abs(p2 - p1))
            if flips:
                per_pair[f"{m1}-{m2}"] = {
                    "n_shared_pairs": len(flips),
                    "label_flip_rate": float(np.mean(flips)),
                    "mean_abs_delta_pic50": (float(np.mean(adps)) if adps else None),
                }
    return per_pair


def _spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    from scipy.stats import spearmanr

    if len(x) < 3:
        return float("nan"), float("nan")
    r, p = spearmanr(x, y)
    return float(r), float(p)


# --------------------------------------------------------------------------
# CATE with metal descriptors
# --------------------------------------------------------------------------


def cate_with_metal(
    raw_rows: Dict[str, List[Tuple[str, str, int, float]]],
    desc: Dict[str, np.ndarray],
    metals: Sequence[str],
    seed: int,
    radius: int,
    nbits: int,
) -> Dict[str, object]:
    """Predict the per-pair flip from ligand ECFP4 alone vs ligand+metal-diff.

    Grouped CV by ligand so a fingerprint never spans train and test.
    """
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score

    pair_eff: Dict[Tuple[str, str], Dict[str, Tuple[int, float]]] = defaultdict(dict)
    for m in metals:
        for smi, cell, y, p in raw_rows[m]:
            pair_eff[(smi, cell)][m] = (y, p)

    rows_smi: List[str] = []
    rows_dmetal: List[np.ndarray] = []
    rows_y: List[int] = []
    for i, m1 in enumerate(metals):
        for m2 in metals[i + 1 :]:
            dm = desc[m2] - desc[m1]
            for (smi, _cell), pm in pair_eff.items():
                if m1 not in pm or m2 not in pm:
                    continue
                y1, _p1 = pm[m1]
                y2, _p2 = pm[m2]
                rows_smi.append(smi)
                rows_dmetal.append(dm)
                rows_y.append(int(y1 != y2))

    y = np.array(rows_y, dtype=int)
    if len(set(y.tolist())) < 2 or len(y) < 40:
        return {"status": "UNDERPOWERED", "n": int(len(y))}

    Xlig = _featurise(rows_smi, radius, nbits)
    Xmetal = np.vstack(rows_dmetal)
    groups = np.array(rows_smi, dtype=object)
    n_groups = len(set(groups.tolist()))
    gkf = GroupKFold(n_splits=min(5, n_groups))

    def _cv(X: np.ndarray) -> Optional[float]:
        oof = np.full(len(y), np.nan)
        for tr, te in gkf.split(X, y, groups):
            if len(set(y[tr].tolist())) < 2:
                continue
            clf = _make_model(y[tr], seed)
            clf.fit(X[tr], y[tr], verbose=False)
            oof[te] = clf.predict_proba(X[te])[:, 1]
        mask = ~np.isnan(oof)
        if mask.sum() < 2 or len(set(y[mask].tolist())) < 2:
            return None
        return float(roc_auc_score(y[mask], oof[mask]))

    auc_lig = _cv(Xlig)
    auc_lig_metal = _cv(np.hstack([Xlig, Xmetal]))
    auc_metal_only = _cv(Xmetal)

    delta = None
    if auc_lig is not None and auc_lig_metal is not None:
        delta = auc_lig_metal - auc_lig

    return {
        "status": "MEASURED",
        "n_pairs": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_groups": n_groups,
        "cv_auc_ligand_only": auc_lig,
        "cv_auc_metal_only": auc_metal_only,
        "cv_auc_ligand_plus_metal": auc_lig_metal,
        "delta_from_metal": delta,
        "verdict": (
            "METAL_CHEMISTRY_ADDS_SIGNAL"
            if (delta is not None and delta > 0.02)
            else (
                "LIGAND_INTRINSIC_NOT_METAL_DRIVEN"
                if (auc_lig is not None and auc_lig > 0.5)
                else "NO_SIGNAL"
            )
        ),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def _wp3_verdict(cate: Dict[str, object], n_pairs: int, pval: float) -> str:
    """Honest verdict: metal chemistry only 'drives transfer' if it ADDS signal
    over ligand chemistry, not merely if the combined AUC clears 0.5."""
    if cate.get("status") != "MEASURED":
        return "NOT_SUPPORTED_OR_UNDERPOWERED"
    auc_lig = cate.get("cv_auc_ligand_only") or 0.0
    delta = cate.get("delta_from_metal")
    if delta is None:
        return "NOT_SUPPORTED_OR_UNDERPOWERED"
    if delta > 0.02:
        return "METAL_CHEMISTRY_ADDS_SIGNAL"
    if auc_lig > 0.5:
        return "LIGAND_INTRINSIC_NOT_METAL_DRIVEN"
    return "NOT_SUPPORTED_OR_UNDERPOWERED"


def run(
    metals: Sequence[str] = METALS,
    seed: int = 42,
    radius: int = 2,
    nbits: int = 2048,
    time_threshold: Optional[float] = None,
) -> Dict[str, object]:
    _load_env_default()
    raw_rows, _dedup = load_cohort(metals, tie_rule="active", time_threshold=time_threshold)
    desc = metal_descriptors(metals)

    dist = distance_matrix(desc, metals)
    transfer = measured_transfer(raw_rows, metals)
    cate = cate_with_metal(raw_rows, desc, metals, seed, radius, nbits)

    # correlate descriptor distance vs measured transfer (data-limited)
    common = sorted(set(dist["pairwise_distance"]) & set(transfer))
    d_vals = [dist["pairwise_distance"][k] for k in common]
    t_vals = [transfer[k]["label_flip_rate"] for k in common]
    rho, pval = _spearman(d_vals, t_vals)

    return {
        "metric": "metal_transfer_chemistry",
        "description": (
            "WP3 (thesis_v2): does metal chemistry drive cross-metal transfer? "
            "Descriptor-distance matrix, distance-vs-measured-transfer correlation, "
            "and a CATE model that adds metal descriptors to ligand ECFP4."
        ),
        "unit": "ROC-AUC (0-1); distances are z-normalised Euclidean; rho is Spearman",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/wp3_metal_transfer_chemistry.py",
        "data_source": "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026, 69(8), 8838)",
        "seed": seed,
        "cohort": {
            "time_threshold": time_threshold,
            "label": "full" if time_threshold is None else f"time_le_{time_threshold}",
        },
        "metal_distance_matrix": dist,
        "measured_transfer_by_metal_pair": transfer,
        "distance_vs_transfer": {
            "n_metal_pairs_with_data": len(common),
            "metal_pairs": common,
            "spearman_rho": rho,
            "spearman_p": pval,
            "caveat": (
                "n=3 metal pairs share matched ligands -- this correlation is "
                "data-limited and reported as an observation, not a finding."
            ),
        },
        "cate_with_metal": cate,
        "acceptance": {
            "distance_vs_transfer_significant": bool(
                not math.isnan(pval) and pval < 0.05 and len(common) >= 5
            ),
            "distance_vs_transfer_note": (
                f"n={len(common)} metal pairs -- below the n>=5 bar, so the "
                "correlation is reported but not accepted as evidence."
            ),
            "cate_auc_gt_0.5": bool(
                cate.get("status") == "MEASURED"
                and (cate.get("cv_auc_ligand_plus_metal") or 0) > 0.5
            ),
            # The load-bearing test is not "is the combined AUC > 0.5" (ligand
            # chemistry alone clears that) but "does metal chemistry ADD signal".
            "metal_adds_signal_delta_gt_0.02": bool(
                cate.get("status") == "MEASURED"
                and (cate.get("delta_from_metal") or 0) > 0.02
            ),
            "verdict": _wp3_verdict(cate, len(common), pval),
        },
        "honest_framing": {
            "MEASURED_vs_PROJECTED": "MEASURED",
            "caveat": (
                "The distance-vs-transfer correlation is n=3 (only 3 metal pairs "
                "share matched ligands); the CATE-with-metal model is the load-"
                "bearing result."
            ),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="WP3: metal chemistry drives transfer.")
    p.add_argument("--metals", nargs="+", default=METALS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument("--time-threshold", type=float, default=None)
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/metal_transfer_chemistry.json",
    )
    args = p.parse_args(argv)

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    rep = run(
        metals=list(args.metals),
        seed=args.seed,
        radius=args.radius,
        nbits=args.nbits,
        time_threshold=args.time_threshold,
    )
    atomic_write_file(Path(args.out_metric), json.dumps(rep, indent=2, default=float))

    dv = rep["distance_vs_transfer"]
    print(f"[wp3] distance-vs-transfer: n={dv['n_metal_pairs_with_data']} rho={dv['spearman_rho']} p={dv['spearman_p']}")
    c = rep["cate_with_metal"]
    print(
        f"[wp3] CATE: lig_only={c.get('cv_auc_ligand_only')} "
        f"metal_only={c.get('cv_auc_metal_only')} "
        f"lig+metal={c.get('cv_auc_ligand_plus_metal')} delta={c.get('delta_from_metal')}"
    )
    print(f"[wp3] verdict: {rep['acceptance']['verdict']}")
    print(f"[wp3] wrote {args.out_metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
