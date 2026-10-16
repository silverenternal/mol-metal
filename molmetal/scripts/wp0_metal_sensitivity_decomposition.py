#!/usr/bin/env python
# Copyright 2026 The Mol-Metal Authors
# SPDX-License-Identifier: Apache-2.0
"""WP0 + WP1 — power analysis, LOMO stratified attribution, matched-pair ATE/CATE.

``todo.json :: thesis_v2_2026_10_17.work_packages[WP0, WP1]``. WP0 and WP1 share
one deliverable (``metrics/by_metric/metal_sensitivity_decomposition.json``) and
one acceptance rule (robust-subset LOMO recovery >= 90% of within-metal), so they
live in one script.

Four questions, all answerable on CPU with zero GPU:

1. **Matched-pair evidence.** A ``(ligand, cell-line)`` pair seen under >= 2
   metals flips its activity label ~18% of the time, against a
   repeated-measurement noise floor of ~9.6%. Re-derived here from the raw CSV,
   never trusted from a stored artifact.

2. **Power.** WP2 (predict metal-sensitivity from ligand chemistry alone) and WP3
   (predict the metal-swap effect) both turn on whether a *classifier* can
   separate on that signal. With the ligand count we actually have, what AUC is
   detectable at 80% power? If the answer sits below the target 0.65, the v2 plan
   needs more data or a different angle first.

3. **Attribution (WP1).** The aggregate LOMO collapse is one number per metal.
   Split each held-out metal's ligands into *metal-sensitive* (label flips across
   the metals it appears under) and *metal-robust* (stable), and measure LOMO AUC
   on each. If robust recovers to >= 90% of within-metal, the transfer penalty is
   attributable to the sensitive subset -- a locatable mechanism.

4. **ATE / CATE (WP1).** On matched pairs, the treatment is the metal. Estimate
   the average treatment effect per metal pair (Delta pIC50, label-flip rate) and
   ask whether the effect is heterogeneous in ligand chemistry (a ligand-feature
   model vs the constant-ATE baseline).

**Cohort.** The cited evidence (268 multi-metal ligands / 668 pairs) is the FULL
cohort with no incubation-time filter. The published transfer penalty
(``metal_activity_transfer.json``) is the ``Time(h) <= 24`` cohort. Both are
computed and reported side by side so neither is silently conflated with the
other.

Scorer is identical to ``metal_activity_transfer_probe.py`` (Morgan FP +
XGBoost) so the numbers are comparable to the published transfer penalty.
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
    collapse_duplicates,
)

# raw row = (smiles, cell_line, label, pic50)
RawRow = Tuple[str, str, int, float]

# --------------------------------------------------------------------------
# cohort loading
# --------------------------------------------------------------------------


def load_cohort(
    metals: Sequence[str],
    tie_rule: str = "active",
    time_threshold: Optional[float] = None,
) -> Tuple[Dict[str, List[RawRow]], Dict[str, Tuple[List[str], np.ndarray]]]:
    """Return (raw_rows, dedup).

    ``raw_rows[m]`` is a list of ``(smiles, cell_line, label, pic50)`` tuples at
    the row level (duplicates kept) -- needed for the noise-floor estimate and
    the matched-pair ATE, both of which live below ligand granularity.

    ``dedup[m]`` is ``(unique_smiles, labels)`` -- one row per ligand, the
    cohort the transfer probe scores on.

    ``time_threshold=None`` keeps every incubation time (the FULL cohort, which
    the cited 268-ligand / 668-pair evidence uses); ``24.0`` reproduces the
    published transfer-penalty cohort.
    """
    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

    raw_rows: Dict[str, List[RawRow]] = {}
    dedup: Dict[str, Tuple[List[str], np.ndarray]] = {}

    for m in metals:
        ds = MetalCytotoxDataset.from_csv(
            filters=CytotoxFilter(
                time_threshold=time_threshold, ic50_min=0.01, metal_whitelist=[m]
            )
        )
        df = ds.df
        smi = df["SMILES_Ligands"].astype(str).to_numpy()
        cell = (
            df["Cell_line"].astype(str).to_numpy()
            if "Cell_line" in df.columns
            else np.array([""] * len(df))
        )
        act = df["active"].astype(bool).to_numpy()
        pic = (
            df["pIC50"].astype(float).to_numpy()
            if "pIC50" in df.columns
            else np.full(len(df), np.nan)
        )
        raw_rows[m] = [
            (str(s), str(c), int(bool(a)), float(p))
            for s, c, a, p in zip(smi, cell, act, pic)
        ]
        s_u, y_u, _n_raw = collapse_duplicates(ds.smiles, ds.active, tie_rule=tie_rule)
        dedup[m] = (list(s_u), np.asarray(y_u, dtype=int))

    return raw_rows, dedup


# --------------------------------------------------------------------------
# matched-pair evidence (the 18.2% vs 9.6% pair)
# --------------------------------------------------------------------------


def _two_prop_z_p(k1: int, n1: int, k0: int, n0: int) -> float:
    """One-sided two-proportion z-test that group 1's rate exceeds group 0's."""
    if n1 == 0 or n0 == 0:
        return float("nan")
    p1 = k1 / n1
    p0 = k0 / n0
    p_pool = (k1 + k0) / (n1 + n0)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n0))
    if se == 0.0:
        return 0.0
    z = (p1 - p0) / se
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def matched_pair_evidence(
    raw_rows: Dict[str, List[RawRow]], metals: Sequence[str]
) -> Dict[str, object]:
    """Recompute the matched-ligand flip rate and the repeated-measurement noise
    floor from the raw rows."""
    pair_labels: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(dict)
    for m in metals:
        for smi, cell, y, _p in raw_rows[m]:
            pair_labels[(smi, cell)][m] = y

    n_pairs_multi = 0
    n_pairs_flip = 0
    for _key, per_metal in pair_labels.items():
        if len(per_metal) < 2:
            continue
        n_pairs_multi += 1
        if len(set(per_metal.values())) > 1:
            n_pairs_flip += 1

    triple: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for m in metals:
        for smi, cell, y, _p in raw_rows[m]:
            triple[(smi, cell, m)].append(y)

    n_triple_rep = 0
    n_triple_flip = 0
    for _key, labels in triple.items():
        if len(labels) < 2:
            continue
        n_triple_rep += 1
        if len(set(labels)) > 1:
            n_triple_flip += 1

    flip_rate = n_pairs_flip / n_pairs_multi if n_pairs_multi else float("nan")
    noise_rate = n_triple_flip / n_triple_rep if n_triple_rep else float("nan")
    p_val = _two_prop_z_p(n_pairs_flip, n_pairs_multi, n_triple_flip, n_triple_rep)

    return {
        "cross_metal_pairs": {
            "definition": "(ligand, cell-line) observed under >= 2 distinct metals",
            "n_pairs": int(n_pairs_multi),
            "n_label_flips": int(n_pairs_flip),
            "flip_rate": float(flip_rate),
        },
        "repeated_measurement_noise_floor": {
            "definition": "(ligand, cell-line, metal) observed more than once",
            "n_repeated": int(n_triple_rep),
            "n_contradictory": int(n_triple_flip),
            "contradiction_rate": float(noise_rate),
        },
        "flip_vs_noise": {
            "observed_flip_rate": float(flip_rate),
            "noise_floor_rate": float(noise_rate),
            "two_proportion_ztest_p_one_sided": float(p_val),
            "interpretation": (
                "Same x (ligand, cell-line), same assay, metal changed -> y changes "
                "above the measurement's own noise floor: p(y|x,m) != p(y|x)."
            ),
        },
    }


# --------------------------------------------------------------------------
# ligand metal-sensitivity map
# --------------------------------------------------------------------------


def sensitivity_map(dedup: Dict[str, Tuple[List[str], np.ndarray]]) -> Dict[str, str]:
    """Classify every ligand as 'sensitive' | 'robust_multi' | 'singleton'.

    - ``sensitive``     : seen under >= 2 metals with >= 2 distinct labels
    - ``robust_multi``  : seen under >= 2 metals, label identical everywhere
    - ``singleton``     : seen under exactly one metal
    """
    per_ligand: Dict[str, List[int]] = defaultdict(list)
    for _m, (s_list, y_arr) in dedup.items():
        for s, y in zip(s_list, y_arr.tolist()):
            per_ligand[s].append(int(y))

    out: Dict[str, str] = {}
    for s, labels in per_ligand.items():
        if len(labels) < 2:
            out[s] = "singleton"
        elif len(set(labels)) > 1:
            out[s] = "sensitive"
        else:
            out[s] = "robust_multi"
    return out


# --------------------------------------------------------------------------
# LOMO with per-ligand score dump
# --------------------------------------------------------------------------


def lomo_with_scores(
    dedup: Dict[str, Tuple[List[str], np.ndarray]],
    seed: int,
    radius: int,
    nbits: int,
) -> Dict[str, object]:
    """Leave-one-metal-out, keeping per-ligand (smiles, label, prob).

    Same leakage rule as the probe: a ligand seen in ANY other metal is dropped
    from the training pool, so no fingerprint straddles the split.
    """
    from sklearn.metrics import roc_auc_score

    folds: List[Dict[str, object]] = []
    pooled: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)

    for held_out, (te_s, te_y) in dedup.items():
        if len(set(te_y.tolist())) < 2:
            folds.append({"metal": held_out, "status": "SKIPPED", "reason": "single-class"})
            continue
        blocked = set(te_s)
        tr_s: List[str] = []
        tr_y: List[int] = []
        for m, (s_list, y_arr) in dedup.items():
            if m == held_out:
                continue
            for s, yv in zip(s_list, y_arr.tolist()):
                if s in blocked:
                    continue
                tr_s.append(s)
                tr_y.append(int(yv))
        if len(set(tr_y)) < 2:
            folds.append(
                {"metal": held_out, "status": "SKIPPED", "reason": "train_pool_single_class"}
            )
            continue
        ytr = np.asarray(tr_y, dtype=int)
        Xtr = _featurise(tr_s, radius, nbits)
        Xte = _featurise(te_s, radius, nbits)
        clf = _make_model(ytr, seed)
        clf.fit(Xtr, ytr, verbose=False)
        prob = clf.predict_proba(Xte)[:, 1]

        for s, yv, pv in zip(te_s, te_y.tolist(), prob.tolist()):
            pooled[held_out].append((str(s), int(yv), float(pv)))

        folds.append(
            {
                "metal": held_out,
                "status": "MEASURED",
                "roc_auc": float(roc_auc_score(te_y, prob)),
                "n_test_unique": int(len(te_y)),
                "n_train_unique": int(len(tr_y)),
            }
        )

    return {"folds": folds, "per_ligand": dict(pooled)}


# --------------------------------------------------------------------------
# stratified attribution
# --------------------------------------------------------------------------


def _auc(labels: Sequence[int], probs: Sequence[float]) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return None
    return float(roc_auc_score(labels, probs))


def stratified_attribution(
    lomo: Dict[str, object],
    sens: Dict[str, str],
    within_auc: Dict[str, Optional[float]],
) -> Dict[str, object]:
    """Pooled LOMO AUC on the sensitive vs robust subsets, plus the recovery
    ratio against the within-metal AUC."""
    per_ligand: Dict[str, List[Tuple[str, int, float]]] = lomo["per_ligand"]  # type: ignore[assignment]

    strata: Dict[str, Dict[str, List]] = {
        "sensitive": {"y": [], "p": []},
        "robust": {"y": [], "p": []},  # robust_multi + singleton
    }
    per_metal: Dict[str, Dict[str, Dict[str, List]]] = defaultdict(
        lambda: {"sensitive": {"y": [], "p": []}, "robust": {"y": [], "p": []}}
    )

    for metal, rows in per_ligand.items():
        for s, yv, pv in rows:
            tag = sens.get(s, "singleton")
            bucket = "sensitive" if tag == "sensitive" else "robust"
            strata[bucket]["y"].append(yv)
            strata[bucket]["p"].append(pv)
            per_metal[metal][bucket]["y"].append(yv)
            per_metal[metal][bucket]["p"].append(pv)

    out: Dict[str, object] = {}
    for bucket in ("sensitive", "robust"):
        y = strata[bucket]["y"]
        p = strata[bucket]["p"]
        out[bucket] = {
            "n": int(len(y)),
            "base_rate": float(np.mean(y)) if y else None,
            "lomo_auc": _auc(y, p),
        }

    within_vals = [v for v in within_auc.values() if v is not None]
    within_mean = float(np.mean(within_vals)) if within_vals else None
    robust_auc = out["robust"]["lomo_auc"]  # type: ignore[index]
    sens_auc = out["sensitive"]["lomo_auc"]  # type: ignore[index]

    recovery = None
    if within_mean and robust_auc is not None and within_mean > 0:
        recovery = robust_auc / within_mean

    out["recovery"] = {
        "within_metal_auc_mean": within_mean,
        "robust_lomo_auc": robust_auc,
        "sensitive_lomo_auc": sens_auc,
        "robust_recovery_ratio": recovery,
        "acceptance_ge_90pct": bool(recovery is not None and recovery >= 0.90),
    }
    out["per_metal"] = {
        m: {
            "sensitive": {
                "n": len(b["sensitive"]["y"]),
                "auc": _auc(b["sensitive"]["y"], b["sensitive"]["p"]),
            },
            "robust": {
                "n": len(b["robust"]["y"]),
                "auc": _auc(b["robust"]["y"], b["robust"]["p"]),
            },
        }
        for m, b in per_metal.items()
    }
    return out


# --------------------------------------------------------------------------
# matched-pair ATE / CATE (WP1)
# --------------------------------------------------------------------------


def matched_ate_cate(
    raw_rows: Dict[str, List[RawRow]],
    metals: Sequence[str],
    seed: int,
    radius: int,
    nbits: int,
) -> Dict[str, object]:
    """Average treatment effect per metal pair on matched pairs, plus a CATE
    heterogeneity check.

    Treatment = metal. For each unordered metal pair (m1, m2) and each
    ``(ligand, cell-line)`` observed under both, the paired effect is
    ``Delta pIC50 = pIC50(m2) - pIC50(m1)`` and the label flip indicator. The ATE
    is the mean over shared pairs.

    CATE heterogeneity: can a ligand-fingerprint model predict the SIGN of the
    per-pair effect better than the constant-ATE baseline (majority sign)? If
    yes, the metal-swap effect is heterogeneous in ligand chemistry -- the hook
    WP2/WP3 hang on. Cross-validated (5-fold, grouped by ligand) AUC.
    """
    pair_eff: Dict[Tuple[str, str], Dict[str, Tuple[int, float]]] = defaultdict(dict)
    for m in metals:
        for smi, cell, y, p in raw_rows[m]:
            pair_eff[(smi, cell)][m] = (y, p)

    ate: Dict[str, object] = {}
    # one row per matched pair, for the CATE heterogeneity check
    cate_smi: List[str] = []
    cate_dp: List[float] = []
    cate_flip: List[int] = []

    for i, m1 in enumerate(metals):
        for m2 in metals[i + 1 :]:
            dps: List[float] = []
            flips: List[int] = []
            for (smi, _cell), per_metal in pair_eff.items():
                if m1 not in per_metal or m2 not in per_metal:
                    continue
                y1, p1 = per_metal[m1]
                y2, p2 = per_metal[m2]
                flip = int(y1 != y2)
                flips.append(flip)
                if not (math.isnan(p1) or math.isnan(p2)):
                    dps.append(p2 - p1)
                    cate_smi.append(smi)
                    cate_dp.append(p2 - p1)
                    cate_flip.append(flip)
            ate[f"{m1}-{m2}"] = {
                "n_shared_pairs": len(flips),
                "label_flip_rate": (float(np.mean(flips)) if flips else None),
                "n_with_pic50": len(dps),
                "mean_delta_pic50": (float(np.mean(dps)) if dps else None),
                "median_delta_pic50": (float(np.median(dps)) if dps else None),
            }

    # CATE: predict the SIGN of the paired effect from ligand chemistry alone.
    # Positives = pairs that flip label; negatives = pairs that do not. Grouped
    # by ligand so the same fingerprint never spans train and test.
    cate_block: Dict[str, object] = {
        "definition": "predict whether a matched pair flips label, from ligand ECFP4 alone",
        "n_pairs": len(cate_flip),
        "n_flip": int(sum(cate_flip)),
    }
    n_neg_pairs = len(cate_flip) - int(sum(cate_flip))
    if len(cate_flip) >= 20 and n_neg_pairs >= 20:
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import roc_auc_score

        Xs = _featurise(cate_smi, radius, nbits)
        ys = np.array(cate_flip, dtype=int)
        groups = np.array(cate_smi, dtype=object)
        n_groups = len(set(groups.tolist()))
        gkf = GroupKFold(n_splits=min(5, n_groups))
        oof = np.full(len(ys), np.nan)
        for tr, te in gkf.split(Xs, ys, groups):
            if len(set(ys[tr].tolist())) < 2:
                continue
            clf = _make_model(ys[tr], seed)
            clf.fit(Xs[tr], ys[tr], verbose=False)
            oof[te] = clf.predict_proba(Xs[te])[:, 1]
        mask = ~np.isnan(oof)
        cate_block.update(
            {
                "n_pos": int((ys == 1).sum()),
                "n_neg": int((ys == 0).sum()),
                "cv_auc": (
                    float(roc_auc_score(ys[mask], oof[mask]))
                    if mask.sum() > 0 and len(set(ys[mask].tolist())) > 1
                    else None
                ),
                "n_scored": int(mask.sum()),
                "n_groups": n_groups,
                "verdict": "HETEROGENEOUS_IN_LIGAND_CHEMISTRY_IF_AUC_GT_0.5",
            }
        )
    else:
        cate_block["verdict"] = "UNDERPOWERED_TOO_FEW_PAIRS"

    return {"ate_by_metal_pair": ate, "cate_ligand_predictability": cate_block}


# --------------------------------------------------------------------------
# power analysis
# --------------------------------------------------------------------------


def auc_power(auc: float, n_pos: int, n_neg: int, alpha: float = 0.05) -> Dict[str, float]:
    """Power to detect ``auc`` vs 0.5 at two-sided ``alpha``.

    Hanley & McNeil (1982) variance for the AUC estimator.
    """
    if n_pos < 1 or n_neg < 1:
        return {
            "auc": auc,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "se": float("nan"),
            "power": float("nan"),
        }
    A = float(auc)
    q1 = A / (2.0 - A)
    q2 = 2.0 * A * A / (1.0 + A)
    var = (
        A * (1.0 - A) + (n_pos - 1) * (q1 - A * A) + (n_neg - 1) * (q2 - A * A)
    ) / (n_pos * n_neg)
    se = math.sqrt(max(var, 1e-12))
    z = (A - 0.5) / se
    z_crit = 1.959963984540054 if alpha == 0.05 else 1.6448536269514722
    power = 0.5 * math.erfc((z_crit - z) / math.sqrt(2.0))
    return {
        "auc": A,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "se": float(se),
        "power": float(power),
    }


def power_analysis(
    sens: Dict[str, str], target_aucs: Sequence[float] = (0.60, 0.65, 0.70)
) -> Dict[str, object]:
    """Power to detect a target AUC for the WP2/WP3 classifiers."""
    n_pos = sum(1 for v in sens.values() if v == "sensitive")
    n_neg = len(sens) - n_pos

    rows = [auc_power(a, n_pos, n_neg) for a in target_aucs]

    need = {}
    for a in target_aucs:
        lo, hi = 0.1, 400.0
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            np_ = max(1, int(round(n_pos * mid)))
            nn_ = max(1, int(round(n_neg * mid)))
            p = auc_power(a, np_, nn_)["power"]
            if p < 0.80:
                lo = mid
            else:
                hi = mid
        scale = 0.5 * (lo + hi)
        need[f"auc_{a}"] = {
            "scale_multiplier_for_80pct_power": scale,
            "n_pos_needed": int(round(n_pos * scale)),
            "n_neg_needed": int(round(n_neg * scale)),
        }

    return {
        "method": "Hanley & McNeil (1982) AUC variance; two-sided alpha=0.05",
        "observed_ligand_class_balance": {"n_sensitive": n_pos, "n_robust": n_neg},
        "power_at_observed_n": rows,
        "n_needed_for_80pct_power": need,
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def run(
    metals: Sequence[str] = METALS,
    seed: int = 42,
    radius: int = 2,
    nbits: int = 2048,
    tie_rule: str = "active",
    time_threshold: Optional[float] = None,
) -> Dict[str, object]:
    from molmetal.scripts.metal_activity_transfer_probe import within_metal

    _load_env_default()
    raw_rows, dedup = load_cohort(metals, tie_rule=tie_rule, time_threshold=time_threshold)
    evidence = matched_pair_evidence(raw_rows, metals)
    sens = sensitivity_map(dedup)

    within_auc: Dict[str, Optional[float]] = {}
    for m in metals:
        s_u, y_u = dedup[m]
        w = within_metal(s_u, y_u, seed, radius, nbits, return_scores=False)
        within_auc[m] = float(w["roc_auc"]) if w.get("status") == "MEASURED" else None

    lomo = lomo_with_scores(dedup, seed, radius, nbits)
    strat = stratified_attribution(lomo, sens, within_auc)
    ate_cate = matched_ate_cate(raw_rows, metals, seed, radius, nbits)
    power = power_analysis(sens)

    n_sens = sum(1 for v in sens.values() if v == "sensitive")
    n_robust_multi = sum(1 for v in sens.values() if v == "robust_multi")
    n_singleton = sum(1 for v in sens.values() if v == "singleton")

    return {
        "cohort": {
            "time_threshold": time_threshold,
            "label": "full" if time_threshold is None else f"time_le_{time_threshold}",
            "n_ligands_total": len(sens),
        },
        "matched_pair_evidence": evidence,
        "ligand_strata": {
            "n_total": len(sens),
            "n_sensitive": n_sens,
            "n_robust_multi": n_robust_multi,
            "n_singleton": n_singleton,
        },
        "within_metal_auc": within_auc,
        "lomo_per_fold": lomo["folds"],
        "stratified_attribution": strat,
        "matched_ate_cate": ate_cate,
        "power_analysis": power,
        "acceptance": {
            "robust_recovery_ge_90pct": bool(
                strat["recovery"]["acceptance_ge_90pct"]  # type: ignore[index]
            ),
            "verdict": (
                "TRANSFER_PENALTY_ATTRIBUTABLE_TO_SENSITIVE_SUBSET"
                if strat["recovery"]["acceptance_ge_90pct"]  # type: ignore[index]
                else "NOT_ATTRIBUTABLE_OR_PARTIAL"
            ),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="WP0+WP1: power, LOMO stratified attribution, matched-pair ATE/CATE."
    )
    p.add_argument("--metals", nargs="+", default=METALS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument("--tie-rule", choices=("active", "inactive"), default="active")
    p.add_argument(
        "--time-threshold",
        type=float,
        default=None,
        help="incubation-time filter in hours; omit for the FULL cohort",
    )
    p.add_argument(
        "--both-cohorts",
        action="store_true",
        help="run the FULL cohort and the Time(h)<=24 cohort and emit both",
    )
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/metal_sensitivity_decomposition.json",
    )
    args = p.parse_args(argv)

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    if args.both_cohorts:
        cohorts = {
            "full": run(
                metals=list(args.metals),
                seed=args.seed,
                radius=args.radius,
                nbits=args.nbits,
                tie_rule=args.tie_rule,
                time_threshold=None,
            ),
            "time_le_24": run(
                metals=list(args.metals),
                seed=args.seed,
                radius=args.radius,
                nbits=args.nbits,
                tie_rule=args.tie_rule,
                time_threshold=24.0,
            ),
        }
        rep: Dict[str, object] = {
            "metric": "metal_sensitivity_decomposition",
            "description": (
                "WP0+WP1 (thesis_v2 prerequisite): matched-pair evidence + power "
                "analysis, LOMO stratified attribution, and matched-pair ATE/CATE. "
                "Both the FULL cohort and the Time(h)<=24 transfer-penalty cohort "
                "are reported so neither is conflated."
            ),
            "unit": "ROC-AUC (0-1, higher better); rates are fractions; pIC50 in log units",
            "scope": "tier2_precious_metal_anticancer",
            "source": "molmetal/scripts/wp0_metal_sensitivity_decomposition.py",
            "model": f"morgan_fp(radius={args.radius}, {args.nbits}bit) + xgboost",
            "data_source": "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026, 69(8), 8838)",
            "seed": args.seed,
            "cohorts": cohorts,
            "honest_framing": {
                "MEASURED_vs_PROJECTED": "MEASURED",
                "caveat": (
                    "The flip-vs-noise gap quantifies that the metal matters; it is "
                    "the v2 foundation, not the headline. The headline is whether the "
                    "intervention effect is predictable (WP1-WP4)."
                ),
            },
        }
        atomic_write_file(Path(args.out_metric), json.dumps(rep, indent=2, default=float))
        for name, c in cohorts.items():
            ev = c["matched_pair_evidence"]
            ls = c["ligand_strata"]
            rec = c["stratified_attribution"]["recovery"]
            print(
                f"[wp0:{name}] ligands={ls['n_total']} sensitive={ls['n_sensitive']} "
                f"pairs={ev['cross_metal_pairs']['n_pairs']} "
                f"flip={ev['cross_metal_pairs']['flip_rate']:.4f} "
                f"noise={ev['repeated_measurement_noise_floor']['contradiction_rate']:.4f}"
            )
            print(
                f"[wp0:{name}] within={rec['within_metal_auc_mean']} "
                f"robust_lomo={rec['robust_lomo_auc']} recovery={rec['robust_recovery_ratio']} "
                f"verdict={c['acceptance']['verdict']}"
            )
        print(f"[wp0] wrote {args.out_metric}")
        return 0

    rep = run(
        metals=list(args.metals),
        seed=args.seed,
        radius=args.radius,
        nbits=args.nbits,
        tie_rule=args.tie_rule,
        time_threshold=args.time_threshold,
    )
    atomic_write_file(Path(args.out_metric), json.dumps(rep, indent=2, default=float))
    print(f"[wp0] cohort={rep['cohort']['label']} verdict={rep['acceptance']['verdict']}")
    print(f"[wp0] wrote {args.out_metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
