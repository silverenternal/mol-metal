#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FU4 — the FSMA K-curve re-run on a FUSED scorer.

=====================================================================
Why this file exists
=====================================================================
``todo.json → directive_2026_10_16`` forbids closing a "something else is
stronger" gap with an honest negative.  ``applies_to`` names three lines; the
third is the **FSMA K-shot main line (C1/C2)**.  On that line there are two
measured scorers with the SAME LOMO / ligand-disjoint protocol and the SAME
label-stratified K-support sampler:

* **A** ``A_ecfp4_xgb`` — Morgan(2, 2048) → XGBoost; adaptation = warm-started
  gradient boosting on the K support ligands
  (``molmetal/scripts/fsma_xgb_warmstart_kshot.py``).  Strong on Rh (+0.132 at
  K=80) and Re (+0.151).
* **B** ``B_dmpnn`` — the in-house metal-blind D-MPNN; adaptation = head-only
  fine-tune on the K support ligands
  (``molmetal/scripts/ex1_metal_injection_contrast.py``, arm ``none``).
  Stronger than A on Ru / Ir / Os.

The two are complementary **per metal**, which is exactly the case the directive
targets: neither dominates, so the final paper model should be their *merge*.
FU1 (feature-level fusion) and FU2 (score-level stacking) do this on the
within-metal Table-1 line.  FU4 is the FSMA-line analogue: run the K-curve with
``merged`` as the adapted scorer and check the acceptance rule

    merged >= max(A, B) on >= 4/5 metals, Holm-significant.

=====================================================================
The merged arm (three variants, all leak-free)
=====================================================================
Every variant fuses the two **adapted** scorers' query probabilities, so the
whole comparison lives on the identical held-out query (paired):

* ``merged_avg``        — mean of the two adapted probabilities.
* ``merged_logit_avg``  — sigmoid of the mean adapted logit (better conditioned
  when the two scorers are differently calibrated).
* ``merged_stack``      — logistic regression on ``[logitA, logitB]`` whose
  training features are **nested-OOF inside the K support** (split the support
  into inner folds; adapt each component on the inner-train, score the
  inner-test).  No support row's meta-feature comes from a model that saw it.
  At K < 20 the inner folds are too small to fit a 2-feature merger, so the
  variant is reported ``SKIPPED`` rather than guessed.

``merged_avg`` / ``merged_logit_avg`` need no meta-training and are the primary
arms; ``merged_stack`` is the learned merger.

=====================================================================
Alignment (the load-bearing detail)
=====================================================================
A and B are built on the **same** cohort rows in the **same** order:
``build_paper_cohort`` + ``collapse_duplicates`` (see
``train_chemprop_lomo_paper_cohort.build_probe_cohort`` and
``ex1_metal_injection_contrast.load_cohort``), and both draw the K support with
the canonical ``_draw_support(yte, None, k, seed, "stratified")``.  The
D-MPNN drops SMILES whose graph fails to build, so the fusion is restricted to
the query rows that are **valid for both** components; the shared mask is
recorded per cell.

=====================================================================
Device / cache honesty
=====================================================================
The published in-house K-curve (``fsma_inhouse_kcurve_ep50.json``) was measured
on ``cuda:1``.  Per the 2026-10-16 user directive all new GPU work is on
``cuda:0``; the ex1 source cache key contains ``str(device)``, so a ``cuda:0``
run is a cache MISS and retrains its own sources.  FU4 therefore owns its
numbers: A, B and merged are all measured here, on one device, and the artifact
records the device + cache dir.  FU4's ``B_dmpnn`` column is NOT expected to
match the ``cuda:1`` K-curve bit-for-bit.

MEASURED only.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")
DEFAULT_SEEDS = (42, 43, 44, 49, 123)
DEFAULT_K = (5, 10, 20, 40, 80)

COMPONENT_A = "A_ecfp4_xgb"
COMPONENT_B = "B_dmpnn"
MERGED_VARIANTS = ("merged_avg", "merged_logit_avg", "merged_stack")

#: FSMA adaptation schedule (20*ceil(K/16) optimizer steps), the historical arm.
ADAPT_STEPS_BY_K = {5: 20, 10: 20, 20: 40, 40: 60, 80: 100}

#: XGB (component A) — mirrors fsma_xgb_warmstart_kshot.XGB_PARAMS.
XGB_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "tree_method": "hist",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _draw_support(te_y, k, seed, sampler):
    from molmetal.scripts.train_chemprop_lomo_paper_cohort import _draw_support as _ds

    return _ds(te_y, None, k, seed, sampler)


def _fit_xgb(X, y, seed, rounds, base=None):
    from xgboost import XGBClassifier

    params = dict(XGB_PARAMS, random_state=int(seed), n_estimators=int(rounds))
    pos = max(1, int(np.asarray(y).sum()))
    neg = max(1, int(len(y) - pos))
    params["scale_pos_weight"] = neg / pos
    clf = XGBClassifier(**params)
    if base is None:
        clf.fit(X, y, verbose=False)
    else:
        clf.fit(X, y, xgb_model=base.get_booster(), verbose=False)
    return clf


def _fit_meta(meta_X, y_tr, kind, seed):
    """Fit the merger on nested-OOF meta-features (same defaults as FU2)."""
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                               random_state=int(seed))
        m.fit(meta_X, y_tr)
        return m
    if kind == "gbdt":
        from xgboost import XGBClassifier

        m = XGBClassifier(
            max_depth=2, n_estimators=100, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=int(seed),
            eval_metric="logloss", n_jobs=1,
        )
        m.fit(meta_X, y_tr)
        return m
    raise ValueError(f"unknown meta_learner {kind!r} (expected logreg|gbdt)")


# ---------------------------------------------------------------------------
# Component B — D-MPNN source model (cached) + head-only adaptation
# ---------------------------------------------------------------------------
def _source_cache_key(metal, seed, args, device):
    from molmetal.data.metal_graph import METAL_FEAT_DIM

    return (
        metal, "none", int(seed), int(args.epochs), float(args.lr), int(args.hidden),
        float(args.dropout), False, False, False, int(METAL_FEAT_DIM), str(device),
    )


def _load_source_sd(metal, seed, args, device):
    """Load (or train + cache) the source state_dict for one (metal, seed)."""
    import torch

    from molmetal.data.metal_graph import METAL_FEAT_DIM
    from molmetal.models.metal_aware_dmpnn import MetalAwareDMPNN
    from molmetal.scripts.ex1_metal_injection_contrast import (
        _SOURCE_CACHE, _source_cache_path, train_source,
    )

    key = _source_cache_key(metal, seed, args, device)
    if key in _SOURCE_CACHE:
        return _SOURCE_CACHE[key]
    cpath = (
        _source_cache_path(args.source_cache_dir, key)
        if args.source_cache_dir is not None else None
    )
    if cpath is not None and Path(cpath).exists():
        sd = torch.load(cpath, map_location="cpu")
        _SOURCE_CACHE[key] = sd
        return sd
    # cache miss -> train on the source pool, then persist
    model = MetalAwareDMPNN(
        metal_injection="none", metal_feat_dim=METAL_FEAT_DIM,
        use_adapter=False, hidden=int(args.hidden), dropout=float(args.dropout),
    ).to(device)
    train_source(model, args._tr_graphs[metal], args._tr_y[metal],
                 int(args.epochs), float(args.lr), device, seed)
    sd = copy.deepcopy(model.state_dict())
    _SOURCE_CACHE[key] = sd
    if cpath is not None:
        Path(cpath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, cpath)
    return sd


def _new_model(args, device):
    from molmetal.data.metal_graph import METAL_FEAT_DIM
    from molmetal.models.metal_aware_dmpnn import MetalAwareDMPNN

    return MetalAwareDMPNN(
        metal_injection="none", metal_feat_dim=METAL_FEAT_DIM,
        use_adapter=False, hidden=int(args.hidden), dropout=float(args.dropout),
    ).to(device)


def _valid_mask(graphs, idx):
    return np.asarray([graphs[int(i)] is not None for i in idx], dtype=bool)


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------
def _run_cell(metal, k, seed, prep, args, device):
    """Row-level scores for one (metal, K, seed): A, B (zero + adapted) + fused."""
    from molmetal.scripts.ex1_metal_injection_contrast import adapt_head, score
    from molmetal.scripts.metal_activity_transfer_probe import _featurise

    tr, te, n_leaked, tr_graphs, te_graphs = prep
    te_y = te["y"]
    adapt_steps = ADAPT_STEPS_BY_K.get(int(k), 20)

    sup_idx = _draw_support(te_y, int(k), int(seed), args.sampler)
    sup_mask = np.zeros(len(te_y), dtype=bool)
    sup_mask[sup_idx] = True
    qry_idx = np.where(~sup_mask)[0]

    # ---- component A: ECFP4 -> XGB ---------------------------------------
    Xtr = _featurise(list(tr["smiles"]), args.radius, args.nbits)
    Xte = _featurise(list(te["smiles"]), args.radius, args.nbits)
    ytr = np.asarray(tr["y"], dtype=int)
    base_a = _fit_xgb(Xtr, ytr, seed, args.base_rounds)
    pA0_all = base_a.predict_proba(Xte)[:, 1]
    warm_a = _fit_xgb(Xte[sup_idx], te_y[sup_idx], seed, args.ft_rounds, base=base_a)
    pA1_all = warm_a.predict_proba(Xte)[:, 1]

    # ---- component B: D-MPNN source + head-only adaptation ---------------
    sd = _load_source_sd(metal, seed, args, device)
    model = _new_model(args, device)
    model.load_state_dict(sd)
    vm = _valid_mask(te_graphs, qry_idx)
    pB0_q = score(model, [te_graphs[int(i)] for i in qry_idx[vm]], device)

    sup_valid_idx = sup_idx[_valid_mask(te_graphs, sup_idx)]
    adapt_head(model, [te_graphs[int(i)] for i in sup_valid_idx], te_y[sup_valid_idx],
               adapt_steps, float(args.adapt_lr), device, seed, scope="head")
    pB1_q = score(model, [te_graphs[int(i)] for i in qry_idx[vm]], device)
    model.load_state_dict(sd)  # restore zero-shot state for the caller

    # ---- align A onto the same (query ∩ valid-B) rows ---------------------
    qry_valid_abs = qry_idx[vm]
    pA0 = np.asarray(pA0_all[qry_valid_abs], dtype=float)
    pA1 = np.asarray(pA1_all[qry_valid_abs], dtype=float)
    pB0_q = np.asarray(pB0_q, dtype=float)
    pB1_q = np.asarray(pB1_q, dtype=float)
    y_q = te_y[qry_valid_abs]

    if not (len(pB0_q) == len(pA0) == len(y_q)):
        raise RuntimeError(f"alignment mismatch: |B|={len(pB0_q)} |A|={len(pA0)} |y|={len(y_q)}")

    stack = _nested_support_stack(
        metal, seed, prep, sup_valid_idx, args, device, adapt_steps, base_a,
        pA1, pB1_q, sd,
    )
    return {
        "metal": metal, "k": int(k), "seed": int(seed),
        "n_support": int(sup_mask.sum()), "n_support_valid": int(len(sup_valid_idx)),
        "n_query": int(len(qry_valid_abs)), "n_invalid_query": int((~vm).sum()),
        "y": y_q,
        "A_zero": pA0, "A_adapt": pA1, "B_zero": pB0_q, "B_adapt": pB1_q,
        "merged_avg": 0.5 * (pA1 + pB1_q),
        "merged_logit_avg": 1.0 / (1.0 + np.exp(-0.5 * (_logit(pA1) + _logit(pB1_q)))),
        "merged_stack": stack,
    }


def _nested_support_stack(metal, seed, prep, sup_valid_idx, args, device,
                          adapt_steps, base_a, pA1_q, pB1_q, sd):
    """logreg on [logitA, logitB], trained on nested-OOF inside the K support.

    Returns the query probability vector, or ``None`` when the support is too
    small (< 20 valid rows) to fit a 2-feature merger without leaking.
    """
    from molmetal.scripts.ex1_metal_injection_contrast import adapt_head, score
    from molmetal.scripts.metal_activity_transfer_probe import _featurise
    from sklearn.model_selection import StratifiedKFold

    if len(sup_valid_idx) < 20:
        return None

    tr, te, _n_leaked, _trg, te_graphs = prep
    te_y = te["y"]
    Xte = _featurise(list(te["smiles"]), args.radius, args.nbits)
    sup_y = te_y[sup_valid_idx]

    model = _new_model(args, device)
    inner = StratifiedKFold(n_splits=args.inner_splits, shuffle=True,
                            random_state=int(seed) + 1000)
    n = len(sup_valid_idx)
    mA = np.full(n, np.nan, dtype=np.float64)
    mB = np.full(n, np.nan, dtype=np.float64)
    for itr, ite in inner.split(np.zeros((n, 1)), sup_y):
        itr_abs = sup_valid_idx[itr]
        ite_abs = sup_valid_idx[ite]
        if len(set(te_y[itr_abs].tolist())) < 2 or len(ite_abs) == 0:
            continue
        # component A: warm-start the SAME source base on the inner-train
        warm = _fit_xgb(Xte[itr_abs], te_y[itr_abs], seed, args.ft_rounds, base=base_a)
        mA[ite] = warm.predict_proba(Xte[ite_abs])[:, 1]
        # component B: adapt a fresh copy of the source model on the inner-train
        model.load_state_dict(sd)
        adapt_head(model, [te_graphs[int(i)] for i in itr_abs], te_y[itr_abs],
                   adapt_steps, float(args.adapt_lr), device, seed, scope="head")
        p = score(model, [te_graphs[int(i)] for i in ite_abs], device)
        if len(p) == len(ite_abs):
            mB[ite] = p
    model.load_state_dict(sd)

    mask = np.isfinite(mA) & np.isfinite(mB)
    if int(mask.sum()) < 2 or len(set(sup_y[mask].tolist())) < 2:
        return None
    meta_X = np.column_stack([_logit(mA[mask]), _logit(mB[mask])])
    meta = _fit_meta(meta_X, sup_y[mask], args.meta_learner, seed)
    Xq = np.column_stack([_logit(pA1_q), _logit(pB1_q)])
    return meta.predict_proba(Xq)[:, 1]


# ---------------------------------------------------------------------------
# Per-metal driver
# ---------------------------------------------------------------------------
def _run_metal(metal, args, device) -> Dict[str, object]:
    from molmetal.scripts.ex1_metal_injection_contrast import _prepare

    t0 = time.time()
    prep = _prepare(args._cohort, metal, False, False)
    tr, te, n_leaked, tr_graphs, _teg = prep
    if len(tr["smiles"]) == 0:
        raise ValueError(
            f"empty source pool for held-out {metal!r}: --metals must contain all "
            f"source metals (LOMO needs >= 2 metals), got {list(args.metals)}"
        )
    args._tr_graphs[metal] = tr_graphs
    args._tr_y[metal] = tr["y"]
    entry: Dict[str, object] = {
        "n_held_out": int(len(te["y"])), "n_leaked": int(n_leaked),
        "pos_rate": float(np.mean(te["y"])),
    }

    arms = (COMPONENT_A, COMPONENT_B) + MERGED_VARIANTS
    by_k: Dict[str, object] = {}
    for k in args.k:
        if len(te["y"]) <= int(k) + 10:
            by_k[str(k)] = {"status": "SKIPPED", "reason": f"n_test={len(te['y'])} <= K+10"}
            continue
        cells = [_run_cell(metal, k, seed, prep, args, device) for seed in args.seeds]

        arm_auc: Dict[str, List[float]] = {a: [] for a in arms}
        for c in cells:
            y = c["y"]
            arm_auc[COMPONENT_A].append(_auc(y, c["A_adapt"]))
            arm_auc[COMPONENT_B].append(_auc(y, c["B_adapt"]))
            arm_auc["merged_avg"].append(_auc(y, c["merged_avg"]))
            arm_auc["merged_logit_avg"].append(_auc(y, c["merged_logit_avg"]))
            arm_auc["merged_stack"].append(
                float("nan") if c["merged_stack"] is None else _auc(y, c["merged_stack"])
            )
        max_ab = [
            float(np.nanmax([aA, aB])) if np.isfinite([aA, aB]).any() else float("nan")
            for aA, aB in zip(arm_auc[COMPONENT_A], arm_auc[COMPONENT_B])
        ]
        zero_a = [_auc(c["y"], c["A_zero"]) for c in cells]
        zero_b = [_auc(c["y"], c["B_zero"]) for c in cells]

        from molmetal.scripts.synthesize_tmqm_pretrain_init_ab_multimetal import _paired_stats

        def _delta(ref, arm):
            d = np.asarray(arm, dtype=float) - np.asarray(ref, dtype=float)
            good = np.isfinite(d)
            return _paired_stats(d[good]) if int(good.sum()) >= 2 else {"status": "NOT_MEASURED"}

        block: Dict[str, object] = {
            "status": "MEASURED",
            "n_seeds": len(cells),
            "n_support": int(cells[0]["n_support"]),
            "n_support_valid": int(cells[0]["n_support_valid"]),
            "n_query": int(cells[0]["n_query"]),
            "per_arm_mean_auc": {
                a: (float(np.nanmean(v)) if np.isfinite(np.asarray(v, dtype=float)).any()
                    else float("nan"))
                for a, v in arm_auc.items()
            },
            "per_seed": {a: [float(x) for x in v] for a, v in arm_auc.items()},
            "zero_shot_mean_auc": {
                COMPONENT_A: float(np.nanmean(zero_a)),
                COMPONENT_B: float(np.nanmean(zero_b)),
            },
            "max_ab_per_seed": max_ab,
            "deltas": {
                "merged_avg_minus_max_ab": _delta(max_ab, arm_auc["merged_avg"]),
                "merged_logit_avg_minus_max_ab": _delta(max_ab, arm_auc["merged_logit_avg"]),
                "merged_stack_minus_max_ab": _delta(max_ab, arm_auc["merged_stack"]),
                "merged_avg_minus_A": _delta(arm_auc[COMPONENT_A], arm_auc["merged_avg"]),
                "merged_avg_minus_B": _delta(arm_auc[COMPONENT_B], arm_auc["merged_avg"]),
            },
        }
        by_k[str(k)] = block
        pa = block["per_arm_mean_auc"]
        print(
            f"[fu4] {metal:>3} K={int(k):>3} A={pa[COMPONENT_A]:.4f} B={pa[COMPONENT_B]:.4f} "
            f"avg={pa['merged_avg']:.4f} logit={pa['merged_logit_avg']:.4f} "
            f"stack={pa['merged_stack']:.4f}", flush=True,
        )
    entry["by_k"] = by_k
    entry["wall_sec"] = round(time.time() - t0, 1)
    return entry


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
def _acceptance(per_metal: Dict[str, object], variant: str, k: int) -> Dict[str, object]:
    """Directive acceptance: merged >= max(A,B) on >=4/5 metals, Holm-significant."""
    from molmetal.scripts.synthesize_tmqm_pretrain_init_ab_multimetal import _holm

    pvals: Dict[str, float] = {}
    ge = 0
    n_res = 0
    for m, v in per_metal.items():
        b = v.get("by_k", {}).get(str(k), {})
        if b.get("status") != "MEASURED":
            continue
        d = b["deltas"][f"{variant}_minus_max_ab"]
        if d.get("status") == "NOT_MEASURED":
            continue  # e.g. merged_stack below K=20, or a 1-seed smoke
        n_res += 1
        if float(d["mean"]) >= 0.0:
            ge += 1
        pvals[m] = float(d["p_one_sided_ttest"])
    holm = _holm(pvals) if pvals else None
    sig = len(holm["significant_after_holm_0.05"]) if holm else 0
    return {
        "variant": variant, "k": int(k),
        "merged_ge_max_count": ge, "n_resolvable": n_res,
        "holm_significant_count": sig,
        "holm": holm,
        "acceptance_met": bool(ge >= 4 and sig >= 1),
        "rule": "directive_2026_10_16: merged >= max(A,B) on >=4/5 metals, Holm p < 0.05",
    }


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, object]:
    from molmetal.scripts.ex1_metal_injection_contrast import load_cohort

    device = args.device
    if str(device).startswith("cuda"):
        import torch

        if args.deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    cohort = load_cohort(tuple(args.metals), args.cohort_spec)
    args._cohort = cohort
    args._tr_graphs = {}
    args._tr_y = {}

    per_metal: Dict[str, object] = {}
    for m in args.metals:
        per_metal[m] = _run_metal(m, args, device)

    rep: Dict[str, object] = {
        "metric": "fsma_fused_kcurve",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/fsma_fused_kcurve.py",
        "description": (
            "FU4: the FSMA LOMO K-shot curve re-run with the FUSED scorer "
            "(D-MPNN-FSMA head-only adaptation ⊕ ECFP4->XGB warm-start) as the "
            "adapted model.  directive_2026_10_16 applies_to 'FSMA K-shot 主线 "
            "(C1/C2)': the final model must be the merge, not the weaker member."
        ),
        "protocol": {
            "cohort_spec": args.cohort_spec,
            "split": "leave-one-metal-out, ligand-disjoint (leaked ligands dropped)",
            "support_sampler": args.sampler,
            "adapt_steps_by_k": ADAPT_STEPS_BY_K,
            "seeds": [int(s) for s in args.seeds],
            "k": [int(k) for k in args.k],
            "components": {
                COMPONENT_A: f"morgan_fp(radius={args.radius}, {args.nbits}bit) + xgboost "
                             f"(base {args.base_rounds} + ft {args.ft_rounds} rounds)",
                COMPONENT_B: f"MetalAwareDMPNN(metal_injection=none, hidden={args.hidden}) "
                             f"head-only FSMA (epochs={args.epochs})",
            },
            "merged_variants": list(MERGED_VARIANTS),
        },
        "device": str(device),
        "deterministic": bool(args.deterministic),
        "source_cache_dir": (None if args.source_cache_dir is None else str(args.source_cache_dir)),
        "per_metal": per_metal,
        "honest_framing": {
            "purpose": (
                "directive_2026_10_16 acceptance on the FSMA line: does the FUSED "
                "adapted scorer reach >= max(A,B) on >=4/5 metals with Holm "
                "significance?"
            ),
            "component_numbers_stay_reported": (
                "A and B are re-measured here and reported alongside merged; the "
                "directive changes the final model's definition, not the facts."
            ),
            "no_pooled_mean": (
                "Ru is ~64% of the deduplicated cohort; the per-metal table is the unit."
            ),
            "device_note": (
                "FU4 owns its device: the published cuda:1 K-curve is NOT the "
                "baseline here; A, B and merged are all measured on one device. "
                "The source-cache key contains str(device), so a cuda:0 run "
                "retrains its own sources."
            ),
            "stack_note": (
                "merged_stack is SKIPPED at K<20 (inner folds too small for a "
                "2-feature merger); it is not imputed."
            ),
        },
    }
    rep["acceptance"] = {
        v: {str(k): _acceptance(per_metal, v, int(k)) for k in args.k}
        for v in MERGED_VARIANTS
    }
    return rep


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FU4 — FSMA K-curve on a fused (D-MPNN ⊕ ECFP4-XGB) scorer."
    )
    p.add_argument("--metals", nargs="+", default=list(METALS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K))
    p.add_argument("--cohort-spec", choices=["24h", "paper_funnel"], default="paper_funnel")
    p.add_argument("--sampler", choices=["stratified", "random"], default="stratified")
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument("--base-rounds", type=int, default=500)
    p.add_argument("--ft-rounds", type=int, default=50)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--adapt-lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--inner-splits", type=int, default=3)
    p.add_argument("--meta-learner", choices=["logreg", "gbdt"], default="logreg")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--source-cache-dir", type=str, default=None)
    p.add_argument("--config", type=str, default=None,
                   help="YAML config (AGENTS.md §5.11); an explicit CLI flag still wins.")
    p.add_argument("--out-metric", default="metrics/by_metric/fsma_fused_kcurve.json")
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = _build_parser()
    args = p.parse_args(argv)
    if getattr(args, "config", None):
        _scripts_dir = str(Path(__file__).resolve().parent)
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        from wfA2_config_loader import apply_yaml_overrides

        args = apply_yaml_overrides(p, args, args.config, argv=list(argv or []))
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    rep = run(args)

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = Path(args.out_metric)
    atomic_write_file(out, json.dumps(rep, indent=2, default=float, ensure_ascii=False))

    print(f"\n{'metal':>5} {'K':>4} {'A':>7} {'B':>7} {'avg':>7} {'logit':>7} {'stack':>7}")
    for m in args.metals:
        for k in args.k:
            b = rep["per_metal"][m]["by_k"].get(str(k), {})
            if b.get("status") != "MEASURED":
                continue
            pa = b["per_arm_mean_auc"]
            print(f"{m:>5} {int(k):>4} {pa[COMPONENT_A]:7.4f} {pa[COMPONENT_B]:7.4f} "
                  f"{pa['merged_avg']:7.4f} {pa['merged_logit_avg']:7.4f} {pa['merged_stack']:7.4f}")
    for variant in MERGED_VARIANTS:
        for k in args.k:
            a = rep["acceptance"][variant][str(k)]
            print(f"acceptance[{variant} K={int(k)}]: {a['acceptance_met']} "
                  f"(>={a['merged_ge_max_count']}/{a['n_resolvable']}, holm-sig {a['holm_significant_count']})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
