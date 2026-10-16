#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FU2 — stack the ECFP4→XGB and D-MPNN scorers into ONE merged scorer.

=====================================================================
Why this file exists
=====================================================================
G1 (``todo.json`` → ``gap_analysis_2026_10_16``): on the paper cohort and the
*shared* folds, the in-house D-MPNN loses to ECFP4→XGBoost on **all five**
metals (Ru −0.0440, Ir −0.1628, Rh −0.0308, Os −0.1380, Re −0.0395;
``metrics/by_metric/dmpnn_within_metal_auc_paper_funnel.json``).  FU1 tests
whether *feature-level* fusion (concatenating the D-MPNN representation onto
ECFP4) closes that gap.  FU2 is the other half of ``directive_2026_10_16``:
if feature fusion is dragged down by the weaker member (as arm 2 was, Ru
−0.0071 in ``fusion_cheap_arms_probe.json``), then a **stacking / routing**
merger — which can down-weight a weak base learner instead of adding its
columns to a tree model — is the more robust way to reach

    merged >= max(component_A, component_B)

without discarding the stronger component.

=====================================================================
Components and the merged scorer
=====================================================================
* ``A_ecfp4_xgb``  — ECFP4 (Morgan r=2, 2048) → XGBoost.  Reproduces the
  BLK-P1 / FU1 arm-1 anchor column exactly (same folds, same ``_make_model``).
* ``B_dmpnn``      — the in-house masked D-MPNN scorer (``MaskedMetalCytotoxModel``)
  trained per fold with the exact BLK-P1 protocol (inverse-class-frequency CE,
  Adam lr=1e-3, grad-clip 5.0).  This is the paper's model as a *base learner*.
* ``merged_stack`` — a meta-learner ``g([A, B])``.  Meta-features are the two
  base learners' probabilities (or their logits, ``--meta-space``); the
  meta-learner is ``logreg`` (default) or a shallow ``gbdt``.

=====================================================================
Leakage control (the load-bearing part)
=====================================================================
A naive stack trains the meta-learner on OOF base predictions produced on the
*same* partition that the meta-learner is evaluated on.  That leaks: the base
model that scored a meta-train row saw the meta-test rows' labels.  FU2 uses
the **strict nested** scheme that ``fuse_dmpnn_gbdt_within_metal.py``
introduced for embeddings, applied here to *predictions*:

For every outer fold ``(tr, te)``:
  1. inner K-fold over ``tr`` — train A and B on each inner-train, predict the
     inner-test → inner-OOF meta-features for the ``tr`` rows (no ``tr`` row's
     meta-feature comes from a base model that saw it);
  2. train A and B on the *full* ``tr`` → meta-features for the ``te`` rows;
  3. fit the meta-learner on (inner-OOF features, ``tr`` labels), predict ``te``.

The A / B columns are the same models as step 2 scored on ``te`` — i.e. the
ordinary (non-nested) OOF AUC, directly comparable to FU1's arm-1 anchor and
BLK-P1's D-MPNN column.  Only ``merged_stack`` pays the nested cost.

=====================================================================
Statistics
=====================================================================
Per split seed: 5-fold mean AUC; folds are shared → deltas are paired.
``delta(m, s) = AUC_merged − AUC_component`` over the 5 split seeds.  The
family correction is a **paired one-sided t-test** (df=4) with a t-based 95%
CI; the percentile bootstrap is ``*_reference_only`` (anti-conservative at
n=5 — see ``project_five_seed_bootstrap_anticonservative``).

``merged`` "wins" a metal when ``mean delta > 0`` AND ``ci95_ttest[0] > 0``
AND the Holm-adjusted p < 0.05.  The directive's acceptance rule is reported
verbatim: ``merged >= max(A, B)`` on **≥4/5 metals** with Holm significance.

Reporting rule: per-metal table only, **no pooled mean** — Ru is ~64% of the
deduplicated cohort, so a pooled mean is a Ru mean wearing a 5-metal label.

Usage
-----
::

    # canonical: every hyperparameter lives in the YAML (AGENTS.md §5.11)
    .venv/bin/python molmetal/scripts/stack_dmpnn_xgb_within_metal.py \
        --config configs/stack_dmpnn_xgb_paper_funnel_gpu.yaml

    # CPU smoke (2 epochs, 1 metal, 1 split seed, 200 ligands)
    .venv/bin/python molmetal/scripts/stack_dmpnn_xgb_within_metal.py \
        --config configs/stack_dmpnn_xgb_paper_funnel_gpu.yaml \
        --metals Ru --split-seeds 42 --epochs 2 --max-fold-n 200
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CPU thread control (perf, not correctness) — identical rationale to BLK-P1 /
# BLK-F1: the D-MPNN forward loops over molecules in Python and torch's default
# intra-op pool spawns ~90 threads that thrash on sub-ms work. Must precede
# torch import.
# ---------------------------------------------------------------------------
import os as _os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from molmetal.scripts.metal_activity_transfer_probe import (  # noqa: E402
    METALS,
    MIN_UNIQUE_FOR_GATE,
    _featurise,
    _make_model,
)
from molmetal.scripts.dmpnn_within_metal_auc import (  # noqa: E402
    MaskedMetalCytotoxModel,
    _load_dedup_cohort,
    _resolve_device,
    _same_folds_as_probe,
)
from molmetal.scripts.finetune_metalcytotox_from_tmqm import (  # noqa: E402
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    _train_one_epoch,
    make_batch,
)
from molmetal.scripts.synthesize_tmqm_pretrain_init_ab_multimetal import (  # noqa: E402
    _holm,
    _paired_stats,
)

# Belt-and-braces with the OMP_* env vars above (torch may already be imported
# transitively by the modules above, making the env vars too late).
import torch as _torch  # noqa: E402

_torch.set_num_threads(1)
try:
    _torch.set_num_interop_threads(1)
except RuntimeError:
    pass

#: The three reported columns.  A and B are the base learners; merged_stack is
#: the meta-learner's output.  Order = report order.
COMPONENT_A = "A_ecfp4_xgb"
COMPONENT_B = "B_dmpnn"
MERGED = "merged_stack"
ARMS: Tuple[str, ...] = (COMPONENT_A, COMPONENT_B, MERGED)

CONFIRMATORY_METALS: Tuple[str, ...] = ("Ru", "Ir")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _valid_indices(idx, graphs) -> np.ndarray:
    return np.asarray([int(i) for i in idx if graphs[int(i)] is not None], dtype=int)


def _class_weighted_criterion(targets_tr, device):
    import torch
    import torch.nn as nn

    pos = max(1, int(np.asarray(targets_tr).sum()))
    neg = max(1, int(len(targets_tr) - pos))
    w = torch.tensor([1.0, neg / pos], dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=w)


def _auc(y_true, prob) -> float:
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y_true, prob))
    except Exception:
        return float("nan")


def _meta_features(p: np.ndarray, space: str, eps: float = 1e-6) -> np.ndarray:
    """Map a probability vector to the meta-learner's input space."""
    p = np.asarray(p, dtype=np.float64)
    if space == "prob":
        return p
    # logit — better conditioned for a linear meta-learner; clipped so a
    # saturated base prob does not become ±inf.
    pc = np.clip(p, eps, 1.0 - eps)
    return np.log(pc / (1.0 - pc))


# ---------------------------------------------------------------------------
# Base learners
# ---------------------------------------------------------------------------
def _fit_xgb(X: np.ndarray, y: np.ndarray, tr: np.ndarray, split_seed: int):
    """Component A: ECFP4 → XGB (the exact BLK-P1 / FU1 arm-1 model)."""
    clf = _make_model(y[tr], split_seed)
    clf.fit(X[tr], y[tr], verbose=False)
    return clf


def _fit_dmpnn(
    graphs, y, tr, train_seed, hidden, depth, dropout, lr, epochs,
    batch_size, max_atoms, max_edges, device,
):
    """Component B: the in-house masked D-MPNN scorer, BLK-P1 protocol."""
    import torch

    rng = np.random.default_rng(train_seed)
    torch.manual_seed(train_seed)
    model = MaskedMetalCytotoxModel(hidden=hidden, depth=depth, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = _class_weighted_criterion(y[tr], device)
    for _ in range(epochs):
        _train_one_epoch(
            model, optimizer, criterion, graphs, y, tr,
            batch_size=batch_size, max_atoms=max_atoms, max_edges=max_edges,
            device=device, rng=rng,
        )
    return model


@_torch.no_grad()
def _dmpnn_probs(
    model, graphs, idx: np.ndarray, batch_size: int, max_atoms: int,
    max_edges: int, device,
) -> np.ndarray:
    """Row-aligned positive-class probability for ``idx`` (all-valid indices)."""
    model.eval()
    out = np.full(len(idx), np.nan, dtype=np.float64)
    if len(idx) == 0:
        return out
    for start in range(0, len(idx), batch_size):
        bi = idx[start : start + batch_size]
        batched = make_batch(
            [graphs[int(i)] for i in bi], [0] * len(bi),
            max_atoms, max_edges, ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
        )
        if batched is None:
            continue
        logits = model(
            batched.atom_feats.to(device), batched.bond_feats.to(device),
            batched.edge_src.to(device), batched.edge_dst.to(device),
        )
        p = _torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        out[start : start + len(bi)] = p
    return out


# ---------------------------------------------------------------------------
# Meta-learner
# ---------------------------------------------------------------------------
def _fit_meta(meta_X: np.ndarray, y_tr: np.ndarray, kind: str, seed: int):
    """Fit the merger on nested-OOF meta-features."""
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression

        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed)
        m.fit(meta_X, y_tr)
        return m
    if kind == "gbdt":
        from xgboost import XGBClassifier

        m = XGBClassifier(
            max_depth=2, n_estimators=100, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=seed,
            eval_metric="logloss", n_jobs=1,
        )
        m.fit(meta_X, y_tr)
        return m
    raise ValueError(f"unknown meta_learner {kind!r} (expected logreg|gbdt)")


# ---------------------------------------------------------------------------
# Per-metal driver
# ---------------------------------------------------------------------------
def _run_metal(metal: str, args: argparse.Namespace, device) -> Dict[str, object]:
    from sklearn.model_selection import StratifiedKFold

    from molmetal.baselines.dmpnn import featurize_smiles_list

    t0 = time.time()
    smiles, y, n_raw = _load_dedup_cohort(metal, getattr(args, "cohort_spec", "24h"))
    if args.max_fold_n and len(y) > args.max_fold_n:
        rng = np.random.default_rng(args.split_seeds[0])
        keep = np.sort(rng.choice(len(y), size=args.max_fold_n, replace=False))
        smiles = [smiles[i] for i in keep]
        y = y[keep]
    n_unique = int(len(y))
    entry: Dict[str, object] = {
        "n_rows_raw": n_raw,
        "n_unique": n_unique,
        "pos_rate": float(y.mean()) if n_unique else float("nan"),
        "resolvable": bool(n_unique >= MIN_UNIQUE_FOR_GATE),
        "meta_learner": args.meta_learner,
        "meta_space": args.meta_space,
        "inner_splits": args.inner_splits,
    }
    if n_unique < 2 * args.n_splits or len(set(y.tolist())) < 2:
        entry["status"] = "SKIPPED"
        entry["reason"] = f"n={n_unique} too small or single-class"
        return entry

    graphs = featurize_smiles_list(list(smiles))
    entry["n_invalid_graphs"] = int(sum(g is None for g in graphs))
    X_ecfp = _featurise(list(smiles), args.radius, args.nbits)

    per_split: List[Dict[str, object]] = []
    per_arm_seed_auc: Dict[str, List[float]] = {a: [] for a in ARMS}
    per_arm_seed_max: List[float] = []  # max(A, B) per split seed

    for split_seed in args.split_seeds:
        folds = _same_folds_as_probe(y, split_seed, args.n_splits)
        aA: List[float] = []
        aB: List[float] = []
        aM: List[float] = []
        for tr, te in folds:
            trv = _valid_indices(tr, graphs)
            tev = _valid_indices(te, graphs)
            if len(trv) == 0 or len(tev) == 0 or len(set(y[tev].tolist())) < 2:
                aA.append(float("nan"))
                aB.append(float("nan"))
                aM.append(float("nan"))
                continue

            # ---- 1. inner-OOF meta-features for the TRAIN rows -------------
            pA_in = np.full(len(trv), np.nan, dtype=np.float64)
            pB_in = np.full(len(trv), np.nan, dtype=np.float64)
            inner = StratifiedKFold(
                n_splits=args.inner_splits, shuffle=True,
                random_state=split_seed + 1000,
            )
            for itr, ite in inner.split(np.zeros((len(trv), 1)), y[trv]):
                itr_abs = trv[itr]
                ite_abs = trv[ite]
                if len(set(y[itr_abs].tolist())) < 2 or len(ite_abs) == 0:
                    continue
                clfA = _fit_xgb(X_ecfp, y, itr_abs, split_seed)
                pA_in[ite] = clfA.predict_proba(X_ecfp[ite_abs])[:, 1]
                mB = _fit_dmpnn(
                    graphs, y, itr_abs, args.train_seeds[0], args.hidden, args.depth,
                    args.dropout, args.lr, args.epochs, args.batch_size,
                    args.max_atoms, args.max_edges, device,
                )
                pB_in[ite] = _dmpnn_probs(
                    mB, graphs, ite_abs, args.batch_size, args.max_atoms, args.max_edges, device
                )

            # ---- 2. base learners on the FULL outer-train → outer-test ------
            clfA_full = _fit_xgb(X_ecfp, y, trv, split_seed)
            pA_te = clfA_full.predict_proba(X_ecfp[tev])[:, 1]
            mB_full = _fit_dmpnn(
                graphs, y, trv, args.train_seeds[0], args.hidden, args.depth,
                args.dropout, args.lr, args.epochs, args.batch_size,
                args.max_atoms, args.max_edges, device,
            )
            pB_te = _dmpnn_probs(
                mB_full, graphs, tev, args.batch_size, args.max_atoms, args.max_edges, device
            )

            # ---- 3. meta-learner on inner-OOF, predict outer-test -----------
            MA_tr = _meta_features(pA_in, args.meta_space)
            MB_tr = _meta_features(pB_in, args.meta_space)
            MA_te = _meta_features(pA_te, args.meta_space)
            MB_te = _meta_features(pB_te, args.meta_space)
            mask = np.isfinite(MA_tr) & np.isfinite(MB_tr)
            if int(mask.sum()) < 2 or len(set(y[trv][mask].tolist())) < 2:
                aM.append(float("nan"))
            else:
                meta_X_tr = np.column_stack([MA_tr[mask], MB_tr[mask]])
                meta_X_te = np.column_stack([MA_te, MB_te])
                meta = _fit_meta(meta_X_tr, y[trv][mask], args.meta_learner, args.train_seeds[0])
                pM_te = meta.predict_proba(meta_X_te)[:, 1]
                aM.append(_auc(y[tev], pM_te))

            aA.append(_auc(y[tev], pA_te))
            aB.append(_auc(y[tev], pB_te))

        def _nanmean(v: List[float]) -> float:
            arr = np.asarray(v, dtype=float)
            return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")

        mA, mB_, mM = _nanmean(aA), _nanmean(aB), _nanmean(aM)
        per_arm_seed_auc[COMPONENT_A].append(mA)
        per_arm_seed_auc[COMPONENT_B].append(mB_)
        per_arm_seed_auc[MERGED].append(mM)
        best = float(np.nanmax([mA, mB_])) if np.isfinite([mA, mB_]).any() else float("nan")
        per_arm_seed_max.append(best)
        per_split.append(
            {
                "split_seed": split_seed,
                "A": mA, "B": mB_, "merged": mM, "max_A_B": best,
                "per_fold_A": aA, "per_fold_B": aB, "per_fold_merged": aM,
            }
        )
        print(
            f"[fu2]   {metal} seed={split_seed} "
            f"A={mA:.4f} B={mB_:.4f} merged={mM:.4f} max={best:.4f}",
            flush=True,
        )

    # ---- paired deltas ------------------------------------------------------
    arrA = np.asarray(per_arm_seed_auc[COMPONENT_A], dtype=float)
    arrB = np.asarray(per_arm_seed_auc[COMPONENT_B], dtype=float)
    arrM = np.asarray(per_arm_seed_auc[MERGED], dtype=float)
    arrMax = np.asarray(per_arm_seed_max, dtype=float)

    def _delta_stats(ref: np.ndarray) -> Dict[str, object]:
        d = arrM - ref
        good = np.isfinite(d)
        if good.sum() < 2:
            return {"status": "NOT_MEASURED"}
        return _paired_stats(d[good])

    deltas = {
        "merged_minus_A": _delta_stats(arrA),
        "merged_minus_B": _delta_stats(arrB),
        "merged_minus_max_AB": _delta_stats(arrMax),
    }

    entry.update(
        {
            "status": "MEASURED",
            "n_splits": args.n_splits,
            "split_seeds": list(args.split_seeds),
            "per_arm_seed_auc": {
                a: [round(float(x), 6) for x in v] for a, v in per_arm_seed_auc.items()
            },
            "per_arm_mean_auc": {
                a: round(float(np.nanmean(v)), 6) for a, v in per_arm_seed_auc.items()
                if np.isfinite(np.asarray(v, dtype=float)).any()
            },
            "deltas": deltas,
            "per_split": per_split,
            "wall_sec": round(time.time() - t0, 1),
        }
    )
    ma = entry["per_arm_mean_auc"]
    entry["three_column"] = {
        "A_ecfp4_xgb": ma.get(COMPONENT_A),
        "B_dmpnn": ma.get(COMPONENT_B),
        "merged": ma.get(MERGED),
        "max_A_B": round(float(np.nanmax([ma.get(COMPONENT_A, np.nan), ma.get(COMPONENT_B, np.nan)])), 6)
        if ma else None,
    }
    entry["three_column"]["merged_minus_max"] = (
        None if entry["three_column"]["max_A_B"] is None or ma.get(MERGED) is None
        else round(ma[MERGED] - entry["three_column"]["max_A_B"], 6)
    )
    entry["three_column"]["merged_ge_max"] = (
        None if entry["three_column"]["merged_minus_max"] is None
        else bool(entry["three_column"]["merged_minus_max"] >= 0.0)
    )
    print(
        f"[fu2] {metal:>3} A={ma.get(COMPONENT_A):.4f} B={ma.get(COMPONENT_B):.4f} "
        f"merged={ma.get(MERGED):.4f} "
        f"d(max)={deltas['merged_minus_max_AB'].get('mean')} wall={entry['wall_sec']}s",
        flush=True,
    )
    return entry


# ---------------------------------------------------------------------------
# Verdicts + family block
# ---------------------------------------------------------------------------
def _verdict(metal_entry: Dict[str, object], delta: Dict[str, object], holm_p: Optional[float]) -> str:
    if not metal_entry.get("resolvable", False):
        return "UNRESOLVABLE"
    if not isinstance(delta, dict) or "mean" not in delta:
        return "NOT_MEASURED"
    ci_lo = delta["ci95_ttest"][0]
    if delta["mean"] > 0 and ci_lo > 0 and (holm_p is not None and holm_p < 0.05):
        return "MERGED_WINS"
    if ci_lo > 0:
        return "POSITIVE_NOT_HOLM_SIGNIFICANT"
    if delta["ci95_ttest"][0] <= 0.0 <= delta["ci95_ttest"][1]:
        return "NULL"
    return "NEGATIVE"


def _family_block(per_metal: Dict[str, object]) -> Dict[str, object]:
    comparisons = ("merged_minus_A", "merged_minus_B", "merged_minus_max_AB")
    out: Dict[str, object] = {"comparisons": {}, "confirmatory_metals": list(CONFIRMATORY_METALS)}
    for comp in comparisons:
        p_conf = {}
        p_expl = {}
        for m, me in per_metal.items():
            if not isinstance(me, dict):
                continue
            d = me.get("deltas", {}).get(comp, {}) if isinstance(me.get("deltas"), dict) else {}
            if isinstance(d, dict) and "p_one_sided_ttest" in d:
                p_expl[m] = float(d["p_one_sided_ttest"])
                if m in CONFIRMATORY_METALS:
                    p_conf[m] = float(d["p_one_sided_ttest"])
        holm_conf = _holm(p_conf) if p_conf else None
        holm_expl = _holm(p_expl) if p_expl else None
        for m, me in per_metal.items():
            if not isinstance(me, dict):
                continue
            d = me.get("deltas", {}).get(comp, {}) if isinstance(me.get("deltas"), dict) else {}
            hp = holm_conf["p_one_sided_holm"].get(m) if holm_conf else None
            me.setdefault("verdicts", {})[comp] = _verdict(me, d, hp)
        out["comparisons"][comp] = {
            "holm_confirmatory_ru_ir": holm_conf,
            "holm_exploratory_5metal": holm_expl,
        }
    return out


def _acceptance_block(per_metal: Dict[str, object]) -> Dict[str, object]:
    """The directive's acceptance rule, evaluated verbatim.

    ``merged >= max(A, B)`` on ≥4/5 metals, with Holm significance on the
    merged-vs-max paired delta.
    """
    resolvable = [m for m, me in per_metal.items() if isinstance(me, dict) and me.get("resolvable")]
    ge_max: List[str] = []
    sig: List[str] = []
    for m in resolvable:
        me = per_metal[m]
        tc = me.get("three_column", {})
        if tc.get("merged_ge_max"):
            ge_max.append(m)
        v = me.get("verdicts", {}).get("merged_minus_max_AB")
        if v == "MERGED_WINS":
            sig.append(m)
    n_res = len(resolvable)
    met = (len(ge_max) >= 4 and len(sig) >= 4) if n_res >= 5 else (
        len(ge_max) >= int(np.ceil(0.8 * n_res)) and len(sig) >= int(np.ceil(0.8 * n_res))
    )
    return {
        "rule": "merged >= max(A, B) on >=4/5 metals AND Holm-significant (paired one-sided t, df=4)",
        "resolvable_metals": resolvable,
        "n_resolvable": n_res,
        "merged_ge_max_metals": ge_max,
        "merged_ge_max_count": len(ge_max),
        "holm_significant_metals": sig,
        "holm_significant_count": len(sig),
        "acceptance_met": bool(met),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, object]:
    if getattr(args, "deterministic", False):
        # cuBLAS needs a fixed workspace for deterministic matmul on CUDA.
        _os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
        _torch.use_deterministic_algorithms(True, warn_only=True)

    device = _resolve_device(args.device)
    report: Dict[str, object] = {
        "metric": "stacked_dmpnn_xgb_within_metal",
        "status": "MEASURED",
        "MEASURED_vs_PROJECTED": "MEASURED",
        "description": (
            "FU2: stack the ECFP4→XGB scorer (component A) and the in-house "
            "D-MPNN scorer (component B) into one merged scorer via a "
            "meta-learner trained under strict nested CV. Closes G1 by routing "
            "around the weaker member instead of discarding it."
        ),
        "unit": "ROC-AUC (0-1, higher better); paired delta of merged vs each component",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/stack_dmpnn_xgb_within_metal.py",
        "data_source": "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026, 69(8), 8838)",
        "cohort_spec": getattr(args, "cohort_spec", "24h"),
        "cohort": (
            "paper_funnel (Krasnov-style): build_paper_cohort(raw, metal=m) then "
            "collapse_duplicates(tie_rule=active); the Table-1 / BLK-P1 cohort"
            if getattr(args, "cohort_spec", "24h") == "paper_funnel"
            else "24h: CytotoxFilter(time_threshold=24, ic50_min=0.01) then collapse_duplicates"
        ),
        "deterministic": bool(getattr(args, "deterministic", False)),
        "fold_protocol": (
            "StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed); "
            "identical to the probe's within_metal, BLK-P1, and FU1."
        ),
        "components": {
            COMPONENT_A: "ECFP4 (Morgan r=2, 2048) -> XGBoost (BLK-P1 / FU1 arm-1 anchor)",
            COMPONENT_B: "in-house masked D-MPNN scorer (MaskedMetalCytotoxModel, BLK-P1 protocol)",
        },
        "merged": (
            "meta-learner g([A, B]) on the base probabilities (or logits), trained "
            "with strict nested CV (inner K-fold OOF features for the meta-train "
            "rows; full outer-train base models for the meta-test rows)"
        ),
        "meta_learner": args.meta_learner,
        "meta_space": args.meta_space,
        "inner_splits": args.inner_splits,
        "split_seeds": list(args.split_seeds),
        "device": str(device),
        "hidden": args.hidden,
        "depth": args.depth,
        "min_unique_for_gate": MIN_UNIQUE_FOR_GATE,
        "per_metal": {},
    }

    for metal in args.metals:
        report["per_metal"][metal] = _run_metal(metal, args, device)

    report["family"] = _family_block(report["per_metal"])
    report["acceptance"] = _acceptance_block(report["per_metal"])

    report["honest_negative"] = {
        "pooled_mean_reported": False,
        "why": (
            "Ru is ~64% of the deduplicated cohort, so a pooled mean is a Ru "
            "mean wearing a 5-metal label. The per-metal table is the unit."
        ),
        "unresolvable_metals": [
            m for m, v in report["per_metal"].items() if not v.get("resolvable", False)
        ],
        "component_numbers_unchanged": (
            "The A and B columns are the same base learners as FU1's arm-1 anchor "
            "and BLK-P1's D-MPNN column; the component-level numbers are reported "
            "verbatim and are NOT rewritten. Only the FINAL model's definition "
            "changes (single D-MPNN -> merged scorer), per directive_2026_10_16."
        ),
        "fallback_if_stacking_does_not_win": (
            "If merged < max(A, B) on the resolvable metals, the honest reading is "
            "that the two scorers are not complementary at this scale and the "
            "merger must be re-designed (FU3 metal-conditioned router), not "
            "reported as a win. MEASURED, not PROJECTED."
        ),
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FU2 — stack ECFP4→XGB and D-MPNN into one within-metal scorer."
    )
    p.add_argument("--metals", nargs="+", default=list(METALS))
    p.add_argument("--split-seeds", type=int, nargs="+", default=[42, 43, 44, 49, 123])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--train-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-atoms", type=int, default=120)
    p.add_argument("--max-edges", type=int, default=240)
    p.add_argument(
        "--inner-splits", type=int, default=3,
        help="inner K-fold used to build leak-free meta-features for the meta-train rows",
    )
    p.add_argument(
        "--meta-learner", choices=["logreg", "gbdt"], default="logreg",
        help="the merger g([A, B]): logistic regression (default) or a shallow GBDT",
    )
    p.add_argument(
        "--meta-space", choices=["prob", "logit"], default="logit",
        help="base-learner output space fed to the meta-learner",
    )
    p.add_argument(
        "--cohort-spec", choices=["24h", "paper_funnel"], default="paper_funnel",
        help="cohort definition; paper_funnel = the Table-1 / BLK-P1 paper cohort",
    )
    p.add_argument(
        "--deterministic", action="store_true",
        help=(
            "Force bit-reproducible kernels (torch.use_deterministic_algorithms, "
            "cudnn.deterministic, CUBLAS_WORKSPACE_CONFIG). Required for CUDA runs."
        ),
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--max-fold-n", type=int, default=0,
        help="smoke only: truncate each metal to N ligands. 0 = full cohort.",
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="YAML config (AGENTS.md §5.11). Keys resolve to the dests above; "
        "an explicit CLI flag still wins.",
    )
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/stacked_dmpnn_xgb_within_metal.json",
    )
    return p


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Module-level entry point for scripts/_config_binding_gate.py."""
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

    print(f"\n{'metal':>5} {'A':>8} {'B':>8} {'merged':>8} {'maxAB':>8} {'d(max)':>9}  verdict")
    for m in args.metals:
        v = rep["per_metal"].get(m, {})
        if v.get("status") != "MEASURED":
            print(f"{m:>5}  SKIPPED: {v.get('reason', '')}")
            continue
        tc = v.get("three_column", {})
        d = v.get("deltas", {}).get("merged_minus_max_AB", {})
        dtxt = f"{d['mean']:+.4f}" if isinstance(d, dict) and "mean" in d else "n/a"
        verdict = v.get("verdicts", {}).get("merged_minus_max_AB", "")
        print(
            f"{m:>5} {tc.get('A_ecfp4_xgb'):8.4f} {tc.get('B_dmpnn'):8.4f} "
            f"{tc.get('merged'):8.4f} {tc.get('max_A_B'):8.4f} {dtxt:>9}  {verdict}"
        )
    acc = rep["acceptance"]
    print(f"\nacceptance_met: {acc['acceptance_met']} "
          f"(merged>=max on {acc['merged_ge_max_count']}/{acc['n_resolvable']} resolvable; "
          f"Holm-sig {acc['holm_significant_count']})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
