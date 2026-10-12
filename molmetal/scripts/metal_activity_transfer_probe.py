#!/usr/bin/env python3
"""Within-metal vs cross-metal anticancer activity — the transfer-penalty probe.

=====================================================================
What this measures, and why both halves are needed
=====================================================================
The field's headline numbers (Krasnov et al., J. Med. Chem. 2026, 69(8),
8838; MetalCytoToxDB) are **within-metal** ROC-AUC: 0.81 for Ru, 0.73
for Ir. That is a same-metal train/test split — the model has already
seen that metal's coordination chemistry during training.

Cross-metal (leave-one-metal-out) is a strictly harder question and
**nobody in the comparison set has published it**. Reporting only one
half produces a wrong conclusion in either direction:

* quoting within-metal AUC as if it were cross-metal overstates us;
* quoting cross-metal AUC as if it were the field's number understates
  the field.

So this script measures BOTH, on the SAME deduplicated cohort, with the
SAME model, and reports the difference as the transfer penalty. The
penalty is the finding; neither half is meaningful alone.

Why deduplication is load-bearing (MEASURED 2026-10-10)
--------------------------------------------------------
Raw MetalCytoToxDB rows: 5,271 across 5 metals, but only **1,908 unique
ligands** — 64% duplication, one ligand measured up to 15 times. Under
a Morgan fingerprint + GBDT scorer, a ligand present in both the train
metals and the held-out metal is simply memorised, so cross-metal
apparent transfer inflates. Measured effect of removing duplicates:

    pooled ROC-AUC   0.6444 (raw)  ->  0.5894 (deduplicated)
    top-5% lift      1.88         ->  1.54
    Os fold AUC      0.6912       ->  0.5318   (47% of Os ligands also
                                                appear in other metals)
    Re fold AUC      0.5701       ->  0.4179   (below chance)

Duplicates are collapsed by majority label. The predicate is `sum * 2 >= n`,
so an exact tie resolves to ACTIVE -- the ANTI-conservative direction for an
activity claim. An earlier version of this docstring said "a tie resolves to
inactive, which is the conservative choice"; that was wrong and is corrected
here 2026-10-11. The effect is real, not cosmetic: 80 ligand groups tie on
MetalCytoToxDB and all 80 are labelled active, and re-running within-metal
under the conservative rule moves Ru 0.8363 -> 0.7929, below Krasnov's
published 0.81, while Ir is unchanged. See
`metrics/by_metric/metal_activity_transfer_tie_sensitivity.json` and the
`collapse_duplicates` docstring below.

Honest-negative carried by this script
--------------------------------------
``Re`` cross-metal lands **below chance** (0.418 after dedup, from 0.928
within-metal). ``Rh`` (n=51 unique) and ``Os`` (n=36 unique) are too small
to resolve at all and are reported but not gated. Any paper claim must
carry the per-metal table, not the pooled mean, because the pooled mean
is dominated by whichever metal is largest (Ru is 64% of the cohort).

Model
-----
Morgan/ECFP4 (radius 2, 2048 bit) + XGBoost, with ``baseline_lomo_cv``'s
exact hyper-parameters so the two scripts' numbers are comparable. NOT
our D-MPNN, deliberately: this probe asks whether the *signal* exists in
the data before any GPU time is spent. Note ``baseline_lomo_cv``'s
per-fold ``hit_rate_top5pct`` is NOT quoted anywhere — its folds are
87-3668 rows, so top-5% is 4-183 molecules and the number is dominated
by k, not by chemistry.

Scaffold-split control (BLK-PA1, added 2026-10-12)
-------------------------------------------------
``within_metal`` is a *random* StratifiedKFold, so its train and test share
Bemis-Murcko scaffolds. That makes the within-vs-cross gap a confounded
quantity: it mixes the "novel metal" penalty with the ordinary "novel
scaffold" penalty. ``--with-scaffold-split`` adds a same-metal,
scaffold-disjoint 5-fold CV (deterministic largest-first grouping) and reports

    net_metal_effect = within_scaffold_auc - cross_metal_auc

which is the metal-shift penalty with the scaffold penalty removed. If
scaffold-split already sits near cross-metal, the "collapse" is ordinary
hard generalisation and is not specific to the metal axis.

Scaffolds are computed on the **desalted largest organic fragment**: the
MetalCytoToxDB rows are multi-fragment SMILES (``ligand.solvent.counterion``,
no explicit metal atom), and a raw Murcko call on them returns empty for
100% of rows (MEASURED 3668/3668 Ru). See :func:`_parent_murcko`.

Permutation null (BLK-PA2, added 2026-10-12)
--------------------------------------------
Three held-out folds have n_test of 36-74, so the "below chance" claims cannot
be judged against a nominal 0.5 — the null distribution of a fixed model's
ranking has real spread at that n. ``--permutation-null N`` shuffles each fold's
test labels N times and reports the null mean/sd/percentiles plus the empirical
p-values, so ``below_chance_significant`` is the honest flag (see
:func:`permutation_null`).

Usage
-----
::

    .venv/bin/python molmetal/scripts/metal_activity_transfer_probe.py \
        --out-metric metrics/by_metric/metal_activity_transfer.json

    # with the BLK-PA1 scaffold-split control:
    .venv/bin/python molmetal/scripts/metal_activity_transfer_probe.py \
        --with-scaffold-split \
        --out-metric metrics/by_metric/metal_activity_transfer_scaffold.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

METALS = ["Ru", "Ir", "Rh", "Os", "Re"]

# Krasnov et al. 2026 published WITHIN-Metal ROC-AUC. Recorded so the
# within-metal half of this script is checked against the literature
# rather than merely asserted. `None` = not reported for that metal.
KRASNOV_WITHIN_METAL_AUC = {"Ru": 0.81, "Ir": 0.73, "Rh": None, "Os": None, "Re": None}

# Emitted on request (--comparability-note) into the base report. This block used
# to be added to metal_activity_transfer.json by hand, which made the canonical
# unreproducible from committed code. It is prose that references other
# artifacts, so it lives here as a constant rather than being recomputed.
COMPARABILITY_NOTE = {
    "verdict": "WITHIN COLUMN IS NOT COMPARABLE TO KRASNOV 2026. The two use different cohorts.",
    "ours": (
        "within_metal_cv is 5-fold CV on the LOMO dedup cohort (Ru n_unique=1255 "
        "pos_rate 0.2574; Ir n_unique=534 pos_rate 0.4532)."
    ),
    "krasnov": (
        "Their 0.81 / 0.73 come from a cancer-only cohort of 15,799 rows / 4,793 "
        "complexes / 591 cell lines (see "
        "metalcytotox_ru_auc.json::paper_protocol_measurement)."
    ),
    "our_number_on_their_cohort": (
        "0.7717 +/- 0.0380 under their DOI-split 10-fold CV, which is 0.0383 BELOW "
        "their LightGBM 0.810 +/- 0.024, with 3/10 folds at or above. That is the "
        "only like-for-like comparison on disk."
    ),
    "tie_sensitivity": (
        "Both sides use the identical collapse predicate (sum*2 >= n, ties -> "
        "ACTIVE), so the tie rule does not confound the cross-metal comparison. It "
        "DOES move the within column: conservative rule gives Ru 0.7880-0.8082 (at "
        "or below Krasnov 0.81) while Ir is unchanged and stays above 0.73 under "
        "both rules and all 5 seeds."
    ),
    "withdrawn": (
        "The CLAIM 'within-metal matches or beats the published result' is "
        "WITHDRAWN for Ru. It is defensible for Ir only."
    ),
    "supersedes_a_bad_citation": (
        "todo.json::withdrawn_claims_2026_10_10 justified this withdrawal by citing "
        "'DOI-grouped 0.6350 / 0.6065'. Those two numbers DO NOT EXIST anywhere on "
        "disk; the only 0.6351 in the repo is an Os per-seed value in "
        "metal_lomo_auc.json, a different experiment. The withdrawal conclusion is "
        "correct; that citation was not traceable."
    ),
}

# Mirrors baseline_lomo_cv's XGBClassifier params exactly.
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "tree_method": "hist",
}

# Below this many unique ligands a metal's AUC is not resolvable and must
# not be gated. At n=36 a 95% CI on AUC is roughly +-0.16.
MIN_UNIQUE_FOR_GATE = 200


def _load_env_default() -> None:
    """``cytotox.DEFAULT_CSV`` honours MOLMETAL_DATA_DIR; mirror env.sh 1b.

    Without this the loader falls back to a /mnt/storage mount that does
    not exist on this host — measured FileNotFoundError 2026-10-10.
    """
    if not os.environ.get("MOLMETAL_DATA_DIR"):
        local = PROJECT_ROOT / ".data" / "molmetal"
        if (local / "MetalCytoToxDB.csv").exists():
            os.environ["MOLMETAL_DATA_DIR"] = str(local)


def collapse_duplicates(
    smiles: np.ndarray, y: np.ndarray, tie_rule: str = "active"
) -> Tuple[List[str], np.ndarray, int]:
    """One row per unique ligand; label = majority, **ties resolve to ACTIVE**.

    Returns ``(unique_smiles, unique_labels, n_raw_rows)``.

    ``tie_rule="active"`` (default) uses the predicate ``sum * 2 >= n``, so an
    exact tie (sum*2 == n) yields ``True`` → label 1. That is the
    **anti-conservative** direction for an activity claim, and earlier versions
    of this docstring (and of the matching trainer) described it as "ties
    resolve to inactive, the conservative choice" — that was wrong, verified
    2026-10-10 by evaluating the expression on n=2/sum=1 and n=4/sum=2 (both →
    label 1). ``tie_rule="inactive"`` uses ``sum * 2 > n`` and is the
    conservative counterpart; the two are compared in
    ``metal_activity_transfer_tie_sensitivity.json``.

    It is not a harmless typo: on MetalCytoToxDB 80 ligand groups tie (Ru 59,
    Ir 19, Rh 2), and all 80 are labelled active under the default rule.
    Re-running within-metal with the conservative rule ``sum * 2 > n`` moves Ru
    0.8363 → 0.7929 (below Krasnov's published 0.81) while Ir is unchanged
    (0.8295 → 0.8319). See
    ``metrics/by_metric/metal_activity_transfer_tie_sensitivity.json``.

    Keep the rule, but state its direction honestly. The tie itself is genuine
    label noise (see ``todo.json :: root_cause_analysis.RC1``), so neither
    direction is "right"; what matters is that no claim is presented as robust
    to a choice it is not robust to.
    """
    import pandas as pd

    if tie_rule not in ("active", "inactive"):
        raise ValueError(f"unknown tie_rule {tie_rule!r} (expected active|inactive)")
    df = pd.DataFrame({"s": [str(x) for x in smiles], "y": np.asarray(y, dtype=int)})
    if tie_rule == "active":
        agg = df.groupby("s")["y"].agg(lambda v: int(v.sum() * 2 >= len(v)))
    else:
        agg = df.groupby("s")["y"].agg(lambda v: int(v.sum() * 2 > len(v)))
    return list(agg.index), agg.to_numpy(dtype=int), len(df)


def _make_model(y_train: np.ndarray, seed: int):
    from xgboost import XGBClassifier

    params = dict(XGB_PARAMS, random_state=seed)
    pos = max(1, int(y_train.sum()))
    neg = max(1, int(len(y_train) - pos))
    params["scale_pos_weight"] = neg / pos
    return XGBClassifier(**params)


def _featurise(smiles: List[str], radius: int, nbits: int) -> np.ndarray:
    from molmetal.baselines.eval_utils import morgan_features

    return morgan_features(np.asarray(smiles), radius, nbits)


def within_metal(
    smiles: List[str],
    y: np.ndarray,
    seed: int,
    radius: int,
    nbits: int,
    n_splits: int = 5,
    return_scores: bool = False,
) -> Dict[str, object]:
    """Stratified k-fold CV inside a single metal — the field's own setting.

    ``return_scores=True`` (BLK-PA6) also returns the pooled out-of-fold
    ``oof_labels`` / ``oof_probs`` — every ligand scored by a model that did
    not see it — so a bootstrap CI can be put on the within-metal AUC.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    if len(set(y.tolist())) < 2 or len(y) < 2 * n_splits:
        return {"status": "SKIPPED", "reason": f"n={len(y)} too small or single-class"}

    X = _featurise(smiles, radius, nbits)
    aucs, aps = [], []
    oof_y: List[int] = []
    oof_p: List[float] = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = _make_model(y[tr], seed)
        clf.fit(X[tr], y[tr], verbose=False)
        prob = clf.predict_proba(X[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], prob)))
        aps.append(float(average_precision_score(y[te], prob)))
        if return_scores:
            oof_y.extend(int(v) for v in y[te])
            oof_p.extend(float(v) for v in prob)
    out: Dict[str, object] = {
        "status": "MEASURED",
        "n_unique": int(len(y)),
        "n_splits": int(n_splits),
        "roc_auc": float(np.mean(aucs)),
        "roc_auc_fold_std": float(np.std(aucs)),
        "pr_auc": float(np.mean(aps)),
        "pos_rate": float(y.mean()),
    }
    if return_scores:
        out["oof_labels"] = oof_y
        out["oof_probs"] = oof_p
    return out


def _parent_murcko(smi: str) -> str:
    """Bemis-Murcko scaffold of the *desalted largest organic fragment*.

    MetalCytoToxDB rows are multi-fragment SMILES
    (``ligand.solvent.counterion``, with no explicit metal atom). Feeding the
    raw string to ``MurckoScaffold.GetScaffoldForMol`` returns ``""`` for
    every row — MEASURED 3668/3668 Ru rows on 2026-10-12 — so a naive
    ``ScaffoldSplitter`` call degenerates to a per-ligand (i.e. random) split.
    We therefore take ``rdMolStandardize.FragmentParent`` (desalt + largest
    fragment), re-parse to sanitise it (``FragmentParent`` leaves RingInfo
    uninitialised, which makes Murcko raise), then take the Murcko scaffold.
    Returns ``""`` when no scaffold can be computed.
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    try:
        parent = rdMolStandardize.FragmentParent(mol)
        parent = Chem.MolFromSmiles(Chem.MolToSmiles(parent))  # re-parse => sanitised
        if parent is None or parent.GetNumAtoms() == 0:
            return ""
        sc = MurckoScaffold.GetScaffoldForMol(parent)
        return Chem.MolToSmiles(sc) if sc is not None else ""
    except Exception:
        return ""


def _scaffold_groups(smiles: List[str]) -> List[List[int]]:
    """Group ligand row-indices by Bemis-Murcko scaffold.

    Uses :func:`_parent_murcko` (desalt + largest fragment + sanitise) because
    the raw MetalCytoToxDB strings are multi-fragment and would otherwise all
    collapse to a single empty-scaffold bucket. Ligands whose scaffold is empty
    (acyclic or unparseable) are bucketed by their own SMILES so they never
    merge across rows.
    """
    groups: Dict[str, List[int]] = {}
    for i, s in enumerate(smiles):
        scaf = _parent_murcko(str(s))
        if not scaf:
            scaf = f"__empty__{s}"
        groups.setdefault(scaf, []).append(i)
    return list(groups.values())


def _largest_first_kfold(groups: List[List[int]], n_splits: int) -> List[List[int]]:
    """Deterministic largest-first assignment of scaffold groups into k folds.

    Groups are sorted by descending size and each is placed greedily into the
    currently smallest fold (row-count greedy). This is the k-way analogue of
    the repo's ``ScaffoldSplitter(strategy='largest_first')``; row-count (not
    group-count) balancing keeps folds of similar size so the per-fold AUC is
    comparable to the random StratifiedKFold in :func:`within_metal`.
    """
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    sizes = [0] * n_splits
    for g in sorted(groups, key=len, reverse=True):
        j = min(range(n_splits), key=lambda k: (sizes[k], k))
        folds[j].extend(g)
        sizes[j] += len(g)
    return folds


def within_metal_scaffold(
    smiles: List[str], y: np.ndarray, seed: int, radius: int, nbits: int, n_splits: int = 5
) -> Dict[str, object]:
    """5-fold CV inside a single metal with **scaffold-disjoint** folds.

    This is the missing control for the metal-shift claim (todo.json BLK-PA1).
    :func:`within_metal` uses a random StratifiedKFold, so its train and test
    share scaffolds; this function instead holds out whole Bemis-Murcko
    scaffolds, which is the *hardest* same-metal generalisation. Comparing the
    two isolates the "novel scaffold" penalty from the "novel metal" penalty.

    Deterministic: the largest-first assignment has no RNG, so ``seed`` only
    affects the XGBoost fit (as it does in :func:`within_metal`).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(set(y.tolist())) < 2 or len(y) < 2 * n_splits:
        return {"status": "SKIPPED", "reason": f"n={len(y)} too small or single-class"}

    groups = _scaffold_groups(smiles)
    folds = _largest_first_kfold(groups, n_splits)
    X = _featurise(smiles, radius, nbits)
    y = np.asarray(y, dtype=int)
    in_test = np.zeros(len(y), dtype=bool)

    aucs, aps, fold_rows = [], [], []
    for k, te in enumerate(folds):
        in_test[:] = False
        in_test[np.asarray(te, dtype=int)] = True
        te_idx = np.flatnonzero(in_test)
        tr_idx = np.flatnonzero(~in_test)
        if (
            len(te_idx) == 0
            or len(set(y[te_idx].tolist())) < 2
            or len(set(y[tr_idx].tolist())) < 2
        ):
            fold_rows.append(
                {"fold": int(k), "status": "SKIPPED", "n_test": int(len(te_idx))}
            )
            continue
        clf = _make_model(y[tr_idx], seed)
        clf.fit(X[tr_idx], y[tr_idx], verbose=False)
        prob = clf.predict_proba(X[te_idx])[:, 1]
        a = float(roc_auc_score(y[te_idx], prob))
        aucs.append(a)
        aps.append(float(average_precision_score(y[te_idx], prob)))
        fold_rows.append(
            {
                "fold": int(k),
                "status": "MEASURED",
                "n_test": int(len(te_idx)),
                "pos_rate_test": float(y[te_idx].mean()),
                "roc_auc": a,
            }
        )

    if not aucs:
        return {"status": "SKIPPED", "reason": "no measurable folds", "per_fold": fold_rows}

    return {
        "status": "MEASURED",
        "n_unique": int(len(y)),
        "n_scaffolds": int(len(groups)),
        "n_splits": int(n_splits),
        "n_folds_measured": int(len(aucs)),
        "roc_auc": float(np.mean(aucs)),
        "roc_auc_fold_std": float(np.std(aucs)),
        "pr_auc": float(np.mean(aps)),
        "pos_rate": float(y.mean()),
        "per_fold": fold_rows,
    }


def majority_baseline(y: np.ndarray) -> Dict[str, object]:
    """Trivial constant-predictor reference (BLK-PA7).

    A model that always predicts the majority class has no ranking power, so
    its ROC-AUC is 0.5 by construction — the honest trivial reference the
    transfer tables must be read against. We also report the base rate and
    the majority-class accuracy, because those are the numbers a reader can
    actually compare a classifier to.
    """
    y = np.asarray(y, dtype=int)
    base = float(y.mean())
    maj = 1 if base >= 0.5 else 0
    return {
        "status": "MEASURED",
        "n": int(len(y)),
        "base_rate": base,
        "majority_class": maj,
        "majority_accuracy": float((y == maj).mean()),
        "roc_auc": 0.5,
        "note": (
            "ROC-AUC is 0.5 by construction for a constant predictor; "
            "majority_accuracy is the class-aware trivial reference."
        ),
    }


def pooled_split(
    smiles: List[str],
    y: np.ndarray,
    seed: int,
    radius: int,
    nbits: int,
    mode: str = "random",
    n_splits: int = 5,
) -> Dict[str, object]:
    """Metal-agnostic 5-fold CV over the pooled cohort (BLK-PA7).

    ``mode='random'`` is the no-shift upper bound (metal identity ignored,
    rows shuffled). ``mode='scaffold'`` keeps the metal axis collapsed but
    holds out whole Bemis-Murcko scaffolds, isolating the *novel-scaffold*
    penalty without the metal penalty. Together with the cross-metal LOMO
    they bracket how much of the collapse is metal shift vs ordinary OOD.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y, dtype=int)
    if len(set(y.tolist())) < 2 or len(y) < 2 * n_splits:
        return {"status": "SKIPPED", "reason": f"n={len(y)} too small or single-class"}

    X = _featurise(smiles, radius, nbits)
    if mode == "scaffold":
        folds = _largest_first_kfold(_scaffold_groups(smiles), n_splits)
    elif mode == "random":
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [te for _, te in skf.split(X, y)]
    else:
        raise ValueError(f"mode must be 'random' or 'scaffold', got {mode!r}")

    aucs, aps = [], []
    for te in folds:
        te = np.asarray(te, dtype=int)
        in_te = np.zeros(len(y), dtype=bool)
        in_te[te] = True
        tr = np.flatnonzero(~in_te)
        if (
            len(te) == 0
            or len(set(y[te].tolist())) < 2
            or len(set(y[tr].tolist())) < 2
        ):
            continue
        clf = _make_model(y[tr], seed)
        clf.fit(X[tr], y[tr], verbose=False)
        prob = clf.predict_proba(X[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], prob)))
        aps.append(float(average_precision_score(y[te], prob)))
    if not aucs:
        return {"status": "SKIPPED", "reason": "no measurable folds"}
    return {
        "status": "MEASURED",
        "mode": mode,
        "n_unique": int(len(y)),
        "n_splits": int(n_splits),
        "n_folds_measured": int(len(aucs)),
        "roc_auc": float(np.mean(aucs)),
        "roc_auc_fold_std": float(np.std(aucs)),
        "pr_auc": float(np.mean(aps)),
        "pos_rate": float(y.mean()),
    }


def cross_metal(
    per_metal: Dict[str, Tuple[List[str], np.ndarray]],
    seed: int,
    radius: int,
    nbits: int,
    return_scores: bool = False,
) -> Dict[str, object]:
    """Leave-one-metal-out. A ligand seen in any other metal is dropped
    from the training pool, so no fingerprint can straddle the split.

    ``return_scores=True`` adds the raw ``test_labels`` / ``test_probs`` to
    each MEASURED fold so a permutation null can be computed downstream
    (callers pop them before persisting the report)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    pooled_p: List[np.ndarray] = []
    pooled_y: List[np.ndarray] = []
    pooled_m: List[np.ndarray] = []
    folds: List[Dict[str, object]] = []

    for held_out in per_metal:
        te_s, te_y = per_metal[held_out]
        if len(set(te_y.tolist())) < 2:
            folds.append({"metal": held_out, "status": "SKIPPED", "reason": "single-class"})
            continue
        tr_s: List[str] = []
        tr_y: List[int] = []
        n_leaked = 0
        for m, (s_list, y_arr) in per_metal.items():
            if m == held_out:
                continue
            blocked = set(te_s)
            for s, yv in zip(s_list, y_arr.tolist()):
                if s in blocked:
                    n_leaked += 1
                    continue
                tr_s.append(s)
                tr_y.append(yv)
        if len(set(tr_y)) < 2:
            folds.append({"metal": held_out, "status": "SKIPPED", "reason": "train_pool_single_class"})
            continue
        ytr = np.asarray(tr_y, dtype=int)
        Xtr = _featurise(tr_s, radius, nbits)
        Xte = _featurise(te_s, radius, nbits)
        clf = _make_model(ytr, seed)
        clf.fit(Xtr, ytr, verbose=False)
        prob = clf.predict_proba(Xte)[:, 1]
        fold: Dict[str, object] = {
            "metal": held_out,
            "status": "MEASURED",
            "roc_auc": float(roc_auc_score(te_y, prob)),
            "pr_auc": float(average_precision_score(te_y, prob)),
            "n_train_unique": int(len(tr_y)),
            "n_test_unique": int(len(te_y)),
            "n_train_rows_dropped_as_leaked": int(n_leaked),
            "pos_rate_test": float(te_y.mean()),
            "resolvable": bool(len(te_y) >= MIN_UNIQUE_FOR_GATE),
        }
        if return_scores:
            fold["test_labels"] = [int(v) for v in te_y.tolist()]
            fold["test_probs"] = [float(v) for v in prob.tolist()]
        folds.append(fold)
        pooled_p.append(prob)
        pooled_y.append(te_y)
        pooled_m.append(np.array([held_out] * len(te_y), dtype=object))

    if not pooled_p:
        return {"status": "INSUFFICIENT_FOLDS", "per_fold": folds}

    p_all = np.concatenate(pooled_p)
    y_all = np.concatenate(pooled_y)
    m_all = np.concatenate(pooled_m)
    order = np.argsort(-p_all, kind="stable")
    y_sorted = y_all[order]
    base = float(y_all.mean())
    n = len(y_all)

    enrichment = {}
    for frac in (0.01, 0.05, 0.10, 0.25, 0.50):
        k = max(1, int(round(frac * n)))
        hits = int(y_sorted[:k].sum())
        hit = hits / k
        se = (hit * (1.0 - hit) / k) ** 0.5
        enrichment[f"top_{int(frac * 100)}pct"] = {
            "k": k,
            "n_hits": hits,
            "hit_rate": hit,
            "base_rate": base,
            "lift": (hit / base) if base > 0 else float("nan"),
            "hit_rate_ci95": [max(0.0, hit - 1.96 * se), min(1.0, hit + 1.96 * se)],
        }

    return {
        "status": "MEASURED",
        "per_fold": folds,
        "pooled": {
            "n": int(n),
            "base_rate": base,
            "roc_auc": float(roc_auc_score(y_all, p_all)),
            "pr_auc": float(average_precision_score(y_all, p_all)),
            "n_tied_scores": int(n - len(np.unique(p_all))),
            "enrichment": enrichment,
            "peak_enrichment_k": max(enrichment.items(), key=lambda kv: kv[1]["lift"])[0],
        },
    }


def permutation_null(
    y_true: List[int], prob: List[float], n_perm: int, seed: int
) -> Dict[str, object]:
    """Exchangeability null for a fixed model's test ranking (BLK-PA2).

    Under H0 the test labels carry no information about the model's scores, so
    shuffling them and recomputing ROC-AUC gives the null distribution the
    observed AUC must be judged against. This is the correct reference for the
    small held-out folds: at n_test=36 a *nominal* 0.5 is not the right line —
    the null itself has real spread (~±0.09), so an observed 0.5195 may be
    entirely inside it.

    Caveat (stated so it is not over-read): this permutes the TEST labels only,
    holding the trained model fixed. It answers "is this model's ranking
    informative on this test set?" — it is not a retrain-under-shuffled-labels
    null, which would cost ``n_perm`` full fits.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    if len(set(y.tolist())) < 2:
        return {"status": "SKIPPED", "reason": "single-class test fold"}
    observed = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        null[i] = roc_auc_score(rng.permutation(y), p)
    p_below = float((null <= observed).mean())
    p_above = float((null >= observed).mean())
    return {
        "status": "MEASURED",
        "n_perm": int(n_perm),
        "n_test": int(len(y)),
        "observed_auc": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std()),
        "null_pct": {
            "2.5": float(np.percentile(null, 2.5)),
            "5": float(np.percentile(null, 5)),
            "50": float(np.percentile(null, 50)),
            "95": float(np.percentile(null, 95)),
            "97.5": float(np.percentile(null, 97.5)),
        },
        "p_below_chance": p_below,
        "p_above_chance": p_above,
        "p_two_sided": float(min(1.0, 2.0 * min(p_below, p_above))),
        "below_chance_significant": bool(p_below < 0.05),
        "above_chance_significant": bool(p_above < 0.05),
    }


def run(
    metals: List[str],
    seed: int,
    radius: int,
    nbits: int,
    include_raw: bool = False,
    with_scaffold_split: bool = False,
    n_permutations: int = 0,
    n_bootstrap: int = 0,
    with_baselines: bool = False,
    tie_rule: str = "active",
    comparability_note: bool = False,
) -> Dict[str, object]:
    _load_env_default()
    import pandas as pd

    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

    raw: Dict[str, Tuple[List[str], np.ndarray]] = {}
    dedup: Dict[str, Tuple[List[str], np.ndarray]] = {}
    dup_stats: Dict[str, object] = {}

    for m in metals:
        ds = MetalCytotoxDataset.from_csv(
            filters=CytotoxFilter(time_threshold=24.0, ic50_min=0.01, metal_whitelist=[m])
        )
        s_raw = [str(x) for x in ds.smiles]
        y_raw = np.asarray(ds.active, dtype=int)
        s_u, y_u, n_raw = collapse_duplicates(ds.smiles, ds.active, tie_rule=tie_rule)
        raw[m] = (s_raw, y_raw)
        dedup[m] = (s_u, y_u)
        dup_stats[m] = {
            "n_rows_raw": int(n_raw),
            "n_unique": int(len(s_u)),
            "duplication_rate": 1.0 - len(s_u) / max(1, n_raw),
            "pos_rate_raw": float(y_raw.mean()),
            "pos_rate_dedup": float(y_u.mean()),
        }

    report: Dict[str, object] = {
        "metric": "metal_activity_transfer",
        "description": (
            "Anticancer cytotoxicity AUC for noble-metal complexes, measured "
            "BOTH within-metal (5-fold CV, the setting Krasnov et al. 2026 "
            "report) and cross-metal (leave-one-metal-out, which the "
            "comparison set does not publish), on the SAME deduplicated "
            "cohort with the SAME scorer. The difference is the transfer "
            "penalty and is the actual finding."
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/metal_activity_transfer_probe.py",
        "model": f"morgan_fp(radius={radius}, {nbits}bit) + xgboost",
        "data_source": "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026, 69(8), 8838)",
        "seed": seed,
        "dedup_rule": "one row per unique ligand; label = majority; ties -> ACTIVE (sum*2>=n; anti-conservative direction; see tie_sensitivity field)",
        "leakage_rule": "a ligand appearing in ANY other metal is dropped from the LOMO train pool",
        "krasnov_within_metal_auc": KRASNOV_WITHIN_METAL_AUC,
        "duplication": dup_stats,
    }

    within: Dict[str, object] = {}
    within_scores: Dict[str, Tuple[List[int], List[float]]] = {}
    for m in metals:
        s_u, y_u = dedup[m]
        within[m] = within_metal(
            s_u, y_u, seed, radius, nbits, return_scores=(n_bootstrap > 0)
        )
        within[m]["krasnov_reported_auc"] = KRASNOV_WITHIN_METAL_AUC.get(m)
        if n_bootstrap > 0 and within[m].get("status") == "MEASURED":
            # pop so the persisted per-metal block stays small (BLK-PA6)
            within_scores[m] = (
                within[m].pop("oof_labels"),
                within[m].pop("oof_probs"),
            )
    report["within_metal_cv"] = within

    cross = cross_metal(
        {m: dedup[m] for m in dedup},
        seed,
        radius,
        nbits,
        return_scores=(n_permutations > 0 or n_bootstrap > 0),
    )
    report["cross_metal_lomo_dedup"] = cross

    # ---- BLK-PA2: permutation null for the below-chance claims ----
    # At n_test=36-74 a nominal 0.5 is the wrong reference; we need the null
    # distribution of a fixed model's test ranking. Pop the raw scores so the
    # persisted per_fold stays small.
    #
    # BLK-PA6: the same block also emits a bootstrap 95% CI on each fold's
    # cross-metal AUC (`--bootstrap-ci N`) and a Holm-corrected version of the
    # permutation p-values (multiple comparison across the 5 metals).
    if n_permutations > 0 or n_bootstrap > 0:
        from molmetal.data.leakage_utils import bootstrap_auc

        perm: Dict[str, object] = {}
        boot: Dict[str, object] = {}
        cross_scores: Dict[str, Tuple[List[int], List[float]]] = {}
        for f in cross.get("per_fold", []):
            if f.get("status") != "MEASURED":
                continue
            yl = f.pop("test_labels", None)
            yp = f.pop("test_probs", None)
            if yl is None or yp is None:
                continue
            metal = str(f["metal"])
            cross_scores[metal] = (yl, yp)
            if n_permutations > 0:
                perm[metal] = permutation_null(yl, yp, n_permutations, seed)
            if n_bootstrap > 0:
                _mean, _lo, _hi, _ = bootstrap_auc(yl, yp, n_bootstrap, seed)
                boot[metal] = {
                    "n_boot": int(n_bootstrap),
                    "observed_auc": float(f["roc_auc"]),
                    "ci95": [float(_lo), float(_hi)],
                    "ci_excludes_chance": bool(float(_lo) > 0.5 or float(_hi) < 0.5),
                }
        if n_permutations > 0:
            report["permutation_null"] = perm
            # Holm-Bonferroni across the metals, on the two-sided empirical p.
            _items = sorted(
                ((m, v.get("p_two_sided", 1.0)) for m, v in perm.items()),
                key=lambda kv: kv[1],
            )
            _n = len(_items)
            _adj: Dict[str, float] = {}
            _running = 0.0
            for _rank, (_m, _p) in enumerate(_items):
                _running = max(_running, min(1.0, (float(_p)) * (_n - _rank)))
                _adj[_m] = _running
            report["permutation_null_holm"] = {
                "correction": "Holm-Bonferroni (family = 5 held-out metals)",
                "n_family": _n,
                "p_two_sided_holm": _adj,
                "significant_after_holm_0.05": [
                    m for m, p in _adj.items() if p < 0.05
                ],
            }
        if n_bootstrap > 0:
            report["cross_metal_bootstrap_ci"] = boot

            # BLK-PA6: the within-metal regime needs the SAME treatment, and the
            # headline "transfer penalty" needs a CI. within_metal and
            # cross_metal score DIFFERENT test sets (within = all ligands via
            # 5-fold OOF; cross = the held-out metal's ligands), so a per-ligand
            # PAIRED test is not defined. We bootstrap each regime's test
            # ligands independently and take the difference distribution; the
            # penalty CI is therefore UNPAIRED and labelled as such.
            within_boot: Dict[str, object] = {}
            penalty_ci: Dict[str, object] = {}
            for m in metals:
                if m not in within_scores:
                    continue
                wl, wp = within_scores[m]
                w_mean, w_lo, w_hi, _ = bootstrap_auc(wl, wp, n_bootstrap, seed)
                within_boot[m] = {
                    "n_boot": int(n_bootstrap),
                    "observed_auc": float(within[m]["roc_auc"]),
                    "ci95": [float(w_lo), float(w_hi)],
                    "ci_excludes_chance": bool(float(w_lo) > 0.5 or float(w_hi) < 0.5),
                }
                if m in cross_scores:
                    cl, cp = cross_scores[m]
                    c_mean, c_lo, c_hi, _ = bootstrap_auc(cl, cp, n_bootstrap, seed)
                    penalty_ci[m] = {
                        "within_auc": float(w_mean),
                        "cross_auc": float(c_mean),
                        "transfer_penalty": float(w_mean - c_mean),
                        "penalty_ci95": [float(w_lo - c_hi), float(w_hi - c_lo)],
                        "ci_excludes_zero": bool(
                            (w_lo - c_hi) > 0.0 or (w_hi - c_lo) < 0.0
                        ),
                        "pairing": "unpaired (independent bootstrap; different test sets)",
                    }
            report["within_metal_bootstrap_ci"] = within_boot
            report["transfer_penalty_ci"] = penalty_ci

    # ---- the finding: transfer penalty per metal ----
    penalty: Dict[str, object] = {}
    for m in metals:
        w = within[m]
        folds = {f["metal"]: f for f in cross.get("per_fold", []) if f.get("status") == "MEASURED"}
        if w.get("status") != "MEASURED" or m not in folds:
            continue
        f = folds[m]
        penalty[m] = {
            "within_auc": w["roc_auc"],
            "cross_auc": f["roc_auc"],
            "transfer_penalty": float(w["roc_auc"] - f["roc_auc"]),
            "cross_below_chance": bool(f["roc_auc"] < 0.5),
            "resolvable": bool(f.get("resolvable", False)),
            "krasnov_reported_auc": KRASNOV_WITHIN_METAL_AUC.get(m),
        }
    report["transfer_penalty"] = penalty

    # ---- BLK-PA7: baselines to read the transfer table against ----
    # A cross-metal AUC of ~0.5 is only meaningful next to (a) the trivial
    # constant predictor, (b) a metal-agnostic model on a random split (the
    # no-shift upper bound), and (c) the same model on a scaffold split (the
    # OOD-without-metal penalty). Together they separate "the model can't
    # learn this cohort" from "the metal axis is the thing that breaks it".
    if with_baselines:
        all_s: List[str] = []
        all_y: List[int] = []
        for m in metals:
            s_u, y_u = dedup[m]
            all_s.extend(s_u)
            all_y.extend(int(v) for v in y_u)
        pool_s, pool_y, _n_pool_raw = collapse_duplicates(all_s, all_y)
        report["baselines"] = {
            "note": (
                "Reference points for the cross-metal table. 'random' is the "
                "no-shift upper bound (metal identity ignored, rows shuffled); "
                "'scaffold' holds out whole Bemis-Murcko scaffolds so the "
                "novel-scaffold penalty is isolated from the metal penalty."
            ),
            "pooled_n_unique": int(len(pool_y)),
            "pooled_pos_rate": float(pool_y.mean()),
            "majority_class": majority_baseline(pool_y),
            "metal_agnostic_random_split": pooled_split(
                pool_s, pool_y, seed, radius, nbits, mode="random"
            ),
            "metal_agnostic_scaffold_split": pooled_split(
                pool_s, pool_y, seed, radius, nbits, mode="scaffold"
            ),
        }

    # ---- BLK-PA1: the same-metal scaffold-split control ----
    # within_metal uses a random StratifiedKFold, so its train and test share
    # scaffolds. Holding out whole Bemis-Murcko scaffolds gives the hardest
    # same-metal generalisation; net_metal_effect = scaffold - cross is the
    # metal-shift penalty with the "novel scaffold" penalty already removed.
    if with_scaffold_split:
        scaf: Dict[str, object] = {}
        for m in metals:
            s_u, y_u = dedup[m]
            scaf[m] = within_metal_scaffold(s_u, y_u, seed, radius, nbits)
        report["within_metal_scaffold_split"] = scaf

        cross_folds = {
            f["metal"]: f
            for f in cross.get("per_fold", [])
            if f.get("status") == "MEASURED"
        }
        net: Dict[str, object] = {}
        for m in metals:
            w = within.get(m, {})
            sc = scaf.get(m, {})
            f = cross_folds.get(m)
            if (
                w.get("status") != "MEASURED"
                or sc.get("status") != "MEASURED"
                or f is None
            ):
                continue
            net[m] = {
                "within_random_auc": w["roc_auc"],
                "within_scaffold_auc": sc["roc_auc"],
                "cross_metal_auc": f["roc_auc"],
                "scaffold_penalty": float(w["roc_auc"] - sc["roc_auc"]),
                "net_metal_effect": float(sc["roc_auc"] - f["roc_auc"]),
                "resolvable": bool(f.get("resolvable", False)),
            }
        report["net_metal_effect"] = net

    resolvable = [m for m, v in penalty.items() if v["resolvable"]]
    hn: Dict[str, object] = {
        "below_chance_cross_metal": [m for m, v in penalty.items() if v["cross_below_chance"]],
        "unresolvable_small_folds": [
            m for m, v in penalty.items() if not v["resolvable"]
        ],
        "note": (
            "Ru is 64% of the deduplicated cohort, so the pooled cross-metal "
            "mean is dominated by Ru and must not be quoted without this "
            "per-metal table."
        ),
    }
    # BLK-PA2: below_chance_cross_metal is a POINT ESTIMATE (AUC < 0.5). At
    # n_test=36-74 that is not evidence of below-chance transfer — the
    # label-permutation null has real spread (~±0.09). When the null was
    # computed, surface the significance-gated verdict so the paper quotes the
    # field that survives the test.
    perm_report = report.get("permutation_null")
    if perm_report:
        hn["below_chance_significant"] = [
            m for m, v in perm_report.items() if v.get("below_chance_significant")
        ]
        hn["above_chance_significant"] = [
            m for m, v in perm_report.items() if v.get("above_chance_significant")
        ]
        hn["note"] += (
            " below_chance_cross_metal is a point estimate only; quote "
            "below_chance_significant / above_chance_significant "
            "(label-permutation null) for any chance-level claim."
        )
    report["honest_negative"] = hn

    if include_raw:
        report["cross_metal_lomo_RAW_undeduplicated"] = cross_metal(raw, seed, radius, nbits)

    if comparability_note:
        report["comparability_note"] = COMPARABILITY_NOTE

    return report


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Within- vs cross-metal activity probe.")
    p.add_argument("--metals", nargs="+", default=METALS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument(
        "--include-raw",
        action="store_true",
        help="also run LOMO on the un-deduplicated rows, to quantify the leakage effect",
    )
    p.add_argument(
        "--with-scaffold-split",
        action="store_true",
        help=(
            "also run a same-metal Bemis-Murcko scaffold-split 5-fold CV "
            "(BLK-PA1 control); adds within_metal_scaffold_split + "
            "net_metal_effect to the report"
        ),
    )
    p.add_argument(
        "--permutation-null",
        type=int,
        default=0,
        metavar="N",
        help=(
            "compute a label-permutation null (N shuffles) for each cross-metal "
            "fold (BLK-PA2); adds a permutation_null section. 0 = off"
        ),
    )
    p.add_argument(
        "--bootstrap-ci",
        type=int,
        default=0,
        metavar="N",
        help=(
            "bootstrap (N resamples) a 95%% CI on each cross-metal fold's AUC "
            "(BLK-PA6); adds a cross_metal_bootstrap_ci section. 0 = off"
        ),
    )
    p.add_argument(
        "--with-baselines",
        action="store_true",
        help=(
            "also emit the BLK-PA7 reference baselines (majority-class, "
            "metal-agnostic random-split upper bound, metal-agnostic "
            "scaffold-split); adds a baselines section"
        ),
    )
    p.add_argument(
        "--tie-rule",
        choices=("active", "inactive"),
        default="active",
        help=(
            "how collapse_duplicates resolves an exact majority tie: 'active' "
            "(sum*2>=n, the default and the anti-conservative direction) or "
            "'inactive' (sum*2>n). The two are compared in "
            "metal_activity_transfer_tie_sensitivity.json"
        ),
    )
    p.add_argument(
        "--comparability-note",
        action="store_true",
        help=(
            "emit the comparability_note block: the within column is not "
            "comparable to Krasnov 2026 (different cohorts), the Ru "
            "'matches-or-beats' claim is withdrawn, and the tie rule's effect on "
            "it. Prose, so it is a constant, not recomputed. On for the canonical "
            "base artifact only"
        ),
    )
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/metal_activity_transfer.json",
    )
    args = p.parse_args(argv)

    rep = run(
        metals=list(args.metals),
        seed=args.seed,
        radius=args.radius,
        nbits=args.nbits,
        include_raw=args.include_raw,
        with_scaffold_split=args.with_scaffold_split,
        n_permutations=args.permutation_null,
        n_bootstrap=args.bootstrap_ci,
        with_baselines=args.with_baselines,
        tie_rule=args.tie_rule,
        comparability_note=args.comparability_note,
    )

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = Path(args.out_metric)
    atomic_write_file(out, json.dumps(rep, indent=2, default=float))

    print(f"{'metal':>5} {'n_uniq':>7} {'within':>8} {'krasnov':>8} {'cross':>8} {'penalty':>9}")
    for m in METALS:
        w = rep["within_metal_cv"].get(m, {})
        pen = rep["transfer_penalty"].get(m)
        if pen is None:
            print(f"{m:>5} {w.get('n_unique', 0):7d} "
                  f"{w.get('roc_auc', float('nan')):8.4f} "
                  f"{str(w.get('krasnov_reported_auc') or 'n/a'):>8} {'-':>8} {'-':>9}")
            continue
        print(f"{m:>5} {w['n_unique']:7d} {pen['within_auc']:8.4f} "
              f"{str(pen['krasnov_reported_auc'] or 'n/a'):>8} {pen['cross_auc']:8.4f} "
              f"{pen['transfer_penalty']:9.4f}"
              + ("  BELOW-CHANCE" if pen["cross_below_chance"] else ""))

    pool = rep["cross_metal_lomo_dedup"].get("pooled")
    if pool:
        print(f"\npooled cross-metal n={pool['n']} AUC={pool['roc_auc']:.4f} "
              f"peak enrichment at {pool['peak_enrichment_k']}")
        for k, c in pool["enrichment"].items():
            print(f"  {k:>10}  k={c['k']:5d}  hit={c['hit_rate']:.3f}  lift={c['lift']:.2f}")
    hn = rep["honest_negative"]
    print(f"\nhonest-negative: below-chance(point est)={hn['below_chance_cross_metal']} "
          f"unresolvable={hn['unresolvable_small_folds']}")
    if "below_chance_significant" in hn:
        print(f"                 below-chance(sig)={hn['below_chance_significant']} "
              f"above-chance(sig)={hn['above_chance_significant']}")

    net = rep.get("net_metal_effect")
    if net:
        print(f"\n{'metal':>5} {'random':>8} {'scaffold':>9} {'cross':>8} "
              f"{'scaf_pen':>9} {'net_metal':>10}")
        for m in METALS:
            v = net.get(m)
            if v is None:
                continue
            print(f"{m:>5} {v['within_random_auc']:8.4f} "
                  f"{v['within_scaffold_auc']:9.4f} {v['cross_metal_auc']:8.4f} "
                  f"{v['scaffold_penalty']:9.4f} {v['net_metal_effect']:10.4f}"
                  + ("" if v["resolvable"] else "  (n too small)"))

    perm = rep.get("permutation_null")
    if perm:
        print(f"\n{'metal':>5} {'n_test':>7} {'obs':>7} {'null_mu':>8} {'null_sd':>8} "
              f"{'null_5%':>8} {'null_95%':>8} {'p_below':>8} {'p_above':>8}  verdict")
        for m in METALS:
            v = perm.get(m)
            if not v or v.get("status") != "MEASURED":
                continue
            npc = v["null_pct"]
            if v["below_chance_significant"]:
                verdict = "BELOW (sig)"
            elif v["above_chance_significant"]:
                verdict = "ABOVE (sig)"
            else:
                verdict = "in null (not resolvable)"
            print(f"{m:>5} {v['n_test']:7d} {v['observed_auc']:7.4f} "
                  f"{v['null_mean']:8.4f} {v['null_sd']:8.4f} {npc['5']:8.4f} "
                  f"{npc['95']:8.4f} {v['p_below_chance']:8.4f} {v['p_above_chance']:8.4f}"
                  f"  {verdict}")

    holm = rep.get("permutation_null_holm")
    if holm:
        print(f"\nHolm-corrected (family={holm['n_family']}): "
              f"significant after Holm = {holm['significant_after_holm_0.05']}")
        for m, p in holm["p_two_sided_holm"].items():
            print(f"  {m:>5} p_holm={p:.4f}")

    boot = rep.get("cross_metal_bootstrap_ci")
    if boot:
        print(f"\n{'metal':>5} {'observed':>9} {'ci95_lo':>8} {'ci95_hi':>8}  "
              f"excludes_chance")
        for m in METALS:
            v = boot.get(m)
            if not v:
                continue
            print(f"{m:>5} {v['observed_auc']:9.4f} {v['ci95'][0]:8.4f} "
                  f"{v['ci95'][1]:8.4f}  {v['ci_excludes_chance']}")

    wboot = rep.get("within_metal_bootstrap_ci")
    if wboot:
        print(f"\nwithin-metal bootstrap CI (BLK-PA6):")
        print(f"{'metal':>5} {'observed':>9} {'ci95_lo':>8} {'ci95_hi':>8}  "
              f"excludes_chance")
        for m in METALS:
            v = wboot.get(m)
            if not v:
                continue
            print(f"{m:>5} {v['observed_auc']:9.4f} {v['ci95'][0]:8.4f} "
                  f"{v['ci95'][1]:8.4f}  {v['ci_excludes_chance']}")

    pen = rep.get("transfer_penalty_ci")
    if pen:
        print(f"\ntransfer-penalty CI (unpaired, BLK-PA6):")
        print(f"{'metal':>5} {'within':>8} {'cross':>8} {'penalty':>9} "
              f"{'ci95_lo':>8} {'ci95_hi':>8}  excl_0")
        for m in METALS:
            v = pen.get(m)
            if not v:
                continue
            print(f"{m:>5} {v['within_auc']:8.4f} {v['cross_auc']:8.4f} "
                  f"{v['transfer_penalty']:9.4f} {v['penalty_ci95'][0]:8.4f} "
                  f"{v['penalty_ci95'][1]:8.4f}  {v['ci_excludes_zero']}")

    bl = rep.get("baselines")
    if bl:
        print(f"\nbaselines (pooled n={bl['pooled_n_unique']}, "
              f"pos_rate={bl['pooled_pos_rate']:.3f}):")
        mc = bl["majority_class"]
        print(f"  majority-class : auc={mc['roc_auc']:.4f} "
              f"acc={mc['majority_accuracy']:.4f} (class={mc['majority_class']})")
        for key, label in (
            ("metal_agnostic_random_split", "metal-agnostic random"),
            ("metal_agnostic_scaffold_split", "metal-agnostic scaffold"),
        ):
            v = bl[key]
            if v.get("status") != "MEASURED":
                print(f"  {label:22s}: {v.get('status')} ({v.get('reason', '')})")
                continue
            print(f"  {label:22s}: auc={v['roc_auc']:.4f} "
                  f"±{v['roc_auc_fold_std']:.4f} pr_auc={v['pr_auc']:.4f}")

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())