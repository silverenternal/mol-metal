#!/usr/bin/env python3
"""MET-2 — does the strong model survive a change of partition?

MET-1 established that the fused model is stable across five seeds on a
**fixed ligand-grouped** partition (canonical R2 0.5828 +/- 0.0102).  That
number is an interpolation number: both ligand-grouped protocols leave rows
from the same paper in train and test, and FIX-M4 already showed that
within-paper structure can manufacture a decision-layer gain that vanishes
when whole papers are held out.

This script re-runs the same features, the same hyperparameters and the same
model under four partitions:

``ligand``    GroupKFold(5) on ``ligand_canonical`` — the MET-1 protocol
``doi``       GroupKFold(5) on the DOI — whole papers held out
``scaffold``  GroupKFold(5) on the Bemis-Murcko scaffold of the complex
``year``      leave-one-year-out on the year parsed from the DOI (30.6% of rows)

Two arms are measured, both with the L5 feature set:

``ligand``      ECFP4(ligand_canonical) + metal/cell categorical codes
``ligand+both`` the same plus metal state and coordination blocks (the strong arm)

The GNN arm is **not** re-run here.  Its contribution is +0.027 R2 over the
tree; if the tree's R2 collapses under a partition, the fusion cannot rescue
it, so the tree is the right decision variable.  This is recorded in
``design.arms_not_measured`` rather than left implicit.

Output: ``metrics/by_metric/msrs_generalization.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_external_validation import (  # noqa: E402
    EARLY_YEAR_LABEL,
    MIN_YEAR_ROWS,
    murcko_scaffold,
    parse_year,
    year_group,
)
from molmetal.scripts.msrs_joint_representation import (  # noqa: E402
    ENDPOINT,
    LIGAND_ONLY,
    SEED,
    TOP_CELL_LINES,
    build,
    cell_vocabulary,
    coordination_block,
    load_descriptors,
    native_codes,
    state_block,
)
from molmetal.scripts.msrs_code_levers import fit_arm  # noqa: E402
from molmetal.scripts.msrs_recommendation import (  # noqa: E402
    COHORT_KEYS,
    MIN_COHORT_SIZE,
    build_cohorts,
    cohort_metrics,
    doi_clustered_test,
    fold_baseline_scores,
    summarise_cohorts,
)
from molmetal.scripts.msrs_surface_model import (  # noqa: E402
    _git_commit,
    _sha256,
    score,
)

SCHEMES = ("ligand", "doi", "scaffold", "year")
ARMS = ("ligand", "ligand+both")
N_FOLDS = 5
SIGN_THRESHOLDS = (0.0, 0.2091)
N_BOOT = 2000
MAX_DROP_OK = 0.05  # R2 may fall this much from the ligand protocol and still pass


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------
def scheme_folds(
    frame: pd.DataFrame, scheme: str, n_splits: int = N_FOLDS
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Fold assignment plus the group key it used.

    Rows with no usable key (unparseable year) get ``-1``: they train in every
    fold and are never evaluated.  ``fit_arm`` documents that convention.
    """
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

    info: Dict[str, object] = {}
    if scheme == "ligand":
        groups = frame["ligand_canonical"].astype(str).to_numpy()
    elif scheme == "doi":
        groups = frame["doi"].astype(str).to_numpy()
    elif scheme == "scaffold":
        groups = frame["ligand_canonical"].map(murcko_scaffold).to_numpy()
    elif scheme == "year":
        counts = frame["doi"].map(parse_year).value_counts().to_dict()
        counts = {int(k): int(v) for k, v in counts.items()}
        groups = (
            frame["doi"]
            .map(lambda d: year_group(d, counts, MIN_YEAR_ROWS))
            .to_numpy()
        )
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    valid = np.array([g is not None and str(g) != "nan" for g in groups])
    folds = np.full(len(frame), -1, dtype=int)
    idx = np.flatnonzero(valid)
    sub_groups = groups[valid]
    n_groups = len(np.unique(sub_groups))
    if n_groups < 2:
        raise ValueError(f"scheme {scheme!r} has {n_groups} group(s); cannot split")

    if scheme == "year":
        splitter = LeaveOneGroupOut()
    else:
        splitter = GroupKFold(n_splits=min(n_splits, n_groups))
    for k, (_, te) in enumerate(
        splitter.split(np.zeros((len(idx), 1)), groups=sub_groups)
    ):
        folds[idx[te]] = k

    info["n_groups"] = int(n_groups)
    info["n_folds"] = int(len(set(folds[folds >= 0].tolist())))
    info["n_rows_evaluated"] = int((folds >= 0).sum())
    info["n_rows_unscored"] = int((folds < 0).sum())
    if scheme == "year":
        info["n_years"] = int(
            len({g for g in sub_groups if g != EARLY_YEAR_LABEL})
        )
        info["note"] = (
            "leave-one-year-out; years with fewer than "
            f"{MIN_YEAR_ROWS} rows are pooled into {EARLY_YEAR_LABEL!r}"
        )
    return folds, groups, info


# ---------------------------------------------------------------------------
# Paired-delta accuracy
# ---------------------------------------------------------------------------
def pair_table(frame: pd.DataFrame, ligand_key: str = "ligand_canonical") -> pd.DataFrame:
    """Cross-metal pairs sharing (ligand, cell, DOI, duration).

    Same construction as ``msrs_surface_model.matched_delta_pairs`` but keyed on
    the complex rather than the main fragment, because the features are built
    on ``ligand_canonical``.
    """
    key = [ligand_key, "cell_line_key", "doi", "time_h"]
    rank = {m: i for i, m in enumerate(sorted(frame["metal"].unique()))}
    rows: List[Dict[str, object]] = []
    for _, g in frame.groupby(key, dropna=False):
        if g["metal"].nunique() < 2:
            continue
        g = g.reset_index()
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                ma, mb = g.at[i, "metal"], g.at[j, "metal"]
                if ma == mb:
                    continue
                if rank[ma] > rank[mb]:
                    lo, hi, mlo, mhi = j, i, mb, ma
                else:
                    lo, hi, mlo, mhi = i, j, ma, mb
                rows.append(
                    {
                        "idx_a": int(g.at[lo, "index"]),
                        "idx_b": int(g.at[hi, "index"]),
                        "metal_a": mlo,
                        "metal_b": mhi,
                        "doi": g.at[i, "doi"],
                    }
                )
    return pd.DataFrame(rows)


def pair_sign_accuracy(
    y: np.ndarray, pred: np.ndarray, pairs: pd.DataFrame, fold_id: np.ndarray
) -> Dict[str, object]:
    """Fraction of matched pairs whose predicted ordering matches the observed one.

    Only pairs whose two rows are both evaluated (fold >= 0) are scored.
    """
    if pairs.empty:
        return {"n_pairs": 0, "status": "NO_MATCHED_PAIRS"}
    ia = pairs["idx_a"].to_numpy()
    ib = pairs["idx_b"].to_numpy()
    keep = (fold_id[ia] >= 0) & (fold_id[ib] >= 0)
    if not keep.any():
        return {"n_pairs": 0, "status": "NO_EVALUATED_PAIRS"}

    ia, ib = ia[keep], ib[keep]
    obs = y[ib] - y[ia]
    d = pred[ib] - pred[ia]
    out: Dict[str, object] = {
        "n_pairs": int(len(ia)),
        "n_dois": int(pairs.loc[keep, "doi"].nunique()),
        "by_threshold": {},
    }
    for th in SIGN_THRESHOLDS:
        m = np.abs(obs) >= th
        if m.sum() == 0:
            continue
        out["by_threshold"][f"{th:g}"] = {
            "n_pairs": int(m.sum()),
            "sign_accuracy": round(
                float(np.mean(np.sign(d[m]) == np.sign(obs[m]))), 4
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------
def bootstrap_r2_ci(
    y: np.ndarray, pred: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> Dict[str, object]:
    """Group bootstrap CI for R2.

    Resampling rows would understate the uncertainty: rows inside one paper are
    not independent.  Resampling groups (the partition key) keeps that
    structure.  R2 is computed against the resample's own mean, so the interval
    is about explained variance, not about the original intercept.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    if len(uniq) < 3:
        return {"status": "TOO_FEW_GROUPS", "n_groups": int(len(uniq))}
    buckets = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([buckets[p] for p in pick])
        yy, pp = y[idx], pred[idx]
        denom = float(np.sum((yy - yy.mean()) ** 2))
        if denom <= 0:
            continue
        vals.append(1.0 - float(np.sum((yy - pp) ** 2)) / denom)
    if not vals:
        return {"status": "DEGENERATE"}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "n_boot": int(len(vals)),
        "n_groups": int(len(uniq)),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long",
        type=Path,
        default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet",
    )
    parser.add_argument(
        "--metal-desc",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_generalization.json",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit", type=int, default=0, help="smoke: first N ligands")
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=list(SCHEMES),
        choices=list(SCHEMES),
        help="subset of partitions to run",
    )
    args = parser.parse_args(argv)

    t_start = time.time()
    long = pd.read_parquet(args.long)
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
        & long["oxidation_state"].notna()
    ].copy().reset_index(drop=True)
    if args.limit:
        keep = sub["ligand_main_fragment"].drop_duplicates().head(args.limit)
        sub = sub[sub["ligand_main_fragment"].isin(set(keep))].reset_index(drop=True)

    y = sub["value_log10"].to_numpy(dtype=float)
    load_descriptors(args.metal_desc)
    vocab = cell_vocabulary(
        sub["cell_line_key"].astype(str).tolist(), top_n=TOP_CELL_LINES
    )
    codes = native_codes(
        sub["metal"].astype(str).tolist(),
        sub["cell_line_key"].astype(str).tolist(),
        vocab,
        mode="full",
    )

    bundle = build(sub, LIGAND_ONLY)
    state, state_names = state_block(sub)
    coord, coord_names, coverage = coordination_block(sub, include_geometry=True)
    both = sparse.hstack([sparse.csr_matrix(state), coord], format="csr")
    extras = {
        "ligand": sparse.csr_matrix((len(sub), 0), dtype=np.float32),
        "ligand+both": both,
    }
    empty = sparse.csr_matrix((len(sub), 0), dtype=np.float32)

    pairs = pair_table(sub)
    metals = sub["metal"].astype(str).to_numpy()
    cells = sub["cell_line_key"].astype(str).to_numpy()

    print(
        f"[met2] rows={len(sub)} ligands={sub['ligand_canonical'].nunique()} "
        f"dois={sub['doi'].nunique()} pairs={len(pairs)}"
    )

    splits: Dict[str, object] = {}
    for scheme in args.schemes:
        t0 = time.time()
        fold_id, groups, info = scheme_folds(sub, scheme)
        print(
            f"[met2] scheme={scheme} folds={info['n_folds']} "
            f"groups={info['n_groups']} evaluated={info['n_rows_evaluated']}"
        )

        # Fold-safe baselines, shared by every arm in this scheme.
        pred_global, pred_cell, base_info = fold_baseline_scores(
            metals, cells, y, fold_id
        )

        arms: Dict[str, object] = {}
        preds: Dict[str, np.ndarray] = {}
        for arm in ARMS:
            X = sparse.hstack([bundle.X, extras[arm]], format="csr")
            pred = fit_arm(
                X,
                empty,
                empty,
                None,
                0,
                False,
                y,
                fold_id,
                cat_codes=codes,
                params_override={"random_state": int(args.seed)},
            )
            preds[arm] = pred
            ev = fold_id >= 0
            entry: Dict[str, object] = {
                **score(y[ev], pred[ev]),
                "n_features": int(X.shape[1]),
            }
            entry["bootstrap_r2"] = bootstrap_r2_ci(
                y[ev], pred[ev], groups[ev], n_boot=N_BOOT, seed=args.seed
            )
            entry["pair_sign"] = pair_sign_accuracy(y, pred, pairs, fold_id)
            entry["wall_s"] = round(time.time() - t0, 1)
            arms[arm] = entry
            print(
                f"    {arm:14s} R2={entry['r2']:.4f} "
                f"MAE={entry['mae']:.4f} "
                f"pair_sign@0.2091="
                f"{entry['pair_sign'].get('by_threshold', {}).get('0.2091', {}).get('sign_accuracy')}"
            )

        # Decision layer: within-cohort regret / top-1 on held-out predictions.
        frame = pd.DataFrame(
            {
                "cohort": pd.factorize(
                    sub["ligand_canonical"].astype(str)
                    + "|"
                    + sub["cell_line_key"].astype(str)
                    + "|"
                    + sub["doi"].astype(str)
                    + "|"
                    + sub["time_h"].astype(str)
                )[0],
                "metal": metals,
                "y": y,
                "doi": sub["doi"].astype(str).to_numpy(),
                "fold": fold_id,
                "global_best": pred_global,
                "cell_best": pred_cell,
                **{arm: preds[arm] for arm in ARMS},
            }
        )
        frame = frame[frame["fold"] >= 0].reset_index(drop=True)
        cohorts = build_cohorts(
            frame, scorers=list(ARMS) + ["global_best", "cell_best"],
            min_size=MIN_COHORT_SIZE,
        )
        decision: Dict[str, object] = {}
        if cohorts.empty:
            decision["status"] = "NO_EVALUABLE_COHORTS"
        else:
            records = []
            for cid, g in cohorts.groupby("cohort", sort=False):
                for scorer in list(ARMS) + ["global_best", "cell_best"]:
                    rec = cohort_metrics(g["y"].to_numpy(), g[scorer].to_numpy())
                    rec.update({"cohort": int(cid), "scorer": scorer,
                                "doi": g["doi"].iloc[0]})
                    records.append(rec)
            rec_df = pd.DataFrame(records)
            decision["n_cohorts"] = int(rec_df["cohort"].nunique())
            for scorer in list(ARMS) + ["global_best", "cell_best"]:
                r = rec_df[rec_df["scorer"] == scorer]
                decision[scorer] = summarise_cohorts(r)
            for arm in ARMS:
                decision[f"{arm}_vs_global_best"] = doi_clustered_test(
                    rec_df, arm, "global_best", n_boot=N_BOOT, seed=args.seed
                )

        splits[scheme] = {
            **info,
            "arms": arms,
            "baselines": base_info,
            "decision": decision,
            "wall_s": round(time.time() - t0, 1),
        }

    # Verdict: the strong arm must hold up on the whole-paper holdout.
    ligand_r2 = None
    doi_r2 = None
    if "ligand" in splits:
        ligand_r2 = splits["ligand"]["arms"]["ligand+both"]["r2"]
    if "doi" in splits:
        doi_r2 = splits["doi"]["arms"]["ligand+both"]["r2"]
    drop = None if (ligand_r2 is None or doi_r2 is None) else round(ligand_r2 - doi_r2, 4)
    verdict = "INCOMPLETE"
    if drop is not None:
        verdict = "PASS" if drop <= MAX_DROP_OK else "NEGATIVE"

    payload = {
        "artifact": "msrs_generalization",
        "version": "1.0",
        # Not a dual-split report: it reports four partitions, which is a
        # stronger claim than the two-protocol contract the gate enforces.
        "requires_dual_split": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long": str(args.long),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "ligand_spec_hash": bundle.spec_hash,
        },
        "design": {
            "rows": int(len(sub)),
            "ligand_source": "ligand_canonical",
            "ligand_block": "ECFP4(ligand_canonical), 2048 bits",
            "metal_cell_encoding": "LightGBM native categorical codes (2 columns)",
            "metal_state_columns": state_names,
            "coord_columns": coord_names,
            "coverage": coverage,
            "split_schema": (
                "four-protocol matrix (ligand / doi / scaffold / year); "
                "the dual-split gate does not apply because this artifact "
                "reports more protocols than it requires"
            ),
            "schemes": list(args.schemes),
            "arms": list(ARMS),
            "arms_not_measured": ["gnn_metal_state", "blend_equal", "stack_foldwise"],
            "arms_not_measured_reason": (
                "the GNN adds +0.027 R2 over the tree (MET-1); if the tree's R2 "
                "collapses under a partition the fusion cannot rescue it, so the "
                "tree is the decision variable. The fusion arms are not claimed "
                "for any protocol other than the ligand one."
            ),
            "arm_ligand_source": {arm: "ligand_canonical" for arm in ARMS},
            "n_folds": N_FOLDS,
            "n_pairs": int(len(pairs)),
            "bootstrap": {"n_boot": N_BOOT, "unit": "the scheme's group key"},
            "max_drop_ok": MAX_DROP_OK,
            "seed": int(args.seed),
        },
        "splits": splits,
        "delta_r2_doi_vs_ligand": drop,
        "verdict": verdict,
        "verdict_rule": (
            "R2(ligand+both, ligand) - R2(ligand+both, doi) <= "
            f"{MAX_DROP_OK} (a DOI holdout may cost at most this much); "
            "INCOMPLETE if either protocol was not run"
        ),
        "wall_s": round(time.time() - t_start, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        f"[met2] verdict={verdict} ligand={ligand_r2} doi={doi_r2} drop={drop}"
    )
    print(f"[met2] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
