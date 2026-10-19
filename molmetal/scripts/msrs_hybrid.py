#!/usr/bin/env python3
"""FIX-M2 — combine the strong tree model with the D-MPNN.

FIX-L5 left the tree model at canonical R² 0.5556; FIX-M1 left the D-MPNN at
0.5031.  Neither is a win on its own, and NEG-20 closed the "swap the model
class" axis.  The remaining question is whether the two make *different*
mistakes: a graph model reads local chemistry, a tree on ECFP4 + metal state
reads global fingerprints, and if their residuals are decorrelated a blend can
beat both.

Arms, all evaluated on the same GroupKFold(5) partition as FIX-L5 / FIX-M1:

  lgb_tree        LightGBM on ligand+both (FIX-L5)          — the strong model
  gnn_metal_state D-MPNN + metal node, state conditioning   — FIX-M1
  blend_equal     0.5 * tree + 0.5 * gnn                    — zero parameters
  blend_foldwise  w fit on the other four folds, applied out of fold
  stack_foldwise  a*tree + b*gnn + c, fit the same way

``blend_equal`` is the primary result precisely because it fits nothing: there
is no weight that could have been tuned on the evaluation rows.  The two
fold-wise arms are standard stacking and carry the usual mild optimism; they
are reported to show whether a learned weight is worth anything at all.

Also reported: the residual correlation, and an ORACLE weight fitted on all
rows (an upper bound, never a result).

Usage::

    .venv/bin/python molmetal/scripts/msrs_hybrid.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from molmetal.scripts.msrs_code_levers import COHORT_KEYS, folds_for
from molmetal.scripts.msrs_decision_policy import MIN_COHORT_SIZE
from molmetal.scripts.msrs_dual_split import (
    SPLIT_CANONICAL_GROUPED,
    SPLIT_FRAGMENT_GROUPED,
    write_report,
)
from molmetal.scripts.msrs_metal_block import paired_group_test
from molmetal.scripts.msrs_surface_model import _git_commit, _sha256, score

TREE_ARM = "ligand+both"
GNN_ARM = "metal_state"
SPLITS = (SPLIT_FRAGMENT_GROUPED, SPLIT_CANONICAL_GROUPED)
GROUP_COL = {
    SPLIT_FRAGMENT_GROUPED: "ligand_main_fragment",
    SPLIT_CANONICAL_GROUPED: "ligand_canonical",
}
MIN_GAIN = 0.005


# ---------------------------------------------------------------------------
# Loading and alignment
# ---------------------------------------------------------------------------
def load_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def align(
    tree: Mapping[str, np.ndarray],
    gnn: Mapping[str, np.ndarray],
    split: str,
    tree_arm: str = TREE_ARM,
    gnn_arm: str = GNN_ARM,
) -> pd.DataFrame:
    """Build one row-aligned frame; raises if the two OOF files disagree."""
    ty, gy = tree["y"], gnn["y"]
    if len(ty) != len(gy):
        raise ValueError(f"row count differs: tree {len(ty)} vs gnn {len(gy)}")
    if not np.allclose(ty, gy, equal_nan=True):
        raise ValueError("labels differ — the two OOF files are not the same cohort")
    for key in ("ligand_canonical", "ligand_main_fragment", "metal", "cell_line_key", "doi"):
        if key in tree and key in gnn and not np.array_equal(tree[key], gnn[key]):
            raise ValueError(f"column {key!r} differs between the two OOF files")

    tkey, gkey = f"{split}__{tree_arm}", f"{split}__{gnn_arm}"
    for k in (tkey, gkey):
        if k not in tree and k not in gnn:
            raise KeyError(f"missing OOF column {k!r}")

    frame = pd.DataFrame(
        {
            "ligand_canonical": tree["ligand_canonical"],
            "ligand_main_fragment": tree["ligand_main_fragment"],
            "metal": tree["metal"],
            "cell_line_key": tree["cell_line_key"],
            "doi": tree["doi"],
            "time_h": tree["time_h"] if "time_h" in tree else gnn["time_h"],
            "y": ty.astype(float),
            "tree": tree[tkey].astype(float),
            "gnn": gnn[gkey].astype(float),
        }
    )
    frame["fold"] = folds_for(frame[GROUP_COL[split]].to_numpy())
    return frame


# ---------------------------------------------------------------------------
# Blend estimators
# ---------------------------------------------------------------------------
def fit_one_weight(
    y: np.ndarray, p_tree: np.ndarray, p_gnn: np.ndarray
) -> float:
    """Least-squares w in ``w*p_tree + (1-w)*p_gnn``, clipped to [0, 1]."""
    d = p_tree - p_gnn
    denom = float(d @ d)
    if denom <= 0.0:
        return 0.5
    w = float((y - p_gnn) @ d / denom)
    return float(np.clip(w, 0.0, 1.0))


def fit_linear(
    y: np.ndarray, p_tree: np.ndarray, p_gnn: np.ndarray
) -> Tuple[float, float, float]:
    """Least-squares ``a*p_tree + b*p_gnn + c`` (no clipping)."""
    design = np.column_stack([p_tree, p_gnn, np.ones_like(p_tree)])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def foldwise(
    frame: pd.DataFrame, kind: str
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Fit on the other folds, predict this fold. Returns preds + per-fold params."""
    pred = np.full(len(frame), np.nan)
    params: List[Dict[str, float]] = []
    y = frame["y"].to_numpy()
    p_tree = frame["tree"].to_numpy()
    p_gnn = frame["gnn"].to_numpy()
    folds = frame["fold"].to_numpy()
    for k in sorted(np.unique(folds)):
        train = folds != k
        test = folds == k
        if kind == "one_weight":
            w = fit_one_weight(y[train], p_tree[train], p_gnn[train])
            pred[test] = w * p_tree[test] + (1.0 - w) * p_gnn[test]
            params.append({"fold": int(k), "w_tree": round(w, 4), "w_gnn": round(1.0 - w, 4)})
        elif kind == "linear":
            a, b, c = fit_linear(y[train], p_tree[train], p_gnn[train])
            pred[test] = a * p_tree[test] + b * p_gnn[test] + c
            params.append(
                {"fold": int(k), "a_tree": round(a, 4), "b_gnn": round(b, 4), "c": round(c, 4)}
            )
        else:
            raise ValueError(f"unknown kind {kind!r}")
    if np.isnan(pred).any():
        raise RuntimeError("fold-wise blend left rows unscored")
    return pred, params


def oracle_weight(y: np.ndarray, p_tree: np.ndarray, p_gnn: np.ndarray) -> float:
    """In-sample optimum — an upper bound, never a result."""
    return fit_one_weight(y, p_tree, p_gnn)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def residual_correlation(frame: pd.DataFrame) -> Dict[str, float]:
    rt = frame["y"] - frame["tree"]
    rg = frame["y"] - frame["gnn"]
    pearson = float(stats.pearsonr(rt, rg).statistic)
    spearman = float(stats.spearmanr(rt, rg).statistic)
    return {
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
        "n_rows": int(len(frame)),
    }


def rank_blend(frame: pd.DataFrame, p_tree: np.ndarray, p_gnn: np.ndarray) -> np.ndarray:
    """Mean within-cohort rank of the two models, scaled to [0, 1].

    Uses no labels and no fitted parameters: it only asks whether the two
    models agree on *which* metal is better inside a cohort, which is the
    quantity the recommendation layer actually consumes.

    Ties are broken toward the tree model with a 1e-9 nudge.  Without it the
    mean rank ties in 158 of 424 cohorts, and ``idxmin`` silently resolves them
    by row order — which inflated top-1 from 0.566 to 0.620 in the first
    version of this arm.  The nudge is far below any real rank gap.
    """
    cohort = frame.groupby(COHORT_KEYS, dropna=False, sort=False).ngroup()
    out = np.zeros(len(frame))
    table = pd.DataFrame(
        {"c": cohort.to_numpy(), "tree": p_tree, "gnn": p_gnn}
    ).reset_index(drop=True)
    for _, g in table.groupby("c", sort=False):
        rt = g["tree"].rank(method="average", pct=True)
        rg = g["gnn"].rank(method="average", pct=True)
        out[g.index.to_numpy()] = 0.5 * (rt.to_numpy() + rg.to_numpy()) + 1e-9 * rt.to_numpy()
    return out


def rank_tie_stats(frame: pd.DataFrame, p_tree: np.ndarray, p_gnn: np.ndarray) -> Dict[str, object]:
    """How often the un-nudged mean rank ties, and what the row order buys."""
    cohort = frame.groupby(COHORT_KEYS, dropna=False, sort=False).ngroup()
    table = pd.DataFrame(
        {
            "c": cohort.to_numpy(),
            "metal": frame["metal"].to_numpy(),
            "tree": p_tree,
            "gnn": p_gnn,
            "y": frame["y"].to_numpy(),
        }
    )
    # Mirror build_cohorts: pool replicates per (cohort, metal), keep real cohorts.
    cells = (
        table.groupby(["c", "metal"], as_index=False, sort=False)
        .agg(tree=("tree", "median"), gnn=("gnn", "median"), y=("y", "median"))
    )
    n_metals = cells.groupby("c")["metal"].nunique()
    cells = cells[cells["c"].isin(n_metals[n_metals >= MIN_COHORT_SIZE].index)]

    n_tied = 0
    first_is_best = []
    for _, g in cells.groupby("c", sort=False):
        rt = g["tree"].rank(method="average", pct=True).to_numpy()
        rg = g["gnn"].rank(method="average", pct=True).to_numpy()
        score = 0.5 * (rt + rg)
        tied = g[score == score.min()]
        if len(tied) > 1:
            n_tied += 1
            first_is_best.append(bool(tied.iloc[0]["y"] == g["y"].min()))
    return {
        "n_cohorts": int(cells["c"].nunique()),
        "n_tied_at_min": int(n_tied),
        "first_row_is_best_when_tied": round(float(np.mean(first_is_best)), 4) if first_is_best else None,
    }


def run_split(
    frame: pd.DataFrame, split: str
) -> Tuple[Dict[str, object], Dict[str, object], pd.DataFrame]:
    group_col = GROUP_COL[split]
    y = frame["y"].to_numpy()
    p_tree = frame["tree"].to_numpy()
    p_gnn = frame["gnn"].to_numpy()

    equal = 0.5 * (p_tree + p_gnn)
    fold_one, params_one = foldwise(frame, "one_weight")
    fold_lin, params_lin = foldwise(frame, "linear")
    oracle = oracle_weight(y, p_tree, p_gnn)
    oracle_pred = oracle * p_tree + (1.0 - oracle) * p_gnn

    preds = {
        "lgb_tree": p_tree,
        "gnn_metal_state": p_gnn,
        "blend_equal": equal,
        "blend_foldwise": fold_one,
        "stack_foldwise": fold_lin,
    }
    out: Dict[str, object] = {}
    for name, pred in preds.items():
        entry = score(y, pred)
        entry["vs_tree"] = paired_group_test(
            frame, np.abs(y - p_tree), np.abs(y - pred), group_col
        )
        out[name] = entry

    best = max(
        ("blend_equal", "blend_foldwise", "stack_foldwise"),
        key=lambda a: out[a]["r2"],
    )
    delta = float(out[best]["r2"] - out["lgb_tree"]["r2"])
    diagnostics = {
        "best_blend": best,
        "delta_r2_vs_tree": round(delta, 4),
        "oracle": {
            "w_tree": round(oracle, 4),
            "r2": round(float(score(y, oracle_pred)["r2"]), 4),
            "note": "in-sample weight — upper bound only, excluded from the verdict",
        },
        "weights": {"one_weight": params_one, "linear": params_lin},
        "residual_correlation": residual_correlation(frame),
        "rank_tie_check": rank_tie_stats(frame, p_tree, p_gnn),
    }
    oof = pd.DataFrame(
        {
            "ligand_canonical": frame["ligand_canonical"],
            "ligand_main_fragment": frame["ligand_main_fragment"],
            "metal": frame["metal"],
            "cell_line_key": frame["cell_line_key"],
            "doi": frame["doi"],
            "time_h": frame["time_h"],
            "y": y,
            **{f"{split}__{k}": v for k, v in preds.items()},
            f"{split}__blend_rank": rank_blend(frame, p_tree, p_gnn),
        }
    )
    return out, diagnostics, oof


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tree-oof",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_joint_representation" / "oof.npz",
    )
    parser.add_argument(
        "--gnn-oof",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_dmpnn_surface" / "oof.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_hybrid.json",
    )
    parser.add_argument(
        "--oof-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_hybrid" / "oof.npz",
    )
    args = parser.parse_args(argv)

    tree = load_arrays(args.tree_oof)
    gnn = load_arrays(args.gnn_oof)

    splits: Dict[str, object] = {}
    diagnostics: Dict[str, object] = {}
    oof_parts: List[pd.DataFrame] = []
    for split in SPLITS:
        print(f"[hybrid] split={split}")
        frame = align(tree, gnn, split)
        entry, diag, part = run_split(frame, split)
        splits[split] = entry
        diagnostics[split] = diag
        oof_parts.append(part)
        print(
            f"    tree={entry['lgb_tree']['r2']:.4f} "
            f"gnn={entry['gnn_metal_state']['r2']:.4f} "
            f"equal={entry['blend_equal']['r2']:.4f} "
            f"foldwise={entry['blend_foldwise']['r2']:.4f} "
            f"stack={entry['stack_foldwise']['r2']:.4f}"
        )

    canonical = diagnostics[SPLIT_CANONICAL_GROUPED]
    best = canonical["best_blend"]
    delta = canonical["delta_r2_vs_tree"]
    verdict = "POSITIVE" if delta >= MIN_GAIN else "NEGATIVE"

    payload = {
        "artifact": "msrs_hybrid",
        "version": "1.0",
        "requires_dual_split": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "tree_oof": str(args.tree_oof.relative_to(PROJECT_ROOT)),
            "tree_oof_sha256": _sha256(args.tree_oof),
            "gnn_oof": str(args.gnn_oof.relative_to(PROJECT_ROOT)),
            "gnn_oof_sha256": _sha256(args.gnn_oof),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
        },
        "design": {
            "tree_arm": TREE_ARM,
            "gnn_arm": GNN_ARM,
            "ligand_source": "ligand_canonical",
            "primary_arm": "blend_equal",
            "min_gain": MIN_GAIN,
            "note": (
                "blend_equal has no fitted parameters and is the primary result. "
                "blend_foldwise / stack_foldwise fit on the other four folds "
                "(standard stacking, mild optimism). ORACLE is in-sample and is "
                "not a result."
            ),
            "arm_ligand_source": {
                "lgb_tree": "ligand_canonical",
                "gnn_metal_state": "ligand_canonical",
                "blend_equal": "ligand_canonical",
                "blend_foldwise": "ligand_canonical",
                "stack_foldwise": "ligand_canonical",
            },
        },
        "splits": splits,
        "diagnostics": diagnostics,
        "best_blend": best,
        "delta_r2_best_blend_vs_tree": delta,
        "verdict": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, payload)

    if args.oof_out:
        left, right = oof_parts
        if not np.allclose(left["y"].to_numpy(), right["y"].to_numpy()):
            raise RuntimeError("row order diverged between splits")
        merged = pd.concat(
            [left, right.drop(columns=["ligand_canonical", "ligand_main_fragment",
                                       "metal", "cell_line_key", "doi", "time_h", "y"])],
            axis=1,
        )
        arrays = {}
        for c in merged.columns:
            values = merged[c].to_numpy()
            if values.dtype == object:
                values = values.astype(str)
            arrays[c] = values
        args.oof_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.oof_out, **arrays)
        print(f"[hybrid] wrote {args.oof_out}")

    print(f"[hybrid] verdict={verdict} best={best} delta={delta:+.4f}")
    print(f"[hybrid] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
