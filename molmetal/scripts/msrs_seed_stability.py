#!/usr/bin/env python3
"""MET-1 — five-seed stability of the strong MSRS model.

FIX-L5 left the tree at canonical R² 0.5556 and FIX-M2 left the equal-weight
blend at 0.5727 / the stack at 0.5807.  All of those are **one seed**
(20261018).  A single seed is a draw, not a result: the paper's main table has
to carry a mean ± standard deviation and a per-seed breakdown.

This script reads the row-level OOF files produced by

    msrs_joint_representation.py --seed k --oof-out tree_seed{k}.npz
    msrs_dmpnn_surface.py --config configs/msrs_dmpnn_surface_seed{k}.yaml \\
        --oof-out gnn_seed{k}.npz

for k in 0..4, rebuilds the five FIX-M2 arms on the same folds, and reports:

  * per-seed R² / MAE / RMSE for each arm and split;
  * mean ± std across seeds;
  * the paired difference vs the tree, one-sided t over the five seeds;
  * how often the blend beats the tree (out of 5);
  * the fold-wise blend weight per seed (is the weight itself stable?).

The fold partition is deterministic (GroupKFold on the ligand key), so the
only thing that changes between seeds is the model's own randomness
(subsample / colsample / dropout / init).  That is the intended estimand.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_seed_stability.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy import stats

from molmetal.scripts.msrs_dual_split import (
    SPLIT_CANONICAL_GROUPED,
    SPLIT_FRAGMENT_GROUPED,
    write_report,
)
from molmetal.scripts.msrs_hybrid import (
    GNN_ARM,
    TREE_ARM,
    align,
    load_arrays,
    run_split,
)
from molmetal.scripts.msrs_surface_model import _git_commit

ARMS = ("lgb_tree", "gnn_metal_state", "blend_equal", "blend_foldwise", "stack_foldwise")
SPLITS = (SPLIT_FRAGMENT_GROUPED, SPLIT_CANONICAL_GROUPED)
SEEDS = (0, 1, 2, 3, 4)
MAX_STD = 0.02


def collect_seed(
    seed: int, tree_dir: Path, gnn_dir: Path
) -> Dict[str, Dict[str, float]]:
    """Rebuild every arm for one seed. Returns ``{split: {arm: r2}}``."""
    tree = load_arrays(tree_dir / f"tree_seed{seed}.npz")
    gnn = load_arrays(gnn_dir / f"gnn_seed{seed}.npz")
    out: Dict[str, Dict[str, float]] = {}
    for split in SPLITS:
        frame = align(tree, gnn, split, tree_arm=TREE_ARM, gnn_arm=GNN_ARM)
        table, diagnostics, _ = run_split(frame, split)
        out[split] = {
            "r2": {arm: float(table[arm]["r2"]) for arm in ARMS},
            "mae": {arm: float(table[arm]["mae"]) for arm in ARMS},
            "rmse": {arm: float(table[arm]["rmse"]) for arm in ARMS},
            "weights_one_weight": [
                float(p["w_tree"]) for p in diagnostics["weights"]["one_weight"]
            ],
            "residual_correlation": float(diagnostics["residual_correlation"]["pearson"]),
        }
    return out


def summarise(values: Sequence[float]) -> Dict[str, object]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
        "n": int(len(arr)),
        "per_seed": [round(float(v), 4) for v in arr],
    }


def paired_vs_tree(
    per_seed: Dict[int, Dict[str, Dict[str, float]]], split: str, arm: str
) -> Dict[str, object]:
    seeds = sorted(per_seed)
    deltas = np.asarray(
        [per_seed[s][split]["r2"][arm] - per_seed[s][split]["r2"]["lgb_tree"] for s in seeds],
        dtype=float,
    )
    if len(deltas) < 2:
        return {"n": int(len(deltas)), "mean_delta": round(float(deltas.mean()), 4)}
    test = stats.ttest_1samp(deltas, 0.0, alternative="greater")
    return {
        "n": int(len(deltas)),
        "mean_delta": round(float(deltas.mean()), 4),
        "std_delta": round(float(deltas.std(ddof=1)), 4),
        "t": round(float(test.statistic), 4),
        "p_one_sided": float(test.pvalue),
        "n_seeds_better": int((deltas > 0).sum()),
        "per_seed_delta": [round(float(d), 4) for d in deltas],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-dir", type=Path, default=PROJECT_ROOT / "results" / "msrs_seed_stability")
    parser.add_argument("--gnn-dir", type=Path, default=PROJECT_ROOT / "results" / "msrs_seed_stability")
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_seed_stability.json",
    )
    args = parser.parse_args(argv)

    per_seed: Dict[int, Dict[str, Dict[str, float]]] = {}
    missing: List[int] = []
    for seed in SEEDS:
        tree_path = args.tree_dir / f"tree_seed{seed}.npz"
        gnn_path = args.gnn_dir / f"gnn_seed{seed}.npz"
        if not tree_path.exists() or not gnn_path.exists():
            missing.append(seed)
            continue
        print(f"[met1] seed={seed}")
        per_seed[seed] = collect_seed(seed, args.tree_dir, args.gnn_dir)
        for split in SPLITS:
            row = per_seed[seed][split]["r2"]
            print(
                f"    {split:18s} tree={row['lgb_tree']:.4f} "
                f"equal={row['blend_equal']:.4f} stack={row['stack_foldwise']:.4f}"
            )
    if not per_seed:
        raise SystemExit("no complete seed pair found — run the two producers first")

    splits_out: Dict[str, object] = {}
    for split in SPLITS:
        arms_out: Dict[str, object] = {}
        for arm in ARMS:
            arms_out[arm] = {
                "r2": summarise([per_seed[s][split]["r2"][arm] for s in sorted(per_seed)]),
                "mae": summarise([per_seed[s][split]["mae"][arm] for s in sorted(per_seed)]),
                "rmse": summarise([per_seed[s][split]["rmse"][arm] for s in sorted(per_seed)]),
            }
            if arm != "lgb_tree":
                arms_out[arm]["vs_tree"] = paired_vs_tree(per_seed, split, arm)
        splits_out[split] = {
            "arms": arms_out,
            "weights_one_weight": {
                str(s): [round(w, 4) for w in per_seed[s][split]["weights_one_weight"]]
                for s in sorted(per_seed)
            },
            "residual_correlation": summarise(
                [per_seed[s][split]["residual_correlation"] for s in sorted(per_seed)]
            ),
        }

    canon = splits_out[SPLIT_CANONICAL_GROUPED]["arms"]
    tree_mean = canon["lgb_tree"]["r2"]["mean"]
    best_blend = max(
        ("blend_equal", "blend_foldwise", "stack_foldwise"),
        key=lambda a: canon[a]["r2"]["mean"],
    )
    best_mean = canon[best_blend]["r2"]["mean"]
    best_std = canon[best_blend]["r2"]["std"]
    better = canon[best_blend]["vs_tree"]["n_seeds_better"]
    required = max(1, int(np.ceil(0.8 * len(per_seed))))
    verdict = (
        "PASS"
        if (best_mean > tree_mean and best_std <= MAX_STD and better >= required)
        else "NEGATIVE"
    )

    payload = {
        "artifact": "msrs_seed_stability",
        "version": "1.0",
        "requires_dual_split": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "tree_dir": str(args.tree_dir),
            "gnn_dir": str(args.gnn_dir),
        },
        "design": {
            "seeds_requested": list(SEEDS),
            "seeds_present": sorted(per_seed),
            "seeds_missing": missing,
            "tree_arm": TREE_ARM,
            "gnn_arm": GNN_ARM,
            "arms": list(ARMS),
            "ligand_source": "ligand_canonical",
            "arm_ligand_source": {arm: "ligand_canonical" for arm in ARMS},
            "folds": "GroupKFold(5) on the ligand key (deterministic across seeds)",
            "statistic": "one-sided paired t over seeds, alternative='greater'",
            "max_std": MAX_STD,
            "verdict_rule": (
                "best blend mean > tree mean AND std <= 0.02 AND "
                ">= ceil(0.8 * seeds_present) seeds better (4/5 when all five are present)"
            ),
        },
        "splits": splits_out,
        "verdict": verdict,
        "best_blend": best_blend,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, payload)
    print(
        f"[met1] verdict={verdict} best={best_blend} "
        f"mean={best_mean:.4f} std={best_std:.4f} better={better}/{len(per_seed)}"
    )
    print(f"[met1] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
