"""Train symbolic-reward formula on 147 MEASURED cells.

Stage 3 of the wf_deflex_symbolic_regression workflow.

This script loads the 63 deterministic MEASURED cells (effective 33 unique
after cohort-collapse, see Phase 1 inventory §3.3), defines a composite
reward target, fits 8 lit-grounded formula families (F1..F8), selects the
Pareto-optimal family by (R^2, complexity), and dumps the best formula as
both a pickle (Python `FitResult`) and a JSON metadata blob.

Lit anchors:
    Cranmer 2023 (PySR, arXiv:2305.01582)        — F2 + complexity-Pareto
    Udrescu 2020 (AI-Feynman, arXiv:1905.11481) — F1, F2 separability
    Sun 2022 (Symbolic Physics Learner,          — known-terms prior
              arXiv:2205.14212)
    Tennie 2024 (hierarchical SR)                 — F8 decision tree
    Dayan 1997 (potential-based shaping)          — F3, F7
    McAllester 1999 (PAC-Bayes)                   — F5
    Schulman 2017 (PPO clipped)                   — F6
    Auger 2013 (DPW)                             — F7

Honest framing (Phase 1 inventory §7):
    PySR unavailable → algebraic enumeration instead of evolutionary search.
    33-cell effective sample is proof-of-concept only.
    Composite reward is heuristic (NOT a held-out oracle); R^2 measures
    in-sample fit quality, NOT downstream lift.

Usage:
    uv run python molmetal/scripts/train_symbolic_regression.py \
        --output-dir molmetal/models/symbolic_reward
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Project root: .../try_triton_on_rocm
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.reward.symbolic_regression import (  # noqa: E402
    FEATURE_COLUMNS,
    FORMULA_FAMILIES,
    FORMULA_FAMILIES as _FFA,  # alias for clarity
    FAMILY_FITTERS,
    FitResult,
    build_reward_target,
    fit_symbolic_reward,
    load_measured_cells,
    loo_cv,
    mask_complete_cases,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 3 symbolic-regression train (Deflex workflow).")
    parser.add_argument("--output-dir",
                        default="molmetal/models/symbolic_reward",
                        help="Where to write the formula pickle + JSON.")
    parser.add_argument("--family", default="ALL",
                        choices=list(FORMULA_FAMILIES) + ["ALL"],
                        help="Formula family to fit (default: ALL → Pareto pick).")
    parser.add_argument("--n-iterations", type=int, default=100,
                        help="Reserved for PySR evolutionary search (unused).")
    parser.add_argument("--report", default=None,
                        help="Optional path for a markdown verdict report.")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_symbolic_regression] output_dir = {output_dir}")

    # 1) Load cells.
    df = load_measured_cells()
    print(f"[train_symbolic_regression] loaded {len(df)} MEASURED cells")
    print(f"[train_symbolic_regression] cohorts = "
          f"{df['cohort'].value_counts().to_dict()}")

    # 2) Per-family subset (drop NaNs in family's required feature columns).
    print(f"[train_symbolic_regression] feature columns ({len(FEATURE_COLUMNS)}): "
          f"{FEATURE_COLUMNS}")

    # 3) Fit each family and collect results.
    all_results: Dict[str, Any] = {}
    for fam, meta in FORMULA_FAMILIES.items():
        cols = meta["features"]
        df_sub, y_sub = mask_complete_cases(df, feature_cols=cols)
        if len(df_sub) < 5:
            print(f"  [{fam}] skipping: only {len(df_sub)} complete-case cells "
                  f"(need ≥5)")
            all_results[fam] = {
                "status": "insufficient_data",
                "n_fit": int(len(df_sub)),
                "lit_anchor": meta["lit"],
                "math_prior": meta["math_prior"],
                "complexity": meta["complexity"],
            }
            continue
        result = fit_symbolic_reward(df_sub, y_sub, family=fam,
                                     n_iterations=args.n_iterations)
        cv = loo_cv(df_sub, y_sub, family=fam)
        all_results[fam] = {
            "status": result.status,
            "n_fit": result.n_fit,
            "r2_in_sample": result.r2,
            "r2_loo": cv["loo_r2"],
            "pearson_loo": cv["loo_pearson_r"],
            "complexity": result.complexity,
            "params": result.params,
            "formula": result.formula,
            "lit_anchor": meta["lit"],
            "math_prior": meta["math_prior"],
        }
        print(f"  [{fam}] n={result.n_fit:>3}  R^2={result.r2:.4f}  "
              f"R^2_LOO={cv['loo_r2']:.4f}  "
              f"Pearson_LOO={cv['loo_pearson_r']:.4f}  "
              f"complexity={result.complexity}  "
              f"lit={meta['lit'][:40]}...")

    # 4) Pick Pareto-optimal family (highest R^2, then lowest complexity).
    valid = {k: v for k, v in all_results.items()
             if v.get("status") == "ok"}
    if valid:
        ranked = sorted(valid.items(),
                        key=lambda kv: (-kv[1]["r2_in_sample"],
                                        kv[1]["complexity"]))
        best_family, best_meta = ranked[0]
        print(f"\n[train_symbolic_regression] BEST family = {best_family}  "
              f"R^2 = {best_meta['r2_in_sample']:.4f}  "
              f"complexity = {best_meta['complexity']}")
    else:
        best_family, best_meta = None, None
        print("[train_symbolic_regression] WARNING: no family fit successfully")

    # 5) Save JSON metadata + pickle.
    meta_blob = {
        "workflow": "wf_deflex_symbolic_regression",
        "stage": 3,
        "n_cells_total": int(len(df)),
        "n_cells_per_cohort": df["cohort"].value_counts().to_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "reward_target": "(1 - SA/10) + diversity_tanimoto + metal_compliance",
        "pysr_available": False,
        "pysr_fallback": "algebraic enumeration of 8 lit-grounded families",
        "all_results": all_results,
        "best_family": best_family,
        "best_result": best_meta,
        "lit_anchors": {
            "Cranmer_2023": "arXiv:2305.01582 (PySR)",
            "Udrescu_2020": "arXiv:1905.11481 (AI-Feynman)",
            "Sun_2022": "arXiv:2205.14212 (Symbolic Physics Learner)",
            "Tennie_2024": "hierarchical symbolic regression (placeholder)",
        },
    }
    json_path = output_dir / "symbolic_reward.json"
    with open(json_path, "w") as f:
        json.dump(meta_blob, f, indent=2, default=str)
    print(f"[train_symbolic_regression] wrote {json_path}")

    if best_family is not None:
        # Re-fit best family to obtain a FitResult for the pickle.
        cols = FORMULA_FAMILIES[best_family]["features"]
        df_sub, y_sub = mask_complete_cases(df, feature_cols=cols)
        best_result = fit_symbolic_reward(df_sub, y_sub, family=best_family,
                                          n_iterations=args.n_iterations)
        pkl_path = output_dir / "symbolic_reward.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "FitResult": best_result,
                "feature_columns": cols,
                "all_results": all_results,
            }, f)
        print(f"[train_symbolic_regression] wrote {pkl_path}")

        # Honest verdict write-up.
        verdict_lines = [
            f"# Symbolic Regression Train — Best Family: {best_family}",
            "",
            f"- **R^2 (in-sample)**: {best_meta['r2_in_sample']:.4f}",
            f"- **R^2 (LOO CV)**: {best_meta['r2_loo']:.4f}",
            f"- **Pearson r (LOO)**: {best_meta['pearson_loo']:.4f}",
            f"- **n_fit**: {best_meta['n_fit']}",
            f"- **complexity**: {best_meta['complexity']}",
            f"- **Lit anchor**: {best_meta['lit_anchor']}",
            f"- **Math prior**: {best_meta['math_prior']}",
            "",
            f"**Formula**:",
            "",
            "```",
            best_meta["formula"],
            "```",
            "",
        ]
        if args.report:
            rpt_path = Path(args.report)
            rpt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(rpt_path, "w") as f:
                f.write("\n".join(verdict_lines))
            print(f"[train_symbolic_regression] wrote {rpt_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
