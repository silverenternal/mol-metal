#!/usr/bin/env python3
"""AUD-F2 — rebuild the 172-d feature stack on every unique MSRS ligand.

The shipped artifact ``features_172d_5metal.npz`` holds 1,000 rows because it
was built with ``--max-rows``.  The builder itself is deterministic, and the
MSRS potency cohort has only 2,846 unique ``ligand_main_fragment`` values, so a
full rebuild costs minutes of CPU.  This script:

1. loads the same exact-potency cohort the code-lever A/B uses,
2. computes ``[RDKit-12 | Multi-PH-32 | PLL-128]`` once per unique fragment,
3. writes ``molmetal/data/features_172d_full.npz`` + a coverage report,
4. runs the acceptance A/B: does the block add ``+0.005`` R2 over AUD-F1?

Failed streams are left as **NaN**, not zero.  ``compute_172d`` zero-fills a
failed stream, which is indistinguishable from a real zero to a tree model;
LightGBM's native NaN handling is the honest representation of "not computed".

Usage::

    .venv/bin/python molmetal/scripts/msrs_172d_rebuild.py
    .venv/bin/python molmetal/scripts/msrs_172d_rebuild.py --skip-lift
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
for _p in (str(_HERE), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from scipy import sparse

from molmetal.scripts.build_172d_features import (
    MULTI_PH_DIM,
    PLL_DIM,
    RDKIT_DIM as RDKIT12_DIM,
    compute_multiph_32d,
    compute_pll_128d,
    compute_rdkit_12d,
)

ENDPOINT = "potency"
FEATURE_DIM = RDKIT12_DIM + MULTI_PH_DIM + PLL_DIM  # 172
MIN_COVERAGE = 0.95
MIN_LIFT = 0.005


# ---------------------------------------------------------------------------
# Cohort + feature build
# ---------------------------------------------------------------------------
def potency_cohort(long_path: Path) -> pd.DataFrame:
    """The exact-measurement potency cohort used by every MSRS A/B."""
    long = pd.read_parquet(long_path)
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
    ].copy()
    return sub.reset_index(drop=True)


def fragment_features(smiles: str) -> Tuple[np.ndarray, Dict[str, bool]]:
    """One 172-d row with NaN on any stream that failed for this molecule."""
    vec = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
    ok = {"rdkit": False, "multiph": False, "pll": False}

    rd = compute_rdkit_12d(smiles)
    if rd is not None and len(rd) == RDKIT12_DIM:
        vec[:RDKIT12_DIM] = np.asarray(rd, dtype=np.float32)
        ok["rdkit"] = True

    mph = compute_multiph_32d(smiles)
    if mph is not None and mph.size == MULTI_PH_DIM:
        vec[RDKIT12_DIM : RDKIT12_DIM + MULTI_PH_DIM] = mph
        ok["multiph"] = True

    pll = compute_pll_128d(smiles)
    if pll is not None and pll.size == PLL_DIM:
        vec[RDKIT12_DIM + MULTI_PH_DIM :] = pll
        ok["pll"] = True

    return vec, ok


def build_table(fragments: Sequence[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Feature matrix + per-stream boolean masks, one row per fragment."""
    n = len(fragments)
    X = np.full((n, FEATURE_DIM), np.nan, dtype=np.float32)
    masks = {k: np.zeros(n, dtype=bool) for k in ("rdkit", "multiph", "pll")}
    for i, smi in enumerate(fragments):
        vec, ok = fragment_features(smi)
        X[i] = vec
        for k, v in ok.items():
            masks[k][i] = v
    return X, masks


def coverage_report(
    fragments: Sequence[str], X: np.ndarray, masks: Dict[str, np.ndarray], wall_s: float
) -> Dict[str, object]:
    """Coverage in the two units that matter: ligands and cohort rows."""
    complete = ~np.isnan(X).any(axis=1)
    per_stream = {k: round(float(v.mean()), 4) for k, v in masks.items()}
    return {
        "n_unique_ligands": int(len(fragments)),
        "n_complete_172d": int(complete.sum()),
        "coverage_complete": round(float(complete.mean()), 4),
        "coverage_per_stream": per_stream,
        "coverage_rdkit_only": per_stream["rdkit"],
        "n_duplicate_fragments": int(len(fragments) - len(set(fragments))),
        "wall_s": round(float(wall_s), 2),
        "imputation": "none — failed streams stay NaN for LightGBM's native handler",
    }


def write_npz(
    out_path: Path,
    fragments: Sequence[str],
    X: np.ndarray,
    masks: Dict[str, np.ndarray],
    report: Dict[str, object],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        fragments=np.asarray(list(fragments), dtype=object),
        rdkit_ok=masks["rdkit"],
        multiph_ok=masks["multiph"],
        pll_ok=masks["pll"],
    )
    with open(out_path.with_suffix(".summary.json"), "w") as fh:
        json.dump(report, fh, indent=2)


# ---------------------------------------------------------------------------
# Acceptance A/B — does the block add +0.005 R2 over AUD-F1?
# ---------------------------------------------------------------------------
def lift_test(
    sub: pd.DataFrame,
    fragments: Sequence[str],
    X172: np.ndarray,
    *,
    out_path: Path,
) -> Dict[str, object]:
    """Same folds, same LightGBM params, one block different.

    ``cat_cell_full`` is the AUD-F1 best full design (ECFP4 + native categorical
    metal/cell).  Two arms add the block: all 172 dims, and only the 160 dims
    that are not already represented by the AUD-F1 RDKit-32 stream.
    """
    from molmetal.scripts.msrs_code_levers import (
        LGB_PARAMS,
        base_blocks,
        fit_arm,
        folds_for,
        native_codes,
    )
    from molmetal.scripts.msrs_surface_model import (
        TOP_CELL_LINES,
        paired_ligand_test,
        score,
    )

    index = {f: i for i, f in enumerate(fragments)}
    # ``--limit`` builds a subset; drop the rows it cannot cover so the A/B stays
    # a like-for-like comparison rather than silently mismatching indices.
    sub = sub[sub["ligand_main_fragment"].isin(index)].reset_index(drop=True)
    row_idx = np.array([index[f] for f in sub["ligand_main_fragment"]], dtype=int)
    block_all = X172[row_idx]
    block_new = block_all[:, RDKIT12_DIM:]

    descriptors = pd.read_csv(_PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv")
    cell_vocab = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    lig, met, cell = base_blocks(sub, descriptors, cell_vocab)
    cat_codes = native_codes(sub, cell_vocab, "full")
    fold_id = folds_for(sub["ligand_main_fragment"].to_numpy())
    y = sub["value_log10"].to_numpy(dtype=float)

    arms: Dict[str, Optional[np.ndarray]] = {
        "cat_cell_full": None,
        "f2_172d": block_all,
        "f2_mph_pll": block_new,
    }
    preds: Dict[str, np.ndarray] = {}
    results: Dict[str, object] = {}
    for name, block in arms.items():
        t0 = time.time()
        pred = fit_arm(
            lig, met, cell, block, 0, False, y, fold_id, cat_codes=cat_codes
        )
        preds[name] = pred
        entry = score(y, pred)
        entry["wall_s"] = round(time.time() - t0, 1)
        results[name] = entry
        print(f"[lift] {name:<14} R2={entry['r2']:.4f} MAE={entry['mae']:.4f}")

    baseline = results["cat_cell_full"]["r2"]
    for name in ("f2_172d", "f2_mph_pll"):
        err_a = np.abs(y - preds["cat_cell_full"])
        err_b = np.abs(y - preds[name])
        test = paired_ligand_test(sub, err_a, err_b)
        results[name]["delta_r2"] = round(float(results[name]["r2"] - baseline), 4)
        results[name]["vs_cat_cell_full"] = test
        print(
            f"[lift] {name:<14} delta={results[name]['delta_r2']:+.4f} "
            f"paired_p={test['p_one_sided']:.3g}"
        )

    best_name = max(("f2_172d", "f2_mph_pll"), key=lambda k: results[k]["r2"])
    best_delta = float(results[best_name]["delta_r2"])
    payload = {
        "artifact": str(out_path),
        "protocol": {
            "rows": int(len(sub)),
            "ligands": int(sub["ligand_main_fragment"].nunique()),
            "folds": "GroupKFold(5) by ligand_main_fragment",
            "params": {k: v for k, v in LGB_PARAMS.items() if k != "objective"},
            "baseline": "cat_cell_full (AUD-F1 best full design, R2 0.3366 in msrs_code_levers.json)",
            "arms": {
                "f2_172d": "cat_cell_full + [RDKit12 | MultiPH32 | PLL128]",
                "f2_mph_pll": "cat_cell_full + [MultiPH32 | PLL128] (drops the 12 dims AUD-F1 already covers)",
            },
        },
        "arms": results,
        "best_arm": best_name,
        "best_delta_r2": round(best_delta, 4),
        "threshold": MIN_LIFT,
        "verdict": "POSITIVE" if best_delta >= MIN_LIFT else "NEGATIVE",
        "reading": (
            f"172d 块在 AUD-F1 之上只带来 {best_delta:+.4f} R2"
            f"（门槛 +{MIN_LIFT}）。"
            + (
                "通过验收。"
                if best_delta >= MIN_LIFT
                else "未通过验收：这三个流与 ECFP4 的信息重叠大于互补。"
            )
        ),
    }
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--long",
        type=Path,
        default=_PROJECT_ROOT / "data" / "msrs" / "measurements.parquet",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_PROJECT_ROOT / "molmetal" / "data" / "features_172d_full.npz",
    )
    p.add_argument(
        "--metrics",
        type=Path,
        default=_PROJECT_ROOT / "metrics" / "by_metric" / "msrs_172d_full.json",
    )
    p.add_argument("--skip-lift", action="store_true", help="build the artifact only")
    p.add_argument("--limit", type=int, default=0, help="smoke: first N fragments")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.long.exists():
        print(f"FAIL: missing {args.long}")
        return 1

    sub = potency_cohort(args.long)
    fragments: List[str] = sub["ligand_main_fragment"].drop_duplicates().tolist()
    if args.limit:
        fragments = fragments[: args.limit]
    print(f"[f2] rows={len(sub)} unique_ligands={len(fragments)}")

    t0 = time.time()
    X, masks = build_table(fragments)
    wall = time.time() - t0
    report = coverage_report(fragments, X, masks, wall)
    report["cohort_rows"] = int(len(sub))
    report["cohort_rows_covered"] = int(
        sub["ligand_main_fragment"].isin(
            {f for f, ok in zip(fragments, ~np.isnan(X).any(axis=1)) if ok}
        ).sum()
    )
    write_npz(args.out, fragments, X, masks, report)
    print(
        f"[f2] coverage complete={report['coverage_complete']:.4f} "
        f"streams={report['coverage_per_stream']} wall={wall:.1f}s"
    )
    print(f"[f2] wrote {args.out}")

    payload: Dict[str, object] = {"coverage": report}
    if not args.skip_lift:
        lift = lift_test(sub, fragments, X, out_path=args.metrics)
        payload["lift"] = lift
        payload["verdict"] = lift["verdict"]
        print(f"[f2] verdict={lift['verdict']} best_delta={lift['best_delta_r2']:+.4f}")
    else:
        payload["verdict"] = (
            "POSITIVE"
            if report["coverage_complete"] >= MIN_COVERAGE
            else "COVERAGE_FAIL"
        )

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[f2] wrote {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
