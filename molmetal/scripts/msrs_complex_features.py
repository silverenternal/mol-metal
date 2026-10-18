#!/usr/bin/env python3
"""FIX-L1 — complex-level ligand blocks, same folds, same params.

The audit found the model featurized ``ligand_main_fragment`` even though
99.93% of the cohort rows are multi-component complexes.  This script measures
the four ways of using the full complex against that baseline:

  main              ECFP4(main fragment)                      — the old design
  complex           ECFP4(all components)                     — one line change
  main+ancillary    ECFP4(main) | ECFP4(everything else)      — explicit split
  full              complex | main | ancillary | counterion | n_components

Every arm is scored on both splits, because the audit showed the same arm moves
0.0606 between fragment-grouped and complex-grouped CV.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_complex_features.py
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import sparse

from molmetal.features.ligand import complex_ligand_blocks
from molmetal.scripts.msrs_dual_split import (
    SPLIT_CANONICAL_GROUPED,
    SPLIT_FRAGMENT_GROUPED,
    write_report,
)
from molmetal.scripts.msrs_code_levers import (
    ENDPOINT,
    LGB_PARAMS,
    fit_arm,
    folds_for,
    native_codes,
)
from molmetal.scripts.msrs_surface_model import (
    TOP_CELL_LINES,
    _git_commit,
    _sha256,
    paired_ligand_test,
    score,
)

#: arm -> (block names, human description)
ARMS: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "main": (("main",), "ECFP4(ligand_main_fragment) — the pre-audit design"),
    "complex": (("canonical",), "ECFP4(ligand_canonical) — all components pooled"),
    "main_plus_ancillary": (("main", "ancillary"), "ECFP4(main) | ECFP4(ancillary)"),
    "full": (
        ("canonical", "main", "ancillary", "counterion", "n_fragments"),
        "complex | main | ancillary | counterion one-hot | n_components",
    ),
}

SPLITS = (SPLIT_FRAGMENT_GROUPED, SPLIT_CANONICAL_GROUPED)


def build_ligand_blocks(
    frame: pd.DataFrame, counterion_vocabulary: Sequence[str]
) -> Dict[str, sparse.csr_matrix]:
    """Every arm's ligand block, built once from the same frame."""
    out: Dict[str, sparse.csr_matrix] = {}
    for arm, (include, _desc) in ARMS.items():
        X, _names, slices = complex_ligand_blocks(
            frame,
            counterion_vocabulary=counterion_vocabulary,
            include=include,
        )
        out[arm] = X
    return out


def run_split(
    frame: pd.DataFrame,
    y: np.ndarray,
    blocks: Dict[str, sparse.csr_matrix],
    codes: np.ndarray,
    groups: np.ndarray,
) -> Dict[str, object]:
    """Fit every arm on one fold assignment and score it.

    Metal and cell line enter only through ``codes`` (LightGBM native
    categorical); the one-hot ``met``/``cell`` matrices the old call passed were
    never read on this path.  They are gone rather than passed as dead weight.
    """
    fold_id = folds_for(groups)
    empty = sparse.csr_matrix((len(frame), 0), dtype=np.float32)
    preds: Dict[str, np.ndarray] = {}
    results: Dict[str, object] = {}
    for arm in ARMS:
        t0 = time.time()
        preds[arm] = fit_arm(
            blocks[arm], empty, empty, None, 0, False, y, fold_id, cat_codes=codes
        )
        entry = score(y, preds[arm])
        entry["n_features"] = int(blocks[arm].shape[1])
        entry["wall_s"] = round(time.time() - t0, 1)
        results[arm] = entry
        print(f"    {arm:20s} R2={entry['r2']:.4f} MAE={entry['mae']:.4f}")

    for arm in ("complex", "main_plus_ancillary", "full"):
        test = paired_ligand_test(frame, np.abs(y - preds["main"]), np.abs(y - preds[arm]))
        results[arm]["delta_r2_vs_main"] = round(
            float(results[arm]["r2"] - results["main"]["r2"]), 4
        )
        results[arm]["vs_main"] = test
    return results


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--metal-desc",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "data" / "metal_chemistry_5.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_complex_features.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="smoke: first N ligands")
    args = parser.parse_args(argv)

    long = pd.read_parquet(args.long)
    sub = long[
        (long["endpoint"] == ENDPOINT)
        & long["value_log10"].notna()
        & (long["censoring"].astype(str) == "none")
        & long["ligand_main_fragment"].notna()
        & long["ligand_canonical"].notna()
    ].copy().reset_index(drop=True)
    if args.limit:
        keep = sub["ligand_main_fragment"].drop_duplicates().head(args.limit)
        sub = sub[sub["ligand_main_fragment"].isin(set(keep))].reset_index(drop=True)

    y = sub["value_log10"].to_numpy(dtype=float)
    cell_vocab = sub["cell_line_key"].value_counts().head(TOP_CELL_LINES).index.tolist()
    codes = native_codes(sub, cell_vocab, "full")

    counterions = sub["counterion"].fillna("").astype(str)
    vocab = counterions.value_counts().head(32).index.tolist()
    blocks = build_ligand_blocks(sub, vocab)
    print(
        f"[complex] rows={len(sub)} ligands={sub['ligand_main_fragment'].nunique()} "
        f"complexes={sub['ligand_canonical'].nunique()} counterion_levels={len(vocab) + 1}"
    )

    splits: Dict[str, object] = {}
    for split in SPLITS:
        groups = (
            sub["ligand_main_fragment"].to_numpy()
            if split == SPLIT_FRAGMENT_GROUPED
            else sub["ligand_canonical"].to_numpy()
        )
        print(f"[complex] split={split}")
        splits[split] = run_split(sub, y, blocks, codes, groups)

    base = splits[SPLIT_FRAGMENT_GROUPED]["main"]["r2"]
    best_arm = max(
        ("complex", "main_plus_ancillary", "full"),
        key=lambda a: splits[SPLIT_CANONICAL_GROUPED][a]["r2"],
    )
    delta_canonical = (
        splits[SPLIT_CANONICAL_GROUPED][best_arm]["r2"]
        - splits[SPLIT_CANONICAL_GROUPED]["main"]["r2"]
    )
    payload = {
        "artifact": "msrs_complex_features",
        "version": "1.1",
        "requires_dual_split": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
        },
        "design": {
            "rows": int(len(sub)),
            "ligands_main_fragment": int(sub["ligand_main_fragment"].nunique()),
            "complexes_canonical": int(sub["ligand_canonical"].nunique()),
            "folds": "GroupKFold(5); both fragment_grouped and canonical_grouped reported",
            "params": {k: v for k, v in LGB_PARAMS.items() if k != "objective"},
            "arms": {arm: desc for arm, (_inc, desc) in ARMS.items()},
            "arm_ligand_source": {
                arm: (
                    "ligand_main_fragment" if arm == "main" else "ligand_canonical"
                )
                for arm in ARMS
            },
            "counterion_vocabulary_size": len(vocab) + 1,
        },
        "splits": splits,
        "best_arm_canonical_split": best_arm,
        "delta_r2_canonical_split": round(float(delta_canonical), 4),
        "verdict": "POSITIVE" if delta_canonical >= 0.05 else "NEGATIVE",
        "reading": (
            f"完整配合物块在 canonical 分组下相对 main 基线 {delta_canonical:+.4f} R2。"
            f"fragment 分组下 main={base:.4f}。"
            "两个分组都必须报：同一臂换分组会差 0.06。"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, payload)
    print(f"[complex] verdict={payload['verdict']} best={best_arm} delta={delta_canonical:+.4f}")
    print(f"[complex] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
