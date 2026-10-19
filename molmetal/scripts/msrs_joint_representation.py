#!/usr/bin/env python3
"""FIX-L5 — do the three representation gains add up?

FIX-L1 (+0.1002), FIX-L2 (+0.0444) and FIX-L3 (+0.0272) were each measured as a
single change against the complex-ligand baseline.  Nothing so far says what
happens when all three are on at once: oxidation state and coordination number
are chemically coupled (CN is capped by the metal's capacity, which depends on
OS), so the increments may overlap.  This script measures the combination on the
same folds before any of it is called the new representation.

Arms:

  ligand            ECFP4(ligand_canonical)                  — FIX-L1
  ligand+metal      + OS / charge / d-electrons / valence    — FIX-L2 ``state``
  ligand+coord      + donor counts + geometry                — FIX-L3 ``coord_full``
  ligand+both       + both blocks                            — the combination

Acceptance: ``ligand+both`` >= best single block + 0.005 on the canonical split.
If not, the blocks are redundant and only the cheapest one should be kept.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_joint_representation.py
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import sparse

from molmetal.features.build import build
from molmetal.features.cell import cell_vocabulary, native_codes
from molmetal.features.coordination import coordination_block
from molmetal.features.metal import load_descriptors, state_block
from molmetal.features.spec import CellSpec, FeatureSpec, LigandSpec, MetalSpec
from molmetal.scripts.msrs_code_levers import ENDPOINT, fit_arm, folds_for
from molmetal.scripts.msrs_dual_split import (
    SPLIT_CANONICAL_GROUPED,
    SPLIT_FRAGMENT_GROUPED,
    write_report,
)
from molmetal.scripts.msrs_metal_block import paired_group_test
from molmetal.scripts.msrs_surface_model import (
    SEED,
    TOP_CELL_LINES,
    _git_commit,
    _sha256,
    score,
)

LIGAND_ONLY = FeatureSpec(
    ligand=LigandSpec(source="ligand_canonical", include=("canonical",)),
    metal=MetalSpec(one_hot=False, columns=()),
    cell=CellSpec(mode="none"),
)

MIN_EXTRA = 0.005
SPLITS = (SPLIT_FRAGMENT_GROUPED, SPLIT_CANONICAL_GROUPED)


def run_split(
    frame: pd.DataFrame,
    y: np.ndarray,
    ligand: sparse.csr_matrix,
    extras: Dict[str, sparse.csr_matrix],
    codes: np.ndarray,
    groups: np.ndarray,
    group_col: str,
    seed: int = SEED,
) -> tuple[Dict[str, object], Dict[str, np.ndarray]]:
    fold_id = folds_for(groups)
    empty = sparse.csr_matrix((len(frame), 0), dtype=np.float32)
    preds: Dict[str, np.ndarray] = {}
    out: Dict[str, object] = {}

    for arm, extra in extras.items():
        t0 = time.time()
        X = sparse.hstack([ligand, extra], format="csr")
        preds[arm] = fit_arm(
            X,
            empty,
            empty,
            None,
            0,
            False,
            y,
            fold_id,
            cat_codes=codes,
            params_override={"random_state": int(seed)},
        )
        entry = score(y, preds[arm])
        entry["n_features"] = int(X.shape[1])
        entry["n_extra_features"] = int(extra.shape[1])
        entry["wall_s"] = round(time.time() - t0, 1)
        out[arm] = entry
        print(f"    {arm:16s} R2={entry['r2']:.4f} MAE={entry['mae']:.4f}")

    singles = ("ligand+metal", "ligand+coord")
    best_single = max(singles, key=lambda a: out[a]["r2"])
    for arm in extras:
        if arm == "ligand":
            continue
        out[arm]["delta_r2_vs_ligand"] = round(
            float(out[arm]["r2"] - out["ligand"]["r2"]), 4
        )
        out[arm]["vs_ligand"] = paired_group_test(
            frame, np.abs(y - preds["ligand"]), np.abs(y - preds[arm]), group_col
        )
    out["ligand+both"]["delta_r2_vs_best_single"] = round(
        float(out["ligand+both"]["r2"] - out[best_single]["r2"]), 4
    )
    out["ligand+both"]["vs_best_single"] = paired_group_test(
        frame,
        np.abs(y - preds[best_single]),
        np.abs(y - preds["ligand+both"]),
        group_col,
    )
    out["_best_single"] = best_single
    return out, preds


def main(argv: Optional[Sequence[str]] = None) -> int:
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
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_joint_representation.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="smoke: first N ligands")
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="LightGBM random_state; the fold partition is deterministic",
    )
    parser.add_argument(
        "--oof-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "msrs_joint_representation" / "oof.npz",
        help="row-level out-of-fold predictions (needed by the hybrid model)",
    )
    args = parser.parse_args(argv)

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
    load_descriptors(args.metal_desc)  # fail fast if the table is missing
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

    metal_block_csr = sparse.csr_matrix(state)
    coord_block_csr = coord
    both = sparse.hstack([metal_block_csr, coord_block_csr], format="csr")
    extras = {
        "ligand": sparse.csr_matrix((len(sub), 0), dtype=np.float32),
        "ligand+metal": metal_block_csr,
        "ligand+coord": coord_block_csr,
        "ligand+both": both,
    }

    print(
        f"[joint] rows={len(sub)} ligands={sub['ligand_main_fragment'].nunique()} "
        f"complexes={sub['ligand_canonical'].nunique()} "
        f"donor_coverage={coverage['has_donor']:.4f} "
        f"spec={bundle.spec_hash[:19]}…"
    )

    splits: Dict[str, object] = {}
    oof: Dict[str, np.ndarray] = {}
    for split, col in (
        (SPLIT_FRAGMENT_GROUPED, "ligand_main_fragment"),
        (SPLIT_CANONICAL_GROUPED, "ligand_canonical"),
    ):
        print(f"[joint] split={split}")
        splits[split], preds = run_split(
            sub, y, bundle.X, extras, codes, sub[col].to_numpy(), col, seed=args.seed
        )
        for arm, values in preds.items():
            oof[f"{split}__{arm}"] = values

    canonical = splits[SPLIT_CANONICAL_GROUPED]
    best_single = canonical["_best_single"]
    extra = canonical["ligand+both"]["delta_r2_vs_best_single"]
    p_one = float(canonical["ligand+both"]["vs_best_single"]["p_one_sided"])
    verdict = "POSITIVE" if extra >= MIN_EXTRA else "NEGATIVE"

    for split in splits:
        splits[split].pop("_best_single", None)

    payload = {
        "artifact": "msrs_joint_representation",
        "version": "1.0",
        "requires_dual_split": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
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
            "min_extra": MIN_EXTRA,
            "arm_ligand_source": {arm: "ligand_canonical" for arm in extras},
        },
        "splits": splits,
        "best_single_block": best_single,
        "delta_r2_both_vs_best_single": round(float(extra), 4),
        "p_one_sided_vs_best_single": p_one,
        "verdict": verdict,
        "reading": (
            f"三块全开相对最佳单块 ({best_single}) {extra:+.4f} R2"
            f"（门槛 +{MIN_EXTRA:.3f}）。单块增量不可假设可加：OS 与配位数化学耦合。"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, payload)

    if args.oof_out:
        arrays = {
            "y": y.astype(np.float64),
            "metal": sub["metal"].to_numpy().astype(str),
            "cell_line_key": sub["cell_line_key"].to_numpy().astype(str),
            "doi": sub["doi"].to_numpy().astype(str),
            "time_h": sub["time_h"].to_numpy(),
            "ligand_canonical": sub["ligand_canonical"].to_numpy().astype(str),
            "ligand_main_fragment": sub["ligand_main_fragment"].to_numpy().astype(str),
        }
        arrays.update({k: v.astype(np.float64) for k, v in oof.items()})
        args.oof_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.oof_out, **arrays)
        print(f"[joint] wrote {args.oof_out}")
    print(f"[joint] verdict={verdict} extra={extra:+.4f}")
    print(f"[joint] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
