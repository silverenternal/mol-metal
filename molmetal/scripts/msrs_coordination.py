#!/usr/bin/env python3
"""FIX-L3 — does the model benefit from knowing what is coordinated?

The ligand fingerprint encodes connectivity, not binding.  A bidentate chelator
and two monodentate ligands of the same formula look similar to ECFP4 but behave
differently at the metal.  ``molmetal.features.coordination`` derives the ligand
side of that distinction (donor atoms, denticity proxy, estimated CN, implied
geometry); this script asks whether the model can use it.

Arms, all on the FIX-L1 complex representation with metal/cell categorical codes:

  ligand        ECFP4(ligand_canonical)                  — FIX-L1 baseline
  coord_counts  ligand | donor counts + est. CN           — no geometry
  coord_full    ligand | donor counts + geometry one-hot  — the full L3 block

Acceptance is +0.005 R2 over ``ligand`` on the canonical-grouped split with
donor coverage >= 90%.  Below that, record NEGATIVE and stop — AUD-F3 already
showed a +0.0024 metal-descriptor extension is not worth another round.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_coordination.py
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
from molmetal.features.spec import CellSpec, FeatureSpec, LigandSpec, MetalSpec
from molmetal.scripts.msrs_code_levers import ENDPOINT, fit_arm, folds_for
from molmetal.scripts.msrs_dual_split import (
    SPLIT_CANONICAL_GROUPED,
    SPLIT_FRAGMENT_GROUPED,
    write_report,
)
from molmetal.scripts.msrs_metal_block import paired_group_test
from molmetal.scripts.msrs_surface_model import (
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

MIN_LIFT = 0.005
MIN_COVERAGE = 0.90
SPLITS = (SPLIT_FRAGMENT_GROUPED, SPLIT_CANONICAL_GROUPED)


def run_split(
    frame: pd.DataFrame,
    y: np.ndarray,
    ligand: sparse.csr_matrix,
    blocks: Dict[str, Optional[sparse.csr_matrix]],
    codes: np.ndarray,
    groups: np.ndarray,
    group_col: str,
) -> Dict[str, object]:
    fold_id = folds_for(groups)
    empty = sparse.csr_matrix((len(frame), 0), dtype=np.float32)
    preds: Dict[str, np.ndarray] = {}
    out: Dict[str, object] = {}

    for arm, extra in blocks.items():
        t0 = time.time()
        X = ligand if extra is None else sparse.hstack([ligand, extra], format="csr")
        preds[arm] = fit_arm(
            X, empty, empty, None, 0, False, y, fold_id, cat_codes=codes
        )
        entry = score(y, preds[arm])
        entry["n_features"] = int(X.shape[1])
        entry["n_coord_features"] = 0 if extra is None else int(extra.shape[1])
        entry["wall_s"] = round(time.time() - t0, 1)
        out[arm] = entry
        print(f"    {arm:14s} R2={entry['r2']:.4f} MAE={entry['mae']:.4f}")

    for arm in blocks:
        if arm == "ligand":
            continue
        out[arm]["delta_r2_vs_ligand"] = round(
            float(out[arm]["r2"] - out["ligand"]["r2"]), 4
        )
        out[arm]["vs_ligand"] = paired_group_test(
            frame, np.abs(y - preds["ligand"]), np.abs(y - preds[arm]), group_col
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--long", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements.parquet"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_coordination.json",
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
        & long["oxidation_state"].notna()
    ].copy().reset_index(drop=True)
    if args.limit:
        keep = sub["ligand_main_fragment"].drop_duplicates().head(args.limit)
        sub = sub[sub["ligand_main_fragment"].isin(set(keep))].reset_index(drop=True)

    y = sub["value_log10"].to_numpy(dtype=float)
    vocab = cell_vocabulary(
        sub["cell_line_key"].astype(str).tolist(), top_n=TOP_CELL_LINES
    )
    codes = native_codes(
        sub["metal"].astype(str).tolist(),
        sub["cell_line_key"].astype(str).tolist(),
        vocab,
        mode="full",
    )

    lig_bundle = build(sub, LIGAND_ONLY)
    ligand = lig_bundle.X

    coord_full, coord_names, coverage = coordination_block(sub, include_geometry=True)
    counts_only, count_names, _ = coordination_block(sub, include_geometry=False)
    blocks: Dict[str, Optional[sparse.csr_matrix]] = {
        "ligand": None,
        "coord_counts": counts_only,
        "coord_full": coord_full,
    }

    print(
        f"[coord] rows={len(sub)} ligands={sub['ligand_main_fragment'].nunique()} "
        f"complexes={sub['ligand_canonical'].nunique()} "
        f"donor_coverage={coverage['has_donor']:.4f} "
        f"geometry_coverage={coverage['has_geometry']:.4f} "
        f"spec={lig_bundle.spec_hash[:19]}…"
    )

    splits: Dict[str, object] = {}
    for split, col in (
        (SPLIT_FRAGMENT_GROUPED, "ligand_main_fragment"),
        (SPLIT_CANONICAL_GROUPED, "ligand_canonical"),
    ):
        print(f"[coord] split={split}")
        splits[split] = run_split(
            sub, y, ligand, blocks, codes, sub[col].to_numpy(), col
        )

    canonical = splits[SPLIT_CANONICAL_GROUPED]
    best_arm = max(("coord_counts", "coord_full"), key=lambda a: canonical[a]["r2"])
    delta = canonical[best_arm]["r2"] - canonical["ligand"]["r2"]
    covered = coverage["has_donor"] >= MIN_COVERAGE
    verdict = "POSITIVE" if (delta >= MIN_LIFT and covered) else "NEGATIVE"

    payload = {
        "artifact": "msrs_coordination",
        "version": "1.0",
        "requires_dual_split": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "long_parquet": str(args.long),
            "long_sha256": _sha256(args.long),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "ligand_spec_hash": lig_bundle.spec_hash,
        },
        "design": {
            "rows": int(len(sub)),
            "ligands_main_fragment": int(sub["ligand_main_fragment"].nunique()),
            "complexes_canonical": int(sub["ligand_canonical"].nunique()),
            "ligand_source": "ligand_canonical",
            "ligand_block": "ECFP4(ligand_canonical), 2048 bits",
            "metal_cell_encoding": "LightGBM native categorical codes (2 columns)",
            "coord_columns_full": coord_names,
            "coord_columns_counts": count_names,
            "coverage": coverage,
            "min_lift": MIN_LIFT,
            "min_coverage": MIN_COVERAGE,
        },
        "splits": splits,
        "best_arm": best_arm,
        "delta_r2_canonical_vs_ligand": round(float(delta), 4),
        "verdict": verdict,
        "reading": (
            f"{best_arm} 在 canonical 分组下相对 FIX-L1 基线 {delta:+.4f} R2，"
            f"供体覆盖率 {coverage['has_donor']:.2%}、几何覆盖率 "
            f"{coverage['has_geometry']:.2%}。门槛 +{MIN_LIFT:.3f} 且覆盖 ≥"
            f"{MIN_COVERAGE:.0%}。"
            "供体检测是元素+电荷启发式，不区分吡啶 N 与吡咯 N。"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, payload)
    print(f"[coord] verdict={verdict} best={best_arm} delta={delta:+.4f}")
    print(f"[coord] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
