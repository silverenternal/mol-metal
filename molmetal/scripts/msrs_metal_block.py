#!/usr/bin/env python3
"""FIX-L2 — is the metal block too small to matter?

The complex-ligand arm gives the model 2,048 ligand bits.  The metal block is
8 numbers: a five-way one-hot and three tabulated descriptors (IE1, IE2,
Pauling EN).  A 1:256 imbalance, and the oxidation state — the one variable that
changes within a metal and drives d-electron count, LFSE and redox potential —
was never in the matrix at all, even though the cohort carries it for 100% of
rows and ``metal_graph`` already builds the vector.

Four arms, all on the FIX-L1 complex representation, all with the same
metal/cell native categorical codes:

  codes_only   ligand | metal_code | cell_code          — the FIX-L1 baseline
  base         codes_only | one-hot(5) + IE1/IE2/EN     — the historical block
  state        base | OS + charge + d-electrons + valence
  rich         base | RDKit periodic/size block (row, group, weight, radii, …)
  full         base | state | rich

Acceptance is +0.005 R2 over ``base`` on the canonical-grouped split.  The
threshold is deliberate: AUD-F3 already measured a metal-descriptor extension at
+0.0024 and it was recorded NEGATIVE, so a repeat of that magnitude is not a
result — it is the same null.

Geometry is *not* in this arm.  ``classify_coord_geometry`` needs a coordination
number, which the cohort does not carry; estimating it from the ligand is FIX-L3.

Usage::

    OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/msrs_metal_block.py
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import sparse, stats

from molmetal.data.metal_graph import rich_metal_block
from molmetal.features.build import build
from molmetal.features.cell import cell_vocabulary, native_codes
from molmetal.features.metal import load_descriptors, metal_block, state_block
from molmetal.features.spec import CellSpec, FeatureSpec, LigandSpec, MetalSpec
from molmetal.scripts.msrs_code_levers import ENDPOINT, fit_arm, folds_for
from molmetal.scripts.msrs_dual_split import (
    SPLIT_CANONICAL_GROUPED,
    SPLIT_FRAGMENT_GROUPED,
    write_report,
)
from molmetal.scripts.msrs_surface_model import (
    TOP_CELL_LINES,
    _git_commit,
    _sha256,
    score,
)

#: Ligand-only spec: metal and cell are supplied as codes, so the builder must
#: not add its own one-hot blocks (that would double-encode the metal).
LIGAND_ONLY = FeatureSpec(
    ligand=LigandSpec(source="ligand_canonical", include=("canonical",)),
    metal=MetalSpec(one_hot=False, columns=()),
    cell=CellSpec(mode="none"),
)

#: arm -> dense metal block pieces appended after the ligand block
ARMS: Dict[str, Tuple[str, ...]] = {
    "codes_only": (),
    "base": ("onehot", "chem"),
    "state": ("onehot", "chem", "state"),
    "rich": ("onehot", "chem", "rich"),
    "full": ("onehot", "chem", "state", "rich"),
}

MIN_LIFT = 0.005
SPLITS = (SPLIT_FRAGMENT_GROUPED, SPLIT_CANONICAL_GROUPED)


def _rich_block(metals: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    rows = [rich_metal_block(m) for m in metals]
    names = [
        "metal_row",
        "metal_group",
        "metal_weight",
        "metal_n_outer",
        "metal_rcov",
        "metal_rvdw",
        "metal_valence",
    ]
    return np.vstack(rows).astype(np.float32), names


def build_metal_arm(
    frame: pd.DataFrame, arm: str, descriptors: pd.DataFrame
) -> Tuple[Optional[np.ndarray], List[str]]:
    pieces: List[np.ndarray] = []
    names: List[str] = []
    for key in ARMS[arm]:
        if key == "onehot":
            block, block_names = metal_block(
                frame["metal"].astype(str).tolist(), descriptors, columns=()
            )
            pieces.append(block)
            names.extend(block_names)
        elif key == "chem":
            block, block_names = metal_block(
                frame["metal"].astype(str).tolist(), descriptors, one_hot=False
            )
            pieces.append(block)
            names.extend(block_names)
        elif key == "state":
            block, block_names = state_block(frame)
            pieces.append(block)
            names.extend(block_names)
        elif key == "rich":
            block, block_names = _rich_block(frame["metal"].astype(str).tolist())
            pieces.append(block)
            names.extend(block_names)
        else:  # pragma: no cover - guarded by ARMS
            raise KeyError(key)
    if not pieces:
        return None, []
    return np.hstack(pieces).astype(np.float32), names


def paired_group_test(
    frame: pd.DataFrame,
    err_a: np.ndarray,
    err_b: np.ndarray,
    group_col: str,
) -> Dict[str, object]:
    """Paired one-sided t-test on per-group mean absolute error (b better)."""
    table = pd.DataFrame(
        {"g": frame[group_col].to_numpy(), "a": err_a, "b": err_b}
    )
    per = table.groupby("g")[["a", "b"]].mean()
    diff = per["a"] - per["b"]
    if float(diff.std(ddof=1)) == 0.0:
        t, p_one = 0.0, 1.0
    else:
        t, p_two = stats.ttest_rel(per["a"], per["b"])
        p_one = float(p_two / 2.0) if t > 0 else float(1.0 - p_two / 2.0)
    return {
        "n_groups": int(len(per)),
        "mean_mae_a": round(float(per["a"].mean()), 4),
        "mean_mae_b": round(float(per["b"].mean()), 4),
        "mean_delta": round(float(diff.mean()), 4),
        "t": round(float(t), 4),
        "p_one_sided": p_one,
        "direction": "b better" if diff.mean() > 0 else "a better",
    }


def run_split(
    frame: pd.DataFrame,
    y: np.ndarray,
    ligand: sparse.csr_matrix,
    dense_blocks: Dict[str, Optional[np.ndarray]],
    codes: np.ndarray,
    groups: np.ndarray,
    group_col: str,
) -> Dict[str, object]:
    fold_id = folds_for(groups)
    empty = sparse.csr_matrix((len(frame), 0), dtype=np.float32)
    preds: Dict[str, np.ndarray] = {}
    out: Dict[str, object] = {}

    for arm in ARMS:
        t0 = time.time()
        dense = dense_blocks[arm]
        X = ligand if dense is None else sparse.hstack(
            [ligand, sparse.csr_matrix(dense)], format="csr"
        )
        preds[arm] = fit_arm(
            X, empty, empty, None, 0, False, y, fold_id, cat_codes=codes
        )
        entry = score(y, preds[arm])
        entry["n_features"] = int(X.shape[1])
        entry["n_dense_metal"] = 0 if dense is None else int(dense.shape[1])
        entry["wall_s"] = round(time.time() - t0, 1)
        out[arm] = entry
        print(f"    {arm:12s} R2={entry['r2']:.4f} MAE={entry['mae']:.4f}")

    for arm in ARMS:
        if arm == "base":
            continue
        out[arm]["delta_r2_vs_base"] = round(
            float(out[arm]["r2"] - out["base"]["r2"]), 4
        )
        out[arm]["vs_base"] = paired_group_test(
            frame, np.abs(y - preds["base"]), np.abs(y - preds[arm]), group_col
        )
    return out


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
        default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_metal_block.json",
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
    descriptors = load_descriptors(args.metal_desc)
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
    built = {arm: build_metal_arm(sub, arm, descriptors) for arm in ARMS}
    dense_blocks = {arm: built[arm][0] for arm in ARMS}
    widths = {arm: built[arm][1] for arm in ARMS}

    print(
        f"[metal] rows={len(sub)} ligands={sub['ligand_main_fragment'].nunique()} "
        f"complexes={sub['ligand_canonical'].nunique()} "
        f"spec={lig_bundle.spec_hash[:19]}…"
    )

    splits: Dict[str, object] = {}
    for split, col in (
        (SPLIT_FRAGMENT_GROUPED, "ligand_main_fragment"),
        (SPLIT_CANONICAL_GROUPED, "ligand_canonical"),
    ):
        print(f"[metal] split={split}")
        splits[split] = run_split(
            sub, y, ligand, dense_blocks, codes, sub[col].to_numpy(), col
        )

    canonical = splits[SPLIT_CANONICAL_GROUPED]
    best_arm = max(
        ("state", "rich", "full"), key=lambda a: canonical[a]["r2"]
    )
    delta = canonical[best_arm]["r2"] - canonical["base"]["r2"]
    verdict = "POSITIVE" if delta >= MIN_LIFT else "NEGATIVE"

    payload = {
        "artifact": "msrs_metal_block",
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
            "arms": {arm: list(ARMS[arm]) for arm in ARMS},
            "arm_dense_columns": {arm: widths[arm] for arm in ARMS},
            "coverage": {
                "oxidation_state": round(float(sub["oxidation_state"].notna().mean()), 4),
                "charge_complex": round(float(sub["charge_complex"].notna().mean()), 4),
                "valence_e": round(float(sub["valence_e"].notna().mean()), 4),
            },
            "min_lift": MIN_LIFT,
            "threshold_source": (
                "AUD-F3 measured +0.0024 for a metal-descriptor extension and was "
                "recorded NEGATIVE; repeating that magnitude is not a result."
            ),
        },
        "splits": splits,
        "best_arm": best_arm,
        "delta_r2_canonical_vs_base": round(float(delta), 4),
        "verdict": verdict,
        "reading": (
            f"最佳金属块 {best_arm} 在 canonical 分组下相对历史 base 块 "
            f"{delta:+.4f} R2（门槛 +{MIN_LIFT:.3f}）。"
            "geometry 不在本臂内：classify_coord_geometry 需要配位数，队列里没有，"
            "留给 FIX-L3 从配体供体原子估计。"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.out, payload)
    print(f"[metal] verdict={verdict} best={best_arm} delta={delta:+.4f}")
    print(f"[metal] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
