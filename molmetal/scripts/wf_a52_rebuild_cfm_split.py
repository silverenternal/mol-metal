"""Rebuild the CFM 5-metal corpus with a deduplicated, ligand-scaffold split.

Why this exists
---------------
``metalcytotox_5metal_multitask_v3/v4.csv`` ship a ``split`` column whose
``scaffold_id`` is the Bemis-Murcko scaffold of the **whole complex**, with
stereochemistry retained.  MEASURED in ``docs/DATA_LEAKAGE_AUDIT.md``: under a
desalted, stereo-blind **ligand** scaffold key, 73.7 % of test rows and 79.6 %
of val rows share a scaffold with train.  The file also carries 2,149
redundant ``(SMILES, metal, cell_line)`` rows.

This script writes a new corpus (default ``v5``) that:

1. de-duplicates to one row per ``(canonical SMILES, metal, cell_line)``
   (keep-first, matching ``build_paper_cohort``'s measurement-level dedup);
2. computes the split key as ``_bemis_murcko_key`` — desalted to the largest
   fragment, ``includeChirality=False`` — with an ``__empty__<canonical>``
   sentinel so acyclic molecules are grouped per molecule rather than into one
   bucket;
3. assigns whole scaffold groups to train/val/test at 70/10/20 by row count,
   largest group first (deterministic, no RNG; the largest chemical series
   stays in train);
4. verifies 0 shared scaffolds and 0 shared canonical SMILES across every pair
   of splits, and that all five metals appear in all three splits.

The output keeps the v3 column layout, replaces ``scaffold_id`` with the
ligand key, and appends ``complex_scaffold_id`` (the old full-complex value)
for traceability.  ``v3`` and ``v4`` are never modified.

Usage
-----
    /home/hugo/codes/mol-metal/.venv/bin/python \
        molmetal/scripts/wf_a52_rebuild_cfm_split.py

    # explicit paths
    ... --input .data/molmetal/metalcytotox_5metal_multitask_v3.csv \
        --output .data/molmetal/metalcytotox_5metal_multitask_v5.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.data.leakage_utils import canonicalize_smiles  # noqa: E402
from molmetal.data.splits import _bemis_murcko_key  # noqa: E402

DEFAULT_INPUT = (
    _REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v3.csv"
)
DEFAULT_OUTPUT = (
    _REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v5.csv"
)
DEFAULT_MANIFEST = (
    _REPO_ROOT
    / ".data"
    / "molmetal"
    / "metalcytotox_5metal_multitask_v5_split_manifest.json"
)

FRACTIONS: Tuple[float, float, float] = (0.70, 0.10, 0.20)
METALS = ("Ru", "Ir", "Rh", "Os", "Re")
SPLITS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ligand_scaffold(smiles: str) -> str:
    """Desalted, stereo-blind Bemis-Murcko key; empty -> per-molecule."""
    key = _bemis_murcko_key(smiles)
    return key or f"__empty__{canonicalize_smiles(smiles)}"


def _assign_groups(
    df: pd.DataFrame,
    key_col: str,
    fractions: Tuple[float, float, float] = FRACTIONS,
) -> Dict[str, str]:
    """Whole-group assignment, largest group first, greedy capacity fill."""
    n = len(df)
    targets = [int(fractions[0] * n), int(fractions[1] * n)]
    targets.append(n - targets[0] - targets[1])
    counts = [0, 0, 0]

    groups = df.groupby(key_col, sort=False).indices
    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    assignment: Dict[str, str] = {}
    for key, _idx in order:
        gaps = [targets[i] - counts[i] for i in range(3)]
        split_idx = int(np.argmax(gaps))
        assignment[key] = SPLITS[split_idx]
        counts[split_idx] += len(_idx)
    return assignment


def _pairwise_overlap(df: pd.DataFrame, col: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        sa = set(df.loc[df["split"] == a, col])
        sb = set(df.loc[df["split"] == b, col])
        out[f"{a}_vs_{b}"] = len(sa & sb)
    return out


def rebuild(input_path: Path, output_path: Path, manifest_path: Path) -> dict:
    raw = pd.read_csv(input_path, low_memory=False)
    n_input = int(len(raw))

    work = raw.copy()
    work["canon"] = work["SMILES_Ligands"].map(canonicalize_smiles)
    work["lig_scaf"] = work["SMILES_Ligands"].map(_ligand_scaffold)

    before = int(len(work))
    work = work.drop_duplicates(subset=["canon", "Metal", "cell_line"], keep="first")
    work = work.reset_index(drop=True)
    n_dropped_dupes = before - len(work)

    assignment = _assign_groups(work, "lig_scaf")
    work["split"] = work["lig_scaf"].map(assignment)

    # scaffold_id becomes the split key; the old value is retained.
    work["complex_scaffold_id"] = work["scaffold_id"]
    work["scaffold_id"] = work["lig_scaf"]

    cols: List[str] = [
        "SMILES_Ligands",
        "Metal",
        "scaffold_id",
        "split",
        "activity",
        "pic50",
        "cell_line",
        "source",
        "IC50_Dark_value",
        "complex_scaffold_id",
    ]
    out = work[cols].copy()

    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, output_path)

    checks = {
        "scaffold_overlap": _pairwise_overlap(out, "scaffold_id"),
        "canonical_overlap": _pairwise_overlap(work, "canon"),
        "metal_by_split": {
            s: {m: int(((out["split"] == s) & (out["Metal"] == m)).sum()) for m in METALS}
            for s in SPLITS
        },
        "split_rows": {s: int((out["split"] == s).sum()) for s in SPLITS},
        "split_unique_molecules": {
            s: int(work.loc[work["split"] == s, "canon"].nunique()) for s in SPLITS
        },
        "split_unique_scaffolds": {
            s: int(out.loc[out["split"] == s, "scaffold_id"].nunique()) for s in SPLITS
        },
        "null_counts": {c: int(out[c].isna().sum()) for c in out.columns},
    }

    failed: List[str] = []
    for pair, n in checks["scaffold_overlap"].items():
        if n:
            failed.append(f"scaffold overlap {pair}={n}")
    for pair, n in checks["canonical_overlap"].items():
        if n:
            failed.append(f"canonical overlap {pair}={n}")
    for s in SPLITS:
        if checks["split_rows"][s] == 0:
            failed.append(f"empty split {s}")
        missing = [m for m, n in checks["metal_by_split"][s].items() if n == 0]
        if missing:
            failed.append(f"split {s} missing metals {missing}")
    for c, n in checks["null_counts"].items():
        if n:
            failed.append(f"nulls in {c}: {n}")

    manifest = {
        "workflow": "Wf-A52",
        "purpose": "Deduplicated, ligand-scaffold-disjoint CFM corpus",
        "input": {
            "path": str(input_path),
            "sha256": _sha256(input_path),
            "rows": n_input,
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "rows": int(len(out)),
        },
        "split_key": {
            "definition": "_bemis_murcko_key (desalted to largest fragment, "
            "includeChirality=False), empty -> '__empty__<canonical SMILES>'",
            "column": "scaffold_id",
            "old_column_retained_as": "complex_scaffold_id",
        },
        "dedupe": {
            "key": ["canonical SMILES", "Metal", "cell_line"],
            "keep": "first",
            "dropped_rows": n_dropped_dupes,
        },
        "assignment": {
            "fractions": list(FRACTIONS),
            "strategy": "largest group first, greedy capacity fill",
            "seed": "not used (deterministic)",
            "n_groups": int(out["scaffold_id"].nunique()),
        },
        "checks": checks,
        "failed": failed,
        "status": "PASS" if not failed else "FAIL",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"FAIL input missing: {args.input}")
        return 1

    m = rebuild(args.input, args.output, args.manifest)
    print(f"input : {m['input']['path']} ({m['input']['rows']:,} rows)")
    print(f"output: {m['output']['path']} ({m['output']['rows']:,} rows)")
    print(f"dedupe: dropped {m['dedupe']['dropped_rows']:,} rows")
    print(f"split : {m['checks']['split_rows']}")
    print(f"metals: {json.dumps(m['checks']['metal_by_split'])}")
    print(f"scaffold overlap: {m['checks']['scaffold_overlap']}")
    print(f"canonical overlap: {m['checks']['canonical_overlap']}")
    print(f"manifest: {args.manifest}")
    if m["status"] != "PASS":
        print("FAIL " + "; ".join(m["failed"]))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
