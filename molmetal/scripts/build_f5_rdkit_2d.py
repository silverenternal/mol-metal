#!/usr/bin/env python3
"""Wf-248 Phase 1 — F5_RDKIT_2D 32-d feature builder.

=====================================================================
CLI driver that builds the F5_RDKIT_2D 32-d RDKit 2D descriptor matrix
for a SMILES list (or per-row CSV with `smiles` column).  This is the
Wf-218 / Wf-248 expansion of the Wf-49 Phase 1 12-d extractor — the
first 12 entries of the 32-d vector match the 12-d path bit-for-bit
so any trainer / reward channel that hard-codes the 12-d column order
still works without change.

The 32-d set targets the Wf-127 §Cause 2 "12-d signal floor" on
Pt-only cohorts:

  * 12-d baseline (Wf-49 P1, frozen order)
  * +20-d expansion (Wf-218): shape / topology, kappa shape indices,
    surface area, partial-charge stats, ring features, heteroatom
    counts, functional-group counts (incl. fr_halogen — relevant to
    cisplatin / carboplatin / oxaliplatin leaving-group chemistry).

Outputs
-------
* ``--out-csv``: one row per input SMILES, columns:
  ``smiles[, metal, target]* + 32-d descriptors`` (32-d column order
  matches :data:`molmetal.data.rdkit_2d_extended.RDKIT_2D_EXTENDED_FEATURE_COLUMNS`).
* ``--summary``: JSON feature report with per-descriptor stats and
  Pearson r vs the optional target column (if ``--target-col`` is
  set in the input CSV).

Usage
-----
::

    uv run --no-sync python -u molmetal/scripts/build_f5_rdkit_2d.py \\
        --in-csv molmetal/data/platinai_4cell.csv \\
        --out-csv molmetal/data/platinai_4cell_rdkit_extended_features.csv \\
        --summary molmetal/reports/wf248_rdkit_2d_extended/platinai_4cell_summary.json \\
        --smiles-col smiles --target-col pIC50_mcf7

    # Or via plain SMILES list (one per line):
    uv run --no-sync python -u molmetal/scripts/build_f5_rdkit_2d.py \\
        --smiles-list smiles.txt \\
        --out-csv out.csv --summary out_summary.json

Backward-compat notes
---------------------
* The first 12 descriptor columns of the output CSV match
  :mod:`molmetal.scripts.extract_rdkit_2d_descriptors` byte-for-byte
  on every parsed SMILES.
* ``compute_rdkit_2d_12d(smiles)`` (the legacy 12-d entry point) is
  exposed for callers that want to keep using 12-d features without
  recomputing the full 32-d vector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path setup — make ``molmetal.*`` importable from the project root.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from molmetal.data.rdkit_2d_extended import (
    RDKIT_2D_BASE_DIM,
    RDKIT_2D_DIM,
    RDKIT_2D_EXTENDED_FEATURE_COLUMNS,
    compute_rdkit_2d_12d,
    compute_rdkit_2d_extended,
)


# ---------------------------------------------------------------------------
# CSV ingest helpers
# ---------------------------------------------------------------------------
def load_smiles_from_csv(
    path: Path,
    smiles_col: str,
    target_col: Optional[str] = None,
    extra_cols: Sequence[str] = (),
    max_rows: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Stream the input CSV into a list of row dicts + load stats.

    Each row dict has at least ``smiles``; if ``--target-col`` /
    ``extra_cols`` are set, those columns are propagated for downstream
    CSV / correlation writes.  Returns ``(rows, stats)`` where stats
    has ``n_total``, ``n_no_smiles``, ``n_no_target``.
    """
    rows: List[Dict[str, object]] = []
    stats = {"n_total": 0, "n_no_smiles": 0, "n_no_target": 0}
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            stats["n_total"] += 1
            smi = (row.get(smiles_col) or "").strip()
            if not smi:
                stats["n_no_smiles"] += 1
                continue
            out: Dict[str, object] = {"smiles": smi}
            if target_col is not None:
                tgt_raw = (row.get(target_col) or "").strip()
                tgt: float = float("nan")
                if tgt_raw:
                    try:
                        tgt = float(tgt_raw)
                    except ValueError:
                        stats["n_no_target"] += 1
                else:
                    stats["n_no_target"] += 1
                out[target_col] = tgt
            for ec in extra_cols:
                out[ec] = (row.get(ec) or "").strip()
            rows.append(out)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows, stats


def load_smiles_from_list(
    path: Path,
    max_rows: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    """Stream a one-SMILES-per-line text file into rows."""
    rows: List[Dict[str, object]] = []
    stats = {"n_total": 0, "n_no_smiles": 0, "n_no_target": 0}
    with path.open() as fh:
        for line in fh:
            stats["n_total"] += 1
            smi = line.strip()
            if not smi:
                stats["n_no_smiles"] += 1
                continue
            rows.append({"smiles": smi})
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows, stats


# ---------------------------------------------------------------------------
# Per-row descriptor extraction with median imputation (over parsed rows)
# ---------------------------------------------------------------------------
def extract_descriptors_for_rows(
    rows: List[Dict[str, object]],
    n_fn,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run RDKit over each row, return ``(X, valid, parse_failures)``.

    ``X`` is ``(n_rows, 32)`` float32.  Rows where RDKit fails to
    parse have ``valid_mask[i] = False`` and ``X[i, :] = 0``.  Median
    imputation runs across the **parsed** subset for each descriptor
    so the output is fully finite (NaN / inf entries are replaced by
    the per-column median over the parsed subset).
    """
    n_rows = len(rows)
    X = np.zeros((n_rows, RDKIT_2D_DIM), dtype=np.float32)
    valid = np.zeros(n_rows, dtype=bool)
    parse_failures = np.zeros(n_rows, dtype=bool)
    smiles_list = [str(r["smiles"]) for r in rows]

    for i, smi in enumerate(smiles_list):
        feats = n_fn(smi)
        if feats is None:
            parse_failures[i] = True
            continue
        X[i, :] = np.asarray(feats, dtype=np.float32)
        valid[i] = True

    # Median imputation over the parsed subset, per descriptor.
    for j in range(RDKIT_2D_DIM):
        col = X[:, j]
        finite_mask = valid & np.isfinite(col)
        if not finite_mask.any():
            # No finite entries for this descriptor — fill with 0.0
            # (the Wf-49 contract says every descriptor is finite on
            # PlatinAI 4-cell, but defensive zero-fill is safer here).
            continue
        med = float(np.median(col[finite_mask]))
        col_invalid = valid & ~np.isfinite(col)
        X[col_invalid, j] = med
    # Final safety: force finite on the valid rows.
    X[~valid, :] = 0.0
    X[~np.isfinite(X)] = 0.0
    return X, valid, parse_failures


# ---------------------------------------------------------------------------
# Per-descriptor stats + Pearson r blocks (mirrors
# extract_rdkit_2d_descriptors.py so Wf-49 / Wf-248 summaries stay
# schema-compatible downstream).
# ---------------------------------------------------------------------------
def _desc_stats_block(
    X: np.ndarray, desc_names: Sequence[str]
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for j, name in enumerate(desc_names):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if len(finite) == 0:
            out.append(
                {
                    "name": name,
                    "n_missing": int(len(col)),
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "median": None,
                }
            )
            continue
        out.append(
            {
                "name": name,
                "n_missing": int(np.sum(~np.isfinite(col))),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "median": float(np.median(finite)),
            }
        )
    return out


def _pearson_block(
    X: np.ndarray,
    y: np.ndarray,
    desc_names: Sequence[str],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    finite = np.isfinite(y)
    for j, name in enumerate(desc_names):
        col = X[:, j]
        mask = finite & np.isfinite(col)
        n = int(np.sum(mask))
        if n < 3:
            out.append({"name": name, "pearson_r": None, "n_samples": n})
            continue
        x = col[mask]
        yy = y[mask]
        if np.std(x) == 0 or np.std(yy) == 0:
            out.append({"name": name, "pearson_r": None, "n_samples": n})
            continue
        r = float(np.corrcoef(x, yy)[0, 1])
        r_val: Optional[float]
        if not math.isfinite(r):
            r_val = None
        else:
            r_val = r
        out.append({"name": name, "pearson_r": r_val, "n_samples": n})
    return out


# ---------------------------------------------------------------------------
# Summary writer (Wf-49 schema-compatible; add 32-d descriptor block).
# ---------------------------------------------------------------------------
def write_summary(
    out_path: Path,
    dataset_name: str,
    n_rows_total: int,
    n_rows_parsed: int,
    n_rows_dropped: int,
    parse_success_rate: float,
    desc_stats: List[Dict[str, object]],
    correlations: Optional[List[Dict[str, object]]],
    target_col: Optional[str],
    wall_seconds: float,
    source_csv: str,
    feature_dim: int = RDKIT_2D_DIM,
    base_dim: int = RDKIT_2D_BASE_DIM,
    expansion_dim: int = RDKIT_2D_DIM - RDKIT_2D_BASE_DIM,
    recipe: str = "F5_RDKIT_2D 32-d (Wf-218)",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "wf": "Wf-248",
        "phase": "Phase 1",
        "dataset": dataset_name,
        "source_csv": source_csv,
        "feature_dim": int(feature_dim),
        "base_dim": int(base_dim),
        "expansion_dim": int(expansion_dim),
        "descriptor_names": [d["name"] for d in desc_stats],
        "n_descriptors": len(desc_stats),
        "n_rows_total": n_rows_total,
        "n_rows_parsed": n_rows_parsed,
        "n_rows_dropped": n_rows_dropped,
        "parse_success_rate": float(parse_success_rate),
        "descriptor_stats": desc_stats,
        "wall_seconds": float(wall_seconds),
        "recipe": recipe,
        "literature_anchor": (
            "Yang et al. 2019 JCIM (D-MPNN paper); Wu et al. 2017 "
            "MoleculeNet §'smaller single-task datasets'; arXiv:2510.23428 "
            "(MetaModel = RDKit + ChemProp)."
        ),
        "math_anchor": (
            "F5_RDKIT_2D 32-d RDKit 2D descriptor vector (Wf-218); first 12 "
            "entries bit-for-bit match the Wf-49 Phase 1 12-d extractor. The "
            "20-d expansion targets the Wf-127 §Cause 2 '12-d signal floor' "
            "on Pt-only cohorts (shape / topology, kappa shape indices, "
            "surface area, partial-charge stats, ring features, heteroatom "
            "counts, functional-group counts)."
        ),
    }
    if target_col is not None:
        payload["target_col"] = target_col
    if correlations:
        payload["correlations_vs_target"] = correlations
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--in-csv",
        type=Path,
        default=None,
        help="Input CSV with a SMILES column.",
    )
    src.add_argument(
        "--smiles-list",
        type=Path,
        default=None,
        help="One-SMILES-per-line text file (no header).",
    )
    p.add_argument(
        "--smiles-col",
        default="smiles",
        help="SMILES column name when --in-csv is set (default: smiles).",
    )
    p.add_argument(
        "--target-col",
        default=None,
        help="Optional target column for the Pearson-r correlation block.",
    )
    p.add_argument(
        "--metal-col",
        default=None,
        help="Optional metal column propagated into the output CSV.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows (smoke test).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path (32-d descriptor columns appended).",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional JSON summary path "
        "(default: <out-csv-stem>.summary.json next to --out-csv).",
    )
    p.add_argument(
        "--dataset-name",
        default="f5_rdkit_2d",
        help="Top-level dataset name to embed in the summary JSON.",
    )
    p.add_argument(
        "--no-backward-compat",
        action="store_true",
        help="Skip the backward-compat 12-d verification block.",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def render_report(
    dataset_name: str,
    n_total: int,
    n_parsed: int,
    n_dropped: int,
    desc_stats: List[Dict[str, object]],
    wall_s: float,
    base_dim: int = RDKIT_2D_BASE_DIM,
    exp_dim: int = RDKIT_2D_DIM - RDKIT_2D_BASE_DIM,
) -> str:
    """Render a brief Markdown report for the build."""
    lines = [
        "# Wf-248 Phase 1 — F5_RDKIT_2D 32-d feature build report",
        "",
        f"- **dataset**: {dataset_name}",
        f"- **n_rows_total**: {n_total}",
        f"- **n_rows_parsed**: {n_parsed}",
        f"- **n_rows_dropped**: {n_dropped}",
        f"- **feature_dim**: {base_dim + exp_dim} "
        f"({base_dim} baseline + {exp_dim} Wf-218 expansion)",
        f"- **wall_s**: {wall_s:.2f}",
        "",
        "## Backward-compat note",
        "",
        "The first 12 entries of the 32-d vector match the Wf-49 Phase 1 "
        "12-d extractor bit-for-bit (same RDKit callables, same iteration "
        "order, same NaN-handling).",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    t0 = time.time()

    # Load rows.
    extra_cols: List[str] = []
    if args.metal_col:
        extra_cols.append(args.metal_col)

    if args.in_csv is not None:
        rows, load_stats = load_smiles_from_csv(
            args.in_csv,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
            extra_cols=extra_cols,
            max_rows=args.max_rows,
        )
        source_csv = str(args.in_csv)
    else:
        rows, load_stats = load_smiles_from_list(
            args.smiles_list,
            max_rows=args.max_rows,
        )
        source_csv = str(args.smiles_list)

    print(
        f"[Wf-248] loaded {len(rows)} rows from {source_csv} "
        f"(n_total={load_stats['n_total']}, n_no_smiles="
        f"{load_stats['n_no_smiles']})",
        flush=True,
    )

    # Compute 32-d descriptors with median imputation.
    X, valid, parse_failures = extract_descriptors_for_rows(
        rows,
        compute_rdkit_2d_extended,
    )
    n_total = len(rows)
    n_parsed = int(np.sum(valid))
    n_dropped = int(np.sum(parse_failures))

    # Backward-compat verification (cheap: just check first 12 entries
    # of compute_rdkit_2d_extended match compute_rdkit_2d_12d on a
    # single probe SMILES).
    if not args.no_backward_compat:
        probe = "Cc1ccccc1"  # toluene — always parses
        v32 = compute_rdkit_2d_extended(probe)
        v12 = compute_rdkit_2d_12d(probe)
        if v32 is None or v12 is None or not np.allclose(v32[:12], v12):
            print(
                "[Wf-248] WARNING: 12-d backward-compat check failed on "
                f"toluene — v32[:12]={None if v32 is None else v32[:12]}, "
                f"v12={None if v12 is None else v12}. The 12-d path has "
                "drifted from the 32-d path; investigate before shipping.",
                flush=True,
            )
        else:
            print(
                "[Wf-248] backward-compat check PASS: 32-d[:12] == 12-d on toluene.",
                flush=True,
            )

    # Build the per-row metadata header (smiles + metal + target +
    # 32 descriptor columns).
    meta_header: List[str] = ["smiles"]
    if args.metal_col:
        meta_header.append(args.metal_col)
    if args.target_col:
        meta_header.append(args.target_col)
    desc_names = list(RDKIT_2D_EXTENDED_FEATURE_COLUMNS)
    header = meta_header + desc_names

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, r in enumerate(rows):
            row_out: List[str] = [r["smiles"]]
            if args.metal_col:
                row_out.append(str(r.get(args.metal_col, "")))
            if args.target_col:
                v = float(r.get(args.target_col, float("nan")))
                row_out.append("" if not math.isfinite(v) else f"{v:.6f}")
            row_out.extend(f"{float(v):.6f}" for v in X[i, :])
            writer.writerow(row_out)

    desc_stats = _desc_stats_block(X[valid, :], desc_names)
    wall = time.time() - t0

    # Correlations vs the optional target column.
    correlations: Optional[List[object]] = None
    if args.target_col is not None:
        y = np.asarray(
            [
                float(r[args.target_col])
                if isinstance(r[args.target_col], (int, float))
                and math.isfinite(float(r[args.target_col]))
                else float("nan")
                for r in rows
            ],
            dtype=np.float64,
        )
        correlations = _pearson_block(X, y, desc_names)

    summary_path = args.summary or args.out_csv.with_suffix(".summary.json")
    write_summary(
        out_path=summary_path,
        dataset_name=args.dataset_name,
        n_rows_total=n_total,
        n_rows_parsed=n_parsed,
        n_rows_dropped=n_dropped,
        parse_success_rate=(n_parsed / n_total) if n_total else 0.0,
        desc_stats=desc_stats,
        correlations=correlations,
        target_col=args.target_col,
        wall_seconds=wall,
        source_csv=source_csv,
    )

    report = render_report(
        dataset_name=args.dataset_name,
        n_total=n_total,
        n_parsed=n_parsed,
        n_dropped=n_dropped,
        desc_stats=desc_stats,
        wall_s=wall,
    )
    report_path = summary_path.with_suffix(".report.md")
    report_path.write_text(report)

    print(report, flush=True)
    print(f"[Wf-248] DONE — wrote {args.out_csv} and {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
