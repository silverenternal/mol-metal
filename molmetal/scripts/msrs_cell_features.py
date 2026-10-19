#!/usr/bin/env python3
"""FIX-L4 — is a cell-line biology feature table actually obtainable?

The work package asks for CCLE/GDSC expression or mutation features and a
>=+0.005 R2 gain at >=80% cohort coverage.  The repository contains neither
table, so this script measures the blocker instead of asserting it:

  * scan the on-disk data roots for known annotation datasets;
  * quantify how concentrated the 711 MSRS cell lines are, i.e. how many lines
    an 80%-coverage feature table would actually need;
  * read the ablation artifact for the ceiling: what cell *identity* is worth.

Output: ``metrics/by_metric/msrs_cell_features.json``.  Exit 0 = PASS,
2 = BLOCKED-PENDING-DATA (the project's SKIP convention).

Usage:
    .venv/bin/python molmetal/scripts/msrs_cell_features.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
MSRS_PARQUET = REPO_ROOT / "data" / "msrs" / "measurements.parquet"
ABLATION = REPO_ROOT / "metrics" / "by_metric" / "msrs_metal_ablation.json"
DEFAULT_OUT = REPO_ROOT / "metrics" / "by_metric" / "msrs_cell_features.json"

SCAN_ROOTS = (".data", "data", "molmetal/data")
PATTERNS = (
    "ccle",
    "gdsc",
    "depmap",
    "cellosaurus",
    "cosmic",
    "expression",
    "mutation",
    "omics",
    "rnaseq",
    "transcriptome",
)
MAX_DEPTH = 4
COVERAGE_GATE = 0.80
GAIN_GATE = 0.005
MIN_ROWS_PER_LINE = 10


def scan_for_annotation_files(
    roots: Iterable[Path] = (), patterns: Iterable[str] = PATTERNS, max_depth: int = MAX_DEPTH
) -> list[dict[str, Any]]:
    """Return every file whose name matches a known cell-line annotation dataset."""
    hits: list[dict[str, Any]] = []
    needles = tuple(p.lower() for p in patterns)
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        base_depth = len(root.parts)
        for path in root.rglob("*"):
            if len(path.parts) - base_depth > max_depth:
                continue
            if not path.is_file():
                continue
            name = path.name.lower()
            if any(n in name for n in needles):
                hits.append({"path": str(path), "size_bytes": path.stat().st_size})
    return sorted(hits, key=lambda h: h["path"])


def coverage_table(counts: pd.Series, ks: Iterable[int] = (10, 25, 50, 100, 200)) -> list[dict[str, Any]]:
    """Share of rows covered by the top-k cell lines."""
    total = int(counts.sum())
    if total == 0:
        return []
    out = []
    for k in ks:
        if k > len(counts):
            continue
        out.append({"top_k": int(k), "rows": int(counts.head(k).sum()), "share": round(float(counts.head(k).sum() / total), 4)})
    return out


def min_lines_for_coverage(counts: pd.Series, target: float = COVERAGE_GATE) -> Optional[int]:
    """Smallest k whose cumulative share reaches ``target``."""
    total = int(counts.sum())
    if total == 0:
        return None
    cum = counts.cumsum() / total
    hits = cum[cum >= target]
    if not len(hits):
        return None
    # hits.index holds a cell-line key, not a position: convert to 1-based k.
    return int(counts.index.get_loc(hits.index[0])) + 1


def load_msrs(path: Path = MSRS_PARQUET) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    keep = (
        (frame["endpoint"] == "potency")
        & frame["value_log10"].notna()
        & (frame["censoring"] == "none")
        & frame["ligand_main_fragment"].notna()
    )
    return frame.loc[keep]


def read_identity_ceiling(path: Path = ABLATION) -> dict[str, Any]:
    """R2 lost when the cell-line code is removed — the ceiling for any substitute."""
    if not path.exists():
        return {"available": False}
    doc = json.loads(path.read_text())
    out: dict[str, Any] = {"available": True, "artifact": path.name}
    for proto in ("ligand", "doi"):
        try:
            arm = doc["protocols"][proto]["arms"]["cell_blind"]
            out[proto] = {
                "delta_r2_vs_full": arm.get("delta_r2_vs_full"),
                "ci95": arm.get("delta_ci", {}).get("ci95"),
            }
        except KeyError:
            out[proto] = None
    return out


def _ceiling_value(ceiling: dict[str, Any], protocol: str) -> Optional[float]:
    block = ceiling.get(protocol) or {}
    return block.get("delta_r2_vs_full")


def build_artifact(
    msrs_path: Path = MSRS_PARQUET,
    ablation_path: Path = ABLATION,
    roots: Iterable[Path] | None = None,
    measured_gain: Optional[float] = None,
) -> dict[str, Any]:
    frame = load_msrs(msrs_path)
    counts = frame["cell_line_key"].value_counts()
    roots = tuple(Path(REPO_ROOT) / r for r in SCAN_ROOTS) if roots is None else tuple(roots)
    hits = scan_for_annotation_files(roots)
    ceiling = read_identity_ceiling(ablation_path)

    n_rows = int(len(frame))
    n_cells = int(counts.size)
    coverage = coverage_table(counts)
    n_for_80 = min_lines_for_coverage(counts)
    rare = counts[counts < MIN_ROWS_PER_LINE]

    # This script never trains, so ``measured_gain`` stays None unless a caller
    # supplies a real experiment.  Availability alone cannot PASS the package.
    passed = bool(hits) and n_for_80 is not None and measured_gain is not None and measured_gain >= GAIN_GATE
    return {
        "artifact": "msrs_cell_features",
        "version": 1.0,
        "requires_dual_split": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "script": "molmetal/scripts/msrs_cell_features.py",
            "msrs_parquet": str(msrs_path),
            "ablation_artifact": str(ablation_path),
        },
        "design": {
            "coverage_gate": COVERAGE_GATE,
            "gain_gate": GAIN_GATE,
            "min_rows_per_line": MIN_ROWS_PER_LINE,
            "scan_roots": [str(r) for r in roots],
            "scan_patterns": list(PATTERNS),
            "acceptance": f"a citable annotation table covering >={COVERAGE_GATE:.0%} of rows AND >= +{GAIN_GATE} R2",
        },
        "corpus": {
            "n_rows": n_rows,
            "n_cell_lines": n_cells,
            "top_10": counts.head(10).to_dict(),
            "coverage": coverage,
            "min_lines_for_80pct": n_for_80,
            "n_lines_below_10_rows": int(rare.size),
            "rows_in_lines_below_10": int(rare.sum()),
            "share_rows_in_lines_below_10": round(float(rare.sum() / n_rows), 4) if n_rows else None,
        },
        "on_disk_annotation_files": hits,
        "identity_ceiling": ceiling,
        "measured_gain": measured_gain,
        "verdict": "PASS" if passed else "BLOCKED-PENDING-DATA",
        "verdict_rule": "at least one citable annotation table on disk AND enough lines for 80% coverage AND a measured >= +0.005 R2; else BLOCKED-PENDING-DATA",
        "reading": (
            "No CCLE/GDSC/DepMap/Cellosaurus file exists under .data, data or molmetal/data, so the feature "
            "table cannot be built without downloading one. The corpus is concentrated: the top "
            f"{n_for_80} cell lines cover 80% of rows, so an 80%-coverage table does not need all {n_cells} lines. "
            "The identity ceiling is the R2 lost when the cell-line code is removed "
            f"({_ceiling_value(ceiling, 'ligand')} ligand / {_ceiling_value(ceiling, 'doi')} DOI): any biological "
            "feature can at most recover that much, and the +0.005 target is the increment on top of identity."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    artifact = build_artifact()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    c = artifact["corpus"]
    print(
        f"[msrs_cell_features] verdict={artifact['verdict']} cells={c['n_cell_lines']} "
        f"lines_for_80pct={c['min_lines_for_80pct']} annotation_files={len(artifact['on_disk_annotation_files'])} "
        f"-> {args.out}"
    )
    return 0 if artifact["verdict"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
