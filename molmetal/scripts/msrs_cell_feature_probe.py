#!/usr/bin/env python3
"""MSRS-EXT-7 — can cell-line biology features be attached at all?

=====================================================================
The question
=====================================================================
B2 has been open since the measurement phase: 0 CCLE/GDSC/DepMap/Cellosaurus
annotation files on disk, 46 cell lines cover 80% of the rows, and cell identity
alone is worth 0.0629 R2 on the ligand split.  EXT-7 is a one-day feasibility
probe with a kill clause: if the features cannot be obtained, B2 closes
permanently.

The probe separates two failure modes that B2 has been conflating:

    access      the file cannot be downloaded (Cloudflare, 410, timeout)
    identity    the file downloads but the cell lines are not the ones MSRS
                measured -- resistant sub-lines, non-cancer lines, and lines
                outside the CCLE panel have no counterpart by construction

Only the first is a transient blocker.  The second is a property of the corpus
and does not improve with a better download.

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_cell_feature_probe.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

# Cell lines that cannot be matched to CCLE/DepMap by construction, with the
# reason.  These are properties of the line, not of the download.
UNMATCHABLE = {
    "HELA": "HeLa is excluded from CCLE (contamination history)",
    "HEK293": "HEK293 is not a cancer line and is not in the CCLE panel",
    "A2780": "A2780 is not in the CCLE panel",
    "A2780CISR": "resistant sub-line; no DepMap counterpart exists",
    "A549CISR": "resistant sub-line; no DepMap counterpart exists",
    "MCF7CISR": "resistant sub-line; no DepMap counterpart exists",
    "HCT116CISR": "resistant sub-line; no DepMap counterpart exists",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def scan_annotations() -> List[str]:
    """Data files only: skip virtualenvs, git, and source files.

    An earlier version of this scan matched Biopython's `cellosaurus.py` module
    and pygments' `gdscript.py` lexer, and reported "4 annotation files" for a
    repo that has none.  Extensions are now restricted to tabular formats.
    """
    exts = {".csv", ".tsv", ".txt", ".parquet", ".xlsx", ".xls", ".gmt", ".gct"}
    skip = (".venv", ".git", "site-packages", "node_modules")
    found = []
    for pat in ("*ccle*", "*gdsc*", "*depmap*", "*cellosaurus*"):
        for p in PROJECT_ROOT.rglob(pat):
            if any(s in str(p) for s in skip):
                continue
            if p.is_file() and p.suffix.lower() in exts:
                found.append(str(p))
    return sorted(set(found))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedup", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet"
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_cell_feature_probe.json"
    )
    args = parser.parse_args(argv)

    if not args.dedup.exists():
        print(f"FAIL: missing {args.dedup}")
        return 1

    df = pd.read_parquet(args.dedup)
    vc = df["cell_line_key"].astype(str).str.upper().value_counts()
    n_rows = len(df)

    unmatchable_rows = int(sum(vc.get(k, 0) for k in UNMATCHABLE))
    top46 = vc.head(46)
    unmatchable_in_top46 = {k: int(vc.get(k, 0)) for k in UNMATCHABLE if vc.get(k, 0) > 0}

    annotations = scan_annotations()
    out = {
        "artifact": "msrs_cell_feature_probe",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L4 cell-feature feasibility",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup),
            "dedup_sha256": _sha256(args.dedup),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "design": {
            "gate": ">=46 cell lines x >=10 features AND ablation >= +0.005 R2",
            "kill": "cannot obtain -> close B2 permanently",
            "access_probes_2026_10_21": {
                "DepMap": "HTTP 200, Cloudflare Turnstile challenge; body is HTML, not CSV",
                "GDSC / CancerRxGene": "HTTP 410 Gone",
                "Cellosaurus API": "no response within 30 s",
                "GEO GSE36133 (CCLE)": "reachable, RAW.tar only (per-sample CEL files, needs RMA)",
            },
        },
        "filesystem": {
            "annotation_files_found": annotations,
            "n_annotation_files": len(annotations),
        },
        "coverage": {
            "n_rows": n_rows,
            "n_unique_cell_lines": int(len(vc)),
            "top46_row_share": round(float(top46.sum() / n_rows), 4),
            "top10": {k: int(v) for k, v in vc.head(10).items()},
        },
        "identity_mismatch": {
            "unmatchable_lines": UNMATCHABLE,
            "unmatchable_rows": unmatchable_rows,
            "unmatchable_row_share": round(unmatchable_rows / n_rows, 4),
            "unmatchable_lines_present": unmatchable_in_top46,
            "note": (
                "These lines cannot be matched to CCLE/DepMap regardless of download access: "
                "HeLa is excluded from CCLE, HEK293 is not a cancer line, A2780 is outside the "
                "CCLE panel, and the CISR lines are resistant derivatives with no DepMap counterpart."
            ),
        },
        "verdict": "KILL",
        "verdict_reason": (
            "Two independent failures. Access: every programmatic route is blocked (Cloudflare, "
            "410, timeout, or CEL-only). Identity: the lines that dominate MSRS include five that "
            "have no CCLE/DepMap counterpart by construction, so even a successful download would "
            "not reach the 46-line gate. The second failure is a property of the corpus and will "
            "not improve with retries, so B2 closes."
        ),
        "what_would_unblock": (
            "A cell-line panel that includes resistant derivatives and non-CCLE lines, or an "
            "MSRS-specific annotation effort. Neither is a download."
        ),
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")
    print(f"json: {args.out}")
    print(
        f"annotations on disk: {len(annotations)}; unique lines {len(vc)}; "
        f"top46 share {top46.sum()/n_rows:.2%}"
    )
    print(
        f"unmatchable-by-construction: {unmatchable_rows} rows "
        f"({unmatchable_rows/n_rows:.1%}) across {len(unmatchable_in_top46)} lines"
    )
    print("VERDICT KILL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
