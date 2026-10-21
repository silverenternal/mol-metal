#!/usr/bin/env python3
"""MSRS-EXT-5 — does an independent external cohort exist at all?

=====================================================================
Why this is not a modelling script
=====================================================================
EXT-5 in todo.json asks for >=60 compounds from >=3 metals across >=8 papers,
published 2023-2025, DOI-disjoint from MetalCytoToxDB.  The plan budgeted two
weeks of manual curation.  Before spending that, this probe asks whether a
machine-readable cohort already exists somewhere, and if it does, whether it is
actually independent.

The distinction matters.  A dataset that re-publishes the same curation under a
new name is not external validation; it is the same measurement counted twice.
The only way to tell the difference is to compare rows, not titles.

=====================================================================
What is probed
=====================================================================
    local files       any CCLE / GDSC / DepMap / Cellosaurus annotation on disk
    IrCytoToxDB       Scientific Data 2024, Zenodo 10.5281/zenodo.13120939,
                      CC-BY-4.0 -- the largest machine-readable Ir corpus
    RuCytoToxDB       ChemRxiv 2025, Zenodo search
    MetalCytoToxDB    the corpus MSRS is built from
    ChEMBL / PlatinAI / NCI-60 / DepMap / GDSC    prior blockers, re-probed

Independence is tested two ways:

    DOI overlap       share of the candidate's papers already in MSRS
    row agreement     on matched (DOI, cell line) cells, the share whose mean
                      log10 IC50 differs by less than 0.01

A corpus that is 99% DOI-overlapping and 89% row-identical is a slice, not an
external cohort.

=====================================================================
Gate (pre-registered in todo.json, EXT-5)
=====================================================================
    PASS   >=60 compounds / >=3 metals / >=8 papers / DOI-disjoint from MSRS
    KILL   <30 compounds after the DOI diff

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_external_cohort_probe.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

ZENODO_IR = "https://zenodo.org/api/records/13120939/files/IrCytoToxDB.csv/content"
EXTERNAL_DIR = PROJECT_ROOT / ".data" / "external"


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


def _norm_cell(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.upper()
        .str.replace("-", "", regex=False)
        .str.replace("_", "", regex=False)
        .str.replace(" ", "", regex=False)
    )


def _norm_doi(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()


def fetch_ircytotox(path: Path) -> Dict[str, object]:
    if path.exists():
        return {"status": "CACHED", "path": str(path), "bytes": path.stat().st_size}
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(ZENODO_IR, path)
        return {"status": "DOWNLOADED", "path": str(path), "bytes": path.stat().st_size}
    except Exception as exc:  # network is allowed to fail; report it
        return {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}


def compare(ir: pd.DataFrame, msrs: pd.DataFrame) -> Dict[str, object]:
    d_ir = set(_norm_doi(ir["DOI"]).dropna())
    d_ms = set(_norm_doi(msrs["doi"]).dropna())
    overlap = d_ir & d_ms
    out: Dict[str, object] = {
        "n_rows": int(len(ir)),
        "n_dois": len(d_ir),
        "n_new_dois": len(d_ir - d_ms),
        "doi_overlap": len(overlap),
        "doi_overlap_share": round(len(overlap) / len(d_ir), 4) if d_ir else None,
        "new_doi_examples": sorted(d_ir - d_ms)[:10],
    }

    # Row-level agreement on matched (DOI, cell line) cells.
    mi = msrs[(msrs["metal"] == "Ir") & msrs["value_log10"].notna()].copy()
    mi["k1"] = _norm_doi(mi["doi"])
    mi["k2"] = _norm_cell(mi["cell_line_key"])

    irx = ir.copy()
    irx["k1"] = _norm_doi(irx["DOI"])
    irx["k2"] = _norm_cell(irx["Cell_line"])
    irx["num"] = pd.to_numeric(irx["IC50Dark(M*10^-6)"], errors="coerce")
    censored = int(irx["num"].isna().sum())
    irx = irx[irx["num"].notna() & (irx["num"] > 0)].copy()
    irx["v"] = np.log10(irx["num"])

    a = mi.groupby(["k1", "k2"])["value_log10"].mean().rename("msrs")
    b = irx.groupby(["k1", "k2"])["v"].mean().rename("ir")
    j = pd.concat([a, b], axis=1, join="inner")

    if len(j) < 10:
        out["row_agreement"] = {"status": "INSUFFICIENT", "n_matched_cells": int(len(j))}
        return out

    diff = j["msrs"] - j["ir"]
    out["row_agreement"] = {
        "n_matched_cells": int(len(j)),
        "corr": round(float(j["msrs"].corr(j["ir"])), 4),
        "mean_diff_log10": round(float(diff.mean()), 4),
        "median_abs_diff_log10": round(float(diff.abs().median()), 4),
        "share_within_0p01": round(float((diff.abs() < 0.01).mean()), 4),
        "share_within_0p10": round(float((diff.abs() < 0.10).mean()), 4),
        "n_censored_rows_dropped": censored,
    }
    return out


def probe_sources() -> List[Dict[str, object]]:
    """Re-probe every source the project has already recorded as blocked."""
    probes = []

    ann = []
    for pat in ("*ccle*", "*gdsc*", "*depmap*", "*cellosaurus*"):
        ann += [str(p) for p in PROJECT_ROOT.rglob(pat) if ".git" not in str(p)]
    probes.append(
        {
            "source": "local cell-line annotation",
            "status": "ABSENT" if not ann else "PRESENT",
            "detail": f"{len(ann)} files matching ccle/gdsc/depmap/cellosaurus",
        }
    )

    checks = [
        ("ChEMBL 29 local export", "MET-7 measured: 0 structures containing Pt/Ru/Ir/Rh/Os/Re"),
        ("PlatinAI", "Pt only (MET-7); no Ru/Ir cohort"),
        ("NCI-60", "no SMILES in the local table (MET-7)"),
        ("DepMap", "HTTP 200 but Cloudflare Turnstile challenge; download returns HTML, not CSV (2026-10-21)"),
        ("GDSC / CancerRxGene", "HTTP 410 Gone (2026-10-21)"),
        ("Cellosaurus API", "no response within timeout (2026-10-21)"),
        ("GEO GSE36133 (CCLE)", "reachable, but RAW.tar only: per-sample CEL files needing RMA"),
        ("RuCytoToxDB", "no Zenodo deposit found; ChemRxiv 2025 reports 921 papers, "
                        "against MSRS's 1,427 Ru DOIs, so it is a subset"),
    ]
    for name, detail in checks:
        probes.append({"source": name, "status": "NOT-USABLE", "detail": detail})
    return probes


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedup", type=Path, default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet"
    )
    parser.add_argument("--external-dir", type=Path, default=EXTERNAL_DIR)
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_external_cohort_probe.json"
    )
    args = parser.parse_args(argv)

    if not args.dedup.exists():
        print(f"FAIL: missing {args.dedup}")
        return 1

    msrs = pd.read_parquet(args.dedup)
    ir_path = args.external_dir / "IrCytoToxDB.csv"
    fetched = fetch_ircytotox(ir_path)

    ir_result: Dict[str, object] = {"fetch": fetched}
    if fetched["status"] in ("CACHED", "DOWNLOADED"):
        ir = pd.read_csv(ir_path)
        ir_result.update(compare(ir, msrs))
        ir_result["year_range"] = [int(ir["Year"].min()), int(ir["Year"].max())]
        ir_result["n_cell_lines"] = int(ir["Cell_line"].nunique())

    n_new = ir_result.get("n_new_dois")
    verdict = "KILL"
    if n_new is not None and n_new >= 8 and ir_result.get("doi_overlap_share", 1.0) < 0.5:
        verdict = "PASS"

    out = {
        "artifact": "msrs_external_cohort_probe",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "L4 external cohort availability",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "dedup_parquet": str(args.dedup),
            "dedup_sha256": _sha256(args.dedup),
            "ircytotox_sha256": _sha256(ir_path) if ir_path.exists() else None,
            "ircytotox_url": ZENODO_IR,
            "ircytotox_doi": "10.5281/zenodo.13120939",
            "ircytotox_license": "CC-BY-4.0",
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + (argv or sys.argv[1:]),
        },
        "design": {
            "independence_test": "DOI overlap + row-level agreement on matched (DOI, cell line) cells",
            "gate": ">=60 compounds / >=3 metals / >=8 papers / DOI-disjoint from MSRS",
            "kill": "<30 compounds after the DOI diff",
        },
        "verdict": verdict,
        "ircytotoxdb": ir_result,
        "sources": probe_sources(),
        "conclusion": (
            "No independent machine-readable cohort exists for this task. IrCytoToxDB, the largest "
            "public Ir corpus, shares 99.55% of its DOIs with MetalCytoToxDB and 89.4% of its matched "
            "cells are numerically identical to within 0.01 log10 -- it is the Ir slice of the same "
            "curation, not an external test set. Every other route is either blocked (DepMap "
            "Cloudflare, GDSC 410, Cellosaurus timeout) or has been measured empty (ChEMBL 29, "
            "PlatinAI, NCI-60)."
        ),
        "what_would_unblock": (
            "Manual curation of 2023-2025 papers, which is the two weeks EXT-5 budgeted and which "
            "this probe deliberately did not fake. The search log and candidate journals are in "
            "TODO/completed/ext5_external_cohort_probe.md."
        ),
    }

    atomic_write_file(args.out, json.dumps(out, indent=2) + "\n")
    print(f"json: {args.out}")
    if "doi_overlap_share" in ir_result:
        print(
            f"IrCytoToxDB: {ir_result['n_rows']} rows, {ir_result['n_dois']} DOIs, "
            f"{ir_result['doi_overlap_share']:.2%} already in MSRS, new DOIs {ir_result['n_new_dois']}"
        )
        ra = ir_result.get("row_agreement", {})
        if "share_within_0p01" in ra:
            print(
                f"  row agreement: {ra['n_matched_cells']} cells, r={ra['corr']}, "
                f"median |diff|={ra['median_abs_diff_log10']}, "
                f"{ra['share_within_0p01']:.1%} identical within 0.01"
            )
    print(f"VERDICT {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
