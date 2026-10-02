#!/usr/bin/env python3
"""Real-label loader for PlatinAI-style Pt complexes via MB-Finder portal API.

=====================================================================
Wf-295B / TODO-119 P1.3 — PlatinAI real-label loader (Strategy S7:
MB-Finder portal partial harvest, top-1 pick from Wf-292D).

Background
----------
Rusanov 2025 (PlatinAI, ChemRxiv 2025-06-20 v2,
DOI 10.26434/chemrxiv-2025-pp32k) reports per-cell-line pIC50 for
~4,078 Pt complexes across 4 cell lines (MCF7 / A549 / A2780 / A2780cis).

Direct ChemRxiv supplementary harvest is blocked by Cloudflare JS challenge
+ institutional login (per Wf-175, Wf-292D §1.1-§1.5).  We instead
source REAL Rusanov 2025 Pt-compound data from the public MB-Finder
portal (https://www.mb-finder.org), which Rusanov 2025 uses as the
hosted interface for the same compound catalogue.

API endpoints discovered (2026-09-29):
  POST https://api.mb-finder.org/api/v2/get-substances   -> paginated list
  POST https://api.mb-finder.org/api/v2/get-count        -> aggregate counts
  POST https://api.mb-finder.org/api/v2/get-tissues      -> tissue taxonomy
  POST https://api.mb-finder.org/api/v2/get-cell-lines   -> per-substance cell coverage
  POST https://api.mb-finder.org/api/v2/get-references   -> per-substance refs
  (per-cell-line IC50 endpoint NOT publicly exposed - honest-negative below)

Aggregate counts confirm scope:
  Pt compounds: 3,616   (target cohort)
  IC50 measurements: 31,996  (across all metals + cell lines)
  Cell lines: 827  References: 2,196

Rate limit: ~1 request / 2-3 s (429 Throttle after burst).

Output schema (.data/molmetal/platinai_4cell_mbfinder.csv):
  mb_id, imageId, smiles, mw, ceilLineCount, bioDataCount,
  has_a2780cis, has_a2780, has_a549, has_mcf7,
  n_cell_lines_with_data, n_biodata_total,
  total_cell_lines_listed, total_cell_lines_covered

Honest framing (per Wf-292D §3 R1):
- Compound IDs + SMILES are REAL Rusanov 2025 data (via MB-Finder)
- `bioDataCount` and per-cell-line `bioDataCount` are REAL counts
- Per-cell-line IC50 NUMERIC VALUES are NOT exposed by the public API
  (only the per-cell-line coverage MATRIX).  We therefore emit
  `has_<line>` boolean flags (real coverage) instead of synthetic pIC50.
- This REPLACES the SYNTHETIC pIC50 loader (Wf-278B) as the closest-to-
  real PlatinAI dataset we can ship without academic outreach (Wf-292D §6).

Usage
-----
::
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \\
        molmetal/scripts/platinai_real_label_loader.py \\
        --out .data --n-target 100 --start-page 850
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MB_API_BASE = "https://api.mb-finder.org/api/v2"
TARGET_CELL_LINES = ["MCF7", "A549", "A2780", "A2780cis"]


def _post(endpoint: str, body: dict, *, max_retries: int = 4, timeout: int = 20) -> Optional[dict]:
    """POST with exponential backoff on 429 throttle. Fallback to GET if 405.

    Wf-305A: detect MB-Finder soft-rate-limit (status 200 with success=false).
    Returns None on success=false so callers can short-circuit pagination
    instead of wasting 400+ empty page requests.
    """
    url = f"{MB_API_BASE}/{endpoint}"
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=body, timeout=timeout)
            if r.status_code == 405:
                # Some endpoints (e.g. get-count) are GET-only
                r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                try:
                    payload = r.json()
                except ValueError:
                    # Non-JSON 200 (e.g. CF challenge HTML) — treat as soft block
                    print(f"      [warn] {endpoint}: non-JSON 200 response "
                          f"({len(r.text)} bytes) — possible CF challenge", flush=True)
                    return None
                if isinstance(payload, dict) and payload.get("success") is False:
                    msg = payload.get("message", "")
                    retry_after = payload.get("retry_after", "?")
                    print(f"      [BLOCK_RATE_LIMIT] {endpoint}: {msg} "
                          f"(retry_after={retry_after}s)", flush=True)
                    return None
                return payload
            if r.status_code == 429:
                wait = 3.0 * (2 ** attempt)
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(2.0 * (attempt + 1))
    if last_err:
        raise last_err
    return None


def fetch_pt_compound_page(page: int, per_page: int = 5) -> List[dict]:
    """Fetch one page of substances (per_page is server-capped at 5)."""
    out = _post("get-substances", {"per_page": per_page, "page": page})
    if not out:
        return []
    return out.get("data", [])


def fetch_cell_lines(substance_id: int) -> List[dict]:
    """Fetch cell lines (with bioDataCount) for a substance."""
    out = _post("get-cell-lines", {"id": substance_id, "per_page": 200})
    if not out:
        return []
    return out.get("data", [])


def fetch_count() -> dict:
    """Aggregate counts."""
    return _post("get-count", {}) or {}


def _build_row(item: dict, cell_lines_data: List[dict]) -> dict:
    """Build a single row dict for a Pt compound + cell-line coverage."""
    cell_line_names = {cl.get("cellLine", cl.get("name", "")) for cl in cell_lines_data}
    n_cell_total = len(cell_lines_data)
    n_biodata_total = sum(int(cl.get("bioDataCount", 0)) for cl in cell_lines_data)
    coverage_flags = {}
    for line in TARGET_CELL_LINES:
        if line == "A2780cis":
            coverage_flags[f"has_a2780cis"] = int(
                any("A2780cis" == cn or "A2780cis" in (cn or "") for cn in cell_line_names)
            )
        else:
            coverage_flags[f"has_{line.lower()}"] = int(
                any(line == cn or (line + " ") in (cn or "") or cn == line
                    for cn in cell_line_names)
            )
    return {
        "mb_id": int(item["id"]),
        "imageId": item.get("imageId", ""),
        "smiles": item.get("smiles", ""),
        "mw": item.get("mw", ""),
        "ceilLineCount": item.get("ceilLineCount", 0),
        "bioDataCount": item.get("bioDataCount", 0),
        **coverage_flags,
        "n_cell_lines_with_data": sum(1 for cl in cell_lines_data if int(cl.get("bioDataCount", 0)) > 0),
        "n_biodata_total": n_biodata_total,
        "total_cell_lines_listed": n_cell_total,
    }


def _has_cell_line(row: dict, cell_line: str) -> bool:
    """Check if row has bio-data for the given cell line."""
    key = "has_a2780cis" if cell_line == "A2780cis" else f"has_{cell_line.lower()}"
    return int(row.get(key, 0)) == 1


def build_loader(n_target: int, start_page: int, per_request_delay: float) -> List[dict]:
    """Iterate Pt pages, filter for Pt compounds, enrich with cell-line coverage.

    Original single-pass loader (Wf-295B). Stop after n_target compounds.
    """
    rows: List[dict] = []
    page = start_page
    total_pages_seen = 0
    total_requests = 0
    n_non_pt = 0
    while len(rows) < n_target and page <= 1547:
        items = fetch_pt_compound_page(page=page, per_page=5)
        total_pages_seen += 1
        total_requests += 1
        if not items:
            page += 1
            time.sleep(per_request_delay)
            continue
        for item in items:
            imageId = item.get("imageId", "")
            if not imageId.startswith("MB-Pt-"):
                n_non_pt += 1
                continue
            sub_id = int(item["id"])
            cell_lines = fetch_cell_lines(sub_id)
            total_requests += 1
            rows.append(_build_row(item, cell_lines))
            if len(rows) >= n_target:
                break
        page += 1
        time.sleep(per_request_delay)
    return rows, total_requests, total_pages_seen, n_non_pt


def build_loader_stratified(
    n_target_per_cell: int,
    cell_lines: List[str],
    start_page: int,
    per_request_delay: float,
    max_pages: int = 1547,
    progress_every: int = 25,
) -> tuple:
    """Stratified fetch: collect n_target_per_cell compounds with bio-data per cell line.

    Wf-301C: per-cell-line balanced sampling across MB-Finder pages.

    For each cell line in `cell_lines`, iterate pages until we collect
    `n_target_per_cell` compounds that have `has_<line>=1`. Returns ALL
    collected rows (deduplicated by mb_id), plus per-cell-line counts.

    Note: the same compound can satisfy multiple cell lines, so the
    dedup'd row count may be less than `n_target_per_cell * len(cell_lines)`.
    Per-cell-line counts track how many SATISFYING compounds per cell line
    were added (one row can be counted for multiple cell lines).
    """
    seen_mb_ids: set = set()
    rows: List[dict] = []
    per_cell_count: Dict[str, int] = {cl: 0 for cl in cell_lines}
    page = start_page
    total_pages_seen = 0
    total_requests = 0
    n_non_pt = 0
    # Stop when all target cell lines reach n_target_per_cell OR end of pages
    target_keys = [("has_a2780cis" if cl == "A2780cis" else f"has_{cl.lower()}")
                   for cl in cell_lines]
    while page <= max_pages:
        # Check if all targets are satisfied
        if all(per_cell_count[cl] >= n_target_per_cell for cl in cell_lines):
            break
        items = fetch_pt_compound_page(page=page, per_page=5)
        total_pages_seen += 1
        total_requests += 1
        if not items:
            page += 1
            time.sleep(per_request_delay)
            continue
        for item in items:
            imageId = item.get("imageId", "")
            if not imageId.startswith("MB-Pt-"):
                n_non_pt += 1
                continue
            sub_id = int(item["id"])
            cell_lines_data = fetch_cell_lines(sub_id)
            total_requests += 1
            row = _build_row(item, cell_lines_data)
            if sub_id in seen_mb_ids:
                continue
            seen_mb_ids.add(sub_id)
            rows.append(row)
            # Update per-cell-line counts based on this row's coverage
            for cl, key in zip(cell_lines, target_keys):
                if per_cell_count[cl] < n_target_per_cell and int(row.get(key, 0)) == 1:
                    per_cell_count[cl] += 1
        # Periodic progress
        if total_pages_seen % progress_every == 0:
            counts_str = "/".join(f"{cl}={per_cell_count[cl]}" for cl in cell_lines)
            print(
                f"      [progress] page={page} pages_seen={total_pages_seen} "
                f"rows={len(rows)} reqs={total_requests} counts: {counts_str}",
                flush=True,
            )
        page += 1
        time.sleep(per_request_delay)
    return rows, per_cell_count, total_requests, total_pages_seen, n_non_pt


def write_outputs(rows: List[dict], out_dir: Path, total_requests: int,
                  total_pages_seen: int, n_non_pt: int, n_target: int,
                  start_page: int, per_request_delay: float,
                  stratified: bool = False,
                  target_cell_lines: Optional[List[str]] = None,
                  per_cell_count: Optional[Dict[str, int]] = None,
                  csv_basename: str = "platinai_4cell_mbfinder") -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{csv_basename}.csv"
    meta_path = out_dir / f"{csv_basename}.meta.json"
    fieldnames = [
        "mb_id", "imageId", "smiles", "mw", "ceilLineCount", "bioDataCount",
        "has_mcf7", "has_a549", "has_a2780", "has_a2780cis",
        "n_cell_lines_with_data", "n_biodata_total", "total_cell_lines_listed",
    ]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    n_with_real_label = sum(1 for r in rows if r["n_cell_lines_with_data"] > 0)
    coverage = {
        line: sum(1 for r in rows if r[f"has_{line.lower()}"])
        for line in ["mcf7", "a549", "a2780", "a2780cis"]
    }
    meta = {
        "n_rows": len(rows),
        "n_target": n_target,
        "start_page": start_page,
        "per_request_delay": per_request_delay,
        "total_api_requests": total_requests,
        "total_pages_seen": total_pages_seen,
        "n_non_pt_pages_skipped": n_non_pt,
        "n_with_real_label": n_with_real_label,
        "source": "MB-Finder portal (https://www.mb-finder.org)",
        "underlying_paper": "Rusanov 2025 PlatinAI (DOI 10.26434/chemrxiv-2025-pp32k)",
        "endpoints_used": [
            "POST /api/v2/get-substances",
            "POST /api/v2/get-cell-lines",
            "POST /api/v2/get-count",
        ],
        "framing": (
            "REAL MB-Finder-hosted Rusanov 2025 Pt compounds + REAL per-cell-line "
            "bio-data COVERAGE MATRIX. Per-cell-line IC50 NUMERIC VALUES are NOT "
            "exposed by the public MB-Finder API (only presence/absence + count). "
            "We emit has_<line> boolean flags (real coverage) instead of synthetic pIC50. "
            "This replaces the SYNTHETIC pIC50 loader (Wf-278B) as the closest-to-real "
            "PlatinAI dataset without academic outreach."
        ),
        "target_cell_lines": target_cell_lines or TARGET_CELL_LINES,
        "coverage": coverage,
        "stratified": stratified,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if stratified and per_cell_count is not None:
        meta["per_cell_count"] = per_cell_count
        meta["per_cell_target"] = {cl: n_target for cl in (target_cell_lines or [])}
        meta["all_targets_met"] = all(
            per_cell_count.get(cl, 0) >= n_target
            for cl in (target_cell_lines or [])
        )
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return {"csv": str(csv_path), "meta": str(meta_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--out", default=".data/molmetal", help="Output directory (or .csv file path)")
    parser.add_argument("--n-target", type=int, default=100,
                        help="Number of Pt compounds to load (single-pass mode)")
    parser.add_argument("--n-target-per-cell", type=int, default=None,
                        help="If set with --cell-lines, run STRATIFIED mode: collect "
                             "this many compounds per cell line")
    parser.add_argument("--cell-lines", type=str, default=None,
                        help="Comma-separated target cell lines for stratified sampling "
                             "(e.g. 'A549,MCF7,A2780,A2780cis')")
    parser.add_argument("--csv-basename", type=str, default="platinai_4cell_mbfinder",
                        help="CSV/Meta basename (default platinai_4cell_mbfinder). "
                             "If --out ends in .csv, this is auto-set from the file stem.")
    parser.add_argument("--start-page", type=int, default=850,
                        help="Starting page (default 850, deep in Pt range)")
    parser.add_argument("--delay", type=float, default=2.5,
                        help="Seconds between API requests (rate-limit buffer)")
    parser.add_argument("--max-pages", type=int, default=1547,
                        help="Maximum pages to scan before stopping "
                             "(stratified mode; default 1547 = full Pt corpus)")
    args = parser.parse_args()
    out_arg = args.out
    # If --out is a .csv path, derive basename + directory from it
    if out_arg.endswith(".csv"):
        out_path = Path(out_arg)
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_arg
        out_dir = out_path.parent
        if args.csv_basename == "platinai_4cell_mbfinder":
            args.csv_basename = out_path.stem
    else:
        out_dir = PROJECT_ROOT / out_arg if not out_arg.startswith("/") else Path(out_arg)

    stratified = bool(args.n_target_per_cell and args.cell_lines)
    if args.n_target_per_cell and not args.cell_lines:
        parser.error("--n-target-per-cell requires --cell-lines")
    if args.cell_lines and not args.n_target_per_cell:
        parser.error("--cell-lines requires --n-target-per-cell")
    cell_lines = (
        [cl.strip() for cl in args.cell_lines.split(",") if cl.strip()]
        if args.cell_lines else None
    )

    if stratified:
        print(f"=== Wf-301C: MB-Finder STRATIFIED fetch ({len(cell_lines)} cell lines) ===")
    else:
        print(f"=== Wf-295B: MB-Finder real-label loader (Strategy S7, Wf-292D top-1) ===")
    print(f"Source: {MB_API_BASE}")
    if stratified:
        print(f"n_target_per_cell: {args.n_target_per_cell}   "
              f"cell_lines: {cell_lines}   start_page: {args.start_page}   "
              f"delay: {args.delay}s")
    else:
        print(f"n_target: {args.n_target}   start_page: {args.start_page}   "
              f"delay: {args.delay}s")
    print()

    print("[1/3] Probing MB-Finder aggregate counts...")
    counts = fetch_count()
    print(f"      compounds={counts.get('compoundsCount')} "
          f"Pt={counts.get('metalCounts',{}).get('Pt')} "
          f"IC50s={counts.get('ic50Count')} cells={counts.get('cellsCount')}")

    print(f"[2/3] Paginating Pt compounds starting page {args.start_page}...")
    if stratified:
        rows, per_cell_count, total_requests, total_pages_seen, n_non_pt = (
            build_loader_stratified(
                args.n_target_per_cell, cell_lines,
                args.start_page, args.delay,
                max_pages=args.max_pages,
            )
        )
        print(f"      Got {len(rows)} unique Pt compounds in {total_requests} "
              f"API requests ({total_pages_seen} pages, {n_non_pt} non-Pt skipped)")
        print(f"      Per-cell-line counts: {per_cell_count}")
    else:
        rows, total_requests, total_pages_seen, n_non_pt = build_loader(
            args.n_target, args.start_page, args.delay
        )
        per_cell_count = None
        print(f"      Got {len(rows)} Pt compounds in {total_requests} API requests "
              f"({total_pages_seen} pages, {n_non_pt} non-Pt skipped)")

    print(f"[3/3] Writing outputs to {out_dir}...")
    n_target_for_meta = args.n_target_per_cell if stratified else args.n_target
    paths = write_outputs(
        rows, out_dir, total_requests, total_pages_seen, n_non_pt,
        n_target_for_meta, args.start_page, args.delay,
        stratified=stratified,
        target_cell_lines=cell_lines,
        per_cell_count=per_cell_count,
        csv_basename=args.csv_basename,
    )
    print(f"      CSV:  {paths['csv']}")
    print(f"      Meta: {paths['meta']}")

    # Smoke verification
    n_nan_smiles = sum(1 for r in rows if not r["smiles"])
    n_real_label = sum(1 for r in rows if r["n_cell_lines_with_data"] > 0)
    print()
    print("=== Smoke verification ===")
    print(f"n_smoke_mols:        {len(rows)}")
    print(f"n_with_real_label:   {n_real_label} (compounds with >=1 cell-line bio-data)")
    print(f"n_nan_smiles:        {n_nan_smiles}")
    print(f"coverage mcf7/a549/a2780/a2780cis: "
          f"{sum(1 for r in rows if r['has_mcf7'])}/"
          f"{sum(1 for r in rows if r['has_a549'])}/"
          f"{sum(1 for r in rows if r['has_a2780'])}/"
          f"{sum(1 for r in rows if r['has_a2780cis'])}")
    if stratified and per_cell_count is not None:
        all_met = all(
            per_cell_count.get(cl, 0) >= args.n_target_per_cell
            for cl in cell_lines
        )
        print(f"stratified_all_targets_met: {all_met}")
    return 0


if __name__ == "__main__":
    sys.exit(main())