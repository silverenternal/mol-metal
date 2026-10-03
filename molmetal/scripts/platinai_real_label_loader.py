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
  POST https://api.mb-finder.org/api/v2/get-ceil-line-bio-data
        -> REAL per-cell-line measured values (numeric value string + DOI
           + method + incubationTime), no auth, per_page server-capped at 5

  CORRECTION (2026-10-03 - "kill the false honest-negative"):
    The line that stood here read:
      "(per-cell-line IC50 endpoint NOT publicly exposed - honest-negative below)"
    It was WRONG, and it was wrong from the day it was written (2026-09-29).
    It was inferred from Wf-292D's Cloudflare-blocked ChemRxiv scraping and
    from probing only get-cell-lines (a coverage COUNT matrix), without ever
    probing get-ceil-line-bio-data. That endpoint is public, needs no auth,
    and returns real measured numeric values, e.g. (verified live 2026-10-03):
      cell_line="A2780cis"  MB-Pt-000017  value="1.2"      doi=10.1021/ic8012359
      cell_line="A2780cis"  MB-Pt-000132  value="0.6 ± 0.1"
      cell_line="A2780"     MB-Pt-000092  value="0.0046 ± 0.0023"
    The old text is kept here on purpose (audit trail); repo-wide copies of
    the false claim are catalogued in TODO/completed/pt_data_route_correction.md.

Aggregate counts confirm scope:
  Pt compounds: 3,616   (target cohort)
  IC50 measurements: 31,996  (across all metals + cell lines)
  Cell lines: 827  References: 2,196

Rate limit: ~1 request / 2-3 s (429 Throttle after burst).

Output schema, coverage mode (.data/molmetal/platinai_4cell_mbfinder.csv):
  mb_id, imageId, smiles, mw, ceilLineCount, bioDataCount,
  has_a2780cis, has_a2780, has_a549, has_mcf7,
  n_cell_lines_with_data, n_biodata_total,
  total_cell_lines_listed, total_cell_lines_covered

Output schema, REAL-numeric mode (--biodata-cell-line, NEW 2026-10-03):
  compound_code, cell_line, value_raw, value_mean, value_sd, smiles, mw,
  substance_id, casNumber, doi, method, incubation_time, biodata_row_id,
  n_measurements, title

Honest framing (corrected 2026-10-03):
- Compound IDs + SMILES are REAL Rusanov 2025 data (via MB-Finder)
- `bioDataCount` and per-cell-line `bioDataCount` are REAL counts
- CORRECTION: the text here used to claim "Per-cell-line IC50 NUMERIC
  VALUES are NOT exposed by the public API (only the per-cell-line coverage
  MATRIX). We therefore emit has_<line> boolean flags (real coverage)
  instead of synthetic pIC50." The first sentence was FALSE (endpoint
  get-ceil-line-bio-data exposes them; see module docstring CORRECTION).
  The second sentence described a real, retained coverage-matrix mode, but
  it had been used to justify a false conclusion. Two modes now exist:
    (a) coverage mode (Wf-295B / Wf-301C, backwards-compatible): has_<line>
        booleans + counts from get-cell-lines;
    (b) REAL-numeric mode (NEW 2026-10-03): fetch_cell_line_biodata_pages()
        + build_structure_index() + join_structures(), exposed via
        --biodata-cell-line, emits the real measured value + SMILES join.
- The old "closest-to-real ... without academic outreach" framing is
  superseded: mode (b) reaches real per-cell-line numeric labels from the
  public API, no outreach required.

Usage
-----
::
    # coverage mode (single-venv rule, AGENTS.md §5.9):
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/platinai_real_label_loader.py \\
        --out .data --n-target 100 --start-page 850

    # REAL per-cell-line numeric labels (NEW 2026-10-03):
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/platinai_real_label_loader.py \\
        --biodata-cell-line A2780cis \\
        --biodata-max-pages 60 \\
        --substances-start-page 1400 --substances-max-pages 147 \\
        --out .data/molmetal --csv-basename platinai_a2780cis_biodata
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


def _post(
    endpoint: str, body: dict, *, max_retries: int = 4, timeout: int = 20
) -> Optional[dict]:
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
                    print(
                        f"      [warn] {endpoint}: non-JSON 200 response "
                        f"({len(r.text)} bytes) — possible CF challenge",
                        flush=True,
                    )
                    return None
                if isinstance(payload, dict) and payload.get("success") is False:
                    msg = payload.get("message", "")
                    retry_after = payload.get("retry_after", "?")
                    print(
                        f"      [BLOCK_RATE_LIMIT] {endpoint}: {msg} "
                        f"(retry_after={retry_after}s)",
                        flush=True,
                    )
                    return None
                return payload
            if r.status_code == 429:
                wait = 3.0 * (2**attempt)
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


# ---------------------------------------------------------------------------
# REAL per-cell-line numeric bio-data mode (NEW 2026-10-03 — corrects the
# false honest-negative that per-cell-line IC50 values were not exposed).
# ---------------------------------------------------------------------------

BIODATA_ENDPOINT = "get-ceil-line-bio-data"  # server spelling is "ceil"


def compound_code_from_image_url(image_url: str) -> str:
    """Extract the compound code from an MB-Finder image URL.

    Endpoint C returns ``compounds.imageUrl`` such as
    ``https://api.mb-finder.org/storage/images/MB-Pt-000017.svg?v=1770252062``.
    The join key is basename minus ``.svg`` with the query stripped ->
    ``MB-Pt-000017``.  Endpoint B carries the same code as ``imageId``
    (verified live 2026-10-03: MB-Pt-000017 joins to substance id 13,
    smiles containing ``[Pt]``).
    """
    if not image_url:
        return ""
    base = str(image_url).split("/")[-1].split("?")[0]
    if base.endswith(".svg"):
        base = base[:-4]
    return base


def parse_measurement_value(value) -> tuple:
    """Parse MB-Finder value strings -> (mean: float|None, sd: float|None).

    Both live forms (verified 2026-10-03):
      "1.2"        -> (1.2, None)
      "0.6 ± 0.1"  -> (0.6, 0.1)
    Unparseable input -> (None, None); callers count these as honest
    parse_failures instead of dropping rows silently.
    """
    if value is None:
        return None, None
    s = str(value).strip()
    if not s:
        return None, None
    for sep in ("±", "+/-"):
        if sep in s:
            mean_s, _, sd_s = s.partition(sep)
            try:
                return float(mean_s.strip()), float(sd_s.strip())
            except ValueError:
                return None, None
    try:
        return float(s), None
    except ValueError:
        return None, None


def fetch_cell_line_biodata(
    cell_line: str, page: int = 1, per_page: int = 5
) -> List[dict]:
    """Fetch ONE page of per-cell-line REAL measured bio-data.

    Endpoint: POST /api/v2/get-ceil-line-bio-data (public, no auth; the
    server caps per_page at 5 — verified live 2026-10-03).  Each ``data``
    row carries a ``compounds`` object (numeric_value / doi / method /
    incubationTime / imageUrl) plus a top-level ``value`` string in the
    form "1.2" or "0.6 ± 0.1".
    """
    payload = _post(
        BIODATA_ENDPOINT,
        {"cell_line": cell_line, "per_page": per_page, "page": page},
    )
    if not payload:
        return []
    return payload.get("data", [])


def flatten_biodata_row(row: dict, cell_line: str) -> dict:
    """Flatten one endpoint-C row into a loader record (no SMILES yet).

    NOTE: the top-level ``id`` on these rows is the BIODATA row id, NOT the
    substance id — the substance id arrives only via the structure join.
    """
    comp = row.get("compounds") or {}
    value_raw = row.get("value")
    if value_raw is None:
        value_raw = comp.get("numeric_value")
    mean, sd = parse_measurement_value(value_raw)
    return {
        "compound_code": compound_code_from_image_url(comp.get("imageUrl", "")),
        "cell_line": cell_line,
        "value_raw": "" if value_raw is None else str(value_raw),
        "value_mean": mean,
        "value_sd": sd,
        "doi": comp.get("doi", "") or "",
        "title": comp.get("title", "") or "",
        "method": comp.get("method", "") or "",
        "incubation_time": comp.get("incubationTime", "") or "",
        "biodata_row_id": row.get("id"),
        "n_measurements": len(row.get("measurements") or []),
    }


def fetch_cell_line_biodata_pages(
    cell_line: str,
    max_pages: int = 300,
    per_request_delay: float = 2.5,
    per_page: int = 5,
    progress_every: int = 25,
) -> tuple:
    """Paginate get-ceil-line-bio-data and flatten every row.

    Returns (records, total_requests).  Stops on: empty page, soft
    rate-limit (``_post`` -> None), ``meta.last_page`` reached, or
    max_pages.  Reuses ``_post`` so the existing retry/429 backoff and
    soft-rate-limit detection apply unchanged.
    """
    records: List[dict] = []
    page = 1
    total_requests = 0
    last_page: Optional[int] = None
    while page <= max_pages:
        payload = _post(
            BIODATA_ENDPOINT,
            {"cell_line": cell_line, "per_page": per_page, "page": page},
        )
        total_requests += 1
        if not payload:
            break
        data = payload.get("data", [])
        if not data:
            break
        for row in data:
            records.append(flatten_biodata_row(row, cell_line))
        last_page = (payload.get("meta") or {}).get("last_page") or last_page
        if progress_every and total_requests % progress_every == 0:
            print(
                f"      [progress] biodata page={page} rows={len(records)} "
                f"reqs={total_requests} last_page={last_page}",
                flush=True,
            )
        if last_page and page >= int(last_page):
            break
        page += 1
        time.sleep(per_request_delay)
    return records, total_requests


def build_structure_index(
    start_page: int,
    max_pages: int,
    per_request_delay: float,
    per_page: int = 5,
) -> tuple:
    """Page get-substances (reusing fetch_pt_compound_page) -> structure map.

    Returns ({compound_code -> {"substance_id", "smiles", "mw", "casNumber"}},
    total_requests).  The full corpus spans 1547 pages of 5 rows; Pt codes
    sit in the high pages (descending order; ~828..1547).  Callers pass a
    bounded window and must treat unmatched codes as honestly unmatched
    (see join_structures).
    """
    index: Dict[str, dict] = {}
    total_requests = 0
    for i in range(max_pages):
        page = start_page + i
        items = fetch_pt_compound_page(page=page, per_page=per_page)
        total_requests += 1
        if not items:
            break
        for item in items:
            code = item.get("imageId") or compound_code_from_image_url(
                item.get("imageUrl", "")
            )
            if not code:
                continue
            index[code] = {
                "substance_id": item.get("id"),
                "smiles": item.get("smiles", "") or "",
                "mw": item.get("mw", "") or "",
                "casNumber": item.get("casNumber"),
            }
        time.sleep(per_request_delay)
    return index, total_requests


def join_structures(records: List[dict], structure_index: Dict[str, dict]) -> tuple:
    """Attach structure fields to bio-data records via the compound code.

    Returns (joined_records, n_matched, n_unmatched).  Unmatched records are
    KEPT (empty smiles, empty substance_id) and counted — a partial index
    window must surface as an honest unmatched count, never as dropped rows.
    """
    joined: List[dict] = []
    n_matched = 0
    for rec in records:
        new = dict(rec)
        struct = structure_index.get(rec.get("compound_code", ""))
        if struct:
            new["substance_id"] = struct.get("substance_id")
            new["smiles"] = struct.get("smiles", "")
            new["mw"] = struct.get("mw", "")
            new["casNumber"] = struct.get("casNumber")
            n_matched += 1
        else:
            new["substance_id"] = ""
            new["smiles"] = ""
            new["mw"] = ""
            new["casNumber"] = None
        joined.append(new)
    return joined, n_matched, len(records) - n_matched


def _build_row(item: dict, cell_lines_data: List[dict]) -> dict:
    """Build a single row dict for a Pt compound + cell-line coverage."""
    cell_line_names = {cl.get("cellLine", cl.get("name", "")) for cl in cell_lines_data}
    n_cell_total = len(cell_lines_data)
    n_biodata_total = sum(int(cl.get("bioDataCount", 0)) for cl in cell_lines_data)
    coverage_flags = {}
    for line in TARGET_CELL_LINES:
        if line == "A2780cis":
            coverage_flags["has_a2780cis"] = int(
                any(
                    "A2780cis" == cn or "A2780cis" in (cn or "")
                    for cn in cell_line_names
                )
            )
        else:
            coverage_flags[f"has_{line.lower()}"] = int(
                any(
                    line == cn or (line + " ") in (cn or "") or cn == line
                    for cn in cell_line_names
                )
            )
    return {
        "mb_id": int(item["id"]),
        "imageId": item.get("imageId", ""),
        "smiles": item.get("smiles", ""),
        "mw": item.get("mw", ""),
        "ceilLineCount": item.get("ceilLineCount", 0),
        "bioDataCount": item.get("bioDataCount", 0),
        **coverage_flags,
        "n_cell_lines_with_data": sum(
            1 for cl in cell_lines_data if int(cl.get("bioDataCount", 0)) > 0
        ),
        "n_biodata_total": n_biodata_total,
        "total_cell_lines_listed": n_cell_total,
    }


def _has_cell_line(row: dict, cell_line: str) -> bool:
    """Check if row has bio-data for the given cell line."""
    key = "has_a2780cis" if cell_line == "A2780cis" else f"has_{cell_line.lower()}"
    return int(row.get(key, 0)) == 1


def build_loader(
    n_target: int, start_page: int, per_request_delay: float
) -> List[dict]:
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
    target_keys = [
        ("has_a2780cis" if cl == "A2780cis" else f"has_{cl.lower()}")
        for cl in cell_lines
    ]
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


def write_outputs(
    rows: List[dict],
    out_dir: Path,
    total_requests: int,
    total_pages_seen: int,
    n_non_pt: int,
    n_target: int,
    start_page: int,
    per_request_delay: float,
    stratified: bool = False,
    target_cell_lines: Optional[List[str]] = None,
    per_cell_count: Optional[Dict[str, int]] = None,
    csv_basename: str = "platinai_4cell_mbfinder",
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{csv_basename}.csv"
    meta_path = out_dir / f"{csv_basename}.meta.json"
    fieldnames = [
        "mb_id",
        "imageId",
        "smiles",
        "mw",
        "ceilLineCount",
        "bioDataCount",
        "has_mcf7",
        "has_a549",
        "has_a2780",
        "has_a2780cis",
        "n_cell_lines_with_data",
        "n_biodata_total",
        "total_cell_lines_listed",
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
            "bio-data COVERAGE MATRIX. CORRECTION (2026-10-03): earlier versions "
            "of this meta claimed per-cell-line IC50 NUMERIC VALUES are NOT "
            "exposed by the public API — that was WRONG. Endpoint "
            "get-ceil-line-bio-data returns real numeric values (see "
            "--biodata-cell-line mode / TODO/completed/pt_data_route_correction.md). "
            "This coverage mode emits has_<line> boolean flags (real coverage) "
            "and is retained for backwards compatibility; for numeric labels "
            "use the biodata mode."
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
            per_cell_count.get(cl, 0) >= n_target for cl in (target_cell_lines or [])
        )
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return {"csv": str(csv_path), "meta": str(meta_path)}


BIODATA_FIELDNAMES = [
    "compound_code",
    "cell_line",
    "value_raw",
    "value_mean",
    "value_sd",
    "smiles",
    "mw",
    "substance_id",
    "casNumber",
    "doi",
    "method",
    "incubation_time",
    "biodata_row_id",
    "n_measurements",
    "title",
]


def write_biodata_outputs(
    joined: List[dict], out_dir: Path, csv_basename: str, meta_extra: dict
) -> Dict[str, str]:
    """Write the REAL-numeric-mode CSV + meta (joined bio-data + structures)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{csv_basename}.csv"
    meta_path = out_dir / f"{csv_basename}.meta.json"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=BIODATA_FIELDNAMES, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(joined)
    n_with_value = sum(1 for r in joined if r["value_mean"] is not None)
    meta = dict(meta_extra)
    meta.update(
        {
            "n_rows": len(joined),
            "n_rows_with_numeric_value": n_with_value,
            "n_rows_parse_failed": len(joined) - n_with_value,
            "n_rows_with_smiles": sum(1 for r in joined if r["smiles"]),
            "source": "MB-Finder portal (https://www.mb-finder.org)",
            "underlying_paper": "Rusanov 2025 PlatinAI (DOI 10.26434/chemrxiv-2025-pp32k)",
            "endpoints_used": [
                "POST /api/v2/get-ceil-line-bio-data",
                "POST /api/v2/get-substances",
            ],
            "framing": (
                "REAL per-cell-line measured values from the public MB-Finder API "
                "(get-ceil-line-bio-data) joined to REAL compound SMILES from "
                "get-substances. This corrects the earlier false honest-negative "
                "('per-cell-line IC50 NUMERIC not exposed'); see "
                "TODO/completed/pt_data_route_correction.md. Rows whose code is "
                "outside the indexed page window keep empty SMILES and are counted "
                "as unmatched (n_matched_structures / n_unmatched_structures)."
            ),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return {"csv": str(csv_path), "meta": str(meta_path)}


def _run_biodata_mode(args) -> int:
    """REAL per-cell-line numeric mode (NEW 2026-10-03)."""
    cell_line = args.biodata_cell_line
    print(
        "=== MB-Finder REAL per-cell-line bio-data mode (corrects the false "
        "honest-negative, 2026-10-03) ==="
    )
    print(f"Source: {MB_API_BASE}")
    print(
        f"cell_line: {cell_line}   biodata_max_pages: {args.biodata_max_pages}   "
        f"delay: {args.delay}s"
    )
    print()
    print(f"[1/4] Paginating {BIODATA_ENDPOINT} for {cell_line}...")
    records, bio_requests = fetch_cell_line_biodata_pages(
        cell_line,
        max_pages=args.biodata_max_pages,
        per_request_delay=args.delay,
    )
    n_parsed = sum(1 for r in records if r["value_mean"] is not None)
    print(
        f"      rows={len(records)} reqs={bio_requests} parsed_numeric={n_parsed} "
        f"parse_failures={len(records) - n_parsed}"
    )
    print(
        f"[2/4] Building substance structure index (pages "
        f"{args.substances_start_page}.."
        f"{args.substances_start_page + args.substances_max_pages - 1})..."
    )
    index, index_requests = build_structure_index(
        args.substances_start_page,
        args.substances_max_pages,
        args.delay,
    )
    print(f"      index_size={len(index)} substance codes reqs={index_requests}")
    print("[3/4] Joining compound_code -> SMILES...")
    joined, n_matched, n_unmatched = join_structures(records, index)
    print(
        f"      matched={n_matched} unmatched={n_unmatched} "
        f"(unmatched rows kept with empty smiles)"
    )
    print(f"[4/4] Writing outputs to {args.out_dir}...")
    meta_extra = {
        "cell_line": cell_line,
        "biodata_max_pages": args.biodata_max_pages,
        "substances_start_page": args.substances_start_page,
        "substances_max_pages": args.substances_max_pages,
        "per_request_delay": args.delay,
        "biodata_api_requests": bio_requests,
        "substances_api_requests": index_requests,
        "total_api_requests": bio_requests + index_requests,
        "n_matched_structures": n_matched,
        "n_unmatched_structures": n_unmatched,
        "coverage_note": (
            "structure index covers a bounded page window; unmatched codes "
            "require a wider --substances-start-page/--substances-max-pages range"
        ),
    }
    paths = write_biodata_outputs(joined, args.out_dir, args.csv_basename, meta_extra)
    print(f"      CSV:  {paths['csv']}")
    print(f"      Meta: {paths['meta']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--out", default=".data/molmetal", help="Output directory (or .csv file path)"
    )
    parser.add_argument(
        "--n-target",
        type=int,
        default=100,
        help="Number of Pt compounds to load (single-pass mode)",
    )
    parser.add_argument(
        "--n-target-per-cell",
        type=int,
        default=None,
        help="If set with --cell-lines, run STRATIFIED mode: collect "
        "this many compounds per cell line",
    )
    parser.add_argument(
        "--cell-lines",
        type=str,
        default=None,
        help="Comma-separated target cell lines for stratified sampling "
        "(e.g. 'A549,MCF7,A2780,A2780cis')",
    )
    parser.add_argument(
        "--csv-basename",
        type=str,
        default="platinai_4cell_mbfinder",
        help="CSV/Meta basename (default platinai_4cell_mbfinder). "
        "If --out ends in .csv, this is auto-set from the file stem.",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=850,
        help="Starting page (default 850, deep in Pt range)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.5,
        help="Seconds between API requests (rate-limit buffer)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1547,
        help="Maximum pages to scan before stopping "
        "(stratified mode; default 1547 = full Pt corpus)",
    )
    parser.add_argument(
        "--biodata-cell-line",
        type=str,
        default=None,
        help="NEW 2026-10-03: fetch REAL per-cell-line numeric "
        "bio-data (get-ceil-line-bio-data) for this cell line, "
        "join compound SMILES via get-substances, write "
        "CSV+meta. Corrects the earlier false claim that "
        "these values were not public.",
    )
    parser.add_argument(
        "--biodata-max-pages",
        type=int,
        default=60,
        help="Max bio-data pages in --biodata-cell-line mode "
        "(server caps 5 rows/page; default 60 = 300 rows)",
    )
    parser.add_argument(
        "--substances-start-page",
        type=int,
        default=1400,
        help="First get-substances page for the structure index "
        "in --biodata-cell-line mode (Pt codes live in the "
        "high, descending pages ~828..1547)",
    )
    parser.add_argument(
        "--substances-max-pages",
        type=int,
        default=147,
        help="Max get-substances pages for the structure index "
        "(default 147 covers pages 1400..1546)",
    )
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
        out_dir = (
            PROJECT_ROOT / out_arg if not out_arg.startswith("/") else Path(out_arg)
        )

    if args.biodata_cell_line:
        args.out_dir = out_dir
        return _run_biodata_mode(args)

    stratified = bool(args.n_target_per_cell and args.cell_lines)
    if args.n_target_per_cell and not args.cell_lines:
        parser.error("--n-target-per-cell requires --cell-lines")
    if args.cell_lines and not args.n_target_per_cell:
        parser.error("--cell-lines requires --n-target-per-cell")
    cell_lines = (
        [cl.strip() for cl in args.cell_lines.split(",") if cl.strip()]
        if args.cell_lines
        else None
    )

    if stratified:
        print(
            f"=== Wf-301C: MB-Finder STRATIFIED fetch ({len(cell_lines)} cell lines) ==="
        )
    else:
        print(
            "=== Wf-295B: MB-Finder real-label loader (Strategy S7, Wf-292D top-1) ==="
        )
    print(f"Source: {MB_API_BASE}")
    if stratified:
        print(
            f"n_target_per_cell: {args.n_target_per_cell}   "
            f"cell_lines: {cell_lines}   start_page: {args.start_page}   "
            f"delay: {args.delay}s"
        )
    else:
        print(
            f"n_target: {args.n_target}   start_page: {args.start_page}   "
            f"delay: {args.delay}s"
        )
    print()

    print("[1/3] Probing MB-Finder aggregate counts...")
    counts = fetch_count()
    print(
        f"      compounds={counts.get('compoundsCount')} "
        f"Pt={counts.get('metalCounts',{}).get('Pt')} "
        f"IC50s={counts.get('ic50Count')} cells={counts.get('cellsCount')}"
    )

    print(f"[2/3] Paginating Pt compounds starting page {args.start_page}...")
    if stratified:
        rows, per_cell_count, total_requests, total_pages_seen, n_non_pt = (
            build_loader_stratified(
                args.n_target_per_cell,
                cell_lines,
                args.start_page,
                args.delay,
                max_pages=args.max_pages,
            )
        )
        print(
            f"      Got {len(rows)} unique Pt compounds in {total_requests} "
            f"API requests ({total_pages_seen} pages, {n_non_pt} non-Pt skipped)"
        )
        print(f"      Per-cell-line counts: {per_cell_count}")
    else:
        rows, total_requests, total_pages_seen, n_non_pt = build_loader(
            args.n_target, args.start_page, args.delay
        )
        per_cell_count = None
        print(
            f"      Got {len(rows)} Pt compounds in {total_requests} API requests "
            f"({total_pages_seen} pages, {n_non_pt} non-Pt skipped)"
        )

    print(f"[3/3] Writing outputs to {out_dir}...")
    n_target_for_meta = args.n_target_per_cell if stratified else args.n_target
    paths = write_outputs(
        rows,
        out_dir,
        total_requests,
        total_pages_seen,
        n_non_pt,
        n_target_for_meta,
        args.start_page,
        args.delay,
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
    print(
        f"n_with_real_label:   {n_real_label} (compounds with >=1 cell-line bio-data)"
    )
    print(f"n_nan_smiles:        {n_nan_smiles}")
    print(
        f"coverage mcf7/a549/a2780/a2780cis: "
        f"{sum(1 for r in rows if r['has_mcf7'])}/"
        f"{sum(1 for r in rows if r['has_a549'])}/"
        f"{sum(1 for r in rows if r['has_a2780'])}/"
        f"{sum(1 for r in rows if r['has_a2780cis'])}"
    )
    if stratified and per_cell_count is not None:
        all_met = all(
            per_cell_count.get(cl, 0) >= args.n_target_per_cell for cl in cell_lines
        )
        print(f"stratified_all_targets_met: {all_met}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
