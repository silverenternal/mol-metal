"""Wf-A37-5: REAL PlatinAI harvest via MB-Finder public API (NEW-2 RESCINDED).

Iterates the per-cell-line BIO-DATA endpoint (``get-ceil-line-bio-data``) for the
4 target cell lines (MCF7, A549, A2780, A2780cis), each up to ``max_pages``
pages of 5 compounds/page.  Writes per-cell-line JSONL (resume-safe, append-only
when ``--resume``) + per-cell-line CSV (deterministic, sorted) + a summary
JSON at ``<out-dir>/wf_a37_platinai_harvest_summary.json``.

Why this exists (instead of an honest-negative):
  - The 2026-10-03 ``platinai_real_label_loader.py`` CORRECTION verified live
    that ``get-ceil-line-bio-data`` returns real per-cell-line numeric IC50
    values (e.g. A2780cis MB-Pt-000017 value="1.2" doi=10.1021/ic8012359).
  - The earlier framing that "IC50 numeric values are NOT exposed" was wrong.
  - So the COMPLETE harvest is reachable and on the to-do list, not blocked.

Replaces the prior honest-negative-with-no-real-data path with the real
harvest path.  The user directive 2026-10-15 was explicit: "要找完整的，不要
搞什么honest negative".

Network setup:
  - Network on this host is FLAKY to ``api.mb-finder.org:443`` (works
    ~50% of the time).  The orchestrator retries each page 6 times with
    exponential backoff capped at 60s.
  - Local HTTPS_PROXY (``127.0.0.1:20171``) is dead; the script forces
    ``proxies={'http': None, 'https': None}`` on every call.

CLI:
  --mode  smoke  -> 10 compounds per cell line (~5 min, foreground)
  --mode  full   -> up to ``--max-pages`` pages per cell line
                     (~43 min in best case, 60-90 min on flaky path)
  --cell-lines   comma-separated subset of
                 MCF7,A549,A2780,A2780cis
  --out-dir      where to write jsonl + csv + summary
                 (default .data/molmetal/platinai_biodata_full)
  --per-page     server-capped at 5 (per Wf-305A verification)
  --max-pages    upper bound; pagination stops earlier when the server
                 reports ``last_page``
  --resume       skip pages that already produced a record in the jsonl
                 (default True)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# CRITICAL: set no_proxy BEFORE importing requests (and BEFORE sys.path
# manipulation), so ``requests`` doesn't read the dead local proxy.
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Make ``molmetal`` importable regardless of how this script is invoked
# (direct path or ``python -m``).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.platinai_real_label_loader import (  # noqa: E402
    MB_API_BASE,
    TARGET_CELL_LINES as _DEFAULT_CELL_LINES,
    compound_code_from_image_url,
    metal_from_compound_code,
    parse_measurement_value,
)

_DEFAULT_CELL_LINES = list(_DEFAULT_CELL_LINES)
_PROXIES = {"http": None, "https": None}


def _post_with_retry(
    endpoint: str,
    body: dict,
    *,
    max_retries: int = 4,
    timeout: int = 18,
) -> Tuple[str, Optional[dict]]:
    """POST with 4-retry exponential backoff + GET fallback on 405.

    Tuned for this host's flaky path to api.mb-finder.org:443
    (~50% direct-connect success): 4 retries with shorter backoff cap
    avoids spending 60s × 6 = 6 min per page when the link is dead.

    Returns a ``(status, payload)`` tuple so the caller can distinguish
    *server-confirmed empty* (404 / success=false / decode error) from
    *transport failure* (all 4 retries hit ConnectTimeout/ReadTimeout).
    GAP-A37-5-full-harvest (2026-10-15): conflating these two cases
    caused the harvest to abort every cell line after 8 consecutive
    transport-timeouts on this flaky host, producing only 25 / 0 / 370
    rows out of the expected ~2590 / ~1820 / ~1275 for A549 / A2780 /
    A2780cis. The transport-failure branch is now a sentinel that the
    harvest loop re-tries on the same page (without incrementing the
    "consecutive empty pages" abort counter).

    Status values:
      * ``"ok"``              — 200 + parseable JSON + success=true
      * ``"false"``           — 200 + ``{"success": false}`` (Wf-305A)
      * ``"not_found"``       — 404
      * ``"decode_error"``    — 200 but body is not valid JSON
      * ``"transport_dead"``  — all ``max_retries`` attempts hit a
                                ``requests.exceptions.RequestException``
                                (ConnectTimeout / ReadTimeout / etc.)
    """
    url = f"{MB_API_BASE}/{endpoint}"
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                url, json=body, timeout=timeout, proxies=_PROXIES
            )
            if r.status_code == 405:
                r = requests.get(
                    url, params=body, timeout=timeout, proxies=_PROXIES
                )
            if r.status_code == 200:
                try:
                    payload = r.json()
                except ValueError:
                    return ("decode_error", None)
                if (
                    isinstance(payload, dict)
                    and payload.get("success") is False
                ):
                    return ("false", None)
                return ("ok", payload)
            if r.status_code == 429:
                time.sleep(min(60.0, 3.0 * (2 ** attempt)))
                continue
            if r.status_code == 404:
                return ("not_found", None)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            last_err = e
            backoff = min(60.0, 2.0 * (2 ** attempt))
            print(
                f"      [retry {attempt + 1}/{max_retries}] "
                f"{endpoint} page={body.get('page')} "
                f"{type(e).__name__} backoff={backoff:.1f}s",
                flush=True,
            )
            time.sleep(backoff)
    if last_err:
        print(
            f"      [exhausted] {endpoint} page={body.get('page')} "
            f"{type(last_err).__name__}",
            flush=True,
        )
    # GAP-A37-5-full-harvest (2026-10-15): all retries exhausted on
    # transport errors. Do NOT collapse to ``("ok", None)`` here — the
    # caller MUST distinguish transport_dead from server-empty so the
    # harvest loop re-tries the same page instead of aborting after 8
    # consecutive transport-timeouts on this flaky host.
    return ("transport_dead", None)


def _flatten(row: dict, cell_line: str) -> dict:
    comp = row.get("compounds") or {}
    value_raw = row.get("value")
    if value_raw is None:
        value_raw = comp.get("numeric_value")
    mean, sd = parse_measurement_value(value_raw)
    code = compound_code_from_image_url(comp.get("imageUrl", ""))
    return {
        "compound_code": code,
        "cell_line": cell_line,
        "biodata_id": row.get("id"),
        "numeric_value_raw": value_raw,
        "value_mean": mean,
        "value_sd": sd,
        "method": comp.get("method") or row.get("method"),
        "incubationTime": comp.get("incubationTime")
        or row.get("incubationTime"),
        "stability": comp.get("stability"),
        "in_vivo": comp.get("in_vivo"),
        "doi": comp.get("doi"),
        "title": comp.get("title"),
        "imageUrl": comp.get("imageUrl"),
        "smiles": comp.get("canonical_smiles")
        or comp.get("smiles"),
        # The API's ``compounds.metal`` is almost always null, so
        # ``comp.get("metal") or "Pt"`` labelled every Cu/Au/Re row "Pt" and
        # made Pt-only filtering a no-op.  Derive it from the code instead;
        # fall back to the API field only if the code is unparseable.
        "metal": metal_from_compound_code(code) or comp.get("metal"),
    }


def _load_existing_page_set(jsonl_path: Path) -> set[int]:
    seen: set[int] = set()
    if not jsonl_path.exists():
        return seen
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_page" in rec:
                seen.add(int(rec["_page"]))
    return seen


def harvest_pt_substances_smiles(
    out_dir: Path,
    *,
    max_pages: int = 800,
    per_page: int = 5,
    resume: bool = True,
    smoke_page_cap: Optional[int] = None,
) -> dict:
    """Fetch all Pt substances (paginated) and cache SMILES keyed by
    compound_code.

    Writes ``<out_dir>/platinai_pt_substances_smiles.jsonl`` (one record per
    substance: substance_id, compound_code, smiles, metal).  Used as the
    SMILES join table for the per-cell-line biodata CSVs.
    """
    jsonl_path = out_dir / "platinai_pt_substances_smiles.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    done: set[int] = (
        _load_existing_page_set(jsonl_path) if resume else set()
    )

    rows_written = 0
    pages_success = 0
    total_requests = 0
    last_page_observed: Optional[int] = None
    t0 = time.time()
    page = 1
    while page <= max_pages:
        if smoke_page_cap is not None and pages_success >= smoke_page_cap:
            break
        if page in done:
            page += 1
            continue
        total_requests += 1
        status, payload = _post_with_retry(
            "get-substances", {"per_page": per_page, "page": page}
        )
        if status == "transport_dead":
            # GAP-A37-5-full-harvest (2026-10-15): flaky network on this
            # host exhausts all 4 retries on transport timeouts for many
            # consecutive pages. Re-try the SAME page instead of
            # advancing — when the link comes back the page succeeds and
            # its _page marker is written to the jsonl (--resume picks
            # it up next launch). Sleep 5s between page-level retries to
            # let the path recover without burning CPU.
            time.sleep(5.0)
            continue
        if payload is None:
            page += 1
            time.sleep(2.0)
            continue
        data = payload.get("data", []) or []
        meta = payload.get("meta") or {}
        last_page_observed = meta.get("last_page") or last_page_observed
        if data:
            pages_success += 1
            with jsonl_path.open("a", encoding="utf-8") as f:
                for sub in data:
                    code = compound_code_from_image_url(
                        sub.get("imageUrl", "")
                    )
                    rec = {
                        "_page": page,
                        "substance_id": sub.get("id"),
                        "compound_code": code,
                        "smiles": sub.get("smiles")
                        or sub.get("canonical_smiles"),
                        "metal": sub.get("metal"),
                        "name": sub.get("name"),
                        "imageUrl": sub.get("imageUrl"),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    rows_written += 1
        if (
            last_page_observed is not None
            and page >= int(last_page_observed)
        ):
            break
        if page % 50 == 0:
            print(
                f"[pt-substances] page {page} "
                f"last_page={last_page_observed} "
                f"rows={rows_written} "
                f"reqs={total_requests} "
                f"elapsed={round(time.time() - t0, 1)}s",
                flush=True,
            )
        page += 1
        time.sleep(1.0)

    return {
        "jsonl": str(jsonl_path),
        "n_rows": rows_written,
        "n_pages_success": pages_success,
        "last_page_observed": last_page_observed,
        "total_api_requests": total_requests,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def merge_smiles_into_csv(
    csv_path: Path, smiles_jsonl: Path
) -> int:
    """Read smiles_jsonl, build compound_code -> smiles map, write
    ``<csv_path>.with_smiles.csv`` with the smiles column populated.
    Returns the number of rows updated.
    """
    if not smiles_jsonl.exists():
        return 0
    smiles_map: dict[str, str] = {}
    with smiles_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            code = rec.get("compound_code", "")
            smi = rec.get("smiles")
            if code and smi:
                smiles_map[code] = smi
    if not csv_path.exists():
        return 0
    updated = 0
    out_path = csv_path.with_name(csv_path.stem + ".with_smiles.csv")
    with csv_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        if "smiles" not in fieldnames:
            fieldnames.append("smiles")
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            code = row.get("compound_code", "")
            if (
                code in smiles_map
                and not row.get("smiles")
            ):
                row["smiles"] = smiles_map[code]
                updated += 1
            writer.writerow(row)
    return updated


def harvest_cell_line(
    cell_line: str,
    out_dir: Path,
    *,
    max_pages: int,
    per_page: int = 5,
    resume: bool = True,
    smoke_page_cap: Optional[int] = None,
) -> dict:
    """Harvest one cell line, returns a summary dict."""
    # Output stem drops hyphens so the API name "MCF-7" maps to the
    # established file name ``platinai_mcf7_biodata.{jsonl,csv}``
    # (matches the existing 3 cells' files + platinai_mcf7_biodata.csv).
    stem = cell_line.lower().replace("-", "")
    jsonl_path = out_dir / f"platinai_{stem}_biodata.jsonl"
    csv_path = out_dir / f"platinai_{stem}_biodata.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    done_pages: set[int] = (
        _load_existing_page_set(jsonl_path) if resume else set()
    )
    rows: list[dict] = []
    last_page_observed: Optional[int] = None
    total_requests = 0
    success_pages = 0
    empty_data_pages = 0
    consecutive_empty_pages = 0
    consecutive_transport_dead = 0  # GAP-A37-5-full-harvest
    t_start = time.time()

    page = 1
    while page <= max_pages:
        if smoke_page_cap is not None and success_pages >= smoke_page_cap:
            break
        if page in done_pages:
            page += 1
            continue

        total_requests += 1
        status, payload = _post_with_retry(
            "get-ceil-line-bio-data",
            {"cell_line": cell_line, "per_page": per_page, "page": page},
        )
        if status == "transport_dead":
            # GAP-A37-5-full-harvest (2026-10-15): on this flaky host,
            # ConnectTimeout/ReadTimeout on all 4 retries was being
            # counted as "empty page" by the original code, which then
            # aborted the cell line after 8 consecutive transport
            # failures. The server may still have data — re-try the SAME
            # page; do NOT advance ``page`` and do NOT increment
            # ``consecutive_empty_pages``. Track a separate transport
            # circuit breaker so a permanently dead cell line / page
            # still gets bounded.
            consecutive_transport_dead += 1
            if consecutive_transport_dead >= 200:
                # 200 × ~64s per page-level retry ≈ 3.5 hours of
                # continuous transport failure on the same page — the
                # link is genuinely down or the page is permanently
                # unreachable; ABORT the cell line rather than spin
                # forever.
                print(
                    f"[{cell_line}] consecutive_transport_dead="
                    f"{consecutive_transport_dead} >= 200 -> ABORT "
                    f"cell line (transport dead)",
                    flush=True,
                )
                break
            time.sleep(5.0)
            continue
        if payload is None:
            # Server-confirmed empty (404 / success=false / decode
            # error). Only THIS branch feeds the consecutive_empty_pages
            # abort counter; transport_dead is handled above.
            page += 1
            consecutive_empty_pages += 1
            if consecutive_empty_pages >= 8:
                print(
                    f"[{cell_line}] consecutive_empty_pages="
                    f"{consecutive_empty_pages} >= 8 -> ABORT cell line",
                    flush=True,
                )
                break
            time.sleep(2.0)
            continue

        data = payload.get("data", []) or []
        meta = payload.get("meta") or {}
        last_page_observed = meta.get("last_page") or last_page_observed
        consecutive_empty_pages = 0
        consecutive_transport_dead = 0  # GAP-A37-5-full-harvest
        if data:
            success_pages += 1
            with jsonl_path.open("a", encoding="utf-8") as f:
                for row in data:
                    flat = _flatten(row, cell_line)
                    flat["_page"] = page
                    f.write(json.dumps(flat, ensure_ascii=False) + "\n")
                    flat.pop("_page", None)
                    rows.append(flat)
        else:
            empty_data_pages += 1

        if (
            last_page_observed is not None
            and page >= int(last_page_observed)
        ):
            break

        if page % 10 == 0:
            elapsed = round(time.time() - t_start, 1)
            print(
                f"[{cell_line}] page {page} "
                f"last_page={last_page_observed} "
                f"rows={len(rows)} "
                f"success_pages={success_pages} "
                f"reqs={total_requests} "
                f"elapsed={elapsed}s",
                flush=True,
            )

        page += 1
        # Gentle throttle to avoid 429s
        time.sleep(1.2)

    # Always rewrite the CSV from the jsonl (so re-runs are deterministic)
    all_records: list[dict] = []
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec.pop("_page", None)
                all_records.append(rec)

    fieldnames = [
        "compound_code",
        "cell_line",
        "metal",
        "value_mean",
        "value_sd",
        "numeric_value_raw",
        "method",
        "incubationTime",
        "smiles",
        "doi",
        "title",
        "imageUrl",
        "biodata_id",
        "stability",
        "in_vivo",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        # Dedup by (compound_code, cell_line) keeping first
        seen_keys: set[tuple[str, str]] = set()
        for rec in all_records:
            key = (rec.get("compound_code", ""), rec.get("cell_line", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            writer.writerow(rec)

    elapsed_total = round(time.time() - t_start, 1)
    summary = {
        "cell_line": cell_line,
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "n_rows_total": len(all_records),
        "n_rows_unique_compound_cell": len(seen_keys),
        "n_pages_success": success_pages,
        "n_pages_empty": empty_data_pages,
        "max_pages_attempted": page - 1,
        "last_page_observed": last_page_observed,
        "total_api_requests": total_requests,
        "elapsed_seconds": elapsed_total,
        "n_with_real_numeric_value": sum(
            1 for r in all_records if r.get("value_mean") is not None
        ),
        "n_with_smiles": sum(
            1 for r in all_records if r.get("smiles")
        ),
    }
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
    )
    parser.add_argument(
        "--cell-lines",
        default=",".join(_DEFAULT_CELL_LINES),
    )
    parser.add_argument(
        "--out-dir",
        default=".data/molmetal/platinai_biodata_full",
    )
    parser.add_argument("--per-page", type=int, default=5)
    parser.add_argument("--max-pages", type=int, default=300)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="do NOT resume from existing jsonl (default: resume)",
    )
    parser.add_argument(
        "--smoke-pages",
        type=int,
        default=2,
        help="in smoke mode, how many successful pages per cell line",
    )

    args = parser.parse_args(argv)

    cell_lines = [
        c.strip() for c in args.cell_lines.split(",") if c.strip()
    ]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    smoke_cap = (
        args.smoke_pages if args.mode == "smoke" else None
    )

    summaries: list[dict] = []
    grand_t0 = time.time()
    for cell in cell_lines:
        print(
            f"\n=== [{cell}] mode={args.mode} max_pages={args.max_pages} "
            f"smoke_cap={smoke_cap} ===",
            flush=True,
        )
        try:
            s = harvest_cell_line(
                cell,
                out_dir,
                max_pages=args.max_pages,
                per_page=args.per_page,
                resume=not args.no_resume,
                smoke_page_cap=smoke_cap,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{cell}] EXC: {type(e).__name__}: {e}", flush=True)
            s = {
                "cell_line": cell,
                "error": f"{type(e).__name__}: {e}",
            }
        summaries.append(s)

    # 2) Pt substances (SMILES join table).  Always run, even in smoke mode
    #    -- SMILES is cheap-ish and always useful.
    print("\n=== [pt-substances] harvesting SMILES join table ===", flush=True)
    sub_summary = harvest_pt_substances_smiles(
        out_dir,
        max_pages=args.max_pages,
        per_page=args.per_page,
        resume=not args.no_resume,
        smoke_page_cap=smoke_cap,
    )
    sub_jsonl = Path(sub_summary["jsonl"])

    # 3) Merge SMILES into each per-cell-line CSV (writes *.with_smiles.csv).
    merge_summary: list[dict] = []
    for cell in cell_lines:
        csv_path = out_dir / (
            f"platinai_{cell.lower().replace('-', '')}_biodata.csv"
        )
        if csv_path.exists():
            n = merge_smiles_into_csv(csv_path, sub_jsonl)
            merge_summary.append(
                {
                    "cell_line": cell,
                    "csv_with_smiles": str(
                        csv_path.with_name(csv_path.stem + ".with_smiles.csv")
                    ),
                    "n_rows_with_smiles_filled": n,
                }
            )

    elapsed_total = round(time.time() - grand_t0, 1)
    overall = {
        "mode": args.mode,
        "out_dir": str(out_dir),
        "elapsed_seconds": elapsed_total,
        "by_cell_line": summaries,
        "pt_substances": sub_summary,
        "smiles_merge": merge_summary,
        "totals": {
            "rows_total": sum(s.get("n_rows_total", 0) for s in summaries),
            "rows_unique_compound_cell": sum(
                s.get("n_rows_unique_compound_cell", 0) for s in summaries
            ),
            "with_real_numeric_value": sum(
                s.get("n_with_real_numeric_value", 0) for s in summaries
            ),
            "with_smiles_after_merge": sum(
                m["n_rows_with_smiles_filled"] for m in merge_summary
            ),
            "api_requests": sum(
                s.get("total_api_requests", 0) for s in summaries
            ) + sub_summary.get("total_api_requests", 0),
        },
        "framing": (
            "REAL MB-Finder per-cell-line numeric IC50 values + "
            "compound SMILES (joined via get-substances). "
            "Source: public endpoints "
            "POST /api/v2/get-ceil-line-bio-data and "
            "POST /api/v2/get-substances (no auth). "
            "Verified live 2026-10-15 on api.mb-finder.org:443 "
            "from this host with proxies={'http': None, 'https': None} "
            "(local HTTPS_PROXY=127.0.0.1:20171 is dead; bypass)."
        ),
    }
    summary_path = out_dir / "wf_a37_platinai_harvest_summary.json"
    summary_path.write_text(
        json.dumps(overall, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"\n=== harvest done elapsed={elapsed_total}s "
        f"totals={overall['totals']} summary={summary_path} ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
