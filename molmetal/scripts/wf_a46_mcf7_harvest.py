"""Wf-A46: rate-limited multi-cell-line biodata harvest (MB-Finder public API).

Why this exists
---------------
The Rusanov-2025 "17,732 IC50 values" figure is a *cross-cell-line* total, not
a 4-cell total: their V2 SI ("Data curation", p. S27) states "a labelled
dataset of 4,078 structurally unique platinum complexes with cytotoxicity data
... IC50 values from 592 cell lines".  ``GET /api/v2/get-count`` agrees:
``{"compoundsCount":7620,"cellsCount":832,"ic50Count":32022,
"metalCounts":{"Cu":2676,"Pt":3614,"Au":1218,"Re":144}}``.

The 4 modelled cells (A549/MCF-7/A2780/A2780cis) are only 8,428 of the 32,022
rows.  This script harvests the largest remaining cell lines until the
cumulative Pt IC50 count crosses 17,732.

Server behaviour (measured 2026-10-16)
--------------------------------------
* ``per_page`` is hard-capped at 5 (requesting 500 still returns 5 rows).
* ``X-RateLimit-Limit: 30`` requests/minute; a 6-worker burst trips a
  cooldown that then 429s even 25 s-spaced requests.  Request *starts* are
  therefore paced through a token bucket under that limit.
* The link is flaky: ConnectTimeout / ReadTimeout are common, and a wedged
  SSL call can sit in ``futex_wait`` forever.  Every page is appended to the
  JSONL as it arrives, and a hard wall-clock watchdog calls ``os._exit(3)``
  so a re-run resumes from the ``_page`` markers.

Dedup key is ``biodata_id`` (unique measurement id), NOT ``compound_code``:
``meta.total`` counts measurements and one compound legitimately has many
(MB-Pt-003232 alone: 205 A549 rows).

CLI
---
  --cell-lines  comma-separated API names (default: the 17-cell list that
                crosses 17,732).  Hyphens matter: "MCF7" -> total=0.
  --out-dir     default .data/molmetal/wf_a37_platinai_biodata
  --workers     concurrent fetchers (default 4)
  --rate-per-min request starts per minute (default 24, under the 30 limit)
  --max-seconds hard wall-clock budget -> os._exit(3); re-run resumes
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

# Bypass the (dead-for-this-host) local proxy BEFORE requests is imported.
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

import requests  # noqa: E402
from requests.adapters import HTTPAdapter  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.platinai_real_label_loader import (  # noqa: E402
    MB_API_BASE,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402
from molmetal.scripts.wf_a37_platinai_full_harvest import _flatten  # noqa: E402

_PROXIES = {"http": None, "https": None}
_BIODATA_URL = f"{MB_API_BASE}/get-ceil-line-bio-data"

# Largest remaining cell lines by ``meta.total`` (measured 2026-10-16).  The
# first four are already complete on disk; the rest take the cumulative count
# from 8,428 past 17,732.
DEFAULT_CELL_LINES = [
    "MCF-7",
    "A549",
    "A2780",
    "A2780cis",
    "HeLa",
    "HCT-116",
    "Hep-G2",
    "MDA-MB-231",
    "HT-29",
    "SW480",
    "SK-OV-3",
    "CH1",
    "HL-60",
    "MRC-5",
    "NCI-H460",
    "HCT-15",
    "PC-3",
    "HEK-293",
    "A375",
    "L1210",
    "HL-7702",
]

# One shared Session with a pool sized above the worker count.
_SESSION = requests.Session()
_SESSION.proxies = _PROXIES
_SESSION.mount(
    "https://",
    HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0),
)

_CSV_FIELDS = [
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


class RateLimiter:
    """Bounded token bucket pacing request *starts* under the server limit.

    A naive "next allowed time += interval" limiter accumulates unbounded debt
    when callers arrive faster than the rate: on 2026-10-16 fast-failing
    transport retries pushed the schedule ~20 min into the future and every
    worker sat in ``time.sleep()`` with zero progress.  This bucket caps the
    burst at ``capacity`` tokens and never drifts past the refill rate.
    """

    def __init__(self, per_minute: float, capacity: int = 2):
        self.rate = max(1.0, per_minute) / 60.0
        self.capacity = float(max(1, capacity))
        self._tokens = float(max(1, capacity))
        self._last = time.time()
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                self._tokens = min(
                    self.capacity,
                    self._tokens + (now - self._last) * self.rate,
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                need = (1.0 - self._tokens) / self.rate
            time.sleep(min(max(need, 0.01), 1.0))


_LIMITER: Optional[RateLimiter] = None
# Updated on every completed page (ok or fail); a monitor thread aborts the
# process if it goes stale, so a wedged worker cannot burn a whole budget.
_PROGRESS = {"t": time.time()}
# Consecutive 429s.  MB-Finder imposes a cooldown that 429s even 25 s-spaced
# requests for minutes at a time; once we are clearly inside one, bail out so
# the caller can back off instead of burning ~90 s per page.
_RATE_LIMITED = {"consecutive": 0}
_RATE_LIMITED_ABORT = 15


def _post_biodata(cell_line: str, page: int, per_page: int, timeout: int):
    """One paced POST. Returns ``(status, payload)``.

    status: "ok" | "empty" | "rate_limited" | "transport_dead" | "http_<n>"
    """
    if _LIMITER is not None:
        _LIMITER.wait()
    try:
        r = _SESSION.post(
            _BIODATA_URL,
            json={"cell_line": cell_line, "per_page": per_page, "page": page},
            timeout=timeout,
        )
    except (requests.exceptions.RequestException, ValueError):
        _PROGRESS["t"] = time.time()
        return ("transport_dead", None)
    _PROGRESS["t"] = time.time()
    if r.status_code == 200:
        try:
            payload = r.json()
        except ValueError:
            return ("transport_dead", None)
        if payload.get("success") is False:
            _RATE_LIMITED["consecutive"] += 1
            if _RATE_LIMITED["consecutive"] >= _RATE_LIMITED_ABORT:
                print(
                    f"[wf-a46] RATE-LIMIT cooldown: "
                    f"{_RATE_LIMITED['consecutive']} consecutive 429 -> "
                    f"os._exit(3); re-run resumes",
                    flush=True,
                )
                os._exit(3)
            return ("rate_limited", payload)
        _RATE_LIMITED["consecutive"] = 0
        return ("ok", payload)
    if r.status_code == 429:
        _RATE_LIMITED["consecutive"] += 1
        if _RATE_LIMITED["consecutive"] >= _RATE_LIMITED_ABORT:
            print(
                f"[wf-a46] RATE-LIMIT cooldown: "
                f"{_RATE_LIMITED['consecutive']} consecutive 429 -> "
                f"os._exit(3); re-run resumes",
                flush=True,
            )
            os._exit(3)
        return ("rate_limited", None)
    _RATE_LIMITED["consecutive"] = 0
    if r.status_code == 404:
        return ("empty", None)
    return (f"http_{r.status_code}", None)


def fetch_page(
    cell_line: str, page: int, per_page: int, timeout: int, retries: int = 4
) -> Optional[list]:
    """Fetch one page. ``None`` => retry this page; a list is definitive."""
    for attempt in range(retries):
        status, payload = _post_biodata(cell_line, page, per_page, timeout)
        if status == "ok":
            return payload.get("data") or []
        if status == "empty":
            return []
        if status == "rate_limited":
            time.sleep(15.0 + 5.0 * attempt)
            continue
        time.sleep(1.5 * (attempt + 1))
    return None


def probe_last_page(cell_line: str, timeout: int) -> tuple:
    """Return ``(total, last_page)``; retries until the API answers."""
    while True:
        status, payload = _post_biodata(cell_line, 1, 5, timeout)
        if status == "ok":
            meta = payload.get("meta") or {}
            return meta.get("total"), meta.get("last_page")
        if status == "rate_limited":
            time.sleep(20.0)
        else:
            time.sleep(4.0)


def load_existing(jsonl_path: Path) -> tuple:
    """Return ``(records, done_pages)`` from an existing JSONL (resume)."""
    records: list = []
    done: set = set()
    if not jsonl_path.exists():
        return records, done
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(rec)
            if "_page" in rec:
                done.add(int(rec["_page"]))
    return records, done


def write_cell_outputs(out_dir: Path, stem: str, records: list) -> None:
    """Atomic-write one cell line's JSONL + deterministic CSV."""
    jsonl = "".join(
        json.dumps(r, ensure_ascii=False) + "\n"
        for r in sorted(records, key=lambda r: r.get("_page", 0))
    )
    atomic_write_file(out_dir / f"platinai_{stem}_biodata.jsonl", jsonl)

    csv_path = out_dir / f"platinai_{stem}_biodata.csv"
    tmp_csv = out_dir / f".platinai_{stem}_biodata.csv.tmp"
    with tmp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        seen: set = set()
        for rec in records:
            key = rec.get("biodata_id")
            if key in seen:
                continue
            seen.add(key)
            writer.writerow(rec)
    os.replace(tmp_csv, csv_path)


def harvest_one(
    cell_line: str,
    args,
    watchdog,
    prev: dict | None = None,
    stem: str | None = None,
) -> dict:
    """Harvest one cell line; returns its summary dict (``pending`` count
    is reported via ``pages_failed_unrecovered``).

    ``stem`` overrides the output filename slug.  The default rule
    (``lower().replace("-", "")``) maps ``K-562`` and ``K562`` to the same
    file, so ``--all-cells`` passes a collision-free stem.

    With ``args.from_disk`` no network call is made: the API meta is carried
    forward from ``prev`` (the previous summary) and only the on-disk records
    are re-aggregated.  This exists so the summary can be regenerated after a
    schema fix while another crawl is saturating the API's rate limit.
    """
    out_dir = Path(args.out_dir)
    if stem is None:
        stem = _slug(cell_line)
    jsonl_path = out_dir / f"platinai_{stem}_biodata.jsonl"

    records, done_pages = load_existing(jsonl_path)
    if args.from_disk:
        if not jsonl_path.exists():
            print(f"[{cell_line}] SKIP (no file on disk)", flush=True)
            return {"cell_line": cell_line, "error": "no file on disk"}
        total = (prev or {}).get("api_meta_total") or len(records)
        # Cells harvested after the last summary write have no prev entry, so
        # api_meta_last_page is absent.  Fall back to the pages actually on
        # disk -- otherwise int(None) raises *after* the DONE print and the
        # cell is recorded as an error, silently dropping its rows from the
        # aggregate (observed: 25,646 rows on disk -> 18,090 in the summary).
        last_page = (prev or {}).get("api_meta_last_page") or (
            max(done_pages) if done_pages else 0
        )
        pending: list = []
        if last_page:
            pending = [
                p for p in range(1, int(last_page) + 1) if p not in done_pages
            ]
    else:
        total, last_page = probe_last_page(cell_line, args.timeout)
        if not last_page:
            print(f"[{cell_line}] FATAL: meta.last_page unavailable", flush=True)
            return {"cell_line": cell_line, "error": "no meta.last_page"}
        pending = [p for p in range(1, int(last_page) + 1) if p not in done_pages]
    print(
        f"[{cell_line}] resume={len(records)} rows / {len(done_pages)} pages; "
        f"meta.total={total} last_page={last_page} pending={len(pending)}",
        flush=True,
    )

    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0}
    t0 = time.time()
    _PROGRESS["t"] = time.time()

    def persist_page(page: int, data: list) -> None:
        with jsonl_path.open("a", encoding="utf-8") as f:
            for row in data:
                flat = _flatten(row, cell_line)
                flat["_page"] = page
                f.write(json.dumps(flat, ensure_ascii=False) + "\n")

    for rnd in range(0 if args.from_disk else args.max_rounds):
        if not pending:
            break
        still_missing: list = []

        def task(page: int) -> None:
            data = fetch_page(cell_line, page, args.per_page, args.timeout)
            with lock:
                _PROGRESS["t"] = time.time()
                if data is None:
                    counters["fail"] += 1
                    still_missing.append(page)
                    return
                counters["ok"] += 1
                persist_page(page, data)
                if counters["ok"] % 50 == 0:
                    print(
                        f"  [{cell_line}] ok={counters['ok']} "
                        f"fail={counters['fail']}",
                        flush=True,
                    )

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(task, pending))

        pending = still_missing
        if pending:
            print(
                f"  [{cell_line}] round {rnd + 1}: {len(pending)} pages "
                f"still failing; sleep 20s",
                flush=True,
            )
            time.sleep(20.0)

    records, _ = load_existing(jsonl_path)
    records.sort(key=lambda r: (r.get("_page", 0),))
    seen: set = set()
    deduped: list = []
    for rec in records:
        key = rec.get("biodata_id")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(rec)

    write_cell_outputs(out_dir, stem, deduped)
    print(
        f"[{cell_line}] DONE rows={len(deduped)} "
        f"(meta.total={total}) pages_missing={len(pending)} "
        f"elapsed={round(time.time() - t0, 1)}s",
        flush=True,
    )
    return {
        "cell_line": cell_line,
        "api_meta_total": total,
        "api_meta_last_page": int(last_page or 0),
        "rows_total": len(deduped),
        "rows_by_metal": dict(
            Counter(r.get("metal") or "?" for r in deduped).most_common()
        ),
        "rows_pt": sum(1 for r in deduped if r.get("metal") == "Pt"),
        "pt_unique_compounds": len(
            {r.get("compound_code") for r in deduped if r.get("metal") == "Pt"}
        ),
        "unique_compounds": len({r.get("compound_code") for r in deduped}),
        "rows_with_value_mean": sum(
            1 for r in deduped if r.get("value_mean") is not None
        ),
        "rows_with_doi": sum(1 for r in deduped if r.get("doi")),
        "pages_fetched_this_run": counters["ok"],
        "pages_failed_unrecovered": len(pending),
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def _slug(cell_line: str) -> str:
    """Filesystem-safe stem that preserves the historical names.

    The old rule was ``lower().replace("-", "")``.  It is kept for every name
    that survives it unchanged, so existing files still resume; anything else
    (``/``, ``(``, spaces, ``*``) collapses to ``_``.  Without this,
    ``BALB/3T3`` resolves to a non-existent ``platinai_balb/3t3_biodata.jsonl``.
    """
    base = cell_line.lower().replace("-", "")
    base = re.sub(r"[^a-z0-9._]+", "_", base).strip("_")
    return base or "cell"


def on_disk_cells(out_dir: Path) -> tuple[list, dict, dict]:
    """Cell lines already on disk, with the stem each file uses.

    Scanning the JSONL rather than trusting the summary keeps the rebuild
    honest after a killed run: ``harvest_one`` writes a cell's file when the
    cell finishes, but the summary is only written at the end of ``main``, so
    an ``os._exit(3)`` leaves completed cells absent from the summary.
    """
    harvested: dict = {}
    rows_by_cell: dict = {}
    for p in sorted(out_dir.glob("platinai_*_biodata.jsonl")):
        try:
            n = 0
            cell = None
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    n += 1
                    if cell is None:
                        cell = json.loads(line)["cell_line"]
        except (json.JSONDecodeError, OSError, KeyError):
            continue
        if cell:
            harvested[cell] = p.name
            rows_by_cell[cell] = n
    stem_by_cell = {
        cell: name[len("platinai_") : -len("_biodata.jsonl")]
        for cell, name in harvested.items()
    }
    return sorted(harvested), stem_by_cell, rows_by_cell


def resolve_all_cells(out_dir: Path) -> tuple[list, dict]:
    """All API cell lines not yet *complete* on disk, largest first.

    Returns ``(ordered_cell_lines, stem_by_cell)``.  Stems must be made
    collision-proof: the default ``lower().replace("-", "")`` slug maps
    ``K-562``/``K562``, ``PC-3``/``PC3`` and ``SK-OV-3``/``SKOV3`` to the same
    filename, so harvesting both would let the second overwrite the first.
    Cells already on disk keep their existing stem (so a re-run resumes); a
    later collision gets ``_<api id>`` appended.

    A cell with a file is re-queued when the file holds fewer rows than
    ``bioDataCount * 0.9``: a killed run leaves a partial file behind, and
    treating file-existence as completeness would skip it forever.
    """
    registry = Path("logs/wf_a46_celllines.json")
    if not registry.exists():
        raise SystemExit(f"[wf-a46] missing {registry}; cannot enumerate cells")
    cells = json.loads(registry.read_text(encoding="utf-8"))

    harvested_cells, stem_by_cell, rows_by_cell = on_disk_cells(out_dir)
    harvested = set(harvested_cells)
    used: set = set(stem_by_cell.values())

    # The registry's bioDataCount is NOT the endpoint's meta.total: it runs
    # systematically higher (HL-60 465 vs 400 returned; MCF-7 2877 vs 2747),
    # so the 0.9 row-count heuristic re-queues cells that are already
    # complete.  The last summary records each cell's real api_meta_total and
    # whether any page failed, so trust that when it is available.
    complete_from_summary: set = set()
    summary_path = out_dir / "harvest_summary_2026-10-16.json"
    if summary_path.exists():
        try:
            doc = json.loads(summary_path.read_text(encoding="utf-8"))
            for s in doc.get("by_cell_line", []):
                if (
                    s.get("cell_line")
                    and "error" not in s
                    and not s.get("pages_failed_unrecovered", 0)
                    and s.get("rows_total") is not None
                    and s.get("rows_total") == s.get("api_meta_total")
                ):
                    complete_from_summary.add(s["cell_line"])
        except (json.JSONDecodeError, OSError):
            pass

    remaining = []
    for c in cells:
        cell = c.get("cellLine")
        if cell not in harvested:
            remaining.append(c)
            continue
        if cell in complete_from_summary:
            continue
        expected = c.get("bioDataCount") or 0
        if expected and rows_by_cell.get(cell, 0) < expected * 0.9:
            remaining.append(c)

    remaining.sort(key=lambda c: (-(c.get("bioDataCount") or 0), c.get("id") or 0))
    for c in remaining:
        cell = c["cellLine"]
        if cell in stem_by_cell:
            continue
        base = _slug(cell)
        stem = base if base not in used else f"{base}_{c.get('id')}"
        stem_by_cell[cell] = stem
        used.add(stem)

    return [c["cellLine"] for c in remaining], stem_by_cell


def main(argv: Optional[list] = None) -> int:
    global _LIMITER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-lines", default=",".join(DEFAULT_CELL_LINES)
    )
    parser.add_argument(
        "--all-cells",
        action="store_true",
        help="harvest every API cell line not yet on disk, largest "
        "bioDataCount first (reads logs/wf_a46_celllines.json)",
    )
    parser.add_argument(
        "--pt-target",
        type=int,
        default=0,
        help="stop once cumulative Pt rows (on-disk + this run) reach N; "
        "0 = no stop",
    )
    parser.add_argument(
        "--out-dir", default=".data/molmetal/wf_a37_platinai_biodata"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rate-per-min", type=float, default=24.0)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--per-page", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument(
        "--stall-seconds",
        type=int,
        default=300,
        help="abort (os._exit 3) if no page completes within this many "
        "seconds; catches a wedged worker that the wall-clock budget would "
        "otherwise only catch much later.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=0,
        help="hard wall-clock budget; on expiry os._exit(3) so a wedged worker "
        "cannot freeze the run. Rows are appended per page, so a re-run "
        "resumes. 0 = no budget.",
    )
    parser.add_argument(
        "--from-disk",
        action="store_true",
        help="no network: re-aggregate the on-disk records and rewrite the "
        "summary, carrying the API meta forward from the existing summary. "
        "Use to regenerate the summary after a schema fix without competing "
        "for the API's rate limit.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.from_disk:
        cells, stem_by_cell, _ = on_disk_cells(out_dir)
        print(
            f"[wf-a46] --from-disk: {len(cells)} cell line(s) on disk",
            flush=True,
        )
    elif args.all_cells:
        cells, stem_by_cell = resolve_all_cells(out_dir)
        print(
            f"[wf-a46] --all-cells: {len(cells)} cell line(s) not yet on disk",
            flush=True,
        )
    else:
        cells = [c.strip() for c in args.cell_lines.split(",") if c.strip()]
        stem_by_cell = {}
    _LIMITER = RateLimiter(args.rate_per_min)

    t_start = time.time()

    if args.max_seconds:
        def _bail() -> None:
            print(
                f"[wf-a46] WATCHDOG: {args.max_seconds}s budget hit -> "
                f"os._exit(3); re-run resumes",
                flush=True,
            )
            os._exit(3)

        watchdog = threading.Timer(args.max_seconds, _bail)
        watchdog.daemon = True
        watchdog.start()
    else:
        watchdog = None

    if args.stall_seconds:
        def _stall_monitor() -> None:
            while True:
                time.sleep(30.0)
                if time.time() - _PROGRESS["t"] > args.stall_seconds:
                    print(
                        f"[wf-a46] STALL: no page completed in "
                        f"{args.stall_seconds}s -> os._exit(3); re-run resumes",
                        flush=True,
                    )
                    os._exit(3)

        threading.Thread(target=_stall_monitor, daemon=True).start()

    prev_by_cell: dict = {}
    prev_path = out_dir / "harvest_summary_2026-10-16.json"
    if prev_path.exists():
        try:
            prev_doc = json.loads(prev_path.read_text(encoding="utf-8"))
            prev_by_cell = {
                s["cell_line"]: s
                for s in prev_doc.get("by_cell_line", [])
                if s.get("cell_line")
            }
        except (json.JSONDecodeError, OSError):
            prev_by_cell = {}

    summaries: list = []
    processed: set = set()

    def _pt_total() -> int:
        cur = sum(s.get("rows_pt", 0) for s in summaries)
        prev = sum(
            s.get("rows_pt", 0)
            for c, s in prev_by_cell.items()
            if c not in processed
        )
        return cur + prev

    target_reached = False
    for cell in cells:
        if args.pt_target and _pt_total() >= args.pt_target:
            print(
                f"[wf-a46] PT TARGET reached ({_pt_total()} >= "
                f"{args.pt_target}); stopping before [{cell}]",
                flush=True,
            )
            target_reached = True
            break
        print(f"\n=== [{cell}] ===", flush=True)
        try:
            summaries.append(
                harvest_one(
                    cell,
                    args,
                    watchdog,
                    prev_by_cell.get(cell),
                    stem=stem_by_cell.get(cell),
                )
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{cell}] EXC {type(e).__name__}: {e}", flush=True)
            summaries.append({"cell_line": cell, "error": str(e)})
        processed.add(cell)

    # Merge the previous summary's entries for cells not touched this run (and
    # for cells that errored, whose on-disk rows are still valid), so the
    # rewritten summary always covers everything on disk.
    merged: dict = {}
    for s in summaries:
        cell = s.get("cell_line")
        if cell and "error" not in s:
            merged[cell] = s
        elif cell and cell in prev_by_cell and "error" not in prev_by_cell[cell]:
            merged[cell] = prev_by_cell[cell]
        elif cell:
            merged[cell] = s
    for cell, s in prev_by_cell.items():
        merged.setdefault(cell, s)
    summaries = sorted(
        merged.values(), key=lambda s: -(s.get("rows_total") or 0)
    )

    rows_total = sum(s.get("rows_total", 0) for s in summaries)
    incomplete = [
        s["cell_line"]
        for s in summaries
        if s.get("pages_failed_unrecovered", 0) or "error" in s
    ]
    metals: Counter = Counter()
    for s in summaries:
        for m, c in (s.get("rows_by_metal") or {}).items():
            metals[m] += c
    rows_pt = sum(s.get("rows_pt", 0) for s in summaries)

    # The label harvest carries no structures; wf_a47 fetches them and wf_a49
    # joins them.  Fold that coverage in when the artefacts are present so this
    # single summary answers "how many rows are training-ready".
    structures = None
    substances_path = out_dir / "substances_harvest_summary.json"
    join_path = out_dir / "pt" / "smiles_join_summary.json"
    if substances_path.exists() and join_path.exists():
        try:
            substances = json.loads(substances_path.read_text(encoding="utf-8"))
            join = json.loads(join_path.read_text(encoding="utf-8"))
            structures = {
                "substances_with_smiles": substances.get("with_smiles"),
                "pt_compounds": join.get("pt_compounds"),
                "pt_compounds_with_smiles": join.get("pt_compounds_with_smiles"),
                "pt_rows": join.get("pt_rows"),
                "pt_rows_with_smiles": join.get("pt_rows_with_smiles"),
                "pt_rows_with_value_and_smiles": join.get(
                    "pt_rows_with_value_and_smiles"
                ),
                "source": "wf_a47 get-substances + wf_a49 join",
            }
        except (json.JSONDecodeError, OSError):
            structures = None

    combined = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "endpoint": f"POST {_BIODATA_URL}",
        "n_cell_lines": len(summaries),
        "rows_total_all_cells": rows_total,
        "rows_by_metal_all_cells": dict(metals.most_common()),
        "rows_pt_all_cells": rows_pt,
        "incomplete_cells": incomplete,
        "wall_seconds": round(time.time() - t_start, 1),
        "by_cell_line": summaries,
        "structures": structures,
        "notes": (
            "IC50 measurements harvested from MB-Finder across the largest cell "
            "lines. NOT all platinum: the API returns no metal field, so the "
            "metal is derived from the MB-<Metal>- compound code. The "
            "Rusanov-2025 '17,732 IC50' figure is a cross-cell-line Pt total "
            "(V2 SI: 4,078 unique Pt complexes, IC50 from 592 cell lines), and "
            "is NOT reachable from this API -- 21 of 832 cell lines already "
            "cover 91% of MB-Finder's 3,614 Pt compounds. Rows deduped by "
            "biodata_id (unique measurement id), NOT compound_code. Structures "
            "are absent here; join get-substances on compound_code (wf_a49). "
            "Server enforces X-RateLimit-Limit: 30/min."
        ),
    }
    atomic_write_file(
        out_dir / "harvest_summary_2026-10-16.json",
        json.dumps(combined, indent=2, ensure_ascii=False),
    )
    if watchdog is not None:
        watchdog.cancel()
    print(
        f"\n[wf-a46] ALL DONE rows_total={rows_total} "
        f"incomplete={incomplete}",
        flush=True,
    )
    # A reached target is a successful stop, not a partial run: without this
    # an old error entry keeps rc=3 and the outer retry loop re-runs, hits the
    # target again immediately, and sleeps for nothing.
    return 0 if target_reached or not incomplete else 3


if __name__ == "__main__":
    sys.exit(main())
