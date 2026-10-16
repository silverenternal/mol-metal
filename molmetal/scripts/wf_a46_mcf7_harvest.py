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
import sys
import threading
import time
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


def harvest_one(cell_line: str, args, watchdog) -> dict:
    """Harvest one cell line; returns its summary dict (``pending`` count
    is reported via ``pages_failed_unrecovered``)."""
    out_dir = Path(args.out_dir)
    stem = cell_line.lower().replace("-", "")
    jsonl_path = out_dir / f"platinai_{stem}_biodata.jsonl"

    records, done_pages = load_existing(jsonl_path)
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

    for rnd in range(args.max_rounds):
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
        "api_meta_last_page": int(last_page),
        "rows_total": len(deduped),
        "unique_compounds": len({r.get("compound_code") for r in deduped}),
        "rows_with_value_mean": sum(
            1 for r in deduped if r.get("value_mean") is not None
        ),
        "rows_with_doi": sum(1 for r in deduped if r.get("doi")),
        "pages_fetched_this_run": counters["ok"],
        "pages_failed_unrecovered": len(pending),
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def main(argv: Optional[list] = None) -> int:
    global _LIMITER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-lines", default=",".join(DEFAULT_CELL_LINES)
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
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [c.strip() for c in args.cell_lines.split(",") if c.strip()]
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

    summaries: list = []
    for cell in cells:
        print(f"\n=== [{cell}] ===", flush=True)
        try:
            summaries.append(harvest_one(cell, args, watchdog))
        except Exception as e:  # noqa: BLE001
            print(f"[{cell}] EXC {type(e).__name__}: {e}", flush=True)
            summaries.append({"cell_line": cell, "error": str(e)})

    rows_total = sum(s.get("rows_total", 0) for s in summaries)
    incomplete = [
        s["cell_line"]
        for s in summaries
        if s.get("pages_failed_unrecovered", 0) or "error" in s
    ]
    combined = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "endpoint": f"POST {_BIODATA_URL}",
        "n_cell_lines": len(summaries),
        "rows_total_all_cells": rows_total,
        "incomplete_cells": incomplete,
        "wall_seconds": round(time.time() - t_start, 1),
        "by_cell_line": summaries,
        "notes": (
            "Pt IC50 measurements harvested from MB-Finder. The Rusanov-2025 "
            "'17,732 IC50' target is a cross-cell-line total (V2 SI: 4,078 "
            "unique Pt complexes, IC50 from 592 cell lines); this run harvests "
            "the largest cell lines until the cumulative count crosses it. "
            "Rows deduped by biodata_id (unique measurement id), NOT "
            "compound_code. Server enforces X-RateLimit-Limit: 30/min."
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
    return 0 if not incomplete else 3


if __name__ == "__main__":
    sys.exit(main())
