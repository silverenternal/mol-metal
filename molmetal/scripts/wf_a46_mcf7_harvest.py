"""Wf-A46: rate-limited concurrent MCF-7 biodata harvest (MB-Finder public API).

Why this exists
---------------
``wf_a37_platinai_full_harvest.py::harvest_cell_line`` is strictly sequential
and assumes a healthy link.  On 2026-10-16 the MB-Finder API:

  * answers only ~33% of requests (the rest ConnectTimeout / ReadTimeout);
  * enforces ``X-RateLimit-Limit: 30`` requests/minute and returns
    ``{"success": false, "message": "You've made too many requests..."}``
    with HTTP 429 once the burst exceeds it.

A sequential pass over MCF-7's 550 pages measured ~1 page / 40 s (~6 h).  A
6-worker unthrottled burst measured 15 ok-pages/min but tripped the 30/min
limit and wedged (39 threads in ``futex_wait``, 24 half-open sockets).

This script therefore *paces* request starts through a token bucket under the
documented 30/min limit, keeps a small worker pool to hide per-request latency,
and wraps the whole fetch in a hard wall-clock watchdog so a wedged SSL call in
one worker cannot freeze the run.

Scope
-----
MCF-7 ONLY.  The other three cells (A549 2,590 / A2780 1,818 / A2780cis 1,273)
are already complete — their existing JSONLs match the API ``meta.total``
exactly.  Row schema comes from ``wf_a37_platinai_full_harvest._flatten`` so all
four files are field-identical.

Dedup key
---------
``biodata_id`` (the unique measurement-row id), NOT ``(compound_code,
cell_line)``.  ``meta.total`` counts *measurements*; one compound legitimately
has many (MB-Pt-003232 alone: 205 A549 / 243 A2780 / 183 A2780cis rows, each a
distinct study with its own value).  Deduping on compound_code would drop ~25%
of the real data.

CLI
---
  --cell-line   API name (default "MCF-7"; "MCF7" returns meta.total=0)
  --out-dir     default .data/molmetal/wf_a37_platinai_biodata
  --workers     concurrent fetchers (default 4)
  --rate-per-min request starts per minute (default 28, under the 30 limit)
  --timeout     per-request seconds (default 15)
  --max-seconds hard wall-clock budget -> os._exit(3) so a re-run resumes
  --max-rounds  retry passes over still-missing pages (default 6)
  --limit-pages optional cap for a smoke run
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

# One shared Session with a pool sized above the worker count.  Creating a fresh
# module-level ``requests.post`` per call churns Sessions/connections and was
# implicated in the 2026-10-16 wedge.
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
    """Token bucket: enforce a minimum interval between request *starts*.

    Serialises start times to ``60 / per_minute`` seconds so a small worker
    pool can overlap in-flight requests without the instantaneous burst
    exceeding the server's 30/min limit.
    """

    def __init__(self, per_minute: float):
        self.interval = 60.0 / max(1.0, per_minute)
        self._lock = threading.Lock()
        self._next_t = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            sleep_for = max(0.0, self._next_t - now)
            self._next_t = max(now, self._next_t) + self.interval
        if sleep_for > 0:
            time.sleep(sleep_for)


_LIMITER: Optional[RateLimiter] = None


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
        return ("transport_dead", None)
    if r.status_code == 200:
        try:
            payload = r.json()
        except ValueError:
            return ("transport_dead", None)
        if payload.get("success") is False:
            return ("rate_limited", payload)
        return ("ok", payload)
    if r.status_code == 429:
        return ("rate_limited", None)
    if r.status_code == 404:
        return ("empty", None)
    return (f"http_{r.status_code}", None)


def fetch_page(
    cell_line: str, page: int, per_page: int, timeout: int, retries: int = 4
) -> Optional[list]:
    """Fetch one page. ``None`` => retry this page; a list (possibly empty)
    is a definitive answer."""
    for attempt in range(retries):
        status, payload = _post_biodata(cell_line, page, per_page, timeout)
        if status == "ok":
            return payload.get("data") or []
        if status == "empty":
            return []
        if status == "rate_limited":
            # Back off hard — the limiter already paces, so a 429 means the
            # window is genuinely saturated.
            time.sleep(15.0 + 5.0 * attempt)
            continue
        # transport_dead / http_5xx
        time.sleep(1.5 * (attempt + 1))
    return None


def probe_last_page(cell_line: str, timeout: int) -> tuple:
    """Return ``(total, last_page)``; retries until the API answers.

    Bounded by the caller's watchdog, so an outage does not abort the run.
    """
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


def write_outputs(out_dir: Path, stem: str, records: list, summary: dict) -> None:
    """Atomic-write the JSONL, a deterministic CSV, and the summary JSON."""
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

    atomic_write_file(
        out_dir / "harvest_summary_2026-10-16.json",
        json.dumps(summary, indent=2, ensure_ascii=False),
    )


def main(argv: Optional[list] = None) -> int:
    global _LIMITER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-line", default="MCF-7")
    parser.add_argument(
        "--out-dir", default=".data/molmetal/wf_a37_platinai_biodata"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rate-per-min", type=float, default=28.0)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--per-page", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--limit-pages", type=int, default=0)
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
    stem = args.cell_line.lower().replace("-", "")
    jsonl_path = out_dir / f"platinai_{stem}_biodata.jsonl"

    _LIMITER = RateLimiter(args.rate_per_min)

    records, done_pages = load_existing(jsonl_path)
    print(
        f"[wf-a46] resume: {len(records)} rows, {len(done_pages)} pages done",
        flush=True,
    )

    t0 = time.time()
    counters = {"ok": 0, "fail": 0}

    # Hard watchdog started before any network call, so a total outage (or a
    # wedged worker) cannot hang the process forever.  Cancelled once the work
    # is done — otherwise a fast no-op re-run still sits until the timer fires
    # and reports a spurious rc=3.
    watchdog = None
    if args.max_seconds:
        def _bail() -> None:
            print(
                f"[wf-a46] WATCHDOG: {args.max_seconds}s budget hit "
                f"ok={counters['ok']} fail={counters['fail']} -> os._exit(3), "
                f"re-run resumes",
                flush=True,
            )
            os._exit(3)

        watchdog = threading.Timer(args.max_seconds, _bail)
        watchdog.daemon = True
        watchdog.start()

    total, last_page = probe_last_page(args.cell_line, args.timeout)
    if not last_page:
        print("[wf-a46] FATAL: meta.last_page unavailable", flush=True)
        return 2
    if args.limit_pages:
        last_page = min(int(last_page), args.limit_pages)
    print(
        f"[wf-a46] cell={args.cell_line} meta.total={total} "
        f"last_page={last_page} workers={args.workers} "
        f"rate={args.rate_per_min}/min",
        flush=True,
    )

    pending = [p for p in range(1, int(last_page) + 1) if p not in done_pages]
    print(f"[wf-a46] pending pages: {len(pending)}", flush=True)

    lock = threading.Lock()

    def persist_page(page: int, data: list) -> None:
        """Append one page's rows (resume-safe). Called under ``lock``."""
        with jsonl_path.open("a", encoding="utf-8") as f:
            for row in data:
                flat = _flatten(row, args.cell_line)
                flat["_page"] = page
                f.write(json.dumps(flat, ensure_ascii=False) + "\n")

    for rnd in range(args.max_rounds):
        if not pending:
            break
        print(
            f"[wf-a46] round {rnd + 1}: {len(pending)} pages, "
            f"{args.workers} workers",
            flush=True,
        )
        still_missing: list = []

        def task(page: int) -> None:
            data = fetch_page(
                args.cell_line, page, args.per_page, args.timeout
            )
            with lock:
                if data is None:
                    counters["fail"] += 1
                    still_missing.append(page)
                    return
                counters["ok"] += 1
                persist_page(page, data)
                if counters["ok"] % 25 == 0:
                    rate = counters["ok"] / max(
                        1e-9, (time.time() - t0) / 60.0
                    )
                    print(
                        f"[wf-a46] ok={counters['ok']} "
                        f"fail={counters['fail']} rate={rate:.1f} pages/min",
                        flush=True,
                    )

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(task, pending))

        pending = still_missing
        if pending:
            print(
                f"[wf-a46] round {rnd + 1} done: {len(pending)} pages still "
                f"failing; sleeping 20s before retry",
                flush=True,
            )
            time.sleep(20.0)

    # Re-read from disk (resumed + appended), dedup by biodata_id, keep page
    # order.  See module docstring for why the key is not compound_code.
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

    elapsed = round(time.time() - t0, 1)
    summary = {
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "endpoint": f"POST {_BIODATA_URL}",
        "cell_line": args.cell_line,
        "per_page": args.per_page,
        "api_meta_total": total,
        "api_meta_last_page": int(last_page),
        "pages_fetched_this_run": counters["ok"],
        "pages_on_disk": len({r.get("_page") for r in deduped}),
        "pages_failed_unrecovered": len(pending),
        "rows_total": len(deduped),
        "unique_compounds": len({r.get("compound_code") for r in deduped}),
        "rows_with_value_mean": sum(
            1 for r in deduped if r.get("value_mean") is not None
        ),
        "rows_with_smiles": sum(1 for r in deduped if r.get("smiles")),
        "rows_with_doi": sum(1 for r in deduped if r.get("doi")),
        "wall_seconds": elapsed,
        "notes": (
            "MCF-7 only. Cell-line spelling 'MCF-7' (hyphen) is required; "
            "'MCF7' returns meta.total=0. Other 3 cells were already complete "
            "and are untouched. Rows deduped by biodata_id (unique measurement "
            "id), NOT compound_code. Server enforces X-RateLimit-Limit: 30/min."
        ),
    }
    write_outputs(out_dir, stem, deduped, summary)
    if watchdog is not None:
        watchdog.cancel()
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    # 0 = every page fetched; 3 = pages still missing (caller should re-run
    # after a cooldown; the JSONL resumes).
    return 0 if not pending else 3


if __name__ == "__main__":
    sys.exit(main())
