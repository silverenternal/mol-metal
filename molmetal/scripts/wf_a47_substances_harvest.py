"""Wf-A47: enumerate MB-Finder ``get-substances`` to attach SMILES to the
biodata harvest.

Why this exists
---------------
``get-ceil-line-bio-data`` does NOT return a structure.  Its ``compounds``
object carries only ``incubationTime / numeric_value / title / doi / method /
stability / in_vivo / imageUrl``.  So every row harvested by wf_a46 has
``smiles = None`` -- including the pre-existing 4-cell files.  The label and
the molecule live behind two different endpoints.

``get-substances`` is the other half: it returns ``imageId`` (= the same
``MB-Pt-000010`` code wf_a46 derives from ``imageUrl``) plus ``smiles``,
``mw``, ``cLogP``, ``casNumber``, ``weight``.  Joining on that code turns the
label-only harvest into a training-ready dataset.

Server behaviour (measured 2026-10-16/17)
-----------------------------------------
* ``per_page`` is hard-capped at 5, so all 7,711 substances = 1,543 pages.
* **Every filter is ignored.**  ``imageId``, ``id``, ``search`` and ``metal``
  all return the unfiltered first page, so there is no way to ask for just
  the Pt cohort -- the whole list must be walked.
* Same rate limit as biodata: ``X-RateLimit-Limit: 30``/min, with a hard
  IP-wide 429 cooldown if the burst is exceeded.  Request starts are paced
  through wf_a46's bounded token bucket.

Resume
------
Appends one JSONL line per substance as each page lands, each tagged with
``_page``.  A re-run reads the existing file, skips pages already present, and
continues.  The file already holds pages 1-721 (harvested earlier by
``wf_a40_smiles_fetcher.py``), so a fresh run starts at page 722.

CLI
---
  --out-dir      default .data/molmetal/wf_a37_platinai_biodata
  --workers      concurrent fetchers (default 4)
  --rate-per-min request starts per minute (default 12, safely under 30)
  --max-seconds  hard wall-clock budget -> os._exit(3); re-run resumes
  --stall-seconds abort if no page completes within this window
"""
from __future__ import annotations

import argparse
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
from molmetal.scripts.wf_a46_mcf7_harvest import RateLimiter  # noqa: E402

_PROXIES = {"http": None, "https": None}
_SUBSTANCES_URL = f"{MB_API_BASE}/get-substances"

OUT_NAME = "platinai_pt_substances_smiles.jsonl"

# Union schema.  Pages 1-721 were written by wf_a40 with the first six keys
# only; the normalisation pass at the end fills the rest with null so the
# final file is single-schema.
_SUB_FIELDS = [
    "_page",
    "substance_id",
    "compound_code",
    "smiles",
    "metal",
    "name",
    "imageUrl",
    "casNumber",
    "weight",
    "mw",
    "cLogP",
    "referenceCount",
    "ceilLineCount",
    "bioDataCount",
]

_SESSION = requests.Session()
_SESSION.proxies = _PROXIES
_SESSION.mount(
    "https://",
    HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=0),
)

_LIMITER: Optional[RateLimiter] = None
# Bumped on every completed page; a monitor thread aborts the process when it
# goes stale so a wedged SSL call cannot burn the whole budget.
_PROGRESS = {"t": time.time()}
_RATE_LIMITED = {"consecutive": 0}
_RATE_LIMITED_ABORT = 15


def _flatten_substance(row: dict, page: int) -> dict:
    """Map one ``get-substances`` row to the on-disk schema."""
    image_url = row.get("imageUrl") or ""
    code = row.get("imageId")
    if not code:
        # Fall back to the filename stem, same rule the loader uses.
        stem = image_url.rsplit("/", 1)[-1].split("?")[0]
        code = stem[:-4] if stem.lower().endswith(".svg") else stem
    return {
        "_page": page,
        "substance_id": row.get("id"),
        "compound_code": code,
        "smiles": row.get("smiles"),
        "metal": row.get("metal"),
        "name": row.get("name"),
        "imageUrl": image_url,
        "casNumber": row.get("casNumber"),
        "weight": row.get("weight"),
        "mw": row.get("mw"),
        "cLogP": row.get("cLogP"),
        "referenceCount": row.get("referenceCount"),
        "ceilLineCount": row.get("ceilLineCount"),
        "bioDataCount": row.get("bioDataCount"),
    }


def _post_substances(page: int, per_page: int, timeout: int):
    """One paced POST.  Returns ``(status, payload)``.

    status: "ok" | "empty" | "rate_limited" | "transport_dead" | "http_<n>"
    """
    if _LIMITER is not None:
        _LIMITER.wait()
    try:
        r = _SESSION.post(
            _SUBSTANCES_URL,
            json={"per_page": per_page, "page": page},
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
        _RATE_LIMITED["consecutive"] = 0
        return ("ok", payload)
    if r.status_code == 429:
        _RATE_LIMITED["consecutive"] += 1
        if _RATE_LIMITED["consecutive"] >= _RATE_LIMITED_ABORT:
            print(
                f"[wf-a47] RATE-LIMIT cooldown: "
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


def fetch_page(page: int, per_page: int, timeout: int, retries: int = 4):
    """Fetch one page.  ``None`` => retry this page; a list is definitive."""
    for attempt in range(retries):
        status, payload = _post_substances(page, per_page, timeout)
        if status == "ok":
            return payload.get("data") or []
        if status == "empty":
            return []
        if status == "rate_limited":
            time.sleep(15.0 + 5.0 * attempt)
            continue
        time.sleep(1.5 * (attempt + 1))
    return None


def probe_last_page(timeout: int) -> tuple:
    """Return ``(total, last_page)``; retries until the API answers."""
    while True:
        status, payload = _post_substances(1, 5, timeout)
        if status == "ok":
            meta = payload.get("meta") or {}
            return meta.get("total"), meta.get("last_page")
        if status == "rate_limited":
            time.sleep(20.0)
        else:
            time.sleep(4.0)


def load_existing(jsonl_path: Path) -> tuple:
    """Return ``(records, done_pages)`` from the existing JSONL (resume)."""
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


def normalise(jsonl_path: Path, records: list) -> dict:
    """Rewrite the file single-schema and report a coverage summary."""
    by_code: dict = {}
    for rec in records:
        code = rec.get("compound_code")
        if code and code not in by_code:
            by_code[code] = rec
    ordered = sorted(
        by_code.values(), key=lambda r: (r.get("_page") or 0, r.get("substance_id") or 0)
    )
    body = "".join(
        json.dumps({k: r.get(k) for k in _SUB_FIELDS}, ensure_ascii=False) + "\n"
        for r in ordered
    )
    atomic_write_file(jsonl_path, body)

    from collections import Counter

    metals = Counter((r.get("compound_code") or "?")[:6] for r in ordered)
    return {
        "unique_substances": len(ordered),
        "with_smiles": sum(1 for r in ordered if r.get("smiles")),
        "by_prefix": dict(metals.most_common()),
    }


def main(argv: Optional[list] = None) -> int:
    global _LIMITER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", default=".data/molmetal/wf_a37_platinai_biodata"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rate-per-min", type=float, default=12.0)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--per-page", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument(
        "--stall-seconds",
        type=int,
        default=300,
        help="abort (os._exit 3) if no page completes within this many "
        "seconds; catches a wedged worker.",
    )
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=0,
        help="hard wall-clock budget; on expiry os._exit(3). 0 = no budget.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / OUT_NAME
    _LIMITER = RateLimiter(args.rate_per_min)
    t0 = time.time()

    if args.max_seconds:
        def _bail() -> None:
            print(
                f"[wf-a47] WATCHDOG: {args.max_seconds}s budget hit -> "
                f"os._exit(3); re-run resumes",
                flush=True,
            )
            os._exit(3)

        wd = threading.Timer(args.max_seconds, _bail)
        wd.daemon = True
        wd.start()
    else:
        wd = None

    if args.stall_seconds:
        def _stall_monitor() -> None:
            while True:
                time.sleep(30.0)
                if time.time() - _PROGRESS["t"] > args.stall_seconds:
                    print(
                        f"[wf-a47] STALL: no page completed in "
                        f"{args.stall_seconds}s -> os._exit(3); re-run resumes",
                        flush=True,
                    )
                    os._exit(3)

        threading.Thread(target=_stall_monitor, daemon=True).start()

    records, done_pages = load_existing(jsonl_path)
    total, last_page = probe_last_page(args.timeout)
    if not last_page:
        print("[wf-a47] FATAL: meta.last_page unavailable", flush=True)
        return 2

    pending = [p for p in range(1, int(last_page) + 1) if p not in done_pages]
    print(
        f"[wf-a47] resume={len(records)} substances / {len(done_pages)} pages; "
        f"meta.total={total} last_page={last_page} pending={len(pending)}",
        flush=True,
    )

    lock = threading.Lock()
    counters = {"ok": 0, "fail": 0}

    def persist_page(page: int, data: list) -> None:
        with jsonl_path.open("a", encoding="utf-8") as f:
            for row in data:
                f.write(
                    json.dumps(_flatten_substance(row, page), ensure_ascii=False)
                    + "\n"
                )

    for rnd in range(args.max_rounds):
        if not pending:
            break
        still_missing: list = []

        def task(page: int) -> None:
            data = fetch_page(page, args.per_page, args.timeout)
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
                        f"  [wf-a47] ok={counters['ok']} "
                        f"fail={counters['fail']} "
                        f"rate={counters['ok'] / max(1e-9, (time.time() - t0) / 60):.1f}"
                        f" pages/min",
                        flush=True,
                    )

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(task, pending))

        pending = still_missing
        if pending:
            print(
                f"  [wf-a47] round {rnd + 1}: {len(pending)} pages still "
                f"failing; sleep 20s",
                flush=True,
            )
            time.sleep(20.0)

    records, done_pages = load_existing(jsonl_path)
    summary = normalise(jsonl_path, records)
    summary.update(
        {
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "endpoint": f"POST {_SUBSTANCES_URL}",
            "api_meta_total": total,
            "api_meta_last_page": int(last_page),
            "pages_present": len(done_pages),
            "pages_failed_unrecovered": len(pending),
            "wall_seconds": round(time.time() - t0, 1),
        }
    )
    atomic_write_file(
        out_dir / "substances_harvest_summary.json",
        json.dumps(summary, indent=2, ensure_ascii=False),
    )
    if wd is not None:
        wd.cancel()
    print(
        f"\n[wf-a47] DONE substances={summary['unique_substances']} "
        f"with_smiles={summary['with_smiles']} "
        f"pages={len(done_pages)}/{last_page} missing={len(pending)}",
        flush=True,
    )
    return 0 if not pending else 3


if __name__ == "__main__":
    raise SystemExit(main())
