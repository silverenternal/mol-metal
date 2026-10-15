"""Wf-A40-1: SMILES fetcher for the real PlatinAI pool (page-by-page scan).

Scans the MB-Finder ``get-substances`` endpoint starting at page 850 (deep in
the Pt range per Wf-301C scout), filters for imageIds in the real label pool,
and caches SMILES to JSON. Stops when:

    1. Every target imageId is covered (early success), or
    2. ``--max-pages`` reached, or
    3. ``--max-empty-pages`` consecutive pages had ZERO Pt compounds (the
       server has scrolled past the end of the Pt cohort).

Network setup (Wf-A37 verification):
  - Local HTTPS_PROXY (``127.0.0.1:20171``) is dead; bypass with
    ``proxies={'http': None, 'https': None}`` on every call.
  - Set ``os.environ["no_proxy"] = "*"`` BEFORE ``requests`` is imported.

Resume: if the output JSON exists, load it and skip imageIds already cached.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

# CRITICAL: bypass dead local HTTPS_PROXY before ``requests`` is imported.
os.environ["no_proxy"] = "*"
os.environ["NO_PROXY"] = "*"

import requests  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.platinai_real_label_loader import (  # noqa: E402
    MB_API_BASE,
    compound_code_from_image_url,
)
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402


_PROXIES = {"http": None, "https": None}


def _post_get_substances(
    page: int, per_page: int = 5, *, max_retries: int = 4, timeout: int = 18
) -> Optional[list]:
    """POST get-substances one page. Returns the ``data`` list (or ``None`` on
    transport/server failure). Per Wf-A37-5 the server caps ``per_page`` at 5.
    """
    url = f"{MB_API_BASE}/get-substances"
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.post(
                url,
                json={"per_page": per_page, "page": page},
                timeout=timeout,
                proxies=_PROXIES,
            )
            if r.status_code == 200:
                try:
                    payload = r.json()
                except ValueError:
                    return None
                if (
                    isinstance(payload, dict)
                    and payload.get("success") is False
                ):
                    # Wf-305A soft rate-limit: server returns 200 with
                    # success=false. Treat as empty page so the consecutive
                    # empty counter advances; the throttle delay handles it.
                    return []
                return payload.get("data") or []
            if r.status_code == 429:
                time.sleep(min(60.0, 3.0 * (2**attempt)))
                continue
            if r.status_code == 404:
                return []
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            last_err = e
            time.sleep(min(60.0, 2.0 * (2**attempt)))
    if last_err:
        print(
            f"      [exhausted] get-substances page={page} "
            f"{type(last_err).__name__}",
            flush=True,
        )
    return None


def load_target_imageids(csv_path: Path) -> set:
    """Extract unique ``compound_code`` values (= MB-Pt imageIds) from the
    real label pool CSV. The column header is ``compound_code``; the value
    IS the imageId (verified live in Wf-295B).
    """
    out: set = set()
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("compound_code") or "").strip()
            if code:
                out.add(code)
    return out


def load_seed_cache(csv_path: Path) -> Dict[str, str]:
    """Load the Wf-295B seed SMILES (100 rows, imageId -> smiles)."""
    out: Dict[str, str] = {}
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = (row.get("imageId") or "").strip()
            smiles = (row.get("smiles") or "").strip()
            if image_id and smiles:
                out[image_id] = smiles
    return out


def load_existing_cache(out_path: Path) -> Dict[str, str]:
    """Resume support: load existing cache JSON if present."""
    if not out_path.exists():
        return {}
    try:
        with out_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def write_cache(cache: Dict[str, str], out_path: Path) -> None:
    """Atomic write via wf386a3 (AGENTS.md §5.6)."""
    payload = json.dumps(cache, indent=2, ensure_ascii=False)
    atomic_write_file(out_path, payload)


def iter_pages(start_page: int, max_pages: int) -> Iterable[int]:
    """Page numbers to scan: [start_page, start_page + max_pages)."""
    return range(start_page, start_page + max_pages)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wf-A40-1: SMILES fetcher for the real PlatinAI pool "
        "(page-by-page scan over MB-Finder get-substances)."
    )
    parser.add_argument(
        "--target-csv",
        default=".data/molmetal/platinai_4cell_real_label_pool.csv",
        help="CSV with the target compound_codes (= imageIds) to fill in.",
    )
    parser.add_argument(
        "--seed-csv",
        default=".data/molmetal/platinai_4cell_mbfinder.csv",
        help="CSV with the Wf-295B seed SMILES (100 rows).",
    )
    parser.add_argument(
        "--out",
        default=".data/molmetal/platinai_smiles_cache.json",
        help="Output JSON cache (imageId -> smiles).",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=850,
        help="Starting page (default 850, Pt range per Wf-301C scout).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=1500,
        help="Upper bound on pages to scan (default 1500).",
    )
    parser.add_argument(
        "--max-empty-pages",
        type=int,
        default=30,
        help="Abort after this many consecutive pages with NO Pt compound.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Sleep seconds between API calls (server throttles at 30-60s).",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=5,
        help="Server-capped at 5 (default 5 per Wf-305A verification).",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Atomic-write the cache to disk every N pages (default 10).",
    )
    args = parser.parse_args(argv)

    target_csv = Path(args.target_csv)
    seed_csv = Path(args.seed_csv)
    out_path = Path(args.out)

    if not target_csv.exists():
        print(f"[err] target csv not found: {target_csv}", flush=True)
        return 2
    if not seed_csv.exists():
        print(f"[err] seed csv not found: {seed_csv}", flush=True)
        return 2

    target_imageids = load_target_imageids(target_csv)
    seed_cache = load_seed_cache(seed_csv)
    existing_cache = load_existing_cache(out_path)

    cache: Dict[str, str] = dict(seed_cache)
    cache.update(existing_cache)  # existing wins over seed

    missing = target_imageids - set(cache.keys())
    print(
        f"[wf-a40] targets={len(target_imageids)} "
        f"seed={len(seed_cache)} resumed={len(existing_cache)} "
        f"covered={len(target_imageids) - len(missing)} "
        f"missing={len(missing)}",
        flush=True,
    )

    if not missing:
        print("[wf-a40] all targets already covered; nothing to scan.",
              flush=True)
        return 0

    consecutive_empty = 0
    pages_scanned = 0
    pages_with_pt = 0
    new_smiles = 0
    last_last_page: Optional[int] = None
    t0 = time.time()

    for page in iter_pages(args.start_page, args.max_pages):
        data = _post_get_substances(page=page, per_page=args.per_page)
        pages_scanned += 1

        if data is None:
            # Transport failure: don't advance empty counter; sleep + retry
            # path (continue without page increment, but DO sleep the same
            # delay to avoid hammering the dead link).
            time.sleep(args.delay)
            continue

        page_pt_compounds: list = []
        for item in data:
            code = (
                item.get("imageId")
                or compound_code_from_image_url(item.get("imageUrl", ""))
            ).strip()
            if not code:
                continue
            if code in target_imageids:
                smiles = (item.get("smiles") or "").strip()
                if smiles and code not in cache:
                    cache[code] = smiles
                    new_smiles += 1
                page_pt_compounds.append(code)
                if code in missing and code in cache:
                    missing.discard(code)

        if page_pt_compounds:
            pages_with_pt += 1
            consecutive_empty = 0
        else:
            consecutive_empty += 1

        # Flush cache every N pages (resume-safe)
        if pages_scanned % max(1, args.flush_every) == 0:
            write_cache(cache, out_path)

        # Stop conditions
        if not missing:
            print(
                f"[wf-a40] all {len(target_imageids)} targets covered at "
                f"page={page}; stopping early.",
                flush=True,
            )
            break
        if consecutive_empty >= args.max_empty_pages:
            print(
                f"[wf-a40] abort: {consecutive_empty} consecutive empty "
                f"pages (>= {args.max_empty_pages}); Pt cohort ended.",
                flush=True,
            )
            break
        if page_pt_compounds:
            print(
                f"[wf-a40] page={page} pt={len(page_pt_compounds)} "
                f"new_smiles={new_smiles} covered="
                f"{len(target_imageids) - len(missing)}/"
                f"{len(target_imageids)} consecutive_empty="
                f"{consecutive_empty}",
                flush=True,
            )

        time.sleep(args.delay)

    # Final flush
    write_cache(cache, out_path)

    elapsed = round(time.time() - t0, 1)
    print(
        f"[wf-a40] DONE pages_scanned={pages_scanned} pages_with_pt="
        f"{pages_with_pt} new_smiles={new_smiles} "
        f"cache_total={len(cache)} covered="
        f"{len(target_imageids) - len(missing)}/{len(target_imageids)} "
        f"elapsed={elapsed}s -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())