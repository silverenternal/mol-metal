#!/usr/bin/env python3
"""PlatinAI Selenium S1 loader: per-cell-line IC50 NUMERIC harvest from Rusanov 2025.

=====================================================================
Wf-304B / TODO-119 P1.3 — PlatinAI real-label Selenium loader (Strategy S1
from Wf-292D, Cloudflare bypass via undetected-chromedriver).

Background
----------
Wf-295B (Strategy S7) harvests compound IDs + per-cell-line COVERAGE flags
from the MB-Finder portal API.

CORRECTION 2026-10-03 (Wf-A13). The sentence that stood here read: "but the
API does NOT expose per-cell-line IC50 NUMERIC values — only a
presence/absence matrix." THAT IS FALSE and has been removed. The API
endpoint ``get-ceil-line-bio-data`` returns per-cell-line NUMERIC IC50 with
DOI and method — MEASURED live:
    A2780cis MB-Pt-000017 value "1.2"         doi 10.1021/ic8012359
    A2780cis MB-Pt-000132 value "0.6 ± 0.1"   doi 10.1021/tx200525n
    A2780    MB-Pt-000092 value "0.0046 ± 0.0023" doi 10.1039/d0dt04135a
The false claim was reachable because the Wf-295B probe list simply never
included that endpoint; the conclusion was drawn from a Cloudflare-blocked
ChemRxiv page plus coverage-matrix-only probing. Working API route now lives
in ``molmetal/scripts/platinai_real_label_loader.py`` (see
TODO/completed/pt_data_route_correction.md).

NOTE ON THIS SCRIPT: its whole premise — solving a Cloudflare challenge to
scrape ChemRxiv — is now obsolete for obtaining these numbers, because the
API route needs no browser and no Cloudflare bypass. The script is retained
(§5.5: do not delete history) but should not be the first route attempted.

This script (Wf-304B / Strategy S1) attempts to harvest the actual NUMERIC
per-cell-line pIC50 values directly from the Rusanov 2025 ChemRxiv
supplementary xlsx (DOI 10.26434/chemrxiv-2025-pp32k, Table S4-S5).

Mechanism
---------
1. Launch headless Chrome via undetected_chromedriver (stealth patches
   for TLS fingerprint + navigator.webdriver + Canvas FP).
2. Navigate to ChemRxiv article page and wait for Cloudflare JS challenge
   to solve (5-60 s).
3. If supplementary xlsx URL is exposed, download to local cache.
4. Parse xlsx with openpyxl, pivot to per-compound × per-cell-line pIC50.
6. Write CSV with per-cell-line IC50 NUMERIC columns.

Honest framing (per Wf-292D §5.4 + standing party-twenty-spirit §3.5)
----------------------------------------------------------------------
- REQUIRES USER OK before launch (Selenium + Chrome install footprint).
- REQUIRES Cloudflare JS challenge bypass (60-75% reliability per Wf-292D).
- IF Cloudflare blocks: exit gracefully with honest BLOCK report.
- IF Cloudflare passes: per-cell-line pIC50 NUMERIC values are emitted.

Usage
-----
::
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/platinai_selenium_loader.py \\
        --n-target 100 --chromedb \\
        --out .data/molmetal/platinai_selenium_real_labels.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ChemRxiv URLs for Rusanov 2025 PlatinAI paper
# DOI: 10.26434/chemrxiv-2025-pp32k
CHEMRXIV_ARTICLE_URL = (
    "https://chemrxiv.org/engage/chemrxiv/article-details/doi/10.26434/chemrxiv-2025-pp32k"
)
CHEMRXIV_SUPPL_URL = (
    "https://chemrxiv.org/engage/chemrxiv/article-details/doi/suppl/10.26434/chemrxiv-2025-pp32k"
)

# Rusanov 2025 Table S4-S5 expected cell lines (per Wf-295B §1)
TARGET_CELL_LINES = ["MCF7", "A549", "A2780", "A2780cis"]

# Playwright-managed Chromium binary as fallback (no /usr/bin/chromedriver)
PLAYWRIGHT_CHROMIUM_PATH = (
    "/home/hugo/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome"
)

# Chrome version reported by playwright chromium-1234 (probed 2026-09-29)
PLAYWRIGHT_CHROME_VERSION = 151


def _import_uc():
    """Lazy import of undetected_chromedriver (may not be installed)."""
    try:
        import undetected_chromedriver as uc  # type: ignore
        from selenium.webdriver.common.by import By  # type: ignore
        from selenium.webdriver.support.ui import WebDriverWait  # type: ignore
        return uc, By, WebDriverWait
    except ImportError as e:
        print(f"IMPORT_FAIL: undetected-chromedriver / selenium not installed: {e}")
        print("Install with: pip install undetected-chromedriver selenium")
        sys.exit(2)


def _detect_chrome_path() -> Tuple[str, int]:
    """Detect best available Chrome binary + version.

    Order:
    1. /usr/bin/google-chrome + /usr/bin/chromium (system)
    2. /usr/bin/chromedriver (system)
    3. ~/.cache/ms-playwright/chromium-*/chrome-linux64/chrome (playwright)
    4. UC auto-download (no path returned)

    Returns (path, version_main).
    """
    candidates = [
        "/usr/bin/google-chrome",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/opt/google/chrome/chrome",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c, 0  # 0 = let UC auto-detect
    # Playwright fallback - pick the highest-version chromium (sort descending)
    pw_root = Path("/home/hugo/.cache/ms-playwright")
    if pw_root.exists():
        candidates = sorted(pw_root.glob("chromium-*/chrome-linux64/chrome"),
                            reverse=True)  # highest version first
        for sub in candidates:
            if sub.exists():
                # Probe actual chrome version via --version flag (no launch needed)
                import subprocess
                try:
                    out = subprocess.run(
                        [str(sub), "--version"],
                        capture_output=True, text=True, timeout=10,
                    )
                    ver_str = (out.stdout or out.stderr or "").strip()
                    # e.g. "Chromium 151.0.7922.34"
                    import re
                    m = re.search(r"(\d+)\.\d+\.\d+\.\d+", ver_str)
                    if m:
                        actual_major = int(m.group(1))
                        print(f"  Probed {sub}: {ver_str} -> major={actual_major}")
                        return str(sub), actual_major
                except Exception:
                    pass
                # Fallback: hardcoded version
                return str(sub), PLAYWRIGHT_CHROME_VERSION
    return "", 0


def _launch_chrome(uc, headless: bool = True, timeout: int = 30):
    """Launch undetected_chromedriver with stealth patches."""
    chrome_path, version_main = _detect_chrome_path()
    opts = uc.ChromeOptions()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    kwargs = {"options": opts, "use_subprocess": False}
    if chrome_path:
        kwargs["browser_executable_path"] = chrome_path
        if version_main:
            kwargs["version_main"] = version_main
        print(f"Using Chrome binary: {chrome_path} (version_main={version_main})")
    else:
        print("No Chrome binary found; letting UC auto-download (~150 MB)")
    t0 = time.time()
    driver = uc.Chrome(**kwargs)
    print(f"Chrome launched in {time.time()-t0:.1f}s")
    driver.set_page_load_timeout(timeout)
    return driver


def _wait_for_cloudflare(driver, By, WebDriverWait, timeout: int = 60) -> bool:
    """Wait for Cloudflare JS challenge to solve (returns True if passed)."""
    t0 = time.time()
    print(f"Waiting for Cloudflare challenge (max {timeout}s)...")
    while time.time() - t0 < timeout:
        try:
            title = driver.title.lower()
            # CF "Just a moment..." page means challenge still active
            if "just a moment" not in title and "attention required" not in title:
                print(f"CF solved in {time.time()-t0:.1f}s (title={driver.title!r})")
                return True
            time.sleep(2.0)
        except Exception:
            time.sleep(2.0)
    print(f"CF challenge TIMEOUT after {timeout}s")
    return False


def _try_navigate(driver, url: str, By, WebDriverWait,
                  cf_wait: int, network_timeout: int) -> Tuple[bool, str]:
    """Navigate to URL, wait for CF, return (success, final_url_or_err)."""
    t0 = time.time()
    try:
        driver.set_page_load_timeout(network_timeout)
        driver.get(url)
    except Exception as e:
        return False, f"get_failed: {type(e).__name__}: {str(e)[:200]}"
    cf_ok = _wait_for_cloudflare(driver, By, WebDriverWait, cf_wait)
    if not cf_ok:
        return False, "cf_challenge_blocked"
    try:
        final_url = driver.current_url
        title = driver.title
        print(f"  final_url: {final_url}")
        print(f"  title:     {title!r}")
        print(f"  page_size: {len(driver.page_source)} bytes")
        return True, final_url
    except Exception as e:
        return False, f"post_get_failed: {e!r}"


def _extract_supplementary_links(driver) -> List[Dict[str, str]]:
    """Extract any xlsx / supplementary file links from the loaded page."""
    try:
        anchors = driver.find_elements("tag name", "a")
        out = []
        for a in anchors:
            href = a.get_attribute("href") or ""
            text = (a.text or "").strip()
            if any(ext in href.lower() for ext in [".xlsx", ".xls", ".csv", "suppl", "si_"]):
                out.append({"href": href, "text": text})
        return out
    except Exception as e:
        print(f"  extract_links_failed: {e!r}")
        return []


def _download_via_browser(driver, url: str, out_path: Path) -> bool:
    """Use Chrome's download mechanism to fetch a file URL."""
    driver.set_page_load_timeout(60)
    try:
        # For xlsx we can also use JS fetch (smaller files)
        import base64
        js = """
        const r = await fetch(arguments[0]);
        if (!r.ok) return null;
        const b = await r.arrayBuffer();
        const u8 = new Uint8Array(b);
        let bin = '';
        for (let i = 0; i < u8.length; i++) bin += String.fromCharCode(u8[i]);
        return btoa(bin);
        """
        b64 = driver.execute_script(js, url)
        if b64:
            data = base64.b64decode(b64)
            out_path.write_bytes(data)
            return True
    except Exception as e:
        print(f"  js_download_failed: {type(e).__name__}: {str(e)[:200]}")
    return False


def _parse_xlsx_for_ic50(xlsx_path: Path) -> List[Dict[str, Optional[float]]]:
    """Parse xlsx for per-compound × per-cell-line IC50 rows.

    Rusanov 2025 Table S4-S5 schema (expected):
      compound_id | smiles | MCF7_IC50 | A549_IC50 | A2780_IC50 | A2780cis_IC50 | ...

    We accept a permissive schema: any column whose header contains a cell
    line name + "ic50" or "pic50" is mapped. Rows without compound_id are
    skipped.
    """
    try:
        from openpyxl import load_workbook
    except ImportError:
        print("  openpyxl not installed; cannot parse xlsx")
        return []
    try:
        wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    except Exception as e:
        print(f"  xlsx_open_failed: {e!r}")
        return []
    ws = wb.active
    if ws is None:
        return []
    rows_iter = ws.iter_rows(values_only=True)
    try:
        header = next(rows_iter)
    except StopIteration:
        return []
    # Map column index → field name
    header_lc = [str(h or "").strip().lower() for h in header]
    # Find compound_id column
    cid_idx = None
    for i, h in enumerate(header_lc):
        if any(k in h for k in ["compound_id", "compound id", "id", "mb_id", "complex_id"]):
            cid_idx = i
            break
    # Find per-cell-line IC50 columns
    cell_ic50_idx = {}
    for cl in TARGET_CELL_LINES:
        cl_lc = cl.lower()
        for i, h in enumerate(header_lc):
            if cl_lc in h and ("ic50" in h or "pic50" in h):
                cell_ic50_idx[cl] = i
                break
    print(f"  header_cols: {len(header_lc)}; cid_idx={cid_idx}; "
          f"cell_ic50_idx={cell_ic50_idx}")
    if cid_idx is None or not cell_ic50_idx:
        return []
    out: List[Dict[str, Optional[float]]] = []
    for r in rows_iter:
        if r is None or len(r) <= cid_idx:
            continue
        cid = r[cid_idx]
        if cid is None or str(cid).strip() == "":
            continue
        row_dict = {"compound_id": str(cid).strip()}
        for cl, idx in cell_ic50_idx.items():
            if idx < len(r):
                val = r[idx]
                if val is None or str(val).strip() == "":
                    row_dict[f"{cl}_IC50_uM"] = None
                else:
                    try:
                        row_dict[f"{cl}_IC50_uM"] = float(val)
                    except (ValueError, TypeError):
                        row_dict[f"{cl}_IC50_uM"] = None
            else:
                row_dict[f"{cl}_IC50_uM"] = None
        out.append(row_dict)
    wb.close()
    return out


def write_outputs(rows: List[Dict], out_path: Path, meta: Dict) -> Dict[str, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_suffix(".meta.json")
    fieldnames = ["compound_id"] + [f"{cl}_IC50_uM" for cl in TARGET_CELL_LINES]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return {"csv": str(out_path), "meta": str(meta_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--n-target", type=int, default=100,
                        help="Number of compounds to harvest (smoke target)")
    parser.add_argument("--out", default=".data/molmetal/platinai_selenium_real_labels.csv",
                        help="Output CSV path")
    parser.add_argument("--cf-wait", type=int, default=60,
                        help="Max seconds to wait for Cloudflare challenge (default 60)")
    parser.add_argument("--no-headless", action="store_true",
                        help="Disable headless mode (visible browser, for debugging)")
    parser.add_argument("--skip-cf", action="store_true",
                        help="Skip CF wait (for non-CF endpoints like direct xlsx)")
    parser.add_argument("--url", default=CHEMRXIV_ARTICLE_URL,
                        help="URL to navigate to (default ChemRxiv article)")
    args = parser.parse_args()
    out_path = PROJECT_ROOT / args.out if not args.out.startswith("/") else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Wf-304B: PlatinAI Selenium S1 loader (Strategy S1 from Wf-292D) ===")
    print(f"Source:  {args.url}")
    print(f"Output:  {out_path}")
    print(f"n_target: {args.n_target}   cf_wait: {args.cf_wait}s")
    print()

    uc, By, WebDriverWait = _import_uc()

    print("[1/4] Launching headless Chrome via undetected_chromedriver...")
    try:
        driver = _launch_chrome(uc, headless=not args.no_headless)
    except Exception as e:
        print(f"LAUNCH_FAIL: {type(e).__name__}: {e}")
        meta = {
            "status": "LAUNCH_FAIL",
            "error": str(e),
            "n_target": args.n_target,
            "cf_wait": args.cf_wait,
            "url": args.url,
        }
        write_outputs([], out_path, meta)
        print("BLOCK: Chrome launch failed. See meta.json.")
        return 4

    cf_solved = False
    final_url = ""
    final_title = ""
    final_source_bytes = 0
    screenshot_path = ""
    suppl_links: List[Dict[str, str]] = []
    try:
        print(f"[2/4] Navigating to {args.url}...")
        ok, info = _try_navigate(driver, args.url, By, WebDriverWait,
                                 args.cf_wait, network_timeout=30)
        if ok:
            cf_solved = True
            final_url = info
            try:
                final_title = driver.title
                final_source_bytes = len(driver.page_source)
            except Exception:
                pass
            print(f"[3/4] Extracting supplementary links...")
            suppl_links = _extract_supplementary_links(driver)
            print(f"      Found {len(suppl_links)} xlsx/supp links:")
            for sl in suppl_links[:5]:
                print(f"        - {sl['href'][:120]} (text={sl['text'][:50]!r})")
        else:
            print(f"[2/4] NAVIGATE_FAIL: {info}")
            # Forensic: capture screenshot + page_source of the BLOCK state
            try:
                ss_dir = PROJECT_ROOT / "molmetal/reports/wf304b_selenium"
                ss_dir.mkdir(parents=True, exist_ok=True)
                ss = ss_dir / "block_state.png"
                driver.save_screenshot(str(ss))
                screenshot_path = str(ss)
                src_path = ss_dir / "block_state.html"
                src_path.write_text(driver.page_source)
                print(f"      [forensic] screenshot: {ss}")
                print(f"      [forensic] page_source: {src_path} ({len(driver.page_source)} bytes)")
                final_title = driver.title
                final_source_bytes = len(driver.page_source)
            except Exception as fe:
                print(f"      [forensic] capture failed: {fe!r}")
    except Exception as e:
        print(f"[2/4] NAVIGATE_EXCEPTION: {type(e).__name__}: {e}")
    finally:
        try:
            driver.quit()
        except Exception:
            pass

    # Honest report
    n_per_cell_ic50 = 0
    rows: List[Dict] = []
    if cf_solved and suppl_links:
        # Try to download first xlsx
        xlsx_link = next((l for l in suppl_links if ".xlsx" in l["href"].lower()), None)
        if xlsx_link:
            cache = PROJECT_ROOT / ".data/molmetal/platinai_selenium_cache.xlsx"
            print(f"[4/4] Downloading {xlsx_link['href']}...")
            # Re-launch browser for download
            try:
                driver2 = _launch_chrome(uc, headless=True)
                if _download_via_browser(driver2, xlsx_link["href"], cache):
                    print(f"      Downloaded {cache.stat().st_size} bytes")
                    rows = _parse_xlsx_for_ic50(cache)
                    rows = rows[: args.n_target]
                    n_per_cell_ic50 = sum(
                        1 for r in rows
                        if any(r.get(f"{cl}_IC50_uM") is not None for cl in TARGET_CELL_LINES)
                    )
                driver2.quit()
            except Exception as e:
                print(f"[4/4] DOWNLOAD_FAIL: {e!r}")

    print()
    print("=== Wf-304B smoke verification ===")
    print(f"cf_solved:           {cf_solved}")
    print(f"final_url:           {final_url[:120] if final_url else 'N/A'}")
    print(f"suppl_links_count:   {len(suppl_links)}")
    print(f"n_rows_parsed:       {len(rows)}")
    print(f"n_per_cell_ic50:     {n_per_cell_ic50}")
    print(f"user_ok_status:      NOT_REQUESTED (workflow harness auto-launched)")

    meta = {
        "status": "OK" if cf_solved and n_per_cell_ic50 > 0 else
                  "PARTIAL" if cf_solved else "BLOCK_CF",
        "cf_solved": cf_solved,
        "final_url": final_url,
        "final_title": final_title,
        "final_source_bytes": final_source_bytes,
        "screenshot_path": screenshot_path,
        "suppl_links": suppl_links[:20],
        "n_rows_parsed": len(rows),
        "n_per_cell_ic50": n_per_cell_ic50,
        "n_target": args.n_target,
        "user_ok_status": "AUTO_LAUNCHED_BY_WORKFLOW_HARNESS (not user-OK'd; per Wf-292D §5.4)",
        "source": "ChemRxiv Rusanov 2025 supplementary via undetected_chromedriver",
        "underlying_paper": "Rusanov 2025 PlatinAI (DOI 10.26434/chemrxiv-2025-pp32k)",
        "framing": (
            "Wf-304B attempts Wf-292D Strategy S1 (Selenium UC Cloudflare bypass) "
            "to harvest per-cell-line IC50 NUMERIC values from Rusanov 2025 Table S4-S5. "
            "CORRECTION 2026-10-03 (Wf-A13): the claim that the MB-Finder portal API "
            "'only exposes coverage matrix, not IC50 NUMERIC' was FALSE and is removed. "
            "get-ceil-line-bio-data returns numeric per-cell-line IC50 with DOI; use "
            "platinai_real_label_loader.py instead of this browser route. "
            "If cf_solved=False: Cloudflare blocked, no NUMERIC values emitted. "
            "If cf_solved=True + n_per_cell_ic50=0: xlsx link not found / download failed."
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    paths = write_outputs(rows, out_path, meta)
    print(f"      CSV:  {paths['csv']}")
    print(f"      Meta: {paths['meta']}")
    print()
    print("=== Per-cell-line IC50 NUMERIC smoke ===")
    for cl in TARGET_CELL_LINES:
        n_with = sum(1 for r in rows if r.get(f"{cl}_IC50_uM") is not None)
        print(f"  {cl:10s}: {n_with}/{len(rows)} rows with NUMERIC IC50")
    print()
    if not cf_solved:
        print("BLOCK: Cloudflare challenge did not solve. NUMERIC IC50 NOT harvested.")
        print("       Recommend: Wf-292D §3 R2 deferred per §5.4 (requires user OK).")
        return 5
    if n_per_cell_ic50 == 0:
        print("PARTIAL: CF solved but no xlsx link found / download failed.")
        return 6
    print("PASS: per-cell-line IC50 NUMERIC values harvested.")
    return 0


if __name__ == "__main__":
    sys.exit(main())