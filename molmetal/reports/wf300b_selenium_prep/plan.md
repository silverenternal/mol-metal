# Wf-300B — PlatinAI S1 (Selenium + undetected-chromedriver) Implementation Prep

**Date:** 2026-09-29
**Workflow:** Wf-300B (TODO-119 P1.4 — PlatinAI per-cell-line IC50 NUMERIC follow-up)
**Strategy:** S1 (Selenium + undetected-chromedriver) — R2 follow-up from Wf-292D §3
**Status:** PREP ONLY — NOT YET LAUNCHED (Wf-292D §5.4 standing directive: requires USER OK before launch)
**Owner:** Hugo Lee + ultracode
**Inherits from:** Wf-295B (MB-Finder S7 partial harvest) + Wf-297B (gap compute)

---

## §0 TL;DR

- **What S7 (Wf-295B) gave us:** 100/100 Pt compounds with REAL coverage matrix (`has_<line>` boolean flags) + REAL `bioDataCount`. Per-cell-line coverage = A549=100, MCF7/A2780/A2780cis=0 (single-paper slice — those 100 compounds came from one Rusanov 2025 paper that focused on A549).
- **What S7 (Wf-295B) DID NOT give us:** per-cell-line IC50 NUMERIC VALUES. The MB-Finder public API exposes only presence/absence + counts, never the IC50 µM float.
- **What S1 (this prep) targets:** direct harvest from Rusanov 2025 Table S4-S5 via Selenium + undetected-chromedriver (Cloudflare bypass). Full 4,078 complexes × 4 cell lines (MCF7 / A549 / A2780 / A2780cis) × real IC50 µM.
- **Wall estimate:** ~2-3 hr (per Wf-292D §5.4 + Wf-295B strategy_report)
- **Risk:** CF bot-detection evolves — 60-75% reliability. May require 1-2 re-runs.
- **Status:** PREP ONLY. NOT LAUNCHED. Standing Wf-292D §5.4 directive: >1 hr work requires explicit USER OK before launch.

---

## §1 Current state (S7 / Wf-295B)

| | |
|---|---|
| Loader file | `molmetal/scripts/platinai_real_label_loader.py` (291 LOC) |
| CSV | `.data/molmetal/platinai_4cell_mbfinder.csv` (100 rows × 13 cols, 11.3 KB) |
| Meta | `.data/molmetal/platinai_4cell_mbfinder.meta.json` (1.1 KB) |
| Endpoints hit | `POST /api/v2/{get-substances, get-cell-lines, get-count}` |
| Coverage | mcf7=0, a549=100, a2780=0, a2780cis=0 |
| Honest gap | per-cell-line IC50 NUMERIC not exposed by public API |

**Gap class:** S7 gives us the *labelled-set skeleton* (compound IDs, SMILES, which cell-lines are covered) but not the *regression target* (pIC50 float per compound × cell-line). For paper-grade parity with Rusanov 2025 we need Table S4-S5 from the ChemRxiv supplementary.

---

## §2 URL Identification — Rusanov 2025 Table S4-S5

**Paper DOI:** `10.26434/chemrxiv-2025-pp32k`
**V2 update:** 2026-03-02 (V1 was 2025-06-20)
**Direct ChemRxiv page:** `https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-pp32k`

**Supplementary files (per WebSearch 2026-09-29):**
- `table_s4._.xlsx` (8.66 MB) — **Comparative statistical analysis of IC50 for 5 reference Pt complexes** (Cisplatin, Oxaliplatin, Carboplatin, Satraplatin, Transplatin) × 4 cell lines × {24h, 48h, 72h, 96h}. Two worksheets: "Summary (All Times)" + "Detailed (By Time)".
- `table_s5._.xlsx` (7.16 MB) — Likely the **per-compound × per-cell-line IC50 matrix** for the 3,725-3,916 unique Pt complexes (need to confirm schema after download).

**Access wall:** ChemRxiv serves via Cloudflare with JS challenge + (sometimes) institutional login (per Wf-175 + Wf-292D §1.1-§1.5). Selenium + undetected-chromedriver is the right bypass.

**Status:** URLs identified; direct download via `curl`/`requests` will hit 403; Selenium path required.

---

## §3 S1 Implementation Plan

### §3.1 Stack

```
pip install selenium undetected-chromedriver webdriver-manager
```

- **selenium** ≥ 4.15 — WebDriver client
- **undetected-chromedriver** ≥ 3.5 — patches `navigator.webdriver` + TLS fingerprint; bypasses Cloudflare's "Headless Chrome" detection
- **webdriver-manager** ≥ 4.0 — auto-downloads matching ChromeDriver binary (avoids version-mismatch errors)
- **Chrome for Testing headless tarball** (~150 MB) — drop to `~/.cache/selenium/chrome/`

### §3.2 Script skeleton

```python
# molmetal/scripts/platinai_selenium_s4s5_loader.py
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time, json, pandas as pd

URL = "https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-pp32k"
TABLE_S4_URL = "https://chemrxiv.org/engage/chemrxiv/public/details/supplementary/.../table_s4._.xlsx"
TABLE_S5_URL = "https://chemrxiv.org/engage/chemrxiv/public/details/supplementary/.../table_s5._.xlsx"

def fetch():
    opts = uc.ChromeOptions()
    opts.headless = True
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--no-sandbox")
    driver = uc.Chrome(options=opts, version_main=128)
    try:
        # 1. Land on paper, wait for CF challenge to clear (~5-15 s)
        driver.get(URL)
        WebDriverWait(driver, 30).until(EC.title_contains("PlatinAI"))
        time.sleep(3)  # let CF JS finish
        # 2. Click through to supplementary tab
        # 3. Resolve direct download URL from <a href> on supplementary panel
        # 4. driver.get(direct_url) — Chrome will save to default DL dir
        # OR: use requests with cookies harvested from driver.get_cookies()
    finally:
        driver.quit()
```

### §3.3 Two viable sub-strategies inside S1

| | A) Pure Selenium download | B) Selenium + requests hybrid |
|---|---|---|
| How | Click download in Chrome, wait for file in DL dir | Selenium loads page to clear CF, harvest cookies, hand to requests for the .xlsx GET |
| Pro | Most robust (mimics user exactly) | Faster (no browser overhead per file) |
| Con | Slow (~10-20 s/file download in headless) | Cookie lifetime ~minutes; may need to re-harvest |
| Wall | ~3 hr for S4+S5 (8.66 + 7.16 MB) | ~1 hr for S4+S5 |

**Recommendation:** start with B (hybrid) for speed; fall back to A if ChemRxiv adds CSRF on direct GET.

### §3.4 Post-download processing

- Parse both .xlsx with `pd.read_excel` (needs `openpyxl`)
- **Table S4:** 5 reference compounds × 4 cell lines × 4 timepoints × {mean, median, SD, 95% CI, CV%} → long-format CSV keyed by `(compound, cell_line, timepoint) → stats_*`
- **Table S5:** expected shape `compound_id × cell_line → IC50_µM` (need to confirm schema on first sheet). Pivot to long format: `(mb_id_or_imageId, cell_line) → ic50_um, pic50`
- **Join key:** MB-Finder `imageId` (e.g. `MB-Pt-002305`) ↔ Rusanov compound ID. May need regex mapping (Rusanov IDs may be `Pt-NNN` or similar).
- **Output:** `.data/molmetal/platinai_4cell_ic50_real.csv` (replaces S7's coverage-only CSV) + `.meta.json`

### §3.5 Wall breakdown (B - hybrid)

| Step | Wall |
|---|---|
| Install selenium + undetected_chromedriver + webdriver_manager | 5 min |
| Download Chrome for Testing tarball | 5 min |
| Smoke test (load paper, clear CF) | 10 min |
| Harvest cookies + GET S4 | 5 min |
| Harvest cookies + GET S5 | 5 min |
| Parse + pivot + join to S7 mb_ids | 30 min |
| Sanity-check + write output | 10 min |
| **Total** | **~70 min** |
| Buffer for CF retune (1-2 reruns) | +60 min |
| **Realistic estimate** | **~2 hr** |

---

## §4 Chrome Driver Status (2026-09-29)

**Verified:** `chromium`, `chromium-browser`, `google-chrome`, `chrome` — **NOT installed** in PATH.
**Verified:** `selenium`, `undetected_chromedriver` — **NOT installed** in `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`.

**Required pre-launch setup (~10 min):**
1. `pip install selenium undetected-chromedriver webdriver-manager openpyxl`
2. Download `chrome-for-testing` headless tarball (~150 MB) → `~/.cache/selenium/chrome/`
3. Smoke test: `python -c "import undetected_chromedriver; d = uc.Chrome(headless=True); d.get('https://www.google.com'); print(d.title); d.quit()"`

---

## §5 Honest Framing (Wf-292D §5.4 standing directive)

**This is PREP ONLY. S1 has NOT been launched.** Per Wf-292D §5.4: "R2 requires user OK (per standing directive: > 1 hr GPU/CPU work)". The 2-hr wall estimate clears this bar.

**If USER OK, the launch plan is:**
1. Run §4 setup (~10 min, gated by user)
2. Run §3.2 script with `--strategy hybrid` (default) or `--strategy selenium-only` (fallback)
3. Compare output coverage to S7: target n_rows ≈ 4,000 compounds × 4 cell lines = 16,000 IC50 measurements (vs S7's 100)
4. Drop S7 CSV as superseded; ship `platinai_4cell_ic50_real.csv` + meta
5. Wire into Wf-127 P4 PlatinAI 4-cell benchmark (TODO-119 P1.5)

**Risk + honest-negative contingency:**
- If CF detection evolves mid-run: 60-75% reliability per Wf-292D
- Fallback: stay on S7 (current 100-row coverage-only CSV) and ship honest-negative
- Alternative fallback: Wf-292D §3 S3 manual PDF harvest (3-12 hr, poor ROI)
- **Failure mode**: ship S7 + honest-negative §6 paper-limitation paragraph citing the Cloudflare bypass attempt

**What S1 unlocks (vs S7):**
- pIC50 regression target per compound × cell-line (not just boolean coverage)
- Full 4,078 × 4 cell coverage (vs 100 compounds × 1 cell)
- Paper-grade parity with Rusanov 2025 Table S5
- Wf-127 P4 / Wf-128 P3 / TODO-119 ship surface activates (currently gated on having IC50 floats)

**What S1 does NOT unlock:**
- Lab-protocol-level details (compound concentration units, assay type beyond IC50)
- Cell-line-specific assay conditions
- Resistance-mechanism metadata (only A2780cis vs A2780 contrast)

---

## §6 Decision Points for USER

1. **Approve launch?** Yes → run §4 setup + §3.2 script; No → stay on S7, ship honest-negative
2. **Sub-strategy?** Hybrid (default, faster) vs Selenium-only (more robust, slower)
3. **Failure threshold?** Give up after N CF blocks (default N=3) or retry indefinitely?
4. **Output schema?** Drop S7 CSV entirely (replace) or keep both (S7 coverage + S1 IC50 join)?
5. **Ship path?** Auto-wire to Wf-127 P4 (TODO-119 P1.5) or hold for separate user review?

---

## §7 References

- Wf-292D strategy report: `molmetal/reports/wf292d_platinai_bypass/strategy_report.md` (S1+S7 picks + §5.4 sign-off gate)
- Wf-295B loader report: `molmetal/reports/wf295b_platinai_loader/loader_report.md` (S7 implementation + gap analysis)
- Wf-297B synthesis: `molmetal/reports/wf297_synthesis/` (gap-compute + remediation plan)
- TODO-119: PlatinAI real-label loader (master task, P1.3 done, P1.4 this prep, P1.5 ship)
- Rusanov 2025 PlatinAI: `https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-pp32k` (V2 2026-03-02)
- Wf-175: prior Cloudflare bypass attempt (negative — informed Wf-292D design)

*Created for Wf-300B / TODO-119 P1.4 — PlatinAI S1 Selenium implementation prep. Inherits standing Wf-292D §5.4 directive: S1 requires explicit USER OK before launch.*