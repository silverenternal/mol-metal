# Wf-337-A1 — Wf-316 TODO directory audit reclassification (Tier-2 marker coverage)

**Date**: 2026-09-29
**Workflow**: Wf-337-A1 (ultracode residual gap closure — Wf-316 audit completion)
**Owner**: Hugo Lee + ultracode agent
**Parent**: AGENTS.md §5.4 (Tier-2 journal target + NO wet-lab planning) + TODO/INDEX.md §0.4 + Wf-316C ship
**Skill alignment**: `party-twenty-spirit` §3.5 (honest framing mandatory) + `mol-metal-mechanism-mapping` (audit-ledger pattern)

---

## §0 TL;DR (per `party-twenty-spirit` §3.5)

| Question | Answer |
|---|---|
| Was Wf-316 TODO directory audit supposed to reclassify wet-lab items? | **YES** — per TODO/INDEX.md §0.4 "Planned follow-up: Wf-316 TODO directory audit should reclassify wet-lab items as `TEMPLATE_NOT_NEEDED_FOR_TIER2` rather than `STILL_PENDING`" |
| Did Wf-316C ship (2026-09-29)? | **YES** — `molmetal/reports/wf316c_wetlab_loi_audit.md` retains 10 wetlab files in `TODO/pending/` with §3.4 "Wetlab drafts / templates / pipeline (10 files — KEEP per Wf-316C)" justification |
| Were all 10 wetlab files actually reclassified? | **YES** — 10/10 carry `TEMPLATE_NOT_NEEDED_FOR_TIER2` marker (8 added by Wf-324C, 2 by Wf-337-W1) |
| Did Wf-337-A1 close the residual gap? | **YES** — INDEX.md §0.4 wetlab-planning-artifacts table updated to reflect all 10 files + canonical label; INDEX.md §18 cross-references Wf-337-A1; 4 NEW tests PASS |
| Pre-flight gate required? | **NO** — CPU-only docs; no GPU launch (per honest framing) |
| Tests passing? | **4/4 PASS** (after fix) in < 1 s |

**One-line verdict (per `party-twenty-spirit` §3.5)**: Wf-337-A1 closes the
Wf-316 audit action item by verifying that all 10 wetlab files are
reclassified (canonical `TEMPLATE_NOT_NEEDED_FOR_TIER2` marker), no item
is still framed as `STILL_PENDING` or softer-`DEPRECATED`, INDEX.md §0.4
+ §18 reflect the reclassification, and 4 NEW tests gate all of the above.

---

## §1 What was supposed to happen (per Wf-337-A1 brief)

Per AGENTS.md §5.4 + TODO/INDEX.md §0.4 + the Wf-316 audit action item:

1. Find all wet-lab items in `TODO/pending/` (10 files: 4 LOIs + 3 SOPs + 1 IC50 pipeline + 2 helpers).
2. Re-classify as `TEMPLATE_NOT_NEEDED_FOR_TIER2` (per AGENTS.md §5.4).
3. Update `TODO/INDEX.md §18` (paper target — Tier-2 journal).
4. Write 4 NEW tests at `molmetal/tests/test_wf337a1_wf316_audit.py`.
5. Run the tests; verify 4/4 PASS.
6. Document this close-out.

---

## §2 What Wf-337-A1 actually delivered

### §2.1 Wetlab catalogue inventory (10 files, MEASURED)

| # | File | Marker (per Wf-324C / Wf-337-W1 / Wf-337-A1) |
|---:|:---|:---|
| 1 | `wetlab_ic50_analysis_pipeline.md` | ✅ marked (Wf-324C) |
| 2 | `wetlab_loi_ornl_protein_metal.md` | ✅ marked (Wf-324C) |
| 3 | `wetlab_loi_pku_cisplatin_ic50.md` | ✅ marked (Wf-324C) |
| 4 | `wetlab_loi_rusanov_2025_platinai.md` | ✅ marked (Wf-324C) |
| 5 | `wetlab_loi_xiehe_nci60.md` | ✅ marked (Wf-324C) |
| 6 | `wetlab_cold_email_template.md` | ✅ marked (Wf-337-W1) |
| 7 | `wetlab_loi_consolidation_review.md` | ✅ marked (Wf-337-W1) |
| 8 | `wetlab_sop_binding_affinity_assay.md` | ✅ marked (Wf-324C) |
| 9 | `wetlab_sop_cellular_uptake_assay.md` | ✅ marked (Wf-324C) |
| 10 | `wetlab_sop_cytotoxicity_assay.md` | ✅ marked (Wf-324C) |

**MEASURED (Wf-337-A1 2026-09-29):** 10/10 files carry the canonical
`TEMPLATE_NOT_NEEDED_FOR_TIER2` marker (8 added by Wf-324C, 2 added by
Wf-337-W1). No wetlab file is unmarked.

### §2.2 TODO/INDEX.md updates (MEASURED)

**Edit 1 — §0.4 wetlab items list** (lines 615-622):
- Replaced `4 LOI drafts (ORNL / PKU / 协和 / wetlab_ic50_analysis_pipeline)`
  with `4 LOI drafts (ORNL / PKU / 协和 / Rusanov-PlatinAI)` (the
  Rusanov-PlatinAI LOI was missing from the §0.4 list per Wf-337-A1 audit).
- Added bullet `1 IC50 analysis pipeline + 2 LOI/cold-email helpers
  (cold email template + 4-LOI consolidation review) → keep templates
  as reference only` (the 2 helpers were missing from the §0.4 list).

**Edit 2 — §0.4 "Planned follow-up" line** (was line 625):
- Replaced `Planned follow-up: Wf-316 TODO directory audit should
  reclassify wet-lab items as TEMPLATE_NOT_NEEDED_FOR_TIER2 rather than
  STILL_PENDING (per AGENTS.md §5.4).` with:
  `**Wf-316 TODO directory audit DONE (2026-09-29)**: all 10 wet-lab
  items in TODO/pending/wetlab_*.md reclassified as
  TEMPLATE_NOT_NEEDED_FOR_TIER2 per AGENTS.md §5.4 (markers added by
  Wf-324C + Wf-337-W1 + Wf-337-W3 + Wf-337-A1; 4 NEW tests in
  molmetal/tests/test_wf337a1_wf316_audit.py all PASS). The "Future
  TODO audit" action line below is now COMPLETED not PLANNED.`

**Edit 3 — §0.4 wetlab-planning-artifacts subsection** (was lines 629-641):
- Replaced subsection header `DEPRECATED for project scope per AGENTS.md
  §5.4 — keep as reference only` with `TEMPLATE_NOT_NEEDED_FOR_TIER2
  per AGENTS.md §5.4 — keep as reference only`.
- Replaced table header `Status (post-AGENTS.md §5.4)` with `Status
  (post-AGENTS.md §5.4 + Wf-316C ship + Wf-337-A1 audit)`.
- Replaced `⚠️ **DEPRECATED**` cell labels with `✅ **TEMPLATE_NOT_NEEDED_FOR_TIER2**` for all 7 rows.
- Added 3 missing rows: `wetlab_loi_rusanov_2025_platinai.md`,
  `wetlab_cold_email_template.md`, `wetlab_loi_consolidation_review.md`
  (the §0.4 table was incomplete at 7 files; Wf-337-A1 fills the gap to 10).
- Replaced `**Action**: Future TODO audit (Wf-316) reclassifies these
  as TEMPLATE_NOT_NEEDED_FOR_TIER2 rather than STILL_PENDING (per
  AGENTS.md §5.4).` with:
  `**Status**: **10/10 wetlab_*.md files reclassified as
  TEMPLATE_NOT_NEEDED_FOR_TIER2** per AGENTS.md §5.4 — DONE 2026-09-29
  by Wf-316C ship wave + Wf-324C + Wf-337-W1 + Wf-337-W3 + **Wf-337-A1
  audit** (4 NEW tests in molmetal/tests/test_wf337a1_wf316_audit.py
  all PASS). Verified by molmetal/reports/wf316c_wetlab_loi_audit.md +
  TODO/completed/337w1_wetlab_markers.md +
  TODO/completed/337w3_wetlab_rescope.md +
  TODO/completed/337a1_wf316_audit.md.`

**Edit 4 — §18 cross-references table** (was lines 1223-1239):
- Added 3 rows for Wf-337-A1 + Wf-337-W1 + Wf-337-W3 ship wave:
  - `TODO/completed/337a1_wf316_audit.md` | (NEW 2026-09-29) | Wf-337-A1 audit: 10/10 wetlab files `TEMPLATE_NOT_NEEDED_FOR_TIER2`; 4 NEW tests PASS
  - `molmetal/tests/test_wf337a1_wf316_audit.py` | (NEW 2026-09-29) | 4 NEW tests: reclassified / no-still-pending / Tier-2 compliance / §18 updated
  - `TODO/completed/337w1_wetlab_markers.md` + `337w3_wetlab_rescope.md` | (2026-09-29) | Wf-337-W1 + Wf-337-W3 marker coverage; complement Wf-337-A1 §18 audit

### §2.3 4 NEW tests at `molmetal/tests/test_wf337a1_wf316_audit.py`

| # | Test name | Asserts | Status |
|---:|---|---|---|
| 1 | `test_all_wetlab_items_reclassified` | All 10 wetlab files on disk + canonical `TEMPLATE_NOT_NEEDED_FOR_TIER2` marker present + catalogue drift check (no extras, no missing) | ✅ PASS |
| 2 | `test_no_items_still_marked_still_pending` | No wetlab file still framed as `STILL_PENDING` / `unscheduled` / softer-`DEPRECATED` (without canonical qualifier) | ✅ PASS |
| 3 | `test_tier2_strategy_compliance` | AGENTS.md §5.4 exists with canonical directive; INDEX.md §0.4 cites AGENTS.md §5.4; INDEX.md contains canonical `TEMPLATE_NOT_NEEDED_FOR_TIER2` token; all 10 wetlab files mentioned in INDEX.md | ✅ PASS |
| 4 | `test_index_section18_updated` | INDEX.md §18 present + cross-refs `337a1`/`Wf-337-A1`; §0.4 wetlab table contains all 10 filenames + uses canonical label; closeout doc on disk | ✅ PASS |

### §2.4 Test run output

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
    -m pytest molmetal/tests/test_wf337a1_wf316_audit.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collecting ... collected 4 items

molmetal/tests/test_wf337a1_wf316_audit.py::test_all_wetlab_items_reclassified PASSED [ 25%]
molmetal/tests/test_wf337a1_wf316_audit.py::test_no_items_still_marked_still_pending PASSED [ 50%]
molmetal/tests/test_wf337a1_wf316_audit.py::test_tier2_strategy_compliance PASSED [ 75%]
molmetal/tests/test_wf337a1_wf316_audit.py::test_index_section18_updated PASSED [100%]

============================== 4 passed in 0.16s ===============================
```

**MEASURED (Wf-337-A1 2026-09-29): 4/4 tests PASS in 0.16 s wall
(CPU-only docs-only; zero GPU launches).**

---

## §3 Test evolution (why tests were fixed mid-Wf)

Initial test 3 (`test_tier2_strategy_compliance`) had two regex bugs
flagged by the first test run:

1. `AGENTS_SECTION_RE = re.compile(r"AGENTS\.md\s*§?\s*5\.4")` did NOT
   match the AGENTS.md-internal `**§5.4 Tier-2 journal target**` header
   (which has no `AGENTS.md` prefix since it IS the source doc).
   **Fix**: widened to `(?:AGENTS\.md\s*)?§\s*5\.4` (cross-doc reference
   OR bare `§5.4` header).
2. `section_04 = re.search(r"###\s*§0\.4.*?(?=###\s*§|\Z)", ...)`
   matched the FIRST `### §0.4 ...` header (line 56 Wf-316E ship-wave
   timeline), not the SECOND (line 609 Paper target + wet-lab directive).
   **Fix**: anchored to the specific header
   `###\s*§0\.4\s+Paper target\s+\+\s+wet-?lab directive` to capture the
   Tier-2 directive section.

After fix: 4/4 tests PASS.

---

## §4 Honest framing (party-twenty-spirit §3.5 + AGENTS.md §5.2)

- **No fabricated GPU runs**: zero GPU launches performed. CPU-only docs
  work + 4 docstring-grep pytest tests.
- **Marker counts MEASURED**:
  - Wetlab files in `TODO/pending/`: 10/10 carry canonical
    `TEMPLATE_NOT_NEEDED_FOR_TIER2` marker (8 from Wf-324C, 2 from Wf-337-W1).
  - INDEX.md §0.4 wetlab table: 10/10 filenames present
    (was 7/10 pre-Wf-337-A1; 3 added by Wf-337-A1).
  - INDEX.md §0.4 wetlab table status cells: 7/7 updated from
    `⚠️ **DEPRECATED**` → `✅ **TEMPLATE_NOT_NEEDED_FOR_TIER2**`.
  - INDEX.md §18 cross-references: 3/3 Wf-337-{A1, W1, W3} rows added.
- **Pre-flight gate**: NOT required for CPU-only docs (per
  `pre_flight_check.py` design — pre-flight is for retrain / panel /
  ablation launches, not docs).
- **Tier-2 acceptance bar**: computational-only per AGENTS.md §5.4;
  wet-lab validation is OUT OF SCOPE.
- **TODO-84 paper rule preserved**: no `paper/*` writes performed in
  this Wf; only `TODO/INDEX.md` + `molmetal/tests/*` +
  `TODO/completed/*` writes.
- **5.4 honest-negative close**: this Wf closes the residual marker gap
  surfaced by the Wf-316 TODO directory audit (TODO/INDEX.md §0.4
  reclassification action item).

---

## §5 Cross-references

- **AGENTS.md §5.4** — Tier-2 journal target + NO wet-lab planning
- **TODO/INDEX.md §0.4** — Tier-2 + wetlab-planning-artifacts table (now 10/10)
- **TODO/INDEX.md §18** — Wf-139 P9 Tier-2 paper submission package
  (cross-refs Wf-337-A1 + Wf-337-W1 + Wf-337-W3)
- **molmetal/reports/wf316c_wetlab_loi_audit.md** — Wf-316C ship audit
  (§3.4 retains 10 wetlab files in pending/ with explicit justification)
- **TODO/completed/337w1_wetlab_markers.md** — 10 wetlab files marker
  coverage (Wf-324C + Wf-337-W1)
- **TODO/completed/337w3_wetlab_rescope.md** — TODO-138/R20-P6/Wf-74 v2
  re-scope markers (Wf-337-W3)
- **molmetal/tests/test_wf337a1_wf316_audit.py** — 4 NEW tests (this Wf)
- **TODO/pending/97_r20_p6_wetlab_ic50.md** — R20-P6 plan (DEPRECATED per §5.4)
- **TODO/completed/97_r20_p6_wetlab_ic50.md** — R20-P6 close-out
  (DEPRECATED + IN-SILICO RE-SCOPE markers added by Wf-337-W3)

---

## §6 Final verdict

**Date:** 2026-09-29 · **Wf-337-A1 status:** ✅ **SHIP** — 10/10 wetlab
files reclassified + INDEX.md §0.4 + §18 updated + 4 NEW tests PASS
(4/4 in 0.16 s) + this close-out doc on disk.

**Next steps (from this close-out, awaiting user):**
1. **No further action** — all wet-lab reclassification gates are
   green; the Wf-316 audit action item at INDEX.md §0.4 is now
   COMPLETED (was PLANNED).
3. **Future Wf-316-style audit passes** will see green markers in TODO/
   INDEX.md sweep.
3. **User may proceed** with Tier-2 journal submission (JCAMD / WIREs
   Comput. Mol. Sci. / etc.) per TODO-84 authorization gate.

---

## §7 Close-out metadata

- **Created**: 2026-09-29
- **Workflow**: Wf-337-A1 (ultracode residual gap closure)
- **Skill alignment**: `party-twenty-spirit` §3.5 + AGENTS.md §5.4
- **Honest framing**: applied throughout (§4) — all marker counts
  MEASURED via grep + pytest; no fabricated / aspirational numbers
- **Files written**:
  - `molmetal/tests/test_wf337a1_wf316_audit.py` (4 NEW tests, 369 LOC)
  - `TODO/completed/337a1_wf316_audit.md` (this doc)
  - `TODO/INDEX.md` — §0.4 list (3 added files), §0.4 wetlab table
    (10 rows, canonical label), §18 cross-references (3 new rows)
- **Files modified**: 1 (`TODO/INDEX.md`)
- **Tests passing**: 4/4 PASS in 0.16 s (CPU-only)
- **GPU launches**: 0 (CPU-only docs)
- **Pre-flight gate**: not required (docs-only work)
- **Wf-316 audit action item status**: ✅ CLOSED 2026-09-29