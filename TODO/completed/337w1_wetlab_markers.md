# Wf-337-W1 — 10 wetlab_*.md Tier-2 DEPRECATED Marker Verification

**Document ID:** Wf-337-W1
**Version:** 1.0 (closeout, 2026-09-29)
**Authoring date:** 2026-09-29
**Owner:** Hugo Lee (Mol-Metal project lead)
**Scope:** Final verification of TEMPLATE_NOT_NEEDED_FOR_TIER2 marker presence
across the 10 `TODO/pending/wetlab_*.md` files. Adds the marker to the
2 files missed by Wf-324C, writes 4 unit tests, and confirms 4/4 PASS.

---

## 1. Why this Wf exists

Per AGENTS.md §5.4 (2026-09-29 user directive: "我们发二区就可以了，不用规划湿实验"),
Tier-2 journal is sufficient and wet-lab validation is **OUT OF SCOPE** for the
Mol-Metal project. All wet-lab planning (4 LOI drafts + 3 wetlab SOPs + 1 IC50
analysis pipeline + 2 LOI/cold-email helpers) should be reclassified as
**TEMPLATE_NOT_NEEDED_FOR_TIER2** — historical references only, NOT for
execution.

Wf-324C added the marker to 8 of the 10 wetlab files but missed 2 LOI-adjacent
helpers (`wetlab_cold_email_template.md` + `wetlab_loi_consolidation_review.md`).
Wf-337-W1 closes this gap:

| File | Status before Wf-337-W1 | Status after Wf-337-W1 |
|---|---|---|
| `wetlab_ic50_analysis_pipeline.md` | MARKED (Wf-324C) | MARKED |
| `wetlab_loi_ornl_protein_metal.md` | MARKED (Wf-324C) | MARKED |
| `wetlab_loi_pku_cisplatin_ic50.md` | MARKED (Wf-324C) | MARKED |
| `wetlab_loi_xiehe_nci60.md` | MARKED (Wf-324C) | MARKED |
| `wetlab_loi_rusanov_2025_platinai.md` | MARKED (Wf-324C) | MARKED |
| `wetlab_sop_binding_affinity_assay.md` | MARKED (Wf-324C) | MARKED |
| `wetlab_sop_cellular_uptake_assay.md` | MARKED (Wf-324C) | MARKED |
| `wetlab_sop_cytotoxicity_assay.md` | MARKED (Wf-324C) | MARKED |
| `wetlab_cold_email_template.md` | **UNMARKED** | **MARKED (Wf-337-W1)** |
| `wetlab_loi_consolidation_review.md` | **UNMARKED** | **MARKED (Wf-337-W1)** |

## 2. Per-file MEASURED status (2026-09-29, after Wf-337-W1)

| File | Marker line | Format | `AGENTS.md §5.4` ref | Tier-2 directive |
|---|---|---|---|---|
| `wetlab_ic50_analysis_pipeline.md` | L12 | Format B (Status) | yes | yes (CN + EN) |
| `wetlab_loi_ornl_protein_metal.md` | L12 | Format A (LOI) | yes | yes (CN + EN) |
| `wetlab_loi_pku_cisplatin_ic50.md` | L12 | Format A (LOI) | yes | yes (CN + EN) |
| `wetlab_loi_xiehe_nci60.md` | L12 | Format A (LOI) | yes | yes (CN + EN) |
| `wetlab_loi_rusanov_2025_platinai.md` | L13 | Format A (LOI) | yes | yes (CN + EN) |
| `wetlab_sop_binding_affinity_assay.md` | L9 | Format B (Status) | yes | yes (CN + EN) |
| `wetlab_sop_cellular_uptake_assay.md` | L9 | Format B (Status) | yes | yes (CN + EN) |
| `wetlab_sop_cytotoxicity_assay.md` | L9 | Format B (Status) | yes | yes (CN + EN) |
| `wetlab_cold_email_template.md` | L14 | Format A (LOI) | yes | yes (CN + EN) |
| `wetlab_loi_consolidation_review.md` | L12 | Format A (LOI) | yes | yes (CN + EN) |

**MEASURED: 10/10 files marked, 10/10 reference AGENTS.md §5.4, 10/10 include
the Tier-2 directive (Chinese verbatim "二区就可以了" + English "wet-lab
validation is OUT OF SCOPE").**

## 3. Tests written (4 tests, all CPU-only)

File: `molmetal/tests/test_wf337w1_wetlab_markers.py`

| # | Test name | Asserts |
|---|---|---|
| 1 | `test_all_10_wetlab_files_exist` | All 10 wetlab files on disk; no drift vs. WETLAB_FILES catalogue |
| 2 | `test_all_10_wetlab_files_have_tier2_marker` | Every file contains the canonical `TEMPLATE_NOT_NEEDED_FOR_TIER2` token |
| 3 | `test_marker_format_consistency` | Marker header matches Format A (`**TEMPLATE_NOT_NEEDED_FOR_TIER2 (YYYY-MM-DD)**`) or Format B (`**Status (YYYY-MM-DD): TEMPLATE_NOT_NEEDED_FOR_TIER2**`), and the marker sits inside a blockquote |
| 4 | `test_tier2_strategy_compliance` | Marker block cites `AGENTS.md §5.4` + the Tier-2 directive; no execution-status leak |

## 4. Test result (2026-09-29)

```
$ /home/hugo/codes/mol-metal/.venv/bin/python -m pytest molmetal/tests/test_wf337w1_wetlab_markers.py -v
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/hugo/codes/mol-metal/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/hugo/codes/mol-metal
configfile: pyproject.toml
collecting ... collected 4 items

molmetal/tests/test_wf337w1_wetlab_markers.py::test_all_10_wetlab_files_exist PASSED [ 25%]
molmetal/tests/test_wf337w1_wetlab_markers.py::test_all_10_wetlab_files_have_tier2_marker PASSED [ 50%]
molmetal/tests/test_wf337w1_wetlab_markers.py::test_marker_format_consistency PASSED [ 75%]
molmetal/tests/test_wf337w1_wetlab_markers.py::test_tier2_strategy_compliance PASSED [100%]

============================== 4 passed in 0.15s ===============================
```

**MEASURED: 4/4 tests PASS in 0.15 s wall (CPU-only).**

## 5. Honest framing (AGENTS.md §5.2 + party-twenty-spirit §3.5)

- **MEASURED (Wf-337-W1 2026-09-29):** 10 wetlab_*.md files exist in
  `TODO/pending/` (8 marked by Wf-324C + 2 newly marked by Wf-337-W1).
- **MEASURED:** All 10 files now have the `TEMPLATE_NOT_NEEDED_FOR_TIER2` marker.
- **MEASURED:** 4/4 new tests PASS in 0.15 s; no GPU work required.
- **PROJECTED:** 0 wetlab files will be executed in the R20 paper cycle per
  §5.4 (Tier-2 acceptance is computational-only).
- **BLOCK:** None. All 10 wetlab files are now compliant with AGENTS.md §5.4.

## 6. Cross-references

- AGENTS.md §5.4 (Tier-2 journal target + NO wet-lab planning, 2026-09-29 user directive)
- Wf-324C (Wf-321..322 docs-only closeout, 4 LOI/SOP Tier-2 markers) — set the precedent + 8-of-10 baseline
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule; outreach NOT gated but send is user's call)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan — DEPRECATED per §5.4)
- `TODO/pending/116_wf74_wetlab_outreach_partial.md` (Wf-74 v2 close-out rewrite — P0 TODO)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (Wf-58 in-silico dry-run)

## 7. Revision history

| Rev | Date | Author | Change |
|---|---|---|---|
| 1.0 | 2026-09-29 | Wf-337-W1 | Add TEMPLATE_NOT_NEEDED_FOR_TIER2 marker to 2 LOI-adjacent files (cold_email_template + loi_consolidation_review). Write 4 tests. Verify 10/10 marked, 4/4 tests PASS. |
