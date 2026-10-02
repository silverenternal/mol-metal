# Wf-305 Synthesis — Exposed Problems Fix Outcomes

**Date:** 2026-09-29
**Workflow:** Wf-305 (Wf-305A/B/C/D parallel + Wf-305E synthesis)
**Owner:** Hugo Lee + ultracode
**Status:** **2 of 4 partial, 1 of 4 NOT-STARTED, 1 of 4 deferred** (see §0)

---

## §0 TL;DR — Per-Agent Summary

| Agent | Task | Status | Output | Note |
|-------|------|--------|--------|------|
| Wf-305A | PlatinAI S7 (MB-Finder stratified) + S6 (Rusanov LOI) | **PARTIAL** | `wf305a_report.md` | S7 BLOCK_RATE_LIMIT (`retry_after=1467s` ≈ 24 min); S6 LOI drafted but **0 sent** (awaiting user OK per TODO-84) |
| Wf-305B | Frontier math wire-in (0/4 → 4/4 outer adapters) | **NOT STARTED / FAILED** | (none) | The 4 adapters (`multi_ph`, `se3_lipman`, `se3_sb`, `transport_relaxation`) exist at **top-level** `molmetal/adapters/` (NOT `_reference/...` per task description). No `__init__.py` at `molmetal/adapters/` exists. Task's expected import paths (`from ._reference.multi_ph import MultiPHScheduler`) **don't resolve** — wire-in blocked on path mismatch |
| Wf-305C | TODO-287A version-pin + TODO-220 drivers + Wf-219 Ru decision | **PARTIAL (3/3 parts addressed but Parts 1+2 unverified, Part 3 deferred)** | `decision.md` + drivers + tests | Part 1: 2 anchor JSONs versioned (`m14_anchor.json`, `sweep_summary.json` both carry `version: 1.0.0`); `test_anchor_version_pin_*` tests in `test_read_anchor_json.py`. Part 2: 2 stub drivers shipped (`wf219_temporal_split_driver.py`, `nci60_9cell_chemprop_driver.py`) + `TODO/pending/220_wf219_temporal_split_nci60_driver.md`. Part 3: Ru recovery DEFERRED — Option C (scaffold-split `min-count=5→10`, ~12 min GPU-1) |
| Wf-305D | drugood DANN multi-task + Vina Pt-stripped | **PARTIAL** (CPU smoke only) | `wf305d_drugood_dann_train.py` + `wf305d_vina_pt_strip.py` + smoke JSONs | DrugOOD: seed=0 trained 1 epoch, `val_auc=NaN` (val/test empty in tiny CSV). Vina: 5/5 docked (REAL_VINA, mean=**-4.60 kcal/mol**, **BELOW field-avg -7.5 to -9.5 band**). Full GPU runs require user OK per `kill_then_adapt` |

---

## §1 Exposed Problems Closed (per Wf-304 honest-negative audit)

### §1.1 Closed

- **TODO-287A version-pin (Wf-304A gap)**: AnchorGuard `version` field now present on the 2 anchors Wf-305C shipped (`wf188_m14_full_metalcytotox/m14_anchor.json`, `wf219_krasnov_aligned/sweep_summary.json`); `test_anchor_version_pin_required_version_passes` + `test_anchor_version_pin_mismatch_raises` exist in `molmetal/tests/test_read_anchor_json.py`. Anchor-version contract **MEASURED**.
- **TODO-220 driver stubs (Wf-304C gap)**: 2 stub drivers + design doc shipped:
  - `molmetal/scripts/wf219_temporal_split_driver.py` (Krasnov temporal-split, --dry-run)
  - `molmetal/scripts/nci60_9cell_chemprop_driver.py` (NCI-60 9-cell chemprop v2 multi-task, --dry-run)
  - `TODO/pending/220_wf219_temporal_split_nci60_driver.md` (design + integration plan)
- **Wf-219 Ru decision documented (Wf-304C Task 1)**: 4-metal aggregate (`0.7755 ± 0.039`, Re/Os/Rh/Ir) ships honestly; Ru tagged `status=timeout` per-seed; recovery plan (Option C ~12 min GPU-1) documented but DEFERRED.

### §1.2 Partial / Not Closed

- **Wf-304C B-frontier wire-in**: 0/4 → **0/4** (no progress; path mismatch blocks wire-in per R17 invariant).
- **Wf-304C D-drugood DANN + Vina Pt-strip**: scripts exist + 1-epoch CPU smoke OK + 5/5 docked smoke OK; **field-avg target NOT met** (Vina mean=-4.60 kcal/mol vs -7.5 to -9.5 band; DrugOOD val_auc=NaN due to no val/test rows in smoke CSV).

### §1.3 S6 Rusanov LOI — drafted only (NOT sent)

`TODO/pending/wetlab_loi_rusanov_2025_platinai.md` exists with subject + body + 60-day cadence. **0 LOI sent at this revision**. Sending requires user OK per `TODO/pending/84_no_paper_task_without_authorization.md`.

---

## §2 Remaining Blockers (consolidated)

| ID | Blocker | Owner | Unblock Cost | Why Not Closed |
|----|---------|-------|--------------|----------------|
| **B1** | Wf-305B frontier math wire-in 0/4 → 0/4 | Wf-305B | ~2-4 h (fix path mismatch) | Adapter paths expected `_reference/...` but actual paths are top-level `molmetal/adapters/*.py`; no `__init__.py` at `molmetal/adapters/`; R17 invariant forbids editing `_reference/`. Reshape task: import from top-level. |
| **B2** | S7 MB-Finder stratified (Wf-305A) | MB-Finder API team | retry after 24 min + contact via mb-finder.org/contact | IP-level soft rate-limit (1467s ≈ 24 min) on first probe; 0/4 cell lines harvested. |
| **B3** | S6 Rusanov outreach (Wf-305A) | User OK | 1 user OK + 1 email send | LOI drafted, NOT sent. NUMERIC IC50 is the **binding constraint** per Wf-297B §6. |
| **B4** | Wf-305D DrugOOD field-avg target 0.55-0.65 | User OK + GPU-1 reservation | ~15-20 min GPU full | Smoke ran but `val_auc=NaN` (tiny CSV has no val/test rows); full run on `molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv` needs user OK. |
| **B5** | Wf-305D Vina field-avg target -7.5 to -9.5 kcal/mol | User OK + pocket diagnostic | ~10-30 min CPU full | Smoke showed mean=-4.60 kcal/mol (BELOW target band); may need different pocket (1h36 may be poor for Pt complexes) or extended pose search. |
| **B6** | Wf-219 Ru recovery (Option C) | User OK + GPU-1 | ~12 min GPU-1 | All 3 Ru seeds timed out at 32-43/50 epochs (per-seed AUCs 0.7858/0.8215/0.8214 on disk); scaffold-split min-count=5→10 keeps 50-epoch protocol fidelity. |
| **B7** | TODO-220 P1 stub → full | User OK + GPU-1 | ~10-15 min GPU full | Stubs ship --dry-run; live runs need user OK. |

---

## §3 Paper §6 Honest-Negative Entries to Add

1. **Frontier math wire-in: 0/4 outer adapters exposed.** Wf-304C audit + Wf-305B path-mismatch blocker. Wf-306+ to fix.
2. **Wf-219 4-metal aggregate (Re/Os/Rh/Ir only).** Ru excluded per honest framing; recovery DEFERRED pending user OK.
3. **S7 MB-Finder rate-limit (retry_after=1467s).** No Pt compounds harvested in Wf-305A run; alternative path = S6 Rusanov NUMERIC outreach.
4. **Wf-305D field-avg targets BELOW field band on smoke.** DrugOOD val_auc=NaN (no val/test split); Vina mean=-4.60 kcal/mol (vs -7.5 to -9.5 field-avg band). Full runs pending user OK.
5. **S6 Rusanov LOI: 0 sent.** Drafted at `TODO/pending/wetlab_loi_rusanov_2025_platinai.md`, awaiting user OK.

---

## §4 Decision Matrix — What to Launch Next

| Priority | Action | When | Notes |
|----------|--------|------|-------|
| P1 | User OK to send S6 Rusanov LOI | Day +0 (now) | High-leverage: NUMERIC IC50 is binding constraint |
| P2 | User OK for Wf-219 Ru recovery Option C | Day +0 (now) | ~12 min GPU-1 closes paper §3 cross-metal 5-metal aggregate |
| P3 | User OK for Wf-305D DrugOOD+Vina full CPU/GPU | Day +0 (now) | ~25-50 min combined, closes paper §4 §5 AUC + Vina field-avg |
| P4 | Wf-305B frontier math wire-in path-reshape | Wf-306A | R17 invariant: do NOT edit _reference/; reshape to top-level `molmetal/adapters/` lazy import |
| P5 | S7 MB-Finder retry after 24 min throttle window | Day +1 | Low-priority fallback vs S6 |
| P6 | TODO-220 P1 stub → full | Wf-306B | Stubs ship --dry-run; live runs when user OK + GPU-1 free |

---

## §5 Cross-references

- `molmetal/reports/wf305a_platinai_path_switch/wf305a_report.md` (Wf-305A report)
- `molmetal/reports/wf305c_ru_recovery/decision.md` (Wf-305C Part 3 decision)
- `TODO/pending/wetlab_loi_rusanov_2025_platinai.md` (S6 LOI draft, NEW)
- `TODO/pending/220_wf219_temporal_split_nci60_driver.md` (TODO-220 design doc, NEW)
- `molmetal/scripts/wf219_temporal_split_driver.py` (stub driver, NEW)
- `molmetal/scripts/nci60_9cell_chemprop_driver.py` (stub driver, NEW)
- `molmetal/scripts/wf305d_drugood_dann_train.py` (Wf-305D Part 1, NEW)
- `molmetal/scripts/wf305d_vina_pt_strip.py` (Wf-305D Part 2, NEW)
- `molmetal/tests/test_read_anchor_json.py` (TODO-287A version-pin tests, EXTENDED)
- `molmetal/reports/wf304b_selenium/selenium_report.md` (S1 BLOCK_CF honest-negative)
- `molmetal/reports/wf304c_todo_impl/` (Wf-304C audit baseline)

---

## §6 Revision history

| Rev | Date       | Author  | Notes |
|-----|------------|---------|-------|
| 1.0 | 2026-09-29 | Wf-305E | Initial synthesis. 2 of 4 partial, 1 of 4 not-started (Wf-305B path mismatch), 1 of 4 deferred (Ru recovery). 7 blockers consolidated. |