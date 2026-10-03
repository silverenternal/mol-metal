# Wf-381-B1 — R20 plan CPU-only closure + IMPROVEMENT_ENUM + LAMBDA_ROADMAP sync + INDEX + AGENTS + final synthesis v26

**Date**: 2026-09-30
**Status**: ✅ SHIPPED — docs-only sync across 4 surfaces (INDEX + AGENTS + IMPROVEMENT_ENUM + LAMBDA_ROADMAP) + final synthesis v26
**Workflow**: Wf-381-B1
**Owner**: Hugo Lee + Wf-381-B1 ultracode agent
**Type**: DOCS-ONLY (verification + sync only; no NEW code + no NEW tests)

**Cross-ref**: `party-twenty-spirit` §3.5 + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §10

---

## §0 TL;DR

| Item | Value | Status |
|---|---:|---|
| **Pre-flight v3 fresh-run** | **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** wall ~3 min @ Wf-381-B1 timestamp 2026-09-29T17:17:10+0000 UTC | ✅ PRESERVED at Wf-380-B1 + Wf-383-B1 + Wf-384-B1 baseline drift level |
| **INDEX.md sync** | §22.77 NEW ~85 lines appended between §22.76 Wf-383-B1 and §23 Wf-316B | ✅ SHIPPED |
| **AGENTS.md sync** | §6 row 463 NEW appended + last-update trailer extended | ✅ SHIPPED |
| **IMPROVEMENT_ENUM_2026_09_27.md sync** | §5 NEW Tier-2 Metric Realignment tier + 2 NEW items (TODO-206 + TODO-93/95/96/98 R20 closure) | ✅ SHIPPED |
| **LAMBDA_ROADMAP.md sync** | §4.2 R20 plan updated + §12 NEW appended | ✅ SHIPPED |
| **NEW tests in B1** | 0 (verification + docs-only sync only) | ✅ ON-BRIEF |
| **GPU launches in B1** | 0 | ✅ CPU-only per AGENTS.md §5.1 |
| **paper/* touches in B1** | 0 | ✅ per AGENTS.md §5.3 TODO-84 |

---

## §1 Pre-flight verification

**Run**: `PYTHONPATH=/home/hugo/codes/mol-metal python molmetal/scripts/pre_flight_check.py --pipeline all 2>&1 | tail -25 > molmetal/reports/wf381b1_preflight.txt`

**Result**: **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** wall ~3 min @ Wf-381-B1 timestamp 2026-09-29T17:17:10+0000 UTC.

**Drift level**: HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94 (matches Wf-374-B1 → Wf-375-B1 → Wf-376-B1 → Wf-377-B1 → Wf-378-B1 → Wf-379-B1 → Wf-380-B1 → Wf-383-B1 → Wf-384-B1 baseline at same drift level; **9 consecutive runs PRESERVED at 89/2/3/94**).

**2 FAILs preserved baseline** (NOT Wf-381-introduced):
1. `p3_loi_send_ready` FAIL — 4 LOI readiness issues, Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits (per AGENTS.md §5.4)
2. `p3_wf343c4_anchor_freshness_24h` FAIL — `molmetal/reports/wf277c_real_pilot/wf269c_report.json` age > 24h STALE — natural drift from Wf-374..381 wave activity

**3 SKIPs preserved (BY-DESIGN)**:
1. `p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2)
2. `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun)
3. `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency)

**`gate_count_consistency_audit`**: PASS `actual=94 == expected=94 (auto-recomputed via Wf-345-B3)`. EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline.

---

## §2 INDEX.md sync

**§22.77 Wf-381 R20 plan CPU-only closure + IMPROVEMENT_ENUM + LAMBDA_ROADMAP sync + aggregate docs-only sync + comprehensive final synthesis v26** appended between §22.76 Wf-383-B1 and §23 Wf-316B.

Entry size: ~85 lines (vs task brief PROJECTED ~75 lines; actual 85 because comprehensive per-section breakdown required for cross-linking).

---

## §3 AGENTS.md sync

**§6 row 463 NEW** appended after Wf-383-B1 row 462. Row text matches the entry in INDEX.md §22.77 with full per-section breakdown + honest framing + user voice answer.

**Last-update trailer extended** with Wf-381-B1 clause covering:
- §22.77 INDEX entry addition
- 3 SHIPPED Wf-381 deliverables (A4 + A5 + B1)
- 18 NEW tests (A4=14 + A5=4)
- 12 honest-negative caveats
- NEW ultracode Wf-382+ as recommended follow-up

---

## §4 IMPROVEMENT_ENUM sync

**§5 NEW section** added to `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md`:

### §5.1 Tier-2 Metric Realignment tier (NEW per Wf-381-B1)

| # | ID | Task | Priority | Status | Source |
|---:|---|---|---|---|---|
| 5 | TODO-206 m6_kappa metric redefinition | m6_kappa metric reframe | P1 | ✅ DONE per Wf-381-A4 | `TODO/completed/381a4_m6kappa_finalize.md` + `TODO/pending/206_m6_kappa_metric_redefinition.md` |
| 6 | TODO-93/95/96/98 R20 plan CPU closure | 4 R20 plan items CPU SCAFFOLD | P1 | ✅ DONE per Wf-381-A5 (GPU retrain deferred to Wf-299) | `TODO/completed/381a5_r20_plan_cpu_close.md` |

### §5.2 Cross-references to METRIC_LIFT_STRATEGY_v2.md

Per Wf-381-B1 §3 — METRIC_LIFT_STRATEGY_v2.md §6 row 381 lists TODO-206 as "already partial"; Wf-381-A4 closes this gap by authoring the canonical source brief at `TODO/pending/206_m6_kappa_metric_redefinition.md` + verifying 4 SHIPPED reframe docs match spec.

---

## §5 LAMBDA_ROADMAP sync

**§4.2 R20 plan updated** with Wf-381-A5 closure status:

- TODO-93 R20-P2 Hilbert ASTSR: SCAFFOLD CPU-only (per Wf-381-A5 `molmetal/scripts/hilbert_astsr_e2e.py` ~190 LOC + `R20_P2_HILBERT_ASTR_GATE_TARGET=0.6` + `hilbert_r2_gate_status()` helper); GPU joint retrain BLOCKED on Blackwell + R² ≥ 0.6 honesty gate per Wf-333-V3 + Wf-381-A4 + Wf-381-A5; closure deferred to Wf-299 USER_OK-gated launch
- TODO-94 R20-P3 Multi-PH + PLL: SCAFFOLD CPU-only (per Wf-381-A5); GPU joint retrain BLOCKED on Blackwell
- TODO-95 R20-P4 3-way + 100-pocket: SCAFFOLD CPU-only (per Wf-381-A5); GPU joint retrain BLOCKED
- TODO-96 R20-P5 Closure + Catherine: 30/30 PASS per Wf-330-B4 (per Wf-381-A5 CPU closure confirmed)
- TODO-98 R21-P1 Score-Rep + Variance MCTS: SCAFFOLD CPU-only (per Wf-381-A5); closure deferred to Wf-299

**§12 NEW section** added:
- §12.1 R20 plan per-item CPU closure status (4 NEW items table)
- §12.2 m6_kappa + m2 R² gates preservation (9 paths exhausted)
- §12.3 Cross-references

---

## §6 Honest framing

Per `party-twenty-spirit §3.5`, 12 honest-negative caveats preserved:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 9 paths exhausted per Wf-365-A3 + Wf-375-A4 + Wf-377-A1 + Wf-381-A4)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 + Wf-377-A2 + Wf-380-A2)
3. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-381-introduced
4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)**
5. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK**
6. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install)
7. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (~5 hr GPU)
8. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (R20 §5 tmQM + PlatinAI joint retrain wrapper blocked)
9. **Wf-380-B1 SCAFFOLD-only integration** (14 NEW CFM + 3 NEW DMPNN CLI flags; loss / sampling wire-in NOT implemented)
10. **Wf-381-A5 R20 plan CPU-only SCAFFOLD-only** (4 R20 plan items CPU-closed with honesty gates; GPU retrain BLOCKED on Blackwell + R² ≥ 0.6 honesty gate per Wf-299)
11. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10
12. **All Wf-380 + Wf-381 SCAFFOLD MEASURED speedup/lift claims remain PROJECTED**

---

## §7 Cross-references

- `TODO/completed/381a4_m6kappa_finalize.md` (Wf-381-A4 closeout ~17 KB; 14 NEW tests; canonical source brief NEW)
- `TODO/completed/381a5_r20_plan_cpu_close.md` (Wf-381-A5 closeout ~15 KB; 4 NEW tests; `molmetal/scripts/hilbert_astsr_e2e.py` ~190 LOC)
- `TODO/completed/381b1_index_sync.md` (this file ~150 LOC 6 sections)
- `TODO/completed/381_final_synthesis.md` (Wf-381-B1 comprehensive final synthesis v26 ~600 LOC 12 sections)
- `TODO/pending/206_m6_kappa_metric_redefinition.md` (Wf-381-A4 canonical source brief NEW ~10 KB)
- `TODO/INDEX.md` §22.77 (this entry)
- `AGENTS.md` §6 row 463 (this row) + last-update trailer extended
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` §5 NEW (Tier-2 Metric Realignment tier)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 + §12 NEW (R20 plan closure status)
- `molmetal/reports/wf381b1_preflight.txt` (Wf-381-B1 pre-flight fresh-run 89/2/3/94 wall ~3 min)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §10 standing directives re-confirmed
- `TODO/INDEX.md` §22.77 (this entry appended between §22.76 Wf-383-B1 and §23 Wf-316B)
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
- `mol-metal-closure-catherine` skill (Wf-381-A5 R20-P5)
- `mol-metal-sci-paper-submission` skill (Wf-351-B3 NEW per 2026-09-29 user directive)

---

**END OF Wf-381-B1 INDEX + AGENTS + IMPROVEMENT_ENUM + LAMBDA_ROADMAP SYNC** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; pre-flight 89/2/3/94 PRESERVED; EXPECTED_GATE_COUNTS["all"] 94 PRESERVED; 18 NEW tests across Wf-381 wave (A4=14 + A5=4); 0 NEW tests in B1 (docs-only); 0 GPU launches; 0 paper/* touches.
