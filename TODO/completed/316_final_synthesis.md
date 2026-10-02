# Wf-316G — TODO directory cleanup final synthesis (aggregates Wf-316A..F)

**Date**: 2026-09-29
**Workflow**: Wf-316G (synthesis aggregation — READ-only on upstream outputs + WRITE of this doc)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ SHIPPED — synthesis aggregation doc; downstream = INDEX.md §0.4/§0.6/§0.8/§22.10/§22.11/§22.13/§22.15 (already codified by Wf-316E + Wf-320I + Wf-322D)
**Parent**: `TODO/INDEX.md` §0 (single source of truth — already updated by Wf-316E / Wf-320I / Wf-322D to include Wf-309..316 + Wf-317..320 + Wf-321..322 ship waves)
**Skill alignment**: `party-twenty-spirit` §3.5 (every Wf cite traces to a MEASURED JSON / report path; PROJECTED vs SHIPPED tag discipline preserved); `mol-metal-gpu-execution` §0 (kill_then_adapt directive preserved; all Wf-316/320/321/322 GPU launches gated)

---

## §0 TL;DR (party-twenty-spirit §3.5 honest framing)

| Metric | Before (Wf-316 start, 2026-09-29 ~09:30 CST) | After (Wf-316G ship, 2026-09-29 ~11:10 CST) | Delta |
|---|---:|---:|---:|
| `pending/` files | **20** (per `ls TODO/pending/ | wc -l` at Wf-316A start) | **13** (per `ls TODO/pending/ | wc -l` post-Wf-322D) | −7 |
| `completed/` files | 338 (per Wf-316F §1) | **408** (per INDEX.md header Wf-322D 2026-09-29 11:06 CST) | +70 |
| `archive/` files | 7 | 8 (+1: Wf-316 pre-existing 178 archive) | +1 |

| Move action | Count | Source |
|---|---:|---|
| `pending/ → completed/` (Wf-316A-B + earlier Wf-310A/313G moves) | **6** | 177, 180, 181, 183, 220, 234 |
| `pending/ → completed/` (Wf-310G v1 → Wf-313B FINAL supersession) | **1** (split as v1 + final = 2 files) | wetlab_loi_send_ready_package |
| `pending/ → archive/` (pre-Wf-316 + verification) | **1** | 178 (TODO superseded by Wf-308A `178b_smoke_fail_on_missing_hardening.md`) |
| **Net removed from `pending/`** | **7** | (6 direct + 1 archive + 1 wetlab_loi_send split) |
| Newly added to `completed/` (closeouts not from pending) | many | Wf-316A 177 dashboard, Wf-310A 181 closeout, Wf-220 3 closeouts, Wf-234 3 closeouts, wetlab_loi_send v1 + final, etc. |

**Final `pending/` categorization** (13 files per `ls TODO/pending/ | wc -l` 2026-09-29 11:06 CST):

| Category | Count | Notes |
|---|---:|---|
| **RULE** (TODO-84 paper rule) | 1 | Active rule — never moves |
| **STILL_PENDING** (work remains, GPU-gated) | 2 | 179 (W₂ PlatinAI 4-cell, R20-P2 GPU) + 182 (G6 per-innovation ablation, 7.5 hr) |
| **TEMPLATE** (wetlab DRAFT/SOP/IC50/LOI + Tier-2 DEPRECATED per Wf-317G + Wf-322B) | 10 | All wetlab_* — DEPRECATED for Tier-2 paper target per AGENTS.md §5.4 |
| **Total** | **13** | |

---

## §1 Per-agent summary (Wf-316A..F + this Wf-316G)

### §1.1 Wf-316A — Pending audit: Wf-123/Wf-124 P3 gap TODOs (177/179/180/181/182/183)

- **Output**: `molmetal/reports/wf316a_pending_audit_wf123_124.md`
- **Verdict**: 4 DONE → moved pending → completed (177/180/181/183); 2 STILL_PENDING → status-hold (179/182)
- **Per-file evidence**:
  - 177 → Wf-313G `completed/177_wf313g_downstream_lift_dashboard.md` (12/12 dashboard tests PASS; 43/43 pre-flight v2 PASS)
  - 180 → Wf-310A `results/wf124_p3_remaining_metals_g4/training_summary_v2.json` (15/15 pairs)
  - 181 → Wf-310A `completed/181_wf181_g5_50epoch_full_sweep_closeout.md` (13/15 + 2 retry OK)
  - 183 → planning doc closeout (G1-G6 all addressed)
  - 179 → STATUS_HOLD (W₂ alignment architectural infeasibility per Wf-313G dashboard §3.4 + Wf-139 P4 close-out)
  - 182 → STATUS_HOLD (G6 60-run ablation; 7.5 hr wall > 1 hr kill_then_adapt threshold)
- **Pre-flight**: frontier pipeline 20/20 PASS

### §1.2 Wf-316B — TODO-220 + TODO-234 audit + move

- **Output**: `molmetal/reports/wf316b_todo220_234_audit/audit.md`
- **Verdict**: 2 DONE → moved pending → completed (220 driver + 234 audit); 3 closeout companions already in completed/
- **Per-file evidence**:
  - 220 → Wf-308C + Wf-310C + Wf-313D: `220_wf219_temporal_split_nci60_driver.md` + 2 acceptance docs; smoke MEASURED 9/9 cells (HCT116 0.6998 highlight + PC3 0.1667 honest-negative)
  - 234 → Wf-307D + Wf-310F: `234_wf234_frontier_math_audit_integration.md` + 2 closeout docs; 7/7 unit tests + 4 smoke tests; GPU Day 3-7 DEFERRED USER_OK
- **Pre-flight**: 48/48 PASS (wall 289.37s)

### §1.3 Wf-316C — Wetlab LOI 4 drafts audit + move

- **Output**: `molmetal/reports/wf316c_wetlab_loi_audit.md`
- **Verdict**: 1 SUPERSEDED → moved wetlab_loi_send_ready_package.md pending → completed (split as v1 + final by Wf-313B); 3 LOI drafts KEPT pending (DRAFT per TODO-84 ACTIVE)
- **Per-file evidence**:
  - wetlab_loi_send_ready_package.md → Wf-313B `completed/wetlab_loi_send_ready_package_final.md` (superset of Wf-310G v1 preserved as `wetlab_loi_send_ready_package_v1.md`)
  - wetlab_loi_ornl/pku/rusanov/xiehe.md → KEPT (DRAFT LOI letters; not sent per TODO-84 ACTIVE)

### §1.4 Wf-316D — Wetlab SOPs + IC50 pipeline + TODO-84 audit

- **Output**: scope absorbed by Wf-317G + Wf-322B (Tier-2 DEPRECATED marker banners on 3 SOPs + 1 IC50 pipeline pending file)
- **Verdict**: 0 moves; 4 wetlab pending files retain in pending/ with Tier-2 marker banners per AGENTS.md §5.4
- **Per-file evidence**:
  - 3 wetlab_sop_*.md + wetlab_ic50_analysis_pipeline.md → DEPRECATED for Tier-2 paper target (computational-only); retained as paper §6 honest-limitations / §7 future-work material

### §1.5 Wf-316E — INDEX.md §0 + §22 sync

- **Output**: `TODO/completed/316_index_md_update.md`
- **Verdict**: ✅ SHIPPED (2026-09-29 10:50 CST) — INDEX.md §0.4/§0.5/§22.10/§22.11/§22.12 codified; pre-flight 52/52 PASS wall 310.5s
- **Sections added**: §0.4 (Wf-296..316 ship-wave timeline, 36 rows); §0.5 (Net effect on TODO/ inventory); §22.10 (Wf-296..305 detailed table, 11 entries); §22.11 (Wf-306..316 detailed table, 25 entries); §22.12 (Honest framing disclaimer)

### §1.6 Wf-316F — TODO directory final-state synthesis

- **Output**: `TODO/completed/316_todo_directory_final_state.md`
- **Verdict**: ✅ SHIPPED (2026-09-29 ~10:30 CST) — canonical ledger of 20 → 14 file transitions; per-file classification (COMPLETED / STILL_PENDING / RULE / TEMPLATE)
- **Pre-flight verification**: 25/25 PASS PRESERVED per Wf-263C (2026-09-28 22:18 CST, wall 68.4s, FRESH per 24-h STALE rule)

### §1.7 Wf-316G (this doc) — Final aggregation

- **Output**: `TODO/completed/316_final_synthesis.md` (this file)
- **Verdict**: ✅ SHIPPED — aggregates Wf-316A-F + Wf-316E INDEX sync + Wf-320I + Wf-322D cross-references
- **Honest framing**: INDEX.md has been comprehensively synced across Wf-316E / Wf-320I / Wf-322D — no additional INDEX.md edit required by this Wf-316G synthesis doc

---

## §2 INDEX.md status (post-Wf-316G)

Per `TODO/INDEX.md` header (2026-09-29 11:06 CST, Wf-322D sync) — already includes:

- **§0.4** Wf-296..316 ship-wave timeline (36 rows, codified by Wf-316E)
- **§0.5** Net effect on TODO/ inventory post-Wf-316E
- **§0.6** Wf-317..320 ship-wave timeline (12 rows, codified by Wf-320I)
- **§0.7** Net effect on TODO/ inventory post-Wf-320I
- **§0.8** Wf-321..322 ship-wave timeline (13 rows, codified by Wf-322D)
- **§0.9** Net effect on TODO/ inventory post-Wf-322D (this is the most current)
- **§22.10** Wf-296..305 detailed table (Wf-316E)
- **§22.11** Wf-306..316 detailed table (Wf-316E)
- **§22.13** Wf-317..320 detailed table (Wf-320I)
- **§22.15** Wf-321..322 detailed table (Wf-322D)
- **§22.12 / §22.14 / §22.16** honest framing disclaimers for each wave

**§0.9 (Wf-322D) — most current inventory snapshot**:
- completed/ → 397 → **408 files** (+11 since Wf-320I: 5 Wf-321 closeouts + 6 Wf-322 closeouts)
- pending/ → **13 files** (unchanged since Wf-316E)
- Wf-309..322 = **60 Wf sub-flows** in 2 days (2026-09-28 + 2026-09-29)
- Pre-flight v3 expanded 52 → 57 gates (Wf-321D added `p3_gate_count_consistency_audit`)
- Pre-flight v3 fresh-run 2026-09-29 11:06 CST wall 65.0s = **42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL** — HONEST-NEGATIVE REGRESSION vs previously-claimed 52/52 (full 15-FAIL gate breakdown in §22.16)
- §9 honest framing catalog **37 entries** PRESERVED (Wf-321/322 docs-only)

**No additional INDEX.md edit required** by Wf-316G — the timeline is comprehensively codified through §0.4/§0.6/§0.8 + §22.10/§22.11/§22.13/§22.15.

---

## §3 Aggregate ledger (party-twenty-spirit §3.5 honest framing)

### §3.1 File moves during Wf-316A..F (20 → 13 pending/ files)

| # | Source file (pending/) | Destination | Action | MEASURED evidence |
|---:|---|---|---|---|
| 1 | `177_wf123_p3_m1m2_downstream_metric_lift.md` | `completed/177_wf123_p3_m1m2_downstream_metric_lift.md` | MOVE | Wf-313G downstream lift dashboard at `molmetal/reports/wf313g_downstream_lift/`; closeout file at `completed/177_wf313g_downstream_lift_dashboard.md` |
| 2 | `178_wf123_p3_metalcytotox_csv_ondisk.md` | `archive/178_wf123_p3_metalcytotox_csv_ondisk.md` | ARCHIVE (superseded) | Wf-303C CSV hygiene closeout at `completed/178_metalcytotox_csv_hygiene_closeout.md`; Wf-308A `178b_smoke_fail_on_missing_hardening.md` (15/15 illegitimate skips fixed) |
| 3 | `180_wf124_p3_remaining_metals_seeds.md` | `completed/180_wf124_p3_remaining_metals_seeds.md` | MOVE | `molmetal/reports/wf265_collect/all_results.json` `g4_per_seed` — 15/15 (metal,seed) pairs MEASURED |
| 4 | `181_wf124_p3_50_epoch_full_sweep.md` | `completed/181_wf124_p3_50_epoch_full_sweep.md` | MOVE | Wf-310A retry + Wf-278A G5 aggregator; closeout at `completed/181_wf181_g5_50epoch_full_sweep_closeout.md` — 15/15 pairs MEASURED post-retry |
| 5 | `183_wf123_wf124_p3_gap_routing_plan.md` | `completed/183_wf123_wf124_p3_gap_routing_plan.md` | MOVE | Planning doc closeout (G1-G6 all addressed via downstream TODOs 177/178/179/181/182) |
| 6 | `220_wf219_temporal_split_nci60_driver.md` | `completed/220_wf219_temporal_split_nci60_driver.md` | MOVE | Wf-305C stub drivers + Wf-308C + Wf-310C + Wf-313D §6.3 + §6.4 acceptance (smoke 9/9 OK; P1 GPU PENDING USER_OK) |
| 7 | `234_wf234_frontier_math_audit_integration.md` | `completed/234_wf234_frontier_math_audit_integration.md` | MOVE | Wf-234 audit (4/4 G1-G4 PASS) + Wf-307D + Wf-310F CPU portion SHIPPED (7/7 smoke tests); GPU DEFERRED USER_OK |
| 8 | `wetlab_loi_send_ready_package.md` | `completed/wetlab_loi_send_ready_package_v1.md` + `completed/wetlab_loi_send_ready_package_final.md` | MOVE (superseded) | Wf-310G draft (v1, preserved) → Wf-313B FINAL |

**Tally**:
- **6 MOVED** to completed/ (177 / 180 / 181 / 183 / 220 / 234)
- **1 SPLIT** (wetlab_loi_send_ready_package.md → v1 + final = 2 files in completed/)
- **1 ARCHIVED** earlier (178)
- **12 KEPT** in pending/ (2 STILL_PENDING + 1 RULE + 10 TEMPLATE)
- Note that one of the original 20 (counted by Wf-316F) was already 178 which got archived earlier, hence effective 19 → 13 (-6 moves; -1 split; +1 archive from prior; -12 keep = total 13)

### §3.2 Per-file kept in pending/ (12 files + 1 wetlab_loi_send → split into completed)

| # | File | Classification | Reason kept | Wall budget |
|---:|---|---|---|---|
| 1 | `179_wf123_p3_w2_pic50_alignment.md` | STILL_PENDING | W₂ alignment architectural infeasibility (12-d RDKit signal floor; ρ ≤ 0.019 → linear R² upper bound 0.0043); Wf-313G dashboard §3.4 + Wf-139 P4 close-out | 4-5 hr + 1 day overhead (USER_OK required) |
| 2 | `182_wf124_p3_per_innovation_ablation.md` | STILL_PENDING | G6 60-run ablation matrix (4 configs × 5 metals × 3 seeds); trainer ablation flags NOT present; 7.5 hr > 1 hr kill_then_adapt threshold | 7.5 hr + 1 day overhead (USER_OK required) |
| 3 | `84_no_paper_task_without_authorization.md` | RULE | 🚫 Active rule — never moves; applies to all workflows | n/a |
| 4 | `wetlab_cold_email_template.md` | TEMPLATE | DRAFT (Wf-310G Part 1, untracked) — user responsibility to send per TODO-84 + Tier-2 DEPRECATED per AGENTS.md §5.4 | n/a |
| 5 | `wetlab_ic50_analysis_pipeline.md` | TEMPLATE | CPU-only pipeline doc — pre-wet-lab data analysis template; Tier-2 DEPRECATED per AGENTS.md §5.4 + Wf-317G + Wf-322B | n/a |
| 6 | `wetlab_loi_consolidation_review.md` | TEMPLATE | DRAFT (Wf-308D Part 2 review helper, untracked) — review-and-decide helper; NO send | n/a |
| 7 | `wetlab_loi_ornl_protein_metal.md` | TEMPLATE | DRAFT LOI letter — not sent per TODO-84 ACTIVE; PI email TBD | n/a |
| 8 | `wetlab_loi_pku_cisplatin_ic50.md` | TEMPLATE | DRAFT LOI letter — not sent per TODO-84 ACTIVE; PI email TBD | n/a |
| 9 | `wetlab_loi_rusanov_2025_platinai.md` | TEMPLATE | DRAFT LOI letter (Wf-305A NEW) — not sent per TODO-84 ACTIVE; PI email TBD | n/a |
| 10 | `wetlab_loi_xiehe_nci60.md` | TEMPLATE | DRAFT LOI letter — not sent per TODO-84 ACTIVE; PI email TBD | n/a |
| 11 | `wetlab_sop_binding_affinity_assay.md` | TEMPLATE | TEMPLATE SOP — institutional-tailored SPR/ITC; Tier-2 DEPRECATED per Wf-317G | n/a |
| 12 | `wetlab_sop_cellular_uptake_assay.md` | TEMPLATE | TEMPLATE SOP — institutional-tailored ICP-MS uptake assay; Tier-2 DEPRECATED per Wf-317G | n/a |
| 13 | `wetlab_sop_cytotoxicity_assay.md` | TEMPLATE | TEMPLATE SOP — institutional-tailored MTT/SRB/CellTiter-Glo; Tier-2 DEPRECATED per Wf-317G | n/a |

**Total**: 2 + 1 + 10 = **13 files** ✓ (matches `ls TODO/pending/ | wc -l` 2026-09-29 11:06 CST)

### §3.3 Duplicate deletions

Wf-316Q inherited duplicate-handling from 2026-09-27 prior cleanup wave (`completed/audit_2026_09_27_todo_cleanup.md`) + Wf-308A smoke-fail hardening. **No duplicates deleted in Wf-316A-F** — only renames + moves per `git status --short` lines.

### §3.4 Pre-flight v3 verification (post-Wf-322D)

- **Wf-316E verify (2026-09-29 10:47 CST)**: 52/52 PASS wall 310.5s (post-Wf-306C v3 upgrade + Wf-307C + Wf-308A + Wf-310H + Wf-312D + Wf-313H + Wf-314A + Wf-314E + Wf-315A + Wf-315F gate additions)
- **Wf-322D verify (2026-09-29 11:06 CST)**: 42/57 PASS / 15 FAIL / 0 SKIP (wall 65.0s) — HONEST-NEGATIVE REGRESSION from previously-claimed 52/52; full 15-FAIL gate breakdown in §22.16. The 15 FAIL root causes: (1) `flow_matching` upstream `configure_reference` not called → 5 pytest gates cascade-FAIL; (2) ADMET HONEST_NEGATIVE tag missing on 4 metric JSONs; (3) TODO-119 help missing `--chemprop-mode v3-fix` tokens; (4) NCI-60 `wf312b_nci60_smoke_aggregate` block missing; (5) PB packages status='DEFERRED' vs expected 'MEASURED_SMOKE'; (6) `p3_gate_count_consistency_audit` gate-count drift (actual=0 vs expected=57). Remains PENDING for future code-class Wf (Wf-323 candidate).

---

## §4 Standing directives (2026-09-29 snapshot — preserved)

1. **"Kill then adapt" GPU directive** (`mol-metal-gpu-execution` §0): explicit fresh user OK required before any GPU launch > 1 hr wall. Applies to: TODO-179 G3 (4-5 hr), TODO-182 G6 (7.5 hr), TODO-220 P1 GPU (1.0 hr Stage B), TODO-234 GPU Day 3-7, Wf-264 CFM property panel, §9.Z 5 CFM TODOs (37.8 hr total).
2. **No paper work without authorization** (TODO-84 active): zero touches to `paper/*` unless user explicitly OKs. `mol-metal-paper-writing` skill is READ-only on paper sections.
3. **Tier-2 journal target + NO wet-lab planning** (AGENTS.md §5.4 NEW 2026-09-29): paper target downgraded JCIM Tier-1 → Tier-2 (JCAMD / J. Mol. Graph. Model. / WIREs Comput. Mol. Sci. / Mol. Divers. / Comput. Biol. Chem.). Wet-lab items reclassified as `TEMPLATE_NOT_NEEDED_FOR_TIER2` per Wf-317G + Wf-322B.
4. **Pre-flight v2 25/25 PASS gate** (TODO-207): any agent launching a retrain / panel / ablation MUST run `pre_flight_check.py` first. STALE flag (24 hr) blocks PASS. **Currently REGRESSION: 42/57 PASS** — see §22.16 for 15-FAIL breakdown.
5. **Honest-negative framing** (`party-twenty-spirit` §3.5): every numeric value MUST trace to a JSON / report path on disk. MEASURED vs PROJECTED tag discipline preserved.

---

## §5 Cross-references

- `TODO/INDEX.md` §0.4 (Wf-296..316 ship-wave timeline) — codified by Wf-316E
- `TODO/INDEX.md` §0.5 (Net effect post-Wf-316E) — codified by Wf-316E
- `TODO/INDEX.md` §0.6/§0.7 (Wf-317..320 ship-wave + net effect) — codified by Wf-320I
- `TODO/INDEX.md` §0.8/§0.9 (Wf-321..322 ship-wave + net effect) — codified by Wf-322D
- `TODO/INDEX.md` §22.10/§22.11 (Wf-296..305 + Wf-306..316 detailed tables) — codified by Wf-316E
- `TODO/INDEX.md` §22.13/§22.14 (Wf-317..320 detailed table + honest framing) — codified by Wf-320I
- `TODO/INDEX.md` §22.15/§22.16 (Wf-321..322 detailed table + honest framing) — codified by Wf-322D
- `TODO/README.md` (top-level navigation + Wf-296 SHIPPED + Wf-309..317 ship wave + Tier-2 strategy re-orientation)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 honest-framing catalog (37 entries)
- `TODO/completed/316_index_md_update.md` (Wf-316E closeout)
- `TODO/completed/316_todo_directory_final_state.md` (Wf-316F final-state synthesis)
- `TODO/completed/317_final_synthesis.md` (Wf-317I final synthesis — Tier-2 strategy)
- `TODO/completed/320_final_synthesis.md` (Wf-320I final synthesis)
- `TODO/completed/322d_index_readme_wf309_322_sync.md` (Wf-322D INDEX.md + README Wf-309..322 sync)
- `molmetal/reports/wf316a_pending_audit_wf123_124.md` (Wf-316A ship report)
- `molmetal/reports/wf316b_todo220_234_audit/audit.md` (Wf-316B ship report)
- `molmetal/reports/wf316c_wetlab_loi_audit.md` (Wf-316C ship report)
- `AGENTS.md` §5.4 (Tier-2 journal target + NO wet-lab planning — NEW 2026-09-29)
- `.claude/skills/party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `.claude/skills/mol-metal-gpu-execution` §0 (kill_then_adapt directive)

---

## §6 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED today (2026-09-29 11:10 CST)**:
  - pending/ count: 20 → 13 files (post-Wf-316A-F + Wf-320I + Wf-322D aggregation)
  - completed/ count: 338 → **408 files** (per INDEX.md header Wf-322D 2026-09-29 11:06 CST)
  - archive/ count: 7 → 8 files (+1: 178 archive)
  - INDEX.md §0.4/§0.5/§0.6/§0.7/§0.8/§0.9 + §22.10/§22.11/§22.13/§22.15/§22.12/§22.14/§22.16 codified (Wf-316E + Wf-320I + Wf-322D)
  - TODO-180 G4 + TODO-181 G5 CLOSED (15/15 MEASURED per Wf-310A)
  - TODO-220 §6.3 + §6.4 acceptance gates SHIPPED smoke (5/5 + 9/9 OK)
  - TODO-234 frontier-math CPU portion SHIPPED (7/7 smoke + 4 forward-pass tests)
  - TODO-313G downstream lift dashboard SHIPPED (12/12 tests; 1/6 PASS + 2/6 PARTIAL + 3/6 HONEST_NEGATIVE)
  - §9 honest framing catalog 9 → 25 → 30 → 36 → **37 entries**
  - 4-LOI send-ready package FINAL SHIPPED; wetlab SOPs/LOIs kept in pending (DRAFT/TEMPLATE per TODO-84 ACTIVE + Tier-2 DEPRECATED per AGENTS.md §5.4)
  - Pre-flight v3 expanded 25 → 32 → 48 → 52 → **57 gates** (Wf-306C + Wf-307C + Wf-308A + Wf-310H + Wf-312D + Wf-313H + Wf-314A + Wf-314E + Wf-315A + Wf-315F + Wf-316E + Wf-321D)

- **PROJECTED / NOT MEASURED today (PENDING user OK or external dependency)**:
  - TODO-179 G3 W₂ alignment (4-5 hr, USER_OK required)
  - TODO-182 G6 per-innovation ablation (7.5 hr, USER_OK required)
  - TODO-220 P1 GPU sweep (2.9 hr Stage A + 1.0 hr Stage B, USER_OK required)
  - TODO-234 frontier-math GPU production wire-in (Wf-310F Day 3-7)
  - TODO-119 P1.4 chemprop v3 LOMO retrain (pre-flight 25/25 PASS preserved; USER_OK required)
  - Wf-263A v3-fallback post-fix AUC delta (verify_report.json NOT YET WRITTEN — IN-FLIGHT PROCESS 3095279)
  - Wf-264 CFM property panel (NO launch script on disk)
  - Wf-267 SA/QED/Vina panel (n=100 stratified)
  - Krasnov 2026 target ≥ 0.86 STILL below for all metals — Ru max mean 0.7860 (n=2 seeds) still −0.074 below

- **HONEST-NEGATIVE REGRESSION (2026-09-29 11:06 CST)**:
  - Pre-flight v3 fresh-run **42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL** (wall 65.0s) — REGRESSION from previously-claimed 52/52 per Wf-316E; full 15-FAIL gate breakdown in INDEX.md §22.16
  - Per AGENTS.md §5.4 + Wf-317G + Wf-322B: wetlab items DEPRECATED for Tier-2 paper target; the 13 pending/ files reflect the new strategy

- **Standing "kill then adapt" GPU directive**: explicit fresh user OK required before any GPU launch > 1 hr wall per `mol-metal-gpu-execution` §0.

---

## §7 Files added/modified by Wf-316G

1. `TODO/completed/316_final_synthesis.md` (this file — synthesis aggregation doc)

## §8 Files NOT touched by Wf-316G

- `TODO/INDEX.md` — no new edit required (already comprehensively codified through Wf-316E / Wf-320I / Wf-322D across §0.4/§0.5/§0.6/§0.7/§0.8/§0.9 + §22.10/§22.11/§22.13/§22.15 + §22.12/§22.14/§22.16)
- `TODO/README.md` — no new edit required (already updated through Wf-317B + Wf-322D)
- `TODO/pending/*` — no pending files modified (13 files retained per Wf-316A-F + Wf-317G + Wf-322B)
- `TODO/completed/*` — no completed files modified (only added this synthesis doc)
- `molmetal/*` — no code paths touched (Wf-316G is documentation-only synthesis)
- `paper/*` — NO paper work per TODO-84 ACTIVE rule

---

**END OF Wf-316G FINAL SYNTHESIS** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; pending/ 20 → 13 per §0; completed/ 338 → 408 per §0; pre-flight v3 42/57 PASS / 15 FAIL / 0 SKIP per §3.4 + INDEX §22.16; §0.4/§0.6/§0.8 + §22.10/§22.11/§22.13/§22.15 already codified.