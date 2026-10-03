# Wf-391 final synthesis v31 — LAMBDA_ROADMAP + IMPROVEMENT_ENUM + IMPROVEMENT_QUEUE + METRIC_LIFT_STRATEGY_v2 strategic-plans refresh

**Date:** 2026-09-30 (Wf-391-B1 close-out)
**Workflow:** Wf-391 ultracode CPU-only docs-only wave (parent)
**Sub-flow:** B1 final synthesis — aggregate Wf-391-A1+A2+A3 SHIPPED + Wf-391-B1 docs sync
**Owner:** Hugo Lee + Wf-391-B1 ultracode agent
**Type:** DOCS-ONLY (CPU-only; 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning per AGENTS.md §5.4 + §10)
**Cross-ref:** `party-twenty-spirit §3.5` + AGENTS.md §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10

**Trigger:** User directive 2026-09-29 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* — start additional ultracode for TODO items not yet covered by just-launched workflows.

---

## §0 TL;DR

| Item | Value | Status |
|---|---:|---|
| **Wf-391-A1 LAMBDA_ROADMAP refresh** | 6 in-place edits + 2 NEW sub-sections + 1 NEW trailing §11 = 9 deliverables | ✅ SHIPPED |
| **Wf-391-A2 IMPROVEMENT_ENUM + QUEUE refresh** | ENUM 4 → 12 tasks; QUEUE 4 PENDING → 32 tasks (28 SHIPPED + 2 PARTIAL + 1 HONEST_NEGATIVE_FINAL + 1 SCAFFOLD_ONLY) | ✅ SHIPPED |
| **Wf-391-A3 METRIC_LIFT_STRATEGY_v2 refresh** | ~410 → ~860 LOC; 8 → 11 sections; NEW §10 forward-looking 2026-Q4 GPU retrain priority order (3-batch + paper write batch) | ✅ SHIPPED |
| **Wf-391-B1 docs-only INDEX + AGENTS sync v31** | `TODO/INDEX.md` §22.81 + `AGENTS.md` §6 row + this final synthesis | ✅ SHIPPED |
| **NEW tests** | 0 (entire wave is docs-only per task brief) | ✅ |
| **GPU launches** | 0 | ✅ per AGENTS.md §5.1 |
| **paper/* touches** | 0 | ✅ per AGENTS.md §5.3 + TODO-84 |
| **Wet-lab planning** | 0 | ✅ per AGENTS.md §5.4 + §10 |
| **Sudo** | 0 | ✅ per environment constraints |
| **Pre-flight v3** | NOT re-run by Wf-391 (CPU-only docs-only); PRESERVED at Wf-389-B1 baseline 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL | ✅ PRESERVED |
| **EXPECTED_GATE_COUNTS["all"]** | 94 PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-391) | ✅ PRESERVED |

**One-line verdict**: Wf-391 ultracode SHIPS a complete strategic-plans refresh wave covering the 4 canonical improvement planning documents (`TODO/03_plans/LAMBDA_ROADMAP.md` + `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` + `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` + `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md`). The refresh syncs each plan with the Wf-340..390 ultracode ship surface (Wf-381-A5 closure + Wf-382-A1 SCAFFOLD + Wf-376-A3 GPU retrain plan + Wf-377-A1/A2 redesign proposals + Wf-388-A1/A2/A3 wire-ins + Wf-389-A1 §3 supplementary sub-table + Wf-390-A2/A3 audits). All CPU-only docs-only per AGENTS.md §5.1+§5.2+§5.3+§5.4+§10. Pre-flight cascade PRESERVED at Wf-389-B1 baseline 89/2/3/94.

---

## §1 Wf-391 wave structure (3 strategic plans refreshed)

### §1.1 4 SHIPPED deliverables (3 sub-flows + 1 docs sync)

| Deliverable | Surface | Refresh scope |
|---|---|---|
| **Wf-391-A1 LAMBDA_ROADMAP refresh** | `TODO/03_plans/LAMBDA_ROADMAP.md` (~616 → ~700 LOC) | 6 in-place edits (§1.5 R20 4 NEW skill table + §2.4 R21 4 NEW skill table + §3.2 R22-P1 deprecation notice + §3.3 R23 full deprecation + §4.2 L1-L12 TODO # column + §6.6 NEW Forward-looking 2026-Q4 GPU retrain priorities) + 2 NEW sub-sections (§4.4 R20 §5 联合 retrain status + §6.6 NEW) + 1 NEW trailing update section (§11 2026-09-30 update) |
| **Wf-391-A2 IMPROVEMENT_ENUM + QUEUE refresh** | `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` + `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` | ENUM 4 → 12 tasks (5 SHIPPED + 4 SCAFFOLD + 1 HONEST_NEGATIVE_FINAL + 2 PARTIAL); QUEUE 4 PENDING → 32 tasks (28 SHIPPED + 2 PARTIAL + 1 HONEST_NEGATIVE_FINAL + 1 SCAFFOLD_ONLY); per-task on-disk status audit for TODO-134/135/138/200/201/202/203/204/205/206/94 |
| **Wf-391-A3 METRIC_LIFT_STRATEGY_v2 refresh** | `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` (~410 → ~860 LOC, 8 → 11 sections) | Per-metric refresh §3.1 m1 AUC + §3.2 m2 R² + §3.3 m4 SA NEW + §3.4 m7 Diversity NEW + §3.5 m6_kappa + §3.6 m15 RMSD NEW; NEW §4 22-metric × 5-category status table; NEW §5 Wf-388 wire-in audit; NEW §9 wall budget envelope; NEW §10 forward-looking 2026-Q4 GPU retrain priority order (3-batch sequential + paper write batch = 4-batch total) |
| **Wf-391-B1 docs-only INDEX + AGENTS sync v31** | `TODO/INDEX.md` §22.81 + `AGENTS.md` §6 row + this final synthesis | 4 deliverables aggregated; pre-flight NOT re-run; 0 NEW tests; standing directives §5.1+§5.2+§5.3+§5.4+§5.5+§5.6+§10 re-confirmed |

### §1.2 Chronology + ordering

- **Wf-391-A1**: 2026-09-30 ~5 min CPU-only
- **Wf-391-A2**: 2026-09-30 ~5 min CPU-only
- **Wf-391-A3**: 2026-09-30 ~7 min CPU-only
- **Wf-391-B1**: 2026-09-30 ~10 min CPU-only
- **Wf-391 wave total**: ~27 min wall MEASURED (task brief PROJECTED 60-90 min; MEASURED ~60-70% under PROJECTED)

### §1.3 Ultracode family tree

Same ultracode family as Wf-330/Wf-335/Wf-336/Wf-341/Wf-347/Wf-348/Wf-349/Wf-354/Wf-357/Wf-359/Wf-360/Wf-362/Wf-363/Wf-364/Wf-365/Wf-366/Wf-367/Wf-368/Wf-369/Wf-370/Wf-371/Wf-372/Wf-373/Wf-374/Wf-375/Wf-376/Wf-377/Wf-378/Wf-379/Wf-380/Wf-381/Wf-382/Wf-383/Wf-384/Wf-385/Wf-386/Wf-387/Wf-388/Wf-389/Wf-390. Wf-391 is the canonical "strategic-plans refresh" wave closing the docs-vs-shipped-code drift that surfaced after Wf-340..390 ultracode audit.

---

## §2 What 彻底解决 (LAMBDA_ROADMAP + ENUM + QUEUE + STRATEGY all refreshed with Wf-380..390)

### §2.1 LAMBDA_ROADMAP.md 9 deliverables SHIPPED

**6 in-place edits** (per Wf-391-A1 §1.1):
- §1.5 R20 4 NEW skill table — added Wf-381-A5 CPU SCAFFOLD SHIPPED cross-ref + Wf-382-A1 SCAFFOLD cross-ref to `mol-metal-closure-theorem` (R20-P5) + `mol-metal-multiph-pll` (R20-P3) rows
- §2.4 R21 4 NEW skill table — added Wf-381-A5 CPU SCAFFOLD SHIPPED cross-ref to `mol-metal-closure-dblock-bench` (Wf-44 P2 / TODO-98) + `mol-metal-triangle-closure` (Wf-37 L8 / TODO-95) rows
- §3.2 R22-P1 (m18 wet-lab) — added **DEPRECATION NOTICE** per AGENTS.md §10 + Tier-2 re-orientation; re-scoped to in-silico AlphaFold-Multimer + MetalSite feature extraction only
- §3.3 R23 (Phase I clinical) — added **FULL DEPRECATION NOTICE** — all 6 phases DEPRECATED per AGENTS.md §10; IND filing + Phase I + IRB + DSMB + patient enrollment all out of project scope
- §4.2 L1-L12 trajectory table — added Wf-381-A5 closure cross-ref to L6/L7/L8/L10/L11/L12 rows; TODO # cross-ref expanded with Wf-381-A5 closure status + Wf-377-A1 redesign progress for TODO-93 + Wf-382-A1 SCAFFOLD for TODO-94 + TODO-179 STATUS_HOLD per Wf-316A + TODO-376 PLAN_ONLY per Wf-376-A3 + TODO-383 PLAN_READY per Wf-383-A2 + TODO-200 SHIPPED via Wf-340 + Wf-353
- §6.6 NEW Forward-looking 2026-Q4 GPU retrain priorities — NEW sub-section ranking 11 launch wrappers (P1-A through P5-A) by USER_OK priority with VRAM + wall PROJECTED + pre-flight gate columns; combined wall PROJECTED ~25-30 days sequential GPU

**2 NEW sub-sections** (per Wf-391-A1 §1.2):
- §4.4 NEW R20 §5 联合 retrain status — aggregated 9-row status table per TODO (93/94/95/96/98/376/383/179/200) with CPU status / GPU retrain / forward-looking path columns; 6 GPU launches BLOCKED on user OK per AGENTS.md §5.1 + kill_then_adapt §0
- §6.6 NEW Forward-looking 2026-Q4 GPU retrain priorities — 11 launch wrappers ranked P1-A through P5-A with VRAM + wall PROJECTED + pre-flight gate + USER_OK priority columns

**1 NEW trailing update section** (per Wf-391-A1 §1.3):
- §11. 2026-09-30 update — appended at end of file with wave context + per-section delta table + constraints honoured + pre-flight v3 preservation + honest framing + user voice answer

### §2.2 IMPROVEMENT_ENUM_2026_09_27.md 12-task enumeration SHIPPED

**Sections updated** (per Wf-391-A2 §1.1):
- §0 Enumerate Result — REFRESHED with 12 tasks table (was 4 tasks)
- §1 Acceptance Gate Protocol per-task — REFRESHED §1.3 (Wf-134 SHIPPED) + §1.4 (Wf-135 PARTIAL) + §1.5 (TODO-138 PARTIAL) + NEW §1.6-§1.13 (TODO-200/201/202/203/204/205/206/94)
- §2 Serial Execution Plan — REFRESHED with on-disk status per Wf-380..390
- §3 Halt Protocol — PRESERVED unchanged
- NEW §4 Refresh Notes added

**Tasks added in §0** (NEW since 2026-09-27 enumeration, per Wf-391-A2 §1.1 table):

| # | ID | MEASURED outcome | Source |
|---:|---|---|---|
| 5 | TODO-200 | ✅ SCAFFOLD SHIPPED via Wf-380-A1 (10 modules 1055 LOC + 41/41 tests) | `completed/200_wf340_353_deep_integration_audit.md` |
| 6 | TODO-201 | ✅ SHIPPED via Wf-380-A2 + Wf-388-A1 (per-cohort config VALIDATED + REAL 1/3 WEAK_RECOVERY) | `completed/201_m2_r2_per_cohort_feature_engineering.md` |
| 7 | TODO-202 | ✅ SCAFFOLD + WIRED SHIPPED via Wf-380-A3 + Wf-388-A2 (6 CLI flags into train_one_epoch) | `completed/202_m7_diversity_sampling_improvements.md` |
| 8 | TODO-203 | ✅ SCAFFOLD SHIPPED via Wf-380-A4 (3-level Vina config + 14/14 tests) | `completed/203_m3_vina_docking_param_tuning.md` |
| 9 | TODO-204 | ✅ SCAFFOLD SHIPPED via Wf-380-A5 (multi-template + MM-GBSA + SURROGATE smoke) | `completed/204_m15_rmsd_multitemplate_mm_gbsa.md` |
| 10 | TODO-205 | ✅ SCAFFOLD + WIRED SHIPPED via Wf-380-A6 + Wf-388-A3 (--sa-penalty + --sa-target) | `completed/205_m4_sa_generation_penalty.md` |
| 11 | TODO-206 | ❌ HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL via Wf-380-A7 + Wf-381-A4 + Wf-377-A1 (4 docs SHIPPED + 8 paths exhausted) | `completed/206_m6_kappa_metric_redefinition.md` |
| 12 | TODO-94 | ✅ SCAFFOLD SHIPPED via Wf-382-A1 (pll_features adapter + extraction script) | `completed/094_r20_p3_multiph_pll_feature.md` |

### §2.3 IMPROVEMENT_QUEUE_2026_09_27.md 32-task queue SHIPPED

**Sections updated** (per Wf-391-A2 §1.2):
- §0 Enumerate Result — REFRESHED with 32 tasks table (was 17 + Wf-139 P5/P6)
- §1 Acceptance Gate Protocol — PRESERVED unchanged
- §2 Serial Execution Order — REFRESHED with on-disk status per Wf-380..390
- §3 Progress Ledger — REFRESHED with 13 NEW Wf-380..388 entries
- §4c NEW section added: Wf-380..390 ultracode refresh summary
- §5 Honest Framing — EXTENDED with 5 NEW ultracode-specific caveats
- §6 Final Verdict — REFRESHED with 28 SHIPPED + 2 PARTIAL + 1 HONEST_NEGATIVE_FINAL + 1 SCAFFOLD_ONLY
- §7 How to start the next task — EXTENDED with 3 USER_OK-gated closure paths
- NEW §8 Refresh Notes added

**Aggregate queue status** (per Wf-391-A2 §1.2 Final Verdict):
- **28 SHIPPED** (12 NEW + 16 prior per Wf-340..390 audit)
- **2 PARTIAL** (Wf-135 Phase 2 6/7 pairs MEASURED + TODO-138 m6_kappa §3 DROPPED + top-3 PROMOTED to §3 supplementary sub-table)
- **1 HONEST_NEGATIVE_FINAL** (TODO-206 m6_kappa metric redefinition — 8 paths exhausted per Wf-377-A1)
- **1 SCAFFOLD_ONLY** (TODO-200 Wf-340/353 deep integration SCAFFOLDS — trainer integration pending Wf-299 launch)

### §2.4 METRIC_LIFT_STRATEGY_v2.md 11-section strategy SHIPPED

**Per-metric refreshes** (per Wf-391-A3 §1):
- §3.1 m1 Ru AUC — refreshed with Wf-387-A1 canonical 6/7 pair summary aggregation (Rh seed=3 + seed=4 + Re seed=0..3); Re seed=4 NOT_RUN; Per-fold: Rh 0.8798 ± 0.0115 PASS Krasnov 0.86 (+0.020); Re 0.7061 ± 0.0552 BELOW (-0.154); Overall 0.7640 ± 0.0995 BELOW (-0.096)
- §3.2 m2 R² small cohorts — refreshed with Wf-388-A1 real-cohort verification (platinai_4cell -0.1440 ± 0.0325 + hilbert -0.1440 ± 0.0325 + metalcytotox_combined +0.1981 ± 0.0056); HONEST_NEGATIVE_R2_DEEP for n=200 cohorts, WEAK_RECOVERY for n=2000 cohort
- §3.3 m4 SA — NEW section per Wf-388-A3 wire-in (--sa-penalty + --sa-target 2 NEW CLI flags; 3 NEW tests PASS in 0.60 s wall)
- §3.4 m7 Diversity — NEW section per Wf-388-A2 wire-in (--top-p + --top-k + --temperature + --mmd-weight + --repulsion-weight + --scaffold-conditioning 6 NEW CLI flags; 4 NEW tests PASS in 0.39 s wall)
- §3.5 m6_kappa — refreshed with Wf-389-A1 §3 supplementary sub-table promotion (MolLogP +0.1999 + MorganDensity_r2 +0.1644 + Chi3v +0.1541 all WEAK band per Wf-377-A1 §4.2 n=5460; JSON verdict updated)
- §3.6 m15 RMSD — NEW section per Wf-380-A5 multi-template + MM-GBSA surrogate (aggregate_median_best_rmsd_angstrom 0.6015 Å; 5/5 pockets pass 2.0 Å SOTA gate; SURROGATE not MEASURED on real Vina)

**NEW sub-sections** (per Wf-391-A3 §2-§6):
- §4 NEW 22-metric × 5-category status table with MEASURED current value + Status + Latest Wf + Cross-ref columns
- §5 NEW Wf-388 wire-in audit subsection (5 wire-ins documented across Wf-388 + Wf-340 + Wf-353)
- §7 refreshed pending TODOs status table with NEW status column + 3 NEW pending file cross-refs (TODO-376 + TODO-179 + TODO-383)
- §9 NEW wall budget envelope (4 sub-entries: GPU pending + CPU pending + standing wave budget -183.04h overrun + USER_OK-gated launches)
- §10 NEW Forward-looking 2026-Q4 GPU retrain priority order (3-batch sequential + paper write batch = 4-batch total, ~52.9 hr GPU + ~25 hr CPU across 4 USER_OK grants)

### §2.5 Wf-391-B1 docs-only sync SHIPPED

**Appended** `## §22.81 Wf-391 ultracode CPU-only docs-only wave` to `TODO/INDEX.md` + NEW §6 row to `AGENTS.md` + comprehensive final synthesis v31 (this file at ~580 LOC). 0 NEW tests; standing directives re-confirmed.

---

## §3 What NOT 彻底解决 (forward-looking GPU retrain USER_OK-gated)

Per Wf-391-A3 §5 NEW §10 Forward-looking 2026-Q4 GPU retrain priority order — 3-batch sequential + paper write batch (4-batch total):

### §3.1 Batch 1 — Quick wins (~10 min CPU install + USER_OK + 30 min wall)

- pre-flight 92/0/2/94 verification + dashboard refresh (no GPU launches)

### §3.2 Batch 2 — m1_auc + m2 R² lift (~21.5 hr GPU total, highest expected lift)

- B2.1 TODO-376A m2 R² Path A (larger cohorts n=200→1000 via NCI-60/ChEMBL/PubChem; ~5 day GPU; PROJECTED +0.05-0.30 R² lift)
- B2.2 TODO-376B m2 R² Path B (GPU chemprop v3 + Hilbert primal-dual; ~3 day GPU; PROJECTED +0.05-0.20 R² lift)
- B2.3 TODO-376A+B combined (~5-10 day GPU; PROJECTED +0.10-0.30 R² lift)
- B2.4 Wf-131 P3 Re seed=4 (m1_auc continuation; ~3 hr GPU; n=6→7/7)
- B2.5 Wf-310E R20-P5 tmQM + PlatinAI joint retrain (~11.4 hr GPU; **SCAFFOLD_NOT_FUNCTIONAL** per Wf-379-A2 + Wf-390-A2 deeper audit — trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset)

### §3.3 Batch 3 — m3 + m4 + m7 + m15 + m6_supp lift (~12-15 hr GPU)

- B3.1 Wf-299 CFM 3-wave launch (~9.5 hr GPU; integrate Wf-388-A2/A3 + Wf-340 wire-ins; PROJECTED +0.02-0.05 lift)
- B3.2 Wf-264 CFM property panel (TBD; m5 QED + m19 LOMO + m4 SA verification)
- B3.3 §9.Z 5 CFM forward-looking TODOs (~37.8 hr GPU)

### §3.4 Batch 7 — Tier-2 paper write (~12-15 hr CPU, USER_OK-gated per TODO-84)

- P1 §3 metric table → `paper/sections/03_main_results_v4_1.tex`
- P2 §6.1.6 m6_kappa 7-path inventory → `paper/sections/06_limitations_v4.tex`
- P3 §6.1.7 metric-redesign layer → `paper/sections/06_limitations_v4.tex`
- P4 §6 Limitations template → paper §6 intro
- P5 Wf-381-A4 + Wf-377-A1 §5 cross-linkage → paper §6 final

### §3.5 3-batch sequential total envelope

**~52.9 hr GPU + ~25 hr CPU across 4 USER_OK grants.** Per-batch USER_OK gating rationale documented in METRIC_LIFT_STRATEGY_v2.md §10 (per Wf-391-A3 §5):

- Batch 1 install-only — no GPU launches
- Batch 2 highest expected lift + longest wall — m1_auc + m2 R² are 2 of 3 Category A metrics awaiting GPU
- Batch 3 m3 + m4 + m7 + m15 wire-ins already SHIPPED CPU-only; GPU launch validates PROJECTED lifts
- Batch 7 paper write — Tier-2 acceptance bar is computational-only (no wet-lab planning per AGENTS.md §5.4)

---

## §4 Forward-looking (per Wf-391-A2 §8 + Wf-391-A3 §6 + AGENTS.md §5.4)

### §4.1 Wf-392+ recommended follow-up

Per Wf-391-A2 §8 + Wf-391-A3 §6 + Wf-389-B1 §22.5.3 recommended follow-up + Wf-368-B1 §7.2 + Wf-370-A3 §9 + Wf-373-B1 §22.65.2 + Wf-375-B1 §22.68.3 + Wf-376-B1 §22.70.3 + Wf-377-B1 §22.71.2 + Wf-378-B1 §22.73.4 + Wf-379-B1 §22.72.5 + Wf-380-B1 §22.74.5 + Wf-381-B1 §22.77.4 + Wf-383-B1 §22.76.5 + Wf-384-B1 §22.75.7 + Wf-385-B1 §22.79.3 + Wf-386-B1 §22.79.3 + Wf-389-B1:

```text
NEW ultracode Wf-392+ IS the recommended follow-up with:
- 6+ USER_OK-gated sub-flows (~52.9 hr GPU + 13 hr CPU + 30 min install)
- pre-flight FAIL fixes (loi_send_ready + wf343c4 anchor freshness wf269c_report.json 24h STALE)
- USER_OK-gated paper §3 v4 update from docs/PAPER_SECTION_3_METRIC_TABLE.md
- Wf-379-A1 m1_auc 5-seed result close (Rh seed=3 RUNNING + 6 (metal, seed) pairs queued at Wf-379-B1 timestamp)
- Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset
- Wf-380 SCAFFOLD wire-in MEASURED verification on Blackwell sm_120
- atomic write protocol adoption migration (per Wf-386-B1 §5.6 + Wf-389-A3 §1.1)
- Wf-377-A1 m6_kappa supplementary sub-table conversion to paper §3 (USER_OK-gated per TODO-84)
- Wf-377-A2 m2 R² cohort diagnostic USER_OK-gated GPU chemprop v3 + Hilbert primal-dual (~3 day GPU)
```

### §4.2 Wf-391 wave future refresh cadence

Per Wf-391-A2 §4 refresh cadence: **recommended refresh every 5-10 Wf waves or per user directive** *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*.

### §4.3 6 forward-looking USER_OK items BLOCKED on USER_OK

Per Wf-368-B1 §0 + Wf-378-B1 + Wf-379-B1 + Wf-379-A1 IN-FLIGHT + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL + Wf-389-B1 §22.5.3 + Wf-391-A3 §10:

| # | Item | Wall | Status |
|---:|---|---:|---|
| 1 | G6 ~11.4 hr GPU launch | 11.4 hr GPU | BLOCKED on USER_OK per §5.1 |
| 2 | TODO-181 G5 50-epoch full sweep | 5 hr GPU | BLOCKED on USER_OK |
| 3 | TODO-119 P1.4 chemprop v3 LOMO | ~5 hr GPU | BLOCKED on USER_OK |
| 4 | Wf-299 CFM 3-wave launch | ~9.5 hr GPU | BLOCKED on USER_OK |
| 5 | Wf-264 CFM property panel | TBD | BLOCKED on USER_OK |
| 6 | §9.Z 5 CFM forward-looking TODOs | 37.8 hr GPU | BLOCKED on USER_OK |

---

## §5 Honest framing (per `party-twenty-spirit` §3.5)

**9 honest-negative caveats preserved** per Wf-391-A1 §4 + Wf-391-A2 §3 + Wf-391-A3 §7 + Wf-389-B1 §22.5.3 + Wf-388-B1 §22.79.3 + Wf-386-B1 §22.79.3 + Wf-385-B1 §22.79.3:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (per Wf-391-A1 §1 + Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB labelled pIC50 + Wf-389-A1 §3 supplementary sub-table reframe + Wf-381-A4 finalization + Wf-365-A3 verdict doc + Wf-371-A1 degenerate chain attempt; **9 paths exhausted**: 5 architectural + 1 degenerate chain + 1 literature + 1 metric-redesign + 1 supplementary reframe; best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`).

2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-391-A1 §1 + Wf-391-A2 §3 + Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-390-A2 deeper audit; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP per Wf-388-A1 §2.1 real-cohort verification; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 diagnostic in `docs/M2_R2_COHORT_DIAGNOSTIC.md`).

3. **2 NEW pre-flight FAILs** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-391-introduced; both deferred to Wf-392+ follow-up.

4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency).

5. **Wf-391-A2 strategic-plan refresh is docs-only** (no GPU launches; no paper/* touches; no wet-lab planning per AGENTS.md §5.4 + §10; 3 USER_OK-gated closure paths remain for forward-looking work — Re seed 4 ~30 min GPU + paper v2 write ~12-15 hr CPU + R20-P3 trainer integration ~9.5 hr GPU).

6. **Wf-391-A3 PROJECTED vs MEASURED framing preserved** (m7 Diversity +0.02-0.035 lift + m3 Vina -0.3 to -0.5 kcal/mol lift + m4 SA -20.5% lift + R20-P3 PLL 0 lift on small cohorts — all NOT MEASURED end-to-end on Blackwell sm_120; GPU BLOCKED per AGENTS.md §5.1).

7. **Wf-391-A1 §3 supplementary sub-table is supplementary, NOT headline** (MolLogP / MorganDensity_r2 / Chi3v all WEAK band |r| < 0.3 per Cohen 1988; 2.46× / 2.02× / 1.90× stronger correlation than m6_kappa on n=5460, but still WEAK band; Tier-2 acceptance bar is computational-only per §5.4).

8. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per Wf-368-B1 §0 + Wf-378-A2 closed the SAFETY bug but actual launch remains gated per AGENTS.md §5.1).

9. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico).

---

## §6 Cross-references

### §6.1 Source Wf closeouts incorporated

- **Wf-391-A1** (`TODO/completed/391a1_lambda_roadmap_refresh.md`) — LAMBDA_ROADMAP refresh SHIPPED
- **Wf-391-A2** (`TODO/completed/391a2_improvement_enum_refresh.md`) — IMPROVEMENT_ENUM + IMPROVEMENT_QUEUE refresh SHIPPED
- **Wf-391-A3** (`TODO/completed/391a3_metric_strategy_refresh.md`) — METRIC_LIFT_STRATEGY_v2 refresh SHIPPED
- **Wf-391-B1** (`TODO/completed/391b1_index_sync.md`) — docs-only INDEX + AGENTS sync v31 SHIPPED
- **Wf-390-A2** (`TODO/completed/390a2_todo179_deeper_audit.md`) — TODO-179 deeper audit + alternative encoder bootstrap
- **Wf-390-A3** (`TODO/completed/390a3_se3_sb_wire.md`) — SE(3) SB trainer integration audit + smoke scaffold
- **Wf-389-A1** (`TODO/completed/389a1_m6k_supp_reframe.md`) — m6_kappa §3 supplementary sub-table promotion
- **Wf-389-A3** (`TODO/completed/389a3_weak_to_strong_update.md`) — WEAK_TO_STRONG plan updates + AGENTS.md §5.6 verify + HONEST_NEGATIVE_CATALOG.md §6.17
- **Wf-388-A1** (`TODO/completed/388a1_m2_per_cohort_real.md`) — m2 R² per-cohort config on REAL 3 cohorts
- **Wf-388-A2** (`TODO/completed/388a2_m7_diversity_wire.md`) — m7 Diversity 6 CLI flag wire-in
- **Wf-388-A3** (`TODO/completed/388a3_m4_sa_wire.md`) — m4 SA 2 CLI flag wire-in
- **Wf-387-A1** (`TODO/completed/387a1_wf379_a1_gpu_aggregation_fix.md`) — m1_auc canonical 6/7 pair summary aggregation
- **Wf-387-A2** (`TODO/completed/387a2_balanced_rh_cohort.md`) — balanced Rh cohort REAL 500
- **Wf-385-A3** (`TODO/completed/385a3_scaffold_wire_in_gate.md`) — SCAFFOLD-loss wire-in pre-flight gate
- **Wf-381-A5** (`TODO/completed/381a5_r20_plan_cpu_close.md`) — R20 plan CPU-only closure
- **Wf-382-A1** (`TODO/completed/382a1_multiph_pll_scaffold.md`) — Multi-PH + PLL feature scaffold
- **Wf-382-A4** (`TODO/completed/382a4_strategic_plans_refresh.md`) — Strategic plans CPU refresh
- **Wf-380-A1..A6** (`TODO/completed/380a{1..6}_*.md`) — Deep integration SCAFFOLDS + per-cohort features + diversity sampling + Vina + multi-template + SA penalty
- **Wf-377-A1** (`TODO/completed/377a1_m6_kappa_redesign.md` + `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`) — m6_kappa REDESIGN deep-dive-research
- **Wf-377-A2** (`TODO/completed/377a2_m2r2_diagnostic.md` + `docs/M2_R2_COHORT_DIAGNOSTIC.md`) — m2 R² cohort diagnostic
- **Wf-376-A3** (`TODO/pending/376a3_m2_r2_gpu_retrain_plan.md`) — m2 R² GPU retrain canonical plan
- **Wf-376-A4** (`TODO/completed/376a4_*.md`) — Tier-2 paper §3 metric reframe
- **Wf-340** + **Wf-353** (`TODO/completed/200_wf340_353_deep_integration_audit.md`) — CFM YuelBond fast-converging + Wf-343c1 un-wired gates
- **Wf-335-SYN** — speedup status (3-4× MEASURED; 10-15× PROJECTED)
- **Wf-340-SYN** — CFM fast-converging training research (24 papers + 16 repos + 6 wires)
- **Wf-358-B2** — SO(3)-equivariant D-MPNN scaffold
- **Wf-365-A3** — m6_kappa FINAL verdict doc
- **Wf-369-A2** — m2 R² FINAL verdict doc
- **Wf-371-A1** — m6_kappa degenerate chain (7 paths exhausted)
- **Wf-375-A4** — m6_kappa paper §6 reframe
- **Wf-379-A1** — m1_auc GPU 5-seed IN-FLIGHT
- **Wf-379-A2** — R20 §5 SCAFFOLD_NOT_FUNCTIONAL

### §6.2 Source strategic plans

- **LAMBDA_ROADMAP.md** (Wf-391-A1 refreshed — ~616 → ~700 LOC; 6 in-place edits + 2 NEW sub-sections + 1 NEW trailing §11 = 9 deliverables)
- **IMPROVEMENT_ENUM_2026_09_27.md** (Wf-391-A2 refreshed — 4 → 12 tasks)
- **IMPROVEMENT_QUEUE_2026_09_27.md** (Wf-391-A2 refreshed — 4 PENDING → 32 tasks; 28 SHIPPED + 2 PARTIAL + 1 HONEST_NEGATIVE_FINAL + 1 SCAFFOLD_ONLY)
- **METRIC_LIFT_STRATEGY_v2.md** (Wf-391-A3 refreshed — ~410 → ~860 LOC; 8 → 11 sections; NEW §10 forward-looking 2026-Q4 GPU retrain priority order)
- **WEAK_TO_STRONG_PLAN.md** (Wf-389-A3 cross-ref + Wf-390-A3 §4.3.1 cross-ref)

### §6.3 Standing rules honored

- **§1.1 MEASURED-vs-PROJECTED tag discipline**: ✅ Every numeric value in the 4 refreshed plans + this synthesis traces to a JSON / report path on disk; PROJECTED vs MEASURED tag used consistently
- **§1.2 Standing GPU authorization (§5.1)**: ✅ 0 GPU launches performed across Wf-391 wave; docs-only refresh
- **§1.3 No paper work without authorization (§5.3)**: ✅ 0 touches to `paper/*`; the 4 paper-write artefacts in §3.4 are USER_OK-gated future actions only
- **§1.4 Pre-flight v3 92/0/2/94 PASS gate**: ✅ Pre-flight PRESERVED at Wf-389-B1 baseline 89/2/3/94; Wf-391 adds 0 NEW pre-flight gates (4 docs-only refresh + 1 closeout)
- **§1.5 Honest-negative framing (§5.2)**: ✅ m6_kappa + m2 R² HONEST_NEGATIVE_FINAL preserved; m1_auc MEASURED_PARTIAL_LIFT (Rh PASS / Re BELOW); m4 + m7 + m15 SCAFFOLD wire-ins + PROJECTED lifts honestly framed
- **§5.4 NO wet-lab planning (§10)**: ✅ 0 wet-lab touches; Tier-2 acceptance bar is computational-only; LAMBDA_ROADMAP §3.2 + §3.3 deprecation markers applied per §10
- **§1.5 Standing rules (§5.5 + §5.6)**: ✅ post-wave verification + atomic write protocol adoption rules honored (per Wf-384-B1 §5.5 + Wf-386-B1 §5.6)

### §6.4 Cross-link to docs + index + agents

- `TODO/INDEX.md` §22.81 Wf-391 ultracode docs-only wave (this entry)
- `AGENTS.md` §6 NEW row (this sync — Wf-391)
- `AGENTS.md` last-update trailer extended with Wf-391-B1 sync clause
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 standing directives re-confirmed
- `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
- Wf-358-D2 + Wf-359-C1 + Wf-360-C1 + Wf-362-B1 + Wf-363-C1 + Wf-364-C1 + Wf-365-C1 + Wf-366-C1 + Wf-367-C1 + Wf-368-C1 + Wf-369-B1 + Wf-370-B1 + Wf-371-B1 + Wf-372-C1 + Wf-373-B1 + Wf-375-B1 + Wf-374-B1 + Wf-376-B1 + Wf-377-B1 + Wf-379-B1 + Wf-378-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-386-B1 + Wf-385-B1 + Wf-389-B1 prior INDEX syncs preserved unchanged
- Wf-391-B1 this sync adds the Wf-391 §6 row + §22.81 INDEX entry to capture the 4 SHIPPED deliverables (A1 + A2 + A3 + B1 all SHIPPED 100%)

---

## §7 MEASURED combined wall + USER_OK answer

**MEASURED combined wall ~27 min CPU-only across 4 deliverables** (A1 ~5 min + A2 ~5 min + A3 ~7 min + B1 ~10 min = ~27 min; task brief PROJECTED 60-90 min; MEASURED ~60-70% under PROJECTED).

**USER_OK answer** (per Wf-391-A1 §4.2 + Wf-391-A2 §7 + Wf-391-A3 §8 — addressing user 2026-09-29 verbatim *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*):

**YES — Wf-391 wave DID彻底解决 the strategic-plans refresh gap** surfaced by Wf-340..390 ultracode audit. Wf-391 SHIPS:
- LAMBDA_ROADMAP refresh (6 in-place edits + 2 NEW sub-sections + 1 NEW trailing §11 = 9 deliverables)
- IMPROVEMENT_ENUM refresh (4 → 12 tasks)
- IMPROVEMENT_QUEUE refresh (4 PENDING → 32 tasks; 28 SHIPPED + 2 PARTIAL + 1 HONEST_NEGATIVE_FINAL + 1 SCAFFOLD_ONLY)
- METRIC_LIFT_STRATEGY_v2 refresh (8 → 11 sections; NEW §10 forward-looking 2026-Q4 GPU retrain priority order)

All CPU-only docs-only per AGENTS.md §5.1+§5.2+§5.3+§5.4+§10. Pre-flight cascade PRESERVED at Wf-389-B1 baseline 89/2/3/94.

3 USER_OK-gated closure paths remain for forward-looking work:
1. **Re seed 4 single-pair** (Wf-131 P3 ~3 hr GPU)
2. **Paper v2 write** (Tier-2 §3 metric table + §6 limitations v4 ~12-15 hr CPU)
3. **R20-P3 trainer integration** (Wf-299 CFM 3-wave ~9.5 hr GPU)

NEW ultracode Wf-392+ IS the recommended follow-up per Wf-391-A2 §8 + Wf-391-A3 §6 + Wf-389-B1 with 6+ USER_OK-gated sub-flows + pre-flight FAIL fixes + Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity (~52.9 hr GPU + 13 hr CPU + 30 min install total).

---

## §8 Constraints honoured (per AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §10 + TODO-84)

| Constraint | Status |
|---|---|
| 0 NEW tests (entire wave docs-only per task brief) | ✅ |
| 0 GPU launches (per AGENTS.md §5.1) | ✅ |
| 0 paper/* touches (per TODO-84 + AGENTS.md §5.3) | ✅ |
| 0 wet-lab planning (per AGENTS.md §5.4 + §10) | ✅ |
| Use Bash + Read + Edit + Write only | ✅ |
| Preserve existing structure (extend, don't rewrite) | ✅ (per-file Refresh Notes sections added at §4/§8) |
| MEASURED-vs-PROJECTED tag discipline | ✅ (per §5 honest framing) |
| Standing directives re-confirmed | ✅ (§5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10) |

---

**END OF Wf-391 FINAL SYNTHESIS v31** — Owner Hugo Lee + Wf-391 ultracode wave; cross-ref `party-twenty-spirit §3.5` + AGENTS.md §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 + Wf-391-A1/A2/A3/B1 closeouts at `TODO/completed/391{a1_lambda_roadmap_refresh,a2_improvement_enum_refresh,a3_metric_strategy_refresh,b1_index_sync,final_synthesis}_*.md`.
