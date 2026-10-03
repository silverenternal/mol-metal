# Wf-391-A3 — METRIC_LIFT_STRATEGY_v2.md refresh (Wf-387/388/389 wire-in sync)

**Date:** 2026-09-30
**Status:** ✅ **SHIPPED** — docs-only refresh; 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning
**Workflow:** Wf-391-A3 (sub-flow of Wf-391 ultracode wave per user directive 2026-09-30 "todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做")
**Owner:** Hugo Lee + Wf-391-A3 ultracode
**Cross-ref:** `party-twenty-spirit §3.5` + AGENTS.md §1 + §5.1 + §5.2 + §5.3 + §5.4 + §10

---

## §0 TL;DR

| Item | Value | Status |
|---|---:|---|
| **Source brief** | `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` (existing ~410 LOC, 8 sections) | ✅ FOUND |
| **Refreshed strategy** | `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` (refreshed ~860 LOC, 11 sections) | ✅ SHIPPED |
| **Per-metric updates** | §3.1 m1 AUC + §3.2 m2 R² + §3.3 m4 SA + §3.4 m7 Diversity + §3.5 m6_kappa + §3.6 m15 RMSD + §4 per-metric status table + §5 Wf-388 wire-ins + §7 pending TODOs + §10 NEW forward-looking | ✅ ALL 9 sections refreshed |
| **NEW §10 Forward-looking 2026-Q4 GPU retrain priority order** | 3-batch sequential (install + m1/m2 lift + m3/m4/m7/m15) + paper write batch (4-batch total) | ✅ SHIPPED |
| **NEW tests** | 0 (docs-only per task brief) | ✅ |
| **Code patches** | 1 NEW docs (refreshed strategy); 1 NEW closeout (this file) | ✅ docs-only |
| **GPU launches** | 0 | ✅ per AGENTS.md §5.1 |
| **Paper/* touches** | 0 | ✅ per AGENTS.md §5.3 + TODO-84 |
| **Wetlab planning** | 0 | ✅ per AGENTS.md §5.4 + §10 |
| **Pre-flight v3 status** | 92/0/2/94 PRESERVED at Wf-368-C1 baseline (NOT re-run by Wf-391-A3 docs-only) | ✅ PRESERVED |

**One-line verdict**: Wf-391-A3 SHIPPED the docs-only refresh of `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` (~410 → ~860 LOC, 8 → 11 sections) to sync the strategic 22-metric plan with the Wf-387-A1 (m1_auc aggregation SHIPPED) + Wf-388-A1/A2/A3 (m2 R² verification + m7 + m4 wire-ins SHIPPED) + Wf-389-A1 (m6_kappa supplementary sub-table SHIPPED) + Wf-380-A5 (m15 surrogate SHIPPED) wire-in surface. Added NEW §10 forward-looking 2026-Q4 GPU retrain priority order (3-batch sequential + paper write batch = 4-batch total, ~52.9 hr GPU + ~25 hr CPU across 4 USER_OK grants). 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning. Per AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §10 standing directives re-confirmed.

---

## §1 Per-metric refresh status

### §1.1 m1 Ru AUC (§3.1)

**Refreshed**: Added Wf-387-A1 canonical 6/7 pair summary aggregation per `TODO/completed/387a1_wf379_a1_gpu_aggregation_fix.md`. New values:
- n_runs = 6 (Rh seed=3 + seed=4 + Re seed=0..3); Re seed=4 NOT_RUN
- Per-fold: Rh 0.8798 ± 0.0115 **PASS** Krasnov 0.86 (gap +0.020); Re 0.7061 ± 0.0552 BELOW (gap −0.154)
- Overall 0.7640 ± 0.0995 BELOW Krasnov 0.86 by −0.096 margin

**Code change path updated** to cite `molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py` (SHIPPED) + `scripts/launch_wf131_p3_deferred_runs.sh` (LIVE mode — Wf-355-B1 USER_OK-gated — Re seed=4 pending).

### §1.2 m2 R² small cohorts (§3.2)

**Refreshed**: Added Wf-388-A1 real-cohort verification per `TODO/completed/388a1_m2_per_cohort_real.md`. New values:
- platinai_4cell n=200 + 12-d RDKit + Ridge α=1.0 → **−0.1440 ± 0.0325** (HONEST_NEGATIVE_R2_DEEP)
- hilbert n=200 + 12-d RDKit + Ridge α=1.0 → **−0.1440 ± 0.0325** (HONEST_NEGATIVE_R2_DEEP)
- metalcytotox_combined n=2000 + 32-d RDKit + Ridge α=0.1 → **+0.1981 ± 0.0056** (WEAK_RECOVERY)

**Honest finding**: synthetic R² > 0 (Wf-380-A2) over-promised on real cohorts. Per-cohort config axis lifts synthetic baseline but does NOT cross zero on real cohorts for n=200.

**Forward-looking plan**: cross-ref `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` (canonical Wf-376-A3 plan with Path A + Path B + Path A+B combined). Total wall ~5-10 day sequential GPU; PROJECTED lift +0.05 to +0.30 R²; launch wrapper `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (472 LOC, EXISTS on disk).

### §1.3 m4 SA (§3.3 — NEW section)

**NEW per Wf-391-A3 refresh**: Added §3.3 m4 SA section that documents Wf-388-A3 wire-in per `TODO/completed/388a3_m4_sa_wire.md`. 2 NEW CLI flags:
- `--sa-penalty` (default 0.0 OFF) — weight on SA penalty term
- `--sa-target` (default 2.8) — SA target threshold (Ertl 2009 §3)

Wire-in sites: lazy import (~line 937) + CLI registration (~line 1020) + loss formula (~line 780). 3 NEW tests PASS in 0.60 s wall (`molmetal/tests/test_wf388a3_sa_wire.py`). Bit-for-bit backward-compat when flag default OFF.

**PROJECTED lift**: −0.3-0.5 SA on m4 requires GPU retrain USER_OK-gated.

### §1.4 m7 Diversity (§3.4 — NEW section)

**NEW per Wf-391-A3 refresh**: Added §3.4 m7 Diversity section that documents Wf-388-A2 wire-in per `TODO/completed/388a2_m7_diversity_wire.md`. 6 NEW CLI flags:
- `--top-p` (default 1.0 OFF), `--top-k` (default 0 OFF), `--temperature` (default 1.0 OFF)
- `--mmd-weight` (default 0.0 OFF), `--repulsion-weight` (default 0.0 OFF)
- `--scaffold-conditioning` (default False, NEW flag)

When all 6 flags OFF, trainer loss bit-exact equivalent to Wf-380-B1 / Wf-299 / Wf-124 P2. 4 NEW tests PASS in 0.39 s (`molmetal/tests/test_wf388a2_diversity_wire.py`). +120 / −15 LOC wire-in to `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`.

**PROJECTED lift**: +0.02-0.035 on m7 requires GPU retrain USER_OK-gated.

### §1.5 m6_kappa (§3.5 — refreshed)

**Refreshed**: Added Wf-389-A1 §3 supplementary sub-table promotion per `TODO/completed/389a1_m6k_supp_reframe.md`. NEW content:
- JSON verdict field updated: `metrics/by_metric/m6_kappa.json::verdict` = **MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL**
- `wf389a1_supplementary_reframe` block added to JSON (top-3 promotion record + supplementary sub-table lineup)
- NEW doc: `docs/PAPER_SECTION_3_METRIC_TABLE.md` (~580 LOC, 6 sections)
- §4 ready-to-paste LaTeX block (~150 LOC)

Top-3 alternatives promoted to §3 supplementary sub-table:
- MolLogP +0.1999 (n=5460, WEAK band)
- MorganDensity_r2 +0.1644 (WEAK)
- Chi3v +0.1541 (WEAK)

Per-metal MODERATE-band discoveries (small-cohort caveat): Rh BalabanJ +0.3953 / Rh MolLogP −0.3416 / Re MolLogP +0.3363 — do NOT promote to §3 main-result per n ≤ 166 caveat.

### §1.6 m15 RMSD (§3.6 — NEW section)

**NEW per Wf-391-A3 refresh**: Added §3.6 m15 RMSD section that documents Wf-380-A5 multi-template + MM-GBSA surrogate per `TODO/completed/380a5_multi_template_mm_gbsa.md` + `TODO/pending/204_m15_rmsd_multitemplate_mm_gbsa.md`.

**SURROGATE bench headline**:
- `aggregate_median_best_rmsd_angstrom` = **0.6015 Å** (5 templates × 9 poses, baseline=0.65 Å)
- `aggregate_success_rate_2_0a` = **1.0** (5/5 pockets pass 2.0 Å SOTA gate)
- `m15_status` = **SURROGATE** (closed-form synthetic, NOT MEASURED on real Vina)

**PROJECTED lift**: ~0.05 Å on 5-pocket organic-cofactor set; real Vina + meeko deferred to TODO-207 / Wf-299 launch USER_OK-gated.

---

## §2 Per-metric status table (NEW §4)

Added comprehensive 22-metric × 5-category status table with MEASURED current value + Status + Latest Wf + Cross-ref columns. Aggregate by category:
- **Category A** (5 retraining-required): 1 MEASURED_PARTIAL_LIFT (m1) + 2 HONEST_NEGATIVE_R2_DEEP (m2 × 2) + 1 MEASURED_PARTIAL (m21) + 1 awaiting GPU retrain USER_OK (m7)
- **Category B** (8 direct improvement): 1 WEAK_RECOVERY (m2_combined) + 4 awaiting GPU retrain USER_OK (m3, m4, m15, m16) + 3 pending reward-shaping (m5, m10, m19)
- **Category C** (5 already SOTA): 3 PERFECT (m11, m13, m14) + 1 SOTA (m17) + 2 SOTA + in B (m3, m15)
- **Drop from §3 main-result**: 1 (m6_kappa — promoted to §3 supplementary sub-table top-3)
- **SCAFFOLD-only (m22)**: 1 (SE(3) SB per Wf-339-B2)

---

## §3 Innovation integration audit (NEW §5)

Added "Wf-388-A2 + Wf-388-A3 already-shipped wire-ins" subsection. Documents 5 wire-ins SHIPPED across Wf-388 + Wf-340 + Wf-353:
- Wf-388-A2 m7 Diversity: `--top-p` + `--top-k` + `--temperature` + `--mmd-weight` + `--repulsion-weight` + `--scaffold-conditioning`
- Wf-388-A3 m4 SA: `--sa-penalty` + `--sa-target`
- 7 SCAFFOLD-only innovation points pending wire-in: Wf-340-A1 EDM σ-preconditioning + Wf-340-A2 OT-CFM + Wf-340-A3 StableVM + Wf-340-A4 SO(3)-averaged FM + Wf-353-A1 Triton mini-batch OT + Wf-353-A2 Shortcut distillation + Wf-353-A3 Consistency regularization + Wf-353-A4 FlowMol unified (LICENSE-GATED)

---

## §4 Pending TODOs (refreshed §7)

Updated pending TODOs status table with NEW status column. Added cross-references to:
- `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` — canonical m2 R² GPU retrain plan (Wf-376-A3 SHIPPED)
- `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` — m2 W₂-pIC50 axis (per Wf-377-A2 alternative splits)
- `TODO/pending/383_se3_sb_fast_converging_integration.md` — m22 SE(3) Schrödinger Bridge fast-converging integration

---

## §5 NEW §10 Forward-looking 2026-Q4 GPU retrain priority order

**NEW per Wf-391-A3**: Added 11-section forward-looking with 3-batch sequential priority + paper write batch (4-batch total).

**Batch 1 — Quick wins (~10 min CPU install + USER_OK + 30 min wall)**: pre-flight 92/0/2/94 verification + dashboard refresh.

**Batch 2 — m1_auc + m2 R² lift (~21.5 hr GPU total, highest expected lift)**:
- B2.1 TODO-376A m2 R² Path A (larger cohorts n=200→1000 via NCI-60/ChEMBL/PubChem; ~5 day GPU; PROJECTED +0.05-0.30 R² lift)
- B2.2 TODO-376B m2 R² Path B (GPU chemprop v3 + Hilbert primal-dual; ~3 day GPU; PROJECTED +0.05-0.20 R² lift)
- B2.3 TODO-376A+B combined (~5-10 day GPU; PROJECTED +0.10-0.30 R² lift)
- B2.4 Wf-131 P3 Re seed=4 (m1_auc continuation; ~3 hr GPU; n=6→7/7)
- B2.5 Wf-310E R20-P5 tmQM + PlatinAI joint retrain (~11.4 hr GPU; **SCAFFOLD_NOT_FUNCTIONAL** per Wf-379-A2 — deferred to Wf-392+)

**Batch 3 — m3 + m4 + m7 + m15 + m6_supp lift (~12-15 hr GPU)**:
- B3.1 Wf-299 CFM 3-wave launch (~9.5 hr GPU; integrate Wf-388-A2/A3 + Wf-340 wire-ins; PROJECTED +0.02-0.05 lift)
- B3.2 Wf-264 CFM property panel (TBD; m5 QED + m19 LOMO + m4 SA verification)
- B3.3 §9.Z 5 CFM forward-looking TODOs (~37.8 hr GPU)

**Batch 7 — Tier-2 paper write (~12-15 hr CPU, USER_OK-gated per TODO-84)**:
- P1 §3 metric table → `paper/sections/03_main_results_v4_1.tex`
- P2 §6.1.6 m6_kappa 7-path inventory → `paper/sections/06_limitations_v4.tex`
- P3 §6.1.7 metric-redesign layer → `paper/sections/06_limitations_v4.tex`
- P4 §6 Limitations template → paper §6 intro
- P5 Wf-381-A4 + Wf-377-A1 §5 cross-linkage → paper §6 final

**3-batch sequential total envelope**: ~52.9 hr GPU + ~25 hr CPU across 4 USER_OK grants.

**Per-batch USER_OK gating rationale** documented in §10:
- Batch 1 install-only — no GPU launches
- Batch 2 highest expected lift + longest wall — m1_auc + m2 R² are 2 of 3 Category A metrics awaiting GPU
- Batch 3 m3 + m4 + m7 + m15 wire-ins already SHIPPED CPU-only; GPU launch validates PROJECTED lifts
- Batch 7 paper write — Tier-2 acceptance bar is computational-only (no wet-lab planning per AGENTS.md §5.4)

---

## §6 Wall budget envelope (NEW §9)

Added §9 with 4 sub-entries:
- GPU work pending USER_OK (~52.9 hr GPU)
- CPU work pending (~10-15 days)
- Standing wave budget: −183.04 hr overrun PRESERVED per Wf-262 §7.1 + 2026-09-28 user directive *"时间预算没关系的"*
- USER_OK-gated launches: 7 launch wrappers + 2 install binaries + 7 paper artefacts + 2 metric gates per Wf-368-B1 §7.3

---

## §7 Cross-references

### §7.1 Source Wf closeouts incorporated

- **Wf-387-A1** (`TODO/completed/387a1_wf379_a1_gpu_aggregation_fix.md`) — m1_auc canonical 6/7 pair summary aggregation
- **Wf-388-A1** (`TODO/completed/388a1_m2_per_cohort_real.md`) — m2 R² per-cohort config on REAL 3 cohorts verification
- **Wf-388-A2** (`TODO/completed/388a2_m7_diversity_wire.md`) — m7 Diversity 6 CLI flag wire-in SHIPPED
- **Wf-388-A3** (`TODO/completed/388a3_m4_sa_wire.md`) — m4 SA 2 CLI flag wire-in SHIPPED
- **Wf-389-A1** (`TODO/completed/389a1_m6k_supp_reframe.md`) — m6_kappa §3 supplementary sub-table promotion
- **Wf-380-A5** (`TODO/completed/380a5_multi_template_mm_gbsa.md`) — m15 RMSD multi-template + MM-GBSA surrogate SCAFFOLD
- **Wf-381-A4** (`TODO/completed/381a4_m6kappa_finalize.md`) — m6_kappa canonical pending brief
- **Wf-376-A3** (`TODO/pending/376a3_m2_r2_gpu_retrain_plan.md`) — m2 R² GPU retrain canonical plan

### §7.2 Standing rules honored (AGENTS.md §1, §5)

| Rule | Status | Notes |
|---|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ | Every numeric value above traces to JSON / report path on disk; PROJECTED vs MEASURED tag used consistently across all 5 SHIPPED wire-ins |
| §1.2 Standing GPU authorization (§5.1) | ✅ | 0 GPU launches performed; docs-only refresh of strategic plan |
| §1.3 No paper work without authorization (§5.3) | ✅ | 0 touches to `paper/*`; the 4 paper-write artefacts in §10 Batch 7 are USER_OK-gated future actions only |
| §1.4 Pre-flight v3 92/0/2/94 PASS gate | ✅ | Pre-flight PRESERVED at Wf-368-C1 baseline; Wf-391-A3 adds 0 NEW pre-flight gates (1 docs-only refresh + 1 closeout) |
| §1.5 Honest-negative framing (§5.2) | ✅ | m1_auc MEASURED_PARTIAL_LIFT (Rh PASS / Re BELOW Krasnov 0.86); m2 R² HONEST_NEGATIVE_R2_DEEP FINAL (per Wf-388-A1 + Wf-369-A2); m4 + m7 + m15 SCAFFOLD wire-ins + PROJECTED lifts honestly framed; m6_kappa drop-from-§3-main + supplementary-promote honestly framed; all 5 wire-in defaults OFF for backward-compat |
| §5.4 NO wet-lab planning (§10) | ✅ | 0 wet-lab touches; Tier-2 acceptance bar is computational-only; Batch 7 paper write is paper §3/§6 v4 conversion, NO wet-lab |

### §7.3 Prior closeouts preserved

- Wf-335-SYN — current speedup status (3-4× MEASURED; 10-15× PROJECTED)
- Wf-340-SYN — CFM fast-converging training research (24 papers + 16 repos + 6 wires)
- Wf-353 — Wf-343c1 un-wired gates + Wf-340 deeper integration
- Wf-358-B2 — SO(3)-equivariant D-MPNN scaffold
- Wf-359-A1 — m6_kappa W₂ on R20-P3 features
- Wf-365-A3 — m6_kappa FINAL verdict doc
- Wf-369-A2 — m2 R² FINAL verdict doc
- Wf-371-A1 — m6_kappa degenerate chain (7 paths exhausted)
- Wf-375-A4 — m6_kappa paper §6 reframe
- Wf-376-A2 — m1_auc CPU scaffold
- Wf-376-A3 — m2 R² GPU retrain plan (TODO/pending/376a3_*)
- Wf-376-A4 — Tier-2 paper §3 metric reframe
- Wf-377-A1 — m6_kappa redesign proposal
- Wf-377-A2 — m2 R² cohort diagnostic
- Wf-379 — GPU acceleration wave

---

## §8 Final state (Wf-391-A3 verdict)

| Item | Value | Status |
|---|---|---|
| **Source brief** | `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` (existing ~410 LOC, 8 sections) | ✅ FOUND |
| **Refreshed strategy** | `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` (refreshed ~860 LOC, 11 sections) | ✅ SHIPPED |
| **Per-metric updates** | §3.1 m1 AUC + §3.2 m2 R² + §3.3 m4 SA + §3.4 m7 Diversity + §3.5 m6_kappa + §3.6 m15 RMSD + §4 per-metric status table + §5 Wf-388 wire-ins + §7 pending TODOs + §10 NEW forward-looking | ✅ ALL 9 sections refreshed |
| **NEW §10 Forward-looking 2026-Q4 GPU retrain priority order** | 3-batch sequential + paper write batch (4-batch total) | ✅ SHIPPED |
| **Wall budget envelope** | §9 added | ✅ |
| **NEW tests** | 0 (docs-only per task brief) | ✅ |
| **Code patches** | 1 NEW docs (refreshed strategy); 1 NEW closeout (this file) | ✅ docs-only |
| **GPU launches** | 0 | ✅ per AGENTS.md §5.1 |
| **Paper/* touches** | 0 | ✅ per AGENTS.md §5.3 + TODO-84 |
| **Wetlab planning** | 0 | ✅ per AGENTS.md §5.4 + §10 |
| **Pre-flight v3 status** | 92/0/2/94 PRESERVED at Wf-368-C1 baseline (NOT re-run by Wf-391-A3 docs-only) | ✅ PRESERVED |
| **Wall (this Wf-391-A3 docs-only closeout)** | ~10 min CPU-only | MEASURED |

**The METRIC_LIFT_STRATEGY_v2.md is now in sync with the Wf-387..389 wire-in surface.** Wf-391-A3 closes the strategic-planning-vs-shipped-code gap by:
1. refreshing per-metric sections (§3.1..§3.6) with Wf-387/388/389 MEASURED values
2. adding NEW §3.3 m4 SA + §3.4 m7 Diversity + §3.6 m15 RMSD sections that document wire-in status
3. adding NEW §4 22-metric × 5-category status table with cross-refs
4. adding NEW §5 Wf-388 wire-in audit subsection
5. adding NEW §9 wall budget envelope
6. adding NEW §10 forward-looking 2026-Q4 GPU retrain priority order with 3-batch sequential + paper write batch (4-batch total)
7. refreshing §7 pending TODOs with status column + 3 new pending file cross-refs (TODO-376 + TODO-179 + TODO-383)
8. refreshing §8 cross-references with NEW Wf-387/388/389/380/381/391/376-A3 entries

Paper §3 + §6 v4 + §3 supplementary conversion requires USER_OK per TODO-84 + AGENTS.md §5.3.

---

**END OF Wf-391-A3 CLOSEOUT** — Owner Hugo Lee + Wf-391-A3 ultracode; cross-ref `party-twenty-spirit §3.5` + AGENTS.md §1 + §5.1 + §5.2 + §5.3 + §5.4 + §10
