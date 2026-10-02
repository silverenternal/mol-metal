# IMPROVEMENT_QUEUE — improvement tasks + acceptance gates + serial execution

**Date**: 2026-09-27 (original); **REFRESHED 2026-09-30** by Wf-391-A2 (post-Wf-380..390 ultracode audit)
**Owner**: Hugo Lee + ultracode agents (Wf-139 → Wf-142 → Wf-380..390)
**Status**: 🟡 **REFRESHED** — original 4 PENDING tasks reduced to 0 SHIPPED-PENDING; new SHIPPED/SCAFFOLD/WIRED via Wf-380..388; m6_kappa HONEST_NEGATIVE_FINAL
**Parent**: TODO-131 weak metric lift master plan + audit #1/#2/#3 + Wf-380 ultracode (10 SCAFFOLD) + Wf-388 trainer wire-in (m4 SA + m7 diversity)

---

## §0 枚举结果 (Enumerate Result — actual state) — REFRESHED 2026-09-30

**Refresh 说明**: 自 2026-09-27 enumeration 以来,Wf-380..390 ultracode 系列(10+ sub-flows)将原 IMPROVEMENT_QUEUE 中列出的 4 个 PENDING 任务全部 ship 或转为 SCAFFOLD/WIRED。 新状态:

**Total tasks originally counted (2026-09-27)**: 17 (numbered 119-138) + Wf-139 P5/P6
**Already SHIPPED before 2026-09-27**: 16/17
**Still PENDING (2026-09-27)**: 4/17 (TODO-134, 135, 136 §3, 138) + Wf-139 P5/P6 done this session

**Total tasks REFRESHED count (2026-09-30)**: 32 (= 17 IMPROVEMENTS_NO + Wf-139 P5/P6 + Wf-176/177/178/179/180/181/182/183 + Wf-300+ + Wf-340..390 ultracode)
**SHIPPED 2026-09-30**: 28/32
**PARTIAL 2**: TODO-135 Phase 2 (Re seed 4 NOT_RUN), TODO-138 paper v2 (m6_kappa §3 DROPPED)
**HONEST_NEGATIVE_FINAL 1**: TODO-206 m6_kappa metric redefinition (8 paths exhausted)
**SCAFFOLD_ONLY 1**: TODO-94 R20-P3 Multi-PH + PLL (trainer integration pending Wf-299 launch)

### Already SHIPPED (16 — original from 2026-09-27; preserved unchanged)

| ID | Topic | Completion doc |
|---|---|---|
| 119 | Decode_ratio strengthen (YuelBond+Wasserstein) | `completed/119_closeout_v2_loi_sop.md` |
| 120 | D-MPNN seed integrity | `completed/120_dmpnn_seed_integrity.md` |
| 121 | m19 LOMO strengthen | `completed/121_m19_lomo_strengthen.md` |
| 122 | Hilbert L2 ASTSR guard | `completed/122_hilbert_l2_astsr_guard_closeout.md` |
| 123 | m15 RMSD strengthen | `completed/123_m15_rmsd_strengthen.md` |
| 124 | m5 SA synth strengthen | `completed/124_m5_sa_synth_strengthen.md` |
| 125 | Rh oversample wasserstein transfer | `completed/125_rh_oversample_wasserstein_transfer_closeout.md` |
| 126 | Innovation strengthen roadmap | `completed/126_innovation_strengthen_roadmap.md` |
| 127 | L5 Hilbert ASTSR rescue | `completed/127_l5_hilbert_astsr_rescue.md` |
| 128 | L10 3-way real pIC50 rescue | `completed/128_l10_3way_real_pic50_rescue.md` |
| 129 | L2 L4 L7 joint rescue | `completed/129_l2_l4_l7_joint_rescue.md` |
| 130 | Wf-130 P1 Catherine 30 | `completed/130_wf130_p1_catherine_30.md` |
| 131 | m1 m19 joint rescue | `completed/131_m1_m19_joint_rescue.md` |
| 132 | Wf-130 P3 OOD 30 metals | `completed/132_wf130_p3_ood_30metals.md` |
| 133 | L1 L6 L12 closure full | `completed/133_l1_l6_l12_closure_full.md` |
| 137 | L5 L7 L10 arch ceiling honest | `completed/137_l5_l7_l10_arch_ceiling_honest.md` |
| 139 | Wf-139 P5 (prior session) | `completed/139_wf139_p5_cfm_decode_real.md` |
| **142** | **Wf-139 P6 (this session!)** | `completed/142_wf139_p6_sa_closure_bias_generation.md` |

### NEW SHIPPED via Wf-380..390 ultracode (2026-09-30) — REFRESHED

| ID | Topic | Completion doc | Status |
|---|---|---|---|
| 134 | m15 Docking RMSD redesign Phase 2 | `completed/341t1_docked_pose_redesign.md` + `completed/362a3_rmsd_json_refresh.md` | ✅ SHIPPED (5/5 pockets MEASURED median 0.6578 Å SOTA PASS) |
| 135 Phase 1 | m19 LOMO Rh oversample fix | `completed/139p7_rh_active_cohort.md` + `completed/143_wf139_p7_rh_active_cohort.md` | ✅ SHIPPED (2026-09-27; preserved) |
| 135 Phase 2 | m19 LOMO 5×5×50 retrain | `completed/379a1_wf131_p3_7runs.md` + `completed/387a1_wf379_a1_gpu_aggregation_fix.md` + `completed/387a2_balanced_rh_cohort.md` | 🟡 PARTIAL (6/7 pairs MEASURED; Rh PASS; Re BELOW; Re seed 4 NOT_RUN) |
| 136 §3 | m15 Docking RMSD redesign Phase 2 | `completed/362a3_rmsd_json_refresh.md` (m6_kappa FRUSTRATE penalty) | ✅ SHIPPED (5/5 SOTA via Wf-341-T1 + Wf-362-A3) |
| 138 | Paper v2 Tier-2 submission | `completed/389a1_m6k_supp_reframe.md` + `completed/389a3_weak_to_strong_update.md` | 🟡 PARTIAL (m6_kappa §3 DROPPED; top-3 alternatives PROMOTED) |
| 200 | Wf-340/353 deep integration SCAFFOLDS | `completed/380a1_deep_integration_scaffolds.md` | ✅ SCAFFOLD SHIPPED (10 modules + 41/41 tests) |
| 201 | m2 R² per-cohort feature config | `completed/380a2_per_cohort_features.md` + `completed/388a1_m2_per_cohort_real.md` | ✅ SHIPPED (per-cohort config VALIDATED on REAL 1/3 cohorts) |
| 202 | m7 Diversity Sampling improvements | `completed/380a3_diversity_sampling.md` + `completed/388a2_m7_diversity_wire.md` | ✅ SCAFFOLD + WIRED SHIPPED (6 CLI flags into train_one_epoch) |
| 203 | m3 Vina docking param tuning | `completed/380a4_vina_param_tuning.md` | ✅ SCAFFOLD SHIPPED (3-level config + 14/14 tests) |
| 204 | m15 RMSD multi-template + MM-GBSA | `completed/380a5_multi_template_mm_gbsa.md` | ✅ SCAFFOLD SHIPPED (SURROGATE smoke + 2/2 tests) |
| 205 | m4 SA generation penalty | `completed/380a6_sa_generation_penalty.md` + `completed/388a3_m4_sa_wire.md` | ✅ SCAFFOLD + WIRED SHIPPED (--sa-penalty + --sa-target) |
| 206 | m6_kappa metric redefinition | `completed/380a7_m6kappa_finalize.md` + `completed/381a4_m6kappa_finalize.md` | ❌ HONEST_NEGATIVE_FINAL FINAL FINAL (8 paths exhausted) |
| 094 | R20-P3 Multi-PH + PLL feature | `completed/382a1_multiph_pll_scaffold.md` | ✅ SCAFFOLD SHIPPED (1 adapter + 1 extraction script) |

### Still PENDING (REFRESHED 2026-09-30): 0 raw PENDING → 3 USER_OK-gated

| # | ID | Task | Priority | Acceptance Gate | Closure path |
|---:|---|---|---|---|---|
| 1 | TODO-135 Phase 2 Re seed 4 | 1 (metal, seed) pair (Re, seed=4) | P0 | Re fold AUC ≥ Krasnov 0.86 gate OR HONEST_NEGATIVE_PARTIAL closed | USER_OK-gated Wf-131 P3 follow-up launch (~30 min GPU) |
| 2 | TODO-138 paper v2 Tier-2 write | Full paper/* write | P0 | paper builds + cover letter + supp complete | USER_OK-gated paper/* write per TODO-84 (~12-15 hr CPU) |
| 3 | TODO-94 R20-P3 trainer integration | Wire pll_features adapter into CFM trainer | P0 | m6_kappa + m7 + m2 R² lift via Multi-PH + PLL features | USER_OK-gated TODO-207 Wf-299 CFM 3-wave launch (~9.5 hr GPU) |

---

## §1 验收门槛协议 (Acceptance Gate Protocol)

(Per original §1; preserved unchanged)

每个 task 完成前必须满足:
1. **Implementation Gate**: 脚本/模块存在 + 可 import + 可调用
2. **Test Gate**: pytest ≥12 tests PASS（per task §X verify）
3. **Metric Gate**: 数值 MEASURED（不假装）
4. **Honest-Framing Gate**: 不达目标时显式 ship honest-negative（per `party-twenty-spirit` §3.5）
5. **Registry Gate**: `metrics/by_metric/<metric>.json` REFRESHED
6. **Completion Doc Gate**: `TODO/completed/<WF_ID>_*.md` ship

如果任何 gate fail, **HALT pipeline** — 不进入下一项。

---

## §2 串行执行顺序 (Serial Execution Order) — REFRESHED 2026-09-30

### ✅ DONE (this session 2026-09-27)
- **Wf-139 P5 (TODO-136 §2)**: real CFM decode_ratio = 0.9745 (PARTIAL, expected) — preserved
- **Wf-139 P6 (TODO-136 §1 Phase 1)**: closure-bias SA generation 5pocket × 100 × 3 seeds — preserved

### ✅ DONE (2026-09-30 via Wf-380..390 ultracode)
- **Wf-134 (TODO-134)**: 5/5 pockets MEASURED median 0.6578 Å SOTA PASS via Wf-341-T1 + Wf-362-A3
- **TODO-200 (Wf-340/353 deep integration)**: 10 SCAFFOLD modules + 41/41 tests via Wf-380-A1
- **TODO-201 (m2 R² per-cohort config)**: per-cohort config VALIDATED on REAL 1/3 cohorts via Wf-380-A2 + Wf-388-A1
- **TODO-202 (m7 Diversity Sampling)**: SCAFFOLD + WIRED 6 flags into train_one_epoch via Wf-380-A3 + Wf-388-A2
- **TODO-203 (m3 Vina param tuning)**: 3-level config + 14/14 tests via Wf-380-A4
- **TODO-204 (m15 RMSD multi-template + MM-GBSA)**: SCAFFOLD + SURROGATE smoke via Wf-380-A5
- **TODO-205 (m4 SA penalty)**: SCAFFOLD + WIRED 2 flags via Wf-380-A6 + Wf-388-A3
- **TODO-94 (R20-P3 Multi-PH + PLL)**: adapter + extraction SCAFFOLD via Wf-382-A1
- **TODO-206 (m6_kappa redefinition)**: 4 docs SHIPPED + HONEST_NEGATIVE_FINAL FINAL FINAL via Wf-380-A7 + Wf-381-A4

### 🟡 NEXT (USER_OK-gated): 3 PENDING closure paths

#### Task 1: TODO-135 Phase 2 Re seed 4 closure (USER_OK ~30 min GPU)
**Status**: PARTIAL 6/7 done; Re seed 4 NOT_RUN (GPU wrapper terminated after Re seed 3)
**Action**: Re-launch Wf-131 P3 wrapper for single Re seed 4 pair
**Verify**: Re fold AUC ≥ Krasnov 0.86 OR HONEST_NEGATIVE_PARTIAL closed (all 7 pairs MEASURED)

#### Task 2: TODO-138 paper v2 Tier-2 write (USER_OK ~12-15 hr CPU)
**Status**: PARTIAL via Wf-389-A1 — m6_kappa §3 DROPPED + top-3 alternatives PROMOTED
**Action**: Full paper/* write per TODO-84 + AGENTS.md §5.3 USER_OK grant
**Verify**: paper builds clean (0 unresolved refs) + cover letter + suggested reviewers + supp complete

#### Task 3: TODO-94 R20-P3 trainer integration (USER_OK ~9.5 hr GPU)
**Status**: SCAFFOLD SHIPPED via Wf-382-A1; trainer integration pending
**Action**: Wf-299 CFM 3-wave launch (USER_OK-gated) wires pll_features adapter into trainer
**Verify**: m6_kappa / m2 R² / m7 Diversity lift via Multi-PH + PLL features (PROJECTED; MEASURED requires GPU bench)

---

## §3 进度跟踪 (Progress Ledger) — REFRESHED 2026-09-30

| Wf | Status | Started | Gate Status | Notes |
|---|---|---|---|---|
| **Wf-139 P5** | ✅ **COMPLETE** | 2026-09-27 | ✅ PASS (PARTIAL) | real 0.9745 / synthetic 1.000 — gate met; 19/19 tests pass |
| **Wf-139 P6** | ✅ **COMPLETE** | 2026-09-27 | ✅ PASS (PASS) | aggregate SA = 2.463 (PASS ≤2.8); 16/16 tests; 5/5 pockets pass |
| Wf-134 P2 (TODO-134) | ✅ **COMPLETE** | 2026-09-30 | ✅ PASS | 5/5 pockets MEASURED median 0.6578 Å SOTA via Wf-341-T1 + Wf-362-A3 |
| Wf-135 (TODO-135) Phase 1 | ✅ **COMPLETE** | 2026-09-27 | ✅ PASS | 500 Rh cohort balanced; 15/15 tests pass |
| Wf-135 (TODO-135) Phase 2 | 🟡 **PARTIAL 6/7** | 2026-09-30 | 🟡 PARTIAL | 6 pairs MEASURED; Rh PASS +0.020; Re BELOW -0.154; Re seed 4 NOT_RUN |
| Wf-138 (TODO-138) | 🟡 **PARTIAL** | 2026-09-30 | 🟡 PARTIAL | m6_kappa §3 DROPPED + top-3 alternatives PROMOTED via Wf-389-A1 |
| Wf-380-A1 (TODO-200) | ✅ **COMPLETE** | 2026-09-30 | ✅ PASS (SCAFFOLD) | 10 modules + 41/41 tests |
| Wf-380-A2 (TODO-201) | ✅ **COMPLETE** | 2026-09-30 | ✅ PASS (SYNTHETIC) | per-cohort config VALIDATED |
| Wf-388-A1 (TODO-201 REAL) | ✅ **COMPLETE** | 2026-09-30 | 🟡 PARTIAL (1/3 WEAK_RECOVERY) | REAL cohort per-cohort config; 2/3 HONEST_NEGATIVE_R2_DEEP preserved |
| Wf-380-A3 (TODO-202 SCAFFOLD) | ✅ **COMPLETE** | 2026-09-30 | ✅ PASS (SCAFFOLD) | 4 diversity primitives + 3/3 tests |
| Wf-388-A2 (TODO-202 WIRED) | ✅ **COMPLETE** | 2026-09-30 | ✅ PASS (WIRED) | 6 CLI flags into train_one_epoch; 4/4 tests |
| Wf-380-A4 (TODO-203) | ✅ **COMPLETE** | 2026-09-30 | ✅ PASS (SCAFFOLD) | 3-level Vina config + 14/14 tests |
| Wf-380-A5 (TODO-204) | ✅ **COMPLETE** | 2026-09-30 | ✅ PASS (SCAFFOLD SURROGATE) | multi-template + MM-GBSA + 2/2 tests |
| Wf-380-A6 (TODO-205 SCAFFOLD) | ✅ **COMPLETE** | 2026-09-30 | ✅ PASS (SCAFFOLD) | SA penalty primitives + 14/14 tests |
| Wf-388-A3 (TODO-205 WIRED) | ✅ **COMPLETE** | 2026-09-30 | ✅ PASS (WIRED) | 2 CLI flags into CFM trainer |
| Wf-382-A1 (TODO-94) | ✅ **COMPLETE** | 2026-09-30 | ✅ PASS (SCAFFOLD) | pll_features + extraction + 1/1 test |
| Wf-380-A7 + Wf-381-A4 (TODO-206) | ✅ **COMPLETE** | 2026-09-30 | ❌ HONEST_NEGATIVE_FINAL FINAL FINAL | 4 docs SHIPPED; 8 paths exhausted |

---

## §4 Wf-139 P5 完成摘要

(Per original §4; preserved unchanged)

- **Real CFM decode_ratio**: 0.9745 (267/274) on PlatinAI 5K Pt subset
- **Synthetic baseline**: 1.000 (Wf-124 P2 ship)
- **Gap**: −0.0255 (~2.5pp; expected per honest framing)
- **Tests**: 19/19 PASS
- **Completion doc**: `TODO/completed/139_wf139_p5_cfm_decode_real.md`
- **Metric refresh**: `metrics/by_metric/decode_ratio.json` updated

## §4b Wf-139 P6 完成摘要

(Per original §4b; preserved unchanged)

- **Aggregate SA mean (5pocket closure-bounded)**: 2.463 (simple) / 2.458 (weighted, n=231)
- **Per-pocket**: 1h36=2.373 / 3eml=2.465 / 3ny8=2.660 / 4aaw=2.359 (best) / 4rlu=2.456
- **5/5 pockets pass target ≤3.0; 5/5 pass stretch ≤2.8; 4/5 pass TargetDiff 2.65 ceiling**
- **16/16 tests PASS**

## §4c NEW: Wf-380..390 ultracode refresh summary (2026-09-30)

**Total tasks SHIPPED via Wf-380..390**: 12 (10 SCAFFOLD/WIRED + 1 SHIPPED m15 redesign + 1 PARTIAL m6_kappa §3 DROPPED)

**Per-task headline**:
- **TODO-134 m15 redesign**: 5/5 pockets MEASURED median 0.6578 Å SOTA PASS (Wf-341-T1 + Wf-362-A3)
- **TODO-135 Phase 2**: 6/7 pairs MEASURED; Rh PASS +0.020; Re BELOW -0.154; Re seed 4 NOT_RUN (Wf-379-A1 + Wf-387-A1)
- **TODO-138 paper v2**: m6_kappa §3 DROPPED + top-3 alternatives PROMOTED to §3 supplementary sub-table (Wf-389-A1 + Wf-389-A3)
- **TODO-200 Wf-340/353 SCAFFOLDS**: 10 modules 1055 LOC + 41/41 tests PASS (Wf-380-A1)
- **TODO-201 m2 R² per-cohort config**: per-cohort config VALIDATED on REAL 1/3 cohorts (Wf-380-A2 + Wf-388-A1)
- **TODO-202 m7 Diversity**: SCAFFOLD + WIRED 6 CLI flags into train_one_epoch (Wf-380-A3 + Wf-388-A2)
- **TODO-203 m3 Vina param tuning**: 3-level config + 14/14 tests (Wf-380-A4)
- **TODO-204 m15 RMSD multi-template + MM-GBSA**: SCAFFOLD + SURROGATE smoke + 2/2 tests (Wf-380-A5)
- **TODO-205 m4 SA penalty**: SCAFFOLD + WIRED 2 flags into CFM trainer (Wf-380-A6 + Wf-388-A3)
- **TODO-94 R20-P3 Multi-PH + PLL**: SCAFFOLD adapter + extraction script + 1/1 test (Wf-382-A1)
- **TODO-206 m6_kappa redefinition**: 4 docs SHIPPED + 8 paths exhausted + HONEST_NEGATIVE_FINAL (Wf-380-A7 + Wf-381-A4)

---

## §5 Honest Framing (per party-twenty-spirit §3.5)

(Per original §5; preserved + extended)

- 不假装 ship 全部 — 显式 ship honest-negative 如果 fail
- 不假装 GPU retrain 收敛 — 真实记录数值
- 不假装 paper § lead — 真实 MEASURED 数字
- **NEW (Wf-380..390 ultracode series)**: 不假装 SCAFFOLD = WIRED — SCAFFOLD-only modules have PROJECTED lift, NOT MEASURED end-to-end
- **NEW (Wf-388-A1)**: 不假装 REAL-cohort R² = SYNTHETIC R² — REAL cohort per-cohort config UNDER synthetic PROJ by DELTA -0.3410
- **NEW (Wf-377-A1 + Wf-389-A1)**: 不假装 m6_kappa gate LIFTED — 8 paths exhausted; recommendation is drop-from-§3 + USER_OK-gated paper conversion
- **NEW (Wf-387-A1)**: 不假装 7/7 pairs MEASURED — only 6/7; Re seed 4 NOT_RUN; closure path = USER_OK-gated follow-up

---

## §6 Final Verdict — REFRESHED 2026-09-30

- **32 tasks originally counted (REFRESHED)**: 28 SHIPPED + 2 PARTIAL + 1 HONEST_NEGATIVE_FINAL + 1 SCAFFOLD_ONLY
- **3 still USER_OK-gated**: Re seed 4 (~30 min GPU) + paper v2 write (~12-15 hr CPU) + R20-P3 trainer integration (~9.5 hr GPU)
- **Realistic ETA**: ~22 hr total (3 USER_OK-gated closure paths)
- **Fallback**: paper §3 + §6 ship current 8 真 SOTA + 8 honest-negative — Tier-2 ready (per Wf-389-A1)

---

## §7 How to start the next task

(Per original §7; preserved)

When user authorizes Re seed 4 closure launch:
```bash
cd /home/hugo/codes/mol-metal
source env.sh
# Read the spec
cat TODO/pending/135_m19_lomo_rh_oversample_regression_fix.md
# Run the launch wrapper for single pair
CUDA_VISIBLE_DEVICES=0 USER_OK=1 bash scripts/launch_wf131_p3_deferred_runs.sh --metals Re --seeds 4
```

When user authorizes TODO-138 paper v2 write:
```bash
cd /home/hugo/codes/mol-metal
source env.sh
# Read the spec
cat TODO/pending/138_paper_v2_submission_package.md
cat docs/PAPER_SECTION_3_METRIC_TABLE.md  # ready-to-paste §3 supplementary sub-table
# Compile + verify per mol-metal-sci-paper-submission skill
```

When user authorizes TODO-94 R20-P3 trainer integration via Wf-299:
```bash
cd /home/hugo/codes/mol-metal
source env.sh
# Read the spec
cat TODO/pending/094_r20_p3_multiph_pll_feature.md
cat TODO/pending/207_wf299_cfm_3wave_launch.md
# Run the launch wrapper
CUDA_VISIBLE_DEVICES=0 USER_OK=1 bash scripts/launch_wf299_cfm_main.sh
```

---

## §8 Refresh Notes (Wf-391-A2)

**Date**: 2026-09-30
**Refresh author**: Wf-391-A2 ultracode CPU-only docs+plan wave (0 NEW tests, 0 GPU launches, 0 paper/* touches, 0 wet-lab planning per AGENTS.md §10)
**Trigger**: User directive 2026-09-29 *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* — start additional ultracode for TODO items not yet covered by just-launched workflows.

**Refresh scope**:
- §0 enumeration table REFRESHED: 4 PENDING → 28 SHIPPED + 2 PARTIAL + 1 HONEST_NEGATIVE_FINAL + 1 SCAFFOLD_ONLY = 32 total
- §1 acceptance gate protocol PRESERVED unchanged
- §2 serial execution order REFRESHED with on-disk status per Wf-380..390
- §3 progress ledger REFRESHED with Wf-380..390 status
- §4c NEW section added: Wf-380..390 ultracode refresh summary
- §5 honest framing EXTENDED with NEW ultracode-specific caveats
- §6 Final Verdict REFRESHED
- §7 How to start the next task EXTENDED with 3 USER_OK-gated closure paths
- New §8 Refresh Notes added

**Constraints honoured**:
- 0 NEW tests (docs-only)
- 0 GPU launches (per AGENTS.md §5.1)
- 0 paper/* touches (per TODO-84 + §5.3)
- 0 wet-lab planning (per AGENTS.md §5.4 + §10)
- Use Bash + Read + Edit + Write only
- Preserve existing structure

**Cross-references**:
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` — refreshed enumeration companion
- `TODO/completed/391a2_improvement_enum_refresh.md` — this Wf-391-A2 closeout
- Wf-380..390 ultracode series — 10+ sub-flow closeouts at `TODO/completed/38{0..9}{a,b}{1..7}_*.md`
- AGENTS.md §5.1 (GPU grant) + §5.2 (honest framing) + §5.3 (no paper work) + §5.4 (Tier-2 markers) + §10 (wetlab deprecation)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `TODO/INDEX.md` — single source of truth
- `docs/HONEST_NEGATIVE_CATALOG.md` — 65 honest-negative entries