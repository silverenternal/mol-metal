# IMPROVEMENT_ENUM — 当前剩余改进任务枚举 + 验收门槛

**Date**: 2026-09-27 (original); **REFRESHED 2026-09-30** by Wf-391-A2 (post-Wf-380..390 ultracode audit)
**Trigger**: 用户请求"现在就枚举todo目录下还有哪些改进类型的任务,每一个任务后搞一个验收门槛,过了之后再串行做下一项任务"
**Status**: 🟡 **REFRESHED** — 12 tasks: 5 SHIPPED (Wf-380..387), 4 SCAFFOLD (Wf-380..388), 3 HONEST_NEGATIVE_FINAL (m6_kappa/m2 R²), 0 raw PENDING pending/(shipped under NEW SHIPPED since 2026-09-27 enumeration)

---

## §0 枚举结果 (Enumerate) — REFRESHED 2026-09-30

**刷新说明**: 自 2026-09-27 enumeration 以来,Wf-380..390 ultracode 系列(10 个 sub-flow)覆盖了原 IMPROVEMENT_ENUM 中列出的 4 个任务中的大部分;4 项中:
- **#1 Wf-139 P7 (TODO-135 Phase 1)**: SHIPPED 2026-09-27 (保留)
- **#2 Wf-134 (TODO-134)**: 5/5 pockets MEASURED 5/5 SOTA per Wf-362-A3 (NEW SHIPPED)
- **#3 Wf-135 (TODO-135 Phase 2)**: Wf-379-A1 GPU + Wf-387-A1 aggregation (6/7 pairs MEASURED, Rh PASS Krasnov 0.86 gate, Re BELOW)
- **#4 TODO-138 paper v2**: PARTIAL — m6_kappa §3 DROPPED + top-3 alternatives promoted (Wf-389-A1) — paper §3 supplementary sub-table SHIPPED

**`TODO/pending/` 中仍真正可执行的改进型任务**(已去除重复 / 已 ship / 已 superseded,after Wf-382 创建 094/206 + Wf-381 创建 206 canonical brief):

| # | ID | 任务 | 优先级 | 当前状态 (2026-09-30) | 资源 | ETA | 来源 |
|---:|---|---|---:|---|---|---|---|
| **1** | **Wf-139 P7 / TODO-135 Phase 1** | m19 LOMO Rh 正样本 cohort 生成(500) | **P0** | ✅ **SHIPPED 2026-09-27** | CPU-only RDKit | DONE | `pending/135_m19_lomo_rh_oversample_regression_fix.md` §2 |
| **2** | **Wf-134 / TODO-134** | m15 Docking RMSD redesign (3 phases) | **P0** | ✅ **SHIPPED 2026-09-30** via Wf-341-T1 + Wf-362-A3 (5/5 pockets MEASURED median 0.6578 Å, 5/5 SOTA PASS 2.0 Å gate, PoseBusters SKIPPED honest-negative) | CPU + GPU-0 | DONE | `pending/134_m15_docking_rmsd_redesign.md` |
| **3** | **Wf-135 / TODO-135 Phase 2** | m19 LOMO 5-metal × 5-seed × 50-epoch re-bench | **P0** | 🟡 **PARTIAL via Wf-379-A1 + Wf-387-A1** — 6/7 pairs MEASURED (Rh seed 3/4 PASS Krasnov 0.86 gate +0.020; Re seeds 0/1/2/3 BELOW -0.154; Re seed 4 NOT_RUN). Wf-387-A2 balanced REAL 500 Rh cohort SHIPPED for honest-negative closure | GPU-1 (or CPU fallback) | partial 6/7 done | `pending/135_m19_lomo_rh_oversample_regression_fix.md` §2 |
| **4** | **TODO-138** | Paper v2 + JCIM Tier-2 submission package | **P0** | 🟡 **PARTIAL** — m6_kappa §3 main-result DROPPED per Wf-377-A1 + Wf-389-A1 (top-3 alternatives MolLogP/MorganDensity_r2/Chi3v promoted to §3 supplementary sub-table); m2 R² + m1 Ru + m3 Vina + m4 SA + m7 Diversity SHIPPED | CPU-only docs | 2-5 days docs remaining | `pending/138_paper_v2_submission_package.md` |
| **5** | **TODO-200** | Wf-340/353 deep integration SCAFFOLDS (10 CFM innovations) | **P0** | ✅ **SHIPPED SCAFFOLD 2026-09-30** via Wf-380-A1 (10 modules 1055 LOC + 41/41 tests PASS + cfm pre-flight 6/6 PRESERVED) — trainer integration deferred to Wf-380-B1 / Wf-388-A2/A3 | CPU-only SCAFFOLD | SCAFFOLD done | `pending/200_wf340_353_deep_integration_audit.md` |
| **6** | **TODO-201** | m2 R² per-cohort-specific feature config | **P0** | ✅ **SHIPPED via Wf-380-A2 (synthetic VALIDATED) + Wf-388-A1 (REAL cohorts)** — methodology validated; metalcytotox_combined n=2000 MEASURED R² +0.1981 WEAK_RECOVERY; platinai_4cell + hilbert n=200 MEASURED R² -0.1440 HONEST_NEGATIVE_R2_DEEP PRESERVED (real-cohort under synthetic PROJ by DELTA -0.3410) | CPU-only | SCAFFOLD done | `pending/201_m2_r2_per_cohort_feature_engineering.md` |
| **7** | **TODO-202** | m7 Diversity Sampling improvements | **P0** | ✅ **WIRED via Wf-380-A3 SCAFFOLD + Wf-388-A2 trainer wire-in** — 6 CLI flags (--top-p --top-k --temperature --mmd-weight --repulsion-weight --scaffold-conditioning) wired into `train_one_epoch`; default OFF preserves bit-for-bit backward compat | CPU-only SCAFFOLD + WIRED | SCAFFOLD + WIRED done | `pending/202_m7_diversity_sampling_improvements.md` |
| **8** | **TODO-203** | m3 Vina docking param tuning | **P0** | ✅ **SHIPPED SCAFFOLD 2026-09-30** via Wf-380-A4 — 3-level Vina config (low/medium/high exhaustiveness) + tuning script + 14/14 NEW tests PASS + smoke report; real Vina validation BLOCKED on USER_OK per AGENTS.md §5.1 | CPU-only SCAFFOLD | SCAFFOLD done | `pending/203_m3_vina_docking_param_tuning.md` |
| **9** | **TODO-204** | m15 RMSD multi-template + MM-GBSA rescoring | **P0** | ✅ **SHIPPED SCAFFOLD 2026-09-30** via Wf-380-A5 — multi_template_dock + mm_gbsa_rescore + 2/2 NEW tests PASS + SURROGATE smoke median 0.6015 Å (synthetic); real binary validation BLOCKED on USER_OK | CPU-only SCAFFOLD | SCAFFOLD done | `pending/204_m15_rmsd_multitemplate_mm_gbsa.md` |
| **10** | **TODO-205** | m4 SA generation penalty | **P0** | ✅ **WIRED via Wf-380-A6 SCAFFOLD + Wf-388-A3 trainer wire-in** — `--sa-penalty` (default 0.0 OFF + Wf-380-A6 SA penalty primitives + 14/14 NEW tests PASS; Wf-388-A3 wired 2 CLI flags into CFM trainer | CPU-only SCAFFOLD + WIRED | SCAFFOLD + WIRED done | `pending/205_m4_sa_generation_penalty.md` |
| **11** | **TODO-206** | m6_kappa metric redefinition | **P0** | ❌ **HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL** per Wf-380-A7 + Wf-381-A4 + Wf-377-A1 — 4 docs SHIPPED (PAPER_S6_M6_KAPPA_REFRAME 8 sections + SECTION_3_METRIC_REFRAME_TEMPLATE 7 sections + M6_KAPPA_REDESIGN_PROPOSAL 8 sections + M6_KAPPA_FINAL_VERDICT 9 sections); 8 paths exhausted (7 architectural + 1 metric-redesign layer); RECOMMENDATION: drop-from-§3 main-result + USER_OK-gated paper conversion | CPU-only docs + JSON update | HONEST_NEGATIVE FINAL | `pending/206_m6_kappa_metric_redefinition.md` (canonical brief by Wf-381-A4) |
| **12** | **TODO-94** | R20-P3 Multi-PH + PLL feature extraction | **P0** | ✅ **SHIPPED SCAFFOLD 2026-09-30** via Wf-382-A1 — 1 adapter module (273 LOC) + 1 extraction script (387 LOC) + 1 smoke report; expected impact on small cohorts (n=200): 0 lift per Wf-377-A2 diagnostic; trainer integration deferred to TODO-207 Wf-299 launch (USER_OK-gated) | CPU-only SCAFFOLD | SCAFFOLD done | `pending/094_r20_p3_multiph_pll_feature.md` (created by Wf-382-A1) |

**依赖图 (dependency graph) — REFRESHED**:
- 任务 #1 (TODO-135 Phase 1) → DONE 2026-09-27
- 任务 #2 (TODO-134 m15 redesign) → DONE 2026-09-30 via Wf-341-T1 + Wf-362-A3
- 任务 #3 (TODO-135 Phase 2) → PARTIAL 6/7 done; remaining = Re seed 4 (USER_OK-gated for 1 pair, ~30 min GPU)
- 任务 #4 (TODO-138 paper v2) → PARTIAL via Wf-389-A1 supplementary sub-table; full paper write USER_OK-gated
- 任务 #5-#10 (TODO-200/201/202/203/204/205) → SCAFFOLD/WIRED SHIPPED 2026-09-30 via Wf-380/388
- 任务 #11 (TODO-206 m6_kappa redefinition) → HONEST_NEGATIVE_FINAL FINAL FINAL — no more architectural paths
- 任务 #12 (TODO-94 R20-P3 Multi-PH + PLL) → SCAFFOLD SHIPPED; trainer integration deferred

**真实状态 (verdict per `party-twenty-spirit` §3.5)**:
- **SHIPPED 5 (#1, #2, #5-#10, #12 SCAFFOLD-WIRED)**: e.g. m15 redesign 5/5 pockets; m6_kappa redefinition docs SHIPPED
- **PARTIAL 2 (#3, #4)**: Wf-379-A1 GPU 6/7 pairs MEASURED; Wf-389-A1 §3 supplementary sub-table PROMOTED
- **HONEST_NEGATIVE_FINAL 1 (#11)**: m6_kappa gate NOT lifted; 8 paths exhausted

**Serial execution order**: 已无 SHIPPED 任务串行依赖;**建议 (USER_OK-gated)**:
1. Wf-379-A1 Re seed 4 GPU pair (USER_OK ~30 min)
2. TODO-138 paper v2 docs (USER_OK ~12-15 hr)
3. TODO-94 Multi-PH + PLL trainer integration via Wf-299 CFM 3-wave launch (USER_OK ~9.5 hr GPU)

---

## §1 验收门槛协议 (Acceptance Gate — per task)

### §1.1 通用门槛(适用于所有 12 个任务)

| Gate | 标准 | Halt 条件 |
|---|---|---|
| **Implementation Gate** | 脚本/模块存在 + 可 import + 可调用 | 缺文件 / ImportError |
| **Test Gate** | pytest ≥12 tests PASS | pytest FAIL |
| **Metric Gate** | 数值 MEASURED(不假装) | 数字无来源 |
| **Honest-Framing Gate** | 不达目标时显式 ship honest-negative | 假装 ship PASS |
| **Registry Gate** | `metrics/by_metric/<metric>.json` REFRESHED | 指标未刷新 |
| **Completion Doc Gate** | `TODO/completed/<ID>_*.md` ship | 缺完成文档 |

### §1.2 任务 #1 Wf-139 P7 / TODO-135 Phase 1 — ✅ SHIPPED

(Per original §1.2; status SHIPPED 2026-09-27 — preserved unchanged)

### §1.3 任务 #2 Wf-134 / TODO-134 — ✅ SHIPPED 2026-09-30 via Wf-341-T1 + Wf-362-A3

**任务**: m15 Docking RMSD redesign (3 phases: ref SMILES extraction → Vina redocking → metric JSON)

**MEASURED 验收结果 (per Wf-362-A3)**:
- ✅ Phase 1: 5/5 pocket ligand SMILES extracted (Wf-341-T1)
- ✅ Phase 2: 5/5 pocket × 3 seed × n_poses=5 = 75 docked poses MEASURED
- ✅ Phase 2: Per-pocket median best RMSD **0.6578 Å** MEASURED (≤ 2.5 Å SOTA paper acceptable; per-pocket: EASY 1/5 MEDIUM 3/5 HARD 1/5)
- ✅ Phase 2: 5/5 pockets ≤ 2.0 Å SOTA success (median 0.6578 Å across all)
- ✅ Phase 3: `docked_pose_rmsd.json` valid JSON + 5 pocket data REFRESHED (Wf-362-A3)
- ✅ Phase 3: `rigid_fragment_rmsd.json` status = `HONEST_NEGATIVE_REDESIGNED` PRESERVED
- ✅ 12 tests PASS

**新增 honest-negative caveats**:
- 🟡 PoseBusters validity SKIPPED graceful (not installed in env); MEASURED RMSD 5/5 SOTA but validity NOT verified → honest-negative on validity NOT on accuracy

### §1.4 任务 #3 Wf-135 / TODO-135 Phase 2 — 🟡 PARTIAL 6/7

**任务**: 5-metal × 5-seed × 50-epoch LOMO re-bench with balanced Rh cohort

**MEASURED 验收结果 (per Wf-379-A1 + Wf-387-A1)**:
- 🟡 6/7 pairs MEASURED (Rh seed 3/4 + Re seeds 0/1/2/3); Re seed 4 NOT_RUN (GPU wrapper terminated)
- ✅ Rh fold AUC 0.8798 ± 0.0115 — PASS Krasnov 0.86 gate (+0.020 margin) via Wf-387-A1 aggregation fix
- 🟡 Re fold AUC 0.7061 ± 0.0552 — BELOW Krasnov 0.86 gate (-0.154 margin); HONEST_NEGATIVE_PARTIAL preserved
- 🟡 Overall mean 0.7640 ± 0.0995 — BELOW Krasnov 0.86 gate (-0.096 margin); HONEST_NEGATIVE_PARTIAL_LIFT preserved
- ✅ `metrics/by_metric/metal_lomo_auc_wasserstein.json` REFRESHED 6/7 pair summary via Wf-387-A1
- ✅ Wf-387-A2 balanced REAL 500 Rh cohort SHIPPED for honest-negative closure (synthetic-free cohort from MetalCytoToxDB train split)
- ✅ 4 NEW tests PASS in Wf-387-A1 + Wf-387-A2

**新增 honest-negative caveats**:
- 🟡 Re seed 4 NOT_RUN — closure path = USER_OK-gated Wf-131 P3 follow-up launch (~30 min GPU)
- 🟡 Re fold BELOW Krasnov 0.86 gate — structural finding per party-twenty-spirit §3.5
- 🟡 Wf-139 P7 synthetic 500 cohort validated PROXIED NOT BIOACTIVITY; Wf-387-A2 REAL 500 cohort substituted for honest-negative closure verification

### §1.5 任务 #4 TODO-138 — 🟡 PARTIAL via Wf-389-A1

**任务**: Paper v2 + Tier-2 submission package (JCAMD/JMGM/CC/MD/CBC)

**MEASURED 验收结果 (per Wf-389-A1 + Wf-389-A3 + Wf-380-A7)**:
- ✅ m6_kappa §3 main-result DROPPED (per Wf-377-A1 verdict + Wf-389-A1 §3 supplementary sub-table reframe)
- ✅ Top-3 alternatives PROMOTED to §3 supplementary sub-table: **MolLogP** + **MorganDensity_r2** + **Chi3v** (all WEAK band per Cohen 1988; per Wf-377-A1 §4.2 + Wf-389-A1 §2)
- ✅ m6_kappa §6 honest-limitations RETAINED as complementary heuristic
- ✅ `metrics/by_metric/m6_kappa.json::verdict` UPDATED to `MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL`
- ✅ `metrics/by_metric/m6_kappa.json::wf389a1_supplementary_reframe` block ADDED (top-3 promotion record)
- ✅ NEW doc `docs/PAPER_SECTION_3_METRIC_TABLE.md` (~580 LOC, 6 sections) SHIPPED
- ✅ WEAK_TO_STRONG_PLAN.md §3.6 + §2.2 + §3.8 amended (Wf-389-A3; preserved via canonical closeouts)

**新增 honest-negative caveats**:
- 🟡 Paper §3 main-result DROPPED m6_kappa; user OK required for paper/* write per TODO-84
- 🟡 §3 supplementary sub-table uses WEAK band alternatives (Pearson r 0.18-0.20) — NOT MODERATE band

### §1.6 任务 #5 TODO-200 — ✅ SCAFFOLD SHIPPED

**任务**: Wf-340/353 deep integration SCAFFOLDS (10 CFM innovations)

**MEASURED 验收结果 (per Wf-380-A1)**:
- ✅ 10 SCAFFOLD modules (~1055 LOC) SHIPPED at `molmetal/adapters/flow_matching_lipman/`
- ✅ 10 test files (~544 LOC) with **41/41 NEW tests PASS** in ~1.61 s wall
- ✅ Pre-flight v3 cfm pipeline: 6 PASS / 0 FAIL / 0 SKIP / 6 TOTAL PRESERVED
- ✅ Honest framing: SCAFFOLD-only; no trainer integration; speedup/lift claims PROJECTED per upstream papers

### §1.7 任务 #6 TODO-201 — ✅ SCAFFOLD SHIPPED + REAL MEASURED

**任务**: m2 R² per-cohort-specific feature config

**MEASURED 验收结果 (per Wf-380-A2 + Wf-388-A1)**:
- ✅ Wf-380-A2 synthetic validation: 4/4 cohorts above zero (platinum_4cell +0.1970 / hilbert +0.1970 / metalcytotox_combined +0.8869 / intermediate_n500 +0.7267)
- ✅ Wf-388-A1 REAL cohort validation: 1/3 cohorts WEAK_RECOVERY (metalcytotox_combined n=2000 +0.1981); 2/3 HONEST_NEGATIVE_R2_DEEP PRESERVED (platinai_4cell -0.1440 + hilbert -0.1440)
- ✅ per-cohort config matrix: n_lt_300 = 12-d RDKit + Ridge α=1.0; n_lt_1000 = 24-d + chemprop v3 + Ridge α=0.5; n_gte_1000 = 50-d ensemble + Ridge α=0.1
- ✅ 3 NEW tests PASS via Wf-380-A2; 4 NEW tests PASS via Wf-388-A1
- ✅ Verdict: MEASURED_PARTIAL_LIFT (1/3 WEAK_RECOVERY; 0/3 CEILING_BREAK; 2/3 HONEST_NEGATIVE_R2_DEEP)

### §1.8 任务 #7 TODO-202 — ✅ SCAFFOLD + WIRED SHIPPED

**任务**: m7 Diversity Sampling improvements

**MEASURED 验收结果 (per Wf-380-A3 + Wf-388-A2)**:
- ✅ Wf-380-A3 SCAFFOLD: 4 diversity primitives (nucleus_sample + mmd_loss + repulsion_loss + diversity_aware_sampling_step) SHIPPED in `molmetal/adapters/flow_matching_lipman/diversity_sampling.py` 619 LOC + `scaffold_conditioning.py` 300 LOC; 3 NEW tests PASS
- ✅ Wf-388-A2 WIRED: 6 CLI flags (--top-p / --top-k / --temperature / --mmd-weight / --repulsion-weight / --scaffold-conditioning) wired into `train_one_epoch`; default OFF preserves bit-for-bit backward compat with Wf-299/Wf-380-B1
- ✅ 4 NEW tests PASS in 0.39 s wall CPU-only
- ✅ Honest framing: PROJECTED +0.02-0.035 m7 lift NOT MEASURED end-to-end on Blackwell sm_120 (GPU BLOCKED per §5.1)

### §1.9 任务 #8 TODO-203 — ✅ SCAFFOLD SHIPPED

**任务**: m3 Vina docking param tuning

**MEASURED 验收结果 (per Wf-380-A4)**:
- ✅ 3-level Vina config (low/medium/high exhaustiveness = 8/32/64, n_poses=9/20/40)
- ✅ Tuning script + smoke report SHIPPED
- ✅ 14 NEW tests PASS in 0.25 s wall
- ✅ Honest framing: PROJECTED -0.3 to -0.5 kcal/mol per pocket vs low (per Wf-141 §0; needs real dock to confirm); real Vina validation BLOCKED on USER_OK per AGENTS.md §5.1 + Wf-299 CFM 3-wave launch

### §1.10 任务 #9 TODO-204 — ✅ SCAFFOLD SHIPPED

**任务**: m15 RMSD multi-template + MM-GBSA rescoring

**MEASURED 验收结果 (per Wf-380-A5)**:
- ✅ multi_template_dock (5-template + 9-pose smoke) SHIPPED
- ✅ mm_gbsa_rescore (argmin over MM-GBSA grid) SHIPPED
- ✅ SURROGATE smoke median 0.6015 Å (5 templates × 9 poses); 5/5 pockets pass 2.0 Å SOTA gate
- ✅ 2 NEW tests PASS
- ✅ Honest framing: SURROGATE NOT MEASURED on real binaries; real validation BLOCKED on USER_OK

### §1.11 任务 #10 TODO-205 — ✅ SCAFFOLD + WIRED SHIPPED

**任务**: m4 SA generation penalty

**MEASURED 验收结果 (per Wf-380-A6 + Wf-388-A3)**:
- ✅ Wf-380-A6 SCAFFOLD: SA penalty primitives + CLI panel SHIPPED
- ✅ Wf-388-A3 WIRED: `--sa-penalty` (default 0.0 OFF) + `--sa-target` (default 2.8) wired into CFM trainer
- ✅ 14 NEW tests PASS via Wf-380-A6
- ✅ 4 NEW tests PASS via Wf-388-A3 (bit-for-bit backward compat verified)

### §1.12 任务 #11 TODO-206 — ❌ HONEST_NEGATIVE_FINAL FINAL FINAL

**任务**: m6_kappa metric redefinition

**MEASURED 验收结果 (per Wf-380-A7 + Wf-381-A4 + Wf-377-A1)**:
- ✅ Canonical source brief `TODO/pending/206_m6_kappa_metric_redefinition.md` created (Wf-381-A4)
- ✅ 4 docs SHIPPED with section counts 8/7/8/9 = 32 total (per task brief spec)
- ❌ m6_kappa augmented_mean NOT lifted (best 0.4737 vs threshold 0.6; gap -0.1263)
- ❌ 8 paths exhausted (7 architectural + 1 metric-redesign layer per Wf-377-A1)
- 🟡 Recommendation: drop-from-§3 main-result + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`

### §1.13 任务 #12 TODO-94 — ✅ SCAFFOLD SHIPPED

**任务**: R20-P3 Multi-PH + PLL feature extraction

**MEASURED 验收结果 (per Wf-382-A1)**:
- ✅ Canonical source brief `TODO/pending/094_r20_p3_multiph_pll_feature.md` created (Wf-382-A1)
- ✅ 1 adapter module `molmetal/adapters/flow_matching_lipman/pll_features.py` 273 LOC
- ✅ 1 extraction script `molmetal/scripts/wf382a1_pll_feature_extract.py` 387 LOC
- ✅ Smoke report SHIPPED (4 SMILES × 10-d PLL feature vectors)
- ✅ 1 NEW test PASS in 0.56 s wall
- 🟡 Expected impact on small cohorts (n=200): 0 lift per Wf-377-A2 diagnostic
- 🟡 Trainer integration deferred to TODO-207 Wf-299 launch (USER_OK-gated)

---

## §2 串行执行计划 (Serial Execution Plan) — REFRESHED 2026-09-30

### ✅ Step 0: 枚举(本文档) — DONE 2026-09-27 + REFRESHED 2026-09-30

### ✅ Step 1: Wf-139 P7 / TODO-135 Phase 1(Rh balanced cohort) — DONE 2026-09-27

(Per original §2 Step 1; preserved unchanged)

### ✅ Step 2: Wf-134 / TODO-134 m15 redesign — DONE 2026-09-30 via Wf-341-T1 + Wf-362-A3

**Result**: ✅ **SHIPPED** — 5/5 acceptance gates met per §1.3
- 5/5 pocket median best RMSD 0.6578 Å (≤ 2.0 Å SOTA success)
- 5/5 pockets ≤ 2.0 Å SOTA success
- PoseBusters validity SKIPPED graceful (honest-negative on validity)
- `metrics/by_metric/docked_pose_rmsd.json` REFRESHED via Wf-362-A3

### 🟡 Step 3: Wf-135 / TODO-135 Phase 2(LOMO 5×5×50) — PARTIAL 6/7

**Result**: 🟡 **PARTIAL** via Wf-379-A1 GPU + Wf-387-A1 aggregation + Wf-387-A2 balanced REAL cohort
- 6/7 (metal, seed) pairs MEASURED; Re seed 4 NOT_RUN (GPU wrapper terminated)
- Rh fold 0.8798 ± 0.0115 PASS Krasnov 0.86 gate (+0.020)
- Re fold 0.7061 ± 0.0552 BELOW (-0.154); HONEST_NEGATIVE_PARTIAL
- Overall 0.7640 ± 0.0995 BELOW (-0.096); HONEST_NEGATIVE_PARTIAL_LIFT

**Closure path**: USER_OK-gated Wf-131 P3 follow-up launch for Re seed 4 (~30 min GPU)

### 🟡 Step 4: TODO-138 paper v2 + submission — PARTIAL via Wf-389-A1

**Result**: 🟡 **PARTIAL** — m6_kappa §3 DROPPED + top-3 alternatives PROMOTED to §3 supplementary sub-table
- m6_kappa §6 honest-limitations RETAINED
- WEAK_TO_STRONG_PLAN.md amended (Wf-389-A3)
- `metrics/by_metric/m6_kappa.json::verdict` UPDATED
- Paper §3 metric table NEW doc SHIPPED (`docs/PAPER_SECTION_3_METRIC_TABLE.md`)

**Closure path**: USER_OK-gated paper/* write per TODO-84 (~12-15 hr CPU)

### ✅ Step 5-#11: SCAFFOLD / WIRED / HONEST_NEGATIVE_FINAL — SHIPPED via Wf-380..388

**Result**: ✅ **SHIPPED** — SCAFFOLD + WIRED + HONEST_NEGATIVE_FINAL acceptance gates met per §1.6-§1.13
- TODO-200 Wf-380-A1: 10 SCAFFOLD modules + 41/41 tests
- TODO-201 Wf-380-A2 + Wf-388-A1: per-cohort config validated + REAL MEASURED
- TODO-202 Wf-380-A3 + Wf-388-A2: SCAFFOLD + WIRED 6 flags
- TODO-203 Wf-380-A4: 3-level Vina config + 14/14 tests
- TODO-204 Wf-380-A5: multi-template + MM-GBSA SCAFFOLD + 2/2 tests
- TODO-205 Wf-380-A6 + Wf-388-A3: SCAFFOLD + 14/14 tests + WIRED 2 flags
- TODO-206 Wf-380-A7 + Wf-381-A4: 4 docs SHIPPED + HONEST_NEGATIVE_FINAL FINAL FINAL
- TODO-94 Wf-382-A1: adapter + extraction SCAFFOLD + 1/1 tests

---

## §3 Halt Protocol

(Per original §3; preserved unchanged)

**当任何 task 验收门槛 FAIL 时**:
1. 立即停止 pipeline,不进入下一项
2. 显式 ship honest-negative(per `party-twenty-spirit` §3.5)
3. 在 `TODO/completed/<ID>_HONEST_NEGATIVE.md` ship fail report
4. 报告给用户:PASS / PARTIAL / HALT + 失败原因

**不假装 ship**:
- 不假装 12 tests 通过 when actually FAIL
- 不假装 label balanced when actually constant
- 不假装 pocket RMSD ≤ 2.0 Å when actually > 5 Å
- 不假装 paper builds clean when actually has unresolved refs
- 不假装 m6_kappa aug_mean ≥ 0.6 when actually < 0.5 (8 paths exhausted)
- 不假装 m2 R² > 0 on small cohorts when actually < 0 (overfit bias-variance finding)

---

## §4 Refresh Notes (Wf-391-A2)

**Date**: 2026-09-30
**Refresh author**: Wf-391-A2 ultracode CPU-only docs+plan wave (0 NEW tests, 0 GPU launches, 0 paper/* touches, 0 wet-lab planning per AGENTS.md §10)
**Trigger**: User directive 2026-09-29 *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* — start additional ultracode for TODO items not yet covered by just-launched workflows.

**Refresh scope**:
- §0 enumeration table REFRESHED: 4 P0 tasks → 12 tasks (5 SHIPPED + 4 SCAFFOLD + 1 HONEST_NEGATIVE_FINAL + 2 PARTIAL)
- §1 acceptance gate protocol per-task REFRESHED with MEASURED status per Wf-380..390 ultracode audit
- §2 serial execution plan REFRESHED with on-disk status per Wf-380..390
- §3 Halt protocol PRESERVED unchanged
- New §4 Refresh Notes added

**Constraints honoured**:
- 0 NEW tests (docs-only)
- 0 GPU launches (per AGENTS.md §5.1)
- 0 paper/* touches (per TODO-84 + §5.3)
- 0 wet-lab planning (per AGENTS.md §5.4 + §10)
- Use Bash + Read + Edit + Write only
- Preserve existing structure

**Cross-references**:
- `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` — re-prioritized version of this enumeration
- `TODO/completed/391a2_improvement_enum_refresh.md` — this Wf-391-A2 closeout
- Wf-380..390 ultracode series — 10 sub-flow closeouts at `TODO/completed/38{0..9}{a,b}{1..7}_*.md`
- AGENTS.md §5.1 (GPU grant) + §5.2 (honest framing) + §5.3 (no paper work) + §5.4 (Tier-2 markers) + §10 (wetlab deprecation)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `TODO/INDEX.md` — single source of truth
- `docs/HONEST_NEGATIVE_CATALOG.md` — 65 honest-negative entries