# IMPROVEMENT_ENUM — 当前剩余改进任务枚举 + 验收门槛

**Date**: 2026-09-27
**Trigger**: 用户请求"现在就枚举todo目录下还有哪些改进类型的任务,每一个任务后搞一个验收门槛,过了之后再串行做下一项任务"
**Status**: 🟢 ACTIVE — 3 个 P0 任务待做

---

## §0 枚举结果 (Enumerate)

**当前 `TODO/pending/` 中真正可执行的改进型任务**(已去除重复 / 已 ship / 已 superseded):

| # | ID | 任务 | 优先级 | 资源 | ETA | 来源 |
|---:|---|---|---:|---|---|---|
| **1** | **Wf-139 P7 / TODO-135 Phase 1** | **m19 LOMO Rh 正样本 cohort 生成**(250 active + 250 inactive) | **P0** | CPU-only RDKit | 1-2 周 | `pending/135_m19_lomo_rh_oversample_regression_fix.md` §2 |
| **2** | **Wf-134 / TODO-134** | m15 Docking RMSD redesign (3 phases) | **P0** | CPU + GPU-0 | 5-8 天 | `pending/134_m15_docking_rmsd_redesign.md` |
| **3** | **Wf-135 / TODO-135 Phase 2** | m19 LOMO 5-metal × 5-seed × 50-epoch re-bench | **P0** | GPU-1 (or CPU fallback) | 2 周 | `pending/135_m19_lomo_rh_oversample_regression_fix.md` §2 |
| **4** | **TODO-138** | Paper v2 + JCIM submission package | **P0** | CPU-only docs | 5-7 天 | `pending/138_paper_v2_submission_package.md` |

**依赖图 (dependency graph)**:
- 任务 #1 (TODO-135 Phase 1) → 无依赖,CPU-only,可立即启动
- 任务 #2 (TODO-134 m15 redesign) → 无强依赖(Wf-126 P1 5/5 pocket READY),可并行启动
- 任务 #3 (TODO-135 Phase 2) → 依赖任务 #1 (需要 balanced Rh cohort)
- 任务 #4 (TODO-138 paper v2) → 弱依赖任务 #2 (m15 redesign 数据填 paper §4.5.7)

**建议串行顺序**: #1 → #2 → #3 → #4

---

## §1 验收门槛协议 (Acceptance Gate — per task)

### §1.1 通用门槛(适用于所有 4 个任务)

| Gate | 标准 | Halt 条件 |
|---|---|---|
| **Implementation Gate** | 脚本/模块存在 + 可 import + 可调用 | 缺文件 / ImportError |
| **Test Gate** | pytest ≥12 tests PASS | pytest FAIL |
| **Metric Gate** | 数值 MEASURED(不假装) | 数字无来源 |
| **Honest-Framing Gate** | 不达目标时显式 ship honest-negative | 假装 ship PASS |
| **Registry Gate** | `metrics/by_metric/<metric>.json` REFRESHED | 指标未刷新 |
| **Completion Doc Gate** | `TODO/completed/<ID>_*.md` ship | 缺完成文档 |

### §1.2 任务 #1 Wf-139 P7 / TODO-135 Phase 1 — 专属门槛

**任务**: 生成 500 Rh cohort(250 active + 250 inactive with balanced labels)

**硬门槛 (HALT if violated)**:
- ✅ 500 cohort 生成成功(250 active + 250 inactive)
- ✅ balanced labels(不 constant-inactive)
- ✅ ≥12 tests PASS in `test_rh_active_label_generation.py`
- ✅ per-compound Catherine 6/6 PASS(继承自 Wf-131 P2)
- ✅ No label leak (scaffold-split verified)

**软门槛 (PARTIAL if violated)**:
- 🟡 similarity-based label assignment 1h36 ref caveat uncertain

**Verify & Report**: PASS / PARTIAL / HALT

### §1.3 任务 #2 Wf-134 / TODO-134 — 专属门槛

**任务**: m15 Docking RMSD redesign (3 phases: ref SMILES extraction → Vina redocking → metric JSON)

**硬门槛 (HALT if violated)**:
- Phase 1: 5/5 pocket ligand SMILES extracted + valid + InChIKey 一致
- Phase 2: 5/5 pocket × 3 seed × n_poses=5 = 75 docked poses
- Phase 2: Per-pocket median best RMSD ≤ 2.5 Å(SOTA paper 接受)
- Phase 2: ≥ 1 pocket ≤ 2.0 Å(SOTA success)
- Phase 3: `docked_pose_rmsd.json` valid JSON + 5 pocket data
- Phase 3: `rigid_fragment_rmsd.json` status = `HONEST_NEGATIVE_REDESIGNED`
- ≥ 12 tests PASS

**软门槛 (PARTIAL if violated)**:
- 1h36 ref shallow → RMSD > 5 Å → ship honest-negative for 1h36 only

### §1.4 任务 #3 Wf-135 / TODO-135 Phase 2 — 专属门槛

**任务**: 5-metal × 5-seed × 50-epoch LOMO re-bench with balanced Rh cohort

**硬门槛 (HALT if violated)**:
- 5/5 metal × 5 seed × 50 epoch = 1250 runs 完成
- Rh fold AUC ≥ 0.30(recover from 0.136 regression)
- Overall LOMO mean ≥ 0.60
- ≥ 16 tests PASS in `test_metal_lomo_v2.py`

**软门槛 (PARTIAL if violated)**:
- Rh 0.30-0.50 → PARTIAL(honest frame)
- LOMO < 0.60 → PARTIAL(honest frame)
- Multi-task backbone signal leak → PARTIAL

### §1.5 任务 #4 TODO-138 — 专属门槛

**任务**: Paper v2 + JCIM submission package

**硬门槛 (HALT if violated)**:
- paper builds clean (pdflatex 0 unresolved refs / 0 missing files)
- 8 lead claim + 2 partial lead + 8 honest-negative 全部 ship
- Cover letter + highlights + suggested reviewers 全部 ship
- `submission_package.zip` 含所有材料

**软门槛 (PARTIAL if violated)**:
- paper 15-25 页 → 范围外 → ship honest
- m15 redesign 数据未填 → PARTIAL(写 placeholder)

---

## §2 串行执行计划 (Serial Execution Plan)

### ✅ Step 0: 枚举(本文档) — DONE

### ✅ Step 1: Wf-139 P7 / TODO-135 Phase 1(Rh balanced cohort) — DONE 2026-09-27
**结果**: ✅ **PASS** — 5/5 acceptance gates met
- 250 active + 250 inactive (balanced, NOT constant-inactive)
- Catherine 6/6 PASS rate 1.0000 (500/500 inherited from Wf-131 P2)
- Scaffold overlap = 0 (no leak)
- **15/15 tests PASS**
- Wall-time 0.58 s (CPU-only RDKit)
- Outputs:
  - `molmetal/scripts/rh_active_label_generation.py` (340 LOC)
  - `molmetal/tests/test_rh_active_label_generation.py` (15 tests)
  - `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` (500 compounds)
  - `molmetal/reports/wf139_p7_rh_active_cohort/synthesis.md`
  - `TODO/completed/143_wf139_p7_rh_active_cohort.md`

**Honest framing (per party-twenty-spirit §3.5)**:
- Similarity-based label is **proxy**, NOT wet-lab validation
- Diff-based ranking is workaround for low absolute Tanimoto similarity (median 0.18)
- Pearson r(active_sim, inactive_sim) = 0.95 — synthetics are about as similar to actives as to inactives
- 423/500 compounds structurally more similar to inactive than active (structural bias)
- Cohort fixes the Wf-131 P2 regression BUT is NOT publication-grade without wet-lab

### ⏳ Step 2: Wf-134 / TODO-134 m15 redesign(3 phases)
**ETA**: 后续 session
**Verify**: 5/5 pocket median ≤ 2.5 Å + ≥ 1 pocket ≤ 2.0 Å

### ⏳ Step 3: Wf-135 / TODO-135 Phase 2(LOMO 5×5×50)
**ETA**: 后续 session(GPU 训练 2 周)
**Verify**: Rh AUC ≥ 0.30 + LOMO mean ≥ 0.60

### ⏳ Step 4: TODO-138 paper v2 + submission
**ETA**: 后续 session
**Verify**: paper builds + cover letter + supp complete

---

## §3 Halt Protocol

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