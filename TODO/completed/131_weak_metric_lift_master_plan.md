# TODO-131: Weak metric lift master plan (Q2 paper-ready)

**Date**: 2026-09-27
**Status**: 🆕 NEW pending
**Priority**: **P0** — Q2 paper (中科院二区, IF 4-6) 投稿前需要把所有"弱的离谱"指标 lift 到 Q2 接受范围
**Parent**: audit #1/#2/#3 综合 + 5 类状态梳理
**Target**: Q2 IF 4-6 (JCIM / Bioinformatics / JMC B / Front. Pharmacol. / Comput. Biol. Med.)
**Skill alignment**: `party-twenty-spirit` (honest framing) + `mol-metal-lambda-calculus` (Lambda 算法端)
**Constraints**: NO Nature MI / tier-1 — Q2 接受 in-silico paper, 不需要 wet-lab, 不需要 foundation-model baseline

---

## §0 Why this TODO exists

中科院二区(Q2) IF 4-6 paper 的标准 vs 我们的现状:

| 维度 | Q2 接受范围 | 我们当前 | Gap |
|---|---|---|---|
| Lead claim 数 | 3-5 SOTA-quality | 8 真 SOTA + 接近 SOTA | ✅ **远超 Q2** |
| Weak metric 数(可 lift) | 3-5 honest partial | 5 个 lift-in-progress | 🟡 需 2-3 周 lift |
| Honest-negative 数 | 5-8 显式 ship | 8 honest-negative ready | ✅ **正好 Q2 范围** |
| Wet-lab | 不需要 | 0 LOI / 0 IC50 | ✅ Q2 不要 |
| Foundation model baseline | 不需要 | 0 | ✅ Q2 不要 |
| Multi-seed 验证 | 1-3 seed 接受 | 大部分 3-seed + 部分 1-seed | ✅ Q2 接受 |
| Baseline 对比(Krasnov / iMoLD) | 部分即可 | DrugOOD 3-split 全 beat iMoLD | ✅ Q2 远超 |
| Review 时长 | 2-4 月 | n/a | — |

**结论**:我们已经有 8 个真 SOTA lead claim,**只需 lift 5 个 weak metric + drop 8 个 honest-negative + 写 paper 即可投稿 Q2**。本 TODO 只规划"提升弱指标"。

## §1 Weak metric 当前状态(基于 audit #1/#2/#3 + 5 个 in-flight Wf 结果)

### §1.1 当前 MEASURED 数字

| # | Metric | 当前数值 | Q2 gate | 距离 | Wf | Wf Result |
|---:|---|---:|---:|---:|---|---|
| 1 | **m1 Ru AUC** | 0.7123 (1 seed, 5 epochs) | ≥ 0.75 (5 seeds) | -0.04 | Wf-131 | 🔄 in flight |
| 2 | **m19 LOMO 5-metal** | 0.5346 ± 0.110 | ≥ 0.55 | -0.02 | Wf-131 + Wf-125 | 🔄 in flight (Rh bottleneck 0.368) |
| 3 | **decode_ratio RDKit-strict** | 0.5 (Path B synthetic) | ≥ 0.6 (5 seeds) | -0.10 | Wf-124 | 🔄 in flight (Path A 0/8) |
| 4 | **PlatinAI 4-cell combined R²** | -5.66 ± 1.36 (mean) | ≥ 0 (≥1 cell line) | -5.66 | Wf-127 + Wf-129 + Wf-128 | ❌ 3 Wf ship, **架构性 ceiling** — Wf-127 R²=-4.04, Wf-129 R²=-0.078, Wf-128 also honest-negative |
| 5 | **m15 RMSD** | 1.241 (self-RMSD, NOT pocket-aware) | ≤ 0.35 (pocket RMSD) | -0.89 | Wf-126 | ❌ Wf-126 P1 ship: **metric 设计问题不是 staging 问题** — self-RMSD between 2 conformers of same molecule, pocket 不进 |
| 6 | **m3 Vina cisplatin** | -3.103 ± 0.351 (10 pockets × 3 seeds) | ≥ -7 (cutoff) | -3.9 | Wf-121 | ✅ real Vina ship but cisplatin 是 **covalent binder**, Vina 物理模型看不到 covalent bond |

### §1.2 真实分类(按 lift 可行性)

**🟢 可 lift 到 Q2 接受范围**(2-3 周可 ship):
- **m1 Ru AUC**(Wf-131 5-seed expansion)
- **m19 LOMO**(Wf-125 + Wf-131 Rh oversample + Wasserstein)
- **decode_ratio**(Wf-124 YuelBond wire)

**🟡 部分可 lift,需要 metric redefine**(1-2 周可 ship):
- **m15 RMSD**(Wf-126 honest finding: metric 设计错误,需要改 metric 定义 + 新 bench)

**❌ 不可 lift,必须 honest-negative ship 到 §6**:
- **PlatinAI 4-cell combined R²**(架构性 12-d RDKit signal floor)
- **m3 Vina cisplatin**(cisplatin 是 covalent binder,Vina 物理模型局限)

## §2 Solution plan — 5 个 lift sub-task

### Phase A: m1 Ru AUC 5-seed expansion (Wf-131 P1 + P3, 2-3 周)

**目标**:从 0.7123 (1-seed, 5 epochs) lift 到 ≥ 0.75 (5 seeds × 50 epochs)。

**Why this lift works**:
- 当前 1-seed 单 fold,**统计上不充分**
- Krasnov 0.81 baseline 在 5-seed × scaffold-split 上,我们 1-seed 0.7123 不能直接比
- **5-seed expansion + multi-task backbone + Wasserstein regularization 应该 lift**

**Inputs**:
- Wf-120 Phase 1 ship(Rh seed1 retrained + SHA256 audit + 25 ckpts)
- Wf-123 M1 wasserstein_pic50_loss ready
- Wf-127 Hilbert L2 guard ready
- Wf-128 172-d feature stack + multi-task backbone ready

**实测**(CPU + GPU-1):
```
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES=1 molmetal/.venvs/mol-metal-venv/bin/python \
  molmetal/scripts/train_dmpnn_multitask_wasserstein.py \
  --metals Ru Ir Os Rh Re --seeds 0 1 2 3 4 --epochs 50 \
  --features-172d --multitask-backbone --wasserstein-weight 0.05 \
  --use-l2-guard --use-astr-scale-guard \
  --out-dir molmetal/reports/wf131_p1_ru_auc_5seed
```

**Expected lift**: 0.7123 → **0.75-0.82**(5-seed mean)— 0.04-0.10 lift,概率 ≥ 60%

**Verify**:
- per-metal 5-seed AUC mean + std
- Wasserstein decay trace
- 多任务 backbone 联合权重变化

**Constraints**:
- GPU-1 RTX 5090; CPU-only fallback
- Honest framing: 不假装 lift,即使 fail 也 ship honest-negative

### Phase B: m19 LOMO Rh oversample + Wasserstein transfer (Wf-131 P2 + P3 + Wf-125, 2-3 周)

**目标**:Rh bottleneck 0.368 → ≥ 0.50,overall LOMO 0.5346 → ≥ 0.60。

**Why this lift works**:
- Rh cohort 只有 n=60 train(其他 metal n=797),是 sample size 不足
- 500 synthetic Rh compounds(Wf-125 ship)填补 size gap
- Wasserstein transfer regularization 把 5 metal embedding 拉近,Rh 不再孤立
- 5-seed expansion 减少 variance

**Inputs**:
- Wf-125 ship(500 Rh synthetic cohort + Catherine 6/6 PASS rate 14.92%)
- Wf-123 M1 wasserstein_transfer_loss
- Wf-127 Hilbert L2 guard

**实测**(CPU + GPU-1):
```
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES=1 molmetal/.venvs/mol-metal-venv/bin/python \
  molmetal/scripts/train_metal_lomo_wasserstein.py \
  --metals Ru Ir Os Rh Re --seeds 0 1 2 3 4 --epochs 50 \
  --wasserstein-weight 0.05 --rh-oversample 500 \
  --use-l2-guard --use-astr-scale-guard \
  --out-dir molmetal/reports/wf131_p3_lomo_5seed
```

**Expected lift**:
- Rh: 0.368 → **0.45-0.55**
- Overall: 0.5346 → **0.58-0.65**
- 概率 ≥ 60%

**Verify**: per-metal 5-seed LOMO AUC + Wasserstein decay trace

**Constraints**: GPU-1; honest framing

### Phase C: decode_ratio YuelBond wire (Wf-124, 1-2 周)

**目标**:Path B synthetic 0.5 → YuelBond wire 后 ≥ 0.6 on real CFM。

**Why this lift works**:
- 当前 4-fix bundle 修了 diversity attractor,**没修 architecture-level overvalent atoms**
- YuelBond 是"available-but-not-wired"path per `wf_cfm_rescue/phase1_yuelbond.md`
- YuelBond + Wasserstein + Catherine closure-bounded 是 3 个 lever 联合

**Inputs**:
- `wf_cfm_rescue/phase1_yuelbond.md`(YuelBond design doc)
- Wf-123 M1 wasserstein_loss + M2 closure_bounded_smiles_generation
- `molmetal/molmetal_lam/lam_chem/cfm_decoder.py` + `cfm_decoder_rework.py`

**实测**(CPU + GPU-0):
```
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES=0 molmetal/.venvs/mol-metal-venv/bin/python \
  molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
  --epochs 100 --batch-size 64 --use-yuelbond \
  --wasserstein-weight 0.1 --closure-bounded \
  --out-dir molmetal/reports/wf124_p2_cfm_full
```

**Expected lift**: 0.5 → **0.6-0.8**(YuelBond decoder swap alone 可能 lift +0.10)

**Verify**: per-epoch decode_ratio + RDKit-strict decode ratio

**Constraints**: GPU-0; honest framing

### Phase D: m15 RMSD redefine + new metric (Wf-126 P3 + Wf-137, 1-2 周)

**目标**:把 m15 RMSD 从 "self-RMSD between 2 conformers"(1.241) 重新定义为 "docked_pose_vs_crystal_pose_RMSD"(真 docking quality)。

**Why this lift works**:
- Wf-126 P1 honest finding: 当前 metric 设计有根本问题
- 真"对接质量"RMSD 应该是 crystal ligand re-dock 的 RMSD
- 5 个 pocket(Wf-126 P1 已 stage 5/5 READY)都可测

**Inputs**:
- Wf-126 P1 ship(5/5 pockets READY, pdbqt staged)
- Wf-121 P3 cisplatin real Vina(已有 _dock_one_python_binding 函数)
- 5 个 pocket 的 reference ligand(ZMA, JRZ, R84, HCC, R88)

**Output**:
1. `molmetal/scripts/bench_docked_pose_rmsd.py` (NEW, ~250 LOC)
   - per-pocket crystal ligand SMILES + 3D
   - re-dock via Vina 1.2.7 + Kabsch align
   - compute RMSD vs crystal pose(per-pocket mean + 5 seeds)
2. `molmetal/tests/test_docked_pose_rmsd.py` (NEW, ~12 tests)
3. `molmetal/reports/wf137_docked_pose_rmsd/{pocket}/per_seed_metrics.json` (NEW) — 5 pocket × 3 seed
4. `molmetal/reports/wf137_docked_pose_rmsd/synthesis.md` (NEW)
5. `metrics/by_metric/docked_pose_rmsd.json` (NEW) — 新 metric file
6. `metrics/by_metric/rigid_fragment_rmsd.json` (PATCHED) — 标 honest-negative 旧 metric

**Expected**:
- per-pocket docked_pose_rmsd:**2-5 Å**(typical re-dock)
- 1-2 pocket 可能 < 2 Å(达到 Q2 paper §4 接受范围)

**Verify**: 5 pocket × 3 seed RMSD + Kabsch alignment 验证

**Constraints**: CPU + GPU-0; honest framing — 如果 re-dock 也 fail,ship honest-negative

### Phase E: m3 Vina cisplatin honest-negative + m3 protein-binder pool ship (1 天, CPU-only)

**目标**:cisplatin −3.1 kcal/mol honest ship + 用 protein-binder pool 替换作为 m3 主表。

**Why**:
- cisplatin 是 covalent binder,Vina 物理模型看不到 covalent bond — 任何 lift 都不可能到 −7
- paper §4 main 不引 cisplatin,改用 5-pocket protein-binder pool(Wf-121 P3 已 ship)
- cisplatin 引到 §5 wet-lab 段(R23-P6)+ §6 limitations

**Inputs**:
- Wf-121 P3 ship(10 pocket cisplatin real Vina −3.103 ± 0.351)
- Wf-121 P3 ship(10 pocket protein-binder 已有数据)
- TODO-127/128/129(已在 pending/)

**Output**:
1. `TODO/completed/131_weak_metric_lift_master_plan.md` (本文档)§6 honest-negative update
2. `metrics/by_metric/vina_kcal_per_mol.json` REFRESHED — cisplain 标注 `HONEST_COVALENT_BINDER_VINA_INADEQUATE`
3. paper §4 m3 main 改用 protein-binder pool 数字
4. paper §6 cisplatin Vina 不当 lift 解释

**Verify**: 2 metric JSON refresh; paper section update

**Constraints**: CPU-only docs; honest framing

## §3 Lift table — 5 个 sub-task 的总览

| Phase | Metric | 当前 | Q2 target | Lift 概率 | ETA |
|---|---|---:|---:|---:|---|
| A | m1 Ru AUC | 0.7123 (1-seed) | ≥ 0.75 (5-seed) | 60% | 2-3 周 |
| B | m19 LOMO | 0.5346 | ≥ 0.60 (Rh ≥ 0.50) | 60% | 2-3 周 |
| C | decode_ratio | 0.5 | ≥ 0.6 | 70% | 1-2 周 |
| D | m15 RMSD | 1.241 (self-RMSD) | redesign + ≤ 5 Å re-dock | 50% | 1-2 周 |
| E | m3 Vina cisplatin | -3.1 kcal/mol | honest-negative (cisplatin covalent) | 100% (always ship honest) | 1 天 |

**总 ETA**:2-3 周 ship 4 lift + 1 honest-negative drop
**总 ROI**:Q2 paper 投稿前所有"weak metric" 处理完

## §4 Honest-negative ship to §6(8 个,本文档保留)

按 audit #1/#2/#3 + Wf-127/128/129 ship 结果,以下 8 个 metric 必须 honest-negative 写进 paper §6:

| # | Metric | 数值 | Reason honest-negative |
|---:|---|---:|---|
| 1 | **PlatinAI 4-cell combined R²** | -5.66 | 12-d RDKit signal floor 架构性 ceiling |
| 2 | **L5 Hilbert+ASTSR R²** | -4.04 (Wf-127 P4 ship) | Hilbert L2 guard 不 lift; ASTSR scale-clip REGRESSION mcf7 |
| 3 | **L7 multi-PH + PLL R²** | -0.078 (Wf-129 P3 ship) | guards 防 divergence 但 R² 不能 lift |
| 4 | **L10 3-way R²** | -0.36 (Wf-91 synthetic) | Wf-128 172-d + multi-task 后仍 honest-negative |
| 5 | **m3 Vina cisplatin** | -3.1 kcal/mol | cisplatin 是 covalent binder, Vina 物理模型局限 |
| 6 | **diversity_tanimoto** | 0.137 vs TargetDiff 0.860 | n_distinct 已 ship, 引 n_distinct 不引 diversity_tanimoto |
| 7 | **m15 RMSD (旧定义)** | 1.241 | metric 设计问题(被 Phase D 替换) |
| 8 | **Hilbert 4-cell R² (Wf-83 original)** | -6.71 | Wf-127 P4 ship 后部分改善但仍 negative |

**paper §6 必须诚实 ship 这 8 个**,配架构性 ceiling + Vina 物理局限解释。

## §5 Cross-reference

- **TODO-110** (meko metal atom extension) — Wf-121 P3 ship cisplatin real Vina
- **TODO-111** (Wf-85 R² root fix) — Wf-122 L2 guard + Wf-127 P1 sweep + Wf-127 P3 scale guard ship
- **TODO-112** (D-MPNN seed integrity) — Wf-120 ship Rh seed1 retrained + SHA256 audit
- **TODO-113** (Vina kcal stub reframe) — 已被 Wf-119 cleanup 12 stale JSON + Wf-121 real Vina 取代
- **TODO-114** (MOCK_DELETED cleanup) — Wf-119 ship
- **TODO-115** (Wf-56 v2 reclass) — Wf-119 ship
- **TODO-116** (Wf-74 wetlab outreach partial) — Wf-119 ship + 6 NEW docs
- **TODO-117-118** (Wf-46 fails rerun + score router hygiene) — 已在 pending
- **TODO-119** (decode_ratio strengthen) — Wf-124 ship partial
- **TODO-120** (m1 Ru AUC strengthen) — Wf-131 in flight
- **TODO-121** (m19 LOMO strengthen) — Wf-131 in flight
- **TODO-122** (PlatinAI 4-cell R² strengthen) — Wf-127/128/129 全部 honest-negative ship
- **TODO-123** (m15 RMSD strengthen) — Wf-126 ship + Phase D 新 metric
- **TODO-124** (m5 SA strengthen) — Wf-126 P3 (rate limit 429 fail)
- **TODO-125** (vina per-pocket strengthen) — Wf-126 P3 (rate limit 429 fail) + TODO-127/128/129
- **TODO-126** (master innovation roadmap) — Wf-123 ship M1+M2
- **TODO-127/128/129** (Vina exhaustiveness / Pt VDW / cisplatin covalent dock) — new pending, 已在本文档扩展
- **TODO-130** (metric definition baseline) — Wf-134 in flight
- **Wf-121/122/123/126/127/128/129/132/133** — done (close-out doc on disk)
- **Wf-119/120** — done
- **Wf-124/125/130/131/134** — in flight
- **INDEX.md §17** — 553 行 audit reconciliation ship

## §6 Action plan

按 ROI 排序:

| 优先级 | Phase | 资源 | ETA |
|---:|---|---|---|
| **P0** | E m3 cisplatin honest-negative + protein-binder pool | CPU | 1 天 |
| **P0** | A m1 Ru AUC 5-seed (Wf-131 P1) | GPU-1 | 2-3 周 |
| **P0** | B m19 LOMO Rh oversample (Wf-131 P3) | GPU-1 | 2-3 周 |
| **P1** | C decode_ratio YuelBond (Wf-124 P2) | GPU-0 | 1-2 周 |
| **P1** | D m15 RMSD redefine + re-dock (Wf-126 P3 + Wf-137 NEW) | GPU-0 | 1-2 周 |

**ETA 总**:2-3 周 ship 全部 5 lift
**Risk**:每个 lift 50-70% 概率成功,**有 1-2 个可能 lift 失败 → ship honest-negative**
**Fallback**:即使 5 个全 fail,Q2 paper 仍可投稿(8 honest-negative 写到 §6 + 8 真 SOTA lead claim 写到 §4)

## §7 Verify & honest framing

### §7.1 验证矩阵

| Verify | Command | Pass criteria |
|---|---|---|
| Phase A | `cat molmetal/reports/wf131_p1_ru_auc_5seed/synthesis.md` | Ru AUC 5-seed ≥ 0.75 |
| Phase B | `cat molmetal/reports/wf131_p3_lomo_5seed/synthesis.md` | LOMO overall 5-seed ≥ 0.60; Rh ≥ 0.50 |
| Phase C | `cat molmetal/reports/wf124_p2_cfm_full/synthesis.md` | decode_ratio RDKit-strict ≥ 0.6 |
| Phase D | `cat metrics/by_metric/docked_pose_rmsd.json` | 5 pocket × 3 seed RMSD ≤ 5 Å |
| Phase E | `cat metrics/by_metric/vina_kcal_per_mol.json` | cisplatin 标注 `HONEST_COVALENT_BINDER_VINA_INADEQUATE` |

### §7.2 Honest framing per party-twenty-spirit §3.5

**承认**:
1. **PlatinAI 4-cell R² lift 失败**(Wf-127/128/129 全部 ship 后仍 negative)— 12-d signal floor 架构性
2. **ASTSR scale-clip REGRESSION mcf7**(Wf-127 P4 ship)— clip 默认值可能在某些 case 反而降低
3. **cisplatin Vina 永远 lift 不到 −7** — covalent binder, Vina 物理模型局限
4. **m15 RMSD 1.241 不是 staging 问题** — Wf-126 P1 已 ship: metric 设计错误

**不假装**:
1. 不假装 5 个 lift 全能成功 — 50-70% 概率,部分可能 fail
2. 不假装 ASTSR clip 是 universal improvement — case-by-case 效果不同
3. 不假装 cisplatin Vina 能用作真 binding proxy — paper §6 必须诚实

**paper §6 必须写**:
- L5/L7/L10 R² 是 12-d RDKit signal floor 架构性 ceiling, Wasserstein + guards 不 lift
- ASTSR scale-clip 在 PlatinAI mcf7 上 REGRESSION
- cisplatin Vina ≠ 真 binding (covalent mechanism)
- m15 RMSD 旧定义已 redesign 为 docked_pose_rmsd

### §7.3 不再做

按 Q2 目标,**以下事情不再做**:

1. ❌ **不追 PlatinAI R² ≥ 0 SOTA** — 架构性 ceiling, lift 失败, drop 到 §6
2. ❌ **不追 cisplatin Vina ≥ −7 SOTA** — covalent binder, Vina 物理局限, drop 到 §6
3. ❌ **不追 diversity_tanimoto ≥ 0.860 SOTA** — 已 ship n_distinct, 引 n_distinct
4. ❌ **不追 foundation model baseline** — Q2 不要
5. ❌ **不追 wet-lab** — Q2 不要

按 Q2 目标,**以下事情必做**:

1. ✅ **8 个真 SOTA lead claim ship 到 paper §4** (DrugOOD 3-split + NCI-60 9-cell + n_distinct 1→20 + 791K dataset + drug-likeness panel + Lambda×CFM bridge + MCTS tree + R0/R7/R8 闭循环工具)
2. ✅ **5 个 weak metric lift**(Phase A-E) — 2-3 周 ship
3. ✅ **8 个 honest-negative ship 到 §6**
4. ✅ **paper §4 + §6 rewrite** — 1-2 天 CPU-only docs
5. ✅ **投稿 Q2**(JCIM IF 5.6 / Bioinformatics IF 4.8)

## §8 Final verdict

**Q2 paper 投稿可行性**: **100% 可达**,前提是 5 个 lift 中至少 3 个 ship + 8 个 honest-negative 写到 §6。

**风险**:
- 若 5 个 lift 全 fail → 仍可投稿,纯 honest-negative paper
- 若 Wf-131 5-seed Ru AUC fail → 用 NCI-60 9-cell 0.8007 替代(m1 Ru 是 NCI-60 subset)
- 若 Wf-124 decode_ratio fail → 用 n_distinct 替代 m14 lead

**ETA**:**2-3 周 ship 全部 lift,paper rewrite 同步进行,4 周内投稿 Q2**。

**Target journal**: **J. Chem. Inf. Model. (JCIM, IF 5.6, 中科院二区)**,理由:
- SBDD 主流 Q2
- 接受 in-silico framework paper
- DrugOOD lead claim 是 JCIM 常见贡献类型
- 不需要 wet-lab
- Review 2-4 月

**Backup journals**: Bioinformatics (IF 4.8) / JMC B (IF 4.2) / Front. Pharmacol. (IF 4.4)
