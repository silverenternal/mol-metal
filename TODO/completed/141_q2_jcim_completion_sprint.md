# TODO-141: Q2 JCIM completion sprint (6 weeks)

**Date**: 2026-09-27
**Status**: 🆕 NEW pending
**Priority**: **P0** — User challenge 2026-09-27: "datasets complete? models trained? benchmark aligned?" — honest audit shows **all three are NOT**.
**Parent**: TODO-140 baseline alignment + TODO-138 paper submission + Wf-139 paper v2
**Target journal**: **JCIM IF 5.6** (Q2 中科院二区)
**Skill alignment**: `party-twenty-spirit` §3.5 + `mol-metal-mechanism-mapping` + `mol-metal-anticancer`

---

## §0 Why this TODO exists

User 三大核心质疑(2026-09-27):

> "我们的数据集是完整的吗,模型训练完全了吗,benchmark 规模对齐了吗?"

### Honest audit(per party-twenty-spirit §3.5):

| 问题 | 答案 | 证据 |
|---|:---:|---|
| 数据集完整吗? | ❌ **NO** | `drugood_ood_auc.json` 里 `data_source = "metacytotox_fallback"` — **DrugOOD 是凑的**,不是 Ji et al. ICLR 2023 真数据 |
| 模型训练完成了吗? | ❌ **NO** | LOMO 5/5 metal × 5 seed × 50 epoch = **125 trainings 设计**,**7/125 = 5.6% 完成**(Ru 5/5 + Ir 2/5) |
| Benchmark 规模对齐了吗? | ❌ **NO** | CrossDocked **5/100 pocket**(5%);DrugOOD fallback ≠ Ji et al. 2023;NCI-60 9-cell `seeds=N/A epochs=N/A` 不可 audit |

### 后果:

**paper §4 大部分 lead claim 站不住**:
- §4.1 "beats iMoLD on DrugOOD" — **假数据**,必须撤或重做
- §4.2 "NCI-60 9-cell 0.8007" — 来源不可 audit,需重 ship
- §4.X "5-metal LOMO multi-task" — **Os/Rh/Re 完全没训完**

**这个状态发 Q2 JCIM 是危险的**:reviewer 一问 baseline protocol 直接穿。

### Sprint 目标:

**6 周内 ship 真正的 Q2 paper** — dataset 完整 + model 训练完整 + benchmark 规模对齐 + honest-negative 全 ship。

## §1 Conceptual alignment(从第一档 140 重申)

### §1.1 三种范式

| 范式 | 代表 Tier-1 | 我们对应层 |
|---|---|---|
| **Pure SBDD** | TargetDiff (ICLR 2023), Pocket2Mol (ICML 2022), DiffSBDD (ICML 2023) | Layer 1(5 pockets Vina) |
| **De novo anticancer** | **iMoLD (NeurIPS 2022)**, DeepCE, DrugCell | Layer 2(DrugOOD OOD + NCI-60) |
| **Metallodrug-specific** | **No Tier-1 baseline exists** | Layer 3(5 metals × MetalCytoToxDB) |
| **HYBRID** | **Mol-Metal (本文)** | 三层 stack |

**结论**:Mol-Metal 是 **HYBRID**,没有 single Tier-1 baseline 能 cover 三层。**orthogonal baseline 策略**,不是 single head-to-head。

### §1.2 我们的 baseline 引用矩阵

| Baseline | Tier-1? | 对齐方式 | Status |
|---|---|---|---|
| **iMoLD (NeurIPS 2022)** | ✅ | **真 head-to-head** DrugOOD OOD 3-split | 🆕 Phase 1 重建(真 DrugOOD) |
| TargetDiff (ICLR 2023) | ✅ | **Cite-only**(Vina/SA/QED published numbers),**不重跑**(cross-domain weak) | ✅ Cite in §2 |
| DeepCE / DrugCell | Q1 | **Cite-only** NCI-60 reference | ✅ Cite in §2 |
| Krasnov 2026 | Q2 | **Cite + numbers**(NCI-60 9-cell 0.81) | ✅ Cite in §4 |
| Metallodrug Tier-1 | ❌ | **No baseline exists** — paper §6 honest ship | 🆕 §6.10 |

## §2 当前状态 audit(per user challenge 2026-09-27)

### §2.1 Dataset 状态

| Dataset | 状态 | 详细 |
|---|---|---|
| MetalCytoToxDB | ✅ On disk | `molmetal/data/metalcytotox_5metal_multitask.csv`(3.8 MB,5207 compounds × 5 metals) |
| NCI-60 9-cell subset | ⚠️ On disk | `molmetal/data/nci60_precious_9cell.csv` — MetalCytoToxDB 跟 NCI-60 panel overlap 的 9 cell,**不是 NCI-60 GI50 raw data** |
| **DrugOOD (真)** | ❌ **NOT ON DISK** | Ji et al. 2023 ICLR data 没下载,`drugood_ood_auc.json` 用 metacytotox_fallback 凑 |
| CrossDocked2020 | ✅ On disk | `/home/hugo/codes/pocket/data/crossdocked2020/`(183K PDB,`split_by_name.pt` ready) |
| CrossDocked 5-pocket subset | ✅ | 1h36/3eml/3ny8/4aaw/4rlu |
| PlatinAI 4-cell | ⚠️ Synthetic | `models/coupling_adapter_platinai5k.json` — synthetic labels |
| BindingDB 1M | ✅ On disk | `/home/hugo/codes/pocket/data/tdc/bindingdb_ic50.csv`(1,072,932 rows) — 备用 wet-lab 数据 |

**关键缺失**:**真 DrugOOD**。 这是 Phase 1 的核心 fix。

### §2.2 Model training 状态

| Metal | Seeds trained | Epochs | ckpt on disk | Status |
|---|:---:|:---:|:---:|---|
| Ru | 5/5 | 30 → 待 extend 50 | ❌ | ✅ Wf-131 P3,需 extend |
| Ir | 2/5 | 30 → 待 extend 50 | ❌ | ⚠️ 需 complete 5/5 |
| **Os** | **0/5** | 0 | ❌ | ❌ **完全没训** |
| **Rh** | **0/5** | 0 | ❌ | ❌ **Wf-131 regression + Wf-145 fix fail** |
| **Re** | **0/5** | 0 | ❌ | ❌ **完全没训** |
| **Total** | **7/125 = 5.6%** | | | ❌ 设计目标 125 trainings |

ckpt 目录只有:
```
.ckpt/affinity_platinai_seed0.pt    (4 MB)
.ckpt/affinity_platinai_seed1.pt    (60 KB — 异常小)
```

**关键缺失**:**Os/Rh/Re 3 个 metal 一个都没训完**。这是 Phase 2 的核心 fix。

### §2.3 NCI-60 9-cell 状态

| 项 | 现状 | 需要 |
|---|---|---|
| `seeds` 字段 | **N/A** | 显式 ship `[0, 1, 2]` |
| `epochs` 字段 | **N/A** | 显式 ship `50` |
| per-cell AUC | 已 ship(9 cell) | re-verify 来源 |
| ckpt | ❌ 无 | 需重训 |

**关键缺失**:**NCI-60 9-cell 训练 seeds 和 epochs 不明**。这是 Phase 3 的核心 fix。

### §2.4 Benchmark 规模

| Benchmark | 我们 | Tier-1 baseline | 对齐? |
|---|---|---|---|
| DrugOOD OOD 3-split | metacytotox_fallback | DrugOOD 50K+ compounds | ❌ **假数据** |
| CrossDocked 100-pocket | 5/100 pockets(5%) | 100 pockets | ⚠️ subset bias |
| NCI-60 9-cell | 9 cell × 5 metal × ? seeds | Krasnov 2026 全 panel | ⚠️ seeds 不可 audit |
| LOMO 5×5×50 | 7/125 trainings | 125 trainings | ❌ 5.6% |

## §3 Sprint plan — 6 weeks, 6 phases

### Phase 1(Week 1-2):真 DrugOOD 数据集 + iMoLD 真 head-to-head

**任务**:

1. **Clone DrugOOD 官方 repo**(Ji et al. ICLR 2023)
   - URL: `https://github.com/tencent-ailab/DrugOOD`
   - 找 iMoLD paper 用的具体 subset(iMoLD assay 0.7211 / scaffold 0.6884 / size 0.6792)
   - 三个 subsets:`DrugOOD_Assay_split`(large)/ `DrugOOD_Scaffold_split`(medium)/ `DrugOOD_Size_split`(small)

3. **Download DrugOOD data**
   - ~50GB raw + processed
   - 落到 `/home/hugo/codes/mol-metal/.data/molmetal/drugood_native/`
   - SHA256 验证

5. **Verify iMoLD reference numbers**
   - iMoLD paper 报 assay 0.7211 / scaffold 0.6884 / size 0.6792
   - 用 iMoLD pretrained model 在 DrugOOD 重测 → 验证 ±0.01 一致

7. **Re-train Mol-Metal 在真 DrugOOD**
   - 用 v4_morgan_mlp + 3 seeds × 30 epoch
   - 3 OOD splits(assay/scaffold/size)
   - 输出新 `metrics/by_metric/drugood_ood_auc.json`:
     ```json
     "data_source": "drugood_native",  // 不是 fallback
     "imold_reference": { ... },
     "results": { ... }
     ```

**Inputs**:
- DrugOOD repo(Ji et al. 2023,publicly available)
- 现有 v4_morgan_mlp training code:`molmetal/scripts/train_drugood.py`
- 现有 v4_morgan_mlp 数据流(`_load_drugood_native()` 路径,需确认存在或写)

**Output**:
- `.data/molmetal/drugood_native/`(NEW, ~50GB)
- `molmetal/scripts/verify_drugood_imold_split.py`(NEW, ~150 LOC)
- `molmetal/tests/test_drugood_imold_split.py`(NEW, ~8 tests)
- `metrics/by_metric/drugood_ood_auc.json` REFRESHED,`data_source = "drugood_native"`
- `molmetal/reports/wf_phase1_drugood_native/synthesis.md`(NEW)

**Verify**:
- `data_source = "drugood_native"` ✅
- iMoLD numbers 在 ±0.01 重现 ✅
- 我们 Mol-Metal 3-seed OOD AUC 重现(可能比 fallback 高/低/一样,都要 honest ship)
- Scaffold overlap = 0(没 leakage)

**资源**:CPU-only(RDKit + Morgan FP + MLP 训练,不需要 GPU)
**ETA**:1-2 周(下载 + 训练)
**Risk**:DrugOOD 网络不可达(China firewall 可能)— fallback 用 TDC DrugOOD 镜像

### Phase 2(Week 2-4):LOMO 5/5 metal × 5/5 seed × 50 epoch 完整

**任务**:

| Metal | 当前 | 目标 | 训练量 |
|---|---|---|---|
| Ru | 5/5 seeds × 30 epoch | extend to 5/5 × 50 epoch | +5 训练 |
| Ir | 2/5 seeds × 30 epoch | complete to 5/5 × 50 epoch | +3 训练 |
| **Os** | **0/5** | 5/5 × 50 epoch | **5 新训练** |
| **Rh** | **0/5** | 5/5 × 50 epoch(用 Phase 1 真 DrugOOD Rh 子集 + honest Wf-145 doc) | **5 新训练** |
| **Re** | **0/5** | 5/5 × 50 epoch | **5 新训练** |
| Total | 7/125 | **125/125** | +23 trainings |

每个 metal 训练完成后:
- 5 ckpt on disk(`affinity_{metal}_seed{0..4}_e50.pt`)
- `molmetal/reports/wf_phase2_lomo_v2/{metal}/per_seed_metrics.json`
- 5-seed mean ± std 写回 `metrics/by_metric/metal_lomo_auc.json`

**Inputs**:
- Wf-131 P3 现有 Ru 5-seed × 30 epoch ckpt(extend to 50 epoch)
- Wf-131 P3 现有 Ir 2-seed × 30 epoch ckpt
- Wf-123 M1 wasserstein_transfer_loss + M2 closure_oversample
- Wf-127 Hilbert L2 guard
- `train_metal_lomo_v2.py`(Wf-145 P2 driver,860 LOC,18/18 tests pass)

**Output**:
- 23 new ckpt on disk
- 5 metal × 5 seed × 50 epoch = **125 trainings** complete
- `metrics/by_metric/metal_lomo_auc.json` REFRESHED 完整
- `molmetal/reports/wf_phase2_lomo_v2/{metal}/synthesis.md`
- 5 metal × 5-seed mean AUC ship:
  - Ru: extend from 0.594 ± 0.015(可能 ±0.01)
  - Ir: 2-seed → 5-seed complete
  - **Os/Rh/Re: 新测,honest ship results**(可能 fail,paper §6)

**Verify**:
- 125 ckpt on disk ✅
- 5 metal × 5-seed mean + std + per-seed ship ✅
- Wasserstein transfer target = Rh real ACTIVE(Phase 1 真 DrugOOD Rh 子集)
- Rh result 可能 fail per TODO-135 §5 honest framing(50% chance Rh < 0.50)

**资源**:GPU-0 RTX PRO 6000 Blackwell 96GB(完全空闲,~10 min × 23 = ~4 hours pure compute,加上 setup/debug = 1-2 weeks)
**ETA**:1-2 周
**Risk**:Rh n=60 sample size ceiling(TODO-135 §5)— Rh AUC 可能 < 0.50,§6 honest ship

### Phase 3(Week 4-5):NCI-60 9-cell seed × 50 epoch audit + extend

**任务**:

1. **Audit 现有 NCI-60 9-cell 训练**
   - `molmetal/reports/wf_nci60_precious_9cell/` 找 per-seed output
   - 确认实际 seeds 数(可能是 1-seed, 也可能是 3-seed, 也可能 ship 数字与训练脱节)
   - 确认实际 epochs 数

3. **如果只 1-seed**:
   - extend 到 3-seed × 50 epoch
   - 9 cell × 5 metal × 3 seeds × 50 epoch = 6750 training iterations
   - 输出新 ckpt + per-seed metrics

5. **如果已经是 3-seed × 50 epoch**:
   - 验证 `seeds: [0, 1, 2]` 显式 ship
   - 验证 `epochs: 50` 显式 ship
   - 确认 per_cell AUC 与现有数字 ±0.01

7. **重新 ship `metrics/by_metric/nci60_precious_9cell_auc.json`**:
   ```json
   "seeds": [0, 1, 2],   // 显式
   "epochs": 50,           // 显式
   "value_overall_mean_auc": 0.8007...,
   "per_cell_line": { ... }  // 9 cell × per_seed
   ```

**Inputs**:
- Wf-36 / Wf-40 现有 NCI-60 9-cell reports(`molmetal/reports/wf_nci60_precious_9cell/`)
- `molmetal/scripts/train_nci60_10cell.py`

**Output**:
- `metrics/by_metric/nci60_precious_9cell_auc.json` REFRESHED with explicit `seeds` + `epochs`
- 9 cell × per-seed × per-metal ckpt(如果 extend)
- `molmetal/reports/wf_phase3_nci60_verify/synthesis.md`

**Verify**:
- `seeds` 和 `epochs` 显式 ship ✅
- 9 cell per_cell AUC 验证 ±0.01 ✅
- Krasnov 2026 baseline 0.81 对比 honest ✅

**资源**:GPU-0 + GPU-1(PID 649993 完成后释放,~5-10 hours pure compute)
**ETA**:1 周
**Risk**:`seeds=N/A` 可能意味着 ship 数字与训练脱节 → Phase 3 主要任务是 audit + honesty

### Phase 4(Week 5):paper §4 lead narrative 重写

**任务**:

1. **§4.1 iMoLD head-to-head**:用 Phase 1 真 DrugOOD 结果 ship
2. **§4.2 NCI-60 9-cell**:用 Phase 3 audit 结果 ship
3. **§4.3 m19 LOMO**:用 Phase 2 真 5/5 metal × 5/5 seed ship
4. **§4.4 n_distinct 1 → 20**:已 ship,保留
5. **§4.5 metallodrug 791K × 16 metals**:已 ship,保留
6. **§4.6 Drug-likeness panel**:已 ship,保留
7. **§4.7 Decode ratio 1.000 (synthetic)**:已 ship,保留 + honest标注 synthetic
8. **§4.8 MCTS tree**:已 ship,保留

**Inputs**:
- Phase 1 + 2 + 3 新 ship 的 metrics JSONs
- 已有 paper_v2.tex(Wf-139 Phase 8)

**Output**:
- `paper/section4_results.tex` REWRITE §4.1-§4.8
- 每个 lead claim 带 verbatim numbers + paths + dataset 来源

**Verify**:
- §4.1 真 DrugOOD 数据 ✅(不是 fallback)
- §4.2 seeds + epochs 显式 ✅
- §4.3 5/5 metal × 5/5 seed ✅
- 每个 lead claim dataset 来源标清 ✅

**资源**:CPU-only docs
**ETA**:1 周
**TODO-84** paper rule — **需 user authorization**

### Phase 5(Week 5-6):paper §6 honest-negative 完整

**任务**:

1. **§6.1 PlatinAI 4-cell R² -5.66**(12-d signal floor)
2. **§6.2 L5 Hilbert+ASTSR R² -4.04**(3 Wf rescue attempt)
3. **§6.3 L7 multi-PH+PLL R² -0.078**
4. **§6.4 L10 3-way R² -0.36 / -40.24**
5. **§6.5 m3 Vina cisplatin -3.1**(covalent ceiling)
6. **§6.6 diversity_tanimoto 0.137** vs TargetDiff 0.860
7. **§6.7 m15 RMSD 1.241**(self-RMSD,redesign honest)
8. **§6.8 Hilbert 4-cell R² -6.71**
9. **§6.9 Wf-145 m19 Rh regression + fix fail**(Phase 2 Rh fail 加入)
10. **§6.10 No Tier-1 metallodrug baseline**

**Inputs**:
- Wf-127/128/129 honest-negative ships
- Wf-126 RMSD honest finding
- Wf-141 v2 cisplatin honest partial
- Wf-143 Pt VDW 0 lift honest
- Wf-145 m19 Rh fail

**Output**:
- `paper/section6_limitations.tex` REWRITE §6.1-§6.10
- 每个 honest-negative 带 verbatim numbers + paths + reason

**Verify**:
- §6.1-§6.10 全部 ship ✅
- 每个 honest 有具体数字 + 来源 ✅
- §6.10 No Tier-1 metallodrug baseline 明确 ✅

**资源**:CPU-only docs
**ETA**:1-2 天(overlap Phase 4)
**TODO-84** paper rule — **需 user authorization**

### Phase 6(Week 6):paper submission package ship

**任务**:

1. **`paper/paper_v2.tex`** 完整 rewrite(Wf-139 + Phase 4-5)
2. **`paper/paper_v2.pdf`** build with pdflatex(target 15-25 pages)
3. **`paper/cover_letter_jcim.tex`** REWRITE(Wf-139 已有,refresh)
4. **`paper/highlights_jcim.txt`** REWRITE(8 bullets for editorial)
5. **`paper/suggested_reviewers.txt`** 5 names(Nature Chem / JCIM / BriefBioinf senior)
6. **`paper/submission_package.zip`** assemble

**Inputs**:
- Phase 4-5 output

**Output**:
- `paper/submission_package_2026_09_27.zip`(NEW)
- 5 file complete

**Verify**:
- paper_v2.pdf build 成功 ✅
- 5 file 完整 ✅
- zip 文件结构正确 ✅

**资源**:CPU-only docs
**ETA**:1-2 天
**TODO-84** paper rule — **需 user authorization**

## §4 总时间表

| Phase | 任务 | 资源 | ETA |
|---|---|---|---|
| **Phase 1** | 真 DrugOOD + iMoLD head-to-head | CPU-only | Week 1-2 |
| **Phase 2** | LOMO 5/5 metal × 5/5 seed × 50 epoch | GPU-0 Blackwell | Week 2-4 |
| **Phase 3** | NCI-60 9-cell seed/epochs audit + extend | GPU-0 + GPU-1 | Week 4-5 |
| **Phase 4** | paper §4 rewrite | CPU-only docs | Week 5 |
| **Phase 5** | paper §6 honest-negative 完整 | CPU-only docs | Week 5-6 |
| **Phase 6** | paper submission package ship | CPU-only docs | Week 6 |
| **Total** | | | **6 weeks** |

## §5 Self-check rigor(per phase gate)

### §5.1 Phase 1 gate

| Check | Standard | Verified |
|---|---|:---:|
| DrugOOD 真数据 on disk | ≥3 OOD split files | ? |
| iMoLD numbers 重现 | ±0.01 of 0.7211/0.6884/0.6792 | ? |
| `data_source = "drugood_native"` | explicit string | ? |
| Scaffold overlap | = 0 | ? |
| Tests PASS | ≥8 | ? |

### §5.2 Phase 2 gate

| Check | Standard | Verified |
|---|---|:---:|
| 125 ckpt on disk | 5 metal × 5 seed | ? |
| 5 metal × 5-seed mean AUC ship | each metal has mean+std | ? |
| Ru ≥ 0.59 | maintain Wf-131 P3 level | ? |
| Ir/Os/Re ≥ 0.50 | reach LOMO 0.50 floor | ? |
| Rh ≥ 0.50 | 50% chance per TODO-135 §5 | ? |
| Tests PASS | ≥18 (Wf-145 v2 driver) | ? |

### §5.3 Phase 3 gate

| Check | Standard | Verified |
|---|---|:---:|
| `seeds` 字段 explicit | `[0, 1, 2]` not N/A | ? |
| `epochs` 字段 explicit | `50` not N/A | ? |
| per-cell AUC 与现有一致 | ±0.01 | ? |
| Krasnov 2026 baseline 0.81 对比 | honest ship | ? |
| Tests PASS | ≥12 (NCI-60 multi-task) | ? |

### §5.4 Phase 4 gate

| Check | Standard | Verified |
|---|---|:---:|
| §4.1 真 DrugOOD 数据 | not fallback | ? |
| §4.2 seeds + epochs 显式 | not N/A | ? |
| §4.3 5/5 metal × 5/5 seed | all complete | ? |
| 每个 lead claim dataset 来源 | explicit | ? |

### §5.5 Phase 5 gate

| Check | Standard | Verified |
|---|---|:---:|
| §6.1-§6.10 全部 ship | 10 honest-negative | ? |
| 每个 honest 有具体数字 + path | explicit | ? |
| §6.10 No Tier-1 metallodrug baseline | explicit | ? |

### §5.6 Phase 6 gate

| Check | Standard | Verified |
|---|---|:---:|
| paper_v2.pdf build 成功 | pdflatex no error | ? |
| 5 file 完整 | tex + cover + highlights + reviewers + zip | ? |
| zip 文件结构正确 | unzippable | ? |

## §6 Risk assessment

| Risk | Likelihood | Mitigation |
|---|:---:|---|
| DrugOOD 网络不可达(China firewall) | 40% | 用 TDC DrugOOD 镜像 或 联系 Ji et al. |
| LOMO Rh 训练 fail(n=60 sample size ceiling) | 50% | Phase 2 §6 honest ship + paper §6.9 |
| LOMO Os/Re 数据质量差 | 30% | 5-seed mean + std honest ship |
| NCI-60 9-cell `seeds=N/A` 揭示训练脱节 | 30% | Phase 3 audit → 重新训练 |
| GPU-0 占用 | 5% | RTX PRO 6000 96GB 现在 0% util |
| TODO-84 paper rule 没授权 | 10% | Phase 4-6 需 user 授权 |

## §7 Resource inventory

### 7.1 数据
- ✅ MetalCytoToxDB(已)
- ✅ CrossDocked2020 full(已,183K PDB)
- ❌ DrugOOD 真数据(Phase 1 需下载)
- ✅ NCI-60 9-cell subset(已)
- ✅ BindingDB 1M(备用)

### 7.2 Compute
- ✅ GPU-0 RTX PRO 6000 Blackwell 96GB(0% util)
- ⚠️ GPU-1 RTX 5090 32GB(PID 649993 Wf-131 P3 占用中,预计 1-3 小时完成)
- ✅ 32 CPU + 94GB RAM + 1.8TB disk
- ✅ Vina 1.2.7 + PoseBusters + RDKit + scikit-learn 已装

### 7.3 Tier-1 baselines(已 clone)
- ✅ TargetDiff(ICLR 2023)— cite-only,**不重跑**
- ✅ Pocket2Mol(ICML 2022)— cite-only
- ✅ DiffSBDD(ICML 2023)— cite-only
- ✅ DecompDiff / MiDi / MolCRAFT(ICLR/ICML 2024)— cite-only
- ❌ iMoLD(NeurIPS 2022)— Phase 1 需 clone

### 7.4 ckpt(当前)
- 仅 2 个:`affinity_platinai_seed0.pt` + `affinity_platinai_seed1.pt`
- Phase 2 需 ship 125 ckpt

## §8 Honest framing per party-twenty-spirit §3.5

**承认**:
1. **当前 paper §4 lead claim 大部分没有 ground truth**(fallback data + 未完成 training + 不可 audit seeds)
2. **DrugOOD fallback 是 fake data** — 必须 Phase 1 重建
3. **LOMO 7/125 trainings** — Os/Rh/Re 一个都没训完
4. **NCI-60 9-cell seeds/epochs N/A** — 来源不可 audit
5. **CrossDocked 5/100 pocket subset bias** — paper 必须讨论
6. **Diversity 0.137 vs TargetDiff 0.860** — gap +0.723
7. **Vina -6.929 vs TargetDiff -8.45** — gap +1.521

**不假装**:
1. 不假装 5/9 lead claim 是"完整 ship" — 大部分需 Phase 1-3 重建
2. 不假装 metacytotox_fallback 是 DrugOOD — 必须撤
3. 不假装 7/125 trainings 是"完成" — 5.6%
4. 不假装 `seeds=N/A epochs=N/A` 是"足够" — 必须 audit
5. 不假装 cross-domain comparison 有意义 — TargetDiff cite-only 不重跑
6. 不假装 Rh n=60 能 lift 到 ≥0.50 — 50% chance fail

**paper §6 必须写**:
- 12-d RDKit signal floor mathematical analysis
- 3 Wf rescue attempt table + honest-negative
- L5/L7/L10 architectural ceiling + 12-d → 172-d REGRESSION honest
- m3 Vina covalent ceiling(物理极限)
- diversity_tanimoto pool saturation + closure-bounded cohort limited diversity
- m15 RMSD redesign honest
- Wf-145 m19 Rh fail + Rh n=60 sample size ceiling
- No Tier-1 metallodrug baseline + first framework establishing this protocol

## §9 Final verdict

**Sprint 价值**:
- **paper §4 lead claim 全部基于真数据**:从 5/9 lead 完整 → 9/9 lead 完整
- **paper §6 honest-negative 完整**:从 8/10 → 10/10
- **接受概率**:从 70-80% → **80-90%**(+10-15%)
- **数据完整**:从 metacytotox_fallback + 7/125 → 真 DrugOOD + 125/125 + NCI-60 完整 provenance
- **模型完整**:从 1/5 metal → 5/5 metal × 5/5 seed × 50 epoch

**Likelihood**:
- Phase 1 DrugOOD reach: **70%**(可能 firewall)
- Phase 2 5/5 metal × 5 seed 全部 reach: **60%**(Rh 50% chance fail)
- Phase 3 NCI-60 9-cell reach: **80%**(只是 audit + extend)
- Phase 4-6 paper ship: **90%**(CPU-only docs)

**接受概率**:
- 现在 ship:70-80%(部分 lead 站不住)
- 6 周后 ship:80-90%(全部 lead 真数据 + honest-negative 完整)

**下一步**:启动 Phase 1(clone DrugOOD 真数据) + Phase 2 准备脚本不跑 GPU。

要不要:
- **A**:立即启动 Phase 1(clone DrugOOD 真数据,~1-2 周 CPU-only),同时 Phase 2 准备脚本
- **B**:先 audit Phase 3 NCI-60 9-cell 当前状态(0.5 天 CPU-only),确认实际 seeds/epochs
- **C**:你 review 这个 sprint plan 后,再说

我推荐 **A**(TODO 文件已 ready,执行不需要再授权,Phase 1 CPU-only 不冲突)。

## §10 Cross-reference

- **TODO-138** paper submission 5-day plan(Wf-139 部分 ship)
- **TODO-140** Tier-1 baseline alignment(已 ship Phase 1)
- **TODO-135** m19 LOMO Rh oversample regression fix(Wf-145 fail,本 TODO Phase 2 处理)
- **TODO-136** m5 + m14 + m15 residual lift(Wf-140 ship)
- **TODO-137** L5/L7/L10 R² architectural ceiling(已 ship)
- **Wf-139** SERIAL Q2 pipeline 9 phases(本 TODO 是 Wf-139 的扩展)
- **Wf-140** m5 SA 5-pocket 已 ship
- **Wf-142** per-pocket high_affinity ex=32 已 ship
- **Wf-144** relaxed_success threshold 已 ship
- **Wf-145** m19 Rh fix fail(Wf-145 + Phase 2 接力)
- **Wf-131** P2 Rh regression origin + Wf-131 P3 partial 5/5 Ru + 2/5 Ir
- **TODO-84** paper rule active — Phase 4-6 需 user authorization

## §11 Related skill alignment

- `party-twenty-spirit` §3.5: 0-tolerance honest-negative framing → §8
- `mol-metal-mechanism-mapping`: framework 三层 stack → §1.1
- `mol-metal-anticancer`: anticancer domain context + NCI-60 panel → §1.1, §2
- `mol-metal-lambda-calculus`: Lambda + CFM + MCTS architecture → §4 lead claims
- `mol-metal-evaluation-rigor`: benchmark scale alignment + self-check → §5
