# 里程碑

按 Phase 推进,每个 Phase 都有可交付成果 + 验证手段。

## Phase 0: 环境就绪 (1-2 天)

- [x] ROCm 7.2 + torch 2.14 + triton-rocm 3.8 (MolFlow-Triton 已完成) (2026-09-11: reports/diffsbdd_targetdiff_1h36.md)
- [x] RDKit 已装在 venv (`uv pip install rdkit` 已用过) (2026-09-11: reports/mmff94_fix.md, reports/f5_molsimplify_plan.md)
- [ ] 验证 RDKit 3D 嵌入吞吐(单核 + 多核)
- [ ] 把 MolFlow-Triton 的 EGNN + scatter 移植到独立 `molmetal/` 子包
- [ ] 写一个 mini-test 验证 forward+backward

**交付**:一个能跑通 forward+backward 的空 `molmetal/` 包

## Phase 1: 数据准备 (2-3 天)

- [ ] 下载 MetalCytoToxDB.csv (6.4 MB) 到 `data/`
- [ ] 探查数据:列分布、缺失值、metal 计数
- [ ] 写 `molmetal/data/cytotox.py`:`MetalCytotoxDataset` 类
  - 过滤 Time ≥ 24h
  - 过滤 IC50 缺失/SE>IC50
  - 二分类:IC50 < 10 μM = active
  - 数值:pIC50 = -log10(IC50 μM)
  - RDKit Mol + ETKDGv3 + MMFF 3D 嵌入
  - 缓存到 `data/qm_3d_cache/` 一次
- [ ] 时间切分 split:pre-2024 train / post-2024 test
- [ ] 化学切分 split:Tanimoto > 0.7 dissimilar molecules
- [ ] sanity 检查:Ru/Ir/Pt 子集的样本数

**交付**:`molmetal.data.MetalCytotoxDataset` + 缓存 + 3 个 split
**验证**:每个 split 的 train/val/test 样本数 + 平均 IC50

## Phase 2: Baseline 复现 (2-3 天)

- [ ] 实现 Morgan FP + XGBoost baseline(复现 Krasnov 2026)
  - target:Ru AUC ≥ 0.81
- [ ] 加 PyG-D-MPNN baseline (Chemprop-style)
  - target:Ru AUC ≥ 0.82(应该会超 Morgan FP,因为 D-MPNN 是 2026 SOTA)
- [ ] 训练 + 评估脚本
  - 输出:per-metal AUROC,overall AUPRC
- [ ] 留一 (Leave-One-Cell-Line-Out) 评估
- [ ] 写 baseline 报告

**交付**:`molmetal/scripts/baseline.py` 跑出报告 + baseline 数字
**验证**:Ru AUC ≥ 0.81 (≥Krasnov),IR AUC ≥ 0.73

## Phase 3: 3D 增强 (3-5 天)

- [ ] `molmetal/models/egnn_layer.py` — 复用 MolFlow-Triton 的 EGNNLayer + dative bond edge type
- [ ] `molmetal/models/dmpnn.py` — D-MPNN with dative edge type
- [ ] `molmetal/models/fusion.py` — concat + cross-attention fusion
- [ ] `molmetal/models/head.py` — pIC50 + activity 双头输出
- [ ] `molmetal/scripts/train.py` — full training loop with EMA + warmup + grad clip
- [ ] mini-test: 1 epoch batch=4,验证 链路
- [ ] Ru subset:50 epochs batch=64
- [ ] 评估 + 与 baseline 对比

**交付**:`molmetal/models/` 完整 + Ru subset 训练报告
**验证**:Ru AUC > 0.86(超过 Krasnov 2026 + 0.05)

## Phase 4: 跨金属泛化 (3-5 天)

- [ ] 训练 Ru + Ir 多金属模型(Krasnov 2026 报告了 "Multi-metal model")
- [ ] 评估:Leave-One-Metal-Out(LOMO)— 在 Pt 上 train,在 Ru/Ir 上 test
- [ ] 化学切分 strictness:Tanimoto 0.7 / 0.8 / 0.9 阈值
- [ ] 时间切分:pre-2024 / post-2024 hit rate

**交付**:LOMO + 时间切分评估
**验证**:post-2024 hit rate > 0.75(超过 PlatinAI 的 0.72)

## Phase 5: 论文-grade 评估(可选,1-2 周)

- [ ] 接入 MB Finder 的 17,732 Pt 数据 → 时间切分 benchmark
- [ ] 在 NCI-60 上验证通用性(顺铂/卡铂/奥沙利铂)
- [ ] tmQM 预训练 + fine-tune 实验
- [ ] 反离子特征消融实验
- [ ] 论文草稿(Methods / Results / Discussion)

**交付**:完整 evaluation suite + 论文 manuscript
**验证**:多个 SOTA 指标提升 + 严格 split 下不掉点

## 风险与失败模式

| 风险 | 概率 | 缓解 |
|---|---|---|
| RDKit 3D 嵌入对金属复合物失败 | 中 | catch + 用 SMILES 2D fallback;统计失败率 |
| EGNN 训练不稳定 | 中 | 用 MolFlow-Triton 的 warmup + EMA + grad clip 组合 |
| 数据类别严重不平衡(Ru active 占少数) | 高 | weighted BCE + class weights |
| 3D conformer 缓存占满 /tmp | 低 | 用 hash 分桶,只 cache 成功的 |
| 时间预算超支 | 中 | Phase 1-3 优先;Phase 4-5 可拆 |

## 关键决策点

| 决策 | 何时决定 | 选项 |
|---|---|---|
| 先做 Ru 子集还是全金属 | Phase 1 末 | (a) Ru first(快迭代) (b) 多金属 first(general) |
| 用不用 tmQM 预训练 | Phase 3 末 | (a) 直接训(简单) (b) 预训练(可能涨点) |
| 是否加反离子特征 | Phase 3 中 | (a) 是 (b) 否(简化) |
| 是否考虑细胞系 embedding | Phase 3 中 | (a) 是 (b) 否 |

## 时间估算

| Phase | 估计时间 | 累计 |
|---|---|---|
| 0 | 1-2 天 | 2 天 |
| 1 | 2-3 天 | 5 天 |
| 2 | 2-3 天 | 8 天 |
| 3 | 3-5 天 | 13 天 |
| 4 | 3-5 天 | 18 天 |
| 5 (optional) | 1-2 周 | 32 天 |

**Phase 0-3 是核心**(~2 周内)。达到 Ru AUC 0.86+ 就足以构成"超过 2026 SOTA" 的论据。

## 与现有 MolFlow-Triton 项目的关系

- **MolFlow-Triton**:ROCm/Triton 验证工具,EGNN + flow matching(已完成)
- **Mol-Metal**:实际科研产出,用 MolFlow-Triton 的 kernel/EGNN/autograd infrastructure
- **复用**:triton kernels, EGNNLayer, scatter_sum (with autograd), EMA / warmup / cosine schedule

**不要**:
- 在 MolFlow-Triton 目录里堆 MetalCytotox 代码(分开!)
- 改 Triton kernel(已稳定)
- 改 EGNNLayer 内部(只在外面包 dative bond type)

**可以**:
- `import models._scatter` / `models.velocity_net.EGNNLayer` 直接复用

## 实验跟踪日志

每次跑实验后,记到 `TODO/experiments/<date>_<experiment>.md`:
- commit hash
- config
- 结果 (AUC, hit rate)
- 观察 + 下一步

## 现在该问你的问题

1. 范围:先 Ru only?还是 5 金属全做?
2. 是否做 Phase 4(跨金属泛化)还是 Phase 4 砍掉直接走论文?
3. 是否需要把 MolFlow-Triton 项目架构调整为 monorepo (让 `molmetal/` 和当前项目并列),还是直接放平级目录?

### Open phases

下列 phase 在 2026-09-11 仍未完成,需结合本次 review 中列出的报告(mapping 见各 phase 自身的 `TODO cross-ref`):

- **Phase 1: 数据准备** — 关联 `reports/f2_tmqm_pretraining.md`、`reports/f3_metalloprotein_coverage.md`、`reports/f1_multi_component_parser.md`(`02_data/datasets.md`)
- **Phase 2: Baseline 复现** — 关联 `reports/mmff94_fix.md`、`reports/h1_sa_score_ertl.md`、`reports/h2_pic50_predictor_calibration.md`、`reports/h3_retrosynthesis_check.md`、`reports/pocket2mol_vs_lambda_1h36.md`(`03_baselines/reproduce_baselines.md`)
- **Phase 3: 3D 增强** — 关联 `reports/h2_pic50_predictor_calibration.md`、`reports/f2_tmqm_pretraining.md`(`04_architecture/`)
- **Phase 4: 跨金属泛化** — 关联 `reports/audit_lambda_upper_bound.md`、`reports/mcts_strengthening.md`(`07_risks/`, `13_lambda_clickchem/plan.md`)
- **Phase 5: 论文-grade 评估** — 关联 `reports/h4_paper_grade_comparison.md`、`reports/h5_lambda_honest_framing.md`、`reports/f4_paper_leakage_section.md`(`01_research/sota_landscape.md`, `06_milestones/`, `07_risks/open_questions.md`)