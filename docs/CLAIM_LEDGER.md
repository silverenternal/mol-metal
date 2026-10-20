# CLAIM LEDGER — 主张 → 证据 → 判定

**日期**：2026-10-20（STR-7）
**规则**：每条主张必须指向一个磁盘上的 artifact 指针；没有指针的写 `UNMEASURED`，
被外部依赖挡住的写 `BLOCKED`。**这张表是正文的上限**——正文不能比它走得远。
**数值来源**：`metrics/by_metric/*.json`；表格由
`molmetal/scripts/paper_metric_tables.py --strict` 机械生成（12 张表 / 0 MISSING）。

| ID | 主张 | 判定 | 关键数值 | 证据 |
|---|---|---|---|---|
| C1 | 金属感知表示（配合物 + 态 + 配位）显著优于配体单独 | **SUPPORTED** | Δ R² +0.1002（p=2.3e-20）；metal_blind 0.4873 vs full 0.5556 | `msrs_complex_features.json:/splits/fragment_grouped/complex/delta_r2_vs_main`；`msrs_metal_ablation.json:/protocols/ligand/arms/metal_blind/r2` |
| C2 | 跨论文（整篇 DOI 留出）绝对强度预测达 R² 0.36 | **SUPPORTED** | 0.3767（MACCS 树）/ 0.3811（融合） | `msrs_feature_expansion_5seed.json:/protocols/doi/r2_seed_averaged`；`msrs_feature_fusion.json:/protocols/doi/fused_r2` |
| C3 | 模型能判断同一队列里"哪个金属更好" | **NOT SUPPORTED**（且低于恒定预测） | 多数类 0.6009；相减 0.5826（平衡 0.5272）；直接 0.5183（平衡 0.5162）；lambdarank 0.0734 | `msrs_delta_metric_audit.json:/routes/level_differencing/balanced`；`msrs_delta_metric_audit.json:/baselines/majority_class_rate`；`msrs_rank_within_cohort.json:/ties/sign_accuracy_ties_half`；`msrs_rank_diagnostics.json:/coverage/metal_pairs_gated` |
| C4 | 树 + GNN 融合提升跨论文 R² | **SUPPORTED（仅 R²）** | DOI +0.0217 [+0.0044,+0.0377]；融合后 Δ 符号 0.5826→0.5505 | `msrs_fusion_protocols.json:/protocols/doi/arms/stack_foldwise/vs_tree` |
| C5 | 不确定性区间校准 | **SUPPORTED（有边界）** | DOI 覆盖 0.9004、中位宽 2.0052；同队列多金属仅 0.8096 | `msrs_calibration.json:/protocols/doi/variants/global/coverage` |
| C6 | 跨金属迁移达到可用水平 | **NOT SUPPORTED**（LOMO）；within-metal 另计 | 可解析 Ru/Ir 0.5974（集成 0.5947）< 0.6；Rh 0.3812 CI 含 0.5；within-metal Ru 0.8271 / Ir 0.8430 | `lomo_resolvability_audit.json:/aggregates/resolvable_only/value`；`lomo_seed_ensemble.json:/aggregates/resolvable_only/auc_of_mean_score` |
| C7 | 生成物可合成且新颖的金属药物 | **NOT SUPPORTED**（全语料真生成池已存在，18/18 epoch，SA≤4.5 = **0.28%**，低于 256 样本的 0.69%；联合闸 0/10,190） | 全语料自由生成池 `cfm_c7_fullgen_2026_10_20`（n_train=26,715，**18 epoch** × 704 = **12,668** 可打分 / 10,190 唯一 SMILES）：SA 6.99、QED 0.124、SA≤4.5 **0.28% [0.21,0.39]**（36/12,668）；对照 n_train=256 池 SA≤4.5 **0.69% [0.42,1.11]**（n=2,330）。单片段率 2.84→12.22% 随 epoch 上升（连通性变差），SA 不动。联合闸（SA≤4.5 且 novelty≥0.90）**0/10,190**，SA 子集平均新颖度 0.8587。`gen_validity`=1.0 是松散原子假象（`sa_scored`=0） | `c7_scale_matched_pools.json:/by_pipeline/generation/by_tier/full_corpus/rate_sa_le_gate`；`.../wilson_ci95`；`.../by_tier/small/rate_sa_le_gate`；`.../tiers_missing`；`c7_fullgen_pool_quality.json:/first_vs_last`；`molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv` |

---

## C1 — 金属感知表示承重

**能写**：「把配合物、金属氧化态与配位环境显式编码，比只用配体指纹提高 R² 0.1002
（片段分组，p=2.3e-20）；去掉金属身份码掉 0.0683，去掉态掉 0.0272，去掉配位掉
0.0130。」

**不能写**：不能写「架构创新」——增益来自**输入里有金属**，不是来自新的网络结构。

## C2 — 跨论文绝对预测

**能写**：「整篇论文留出下 R² 0.3767（5 seed，MACCS+RDKit 树），与 GNN 融合后
0.3811；相对 ECFP4 树 0.3380 的配对增益 +0.0387 [+0.0241,+0.0535]，置换控制
0.3275 证明增益来自描述符内容而非维度。」

**不能写**：「配体分组的 0.5593 是跨论文性能」——跨论文就是 0.3767。也不能写
「0.3767 代表可部署精度」——DOI 是论文代理，见 C2 的 `feature_headroom_audit`：
DOI offset 值 +0.1109 R²，而金属化学在 DOI 之后只有 −0.0003。

## C3 — 排序（哪个金属更好）

**能写**：「门内 218 对的多数类率是 **0.6009**（60.1% 正号，只有 Os–Ru 与
Ir–Rh 两个金属对）。水平相减 0.5826 **低于恒定预测**，平衡符号准确率
**0.5272 [0.4359, 0.6198]**（TPR 0.80 / TNR 0.25）；Δ 直接回归 0.5183，
平衡 **0.5162 [0.3939, 0.6430]**；lambdarank 0.0734 是**平局假象**——218 对里
181 对（83.03%）两行拿到完全相同的分数，按平局 0.5 计是 **0.4885**，即金属盲。
门内只有 **3/10** 个金属对比（Os–Ru 153、Ir–Rh 64、Rh–Ru 1），七个对比缺失，
五金属排序在本队列上**不可识别**。**没有任何路线超过恒定预测**（0.6009），
平衡准确率与 0.5 无差别；逐对比看，Os–Ru 是金属先验赢（0.6209 vs 0.4641），
Ir–Rh 是学习模型赢（0.6562 vs 0.5625），按对比取优的上界也只有 0.6284 < 0.65。」

**不能写**：「0.5826 接近 0.65 的门槛」——它比「永远答正号」低 0.018；也不能
写「模型学到了排序」：它学到的是基准率。也不能写「直接回归显著更差」：区间重叠。

## C4 — 融合

**能写**：「融合在 DOI 上 +0.0217（CI 排除 0），ligand +0.0401、scaffold +0.0310；
year 的 CI 含 0。**融合不改善排序**（0.5826→0.5505）。」

**不能写**：「融合是全面更强的模型」——它在排序上更差，且 ligand 之外的增益有一半
来自 GNN 的 seed 平均而非融合本身。

## C5 — 校准

**能写**：「DOI 留出、α=0.10 的 split conformal 覆盖 0.9004，中位宽 2.0052 log10；
点校准斜率 0.9256。」

**不能写**：「区间在任何子群都可行动」——同一队列多金属（851 行）覆盖只有 0.8096，
单金属 0.9050。这个子群必须单独报。

## C6 — 跨金属迁移

**能写**：「within-metal 分类在 5 seed 上稳定超过 Krasnov：Ru 0.8271（0.8128–0.8363）
vs 0.81，Ir 0.8430（0.8295–0.8597）vs 0.73；最差 seed 仍过线。」

**不能写**：「跨金属迁移可用」——按项目自己的 `MIN_UNIQUE_FOR_GATE=200`，
只有 Ru（1,311）与 Ir（538）可入闸，其 LOMO 为 **0.5974**（seed 集成 0.5947），
低于 0.6 目标 0.0026，且两个 CI 都包含 0.6。Rh（51）/Os（36）/Re（74）低于阈值，
**不得入闸**：Rh 的 0.3812 的 CI 是 [0.2217, 0.5407]，包含 0.5，它是测量地板。
也不能写「差 0.04」——那是把三个不可解析的折等权平均进去的结果。

## C7 — 生成

**能写**：「全语料的自由生成池现在存在（`cfm_c7_fullgen_2026_10_20`，n_train=26,715，
**18/18 个 epoch** × 704 = 12,669 行 / 12,668 个可打分、10,190 个唯一 SMILES，SA 6.99、
QED 0.124）。它的 SA≤4.5 是 **0.28% [0.21,0.39]**（36/12,668），**低于** n_train=256 池的
0.69% [0.42,1.11]（n=2,330）：把语料放大到全量**没有**改善可合成性。18 个 epoch 内单片段率
从 2.84% 升到 **12.22%**（合计 8.51%），SA 不动，说明连通性随训练**变差**而不是变好。
联合闸（SA≤4.5 且 novelty≥0.90）是 **0/10,190**，SA 通过子集的平均新颖度只有 0.8587，
低于 0.90——通过 SA 的 36 个分子都是训练集的近亲。」

**不能写**：「生成器做不出可合成的金属配合物」——虽然现在有 **18/18 个 epoch**（`exit=0`，
wall 11,232 s），仍**没有**证据说这是收敛值或天花板；也**不能**写「全语料 SA≤4.5 可达
33.7%」——那是重建池的数字，v1.0 已撤回；也不能用 `gen_validity=1.0` 当质量证据——该池
`sa_scored`=0，1.0 是松散原子假象。`--sa-penalty` 仍然零梯度、无可微代理，机制不存在。
对接仍 BLOCKED。

---

## UNMEASURED（完全没有数字）

| 主张 | 为什么没有 | 需要什么 |
|---|---|---|
| 外部数据集上的绝对精度 | PlatinAI 只有 Pt、MSRS 无 Pt、NCI-60 无 SMILES，三键交集 0 | 一个新的、含多贵金属且带 SMILES 的外部集 |
| 细胞系特征带来的增益 | 0 个 CCLE/GDSC 文件；46/711 细胞系覆盖 80% 数据 | 细胞系分子特征表 |
| 湿实验验证 | 硬性不做（§5.4） | — |

## BLOCKED（有依赖挡着）

| 主张 | 阻断原因 | 证据 |
|---|---|---|
| 贵金属配合物的对接打分 | Vina 无贵金属参数 | catalog §6.52d |
| SA 引导解码 | 无梯度 + 无可微代理 + 单 checkpoint | catalog §6.140 |
| 3-seed 生成扫描 | 每 run 只有一个 `seed0` checkpoint | catalog §6.140 |

---

## 与表格的对应

| 台账 | 表 |
|---|---|
| C1 | T4_mechanism |
| C2 | T2_generalization（单 seed ECFP4）、T11_generalization_strong（5 seed 三臂） |
| C3 | T12_ranking |
| C4 | T11_generalization_strong、T2_generalization |
| C5 | T3_calibration |
| C6 | T10_external、`metal_activity_transfer_seed_stability.json` |
| C7 | T7_generation |

**重生成命令**：

```bash
OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/paper_metric_tables.py --strict
```

退出码 0 = 0 MISSING。`manifest.json` 记录每个源文件的 sha256；重跑 artifact 后
必须重跑本脚本，否则表会与产物脱节。
