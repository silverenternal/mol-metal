# CLAIM LEDGER — 主张 → 证据 → 判定

**日期**：2026-10-21（EXT-8；C1–C7 为 2026-10-20 STR-7，新增 C8–C10）
**规则**：每条主张必须指向一个磁盘上的 artifact 指针；没有指针的写 `UNMEASURED`，
被外部依赖挡住的写 `BLOCKED`。**这张表是正文的上限**——正文不能比它走得远。
**数值来源**：`metrics/by_metric/*.json`；表格由
`molmetal/scripts/paper_metric_tables.py --strict` 机械生成（12 张表 / 0 MISSING）。

| ID | 主张 | 判定 | 关键数值 | 证据 |
|---|---|---|---|---|
| C1 | 金属感知表示（配合物 + 态 + 配位）显著优于配体单独 | **SUPPORTED** | Δ R² +0.1002（p=2.3e-20）；metal_blind 0.4873 vs full 0.5556 | `msrs_complex_features.json:/splits/fragment_grouped/complex/delta_r2_vs_main`；`msrs_metal_ablation.json:/protocols/ligand/arms/metal_blind/r2` |
| C2 | 跨论文（整篇 DOI 留出）绝对强度预测达 R² 0.36 | **SUPPORTED** | 0.3767（MACCS 树）/ 0.3811（融合） | `msrs_feature_expansion_5seed.json:/protocols/doi/r2_seed_averaged`；`msrs_feature_fusion.json:/protocols/doi/fused_r2` |
| C3 | 模型能判断同一队列里"哪个金属更好" | **NOT SUPPORTED**（且低于恒定预测；balanced 口径下 listwise 显著优于 0.5，但 raw 仍输） | 多数类 0.6009；相减 0.5826（平衡 0.5272）；直接 0.5183（平衡 0.5162）；lambdarank 0.0734。放宽配对键到 (ligand, cell) 得 360 对、多数类 0.6222：pointwise 平衡 0.4705，**listwise 平衡 0.5985 [0.5262,0.6663]**，raw 0.5417 仍低于 0.6222 | `msrs_delta_metric_audit.json:/routes/level_differencing/balanced`；`msrs_delta_metric_audit.json:/baselines/majority_class_rate`；`msrs_rank_within_cohort.json:/ties/sign_accuracy_ties_half`；`msrs_rank_diagnostics.json:/coverage/metal_pairs_gated`；`msrs_listwise_ranker.json:/pairs/ligand_cell/overall/balanced_accuracy` |
| C4 | 树 + GNN 融合提升跨论文 R² | **SUPPORTED（仅 R²）** | DOI +0.0217 [+0.0044,+0.0377]；融合后 Δ 符号 0.5826→0.5505 | `msrs_fusion_protocols.json:/protocols/doi/arms/stack_foldwise/vs_tree` |
| C5 | 不确定性区间校准 | **SUPPORTED（有边界）** | DOI 覆盖 0.9004、中位宽 2.0052；同队列多金属仅 0.8096 | `msrs_calibration.json:/protocols/doi/variants/global/coverage` |
| C6 | 跨金属迁移达到可用水平 | **NOT SUPPORTED**（LOMO）；within-metal 另计 | 可解析 Ru/Ir 0.5974（集成 0.5947）< 0.6；Rh 0.3812 CI 含 0.5；within-metal Ru 0.8271 / Ir 0.8430 | `lomo_resolvability_audit.json:/aggregates/resolvable_only/value`；`lomo_seed_ensemble.json:/aggregates/resolvable_only/auc_of_mean_score` |
| C7 | 生成物可合成且新颖的金属药物 | **NOT SUPPORTED**（全语料真生成池已存在，18/18 epoch，SA≤4.5 = **0.28%**，低于 256 样本的 0.69%；联合闸 0/10,190） | 全语料自由生成池 `cfm_c7_fullgen_2026_10_20`（n_train=26,715，**18 epoch** × 704 = **12,668** 可打分 / 10,190 唯一 SMILES）：SA 6.99、QED 0.124、SA≤4.5 **0.28% [0.21,0.39]**（36/12,668）；对照 n_train=256 池 SA≤4.5 **0.69% [0.42,1.11]**（n=2,330）。成分均值 22.17→**3.45**（语料 3.34），但第二成分是单原子的占 **60.5%**（语料 1.5%）、第二成分均 **1.98** 原子（语料 12.09）。键密度 1.137（语料 1.069），3 元环 **67.5%** vs 语料 **0.01%**。联合闸（SA≤4.5 且 novelty≥0.90）**0/10,190**，SA 子集平均新颖度 0.8587。键头 teacher-forced `bond_f1` 0.173→0.605，但重建池只有 **1.35/1.49** 个成分（语料 3.34）。`gen_validity`=1.0 是松散原子假象（`sa_scored`=0） | `c7_scale_matched_pools.json:/by_pipeline/generation/by_tier/full_corpus/rate_sa_le_gate`；`.../wilson_ci95`；`.../by_tier/small/rate_sa_le_gate`；`.../tiers_missing`；`c7_fullgen_pool_quality.json:/reference`；`.../per_epoch/17`；`.../head_on_real_geometries`；`.../bond_head_trajectory`；`molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv` |
| C8 | 跨金属外推失败的主因是结构（排序）不迁移，不是水平估不准 | **SUPPORTED** | offset 占平方误差 0.045–0.100，scale 0.198–0.272，**shape 0.653–0.757**；完美水平只值 +0.078 R² 且 4/5 金属仍为负 | `msrs_lomo_error_decomposition.json:/dominant_term`；`.../mechanism`；`msrs_metal_descriptor_level.json:/counterfactual_summary` |
| C9 | 公开语料里不存在独立的跨金属外部 cohort | **SUPPORTED**（领域事实） | IrCytoToxDB 与 MSRS 的 DOI 重叠 221/222 = 99.55%，仅 1 个新 DOI；匹配的 653 个 (DOI,细胞系) 格里 89.4% 数值相同（\|Δ\|<0.01），中位 \|Δ\|=0.0，r=0.9611 | `msrs_external_cohort_probe.json:/ircytotoxdb/doi_overlap_share`；`.../ircytotoxdb/row_agreement` |
| C10 | 给新金属少量测量后校准，可以让模型可用 | **NOT SUPPORTED**（且校准使结果变差） | ligand-disjoint 切分上 k=50 时 R²≥0 的金属 **0/5**；k=5 时"比零样本更差"的重复占 55–95% | `msrs_kshot_calibration.json:/endpoints/potency/splits/metal_out_ligand_disjoint/verdict_at_gate_k` |

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

**UP 轮补充（2026-10-21，可写但要带口径）**：「把配对键从 (ligand, cell, DOI, time)
放宽到 (ligand, cell) 后，同一批折上有 360 对、4 个金属对，多数类率升到 **0.6222**。
pointwise 平衡准确率反而降到 **0.4705**；改成 listwise（`lambdarank`，group =
(ligand, cell)）后平衡准确率 **0.5985 [0.5262, 0.6663]**，是这条线上第一次 CI 不含 0.5
的排序结果。但 raw 符号准确率只有 **0.5417**，仍低于恒定预测 0.6222，预注册的三条
门槛（balanced ≥0.58 ✓、CI 下界 >0.52 ✓、raw ≥0.63 ✗）没有同时满足。」

**不能写**：「0.5826 接近 0.65 的门槛」——它比「永远答正号」低 0.018；也不能
写「模型学到了排序」：它学到的是基准率。也不能写「直接回归显著更差」：区间重叠。
也不能写「排序已经可用了」——raw 口径仍输给恒定预测；更不能把 0.5985 写成
「比 0.5826 更好」，两者配对键不同、样本不同，不是同一估计量的两次测量。

## C4 — 融合

**能写**：「融合在 DOI 上 +0.0217（CI 排除 0），ligand +0.0401、scaffold +0.0310；
year 的 CI 含 0。**融合不改善排序**（0.5826→0.5505）。」

**不能写**：「融合是全面更强的模型」——它在排序上更差，且 ligand 之外的增益有一半
来自 GNN 的 seed 平均而非融合本身。

**来源订正（2026-10-21）**：这四个协议来自 `msrs_dmpnn_protocols.py` 的模块常量
`SCHEMES = ("ligand", "doi", "scaffold", "year")`，**不是**来自配置里的 `eval.splits`。
`configs/msrs_dmpnn_protocols_gpu0.yaml` 曾写着 `eval.splits: [canonical_grouped]`，
而该脚本从不读这个键，`canonical_grouped` 也不在它跑的四个协议里——配置描述的是
另一个实验。该键（连同 `eval.baseline_arm`、`arms.*`、`output.path`）已删除，理由写在
配置注释里。**数字不受影响**（键从未被读），但任何引用「这份配置选了什么协议」的
说法都必须改指向 `SCHEMES`。

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

**EXT-1 补上的机制（2026-10-21）**：把 LOMO 的平方误差按 Murphy 恒等式拆成
offset / scale / shape 三项后，**shape 占 0.653–0.757，offset 只占 0.045–0.100**。
所以"跨金属外推失败"不是"新金属的活性水平估不准"，而是"配体的 SAR 本身几乎不迁移"。
两条限制必须一起写：

- 表观迁移主要来自**配体重叠**。potency 的合并 ρ 在 metal_out 是 +0.296，
  去掉配体重叠后掉到 +0.158；cisplatin 从 +0.269 掉到 +0.062。
- 任何校准的上限是 ρ²，在诚实切分上只有 0.002–0.131。**不要写"校准后可用"。**

## C7 — 生成

**能写**：「全语料的自由生成池现在存在（`cfm_c7_fullgen_2026_10_20`，n_train=26,715，
**18/18 个 epoch** × 704 = 12,669 行 / 12,668 个可打分、10,190 个唯一 SMILES，SA 6.99、
QED 0.124）。它的 SA≤4.5 是 **0.28% [0.21,0.39]**（36/12,668），**低于** n_train=256 池的
0.69% [0.42,1.11]（n=2,330）：把语料放大到全量**没有**改善可合成性。
联合闸（SA≤4.5 且 novelty≥0.90）是 **0/10,190**，SA 通过子集的平均新颖度只有 0.8587，
低于 0.90——通过 SA 的 36 个分子都是训练集的近亲。
成分数会收敛、成分大小不会：语料的金属是单独一个 component（训练集 69.1% 是 3 个
component，均值 3.34，单一 component 只占 0.07%），所以「单片段」在这个数据集里**不是**
质量指标。生成池的成分均值从 22.17 降到 **3.45**（语料 3.34），但 60.5% 的多成分分子
第二成分是**单个原子**（语料 1.5%），第二成分平均 **1.98** 个原子（语料 12.09）——
是一个融合大块加零散原子，不是几个配体。
键密度到 epoch 17 已达 1.137 键/重原子（真实语料 1.069），但 3 元环占比 **67.5%**，
真实语料只有 **0.01%**（6 元环 2.8% vs 87.5%）：模型学到的是键的**数量**，不是键的**拓扑**。
键头在真实几何上 teacher-forced 反而在变好（`bond_f1` 0.173→0.605），但同样过连接：
两个重建池平均只有 **1.35 / 1.49** 个成分（语料 3.34），说明过度成键从键头就开始了，
不能只归咎于采样器。
Wf-A107 的 5-epoch 方向测试把「拓扑没修好」钉在环先验上：环模板先验（OPT-5）在
1e-4 / 1e-3 / 1e-2 三个权重下都没有把 5/6 占比推上去（0.172 → 0.182 / 0.185 / 0.191），
却把键存在 AP 从 **0.5617** 压到 **0.5365 / 0.4513 / 0.4670**；在键头自己的图（真几何，
阈值 0.5）上，权重每升一级环/mol 从 **23.8 → 28.0 → 48.8 → 76.1**，精确率
**0.555 → 0.227**、召回钉在 0.90–0.93——加环的办法是加边。同一协议里键头的
RBF 距离 + 成对化学特征（Wf-A72 Op2/Op3，先验关闭）在 **3 个 seed** 上都降低 3 元环占比
（配对均值 **−0.179**，3/3，p=0.034），键存在 AP 配对均值 **+0.137**（3/3 同号，但
p=0.106，且单 seed 的 +0.194 与 run 间离散度 0.133 同量级，**不得**当作效应量），
5/6 占比 +0.050（2/3，p=0.197，未与噪声分离）。
Wf-A107b 把先验改成逐边环隶属 BCE（`formulation=edge_ring`，梯度对图密度是 intensive 的：
探针 0.225 → 0.084，2.7x；count 是 7,540x）后，3 个 seed 上键 AP **不变**（配对均值
−0.001，p=0.92）、3 元环占比 **−0.142**（3/3，p=0.016）、5/6 占比 **+0.072**（3/3，
p=0.065）；但同一检查在键头自己的图上全部为 null（3 元环 +0.002，p=0.98），所以
**不能**说键头学会了局部闭环——效应只出现在解码之后，机制未解释。」

**不能写**：「生成器做不出可合成的金属配合物」——虽然现在有 **18/18 个 epoch**（`exit=0`，
wall 11,232 s），仍**没有**证据说这是收敛值或天花板；也**不能**写「全语料 SA≤4.5 可达
33.7%」——那是重建池的数字，v1.0 已撤回；也不能用 `gen_validity=1.0` 当质量证据——该池
`sa_scored`=0，1.0 是松散原子假象。**也不能再把单片段率当质量趋势**——它在这个数据集里
既不表示「在学会成键」也不表示「连通性变差」，两种读法都已撤回。
**也不能写「环模板先验修好了环拓扑」**——三个权重下 5/6 占比最多 +0.019，而键 AP 最多 −0.095；
**也不能把 Wf-A96 的 rings/mol 0 → 7.05 记成先验的功劳**——匹配的无先验对照在同一协议、
同一 epoch 自己就到 **6.542**，先验的边际贡献是 **+0.51**（§6.152）。
`--sa-penalty` 仍然零梯度、无可微代理，机制不存在。
对接仍 BLOCKED。**也不能说这是「六项全开」的结果**：逐 epoch 计数器只有 epoch 4 记录为干净，
epochs 5-17 每轮都有 1-6 步某个加权项返回 0（`EnergyConservation` 31 步 / `SBOO3Term` 52 /
`FmOdeTerm` 19），epochs 0-3 是未记录而非零（计数器在 run 中途才落地）。

## C8 — 失败在哪：水平还是结构

**能写**：「把留出金属的平方误差按 Murphy 恒等式拆成三项，shape（排序）占
0.653–0.757，scale（压缩）占 0.198–0.272，offset（水平）只占 0.045–0.100。
因此零样本跨金属外推的瓶颈不是"新金属的活性水平估不准"：即使把真实水平
直接给模型（oracle offset），R² 也只提高 +0.078，五个金属里仍有四个是负的。
预测是压缩的（sd_pred 0.31–0.54 vs sd_obs 0.64–0.96，slope 0.24–1.16），
所以任何校准的算术上限是 ρ²，在 ligand-disjoint 切分上只有 0.002–0.131。」

**不能写**：「校准一下就能用」——见 C10。也不能写「中心化后 R²>0.10 说明迁移成立」：
预注册门要求 ≥3/5 金属，实测 2/5（Os +0.316、Re +0.106），判 PARTIAL；
去掉配体重叠后只剩 1/5（Os +0.080），判 SAR_FAILS。

## C9 — 外部 cohort 到底存不存在

**能写**：「这个领域目前没有机器可读的独立外部 cohort。最大的公开 Ir 语料
IrCytoToxDB（Scientific Data 2024，2,694 条 / 222 篇 / 127 个细胞系，CC-BY-4.0）
与 MSRS 的 DOI 重叠是 221/222 = 99.55%，只有 1 个新 DOI；在 653 个匹配的
(DOI, 细胞系) 格里，89.4% 的数值与 MSRS 相同到 0.01 log10 以内，中位差 0.0。
也就是说它不是外部测试集，是同一份 curation 的 Ir 切片。」

**不能写**：「我们没找到外部数据是因为搜得不够」——本轮逐一实测了
ChEMBL 29 本地库（0 个目标金属）、PlatinAI（只有 Pt）、NCI-60（无 SMILES）、
DepMap（Cloudflare Turnstile）、GDSC（HTTP 410）、Cellosaurus（超时）、
GEO GSE36133（只有 CEL 原始文件）、RuCytoToxDB（无 Zenodo 存款，921 篇 ⊂ MSRS 的 1,427 个 Ru DOI）。
唯一剩下的路径是人工整理 2023–2025 文献，**没有做，也没有编造任何行**。

## C10 — k-shot 校准

**能写**：「在 ligand-disjoint 切分上，用 k 条新金属测量做 offset / linear / isotonic
校准，k=50 时 R²≥0 的金属是 0/5；而且小 k 校准是**负作用**：k=5 时，
offset-only 校准在 55–95% 的重复里比不校准更差。机制与 C8 一致——
零样本的偏差本来就小（占平方误差 5–10%），用少量点去估它，估出来的噪声比要修的偏差还大。」

**不能写**：「给新金属测 20 个化合物就能用」。

---

## UNMEASURED（完全没有数字）

| 主张 | 为什么没有 | 需要什么 |
|---|---|---|
| 湿实验验证 | 硬性不做（§5.4） | — |

（「外部数据集上的绝对精度」与「细胞系特征带来的增益」已从 UNMEASURED 移入
BLOCKED：2026-10-21 的 EXT-5 / EXT-7 给出了实测原因，见下。）

## BLOCKED（有依赖挡着）

| 主张 | 阻断原因 | 证据 |
|---|---|---|
| 外部数据集上的绝对精度 | **不存在独立外部 cohort**：IrCytoToxDB 与 MSRS 的 DOI 重叠 99.55%、89.4% 数值相同；其余来源逐一实测不可用 | `msrs_external_cohort_probe.json:/conclusion` |
| 细胞系特征带来的增益 | **双重阻断**：0 个注释文件；且 21.9% 的行来自 CCLE/DepMap 里按构造不存在的系（HELA / HEK293 / A2780 / A2780CISR / A549CISR） | `msrs_cell_feature_probe.json:/identity_mismatch` |
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
