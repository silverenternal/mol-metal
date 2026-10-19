# 强指标论文的指标缺口分析

日期：2026-10-19
范围：`metrics/by_metric/*.json`（342 个产物）+ `docs/00_MODEL_NARRATIVE.md`
规则：每个"当前值"都是 MEASURED 并附产物路径；每个"目标值"都是 PROPOSED，不是测量值。

---

## 0. 一句话结论

我们已经**有 5 个可以直接当头条的强指标**（生成、对接、跨金属迁移、光毒性、表示增益），
但**核心的金属替换预测线只有中等强度，且缺少同协议外部锚点**；决策层与不确定性层是明确的弱项。
要发强指标论文，不需要再造模型，需要做三件事：

1. 把强项补齐到"5 seed + 多协议 + 外部基线"的可发表标准；
2. 把弱项如实降级为边界，不要包装成卖点；
3. 补上现在完全没有的 5 类指标（外部数据、决策曲线、金属消融、统计功效、数据卡）。

**2026-10-20 强化阶段进展（MEASURED）**：STR-1..STR-4 已跑完。跨论文 DOI R²
从 0.3321 抬到 **0.3767**（MACCS+RDKit，5 seed，配对 Δ +0.0387 [+0.0241,+0.0535]；
置换控制 0.3275 证明是内容而非宽度），与 GNN 融合后 **0.3811**——**R² 门槛 0.36
已过**。但 Δ 符号停在 **0.5826**（融合 0.5505），离 0.65 的门槛仍有 0.067。删失
训练（STR-3）8/8 组合为负，Δ 直接回归（STR-5）0.5183 且无门槛只有 0.4988。也
就是说：**"预测得准"已经成立，"排序得对"没有**，两者在本语料上是分离的，而且
三条排序路线（相减 / 融合 / 直接回归）都关掉了。

---

## 1. "强指标"的判据（PROPOSED）

一条指标能进主表，需要满足 ≥4/5：

| 编号 | 判据 | 说明 |
|---|---|---|
| S1 | 同协议外部基线 | 同数据、同划分、同指标；不能用"别人论文里的不同任务数字" |
| S2 | 5 seed 均值±标准差 | 单 seed 的 0.58 不是结果，是抽样 |
| S3 | ≥2 种互不重叠的划分 | 至少 ligand-grouped + DOI/scaffold/temporal 之一 |
| S4 | 效应量 + 置信区间 + 多重校正 | 配对检验，DOI 聚类，Holm |
| S5 | 可追溯 | 每个数字能追到一个 artifact 的字段 |

现有指标里，满足 S1 的只有跨金属迁移线（Krasnov）和 MoleculeNet；满足 S2 的只有 MoleculeNet、
跨金属迁移、NCI-60；MSRS 主线最多满足 S3，而且它的两个 split 都是配体划分、高度相关，
不构成真正的双协议证据（单 seed、无外部基线）。

---

## 2. 现有指标盘点

### 2.1 已经够强（可直接当头条）

| 指标 | 当前值 | 对照 | 出处 | 缺什么 |
|---|---|---|---|---|
| MoleculeNet 均值（D-MPNN, 5-seed） | **0.8149** | 内部 gate 0.80 | `moleculenet_auc_dmpnn_5seed.json` | 无（bace 单任务仍 −0.036） |
| 对接 RMSD（5 pockets × 3 seeds） | **0.6578 Å**，5/5 ≤2.0 Å | DiffDock 38.2% <2 Å | `docked_pose_rmsd.json` | 只有 5 个 pocket，且是 redocking |
| decode_ratio（real） | **0.998**（5979/5991） | TargetDiff ≥0.5 | `decode_ratio.json` | 不是化学指标，需与 validity 联报 |
| Ir within-metal AUC | **0.8295–0.8597** | Krasnov 0.73 | `metal_activity_transfer_seed_stability.json` | 只有 534 行，需外部复核 |
| Ru within-metal AUC | **0.8128–0.8363** | Krasnov 0.81（持平） | 同上 + `metal_activity_transfer_tie_sensitivity.json` | 平票规则敏感，只能写"持平" |
| 光毒性 switch 差（Ru−Ir） | **+0.1251** [0.0395, 0.2048] | 无外部基线 | `msrs_phototoxicity.json` | 描述性，PI 本身不可预测 |
| 全配合物表征增益（fragment） | **+0.1002 R²**（p=2.3e-20） | ligand_main_fragment 基线 | `msrs_complex_features.json` | 单 seed；canonical 上是 +0.1040 |

**结论：这 7 条里，生成/对接/迁移/光毒性四条足以支撑一篇强指标论文的半张表。**

### 2.2 中等（能写，但不能当头条）

| 指标 | 当前值 | 为什么不够强 | 出处 |
|---|---|---|---|
| MSRS R²（canonical） | **0.5828 ± 0.0102**（堆叠）/ 0.5555 ± 0.0026（树），5 seed | 已满足 S2/S4/S5；仍只有配体划分、无同协议外部基线 | `msrs_seed_stability.json` |
| MSRS R²（fragment） | **0.5137 ± 0.0048** / 0.4832 ± 0.0018，5 seed | 同上 | 同上 |
| Δ 符号准确率 | **0.6193** [0.5085, 0.7137]（DOI，阈值 0.2091） | 只比随机高 0.12；MAE(Δ) 0.51 vs 恒零 0.53；校准斜率 0.32、R² 0.004 | `msrs_delta_metrics.json` |
| top-1 准确率 | 0.5660 | 随机 0.50；424 队列只有 61 DOI | `msrs_hybrid_decision.json` |
| 决策 regret | 0.1186 | 全局最优基线 0.1198，差异不显著；50% 覆盖下 0.1115 vs 0.1067（更差） | `msrs_hybrid_decision.json` + `msrs_decision_curve.json` |
| 决策效用曲线 | 50% 覆盖 regret 0.1115，Holm p=1.0 | 弃权不买准确率；置信度 margin 仅 0.11 log10 | `msrs_decision_curve.json` |
| LOMO 跨金属 AUC | 0.5575 / 0.5845 | 只有 3 个金属对有数据；无外部基线 | `metal_lomo_auc_wasserstein.json` |
| NCI-60 9 细胞系 AUC | 0.7294 ± 0.2137 | Krasnov 0.81（−0.08）；2 个细胞系 n<10 | `nci60_precious_9cell_auc.json` |
| Krasnov 对齐 chemprop | 0.7812 | 3 金属，9/9 run；仍是内部复现 | `wf219_...scaffold.json` |

### 2.3 弱（必须降级或补强）

| 指标 | 当前值 | 问题 | 出处 |
|---|---|---|---|
| 决策 regret vs global_best | 0.1186 vs 0.1198 | 差 0.0012，Holm p=1.0；MET-5 曲线同样 p=1.0 | `msrs_hybrid_decision.json` + `msrs_decision_curve.json` |
| 时间划分 Δ 符号 | **0.4786** | 低于随机；说明跨年份方向不可学 | `msrs_external_validation.json` |
| conformal 区间宽度 | 中位 **2.0052** log10（DOI，global） | 约 100 倍 IC50 跨度；coverage 0.9004 是 conformal 构造保证，不是能力 | `msrs_calibration.json` |
| 学出来的难度函数（difficulty） | coverage **0.8789**、Winkler **2.8587** | 比 global 更差：宽度只降 4.3%，coverage 掉 2.2 点，PIT KS 涨 36 倍 | 同上 |
| 同队列多金属的条件覆盖 | **0.8096**（n=851） | 交换性被队列结构打破；单金属队列 0.9050 | 同上 |
| 光毒性 PI 预测 | OOF R² < 0 | 只能描述，不能预测 | `msrs_surface_model.json` |
| Vina（贵金属） | BLOCKED | 力场缺贵金属参数 | `vina_status.json` |
| 生成 SA | 生成物 **6.4423**（目标 ≤4.5）；4.71 是 metallo_drugs 训练队列 | 生成物比真实语料更难合成；QED 0.0944 | `cfm_generation_metrics.json` |
| 生成 novelty | max T <0.9 比例 **1.000**，平均 novelty **0.8517** | 口径依赖；高 novelty 部分来自「结构奇怪」 | 同上 |
| 生成对接 | **BLOCKED**（1/10 pocket，run died） | 无法计算成功率 | `cfm_vina_generated.json` |

### 2.4 缺失（完全没有 artifact）

| 缺的指标 | 为什么必须有 | 现状 |
|---|---|---|
| ~~强模型的 DOI/scaffold/year 泛化~~ | ~~配体分组会泄漏论文内一致性~~ | **已补（MET-2）：DOI 0.3321 / scaffold 0.4503 / year 0.3739** |
| ~~强模型 5-seed 稳定性~~ | ~~0.5807 是单 seed~~ | **已补：0.5828 ± 0.0102，MET-1** |
| ~~Δ 的 MAE / 校准斜率 / 分层~~ | ~~符号准确率不能说明幅度~~ | **已补（MET-3）：MAE(Δ) 0.5089、斜率 0.32、R² 0.0039，负** |
| ~~回归校准（斜率、区间分数）~~ | ~~coverage 好不代表 sharpness 好~~ | **已补（MET-4）：DOI 斜率 0.9256、r2_fit 0.3363、Winkler 2.74，负** |
| ~~决策曲线（regret-coverage、net benefit）~~ | ~~单点 regret 不能说明可用性~~ | **已补（MET-5）：50% 覆盖 regret 0.1115 vs 0.1067，Holm p=1.0，负** |
| ~~外部独立数据集~~ | ~~所有验证都是同一语料的重划分~~ | **已测（MET-7）但 BLOCKED**：三键交集 **0**（金属 Pt vs 五金属不重叠）；配体 144 / 细胞系 3 有重叠，909 个 Pt 配合物、2,727 个标签只能做「未见金属外推」 |
| ~~最终模型的 metal-blind 消融~~ | ~~证明"金属感知"承重，而不只是相关~~ | **已补（MET-6）：metal-blind −0.0683（ligand）/ −0.0693（DOI），p=0.0099；金属标签本身只值 0.006** |
| ~~统计功效 / 最小可检测效应~~ | ~~61 DOI 下 p=0.054 到底意味着什么~~ | **已补（MET-8）：regret MDE 0.050（62 DOI），符号 MDE 0.130（53 DOI），MAE MDE 0.077** |
| ~~生成 novelty / uniqueness~~ | ~~validity 高不等于分子新~~ | **已补（MET-9）：max T<0.9 比例 1.000、平均 novelty 0.8517、IntDiv1 0.8526（同 n 训练 0.8234）；但 SA 6.4423 未达标、对接 BLOCKED，整体负** |
| ~~数据卡 + 泄漏审计~~ | ~~强指标论文的标配~~ | **已补（MET-10）：`DATASET_CARD.md` + `DATA_LEAKAGE_AUDIT.md` §8；ligand 划分 89.43% 测试行共享 DOI** |

---

## 3. 目标值（PROPOSED）

这些是"投二区强指标论文"的门槛，不是当前测量值。定得太高会逼出 p-hacking，所以每条都同时给
"达不到时怎么写"。

| 指标 | 当前 | 目标 | 达不到时的写法 |
|---|---|---|---|
| canonical R² | **0.5828 ± 0.0102** | ≥0.60，5-seed std ≤0.02（std 已达标） | 报 0.5828 ± 0.0102，强调 +0.1002 的表示增益 |
| DOI/scaffold R² | **DOI 0.3321 / scaffold 0.4503 / year 0.3739** | ≥0.50，相对 random 下降 ≤0.10 | 未达标：报下降幅度，全部主张收缩到内插 |
| Δ 符号（0.2091） | **0.6193 [0.5085, 0.7137]** | ≥0.65，CI 下界 >0.55 | 未达标：报 0.62 + CI，写明仅略高于随机 |
| Δ MAE | **0.5089**（恒零预测器 0.5299） | ≤0.30 log10 | 未达标：模型只比恒零好 4.0%，幅度不可预测 |
| 决策 regret | 0.1186 | **≤0.10**，Holm p<0.05，覆盖 ≥50% | 未达标：50% 覆盖 0.1115，Holm p=1.0；降级为边界，不作为卖点 |
| top-1 | 0.5660 | **≥0.65** | 同上 |
| conformal 宽度 | **2.0052**（DOI global）/ 1.9195（difficulty） | **中位 ≤1.0 log10** | 未达标：报 coverage 0.90 + 宽度 2.01，写明区间只能当限制 |
| LOMO AUC | 0.5575 | **≥0.65** | 报"未见金属不可外推" |
| 外部数据集 | **三键交集 0**（配体 144 / 细胞系 3 有重叠） | ≥200 行，R²≥0.5 或 AUC≥0.75 | 标 BLOCKED-PENDING-DATA：金属不重叠；909 个 Pt 配合物是另一个估计量 |
| 生成 novelty | **平均 0.8517**（max T<0.9 比例 1.000） | **≥0.90**（1−max Tanimoto to train） | 未达标：报双口径 + IntDiv1 0.8526，写明高 novelty 部分来自结构异常（max T 仅 0.11–0.19） |
| 对接成功率 | 5/5 ≤2 Å | ≥50% on ≥20 pockets | 报 5 个 pocket 的现状 |

**每个 null 的 MDE**（MET-8，`msrs_power.json`）：决策 regret 0.050（62 DOI）、
Δ 符号 0.130（53 DOI）、Δ MAE 0.077（53 DOI）、DOI R² 0.064（1,784 DOI）。
判读规则：观测效应 < MDE 时写「低于可检测下限」，不要写「不显著」。

---

## 4. 论文主张 → 必需指标

| 主张 | 必需指标 | 现状 |
|---|---|---|
| C1 全配合物表征优于配体片段 | R² 增量 + 配对 CI + 5 seed + 两 split | 有 +0.1002（单 seed）；5 seed 只在最终模型上补了（MET-1），表征臂本身仍单 seed |
| C2 金属状态/配位是承重的 | metal-blind 消融 + permutation null | **已补（MET-6）**：metal-blind −0.0683 [0.0516, 0.0903]，DOI −0.0693 [0.0509, 0.0884]，block p=0.0099；但 cell-blind 同为 −0.0629 |
| C3 跨金属迁移有代价但可量化 | within-metal vs LOMO + seed spread + 平票敏感性 | **齐全** |
| C4 光毒性可切换且金属依赖 | switch 率 + Wilson CI + 跨金属配对 | **齐全**（描述性） |
| C5 金属替换推荐可用 | regret/top-1/NDCG + 决策曲线 + 显著性 | **不成立**（n.s.） |
| C6 生成+对接可用 | validity/decode/novelty/SA/docking | **已补（MET-9）但为负**：novelty 平均 0.8517、SA 6.4423（>4.5）、对接 BLOCKED；C6 只能作为限制陈述 |
| C7 结论跨论文成立 | DOI/scaffold/year 三协议 | **已测（MET-2），结论为否**：DOI 掉 0.2235、year Δ 符号 0.4286 |

**取舍建议：** 主表放 C1/C3/C4/C6，C2 作为机制支撑，C5 放"边界与负结果"，C7 作为泛化表。
这样论文的强项全部有 5-seed/外部基线，弱项不伪装成强项。

---

## 5. 补齐计划

见 `todo.json :: work_packages` 的 MET-1..MET-11。优先级：

- **P0（不做就不能投）**：MET-1 5-seed（DONE）、MET-2 四协议（DONE，负）、MET-3 Δ 指标（DONE，负）、MET-4 校准（DONE，负）、MET-6 金属消融（DONE，PASS）、MET-10 数据卡（DONE）。**P0 全部完成。**
- **P1（决定论文上限）**：MET-5 决策曲线（DONE，负）、MET-7 外部数据（DONE，BLOCKED：三键交集 0，金属不重叠）、MET-8 统计功效（DONE）。
- **P2（生成线专用）**：MET-9 生成指标（DONE，负：SA 6.4423 > 4.5、平均 novelty 0.8517 < 0.90、对接 BLOCKED）。
- **P3（打包）**：MET-11 指标表自动生成（DONE：10 张表 / 106 行 / 0 MISSING，`metrics/paper_tables/`）。

**强化阶段（2026-10-19 起）**：测量已全部完成，计划改为 `todo.json :: work_packages`
的 STR-1..STR-7。判据写死在两处：DOI R² ≥0.36（配对 CI 下界 >0）与 Δ 符号 ≥0.65
（DOI 聚类 CI 下界 >0.55）。任一失败即按边界论文定稿，不再投入 STR-5/STR-6。

**STR-1（5 seed，DONE / STABLE）**：seed 平均预测把 R² 抬到 ligand 0.5596 /
DOI **0.3380** / scaffold 0.4530 / year 0.3780（seed SD 0.0026–0.0066）；Δ 符号
ligand 0.6101 / DOI **0.5826** / scaffold 0.6330 / year 0.4082（seed SD
0.0179–0.0387，max 0.0387 < 0.05）。**已发布的单 seed DOI 符号 0.6193 高于全部
五个 seed（0.5183–0.5963）**，是偏高抽样，主表必须改用 0.5826。决策 regret
0.1279（单 seed 0.1168），p_holm 仍 1.0，MET-5 的 null 没有翻转。

**STR-2（四协议融合，DONE / PARTIAL）**：5 树 + 5 GNN 集成后 DOI stack R²
**0.3597**（gate 0.36，**差 0.0003**），equal 臂 0.3606；配对 ΔR²
**+0.0217 [+0.0044, +0.0377]** 排除 0。ligand 0.5997（Δ+0.0401）、scaffold
0.4839（Δ+0.0310）、year 0.3968（CI 含 0）。**单模型 stack 只有 DOI
0.3485 ± 0.0046**——集成贡献了约一半增益。融合把 DOI Δ 符号从 0.5826 压到
**0.5413**，R² 与排序方向相反，推荐线不能改用融合模型。

**STR-3（删失训练，DONE / KILL）**：把 4,918 行右删失（bound）按 `tobit` / `hinge`
两种目标加进训练，评估仍在同 21,404 行。**8/8 协议×目标组合为负，8/8 CI 排除 0**：
DOI tobit 0.2400（−0.0933 [−0.1244,−0.0635]）、hinge 0.2828（−0.0504
[−0.0733,−0.0276]）；ligand 0.4962 / 0.5326（树 0.5554）。**删失 bound 编码论文的
剂量设计，不是随机缺失**——加入后把预测面整体上拉。条件均值目标下继续丢弃删失行
是正确的。

**STR-4（特征扩展，DONE / R² 过线、符号未过）**：三臂单 seed 筛选——ECFP6(4096)
DOI 0.3321（KILL，−0.0012）、`assay`（time_h + is_cancer_line）0.3410（+0.0078，
CI 含 0）、**MACCS(167) + 20 RDKit 描述符 DOI 0.3706（PASS）**。5 seed 复现：
DOI **0.3767**（seed mean 0.3697 ± 0.0022），配对 Δ **+0.0387 [+0.0241, +0.0535]**；
ligand 0.5593（−0.0003）、scaffold 0.4733（+0.0204 [+0.0077,+0.0337]）、year
0.3946（+0.0166，CI 含 0）。**置换控制**（新增块整块按行打乱，5 seed）**0.3275 ±
0.0023**，低于原树 0.3332 → 增益来自描述符内容，不是「多 187 列」的宽度效应。与
GNN 融合后 DOI **0.3811**（+0.0431 [+0.0260,+0.0604]）。**但 DOI Δ 符号仍是
0.5826（融合后 0.5505）< 0.65**——R² 与排序分离，这是本阶段最硬的负结果。语料无
`method` 列，已记录而非静默补齐。

**STR-5（Δ 直接回归，DONE / KILL）**：复用 FIX-M4 的 `pair_full` 臂（k=2 时行势
模型与逐对回归是同一对象），只加逐对落盘，按 STR-5 的门槛重算。DOI 留出 427 对 /
门槛内 218 对：符号 **0.5183 [0.3902, 0.6539]**、MAE(Δ) **0.5559**（恒零预测器
0.5299）、bias −0.173、校准斜率 0.166；**无门槛 427 对符号 0.4988 ≈ 随机**。同一批
218 对上的水平模型：0.5826 [0.4670, 0.6805]、MAE 0.5115。**两条 kill 线同时触发**
（<0.62 与 ≥0.50）。区间与水平模型重叠，所以不主张「直接回归显著更差」，只主张
它自己没过门槛。**排序线到此关闭：相减 0.5826 / 融合 0.5505 / 直接回归 0.5183。**

**STR-6（SA 引导解码，DONE / KILL）**：计划中的 3×200 A/B **无法执行**——三处
独立阻断：(a) `--sa-penalty` 零梯度（sascorer 无梯度、代理被
`torch.tensor(float(...))` 包住，开=关逐位相同）；(b) 无可微 SA 代理（2026-10-04
删除）；(c) 每 run 只有 `seed0` checkpoint。600 样本补测也失败：该 checkpoint 的
权重含 `time_embed_proj` / `pocket_per_layer_injector`，而它自己的 `args` 不记录
这两个开关，**无法从其记录的配置重建**。改用产量审计（2,396 个无引导分子）：
**SA≤4.5 只有 16 个 = 0.67% [0.41%, 1.08%]**（panel of record 0/66，最小值
4.751）；门内 16 个的 novelty **0.8757 < 0.90**。**两条门槛同时不过**，且
"SA≤4.5" 只能作为 0.67% 手挑子集的性质来写，不能作为生成器的性质。

**STR-7（打包，DONE / PASS）**：`paper_metric_tables.py --strict` 生成 **12 张表、
0 MISSING**；新增 T11（5-seed 三臂泛化矩阵：ECFP4 树 0.3380 / MACCS+RDKit 0.3767 /
融合 0.3811）与 T12（排序三路线同一 0.65 门槛：0.5826 / 0.5505 / 0.5183）。
`docs/CLAIM_LEDGER.md` 把 C1–C7 逐条绑定到 artifact 指针（11 条，由
`test_claim_ledger.py` 解析验证）。**C6 被更正**：LOMO 0.5575±0.0949 低于 0.7 门槛，
「跨金属迁移」不可写；可写的是 within-metal（Ru 0.8271 / Ir 0.8430）。

**S5 现在是机械的。** `molmetal/scripts/paper_metric_tables.py` 从
`metrics/by_metric/*.json` 直接生成 T1–T9，每个单元格带 artifact pointer，
解析失败即 `MISSING`（`--strict` 退出码 1）。它保证「表 == 产物」，不保证
「产物是新的」——manifest 只记 sha256，重跑 artifact 后必须重跑脚本。

---

## 6. 不要做的事

1. **不要把 DOI 当特征。** `feature_headroom_audit.json` 实测：DOI offset 带来 +0.1109 R²，
   但金属化学在 DOI 之后只有 **−0.0003**。DOI 是论文代理，不是可部署信号。
2. **不要继续在 61 个 DOI 上调决策层。** FIX-M4 已证明配体分组的 +0.02 在整篇论文留出后归零。
3. **不要把配体分组的 R² 0.5596 写成"跨论文"性能。** 跨论文（DOI）是 0.3380。
4. **不要把 NCI-60 的 0.7294 写成"接近 Krasnov 0.81"。** 差 0.08，且 2 个细胞系 n<10。
5. **不要单列 bond_f1 或 decode_ratio。** 前者会塌向全正，后者不是化学指标。
6. **不要把 SA≤4.5 写成生成器的性质。** 它只对 0.67% 的手挑子集成立（16/2,396），
   而且那个子集的 novelty 0.8757 还没到 0.90。报 SA 必须同时报产量。
7. **不要把 `--sa-penalty` 写成训练杠杆。** 它零梯度，开与关的参数更新逐位相同
   （`sa_penalty.py` 自己的 docstring 就写着）。
