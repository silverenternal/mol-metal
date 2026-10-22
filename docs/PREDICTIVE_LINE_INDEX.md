# PREDICTIVE LINE — 单一索引（single source of truth）

**建立**：2026-10-21（用户指令「把预测线的文档整理一下，不要让项目那么乱」）
**作用**：预测线（MSRS 跨金属 pIC50 / Krasnov / chemprop / D-MPNN）每个数字只有一个权威出处。
**规则**：本文档里的每个数字都由 `molmetal/tests/test_predictive_line_index.py` 对着 artifact 校验；
改了 artifact 不改这里 → 测试红。**引用前先看 §3 撤回日志**——预测线最容易犯的错是引用一个已经撤回的数。
**不覆盖**：生成线见 `cfm_todo.json`；论文主张台账见 `docs/CLAIM_LEDGER.md`。

---

## §1 一句话结论

- **跨论文绝对强度预测过关**：整篇 DOI 留出 R² **0.3767**（5 seed，MACCS+RDKit 树）/ **0.3811**（+GNN 融合），门槛 0.36。
- **"换哪个金属更好"点估计不可学、balanced 口径可学**：门内 218 对的多数类率 **0.6009**，三条点估计路线（相减 0.5826 / 融合 0.5505 / 直接回归 0.5183）**全部低于恒定预测**；但同一批折改 listwise 目标后 balanced **0.5985 [0.5262, 0.6663]**，首次显著优于 0.5（raw 仍只有 0.5417）。**这是"方向学到了、精度没到"，不是"可用了"。** 详见 §2.7。
- **跨金属迁移不到可用水平**：可入闸的 Ru/Ir LOMO **0.5974** < 0.6；within-metal 稳定超过 Krasnov（Ru 0.8271 vs 0.81，Ir 0.8430 vs 0.73）。
- **加性约束不是 metal_out 的瓶颈**（UP-1，2026-10-21）：放开 ligand×metal 交互三个臂全部**更差**（free −0.097 / pair −0.113 / product −0.164 R²），最好的 free_onehot 只 +0.0204 且 CI 含 0。**方差分解里 0.712 的交互份额，在留出金属上不可外推。**
- **校准的失败层是 cohort size，不是金属**（UP-6）：`cohort_metal` Mondrian 把最差层 0.8096 → **0.8956**，中位宽只涨 0.65%（2.0052 → 2.0183）。

按 `todo.json::north_star.decision_rule`（门槛在看到结果前写死）：4 条里过 2 条 → **边界论文**。
UP 轮之后仍是 2/4：UP-6 是**已过项的内部加固**，其余八项没有一个过门。

- **"shape 学不会"的真实原因是支持度，不是模型**（UP-10，2026-10-21）：10 个金属对里 **7 个的配对组数是 0**，只有 Ru–Os（354 组）与 Ir–Rh（170 组）过 20 组门槛；跨 ≥2 金属的配体只占 **3.64%**。而且这两个"有支持"的对，典型替换效应**不大于自身化验重复散布**（可分性 0.592 / **0.115**）。
- **交互无法外推到未观测金属对**（UP-11）：201 个跨金属配体**没有一个跨 ≥3 个金属**，三个金属对的配体集合**两两不相交**，所以 Δ 矩阵的完整行是 0 行 —— 秩-1 分解**不可识别**，不是份额小。退一步做配体不相交的跨对迁移，两个方向的 R²_Δ 全为负（−0.0952 / −0.2256），符号准确率 0.49 / 0.53，CI 全含 0.5。
- **配对是论文内现象，整篇留出把它整体拿走**（UP-14）：语料里 5.47% 的行属于跨金属单元，但 **DOI 留出下只有 0.57%（120/21,045）的测试行在训练折里有配对证据**。配对的两行几乎总在同一篇论文里，于是同折。冻结的 kill 规则（<5%）在 DOI 协议上触发。
- **交互在金属被观测到时是真实收益，在留出金属上不存在**（UP-12）：配体分组下自由交互比加性高 **+0.0391**（0.2519 vs 0.2128），同一模型在 metal_out 上是 −0.097；秩-1 支持受限的交互在配体分组下反而 **−0.0174**。metal_out 的两阶段预测与 stage1 **逐位相同**（`max|Δ| = 0`）—— 该协议结构上看不见这个杠杆。
- **0.5985 只在合并后显著，没有任何一个金属对单独过半**（UP-16）：Os–Ru 0.5807 [0.4990, 0.6548]、Ir–Rh 0.5600 [0.4799, 0.6190]、Rh–Ru 0.5000（n=16、1 个负例）。且 360 对 100% 落在支持组内是**定义使然**（排序对本身就是同组两金属两行），不是测出来的覆盖率。
- **金属图是一棵树，而且「有效边」与「精确边」不是同一个集合**（UP-18）：观测子图 4 节点 3 边、无环，Re 是孤立点；树上 node-splitting 无定义、残差自由度 0。按效应 {Ru–Os +0.134 (t=3.36), Ru–Rh +1.077 (t=2.58)}，按精度 {Ru–Os 0.59, Ir–Rh 0.40} —— 只在 Ru–Os 上重合。UP-10 判 OK 的 Ir–Rh 均值与 0 不可区分（t=0.96）。
- **metal_out 的负 R² 是行构成，不是模型缺陷**（UP-19）：**94.5% 的测试行没有任何 (配体, 细胞系) 伙伴在训练金属下**，这部分 R² = **−0.0913**；有伙伴的 **5.47%（1,151 行）上 R² = 0.3299**、去中心化 0.3806，高于同模型 within_metal 的 0.2128。headline −0.0734 是这两块按 SSE/SST 加权的结果。
- **shape 里结构与论文批次各自都超过 50%**（UP-17）：配体×金属 0.8015 [0.7884, 0.8151]、论文×金属 0.5484 [0.5801, 0.6297]，**两块在自己的优先顺序下都过 0.5、在对方的顺序下都归 0**；细胞×金属两种顺序都是 0.0。冻结的 kill 规则触发，但它的推论（"建模即拟合噪声"）**不成立**。
- **CQR 修好了覆盖、没修好宽度**（UP-20）：覆盖 **0.9000**、最差金属组 **0.8716** 双达标，但中位宽度 **1.9871** 只比 global（2.0073）窄 **1.0%**，离 1.80 差 10.4%。difficulty 的 1.8981 是拿 **0.8767** 覆盖换的。kill 触发 ⇒ DP-7 放弃宽度目标。
- **论文偏移真实但不可从论文描述预测**（UP-21）：留出论文偏移 sd **0.2400**、平均 |偏移| **0.2255**，但 LOPO ridge **R² = −0.0259**，校正后 DOI R² 从 0.3670 掉到 **0.3557**、偏移 MAE 反而涨 4.55%。允许用测试论文自己的 y 统计量则跳到 **0.5405**——**信息在标签里，不在描述里**。kill 触发 ⇒ DOI 线关闭。
- **UP-8 的 1/5 是采样粒度，不是缺配对点**（UP-22）：换成行级采样后 n_paired=3 有 **3/5** 金属 R² > 0（Rh +0.0631 / Os +0.2830 / Re +0.1745），但 n_paired 从 0 增到 3 的平均 R² 是 **0.0407 → 0.0306**，不单调。**Re 有 0 个配对行却拿到 +0.1734**，增益在定义上不来自配对。gate FAIL（secondary），kill 未触发。
- **噪声标签不是排序的瓶颈**（UP-23）：deadzone（|Δ| < 该对 noise SD）在 651 对里保留 **315** 对；把噪声对并成同 label 后 lambdarank **0.5985 → 0.5612**（保留子集上 0.6050 → 0.5615），因为 **279/1,586 个组（17.6%）被整组并成一个 label**、梯度归零。kill 触发（0.6050 < 0.62）⇒ **DP-8 = yes，排序线关闭**。
- **换 pairwise 损失与边感知负采样只值 +0.008 AUC**（UP-24）：同学习器同划分，BCE **0.5449** → pairwise_edge **0.5529**（Ru +0.0271 / Ir −0.0112）；五金属均值反而从 0.6185 掉到 0.5352。**0.5974 与 0.63 的差距不是损失或负采样造成的**；该项的绝对门槛按 D-MPNN 冻结、学习器却是特征模型，属计划口径错误，kill 文本触发但未适用。
- **43 个 headline 数字里 5 个不可识别、17 个只部分可识别**（UP-25）：主表按 ESTIMAND（pooled 18 / per_pair 8 / conditional 7 / edge_level 7 / centered 3）与 IDENTIFIABLE（yes 21 / partial 17 / no 5）重排，值全部从 artifact 读出并与本索引的机器校验块逐个核对（43/43 一致）。**"partial" 是最容易误用的一档**：0.5985 / 0.5974 / 0.5807 都只在有配对证据的子群上成立。
- **论文偏移不需要预测，只需要测**（UP-26）：给新论文 **1 行**已测值，DOI R² 从 **0.3674 → 0.5559**；给 **10 行**到 **0.6839**（同批行不校正为 0.4005，校正贡献 **+0.2834**）；给 50 行到 0.7243。完美去除论文偏移的上界是 0.7337（独立算术核对），所以 10 行已经拿到这个上界的 89%。**前三轮全部在"零校准行"下评估，等于自愿放弃了部署时真实存在的信息。**
- **破坏可交换性的是论文随机效应，不是协变量漂移**（UP-27）：用 23 维论文元特征训练"测试论文 vs 校准论文"分类器，**AUC 只有 0.5678**、加权后有效样本量仍是校准集的 **94.5%** —— 两个总体在 x 上几乎不可分。于是加权 conformal 只把宽度从 1.9871 挪到 **1.9849**（−0.1%），按年覆盖漂移曲线每年变动 ≤0.003。ACI 能买到覆盖（**0.9106**），代价是宽度 **2.0572**。**能被 reweight 的偏移才叫协变量漂移；这个偏移不在 x 里。**
- **平局作为第三类结果也救不了排序**（UP-29）：Davidson 模型（δ 由数据估到 **2.28**，五个折 1.86–2.75）在保留子集上 balanced **0.5312**，与 Bradley-Terry 的 **0.5317** 无差别，低于 lambdarank 的 0.5985，kill 触发。模型自己给出的诊断是：δ 拟合到边际平局率后，**98.7% 的配对被判为平局** —— 分数分辨率低于噪声下限，平局类吸收了全部概率。**UP-23 的"噪声标签不是瓶颈"换一个机制重做，结论不变。**

**UP 第四轮记分牌（部署阶段，门槛在 2026-10-22T15:45+08:00 写死）**

| 项 | 问题 | 判决 | 一个数字 |
|---|---|---|---|
| UP-26 | 新论文的偏移能不能用少量已测行估出来 | **PASS**（4/4） | n_cal=10 R² **0.6839** ≥0.45；n_cal=0 复现 **0.3674**（目标 0.3670±0.005）；7 点全单调 |
| UP-27 | 加权 conformal + ACI 能不能同时拿覆盖与宽度 | **FAIL**（宽度、最差组） | 加权只把宽度挪 −0.0022；ACI 覆盖 0.9106 但宽度 2.0572；密度比 AUC 0.5678 |
| UP-29 | 把平局当第三类结果能不能救排序 | **FAIL**（kill 触发） | Davidson **0.5312** vs BT 0.5317 vs lambdarank 0.5985；δ=2.28；98.7% 判平局 |

**UP 强化轮记分牌（第一轮 9 项，门槛在 2026-10-21T15:10+08:00 写死）**

| 项 | 问题 | 判决 | 一个数字 |
|---|---|---|---|
| UP-1 | 加性约束是不是 shape 的瓶颈 | FAIL | free −0.097 / pair −0.113 / product −0.164 R² |
| UP-2 | 排序能不能靠扩样本 + 换目标救 | FAIL（但见下） | balanced 0.4705 → **0.5985**，raw 0.5417 < 0.6222 |
| UP-3 | 3D 配位几何是不是缺的输入 | FAIL | 语料里金属原子 0/200；补给体 ΔR² −0.005 |
| UP-4 | eRAC 的 Z_eff 能不能救水平项 | FAIL | 四个 cell 全负；符号检验 p=0.50 |
| UP-5 | Pt 预训练能不能补样本 | FAIL | +0.0095 vs 门槛 +0.03 |
| UP-6 | 条件覆盖能不能修 | **PASS** | 最差层 0.8096 → **0.8956**，宽度 +0.65% |
| UP-7 | 辅助终点能不能当正则 | FAIL | 干净臂 +0.0132；过门的臂泄漏率 1.000 |
| UP-8 | k 点适配能不能救少样本 | FAIL | k=50 可用金属仍 1/5（Os 0.348 → 0.196；Ru −0.078 → −0.476） |
| UP-9 | 第三个基学习器能不能加性 | FAIL | stack3 − stack2 = −0.0009 |

**唯一的真增益仍是 UP-2b 的 balanced 0.5985**，而它在 raw 口径上仍输给恒定预测。

**UP 第二轮记分牌（支持度阶段，门槛在 2026-10-21T20:30+08:00 写死）**

| 项 | 问题 | 判决 | 一个数字 |
|---|---|---|---|
| UP-10 | 配对支持到底有多少 | **PASS**（审计无阈值） | 10 个金属对里 **7 个组数为 0**，只有 Ru–Os 354 / Ir–Rh 170 |
| UP-17 | shape 是不是论文批次 | **FAIL**（gate 0.0716 > 0.05） | 配体×金属 0.8015 vs 论文×金属 0.5484，两块各自在自己顺序下都 >0.5 |
| UP-11 | 交互能不能外推到未观测金属对 | **NOT_IDENTIFIABLE** | 201 个跨金属配体**全部只跨 2 个金属**；三对两两共享配体 = 0 |
| UP-12 | 两阶段能不能只在配对数据上开交互 | **FAIL**（gate 退化） | metal_out 上交互**恒为 0**；within-metal 秩-1 −0.0174、自由交互 +0.0391 |
| UP-14 | 支持度分层后 headline 还剩什么 | **PASS**（无阈值） | DOI 下只有 **0.57%** 的测试行在训练折里有配对证据 |
| UP-16 | 0.5985 是不是只在支持组内 | **PASS**（无阈值） | 360 对 100% 支持（定义使然）；**没有一个金属对单独过半** |

**UP 第三轮记分牌（可识别性阶段，门槛在 2026-10-22T12:42+08:00 写死）**

| 项 | 问题 | 判决 | 一个数字 |
|---|---|---|---|
| UP-18 | 金属图是什么形状，哪条边携带水平信息 | **FAIL**（2/4 断言） | 观测子图是树；t（簇）Ru–Os 3.36 / Ru–Rh 2.58 / Ir–Rh 0.96；kill 未触发 |
| UP-19 | metal_out 的负 R² 是模型还是估计目标 | **PASS** | 可识别行仅 **5.47%**，其上 R² = **0.3299**（去中心化 0.3806）；其余 94.5% 上 −0.0913 |
| UP-20 | CQR 能不能同时拿宽度和覆盖 | **FAIL**（kill 触发） | 覆盖 **0.9000** ✓、最差组 0.8716 ✓，宽度 **1.9871** > 1.80（只比 global 窄 1.0%） |
| UP-21 | 论文元特征能不能预测论文偏移 | **FAIL**（kill 触发） | DOI R² **0.3557** < 0.40；偏移 LOPO R² **−0.0259**；泄漏臂 +0.1735 |
| UP-22 | UP-8 的 1/5 是不是因为缺配对点 | **FAIL**（secondary） | n_paired=3 时 3/5 金属 >0 ✓，但 0→3 平均 R² **0.0407 → 0.0306** 不单调；Re 有 0 个配对行却 +0.1734 |
| UP-23 | 噪声标签是不是排序上不去的主因 | **FAIL**（kill 触发） | deadzone 保留 315/651；lambdarank **0.5985 → 0.5612**（变差）；BT 臂全部 ≤0.5411 |
| UP-24 | 换 pairwise logistic + 边感知负采样能不能救 LOMO | **FAIL**（kill 文本触发但未适用） | 同学习器 BCE 0.5449 vs pairwise_edge 0.5529（**+0.0080**）；绝对门槛 0.63 不可达 |
| UP-25 | 三轮数字按可识别性重排后主表长什么样 | **PASS**（三条断言） | 43 行：yes 21 / partial 17 / **no 5**；43 个 id 与索引机器校验逐个一致 |

---

## §2 权威数字

### 2.1 四协议 × 三臂（5 seed）

| 协议 | ECFP4 树 | MACCS+RDKit 树 | +GNN 融合 |
|---|---:|---:|---:|
| ligand（**内插，不可当泛化报**） | 0.5596 | 0.5593 | 0.5935 |
| **doi（判据协议）** | 0.3380 | **0.3767** | **0.3811** |
| scaffold | 0.4530 | 0.4733 | 0.4903 |
| year | 0.3780 | 0.3946 | 0.4011 |

DOI 相对 ECFP4 的配对增益 **+0.0431 [+0.0260, +0.0604]**（DOI 聚类配对 bootstrap，2000 次）。
表：`metrics/paper_tables/T11_generalization_strong.md`

### 2.2 金属感知表示（C1）

| 臂 | R² | Δ vs full | p |
|---|---:|---:|---:|
| full（配合物+态+配位+金属码） | 0.5556 | — | — |
| metal_blind（去掉金属相关一切） | 0.4873 | **0.0683** | 0.0099 |
| no_state | 0.5284 | 0.0272 | <0.001 |
| no_coord | 0.5426 | 0.0130 | 0.029 |
| no_metal_code（只去身份码） | 0.5492 | 0.0064 | 0.005 |

全配合物表征 vs 配体单片段：**+0.1002 R²（p=2.3e-20）**。
**不能写"架构创新"**——增益来自输入里有金属。

### 2.3 排序（C3，DOI 留出，|Δ|≥0.2091，218 对）

| 路线 | raw sign | balanced | 95% CI (balanced) |
|---|---:|---:|---|
| 水平相减 | 0.5826 | 0.5272 | [0.4359, 0.6198] |
| 融合 | 0.5505 | — | — |
| Δ 直接回归 | 0.5183 | 0.5162 | [0.3939, 0.6430] |
| **恒定预测（多数类）** | **0.6009** | 0.5 | — |

门内只有 3/10 个金属对：Os–Ru 153、Ir–Rh 64、Rh–Ru 1。**五金属排序在本队列上不可识别。**

### 2.4 校准（C5）

DOI 留出、α=0.10 split conformal：覆盖 **0.9004**，中位宽 **2.0052** log10。
同队列多金属子群（n=851）覆盖只有 0.8096 —— 必须单独报。

### 2.5 跨金属迁移（C6）

| | within-metal（5 seed 均值） | Krasnov 2026 | LOMO |
|---|---:|---:|---:|
| Ru（n_unique 1,311） | **0.8271** | 0.81 | 可入闸 |
| Ir（n_unique 538） | **0.8430** | 0.73 | 可入闸 |
| Rh / Os / Re（51 / 36 / 74） | 0.6974 / 0.4960 / 0.9480 | — | 低于 `MIN_UNIQUE_FOR_GATE=200`，**不得入闸** |

可解析子集 LOMO：**0.5974**（seed 集成 0.5947），目标 0.6，两个 CI 都含 0.6。
**不能写"差 0.04"** —— 那是把三个不可解析的折等权平均进去。

**LOMO 有五个 artifact，只有一个能引：**

| artifact | value | 是什么 | 能不能引 |
|---|---:|---|---|
| `metal_lomo_auc.json` | 0.5346 ± 0.1096 | `train_metal_lomo.py`，5 折 × 3 seed 原始 m19 | 历史值，已被后续覆盖 |
| `metal_lomo_auc_v2.json` | `null`（`DEFERRED_PARTIAL_RUN`） | `train_metal_lomo_v2.py`，平衡 Rh 队列重测，**没跑完** | **不可引** |
| `metal_lomo_auc_wasserstein.json` | 0.5575 ± 0.0949 | `train_metal_lomo_wasserstein.py`；`real_only_fold_value` = 0.3812 | 只能引可解析子集口径 |
| `lomo_resolvability_audit.json` | **0.5974** | 把 Rh/Os/Re 三个低于 `MIN_UNIQUE_FOR_GATE=200` 的折剔除后重聚合 | **这是可引的那个** |
| `lomo_seed_ensemble.json` | 0.5947 | 同一可解析子集，但按「先平均分数再算 AUC」（部署口径） | 可引，作为 0.5974 的稳健性 |

**已知冲突（未修，需要裁定）——不是「pin 过时」，是两个不同的估计量：**

| 出处 | 数 | 估计量 |
|---|---:|---|
| `configs/locks/protocol_lock.yaml::m19_lomo` | — | 锁定协议写的是 `target_metric: mean test AUC over 5 metals`，`n_folds: 5` |
| `molmetal/scripts/registry.py::LINE_METRICS` → `metal_lomo_auc_wasserstein` | **0.5575** | 5 个金属**等权**平均（含 Rh 51 / Os 36 / Re 74 三个低于 `MIN_UNIQUE_FOR_GATE=200` 的折） |
| `docs/CLAIM_LEDGER.md`（STR-7 / FIX-C6）→ `lomo_resolvability_audit` | **0.5974** | 只对可入闸的 Ru(1,311) / Ir(538) 平均 |

所以 `registry.py cite` 说引 0.5575、台账说引 0.5974，两边都符合各自的定义。**以本索引和台账为准**
（0.5974），因为把三个测量地板上的折等权平均进去会低估。但要真正消除冲突，需要改
`protocol_lock.yaml` 的 `target_metric` 定义，而不是只改一个 pin——那会改变锁定协议，
本次没有动，留作待裁定项。

### 2.6 机器校验块

测试从下面这个代码块读取，逐条比对 artifact。**不要手改数值，改 artifact 后同步这里。**

```json predictive-line-index
[
  {"id": "G.ligand.ecfp4", "value": 0.5596, "artifact": "metrics/by_metric/msrs_seed_ensemble.json", "path": "protocols.ligand.ensemble.r2"},
  {"id": "G.doi.ecfp4", "value": 0.3380, "artifact": "metrics/by_metric/msrs_seed_ensemble.json", "path": "protocols.doi.ensemble.r2"},
  {"id": "G.doi.maccs", "value": 0.3767, "artifact": "metrics/by_metric/msrs_feature_expansion_5seed.json", "path": "protocols.doi.r2_seed_averaged"},
  {"id": "G.doi.fused", "value": 0.3811, "artifact": "metrics/by_metric/msrs_feature_fusion.json", "path": "protocols.doi.fused_r2"},
  {"id": "G.doi.fusion_delta", "value": 0.0431, "artifact": "metrics/by_metric/msrs_feature_fusion.json", "path": "protocols.doi.vs_tree.delta_r2"},
  {"id": "G.doi.fusion_ci_low", "value": 0.0260, "artifact": "metrics/by_metric/msrs_feature_fusion.json", "path": "protocols.doi.vs_tree.ci_low"},
  {"id": "G.scaffold.ecfp4", "value": 0.4530, "artifact": "metrics/by_metric/msrs_seed_ensemble.json", "path": "protocols.scaffold.ensemble.r2"},
  {"id": "G.year.ecfp4", "value": 0.3780, "artifact": "metrics/by_metric/msrs_seed_ensemble.json", "path": "protocols.year.ensemble.r2"},
  {"id": "C1.complex_delta", "value": 0.1002, "artifact": "metrics/by_metric/msrs_complex_features.json", "path": "splits.fragment_grouped.complex.delta_r2_vs_main"},
  {"id": "C1.full", "value": 0.5556, "artifact": "metrics/by_metric/msrs_metal_ablation.json", "path": "protocols.ligand.arms.full.r2"},
  {"id": "C1.metal_blind", "value": 0.4873, "artifact": "metrics/by_metric/msrs_metal_ablation.json", "path": "protocols.ligand.arms.metal_blind.r2"},
  {"id": "C1.metal_blind_delta", "value": 0.0683, "artifact": "metrics/by_metric/msrs_metal_ablation.json", "path": "protocols.ligand.arms.metal_blind.delta_r2_vs_full"},
  {"id": "C3.majority_class", "value": 0.6009, "artifact": "metrics/by_metric/msrs_delta_metric_audit.json", "path": "baselines.majority_class_rate"},
  {"id": "C3.diff_raw", "value": 0.5826, "artifact": "metrics/by_metric/msrs_delta_metric_audit.json", "path": "routes.level_differencing.raw_sign"},
  {"id": "C3.diff_balanced", "value": 0.5272, "artifact": "metrics/by_metric/msrs_delta_metric_audit.json", "path": "routes.level_differencing.balanced"},
  {"id": "C3.direct_raw", "value": 0.5183, "artifact": "metrics/by_metric/msrs_delta_metric_audit.json", "path": "routes.direct_regression.raw_sign"},
  {"id": "C3.direct_balanced", "value": 0.5162, "artifact": "metrics/by_metric/msrs_delta_metric_audit.json", "path": "routes.direct_regression.balanced"},
  {"id": "C5.coverage", "value": 0.9004, "artifact": "metrics/by_metric/msrs_calibration.json", "path": "protocols.doi.variants.global.coverage"},
  {"id": "C5.median_width", "value": 2.0052, "artifact": "metrics/by_metric/msrs_calibration.json", "path": "protocols.doi.variants.global.median_width"},
  {"id": "C6.lomo_resolvable", "value": 0.5974, "artifact": "metrics/by_metric/lomo_resolvability_audit.json", "path": "aggregates.resolvable_only.value"},
  {"id": "C6.lomo_ensemble", "value": 0.5947, "artifact": "metrics/by_metric/lomo_seed_ensemble.json", "path": "aggregates.resolvable_only.auc_of_mean_score"},
  {"id": "C6.lomo_raw_v1", "value": 0.5346, "artifact": "metrics/by_metric/metal_lomo_auc.json", "path": "value"},
  {"id": "C6.lomo_wasserstein", "value": 0.5575, "artifact": "metrics/by_metric/metal_lomo_auc_wasserstein.json", "path": "value"},
  {"id": "C6.lomo_wasserstein_real_only", "value": 0.3812, "artifact": "metrics/by_metric/metal_lomo_auc_wasserstein.json", "path": "real_only_fold_value"},
  {"id": "C6.ru_within", "value": 0.8271, "artifact": "metrics/by_metric/metal_activity_transfer_seed_stability_v2.json", "path": "within_metal_stability.Ru.mean", "tol": 0.0002},
  {"id": "C6.ir_within", "value": 0.8430, "artifact": "metrics/by_metric/metal_activity_transfer_seed_stability_v2.json", "path": "within_metal_stability.Ir.mean", "tol": 0.0002},
  {"id": "C7.sa_le_4p5_full_corpus", "value": 0.0028, "artifact": "metrics/by_metric/c7_scale_matched_pools.json", "path": "by_pipeline.generation.by_tier.full_corpus.rate_sa_le_gate"},
  {"id": "C8.dominant_share", "value": 0.7052, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "dominant_term.mean_share"},
  {"id": "C8.offset_share", "value": 0.1002, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out.verdict.mean_offset_share"},
  {"id": "C8.scale_share", "value": 0.2060, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out.verdict.mean_scale_share"},
  {"id": "C8.shape_share", "value": 0.6938, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out.verdict.mean_shape_share"},
  {"id": "C8.ceiling_r2", "value": 0.1524, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out.verdict.mean_ceiling_r2"},
  {"id": "C8.rho_metal_out", "value": 0.2960, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out.pooled.centred.pooled_spearman"},
  {"id": "C8.rho_ligand_disjoint", "value": 0.1575, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out_ligand_disjoint.pooled.centred.pooled_spearman"},
  {"id": "C8.metals_above_gate", "value": 2, "artifact": "metrics/by_metric/msrs_lomo_error_decomposition.json", "path": "endpoints.potency.splits.metal_out.verdict.n_metals_above_threshold"},
  {"id": "C8.perfect_level_gain", "value": 0.0784, "artifact": "metrics/by_metric/msrs_metal_descriptor_level.json", "path": "counterfactual_summary.mean_gain_over_cells"},
  {"id": "C8.metals_still_negative_after_level", "value": 4, "artifact": "metrics/by_metric/msrs_metal_descriptor_level.json", "path": "counterfactual_summary.max_metals_still_negative"},
  {"id": "C10.k50_metals_nonnegative", "value": 0, "artifact": "metrics/by_metric/msrs_kshot_calibration.json", "path": "endpoints.potency.splits.metal_out_ligand_disjoint.verdict_at_gate_k.n_metals_r2_nonnegative"},
  {"id": "C9.ir_doi_overlap", "value": 0.9955, "artifact": "metrics/by_metric/msrs_external_cohort_probe.json", "path": "ircytotoxdb.doi_overlap_share"},
  {"id": "C9.ir_new_dois", "value": 1, "artifact": "metrics/by_metric/msrs_external_cohort_probe.json", "path": "ircytotoxdb.n_new_dois"},
  {"id": "C9.ir_identical_rows", "value": 0.8943, "artifact": "metrics/by_metric/msrs_external_cohort_probe.json", "path": "ircytotoxdb.row_agreement.share_within_0p01"},
  {"id": "C9.ir_median_abs_diff", "value": 0.0, "artifact": "metrics/by_metric/msrs_external_cohort_probe.json", "path": "ircytotoxdb.row_agreement.median_abs_diff_log10"},
  {"id": "C9.unmatchable_rows", "value": 0.2194, "artifact": "metrics/by_metric/msrs_cell_feature_probe.json", "path": "identity_mismatch.unmatchable_row_share"},
  {"id": "UP1.reference_r2", "value": -0.0719, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.reference_weighted_r2"},
  {"id": "UP1.free_desc_delta", "value": -0.0971, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.candidates.free_desc.delta_r2_weighted"},
  {"id": "UP1.pair_desc_delta", "value": -0.1128, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.candidates.pair_desc.delta_r2_weighted"},
  {"id": "UP1.product_desc_delta", "value": -0.1637, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.candidates.product_desc.delta_r2_weighted"},
  {"id": "UP1.free_onehot_delta", "value": 0.0204, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.candidates.free_onehot.delta_r2_weighted"},
  {"id": "UP1.free_onehot_ci_low", "value": -0.0154, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.candidates.free_onehot.bootstrap.ci95.0"},
  {"id": "UP1.free_onehot_ligand_disjoint", "value": -0.0202, "artifact": "metrics/by_metric/msrs_interaction_ab.json", "path": "primary.candidates.free_onehot.delta_r2_ligand_disjoint"},
  {"id": "UP2a.ligand_cell_n_pairs", "value": 360, "artifact": "metrics/by_metric/msrs_ranking_expanded.json", "path": "pairs.ligand_cell.overall.n_pairs"},
  {"id": "UP2a.ligand_cell_raw", "value": 0.5333, "artifact": "metrics/by_metric/msrs_ranking_expanded.json", "path": "pairs.ligand_cell.overall.raw_sign_accuracy"},
  {"id": "UP2a.ligand_cell_balanced", "value": 0.4705, "artifact": "metrics/by_metric/msrs_ranking_expanded.json", "path": "pairs.ligand_cell.overall.balanced_accuracy"},
  {"id": "UP2a.ligand_cell_ceiling_raw", "value": 0.6222, "artifact": "metrics/by_metric/msrs_ranking_expanded.json", "path": "pairs.ligand_cell.additive_ceiling.raw_sign_accuracy"},
  {"id": "UP2b.ligand_cell_raw", "value": 0.5417, "artifact": "metrics/by_metric/msrs_listwise_ranker.json", "path": "pairs.ligand_cell.overall.raw_sign_accuracy"},
  {"id": "UP2b.ligand_cell_balanced", "value": 0.5985, "artifact": "metrics/by_metric/msrs_listwise_ranker.json", "path": "pairs.ligand_cell.overall.balanced_accuracy"},
  {"id": "UP2b.ligand_cell_balanced_ci_low", "value": 0.5262, "artifact": "metrics/by_metric/msrs_listwise_ranker.json", "path": "pairs.ligand_cell.bootstrap.balanced_ci95.0"},
  {"id": "UP2b.ligand_cell_balanced_ci_high", "value": 0.6663, "artifact": "metrics/by_metric/msrs_listwise_ranker.json", "path": "pairs.ligand_cell.bootstrap.balanced_ci95.1"},
  {"id": "UP2b.strict_balanced", "value": 0.5055, "artifact": "metrics/by_metric/msrs_listwise_ranker.json", "path": "pairs.strict.overall.balanced_accuracy"},
  {"id": "UP6.reference_worst_stratum", "value": 0.8096, "artifact": "metrics/by_metric/msrs_calibration_shift.json", "path": "design.reference.worst_stratum"},
  {"id": "UP6.cohort_metal_coverage", "value": 0.8996, "artifact": "metrics/by_metric/msrs_calibration_shift.json", "path": "variants.cohort_metal.coverage"},
  {"id": "UP6.cohort_metal_worst_stratum", "value": 0.8956, "artifact": "metrics/by_metric/msrs_calibration_shift.json", "path": "variants.cohort_metal.worst_stratum.coverage"},
  {"id": "UP6.cohort_metal_width", "value": 2.0183, "artifact": "metrics/by_metric/msrs_calibration_shift.json", "path": "variants.cohort_metal.median_width"},
  {"id": "UP6.width_change", "value": 0.0065, "artifact": "metrics/by_metric/msrs_calibration_shift.json", "path": "gate.width_change_vs_reference"},
  {"id": "UP6.metal_worst_stratum", "value": 0.8331, "artifact": "metrics/by_metric/msrs_calibration_shift.json", "path": "variants.metal.worst_stratum.coverage"},
  {"id": "UP3.explicit_metal_share", "value": 0.0, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.explicit_metal_share"},
  {"id": "UP3.multi_fragment_share", "value": 1.0, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.multi_fragment_share"},
  {"id": "UP3.organic_embed", "value": 1.0, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.organic_embed_success"},
  {"id": "UP3.complete_embed", "value": 0.065, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.complete_embed_success"},
  {"id": "UP3.legal_embed", "value": 0.25, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.legal_embed_success"},
  {"id": "UP3.needs_carbanion", "value": 0.335, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.needs_aromatic_carbanion"},
  {"id": "UP3.needs_aromatic_n", "value": 0.84, "artifact": "metrics/by_metric/msrs_complex3d_probe.json", "path": "summary.needs_aromatic_n"},
  {"id": "UP3b.hetero_r2", "value": 0.2504, "artifact": "metrics/by_metric/msrs_dmpnn_donor_ab.json", "path": "results.hetero.r2"},
  {"id": "UP3b.carbanion_r2", "value": 0.2454, "artifact": "metrics/by_metric/msrs_dmpnn_donor_ab.json", "path": "results.hetero_carbanion.r2"},
  {"id": "UP3b.delta_r2", "value": -0.005, "artifact": "metrics/by_metric/msrs_dmpnn_donor_ab.json", "path": "delta_r2"},
  {"id": "UP4.potency_metal_out_raw", "value": -0.0734, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.potency|metal_out.weighted_r2_raw"},
  {"id": "UP4.potency_metal_out_erac", "value": -0.0744, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.potency|metal_out.weighted_r2_erac"},
  {"id": "UP4.potency_ligdisj_raw", "value": -0.15, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.potency|metal_out_ligand_disjoint.weighted_r2_raw"},
  {"id": "UP4.potency_ligdisj_erac", "value": -0.1739, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.potency|metal_out_ligand_disjoint.weighted_r2_erac"},
  {"id": "UP4.cis_metal_out_raw", "value": -0.0793, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.cisplatin_ratio|metal_out.weighted_r2_raw"},
  {"id": "UP4.cis_metal_out_erac", "value": -0.0904, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.cisplatin_ratio|metal_out.weighted_r2_erac"},
  {"id": "UP4.pooling_none", "value": -0.0806, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.potency|metal_out.pooling_spectrum.none"},
  {"id": "UP4.pooling_one_hot", "value": -0.0708, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "cells.potency|metal_out.pooling_spectrum.one_hot"},
  {"id": "UP4.sign_p", "value": 0.5034, "artifact": "metrics/by_metric/msrs_erac_metal.json", "path": "sign_test.p_two_sided"},
  {"id": "UP5.scratch_r2", "value": 0.2504, "artifact": "metrics/by_metric/msrs_pretrain_transfer.json", "path": "results.scratch.r2"},
  {"id": "UP5.pt_r2", "value": 0.2599, "artifact": "metrics/by_metric/msrs_pretrain_transfer.json", "path": "results.pt_pretrain.r2"},
  {"id": "UP5.delta_r2", "value": 0.0095, "artifact": "metrics/by_metric/msrs_pretrain_transfer.json", "path": "delta_r2"},
  {"id": "UP5.pt_inner_val_mae", "value": 0.4204, "artifact": "metrics/by_metric/msrs_pretrain_transfer.json", "path": "pretrain.inner_val_mae"},
  {"id": "UP8.zero_shot_os", "value": 0.3479, "artifact": "metrics/by_metric/msrs_meta_kshot.json", "path": "results.potency.splits.metal_out.per_metal.Os.zero_shot_r2"},
  {"id": "UP8.k50_offset_ru", "value": -0.4761, "artifact": "metrics/by_metric/msrs_meta_kshot.json", "path": "results.potency.splits.metal_out.per_metal.Ru.arms.offset.50.mean_r2"},
  {"id": "UP8.k50_offset_os", "value": 0.1958, "artifact": "metrics/by_metric/msrs_meta_kshot.json", "path": "results.potency.splits.metal_out.per_metal.Os.arms.offset.50.mean_r2"},
  {"id": "UP8.k50_offset_ru_worse", "value": 0.5, "artifact": "metrics/by_metric/msrs_meta_kshot.json", "path": "results.potency.splits.metal_out.per_metal.Ru.arms.offset.50.share_worse_than_zero_shot"},
  {"id": "UP8.k50_offset_os_worse", "value": 0.85, "artifact": "metrics/by_metric/msrs_meta_kshot.json", "path": "results.potency.splits.metal_out.per_metal.Os.arms.offset.50.share_worse_than_zero_shot"},
  {"id": "UP9.ligand_both", "value": 0.3321, "artifact": "metrics/by_metric/msrs_final_stack.json", "path": "results.ligand_both.r2"},
  {"id": "UP9.knn", "value": 0.1326, "artifact": "metrics/by_metric/msrs_final_stack.json", "path": "results.knn.r2"},
  {"id": "UP9.stack2", "value": 0.3337, "artifact": "metrics/by_metric/msrs_final_stack.json", "path": "results.stack2.r2"},
  {"id": "UP9.stack3", "value": 0.3328, "artifact": "metrics/by_metric/msrs_final_stack.json", "path": "results.stack3.r2"},
  {"id": "UP9.stack3_vs_single", "value": 0.0007, "artifact": "metrics/by_metric/msrs_final_stack.json", "path": "stack3_vs_best_single"},
  {"id": "UP7.single_potency", "value": 0.2365, "artifact": "metrics/by_metric/msrs_multitask_aux.json", "path": "results.single.potency.r2"},
  {"id": "UP7.aux_photo_potency", "value": 0.2497, "artifact": "metrics/by_metric/msrs_multitask_aux.json", "path": "results.aux_photo.potency.r2"},
  {"id": "UP7.aux_both_potency", "value": 0.2723, "artifact": "metrics/by_metric/msrs_multitask_aux.json", "path": "results.aux_both.potency.r2"},
  {"id": "UP7.photo_only_r2", "value": -0.0432, "artifact": "metrics/by_metric/msrs_multitask_aux.json", "path": "results.photo_only.phototoxicity.r2"},
  {"id": "UP7.cis_only_r2", "value": 0.1318, "artifact": "metrics/by_metric/msrs_multitask_aux.json", "path": "results.cis_only.cisplatin_ratio.r2"},
  {"id": "UP7.cis_leak", "value": 1.0, "artifact": "metrics/by_metric/msrs_multitask_aux.json", "path": "leakage.cisplatin_ratio_pairs_also_in_potency"},
  {"id": "UP7.aux_both_delta", "value": 0.0358, "artifact": "metrics/by_metric/msrs_multitask_aux.json", "path": "potency_delta_vs_single.aux_both"},
  {"id": "UP7.aux_photo_delta", "value": 0.0132, "artifact": "metrics/by_metric/msrs_multitask_aux.json", "path": "potency_delta_vs_single.aux_photo"},
  {"id": "UP10.potency.pairs_observed", "value": 2, "artifact": "metrics/by_metric/msrs_support_audit.json", "path": "endpoints.potency.n_pairs_observed"},
  {"id": "UP10.potency.share_ligands_ge2", "value": 0.0364, "artifact": "metrics/by_metric/msrs_support_audit.json", "path": "endpoints.potency.share_ligands_ge2_metals"},
  {"id": "UP10.potency.ruos_groups", "value": 354, "artifact": "metrics/by_metric/msrs_support_audit.json", "path": "endpoints.potency.pairs.2.n_groups"},
  {"id": "UP10.potency.ruos_separability", "value": 0.592, "artifact": "metrics/by_metric/msrs_support_audit.json", "path": "endpoints.potency.pairs.2.separability"},
  {"id": "UP10.potency.irrh_groups", "value": 170, "artifact": "metrics/by_metric/msrs_support_audit.json", "path": "endpoints.potency.pairs.4.n_groups"},
  {"id": "UP10.potency.irrh_separability", "value": 0.115, "artifact": "metrics/by_metric/msrs_support_audit.json", "path": "endpoints.potency.pairs.4.separability"},
  {"id": "UP10.potency.rurh_groups", "value": 15, "artifact": "metrics/by_metric/msrs_support_audit.json", "path": "endpoints.potency.pairs.1.n_groups"},
  {"id": "UP10.photo.pairs_observed", "value": 0, "artifact": "metrics/by_metric/msrs_support_audit.json", "path": "endpoints.phototoxicity.n_pairs_observed"},
  {"id": "UP10.cis.ruos_groups", "value": 128, "artifact": "metrics/by_metric/msrs_support_audit.json", "path": "endpoints.cisplatin_ratio.pairs.2.n_groups"},
  {"id": "UP17.additive_r2", "value": 0.2837, "artifact": "metrics/by_metric/msrs_shape_decomposition.json", "path": "additive_fit.oof_r2"},
  {"id": "UP17.noise_sigma2", "value": 0.043712, "artifact": "metrics/by_metric/msrs_shape_decomposition.json", "path": "additive_fit.noise_sigma2"},
  {"id": "UP17.ligand_metal_indep", "value": 0.7556, "artifact": "metrics/by_metric/msrs_shape_decomposition.json", "path": "independent.ligand_x_metal.share_of_residual"},
  {"id": "UP17.ligand_metal_forward", "value": 0.8015, "artifact": "metrics/by_metric/msrs_shape_decomposition.json", "path": "independent.ligand_x_metal.share_forward"},
  {"id": "UP17.paper_metal_indep", "value": 0.5258, "artifact": "metrics/by_metric/msrs_shape_decomposition.json", "path": "independent.paper_x_metal.share_of_residual"},
  {"id": "UP17.paper_metal_batch_first", "value": 0.5484, "artifact": "metrics/by_metric/msrs_shape_decomposition.json", "path": "independent.paper_x_metal.share_batch_first"},
  {"id": "UP17.cell_metal_forward", "value": 0.0, "artifact": "metrics/by_metric/msrs_shape_decomposition.json", "path": "independent.cell_x_metal.share_forward"},
  {"id": "UP17.unexplained_structure_first", "value": 0.0716, "artifact": "metrics/by_metric/msrs_shape_decomposition.json", "path": "gate.unexplained_share.structure_first"},
  {"id": "UP17.unexplained_batch_first", "value": 0.041, "artifact": "metrics/by_metric/msrs_shape_decomposition.json", "path": "gate.unexplained_share.batch_first"},
  {"id": "UP11.complete_case_ligands", "value": 0, "artifact": "metrics/by_metric/msrs_pair_delta.json", "path": "rank_structure.complete_case_ligands"},
  {"id": "UP11.span_2_metals", "value": 201, "artifact": "metrics/by_metric/msrs_pair_delta.json", "path": "rank_structure.ligand_metal_span_histogram.2"},
  {"id": "UP11.ruos_ligands", "value": 118, "artifact": "metrics/by_metric/msrs_pair_delta.json", "path": "rank_structure.n_ligands_per_pair.Ru-Os"},
  {"id": "UP11.irrh_ligands", "value": 73, "artifact": "metrics/by_metric/msrs_pair_delta.json", "path": "rank_structure.n_ligands_per_pair.Ir-Rh"},
  {"id": "UP11.ruos_irrh_overlap", "value": 0, "artifact": "metrics/by_metric/msrs_pair_delta.json", "path": "rank_structure.pair_ligand_overlap.Ru-Os & Ir-Rh"},
  {"id": "UP11.irrh_to_ruos_raw_r2", "value": -0.3619, "artifact": "metrics/by_metric/msrs_pair_delta.json", "path": "transfer.Ru-Os.arms.raw.r2_delta"},
  {"id": "UP11.irrh_to_ruos_bridged_r2", "value": -0.0952, "artifact": "metrics/by_metric/msrs_pair_delta.json", "path": "transfer.Ru-Os.arms.bridged[ie1_kj].r2_delta"},
  {"id": "UP11.ruos_to_irrh_raw_sign", "value": 0.4658, "artifact": "metrics/by_metric/msrs_pair_delta.json", "path": "transfer.Ir-Rh.arms.raw.sign_accuracy"},
  {"id": "UP11.ruos_to_irrh_bridged_r2", "value": -0.2256, "artifact": "metrics/by_metric/msrs_pair_delta.json", "path": "transfer.Ir-Rh.arms.bridged[ie1_kj].r2_delta"},
  {"id": "UP11.ie1_diff_ruos", "value": 129.8, "artifact": "metrics/by_metric/msrs_pair_delta.json", "path": "transfer.Ru-Os.bridges.ie1_kj.descriptor_diff_test"},
  {"id": "UP12.metal_out_max_delta", "value": 0, "artifact": "metrics/by_metric/msrs_two_stage.json", "path": "metal_out.max_abs_delta_over_metals"},
  {"id": "UP12.metal_out_improvement", "value": 0.0, "artifact": "metrics/by_metric/msrs_two_stage.json", "path": "gate.improvement"},
  {"id": "UP12.additive_within", "value": 0.2128, "artifact": "metrics/by_metric/msrs_two_stage.json", "path": "within_metal.aggregate.additive.weighted_r2"},
  {"id": "UP12.two_stage_within", "value": 0.1954, "artifact": "metrics/by_metric/msrs_two_stage.json", "path": "within_metal.aggregate.two_stage.weighted_r2"},
  {"id": "UP12.free_within", "value": 0.2519, "artifact": "metrics/by_metric/msrs_two_stage.json", "path": "within_metal.aggregate.free.weighted_r2"},
  {"id": "UP12.ir_drop", "value": -0.0501, "artifact": "metrics/by_metric/msrs_two_stage.json", "path": "within_metal.per_metal.Ir.delta_two_stage_vs_additive"},
  {"id": "UP14.doi_pair_in_train_share", "value": 0.0057, "artifact": "metrics/by_metric/msrs_support_stratified.json", "path": "protocols.doi.unit_train_evidence.has_pair_in_train.share_of_rows"},
  {"id": "UP14.doi_pair_in_train_rows", "value": 120, "artifact": "metrics/by_metric/msrs_support_stratified.json", "path": "protocols.doi.unit_train_evidence.has_pair_in_train.n"},
  {"id": "UP14.doi_pooled_r2", "value": 0.1629, "artifact": "metrics/by_metric/msrs_support_stratified.json", "path": "protocols.doi.overall.r2"},
  {"id": "UP14.doi_pair_in_train_sse_share", "value": 0.0049, "artifact": "metrics/by_metric/msrs_support_stratified.json", "path": "protocols.doi.unit_train_evidence.has_pair_in_train.share_of_sse"},
  {"id": "UP14.cross_metal_rows", "value": 1151, "artifact": "metrics/by_metric/msrs_support_stratified.json", "path": "row_flags_corpus.n_cross_metal_unit"},
  {"id": "UP14.metal_out_weighted_r2", "value": -0.0734, "artifact": "metrics/by_metric/msrs_support_stratified.json", "path": "protocols.metal_out.weighted_r2_per_metal"},
  {"id": "UP16.n_pairs", "value": 360, "artifact": "metrics/by_metric/msrs_ranking_support.json", "path": "strata.listwise_ligand_cell.n_pairs"},
  {"id": "UP16.osru_balanced", "value": 0.5807, "artifact": "metrics/by_metric/msrs_ranking_support.json", "path": "strata.listwise_ligand_cell.by_metal_pair.Os-Ru.balanced_accuracy"},
  {"id": "UP16.osru_ci_low", "value": 0.499, "artifact": "metrics/by_metric/msrs_ranking_support.json", "path": "strata.listwise_ligand_cell.by_metal_pair.Os-Ru.balanced_wilson95.0"},
  {"id": "UP16.irrh_balanced", "value": 0.56, "artifact": "metrics/by_metric/msrs_ranking_support.json", "path": "strata.listwise_ligand_cell.by_metal_pair.Ir-Rh.balanced_accuracy"},
  {"id": "UP16.irrh_ci_low", "value": 0.4799, "artifact": "metrics/by_metric/msrs_ranking_support.json", "path": "strata.listwise_ligand_cell.by_metal_pair.Ir-Rh.balanced_wilson95.0"},
  {"id": "UP16.drop_osru_balanced", "value": 0.648, "artifact": "metrics/by_metric/msrs_ranking_support.json", "path": "strata.listwise_ligand_cell.leave_one_pair_out.Os-Ru.balanced_accuracy"},
  {"id": "UP16.osru_n_needed", "value": 642, "artifact": "metrics/by_metric/msrs_ranking_support.json", "path": "strata.listwise_ligand_cell.sample_size.Os-Ru.n_for_half_width_0p05"},
  {"id": "UP18.n_edges", "value": 3, "artifact": "metrics/by_metric/msrs_metal_graph.json", "path": "graph.n_edges"},
  {"id": "UP18.is_tree", "value": true, "artifact": "metrics/by_metric/msrs_metal_graph.json", "path": "graph.is_tree_observed_subgraph"},
  {"id": "UP18.ruos_t_cluster", "value": 3.364, "artifact": "metrics/by_metric/msrs_metal_graph.json", "path": "edges_by_pair.Ru-Os.t_cluster"},
  {"id": "UP18.ruroh_t_cluster", "value": 2.577, "artifact": "metrics/by_metric/msrs_metal_graph.json", "path": "edges_by_pair.Ru-Rh.t_cluster"},
  {"id": "UP18.irrh_t_cluster", "value": 0.958, "artifact": "metrics/by_metric/msrs_metal_graph.json", "path": "edges_by_pair.Ir-Rh.t_cluster"},
  {"id": "UP18.ruos_precision_share_cluster", "value": 0.5903, "artifact": "metrics/by_metric/msrs_metal_graph.json", "path": "edges_by_pair.Ru-Os.precision_share_cluster"},
  {"id": "UP18.ruroh_precision_share_cluster", "value": 0.0054, "artifact": "metrics/by_metric/msrs_metal_graph.json", "path": "edges_by_pair.Ru-Rh.precision_share_cluster"},
  {"id": "UP18.kill_triggered", "value": false, "artifact": "metrics/by_metric/msrs_metal_graph.json", "path": "gate.kill.triggered"},
  {"id": "UP18.identifiable_dim_holdout_ru", "value": 1, "artifact": "metrics/by_metric/msrs_metal_graph.json", "path": "graph.per_metal_out.Ru.identifiable_level_dim_train"},
  {"id": "UP19.identifiable_rows", "value": 1151, "artifact": "metrics/by_metric/msrs_identifiability.json", "path": "metal_out.pooled.descriptors.identifiable_unit_rows.n"},
  {"id": "UP19.identifiable_share", "value": 0.0547, "artifact": "metrics/by_metric/msrs_identifiability.json", "path": "decomposition_summary.identifiable_row_share"},
  {"id": "UP19.pooled_all_r2_abs", "value": -0.064, "artifact": "metrics/by_metric/msrs_identifiability.json", "path": "metal_out.pooled.descriptors.all_rows.r2_abs"},
  {"id": "UP19.pooled_identifiable_r2_abs", "value": 0.3299, "artifact": "metrics/by_metric/msrs_identifiability.json", "path": "metal_out.pooled.descriptors.identifiable_unit_rows.r2_abs"},
  {"id": "UP19.pooled_identifiable_r2_centred", "value": 0.3806, "artifact": "metrics/by_metric/msrs_identifiability.json", "path": "metal_out.pooled.descriptors.identifiable_unit_rows.r2_centred"},
  {"id": "UP19.ru_centred_identifiable", "value": 0.4078, "artifact": "metrics/by_metric/msrs_identifiability.json", "path": "metal_out.per_metal.Ru.representations.descriptors.identifiable_unit_rows.r2_centred"},
  {"id": "UP19.re_identifiable_rows", "value": 0, "artifact": "metrics/by_metric/msrs_identifiability.json", "path": "metal_out.per_metal.Re.n_identifiable_unit_rows"},
  {"id": "UP19.gate_pass", "value": true, "artifact": "metrics/by_metric/msrs_identifiability.json", "path": "gate.gate_pass"},
  {"id": "UP20.cqr_coverage", "value": 0.9, "artifact": "metrics/by_metric/msrs_cqr.json", "path": "aggregate.cqr.coverage_mean"},
  {"id": "UP20.cqr_width", "value": 1.9871, "artifact": "metrics/by_metric/msrs_cqr.json", "path": "aggregate.cqr.median_width_mean"},
  {"id": "UP20.cqr_worst_group", "value": 0.8716, "artifact": "metrics/by_metric/msrs_cqr.json", "path": "aggregate.cqr.worst_group_coverage_mean"},
  {"id": "UP20.mondrian_worst_group", "value": 0.8961, "artifact": "metrics/by_metric/msrs_cqr.json", "path": "aggregate.cqr_metal_mondrian.worst_group_coverage_mean"},
  {"id": "UP20.difficulty_width", "value": 1.8981, "artifact": "metrics/by_metric/msrs_cqr.json", "path": "aggregate.difficulty.median_width_mean"},
  {"id": "UP20.global_width", "value": 2.0073, "artifact": "metrics/by_metric/msrs_cqr.json", "path": "aggregate.global.median_width_mean"},
  {"id": "UP20.gate_pass", "value": false, "artifact": "metrics/by_metric/msrs_cqr.json", "path": "gate.gate_pass"},
  {"id": "UP20.kill_triggered", "value": true, "artifact": "metrics/by_metric/msrs_cqr.json", "path": "gate.kill.triggered"},
  {"id": "UP21.base_r2", "value": 0.367, "artifact": "metrics/by_metric/msrs_paper_meta_offset.json", "path": "aggregate.base.r2_mean"},
  {"id": "UP21.ridge_design_r2", "value": 0.3557, "artifact": "metrics/by_metric/msrs_paper_meta_offset.json", "path": "aggregate.ridge_design.r2_mean"},
  {"id": "UP21.leak_arm_r2", "value": 0.5405, "artifact": "metrics/by_metric/msrs_paper_meta_offset.json", "path": "aggregate.ridge_design+outcome.r2_mean"},
  {"id": "UP21.lopo_ridge_r2", "value": -0.0259, "artifact": "metrics/by_metric/msrs_paper_meta_offset.json", "path": "per_seed.20261022.offset_diagnostics_fold0.lopo_ridge_r2"},
  {"id": "UP21.test_offset_sd", "value": 0.24, "artifact": "metrics/by_metric/msrs_paper_meta_offset.json", "path": "per_seed.20261022.offset_diagnostics_fold0.test_offset_sd"},
  {"id": "UP21.kill_triggered", "value": true, "artifact": "metrics/by_metric/msrs_paper_meta_offset.json", "path": "gate.kill.triggered"},
  {"id": "UP22.re_n_paired_rows", "value": 0, "artifact": "metrics/by_metric/msrs_paired_kshot.json", "path": "aggregate.Re.n_paired_rows"},
  {"id": "UP22.rh_np0_r2", "value": 0.0709, "artifact": "metrics/by_metric/msrs_paired_kshot.json", "path": "aggregate.Rh.ridge|0.mean_r2"},
  {"id": "UP22.re_np3_r2", "value": 0.1745, "artifact": "metrics/by_metric/msrs_paired_kshot.json", "path": "aggregate.Re.ridge|3.mean_r2"},
  {"id": "UP22.primary_metals_above_zero", "value": 3, "artifact": "metrics/by_metric/msrs_paired_kshot.json", "path": "gate.checks.primary.value"},
  {"id": "UP22.mean_r2_np0", "value": 0.0407, "artifact": "metrics/by_metric/msrs_paired_kshot.json", "path": "gate.checks.secondary.values.0"},
  {"id": "UP22.mean_r2_np3", "value": 0.0306, "artifact": "metrics/by_metric/msrs_paired_kshot.json", "path": "gate.checks.secondary.values.3"},
  {"id": "UP22.kill_triggered", "value": false, "artifact": "metrics/by_metric/msrs_paired_kshot.json", "path": "gate.kill.triggered"},
  {"id": "UP23.n_pairs", "value": 651, "artifact": "metrics/by_metric/msrs_ranking_hierarchical.json", "path": "design.n_pairs"},
  {"id": "UP23.n_pairs_kept", "value": 315, "artifact": "metrics/by_metric/msrs_ranking_hierarchical.json", "path": "design.n_pairs_kept"},
  {"id": "UP23.n_pairs_scored_kept", "value": 315, "artifact": "metrics/by_metric/msrs_ranking_hierarchical.json", "path": "design.n_pairs_scored_kept"},
  {"id": "UP23.n_groups_all_tied", "value": 279, "artifact": "metrics/by_metric/msrs_ranking_hierarchical.json", "path": "design.label_info.lambdarank_deadzone.n_groups_all_tied"},
  {"id": "UP23.lambdarank_none_balanced", "value": 0.5985, "artifact": "metrics/by_metric/msrs_ranking_hierarchical.json", "path": "arms.lambdarank_none.all_pairs.overall.balanced_accuracy"},
  {"id": "UP23.lambdarank_deadzone_balanced", "value": 0.5612, "artifact": "metrics/by_metric/msrs_ranking_hierarchical.json", "path": "arms.lambdarank_deadzone.all_pairs.overall.balanced_accuracy"},
  {"id": "UP23.bt_hier_balanced", "value": 0.4685, "artifact": "metrics/by_metric/msrs_ranking_hierarchical.json", "path": "arms.bt_hier.retained.overall.balanced_accuracy"},
  {"id": "UP23.kill_triggered", "value": true, "artifact": "metrics/by_metric/msrs_ranking_hierarchical.json", "path": "gate.kill.triggered"},
  {"id": "UP24.bce_auc", "value": 0.5449, "artifact": "metrics/by_metric/msrs_lomo_pairwise.json", "path": "aggregate.bce.auc_overall_resolvable"},
  {"id": "UP24.pairwise_edge_auc", "value": 0.5529, "artifact": "metrics/by_metric/msrs_lomo_pairwise.json", "path": "aggregate.pairwise_edge.auc_overall_resolvable"},
  {"id": "UP24.bce_auc_all_metals", "value": 0.6185, "artifact": "metrics/by_metric/msrs_lomo_pairwise.json", "path": "aggregate.bce.auc_all_metals"},
  {"id": "UP24.pairwise_edge_auc_all_metals", "value": 0.5352, "artifact": "metrics/by_metric/msrs_lomo_pairwise.json", "path": "aggregate.pairwise_edge.auc_all_metals"},
  {"id": "UP24.bce_ru_auc", "value": 0.548, "artifact": "metrics/by_metric/msrs_lomo_pairwise.json", "path": "aggregate.bce.per_metal.Ru"},
  {"id": "UP24.kill_applied", "value": false, "artifact": "metrics/by_metric/msrs_lomo_pairwise.json", "path": "gate.kill.applied"},
  {"id": "UP24.kill_triggered_by_rule_text", "value": true, "artifact": "metrics/by_metric/msrs_lomo_pairwise.json", "path": "gate.kill.triggered_by_rule_text"},
  {"id": "UP25.n_rows", "value": 43, "artifact": "metrics/by_metric/msrs_main_table.json", "path": "design.n_rows"},
  {"id": "UP25.n_not_identifiable", "value": 5, "artifact": "metrics/by_metric/msrs_main_table.json", "path": "design.by_identifiable.no"},
  {"id": "UP25.n_partial", "value": 17, "artifact": "metrics/by_metric/msrs_main_table.json", "path": "design.by_identifiable.partial"},
  {"id": "UP25.n_cross_checked", "value": 43, "artifact": "metrics/by_metric/msrs_main_table.json", "path": "checks.n_cross_checked"},
  {"id": "UP25.gate_pass", "value": true, "artifact": "metrics/by_metric/msrs_main_table.json", "path": "gate.gate_pass"},
  {"id": "UP26.r2_n0", "value": 0.3674, "artifact": "metrics/by_metric/msrs_paper_calibration.json", "path": "aggregate.0.eb_random.r2_mean"},
  {"id": "UP26.r2_n1", "value": 0.5559, "artifact": "metrics/by_metric/msrs_paper_calibration.json", "path": "aggregate.1.eb_random.r2_mean"},
  {"id": "UP26.r2_n10", "value": 0.6839, "artifact": "metrics/by_metric/msrs_paper_calibration.json", "path": "aggregate.10.eb_random.r2_mean"},
  {"id": "UP26.r2_n50", "value": 0.7243, "artifact": "metrics/by_metric/msrs_paper_calibration.json", "path": "aggregate.50.eb_random.r2_mean"},
  {"id": "UP26.matched_zero_n10", "value": 0.4005, "artifact": "metrics/by_metric/msrs_paper_calibration.json", "path": "aggregate.10.eb_random.r2_matched_zero_mean"},
  {"id": "UP26.delta_n10", "value": 0.2834, "artifact": "metrics/by_metric/msrs_paper_calibration.json", "path": "aggregate.10.eb_random.delta_vs_matched_zero_mean"},
  {"id": "UP26.affine_n10", "value": 0.7049, "artifact": "metrics/by_metric/msrs_paper_calibration.json", "path": "aggregate.10.affine.r2_mean"},
  {"id": "UP26.maxmin_n1", "value": 0.5787, "artifact": "metrics/by_metric/msrs_paper_calibration.json", "path": "aggregate.1.eb_maxmin.r2_mean"},
  {"id": "UP26.kappa", "value": 0.7781, "artifact": "metrics/by_metric/msrs_paper_calibration.json", "path": "gate.checks.report.kappa_mean"},
  {"id": "UP26.gate_pass", "value": true, "artifact": "metrics/by_metric/msrs_paper_calibration.json", "path": "gate.gate_pass"},
  {"id": "UP27.cqr_coverage", "value": 0.9000, "artifact": "metrics/by_metric/msrs_conformal_shift.json", "path": "aggregate.cqr.coverage_mean"},
  {"id": "UP27.cqr_width", "value": 1.9871, "artifact": "metrics/by_metric/msrs_conformal_shift.json", "path": "aggregate.cqr.median_width_mean"},
  {"id": "UP27.weighted_cqr_width", "value": 1.9849, "artifact": "metrics/by_metric/msrs_conformal_shift.json", "path": "aggregate.weighted_cqr.median_width_mean"},
  {"id": "UP27.aci_cqr_coverage", "value": 0.9106, "artifact": "metrics/by_metric/msrs_conformal_shift.json", "path": "aggregate.aci_cqr_g0p01.coverage_mean"},
  {"id": "UP27.aci_cqr_width", "value": 2.0572, "artifact": "metrics/by_metric/msrs_conformal_shift.json", "path": "aggregate.aci_cqr_g0p01.median_width_mean"},
  {"id": "UP27.density_ratio_auc", "value": 0.5678, "artifact": "metrics/by_metric/msrs_conformal_shift.json", "path": "gate.checks.attribution.density_ratio_auc_mean"},
  {"id": "UP27.ess_fraction", "value": 0.945, "artifact": "metrics/by_metric/msrs_conformal_shift.json", "path": "gate.checks.attribution.ess_fraction_of_cal"},
  {"id": "UP27.gate_pass", "value": false, "artifact": "metrics/by_metric/msrs_conformal_shift.json", "path": "gate.gate_pass"},
  {"id": "UP27.kill_triggered", "value": false, "artifact": "metrics/by_metric/msrs_conformal_shift.json", "path": "gate.kill.triggered"},
  {"id": "UP29.bt_balanced", "value": 0.5317, "artifact": "metrics/by_metric/msrs_davidson_ties.json", "path": "results.bt_baseline.balanced_accuracy"},
  {"id": "UP29.davidson_balanced", "value": 0.5312, "artifact": "metrics/by_metric/msrs_davidson_ties.json", "path": "results.davidson_none.balanced_accuracy"},
  {"id": "UP29.davidson_ci_lo", "value": 0.4438, "artifact": "metrics/by_metric/msrs_davidson_ties.json", "path": "results.davidson_none.cluster_bootstrap.lo"},
  {"id": "UP29.delta", "value": 2.2784, "artifact": "metrics/by_metric/msrs_davidson_ties.json", "path": "results.davidson_none.delta_mean"},
  {"id": "UP29.tie_rate", "value": 0.9873, "artifact": "metrics/by_metric/msrs_davidson_ties.json", "path": "results.davidson_none.predicted_tie_rate"},
  {"id": "UP29.gate_pass", "value": false, "artifact": "metrics/by_metric/msrs_davidson_ties.json", "path": "gate.gate_pass"},
  {"id": "UP29.kill_triggered", "value": true, "artifact": "metrics/by_metric/msrs_davidson_ties.json", "path": "gate.kill.triggered"}
]
```

### 2.7 UP 强化轮（2026-10-21，四臂全部预注册门槛）

三个 artifact 的门槛都写在 `todo.json::gates_frozen_at = 2026-10-21T15:10+08:00`，**在看到任何结果之前**。

**UP-1 交互约束消融**（`msrs_interaction_ab.json`，6 臂 × 5 金属 × 2 split × 2 endpoint，单 seed 筛选 + 5 seed 确认）：

| 臂 | 约束 | ΔR² vs additive_desc（potency/metal_out） | 95% CI |
|---|---|---:|---|
| additive_desc（参照，= 生产模型） | `[[lig],[met],[cell]]` | — | — |
| free_desc | 无 | **−0.0971** | [−0.1438, −0.0682] |
| pair_desc | `[[lig,met],[cell]]` | **−0.1128** | [−0.1674, −0.0814] |
| product_desc | 无（显式乘积特征） | **−0.1637** | [−0.3715, −0.0530] |
| free_onehot | 无 | **+0.0204** | [−0.0154, +0.0440] |

参照臂 R² = **−0.0719**，与 `msrs_lomo_surface.json` 的 Ru descriptors −0.0777 同源，**说明参照臂复现了生产模型**，Δ 是可信的。
判据（best ≥ +0.05 且 CI 下界 > 0 且 ligand-disjoint 同号）**三项全不满足** ⇒ `CONSTRAINT_IS_NOT_THE_LEVER`。
**关键机制**：留出金属上，`additive_onehot`（−0.0708）与 `additive_desc`（−0.0719）几乎相同 —— 金属描述符块在 metal_out 上贡献 **0.0011 R²**。放开交互只会把训练金属之间的差异当信号外推，方差涨得比偏差掉得快。

**UP-2a 排序重测**（`msrs_ranking_expanded.json`，不重训，只重算已有 5-seed OOF）：

| 配对键 | n 对 | raw | balanced | 加性上限 raw | balanced |
|---|---:|---:|---:|---:|---:|
| strict（ligand, cell, doi, time_h） | 218 | 0.5826 | 0.5272 | 0.6055 | 0.5 |
| ligand_cell（ligand, cell） | **360** | 0.5333 | **0.4705** | 0.6222 | 0.5 |

放宽配对键把 n 从 218 提到 360，**结论更差**：balanced 0.4705 低于 0.5。**加性上限的 balanced 恒为 0.5**，所以任何"balanced > 0.5"都是非加性信号 —— 这条对照基线是判据的核心。

**UP-2b listwise 排序**（`msrs_listwise_ranker.json`，LightGBM `lambdarank`，group = (ligand, cell)，DOI 分组 5 折，同一批对）：

| 配对键 | n 对 | raw | balanced | 95% CI (balanced) |
|---|---:|---:|---:|---|
| strict | 218 | 0.4312 | 0.5055 | — |
| ligand_cell | 360 | 0.5417 | **0.5985** | **[0.5262, 0.6663]** |

**这是本线第一次有 balanced 显著高于 0.5 的排序结果**（pointwise 同批对只有 0.4705，+0.128）。
但预注册三条要求同时满足：`balanced ≥ 0.58` ✓、`CI 下界 > 0.52` ✓、`raw ≥ 0.63` ✗（0.5417）⇒ verdict 仍为 **FAIL**。
方向自检 `corr(score, y) = −0.156`，分数没反号。strict 键下 balanced 只有 0.5055，**说明增益依赖放宽后的配对定义**，这是它的脆弱点。

**UP-6 条件覆盖**（`msrs_calibration_shift.json`，复用 `msrs_generalization` 的 OOF，不重训）：

| 变体 | 覆盖 | 中位宽 | 最差层 | 最差层覆盖 |
|---|---:|---:|---|---:|
| global（参照） | 0.9004 | 2.0052 | cohort_size=2 (n=851) | 0.8096 |
| metal | 0.9002 | 2.0231 | cohort_size=2 | 0.8331 |
| difficulty | 0.8789 | 1.9195 | cohort_size=2 | 0.8073 |
| cohort | 0.8994 | 1.9847 | Os (n=900) | 0.8844 |
| **cohort_metal** | 0.8996 | 2.0183 | Os (n=900) | **0.8956** |

**metal-Mondrian 只把 0.8096 抬到 0.8331 —— 失败层不是金属。** 按 cohort size 分箱后最差层才真的被抬起来。
`cohort_metal` 宽度只涨 **0.65%**，过门槛的第二分支（宽度增幅 <5% 且最差层 ≥0.88）⇒ **PASS**。
**宽度本身改不了**：90% 区间宽 2.0052 ⇒ 残差 SD ≈ 0.61，与 R²=0.38 一致；校准只能改条件覆盖，不能凭空缩窄。

### 2.8 UP-3 探针：3D 配位图在这份数据上做不出来（2026-10-21）

`msrs_complex3d_probe.json`，200 个金属分层抽样的 (配体, 金属) 对：

| 量 | 值 | 含义 |
|---|---:|---|
| `explicit_metal_share` | **0.000** | 配体 SMILES 里**根本没有金属原子** |
| `multi_fragment_share` | **1.000** | 每个"配合物"都是多点断开串（中位 3 段，平均 2.23 个带电荷辅片段） |
| `organic_embed_success` | 1.000 | 有机片段能嵌入 —— 但片段之间不相连，相对几何完全不受约束 |
| `legal_embed_success` | 0.250 | 只连 RDKit 能表示的给体（非芳香杂原子 + 卤素）时的成功率 |
| `complete_embed_success` | **0.065** | 连上真正会配位的给体（含芳香 N 0.84 / Cp、芳基碳负离子 0.335）后 |
| `complete_fail_graph_rejected` | **0.935** | 失败全部发生在 sanitize（kekulize）阶段，不是嵌入阶段 |

**判据用的是 complete 臂**：一个把 Cp / 吡啶给体丢掉的构造，描述的不是表里那个物种。
⇒ `ORGANIC_ONLY_FALLBACK`：退回「2D 图 + 显式金属节点 + 配位键」，而这条退路**已经存在**
（`molmetal/data/metal_graph.py:303-310`），就是 `msrs_dmpnn_surface.json` 的 `metal_state` 臂。

### 2.9 UP-9 集成：第三个基学习器不可加（2026-10-21）

`msrs_final_stack.json`，DOI 留出，折号直接读 OOF store（不重新划分），kNN 只在折内取邻居：

| 臂 | DOI R² | MAE |
|---|---:|---:|
| ligand | 0.2898 | 0.4935 |
| ligand + metal + cell | **0.3321** | 0.4786 |
| Tanimoto kNN（k=25, ECFP4 r=2） | 0.1326 | 0.5426 |
| stack2（两棵树的折内 ridge） | **0.3337** | 0.4801 |
| stack3（+ kNN） | 0.3328 | 0.4803 |

`stack3` 只比最好单臂高 **+0.0007**，而且**低于 stack2 0.0009** —— kNN 在两棵树在场后贡献为负。
门槛 0.42，差 0.087 ⇒ **FAIL**，UP-9 自己的 kill clause 生效：「集成 < 最好单臂 → 增益不可加，以单臂为准」。

**口径提醒**：这批数低于 0.3767 / 0.3811 的 headline，因为 DOI OOF store 里只有树臂（这个 split 没有 GNN OOF），且 ligand+both 用的是 ECFP4 而非 MACCS+RDKit。**两组数不可比，本 artifact 不替换 headline。**

### 2.10 UP-7 多任务辅助终点：过门的那一臂不独立，独立的那一臂不过门（2026-10-21）

`msrs_multitask_aux.json`，DOI GroupKFold(5)，共享编码器 + 3 输出头，**每 epoch 恰好 15,544 行**（辅助行按自然比例挤占，见 `budget_rows_per_epoch`），potency 只在留出折上评分：

| 臂 | potency R² | Δ vs single | phototoxicity R² | cisplatin_ratio R² |
|---|---:|---:|---:|---:|
| single | 0.2365 | — | — | — |
| aux_photo | 0.2497 | +0.0132 | −0.0734 | — |
| **aux_both** | **0.2723** | **+0.0358** | −0.0430 | **0.2442** |
| photo_only（参照） | — | — | −0.0432 | — |
| cis_only（参照） | — | — | — | 0.1318 |

**两条数值判据在 `aux_both` 上都过了**（potency +0.0358 ≥ 0.02；两个辅助终点相对各自单任务都没掉）。裁决仍然是 **FAIL**，原因是这个"过门"不构成辅助任务迁移的证据：

- `cisplatin_ratio` 的 **(配体, 细胞系) 对与 potency 的重叠率 = 1.000**（artifact `leakage`）。它的标签是同一批测量在顺铂敏感/耐药对上的比值 —— 它是 potency 标签的变换，不是另一个域的监督。
- 唯一独立的辅助任务 `phototoxicity` 只给 **+0.0132**，不到 0.02；而且它在**所有臂里 R² 都是负的**（−0.043 到 −0.073），说明共享编码器根本没学到这个终点，一个预测不了的辅助任务提供不了有用梯度。
- `aux_photo` 还把 phototoxicity 从 −0.0432 拖到 −0.0734（−0.030 > 0.02 容差），这是"辅助头被主任务挤掉"的直接证据。

**结论**：辅助任务在本数据上**没有跨任务迁移价值**；`aux_both` 的 +0.0358 是"同源标签加密度"的效应，写论文时不能当作多任务正则的收益引用。

**逐折复核**（`molmetal/reports/up7_fold_check/fold_check.json`，只重跑 `single` 与 `aux_both`）：pooled 数字**逐位复现**（0.2365 / 0.2723），逐折 Δ = **+0.0640 / −0.0320 / +0.0723 / +0.0674 / +0.0118**，4/5 折为正。方向一致但**符号检验 p≈0.16**，且 phototoxicity 的逐折 R² 在 **−0.230 到 +0.103** 之间跳 —— 辅助头本身是不稳定的，这比 pooled 均值更能说明它提供不了稳定梯度。

### 2.11 UP-10 支持度审计：10 个金属对里只有 2 个有配对证据（2026-10-21）

`msrs_support_audit.json`。一个"配对组"的定义是 `(ligand_canonical, cell_line_key)` 下**至少两个金属**都有测量；`Δ = mean(y|metal_a) − mean(y|metal_b)`。这是所有"金属替换效应"建模的样本单位 —— 第一轮 9 个包全部默认它够用，没人数过。

**potency（21,777 行 / 5,965 配体 / 717 细胞系 / 1,801 论文）**

| 金属对 | 组数 | 配体 | 细胞系 | 中位 \|Δ\| | 可分性 \|Δ\|/SD | 交互份额 [95% CI] | 状态 |
|---|---:|---:|---:|---:|---:|---|---|
| Ru–Os | **354** | 121 | 36 | 0.2769 | **0.592** | 0.7732 [0.7045, 0.8122] | OK |
| Ir–Rh | **170** | 73 | 33 | 0.1908 | **0.115** | 0.1748 [0.0, 0.4145] | OK |
| Ru–Rh | 15 | 10 | 8 | 0.7013 | 2.464 | 0.9086 [0.6962, 0.9501] | UNSUPPORTED（<20 组） |
| 其余 7 对 | **0** | 0 | 0 | — | — | — | EMPTY |

只有 **217 / 5,965 = 3.64%** 的配体在 ≥2 个金属下被测过。**7 个金属对的组数是零，不是"少"，是零** —— 那 7 对在语料里从未被同一配体、同一细胞系、同一论文下比较过。

**可分性是这里最刺眼的数**：Ir–Rh 的 `separability = 0.115`，即**典型替换效应（中位 |Δ| = 0.19 log 单位）小于该对自身的化验重复散布（noise_sd_delta = 0.40）**。Ru–Os 好一些（0.592）也仍是同一量级。逐金属噪声地板（来自未去重表的重复组）：Ru σ=0.2219（235 组）、Ir 0.1403（52）、Rh **0.3764（仅 4 组）**、Os 0.0449（20）、Re 0.2388（5）。**Rh 的噪声地板几乎没被约束住**，而 Ir–Rh 是仅有的两个"有支持"对之一。

**另外两个端点更差**：

| 端点 | 行数 | ≥2 金属的配体 | 有支持的金属对 |
|---|---:|---:|---|
| potency | 21,777 | 217（3.64%） | Ru–Os（354 组，sep 0.592）、Ir–Rh（170 组，sep 0.115） |
| cisplatin_ratio | 11,723 | 65（1.95%） | Ru–Os（128 组，sep 0.726）、Ir–Rh（34 组，sep 0.443） |
| phototoxicity | 1,150 | **0（0.00%）** | **无** |

phototoxicity 的 **479 个配体里没有一个在两种金属下被测过** —— 这个端点在结构上就不可能做金属替换分析，只能做单金属内的配体排序。

**金属留出场景（这是 metal_out 协议的真正约束）**：

| 留出金属 | 训练侧可用金属对 | 训练侧组数 |
|---|---|---:|
| Ru | Ir–Rh | 170 |
| Ir | Ru–Os | 354 |
| Rh | Ru–Os | 354 |
| Os | Ir–Rh | 170 |
| Re | Ru–Os + Ir–Rh | 524 |

**四个留出场景的训练侧只剩 1 个金属对**。一个金属对给出 1 个约束，金属描述符块最多只能定 1 个自由度 —— 这解释了 §2.7 里 `additive_onehot` 与 `additive_desc` 在 metal_out 上只差 0.0011 R²：**不是描述符不好，是训练侧没有第二个金属对来校准它**。

`metal_lomo.csv` 队列（5,460 行 / 1,968 配体）是同一个事实的另一面：只有 **42 个配体（2.13%）** 跨 ≥2 金属，且只有 Ru–Os（17 个共享配体）与 Ir–Rh（25 个）两个对非零（SMILES 未规范化，是下界）。

### 2.12 UP-17 shape 来源分解：结构块与论文批次块**各自**都超过 50%，顺序决定谁是第一（2026-10-21）

`msrs_shape_decomposition.json`。在**加性模型（ECFP4 + 金属块 + 细胞块 + **DOI 主效应**）的 OOF 残差**上做四路分解，不是原始 y。加性 OOF R² = 0.2837，残差方差 0.4145，噪声地板 σ² = 0.0437（df=371，取自**未去重**表）。

| 分量 | 独立矩估计份额 | 结构优先顺序 [CI] | 批次优先顺序 [CI] |
|---|---:|---|---|
| ligand × metal | **0.7556** | **0.8015** [0.7884, 0.8151] | 0.0 [0.0, 0.0] |
| cell × metal | 0.0745 | 0.0 [0.0, 0.0278] | 0.0 [0.0, 0.0] |
| ligand × cell | **0.8733** | 0.1269 [0.1203, 0.1627] | 0.4106 [0.3386, 0.3887] |
| paper × metal | **0.5258** | 0.0 [0.0, 0.0] | **0.5484** [0.5801, 0.6297] |

**冻结门槛（`gates_frozen_at = 2026-10-21T20:30+08:00`）：四个分量之和与残差方差相差 <5%，且每个分量有 CI。**

| 顺序 | 1 − explained_total | <5%？ |
|---|---:|---|
| 结构优先 | 0.0716 | ✗ |
| 批次优先 | 0.0410 | ✓ |

⇒ `gate_pass = false`（两个顺序都要过才算过）。结构优先顺序下 7.16% 的残差方差没有被这四块解释，超过 5% 的容差。

**kill 条款：「paper×metal > 0.5 ⇒ shape 主要是论文批次」。它触发了，但它的推论不成立。**

- 批次优先顺序：paper×metal = **0.5484** > 0.5 ✓ 触发。
- 同一顺序下 ligand×metal = **0.0**；反过来结构优先顺序下 ligand×metal = **0.8015**、paper×metal = **0.0**。
- 两个块**各自在自己的优先顺序下都超过 0.5**，且 ligand×metal 的 0.8015 > paper×metal 的 0.5484。

⇒ `verdict = SHAPE_SHARED_STRUCTURE_AND_PAPER_BATCH`，`kill_triggered = true`，`kill_consequence_valid = false`。
**"任何建模都是在拟合噪声"这句话在这里是错的**：批次块真实存在（0.53–0.55），但配体特异的结构块更大（0.76–0.80），所以学金属替换效应不是学噪声。

**顺序敏感度本身就是结论**：`order_spread` = ligand×metal [0.8015, 0.0]、paper×metal [0.0, 0.5484]、ligand×cell [0.1269, 0.4106]。四个块在构造上互相重叠（一个 ligand-metal 单元同时也是一个 ligand-cell 单元），所以**不存在唯一的"主导分量"**。论文里只能写成范围，不能写成单点。

**两条必须一起报的读数**：
- `cell × metal` 在任何顺序下都是 **0.0**（独立估计也只有 0.0745）—— **细胞系特异的金属响应不存在**，这与 §2.7 的 `pair_desc`（放开 ligand×metal 交互）反而变差 −0.1128 一致。
- 独立矩估计之和不等于 1（0.7556 + 0.0745 + 0.8733 + 0.5258 = **2.2292**），这不是错误，是重叠块的定义。**独立份额只能当上界读**。

**CI 的一个诚实的瑕疵**：批次优先顺序下 bootstrap 区间不包住点估计（paper×metal 点 0.5484、CI [0.5801, 0.6297]；ligand×cell 点 0.4106、CI [0.3386, 0.3887]）。原因是按配体整簇重采样会复制配体、放大组大小 n，而可靠性权重 `w = σ²_b/(σ²_b + σ²_noise/n)` 依赖 n，所以**被重采样的估计量不是原点估计量**。区间只能当方向性证据（paper×metal 在批次优先下稳定高于 0.5），不能当 0.5484 的误差棒。

### 2.13 UP-11 交互迁移：低秩检验**不可识别**，不是失败；配体不相交的跨对迁移全部为负（2026-10-21）

`msrs_pair_delta.json`（GPU0，`device = cuda:0`）。

**冻结的检验跑不了，原因比"rank-1 份额小"更硬。** Δ 矩阵的行是配体、列是金属对，要估计共享剖面 `u_l` 至少需要**一个配体出现在两列里**。实测：

| 金属对 | 配体数 | 与其它对共享的配体 |
|---|---:|---:|
| Ru–Os | 118 | **0** |
| Ir–Rh | 73 | **0** |
| Ru–Rh | 10 | **0** |

`ligand_metal_span_histogram = {"2": 201}` —— **201 个跨金属配体，每一个都恰好只跨 2 个金属，没有一个跨 ≥3**。三个双金属组（Os,Ru）125 /（Ir,Rh）75 /（Rh,Ru）14 两两不相交。所以 `complete_case_ligands = 0`，SVD 无从做起，`rank1_share` **无定义**（不是 0，也不是小于 0.30）。

⇒ `verdict = NOT_IDENTIFIABLE`。kill 条款「rank-1 < 0.30 → 7/10 金属对不可学」在形式上没有触发，因为分子分母都不存在；但结论方向相同。

**唯一还能操作的版本：配体不相交的跨对迁移。** 在 A 对的配体上拟合 Δ，在 B 对**完全不同的配体**上预测。这比原计划的秩-1 桥更严（配体和金属对同时未见），门槛按原值套用。

| 方向 | 训练/测试配体 | 臂 | R²_Δ [95% CI] | 符号准确率 [95% CI] |
|---|---|---:|---:|---|
| Ir–Rh → Ru–Os | 73 / 118 | Δ=0 | −0.1227 [−0.2621, −0.0366] | — |
| | | raw | −0.3619 [−0.6064, −0.1825] | 0.5085 [0.4153, 0.6017] |
| | | **bridged（ie1_kj）** | **−0.0952** [−0.2327, −0.0044] | 0.4915 [0.3983, 0.5847] |
| Ru–Os → Ir–Rh | 118 / 73 | Δ=0 | −0.0317 [−0.2116, −0.0001] | — |
| | | raw | −0.4245 [−1.0069, −0.1584] | 0.4658 [0.3562, 0.5753] |
| | | **bridged（ie1_kj）** | **−0.2256** [−0.5855, −0.0446] | 0.5342 [0.4247, 0.6438] |

**没有一个臂的 R²_Δ 是正的**，全部低于"预测测试集均值"。符号准确率的 CI 全部覆盖 0.5。门槛（符号 ≥0.60、R²_Δ ≥0.10、比 Δ=0 高 ≥0.05）三项全不满足。

**桥参数本身也不稳**（这是 1 参数模型的固有脆弱）：
- `pauling_en` 对 Ru–Os **完全无效** —— Ru / Os / Ir 的 Pauling 电负性都是 2.20，差值 0.0，scale 被压成 0，bridged 臂退化成 Δ=0 臂。
- `ie2_kj` 在 Ru–Os → Ir–Rh 上给出 scale = **−7.0**，R²_Δ 崩到 **−11.06**。两个小描述符差之比不是标定。
- 只有 `ie1_kj` 在两个对之间**符号相反**（Ru→Os +129.8，Ir→Rh −160.3），是唯一有区分力的桥；它的两个方向 R² 分别是 −0.0952 和 −0.2256。

**结论**：交互既不能从语料里分解出共享剖面（不可识别），也不能在配体不相交的两个金属对之间迁移（R²_Δ 全负、符号在 0.5 附近）。**7/10 金属对没有任何可用的外推路径**，这与 UP-10 的支持度边界一致。



### 2.14 UP-12 两阶段残差模型：metal_out 上交互项**恒等于 0**，在能生效的协议里也输给加性（2026-10-22）

`msrs_two_stage.json`（CPU，21,045 行 / 2,846 配体）。

Stage 1 就是 `msrs_lomo_surface` 的 descriptors 加性臂（`interaction_constraints [[ligand],[metal],[cell]]`）。Stage 2 是秩-1 交互 `λ·(w·ECFP4_l)·v_m`：`w` 只在**配对残差差** `d = resid(b) − resid(a)` 上拟合（残差取训练侧 OOF），`v` 由支持对固定（`v_b − v_a = 1`、`Σv = 0`），**无支持金属 `v = 0`**。

**metal_out（冻结门槛）**：留出金属在训练侧不属于任何支持对 ⇒ `v_held = 0` ⇒ 交互项逐行恒为 0。实测 `max |pred_two_stage − pred_stage1| = 0`（全部 21,045 行），五个金属的 R² 与 `msrs_lomo_surface` **逐位相同**：

| 留出金属 | Ru | Ir | Rh | Os | Re | 加权 |
|---|---:|---:|---:|---:|---:|---:|
| stage1 = stage1+stage2 | −0.0777 | −0.1283 | −0.1589 | +0.3479 | −0.1034 | −0.0734 |

⇒ `improvement = 0.0000`（门槛 +0.03），`gate_pass = false`，`gate_failure_mode = DEGENERATE_IDENTICAL_PREDICTIONS`。**这不是"两阶段没用"，是 metal_out 这个协议看不见这个杠杆**：任何以金属为自变量的交互项，在留出金属上都没有系数可乘。gate 退化这件事本身，是这一项的主要产出。

**within_metal（配体分组 5 折，金属在训练侧）**：这才是杠杆能起作用的地方。

| 臂 | 加权 R² | 加权 MAE |
|---|---:|---:|
| 加性（stage1） | 0.2128 | 0.5093 |
| 加性 + 秩-1 交互 | 0.1954 | 0.5168 |
| **自由交互（UP-1 的 free 臂）** | **0.2519** | 0.4933 |

按金属（加性 → 秩-1 → 自由）：Ru 0.2503 → 0.2362 → 0.2727；Ir 0.0974 → 0.0473 → 0.1547；Rh 0.0637 → **0.0890** → 0.2265；Os 0.3290 → 0.3297 → 0.3811；Re 0.0934 → 0.0934 → 0.1992。

**三件事同时成立**：(1) 金属在训练侧时，**不受约束**的交互比加性高 **+0.0391**；(2) 秩-1 支持受限的交互反而低 **−0.0174**，Ir 单独掉 **−0.0501**；(3) 同一个自由交互模型在 metal_out 上掉到 −0.097（UP-1）。所以交互**存在且可学**，但**只在金属被观测到时**；把参数化收紧并不能换来外推能力。

**秩-1 为什么输**：`stage2_r2_inner` 在五折里是 **−0.0138 到 +0.0650**，这个投影对**留出配体**的 Δ 几乎没有预测力；`λ_raw` 是 **1.77–4.68**，闭式收缩要求把投影放大 2–5 倍才能匹配 Δ 的量级 —— 方向不对，放大只是放大噪声。

kill（套用到 within_metal，因为冻结的 metal_out gate 已退化）：improvement −0.0174 < 0.01、Ir 掉 −0.0501 < −0.02 ⇒ **触发**。

### 2.15 UP-14 支持度分层：DOI 协议下只有 **0.57%** 的测试行在训练折里有配对证据（2026-10-22）

`msrs_support_stratified.json`（CPU，21,045 行；加性 descriptors 臂）。

两个轴，每个轴两种定义：**corpus**（全表统计，描述性）与 **train_evidence**（只看该折训练行，这才是模型能用的证据）。

语料层面：跨 ≥2 金属的单元占 **5.47%** 的行（1,151 行），金属属于支持对的行占 **96.41%**（20,289 行）—— 后者几乎覆盖全表，因为它只要求"这个金属在某个支持对里出现过"，不代表这一行有配对证据。**真正的数字是 fold-aware 的那个。**

| 协议 | 池化 R² | 行级：训练折里有配对 | 无配对 |
|---|---:|---|---|
| DOI（`GroupKFold` on doi） | 0.1629 | **0.57%（120 行）**，R² 0.3414，占 SSE **0.49%** | 99.43%，R² 0.1585，占 SSE **99.51%** |
| within_metal（配体分组） | 0.2530 | **0 行**（构造使然） | 100%，R² 0.2530 |
| metal_out | −0.0228（加权 −0.0734） | **0 行**（构造使然） | 100%，R² −0.0228 |

**为什么 DOI 下只剩 0.57%**：同一配体同一细胞系在两个金属下的测量，绝大多数出自**同一篇论文**，于是落在**同一个折**里。配对是**论文内现象**，整篇留出把它整体拿走了。这与 UP-17 的 `paper×metal = 0.5484` 是同一件事的两面。

**金属轴**（金属是否在训练侧的支持对里）：DOI 与 within_metal 下都是 96.41% / 3.59%（Re 是那 3.59%）。Re 在 within_metal 的 R² 是 **0.0934**，其余四个金属 **0.2563** —— Re 没有配对证据，只能靠加性。

kill 规则（支持行 <5% ⇒ 该指标退化为 case study）**在 DOI 协议上触发**：0.57% < 5%。

**一个必须说清的估计量区别**：本表 `metal_out.overall.r2 = −0.0228` 是**池化**（1 − SSE/SST 全部行），`weighted_r2_per_metal = −0.0734` 是**逐金属 R² 按 n 加权**，与 `msrs_lomo_surface` 同一个估计量。两者不可互换，引用时写明是哪一个。

### 2.16 UP-16 排序的支持度分解：360 对**定义上**全在支持组内，且没有一个金属对单独过半（2026-10-22）

`msrs_ranking_support.json`（纯重切，不重训）。

**支持度是协议的性质，不是测量结果**：一个排序对就是同一 `(ligand, cell)` 组在两金属下的两行，所以它**必然**落在"该组有 ≥2 金属"里。100% 支持是**同义反复**；不存在 unsupported 层，也无法在不补数据的情况下构造出来。

| 金属对 | n | TPR | TNR | balanced | Wilson 95% | 达到半宽 <0.05 所需 n |
|---|---:|---:|---:|---:|---|---:|
| Ir–Rh | 86 | 0.120 | 1.000 | 0.5600 | [0.4799, 0.6190] | 130（现有 86） |
| Os–Ru | 258 | 0.384 | 0.778 | 0.5807 | [0.4990, 0.6548] | 642（现有 258） |
| Rh–Ru | 16 | 1.000 | 0.000 | 0.5000 | [0.3981, 0.8967] | 226（现有 16） |

池化的 0.5985 来自存储的配体簇 bootstrap：**[0.5262, 0.6663]**，不含 0.5。但**没有任何一个金属对单独把自己的 CI 抬离 0.5**：Os–Ru 的下界 0.4990、Ir–Rh 的 0.4799，Rh–Ru 只有 1 个负例。**显著性来自合并，不来自任何一对。**

留一金属对（精确重算，不是加权平均）：去掉 Ir–Rh → 0.6034（n=274）；去掉 Os–Ru → **0.6480**（n=102）；去掉 Rh–Ru → 0.5788（n=344）。去掉 Os–Ru 反而**升高**，所以冻结的 kill 规则（"若 Os–Ru 单独解释全部信号"）**不触发**；但 0.6480 建立在 Ir–Rh 的 36 个负例上，TPR 只有 0.323，不能当成更强的结论。

**样本量需求**（按各对自身的 TPR/TNR 与正例率反解，Jeffreys 平滑避免 TNR=1.0 声称零方差）：Os–Ru 要 **642** 对（现有 258），Ir–Rh 要 **130**（现有 86），Rh–Ru 要 **226**（现有 16）。**现有语料最多只能把 Os–Ru 做到一半。**

⇒ DP-5 = **YES**：排序数字必须写成"3 个已观测金属对内"，不能写成跨金属排序能力。

### 2.17 UP-18 金属图的图论诊断：按精度排边，「有效」与「精确」是两个不同的集合（2026-10-22）

`msrs_metal_graph.json`（CPU，potency，21,045 行）。UP-10 用 `n_groups >= 20` 判边，本项改用三条边各自的均值 Δ 与三种标准误。

| 边 | 组数 | 配体 | mean Δ | sd Δ | ICC(配体内) | deff | t（化验噪声） | t（经验） | **t（配体簇 bootstrap）** | 95% CI | 精度份额（簇） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Ru–Os | 354 | 121 | +0.1340 | 0.4641 | 0.6615 | 2.274 | 11.14 | 5.43 | **3.364** | [0.0618, 0.2164] | 0.5903 |
| Ir–Rh | 170 | 73 | −0.0461 | 0.4395 | 0.7378 | 1.980 | 1.50 | 1.37 | **0.958** | [−0.1382, 0.0454] | 0.4044 |
| Ru–Rh | 15 | 10 | +1.0767 | 1.4003 | 0.9865 | 1.493 | 9.54 | 2.98 | **2.577** | [0.3931, 2.0333] | 0.0054 |

**UP-10 的判定维度确实错了，但错法与计划里写的不同。** 计划说「`n≥20` 把 t=9.54 的 Ru–Rh 判死、把 t=1.50 的 Ir–Rh 判活」。实测：

- t=9.54 只在**化验噪声**口径下成立。Ru–Rh 的 sd Δ 是 **1.4003**（ICC 0.9865，deff 1.493），配体簇校正后 t 掉到 **2.577**。它仍 ≥2（kill 未触发），但远不到 9.54。
- Ir–Rh 的 t 在三种口径下都 <1.5，**它的均值与 0 不可区分**。UP-10 判它 OK 的依据是组数，不是效应。
- 所以**「携带非零效应的边」= {Ru–Os, Ru–Rh}**，**「精度份额高的边」= {Ru–Os (0.59), Ir–Rh (0.40)}**。两个集合只在 Ru–Os 上重合。Ru–Rh 的效应最大但精度份额只有 **0.0054**，因为它的组间散布（sd 1.40）几乎全是配体间的真实差异（ICC 0.99），把均值的不确定度放大了。

**图结构**：观测子图（有边的金属）4 节点 3 边，**是树**（E=V−1，无环）；Re 度数为 0，是全图的孤立点（全图 = 2 个连通分量，E=V−2）。割点 = 全部 4 个节点。**树上没有环 ⇒ 没有间接路径 ⇒ node-splitting 无定义**（`NOT_APPLICABLE`，残差自由度 0）。不一致性不是"不存在"，是"不可检验"。

**水平解**（GLS，约束 Σu=0）：Ru +0.5834、Os +0.4494、Ir −0.5394、Rh −0.4933；对应 t 只有 **1.373–1.789**，**没有任何一个金属相对总均值的偏离是显著的**。树上是方阵（3 边 + 1 约束 = 4 未知），所以**点估计与权重无关**（两种加权逐位相同），精度只影响 SE。

**metal_out 的训练侧可识别维数**（V − 连通分量数）：留出 Ru → 1（只剩 Ir–Rh）；留出 Rh → 1（只剩 Ru–Os）；留出 Ir → 2；留出 Os → 2；留出 Re → 3。留出节点自身永远与训练图断开（它的边全被删掉），所以**留出金属的水平在任何场景下都不可识别**。

**Gate：FAIL（2/4 条断言）。** `assert_tree` ✓、`assert_edge_count` ✓、`assert_ruos_dominant` ✗（0.5903 < 0.80；只有化验噪声口径 0.8592 才过）、`assert_ruroh_survives_cluster` ✗（2.577 < 3.0）。**kill 未触发**（2.577 ≥ 2），所以「按精度有效边只有两条」这句不撤回 —— 但要说清是哪两条：按效应是 {Ru–Os, Ru–Rh}，按精度是 {Ru–Os, Ir–Rh}。

**Gate 写错了一处**：`assert_ruos_dominant` 的 0.80 是按化验噪声口径定的（它复现了计划里的 11.14/9.54/1.50），而同一张 gate 里另一条却要求"聚类校正后"。本项按簇口径为主、同时报噪声口径，两者都写进 artifact，不掩饰。

### 2.18 UP-19 metal_out 的可识别子集：**98.5% 的测试行没有配对伙伴**，有伙伴的那 5.47% 上 R² = 0.33（2026-10-22）

`msrs_identifiability.json`（CPU，1 min 47 s；与 `msrs_lomo_surface` 同模型同 seed，逐金属 R² **逐位复现**，`max_abs_delta = 0.0`）。

**分解恒等式**（逐金属，`r2_score` 的基线就是该金属测试均值）：

```
SSE_abs = SSE_centred + n·bias²
R²_centred = R²_abs + bias²/Var(y)
```

`R²_centred` 就是"若有一个 oracle 告诉你该金属的偏移，模型能拿多少分"——即**不依赖那个不可识别水平的那部分**。

| 留出金属 | n_test | 可识别行 | 占比 | R²_abs | bias | R²_centred | 偏移惩罚 | **R²_centred（仅可识别行）** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ru | 14,785 | 406 | 2.7% | −0.0777 | −0.2096 | −0.0031 | 0.0745 | **+0.4078** |
| Ir | 3,666 | 177 | 4.8% | −0.1283 | +0.1778 | −0.0513 | 0.0770 | **+0.2883** |
| Rh | 959 | 192 | 20.0% | −0.1589 | +0.3358 | +0.0565 | 0.2154 | **+0.2941** |
| Os | 879 | 376 | 42.8% | +0.3479 | −0.0562 | +0.3520 | 0.0041 | **+0.4281** |
| Re | 756 | **0** | 0.0% | −0.1034 | −0.2832 | +0.0855 | 0.1890 | NOT_EVALUABLE |
| **池化（全部行）** | 21,045 | 1,151 | 5.47% | **−0.0640** | — | **+0.0160** | 0.0800 | — |
| **池化（仅可识别）** | 1,151 | 1,151 | 100% | **+0.3299** | — | **+0.3806** | 0.0507 | — |

**这是本轮第一个正结果，但它不是"模型变好了"。** 三个事实同时成立：

1. **可识别子集上模型是有效的**：池化 R²_abs = **0.3299**、去中心化 **0.3806**，高于同一模型在 within_metal 的 0.2128。也就是说，当同一 (配体, 细胞系) 在别的金属下被测量过时，模型对该金属的配体排序**是有预测力的**。
2. **不可识别子集把 headline 拉成负数**：**94.5%** 的测试行（19,894 行）没有配对伙伴。它们的 R² 由池化恒等式（SSE 与 SST 分别相减）反解为 **-0.0913**，与 headline 同号同量级。所以 −0.0734 的构成是"5.47% 的行 +0.33"与"94.5% 的行 −0.09"的加权，**不是模型全面失效**。
3. **偏移项本身很大**：全行去中心化 R² 从 −0.064 升到 +0.016，惩罚 0.080；Rh 的惩罚是 **0.2154**（bias 0.3358）。这一项正是不可识别的金属水平。

**Gate：PASS。** primary 0.3806 ≥ 0.20（参照 within_metal 加性 0.2128）；secondary **4/4 个可评估金属** > 0.10（Re 有 0 个可识别行，计为 NOT_EVALUABLE 但仍留在 5 的分母里）。**kill 未触发**。

**必须同时说的三条边界**：

- **去中心化用了留出标签**，所以 0.3806 是 oracle-offset 上界，不是可部署分数。它能支撑的结论是"金属内排序可迁移"，不是"新金属的绝对活性可预测"。
- **可识别子集是选出来的**：能进这个子集的配体，都是被至少两个金属测过的配体（多为研究充分的化合物）。0.33 不是对任意新配体的能力。
- **Re 完全没有可识别行**：Re 的 756 行里没有任何 (配体, 细胞系) 在别的金属下出现过，所以 Re 一行都没有迁移，只能靠加性模型。

### 2.19 UP-20 CQR 条件覆盖：覆盖拿到了，宽度没降下来（2026-10-22）

`msrs_cqr.json`（CPU，5 seed；LightGBM `objective=quantile`，alpha ∈ {0.05, 0.95}；cross-conformal——外层折 k 用**其它折**的 OOF 分数校准，所以校准集与评分集不相交）。

| 臂 | 边际覆盖 | 中位宽度 | 最差金属组覆盖 | Winkler |
|---|---:|---:|---:|---:|
| **cqr** | **0.9000** | **1.9871** | **0.8716** | 2.6315 |
| cqr_metal_mondrian | 0.8999 | 1.9802 | **0.8961** | 2.6357 |
| global（现状基线） | 0.9003 | 2.0073 | 0.8600 | 2.7346 |
| metal | 0.9001 | 2.0329 | 0.8970 | 2.7334 |
| difficulty-normalized | 0.8767 | **1.8981** | 0.8623 | 2.8516 |

**结论：CQR 修好了覆盖，没修好宽度。**

1. **覆盖与最差组都达标**：0.9000（名义 0.90，差 0.0000）、最差组 0.8716（门槛 0.87）。这两项是 UP-6 之后第一次同时成立。
2. **宽度只比 global 窄 1.0%**（2.0073 → 1.9871），离门槛 1.80 差 **10.4%**。计划里写的"当前最好 1.9195"在本次同表复算是 difficulty 的 **1.8981**——CQR 比它**宽** 4.7%。
3. **difficulty 的窄宽度是买来的**：它用 **0.8767** 的覆盖换 1.8981 的宽度，与 CQR 差 **0.0233** 覆盖。gate 里 `beats_difficulty_at_coverage` 明确要求"在各自覆盖下比较"，所以这条判 FAIL 而不是 PASS。**一个更窄但覆盖更低的区间不是胜利。**
4. **metal-Mondrian 抬最差组几乎不要钱**：最差组 0.8716 → **0.8961**（+0.0245），宽度只涨 −0.0035（1.9802 vs 1.9871）。这与 UP-6 的 cohort-Mondrian 结论一致：**分层能修最差层，修不了宽度**。

**Gate：FAIL（4 条过 2 条）。** `median_width` ✗（1.9871 > 1.80）、`beats_difficulty_at_coverage` ✗；`marginal_coverage` ✓、`worst_metal_group_coverage` ✓。**kill 触发**（宽度 1.9871 > 1.82）⇒ **DP-7 走「放弃宽度目标」分支**：区间宽度不再是本轮的成功判据，论文只报"覆盖 0.90 可达、宽度由数据分辨率决定（中位 1.99 log10，约两个数量级）"。

### 2.20 UP-21 论文元特征偏移：偏移是真的，但不是论文描述的函数（2026-10-22）

`msrs_paper_meta_offset.json`（CPU，5 seed；23 维论文级元特征 = 年份 + 5 金属比例 + 细胞系分布熵 + 16 个 `crc32` scaffold 桶；**不含行数、不含任何 y 统计量**；DOI GroupKFold(5)，与 0.3767 同一划分）。

| 臂 | DOI R²（5 seed 均值） | 相对 base | 说明 |
|---|---:|---:|---|
| base（无偏移校正） | **0.3670** | — | 复现 0.3767 的 5 seed 区间 |
| ridge_design | 0.3557 | **−0.0113** | 主臂：23 维元特征 |
| lgbm_design | 0.3522 | −0.0148 | 换非线性，更差 |
| ridge_design+counts | 0.3538 | −0.0132 | 加行数/配体数（no-leak 禁止，仅参考） |
| ridge_design+outcome | **0.5405** | **+0.1735** | 泄漏上界：用测试论文自己的 y 均值/分位 |
| ridge_no_year | 0.3561 | −0.0109 | 去年份 |
| ridge_no_metal | 0.3572 | −0.0098 | 去金属组成 |
| ridge_no_scaffold | 0.3643 | −0.0027 | 去 scaffold 桶（损失最小） |

**论文偏移的规模**（fold 0 诊断）：训练论文偏移 sd **0.2281**、测试论文偏移 sd **0.2400**、平均绝对值 **0.2255**。偏移是真实存在且量级不小的——这正是 UP-17 里"论文×金属 0.5484"的同一个东西。

**但它不是元特征的函数**：留一论文（LOPO）重拟合的 ridge **R² = −0.0259**，即对留出论文的偏移**比恒预测 0 还差**。校正后偏移 MAE 从 0.1622 涨到 **0.1695**（相对变差 **4.55%**）。

**泄漏臂是这条结论的证明而不是反例**：一旦允许用测试论文自己的 y 均值与分位数，R² 从 0.3670 跳到 **0.5405**（+0.1735）。**信息在论文的标签里，不在论文的描述里。** 缺的不是"论文身份"（DOI one-hot 本来就没有），是"论文的结果"——而后者在留出时按定义不存在。

**Gate：FAIL（3 条过 1 条）。** `doi_r2` ✗（0.3557 < 0.42）、`offset_mae_drop` ✗（−0.0451 < 0.10）、`no_leak` ✓。**kill 触发**（0.3557 < 0.40）⇒ **DOI 线关闭**，不再尝试其它偏移模型。论文写"整篇留出 R² ≈ 0.37，剩余损失的主要成分是不可迁移的论文级偏移"，不再写"可用元特征校正"。

### 2.21 UP-22 配对 k-shot：UP-8 的 1/5 来自**采样方式**，不来自"没有配对点"（2026-10-22）

`msrs_paired_kshot.json`（CPU，5 seed × 7 个 n_paired × 3 臂 × 5 金属 × 20 次重复；k=50；对偶 ridge 与 `msrs_meta_kshot.adapt_ridge` 的 primal 形式等价）。

| 留出金属 | 零样本 R² | n_paired=0（行级采样） | n_paired=3 | 配对行总数 | UP-8 的整 DOI 组采样（n=0） |
|---|---:|---:|---:|---:|---:|
| Ru | −0.0777 | −0.1809 | −0.2040 | 406 | **−0.3161** |
| Ir | −0.1283 | −0.1404 | −0.1635 | 177 | **−0.2669** |
| Rh | −0.1589 | **+0.0709** | **+0.0631** | 192 | −0.0840 |
| Os | +0.3479 | +0.2805 | +0.2830 | 376 | +0.2816 |
| Re | −0.1034 | **+0.1734** | **+0.1745** | **0** | −0.0716 |

**两个结论同时成立，而且互相矛盾——这正是这一项的价值。**

1. **primary PASS（3/5）**：n_paired=3 时 Rh +0.0631、Os +0.2830、Re +0.1745 均 > 0，UP-8 的 1/5 变成 3/5。
2. **secondary FAIL（不单调）**：跨金属平均 R² 在 n_paired = 0/1/2/3 上是 **0.0407 / 0.0313 / 0.0432 / 0.0306**，既不单调也不上升。
3. **增益全部发生在 n_paired=0**：Rh 从 −0.1589 到 **+0.0709**、Re 从 −0.1034 到 **+0.1734**，都是在**一个配对点都没加**的情况下发生的。**Re 的配对行总数是 0**，所以 Re 的增益在定义上不可能来自配对。
4. **成因是采样粒度**：UP-8 按**整篇 DOI** 抽 k 个点（右列 −0.3161 / −0.2669 / −0.0840 / +0.2816 / −0.0716），本项按**行**抽。同样的 k=50、同样的臂，只换采样单位就把 1/5 变成 3/5。**UP-8 测到的是"整篇留出时少样本适配难"，不是"语料里缺配对点"。**
5. **但"配对是机制"没有被证实**：0→3 的增量在 Rh 上是 **−0.0078**，Os +0.0025、Re +0.0011，平均从 0.0407 掉到 0.0306。给模型配对点没有换来可测量的交互项。

**Gate：FAIL。** primary ✓（3 ≥ 3）、secondary ✗（非单调）。**kill 未触发**（3 ≥ 2），所以 k-shot 线**不关闭**；但论文只能写"新金属少样本适配在行级采样下 3/5 金属可用"，**不能**写"配对校准点有用"。

### 2.22 UP-23 deadzone 标签 + 层级 Bradley-Terry：噪声标签不是 0.5985 上不去的原因，去掉它们反而更差（2026-10-22）

`msrs_ranking_hierarchical.json`（CPU，5 seed；GroupKFold(5) on DOI，与 UP-2b 同一划分；评分器 `msrs_ranking_expanded.score_pairs` 逐字复用）。

**deadzone 的定义**：金属对 $(a,b)$ 的 `noise_sd_delta` 记为 τ，`|Δ| < τ` 的对**测量上不可分辨**。τ 取自 `msrs_metal_graph.json`：Ru–Os **0.2264**、Ir–Rh **0.4017**、Ru–Rh **0.4369**。

| 对表 | 总数 | deadzone 保留 | 保留率 |
|---|---:|---:|---:|
| 全部（训练用） | 651 | **315** | 48.4% |
| 过 |Δ|≥0.2091 的评分集 | 360 | **315** | 87.5% |

| 臂 | 全部 360 对 balanced | deadzone 保留 315 对 balanced | 说明 |
|---|---:|---:|---|
| **lambdarank_none** | **0.5985** | **0.6050** | 逐位复现 UP-2b 的 0.5985 ✓ |
| lambdarank_deadzone | 0.5612 | 0.5615 | 同样本、同目标，只把噪声对并成同 label |
| bt_none | 0.5369 | 0.5297 | 成对 logistic，651 对 |
| bt_exclude | 0.5411 | 0.5317 | 臂 A：丢弃 deadzone 对（315 对） |
| bt_weight | 0.5056 | 0.4974 | 臂 B：w = min(1, |Δ|/τ) |
| bt_hier | 0.4631 | 0.4685 | 加金属指示（加性项）后**最差** |

**结论：噪声标签不是瓶颈，把它们从训练里拿掉会让模型变差 0.0435。**

1. **复现是干净的**：`lambdarank_none` 在全部 360 对上给出 **0.5985**，与 UP-2b 逐位相同，说明本脚本的折、特征、评分没有漂移。它在保留子集上是 **0.6050** —— 保留子集**更容易**，不是更难。
2. **deadzone 标签直接伤到模型**：0.5985 → **0.5612**（−0.0373；在保留子集上 0.6050 → 0.5615，−0.0435）。原因在 label 结构：**1,586 个多金属组里有 279 个（17.6%）被整组并成一个 label**，这些组对 lambdarank 的梯度贡献为 0。把"测量上不可分辨"翻译成"训练时不可分辨"，等于主动丢掉 17.6% 的排序信息。
3. **成对 logistic 全部 ≤0.5411**，而且样本量解释得通：BT 臂每个训练折只有 ~520 对，lambdarank 有 21k 行。这一列不是"架构输了"，是"样本输了"，所以它不能用来支持任何关于 loss 的结论。
4. **`bt_hier` 最差（0.4631）**，且它的加性项在树上**饱和**：4 节点 3 边的图上 `u_Ru − u_Rh` 恰好复现 Ru–Rh 的均值 Δ，没有残差自由度可供"借强度"。UP-18 已经证明这一点，这里只是再次实测到它不带来增益。

**per-pair Wilson 有一个陷阱**：保留子集上唯一 Wilson 下界 > 0.5 的是 **Rh–Ru（15/16 = 0.9375，下界 0.7167）**，但这一对的多数类率**也是 0.9375**（16 对里 15 对同号），所以它的 balanced accuracy 恰好是 **0.5**。**Wilson 下界在极不平衡的对上测的是"会不会猜多数"，不是"有没有分辨力"。**

**Gate：FAIL（3 条过 1 条）。** `balanced_ge_0.65` ✗（0.6050）、`per_pair_wilson_lo_gt_0.5` ✗（1 对，需 2 对）、`n_pairs_retained_ge_240` ✓（315）。**kill 触发**（0.6050 < 0.62）⇒ **DP-8 走 yes 分支：排序线关闭**。用全部 360 对的口径读也是 0.5985 < 0.62，两种读法都触发，不存在口径争议。

### 2.23 UP-24 LOMO 的 pairwise logistic：换损失与换负采样都只值 +0.008 AUC（2026-10-22）

`msrs_lomo_pairwise.json`（CPU，5 seed；队列 = `metal_lomo.csv` 经 `collapse_ligand_duplicates`，**2,010 行**，折大小 Ru 1,311 / Ir 538 / Os 36 / Rh 51 / Re 74 —— 与 0.5974 那次审计**逐折相同**）。

| 臂 | Ru/Ir 均值（可解析） | Ru | Ir | 五金属均值 | 说明 |
|---|---:|---:|---:|---:|---|
| **bce** | **0.5449** | 0.5480 | 0.5418 | **0.6185** | 点式 logistic（同划分基线） |
| pairwise | 0.5450 | 0.5434 | 0.5466 | 0.5798 | 负样本均匀抽 |
| pairwise_edge | **0.5529** | **0.5751** | 0.5306 | 0.5352 | 负样本优先同配体/异金属 |

**答案是"两者都不是"，而且这一项有一个必须先说清的方法学缺陷。**

1. **同学习器下 pairwise 只值 +0.0080**：0.5449 → 0.5529。Ru 涨 +0.0271，Ir 跌 −0.0112 —— 一个换另一个，不是一致增益。
2. **边感知负采样把五金属均值拉低 0.0833**（0.6185 → 0.5352）。它把小折（Os 0.789 → 0.582、Rh 0.851 → 0.722、Re 0.363 → 0.267）全部变差，只换来 Ru 的提升。**"让模型必须分辨金属"的代价是它不再能靠配体走捷径，而小折没有足够的金属内样本重新学会。**
3. **绝对门槛 0.63 不可达，原因在学习器不在损失**：本脚本是 ECFP4 上的线性模型，它自己的 BCE 基线是 **0.5449**，比 D-MPNN 的 0.5974 低 0.0525。**用这条 0.61 的 kill 线去关闭 LOMO，测到的是"线性模型弱"，不是"pairwise 损失无效"。** 所以 artifact 里两个读数都记：`triggered_by_rule_text = true`、`applied = false`，并写明理由。这是一处**计划自身的口径错误**（门槛按 D-MPNN 冻结，代码点却指定特征模型），不是事后调整。
4. **能支撑的结论只有一条**：在**同一学习器、同一划分、同一 seed** 下，把 BCE 换成 pairwise logistic、并把负样本换成同配体异金属，Ru/Ir 均值从 0.5449 到 0.5529。**0.5974 与 0.63 之间的差距不是损失函数或负采样造成的。**

**Gate：FAIL（3 条过 1 条）。** `auc_overall_ge_0.63` ✗（0.5529）、`auc_per_metal_ru_ir_ge_0.60` ✗（Ru 0.5751 / Ir 0.5306）、`must_report_bce_baseline` ✓。**kill 文本触发（0.5529 < 0.61）但不适用**，理由见第 3 条；**LOMO 线的去留不由这一项裁定**，仍按 0.5974 的 D-MPNN 结果记。

### 2.24 UP-25 可识别性分层主表：43 个 headline 数字里 5 个不可识别、17 个只部分可识别（2026-10-22）

`msrs_main_table.json` + [`docs/PREDICTIVE_LINE_MAIN_TABLE.md`](PREDICTIVE_LINE_MAIN_TABLE.md)（CPU，纯汇总，不重训）。

主表把三轮的 headline 数字按**估计目标**重排，每个数字带 `ESTIMAND`（pooled / centered / per_pair / conditional / edge_level）与 `IDENTIFIABLE`（yes / partial / no）。**值一律从 artifact 读出、不手抄**，再与本索引的机器校验块交叉核对。

| 维度 | 分布 |
|---|---|
| 行数 | **43** |
| ESTIMAND | pooled 18 / per_pair 8 / conditional 7 / edge_level 7 / centered 3 |
| IDENTIFIABLE | **yes 21 / partial 17 / no 5** |

**三条断言全 PASS**：每行都有 ESTIMAND + IDENTIFIABLE + artifact 路径；43 个 artifact 全部在 registry 里；**43 个 id 与机器校验块逐个一致**（`mismatches = []`）。

**"不可识别"的 5 个**（`G.ligand.ecfp4` 0.5596、`UP14.metal_out_weighted_r2` −0.0734、`UP19.pooled_all_r2_abs` −0.0640、`UP21.lopo_ridge_r2` −0.0259、`UP11.complete_case_ligands` 0）**不是"估计得不好"，是目标本身依赖留出金属自己携带的水平**（或依赖一个语料里不存在的完整行）。它们可以出现在方法讨论里，**不能**作为能力陈述引用。

**"部分可识别"的 17 个**是最容易误用的一档：`UP2b.ligand_cell_balanced` 0.5985、`C6.lomo_resolvable` 0.5974、`UP16.osru_balanced` 0.5807 都属于"只在有配对证据的子群上成立"。引用时必须同时给出子群定义，否则就是把部分识别读成全体识别。

---

### 2.25 UP-26 每篇论文的 EB 自校准：**10 行已测值把 DOI R² 从 0.367 抬到 0.684**（2026-10-22）

`msrs_paper_calibration.json`（CPU 4.4 min，5 seed × 20 repeats）。

UP-21 证明论文偏移 sd 0.24 且**不可**从论文描述预测。UP-26 换问题：偏移不必预测，部署时新论文本来就有若干已测行。估计量是正态-正态层级下的后验均值

```
sigma2_noise  组内残差方差   （训练折，逐折重估）
tau2_paper    论文间偏移方差 （训练折，逐折重估）
kappa         = sigma2_noise / tau2_paper
offset_hat    = n_cal / (n_cal + kappa) · mean(y − y_hat | 该论文的校准行)
```

**kappa = 0.7781**（五折 0.7332–0.8087），`sigma2_noise` 0.164–0.170、`tau2_paper` 0.211–0.224。这个收缩因子不是调出来的，是估出来的。

| n_cal | eb_random | eb_maxmin | affine | 同批行零校正 | 校正增量 | 评估行数 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.3674 | 0.3674 | 0.3674 | 0.3674 | 0 | 21,309 |
| 1 | **0.5559** | 0.5787 | 0.4258 | 0.3692 | +0.1867 | 19,620 |
| 2 | 0.6094 | 0.6396 | 0.5210 | 0.3713 | +0.2381 | 18,077 |
| 5 | 0.6584 | 0.6704 | 0.6565 | 0.3804 | +0.2781 | 14,223 |
| 10 | **0.6839** | 0.6829 | **0.7049** | 0.4005 | **+0.2834** | 9,885 |
| 20 | 0.7038 | 0.7087 | 0.7295 | 0.4254 | +0.2784 | 5,625 |
| 50 | 0.7243 | **0.7488** | 0.7475 | 0.4347 | +0.2895 | 2,782 |

**必须分开报的两件事**（否则会把行构成读成模型增益）：n_cal 越大，评估行越少、越偏向大论文，所以"同批行零校正"本身从 0.3674 升到 0.4347。**真实增益是最后一列**，在 n_cal=10 上是 +0.2834；绝对 R² 从 0.3674 到 0.6839 里有 +0.0331 是行构成、+0.2834 是校正。

**三条可引用的读数**：

1. **1 行就拿到一半以上**（0.5559）。部署时不需要 50 个对照样，1 个就足以把论文水平对齐。
2. **随机抽样在 n_cal=10 与 MaxMin 打平**（0.6839 vs 0.6829），但在两端 MaxMin 更好（n=1 +0.0228、n=50 +0.0245）。因为目标是**均值**，随机是无偏估计；MaxMin 的优势在斜率，所以从 n_cal=10 起 `affine` 反超（0.7049 vs 0.6839），n=50 时 0.7475 vs 0.7243。
3. **上界是 0.7337**：独立算术核对（把每个测试论文的残差均值全部减掉、评估行不变）给出 0.7337，实测 0.6839 是其 89%。**这个数字封住了"再多校准行能到 0.9"的想象**。

**必须声明的口径**：n_cal=10 时 9,885 个评估行里只有 **1,017 行（10.3%）** 来自"只剩 1 行可评"的小论文，其余 89.7% 来自 ≥12 行的论文，所以结论不是被 2 行论文撑起来的。

### 2.26 UP-27 漂移下的覆盖：加权 conformal 无效，ACI 只买到覆盖（2026-10-22）

`msrs_conformal_shift.json`（CPU 6.8 min，5 seed）。

UP-20 在可交换 conformal 下拿到覆盖 0.9000、宽度 1.9871。UP-27 改的是**假设**而不是分位数模型：

| 臂 | 协议 | 覆盖 | 中位宽 | 最差金属组 | Winkler |
|---|---|---:|---:|---:|---:|
| cqr | doi | 0.9000 | 1.9871 | 0.8716 | 2.6315 |
| **weighted_cqr** | doi | 0.8998 | **1.9849** | 0.8693 | 2.6306 |
| weighted_global | doi | 0.8991 | 2.0108 | 0.8580 | 2.7330 |
| global | doi | 0.9003 | 2.0073 | 0.8600 | 2.7346 |
| difficulty | doi | 0.8767 | 1.8981 | 0.8623 | 2.8516 |
| **aci_cqr (γ=0.01)** | rolling | **0.9106** | 2.0572 | 0.8818 | 2.6366 |
| online_cqr（α 固定） | rolling | 0.9104 | 2.0551 | 0.8820 | 2.6362 |

**加权臂为什么无效，有直接证据**：似然比分类器的 **AUC = 0.5678**，加权后的有效样本量 **ESS = 1,348.7 / 1,428.9 = 94.5%**。两个论文总体在 23 维元特征上几乎不可分 —— 因为 DOI 折是**随机**划分论文，本来就没有协变量漂移。按年覆盖曲线给出同一个结论：CQR 逐年覆盖在 **0.7711（2011）到 0.9451（2005）** 之间摆动，加权后每一年变动 **≤0.003**（2011 年 0.7711 → 0.7689）。

**这是 UP-26 的镜像**：论文偏移在 x 里**不存在**（AUC 0.57），但在 y 里**存在且可测**（10 行换 +0.28 R²）。所以正确的说法不是"模型对论文漂移不够鲁棒"，而是**论文效应必须测量、不能建模**。

**ACI 的代价是诚实的**：覆盖从 0.9000 抬到 0.9106，宽度同时从 1.9871 涨到 2.0572。理论就是这么预言的（在线更新只保证长期边际覆盖），这里没有任何超额收益。gate 在宽度与最差组上 FAIL，kill（宽 >2.00 或覆盖 <0.88）**未触发** —— 覆盖本身是可达的，达不到的是宽度。

### 2.27 UP-29 平局作为第三类结果：Davidson 与 Bradley-Terry 无差别（2026-10-22）

`msrs_davidson_ties.json`（CPU 31 s）。

UP-23 把近平局行**删掉**或**降权**，两者都让排序变差，于是结论写成"噪声标签不是瓶颈"。UP-29 用 Davidson (1970) 的三结果似然重开这条线 —— 平局有它自己的概率，而不是"没有信息"：

```
P(a wins) = 1 / (1 + e^-s + δ e^(-s/2))
P(tie)    = δ e^(-s/2) / (1 + e^-s + δ e^(-s/2)),   s = f(x_a) − f(x_b)
```

| 臂 | balanced（保留子集，n=315） | 簇 bootstrap 95% | δ | 判为平局的比例 |
|---|---:|---|---:|---:|
| bt_baseline（UP-23 bt_exclude 复现） | **0.5317** | [0.4433, 0.6193] | — | 0 |
| davidson_none | **0.5312** | [0.4438, 0.6135] | 2.28 | **0.9873** |
| davidson τ×0.5 | 0.4968 | [0.4092, 0.5893] | 0.96 | 0.9626 |
| davidson τ×1.5 | 0.5225 | [0.4375, 0.6041] | 3.72 | 1.0000 |

**bt_baseline 逐位复现 UP-23 的 0.5317**，这是本包唯一需要的协议一致性证明。Davidson 与它差 **−0.0005**，即机制换了、结果没变。

**δ 自己给出了诊断**：拟合到边际平局率（336/651 = 51.6%）后 δ=2.28，此时**98.7% 的配对被判为平局**。也就是说，这个模型在 315 个"定义上不是平局"的评估对上，有 98.7% 直接放弃判断。**这不是拟合失败，是模型在说：分数分辨率低于噪声下限。** 三分类准确率 0.0127 是这句话的量化形式。

**kill 触发（0.5312 < 0.58）⇒ 排序线在第二次、不同机制下仍然关闭**。DP-8 维持 YES，且现在是"两个机制、同一个结论"，比 UP-23 单次更强。

---

## §3 撤回日志（引用前必看）

| 日期 | 撤回的数 | 换成 | 为什么 |
|---|---|---|---|
| 2026-10-19 | **0.5828 ± 0.0102**（ligand-grouped R²，曾被叫作 "canonical"） | DOI 留出 **0.3380** | ligand 划分里 **89.43%** 的测试行与训练集共享同一篇 DOI，学的是论文内实验偏移。同一套特征/超参/模型，换整篇留出即掉 0.20 |
| 2026-10-19 | **Δ 符号 0.6193** | **0.5826**（5 seed） | 单 seed 高抽样：5 个 seed 落在 0.5183–0.5963，0.6193 比全部五个都高 |
| 2026-10-20 | 「0.5826 接近 0.65 门槛」 | 多数类率 **0.6009** | 门内 218 对只有两个金属对，正号占 60.1%。0.5826 **低于"永远答正号"**，不是"接近门槛" |
| 2026-10-20 | 生成 **SA≤4.5 = 33.7%** | **0.28%** | 33.7% 是键头重建池（给定真实 tmQM 几何只跑键头），从不调用采样器 |
| 2026-10-19 | 「LOMO 0.5575，差 0.04」 | 可解析子集 **0.5974** | 0.5575 把 Rh(51)/Os(36)/Re(74) 三个低于 `MIN_UNIQUE_FOR_GATE=200` 的折等权平均进去 |
| 2026-10-21 | 「换哪个金属更好**不可学**」（无条件全称） | 「**点估计不可学（raw 低于多数类率），balanced 口径可学但未达可用**」 | UP-2b 在同一批折上改 listwise 目标后 balanced 0.5985 [0.5262, 0.6663]，CI 不含 0.5。**但没有撤回 0.5826/0.5333 那三行**：它们仍是 raw 口径的正确读数。说"不可学"而不说口径，是这次的错误 |
| 2026-10-21 | 「交互份额 0.712 ⇒ 放开加性约束能救 metal_out」 | **约束不是杠杆**（free −0.097 / pair −0.113 / product −0.164） | 方差分解测的是**观测到的**交互，metal_out 要求把它**外推到没见过的金属**；这两件事不是同一个量。以后引用 0.712 必须同时说它只描述训练金属内部 |

**这五条里前两条是同一个根因**：单 seed + ligand 划分。凡是看到预测线数字没有写协议、没有写 seed 数的，先假定它属于这一类。

---

## §4 脚本地图

每个 artifact 的 `provenance.script` 是权威。下表按用途分组。

### 4.1 主模型（in-house D-MPNN）

| 脚本 | 用途 | 引用数 |
|---|---|---:|
| `molmetal/baselines/dmpnn.py` | **主模型实现本体**（不是脚本） | — |
| `train_dmpnn_multitask_wasserstein.py` | 5 金属多任务 + Wasserstein pIC50 正则；论文 §3.2 折划分 | 434 |
| `train_dmpnn_multitask.py` | 多任务 D-MPNN 基础 CLI（回归头 + 分类头） | 483 |
| `train_dmpnn_pic50_real.py` | 每金属 pIC50 回归基线（alpha=1.0） | 12 |
| `pre_featurize_dmpnn.py` | 特征预计算缓存 | 37 |

### 4.2 LOMO 家族（**5 个训练器，容易混淆**）

| 脚本 | 是什么 | 产出 | 引用数 |
|---|---|---|---:|
| `train_metal_lomo.py` | **原始 m19 LOMO**（5 折 × 3 seed，gate 0.70） | `metal_lomo_auc.json` | 157 |
| `train_metal_lomo_v2.py` | Wf-145 平衡 Rh 队列重测 | `metal_lomo_auc_v2.json` | 47 |
| `train_metal_lomo_wasserstein.py` | Wf-125 W₂ 迁移正则（**W₂ 默认关闭**，Wf-E1 证明旧实现 detach 后零梯度） | `metal_lomo_auc_wasserstein.json` | 91 |
| `train_multimetal_lomo.py` | Wf-32 **另一个架构**（SE(3)-SB backbone）的 LOMO 包装器 | — | 25 |
| `train_chemprop_lomo_paper_cohort.py` | chemprop v2 **仅作参考对照**（PA3），不是主模型 | `metal_lomo_chemprop_v2.json` | 44 |

**要点**：`metal_lomo_auc_wasserstein.json` 的 headline 是 **half-synthetic**，引用必须带 `real_only_fold_value`。

### 4.3 MSRS 泛化 / 排序 / 校准

| artifact | 脚本 |
|---|---|
| `msrs_seed_ensemble.json` | `msrs_seed_ensemble.py` |
| `msrs_generalization.json` | `msrs_generalization.py` |
| `msrs_feature_expansion_5seed.json` + `msrs_feature_fusion.json` | `msrs_feature_expansion_ensemble.py` |
| `msrs_delta_metric_audit.json` | `msrs_delta_metric_audit.py` |
| `msrs_delta_direct.json` | `msrs_delta_direct.py` |
| `msrs_calibration.json` | `msrs_calibration.py` |
| `msrs_complex_features.json` | `msrs_complex_features.py` |
| `msrs_metal_ablation.json` | `msrs_metal_ablation.py` |
| `msrs_rank_within_cohort.json` | `msrs_rank_within_cohort.py` |

### 4.4 活性迁移（Krasnov 对照）

| artifact | 脚本 |
|---|---|
| `metal_activity_transfer.json` / `_baselines` / `_scaffold` / `_permutation_null` / `_significance` | `metal_activity_transfer_probe.py` |
| `metal_activity_transfer_seed_stability_v2.json` | `metal_activity_transfer_seed_stability.py` |
| `lomo_resolvability_audit.json` | `lomo_resolvability_audit.py` |
| `lomo_seed_ensemble.json` | `lomo_seed_ensemble.py` |

### 4.5 零引用脚本（2026-10-21 实测，git grep 全仓库）

`pa16_nci60_funnel.py`、`train_dmpnn_pic50_ablation.py` —— 仓库内无任何引用。
**没有删除**（可能是手工入口），但任何新工作不应再依赖它们。

---

## §5 数据阻断（2026-10-21 EXT 阶段后：B1 已从"缺数据"升级为"领域事实"）

| ID | 阻断 | 实测 | 需要什么 |
|---|---|---|---|
| B1 | 外部独立 cohort | **不存在。** IrCytoToxDB（Scientific Data 2024，2,694 条 / 222 篇 / 127 系，CC-BY-4.0）与 MSRS 的 DOI 重叠 **221/222 = 99.55%**，仅 1 个新 DOI；653 个匹配 (DOI,细胞系) 格里 **89.4%** 数值相同到 0.01 log10，中位差 **0.0**，r=0.9611。其余来源：ChEMBL 29 本地库 0 个目标金属、PlatinAI 只有 Pt、NCI-60 无 SMILES、DepMap Cloudflare Turnstile、GDSC HTTP 410、Cellosaurus 超时、GEO GSE36133 只有 CEL 原始文件、RuCytoToxDB 无 Zenodo 存款（921 篇 ⊂ MSRS 的 1,427 个 Ru DOI） | 人工整理 2023–2025 文献（EXT-5 的两周预算，**未做，未编造**） |
| B2 | 细胞系生物学特征 | **永久关闭。** 0 个注释文件；21.9% 的行来自 CCLE/DepMap 里按构造不存在的系（HELA / HEK293 / A2780 / A2780CISR / A549CISR）；top-46 覆盖 80.22% | 一份包含耐药衍生系与非 CCLE 系的注释，或 MSRS 自建注释——都不是下载 |

**不要用 Pt 的 PlatinAI 冒充外部验证。也不要用 IrCytoToxDB 冒充外部验证**——
它是同一份 curation 的 Ir 切片（见 C9）。

### 5.1 EXT 阶段一句话

跨金属外推的失败位置已经确定：**shape（排序）占平方误差 0.653–0.757，
offset（水平）只占 0.045–0.100**。所以"校准一下就能用"在算术上不成立
（完美水平只值 +0.078 R²，4/5 金属仍为负）；k=50 校准时 R²≥0 的金属是 **0/5**，
且小 k 校准在 55–95% 的重复里比不校准更差。表观迁移主要来自配体重叠：
potency 的合并 ρ 从 0.296（metal_out）掉到 0.158（ligand_disjoint）。
详见 `TODO/completed/ext_phase_results.md`。

---

## §6 怎么重跑

```bash
# 表格（12 张 / 0 MISSING；manifest 记 sha256）
OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/paper_metric_tables.py --strict

# 引用等级（CITE / NUM / NO）
.venv/bin/python molmetal/scripts/registry.py cite

# 本索引自校验
.venv/bin/python -m pytest molmetal/tests/test_predictive_line_index.py -q
```

重跑任何 artifact 之后，**必须先重跑 `paper_metric_tables.py --strict`**，否则表和产物脱节。
