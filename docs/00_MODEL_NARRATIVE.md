# 最终模型叙事（置顶）

> **置顶文档。** 本文件是论文叙事的**唯一权威版本**，也是收尾打包的依据。
> 与 `README.md` / `docs/ARCHITECTURE.md` / `docs/DESIGN_BOUNDARY.md` 冲突时以本文件为准。
>
> 冻结 2026-10-10。AGENTS.md §5.2：本文每个数字都追到 `metrics/by_metric/*.json` 或 `molmetal/reports/` 下的记录文件。
>
> **修订记录：** 本文件 2026-10-10 首版把"跨金属迁移"当成了项目主线，理由是那条线没有外部基线、所以只有我们能讲。**这个判断是错的**——没有基线不等于它是主线，只等于它不适合当头条。同协议口径下项目有 **9 项 competitive/winning、5 项 behind**，而首版一份都没报。项目体量是 `TODO/04_history/completed/` 下 **297+ 个已交付任务**。详见 §7。

---

## 1. 这个项目实际是怎么组合的

> 首版把三者画成"算法 → 生成 → 价值"的**串联流水线**，这是错的理解。
> 真实的结构是：**D-MPNN 是贯穿所有轴的共享表征引擎，Lambda Calculus 提供化学知识层，
> CFM/YuelBond 是以 D-MPNN 为 bond head 的结构生成器。** 生成器与打分器**共用同一个骨干**。

```
┌─────────────────────────────────────────────────────────────┐
│  化学知识层 — Lambda Calculus                                │
│  反应程序空间上的证明搜索（β-NF）+ Catherine 闭包性质         │
│  作用：定义"哪些变换是合法的"，而不是直接打分                │
└───────────────────────────┬─────────────────────────────────┘
                            │ 合法反应程序 / 12-tile 库
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  生成层 — CFM / YuelBond                                     │
│  EGNN velocity field  +  DMPNNBondHead  ← 骨干就是 D-MPNN    │
└───────────────────────────┬─────────────────────────────────┘
                            │ 几何 + 键拓扑
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  共享表征引擎 — D-MPNN（同一个骨干，不是"两个模型之一"）      │
│  ┌────────────┬─────────────┬──────────┬──────────────────┐  │
│  │ CFM bond   │ MoleculeNet │ 金属活性 │ pIC50 / 亲和力    │  │
│  │ head       │ 骨干 0.8149 │ 分类/回归│ 回归             │  │
│  └────────────┴─────────────┴──────────┴──────────────────┘  │
│  外加：tmQM 预训练编码器（下游共同初始化源）、checkpoint 守卫  │
└─────────────────────────────────────────────────────────────┘
```

**关键点：D-MPNN 在项目里承担 7 个角色**；`grep -rli dmpnn molmetal/scripts` 命中 **119** 个脚本（另有 tests 180 / configs 25，计数于 2026-10-10）。
"XGB 在某个子任务上更好"只与其中 1 个角色相关，**不能据此把 D-MPNN 降级**——
这个判断我犯过一次，已写入 `todo.json :: dmpnn_roles_2026_10_10`。

---

## 2. 论文该讲什么：同协议口径的真实战绩

> 数据源 `docs/ALIGNED_COMPARISON.md`（2026-10-08），每行都能追到 `metrics/by_metric/*.json`。
> 原则：**同数据集 + 同协议 + 同指标才可比**（`COMPARISON_BASELINES.md` §0）。

### 2.1 可以当头条的（同协议，competitive 或 winning）

| 指标 | 我们（MEASURED） | 同协议基线 | 结论 | 出处 |
|---|---|---|---|---|
| **MoleculeNet 均值**（D-MPNN, 5-seed） | **0.8149** | 0.80 gate | **过 gate**（Morgan 骨干时仅 0.7954） | `moleculenet_auc_dmpnn_5seed.json` |
| ├ bbbp | 0.9065 ± 0.0227 | 0.940 | −0.034 | 同上 |
| ├ clintox | 0.8209 ± 0.0122 | 0.894 | −0.073（已从 Morgan 的 −0.192 补回） | 同上 |
| └ hiv | 0.7900 ± 0.0109 | 0.823 | −0.033（已从 −0.076 补回） | 同上 |
| **m15  docking RMSD** | **0.6578 Å**（5/5 pockets ≤ 2.0 Å） | DiffDock 38.2% < 2 Å | **通过率 100% vs 38.2%** | `docked_pose_rmsd.json` |
| **m14  decode_ratio** | **0.9647–0.998** | TargetDiff 文献 ≥ 0.5 | **约 2×** | `decode_ratio.json` |
| **m21  NCI-60 9-cell AUC** | **0.8007 ± 0.0160** | Krasnov 0.81 | **≈ 打平**（−0.009） | `nci60_precious_9cell_auc.json` |
| **Krasnov 对齐 chemprop（3 金属）** | **0.7812**（Re 0.8027 / Os 0.7884 / Rh 0.7525） | Krasnov per-metal | scaffold split, 9/9 runs | `wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json` |
| **m1 Ru AUC 论文协议复现** | 0.7717 ± 0.0380 | Krasnov 0.810 ± 0.024 | −0.038，**跨队列**（我们 591 vs 论文 23 细胞系） | `metalcytotox_ru_auc.json::paper_protocol_measurement` |
| **m1 vs 自家单任务基线** | 0.7613 | 0.7123 | **+0.049 内部同口径** | `metalcytotox_ru_auc.json` |
| **m13  RDKit validity** | **1.000** | SOTA 0.99–1.00 | 达标 | `rdkit_validity.json` |
| **n_distinct** | **20** | gate ≥ 3 | **6.7× gate** | `n_distinct.json` |

### 2.2 真正落后的（同协议，5 项）

| 指标 | 我们 | 基线 | gap | 性质 |
|---|---|---|---|---|
| MoleculeNet bace | 0.7421 ± 0.0446 | 0.876 | −0.134 | **唯一真实架构级落后**。5-seed 收紧后仍稳定；attentivefp 0.7636 与 D-MPNN 打平（Δ +0.0131），换骨干无意义 |
| m19 跨金属 LOMO real-only | 0.5845 ± 0.0764 | 无同协议基线 | — | 见 §2.3 |

其余 3 项（clintox / hiv / bbbp）**换 D-MPNN 骨干后大部分已补回，现在是调参不是架构问题**。

### 2.3 关于跨金属这条线——它不是主线，但结论是真的

**头条数字已做种子稳定性验证**（`metrics/by_metric/metal_activity_transfer_seed_stability.json`，
seeds 42/43/44/49/123，同 cohort 同打分器）：

| 金属 | within-metal 区间 | 5-seed spread | Krasnov 已发表 |
|---|---|---|---|
| **Ru** | **0.8128 – 0.8363** | **0.0235** | 0.81 — **持平，见平票敏感性** |
| **Ir** | **0.8295 – 0.8597** | **0.0302** | 0.73 — **稳健超过** |
| Re | 0.9083 – 0.9819 | 0.0736 | — |
| Rh | 0.6429 – 0.7381 | 0.0952 | — |
| Os | 0.3133 – 0.5867 | **0.2734** | — |

**Ru 最差的那个种子（0.8128）高于 Krasnov 的 0.81，Ir 五个种子全部远高于 0.73。**
但这句话**只在当前去重规则下成立，而该规则对平票的处理是承重的** ——
见下方「平票规则的敏感性」，**论文里 Ru 必须写成"与 Krasnov 持平"而不是"超过"**。

**摆幅本身就是可分辨性的判据**：Os 的 spread 是 Ru 的 **11.6 倍**，那不是测量、是噪声。
Ru/Ir 恰好是唯二样本量够分辨的金属（n=1255 / 534，其余 51 / 36 / 74）。
这比"n<200 不可分辨"的事后声明硬得多——**用摆幅排序，不用样本量声明。**

跨金属 pooled 反倒极稳：0.5975 – 0.6038（3 seed，spread **0.0063**）。

### 平票规则的敏感性（`metal_activity_transfer_tie_sensitivity.json`）

去重折叠的判据是 `sum*2 >= n`，所以**恰好平票的配体会被判为 ACTIVE** ——
对活性主张而言这是**不保守**的那一侧。代码里原本有四处注释把它写成
"ties -> inactive, the conservative choice"，**那是错的**，2026-10-10 更正。

实测 80 个配体组平票（Ru 59 / Ir 19 / Rh 2，占去重语料 4.1%），当前规则下全部判 active。
换成保守规则（`sum*2 > n`）后：

| | 保守 ties→inactive | 当前 ties→active | Krasnov |
|---|---|---|---|
| **Ir** | 0.8237 – 0.8374 | 0.8295 – 0.8597 | 0.73 |
| **Ru** | **0.7880 – 0.8082** | 0.8128 – 0.8363 | 0.81 |

- **Ir 的主张稳健**：两种规则、每个种子都远高于 0.73。可以放心写"超过"。
- **Ru 的主张不稳健**：保守规则下 0.7880–0.8082，整体略低于 0.81。
  **"Ru 超过 Krasnov" 的方向由平票规则决定，不由化学决定。**

**论文写法**：Ir 写"在两种平票规则下均超过已发表值"；Ru 写"与已发表的 0.81 持平"，
并把两种规则的区间都报出来。平票本身是真实的标签噪声（见 `RC1`），
所以两个方向都不"正确"——但**不能把一个不稳健的主张写成稳健的**。

**正确用法**：这条线是 §4 结果里的一个**量化发现**（跨金属迁移代价 = 金属内 − LOMO，
同一语料同一打分器），不是项目头条，理由是 `registry/protocols.yaml` 对 m19_lomo 记
`baseline_paper: null` / `NOT ALIGN` —— **它是项目里唯一没有同协议外部基线的轴**。

论文位置：§4 结果 2（次要发现），per-metal 表 + spread 列 + 平票敏感性。

---

## 3. 细分任务 → 用哪个组件

| 子任务 | 指定组件 | 依据 |
|---|---|---|
| 反应程序搜索 | MCTS (VariancePUCT + replay) | 9/9 success，`n_distinct` 9.11/10 |
| 分子生成 / 成键 | CFM (EGNN + DMPNNBondHead) | bond 感知；**须与生成能力联合报告**（§5） |
| 通用性质预测 | D-MPNN 骨干 | MoleculeNet 均值 **0.8149**（5-seed，过 0.80 gate） |
| 金属内活性 | XGB (Morgan FP + GBDT) 胜出 | Ru 0.8363 / Ir 0.8295 |
| 跨金属迁移 | D-MPNN (LOMO) | 0.5845 ± 0.0764 |
| 分子对接 | 自建 Vina 流程 | 中位 RMSD **0.6578 Å**，5/5 ≤ 2.0 Å |
| NCI-60 9 细胞系 | 自建多细胞系轴 | **0.8007 ± 0.0160** ≈ Krasnov |
| MoleculeNet bace | attentivefp（与 D-MPNN 打平） | Δ +0.0131，在噪声内 |

---

## 4. 收尾清单：投稿前只剩 3 件事

> 相比首版的"4 个缺口"，这里砍掉了一个虚的（G2，见 §6）。

### 先看清 blocker 的真实分布（2026-10-10 全量补状态后）

29 个 blocker 里 **13 个已关闭/已定性、3 个部分解决、2 个待核实，真正 open 的只有 11 个**：

| 桶 | 数量 | 说明 |
|---|---|---|
| closed_or_settled | **13** | 含 4 条"披露要求"类（C3/L2/L3 等）—— 它们永远为真，是论文措辞约束，不是缺陷 |
| partial | 3 | C14 / L1 / I4 |
| needs_verification | 2 | C4 (m5 QED)、C6 (m7 diversity) —— 本轮未核实，非结论 |
| **open** | **11** | 其中 4 条同属一个 CFM 问题，3 条是文档同步 |

**真正待收尾的工程问题只有一个**：CFM 生成目标函数（C11 崩塌 / C12 八项 loss 恒零 /
C13 warmup 冻结整条 loss / C15 目标未定义）——这四条是**同一个问题的四个面**。
其余是文档同步与 SSOT 刷新。

> ⚠️ **汇报状态前读 `todo.json :: blocker_status_sweep_2026_10_10.rollup`，
> 不要用 `len(blockers)`。** 首版把 29 当成"29 个未解问题"汇报，是取样错误。

### R1 — bace：决定**接受并披露**，不再追 ⚠ 已决策 2026-10-10

−0.134 vs chemprop-D-MPNN，5-seed std 0.0446，稳定负 gap。**决定不修，写进局限。**

理由（三条都已实测，不是推测）：

1. **换骨干无效**：attentivefp 0.7636 ± 0.0398 vs D-MPNN 0.7505 ± 0.0456，
   Δ +0.0131 在噪声内 —— 打平说明问题不在骨干选择。
2. **最自然的机制假说已被证伪**：unconditional `pos_weight=neg/pos` 压制 bace 的
   假说，用 `--class-weight balanced` + dropout 0.1 + wd 1e-5 + patience 20 实测：
   bace **0.7421 → 0.6805（更差）**；同配置 bbbp 无对照臂复现 −0.013（噪声内），
   把 harness 漂移限定在 ±0.015。**所以 class-weighting 不是原因，不要再当修复方案。**
3. **BLK-D1 已按 5-seed 0.7505 ± 0.0456 关闭**为 `CLOSED_HONEST_NEGATIVE`，
   5 seed 相对 3 seed 只收紧了方差、没有移动均值 —— gap 是稳定的，不是种子运气。

**论文写法**（§6 局限，一条即可）：
> bace 上我们比 chemprop-D-MPNN 低 0.134（5-seed 0.7421 ± 0.0446 vs 0.876）。
> 该差距在更换骨干（attentivefp 打平）与类别权重调整（实测更差）后均未消除，
> 因此我们将其报告为架构层面的未解项，而非调参问题。其余三个 MoleculeNet 任务的
> 残差（clintox −0.073 / hiv −0.033 / bbbp −0.034）已通过 D-MPNN 骨干从
> Morgan 口径的 −0.192 / −0.076 / −0.068 大幅收窄，属调参范畴。

**不要再开的口子**：class-weighting（已证伪）、换骨干（已证伪）。若将来重启，
唯一没试过的是 D-MPNN 侧的结构改动，且必须先给出为什么其它三任务能收窄而 bace 不能的假设。

### R2 — `DESIGN_BOUNDARY.md` 的 "可达性 ≥0.95" 与盘上数据矛盾 ✅ 已更正 2026-10-10
`ood_reachability_summary.json` 实测：30 个 d-block 元素 **20 个**过 0.40 gate，
macro **0.4989**，5d 块通过率 0.5，未通过者由 **Pt/Pd/Au/Ir/Os** 主导——正是目标金属。
需按 §5.5 更正文档表述，或说明两者是不同 cohort。**这是文档准确性问题，不是实验缺陷。**

### R3 — 两处 SSOT 同名两值，易误引 ✅ 已加消歧注记 2026-10-10
`moleculenet_auc_dmpnn_5seed_rescue.json::runs.bace.rescue` = 0.6805，
而 `wf_fix3lines_bace_50ep_2026_10_09/metrics.json` = 0.7505，**不是同一个臂**。
配对结论只能用后者（与 attentivefp 逐 seed 同 split）。同理 `metal_lomo_auc_wasserstein.json`
的 0.5575 与 centering 跑的 0.5845 并存。按 §5.5 刷新 canonical 文件。

---

## 5. 成键指标 `bond_f1` 必须联合报告（方法论，不是缺陷）

生成侧最高 F1 是 **0.7836**，但它来自一个**完全不生成**的 run（`gen_validity = 0.0000`，
训练器自记 "carries no gradient this run"）。两臂同 epoch(29)：

| arm | bond_f1 | precision | recall | exact_match | decode_ratio | gen_validity |
|---|---|---|---|---|---|---|
| fm=false | 0.7836 | 0.6817 | 0.9215 | 0.0000 | 0.2500 | 0.0000 |
| **fm=true** | **0.6703** | 0.5390 | 0.8861 | 0.0000 | **0.5833** | **0.0625** |

`bond_pos_rate = 0.2986`，全正基线 F1 ≈ 0.460。两臂 recall 都 ~0.89–0.92 而 `exact_match` 皆 0
—— bond head 塌向"全预测为正"，F1 高低实际由 precision 决定，而 precision 跟着生成能力走。

**论文里生成侧该写的最强表述是 fm=true 的 `decode_ratio 0.5833` vs `0.2500`——
flow-matching 目标把残余生成能力翻了一倍以上**（代价是 bond head Δ −0.113，Pareto 权衡）。
`bond_f1` 必须与 `decode_ratio` / `gen_validity` / `exact_match` 联合报告，不能单列。

---

## 6. 被撤回的判断（四个）

| 首版判断 | 撤回理由 |
|---|---|
| "跨金属迁移是项目主线" | 无同协议基线的数字不适合当头条。同口径下项目有 9 项 competitive/winning |
| "MCTS 价值通道无影响"（诚实负面） | **样本不足下的过度解读**。3 targets × 3 seeds、`max_depth: 3`（观测到达 4）、50 simulations，9 次运行。通道确实跑通（737 次评估、方差非零）。**结论只能是"本规模下未测出差异"，不是"无影响"**。见 `todo.json :: BLK-P2` |
| "29 个 blocker" | 我数了总数没读状态。29 个里 **20 个 status 字段为空**，真实状态分布见 `todo.json` |
| "Ru 0.8363 超过 Krasnov 0.81，种子稳健" | **数值本身没错，但结论承重于去重的平票规则，而该规则的注释把方向写反了。** 判据 `sum*2>=n` 让平票判 **ACTIVE**（不保守），四处 docstring/comment 曾称其为 "ties -> inactive, the conservative choice"。实测 80/1950 配体组平票；换保守规则后 Ru 降至 0.7880–0.8082，跌破 0.81。**正确表述：Ir 稳健超过，Ru 持平。** 见 §2.3 平票敏感性与 `metal_activity_transfer_tie_sensitivity.json` |

---

## 7. 收尾打包建议

**论文主线（按强度排序）**：

1. **通用性质预测**：D-MPNN 骨干 MoleculeNet 均值 **0.8149**（5-seed，过 gate），相对 Morgan 骨干 +0.0195
2. **结构生成可用性**：decode_ratio **0.9647–0.998** vs 文献 ~0.5；RDKit validity **1.000**；n_distinct **20**（6.7× gate）
3. **对接精度**：中位 RMSD **0.6578 Å**，5/5 pockets 通过 —— 对 DiffDock 的 38.2% 通过率
4. **金属活性**：NCI-60 9-cell **0.8007 ± 0.0160** ≈ Krasnov；金属内 **Ir 在两种平票规则下均超过 Krasnov**
   （保守 0.8237–0.8374 / 当前 0.8295–0.8597 vs 0.73）；**Ru 与 Krasnov 持平**（保守 0.7880–0.8082 /
   当前 0.8128–0.8363 vs 0.81，方向取决于平票规则）
5. **方法学贡献**：跨金属迁移代价的量化（次要，带 per-metal 表与不可分辨声明）

**局限章节（真的只有这些）**：
1. bace −0.134（唯一真实架构级落后，两条修复路径均已实测证伪）
2. m19 跨金属 Os/Re 低于随机，且 Rh/Os/Re 因 n=51/36/74 不可分辨
3. 5d 贵金属可达性不足（closure reachability 0.359–0.652，gate 0.85 全部未过）
4. 标签噪声给出 ~92% 的 Bayes 上限；**去重平票规则承重**——80/1950 配体组平票，
   全部判 active，换保守规则后 Ru 跌破 Krasnov 0.81。Ru 的"超过"不可主张，Ir 的可以。

---

## 8. 相关记录

- 同协议总账：[`docs/ALIGNED_COMPARISON.md`](ALIGNED_COMPARISON.md) ← **写论文前必读**
- 指标 SSOT：`metrics/by_metric/`（161 个文件，计数于 2026-10-10）
- 诚实负面台账：`docs/HONEST_NEGATIVE_CATALOG.md`
- blocker 全集：`todo.json`；D-MPNN 七角色：`todo.json :: dmpnn_roles_2026_10_10`
- 根因分析：`todo.json :: root_cause_analysis_2026_10_10`
- 工程纪律审计：`docs/STANDING_DIRECTIVES.md` §I
