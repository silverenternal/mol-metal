# MET-9 — 生成侧指标补齐

**日期**：2026-10-19
**状态**：DONE / NEGATIVE
**脚本**：`molmetal/scripts/cfm_generation_metrics.py`
**测试**：`molmetal/tests/test_cfm_generation_metrics.py`（7 个，全过）
**产物**：`metrics/by_metric/cfm_generation_metrics.json`
**输入**：
`molmetal/reports/cfm_fullcorpus_2026_10_10/generated_smiles.csv`（66 个存活样本）、
`.data/molmetal/metalcytotox_5metal_multitask_v4.csv`（train split）、
`cfm_free_generation_panel_seed0.json`（SA/QED/yield）、
`m7_diversity.json`（canonical IntDiv1）、
`cfm_vina_generated.json`（对接状态）

## 为什么做

生成线已有的强数字是「chemistry validity 1.000（存活者）」「decode_ratio
0.998」，但这两条都不是可发表的生成质量：前者只描述存活下来的分子，后者是
重建指标。缺的是 **novelty、uniqueness、IntDiv1、生成物 SA/QED、金属合规、
对接覆盖**。MET-9 把这六项补到一个 artifact 里，并把口径写清楚。

## 做法

- 生成集：`cfm_fullcorpus_2026_10_10` epoch 10 / seed 0，192 次尝试 → **66 个
  存活**（yield 0.344）。
- 训练参照：CFM v4 的 `split == train`（18,137 行 → 4,597 个唯一 canonical）。
- 指纹：Morgan r=2 / 1024 bit。
- Novelty：`1 − max Tanimoto to train`；同时报「max Tanimoto < 0.9 的比例」
  （标准口径）与「1 − 平均 max Tanimoto」（严格口径）。
- IntDiv1：MOSES 定义（含自配对）；并与**同 n 的训练子样本**（40 次抽样）比较。
- SA/QED：直接读 `cfm_free_generation_panel_seed0.json`，不重算（同一批 66 个）。
- 金属合规：RDKit 数每个分子里的 Ru/Ir/Rh/Os/Re/Pt 原子数。
- 对接：读 `cfm_vina_generated.json::RUN_STATUS`，不做任何外推。

## MEASURED

### 队列

| 项 | 值 |
|---|---:|
| 尝试次数 | 192 |
| 存活 | 66 |
| yield | 0.344 |
| RDKit 可解析 | 66（1.000） |
| 唯一 canonical | 66（uniqueness 1.000） |

### Novelty（对 CFM train）

| 项 | 值 |
|---|---:|
| mean max Tanimoto | **0.1483** |
| median | 0.1493 |
| min / max | 0.1078 / 0.1848 |
| mean novelty（1 − mean max T） | **0.8517** |
| fraction max T < 0.9 | **1.000** |
| fraction max T < 0.7 | **1.000** |

### Diversity

| 项 | 值 |
|---|---:|
| IntDiv1（n=66） | **0.8526** |
| 训练同 n 子样本（40 次） | 0.8234 ± 0.0073 |
| Δ vs 同 n 训练 | **+0.0292** |
| m7 canonical（n=37，另一参照语料） | 0.8690 |
| m7 的 n=37 训练参照（tmQM） | 0.8845 → Δ −0.0155 |

### 化学质量与合规

| 项 | 值 | 目标 | 判定 |
|---|---:|---:|:-:|
| SA（Ertl） | **6.4423** | ≤4.5 | **FAIL** |
| QED | 0.0944 | — | 描述 |
| 恰好 1 个金属 | **66/66（1.000）** | — | PASS |
| 金属分布 | Ir 18 / Ru 16 / Rh 15 / Re 13 / Os 4 | — | 偏斜 |

### 对接

| 项 | 值 |
|---|---|
| 状态 | **BLOCKED** |
| 原因 | `INCOMPLETE_RUN_DIED` |
| 计划 / 完成 pocket | 10 / **1** |
| 成功率 | 不可计算 |

## 判定

**NEGATIVE。** SA 6.4423 > 4.5；对接 BLOCKED（1/10 pocket）。即使 novelty
（fraction 口径）与 IntDiv1 过线，整体也不能 PASS。

## 诚实读数

1. **真正该报的 yield 是 0.344，不是 1.000。** 1.000 是 66 个存活者的
   chemistry validity；192 次尝试里只有 66 次产出可用分子。把 1.000 当头条
   是幸存者偏差。
2. **Novelty 依赖口径，必须两个都报。** 按「max Tanimoto < 0.9 的比例」是
   **1.000**；按「1 − 平均 max Tanimoto」是 **0.852**，低于 0.90 目标。
   说「novelty 1.0」会误导；说「novelty 0.852」也不完整。正确写法是
   「100% 的生成物与训练集的最大相似度 <0.19（平均 0.148），平均 novelty 0.852」。
3. **Novelty 高有一部分是「生成得奇怪」的副产品。** 最大相似度只有 0.108–0.185，
   因为金属中心被小的无机配体填满（`[Re](OH)(OH)(N(O)O)` 之类），这类结构
   在训练集里本来就没有。**novel 不等于有用。**
4. **SA 6.44 与 QED 0.094 指向同一个瓶颈：配体真实性。** 键存在 F1 0.996、
   连接性 0.970（比真实语料还好），说明键头在工作；问题在「填配位球」而不是
   「造药-like 配体」。这是生成线最该修的地方，不是 novelty。
5. **IntDiv1 的两个参照给出相反符号，原因是参照语料不同。** 对**自己的训练语料**
   同 n 比较是 **+0.029**（生成物略更多样）；m7 的 n=37 比较用的是 tmQM d-block
   子集，得到 **−0.016**。两个都真实，但**不能拿 0.853 去对 MOSES 0.857**——
   n 与语料都不同。
6. **金属合规 1.000 是真结果，但很弱。** 模型从不出 0 金属或 2 金属；分布偏向
   Ir/Ru（34/66），Os 只有 4 个。合规只说明「有一个金属」，不说明「金属选得对」。
7. **对接是 BLOCKED，不是 FAIL。** 10 个 pocket 只完成 1 个，进程死亡，没有
   成功率可言。贵金属力场缺失是另一条独立的 BLOCKED（`vina_status.json`）。
8. **单 seed、单 checkpoint、n=66。** 这是第一次真实测量，不是定论；跨 3 个
   full-corpus checkpoint 复现是下一步。

## 对主表的影响

- 生成线主表只能写：**yield 0.344；novelty（max T <0.9）100%、平均 novelty
  0.852；IntDiv1 0.853（同 n 训练 0.823）；SA 6.44（FAIL）；金属合规 1.000；
  对接 BLOCKED**。
- **不能写**「chemistry validity 1.000」「novelty 1.0」「SA 4.71」（后者是
  metallo_drugs 训练队列，不是生成物）。
- 生成线的强项只剩 validity（存活者）与 decode_ratio；这两条都不足以支撑
  「可合成、可对接的金属药物生成」的主张。按 §5.2，这条线应作为**边界与负结果**
  呈现。

## 复现

```bash
OMP_NUM_THREADS=1 .venv/bin/python molmetal/scripts/cfm_generation_metrics.py
OMP_NUM_THREADS=1 .venv/bin/python -m pytest molmetal/tests/test_cfm_generation_metrics.py -q
```
