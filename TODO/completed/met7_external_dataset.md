# MET-7 — 外部独立数据集验证

**日期**：2026-10-19
**状态**：BLOCKED-PENDING-DATA（已测量，不是断言）
**脚本**：`molmetal/scripts/msrs_external_dataset.py`
**测试**：`molmetal/tests/test_msrs_external_dataset.py`（12 个，全过）
**产物**：`metrics/by_metric/msrs_external_dataset.json`

## 为什么做

此前 MET-7 的状态是「BLOCKED-PENDING-DATA」，但没有 artifact 记录**为什么**
blocked。§5.2 要求每个结论可追溯，所以「没有外部数据」本身也必须是一个测量：
交集的三个键各有多少、哪一步归零、如果放宽到「同配体」还剩多少行。

预先登记的估计量（`todo.json::MET-7.method`）：

> 同配体 + **同金属** + 同细胞系，两边都有数值标签。

## 做法

1. **参照队列**：`data/msrs/measurements.parquet`（21,404 行、5 金属、3,169 个
   `ligand_main_fragment`、711 个细胞系）。
2. **候选外部队列**（磁盘上实际存在的）：
   - `PlatinAI`：`.data/molmetal/platinai_4cell.csv`，4,078 个配合物，4 个细胞系，
     `pIC50` 标签；
   - `NCI-60 GI50`：`.data/molmetal/NCI60_GI50/GI50.csv`（421 MB）。
3. **配体键**：外部 SMILES 是**含金属的完整配合物**（`[Pt+6]`、dative bond），
   MSRS 的 `ligand_main_fragment` 是**已去金属**的主片段。直接比字符串会得到
   假的 0，所以先删掉金属原子，再按 MSRS 的 `MIN_LIGAND_HEAVY_ATOMS=6` 取主片段。
4. **金属键**：精确元素符号。
5. **细胞系键**：MSRS `cell_line_key` 小写后精确匹配。
6. **不做任何插补**：任一键不匹配的行直接排除，不改标签、不换金属、不外推。

## MEASURED

### PlatinAI × MSRS

| 键 | 交集 | 说明 |
|---|---:|---|
| 金属 | **0** | PlatinAI 只有 Pt；MSRS 是 Ru/Ir/Rh/Os/Re，**Pt 训练行为 0** |
| 配体（主片段） | **144** | 2,612 个外部主片段 vs 3,169 个 MSRS 主片段 |
| 细胞系 | **3 / 4** | MCF7（MSRS 3,586 行）、A549（4,884）、A2780（2,063）；A2780cis **0 行** |
| **三键交集（登记估计量）** | **0** | 金属键归零 |

放宽到「同配体 + 同细胞系、不管金属」：

| 项 | 值 |
|---|---:|
| 共享配体的外部配合物 | **909** |
| 可用的（配合物 × 共享细胞系）标签 | **2,727** |

这 909 / 2,727 是真实的、>200 行的外部数据，但它们是 **Pt** 配合物。用它做
验证等于测「未见金属的外推」（LOMO AUC 0.5575 已经说明这件事很难），不是
MET-7 登记的「同金属」估计量。**所以不计入，也不拿它冒充外部验证。**

### NCI-60 GI50

| 项 | 值 |
|---|---|
| 列 | `RELEASE_DATE, EXPID, PREFIX, NSC, CONCENTRATION_UNIT, ... CELL_NAME, AVERAGE` |
| SMILES 列 | 无 |
| 金属列 | 无 |
| 可用行 | **0** |

NCI-60 只有 `NSC` 编号，磁盘上没有 NSC→结构的映射，也没有金属字段，无法与
MSRS 做任何键的连接。

## 判定

```
verdict = BLOCKED-PENDING-DATA
rule    = triple_same_metal_rows >= 200 AND (R2 >= 0.5 OR AUC >= 0.75)
measured triple_same_metal_rows = 0
```

**这不是「数据不够多」，是「金属不重叠」。** 现有任何外部队列都无法检验
MSRS 的五金属模型，因为外部队列里没有这五种金属的配合物。

## 诚实读数

1. **外部验证仍然缺失。** 论文里所有「泛化」结论仍然只是同一语料的重划分
   （ligand / DOI / scaffold / year）。MET-2 的 DOI 0.3321 是「跨论文」，不是
   「跨数据库」。
2. **配体层面的重叠是真的。** 144 个主片段、909 个配合物共享配体，说明外部
   化学空间并非完全不相交；挡住验证的是金属，不是配体。
3. **Pt 零样本是一个不同的估计量。** 它可以做（909 个配合物、2,727 个标签），
   但必须写成「未见金属外推」，且需要先解决标签尺度（PlatinAI `pIC50` 是
   −log10(IC50/M)，MSRS 是 log10(IC50/µM)）与辅 ligand 差异。
4. **NCI-60 不是「没试过」，是「结构上不可能」。** 没有 SMILES、没有金属，
   任何声称用它做外部验证的说法都需要先补一个 NSC→结构的映射。
5. **A2780cis 在 MSRS 里是 0 行。** 即使金属问题解决，顺铂耐药株这一列也用
   不上。
6. **脚本只读不训练。** 它不会替你跑一个模型再把结果说成「外部验证」。

## 复现

```bash
.venv/bin/python molmetal/scripts/msrs_external_dataset.py
# exit 0 = PASS, 2 = BLOCKED-PENDING-DATA（与 §4 的 SKIP 约定一致）
OMP_NUM_THREADS=1 .venv/bin/python -m pytest molmetal/tests/test_msrs_external_dataset.py -q
```
