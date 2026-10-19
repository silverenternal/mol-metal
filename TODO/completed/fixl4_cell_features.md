# FIX-L4 — 细胞系生物学特征（外部数据）

**日期**：2026-10-19
**状态**：BLOCKED-PENDING-DATA（已测量，不是断言）
**脚本**：`molmetal/scripts/msrs_cell_features.py`
**测试**：`molmetal/tests/test_msrs_cell_features.py`（12 个，全过）
**产物**：`metrics/by_metric/msrs_cell_features.json`

## 为什么做

`todo.json::FIX-L4` 要求给 711 个细胞系加 CCLE/GDSC 表达或突变特征，验收线是
「覆盖率 ≥80% 且 R² 提升 ≥ +0.005」。此前的状态是 BLOCKED，理由是「仓库内
没有」——但这句话没有 artifact。本任务把 blocker 变成可核对的三件事：

1. 磁盘上到底有没有可引用的注释表；
2. 要覆盖 80% 的行，实际需要多少个细胞系；
3. 细胞系**身份**本身值多少 R²（任何替代特征的上限）。

## 做法

- **扫描**：`.data`、`data`、`molmetal/data` 三棵目录，深度 ≤4，文件名匹配
  `ccle / gdsc / depmap / cellosaurus / cosmic / expression / mutation / omics /
  rnaseq / transcriptome`（大小写不敏感）。
- **集中度**：按 potency 模型行（21,404 行）统计每个 `cell_line_key` 的行数，
  计算 top-k 累计占比与达到 80% 所需的最小 k。
- **身份上限**：读 `msrs_metal_ablation.json` 的 `cell_blind` 臂——去掉细胞系
  类别码后损失的 R²。这是「任何细胞表示最多能替代多少」的上界。
- **不训练**：脚本没有 `measured_gain` 输入时一律 BLOCKED。有文件不等于能过
  验收线，只有真跑出 ≥+0.005 才算 PASS。

## MEASURED

### 磁盘

| 项 | 值 |
|---|---:|
| 匹配到的注释文件 | **0** |
| 扫描根 | `.data`、`data`、`molmetal/data` |
| 深度上限 | 4 |

### 语料集中度（21,404 行 / 711 个细胞系）

| top-k | 覆盖行数 | 占比 |
|---:|---:|---:|
| 10 | 11,423 | 53.37% |
| 25 | 15,215 | 71.08% |
| 50 | 17,374 | 81.17% |
| 100 | 19,078 | 89.13% |
| 200 | 20,307 | 94.87% |

- **达到 80% 只需 46 个细胞系**（不是 711 个）。
- 539 个细胞系各 <10 行，合计只占 6.21% 的行。
- 最大的 10 个：A549 2,321 / HELA 2,162 / MCF7 1,880 / HEPG2 1,045 /
  A2780 1,036 / MDAMB231 917 / HCT116 666 / A2780CISR 511 / SW480 450 /
  HEK293 435。

### 身份上限（来自 MET-6 消融）

| 协议 | 去掉细胞系码的 ΔR² | 95% CI |
|---|---:|---|
| ligand | **0.0629** | [0.0502, 0.0806] |
| DOI | **0.0442** | [0.0258, 0.0620] |

## 判定

```
verdict = BLOCKED-PENDING-DATA
rule    = 有注释表 AND 能覆盖 80% AND 实测 ΔR² >= +0.005
measured: files=0, lines_for_80pct=46, measured_gain=None
```

**下载 46 个细胞系的 CCLE/GDSC 表就能满足覆盖率门槛**——这是唯一一条不需要
新实验的路径。但「+0.005」必须真跑一次消融，脚本本身不产出这个数。

## 诚实读数

1. **覆盖率门槛比想象的低。** 46 个细胞系就够 80%，所以「711 个系没法全部
   注释」不是有效的 blocker；真正的 blocker 是磁盘上一张表都没有。
2. **身份已经很值钱。** cell_blind 掉 0.0629（ligand）/ 0.0442（DOI）。生物
   特征要证明的是**在身份之上**再 +0.005，而不是替代身份。任何把「cell_blind
   掉 0.06」当成「生物特征能加 0.06」的说法都是错的。
3. **46 个系之外的 665 个系只占 19%。** 即使特征表只覆盖常见系，也足以训练
   一个可用的主模型；稀有系仍只能靠类别码。
4. **没有 tissue 列。** MSRS 只有 `cell_line_key` 和 `is_cancer_line`，连
   「组织来源」这种最低成本的注释都没有；FIX-L4 若只做 tissue one-hot，也
   需要外部映射表。
5. **不要凭记忆硬编码。** 脚本刻意只扫描磁盘、不内置任何 CCLE 数值；没有
   文件就报 0。
6. **PASS 需要一次真实验。** 有文件、覆盖够，仍然 BLOCKED——因为
   `measured_gain` 只能由一次带特征的消融提供。

## 复现

```bash
.venv/bin/python molmetal/scripts/msrs_cell_features.py
# exit 0 = PASS, 2 = BLOCKED-PENDING-DATA
OMP_NUM_THREADS=1 .venv/bin/python -m pytest molmetal/tests/test_msrs_cell_features.py -q
```
