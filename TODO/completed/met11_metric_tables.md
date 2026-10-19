# MET-11 — 指标表自动生成

**日期**：2026-10-19
**状态**：DONE / PASS
**脚本**：`molmetal/scripts/paper_metric_tables.py`
**测试**：`molmetal/tests/test_paper_metric_tables.py`（19 个，全过）
**产物**：`metrics/paper_tables/`（10 张表 × md + csv + `manifest.json`）

## 为什么做

论文里每个数字都必须能追到 `metrics/by_metric/*.json` 的某个字段（§5.2 / S5）。
手工抄表有两个失败模式：抄错，以及在 artifact 重跑后忘记更新。两种失败都不会
报错，只会让正文和产物悄悄分叉。MET-11 把表变成**声明**：每个单元格写死
「文件名 + JSON pointer」，运行时解析；解析不到就是 `MISSING`，不是空白，也不
是上一次的旧值。

## 做法

- **声明式 spec**：每张表的每行是一组 `Cell`，`Cell(file, pointer, fmt)`。
  脚本里没有任何测量数字——只有路径。
- **pointer 解析**支持三种写法：
  - 普通路径：`/splits/doi/arms/ligand+both/r2`
  - 列表下标：`/curve/4`
  - 列表过滤：`/curve[coverage=0.5]/model/mean_regret`
  - 转义：`~1` → `/`，`~0` → `~`
- **派生量**：`delta` 单元格在同文件内做 `a − b`（例如每个协议的 R² 减去
  ligand 协议 R²），来源列同时写出两个 pointer。
- **缺失语义分三档**：
  - `MISSING` — 文件不存在 / pointer 解析失败 → 计入 `missing_total`，
    `--strict` 时退出码 1；
  - `-` — 该行按定义没有这一列（如 `full` 臂没有「相对 full 的 Δ」）；
  - `-`（optional）— 字段按设计不存在（如 `delta_mae_doi` 没有
    `target_effect`），记入 `optional_absent`，不阻断 `--strict`。
- **manifest**：每张表的行数、列名、输出路径、缺失清单，以及 **11 个源
  artifact 的 sha256**。表是快照，sha256 是判断快照是否过期的唯一依据。
- **重建幂等**：表体不含时间戳；`manifest.json` 只在内容真的变化时才更新
  `generated_utc`。同样的 artifact 重建两次，输出逐字节相同。

## MEASURED

| 表 | 行数 | 内容 | 缺失 |
|---|---:|---|---:|
| T1 main_predictive | 14 | 5-seed R²、四协议 R²、ΔR²、Δ 符号/MAE、regret/top-1、LOMO AUC | 0 |
| T2 generalization | 4 | 四协议的组数、评分数、R²、bootstrap CI、相对 ligand 的 ΔR² | 0 |
| T3 calibration | 8 | DOI/ligand × point/global/metal/difficulty 的 coverage、宽度、Winkler、PIT KS | 0 |
| T4 mechanism | 18 | 2 协议 × 9 臂的 R²、ΔR²、CI、p、特征数 | 0 |
| T5 decision | 8 | 2 协议 × 2 覆盖率 × model/global_best 的 regret、top-1、NB(0.5)、Holm p | 0 |
| T6 power | 7 | 每个主指标的观测值、CI、MDE、聚类数、所需 n、是否 powered | 0（1 optional） |
| T7 generation | 11 | yield、novelty 双口径、IntDiv1、SA、QED、金属合规、对接状态 | 0 |
| T8 data | 23 | 语料计数、删失、重复键、四协议 × 三种键的跨折率 | 0 |
| T9 seed_stability | 5 | 5 个臂的 R²/MAE 均值±SD、相对 tree 的 Δ、单侧 p | 0 |
| T10 external | 8 | 外部队列三键交集（金属 0 / 配体 144 / 细胞系 3）、三键行 0、NCI-60 可用 0 | 0 |

合计 **106 行**、**10 张表**、**11 个源 artifact**、**0 MISSING**、**1 optional
absent**（`msrs_power.json:/metrics/delta_mae_doi/target_effect`，该指标按设计
没有目标效应量）。

抽查（表值 == artifact 值）：

| 数字 | 表 | artifact 路径 |
|---:|---|---|
| 0.5828 | T1 | `msrs_seed_stability.json:/splits/canonical_grouped/arms/stack_foldwise/r2/mean` |
| 0.3321 | T1 | `msrs_generalization.json:/splits/doi/arms/ligand+both/r2` |
| 0.4503 | T1 | `msrs_generalization.json:/splits/scaffold/arms/ligand+both/r2` |
| 0.3739 | T1 | `msrs_generalization.json:/splits/year/arms/ligand+both/r2` |
| 0.6193 | T1 | `msrs_delta_metrics.json:/protocols/doi/thresholds/0.2091/sign_accuracy` |
| 0.5089 | T1 | `msrs_delta_metrics.json:/protocols/doi/thresholds/0.2091/mae_delta` |
| 0.1168 | T1 | `msrs_decision_curve.json:/protocols/doi/full_coverage/model/mean_regret` |
| 0.5600 | T1 | `msrs_decision_curve.json:/protocols/doi/full_coverage/model/top1_accuracy` |
| 0.5575 | T1 | `metal_lomo_auc_wasserstein.json:/value` |

## 诚实读数

1. **表正确 ≠ 结论正确。** 自动表只保证「表里的数字等于 artifact 里的数字」。
   如果 artifact 本身是旧的，表会把旧数字忠实地搬进论文。manifest 只记
   sha256，不判断新鲜度；重跑 artifact 后必须重跑本脚本，这一步没有自动化钩子。
2. **T1 混合了多次运行。** 5-seed 来自 `msrs_seed_stability`，四协议来自
   `msrs_generalization`，Δ 指标来自 `msrs_delta_metrics`，决策来自
   `msrs_decision_curve`。它们是同一语料、同一臂名，但不是同一次 fit。把它们
   放进一张表是阅读便利，不是「一个模型的一行结果」。
3. **四舍五入到 4 位。** 小于 1e-4 的差异在表里不可见；需要更细的对比时必须
   回读 artifact，不能用表。
4. **optional 缺失在表体里不可见。** T6 的 `delta_mae_doi` 目标列显示 `-`，
   只有 manifest 的 `optional_absent` 记录它「按设计不存在」。这是有意的：
   把「没有这一项」和「路径写错了」混为一谈会让 `--strict` 永远失败。
5. **表不含任何论文措辞。** 「边界」「限制」「不作为卖点」这些判断仍在
   `METRIC_GAP_ANALYSIS.md` 和 catalog 里；表只给数字和出处。两者的分工必须
   保持，否则表会变成第二份互相矛盾的叙事。
6. **没有自动重算。** 脚本不调用任何训练/评估入口，只读现有 artifact。这是
   刻意的：自动重算会把「生成表」变成「跑 6 小时任务」，也会让表的内容取决于
   运行环境。

## 复现

```bash
.venv/bin/python molmetal/scripts/paper_metric_tables.py --strict
# 或只重建部分表
.venv/bin/python molmetal/scripts/paper_metric_tables.py --only T1 --only T3
# 测试
OMP_NUM_THREADS=1 .venv/bin/python -m pytest molmetal/tests/test_paper_metric_tables.py -q
```

`--strict` 在任何 `MISSING` 单元格上退出码为 1；optional absent 不触发失败。
