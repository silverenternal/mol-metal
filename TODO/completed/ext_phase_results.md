# EXT 阶段结果 — 金属外推的机制、能力与外部验证

**日期**：2026-10-21
**范围**：EXT-1..EXT-8（`todo.json`）
**规则**：每个数字都有 artifact:jsonpath；PROJECTED 与 MEASURED 分开标注。
**门槛**：全部写死于 2026-10-21T12:00+08:00，早于任何 EXT 结果产生。

---

## 0. 一句话结论

**按预注册规则，本阶段落在 boundary path。** 但这个 boundary 比规划时设想的具体得多：

> 跨金属外推失败的主因不是"新金属的活性水平估不准"（那只占 5–10% 的平方误差），
> 而是**配体 SAR 本身几乎不迁移**（占 65–76%）。而且表观上的那点迁移，主要来自
> 同一配体在不同金属下被重复测量；把配体重叠去掉之后，五个金属里有四个的
> 秩相关掉到 0.05–0.21。
>
> 更关键的是：**这个领域目前不存在可用的独立外部 cohort**。最大的公开 Ir 数据集
> IrCytoToxDB 与 MSRS 共享 99.55% 的 DOI，匹配上的细胞格有 89.4% 数值完全相同。

一区路径（EXT-1 SAR_TRANSFERS + EXT-3 PASS + EXT-5 PASS）**被测量关闭**，
不是被判断关闭。

---

## 1. EXT-1 — 误差三分量分解

`metrics/by_metric/msrs_lomo_error_decomposition.json`
`molmetal/scripts/msrs_lomo_error_decomposition.py`

### 方法

Murphy 恒等式，对每个留出金属精确成立：

```
E[(p−y)²] = (μ_p−μ_y)² + (σ_p−σ_y)² + 2σ_pσ_y(1−ρ)
             offset²        scale²        shape
```

三臂：原始目标（复现旧结果）、每金属中心化目标（水平与尺度按构造消掉）、
中心化 + 每金属斜率。

### 结果

| 单元 | offset | scale | shape | 合并 ρ | 校准上限 ρ² |
|---|---:|---:|---:|---:|---:|
| potency / metal_out | 0.100 | 0.206 | **0.694** | +0.296 | 0.152 |
| potency / ligand_disjoint | 0.075 | 0.272 | **0.653** | +0.158 | 0.050 |
| cisplatin / metal_out | 0.045 | 0.198 | **0.757** | +0.269 | 0.108 |
| cisplatin / ligand_disjoint | 0.053 | 0.230 | **0.717** | +0.062 | 0.034 |

**主项是 shape（均值 0.705），不是 offset（均值 0.068）。**

分流判决（预注册：中心化后 R²>0.10 的金属 ≥3/5 → SAR_TRANSFERS）：

| 单元 | 判决 | 过线金属 |
|---|---|---|
| potency / metal_out | **PARTIAL** | Os +0.316、Re +0.106（2/5） |
| potency / ligand_disjoint | SAR_FAILS | Os +0.080（1/5） |
| cisplatin / metal_out | **PARTIAL** | Rh +0.144、Os +0.329（2/5） |
| cisplatin / ligand_disjoint | SAR_FAILS | Os +0.027（1/5） |

### 这条结果推翻了规划里的前提

`todo.json` 的 EXT-1 段落写的是"水平项和结构项是两回事，先拆开"，
隐含假设是水平项会占大头。**实测 offset 只占 5–10%，假设被证伪。**
这条已写进 artifact 的 `caveats` 与 `dominant_term`。

### 为什么"每金属中心化"的 R² 仍然低

因为预测是**压缩的**：sd_pred 0.31–0.54 vs sd_obs 0.64–0.96，slope 0.24–1.16
（五分之四低于 0.7）。中心化只去掉水平，不去掉压缩。所以
`calibration_ceiling_r2 = ρ²` 才是任何校准能到的上限，而它在诚实切分上只有
0.002–0.131。

---

## 2. EXT-2 — 静态金属描述符能否定水平

`metrics/by_metric/msrs_metal_descriptor_level.json`
`molmetal/scripts/msrs_metal_descriptor_level.py`

### 方法

16 列描述符：7 列 RDKit 周期表（可复现）、3 列 NIST/Pauling（已有）、
6 列定义性（group/period/d 电子数/自旋，标注为"非测量"）。
两种估计器：全列 ridge（alpha 嵌套 LOOCV）、单列 OLS（更公平，5 个点只支持 1 维）。
零假设：120 种 metal→mean 置换。

### 结果

| 终点 | ridge LOOCV R² | 最好单列 | 单列 LOOCV R² | 置换 p |
|---|---:|---|---:|---:|
| potency | +0.361 | `rvdw_ang` | +0.894 | **0.167** |
| cisplatin_ratio | −0.475 | `ie1_kj` | −0.410 | **0.133** |

**判决 KILL。**

判据不是"R² 没到 0.5"——单列 0.894 过了。判据是 todo 里自己写的 note：
"n=5 时 R²=0.5 是弱门槛；必须同时报 permutation p 与逐折预测"。
0.894 在 120 次置换里有 16.7% 能复现，且在第二个终点上反号。

### 完美水平的反事实

即使**完全知道**新金属的水平（oracle offset），R² 只提高 **+0.078**，
而且 4/5 金属仍然是负的。半数偏差修正只提高 +0.059。

---

## 3. EXT-3 — k-shot 校准曲线

`metrics/by_metric/msrs_kshot_calibration.json`
`molmetal/scripts/msrs_kshot_calibration.py`

### 门槛修订（记录在案）

原门槛是"k=50 恢复 ≥60% 的 level gap"。EXT-1 测出 level gap 只占 5–10%，
这个门槛的分母退化，失去意义。修订后的门槛在**脚本首次运行之前**写入：

> k=50 线性校准在 ligand-disjoint 切分上让 R² ≥ 0 的金属 ≥3/5。

### 结果

| 单元 | k=50 时 R²≥0 的金属 |
|---|---|
| potency / metal_out | 1/5（仅 Os） |
| potency / ligand_disjoint | **0/5** |
| cisplatin / metal_out | 2/5 |
| cisplatin / ligand_disjoint | **0/5** |

**判决 KILL。**

### 更强的发现：小 k 校准是负作用

offset-only 校准，ligand-disjoint 切分，中位 R² 在每个 k 上都是负的；
"比零样本更差"的重复占比：

| k | 5 | 20 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|
| Ru | 75% | 60% | 55% | 55% | 45% |
| Ir | 55% | 45% | 55% | 20% | 15% |
| Rh | 65% | 45% | 25% | 30% | 0% |
| Os | 95% | 75% | 70% | 85% | 70% |
| Re | 75% | 70% | 80% | 50% | 60% |

机制：零样本的偏差本来就小（offset 占 5–10%），用 k 个点去估它，
估出来的噪声比要修的偏差还大。**"给新金属测 20 个化合物就能用"这句话，
在这份数据上是错的。**

---

## 4. EXT-4 — 金属嵌入的可识别性

`metrics/by_metric/msrs_metal_embedding.json`
`molmetal/scripts/msrs_metal_embedding.py`

| 终点 | 配体总数 | ≥2 金属 | 占比 | 全 5 金属 |
|---|---:|---:|---:|---:|
| potency | 2,846 | 379 | 13.3% | **1** |
| cisplatin_ratio | 1,616 | 169 | 10.5% | **1** |

**判决 KILL。** 潜伏金属向量只能从跨金属配体估计；13% 的覆盖率下，
留出金属的载荷不可识别。留一金属的"其他四金属残差均值"预测 R² = **−2.19**。
另外，5 个金属中心化后秩上界是 4，所以 8 维嵌入按构造不可识别。

---

## 5. EXT-5 — 外部队列探针

`metrics/by_metric/msrs_external_cohort_probe.json`
`molmetal/scripts/msrs_external_cohort_probe.py`

### IrCytoToxDB（Scientific Data 2024，Zenodo 10.5281/zenodo.13120939，CC-BY-4.0）

| 指标 | 值 |
|---|---:|
| 行数 / DOI / 细胞系 | 2,694 / 222 / 127 |
| 年份 | 2008–2022 |
| 与 MSRS 的 DOI 重叠 | **221/222 = 99.55%** |
| 新 DOI | **1** |
| 匹配的 (DOI, 细胞系) 格 | 653 |
| 相关系数 | 0.9611 |
| 中位 \|Δlog10\| | **0.0** |
| \|Δ\|<0.01 的占比 | **89.4%** |

**判决 KILL。** 这不是外部队列，是同一份 curation 的 Ir 切片。

### 其余来源

| 来源 | 状态 |
|---|---|
| RuCytoToxDB | Zenodo 无存款；921 篇 vs MSRS 的 1,427 个 Ru DOI → 子集 |
| ChEMBL 29 本地库 | 0 个目标金属结构（MET-7） |
| PlatinAI | 只有 Pt |
| NCI-60 | 无 SMILES（MET-7） |
| DepMap | HTTP 200 但 Cloudflare Turnstile，下载得到 HTML |
| GDSC / CancerRxGene | HTTP 410 Gone |
| Cellosaurus API | 30 s 无响应 |
| GEO GSE36133 (CCLE) | 可达，但只有 RAW.tar（逐样本 CEL，需 RMA） |

**结论：这个任务目前不存在机器可读的独立外部 cohort。**
这是领域事实，不是本项目的数据缺口——它同时解释了为什么这个领域所有 ML 结果
都测在同一批 1,921 篇论文上。

---

## 6. EXT-6 — 外部队列检验

**BLOCKED。** EXT-5 没有产出可用 cohort，因此零样本/k-shot 的外部检验无法执行。
没有生成任何替代数字，也没有用 IrCytoToxDB 冒充外部检验。

---

## 7. EXT-7 — 细胞系特征可行性

`metrics/by_metric/msrs_cell_feature_probe.json`
`molmetal/scripts/msrs_cell_feature_probe.py`

| 指标 | 值 |
|---|---:|
| 磁盘上的注释文件 | **0** |
| 唯一细胞系 | 739 |
| top-46 覆盖 | 80.22% |
| 按构造无法匹配的系 | 5 个（HELA / HEK293 / A2780 / A2780CISR / A549CISR） |
| 这些系覆盖的行 | **9,253 / 42,167 = 21.9%** |

**判决 KILL，B2 永久关闭。** 两种独立失败：访问（Cloudflare/410/超时/CEL-only）
与身份（21.9% 的行来自 CCAR/DepMap 里根本不存在的系）。第二种不会因为重试而改善。

---

## 8. 分流结论

| 规则 | 条件 | 实测 | 结果 |
|---|---|---|---|
| zone1 | EXT-1 SAR_TRANSFERS + EXT-3 PASS + EXT-5 PASS + EXT-6 PASS | PARTIAL + KILL + KILL + BLOCKED | ✗ |
| zone2 | EXT-1 SAR_TRANSFERS + EXT-3 PASS，外部失败 | PARTIAL + KILL | ✗ |
| **boundary** | EXT-1 SAR_FAILS 或 EXT-3 KILL | EXT-3 = KILL | **✓ 落在这里** |

一区的两条腿（可部署的跨金属外推、独立外部验证）**都被测量关闭**。

---

## 9. 这个 boundary 比规划时想的强在哪

规划时的 boundary 是"我们没做到"。现在的 boundary 有三条机制：

1. **失败的位置是确定的**：shape 占 65–76%，offset 只占 5–10%。所以任何
   "校准一下就能用"的方案在算术上就不成立（完美水平只值 +0.078 R²）。
2. **表观迁移的来源是确定的**：metal_out 的 ρ=0.30 在去掉配体重叠后掉到 0.16；
   cisplatin 从 0.27 掉到 0.06。迁移主要来自同一配体被重复测量，不是化学。
3. **外部验证的缺失是领域性的**：99.55% DOI 重叠 + 89.4% 数值相同，
   说明公开语料只有一份 curation。这不是本项目的疏忽。

这三条都可以进正文，而且都不需要新数据。

---

## 10. 没有做的事（明确列出）

- 没有做 2023–2025 论文的人工整理（EXT-5 的两周预算）。这是唯一剩下的
  EXT-5 路径，也是唯一能解 B1 的动作。**没有编造任何化合物行。**
- 没有换学习器重跑 LOMO。EXT-3 的 KILL 是 LightGBM + 加性约束下的结论；
  换模型类是一次新的预注册检验，不是本轮的事。
- 没有碰 `paper/*` 与 `paper_a/*`。

---

## 11. 产物清单

| 包 | artifact | 脚本 |
|---|---|---|
| EXT-1 | `msrs_lomo_error_decomposition.json` | `msrs_lomo_error_decomposition.py` |
| EXT-2 | `msrs_metal_descriptor_level.json` | `msrs_metal_descriptor_level.py` |
| EXT-3 | `msrs_kshot_calibration.json` | `msrs_kshot_calibration.py` |
| EXT-4 | `msrs_metal_embedding.json` | `msrs_metal_embedding.py` |
| EXT-5 | `msrs_external_cohort_probe.json` | `msrs_external_cohort_probe.py` |
| EXT-6 | —（BLOCKED） | — |
| EXT-7 | `msrs_cell_feature_probe.json` | `msrs_cell_feature_probe.py` |
| EXT-8 | 本文件 + 台账 + 索引 | — |
