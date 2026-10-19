# FIX-M4 — 反对称 pairwise reranker

日期：2026-10-19
状态：DONE-NEGATIVE（配体分组下 +0.02 但未过门槛；整篇论文留出后归零）

## 问题

FIX-M3 给出两条事实：

1. 队列内只有 `队列×金属交互` 能改变排序，金属主效应只占 3.4%；
2. 当前所有模型（含 FIX-M2 的全部融合臂）都停在这个主效应水平。

所以下一步只能改目标函数：不再预测 y，而是直接学「同一队列里金属 a 是否优于金属 b」。

FIX-E3 曾试过 `pair_delta`，但实现有实质缺陷：

- 配对特征写成 `[X[a], cell_a, metal_a code, metal_b code]` —— **不对称**。金属 B 只贡献
  一个身份编码，模型看不到 B 的氧化态、电荷、d 电子数、配位环境；
- 用 500 棵梯度提升树拟合约 700 个训练对 —— 容量远大于样本。

## 方法

**反对称表征。** 交换 a、b 必须同时取负特征与目标：

```python
d_onehot = onehot[b] - onehot[a]                    # pair_identity
parts   += [state[b] - state[a], coord[b] - coord[a]]  # pair_state
ctx      = 0.5 * (ligand[a] + ligand[b])
parts   += outer(ctx, d_onehot)                     # pair_full
```

`f(b) - f(a) = -[f(a) - f(b)]` 由构造保证，所以同一对不会给出两个答案。

**闭式 ridge + 内层 GroupKFold 选 α。** α ∈ {0.1, 1, 10, 100, 1000}，只在训练折内部
用 3 折选择。选中的 α 几乎总是 100 或 1000 —— 与「容量是敌人」一致。

**只对多金属队列建对。** `g["m"].nunique() < 2` 直接跳过，避免把同一金属的重复行
当成零 delta 训练。

**评估。** 424 个可评估队列 / 834 个有序对；regret 按队列逐条算再取均值（不是把 851
行当一个队列）。DOI 聚类单侧 t 检验 + 2000 次 bootstrap。

**三个协议。** `fragment_grouped` / `canonical_grouped` 是双 split 门禁要求的两个
配体分组协议。第三个 `doi_grouped` 整篇论文留出（FIX-E4 的部署条件）是在看到
canonical 结果之后才加的，明确标为 post hoc，不参与主判定。

## 结果（MEASURED）

### 主表（mean within-cohort regret，越低越好）

| split | main_effect | cell_effect | pair_identity | pair_state | pair_full |
|---|---|---|---|---|---|
| canonical | 0.1242 | 0.1231 | 0.1182 | **0.1069** | 0.1125 |
| fragment | 0.1199 | 0.1295 | 0.1200 | **0.1069** | 0.1107 |

`pair_state` 在两个 split 上给出**完全相同**的 0.1069。不是巧合：金属状态块与配体
无关，队列内排序只由状态差决定，两个 split 的分折边界不同但打分函数在各自折外一致。

### 显著性（vs main_effect）

| split | arm | Δ regret | t | p（单侧，DOI 聚类） | bootstrap 95% |
|---|---|---|---|---|---|
| canonical | pair_identity | +0.0069 | +0.878 | 0.1918 | [−0.0074, +0.0223] |
| canonical | **pair_state** | **+0.0208** | **+1.632** | **0.05393** | [−0.0014, +0.0491] |
| canonical | pair_full | +0.0089 | +0.340 | 0.3676 | [−0.0412, +0.0583] |
| fragment | pair_identity | −0.0002 | −1.000 | 0.8393 | [−0.0006, 0.0000] |
| fragment | **pair_state** | **+0.0179** | **+1.651** | **0.05204** | [−0.0006, +0.0426] |
| fragment | pair_full | −0.0007 | −0.025 | 0.5100 | [−0.0566, +0.0559] |

### top-1 / pairwise accuracy（canonical）

| arm | top-1 | pairwise acc |
|---|---|---|
| main_effect | 0.5660 | 0.5492 |
| cell_effect | 0.5590 | 0.5396 |
| pair_identity | 0.5802 | 0.5635 |
| pair_state | 0.5896 | 0.5731 |
| pair_full | **0.6203** | **0.6043** |

`pair_full` 的 top-1 最高（0.6203）但 regret 只有 0.1125 且检验不显著（p=0.368）：
它把最难的队列推对了，却在容易的队列上犯错，净值不如 `pair_state`。

### 事后补充：整篇论文留出（DOI-grouped）

上面的两个 split 都是**配体分组**，同一篇论文的行会同时出现在训练和测试折里。
FIX-E4 的部署条件是整篇论文留出。这个协议是看到 canonical 结果之后才加的，
因此明确标为 post hoc，不参与主判定。

| arm | mean regret | top-1 | pairwise acc |
|---|---|---|---|
| main_effect | **0.1199** | 0.5755 | 0.5588 |
| cell_effect | 0.1358 | 0.5472 | 0.5276 |
| pair_identity | 0.1496 | 0.4811 | 0.4628 |
| pair_state | 0.1441 | 0.5377 | 0.5204 |
| pair_full | 0.1568 | 0.5307 | 0.5108 |

| arm | Δ regret vs main_effect | t | p（单侧） | bootstrap 95% |
|---|---|---|---|---|
| pair_identity | **−0.0440** | −2.500 | 0.9924 | [−0.0801, −0.0139] |
| pair_state | −0.0011 | −0.068 | 0.5270 | [−0.0322, +0.0271] |
| pair_full | −0.0244 | −0.817 | 0.7915 | [−0.0851, +0.0307] |

**整篇论文留出后，没有一个配对臂优于主效应。** `pair_identity` 显著更差
（−0.044，区间不含 0），`pair_state` 退回到与主效应无法区分（−0.0011），
`pair_full` 同样为负。


## 判定

**NEGATIVE。** 预注册判据是 canonical 上最优配对臂 vs `main_effect` 的 DOI 聚类单侧
p < 0.05。实测 **p = 0.05393**（fragment 0.05204），差 0.004。

更关键的是事后加的 DOI-grouped 协议：**整篇论文留出后，配对模型不再有任何增益**
（`pair_state` −0.0011，p=0.527；`pair_identity` 显著更差 −0.044，p=0.992；
`pair_full` −0.0244）。因此 canonical/fragment 上的 +0.02 不能解释为「金属状态差
可泛化」，只能解释为**同一篇论文内部的偏移**——模型学到的是这篇论文的金属排序，
换一篇就不成立。

必须同时记录的四件事：

1. **方向一致，但只在配体分组内。** 两个配体 split 同向（+0.0208 / +0.0179），
   整篇论文留出后归零。
2. **幅度一致，但只在配体分组内。** `pair_state` 在两个配体 split 上给出同一个
   0.1069；DOI split 上是 0.1441，比主效应 0.1199 更差。
3. **预注册门槛未过。** 61 个 DOI 上的单侧检验，t≈1.63；bootstrap 区间下界略低于 0。
4. **DOI 协议是事后加的。** 它不改变主判定，但它是唯一一个模拟部署条件的协议，
   其负结果比主判定的「差 0.004」更重要。

`pair_state` vs `cell_effect` 在配体分组下显著（p=0.0154 / 0.0265），但 `cell_effect`
是更弱的基线、且是事后挑选的比较，不构成结论。

## 诚实读数

1. **这不是四条轴里第一个「接近显著」的决策层结果，而是一个被更强协议否定的结果。**
   表示（FIX-L*）、模型类（FIX-M1）、融合（FIX-M2）都是零；目标函数在配体分组下
   给出 +0.02，但整篇论文留出后归零。**四条轴仍然全部为零。**
2. **`pair_identity` 在 DOI split 上显著更差（−0.044）**，说明在未见论文上按金属身份
   做配对排序会主动伤害推荐；主效应反而更稳。这是一个比「无增益」更强的负结果。
3. **0.1069 是均值，不是可部署的推荐器。** 中位 regret 为 0，改善集中在少数队列。
4. **`pair_full` 的配体上下文交互没有兑现。** 两个配体 split 上都退回主效应
   （p=0.368 / 0.510），DOI split 上更差。~700 个训练对不足以学配体特异的金属偏好，
   与 FIX-M3 的交互方差不可识别一致。
5. **`pair_identity` 在配体 split 上基本等于主效应**（+0.0069, p=0.19），在 DOI split
   上显著更差。**配体 split 的增益全部来自状态差，且不跨论文。**

## 下一步（未执行）

- **不要**在同一批 61 个 DOI 上继续换特征或调 α 去追 p<0.05。DOI split 已经说明
  这条路是过拟合。
- 若还要做决策层，需要的是**新论文**（新 DOI），且必须用 DOI-grouped 协议做预注册
  单次检验，而不是配体分组。
- 当前证据支持把主张收缩到回归面 + 边界（open_decisions 的 (c)）。

## 产物

- `molmetal/scripts/msrs_pairwise_reranker.py`
- `molmetal/tests/test_msrs_pairwise_reranker.py`（9 个测试：反对称性 ×3 臂、自身配对为零、
  跳过同金属/单例、ridge 恢复线性关系、`pair_accuracy` 检测符号翻转）
- `metrics/by_metric/msrs_pairwise_reranker.json`
