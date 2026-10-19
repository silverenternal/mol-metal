# FIX-M4 — 反对称 pairwise reranker

日期：2026-10-19
状态：DONE-NEGATIVE-BOUNDARY（方向一致、幅度一致、**未过预注册门槛**）

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
行当一个队列）。DOI 聚类单侧 t 检验 + 2000 次 bootstrap。两个 split 都报。

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

## 判定

**NEGATIVE（边界证据）。** 预注册判据是 canonical 上最优配对臂 vs `main_effect` 的
DOI 聚类单侧 p < 0.05。实测 **p = 0.05393**（fragment 0.05204），差 0.004。

必须同时记录的三件事：

1. **方向一致。** 两个独立 split 都指向同一结论：金属状态差带来 +0.018～0.021 regret。
2. **幅度一致。** 0.0179 vs 0.0208，且 `pair_state` 在两个 split 上给出同一个 0.1069。
   这不是一个 split 的噪声。
3. **仍不足以称为发现。** 61 个 DOI 上的单侧检验，t≈1.63；bootstrap 区间下界略低于 0。
   按预注册判据这是负结果，不能改口径。

`pair_state` vs `cell_effect` 是显著的（p=0.0154 / 0.0265），但 `cell_effect` 是更弱的
基线，且这是事后挑选的比较，不构成主结论。

## 诚实读数

1. **这是四条轴里第一次出现「接近显著」的决策层结果。** 表示（FIX-L*）、模型类（FIX-M1）、
   融合（FIX-M2）都是零；目标函数是第一个给出同向、同幅度、跨 split 稳定信号的轴。
2. **0.1069 仍然是均值，不是可部署的推荐器。** 中位 regret 为 0，说明改善集中在少数
   队列；没有任何一个臂把分布整体左移。
3. **不能宣称「打败树模型 0.1186」。** 判据未过，且两臂的差在 DOI 层面只有 t≈1.63。
4. **`pair_full` 的配体上下文交互没有兑现。** 它在两个 split 上都退回到主效应水平
   （p=0.368 / 0.510），说明 ~700 个训练对不足以学配体特异的金属偏好。这与 FIX-M3 的
   交互方差估计（0.09 上限、大部分不可识别）一致。
5. **`pair_identity` 基本等于主效应。** canonical 上 +0.0069（p=0.19），fragment 上
   −0.0002。**所有增益来自状态差，不是来自「用配对目标重训」。** 这是本波次最有信息量
   的一条：目标函数本身不是杠杆，金属化学的差才是。

## 下一步（未执行）

- 若要把 p 从 0.054 推过 0.05，需要更多 DOI 或更多队列，而不是更多模型容量。
- `pair_state` 的特征只有 5 个金属状态列 + one-hot 差，已接近可解释上限；再堆特征会
  重演 `pair_full` 的过拟合。
- 若要继续，应先做**预注册的重复**：固定 `pair_state`、固定 α 选择规则，在新增 DOI
  上单次检验，而不是在同一批数据上换特征再试。

## 产物

- `molmetal/scripts/msrs_pairwise_reranker.py`
- `molmetal/tests/test_msrs_pairwise_reranker.py`（9 个测试：反对称性 ×3 臂、自身配对为零、
  跳过同金属/单例、ridge 恢复线性关系、`pair_accuracy` 检测符号翻转）
- `metrics/by_metric/msrs_pairwise_reranker.json`
