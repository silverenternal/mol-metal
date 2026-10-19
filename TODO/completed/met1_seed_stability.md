# MET-1 — 强模型 5-seed 稳定性

**日期**：2026-10-19
**状态**：DONE / PASS
**脚本**：`molmetal/scripts/msrs_seed_stability.py`
**测试**：`molmetal/tests/test_msrs_seed_stability.py`（4 个）
**产物**：`metrics/by_metric/msrs_seed_stability.json`
**原始 OOF**：`results/msrs_seed_stability/{tree,gnn}_seed{0..4}.{json,npz}`

## 为什么做

之前主表上唯一的强模型数字是 **单 seed 20261018** 的 R² 0.5807。单 seed 无法回答两件事：

1. 这个数字是模型能力还是那一次初始化的运气；
2. 融合权重（树 vs GNN）跨 seed 是否稳定——如果权重在 seed 间乱跳，融合就是伪增益。

## 做法

- 固定 **同一个 GroupKFold(5) 折划分**（按配体键），5 个 seed 只改变模型随机性。
  **因此 std 度量的是训练方差，不是划分方差。** 划分方差由 MET-2 的四协议回答。
- 树：`msrs_joint_representation.py --seed k`（LightGBM `random_state`）。
- GNN：`msrs_dmpnn_surface.py` 的 `metal_state` 单臂，`configs/msrs_dmpnn_surface_seed{0..4}.yaml`
  （`train.seed = k`，其余超参与 FIX-M1 完全一致）。
- 融合复用 FIX-M2 的 `msrs_hybrid.py`：等权 / 折内单权重 / 折内线性堆叠。
- 统计：seed 间**配对单侧 t 检验**（`alternative="greater"`）。不用 5 点 bootstrap
  （已知反保守，见 memory `project_five_seed_bootstrap_anticonservative`）。

## MEASURED（5 seed，mean ± std）

**canonical_grouped**

| 臂 | R² | MAE | RMSE | Δ vs tree | p（单侧配对 t） |
|---|---|---|---|---|---|
| lgb_tree | 0.5555 ± 0.0026 | 0.3855 | 0.5083 | — | — |
| gnn_metal_state | 0.5032 ± 0.0248 | 0.4023 | 0.5372 | **−0.0522** | 0.996 |
| blend_equal | 0.5759 ± 0.0116 | 0.3745 | 0.4964 | +0.0205 | 0.0071 |
| blend_foldwise | 0.5775 ± 0.0101 | 0.3745 | 0.4955 | +0.0221 | 0.0034 |
| **stack_foldwise** | **0.5828 ± 0.0102** | **0.3702** | **0.4924** | **+0.0273** | **0.0016** |

**fragment_grouped**

| 臂 | R² | MAE | RMSE | Δ vs tree | p（单侧配对 t） |
|---|---|---|---|---|---|
| lgb_tree | 0.4832 ± 0.0018 | 0.4182 | 0.5481 | — | — |
| gnn_metal_state | 0.4331 ± 0.0164 | 0.4340 | 0.5740 | −0.0501 | 0.999 |
| blend_equal | 0.5071 ± 0.0075 | 0.4067 | 0.5352 | +0.0239 | 0.0013 |
| blend_foldwise | 0.5101 ± 0.0055 | 0.4060 | 0.5336 | +0.0269 | 2.9e-4 |
| **stack_foldwise** | **0.5137 ± 0.0048** | **0.4029** | **0.5317** | **+0.0304** | **1.1e-4** |

残差相关（树 vs GNN）：canonical 0.804 ± 0.003，fragment 0.820 ± 0.008 —— 两模型错在
不同行，融合有结构性空间。

## 判定

判据：最优融合臂 mean > tree mean、std ≤ 0.02、≥ 4/5 seed 更好。**三项全部满足**
（0.5828 > 0.5555，std 0.0102，5/5）。

## 诚实读数

1. **单 seed 0.5807 站得住**：5-seed 均值 0.5828，std 0.0102。之前那个数字不是运气。
2. **std 只有 0.0102，但它不是划分方差。** 折划分被固定了。跨论文/跨 scaffold 的
   方差会大得多（旧特征上 DOI 0.165 vs 配体 0.25 已经说明），MET-2 才能回答。
3. **GNN 单模在所有 5 个 seed、两个 split 上都显著差于树**（−0.052 / −0.050，0/5 seed
   更好）。**不能写「D-MPNN 打赢 LightGBM」。** 能写的只有「融合比任一单模好」。
4. **融合增益 0.027 在 fragment 上更稳**（std 0.0097 → 0.0053，p 0.0016 → 1.1e-4）。
   canonical 的 seed 3 只有 +0.0128，是五个里最小的。
5. **折内单权重在 seed 间落在 0.53–0.75，偏向树**。权重不稳但方向一致，这解释了
   为什么 `blend_equal`（固定 0.5）稳定地略低于 `blend_foldwise`。
6. **这是内插性能，不是泛化性能。** 主表若要写「泛化」，必须等 MET-2 的 DOI/scaffold/
   year 数字；在此之前只能写「同分布 5-seed 稳定性」。

## 对主表的影响

- R² 0.5828 ± 0.0102（canonical）/ 0.5137 ± 0.0048（fragment）现在满足 S2（5-seed）、
  S4（效应量 + CI + 配对检验）、S5（可追溯）。
- 仍缺 S1（同协议外部基线）与 S3（互不重叠的第二类划分）。**所以它还是 medium，
  不是 strong。** MET-2 补 S3，MET-7 补 S1。
