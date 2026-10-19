# FIX-M2 — 强模型与 D-MPNN 的融合

日期：2026-10-19
状态：DONE-MIXED（回归通过，决策层无变化）

## 为什么做

FIX-M1 把 D-MPNN 接进 MSRS，但只到 parity（0.5031 vs LightGBM 0.5012），
NEG-20 关闭了「换模型类」这条轴。可两个模型的残差相关只有 0.82——它们错的不是同一批行。
如果融合能抬高绝对 R²，主模型仍然留在链路里，而且论文可以诚实地说「金属感知的图模型
与指纹树模型互补」，而不是「D-MPNN 打赢了 LightGBM」。

## 做了什么

- `molmetal/scripts/msrs_hybrid.py`：读 FIX-L5（`results/msrs_joint_representation/oof.npz`）
  与 FIX-M1（`results/msrs_dmpnn_surface/oof.npz`）的行级 OOF。两者是同一 21,404 行、
  同一 GroupKFold(5) 划分；`align()` 逐列核对标签、金属、配体、细胞系、DOI，不一致就报错。
- 五臂：
  - `lgb_tree` = FIX-L5 的 `ligand+both`
  - `gnn_metal_state` = FIX-M1 的最佳臂
  - `blend_equal` = 0.5/0.5，**零参数**，主结果
  - `blend_foldwise` = 单权重，在其他四折上拟合
  - `stack_foldwise` = `a*tree + b*gnn + c`，同样折外拟合
- 另加 `blend_rank`（队列内秩平均）送决策层；该臂不参与 R² 表，因为秩不在 y 的尺度上。
- `msrs_joint_representation.py` 新增 `--oof-out`，把四臂 × 双 split 的逐行预测落盘。

## 结果（MEASURED）

### 回归面

| split | tree | gnn | equal | foldwise | stack |
|---|---|---|---|---|---|
| canonical_grouped | 0.5556 | 0.5031 | **0.5727** | 0.5740 | **0.5807** |
| fragment_grouped | 0.4856 | 0.4263 | 0.5050 | 0.5089 | 0.5128 |

- 等权：canonical +0.0171（配对 p=4.8e-12），fragment +0.0194（p=1.9e-09）。
- 堆叠：canonical +0.0251（p=5.1e-34），fragment +0.0272。
- 残差相关 0.818（canonical）/ 0.821（fragment）——确实不是同一个模型。
- 折内单权重在 5 折间稳定：0.596–0.712（均值 0.651），不是某一折的偶然。
- 两个 split 同向，这一点比单 split 的显著 p 更重要。

### 决策层（424 cohort，canonical，regret 越低越好）

| scorer | regret | top-1 |
|---|---|---|
| global_best | 0.1241 | 0.566 |
| lgb_tree | **0.1186** | 0.566 |
| gnn_metal_state | 0.1219 | 0.566 |
| blend_equal | 0.1225 | 0.566 |
| blend_foldwise | 0.1224 | 0.559 |
| stack_foldwise | 0.1223 | 0.561 |
| blend_rank（tie-broken） | 0.1186 | 0.566 |

所有融合臂对树模型的 DOI 聚类 Holm p = 1.0。

## 一个必须记录的假象（DEF-7 / NEG-22）

秩平均臂最初给出 regret **0.1099**、top-1 **0.6203**，看起来是决策层的突破。它不是。

- w=0.5 时 424 个队列里有 **158 个**在最小值处二选一并列。
- `regret_matrix` 用 `idxmin`，并列时取**行序第一行**。
- 在这 158 个队列里，首行恰好是真最优的比例是 **65.2%**（行序不是随机的）。
- 把权重挪到 0.4 或 0.6，收益立刻消失（0.1213 / 0.1186）。
- 加确定性 tie-break（并列时归树模型）后，秩融合与树模型**逐位相同**：0.1186 / 0.5660。

所以：决策层的表面增益是评估器 artifact。已在 `rank_blend` 内加 1e-9×tree_rank 的
确定性 tie-break，并在 `msrs_hybrid.json` 的 `diagnostics.rank_tie_check` 里留下证据。
`regret_matrix` 本身未改，以免动到历史连续 scorer 的数字；DEF-7 记录了影响范围。

## 诚实读数

1. **融合是真的**：零参数等权已经把强模型抬高 0.0171，两个 split 都显著，权重跨折稳定。
2. **决策层是假的**：所有融合臂 regret 0.1223–0.1225，比树模型 0.1186 更差；Holm p=1.0。
3. **这是 NEG-17 的第四次复现**，也是最强的一次——换表示、换模型类、换融合方式，
   队列内金属排序都没有离开单一强模型。
4. **论文线继续冻结**：解冻判据要求 regret<0.1198 且 Holm p<0.05，当前最好 0.1186 但 p=1.0。

## 产物

- `molmetal/scripts/msrs_hybrid.py`
- `molmetal/tests/test_msrs_hybrid.py`（8 个测试：权重可恢复、折外拟合不看测试折、tie-break、对齐拒绝）
- `metrics/by_metric/msrs_hybrid.json`
- `metrics/by_metric/msrs_hybrid_decision.json`
- `results/msrs_hybrid/oof.npz`
- `results/msrs_joint_representation/oof.npz`（新增）
- 两个 artifact 门禁 PASS（10 个 dual-split artifact / 10 个粒度一致 artifact）
