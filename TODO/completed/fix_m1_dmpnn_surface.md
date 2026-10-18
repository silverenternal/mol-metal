# FIX-M1 — 自研 D-MPNN + metal_graph 接进 MSRS

日期：2026-10-19
状态：DONE-NEGATIVE（验收字面通过，实质 parity）

## 做了什么

- `molmetal/scripts/msrs_dmpnn_surface.py`：把 `molmetal.models.metal_aware_dmpnn`
  与 `molmetal.data.metal_graph.build_metal_graph` 接到 MSRS 的 21,404 行上。
  图从 `ligand_canonical`（完整配合物）构建，金属作为追加节点 + `cap` 条配位边。
- 三臂，只改条件向量与注入位置：
  - `ligand_cell`：金属盲，cell one-hot，readout 注入（对照）
  - `metal_identity`：金属 one-hot + 节点/边，message 注入
  - `metal_state`：再加氧化态/电荷/d 电子/配位几何
- 同折同评估：`folds_for` 的 GroupKFold(5)，双 split（FIX-E1）；
  LightGBM `complex` 基线在**同一脚本、同一折**内重算。
- `configs/msrs_dmpnn_surface_cpu0.yaml`（3 epoch 接线 smoke）与
  `configs/msrs_dmpnn_surface_gpu0.yaml`（60 epoch 正式跑）。
- OOF 落盘 `results/msrs_dmpnn_surface/oof.npz`；
  `molmetal/scripts/msrs_dmpnn_decision.py` 用同一 OOF 跑决策层 regret（FIX-M1b）。

## 结果（MEASURED）

| split | lgb_complex | ligand_cell | metal_identity | metal_state |
|---|---|---|---|---|
| canonical_grouped | 0.5012 | 0.4553 | 0.4726 | **0.5031** |
| fragment_grouped | 0.4368 | 0.4107 | 0.4237 | 0.4263 |

- 金属分解在 GNN 内复现：identity − blind = +0.0173，state − identity = +0.0305。
- 最佳臂只比同折 LightGBM 高 +0.0019；per-ligand MAE 配对 p=0.60。
- 同一 canonical split 上 FIX-L5 的树模型是 0.5556 → D-MPNN 不是更强的表示。
- 400 配体 pilot 曾给出 +0.0705；全量只剩 +0.0019。**pilot 结论作废。**

决策层（424 cohort，canonical，regret 越低越好）：

| scorer | regret | top-1 |
|---|---|---|
| global_best | 0.1241 | 0.566 |
| lgb_complex | 0.1212 | 0.538 |
| metal_identity | **0.1190** | 0.592 |
| metal_state | 0.1219 | 0.566 |

对 global_best 与 lgb 的 DOI 聚类单侧 t 检验：**Holm p 全为 1.0**。

## 读法

1. 架构轴关闭。表示层（L1–L5）把树模型推到 0.5556，模型类（M1）只到 0.5031。
   瓶颈不是消息传递容量，也不是「金属没进图」。
2. `metal_state` 的 R² 最好，但决策 regret 反而比 `metal_identity` 差 ——
   NEG-17（回归 ≠ 排序）的第三次复现。
3. 决策层仍然没有赢过「训练折金属均值」。论文解冻判据
   （regret < 0.1198 且 DOI 聚类 Holm p < 0.05）不满足，论文线保持冻结。

## 复现

```bash
# 正式跑（GPU0，约 45 min）
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
  .venv/bin/python molmetal/scripts/msrs_dmpnn_surface.py \
  --config configs/msrs_dmpnn_surface_gpu0.yaml

# 决策层（秒级，读 OOF）
.venv/bin/python molmetal/scripts/msrs_dmpnn_decision.py
```

## 产物

- `metrics/by_metric/msrs_dmpnn_surface.json`
- `metrics/by_metric/msrs_dmpnn_decision.json`
- `results/msrs_dmpnn_surface/oof.npz`
- `molmetal/tests/test_msrs_dmpnn_surface.py`（11 tests）
