# Baseline 复现计划

## 目标

在我们 GPU 上复现 Krasnov 2026 的 baseline AUC,作为后续 3D 增强模型的对比基线。

## Baseline 1: Morgan FP + XGBoost (Krasnov 2026 复现)

### 数据
- 来源:MetalCytoToxDB.csv,只取 Ru 子集
- 过滤:Time ≥ 24h, IC50_Dark_value not null, IC50 > 0.01
- 标签:`active = (IC50 < 10 μM)`
- 划分:80/10/10 随机 split(seed=42)

### 特征
- Morgan FP (radius=2, nBits=2048)
- 可选:加 RDKit 描述符(MolWt、LogP、TPSA、HBA、HBD 等)

### 模型
```python
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=negative_count / positive_count,
    random_state=42,
)
```

### 评估
- ROC-AUC(主指标)
- PR-AUC
- per-metal 表

### 预期
- Ru AUC ~ 0.81 (复现 Krasnov)
- Ir AUC ~ 0.73
- 多金属 AUC ~ 0.78

## Baseline 2: Morgan FP + RandomForest (Krasnov 论文里也有)

- 同上,但用 sklearn `RandomForestClassifier(n_estimators=500)`
- 预期:Ru AUC ~ 0.78-0.80(略低于 XGBoost)

## Baseline 3: D-MPNN with PyG (自家实现)

### 消息传递
```python
# Directed edge message (Chemprop 风格)
def message(h_i, h_j, edge_type):
    m_ij = nn.Sequential(
        nn.Linear(2*h_dim + edge_type_dim, h_dim),
        nn.SiLU(),
    )(concat([h_i, h_j, edge_type_onehot]))
    return m_ij
```

### 聚合(sum over incoming edges)
```python
agg_i = scatter_add(src=messages, index=dst, dim=0)
```

### 实现
- 不引入 chemprop 包,自家写 ~250 行
- 用项目的 `models._scatter.scatter_sum`

### 训练
- batch=32, lr=1e-3, epochs=50
- EMA + warmup + grad clip(同 MolFlow-Triton)

### 预期
- Ru AUC ~ 0.82-0.85 (略超 Morgan + XGBoost)

## Baseline 4: EGNN-only (用项目已有 EGNN)

### 输入
- atom features + 3D coords (RDKit ETKDGv3)
- edge: covalent + dative

### 训练
- 同 MolFlow-Triton setup,但 loss 改成 BCE
- 50 epochs

### 预期
- Ru AUC ~ 0.80-0.83(应该比 Morgan 好,但可能比 D-MPNN 略差 — 因为 3D 对小分子贡献有限)

## Baseline 5: D-MPNN + EGNN fusion (我们的目标模型)

### 融合
- per-atom 特征 concat → MLP → graph-level → pIC50 head

### 训练
- 同上 + 多任务(pIC50 + active/inactive)

### 预期目标
- Ru AUC ≥ 0.86(超过 Krasnov 0.81)
- Ir AUC ≥ 0.78

## 评估脚本 (`molmetal/scripts/eval.py`)

### 输出格式
```python
{
  "metal": "Ru",
  "n_samples": 9421,
  "auc": 0.864,
  "ap": 0.851,
  "rmse_pic50": 0.45,
  "hit_rate_top5pct": 0.62,
  "per_cell_line": {"A549": 0.84, "HeLa": 0.89, ...},
}
```

### Split 模式
- 随机:80/10/10
- 时间:pre-2024 / post-2024
- 化学:Tanimoto > 0.7 dissimilar
- LOMO: leave-one-cell-line-out
- LOMO-metal: leave-one-metal-out (only for multi-metal model)

### 报告生成
- Markdown table(每个 split × 每个 metal)
- 图表(loss curve、AUC distribution、confusion matrix)

## 与 Krasnov 2026 数字对照表

| 模型 | Ru AUC | Ir AUC | Pt AUC(参考 MB Finder) |
|---|---|---|---|
| Morgan FP + XGBoost (Krasnov 2026) | 0.81 | 0.73 | N/A |
| Morgan FP + XGBoost (our reproduce) | TBD | TBD | TBD |
| D-MPNN (2D) | TBD | TBD | TBD |
| EGNN (3D) | TBD | TBD | TBD |
| **D-MPNN + EGNN fusion (target)** | **≥0.86** | **≥0.78** | TBD |

## 验证流程

1. 跑 baseline 1 (XGBoost) → 验证能 reproduce ~0.81
2. 跑 baseline 3 (D-MPNN) → 应该 ≥0.82
3. 跑 baseline 4 (EGNN) → 应该 ≥0.80
4. 跑 target model (D-MPNN+EGNN) → 应该 ≥0.86
5. 若 target < baseline 3 → 复盘 fusion 设计

## 数据注意事项

- **细胞系偏差**:某些 cell line 数据很多(HeLa, A549),有些很少 — 注意 per-cell-line 模型的样本量
- **重复样本**:同一 (compound, cell_line) 可能有多个 IC50 取均值
- **重复金属**:同一 (compound, metal, ligands) 在不同 cell_line 是不同样本,保留
- **inactive 主动平衡**:Ru subset active ratio — 决定用 weighted BCE 还是 focal loss

## 下一步

1. 实现 `molmetal/scripts/baseline.py`
2. 写报告生成函数
3. 跑 5 个 baseline 数字
4. 输出 baseline_report.md(在 TODO/experiments/ 下)

## 代码估计

| 文件 | 行数 |
|---|---|
| `molmetal/scripts/baseline.py` (4 baselines) | 400 |
| `molmetal/scripts/eval.py` | 300 |
| `molmetal/scripts/report.py` | 150 |
| **总计** | 850 |