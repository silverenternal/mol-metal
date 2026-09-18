# 模型设计:D-MPNN + EGNN 混合

## 目标

在 MetalCytoToxDB Ru/Ir 子集上,AuC > Krasnov 2026 (Ru 0.81, Ir 0.73)。

## 总体架构

```
SMILES  ──[RDKit + ETKDGv3]──>  3D conformer
                                  │
                                  ▼
                          ┌──── Atom Node ────┐
                          │  one-hot Z +       │
                          │  charge + spin +   │
                          │  dative flag       │
                          └─────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        2D D-MPNN           3D EGNN           Charge/spin cond
        (Morgan FP)         (positions)       (e.g., +2 state)
              │                   │
              └───── concat ──────┘
                       │
                          ▼
              cross-attention MLP fusion
                          │
                          ▼
                per-atom features → sum-pool
                          │
                          ▼
                    pIC50 / activity head
```

## 各模块设计

### 1. Atom encoder
```python
# node features (per atom)
z_onehot = nn.Embedding(100, 64)  # atomic number
charge_proj = nn.Linear(1, 16)    # formal charge
spin_proj = nn.Linear(1, 16)      # spin multiplicity
dative_flag = nn.Embedding(2, 16) # is metal-ligand bond
coord_proj = nn.Linear(3, 64)    # 3D position (invariant: use ||x||² or RBF)
atom_feat = concat([z_onehot, charge, spin, dative, coord_rbf])  # ~160-dim
```

### 2. 2D D-MPNN stream (per atom + edge)
- 输入:SMILES → bond graph → 消息传递
- 边类型:**covalent** vs **dative**(参考 tmGNN-XAI)
- 消息:`m_ij = MLP(h_i || h_j || edge_type_oh)`
- 聚合:sum over neighbours
- 输出:per-atom 128-dim

### 3. 3D EGNN stream
- 输入:坐标 + 边图
- EGNNLayer 消息:`m_ij = MLP(h_i || h_j || ||x_i - x_j||², ||x_i - x_j||)`
- 向量贡献:`(x_i - x_j) * phi_ij`,sum 聚合到 dst
- 输出:per-atom 128-dim + per-atom 3D velocity(用于 equivariance 检查)
- 复用 `models/velocity_net.py::EGNNLayer` + `models/_scatter.scatter_sum`

### 4. Cross-attention fusion
```python
h_2d = dmpnn_out         # (N, 128)
h_3d = egnn_out          # (N, 128)
# Cross-attention: 2D attends to 3D positions, 3D attends to 2D graph
fused = CrossAttention(h_2d, h_3d)  # (N, 128)
```

简化版:concatenation + MLP
```python
fused = MLP(concat([h_2d, h_3d]))  # (N, 128)
```

### 5. Readout
- sum-pool per-atom features → graph-level
- concat with: metal one-hot + oxidation state + counterion embedding
- MLP → pIC50 (回归) 或 activity logit (分类)

## 训练目标

### 任务 1: 回归 (pIC50)
```python
loss = F.mse_loss(pred_pic50, true_pic50)
```

### 任务 2: 分类 (active / inactive)
```python
logit = pred_pic50_to_logit  # 阈值 10 μM
loss = F.binary_cross_entropy_with_logits(logit, active_label)
```

### 多任务 (可选)
```python
total_loss = α * reg_loss + (1-α) * cls_loss
```

## 损失函数细节

```python
class MetalCytotoxLoss(nn.Module):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
    def forward(self, pred_pic50, true_pic50, active_label, mask=None):
        # mask: 1 = real sample, 0 = padded
        reg = F.mse_loss(pred_pic50, true_pic50, reduction='none')
        cls = F.binary_cross_entropy_with_logits(
            pred_pic50_to_logit(pred_pic50), active_label, reduction='none')
        per_sample_loss = self.alpha * reg + (1-self.alpha) * cls
        return (per_sample_loss * mask).sum() / mask.sum().clamp(min=1)
```

## 训练配置

```yaml
model:
  hidden_dim: 128
  n_layers_dmpnn: 3
  n_layers_egnn: 3
  n_rbf: 32
  edge_cutoff: 6.0   # Å, 贵金属复合物金属-配体键 ~ 2.0 Å
  rbf_max: 8.0       # Å

train:
  lr: 1.0e-4
  epochs: 50
  batch_size: 32
  warmup_steps: 200
  grad_clip_norm: 1.0
  ema_decay: 0.999
```

## 评估指标

### 内部评估
- **ROC-AUC** (主指标,Krasnov 用的)
- **PR-AUC** (数据不平衡时更稳)
- **Hit Rate @ top 5%** (虚拟筛选场景)
- **Pearson r** / **RMSE** (回归)
- **AUROC per metal**(Ru, Ir, Pt,...)

### 外部评估(发 paper 用)
- 时间切分:**train pre-2024, test post-2024** → hit rate
- 化学切分:**train, test on Tanimoto-dissimilar (70% threshold)**
- Leave-one-cell-line-out
- Leave-one-metal-out

## 与 Krasnov 2026 的对比维度

| 维度 | Krasnov | 我们 |
|---|---|---|
| 表示 | Morgan FP / RDKit desc | D-MPNN + EGNN + 3D |
| 3D 几何 | ❌ | ✅ (ETKDGv3 嵌入) |
| Dative 键 | ❌ | ✅ special edge type |
| 任务 | 二分类 (active/inactive) | 回归 + 分类 + 多任务 |
| 反离子 | ❌ | ✅ (探索) |
| 化学 split | 随机 | 严格 Tanimoto / 时间 |
| 评估指标 | AUC | AUC + Hit Rate + AUPRC |

## 复用项目里已有代码

- `models/_scatter.py::scatter_sum` — EGNN scatter,带 autograd
- `models/velocity_net.py::EGNNLayer` — 已有,小改即可加 dative edge type
- `models/encoder.py::MolEncoder` — D-MPNN-like,可借鉴 message passing
- `models/conditioner.py::Conditioner` — charge/spin conditioning
- `flow_matching/loss.py::ConditionalFlowMatchingLoss` — multi-task 模式

## 新代码估计

| 文件 | 行数估计 |
|---|---|
| `molmetal/data/cytotox.py` | 300 |
| `molmetal/data/featurize.py` (RDKit 3D + 描述符) | 200 |
| `molmetal/models/dmpnn.py` | 250 |
| `molmetal/models/fusion.py` (D-MPNN + EGNN hybrid) | 200 |
| `molmetal/models/loss.py` | 100 |
| `molmetal/scripts/train.py` | 250 |
| `molmetal/scripts/eval.py` (LOO + 时间 split) | 200 |
| **总计** | ~1500 |

## 关键技术风险

1. **3D 几何生成质量**:RDKit ETKDGv3 对金属复合物可能产生畸形构象
   - 缓解:**用 MMFF 力场 refine**(类似 PlatinAI 论文);**可视化检查**首批样本
2. **金属 SMILES 表示**:`[Pt](N)(N)Cl` 在 RDKit 里要特殊处理
   - 缓解:用 `rdkit.Chem.MolFromSmiles` + `Chem.AllChem.EmbedMolecule`,失败则跳过
3. **EGNN 训练不稳定**:我们已知需要 (a) 几何归一化,(b) grad clip,(c) EMA
   - 缓解:复用 MolFlow-Triton 的 warmup + cosine + EMA 调度

## 与 MolFlow-Triton 项目的关联

**复用**:Triton scatter kernel、EGNN layer、autograd-wrapped scatter、ema setup
**新增**:D-MPNN 流、cross-attention fusion、MetalCytoToxDB loader、3D conformer generation

**项目结构建议**:
```
/home/hugo/codes/try_triton_on_rocm/
├── (现有 MolFlow-Triton)
├── molmetal/                  ← 新项目
│   ├── data/
│   ├── models/
│   ├── scripts/
│   └── configs/
├── TODO/
├── data/
│   └── MetalCytoToxDB.csv     ← 共享数据目录
└── checkpoints/molmetal/
```

避免污染 MolFlow-Triton 的 main 目录 — `molmetal/` 是独立子项目。