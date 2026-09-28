# R17 Tier-1 AUC 失败修复方案（必 claim 贵金属 AUC 领先）

**Date**: 2026-09-25
**Owner**: Hugo Lee + ultracode agents
**Constraint**: **必须** claim 贵金属 AUC 领先 — 这是 paper §4 的 headline

---

## 1. 失败诊断（honest 总结）

| Metric | Our | Gate | Tier-1 SOTA | Δ to gate | Δ to SOTA |
|---|---|---|---|---|---|
| **m1 Ru AUC** (MetalCytoToxDB) | 0.712 | ≥ 0.86 | Krasnov 0.81 | -0.148 | -0.098 |
| **m2 Pt MCF-7 AUC** (PlatinAI) | 0.584 | match | Rusanov per-line | unclear | -0.1 to -0.2 |
| **m2 Pt A549 AUC** | 0.630 | match | Rusanov per-line | unclear | -0.05 to -0.15 |
| **m2 Pt A2780 AUC** | **0.350** | match | Rusanov per-line | unclear | **worst than random** |
| **m2 Pt A2780cis AUC** | 0.648 | ≥ 0.85 | Rusanov per-line | -0.202 | -0.05 to -0.10 |

**真正问题**:
- m1 用 1 seed × 5 epochs → 远不够
- m2 A2780 0.350 = 数据/标签问题（agent 推荐 drop）
- m2 A2780cis 0.648 = 模型未针对 cisplatin-resistance 训练

**目标**:
- m1 ≥ 0.86 (beat Krasnov by +0.05)
- m2 A2780cis ≥ 0.85 (paper highlight)
- m2 A2780: drop or investigate (potential data bug)

---

## 2. 修复策略（4 阶段，2 周）

### Stage 1 — 解决 m1 Ru AUC（4-5 天）

**Strategy**: 把 1-seed × 5-epochs D-MPNN 升级到 **3-seed × 100-epochs + tmQM 预训练 + 多任务**

**Tasks**:

1. **数据扩增（1 天）**
   - 从 `.data/molmetal/MetalCytoToxDB.csv` 取 Ru subset
   - 联合 Ir / Rh / Re / Os 4 个 metal 的数据（5-way multi-task）
   - 总样本从 ~3K 扩到 ~15K

2. **D-MPNN 多任务架构（2 天）**
   ```python
   # molmetal/models/v4_multitask_dmpnn.py
   class V4MultiTaskDMPNN(nn.Module):
       """5-way multi-task D-MPNN for precious-metal cytotoxicity.
       
       Shared encoder (D-MPNN), 5 classification heads (Ru, Ir, Rh, Re, Os)
       Each head: 256 → 128 → 2 (active/inactive)
       Auxiliary: coordination geometry classifier (5 classes)
       """
       def __init__(self, hidden_dim=300, n_layers=3):
           self.encoder = DMPNNEncoder(hidden_dim, n_layers)
           self.metal_embed = nn.Embedding(5, 64)  # 5 metals
           self.heads = nn.ModuleDict({
               m: nn.Sequential(nn.Linear(hidden_dim + 64, 128), nn.ReLU(),
                                nn.Linear(128, 2))
               for m in ['Ru', 'Ir', 'Rh', 'Re', 'Os']
           })
           self.coord_head = nn.Linear(hidden_dim + 64, 5)  # coord geometry
   ```

3. **tmQM 预训练（1 天）**
   - 加载 tmQM 108k complexes
   - 在 V4 encoder 上做 masked atom prediction (BERT-style)
   - 预训练 ckpt → 用于 MetalCytoToxDB fine-tune

4. **3-seed × 100-epoch training（后台跑 12h）**
   - lr=1e-3, batch=128
   - AdamW + cosine annealing
   - Mixed precision (BF16)
   - Scaffold split (per Krasnov 2026 protocol)

**预期 Ru AUC**: 0.712 → **0.83-0.86**（多任务 + tmQM 预训练 + 3-seed）

**Gate target**: ≥ 0.86
**SOTA target**: beat Krasnov 0.81 by +0.05

### Stage 2 — 解决 m2 Pt A2780cis AUC（4-5 天）

**Strategy**: **cisplatin-resistance-specific fine-tune** + **多任务 + tmQM 预训练 + 数据平衡**

**Tasks**:

1. **A2780 数据诊断（1 天）**
   - 检查 `molmetal/data/platinai_4cell.csv` 中 A2780 标签分布
   - 检查 scaffold split 是否 leakage
   - **若数据有 bug → drop A2780，重跑 3-cell-line**（MCF-7 / A549 / A2780cis）
   - **若数据 OK → 调训练策略**

2. **cisplatin-resistance features（2 天）**
   ```python
   # molmetal/molmetal_lam/lam_chem/cisplatin_resistance.py
   def compute_resistance_features(mol: Chem.Mol) -> dict:
       """Features specific to cisplatin-resistance prediction.
       
       - Lipophilicity (logP)
       - Reduction potential (proxy by Hammett-like substituent constants)
       - Glutathione affinity (proxy by thiol count)
       - DNA adduct geometry (proxy by N-donor count + trans angle)
       - Trans-influence proxies (cis vs trans isomerism)
       """
       return {
           'logP': Descriptors.logP(mol),
           'thiol_count': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[SH]'))),
           'n_donor_count': len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH,NH2,NH3+]'))) + len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]'))),
           'cis_trans_pattern': is_cis_or_trans(mol),  # for Pt square_planar
       }
   ```

3. **resistance-aware multi-task（2 天）**
   - 3-cell-line multi-task (drop A2780 if data bug)
   - 加 resistance features to D-MPNN feature vector
   - tmQM 预训练 (复用 Stage 1 ckpt)
   - **per-cell-line heads + 共享 backbone**

4. **5-seed × 100-epoch training（后台 12h）**
   - focal loss for class imbalance
   - LR warmup 200 + cosine
   - BF16 mixed precision

**预期 A2780cis AUC**: 0.648 → **0.82-0.88**（cisplatin-resistance features + tmQM pretrain + balanced training）

**Gate target**: ≥ 0.85

### Stage 3 — A2780 数据问题处理（1 天）

**Strategy**: 决定 A2780 是 drop 还是 fix

**Tasks**:
1. 如果 A2780 标签错误 / scaffold leak → drop A2780，paper 报告 3-cell-line
2. 如果 A2780 数据 OK 但模型 fail → 增加 train epochs + class re-weighting
3. 写 paper §4.5 honest negative framing: "Mol-Metal achieves strong AUC on MCF-7/A549/A2780cis but near-chance on A2780 (parental), suggesting A2780 may have structural label noise or our scaffold split inadvertently isolates it"

### Stage 4 — Paper §4 abstract 重写（1 天）

**Strategy**: 真正 claim Tier-1 AUC 领先

**Tasks**:
1. 重写 paper §4 abstract: claim Ru AUC ≥ 0.86 + Pt A2780cis AUC ≥ 0.85
2. 重写 paper §4.1 Tier-1 panel figure (per-cell-line bar + Ru vs Krasnov)
3. 更新 TODO/02_metrics/METRICS_TARGETS.md Table 2 — Tier-1 row 用实测值
4. 更新 `mol-metal-cfm-training` SKILL — citation Krasnov/Rusanov
5. 更新 `mol-metal-anticancer` SKILL — Tier-1 achievement section

---

## 3. 总时间线

```
Day 1   : Stage 1 数据扩增 + D-MPNN 多任务架构
Day 2-3 : Stage 1 tmQM 预训练 + 3-seed 启动（后台）
Day 4   : Stage 2 A2780 数据诊断
Day 5-6 : Stage 2 cisplatin-resistance features
Day 7   : Stage 2 multi-task fine-tune 启动（后台）
Day 8   : Stage 3 A2780 数据问题处理
Day 9-10: Stage 4 paper §4 abstract 重写
```

**关键后台任务**: Stage 1 训练 12h + Stage 2 训练 12h = 24h GPU

---

## 4. 数据 + 模型细节

### 4.1 数据增强

```python
# molmetal/scripts/build_metacytotox_multitask.py
import pandas as pd

# 加载 MetalCytoToxDB
df = pd.read_csv('.data/molmetal/MetalCytoToxDB.csv')

# 5-way multi-task
metals = ['Ru', 'Ir', 'Rh', 'Re', 'Os']
df_5metal = df[df['metal'].isin(metals)].copy()

# Active/inactive label
df_5metal['active'] = (df_5metal['ic50_nm'] < 10000).astype(int)

# Scaffold split (Bemis-Murcko)
from rdkit.Chem.Scaffolds import MurckoScaffold
df_5metal['scaffold'] = df_5metal['smiles'].apply(
    lambda s: MurckoScaffold.MurckoScaffoldSmiles(mol=s, includeChirality=False)
)
train, val, test = scaffold_split(df_5metal, frac=[0.7, 0.1, 0.2])
```

### 4.2 tmQM 预训练

```python
# molmetal/scripts/pretrain_v4_on_tmqm.py
import pandas as pd
import torch

# 加载 tmQM
df_tmqm = pd.read_csv('.data/molmetal/tmQM/tmqm_parsed.csv')

# 任务: masked atom prediction (BERT-style)
def mask_atoms(smiles: str, mask_ratio: float = 0.15) -> tuple:
    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()
    n_mask = max(1, int(n_atoms * mask_ratio))
    masked_indices = random.sample(range(n_atoms), n_mask)
    masked_mol = copy.deepcopy(mol)
    for idx in masked_indices:
        masked_mol.GetAtomWithIdx(idx).SetAtomicNum(1)  # mask as H
    return Chem.MolToSmiles(masked_mol), masked_indices

# 预训练
model = V4MultiTaskDMPNN()
optimizer = AdamW(model.parameters(), lr=1e-3)
for step in range(10000):
    batch = sample_batch(df_tmqm, batch_size=128)
    pred_atoms = model(batch['masked_smiles'])
    loss = CrossEntropy(pred_atoms, batch['true_atoms'])
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

### 4.3 cisplatin-resistance features

```python
# molmetal/molmetal_lam/lam_chem/cisplatin_resistance.py
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

def compute_resistance_features(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(10)
    
    features = []
    # 1. logP (lipophilicity)
    features.append(Descriptors.MolLogP(mol))
    # 2. TPSA (polar surface area)
    features.append(Descriptors.TPSA(mol))
    # 3. Thiol count (GSH affinity proxy)
    features.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts('[SH]'))))
    # 4. N-donor count (DNA binding proxy)
    features.append(len(mol.GetSubstructMatches(Chem.MolFromSmarts('[NH,NH2]'))))
    # 5. Aromatic ring count (DNA intercalation)
    features.append(rdMolDescriptors.CalcNumAromaticRings(mol))
    # 6. Molecular weight
    features.append(Descriptors.MolWt(mol))
    # 7. Rotatable bonds
    features.append(Descriptors.NumRotatableBonds(mol))
    # 8. H-bond donors
    features.append(Descriptors.NumHDonors(mol))
    # 9. H-bond acceptors
    features.append(Descriptors.NumHAcceptors(mol))
    # 10. Pt square_planar flag (cis vs trans)
    pt_pattern = Chem.MolFromSmarts('[Pt](-*)(-*)(-*)(-*)')
    features.append(1.0 if mol.HasSubstructMatch(pt_pattern) else 0.0)
    return np.array(features)
```

### 4.4 focal loss for class imbalance

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()
```

---

## 5. 风险 register

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| **3-seed × 100-epoch 仍 < 0.80** | Medium | High | 加 multi-task (5-way) + tmQM pretrain → 期望 +0.05 |
| **A2780 数据 bug 不能 fix** | Medium | Medium | drop A2780，paper 报 3-cell-line |
| **cisplatin-resistance features 不相关** | Low | High | feature importance analysis + 退回到 pure D-MPNN |
| **tmQM pretrain OOM** | Low | Low | gradient accumulation + bf16 |
| **GPU Blackwell 12h 不够** | Low | Medium | reduce epochs to 50 |

---

## 6. Success Criteria

| Metric | Before | Target (after Stage 1+2) | Gate |
|---|---|---|---|
| **m1 Ru AUC** | 0.712 | **≥ 0.86** | beat Krasnov 0.81 by +0.05 |
| **m2 Pt MCF-7 AUC** | 0.584 | ≥ 0.70 | match Rusanov per-line |
| **m2 Pt A549 AUC** | 0.630 | ≥ 0.75 | match Rusanov per-line |
| **m2 Pt A2780 AUC** | 0.350 | **drop or ≥ 0.65** | if data OK, recover |
| **m2 Pt A2780cis AUC** | 0.648 | **≥ 0.85** | paper highlight |

**如果 Stage 1 + 2 都 succeed**, paper §4 可以 claim:
> "Mol-Metal achieves Ru AUC 0.86+ on MetalCytoToxDB (beating Krasnov 2026 baseline by +0.05), and Pt A2780cis AUC 0.85+ on PlatinAI 4-cell-line panel (matching Rusanov 2025 cisplatin-resistant cell line)."

---

## 7. ultracode 编排

### ultracode #1: Stage 1 + Stage 2 (并行)
- Stage 1: 3-seed × 100-epoch multi-task D-MPNN
- Stage 2: cisplatin-resistance features + A2780 诊断 + 3-cell-line fine-tune

### ultracode #2: Stage 3 + Stage 4 (并行)
- Stage 3: A2780 数据 fix or drop
- Stage 4: paper §4 abstract 重写

### 单个 ultracode 总 effort: ~2 weeks wall time + 24h GPU

---

## 8. 与 R18 的关系

| R18 主推 | 是否帮 Stage 1+2 |
|---|---|
| **W-Flow** (100× faster ODE) | ❌ 不相关（这是分子生成加速，不是 property prediction） |
| **MacroGuide PH** (ring topology) | ❌ 不相关（不是 cytotoxicity prediction） |
| **Coord-Aware EGNN** | ✅ **能直接帮** Stage 1 Ru AUC（coord geometry classifier 作为 auxiliary loss） |
| **Quotient-Space Diffusion** | ❌ 不相关 |
| **Score-Repellent MC** | ❌ 不相关 |

**结论**: **不要等 R18**。R17 Tier-1 AUC fix 不依赖 R18 数学定理迁移。Coord-Aware EGNN 是唯一相关项，但 Stage 1 直接用 multi-task D-MPNN 已经够。

**顺序**:
1. 现在（不依赖 R18）: 启动 Stage 1 + 2（2 周）
2. R18 P0-1 SE(3) SB 主推（3 周）可以并行
3. R18 P1-1 multi-PH + Coord-Aware GNN（2 周）可以增强

---

## 9. 下一步

1. ✅ 写本规划 `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md`（本文）
2. ⏳ 启动 ultracode #1: Stage 1 (Ru AUC) + Stage 2 (A2780cis) 并行
3. ⏳ 启动 ultracode #2: Stage 3 (A2780 数据) + Stage 4 (paper abstract) 并行
4. ⏳ 等 2 周，看 Ru AUC + A2780cis AUC 是否达成
5. ⏳ 如果达成，重写 paper §4 abstract + update TODO/02_metrics/METRICS_TARGETS.md Table 2
6. ⏳ 如果不达成，drop Tier-1 claim 改用 honest negative framing
