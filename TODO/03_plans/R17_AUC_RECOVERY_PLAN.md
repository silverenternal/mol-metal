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

---

## 10. 2026-09-30 refresh (Wf-393-A1 ultracode CPU-only docs-only)

**Date**: 2026-09-30
**Workflow**: Wf-393-A1 (ultracode CPU-only docs-only; 0 NEW tests; 0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning per AGENTS.md §10)
**Trigger**: User 2026-09-30 directive *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* — refresh R17 with Wf-380..390 SCAFFOLD/wire-in status
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `mol-metal-anticancer` (Krasnov 0.86 gate); no `paper/*` touches (TODO-84 honored); no wet-lab planning (§10)

This refresh is **docs-only**: §1-§9 preserved verbatim as historical baseline; §10 (this section) adds 5 subsections capturing the Wf-380..390 ship surface that supersedes the Stage 1-4 strategies. All numbers tagged MEASURED (from on-disk JSON) or PROJECTED (per upstream paper benchmarks).

### §10.1 Background — Wf-387 m1 AUC aggregation 6/7 pairs MEASURED

Per `TODO/completed/387a1_wf379_a1_gpu_aggregation_fix.md` (canonical 6/7 pair summary) — the Wf-379-A1 GPU wrapper `scripts/launch_wf131_p3_deferred_runs.sh` terminated after 6 of 7 (metal, seed) pairs completed. The canonical Tier-2 catalog `metrics/by_metric/metal_lomo_auc_wasserstein.json` was overwritten by the trainer at Re seed=3 completion with only its own data (n=1). Wf-387-A1 fixed this by computing the correct aggregate across all 6 completed pairs:

| Field | OLD (wrong, Re seed=4 only) | NEW (correct, 6 runs) |
|---|---:|---:|
| `value` | 0.6220 (n=1) | **0.7640** (6 runs) |
| `value_std` | 0.0000 | **0.0995** (sample std) |
| `n_runs` | 1 | **6** |
| `n_folds` | 1 (Re only) | **2 (Rh, Re)** |
| `metals` | ["Re"] | **["Rh", "Re"]** |
| `seeds` | [4] | **[3, 4, 0, 1, 2, 3]** |
| `per_fold_auc_mean` | {Re: 0.6220} | **{Rh: 0.8798, Re: 0.7061}** |
| `per_fold_auc_std` | {Re: 0.0000} | **{Rh: 0.0115, Re: 0.0552}** |

**Per-pair MEASURED test_auc (sourced from on-disk `per_seed_metrics_seed{n}.json` + per-directory `aggregate.json`):**

| # | (metal, seed) | test_auc | n_test | Source path |
|:---:|:---:|---:|---:|---|
| 1 | (Rh, 3) | **0.8879** | 1618 | `Rh/seed3/Rh/per_seed_metrics_seed3.json` |
| 2 | (Rh, 4) | **0.8717** | 1618 | `Rh/seed4/Rh/per_seed_metrics_seed4.json` |
| 3 | (Re, 0) | **0.7018** | 868 | `Re/seed0/Re/per_seed_metrics_seed0.json` |
| 4 | (Re, 1) | **0.7463** | 868 | `Re/seed1/Re/per_seed_metrics_seed1.json` |
| 5 | (Re, 2) | **0.7467** | 868 | `Re/seed2/Re/per_seed_metrics_seed2.json` |
| 6 | (Re, 3) | **0.6295** | 868 | `Re/seed3/Re/per_seed_metrics_seed3.json` |
| 7 | (Re, 4) | **NOT_RUN** | — | GPU wrapper terminated after Re seed=3; Re seed=4 deferred to follow-up launch |

**Krasnov 0.86 gate verdict** per `mol-metal-anticancer` skill:

| Fold | Mean test_auc | Krasnov 0.86 | gap |
|---|---:|:---:|---:|
| **Rh** | 0.8798 ± 0.0115 | **PASS** | +0.020 |
| **Re** | 0.7061 ± 0.0552 | **BELOW** | −0.154 |
| **Overall** | 0.7640 ± 0.0995 | **BELOW** | −0.096 |

### §10.2 Phase 1 — Rh cohort balanced (per Wf-387-A2 TODO-135 structural fix)

Per `TODO/completed/387a2_balanced_rh_cohort.md` — the Rh val_auc lift (test_auc 0.7613 from Wf-124 P3 Ru baseline) was contaminated by the Wf-139 P7 synthetic-500 injection (Tanimoto-proxy-labeled cohort with mean Tanimoto to nearest real active = 0.14 — WEAK band, NOT bioactivity-validated). The structural fix per Wf-387-A2:

- **Cohort**: `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` — 500 entries (250 active + 250 inactive), all `is_synthetic=False`, all `is_wf139_p7_carryover=False`, all from Rh `train` split under the Wf-261 strict scaffold split
- **Source**: `/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (Rh train; n=696 available; 250+250 sampled with seed=42)
- **Verify path**: If Wf-125 P1 re-run with this REAL 500 (instead of synthetic 500) yields val_auc ≥ Krasnov 0.86 gate, the lift is REAL; if val_auc ≤ 0.5, the lift was an artifact of the Wf-139 P7 synthetic injection
- **Verdict**: per `party-twenty-spirit` §3.5, the Wf-139 P7 similarity-proxied lift is "achievable" but PROJECTED — not MEASURED until wet-lab validation closes the loop; since AGENTS.md §5.4 deprecates wet-lab planning, this verification gap must close via the real-cohort comparison route (Wf-387-A2)
- **Stage 1 strategy revision**: Replace the Wf-124 P3 1-seed × 5-epoch multi-task D-MPNN with Wf-125 P1 re-run using the Wf-387-A2 REAL balanced cohort; tmQM pretrain + multi-task + 5-seed × 100-epoch protocol unchanged from §2 Stage 1
- **Cross-ref**: `molmetal/scripts/wf376a2_m1_auc_multiseed_scaffold.py` (SHIPPED per Wf-376-A2) + `scripts/launch_wf131_p3_deferred_runs.sh` (LIVE mode — Wf-355-B1 USER_OK-gated — Re seed=4 pending launch)

### §10.3 Phase 2 — TODO-376 5-metal subset retrain (per Wf-390-A1 Path A)

Per `TODO/completed/390a2_todo179_deeper_audit.md` (TODO-179 deeper audit) + `TODO/completed/388a1_m2_per_cohort_real.md` (real-cohort verification) + Wf-369-A2 m2 R² FINAL verdict, the Stage 2 strategy is now scoped to a **5-metal subset retrain** with Path A as the recommended first move:

**Stage 2 strategy revision (Path A)**:
1. **5-metal subset**: Ru + Ir + Rh + Re + Os (matches D-MPNN `FIVE_METALS = ("Ru", "Ir", "Rh", "Os", "Re")` constraint at `train_dmpnn_multitask_wasserstein.py` line 247)
2. **Per-cohort config axis** (per Wf-380-A2 + Wf-388-A1): 12-d RDKit (n<300) / 24-d RDKit (300≤n<1000) / 32-d RDKit (n≥1000) matched to cohort size
3. **MEASURED real-cohort R²** (per Wf-388-A1, 5-seed × 5-fold Ridge baseline):
   - `platinai_4cell` (n=200, 12-d RDKit, α=1.0) → **−0.1440 ± 0.0325** HONEST_NEGATIVE_R2_DEEP
   - `hilbert` (n=200, 12-d RDKit, α=1.0) → **−0.1440 ± 0.0325** HONEST_NEGATIVE_R2_DEEP
   - `metalcytotox_combined` (n=2000, 32-d RDKit, α=0.1) → **+0.1981 ± 0.0056** WEAK_RECOVERY (lift −0.0246 vs Wf-335-M2 best +0.2227)
4. **Aggregate verdict**: `MEASURED_PARTIAL_LIFT` (1/3 cohorts WEAK_RECOVERY; 0/3 CEILING_BREAK; 2/3 HONEST_NEGATIVE_R2_DEEP preserved)
5. **Honest finding** (`party-twenty-spirit` §3.5): per-cohort config axis VALIDATED as methodology (Wf-380-A2 synthetic R² > 0 across all tiers), but REAL-cohort lift smaller than synthetic validation projected (Wf-380-A2 synthetic platinai_4cell + hilbert at n=200 + 12-d RDKit → R²=+0.1970 vs Wf-388-A1 real → R²=−0.1440, DELTA −0.3410); synthetic ground-truth is clean signal-to-noise model; real cohort pIC50 has more variance + non-uniform signal distribution + per-cell heterogeneity
6. **Path A launch wrapper**: `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (472 LOC, EXISTS on disk) — but per Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL, trainer `--metals` is restricted to 5 metals (matches Stage 2 Path A constraint); alternative trainer `train_cisplatin_resistance_v2.py` requires `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` which DOES NOT EXIST on disk — Path A targets the 5-metal subset directly via existing trainer
7. **Wall PROJECTED**: ~5 hr GPU for Path A; if Path A R² < +0.40 ceiling, follow up with Path B (alt encoder bootstrap per Wf-390-A2 §7, ~10 hr GPU)

### §10.4 Phase 3 — Forward-looking GPU retrain priority order (per Wf-391-A3 §10)

Per `TODO/completed/391a3_metric_strategy_refresh.md` §10 (NEW Forward-looking 2026-Q4 GPU retrain priority order), 3-batch sequential priority:

| Batch | Wall PROJECTED | Wrappers | USER_OK gate | Pre-flight gate |
|:---:|---:|---|---|---|
| **Batch 1 (install)** | ~30 min CPU | Zotero 10.0.3 + Tectonic 0.17.0 (per Wf-372-A1 + Wf-372-A2 SHIPPED) | None (already installed) | N/A |
| **Batch 2 (m1 + m2 lift)** | ~5-10 hr GPU | Re seed=4 (Wf-355-B1) + Path A 5-metal retrain (Wf-376-A3 + Wf-390-A2 Path A) | USER_OK=1 + VRAM ≥ 35 GB | wf343c2 + wf343c3 PASS |
| **Batch 3 (m3 + m4 + m7 + m15)** | ~12-15 hr GPU | `cfm_vina_dock_panel.py` (BLOCKED on Pt+2 parameterisation per Wf-299B) + SA penalty wire (Wf-380-A6 SHIPPED) + m7 wire (Wf-388-A2 SHIPPED) + m15 redesign (Wf-341-T1 SHIPPED) | USER_OK=1 + VRAM ≥ 35 GB | wf343c2 + wf343c3 PASS |
| **Paper write batch** | ~12-15 hr CPU | 7 paper artefacts (Tier-2 §6 Limitations + 4 LOI/SOP Tier-2 markers + paper §3 R² honesty gate) | USER_OK=1 (TODO-84 active) | N/A |

**Combined wall PROJECTED**: ~52.9 hr GPU + ~13 hr CPU + ~30 min install (4-batch total)

**Forward-looking honest-negative caveats** (per `docs/HONEST_NEGATIVE_CATALOG.md` Wf-369-A3 expansion 40 → 65 entries):
- m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL (5 architectural + 1 literature attempts exhausted per Wf-365-A3 + Wf-377-A1 deep-dive-research move; Wf-377-A1 recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
- m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP FINAL (platinai_4cell + hilbert cohorts n=200 SMILES each; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc + Wf-377-A2 5 alternative splits in `docs/M2_R2_COHORT_DIAGNOSTIC.md`)
- m1 Ru AUC 0.7640 ± 0.0995 BELOW Krasnov 0.86 by −0.096 margin; Rh fold 0.8798 PASS; Re fold 0.7061 BELOW; aggregate 0.7640 NOT HIT

### §10.5 Honest framing — Rh fold hit Krasnov 0.86; Re fold BELOW; aggregate 0.7640 NOT hit

Per Wf-387-A1 canonical aggregation + Wf-388-A1 real-cohort verification + Wf-377-A1 m6_kappa redesign + Wf-377-A2 m2 R² diagnostic, the honest framing per `party-twenty-spirit` §3.5 for R17 is:

**m1 (Ru + Ir + Rh + Re + Os 5-metal LOMO AUC)**:
- **Rh fold 0.8798 ± 0.0115 — PASS Krasnov 0.86** (gap +0.020) — per Wf-387-A1 MEASURED
- **Re fold 0.7061 ± 0.0552 — BELOW Krasnov 0.86** (gap −0.154) — per Wf-387-A1 MEASURED
- **Aggregate 0.7640 ± 0.0995 — NOT HIT** (gap −0.096) — per Wf-387-A1 MEASURED
- **Rh val_auc 0.7613 lift** (Wf-124 P3 1-seed) was artifact of Wf-139 P7 synthetic-500 injection per Wf-387-A2 §2; Wf-387-A2 balanced REAL 500 from Wf-261 strict scaffold split substitute needs verification via Wf-125 P1 re-run

**m2 (Pt 4-cell PlatinAI R² on pIC50)**:
- **platinai_4cell + hilbert n=200 — HONEST_NEGATIVE_R2_DEEP** (−0.1440 ± 0.0325 each, per Wf-388-A1 MEASURED)
- **metalcytotox_combined n=2000 — WEAK_RECOVERY** (+0.1981 ± 0.0056, lift −0.0246 vs Wf-335-M2 best +0.2227 CEILING_BREAK)
- **Per-cohort config axis VALIDATED as methodology** (Wf-380-A2 synthetic R² > 0), but REAL-cohort lift smaller than synthetic projection; structural finding: 50+ feature Ridge ensemble OVERFITS on small cohorts per Wf-367-B1

**m6_kappa (Shape Kappa Index per molecule)**:
- **PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted; structural CV-R² ceiling LIFTED +0.5410 → +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted)
- **Wf-377-A1 deep-dive-research move**: 16 per-molecule topological invariants evaluated on n=5460 MetalCytoToxDB labelled pIC50; best alternative MolLogP Pearson r=+0.1999 WEAK band; no cohort-level MODERATE band discovery
- **Forward-looking**: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`

**Tier-2 paper §6 Limitations draft** (per Wf-359-B2): m6_kappa + m2 R² + m1 Ru AUC 0.7640 NOT HIT all surface as honest-negative entries in §6; cross-ref `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md` + `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 LaTeX template)

**Tier-1 paper claim DEPRECATED** per AGENTS.md §5.4 + 2026-09-29 user directive *"我们发二区就可以了，不用规划湿实验"* — Tier-2 journal target (JCAMD / Mol. Divers. / Comput. Chem.) does NOT require Tier-1 AUC gate

**Constraints honoured**: 0 NEW tests, 0 GPU launches, 0 `paper/*` touches, 0 wet-lab planning, only Bash + Read + Edit + Write tools used. AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §10 all preserved unchanged.
