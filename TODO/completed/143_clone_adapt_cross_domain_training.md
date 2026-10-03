# TODO-143: Clone + adapt cross-domain training infrastructure

**Date**: 2026-09-27
**Status**: 🆕 NEW pending
**Priority**: **P0** — User challenge 2026-09-27 "先调研有没有对应的现成的项目,先 clone 下来看看能不能直接适配"
**Parent**: TODO-142 cross-domain recommendation + TODO-141 §2.1 audit
**Target journal**: **JCIM IF 5.6** (Q2 中科院二区)
**Skill alignment**: `party-twenty-spirit` §3.5 + `mol-metal-mechanism-mapping`

---

## §0 Why this TODO exists

User 核心指令(2026-09-27):
> "最重要的一点,现在 todo 目录里不是有很多建议或者说我们按理来说应该要做很多东西,现在调研一下有没有对应的现成的项目,先 clone 下来看看能不能直接适配到我们的项目"

**承认**:之前我一直在 propose 自建实现,没有先调研现成项目。正确做法是**先 clone 现成项目,评估能否 drop-in / 适配**。

**已调研 + clone 的项目**:

| Repo | ⭐ | 用途 | 状态 |
|---|---:|---|---|
| `mtl_uncertainty_loss` (elna4os) | 1 | Kendall 2018 MTL loss | ✅ cloned + ready |
| `DANN` (fungtion) | 949 | Gradient Reversal Layer | ⚠️ Py2.7 old |
| `DANN_py3` (fungtion) | 152 | DANN Py3 fork | ✅ cloned + ready |
| `Sheaf-FRL` (SPAICOM) | 0 | Sheaf-FRL federated learning | ✅ cloned + ready |
| `chemprop_v2` (chemprop) | 1.5K | Full chemprop v2 lib | ✅ cloned + ready |
| **Multiflow**(本地已 clone) | - | contact_sheaf + uncertainty_calibration | ✅ 已 ship |

---

## §1 调研清单 — 每个建议对应现成项目

### §1.1 工程 4 机制 — 现成实现

| 机制 | 现成项目 | 状态 |
|---|---|---|
| **1. BatchNorm→GroupNorm** | ❌ **无现成** (chemprop v2 仍用 BatchNorm1d) | 需要自实现 |
| **2. Kendall MTL uncertainty** | ✅ `mtl_uncertainty_loss` (elna4os) | drop-in |
| **3. Per-domain WeightedRandomSampler** | ✅ PyTorch built-in | drop-in |
| **4. DANN domain adversarial** | ✅ `DANN_py3` (fungtion) | extract GRL + adapt |

### §1.2 理论 3 类工具 — 现成实现

| 工具 | 现成项目 | 状态 |
|---|---|---|
| **Sheaf-FRL** (2026) | ✅ `SPAICOM/sheaf-based-federated-representation-learning` | extract SheafLaplacian |
| **Sheaf-FMTL** (TMLR 2025) | ⚠️ not in SPAICOM (only FRL) | partial — only FRL |
| **SGPC** (AAAI 2026) | ❌ not found | 自实现或 skip |
| **TopoAD / pTU** (2025) | ❌ not found | 自实现或 skip |
| **Kan extensions** (2026) | ❌ not found | 自实现或 skip |

### §1.3 Multiflow 已有(本地已 clone)

| 模块 | 路径 | 用途 |
|---|---|---|
| `pocket_core/contact_sheaf.py` | `/home/hugo/codes/multiflow/pocket_core/` | pocket-ligand sheaf atlas |
| `scripts/build_uncertainty_calibration_applicability_panel.py` | `/home/hugo/codes/multiflow/scripts/` | 4-domain uncertainty calibration |

---

## §2 详细 clone 资源 audit

### §2.1 mtl_uncertainty_loss(elna4os,⭐1)

**路径**:`/home/hugo/codes/mol-metal/.cloned/mtl_uncertainty/`

**核心代码**(28 行!):
```python
class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """Kendall et al. 2018 (CVPR) https://arxiv.org/abs/1705.07115"""
    def __init__(self, num_tasks: int):
        super().__init__()
        self.num_tasks = num_tasks
        self.log_sigmas = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        task_losses = torch.stack(losses)
        sigmas = self.log_sigmas.exp()
        return (task_losses / (2 * sigmas ** 2) + self.log_sigmas).sum()
```

**Verification**:✅ 直接 implement Kendall 2018 同方差不确定性,no delta
**License**: MIT

**适配 Mol-Metal**:
1. Copy `mtl_uncertainty_loss/loss.py` → `molmetal/molmetal_lam/sbdd_env/uncertainty_loss.py`
2. 在 `molmetal/molmetal_lam/sbdd_env/multi_task.py` 替换 hardcoded 加权 → `UncertaintyWeightedMultiTaskLoss(num_tasks=45)`
3. 45 task heads (9 cell × 5 metal) 自动学习权重

**Verify gate**:
- ✅ log_sigmas std > 0.1 after 50 epoch(不退化)
- ✅ Multi-task metric aggregate 与 baseline ±0.01

### §2.2 DANN_py3(fungtion,⭐152)

**路径**:`/home/hugo/codes/mol-metal/.cloned/DANN_py3/`

**核心代码**(`ReverseLayerF`):
```python
from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
```

**架构**:
- `feature` 提取器(backbone)→ `class_classifier`(task heads)
- `ReverseLayerF.apply(feature, alpha)` → `domain_classifier`(4-domain classifier)
- Loss: `L_total = L_task - λ * L_domain`(GRL 让 domain loss 反向优化)

**Note**: DANN_py3 是 CNN for MNIST/MNIST-M,我们要 **adapt 到 D-MPNN backbone**

**适配 Mol-Metal**:
1. Copy `ReverseLayerF` → `molmetal/molmetal_lam/sbdd_env/gradient_reversal.py`
2. 在 D-MPNN backbone 后加 4-class domain head
3. Loss: `L_total = L_task - λ * L_domain`(λ 从 0.1 grid search)
4. 训练脚本 `molmetal/scripts/train_dmpnn_multitask_dann.py`

**Verify gate**:
- ✅ Domain classifier AUC < 0.6 on val(对抗成功)
- ✅ Main task AUC 不下降(对抗没破坏 task signal)
- ✅ OOD 跨域 AUC ≥ baseline

### §2.3 Sheaf-FRL(SPAICOM,⭐0)

**路径**:`/home/hugo/codes/mol-metal/.cloned/Sheaf-FRL/`

**结构**:
- `src/agents/`、`src/communication/`、`src/datamodules/`、`src/mutualinfo/`、`src/orchestrators/`、`src/utils/`
- `scripts/` 40+ experiment scripts
- 配置:`config/`、`pyproject.toml`、`uv.lock`

**核心**:Sheaf-Laplacian "gluing regularizer" on shared pilot samples between domains, Procrustes 闭式更新限制映射

**Note**: 这是 federated learning 框架,我们要 **lift 核心 sheaf theory 实现**(sheaf laplacian + Procrustes update),包装成 Mol-Metal 用的工具

**适配 Mol-Metal**:
1. Extract `src/agents/` 中 sheaf layer + Procrustes update
2. Wrap as `molmetal/molmetal_lam/sbdd_env/sheaf_alignment.py`
3. 用 4 数据域(MetalCytoToxDB / DrugOOD / NCI-60 / BindingDB)的 shared molecular fingerprints 作为 pilot samples
4. Add as auxiliary loss: `L_sheaf = λ_sheaf * tr(X^T L_sheaf X)` where L_sheaf is sheaf Laplacian

**Verify gate**:
- ✅ Sheaf Laplacian eigenvalues non-negative
- ✅ Procrustes update convergence < 100 iter
- ✅ Cross-domain OOD AUC improvement vs DANN baseline

### §2.4 chemprop_v2(chemprop,⭐1.5K)

**路径**:`/home/hugo/codes/mol-metal/.cloned/chemprop_v2/`

**完整模块**:
- `chemprop/models/model.py` — MPNN backbone
- `chemprop/models/multi.py` — **MulticomponentMPNN(多任务,MulticomponentMessagePassing)**
- `chemprop/nn/message_passing/` — message passing layers
- `chemprop/nn/agg.py` — aggregation
- `chemprop/nn/ffn.py` — feed-forward
- `chemprop/nn/predictors.py` — task predictors
- `chemprop/uncertainty/` — **calibrator, estimator, evaluator**

**Note**: chemprop v2 仍用 `BatchNorm1d`(没有 GroupNorm),需要**自定义修改**。

**License**: MIT
**Recent paper**: JCIM 2025 ("Chemprop v2: An Efficient, Modular Machine Learning Package for Chemical Property Prediction")

**适配 Mol-Metal**:
1. **Option A**(小改动):保留我们 `molmetal/baselines/dmpnn_multitask.py`,只 import chemprop 的 `models/multi.py` 作为参考 + 加 GroupNorm + DANN
2. **Option B**(大重构):直接用 chemprop v2 替换我们的 D-MPNN → 9 cell × 5 metal = 45 task heads 全部在 chemprop v2 上重训

**Verify gate**:
- ✅ chemprop v2 successfully loads our MetalCytoToxDB data
- ✅ Multi-task AUC ≥ current D-MPNN baseline
- ✅ MPNN backbone compatible with our featurization

### §2.5 Multiflow 已 clone(本地)

| 模块 | 路径 | 复用价值 |
|---|---|---|
| `contact_sheaf.py` | `/home/hugo/codes/multiflow/pocket_core/` | **Sheaf atlas 数据结构可复用**(虽然没有 federated 实现) |
| `build_uncertainty_calibration_applicability_panel.py` | `/home/hugo/codes/multiflow/scripts/` | **uncertainty calibration framework** for binding/admet/pose/postprocess domains |

**Lift 到 Mol-Metal**:
- `contact_sheaf.ContactSection` dataclass → Mol-Metal `molmetal_lam/sbdd_env/sheaf_section.py`
- `build_uncertainty_calibration_applicability_panel.py` 4-domain logic → Mol-Metal 5-metal + 9-cell uncertainty reporting

---

## §3 Drop-in vs 自实现决策

### §3.1 全部 drop-in(1 周)

| 机制 | 决策 | 时间 |
|---|---|---|
| Kendall MTL | drop-in `mtl_uncertainty_loss` | 0.5 天 |
| WeightedRandomSampler | drop-in PyTorch built-in | 0.5 天 |
| DANN GRL | drop-in `ReverseLayerF` from DANN_py3 | 0.5 天 |
| BatchNorm→GroupNorm | **自实现**(chemprop 也没 GroupNorm) | 0.5 天 |

### §3.2 部分 lift(2 周)

| 模块 | 决策 | 时间 |
|---|---|---|
| Multiflow uncertainty_calibration | lift 4-domain logic | 1 天 |
| Multiflow contact_sheaf | lift ContactSection dataclass | 0.5 天 |
| chemprop v2 reference(不改)| 读自记录 reference impl | 0.5 天 |

### §3.3 Major integration(可选,4-6 周)

| 模块 | 决策 | 时间 |
|---|---|---|
| Sheaf-FRL | lift sheaf laplacian + Procrustes | 5-7 天 |
| chemprop v2 full replace | 重构所有 D-MPNN 训练 | 3-4 周 |

---

## §4 Phase 1 实施 — drop-in 集成(1 周)

### Day 1-2: BatchNorm→GroupNorm(自实现)

**Step 1.1**: grep 我们的 D-MPNN 中所有 BatchNorm
```bash
grep -rn "BatchNorm" /home/hugo/codes/mol-metal/molmetal/baselines/dmpnn*.py
grep -rn "BatchNorm" /home/hugo/codes/mol-metal/molmetal/models/dmpnn.py
grep -rn "BatchNorm" /home/hugo/codes/mol-metal/molmetal/models/v4_multitask_dmpnn.py
```

**Step 1.2**: 写 `molmetal/molmetal_lam/sbdd_env/groupnorm_replacement.py`
```python
def replace_batchnorm_with_groupnorm(module, num_groups=32):
    """Recursively replace all BatchNorm1d/2d with GroupNorm."""
    for name, child in module.named_children():
        if isinstance(child, (nn.BatchNorm1d, nn.BatchNorm2d)):
            num_channels = child.num_features
            gn = nn.GroupNorm(
                num_groups=min(num_groups, num_channels),
                num_channels=num_channels,
            )
            setattr(module, name, gn)
        else:
            replace_batchnorm_with_groupnorm(child, num_groups)
    return module
```

**Step 1.3**: 在 `train_dmpnn_multitask.py` 加 hook 调用
```python
model = replace_batchnorm_with_groupnorm(model, num_groups=32)
```

**Verify gate**:
- ✅ grep `BatchNorm` 在 D-MPNN = 0 hits
- ✅ Forward pass shape 不变
- ✅ 100K subsample pipeline converges

### Day 3: Kendall MTL 集成(drop-in)

**Step 3.1**: Copy `mtl_uncertainty_loss/loss.py` 到项目内
```bash
cp /home/hugo/codes/mol-metal/.cloned/mtl_uncertainty/mtl_uncertainty_loss/loss.py \
   /home/hugo/codes/mol-metal/molmetal/molmetal_lam/sbdd_env/uncertainty_loss.py
```

**Step 3.2**: 在 `train_dmpnn_multitask.py` 加 Kendall loss
```python
from molmetal_lam.sbdd_env.uncertainty_loss import UncertaintyWeightedMultiTaskLoss

# 45 task heads (9 cell × 5 metal) auto-balance
loss_fn = UncertaintyWeightedMultiTaskLoss(num_tasks=45)

# Per-task losses
losses = [cls_loss[i] for i in range(num_cells)] + [reg_loss[i] for i in range(num_cells)]
# Apply per-metal × per-cell  (45 task total = 9 cell × 5 metal × 2 [cls+reg])
# Or simplify to 45 = 9 cell × 5 metal classification only
total_loss = loss_fn(losses)
```

**Verify gate**:
- ✅ log_sigmas std > 0.1 after 50 epoch
- ✅ Multi-task metric aggregate ≥ baseline

### Day 4: Per-domain WeightedRandomSampler

**Step 4.1**: 4-domain dataset combine
```python
from torch.utils.data import WeightedRandomSampler, ConcatDataset

datasets = [
    MetalCytoToxDB(),   # 5,207
    DrugOODNative(),    # 1,580,000
    NCI60GI50(),        # 4,850,000
    BindingDBIC50(),    # 1,072,000
]
concat = ConcatDataset(datasets)
# α = 0.5 soft balancing
weights = []
for i, ds in enumerate(datasets):
    n_d = len(ds)
    n_target = max(len(d) for d in datasets)
    w_d = (n_target / n_d) ** 0.5
    weights.extend([w_d / len(datasets)] * n_d)
sampler = WeightedRandomSampler(weights, num_samples=100000, replacement=True)
loader = DataLoader(concat, batch_size=32, sampler=sampler)
```

**Verify gate**:
- ✅ Per-domain batch composition within ±5% of weights
- ✅ Multi-epoch training stable

### Day 5-6: DANN 集成(drop-in GRL)

**Step 5.1**: Copy `ReverseLayerF` from DANN_py3
```bash
# Extract the GRL function
cat /home/hugo/codes/mol-metal/.cloned/DANN_py3/functions.py \
   > /home/hugo/codes/mol-metal/molmetal/molmetal_lam/sbdd_env/gradient_reversal.py
```

**Step 5.2**: 在 D-MPNN 后加 domain classifier
```python
from molmetal_lam.sbdd_env.gradient_reversal import ReverseLayerF

class DMPNNDANN(nn.Module):
    def __init__(self, backbone, num_tasks=45, num_domains=4):
        super().__init__()
        self.backbone = backbone
        self.task_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_tasks)])
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains),
        )

    def forward(self, x, alpha=1.0):
        features = self.backbone(x)
        task_outputs = [head(features) for head in self.task_heads]
        reversed_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        return task_outputs, domain_output

# Loss
task_loss = sum(cls_losses) / len(cls_losses)
domain_loss = F.cross_entropy(domain_output, domain_label)
total_loss = task_loss - lambda_grl * domain_loss
```

**Verify gate**:
- ✅ Domain classifier AUC < 0.6 on val(对抗成功)
- ✅ Main task AUC 不下降
- ✅ OOD 跨域 AUC ≥ baseline

### Day 7: 100K subsample 验证

**Step 7.1**: Run with new pipeline on 100K subsample
```bash
molmetal/scripts/train_dmpnn_multitask_v2.py \
    --data-config 4domain_100k \
    --epochs 30 \
    --seeds 0 1 2 \
    --use-groupnorm --use-kendall-loss --use-weighted-sampler --use-dann \
    --lambda-grl 0.1 \
    --out-dir molmetal/reports/wf146_phase1_engineering
```

**Verify gate**:
- ✅ Pipeline runs end-to-end
- ✅ Multi-task loss converges
- ✅ Domain classifier AUC < 0.6

---

## §5 Phase 2 实施 — Multiflow lift + chemprop ref(1-2 周)

### Week 2 Day 1-2: Multiflow uncertainty_calibration lift

**Step 1**: 读 multiflow `build_uncertainty_calibration_applicability_panel.py`
- 文件 200+ 行,4-domain uncertainty framework
- Lift `DOMAIN_SPECS` dataclass → Mol-Metal 5-metal + 9-cell + 4-source format

**Step 2**: 写 `molmetal/scripts/build_cross_domain_uncertainty_panel.py`
- 5 metal × 9 cell × 4 source = 180 domain combinations
- Per-domain: probability / label / uncertainty / OOD detection

**Verify gate**:
- ✅ Per-domain panel ship with calibration curves
- ✅ Brier score + ECE reported per domain

### Week 2 Day 3-4: Multiflow contact_sheaf lift

**Step 1**: 读 `multiflow/pocket_core/contact_sheaf.py`
- `ContactSection` dataclass 复用
- `overlap_residual` 计算可作为 cross-domain molecule similarity metric

**Step 2**: 写 `molmetal/molmetal_lam/sbdd_env/sheaf_section.py`
- Lift `ContactSection` + `overlap_residual`
- Adapt 为分子-分子 across-domain 距离度量

**Verify gate**:
- ✅ Section dataclass produces reasonable molecule similarity

### Week 2 Day 5: chemprop v2 reference 读

**Step**: 读 `chemprop_v2/chemprop/models/multi.py` 多任务实现
- 参考多任务架构,不强加用 chemprop
- 记录 `MulticomponentMPNN` 设计到 Mol-Metal `multi_task_design_notes.md`

---

## §6 Phase 3 实施 — Sheaf-FRL lift(2-3 周,可选)

### Week 3: Sheaf-FRL 提取 + Procrustes 闭式更新

**Step 1**: 读 `Sheaf-FRL/src/agents/` 找 sheaf layer + Procrustes
- extract `SheafAlignment` class
- extract `procrustes_update` function

**Step 2**: 写 `molmetal/molmetal_lam/sbdd_env/sheaf_alignment.py`
```python
class SheafAlignment(nn.Module):
    """Sheaf-Laplacian based domain alignment (SPAICOM 2026)."""
    def __init__(self, num_domains=4, hidden_dim=256):
        super().__init__()
        # Learnable restriction maps per domain pair
        self.restriction_maps = nn.ParameterDict({
            f"R_{i}_{j}": nn.Parameter(torch.empty(hidden_dim, hidden_dim))
            for i in range(num_domains) for j in range(num_domains) if i != j
        })
        # Initialize with random orthogonal
        for R in self.restriction_maps.values():
            nn.init.orthogonal_(R)

    def sheaf_laplacian(self, X_per_domain):
        """Compute sheaf Laplacian from per-domain representations."""
        # ... sheaf Laplacian construction ...

    def procrustes_update(self, X_i, X_j):
        """Closed-form Procrustes alignment between domain i and j."""
        # X_j ≈ R_i_j @ X_i
        U, _, Vt = torch.linalg.svd(X_j @ X_i.T)
        R = U @ Vt  # orthogonal
        return R

    def forward(self, X_per_domain):
        """Compute sheaf-Laplacian regularization loss."""
        L = self.sheaf_laplacian(X_per_domain)
        # Tr(X^T L X) for each domain pair
        loss = 0
        for X in X_per_domain.values():
            loss += torch.trace(X.T @ L @ X)
        return loss
```

**Step 3**: Integrate with D-MPNN
- After D-MPNN forward, get per-domain representations X_d
- L_sheaf = λ_sheaf * sheaf_alignment(X_d)
- L_total = L_task + L_sheaf

**Verify gate**:
- ✅ Sheaf Laplacian eigenvalues non-negative
- ✅ Procrustes update convergence < 100 iter
- ✅ Cross-domain OOD AUC ≥ DANN baseline

### Week 4: 完整 5-metal × NCI-60 + DrugOOD + BindingDB 训练

(假设 -7天或更久)

---

## §7 文件路径 + 资源地图

### §7.1 外部 cloned repos

```
/home/hugo/codes/mol-metal/.cloned/
├── DANN/                   # fungtion Py2.7 旧版
├── DANN_py3/               # fungtion Py3 fork,ReverseLayerF in functions.py
├── mtl_uncertainty/        # elna4os Kendall 2018 loss impl
├── Sheaf-FRL/              # SPAICOM sheaf theory for federated learning
└── chemprop_v2/           # chemprop standard library v2.x
```

### §7.2 Multiflow 已有(本地已 clone)

```
/home/hugo/codes/multiflow/
├── pocket_core/contact_sheaf.py          # sheaf atlas, ContactSection dataclass
├── scripts/build_uncertainty_calibration_applicability_panel.py  # 4-domain uncertainty
└── scripts/build_*.py  # 30+ build/audit scripts (lift as needed)
```

### §7.3 计划 drop-in 的 Mol-Metal 文件

```
molmetal/molmetal_lam/sbdd_env/
├── uncertainty_loss.py        # NEW from mtl_uncertainty_loss
├── gradient_reversal.py      # NEW from DANN_py3
├── groupnorm_replacement.py   # NEW (self-implement)
└── sheaf_alignment.py         # NEW Phase 3 from Sheaf-FRL

molmetal/scripts/
├── build_cross_domain_uncertainty_panel.py  # NEW from multiflow
├── train_dmpnn_multitask_v2.py              # NEW with all 4 mechanisms
└── verify_dann_domain.py                    # NEW for DANN sanity
```

---

## §8 Self-check rigor

### Phase 1 (drop-in) gates

| Check | Standard | Verified |
|---|---|:---:|
| **BatchNorm→GroupNorm** | `grep BatchNorm` in D-MPNN = 0 | ? |
| **Kendall log_sigmas** | std > 0.1 after 50 epoch | ? |
| **Per-domain batch balance** | within ±5% of weights | ? |
| **DANN domain AUC** | < 0.6 on val(对抗成功)| ? |
| **100K pipeline** | converges within 30 epoch | ? |
| **Multi-task metric** | ≥ current D-MPNN baseline | ? |
| **Tests PASS** | ≥16 new tests | ? |

### Phase 2 (lift) gates

| Check | Standard | Verified |
|---|---|:---:|
| **Multiflow uncertainty lift** | per-domain calibration curve ship | ? |
| **Multiflow contact_sheaf lift** | section dataclass produces reasonable similarity | ? |
| **chemprop v2 reference** | multi-task notes shipped | ? |

### Phase 3 (Sheaf-FRL) gates

| Check | Standard | Verified |
|---|---|:---:|
| **Sheaf Laplacian eigenvalues** | ≥ 0 | ? |
| **Procrustes convergence** | < 100 iter | ? |
| **OOD cross-domain AUC** | ≥ DANN baseline | ? |
| **5-metal × NCI-60 + DrugOOD + BindingDB** | 7.5M training complete | ? |

---

## §9 Resource + Risk

**资源**:
- ✅ GPU-0 RTX PRO 6000 Blackwell 96GB(0% util)
- ⚠️ GPU-1 RTX 5090(被 Wf-131 P3 占用,~3 小时运行)
- ✅ 32 CPU + 94 GB RAM + 1.8 TB disk
- ✅ 5 cloned repos total ~200 MB code
- ✅ Multiflow 本地已 clone

**风险**:
- **Sheaf-FRL Procrustes 实现复杂**(5-7 天,可能 fail)
- **chemprop v2 完全替换工作量大**(3-4 周,可能 delay)
- **DANN λ grid search**(5-10 runs × 3 小时 = 15-30 小时 GPU)
- **100K subsample 可能不够**(收敛 OK 但泛化不足)

**Likelihood**:
- Phase 1 完成:**90%**(4 个 drop-in 都成熟)
- Phase 2 完成:**75%**(multiflow lift 已经写好,chemprop ref 读自即可)
- Phase 3 完成:**55%**(Sheaf-FRL 是 2026 新,实现风险高)

---

## §10 Honest framing per party-twenty-spirit §3.5

**承认**:
1. **之前我 propose 自建时没有先调研现成** — 用户 push 我做 clone 是对的
2. **mtl_uncertainty_loss / DANN_py3 / Sheaf-FRL / chemprop v2 都是成熟现成项目** — 我应该一开始就 clone
3. **Multiflow 本地已有 contact_sheaf + uncertainty_calibration** — 我之前不知道
4. **Phase 3 Sheaf-FRL 仍是 experimental**(2026 新方法)
5. **Phase 1-2 是必须做,Phase 3 是 nice-to-have**

**不假装**:
1. 不假装 GroupNorm 现成 — 必须自实现
2. 不假装 Sheaf-FMTL / SGPC / TopoAD / Kan 现成 — 没找到
3. 不假装 drop-in 0 工作量 — 还是需要 ~1 周
4. 不假装 chemprop v2 替换简单 — 实际 3-4 周

**paper §6 必须 ship**:
- "Mol-Metal multi-domain training extended with 4 engineering mechanisms (GroupNorm + Kendall + WeightedRandomSampler + DANN) integrated via drop-in from open-source reference impls (mtl_uncertainty_loss + DANN_py3 + chemprop v2 reference); full 7.5M training deferred to R20."

---

## §11 Cross-reference

- **TODO-140**:Tier-1 baseline alignment
- **TODO-141**:Q2 JCIM completion sprint(本 TODO 扩展其 Phase 2)
- **TODO-142**:Cross-domain recommendation(本 TODO 是其 clone+adapt 实施)
- **Wf-131 P3**:LOMO 5/5 metal × 5 seed × 50 epoch (7/125 done)
- **Wf-145**:m19 Rh fail (TODO-135 §5)
- **Wf-126 / Wf-140 / Wf-142 / Wf-144**:CrossDocked 5 pocket Vina work
- **TODO-84**:paper rule active — Phase 4+ docs 待 user 授权

---

## §12 Final verdict

**Clone + Adapt 资源**:
- ✅ 5 个外部 repo cloned
- ✅ Multiflow 本地已有
- ✅ Path bug 已修

**Phase 1 drop-in 集成**(1 周):
- **mtl_uncertainty_loss** → 替换 hardcoded MTL 加权
- **DANN_py3 ReverseLayerF** → domain adversarial head
- **PyTorch built-in** WeightedRandomSampler → per-domain balance
- **自实现** GroupNorm replacement (因为 chemprop v2 也没有)

**Phase 2 lift**(1-2 周):
- Multiflow `uncertainty_calibration_applicability_panel.py` lift
- Multiflow `contact_sheaf.py` lift
- chemprop v2 reference 读

**Phase 3 Sheaf-FRL**(2-3 周,可选,高风险):
- Extract Sheaf-Laplacian + Procrustes
- Add auxiliary loss on D-MPNN

**ROI 决策**:
- **Phase 1 + Phase 2**:2-3 周 → paper 接受率 **85-90%**
- **Phase 1 + Phase 2 + Phase 3**:5-6 周 → paper 接受率 **85-90%**(高风险 + ROI 不明显提升)

**下一步**:用户决定:
- **A**:立即启动 Phase 1 drop-in 集成(1 周)— **推荐**
- **B**:Phase 1 + Phase 2(2-3 周)
- **C**:Phase 1 + 2 + 3(5-6 周,高风险)

我推荐 **A** — drop-in 是 highest ROI 路径,且符合你"先调研现成"的指令。

---

## §13 Related skill alignment

- `party-twenty-spirit` §3.5:0-tolerance honest-negative framing → §10
- `mol-metal-mechanism-mapping`:3 层 stack + 4 数据域 → §1
- `mol-metal-anticancer`:anticancer domain context → §3
- `mol-metal-lambda-calculus`:Lambda + CFM + MCTS 与新工具集成 → §4-6
- `mol-metal-evaluation-rigor`:LODO + per-domain breakdown → §8
