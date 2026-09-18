# Flow Matching — 开山鼻祖 clone-adapter 计划

## 选定的"开山鼻祖" paper

**Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023).**
*Flow Matching for Generative Modeling.* **ICLR 2023**.
- arXiv: 2210.02747 (提交 2022-10-06)
- 作者:Meta AI (FAIR) + Weizmann Institute
- 引用: ~4600+ (FM 领域最高)
- **官方代码**:`github.com/facebookresearch/flow_matching`
- 配套教程:Lipman et al. 2024, *Flow Matching Guide and Code* (arXiv 2412.06264)

**为什么这是开山鼻祖**:
- 提出 **Flow Matching** 范式(不是 Stochastic Interpolants / Rectified Flow 的并行工作)
- 提出 **Conditional Flow Matching (CFM)** loss 的边际化技巧
- 证明 FM 与 diffusion 的关系 — diffusion 是 FM 的特例
- 提出 **Optimal Transport (OT) conditional path** — 直线路径,比 diffusion 短 70% 步数
- 在 ImageNet 64x64 上 FID 14.45 (vs DDPM 17.36)

**为什么选这个 clone(而不是 FLOWR、TargetDiff、DiffDock)**:
1. **基础性**:这是 flow matching 的"母论文",所有后续工作(FLOWR/TargetDiff 内部都基于这个)都引用它
2. **简洁性**:官方代码 ~3000 行,核心 CFM loss + OT path scheduler 几百行,易于适配
3. **抽象稳定**:FM 的数学抽象(条件向量场、OT 路径)比 SOTA 应用层更稳定
4. **易扩展**:我们的算法优化在它之上叠加,不需要修改其核心

## 复用官方代码 — 策略

```
facebookresearch/flow_matching     (cloned as-is)
        │
        │  我们用它的:
        │   • flow_matching.path.scheduler.ConditionalOTScheduler
        │   • flow_matching.path.AffineLinearPath
        │   • flow_matching.solver.ode_solver.ODESolver (Euler, RK4)
        │   • flow_matching.loss.ConditionalFlowMatchingLoss
        │   • flow_matching.path.utils.TrajectoryReference
        │
        │  我们不直接用它的:
        │   • 它的 model 实现(我们的 EGNN 替换)
        │   • 它的 image/text example
        │   • 它的 Riemannian FM (暂时用不上)
        ▼
molmetal/adapters/flow_matching_lipman/
   ├── lipman_adapter.py         # 包装 facebookresearch/flow_matching
   ├── egnn_velocity.py          # 我们的 EGNN as velocity field
   ├── path_schedulers.py        # OT / Linear / Conditional wrappers
   ├── solvers.py                # ODE solver wrappers
   └── io.py                     # Pocket/Molecule <-> FM input
        │
        ▼
molmetal/ports/generator.py  (MoleculeGenerator Protocol)
        │
        ▼
FlowMatchingAdapter : MoleculeGenerator
```

## 我们要 clone 什么

### 主仓库
```bash
git clone https://github.com/facebookresearch/flow_matching.git \
    molmetal/references/flow_matching

# 关联项目 (已 clone 进 references)
git clone https://github.com/jule-c/flowr.git \
    molmetal/references/FLOWR
# (注意:原文档写错,insitro/FLOWR 不存在。FLOWR 是 Cremer et al. 2025/2026)
git clone https://github.com/jule-c/flowr_root.git \
    molmetal/references/flowr_root
```

### 我们的包装层(Phase 0 第一周)

```python
# molmetal/adapters/flow_matching_lipman/lipman_adapter.py
"""
Adapter for the open-source Facebook Research Flow Matching library.

Lipman et al. 2023 — ICLR 2023 — the "founding paper" of flow matching.

We treat the cloned library as a black box for the path/loss/solver
machinery, and plug in our own EGNN-based velocity field.
"""
from __future__ import annotations
from typing import List, Optional

import torch

# Imported from the cloned reference
from flow_matching.path import (
    AffineLinearPath,
    MixtureDiscreteEpsilonPath,  # not used yet, but available
)
from flow_matching.path.scheduler import (
    CondOTScheduler,         # Optimal Transport — Lipman et al. §4.7
    PolynomialConvexScheduler,
    VPScheduler,             # for diffusion path — Lipman et al. §4.5
)
from flow_matching.loss import (
    ConditionalFlowMatchingLoss,  # the core CFM loss — Lipman et al. §4.5
)
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.solver import Solver, ModelWrapper
from flow_matching.integrator import odeint, EulerSimulator, RK4Simulator
from flow_matching.integrator.ode import torchdiffeq_odeint

from molmetal.domain import Pocket, Molecule
from molmetal.ports import MoleculeGenerator, GenerationConfig


class EGNNVelocityField(torch.nn.Module):
    """
    The velocity field v_θ(x, t) parameterised by our EGNN.

    Input:  pocket + ligand coords at time t
    Output: per-atom 3D velocity

    We follow the conditional flow matching convention:
      x_t = (1 - σ_t) * x_0 + σ_t * x_1
      target_v = x_1 - x_0   (for OT path: straight line)
    """
    def __init__(self, egnn: nn.Module, hidden_dim: int = 128):
        super().__init__()
        self.egnn = egnn
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # Time + node feature → 3D velocity per node
        self.vel_head = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        x_t: torch.Tensor,         # (N_atoms, 3)
        atom_types: torch.Tensor,  # (N_atoms,)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: Optional[torch.Tensor],
        t: torch.Tensor,           # scalar in [0, 1]
    ) -> torch.Tensor:
        # 1. Encode t as a per-node feature
        t_emb = self.time_mlp(t.view(1, 1)).expand(x_t.shape[0], -1)
        # 2. Run EGNN
        h = self.egnn(atom_types, x_t, edge_index, t_emb, edge_attr=edge_attr)
        # 3. Project to 3D velocity
        return self.vel_head(h)


class LipmanFlowMatchingAdapter(MoleculeGenerator):
    """
    MoleculeGenerator port implementation wrapping the Facebook
    Research Flow Matching library (Lipman et al. 2023).

    Public API matches our MoleculeGenerator Protocol; the internal
    training/sampling uses the cloned library.
    """
    @property
    def name(self) -> str:
        return "LipmanFlowMatching_v1"

    def __init__(
        self,
        pocket_encoder: nn.Module,    # pocket → graph embedding
        atom_decoder: nn.Module,      # graph embedding → atom types
        ref_repo_path: str = "molmetal/references/flow_matching",
        device: str = "cuda",
    ):
        self.pocket_encoder = pocket_encoder
        self.atom_decoder = atom_decoder
        # Import the cloned library from the local path
        sys.path.insert(0, ref_repo_path)
        from flow_matching.path import AffineLinearPath
        from flow_matching.path.scheduler import CondOTScheduler
        from flow_matching.loss import ConditionalFlowMatchingLoss
        from flow_matching.solver import ODESolver
        from flow_matching.integrator import EulerSimulator
        # ... see io.py for full imports
        # (we keep these references as self._fm_path etc.)

    def setup(self, device: str = "cuda") -> None:
        """Initialise the CFM components from Lipman et al."""
        self._path = AffineLinearPath()  # x_t = (1-t) * x_0 + t * x_1
        self._scheduler = CondOTScheduler()  # OT path
        self._loss_fn = ConditionalFlowMatchingLoss(
            path=self._path, scheduler=self._scheduler
        )
        # Our velocity field
        self._velocity_field = EGNNVelocityField(...).to(device)

    def train_step(
        self,
        pocket: Pocket,
        mols: List[Molecule],     # ground truth
    ) -> float:
        """One training step using the CFM loss from Lipman et al."""
        # 1. Encode pocket
        pocket_emb = self.pocket_encoder(pocket)
        # 2. Sample (x_0, x_1, t)
        x_0 = torch.randn_like(mols[0].coords)  # noise
        x_1 = torch.stack([m.coords for m in mols])  # data
        t = torch.rand(x_1.shape[0], device=x_1.device)
        # 3. Compute x_t via the path
        x_t = self._path.sample(t, x_0, x_1)
        # 4. Predict velocity with our EGNN
        v_pred = self._velocity_field(x_t, atom_types, edge_index, t)
        # 5. CFM loss (from Lipman et al. §4.5)
        loss = self._loss_fn(v_pred, x_1, x_0, t)
        return loss

    def generate(
        self,
        pocket: Pocket,
        config: GenerationConfig,
    ) -> List[Molecule]:
        """Sample N molecules by integrating the ODE dx/dt = v_θ(x, t)."""
        # 1. Sample noise for each atom
        x_0 = torch.randn(...)
        # 2. Integrate using Euler / RK4 (from flow_matching.integrator)
        trajectory = odeint(
            self._velocity_field,
            x_0,
            t=torch.linspace(0, 1, config.n_steps + 1),
            method="euler",
        )
        # 3. Decode atom types
        atom_logits = self.atom_decoder(trajectory[-1], ...)
        atom_types = atom_logits.argmax(dim=-1)
        # 4. Build Molecule objects
        return [Molecule(coords=traj, atom_types=t, ...) for ...]
```

## 算法优化空间(后续)

我们的 port + 官方实现先跑通 baseline。然后在这之上优化:

| 优化点 | 出处 | 引入时机 |
|---|---|---|
| **OT path + AffineLinearPath** | Lipman 2023 §4.7 | Phase 1 (默认就有) |
| **Optimal Transport coupling (minibatch)** | Tong et al. 2023, *"Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport"* (FM 引用 ~500) | Phase 2 (高频) |
| **Mini-batch OT** | 同一篇 | Phase 2 |
| **Probability path interpolation design** | 我们自己的实验 | Phase 2-3 |
| **Stochastic interpolant** (Albergo 2023, 同期工作) | arXiv 2303.08797 | 可选(Phase 3) |
| **Riemannian FM** (Chen 2024, Flow Matching on General Geometries) | arXiv 2302.03660 | **重要** — 用于 SBDD 3D |
| **Riemannian Rectified Flow Matching** | 后续工作 | Phase 3 |
| **Discrete Flow Matching** (Gat 2024) | arXiv 2407.15595 | Phase 3-4 — 离散原子类型 |
| **Manifold constraint projection** | 我们的 | Phase 2 — 把分子约束在 manifold 上(化学有效) |
| **Guidance for binding** | 我们的(用 DiffDock / Vina 作 classifier) | Phase 2 |
| **Flow Map Matching** (Boffi 2024) | arXiv 2404.18022 | 实验性 |
| **Non-isotropic noise** | 我们 | Phase 2-3 |

**对贵金属 anticancer 的具体优化**(Phase 3-4):

1. **Geometric prior**:用 square-planar Pt(II) 几何先验约束 flow path
2. **Dative bond edge type**:在 EGNN 消息传递里区分 covalent vs dative
3. **Multi-objective guidance**:同时引导 binding (Vina) + IC50 (ML predictor) + QED + SA
4. **Latent space shaping**:FM 的 latent space 加入 metal 维度

## 与 MolFlow-Triton 的关系

**MolFlow-Triton 现状**:
- flow_matching 基础设施已有(`flow_matching/loss.py`, `flow_matching/sampler.py`)
- 我们的实现是 toy 的(单 batch QM9 训练)

**新计划**:
- facebookresearch/flow_matching 替换我们 toy 的实现
- 我们的 EGNN (`models/velocity_net.py::EGNNLayer`)作为 velocity field network
- 我们的 `_scatter.py` 给 EGNN 提供 Triton 加速
- 我们的 Vina-style guidance (Vina + QED + SA) 作 conditioning

## Clone + 适配 + 写接口 的实施步骤

### Phase 0(1 周)— 搭骨架

1. **clone 仓库**:
   ```bash
   git clone https://github.com/facebookresearch/flow_matching.git \
       molmetal/references/flow_matching
   ```
2. **写 domain dataclasses** (Pocket, Molecule, Complex) — 已有部分(分子),加 Pocket
3. **定义 MoleculeGenerator port** (Protocol)
4. **写 mock adapter + e2e test** (用 mock 跑通流程)
5. **写 LipmanFlowMatchingAdapter skeleton** — import cloned lib 但先用 toy 数据

### Phase 1(2 周)— 接通官方库

1. **Path / scheduler 配置**:
   - AffineLinearPath + CondOTScheduler (Lipman 2023 §4.7)
   - 测试在不同 data 上 path 的形状
2. **Loss 接通**:ConditionalFlowMatchingLoss
3. **EGNN velocity field**:接入我们已有的 EGNNLayer
4. **Sample with ODE solver**:EulerSimulator / RK4Simulator
5. **e2e on toy 3D** (8-atom 单分子,验证)

### Phase 2(1-2 周)— 3D molecular generation

1. **Pocket encoder** — 蛋白→graph embedding(用我们 EGNN)
2. **Atom decoder** — graph → atom types
3. **训练** on toy 分子集 — 验证 loss 下降、generated mols 有效
4. **接 docking loop** — generate → dock → score
5. **Mini benchmark** — 100 个 pocket,看 Vina score 分布

### Phase 3(2-3 周)— 优化 + 真实数据

1. **加 Rectified Flow / mini-batch OT**
2. **Property guidance** (classifier-free guidance with IC50 predictor)
3. **Metal complex prior** — 把金属配位作为 manifold constraint
4. **接 MMP 真实数据**

## 关键文件路径

```
molmetal/
├── references/
│   └── flow_matching/                # git cloned, AS-IS
├── adapters/
│   └── flow_matching_lipman/
│       ├── __init__.py
│       ├── lipman_adapter.py         # MoleculeGenerator implementation
│       ├── egnn_velocity.py          # our EGNN as v_θ
│       ├── path_io.py                # Molecule coords <-> FM tensor
│       ├── schedulers.py             # OT / Linear / Diffusion wrappers
│       └── tests/
├── ports/
│   └── generator.py                  # MoleculeGenerator Protocol
├── domain/
│   ├── pocket.py
│   ├── molecule.py
│   └── complex.py
├── orchestration/
│   └── closed_loop.py                # 编排 generate → dock → score
└── scripts/
    └── train_flow_matching.py        # 训练入口
```

## 测试策略

```python
# tests/test_lipman_adapter.py

def test_path_scheduler():
    """Verify the cloned library's path + scheduler produces expected shapes."""
    from flow_matching.path import AffineLinearPath
    from flow_matching.path.scheduler import CondOTScheduler
    path = AffineLinearPath()
    scheduler = CondOTScheduler()
    x_0 = torch.randn(4, 8, 3)  # 4 molecules, 8 atoms, 3D
    x_1 = torch.randn(4, 8, 3)
    t = torch.rand(4)
    x_t = path.sample(t, x_0, x_1)
    assert x_t.shape == (4, 8, 3)


def test_cfm_loss():
    """Verify CFM loss is the standard formulation."""
    from flow_matching.loss import ConditionalFlowMatchingLoss
    loss_fn = ConditionalFlowMatchingLoss()
    v_pred = torch.randn(4, 8, 3)
    x_1 = torch.randn(4, 8, 3)
    x_0 = torch.randn(4, 8, 3)
    t = torch.rand(4)
    loss = loss_fn(v_pred, x_1, x_0, t)
    assert loss.requires_grad
    assert loss.item() > 0


def test_egnn_velocity_field():
    """Our EGNN + Lipman's loss works end-to-end."""
    from molmetal.adapters.flow_matching_lipman import (
        LipmanFlowMatchingAdapter, EGNNVelocityField
    )
    from flow_matching.loss import ConditionalFlowMatchingLoss
    v_field = EGNNVelocityField(...)
    loss_fn = ConditionalFlowMatchingLoss()
    # ... train step ...


def test_generate_3d_molecule():
    """Adapter produces a chemically valid Molecule."""
    adapter = LipmanFlowMatchingAdapter(...)
    adapter.setup()
    pocket = ...
    mols = adapter.generate(pocket, GenerationConfig(n_samples=10))
    assert len(mols) == 10
    for m in mols:
        assert m.coords.shape[1] == 3
        # RDKit validity check
        assert RDKit_is_valid(m.to_rdkit())
```

## 与 DiffDock 的关系

**DiffDock 是 docking port** — 它不生成,只对接。
我们的 flow matching adapter 只生成,不解 dock。
闭环:generate(dock 目标) → dock(DiffDock) → score(weighted) → refine。

DiffDock 还是要 clone,但优先级排在 flow matching 之后。

## 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| facebookresearch/flow_matching 依赖 torchdiffeq | 低 | torchdiffeq 是 PyTorch native,ROCm 兼容 |
| Clone 仓库很大(2-3k 行) | 已存在 | 我们只 import 需要的子模块 |
| Lipman 2023 是 image-focused,3D molecular 例子少 | 中 | 参考 SynthAether / atong01 / g4vrel 3D 实现 |
| facebookresearch/flow_matching 文档少 | 中 | 看 README + examples 目录 |
| 我们 EGNN 与 FM 库不兼容 | 低 | EGNN 是纯 torch.nn.Module,无外部依赖 |

## 立即可做(不等回答)

```bash
# 1. Clone
mkdir -p molmetal/references
git clone https://github.com/facebookresearch/flow_matching.git \
    molmetal/references/flow_matching

# 2. 探查结构
ls molmetal/references/flow_matching/flow_matching/
ls molmetal/references/flow_matching/examples/

# 3. 写 domain dataclass (Pocket, Molecule, Complex)

# 4. 写 port (MoleculeGenerator Protocol)

# 5. 写 LipmanFlowMatchingAdapter skeleton (import cloned lib)

# 6. E2E test on toy 3D
```

## 待你回答

1. **要不要把 FLOWR/DiffDock 也 clone**?
   - (a) 先 FM,DiffDock 之后
   - (b) 现在都 clone
2. **第一靶点集**(沿用之前的 MMP2/9 还是改)?
3. **接闭环 vs 只生成**?
   - (a) 这次只做 Generator port,docking 后面
   - (b) 一起做完整闭环

回答后我开 Phase 0 clone。