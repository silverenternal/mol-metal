# Flow Matching 集成到 molmetal 的具体步骤

## 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│   molmetal/                                                    │
│                                                              │
│   ┌──────────┐    ┌────────────────────────────┐          │
│   │ domain/  │    │  ports/  (我们定义接口)        │          │
│   │ Pocket   │    │  MoleculeGenerator (Protocol) │          │
│   │ Molecule │◄───│  DockingEngine   (Protocol)  │          │
│   │ Complex  │    │  PropertyPredictor(Protocol)  │          │
│   └──────────┘    │  ScoringFunction (Protocol)  │          │
│        ▲           │  DesignLoop      (Protocol)  │          │
│        │           └────────────┬───────────────┘          │
│        │                        │                          │
│   ┌────┴────────────────────────────────────────┐         │
│   │  adapters/  (具体实现,from cloned repos)       │         │
│   │  ┌──────────────────────────────────────┐   │         │
│   │  │ LipmanFlowMatchingAdapter  ← 我们写   │   │         │
│   │  │  import flow_matching (cloned)        │   │         │
│   │  │  import models.egnn (我们的 EGNN)     │   │         │
│   │  └──────────────────────────────────────┘   │         │
│   │  ┌──────────────────────────────────────┐   │         │
│   │  │ DiffDockAdapter (后续 Phase)          │   │         │
│   │  └──────────────────────────────────────┘   │         │
│   │  ┌──────────────────────────────────────┐   │         │
│   │  │ EGNNPropertyPredictor (我们的)        │   │         │
│   │  └──────────────────────────────────────┘   │         │
│   └─────────────────────────────────────────┘         │
│                                                              │
│   ┌──────────────────┐                                  │
│   │  references/     │   ← git clone as-is             │
│   │  flow_matching/  │                                  │
│   │  (Lipman 2023)   │                                  │
│   └──────────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

## Lipman 2023 在 molmetal 里的位置

我们**不重写** flow matching — 我们**调用** facebookresearch/flow_matching 库。

```python
# 我们的 adapter (molmetal/adapters/flow_matching_lipman/lipman_adapter.py)
# 1. import cloned library
import sys
sys.path.insert(0, "molmetal/references/flow_matching")
from flow_matching.path import AffineLinearPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.loss import ConditionalFlowMatchingLoss
from flow_matching.solver import ODESolver
from flow_matching.integrator import EulerSimulator, RK4Simulator

# 2. 我们的 EGNN (from MolFlow-Triton)
from models.velocity_net import EGNNLayer, VelocityNet
# 但只要 EGNNLayer,不要 velocity net 的全包装

# 3. 我们的 adapter 实现 MoleculeGenerator port
class LipmanFlowMatchingAdapter(MoleculeGenerator):
    ...
```

## 关键 Cloned 库的 API(只列我们要用的)

### `flow_matching.path.AffineLinearPath`
**Lipman 2023 §4.8** — 最常用的 linear path
```python
from flow_matching.path import AffineLinearPath
path = AffineLinearPath()  # 默认: x_t = (1-t) * x_0 + t * x_1
x_t = path.sample(t, x_0, x_1)  # t: (B,), x_0/x_1: (B, N, 3)
target_v = x_1 - x_0  # 直线路径的速度场
```

### `flow_matching.path.scheduler.CondOTScheduler`
**Lipman 2023 §4.7** — Optimal Transport path
```python
from flow_matching.path.scheduler import CondOTScheduler
scheduler = CondOTScheduler()
# 与 AffineLinearPath 配合用
# CFM loss 内部用 scheduler 的 OT 配对
```

### `flow_matching.loss.ConditionalFlowMatchingLoss`
**Lipman 2023 §4.5** — 核心 loss
```python
from flow_matching.loss import ConditionalFlowMatchingLoss
loss_fn = ConditionalFlowMatchingLoss(path=path, scheduler=scheduler)
loss = loss_fn(v_pred, x_1, x_0, t)
# CFM loss = ||v_θ(x_t, t) - (x_1 - x_0)||²
# (Lipman 2023 Theorem 1: 这与 FM loss 梯度相同)
```

### `flow_matching.integrator.EulerSimulator` / `RK4Simulator`
ODE 求解器(从噪声积分到数据)
```python
from flow_matching.integrator import EulerSimulator
simulator = EulerSimulator(velocity_model=...)
trajectory = simulator.simulate(x_0, t=torch.linspace(0, 1, 100))
# trajectory: (T+1, B, N, 3)
```

## 我们的 adapter 完整代码骨架(50 行核心)

```python
# molmetal/adapters/flow_matching_lipman/lipman_adapter.py
"""LipmanFlowMatchingAdapter — wraps facebookresearch/flow_matching.

Lipman et al. 2023, ICLR 2023, "Flow Matching for Generative Modeling"
arXiv:2210.02747  |  Official code: github.com/facebookresearch/flow_matching
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

# 1. Import the cloned library (Lipman 2023 official)
_REF_PATH = Path(__file__).parent.parent.parent / "references" / "flow_matching"
if str(_REF_PATH) not in sys.path:
    sys.path.insert(0, str(_REF_PATH))

from flow_matching.path import AffineLinearPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.loss import ConditionalFlowMatchingLoss
from flow_matching.integrator import EulerSimulator

# 2. Import our domain types
from molmetal.domain import Pocket, Molecule
from molmetal.ports import MoleculeGenerator, GenerationConfig

# 3. Import our EGNN for the velocity field
from models.velocity_net import EGNNLayer


class EGNNVelocityField(nn.Module):
    """Our EGNN parameterised as v_θ(x, t) for 3D molecular flow matching."""

    def __init__(self, hidden_dim=128, edge_mlp_hidden=None, n_layers=3):
        super().__init__()
        edge_mlp_hidden = edge_mlp_hidden or hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.atom_embed = nn.Embedding(100, hidden_dim)  # up to element 100
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim=hidden_dim, edge_mlp_hidden=edge_mlp_hidden)
            for _ in range(n_layers)
        ])
        self.vel_head = nn.Linear(hidden_dim, 3, bias=False)
        nn.init.zeros_(self.vel_head.weight)  # zero-init: start at zero velocity

    def forward(
        self,
        x_t: torch.Tensor,         # (B, N, 3)
        atom_types: torch.Tensor,  # (B, N)
        edge_index: torch.Tensor,  # (B, 2, E)
        t: torch.Tensor,           # (B,) or (B, 1)
        edge_mask: torch.Tensor | None = None,  # (B, E)
    ) -> torch.Tensor:
        b, n = x_t.shape[:2]
        # Broadcast t to per-atom
        t_per_atom = self.time_mlp(t.view(-1, 1)).unsqueeze(1).expand(b, n, -1)
        h = self.atom_embed(atom_types) + t_per_atom
        last_v = torch.zeros_like(x_t)
        for layer in self.layers:
            h, last_v = layer(
                h, x_t, edge_index,
                cond_per_node=None,
                edge_mask=edge_mask,
            )
        # Equivariant velocity = zero-init head (vel_head == 0) + aggregated vector messages
        return self.vel_head(h) + last_v


class LipmanFlowMatchingAdapter(MoleculeGenerator):
    """MoleculeGenerator port = Lipman 2023 CFM + our EGNN velocity field."""

    def __init__(
        self,
        ref_repo_path: str = "molmetal/references/flow_matching",
        hidden_dim: int = 128,
        device: str = "cuda",
    ):
        self._ref_repo = Path(ref_repo_path)
        self._hidden_dim = hidden_dim
        self._device = device

        # Will be initialised in setup()
        self.velocity_field: EGNNVelocityField | None = None
        self.path: AffineLinearPath | None = None
        self.scheduler: CondOTScheduler | None = None
        self.loss_fn: ConditionalFlowMatchingLoss | None = None
        self.simulator: EulerSimulator | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    @property
    def name(self) -> str:
        return "LipmanFlowMatching_v1"

    def setup(self, device: str = "cuda") -> None:
        """Initialise all components from the cloned Lipman library."""
        # Inject cloned library into path
        if str(self._ref_repo) not in sys.path:
            sys.path.insert(0, str(self._ref_repo))

        from flow_matching.path import AffineLinearPath
        from flow_matching.path.scheduler import CondOTScheduler
        from flow_matching.loss import ConditionalFlowMatchingLoss
        from flow_matching.integrator import EulerSimulator

        self.path = AffineLinearPath()
        self.scheduler = CondOTScheduler()
        self.loss_fn = ConditionalFlowMatchingLoss(
            path=self.path, scheduler=self.scheduler
        )
        self.simulator = EulerSimulator()

        self.velocity_field = EGNNVelocityField(
            hidden_dim=self._hidden_dim,
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            self.velocity_field.parameters(), lr=1e-4
        )
        self._device = device

    def train_step(
        self,
        pocket: Pocket,
        mols: list[Molecule],
    ) -> float:
        """One CFM training step (Lipman 2023 §4.5)."""
        b = len(mols)
        # 1. Concatenate coords
        max_n = max(m.coords.shape[0] for m in mols)
        x_1 = torch.zeros(b, max_n, 3, device=self._device)
        atom_types = torch.zeros(b, max_n, dtype=torch.long, device=self._device)
        node_mask = torch.zeros(b, max_n, dtype=torch.bool, device=self._device)
        for i, m in enumerate(mols):
            n = m.coords.shape[0]
            x_1[i, :n] = m.coords.to(self._device)
            atom_types[i, :n] = m.atom_types.to(self._device)
            node_mask[i, :n] = True

        # 2. Sample noise + t (Lipman 2023 CFM)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(b, device=self._device)

        # 3. CFM loss (from cloned library)
        # The library's loss internally:
        #   - samples x_t = path.sample(t, x_0, x_1)
        #   - target = path.target_velocity(t, x_0, x_1)  = x_1 - x_0
        #   - loss = MSE(v_pred, target)
        # We need to pass x_t explicitly to our velocity field:
        from flow_matching.path import AffineLinearPath
        x_t = self.path.sample(t, x_1=x_1, x_0=x_0)
        v_pred = self.velocity_field(x_t, atom_types, edge_index, t)
        target = self.path.target_velocity(t, x_1=x_1, x_0=x_0)  # = x_1 - x_0
        loss = ((v_pred - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def generate(
        self,
        pocket: Pocket,
        config: GenerationConfig,
    ) -> list[Molecule]:
        """Sample N molecules by integrating the learned ODE."""
        from flow_matching.integrator import EulerSimulator
        # TODO: build edge_index, edge_mask from pocket
        edge_index = ...  # from pocket
        edge_mask = ...

        # 1. Sample noise
        # For each of n_samples, sample x_0 ~ N(0, I) of size (n_atoms, 3)
        x_0 = torch.randn(config.n_samples, pocket.n_atoms, 3, device=self._device)
        atom_types = ...  # from pocket?

        # 2. Integrate ODE (Lipman 2023)
        t_steps = torch.linspace(0, 1, config.n_steps + 1, device=self._device)
        # The library's simulator expects a ModelWrapper
        from flow_matching.solver import ModelWrapper
        class _Wrapper(ModelWrapper):
            def forward(self, x_t, t):
                return self.velocity_field(x_t, atom_types, edge_index, t)
        wrapper = _Wrapper(self.velocity_field)
        trajectory = EulerSimulator().simulate(x_0, t_steps, wrapper)
        # trajectory: (n_steps+1, n_samples, n_atoms, 3)

        # 3. Decode atom types (separate from coords)
        # TODO: use pocket-conditioned decoder

        # 4. Build Molecule objects
        mols = []
        for i in range(config.n_samples):
            mols.append(Molecule(
                coords=trajectory[-1, i],  # final step
                atom_types=atom_types[i],
                bonds=torch.zeros(2, 0, dtype=torch.long),  # TODO
                bond_types=torch.zeros(0, dtype=torch.long),
                formal_charges=torch.zeros_like(atom_types[i]),
            ))
        return mols

    def get_metadata(self) -> dict:
        return {
            "model": "LipmanFlowMatching_v1",
            "paper": "Lipman et al. 2023, ICLR 2023",
            "arxiv": "2210.02747",
            "official_code": "github.com/facebookresearch/flow_matching",
            "path": "AffineLinearPath (Lipman 2023 §4.8)",
            "scheduler": "CondOTScheduler (Lipman 2023 §4.7)",
            "loss": "ConditionalFlowMatchingLoss (Lipman 2023 §4.5)",
            "velocity_field": "EGNN (MolFlow-Triton)",
        }
```

## 训练入口(轻量,只验证管道)

```python
# molmetal/scripts/train_flow_matching.py
"""Smoke-test: train LipmanFMAdapter on toy 8-atom molecules.

Goal: verify forward+backward+sample roundtrip works end-to-end
       with the cloned facebookresearch/flow_matching library.
"""
import sys
import torch

sys.path.insert(0, ".")

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.domain import Pocket, Molecule


def make_toy_mol(seed=0):
    """Random 8-atom molecule, ~1Å bond lengths, in 2D plane + z jitter."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.zeros(8, 3)
    coords[:4, 0] = torch.arange(4).float() * 1.5
    coords[4:, 0] = torch.arange(4).float() * 1.5
    coords[:4, 1] = 0.0
    coords[4:, 1] = 2.0
    coords += torch.randn(8, 3, generator=g) * 0.1
    return Molecule(
        coords=coords,
        atom_types=torch.randint(1, 10, (8,)),
        bonds=torch.tensor([[0,1,2,3,4,5,6],[1,2,3,4,5,6,7]], dtype=torch.long),
        bond_types=torch.tensor([1,1,1,1,2,1,1], dtype=torch.long),
        formal_charges=torch.zeros(8, dtype=torch.long),
    )


def main():
    adapter = LipmanFlowMatchingAdapter(
        ref_repo_path="molmetal/references/flow_matching",
        hidden_dim=128,
    )
    adapter.setup(device="cuda")

    # 32 toy molecules
    mols = [make_toy_mol(i) for i in range(32)]
    pocket = Pocket(  # dummy pocket
        pdb_id="toy",
        coords=torch.zeros(10, 3),
        atom_types=torch.zeros(10, dtype=torch.long),
        residue_ids=torch.zeros(10, dtype=torch.long),
        chain_ids=torch.zeros(10, dtype=torch.long),
        mask=torch.ones(10, dtype=torch.bool),
        center=torch.zeros(3),
        radius=6.0,
    )

    for step in range(200):
        loss = adapter.train_step(pocket, mols)
        if step % 20 == 0:
            print(f"step {step}: loss = {loss:.4f}")

    print("\n=== Generate 5 molecules ===")
    samples = adapter.generate(pocket, GenerationConfig(n_samples=5, n_steps=20))
    for i, s in enumerate(samples):
        print(f"  mol {i}: {s.coords.shape} atoms, mean |coord| = {s.coords.abs().mean():.3f}")


if __name__ == "__main__":
    main()
```

## 预期 smoke-test 结果

- step 0: loss ~ 1.0 (random velocity predicts noise)
- step 100: loss ~ 0.3
- step 200: loss ~ 0.1
- 生成的分子坐标应该在 toy 数据附近(不在 0 因为还没训够)

如果这些都过,Phase 0 验收。

## 优先级与时间

| Phase | 内容 | 估计时间 |
|---|---|---|
| **0a** | Clone + domain dataclasses + port + mock | 3 天 |
| **0b** | LipmanFMAdapter skeleton + 训练脚本 + toy smoke-test | 3 天 |
| **1** | 接 DiffDock + closed loop + 5-pocket benchmark | 2 周 |
| **2** | 真实 MMP 数据 + 算法优化 (mini-batch OT, guidance) | 2 周 |
| **3** | 金属特化 + 论文级 benchmark | 2-3 周 |

**总 6-8 周达到论文可发表状态**。

## 与 MolFlow-Triton 的边界

**MolFlow-Triton** 是**基础研究**:
- flow_matching 原理验证 (QM9 conditional generation)
- Triton kernel autograd 修复
- 已经跑通且有 baseline 数字

**molmetal** 是**应用层**:
- 用 Lipman 2023 官方库 + MolFlow-Triton 的 EGNN → 3D 分子生成
- 接 docking + 闭环
- 算法优化在 Lipman library 之上做

**molmetal 不重写 MolFlow-Triton 的东西**,但 import 它的 EGNN / scatter / _scatter 复用。

## 待你回答的 3 个问题

1. **是否同时 clone DiffDock**(还是先纯 Generator,docking 以后)?
   - (a) 先 FM(2 周)— 跑通生成,再 docking
   - (b) 一起 clone(3-4 周)— 完整闭环
2. **第一靶点集**(沿用 MMP2/9 推荐)?
   - (a) MMP2/9 (推荐)
   - (b) CrossDocked100(通用)
3. **接 DiffDock 用什么 docker port**:
   - (a) clone 整个 DiffDock(占用大)
   - (b) 只用 DiffDock 的 inference 模块,其他不要
   - (c) 暂时用 Vina(纯 CPU,慢)作为 docking baseline

回答后开 Phase 0a。