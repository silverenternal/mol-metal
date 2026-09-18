# 抽象接口(Ports) — Hexagonal Architecture

> **Status (2026-09-11):** synced with code in `molmetal/ports/__init__.py`,
> `molmetal/adapters/`, `molmetal/baselines/`, and `molmetal/orchestration/`.
> Cross-referenced with `TODO/04_architecture/model_design.md`,
> `TODO/11_design_loop/closed_loop_design.md`, and
> `TODO/12_flow_matching/integration_with_molflow.md`.

## 设计原则

- **我们的代码定义接口**(port),**不依赖**任何具体 SOTA 项目
- 别人的项目是 **adapter**(plug-in implementation)
- 想换底层模型 = 改一行 import
- 单元测试用 **mock adapter**

## 1. Port 表 (实际实现于 `molmetal/ports/__init__.py`)

`molmetal.ports.__init__` 现在一共暴露 **5 个 Port (typing.Protocol)** + **5 个 config dataclass**:

| # | Port (Protocol)         | Config dataclass                | 关键方法                                                                 | 状态 |
|---|-------------------------|---------------------------------|--------------------------------------------------------------------------|------|
| 1 | `MoleculeGenerator`     | `GenerationConfig`              | `setup`, `generate`, `train_step`, `name`, `get_metadata`                | REAL |
| 2 | `DockingEngine`         | `DockingConfig`                 | `setup`, `dock`, `name`, `get_metadata`                                  | REAL |
| 3 | `PropertyPredictor`     | `PropertyPrediction` (out)      | `setup`, `predict`, `name`, `get_metadata`                               | REAL |
| 4 | `ScoringFunction`       | `ScoredCandidate` (out)         | `setup`, `score`, `name`, `get_metadata`                                 | REAL |
| 5 | `DesignLoop`            | `DesignLoopConfig`              | `run`, `get_history`, `name`, `setup`                                    | REAL (协议定义); 编排实现见 `molmetal/orchestration/` |

> 历史 doc 里出现过 `Refiner` port,目前 **不在 v1 ports 中** — refinement 通过
> `DesignLoop.run()` 的多轮 iterate 实现,无需独立 Protocol。

## 2. 接口一览 (sync'd)

```
Pocket          Molecule        Protein-Ligand
  │                │                Complex
  │                │                │
  ▼                ▼                ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Pocket  │  │ Molecule │  │ Complex  │  domain dataclasses
└──────────┘  └──────────┘  └──────────┘
       ▲              ▲            ▲
       │              │            │
┌──────┴──────────────┴────────────┴───────┐
│                                          │
│  MoleculeGenerator (port)                │
│  DockingEngine       (port)              │  port interfaces
│  PropertyPredictor   (port)              │
│  ScoringFunction     (port)              │
│  DesignLoop          (port)              │
│                                          │
└────────┬─────────────┬─────────────┬──────┘
         │             │             │
    ┌────┴──┐      ┌───┴──┐     ┌────┴───┐
    │Lipman │      │Diff- │     │RDKit / │  concrete
    │FM     │      │Dock  │     │EGNN /  │  adapters
    │Adptr  │      │Equi- │     │Mock    │  (cloned
    │       │      │Bind  │     │        │  or stub)
    └───────┘      └──────┘     └────────┘
```

## 3. Domain dataclasses (不可变数据)

```python
# molmetal/domain/pocket.py
@dataclass(frozen=True)
class Pocket:
    pdb_id: str
    coords: torch.Tensor          # (N_atoms, 3) 蛋白质原子坐标
    atom_types: torch.Tensor      # (N_atoms,) 原子序数
    residue_ids: torch.Tensor     # (N_atoms,) 残基号
    chain_ids: torch.Tensor       # (N_atoms,) 链 ID
    mask: torch.Tensor            # (N_atoms,) 1 = backbone/sidechain atom
    center: torch.Tensor          # (3,) binding site centroid
    radius: float = 6.0           # Å

    @property
    def n_atoms(self) -> int:
        return self.coords.shape[0]

    def to(self, device) -> "Pocket": ...  # immutable → return new

    @classmethod
    def from_pdb_file(cls, path: str, ligand_center: torch.Tensor,
                      radius: float = 6.0) -> "Pocket":
        """Parse PDB and extract pocket around ligand center."""

# molmetal/domain/molecule.py
@dataclass(frozen=True)
class Molecule:
    """3D molecular structure."""
    coords: torch.Tensor          # (N_atoms, 3)
    atom_types: torch.Tensor      # (N_atoms,)
    bonds: torch.Tensor          # (2, N_bonds)
    bond_types: torch.Tensor      # (N_bonds,)
    formal_charges: torch.Tensor  # (N_atoms,)
    smiles: str = ""
    qed: float = 0.0
    sa_score: float = 0.0
    logp: float = 0.0

    @property
    def n_atoms(self) -> int:
        return self.coords.shape[0]

    def to(self, device) -> "Molecule": ...

    @classmethod
    def from_rdkit_mol(cls, mol, conformer_id=0) -> "Molecule": ...

    @classmethod
    def from_smiles(cls, smiles: str) -> "Molecule":
        """Embed 3D with RDKit ETKDGv3 + MMFF."""

    def to_rdkit(self) -> "Chem.Mol": ...
    def to_sdf(self, path: str) -> None: ...

# molmetal/domain/complex.py
@dataclass(frozen=True)
class Complex:
    """Protein-ligand complex with predicted pose."""
    pocket: Pocket
    molecule: Molecule
    pose_confidence: float = 0.0
    vina_score: Optional[float] = None
    binding_affinity: Optional[float] = None  # pIC50 / pKi
```

## 4. Generator port (de novo design)

```python
# molmetal/ports/__init__.py
from typing import Protocol, List, Optional
from dataclasses import dataclass, field

@dataclass(frozen=True)
class GenerationConfig:
    n_samples: int = 100
    n_steps: int = 50
    temperature: float = 1.0
    conditioning: dict = field(default_factory=dict)  # pocket embeddings etc.
    seed: int = 42


class MoleculeGenerator(Protocol):
    """Port: de novo molecule generation conditioned on a pocket."""

    @property
    def name(self) -> str:
        """Identifier: 'LipmanFlowMatching_v1', 'MockGenerator_v0', etc."""
        ...

    def setup(self, device: str = "cuda") -> None:
        """Load weights, prepare buffers. Called once at startup."""
        ...

    def generate(
        self,
        pocket: Pocket,
        config: GenerationConfig,
    ) -> List[Molecule]:
        """Generate `n_samples` molecules fitting the pocket."""
        ...

    def train_step(
        self,
        pocket: Pocket,
        mols: List[Molecule],
    ) -> float:
        """One optimisation step; returns the loss value (for logging)."""
        ...

    def get_metadata(self) -> dict:
        """Return training dataset, config, citation info (for reproducibility)."""
        ...


# Concrete adapter: LipmanFlowMatchingAdapter
# (molmetal/adapters/flow_matching_lipman/__init__.py)
class LipmanFlowMatchingAdapter(MoleculeGenerator):
    """Wraps facebookresearch/flow_matching (Lipman et al. 2023 ICLR)."""

    def __init__(self, ref_repo_path: str = "molmetal/references/flow_matching",
                 hidden_dim: int = 128, n_layers: int = 3,
                 max_atomic_number: int = 100, lr: float = 1e-4,
                 atom_loss_weight: float = 0.1): ...

    def setup(self, device=None): ...   # auto-detect ROCm/CUDA via molmetal.utils.device
    def generate(self, pocket, config): ...   # joint atom-type + coord sampling (Categorical)
    def train_step(self, pocket, mols, atom_loss_weight=None): ...   # CFM + masked CE
```

**Joint atom-type + coord FM (NEW)**:`LipmanFlowMatchingAdapter` 同时预测
per-atom **坐标速度** 和 **原子序数 logits**:

- `EGNNVelocityField.forward` 返回 `{"vel", "atom_logits", "h"}`
- `train_step` 返回 **float total loss** = CFM + α · atom-CE
  (Protocol 契约);分量写入 `self.last_losses = {"cfm", "atom", "total"}`
- `generate` 用 `Categorical(probs=softmax(atom_logits))` 采样真实原子序数
  (placeholder `randint(1, 10)` 已删除);Z=0 的 padding 位置 logits 被置为 -inf

## 5. Docking port

```python
# molmetal/ports/__init__.py
@dataclass(frozen=True)
class DockingConfig:
    n_poses: int = 10             # top-K poses to return
    exhaustiveness: int = 8
    use_confidence: bool = True
    seed: int = 42


class DockingEngine(Protocol):
    """Port: predict ligand binding pose(s) in a pocket."""

    @property
    def name(self) -> str: ...
    def setup(self, device: str = "cuda") -> None: ...
    def dock(
        self,
        molecule: Molecule,
        pocket: Pocket,
        config: DockingConfig,
    ) -> List[Complex]:
        """Return top-N poses (each is a Complex with transformed molecule)."""
        ...
    def get_metadata(self) -> dict: ...


# Concrete adapters (Phase-1 STUB):
class DiffDockAdapter(DockingEngine):
    """Corso et al. ICLR 2023. STUB: random SE(3) + U(-12,-4) vina_score."""

class EquiBindAdapter(DockingEngine):
    """Stärk et al. NeurIPS 2022. STUB: random SE(3) + U(-11,-5) vina_score."""

class MockDocker(DockingEngine):
    """Random SE(3) pose sampler with constant-ish vina_score."""
```

两个 STUB 的差异(用于基准比较):
- DiffDock STUB 的 translation σ = 2 Å;pose_confidence ~ U(0, 1)
- EquiBind STUB 的 translation σ = 3 Å;pose_confidence ~ U(0.4, 1.0)

## 6. Property predictor port

```python
# molmetal/ports/__init__.py
@dataclass(frozen=True)
class PropertyPrediction:
    qed: float = 0.0
    sa_score: float = 0.0
    logp: float = 0.0
    mol_weight: float = 0.0
    tpsa: float = 0.0
    num_h_donors: int = 0
    num_h_acceptors: int = 0
    num_rotatable_bonds: int = 0
    binding_affinity_pic50: Optional[float] = None  # from EGNN/D-MPNN
    metal_binding_score: Optional[float] = None     # for our metal extension


@runtime_checkable
class PropertyPredictor(Protocol):
    """Port: predict 2D/3D properties of a Molecule (or Complex)."""

    @property
    def name(self) -> str: ...
    def setup(self, device: str = "cuda") -> None: ...
    def predict(
        self,
        molecule: Molecule,
        complex: Optional[Complex] = None,
    ) -> PropertyPrediction: ...
    def get_metadata(self) -> dict: ...


# Concrete adapters:
class RDKitPropertyPredictor:
    """REAL — RDKit descriptors (QED, logP, MW, TPSA, Lipinski, SA score)."""

class EGNNPropertyPredictor:
    """STUB — allocates EGNN, loads checkpoint if provided; pIC50 inference TODO."""

class MockPredictor:
    """REAL-for-tests — RDKit QED/logP + random sa_score + synthetic pIC50."""
```

**RDKit PropertyPredictor 字段覆盖**:
- `qed` (Descriptors.qed), `logp` (Crippen.MolLogP), `mol_weight` (Descriptors.MolWt)
- `tpsa` (rdMolDescriptors.CalcTPSA)
- `num_h_donors`, `num_h_acceptors`, `num_rotatable_bonds` (Lipinski.*)
- `sa_score`:`rdkit.Contrib.SA_Score.sascorer` if importable,否则 fallback 到
  `1 / (1 + NumAromaticRings)` proxy(在 `get_metadata` 中标注 `sa_score_source`)
- `binding_affinity_pic50` / `metal_binding_score`: **始终 `None`**(留给 EGNN)

## 7. Scorer port (composite)

```python
# molmetal/ports/__init__.py
@dataclass(frozen=True)
class ScoredCandidate:
    molecule: Molecule
    complex: Optional[Complex]
    property_pred: PropertyPrediction
    combined_score: float
    rank: int = 0


class ScoringFunction(Protocol):
    """Port: combine multiple signals into a single ranking score."""

    @property
    def name(self) -> str: ...
    def setup(self) -> None: ...
    def score(
        self,
        candidates: List[Tuple[Molecule, Optional[Complex], PropertyPrediction]],
    ) -> List[ScoredCandidate]: ...
    def get_metadata(self) -> dict: ...


# Concrete: MockScorer (weighted-sum, default w_qed=w_sa=w_pic50=w_vina=1.0)
class MockScorer(ScoringFunction):
    """combined = w_qed*qed + w_sa*sa + w_pic50*pic50 − w_vina*vina
    Sorts descending and assigns 1-indexed rank."""
```

`ScoredCandidate` 在协议里 **不带独立 `vina_score` / `binding_pic50` 字段**;
下游 scorer 通过 `complex.vina_score` / `property_pred.binding_affinity_pic50`
访问(对比老 doc 的字段冗余)。这个差异已在 `mock.MockScorer` 中实现。

## 8. DesignLoop port (orchestration)

```python
# molmetal/ports/__init__.py
@dataclass(frozen=True)
class DesignLoopConfig:
    n_iterations: int = 3                # outer refinement rounds
    n_samples_per_round: int = 1000
    n_top_k: int = 100                    # keep top-K between rounds
    convergence_threshold: float = 0.01
    seed: int = 42


class DesignLoop(Protocol):
    """Port: orchestrate generate → score → refine cycles."""

    def __init__(
        self,
        generator: MoleculeGenerator,
        docker: DockingEngine,
        predictor: PropertyPredictor,
        scorer: ScoringFunction,
    ): ...

    @property
    def name(self) -> str: ...
    def setup(self, device: str = "cuda") -> None: ...
    def run(
        self, pocket: Pocket, config: DesignLoopConfig
    ) -> List[ScoredCandidate]: ...
    def get_history(self) -> List[dict]: ...
```

**实际实现** 在 `molmetal/orchestration/closed_loop.py`(ClosedLoop / LamClickDesignLoop)
和 `molmetal/orchestration/design_loop.py`(Phase-0 baseline)。详见
`TODO/11_design_loop/closed_loop_design.md`。

## 9. Adapter 清单 (实际文件 → 状态)

| 文件                                                       | 类                          | Port 实现                    | 状态                     |
|------------------------------------------------------------|-----------------------------|------------------------------|--------------------------|
| `molmetal/adapters/flow_matching_lipman/__init__.py`       | `LipmanFlowMatchingAdapter` | `MoleculeGenerator`          | **REAL** — joint atom-type + coord FM |
| `molmetal/adapters/rdkit_predictor.py`                     | `RDKitPropertyPredictor`    | `PropertyPredictor`          | **REAL** — RDKit descriptors |
| `molmetal/adapters/egnn_predictor.py`                      | `EGNNPropertyPredictor`     | `PropertyPredictor`          | **STUB** — checkpoint loader only, no pIC50 inference |
| `molmetal/adapters/egnn_rocm.py`                           | `EGNN` (SE(3) layer)        | n/a (内部组件)               | REAL — ROCm-friendly scatter_sum |
| `molmetal/adapters/diffdock.py`                            | `DiffDockAdapter`           | `DockingEngine`              | **STUB** — random SE(3) poses + U(-12,-4) vina |
| `molmetal/adapters/equibind.py`                            | `EquiBindAdapter`           | `DockingEngine`              | **STUB** — random SE(3) poses + U(-11,-5) vina |
| `molmetal/adapters/mock.py`                                | `MockGenerator`             | `MoleculeGenerator`          | REAL-for-tests — 8-atom random molecules |
| `molmetal/adapters/mock.py`                                | `MockDocker`                | `DockingEngine`              | REAL-for-tests — random SE(3) |
| `molmetal/adapters/mock.py`                                | `MockPredictor`             | `PropertyPredictor`          | REAL-for-tests — RDKit + synthetic |
| `molmetal/adapters/mock.py`                                | `MockScorer`                | `ScoringFunction`            | REAL-for-tests — weighted sum |
| `molmetal/baselines/dmpnn.py`                              | (D-MPNN trainer)            | n/a — standalone trainer     | **REAL** — vanilla D-MPNN baseline |
| `molmetal/baselines/dmpnn_attentive.py`                    | (AttentiveFP-style)         | n/a — standalone trainer     | **REAL** — attentive D-MPNN baseline |
| `molmetal/baselines/morgan_xgb.py`                         | (Morgan + XGBoost)          | n/a — standalone trainer     | REAL — fingerprint baseline |
| `molmetal/baselines/rf_baseline.py`                        | (Random Forest)             | n/a — standalone trainer     | REAL — RF baseline |

注意:**`molmetal/baselines/` 不实现 port**;它们是独立的 trainer/CLI,产 model
checkpoint 供下游 (例如 `EGNNPropertyPredictor(checkpoint_path=...)`) 加载。
正式把它们包装成 port adapters 是后续 Phase 的工作。

## 10. Port Compliance Matrix (适配器 → 协议)

| Adapter                          | MoleculeGenerator | DockingEngine | PropertyPredictor | ScoringFunction | DesignLoop |
|----------------------------------|-------------------|---------------|-------------------|------------------|------------|
| `LipmanFlowMatchingAdapter`      | YES (REAL)        | —             | —                 | —                | —          |
| `RDKitPropertyPredictor`         | —                 | —             | YES (REAL)        | —                | —          |
| `EGNNPropertyPredictor`          | —                 | —             | YES (STUB)        | —                | —          |
| `DiffDockAdapter`                | —                 | YES (STUB)    | —                 | —                | —          |
| `EquiBindAdapter`                | —                 | YES (STUB)    | —                 | —                | —          |
| `MockGenerator`                  | YES (test)        | —             | —                 | —                | —          |
| `MockDocker`                     | —                 | YES (test)    | —                 | —                | —          |
| `MockPredictor`                  | —                 | —             | YES (test)        | —                | —          |
| `MockScorer`                     | —                 | —             | —                 | YES (test)       | —          |
| `ClosedLoop` (orchestration)     | —                 | —             | —                 | —                | YES (REAL) |

## 11. Design Decisions (近期变更)

1. **Joint atom-type + coord FM** (2026-09)
   - `EGNNVelocityField` 增加 `atom_head = nn.Linear(hidden, max_z)` (zero-init)
   - `train_step` 增加 masked cross-entropy over atom types
   - `generate` 用 `Categorical(softmax(atom_logits))` 替换 `randint(1, 10)`
     placeholder,Z=0 位置 logits = -inf
   - `train_step` 返回 `float total loss = CFM + α·atom-CE`(Protocol 契约),
     分量写入 `self.last_losses`

2. **ROCm-first device selection**
   - 新模块 `molmetal/utils/device.py`:`ROCM_AVAILABLE`, `DEFAULT_DEVICE`,
     `get_device()`, `verify_rocm_active()`, `device_guard(...)`
   - 所有 adapter 的 `setup()` 通过 `get_device()` 解析设备,不再硬编码 `"cpu"`
   - `LipmanFlowMatchingAdapter` 额外把 `device_info = verify_rocm_active()`
     存到 `self.device_info`,供 smoke test 断言 GPU 真在用

3. **`LipmanFlowMatchingAdapter.train_step` 签名契约**
   - 旧:返回 `dict` 包含 `{cfm, atom, total}`
   - 新:返回 `float = total` (Protocol 要求),分量写入 `self.last_losses`
   - 旧 docstring 里的 `List[float]` 概念已废弃

4. **`ScoredCandidate` 字段收敛**
   - 旧 doc 把 `vina_score` / `binding_pic50` 独立列在 `ScoredCandidate`
   - 现代码只保留 `molecule / complex / property_pred / combined_score / rank`;
     vina_score 从 `complex.vina_score` 取,pic50 从 `property_pred.binding_affinity_pic50` 取

5. **MockDocker SE(3) 一致性**
   - 旧 doc 用 N(0, 5) translation
   - 现 `mock.MockDocker` 仍用 N(0, 5);`DiffDockAdapter` 用 N(0, 2);`EquiBindAdapter` 用 N(0, 3)
   - 三者 vina_score / pose_confidence 分布不同,基准能区分

6. **`__init__.py` 集中化**
   - 所有 5 个 Port + 5 个 config dataclass 都从
     `molmetal/ports/__init__.py` 单一入口导出
   - 旧 doc 假设分文件 (`generator.py / docking.py / predictor.py / scorer.py / design_loop.py / metal_complex.py`)
     — **未拆分**,仅 `__init__.py` 一个文件

7. **`Refiner` port 废弃**
   - 旧 doc 列出独立 `Refiner` Protocol
   - v1 通过 `DesignLoop.run()` 多轮 `n_iterations` 内置 refinement;没有单独 port

8. **`metal_complex.MetalLigandGenerator` 延后**
   - 旧 doc 列出 `MetalCoordination` + `PtCoordinationAdapter`
   - v1 未实现(等 Phase-3 metal-priorit 工作启动)

## 12. Port Compliance Matrix 中的常见问题(后续工作)

- `EGNNPropertyPredictor.predict()` 实际 forward pass(TODO in `predict` docstring);
  目前无论有没有 checkpoint,`binding_affinity_pic50` 都返回 `None`
- `DiffDockAdapter` / `EquiBindAdapter` 实际推理路径在 `setup(checkpoint_path=...)` 时
  会抛 `NotImplementedError`;`checkpoint_path=None` 才走 STUB
- 没有 `Refiner` / `MetalLigandGenerator` port 实现

## 13. Adapter 注册表 (建议用法)

```python
# molmetal/adapters/__init__.py (sync'd)
from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.adapters.rdkit_predictor import RDKitPropertyPredictor
from molmetal.adapters.egnn_predictor import EGNNPropertyPredictor
from molmetal.adapters.diffdock import DiffDockAdapter
from molmetal.adapters.equibind import EquiBindAdapter
from molmetal.adapters.mock import (
    MockGenerator,
    MockDocker,
    MockPredictor,
    MockScorer,
)

# Each adapter implements one or more ports
ADAPTERS = {
    "generator.lipman_fm": LipmanFlowMatchingAdapter,
    "generator.mock": MockGenerator,
    "docking.diffdock": DiffDockAdapter,
    "docking.equibind": EquiBindAdapter,
    "docking.mock": MockDocker,
    "predictor.rdkit": RDKitPropertyPredictor,
    "predictor.egnn": EGNNPropertyPredictor,
    "predictor.mock": MockPredictor,
    "scorer.mock": MockScorer,
}
```

## 14. 测试策略

```python
# tests/test_ports.py
def test_generator_protocol():
    """Any MoleculeGenerator must implement generate(pocket, config) -> List[Molecule]."""
    for name, cls in ADAPTERS.items():
        if "generator" not in name:
            continue
        gen = cls(**_defaults_for(name))
        assert hasattr(gen, "setup")
        assert hasattr(gen, "generate")
        assert hasattr(gen, "name")
        assert hasattr(gen, "train_step")  # NEW in MoleculeGenerator Protocol

def test_docking_protocol():
    """Any DockingEngine must produce Complex with valid 3D coordinates."""
    ...

def test_predictor_protocol():
    """PropertyPredictor must satisfy runtime_checkable: isinstance() check works."""
    ...

def test_loop_runs_with_any_combination():
    """DesignLoop should work with any combination of adapters."""
    pocket = mock_pocket()
    for gen_name, doc_name in [("lipman_fm", "diffdock"), ("mock", "mock")]:
        gen = ADAPTERS[f"generator.{gen_name}"](...)
        doc = ADAPTERS[f"docking.{doc_name}"](...)
        loop = ClosedLoop(gen, doc, RDKitPropertyPredictor(), MockScorer())
        result = loop.run(pocket, DesignLoopConfig(n_iterations=1, n_samples_per_round=10))
        assert len(result) > 0
        assert result[0].rank == 1
```

## 15. 文件组织 (实际)

```
molmetal/
├── domain/                          # Pocket / Molecule / Complex dataclasses
├── ports/
│   └── __init__.py                  # ★ all 5 Protocols + 5 config dataclasses (single file)
├── adapters/
│   ├── __init__.py                  # (empty — import explicit submodules)
│   ├── flow_matching_lipman/
│   │   └── __init__.py              # LipmanFlowMatchingAdapter + EGNNVelocityField
│   ├── rdkit_predictor.py           # RDKitPropertyPredictor
│   ├── egnn_predictor.py            # EGNNPropertyPredictor (stub) + EGNNConfig
│   ├── egnn_rocm.py                 # minimal SE(3) EGNN (ROCm-friendly scatter_sum)
│   ├── diffdock.py                  # DiffDockAdapter (stub)
│   ├── equibind.py                  # EquiBindAdapter (stub)
│   └── mock.py                      # MockGenerator / MockDocker / MockPredictor / MockScorer
├── baselines/                       # standalone trainers — not port adapters
│   ├── dmpnn.py
│   ├── dmpnn_attentive.py
│   ├── morgan_xgb.py
│   ├── rf_baseline.py
│   └── eval_utils.py
├── orchestration/                   # ClosedLoop / design_loop implementations
│   ├── closed_loop.py
│   └── design_loop.py
├── models/                          # MolFlow-Triton EGNN / scatter / velocity_net (re-used)
├── utils/
│   └── device.py                    # ROCm-first helpers (NEW)
├── references/                      # cloned upstream code (DiffDock, EquiBind, flow_matching)
├── scripts/
└── tests/
```

## 16. 与 MolFlow-Triton 项目的边界

**复用**:
- `models._collision.scatter_sum`(Triton kernel)
- `models.velocity_net.EGNNLayer`
- `flow_matching.*`(loss / sampler / flow matching infrastructure)

**不污染 MolFlow-Triton**:
- `molmetal/` 是独立子目录
- 可以 import MolFlow-Triton 的模块,但反过来不行

## 17. 跨引用

- 架构总览 → `TODO/04_architecture/model_design.md`
- 闭环编排细节 → `TODO/11_design_loop/closed_loop_design.md`
- FM 与 MolFlow-Triton 整合 → `TODO/12_flow_matching/integration_with_molflow.md`
- 数据 / Splits → `molmetal/data/cytotox.py`, `molmetal/data/splits.py`
- 风险 & 待回答问题 → `TODO/07_risks/open_questions.md` (A1-A4, B1-B3, C1-C5, D1-D4, E1-E2, F1-F3, G1-G3)
- 里程碑勾选 → `TODO/06_milestones/milestones.md`
- 靶点选择 → `TODO/10_targets/target_selection.md`

## 18. 接下来

1. 推进 `EGNNPropertyPredictor` 实际 forward pass(等 checkpoint 训练落地)
2. Phase-2: `DiffDockAdapter` / `EquiBindAdapter` 接入真实 checkpoint(已 clone 仓库即可挂)
3. 把 `molmetal/baselines/dmpnn.py` 等包装成 `PropertyPredictor` port adapters
4. Phase-3: `MetalLigandGenerator` port + `PtCoordinationAdapter` 落地

## 19. 待回答 (legacy)

1. 哪个靶点集?通用 (CrossDocked) / 单个 / KRAS-MYC? → `TODO/10_targets/target_selection.md` 答:**MMP2/9 + CrossDocked100**
2. Clone 优先级 1-3: FLOWR / DiffDock / TargetDiff 都同意? → 当前已 clone **flow_matching**, DiffDock + EquiBind 在 `references/` 标记待 clone; TargetDiff **未 clone**
3. 接口用 Protocol / ABC / Pydantic? → **Protocol + frozen dataclass** (per `molmetal/ports/__init__.py`)
4. 开环 / 闭环 / 两步实用? → **闭环**(`DesignLoop.run()` 多轮 iterate);详见 `TODO/11_design_loop/closed_loop_design.md`