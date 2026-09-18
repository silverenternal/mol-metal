# De Novo + SBDD 闭环设计 — Architecture

## 总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    DesignLoop (orchestration)                  │
│                                                                 │
│  ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    │
│  │Generator │ →  │Dock top │ →  │ Predict  │ →  │  Score  │ ──┐│
│  │(FLOWR)   │    │  -K     │    │(EGNN)    │    │(weight) │   ││
│  └──────────┘    └─────────┘    └──────────┘    └─────────┘   ││
│       ↑                                                       ││
│       │                ┌──────────────┐                        ││
│       └────────────────│  Refinement  │────────────────────────┘│
│                        │  (re-generate) │                        │
│                        └──────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
```

## Phase 0: 简单开环(基线)

```
input: Pocket
       │
       ▼
  Generator (FLOWR or TargetDiff)  ──→  N=1000 Molecules
       │
       ▼
  Top-K (K=100) by QED (cheap)
       │
       ▼
  Docking (DiffDock or EquiBind)  ──→  100 Complexes with poses
       │
       ▼
  Scoring (weighted: Vina + QED + SA)  ──→  ranked top-100
       │
       ▼
  output: top-10 candidates with poses + scores
```

**优点**:简单,易实现
**缺点**:第一次生成的所有分子可能都不在口袋里 — 需要 refine

## Phase 1: 闭环 refine

```
Iter 1: pocket → FLOWR → 1000 mols → dock top-100 → rank
Iter 2: 拿 top-10 mols 的 SMILES → FLOWR 重新生成(以 top-10 为种子 / 条件)
       → 1000 新 mols → dock → rank
Iter 3: 同上
output: 每代 top-1, 跨代 best
```

**优点**:逐步收敛到好分子
**缺点**:
- 容易 mode collapse(所有分子越来越相似)
- 需要 careful diversity regularization

## Phase 2: 实用两步(行业主流)

```
Step A: 大规模生成(offline)
  pocket → FLOWR (n=10,000) → raw Molecules
  → Docking (Vina/EquiBind for speed) → Vina scores
  → Top-100 candidates

Step B: 精修(online)
  Top-100 → DiffDock (high quality, slow) → 重新对接 + confidence
  → PropertyPredictor (EGNN for IC50) → 综合评分
  → Top-10 输出

优点: 速度快 + 质量高
缺点: Step A 和 Step B 需要不同的工具
```

## Phase 3: 金属特化 (我们的差异化)

```
Input: MMP2 口袋 (Zn²⁺ binding site) + 候选配体
  │
  ▼
MetalCoordinationAdapter: 枚举 Zn-N/O 配位组合
  │
  ▼
生成含金属配位的复合物(MoleculeWithMetal)
  │
  ▼
Docking (DiffDock with custom metal-aware scoring)
  │
  ▼
Scoring: Vina + Zn-binding score + IC50(EGNN)
  │
  ▼
Top-10 Zn-MMP2 抑制剂候选
```

## 评估指标

### Per-pocket 评估
- **Vina score** (kcal/mol, lower is better)
- **PoseBusters validity** (化学合理性)
- **QED** (drug-likeness, 0-1)
- **SA score** (synthetic accessibility, 0-1)
- **LogP, MW, HBA, HBD, RotB** (Lipinski)
- **Top-K success rate** (top-1 / top-3 / top-10 having Vina < -8.0)

### Cross-pocket 综合
- **Per-pocket AUC** (active vs decoy classification)
- **Recall@K** (known actives retrieved in top-K)
- **Diversity** (Tanimoto distance between top-K)
- **Novelty** (max Tanimoto to training set)

### 闭环特有
- **Convergence** (top-1 stability across iterations)
- **Diversity preservation** (mean pairwise Tanimoto)
- **Improvement curve** (best score vs iteration)

## 与 MolFlow-Triton 集成

```
MolFlow-Triton (flow matching infrastructure)
        ↓
molmetal.adapters.flour.flowr_runner
        ↓
molmetal.adapters.flour.adapter (FlowrAdapter : MoleculeGenerator)
        ↓
molmetal.ports.generator (Protocol)
        ↓
molmetal.orchestration.closed_loop (uses MoleculeGenerator)
```

复用部分:
- `models._scatter.scatter_sum`(Triton 算子)
- `models.velocity_net.EGNNLayer`(3D equivariant message passing)
- `flow_matching.loss.ConditionalFlowMatchingLoss`
- `flow_matching.sampler.FlowMatchingSampler`(如果换成 flow matching SBDD)
- `data._base.BaseMoleculeDataset`(自定义 dataset 基类)
- `utils.chem_utils.positions_to_smiles`(验证生成化学合理性)

## 文件结构 (初步)

```
molmetal/
├── __init__.py
├── domain/
│   ├── __init__.py
│   ├── pocket.py            # Pocket dataclass + PDB loader
│   ├── molecule.py          # Molecule dataclass + RDKit interop
│   └── complex.py           # Complex dataclass
├── ports/
│   ├── __init__.py
│   ├── generator.py         # MoleculeGenerator Protocol
│   ├── docking.py          # DockingEngine Protocol
│   ├── predictor.py         # PropertyPredictor Protocol
│   ├── scorer.py           # ScoringFunction Protocol
│   ├── design_loop.py      # DesignLoop Protocol
│   └── metal_complex.py    # MetalLigandGenerator Protocol
├── adapters/
│   ├── __init__.py          # ADAPTERS dict
│   ├── flour/
│   │   ├── __init__.py
│   │   ├── adapter.py       # FlowrAdapter
│   │   └── io.py           # Pocket <-> FLOWR input
│   ├── targetdiff/
│   ├── diffdock/
│   ├── equibind/
│   ├── rdkit_predictor.py
│   └── egnn_predictor.py
├── orchestration/
│   ├── __init__.py
│   ├── closed_loop.py
│   └── two_step.py
├── references/              # git cloned projects
│   ├── FLOWR/
│   ├── DiffDock/
│   └── TargetDiff/
├── scripts/
│   ├── run_loop.py
│   ├── eval_loop.py
│   └── report.py
└── tests/
    ├── test_ports.py
    ├── test_adapters.py
    └── test_e2e.py
```

## 测试策略

### 单元测试
```python
# test_ports.py
def test_generator_protocol():
    for name, cls in ADAPTERS.items():
        if "generator" not in name: continue
        gen = cls("tests/mock", "tests/mock.ckpt")
        assert isinstance(gen, MoleculeGenerator)

# test_adapters.py
def test_rdkit_predictor():
    pred = RDKitPredictor()
    pred.setup()
    mol = Molecule.from_smiles("CCO")
    p = pred.predict(mol)
    assert 0 <= p.qed <= 1
    assert p.mol_weight > 0

# test_e2e.py
def test_loop_with_mock():
    """ClosedLoop with mock adapters should run end-to-end."""
    gen = MockGenerator(returning=mock_mols(5))
    doc = MockDocker(returning=mock_complexes(5))
    pred = RDKitPredictor()
    sco = WeightedSumScorer()
    loop = ClosedLoop(gen, doc, pred, sco)
    result = loop.run(mock_pocket(), DesignLoopConfig(n_iterations=1, n_samples_per_round=5))
    assert len(result) > 0
    assert result[0].rank == 1
```

## 实施阶段 (按 Phase)

### Phase 0 (1 周)
- [x] TODO/ 规划文档(已完成)
- [ ] Clone 3 个参考仓库
- [ ] 写 domain dataclasses
- [ ] 写 ports (Protocol)
- [ ] 写 mock adapters
- [ ] 写 RDKit adapter for predictor
- [ ] 写 closed loop
- [ ] E2E test pass

### Phase 1 (2 周)
- [ ] 写 FlowrAdapter (real FLOWR integration)
- [ ] 写 DiffDockAdapter
- [ ] 跑 CrossDocked100 baseline
- [ ] 跟 FLOWR 论文数字对比

### Phase 2 (1 周)
- [ ] 写 EGNNPropertyPredictor (用我们已有的 EGNN)
- [ ] 集成到 closed loop
- [ ] 跑 MMP2/9 靶点测试
- [ ] 评估金属特化 (含 Zn 配位的 complex)

### Phase 3 (2 周)
- [ ] 写 MetalCoordinationAdapter
- [ ] 接 KRAS / BRAF 难例
- [ ] Benchmark 报告
- [ ] 论文草稿

## 风险

| 风险 | 概率 | 缓解 |
|---|---|---|
| FLOWR 不兼容 ROCm | 中 | 装 CPU 跑小的测试;GPU 仅在跑 benchmark 时用 |
| FLOWR 接口太复杂适配难 | 高 | 写一个 thin wrapper 暴露 1-2 个核心方法 |
| DiffDock 显存爆 | 中 | EquiBind 作为 fallback |
| 对接 score 与 IC50 相关性弱 | 中 | 报告里讨论,作 Vina 局限说明 |
| 我们 EGNN 没训 | 已完成(MolFlow-Triton) | 复用 |
| 闭环 mode collapse | 中 | diversity penalty |

## 现在该你回答的 3 个问题

1. **第一靶点集**:
   - (a) 通用 CrossDocked100 (paper 比较)
   - (b) PARP1(癌症药经典)
   - (c) BRAF / KRAS(难例)
   - (d) **MMP2/9**(金属依赖,与贵金属契合,niche)
2. **clone 顺序确认**:
   - (a) FLOWR → DiffDock → TargetDiff
   - (b) DiffDock → FLOWR → TargetDiff
3. **接口风格**:
   - (a) `typing.Protocol`(PEP 544,推荐)
   - (b) `abc.ABC`(显式 abstract)
   - (c) Pydantic BaseModel(带 schema 校验)

回答这三个,我开始 clone + 写代码。

### Multi-reward pipeline 2026-09-11

闭环 Scoring 阶段从单一加权 Vina+QED+SA 升级为多通道 reward aggregator,形式为 `r = w1·vina + w2·sa + w3·posebusters + w4·pic50 + w5·retro`(对应 `molmetal/molmetal_lam/search_alg/proof_search.py` 中的 `RewardAggregator`,Vina 自动取负、SA 由 [1,10] 倒置为 [0,1])。集成点:Phase 0/1/2 的 Scoring 节点直接替换为 `RewardAggregator` 实例,权重由 `molmetal/orchestration/closed_loop.py` 接收 `DesignLoopConfig.reward_weights` 注入;每通道独立 try/except,失败回 0.0 防止 docking/property predictor 崩溃污染整轮。Phase 3 金属特化场景额外叠加 `bonus_typed`(含 Zn-N/O 配位)+`bonus_binder` 项。闭环 ranking 复用 MCTS 的 `score_final`,从而把 de novo/SBDD 设计与 lambda 演算 proof search 的多 reward 通道统一为同一评分语义,便于跨任务对比 top-K 与收敛曲线。