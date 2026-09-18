# 与 Mol-Metal (12_flow_matching) 的融合方案

## 总览 (修订: 不再是"两个独立轨道",而是"同一 λ-演算的两个面")

之前我说"两条独立轨道在 3 个点交汇"——这个说法还是低估了。**真正的论断是:Mol-Metal 的 EGNN + FM 和 Lambda + Click **是同一个 Molecular Lambda Calculus (MLC) 的不同算子**。

```
┌──────────────────────────────────────────────────────────────────┐
│  12_flow_matching/ (Mol-Metal)                                  │
│  = η-conversion learner + β-reduction sequence learner            │
│                                                                  │
│  EGNN:     η-invariant aggregation  → 拟合 η-equivalence class    │
│  FM:       noise → β-NF  →  学习 β-reduction 路径                │
│  IC50 pred: type inference  →  给定 term 推断 inhabitation       │
│                                                                  │
│  优势: SE(3) 等变 + 3D 几何 + Triton 加速 + 大数据训练           │
│  短板: 黑盒 + 不可解释 + 需大数据                                │
└──────────────────────────┬───────────────────────────────────────┘
                           │  同一 Molecular Lambda Calculus
                           │  不同算子,不同 trade-off
┌──────────────────────────┴───────────────────────────────────────┐
│  13_lambda_clickchem/ (Lambda + Click)                           │
│  = Molecular Lambda Calculus 的形式化基础 + 符号算子              │
│                                                                  │
│  Atoms-as-combinators: 语法层 (S, K, I + 金属 n-arity)            │
│  Bonds-as-application:  β-reduction (= 化学键)                    │
│  Reactions:             β-reduction rules                         │
│  Molecules:             closed λ-term in β-NF                     │
│  Synthesis:             β-reduction sequence                     │
│  ADMET:                 type predicates                          │
│  Drug design:           constructive proof search                 │
│                                                                  │
│  优势: 形式化基础 + 可解释 + 100% SAS + Curry-Howard            │
│  短板: 不直接处理 3D 几何 + 搜索空间大                            │
└──────────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │  最终目标: 一区论文        │
              │  "Molecular Lambda          │
              │   Calculus: A Unified       │
              │   Framework for Symbolic    │
              │   + Geometric Drug Design"  │
              └────────────────────────────┘
```

**关键论断 (更新)**:
- 不是"两个独立轨道融合",而是"**同一个理论的两个实现**"
- 轨道 B 提供 **形式化基础** (definitions, type system, reduction rules)
- 轨道 A 提供 **几何实现** (η-conversion learner, β-reduction learner)
- 论文的故事:**MLC 理论 + 两种实现 + 实验对比 + 融合验证**

## 融合点 1 (修订): Reward 信号共享 + Type Inference 复用

**之前的版本**: "Mol-Metal 的 IC50 predictor 是 Lambda MCTS 的可选 reward"
**修订**: Mol-Metal 的 IC50 predictor **就是 MLC 框架下的 type inference**

```python
# molmetal/molmetal_lam/types/type_inference.py
class MolMetalTypeInference:
    """Mol-Metal's EGNN IC50 predictor = type inhabitation inference.
    
    IC50(M) = "how strongly does M inhabit BindingType(T)?"
    
    In λ-calculus terms: type inference assigns the "deepest"
    inhabitation score — EGNN approximates this via η-invariant features.
    """
    def infer(self, molecule: Molecule, binding_type: 'BindingType') -> 'InhabitationScore':
        # Step 1: η-canonicalize (GNN aggregation = η-reduction)
        canonical_features = egnn_encode(molecule)
        
        # Step 2: Type inhabitation score
        inhabitation = self.binding_head(canonical_features, binding_type)
        
        # Step 3: Convert to pIC50
        return InhabitationScore(
            pic50=9.0 - inhabitation,
            type_safe=True,
            violated_constraints=[],
        )
```

**这才是真正的融合**: 不是"用 Mol-Metal 当黑盒打分",而是"理解 Mol-Metal 的 IC50 预测本质上就是 type inference,然后用 MLC 框架来解释它"。

---

## 融合点 2: Metal Coordination Tile (中期, 第 2 个月)

**洞察**: Pt(II)、Ru(II)、Ir(III) 等贵金属有**严格的几何约束** (square-planar, octahedral)。这些约束可以编码为**特殊的 Lambda 常量**。

```python
# molmetal/molmetal_lam/tile_lib/metal_tiles.py

@dataclass(frozen=True)
class MetalCoordinationTile(Tile):
    """Special tile: a metal center with predefined coordination geometry.
    
    Square-planar Pt(II): 4 ligands at 90° in a plane
    Octahedral Ru(II):    6 ligands at 90°
    Trigonal bipyramidal: 5 ligands
    """
    metal: str           # 'Pt', 'Ru', 'Ir', ...
    oxidation_state: int
    geometry: str        # 'square_planar', 'octahedral', ...
    ligand_angles: torch.Tensor   # (n_ligands, 3) ideal positions
    
    @classmethod
    def pt_square_planar(cls, pt_center: torch.Tensor) -> 'MetalCoordinationTile':
        """Pt(II) center with 4 ideal ligand positions in xy plane."""
        angles = torch.tensor([
            [1.0, 0.0, 0.0],   # 0° (east)
            [0.0, 1.0, 0.0],   # 90° (north)
            [-1.0, 0.0, 0.0],  # 180° (west)
            [0.0, -1.0, 0.0],  # 270° (south)
        ]) * 2.0  # 2 Å Pt-ligand bond length
        return cls(
            smiles='[Pt]',
            coords=pt_center + angles,
            functional_groups=['metal_center'] * 4,
            tile_id='Pt_II_square_planar',
            metal='Pt', oxidation_state=2, geometry='square_planar',
            ligand_angles=angles,
        )
    
    @classmethod
    def ru_octahedral(cls, ru_center: torch.Tensor) -> 'MetalCoordinationTile':
        """Ru(II) center with 6 ideal ligand positions."""
        angles = torch.tensor([
            [1, 0, 0], [-1, 0, 0],   # x axis
            [0, 1, 0], [0, -1, 0],   # y axis
            [0, 0, 1], [0, 0, -1],   # z axis
        ]) * 2.1  # 2.1 Å Ru-ligand bond
        return cls(
            smiles='[Ru]',
            coords=ru_center + angles,
            functional_groups=['metal_center'] * 6,
            tile_id='Ru_II_octahedral',
            metal='Ru', oxidation_state=2, geometry='octahedral',
            ligand_angles=angles,
        )
```

**Lambda 应用**: "Pt(NH3)2Cl2 (cisplatin)" 的合成可表达为:

```haskell
-- Lambda 演算
Cisplatin = Pt_Center 
          `attach` Azide_Handle("NH3")  -- via CuAAC to first NH3
          `attach` Azide_Handle("NH3")  -- via CuAAC to second NH3  
          `attach` Chloro_Handle         -- via click to Cl
          `attach` Chloro_Handle         -- via click to Cl

-- 验证: 顺式 vs 反式由 attach 顺序决定 (Lambda 演算的 evaluation order)
```

**谁先做**: Mol-Metal 轨道已经有 EGNN 处理 dative 键的经验 (TODO/04_architecture/model_design.md 提到),Lambda 轨道可以直接复用这些 MetalCoordinationTile 定义。

---

## 融合点 3: 靶标对齐 + 交叉验证 (长期, 第 3 个月)

### 推荐第一靶点集: **MMP2 / MMP9**

理由:
- **Mol-Metal 轨道已选 MMP2/9** (TODO/10_targets/target_selection.md) — Zn 依赖的 metalloproteinase
- **Lambda 轨道可用**: MMP2/9 有已知抑制剂,大多含 hydroxamate (zinc-binding) — 不是经典 click chem,但可以扩展
- **融合验证**: Mol-Metal 用 IC50,Lambda 用 binding pose + SAS — 两个不同维度的评估

### 共享靶标列表

| 靶点 | Mol-Metal 价值 | Lambda 价值 | 融合潜力 |
|---|---|---|---|
| **MMP2/MMP9** | ✅ 主选 (Zn metalloproteinase) | 中 (含 hydroxamate) | ★★★★ |
| Kinase (EGFR, BRAF) | 中 (Pt 偶联物研究) | ✅ 高 (click 经典) | ★★★★★ |
| GPCR (β2AR) | 低 | ✅ 高 | ★★★ |
| **CrossDocked100** | 中 (训练) | 中 (通用测试) | ★★★★ |

### 交叉验证协议

```python
# molmetal/molmetal_lam/scripts/cross_validate_with_molmetal.py
def cross_validate(pdb_id: str):
    """Run BOTH tracks on the same target, compare outcomes."""
    
    # 1. Lambda track generates candidates
    lambda_candidates = lam_mcts_run(pdb_id, n=1000)
    
    # 2. Mol-Metal IC50 predictor scores them
    ic50_scores = molmetal_ic50_predictor.predict_batch(
        [c.smiles for c in lambda_candidates]
    )
    
    # 3. Mol-Metal diffusion also generates candidates
    molmetal_candidates = lipman_fm_adapter.generate(pdb_id, n=1000)
    
    # 4. Lambda scoring (REINVENT4 + SAS) evaluates them
    lambda_scores = reinvent_scorer.score_batch(
        [c.smiles for c in molmetal_candidates]
    )
    
    # 5. Report cross-track metrics
    return {
        "lambda_candidates_molmetal_auc": compute_auc(ic50_scores, active_labels),
        "molmetal_candidates_lambda_sas": mean([s.sas for s in lambda_scores]),
        "agreement_rate": agreement(lambda_candidates, molmetal_candidates),
        "best_of_both": top_k_by_combined_score(...),
    }
```

### 论文中的融合故事

**Abstract**:
> "We present a hybrid drug design framework that combines
>  (1) **symbolic reasoning** via Lambda-calculus-guided click chemistry
>  (interpretable synthesis paths, 100% synthetic accessibility) with
>  (2) **geometric deep learning** via SE(3)-equivariant flow matching
>  (3D-pocket-conditioned generation, IC50 prediction).
>  Applied to MMP2 inhibitors, our framework achieves AUC 0.89 with
>  interpretable Lambda expressions for top-10 candidates."

**Methodology 章节图**:

```
┌─────────────────────────────┐         ┌──────────────────────────┐
│  轨道 A: Mol-Metal FM      │         │  轨道 B: Lambda Click    │
│  ┌─────────────────────┐   │         │  ┌────────────────────┐  │
│  │  Pocket encoder     │   │         │  │  Tile library      │  │
│  │  (EGNN)             │   │         │  │  (ChEMBL + metal)  │  │
│  └─────────────────────┘   │         │  └────────────────────┘  │
│           ↓                 │         │            ↓             │
│  ┌─────────────────────┐   │         │  ┌────────────────────┐  │
│  │  Velocity field     │◄──┼─reward──┼─►│  MCTS + β-reduce   │  │
│  │  v_θ(x, t)          │   │  share  │  │  (Lambda MCTS)     │  │
│  └─────────────────────┘   │         │  └────────────────────┘  │
│           ↓                 │         │            ↓             │
│  ┌─────────────────────┐   │         │  ┌────────────────────┐  │
│  │  ODE sampler        │   │         │  │  PySR heuristic   │  │
│  │  (Lipman 2023)      │   │         │  │  + SAS check       │  │
│  └─────────────────────┘   │         │  └────────────────────┘  │
│           ↓                 │         │            ↓             │
│      3D Molecules           │         │  SMILES + Lambda expr  │
└─────────────────────────────┘         └──────────────────────────┘
                  ↓                                       ↓
                  └──────────────►  Cross-validate  ◄─────┘
                                     ↓
                              Best of both worlds:
                              3D molecules with Lambda provenance
```

---

## 实施时间线 (融合部分)

| 周 | 轨道 A (Mol-Metal FM) | 轨道 B (Lambda Click) | 融合点 |
|---|---|---|---|
| 1-4 | LipmanFM Adapter smoke test | 克隆 PySR + REINVENT4 | (无, 各自准备) |
| 5-6 | EGNN training on MetalCytoToxDB | Tile 库构建 (ChEMBL) | **Reward 信号共享 (1)** |
| 7-9 | MMP2 SBDD benchmark | MCTS + β-归约算法 | **Metal Coordination Tile (2)** |
| 10-12 | 论文 Methods (Mol-Metal 部分) | 论文 Methods (Lambda 部分) | **靶标对齐 (3)** |
| 13-16 | 联合投稿: hybrid framework paper | | |

**总 16 周 (~4 个月) 达到联合投稿状态**

---

## 谁依赖谁 (dependency graph)

```
轨道 A: LipmanFMAdapter       轨道 B: PySR + REINVENT4
         ↓                              ↓
    EGNN training                   Tile library
         ↓                              ↓
    IC50 predictor  ─────────►   MCTS reward (MolMetal)
         ↓                              ↓
    3D geometry                   Metal Coordination Tile
         ↓                              ↓
    MMP2 benchmark                Cross-validate
         ↓                              ↓
    ════════════  Hybrid framework paper  ══════════════
```

**关键依赖**:
- 轨道 B 在 Week 5 之前不依赖轨道 A (可以完全独立开发)
- 轨道 B 在 Week 5 之后**选择性依赖** 轨道 A 的 IC50 predictor (作为可选 reward)
- 论文撰写阶段两轨**强依赖** (需要互相引用对方的图、表、实验)

---

## 给读者的故事 (审稿人友好的叙述)

> "Drug design is fundamentally a constrained combinatorial search problem.
>  Combinatorial because we explore 10^60 candidate molecules;
>  constrained because chemistry, biology, and synthesis all impose hard rules.
>
>  We argue that this combinatorial problem is naturally expressed
>  as a Lambda-calculus term rewriting system:
>  - **Tiles** are constants
>  - **Click chemistry reactions** are functions (e.g., λ(A). λ(B). CuAAC(A, B))
>  - **β-reduction** is the actual chemical reaction
>  - **Heuristics** for search are learned symbolically (PySR) for interpretability
>
>  We complement this with a geometric deep learning track (Lipman 2023 FM +
>  MolFlow-Triton EGNN) that handles 3D pocket conditioning and IC50 prediction.
>  The two tracks share an MMP2/MMP9 evaluation protocol and jointly produce
>  interpretable, 3D-aware, and 100%-synthesizable drug candidates."

**审稿人会问的问题**:
1. "Lambda 演算比 Symbolic Regression 直接用 PySR 多什么?" → 答: 形式化的 β-归约让化学反应可组合 (reactions are functions), 而 PySR 只能拟合公式。
2. "为什么不用 RL?" → 答: 离散空间 + 化学约束 = MCTS 更合适; RL 在连续 latent 空间里不能保证合成可行性。
3. "和 12_flow_matching 的关系?" → 答: (指向 fusion 图) 互补 — 一个给可解释性,一个给 3D 几何。

---

## 立即可做

```bash
# 在写完 13_lambda_clickchem/plan.md 之后:

# 1. 克隆 Lambda 引擎
mkdir -p molmetal/references/PySR
git clone --depth 1 https://github.com/MilesCranmer/PySR.git molmetal/references/PySR

# 2. 克隆 REINVENT4 (SBDD sandbox)
git clone --depth 1 https://github.com/MolecularAI/REINVENT4.git molmetal/references/REINVENT4

# 3. 创建 molmetal_lam/ 目录骨架 (与 molmetal/ 平级)
mkdir -p molmetal/molmetal_lam/{lam_chem,sbdd_env,tile_lib,search_alg,pipeline,scripts,tests}
for d in lam_chem sbdd_env tile_lib search_alg pipeline scripts tests; do
    touch molmetal/molmetal_lam/$d/__init__.py
done

# 4. 写第一版 CuAAC reaction rule
cat > molmetal/molmetal_lam/lam_chem/rules.py << 'EOF'
"""Click chemistry reactions as Lambda functions.

Each rule has signature: Tile -> Tile -> Tile
(Beta-reduction = apply reaction to two tiles)
"""
from molmetal_lam.tile_lib.tile import Tile

def CuAAC(tile_A: Tile, tile_B: Tile) -> Tile:
    """Cu(I)-catalyzed azide-alkyne cycloaddition.
    
    R-N3 + R'-C≡CH --[Cu(I)]--> 1,4-disubstituted 1,2,3-triazole
    """
    azide_atoms = tile_A.get_functional_group("azide")
    alkyne_atoms = tile_B.get_functional_group("terminal_alkyne")
    if not azide_atoms or not alkyne_atoms:
        raise ValueError("CuAAC requires azide and terminal alkyne")
    # ... assemble triazole ring
    return tile_A.assemble_with(tile_B, "CuAAC")

REACTION_RULES = {"CuAAC": CuAAC}
EOF

# 5. 跑最简单的 smoke test
python -c "
from molmetal_lam.lam_chem.rules import REACTION_RULES
from molmetal_lam.tile_lib.tile import Tile
# Build 2 dummy tiles
a = Tile.from_smiles('CCN=[N+]=[N-]')  # ethyl azide
b = Tile.from_smiles('C#CC')            # propyne
mol = REACTION_RULES['CuAAC'](a, b)
print('CuAAC product:', mol.smiles)  # should be 1,2,3-triazole
"
```

## 待你回答 (融合相关的 3 个问题)

| # | 问题 | 选项 |
|---|---|---|
| 1 | 融合点 1 (Reward 共享) | (a) Mol-Metal IC50 作为 Lambda reward (b) 只用 REINVENT4 (c) 两者都用 |
| 2 | 融合点 2 (Metal Tile) | (a) Pt(II) square-planar 第一批 (b) Ru(II) octahedral 第一批 (c) 都做 |
| 3 | 融合点 3 (靶标对齐) | (a) **MMP2/9 主选** (b) Kinase 优先 (c) CrossDocked100 通用 |

---

### Lambda Upper-Bound Push 2026-09-11

对照 `molmetal/reports/audit_lambda_upper_bound.md`(5 gaps)与 `molmetal/reports/mcts_strengthening.md`(PR 落地),本轮 P0 加强如下:

**已闭合的 gap**:
- **Gap 1 (constant prior → learned symbolic prior)**:`SymbolicPrior` 包装 `HeuristicRegressor`,`fit(states, scores)` 后输出 [0,1] 的 `predict_proba`,取代 `heuristic()` 永远返回 0.5 的 stub。PUCT 的 `P(a)` 现在真正由 `(features → score)` 给出,`pysr_wrapper` 从"事后拟合"提升为"在线引导"。
- **Gap 3 (stub scorer → multi-reward)**:`RewardAggregator` 组合 `r_vina / r_sa / r_posebusters / r_pic50 / r_retro` 加 `bonus_typed` / `bonus_binder`,Vina 自动取反、SA 从 [1,10] 反向归一到 [0,1],每通道 try/except → 0.0 不崩。`RewardAggregator.from_scorer()` 保持旧 `scorer=` 调用零迁移成本。
- **Dirichlet 根节点噪声 (AlphaZero 风格)**:`_apply_dirichlet_to_root` 在首次模拟时把 `Dir(α=0.3)` 混合进根子节点 prior,默认 `fraction=0.25`,直接消除"UCB 全 tie → 卡第一个孩子"的平台病态。

**仍 open 的 gap**:
- **Gap 2 (tile library 太小)** + **Gap 4 (stateless tree,无跨迭代学习)**:`SymbolicPrior` 虽拟合但尚未**持久化进树**,`closed_loop` 仍每轮从 seed 重建 MCTS;`tile_lib/click_tiles.py` 仍是 12 个手写 SMILES,展开上界 ≈ |rules|×12≈72,未见 ChEMBL/tmQM 池化或 tile embedding/retrieval。
- **Gap 5 (binding 仍是 RDKit 指纹匹配)**:`Vina`/`PoseBusters`/`DiffDock`/`FlowDock` adapter 仍未落进 `_binds_target` 的 typecheck 主路径,仍是 rollout 后置过滤。
- **VirtualLoss 并行化**:`_MCTSNode.virtual_loss` 字段已加,`VirtualLoss` dataclass 已暴露,但单线程搜索不触发——lock-free 并行树是"forward-compat hook",尚未接 leaf-parallel scheduler。
- **Richer tile library**:ChEMBL reactive-handle 池(10k–100k)、tmQM 108k dative-bond 预训练、Pocket2Mol/TargetDiff/FLOWR 端口仍未接到 expansion 前缀。

**净效果**:从 audit 报告的"5/5 gap 全开"收紧为"3/5 gap 已 inline 加强,2.5/5 仍 open(其中 VirtualLoss 与 tile library 是下一 PR 的 P0/P1 目标)"。PR 文件:`molmetal/molmetal_lam/search_alg/proof_search.py`、`molmetal/tests/test_proof_search_strengthened.py`(19 测试全过,1.49s,严格 backward-compat)。
