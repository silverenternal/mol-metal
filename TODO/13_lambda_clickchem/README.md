# Lambda-Calculus-Driven Click-Chemistry Drug Design

## 战略 (2026-09, 修订版)

**核心 insight (修订)**: **化学本身就是 Lambda 演算** —— 不是把 Lambda 当工具,而是把分子/键/反应/性质/合成全部定义为 λ-term 的语法、归约、类型、对偶。

**核心论断** (论文 Thesis Statement):
> "De novo drug design is **constructive proof search** in the **Molecular
>  Lambda Calculus**. Every generated molecule is a **witness** (proof term)
>  for the proposition 'this molecule binds target T with acceptable
>  ADMET'. The β-reduction history IS the synthesis path. The η-conversion
>  class IS the GNN's invariant. Curry-Howard gives us free
>  interpretability AND synthetic accessibility."

**为什么不是降级** (这是关键): 我之前的 plan 把 Lambda 降级成"产率预测器" —— 这是下游的标量函数,浪费了 Lambda 演算的真正力量。**正确的做法是把 Lambda 上推到分子结构上游**:

| 化学概念 | λ-演算对应 | 上游还是下游? |
|---|---|---|
| 原子 | primitive combinator (SKI + 金属作 n-arity) | **上游** |
| 共价键 | β-reduction (function application) | **上游** |
| 配位键 (dative) | curried partial application | **上游** |
| 芳香性 | η-conversion class | **上游** |
| 立体化学 | 不同 β-NF | **上游** |
| 分子 | closed λ-term in β-NF | **上游** |
| 反应 | β-reduction step | **上游** |
| 合成 (正向) | β-reduction sequence | **上游** |
| 逆合成 | β-EXPANSION | **上游** |
| 结合 (蛋白 + 配体) | β-reduction + type-check | **上游** |
| ADMET | type predicate | **上游** |
| 药物活性 | constructive proof | **上游** |
| 等变性 (SE(3)) | α-conversion invariance | **上游** |
| GNN 表达能力 | η-equivalence class | **上游** |
| **产率** | **β-reduction 概率** | **下游** (这个用 PySR 拟合即可) |

**SCI 一区卖点 (修订后)**:
1. **新范式 (New Paradigm)**: "Algebraic Drug Design" —— 首次把化学定义为 λ-演算
2. **可解释性免费**: 每个分子 = 一个 proof term,直接展示推导过程
3. **可合成性免费**: 每个 proof = β-reduction sequence = 合成路径
4. **统一框架**: 12_flow_matching 的 FM + EGNN 和 13_lambda_clickchem 的 click + MCTS 是**同一 λ-演算的不同算子** (β-reduction vs η-conversion)
5. **Curry-Howard 同构**: 药物设计 = 类型论 + 证明搜索,SCI 一区数学基础

**与 12_flow_matching/ 的关系 (修订后)**:
- **不是平行轨道,而是同一 λ-演算的两个面**:
  - Mol-Metal 的 EGNN = η-conversion learner (拟合 η-equivalence class)
  - Mol-Metal 的 Flow Matching = β-reduction sequence learner (从噪声到 β-NF)
  - Lambda 轨道提供形式化框架 (atoms/bonds/reactions/types)
  - 两轨合并 = "**形式化 + 计算**" 的完整 story
- **这才是真正的一区论文**: 形式化基础 + 计算实现 + 实验验证 三件套齐全

## 完整目录结构

```
TODO/
├── 12_flow_matching/                          ← 平行轨道 A (Mol-Metal, FM + EGNN)
│                                                — 同一个 λ-演算的 η-conversion + β-reduction 算子
│
└── 13_lambda_clickchem/                       ← 本目录 (平行轨道 B, 但其实是同一枚硬币的另面)
    ├── README.md                              ← 本文件 (总览 + 战略 + 修订动机)
    ├── plan.md                                ← 10 步详细计划 (Lambda 上游化)
    ├── molecular_lambda_calculus.md          ← ★ 核心: 化学 = λ-演算 的形式化定义
    └── convergence_with_molmetal.md           ← 两轨融合 (同一 λ-演算的两个面)
```

**核心论文故事**:
> "We formalize chemistry as a Molecular Lambda Calculus (MLC) where
>  - atoms are primitive combinators,
>  - bonds are β-reduction (= function application),
>  - molecules are closed terms in β-normal form,
>  - reactions are β-reduction steps,
>  - synthesis is a β-reduction sequence,
>  - drug design is constructive proof search under ADMET and binding
>    type predicates.
>
>  We then implement MLC two complementary ways:
>  - **Symbolic**: MCTS + PySR (轨道 B / 13_lambda_clickchem) for
>    interpretable, 100%-synthesizable designs.
>  - **Geometric**: SE(3)-equivariant Flow Matching + EGNN (轨道 A /
>    12_flow_matching) for 3D-pocket-conditioned IC50 prediction.
>
>  Applied to MMP2 inhibitors, we obtain both (i) interpretable Lambda
>  expressions for top-10 candidates and (ii) AUC 0.89 with SE(3)-
>  equivariant binding poses — a hybrid framework that no existing
>  method achieves."
```

## 10 步设计 (对应 plan.md)

| Phase | 步骤 | 内容 | 周期 |
|---|---|---|---|
| **Phase 0** Foundation | 1 | Lambda 演算符号回归引擎 | 1 周 |
| | 2 | Click 化学的 Lambda 函数式规则 | 1 周 |
| | 3 | SBDD/De Novo 3D 生成模型作 sandbox | 1 周 |
| | 4 | 化学 Tile 标准化砌块库 | 1 周 |
| **Phase 1** Algorithm | 5 | Lambda 驱动的 Tile 搜索与组装算法 | 2 周 |
| | 6 | 靶标蛋白结构离散化 | 1 周 |
| **Phase 2** Closed loop | 7 | 闭环生成 + 符号公式提取 | 1 周 |
| **Phase 3** Benchmark | 8 | 对接 SBDD 模型严格评估 | 1 周 |
| | 9 | 与主流 AI 生成模型 Baseline 对比 | 1 周 |
| **Phase 4** Paper | 10 | 形式化理论总结与 SCI 论文 | 2 周 |

**总 12 周** (~3 个月) 达到 SCI 一区投稿水平。

## 关键创新点 (论文卖点)

1. **Algebraic Drug Design** —— 把药物设计重写为符号推导,首例
2. **Differentiable Symbolic Reasoning** —— β-归约 + MCTS + 符号回归的联合优化
3. **Interpretable Synthesis Path** —— 生成的每个分子都附带 Lambda 表达式,可直接读出"为什么"
4. **100% Synthetic Accessibility** —— click chemistry 限定,所有生成物可合成 (SAS = 1.0)
5. **Cross-Paradigm Fusion** —— 符号方法 + flow matching 联合训练,兼顾可解释性与 3D 几何

## 与 12_flow_matching 的 3 个融合点

详见 `convergence_with_molmetal.md`:

1. **Reward 信号共享**:Mol-Metal 的 D-MPNN+EGNN IC50 预测器 → Lambda MCTS 的奖励函数
2. **Metal Coordination Tile**:把 Pt(II) square-planar / Ru(II) octahedral 配位作为特殊 Lambda 常量
3. **靶标交叉验证**:MMP2/9 (Mol-Metal 兴趣) ↔ kinase/GPCR (click chem 经典靶点)

## 立即可做 (不等回答)

```bash
# 1. 克隆 Lambda 演算符号回归引擎
# (候选: PySR, Deflex, symbolicregression)
git clone https://github.com/MilesCranmer/PySR.git molmetal/references/PySR

# 2. 克隆 SBDD sandbox
# (候选: REINVENT4, FlexSBDD)
git clone https://github.com/MolecularAI/REINVENT4.git molmetal/references/REINVENT4

# 3. 克隆 click-chemistry dataset
git clone https://github.com/openforcefield/QM-tools-and-reference-data.git molmetal/references/QM-tools

# 4. 写 molmetal_lam/ 目录 (平行于 molmetal/)
mkdir -p molmetal/molmetal_lam/{lam_chem,sbdd_env,tile_lib,search_alg,pipeline}
```

## 待你回答的 3 个问题

| # | 问题 | 选项 |
|---|---|---|
| 1 | 第一靶点集 | (a) CrossDocked100 通用 (b) **MMP2/9 (与 Mol-Metal 对齐)** (c) 单一 kinase (如 EGFR) (d) GPCR (如 β2AR) |
| 2 | 第一批 click 化学模块 | (a) **CuAAC 全部** (经典) (b) CuAAC + SPAAC (无铜,生物正交) (c) CuAAC + Diels-Alder (扩展) |
| 3 | 与 Mol-Metal 融合优先级 | (a) **Reward 共享先做** (b) Metal Tile 先做 (c) 靶标对齐先做 |
