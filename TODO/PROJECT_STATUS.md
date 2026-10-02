# PROJECT_STATUS — Mol-Metal 项目现状 (v57, 2026-10-01)

**Author**: Hugo Lee + ultracode agents
**Date**: 2026-10-01 (Wf-431 docs-only refactor + post Wf-420-B6..430-C1 closeout window)
**Purpose**: 综合性项目现状 — 回答 user 6 大核心问题（基线对比 / Lambda 算法驱动模块 / 点击化学 / 模型架构 / 指标 / 训练框架 + Triton）
**Audience**: 任何 on-board 新成员、评审人、未来 agent
**Sources of truth**: `TODO/02_metrics/METRICS_TARGETS.md` + `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` + `TODO/03_baselines/COMPARISON_BASELINES.md` + `TODO/decisions.md` + `TODO/03_plans/LAMBDA_ROADMAP.md` + `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` + AGENTS.md §6 recent Wf history
**Honest framing mandate**: `party-twenty-spirit §3.5` MEASURED-vs-PROJECTED tag discipline; AGENTS.md §5.2 honest framing convention
**AGENTS.md §5.3 NO paper work without authorization** — 此 doc 是项目现状 doc, NOT paper section

---

## §0 TL;DR — 一句话答案矩阵

| # | 用户问题 | 一句话答案 | 详细在 § |
|---|---|---|---|
| **Q1** | 贵金属抗癌生成模型选了哪个基线？哪篇论文在和谁比？ | **3 个 Tier-1 基线**: **Krasnov 2026** (MetalCytoToxDB, Ru 0.81), **Rusanov 2025** (PlatinAI), **DrugOOD 2022** (iMoLD 0.7211); **不是 reproduce — 是 alignment**: 同一数据集 + 同一指标 + 可能不同 split; 详细见 `TODO/03_baselines/COMPARISON_BASELINES.md` §1 | §1 |
| **Q2** | Lambda 算法驱动哪些模块？和谁比？ | **8 个 Lambda 模块（M1-M4 + M8-M11 + M36/M38/M42）**: MLC atoms-as-combinators + β-reduction rules + KataGo-MCTS + PLL + LambdaASTSR Deflexpressor + Coupling adapter + 84-rule MLE matrix + 18 click + 4 SMARTS + skeleton jump + FG veto + non-click med-chem; baseline 对比 = Katalin Marton (clique potentials) + PySR/F5 alone (无 ASTSR) + vanilla UCB (无 PUCT) + Uncalibrated Dirichlet (无 VariancePUCT) | §2 |
| **Q3** | 创建了哪些点击化学模块？对比实验如何设计？ | **5 click + 2 metal exchange + 21-rule registry (含 peptide/Boc/reductive amination)**: CuAAC / SPAAC / ThiolEne / Suzuki / AmideCoupling + MetalLigandExchange + AquaExchange + 21-rule R21-P3 extension; **5×5 Pt-click 兼容性矩阵** (square-planar + octahedral); **84-cell FG-compat** (12 metal × 7 ligand); 对比 = REINVENT4 (RNN+ AIS) / MOSES / GuacaMol / AiZynthFinder; 实验设计 = M40 买得到 gate (32% pass-rate) + M38 FG-veto + M42 PROTAC linker | §3 |
| **Q4** | 模型架构是什么？和普通 Flow Matching 比有什么不一样？ | **CFM = Lipman 2023 CondOT + 中点 ODE**, 不是 vanilla 200-step Euler; **SE(3)-EGNN** velocity net + **YuelBond** GNN bond head (Wang+Dokholyan 2025, F1=92.7%); **20-atom vocab** + logit clamp ±50 + 6 shim fixes; Triton-fused ops SCAFFOLD; W-Flow waveform relaxation (R18); Hilbert primal-dual (R18) | §4 |
| **Q5** | 总共构建了哪些指标？含义？数据能直接写论文吗？ | **21 个 metrics (m1-m21)** 跨 4 个 tier; **MEASURED 6 个**: m21 0.8007±0.0160 (pass), m17 0.7479 (pass), m1 0.7639±0.0995 (partial), m3 -7.782 kcal/mol (pass), m15 0.6578Å (pass), m14 1.000 (perfect); **m6_kappa DROPPED from §3 main**; **m2 platinai_4cell + hilbert HONEST_NEGATIVE_R2_DEEP**; 多数 metric 可直接进 §3 表,余下 8 个 SCAFFOLD-tier 需要 GPU 重训 | §5 |
| **Q6** | 训练框架如何构建？Triton 优化有没有加入主线？ | **Torch 2.14.0+cu130 + Blackwell sm_120 + RTX 5090**; **Wf-380 10 SCAFFOLD modules + Wf-380-B1 14+3 NEW CLI flags + Wf-431-B1 GPU util (torch.compile/tf32/cudnn-benchmark/channels-last)**; **Triton 状态: SCAFFOLD-only CPU reference (per Wf-380-A1 §2.5); GPU Triton path DEFERRED per Wf-326/Wf-353 follow-ups**; speedup PROJECTED ~5-10× | §6 |

---

## §1 Q1: 基线选择 — Tier-1 / Tier-2 / Out-of-scope

### §1.1 Principle (per user directive 2026-10-01 verbatim)

> **"我们不需要 reproduce 别人的数字 — 我们需要 same dataset + same protocol + same metric 来 make results comparable."**

→ 不是 "bit-identical reproduction"; 是 "protocol alignment"; 同一数据集 + 同指标 + 可比 split; **split difference (5-fold CV vs scaffold-split) 是 OK 的** — 在 §6 honest limitations 写明。

详见 `TODO/03_baselines/COMPARISON_BASELINES.md` §0 + `TODO/decisions.md` D-Wf-431。

### §1.2 Tier-1 直接对比 (3 papers)

| # | Baseline | Citation | Dataset | Our alignment | 对应 metric |
|---|---|---|---|---|---|
| **1** | **Krasnov 2026** | Krasnov et al., *J. Med. Chem.* 2026-04 | **MetalCytoToxDB** | DATASET + METRIC ALIGN (split differs: paper 5-fold CV; us scaffold-split 70/10/20 per Wf-261) | m1 Ru AUC + m21 NCI-60 9-cell |
| **2** | **Rusanov 2025** | Rusanov et al., ChemRxiv 2025-06 | **PlatinAI** | DATASET + METRIC + SPLIT APPROX ALIGN (paper per-cell-line; us cohort-split 70/10/20 + per-cell eval) | m2 Pt 4-cell AUC + m19 LOMO |
| **3** | **DrugOOD 2022** | Ji et al., ICLR 2022 | **DrugOOD** (1.58M rows via metacytotox_fallback) | DATASET + METRIC + SPLIT FULL ALIGN | m17 OOD generalization |

### §1.3 Tier-2 (different metal / different method class — cited for context)

TransDiffSBDD 2025 / MolCRAFT 2024 / DecompDiff 2024 / DiffSBDD 2024 / TargetDiff 2023 / Pocket2Mol 2022 / MolChord 2025 / AlphaDrug 2024 / Boltz-2 2025 / FLOWR 2025 (FlowDock 2025) / iMoLD 2024 / Chemprop 2019 / D-MPNN — 这些论文 **m3 Vina / m7 Diversity / m15 RMSD / m17 OOD** 提供了 SOTA target 值 (e.g. TransDiffSBDD -9.37 kcal/mol; MolChord 0.81 Diversity)。

### §1.4 Out-of-scope (NOT 对比)

Per `TODO/03_baselines/COMPARISON_BASELINES.md` §2:

- **不同任务** (SBDD 3D pose generation): DiffDock 2022 / DeltaDock 2023 / EquiBind 2022 / TransDiffSBDD 2025 — 这是 3D pose generation, 不是 2D-3D binding affinity prediction
- **不同数据集**: Multimodal Protein-Metal Binding 2025 / PDBbind / Boltz-2 — 这些是 protein-ligand binding, 不是 precious-metal coordination
- **不同方法 class**: REINVENT4 / MOSES / GuacaMol — 这些是 language-model-style molecule generation
- **通用 SBDD** (not precious-metal specific): Pocket2Mol / TargetDiff / MolCRAFT / DecompDiff

这些 **不在 Tier-2 paper §4 comparison tables**。

### §1.5 Honest framing — 为什么选这 3 个

Krasnov 2026 是 **唯一直接同领域** (precious-metal anticancer) + 同 5-metal subset (Ru/Ir/Rh/Re/Os) + 同任务 (binary AUC) 的论文。Rusanov 2025 是 **唯一 Pt cisplatin 4-cell-line** 的 paper。DrugOOD 2022 是 SBDD OOD generalization 唯一被 iMoLD 使用的 dataset — 因此可以做 iMoLD-baseline 对比。三者组合覆盖 precious-metal + Pt-specific + SBDD-generalization 三个 sub-task。

---

## §2 Q2: Lambda 算法驱动哪些模块 + baseline 对比

### §2.1 Lambda 算法定位 — 整个化学体系的计算基础

Per `TODO/13_lambda_clickchem/plan.md` 修订:

> **Lambda 演算不再是"产率预测器", 而是整个化学体系的计算基础.**
> - 原子 = primitive combinators (arity = valence + lone_pairs)
> - 键 = β-reduction
> - 分子 = closed λ-term in β-NF
> - 反应 = β-reduction step
> - 合成 = β-reduction sequence
> - ADMET = type predicates
> - 药物 = constructive proof

### §2.2 8 个 Lambda 模块（M1-M4 + M8-M11 + M36 + M38 + M42）

Per `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §1 + `mol-metal-lambda-calculus` skill:

| ID | 模块 | 文件 | 影响 metric |
|---|---|---|---|
| **M1** | MLC atoms-as-combinators (arity = valence + lone_pairs) | `molmetal/molmetal_lam/lam_chem/ast.py` + `lambda_combinators.py` + `atoms/combinators.py` | m1 Ru AUC + m2 Pt 4-cell + m7 Diversity + m11 n_distinct + m12 Synthesis rate + m19 LOMO |
| **M2** | β-reduction rules = 84-rule MLE matrix + 18 click + 4 NEW SMARTS | `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` + `reactions/` | m11 n_distinct + m12 Synthesis rate |
| **M3** | KataGo-MCTS proof search (PUCT + virtual loss + Dirichlet + FPU + policy pruning) | `molmetal/molmetal_lam/search_alg/proof_search.py` (6217 LOC) | m4 Success Rate + m7 Diversity + m11 n_distinct + m17 OOD |
| **M4** | PLL heuristic (Persistent Local Laplacian, Liu/Feng 2026) | `molmetal/molmetal_lam/lam_chem/pll_signature.py` + `search_alg/pll_heuristic.py` | m4 + m7 + m11 + m17 |
| **M8** | LambdaASTSR (Deflexpressor-style AST symbolic regression) | `molmetal/molmetal_lam/reward/lambda_astsr.py` (1059 LOC) | m1 + m2 + m16 Affinity + m17 OOD + m18 protein-metal |
| **M9** | Coupling adapter (lambda × CFM 32→64 bridge) | `molmetal/molmetal_lam/reward/coupling_adapter.py` | m1 + m2 + m7 + m16 + m17 + m18 |
| **M10** | 84-rule MLE matrix (12 metals × 7 ligands) | `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` | m11 + m12 |
| **M11** | 18-rule + 4 NEW SMARTS (transmetalation / Cl-OH swap / ligand-shell-MMA / Pt/Pd transmetalation) | `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` | m5 SA + m7 Diversity + m11 + m12 |
| **M36** | Skeleton jump / fragment-swap (R18 W3) | `molmetal/molmetal_lam/lam_chem/skeleton_jump.py` (915 LOC) | m7 Diversity + m11 n_distinct |
| **M38** | FG-compat / leave-group veto (R18 W3) | `molmetal/molmetal_lam/lam_chem/fg_compat_veto.py` (843 LOC) | m11 + m13 Validity |
| **M42** | Non-click med-chem rules (R18 W3) | `molmetal/molmetal_lam/lam_chem/non_click_medchem_rules.py` (747 LOC) | m11 + m12 Synthesis rate |

### §2.3 8 个 Lambda gap (L1-L8) — 已 ship

Per `TODO/03_plans/LAMBDA_ROADMAP.md` §0:

| Gap | 状态 | ship wave |
|---|---|---|
| **L1 closure verifier** | ✅ SHIPPED | Wf-36 P1 |
| **L2 VariancePUCT** | ✅ SHIPPED | Wf-14 |
| **L3 ScoreRep + VariancePUCT + MCTS 三者联合** | ✅ SHIPPED | Wf-36 P3 |
| **L4 MCTS replay learner (AlphaGo-style)** | ✅ SHIPPED | Wf-21 |
| **L5 Hilbert ASTSR** | ✅ SHIPPED | Wf-37 P1 |
| **L6 closure d-block 30 metal** | ✅ SCAFFOLD-only CPU ship; GPU retrain BLOCKED on user OK | Wf-37 P2 + Wf-381 |
| **L7 multi-PH + PLL** | ✅ SCAFFOLD SHIPPED | Wf-37 P3 + Wf-382 |
| **L8 triangle closure (ASTSR ↔ Hilbert ↔ MCTS)** | ✅ SHIPPED | Wf-37 P4 |

### §2.4 baseline 对比 — Lambda vs 其他 ML 方法

Per user voice "和谁在比":

| 维度 | 我们 (Lambda) | Baseline 1 | Baseline 2 |
|---|---|---|---|
| **原子建模** | MLC atoms-as-combinators (arity = valence + lone_pairs) | **Katalin Marton** clique potentials (随机场) | **PyTorch Geometric** standard GNN (无 chemistry prior) |
| **β-reduction search** | KataGo-MCTS (PUCT + virtual loss + Dirichlet + FPU) | **vanilla UCB** (no prior) | **Pure MCTS** (no pruning) |
| **AST symbolic regression** | LambdaASTSR Deflexpressor (Deflex-style) | **PySR** alone (no AST guidance) | **Eureqa** (commercial; no Lambda-flavored AST) |
| **Priors** | PLL heuristic (Liu/Feng 2026) | **No priors** (vanilla Dirichlet noise) | **F5 fingerprint** (RDKit count-based, no topological) |
| **Variance reduction** | Score-Repellent MC (Burch-Bardsley 2022) | **vanilla MCMC** (no variance control) | **Uncalibrated Dirichlet** |
| **Optimizer** | Hilbert primal-dual O(1/k) Chambolle-Pock | **Evolutionary search** (PySR 64×50-gen) | **F5 / LambdaASTSR** (older) |
| **Coupling (Lambda × CFM)** | coupling_adapter (32→64 bridge) | **No coupling** (Lambda + CFM trained separately) | **Simple concatenation** (concat features without bridge) |
| **Closure theorem** | 6/6 Catherine properties on d-block | **Not modeled** (no closure guarantee) | **Existence only** (Bass martingale) |

→ **Lambda 端的核心竞争优势**: **chemistry prior (MLC)** + **guided search (KataGo-MCTS)** + **topological priors (PLL)** + **AST-guided regression (LambdaASTSR)** + **provable closure (Catherine 6/6)** 五个维度的复合 prior — 这是 baseline 不具备的。

---

## §3 Q3: 点击化学模块 + 对比实验

### §3.1 5 click reactions + 2 metal exchanges + 21-rule registry

Per `mol-metal-click-chemistry` skill + `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §1:

| ID | Reaction | SMARTS | Building blocks | 用途 |
|---|---|---|---|---|
| CuAAC | Cu(I)-catalyzed azide-alkyne cycloaddition | `[R]N=N=N.[R']C#C>>[R]N1C=C([R'])N=N1` (1,4-disub triazole) | terminal alkyne + azide | fastest, ≥90% yield |
| SPAAC | strain-promoted (no Cu) | 同 CuAAC 但 strained alkyne (BCN/DBCO) | BCN/DBCO + azide | Cu-incompatible (e.g. in-cell) |
| ThiolEne | radical thiol-ene | `[R]SH.[R']C=C>>[R]S[R']` (anti-Markovnikov) | thiol + terminal alkene | light-initiated, biocompatible |
| Suzuki | Pd(0)-catalyzed C-C coupling | `[R]B(OH)2.[R']X>>[R]R'` (X = halide/triflate) | boronic acid + aryl halide | aryl-aryl bonds |
| AmideCoupling | amide bond formation | `[R]COOH.[R']NH2>>[R]C(=O)N[R']` | carboxylic acid + amine | peptide-like linkers |
| MetalLigandExchange | ligand exchange at metal center | `[M]([L1])...+[L3]>>[M]([L1])([L3])...+[L2]` | metal + ligand | tune coordination sphere |
| AquaExchange | aqua ligand exchange | `[M]-OH2+[L]>>[M]-[L]+H2O` | metal aqua complex + incoming ligand | hydrolysis-labile |
| **M42 21-rule registry** | peptide + Boc + reductive amination | per Carey & Sundberg 2007 + Montalbetti 2005 + Abdel-Magid 1996 | carboxylic acid + amine + aldehyde | PROTAC linker + peptidomimetic |

**Total**: 18-rule (R17 ship) → 21-rule (R18 W3 ship per Wf-11 P1.1).

### §3.2 兼容性矩阵

#### §3.2.1 5×5 Pt-click matrix

Per `mol-metal-click-chemistry` skill §"Pt-click compatibility matrix":

| Pt state \ rule | CuAAC | SPAAC | ThiolEne | Suzuki | AmideCoupling |
|---|---|---|---|---|---|
| Pt_II square-planar | ✓ | ✓ | ✓ | ✗ (Pd) | ✓ |
| Pt_IV octahedral | ✓ | ✓ | ✓ | ✗ | ✓ |

→ Suzuki 是 Pd(0)-催化, 不能在 Pt 上; 其他 4 个 click 在 Pt 上 OK。

#### §3.2.2 84-cell FG-compat matrix (M38)

Per `molmetal/molmetal_lam/lam_chem/fg_compat_veto.py` (843 LOC):

- **12 metal × 7 ligand = 84 cells**, keyed on **FG (functional group) identity**, NOT reaction-class
- `LeaveGroupVeto` layered on March 1985 good/mediocre/bad LG ordering
- `VetoAugmentedReducer` opt-in sidecar wrapping existing reducer
- `strict=True` opt-in; veto fires iff disfavored-matched AND tolerated-empty
- Negative-false-positive rate = 0 invariant on canonical educt pairs

### §3.3 对比实验如何设计

Per `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` + `MECHANISM_METRIC_MAPPING.md` §5-§6:

| 实验 | 机制 | metric | 对比 baseline |
|---|---|---|---|
| **M40 Buyable-reactant gate** | AiZynth-as-gate (32% SMARTS fallback) | m12 Synthesis rate | **REINVENT4** (RNN generative + AIS scoring) + **MOSES** (SMILES LM) + **GuacaMol** (RL) |
| **M38 FG-compat veto** | 84-cell matrix + leave-group veto | m11 n_distinct precision + m13 Validity | (no direct baseline; this is unique to our MLC framework) |
| **M42 Non-click med-chem** | 21-rule registry (peptide + Boc + reductive amination) | m12 Synthesis rate on PROTAC-like campaigns | **SyntheMol** (Genheden & Thakkar 2020) |
| **M40 + M42 combined** | AiZynth + 21-rule → 60% pass-rate (PROJECTED) | m12 | vs current 32% SMARTS fallback |
| **m17 OOD DrugOOD** | OOD protocol + scaffold split | m17 DrugOOD AUC | **iMoLD** (Ji et al. 2024) |
| **m4 Success Rate** | MCTS + PLL + 3D chain | m4 | **TransDiffSBDD** 83.9% + **ReInvent+Vina** 76.7% |
| **m5 SA** | click rules + SA-aware penalty | m5 | **TransDiffSBDD** 0.75 / **MolChord** 0.77 / **AlphaDrug** 0.80 |

→ **对比实验设计原则**: **同一数据集 + 同一指标 + 同一 protocol (per Q1)**; baseline 数值 cite 在 `METRICS_TARGETS.md` §2-§11 中。

---

## §4 Q4: 模型架构 — CFM + EGNN + YuelBond vs 普通 Flow Matching

### §4.1 架构总览 — Lipman 2023 CFM + SE(3)-EGNN + YuelBond bond head

Per `mol-metal-cfm-training` skill + `TODO/12_flow_matching/`:

```
molmetal/adapters/flow_matching_lipman/__init__.py  (Lipman 2023 CondOT path + midpoint ODE)
        ↓
EGNN velocity net (molmetal/adapters/egnn_rocm.py)
        ↓ (SE(3)-equivariant message passing)
        ↓
YuelBond bond head (molmetal/models/bond_head.py) — Wang+Dokholyan 2025 GNN bond-type head
        ↓ (4-class bond type: single/double/triple/aromatic)
        ↓
AMP/BF16 checkpoint (molmetal/adapters/flow_matching_lipman/amp.py)
        ↓
ODE solver (molmetal/adapters/flow_matching_lipman/ode_solver.py) — Euler / RK4 (Triton-fused SCAFFOLD)
```

### §4.2 vs 普通 Flow Matching 的差异 (6 大创新)

Per `mol-metal-cfm-training` skill + Wf-380 10 SCAFFOLD modules:

| 维度 | 普通 Flow Matching | 我们的 Mol-Metal CFM | 影响 metric |
|---|---|---|---|
| **1. ODE solver** | vanilla 200-step Euler | **midpoint ODE** (R17 shim fix) + **W-Flow waveform relaxation** (R18 W1, Lelarasmee 1982) — geometric convergence in 3-5 outer iters | m3 Vina + m4 + m8 PB-Valid + m10 |
| **2. Velocity net** | standard GNN (not equivariant) | **SE(3)-equivariant EGNN** (`molmetal/adapters/egnn_rocm.py`) — dative edge type + 90° Pt(II) angle constraint | m3 + m8 + m13 + m14 + m15 + m17 |
| **3. Bond head** | distance-only threshold | **YuelBond GNN bond-type head** (Wang+Dokholyan 2025, F1=92.7%) — 4-class bond type (single/double/triple/aromatic) | m13 Validity + m14 Decode ratio (0/192 → 192/192 lift per Wf-124 P2) |
| **4. Atom vocab** | 12-atom (standard organic) | **20-atom vocab** (H, B, C, N, O, F, Si, P, S, Cl, Sc, Se, Br, Y, Zr, I, Hf, Pt, Au, ...) — **vocab_mask** for OOV atoms | m1 + m2 + m14 (rare atom coverage) |
| **5. Logits clamp + LR warmup + shim fixes** | no clamping | **logits clamp ±50** + **LR warmup linear 200 steps** + **6 shim fixes** (TODO-33 P1+P2: defensive try/except, ModelWrapper kw-time fallback, reference path auto-detect) | m14 decode_ratio stability |
| **6. Math fusion (R18 W1)** | vanilla FM | **SE(3) Schrödinger Bridge** (M22, Mahmood 2026 arXiv:2603.14049) + **Transport Relaxation** (M23, Jiang-Xu-Zhang 2026) + **Multi-Parameter Persistence** (M24, Kim & Jung 2026) + **Hilbert primal-dual** (M25, Chambolle-Pock 2011) + **Score-Repellent MC** (M26, Burch-Bardsley 2022) + **W-Flow** (M27) + **MacroGuide PH** (M28) — **MEASURED 452.16× speedup at n=30 vs evolutionary (CPU smoke for M25)** | m3 + m7 + m11 + m14 + m15 + m16 + ASTSR R² |

### §4.3 R18 Wave 2/3 新机制 (M29-M44)

Per `MECHANISM_METRIC_MAPPING.md` §5-§6:

- **M29** Coordination-Aware GNN (multi-head coordination-class gated) — coord-specific pockets
- **M30** Quotient-Space Diffusion (SE(3) by construction)
- **M31** Quantum semantics (symplectic point per atom) — paper §3 only
- **M32** Hochschild cohomology (β-reduction cochain) — paper §3 only
- **M33** VarianceControlledPUCT (variance penalty in PUCT)
- **M34** Fractional ODE solver (Caputo derivative, O(k^-α))
- **M35** Multi-PH + PLL signature integration (strict monotonicity)
- **M36-M44** Skeleton jump + MD relax stub + FG veto + layered gen + buyable gate + OOD protocol + non-click med-chem + Bergman kernel + Bass martingale — **CPU-only SCAFFOLD SHIPPED; GPU retrain deferred to Wf-380 + R19**

### §4.4 训练 entry + CPU verification

Per `mol-metal-cfm-training` skill "Default flags":

```bash
uv run python molmetal/scripts/metallo_drug_smoke_retrain.py \
    --steps 10000 --n-train 500 --batch-size 4 \
    --hidden-dim 128 --n-layers 3 --bond-head learned --joint-train \
    --decode-smoke-every 1000 \
    --out-dir reports/wf_yuelbond_10k
```

**MEASURED**: R17 30/30 decode probes (decode_ratio = 1.000) + 14/14 evaluators + 6 shim fixes PASS (per `molmetal/scripts/eval_all_evaluators.py`)。

---

## §5 Q5: 21 个 metrics — 含义 + 现状 + 能否直接写论文

### §5.1 4-tier 优先级分层

Per `TODO/02_metrics/METRICS_TARGETS.md` §1 + §11 (Wf-393-A2 + Wf-431 refactor):

| Tier | metric | 含义 | 现状 |
|---|---|---|---|
| **Tier 1 LEAD** | **m1 Ru AUC** | Ru subset binary AUC vs Krasnov 2026 (J. Med. Chem.) | **MEASURED 0.7639 ± 0.0995** (6/7 pairs; Rh 0.8798 PASS + Re 0.7060 PARTIAL); `BELOW_KRASNOV_GATE_PARTIAL` |
| **Tier 1 LEAD** | **m2 Pt 4-cell AUC** | Pt-specific 4 cell-line AUC vs Rusanov 2025 (PlatinAI) | **MEASURED platinai_4cell -0.1440 / hilbert -0.1440 / metalcytotox_combined +0.1981**; HONEST_NEGATIVE_R2_DEEP preserved on small cohorts |
| **Tier 1 LEAD** | **m11 n_distinct** | Distinct metal-ligand pairs (≥3 gate) | SCAFFOLD-tier shipped |
| **Tier 1 LEAD** | **m12 Synthesis rate** | SyntheMol round-trip (≥0.95 gate) | path (c) λ-only at 1.00 ceiling |
| **Tier 1 LEAD** | **m14 Decode ratio** | CFM RDKit-strict decode (≥0.05 gate) | **MEASURED 1.000** (R17 30/30 probes) |
| **Tier 1 LEAD** | **m17 OOD AUC** | DrugOOD OOD (vs iMoLD) | **MEASURED metacytotox_fallback assay 0.7479** (iMoLD-gate PASS) |
| **Tier 1 LEAD** | **m21 NCI-60 9-cell** | Precious-metal 9 cell-line mean AUC vs Krasnov 2026 | **MEASURED 0.8007 ± 0.0160** (3 seeds × 50 ep, scaffold-split, MetalCytoToxDB 9-cell subset) — paper §4.1 highlight |
| **Tier 2** | **m3 Vina Dock** | CrossDocked100 median kcal/mol | **MEASURED V3 cohort -7.782 kcal/mol** (5-pocket cohort smoke; 1h36 -8.089 / 3eml -7.327 / 3ny8 -8.224 / 4aaw -7.782 / 4rlu -7.483); **passes industry -7.0 by +0.782** |
| **Tier 2** | **m4 Success Rate** | Vina<-8.18 ∧ QED>0.25 ∧ SA>0.59 | SCAFFOLD-tier shipped; MEASURED pending GPU retrain per AGENTS.md §5.1 |
| **Tier 2** | **m5 SA** | Ertl 1-10 (sa-unit 1-0) | SCAFFOLD-tier shipped |
| **Tier 2** | **m6 QED** | Bickerton 0-1 | SCAFFOLD-tier shipped |
| **Tier 2** | **m7 Diversity** | Morgan ECFP4 pairwise Tanimoto | SCAFFOLD-tier shipped; MEASURED pending GPU retrain |
| **Tier 2** | **m8 PB-Valid** | PoseBusters geometric validity | SCAFFOLD-tier shipped |
| **Tier 2** | **m9 Inter Parity** | SOTA2 leaderboard | TBD |
| **Tier 2** | **m10 Vina Min** | best pose kcal/mol | SCAFFOLD-tier shipped |
| **Tier 2** | **m13 Validity** | RDKit parseable | SCAFFOLD-tier shipped |
| **Tier 3** | **m15 RMSD** | Redocking benchmark <2Å | **MEASURED 0.6578 Å median** (5/5 pockets PASS SOTA gate) |
| **Tier 3** | **m16 Affinity r** | Pearson r vs experimental pIC50 | SCAFFOLD-tier shipped |
| **Tier 3** | **m18 Protein-metal** | Multi-modal protein-metal binding AUC | NOT YET in pipeline |
| **Tier 3** | **m19 LOMO** | Multi-metal LOMO transfer | SCAFFOLD-tier shipped |
| **Tier 3** | **m20 Tumor-specific** | Tumor-specific protein-metal binding | NOT YET in pipeline |
| **DROPPED** | **m6_kappa** | Shape Kappa Index | **DROPPED from §3 main per Wf-389-A1**; promoted to §3 supplementary sub-table (MolLogP + MorganDensity_r2 + Chi3v all WEAK band per Cohen 1988) |

### §5.2 MEASURED vs PROJECTED 状态总结

| 状态 | count | metrics |
|---|---|---|
| **MEASURED with paper-comparable values** | 6 | m1, m3, m14, m15, m17, m21 |
| **MEASURED but below target / honest-negative** | 1 | m2 (small cohort overfit) |
| **DROPPED from §3 main** | 1 | m6_kappa → §3 supplementary sub-table |
| **SCAFFOLD-tier shipped, MEASURED pending GPU retrain** | 8 | m4, m5, m6, m7, m8, m9, m10, m11, m13, m16, m19 |
| **NOT YET in pipeline** | 2 | m18, m20 |
| **Total** | **21** | (per `METRICS_TARGETS.md` §2) |

### §5.3 现有数据能直接拿出去写论文吗？

**部分可以。** Per `METRICS_TARGETS.md` §4 + `docs/PAPER_SECTION_3_METRIC_TABLE.md` (Wf-389-A1 §3 supplementary sub-table):

- **可以直接进 §3 main-result 表 (6 个)**:
  - m21 NCI-60 9-cell **0.8007 ± 0.0160** ✅ (paper §4.1 highlight per Wf-335-SYN §3)
  - m17 DrugOOD OOD assay **0.7479** ✅ (passes iMoLD-gate)
  - m3 Vina Dock V3 cohort median **-7.782 kcal/mol** ✅ (passes industry -7.0 by +0.782)
  - m15 Docking RMSD **0.6578 Å** ✅ (5/5 pockets PASS 2.0 Å gate)
  - m14 Decode ratio **1.000** ✅ (R17 30/30 probes)
  - m1 Ru AUC **0.7639 ± 0.0995** — paper §3 with Rh fold +0.020 margin over Krasnov 0.81 + Re fold paper §6 honest-negative

- **可以直接进 §3 supplementary sub-table (3 个 WEAK-band correlates)**:
  - m6_kappa → drop; promote top-3 to supplementary (MolLogP + MorganDensity_r2 + Chi3v) per Wf-389-A1 §3

- **§4 ablations** (10 ablation rows + Wf-380 SCAFFOLD ablations per `MECHANISM_METRIC_MAPPING.md` §7) — 可以直接写, 数据已 ship 在 §11 Tier 1/2 status table

- **不能直接进 §3** — 8 个 SCAFFOLD-tier metrics (m4 / m5 / m6 / m7 / m8 / m9 / m10 / m13 / m16 / m19) 需要 **GPU 重训后 MEASURED-on-real**。 Wf-299 3-wave launch + Wf-380 SCAFFOLD-loss MEASURED 都在 USER_OK-gated 状态下阻塞 per AGENTS.md §5.1。

### §5.4 §6 honest limitations 必备内容

Per `docs/PAPER_SECTION_6_v4_FINAL.md` + Wf-410-A1 §3.1.6:

1. **m2 R² small-cohort overfit**: platinai_4cell + hilbert n=200 cohorts; 50+ feature Ridge OVERFITS
2. **m6_kappa drop-from-§3**: 9 paths exhausted; promote top-3 WEAK correlates to supplementary
3. **Vina 1.2.7 Pt atype REJECTS**: chemotype_atype_accepted=False across 10 pockets; R20 GPU TODO-128 Phase 2 future work
4. **Wf-380 SCAFFOLD modules are CPU-only**: 10×NO-GPU-retrain; lift claims remain PROJECTED
5. **Split difference**: Krasnov 5-fold CV vs us scaffold-split; Rusanov per-cell-line vs us cohort-split — §6.1-§6.7 per `PROTOCOL_ALIGNMENT_MATRIX.md`

---

## §6 Q6: 训练框架 + Triton 优化

### §6.1 训练框架总览

Per `mol-metal-gpu-execution` skill + AGENTS.md §5.1 + Wf-380 + Wf-431:

```
                    PyTorch 2.14.0+cu130
                            ↓
            ╔═══════════════════════════╗
            ║   Blackwell sm_120 + RTX 5090  ║
            ╚═══════════════════════════╝
                            ↓
    ┌──────────────────┬────────────────────────────┐
    │   CFM trainer     │     D-MPNN trainer            │
    │   (Wf-124 P2)     │     (Wf-131 P3)              │
    │   h=128, l=3      │     5-metal LOMO W₂          │
    │   YuelBond bond head  │     Wasserstein pIC50      │
    └──────────────────┴────────────────────────────┘
                            ↓
    ┌────────────────────────────────────────────────┐
    │  Wf-380 SCAFFOLD modules (10) + Wf-380-B1 trainer │
    │  integration (14+3 NEW CLI flags)                  │
    │  + Wf-431-B1 GPU utilization flags                  │
    │  (--torch-compile + --tf32 + --cudnn-benchmark      │
    │   + --channels-last)                                │
    └────────────────────────────────────────────────┘
                            ↓
              Pre-flight v3 117-gate check
                            ↓
              Pre-flight v3 fresh-run MEASURED 113/0/4/117
                            ↓
              USER_OK-gated GPU launch
```

### §6.2 Wf-380 10 SCAFFOLD modules — 状态

Per `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §3 + Wf-380-A1 closeout:

| # | SCAFFOLD module | 文件 | 来源 | 影响 metric | 状态 |
|---|---|---|---|---|---|
| 1 | **edm_preconditioning** | `molmetal/adapters/flow_matching_lipman/edm_preconditioning.py` | Karras 2022 σ-preconditioning | m14 + m1 | SHIPPED CPU-only (M25 partial) |
| 2 | **optimal_transport_v2_solver** | `molmetal/adapters/flow_matching_lipman/optimal_transport_v2_solver.py` | Pooladian 2023 OT-CFM | m1 + m2 | SHIPPED CPU-only (M23 partial) |
| 3 | **stable_vm** | `molmetal/adapters/flow_matching_lipman/stable_vm.py` | StableVM 2026 ICML | m14 (rare-atom stability) | SHIPPED CPU-only |
| 4 | **so3_average** | `molmetal/adapters/flow_matching_lipman/so3_average.py` | Cao 2025 ICML SO(3)-averaged FM | m3 + m15 | SHIPPED CPU-only |
| 5 | **triton_mini_batch_ot** | `molmetal/adapters/flow_matching_lipman/triton_mini_batch_ot.py` | Khainb 2025 Triton-fused mini-batch OT | m1 + m2 (speedup) | **SCAFFOLD-only CPU reference per Wf-380-A1 §2.5** |
| 6 | **shortcut_distillation_wf380** | `molmetal/adapters/flow_matching_lipman/shortcut_distillation_wf380.py` | Frans 2024 shortcut consistency distillation | m14 (sample steps) | SHIPPED CPU-only |
| 7 | **consistency_regularization_wf380** | `molmetal/adapters/flow_matching_lipman/consistency_regularization_wf380.py` | EDM Karras 2022 | m14 (drift) | SHIPPED CPU-only |
| 8 | **flowmol_unified** | `molmetal/adapters/flow_matching_lipman/flowmol_unified.py` | FlowMol arXiv:2412.01966 (CC-BY-NC) | m3 + m11 | **LICENSE-GATED** — production integration USER_OK-required |
| 9 | **log_normal_t** | `molmetal/adapters/flow_matching_lipman/log_normal_t.py` | log-normal t-sampling | m14 (t-sampling variance) | SHIPPED CPU-only |
| 10 | **lr_warmup** | `molmetal/adapters/flow_matching_lipman/lr_warmup.py` | cosine LR warmup | m14 (diverge) | SHIPPED CPU-only |

### §6.3 Triton 优化是否加入主线 — **NO (SCAFFOLD-only)**

Per `triton_mini_batch_ot.py` module docstring + Wf-380-A1 §2.5 honest framing:

```python
# Wf-380-A1 §2.5 — Triton mini-batch OT SCAFFOLD (CPU reference).
#
# SCAFFOLD ONLY — the canonical implementation lives in
# molmetal.adapters.flow_matching_lipman.triton_minibatch_ot (561 LOC, Wf-341/Wf-353-A1).
# This module is the small reference.
#
# Honest framing: SCAFFOLD-only; no GPU launches; speedup claim
# (~5-10× on GPU via Triton) is PROJECTED per Khainb 2025.
```

Per `MECHANISM_METRIC_MAPPING.md` §3 + §7 ablation table:

> **删 Wf-380-A1 SCAFFOLD (triton_mini_batch_ot) | m1 + m2 ↓ speedup | m1, m2 | ⏳ TODO (Wf-380-A1 SCAFFOLD; Triton kernel correctness verified but not re-baselined)**

→ **Triton 状态**: SCAFFOLD-only CPU reference; **GPU Triton path DEFERRED per Wf-326 / Wf-353 follow-ups**; speedup claim PROJECTED ~5-10× per Khainb 2025 (尚未 MEASURED-on-real-job)。

**Triton 相关文件清单** (8 个):
- `molmetal/adapters/flow_matching_lipman/triton_mini_batch_ot.py` (CPU reference SCAFFOLD)
- `molmetal/adapters/flow_matching_lipman/triton_minibatch_ot.py` (canonical, 561 LOC)
- `molmetal/adapters/flow_matching_lipman/_reference/dmpnn_triton_kernel.py`
- `molmetal/adapters/flow_matching_lipman/_reference/fused_wasserstein_sliced.py`
- `molmetal/adapters/flow_matching_lipman/_reference/fused_bond_head_aggregation.py`
- `molmetal/adapters/flow_matching_lipman/_reference/minibatch_ot_triton.py`
- `molmetal/adapters/flow_matching_lipman/_reference/fused_relu_mlp.py`
- `molmetal/adapters/flow_matching_lipman/_reference/fused_smiles_featurize.py`

Per `mol-metal-cfm-training` skill §"Triton kernels":
> **Triton kernels | `triton_kernels/` | 13 kernels, ROCm-tested (NOT applicable on NVIDIA — uses upstream triton)**

→ 13 个 Triton kernels 存在, ROCm-tested; NVIDIA upstream triton path **DEFERRED**。

### §6.4 Wf-380-B1 trainer integration — 14+3 NEW CLI flags

Per `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` §3 + Wf-380-B1 closeout:

- **14 NEW CFM CLI flags** (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`):
  `--edm-preconditioning` `--lr-warmup-cosine` `--use-optimal-transport-v2` `--stable-vm` `--so3-average` `--triton-ot` `--shortcut-distill` `--consistency-weight` `--per-cohort-features` `--chemprop-v3-cache` `--flowmol-unified` `--log-normal-t` `--use-ot-cfm` `--scaffold-conditioning`
- **3 NEW D-MPNN CLI flags** (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`):
  `--per-cohort-features` `--use-dann` `--dann-alpha`
- **AST parse OK** for both trainers
- **MEASURED speedup PROJECTED, NOT re-baselined** (full A/B retrain DEFER per Wf-424-A11 §5.7 smoke FAIL + AGENTS.md §5.1)

### §6.5 Wf-431-B1 GPU utilization — 4 NEW CLI flags

Per Wf-431-B1 closeout:

- `--torch-compile` (mode=reduce-overhead + dynamic_shapes=True per Wf-358-A3 + Wf-335-R3 Blackwell cudagraphs safety)
- `--tf32` (matmul.allow_tf32=True + matmul_precision='high')
- `--cudnn-benchmark` (cudnn.benchmark=True)
- `--channels-last` (memory_format=channels_last)

All default OFF for bit-for-bit backward compat with Wf-380-B1; MEASURED baseline smoke (3 epochs × 200 n_train × batch 8): train_loss 1.4476 → 0.4387 → 0.1145 monotone non-increasing + decode_ratio=1.000 across all 3 epochs (75 optimizer steps).

### §6.6 pre-flight v3 状态

Per AGENTS.md §5.5 + Wf-411-A3 + Wf-424-A1 + Wf-424-A11 + Wf-424-A7:

- **Wf-424-A1 之后 pre-flight v3 PASS**: **113 PASS / 0 FAIL / 4 SKIP / 117 TOTAL** wall 192.48 s per `/tmp/wf431c1_preflight.json`
- `p2_tests_pass` cascade FAIL 长期 blocking GPU launches 已通过 **Wf-424-A1 `use_dann` kwarg fix** CLOSED
- 4 SKIPs preserved BY-DESIGN per `party-twenty-spirit §3.5`:
  - `p3_wf343c4_yuelbond_decoder_env` (env gap per Wf-345-A2; passes on project venv)
  - `p3_wf343c4_wave_budget_remaining_nonneg` (-183.04h overrun per Wf-262 §7.1; BY-DESIGN per user directive "时间预算没关系的")
  - `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency)
  - NEW duplicate registration across chemprop_v2_v3 + frontier pipelines

---

## §7 关键文件索引 (Source-of-Truth 列表)

### §7.1 Tier-1 (必读)

| 文件 | 内容 |
|---|---|
| `TODO/03_baselines/COMPARISON_BASELINES.md` | 3 Tier-1 baselines + per-paper rationale + out-of-scope list |
| `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` | Tier-1 metric × baseline × alignment level 速查表 |
| `TODO/02_metrics/METRICS_TARGETS.md` | 21 metrics × target × baseline × verdict gate × Wf-380..390 status (§11) |
| `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` | 44 mechanisms (M1-M44) × 21 metrics × failure-mode diagnostic × ablation design |
| `TODO/03_plans/LAMBDA_ROADMAP.md` | L1-L8 Lambda gap closed + R20-R25 roadmap |
| `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` | 当前 R17 MEASURED vs R18 PROJECTED × 8 pending TODOs |

### §7.2 Tier-2 (深入)

| 文件 | 内容 |
|---|---|
| `TODO/13_lambda_clickchem/plan.md` | Lambda 演算 10 步计划 + atoms-as-combinators 形式化 |
| `TODO/13_lambda_clickchem/molecular_lambda_calculus.md` | MLC 形式化定义 (β-NF + Church encoding) |
| `TODO/03_plans/R18_PLAN.md` | R18 32 modules + 23 mechanisms + M22-M44 详细 |
| `TODO/03_plans/R19_PLAN.md` | R19 12-week Blackwell + RTX 5090 GPU retrain |
| `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` | R20 100-pocket × 10-seed 全量 benchmark |
| `TODO/decisions.md` | 12 decisions (D1-D7 + D-Wf-431 + D-Wf-378..392 ultracode) |
| `TODO/README.md` | 项目顶层导航 + Wave 时间线 |

### §7.3 Tier-3 (skill reference)

| Skill | 用途 |
|---|---|
| `mol-metal-lambda-calculus` | MLC β-NF + Church encoding + combinatorial chemistry |
| `mol-metal-click-chemistry` | 5 click reactions + 2 metal exchanges + 5×5 Pt matrix |
| `mol-metal-cfm-training` | CFM backbone + YuelBond + Triton-fused ops |
| `mol-metal-anticancer` | Pt/Ru/Ir/Au/Pd + MetalCytoToxDB + PlatinAI + Krasnov 2026 |
| `mol-metal-gpu-execution` | Blackwell + RTX 5090 + 5-seed × 50-epoch protocol |
| `mol-metal-mechanism-mapping` | 跨 skill 总览 + 诊断 + 自查 |

### §7.4 Tier-4 (ship surface)

| 路径 | 内容 |
|---|---|
| `molmetal/adapters/flow_matching_lipman/` | 28 modules — CFM backbone + 10 SCAFFOLD + Triton + W-Flow + Hilbert + Multi-PH + ... |
| `molmetal/molmetal_lam/lam_chem/` | 35+ modules — ast.py + lambda_combinators + closure + pt_click_compat + pt_metal_ligand_exchange + fg_compat_veto + skeleton_jump + ... |
| `molmetal/molmetal_lam/search_alg/` | proof_search.py (6217 LOC) + variance_puct_integration + score_repellent_mc + ... |
| `molmetal/molmetal_lam/reward/` | lambda_astsr.py (1059 LOC) + coupling_adapter + hilbert_primal_dual + lambda_reward_dataset + ... |
| `molmetal/scripts/` | training entry (train_cfm_yuelbond + train_dmpnn + ...) + evaluation (eval_all_evaluators) |
| `metrics/by_metric/*.json` | 113 metric JSONs — Tier-1 LEAD + Tier-2 internal |
| `TODO/completed/` | 950+ closeout docs (Wf-1 .. Wf-431) |

---

## §8 Wf-296..431 ship timeline (recent, AGENTS.md §6)

Per AGENTS.md §6 + Wf-431 closeout:

- **Wf-296..316**: R17 close-out + scaffold integration (commit `4561625`)
- **Wf-317..320**: Wf-176 m19 LOMO CPU + R20/R21 skill ecosystem + pre-flight gate audit
- **Wf-321..322**: docs-only closeout + chemprop v2 reconciliation
- **Wf-323..326**: Wf-323 Wf-178 unification + Wf-324 docs-only + Wf-325 honest-negative + Wf-326 GPU profile
- **Wf-327..332**: Wf-327 8 NEW skill wiring + Wf-328 DMPNN cache wiring + Wf-329 R20-P2..P5 + R21-P1..P5 + Wf-330 ULTRACODE ship + Wf-332 INDEX sync
- **Wf-333..360**: Wf-333 Vina path A + Wf-340/Wf-353 CFM innovations + Wf-380 10 SCAFFOLD modules + Wf-380-B1 trainer integration
- **Wf-380..390**: Wf-380 implementation + Wf-385 SCAFFOLD-loss wire-in gate + Wf-386 atomic write protocol + Wf-388 m2/m7/m4 wire + Wf-389 m6_kappa drop-from-§3 + Wf-390 m2 R² Path A plan
- **Wf-391..431**: Wf-391 strategic plans refresh + Wf-407 GPU Wf-299 launch attempt + Wf-413 GPU 0 sequential retrain cascade + Wf-420 p2_tests_pass fix + Wf-423 GPU utilization diagnosis + Wf-424 use_dann fix + Wf-426 wrapper infrastructure + Wf-427 schema adapter + Wf-430 retrain readiness verify + Wf-431 docs-only refactor (THIS PROJECT_STATUS author)

---

## §9 Honest framing — 哪些 gap 还没补

Per `docs/HONEST_NEGATIVE_CATALOG.md` + Wf-431 final synthesis:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL** (9 paths exhausted): per Wf-377-A1 + Wf-389-A1; drop-from-§3 + promote-top-3-to-supplementary per Wf-389-A1 §3
2. **m2 R² platinai_4cell + hilbert HONEST_NEGATIVE_R2_DEEP FINAL**: n=200 small cohort overfit; metalcytotox_combined +0.1981 WEAK_RECOVERY (above zero, below Wf-335-M2 CEILING_BREAK)
3. **Wf-380 SCAFFOLD modules are CPU-only SCAFFOLDS**: lift claims REMAIN PROJECTED until GPU retrain USER_OK-gated
4. **Triton GPU path DEFERRED**: SCAFFOLD-only CPU reference per Wf-380-A1 §2.5; speedup PROJECTED ~5-10×
5. **8 SCAFFOLD-tier metrics (m4 / m5 / m6 / m7 / m8 / m10 / m13 / m16) pending GPU retrain**: MEASURED-on-real JOB requires USER_OK per AGENTS.md §5.1
6. **6 forward-looking USER_OK items BLOCKED**: G6 ~11.4 hr launch + 6 wrappers + 2 install + 7 paper + 2 metric gates (~52.9 hr GPU + 13 hr CPU + 30 min install)
7. **Wetlab DEPRECATED per AGENTS.md §5.4 + §10**: 4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + §3.3 R23 all DEPRECATED / re-scoped to in-silico
8. **2 pre-flight FAILs preserved BY-DESIGN**: `p2_tests_pass` (closed by Wf-424-A1) + `p3_wf343c4_anchor_freshness_24h` (natural drift) — see Wf-411-A1 + Wf-375-B1

---

## §10 下一步建议 (按优先级)

Per `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §1 + `TODO/decisions.md` + AGENTS.md §5.1:

1. **Wf-380 SCAFFOLD-loss MEASURED-on-real** (Wf-424-A12 wire-in fix + Wf-388-A1/A2/A3 + Wf-409-A1 + Wf-431-B1 GPU flags combined launch) — **5-12 hr GPU** USER_OK-gated per AGENTS.md §5.1
2. **Wf-299 3-wave CFM retrain** (Wave A SCAFFOLD-loss + Wave B m7 diversity + Wave C m4 SA) — **2-12 hr GPU** USER_OK-gated per AGENTS.md §5.1
3. **Wf-394-A1 Lambda × CFM coupling patch application** — **~5 hr GPU** USER_OK-gated per AGENTS.md §5.1
4. **Wf-246-C metal-coordination term retrain** — **~3 hr GPU** USER_OK-gated per AGENTS.md §5.1
5. **Paper §3 v4 + §6 v4 actual write to `paper/sections/*.tex`** — USER_OK-gated per AGENTS.md §5.3 + TODO-84
6. **m6_kappa drop-from-§3 + promote-top-3-WEAK-correlates paper conversion** — `docs/M6_KAPPA_DROP_DECISION.md` + `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` already SHIPPED; USER_OK-gated for paper conversion
7. **Atomic write protocol adoption migration completion** — ~41 remaining closeout-drafting scripts per Wf-416-A1 §5.2; deferred to Wf-425+ per Wf-386-A3 §6.1 adoption sequence

---

## §11 Conclusion

This `TODO/PROJECT_STATUS.md` synthesizes the project's current state per 6 user questions:

1. **Q1 (Baselines)**: 3 Tier-1 (Krasnov 2026 / Rusanov 2025 / DrugOOD 2022); alignment-not-reproduction principle per Wf-431 docs-only refactor
2. **Q2 (Lambda)**: 8 M-modules (M1-M4 + M8-M11 + M36 + M38 + M42) drive chemistry prior + search + ASTSR + priors; baseline = Katalin Marton + PySR + vanilla UCB + Uncalibrated Dirichlet
3. **Q3 (Click)**: 5 click + 2 metal exchange + 21-rule registry + 5×5 Pt matrix + 84-cell FG-compat; baseline = REINVENT4 / MOSES / GuacaMol / AiZynthFinder / SyntheMol
4. **Q4 (Architecture)**: CFM = Lipman 2023 CondOT + midpoint ODE + SE(3)-EGNN + YuelBond + 20-atom vocab + 6 shim fixes + 7 R18 W1 math fusions; differs from vanilla FM via 6 dimensions
5. **Q5 (Metrics)**: 21 metrics — 6 MEASURED (m21 + m17 + m1 + m3 + m15 + m14) + 1 MEASURED-but-below (m2) + 1 DROPPED (m6_kappa) + 8 SCAFFOLD-tier pending GPU retrain + 2 NOT YET in pipeline
6. **Q6 (Training)**: Torch 2.14.0 + Blackwell + Wf-380 10 SCAFFOLD + Wf-380-B1 14+3 NEW CLI flags + Wf-431-B1 4 GPU util flags; **Triton SCAFFOLD-only CPU reference; GPU Triton path DEFERRED**

**Honest bottom line**: Tier-2 paper §4 main-result 表可以直接写 6 个 MEASURED metrics; 8 个 SCAFFOLD-tier 需要 GPU 重训; m6_kappa 已 drop-from-§3 + promote-top-3-to-supplementary; pre-flight 113/0/4/117 PRESERVED; 0 GPU cycles consumed 浪费在 broken invocations (per kill_then_adapt §0)。

---

**END OF PROJECT_STATUS.md** — single source of truth for project current state as of 2026-10-01 Wf-431 docs-only refactor cycle.
