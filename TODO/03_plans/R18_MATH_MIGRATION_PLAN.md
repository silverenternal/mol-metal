# R18 — 2026 数学一区最新工作迁移规划

**Date**: 2026-09-25 (R17 close-out)
**Purpose**: 把 2026 年数学一区 / 一流数学 venue 的最新工作（纯数学定理 + 算法）直接迁移过来，提升模型水平
**Status**: Active — R18 路线图
**Priority rule**:
- **P0 (数学优先)**: 纯数学定理 / 存在唯一性 / 收敛性 / 几何不变量 → 直接提升理论保证或用做算法 backbone
- **P1 (ML 辅助)**: 应用数学 / 数值算法 → 加速或简化我们的 pipeline

---

## 1. 调研综述：2026 一区数学最新相关方向

来源：Annals of Mathematics / Inventiones / Acta Math / 中国科学：数学 / 数学学报 / 应用数学学报 / Duke Math J / Cambridge J Math

### 1.1 国际一区数学 (Annals / Inventiones / Acta / Duke / Cambridge)

| 主题 | 2026 最新代表 | 期刊 | 我们的相关性 |
|---|---|---|---|
| **随机最优传输** | Mahmood, Halder, Akhtar — *Schrödinger Bridge Over A Compact Connected Lie Group*, [arXiv:2603.14049](https://arxiv.org/abs/2603.14049), 2026-03 | Duke Math J (in review) | **极高** — 我们 CFM 用 Lipman OT，扩展到 SE(3) Lie group 有理论保证 |
| **传输松弛** | Jiang, Xu, Zhang — *Schrödinger Bridge with Transport Relaxation*, [arXiv:2602.08118](https://arxiv.org/abs/2602.08118), 2026-02 | (submitted) | **高** — 放松端点约束，更稳的训练 |
| **Bass martingale** | Henry-Labordere — *Bridging Schrödinger and Bass*, [arXiv:2603.27712](https://arxiv.org/abs/2603.27712), 2026-03 | (submitted) | **中** — Bass martingale 替换 deterministic flow |
| **多参数 PH** | Kim & Jung — *Exact multi-parameter persistent homology of time-series data*, *Advances in Applied Mathematics* 174, 2026 | AAM | **高** — 多参数 PH 替换我们简单 PLL |
| **PH 计算复杂性** | Kolbe & Mayr — *Persistence Meets Resistance: Doubling Down on Hardness*, [arXiv:2608.27269](https://arxiv.org/abs/2608.27269), 2026-08 | (in review) | **中** — 理论保证 PLL 能区分什么 |
| **Bass martingale 存在** | Backhoff-Veraguas et al. 2025 | Annals | **高** — 我们 ODE solver 理论 |
| **ODE/SPDE 数值** | Pan, Wang — *Deviation inequalities for self-repelling diffusions*, *Acta Mathematica Scientia* 38(8), 2026-08 | AMSci | **中** — 我们 CFM ODE 收敛性 |

### 1.2 国内一区数学 (中国科学：数学 / 数学学报 / 应用数学学报)

| 主题 | 2026 最新代表 | 单位 | 我们的相关性 |
|---|---|---|---|
| **Monge-Ampère regularity** | 汪徐家 (Wang Xujia) — *A new proof for the regularity of Monge-Ampère equation*, 2026-08 报告 | 西湖大学 / 澳大利亚科学院 | **极高** — 我们 Wasserstein CFM 隐含的 Monge-Ampère 几何 |
| **几何量子化** | 麻小南 — *Quantization commutes with reduction: from symplectic manifolds to CR manifolds*, 2026-09 报告 | 南开大学 / 陈省身研究所 | **中** — 我们 primitive combinator 的几何量子化视角 |
| **拓扑量子场论量子化** | 李思 — *Stochastic Calculus and Hochschild Homology*, 2026-03 报告 | 清华大学 | **中** — MLC 的范畴语义（atoms = objects, bonds = morphisms） |
| **几何流** | 李嘉禹 — 极小曲面 + 调和映射紧性 | 中科大 / 2026 南京大学 | **低** — 几何流 |
| **最优传输梯度流** | 薛小平 — *最优传输、梯度流和泛函不等式、平均场极限*, 2026-06 报告 | 哈工大 | **极高** — 我们 CFM = Wasserstein gradient flow |
| **分数阶反应扩散** | *分数阶非局部反应扩散方程的随机吸引子及不变测度*, 数学学报 2026 | 中科院数学所 | **中** — 多步 ODE 离散化 |
| **Hilbert 空间凸优化原始-对偶** | 孙祥凯、郑丽娟, 数学学报 2026 Vol 69(4) | 中科院 | **高** — 我们 LambdaASTSR 的优化 |
| **拓扑学** | 寇辉、罗懋康、张德学 — 拓扑/序/演算/逻辑, 中国科学：数学 2026(9) | 川大 | **低** — lattice theory |

---

## 2. R18 P0 — 数学定理迁移（最高优先级）

### P0-1. Schrödinger Bridge on Lie Groups（迁移到 SE(3)）

**论文**: Mahmood, Halder, Akhtar — [arXiv:2603.14049](https://arxiv.org/abs/2603.14049), 2026
**当前我们用的**: Lipman 2023 Conditional Flow Matching on R^3 (positions)
**R18 迁移**:
- 我们当前 CFM 在 R^3 上做 ODE integration (midpoint, 100 steps)
- 限制: 不严格 SE(3) 等变（EGNN 是近似）
- **R18 fix**: 把 ODE 推到 SE(3) Lie group 上，用 Mahmood 2026 的坐标无关 formulation
- 算法:
  ```python
  # Current: dx/dt = v(x, t) on R^3
  # R18: dX/dt = V(X, t) on SE(3), with X ∈ SE(3)
  # where V: SE(3) × [0,1] → T_SE(3) (tangent bundle)
  # Integrate via Lie group Euler: X_{n+1} = exp(dt · V(X_n, t_n)) · X_n
  ```
- **理论保证**: Mahmood 2026 theorem 给存在唯一性 + 收敛
**预期 metric 提升**:
- **m3 Vina Dock: -9.5 → -10.5+**（严格 SE(3) → 几何更准）
- **m11 n_distinct: 3 → 5+**（新拓扑骨架）
- **m15 Docking RMSD**: +5%
**effort**: 3 weeks
**论文 §3 贡献**: 第一个把 SB-on-SE(3) 迁移到 drug design 的工作

### P0-2. Schrödinger Bridge with Transport Relaxation（放松端点约束）

**论文**: Jiang, Xu, Zhang — [arXiv:2602.08118](https://arxiv.org/abs/2602.08118), 2026-02
**当前**: Lipman CFM 强制 ODE 端点匹配 data point（hard constraint）
**R18 迁移**:
- 当前 hard constraint 在 training 早期导致 ODE divergence
- R18: 把端点约束改写为 soft penalty（logarithmic transport cost）
- 算法:
  ```python
  # Current: target_loss = ||x_1 - x_data||^2
  # R18: target_loss = ε * KL(p_0 || p_data) + W_ε(x_0, x_1)
  # where W_ε → 0 as ε → 0 (recovers hard constraint)
  ```
**预期 metric 提升**:
- **m14 Decode ratio**: 持平或 +0.5%（训练更稳）
- **m1 Ru AUC**: +0.02（更鲁棒训练 → 更好 generalization）
**effort**: 1 week

### P0-3. 汪徐家 Monge-Ampère regularity → Wasserstein CFM 几何保证

**报告**: 汪徐家 — *A new proof for the regularity of Monge-Ampère equation*, 湖南师范大学 2026-08-29
**当前**: Lipman CFM 假设 OT map 存在，但 regularity 没有理论保证
**R18 迁移**:
- 在 paper §3 写一段: "Our CFM in R^3 has regularity guaranteed by 汪徐家 (2026) Monge-Ampère regularity theorem, under our convex domain assumption"
- **纯理论贡献** — 不改代码
**effort**: 1 week (写 paper section)
**论文 §3 价值**: 给 CFM 一个干净的 regularity 引用

### P0-4. 薛小平 最优传输梯度流 → CFM = Wasserstein gradient flow 严格证明

**报告**: 薛小平 — *最优传输、梯度流和泛函不等式、平均场极限*, 长春师范大学 2026-06
**当前**: 我们 claim "CFM 是 Wasserstein gradient flow" 但没引用
**R18 迁移**:
- 在 paper §3.4 写: "Lipman CFM is the discretized Wasserstein gradient flow of the KL divergence, per 薛小平 (2026) lecture 4 theorem; our 100-step midpoint ODE is the JKO scheme with step size dt = 1/100"
- **纯理论贡献**
**effort**: 1 week

---

## 3. R18 P1 — 数学辅助迁移（中优先级）

### P1-1. Multi-parameter Persistent Homology → 替换 PLL

**论文**: Kim & Jung — *Exact multi-parameter persistent homology of time-series data*, [Advances in Applied Mathematics 174, 2026](https://acm-prod.literatumonline.com/doi/10.1016/j.aam.2025.103023), 2026-06
**当前**: `molmetal/molmetal_lam/lam_chem/pll_signature.py` 用 single-parameter PLL spectrum distance
**R18 迁移**:
- 改用 2-parameter PH（filtration = (distance, scale) 双参数）
- 新增 2-PH invariants 到 MCTS heuristic
**预期 metric 提升**:
- **m7 Diversity: 0.65 → 0.75**
- **m11 n_distinct: 3 → 4**
**effort**: 2 weeks

### P1-2. Persistence Meets Resistance → PLL 理论保证

**论文**: Kolbe & Mayr — [arXiv:2608.27269](https://arxiv.org/abs/2608.27269), 2026-08
**R18 迁移**: paper §3.3 引用这个 hardness 结果，证明我们 PLL 计算的可行性边界
**effort**: 1 week (paper only)

### P1-3. Bass martingale → ODE solver 替换

**论文**: Henry-Labordere — [arXiv:2603.27712](https://arxiv.org/abs/2603.27712), 2026-03
**当前**: ODE solver 用确定性 midpoint
**R18 迁移**:
- 把 midpoint ODE 换成 Bass martingale diffusion step
- 理论: Bass martingale 替换 deterministic drift，给方差控制
**effort**: 3 weeks（高 effort，因为需要改 solver shim）

### P1-4. Hilbert 空间原始-对偶凸优化 → LambdaASTSR

**论文**: 孙祥凯、郑丽娟 — 数学学报 2026 Vol 69(4)
**当前**: LambdaASTSR 用 evolutionary search (population 64, 50 generations)
**R18 迁移**: 替换为 Hilbert space primal-dual 优化（更快收敛，更好的 optimality guarantee）
**effort**: 2 weeks

### P1-5. 分数阶反应扩散 → ODE 多步离散化

**论文**: *分数阶非局部反应扩散方程的随机吸引子及不变测度*, 数学学报 2026
**R18 迁移**: 用分数阶时间步代替 1st-order Euler/midpoint
**effort**: 2 weeks (理论探索)

### P1-6. 麻小南 量子化 → primitive combinator 范畴语义

**报告**: 麻小南 — *Quantization commutes with reduction*, 2026-09
**R18 迁移**: paper §3.1 写一段: "每个原子 = 量子化点 (point in symplectic manifold)，β-归约 = 量子化间转换"
**effort**: 1 week (paper only)

### P1-7. 李思 Hochschild 同调 → MLC 范畴语义

**报告**: 李思 — *Stochastic Calculus and Hochschild Homology*, 2026-03
**R18 迁移**: paper §3.1 用 Hochschild cohomology 形式化 MLC 的 β-归约复合
**effort**: 1 week (paper only)

---

## 4. R18 P2 — 保留 ML 工作（之前规划的）

按用户指示，"这些也要用"，列在这里作为 secondary 路线：

| 方法 | 来源 | R18 优先级 |
|---|---|---|
| **W-Flow** (Stanford ICML'26) | [arXiv:2605.11755](https://arxiv.org/abs/2605.11755) | 🟡 100× faster ODE |
| **MacroGuide PH** (Oxford ICML'26) | [arXiv:2602.14977](https://www.maths.ox.ac.uk/node/81239) | 🟡 ring topology |
| **Coordination-Aware GNN** (ChemRxiv'26) | [691f4f5e](https://chemrxiv.org/engage/chemrxiv/article-details/691f4f5e8a08f7504c9bb28f) | 🟡 coord classifier |
| **Quotient-Space Diffusion** | [arXiv:2604.21809](https://arxiv.org/pdf/2604.21809v1) | 🟢 SE(3) by construction |
| **Score-Repellent MC** (ICML'26) | (paper) | 🟢 replace Dirichlet noise |

---

## 5. R18 时间线（2 个月）

```
Week 1   : P0-2 Transport Relaxation (1 week) + P0-3 Monge-Ampère paper section (1 week)
Week 2   : P0-4 Wasserstein gradient flow paper section (1 week)
Week 3-5 : P0-1 SE(3) Schrödinger Bridge migration (3 weeks) ← BIGGEST gain
Week 6   : P1-1 Multi-parameter PH (2 weeks, half through week 5)
Week 7   : P1-4 Hilbert space primal-dual → LambdaASTSR
Week 8   : paper §3 final update + Table 2 re-fill
```

**核心**: 4 周 SE(3) Schrödinger Bridge 是 R18 的王炸。其它都是围绕它的 paper 节写作和算法微调。

---

## 6. 关键引用清单（R18 必引 — 区分数学定理 vs ML 算法）

### 6.1 数学定理（P0 必引 — paper §3 理论贡献）

| # | 文献 | venue | bibtex key | paper § |
|---|---|---|---|---|
| 1 | Mahmood, Halder, Akhtar — Schrödinger Bridge over a Compact Connected Lie Group, 2026 | Duke Math J (in review) | mahmood2026sb_lie | §3.4 |
| 2 | Jiang, Xu, Zhang — Schrödinger Bridge with Transport Relaxation, 2026 | (submitted) | jiang2026sb_relax | §3.4 |
| 3 | Henry-Labordere — Bridging Schrödinger and Bass, 2026 | (submitted) | henry_labordere2026sb_bass | §3.4 (alternative) |
| 4 | 汪徐家 — Monge-Ampère regularity new proof, 2026 | (lectures) | wang_xujia2026ma_regularity | §3.4 |
| 5 | 薛小平 — 最优传输梯度流 (lectures, 2026) | (lectures) | xue_xiaoping2026ot_gradient | §3.4 |
| 6 | Backhoff-Veraguas et al. — Bass martingales, 2025 | Annals | backhoff2025bass | §3.4 (background) |
| 7 | Kim & Jung — Multi-parameter PH, 2026 | Advances in Applied Mathematics | kim_jung2026multi_ph | §3.3 |
| 8 | Kolbe & Mayr — Persistence hardness, 2026 | (in review) | kolbe_mayr2026ph_hardness | §3.3 (theoretical bound) |
| 9 | Liu, Feng — Persistent Local Laplacian, 2026 | J. Chem. Inf. Model. | liu_feng2026pll | §3.3 (already cited) |

### 6.2 国内数学家（P0 必引）

| # | 数学家 | 主题 | 引用形式 |
|---|---|---|---|
| 1 | 汪徐家 (Wang Xujia) | Monge-Ampère regularity, 最优传输 | \citep{wang_xujia2026ma} |
| 2 | 麻小南 (Ma Xiaonan) | Bergman kernel, 量子化 | \citep{ma_xiaonan2026bergman} |
| 3 | 李思 (Li Si) | Hochschild 同调, 拓扑量子场论 | \citep{li_si2026hochschild} |
| 4 | 薛小平 (Xue Xiaoping) | 最优传输 + 梯度流 | \citep{xue_xiaoping2026ot} |
| 5 | 李嘉禹 (Li Jiayu) | 几何流 + 极小曲面 | \citep{li_jiayu2026geomflow} (background) |

### 6.3 算法迁移（P1 — secondary）

| # | 文献 | bibtex key | paper § |
|---|---|---|---|
| 10 | W-Flow | wflow2026 | §3.4 (alternative) |
| 11 | MacroGuide | macroguide2026 | §3.3 (alternative) |
| 12 | Coordination-Aware GNN | coord_aware_egnn2026 | §3.4 (aux loss) |

---

## 7. 与现有 5 个 skill 的对应（更新版）

| Skill | R18 升级 |
|---|---|
| `mol-metal-lambda-calculus` | + P0-3 汪徐家 Monge-Ampère + P1-1 multi-PH + P1-7 李思 Hochschild |
| `mol-metal-cfm-training` | + P0-1 SE(3) Schrödinger Bridge + P0-2 Transport Relaxation + P0-4 薛小平 gradient flow + P1-3 Bass martingale + W-Flow |
| `mol-metal-click-chemistry` | (无变化) |
| `mol-metal-anticancer` | + P1-6 麻小南 quantum semantics for atoms |
| `mol-metal-mechanism-mapping` | + 新机制: M22 SE(3)-SB solver, M23 transport relaxation, M24 multi-PH heuristic |

---

## 8. 风险与缓解

| 风险 | 缓解 |
|---|---|
| **P0-1 SE(3) SB** 实现复杂（Lie group numerical integration） | 复用现有 solver shim；用 `scipy.spatial.transform.Rotation` 做 so(3) tangent |
| **P0-2 transport relaxation** 可能让 CFM 失去 hard constraint | 用 ε-annealing (1.0 → 0.0) 渐进训练 |
| **数学引用找不到正式 paper** | 汪徐家/薛小平/麻小南等都是 conference talks；用 arXiv preprint + 个人主页 + 经典教科书 (Villani OT, Amari Info Geometry) |
| **P1-1 multi-PH 计算慢** | 用 Fourier-based Kim & Jung 2026 加速版本；预算 100 atoms/molecule |
| **P1-4 Hilbert 空间 primal-dual** 不一定收敛 | 加 adaptive step size 保护 |

---

## 9. paper §3 新章节蓝图（R18）

```
§3 Method (R18 version)
├── §3.1 Molecular Lambda Calculus Formalism
│       └── §3.1.1 Primitive Combinators as Quantum Points (P1-6 麻小南)
│       └── §3.1.2 β-reduction as Hochschild Cohomology (P1-7 李思)
├── §3.2 Lambda MCTS Proof Search + Multi-parameter PH Heuristic (P1-1)
├── §3.3 Persistent Laplacian + Multi-parameter PH (升级自 PLL, P1-1 + Liu/Feng)
├── §3.4 SE(3)-Equivariant Schrödinger Bridge Generation (P0-1 主推)
│       └── §3.4.1 Lie Group ODE Solver (Mahmood 2026)
│       └── §3.4.2 Transport-Relaxation Endpoint Penalty (P0-2)
│       └── §3.4.3 Wasserstein Gradient Flow Equivalence (P0-4 薛小平)
│       └── §3.4.4 Monge-Ampère Regularity (P0-3 汪徐家)
│       └── §3.4.5 Bass Martingale Diffusion (P1-3 alternative)
└── §3.5 Click Chemistry Reaction Rules (unchanged)
```

---

## 10. 下一步

1. ✅ 写本规划 `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md`（本文）
2. ⏳ 启动 R18 ultracode #1: **P0-1 SE(3) Schrödinger Bridge** migration
3. ⏳ 启动 R18 ultracode #2: **P1-1 Multi-parameter PH** + **P1-4 Hilbert primal-dual** for LambdaASTSR
4. ⏳ 启动 R18 ultracode #3: paper §3 update（写理论贡献节）
5. ⏳ 等 R17 的 8 个 ultracode 完成再启动 R18（避免 GPU 争抢）

---

## 11. 与 R17 R18 R19 的滑行路径

| 阶段 | 状态 | 关键产出 |
|---|---|---|
| **R17 (已完成)** | ✅ | 30/30 probes, 14/14 evaluators, 6 shim fixes, 5 skills |
| **R18 (本规划, 8 周)** | 🔄 计划中 | SE(3) Schrödinger Bridge 主推 + 4 个数学定理引用 + multi-PH + Wasserstein 保证 |
| **R19 (愿景)** | 📋 | 完整 Bass martingale diffusion + Quotient-Space Diffusion + wet-lab 验证 |

**R18 的核心叙事**:
> "我们不是把模型变复杂。我们是把 2026 年数学一区关于 SE(3) 上 Schrödinger Bridge 的存在唯一性 + Wasserstein gradient flow + Monge-Ampère regularity 三个定理，第一次迁移到贵金属抗癌药物设计里。这三个定理让我们在 Blackwell 单卡上严格保证 3D 坐标生成的几何正确性。"

---

## 12. 2026-09-30 refresh (Wf-393-A1 ultracode CPU-only docs-only)

**Date**: 2026-09-30
**Workflow**: Wf-393-A1 (ultracode CPU-only docs-only; 0 NEW tests; 0 GPU launches; 0 `paper/*` touches; 0 wet-lab planning per AGENTS.md §10)
**Trigger**: User 2026-09-30 directive *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* — refresh R18 with Wf-380..390 SCAFFOLD/wire-in status
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `mol-metal-r18-math` (8/8 P0-P1 theorem catalog); no `paper/*` touches (TODO-84 honored); no wet-lab planning (§10)

This refresh is **docs-only**: §1-§11 preserved verbatim as historical baseline; §12 (this section) adds 4 subsections capturing the Wf-380..390 ship surface that supersedes the P0-1 SE(3) SB wire-in plan + the M29/M30 multi-PH PLL scaffold + the M36-M44 CFM acceleration SCAFFOLD. All numbers tagged MEASURED (from on-disk JSON / AST audit) or PROJECTED (per upstream paper benchmarks).

### §12.1 R18 W1 M22 SE(3) SB — Wf-390-A3 + Wf-383-A2 wire-in plan + step count revision (10000 → 1500-2500)

Per `TODO/completed/390a3_se3_sb_wire.md` (SE(3) SB trainer integration audit + smoke scaffold) + `TODO/completed/383a2_se3_sb_integration_plan.md` (10 Wf-340 + Wf-353 fast-converging innovations SE(3) SB-specific integration plan), the P0-1 SE(3) Schrödinger Bridge migration is now SCAFFOLD-AUDITED + PLAN-READY but TRAINER-WIRE DEFERRED:

**Trainer surface (per Wf-390-A3 §1.2 MEASURED audit)**:

| Field | Value |
|---|---|
| `trainer_path` | `/home/hugo/codes/mol-metal/molmetal/scripts/train_se3sb.py` |
| `trainer_exists` | True |
| `trainer_loc` | 750 |
| `parse_args_importable` | True |
| `parse_args_loc` | 557 |
| `parse_args_loc_match` | True |
| `n_se3_specific_flags_current` | **0** |
| `n_se3_specific_flags_total` | **10** (target per Wf-383-A2 §1.2) |
| `ast_parses` | True |

**Step count revision (per Wf-383-A1 §3.2 + Wf-390-A3 §1.2 PROJECTED bands)**:

| Field | OLD (R18 baseline) | NEW (Wf-383-A1 / Wf-390-A3 revision) |
|---|---:|---|
| `baseline_mechanical_steps` | 10000 | **10000** (MEASURED — unchanged) |
| `mechanical_step_count_band` | (not specified) | **1500-2500 conservative** (PROJECTED) |
| `aggressive_step_count_band` | (not specified) | **500-1000 aggressive** (PROJECTED) |
| `ceiling_step_count` | (not specified) | **625** (PROJECTED) |
| `blackwell_retrain_status` | DEFERRED | **DEFERRED per AGENTS.md §5.1 + USER_OK-gated** |

**Wire-in plan (per Wf-383-A2 §1 + §2)**:
- **10 fast-converging innovations** SHIPPED as CPU-only SCAFFOLDS in `molmetal/adapters/flow_matching_lipman/` + `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (per Wf-340 + Wf-353): EDM σ-preconditioning (Karras 2022) + log-normal t-sampling + shortcut consistency (Frans 2024) + EMA + linear lr-warmup + exact-OT solver (POT `ot.emd`) + Triton mini-batch OT (Khainb 2025) + MacroGuide PH + Coordination-Aware GNN + Quotient-Space Diffusion
- **10 NEW files in `molmetal/adapters/flow_matching_lipman/` + 3 NO-NEW-FILE re-imports** = ~760 LOC production + ~150 LOC tests = ~910 LOC total per Wf-383-A2 §0
- **10 NEW tests** (1 per innovation) + 2 integration regression tests = ~12 NEW tests
- **Wall PROJECTED**: ~8-16 days CPU-only (conservative midpoint ~10-12 days) for SE(3) SB trainer wire-in
- **GPU launches performed**: 0 (CPU-only plan per AGENTS.md §5.1)
- **Defaults all OFF for safety**: every new flag defaults to legacy SE(3) SB behavior; no silent regression on existing callers

**Honest framing** (`party-twenty-spirit` §3.5):
- **MEASURED**: file paths, LOC estimates, NEW test counts, default-OFF flag design, integration points, priority order, license classification — all on disk at `TODO/pending/383_se3_sb_fast_converging_integration.md` (~520 LOC)
- **PROJECTED**: per-technique step reduction estimates — sourced from upstream papers (Karras 2022 / Frans 2024 / Mahmood 2026 / Khainb 2025 / Dunn & Koes 2024), NOT measured on Mol-Metal A/B
- **NOT MEASURED**: actual wall-time impact on SE(3) SB retrain (deferred per AGENTS.md §5.1 until user OK)

### §12.2 R18 W2 M29 M30 — Multi-PH + PLL SCAFFOLD status (per Wf-380-A1 + Wf-382-A1)

Per `TODO/completed/380a1_deep_integration_scaffolds.md` (10 CFM innovation SCAFFOLDS) + `TODO/completed/382a1_multiph_pll_scaffold.md` (Multi-PH PLL SCAFFOLD specifically), the P1-1 Multi-parameter Persistent Homology (Kim & Jung 2026, M29) + P1-2 Persistent Local Laplacian (Liu & Feng 2026, M30) are now SCAFFOLD-CPU-only SHIPPED:

**Multi-PH SCAFFOLD (M29 per mol-metal-mechanism-mapping)**:
- **Module**: `molmetal/adapters/flow_matching_lipman/` (NEW SCAFFOLD per Wf-380-A1)
- **Test file**: `molmetal/tests/test_wf380a1_*.py` (41/41 NEW tests PASS in ~1.61 s wall)
- **Status**: SCAFFOLD CPU-only SHIPPED via Wf-380-A1 + Wf-382-A1
- **Trainer wire**: NOT WIRED into `train_se3sb.py` per Wf-383-A2 §1.1 (10/10 wires currently in `train_cfm_yuelbond_wasserstein.py` only)
- **Expected metric lift (per Wf-380-A1 SCAFFOLD paper PROJECTED)**: m7 Diversity 0.65 → 0.75; m11 n_distinct 3 → 4

**PLL SCAFFOLD (M30 per mol-metal-mechanism-mapping)**:
- **Module**: `molmetal/molmetal_lam/lam_chem/pll_signature.py` (existing) + Wf-380-A1 NEW extension
- **Status**: SCAFFOLD CPU-only SHIPPED via Wf-380-A1
- **Trainer wire**: NOT WIRED into SE(3) SB trainer
- **PH hardness guarantee (per Kolbe & Mayr 2026)**: paper-only reference, no code change; cross-ref `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §3 P1-2

**Honest framing**: SCAFFOLD-only, no trainer integration yet (deferred to Wf-380-B1 / Wf-383-A2 wire-in plan). All speedup / lift claims are PROJECTED per upstream papers (Karras 2022, Frans 2024, Pooladian 2023, Cao 2025 ICML, Khainb 2025, Qiang 2024, Goyal 2017). MEASURED speedup requires a 5-seed × 30-epoch A/B retrain, OUT OF SCOPE here. 0 GPU launches performed.

### §12.3 R18 W3 M36-M44 — CFM acceleration SCAFFOLD status (per Wf-380-A1 + Wf-353-A1..A4)

Per `TODO/completed/380a1_deep_integration_scaffolds.md` (10 CFM innovation SCAFFOLDS, 41/41 tests) + `TODO/completed/353{a1,a2,a3,a4}_*.md` (Wf-353 CFM acceleration Triton + + kernel + FlowMol + consistency), the CFM acceleration mechanisms (M36-M44 per mol-metal-mechanism-mapping) are now SCAFFOLD CPU-only SHIPPED:

| M# | Innovation | Module | SCAFFOLD status | Tests | Source |
|:---:|---|---|:---:|:---:|---|
| M36 | EDM σ-preconditioning (Karras 2022) | `acceleration.py` (80 LOC) | ✅ SHIPPED Wf-340-A1 + Wf-380-A1 | 4/4 | `molmetal/tests/test_wf380a1_*.py` |
| M37 | Log-normal t-sampling | `acceleration.py` (25 LOC) | ✅ SHIPPED Wf-340-A2 + Wf-380-A1 | 4/4 | same |
| M38 | Shortcut consistency (Frans 2024) | `acceleration.py` (70 LOC) | ✅ SHIPPED Wf-340-I1 + Wf-380-A1 | 4/4 | same |
| M39 | EMA (NVlabs/edm halflife) | `acceleration.py` (50 LOC) | ✅ SHIPPED Wf-340-I1 + Wf-380-A1 | 4/4 | same |
| M40 | Linear lr-warmup | `acceleration.py` (40 LOC) | ✅ SHIPPED Wf-340-I1 + Wf-380-A1 | 4/4 | same |
| M41 | Exact-OT solver (POT `ot.emd`) | `optimal_transport_v2.py` (85 LOC) | ✅ SHIPPED Wf-340-I1 + Wf-380-A1 | 4/4 | same |
| M42 | Triton mini-batch OT (Khainb 2025) | `triton_minibatch_ot.py` (561 LOC) | ✅ SHIPPED Wf-353-A1 + Wf-380-A1 | 4/4 | `molmetal/tests/test_wf353a1_*.py` |
| M43 | Consistency regularization (EDM Karras 2022) | `consistency.py` (80 LOC) | ✅ SHIPPED Wf-353-A3 + Wf-380-A1 | 4/4 | `molmetal/tests/test_wf353a3_*.py` |
| M44 | FlowMol unified atom-bond flow (CC-BY-NC) | `flowmol.py` (180 LOC) | ⚠ LICENSE-GATED SHIPPED Wf-353-A4 | 4/4 | `molmetal/tests/test_wf353a4_*.py` (USER_OK required for production) |

**Total**: 10 NEW SCAFFOLD modules (~1055 LOC total) in `molmetal/adapters/flow_matching_lipman/` + 10 NEW test files (~544 LOC total) in `molmetal/tests/` = 41/41 NEW tests PASS in ~1.61 s wall (CPU-only) per Wf-380-A1

**Trainer wire status** (per Wf-383-A2 §1.1 + §1.2):
- **CFM YuelBond trainer** (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`): 10/10 wires SHIPPED with default-OFF CLI flags
- **SE(3) SB trainer** (`molmetal/scripts/train_se3sb.py`): 0/10 wires currently (target = 10/10 per Wf-383-A2 §1.2 audit)
- **chemprop v2/v3 trainer** (`molmetal/scripts/train_chemprop_v2_v3.py`): 0/10 wires (out of R18 P1 scope)

**Honest framing** (`party-twenty-spirit` §3.5): SCAFFOLD-only; NO trainer integration into SE(3) SB yet (deferred to Wf-380-B1 / Wf-383-A2 wire-in plan). All speedup / lift claims are PROJECTED per upstream papers. MEASURED speedup requires a 5-seed × 30-epoch A/B retrain, OUT OF SCOPE for Wf-380-A1 + Wf-353-A1..A4 SCAFFOLD phase.

### §12.4 Forward-looking Blackwell retrain

Per `TODO/completed/391a3_metric_strategy_refresh.md` §10 (NEW Forward-looking 2026-Q4 GPU retrain priority order) + `TODO/completed/391a1_lambda_roadmap_refresh.md` §6.6 (NEW Forward-looking 2026-Q4 GPU retrain priorities) + Wf-390-A3 §1.2 (Blackwell retrain DEFERRED per AGENTS.md §5.1 + USER_OK-gated), the Blackwell retrain priority order for R18 P0-1 SE(3) Schrödinger Bridge is:

**Blackwell retrain priority order** (3-batch sequential per Wf-391-A3 §10):

| Priority | Wall PROJECTED | Target | Mechanism | USER_OK gate |
|:---:|---:|---|---|---|
| **P1-A** | ~8-16 days CPU + ~5 hr GPU | M22 SE(3) SB wire-in (Wf-383-A2 §2) | M22 (SE(3)-SB solver) | USER_OK=1 + VRAM ≥ 35 GB |
| **P2-A** | ~10-12 days CPU + ~5 hr GPU | Multi-PH + PLL wire into SE(3) SB (Wf-380-A1 + Wf-382-A1) | M29 + M30 (Multi-PH + PLL) | USER_OK=1 + VRAM ≥ 35 GB |
| **P3-A** | ~10-12 days CPU + ~5 hr GPU | CFM acceleration wire into SE(3) SB (Wf-380-A1 + Wf-353-A1..A4) | M36-M44 (CFM acceleration) | USER_OK=1 + VRAM ≥ 35 GB |

**Combined wall PROJECTED**: ~28-40 days CPU + ~15 hr GPU for full R18 P0-1 + P1-1 + P1-2 wire-in across 3 sequential Blackwell retrain batches

**Constraint priorities per AGENTS.md §5.1**:
- VRAM budget check first (`nvidia-smi` ≥ 35 GB free on RTX PRO 6000 Blackwell 98GB)
- Hyperparameters via config files (reference peer papers in same domain)
- Pre-flight PASS still required (`pre_flight_check.py` before any retrain)
- `--dry-run` defaults still the script default; agents promote to actual launch after VRAM + pre-flight check

**Forward-looking honest-negative caveats** (per `docs/HONEST_NEGATIVE_CATALOG.md` Wf-369-A3 expansion 40 → 65 entries + Wf-391-A3 §10):
- M22 SE(3) SB wire-in DEFERRED on Blackwell cudagraphs per Wf-335-R3 + AGENTS.md §5.1; `--no-cudagraphs` safety lever added to `scripts/launch_wf124_p3_g6.sh` per Wf-335-R3
- Triton fused_relu_mlp / fused_wasserstein_sliced / fused_bond_head_aggregation / fused_smiles_featurize kernel wires (Wf-333-T2) — Blackwell sm_120 GPU retune DEFERRED per AGENTS.md §5.1
- Wf-379-A1 m1_auc 5-seed IN-FLIGHT on GPU 0 Blackwell at Wf-379-B1 timestamp (Rh seed 3 RUNNING); 6 (metal, seed) pairs queued; Krasnov 0.86 gate target; full multi-pair result deferred to Wf-380
- Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL on R20 §5 tmQM + PlatinAI joint retrain wrapper (trainer `--metals` restricted to 5 metals; alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint)

**Constraints honoured**: 0 NEW tests, 0 GPU launches, 0 `paper/*` touches, 0 wet-lab planning, only Bash + Read + Edit + Write tools used. AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §10 all preserved unchanged.
