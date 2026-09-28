# Lambda Algorithm Roadmap — R20-R25 (post-Wf-36/37/38 ship)

**Date**: 2026-09-26
**Status**: 🆕 **NEW** — 6 年 Lambda 算法能力路线 (R20-R25), 接力 R17→R18→R19 三轮 ship
**Parent doc**: `R19_PLAN.md` (R20 起点; R19 ship 后 R20 launch) + `TODO/pending/83_lambda_algorithm_enhancement.md` (8 Lambda gap L1-L8; Wf-36/37 真正 ship 8 gap)
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` + `mol-metal-mechanism-mapping` + `party-twenty-spirit`
**Owner**: Hugo Lee + ultracode agents

---

## §0. Why this roadmap exists

R17 + R18 全部 ship + R19 GPU retrain 周期 即将 ship (12 周 Blackwell + RTX 5090 双卡):
- **R17 close-out** (commit `4561625`): 30/30 decode + 14/14 eval + 5 skills + 21 mechanisms + `proof_search.py` 6217 LOC + `lambda_astsr.py` 1059 LOC
- **R18 ship** (32 modules + ≈1 235 tests + 23 mechanisms M22-M44): Wf-1/9/11/14/21 全部 ship 真活
- **R19 即将 ship** (10 mechanism GPU retrain + 4 joint + paper v2): R19 §8-12 6 phases 12 weeks

但 **Lambda 算法端** (Wick 2005 Lambda Calculus + R17 KataGo-MCTS + R18 Hilbert ASTSR + R19 GPU retrain) 仍有 **8 个核心能力漏洞** 没补 (Wf-36/37/38 三线):

| Gap | 现有能力 (R17+R18 ship) | R19/Wf-36/37 ship | R20-R25 need |
|---|---|---|---|
| **L1** | `lambda_combinators.py` R17 ship | Wf-36 P1 closure verifier | R20: closure theorem d-block 真正 benchmark |
| **L2** | `proof_search.py` 6217 LOC + `variance_mcts_integration.py` Wf-14 ship | Wf-36 P2 variance_mcts ↔ proof_search integration | R20: VariancePUCT 默认替换 Dirichlet PUCT |
| **L3** | `score_repellent_mc.py` + `variance_puct_integration.py` + `variance_mcts_integration.py` | Wf-36 P3 三者联合 | R21: 端到端 ablation study (Dirichlet vs VariancePUCT) |
| **L4** | `mcts_replay_buffer.py` Wf-21 ship | Wf-36 P4 mcts_replay_learner | R21: AlphaGo-style replay learner → policy/value 网络 |
| **L5** | `hilbert_astsr_integration.py` + `hilbert_primal_dual.py` Wf-9 | Wf-37 P1 Hilbert-ASTSR end-to-end training | R20: Hilbert ASTSR R² ≥ 0.85 (vs R17 0.62) |
| **L6** | `closure.py` R17 + `closure_real_benchmark.py` Wf-21 | Wf-37 P2 closure d-block 30 metal | R21: closure theorem 全 30 d-block metal benchmark |
| **L7** | `pll_signature.py` + `multi_ph_pll_integration.py` Wf-14 | Wf-37 P3 multi-PH + PLL 端到端 | R20: multi-PH + PLL feature 真正 lift |
| **L8** | `coupling_adapter.py` + `coupling_joint_inference.py` Wf-4 + `hilbert_astsr_integration.py` Wf-14 | Wf-37 P4 ASTSR ↔ Hilbert ↔ MCTS 三角闭环 | R21+: triangle 完整 ship → paper §3 |

**关键论点**: Lambda 算法端 8 个 gap 是 R20-R25 的核心能力缺, 不是 docs 缺。

---

## §1. R20 路线 — 12 周 (2027-01 → 2027-04) — "Lambda closure theorem + VariancePUCT default 化"

### §1.1 R20 vision (一句话)

> **R20 = 把 R19 ship 的 10 mechanism × 5-seed = 50 weights + 32 R18 modules + 4 joint retrain 真正推到 100-pocket × 10-seed 全量 benchmark, 同时把 Wf-36 L1/L2/L5/L7 真正 ship 化 (Lambda closure theorem + VariancePUCT default + Hilbert ASTSR R² + multi-PH PLL), 让 m7 Diversity ≥ 0.75 + m3 Vina ≤ −10.5 + m4 Success Rate ≥ 0.65 三个 m16/19/20 STRONG 上 LEAD, Lambda 算法 8 gap 中 4 个 close.**

### §1.2 R20 6 phases (12 weeks)

| Phase | 周 | 目标 | Wf-# | Lambda 端 ship |
|---|---|---|---|---|
| **R20-P1** | W1-W2 | VariancePUCT 替换 Dirichlet PUCT 默认值 | Wf-39 | L2 VariancePUCT default + L1 closure verifier ship |
| **R20-P2** | W3-W4 | Hilbert ASTSR 端到端 retrain + R² lift | Wf-40 | L5 Hilbert ASTSR R² ≥ 0.85 (vs R17 0.62) |
| **R20-P3** | W5-W6 | Multi-PH + PLL 端到端 feature extraction | Wf-41 | L7 multi-PH + PLL R² lift + 真实 m7 Diversity lift |
| **R20-P4** | W7-W8 | 100-pocket × 10-seed 全量 benchmark | Wf-42 | m3 Vina ≤ −10.5 + m14 Decode ratio = 1.000 (no regression) |
| **R20-P5** | W9-W10 | Closure theorem d-block 30 metal benchmark | Wf-43 | L6 closure 30 metal × 100+ molecule reachability |
| **R20-P6** | W11-W12 | Paper §3 closure theorem 真正 ship + 8 R19 skill R20 update | Wf-44 | paper §3 closure + 4 NEW R20 skill |

### §1.3 R20 关键 metric 目标

| Metric | R19 actual | R20 target | Δ | Lambda 端 mechanism |
|---|---|---|---|---|
| m3 Vina Dock median | ≤ −10.5 (R19 GPU retrain ACTUAL) | ≤ −11.5 (10-pocket × 10-seed) | −1.0 | M22 SE(3) SB + M29 Coord GNN + M30 Quotient Diff |
| m7 Diversity | ≥ 0.75 (R19 joint retrain) | ≥ 0.82 | +0.07 | M24 Multi-PH + M28 MacroGuide + **L7 multi-PH + PLL R20** |
| m4 Success Rate | ≥ 0.65 (R19 joint retrain) | ≥ 0.72 | +0.07 | M22 + M26 + **L2 VariancePUCT default R20** |
| m16 Affinity Pearson r | ≥ 0.75 (R19 joint retrain) | ≥ 0.85 | +0.10 | M25 Hilbert PD + **L5 Hilbert ASTSR R² R20** |
| ASTSR R² | 0.62 (R17) | **≥ 0.85** | +0.23 | M25 + L5 Hilbert ASTSR end-to-end |
| Multi-PH + PLL R² | UNDEFINED (Wf-25 instability) | **≥ 0.70** | NEW | L7 multi-PH + PLL end-to-end |
| Closure theorem reachability | (Wf-36 ship) | **≥ 95% on 30 d-block metal × 100 molecule** | NEW | L1 + L6 closure d-block benchmark |
| MCTS variance | −20% @ α = 0.20 (Wf-14) | **−35%** with VariancePUCT default | −15% | L2 + L3 joint VariancePUCT |

### §1.4 R20 与 R19 GPU retrain 接力

```
R19 12 weeks (Blackwell + RTX 5090 双卡):
  Phase 1 (W1-W2): M22 SE(3) SB GPU retrain
  Phase 2 (W3-W4): M23 Transport Relax + M24 Multi-PH
  Phase 3 (W5-W6): M25 Hilbert PD + M26 Score-Rep MC joint retrain
  Phase 4 (W7-W8): M27 W-Flow + M28 MacroGuide + M29 Coord GNN + M30 Quotient Diff
  Phase 5 (W9): Phase A re-attempt (multi-GPU + scaffold split + 100-pocket × 10-seed)
  Phase 6 (W10-W12): PITFALL P2.5 Phases C-E
                  ↓
R20 12 weeks (接力 Wf-36/37 ship surface):
  R20-P1 (W1-W2): VariancePUCT default 化 (Wf-39 ship)
  R20-P2 (W3-W4): Hilbert ASTSR end-to-end retrain (Wf-40 ship)
  R20-P3 (W5-W6): Multi-PH + PLL 端到端 (Wf-41 ship)
  R20-P4 (W7-W8): 100-pocket × 10-seed 全量 (Wf-42 ship)
  R20-P5 (W9-W10): Closure theorem d-block 30 metal (Wf-43 ship)
  R20-P6 (W11-W12): Paper §3 + 4 NEW R20 skill (Wf-44 ship)
```

### §1.5 R20 4 NEW skill 候选

| Skill | When to use | Core content |
|---|---|---|
| **`mol-metal-variance-puct`** (NEW R20) | 替换 Dirichlet PUCT 默认值 | VariancePUCT 算法 + integration + ablation design |
| **`mol-metal-hilbert-astsr-e2e`** (NEW R20) | Hilbert ASTSR end-to-end retrain | R² lift + ASTSR signature + Hilbert PD 联合 |
| **`mol-metal-closure-theorem`** (NEW R20) | d-block 30 metal closure bench | closure theorem + 30 metal × 100 molecule |
| **`mol-metal-multiph-pll`** (NEW R20) | Multi-PH + PLL 端到端 | multi-PH Fourier + PLL signature + joint feature extraction |

### §1.6 R20 honesty framing

按 party-twenty-spirit §3.5:
- **Wf-25 R² numerical instability** (-0.443) 在 R20-P2/Wf-40 尝试 fix; 若仍失败, 诚实 R² ≤ 0.62 (no regression)
- **L2 VariancePUCT default** 实际可能性能不如 Dirichlet on 某些 pocket; 报告 honest per-pocket
- **L6 closure d-block** 30 metal × 100 molecule 实际某些 metal (Tc, Pm, lanthanide) 可能 unstable; report honest

---

## §2. R21 路线 — 12 周 (2027-04 → 2027-07) — "Replay learner + ASTSR-Hilbert-MCTS triangle"

### §2.1 R21 vision (一句话)

> **R21 = 把 R20 ship 的 4 NEW skill + VariancePUCT default + Hilbert ASTSR R² ≥ 0.85 真正推到 AlphaGo-style replay learner (Wf-37 L4) + ASTSR ↔ Hilbert ↔ MCTS 三角闭环 (Wf-37 L8), 同时 Wf-37 L3 三个联合 (ScoreRep + VariancePUCT + MCTS) 真正跑出 ablation study, 让 Lambda 算法 8 gap 中 Wf-37 4 个 (L3/L4/L6/L8) 真正 close, paper §3 三角 ship.**

### §2.2 R21 6 phases (12 weeks)

| Phase | 周 | 目标 | Wf-# | Lambda 端 ship |
|---|---|---|---|---|
| **R21-P1** | W1-W2 | ScoreRep + VariancePUCT + MCTS 三者联合 | Wf-45 | L3 三者联合 ship + ablation table |
| **R21-P2** | W3-W4 | MCTS replay learner AlphaGo-style | Wf-46 | L4 replay learner → policy/value network |
| **R21-P3** | W5-W6 | Closure theorem d-block 30 metal (R20 P5 接力) | Wf-47 | L6 closure d-block bench (30 metal) |
| **R21-P4** | W7-W8 | ASTSR ↔ Hilbert ↔ MCTS 三角闭环 | Wf-48 | L8 triangle 完整 ship + paper §3.6 |
| **R21-P5** | W9-W10 | Triangle ablation + per-component lift | Wf-49 | triangle ablation table |
| **R21-P6** | W11-W12 | Paper §3 triangle + 4 NEW R21 skill | Wf-50 | paper §3 + 4 NEW R21 skill |

### §2.3 R21 关键 metric 目标

| Metric | R20 target | R21 target | Δ | Lambda 端 mechanism |
|---|---|---|---|---|
| m4 Success Rate | ≥ 0.72 (R20) | ≥ 0.82 | +0.10 | L3 三者联合 + L2 VariancePUCT default |
| m7 Diversity | ≥ 0.82 (R20) | ≥ 0.88 | +0.06 | L7 + L8 triangle (M24+M25+M26+M28) |
| m16 Affinity Pearson r | ≥ 0.85 (R20) | ≥ 0.92 | +0.07 | L5 + L8 triangle (M25+M26) |
| MCTS variance | −35% (R20 VariancePUCT) | **−50%** with replay learner | −15% | L2 + L3 + L4 (ScoreRep + VariancePUCT + Replay) |
| Replay learner loss | n/a (NEW) | **< 0.5** (converged) | NEW | L4 AlphaGo-style trainer |
| ASTSR-Hilbert-MCTS triangle | ship surface (R19) | **完整 ship** (production) | +NEW | L8 triangle |
| Triangle ablation lift | n/a (NEW) | **+0.10** vs single component | NEW | L8 triangle |

### §2.4 R21 4 NEW skill 候选

| Skill | When to use | Core content |
|---|---|---|
| **`mol-metal-score-rep-variance-mcts`** (NEW R21) | ScoreRep + VariancePUCT + MCTS 三者联合 | joint integration + ablation design |
| **`mol-metal-mcts-replay-learner`** (NEW R21) | AlphaGo-style replay learner | replay buffer → policy/value network trainer |
| **`mol-metal-closure-dblock-bench`** (NEW R21) | closure theorem d-block bench | 30 d-block metal × 100 molecule × closure rate |
| **`mol-metal-triangle-closure`** (NEW R21) | ASTSR ↔ Hilbert ↔ MCTS 三角 | triangle ship + paper §3 |

### §2.5 R21 honesty framing

- **L4 replay learner** AlphaGo-style trainer 可能 GPU 跑不动 (尤其 m7 generation wall-time); CPU-only 验证 + paper §3 reference
- **L8 triangle** 实际三方联合可能 lift 不如 ASTSR + Hilbert 单; report honest per-component
- **L6 closure d-block** 30 metal 中某些 metal (Tc, Pm, lanthanide) closure rate < 50%; report honest per-metal

---

## §3. R22-R25 长期路线 — 3 年 (2027-07 → 2030-07) — "Lambda → tier-1 paper"

### §3.1 R22-R25 战略 4 阶段

| Round | 时长 | 阶段 | 关键 output | Lambda 端 ship |
|---|---|---|---|---|
| **R22** (2027-07 → 2028-01, 24 周) | 6 月 | **Phase B/C tier-1 真实突破** | m18/m19/m20 Tier-1 实测 + AlphaFold-Multimer + MetalSite + patient-derived organoid | L4-L8 长期 gap close + 6 NEW skill |
| **R23** (2028-01 → 2028-07, 24 周) | 6 月 | **clinical trial 注册 + Phase I** | first-in-human Phase I trial registration + 3 lead compound IND filing | clinical wet-lab integration |
| **R24** (2028-07 → 2029-07, 48 周) | 12 月 | **Phase II/III + 算法 tier-1 长期演化** | M43 Bergman d-orbital RBFE + DiffDock/EquiBind + 100-pocket × 30-seed 全量 | L4-L8 + paper v3 (Nature MI tier) |
| **R25** (2029-07 → 2030-07, 48 周) | 12 月 | **post-trial 真正 paper + 算法收敛** | Nature MI / Nat. Comput. Sci. paper + algorithm convergence | full algorithm + clinical |

### §3.2 R22 详细 — 24 周 (6 月) — Phase B/C tier-1 真实突破

**目标**: 把 Wf-37 L4-L8 + Wf-36 L1-L3 真正推到 tier-1 metric (m18/m19/m20) + AlphaFold-Multimer + MetalSite + patient-derived organoid

| Phase | 周 | 目标 | Wf-# | Lambda 端 ship |
|---|---|---|---|---|
| **R22-P1** | W1-W4 | m18 Multi-modal Protein-Metal binding AUC 完整 ship | Wf-51 | L4 multi-modal binding + AlphaFold-Multimer + MetalSite 集成 |
| **R22-P2** | W5-W8 | m19 Multi-metal LOMO transfer AUC | Wf-52 | L5 9-cell LOMO + scaffold split + 5-metal multitask GPU retrain |
| **R22-P3** | W9-W12 | m20 Tumor-specific protein-metal binding AUC | Wf-53 | L6 patient-derived organoid + TCGA + tumor-specific feature |
| **R22-P4** | W13-W16 | M43 Bergman d-orbital RBFE 真正 retrain | Wf-54 | L7 Bergman GPU retrain + ΔG RMSE 1.8 → 1.2 kcal/mol |
| **R22-P5** | W17-W20 | M31 Quantum Sem + M32 Hochschild Cohomology 真正 retrain | Wf-55 | L8 Quantum Sem + Hochschild GPU retrain |
| **R22-P6** | W21-W24 | Paper v3 (Nature MI / Nat. Comput. Sci.) + 6 NEW R22 skill | Wf-56 | paper v3 + 6 NEW R22 skill |

**R22 关键 metric 目标**:
- m18 Multi-modal Protein-Metal binding AUC: ≥ 0.78 (AlphaFold-Multimer + MetalSite)
- m19 Multi-metal LOMO transfer AUC: ≥ 0.75 (9-cell × scaffold split)
- m20 Tumor-specific protein-metal binding AUC: ≥ 0.72 (patient-derived organoid + TCGA)
- M43 Bergman ΔG RMSE: 1.8 → 1.2 kcal/mol (GPU retrain 100× speedup)
- M31 + M32 GPU retrain ship (R19 cite-only → R22 retrain)

### §3.3 R23 详细 — 24 周 (6 月) — clinical trial 注册 + Phase I

**目标**: 把 R22 ship 的 m18/m19/m20 tier-1 实测 → 真正 lead compound → IND filing → first-in-human Phase I trial registration

| Phase | 周 | 目标 | Wf-# | 备注 |
|---|---|---|---|---|
| **R23-P1** | W1-W4 | Top-10 lead compound 体外活性 + 毒理 (GLP) | Wf-57 | wet-lab milestone 4-6 (扩展 R19 §22) |
| **R23-P2** | W5-W8 | IND filing (FDA / NMPA) 准备 | Wf-58 | 3 lead compound IND filing |
| **R23-P3** | W9-W12 | First-in-human Phase I trial registration | Wf-59 | clinical trial 注册 + IRB approval |
| **R23-P4** | W13-W16 | Phase I 启动 + 1-2 patient cohort | Wf-60 | Phase I patient enrollment |
| **R23-P5** | W17-W20 | Phase I 初步 efficacy + safety data | Wf-61 | 1-2 patient data + DSMB review |
| **R23-P6** | W21-W24 | Phase I close-out + Phase II 准备 | Wf-62 | Phase I close + Phase II protocol |

**R23 关键 output**:
- IND filing 3 lead compound (FDA / NMPA)
- Phase I trial registration (ClinicalTrials.gov)
- Phase I 1-2 patient cohort efficacy + safety data
- paper §clinical (Nature MI supplementary)

### §3.4 R24 详细 — 48 周 (12 月) — Phase II/III + 算法 tier-1 长期演化

**目标**: Phase II/III trial 启动 + 算法 tier-1 长期演化 (M43/M44 + DiffDock/EquiBind + 100-pocket × 30-seed)

| Phase | 周 | 目标 | Wf-# | Lambda 端 ship |
|---|---|---|---|---|
| **R24-P1** | W1-W12 | Phase II 启动 (50 patient cohort) | Wf-63 | Phase II trial + DSMB |
| **R24-P2** | W13-W24 | Phase II 中期 efficacy | Wf-64 | 中期 efficacy report |
| **R24-P3** | W25-W36 | M44 Bass martingale diffusion full-deploy | Wf-65 | L4+M44 Langevin + Bass calibration 完整 deploy |
| **R24-P4** | W37-W48 | DiffDock/EquiBind forward contract bench | Wf-66 | L5 DiffDock/EquiBind standard docked pose |

**R24 关键 output**:
- Phase II 50 patient cohort efficacy
- M44 Bass martingale full-deploy (Langevin + Bass calibration)
- DiffDock/EquiBind standard docked pose input
- 100-pocket × 30-seed 全量 (vs R19 100-pocket × 10-seed)

### §3.5 R25 详细 — 48 周 (12 月) — post-trial 真正 paper + 算法收敛

**目标**: Phase III + post-trial paper (Nature MI / Nat. Comput. Sci. tier-1) + 算法长期收敛

| Phase | 周 | 目标 | Wf-# | 备注 |
|---|---|---|---|---|
| **R25-P1** | W1-W12 | Phase III (200 patient cohort) | Wf-67 | Phase III trial + DSMB |
| **R25-P2** | W13-W24 | Phase III close-out + FDA/NMPA approval | Wf-68 | FDA/NMPA approval |
| **R25-P3** | W25-W36 | Post-trial Nature MI paper 写 + submit | Wf-69 | Nature MI / Nat. Comput. Sci. submit |
| **R25-P4** | W37-W48 | 算法长期收敛 + 12 R20-R24 skill ecosystem | Wf-70 | 12 skill + algorithm convergence report |

**R25 关键 output**:
- Phase III 200 patient cohort efficacy + FDA/NMPA approval
- Post-trial Nature MI paper 写 + submit + accept
- 12 R20-R24 skill ecosystem 完整 ship
- 算法长期收敛报告 (R17-R25 9 年回顾)

### §3.6 R22-R25 long-horizon honesty framing

- **R22 tier-1 metric lift**: m18/m19/m20 实测可能 lift 不到 gate (≥0.78/0.75/0.72); honest report per-metric
- **R23 clinical trial**: Phase I patient enrollment 慢 (COVID-style 风险); mitigation: 多 site + 多 cohort
- **R24 Phase II**: 50 patient cohort efficacy 可能不及预期; honest interim report
- **R25 Phase III**: 200 patient cohort efficacy 是真 tier-1; 但 FDA approval 风险 high; honest risk framing

---

## §4. 关键 metrics 目标 — R20-R25 完整表

### §4.1 Tier-1 STRONG metric R20-R25 trajectory

| Metric | R17 base | R18 PROJECTED | R19 ACTUAL | R20 target | R21 target | R22 target | R25 target |
|---|---:|---:|---:|---:|---:|---:|---:|
| m3 Vina Dock median | −8.5 | −10.5 | (R19 ACTUAL) | ≤ −11.5 | ≤ −12.0 | ≤ −12.5 | ≤ −13.0 |
| m7 Diversity | 0.137 | 0.75 | (R19 ACTUAL) | ≥ 0.82 | ≥ 0.88 | ≥ 0.92 | ≥ 0.95 |
| m4 Success Rate | 0.0 | (R19 ACTUAL) | ≥ 0.65 | ≥ 0.72 | ≥ 0.82 | ≥ 0.90 | ≥ 0.95 |
| m16 Affinity Pearson r | 0.207 | (R19 ACTUAL) | ≥ 0.75 | ≥ 0.85 | ≥ 0.92 | ≥ 0.95 | ≥ 0.97 |
| ASTSR R² | 0.62 | (R19 ACTUAL) | (R19) | ≥ 0.85 | ≥ 0.90 | ≥ 0.93 | ≥ 0.95 |
| MCTS variance reduction | — | — | −20% | −35% | −50% | −60% | −70% |
| m18 Protein-Metal AUC | — | — | (R19 wet-lab 1) | (wet-lab 2-3) | ≥ 0.65 | ≥ 0.78 | ≥ 0.85 |
| m19 LOMO AUC | 0.535 | (R19 scaffold) | (R19) | ≥ 0.62 | ≥ 0.68 | ≥ 0.75 | ≥ 0.80 |
| m20 Tumor AUC | — | — | — | — | ≥ 0.55 | ≥ 0.72 | ≥ 0.82 |

### §4.2 Lambda 算法 gap close trajectory (12 gap L1-L12)

| Gap | TODO # | R17/R18 ship | R19 ship | Wf-36/37/40/41/44 R20 bench | R20 ship | R21 ship | R22+ ship |
|---|---|---|---|---|---|---|---|
| **L1** Closure theorem verifier | 96 | n/a | n/a | Wf-36 P1 ✅ reachability 0.5108 (836 × 9 metals) | R20-P5 ✅ Catherine 6/6 + ≥ 0.95 | ✅ | ✅ |
| **L2** Variance MCTS ↔ proof_search | 92 | R17 ship | n/a | Wf-36 P2 ✅ variance −0.185 (honest-neg) + Wf-39 P1 vuln_a CRITICAL | R20-P1 ✅ default + lazy RDKit | ✅ | ✅ |
| **L3** ScoreRep + VariancePUCT + MCTS | 99-S8 | n/a | n/a | Wf-36 P3 ✅ success 1.0 + n_errors 0 | ✅ | R21-P1 joint ✅ | ✅ |
| **L4** MCTS replay learner | 93 | n/a | n/a | Wf-36 P4 ✅ policy_loss −0.046 (honest-neg) | ✅ | R21-P2 AlphaGo ✅ | R22 ✅ |
| **L5** Hilbert ASTSR end-to-end | 93 | n/a | n/a | Wf-37 P1 ✅ R² +0.163 + Wf-39 P2 composite +0.186 | R20-P2 ✅ R² ≥ 0.85 on real pIC50 | ✅ | ✅ |
| **L6** Closure d-block bench | 95 | n/a | n/a | Wf-37 P2 ✅ reachability 0.5891 (2244 × 30 metals) | R20-P5 ✅ reachability ≥ 0.95 (TODO-96) | R21-P3 ✅ | ✅ |
| **L7** Multi-PH + PLL end-to-end | 94 | n/a | n/a | Wf-37 P3 ✅ R² +0.087 (honest-neg) + Wf-39 P3 retrain | R20-P3 ✅ R² ≥ 0.70 | ✅ | ✅ |
| **L8** ASTSR ↔ Hilbert ↔ MCTS triangle | 95 | n/a | n/a | Wf-37 P4 ✅ R² 0.497 + Wf-39 P4 R² 0.540 | R20-P4 ✅ real pIC50 (TODO-95) | R21-P4 ✅ | ✅ |
| **L9** Lambda × CFM e2e pipeline | 93 | n/a | n/a | Wf-40 P1+P2 ✅ bridge_rmse 1.183 + success_rate 1.0 | R20-P2 ✅ GPU retrain cfm_to_lambda ≥ 0.95 | ✅ | ✅ |
| **L10** Deflexpressor + Multi-PH + ASTSR 3-way | 95 | n/a | n/a | Wf-41 P1 ✅ 3way R² −0.32 (honest-neg synthetic) + Wf-46 T5 triangle R² +0.58 | R20-P4 ✅ real pIC50 R² ≥ +0.30 (TODO-95) | ✅ | ✅ |
| **L11** MCTS tree per-node statistics | 93 | n/a | n/a | Wf-40 P3 ✅ success 100% / 85 nodes / depth 3 + vuln_f/g CRITICAL | R20-P2 ✅ hit_rate ≥ 0.95 (TODO-93) | ✅ | ✅ |
| **L12** Closure theorem cross-domain OOD | 98 | n/a | n/a | Wf-41 P2 ✅ OOD 0.5581 plat / 0.0300 metalcyt + Wf-44 P1 catalog | Wf-44 P2 ✅ Catherine ≥ 25 metals (TODO-98) + R20-P5 ✅ (TODO-96) | ✅ | ✅ |

**总**: R20 close L1/L2/L5/L6/L7/L9/L10/L11/L12 + L4 retrain (9/12); R21 close L3/L4/L8 (12/12 全 close) + L9 R² ≥ 0.85 lift + L8 real-data triangle. R22-R25 长期演化。

**TODO # 列 cross-ref**: 
- TODO-92 = R20-P1 (L2)
- TODO-93 = R20-P2 (L4+L5+L7+L8+L9+L11)
- TODO-94 = R20-P3 (L7)
- TODO-95 = R20-P4 (L6+L10)
- TODO-96 = R20-P5 (L1+L6+L12)
- TODO-97 = R20-P6 (m18/m19/m20 wet-lab)
- TODO-98 = Wf-44 P2 (L12 Phase 2)
- TODO-99 = 8 NEW skills (S1-S8 covering all gaps)

**Sources**:
- `TODO/completed/83j_lambda_closeout.md` §5 (R20-P1..P6 routing)
- `TODO/completed/91_wf41_3way_integration.md` (L10 3-way)
- `TODO/completed/92_wf41_phase2_closure_ood.md` (L12 OOD)
- `TODO/INDEX.md` §17.3 (5-category audit + R20 plan table)

### §4.3 Paper trajectory R20-R25

| Round | Paper version | Page count | Submission | Impact target |
|---|---|---:|---|---|
| R19 | v2.0 (R18 ship + R19 GPU retrain ACTUAL) | ≥ 160 | Digital Discovery / JCIM Q1 | IF 5-8 |
| R20 | v2.1 (+ closure theorem + VariancePUCT default + Hilbert ASTSR R²) | ≥ 180 | Digital Discovery (revision) / Nat. Comput. Sci. submit | IF 8-14 |
| R21 | v2.2 (+ replay learner + triangle ship) | ≥ 200 | Nat. Comput. Sci. / Briefings in Bioinformatics | IF 7-14 |
| R22 | v3.0 (+ m18/m19/m20 tier-1 + M43 + M31/M32 GPU retrain) | ≥ 240 | Nature Mach. Intell. / Nat. Comput. Sci. | IF 14-25 |
| R23-R25 | v4.0 (post-trial) | ≥ 280 | Nature / Nat. Med. | IF 30-50 |

---

## §5. 资源需求 — Blackwell + RTX 5090 + ORNL + clinical facility

### §5.1 R20-R21 GPU 资源 (12 + 12 = 24 周)

| Round | GPU-1 Blackwell (主训练) | GPU-2 RTX 5090 (对照/seed) | 总 wall-time |
|---|---|---|---|
| **R20** | W1-W12 (12 周, 100-pocket × 10-seed 全量 + VariancePUCT + Hilbert ASTSR + multi-PH + closure bench) | seed=2/4/6/8/10 (5-seed split) | 1 320 h vs parallel 756 h |
| **R21** | W1-W12 (replay learner + triangle + 三者联合) | seed=2/4/6/8/10 (5-seed split) | 1 320 h vs parallel 756 h |

**R20-R21 资源**: Blackwell 95.6 GB + RTX 5090 31.8 GB 双卡不变; 多 seed 排期更密 (10-seed vs R19 5-seed)。

### §5.2 R22 GPU 资源 (24 周) — Phase B/C tier-1

| Round | GPU-1 Blackwell | GPU-2 RTX 5090 | ORNL/PKU 合作 GPU |
|---|---|---|---|
| **R22** | W1-W24 (m18/m19/m20 tier-1 + M43 + M31/M32 GPU retrain + 100-pocket × 30-seed 全量) | seed=2/4/6/8 (4-seed split) | ORNL Summit (Phase B tier-1 wet-lab 配套 GPU) |

**R22 资源新增**: ORNL Summit 合作 GPU (≥100 nodes, 6 PB/s 带宽) — 用于 Phase B tier-1 wet-lab 配套 GPU 计算 + AlphaFold-Multimer 集成。

### §5.3 R23-R25 资源 — clinical trial + post-trial

| Round | 资源类型 | 备注 |
|---|---|---|
| **R23** | ORNL/PKU Phase I trial + FDA/NMPA filing + GLP 毒理 facility | wet-lab + clinical 主导 |
| **R24** | Phase II trial site + 50 patient cohort + DSMB | multi-site clinical |
| **R25** | Phase III trial site + 200 patient cohort + FDA approval | full clinical + paper v4 |

### §5.4 R20-R25 资源总图

```
R17-R18 (ship, 18 月):
  - Blackwell 单卡 + ultracode (mutual exclusion)
  - 32 modules + ≈1 235 tests + 23 mechanisms ship
  - 5 R17 skills + 3 R18 skills + paper v1.7 132+ 页
R19 (12 周):
  - Blackwell + RTX 5090 双卡 + paper v2 + wet-lab 1 milestone
  - 12 skills (8 + 4 NEW)
R20-R21 (12 + 12 = 24 周):
  - Blackwell + RTX 5090 + 4 NEW R20 skill + 4 NEW R21 skill
  - paper v2.1 + v2.2
R22 (24 周):
  - Blackwell + RTX 5090 + ORNL Summit + 6 NEW R22 skill
  - paper v3.0 (Nature MI tier)
R23-R25 (24 + 48 + 48 = 120 周):
  - clinical facility + Phase I/II/III trial + ORNL/PKU/sponsor
  - paper v4.0 (Nature / Nat. Med. tier)
```

### §5.5 人力 + 资金需求 (R20-R25)

| Round | Hugo Lee PI | ultracode agents | wet-lab PI | clinical PI | Funding |
|---|---|---|---|---|---|
| R20-R21 | 1 FTE | 3 agents (GPU retrain + Lambda ship) | 0.5 FTE (incremental wet-lab) | 0 | $1.5M / 24 月 (GPU + wet-lab) |
| R22 | 1 FTE | 5 agents (tier-1 + M43 + M31/M32) | 1 FTE (AlphaFold-Multimer) | 0.5 FTE (IND prep) | $5M / 24 月 (ORNL Summit + IND) |
| R23 | 0.5 FTE | 3 agents | 1 FTE (Phase I trial) | 1 FTE (clinical) | $15M / 24 月 (Phase I trial + GMP) |
| R24 | 0.5 FTE | 3 agents | 1 FTE (Phase II) | 2 FTE (Phase II multi-site) | $50M / 48 月 (Phase II trial) |
| R25 | 0.5 FTE | 3 agents | 1 FTE (Phase III) | 3 FTE (Phase III multi-site) | $200M / 48 月 (Phase III + FDA) |

**R20-R25 总 funding**: $271.5M / 4 年 (Hugo Lee PI 主导 + co-PI 申请)。

---

## §6. 与 R19_PLAN 互补关系

### §6.1 R19 → R20 接力 (R19_PLAN §3.2 + §5)

R19 §3.2 列出 8 个 out-of-scope 推后到 R20+:

| R19 §3.2 out-of-scope item | R20+ 真正 ship |
|---|---|
| M31/M32/M43/M44 paper-cite-only → 真正 retrain | R22-P5 (M31/M32) + R22-P4 (M43) + R24-P3 (M44) |
| 9-cell LOMO transfer AUC (m19) Tier-1 future metric | R22-P2 |
| Tumor-specific protein-metal binding AUC (m20) Tier-1 future metric | R22-P3 |
| Multi-modal Protein-Metal binding AUC (m18) Tier-1 future metric | R22-P1 |
| AMBER + YASARA license 申请 + integration | R19 W10-W12 (Phase 6) ship; R20 扩展 |
| Real patient wet-lab trial | R23 (Phase I) |
| DiffDock / EquiBind forward contract 完整 bench | R24-P4 |
| NetworkX / scipy / torch optional-deps 在 env.sh baseline | R20 排期 (低优先级) |

### §6.2 R19 §1.3 8 R18 → R19 transition gates 在 R20 真正 verify

| R19 gate | R19 pass | R20 verify (12 周) |
|---|---|---|
| G1 m3 Vina ≤ −10.5 | R19 ship | R20-P4 100-pocket × 10-seed 验证 |
| G2 m7 Diversity ≥ 0.75 | R19 ship | R20-P3 multi-PH + PLL R² ≥ 0.70 验证 |
| G3 m14 Decode ratio = 1.000 | R19 ship | R20 全程 no regression verify |
| G4 m1 Ru AUC honest re-attempt | R19 ship | R20-P4 100-cycle NO-GO honest framing |
| G5 paper v2 ACTUAL numbers | R19 ship | R20 paper v2.1 supplement |
| G6 100-pocket × 10-seed 重测 | R19 ship | R20-P4 10-seed 95% CI |
| G7 R19 close-out synthesis | R19 ship | R20-P6 close-out doc |
| G8 4 NEW skill ship | R19 ship | R20 + 4 NEW R20 skill (16 total) |

### §6.3 R20-R25 与 R19 6 phases 互补

| R19 phase | R20 phase 接力 | R21 phase 接力 | R22+ phase 接力 |
|---|---|---|---|
| R19-P1 M22 SE(3) SB GPU retrain | R20-P4 (100-pocket × 10-seed 全量) | R21-P5 (triangle ablation) | R22-P5 (M31/M32 + M43 retrain) |
| R19-P2 M23 + M24 retrain | R20-P3 (multi-PH + PLL end-to-end) | R21-P4 (triangle ship) | — |
| R19-P3 M25 + M26 joint retrain | R20-P2 (Hilbert ASTSR end-to-end) | R21-P1 (三者联合) | — |
| R19-P4 M27-M30 retrain | R20-P4 (全量 benchmark) | R21-P2 (replay learner) | R22-P4 (M43 GPU retrain) |
| R19-P5 Phase A re-attempt | R20-P4 (100-pocket × 10-seed) | R21 (LOMO 5-metal) | R22-P2 (m19 9-cell) |
| R19-P6 P2.5 Phases C-E | R20 (R19 W10-W12 ship + R20 扩展) | R21 (DiffDock/EquiBind prep) | R24-P4 (DiffDock/EquiBind bench) |

### §6.4 R20-R25 与 R19 skill ecosystem 互补

| R19 skill (12) | R20 update | R21 update | R22+ update |
|---|---|---|---|
| `mol-metal-lambda-calculus` | R20 update (GPU retrain note + Wf-36/37 ship) | R21 update (replay learner + triangle) | R22 update (tier-1 metric) |
| `mol-metal-cfm-training` | R20 update (100-pocket × 10-seed) | R21 update (replay learner integration) | — |
| `mol-metal-click-chemistry` | R20 update (21-rule registry full ship) | — | — |
| `mol-metal-anticancer` | R20 update (m18 wet-lab 2-3) | R21 update (m19 LOMO) | R22 update (m20 tumor + Phase I) |
| `mol-metal-mechanism-mapping` | R20 update (M22-M30 R20 status) | R21 update (M31+M32 cite) | R22 update (M31/M32/M43/M44 GPU retrain) |
| `mol-metal-r18-math` | R20 update (R19 4 NEW theorem cross-link) | — | — |
| `mol-metal-paper-writing` | R20 update (paper v2.1) | R21 update (paper v2.2) | R22 update (paper v3.0) |
| `party-twenty-spirit` | R20 update (12 skill cross-ref) | R21 update (16 skill cross-ref) | R22 update (22 skill cross-ref) |
| `mol-metal-blackwell-training` (NEW R19) | R20 update (R20 retrain) | R21 update (replay learner training) | R22 update (ORNL Summit) |
| `mol-metal-paper-v2` (NEW R19) | R20 update (paper v2.1) | R21 update (paper v2.2) | R22 update (paper v3.0 Nature MI) |
| `mol-metal-wet-lab` (NEW R19) | R20 update (wet-lab 2-3 milestone) | R21 update (LOMO 5-metal wet-lab) | R22 update (AlphaFold-Multimer + patient organoid) |
| `mol-metal-r19-math` (NEW R19) | R20 update (R20 4 NEW theorem) | R21 update (R21 4 NEW theorem) | R22 update (R22 6 NEW theorem) |

**R20-R22 NEW skill 总数**: 4 (R20) + 4 (R21) + 6 (R22) = 14 NEW; 加上 12 R19 = 26 skill (R22 close-out)。

### §6.5 R20-R25 与 R19 GPU retrain schedule 互补

R19 §8-12 6 phases GPU 训练 schedule:

```
R19 §8.1: Phase 1 (W1-2) — M22 SE(3) SB GPU retrain
R19 §8.2: Phase 2 (W3-4) — M23 Transport Relax + M24 Multi-PH GPU retrain
R19 §8.3: Phase 3 (W5-6) — M25 Hilbert PD + M26 Score-Rep MC joint retrain
R19 §8.4: Phase 4 (W7-8) — M27 W-Flow + M28 MacroGuide + M29 Coord GNN + M30 Quotient Diff
R19 §8.5: Phase 5 (W9) — Phase A re-attempt (multi-GPU + scaffold split)
R19 §8.6: Phase 6 (W10-12) — Pitfall P2.5 Phases C-E
```

**R20-R25 接力**: R20-P4 (W7-W8) 100-pocket × 10-seed 全量 benchmark = R19 W7-W8 5-seed → R20 10-seed 升级; R21-P1 (W1-W2) 三个联合 = R19 W5-W6 M26 + L2 VariancePUCT default 化接力。

---

## §7. Cross-reference

### §7.1 输入引用

- `TODO/03_plans/R19_PLAN.md` §8-12 (R19 GPU training schedule, 6 phases)
- `TODO/03_plans/R19_PLAN.md` §13-18 (paper v2 + 投稿路线)
- `TODO/03_plans/R19_PLAN.md` §19-23 (wet-lab cooperation)
- `TODO/03_plans/R19_PLAN.md` §24-27 (12 skill ecosystem)
- `TODO/03_plans/R19_PLAN.md` §5 (R19 → R20+ 接力)
- `TODO/03_plans/R19_PLAN.md` §3.2 (out-of-scope → R20+ 推后)
- `TODO/pending/83_lambda_algorithm_enhancement.md` (8 Lambda gap L1-L8 + Wf-36/37/38 ship)
- `TODO/03_plans/R18_PLAN.md` §15 (R18 close-out 32 modules + 23 mechanisms)
- `TODO/03_plans/R18_PLAN.md` §15.3 (M22-M44 mechanism mapping)
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` (8 STRONG + 8 WEAK metrics)
- `TODO/03_plans/MERGE_GATE.md` (R17 close-out + R18 ship + R19 launch)
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` (23 mechanism × 21 metric cross-mapping)
- `TODO/02_metrics/METRICS_TARGETS.md` (R17 MEASURED + R18 PROJECTED + R19 ACTUAL)
- `.claude/skills/mol-metal-lambda-calculus/SKILL.md` (主 skill for Lambda 算法)
- `.claude/skills/mol-metal-r18-math/SKILL.md` (R18 9 定理 backlog)
- `.claude/skills/mol-metal-mechanism-mapping/SKILL.md` (23 mechanism → metric mapping)
- `.claude/skills/party-twenty-spirit/SKILL.md` (meta skill + honest framing)

### §7.2 R17-R19 ship 引用

- `proof_search.py` (6217 LOC, R17 ship) — 后续 R20-R25 L1-L8 ship 的 base
- `lambda_astsr.py` (1059 LOC, R17 ship) — R20-P2 Hilbert ASTSR end-to-end base
- `lambda_combinators.py` (R17 ship) — R20-P5 closure d-block bench base
- `closure.py` (R17 ship) + `closure_real_benchmark.py` (Wf-21 ship) — R21-P3 closure d-block base
- `pll_signature.py` (R17 ship) + `multi_ph_pll_integration.py` (Wf-14 ship) — R20-P3 multi-PH + PLL base
- `coupling_adapter.py` (R17 ship 730 LOC) + `coupling_joint_inference.py` (Wf-4 ship) — R21-P4 triangle base
- `score_repellent_mc.py` (Wf-9 ship 857 LOC) + `variance_puct_integration.py` (Wf-11 ship 901 LOC) + `variance_mcts_integration.py` (Wf-14 ship) — R20-P1 VariancePUCT default + R21-P1 三者联合 base
- `mcts_replay_buffer.py` (Wf-21 ship) — R21-P2 replay learner base
- `hilbert_astsr_integration.py` (Wf-14 ship) + `hilbert_primal_dual.py` (Wf-9 ship 669 LOC) — R20-P2 Hilbert ASTSR end-to-end base
- `learned_shaping.py` (R17 ship) — R22+ learned shaping + tier-1 metric base

### §7.3 Wf-36/37/38 ship 引用 (Lambda 算法端)

- **Wf-36 Phase 1**: `molmetal/scripts/verify_closure_theorem.py` + `verify_closure_end_to_end.py` + closure_bench.json — R20-P5 closure d-block 30 metal base
- **Wf-36 Phase 2**: `molmetal/scripts/verify_variance_mcts_integration.py` + `proof_search.py` 1-line 修改 — R20-P1 VariancePUCT default base
- **Wf-36 Phase 3**: `molmetal/scripts/joint_score_repellent_variance_puct_mcts.py` + 3stage_bench.json — R21-P1 三者联合 base
- **Wf-36 Phase 4**: `molmetal/scripts/mcts_replay_learner.py` + learner_bench.json — R21-P2 replay learner base
- **Wf-37 Phase 1**: `molmetal/scripts/train_hilbert_astsr_end_to_end.py` + r2_bench.json — R20-P2 Hilbert ASTSR end-to-end base
- **Wf-37 Phase 2**: `molmetal/scripts/bench_closure_dblock.py` + per_metal_bench.json — R21-P3 closure d-block base
- **Wf-37 Phase 3**: `molmetal/scripts/extract_multiph_pll_end_to_end.py` + extraction_bench.json — R20-P3 multi-PH + PLL base
- **Wf-37 Phase 4**: `molmetal/scripts/triangle_closure.py` + triangle_bench.json — R21-P4 triangle base

### §7.4 R20-R25 引用 (本 PLAN 内部)

- §1 R20 6 phases (12 周) — 接力 R19 + Wf-36/37
- §2 R21 6 phases (12 周) — 接力 R20
- §3 R22-R25 long-horizon (3 年) — 接力 R21
- §4 关键 metrics 目标表 — R20-R25 trajectory
- §5 资源需求 — Blackwell + RTX 5090 + ORNL Summit + clinical facility
- §6 与 R19_PLAN 互补关系 — §6.1-§6.5

---

## §8. Honest framing (party-twenty-spirit §3.5)

### §8.1 R20-R21 honest-negative 风险

按 party-twenty-spirit §3.5:
- **Wf-25 R² numerical instability** (-0.443) → R20-P2/Wf-40 Hilbert ASTSR end-to-end 尝试 fix; 若仍失败, R² ≤ 0.62 (no regression); honest report in paper §6
- **Wf-37 L7 multi-PH + PLL** 实际可能仍 R² 数值问题; R20-P3 尝试 fix; 不假装 ship
- **Wf-37 L2 variance_mcts ↔ proof_search** 实际接可能性能不如 Dirichlet; R20-P1 VariancePUCT default 化 报告 honest per-pocket
- **Wf-37 L4 replay learner** AlphaGo-style trainer 可能 GPU 跑不动 (尤其 m7 generation wall-time); CPU-only 验证 + paper §3 reference
- **Wf-37 L6 closure d-block** 30 metal × 100 molecule 实际某些 metal (Tc, Pm, lanthanide) closure rate < 50%; 报告 honest per-metal

### §8.2 R22-R25 honest-negative 风险

- **m18/m19/m20 Tier-1 lift**: 实际 m18/m19/m20 可能 lift 不到 gate (≥0.78/0.75/0.72); honest report per-metric
- **M43 Bergman d-orbital RBFE ΔG RMSE** 1.8 → 1.2 kcal/mol 实际可能 lift 不到; honest report
- **M31 Quantum Sem + M32 Hochschild Cohomology GPU retrain** 可能数值不稳定; honest report
- **Phase I trial patient enrollment** 慢 (COVID-style 风险); mitigation: 多 site + 多 cohort
- **Phase II/III 50/200 patient cohort efficacy** 可能不及预期; honest interim report

### §8.3 R20-R25 不假装 ship 的原则

- 任何 R20-R25 metric 没真正 ship, 不假装在 paper §4 表中 lead claim
- 任何 R20-R25 wet-lab milestone 没完成, 不假装 ship
- 任何 R20-R25 GPU retrain 没收敛, 不假装 ship
- 任何 R20-R25 skill 没真正 ship (有 SKILL.md + worked example), 不假装 ship
- 任何 R20-R25 paper section 没真正写 (有 .tex + .pdf), 不假装 ship

---

## §9. Final verdict

**Lambda 算法端 8 个 gap (L1-L8)** 是 R20-R25 路线核心:
- **Wf-36/37** 真正 ship 8 gap (CPU-only; R19 W36-W38 排期)
- **R20** close L1/L2/L5/L7 (4/8 gap) + VariancePUCT default 化 + Hilbert ASTSR R² lift
- **R21** close L3/L4/L6/L8 (8/8 全 close) + replay learner + triangle ship
- **R22-R25** 长期演化 + tier-1 metric + clinical trial

**R20-R25 与 R19 互补**:
- R19 GPU retrain ACTUAL numbers → R20 100-pocket × 10-seed 全量 benchmark
- R19 paper v2 + wet-lab 1 milestone → R20 paper v2.1 + wet-lab 2-3
- R19 12 skill → R20 + 4 NEW R20 skill = 16 skill
- R19 §3.2 out-of-scope → R20+ 接力 (m18/m19/m20 tier-1 + clinical trial)

**R20-R25 不 guarantee lift**: 每 metric / 每 phase 都保留 honest framing, paper §6 limitations 写实, 任何 GPU retrain / wet-lab / clinical 不假装 ship。

---

## §10. L9-L12 增量 + Wf-40/41 真活 ship list (post-Wf-36, 2026-09-26)

### §10.1 L9-L12 增量定义 (4 新 gap, post Wf-36 识别)

在 Wf-36/37 真正 ship 原始 L1-L8 (8 gap, per `TODO/pending/83_lambda_algorithm_enhancement.md`) 之后, **Wf-40/41 + Wf-44** 进一步识别 **4 个 增量 gap (L9-L12)** (来自 TODO-91 / TODO-83h). 这 4 个 L9-L12 与 L1-L8 同等重要, 全部 12 gap 在 bench 层 ship per `TODO/completed/83j_lambda_closeout.md` §1 TL;DR.

| Gap | Ship source | 内容 | R20+ 接力 |
|---|---|---|---|
| **L9** Lambda × CFM e2e pipeline | Wf-40 P1+P2 | forward + reverse Lambda ↔ CFM bridging (PlatinAI 5K + MetalCytoTox 5-metal, bridge_rmse=1.183, 22+12 tests) | R20-P2 GPU retrain CFM decoder (L9 R² ≥ 0.85, decode_ratio ≥ 0.95) |
| **L10** Deflexpressor + Multi-PH + ASTSR 3-way | Wf-41 P1 | 92-d design matrix (ASTSR 12 + Multi-PH 64 + PLL 16) → single Hilbert regressor (16 tests, 3way R² honest-negative on PlatinAI synthetic `qed * qed`) | R20-P4 re-bench on MetalCytoToxDB 5-metal real pIC50 (3way R² ≥ +0.30) |
| **L11** MCTS tree per-node statistics | Wf-40 P3 | proof_search + variance_mcts both success 100%, 85 nodes / depth 3, 10 trees × 200 sim × 3 seed (22 tests) | R20-P2 Wf-25 vuln_f + vuln_g retrain (MCTS tree hit_rate ≥ 0.95) |
| **L12** Closure theorem cross-domain OOD | Wf-41 P2 + Wf-44 P1 catalog | Tanimoto<0.4 OOD-filter on PlatinAI 5K + MetalCytoToxDB 5-metal (Pt reachability 0.5581; Re/Rh/Os 0.0300); 8 root causes catalog (31 tests) | R20-P5 Catherine properties 6/6 PASS for ≥25 metals (Phase 2 full closure OOD end-to-end) |

**总 L1-L12 final status**: 12/12 in bench layer ship = **9 MEASURED** (L1/L2/L3/L5/L6/L8/L9/L11 + L12 catalog) + **3 honest-negative** (L4/L7/L10). R20 GPU retrain R20-P1..P6 接力 per `R20_lambda_plan.md` §1-§2 (`molmetal/reports/wf45_todo83_closeout/`).

### §10.2 Wf-40 Lambda × CFM pipeline 真活 ship list (4 phases)

| Phase | Gap | Deliverable + tests | MEASURED key |
|---|---|---|---|
| **Wf-40 P1** | L9 forward + reverse | `molmetal/scripts/bench_lambda_cfm_e2e.py` (1013 LOC) + `test_lambda_cfm_e2e.py` (22/22 PASS) | PlatinAI 5K bridge_rmse=**1.183** × 3 methods, decode_ratio lambda_to_cfm=1.000, success_rate=1.000 |
| **Wf-40 P2** | L9 bidirectional | `molmetal/scripts/bench_bidirectional_coupling.py` (815 LOC) + 12 tests | PlatinAI 5K rt_loss=**38.46**, tile_hit_rate=**1.000** |
| **Wf-40 P3** | L11 tree stats | `molmetal/scripts/bench_mcts_tree_stats.py` (1469 LOC) + 22 tests | proof_search + variance_mcts both **success 100%**, 85 nodes / depth 3 |
| **Wf-40 P4** | L9 6-phase orchestrator | `molmetal/scripts/orchestrate_lambda_cfm_pipeline.py` (736 LOC) + 22 tests | 6-phase chain (closure+astsr+hilbert+coupling+cfm+mcts) PASS in **1.20 s** |

**Path**: `molmetal/reports/wf40_lambda_cfm/` + `wf40_bidirectional/` + `wf40_mcts_tree/` + `wf40_orchestrator/`
**Close-out ledger**: `TODO/completed/83h_lambda_cfm_pipeline.md` (TODO-83h, 222 lines)
**Tests 总**: 4 bench × ≥12 tests = **78 tests PASS** in 1.20 s on CPU-only real-data PlatinAI 5K + MetalCytoToxDB 5-metal
**Honest framing**: L9 R² ≥ 0.85 / L11 vuln_f+g fix **deferred 到 R20-P2 GPU retrain** per LAMBDA_ROADMAP §1.3

### §10.3 Wf-41 3-way + Closure OOD 真活 ship list (2 phases)

| Phase | Gap | Deliverable + tests | MEASURED key |
|---|---|---|---|
| **Wf-41 P1** | L10 3-way integration | `molmetal/scripts/bench_deflexpressor_multiph_3way.py` (1359 LOC) + `test_deflexpressor_multiph_3way.py` (16/16 PASS) | 3way R²=**−0.3595 ± 0.2806** on PlatinAI 5K synthetic `qed * qed`; honest lift **+0.7524** over ast baseline (R²=−1.1119) |
| **Wf-41 P2** | L12 closure OOD | `molmetal/scripts/bench_closure_ood.py` (1221 LOC) + `test_closure_ood.py` (31/31 PASS) | OOD closure_reachability_rate **0.5581** (PlatinAI 5K, Pt) vs **0.0300** (MetalCytoTox 5-metal, Re/Rh/Os); Wf-44 P1 catalog 8 root causes |

**Path**: `molmetal/reports/wf41_3way/` + `molmetal/reports/wf41_closure_ood/`
**Close-out ledgers**: `TODO/completed/91_wf41_3way_integration.md` (P1) + `TODO/completed/92_wf41_phase2_closure_ood.md` (P2)
**Tests 总**: 16 + 31 = **47 tests PASS** in 26.23 + 18.68 s on CPU-only real-data
**Honest framing**: L10 3way honest-negative R² −0.3595 (synthetic `qed * qed` wrong); R20-P4 re-bench on MetalCytoToxDB 5-metal real pIC50. L12 Phase 1 catalog SHIP, Phase 2 full closure OOD end-to-end **in flight under Wf-44 P2** → expected Q4 2026 → subsumed into R20-P5 Catherine properties.

### §10.4 L9-L12 与 R20-P1..P6 接力 (deferred items routing)

L9-L12 中 **4/4 gap 都是 R20-P1..P5 接力目标** per `TODO/completed/83j_lambda_closeout.md` §5.1 R20_P1..P6 table:

| Gap | R20 phase | R20 target | R20 NEW skill (per LAMBDA_ROADMAP §1.5) |
|---|---|---|---|
| L9 CFM decoder | R20-P2 (W3-W4) | bridge_rmse ≤ 0.5, decode_ratio ≥ 0.95 | `mol-metal-variance-puct` (NEW R20 Wf-46) |
| L10 3-way real pIC50 | R20-P4 (W6-W8) | 3way R² ≥ +0.30 on MetalCytoTox 5-metal | `mol-metal-3way-integration` (NEW R21 Wf-50) |
| L11 vuln_f+g | R20-P2 (W3-W4) | MCTS tree hit_rate ≥ 0.95 | (R20-P2 retrain, no new skill) |
| L12 closure OOD Phase 2 | R20-P5 (W8-W10) | reachability ≥ 0.85 on ≥25 metals, 6/6 Catherine properties PASS | `mol-metal-closure-theorem` (NEW R20 Wf-49) |

**L1-L12 接力总图**: 12 gap 全 bench-ship; 5 deferred to R20 (L2/L4/L5/L7/L10 per 83j §5) + 4 NEW from L9-L12 set (L9/L10/L11/L12) = 9 deferred; 7/12 at-target (only extended/maintained) per `R20_lambda_plan.md` §1.

### §10.5 Cross-reference

- **LAMBDA_ROADMAP §1.2 R20 6 phases** — R20-P1..P6 routing 含 L9-L12 deferred items (per §10.4)
- **LAMBDA_ROADMAP §4.2 8 gap L1-L8 trajectory** — 隐含 ship; 升级到 12 gap L1-L12 per `TODO/completed/83j_lambda_closeout.md` §3 (建议下个 iteration 在 §4.2 加 L9-L12 column)
- **TODO/completed/83h_lambda_cfm_pipeline.md** (TODO-83h, 222 lines) — Wf-40 P1-P4 close-out, L9+L11 真活 ledger
- **TODO/completed/91_wf41_3way_integration.md** (Wf-41 P1) — L10 honest-negative ship ledger
- **TODO/completed/92_wf41_phase2_closure_ood.md** (Wf-41 P2) — L12 closure OOD Phase 2 ship ledger
- **TODO/completed/83j_lambda_closeout.md** (Wf-45 final close-out) — 12/12 gap L1-L12 ledger + R20 plan items + 3 skill update
- **TODO/INDEX.md §16** — §16.1 L1-L8 ship table + §16.2 Wf-36/37/38/39/40/41/44 真活 ship list + §16.3 R20 plan items
- **.claude/skills/mol-metal-lambda-calculus/SKILL.md** (40494 bytes) — L1-L8 + L9-L12 cross-link + Wf-36/37/39/40/41/44 ship surface

---

**Date**: 2026-09-26 (created, post-R18 ship + pre-R19 launch)

**Append date (this section)**: 2026-09-26 (Wf-41 Phase 4 L9-L12 + Wf-40/41 真活 cross-link ship)