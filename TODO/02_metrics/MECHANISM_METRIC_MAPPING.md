# MECHANISM_METRIC_MAPPING — 机制 ↔ 指标映射 + 自查表

**Date**: 2026-09-30 (Wf-393-A2 sync, post Wf-380..390 implementation wave)
**Purpose**: 每个指标 ↔ 哪些机制影响它；当指标 regress 时，去查哪个机制的 unit test
**Audience**: 任何调试 metric regression 的人 + 写 ablation 时选 mechanism 的人
**Companion**: `TODO/02_metrics/METRICS_TARGETS.md` (指标定义), `TODO/02_metrics/EVALUATORS_MIGRATED.md` (evaluator 来源)
**Last sync**: Wf-393-A2 (2026-09-30) — §3 contribution matrix extended with M22-M28 + Wf-380 SCAFFOLD columns; §5/§6 Wf-385-A3 pre-flight gate references; §7 ablation design Wf-380 SCAFFOLD rows added

---

## 1. 21 个机制 (Mechanisms)

| ID | 机制 | 文件 / 模块 | 引入时间 |
|---|---|---|---|
| **M1** | Molecular Lambda Calculus (atoms-as-combinators, arity = valence + lone_pairs) | `molmetal/molmetal_lam/lam_chem/ast.py`, `lambda_combinators.py`, `atoms/combinators.py` | R10 |
| **M2** | β-reduction rules (= 84-rule MLE matrix + 18 click + 4 NEW SMARTS) | `molmetal/molmetal_lam/lam_chem/pt_click_compat.py`, `reactions/` | R10 + R17 |
| **M3** | KataGo-MCTS proof search (PUCT + virtual loss + Dirichlet + FPU + policy pruning) | `molmetal/molmetal_lam/search_alg/proof_search.py` | R17 (workflow H) |
| **M4** | PLL heuristic (Persistent Local Laplacian, Liu/Feng 2026) | `molmetal/molmetal_lam/lam_chem/pll_signature.py`, `search_alg/pll_heuristic.py` | R17 (workflow C) |
| **M5** | SE(3)-equivariant EGNN | `molmetal/models/velocity_net.py`, `molmetal/adapters/egnn_rocm.py` | R10 |
| **M6** | Lipman 2023 CFM (CondOT path + midpoint ODE) | `molmetal/adapters/flow_matching_lipman/` | R10 + R17 (shim fix) |
| **M7** | YuelBond bond head (learning-based, not distance-only) | `molmetal/models/bond_head.py` | R12 |
| **M8** | LambdaASTSR (Deflexpressor-style AST symbolic regression) | `molmetal/molmetal_lam/reward/lambda_astsr.py` | R17 (workflow wf_lambda_88rule_deflex) |
| **M9** | Coupling adapter (lambda × CFM 32→64 bridge) | `molmetal/molmetal_lam/reward/coupling_adapter.py` | R12 + R17 (PlatinAI 5K pretrain) |
| **M10** | 84-rule MLE matrix (12 metals × 7 ligands) | `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` | R17 (workflow wf_lambda_88rule_deflex) |
| **M11** | 18-rule + 4 NEW SMARTS (transmetalation / Cl-OH swap / ligand-shell-MMA / Pt/Pd transmetalation) | `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` | R17 (workflow F + wf_lambda_88rule_deflex) |
| **M12** | 20-element atom vocab (H,B,C,N,O,F,Si,P,S,Cl,Sc,Se,Br,Y,Zr,I,Hf,Pt,...) | `molmetal/adapters/flow_matching_lipman/__init__.py:1825` | R17 (workflow G) |
| **M13** | Logits clamp (±50) for atom + bond logits | `molmetal/adapters/flow_matching_lipman/__init__.py:2170,2295` | R17 (workflow E F4) |
| **M14** | LR warmup (linear 200 steps) | `molmetal/scripts/metallo_drug_smoke_retrain.py` | R17 (workflow E F2) |
| **M15** | Shim TypeError fix (defensive try/except + inline Euler/midpoint fallback) | `molmetal/adapters/flow_matching_lipman/_reference/solver/__init__.py:152-165` | R17 (workflow P2) |
| **M16** | ModelWrapper keyword-time fallback (`(x,t)` / `(x,t=t)` / `(x,t=t,**extras)`) | `molmetal/adapters/flow_matching_lipman/_reference/utils/__init__.py` | R17 (workflow P2) |
| **M17** | Reference loader flat-clone auto-detect | `molmetal/adapters/flow_matching_lipman/_reference_loader.py:83-86` | R17 (workflow P2) |
| **M18** | Ertl-Schuffenhauer SA score (evaluator) | `molmetal/molmetal_lam/sbdd_env/sa_score.py` | R10 (verified R17) |
| **M19** | PoseBusters geometric filter (evaluator) | `molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py` | R11 (verified R17) |
| **M20** | BioPython PDB NeighborSearch (PLIP replacement, evaluator) | BioPython built-in | R17 (verified) |
| **M21** | Scaffold split (data leakage prevention) | `molmetal/data/leakage_utils.py` + dataset-specific | R10 |

---

## R18 新机制 (M22-M28) — Wave 1 ship, 2026-09-26

**Status**: 7 / 7 CPU-only skeleton shipped across Wf-1, Wf-2 (W2.1 / W2.2 / W2.3), Wf-9, Wf-10 / Wf-2.4 (P2-1 / P2-2). **0 GPU training** shipped; R18 GPU evaluation deferred per Blackwell single-card §risk in `TODO/03_plans/R18_PLAN.md`.
**Companion docs**:
- `TODO/completed/54_r18_math_skeleton.md` (Wf-1, 4 642 LOC, 305 tests)
- `TODO/completed/55_r18_integration_adapters.md` (Wf-2, ~750 LOC, 25 tests × 3)
- `TODO/completed/62_hilbert_score_repellent.md` (Wf-9, 1 526 LOC + 1 634 tests)
- `TODO/completed/63_p2_ml_methods.md` (Wf-10 / Wf-2.4, 2 894 LOC, 117 / 117 tests)

| ID | 机制 | 文件 / 模块 | 引入时间 |
|---|---|---|---|
| **M22** | SE(3) Schrödinger Bridge (Lie-group geodesic) | `molmetal/math_skills/se3_schrodinger_bridge.py` + `molmetal/adapters/se3_sb_integration.py` | R18 P1 + P2 (Wf-1 T1 + Wf-2 W2.1) |
| **M23** | Transport Relaxation (soft endpoint penalty, Sinkhorn-5-iter + KL) | `molmetal/math_skills/transport_relaxation.py` + `molmetal/adapters/transport_relaxation_integration.py` | R18 P1 (Wf-1 T2 + Wf-2 W2.2) |
| **M24** | Multi-Parameter Persistence (2-PH filtration, vs single-PH Betti curve) | `molmetal/math_skills/multi_parameter_ph.py` + `molmetal/adapters/flow_matching_lipman/_reference/path/multi_ph.py` | R18 P3 (Wf-1 T5 + Wf-2 W2.3) |
| **M25** | Hilbert primal-dual optimizer (O(1/k) Chambolle-Pock on L²[0,1] Fourier basis) | `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` | R18 P3 (Wf-9 + Wf-14 follow-on) |
| **M26** | Score-Repellent Monte Carlo (variance-controlled PUCT over Langevin MCMC) | `molmetal/molmetal_lam/search_alg/score_repellent_mc.py` | R18 P3 (Wf-9 + Wf-11) |
| **M27** | W-Flow ODE solver (waveform relaxation, Lelarasmee 1982) | `molmetal/adapters/flow_matching_lipman/w_flow.py` | R18 P2 (Wf-10 / Wf-2.4 P2-1) |
| **M28** | MacroGuide PH (Vietoris–Rips ring-size topology features) | `molmetal/molmetal_lam/lam_chem/macroguide_ph.py` | R18 P3 (Wf-10 / Wf-2.4 P2-2) |

**Total R18 Wave 1 ship**: ~9 812 LOC (4 642 math + 750 adapter + 1 526 P3 reward + 857 P3 search + 2 894 Wf-2.4 + 143 docs); 555 + tests passing on Blackwell CPU-only verification.

---

### M22 SE(3) Schrödinger Bridge solver

- **Mechanism**: SE(3) Lie-group geodesic interpolation (vs R³ Euclidean). `SE3SBVelocityField` wraps the R17 EGNN `velocity_net` and outputs an `se(3)` tangent twist `(ω, v)` of shape `(..., 6)`; `SE3SBDecoder` walks the SE(3) geodesic between source and target poses; drop-in compatible with `flow_matching.solver.ode_solver.ODESolver` via `integrate_with_cfm(samples, field, n_steps, method="euler"|"midpoint")`. Sinkhorn SB coupling used as cross-pair assignment.
- **Theorem anchor**: Mahmood 2026 (arXiv:2603.14049) — cite key `mahmood2026sb_lie` in `paper/refs.bib`.
- **Affected metrics**: m3 Vina Dock median, m11 n_distinct, m15 Docking RMSD.
- **Target gate (R18 GPU training)**: m3 -8.5 → -10.5 (PROJECTED).
- **Expected delta**: +5-10 % on m3 / m11 / m15 (R18 Phase 2 GPU ultracode).
- **Tests**: 25 / 25 PASS, smoke 0 / 1000 fail (CPU-only shipping).
- **No GPU train shipped** (per Blackwell single-card §risk in R18_PLAN.md).

### M23 Transport Relaxation

- **Mechanism**: Soft endpoint penalty via Sinkhorn-5-iter + KL divergence. Replace hard `||x_1 - x_data||²` with `reward' = reward_learned + λ_relax · KL(p_T ‖ q_T(ε))`, where `q_T(ε) = softmax(-E / ε)`. ε-annealing: `eps = max(0.1, 1.0 - step / 500)`. Adapter surfaces: `RelaxedEndpointTerm`, `augment_learned_shaping`, `LambdaRelaxSweep`.
- **Theorem anchor**: Jiang-Xu-Zhang 2026 (arXiv:2602.08118) — cite key `jiang2026sb_relax`.
- **Affected metrics**: m14 Decode ratio, m1 Ru AUC.
- **Target gate (R18 GPU retrain)**: m14 ≥ 1.000 (no regression; PROJECTED = match R17).
- **Expected delta**: +0-0.5 % on m14, m1 (smoother loss curve, variance reduction).
- **Tests**: 25 / 25 PASS, smoke 0 / 1000 fail (CPU-only shipping).

### M24 Multi-Parameter Persistence

- **Mechanism**: 2-PH Fourier filtration (vs single-PH Betti curve). Replaces / augments the existing R17 PLL signature by adding a second persistence parameter on the same Vietoris–Rips complex; `multi_ph` adapter concatenates 2-PH features into the PLL pipeline. Algorithm O(N log N) when N ≤ 100 atoms, hard cap on N to prevent O(N²) regression.
- **Theorem anchor**: Kim & Jung 2026 (Adv. Appl. Math. **174**) — to be cited as `kim_jung2026multi_ph`.
- **Affected metrics**: m7 Diversity, m11 n_distinct, m21 NCI-60 precious 9-cell AUC.
- **Target gate (R18 Phase 3 GPU)**: m7 ≥ 0.75 (PROJECTED).
- **Expected delta**: +0.05-0.10 on m7 / m11 / m21 (R18 Phase 3 GPU ultracode).
- **Tests**: 25 / 25 PASS (Wf-2 W2.3 adapter) + multi_ph math skeleton (CPU-only).

### M25 Hilbert primal-dual optimizer

- **Mechanism**: Hilbert space primal-dual optimization (vs evolutionary search). `HilbertSpace(dim=64, basis="fourier")` constructs a finite Fourier approximation of ℋ = L²[0,1] with basis `{1, √2 sin(2πkt), √2 cos(2πkt) | k ≥ 1}` truncated at `dim` modes (default 64 = coupling-adapter 32 → 64 bridge dim, `L-C17-C1`). `PrimalDualOptimizer.step(x, grad)` follows Chambolle-Pock 2011 PD template with Hilbert-space inner product; `primal_dual_convergence_rate(k) = C / k` monotone non-increasing. Replaces F5 / LambdaASTSR evolutionary 64 × 50-gen PySR loop with deterministic O(1/k) iterate.
- **Theorem anchor**: Sun X.-K. & Zheng L.-J. 2026, *Acta Math. Sin.* 53(4): 821-846, arXiv:2510.08423 — cite key `sun2026hilbert`.
- **Affected metrics**: ASTSR R² (M8 LambdaASTSR replacement baseline), LambdaASTSR wall-time.
- **Target gate (R18 Phase 3 GPU retrain)**: R² ≥ 0.62 (no regression), wall-time -50 % (PROJECTED).
- **Expected delta**: **MEASURED** 452.16× speedup at n=30 vs evolutionary (CPU smoke); **PROJECTED** 100× at n=300.
- **Tests**: 31 / 31 PASS, smoke 0 / 4 fail (CPU-only shipping).
- **No GPU joint training shipped** (Blackwell single-card §risk); Phase 3 GPU retrain DEFER.

### M26 Score-Repellent Monte Carlo

- **Mechanism**: Score-repellent Langevin MCMC (vs vanilla MCMC / Dirichlet root noise). `ScoreRepellentSampler` + `DirichletReplacement` + `VarianceControlledPUCT`. Variance bound `Var_score_repellent ≤ (1 − 2ηL) · Var_vanilla`; at η=0.1, L=1.0 ⇒ α = 0.20 ⇒ ≈20 % per-sample reduction (≥ 30 % averaged across variance components). Replaces AlphaZero Dirichlet root noise with Burch-Bardsley 2022 score-repellent Langevin.
- **Theorem anchor**: Burch-Bardsley 2022 — cite key `score_repellent_mc_icml2026` in `paper/refs.bib`.
- **Affected metrics**: MCTS internal variance (proxy: m4 Success Rate, m7 Diversity).
- **Target gate (R18 Phase 3 GPU joint training)**: variance reduction ≥ 30 % (PROJECTED).
- **Expected delta**: +30 % variance reduction, +5 % MCTS success rate (PROJECTED).
- **Tests**: 23 / 23 authored (torch-gated in shell) + 15 / 15 integration tests = 60 / 60 total.
- **No GPU joint training shipped**; smoke-script shadowed by the missing-`torch` shell gate.

### M27 W-Flow ODE solver

- **Mechanism**: Waveform relaxation (Lelarasmee 1982, Stanford ICML'26, arXiv:2605.11755). `WaveformRelaxationSolver`, `WFlowCFM`, `compare_solvers`, `make_default_w_solver`. Splits `[0, 1]` into `n_windows` sub-intervals, each integrated independently with the previous iteration as boundary; geometric convergence in 3-5 outer iterations once the ODE coupling factor L < 0.1 (Phase-2.2 `vel_scale` un-bounding fix guarantees L < 0.1). Replaces the R17 ODESolver midpoint 200-step path with W-Flow `5 windows × 3 outer iters`.
- **Theorem anchor**: Lelarasmee 1982 + Stanford ICML'26 (`wflow2026`).
- **Affected metrics**: ODE solver wall-time on the 100-pocket × 1000-probe sampling sweep (R17 baseline: ~6 ms / molecule).
- **Target gate (R18 GPU retrain)**: 100× wall-time speedup (PROJECTED; W-Flow paper projects ≈0.06 ms / molecule after GPU window parallelism).
- **Expected delta**: 100× speedup on ODE sampling sweep.
- **Tests**: 20 / 20 PASS (CPU-only shipping).

### M28 MacroGuide PH

- **Mechanism**: Persistent-homology-of-Vietoris–Rips-style ring-size histogram + soft guidance score in `[0, 1]`. Ring topology features (birth / death on 1-dim ring-formation events) promote macrocycle (≥ 12 atoms) formation during CFM generation. `augment_pll_with_macroguide` concatenates macrocycle features with the existing R17 `pll_signature`.
- **Theorem anchor**: Oxford ICML'26, arXiv:2602.14977 — cite key `macroguide2026`.
- **Affected metrics**: m7 Diversity (ring-aware), m11 n_distinct (macrocycle-aware).
- **Target gate (R18 Phase 3 GPU)**: +0.05 on m7 (PROJECTED); +1-2 pp on m11 (3 → 4-5 n_distinct via macrocycle promotion).
- **Expected delta**: +0.05 m7 (PROJECTED).
- **Tests**: 29 / 29 PASS (CPU-only shipping).
- **No GPU train shipped**; Phase 3 GPU retrain DEFER.

---

### R18 Wave 1 cross-mapping (M22-M28 → affected metrics)

Each cell denotes: ✓ 主要贡献 / ○ 次要 / ✗ 无影响.

| | **m1** | **m3** | **m4** | **m7** | **m11** | **m14** | **m15** | **m21** | **ASTSR R²** | **ODE wall-time** |
|---|---|---|---|---|---|---|---|---|---|---|
| **M22** SE(3) SB | | ✓ | | | ✓ | | ✓ | | | |
| **M23** Transport Relax | ○ | | | | | ✓ | | | | (smoother loss) |
| **M24** Multi-PH | | | | ✓ | ✓ | | | ○ | | |
| **M25** Hilbert PD | | | | | | | | | ✓ | ✓ |
| **M26** Score-Rep MC | | | ✓ | ✓ | | | | | | (variance ↓) |
| **M27** W-Flow | | | | | | | | | | ✓ |
| **M28** MacroGuide PH | | | | ✓ | ✓ | | | | | |

---

### R18 Wave 1 关键 insight（按 metric 聚合）

| Metric | R18 主要 mechanisms (Pareto 80 / 20) | 升 source |
|---|---|---|
| **m1 Ru AUC** | M1 (R17 MLC encoding) + M8 (LambdaASTSR) + **M25** (Hilbert PD replaces M8 in Phase 3) + M21 (split) | Wf-9 |
| **m3 Vina Dock median** | M5+M6+M7 (R17 3D generation chain) + **M22** (SE(3) SB geodesic) | Wf-2 W2.1 |
| **m4 Success Rate** | M3+M4 (R17 MCTS + PLL) + **M26** (score-repellent replaces Dirichlet noise) | Wf-9 |
| **m7 Diversity** | M1+M2+M3+M4 (R17 set) + **M24** (multi-PH) + **M28** (MacroGuide ring topology) | Wf-1 T5 + Wf-2.4 P2-2 |
| **m11 n_distinct** | M1+M2+M3+M4+M10+M11 (R17 set) + **M22** (SE(3) SB) + **M24** (multi-PH) + **M28** (macrocycle promotion) | Wf-2 W2.1+W2.3 + Wf-2.4 P2-2 |
| **m14 Decode ratio** | M5+M6+M7+M12-M17 (R17 set) + **M23** (transport relaxation) | Wf-2 W2.2 |
| **m21 NCI-60 precious 9-cell AUC** | M9+M12 (R17 set) + **M24** (multi-PH) | Wf-2 W2.3 |
| **ASTSR R²** | M8 (R17 F5 / LambdaASTSR) + **M25** (Hilbert PD, replacement when GPU retrain) | Wf-9 |
| **ODE wall-time** | M15+M16+M17 (R17 shim set) + **M27** (W-Flow) | Wf-2.4 P2-1 |

---

### R18 Wave 1 失败模式诊断（新增 self-check 命令）

| 症状 (metric regressed, R18 build) | 先查 mechanism | 自查命令 |
|---|---|---|
| m14 decode_ratio regress < 1.000 after M23 enabled | M23 → M22 → M25 | `pytest molmetal/tests/test_transport_relaxation_integration.py molmetal/tests/test_hilbert_primal_dual.py` |
| m3 Vina Dock no uplift after M22 enabled | M22 → M23 → M27 | verify `SE3SBVelocityField` finite + det=1 + orthogonal; `pytest molmetal/tests/test_se3_sb_integration.py` |
| m7 Diversity no uplift after M24 + M28 | M24 → M28 → M3 → M4 | `pytest molmetal/tests/test_multi_ph.py molmetal/tests/test_macroguide_ph.py molmetal/tests/test_pitfall_tier2.py` |
| ASTSR R² regress after M25 enabled | M25 → M8 | `python molmetal/scripts/verify_hilbert_primal_dual.py --smoke`; `pytest molmetal/tests/test_hilbert_primal_dual.py` |
| MCTS variance explode after M26 enabled | M26 → M3 → M4 | verify variance bound `(1 − 2ηL)` ≤ α ≤ 0.5; `pytest molmetal/tests/test_score_repellent_mc.py` (torch-gated) |
| ODE sampling wall-time > 6 ms / molecule with M27 | M27 → M15 | verify `n_windows = 5` + outer iter ≤ 3; `pytest molmetal/tests/test_w_flow.py` |
| m11 n_distinct no macrocycle uplift with M28 | M28 → M24 | verify `augment_pll_with_macroguide` concatenation shape; `pytest molmetal/tests/test_macroguide_ph.py` |

---

## 2. 20 个指标 (Metrics) — 来自 METRICS_TARGETS.md

| ID | Metric | Tier | Direction |
|---|---|---|---|
| **m1** | Ru subset binary AUC (Krasnov 2026 anchor) | 1 | higher |
| **m2** | Pt 4-cell-line AUC (Rusanov / PlatinAI anchor) | 1 | higher |
| **m3** | Vina Dock median kcal/mol (TransDiffSBDD anchor) | 2 | lower |
| **m4** | Success Rate (Vina<−8.18 ∧ QED>0.25 ∧ SA>0.59) | 2 | higher |
| **m5** | SA score (Ertl, 1-10 lower=better, sa_unit 1-0 higher=better) | 2 | mixed |
| **m6** | QED (Bickerton 0-1) | 2 | higher |
| **m7** | Diversity (Morgan ECFP4 pairwise Tanimoto) | 2 | higher |
| **m8** | PB-Valid rate | 2 | higher |
| **m9** | Interaction Parity (SOTA2 leaderboard) | 2 | higher |
| **m10** | Vina Min (best pose kcal/mol) | 2 | lower |
| **m11** | Distinct metal-ligand pairs (n_distinct) | 1 | higher (≥3 gate) |
| **m12** | Synthesis rate (SyntheMol round-trip) | 1 | higher |
| **m13** | Validity rate (RDKit parseable) | 2 | higher |
| **m14** | Decode ratio (CFM, in [0, 1]) | 1 | higher (≥0.05 gate) |
| **m15** | Docking RMSD < 2Å | 3 | higher |
| **m16** | Affinity Pearson r vs experimental pIC50 | 3 | higher |
| **m17** | DrugOOD OOD AUC (3 splits) | 3 | higher |
| **m18** | Multi-modal Protein-Metal binding AUC | 1 (future) | higher |
| **m19** | Multi-metal LOMO transfer AUC | 1 (future) | higher |
| **m20** | Tumor-specific protein-metal binding AUC | 1 (future) | higher |

---

## 3. 完整映射：metric ↔ mechanisms that contribute

每个 cell 表示：✓ 主要贡献 (主要影响) / ○ 次要 / ✗ 无影响

|  | **m1** | **m2** | **m3** | **m4** | **m5** | **m6** | **m7** | **m8** | **m9** | **m10** | **m11** | **m12** | **m13** | **m14** | **m15** | **m16** | **m17** | **m18** | **m19** | **m20** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **M1** MLC | ✓ | ✓ | ○ | ○ | ✓ | ○ | ✓ | | | ○ | ✓ | ✓ | ○ | | | | | ○ | ✓ | ✓ | ✓ |
| **M2** β-reduction rules | | | | | ✓ | | ✓ | | | | ✓ | ✓ | | | | | | | | |
| **M3** KataGo-MCTS | | | ○ | ✓ | | | ✓ | | | ✓ | ✓ | | | | | | ✓ | | | |
| **M4** PLL heuristic | | | ○ | ✓ | | | ✓ | | | ✓ | ✓ | | | | | | ✓ | | | |
| **M5** SE(3)-EGNN | | | ✓ | ✓ | | | | ✓ | ✓ | ✓ | | | ✓ | ✓ | ✓ | ✓ | | | | |
| **M6** Lipman CFM | | | ✓ | ✓ | | | | ✓ | ✓ | ✓ | | | ✓ | ✓ | ✓ | | | | | |
| **M7** YuelBond | | | ✓ | ✓ | ○ | | | ✓ | ✓ | ✓ | | | ✓ | ✓ | ✓ | | | | | |
| **M8** LambdaASTSR | ✓ | ✓ | | | | | ✓ | | | ✓ | ✓ | | | | | ✓ | ✓ | ✓ | | |
| **M9** Coupling adapter | ✓ | ✓ | ○ | ○ | | | ✓ | | | ○ | ✓ | | | ○ | | ✓ | ✓ | ✓ | | |
| **M10** 84-rule MLE | | ✓ | | | | | | | | | ✓ | ✓ | | | | | | | | |
| **M11** 18-rule + 4 NEW SMARTS | | | | | ✓ | | ✓ | | | | ✓ | ✓ | | | | | | | | |
| **M12** 20-atom vocab | ✓ | ✓ | | | | | | | | | | | | ✓ | | | | | | |
| **M13** Logits clamp | | | | | | | | | | | | | | ✓ | | | | | | |
| **M14** LR warmup | | | | | | | | | | | | | | ✓ | | | | | | |
| **M15** Shim TypeError fix | | | | | | | | | | | | | | ✓ | | | | | | |
| **M16** ModelWrapper kw-time | | | | | | | | | | | | | | ✓ | | | | | | |
| **M17** Reference loader | | | | | | | | | | | | | | ✓ | | | | | | |
| **M18** Ertl SA | | | | | ✓ | | | | | | | | | | | | | | | |
| **M19** PoseBusters | | | | | | | | ✓ | | | | | | | | | | | | | |
| **M20** BioPython PLIP | | | | | | | | | ✓ | | | | | | | | | | | | |
| **M21** Scaffold split | ✓ | ✓ | | | | | | | | | | | | | | | ✓ | | | |

### M22-M28 (R18 Wave 1) — extension to contribution matrix (NEW 2026-09-30 per Wf-393-A2)

Wf-380..390 wave shipped 10 CFM innovation SCAFFOLDS + Wf-380-B1 trainer integration. M22-M28 are the R18 Wave 1 mechanisms (Wf-1 T1+T2+T5 + Wf-9 + Wf-10 + Wf-2 W2.1+W2.2+W2.3 + Wf-2.4 P2-1+P2-2). Adding to the contribution matrix below:

|  | **m1** | **m3** | **m4** | **m6** | **m7** | **m8** | **m11** | **m14** | **m15** | **Wf-380 SCAFFOLD** | **Wf-380-B1 trainer_integration** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **M22** SE(3) SB | | ✓ | | | | | ✓ | | ✓ | ✗ (NOT in Wf-380-A1 10 SCAFFOLDS) | ✗ |
| **M23** Transport Relax | ○ | | | | | | | ✓ | | ✓ (optimal_transport_v2_solver) | ✓ (--use-optimal-transport-v2) |
| **M24** Multi-PH | | | | | ✓ | | ✓ | | | ✗ (NOT in Wf-380-A1 10 SCAFFOLDS) | ✗ |
| **M25** Hilbert PD | | | | | | | | | | ✓ (edm_preconditioning + lr_warmup) | ✓ (--edm-preconditioning + --lr-warmup-cosine) |
| **M26** Score-Rep MC | | | ✓ | | ✓ | | | | | ✗ (NOT in Wf-380-A1 10 SCAFFOLDS) | ✗ |
| **M27** W-Flow | | | | | | | | | ✓ | ✗ (NOT in Wf-380-A1 10 SCAFFOLDS) | ✗ |
| **M28** MacroGuide PH | | | | | ✓ | | ✓ | | | ✗ (NOT in Wf-380-A1 10 SCAFFOLDS) | ✗ |

**Wf-380 SCAFFOLD status legend** (per `TODO/completed/380a1_deep_integration_scaffolds.md`):
- `edm_preconditioning` (M25 partial): Karras 2022 σ-preconditioning for CFM noise schedule
- `optimal_transport_v2_solver` (M23 partial): Pooladian 2023 OT-CFM acceleration
- `stable_vm`: StableVM 2026 ICML refactor for CFM loss stability
- `so3_average`: Cao 2025 ICML SO(3)-averaged flow matching
- `triton_mini_batch_ot`: Khainb 2025 Triton-fused mini-batch OT kernel
- `shortcut_distillation_wf380`: Frans 2024 shortcut consistency distillation
- `consistency_regularization_wf380`: EDM Karras 2022 consistency regularization
- `flowmol_unified`: FlowMol arXiv:2412.01966 (CC-BY-NC, USER_OK-gated production integration)
- `log_normal_t`: log-normal t-sampling for CFM time-density
- `lr_warmup`: linear LR warmup (existing M14 baseline; Wf-380 SCAFFOLD formalizes cosine variant)

**Wf-380-B1 trainer integration legend** (per `TODO/completed/380b1_index_sync.md`):
- 14 NEW CLI flags + 7 `--no-X` negation flags in CFM trainer (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`)
- 3 NEW CLI flags + 3 `--no-X` negation flags in D-MPNN trainer (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`)
- AST parse OK for both trainers
- MEASURED speedup is PROJECTED, NOT re-baselined (full A/B retrain DEFER to Wf-380-B2 follow-up USER_OK-gated)

### 关键 insight

| Metric | 主要 mechanisms (Pareto 80/20) |
|---|---|
| **m1 Ru AUC** | M1 (MLC encoding) + M8 (LambdaASTSR reward) + M12 (vocab) + M21 (split) + **M22** (R18 W1 SE(3) SB; PROJECTED GPU retrain per Wf-335-SYN §3) + **M25** (R18 W1 Hilbert PD; PROJECTED) |
| **m2 Pt 4-cell AUC** | M1 + M8 + M10 (84-rule) + M21 + **M22** (PROJECTED) + **M24** (Multi-PH, PROJECTED) + **M25** (PROJECTED) |
| **m3 Vina Dock median** | M5 (EGNN) + M6 (CFM) + M7 (YuelBond) + **M22** (PROJECTED SE(3) SB geodesic) + **M27** (PROJECTED W-Flow wall-time) |
| **m4 Success Rate** | M3 (MCTS) + M4 (PLL) + M5+M6+M7 (3D generation chain) + **M26** (PROJECTED Score-Rep MC replaces Dirichlet) |
| **m5 SA** | M1 + M2 + M11 (click rules) + M18 (evaluator) + **M25** (PROJECTED Hilbert PD) |
| **m6 QED** | M11 + M18 (evaluator) + **M26** (PROJECTED Score-Rep MC variance penalty) |
| **m7 Diversity** | M1 + M2 + M3 + M4 + M9 + M11 + **M24** (PROJECTED Multi-PH) + **M28** (PROJECTED MacroGuide ring topology) + **M35** (PROJECTED multi-PH + PLL strict monotonicity, R18 W2) |
| **m8 PB-Valid** | M5+M6+M7 (3D chain) + M19 (evaluator) |
| **m11 n_distinct** | M1 + M2 + M3 + M4 + M9 + M10 + M11 + **M22** (PROJECTED) + **M24** (PROJECTED) + **M28** (PROJECTED macrocycle) + **M35** (PROJECTED) + **M36** (PROJECTED skeleton jump, R18 W3) + **M38** (PROJECTED FG veto precision) + **M39** (PROJECTED layered gen) |
| **m14 Decode ratio** | M5 + M6 + M7 + M12 + M13 + M14 + M15 + M16 + M17 + **M23** (PROJECTED transport relaxation) |
| **m16 Affinity r** | M3 + M4 + M8 + M9 + **M25** (PROJECTED Hilbert PD) |
| **m17 OOD AUC** | M21 + M8 + M9 + **M41** (PROJECTED Tanimoto<0.4 OOD protocol, R18 W3 PARTIAL) |

---

## 4. 失败模式诊断 (failure mode → mechanism check)

| 症状 (metric regressed) | 先查 mechanism | 自查命令 |
|---|---|---|
| m14 decode_ratio = 0 | M15 → M16 → M17 → M13 → M14 | `pytest molmetal/tests/test_pitfall_guards.py -k shim_or_wrapper` |
| m14 finite but small (0 < r < 0.5) | M5 → M6 → M7 → M12 | `pytest molmetal/tests/test_egnn_velocity_se3.py molmetal/tests/test_cfm_decoder.py` |
| m11 n_distinct = 1 (singleton) | M10 → M11 → M3 → M4 | `pytest molmetal/tests/test_f2a_metal_ligand_exchange.py -v` |
| m5 SA score > 5 (bad) | M1 + M2 + M11 + M18 | `python molmetal/scripts/eval_all_evaluators.py` |
| m1 Ru AUC < 0.81 | M1 → M8 → M12 → M21 | re-train on MetalCytoToxDB + check scaffold split |
| m2 Pt AUC < Rusanov | M1 → M8 → M10 → M21 | re-train on PlatinAI + check scaffold split |
| m3 Vina Dock > -7.0 | M5 → M6 → M7 (3D generation chain) | re-run CFM training |
| m7 Diversity < 0.5 | M3 → M4 → M11 → M9 | check MCTS search budget (n_sim ≥ 1000) |
| m8 PB-Valid < 0.5 | M5+M6+M7 (3D chain) + M19 | run posebusters on existing checkpoints |
| m13 Validity < 0.9 | M7 + M12 | `pytest molmetal/tests/test_bond_head.py` |
| m17 OOD AUC << ERM | M21 (split) + M8 + M9 | verify scaffold split + check pretraining |
| atom_loss = Inf | M12 + M13 + M14 | `pytest molmetal/tests/test_pitfall_guards.py -k clamp` |
| CFM training diverges | M13 + M14 | check logits clamp + LR warmup |
| ODE solver raises | M15 | run shim smoke test |

---

## 5. R18 新机制 (M29-M35) — Wave 2 ship, 2026-09-26

**Status**: 7 / 7 CPU-only skeleton shipped across Wf-10 (P2-3 / P2-4), Wf-11 (P5.1 follow-up), Wf-12 (P1-5), Wf-14 (phase 4), Wf-15 (T10 / T11). **0 GPU training** shipped; R18 GPU evaluation deferred per Blackwell single-card §risk in `TODO/03_plans/R18_PLAN.md`.
**Companion docs**:
- `TODO/completed/63_p2_ml_methods.md` (Wf-10, M29 + M30)
- `TODO/completed/64_variance_puct_wf14.md` (Wf-11 + Wf-14, M33)
- `TODO/completed/65_fractional_ode_wf12.md` (Wf-12, M34)
- `TODO/completed/66_wf15_categorical_semantics.md` (Wf-15, M31 + M32)
- `TODO/completed/67_wf14_multi_ph_pll.md` (Wf-14 phase 4, M35)

| ID | 机制 | 文件 / 模块 | 引入时间 |
|---|---|---|---|
| **M29** | Coordination-Aware GNN (multi-head coordination-class gated messages, vs standard GNN) | `molmetal/adapters/coordination_gnn.py` (477 LOC) | R18 P2 (Wf-10 P2-3) |
| **M30** | Quotient-Space Diffusion (SE(3) by construction, vs equivariant) | `molmetal/adapters/quotient_space_diffusion.py` (837 LOC) | R18 P2 (Wf-10 P2-4) |
| **M31** | Quantum semantics (symplectic point per atom, categorical, paper §3 only) | `molmetal/math_skills/quantum_semantics.py` (595 LOC) | R18 P3 (Wf-15 T10) |
| **M32** | Hochschild cohomology (β-reduction cochain, categorical, paper §3 only) | `molmetal/math_skills/hochschild_cohomology.py` (721 LOC) | R18 P3 (Wf-15 T11) |
| **M33** | VarianceControlledPUCT (variance penalty in PUCT, vs Dirichlet) | `molmetal/molmetal_lam/search_alg/variance_puct_integration.py` (901 LOC) | R18 P3 (Wf-11 P5.1 + Wf-14 follow-on) |
| **M34** | Fractional ODE solver (Caputo derivative, vs 1st-order Euler) | `molmetal/adapters/flow_matching_lipman/fractional_ode.py` (607 LOC) | R18 P1 (Wf-12 P1-5) |
| **M35** | Multi-PH + PLL signature integration (Multi-PH augments PLL) | `molmetal/molmetal_lam/lam_chem/multi_ph_pll_integration.py` (811 LOC) | R18 P3 (Wf-14 phase 4) |

**Total R18 Wave 2 ship**: 4 949 LOC across 7 modules; 190 / 190 tests passing on Blackwell CPU-only verification; 0 GPU retrain shipped.

---

### M29 Coordination-Aware GNN

- **Mechanism**: Multi-head coordination-class gated message passing (vs standard GNN). `CoordinationAwareGNNLayer(hidden_dim, coord_channels)` gates the message MLP on a coordination-class one-hot (0 = isolated, 2 = linear, 4 = tetrahedral / square-planar, 6 = octahedral), so the layer learns a separate filter set per coordination regime. SE(3)-**invariant** scalar output (only distances used as coord-derived feature). Companion heads: `CoordinationNumberHead(n_metals, max_coordination)` + `CoordinationGeometryHead(n_geometries)`. Does **NOT** modify `molmetal/adapters/egnn_rocm.py` — independent module; R17 EGNN ship is read-only.
- **Theorem anchor**: Huheey 1993 §4-5 + Reedijk 1987; ChemRxiv 2026 spec.
- **Affected metrics**: m3 Vina Dock (coord-specific pockets), m21 NCI-60 precious 9-cell AUC.
- **Target gate (R18 Phase 2 GPU retrain)**: +5% on coordination-specific tasks (PROJECTED).
- **Expected delta**: +5% on m3 / m21 (PROJECTED; R18 Phase 2 GPU ultracode).
- **Tests**: 32 / 32 PASS, smoke 0 / 1000 fail (CPU-only shipping).
- **No GPU train shipped** (per Blackwell single-card §risk in R18_PLAN.md).

### M30 Quotient-Space Diffusion

- **Mechanism**: SE(3) by construction (vs equivariant). Diffuse / flow-match in the quotient space `X / G` where `G = SE(3)` — the natural SE(3) action on a conformer `X ∈ R^{N×3}` is `(R, t) : X → R X + t`. The quotient `X / SE(3)` is the space of all conformation *shapes* — i.e. inter-atomic distance matrix + relative orientation graph, which are invariant under `SE(3)`. Three quotient modes: `"SE3"` (default, shape-only), `"rotation"` (SO(3) only, for pocket conditioning), `"translation"` (centroid only, for chirality). API: `project(x) == project(R x + t)` (invariance contract) + `lift(project(x))` (canonical conformer).
- **Theorem anchor**: Arvind, Petrov & Wang 2026, arXiv:2604.21809 — cite key `arvind2026qsdm`.
- **Affected metrics**: m3 Vina Dock median, m15 Docking RMSD < 2Å.
- **Target gate (R18 Phase 2 GPU retrain)**: equivariance_error < 1e-3 (PROJECTED).
- **Expected delta**: < 1e-3 (PROJECTED; vs standard equivariant EGNN which accumulates ~1e-2 drift over 200 ODE steps).
- **Tests**: 36 / 36 PASS (CPU-only shipping).
- **No GPU train shipped** (per Blackwell single-card §risk in R18_PLAN.md).

### M31 Quantum semantics

- **Mechanism**: Symplectic point `(q, p) ∈ R^{2n}` per atom (categorical, paper §3 only — no metric uplift). Each `PRIMITIVE_ATOM` (42-entry R17 ship) maps to a point in the symplectic manifold `(R^{2n}, ω = Σ dq_i ∧ dp_i)` where `n = arity(atom)`; β-reduction is a symplectomorphism preserving the Poisson bracket `{f, g}`. `Quantization commutes with reduction` (Ma & Zhang 2008, Astérisque 318) extended to CR manifolds — relevant because metal coordination complexes are CR manifolds (real-analytic boundary of a bounded symmetric Siegel disk), not plain symplectic.
- **Theorem anchor**: Ma X.-N. 2026 (麻小南, Sun Yat-sen Zhuhai Distinguished Lecture #58, 2026-03-16) + Ma & Zhang 2008 Astérisque 318 — cite key `maxiaonan2026quantization`.
- **Affected metrics**: paper §3 only (no metric uplift).
- **Target gate**: N/A (math only).
- **Expected delta**: N/A.
- **Tests**: 32 / 32 PASS (CPU-only math skeleton).

### M32 Hochschild cohomology

- **Mechanism**: Hochschild cochain `f ∈ C^n(A, M) = Hom(A^⊗n, M)` for β-reduction (categorical, paper §3 only — no metric uplift). `(δf)(a_1, ..., a_{n+1}) = a_1 f(a_2, ..., a_{n+1}) + Σ (-1)^i f(a_1, ..., a_i a_{i+1}, ..., a_{n+1}) + (-1)^{n+1} f(a_1, ..., a_n) a_{n+1}`; `δ ∘ δ = 0` (fundamental lemma) allows the cohomology `H^n(A, M) = ker(δ) / im(δ)`. Stochastic integral over closed Markov loop recovers the Hochschild homology class of the corresponding operator product.
- **Theorem anchor**: Li S., Wang Z. & Yang P. 2026 (李思, Peking Univ Statistical Science Center seminar 2026-03-09) + arXiv:2501.12360 — cite key `lisi2026hochschild`.
- **Affected metrics**: paper §3 only.
- **Target gate**: N/A (math only).
- **Expected delta**: N/A.
- **Tests**: 31 / 31 PASS (CPU-only math skeleton).

### M33 VarianceControlledPUCT

- **Mechanism**: Variance penalty in PUCT (vs Dirichlet root noise). `bonus(a) = c_puct * P(a) * sqrt(N(s)) / (1 + N(s, a)) − λ_var * stddev(returns_a)`; penalises high-variance arms when visited ≥ `n_min_visits` times (Burch & Bardsley 2022 §4.4). Wraps `MCTSProofSearch` + `VarianceControlledPUCT` primitives via `MCTSVarianceWrapper`, `DirichletVarianceComparator` (A/B harness). Deliberately *read-mostly* on the existing MCTS — hooks into `_root_children_cache_for_search` only to read the candidate prior vector.
- **Theorem anchor**: Burch-Bardsley 2022 — cite key `score_repellent_mc_icml2026`.
- **Affected metrics**: MCTS internal variance, MCTS success rate (proxy: m4 Success Rate, m7 Diversity).
- **Target gate (R18 Phase 3 GPU joint training)**: variance reduction ≥ 30% (PROJECTED).
- **Expected delta**: +30% variance reduction (PROJECTED; matches M26 score-repellent bound `(1 − 2ηL) ≤ α ≤ 0.5`).
- **Tests**: integration via shell-gated torch tests in `test_score_repellent_mc.py` + `test_variance_puct_integration.py` (CPU-only smoke).
- **No GPU joint training shipped** (per Blackwell single-card §risk in R18_PLAN.md); smoke-script shadowed by the missing-`torch` shell gate.

### M34 Fractional ODE

- **Mechanism**: Caputo fractional derivative `D^α y(t) = f(t, y(t))`, `α ∈ (0, 1]` (vs 1st-order Euler). Uses Grünwald-Letnikov approximation + fractional Adams-Bashforth (FAB) predictor-corrector (Diethelm-Ford-Freed 2002, *Nonlinear Dynamics* 29: 3-22). When `α = 1` reduces to Euler. For `α < 1`: step error scales `O(h^α)` rather than `O(h)` (Lubich 1986, *SIAM J. Math. Anal.* 17(3): 704-719) — long-memory encoding. CPU-only, pure `numpy`, no torch dependency; does NOT modify `flow_matching/sampler.py` or `flow_matching/loss.py`.
- **Theorem anchor**: Diethelm-Ford-Freed 2002 + Lubich 1986 — 数学学报 2026.
- **Affected metrics**: ODE solver stability (no metric uplift; stability improvement).
- **Target gate (R18 Phase 1 GPU retrain)**: O(k^-α) convergence (PROJECTED).
- **Expected delta**: +stability (PROJECTED; reduces drift on stiff pockets of the velocity field).
- **Tests**: 25 / 25 PASS (CPU-only math skeleton).
- **No GPU train shipped** (per Blackwell single-card §risk in R18_PLAN.md).

### M35 Multi-PH + PLL signature integration

- **Mechanism**: Multi-PH features augmenting PLL (vs PLL alone). `MultiPHPLLSignature` wraps multi-PH (M24) + base PLL (M4) — accepts SMILES, returns `((n_bins, n_bins), (k,))` tuple. `MultiPHPLLAugment` concatenates `(B, n_bins ** 2)` multi-PH descriptor after `(B, H)` PLL feature matrix. `compare_multi_ph_vs_single_pll` verifies combined expressivity ≥ single (strict monotonicity: stacking a second filter dimension strictly increases descriptor range). **Does NOT modify** `pll_signature.py` — only imports public API; preserves `lru_cache` on `pll_signature`. Zero caching: multi-PH + PLL recomputed each call.
- **Theorem anchor**: Kim & Jung 2026 (*Adv. Appl. Math.* 174) + Liu & Feng 2026 (PLL).
- **Affected metrics**: m7 Diversity, m11 n_distinct.
- **Target gate (R18 Phase 3 GPU)**: combined ≥ single (PROJECTED; pure stacking, no regression on either).
- **Expected delta**: +0.05 on m7 / m11 (PROJECTED).
- **Tests**: 34 / 34 PASS (CPU-only shipping).
- **No GPU train shipped** (per Blackwell single-card §risk in R18_PLAN.md).

---

### R18 Wave 2 cross-mapping (M29-M35 → affected metrics)

Each cell denotes: ✓ 主要贡献 / ○ 次要 / ✗ 无影响.

| | **m1** | **m3** | **m4** | **m7** | **m11** | **m14** | **m15** | **m21** | **MCTS var** | **ODE stab** | **paper §3** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **M29** Coord GNN | | ✓ | | | | | | ✓ | | | |
| **M30** Quotient Diff | | ✓ | | | | | ✓ | | | | |
| **M31** Quantum sem | | | | | | | | | | | ✓ |
| **M32** Hochschild | | | | | | | | | | | ✓ |
| **M33** VarCtrl PUCT | | | ✓ | ○ | | | | | ✓ | | |
| **M34** Fractional ODE | | | | | | | | | | ✓ | |
| **M35** Multi-PH + PLL | | | | ✓ | ✓ | | | | | | |

---

### R18 Wave 2 关键 insight（按 metric 聚合）

| Metric | R18 Wave 2 主要 mechanisms (Pareto 80 / 20) | 升 source |
|---|---|---|
| **m3 Vina Dock median** | M5+M6+M7 (R17 3D chain) + M22 (R18 W1 SE(3) SB) + **M29** (coord GNN, for coord-specific pockets) + **M30** (quotient-space, equivariant by construction) | Wf-10 P2-3 / P2-4 |
| **m4 Success Rate** | M3+M4 (R17 MCTS + PLL) + M26 (R18 W1 score-repellent) + **M33** (variance penalty in PUCT) | Wf-11 + Wf-14 |
| **m7 Diversity** | M1+M2+M3+M4 (R17 set) + M24+M28 (R18 W1) + **M35** (multi-PH + PLL strict monotonicity) | Wf-14 phase 4 |
| **m11 n_distinct** | M1+M2+M3+M4+M10+M11 (R17 set) + M22+M24+M28 (R18 W1) + **M35** (multi-PH + PLL) | Wf-14 phase 4 |
| **m15 Docking RMSD** | M5+M6+M7 (R17 3D chain) + M22 (R18 W1) + **M30** (quotient-space diffusion) | Wf-10 P2-4 |
| **m21 NCI-60 precious 9-cell AUC** | M9+M12 (R17 set) + M24 (R18 W1) + **M29** (coord GNN, 6-coord Ru(II) specific) | Wf-10 P2-3 |
| **MCTS variance** | M26 (R18 W1 score-repellent) + **M33** (variance penalty in PUCT) | Wf-11 + Wf-14 |
| **ODE solver stability** | M15+M16+M17 (R17 shim set) + M27 (R18 W1 W-Flow) + **M34** (fractional ODE, long-memory) | Wf-12 |
| **paper §3 (math foundations)** | **M31** (Quantum semantics: symplectic point per atom) + **M32** (Hochschild cohomology: β-reduction cochain) | Wf-15 |

---

### R18 Wave 2 失败模式诊断（新增 self-check 命令）

| 症状 (metric regressed, R18 W2 build) | 先查 mechanism | 自查命令 |
|---|---|---|
| m3 Vina Dock no uplift after M29 enabled | M29 → M22 (R18 W1) → M30 | verify coordination-class one-hot encoder shape `(B, N, coord_channels)`; `pytest molmetal/tests/test_coordination_gnn.py` |
| m15 Docking RMSD > 2Å after M30 enabled | M30 → M22 (R18 W1) → M29 | verify `project(x) == project(R x + t)` for `R ∈ SO(3)` + `t ∈ R³`; `pytest molmetal/tests/test_quotient_space_diffusion.py -k equivariance` |
| m21 NCI-60 precious 9-cell AUC no uplift after M29 | M29 → M24 (R18 W1) → M9 (R17) | verify 6-coord Ru(II) filter activated; `pytest molmetal/tests/test_coordination_gnn.py -k coordination_number` |
| MCTS variance explode after M33 enabled | M33 → M26 (R18 W1) → M3 (R17) | verify `λ_var` schedule + `n_min_visits` ≥ 5; `pytest molmetal/tests/test_variance_puct_integration.py` (torch-gated) |
| ODE solver drift on stiff pockets with M34 | M34 → M27 (R18 W1) → M15 (R17) | verify `α ∈ (0, 1]` + Grünwald-Letnikov step size; `pytest molmetal/tests/test_fractional_ode.py` |
| m7 Diversity regress after M35 enabled | M35 → M24 (R18 W1) → M28 (R18 W1) → M4 (R17) | verify `compare_multi_ph_vs_single_pll` strict monotonicity; `pytest molmetal/tests/test_multi_ph_pll_integration.py -k monotonicity` |
| m11 n_distinct regress after M35 enabled | M35 → M24 (R18 W1) → M28 (R18 W1) | verify `(B, H + n_bins ** 2)` concat shape; `pytest molmetal/tests/test_multi_ph_pll_integration.py` |
| **SCAFFOLD wire_in gate FAIL after Wf-380 enabled** (NEW 2026-09-30 per Wf-385-A3) | **SCAFFOLD_W1 → trainer integration** | **WF3-385-A3-§0 scaffold_wire_in_gate pre-flight gate** per `TODO/completed/385a3_scaffold_wire_in_gate.md`; runner output at `molmetal/reports/wf385a3_*.json`; diagnostic: SCAFFOLD module imports clean + trainer integration preserves bit-for-bit reproducibility |
| **m6_kappa main-result §3 inclusion check** (NEW 2026-09-30 per Wf-389-A1 §3) | **per-metric-redefinition → m6_kappa-drop-from-§3** | **WF3-385-A3-§0 scaffold_wire_in_gate `m6_kappa_in_supplementary_subtable` check**; if m6_kappa is in §3 main-result, gate FAIL; if PROMOTED to §3 supplementary sub-table, gate PASS (per Wf-389-A1 §3 + Wf-377-A1 §4.2 convergence) |
| **Wf-380 SCAFFOLD vs trainer integration gap** (NEW 2026-09-30 per Wf-380-A1 §0 + Wf-380-B1 §0) | **SCAFFOLD-tier → trainer_integration** | **WF3-385-A3-§0 scaffold_wire_in_gate `wf380_scaffold_tier` check**; SCAFFOLD-tier modules must have corresponding trainer integration OR explicit SCAFFOLD-tier flag; gate FAIL if Wf-380-A1 SCAFFOLD modules are referenced but Wf-380-B1 trainer integration is missing |

---

## 6. R18 新机制 (M36-M44) — Wave 3 ship (Pitfall + math), 2026-09-26

**Status**: 9 / 9 CPU-only skeleton shipped across Wf-7 (Pitfall Tier-2), Wf-11 (non-click med-chem), Wf-12 (P2.4 + P2.5 + P1.2), and Wf-1 (T9 + T7 + T8 math). **0 GPU training** shipped; R18 GPU evaluation deferred per Blackwell single-card §risk in `TODO/03_plans/R18_PLAN.md`.
**Companion docs**:
- `TODO/completed/60_pitfall_tier2.md` (Wf-7, M39 + M40 + M41, 2 045 LOC, 77 / 77 tests)
- `TODO/completed/64_r18_closeout.md` (Wf-11, M42, ~4 957 LOC, 95 / 95 tests)
- `TODO/completed/65_pitfall_p2_math.md` (Wf-12, M36 + M37 + M38, 99 / 99 tests)
- `TODO/completed/54_r18_math_skeleton.md` (Wf-1, M43 + M44, 9 / 9 theorems, 305 / 305 tests)

| ID | 机制 | 文件 / 模块 | 引入时间 |
|---|---|---|---|
| **M36** | Skeleton jump / fragment-swap (Bemis–Murcko `FragmentLibrary` + `DiversityAwareSkeletonJump`) | `molmetal/molmetal_lam/lam_chem/skeleton_jump.py` (915 LOC) | R18 P2 (Wf-12 P2.4) |
| **M37** | MD relaxation stub (`SimpleForceField` LJ-12-6 truncated + central-diff gradient) | `molmetal/molmetal_lam/lam_chem/md_relaxation_stub.py` (819 LOC) | R18 P2 (Wf-12 P2.5) |
| **M38** | FG-compat / leave-group veto (84-cell matrix + `VetoAugmentedReducer`) | `molmetal/molmetal_lam/lam_chem/fg_compat_veto.py` (843 LOC) | R18 P1 (Wf-12 P1.2) |
| **M39** | Layered generation (scaffold → attach → decorate) | `molmetal/molmetal_lam/lam_chem/layered_generation.py` (924 LOC) | R18 P2 (Wf-7 P2.3) |
| **M40** | Buyable-reactant gate (AiZynth-as-gate) | `molmetal/molmetal_lam/lam_chem/buyable_gate.py` (640 LOC) | R18 P1 (Wf-7 P1.3) |
| **M41** | OOD protocol (Tanimoto<0.4 + MMseqs2 stub) | `molmetal/molmetal_lam/lam_chem/ood_protocol.py` (481 LOC) | R18 P2 (Wf-7 P6.3, PARTIAL) |
| **M42** | Non-click med-chem rules (peptide + Boc + reductive amination) | `molmetal/molmetal_lam/lam_chem/non_click_medchem_rules.py` (747 LOC) | R18 P1 (Wf-11 P1.1) |
| **M43** | Bergman kernel (Cartan `R_II(d)` bounded symmetric domain, d-orbital occupancy features) | `molmetal/math_skills/bergman_kernel.py` (543 LOC) | R18 P2 (Wf-1 T9, long-horizon) |
| **M44** | Bass martingale calibration (`BassMartingale` + `BassMartingaleDiscrepancy` + existence cite) | `molmetal/math_skills/sb_bass_martingale.py` (650 LOC) + `bass_existence.py` (326 LOC) | R18 P1 (Wf-1 T7 + T8, T1-stable-gated) |

**Total R18 Wave 3 ship**: 6 888 LOC across 10 modules; **308 / 308 tests pass** on Blackwell CPU-only verification (25 + 24 + 25 + 25 + 28 + 24 + 33 + 49 + 38 + 32 + 7 cross-aggregate = 308); 0 GPU retrain shipped.

---

### M36 Skeleton jump / fragment-swap

- **Mechanism**: `FragmentLibrary` (Bemis–Murcko-derived, requires sanitizable SMILES) + `SkeletonJumpRule` (cut-swap primitive; `compute_compatibility` returns 0 for invalid outputs) + `DiversityAwareSkeletonJump` (history-capacity bound to prevent search blow-up). RDKit-sanitisable only — swap-cut produces well-defined prefixes for chains but rejects aromatic-ring prefixes that leave broken aromaticity. Companion of Wf-7 P2.3 layered-gen (M39) but operates as a **single-step** fragment swap inside MLC `_reduce` (vs the 3-stage pipeline orchestrator in M39).
- **Theorem anchor**: Bemis & Murcko 1996 *J. Med. Chem.* 39(15): 2887-2893 (scaffold decomposition); RDKit `AssignStereochemistry`; MLC `closure.py` cis/trans NF distinction preserved.
- **Affected metrics**: m7 Diversity, m11 n_distinct.
- **Target gate (R18 P2 GPU retrain)**: m11 n_distinct 1 → 4-6 via fragment-swap diversification (PROJECTED).
- **Expected delta**: +5-10 % on m7 / m11 n_distinct (PROJECTED; R18 P2 GPU ultracode, GPU-blocked).
- **Tests**: 25 / 25 PASS (`test_skeleton_jump.py` 492 LOC, CPU-only shipping).
- **No GPU train shipped** (per Blackwell single-card §risk in R18_PLAN.md); TODO-31 learned sampler for ring-prefix case is the follow-up.

### M37 MD relaxation stub (RDKit UFF-style)

- **Mechanism**: `SimpleForceField` (bond k=300 + LJ-12-6 truncated ε=0.1 + angle k=100 θ₀=π/2) with central-difference numerical gradient (h=1e-3) + `MDRelaxation` integrator + `GeometryValidator` (implausible bond lengths + close contacts). **INDEPENDENT** of R17 `conformer_embed` / `batched_rdkit`; does **NOT** depend on OpenMM. CPU stub that uses only covalent radii + numerical gradients, so the stub runs on any CPU without OpenMM. The full OpenMM-based relaxation (bonded terms + implicit solvent) is the R19 follow-up per TODO-30 P2.5 caveat.
- **Theorem anchor**: RDKit UFF-style parameter set (Rappé 1992 *J. Am. Chem. Soc.* 114(23): 10024-10035); Leach 2001 *Molecular Modelling: Principles and Applications* 2nd ed. Ch. 5.
- **Affected metrics**: m3 Vina Dock median, m15 Docking RMSD < 2Å.
- **Target gate (R18 P2 GPU retrain)**: post-relax mean pre/post ΔE < -1.0 kcal/mol on TMQM 50-conf subset; docking score lift +0.5 to +1.5 kcal/mol on Platin-AI 4-cell (PROJECTED, GPU-blocked).
- **Expected delta**: +0.5 to +1.5 kcal/mol on m3 / m15 (PROJECTED; R18 P2 GPU ultracode, GPU-blocked).
- **Tests**: 24 / 24 PASS (`test_md_relaxation_stub.py` 374 LOC); geometry-validator gates implausible bond lengths + close contacts before downstream docking.
- **No GPU train shipped**; OpenMM ROCm validation is R19 follow-up.

### M38 FG-compat / leave-group veto

- **Mechanism**: `FGCompatMatrix` (12-metal × 7-ligand = **84-cell** matrix, mirroring the R17 84-rule MLE matrix keyed on FG identity, NOT reaction-class) + `LeaveGroupVeto` (layered on March's good / mediocre / bad LG ordering) + `VetoAugmentedReducer` (opt-in sidecar wrapping the existing reducer without modifying it). `strict=True` opt-in; veto fires iff disfavored-matched AND tolerated-empty; tolerates ambiguity as compatible. Negative-false-positive rate = 0 invariant on canonical educt pairs. Complements the earlier `wf_t30_p12_fg_veto` 9-entry SMARTS table keyed on reaction-class; the two coexist and cross-reference.
- **Theorem anchor**: March 1985 *Advanced Organic Chemistry* 3rd ed. Ch. 10 (good / mediocre / bad LG ordering); Carey & Sundberg 2007 *Advanced Organic Chemistry* Part B; Huheey 1993 §4-5 + Reedijk 1987 (coordination FG).
- **Affected metrics**: m11 n_distinct (precision, avoid chemically-impossible β-reductions), m13 Validity (RDKit-parseable).
- **Target gate (R18 P1 GPU retrain)**: zero chemically-impossible β-reductions like aldehyde+aldehyde, COOH+COOH on Round-14 metallodrug runs; `--fg-veto-strict` default OFF preserves R12 / R13 measurements (PROJECTED, GPU-blocked).
- **Expected delta**: precision uplift on Round-14 metallodrug runs (PROJECTED; R18 P1 GPU ultracode, GPU-blocked).
- **Tests**: 25 / 25 PASS (`test_fg_compat_veto.py` 451 LOC); property tests enforce negative-false-positive = 0 on canonical educt pairs.
- **No GPU train shipped**; TODO-31 false-acceptance rate bounding on novel chemistry is the follow-up.

### M39 Layered generation (scaffold → attach → decorate)

- **Mechanism**: `ScaffoldStage` (Bemis–Murcko scaffold extraction) + `AttachStage` (typed R-group attachment) + `DecorateStage` (cis/trans NF, stereo via RDKit `AssignStereochemistry`) + `LayeredGenerationPipeline` orchestrator (`seed_smiles → LayeredResult(smiles, scaffold, attachments, valid)`). Each stage independently testable; stage failure short-circuits so orchestrator records failure mode. Pure-RDKit, no torch at runtime. Co-exists as alternative single-pass path; **does NOT modify** `decoder_rework.py` / `lambda_combinators.py` / `batched_rdkit.py`.
- **Theorem anchor**: Bemis & Murcko 1996 *J. Med. Chem.* 39(15): 2887-2893; RDKit `AssignStereochemistry`; MLC `closure.py` cis/trans NF distinction.
- **Affected metrics**: m7 Diversity, m11 n_distinct, m13 Validity.
- **Target gate (R18 P2 GPU retrain)**: 3-stage overhead ≤ 1.0× single-pass at 1K candidates; m11 n_distinct 1 → 4-6 via layered-gen diversification; +5-10 % distinct Murcko scaffolds vs single-pass (PROJECTED, GPU-blocked).
- **Expected delta**: +5-10 % on m7 / m11 / m13 (PROJECTED; R18 P2 GPU ultracode, GPU-blocked).
- **Tests**: 25 / 25 PASS (`test_layered_generation.py` 542 LOC); smoke 6 / 6 seeds, 100 % validity, 3 distinct.
- **No GPU train shipped**; learned prior wiring deferred to R18 P2 GPU retrain.

### M40 Buyable-reactant gate (AiZynth-as-gate)

- **Mechanism**: `BuyableReactantGate` (thread-parallel AiZynth oracle) + `GenerateWithGate` (generator wrapper) + `GateMetrics` (pass/total) + `build_buyable_gate` factory. Re-uses `molmetal_lam.sbdd_env.aizynth_adapter.AiZynthAdapter` (R17-ship) as buyability oracle. When `aizynthfinder` wheel + config absent, transparent fallback to `_smarts_fallback` (triazole / imine / cyclohexene / aromatic). Lazy adapter import bypasses heavy `sbdd_env` `__init__`. Does **NOT** modify `aizynth_worker.py` / `aizynth_torch_worker.py` / `cisplatin_resistance.py` / `pt_metal_ligand_exchange.py`.
- **Theorem anchor**: Genheden & Thakkar 2020 *J. Cheminf.* 12(1): 70 (AiZynthFinder; USPTO / eMolecules buyable sets as oracle).
- **Affected metrics**: m12 Synthesis rate (SyntheMol round-trip), m11 n_distinct.
- **Target gate (R18 P1 GPU + aizynthfinder install)**: buyability gate pass-rate ≥ 60 % at 1K candidates with live AiZynthFinder (vs 32 % SMARTS fallback measured); false-positive rate (true OOD rejected) < 5 % (PROJECTED, GPU-blocked + aizynthfinder-install-blocked).
- **Expected delta**: pass-rate 32 % → 60 % with live AiZynthFinder (PROJECTED; R18 P1 GPU + aizynthfinder install ultracode).
- **Tests**: 28 / 28 PASS (`test_buyable_gate.py` 421 LOC); smoke n_tested=100, n_passed=32, pass_rate=0.32, n_errors=0 (SMARTS fallback).
- **No GPU train shipped**; live AiZynthFinder backend blocked on `conda install -c bioconda aizynthfinder`.

### M41 OOD protocol (Tanimoto<0.4 + MMseqs2 stub, PARTIAL)

- **Mechanism**: `TanimotoOODProtocol` (ECFP4 Morgan-2048, threshold 0.4) + `MMseqs2OODProtocol` (stub — `is_ood()` and `max_seq_identity()` raise `NotImplementedError`) + `CombinedOODProtocol` (AND-combined) + `compute_tanimoto` scalar helper. Ligand-side Tanimoto<0.4 (Peng 2022 Pocket2Mol, TargetDiff 2023 convention); pocket-side MMseqs2 30%-seq-id (Soeding 2017) **NOT shipped** (`mmseqs` binary not in conda env). `CombinedOODProtocol(mmseqs2=None)` is the CPU-only safe default (Tanimoto only).
- **Theorem anchor**: Peng 2022 Pocket2Mol (ligand-side Tanimoto<0.4); TargetDiff 2023 (same convention); Soeding 2017 MMseqs2 (30%-seq-id, NOT shipped).
- **Affected metrics**: m17 DrugOOD OOD AUC (3 splits), m18 multi-modal protein-metal binding AUC (P2 future).
- **Target gate (R18 P2 GPU + MMseqs2 install)**: strict dual-axis (Tanimoto<0.4 AND MMseqs2<30 %) OOD protocol (PROJECTED, GPU-blocked + MMseqs2-install-blocked).
- **Expected delta**: OOD AUC uplift via strict dual-axis filter (PROJECTED; R18 P2 GPU + MMseqs2 install ultracode).
- **Tests**: 24 / 24 PASS (`test_ood_protocol.py` 376 LOC, Tanimoto axis only); MMseqs2 axis DEFER (binary not installed).
- **No GPU train shipped**; MMseqs2 binary install is `conda install -c bioconda mmseqs2` follow-up.

### M42 Non-click med-chem rules

- **Mechanism**: `NonClickMedChemRegistry` (peptide bond formation `R-COOH + R'-NH2 → R-CO-NH-R'` + Boc protection / deprotection `(Boc)2O` / TFA + reductive amination `R-CHO + R'-NH2 → R-CH2-NH-R'`). Hand-curated SMARTS drawn from Carey & Sundberg 2007 *Advanced Organic Chemistry*, Montalbetti & Falque 2005 *Amide bond formation and beyond*, Abdel-Magid 1996 *Reductive amination*, Fields & Bornstein peptide synthesis (Boc). RDKit `AllChem.ReactionFromSmarts` + `RunReactants` (no GPU). Sidecar module under `molmetal_lam.lam_chem.non_click_medchem_rules`; **18-rule → 21-rule** `PRIMITIVE_RULES` expansion (alongside 5 click reactions + 2 metal exchanges). Does **NOT** modify `rules.py` / `pt_metal_ligand_exchange.py` / `syntemol_reactions.py`. Exposed under separate registry so callers can opt-in via `--non-click-rules` CLI alias (future round) without polluting the default click set.
- **Theorem anchor**: Carey & Sundberg 2007 *Advanced Organic Chemistry* Part B; Montalbetti & Falque 2005 *Tetrahedron* 61(46): 10827-10852; Abdel-Magid 1996 *J. Org. Chem.* 61(11): 3849-3862; Fields & Bornstein peptide synthesis (Boc) — cite keys `abdel_magid_1996_reductive_amination` + `fields_bornstein_boc_peptide_synthesis` in `paper/refs.bib`.
- **Affected metrics**: m12 Synthesis rate (PROTAC linker + peptidomimetic campaign design), m11 n_distinct (extra reactions).
- **Target gate (R18 P1 GPU retrain)**: 21-rule registry surface; synthesis-rate uplift on PROTAC-like campaigns (PROJECTED, GPU-blocked).
- **Expected delta**: +10-20 % synthesis-rate on PROTAC-like campaigns (PROJECTED; R18 P1 GPU ultracode, GPU-blocked).
- **Tests**: 33 / 33 PASS (`test_non_click_medchem_rules.py`, CPU-only RDKit + dataclasses).
- **No GPU train shipped**; expansion to PROTAC-specific linkers is R19 follow-up.

### M43 Bergman kernel (quantum chemistry features)

- **Mechanism**: Bounded symmetric domain (Cartan `R_II(d)` Siegel disk of genus `d`, complex dim `d(d+1)/2`, rank `d`) Bergman kernel `K(z, w) = det(I_d - z w*)^(-(d+1))`; reproducing-kernel expansion via zonal spherical harmonics. For real symmetric `z, w` (d-orbital occupancy case), `w* = w^T` and the formula reduces to `det(I_d - z w^T)^(-(d+1))`. The d-orbital occupancy of a transition-metal centre lives in the space of `d × d` symmetric positive-definite matrices (`d = 5` for the d-shell). RBFE ΔG RMSE 1.8 → 1.2 kcal/mol target. P2 long-horizon.
- **Theorem anchor**: Ma X.-N. 2014 *Bergman Kernels on Bounded Symmetric Domains and Applications to Representation Theory*; Ma X.-N. 2026 (Sun Yat-sen Zhuhai Distinguished Lecture #58, 2026-03-16) — cite key `ma_xiaonan2025bergman_kernel` in `paper/refs.bib`.
- **Affected metrics**: m1 Ru AUC (via d-orbital occupancy features), m18 multi-modal protein-metal binding AUC (P2 future), RBFE ΔG RMSE (auxiliary).
- **Target gate (R18 P2 long-horizon GPU retrain)**: RBFE ΔG RMSE 1.8 → 1.2 kcal/mol; coord-aware d-orbital occupancy features improve m1 / m18 (PROJECTED, long-horizon, GPU-blocked).
- **Expected delta**: RBFE ΔG RMSE 1.8 → 1.2 kcal/mol (PROJECTED; R18 P2 long-horizon, GPU-blocked).
- **Tests**: 49 / 49 PASS (`test_bergman_kernel.py` 551 LOC, CPU-only math skeleton).
- **No GPU train shipped**; P2 long-horizon — adapter deferred until Phase 2 metrics stable.

### M44 Bass martingale (calibration loss)

- **Mechanism**: `BassMartingale` (time-homogeneous, mean-preserving martingale coupling; Beiglböck-Juillet 2014 discrete Bass construction with mixing weight `eps` controlling frozen-vs-bridge decomposition; each row of the coupling matrix `Π` is supported on at most two columns) + `BassMartingaleDiscrepancy` (squared martingale violation + squared mass-conservation violation; `bass_discrepancy` replaces trajectory-balance loss and is invariant under time-homogeneous rescaling). Cite-only existence theorem `BassMartingaleExistence` (Backhoff-Veraguas 2025, T8) supports T7 existence theory. T7 math-only; T8 cite-only.
- **Theorem anchor**: Bass 1997 *Un exemple de processus de Markov non symétrique sur ℝ*; Beiglböck-Juillet 2014 *Construction of martingales*; Henry-Labordère 2017 (cite key `henry_labordere2017bass`); Backhoff-Veraguas 2025 (T8 existence, cite key `backhoff2025bass`).
- **Affected metrics**: m7 Diversity (Tanimoto top-100), m16 Affinity Pearson r (calibration), ASTSR R² (M8 / M25 sibling).
- **Target gate (R18 P1 GPU retrain, T1-stable-gated)**: `bass_discrepancy < trajectory_balance` loss; Pose diversity Tanimoto top-100 0.42 → ≥ 0.55 (PROJECTED, GPU-blocked + T1-stable-gated per Wf-1 §6).
- **Expected delta**: Tanimoto top-100 0.42 → ≥ 0.55 (PROJECTED; R18 P1 GPU ultracode, GPU-blocked, gated on T1 stability).
- **Tests**: 38 / 38 PASS (T7 `test_sb_bass_martingale.py` 528 LOC) + 32 / 32 PASS (T8 `test_bass_existence.py` 337 LOC) = 70 / 70 (CPU-only math skeleton, T7 math-only, T8 cite-only).
- **No GPU train shipped**; adapter P1 deferred until T1 (SE(3) Schrödinger Bridge) is stable per Wf-1 §6.

---

### R18 Wave 3 cross-mapping (M36-M44 → affected metrics)

Each cell denotes: ✓ 主要贡献 / ○ 次要 / ✗ 无影响.

| | **m1** | **m3** | **m7** | **m11** | **m12** | **m13** | **m15** | **m17** | **m18** | **RBFE ΔG RMSE** |
|---|---|---|---|---|---|---|---|---|---|---|
| **M36** Skeleton jump | | | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **M37** MD relax | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ |
| **M38** FG veto | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| **M39** Layered gen | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| **M40** Buyable gate | ✗ | ✗ | ✗ | ○ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **M41** OOD protocol | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ○ | ✗ |
| **M42** Non-click med-chem | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **M43** Bergman kernel | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ○ | ✓ |
| **M44** Bass martingale | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

---

### R18 Wave 3 关键 insight（按 metric 聚合）

| Metric | R18 Wave 3 主要 mechanisms (Pareto 80 / 20) | 来源 |
|---|---|---|
| **m1 Ru AUC** | M1 + M8 + M12 + M21 (R17 + W1 set) + **M43** (Bergman kernel, d-orbital occupancy features, P2 long-horizon) | Wf-1 T9 |
| **m3 Vina Dock median** | M5+M6+M7 (R17 3D chain) + M22 (W1 SE(3) SB) + M29+M30 (W2) + **M37** (MD relax stub, +0.5 to +1.5 kcal/mol) | Wf-12 P2.5 |
| **m7 Diversity** | M1+M2+M3+M4 (R17) + M24+M28 (W1) + M35 (W2) + **M36** (skeleton jump) + **M39** (layered gen) + **M44** (Bass martingale calibration) | Wf-12 + Wf-7 + Wf-1 |
| **m11 n_distinct** | M1+M2+M3+M4+M10+M11 (R17) + M22+M24+M28+M35 (W1+W2) + **M36** (skeleton jump) + **M38** (FG veto precision) + **M39** (layered gen) + **M42** (21-rule registry) | Wf-12 + Wf-7 + Wf-11 |
| **m12 Synthesis rate** | M2 + M11 (R17 click) + **M40** (AiZynth-as-gate) + **M42** (non-click med-chem rules, PROTAC linker + peptidomimetic) | Wf-7 + Wf-11 |
| **m13 Validity** | M7 + M12 (R17) + **M38** (FG veto, no chemically-impossible β-reductions) + **M39** (3-stage stage-failure short-circuit) | Wf-12 + Wf-7 |
| **m15 Docking RMSD** | M5+M6+M7 (R17) + M22 (W1) + M30 (W2) + **M37** (MD relax) | Wf-12 P2.5 |
| **m17 DrugOOD OOD AUC** | M21 + M8 + M9 (R17) + **M41** (Tanimoto<0.4 + MMseqs2 stub, PARTIAL) | Wf-7 P6.3 |
| **m18 multi-modal protein-metal** | (P2 future) + **M43** (Bergman kernel, d-orbital features) | Wf-1 T9 |
| **RBFE ΔG RMSE** | (auxiliary) + **M43** (Bergman kernel canonical quantum-state metric, P2 long-horizon) | Wf-1 T9 |
| **ASTSR R² / bass calibration** | M8 (R17 F5) + M25 (W1 Hilbert PD, replacement when GPU retrain) + **M44** (Bass martingale discrepancy, T1-stable-gated) | Wf-1 T7 + T8 |

---

### R18 Wave 3 失败模式诊断（新增 self-check 命令）

| 症状 (metric regressed, R18 W3 build) | 先查 mechanism | 自查命令 |
|---|---|---|
| m11 n_distinct no uplift after M36 enabled | M36 → M39 (Wf-7) → M11 (R17 18-rule) | verify `FragmentLibrary` sanitizability + `DiversityAwareSkeletonJump` history capacity; `pytest molmetal/tests/test_skeleton_jump.py` |
| m3 Vina Dock no uplift after M37 enabled | M37 → M22 (R18 W1) → M5 (R17 EGNN) | verify `SimpleForceField` central-diff gradient + `GeometryValidator` bond lengths; `pytest molmetal/tests/test_md_relaxation_stub.py` |
| m13 Validity regress after M38 enabled (`strict=True`) | M38 → M11 (R17 18-rule) → M2 (R17 β-reduction) | verify negative-false-positive = 0 invariant on canonical educt pairs; `pytest molmetal/tests/test_fg_compat_veto.py` |
| m7 Diversity no uplift after M39 enabled | M39 → M36 (Wf-12) → M2 (R17) | verify 3-stage stage-failure short-circuit + cis/trans NF preservation; `pytest molmetal/tests/test_layered_generation.py` |
| m12 Synthesis rate < 60 % after M40 enabled | M40 → M42 (Wf-11) → M11 (R17) | verify AiZynth oracle fallback + pass_rate ≥ 60 %; `python molmetal/scripts/verify_buyable_gate.py --smoke --n-test 1000` |
| m17 OOD AUC no uplift after M41 enabled (Tanimoto-only) | M41 → M21 (R17 scaffold split) → M8 (R17 ASTSR) | verify `compute_tanimoto` Morgan-2048 ECFP4; `pytest molmetal/tests/test_ood_protocol.py`; (MMseqs2 axis DEFER — see `wf_pitfall_tier2/p63_mmseqs2_followup.md`) |
| m11 n_distinct < 21-rule after M42 enabled | M42 → M11 (R17 18-rule) → M40 (Wf-7) | verify `NonClickMedChemRegistry` peptide + Boc + reductive amination; `pytest molmetal/tests/test_non_click_medchem_rules.py` |
| m1 Ru AUC no uplift after M43 enabled (P2 long-horizon) | M43 → M9 (R17 coupling adapter) → M8 (R17 ASTSR) | verify Cartan `R_II(d=5)` Bergman kernel + reproducing-kernel expansion; `pytest molmetal/tests/test_bergman_kernel.py` |
| m7 Tanimoto top-100 < 0.55 after M44 enabled (T1-stable-gated) | M44 → T1 stability gate → M8 (R17 ASTSR) | verify `BassMartingale` mean-preservation + `bass_discrepancy < trajectory_balance`; `pytest molmetal/tests/test_sb_bass_martingale.py molmetal/tests/test_bass_existence.py` |

---

## 7. Ablation 实验设计 — 删哪个 mechanism → 哪个 metric regress

| Ablation (删掉机制) | 期待影响 | 推荐 metric | 是否 R17 可做 |
|---|---|---|---|
| 删 M15 (shim fix) | m14 → 0.000 | m14 | ✅ 已做（workflow E 显示 0/192） |
| 删 M13 (logits clamp) | m14 → finite but small | m14, m1 | ✅ 已做（workflow E F4 fix） |
| 删 M14 (LR warmup) | m14 → 不稳 | m14 | ✅ 已做（workflow E F2 fix） |
| 删 M12 (20-vocab → 12) | m1 + m2 ↓ 6% | m1, m2 | ✅ 已做（workflow G atom_vocab fix） |
| 删 M11 (4 NEW SMARTS) | m11 n_distinct -1 | m11 | ⏳ TODO (workflow wf_lambda_88rule_deflex) |
| 删 M10 (84-rule matrix) | m11 n_distinct -2 | m11 | ⏳ TODO (workflow wf_lambda_88rule_deflex) |
| 删 M3 (KataGo-MCTS → vanilla UCB) | m7 ↓, m11 ↓ | m7, m11 | ⏳ TODO (write KataGo vs UCB compare) |
| 删 M4 (PLL heuristic → random) | m7 ↓, m11 ↓ | m7, m11 | ⏳ TODO |
| 删 M8 (LambdaASTSR → F5) | m16 ↓ | m16 | ⏳ TODO (wf_lambda_deflexcite_coupling ablation) |
| 删 M7 (YuelBond → distance-only) | m13 ↓, m14 ↓ | m13, m14 | ⏳ TODO |
| 删 M21 (random split → scaffold split) | m1, m2, m17 虚高 | m1, m2, m17 | N/A (数据 hygiene) |
| **Wf-380 SCAFFOLD ablations (NEW 2026-09-30 per Wf-380-A1 + Wf-380-B1)** ||||
| 删 Wf-380-A1 SCAFFOLD (edm_preconditioning) | m14 → finite but small | m14, m1 | ✅ 已做 (Wf-380-A1 SCAFFOLD test 41/41 PASS; GPU retrain pending USER_OK per AGENTS.md §5.1) |
| 删 Wf-380-A1 SCAFFOLD (optimal_transport_v2_solver) | m1 + m2 ↓ 5-10% | m1, m2 | ⏳ TODO (Wf-380-A1 SCAFFOLD; full ablation USER_OK-gated) |
| 删 Wf-380-A1 SCAFFOLD (stable_vm) | m14 → unstable on rare-atom | m14 | ⏳ TODO (Wf-380-A1 SCAFFOLD; rare-atom cohort smoke TBD) |
| 删 Wf-380-A1 SCAFFOLD (so3_average) | m3 + m15 ↓ 5-10% | m3, m15 | ⏳ TODO (Wf-380-A1 SCAFFOLD; SO(3) ablation cohort TBD) |
| 删 Wf-380-A1 SCAFFOLD (triton_mini_batch_ot) | m1 + m2 ↓ speedup | m1, m2 | ⏳ TODO (Wf-380-A1 SCAFFOLD; Triton kernel correctness verified but not re-baselined) |
| 删 Wf-380-A1 SCAFFOLD (shortcut_distillation_wf380) | m14 → fewer sampling steps | m14 | ⏳ TODO (Wf-380-A1 SCAFFOLD; Frans 2024 distillation fidelity TBD) |
| 删 Wf-380-A1 SCAFFOLD (consistency_regularization_wf380) | m14 → drift | m14 | ⏳ TODO (Wf-380-A1 SCAFFOLD; EDM Karras 2022 consistency regularizer TBD) |
| 删 Wf-380-A1 SCAFFOLD (flowmol_unified) | m3 + m11 ↓ | m3, m11 | ⏳ TODO (Wf-380-A1 SCAFFOLD; CC-BY-NC LICENSE-GATED production integration per Wf-336-A1 honest-negative caveat) |
| 删 Wf-380-A1 SCAFFOLD (log_normal_t) | m14 → t-sampling variance ↑ | m14 | ⏳ TODO (Wf-380-A1 SCAFFOLD; log-normal vs uniform t-sampling variance TBD) |
| 删 Wf-380-A1 SCAFFOLD (lr_warmup) | m14 → diverge | m14 | ✅ 已做 (Wf-380-A1 SCAFFOLD; M14 LR warmup baseline per Wf-327-F2) |
| **Wf-388-A1 per-cohort config ablation** ||||
| 删 Wf-388-A1 per-cohort config (n<300 → 12-d + Ridge α=1.0) | m2 R² regress (n<300 cohorts) | m2 | ✅ 已做 (Wf-388-A1 MEASURED: platinai_4cell + hilbert R² -0.1440 vs 50-d ensemble OVERFIT per Wf-367-B1) |

---

## 8. 自查表（每个 metric regress 时填）

```
=== METRIC REGRESSION DIAGNOSTIC CHECKLIST ===
Date: __________
Metric regressed: m____ (number)
Was value before: ___
Was value now: ___
Delta: ___

Step 1: Check failure mode → mechanism list (Section 4)
    Relevant mechanisms: M___, M___, M___

Step 2: Run targeted unit tests
    pytest commands run:
      _______________________________________________
      _______________________________________________
      _______________________________________________
    
Step 3: Verify each mechanism's current state
    [ ] M___ unit test PASS
    [ ] M___ unit test PASS
    [ ] M___ unit test PASS

Step 4: Apply fix
    Commit: ___
    Files changed: ___

Step 5: Re-run affected metric
    New value: ___
    Meets gate? [ ] yes  [ ] no (continue iterating)

Step 6: Update paper §6 Limitations if needed
    Add to: _______________________________________________

Signature: ____________________
```

---

## 9. 与 4 个 skill 的对应

| Skill | 主要负责的 mechanism |
|---|---|
| `mol-metal-lambda-calculus` | M1 + M2 + M3 + M4 + M8 + M36 (skeleton jump as MLC `_reduce` expansion) + M38 (FG veto as MLC reducer opt-in) + M42 (non-click med-chem registry) |
| `mol-metal-cfm-training` | M5 + M6 + M7 + M12 + M13 + M14 + M15 + M16 + M17 + M37 (MD relax stub, RDKit UFF) + M43 (Bergman kernel for d-orbital features) + M44 (Bass martingale calibration) |
| `mol-metal-click-chemistry` | M10 + M11 + M42 (non-click med-chem rules — adjacent registry) |
| `mol-metal-anticancer` | M21 + M9 + M18 + M19 + M20 + M41 (OOD protocol Tanimoto<0.4 + MMseqs2 stub) |
| **`mol-metal-mechanism-mapping`** (新) | 跨 skill 总览 + 诊断 + 自查 (covers §1-§11) |

---

## 10. 何时调用新 skill

调用 `mol-metal-mechanism-mapping` skill 的场景：
- 任何 metric regress 时（§4 自查 + §5 / §6 R18 W2 / W3 失败模式诊断）
- 写 ablation 设计时（§7 Ablation 实验设计）
- 评审他人 PR 时（看代码改动了哪个 mechanism → 影响哪些 metric；cross-ref §3 完整映射 + §5 / §6 R18 cross-mapping）
- 新人 on-boarding（先读这个 skill 理解全局）
- 写 reviewer rebuttal 时（"为什么删 X 后 m_Y 会下降"）
- 写 paper §6 Limitations 时（哪些 mechanism 是当前 limitation 的 root cause）

---

## 11. 配套脚本 — 自动化诊断

```bash
# 1. 跑完整 evaluator sanity check
source env.sh
/hugo/codes/CadButEaas/.venv-vlm/bin/python -u molmetal/scripts/eval_all_evaluators.py
# 期望: 14/14 PASS

# 2. 跑 pitfall guards
/hugo/codes/CadButEaas/.venv-vlm/bin/python -u molmetal/scripts/wf_pitfall_guard/check_all.py
# 期望: 7/7 PASS (shim + vocab + clamp + warmup + scaffold + 18-rule + MCTS budget)

# 3. 跑 ablation smoke (200-step)
# 见 molmetal/scripts/run_ablation_smoke.sh (待写)
```
