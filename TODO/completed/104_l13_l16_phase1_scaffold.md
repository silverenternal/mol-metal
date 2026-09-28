# Wf-78 — L13-L16 Phase 1 Scaffolding Close-out

**Date**: 2026-09-26
**Workflow**: Wf-78 (Phases 1+2+3 — R22-P1 L13 / R22-P2 L14 / R22-P3 L15 / R22-P4 L16 Phase 1 CPU-only initial code + smoke tests)
**Status**: ✅ **SHIP — 4 NEW L13-L16 modules** (hierarchical-multimetal / neural-ode-kinetics / pareto-multiobj / wetlab-bayesian) **+ 73 tests PASS** + **smoke PASS** for all 4
**Priority**: HIGH (drives L13-L16 R22 first-close candidates per `r22_l13_l16_gap_analysis.md` §7.1; L13 = 188 scan signals / L16 = 70 / L15 = 52 / L14 = 20 of 330 total)
**Owner**: Hugo Lee + Wf-78 single-pass Claude Code author (3 phases)
**Parent**:
- `TODO/pending/r22_l13_l16_gap_analysis.md` (Wf-72 Phase 1, 311 lines, 4 candidate Lambda gaps + 330 scan signals)
- `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` (Wf-72 Phase 2, ~440 lines, 6 plan items)
- `TODO/completed/101_r22_plan_l13_l16.md` (Wf-72 P1+P2+P3 close-out)
- `TODO/completed/103_todo_final_consolidation.md` (Wf-76 TODO ledger final; precedes Wf-78)
- `TODO/INDEX.md` §17.5 (TODO-103..106 R22 plan rows)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (12-row L1-L12 trajectory, proposed L13-L16 4-row extension)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate, CRITICAL — honored)
**Constraint**: CPU-only initial code; synthetic mock + smoke test only; **no GPU runs**; **no wet-lab hands**; **no paper work touched** (TODO-84 paper-gate honored)
**Skill alignment**: `party-twenty-spirit` §3.5 (0-tolerance honest framing) + `mol-metal-lambda-calculus` (closure.py / lam_chem/) + `mol-metal-r18-math` (M26 Score-Rep / T1 SE3 SB / T7 SB-Bass) + `mol-metal-click-chemistry` (L13 dative-bond SMARTS deferred to Wf-79) + `mol-metal-neural-ode-kinetics` (S10 SKILL.md, Wf-76 P1 ship) + `mol-metal-anticancer` (trans-effect / redox cycling)

---

## §1 TL;DR

Wf-78 (3 phases) ships the **initial CPU-only scaffolding** for all 4 L13-L16 candidate Lambda gaps, pinning the algorithm shape for downstream Wf-76 P2 (GPU retrain) and Wf-79 P1.3 (dative-bond SMARTS + GPU-0 sweep):

| Phase | Deliverable | LOC | Tests | Status |
|---:|---|---:|---:|---|
| **Wf-78 P1** | `molmetal/molmetal_lam/reward/hierarchical_multimetal.py` (L13) | **512 LOC** | **16/16 PASS** | ✅ SHIP |
| **Wf-78 P2** | `molmetal/molmetal_lam/reward/neural_ode_kinetics.py` (L14) | **981 LOC** | **17/17 PASS** | ✅ SHIP |
| **Wf-78 P3a** | `molmetal/molmetal_lam/reward/pareto_multiobj.py` (L15) | **560 LOC** | **20/20 PASS** | ✅ SHIP |
| **Wf-78 P3b** | `molmetal/molmetal_lam/reward/wetlab_bayesian.py` (L16) | **624 LOC** | **20/20 PASS** | ✅ SHIP |
| **Aggregate** | **4 NEW modules + 4 NEW test files** | **2 677 LOC** | **73/73 PASS** | ✅ **SHIP** |

**Headline (MEASURED, 2026-09-26)**:

| Metric | Value | Source |
|---|---:|---|
| Modules shipped | **4 NEW** | `molmetal_lam/reward/{hierarchical_multimetal,neural_ode_kinetics,pareto_multiobj,wetlab_bayesian}.py` |
| Aggregate LOC | **2 677 LOC** | sum of 4 modules (512 + 981 + 560 + 624) |
| Test files | **4 NEW** | `tests/test_{hierarchical_multimetal,neural_ode_kinetics,pareto_multiobj,wetlab_bayesian}.py` |
| Aggregate tests | **73** | sum of 4 test files (16 + 17 + 20 + 20) |
| Test runtime | **0.43 s wall** | `pytest molmetal/tests/test_{hierarchical_multimetal,neural_ode_kinetics,pareto_multiobj,wetlab_bayesian}.py` (today) |
| R17 surface modified | **0 files** | closure.py / lam_chem/ / reward/lambda_astsr.py carry from Wf-36/Wf-37/Wf-49 bit-identical |
| Paper files touched | **0** | TODO-84 paper-gate honored |
| GPU retrain executed | **0** | CPU-only initial code |
| Wet-lab hands | **0** | R22-P4 6-mo wet-lab deferred to R23-P3 per `r22_p1_p2_p3_p4_p5_p6_plan.md` §6 |
| Reachability gate lifted | **N/A** | L13 structural blockers unchanged; Wf-79 R22-P1 P1.3 GPU plan is the lift attempt |
| M26 Score-Rep multi-task loss wire-up | **3-term (α·R²_static + β·MSE_rate + γ·CE_t1/2)** | L14 §1.2, α=0.5/β=0.3/γ=0.2 per S10 SKILL.md §2.4 anchor |
| NSGA-III reference directions | **15 simplex points** (4-obj p=2, Das-Dennis) | L15 §2.1, Deb-Jain 2014 §IV.B |
| q-EI batch acquisition | **q default = 5** | L16 §2.2, Hernández-Lobato 2017 |

---

## §2 Per-deliverable

### §2.1 Wf-78 Phase 1 — L13 hierarchical multi-metal (R22-P1 P1.1+P1.2)

**Status**: ✅ SHIP — `molmetal/molmetal_lam/reward/hierarchical_multimetal.py` (512 LOC, 8 public API) + `molmetal/tests/test_hierarchical_multimetal.py` (16 tests PASS) ship today.

**Architecture**: Outer × inner loop over `TargetSpec` × `MetalPool` (8 d-block centers: Pt/Pd/Ru/Rh/Ir/Os/Au/Re spanning 4d + 5d) with `max_outer_depth=2` default. The P1.1 + P1.2 scaffold pins the **shape** of the L13 algorithm without lifting the reachability gate — the structural blockers (valence-BNF dative-bond rejection, tautomerisation redex exhaustion, sample sparsity) remain, addressed by Wf-79 R22-P1 P1.3 GPU phases.

**8 public API symbols**: `MetalPool`, `TargetSpec`, `HierarchicalCoordConfig`, `HierarchicalCoordResult`, `HierarchicalCoordinator`, `synthetic_mock_score`, `smoke_hierarchical`, `SUPPORTED_METALS`.

**Smoke (3 targets × 8 metals × depth 2 = 48 pairs, deterministic SHA-256 mock, <5 s CPU)**.

**Why L13 first**: 188 scan signals = 57% of 330 total, top candidate per `r22_l13_l16_gap_analysis.md` §7.1. Closes the structural reachability ceiling (0.5744-0.5962 ≪ gate 0.95) per `completed/96_r20_p5_closure_catherine.md` §3.2.

### §2.2 Wf-78 Phase 2 — L14 Graph Neural ODE kinetics (R22-P2 P2.1+P2.2+P2.3)

**Status**: ✅ SHIP — `molmetal/molmetal_lam/reward/neural_ode_kinetics.py` (981 LOC, 13 public API + 10 module-level constants) + `molmetal/tests/test_neural_ode_kinetics.py` (17 tests PASS, 0.13 s wall).

**Architecture**: 4 ODE drivers (`BindingKinetics` 3-state ligand-exchange / `DissociationKinetics` 4-step cisplatin hydrolysis cascade 6-state / `CellularUptake` Ctr1+OCT2+passive+efflux 2-state Michaelis-Menten / `RedoxCycling` Pt(II)↔Pt(IV) + Ru(II)↔Ru(III) 2-state Nernst-driven) + `MultiTaskLoss` 3-term (α·R²_static + β·MSE_rate + γ·CE_t1/2 with α=0.5, β=0.3, γ=0.2) per S10 SKILL.md §2.4. NumPy forward-Euler integrator (CPU only); torchdiffeq deferred to Wf-76 P2.1 GPU-1 retrain.

**13 public API + 10 constants**: BindingKinetics / DissociationKinetics / CellularUptake / RedoxCycling / MultiTaskLoss / KineticsTrajectory / rate_scale_for_metal / synthetic_mock_kinetics / smoke_neural_ode_kinetics / + 10 module-level constants (SUPPORTED_KINETIC_METALS / KINETIC_REGIME / AQUATION_HALF_LIVES_S / REDOX_POTENTIALS_V / UPTAKE_FLUX_FMOL_CELL_HR / STATIC_R2_PLATINAI_4CELL / LIGAND_EXCHANGE_RATE_CONSTANTS / HYDROLYSIS_CASCADE_RATE_CONSTANTS / MULTI_TASK_LOSS_WEIGHTS / T_HALF_CLASS_BOUNDS_S).

**Pt cisplatin anchor**: t1/2 = 7200 s = 2 hr per Reedijk 1987 (cross-check PASS).

**Why L14 second**: 20 scan signals = 6% of 330, MEDIUM-priority candidate per gap analysis. Lifts static-ASTR collapse detector (PlatinAI 4-cell R² ≈ -0.02 at epoch 4 per `completed/93_r20_p2_hilbert_astsr_gpu.md` §1.1) into continuous-time kinetics dimension.

### §2.3 Wf-78 Phase 3a — L15 multi-objective Pareto front (R22-P3 P3.1+P3.2)

**Status**: ✅ SHIP — `molmetal/molmetal_lam/reward/pareto_multiobj.py` (560 LOC, 10 public API) + `molmetal/tests/test_pareto_multiobj.py` (20 tests PASS).

**Architecture**: 4 objectives uniformly MAX-convention (after sign-flip on minimised axes) — toxicity (min → max `-toxicity`, Williamson + Bajorath 2009) / efficacy (max, PlatinAI-5K AUC) / admet (max, QED × SA × Lipinski × logS) / selectivity (max, CancerScreenDB ΔAUC). Composed on top of `molmetal_lam/search_alg/pareto.py` (439-LOC NSGA-II primitives — `dominates`, `non_dominated_set`, `pareto_front`, `hypervolume`, `rank_population`).

**10 public API symbols**: `ParetoObjective`, `ParetoConfig`, `build_pareto_vector`, `pareto_objectives_from_table`, `nsga3_reference_directions` (15 simplex points for 4-obj p=2 Das-Dennis), `mcts_nsga3_bridge` (VariancePUCT → NSGA-III offspring niching per Deb-Jain 2014 §IV.B), `per_cell_line_breakdown`, `per_metal_pareto_rank` (30-row d-block Sc..Tl table), `compute_hypervolume_indicator` (Zitzler+Thiele 1999), `build_synthetic_metallodrug_pop` (deterministic 100-cand mock with per-metal reachability bias Pt=0.9/Au=0.85/Ru=Ir=0.7/Pd=0.6/Os=Rh=0.55), `synthesize_pareto_front`.

**Why L15 third**: 52 scan signals = 16% of 330, MEDIUM-priority candidate. Closes the m18/m19/m20 tier-1 metric ceiling per `INDEX §17.1`.

### §2.4 Wf-78 Phase 3b — L16 wet-lab-in-the-loop Bayesian opt (R22-P4 P4.1)

**Status**: ✅ SHIP — `molmetal/molmetal_lam/reward/wetlab_bayesian.py` (624 LOC, 11 public API) + `molmetal/tests/test_wetlab_bayesian.py` (20 tests PASS).

**Architecture**: Three-layer (in-silico D-MPNN top-20 → 4-D ARD feature vector × Gaussian-process surrogate Matern 5/2 kernel Williams+Rasmussen 2006 §4.2 × q-EI batch acquisition Hernández-Lobato 2017). Per-institution protocol table — ORNL (protein-metal binding SPR/ITC, 8wk) + PKU (cisplatin IC50 7wk) + 协和 (NCI-60 GI50 panel 12wk, IRB-required). Iterative `cycles × q` loop: fit GP → score all 20 → pick top-q → simulate wet-lab measurement → repeat.

**11 public API symbols**: `WetLabConfig`, `Institution`, `build_institution_table`, `GaussianProcessSurrogate`, `qExpectedImprovement`, `build_synthetic_in_silico_top20`, `build_synthetic_wetlab_measurements`, `compute_silico_wetlab_agreement`, `run_iterative_loop`, `synthesize_wetlab_loop`, `main`.

**Cross-ref**: complements `molmetal_lam/lam_chem/wetlab_protocol.py` (R20-P6 plumbing — Tier 1 assay dataclass) + `wetlab_extensions.py` + `wetlab_reward_channel.py`. **Actual wet-lab outreach is DEFERRED to R23-P3** per `r22_p1_p2_p3_p4_p5_p6_plan.md` §6.

**Why L16 fourth**: 70 scan signals = 21% of 330, HIGH-priority candidate. Closes the wet-lab cooperation gap (TODO-97 P6.1 ORNL/PKU/协和 outreach 0 progress today).

---

## §3 MEASURED vs PROJECTED (per `party-twenty-spirit §3.5`)

### §3.1 MEASURED today (CPU-only docs)

| Item | Value | Source |
|---|---:|---|
| L13 module LOC | **512** | `wc -l molmetal_lam/reward/hierarchical_multimetal.py` |
| L13 tests | **16/16 PASS** | `pytest molmetal/tests/test_hierarchical_multimetal.py` (today) |
| L14 module LOC | **981** | `wc -l molmetal_lam/reward/neural_ode_kinetics.py` |
| L14 tests | **17/17 PASS** | `pytest molmetal/tests/test_neural_ode_kinetics.py` (today) |
| L15 module LOC | **560** | `wc -l molmetal_lam/reward/pareto_multiobj.py` |
| L15 tests | **20/20 PASS** | `pytest molmetal/tests/test_pareto_multiobj.py` (today) |
| L16 module LOC | **624** | `wc -l molmetal_lam/reward/wetlab_bayesian.py` |
| L16 tests | **20/20 PASS** | `pytest molmetal/tests/test_wetlab_bayesian.py` (today) |
| Aggregate LOC | **2 677** | sum of 4 modules |
| Aggregate tests | **73/73 PASS** | `pytest` aggregate (0.43 s wall, today) |
| L13 metal pool | **8 d-block** (Pt/Pd/Ru/Rh/Ir/Os/Au/Re) | `hierarchical_multimetal.SUPPORTED_METALS` |
| L13 outer depth default | **2** | matches R22-P1 P1.2 hierarchical BFS depth-≥2 spec |
| L14 ODE drivers | **4** (ligand-exchange / hydrolysis cascade / cellular uptake / redox cycling) | `neural_ode_kinetics.py` |
| L14 multi-task loss weights | α=0.5 / β=0.3 / γ=0.2 | S10 SKILL.md §2.4 anchor |
| L14 Pt cisplatin t1/2 anchor | **7200 s = 2 hr** | Reedijk 1987 (cross-check PASS) |
| L14 integrator | **NumPy forward-Euler** (CPU) | torchdiffeq deferred to Wf-76 P2.1 GPU-1 retrain |
| L15 objectives | **4** (toxicity / efficacy / admet / selectivity) | uniform MAX-convention |
| L15 NSGA-III reference directions | **15 simplex points** (4-obj p=2) | Das-Dennis Deb-Jain 2014 §IV.B |
| L15 per-metal reachability bias | Pt=0.9 / Au=0.85 / Ru=Ir=0.7 / Pd=0.6 / Os=Rh=0.55 | `build_synthetic_metallodrug_pop` |
| L16 q-EI batch size default | **q=5** | `qExpectedImprovement` |
| L16 Matern 5/2 kernel | **CPU-only deterministic NumPy** | Williams+Rasmussen 2006 §4.2 |
| L16 institution table | **3 rows** (ORNL/PKU/协和) | `build_institution_table` |
| R17 surface modified | **0 files** | closure.py / lam_chem/ / reward/lambda_astsr.py bit-identical |
| Paper files touched | **0** | TODO-84 paper-gate honored |
| GPU retrain executed | **0** | CPU-only initial code |
| Wet-lab hands | **0** | R22-P4 6-mo deferral to R23-P3 |
| Reachability gate lifted | **N/A** | L13 structural blockers unchanged |

### §3.2 PROJECTED (reserved for Wf-76 P2 GPU-1 + Wf-79 P1.3 GPU-0 + R23-P3 wet-lab)

| Item | Why PROJECTED | When MEASURED |
|---|---|---|
| L13 reachability ≥ 0.95 (30-metal × 1000 sweep) | requires chemistry-aware dative-bond SMARTS + hierarchical BFS depth ≥ 2 + GPU-0 sweep | R22-P1 P1.3 (Wf-79) 3-4 wk |
| L14 ligand-exchange t1/2 RMSE ≤ 0.5 | requires torchdiffeq.odeint + dopri5 + adjoint sensitivity + Reedijk 2008 literature kinetics data | R23-P2 4 wk GPU-1 |
| L14 multi-task loss uplift | requires 3-task GPU retrain on real PlatinAI + Ctr1+OCT2 + Pt(II)/Pt(IV) data | R23-P2 4 wk GPU-1 |
| L15 hypervolume ≥ 0.6 (3-objective NSGA-III bench) | requires m18/m19/m20 metric definition + real 4-D scorer + GPU-0 sweep | R22-P3 (Wf-77) 3 wk |
| L15 per-cell-line Pareto ranking | requires PlatinAI-5K AUC + CancerScreenDB ΔAUC real data | R22-P3 3 wk |
| L16 in-silico-vs-wet-lab Pearson r ≥ 0.7 | requires 6-mo wet-lab cooperation (R20-P6 TODO-97) + GP/q-EI BO | R23-P3 (post-R23-P1 LOI) |
| L16 per-institution cycle throughput | requires ORNL SPR/ITC + PKU IC50 + 协和 GI50 wet-lab cycles | R23-P3 6 mo wet-lab + 2 wk CPU BO |
| LAMBDA_ROADMAP §4.2 4-row extension (L13-L16) | §6 of `r22_l13_l16_gap_analysis.md` proposes structure | Wf-72 Phase 4 (next cycle) |
| m22 m18 Protein-Metal AUC ≥ 0.78 (R22 target) | requires 5-metal multitask pretrain + 5-seed GPU retrain | R22 timeline |
| m23 PlatinAI 3-cell AUC ≥ 0.65 (A2780cis R22 target) | requires multi-seed retrain + scaffold split | R22 timeline |
| m24 cross-dataset macro R² ≥ 0.20 (R22 target) | requires Hilbert + ASTSR joint retrain per Wf-71 P3 baseline | R22 timeline |

---

## §4 Before vs After

| Aspect | Before (Wf-72..77 pre-Wf-78) | After (Wf-78 P1+P2+P3 ship) |
|---|---|---|
| **Lambda gap coverage** | L1-L12 algorithmic gaps (closure + MCTS + triangle + CFM + 3-way + OOD) per LAMBDA_ROADMAP §4.2 | **L1-L16** — adds coordination depth (L13), kinetics (L14), Pareto (L15), wet-lab loop (L16) |
| **L13-L16 module existence** | 0 modules | **4 NEW modules** (`hierarchical_multimetal.py` / `neural_ode_kinetics.py` / `pareto_multiobj.py` / `wetlab_bayesian.py`) |
| **L13-L16 unit-test coverage** | 0 tests | **73 tests** (16 + 17 + 20 + 20), all PASS in 0.43 s wall |
| **L13 outer × inner loop** | not implemented | **8-metal pool × 2-depth outer × target_id spec** with per-metal + per-target breakdown |
| **L14 ODE driver stack** | not implemented (only static ASTSR R² ≈ -0.02) | **4 ODE drivers** + NumPy forward-Euler integrator + 3-task multi-task loss wire-up |
| **L15 NSGA-III integration** | partial (only NSGA-II primitives in `search_alg/pareto.py`) | **NSGA-III reference directions** (15 simplex) + `mcts_nsga3_bridge` + per-cell-line + per-metal breakdown |
| **L16 in-silico ↔ wet-lab loop** | docs only (ORNL/PKU/协和 LOI deferral per TODO-97) | **GP + q-EI + 3-institution protocol table + cycle loop + silico-wet-lab agreement** |
| **R22-P1 P1.1+P1.2 status** | doc-only (per `r22_p1_p2_p3_p4_p5_p6_plan.md` §2) | **code SHIP** — P1.3 GPU-0 sweep deferred to Wf-79 |
| **R22-P2 P2.1+P2.2+P2.3 status** | doc-only (per `r22_p1_p2_p3_p4_p5_p6_plan.md` §3) | **code SHIP** — P2.3 GPU-1 retrain deferred to Wf-76 P2.1 |
| **R22-P3 P3.1+P3.2 status** | doc-only (per `r22_p1_p2_p3_p4_p5_p6_plan.md` §4) | **code SHIP** — P3.2 GPU-0 sweep deferred to Wf-77 |
| **R22-P4 P4.1 status** | doc-only (per `r22_p1_p2_p3_p4_p5_p6_plan.md` §5) | **code SHIP** — P4.2..P4.5 wet-lab cooperation deferred to R23-P3 |
| **Smoke verification** | n/a (no L13-L16 modules) | **4 smoke PASS** — hierarchical × neural-ode × pareto × wetlab each print headline + run deterministic synthetic mock |
| **Honest framing** | proposed in `r22_l13_l16_gap_analysis.md` §8 | **embodied** in 4 module docstrings + 73 test invariants + this close-out §3 MEASURED-vs-PROJECTED |

**Net effect**: Wf-78 ships **4 NEW modules + 2 677 LOC + 73 tests PASS** that convert L13-L16 from CANDIDATE gaps (per `r22_l13_l16_gap_analysis.md` §7.1) into **shape-pinned scaffolding** ready for Wf-76 P2 / Wf-77 / Wf-79 / R23-P3 GPU + wet-lab phases. The 4 active R22-P plan items (R22-P1..P4) now have CPU-only initial code in `molmetal_lam/reward/`, each with deterministic smoke tests that pin the algorithm shape without claiming MEASURED numbers.

---

## §5 Honest framing per `party-twenty-spirit §3.5`

| Honest finding | Evidence |
|---|---|
| **CPU-only initial code, no GPU retrain executed** | All 4 modules use NumPy forward-Euler / SHA-256 mock / deterministic synthetic mocks; torchdiffeq + dative-bond SMARTS + GP/q-EI real-data fit deferred to Wf-76 P2 / Wf-77 / Wf-79 |
| **No wet-lab hands touched** | R22-P4 P4.1 LOI/NDA + P4.2 cell culture + P4.3 GP/q-EI fit + P4.4 in-silico vs wet-lab agreement + P4.5 paper §5 claim ALL deferred to R23-P3 (6-mo wet-lab) per `r22_p1_p2_p3_p4_p5_p6_plan.md` §6 |
| **Reachability gate NOT lifted by L13** | The structural blockers (valence-BNF dative-bond rejection, tautomerisation redex exhaustion, sample sparsity) remain; L13 P1.1+P1.2 shape-pinning is a **prerequisite** for the Wf-79 R22-P1 P1.3 GPU-0 sweep that ATTEMPTS the lift |
| **L13-L16 are still CANDIDATE shapes** | Per `pending/r22_l13_l16_gap_analysis.md` §0 + §8; Wf-78 ships CPU-only scaffolding, NOT MEASURED gate-lift numbers |
| **Pattern-match signal counts are NOT quality measures** | 188 L13 hits ≠ "L13 is the right shape"; 70 L16 hits ≠ "L16 will close the wet-lab gap" — regex match density ≠ quality per `gap_analysis.md §8` |
| **R22-P1..P4 PROJECTED, NOT MEASURED** | All 4 active plans ship today as CPU-only docs; MEASURED numbers come after Wf-76 P2 / Wf-77 / Wf-79 / R23-P3 GPU + wet-lab phases |
| **No paper work touched** | Per `pending/84_no_paper_task_without_authorization.md` — no `paper/sections/*.tex`, no `paper/main.tex` rebuilds, no reviewer rebuttal |
| **No R17 surface modified** | `closure.py` + `lam_chem/*` + `reward/lambda_astsr.py` carry from Wf-36/Wf-37/Wf-49 bit-identical; L13-L16 modules live in NEW files (`molmetal_lam/reward/{hierarchical_multimetal,neural_ode_kinetics,pareto_multiobj,wetlab_bayesian}.py`) |
| **4 smoke tests deterministic on `seed=0`** | All 4 modules pin `seed` config; reproduce across runs; SHA-256 mock + NumPy forward-Euler + Das-Dennis simplex + Matern 5/2 all deterministic on integer seed |
| **Per-metal reachability bias is a MOCK convention** | L15 `build_synthetic_metallodrug_pop` Pt=0.9/Au=0.85/Ru=Ir=0.7/Pd=0.6/Os=Rh=0.55 are per-Wf-78 mock anchors, NOT MEASURED reachability — Wf-77 P3.2 GPU-0 sweep is the real measurement |
| **M26 Score-Rep multi-task loss is WIRED, not TRAINED** | L14 `MultiTaskLoss.loss()` returns the 3-term weighted sum; the underlying `R²_static` / `MSE_rate` / `CE_t1/2` are computed from synthetic mock priors, not from real GPU retrain outputs |
| **L15 NSGA-III reference directions are 4-obj p=2 only** | 15 simplex points; `p=3` or higher-p reference directions not shipped today — Wf-77 P3.2 GPU-0 sweep will exercise p=3 if real hypervolume requires |
| **L16 institution table is STATIC** | ORNL/PKU/协和 3-row table; new institutions (e.g. AstraZeneca / Genentech / NIH NCI) not in scope; Wf-76 P2.2 + R23-P3 wet-lab phase adds institutions |

**Per `party-twenty-spirit §3.5`**: This Wf-78 cycle (Phases 1+2+3) ships **honest framing** — no MEASURED numbers fabricated, no PROJECTED numbers labeled as MEASURED, no GPU/wet-lab phases silently claimed ship. The 4 NEW modules are **CPU-only initial scaffolding** with deterministic smoke tests + 73 PASS unit tests; the 4 R22-P plan items (R22-P1..P4) are PROJECTED until Wf-76 P2 / Wf-77 / Wf-79 / R23-P3 GPU + wet-lab phases. The 4 L13-L16 candidates remain **CANDIDATE shapes** awaiting Wf-72 Phase 4 LAMBDA_ROADMAP §4.2 4-row extension (next cycle).

---

## §6 Cross-references

### §6.1 Wf-78 cycle primary deliverables (this cycle ship surface)

- **`molmetal/molmetal_lam/reward/hierarchical_multimetal.py`** (Wf-78 P1, 512 LOC) — L13 outer × inner loop + 8-metal pool + 16 tests PASS
- **`molmetal/molmetal_lam/reward/neural_ode_kinetics.py`** (Wf-78 P2, 981 LOC) — L14 4 ODE drivers + multi-task loss wire-up + 17 tests PASS
- **`molmetal/molmetal_lam/reward/pareto_multiobj.py`** (Wf-78 P3a, 560 LOC) — L15 NSGA-III integration + 4-objective Pareto + 20 tests PASS
- **`molmetal/molmetal_lam/reward/wetlab_bayesian.py`** (Wf-78 P3b, 624 LOC) — L16 GP + q-EI + 3-institution wet-lab loop + 20 tests PASS
- **`molmetal/tests/test_hierarchical_multimetal.py`** (16 tests, 0.12 s wall)
- **`molmetal/tests/test_neural_ode_kinetics.py`** (17 tests, 0.13 s wall)
- **`molmetal/tests/test_pareto_multiobj.py`** (20 tests, 0.09 s wall)
- **`molmetal/tests/test_wetlab_bayesian.py`** (20 tests, 0.09 s wall)
- **`molmetal/reports/wf78_l13_scaffold/synthesis.md`** (Wf-78 P1 ship surface)
- **`molmetal/reports/wf78_l14_scaffold/synthesis.md`** (Wf-78 P2 ship surface)
- **`molmetal/reports/wf78_l15_l16_scaffold/synthesis.md`** (Wf-78 P3 ship surface)
- **`TODO/completed/104_l13_l16_phase1_scaffold.md`** (this file)

### §6.2 Wf-72/Wf-76/Wf-77/Wf-79 cycle dependencies (next-cycle handoff)

- **`TODO/pending/r22_l13_l16_gap_analysis.md`** (Wf-72 P1, 311 lines) — L13-L16 candidate Lambda gaps + 330 scan signals + LAMBDA_ROADMAP §4.2 4-row extension proposal
- **`TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md`** (Wf-72 P2, ~440 lines) — R22-P1..P6 plan items with 4-5 phases each + cross-skill strategy + MEASURED-vs-PROJECTED
- **`molmetal/reports/wf72_5cat_extend/audit_22_to_25.json`** (Wf-72 P3, 426 lines JSON) — 5-cat audit extension 22 → 25 metrics + per-metric audit signal + Q1-Q5 status
- **`TODO/completed/101_r22_plan_l13_l16.md`** (Wf-72 close-out) — Phases 1+2+3 ship surface
- **`TODO/completed/103_todo_final_consolidation.md`** (Wf-76 TODO ledger final) — precedes Wf-78
- **`TODO/completed/101_wf73_p2_ood_permetal.md`** (Wf-73 P2, 125 lines) — per-metal OOD split
- **`TODO/completed/102_wf73_p3_ood_perelement.md`** (Wf-73 P3, 197 lines) — per-element OOD split
- **`TODO/completed/97b_wetlab_prework_docs.md`** (Wf-74 wet-lab pre-work — 3 LOI + 3 SOP)
- **`.claude/skills/mol-metal-neural-ode-kinetics/SKILL.md`** (S10 SKILL.md, Wf-76 P1 ship; 5 paper refs + 17-test smoke) — L14 multi-task loss anchor
- **`.claude/skills/mol-metal-lambda-calculus/SKILL.md`** (closure.py / lam_chem/) — L13 cross-ref
- **`.claude/skills/mol-metal-r18-math/SKILL.md`** (M26 Score-Rep / T1 SE3 SB / T7 SB-Bass) — L14 cross-ref
- **`.claude/skills/mol-metal-click-chemistry/SKILL.md`** (L13 dative-bond SMARTS deferred to Wf-79) — L13 cross-ref
- **`.claude/skills/mol-metal-anticancer/SKILL.md`** (kinetic trans-effect / redox cycling) — L14 cross-ref
- **`.claude/skills/party-twenty-spirit/SKILL.md` §3.5** (0-tolerance honest framing) — applied throughout

### §6.3 R20/R21 close-out files (parent context)

- `TODO/completed/r20_status_ledger_2026_09_26.md` (Wf-70 P1, 308 lines, 50 sub-task R20 ledger)
- `TODO/completed/100_r20_r21_consolidation.md` (Wf-70 P2, L1-L8 → L1-L12 expansion)
- `TODO/completed/100_wf71_p3_wf49_cross_round2.md` (Wf-71 P3, F5_RDKIT_2D cross-dataset macro R² = +0.1248)
- `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` §1.1 (static ASTSR R² ≈ -0.02 collapse detector at epoch 4 — the L14 kinetics anchor)
- `TODO/completed/96_r20_p5_closure_catherine.md` §3.2 (reachability 0.5962 NOT gate 0.95 — the L13 anchor)
- `TODO/completed/95_r20_p4_3way_100pocket.md` (3-way + 100-pocket L10 baseline)
- `TODO/completed/91_wf41_3way_integration.md` (L10 3-way R²=-0.3595 ± 0.2806 baseline)
- `TODO/completed/87j_wf44_root_cause_fixes.md` C5 (target distribution extreme TmQM-heavy)

### §6.4 R22-P plan files + LAMBDA_ROADMAP + INDEX

- `molmetal/reports/wf72_r22_plans/R22-P1_template.json` (R22-P1 P1.1..P1.5 template)
- `molmetal/reports/wf72_r22_plans/R22-P2_template.json` (R22-P2 P2.1..P2.4 template)
- `molmetal/reports/wf72_r22_plans/R22-P3_template.json` (R22-P3 P3.1..P3.3 template)
- `molmetal/reports/wf72_r22_plans/R22-P4_template.json` (R22-P4 P4.1..P4.5 template — 6-mo wet-lab)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (12-row L1-L12 trajectory, proposed L13-L16 4-row extension per `gap_analysis.md §6`)
- `TODO/INDEX.md` §17.5 (TODO 100-107 R22 plan items) + §17.6 (L13-L16 candidate Lambda gaps, 4 candidates, 330 scan signals)

### §6.5 Paper-gate + skill rules (CRITICAL — no exceptions)

- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate, CRITICAL — honored)
- `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 (0-tolerance honest framing)
- `.claude/skills/mol-metal-paper-writing/SKILL.md` §3 (MEASURED/PROJECTED tag discipline)
- `.claude/skills/mol-metal-anticancer/SKILL.md` §3 (tier-1 metric honesty for m18/m19/m20)

---

## §7 Net effect summary

| Deliverable | LOC | Tests | Status | Deferral |
|---|---:|---:|---|---|
| `hierarchical_multimetal.py` (L13) | 512 | 16/16 PASS | ✅ SHIP | Wf-79 R22-P1 P1.3 GPU-0 sweep |
| `neural_ode_kinetics.py` (L14) | 981 | 17/17 PASS | ✅ SHIP | Wf-76 P2.1 GPU-1 torchdiffeq retrain |
| `pareto_multiobj.py` (L15) | 560 | 20/20 PASS | ✅ SHIP | Wf-77 R22-P3 P3.2 GPU-0 sweep |
| `wetlab_bayesian.py` (L16) | 624 | 20/20 PASS | ✅ SHIP | R23-P3 6-mo wet-lab + 2 wk CPU BO |
| **Aggregate (4 NEW modules)** | **2 677** | **73/73 PASS** | ✅ **SHIP** | 4 GPU/wet-lab phases deferred to Wf-76 P2 / Wf-77 / Wf-79 / R23-P3 |

**One-line verdict (per `party-twenty-spirit §3.5`)**: Wf-78 ships **4 NEW L13-L16 modules + 73 tests PASS** that pin the algorithm shape for downstream GPU + wet-lab phases — but **0 NEW MEASURED numbers**: this is a CPU-only docs cycle, not a GPU/wet-lab ship. The 4 R22-P plan items (R22-P1..P4) now have shape-pinned scaffolding ready for Wf-76 P2 / Wf-77 / Wf-79 / R23-P3, which is when MEASURED gate-lift numbers will arrive.

**Net effect on R22 readiness**: ✅ TODO ledger final (Wf-76) + ✅ 4 NEW L13-L16 modules + ✅ 73 tests PASS (Wf-78) + ⏳ R22-P1..P4 GPU retrain (deferred to Wf-77 / Wf-79) + ⏳ R22-P4 wet-lab cooperation (deferred to R23-P3).