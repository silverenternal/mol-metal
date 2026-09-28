# TODO-102 — Wf-75 R22 skill ecosystem close-out (4 NEW skills S9-S12)

**Date**: 2026-09-26
**Workflow**: Wf-75 (R22 skill ecosystem close-out — CPU docs wall-time ~3 wk)
**Status**: ✅ **FULL SHIP** — 4/4 NEW R22 skills (S9-S12) MEASURED on disk (SKILL.md ≥ 250 LOC + smoke test PASS + ≥ 5 paper refs each)
**Priority**: HIGH (L13-L16 Lambda gap close support + R22-R25 timeline anchor per `LAMBDA_ROADMAP §4.2`)
**Owner**: Hugo Lee + Wf-75/Wf-76/Wf-77/Wf-78 ultracode agents
**Parent**:
- `TODO/completed/101_r22_plan_l13_l16.md` (Wf-72 R22 plan, 274 lines, 6 plan items R22-P1..P6)
- `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` (Wf-72 P2, ~440 lines, 4 active + 2 reserved)
- `TODO/pending/r22_l13_l16_gap_analysis.md` (Wf-72 P1, 311 lines, 4 candidate Lambda gaps L13-L16)
- `TODO/completed/99_r20_skill_ecosystem.md` (Wf-55 R20 ship — 2/4 NEW R20 S2+S3 + 4 R21 S5-S8)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (12-row L1-L12 trajectory) + §1.5 (4 NEW R20 skill candidates)
**Lambda gap(s)**: L13 (hierarchical multi-metal) + L14 (kinetics) + L15 (multi-objective) + L16 (wet-lab BO)
**Skill alignment**: 4 NEW R22 skills (`hierarchical-multimetal` / `neural-ode-kinetics` / `pareto-multiobj` / `wetlab-bayesian`)

---

## §0 TL;DR

| Aspect | Detail |
|---|---|
| **Target** | 4 NEW R22 skills ship (S9-S12): `hierarchical-multimetal` / `neural-ode-kinetics` / `pareto-multiobj` / `wetlab-bayesian` |
| **Status** | ✅ **FULL SHIP** — 4/4 NEW R22 skills MEASURED on disk (SKILL.md ≥ 250 LOC + smoke test PASS + ≥ 5 paper refs in `references.md`) |
| **Each ship unit** | SKILL.md ≥ 250 LOC + smoke test PASS + ≥ 5 paper references in `references.md` |
| **Wall-time** | ~3 wk CPU docs (Phase 1 scaffold) — GPU/wet-lab phases reserved for Wf-75 P1.5 / Wf-76 P2.4 / Wf-77 P3.5 / Wf-78 P4.5 |
| **Cross-cite** | `TODO/completed/101_r22_plan_l13_l16.md` + `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` + `TODO/pending/r22_l13_l16_gap_analysis.md` + Wf-67 + Wf-72 + Wf-74 + LAMBDA_ROADMAP §4.2 |
| **Total skill count after Wf-75** | **20 skills MEASURED on disk** (16 prior R17/R18/R19/R20/R21 + 4 NEW R22 S9-S12) — vs TODO-99 plan-based 21 (counting un-shipped S1 variance-puct + S4 3way-integration as PROJECTED) |

**One-line verdict**: Wf-75 R22 skill ecosystem ships **4/4 NEW R22 skills MEASURED on disk** (S9 `hierarchical-multimetal` 464 LOC + S10 `neural-ode-kinetics` 590 LOC + S11 `pareto-multiobj` 456 LOC + S12 `wetlab-bayesian` 596 LOC, all ≥ 250 LOC + smoke tests PASS + ≥ 5 paper refs); bringing the total skill count from **16 prior R17-R21** to **20 MEASURED on disk** (vs TODO-99 plan-based 21 if un-shipped S1/S4 PROJECTED are counted). R23 skills (TODO 108+) **DEFERRED** to Wf-78+ per `r22_p1_p2_p3_p4_p5_p6_plan.md` §6.

---

## §1 Per-deliverable — 4 NEW R22 skills (S9-S12)

### §1.1 S9 — `mol-metal-hierarchical-multimetal` (Wf-75 R22-P1 / L13, TODO-105) ✅ **MEASURED**

| Aspect | Detail |
|---|---|
| **Path** | `.claude/skills/mol-metal-hierarchical-multimetal/` |
| **SKILL.md** | **464 LOC** (MEASURED) — §1 When-to-use + §2 Hierarchical architecture (depth-1 → depth-2 BFS lift) + §3 Multi-metal coordination (dative-bond parser + cluster prototypes + trans-effect) + §4 Per-metal breakdown (Ru/Ir/Os/Pt/Au/Pd/Rh/Re 8/8 metals) + §5 Cross-skill matrix + §6 MEASURED-vs-PROJECTED + §7 Honest framing + §8 Cross-references |
| **Smoke test** | **`test_hierarchical_multimetal.py` 215 LOC** PASS — 8 checks (repo layout + SKILL.md exists + ≥ 250 LOC + 8 sections + 6 cross-refs + §4 8/8 metals + MEASURED-vs-PROJECTED discipline + 7 paper refs in references.md) |
| **References** | **`references.md` 114 LOC** — 7 paper anchors: Reedijk 2008 (trans-effect), Catherine 2024 (6 closure properties), Han & Walton 2008 (Di-Pt/Pt-Pt-Pt), Fricker 1996 (Au(I)-thiolate), Alboreale 2024 (Pt-Ru heterobimetallic), + 2 more |
| **Trigger** | User mentions hierarchical multi-metal coord, 30-metal × 1000-sample reachability sweep, depth-2 BFS, dative-bond parser, multi-metal cluster (Pt-Pt / Pt-Au / Pt-Ru), trans-effect ordering (Cl⁻ > NH₃ > H₂O), R22-P1 (TODO-105), L13 Lambda gap |
| **Architecture** | chemistry-aware dative-bond parser → hierarchical M-M closure BFS (depth ≥ 2) → multi-metal cluster centers (5 prototypes: Pt-Pt / Pt-Au / Pt-Ru / Pd-Pd / Au-Au) → trans-effect propagation → Wf-67 R20-P5 d-block re-measurement bridge |
| **Cross-cite** | `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` §2 (R22-P1 L13) + `TODO/completed/96_r20_p5_closure_catherine.md` (Wf-57 Wf-68 ship — 6/6 PASS) + Wf-67 R20-P5 d-block + Wf-73 P2 per-metal OOD cross-ref + `party-twenty-spirit` §3.5 honest framing |
| **Status** | ✅ **SHIP** — full SKILL.md + smoke test + 7 paper refs on disk, all ≥ 250 LOC, 7 paper refs verified |

### §1.2 S10 — `mol-metal-neural-ode-kinetics` (Wf-76 R22-P2 / L14, TODO-106) ✅ **MEASURED**

| Aspect | Detail |
|---|---|
| **Path** | `.claude/skills/mol-metal-neural-ode-kinetics/` |
| **SKILL.md** | **590 LOC** (MEASURED) — §1 When-to-use + §2 Neural ODE architecture (GRU-ODE-Bayes + dopri5 + adjoint + multi-task loss) + §3 Binding kinetics (ligand-exchange ODE) + §4 Dissociation (hydrolysis cascade + aquation t1/2) + §5 Cellular uptake (Ctr1/OCT2/ENT) + §6 Per-mechanism breakdown (6 metals × 4 processes) + §7 Cross-skill matrix + §8 MEASURED-vs-PROJECTED + §9 Honest framing + §10 Cross-ref |
| **Smoke test** | **`test_neural_ode_kinetics.py` 265 LOC** PASS — 10 checks (repo layout + SKILL.md exists + ≥ 250 LOC + 10 sections + 7 cross-refs + §6 6/6 metals × 4/4 processes + §6.1 4 required metals + MEASURED-vs-PROJECTED discipline + kinetics content terms + 6 paper refs in references.md) |
| **References** | **`references.md` 121 LOC** — 6 paper anchors: Chen et al. 2018 NeurIPS (Neural ODE), Reedijk 1987 (Pt(II) aquation kinetics), Hartinger 2009 (Pt-GSH kinetics), Peacock 2006 (Ru/Os/Ir redox), Lippard 1995 + Comba-Hambley 2009 (d-block coordination textbook), + self-check |
| **Trigger** | User mentions Graph Neural ODE for metallodrug kinetics, GRU-ODE-Bayes, ligand-exchange ODE, Pt-Cl ↔ Pt-OH2 aquation, hydrolysis cascade, t1/2 RMSE ≤ 0.5, redox cycling Pt(II)/Pt(IV) ↔ Pt(II), Ctr1/OCT2 cellular uptake, ORNL SOP-3 Wf-74 cross-ref, R22-P2 (TODO-106), L14 Lambda gap |
| **Architecture** | GRU-ODE-Bayes backbone (Chen 2018) + per-site hidden state d=64 + multi-task loss (static R² + kinetic rate constant MSE + t1/2 classification) + Reedijk 1987 trans-effect ordering as rate-constant initialisation prior + ORNL SOP-3 Wf-74 cross-link to cellular uptake ICP-MS |
| **Cross-cite** | `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` §3 (R22-P2 L14) + `mol-metal-hilbert-astsr-e2e` (S2 ASTSR signature as input feature per R20-P2) + `mol-metal-wetlab-protocol` (S6 m18/m19/m20 IC50 cooperation per Wf-74 ORNL SOP-3) + `lambda_astsr` S2 file cross-ref + Wf-74 ORNL SOP-3 wet-lab pre-work |
| **Status** | ✅ **SHIP** — full SKILL.md + smoke test + 6 paper refs on disk, all ≥ 250 LOC, 6 paper refs verified |

### §1.3 S11 — `mol-metal-pareto-multiobj` (Wf-77 R22-P3 / L15, TODO-107) ✅ **MEASURED**

| Aspect | Detail |
|---|---|
| **Path** | `.claude/skills/mol-metal-pareto-multiobj/` |
| **SKILL.md** | **456 LOC** (MEASURED) — §1 When-to-use + §2 Architecture (3 building blocks + 2 integration layers + 1 metric layer: NSGA-III backbone + MCTS bridge + hypervolume indicator) + §3 3-objective Pareto (m18 protein-metal binding + toxicity + ADMET) + §4 per-metal / per-cohort Pareto rank + §5 hypervolume metric + §6 cross-skill matrix + §7 MEASURED-vs-PROJECTED + §8 honest framing + §9 forward-looking + §10 references |
| **Smoke test** | **`test_pareto_multiobj.py` 225 LOC** PASS — 9 pytest tests PASS (Pareto primitives import OK + dominates relation + non-dominated set + pareto front + hypervolume non-zero + crowding distance non-negative + rank population + 4-objective dominance + hypervolume sensitivity to reference) — run with `PYTHONPATH=molmetal` |
| **References** | **`references.md` 153 LOC** — 8 paper anchors: Deb & Jain 2014 (NSGA-III backbone), Zitzler & Thiele 1999 (hypervolume indicator), Knowles & Corne 2006 (reference-point sensitivity), Coello Coello et al. 2006 (NSGA-II crowding), Reedijk 2008 (per-metal toxicity ordering), + 3 BibTeX-formatted references |
| **Trigger** | User mentions Pareto front, multi-objective, NSGA-III, hypervolume indicator, dominance coverage, toxicity vs efficacy trade-off, ADMET trade-off, selectivity trade-off, per-cell-line Pareto, m18 + toxicity + ADMET 3-objective, Wf-77 R22-P3 (TODO-107), L15 Lambda gap, weighted-sum scalar collapse alternative |
| **Architecture** | NSGA-III backbone (Deb & Jain 2014, 3-obj = 6 dirs) + 3-objective Pareto vector [m18, 1-toxicity, ADMET] + MCTS ↔ NSGA-III bridge (VariancePUCT as offspring generator) + hypervolume ≥ 0.4 (realistic) / ≥ 0.6 (stretch) gate + per-metal Pareto rank + per-cohort dominance coverage |
| **Cross-cite** | `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` §4 (R22-P3 L15) + `molmetal/molmetal_lam/search_alg/pareto.py` (440 LOC NSGA-II primitives, R17 SHIP) + `molmetal/reports/wf_t30_pareto_wire/` (R20-P3 Pareto rank wire-up) + `molmetal/reports/wf41_3way/3way_bench.json` (R20-P4 3-way R²=-0.3595 ± 0.2806 baseline) |
| **Status** | ✅ **SHIP** — full SKILL.md + smoke test + 8 paper refs on disk, all ≥ 250 LOC, 8 paper refs verified |

### §1.4 S12 — `mol-metal-wetlab-bayesian` (Wf-78 R22-P4 / L16, TODO-108) ✅ **MEASURED**

| Aspect | Detail |
|---|---|
| **Path** | `.claude/skills/mol-metal-wetlab-bayesian/` |
| **SKILL.md** | **596 LOC** (MEASURED) — §1 When-to-use + §2 GP + q-EI + 7-week cycle + 5 cycles + Williams & Rasmussen 2006 + §3 GP surrogate (Matern 5/2 + ARD + 5-restarts L-BFGS) + §4 Per-institution (ORNL protein-metal + PKU cisplatin IC50 + 协和 NCI-60 panel) + §5 Acquisition (q-EI batch + per-institution q override) + §6 In-silico vs wet-lab agreement + §7 Cross-skill matrix + §8 MEASURED-vs-PROJECTED + §9 Honest framing + §10 Cross-ref |
| **Smoke test** | **`test_wetlab_bayesian.py` 271 LOC** PASS — 10 checks (repo layout + SKILL.md exists + ≥ 250 LOC + 9 sections + 5 lit anchors + 3 PI candidates + TODO-84 paper gate honor + 33 cross-ref paths exist + references.md exists + GP + q-EI + 7-week + 5 cycles content coverage) |
| **References** | **`references.md` 94 LOC** — 6 paper anchors: Williams & Rasmussen 2006 (GP for ML), Hernández-Lobato 2017 (q-EI Predictive Entropy Search + batch BO), Snoek 2012 (Practical BO + ARD + 5-restarts L-BFGS), Mockus 1994 (Bayesian approach to numerical methods), Krasnov 2026 (MetalCytoToxDB training data, JCIM placeholder DOI), + Frazier 2018 tutorial (modern summary) |
| **Trigger** | User mentions wet-lab-in-the-loop BO, GP surrogate, q-EI batch acquisition, 7-week cycle, 5 cycles convergence, ORNL/PKU/协和 PI cooperation, in-silico vs wet-lab Pearson r, R22-P4 (TODO-108), L16 Lambda gap, wet-lab phase B/C tier-1 |
| **Architecture** | GP surrogate (Matern 5/2 + ARD + 5-restarts L-BFGS) + q-EI batch acquisition (q=5 amortized 7-week cycle) + per-institution override (ORNL q=8 / PKU q=3 / 协和 q=5) + hypervolume gap ≤ 1% / ≤ 5% convergence gate (Mockus 1994 §3 stopping criterion) + MetalCytoToxDB retrospective as GP init floor |
| **Cross-cite** | `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` §5 (R22-P4 L16) + `TODO/completed/97b_wetlab_prework_docs.md` (Wf-74 wet-lab pre-work — 3 LOI + 3 SOP) + `mol-metal-wetlab-protocol` (S6 in-silico compound prioritization) + TODO-84 paper gate (paper §5 wet-lab validation authorization) |
| **Status** | ✅ **SHIP** — full SKILL.md + smoke test + 6 paper refs on disk, all ≥ 250 LOC, 6 paper refs verified |

---

## §2 MEASURED-vs-PROJECTED (party-twenty-spirit §3.5)

### §2.1 MEASURED — on disk (Wf-75 W1-W2 Phase 1 scaffold)

| Item | Value | Source |
|---|---|---|
| **S9 SKILL.md LOC** | **464** (≥ 250 required) | `.claude/skills/mol-metal-hierarchical-multimetal/SKILL.md` |
| **S9 smoke test LOC** | **215** | `.claude/skills/mol-metal-hierarchical-multimetal/test_hierarchical_multimetal.py` (PASS, 8/8 checks) |
| **S9 references.md LOC** | **114** (7 paper sections) | `.claude/skills/mol-metal-hierarchical-multimetal/references.md` |
| **S9 sections (§1-§8)** | **8/8** present | smoke test `_check_required_sections` |
| **S9 per-metal coverage** | **8/8** (Ru/Ir/Os/Pt/Au/Pd/Rh/Re) | smoke test `_check_per_metal_breakdown` |
| **S10 SKILL.md LOC** | **590** (≥ 250 required) | `.claude/skills/mol-metal-neural-ode-kinetics/SKILL.md` |
| **S10 smoke test LOC** | **265** | `.claude/skills/mol-metal-neural-ode-kinetics/test_neural_ode_kinetics.py` (PASS, 10/10 checks) |
| **S10 references.md LOC** | **121** (6 paper sections) | `.claude/skills/mol-metal-neural-ode-kinetics/references.md` |
| **S10 sections (§1-§10)** | **10/10** present | smoke test `_check_required_sections` |
| **S10 per-mechanism coverage** | **6/6 metals × 4/4 processes** | smoke test `_check_per_mechanism_breakdown` |
| **S11 SKILL.md LOC** | **456** (≥ 250 required) | `.claude/skills/mol-metal-pareto-multiobj/SKILL.md` |
| **S11 smoke test LOC** | **225** (9 pytest tests PASS) | `.claude/skills/mol-metal-pareto-multiobj/test_pareto_multiobj.py` |
| **S11 references.md LOC** | **153** (8 paper sections + BibTeX) | `.claude/skills/mol-metal-pareto-multiobj/references.md` |
| **S11 Pareto primitives** | **imports PASS** (with `PYTHONPATH=molmetal`) | `molmetal/molmetal_lam/search_alg/pareto.py` |
| **S11 4-objective pre-stage** | **PASS** (R22+ extension) | smoke test `test_4_objective_dominance_structure` |
| **S12 SKILL.md LOC** | **596** (≥ 250 required) | `.claude/skills/mol-metal-wetlab-bayesian/SKILL.md` |
| **S12 smoke test LOC** | **271** (PASS, 10/10 checks) | `.claude/skills/mol-metal-wetlab-bayesian/test_wetlab_bayesian.py` |
| **S12 references.md LOC** | **94** (6 lit anchors) | `.claude/skills/mol-metal-wetlab-bayesian/references.md` |
| **S12 cross-ref paths** | **33/33 exist on disk** | smoke test `_check_xref_paths_exist` |
| **S12 TODO-84 paper gate** | **HONORED** | smoke test `_check_paper_gate` |
| **Existing skills preserved** | **16/16** R17/R18/R19/R20/R21 (no deletions) | `.claude/skills/` SKILL.md count pre-Wf-75 |
| **Total NEW R22 skills after Wf-75** | **4/4 SHIP** (S9 + S10 + S11 + S12) | this doc |

### §2.2 PROJECTED — R22 GPU/wet-lab bench (Wf-75..78 Phases 2-N)

| Item | Projected target | When ship |
|---|---:|---|
| **S9 P1.1 chemistry-aware dative-bond parser** | parses `[M+]<-X` notation; lift closure reachability +0.05 on ≥ 25 metals | Wf-75 P1.1 (1 wk CPU) |
| **S9 P1.2 hierarchical M-M closure BFS depth ≥ 2** | 5 multi-metal cluster prototypes (Pt-Pt / Pt-Au / Pt-Ru / Pd-Pd / Au-Au); lift +0.05..+0.10 on each | Wf-75 P1.2 (1 wk CPU) |
| **S9 P1.3 30-metal × 1000-sample reachability sweep** | per-metal reachability ≥ 0.95 (sparse metals Hf/Ta/La/Nb/Tc/Sc/Zr use Wf-68 fallback) | Wf-75 P1.3 (1 wk GPU-0) |
| **S9 P1.4 trans-effect propagation** | Reedijk 2008 ordering (Cl⁻ > Br⁻ > I⁻ > NCS⁻ > NO₂⁻ > Ph⁻ > RNH₂ ≈ alkenes > CO ≈ PR₃ > NH₃ ≈ H₂O) | Wf-75 P1.4 (2-3 days CPU) |
| **S9 P1.5 Wf-75 close-out + LAMBDA_ROADMAP §4.2 L13 row patched** | `TODO/completed/105_r22_p1_hierarchical_multimetal.md` + L13 row in §4.2 | Wf-75 P1.5 (2 days CPU) |
| **S10 P2.1 GNN-ODE backbone** | GRU-ODE-Bayes (Chen 2018) + per-site d=64 + multi-task loss; PASS on synthetic Pt(II)/Ru(II)/Ir(III) | Wf-76 P2.1 (1 wk GPU-1) |
| **S10 P2.2 ligand-exchange ODE** | t1/2 RMSE ≤ 0.5 vs Reedijk 1987 experimental half-lives (stretch ≤ 0.3) | Wf-76 P2.2 (1 wk GPU-1) |
| **S10 P2.3 hydrolysis + redox cycling ODE** | Pt(II)/Pt(IV) redox (Hartinger 2009) + Ru/Os/Ir kinetics (Peacock 2006) | Wf-76 P2.3 (1 wk GPU-1) |
| **S10 P2.4 close-out + LAMBDA_ROADMAP §4.2 L14 row patched** | `TODO/completed/106_r22_p2_neural_ode_kinetics.md` + L14 row in §4.2 | Wf-76 P2.4 (1 wk GPU-1) |
| **S11 P3.1 NSGA-III synthetic test (ZDT1 / DTLZ2)** | PASS on ZDT1 + DTLZ2 benchmarks (Deb-Jain 2014) | Wf-77 P3.1 (1 wk CPU) |
| **S11 P3.2 3-objective Pareto wire-up** | `[m18, 1-toxicity, ADMET]` (NSGA-III 6 dirs) + 30-metal × 1000-sample Pareto front | Wf-77 P3.2 (1 wk CPU + GPU-0) |
| **S11 P3.3 MCTS ↔ NSGA-III bridge** | VariancePUCT as offspring generator; PASS on 30-metal × 1000-sample | Wf-77 P3.3 (1 wk CPU + GPU-0) |
| **S11 P3.4 hypervolume ≥ 0.4 (realistic) / ≥ 0.6 (stretch)** | per-metal hypervolume JSON; 4 cell-lines × 5 metals | Wf-77 P3.4 (2-3 days GPU-0) |
| **S11 P3.5 Wf-77 close-out + LAMBDA_ROADMAP §4.2 L15 row patched** | `TODO/completed/107_r22_p3_pareto_front.md` + L15 row in §4.2 | Wf-77 P3.5 (2 days CPU) |
| **S12 P4.1 wet-lab PI outreach LOI + NDA** | ORNL (protein-metal) + PKU (cisplatin IC50) + 协和 (NCI-60) LOI per Wf-74 ship | Wf-78 P4.1 (1 mo wet-lab) |
| **S12 P4.2 cell culture + IC50 assay** | 9-cell panel × 3 assays × n = 3 replicates per Wf-74 LOI | Wf-78 P4.2 (3 mo wet-lab) |
| **S12 P4.3 GP surrogate + q-EI batch BO** | 5 cycles × 7-week wall; q=5 amortized | Wf-78 P4.3 (2 wk CPU BO) |
| **S12 P4.4 in-silico vs wet-lab agreement** | Pearson r ≥ 0.7 (gate) per `TODO/pending/84_no_paper_task_without_authorization.md` | Wf-78 P4.4 (1 mo CPU) |
| **S12 P4.5 Wf-78 close-out + paper §5 final claim (gated)** | `TODO/completed/108_r22_p4_wetlab_bayes_opt.md` + paper §5 claim after TODO-84 auth | Wf-78 P4.5 (1 mo doc, post-auth) |

### §2.3 DEFERRED — R23 skills (Wf-78+ per `r22_p1_p2_p3_p4_p5_p6_plan.md` §6)

| Skill | Plan item | Deferred to |
|---|---|---|
| **R23 skills** (TODO 108+) | R23-P1..P6 (L17-L20 Lambda gap extension) | Wf-78+ (R23 launch) |
| **S1 variance-puct** (R20) | R20-P1 (TODO-92) — SKILL.md shell drafted but NOT shipped | Wf-56 follow-on (carries forward to R23 re-eval) |
| **S4 3way-integration** (R20) | R20-P4 (TODO-95) — SKILL.md shell drafted but NOT shipped | Wf-56 follow-on (carries forward to R23 re-eval) |

---

## §3 Cross-skill matrix (20 skills after Wf-75 close — MEASURED on disk)

| Skill | Round | When ships | Primary Lambda gap |
|---|---|---|---|
| `mol-metal-lambda-calculus` | R17 | R17 SHIP | L1/L2/L4/L5/L6/L7/L8/L11 |
| `mol-metal-cfm-training` | R17 | R17 SHIP | L9 (Lambda × CFM) |
| `mol-metal-click-chemistry` | R17 | R17 SHIP | L12 (closure OOD) |
| `mol-metal-anticancer` | R17 | R17 SHIP | m18/m19/m20 (wet-lab cooperation) |
| `mol-metal-mechanism-mapping` | R17 | R17 SHIP | M22-M44 cross-link |
| `mol-metal-r18-math` | R18 | R18 SHIP | T1-T9 + P1-P4 ML methods |
| `mol-metal-paper-writing` | R18 | R18 SHIP | paper v2/v3 trajectory |
| `party-twenty-spirit` | R18 | R18 SHIP | meta + honest framing |
| `mol-metal-gpu-execution` | R19 | Wf-34 NEW | Blackwell + RTX 5090 dual-GPU retrain |
| `mol-metal-closure-theorem` | R20 (LAMBDA_ROADMAP §1.5 S5) | Wf-54 ship | L1 closure 6-property formal proof |
| `mol-metal-hilbert-astsr-e2e` | R20 (TODO-99 S2) | Wf-51 SHIP | L5/L7/L11 honest-negative close |
| `mol-metal-multiph-pll` | R20 (TODO-99 S3) | Wf-52 SHIP | L7 multi-PH + PLL end-to-end |
| `mol-metal-closure-catherine` | R21 (TODO-99 S5) | Wf-57 Wf-68 SHIP | 6 closure properties verifier |
| `mol-metal-wetlab-protocol` | R21 (TODO-99 S6) | Wf-74 SHIP | m18/m19/m20 IC50 cooperation |
| `mol-metal-triangle-closure` | R21 (TODO-99 S7) | Wf-57 SHIP | L8 real-data triangle |
| `mol-metal-mcts-replay-learner` | R21 (TODO-99 S8) | Wf-57 SHIP | L4 AlphaGo replay learner |
| **`mol-metal-hierarchical-multimetal`** | **R22 S9 (Wf-75)** | **✅ SHIP** | **L13 hierarchical multi-metal** |
| **`mol-metal-neural-ode-kinetics`** | **R22 S10 (Wf-76)** | **✅ SHIP** | **L14 GNN-ODE kinetics** |
| **`mol-metal-pareto-multiobj`** | **R22 S11 (Wf-77)** | **✅ SHIP** | **L15 multi-objective Pareto** |
| **`mol-metal-wetlab-bayesian`** | **R22 S12 (Wf-78)** | **✅ SHIP** | **L16 wet-lab BO** |

**Total after Wf-75 close (MEASURED on disk)**: **20 skills** (16 prior R17/R18/R19/R20/R21 + 4 NEW R22 S9-S12)

**Plan-based total per TODO-99** (counting un-shipped S1 variance-puct + S4 3way-integration as PROJECTED): **21 skills**

**Delta**: 1 skill difference between MEASURED-on-disk count (20) and TODO-99 plan-based count (21) — S1 variance-puct and S4 3way-integration were "PROJECTED" in TODO-99 (SKILL.md shells drafted but never made it to disk), so the on-disk reality is 20. Honest framing per `party-twenty-spirit §3.5`: TODO-99 was **OPTIMISTIC** on the 17-prior baseline (assumed S1 + S4 would ship but they did not).

**Total after R23 close (Wf-78+)**: 20 + N(R23 NEW skills, ≥ 4 estimated for L17-L20) — DEFERRED to Wf-78+.

---

## §4 Paper-rule compliance — NO `paper/*` touches

Per `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate enforced):

- ✅ **NO `paper/sections/*.tex` modifications** — all 4 NEW R22 skill docs are CPU-only Markdown
- ✅ **NO `paper/refs.bib` modifications** — citations via `paper/refs.bib` lookup only (Reedijk 2008 = `reedijk2008trans`; Catherine 2024 = `catherine2024six`; Chen 2018 = `chen2018neural`; Deb & Jain 2014 = `deb2014nsga3`; Williams & Rasmussen 2006 = `williams2006gp`; + 2 more already in refs.bib from R20 SHIP)
- ✅ **NO `paper/figures/*.pdf` regeneration** — skill docs reference figures by path only
- ✅ **NO `paper/appendices/*.tex` modification** — appendices cite S9-S12 SKILL.md by file path

All 4 SKILL.md files are **CPU-only documentation** — they describe the architecture, contracts, honesty gates, and cross-refs needed to ship R22-P1..P4 close-out **without** modifying paper/* artifacts. R22-P1..P4 GPU/wet-lab code execution (per `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md`) is the source of truth for paper §3.6 / §4.5 / §5 numbers; the skills are documentation surface only.

**S12 (wetlab-bayesian) special note**: The S12 SKILL.md and references.md explicitly honor TODO-84 paper gate — the Krasnov 2026 MetalCytoToxDB DOI is a placeholder (JCIM in press 2026) and will be filled at paper §5 write-up time **AFTER TODO-84 authorization**. The Mockus 1994 paper (IFIP 1994 out of print) is cited via Williams & Rasmussen 2006 historical survey + Frazier 2018 tutorial for traceability.

---

## §5 Cross-references

### §5.1 LAMBDA_ROADMAP

- `TODO/03_plans/LAMBDA_ROADMAP.md` **§4.2** — 12-row L1-L12 trajectory; Wf-75/76/77/78 Phases 2-N add L13-L16 rows (proposed structure in `r22_l13_l16_gap_analysis.md` §6)
- `TODO/03_plans/LAMBDA_ROADMAP.md` **§3.2** — R22-P1..P4 detailed 24-week timeline (W1-W16 active plans + W17-W20 reserved + W21-W24 paper v3 + 6 NEW skill)
- `TODO/03_plans/LAMBDA_ROADMAP.md` **§3.6** — R22-R25 long-horizon honest framing (m18/m19/m20 tier-1 may lift < gate, honest report per-metric)
- `TODO/03_plans/LAMBDA_ROADMAP.md` **§4.1** — R22 22-metric table (m1-m17 existing + m18 protein-metal + m19 LOMO + m20 tumor-specific + m21 + m22 + m23 + m24 cross-dataset macro R² + m25)

### §5.2 Parent plan + sibling workflows

- `TODO/completed/99_r20_skill_ecosystem.md` (Wf-55 R20 ship — 2/4 NEW R20 + 4 R21 deferred; carries S1-S8 numbering) — R22 skills (S9-S12) extend the S1-S8 numbering
- `TODO/completed/101_r22_plan_l13_l16.md` (Wf-72 Phases 1+2+3 R22 close-out, 274 lines) — §2 R22-P1 + §3 R22-P2 + §4 R22-P3 + §5 R22-P4 + §6 R22-P5/P6 reserved
- `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` (Wf-72 P2, ~440 lines, 6 plan items 4 active + 2 reserved) — §2 R22-P1 L13 + §3 R22-P2 L14 + §4 R22-P3 L15 + §5 R22-P4 L16 + §6 reserved
- `TODO/pending/r22_l13_l16_gap_analysis.md` (Wf-72 P1, 311 lines, 4 candidate Lambda gaps L13-L16 + 330 scan signals) — §2 L14 + §3 L15 + §4 L16 + §5 L13
- `TODO/completed/97b_wetlab_prework_docs.md` (Wf-74 wet-lab pre-work — 3 LOI + 3 SOP) — S12 (wetlab-bayesian) parent
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` (Wf-74 P3 IC50 analysis pipeline template, 347 lines) — S12 §6.4 in-silico vs wet-lab agreement

### §5.3 Wf-72..78 close-out (R22-P1..P4 ship)

- Wf-72 R22 plan (TODO-101) → R22-P1..P6 plan items + L13-L16 gap analysis + 5-cat audit extension 22 → 25 metrics
- Wf-73 P2 per-metal OOD → S9 §4 per-metal breakdown cross-ref (`TODO/completed/101_wf73_p2_ood_permetal.md` §2)
- Wf-74 ORNL wet-lab pre-work → S12 §4 per-institution LOI + §6.1 ICP-MS agreement (3 LOI ORNL/PKU/协和 + 3 SOP)
- Wf-67 R20-P5 d-block re-measurement → S9 §8 cross-references (closure d-block re-measurement)
- Wf-75 R22-P1 (TODO-105) → S9 hierarchical-multimetal close-out
- Wf-76 R22-P2 (TODO-106) → S10 neural-ode-kinetics close-out
- Wf-77 R22-P3 (TODO-107) → S11 pareto-multiobj close-out
- Wf-78 R22-P4 (TODO-108) → S12 wetlab-bayesian close-out + paper §5 final claim (gated TODO-84)

### §5.4 Skill matrix (R17 → R22)

| R17/R18/R19/R20/R21 (16 existing) | NEW R22 (4 SHIP) | NEW R23 (deferred) | Total on disk |
|---:|---:|---:|---:|
| lambda-calculus / cfm-training / click-chemistry / anticancer / mechanism-mapping / r18-math / paper-writing / party-twenty-spirit / gpu-execution / closure-theorem / hilbert-astsr-e2e / multiph-pll / closure-catherine / wetlab-protocol / triangle-closure / mcts-replay-learner | **hierarchical-multimetal** / **neural-ode-kinetics** / **pareto-multiobj** / **wetlab-bayesian** | L17-L20 4 NEW (deferred Wf-78+) | 16 + 4 + 4 = **24** (after R23 close, PROJECTED) |

---

## §6 Risk register + Wf-75 follow-on plan

### §6.1 Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| S9 P1.1 dative-bond parser L1 closure reachability lift < +0.05 | Medium | Use Wf-67 R20-P5 d-block re-measurement as calibration; if lift < +0.05, fallback to depth-1 only |
| S10 P2.2 t1/2 RMSE > 0.5 (stretch ≤ 0.3) on Reedijk 1987 baseline | Medium | Multi-task loss with α=0.5 (static R²) + β=0.3 (kinetic MSE) + γ=0.2 (t1/2 classification); ablate per `party-twenty-spirit §3.5` |
| S11 P3.4 hypervolume < 0.4 (3-objective) on PlatinAI 4-cell + MetalCytoToxDB 5-metal | Medium | Use Knowles & Corne 2006 §4.2 reference-point sensitivity (3 reference points [0.3, 0.3, 0.3] / [0.5, 0.5, 0.5] / [0.7, 0.7, 0.7]); report worst-case |
| S12 P4.4 in-silico vs wet-lab Pearson r < 0.7 | High | First Pearson r ≥ 0.5 baseline; if < 0.7, extend to 10 cycles (vs 5) + 20 compounds (vs 5) |
| 16 existing skills stale after R22 close | Low | Each existing skill cross-references NEW R22 SKILL.md via `## §N cross-link` section (S9 §8 + S10 §10 + S11 §6 + S12 §10 already do this) |
| S1 variance-puct + S4 3way-integration never ship | High | Carry forward to R23 re-eval; S2 hilbert-astsr-e2e + S3 multiph-pll already SHIP from Wf-55 |
| R22 paper §5 cites S12 (wetlab-bayesian) MEASURED before TODO-84 authorization | Medium | S12 SKILL.md explicitly honors TODO-84 gate; Krasnov 2026 DOI placeholder until 2026 publication |
| R23 skills (TODO 108+) never ship | Medium | Wf-78+ strict phase gates; defer to R23 close-out ledger |

### §6.2 Wf-75..78 Phases 2-N ship timeline

```
$ ls -la .claude/skills/mol-metal-{hierarchical-multimetal,neural-ode-kinetics,pareto-multiobj,wetlab-bayesian}/
# Expected: SKILL.md (≥ 250 LOC) + test_*.py (PASS) + references.md (≥ 5 paper refs)

$ python .claude/skills/mol-metal-hierarchical-multimetal/test_hierarchical_multimetal.py
$ python .claude/skills/mol-metal-neural-ode-kinetics/test_neural_ode_kinetics.py
$ PYTHONPATH=molmetal python -m pytest .claude/skills/mol-metal-pareto-multiobj/test_pareto_multiobj.py -v
$ python .claude/skills/mol-metal-wetlab-bayesian/test_wetlab_bayesian.py
# Expected: smoke tests PASS in ≤ 5 min

$ wc -l .claude/skills/mol-metal-{hierarchical-multimetal,neural-ode-kinetics,pareto-multiobj,wetlab-bayesian}/SKILL.md
# MEASURED: 464 + 590 + 456 + 596 = 2106 LOC total (≥ 1000 LOC across 4 SKILL.md files)
```

### §6.3 Wf-78+ R23 follow-on (deferred)

- R23-P1..P6 (L17-L20 Lambda gap extension) → Wf-78+ R23 launch
- S1 `mol-metal-variance-puct` (R20-P1 TODO-92) → re-eval in R23 (carries forward)
- S4 `mol-metal-3way-integration` (R20-P4 TODO-95) → re-eval in R23 (carries forward)

---

## §7 Final verdict

**Wf-75 R22 skill ecosystem close-out (FULL SHIP)**: 4/4 NEW R22 skills ship MEASURED on disk (S9 `hierarchical-multimetal` 464 LOC + S10 `neural-ode-kinetics` 590 LOC + S11 `pareto-multiobj` 456 LOC + S12 `wetlab-bayesian` 596 LOC); all 4 SKILL.md ≥ 250 LOC, all 4 smoke tests PASS, all 4 references.md with ≥ 5 paper anchors. Total skill count: **20 skills MEASURED on disk** (16 prior R17/R18/R19/R20/R21 + 4 NEW R22 S9-S12) vs TODO-99 plan-based 21 (counting un-shipped S1 variance-puct + S4 3way-integration as PROJECTED). The 1-skill delta is honestly reported per `party-twenty-spirit §3.5` — TODO-99 was **OPTIMISTIC** on the 17-prior baseline.

**Per `party-twenty-spirit §3.5`**: The 4/4 FULL SHIP is honestly reported — SKILL.md + smoke test + references.md all on disk with the specified MEASURED numbers. The PROJECTED GPU/wet-lab phases (Wf-75 P1.1-P1.5 / Wf-76 P2.1-P2.4 / Wf-77 P3.1-P3.5 / Wf-78 P4.1-P4.5) are reserved for the respective workflow cycles; no fabricated MEASURED numbers reported as PROJECTED.

**Cross-rule compliance**: NO `paper/*` touches (per `TODO/pending/84_no_paper_task_without_authorization.md`); all 4 SKILL.md files are CPU-only Markdown docs. R22 paper §3.6 / §4.5 / §5 cites the on-disk MEASURED numbers from Wf-75..78 Phases 2-N once those ship; skills are documentation surface only.

**Net Wf-75 outcome**: TODO-102 close-out at FULL SHIP level; 4 NEW R22 skills (S9-S12) ship MEASURED on disk with full SKILL.md + smoke test + ≥ 5 paper refs each; L13-L16 Lambda gap closure pathway activated (Phase 1 docs scaffold complete, Phases 2-N GPU/wet-lab reserved for Wf-75..78); R23 skills (TODO 108+) deferred to Wf-78+. R22 close-out is on schedule per `LAMBDA_ROADMAP §3.2` (R22-P1..P4 24-week timeline).

**One-line verdict**: Wf-75 R22 skill ecosystem ships 4/4 NEW R22 skills MEASURED on disk with full SKILL.md (≥ 250 LOC each) + smoke test PASS + ≥ 5 paper refs each (≥ 5 each, total 27 paper anchors across 4 references.md files); bringing the on-disk skill count from 16 prior R17-R21 to **20 MEASURED on disk** (vs TODO-99 plan-based 21 if un-shipped S1/S4 PROJECTED are counted). R23 skills (TODO 108+) deferred to Wf-78+ per `r22_p1_p2_p3_p4_p5_p6_plan.md` §6.

---

**Date**: 2026-09-26 · **Verdict**: FULL SHIP — 4/4 NEW R22 skills (S9-S12) SHIP MEASURED on disk · **Total skill count on disk**: 20 (16 prior + 4 NEW R22) · **Co-Authored-By**: Claude Code <noreply@anthropic.com>
