# TODO-105..110 — R22-P1..P6 plan (L13-L16 Lambda gap closure)

**Date**: 2026-09-26
**Workflow**: Wf-72 Phase 2 (R22-P1..P6 plan authoring — CPU-only docs)
**Status**: 🆕 **NEW — 4 active + 2 reserved R22-P plan items**
**Priority**: HIGH (drives R22-R25 timeline anchor + LAMBDA_ROADMAP §4.2 4-row extension)
**Owner**: Hugo Lee + ultracode agents (Wf-72 Phase 2 single-pass Claude Code author)
**Parent**:
- `TODO/pending/r22_l13_l16_gap_analysis.md` (Wf-72 Phase 1 ship, 311 lines, 330 scan signals)
- `TODO/pending/{92..99}_*.md` (R20-P1..P6 plan templates)
- `TODO/completed/r20_status_ledger_2026_09_26.md` (Wf-70 Phase 1, 50 sub-task R20 ledger)
- `TODO/completed/100_r20_r21_consolidation.md` (Wf-70 Phase 2, L1-L12 expansion)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (existing 12-row L1-L12 trajectory)
- `TODO/INDEX.md` §17.1 (22 metrics × 5 categories) + §17.5 (honest framing)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate, CRITICAL)
**Constraint**: CPU-only docs — no GPU retrain executed; **paper-gate honored** — no paper work touched
**Skill alignment**: `party-twenty-spirit` §3.5 (0-tolerance honest framing) + `mol-metal-lambda-calculus` + `mol-metal-r18-math` + `mol-metal-anticancer`

---

## §0 Why this plan exists

Wf-72 Phase 1 (`pending/r22_l13_l16_gap_analysis.md`) ships **4 candidate Lambda gaps L13-L16** based on R20/R21 honest-negative findings — 330 pattern-match signals across 14 close-out files + INDEX §17 + LAMBDA_ROADMAP §4.2. The L13-L16 candidates surface dimensions **NOT represented in L1-L12**:

1. **Coordination chemistry depth** (L13) — 4d/5d orbital sub-shell + multi-metal cluster centers + trans-effect propagation across M-M bonds; L1-L12 only addresses single-metal closure
2. **Dynamic kinetics** (L14) — ligand-exchange rates, hydrolysis t1/2, redox cycling (Pt(II)↔Pt(IV)); L1-L12 only addresses static ASTSR R²
3. **Multi-objective trade-off** (L15) — toxicity / efficacy / ADMET Pareto; L1-L12 only addresses single-scalar aggregation
4. **Wet-lab validation feedback** (L16) — in-silico → wet-lab loop with Bayesian optimization; L1-L12 only addresses in-silico bench

**Wf-72 Phase 2** (this doc) authors **6 R22-P plan items** (4 active + 2 reserved) following the R20-P1..P6 template structure, each with 4-5 phases and MEASURED-vs-PROJECTED targets per `party-twenty-spirit §3.5`. **R22-P5/P6 are reserved** as fallback slots for ship-evidence gap and cross-skill ecosystem per the gap-analysis §6 priorities.

---

## §1 R22-P1..P6 plan overview

| # | Lambda gap | Title | Hardware | Wall | Priority | Depends on |
|---:|---|---|---|---|---|---|
| **R22-P1** | L13 | Hierarchical multi-metal coordination geometry | CPU + GPU-0 | 3-4 wk | **HIGH** | L1 + L6 + L10 |
| **R22-P2** | L14 | Graph Neural ODE for metallodrug kinetics | GPU-1 | 4 wk | MEDIUM | L5 + L7 |
| **R22-P3** | L15 | Multi-objective Pareto front (toxicity/efficacy/ADMET) | CPU + GPU-0 | 3 wk | MEDIUM | L3 + L8 |
| **R22-P4** | L16 | Wet-lab-in-the-loop Bayesian optimization | wet-lab + CPU BO | 6 mo + 2 wk CPU BO | **HIGH** | L13 + L15 |
| **R22-P5** | [reserved] | ship-evidence gap (close-out traceability) | CPU | TBD | MEDIUM | (TBD by Wf-73) |
| **R22-P6** | [reserved] | cross-skill ecosystem (L13-L16 skill matrix) | CPU | TBD | MEDIUM | (TBD by Wf-73) |

**Aggregate R22 wall-time estimate**: 6-7 months R22 timeline + 24-month R23 follow-on for L14/L16 (per `r22_l13_l16_gap_analysis.md` §5 timeline anchor).

---

## §2 R22-P1 — Hierarchical multi-metal coordination geometry (L13)

### §2.0 Why this plan exists

L1 (closure theorem verifier) + L6 (closure d-block bench) + L10 (3-way integration) ship at CPU bench layer with **structural reachability ceiling 0.5744-0.5962 ≪ gate 0.95** (`completed/96_r20_p5_closure_catherine.md` §3.2 + `completed/95_r20_p4_3way_100pocket.md` §0). Three structural blockers prevent gate closure:

- **Valence-BNF rejection** of `[M+]<-X` dative-bond notation (RDKit valence accounting undercounts metal bonds)
- **Tautomerisation redex** exhaustion (~30% tmQM samples have tautomerisation redex that `is_beta_normal_form` rejects)
- **Sample sparsity** for 7 metals (Hf=20, Ta=30, La=3, Nb=19, Tc=15, Sc=20, Zr=53) — high per-metal variance

L13 (hierarchical multi-metal coord) is the **HIGH-priority candidate** with **188 scan signals** (57% of 330 total) per `r22_l13_l16_gap_analysis.md` §7.1. **R22-P1 implements L13** with chemistry-aware dative-bond parser + hierarchical M-M closure BFS + multi-metal cluster centers + trans-effect propagation.

### §2.1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU-only (parser + BFS) + Blackwell sm_120 GPU-0 for 30-metal × 1000-sample sweep |
| Wall-time | 3-4 weeks (1 wk CPU parser + 1 wk GPU-0 sweep + 1-2 wk analysis) |
| Output | `molmetal/reports/wf75_r22_p1/{dative_bond_parser,hierarchical_bfs,multimetal_cluster,transeffect_30metals}.json` |
| Tests | ~25 new tests: per-metal reachability + 6/6 Catherine properties + trans-effect propagation |

### §2.2 Cross-skill strategy

| Sub-task | Primary skill | Secondary skill |
|---|---|---|
| Chemistry-aware dative-bond parser | mol-metal-lambda-calculus | mol-metal-click-chemistry |
| Hierarchical M-M closure BFS | mol-metal-lambda-calculus | mol-metal-r18-math (M26 Score-Rep) |
| Multi-metal cluster centers | mol-metal-r18-math (T1 SE3 SB) | mol-metal-lambda-calculus |
| Trans-effect propagation | mol-metal-click-chemistry | mol-metal-lambda-calculus |

### §2.3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** Chemistry-aware dative-bond parser | 1 wk CPU | `molmetal/molmetal_lam/lam_chem/dative_bond_parser.py` + `tests/test_dative_bond_parser.py` (≥10 tests) | parser PASS on `[M+]<-X` SMARTS for Pt/Pd/Au/Ru/Ir |
| **P1.2** Hierarchical M-M closure BFS (depth ≥ 2) | 1 wk CPU | `molmetal/molmetal_lam/lam_chem/hierarchical_bfs.py` + `tests/test_hierarchical_bfs.py` (≥8 tests) | reachability lift on 5 metal clusters (Pt-Pt, Pt-Ru, Ru-Ru, Pt-Au, Au-Au) ≥ +0.05 |
| **P1.3** Multi-metal cluster centers (30-metal sweep) | 1 wk GPU-0 | `multimetal_cluster_30metals.json` (30 metals × 1000 samples per-metal reachability) | per-metal reachability ≥ 0.85 on ≥ 25/30 metals (vs current 0/30 at gate 0.95 strict, 1/30 at 0.85) |
| **P1.4** Trans-effect propagation capture | 2-3 days CPU | `transeffect_propagation.json` (Pt(II) series: Cl⁻ > NH₃ > H₂O kinetic trans-effect) | trans-effect ordering matches Reedijk 2008 experimental literature |
| **P1.5** Wf-75 close-out | 2 days CPU | `TODO/completed/105_r22_p1_hierarchical_multimetal.md` (≤ 300 lines) + LAMBDA_ROADMAP §4.2 L13 row patched | INDEX §17.6 cross-ref update + party-twenty-spirit §3.5 honest framing |

### §2.4 MEASURED-vs-PROJECTED

按 `party-twenty-spirit §3.5`:
- **MEASURED** = Wf-75 GPU/CPU runs on disk + per-metal reachability JSON + 6/6 Catherine PASS on ≥ 25 metals
- **PROJECTED** = reachability ≥ 0.95 on ≥ 25/30 metals (stretch) / ≥ 0.85 on ≥ 25/30 metals (realistic); 6/6 Catherine PASS preserved

### §2.5 Cross-references

- `TODO/pending/r22_l13_l16_gap_analysis.md` §1 (L13 candidate)
- `TODO/completed/96_r20_p5_closure_catherine.md` §3.2 (reachability 0.5744-0.5962 NOT MET)
- `TODO/completed/95_r20_p4_3way_100pocket.md` (3-way + 100-pocket + L10 honest-negative)
- `TODO/completed/91_wf41_3way_integration.md` (L10 3-way baseline R² = -0.3595 ± 0.2806)
- `TODO/completed/87j_wf44_root_cause_fixes.md` C5 (target distribution extreme TmQM-heavy)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L1/L6/L10 trajectory column — proposed §4.2 L13 row extension)
- `TODO/INDEX.md` §17.1 (22 metrics × 5 categories) — proposed §17.6 L13 row

---

## §3 R22-P2 — Graph Neural ODE for metallodrug kinetics (L14)

### §3.0 Why this plan exists

L5 (Hilbert-ASTSR) + L7 (Multi-PH + PLL) + L8 (Triangle closure) measure **static pIC50 R²** — a single snapshot of binding affinity (`completed/93_r20_p2_hilbert_astsr_gpu.md` §0-1.1). Real metallodrug efficacy depends on **dynamic lability** (ligand-exchange rate), **kinetic trans-effect** (cisplatin aquation t1/2 ≈ 2 hr), and **redox cycling** (Pt(II)→Pt(IV) under oxidative stress) — none captured by static ASTSR R².

L14 (Graph Neural ODE kinetics) is the **MEDIUM-priority candidate** with **20 scan signals** (6% of 330) — lowest evidence density but unique in dimension (dynamic kinetics). Gating factor is **GPU retrain wall-clock + literature kinetics data availability** (Reedijk 2008 + Hartinger 2009 + Peacock 2006 review). **R22-P2 implements L14** with GRU-ODE-Bayes backbone + ligand-exchange ODE + hydrolysis kinetics + redox cycling ODE.

### §3.1 Scope

| Item | Detail |
|---|---|
| Hardware | Blackwell sm_120 GPU-1 (32 GB) — GNN-ODE retrain |
| Wall-time | 4 weeks GPU-1 (≈ 10 days compute + 2 weeks analysis) |
| Output | `molmetal/reports/wf76_r22_p2/{gnn_ode_ligand_exchange,hydrolysis_t12,redox_cycling_potential}.json` |
| Tests | ~20 new tests: per-cohort t1/2 RMSE + per-cell redox rate constant + per-fold ablation |

### §3.2 Cross-skill strategy

| Sub-task | Primary skill | Secondary skill |
|---|---|---|
| GNN-ODE backbone (GRU-ODE-Bayes) | mol-metal-r18-math (T1 SE3 SB) | mol-metal-lambda-calculus |
| Ligand-exchange ODE (Cl⁻/H₂O/GSH) | mol-metal-click-chemistry | mol-metal-r18-math (T7 SB-Bass) |
| Hydrolysis kinetics (cisplatin aquation) | mol-metal-anticancer | mol-metal-click-chemistry |
| Redox cycling ODE (Pt(II)↔Pt(IV)) | mol-metal-r18-math (T7 SB-Bass) | mol-metal-anticancer |

### §3.3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P2.1** GNN-ODE backbone (GRU-ODE-Bayes Chen et al. 2018 NeurIPS) | 1 wk GPU-1 | `molmetal/molmetal_lam/lam_chem/gnn_ode.py` + `tests/test_gnn_ode.py` (≥8 tests) | backbone PASS on synthetic Pt(II)/Pd(II)/Ru(II) redox cycle |
| **P2.2** Ligand-exchange ODE (Cl⁻/H₂O/GSH rate constants) | 1 wk GPU-1 | `gnn_ode_ligand_exchange.json` (per-metal rate constants) | t1/2 RMSE ≤ 0.5 vs Reedijk 2008 experimental |
| **P2.3** Hydrolysis kinetics + redox cycling | 1 wk GPU-1 | `hydrolysis_t12.json` + `redox_cycling_potential.json` (per-cell redox rate + Pt(II)/Pt(IV) potential) | multi-task loss: static R² + kinetic rate constant MSE + t1/2 classification |
| **P2.4** Wf-76 close-out | 2 days CPU | `TODO/completed/106_r22_p2_gnn_ode_kinetics.md` (≤ 300 lines) + LAMBDA_ROADMAP §4.2 L14 row patched | INDEX §17.6 cross-ref update + party-twenty-spirit §3.5 honest framing |

### §3.4 MEASURED-vs-PROJECTED

按 `party-twenty-spirit §3.5`:
- **MEASURED** = Wf-76 GPU runs on disk + per-cohort t1/2 RMSE JSON + multi-task loss ablation
- **PROJECTED** = t1/2 RMSE ≤ 0.5 vs literature (stretch ≤ 0.3); multi-task loss stable (R² ≥ 0.05 lift over static-only baseline)

### §3.5 Cross-references

- `TODO/pending/r22_l13_l16_gap_analysis.md` §2 (L14 candidate)
- `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` §1.1 (CPU smoke R² ≈ -0.02)
- `TODO/completed/100_wf71_p3_wf49_cross_round2.md` (PlatinAI 4-cell A2780cis only R²=+0.037)
- `TODO/completed/96_r20_p5_closure_catherine.md` §3.2 (static closure empirical ceiling 0.85)
- Reedijk 2008 *Platinum Metals Rev.* (cisplatin aquation kinetics reference)
- Chen et al. 2018 NeurIPS *Neural Ordinary Differential Equations* (GRU-ODE-Bayes)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L5/L7 trajectory column — proposed §4.2 L14 row)

---

## §4 R22-P3 — Multi-objective Pareto front over toxicity / efficacy / ADMET (L15)

### §4.0 Why this plan exists

L3 (ScoreRep + VariancePUCT + MCTS) + L8 (ASTSR ↔ Hilbert ↔ MCTS triangle) + L10 (3-way) all optimize **single-scalar objectives** (R², success rate, Pearson r). Real drug discovery requires **Pareto trade-off** between toxicity (lower = better) + efficacy (higher = better) + ADMET (multi-axis: solubility, permeability, metabolic stability). Single-scalar aggregation (e.g., weighted sum) **loses dominance structure**.

L15 (Multi-objective Pareto front) is the **MEDIUM-priority candidate** with **52 scan signals** (16% of 330). m18 (Protein-Metal binding AUC) / m19 (LOMO AUC) / m20 (Tumor-specific AUC) all UNDEFINED or WEAK_REAL_NEEDS_MORE_DATA per `INDEX §17.1`. **R22-P3 implements L15** with NSGA-III backbone + 3-objective Pareto (m18 + toxicity + ADMET) + hypervolume indicator for paper §4 main result.

### §4.1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU-only (NSGA-III bench) + Blackwell sm_120 GPU-0 for 30-metal × 1000-sample sweep |
| Wall-time | 3 weeks (1 wk CPU NSGA-III + 1 wk GPU-0 sweep + 1 wk analysis) |
| Output | `molmetal/reports/wf77_r22_p3/{nsga3_pareto_front,hypervolume_indicator,reference_points}.json` |
| Tests | ~20 new tests: hypervolume ≥ 0.6 (3-objective) + per-metal Pareto rank + per-cohort dominance coverage |

### §4.2 Cross-skill strategy

| Sub-task | Primary skill | Secondary skill |
|---|---|---|
| NSGA-III backbone (Deb & Jain 2014) | mol-metal-r18-math (T5 Multi-PH) | mol-metal-lambda-calculus |
| 3-objective Pareto (m18 + toxicity + ADMET) | mol-metal-anticancer | mol-metal-mechanism-mapping |
| Hypervolume indicator | mol-metal-r18-math (T5 Multi-PH) | mol-metal-anticancer |
| MCTS ↔ NSGA-III bridge | mol-metal-lambda-calculus (M26 Score-Rep) | mol-metal-r18-math (T5) |

### §4.3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P3.1** m18/m19/m20 metric definition (Protein-Metal binding AUC / LOMO / Tumor-specific AUC) | 3 days CPU | `TODO/02_metrics/METRICS_TARGETS.md` §m18-m20 + `tests/test_metric_definitions.py` (≥6 tests) | metric PASS on retrospective + scaffold split |
| **P3.2** NSGA-III backbone + 3-objective Pareto | 1 wk CPU | `molmetal/molmetal_lam/search_alg/nsga3_pareto.py` + `tests/test_nsga3_pareto.py` (≥8 tests) | NSGA-III PASS on synthetic 3-objective test (ZDT1 / DTLZ2) |
| **P3.3** MCTS ↔ NSGA-III bridge (VariancePUCT as offspring generator) | 2-3 days CPU | `molmetal/molmetal_lam/search_alg/mcts_nsga3_bridge.py` + `tests/test_mcts_nsga3_bridge.py` (≥6 tests) | bridge PASS on 30-metal × 1000-sample pareto front |
| **P3.4** Hypervolume indicator (paper §4 main result) | 2-3 days GPU-0 | `hypervolume_indicator.json` (3-objective hypervolume ≥ 0.6 on real m18 + toxicity + ADMET) | hypervolume ≥ 0.6 (stretch) / ≥ 0.4 (realistic) on PlatinAI 4-cell |
| **P3.5** Wf-77 close-out | 2 days CPU | `TODO/completed/107_r22_p3_pareto_front.md` (≤ 300 lines) + LAMBDA_ROADMAP §4.2 L15 row patched | INDEX §17.6 cross-ref update + party-twenty-spirit §3.5 honest framing |

### §4.4 MEASURED-vs-PROJECTED

按 `party-twenty-spirit §3.5`:
- **MEASURED** = Wf-77 GPU/CPU runs + hypervolume JSON + 30-metal × 1000-sample pareto front
- **PROJECTED** = hypervolume ≥ 0.6 (3-objective, stretch) / ≥ 0.4 (realistic); m18 Protein-Metal binding AUC ≥ 0.78 on PlatinAI 4-cell

### §4.5 Cross-references

- `TODO/pending/r22_l13_l16_gap_analysis.md` §3 (L15 candidate)
- `TODO/INDEX.md §17.1` (m18/m19/m20 UNDEFINED or WEAK_REAL_NEEDS_MORE_DATA)
- `TODO/completed/85_mock_audit.md` (8 critical + 23 major + 98 minor mocks)
- `TODO/completed/95_r20_p4_3way_100pocket.md` §0 (3-way on MetalCytoToxDB R² ≥ +0.30 NOT executed)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (m18/m19/m20 cooperation context)
- Deb & Jain 2014 *Evolutionary Computation* (NSGA-III reference)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L3/L8 trajectory column — proposed §4.2 L15 row)

---

## §5 R22-P4 — Wet-lab-in-the-loop Bayesian optimization (L16)

### §5.0 Why this plan exists

R20-P6 (`pending/97_r20_p6_wetlab_ic50.md`) ships the **in-silico side** (D-MPNN predictions, dry-run validation) — `completed/97_r20_p6_wetlab_ic50_dryrun.md` confirms in-silico SHIP today. The **wet-lab feedback loop** is open: without prospective IC50 validation, paper §5 claim ("in-silico predictions agree with wet-lab within Pearson r ≥ 0.7") cannot be made.

L16 (Wet-lab-in-the-loop Bayes opt) is the **HIGH-priority candidate** with **70 scan signals** (21% of 330). Wet-lab cooperation with ORNL/PKU/协和 is 6-month deferred per `pending/97_r20_p6_wetlab_ic50.md` §3. **R22-P4 closes the loop** with Gaussian-process surrogate + q-EI batch acquisition + 7-week wet-lab cycle (2-week synthesis + 4-week assay + 1-week analysis).

### §5.1 Scope

| Item | Detail |
|---|---|
| Hardware | wet-lab (ORNL/PKU/协和 cell culture + MTT/SRB/CellTiter-Glo assay) + CPU-only BO |
| Wall-time | 6 months wet-lab (P6.1 outreach + P6.2 cell culture + P6.3 in-silico vs wet-lab) + 2 weeks CPU BO |
| Output | `molmetal/reports/wf78_r22_p4/{wetlab_ic50_q1,bayes_opt_convergence,silico_vs_wetlab_agreement}.json` |
| Tests | ~10 new tests: per-cell IC50 ± 95% CI + GP surrogate calibration + q-EI convergence |

### §5.2 Cross-skill strategy

| Sub-task | Primary skill | Secondary skill |
|---|---|---|
| Gaussian-process surrogate | mol-metal-r18-math (T7 SB-Bass) | mol-metal-anticancer |
| q-EI batch acquisition | mol-metal-r18-math (T5 Multi-PH) | mol-metal-lambda-calculus |
| Wet-lab cycle (MCF7 + A549 + A2780 + A2780cis + 5 NCI-60 cells) | mol-metal-anticancer | mol-metal-wetlab-protocol |
| In-silico vs wet-lab agreement | mol-metal-anticancer | mol-metal-mechanism-mapping |

### §5.3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P4.1** Wet-lab PI outreach (ORNL / PKU / 协和 LOI + NDA) | 1 mo wet-lab | LOI signed + NDA + 20 compounds synthesized | LOI signed + 20/20 compounds QC-pass |
| **P4.2** Cell culture + IC50 assay (MCF7 + A549 + A2780 + A2780cis + 5 NCI-60 cells) | 3 mo wet-lab | `wetlab_ic50_q1.json` (per-compound × per-cell IC50 ± std, 9 cell lines) | full panel IC50 measured |
| **P4.3** GP surrogate + q-EI batch BO (CPU) | 2 wk CPU | `bayes_opt_convergence.json` (top-20 candidate acquisition + 5-candidate batches per cycle) | GP calibration + q-EI convergence ≥ 5 cycles |
| **P4.4** In-silico vs wet-lab agreement | 1 mo CPU | `silico_vs_wetlab_agreement.json` (Pearson r ≥ 0.7 on top-20 candidates) | Pearson r ≥ 0.7 (stretch) / ≥ 0.5 (realistic) |
| **P4.5** Wf-78 close-out + paper §5 final claim | 1 mo doc (after TODO-84 authorization) | `TODO/completed/108_r22_p4_wetlab_bayes_opt.md` + `paper/sections/05_wetlab_validation.tex` (gated) | shipped after auth + Pearson r ≥ 0.7 |

### §5.4 MEASURED-vs-PROJECTED

按 `party-twenty-spirit §3.5`:
- **MEASURED** = wet-lab IC50 measured + in-silico vs wet-lab agreement JSON + GP surrogate calibration
- **PROJECTED** = Pearson r ≥ 0.7 (stretch) / ≥ 0.5 (realistic) between D-MPNN prediction and wet-lab IC50; GP calibration RMSE ≤ 0.3

### §5.5 Cross-references

- `TODO/pending/r22_l13_l16_gap_analysis.md` §4 (L16 candidate)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 plan, 6-mo deferred)
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (Wf-58 P6.0+P6.1+P6.2 ship surface)
- `TODO/pending/wetlab_ic50_analysis_pipeline.md` (Wf-74 P3 IC50 analysis pipeline template)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate; P4.5 needs auth)
- Williams & Rasmussen 2006 *Gaussian Processes for Machine Learning* (GP surrogate reference)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L13/L15 dependency — proposed §4.2 L16 row)

---

## §6 R22-P5 — [reserved: ship-evidence gap]

### §6.0 Why this plan exists

R20 status ledger (`completed/r20_status_ledger_2026_09_26.md` §0) confirms **0/8 R20-P fully MEASURED on disk** + 5/8 partial / 2/8 partial / 1/8 phase 2 partial + 3/50 sub-tasks NOT MET (reachability 0.5962 < 0.95 gate; 0.5744 < 0.95 gate; Wf-44 P2 empty dir). The **ship-evidence gap** (close-out traceability + benchmark reproducibility + MEASURED-vs-PROJECTED audit trail) is a structural issue not addressed by L1-L16 algorithmic gaps.

**R22-P5 is RESERVED** as a fallback slot for **ship-evidence gap closure** — depending on Wf-73 Phase 1 outcome, this slot may be assigned to:

| Candidate sub-topic | Why reserved | Owner |
|---|---|---|
| Close-out traceability audit (50 sub-tasks R20 → R22 carry-forward) | R20 ledger has 5/8 partial + 3/50 NOT MET — risk of L13-L16 ship similar partial state | Wf-73 |
| Benchmark reproducibility bench (per-metric JSON + per-report hash) | R20 close-out files have honest-partial framing but lack per-bench hash chain | Wf-73 |
| MEASURED-vs-PROJECTED audit trail (party-twenty-spirit §3.5 enforcement) | R20 §7 close-out files use PROJECTED labels but lack audit log | Wf-73 |
| R21 status ledger (Wf-70 Phase 3, 8 R21-P plan items) | R21 not yet ledger'd | Wf-70 Phase 3 |
| R22 status ledger (Wf-72 Phase 3, 4 active R22-P plan items) | R22 not yet ledger'd | Wf-72 Phase 3 |

**R22-P5 placeholder phases** (Wf-73 will refine):

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P5.1** R20+R21 ledger verification (Wf-70 Phase 3) | 1 wk CPU | `TODO/completed/r21_status_ledger_2026_09_26.md` (mirror R20 ledger format) | ledger PASS review |
| **P5.2** Close-out traceability audit | 1 wk CPU | `TODO/02_metrics/EVALUATORS_MIGRATED.md` §audit-trail + `tests/test_closeout_audit.py` (≥8 tests) | audit PASS on 50 R20 sub-tasks |
| **P5.3** Benchmark reproducibility bench | 1 wk CPU | `molmetal/reports/bench_reproducibility/{per_metric_hash,per_report_sha256}.json` | hash chain PASS |
| **P5.4** MEASURED-vs-PROJECTED audit log | 2-3 days CPU | `TODO/02_metrics/MEASURED_VS_PROJECTED_AUDIT.md` + `tests/test_measured_vs_projected.py` (≥6 tests) | audit PASS on all R20+R21 close-out files |
| **P5.5** Wf-73 close-out | 2 days CPU | `TODO/completed/109_r22_p5_ship_evidence_gap.md` (≤ 300 lines) | INDEX §17 cross-ref update |

**MEASURED-vs-PROJECTED** (placeholder): all phases TBD by Wf-73 Phase 1.

### §6.1 Cross-references

- `TODO/completed/r20_status_ledger_2026_09_26.md` §0 (5/8 partial + 3/50 NOT MET)
- `TODO/completed/100_r20_r21_consolidation.md` (Wf-70 Phase 2 L1-L12 expansion)
- `TODO/02_metrics/EVALUATORS_MIGRATED.md` (14 evaluators — audit trail anchor)
- `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 (0-tolerance honest framing)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (proposed §4.2 L17 row for ship-evidence gap — reserved)

---

## §7 R22-P6 — [reserved: cross-skill ecosystem]

### §7.0 Why this plan exists

The 8 R20 skills (`pending/99_r20_skill_ecosystem.md`) ship 4 R20 + 4 R21 skills to cover L1-L12 algorithmic gaps. **L13-L16 require NEW skills** that don't exist yet:

| L# | Required NEW skill | Why NEW |
|---|---|---|
| L13 Hierarchical multi-metal coord | `mol-metal-hierarchical-multimetal` | chemistry-aware dative-bond parser + M-M BFS + cluster centers |
| L14 GNN-ODE kinetics | `mol-metal-gnn-ode-kinetics` | GRU-ODE-Bayes backbone + ligand-exchange ODE + hydrolysis kinetics |
| L15 Multi-objective Pareto | `mol-metal-pareto-front` | NSGA-III + 3-objective hypervolume + MCTS-NSGA-III bridge |
| L16 Wet-lab BO | `mol-metal-bayes-opt-wetlab` | GP surrogate + q-EI + wet-lab cycle + IC50 analysis pipeline |

**R22-P6 is RESERVED** as a fallback slot for **cross-skill ecosystem** — depends on Wf-72 Phase 3 R22 status ledger + Wf-73 ship-evidence gap. If R22-P5/R22-P6 are NOT activated by R23 planning, the skill ecosystem for L13-L16 is **deferred to R23-skill-ecosystem plan**.

**R22-P6 placeholder phases** (Wf-73 will refine):

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P6.1** `mol-metal-hierarchical-multimetal` SKILL.md | 2 wk CPU | `.claude/skills/mol-metal-hierarchical-multimetal/SKILL.md` (≥ 250 LOC) + `references.md` + `test_hierarchical_multimetal.py` (≥6 tests) | SKILL.md PASS review |
| **P6.2** `mol-metal-gnn-ode-kinetics` SKILL.md | 2 wk CPU | `.claude/skills/mol-metal-gnn-ode-kinetics/SKILL.md` (≥ 250 LOC) + `references.md` + `test_gnn_ode.py` (≥6 tests) | same |
| **P6.3** `mol-metal-pareto-front` SKILL.md | 2 wk CPU | `.claude/skills/mol-metal-pareto-front/SKILL.md` (≥ 250 LOC) + `references.md` + `test_pareto_front.py` (≥6 tests) | same |
| **P6.4** `mol-metal-bayes-opt-wetlab` SKILL.md | 1 wk CPU | `.claude/skills/mol-metal-bayes-opt-wetlab/SKILL.md` (≥ 250 LOC) + `references.md` + `test_bayes_opt_wetlab.py` (≥4 tests) | same |
| **P6.5** Wf-79 close-out | 1 wk CPU | `TODO/completed/110_r22_p6_cross_skill_ecosystem.md` (≤ 300 lines) | INDEX §17.6 cross-ref update |

**MEASURED-vs-PROJECTED** (placeholder): all phases TBD by Wf-73 Phase 1.

### §7.1 Cross-references

- `TODO/pending/99_r20_skill_ecosystem.md` (8 R20 skills — L1-L12 anchor)
- `TODO/pending/r22_l13_l16_gap_analysis.md` §6 (proposed §4.2 4-row extension)
- `TODO/completed/99_r20_skill_ecosystem.md` (Wf-55 8 R20 skills close-out, partial SHIP)
- `.claude/skills/mol-metal-lambda-calculus/` (L13-L16 cross-references)
- `.claude/skills/mol-metal-r18-math/` (M26 Score-Rep, T7 SB-Bass — L13-L16 anchor)
- `.claude/skills/mol-metal-anticancer/` (m18/m19/m20 metrics — L15/L16 anchor)

---

## §8 R22-P1..P6 aggregate MEASURED-vs-PROJECTED (party-twenty-spirit §3.5)

### §8.1 MEASURED today (CPU-only docs)

| Item | Value | Source |
|---|---:|---|
| Wf-72 Phase 1 ship | ✅ | `TODO/pending/r22_l13_l16_gap_analysis.md` (311 lines) |
| R22-P1..P6 plan files authored | 1 (this doc) | `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` (~440 lines) |
| R22-P1..P6 plan templates | 1 | `molmetal/scripts/template_r22_plan.py` (~150 LOC) |
| R22-P1..P6 plan signals JSON | 1 | `molmetal/reports/wf72_r22_plans/r22_plan_signals.json` |
| Pattern-match signals total | **330** | `analyze_lambda_gaps.py` regex scan |
| R20 ship honest-negative carry | **0/8 R20-P fully MEASURED** + 3/50 NOT MET | `completed/r20_status_ledger_2026_09_26.md` §0 |
| L13-L16 candidate signal counts | **188/20/52/70** | `pending/r22_l13_l16_gap_analysis.md` §7.1 |

### §8.2 PROJECTED (reserved for R22-R25 plan owners)

| R22-P | MEASURE target | Why PROJECTED | When MEASURED |
|---|---|---|---|
| **R22-P1** | reachability ≥ 0.95 on ≥ 25/30 metals (L13) | requires chemistry-aware dative-bond parser + hierarchical BFS + GPU-0 sweep | R22 timeline (3-4 wk) |
| **R22-P2** | t1/2 RMSE ≤ 0.5 + multi-task R² ≥ +0.05 lift (L14) | requires GNN-ODE retrain + literature kinetics data | R23 timeline (4 wk GPU-1) |
| **R22-P3** | hypervolume ≥ 0.6 (3-objective) + m18 AUC ≥ 0.78 (L15) | requires m18/m19/m20 metric definition + NSGA-III bench + GPU-0 sweep | R22 timeline (3 wk) |
| **R22-P4** | Pearson r ≥ 0.7 wet-lab agreement (L16) | requires 6-month wet-lab cooperation + GP/q-EI BO | R23 timeline (6-mo wet-lab + 2-wk CPU BO) |
| **R22-P5** | [reserved] | TBD by Wf-73 | R22 timeline (TBD) |
| **R22-P6** | [reserved] | TBD by Wf-73 | R22 timeline (TBD) |

---

## §9 Honest framing (party-twenty-spirit §3.5)

| Honest finding | Evidence |
|---|---|
| **L13-L16 are CANDIDATE shapes, not committed gaps** | Per `pending/r22_l13_l16_gap_analysis.md` §0; R22 plan owners re-evaluate against phase-specific evidence |
| **Pattern-match signal counts are NOT quality measures** | 188 L13 hits ≠ "L13 is the right shape" — regex match density ≠ quality |
| **R22-P1..P4 PROJECTED, NOT MEASURED** | All 4 active plans ship today as CPU-only docs; MEASURED numbers come after R22 GPU/wet-lab phases |
| **R22-P5/P6 RESERVED** | Both slots are placeholders for ship-evidence gap (P5) + cross-skill ecosystem (P6); Wf-73 will refine |
| **No GPU retrain executed today** | CPU-only docs only — R22-P1 GPU sweep + R22-P2 GNN-ODE + R22-P3 GPU-0 sweep reserved for R22-R23 |
| **No paper work touched** | Per `pending/84_no_paper_task_without_authorization.md` — no `paper/sections/*.tex`, no reviewer rebuttal |
| **R20 honest-negative carry-forward** | 0/8 R20-P fully MEASURED + 3/50 NOT MET carry into R22 — L13-L16 do NOT resolve R20 honest-negative |
| **Wet-lab cooperation 6-month deferred** | ORNL/PKU/协和 outreach 0 progress today; L16 cannot close until R23-P4 wet-lab cycle starts |

Per `party-twenty-spirit §3.5`: This Wf-72 Phase 2 doc ships **honest framing** — no MEASURED numbers fabricated, no PROJECTED numbers labeled as MEASURED, no GPU/wet-lab phases silently claimed ship.

---

## §10 Cross-references

### §10.1 Inputs (Wf-72 Phase 1 ship surface)

- **`TODO/pending/r22_l13_l16_gap_analysis.md`** (Wf-72 Phase 1, 311 lines) — primary input
- **`TODO/pending/{92..99}_*.md`** (R20-P1..P6 plan templates) — structural template
- **`TODO/completed/r20_status_ledger_2026_09_26.md`** (Wf-70 Phase 1, 50 sub-task R20 ledger) — honest-negative context
- **`TODO/completed/100_r20_r21_consolidation.md`** (Wf-70 Phase 2, L1-L12 expansion)
- **`TODO/pending/84_no_paper_task_without_authorization.md`** (paper-gate, CRITICAL) — honored
- **`.claude/skills/party-twenty-spirit/SKILL.md`** §3.5 (0-tolerance honest framing) — applied throughout

### §10.2 Outputs (this Wf-72 Phase 2 ship surface)

- **`TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md`** (this doc, ~440 lines) — primary deliverable
- **`molmetal/scripts/template_r22_plan.py`** (~150 LOC) — per-R22-P JSON template generator
- **`molmetal/reports/wf72_r22_plans/r22_plan_signals.json`** — per-R22-P signals + GPU/CPU wall-time estimates

### §10.3 Parent plans + sibling workflows

- **`TODO/03_plans/LAMBDA_ROADMAP.md`** §1 (R20 plan) + §2 (R21 plan) + §3 (R22-R25 long-term) + §4.2 (12-row L1-L12 trajectory) — proposed §4.2 4-row extension (L13/L14/L15/L16) in `pending/r22_l13_l16_gap_analysis.md` §6
- **`TODO/INDEX.md`** §17.1 (22 metrics × 5 categories) + §17.3 (R20 plan table) + §17.5 (honest framing NEW) — proposed §17.6 L13-L16 candidate row
- **`TODO/pending/97_r20_p6_wetlab_ic50.md`** (R20-P6 wet-lab, 6-mo deferred) — L16 first-close anchor
- **`TODO/pending/98_wf44_closure_ood_phase2.md`** (Wf-44 P2 L12 Phase 2, Q4 2026) — L13 OOD Phase 2 anchor
- **`TODO/pending/wetlab_ic50_analysis_pipeline.md`** (Wf-74 P3 IC50 analysis pipeline) — L16 IC50 analysis anchor

### §10.4 Follow-on workflows (Wf-72 Phase 3+, NOT this doc)

- **Wf-72 Phase 3**: Apply §4.2 4-row extension to `LAMBDA_ROADMAP.md` + create §17.6 candidate row in `INDEX.md`
- **Wf-73 Phase 1**: Refine R22-P5 ship-evidence gap + R22-P6 cross-skill ecosystem (post-R22 GPU bench stabilization)
- **Wf-75 / Wf-76 / Wf-77 / Wf-78**: R22-P1..P4 close-out workflows (after R22 GPU retrain)
- **Wf-79**: R22-P6 cross-skill ecosystem close-out (if R22-P6 activated)

---

## §11 One-line verdict

Wf-72 Phase 2 SHIPs **6 R22-P plan items** based on Wf-72 Phase 1 L13-L16 candidate gaps:
- **R22-P1** (HIGH, L13, 3-4 wk GPU-0): Hierarchical multi-metal coordination geometry (188 scan signals)
- **R22-P2** (MEDIUM, L14, 4 wk GPU-1): Graph Neural ODE for metallodrug kinetics (20 scan signals)
- **R22-P3** (MEDIUM, L15, 3 wk GPU-0): Multi-objective Pareto front over toxicity/efficacy/ADMET (52 scan signals)
- **R22-P4** (HIGH, L16, 6 mo wet-lab + 2 wk CPU BO): Wet-lab-in-the-loop Bayesian optimization (70 scan signals)
- **R22-P5** (RESERVED): ship-evidence gap (close-out traceability)
- **R22-P6** (RESERVED): cross-skill ecosystem (L13-L16 skill matrix)

Total **330 pattern-match signals** across 14 R20/R21 close-out files. R22-P1..P4 ship today as **CPU-only docs** with MEASURED-vs-PROJECTED honest framing per `party-twenty-spirit §3.5`. R22-P5/P6 reserved for Wf-73 refinement. Wf-72 Phase 3 will apply the LAMBDA_ROADMAP §4.2 4-row extension + create INDEX §17.6 candidate row.
