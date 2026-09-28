# Wf-72 Phase 1 — R22 L13-L16 Lambda Gap Analysis

**Date**: 2026-09-26
**Workflow**: Wf-72 (Phase 1: L13-L16 Lambda gap analysis based on R20/R21 ship results)
**Status**: 🆕 **NEW** — 4 candidate Lambda gaps (L13-L16) identified for R22+ roadmap extension
**Priority**: HIGH (drives R22-P3..P4 plan items + Wf-72 follow-on phases)
**Owner**: Hugo Lee + Wf-72 ultracode agents
**Parent**:
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (existing 12-row L1-L12 trajectory)
- `TODO/completed/r20_status_ledger_2026_09_26.md` (Wf-70 Phase 1, 50 sub-task R20 ledger)
- `TODO/completed/100_r20_r21_consolidation.md` (Wf-70 Phase 2, L1-L12 expansion)
- `TODO/completed/{93,94,95,96,97,98,99}_r20_p*.md` (7 R20 close-out files)
- `TODO/completed/87j_wf44_root_cause_fixes.md` (Wf-44 root causes C1-C8)
- `TODO/completed/85_mock_audit.md` (Wf-42 mock inventory: 8 critical + 23 major + 98 minor)
- `TODO/INDEX.md` §17.1 (22 metrics × 5 categories)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6 wet-lab plan, 6-mo deferred)
**Constraint**: CPU-only docs — no GPU retrain executed; **paper-gate (TODO-84) honored** — no paper work touched
**Skill alignment**: `party-twenty-spirit` §3.5 (0-tolerance honest framing) + `mol-metal-lambda-calculus`

---

## §0 Why L13-L16 are needed

`LAMBDA_ROADMAP §4.2` ships a **12-row L1-L12 trajectory** that closes **9/12 gaps in R20 + 3/12 in R21 = 12/12 全 close by R21+**. Wf-70 Phase 2 (TODO-100) confirmed this grid is **honest-negative in 3 dimensions**:

1. **Structural reachability ceiling 0.5744-0.5962 ≪ gate 0.95**: Wf-67 P2 (L6) returns 0.5962 aggregate over 30 metals × 5000 tmQM samples, vs gate 0.95. The gate is **structurally blocked** by (a) valence-BNF conservative rejection of dative-bond notation, (b) tautomerisation redex exhaustion (~30%), (c) sample sparsity for 7 metals (n<30). No L1-L12 row addresses this.
2. **GPU retrain deferred across R20**: 5/8 R20-P (R20-P2/P3/P4/P5/P6) ship code + skill + smoke + close-out, but **0/8 R20-P fully MEASURED on disk**. The static R² / reachability metrics in L1-L12 measure a snapshot, not the dynamic lability / kinetics that real metallodrugs exhibit. No L1-L12 row addresses this.
3. **m18/m19/m20 tier-1 metric ceiling + 6-mo wet-lab deferral**: `INDEX §17.1` catalogues **~6 UNDEFINED metrics** including m18/m19/m20 (Protein-Metal AUC / LOMO AUC / Tumor-specific AUC). R20-P6 (TODO-97) wet-lab cooperation with ORNL/PKU/协和 is deferred 6 months. No L1-L12 row addresses the in-silico → wet-lab validation feedback loop.

The 12-row L1-L12 grid **does not contain candidates for these dimensions** because:
- L1-L12 trace **algorithmic capabilities** (closure theorem, VariancePUCT, ASTSR-Hilbert, MCTS triangle, CFM coupling, replay learner). They are all **in-silico algorithmic gaps**.
- **Coordination chemistry depth** (4d/5d orbital + multi-metal cluster centers), **dynamic kinetics** (ligand-exchange rates, hydrolysis t1/2), **multi-objective trade-off** (toxicity/efficacy/ADMET), and **wet-lab validation feedback** are **NOT represented** in L1-L12.

L13-L16 fill this gap as **candidates**. Per `party-twenty-spirit §3.5`, these are **CANDIDATE** shapes — R22+ plan owners should re-evaluate against their phase-specific evidence. MEASURED signal counts reflect pattern-match hits in close-out files; they are NOT a guarantee of the right shape.

---

## §1 L13 — Hierarchical multi-metal coordination geometry

### §1.1 One-line definition
> Replace single-metal-center closure (current L1/L6/L12) with **hierarchical multi-metal coordination geometry** — 4d/5d orbital sub-shell + multi-metal cluster centers + trans-effect propagation across M-M bonds.

### §1.2 Evidence (R20/R21 honest-negative)

| Source | Finding |
|---|---|
| `completed/96_r20_p5_closure_catherine.md` §1.2 | Closure reachability aggregate **0.5962** vs gate **0.95**; per-metal worst V=0.1389 (valence-BNF rejection of V=O/V-S dative bonds) |
| `completed/96_r20_p5_closure_catherine.md` §3.2 | Reachability lift Wf-36 P1 (0.5108) → Wf-67 P2 (0.5962) = +0.0854; gate structurally NOT MET |
| `completed/95_r20_p4_3way_100pocket.md` | 100-pocket × 10-seed BLOCKED on meko 0.8.0 Pt/Au/Ru; reachability 0.5962 NOT gate 0.95 |
| `completed/91_wf41_3way_integration.md` (L10) | 3-way R²=-0.3595 ± 0.2806 on synthetic; +0.7524 lift over AST alone (synthetic) |
| `completed/87j_wf44_root_cause_fixes.md` C5 | Target distribution extreme (TmQM-heavy metals) confirmed but **deferred** — Yeo-Johnson power transform needed |

**Structural reasons** (per `96_r20_p5_closure_catherine.md §3.2`):
- **Valence-BNF rejection**: closure theorem conservatively rejects `[M+]<-X` dative-bond notation because RDKit valence accounting undercounts metal bonds.
- **Tautomerisation redex**: ~30% tmQM samples possess tautomerisation redex that `is_beta_normal_form` predicate rejects.
- **Sample sparsity**: 7 metals have n<30 in tmQM (Hf=20, Ta=30, La=3, Nb=19, Tc=15, Sc=20, Zr=53) — high per-metal variance.

### §1.3 Candidate approach
- **Hierarchical M-M closure BFS**: traverse M-M bond graph at depth ≥ 2 (vs current max_depth=1)
- **Dative-bond chemistry-aware parser**: treat `[M+]<-X` as `M-X` covalent for valence accounting
- **Multi-metal cluster centers**: square-planar Pt(II) + axial ligand, octahedral Pt(IV), dinuclear Rh/Ru paddlewheel
- **Trans-effect propagation**: capture kinetic trans-effect (Cl⁻ > NH₃ > H₂O in Pt(II) series)

### §1.4 Priority + first-close horizon
- **Priority**: HIGH (structural ceiling on L1/L6/L12)
- **R first close**: R22-P4 (Wf-54, 24-week R22 timeline)
- **Depends on**: L1 (closure theorem verifier) + L6 (closure d-block bench) + L10 (3-way)
- **Scan signal count**: 188 (structural_ceiling=144 + unmet_gate=44)

---

## §2 L14 — Graph Neural ODE for metallodrug kinetics / dynamic bond lability

### §2.1 One-line definition
> Replace static ASTSR R² benchmarks (current L5/L7/L8) with **Graph Neural ODE for metallodrug kinetics** — continuous-time dynamics over ASTSR signature, modeling ligand-exchange rates, hydrolysis t1/2, and redox cycling (Pt(II)↔Pt(IV), Ru(II)↔Ru(III)).

### §2.2 Evidence (R20/R21 honest-negative)

| Source | Finding |
|---|---|
| `completed/93_r20_p2_hilbert_astsr_gpu.md` §0 | L5 R² ≥ 0.85 on PlatinAI 4-cell PROJECTED @ Wf-51 GPU retrain (DEFERRED) |
| `completed/93_r20_p2_hilbert_astsr_gpu.md` §1.1 | MEASURED on CPU smoke: real PlatinAI 4-cell 1-seed × 5-epoch R² ≈ -0.02 (collapse detector fires at epoch 4) |
| `completed/93_r20_p2_hilbert_astsr_gpu.md` §1.2 | L7 Multi-PH + PLL R² ≥ 0.70 PROJECTED; CPU smoke 32-bin Fourier + 128-d PLL joint dim 92 R² ≈ +0.04 |
| `completed/96_r20_p5_closure_catherine.md` §3.2 | Static closure reaches empirical ceiling 0.85 per-metal best (Cd) — even with closure-theorem relaxation, the gate would land at 0.85-0.90 |
| `completed/100_wf71_p3_wf49_cross_round2.md` | PlatinAI 4-cell per-cohort: only A2780cis (cisplatin-resistant) shows R²=+0.037; 3/4 cohorts at predict-the-mean |

**Structural reason**: the current L5/L7/L8 stack measures **static pIC50 R²** — a single snapshot of binding affinity. Real metallodrug efficacy depends on **dynamic lability** (ligand-exchange rate), **kinetic trans-effect** (cisplatin aquation t1/2 ≈ 2 hr), and **redox cycling** (Pt(II)→Pt(IV) under oxidative stress). No current gap addresses continuous-time dynamics.

### §2.3 Candidate approach
- **Graph Neural ODE backbone** (Chen et al. 2018 NeurIPS, GRU-ODE-Bayes): each ASTSR site node carries a hidden state; edges (chemical bonds) carry dynamics; ODE solver (dopri5) integrates forward in time
- **Ligand-exchange ODE**: `d[Pt-X]/dt = -k_ex * [Pt-X] + k_ex' * [Pt-X']` for X = Cl⁻, H₂O, GSH
- **Hydrolysis kinetics**: cisplatin Cl⁻/H₂O exchange rates from experimental literature (Reedijk 2008)
- **Redox cycling ODE**: Pt(II) ⇌ Pt(IV) potential + ROS-dependent transition rate
- **Multi-task loss**: static R² (current L5) + kinetic rate constant MSE + t1/2 classification

### §2.4 Priority + first-close horizon
- **Priority**: MEDIUM (post-R21; gating factor is GPU retrain wall-clock)
- **R first close**: R23-P2 (12-month R23 timeline; after R22 GPU bench stabilizes)
- **Depends on**: L5 (Hilbert ASTSR) + L7 (Multi-PH + PLL) — provides ASTSR signature as input features
- **Scan signal count**: 20 (deferred_phase=20)

---

## §3 L15 — Multi-objective Pareto front over toxicity / efficacy / ADMET

### §3.1 One-line definition
> Replace single-scalar optimization across 22 metrics (current L3/L8/L10) with **Multi-objective Pareto front** jointly optimizing toxicity + efficacy + ADMET — dominance-aware search, NSGA-II/III backbone, hypervolume indicator for paper §4 reporting.

### §3.2 Evidence (R20/R21 honest-negative)

| Source | Finding |
|---|---|
| `TODO/INDEX.md §17.1` | 22 metrics × 5 categories: 7 STRONG_REAL / 11 WEAK_REAL / 5+ MOCK / ~6 UNDEFINED / R20 PROJECTED |
| `TODO/INDEX.md §17.1` | m18 (Protein-Metal AUC), m19 (LOMO AUC), m20 (Tumor-specific AUC) all UNDEFINED or WEAK_REAL_NEEDS_MORE_DATA |
| `completed/85_mock_audit.md` | 8 critical + 23 major + 98 minor mock sites; toxicity/efficacy/ADMET each treated as single scalar |
| `completed/95_r20_p4_3way_100pocket.md` §0 | 3-way on MetalCytoToxDB R² ≥ +0.30 NOT executed; current single-scalar optimization fails to capture trade-offs |
| `TODO/INDEX.md §17.5` | m4 Success Rate (≥0.65 R19) + m7 Diversity (≥0.75 R19) are in tension; single-scalar sum-of-weights loses Pareto structure |

**Structural reason**: current L3 (ScoreRep + VariancePUCT + MCTS), L8 (ASTSR ↔ Hilbert ↔ MCTS triangle), L10 (3-way) all optimize **single-scalar objectives** (R², success rate, Pearson r). Real drug discovery requires **Pareto trade-off** between toxicity (lower = better) + efficacy (higher = better) + ADMET (multi-axis: solubility, permeability, metabolic stability). Single-scalar aggregation (e.g., weighted sum) loses dominance structure.

### §3.3 Candidate approach
- **NSGA-III backbone** (Deb & Jain 2014): reference-point based multi-objective EA
- **3-objective Pareto**:
  - Obj 1: m18 Protein-Metal binding AUC ≥ 0.78 (R22 target)
  - Obj 2: Toxicity (hERG IC50, AMES, DILI) — minimize
  - Obj 3: ADMET composite (solubility ≥ 10 μM, permeability ≥ -5 logP, metabolic stability ≥ 30 min)
- **Hypervolume indicator**: NSGA-III primary metric; paper §4 main result
- **Skill integration**: reuses `mol-metal-anticancer` + `mol-metal-mechanism-mapping` + new `mol-metal-pareto-front`
- **MCTS ↔ NSGA-III bridge**: use VariancePUCT (L2) as the search operator for offspring generation within NSGA-III

### §3.4 Priority + first-close horizon
- **Priority**: MEDIUM (depends on m18/m19/m20 metric definition + first GPU retrain)
- **R first close**: R22-P3 (Wf-53, 24-week R22 timeline)
- **Depends on**: L3 (ScoreRep + VariancePUCT + MCTS) + L8 (ASTSR ↔ Hilbert ↔ MCTS triangle) — provides search backbone
- **Scan signal count**: 52 (design_gap=26 + deferred_phase=24 + tier1_metric_undefined=2)

---

## §4 L16 — Wet-lab-in-the-loop Bayesian optimization (m18/m19/m20 validation)

### §4.1 One-line definition
> Close the in-silico → wet-lab validation feedback loop with **Bayesian optimization over D-MPNN predictions** — Gaussian-process surrogate on top-20 generated candidates, batch IC50 acquisition function, ORNL/PKU/协和 cooperation on cell-line panel.

### §4.2 Evidence (R20/R21 honest-negative)

| Source | Finding |
|---|---|
| `pending/97_r20_p6_wetlab_ic50.md` §0 | Wet-lab IC50 cooperation with ORNL/PKU/协和 **deferred 6 months** (R20-P6 plan) |
| `pending/97_r20_p6_wetlab_ic50.md` §3 | 6-month wall: P6.1 outreach + synthesis 1mo / P6.2 cell culture + IC50 assay 3mo / P6.3 in-silico-vs-wet-lab agreement Pearson r ≥ 0.7 |
| `completed/97_r20_p6_wetlab_ic50_dryrun.md` | In-silico dry-run + D-MPNN prediction layer SHIP today; **wet-lab cooperation 0 progress** |
| `completed/98_wf44_closure_ood_phase2.md` | Cross-domain OOD Phase 2 re-measurement **NOT RUN**; Wf-44 P2 reserved Q4 2026 |
| `completed/96_r20_p5_closure_catherine.md` §3.3 | MetalCytoToxDB 5-metal OOD reachability 0.0300 floor (Wf-41 P2 carry); Wf-44 P1 catalog C1-C8 |
| `completed/96_r20_p5_closure_catherine.md` §1.3 | PlatinAI 5K reachability 0.5581; aggregate OOD 0.2742 |

**Structural reason**: R20-P6 (TODO-97) ships the **in-silico side** (D-MPNN predictions, dry-run validation) but the **wet-lab feedback loop** is open. Without prospective IC50 validation, the paper §5 claim ("in-silico predictions agree with wet-lab within Pearson r ≥ 0.7") cannot be made. No current L1-L12 gap closes this loop.

### §4.3 Candidate approach
- **Gaussian-process surrogate** (Williams & Rasmussen 2006) on top-20 D-MPNN-predicted candidates with uncertainty quantification
- **Batch acquisition function**: q-EI (q Expected Improvement) for 5-candidate batches per wet-lab cycle
- **Wet-lab cycle**: 2-week compound synthesis + 4-week IC50 assay + 1-week analysis = 7-week cycle
- **Tier-1 metric uplift**: m18/m19/m20 AUC target R22 ≥ 0.72 (paper §3.5 lead claim)
- **Skill integration**: reuses `mol-metal-anticancer` + new `mol-metal-bayes-opt-wetlab`
- **ORNL/PKU/协和 cooperation**: NDA + sample shipment protocol (per TODO-97 P6.1)

### §4.4 Priority + first-close horizon
- **Priority**: HIGH (gates paper §5 final claim + clinical trial feasibility)
- **R first close**: R23-P3 (24-week R23 timeline; clinical-trial Phase A starts after R22 GPU bench stabilizes)
- **Depends on**: L13 (hierarchical multi-metal coord — provides diverse candidate pool) + L15 (Pareto front — provides dominance-aware selection)
- **Scan signal count**: 70 (unmet_gate=32 + deferred_phase=30 + design_gap=8)

---

## §5 L13-L16 priority ranking + R22-R25 timeline anchor

| Priority | Candidate | R first close | Depends on | Key risk |
|---|---|---|---|---|
| **HIGH** | **L13** Hierarchical multi-metal coord | R22-P4 | L1 + L6 + L10 | dative-bond parser design risk (chemistry-aware vs generic) |
| **HIGH** | **L16** Wet-lab-in-the-loop Bayes opt | R23-P3 | L13 + L15 | wet-lab cooperation timeline (6-mo R20-P6 → 24-mo R23-P3) |
| MEDIUM | L14 Graph Neural ODE kinetics | R23-P2 | L5 + L7 | GPU retrain wall-clock + literature kinetics data availability |
| MEDIUM | L15 Multi-objective Pareto | R22-P3 | L3 + L8 | hypervolume metric definition + reference-point selection |

**Aggregate scan signal counts** (330 total pattern-match hits):
- L13: 188 (57%) — highest evidence density; structural ceiling + gate NOT MET
- L16: 70 (21%) — second; wet-lab deferral + OOD honest-negative
- L15: 52 (16%) — third; m18/m19/m20 UNDEFINED + Pareto references
- L14: 20 (6%) — lowest; GPU retrain deferral

---

## §6 LAMBDA_ROADMAP §4.2 proposed extension (12-row → 16-row)

| Gap | Title | Existing L1-L12 anchor | New L13-L16 row |
|---|---|---|---|
| L1-L8 (R17-R18 algorithmic) | closure + MCTS + triangle | existing | unchanged |
| L9-L12 (R19-R20 algorithmic) | CFM e2e + 3-way + MCTS tree + OOD | existing | unchanged |
| **L13** NEW | Hierarchical multi-metal coord | — | **NEW row** |
| **L14** NEW | Graph Neural ODE kinetics | — | **NEW row** |
| **L15** NEW | Multi-objective Pareto front | — | **NEW row** |
| **L16** NEW | Wet-lab-in-the-loop Bayes opt | — | **NEW row** |

**Proposed §4.2 row addition (4 new rows)** — for Wf-72 Phase 2 LAMBDA_ROADMAP patch:

| Gap | TODO # | R17/R18 ship | R19 ship | Wf-36/37/40/41/44 R20 bench | R20 ship | R21 ship | R22+ ship |
|---|---|---|---|---|---|---|---|
| **L13** Hierarchical multi-metal coord | TODO-101 (Wf-72 P1 close) | n/a | n/a | n/a | n/a | n/a | R22-P4 ✅ |
| **L14** Graph Neural ODE kinetics | TODO-102 | n/a | n/a | n/a | n/a | n/a | R23-P2 ✅ |
| **L15** Multi-objective Pareto | TODO-103 | n/a | n/a | n/a | n/a | n/a | R22-P3 ✅ |
| **L16** Wet-lab-in-the-loop Bayes | TODO-104 | n/a | n/a | n/a | n/a | n/a | R23-P3 ✅ |

(Detailed 8-stage cells to be filled by Wf-72 Phase 2 once R22+ plan items ship concrete bench targets.)

---

## §7 MEASURED vs PROJECTED (per `party-twenty-spirit §3.5`)

### §7.1 MEASURED today (CPU-only docs)

| Item | Value | Source |
|---|---:|---|
| R20 close-out files scanned | **12** | `completed/{93,94,95,96,97,98,99,100}_*.md` + `completed/{100_wf71,87j,85}_*.md` + `r20_status_ledger_2026_09_26.md` |
| LAMBDA_ROADMAP §4.2 existing row count | **12** (L1-L12) | `TODO/03_plans/LAMBDA_ROADMAP.md §4.2` |
| Pattern-match signals total | **330** | `analyze_lambda_gaps.py` regex scan |
| L13 signal count | **188** | scan output |
| L14 signal count | **20** | scan output |
| L15 signal count | **52** | scan output |
| L16 signal count | **70** | scan output |
| R20-P fully MEASURED | **0/8** | `r20_status_ledger_2026_09_26.md` §0 |
| Honest-negative carry (R20 status) | **15/50 VERIFIED, 30/50 DEFERRED, 3/50 NOT MET, 2/50 IN FLIGHT** | same |
| INDEX §17.1 metric categories | **5** (STRONG_REAL=7 / WEAK_REAL=11 / MOCK=5+ / UNDEFINED=~6 / PROJECTED=R20) | `INDEX §17.1` |

### §7.2 PROJECTED (reserved for R22-R25 plan owners)

| Item | Why PROJECTED | When MEASURED |
|---|---|---|
| L13 hierarchical multi-metal coord 30-metal reachability ≥ 0.95 | requires chemistry-aware dative-bond parser (out-of-scope for R20) | R22-P4 (Wf-54) |
| L14 GNN-ODE kinetics ligand-exchange t1/2 RMSE ≤ 0.5 | requires GPU retrain wall-clock + literature kinetics data | R23-P2 (post-R22 GPU stabilization) |
| L15 Pareto front hypervolume ≥ 0.6 (3-objective) | requires m18/m19/m20 metric definition + NSGA-III benchmark | R22-P3 (Wf-53) |
| L16 wet-lab in-silico-vs-wet-lab Pearson r ≥ 0.7 | requires 6-month wet-lab cooperation (R20-P6 TODO-97) | R23-P3 (post-R23-P1 LOI signed) |
| L13-L16 row insertion into LAMBDA_ROADMAP §4.2 | Wf-72 Phase 2 doc patch (this doc proposes structure) | Wf-72 Phase 2 (next cycle) |

---

## §8 Honest framing — 0-tolerance per `party-twenty-spirit §3.5`

| Honest finding | Evidence |
|---|---|
| **L13-L16 are CANDIDATE shapes, not committed gaps** | Per this doc §0; R22+ plan owners should re-evaluate against phase-specific evidence before adopting. |
| **Pattern-match signal counts are NOT a quality measure** | 188 L13 hits ≠ "L13 is the right shape" — it reflects regex match density on close-out files. Quality assessment requires R22 plan owner judgment. |
| **No GPU retrain executed for L13-L16 today** | CPU-only docs analysis. L13 dative-bond parser, L14 GNN-ODE kinetics, L15 NSGA-III bench, L16 wet-lab cycle all require GPU/wet-lab phases reserved for R22-R25. |
| **No LAMBDA_ROADMAP §4.2 patch executed today** | §6 proposes the 4 new rows; Wf-72 Phase 2 (next cycle) applies the patch. |
| **No paper work touched** | Per `pending/84_no_paper_task_without_authorization.md` — no `paper/sections/*.tex`, no `paper/main.tex` rebuilds, no reviewer rebuttal. |
| **R20 ship honest-negative carry-forward** | 0/8 R20-P fully MEASURED; 3/50 sub-tasks NOT MET (reachability 0.5962 < 0.95 gate; 0.5744 < 0.95 gate; Wf-44 P2 empty dir); 30/50 DEFERRED (GPU/wet-lab). L13-L16 candidates do NOT resolve these — they surface different dimensions (coordination depth, kinetics, multi-objective, wet-lab loop). |
| **R20-P1 is the ONLY R20-P without close-out file** | `completed/92_r20_p1_*.md` NOT yet written — sign-flip audit SHIP, lazy RDKit import partial, GPU trajectory + close-out pending. |
| **Wet-lab cooperation 6-month deferred** | ORNL/PKU/协和 outreach 0 progress today (TODO-97 P6.1). L16 candidate cannot close this gap until R23-P3 wet-lab cooperation starts. |

**Per `party-twenty-spirit §3.5`**: This Wf-72 Phase 1 doc ships **honest framing** — no MEASURED numbers fabricated, no PROJECTED numbers labeled as MEASURED, no GPU/wet-lab phases silently claimed ship. The 4 L13-L16 candidates are **CANDIDATE shapes** based on R20/R21 honest-negative findings, awaiting R22+ plan owner judgment.

---

## §9 Cross-references

### §9.1 Inputs (R20/R21 ship surface)

- **`TODO/completed/r20_status_ledger_2026_09_26.md`** (Wf-70 Phase 1, 308 lines, 50 sub-task R20 ledger) — primary input
- **`TODO/completed/100_r20_r21_consolidation.md`** (Wf-70 Phase 2, 12-row L1-L12 expansion) — primary input
- **`TODO/completed/93_r20_p2_hilbert_astsr_gpu.md`** (Wf-56 R20-P2, L5+L7+L8+L11) — honest-partial SHIP
- **`TODO/completed/94_r20_p3_multiph_pll_feature.md`** (Wf-66 R20-P3, L7+L3) — honest-partial SHIP
- **`TODO/completed/95_r20_p4_3way_100pocket.md`** (Wf-67 R20-P4, L6+L10) — honest-negative partial SHIP
- **`TODO/completed/96_r20_p5_closure_catherine.md`** (Wf-68+Wf-57+Wf-67 R20-P5, L1+L6+L12) — Catherine 6/6 MEASURED, reachability 0.5962 NOT MET
- **`TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md`** (Wf-58 R20-P6, m18/m19/m20) — in-silico SHIP, wet-lab deferred
- **`TODO/completed/98_wf44_closure_ood_phase2.md`** (Wf-57 Wf-44 P2, L12 Phase 2) — partial SHIP
- **`TODO/completed/99_r20_skill_ecosystem.md`** (Wf-55 R20 skills, S1-S8) — partial SHIP
- **`TODO/completed/100_wf71_p3_wf49_cross_round2.md`** (Wf-71 P3 Wf-49 cross-dataset) — F5_RDKIT_2D R² ≥ 0 mean macro
- **`TODO/completed/87j_wf44_root_cause_fixes.md`** (Wf-59 Wf-44 root-cause C1+C4+C6 fixes) — 52 tests PASS
- **`TODO/completed/85_mock_audit.md`** (Wf-42 mock inventory) — 8 critical + 23 major + 98 minor

### §9.2 Outputs (this Wf-72 Phase 1 ship surface)

- **`TODO/pending/r22_l13_l16_gap_analysis.md`** (this doc, ~400 lines) — primary deliverable
- **`molmetal/scripts/analyze_lambda_gaps.py`** (~340 LOC) — regex-based signal extractor
- **`molmetal/reports/wf72_l13_l16/gap_signals.json`** (per-candidate signal count + supporting files)
- **`molmetal/reports/wf72_l13_l16/synthesis.md`** (machine-readable summary table)

### §9.3 Parent plans + sibling workflows

- **`TODO/03_plans/LAMBDA_ROADMAP.md`** §1 (R20-R21 plan) + §2 (R21 plan) + §3 (R22-R25 long-term) + §4.2 (12-row L1-L12 trajectory) — proposed §4.2 4-row extension in §6
- **`TODO/INDEX.md`** §17.1 (22 metrics × 5 categories) + §17.3 (R20 plan table) + §17.5 (honest framing NEW) — proposed §17.6 L13-L16 candidate row
- **`TODO/pending/97_r20_p6_wetlab_ic50.md`** (R20-P6 wet-lab, 6-mo deferred) — L16 first-close anchor
- **`TODO/pending/98_wf44_closure_ood_phase2.md`** (Wf-44 P2 L12 Phase 2, Q4 2026) — L13 OOD Phase 2 anchor
- **`TODO/pending/84_no_paper_task_without_authorization.md`** (paper-gate, CRITICAL) — honored
- **`.claude/skills/party-twenty-spirit/SKILL.md`** §3.5 (0-tolerance honest framing) — applied throughout

### §9.4 Phase 2 plan (Wf-72 next cycle, NOT this doc)

- Apply §4.2 4-row extension to `LAMBDA_ROADMAP.md`
- Apply §17.6 candidate row to `INDEX.md`
- Create `TODO/pending/{101,102,103,104}_*.md` 4 NEW R22-P plan files (L13-L16)
- Re-evaluate L13-L16 candidate shapes against R22 GPU bench + wet-lab cooperation timeline
- CPU-only — no GPU retrain; paper-gate honored

---

## §10 One-line verdict

Wf-72 Phase 1 SHIPs **4 candidate Lambda gaps L13-L16** based on R20/R21 ship honest-negative findings:
- **L13** (HIGH, R22-P4): Hierarchical multi-metal coordination geometry (188 scan signals)
- **L14** (MEDIUM, R23-P2): Graph Neural ODE for metallodrug kinetics (20 scan signals)
- **L15** (MEDIUM, R22-P3): Multi-objective Pareto front over toxicity/efficacy/ADMET (52 scan signals)
- **L16** (HIGH, R23-P3): Wet-lab-in-the-loop Bayesian optimization (70 scan signals)

Total **330 pattern-match signals** across 14 R20/R21 close-out files + INDEX §17 + LAMBDA_ROADMAP §4.2. L13-L16 are **CANDIDATE shapes** awaiting R22+ plan owner judgment. Wf-72 Phase 2 will apply the §4.2 4-row extension + create 4 NEW R22-P plan files.
