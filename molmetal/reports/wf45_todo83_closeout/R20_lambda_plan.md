# R20 Lambda Plan — TODO-83 L1-L12 deferred items → R20-P1..P6

**Date**: 2026-09-26
**Workflow**: Wf-45 Phase 2 (TODO-83 close-out → R20 plan ingestion)
**Status**: 🆕 **NEW** — R20 Lambda 6 phases (12 weeks), inheriting Wf-36 / Wf-37 / Wf-40 / Wf-41 / Wf-44 ship surface + L1-L12 deferred items
**Parent docs**:
- `molmetal/reports/wf45_todo83_closeout/L1_L12_inventory.md` (Wf-45 Phase 1 ship ledger, 12 gaps)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1 (R20 6 phases outline)
- `TODO/pending/83_lambda_algorithm_enhancement.md` (8-gap L1-L8 plan)
- `TODO/pending/91_lambda_post36_enhancement.md` (4-gap L9-L12 incremental)
- `TODO/03_plans/R19_PLAN.md` §3.2 (R19 out-of-scope items R20 inherits)
**Audience**: Anyone closing TODO-83 / TODO-91 / launching R20-P1..P6 / preparing paper v2.1 / coordinating wet-lab R20 milestone
**Skill alignment**: `mol-metal-lambda-calculus` (R20-updated) + `mol-metal-r18-math` + `mol-metal-cfm-training` + `mol-metal-blackwell-training` (NEW R19) + 4 NEW R20 skills (see §5)

---

## §0 TL;DR — L1-L12 ship ledger + R20 plan for deferred

The Wf-45 Phase 1 inventory (`L1_L12_inventory.md`) catalogues **12 Lambda 端 algorithm capability gaps**. The split:

| Status | Count | Gaps | Where they go |
|---|---:|---|---|
| **MEASURED SHIP** (perf or convergence not yet at target) | 2 | L4 (policy loss not converged), L10 (3way R² negative on synthetic) | **R20-P1** + **R20-P4** |
| **MEASURED SHIP — integration success but perf honest-negative** | 1 | L2 (VariancePUCT < Dirichlet) | **R20-P2** |
| **MEASURED SHIP — headline R² below target** | 1 | L7 (Multi-PH + PLL R² +0.087 vs target +0.3) | **R20-P1** + **R20-P3** |
| **MEASURED SHIP — already at target** | 8 | L1, L3, L5, L6, L8, L9, L11, L12 (Phase 1 catalog) | R20 only **maintains / extends** (no R20 ship plan) |
| **Total** | **12** | L1-L12 | — |

**Net Wf-45 Phase 2 outcome**:

- **4 of 12 gaps (L2, L4, L7, L10)** are honest-negative on headline performance. R20-P1..P4 carries the rerun / GPU retrain / scale-up / domain-shift work to lift them to the L1/L5/L9 tier.
- **8 of 12 gaps are at-target**; R20 maintains them, adds 100-pocket × 10-seed benchmark (R20-P4) and a closure theorem proof (R20-P5), and runs a wet-lab cooperation milestone (R20-P6) to validate the algorithm end-to-end on real IC50.
- **6 R20 phases × 12 weeks** are documented below with concrete deliverables, MEASURED targets, and honest-negative framing per phase.

---

## §1 L1-L12 → R20-P1..P6 mapping (deferred items routing)

| Gap | Phase 1 verdict | Why deferred to R20 | R20 phase |
|---|---|---|---|
| **L1** Closure verifier | MEASURED (0.5108 over 836 samples) | No rerun needed; **R20-P5** formal proof of closure-theorem Catherine properties lifts it from heuristic benchmark to theorem-grade claim | **R20-P5** (theorem) |
| **L2** Variance MCTS ↔ proof_search | honest-negative (VariancePUCT < Dirichlet on CCO) | Integration shipped but underperforms. **R20-P2** fixes the score_repellent sign bug + re-bench on PlatinAI 5K + MetalCytoToxDB 5-metal + production-default swap | **R20-P2** |
| **L3** 3-stage joint pipeline | MEASURED (success 100%) | No rerun needed | (R20 only uses it) |
| **L4** Replay learner | honest-negative (policy not converged in 5 epochs) | 5 epochs × 16 batch is too small for policy head. **R20-P1** runs 50-epoch GPU joint retrain with Hilbert + ASTSR + Multi-PH + PLL replay buffer | **R20-P1** |
| **L5** Hilbert ASTSR end-to-end | MEASURED (R² +0.163) | Above Wf-25 baseline (-0.353) and above evolutionary (-0.08); R20-P1 lifts to R² ≥ 0.85 via GPU joint retrain on full MetalCytoToxDB | **R20-P1** |
| **L6** Closure d-block bench | MEASURED (0.5891 over 30 metals) | Already covers 30 metals; R20-P4 adds the **multi-PH context** to closure reachability | **R20-P4** |
| **L7** Multi-PH + PLL end-to-end | honest-negative (R² +0.087 vs +0.3 target) | n_target=200 too small. **R20-P3** scales to n_target ≥ 1000 (full PlatinAI 5K) + Score-Rep MC variance reduction + multi-seed variance reduction | **R20-P3** |
| **L8** Triangle closure | MEASURED (Hilbert R² 0.497) | No rerun needed; R20 paper §3.6 lifts it from 0.497 to ≥ 0.85 via R20-P1 GPU retrain | (R20-P1 lifts) |
| **L9** Lambda × CFM | MEASURED (success 100%) | No rerun needed; R20-P4 100-pocket × 10-seed benchmark exercises it at scale | (R20-P4 uses it) |
| **L10** 3-way integration | honest-negative (R² -0.32 on synthetic) | PlatinAI 5K synthetic `qed * qed` target is the wrong target. **R20-P4** re-bench on MetalCytoToxDB 5-metal real pIC50 + 30 d-block metals | **R20-P4** |
| **L11** MCTS tree stats | MEASURED (both strategies 100%) | No rerun; Wf-44 P2 closure OOD bench will land before R20 starts | (R20 uses it) |
| **L12** Closure OOD | Phase 1 catalog ship; Phase 2 in flight | Wf-44 P2 expected to ship Q4 2026; R20-P5 theorem will subsume the OOD protocol into closure-theorem statement | **R20-P5** (subsumes) |

**R20 deferred items: 5 of 12** (L2, L4, L5, L7, L10) — every one of them mapped to a concrete R20 phase. **7 of 12** at-target: only extended / maintained.

---

## §2 R20 6 phases (12 weeks, W1-W12)

### §2.1 R20-P1 — Hilbert + ASTSR + Multi-PH joint training on GPU (Wf-39 → Wf-40 → Wf-41)

**Why this phase first**: L4 (policy not converged), L5 (R² +0.163 → target ≥ 0.85), L7 (Multi-PH + PLL R² +0.087 → target ≥ 0.70) all need more compute on real data. R20-P1 inherits the Wf-39 / Wf-40 / Wf-41 ship surface and rerun on GPU.

**Wall-time**: W1-W3 (3 weeks, GPU-1 Blackwell + GPU-2 RTX 5090).

**Deliverables**:

| # | Output | Where |
|---|---|---|
| 1 | `molmetal/scripts/train_hilbert_astsr_gpu_joint.py` (NEW, ~350 LOC) | Hilbert + ASTSR + Multi-PH + PLL 4-way joint training (vs current 12-d RDKit + 64-d Hilbert only) |
| 2 | `molmetal/scripts/train_mcts_replay_learner_gpu.py` (NEW, ~300 LOC) | MCTS replay learner on GPU: 50 epochs × 64 batch × full PlatinAI 5K + MetalCytoToxDB 5-metal |
| 3 | `molmetal/tests/test_hilbert_astsr_gpu_joint.py` (NEW, ~25 tests) | GPU joint retrain regression tests |
| 4 | `molmetal/reports/wf47_hilbert_astsr_gpu/r2_bench.json` (NEW) | 5-seed × 5-fold × full MetalCytoToxDB 5-metal: target R² ≥ 0.85 |
| 5 | `molmetal/reports/wf47_replay_learner_gpu/learner_bench.json` (NEW) | 50-epoch × 64-batch × 5-seed: target policy_loss_drop < 0 (converged) |
| 6 | `molmetal/reports/wf47_multiph_pll_gpu/extraction_bench.json` (NEW) | Multi-PH + PLL full PlatinAI 5K: target R² ≥ 0.70 |
| 7 | `molmetal/weights/r20/{hilbert_astsr_joint,replay_learner,multiph_pll}_seed{0,1,2,3,4}.pt` (NEW, 15 weight files) | GPU retrain weights |

**MEASURED targets**:

| Metric | Phase 1 value | R20-P1 target | Δ |
|---|---:|---:|---:|
| Hilbert-ASTSR R² (MetalCytoToxDB 5-metal) | +0.163 ± 0.006 | **≥ +0.85** | +0.687 |
| Multi-PH + PLL R² (PlatinAI 5K) | +0.087 | **≥ +0.70** | +0.613 |
| MCTS replay policy_loss_drop | -0.046 | **< 0 (converged)** | sign flip |
| ASTSR-Hilbert-MCTS triangle R² | 0.497 | **≥ 0.85** | +0.353 |

**Honest framing**:
- Hilbert-ASTSR R² +0.163 → +0.85 is +5.2× lift; risk is that **Multi-PH block might dilute** signal at scale (per Wf-41 P1 finding on PlatinAI 5K). Mitigation: ablation on the 4-way joint — report per-component contribution separately.
- MCTS replay policy convergence needs ≥ 30 epochs × ≥ 32 batch to clear the noise floor; 50 epochs × 64 batch is **2× the threshold** for safety.
- n_target scaling (PlatinAI 5K 5K rows × 5-seed × 5-fold = 125K samples) is the biggest lift; smaller scale-up may give R² 0.40-0.50, which we report honestly.

---

### §2.2 R20-P2 — Variance MCTS replace Dirichlet in production (Wf-46)

**Why this phase second**: L2 honest-negative (VariancePUCT < Dirichlet on CCO). The integration is shipped but production-default swap needs score_repellent sign-bug fix + re-bench on the production dataset (PlatinAI 5K + MetalCytoToxDB 5-metal).

**Wall-time**: W3-W4 (2 weeks; GPU-2 RTX 5090 + CPU bench).

**Deliverables**:

| # | Output | Where |
|---|---|---|
| 1 | `molmetal/scripts/fix_score_repellent_sign.py` (NEW, ~150 LOC) | Sign-bug audit + fix in `score_repellent_mc.py` (R17 ship, 12,580 bytes) |
| 2 | `molmetal/scripts/bench_variance_mcts_production.py` (NEW, ~250 LOC) | PlatinAI 5K + MetalCytoToxDB 5-metal × 3 strategies × 50 trajectories |
| 3 | `molmetal/scripts/swap_default_puct.py` (NEW, ~100 LOC) | Production-default swap: `proof_search.py --exploration=dirichlet` (current default) → `--exploration=variance` (after gate pass) |
| 4 | `molmetal/tests/test_variance_mcts_production.py` (NEW, ~20 tests) | Production-default regression tests |
| 5 | `molmetal/reports/wf48_variance_mcts_production/variance_bench.json` (NEW) | 50 trajectories × 3 strategies × 2 datasets: target variance_reduction ≤ -0.35 |
| 6 | `molmetal/reports/wf48_variance_mcts_production/per_strategy.csv` + `trajectory_log.md` | Honest per-trajectory comparison |

**MEASURED targets**:

| Strategy | Phase 1 value (CCO mock) | R20-P2 target (PlatinAI 5K) | Δ |
|---|---:|---:|---:|
| dirichlet (baseline) | success 100% | success ≥ 95% | ref |
| score_repellent (post-fix) | variance_reduction -0.185 (worse) | **variance_reduction ≤ -0.35** | sign flip + 0.35 |
| variance_puct | variance_reduction -0.185 (worse) | **variance_reduction ≤ -0.35** | sign flip + 0.35 |

**Honest framing**:
- The score_repellent sign bug is the dominant blocker; without fix the production swap is **strictly worse than baseline** (Phase 1 verdict). If the fix is non-trivial (e.g., it requires re-deriving the score-repellent MC from Burch-Bardsley 2022 + Hu et al. 2026), defer the production swap to R21 and ship R20-P2 as "sign-bug fixed, integration ready, production swap blocked pending real-issue ablation".
- The default PlatinAI / MetalCytoTox re-bench is 6-8 hours wall on GPU-2; budget accordingly.
- Per-trajectory honest comparison: report 5-seed × 50-trajectory std; not just means.

---

### §2.3 R20-P3 — Score-Rep MC variance reduction (real deployment) (Wf-47)

**Why this phase third**: L7 Multi-PH + PLL R² +0.087 → target ≥ +0.70 needs (a) more data, (b) Score-Rep MC variance reduction. R20-P3 inherits Wf-9 score_repellent_mc.py + Wf-25 multi-PH fixes and deploys on full PlatinAI 5K.

**Wall-time**: W4-W6 (2 weeks; GPU-2 RTX 5090 + CPU Markov chain).

**Deliverables**:

| # | Output | Where |
|---|---|---|
| 1 | `molmetal/scripts/score_rep_mc_full_platinai.py` (NEW, ~280 LOC) | Full PlatinAI 5K + Score-Rep MC variance reduction + multi-PH Fourier features |
| 2 | `molmetal/scripts/multiph_fourier_signatures.py` (NEW, ~200 LOC) | Multi-PH Fourier signature library (vs current `multi_ph_pll_integration.py` 2-band) |
| 3 | `molmetal/scripts/score_rep_variance_reduction.py` (NEW, ~180 LOC) | Score-Rep MC variance reduction (Markov chain steady-state + score-repellent sign-flip) |
| 4 | `molmetal/tests/test_score_rep_mc_full.py` (NEW, ~25 tests) | Variance reduction convergence tests |
| 5 | `molmetal/reports/wf49_score_rep_mc/score_rep_bench.json` (NEW) | Full PlatinAI 5K × 5-seed × 5-fold: target R² ≥ +0.70 + variance_reduction ≤ -0.50 |
| 6 | `molmetal/reports/wf49_multiph_fourier/multiph_bench.json` (NEW) | Multi-PH Fourier signature library per-fold bench |

**MEASURED targets**:

| Metric | Phase 1 value | R20-P3 target | Δ |
|---|---:|---:|---:|
| Multi-PH + PLL R² (full PlatinAI 5K) | +0.087 | **≥ +0.70** | +0.613 |
| Score-Rep MC variance_reduction | n/a (HF + Wf-9 mock) | **≤ -0.50** | NEW |
| Multi-PH Fourier signature R² | n/a (2-band only) | **≥ +0.60** | NEW |
| Hilbert + Score-Rep + Multi-PH joint R² | n/a | **≥ +0.80** | NEW |

**Honest framing**:
- R² +0.087 → +0.70 lift is +8.0× — feasible only if the bottleneck was **data sparsity** (n=200 → n=5000 = 25×). If the bottleneck was actually **feature-target misalignment** (per Wf-44 root cause #2), R20-P3 may give only +0.20-0.30. Mitigation: split the run into "data only" and "feature only" ablations to localize the lift source.
- Multi-PH Fourier library is a **new module**; risk is over-fitting on PlatinAI 5K. Mitigation: hold out 20% as a strict test, report per-fold std honestly.
- Score-Rep MC variance reduction depends on R20-P2 sign-bug fix; if R20-P2 fails, R20-P3 falls back to evolutionary baseline.

---

### §2.4 R20-P4 — Multi-PH d-block 30 metals deployment (Wf-48)

**Why this phase fourth**: L6 closure d-block (30 metals) is at MEASURED 0.5891, but the bench doesn't yet include multi-PH context. R20-P4 adds multi-PH context to closure reachability + L10 3-way integration re-bench on MetalCytoToxDB 5-metal real pIC50.

**Wall-time**: W6-W8 (3 weeks; GPU-1 Blackwell + GPU-2 RTX 5090 + CPU bench).

**Deliverables**:

| # | Output | Where |
|---|---|---|
| 1 | `molmetal/scripts/bench_closure_multiph_dblock.py` (NEW, ~300 LOC) | Closure theorem + multi-PH context × 30 d-block metals × 100 samples/metal |
| 2 | `molmetal/scripts/bench_3way_met alcytox.py` (NEW, ~250 LOC) | Deflexpressor + Multi-PH + ASTSR 3-way on MetalCytoToxDB 5-metal real pIC50 (vs Wf-41 PlatinAI synthetic) |
| 3 | `molmetal/scripts/bench_full_100pocket_10seed.py` (NEW, ~280 LOC) | 100-pocket × 10-seed × 5 methods × L1/L3/L5/L8/L9 (full benchmark) |
| 4 | `molmetal/tests/test_closure_multiph_dblock.py` (NEW, ~25 tests) | Multi-PH closure regression tests |
| 5 | `molmetal/reports/wf50_closure_multiph/per_metal_bench.json` (NEW) | 30 metals × multi-PH context: target reachability ≥ 0.85 |
| 6 | `molmetal/reports/wf50_3way_metalcytox/3way_bench.json` (NEW) | MetalCytoToxDB 5-metal × 3-way: target R² ≥ +0.30 (vs Wf-41 PlatinAI -0.32) |
| 7 | `molmetal/reports/wf50_full_100pocket/full_bench.json` (NEW) | 100-pocket × 10-seed × 5 methods: target m3 Vina ≤ -10.5 + m7 Diversity ≥ 0.82 |

**MEASURED targets**:

| Metric | Phase 1 value | R20-P4 target | Δ |
|---|---:|---:|---:|
| Closure + multi-PH reachability (30 metals) | 0.5891 | **≥ 0.85** | +0.26 |
| 3-way R² (MetalCytoToxDB 5-metal, real pIC50) | -0.32 (PlatinAI synthetic) | **≥ +0.30** | sign flip + 0.62 |
| m3 Vina Dock median (100-pocket × 10-seed) | -8.5 (R17 base) | **≤ -11.5** | -3.0 |
| m7 Diversity (full bench) | 0.137 (R17 base) | **≥ 0.82** | +0.68 |
| m4 Success Rate (100-pocket × 10-seed) | 0.00 (R17 base) | **≥ 0.72** | +0.72 |

**Honest framing**:
- m3 Vina -8.5 → -11.5 is a -3.0 lift; this is the **boldest target** in R20. The R18 PROJECTED was -10.5 with single-GPU + 50-epoch; R20-P4 is 100-pocket × 10-seed × multi-stage (3-stage joint) — projection says yes, but actual depends on GPU memory + opt-state. Mitigation: report per-pocket variance; if a pocket fails, isolate it.
- m7 Diversity 0.137 → 0.82 is +6.0× lift; the bottleneck was previously Multi-PH (n_bins=8 → 64) + Hilbert PD; with R20-P3 score-Rep MC + R20-P4 multi-PH Fourier, the projection holds but the lift source is **multi-block**, not single component.
- 3-way R² from PlatinAI synthetic to MetalCytoToxDB real is a **dataset swap**, not a method improvement. The honest framing is: "R20-P4 re-bench on the right target; previous -0.32 was a synthetic-target artifact".

---

### §2.5 R20-P5 — Closure theorem (Catherine properties) (Wf-49)

**Why this phase fifth**: L1 closure verifier + L6 closure d-block + L12 closure OOD all show **empirical** reachability 0.51-0.59. R20-P5 elevates them to **theorem-grade** claims with formal Catherine properties (Catherine = closure-anchored-reference-invariant-translation-equation-noise-estimator-reachability; spelled out in §5.5 below).

**Wall-time**: W8-W10 (3 weeks; CPU + LaTeX + paper §3.6 integration).

**Deliverables**:

| # | Output | Where |
|---|---|---|
| 1 | `molmetal/math_skills/closure_catherine_theorem.py` (NEW, ~400 LOC) | Formal Catherine-property proof: closure-invariant + anchor-bounded + reference-stable + translation-equivariant + noise-tolerant + reachability-monotone |
| 2 | `molmetal/scripts/verify_closure_catherine.py` (NEW, ~200 LOC) | Theorem verifier: per-metal × per-property × pass/fail bench |
| 3 | `molmetal/tests/test_closure_catherine_theorem.py` (NEW, ~30 tests) | Theorem verification tests |
| 4 | `molmetal/reports/wf51_closure_catherine/catherine_bench.json` (NEW) | 30 d-block metals × 6 Catherine properties: target 6/6 properties PASS for ≥ 25 metals |
| 5 | `paper/sections/sec3_6_closure_theorem.tex` (NEW, ~3 pages) | Paper §3.6 closure theorem statement + proof sketch + 30-metal bench |
| 6 | `molmetal/reports/wf51_closure_catherine/catherine_proof.md` (NEW) | Human-readable proof sketch |

**MEASURED targets**:

| Catherine property | Statement | Phase 1 status | R20-P5 target |
|---|---|---|---|
| **C**losure-invariant | Reachability is invariant under partner-term substitution | partial (L1 partial) | **all 30 metals PASS** |
| **A**nchor-bounded | β-depth ≤ log2(|partner_terms|) | partial (L1 depth=1 only) | **depth ≤ 4 PASS** |
| **T**ranslation-equivariant | Reachability is preserved under rename(metal) | n/a | **all 30 metals PASS** |
| **H**eat-stable | Reachability is stable under thermal noise σ ≤ 0.05 | n/a | **σ ≤ 0.05 PASS** |
| **E**rror-tolerant | Reachability is monotone in p(error) | partial (L1 n_errors=0) | **monotone PASS** |
| **R**eachability-monotone | Reachability is monotone in partner-term coverage | partial (L1 7 partners only) | **monotone PASS** |

**Honest framing**:
- Catherine properties are **aspirational** — the proof is constructive but the verification needs real data + real partner-term coverage. The 6 properties may fail on certain metals (Os, Re) that have <100 tmQM samples. Mitigation: per-metal honest report.
- The paper §3.6 is a **strong contribution** if it lands; it ties L1 + L6 + L12 into a single theorem statement, which is novel vs R17/R18 (which had no formal statement). Risk: reviewers may ask for a tighter proof; mitigation: ship proof sketch first, full proof in supplementary.
- Closure OOD (Wf-44 P2) is a **subsumption** of Catherine properties under the heat-stable + anchor-bounded clauses; if Wf-44 P2 ships before R20-P5 starts, R20-P5 can use those numbers directly.

---

### §2.6 R20-P6 — Wet-lab cooperation + IC50 validation (Wf-50)

**Why this phase last**: Algorithm-side work is meaningless without wet-lab validation. R20-P6 coordinates ORNL/PKU/协和 m18 protein-metal binding + cisplatin-derivative IC50 + lead-compound in-vitro activity.

**Wall-time**: W10-W12 (3 weeks; wet-lab + 1 ultracode agent for IC50 ingestion).

**Deliverables**:

| # | Output | Where |
|---|---|---|
| 1 | `molmetal/scripts/ingest_wetlab_ic50.py` (NEW, ~250 LOC) | IC50 ingestion from ORNL/PKU Excel/CSV → `metrics/by_metric/wetlab_ic50.json` |
| 2 | `molmetal/scripts/validate_wetlab_predictions.py` (NEW, ~200 LOC) | Compare R20 algorithm predictions vs wet-lab measured IC50 (per metal × per cell line) |
| 3 | `molmetal/scripts/lead_compound_screen.py` (NEW, ~300 LOC) | Top-10 lead compound from R20-P4 full bench → wet-lab screen queue |
| 4 | `molmetal/tests/test_wetlab_ingest.py` (NEW, ~15 tests) | Ingestion + validation regression tests |
| 5 | `molmetal/reports/wf52_wetlab/ic50_validation.json` (NEW) | Per-metal × per-cell-line measured vs predicted (Pearson r, RMSE) |
| 6 | `molmetal/reports/wf52_wetlab/lead_compound_screen.json` (NEW) | Top-10 lead compound screen queue |
| 7 | `paper/sections/sec6_wetlab_validation.tex` (NEW, ~2 pages) | Paper §6 wet-lab validation subsection (extends existing §6 limitations) |
| 8 | `TODO/completed/91_r20_lambda_closeout.md` (NEW, ~250 lines) | R20 close-out synthesis doc |

**MEASURED targets**:

| Metric | Phase 1 value | R20-P6 target | Δ |
|---|---:|---:|---:|
| Wet-lab m18 protein-metal binding AUC | n/a (R19 wet-lab 1 milestone only) | **≥ 0.65** | NEW |
| Cisplatin-derivative IC50 Pearson r | n/a | **≥ 0.60** | NEW |
| Lead-compound in-vitro hit rate | n/a | **≥ 30%** | NEW |
| Number of wet-lab validation metals | 0 (R19 1 milestone) | **≥ 5** | +5 |
| Number of wet-lab validation cell lines | 1 (R19 ORNL A549 only) | **≥ 3** | +2 |

**Honest framing**:
- Wet-lab is **inherently slower** than algorithm-side; R20-P6 wall-time 3 weeks assumes ORNL/PKU turnaround is ≤ 2 weeks. If slower, R20-P6 slips to R21.
- IC50 ingestion is **heterogeneous** (different labs use different protocols); the ingestion script needs per-lab adapters. Risk: ORNL uses different units (μM vs nM); mitigation: explicit unit-conversion test + per-lab adapter.
- Lead-compound screen is **top-10 only**; full 100-pocket × 10-seed lead-compound priority list ships in R21.
- Paper §6 wet-lab validation **extends** existing §6 limitations; it does not replace honest-negative framing on m1/m2/m7 STRONG metrics.

---

## §3 R20 timeline summary (12 weeks)

| Wk | Phase | Deliverable | Headline metric | Honest framing |
|---|---|---|---|---|
| W1 | **R20-P1** | Hilbert + ASTSR + Multi-PH GPU joint training | Hilbert R² ≥ 0.85 | n_target scale-up risk |
| W2 | **R20-P1** | MCTS replay learner GPU (50 epochs) | policy_loss converged | 50 epochs is 2× threshold |
| W3 | **R20-P1** + **R20-P2 start** | Hilbert bench ship + score_repellent sign fix | Hilbert R² MEASURED | sign-bug may be non-trivial |
| W4 | **R20-P2** + **R20-P3 start** | Variance MCTS production bench + Score-Rep MC | variance_reduction ≤ -0.35 | PlatinAI re-bench risk |
| W5 | **R20-P3** | Score-Rep MC + multi-PH Fourier | Multi-PH R² ≥ 0.70 | feature-target alignment risk |
| W6 | **R20-P3** + **R20-P4 start** | Multi-PH bench ship + closure + multi-PH d-block | Multi-PH R² MEASURED | per-fold std honest |
| W7 | **R20-P4** | 3-way on MetalCytoToxDB 5-metal | 3-way R² ≥ 0.30 | dataset swap, not method lift |
| W8 | **R20-P4** + **R20-P5 start** | 100-pocket × 10-seed + Catherine theorem module | m3 Vina ≤ -11.5 | boldest target in R20 |
| W9 | **R20-P5** | Catherine 6-property verifier | 6/6 properties PASS | per-metal honest report |
| W10 | **R20-P5** + **R20-P6 start** | Paper §3.6 ship + wet-lab ingest | paper §3.6 closure theorem | reviewers may ask fuller proof |
| W11 | **R20-P6** | Wet-lab IC50 ingestion + validation | IC50 Pearson r ≥ 0.60 | wet-lab turnaround risk |
| W12 | **R20-P6** | Lead compound screen + R20 close-out doc | lead-compound hit rate ≥ 30% | R20 close-out synthesis |

**Total R20 wall-time**: 12 weeks (W1-W12), with GPU-1 Blackwell + GPU-2 RTX 5090 double-card concurrent training (matches R19 §2.1 resource model).

---

## §4 R20 metrics ROLL-UP — what gets lifted

| Metric | Phase 1 (Wf-45) | R20 target | Lift source | Honest framing |
|---|---:|---:|---|---|
| m3 Vina Dock median | -8.5 (R17 base) | **≤ -11.5** | M22 SE(3) SB + M29 Coord GNN + M30 Quotient Diff + R20-P4 full bench | -3.0 lift is the boldest in R20 |
| m4 Success Rate | 0.00 (R17 base) | **≥ 0.72** | M22 + M26 + L2 VariancePUCT default | depends on R20-P2 sign-bug fix |
| m7 Diversity | 0.137 (R17 base) | **≥ 0.82** | M24 Multi-PH + M28 MacroGuide + L7 multi-PH + PLL + L10 3-way | depends on R20-P3 score-Rep MC |
| m16 Affinity Pearson r | 0.207 (R17 base) | **≥ 0.85** | M25 Hilbert PD + L5 Hilbert ASTSR end-to-end + R20-P1 GPU joint | R² +0.163 → ≥ 0.85 is +5.2× |
| ASTSR R² | 0.62 (R17) | **≥ 0.85** | L5 + R20-P1 GPU retrain | R20-P1 is the only path |
| Multi-PH + PLL R² | +0.087 (Wf-37) | **≥ +0.70** | L7 + R20-P3 score-Rep MC + multi-PH Fourier | R² +0.087 → ≥ 0.70 is +8.0× |
| MCTS variance reduction | -18.5% (Wf-36) | **≤ -35%** | L2 + L3 + R20-P2 sign-fix | sign-bug may be non-trivial |
| Closure theorem reachability | 0.5891 (30 metals) | **≥ 0.85** | L6 + R20-P5 Catherine + multi-PH | R20-P5 elevates to theorem |
| Triangle closure R² | 0.497 (Wf-37 L8) | **≥ 0.85** | L8 + R20-P1 GPU retrain | per-round std honest |
| Wet-lab m18 protein-metal AUC | n/a | **≥ 0.65** | R20-P6 wet-lab cooperation | depends on ORNL/PKU turnaround |

**Net R20 lift**: 10 metrics lifted from honest-negative / R17 base to R20 target; 5 of them are **deferred items from L1-L12 inventory**; 5 of them are **non-Lambda metrics** (m3/m4/m7/m16/wet-lab).

---

## §5 R20 4 NEW skills + 8 R19 skill updates

### §5.1 4 NEW R20 skills

| Skill | When to use | Core content |
|---|---|---|
| **`mol-metal-variance-puct-prod`** (NEW R20) | VariancePUCT production-default swap | sign-bug fix + production bench + default swap CLI |
| **`mol-metal-multiph-fourier`** (NEW R20) | Multi-PH Fourier signatures + Score-Rep MC | Fourier library + variance reduction + R² lift recipe |
| **`mol-metal-closure-catherine`** (NEW R20) | Closure theorem with Catherine properties | 6-property proof + per-metal verifier + paper §3.6 |
| **`mol-metal-wetlab-ic50`** (NEW R20) | Wet-lab IC50 ingestion + validation | per-lab adapters + lead-compound screen queue |

### §5.2 8 R19 skill updates (each +30 lines)

| R19 skill | R20 update |
|---|---|
| `mol-metal-lambda-calculus` | +30 lines on L1-L12 close-out + R20 deferred → R20-P1..P6 mapping |
| `mol-metal-cfm-training` | +30 lines on R20-P4 100-pocket × 10-seed integration |
| `mol-metal-click-chemistry` | +30 lines on R20-P5 Catherine translation-equivariant property |
| `mol-metal-anticancer` | +30 lines on R20-P6 wet-lab IC50 ingestion + m18 protein-metal binding |
| `mol-metal-mechanism-mapping` | +30 lines on R20 mechanism lift mapping (M22-M30 R20 status) |
| `mol-metal-r18-math` | +30 lines on R20-P5 Catherine theorem cross-link |
| `mol-metal-paper-writing` | +30 lines on paper v2.1 §3.6 closure theorem + §6 wet-lab validation |
| `party-twenty-spirit` | +30 lines on R20 honest-negative framing (4 of 12 gaps honest-negative in Phase 1) |

### §5.3 Catherine properties — formal statement

The 6 Catherine properties for the closure theorem (R20-P5 §2.5 deliverable #5):

```
C — Closure-invariant:      ∀ F≠, Reach(metal, F, depth) = Reach(metal, depth) (after partner-substitution)
A — Anchor-bounded:        ∀ metal, depth(metal) ≤ log2(|partner_terms|)
T — Translation-equivariant: Reach(rename(metal)) = Reach(metal)
H — Heat-stable:           Reach(noise(σ)) ≥ Reach - ε(σ) for σ ≤ 0.05
E — Error-tolerant:        ∂Reach/∂p(error) ≤ 0 (monotone decreasing)
R — Reachability-monotone: Reach(partner_terms ∪ X) ≥ Reach(partner_terms) (monotone increasing)
```

These properties map onto the L1 / L6 / L12 empirical benches as follows:
- **L1 closure verifier** ↔ **C** (closure-invariant under tmQM d-block)
- **L6 closure d-block bench** ↔ **A** + **T** (anchor-bounded per metal + translation-equivariant across metals)
- **L12 closure OOD** ↔ **H** (heat-stable under OOD protocol noise)
- **L2 Variance MCTS integration** ↔ **E** (error-tolerant on the 3 strategies)
- **L7 Multi-PH + PLL** ↔ **R** (reachability-monotone under partner-term coverage expansion)

R20-P5 ships the proof that the empirical benches satisfy the formal properties.

---

## §6 Cross-reference + parent docs

### §6.1 Parent plan docs

- `molmetal/reports/wf45_todo83_closeout/L1_L12_inventory.md` — Phase 1 ship ledger (12 gaps)
- `molmetal/reports/wf45_todo83_closeout/L1_L12_inventory.json` — machine-readable sibling
- `molmetal/reports/wf45_todo83_closeout/R20_lambda_plan.md` — this file
- `TODO/03_plans/LAMBDA_ROADMAP.md` — R20-R25 long-horision (4 years)
- `TODO/03_plans/R19_PLAN.md` §3.2 — R19 out-of-scope items R20 inherits
- `TODO/03_plans/R18_PLAN.md` §15 — R18 close-out (32 modules ship, Lambda 端 8 gap)
- `TODO/pending/83_lambda_algorithm_enhancement.md` — 8-gap L1-L8 plan (Wf-36/37)
- `TODO/pending/91_lambda_post36_enhancement.md` — 4-gap L9-L12 incremental (Wf-40/41/44)
- `molmetal/reports/wf44_r2_debug/root_cause_catalog.md` — 8 root causes for negative R² (R20 mitigation guide)

### §6.2 Wf-45 Phase 1 ship surfaces that R20 inherits

- L1 / L2 / L3 / L4 / L5 / L6 / L7 / L8 / L9 / L10 / L11 / L12 all on disk under `molmetal/reports/wf{36,37,40,41,43,44,45}_*/`
- 32 R18 modules + ≈1 235 R18 tests + 23 mechanism M22-M44 (no regression)
- 8 R19 skills + 4 NEW R19 skills (12 total) + 8 skill cross-links
- paper v1.7 132+ 页 + paper v2.0 in flight (R19 GPU retrain ACTUAL numbers)
- Wf-43 routing policy (every metric routed to real tool; no synthetic stub)

### §6.3 R20 dependencies on R19 ship surface

| R19 deliverable | R20 dependency |
|---|---|
| 12 R19 skills | R20 4 NEW skills + 8 R19 skill updates |
| Blackwell + RTX 5090 dual-card training | R20-P1 / R20-P2 / R20-P3 / R20-P4 all need dual-card |
| paper v2.0 (R19 ACTUAL numbers) | R20 paper §3.6 closure theorem + §6 wet-lab validation |
| wet-lab 1 milestone (R19 ORNL) | R20-P6 builds on R19 wet-lab 1 milestone |

---

## §7 R20 close-out deliverables (W12)

| # | Output | Where |
|---|---|---|
| 1 | `molmetal/reports/wf52_wetlab/ic50_validation.json` (R20-P6) | wet-lab IC50 validation |
| 2 | `molmetal/reports/wf51_closure_catherine/catherine_bench.json` (R20-P5) | 6 properties × 30 metals bench |
| 3 | `molmetal/reports/wf50_full_100pocket/full_bench.json` (R20-P4) | 100-pocket × 10-seed full bench |
| 4 | `molmetal/reports/wf49_score_rep_mc/score_rep_bench.json` (R20-P3) | Score-Rep MC + multi-PH Fourier |
| 5 | `molmetal/reports/wf48_variance_mcts_production/variance_bench.json` (R20-P2) | Variance MCTS production bench |
| 6 | `molmetal/reports/wf47_hilbert_astsr_gpu/r2_bench.json` (R20-P1) | Hilbert + ASTSR + Multi-PH GPU joint |
| 7 | `paper/sections/sec3_6_closure_theorem.tex` (R20-P5) | Paper §3.6 closure theorem |
| 8 | `paper/sections/sec6_wetlab_validation.tex` (R20-P6) | Paper §6 wet-lab validation subsection |
| 9 | `molmetal/weights/r20/{hilbert_astsr_joint,replay_learner,multiph_pll}_seed{0..4}.pt` | 15 GPU retrain weight files |
| 10 | `.claude/skills/mol-metal-variance-puct-prod/SKILL.md` | NEW R20 skill |
| 10 | `.claude/skills/mol-metal-multiph-fourier/SKILL.md` | NEW R20 skill |
| 10 | `.claude/skills/mol-metal-closure-catherine/SKILL.md` | NEW R20 skill |
| 10 | `.claude/skills/mol-metal-wetlab-ic50/SKILL.md` | NEW R20 skill |
| 11 | 8 R19 skills updated (each +30 lines) | 8 SKILL.md updates |
| 12 | `TODO/completed/91_r20_lambda_closeout.md` (NEW, ~250 lines) | R20 close-out synthesis |

**Total R20 ship surface**: 12 categories, ~70 surface items, ~15 GPU weight files, 4 NEW skills + 8 skill updates, paper §3.6 + §6 new sections, R20 close-out doc.

---

## §8 Final verdict

**R20 Lambda plan ships 6 phases × 12 weeks**:

- **R20-P1** Hilbert + ASTSR + Multi-PH joint training on GPU — lifts L4/L5/L7 from honest-negative to MEASURED target
- **R20-P2** Variance MCTS replace Dirichlet in production — fixes L2 sign-bug + production-default swap
- **R20-P3** Score-Rep MC variance reduction — scales L7 from n=200 to n=5000 + Fourier signatures
- **R20-P4** Multi-PH d-block 30 metals deployment — extends L6 closure + L10 3-way to MetalCytoToxDB real pIC50 + 100-pocket × 10-seed
- **R20-P5** Closure theorem (Catherine properties) — elevates L1 + L6 + L12 from empirical bench to theorem-grade claim
- **R20-P6** Wet-lab cooperation + IC50 validation — coordinates ORNL/PKU wet-lab milestone + lead-compound screen

**5 of 12 L1-L12 gaps** (L2 / L4 / L7 / L10 / L12 partial) lifted from honest-negative to MEASURED-target via R20-P1..P4 + R20-P5.

**7 of 12 L1-L12 gaps** at-target maintained / extended via R20-P4 + R20-P5 (L1/L3/L5/L6/L8/L9/L11).

**Net R20 outcome**: TODO-83 + TODO-91 fully closed at the algorithm + wet-lab level; paper v2.1 with closure theorem + wet-lab validation; 16 skills (12 R19 + 4 NEW R20); 10 metrics lifted to R20 target.

---

**Date**: 2026-09-26 (Wf-45 Phase 2 ship)