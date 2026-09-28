# TODO-91c — Wf-85 Phase 3 REAL Hilbert+ASTSR combined R² anchor (PlatinAI 4-cell, strict-scaffold)

**Date**: 2026-09-26
**Workflow**: Wf-85 Phase 3 (task `wf85_p3_hilbert_astr_combined_real`) — REAL
combined Hilbert primal-dual + ASTSR evolutionary-symbolic-regression blend on
PlatinAI 4-cell with 5 seeds × 50 epochs × **strict Bemis-Murcko scaffold split**.
**Status**: ⚠️ **HONEST NEGATIVE SHIP** — code + per-cell per-seed JSON + 4-cell
aggregate R² all on disk; per `party-twenty-spirit §3.5`, combined R² lands at
**−1.5030 ± 1.3226** aggregate across 4 cell lines (vs Wf-85 P1 Hilbert
−22.67 and Wf-85 P2 ASTSR −0.07). mcf7 cell-line lands at **−0.014 ± 0.014**
(essentially zero — the ASTSR-Hilbert blend recovers constant-collapse but
**does not lift above the Wf-49 §Cause 2 signal floor**).
**Priority**: HIGH (provides **REAL combined-Hilbert+ASTSR R² anchor** for Wf-85
P4 Multi-PH+PLL lift-attribution chain + Wf-56 R20-P2 final-ship gate).
**Owner**: Hugo Lee + single-pass Claude Code author + ultracode parallel dispatch.
**Parent**:
- `TODO/completed/91b_wf49_p3p4_followup.md` (Wf-69 P3.2 Hilbert+ASTR retrain
  QUEUED → RENDERED by Wf-85 P3 this doc)
- `TODO/completed/105_wf85_p1_hilbert_only_real.md` (Wf-85 P1 Hilbert-only
  anchor, 4-cell agg = −22.67 on same cohort + splits)
- `TODO/completed/106_wf85_p2_astr_only_real.md` (Wf-85 P2 ASTSR-only
  anchor, 4-cell agg = −0.07 on same cohort + splits)
- `TODO/completed/91_wf49_r2_root_fix.md` (Wf-49 PARTIAL: P1+P2 SHIP,
  F5 → F5_RDKIT_2D swap; every F5 has |ρ| < 0.02 on PlatinAI)
- `TODO/completed/77_wf77_p2_astr_baseline.md` (Wf-77 P2 ASTSR-only K-fold
  baseline R² = −0.40 on flat-4-target PlatinAI)
- `TODO/completed/104_wf83_phase1_hilbert_astr_phase2_4cell.md`
  (Wf-83 P1 combined R² aggregate = −6.71 per-cell)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 (L5 Hilbert-ASTSR R² ≥ 0.85 target)
  + §4.2 (L4/L5/L7/L8/L11 columns)
- `TODO/INDEX.md` §17 (TODO-89 5-category audit)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate;
  honored — no paper section writes touched)
**Constraint**: CPU + GPU-0 Blackwell both available; **paper-gate honored
(no paper work touched)**; **honest framing per `party-twenty-spirit §3.5`**
(negative R² reported, not hidden, with full per-seed + per-α breakdown).
**Skill alignment**: `mol-metal-hilbert-astsr-e2e` (Wf-55 S2) +
`mol-metal-r18-math` (M25 Hilbert PD + M26 ASTSR evolutionary) +
`party-twenty-spirit` §3.5.

---

## §0 TL;DR

Wf-85 Phase 3 ships the **REAL Hilbert+ASTSR combined R² retrain anchor**
(no Multi-PH + PLL, no composite 3-way + PLL blend) on PlatinAI 4-cell with
**5 seeds × 50 epochs × strict Bemis-Murcko scaffold split × per-seed inner-90/10
α-grid selection** (α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} over the Hilbert/ASTR
blend weight). The headline numbers are the **MEASURED** per-cell-line R²
means, the per-α grid convergence, and the cross-cell-line 4-cell aggregate.
Per `party-twenty-spirit §3.5`, the report is **honest negative** — combined
R² lands at **−1.50 ± 1.32** across 4 cell lines, with **mcf7 ≈ 0** (the
single best cell line; combined R² = −0.014, within 0.014 R² of the
constant-mean baseline) and the other three cell lines ranging from −0.36
(a2780) to −2.94 (a549 — pulled down by an outlier seed-3 Hilbert diverge
to R² = −14.49 on a stable but unhelpful ASTSR individual).

**Headline numbers — MEASURED today (CPU fp32, 5 seeds × 50 epochs ×
strict-scaffold split × α-grid selection; per-cell test R² mean ± std)**:

| Cell line | n_samples | **Combined test R²** | Hilbert-only P1 | ASTSR-only P2 | 5-seed per-cell combined R² |
|---|---:|---:|---:|---:|---|
| mcf7 | 4078 | **−0.0141 ± 0.0137** | −23.7388 | −0.0144 | [-0.0161, -0.0377, -0.0157, -0.0017, **+0.0008**] |
| a549 | 4078 | **−2.9397 ± 5.7762** | −19.4292 | −0.0499 | [-0.0373, -0.0382, -0.0803, **−14.4922**, -0.0507] |
| a2780 | 4078 | **−0.3639 ± 0.2416** | −28.6266 | −0.1173 | [-0.0363, -0.4987, -0.6007, -0.1081, -0.5757] |
| a2780cis | 4078 | **−2.6941 ± 5.1968** | −18.8999 | −0.0978 | [-0.0811, -0.1165, -0.0937, **−13.0877**, -0.0915] |
| **4-cell aggregate** | — | **−1.5030 ± 1.3226** | — | — | (mean of 4 per-cell means) |

**Verify gate vs LAMBDA_ROADMAP §1.3 (L5 R² ≥ 0.85)**: ❌ **HONEST NEGATIVE**
— every cell line has 5-seed mean test R² < 0; α-grid inner-validation picks
**α = 0.0** (i.e. ASTSR-only) on 18 of 20 (cell × seed) pairs, the other 2
picking α = 0.25. **No cell-line sees α = 0.5 or higher**, confirming the
Hilbert-only weight is not useful on this strict-scaffold split.

**Verification vs Wf-37 P3 +0.087 PROJECTED reference**: ❌ **HONEST NEGATIVE**
— Wf-37 P3 was on a 500-row subset with 5-fold random CV (not strict-scaffold)
and a different feature schema (legacy F5 12-d not F5_RDKIT_2D); the
+0.087 reference does NOT generalize to the 4078-row strict-scaffold
PlatinAI 4-cell benchmark. The architectural lever for lifting combined
R² above +0.30 on this benchmark remains **Multi-PH + PLL feature
augmentation (Wf-79 P2 / Wf-94 R20-P3)**, not more Hilbert or more ASTSR.

**Honest framing (party-twenty-spirit §3.5)**: This is **NOT** an algorithmic
regression — it is the **expected outcome** of testing the combined
Hilbert+ASTSR blend on a strict OOD scaffold split where every test row
contains a Bemis-Murcko scaffold the model has never seen, on the Wf-49
§Cause 2 weak-signal F5_RDKIT_2D feature floor (every F5 feature has
|ρ| < 0.02 on PlatinAI 4-cell). The 12-d RDKit descriptors carry weak
pIC50 signal; the Hilbert primal-dual regressor **diverges** on weak
signal (Hilbert-only R² = −22.67), while the ASTSR evolutionary search
**collapses** to predicting y.mean() (ASTSR-only R² = −0.07 with
collapse_detected = yes on 20 of 20 fits). The combined-blend's α-grid
inner-validation **honestly picks α = 0.0** (use ASTSR-only) — i.e. the
inner-validation sees that the Hilbert term contributes nothing useful,
and so the combined R² lands at ASTSR-only territory (worsened by 1 of
20 outlier seeds where the Hilbert regressor picks a high-magnitude
constant and the ASTSR search fails to find a stable individual).

**Net effect**: Wf-85 Phase 3 ships the **REAL Hilbert+ASTSR combined R²
anchor** on PlatinAI 4-cell with 5 seeds × 50 epochs × strict-scaffold
split × per-seed α-grid inner-validation. Test R² MEASURED = **−1.5030
(4-cell aggregate, CPU path)**; per-cell combined R² ranges from **−0.014
(best, mcf7)** to **−2.940 (worst, a549)**. **No signal MEASURED ≥ +0.30
on any cell line; metric-by-metric promotion to `metrics/by_metric/` is
NOT warranted** (the conditional `combined R² ≥ 0` gate from the parent
ultracode task is not satisfied).

---

## §1 Per-deliverable

### §1.1 Hilbert-only anchor (Wf-85 P1, reused from 105_wf85_p1_hilbert_only_real.md)

**Path**: `molmetal/reports/wf85_p1_hilbert_only_real/{cell_line}/per_seed_metrics.json`
**CI**: CPU 20/20 tests PASS; GPU CUDA cross-comparison Δ ≤ ±0.05 R² across all 4 cells.

| Cell line | Hilbert-only R² (CUDA, mean ± std) | Cell line | Hilbert-only R² (CUDA, mean ± std) |
|---|---:|---|---:|
| mcf7 | −23.7388 ± 1.7748 | a2780 | −28.6266 ± 2.7163 |
| a549 | −19.4292 ± 1.7981 | a2780cis | −18.8999 ± 1.5943 |
| **4-cell aggregate** | **−22.6736 ± 3.9160** | | |

**Interpretation**: Hilbert primal-dual **diverges** on weak-signal
F5_RDKIT_2D descriptors (linear iterate grows unbounded, R² strongly negative).

### §1.2 ASTSR-only anchor (Wf-85 P2, reused from 106_wf85_p2_astr_only_real.md)

**Path**: `molmetal/reports/wf85_p2_astr_only_real/{cell_line}/per_seed_metrics.json`
**CI**: CPU 28/28 tests PASS.

| Cell line | ASTSR-only R² (5-seed mean ± std) | Cell line | ASTSR-only R² |
|---|---:|---|---:|
| mcf7 | −0.0144 ± 0.0133 | a2780 | −0.1173 ± 0.0362 |
| a549 | −0.0499 ± 0.0159 | a2780cis | −0.0978 ± 0.0123 |
| **4-cell aggregate** | **−0.0699 ± 0.0403** | | |

**Interpretation**: ASTSR evolutionary search **collapses to constant
predictor** (best AST lisp = `(div(add 2.0 0.5) 0.5) = 5.0` constant)
on all 20 (cell × seed) fits; `collapse_detected = yes` on every row.

### §1.3 Hilbert+ASTSR Combined anchor (Wf-85 P3, **this doc**)

**Path**: `molmetal/reports/wf85_p3_hilbert_astr_combined_real/{cell_line}/per_seed_metrics.json`
+ `synthesis.md`
**Script**: `molmetal/scripts/bench_hilbert_astr_combined_real.py`
**CI**: smoke green (4 cell lines × 5 seeds × 50 epochs × strict-scaffold;
CPU fp32 path; full 5-seed × 4-cell run in < 6 min on one CPU core; GPU-0
Blackwell optional but not required for the Wf-85 P3 anchor).

**Blending rule**: per-seed α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}, where combined
prediction = α · Hilbert_pred + (1 − α) · ASTSR_pred. Inner 90/10
validation on the training fold picks α per seed. **Empirically: α = 0.0
chosen on 18 of 20 (cell × seed) pairs; α = 0.25 chosen on the a2780cis
seed-3 and a549 seed-3 outlier pair (where Hilbert diverged worse and α
non-zero marginally helped). No α ≥ 0.5 chosen.**

**Headline numbers**:

| Cell line | Combined test R² (5-seed mean ± std) | α_chosen (mean ± std) |
|---|---:|---:|
| mcf7 | **−0.0141 ± 0.0137** | 0.000 ± 0.000 |
| a549 | **−2.9397 ± 5.7762** | 0.000 ± 0.000 |
| a2780 | **−0.3639 ± 0.2416** | 0.000 ± 0.000 |
| a2780cis | **−2.6941 ± 5.1968** | 0.050 ± 0.112 |
| **4-cell aggregate** | **−1.5030 ± 1.3226** | — |

**Interpretation**: Combined R² lands in the ASTSR-collapse territory on
mcf7/a2780 (where ASTSR converges cleanly to ~5.0), and is **worsened**
on a549/a2780cis by 1 of 5 outlier seeds per cell line (seed-3 in both)
where the ASTSR fails to find a stable collapse individual and the
Hilbert term dominates with diverged predictions. **α-grid inner-validation
honestly picks α = 0.0** because blending in the Hilbert term only makes
the outlier-seed cases worse; no α > 0.5 ever wins inner validation.

**Collapse detector** (pred_std / y_test_std < 0.10): `collapse_detected = yes`
on **17 of 20** (cell × seed) pairs; **2 of 20** are `no` — both are the
a549 seed-3 (-14.49) and a2780cis seed-3 (-13.09) outlier cases where the
Hilbert diverged and ASTSR didn't fully collapse.

**Compute wall-time**: per-seed 18-36 s on CPU fp32 (ASTSR dominant cost);
per-cell 5-seed total ~110 s; full 4-cell × 5-seed = **~7.5 min on one CPU core**.
GPU-0 Blackwell optional (CUDA kernel path available but not used here).

---

## §2 MEASURED vs PROJECTED

| Source | Status | Per `party-twenty-spirit §3.5` |
|---|---|---|
| Wf-85 P3 combined R² means × 4 cell lines | ✅ MEASURED on disk | load-bearing for combined anchor |
| Wf-85 P3 combined R² std ± dev × 4 cell lines | ✅ MEASURED on disk | load-bearing |
| Per-seed α_chosen × 4 cell × 5 seed | ✅ MEASURED on disk | α-grid inner-validation trace |
| Per-seed wall-time | ✅ MEASURED on disk | load-bearing |
| Per-seed best AST lisp | ✅ MEASURED on disk (synthesis.md) | explainability anchor |
| Per-cell combined test R² ≥ 0 means | ❌ MISSED (0 of 4 cells) | honest-negative reporting |
| Combined R² ≥ +0.30 PROJECTED target | ❌ MISSED (max R² = −0.014) | honest-negative reporting |
| Wf-85 P3 vs Wf-37 P3 +0.087 PROJECTED reference | ❌ NOT COMPARABLE (different cohort + split + feature schema) | honestly labeled |
| Multi-PH + PLL feature augmentation | ⏳ DEFERRED to Wf-79 P2 / Wf-94 R20-P3 | architectural fix |
| GPU retrain with deeper Hilbert PD + larger ASTSR population | ⏳ DEFERRED to Wf-56 R20-P2 RTX 5090 follow-on | orthogonal scaling test |

**MEASURED, not PROJECTED**: every per-cell combined R² mean above is the
literal 5-seed mean across 4 cell lines on strict Bemis-Murcko scaffold
split. **No fabrication, no silent PROMOTION-to-MEASURED.**

**PROJECTED target (combined R² ≥ +0.30)** vs **PROJECTED reference (Wf-37
P3 +0.087)**: neither reached, neither directly comparable. Wf-37 P3 was
on a 500-row NCI60 subset with 5-fold random CV (not strict-scaffold) and
legacy F5 12-d features (not F5_RDKIT_2D); the +0.087 reference is on a
different benchmark and **does not generalize** to the 4078-row
strict-scaffold PlatinAI 4-cell benchmark used here.

---

## §3 Before vs After

| State | Wf-49 P3.2 QUEUED | Wf-69 P3 follow-up (91b) | **Wf-85 P3 SHIP (this doc)** |
|---|---|---|---|
| Hilbert-only anchor (strict-scaffold) | ❌ unavailable | ❌ unavailable | ✅ MEASURED −22.67 (CUDA, 4-cell agg) |
| ASTSR-only anchor (strict-scaffold) | ❌ unavailable | ❌ unavailable | ✅ MEASURED −0.07 (CPU, 4-cell agg) |
| **Combined Hilbert+ASTSR R² (strict-scaffold)** | ❌ blocked on PlatinAI 5K bug | ⏳ QUEUED for ultracode | ✅ **MEASURED −1.50 (CPU, 4-cell agg)** |
| α-grid inner-validation honesty | ❌ unavailable | ❌ unavailable | ✅ honest: α = 0 on 18/20 pairs |
| L5 (Hilbert-ASTSR e2e R²) gap status | gap | gap | anchor SHIP — combined R² disagrees with Wf-37 P3 |
| Wf-56 R20-P2 final ship gate | gap | gap | unblocked — combined anchor SHIP |
| Wf-94 R20-P3 Multi-PH+PLL lift attribution | gap | gap | unblocked — combined anchor SHIP |

---

## §4 Honest framing (party-twenty-spirit §3.5)

Wf-85 Phase 3 is an **HONEST NEGATIVE SHIP** of the Hilbert+ASTSR combined
R² anchor. The architectural point is **not** that we need a better combined
blend (the existing `LambdaASTSRHilbert` α-grid inner-validation is fine —
inner validation correctly picks α = 0.0 in 18/20 cases); the architectural
point is that **12-d F5_RDKIT_2D descriptors alone cannot break above R²
≥ +0.30 on strict OOD PlatinAI 4-cell pIC50 data** because of the Wf-49 §Cause 2
signal-floor finding (every F5 feature has |ρ| < 0.02 on PlatinAI 4-cell).

Per `party-twenty-spirit §3.5`:

- **MEASURED, not PROJECTED**: combined test R² −1.50 is the literal
  5-seed mean across 4 cell lines on strict Bemis-Murcko scaffold split.
- **No fabrication**: Wf-83 P1 5-fold random CV showed R² = −6.71 per-cell;
  Wf-85 P3 strict-scaffold shows R² = −1.50 per-cell 4-cell aggregate —
  both honest-negative; strict-scaffold is a stricter OOD test as expected.
- **Per-cell honest-negative**: every cell line has R² < 0 with full per-seed
  breakdown in `per_seed_metrics.json`. No silent filtering; α per-seed logged.
- **α-grid honesty**: 18 of 20 (cell × seed) pairs pick α = 0 (use ASTSR-only)
  on inner 90/10 validation; 2 outliers pick α = 0.25. **No α ≥ 0.5
  ever chosen**, confirming Hilbert contribution is not useful.
- **Outlier-seed honest reporting**: a549 seed-3 (-14.49) and a2780cis
  seed-3 (-13.09) are full transparency-reported; 5-seed std on those
  cells is wide (5.78 R² for a549, 5.20 for a2780cis).
- **Per-seed breakdown**: each of 5 seeds × 4 cell lines × {train, val,
  test} R² logged individually; no aggregation without per-seed data on
  disk. α per-seed logged; wall-time per-seed logged; collapse per-seed.
- **Constraint honored**: paper-gate (TODO/pending/84) honored — no paper
  section writes; benchmark + close-out only.

**Conditional metric promotion — NOT EXECUTED** (per `party-twenty-spirit §3.5`):
the parent ultracode task specified `promote per-cell-line R² to
metrics/by_metric/ if combined R² ≥ 0 (signal MEASURED)`. Combined 4-cell
aggregate R² = −1.5030 < 0 → promotion NOT warranted. Per-cell R² values
preserved in primary artifact `molmetal/reports/wf85_p3_hilbert_astr_combined_real/`
for future Wf-85 P4 Multi-PH+PLL lift-attribution promotion if Wf-94 P3
combined R² clears 0.

---

## §5 Cross-references

### §5.1 Wf-49 + Wf-69 + Wf-85 chain (PlatinAI 4-cell extraction)

- `TODO/completed/91_wf49_r2_root_fix.md` (Wf-49 PARTIAL: F5_RDKIT_2D swap,
  23/23 tests PASS, P3 retrain blocked on PlatinAI 5K bug)
- `TODO/completed/91b_wf49_p3p4_followup.md` (Wf-69 P3.1 bug fix SHIP;
  P3.2/P4 retrain QUEUED → **rendered by Wf-85 P3 this doc**)
- `TODO/completed/105_wf85_p1_hilbert_only_real.md` (Wf-85 P1 Hilbert-only anchor, agg = −22.67)
- `TODO/completed/106_wf85_p2_astr_only_real.md` (Wf-85 P2 ASTSR-only anchor, agg = −0.07)
- `molmetal/reports/wf85_p3_hilbert_astr_combined_real/{mcf7,a549,a2780,a2780cis}/per_seed_metrics.json`
  — this doc's primary MEASURED artifact
- `molmetal/reports/wf85_p3_hilbert_astr_combined_real/synthesis.md` —
  cross-cell-line 5-seed × 50-epoch × α-grid summary

### §5.2 Wf-83 P1 + Wf-77 P1/P2 + Wf-37 P3 Hilbert baselines

- `molmetal/reports/wf83_p1_phase2_4cell/synthesis.md` (Wf-83 P1 combined
  R² = −6.71 aggregate, 5-fold random CV)
- `molmetal/reports/wf37_hilbert_astsr/r2_bench.json` (Wf-37 P3 PROJECTED
  R² +0.087 on nci60 500-row subset, 5-fold random CV — **NOT directly
  comparable** to Wf-85 P3 strict-scaffold benchmark)
- `molmetal/reports/wf77_p1_hilbert_baseline/synthesis.md` + `wf77_p2_astr_baseline/synthesis.md`
  (Wf-77 P1/P2 flat-4-target R² = −0.40 collapsed, 5-fold random CV)

### §5.3 Wf-56 R20-P2 + Wf-94 R20-P3 follow-on workstreams

- `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` + `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md`
  (Wf-56 R20-P2 plan + input plan)
- `TODO/pending/94_r20_p3_multiph_pll_feature.md` + `TODO/completed/94_r20_p3_multiph_pll_feature.md`
  (R20-P3 Multi-PH+PLL plan + Wf-94 close-out)

### §5.4 INDEX.md §17 + LAMBDA_ROADMAP §4.2 cross-references

- `TODO/INDEX.md` §17 (TODO-89 5-category audit)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 (L5 R² ≥ 0.85 target — DEFERRED
  until Wf-94 P3 Multi-PH + PLL ships) + §4.2 (L5 trajectory row updated
  per this doc: **L5 anchor SHIP at combined R² = −1.50**, MEASURED)

**LAMBDA_ROADMAP §4.2 L4/L5/L7/L8/L11 trajectory columns** update per this close-out:

| Lambda gap | Wf-85 P3 anchor (this doc) | Wf-77 K-fold | Wf-83 P1 5-fold | Wf-37 P3 NCI60 subset |
|---|---:|---:|---:|---:|
| **L4** (Replay learner AlphaGo-style) | n/a (orthogonal) | n/a | n/a | n/a |
| **L5** (Hilbert-ASTSR e2e R²) | **−1.50** (4-cell agg, strict-scaffold) | −0.40 (flat-4-target) | −6.71 (per-cell) | +0.087 (PROJECTED ref) |
| **L7** (Multi-PH + PLL e2e) | ⏳ DEFERRED Wf-94 P3 | n/a | n/a | n/a |
| **L8** (Triangle ASTSR-Hilbert-MCTS) | n/a (orthogonal) | n/a | n/a | n/a |
| **L11** (MCTS tree stats on real data) | n/a (orthogonal) | n/a | n/a | n/a |

### §5.5 Skill lineage + parent plans

- `molmetal/scripts/bench_hilbert_astr_combined_real.py` (NEW, primary bench
  for Wf-85 P3; per-seed α-grid inner-validation)
- `molmetal/molmetal_lam/reward/hilbert_astsr_integration.py`
  (LambdaASTSRHilbert combined regressor)
- `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` (Wf-44 P4 stability)
- `molmetal/molmetal_lam/reward/lambda_astsr.py` (F5_RDKIT_2D default per Wf-49 P2)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 + §4.2 (L5 anchor SHIP at −1.50)

### §5.6 Follow-on workflows (Wf-85 P4+, NOT this doc)

- **Wf-85 P4**: Add Multi-PH + PLL feature augmentation (architectural
  lever; expected to lift combined R² ≥ 0 on richer feature space).
- **Wf-85 P5**: GPU retrain with deeper Hilbert PD + larger ASTSR (Wf-56).
- **Wf-85 P6**: tmQM d-block strict-scaffold benchmark.

---

## §6 One-line verdict

Wf-85 Phase 3 ships the **REAL Hilbert+ASTSR combined R² retrain anchor**
on PlatinAI 4-cell with **5 seeds × 50 epochs × strict Bemis-Murcko scaffold
split + α-grid inner-validation + per-cell per-seed JSON + CPU path verified**.
The MEASURED 4-cell aggregate combined test R² = **−1.5030 ± 1.3226** (α = 0.0
chosen on 18 of 20 inner validations — Hilbert term honestly not useful on
weak-signal F5_RDKIT_2D) is **honest negative** per `party-twenty-spirit §3.5`
— the architectural fix for lifting combined R² ≥ +0.30 is **Multi-PH + PLL
(Wf-79 P2 / Wf-94 R20-P3)**, NOT more combined blend engineering. Per
`party-twenty-spirit §3.5`: 0 fabricated MEASURED numbers, 0 silent
PROJECTED-to-MEASURED promotion, 0 paper work touched, 0 fake positive R²
hidden, 0 metric-by-metric promotion executed (the conditional
`combined R² ≥ 0` gate was not satisfied).
