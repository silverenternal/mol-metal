# Wf-361-B1 — m6_kappa MAML Cross-Scaffold Meta-Learning Exploration

**Date:** 2026-09-29
**Status:** SHIPPED (CPU-only; HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved)
**Files:**
- `molmetal/scripts/wf361b1_m6_kappa_maml.py` (~620 LOC, closed-form Ridge MAML)
- `molmetal/tests/test_wf361b1_m6_kappa_maml.py` (4 NEW tests)
- `molmetal/reports/wf361b1_m6_kappa_maml/result.json` (MEASURED CV-R² + augmented_mean)
- `molmetal/reports/wf361b1_m6_kappa_maml/smoke_log.txt` (smoke run output)

## §1 Context

Per Wf-359-A1: m6_kappa `augmented_mean` DECREASED from 0.4386 → **0.3561** with R20-P3
features on platinai_4cell. The Wf-335-M1 chemprop+PLL CV-R² +0.6814 lifted the
**structural ceiling** but `augmented_mean` remained BELOW the 0.6 threshold. Wf-361-B1
explores MAML cross-scaffold meta-learning as a **NEW architectural path** beyond
feature engineering.

## §2 MAML design

- **Tasks:** top-N Murcko scaffolds by frequency (default N=5).
- **Inner loop:** per-scaffold adaptation via closed-form Ridge (alpha_inner=0.1) on
  K_shot support samples (default K_shot=8) from the held-out scaffold.
- **Outer loop:** meta-Ridge (alpha_meta=1.0) trained on other scaffolds, blended
  70/30 with inner-fitted coefficients (closed-form first-order MAML approximation).
- **LOSO validation:** leave-one-scaffold-out across top-N scaffolds; per-scaffold
  augmented_mean is weighted-aggregated.
- **Features:** chemprop v2 graph + PLL signature (joint_dim=248), bit-identical to
  Wf-335-M1 chemprop+PLL surface.

## §3 MEASURED results (Wf-361-B1)

| Metric | Value |
|---|---:|
| `n_samples` | 60 |
| `n_features` | 248 |
| `n_scaffolds` | 5 |
| `baseline_mean` (m6_kappa) | **0.4511** |
| `augmented_mean_meta` | 0.4239 |
| `augmented_mean_adapted` | **0.4288** |
| `delta_adapted_vs_baseline` | -0.0223 |
| `delta_adapted_vs_meta` | **+0.0048** (MAML inner-loop lift over meta-only) |
| `cv_r2_mean_meta` | -1.4143 |
| `cv_r2_mean_adapted` | **+0.2909** (lift from -1.41 → +0.29 via cross-scaffold transfer) |
| `pass_rate_baseline` | 0.2167 |
| `pass_rate_augmented_adapted` | **0.2500** (+0.0333 lift) |
| `verdict` | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL** |
| `wall_seconds` (3 epochs) | 129.40 |

## §4 Honest framing

Per `party-twenty-spirit` §3.5:

- `augmented_mean_adapted = 0.4288` is BELOW the 0.6 threshold → **gate NOT lifted**.
- Verdict: **HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved** (the Wf-335-M1 verdict
  is unchanged; Wf-361-B1 adds another data point on the architectural exploration
  axis, not a ceiling-break).
- **Measured lift:** MAML inner-loop adaptation gives +0.0048 augmented_mean and
  +1.7 CV-R² over the meta-only baseline. Significant structural recovery via
  cross-scaffold transfer but **insufficient to lift augmented_mean to ≥ 0.6**.
- **Honest framing tag:** another path tried; architectural ceiling partially
  lifted via cross-task meta-learning but **augmented_mean remains 0.43 < 0.6**.

## §5 Comparison with prior Wf

| Workflow | augmented_mean | cv_r2 | verdict |
|---|---:|---:|---|
| Wf-325D baseline | 0.4719 | n/a | HONEST_NEGATIVE (anchor) |
| Wf-335-M1 chemprop+PLL | 0.4386 | **+0.6814** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| Wf-359-A1 R20-P3 (Multi-PH+PLL) | 0.3561 | n/a | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| **Wf-361-B1 MAML cross-scaffold** | **0.4288** | +0.2909 | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL** |

Wf-361-B1 recovers augmented_mean closer to the Wf-335-M1 baseline (+0.0727 above
Wf-359-A1) but still below 0.6. **The structural ceiling is partially re-built via
cross-task transfer but augmented_mean remains BELOW threshold.**

## §6 Cross-references

- Wf-359-A1 baseline 0.3561: `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/m6_kappa_augmented_mean.json`
- Wf-335-M1 chemprop+PLL anchor: `molmetal/reports/wf335m1_chemprop_pll/result.json`
- Wf-333-V4 candidate bench: `TODO/completed/333v4_m6_kappa_alt.md`
- MAML module: `molmetal/scripts/wf361b1_m6_kappa_maml.py`
- Tests: `molmetal/tests/test_wf361b1_m6_kappa_maml.py` (4/4 PASS)
- JSON output: `molmetal/reports/wf361b1_m6_kappa_maml/result.json`

## §7 Standing directives preserved

- §5.1 GPU authorization: **NOT INVOKED** — CPU-only scaffold + sklearn Ridge.
- §5.2 Honest framing: **PRESERVED** — all values MEASURED + tagged.
- §5.3 No paper work: **NOT TOUCHED** — `paper/*` untouched.
- §5.4 Tier-2 markers: **UNCHANGED** — no wet-lab planning, computational only.

## §8 Tests (4 NEW)

| # | Test | Status |
|---|---|---|
| 1 | `test_inner_outer_loop_logic` | PASS |
| 2 | `test_per_scaffold_adaptation_step` | PASS |
| 3 | `test_cross_scaffold_loso_validation` | PASS |
| 4 | `test_backward_compat_no_flag` | PASS |

Run with: `PYTHONPATH=/home/hugo/codes/mol-metal python -m pytest
molmetal/tests/test_wf361b1_m6_kappa_maml.py -v` (4 PASS in 1.68s).

## §9 Remaining work for Wf-362+

The m6_kappa `augmented_mean < 0.6` honest-negative is now **3 workflows deep**:
Wf-335-M1 (chemprop+PLL), Wf-359-A1 (R20-P3), Wf-361-B1 (MAML cross-scaffold). All
attempted architectural paths preserve the HONEST_NEGATIVE_PARTIAL_STRUCTURAL verdict.

Forward-looking options for Wf-362+:
1. **Non-Ridge meta-learner** (gradient-boosted trees or small MLP as the base
   learner; closed-form Ridge is too linear to capture the m6_kappa signal).
2. **Cross-domain transfer** from a different cohort (e.g. NCI-60 9-cell) with
   matched scaffold groups (cross-cohort MAML).
3. **Active learning** — selective query of m6_kappa via oracle + iterative
   Ridge refit on top-K augmented predictions.
4. **Drop m6_kappa** as a Tier-2 paper metric (per AGENTS.md §5.4 computational-only
   focus); replace with m1_auc (A2780 cisplatin) which has multiple CEILING_BREAK
   anchors per Wf-335-M2.

**Recommendation:** Option 4 — re-scope m6_kappa as Tier-2-out-of-scope (the
Shape Kappa Index is a per-molecule druglikeness heuristic from ChemAxon, not a
direct fit target for the m6 family of metrics; the Tier-2 paper §3 already has
m1_auc as the headline anticancer fit metric).