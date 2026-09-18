# WF-pIC50-Margin-Fix — VERDICT

Date: 2026-09-14
Project: /home/hugo/codes/try_triton_on_rocm
Owner: model/property-prediction line (TODO-18 / WF-Extra-1 follow-up)

---

## 1. Verdict

**SUCCESS on the targeted metric (Option A, hinge margin 0.25 → 1.0).**

| Metric | WF-Extra-1 (margin=0.25) | WF-pIC50-Margin-Fix (margin=1.0) | Delta | Status |
|---|---:|---:|---:|---|
| test_pearson_r (mean ± std, 3 seeds) | 0.198 ± 0.228 | **0.181 ± 0.224** | -0.017 | unchanged (within noise) |
| test_rmse (mean ± std, pIC50) | 0.687 ± 0.042 | **0.705 ± 0.048** | +0.018 | unchanged (within noise) |
| test_spearman_rho (mean ± std) | 0.201 ± 0.197 | **0.161 ± 0.218** | -0.040 | unchanged (within noise) |
| **bound_aware_accuracy (pooled, 67/67)** | **0.000** | **1.000** | **+1.000** | **FIX** |
| bound_aware_accuracy (mean per-seed) | 0.000 | 1.000 ± 0.000 | +1.000 | FIX |
| pred_std per-seed range | [0.04, 0.16] | [0.084, 0.320] | widened | improvement |
| wall-clock (3 seeds, 50 epochs) | ~7 min CPU | ~7 min CPU | n/a | within budget |

The hinge margin=1.0 pIC50 is now **above the regression-to-mean pull (~0.80 pIC50)** that dominated the previous configuration, so the hinge actually fires hard enough to constrain the censored predictions without collapsing pred_std. The Option B follow-up (zero-margin penalty) was **NOT** required and was held in reserve.

---

## 2. Recommendation

**Adopt Option A as the default censor-aware loss configuration for the
Attentive D-MPNN pIC50 predictor on the HeLa48h/dark cohort.** The
following updates are recommended:

1. **Code**: ship the `CENSOR_MARGIN_DEFAULT = 1.0` value and the new
   `--censored-margin` CLI flag in `molmetal/scripts/retrain_pic50_neural.py`
   as the supported default. Document the value in the script docstring.
2. **TODO-18**: keep the "WF-pIC50-Margin-Fix Option A succeeded" entry as
   the active accepted-result block (already appended in append-only style).
3. **Paper §4.6**: the anticancer-specific paragraph now states the
   updated bound-aware accuracy (1.00 ± 0.00) and the margin-fix result;
   the Round-12 pilot is still the gate for per-pocket pIC50 numbers.
4. **Reward channel**: the neural predictor is now eligible to be used
   as a reward (censor compliance is satisfied), but only behind a guard
   that warns when `bound_aware_accuracy < 0.95` on the censored
   held-out set.

**Do NOT** promote the neural predictor to a primary reward channel
without first closing the seed-fragility gap (σ/μ = 1.23 on Pearson r
across 3 seeds). The conditioned ridge baseline (RMSE 0.541 ± 0.089,
Pearson 0.572 ± 0.068) remains the credible ceiling for exact-row
metrics on the same cohort.

---

## 3. Follow-ups

### 3.1 Out-of-scope for this PR (deferred)

- **F1 — Seed-fragility on Pearson r.** σ/μ = 1.23 across 3 seeds is
  unchanged from WF-Extra-1 (1.15). The collapse on seed=42 (r=-0.05)
  persists. Likely fixes (in order of expected effort vs payoff):
  - dropout ↑ 0.1 → 0.3 + weight decay 1e-4;
  - depth ↑ 3 → 4 with hidden ↑ 128 → 192;
  - per-sample loss weighting that splits censored vs exact contribution;
  - auxiliary multi-task head that predicts pIC50 plus the censor
    direction as a separate classifier.
  This is a model-class problem, not a loss-plumbing problem; budget
  would be a separate PR with ≥5 seeds.

- **F2 — Ridge-baseline gap on RMSE/Pearson.** Neural 0.705/0.181 vs
  ridge 0.541/0.572. The model is under-fitting / collapsing. Closing
  the gap requires more data (other cell/time cohorts) or a
  fundamentally different head. Out of scope for the margin-fix PR.

- **F3 — Cross-cohort generalisation.** HeLa24h, HeLa72h, MCF7, A549
  are not measured. The penalty-loss design is a single-cohort artefact
  until a transfer study is run.

### 3.2 Should-be-done-soon (≤1 day, low risk)

- **F4 — Sweep margin at [0.5, 1.0, 1.5, 2.0].** The 1.0 sweet spot is
  inferred from the regression-to-mean pull (~0.80), not measured.
  Running a 4-value × 3-seed × 50-epoch sweep (12 runs, ~28 min wall
  on CPU) would identify whether 1.0 is at the local optimum or if
  1.5 / 2.0 trades a small amount of exact-row RMSE for tighter
  bound-aware accuracy. Output: a margin-vs-metric table that
  can be added to §4.6 as a sensitivity study.

- **F5 — Try Option B in isolation, on the same 3 seeds, to confirm
  it is no better than Option A.** This is a 7-min CPU run that closes
  the design space; if Option B matches Option A, ship both and let
  the user pick; if Option B is worse, the paper can claim
  "Option A is sufficient".

- **F6 — Add a unit test in
  `molmetal/tests/test_pic50_censor_loss.py` (or extend the existing
  one) that asserts `bound_aware_accuracy == 1.0` on a synthetic
  censored toy dataset when margin ≥ 1.0, and `== 0.0` when margin <
  0.1 — locks in the regression so future margin changes do not
  silently re-introduce the WF-Extra-1 defect.

### 3.3 Should-be-done-eventually (≥1 week, higher risk)

- **F7 — Add Option C (combined hinge + L1 + zero-margin).** Listed in
  the brief as "best of both". Not pursued here because Option A
  already fixed the targeted metric, but a future ablation would
  benefit from having it on the table.

- **F8 — Move from CPU retraining to gfx1101 GPU retraining.** The
  subagent Python process did not see CUDA. Re-running with a
  CUDA-visible Python would shrink the 3-seed × 50-epoch wall-clock
  from ~7 min to ~30 s and unlock larger sweeps (e.g. 10 seeds ×
  100 epochs).

- **F9 — Calibration of neural pIC50 for use as a reward.** Even with
  bound-aware accuracy = 1.0, the raw Pearson r is too low to use the
  neural predictor as a top-level reward. A Platt-scale or isotonic
  calibration against the conditioned ridge baseline would let the
  neural predictor be used as a *secondary* reward channel for
  *censor-rich* pockets where the ridge baseline has no coverage.

---

## 4. Honest framing (MEASURED vs PROJECTED)

**MEASURED in this verification (2026-09-14, 3 seeds × 50 epochs,
FROZEN HeLa48h/dark cohort, hinge margin = 1.0)**:
- test_rmse = 0.705 ± 0.048 pIC50
- test_pearson_r = 0.181 ± 0.224
- test_spearman_rho = 0.161 ± 0.218
- bound_aware_accuracy = **1.000 (67/67 censored test rows)** — flipped
  from 0.000 in WF-Extra-1
- pred_std per seed: [0.084, 0.320] — non-trivial spread, no collapse
- 3-seed wall-clock (CPU): ~7 min (well under 30-min budget per option)
- script exit code: 0
- per-seed breakdown: seed=42 r=-0.049 18/18 compliant; seed=0 r=+0.484
  33/33 compliant; seed=1234 r=+0.109 16/16 compliant

**PROJECTED (not measured in this run)**:
- Sweet-spot margin value (0.5 / 1.0 / 1.5 / 2.0 sweep).
- Effect of Option B / Option C as alternatives.
- Cross-cohort transfer (HeLa24h, HeLa72h, MCF7, A549).
- Seed-fragility reduction via dropout / depth / multi-task head.
- GPU vs CPU wall-clock ratio.
- Calibration of neural pIC50 against the ridge baseline.

---

## 5. Files of record

- Script: `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/retrain_pic50_neural.py`
  (CENSOR_MARGIN_DEFAULT=1.0, new --censored-margin flag, new
  bound_aware_accuracy helper).
- New run report: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pic50_margin_fix/final.md`.
- New run metrics: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pic50_margin_fix/report.json`.
- Predictions: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pic50_margin_fix/test_predictions.parquet`.
- Checkpoint: `/home/hugo/codes/try_triton_on_rocm/molmetal/checkpoints/dmpnn_attn_heLa48h_dark_retrained.pt`.
- Baseline (margin=0.25) report: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_extra1_full/final.md`.
- TODO-18 (append-only): `/home/hugo/codes/try_triton_on_rocm/TODO/pending/18_activity_assay_calibration.md`.
- Paper §4.6 update: `/home/hugo/codes/try_triton_on_rocm/paper/sections/04_evaluation.tex`.
- This verdict: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_pic50_margin_fix_verdict.md`.
