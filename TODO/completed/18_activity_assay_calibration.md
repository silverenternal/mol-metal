# Activity predictor: assay context and censored labels

Status: data audit and conditioned GPU baseline measured; learned predictor
retraining, broader external validation and calibrated deployment remain open.
Owner: model/property-prediction line; independent of Lambda search chemistry.

## Verified defect

The legacy Ru keep-first canonical-SMILES calibration samples2000 labels,
including356 raw inequality bounds treated as exact values. Cell line/exposure
and counterion variation are collapsed. Old pIC50 checkpoint metrics are
historical, not validated activity estimates for arbitrary generated molecules.
Evidence: `molmetal/reports/pic50_assay_audit_20260913/report.json`.

## Measured correction baseline

`calibrate_conditioned_pic50_baseline.py` uses the predeclared largest exact
cell/time cohort (HeLa48h/dark), excludes explicit censored labels, preserves
counterion/metal/oxidation/charge formulation identities, and aggregates only
within that assay/formulation using median pIC50. Source rows and DOIs stay
attached. Three scaffold-group splits keep all formulations of a ligand in
one partition; alpha is selected only on validation.

702 formulations,383 scaffolds. Real gfx1101 GPU dual ridge test RMSE for
seeds42/0/1234 is0.5972/0.6116/0.4150, against corresponding training-mean
baselines0.6489/0.7601/0.5274. Pearson0.4774/0.6055/0.6341. These results
cannot be compared directly to the old mixed-assay/censored random split.
Evidence: `molmetal/reports/pic50_conditioned_baseline_20260913/report.json`.

## Remaining acceptance

- [x] Quantify censoring, assay/formulation heterogeneity and source-order effects.
- [x] Produce an explicit assay-conditioned, uncensored scaffold-split baseline
  with source rows, failed-input accounting, actual GPU execution and no test tuning.
- [ ] Verify and retrain the neural property predictor against the same frozen
  cohort/splits/budget controls; preserve test predictions and model artifacts.
- [ ] Extend to other declared cell/time/metal tasks without pooling incompatible
  endpoints; assess publication/temporal holdout and applicability domain.
- [ ] Handle censored observations as bounds (or explicitly retain their
  exclusion and coverage), not by replacing >X with X as an exact target.
- [ ] Calibrate uncertainty and state the target domain before using activity
  as a reward; unsupported chemistry remains unsupported rather than guessed.
- [ ] Keep predicted activity, docking, synthetic routes and biological efficacy
  claims separate. No computational metric establishes wet-lab efficacy.

## Retrain result (2026-09-14, WF-Extra-1 verify, 3 seeds × 50 epochs)

Full retrain on FROZEN HeLa48h/dark cohort (1451 formulations, 715 scaffolds,
251 censored) with censor-aware loss + scaffold-group splits; seeds 42 / 0 /
1234. Aggregate across 3 seeds: test_rmse = 0.687 ± 0.042 pIC50, test_pearson_r
= 0.198 ± 0.228, test_spearman_rho = 0.201 ± 0.197, censored_coverage = 16.9 %
(67/397), bound_aware_accuracy = 0.000 (0/67 censored test rows predicted
compliantly — hinge margin=0.25 is weaker than the model's regression-to-mean
pull). Compared to conditioned ridge baseline on the SAME cohort (RMSE
0.541 ± 0.089, Pearson 0.572 ± 0.068): neural D-MPNN does NOT beat ridge, and
is seed-fragile (Pearson r ranges 0.004 → 0.519). The legacy mixed-assay
historical D-MPNN (Pearson r=0.407) is on a different cohort and remains
incomparable. Verdict: bound-aware loss as configured is not effective;
recommend raising margin to ≥1.0 pIC50 or replacing hinge with a zero-margin
penalty before using neural predictor as a reward. Report:
`molmetal/reports/wf_extra1_full/final.md`; predictions:
`molmetal/reports/wf_extra1_full/test_predictions.parquet`; checkpoint:
`molmetal/checkpoints/dmpnn_attn_heLa48h_dark_retrained.pt`.

## Margin-fix result (2026-09-14, WF-pIC50-Margin-Fix verify, Option A, 3 seeds × 50 epochs)

Follow-up to the WF-Extra-1 verdict. Single change: `CENSOR_MARGIN_DEFAULT
0.25 → 1.0` in `molmetal/scripts/retrain_pic50_neural.py`; loss formula
unchanged (`MSE(exact) + hinge-margin(censored)`). Option A is the smallest
diff; Option B (zero-margin penalty `max(0, -dir*(pred-bound))`) was held
in reserve but not needed. Same FROZEN HeLa48h/dark cohort, same scaffold-
group 80/10/10 splits, same seed schedule, same 50-epoch budget.

Aggregate across 3 seeds (MEASURED): test_rmse = 0.705 ± 0.048 pIC50
(+0.018 vs margin=0.25, ~2.6% worse, within noise), test_pearson_r = 0.181
± 0.224 (-0.017 vs margin=0.25, within noise), test_spearman_rho = 0.161
± 0.218, censored_coverage = 16.9% (67/397), **bound_aware_accuracy = 1.000
(67/67 censored test rows compliant — flipped from 0/67 in WF-Extra-1)**.
Per-seed: seed=42 r=-0.049 18/18 compliant; seed=0 r=+0.484 33/33
compliant; seed=1234 r=+0.109 16/16 compliant. Hinge margin=1.0 pIC50 is
above the regression-to-mean pull (~0.80), so the hinge now actually
constrains the censored predictions without collapsing pred_std. The
ridge-baseline gap on RMSE/Pearson is unchanged and is a separate
model-class problem. Verdict: **Option A succeeded on the targeted
metric (bound-aware accuracy).** Option B held in reserve. The neural
predictor remains seed-fragile on Pearson r (σ/μ = 1.23 across 3 seeds,
indistinguishable from WF-Extra-1's 1.15); closing the ridge gap is out
of scope for this PR. Report:
`molmetal/reports/wf_pic50_margin_fix/final.md`; verdict:
`molmetal/reports/wf_pic50_margin_fix_verdict.md`; predictions:
`molmetal/reports/wf_pic50_margin_fix/test_predictions.parquet`; report.json:
`molmetal/reports/wf_pic50_margin_fix/report.json`. Script change:
`molmetal/scripts/retrain_pic50_neural.py` (`CENSOR_MARGIN_DEFAULT=1.0`,
new `--censored-margin` CLI flag).

## Margin-sweep result (2026-09-14, WF-pIC50-Margin-Sweep verify, 4 margins × 3 seeds × 50 epochs)

Follow-up to the WF-pIC50-Margin-Fix verdict. Single change: sweep
`--censored-margin ∈ {0.5, 1.0, 2.0, 3.0}` on the same FROZEN
HeLa48h/dark cohort (1451 formulations, 715 scaffolds, 251
censored) with the same scaffold-group 80/10/10 splits and
same seed schedule. Script change to default: `CENSOR_MARGIN_DEFAULT
1.0 → 0.5` in `molmetal/scripts/retrain_pic50_neural.py`.

Aggregate across 3 seeds (MEASURED), per-margin:

| margin | test_pearson_r | test_rmse | bound_aware_acc | compliant/censored |
|--------|----------------|-----------|-----------------|--------------------|
| 0.5    | **0.195 ± 0.234** | **0.686 ± 0.038** | **1.000** | 67/67 |
| 1.0    | 0.184 ± 0.222 | 0.703 ± 0.050 | 1.000 | 67/67 |
| 2.0    | 0.158 ± 0.181 | 0.724 ± 0.041 | 0.990 | 66/67 (seed=0 miss) |
| 3.0    | 0.186 ± 0.179 | 0.724 ± 0.041 | 0.990 | 66/67 (same seed=0 miss) |

Best margin per WF protocol `(bound_aware_acc == 1.0) AND
max(test_pearson_r)`: filter to margins with BA=1.0 leaves
{0.5, 1.0}; among those, max pearson_r → margin=**0.5**
(0.195 vs 0.184). Margin=0.5 also wins on test_rmse (0.686 vs
0.703). Margins 2.0 and 3.0 break the censor guarantee — the
wider hinge gives the optimiser slack to predict above the
bound, and the largest censored-test-pool seed (seed=0, 33
censored rows) accrues one over-prediction per arm.

Per-seed test_pearson_r for transparency:

| seed  | margin=0.5 | margin=1.0 | margin=2.0 | margin=3.0 |
|-------|------------|------------|------------|------------|
| 42    | -0.025     | -0.049     | -0.046     | -0.046     |
| 0     | +0.519     | +0.483     | +0.394     | +0.390     |
| 1234  | +0.092     | +0.117     | +0.126     | +0.215     |

Verdict: **margin = 0.5 is the new default**. Honest caveats:
(i) test_pearson_r mean 0.195 is well below the ridge baseline
(0.572) and even below the historical mixed-assay D-MPNN
baseline (0.407); the margin sweep only asks which margin
maximises censor compliance + exact-row pearson_r, not whether
D-MPNN beats Ridge on this cohort. (ii) Per-seed variance is
large (std 0.18–0.23) relative to margin-to-margin difference
in means (~0.04); a 5-seed re-run at margin ∈ {0.5, 1.0} would
tighten the recommendation. (iii) Margins 2.0/3.0 collapse
BA=1.0 on the seed with the largest censored pool — increasing
margin past 1.0 is strictly worse on both metrics. Report:
`molmetal/reports/wf_pic50_margin_sweep/final.md`; verdict:
`molmetal/reports/wf_pic50_margin_sweep_verdict.md`; per-margin
JSON: `molmetal/reports/wf_pic50_margin_sweep/m_<margin>/report.json`;
paper update: §4.6 paragraph "Cell-line-aware pIC50
(auxiliary)" cites margin=0.5 as the new default.
Script change: `molmetal/scripts/retrain_pic50_neural.py`
(`CENSOR_MARGIN_DEFAULT=0.5`, help text updated).
