# CFM cohort-matched comparison — VERDICT

- **A**: `cfm_fullcorpus_seed1` (molmetal/reports/cfm_fullcorpus_seed1/per_epoch_metrics.json)
- **B**: `cfm_fullcorpus_seed2` (molmetal/reports/cfm_fullcorpus_seed2/per_epoch_metrics.json)
- **cohort**: n_train=26715, n_val=192
- **reading**: `epoch` at epoch index 14 (compared budget = 15 epochs)
- **formula**: `(gen_validity_val / gen_validity_n) * gen_single_fragment_ratio`

## Axes

| axis | status | detail |
|---|---|---|
| `cohort` | PASS | both runs trained on n_train=26715, n_val=192 |
| `epoch_budget` | PASS | both runs recorded a row at epoch 14 (compared budget = 15 epochs) |
| `not_measured` | PASS | the contributing row of each run carries a generation measurement (gen_validity_n > 0) |
| `reading` | PASS | the fixed-epoch reading compares one index for both arms, so the two series need not cover the same epochs |
| `history` | PASS | both runs recorded a contiguous epoch history from 0 and their contributing rows carry a finite train_loss |
| `provenance` | PASS | both arms read the GENERATION pool, their pool construction matches, and no on-disk pool contradicts the per-epoch metric |

## Result

- e2e(A) = **0.38020833333333337** (MEASURED, epoch 14)
- e2e(B) = **0.4518229166666667** (MEASURED, epoch 14)
- delta (B - A) = **0.07161458333333331** (MEASURED)
- source: `molmetal/reports/cfm_fullcorpus_seed2/per_epoch_metrics.json -> per_epoch_metrics[epoch==14] minus molmetal/reports/cfm_fullcorpus_seed1/per_epoch_metrics.json -> per_epoch_metrics[epoch==14]`

## Bar

- bar = **0.5260416666666666** (MEASURED) — molmetal/reports/cfm_fullcorpus_seed1/per_epoch_metrics.json -> per_epoch_metrics[epoch==13]
- B meets the bar: **False**; A meets the bar: **False**

## Caveat

- one run is one run: no error bar, finite generation pool (gen_eval_n=256) => sampling noise of unknown sign; 2 seeds is still n=2
