# CFM cohort-matched comparison — VERDICT

- **A**: `cfm_r9_fix_v2` (molmetal/reports/cfm_r9_fix_v2/per_epoch_metrics.json)
- **B**: `cfm_r8_cosine_40ep` (molmetal/reports/cfm_r8_cosine_40ep/per_epoch_metrics.json)
- **cohort**: n_train=26715, n_val=6714
- **reading**: `epoch` at epoch index 8 (compared budget = 9 epochs)
- **formula**: `(gen_validity_val / gen_validity_n) * gen_single_fragment_ratio`

## Axes

| axis | status | detail |
|---|---|---|
| `cohort` | PASS | both runs trained on n_train=26715, n_val=6714 |
| `epoch_budget` | PASS | both runs recorded a row at epoch 8 (compared budget = 9 epochs) |
| `not_measured` | PASS | the contributing row of each run carries a generation measurement (gen_validity_n > 0) |
| `reading` | PASS | the fixed-epoch reading compares one index for both arms, so the two series need not cover the same epochs |
| `history` | PASS | both runs recorded a contiguous epoch history from 0 and their contributing rows carry a finite train_loss |
| `provenance` | PASS | both arms read the GENERATION pool, their pool construction matches, and no on-disk pool contradicts the per-epoch metric |

## Result

- e2e(A) = **0.24479166666666669** (MEASURED, epoch 8)
- e2e(B) = **0.057291666666666664** (MEASURED, epoch 8)
- delta (B - A) = **-0.18750000000000003** (MEASURED)
- source: `molmetal/reports/cfm_r8_cosine_40ep/per_epoch_metrics.json -> per_epoch_metrics[epoch==8] minus molmetal/reports/cfm_r9_fix_v2/per_epoch_metrics.json -> per_epoch_metrics[epoch==8]`

## Bar

- bar = **0.41145833333333337** (MEASURED) — molmetal/reports/cfm_r9_fix_v2/per_epoch_metrics.json -> per_epoch_metrics[epoch==10]
- B meets the bar: **False**; A meets the bar: **False**

## Caveat

- one run is one run: no error bar, finite generation pool (gen_eval_n=64) => sampling noise of unknown sign; 2 seeds is still n=2
