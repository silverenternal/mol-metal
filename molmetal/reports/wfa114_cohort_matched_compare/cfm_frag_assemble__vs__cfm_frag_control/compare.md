# CFM cohort-matched comparison — VERDICT

- **A**: `cfm_frag_assemble` (molmetal/reports/cfm_frag_assemble/per_epoch_metrics.json)
- **B**: `cfm_frag_control` (molmetal/reports/cfm_frag_control/per_epoch_metrics.json)
- **cohort**: n_train=256, n_val=192
- **reading**: `epoch` at epoch index 29 (compared budget = 30 epochs)
- **formula**: `(gen_validity_val / gen_validity_n) * gen_single_fragment_ratio`

## Axes

| axis | status | detail |
|---|---|---|
| `cohort` | PASS | both runs trained on n_train=256, n_val=192 |
| `epoch_budget` | PASS | both runs recorded a row at epoch 29 (compared budget = 30 epochs) |
| `not_measured` | PASS | the contributing row of each run carries a generation measurement (gen_validity_n > 0) |
| `reading` | PASS | the fixed-epoch reading compares one index for both arms, so the two series need not cover the same epochs |
| `history` | PASS | both runs recorded a contiguous epoch history from 0 and their contributing rows carry a finite train_loss |
| `provenance` | PASS | both arms read the GENERATION pool, their pool construction matches, and no on-disk pool contradicts the per-epoch metric |

## Result

- e2e(A) = **0.015625** (MEASURED, epoch 29)
- e2e(B) = **0.03125** (MEASURED, epoch 29)
- delta (B - A) = **0.015625** (MEASURED)
- source: `molmetal/reports/cfm_frag_control/per_epoch_metrics.json -> per_epoch_metrics[epoch==29] minus molmetal/reports/cfm_frag_assemble/per_epoch_metrics.json -> per_epoch_metrics[epoch==29]`

## Caveat

- one run is one run: no error bar, finite generation pool (gen_eval_n=32) => sampling noise of unknown sign; 2 seeds is still n=2
