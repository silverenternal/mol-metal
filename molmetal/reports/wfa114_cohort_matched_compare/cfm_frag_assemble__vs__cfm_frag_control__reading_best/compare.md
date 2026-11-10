# CFM cohort-matched comparison — VERDICT

- **A**: `cfm_frag_assemble` (molmetal/reports/cfm_frag_assemble/per_epoch_metrics.json)
- **B**: `cfm_frag_control` (molmetal/reports/cfm_frag_control/per_epoch_metrics.json)
- **cohort**: n_train=256, n_val=192
- **reading**: `best` at epoch index 29 (compared budget = 30 epochs)
- **formula**: `(gen_validity_val / gen_validity_n) * gen_single_fragment_ratio`

## Axes

| axis | status | detail |
|---|---|---|
| `cohort` | PASS | both runs trained on n_train=256, n_val=192 |
| `epoch_budget` | PASS | both runs recorded a row at epoch 29 (compared budget = 30 epochs) |
| `not_measured` | PASS | the contributing row of each run carries a generation measurement (gen_validity_n > 0) |
| `reading` | PASS | both runs cover the same recorded epochs, so their best epochs are comparable under the same reading |
| `history` | PASS | both runs recorded a contiguous epoch history from 0 and their contributing rows carry a finite train_loss |
| `provenance` | PASS | both arms read the GENERATION pool, their pool construction matches, and no on-disk pool contradicts the per-epoch metric |

## Result

- e2e(A) = **0.1953125** (MEASURED, epoch 16)
- e2e(B) = **0.28125** (MEASURED, epoch 24)
- delta (B - A) = **0.0859375** (MEASURED)
- source: `molmetal/reports/cfm_frag_control/per_epoch_metrics.json -> per_epoch_metrics[epoch==24] minus molmetal/reports/cfm_frag_assemble/per_epoch_metrics.json -> per_epoch_metrics[epoch==16]`
- note: the best reading is a MAX over a noisy series (the census labels it 'optimistically biased'); it is reported as a secondary reading and never as a bar

## Caveat

- one run is one run: no error bar, finite generation pool (gen_eval_n=32) => sampling noise of unknown sign; 2 seeds is still n=2
