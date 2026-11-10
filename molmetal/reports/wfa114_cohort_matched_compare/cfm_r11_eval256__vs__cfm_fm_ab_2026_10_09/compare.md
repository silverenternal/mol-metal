# CFM cohort-matched comparison — REFUSED

- **A**: `cfm_r11_eval256` (molmetal/reports/cfm_r11_eval256/per_epoch_metrics.json)
- **B**: `cfm_fm_ab_2026_10_09` (molmetal/reports/cfm_fm_ab_2026_10_09/per_epoch_metrics.json)
- **cohort**: n_train=26715, n_val=6714
- **reading**: `epoch` at epoch index 19 (compared budget = 20 epochs)
- **formula**: `(gen_validity_val / gen_validity_n) * gen_single_fragment_ratio`

## Axes

| axis | status | detail |
|---|---|---|
| `cohort` | FAIL | the two runs are not the same cohort: n_train differs (A=26715, B=256); n_val differs (A=6714, B=192) |
| `epoch_budget` | PASS | both runs recorded a row at epoch 19 (compared budget = 20 epochs) |
| `not_measured` | PASS | the contributing row of each run carries a generation measurement (gen_validity_n > 0) |
| `reading` | PASS | the fixed-epoch reading compares one index for both arms, so the two series need not cover the same epochs |
| `history` | FAIL | cfm_r11_eval256's recorded epochs start at 19, not 0: 19 epoch(s) are invisible, so its epoch index is not the same training amount as the other run's; cfm_r11_eval256's contributing row carries train_loss = nan |
| `provenance` | FAIL | the two runs' pools were not constructed the same way: gen_solver ('midpoint/50' vs 'rk4/100'), gen_eval_n (256 vs 16), gen_eval_seeds ('0,1,2,3' vs '0,1,2') |

## Refusal

- unmatched axes: cohort, history, provenance
  - `cohort`: the two runs are not the same cohort: n_train differs (A=26715, B=256); n_val differs (A=6714, B=192)
  - `history`: cfm_r11_eval256's recorded epochs start at 19, not 0: 19 epoch(s) are invisible, so its epoch index is not the same training amount as the other run's; cfm_r11_eval256's contributing row carries train_loss = nan
  - `provenance`: the two runs' pools were not constructed the same way: gen_solver ('midpoint/50' vs 'rk4/100'), gen_eval_n (256 vs 16), gen_eval_seeds ('0,1,2,3' vs '0,1,2')

## Nearest valid pair on disk

- `cfm_blk_c5_repair_2026_10_09` vs `cfm_fm_ab_2026_10_09` at epoch index 29
- e2e(A) = 0.020833333333333332, e2e(B) = 0.041666666666666664
- the nearest valid pair on disk in cfm_fm_ab_2026_10_09's cohort (n_train=256, n_val=192)

## Caveat

- one run is one run: no error bar, finite generation pool (gen_eval_n=256/16) => sampling noise of unknown sign; 2 seeds is still n=2
