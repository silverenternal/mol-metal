# tmQM DMPNN encoder pretraining (CN MAE=0.132)

**Status:** completed
**Completed:** 2026-09-12
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/f2_tmqm_pretraining.md
**Owner:** (unset)

## What was delivered
Pretrained a D-MPNN (directed message-passing neural network) encoder on
21,617 Pt/Ru/Ir complexes from the tmQM transition-metal dataset (cloned at
`molmetal/references/tmQM/`). The encoder has two heads:
(1) coordination-number (CN) regression, (2) Wiberg bond-order (BO)
regression. The pretrained checkpoint is intended as the default init for the
EGNN velocity field used by the Lipman FM adapter (wire-up is pending; see
TODO/pending/08_tmqm_egnn_wireup.md).

## Hard numbers (f2_tmqm_pretraining.md + f2_tmqm_stats.json)
- Dataset: **21,617** Pt/Ru/Ir tmQM complexes (NOT 86k — that's the full tmQM;
  this subset is restricted to the three metals that drive the lambda_clickchem reward)
- Pretraining: 10 epochs on ROCm 7.2, ~63 s wall-clock total
- **CN regression MAE = 0.132** (R² = 0.986)
- **Wiberg BO regression MAE = 0.176** (R² = 0.924)
- 97% exact-match CN on 2,161 held-out complexes
- Encoder + heads checkpoint size: ~14 MB

## Lessons learned
- The 21,617 number is correct (Pt/Ru/Ir subset); the full tmQM corpus is ~86k
  complexes but we restrict to the three metals that drive the reward signal.
- Wire-up into `molmetal/adapters/flow_matching_lipman/egnn_velocity.py` is
  NOT yet done — see TODO/pending/08_tmqm_egnn_wireup.md. The current
  EGNNVelocityField still uses random-init weights.

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/f2_tmqm_pretraining.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/f2_tmqm_stats.json
- /home/hugo/codes/try_triton_on_rocm/molmetal/references/tmQM/  (cloned)