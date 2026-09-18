# Fine-tune tmQM encoder + wire into EGNNVelocityField

**Status:** pending
**Priority:** high
**Effort:** 1d
**Owner:** (unset)
**Depends on:** none
**Blockers:** tmQM pretrained checkpoint available at molmetal/checkpoints/tmqm_dmpnn.pt  (verify exists; see TODO/completed/05)
**Created:** 2026-09-12

## Goal
Take the tmQM-pretrained D-MPNN encoder (CN MAE=0.132, R²=0.986; see
TODO/completed/05) and (a) fine-tune on MetalCytoToxDB Ru subset, then
(b) wire its state-dict into `EGNNVelocityField` as the encoder initialiser.
The current `EGNNVelocityField` uses random-init weights; the metal-coordination
prior is theoretically motivated but not delivered to the velocity field.

## File(s) to edit
- /home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/egnn_velocity.py  (load + apply init)
- new: fine-tune script under molmetal/scripts/  (uses molmetal/data/)

## Success criterion
`egnn_velocity.py` loads `checkpoints/tmqm_dmpnn.pt` state-dict into
`encoder.linear` (or equivalent); fine-tune epoch on MetalCytoToxDB Ru
subset improves test AUC by ≥+0.01 over random-init EGNNVelocityField;
`f2_tmqm_pretraining.md` marked PHASE-2 DELIVERED.

## Related reports
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/f2_tmqm_pretraining.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/f2_tmqm_stats.json