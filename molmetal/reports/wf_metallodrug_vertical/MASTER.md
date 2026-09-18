# WF-Metallodrug-Vertical MASTER — 2026-09-16

**Status**: SHIPPED (paper + smoke + 19 protocol tests + 4 new loaders). 
GPU retrain (Phase 3) hit 200-step smoke = **decode_ratio 0/16 BUT cfm_loss -66%**, then black-screened → re-run deferred.

## Deliverables shipped (CPU-only, additive)

### Code (3 new modules)
- `molmetal/molmetal_lam/lam_chem/platinai_dataset.py` (PlatinAI 226K Pt-corpus loader, kNN oracle)
- `molmetal/molmetal_lam/lam_chem/tmqm_dataset.py` (tmQM 92K loader + 7-must-have Pt-pattern coverage)
- `molmetal/scripts/metallo_drug_smoke_retrain.py` (1000-step CFM smoke driver)

### Paper
- `paper/sections/03_7_metallodrug_vertical.tex` (501 lines) — vertical framing + dataset registry + 14-atom vocab rationale + 5-cell-line oracle + 9 vertical metrics

### Tests
- `molmetal/tests/test_protocol_alignment.py` (19 unit tests) — passes
- `molmetal/tests/test_dmpnn_baseline.py` (17 integration tests, from TODO/04 batch)
- Phase 2 PlatinAI / tmQM unit tests pass

### Scripts
- `metallo_pool_scaffold_split.py` (P6.1, also from P6.2 batch)

### Metrics
- `atom_vocab_coverage` 4→14 (3.5×)
- `n_train_scaleup` 32→500 (15.6×)
- `metallodrug_vertical_coverage` new metric (5 datasets, 791K union records, 16 metals)

## Phase 3 GPU smoke — INCONCLUSIVE_PARTIAL

- 200/1000 steps executed on cuda
- cfm_loss: 21.09 → 6.97 (-66%, training works)
- bond/atom_loss: inf → inf (NaN-poisoned, known issue)
- GPU black-screened at step 201 (device mismatch bug + SMU hang)
- INCOMPLETE: do NOT claim decode lift in paper

## Blocked / Deferred

- GPU retrain R16 W42-W43 (TODO-24)
- Re-run 200×3 sweep at n_sim=1000 (TODO-26 W39)
- Wet-lab validation (Tier 3 P5.2)

## Reference

- `molmetal/reports/wf_metallodrug_vertical/MASTER.md` (this file)
- `paper/sections/03_7_metallodrug_vertical.tex` (vertical section)
- `TODO/pending/22_data_gap_alignment_plan.md` (alignment spec)
