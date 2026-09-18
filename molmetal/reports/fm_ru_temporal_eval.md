# Flow Matching (Ru) — Temporal OOD Evaluation

## Experiment Config

| Parameter | Value |
|-----------|-------|
| Metal | Ru |
| Epochs | 50 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Hidden dim | 128 |
| EGNN layers | 3 |
| Max atoms (OOM guard) | 40 |
| Generation budget | 1000 |
| Seed | 42 |

## Dataset Split (TemporalSplitter, cutoff=2024)

| Split | Size |
|-------|------|
| Train | 2432 |
| Val | 338 |
| Test (OOD) | 290 |

## Training

- **Initial loss**: 232632.141167
- **Final loss**: 7.089718
- **Loss ratio (final/initial)**: 0.0000
- **Wall-clock (training)**: 23.6s
- **GPU used**: True

## Generation Quality

> **Phase-0 limitation**: ``LipmanFlowMatchingAdapter.generate()`` produces
> placeholder atoms (random element types, no bonds, no SMILES) because the
> full ligand-decoder (SMILES/atom-type decoding from the velocity field) is
> Phase 1 work. Generated-molecule metrics below are therefore 0 / empty.
> The **training set reference distribution** (right column) shows what the
> model was optimised on.

| Metric | Generated (Phase-0 placeholder) | Training Reference |
|--------|--------------------------------|--------------------|
| Valid molecules (RDKit-parseable) | 0 / 1000 | 271 |
| QED mean | 0.0000 | 0.5545 |
| QED std | 0.0000 | 0.1684 |
| QED median | 0.0000 | 0.5904 |
| Fraction drug-like (QED ≥ 0.5) | 0.000 | 0.683 |
| MolLogP mean | 0.0000 | -1.2814 |
| MolWt mean | 0.0 | 298.3 |
| TPSA mean | 0.00 | 52.08 |
| Predicted pIC50 mean | 0.0000 | 4.3201 |
| Predicted pIC50 std | 0.0000 | 0.6277 |
| Predicted pIC50 median | 0.0000 | 4.2717 |
| Hit rate @ top-5% (pIC50 ≥ 6) | 0.000 | 0.018 |

## Checkpoint

Saved to: `/home/hugo/codes/try_triton_on_rocm/molmetal/checkpoints/fm_ru_temporal.pt`

## Training Loss Curve

![Loss curve](./fm_ru_train_loss.png)

## Notes

- Temporal split: train = year < 2024, test (OOD) = year ≥ 2024.
- pIC50 regressor = MLP on Morgan FP (radius=2, 2048 bits), trained on training set.
- Training reference = 1000-molecule random sample from the training set.
- Hit rate = fraction with predicted pIC50 ≥ 6.
- Phase-1: replace placeholder generation with a proper SMILES/element decoder.
