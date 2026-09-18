# Pocket-Conditioned Flow Matching — CrossDocked2020 Evaluation

## Experiment Config

| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch size | 8 |
| Learning rate | 0.0001 |
| Hidden dim | 128 |
| EGNN layers | 3 |
| Max atoms per ligand (OOM guard) | 40 |
| Max atoms per pocket (OOM guard) | 200 |
| Generation budget | 100 |
| Seed | 42 |
| Dataset | CrossDocked2020 |
| Target filter | MMP2/MMP9 |

## Dataset

- **Pairs loaded**: 200
- **Split**: train

## Training

- **Initial loss**: 381.532628
- **Final loss**: 193.019047
- **Loss ratio (final/initial)**: 0.5059
- **Wall-clock (training)**: 36.1s
- **GPU used**: True

## Generation Quality

> **Phase-0 limitation**: ``generate()`` produces placeholder atoms (random
> element types, no bonds, no SMILES). Full ligand decoding is Phase 1 work.
> The **training reference** column shows the distribution of the actual
> training ligands (ground truth).

| Metric | Generated (Phase-0 placeholder) | Training Reference |
|--------|--------------------------------|--------------------|
| Valid molecules | 0 / 100 | 200 |
| QED mean | 0.0000 | 0.5490 |
| QED std | 0.0000 | 0.1665 |
| Fraction drug-like (QED ≥ 0.5) | 0.000 | 0.630 |
| MolLogP mean | 0.0000 | 1.8831 |
| MolWt mean | 0.0 | 367.2 |
| TPSA mean | 0.00 | 98.85 |
| Predicted pIC50 mean | 0.0000 | 3.8336 |
| Predicted pIC50 std | 0.0000 | 1.0797 |
| Hit rate @ pIC50 ≥ 6 | 0.000 | 0.030 |

## Checkpoint

Saved to: `/home/hugo/codes/try_triton_on_rocm/molmetal/checkpoints/fm_pocket.pt`

## Training Loss Curve

![Loss curve](./fm_pocket_train_loss.png)

## Notes

- Pocket-conditioned velocity field: shared EGNN for ligand + pocket,
  cross-context via pocket → ligand global pooling.
- pIC50 regressor = MLP on Morgan FP (radius=2, 2048 bits), trained on
  Ru subset of MetalCytoToxDB as a proxy for drug-likeness.
- Phase-1: replace placeholder generation with SMILES/element decoder.
