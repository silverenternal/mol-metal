# T3 MetalHybrid V3 — Ru Temporal Split

## TL;DR
- Test ROC-AUC on Ru temporal split (n=290, post-2024) = **0.4708** (PR-AUC 0.231, pIC50 MAE 0.645).
- Beats raw D-MPNN baseline (0.5135) only marginally; both regress vs val (0.6733/0.7019) under temporal drift.
- All 4 existing `test_metal_hybrid_v3.py` unit tests pass; no retraining performed.

## Model architecture
`MetalHybridV3Model` (968k params) =
1. **D-MPNN encoder** (DirectedMPNN, hidden=128, 3 layers) — initialised from tmQM pretraining checkpoint.
2. **EGNN coordinate-refinement stream** (3 layers, hidden=128) — produces 3-D-aware per-atom features; gradient-detached coord-MLP adds bounded Δx (max 0.3 Å) via `_CoordRefineHead`.
3. **Cross-attention fusion** (`fusion_gate`) between 2-D (D-MPNN) and 3-D (EGNN) pooled graphs.
4. **Multi-task head**: shared MLP → (a) pIC50 regression (L1), (b) 2-class active/inactive softmax (BCE). Loss = α·BCE + (1−α)·L1 with detached coord-loss term γ=0.1.

## Test AUC (Ru temporal)
| Model | n_test | ROC-AUC | PR-AUC |
|---|---|---|---|
| **MetalHybrid V3 (this ckpt)** | 290 | **0.4708** | 0.2307 |
| D-MPNN baseline | 290 | 0.5135 | 0.2612 |
| Hybrid val (pre-cutoff) | 338 | 0.6733 | 0.3701 |
| D-MPNN val (pre-cutoff) | 338 | 0.7019 | 0.4771 |

Temporal split (cutoff=2024) shows classic post-cutoff regression: both models lose ~0.2 AUC. Hybrid does not improve over D-MPNN on OOD; pIC50 MAE 0.645 is reasonable but not SOTA.

## Known limitations
- **Single seed**: checkpoint was best epoch (11/12) of one run; seed averaging would tighten CI.
- **Coord signal weak**: EGNN layers detach the coord gradient; the model effectively uses 3-D only as pooling bias. Likely why hybrid ≈ D-MPNN.
- **No scaffold/ligand_dedup comparison reported here** — temporal is hardest OOD; results are worse than the 0.71 val figures.
- **Test pos_rate 0.248** with only 290 rows → AUC variance ~±0.04.
- **Negative result**: cross-attention + EGNN coord refinement does not recover temporal drift; need either temporal augmentation, meta-learning, or richer 3-D pretraining.

## Files
- Code: `molmetal/models/metal_hybrid_v3.py`, `molmetal/scripts/train_metal_hybrid.py`
- Checkpoint: `molmetal/checkpoints/metal_hybrid_v3_crossattn_ru_temporal.pt` (3.9 MB)
- Baseline: `molmetal/reports/baseline_ru_dmpnn_temporal.json`
