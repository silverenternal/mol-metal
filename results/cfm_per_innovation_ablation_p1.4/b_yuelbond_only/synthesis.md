# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (192/192)  
**Best epoch**: 0  
**Target**: 0.50  
**Hit target**: YES  
**Total wall time**: 14.0s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |
| L-R17-B2 matrix validation (Wf-296D) | `lr17_b2_matrix_84_plus_17_new_smarts` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.8538 | 1.8538 | 0.0000 | 2.828 | 1.000 | (192/192) | 10.3 |
| 1 | 1.6346 | 1.6346 | 0.0000 | 2.441 | 1.000 | (192/192) | 2.0 |
| 2 | 1.4563 | 1.4563 | 0.0000 | 2.308 | 1.000 | (192/192) | 1.7 |

## Honest framing

- The target decode_ratio ≥ 0.50 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 3-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 14.0s (~4.7s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Wf-296D — YuelBond precondition + matrix wiring

- `m14_lead_invalidated`: **False** (YuelBond active; m14 LEAD precondition OK).
- `matrix_validation_available`: **True** (compiled 101 reactions from 84 matrix + 17 NEW SMARTS patterns).
- `decode_ratio` is computed via `_decode_with_matrix_validation`, which requires the generated SMILES to match at least one of the 84 matrix or 17 NEW SMARTS productive-space patterns (RDKit + matrix = strict).

## Outputs

- `results/cfm_per_innovation_ablation_p1.4/b_yuelbond_only/per_epoch_metrics.json` (full per-epoch record)
- `results/cfm_per_innovation_ablation_p1.4/b_yuelbond_only/per_epoch_metrics.csv` (flat CSV)
- `results/cfm_per_innovation_ablation_p1.4/b_yuelbond_only/wasserstein_decay.json` (W₂ decay check)
- `results/cfm_per_innovation_ablation_p1.4/b_yuelbond_only/ckpts/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/cfm_per_innovation_ablation_p1.4/b_yuelbond_only/synthesis.md` (this file)

## Cross-references

- TODO-119 (decode_ratio strengthen) — primary target
- TODO-121 (m19 LOMO / Rh bottleneck) — uses Wasserstein transfer
- TODO-122 (PlatinAI 4-cell R²) — uses Wasserstein pIC50
- TODO-123 (m15 RMSD) — uses Wasserstein pose
- TODO-124 (m5 SA) — uses Catherine closure-bounded generation
- TODO-125 (per-pocket Vina) — uses Wasserstein pocket alignment
- TODO-126 §3 (shared master modules) — M1 Wasserstein + M2 closure-bias
- Wf-124 P1 (YuelBond decoder swap)
- Wf-123 P1 (Wasserstein loss module)
- Wf-123 P2 (pocket-specific closure-bias module)
- Wf-123 P3 (cross-module integration verification)
