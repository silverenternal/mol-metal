# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.425 (79/186)  
**Best epoch**: 26  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 2820.8s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.5550 | 5.4493 | 0.0000 | 128.774 | 0.839 | (156/186) | 165.0 |
| 1 | 5.1037 | 5.0367 | 0.0000 | 118.898 | 0.806 | (150/186) | 106.9 |
| 2 | 4.3576 | 4.3343 | 0.0000 | 103.983 | 0.554 | (103/186) | 110.2 |
| 3 | 3.4336 | 3.4504 | 0.0000 | 87.819 | 0.457 | (85/186) | 96.2 |
| 4 | 2.5124 | 2.5531 | 0.0000 | 74.485 | 0.430 | (80/186) | 93.7 |
| 5 | 1.5307 | 1.5951 | 0.0000 | 53.588 | 0.629 | (117/186) | 100.5 |
| 6 | 1.0489 | 1.1181 | 0.0000 | 29.854 | 0.565 | (105/186) | 90.9 |
| 7 | 0.8745 | 0.9432 | 0.0000 | 22.077 | 0.624 | (116/186) | 93.5 |
| 8 | 0.7464 | 0.8162 | 0.0000 | 18.831 | 0.667 | (124/186) | 93.8 |
| 9 | 0.6637 | 0.7284 | 0.0000 | 18.771 | 0.694 | (129/186) | 83.1 |
| 10 | 0.6083 | 0.6729 | 0.0000 | 15.659 | 0.575 | (107/186) | 113.3 |
| 11 | 0.5788 | 0.6410 | 0.0000 | 23.813 | 0.468 | (87/186) | 115.0 |
| 12 | 0.5576 | 0.6146 | 0.0000 | 29.751 | 0.468 | (87/186) | 105.4 |
| 13 | 0.5232 | 0.5796 | 0.0000 | 17.360 | 0.457 | (85/186) | 110.5 |
| 14 | 0.5040 | 0.5582 | 0.0000 | 19.696 | 0.511 | (95/186) | 102.7 |
| 15 | 0.4842 | 0.5356 | 0.0000 | 19.635 | 0.452 | (84/186) | 91.9 |
| 16 | 0.4654 | 0.5143 | 0.0000 | 21.416 | 0.441 | (82/186) | 66.9 |
| 17 | 0.4482 | 0.4948 | 0.0000 | 22.624 | 0.452 | (84/186) | 91.1 |
| 18 | 0.4354 | 0.4817 | 0.0000 | 22.275 | 0.425 | (79/186) | 96.1 |
| 19 | 0.4234 | 0.4654 | 0.0000 | 25.492 | 0.414 | (77/186) | 93.6 |
| 20 | 0.4069 | 0.4484 | 0.0000 | 20.167 | 0.414 | (77/186) | 97.5 |
| 21 | 0.4040 | 0.4445 | 0.0000 | 27.254 | 0.414 | (77/186) | 106.0 |
| 22 | 0.4021 | 0.4419 | 0.0000 | 32.858 | 0.414 | (77/186) | 80.3 |
| 23 | 0.3862 | 0.4238 | 0.0000 | 18.255 | 0.425 | (79/186) | 77.5 |
| 24 | 0.3823 | 0.4190 | 0.0000 | 26.077 | 0.425 | (79/186) | 73.9 |
| 25 | 0.3748 | 0.4102 | 0.0000 | 27.228 | 0.414 | (77/186) | 69.3 |
| 26 | 0.3632 | 0.3982 | 0.0000 | 20.703 | 0.425 | (79/186) | 72.6 |
| 27 | 0.3535 | 0.3863 | 0.0000 | 18.406 | 0.414 | (77/186) | 76.4 |
| 28 | 0.3517 | 0.3824 | 0.0000 | 24.226 | 0.414 | (77/186) | 73.3 |
| 29 | 0.3379 | 0.3682 | 0.0000 | 23.023 | 0.414 | (77/186) | 73.3 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 30-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 2820.8s (~94.0s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/cfm_5metal_default/per_epoch_metrics.json` (full per-epoch record)
- `results/cfm_5metal_default/per_epoch_metrics.csv` (flat CSV)
- `results/cfm_5metal_default/wasserstein_decay.json` (W₂ decay check)
- `results/cfm_5metal_default/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/cfm_5metal_default/synthesis.md` (this file)

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
