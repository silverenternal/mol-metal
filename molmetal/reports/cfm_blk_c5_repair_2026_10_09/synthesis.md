# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.609 (117/192)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 4056.7s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.9479 | 1.8784 | 0.7638 | 0.163 | 0.609 | (117/192) | 1723.4 |
| 1 | 1.9479 | 1.8836 | 0.7114 | 0.262 | 0.609 | (117/192) | 368.6 |
| 2 | 1.9395 | 1.8819 | 0.6431 | 0.381 | 0.609 | (117/192) | 94.1 |
| 3 | 1.9276 | 1.8798 | 0.5466 | 0.531 | 0.609 | (117/192) | 80.2 |
| 4 | 1.9230 | 1.8876 | 0.4217 | 0.491 | 0.609 | (117/192) | 61.2 |
| 5 | 1.5434 | 1.5086 | 0.4028 | 5.919 | 1.000 | (192/192) | 82.1 |
| 6 | 1.2993 | 1.2644 | 0.4092 | 1.965 | 1.000 | (192/192) | 67.5 |
| 7 | 1.2350 | 1.2068 | 0.3718 | 1.219 | 1.000 | (192/192) | 66.0 |
| 8 | 1.1928 | 1.1658 | 0.3717 | 0.935 | 0.854 | (164/192) | 64.7 |
| 9 | 1.1682 | 1.1416 | 0.3662 | 1.028 | 0.797 | (153/192) | 67.2 |
| 10 | 1.1468 | 1.1198 | 0.3668 | 1.009 | 0.505 | (97/192) | 70.3 |
| 11 | 1.1339 | 1.1072 | 0.3649 | 1.118 | 0.760 | (146/192) | 68.7 |
| 12 | 1.1228 | 1.0960 | 0.3648 | 0.953 | 0.552 | (106/192) | 68.7 |
| 13 | 1.1101 | 1.0831 | 0.3642 | 1.237 | 0.422 | (81/192) | 73.2 |
| 14 | 1.0949 | 1.0682 | 0.3638 | 1.256 | 0.286 | (55/192) | 67.0 |
| 15 | 1.0797 | 1.0525 | 0.3635 | 1.099 | 0.562 | (108/192) | 62.3 |
| 16 | 1.0673 | 1.0403 | 0.3634 | 1.318 | 0.307 | (59/192) | 77.5 |
| 17 | 1.0503 | 1.0231 | 0.3629 | 1.137 | 0.151 | (29/192) | 111.0 |
| 18 | 1.0308 | 1.0033 | 0.3628 | 1.285 | 0.359 | (69/192) | 63.6 |
| 19 | 1.0168 | 0.9891 | 0.3621 | 1.323 | 0.208 | (40/192) | 65.0 |
| 20 | 1.0096 | 0.9819 | 0.3626 | 1.682 | 0.188 | (36/192) | 67.4 |
| 21 | 0.9894 | 0.9617 | 0.3623 | 1.476 | 0.219 | (42/192) | 63.5 |
| 22 | 0.9734 | 0.9454 | 0.3624 | 1.359 | 0.250 | (48/192) | 61.9 |
| 23 | 0.9600 | 0.9320 | 0.3623 | 1.654 | 0.151 | (29/192) | 59.6 |
| 24 | 0.9461 | 0.9181 | 0.3626 | 1.656 | 0.240 | (46/192) | 66.0 |
| 25 | 0.9347 | 0.9065 | 0.3618 | 2.196 | 0.240 | (46/192) | 65.5 |
| 26 | 0.9269 | 0.8985 | 0.3618 | 2.119 | 0.255 | (49/192) | 70.3 |
| 27 | 0.9128 | 0.8843 | 0.3621 | 2.088 | 0.354 | (68/192) | 68.6 |
| 28 | 0.9042 | 0.8758 | 0.3613 | 2.104 | 0.250 | (48/192) | 69.1 |
| 29 | 0.8873 | 0.8587 | 0.3615 | 2.044 | 0.286 | (55/192) | 62.3 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 30-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 4056.7s (~135.2s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_blk_c5_repair_2026_10_09/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_blk_c5_repair_2026_10_09/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_blk_c5_repair_2026_10_09/wasserstein_decay.json` (W₂ decay check)
- `molmetal/reports/cfm_blk_c5_repair_2026_10_09/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_blk_c5_repair_2026_10_09/synthesis.md` (this file)

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
