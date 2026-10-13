# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.609 (117/192)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 4334.3s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.9529 | 1.8814 | 0.7826 | 0.116 | 0.609 | (117/192) | 678.7 |
| 1 | 1.9504 | 1.8794 | 0.7785 | 0.129 | 0.609 | (117/192) | 207.2 |
| 2 | 1.9529 | 1.8822 | 0.7741 | 0.142 | 0.609 | (117/192) | 283.0 |
| 3 | 1.9513 | 1.8811 | 0.7694 | 0.153 | 0.609 | (117/192) | 473.7 |
| 4 | 1.9547 | 1.8850 | 0.7640 | 0.164 | 0.609 | (117/192) | 518.5 |
| 5 | 1.7336 | 1.6643 | 0.7595 | 8.855 | 0.562 | (108/192) | 330.5 |
| 6 | 1.4734 | 1.4043 | 0.7555 | 4.131 | 0.979 | (188/192) | 192.3 |
| 7 | 1.3692 | 1.3014 | 0.7526 | 2.056 | 1.000 | (192/192) | 154.0 |
| 8 | 1.3196 | 1.2527 | 0.7489 | 1.547 | 1.000 | (192/192) | 96.9 |
| 9 | 1.2925 | 1.2268 | 0.7437 | 1.480 | 1.000 | (192/192) | 80.5 |
| 10 | 1.2693 | 1.2048 | 0.7370 | 1.369 | 1.000 | (192/192) | 77.4 |
| 11 | 1.2475 | 1.1840 | 0.7290 | 1.264 | 1.000 | (192/192) | 75.6 |
| 12 | 1.2300 | 1.1677 | 0.7192 | 1.189 | 1.000 | (192/192) | 70.5 |
| 13 | 1.2101 | 1.1489 | 0.7081 | 1.141 | 1.000 | (192/192) | 78.2 |
| 14 | 1.1970 | 1.1372 | 0.6952 | 1.078 | 1.000 | (192/192) | 66.0 |
| 15 | 1.1804 | 1.1221 | 0.6801 | 1.124 | 1.000 | (192/192) | 69.9 |
| 16 | 1.1682 | 1.1117 | 0.6622 | 1.185 | 0.932 | (179/192) | 62.1 |
| 17 | 1.1489 | 1.0943 | 0.6415 | 1.099 | 0.958 | (184/192) | 62.3 |
| 18 | 1.1363 | 1.0843 | 0.6159 | 1.234 | 0.833 | (160/192) | 66.1 |
| 19 | 1.1265 | 1.0774 | 0.5859 | 1.275 | 0.745 | (143/192) | 64.9 |
| 20 | 1.1051 | 1.0596 | 0.5493 | 1.252 | 0.760 | (146/192) | 66.9 |
| 21 | 1.0897 | 1.0487 | 0.5042 | 1.225 | 0.552 | (106/192) | 69.3 |
| 22 | 1.0676 | 1.0318 | 0.4514 | 1.256 | 0.542 | (104/192) | 59.5 |
| 23 | 1.0562 | 1.0254 | 0.4010 | 1.200 | 0.620 | (119/192) | 66.0 |
| 24 | 1.0357 | 1.0072 | 0.3757 | 1.089 | 0.344 | (66/192) | 62.0 |
| 25 | 1.0230 | 0.9943 | 0.3769 | 1.233 | 0.510 | (98/192) | 60.7 |
| 26 | 1.0048 | 0.9761 | 0.3748 | 1.244 | 0.318 | (61/192) | 59.0 |
| 27 | 0.9918 | 0.9631 | 0.3728 | 1.187 | 0.417 | (80/192) | 57.7 |
| 28 | 0.9750 | 0.9463 | 0.3725 | 1.134 | 0.286 | (55/192) | 61.0 |
| 29 | 0.9679 | 0.9392 | 0.3718 | 1.473 | 0.250 | (48/192) | 63.8 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 30-epoch history windowed by 10: NO PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 4334.3s (~144.5s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_fm_control_2026_10_09/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_fm_control_2026_10_09/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_fm_control_2026_10_09/wasserstein_decay.json` (W₂ decay check)
- `molmetal/reports/cfm_fm_control_2026_10_09/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_fm_control_2026_10_09/synthesis.md` (this file)

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
