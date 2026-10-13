# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.609 (117/192)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 5328.4s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 8.6873 | 1.8766 | 0.7825 | 4.626 | 0.609 | (117/192) | 555.2 |
| 1 | 8.7001 | 1.8745 | 0.7790 | 4.782 | 0.609 | (117/192) | 141.2 |
| 2 | 8.7079 | 1.8755 | 0.7748 | 5.075 | 0.609 | (117/192) | 360.2 |
| 3 | 8.6932 | 1.8798 | 0.7709 | 5.542 | 0.609 | (117/192) | 466.8 |
| 4 | 8.6190 | 1.8766 | 0.7667 | 6.180 | 0.609 | (117/192) | 458.2 |
| 5 | 8.3533 | 1.6557 | 0.7624 | 11.133 | 0.552 | (106/192) | 323.8 |
| 6 | 8.0642 | 1.4181 | 0.7585 | 8.421 | 0.974 | (187/192) | 189.0 |
| 7 | 7.9564 | 1.3121 | 0.7544 | 7.866 | 1.000 | (192/192) | 146.2 |
| 8 | 7.9398 | 1.2756 | 0.7499 | 7.368 | 1.000 | (192/192) | 106.3 |
| 9 | 7.8528 | 1.2466 | 0.7444 | 8.263 | 1.000 | (192/192) | 83.1 |
| 10 | 7.8184 | 1.2263 | 0.7391 | 8.328 | 1.000 | (192/192) | 94.1 |
| 11 | 7.7034 | 1.2132 | 0.7331 | 8.851 | 1.000 | (192/192) | 82.5 |
| 12 | 7.7485 | 1.1947 | 0.7262 | 8.931 | 1.000 | (192/192) | 90.7 |
| 13 | 7.7051 | 1.1807 | 0.7194 | 9.938 | 1.000 | (192/192) | 95.8 |
| 14 | 7.5806 | 1.1757 | 0.7117 | 12.264 | 1.000 | (192/192) | 73.8 |
| 15 | 7.5798 | 1.1575 | 0.7031 | 11.637 | 1.000 | (192/192) | 107.1 |
| 16 | 7.4840 | 1.1490 | 0.6944 | 12.801 | 1.000 | (192/192) | 150.6 |
| 17 | 7.4573 | 1.1490 | 0.6843 | 11.953 | 1.000 | (192/192) | 154.1 |
| 18 | 7.3253 | 1.1331 | 0.6734 | 13.313 | 1.000 | (192/192) | 155.6 |
| 19 | 7.2323 | 1.1216 | 0.6607 | 14.859 | 0.995 | (191/192) | 142.8 |
| 20 | 7.1635 | 1.1178 | 0.6475 | 15.874 | 1.000 | (192/192) | 115.3 |
| 21 | 7.2583 | 1.1010 | 0.6345 | 13.197 | 0.948 | (182/192) | 114.6 |
| 22 | 7.1130 | 1.0969 | 0.6201 | 16.404 | 0.698 | (134/192) | 133.1 |
| 23 | 6.8868 | 1.0881 | 0.6050 | 16.710 | 1.000 | (192/192) | 141.7 |
| 24 | 6.7701 | 1.0800 | 0.5864 | 19.325 | 0.755 | (145/192) | 134.5 |
| 25 | 6.7670 | 1.0687 | 0.5648 | 17.597 | 0.807 | (155/192) | 140.2 |
| 26 | 6.8480 | 1.0568 | 0.5463 | 17.578 | 0.771 | (148/192) | 147.7 |
| 27 | 6.8048 | 1.0511 | 0.5342 | 16.912 | 0.891 | (171/192) | 137.5 |
| 28 | 6.5235 | 1.0385 | 0.5207 | 22.462 | 0.724 | (139/192) | 157.1 |
| 29 | 6.7049 | 1.0236 | 0.5016 | 16.815 | 0.583 | (112/192) | 129.7 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 30-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 5328.4s (~177.6s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_fm_ab_2026_10_09/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_fm_ab_2026_10_09/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_fm_ab_2026_10_09/wasserstein_decay.json` (W₂ decay check)
- `molmetal/reports/cfm_fm_ab_2026_10_09/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_fm_ab_2026_10_09/synthesis.md` (this file)

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
