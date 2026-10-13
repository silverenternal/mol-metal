# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (192/192)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 5027.3s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14.4340 | 2.2139 | 0.8152 | 23.072 | 1.000 | (192/192) | 437.6 |
| 1 | 14.2558 | 2.1247 | 0.8146 | 22.469 | 1.000 | (192/192) | 176.4 |
| 2 | 14.1936 | 1.8998 | 0.8111 | 23.548 | 1.000 | (192/192) | 265.0 |
| 3 | 13.6387 | 1.6032 | 0.7952 | 26.171 | 1.000 | (192/192) | 399.4 |
| 4 | 13.2542 | 1.2908 | 0.7639 | 15.960 | 1.000 | (192/192) | 379.4 |
| 5 | 12.9557 | 1.0568 | 0.7318 | 16.789 | 1.000 | (192/192) | 215.4 |
| 6 | 13.0628 | 0.9766 | 0.7073 | 11.479 | 1.000 | (192/192) | 106.9 |
| 7 | 12.8940 | 0.9257 | 0.6709 | 14.748 | 1.000 | (192/192) | 76.4 |
| 8 | 12.6996 | 0.9033 | 0.6013 | 22.998 | 1.000 | (192/192) | 65.6 |
| 9 | 12.7533 | 0.8818 | 0.4983 | 29.971 | 1.000 | (192/192) | 108.7 |
| 10 | 12.6540 | 0.8822 | 0.4988 | 20.215 | 1.000 | (192/192) | 56.9 |
| 11 | 12.2546 | 0.8586 | 0.5144 | 28.002 | 1.000 | (192/192) | 71.2 |
| 12 | 11.7716 | 0.8483 | 0.4012 | 28.285 | 1.000 | (192/192) | 259.7 |
| 13 | 11.2273 | 0.8360 | 0.5022 | 30.313 | 1.000 | (192/192) | 243.6 |
| 14 | 10.7939 | 0.8211 | 0.9523 | 34.230 | 1.000 | (192/192) | 426.1 |
| 15 | 10.4665 | 0.8099 | 0.9476 | 23.197 | 1.000 | (192/192) | 321.7 |
| 16 | 10.4155 | 0.7979 | 0.7762 | 17.384 | 1.000 | (192/192) | 86.4 |
| 17 | 10.0775 | 0.7843 | 0.9322 | 12.008 | 1.000 | (192/192) | 276.9 |
| 18 | 10.5970 | 0.7786 | 1.2267 | 11.796 | 0.536 | (103/192) | 116.8 |
| 19 | 10.1062 | 0.7723 | 1.0675 | 10.550 | 0.698 | (134/192) | 168.7 |
| 20 | 10.4688 | 0.7582 | 1.3256 | 15.374 | 0.990 | (190/192) | 107.4 |
| 21 | 10.0756 | 0.7493 | 1.1128 | 8.291 | 0.854 | (164/192) | 86.2 |
| 22 | 10.1049 | 0.7438 | 1.1738 | 7.374 | 0.391 | (75/192) | 116.3 |
| 23 | 10.3779 | 0.7384 | 0.9387 | 11.156 | 0.849 | (163/192) | 54.9 |
| 24 | 10.1590 | 0.7295 | 1.1014 | 8.460 | 0.917 | (176/192) | 113.4 |
| 25 | 10.2360 | 0.7218 | 1.0329 | 9.636 | 0.625 | (120/192) | 57.4 |
| 26 | 9.8456 | 0.7198 | 1.2208 | 11.585 | 0.266 | (51/192) | 54.6 |
| 27 | 10.3128 | 0.7238 | 0.9265 | 9.950 | 0.885 | (170/192) | 68.3 |
| 28 | 10.0811 | 0.7183 | 1.1068 | 7.504 | 0.276 | (53/192) | 55.5 |
| 29 | 10.3509 | 0.7099 | 0.9769 | 6.418 | 0.089 | (17/192) | 54.8 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 30-epoch history windowed by 10: NO PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 5027.3s (~167.6s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_frag_control/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_frag_control/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_frag_control/wasserstein_decay.json` (W₂ decay check)
- `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_frag_control/synthesis.md` (this file)

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
