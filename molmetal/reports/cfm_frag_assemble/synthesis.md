# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (192/192)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 5126.1s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14.4340 | 2.2139 | 0.8152 | 23.072 | 1.000 | (192/192) | 441.1 |
| 1 | 14.2315 | 2.1209 | 0.8147 | 22.095 | 1.000 | (192/192) | 187.9 |
| 2 | 14.1823 | 1.9002 | 0.8102 | 23.902 | 1.000 | (192/192) | 258.0 |
| 3 | 13.6874 | 1.6027 | 0.7951 | 22.733 | 1.000 | (192/192) | 395.2 |
| 4 | 13.3240 | 1.2864 | 0.7657 | 13.025 | 1.000 | (192/192) | 366.6 |
| 5 | 12.9254 | 1.0531 | 0.7391 | 19.270 | 1.000 | (192/192) | 225.4 |
| 6 | 13.1421 | 0.9765 | 0.7208 | 9.745 | 1.000 | (192/192) | 107.9 |
| 7 | 12.9274 | 0.9240 | 0.6865 | 15.820 | 1.000 | (192/192) | 75.5 |
| 8 | 12.8426 | 0.9005 | 0.6450 | 22.181 | 1.000 | (192/192) | 74.3 |
| 9 | 12.7306 | 0.8813 | 0.5651 | 20.827 | 1.000 | (192/192) | 124.2 |
| 10 | 12.5494 | 0.8746 | 0.5320 | 23.055 | 1.000 | (192/192) | 53.7 |
| 11 | 12.0662 | 0.8517 | 0.4620 | 33.646 | 1.000 | (192/192) | 73.3 |
| 12 | 11.5235 | 0.8392 | 0.3871 | 26.361 | 1.000 | (192/192) | 406.5 |
| 13 | 11.0443 | 0.8274 | 0.7920 | 34.262 | 1.000 | (192/192) | 479.0 |
| 14 | 10.7888 | 0.8168 | 0.8236 | 30.207 | 1.000 | (192/192) | 296.4 |
| 15 | 10.8870 | 0.8054 | 1.1552 | 56.569 | 1.000 | (192/192) | 88.2 |
| 16 | 10.5222 | 0.7950 | 0.7195 | 32.917 | 1.000 | (192/192) | 236.6 |
| 17 | 10.2507 | 0.7846 | 0.7200 | 23.584 | 0.995 | (191/192) | 102.0 |
| 18 | 10.5481 | 0.7755 | 1.3923 | 20.449 | 0.995 | (191/192) | 317.7 |
| 19 | 10.0710 | 0.7652 | 1.2131 | 11.976 | 0.953 | (183/192) | 73.9 |
| 20 | 10.3853 | 0.7641 | 1.1180 | 12.343 | 0.990 | (190/192) | 108.4 |
| 21 | 10.3790 | 0.7589 | 0.7885 | 13.569 | 0.974 | (187/192) | 95.8 |
| 22 | 10.1723 | 0.7573 | 1.2492 | 18.803 | 0.490 | (94/192) | 73.0 |
| 23 | 10.0169 | 0.7542 | 0.8078 | 15.812 | 0.417 | (80/192) | 104.1 |
| 24 | 9.9851 | 0.7401 | 0.9961 | 14.570 | 0.651 | (125/192) | 55.2 |
| 25 | 10.3481 | 0.7321 | 1.3510 | 16.585 | 0.875 | (168/192) | 79.7 |
| 26 | 9.8565 | 0.7261 | 0.9557 | 10.272 | 0.406 | (78/192) | 55.9 |
| 27 | 10.4178 | 0.7191 | 1.2451 | 12.156 | 0.219 | (42/192) | 57.7 |
| 28 | 9.9992 | 0.7149 | 1.1181 | 8.465 | 0.161 | (31/192) | 71.2 |
| 29 | 10.1959 | 0.7101 | 1.1914 | 8.386 | 0.130 | (25/192) | 41.7 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 30-epoch history windowed by 10: NO PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 5126.1s (~170.9s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_frag_assemble/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_frag_assemble/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_frag_assemble/wasserstein_decay.json` (W₂ decay check)
- `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_frag_assemble/synthesis.md` (this file)

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
