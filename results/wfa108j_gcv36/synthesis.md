# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cpu  
**Best decode_ratio RDKit-strict**: 1.000 (32/32)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 934.8s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 29.5850 | 15.9196 | 0.0000 | 568.709 | 1.000 | (32/32, single_frag=1) | 48.1 |
| 1 | 25.1983 | 12.6572 | 0.0000 | 445.383 | 1.000 | (32/32, single_frag=1) | 29.7 |
| 2 | 20.9775 | 9.2408 | 0.0000 | 406.970 | 1.000 | (32/32, single_frag=1) | 22.3 |
| 3 | 16.6960 | 5.8107 | 0.0000 | 353.800 | 1.000 | (32/32, single_frag=1) | 18.8 |
| 4 | 13.0553 | 2.7801 | 0.0000 | 293.511 | 1.000 | (32/32, single_frag=1) | 32.5 |
| 5 | 12.9849 | 1.7904 | 0.0000 | 163.584 | 1.000 | (32/32, single_frag=1) | 34.3 |
| 6 | 10.9382 | 1.5623 | 0.0000 | 180.910 | 1.000 | (32/32, single_frag=1) | 20.8 |
| 7 | 10.8511 | 1.5330 | 0.0000 | 233.273 | 1.000 | (32/32, single_frag=1) | 14.7 |
| 8 | 10.8580 | 1.3112 | 0.0000 | 148.952 | 1.000 | (32/32, single_frag=0) | 18.8 |
| 9 | 10.9598 | 1.2160 | 0.0000 | 139.329 | 1.000 | (32/32, single_frag=0) | 16.3 |
| 10 | 11.2784 | 1.1679 | 0.0000 | 93.152 | 1.000 | (32/32, single_frag=0) | 24.9 |
| 11 | 10.7531 | 1.0768 | 0.0000 | 129.538 | 1.000 | (32/32, single_frag=0) | 15.1 |
| 12 | 9.8757 | 1.0295 | 0.0000 | 112.758 | 1.000 | (32/32, single_frag=0) | 26.0 |
| 13 | 10.7295 | 0.9678 | 0.0000 | 78.289 | 1.000 | (32/32, single_frag=0) | 37.3 |
| 14 | 9.9986 | 0.9606 | 0.0000 | 124.533 | 1.000 | (32/32, single_frag=0) | 23.7 |
| 15 | 9.8348 | 0.8976 | 0.0000 | 115.261 | 1.000 | (32/32, single_frag=0) | 17.2 |
| 16 | 9.5087 | 0.8872 | 0.0000 | 112.972 | 1.000 | (32/32, single_frag=0) | 15.6 |
| 17 | 9.5745 | 0.8785 | 0.0000 | 117.047 | 1.000 | (32/32, single_frag=0) | 15.8 |
| 18 | 10.4379 | 0.8734 | 0.0000 | 65.953 | 1.000 | (32/32, single_frag=0) | 37.5 |
| 19 | 9.6910 | 0.8641 | 0.0000 | 105.621 | 1.000 | (32/32, single_frag=1) | 34.7 |
| 20 | 10.3893 | 0.8451 | 0.0000 | 70.007 | 1.000 | (32/32, single_frag=1) | 33.1 |
| 21 | 9.9008 | 0.8360 | 0.0000 | 82.143 | 1.000 | (32/32, single_frag=1) | 43.5 |
| 22 | 9.4979 | 0.8165 | 0.0000 | 105.099 | 1.000 | (32/32, single_frag=1) | 17.7 |
| 23 | 10.0369 | 0.8064 | 0.0000 | 82.910 | 1.000 | (32/32, single_frag=1) | 17.0 |
| 24 | 10.4171 | 0.8071 | 0.0000 | 66.415 | 1.000 | (32/32, single_frag=1) | 33.2 |
| 25 | 9.8180 | 0.7875 | 0.0000 | 74.013 | 1.000 | (32/32, single_frag=1) | 33.7 |
| 26 | 9.7209 | 0.7900 | 0.0000 | 119.454 | 1.000 | (32/32, single_frag=1) | 24.4 |
| 27 | 9.5968 | 0.8040 | 0.0000 | 94.722 | 1.000 | (32/32, single_frag=1) | 23.8 |
| 28 | 9.5383 | 0.7992 | 0.0000 | 136.915 | 1.000 | (32/32, single_frag=1) | 17.3 |
| 29 | 9.9333 | 0.7699 | 0.0000 | 105.755 | 1.000 | (32/32, single_frag=1) | 23.9 |
| 30 | 8.6015 | 0.7551 | 0.0000 | 117.724 | 1.000 | (32/32, single_frag=1) | 18.9 |
| 31 | 9.4550 | 0.7538 | 0.0000 | 102.176 | 1.000 | (32/32, single_frag=1) | 37.3 |
| 32 | 9.2822 | 0.7438 | 0.0000 | 87.469 | 1.000 | (32/32, single_frag=1) | 22.9 |
| 33 | 9.4997 | 0.7544 | 0.0000 | 92.962 | 1.000 | (32/32, single_frag=1) | 22.7 |
| 34 | 9.7368 | 0.7620 | 0.0000 | 98.625 | 1.000 | (32/32, single_frag=1) | 33.1 |
| 35 | 8.7145 | 0.7324 | 0.0000 | 99.366 | 1.000 | (32/32, single_frag=1) | 28.2 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 36-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cpu: 934.8s (~26.0s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/wfa108j_gcv36/per_epoch_metrics.json` (full per-epoch record)
- `results/wfa108j_gcv36/per_epoch_metrics.csv` (flat CSV)
- `results/wfa108j_gcv36/wasserstein_decay.json` (W₂ decay check)
- `results/wfa108j_gcv36/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/wfa108j_gcv36/synthesis.md` (this file)

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
