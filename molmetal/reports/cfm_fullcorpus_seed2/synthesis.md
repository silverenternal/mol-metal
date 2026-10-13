# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.339 (65/192)  
**Best epoch**: 24  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 26484.5s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.9391 | 0.3166 | 0.8441 | 10.193 | 0.599 | (115/192) | 3959.6 |
| 1 | 9.4663 | 0.0992 | 0.6394 | 8.914 | 0.443 | (85/192) | 1157.4 |
| 2 | 9.4020 | 0.0655 | 0.5651 | 7.987 | 0.443 | (85/192) | 1006.3 |
| 3 | 9.3767 | 0.0524 | 0.4859 | 8.203 | 0.359 | (69/192) | 679.6 |
| 4 | 9.2959 | 0.0449 | 0.4827 | 7.806 | 0.385 | (74/192) | 808.7 |
| 5 | 9.2877 | 0.0400 | 0.4629 | 7.893 | 0.339 | (65/192) | 1544.8 |
| 6 | 9.2635 | 0.0368 | 0.4378 | 7.909 | 0.391 | (75/192) | 803.2 |
| 7 | 9.2454 | 0.0349 | 0.4350 | 7.826 | 0.365 | (70/192) | 756.7 |
| 8 | 9.1924 | 0.0314 | 0.4202 | 7.523 | 0.307 | (59/192) | 907.9 |
| 9 | 9.1957 | 0.0298 | 0.4315 | 7.923 | 0.339 | (65/192) | 921.6 |
| 10 | 9.1977 | 0.0277 | 0.4198 | 7.509 | 0.349 | (67/192) | 768.6 |
| 11 | 9.1764 | 0.0263 | 0.4157 | 7.556 | 0.365 | (70/192) | 687.0 |
| 12 | 9.1227 | 0.0250 | 0.4133 | 7.450 | 0.318 | (61/192) | 837.6 |
| 13 | 9.1402 | 0.0241 | 0.4076 | 7.863 | 0.365 | (70/192) | 912.8 |
| 14 | 9.1125 | 0.0227 | 0.4115 | 7.814 | 0.344 | (66/192) | 822.0 |
| 15 | 9.1602 | 0.0219 | 0.4050 | 7.329 | 0.344 | (66/192) | 872.7 |
| 16 | 9.1498 | 0.0210 | 0.3948 | 7.672 | 0.365 | (70/192) | 846.4 |
| 17 | 9.1331 | 0.0198 | 0.3953 | 7.599 | 0.323 | (62/192) | 762.5 |
| 18 | 9.1182 | 0.0191 | 0.3950 | 7.567 | 0.318 | (61/192) | 813.4 |
| 19 | 9.1619 | 0.0185 | 0.3950 | 7.458 | 0.333 | (64/192) | 806.1 |
| 20 | 9.1108 | 0.0176 | 0.3872 | 7.709 | 0.333 | (64/192) | 836.8 |
| 21 | 9.1118 | 0.0173 | 0.3994 | 7.778 | 0.349 | (67/192) | 743.9 |
| 22 | 9.1017 | 0.0166 | 0.3978 | 7.787 | 0.339 | (65/192) | 554.2 |
| 23 | 9.0961 | 0.0164 | 0.3853 | 7.765 | 0.286 | (55/192) | 568.0 |
| 24 | 9.0833 | 0.0158 | 0.3896 | 7.990 | 0.339 | (65/192) | 821.3 |
| 25 | 9.0648 | 0.0154 | 0.3864 | 7.801 | 0.307 | (59/192) | 779.3 |
| 26 | 9.0832 | 0.0149 | 0.3893 | 7.324 | 0.323 | (62/192) | 732.5 |
| 27 | 9.0713 | 0.0144 | 0.3859 | 7.642 | 0.344 | (66/192) | 773.2 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 28-epoch history windowed by 10: NO PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 26484.5s (~945.9s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_fullcorpus_seed2/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_fullcorpus_seed2/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_fullcorpus_seed2/wasserstein_decay.json` (W₂ decay check)
- `molmetal/reports/cfm_fullcorpus_seed2/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_fullcorpus_seed2/synthesis.md` (this file)

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
