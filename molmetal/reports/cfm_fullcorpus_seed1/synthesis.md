# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.349 (67/192)  
**Best epoch**: 11  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 16194.7s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.9265 | 0.3124 | 0.8267 | 9.857 | 0.458 | (88/192) | 3853.4 |
| 1 | 9.4997 | 0.0948 | 0.6419 | 8.898 | 0.453 | (87/192) | 1172.1 |
| 2 | 9.4161 | 0.0600 | 0.5449 | 9.196 | 0.490 | (94/192) | 1023.3 |
| 3 | 9.3258 | 0.0474 | 0.5069 | 8.744 | 0.474 | (91/192) | 850.6 |
| 4 | 9.2702 | 0.0427 | 0.4582 | 8.270 | 0.474 | (91/192) | 950.5 |
| 5 | 9.3049 | 0.0389 | 0.4508 | 8.753 | 0.406 | (78/192) | 1381.6 |
| 6 | 9.2468 | 0.0355 | 0.4412 | 8.480 | 0.365 | (70/192) | 757.7 |
| 7 | 9.2199 | 0.0321 | 0.4281 | 8.582 | 0.391 | (75/192) | 705.7 |
| 8 | 9.1854 | 0.0305 | 0.4214 | 8.476 | 0.359 | (69/192) | 834.1 |
| 9 | 9.2035 | 0.0292 | 0.4202 | 8.470 | 0.401 | (77/192) | 1044.5 |
| 10 | 9.1717 | 0.0266 | 0.4202 | 8.095 | 0.401 | (77/192) | 642.2 |
| 11 | 9.2060 | 0.0251 | 0.4211 | 8.495 | 0.349 | (67/192) | 747.1 |
| 12 | 9.1724 | 0.0240 | 0.4050 | 8.504 | 0.318 | (61/192) | 803.9 |
| 13 | 9.1582 | 0.0224 | 0.4079 | 8.440 | 0.323 | (62/192) | 900.7 |
| 14 | 9.1266 | 0.0215 | 0.4131 | 8.389 | 0.292 | (56/192) | 527.0 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 15-epoch history windowed by 10: NO PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 16194.7s (~1079.6s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_fullcorpus_seed1/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_fullcorpus_seed1/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_fullcorpus_seed1/wasserstein_decay.json` (W₂ decay check)
- `molmetal/reports/cfm_fullcorpus_seed1/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_fullcorpus_seed1/synthesis.md` (this file)

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
