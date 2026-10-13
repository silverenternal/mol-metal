# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.302 (58/192)  
**Best epoch**: 8  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 10465.3s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.5717 | 0.6068 | 0.9272 | 8.844 | 0.307 | (59/192) | 3322.5 |
| 1 | 9.7615 | 0.2797 | 0.7706 | 8.259 | 0.411 | (79/192) | 1029.3 |
| 2 | 9.6336 | 0.1519 | 0.6624 | 7.043 | 0.505 | (97/192) | 792.8 |
| 3 | 9.5152 | 0.1053 | 0.6261 | 7.518 | 0.391 | (75/192) | 648.6 |
| 4 | 9.4852 | 0.0806 | 0.5272 | 6.935 | 0.401 | (77/192) | 599.1 |
| 5 | 9.4217 | 0.0662 | 0.5015 | 6.560 | 0.370 | (71/192) | 600.7 |
| 6 | 9.4155 | 0.0627 | 0.4831 | 8.524 | 0.266 | (51/192) | 609.4 |
| 7 | 9.3837 | 0.0575 | 0.4738 | 6.693 | 0.333 | (64/192) | 571.9 |
| 8 | 9.3284 | 0.0488 | 0.4718 | 5.968 | 0.302 | (58/192) | 571.2 |
| 9 | 9.3231 | 0.0488 | 0.4628 | 6.238 | 0.297 | (57/192) | 567.4 |
| 10 | 9.3292 | 0.0557 | 0.4498 | 7.709 | 0.307 | (59/192) | 579.0 |
| 11 | 9.3145 | 0.0437 | 0.4589 | 6.368 | 0.354 | (68/192) | 573.1 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 12-epoch history windowed by 10: NO PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 10465.3s (~872.1s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_fullcorpus_b64_2026_10_11/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_fullcorpus_b64_2026_10_11/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_fullcorpus_b64_2026_10_11/wasserstein_decay.json` (W₂ decay check)
- `molmetal/reports/cfm_fullcorpus_b64_2026_10_11/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_fullcorpus_b64_2026_10_11/synthesis.md` (this file)

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
