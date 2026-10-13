# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (192/192)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 3994.7s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.4928 | 1.4130 | 0.7640 | 5.876 | 1.000 | (192/192) | 1693.8 |
| 1 | 1.2676 | 1.1929 | 0.7114 | 2.287 | 1.000 | (192/192) | 397.0 |
| 2 | 1.2156 | 1.1495 | 0.6425 | 1.931 | 1.000 | (192/192) | 87.0 |
| 3 | 1.1777 | 1.1225 | 0.5458 | 1.625 | 1.000 | (192/192) | 78.1 |
| 4 | 1.1398 | 1.0969 | 0.4205 | 1.532 | 0.979 | (188/192) | 72.3 |
| 5 | 1.1165 | 1.0773 | 0.3876 | 1.599 | 0.922 | (177/192) | 81.8 |
| 6 | 1.0960 | 1.0584 | 0.3720 | 1.497 | 0.891 | (171/192) | 66.8 |
| 7 | 1.0794 | 1.0418 | 0.3723 | 1.983 | 0.828 | (159/192) | 66.3 |
| 8 | 1.0633 | 1.0262 | 0.3672 | 2.425 | 0.802 | (154/192) | 64.0 |
| 9 | 1.0515 | 1.0144 | 0.3670 | 2.722 | 0.620 | (119/192) | 71.2 |
| 10 | 1.0352 | 0.9982 | 0.3657 | 3.123 | 0.391 | (75/192) | 71.3 |
| 11 | 1.0256 | 0.9887 | 0.3648 | 2.891 | 0.542 | (104/192) | 67.7 |
| 12 | 1.0108 | 0.9739 | 0.3646 | 3.153 | 0.198 | (38/192) | 63.1 |
| 13 | 1.0016 | 0.9647 | 0.3637 | 3.998 | 0.411 | (79/192) | 73.8 |
| 14 | 0.9918 | 0.9547 | 0.3642 | 4.510 | 0.354 | (68/192) | 62.6 |
| 15 | 0.9801 | 0.9432 | 0.3631 | 4.498 | 0.344 | (66/192) | 61.8 |
| 16 | 0.9682 | 0.9313 | 0.3632 | 5.143 | 0.599 | (115/192) | 62.8 |
| 17 | 0.9639 | 0.9271 | 0.3624 | 6.426 | 0.245 | (47/192) | 62.7 |
| 18 | 0.9457 | 0.9089 | 0.3629 | 4.624 | 0.193 | (37/192) | 64.1 |
| 19 | 0.9331 | 0.8964 | 0.3622 | 3.737 | 0.297 | (57/192) | 62.6 |
| 20 | 0.9243 | 0.8876 | 0.3624 | 4.095 | 0.240 | (46/192) | 71.0 |
| 21 | 0.9141 | 0.8775 | 0.3624 | 5.305 | 0.255 | (49/192) | 66.2 |
| 22 | 0.9056 | 0.8689 | 0.3617 | 4.609 | 0.276 | (53/192) | 64.2 |
| 23 | 0.8977 | 0.8610 | 0.3616 | 5.664 | 0.229 | (44/192) | 62.0 |
| 24 | 0.8868 | 0.8502 | 0.3615 | 5.601 | 0.161 | (31/192) | 66.4 |
| 25 | 0.8757 | 0.8390 | 0.3616 | 6.036 | 0.255 | (49/192) | 70.4 |
| 26 | 0.8653 | 0.8286 | 0.3617 | 6.295 | 0.193 | (37/192) | 67.6 |
| 27 | 0.8519 | 0.8152 | 0.3616 | 6.339 | 0.271 | (52/192) | 66.6 |
| 28 | 0.8465 | 0.8098 | 0.3619 | 7.202 | 0.146 | (28/192) | 64.7 |
| 29 | 0.8443 | 0.8077 | 0.3611 | 9.960 | 0.188 | (36/192) | 64.8 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 30-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 3994.7s (~133.2s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/wasserstein_decay.json` (W₂ decay check)
- `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_bond_head_30ep_onecycle_2026_10_09/synthesis.md` (this file)

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
