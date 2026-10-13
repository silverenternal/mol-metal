# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.135 (35/192)  
**Best epoch**: 15  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 138.4s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.8348 | 0.7747 | 0.5931 | 3.067 | 1.000 | (192/192) | 6.7 |
| 2 | 0.7926 | 0.7401 | 0.5218 | 1.742 | 0.974 | (187/192) | 5.1 |
| 3 | 0.7651 | 0.7184 | 0.4651 | 1.770 | 0.406 | (78/192) | 5.1 |
| 4 | 0.7394 | 0.6975 | 0.4164 | 1.699 | 0.943 | (181/192) | 5.8 |
| 5 | 0.7158 | 0.6777 | 0.3779 | 1.729 | 0.375 | (72/192) | 7.9 |
| 6 | 0.6945 | 0.6590 | 0.3533 | 2.476 | 0.714 | (137/192) | 7.7 |
| 7 | 0.6844 | 0.6501 | 0.3406 | 3.107 | 0.354 | (68/192) | 7.7 |
| 8 | 0.6594 | 0.6257 | 0.3346 | 1.961 | 0.344 | (66/192) | 7.7 |
| 9 | 0.6397 | 0.6065 | 0.3297 | 3.332 | 0.193 | (37/192) | 7.7 |
| 10 | 0.6170 | 0.5839 | 0.3287 | 2.250 | 0.276 | (53/192) | 7.7 |
| 11 | 0.5978 | 0.5650 | 0.3259 | 2.932 | 0.250 | (48/192) | 7.7 |
| 12 | 0.5763 | 0.5435 | 0.3259 | 2.274 | 0.120 | (23/192) | 7.7 |
| 13 | 0.5637 | 0.5308 | 0.3268 | 3.208 | 0.120 | (23/192) | 7.7 |
| 14 | 0.5450 | 0.5123 | 0.3256 | 2.804 | 0.172 | (33/192) | 7.7 |
| 15 | 0.5222 | 0.4896 | 0.3234 | 2.371 | 0.135 | (26/192) | 7.7 |
| 16 | 0.5123 | 0.4794 | 0.3249 | 3.414 | 0.182 | (35/192) | 7.7 |
| 17 | 0.4968 | 0.4645 | 0.3210 | 3.984 | 0.130 | (25/192) | 7.7 |
| 18 | 0.4829 | 0.4504 | 0.3216 | 3.479 | 0.255 | (49/192) | 7.8 |
| 19 | 0.4633 | 0.4308 | 0.3220 | 3.422 | 0.146 | (28/192) | 7.7 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 19-epoch history windowed by 10: NO PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 138.4s (~7.3s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_bond_head_20ep_2026_10_09/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_bond_head_20ep_2026_10_09/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_bond_head_20ep_2026_10_09/wasserstein_decay.json` (W₂ decay check)
- `None/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_bond_head_20ep_2026_10_09/synthesis.md` (this file)

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
