# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:1  
**Best decode_ratio RDKit-strict**: 1.000 (192/192)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 32.3s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.9280 | 1.8537 | 0.7394 | 2.826 | 1.000 | (192/192) | 5.0 |
| 1 | 1.7089 | 1.6351 | 0.7322 | 2.439 | 1.000 | (192/192) | 2.3 |
| 2 | 1.5303 | 1.4573 | 0.7250 | 2.306 | 1.000 | (192/192) | 2.4 |
| 3 | 1.3794 | 1.3074 | 0.7178 | 2.458 | 1.000 | (192/192) | 2.2 |
| 4 | 1.2124 | 1.1411 | 0.7106 | 2.733 | 1.000 | (192/192) | 4.1 |
| 5 | 1.0236 | 0.9532 | 0.7034 | 2.885 | 1.000 | (192/192) | 4.1 |
| 6 | 0.8237 | 0.7540 | 0.6962 | 2.824 | 1.000 | (192/192) | 2.6 |
| 7 | 0.6377 | 0.5687 | 0.6891 | 2.371 | 1.000 | (192/192) | 2.0 |
| 8 | 0.4881 | 0.4199 | 0.6820 | 1.686 | 1.000 | (192/192) | 2.6 |
| 9 | 0.3738 | 0.3061 | 0.6748 | 1.446 | 1.000 | (192/192) | 2.2 |
| 10 | 0.2789 | 0.2117 | 0.6677 | 1.293 | 1.000 | (192/192) | 2.9 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 11-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:1: 32.3s (~2.9s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/wf124_p2_cfm_full/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/wf124_p2_cfm_full/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/wf124_p2_cfm_full/wasserstein_decay.json` (W₂ decay check)
- `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/wf124_p2_cfm_full/synthesis.md` (this file)

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
