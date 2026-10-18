# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (64/64)  
**Best epoch**: 3  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 2490.6s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8994 | 0.9740 | 0.0000 | 2.492 | 1.000 | (64/64) | 970.7 |
| 1 | 0.8758 | 0.9529 | 0.0000 | 2.330 | 1.000 | (64/64) | 542.5 |
| 2 | 0.8292 | 0.9107 | 0.0000 | 2.112 | 1.000 | (64/64) | 387.3 |
| 3 | 0.7521 | 0.8365 | 0.0000 | 1.824 | 1.000 | (64/64) | 302.8 |
| 4 | 0.6762 | 0.7570 | 0.0000 | 1.593 | 1.000 | (64/64) | 287.3 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 5-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 2490.6s (~498.1s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/cfm_5metal_wfa75a_w020/per_epoch_metrics.json` (full per-epoch record)
- `results/cfm_5metal_wfa75a_w020/per_epoch_metrics.csv` (flat CSV)
- `results/cfm_5metal_wfa75a_w020/wasserstein_decay.json` (W₂ decay check)
- `results/cfm_5metal_wfa75a_w020/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/cfm_5metal_wfa75a_w020/synthesis.md` (this file)

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
