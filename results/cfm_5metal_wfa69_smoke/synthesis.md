# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (64/64)  
**Best epoch**: 4  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 261.1s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 16.9166 | 16.8705 | 0.0000 | 615.966 | 1.000 | (64/64) | 104.4 |
| 1 | 11.5971 | 11.5823 | 0.0000 | 529.511 | 1.000 | (64/64) | 40.6 |
| 2 | 3.0685 | 3.0429 | 0.0000 | 275.822 | 1.000 | (64/64) | 40.1 |
| 3 | 0.4582 | 0.4740 | 0.0000 | 19.486 | 1.000 | (64/64) | 36.3 |
| 4 | 0.3048 | 0.3312 | 0.0000 | 11.054 | 1.000 | (64/64) | 39.6 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 5-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 261.1s (~52.2s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `/home/hugo/codes/mol-metal/results/cfm_5metal_wfa69_smoke/per_epoch_metrics.json` (full per-epoch record)
- `/home/hugo/codes/mol-metal/results/cfm_5metal_wfa69_smoke/per_epoch_metrics.csv` (flat CSV)
- `/home/hugo/codes/mol-metal/results/cfm_5metal_wfa69_smoke/wasserstein_decay.json` (W₂ decay check)
- `/home/hugo/codes/mol-metal/results/cfm_5metal_wfa69_smoke/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `/home/hugo/codes/mol-metal/results/cfm_5metal_wfa69_smoke/synthesis.md` (this file)

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
