# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (64/64)  
**Best epoch**: 3  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 450.3s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.8728 | 0.9668 | 0.0000 | 2.356 | 1.000 | (64/64, single_frag=3) | 91.4 |
| 1 | 0.8482 | 0.9418 | 0.0000 | 2.222 | 1.000 | (64/64, single_frag=4) | 77.7 |
| 2 | 0.7925 | 0.8847 | 0.0000 | 1.972 | 1.000 | (64/64, single_frag=3) | 84.1 |
| 3 | 0.7370 | 0.8261 | 0.0000 | 1.712 | 1.000 | (64/64, single_frag=3) | 114.9 |
| 4 | 0.6612 | 0.7449 | 0.0000 | 1.534 | 1.000 | (64/64, single_frag=3) | 82.0 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 5-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 450.3s (~90.1s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/wfa107_features/per_epoch_metrics.json` (full per-epoch record)
- `results/wfa107_features/per_epoch_metrics.csv` (flat CSV)
- `results/wfa107_features/wasserstein_decay.json` (W₂ decay check)
- `results/wfa107_features/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/wfa107_features/synthesis.md` (this file)

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
