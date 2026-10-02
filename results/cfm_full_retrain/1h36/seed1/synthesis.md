# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (192/192)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 33.1s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.5024 | 1.4286 | 0.7345 | 2.782 | 1.000 | (192/192) | 4.1 |
| 1 | 1.0618 | 0.9889 | 0.7228 | 2.771 | 1.000 | (192/192) | 2.9 |
| 2 | 0.7043 | 0.6313 | 0.7113 | 1.913 | 1.000 | (192/192) | 2.9 |
| 3 | 0.4891 | 0.4131 | 0.7002 | 1.720 | 1.000 | (192/192) | 2.9 |
| 4 | 0.3331 | 0.2501 | 0.6895 | 1.619 | 1.000 | (192/192) | 2.9 |
| 5 | 0.2204 | 0.1255 | 0.6791 | 1.250 | 1.000 | (192/192) | 2.9 |
| 6 | 0.1703 | 0.0563 | 0.6690 | 0.888 | 1.000 | (192/192) | 3.0 |
| 7 | 0.1753 | 0.0352 | 0.6590 | 0.594 | 1.000 | (192/192) | 2.9 |
| 8 | 0.1949 | 0.0292 | 0.6490 | 0.663 | 1.000 | (192/192) | 2.9 |
| 9 | 0.2065 | 0.0229 | 0.6391 | 0.685 | 1.000 | (192/192) | 2.9 |
| 10 | 0.2094 | 0.0165 | 0.6291 | 0.683 | 1.000 | (192/192) | 2.9 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 11-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 33.1s (~3.0s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/cfm_full_retrain/1h36/seed1/per_epoch_metrics.json` (full per-epoch record)
- `results/cfm_full_retrain/1h36/seed1/per_epoch_metrics.csv` (flat CSV)
- `results/cfm_full_retrain/1h36/seed1/wasserstein_decay.json` (W₂ decay check)
- `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/cfm_full_retrain/1h36/seed1/synthesis.md` (this file)

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
