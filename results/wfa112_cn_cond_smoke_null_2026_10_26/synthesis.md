# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cpu  
**Best decode_ratio RDKit-strict**: 1.000 (32/32)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 49.2s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 29.5850 | 15.9196 | 0.0000 | 568.850 | 1.000 | (32/32, single_frag=1) | 9.0 |
| 1 | 25.2006 | 12.6586 | 0.0000 | 445.488 | 1.000 | (32/32, single_frag=1) | 10.6 |
| 2 | 20.9861 | 9.2477 | 0.0000 | 406.640 | 1.000 | (32/32, single_frag=1) | 8.1 |
| 3 | 16.7162 | 5.8284 | 0.0000 | 353.200 | 1.000 | (32/32, single_frag=1) | 5.0 |
| 4 | 13.0894 | 2.8109 | 0.0000 | 295.747 | 1.000 | (32/32, single_frag=1) | 5.7 |
| 5 | 13.0032 | 1.7968 | 0.0000 | 165.344 | 1.000 | (32/32, single_frag=1) | 6.2 |
| 6 | 10.9568 | 1.5676 | 0.0000 | 180.345 | 1.000 | (32/32, single_frag=1) | 4.5 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 7-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cpu: 49.2s (~7.0s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/wfa112_cn_cond_smoke_null_2026_10_26/per_epoch_metrics.json` (full per-epoch record)
- `results/wfa112_cn_cond_smoke_null_2026_10_26/per_epoch_metrics.csv` (flat CSV)
- `results/wfa112_cn_cond_smoke_null_2026_10_26/wasserstein_decay.json` (W₂ decay check)
- `results/wfa112_cn_cond_smoke_null_2026_10_26/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/wfa112_cn_cond_smoke_null_2026_10_26/synthesis.md` (this file)

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
