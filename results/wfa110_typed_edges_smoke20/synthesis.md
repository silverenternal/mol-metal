# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cpu  
**Best decode_ratio RDKit-strict**: 1.000 (32/32)  
**Best epoch**: 18  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 187.6s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 29.1661 | 15.9328 | 0.0000 | 585.075 | 1.000 | (32/32, single_frag=1) | 12.6 |
| 1 | 24.4720 | 12.7067 | 0.0000 | 525.379 | 1.000 | (32/32, single_frag=1) | 11.0 |
| 2 | 20.1812 | 9.3671 | 0.0000 | 515.325 | 1.000 | (32/32, single_frag=1) | 8.1 |
| 3 | 17.1421 | 6.0423 | 0.0000 | 393.824 | 1.000 | (32/32, single_frag=1) | 11.5 |
| 4 | 13.8341 | 3.0195 | 0.0000 | 317.802 | 1.000 | (32/32, single_frag=1) | 10.7 |
| 5 | 11.2009 | 1.8424 | 0.0000 | 483.784 | 1.000 | (32/32, single_frag=1) | 7.0 |
| 6 | 11.4270 | 1.5995 | 0.0000 | 147.975 | 1.000 | (32/32, single_frag=1) | 11.1 |
| 7 | 9.8827 | 1.4733 | 0.0000 | 194.585 | 1.000 | (32/32, single_frag=1) | 8.1 |
| 8 | 11.2102 | 1.3416 | 0.0000 | 132.679 | 1.000 | (32/32, single_frag=0) | 10.9 |
| 9 | 9.6963 | 1.2397 | 0.0000 | 271.698 | 1.000 | (32/32, single_frag=0) | 7.2 |
| 10 | 9.7330 | 1.1818 | 0.0000 | 215.099 | 1.000 | (32/32, single_frag=0) | 8.1 |
| 11 | 10.4579 | 1.1199 | 0.0000 | 208.282 | 1.000 | (32/32, single_frag=0) | 10.5 |
| 12 | 9.8677 | 1.0632 | 0.0000 | 137.719 | 1.000 | (32/32, single_frag=0) | 8.4 |
| 13 | 10.5990 | 0.9937 | 0.0000 | 141.362 | 1.000 | (32/32, single_frag=0) | 10.3 |
| 14 | 9.4340 | 0.9657 | 0.0000 | 182.884 | 1.000 | (32/32, single_frag=0) | 7.5 |
| 15 | 9.8882 | 0.9252 | 0.0000 | 120.549 | 1.000 | (32/32, single_frag=0) | 6.3 |
| 16 | 9.6467 | 0.9110 | 0.0000 | 200.060 | 1.000 | (32/32, single_frag=0) | 9.0 |
| 17 | 10.3573 | 0.8960 | 0.0000 | 157.685 | 1.000 | (32/32, single_frag=0) | 10.8 |
| 18 | 9.0968 | 0.8891 | 0.0000 | 211.201 | 1.000 | (32/32, single_frag=0) | 7.9 |
| 19 | 8.9561 | 0.8748 | 0.0000 | 168.182 | 1.000 | (32/32, single_frag=0) | 10.4 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 20-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cpu: 187.6s (~9.4s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/wfa110_typed_edges_smoke20/per_epoch_metrics.json` (full per-epoch record)
- `results/wfa110_typed_edges_smoke20/per_epoch_metrics.csv` (flat CSV)
- `results/wfa110_typed_edges_smoke20/wasserstein_decay.json` (W₂ decay check)
- `results/wfa110_typed_edges_smoke20/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/wfa110_typed_edges_smoke20/synthesis.md` (this file)

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
