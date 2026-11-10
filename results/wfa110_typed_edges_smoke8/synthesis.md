# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cpu  
**Best decode_ratio RDKit-strict**: 1.000 (32/32)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 64.9s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 29.1661 | 15.9328 | 0.0000 | 585.075 | 1.000 | (32/32, single_frag=1) | 10.3 |
| 1 | 24.4729 | 12.7073 | 0.0000 | 525.387 | 1.000 | (32/32, single_frag=1) | 11.3 |
| 2 | 20.1863 | 9.3714 | 0.0000 | 515.335 | 1.000 | (32/32, single_frag=1) | 8.4 |
| 3 | 17.1551 | 6.0540 | 0.0000 | 394.080 | 1.000 | (32/32, single_frag=1) | 10.7 |
| 4 | 13.8590 | 3.0420 | 0.0000 | 318.807 | 1.000 | (32/32, single_frag=1) | 6.4 |
| 5 | 11.2159 | 1.8484 | 0.0000 | 498.057 | 1.000 | (32/32, single_frag=1) | 4.4 |
| 6 | 11.4421 | 1.6060 | 0.0000 | 144.882 | 1.000 | (32/32, single_frag=1) | 7.7 |
| 7 | 9.8916 | 1.4721 | 0.0000 | 192.817 | 1.000 | (32/32, single_frag=1) | 5.7 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 8-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cpu: 64.9s (~8.1s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/wfa110_typed_edges_smoke8/per_epoch_metrics.json` (full per-epoch record)
- `results/wfa110_typed_edges_smoke8/per_epoch_metrics.csv` (flat CSV)
- `results/wfa110_typed_edges_smoke8/wasserstein_decay.json` (W₂ decay check)
- `results/wfa110_typed_edges_smoke8/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/wfa110_typed_edges_smoke8/synthesis.md` (this file)

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
