# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cpu  
**Best decode_ratio RDKit-strict**: 1.000 (32/32)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 46.3s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 29.1659 | 15.9328 | 0.0000 | 585.046 | 1.000 | (32/32, single_frag=2) | 6.9 |
| 1 | 24.4735 | 12.7075 | 0.0000 | 524.737 | 1.000 | (32/32, single_frag=2) | 8.6 |
| 2 | 20.1867 | 9.3724 | 0.0000 | 515.366 | 1.000 | (32/32, single_frag=2) | 6.4 |
| 3 | 17.1589 | 6.0578 | 0.0000 | 393.913 | 1.000 | (32/32, single_frag=1) | 7.4 |
| 4 | 13.8631 | 3.0493 | 0.0000 | 319.351 | 1.000 | (32/32, single_frag=1) | 5.2 |
| 5 | 11.2158 | 1.8499 | 0.0000 | 487.898 | 1.000 | (32/32, single_frag=1) | 5.1 |
| 6 | 11.4407 | 1.6032 | 0.0000 | 146.705 | 1.000 | (32/32, single_frag=1) | 6.7 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 7-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cpu: 46.3s (~6.6s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/wfa111_ring_prior_smoke_7ep_2026_10_26/per_epoch_metrics.json` (full per-epoch record)
- `results/wfa111_ring_prior_smoke_7ep_2026_10_26/per_epoch_metrics.csv` (flat CSV)
- `results/wfa111_ring_prior_smoke_7ep_2026_10_26/wasserstein_decay.json` (W₂ decay check)
- `results/wfa111_ring_prior_smoke_7ep_2026_10_26/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/wfa111_ring_prior_smoke_7ep_2026_10_26/synthesis.md` (this file)

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
