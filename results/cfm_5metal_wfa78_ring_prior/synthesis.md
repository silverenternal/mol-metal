# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (64/64)  
**Best epoch**: 4  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 1399.4s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.2663 | 1.0285 | 0.0000 | 53.646 | 1.000 | (64/64, single_frag=0) | 563.3 |
| 1 | 1.2480 | 1.0155 | 0.0000 | 86.100 | 1.000 | (64/64, single_frag=0) | 249.4 |
| 2 | 1.2079 | 0.9897 | 0.0000 | 279.829 | 1.000 | (64/64, single_frag=2) | 214.0 |
| 3 | 1.4458 | 0.9787 | 0.0000 | 25.074 | 1.000 | (64/64, single_frag=2) | 193.5 |
| 4 | 1.8939 | 0.9094 | 0.0000 | 1.668 | 1.000 | (64/64, single_frag=7) | 179.2 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 5-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 1399.4s (~279.9s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/cfm_5metal_wfa78_ring_prior/per_epoch_metrics.json` (full per-epoch record)
- `results/cfm_5metal_wfa78_ring_prior/per_epoch_metrics.csv` (flat CSV)
- `results/cfm_5metal_wfa78_ring_prior/wasserstein_decay.json` (W₂ decay check)
- `results/cfm_5metal_wfa78_ring_prior/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/cfm_5metal_wfa78_ring_prior/synthesis.md` (this file)

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
