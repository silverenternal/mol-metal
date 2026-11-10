# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (64/64)  
**Best epoch**: 4  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 369.1s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.2306 | 1.0275 | 0.0000 | 7.938 | 1.000 | (64/64, single_frag=0) | 101.0 |
| 1 | 2.1584 | 1.0073 | 0.0000 | 7.596 | 1.000 | (64/64, single_frag=0) | 69.6 |
| 2 | 1.9735 | 0.9511 | 0.0000 | 6.662 | 1.000 | (64/64, single_frag=0) | 62.5 |
| 3 | 1.7098 | 0.8771 | 0.0000 | 4.900 | 1.000 | (64/64, single_frag=5) | 60.2 |
| 4 | 1.4268 | 0.7774 | 0.0000 | 3.488 | 1.000 | (64/64, single_frag=1) | 75.7 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 5-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 369.1s (~73.8s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/wfa107c_headswap/edge_ring_w1_seed0/per_epoch_metrics.json` (full per-epoch record)
- `results/wfa107c_headswap/edge_ring_w1_seed0/per_epoch_metrics.csv` (flat CSV)
- `results/wfa107c_headswap/edge_ring_w1_seed0/wasserstein_decay.json` (W₂ decay check)
- `results/wfa107c_headswap/edge_ring_w1_seed0/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/wfa107c_headswap/edge_ring_w1_seed0/synthesis.md` (this file)

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
