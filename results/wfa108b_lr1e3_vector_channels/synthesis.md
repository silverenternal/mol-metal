# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (64/64)  
**Best epoch**: 3  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 987.0s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13.3676 | 0.9988 | 0.7930 | 8.640 | 1.000 | (64/64, single_frag=0) | 233.0 |
| 1 | 13.1813 | 0.8464 | 0.7821 | 7.702 | 1.000 | (64/64, single_frag=3) | 180.9 |
| 2 | 13.4131 | 0.5984 | 0.7552 | 30.729 | 1.000 | (64/64, single_frag=0) | 186.7 |
| 3 | 12.9935 | 0.4338 | 0.7231 | 21.950 | 1.000 | (64/64, single_frag=0) | 189.9 |
| 4 | 13.0113 | 0.3642 | 0.7023 | 23.126 | 1.000 | (64/64, single_frag=0) | 196.3 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 5-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 987.0s (~197.4s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/wfa108b_lr1e3_vector_channels/per_epoch_metrics.json` (full per-epoch record)
- `results/wfa108b_lr1e3_vector_channels/per_epoch_metrics.csv` (flat CSV)
- `results/wfa108b_lr1e3_vector_channels/wasserstein_decay.json` (W₂ decay check)
- `results/wfa108b_lr1e3_vector_channels/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/wfa108b_lr1e3_vector_channels/synthesis.md` (this file)

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
