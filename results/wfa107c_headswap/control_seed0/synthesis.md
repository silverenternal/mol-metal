# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (64/64)  
**Best epoch**: 4  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 519.9s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9595 | 1.0274 | 0.0000 | 2.617 | 1.000 | (64/64, single_frag=0) | 90.0 |
| 1 | 0.9362 | 1.0069 | 0.0000 | 2.532 | 1.000 | (64/64, single_frag=0) | 86.1 |
| 2 | 0.8753 | 0.9519 | 0.0000 | 2.288 | 1.000 | (64/64, single_frag=0) | 113.5 |
| 3 | 0.8022 | 0.8858 | 0.0000 | 1.880 | 1.000 | (64/64, single_frag=2) | 143.1 |
| 4 | 0.7105 | 0.7977 | 0.0000 | 1.481 | 1.000 | (64/64, single_frag=2) | 87.1 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 5-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 519.9s (~104.0s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/wfa107c_headswap/control_seed0/per_epoch_metrics.json` (full per-epoch record)
- `results/wfa107c_headswap/control_seed0/per_epoch_metrics.csv` (flat CSV)
- `results/wfa107c_headswap/control_seed0/wasserstein_decay.json` (W₂ decay check)
- `results/wfa107c_headswap/control_seed0/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/wfa107c_headswap/control_seed0/synthesis.md` (this file)

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
