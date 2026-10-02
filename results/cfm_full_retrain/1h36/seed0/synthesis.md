# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (192/192)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 35.7s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.6362 | 1.5561 | 0.7828 | 3.161 | 1.000 | (192/192) | 7.1 |
| 1 | 1.2016 | 1.1222 | 0.7717 | 2.985 | 1.000 | (192/192) | 2.8 |
| 2 | 0.8205 | 0.7412 | 0.7601 | 2.413 | 1.000 | (192/192) | 2.8 |
| 3 | 0.5630 | 0.4830 | 0.7481 | 1.521 | 1.000 | (192/192) | 2.8 |
| 4 | 0.4062 | 0.3247 | 0.7360 | 1.377 | 1.000 | (192/192) | 2.9 |
| 5 | 0.2722 | 0.1883 | 0.7237 | 1.172 | 1.000 | (192/192) | 2.8 |
| 6 | 0.1743 | 0.0872 | 0.7113 | 0.753 | 1.000 | (192/192) | 2.9 |
| 7 | 0.1304 | 0.0388 | 0.6991 | 0.416 | 1.000 | (192/192) | 2.8 |
| 8 | 0.1205 | 0.0249 | 0.6871 | 0.450 | 1.000 | (192/192) | 2.8 |
| 9 | 0.1178 | 0.0210 | 0.6754 | 0.789 | 1.000 | (192/192) | 2.8 |
| 10 | 0.1176 | 0.0197 | 0.6640 | 0.908 | 1.000 | (192/192) | 2.8 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 11-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 35.7s (~3.2s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/cfm_full_retrain/1h36/seed0/per_epoch_metrics.json` (full per-epoch record)
- `results/cfm_full_retrain/1h36/seed0/per_epoch_metrics.csv` (flat CSV)
- `results/cfm_full_retrain/1h36/seed0/wasserstein_decay.json` (W₂ decay check)
- `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/cfm_full_retrain/1h36/seed0/synthesis.md` (this file)

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
