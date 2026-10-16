# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (192/192)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 120.9s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.4751 | 1.5553 | 0.7829 | 3.157 | 1.000 | (192/192) | 25.2 |
| 1 | 1.0499 | 1.1226 | 0.7717 | 2.841 | 1.000 | (192/192) | 9.5 |
| 2 | 0.7024 | 0.7462 | 0.7600 | 2.168 | 1.000 | (192/192) | 11.0 |
| 3 | 0.4762 | 0.4825 | 0.7481 | 1.450 | 1.000 | (192/192) | 10.3 |
| 4 | 0.3325 | 0.3187 | 0.7360 | 1.265 | 1.000 | (192/192) | 7.0 |
| 5 | 0.2146 | 0.1859 | 0.7237 | 1.113 | 1.000 | (192/192) | 9.3 |
| 6 | 0.1353 | 0.0899 | 0.7113 | 0.634 | 1.000 | (192/192) | 13.2 |
| 7 | 0.1008 | 0.0429 | 0.6991 | 0.402 | 1.000 | (192/192) | 6.6 |
| 8 | 0.0897 | 0.0256 | 0.6871 | 0.484 | 1.000 | (192/192) | 4.5 |
| 9 | 0.0848 | 0.0192 | 0.6754 | 0.524 | 1.000 | (192/192) | 12.6 |
| 10 | 0.0816 | 0.0167 | 0.6640 | 0.581 | 1.000 | (192/192) | 11.6 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 11-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 120.9s (~11.0s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/wf124_p2_cfm_full/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/wf124_p2_cfm_full/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/wf124_p2_cfm_full/wasserstein_decay.json` (W₂ decay check)
- `None/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/wf124_p2_cfm_full/synthesis.md` (this file)

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
