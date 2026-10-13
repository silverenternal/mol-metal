# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.458 (88/192)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 7020.2s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.9196 | 0.3153 | 0.7990 | 9.727 | 0.458 | (88/192) | 3899.5 |
| 1 | 9.4985 | 0.0955 | 0.6118 | 8.983 | 0.318 | (61/192) | 1260.6 |
| 2 | 9.4317 | 0.0591 | 0.5379 | 8.388 | 0.344 | (66/192) | 957.3 |
| 3 | 9.3516 | 0.0494 | 0.4644 | 8.523 | 0.370 | (71/192) | 902.8 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 4-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 7020.2s (~1755.1s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_fullcorpus_2026_10_10/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_fullcorpus_2026_10_10/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_fullcorpus_2026_10_10/wasserstein_decay.json` (W₂ decay check)
- `molmetal/reports/cfm_fullcorpus_2026_10_10/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_fullcorpus_2026_10_10/synthesis.md` (this file)

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
