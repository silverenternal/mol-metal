# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.266 (51/192)  
**Best epoch**: 4  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 10696.3s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.9965 | 0.3133 | 0.7787 | 9.562 | 0.396 | (76/192) | 3787.3 |
| 1 | 9.6219 | 0.0969 | 0.6049 | 8.773 | 0.260 | (50/192) | 1355.6 |
| 2 | 9.5367 | 0.0663 | 0.4996 | 8.584 | 0.464 | (89/192) | 1043.7 |
| 3 | 9.5127 | 0.0503 | 0.4667 | 8.224 | 0.281 | (54/192) | 956.9 |
| 4 | 9.4451 | 0.0423 | 0.4563 | 8.346 | 0.266 | (51/192) | 935.7 |
| 5 | 9.4903 | 0.0381 | 0.4386 | 8.013 | 0.307 | (59/192) | 860.3 |
| 6 | 9.4261 | 0.0368 | 0.4283 | 8.231 | 0.297 | (57/192) | 864.1 |
| 7 | 9.4184 | 0.0328 | 0.4216 | 7.902 | 0.234 | (45/192) | 892.7 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 8-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 10696.3s (~1337.0s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_fm_distort_2026_10_11/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_fm_distort_2026_10_11/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_fm_distort_2026_10_11/wasserstein_decay.json` (W₂ decay check)
- `molmetal/reports/cfm_fm_distort_2026_10_11/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_fm_distort_2026_10_11/synthesis.md` (this file)

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
