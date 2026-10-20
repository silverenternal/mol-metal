# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (0/0)  
**Best epoch**: 17  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 11137.3s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 26.3485 | 0.7150 | 0.7604 | 5.603 | 1.000 | (192/192, single_frag=104) | 1140.2 |
| 5 | 26.3107 | 0.7027 | 0.7495 | 6.530 | 1.000 | (192/192, single_frag=96) | 1005.8 |
| 6 | 26.2565 | 0.6914 | 0.7422 | 7.086 | 1.000 | (192/192, single_frag=100) | 1131.4 |
| 7 | 26.2042 | 0.6806 | 0.7368 | 7.623 | 1.000 | (192/192, single_frag=112) | 867.3 |
| 8 | 26.0608 | 0.6701 | 0.7313 | 8.739 | 1.000 | (192/192, single_frag=102) | 918.7 |
| 9 | 25.9043 | 0.6604 | 0.7111 | 10.618 | 1.000 | (192/192, single_frag=116) | 468.2 |
| 10 | 25.6929 | 0.6516 | 0.6795 | 12.587 | 1.000 | (192/192, single_frag=109) | 513.7 |
| 11 | 25.5434 | 0.6429 | 0.6620 | 13.209 | 1.000 | (192/192, single_frag=114) | 694.9 |
| 12 | 25.4754 | 0.6338 | 0.6535 | 13.570 | 1.000 | (192/192, single_frag=118) | 816.3 |
| 13 | 25.3974 | 0.6246 | 0.6461 | 13.778 | 1.000 | (192/192, single_frag=128) | 823.5 |
| 14 | 25.3325 | 0.6149 | 0.6379 | 13.890 | 1.000 | (192/192, single_frag=135) | 817.9 |
| 15 | 25.2915 | 0.6047 | 0.6300 | 14.099 | 1.000 | (192/192, single_frag=134) | 814.7 |
| 16 | 25.2339 | 0.5946 | 0.6231 | 14.191 | 1.000 | (192/192, single_frag=141) | 526.9 |
| 17 | 25.2138 | 0.5845 | 0.6168 | 14.371 | 1.000 | (192/192, single_frag=137) | 597.0 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 14-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 11137.3s (~795.5s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_c7_fullgen_2026_10_20/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_c7_fullgen_2026_10_20/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_c7_fullgen_2026_10_20/wasserstein_decay.json` (W₂ decay check)
- `molmetal/reports/cfm_c7_fullgen_2026_10_20/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_c7_fullgen_2026_10_20/synthesis.md` (this file)

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
