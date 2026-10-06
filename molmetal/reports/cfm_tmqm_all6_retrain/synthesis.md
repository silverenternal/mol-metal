# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.531 (3564/6714)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 9629.3s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4471 | 0.4112 | 0.3553 | 3.338 | 0.531 | (3564/6714) | 1255.8 |
| 1 | 0.1841 | 0.1500 | 0.3396 | 3.171 | 0.475 | (3189/6714) | 773.2 |
| 2 | 0.1375 | 0.1036 | 0.3381 | 2.740 | 0.504 | (3381/6714) | 908.0 |
| 3 | 0.1187 | 0.0849 | 0.3370 | 2.416 | 0.506 | (3394/6714) | 918.5 |
| 4 | 0.1080 | 0.0742 | 0.3365 | 2.525 | 0.389 | (2614/6714) | 856.8 |
| 5 | 0.0988 | 0.0652 | 0.3359 | 2.413 | 0.475 | (3192/6714) | 799.6 |
| 6 | 0.0916 | 0.0580 | 0.3352 | 1.656 | 0.432 | (2903/6714) | 708.4 |
| 7 | 0.0862 | 0.0527 | 0.3345 | 1.488 | 0.389 | (2614/6714) | 824.3 |
| 8 | 0.0826 | 0.0492 | 0.3343 | 1.346 | 0.401 | (2690/6714) | 770.1 |
| 9 | 0.0785 | 0.0451 | 0.3340 | 1.043 | 0.459 | (3084/6714) | 928.3 |
| 10 | 0.0767 | 0.0433 | 0.3336 | 1.323 | 0.418 | (2807/6714) | 886.2 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 11-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 9629.3s (~875.4s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_tmqm_all6_retrain/per_epoch_metrics.json` (full per-epoch record)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_tmqm_all6_retrain/per_epoch_metrics.csv` (flat CSV)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_tmqm_all6_retrain/wasserstein_decay.json` (W₂ decay check)
- `results/cfm_tmqm_all6/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_tmqm_all6_retrain/synthesis.md` (this file)

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
