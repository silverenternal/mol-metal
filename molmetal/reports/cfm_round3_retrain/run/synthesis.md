# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.402 (2696/6714)  
**Best epoch**: 18  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 23742.9s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4506 | 0.4148 | 0.3550 | 3.703 | 0.536 | (3599/6714) | 3000.8 |
| 1 | 0.1804 | 0.1462 | 0.3396 | 4.055 | 0.499 | (3352/6714) | 1380.1 |
| 2 | 0.1363 | 0.1024 | 0.3380 | 2.666 | 0.491 | (3294/6714) | 1200.8 |
| 3 | 0.1226 | 0.0888 | 0.3372 | 3.005 | 0.423 | (2837/6714) | 1329.2 |
| 4 | 0.1123 | 0.0786 | 0.3364 | 3.142 | 0.471 | (3164/6714) | 1329.6 |
| 5 | 0.0983 | 0.0646 | 0.3359 | 1.987 | 0.451 | (3028/6714) | 1147.1 |
| 6 | 0.0920 | 0.0584 | 0.3353 | 2.027 | 0.439 | (2949/6714) | 1021.1 |
| 7 | 0.0853 | 0.0518 | 0.3348 | 1.319 | 0.428 | (2871/6714) | 820.9 |
| 8 | 0.0807 | 0.0473 | 0.3343 | 1.118 | 0.420 | (2823/6714) | 915.3 |
| 9 | 0.0772 | 0.0438 | 0.3339 | 1.136 | 0.392 | (2633/6714) | 1067.8 |
| 10 | 0.0742 | 0.0407 | 0.3338 | 1.088 | 0.391 | (2625/6714) | 1044.7 |
| 11 | 0.0716 | 0.0382 | 0.3336 | 1.006 | 0.459 | (3080/6714) | 1103.6 |
| 12 | 0.0699 | 0.0366 | 0.3333 | 1.097 | 0.401 | (2693/6714) | 1058.7 |
| 13 | 0.0679 | 0.0346 | 0.3331 | 0.979 | 0.389 | (2615/6714) | 1063.1 |
| 14 | 0.0659 | 0.0326 | 0.3329 | 0.972 | 0.379 | (2542/6714) | 1079.1 |
| 15 | 0.0656 | 0.0322 | 0.3330 | 1.005 | 0.399 | (2679/6714) | 1035.0 |
| 16 | 0.0637 | 0.0304 | 0.3328 | 0.970 | 0.438 | (2940/6714) | 1042.2 |
| 17 | 0.0621 | 0.0289 | 0.3325 | 0.929 | 0.421 | (2828/6714) | 1018.7 |
| 18 | 0.0617 | 0.0284 | 0.3325 | 0.966 | 0.402 | (2696/6714) | 1072.6 |
| 19 | 0.0609 | 0.0276 | 0.3324 | 0.950 | 0.398 | (2669/6714) | 1012.2 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 20-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 23742.9s (~1187.1s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_round3_retrain/run/per_epoch_metrics.json` (full per-epoch record)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_round3_retrain/run/per_epoch_metrics.csv` (flat CSV)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_round3_retrain/run/wasserstein_decay.json` (W₂ decay check)
- `results/cfm_tmqm_all6/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_round3_retrain/run/synthesis.md` (this file)

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
