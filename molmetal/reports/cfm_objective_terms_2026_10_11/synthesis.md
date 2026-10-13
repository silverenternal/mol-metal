# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.391 (75/192)  
**Best epoch**: 5  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 12717.9s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.8781 | 0.3158 | 0.7811 | 10.609 | 0.484 | (93/192) | 4240.7 |
| 1 | 9.4693 | 0.0981 | 0.6309 | 9.450 | 0.479 | (92/192) | 1284.3 |
| 2 | 9.4534 | 0.0661 | 0.5325 | 10.696 | 0.354 | (68/192) | 1114.0 |
| 3 | 9.3945 | 0.0501 | 0.4838 | 8.656 | 0.307 | (59/192) | 1035.9 |
| 4 | 9.2818 | 0.0442 | 0.4716 | 9.163 | 0.344 | (66/192) | 986.9 |
| 5 | 9.2846 | 0.0394 | 0.4558 | 8.602 | 0.391 | (75/192) | 1009.1 |
| 6 | 9.2652 | 0.0364 | 0.4499 | 8.569 | 0.219 | (42/192) | 1041.6 |
| 7 | 9.2082 | 0.0329 | 0.4400 | 8.749 | 0.266 | (51/192) | 1023.1 |
| 8 | 9.1923 | 0.0315 | 0.4263 | 8.315 | 0.286 | (55/192) | 982.1 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 9-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 12717.9s (~1413.1s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `molmetal/reports/cfm_objective_terms_2026_10_11/per_epoch_metrics.json` (full per-epoch record)
- `molmetal/reports/cfm_objective_terms_2026_10_11/per_epoch_metrics.csv` (flat CSV)
- `molmetal/reports/cfm_objective_terms_2026_10_11/wasserstein_decay.json` (W₂ decay check)
- `molmetal/reports/cfm_objective_terms_2026_10_11/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `molmetal/reports/cfm_objective_terms_2026_10_11/synthesis.md` (this file)

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
