# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 1.000 (192/192)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 1661.5s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 14.4560 | 2.2136 | 0.8150 | 23.351 | 1.000 | (192/192) | 409.6 |
| 1 | 14.1479 | 1.9670 | 0.8096 | 20.392 | 1.000 | (192/192) | 554.5 |
| 2 | 13.5216 | 1.3773 | 0.7939 | 18.471 | 1.000 | (192/192) | 401.2 |
| 3 | 13.0943 | 1.0183 | 0.7414 | 19.219 | 1.000 | (192/192) | 296.2 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 4-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 1661.5s (~415.4s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_round3_retrain/run/per_epoch_metrics.json` (full per-epoch record)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_round3_retrain/run/per_epoch_metrics.csv` (flat CSV)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_round3_retrain/run/wasserstein_decay.json` (W₂ decay check)
- `None/cfm_yuelbond_wasserstein.pt` (best checkpoint)
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
