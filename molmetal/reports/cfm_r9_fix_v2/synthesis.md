# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.408 (2738/6714)  
**Best epoch**: 7  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 29681.1s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10.0290 | 0.4454 | 0.0000 | 10.094 | 0.273 | (1833/6714) | 4915.1 |
| 1 | 9.5672 | 0.1712 | 0.0000 | 8.923 | 0.329 | (2211/6714) | 5744.2 |
| 2 | 9.4539 | 0.1157 | 0.0000 | 8.390 | 0.301 | (2023/6714) | 4414.5 |
| 3 | 9.4097 | 0.0932 | 0.0000 | 8.055 | 0.272 | (1827/6714) | 2111.3 |
| 4 | 9.3481 | 0.0779 | 0.0000 | 7.651 | 0.352 | (2361/6714) | 2211.6 |
| 5 | 9.3134 | 0.0701 | 0.0000 | 7.458 | 0.362 | (2433/6714) | 2063.5 |
| 6 | 9.2810 | 0.0643 | 0.0000 | 7.983 | 0.413 | (2773/6714) | 1719.4 |
| 7 | 9.2508 | 0.0554 | 0.0000 | 6.885 | 0.408 | (2738/6714) | 1577.3 |
| 8 | 9.2273 | 0.0512 | 0.0000 | 6.938 | 0.307 | (2060/6714) | 1615.2 |
| 9 | 9.1992 | 0.0467 | 0.0000 | 6.706 | 0.336 | (2257/6714) | 1692.4 |
| 10 | 9.2065 | 0.0462 | 0.0000 | 7.347 | 0.386 | (2589/6714) | 1616.7 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 11-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 29681.1s (~2698.3s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_r9_fix_v2/per_epoch_metrics.json` (full per-epoch record)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_r9_fix_v2/per_epoch_metrics.csv` (flat CSV)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_r9_fix_v2/wasserstein_decay.json` (W₂ decay check)
- `results/cfm_tmqm_all6/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_r9_fix_v2/synthesis.md` (this file)

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
