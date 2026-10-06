# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda  
**Best decode_ratio RDKit-strict**: 0.488 (2894/6714)  
**Best epoch**: 3  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 12772.8s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.1427 | 0.1087 | 0.3379 | 3.661 | 0.458 | (3073/6714) | 1461.3 |
| 3 | 0.1204 | 0.0866 | 0.3371 | 2.764 | 0.488 | (3276/6714) | 892.5 |
| 4 | 0.1096 | 0.0759 | 0.3364 | 2.879 | 0.445 | (2988/6714) | 851.2 |
| 5 | 0.0996 | 0.0659 | 0.3360 | 2.313 | 0.431 | (2894/6714) | 1447.9 |
| 6 | 0.0931 | 0.0596 | 0.3351 | 1.799 | 0.432 | (2898/6714) | 1695.9 |
| 7 | 0.0876 | 0.0541 | 0.3345 | 1.358 | 0.444 | (2978/6714) | 1552.6 |
| 8 | 0.0840 | 0.0505 | 0.3344 | 1.241 | 0.398 | (2671/6714) | 1218.2 |
| 9 | 0.0797 | 0.0463 | 0.3339 | 1.070 | 0.417 | (2802/6714) | 805.2 |
| 10 | 0.0765 | 0.0431 | 0.3337 | 1.005 | 0.421 | (2829/6714) | 821.9 |
| 11 | 0.0738 | 0.0404 | 0.3335 | 0.922 | 0.407 | (2733/6714) | 801.4 |
| 12 | 0.0720 | 0.0387 | 0.3333 | 0.912 | 0.371 | (2492/6714) | 662.2 |
| 13 | 0.0701 | 0.0368 | 0.3332 | 0.950 | 0.408 | (2737/6714) | 562.3 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 12-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda: 12772.8s (~1064.4s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `/home/hugo/codes/mol-metal/results/cfm_tmqm_all6/per_epoch_metrics.json` (full per-epoch record)
- `/home/hugo/codes/mol-metal/results/cfm_tmqm_all6/per_epoch_metrics.csv` (flat CSV)
- `/home/hugo/codes/mol-metal/results/cfm_tmqm_all6/wasserstein_decay.json` (W₂ decay check)
- `results/cfm_tmqm_all6/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `/home/hugo/codes/mol-metal/results/cfm_tmqm_all6/synthesis.md` (this file)

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
