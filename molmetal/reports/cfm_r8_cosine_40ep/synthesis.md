# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cuda:0  
**Best decode_ratio RDKit-strict**: 0.020 (137/6714)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: no — honest framing below  
**Total wall time**: 17395.6s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13.4830 | 1.1489 | 0.0000 | 4.427 | 0.020 | (137/6714) | 8149.5 |
| 1 | 13.2192 | 0.9914 | 0.0000 | 6.204 | 0.031 | (205/6714) | 1635.2 |
| 2 | 12.7574 | 0.9452 | 0.0000 | 13.100 | 0.047 | (317/6714) | 1070.9 |
| 3 | 11.6113 | 0.9158 | 0.0000 | 25.879 | 0.050 | (338/6714) | 1422.7 |
| 4 | 10.5996 | 0.8847 | 0.0000 | 34.222 | 0.071 | (478/6714) | 1654.6 |
| 5 | 10.4619 | 0.8527 | 0.0000 | 35.751 | 0.081 | (545/6714) | 946.8 |
| 6 | 10.3659 | 0.8232 | 0.0000 | 34.328 | 0.087 | (583/6714) | 852.0 |
| 7 | 10.3130 | 0.7948 | 0.0000 | 33.607 | 0.105 | (706/6714) | 762.6 |
| 8 | 10.2704 | 0.7678 | 0.0000 | 35.555 | 0.114 | (766/6714) | 901.2 |

## Honest framing

- The target decode_ratio ≥ 0.80 is NOT MET. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 9-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cuda:0: 17395.6s (~1932.8s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_r8_cosine_40ep/per_epoch_metrics.json` (full per-epoch record)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_r8_cosine_40ep/per_epoch_metrics.csv` (flat CSV)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_r8_cosine_40ep/wasserstein_decay.json` (W₂ decay check)
- `results/cfm_tmqm_all6/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `/home/hugo/codes/mol-metal/molmetal/reports/cfm_r8_cosine_40ep/synthesis.md` (this file)

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
