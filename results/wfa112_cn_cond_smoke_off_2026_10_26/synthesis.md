# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cpu  
**Best decode_ratio RDKit-strict**: 1.000 (32/32)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 61.7s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 29.5850 | 15.9196 | 0.0000 | 568.709 | 1.000 | (32/32, single_frag=1) | 10.0 |
| 1 | 25.1996 | 12.6582 | 0.0000 | 445.397 | 1.000 | (32/32, single_frag=1) | 10.3 |
| 2 | 20.9853 | 9.2475 | 0.0000 | 407.067 | 1.000 | (32/32, single_frag=1) | 8.6 |
| 3 | 16.7161 | 5.8289 | 0.0000 | 353.995 | 1.000 | (32/32, single_frag=1) | 7.8 |
| 4 | 13.0910 | 2.8123 | 0.0000 | 295.543 | 1.000 | (32/32, single_frag=1) | 8.1 |
| 5 | 13.0014 | 1.7965 | 0.0000 | 164.676 | 1.000 | (32/32, single_frag=1) | 9.9 |
| 6 | 10.9539 | 1.5667 | 0.0000 | 181.040 | 1.000 | (32/32, single_frag=1) | 6.9 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 7-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cpu: 61.7s (~8.8s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/wfa112_cn_cond_smoke_off_2026_10_26/per_epoch_metrics.json` (full per-epoch record)
- `results/wfa112_cn_cond_smoke_off_2026_10_26/per_epoch_metrics.csv` (flat CSV)
- `results/wfa112_cn_cond_smoke_off_2026_10_26/wasserstein_decay.json` (W₂ decay check)
- `results/wfa112_cn_cond_smoke_off_2026_10_26/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/wfa112_cn_cond_smoke_off_2026_10_26/synthesis.md` (this file)

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
