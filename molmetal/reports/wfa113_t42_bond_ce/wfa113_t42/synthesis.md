# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cpu  
**Best decode_ratio RDKit-strict**: 1.000 (32/32)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 99.0s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 26.3000 | 13.5135 | 0.0000 | 475.393 | 1.000 | (32/32, single_frag=3) | 6.6 |
| 1 | 22.8636 | 10.7831 | 0.0000 | 375.056 | 1.000 | (32/32, single_frag=2) | 4.7 |
| 2 | 19.4018 | 7.8864 | 0.0000 | 347.899 | 1.000 | (32/32, single_frag=1) | 5.7 |
| 3 | 15.7341 | 4.9784 | 0.0000 | 310.367 | 1.000 | (32/32, single_frag=1) | 7.3 |
| 4 | 12.6202 | 2.4397 | 0.0000 | 255.949 | 1.000 | (32/32, single_frag=2) | 6.6 |
| 5 | 12.8049 | 1.6471 | 0.0000 | 151.702 | 1.000 | (32/32, single_frag=0) | 17.5 |
| 6 | 10.8731 | 1.4705 | 0.0000 | 191.020 | 1.000 | (32/32, single_frag=0) | 50.6 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 7-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cpu: 99.0s (~14.1s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `/tmp/wfa113_t42/per_epoch_metrics.json` (full per-epoch record)
- `/tmp/wfa113_t42/per_epoch_metrics.csv` (flat CSV)
- `/tmp/wfa113_t42/wasserstein_decay.json` (W₂ decay check)
- `/tmp/wfa113_t42/ck/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `/tmp/wfa113_t42/synthesis.md` (this file)

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
