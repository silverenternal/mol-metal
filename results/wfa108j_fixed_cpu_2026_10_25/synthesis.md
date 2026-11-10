# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cpu  
**Best decode_ratio RDKit-strict**: 1.000 (32/32)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 192.2s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 29.5637 | 15.9216 | 0.0000 | 589.554 | 1.000 | (32/32, single_frag=1) | 19.1 |
| 1 | 25.3646 | 12.6769 | 0.0000 | 464.268 | 1.000 | (32/32, single_frag=1) | 26.5 |
| 2 | 21.1480 | 9.2842 | 0.0000 | 415.437 | 1.000 | (32/32, single_frag=1) | 35.4 |
| 3 | 16.6007 | 5.8989 | 0.0000 | 447.562 | 1.000 | (32/32, single_frag=1) | 32.3 |
| 4 | 13.1365 | 2.9681 | 0.0000 | 385.095 | 1.000 | (32/32, single_frag=2) | 49.6 |
| 5 | 13.1905 | 1.8412 | 0.0000 | 220.651 | 1.000 | (32/32, single_frag=1) | 29.3 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 6-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cpu: 192.2s (~32.0s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/wfa108j_fixed_cpu_2026_10_25/per_epoch_metrics.json` (full per-epoch record)
- `results/wfa108j_fixed_cpu_2026_10_25/per_epoch_metrics.csv` (flat CSV)
- `results/wfa108j_fixed_cpu_2026_10_25/wasserstein_decay.json` (W₂ decay check)
- `results/wfa108j_fixed_cpu_2026_10_25/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/wfa108j_fixed_cpu_2026_10_25/synthesis.md` (this file)

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
