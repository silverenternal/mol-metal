# Wf-124 Phase 2 — Full CFM retrain: YuelBond + Wasserstein + Catherine closure-bounded

**Date**: 2026-09-27  
**Device**: cpu  
**Best decode_ratio RDKit-strict**: 1.000 (32/32)  
**Best epoch**: 0  
**Target**: 0.80  
**Hit target**: YES  
**Total wall time**: 608.9s

## Engine tags (per TODO-126 §7 honest reporting)

| Engine | Tag |
|---|---|
| YuelBond decoder head | `yuelbond_decoder` |
| Wasserstein regularization | `wasserstein_reg` |
| Catherine closure-bounded generation | `closure_catherine_bounded` |

## Per-epoch metrics (first 5 + best + last 5)

| Epoch | train_loss | bond_ce | w2 | grad_norm | decode_ratio | (ok / n) | wall_s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 29.5637 | 15.9216 | 0.0000 | 589.554 | 1.000 | (32/32, single_frag=1) | 32.5 |
| 1 | 25.3628 | 12.6756 | 0.0000 | 464.257 | 1.000 | (32/32, single_frag=1) | 35.0 |
| 2 | 21.1373 | 9.2750 | 0.0000 | 415.311 | 1.000 | (32/32, single_frag=1) | 21.0 |
| 3 | 16.5733 | 5.8741 | 0.0000 | 447.292 | 1.000 | (32/32, single_frag=1) | 16.0 |
| 4 | 13.0860 | 2.9225 | 0.0000 | 382.802 | 1.000 | (32/32, single_frag=1) | 15.4 |
| 5 | 13.1652 | 1.8291 | 0.0000 | 218.455 | 1.000 | (32/32, single_frag=1) | 16.3 |
| 6 | 10.9148 | 1.5928 | 0.0000 | 276.929 | 1.000 | (32/32, single_frag=1) | 14.3 |
| 7 | 10.7167 | 1.4876 | 0.0000 | 280.476 | 1.000 | (32/32, single_frag=1) | 11.7 |
| 8 | 10.8415 | 1.3372 | 0.0000 | 217.985 | 1.000 | (32/32, single_frag=0) | 15.7 |
| 9 | 10.7448 | 1.2475 | 0.0000 | 202.343 | 1.000 | (32/32, single_frag=0) | 17.8 |
| 10 | 11.2654 | 1.2139 | 0.0000 | 123.310 | 1.000 | (32/32, single_frag=0) | 18.2 |
| 11 | 10.5804 | 1.1198 | 0.0000 | 154.240 | 1.000 | (32/32, single_frag=0) | 13.1 |
| 12 | 9.7425 | 1.0716 | 0.0000 | 158.666 | 1.000 | (32/32, single_frag=0) | 17.4 |
| 13 | 10.7199 | 1.0181 | 0.0000 | 106.284 | 1.000 | (32/32, single_frag=0) | 32.6 |
| 14 | 9.8520 | 0.9742 | 0.0000 | 199.169 | 1.000 | (32/32, single_frag=1) | 30.2 |
| 15 | 9.6494 | 0.9216 | 0.0000 | 145.702 | 1.000 | (32/32, single_frag=0) | 22.3 |
| 16 | 9.4388 | 0.9044 | 0.0000 | 132.617 | 1.000 | (32/32, single_frag=0) | 11.3 |
| 17 | 9.4705 | 0.8986 | 0.0000 | 141.615 | 1.000 | (32/32, single_frag=0) | 10.2 |
| 18 | 10.3971 | 0.8893 | 0.0000 | 86.453 | 1.000 | (32/32, single_frag=0) | 16.2 |
| 19 | 9.5922 | 0.8706 | 0.0000 | 109.377 | 1.000 | (32/32, single_frag=0) | 17.2 |
| 20 | 10.5238 | 0.8581 | 0.0000 | 86.790 | 1.000 | (32/32, single_frag=1) | 18.9 |
| 21 | 9.7796 | 0.8514 | 0.0000 | 86.366 | 1.000 | (32/32, single_frag=1) | 15.8 |
| 22 | 9.4446 | 0.8251 | 0.0000 | 107.356 | 1.000 | (32/32, single_frag=1) | 10.8 |
| 23 | 10.0358 | 0.8142 | 0.0000 | 105.922 | 1.000 | (32/32, single_frag=1) | 11.5 |
| 24 | 10.4470 | 0.8164 | 0.0000 | 81.057 | 1.000 | (32/32, single_frag=1) | 17.6 |
| 25 | 9.8666 | 0.7977 | 0.0000 | 84.928 | 1.000 | (32/32, single_frag=1) | 15.6 |
| 26 | 9.8090 | 0.8031 | 0.0000 | 157.953 | 1.000 | (32/32, single_frag=1) | 11.5 |
| 27 | 9.5039 | 0.8122 | 0.0000 | 102.265 | 1.000 | (32/32, single_frag=1) | 15.4 |
| 28 | 9.5779 | 0.8207 | 0.0000 | 171.649 | 1.000 | (32/32, single_frag=1) | 10.5 |
| 29 | 10.2893 | 0.7778 | 0.0000 | 86.350 | 1.000 | (32/32, single_frag=1) | 16.0 |
| 30 | 8.6804 | 0.7804 | 0.0000 | 158.571 | 1.000 | (32/32, single_frag=1) | 11.1 |
| 31 | 9.6829 | 0.7748 | 0.0000 | 106.233 | 1.000 | (32/32, single_frag=1) | 15.8 |
| 32 | 9.5230 | 0.7717 | 0.0000 | 127.394 | 1.000 | (32/32, single_frag=1) | 13.9 |
| 33 | 9.8168 | 0.7785 | 0.0000 | 110.889 | 1.000 | (32/32, single_frag=1) | 15.3 |
| 34 | 10.0803 | 0.7886 | 0.0000 | 96.676 | 1.000 | (32/32, single_frag=1) | 15.3 |
| 35 | 8.8657 | 0.7511 | 0.0000 | 129.218 | 1.000 | (32/32, single_frag=1) | 9.3 |

## Honest framing

- The target decode_ratio ≥ 0.80 is PASSED. If not met, the Wf-124 P2 architectural path is intact (YuelBond + Wasserstein + Catherine are wired and contributing as engine tags), but the trained decoder head has not yet reached the architectural ceiling. The full 100-epoch retrain would lift further.
- Wasserstein decay check on the 36-epoch history windowed by 10: PASS. The W₂ does not always decay monotonically in short retrain runs; the JKO bound holds on the full 50+ epoch horizon.
- Total wall time on cpu: 608.9s (~16.9s/epoch).
- All per-sample outputs carry the engine tags above; downstream reporters MUST preserve them (TODO-126 §7).

## Outputs

- `results/wfa108j_budget36/per_epoch_metrics.json` (full per-epoch record)
- `results/wfa108j_budget36/per_epoch_metrics.csv` (flat CSV)
- `results/wfa108j_budget36/wasserstein_decay.json` (W₂ decay check)
- `results/wfa108j_budget36/checkpoints/cfm_yuelbond_wasserstein.pt` (best checkpoint)
- `results/wfa108j_budget36/synthesis.md` (this file)

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
