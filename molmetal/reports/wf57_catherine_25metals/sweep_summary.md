# Wf-57 Phase 3 — Catherine 6/6 Properties × 25+ d-block metals

**Date**: 2026-09-26T08:42:06Z
**Wall-time total**: 6.91s
**#metals**: 25 / **#seeds**: 5 / **max_depth**: 1 / **max_products**: 64
**Mode**: GPU-0 Blackwell (real closure BFS)

## Headline verdict

- **25 / 25 metals × 6 / 6 properties PASS** (target >= 25 / 25)
- **0 metals** with at least one Catherine property FAIL.

## Per-property pass-rate

| Property | #metals PASS / #metals total | Pass-rate |
|---|---:|---:|
| P1_reflexivity | 25 / 25 | 100.00% |
| P2_transitivity | 25 / 25 | 100.00% |
| P3_symmetry | 25 / 25 | 100.00% |
| P4_dblock | 25 / 25 | 100.00% |
| P5_reachability | 25 / 25 | 100.00% |
| P6_composability | 25 / 25 | 100.00% |

## Per-metal 6/6 verdict

| Metal | 6/6 | P1 | P2 | P3 | P4 | P5 | P6 | wall[s] |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| Sc | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Ti | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| V | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Cr | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Mn | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Fe | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Co | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Ni | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Cu | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Zn | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Y | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Zr | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Nb | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Mo | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Tc | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Ru | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Rh | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Pd | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Ag | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Cd | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Hf | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Ta | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| W | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Re | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Os | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |

## Per-metal bench summary

| Metal | n_visited | n_products | max_depth_seen | witness_rate | well_typed_rate | coord_violations | reachability_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sc | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Ti | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| V | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Cr | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Mn | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Fe | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Co | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Ni | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Cu | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Zn | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Y | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Zr | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Nb | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Mo | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Tc | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Ru | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Rh | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Pd | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Ag | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Cd | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Hf | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Ta | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| W | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Re | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |
| Os | 90 | 50 | 1 | 1.000 | 1.000 | 0 | 0.281 |

## File outputs

- `per_metal_pass.json` — full per-metal 6/6 verdict + summary
- `sweep_summary.md` — this file
