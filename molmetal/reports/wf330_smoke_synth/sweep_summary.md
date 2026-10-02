# Wf-57 Phase 3 — Catherine 6/6 Properties × 25+ d-block metals

**Date**: 2026-09-29T05:42:09Z
**Wall-time total**: 0.88s
**#metals**: 3 / **#seeds**: 1 / **max_depth**: 1 / **max_products**: 32
**Mode**: CPU-only synthetic fallback

## Headline verdict

- **3 / 3 metals × 6 / 6 properties PASS** (target >= 25 / 25)
- **0 metals** with at least one Catherine property FAIL.

## Per-property pass-rate

| Property | #metals PASS / #metals total | Pass-rate |
|---|---:|---:|
| P1_reflexivity | 3 / 3 | 100.00% |
| P2_transitivity | 3 / 3 | 100.00% |
| P3_symmetry | 3 / 3 | 100.00% |
| P4_dblock | 3 / 3 | 100.00% |
| P5_reachability | 3 / 3 | 100.00% |
| P6_composability | 3 / 3 | 100.00% |

## Per-metal 6/6 verdict

| Metal | 6/6 | P1 | P2 | P3 | P4 | P5 | P6 | wall[s] |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| Sc | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| Ti | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |
| V | YES | PASS | PASS | PASS | PASS | PASS | PASS | 0.00 |

## Per-metal bench summary

| Metal | n_visited | n_products | max_depth_seen | witness_rate | well_typed_rate | coord_violations | reachability_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sc | 8 | 7 | 1 | 1.000 | 1.000 | 0 | 1.000 |
| Ti | 8 | 7 | 1 | 1.000 | 1.000 | 0 | 1.000 |
| V | 8 | 7 | 1 | 1.000 | 1.000 | 0 | 1.000 |

## File outputs

- `per_metal_pass.json` — full per-metal 6/6 verdict + summary
- `sweep_summary.md` — this file
