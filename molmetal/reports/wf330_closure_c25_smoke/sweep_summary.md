# Wf-57 Phase 3 — Catherine 6/6 Properties × 25+ d-block metals

**Date**: 2026-09-29T10:37:50Z
**Wall-time total**: 1.24s
**#metals**: 3 / **#seeds**: 1 / **max_depth**: 1 / **max_products**: 32
**Mode**: GPU-0 Blackwell (real closure BFS)

## Headline verdict

- **0 / 3 metals × 6 / 6 properties PASS** (target >= 25 / 25)
- **3 metals** with at least one Catherine property FAIL.

## Per-property pass-rate

| Property | #metals PASS / #metals total | Pass-rate |
|---|---:|---:|
| P1_reflexivity | 3 / 3 | 100.00% |
| P2_transitivity | 0 / 3 | 0.00% |
| P3_symmetry | 0 / 3 | 0.00% |
| P4_dblock | 3 / 3 | 100.00% |
| P5_reachability | 3 / 3 | 100.00% |
| P6_composability | 3 / 3 | 100.00% |

## Per-metal 6/6 verdict

| Metal | 6/6 | P1 | P2 | P3 | P4 | P5 | P6 | wall[s] |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| Sc | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Ti | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| V | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |

## Per-metal bench summary

| Metal | n_visited | n_products | max_depth_seen | witness_rate | well_typed_rate | coord_violations | reachability_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sc | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.562 |
| Ti | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.562 |
| V | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.562 |

## Diagnostics — metals with one or more FAIL properties

### Sc

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Ti

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### V

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

## File outputs

- `per_metal_pass.json` — full per-metal 6/6 verdict + summary
- `sweep_summary.md` — this file
