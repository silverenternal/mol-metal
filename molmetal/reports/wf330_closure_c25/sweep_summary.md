# Wf-57 Phase 3 — Catherine 6/6 Properties × 25+ d-block metals

**Date**: 2026-09-29T05:42:16Z
**Wall-time total**: 2.47s
**#metals**: 25 / **#seeds**: 1 / **max_depth**: 1 / **max_products**: 1000
**Mode**: GPU-0 Blackwell (real closure BFS)

## Headline verdict

- **0 / 25 metals × 6 / 6 properties PASS** (target >= 25 / 25)
- **25 metals** with at least one Catherine property FAIL.

## Per-property pass-rate

| Property | #metals PASS / #metals total | Pass-rate |
|---|---:|---:|
| P1_reflexivity | 25 / 25 | 100.00% |
| P2_transitivity | 0 / 25 | 0.00% |
| P3_symmetry | 0 / 25 | 0.00% |
| P4_dblock | 25 / 25 | 100.00% |
| P5_reachability | 25 / 25 | 100.00% |
| P6_composability | 25 / 25 | 100.00% |

## Per-metal 6/6 verdict

| Metal | 6/6 | P1 | P2 | P3 | P4 | P5 | P6 | wall[s] |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| Sc | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Ti | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| V | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Cr | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Mn | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Fe | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Co | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Ni | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Cu | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Zn | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Y | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Zr | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Nb | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Mo | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Tc | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Ru | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Rh | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Pd | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Ag | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Cd | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Hf | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Ta | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| W | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Re | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |
| Os | NO | PASS | FAIL | FAIL | PASS | PASS | PASS | 0.00 |

## Per-metal bench summary

| Metal | n_visited | n_products | max_depth_seen | witness_rate | well_typed_rate | coord_violations | reachability_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sc | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Ti | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| V | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Cr | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Mn | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Fe | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Co | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Ni | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Cu | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Zn | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Y | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Zr | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Nb | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Mo | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Tc | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Ru | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Rh | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Pd | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Ag | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Cd | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Hf | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Ta | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| W | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Re | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |
| Os | 18 | 10 | 1 | 0.000 | 0.000 | 0 | 0.018 |

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

### Cr

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Mn

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Fe

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Co

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Ni

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Cu

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Zn

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Y

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Zr

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Nb

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Mo

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Tc

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Ru

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Rh

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Pd

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Ag

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Cd

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Hf

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Ta

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### W

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Re

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

### Os

- **P2_transitivity** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: witness_rate=0.0000 < 1.0 indicates a term in the visited set has no finite witness; the closure operator is not transitive at that term
- **P3_symmetry** FAIL — n_samples=10, n_violations=10, wall=0.0000s
  - Note: well_typed_rate=0.0000 < 1.0: closure is non-idempotent — products emitted by one BFS layer may not be re-emittable on a second pass

## File outputs

- `per_metal_pass.json` — full per-metal 6/6 verdict + summary
- `sweep_summary.md` — this file
