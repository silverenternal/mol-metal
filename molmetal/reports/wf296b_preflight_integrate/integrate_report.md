# Wf-296B — pre_flight_check.py auto-verify integration of new tests

**Date**: 2026-09-29
**Worker**: Wf-296B
**Target file**: `molmetal/scripts/pre_flight_check.py`
**Test gate updated**: `p2_tests_pass` (line ~775)

## Summary

Integrated 4 newly-collectable test files (from Wf-283B / Wf-285B / Wf-290 / Wf-268A)
into the existing `p2_tests_pass` gate so that regressions in those subsystems are
caught by pre-flight. Pre-flight gate count unchanged (25/25 PASS). Net new
gate coverage: 672 passing tests added to preflight (~3× more test work
exercised per preflight run).

## Diff summary

Single Edit to `molmetal/scripts/pre_flight_check.py` at line ~775 (the
`p2_tests_pass` gate's `gate_pytest` call). The 3 baseline tests from Wf-259
remain; 4 new tests appended; timeout bumped 300s → 600s (pytest run now
covers 7 files / 699 tests vs. 3 files / 27 tests).

### Test files added (4)

| Test file | Origin | Tests added | Status |
|-----------|--------|-------------|--------|
| `test_kendall_dann_wiring.py` | Wf-285B | 3 | all PASS |
| `test_no_kendall_no_dann_flags.py` | Wf-283B | (incl. in 672) | all PASS |
| `test_wf268a_cfm_ablation_flags.py` | Wf-268A | (incl. in 672) | all PASS |
| `test_metal_ligand_exchange_matrix.py` | Wf-290F4 | (incl. in 672) | all PASS |

### Test files NOT added (1, with reason)

| Test file | Origin | Reason |
|-----------|--------|--------|
| `test_f2a_metal_ligand_exchange.py` | Wf-290F1 | 33/309 tests fail pre-existing `SyntheMolReactionRule.apply()` AttributeError; file is collectable (post-Wf-290F1 import fix) but upstream module mismatch. Tracked for Wf-297 follow-up. |

## Verification run

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py
...
  "summary": { "pass": 25, "fail": 0, "total": 25, ... }
  "p2_tests_pass": PASS pytest PASS: 699 passed, 1 warning in 105.68s
```

Pre-flight: **25/25 PASS** (gate count unchanged from Wf-259 baseline).
`p2_tests_pass` now runs **699 tests in 105.68s** (up from 27 tests in <60s).

## Output (per task spec)

```
n_tests_added=4
preflight_25_25=PASS
n_new_gates=0  (added to existing p2_tests_pass gate; no new gate introduced)
```

## Notes

- Gate augmentation (vs. new gate) chosen because the new tests are
  pytest-collected regression guards with the same priority/P2 classification
  as the existing `p2_tests_pass` entries. Introducing a separate gate would
  duplicate pytest collection overhead.
- timeout bumped 300s → 600s because 7 files × ~30s median = 210s, plus
  chemprop cold-start; 600s gives 2.85× margin (consistent with the Wf-294A
  90s → chemprop --quick precedent for cold-start budgeting).
- The 33 pre-existing failures in `test_f2a_metal_ligand_exchange.py` are
  out of scope for this integration (different workflow, different module
  surface). See Wf-297 follow-up.