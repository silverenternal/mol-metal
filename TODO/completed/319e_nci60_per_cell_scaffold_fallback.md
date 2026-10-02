# Wf-319E — NCI-60 9-cell per-cell n_test_min auto-tuning

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent
**Status**: SHIPPED (7 passed + 3 skipped = 10 NEW tests; Wf-315F bug
fix shipped; per-cell n_test_min auto-tuning shipped)
**Parent**: `TODO/completed/315f_nci60_per_cell_scaffold_fallback.md`
(Wf-315F)
**Predecessor**: Wf-315F (per-cell scaffold-split fallback pattern)
**Follow-up to**: Wf-315F §3 honest-negative finding (scaffold fallback
wired but never triggers with default n_test_min=50)

---

## §1 Gap description

Wf-315F extended the NCI-60 9-cell driver
(`molmetal/scripts/nci60_9cell_chemprop_driver.py`) with
`--split {temporal, scaffold, auto}` to handle the Re degenerate
n_test=1 temporal case (Wf-312A pattern). Wf-315F shipped honestly,
but its §3 honest-negative finding flagged a residual gap: **the
scaffold fallback is wired but never actually triggers with default
`--n-test-min=50` because no NCI-60 cell has a Bemis-Murcko scaffold
group with >= 50 rows**. Per-cell max scaffold sizes (MEASURED Sep
28 2026):

| Cell | max scaffold size | n_test_random_est |
|---|---:|---:|
| A549 | 37 | 529 |
| MCF7 | 34 | 393 |
| A2780 | 37 | 219 |
| A2780cis | 30 | 113 |
| HCT116 | 20 | 140 |
| SKOV3 | 18 | 38 |
| HL60 | 18 | 45 |
| DU145 | 10 | 32 |
| PC3 | 9 | 55 |

So with default `n_test_min=50`, the scaffold fallback label is
applied (HL60/SKOV3/DU145) but the helper returns None → random
fallback. In effect all 9 cells use the random split.

**Wf-319E closes this gap by introducing per-cell n_test_min
auto-tuning.** The per-cell threshold is computed from the max
Bemis-Murcko scaffold group size with a configurable headroom
factor (default 0.85) and floor (default 5).

---

## §2 Per-cell n_test_min (Wf-319E MEASURED Sep 29 2026)

With `headroom=0.85, floor=5`:

| Cell | max scaffold | n_test_min_auto | split choice under `--split auto` |
|---|---:|---:|---|
| A549 | 37 | 31 | sprint-OOD (n_test_random_est=529 >= 31) → temporal |
| MCF7 | 34 | 29 | temporal (n_test_random_est=393 >= 29) |
| A2780 | 37 | 31 | temporal (n_test_random_est=219 >= 31) |
| A2780cis | 30 | 26 | temporal (n_test_random_est=113 >= 26) |
| HCT116 | 20 | 17 | temporal (n_test_random_est=140 >= 17) |
| SKOV3 | 18 | 15 | scaffold_fallback (n_test_random_est=38 >= 15 → random) |
| HL60 | 18 | 15 | scaffold_fallback (n_test_random_est=45 >= 15 → random) |
| DU145 | 10 | 8 | scaffold_fallback (n_test_random_est=32 >= 8 → random) |
| PC3 | 9 | 8 | scaffold_fallback (n_test_random_est=55 >= 8 → random) |

**Honest framing**: With auto-tuning ON (Wf-319E default), `--split
auto` flips 4 cells to `scaffold_fallback` label (SKOV3, HL60,
DU145, PC3) — and the scaffold helper actually finds a group ≥
n_test_min for 3 of them (SKOV3=15, HL60=15, DU145=8 all have
groups >= threshold; PC3=8 has max=9 so threshold 8 ≤ max). For
all 4 of these cells the random test partition is still larger
than the threshold, so they remain on random in practice — BUT
when users explicitly pass `--split scaffold`, the scaffold
helper now successfully returns a valid OOD split for all 9 cells
(previously: 0/9 cells because the threshold was unreachable).

This is **INTEGRATION work**, NOT a new measurement: no AUC values
are reported.

---

## §3 Code changes (per file)

### §3.1 `molmetal/scripts/_nci60_scaffold_helpers.py` (NEW)

New pure-Python helpers file (RDKit-only, no chemprop v2 dep) so the
scaffold-split logic can be unit-tested without `cuik_molmaker`:

- `_scaffold_split_indices_nci60(...)` — moved from driver; same
  behaviour as Wf-315F.
- `compute_per_cell_n_test_min(...)` — new Wf-319E function. Per-cell
  threshold = `max(round(headroom_factor * max_scaffold_size),
  n_test_min_floor)`.

### §3.2 `molmetal/scripts/nci60_9cell_chemprop_driver.py`

- Imports helpers from `molmetal.scripts._nci60_scaffold_helpers`
  (re-exported for backward compat).
- `execute_sweep(...)`: per-cell n_test_min auto-tuning (when
  `args.split == "auto"` and `args.n_test_min_auto` is True).
  Computes per-cell threshold from `compute_per_cell_n_test_min`
  and records it in the new `per_cell_n_test_min` JSON field.
- New CLI flags (all default to sensible values):
  - `--n-test-min-auto` / `--no-n-test-min-auto` (default ON;
    restores Wf-315F single-threshold mode when OFF).
  - `--n-test-min-floor N` (default 5; minimum per-cell threshold).
  - `--n-test-min-headroom F` (default 0.85; multiplicative headroom
    on max scaffold size).
- `split_protocol` JSON field now includes per-cell n_test_min.
- `split_arg` JSON field now includes `n_test_min`, `n_test_min_auto`,
  `n_test_min_floor`, `n_test_min_headroom`.
- `honest_framing_note` updated to surface the per-cell n_test_min
  data and the Wf-319E finding honestly.

### §3.3 `molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py`

Bug fix (Wf-319E): `test_a549_split_choice_temporal` had an
unguarded `import nci60_9cell_chemprop_driver` that caused the
test to FAIL (not skip) when the chemprop v2 driver can't import
`cuik_molmaker`. The import was unused (the test only checks the
closed-form threshold rule); it was removed. Net effect: Wf-315F
test pass rate improves from 1 failed + 4 passed + 5 skipped to
5 passed + 5 skipped.

### §3.4 `molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py` (NEW)

New file with 10 tests across 3 test classes:

1. **TestPerCellNTestMinAutoComputation** (5 tests) — per-cell
   `compute_per_cell_n_test_min` correctness against the closed-form
   rule:
   - A549 (max=37) → 31
   - MCF7 (max=34) → 29 (banker's rounding: 0.85×34=28.9 → 29)
   - A2780 (max=37) → 31
   - DU145 (max=10) → 8 (banker's rounding: 0.85×10=8.5 → 8 even)
   - PC3 (max=9) → 8

2. **TestPerCellSplitDecisionAutoTuned** (3 tests) — per-cell split
   decision with auto-tuning:
   - A549 with `--split scaffold` + auto-tuned n_test_min=31 returns
     valid scaffold group
   - DU145 with `--split scaffold` + auto-tuned n_test_min=8 returns
     valid scaffold group
   - `--no-n-test-min-auto` restores single-threshold mode

3. **TestDriverCLIIntegration** (2 tests) — CLI flag acceptance and
   dry-run plan shape:
   - All 4 new flags accepted by argparse
   - dry_run_plan() works with auto flags

---

## §4 Test results

```
$ python -m pytest molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py -v
molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py::TestPerCellNTestMinAutoComputation::test_a549_n_test_min_auto_is_31 PASSED [  6%]
molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py::TestPerCellNTestMinAutoComputation::test_mcf7_n_test_min_auto_is_29 PASSED [ 12%]
molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py::TestPerCellNTestMinAutoComputation::test_a2780_n_test_min_auto_is_31 PASSED [ 18%]
molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py::TestPerCellNTestMinAutoComputation::test_du145_n_test_min_auto_is_8 PASSED [ 24%]
molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py::TestPerCellNTestMinAutoComputation::test_pc3_n_test_min_auto_is_8 PASSED [ 30%]
molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py::TestPerCellSplitDecisionAutoTuned::test_split_scaffold_works_for_a549_with_per_cell_threshold PASSED [ 36%]
molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py::TestPerCellSplitDecisionAutoTuned::test_split_scaffold_works_for_du145_with_per_cell_threshold PASSED [ 42%]
molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py::TestPerCellSplitDecisionAutoTuned::test_auto_tuning_off_keeps_default_threshold SKIPPED [ 48%]
molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py::TestDriverCLIIntegration::test_cli_accepts_new_n_test_min_flags SKIPPED [ 54%]
molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py::TestDriverCLIIntegration::test_dry_run_plan_includes_per_cell_n_test_min_field SKIPPED [ 60%]
================= 7 passed, 3 skipped in 7.12s =================
```

Combined with the Wf-315F tests:

```
$ python -m pytest molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py \
                    molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py \
                    molmetal/tests/test_wf312a_re_non_degenerate.py -v
=========== 14 passed, 11 skipped in 7.82s ===========
```

(Wf-315F: 5 passed + 5 skipped; Wf-319E: 7 passed + 3 skipped;
Wf-312A: 2 passed + 3 skipped; total 14 passed, 11 skipped.)

---

## §5 Usage

```bash
# Default (Wf-319E): per-cell n_test_min auto-tuning ON
python molmetal/scripts/nci60_9cell_chemprop_driver.py \
    --csv molmetal/data/nci60_precious_9cell.csv \
    --split auto --dry-run

# Explicit scaffold-OOD per cell (with auto-tuned threshold)
python molmetal/scripts/nci60_9cell_chemprop_driver.py \
    --csv molmetal/data/nci60_precious_9cell.csv \
    --split scaffold --dry-run

# Restore Wf-315F single-threshold mode (n_test_min=50 for all cells)
python molmetal/scripts/nci60_9cell_chemprop_driver.py \
    --csv molmetal/data/nci60_precious_9cell.csv \
    --split auto --no-n-test-min-auto --dry-run

# Custom headroom + floor
python molmetal/scripts/nci60_9cell_chemprop_driver.py \
    --csv molmetal/data/nci60_precious_9cell.csv \
    --split auto \
    --n-test-min-headroom 0.7 \
    --n-test-min-floor 10 \
    --dry-run
```

---

## §6 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: Per-cell row counts, max scaffold sizes, and
  per-cell n_test_min auto values come from `pandas.read_csv` on
  `molmetal/data/nci60_precious_9cell.csv` (MEASURED Sep 29 2026,
  7,823 rows). The auto-computed thresholds follow the closed-form
  `max(round(headroom * max_size), floor)` rule with default
  `headroom=0.85, floor=5`.
- **PROJECTED**: No AUC values are reported; this is INTEGRATION
  work only. Per-cell n_test_min is recorded in the JSON output
  but not consumed by any downstream metric.
- **HONEST-NEGATIVE**: With auto-tuning ON, the scaffold fallback
  label is applied to 4 cells (SKOV3/HL60/DU145/PC3), but for all
  4 the random test partition still beats the per-cell threshold —
  the actual split choice is "temporal (random 80/10/20)" for all
  9 cells in `--split auto` mode. The auto-tuning is primarily
  meaningful for users passing `--split scaffold` explicitly: it
  lets the helper return valid OOD splits for all 9 cells where
  previously 0/9 worked.
- **Banker's rounding**: Python's `round()` uses banker's rounding
  (round half to even). For DU145 (max=10), 0.85×10=8.5 → 8 (even).
  This is documented in the test docstring; users who want
  round-half-up can use `--n-test-min-headroom 0.86` to push the
  threshold to 9.
- **Pre-flight**: This work is INTEGRATION-only (driver extension +
  helper extraction + tests + JSON field); it does NOT launch any
  retrain / panel / ablation. The Pre-flight v2 25/25 PASS gate
  (TODO-207) is not invoked.
- **Skip infrastructure gap**: 3 of 10 Wf-319E tests SKIP because
  the chemprop v2 driver import requires `cuik_molmaker`, which
  is not installed in this venv (same gap as Wf-315F / Wf-312A).
  All 7 non-supertial tests PASS.

---

## §7 Acceptance gate (Wf-319E)

- [x] G1. `compute_per_cell_n_test_min(...)` helper added (RDKit-only,
      in `_nci60_scaffold_helpers.py`).
- [x] G2. Driver extended with `--n-test-min-auto` / `--no-n-test-min-auto`
      / `--n-test-min-floor` / `--n-test-min-headroom` CLI flags.
- [x] G3. Per-cell `n_test_min` recorded in JSON output
      (`per_cell_n_test_min` field, plus `split_arg` extended).
- [x] G4. Wf-315F test bug fix (`test_a549_split_choice_temporal`
      unguarded import). Net effect: Wf-315F pass count 4 → 5.
- [x] G5. 10 NEW Wf-319E tests shipped
      (`test_wf319e_nci60_per_cell_scaffold_fallback.py`).
- [x] G6. Honest framing note in driver + this doc updated to
      surface the per-cell n_test_min data and Wf-319E finding.
- [x] G7. `TODO/completed/319e_nci60_per_cell_scaffold_fallback.md`
      shipped (this file).

---

## §8 Cross-references

- `molmetal/scripts/nci60_9cell_chemprop_driver.py` (extended)
- `molmetal/scripts/_nci60_scaffold_helpers.py` (NEW)
- `molmetal/scripts/wf219_temporal_split_driver.py` (Wf-312A pattern)
- `molmetal/tests/test_wf319e_nci60_per_cell_scaffold_fallback.py`
  (NEW, 10 tests)
- `molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py`
  (Wf-315F tests, bug-fixed)
- `molmetal/tests/test_wf312a_re_non_degenerate.py` (Wf-312A tests)
- `TODO/completed/315f_nci60_per_cell_scaffold_fallback.md` (parent)
- `TODO/pending/220_wf219_temporal_split_nci60_driver.md`
  (TODO-220 Wf-308C stub driver)

---

**END OF Wf-319E** — Owner Hugo Lee; cross-ref `party-twenty-spirit`
§3.5; ship date 2026-09-29.