# Wf-315F — NCI-60 9-cell per-cell scaffold-split fallback

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent
**Status**: SHIPPED (10/10 tests pass, dry-run verified)
**Parent**: `TODO/pending/220_wf219_temporal_split_nci60_driver.md` (TODO-220 Wf-308C stub driver)
**Predecessor**: Wf-312A (Re degenerate temporal-split fix in `wf219_temporal_split_driver.py`)

---

## §1 Gap description

Wf-308C shipped the dry-run + execute paths for the NCI-60 9-cell
chemprop v2 multi-task driver, but **only** with a random 80/10/20
split per cell. Wf-312A added the `--split {temporal, scaffold,
auto}` pattern to the metal-cohort driver to handle the Re
degenerate-n_test=1 temporal case.

**Wf-315F wires the same pattern into the NCI-60 9-cell driver** so
any per-cell cohort shrink can fall back to Bemis-Murcko scaffold-OOD
instead of the random holdout. NCI-60 has no temporal Year column
(only `smiles, cell_line, pIC50, active`), so "temporal" here maps
to the random 80/10/20 split — but the Wf-312A honest-fallback
contract is preserved.

This is **INTEGRATION work**, NOT a new measurement: no AUC values
are reported.

---

## §2 Per-cell row distribution (MEASURED Sep 28 2026)

From `molmetal/data/nci60_precious_9cell.csv` (7,823 rows total):

| Cell | n_rows | n_test_random_est | auto decision |
|---|---:|---:|---|
| A549 | 2647 | 529 | temporal |
| MCF7 | 1964 | 393 | temporal |
| A2780 | 1095 | 219 | temporal |
| HCT116 | 701 | 140 | temporal |
| A2780cis | 564 | 113 | temporal |
| PC3 | 275 | 55 | temporal |
| HL60 | 225 | 45 | scaffold_fallback |
| SKOV3 | 191 | 38 | scaffold_fallback |
| DU145 | 161 | 32 | scaffold_fallback |

`n_test_random_est = round(0.20 × n_rows)` is the closed-form 80/10/20
test size used in `execute_sweep` to drive the per-cell auto
decision (default `--n-test-min=50`).

---

## §3 Honest-negative finding (Wf-315F surfaced)

Per-cell Bemis-Murcko scaffold group max sizes (MEASURED):

| Cell | max scaffold size | n_scaffolds |
|---|---:|---:|
| A549 | 37 | 1352 |
| MCF7 | 34 | 978 |
| A2780 | 37 | 471 |
| A2780cis | 30 | 272 |
| HCT116 | 20 | 361 |
| SKOV3 | 18 | 83 |
| HL60 | 18 | 128 |
| DU145 | 10 | 96 |
| PC3 | 9 | 167 |

**No cell has a scaffold group with >= 50 rows.** This means:

1. With default `--n-test-min=50` (consistent with Wf-312A pattern):
   the 3 borderline cells (HL60, SKOV3, DU145) flip to
   `scaffold_fallback` in auto mode. The scaffold helper then
   returns `None` (no group >= 50 rows) → `random_80_10_20_fallback`
   is used. **In effect, all 9 cells use the random split.**
2. With `--n-test-min=10` and `--split scaffold`: only A549, MCF7,
   A2780 produce a valid scaffold-OOD split. Other cells fall back
   to random.
3. With `--split scaffold --n-test-min=5`: more cells qualify but
   the scaffold-OOD test set becomes small (e.g. DU145 n_test = 5).

This is the Wf-315F honest framing: **scaffold fallback is wired
correctly but the NCI-60 per-cell scaffold distribution makes
strict scaffold-OOD evaluation practical only for the 3 largest
cells (A549/MCF7/A2780) and only at small n_test_min.** This is
fundamentally different from the Re cohort (868 rows / 123
scaffolds / max size 50+) where Wf-312A's scaffold fallback
worked cleanly.

---

## §4 Code changes (per file)

### §4.1 `molmetal/scripts/nci60_9cell_chemprop_driver.py`

- Added `split: str` field to `Nci60CellRun` dataclass (default
  `"temporal_random_80_10_20"`).
- Added local `_scaffold_split_indices_nci60(...)` helper that mirrors
  `wf219_temporal_split_driver._scaffold_split_indices(...)` but is
  parameterised by `n_test_min` (default 50) and adapts the Wf-312A
  scaffold-OOD pattern to NCI-60's lack of a Year column.
- Refactored `run_one_per_cell(...)` to accept `split_mode:
  {temporal, scaffold, auto}` and `n_test_min: int` (default 50).
  The three branches pick train/val/test accordingly and record
  `chosen_split_label` in the returned `Nci60CellRun.split` field.
- `execute_sweep(...)` computes `per_cell_split_choice` per cell
  before the training loop (using closed-form n_test_random_est) and
  passes `effective_split_mode` to `run_one_per_cell` accordingly.
- The aggregate JSON now includes `per_cell_split_choice`,
  `split_protocol`, and `split_arg` fields.
- Added two new CLI flags:
  - `--split {temporal, scaffold, auto}` (default `temporal`,
    preserves Wf-308C behaviour).
  - `--n-test-min N` (default 50; matches Wf-312A threshold).
- Updated honest framing note to reflect the per-cell scaffold
  distribution honest-negative finding.

### §4.2 `molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py`

New file with 10 tests across 3 test classes:

1. **TestPerCellSplitDecision** (5 tests) — per-cell auto decision
   per the closed-form 0.20 × n_rows rule:
   - A549 (n=2647) → temporal
   - MCF7 (n=1964) → temporal
   - A2780 (n=1095) → temporal
   - HCT116 (n=701) → temporal
   - HL60 (n=225, n_test_est=45) → scaffold_fallback (borderline)

2. **TestScaffoldFallbackCorrectness** (3 tests) — scaffold helper
   correctness using `n_test_min=10` to exercise A549/MCF7/A2780:
   - `_scaffold_split_indices_nci60` returns valid splits for A549
   - scaffold split is disjoint (train/val/test union = sum of parts)
   - same seed produces identical splits (regression guard)

3. **TestNCI60PerCellIntegration** (2 tests) — integration:
   - `--split {temporal, scaffold, auto}` is accepted by the CLI
     (and bogus values rejected with SystemExit)
   - `run_one_per_cell` records `chosen_split_label` in the returned
     `Nci60CellRun.split` field for both `temporal` and `scaffold`
     modes on a tiny synthetic cohort.

---

## §5 Test results

```
$ python -m pytest molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py -v
molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py::TestPerCellSplitDecision::test_a549_split_choice_temporal PASSED [ 10%]
molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py::TestPerCellSplitDecision::test_mcf7_split_choice_temporal PASSED [ 20%]
molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py::TestPerCellSplitDecision::test_a2780_split_choice_temporal PASSED [ 30%]
molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py::TestPerCellSplitDecision::test_hct116_split_choice_temporal PASSED [ 40%]
molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py::TestPerCellSplitDecision::test_hl60_split_choice_scaffold_fallback PASSED [ 50%]
molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py::TestScaffoldFallbackCorrectness::test_scaffold_split_returns_non_degenerate_for_a549 PASSED [ 60%]
molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py::TestScaffoldFallbackCorrectness::test_scaffold_split_disjoint_train_val_test PASSED [ 70%]
molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py::TestScaffoldFallbackCorrectness::test_scaffold_split_seed_deterministic PASSED [ 80%]
molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py::TestNCI60PerCellIntegration::test_per_cell_split_choice_recorded_in_dry_run_plan PASSED [ 90%]
molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py::TestNCI60PerCellIntegration::test_run_one_per_cell_split_field_records_chosen_split PASSED [100%]
======================== 10 passed, 1 warning in 7.16s =========================
```

---

## §6 Usage

```bash
# Default (preserves Wf-308C behaviour): random 80/10/20 per cell
python molmetal/scripts/nci60_9cell_chemprop_driver.py \
    --csv molmetal/data/nci60_precious_9cell.csv \
    --split temporal --dry-run

# Scaffold-OOD per cell (future-proofed)
python molmetal/scripts/nci60_9cell_chemprop_driver.py \
    --csv molmetal/data/nci60_precious_9cell.csv \
    --split scaffold --n-test-min 10 --dry-run

# Auto: temporal when healthy; scaffold fallback for borderline
python molmetal/scripts/nci60_9cell_chemprop_driver.py \
    --csv molmetal/data/nci60_precious_9cell.csv \
    --split auto --n-test-min 50 --dry-run
```

---

## §7 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: All per-cell row counts and per-cell max scaffold
  sizes come from `pandas.read_csv` on
  `molmetal/data/nci60_precious_9cell.csv` (MEASURED Sep 28 2026,
  7,823 rows).
- **PROJECTED**: No AUC values are reported; this is INTEGRATION
  work only (the driver now has the `--split` flag wired and the
  per-cell split_choice recorded in JSON).
- **HONEST-NEGATIVE**: With default `--n-test-min=50`, all 9 cells
  effectively use the random split because no cell has a scaffold
  group >= 50 rows. This is a real constraint of the NCI-60
  per-cell distribution and is documented honestly in the driver's
  `honest_framing_note`.
- **Pre-flight**: This work is INTEGRATION-only (driver extension +
  tests); it does NOT launch any retrain / panel / ablation. The
  Pre-flight v2 25/25 PASS gate (TODO-207) is not invoked.

---

## §8 Acceptance gate (Wf-315F)

- [x] G1. `--split {temporal, scaffold, auto}` wired into
  `nci60_9cell_chemprop_driver.py` (CLI + argparse + JSON output).
- [x] G2. `_scaffold_split_indices_nci60` helper added locally
  (mirrors Wf-312A pattern but adapts to NCI-60's lack of Year
  column).
- [x] G3. Per-cell `split_choice` recorded in JSON output
  (`per_cell_split_choice` field).
- [x] G4. `Nci60CellRun.split` field added; populated by all three
  split modes.
- [x] G5. 10/10 NEW tests pass (`test_wf315f_nci60_per_cell_scaffold_fallback.py`).
- [x] G6. Honest framing note in driver updated to surface the
  per-cell scaffold distribution finding honestly.
- [x] G7. `TODO/completed/315f_nci60_per_cell_scaffold_fallback.md`
  shipped (this file).

---

## §9 Cross-references

- `molmetal/scripts/nci60_9cell_chemprop_driver.py` (extended)
- `molmetal/scripts/wf219_temporal_split_driver.py` (Wf-312A pattern)
- `molmetal/tests/test_wf315f_nci60_per_cell_scaffold_fallback.py`
  (new, 10 tests)
- `molmetal/tests/test_wf312a_re_non_degenerate.py` (Wf-312A test)
- `TODO/pending/220_wf219_temporal_split_nci60_driver.md` (parent
  TODO-220 Wf-308C stub driver)
- `TODO/completed/312a_re_non_degenerate.md` (Wf-312A ship)

---

**END OF Wf-315F** — Owner Hugo Lee; cross-ref `party-twenty-spirit`
§3.5; ship date 2026-09-29.
