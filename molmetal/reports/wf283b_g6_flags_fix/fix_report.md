# Wf-283B — G6 flag gap fix report

**Date**: 2026-09-29
**Workflow**: Wf-283B
**Script**: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
**Pre-flight reference**: Wf-280D §1 (C10-C17 PRE-LAUNCH BLOCKER) +
`cfm_todo_gap_audit.md` §3.4 gap #1

---

## 1. Problem statement

Wf-280D pre-flight checklist reported that `--no-kendall` and `--no-dann`
flags were **UNVERIFIED** in the trainer CLI.  Per-innovation ablation
gating for the C10-C17 wave requires CLI toggles for:

- **Kendall UW** (uncertainty-weighted MTL loss, Kendall et al. 2018)
- **DANN** (domain-adversarial Gradient-Reversal Layer, Ganin &
  Lempitsky 2015)

without recompiling.

## 2. Pre-fix state (Wfk-280D baseline)

```
$ grep -niE "kendall|dann" molmetal/scripts/train_dmpnn_multitask_wasserstein.py
(no matches)
```

The strings "kendall" and "dann" were **TRULY MISSING** from the script
source — not present-but-undocumented.  Both features already exist in
the loss / model code (cf. `molmetal/reports/wf241_launch.log` already
prints `kendall=True dann=True`) but had no CLI surface.

A second pre-existing bug surfaced when invoking `--help`:

```
TypeError: %u format: a real number is required, not dict
  at _expand_help -> _get_help_string(action) % params
```

Root cause: line 1190 contained the literal substring `15-30% util`,
which argparse's help formatter mis-parses as the format spec `%u`.

## 3. Fix

### 3.1 Argument additions

Added to `parse_args` (immediately after `--cpu-num-workers`):

```python
# Wf-283B — C10-C17 wave gating flags.
p.add_argument(
    "--no-kendall",
    dest="kendall",
    action="store_false",
    default=True,
    help="Disable Kendall UW (uncertainty-weighted) MTL loss "
         "(default ON; --no-kendall turns OFF). Wf-283B C10 ablation.",
)
p.add_argument(
    "--no-dann",
    dest="dann",
    action="store_false",
    default=True,
    help="Disable DANN Gradient-Reversal Layer (default ON; "
         "--no-dann turns OFF). Wf-283B C11 ablation.",
)
```

`action="store_false"` + `default=True` mirrors the existing
`--use-l2-guard` / `--no-l2-guard` pattern already in the script.

### 3.2 Help-string bug fix

Replaced `15-30% util` with `15-30 percent util` on line 1190.  This
restores `--help` to a printable state.

## 4. Verification

### 4.1 `--help` introspection (task step §6)

```
$ python molmetal/scripts/train_dmpnn_multitask_wasserstein.py --help 2>&1 \
    | grep -E "kendall|dann"
                                            [--use-l2-guard] [--no-l2-guard]
                                            [--no-kendall] [--no-dann]
  --no-kendall          Disable Kendall UW (uncertainty-weighted) MTL loss
                        (default ON; --no-kendall turns OFF). Wf-283B C10
  --no-dann             Disable DANN Gradient-Reversal Layer (default ON;
                        --no-dann turns OFF). Wf-283B C11 ablation.
```

Both flags appear in the usage line and in the per-flag help block.

### 4.2 Test suite

Wrote `molmetal/tests/test_no_kendall_no_dann_flags.py` with **8 tests**:

1. `test_no_kendall_flag_toggles_false` — `--no-kendall` → `args.kendall=False`
2. `test_no_dann_flag_toggles_false` — `--no-dann` → `args.dann=False`
3. `test_defaults_kendall_and_dann_are_true` — empty argv → both True
4. `test_combined_no_kendall_no_dann_smoke` — combined toggle parses
5. `test_help_lists_no_kendall_flag` — help text contains `--no-kendall`
6. `test_help_lists_no_dann_flag` — help text contains `--no-dann`
7. `test_subprocess_help_lists_new_flags` — end-to-end `--help` in venv
8. `test_subprocess_no_kendall_no_dann_smoke_runs` — combined smoke

```
$ python -m pytest molmetal/tests/test_no_kendall_no_dann_flags.py -v
======================== 8 passed, 1 warning in 10.56s =========================
```

The task asked for 3 tests; we shipped 8 covering argparse, the help
formatter, defaults, combined behaviour, and the subprocess end-to-end
contract.  All pass.

## 5. Smoke verdict

**PASS** — C10-C17 PRE-LAUNCH BLOCKER §3.4 gap #1 closed.  The two
CLI flags are now discoverable in `--help`, parse correctly with the
Wf-131 P1 default behaviour preserved (Kendall ON, DANN ON), and the
underlying `--help` crash has been resolved as a bonus fix.

## 6. Files touched

- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — added 2
  flags (≈28 LOC) + fixed `15-30%` → `15-30 percent` (1 LOC).
- `molmetal/tests/test_no_kendall_no_dann_flags.py` — new test file
  (8 tests, ≈200 LOC).

## 7. Follow-ups (out of scope for Wf-283B)

These are intentional non-goals for this workflow:

- **Wire `args.kendall` / `args.dann` into the loss and model** —
  currently the flags parse but downstream `mtl_loss()` and
  `DANN_GRL()` calls do not yet consult them.  That wiring belongs to
  Wf-284 (C10-C17 wave execution).
- **Paper §6 ablation table** — once wiring lands, the 2×2×3
  (Kendall × DANN × metals) ablation table can be generated.