# Wf-285C — `--no-enforce-budget` CLI flag report

**Author:** Wf-285C (Wf-283E P1.3 follow-up)
**Date:** 2026-09-29
**Scope:** `molmetal/scripts/wave_budget_tracker.py` — add the
`--no-enforce-budget` CLI flag with `action="store_true" default=False`,
wired into the existing `compute_wave_status(enforce_budget=...)` and
`summarise(enforce_budget=...)` plumbing.

---

## 1. Flag addition

`molmetal/scripts/wave_budget_tracker.py` — `parse_argv()` argparser now
exposes the flag:

```python
p.add_argument(
    "--no-enforce-budget",
    action="store_true",
    help=(
        "Disable DEFERRED labeling for waves that exceed the stated "
        "budget. All waves stay PLANNED; honest_framing still "
        "reports the over-budget figure. Per user 2026-09-28 "
        "'时间预算没关系的': budget is a tracker, not a hard blocker."
    ),
)
```

Wiring in `main()` (line ~888):

```python
enforce_budget = not args.no_enforce_budget
wave_dicts = compute_wave_status(
    args.budget_hr, waves, actual_hr_by_name,
    enforce_budget=enforce_budget,
)
```

The pre-existing `compute_wave_status(enforce_budget=...)` and
`summarise(enforce_budget=...)` parameters are consumed by this flag
(Wf-262 §0.5 honest-gap fix shipped the helpers in Wf-268; Wf-285C wires
the public CLI surface).

## 2. New tests added (3 tests)

`molmetal/tests/test_wave_budget_tracker.py` — Test 13:

* `test_no_enforce_budget_flag` — end-to-end CLI smoke. 4 waves ×
  total 14.0h against a 10.0h budget with `--no-enforce-budget` →
  `deferred_waves == []`, every status is `PLANNED`, and
  `honest_framing` mentions both `--no-enforce-budget` and
  `user opted out`.
* `test_budget_enforced_when_flag_off` — default behaviour (no flag)
  with the same 4-wave / 10h-budget shape → `Wave-4` is `DEFERRED`,
  `honest_framing` mentions `exceeded` and `DEFERRED`, but NOT
  `--no-enforce-budget`.
* `test_no_enforce_budget_default_false` — `parse_argv` round-trip:
  default is `False` (enforcement ON), explicit `--no-enforce-budget`
  flips to `True`.

## 3. Test run

```
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_wave_budget_tracker.py -v 2>&1 | tail -20
```

Result: **31 passed, 1 warning in 104.50s (0:01:44)** — all tests pass
including the 2 new + 1 default-flag spot-check.

## 4. `--help` smoke

```
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wave_budget_tracker.py --help 2>&1 | grep -A 1 "no-enforce"
```

Output (excerpt):

```
                              [--no-default-glob] [--no-enforce-budget]
                              [--verbose]
...
  --no-enforce-budget   Disable DEFERRED labeling for waves that exceed the
                        stated budget. All waves stay PLANNED; honest_framing
                        still reports the over-budget figure. Per user
                        2026-09-28 '时间预算没关系的': budget is a tracker, not a hard
```

Exit code: **0** (clean).

## 5. Honest framing on opt-out

`summarise()` branch when `enforce_budget=False`:

```
10.0h budget exceeded by 4.0h (--no-enforce-budget; user opted out
of DEFERRED labeling); all 4 wave(s) PLANNED.
```

This preserves the PROJECTED-vs-MEASURED ledger (over-budget figure is
still reported) while respecting the user's "时间预算没关系的" opt-out
directive (Wf-262 §0.5 honest-gap fix).

## 6. Files touched

* `/home/hugo/codes/mol-metal/molmetal/scripts/wave_budget_tracker.py` — flag already present at
  `parse_argv()` line ~791; wiring in `main()` at line ~888.
* `/home/hugo/codes/mol-metal/molmetal/tests/test_wave_budget_tracker.py` — added 3 tests
  under Test 13 (Wf-285C §2).
* `/home/hugo/codes/mol-metal/molmetal/reports/wf285c_no_enforce_budget/flag_report.md` — this report.

## 7. Verification summary

* `flag_added=True` — `--no-enforce-budget` parsed correctly via `parse_argv()`.
* `n_tests_pass=31/31` — full suite passes including the 3 new tests.
* `help_exits_0=True` — `--help` exits cleanly (rc=0).
* `enforcement_verified=True` — default behaviour still marks `DEFERRED`;
  `--no-enforce-budget` flips to all-`PLANNED`.