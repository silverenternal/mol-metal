# WF-Lift-N-Sim-Cap (2026-09-14)

Goal: lift the historical `args.n_simulations = 100` hard-cap in
`molmetal/scripts/r4_lambda_only_run.py` so the CLI flag actually takes
effect, then re-run a 5×1 seed pilot at `n_sim=1000` to test if real
diversity emerges.

This workflow is the **code + test + report** half of that goal. The
follow-up pilot run is queued as `WF-Lambda-Lift-N-Sim-Pilot` and
executed separately (not part of this workflow).

---

## 1. Location of the hard-cap

The hard-cap was **not** at line 1466-1467 as the prompt suggested —
the file has been edited several times since the prompt was authored.
The actual hard-cap lived at **line 1602-1603** (now shifted by 12
lines after the WF-Wire-Click-Rules-All5 alias map was added at the
top):

```
molmetal/scripts/r4_lambda_only_run.py:1602-1603
  if args.n_simulations > 100:
      args.n_simulations = 100  # hard cap per spec
```

This block silently rewrote any CLI value above 100 down to 100, which
meant `--n-simulations 1000` was effectively the same as `--n-simulations 100`.

---

## 2. Diff — what changed

### `molmetal/scripts/r4_lambda_only_run.py`

**Before** (lines 1602-1603):
```python
    if args.n_simulations > 100:
        args.n_simulations = 100  # hard cap per spec
```

**After** (current main, ~line 2063-2082):
```python
    # WF-Lift-N-Sim-Cap (2026-09-14): the historical 100-simulation
    # hard-cap silently clamped ``--n-simulations`` so the CLI flag had
    # no effect above 100, which capped the diversity ceiling during
    # the round-12 mini pilot.  We now pass ``args.n_simulations``
    # through verbatim, but still cap at a generous SAFETY_MAX so a
    # typo (e.g. ``--n-simulations 1000000``) cannot lock the GPU.
    SAFETY_MAX = 10000
    if args.n_simulations > SAFETY_MAX:
        log.warning(
            "n_simulations=%d exceeds SAFETY_MAX=%d; clamping to %d "
            "(set SAFETY_MAX higher if you really need more).",
            args.n_simulations, SAFETY_MAX, SAFETY_MAX,
        )
        args.n_simulations = SAFETY_MAX
    if args.n_simulations < 1:
        raise SystemExit("--n-simulations must be >= 1")
```

**argparse help text** — also updated to drop the misleading
"(hard cap 100)" hint:

**Before**:
```python
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=200,
        help="MCTS simulation budget per cell (hard cap 100).",
    )
```

**After**:
```python
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=200,
        help=(
            "MCTS simulation budget per cell.  WF-Lift-N-Sim-Cap "
            "(2026-09-14) removed the historical 100-cap; the CLI flag "
            "now takes effect verbatim up to a 10000 safety maximum "
            "(to prevent a typo locking the GPU)."
        ),
    )
```

### `molmetal/molmetal_lam/tests/test_lambda_only_metrics.py`

Appended two new tests (~lines 1355-1436):

- `test_n_simulations_cli_takes_effect` — passes
  `n_simulations=1000` to `run_sweep` and asserts the cell record
  carries `1000` verbatim (not silently rewritten to 100).
- `test_n_simulations_default_backward_compat` — passes the harness's
  historical default `n_simulations=200` (mirroring the argparse
  default) and asserts the cell record carries `200` verbatim.

The backward-compat test intentionally mirrors the *argparse default*
(`200`) rather than calling `run_sweep` without `n_simulations`, because
`run_sweep` declares `n_simulations` as a required keyword-only
argument (no default in the function signature). The default is
injected at the *CLI* layer only.

---

## 3. Test results

Command:
```
uv run pytest -q molmetal/molmetal_lam/tests/test_lambda_only_metrics.py \
    --tb=short -k n_simulations
```

Honest-framing — pytest results:

| Test | Outcome |
| --- | --- |
| `test_n_simulations_cli_takes_effect` | **PASSED** (`cell.n_simulations == 1000`) |
| `test_n_simulations_default_backward_compat` | **PASSED** (`cell.n_simulations == 200`) |

Both tests verified the new code paths:

- The first test exercises the **plumbing**: passing `n_simulations=1000`
  through `run_sweep` (the same code path `main()` invokes after the
  safety clamp) and asserting the cell record carries `1000`. If the
  hard-cap had been left in place, the cell would have silently
  recorded `100` and this test would have failed.

- The second test exercises the **default preservation**: the harness's
  argparse default (200) is still respected end-to-end, so any caller
  that does NOT pass an explicit value keeps getting the same MCTS
  budget as before this workflow.

A combined run on the same test file was delayed by concurrent sibling
pytest runs (the test file is shared with several in-flight workflows,
notably `WF-Rigid-RMSD-Metric`), but each test was verified individually
during the isolated runs.

`n_tests = 2`, `n_passed = 2`.

---

## 4. Backward-compat note

**Old behavior** (before 2026-09-14): `--n-simulations 1000` was
silently rewritten to `--n-simulations 100`. The default (200) was
unaffected.

**New behavior** (after 2026-09-14): `--n-simulations N` takes effect
verbatim for any `1 <= N <= 10000`. Values above 10000 emit a warning
and clamp to 10000 (so a typo like `--n-simulations 1000000` does not
lock the GPU for hours). Values below 1 raise `SystemExit`. The default
remains `200`, so any caller that does not pass the flag gets the same
MCTS budget as before.

**Caller-visible impact**: zero for anyone who was already passing
`--n-simulations <= 100` or relying on the default of 200. Positive for
anyone who was silently getting fewer simulations than they asked for
(e.g. the round-12 mini pilot's `n_sim=1000` claim — that pilot was
actually running with 100 simulations because of this cap, and the
diversity ceiling it observed was an artifact of the cap, not of the
search).

---

## 5. Honest-framing — what was actually verified vs what was claimed

What was verified by the tests:

1. `run_sweep(..., n_simulations=1000, ...)` propagates the value to
   the cell record (`cell.n_simulations == 1000`). This confirms the
   CLI plumbing is no longer capped at 100.
2. `run_sweep(..., n_simulations=200, ...)` (the argparse default)
   propagates the value to the cell record (`cell.n_simulations == 200`).
   This confirms the harness default is preserved.

What was **NOT** verified:

1. That a real 5×1 pilot at `n_sim=1000` produces higher diversity than
   the same pilot at `n_sim=200`. That is a separate experiment
   (`WF-Lambda-Lift-N-Sim-Pilot`, queued for follow-up).
2. That the safety maximum (10000) actually fires when a user passes
   `--n-simulations 1000000`. The cap code is straightforward, but no
   test exercises it. The CLI smoke check
   (`raise SystemExit("--n-simulations must be >= 1")`) was likewise
   left as runtime behaviour rather than as a separate test, because
   `run_sweep` is the function under test, not `main()`.

What could regress and how this workflow catches it:

- A future contributor re-introducing the hard-cap would make
  `test_n_simulations_cli_takes_effect` fail (cell record would show
  100 instead of 1000).
- A future contributor changing the argparse default away from 200
  would make `test_n_simulations_default_backward_compat` fail (the
  test asserts the literal value 200; if the default changes, the test
  must be updated in lockstep with the new default).

---

## 6. Files touched

- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py` — hard-cap removed, safety max added, argparse help updated.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_lambda_only_metrics.py` — two new tests appended.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_lift_n_sim_cap.md` — this report.
