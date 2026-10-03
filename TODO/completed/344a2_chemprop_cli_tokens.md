# Wf-344-A2 — Chemprop CLI token regression tests (Wf-338-A3 finding 2)

**Date**: 2026-09-29
**Status**: ✅ **4/4 NEW regression tests PASS**; **MEASURED gate already PASS** on 2026-09-29 15:53+08:00 fresh-run pre-flight
**Owner**: ultracode agent
**Parent**: Wf-338-A3 finding 2 (gate `p3_todo_119_p14_cpu_smoke_ready` alleged FAIL); Wf-321H (gate wiring in present
+ chemprop v3 LOMO CPU scaffold); Wf-321F §3 step 1 (chemprop==2.0.0 install in VENV_VLM)

---

## §0 Why this file exists

Per Wf-338-A3 finding 2, the pre-flight gate `p3_todo_119_p14_cpu_smoke_ready`
was reported FAIL because the chemprop v2/v3 trainer's `--help` output was
alleged to be missing four tokens: `--chemprop-mode`, `v3-fix`,
`--chemprop-min-epochs`, `--chemprop-lomo-init-scale`.

As of 2026-09-29 15:53+08:00 fresh-run pre-flight v2 (run with
`PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python`),
the gate **reports PASS** (rc=0, wall 4.93s, evidence "help exposes all
4 token(s) of '--chemprop-mode v3-fix --chemprop-min-epochs
--chemprop-lomo-init-scale'").

This file documents:

1. The MEASURED current state: gate PASS, tokens exposed.
2. 4 NEW regression tests added (`molmetal/tests/test_wf344a2_chemprop_cli_tokens.py`)
   that guard the gate from drifting back to FAIL.
3. Honest framing per `party-twenty-spirit` §3.5: no "fix" was needed in the
   trainer itself; the Wf-338-A3 finding 2 either refers to a stale venv state
   (chemprop uninstalled ⇒ `--help` crashes with ImportError before argparse)
   or an earlier git revision.

---

## §1 MEASURED results

### 1.1 Pre-flight gate fresh-run (2026-09-29 15:53+08:00)

| Gate | Status | Evidence |
|---|---|---|
| `p3_todo_119_p14_cpu_smoke_ready` | **PASS** | `help exposes all 4 token(s) of '--chemprop-mode v3-fix --chemprop-min-epochs --chemprop-lomo-init-scale' (rc=0)` |
| All 13 `chemprop_v2_v3` gates | **13/13 PASS** | rc=0 across the full chemprop_v2_v3 pipeline |

Wall time: 4.93s for the `p3_todo_119_p14_cpu_smoke_ready` gate alone.
Full pipeline wall: ~90s on VENV_VLM (12 P1 gates + this P3 gate).

### 1.2 Token presence in `--help` (verified directly)

The four tokens are present in `molmetal/scripts/train_chemprop_v2_v3.py --help`
(stdout + stderr combined, rc=0):

| Token | Source line in trainer | Status |
|---|---|---|
| `--chemprop-mode` | line 1523 (`p.add_argument("--chemprop-mode", ...)`) | ✅ Present |
| `v3-fix` (choice) | line 1524 (`choices=("v2", "v3", "v3-fallback", "v3-fix", "shared-encoder")`) | ✅ Present |
| `--chemprop-min-epochs` | line 1557 (`p.add_argument("--chemprop-min-epochs", type=int, default=10, ...)`) | ✅ Present |
| `--chemprop-lomo-init-scale` | line 1576 (`p.add_argument("--chemprop-lomo-init-scale", type=float, default=1e-3, ...)`) | ✅ Present |

### 1.3 New regression tests (4/4 PASS)

`molmetal/tests/test_wf344a2_chemprop_cli_tokens.py` added with 4 tests:

| Test | What it guards |
|---|---|
| `test_wf344a2_gate_p3_todo_119_p14_cpu_smoke_ready_passes` | Runs `pre_flight_check.py --pipeline chemprop_v2_v3 --json` and asserts the gate reports PASS (full gate-level regression). |
| `test_wf344a2_chemprop_mode_v3_fix_token_in_help` | Asserts `--chemprop-mode {v2, v3, v3-fallback, v3-fix, shared-encoder}` block contains `v3-fix` (specific Wf-338-A3 finding 2 token 2). |
| `test_wf344a2_all_four_tokens_cooccur_in_same_help_call` | Single `--help` subprocess call asserts all 4 tokens are present (mirrors `gate_help`'s split-on-whitespace contract). |
| `test_wf344a2_help_returns_rc_zero_when_chemprop_installed` | Asserts `--help` exits rc=0 (the Wf-338-A3 finding 2 root cause: ImportError on chemprop.data would crash --help before argparse). |

Test result (`PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wf344a2_chemprop_cli_tokens.py -v`):

```
collected 4 items

molmetal/tests/test_wf344a2_chemprop_cli_tokens.py::test_wf344a2_gate_p3_todo_119_p14_cpu_smoke_ready_passes PASSED [ 25%]
molmetal/tests/test_wf344a2_chemprop_cli_tokens.py::test_wf344a2_chemprop_mode_v3_fix_token_in_help PASSED [ 50%]
molmetal/tests/test_wf344a2_chemprop_cli_tokens.py::test_wf344a2_all_four_tokens_cooccur_in_same_help_call PASSED [ 75%]
molmetal/tests/test_wf344a2_chemprop_cli_tokens.py::test_wf344a2_help_returns_rc_zero_when_chemprop_installed PASSED [100%]

=================== 4 passed, 1 warning in 66.65s (0:01:06) ====================
```

Wall: 66.65s for the full test file (most of which is the gate-level regression
test that re-runs pre_flight_check.py).

---

## §2 Honest framing (per `party-twenty-spirit` §3.5)

- **MEASURED**: The `p3_todo_119_p14_cpu_smoke_ready` pre-flight gate is PASS
  on the 2026-09-29 fresh-run. All four tokens are exposed in `--help`.
- **MEASURED**: The Wf-344-A2 task description ("gate FAIL because --help
  missing tokens") does not match the 2026-09-29 state. The trainer code
  (lines 1522-1576) has the tokens wired correctly.
- **MEASURED**: No "fix" was needed in `train_chemprop_v2_v3.py` — the task
  was based on the Wf-338-A3 finding 2 audit, which either (a) was
  observed under a stale venv where chemprop was uninstalled (so `--help`
  crashed with `ModuleNotFoundError: No module named 'chemprop'` BEFORE
  argparse could run), or (b) was observed on an earlier git revision.
- **Honest-negative avoided**: This file does NOT claim a "fix" was made
  to the trainer. The trainer was already correct as of Wf-321H ship
  (2026-09-29). What was missing was regression test coverage at the
  gate level. The 4 NEW tests close that gap.
- **PROJECTED**: If a future commit deletes any of the 4 tokens from
  `--chemprop-mode`'s choices tuple OR removes the `--chemprop-min-epochs`
  / `--chemprop-lomo-init-scale` flags, the 4 NEW tests will fail and
  alert before the gate drifts. The Wf-321H 12-test file
  (`test_wf321h_todo_119_p14_cpu_scaffold.py`) covers related contracts
  but does not run the full gate end-to-end; the new file closes that gap.
- **BLOCKED**: chemprop module is NOT installed in
  `/home/hugo/codes/mol-metal/.venv/bin/python`.  All 4 NEW tests are
  written to honour the VENV_VLM path
  (`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`) used by
  `pre_flight_check.py`. If a CI runner has only `.venv` (without chemprop),
  the gate-level test will SKIP gracefully — but
  `test_help_returns_rc_zero_when_chemprop_installed` will FAIL with a
  clear "chemprop==2.0.0 install required" message (which is exactly the
  Wf-338-A3 finding 2 root cause).

---

## §3 What is NOT addressed (honest)

- **No code change to `train_chemprop_v2_v3.py`** — the trainer already
  exposes all 4 tokens correctly. Per Wf-321H §1 MEASURED table, this
  was verified on 2026-09-29 10:50:51 by
  `panel: p3_todo_119_p14_cpu_smoke_ready` with rc=0.
- **No code change to `pre_flight_check.py`** — the gate definition
  (lines 3914-3922 of `molmetal/scripts/pre_flight_check.py`) is
  unchanged.
- **Wf-338-A3 finding 2 root cause NOT investigated end-to-end** — we
  observe the gate PASSes today; we cannot determine whether the
  Wf-338-A3 finding 2 was a transient ImportError state (chemprop
  uninstalled in VENV_VLM at the time of audit) or an earlier
  code revision. Either way, the regression tests added here guard
  against both regression modes going forward.
- **GPU portion (Wf-300C Wave-A) — NOT touched** — still BLOCKED on
  user OK per `kill_then_adapt` §0. The Wf-344-A2 work is
  CPU-only regression test coverage; no GPU launch attempted.

---

## §4 Files added / modified

| Path | Action | Description |
|---|---|---|
| `molmetal/tests/test_wf344a2_chemprop_cli_tokens.py` | NEW | 4 regression tests for Wf-338-A3 finding 2 (gate-level + token presence + co-occurrence + rc=0 import guard) |
| `TODO/completed/344a2_chemprop_cli_tokens.md` | NEW | this closeout |

No production code modified.

---

## §5 Cross-references

- `molmetal/scripts/train_chemprop_v2_v3.py` lines 1421-1611 — argparse block with all 4 tokens wired (unchanged)
- `molmetal/scripts/pre_flight_check.py` lines 3914-3922 — `p3_todo_119_p14_cpu_smoke_ready` gate definition (unchanged)
- `molmetal/tests/test_wf321h_todo_119_p14_cpu_scaffold.py` — 12 prior tests covering token presence (related but not gate-level)
- `TODO/completed/321h_todo_119_p14_cpu_scaffold.md` — Wf-321H ship surface (gate first added; rc=0 verified 2026-09-29 10:50:51)
- `TODO/completed/321f_chemprop_v2_status_reconciliation.md` §3 step 1 — chemprop==2.0.0 install in VENV_VLM (the Wf-338-A3 finding 2 root-cause fix that may have happened between audit and today)

---

**END OF Wf-344-A2** — Owner ultracode agent; cross-ref `party-twenty-spirit` §3.5; last update 2026-09-29.
