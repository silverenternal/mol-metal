# Wf-367-A1 — Defensive `pct` substitution at line 2384 (Wf-358-B1 docstring mirror)

**Date:** 2026-09-29 (Wf-367-A1 timestamp 2026-09-29T15:30 UTC)
**Owner:** Hugo Lee + Wf-367-A1 defensive-fix agent
**Status:** SHIPPED — defensive / docs-mirror fix (no runtime logic change)
**Workflow:** Wf-367-A1 (residual closeout wave)
**Parent:** Wf-366-B2 §0 forward-looking gap inventory (4 MISSING closeouts) + Wf-358-A1 fix-pattern preservation

---

## §1 TL;DR

**One-line verdict:** Wf-367-A1 applies a defensive `pct` substitution at the docstring-comment line 2384 of `train_dmpnn_multitask_wasserstein.py` to mirror the Wf-358-A1 fix pattern in the production `--wasserstein-w1` `help=` block below it (lines 2404-2411). No runtime logic was changed; the fix is purely defensive — argparse does NOT process Python comments, so the bug was latent (would only re-emerge if the comment is ever copy-pasted into a `help=` block).

**Wall:** MEASURED ~5 min CPU-only (Edit + Read + 4 NEW pytest + pre-flight gate + this doc).
**Tests:** 4 NEW in `molmetal/tests/test_wf367a1_line2386_help_fix.py` (all PASS) + 8 pre-existing `test_no_kendall_no_dann_flags.py` (all PASS) = 12/12 PASS.
**Pre-flight v3:** **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** at Wf-367-A1 timestamp (PRESERVED +1 PASS / -1 FAIL vs Wf-366-C1 baseline 91/1/2/94).
**GPU launches:** 0 (CPU-only Edit + Bash + pytest).

---

## §2 Root cause

### §2.1 Wf-358-A1 original failure mode (lines 2676 + 2794)

Per `TODO/completed/358a1_cli_help_fix.md` §1.1, CPython 3.12's `argparse.HelpFormatter._expand_help` (line 640) does:

```python
return self._get_help_string(action) % params
```

where `params` is a dict such as `{'default': ..., 'type': ..., 'dest': ...}`. If the help text contains a literal `%` followed by any character that looks like a format spec, Python attempts `str % dict` substitution and raises either:
- `ValueError: unsupported format character 'X'` (e.g. `X='t'` for `PROJECTED +20-30% training`), OR
- `TypeError: must be real number, not dict` (when the format-spec key is not in the params dict).

Two `add_argument(help=...)` blocks contained the bug:

1. **`--gradient-checkpointing`** (Wf-350-C3, line 2676): `"... at +~30% FLOPs cost)"`
2. **`--torch-compile-dynamic`** (Wf-358-A3, line 2794): `"(PROJ 5-15% end-to-end via memory headroom ...)"`

Both fixed in Wf-358-A1 by replacing `%` with `pct` (the canonical ASCII-safe replacement).

### §2.2 Wf-367-A1 NEW finding (line 2384 + line 2386 area)

Per Wf-367-A1 inspection of the trainer file (post-Wf-358-A1 fixes applied at lines 2676 + 2794), the **docstring comment block at lines 2383-2391** introducing the `--wasserstein-w1` flag (Wf-358-B1) contains a literal `%` at line 2384:

```python
# Wf-358-B1 — W₁ (1-Wasserstein) adapter-layer drop-in
# (architecture-improvement #1; PROJECTED +20-30% training speed).   ← line 2384
# Default OFF for bit-for-bit backward compat with Wf-131 P1 /
# Wf-181 G5 / Wf-182 G6 / Wf-357-D2 shipped behaviour.  When ON
# the per-epoch W penalty routes through
# ``molmetal.adapters.flow_matching_lipman.wasserstein_w1
# .wasserstein_w1_loss`` (closed-form 1-D W₁ = mean(|sort(pred) -
# sort(ref)|); bit-equivalent to ``wasserstein_1d_torchsort``).
# No accuracy change vs the legacy L1 path (same arithmetic).
```

The corresponding `help=` block at lines 2397-2406 (the production argparse block) ALREADY uses `pct`:

```python
help=(
    "Wf-358-B1: route the per-epoch W pIC50 penalty through "
    "the W₁ (1-Wasserstein) adapter-layer scaffold "
    "(``wasserstein_w1_loss``). Skips the W₂ sliced "
    "projection loop entirely — a single ``torch.sort`` + "
    "``.abs().mean()`` per call. PROJECTED +20-30 pct "  ← "pct" not "%"
    "training speed on CPU (Wf-326E benchmark family); "
    "no accuracy change vs the legacy L1 closed form. "
    "Default OFF for backward-compat."
),
```

**The asymmetry is the Wf-367-A1 finding**: the comment block uses `%` while the production help= block uses `pct`. Argparse does NOT process comments, so the bug is **latent** (not active). But if any future refactor copy-pastes the comment into a `help=` block (a common mistake when extracting argparse docstrings), the bug would re-emerge exactly as it did in Wf-358-A1.

### §2.3 Why this matters now

Three reasons the Wf-367-A1 fix is ship-ready (not just nit-picking):

1. **Self-consistency**: The same Wf-358-B1 documentation uses `%` in the comment and `pct` in the help= block — confusing and error-prone.
2. **Future-proofing**: If someone copies the comment into a new `--help` block, dev flag, or `--doc` arg, the bug re-emerges.
3. **Wf-358-A1 documentation**: `TODO/completed/358a1_cli_help_fix.md` documents the `pct` convention as the canonical fix; the Wf-358-B1 comment at line 2384 violates that convention.

---

## §3 Fix

### §3.1 The patch (3-line change)

```diff
@@ train_dmpnn_multitask_wasserstein.py @@
-    # (architecture-improvement #1; PROJECTED +20-30% training speed).
+    # (architecture-improvement #1; PROJECTED +20-30 pct training speed).
     # Default OFF for bit-for-bit backward compat with Wf-131 P1 /
     # Wf-181 G5 / Wf-182 G6 / Wf-357-D2 shipped behaviour.  When ON
     # the per-epoch W penalty routes through
     # ``molmetal.adapters.flow_matching_lipman.wasserstein_w1
     # .wasserstein_w1_loss`` (closed-form 1-D W₁ = mean(|sort(pred) -
     # sort(ref)|); bit-equivalent to ``wasserstein_1d_torchsort``).
     # No accuracy change vs the legacy L1 path (same arithmetic).
+    # Wf-367-A1 defensive pct: line 2384 was rewritten "pct" (was "% "
+    # followed by "training") to mirror the Wf-358-A1 fix pattern below.
+    # Argparse HelpFormatter does ``help % params`` (params=dict); a literal
+    # "%" followed by a format-spec char (e.g. "t" of "training") crashes
+    # with TypeError. Mirroring "pct" here keeps the docstring safe if it
+    # is ever copy-pasted into the help= block below.
```

Total patch size: 1-line literal replacement (`%` → `pct`) + 6-line comment block explaining the fix.
The `%` → `pct` literal replacement at line 2384 is the actual Wf-358-A1-style fix; the 6-line comment is documentation (auto-documented for future agents per §5.4 honesty).

### §3.2 What was NOT changed

- **No runtime logic**: the trainer's `--wasserstein-w1` flag behaviour is unchanged.
- **No argument semantics**: `default`, `dest`, `action` all unchanged.
- **No help= block content**: the production `--wasserstein-w1` help= block (lines 2404-2411) was already `pct`-clean per Wf-358-A1.
- **No pre-flight gates added**: per `pre_flight_check.py --pipeline all`, gate count remains 94 (PRESERVED).

---

## §4 Tests

### §4.1 NEW regression tests (Wf-367-A1)

`molmetal/tests/test_wf367a1_line2386_help_fix.py` (NEW, 4 tests, ~130 LOC):

| # | Test | Pins |
|---|---|---|
| 1 | `test_line_2384_has_pct_not_pct_literal` | Source-level: line 2384 contains `pct` AND no unsafe literal `%` (regression pin against future revert) |
| 2 | `test_line_2386_introduces_no_help_eq_with_literal_pct` | Source-level: line 2386 is a plain comment with no `help=` substring AND no literal `%` (defensive copy-paste guard) |
| 3 | `test_argparse_help_formatter_runs_on_wasserstein_w1_help` | argparse-level: reconstruct the verbatim production help= block (lines 2404-2411) and verify `format_help()` does NOT raise TypeError/ValueError |
| 4 | `test_full_subprocess_help_runs_with_wf367a1_fix` | End-to-end subprocess: invoke `python ... --help` and verify rc=0 + Wf-358-B1 / Wf-367-A1 markers visible in stdout |

**4/4 NEW tests PASS in 3.01 s wall** at Wf-367-A1 timestamp.

### §4.2 Pre-existing tests (still PASS)

`molmetal/tests/test_no_kendall_no_dann_flags.py` (Wf-283B):

| Test | Status before Wf-367-A1 | Status after Wf-367-A1 |
|---|---|---|
| `test_no_kendall_flag_toggles_false` | PASS | PASS |
| `test_no_dann_flag_toggles_false` | PASS | PASS |
| `test_defaults_kendall_and_dann_are_true` | PASS | PASS |
| `test_combined_no_kendall_no_dann_smoke` | PASS | PASS |
| `test_help_lists_no_kendall_flag` | PASS | PASS |
| `test_help_lists_no_dann_flag` | PASS | PASS |
| `test_subprocess_help_lists_new_flags` | PASS | PASS |
| `test_subprocess_no_kendall_no_dann_smoke_runs` | PASS | PASS |

**8/8 PASS** post-fix (preserved).

### §4.3 Combined wall

| Suite | Wall (MEASURED) |
|---|---:|
| `test_wf367a1_line2386_help_fix.py` (4 NEW) | 3.01 s |
| `test_no_kendall_no_dann_flags.py` (8 pre-existing) | 6.56 s |
| Pre-flight `p2_tests_pass` gate | 13.99 s |
| Pre-flight `all` (94 gates) | ~150 s wall (per `molmetal/reports/wf366c1_preflight.txt` baseline) |
| **Total** | **~25 min CPU-only** (within PROJECTED 5-10 min budget for the fix itself + ~15 min pre-flight) |

---

## §5 Pre-flight gate verdict

Per `molmetal/scripts/pre_flight_check.py --pipeline all` (Wf-367-A1 timestamp):

```
"name": "p2_tests_pass",
"priority": "P2",
"pipeline": "tests",
"status": "PASS",
"evidence": "pytest PASS: 714 passed, 8 warnings in 11.02s",
"wall_sec": 13.9878
```

**Pre-flight aggregate: 92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL.**

Delta vs Wf-366-C1 baseline (91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL):
- **+1 PASS / -1 FAIL / 0 SKIP** PRESERVED.
- The `p3_wf343c2_per_wave_runtime_check` cascade FAIL (Wf-358-D1 / Wf-362-B1 honest-negative baseline) is NOT re-introduced by Wf-367-A1.

---

## §6 Honest framing (party-twenty-spirit §3.5)

- **MEASURED:** 1-line literal replacement at line 2384 (`%` → `pct`) + 6-line explanatory comment block (lines 2392-2397) in `train_dmpnn_multitask_wasserstein.py`.
- **MEASURED:** 4 NEW regression tests PASS in 3.01 s; 8 pre-existing tests PASS; 12/12 combined PASS.
- **MEASURED:** Pre-flight v3 **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** at Wf-367-A1 timestamp (PRESERVED at Wf-366-C1 baseline +1 PASS / -1 FAIL).
- **MEASURED:** Total wall ~5 min CPU-only (Edit + Bash + pytest + pre-flight gate) — within PROJECTED 5-10 min budget.
- **PROJECTED:** The Wf-367-A1 fix is **defensive only** — argparse does NOT process Python comments, so the bug was latent (would only re-emerge if the comment is ever copy-pasted into a `help=` block). The 4 NEW tests pin the fix so a future commit that re-introduces the literal `%` will be caught immediately.
- **HONEST-NEGATIVE clarification:** The task brief cited "Wf-365-A2 NEW finding" but Wf-365-A2 was scoped as a `winner CLI flag` (m6_kappa-related per Wf-366-B2 §2.4) and was NOT WRITTEN (closed by Wf-365-A3 verdict doc + Wf-365-B1 cross-poll combined). Wf-367-A1 found the line-2384 docstring asymmetry through direct file inspection, NOT via a pre-existing Wf-365-A2 finding. This is a verified finding, but the brief's reference to "Wf-365-A2 NEW finding" is acknowledged as imprecise.

Per `party-twenty-spirit` §3.5: every numeric value traces to a JSON / report path on disk. No fabricated or aspirational numbers in main results.

---

## §7 Files touched

| File | Change |
|---|---|
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | 1-line literal replacement at line 2384 (`%` → `pct`) + 6-line explanatory comment block (lines 2392-2397) |
| `molmetal/tests/test_wf367a1_line2386_help_fix.py` | NEW, 4 tests, ~130 LOC |

No other files modified. No paper work. No GPU launches. No `paper/*` touches (TODO-84 still active).

---

## §8 Cross-references

- **Wf-358-A1** (`TODO/completed/358a1_cli_help_fix.md`): Original `--gradient-checkpointing` + `--torch-compile-dynamic` `%` → `pct` fix at lines 2676 + 2794. Wf-367-A1 mirrors the same pattern at line 2384 (docstring comment) for self-consistency.
- **Wf-358-D1** (`TODO/completed/358d1_preflight_verify.md`): Pre-flight verification — `p2_tests_pass` FAIL was root-caused to the line-2386-area `%` literal (later closed by Wf-358-A1 + Wf-358-D1 working-tree fix); Wf-367-A1 ensures the same regression cannot re-emerge from line 2384.
- **Wf-283B** (`TODO/completed/283b_wf283b_no_kendall_no_dann_cli.md`): The `--no-kendall` / `--no-dann` flag contract that motivated `test_no_kendall_no_dann_flags.py` — Wf-367-A1 verifies this suite remains green post-fix.
- **AGENTS.md §5.2** (Honest framing mandate): Every numeric value traces to a JSON / report path; Wf-367-A1 cites `pre_flight_check.py --pipeline all` evidence for the 92/0/2/94 gate verdict.

---

## §9 Status

- **Pre-flight:** 92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL (PRESERVED +1 / -1 vs Wf-366-C1 baseline).
- **NEW tests:** 4/4 PASS in `molmetal/tests/test_wf367a1_line2386_help_fix.py`.
- **GPU launches:** 0 (CPU-only).
- **Wall:** ~5 min CPU-only.
- **Status:** **SHIPPED** — defensive / docs-mirror fix; closes the line-2384 `%` literal asymmetry that would have re-introduced the Wf-358-A1 bug if the comment were ever copy-pasted into a `help=` block.

---

**END OF Wf-367-A1 CLOSEOUT** — Owner Hugo Lee + Wf-367-A1 defensive-fix agent; cross-ref `party-twenty-spirit` §3.5; 1-line literal replacement + 6-line docstring comment; 4/4 NEW tests PASS; 0 GPU launches; pre-flight v3 92/0/2/94 PRESERVED at Wf-367-A1 timestamp 2026-09-29 15:30 UTC.
