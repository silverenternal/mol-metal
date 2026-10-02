# Wf-358-A1 — CLI help-text `%`-format bug fix

**Date:** 2026-09-29
**Owner:** Hugo Lee (assisted by Claude Code)
**Status:** SHIPPED
**Workflow:** Wf-358-A1 (ultracode residual closeout wave)

---

## §1 Root cause

`molmetal/scripts/train_dmpnn_multitask_wasserstein.py` raised
`TypeError: must be real number, not dict` whenever the trainer's
`argparse.ArgumentParser` was asked to format `--help` output.

### §1.1 The exception

CPython 3.12's `argparse.HelpFormatter._expand_help` (line 640) does::

    return self._get_help_string(action) % params

where `params` is a dict such as `{'default': ..., 'type': ...,
'dest': ...}`. If the help text contains a literal `%` followed by any
character that looks like a format spec, Python attempts
`str % dict` substitution and raises `TypeError: must be real number,
not dict` because the format-spec key is not in the dict.

### §1.2 The two offending help strings

Two `add_argument(help=...)` blocks contained literal `%` characters
that triggered the bug:

1. **`--gradient-checkpointing`** (Wf-350-C3, line 2676):
   `"... at +~30% FLOPs cost)"` — `30%` is parsed as a format spec.
2. **`--torch-compile-dynamic`** (Wf-358-A3, line 2794):
   `"(PROJ 5-15% end-to-end via memory headroom ...)"` — `5-15%` triggers
   the same path.

### §1.3 Impact

- `test_subprocess_help_lists_new_flags` (Wf-283B) FAIL — `--help` exited
  non-zero with the TypeError in stderr.
- `test_subprocess_no_kendall_no_dann_smoke_runs` (Wf-283B) FAIL — same
  failure mode under combined `--no-kendall --no-dann`.
- The `p2_tests_pass` pre-flight gate was blocked from going green
  because the pytest collection cycle hit the TypeError before the
  `--no-kendall` / `--no-dann` flag contract could be verified.

---

## §2 Fix

Two 1-line literal replacements (no logic changes):

```diff
@@ line 2676 (--gradient-checkpointing help string) @@
-            "(memory ~O(sqrt(depth)) at +~30% FLOPs cost). Opt in for "
+            "(memory ~O(sqrt(depth)) at plus ~30 pct FLOPs cost). Opt in for "

@@ line 2794 (--torch-compile-dynamic help string) @@
-            "(PROJ 5-15% end-to-end via memory headroom + amortised "
+            "(PROJ 5-15 pct end-to-end via memory headroom + amortised "
```

Total patch size: 2 lines changed (out of 154,726 LOC in the file).
No argument semantics altered — `default`, `dest`, `action` all unchanged.
The `pct` token is the canonical ASCII-safe replacement that survives
argparse's `% params` substitution without becoming a format spec.

---

## §3 Tests

### §3.1 Existing tests (now PASS)

`molmetal/tests/test_no_kendall_no_dann_flags.py` (Wf-283B):

| Test | Status before | Status after |
|---|---|---|
| `test_no_kendall_flag_toggles_false` | PASS | PASS |
| `test_no_dann_flag_toggles_false` | PASS | PASS |
| `test_defaults_kendall_and_dann_are_true` | PASS | PASS |
| `test_combined_no_kendall_no_dann_smoke` | PASS | PASS |
| `test_help_lists_no_kendall_flag` | PASS | PASS |
| `test_help_lists_no_dann_flag` | PASS | PASS |
| `test_subprocess_help_lists_new_flags` | **FAIL** | **PASS** |
| `test_subprocess_no_kendall_no_dann_smoke_runs` | **FAIL** | **PASS** |

8/8 PASS post-fix.

### §3.2 NEW regression tests (Wf-358-A1)

`molmetal/tests/test_wf358a1_cli_help_fix.py` (NEW, 90 LOC):

1. `test_subprocess_help_exit_zero_with_new_flags` — verifies
   `--help` exits 0 AND stdout contains
   `--gradient-checkpointing` (Wf-350-C3) + `--torch-compile-dynamic`
   (Wf-358-A3), the two flags that were inaccessible while the bug was
   live.
2. `test_subprocess_help_does_not_raise_typeerror` — verifies
   `--help` does NOT raise `TypeError: must be real number, not dict`
   (the exact signature from the original failure mode). Pins the
   regression so a future commit re-introducing a literal `%` in any
   `help=` string will be caught immediately.

2/2 NEW tests PASS.

---

## §4 Pre-flight gate verdict

`p2_tests_pass` gate (Wf-358-A1 closeout, fresh at 2026-09-29
13:33 UTC):

```
{
  "name": "p2_tests_pass",
  "priority": "P2",
  "pipeline": "tests",
  "status": "PASS",
  "evidence": "pytest PASS: 714 passed, 8 warnings in 10.86s",
  "wall_sec": 13.7497
}
```

Pre-flight aggregate: **91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL**.
The 1 remaining FAIL is `wf343c2_per_wave_check.py` for the
Wf-347A1 pre-retrain smoke wave — unrelated to the argparse fix.

---

## §5 Honest framing (party-twenty-spirit §3.5)

- **Patch size:** MEASURED 2 lines changed (1 line per offending help
  string).
- **Tests:** MEASURED 2 NEW + 2 fixed = 4 tests; 8/8 PASS in the
  pre-existing suite; 714/714 PASS in pre-flight `p2_tests_pass`.
- **Wall:** MEASURED ~6-7 s per pytest invocation; total ~30 s for
  the four pytest runs (including 2 fixed + 2 NEW + 1 pre-flight).
- **GPU launches:** 0 (CPU-only Edit + Bash + pytest).

---

## §6 Files touched

- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (line 2676 + line 2794)
- `molmetal/tests/test_wf358a1_cli_help_fix.py` (NEW, 90 LOC)

---

## §7 Source pointers

- Original failure: Wf-357-A1 finding (p2_tests_pass gate FAIL root-cause
  analysis).
- Wf-283B contract: `--no-kendall` / `--no-dann` flag coverage
  (`TODO/completed/283b_wf283b_no_kendall_no_dann_cli.md`).
- Wf-350-C3 origin: `--gradient-checkpointing` flag (cfm_yuelbond +
  Wasserstein joint trainer, Wf-350-C3 closeout).
- Wf-358-A3 origin: `--torch-compile-dynamic` flag (torch.compile
  dynamic_shapes guard, Wf-358-A3 closeout).

---

**END OF Wf-358-A1 CLOSEOUT**