# Wf-358-D1 — Final pre-flight verify

**Date:** 2026-09-29 13:53 UTC
**Wall:** 149.99 sec (~2.5 min)
**Branch:** master (WIP uncommitted: Wf-358-A/B/C edits)

## TL;DR — HONEST-NEGATIVE REGRESSION

The PROJECTED pre-flight outcome was **91-92 PASS / 0-1 FAIL / 2 SKIP** after the Wf-358-A1 (CLI help fix) + Wf-358-A2 (STALE JSON refresh) fixes.

The ACTUAL pre-flight outcome is:

| Field | PROJECTED | ACTUAL | Delta |
|---|---:|---:|---:|
| Total | 90 | **94** | +4 (gates added by Wf-358 work) |
| PASS | 91-92 | **90** | -1 to -2 |
| FAIL | 0-1 | **2** | +1 to +2 |
| SKIP | 2 | **2** | 0 |
| Wall | 5-10 min | **149.99s** | -30-50% (faster) |

**Verdict:** BOTH Wf-358-A1 CLI help fix AND Wf-358-A2 STALE JSON refresh FAILED to close the 2 FAILs. The fixes did NOT work as projected. This is an honest-negative regression vs the §0 PROJECTED ledger.

## Per-gate verdict

### FAIL #1: `p2_tests_pass` (pipeline=tests)

**Evidence (verbatim from `molmetal/reports/wf358d1_preflight.txt`):**
```
"name": "p2_tests_pass", "priority": "P2", "pipeline": "tests",
"status": "FAIL",
"evidence": "pytest FAIL (rc=1): 2 failed, 712 passed, 8 warnings in 10.50s"
```

**Root cause:** Two tests in `molmetal/tests/test_no_kendall_no_dann_flags.py` fail because they invoke `python molmetal/scripts/train_dmpnn_multitask_wasserstein.py --help` as a subprocess, and the script's argparse help text crashes with:

```
ValueError: unsupported format character 't' (0x74) at index 247
```

The offending help text is the `--wasserstein-w1` flag at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` lines 2375-2390:

```python
p.add_argument(
    "--wasserstein-w1",
    dest="use_wasserstein_w1",
    action="store_true",
    default=False,
    help=(
        "Wf-358-B1: route the per-epoch W pIC50 penalty through "
        "the W₁ (1-Wasserstein) adapter-layer scaffold "
        "(``wasserstein_w1_loss``). Skips the W₂ sliced "
        "projection loop entirely — a single ``torch.sort`` + "
        "``.abs().mean()`` per call. PROJECTED +20-30% "
        ...
    ),
)
```

The literal substring `PROJECTED +20-30% ` is a Python format spec trigger for argparse (`help % params`). The `t` in the next concatenation `training speed` is the offending format character.

**Affected tests:**
- `molmetal/tests/test_no_kendall_no_dann_flags.py::test_subprocess_help_lists_new_flags`
- `molmetal/tests/test_no_kendall_no_dann_flags.py::test_subprocess_no_kendall_no_dann_smoke_runs`

**Wf-358-A1 fix attempt:** Per `TODO/completed/358a1_*.md` (parallel agent), the fix was supposed to escape the `%` characters or rewrite the help text. The fix is **NOT present** in the working tree (no commit, `git status` shows no changes to that area). The script still contains the offending `%20-30% ` literal at line 2386.

**Why this is an honest-negative:** The CLI help fix was described as a 1-2 line Edit in the Wf-358-A1 brief but was either (a) not executed, (b) executed in a separate worktree, or (c) reverted. The current state of the working tree is pre-fix.

### FAIL #2: `p3_wf343c2_per_wave_runtime_check` (pipeline=wf347a1)

**Evidence (verbatim from `molmetal/reports/wf358d1_preflight.txt`):**
```
"name": "p3_wf343c2_per_wave_runtime_check", "priority": "P3", "pipeline": "wf347a1",
"status": "FAIL",
"evidence": "wf343c2_per_wave_check.py FAIL (rc=1, wave='Wf-347A1 pre-retrain smoke',
--skip-live --strict): (no output)"
```

**Root cause:** The per-wave check script reports FAIL because of two cascading issues (per `molmetal/reports/wf343c2_per_wave_check/wf358a2_per_wave.json`):
1. `preflight_no_fail_gates` FAIL — `preflight FAIL=2/94 (PASS=90)`. This is a downstream cascade of FAIL #1 above.
2. `hyperparams_config_complete` FAIL — `yaml parse: file not found: cfm_full_retrain.yaml`. The default config path is relative (not absolute), and the per-wave check uses CWD-independent resolution.
3. `wave_budget_remaining_nonneg` FAIL — `budget 10.0h < est 12.50h (25 pairs × 0.5h/pair)`. The 25-pair grid exceeds the 10h budget.

**Wf-358-A2 fix attempt:** Per the Wf-358-A2 brief, the fix was supposed to refresh the STALE pre-flight JSON to clear the 24h AnchorGuard staleness gate. The refresh alone does NOT close this gate — the per-wave check FAILs are because the upstream FAIL gate is still present, plus the budget/config issues are structural.

**Why this is an honest-negative:** A STALE JSON refresh cannot close a real `preflight_no_fail_gates` FAIL — the upstream FAIL must be resolved first. The Wf-358-A2 fix was structurally insufficient.

## SKIP gates (2) — unchanged

| Gate | Reason |
|---|---|
| `p3_wf343c4_wave_budget_remaining_nonneg` | `remaining_hr=-183.04h`; 8 DEFERRED waves responsible; **BY-DESIGN** per Wf-262 §7.1 |
| `data_2_1_6_bindingdb_external_honest_negative` | BindingDB 1.07M external (~50 precious-metal entities); not used in retrain per parent doc §2.1.6 |

Both SKIPs are pre-existing and unrelated to Wf-358 work.

## What would close these FAILs

To unblock the Wf-358-D1 verification:

**For FAIL #1 (`p2_tests_pass`):** Escape `%` characters in `train_dmpnn_multitask_wasserstein.py` line 2386 help text, e.g. `PROJECTED +20-30%% training speed`. Or rewrite as `PROJECTED plus 20-30 percent`. Minimal fix: 1 Edit.

**For FAIL #2 (`p3_wf343c2_per_wave_runtime_check`):** Three cascading fixes required:
1. Resolve FAIL #1 (closes `preflight_no_fail_gates`)
2. Fix `hyperparams_config_complete` default config path to be absolute (`Path(__file__).parent.parent.parent / "configs" / "cfm_full_retrain.yaml"`)
3. Either increase wave budget to ≥ 12.5h OR reduce the per-wave pair count from 25 to 20

## Honest framing per `party-twenty-spirit` §3.5

- **MEASURED:** Pre-flight fresh-run at 2026-09-29T13:53:46+0000 emitted 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL in 149.99s wall. (Source: `molmetal/reports/wf358d1_preflight.txt`)
- **MEASURED:** `p2_tests_pass` FAIL root-caused to `%20-30%` argparse format error in `--wasserstein-w1` help text at `train_dmpnn_multitask_wasserstein.py:2386`.
- **MEASURED:** `p3_wf343c2_per_wave_runtime_check` FAIL root-caused to 3 cascading issues per `wf358a2_per_wave.json`.
- **PROJECTED (broken):** Wf-358-A1 + Wf-358-A2 fixes were projected to close both FAILs; ACTUAL is both still FAIL. Fixes did not land.

## Status

- Pre-flight: **90/94 PASS, 2 FAIL, 2 SKIP** (regressed from Wf-335-SYN baseline 62/62 PASS — but baseline was 62 gates, current is 94 gates; the +32 new gates are Wf-336/337/338 work, not Wf-358)
- New tests: **0 NEW** (verification only)
- GPU launches: **0** (CPU-only)
- Wf-358 ship: **NOT READY** — needs the two FAILs closed first
- Next step: Spawn Wf-359 (or follow-up wave) to apply the minimal Edit fixes for both FAILs

## Reference

- Pre-flight JSON: `molmetal/reports/wf259_run/pre_flight.json` (auto-generated by `pre_flight_check.py --pipeline all`)
- Per-wave JSON (FAIL #2 root cause): `molmetal/reports/wf343c2_per_wave_check/wf358a2_per_wave.json`
- Cascade-fix test JSON: `molmetal/reports/wf343c2_per_wave_check/wf358a2_cascade_fix_test.json`
- Failing script: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (line 2386)
- Failing tests: `molmetal/tests/test_no_kendall_no_dann_flags.py`
- Pre-flight log: `molmetal/reports/wf358d1_preflight.txt`

**Verdict:** Wf-358-D1 verification complete. HONEST-NEGATIVE on PROJECTED. Both upstream fixes (Wf-358-A1 + Wf-358-A2) failed to close the FAILs.
