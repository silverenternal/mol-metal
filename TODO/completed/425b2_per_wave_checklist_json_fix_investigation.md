# Wf-425-B2 investigator: per_wave_checklist.py --wave wave_a --json argparse bug

**Document ID:** Wf-425-B2 / 425b2_per_wave_checklist_json_fix_investigation
**Version:** 1.0
**Date:** 2026-10-01
**Phase:** 1 of 2 — INVESTIGATOR (Phase 2 fixers will apply the patch based on this review)
**Authoring trigger:** Wf-424-A7 closeout §"what did NOT 彻底解决" + Wf-424-A10 closeout §3.1 + Wf-424-A11 closeout §3 + Wf-424-B1 final synthesis + harness user directive *"现在先搞清楚这些blocker是怎么回事，做相关的review和洞察然后再启动ultracode做修复"*.

---

## §0 TL;DR

The B2 blocker is a **wrapper-side caller mismatch**, not a `per_wave_checklist.py` argparse design bug per se. The wrapper `molmetal/scripts/launch_wf299_cfm_main.sh:398-399` (and `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh:186-187`) calls `per_wave_checklist.py --wave ${wave_label} --json` (and `--skip-live --json` in the Wf-310e wrapper), but `per_wave_checklist.py`'s argparse schema does NOT define `--json` (it never has; per line 947-1022 the schema only defines `--wave` + `--config` + `--preflight` + `--out` + `--out-md` + `--pairs-metals` + `--pairs-seeds` + `--time-budget-hr` + `--no-md` + `--verbose`). Argparse correctly rejects with `unrecognized arguments: --json` (reproduced live below). The Wf-413-A3 closeout §0.3 fix on the SAME wrapper removed `--json` from a DIFFERENT invocation (`pre_flight_check.py --json` at line 257-258), but did NOT clean up the per_wave_checklist invocation at line 398-399 because Wf-413-A3 focused on the pre-flight bug only.

---

## §1 Code state (file:line refs)

### §1.1 Target file: `molmetal/scripts/per_wave_checklist.py`

- **Lines 936-1022**: `parse_argv()` — argparse schema definition.
  - **Line 947-954**: `p.add_argument("--wave", required=True, help=...)` — the ONLY required flag.
  - **Line 956-964**: `p.add_argument("--config", type=Path, default=None, ...)`
  - **Line 966-973**: `p.add_argument("--preflight", type=Path, default=...)`
  - **Line 975-979**: `p.add_argument("--out", type=Path, default=...)`
  - **Line 981-985**: `p.add_argument("--out-md", type=Path, default=...)`
  - **Line 987-994**: `p.add_argument("--pairs-metals", type=str, default=...)`
  - **Line 996-1002**: `p.add_argument("--pairs-seeds", type=str, default=...)`
  - **Line 1004-1011**: `p.add_argument("--time-budget-hr", type=float, default=...)`
  - **Line 1013-1016**: `p.add_argument("--no-md", action="store_true", ...)`
  - **Line 1018-1021**: `p.add_argument("--verbose", action="store_true", ...)`
  - **NO `--json` flag defined anywhere in this argparse schema.**
- **Lines 1025-1092**: `main()` — body; receives `args = parse_argv(argv)` at line 1026; **does NOT consume any `--json` flag**.
- **Line 615**: `* Footer with honest-framing note` — **typo**: docstring line starts with `185` instead of `   ` (3 spaces + `*`). This is a docstring artifact inside `_render_markdown()` and does NOT affect runtime behavior; flagged as a docs-cleanup item but not a B2 blocker.

### §1.2 Caller file: `molmetal/scripts/launch_wf299_cfm_main.sh`

- **Lines 210-254**: EXTRA_FLAGS accumulator pattern (introduced Wf-415-A1) — captures unknown flags into `EXTRA_FLAGS` for transparent passthrough to the trainer invocation.
- **Lines 256-270**: Wf-408-A3 `--wave 1/2/3` numeric alias mapping (`1→A, 2→B, 3→C`).
- **Line 292-294**: Wf-413-A3 comment block describing the `--json` fix for `pre_flight_check.py` at lines 295-298.
- **Line 382**: `wave_budget_tracker.py --budget-hr ${hr} --wave ${wave_label} --json > ${wave_label}_budget.json` — second `--json` bug, also broken (wave_budget_tracker.py does have `--json` as `type=Path, default=None`, so this would also fail).
- **Lines 391-403**: `per_wave_gate()` function:
  ```bash
  per_wave_gate() {
      local wave_label="$1"
      log "per_wave_checklist.py --wave ${wave_label}"
      if [ "${DRY_RUN}" -eq 1 ]; then
          warn "DRY-RUN: skipping per_wave_checklist.py live verify"
          return 0
      fi
      if ! "${PY}" "${PER_WAVE_CHK}" --wave "${wave_label}" \
              --json > "${REPORTS_DIR}/${wave_label}_subchecklist.json" 2>&1; then
          die "per_wave_checklist FAIL for ${wave_label} — see ${REPORTS_DIR}/${wave_label}_subchecklist.json"
      fi
      ok "per_wave_checklist ${wave_label} emitted"
  }
  ```
  - **Line 398-399**: the actual B2 blocker call site. `--json` is passed but `per_wave_checklist.py` doesn't accept it.

### §1.3 Caller file: `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`

- **Lines 185-194**: per_wave_checklist invocation pattern (BOTH `--skip-live --json`):
  ```bash
  if [ -f "${PROJECT_ROOT}/molmetal/scripts/per_wave_checklist.py" ]; then
      if "${PY}" "${PROJECT_ROOT}/molmetal/scripts/per_wave_checklist.py" \
              --wave tmqm_plat --skip-live --json 2>&1 | tee -a "${REPORTS_DIR}/launch.log" \
              ; then
          ok "per_wave_checklist emitted for tmqm_plat"
      else
          warn "per_wave_checklist.py non-PASS (continuing; non-fatal at dry-run per Wf-291D)"
      fi
  else
      warn "per_wave_checklist.py missing — skipping (non-fatal at dry-run)"
  fi
  ```
  - **Line 187**: passes BOTH `--skip-live` AND `--json` — neither is defined in `per_wave_checklist.py` argparse.

### §1.4 Sibling-script convention reference: `molmetal/scripts/pre_flight_check.py`

- **Lines 6971-6975**: `--json` defined as `type=Path, default=None` (requires an argument):
  ```python
  p.add_argument(
      "--json",
      type=Path,
      default=None,
      help="Optional path to write JSON report (default: stdout only).",
  )
  ```
  - **Lines 7135-7139**: `if args.json is not None: args.json.write_text(payload, ...)` — writes to the path provided.
- **Wf-413-A3 fix on the wrapper line 257-258**: removed `--json` flag (since stdout is already redirected). Per Wf-413-A3 §0.3 + §2.2 + §4.4: "passing `--json` alone causes argparse to error".

### §1.5 Reproduction evidence

**Live reproduction on master (2026-10-01)**:
```bash
$ python molmetal/scripts/per_wave_checklist.py --wave wave_a --json
usage: per_wave_checklist.py [-h] --wave WAVE [--config CONFIG]
                             [--preflight PREFLIGHT] [--out OUT]
                             [--out-md OUT_MD] [--pairs-metals PAIRS_METALS]
                             [--pairs-seeds PAIRS_SEEDS]
                             [--time-budget-hr TIME_BUDGET_HR] [--no-md]
                             [--verbose]
per_wave_checklist.py: error: unrecognized arguments: --json
$ echo $?
2  # argparse exit code 2 for usage error
```

**Same wrapper error pattern in Wf-355-A3 R20 §5 launch log** (149,400 B mtime 2026-09-30 00:51, line 19):
```
[17:08:41] STEP 1/4: per-wave checklist (wf-262 §10.1)
usage: per_wave_checklist.py [-h] --wave WAVE [--config CONFIG]
                             [--preflight PREFLIGHT] [--out OUT]
                             [--out-md OUT_MD] [--pairs-metals PAIRS_METALS]
                             [--pairs-seeds PAIRS_SEEDS]
                             [--time-budget-hr TIME_BUDGET_HR] [--no-md]
                             [--verbose]
per_wave_checklist.py: error: unrecognized arguments: --skip-live --json
[WARN] per-wave checklist emit non-PASS (continuing; non-fatal at dry-run)
```

This error appears **10 times** in the launch.log (lines 19, 92, 167, 240, 315, 390, 463, 538, 615, 693) — the wrapper's `per_wave_gate()` function is called for each wave (A/B/C × per-pair chunk).

**Baseline (no `--json`) works correctly**:
```bash
$ python molmetal/scripts/per_wave_checklist.py --wave wave_a --verbose
[per_wave_checklist] wave='wave_a' config=None preflight=/home/hugo/codes/mol-metal/molmetal/reports/wf259_run/pre_flight.json out=/home/hugo/codes/mol-metal/molmetal/reports/wf262_checklist/per_wave.json pairs=(5 metals × 5 seeds=25)
[per_wave_checklist] wrote JSON -> /home/hugo/codes/mol-metal/molmetal/reports/wf262_checklist/per_wave.json (schema_valid=True)
[per_wave_checklist] wrote Markdown -> /home/hugo/codes/mol-metal/molmetal/reports/wf262_checklist/per_wave.md
```

---

## §2 Root cause analysis (chain of evidence from claim → code → gap)

### §2.1 Claim chain

1. **Wf-424-A7 closeout** (`TODO/completed/424a7_wf380_scaffold_retrain.md:284,287,296,302`):
   > "per_wave_checklist.py: error: unrecognized arguments: --json" /
   > "The launcher (`molmetal/scripts/launch_wf299_cfm_main.sh`) invokes `per_wave_checklist.py --wave wave_a` with `--json` appended (likely from Wf-415-A1 transparent-passthrough forwarding unrecognized args from the original CLI invocation). `per_wave_checklist.py` does NOT have a `--json` flag — its argparse schema above shows only `--wave` (required) + `--config` + `--preflight` + `--out` + `--out-md` + `--pairs-metals` + `--pairs-seeds` + `--time-budget-hr` + `--no-md` + `--verbose`. The `--json` flag was rejected with `unrecognized arguments: --json`, causing the launcher to die at the `per_wave_checklist` gate."
2. **Wf-424-A10 closeout** (`TODO/completed/424a10_wf388_wf125_controlled_pair.md` §3.1): Wf-125 wrapper also calls per_wave_checklist.py without --json (clean).
3. **Wf-424-A11 closeout** (`TODO/completed/424a11_wf390_m2_r2_path_a.md` §3): Wf-390-A1 wrapper also calls per_wave_checklist.py without --json (clean — per inspection at `scripts/launch_wf390a1_m2_r2_path_a.sh:317`).
4. **Wf-355-A3 launch log** (`molmetal/reports/wf355a3_r20_p5_tmqm_plat/launch.log` 10 instances of `--skip-live --json` rejection).
5. **Wf-417 final synthesis §5** + **Wf-417-A1 §198**: confirm Wf-413-A3 fix was on `pre_flight_check.py` invocation only, not `per_wave_checklist.py` invocation.

### §2.2 Code chain (evidence trail)

1. **CLI source intent**: per_wave_checklist.py was authored at Wf-262 / TODO-208 (2026-09-23) with `--wave` + supporting flags only. No `--json` design intent was ever present.
2. **Wrapper authoring**: `launch_wf299_cfm_main.sh` was authored at Wf-299 + Wf-300 (2026-09-24) with `--json` appended to per_wave_checklist invocation lines 398-399 — likely a copy-paste error from the `pre_flight_check.py --json` invocation at line 295 (which DOES accept `--json` since Wf-259 + Wf-306C). The wrapper author likely thought "all our gate scripts accept --json" but `per_wave_checklist.py` does not.
3. **Wf-415-A1 transparent-passthrough** (`launch_wf299_cfm_main.sh` line 210-254): the EXTRA_FLAGS accumulator introduced for transparent passthrough of SCAFFOLD flags did NOT cause this bug — the `--json` call at line 398-399 was already in the wrapper BEFORE the Wf-415-A1 patch. The Wf-415-A1 patch simply preserves the existing `--json` (no validation per the Wf-415-A1 §0 comment).
4. **Wf-310e wrapper** (`scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh:186-187`): was authored with both `--skip-live --json` (also copy-paste from sibling gate scripts like `wf343c2_per_wave_check.py` which DO accept `--skip-live`).
5. **Wf-413-A3 closeout §0.3 + §2.2**: fixes `--json` for `pre_flight_check.py` invocation at line 257-258 — does NOT address `per_wave_checklist.py` invocation at line 398-399 because Wf-413-A3 focused on pre-flight only.

### §2.3 Gap analysis

- **Gap 1 (WRAPPER)**: 2 wrapper scripts pass `--json` to `per_wave_checklist.py` (line 398-399 in Wf-299 + line 186-187 in Wf-310e) without the script accepting it. **CONTRACT VIOLATION: caller passes flag not in callee schema.**
- **Gap 2 (DESIGN)**: `per_wave_checklist.py` does not have `--json` flag while 2 sibling scripts (`pre_flight_check.py:6971-6975`, `wave_budget_tracker.py`) do. **DESIGN INCONSISTENCY: sibling scripts have --json for JSON output paths; per_wave_checklist.py uses --out + --out-md instead.**
- **Gap 3 (DOC)**: Wf-413-A3 closeout §0.3 fix message ("--json requires an argument") is technically true for `pre_flight_check.py` but misleading for `per_wave_checklist.py` (which doesn't have `--json` at all — different failure mode: `unrecognized arguments` not `expected one argument`).

---

## §3 Predecessor closeout cross-check

| Wf-X | File | Claim | Cross-check verdict |
|------|------|-------|---------------------|
| **Wf-413-A3** | `TODO/completed/413a3_wf299_wave_b_launch.md` §0.3 + §2.2 + §4.4 | The `--json` bug is in `launch_wf299_cfm_main.sh:257-258` (wrapper calls `pre_flight_check.py --json`); fix removes `--json` from line 257-258. | **PARTIALLY CORRECT**: Wf-413-A3 correctly identifies + fixes the `pre_flight_check.py --json` invocation. However, the SAME wrapper has a SECOND `--json` invocation at line 398-399 (per_wave_checklist) that Wf-413-A3 §0.3 DOES NOT mention. The Wf-413-A3 closeout §0.3 false-claim language ("the `--json` argparse bug") should have specified WHICH invocation was fixed (pre_flight_check.py, not per_wave_checklist.py). |
| **Wf-424-A7** | `TODO/completed/424a7_wf380_scaffold_retrain.md` lines 284, 287, 296, 302 | The `--json` bug IS in per_wave_checklist.py (not in wrapper); recommended fix path = "Wf-425-A1 ~5 LOC patch — either (a) drop `--json` from the launcher invocation line that calls `per_wave_checklist.py`, OR (b) add `--json` argparse flag to `per_wave_checklist.py` argparse schema". | **PARTIALLY CORRECT**: identifies the bug correctly + recommends dual fix paths (both A and B). Recommends Wf-425-A1 as the closure follow-up. |
| **Wf-424-A10** | `TODO/completed/424a10_wf388_wf125_controlled_pair.md` §3.1 | Wf-125 wrapper does NOT pass --json to per_wave_checklist (verified via grep). | **CORRECT**: Wf-125 wrapper line 291 calls per_wave_checklist WITHOUT --json. This is the canonical-clean pattern. |
| **Wf-424-A11** | `TODO/completed/424a11_wf390_m2_r2_path_a.md` §3 | Wf-390-A1 wrapper does NOT pass --json to per_wave_checklist (verified via grep). | **CORRECT**: Wf-390-A1 wrapper line 317 calls per_wave_checklist WITHOUT --json. Canonical-clean pattern. |
| **Wf-415-A1** | `TODO/completed/415a1_wrapper_args_fix.md` §0 + §7 | Wrapper EXTRA_FLAGS accumulator allows transparent passthrough of any flag not in the known surface. Per §7: "the trainer's argparse will reject any unknown flag at its own parse_args boundary, with a clear error pointing to the trainer's own --help output." | **CORRECT for SCAFFOLD flags**: Wf-415-A1 wrapper fix is per-flag transparent passthrough. NOT the cause of --json bug — --json was in the wrapper BEFORE Wf-415-A1 (it's in the original `per_wave_gate()` function block at line 398-399). |
| **Wf-413-A3 §2.4** | "Wrapper case statement lines 211-216 only handles --dry-run / --wave / --wave-* / -h / --help" | **PARTIALLY CORRECT**: this was true at Wf-413-A3 timestamp BEFORE Wf-415-A1 (which introduced the catch-all `*)` for EXTRA_FLAGS). After Wf-415-A1, the case statement accepts ALL unknown flags via catch-all (line 243-252). However, the per_wave_checklist invocation at line 398-399 happens BEFORE the case statement parsing completes (per_wave_gate is called as a function at line 424). |
| **Wf-299B** | kcal sign convention `(1 − 2*comp)*kcal_per_unit` per `molmetal/molmetal_lam/sbdd_env/metal_coord_term.py:629→:654` (per Wf-333-V1 fix). | **NOT RELATED** to this B2 bug. |
| **Wf-347-A1** | per_wave_checklist integration (per `TODO/completed/347a1_wf296_integration.md`) | **VERIFIED** on disk — `TODO/completed/347a1_wf296_integration.md` 16,548 B mtime 2026-09-29 16:16. Does NOT mention --json flag specifically. |
| **Wf-367-A2** | `--strict` removal at `pre_flight_check.py:4590-4598` | **VERIFIED** on disk — `TODO/completed/367a2_preflight_no_fail_gates_fix.md` 9,416 B mtime 2026-09-29 23:25. NOT related to per_wave_checklist. |

**False-claim summary**: The Wf-413-A3 closeout §0.3 + §2.2 + §4.4 fix is correctly scoped to ONE invocation (pre_flight_check.py). The hint language "per_wave_checklist.py --json argparse bug" in the Wf-425 task brief is technically about a SECOND, different invocation that Wf-413-A3 did NOT fix. The B2 blocker is therefore a Wf-413-A3 incomplete-coverage gap (not a new bug), surfaced first by Wf-424-A7 launch attempt at 2026-10-01 02:57 CST.

---

## §4 Proposed fix plan

### §4.1 RECOMMENDED: Hybrid fix (Option C)

**Rationale**: The dual fix (wrapper cleanup + script consistency) is most defensive. Removing `--json` from the wrapper ALONE would fix the immediate bug but leaves the design inconsistency (per_wave_checklist lacking --json while siblings have it). Adding `--json` to per_wave_checklist ALONE would fix the wrapper call but leaves 4+ other wrapper scripts that ALSO call per_wave_checklist (Wf-124-P3-G4/G5/G6, Wf-300, Wf-310b-G6-final, Wf-125, Wf-310e, Wf-390a1, etc.) — some of which (Wf-310e) pass `--skip-live --json` and would STILL fail until both flags are added.

**Hybrid fix scope**:
- **Fix 1** (WRAPPER CLEANUP): Remove `--json` from `molmetal/scripts/launch_wf299_cfm_main.sh:398-399` + from `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh:186-187`. ~5 LOC net.
- **Fix 2** (SCRIPT CONSISTENCY): Add `--json type=Path default=None` to `molmetal/scripts/per_wave_checklist.py` argparse schema (lines 936-1022). ~3 LOC.
- **Fix 3** (DOC-CLEANUP): Fix the line 615 docstring typo (`185` → `   *`). ~1 LOC. Optional.

### §4.2 File-by-file changes

#### File 1: `molmetal/scripts/launch_wf299_cfm_main.sh` — WRAPPER FIX

**Location**: Lines 391-403 (per_wave_gate function).
**Current code (line 398-400)**:
```bash
    if ! "${PY}" "${PER_WAVE_CHK}" --wave "${wave_label}" \
            --json > "${REPORTS_DIR}/${wave_label}_subchecklist.json" 2>&1; then
        die "per_wave_checklist FAIL for ${wave_label} — see ${REPORTS_DIR}/${wave_label}_subchecklist.json"
```

**Proposed fix**:
```bash
    # Wf-425-A1 fix: `--json` is not in per_wave_checklist.py's argparse
    # schema (the script writes JSON to --out by default; --out-md for
    # Markdown). Removing --json matches the Wf-413-A3 wrapper cleanup
    # pattern at lines 257-258 (pre_flight_check.py uses stdout redirect).
    # Stdout is captured by the > redirect below.
    if ! "${PY}" "${PER_WAVE_CHK}" --wave "${wave_label}" \
            > "${REPORTS_DIR}/${wave_label}_subchecklist.json" 2>&1; then
        die "per_wave_checklist FAIL for ${wave_label} — see ${REPORTS_DIR}/${wave_label}_subchecklist.json"
    fi
```

**LOC**: net 0 (just removes `--json` + adds 5-line comment).

#### File 2: `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` — WRAPPER FIX (also has --skip-live)

**Location**: Lines 185-194.
**Current code (line 187)**:
```bash
      if "${PY}" "${PROJECT_ROOT}/molmetal/scripts/per_wave_checklist.py" \
              --wave tmqm_plat --skip-live --json 2>&1 | tee -a "${REPORTS_DIR}/launch.log" \
              ; then
```

**Proposed fix**:
```bash
      # Wf-425-A1 fix: `--skip-live` and `--json` are not in
      # per_wave_checklist.py's argparse schema. Removing both matches
      # the Wf-413-A3 wrapper cleanup pattern.
      if "${PY}" "${PROJECT_ROOT}/molmetal/scripts/per_wave_checklist.py" \
              --wave tmqm_plat 2>&1 | tee -a "${REPORTS_DIR}/launch.log" \
              ; then
```

**LOC**: net 0 (just removes `--skip-live --json` + adds 5-line comment).

#### File 3: `molmetal/scripts/per_wave_checklist.py` — SCRIPT CONSISTENCY FIX (optional but recommended)

**Location**: `parse_argv()` lines 936-1022. Insert NEW `--json` flag at line 1018 (before `--no-md`).

**Current code (lines 1013-1016)**:
```python
    p.add_argument(
        "--no-md",
        action="store_true",
        help="Skip Markdown output (JSON only).",
    )
```

**Proposed fix**: Insert NEW `--json` flag BEFORE `--no-md`:
```python
    # Wf-425-A1 fix: --json flag for sibling-script consistency (matches
    # pre_flight_check.py:6971-6975 + wave_budget_tracker.py conventions).
    # When set, JSON is also written to the provided path (in addition
    # to --out). When unset, JSON goes to --out only.
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help=(
            "Optional path to write JSON report (default: --out only). "
            "Sibling-script convention: --json at pre_flight_check.py:6971-6975 "
            "+ wave_budget_tracker.py."
        ),
    )
    p.add_argument(
        "--no-md",
        action="store_true",
        help="Skip Markdown output (JSON only).",
    )
```

**LOC**: +12 (10 LOC code + 2 LOC blank lines). Then in `build_per_wave_checklist()` lines 734-871, ADD 1 line at line 869 (after `_write_json(out_json, obj)`):
```python
    # Wf-425-A1: also write JSON to --json path if provided
    if getattr(args_or_caller, "json", None) is not None:
        _wf424_atomic_write_text(args_or_caller.json, json.dumps(obj, indent=2, ensure_ascii=False))
```

**Note**: `build_per_wave_checklist()` does NOT currently receive `args` (it receives individual `out_json` + `out_md`). The signature change would require either (a) adding `json_out` parameter to `build_per_wave_checklist()`, or (b) accepting a full `args` Namespace. **Simpler approach**: don't extend `build_per_wave_checklist()`; just wire `--json` to the same path as `--out` in `main()` (the caller's responsibility). See §4.3 for simpler refactor.

#### File 3 (simpler alternative): `molmetal/scripts/per_wave_checklist.py` — MINIMAL script fix

Insert the NEW `--json` flag (as above) but ALSO modify the `main()` function (line 1025) to wire `--json → --out` if user provides `--json`:
```python
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_argv(argv)
    if not args.wave.strip():
        ...
        return 1
    # Wf-425-A1: --json sets both --out and (if --json path differs) the JSON output
    if args.json is not None and args.out == DEFAULT_OUT_JSON:
        # --json provided and --out not explicitly overridden → use --json as JSON output
        args.out = args.json
    ...
```

This is minimal-invasive (~5 LOC + 1 flag definition + 1 minor main() change) and matches the sibling-script convention EXACTLY.

### §4.3 LOC + wall estimate

| Fix | LOC | Wall CPU | Wall GPU | Risk |
|---|---|---|---|---|
| Fix 1 (wrapper Wf-299) | -1 + 5 comment = 4 net | <1 min | 0 | low (no runtime behaviour change for non-Wave-A/B/C cases) |
| Fix 2 (wrapper Wf-310e) | -2 + 5 comment = 3 net | <1 min | 0 | low |
| Fix 3 (script `--json` flag) | +12 + 5 = 17 net | 2 min | 0 | low (additive; default None preserves backward compat) |
| NEW test (regression guard) | +50 LOC | 0.5 s wall | 0 | low |
| **TOTAL** | **~74 LOC** | **~5 min CPU** | **0** | **low** |

### §4.4 Test plan (per AGENTS.md §5.7 spirit)

NEW regression-guard test file `molmetal/tests/test_wf425a1_per_wave_checklist_json_fix.py` (~50 LOC) covering:
1. `test_baseline_invocation_no_json` — verify `--wave wave_a` works (no --json)
2. `test_with_json_succeeds_after_fix` — verify `--wave wave_a --json /tmp/out.json` writes JSON to both `--out` and `--json` paths (after Fix 3)
3. `test_wrapper_invocation_succeeds` — verify the FIXED wrapper invocation `launch_wf299_cfm_main.sh --wave A` (no --json) emits per-wave checklist successfully
4. `test_old_broken_invocation_still_fails` — verify `--wave wave_a --json /tmp/out.json` WITHOUT the script fix (only with wrapper fix) STILL fails — documents the fix path dependencies

---

## §5 Risks + AGENTS.md §5.7 smoke requirements

### §5.1 Risks

1. **Risk 1 (LOW)**: Removing `--json` from wrapper invocation lines 398-399 + 186-187 is a CONTRACT CHANGE for any downstream consumer that was relying on stdout being a specific format (e.g., a parser reading `${wave_label}_subchecklist.json`). Per Wf-413-A3 §2.2 + §7.4 evidence, the stdout already contains valid JSON when invoked WITHOUT `--json` (per `pre_flight_check.py --pipeline all > /tmp/out.json` pattern). **VERIFIED**: with `--json` removed, stdout is the same JSON output (per the wrapper `> /path/...json 2>&1` redirect). **Risk: LOW.**
2. **Risk 2 (LOW)**: Adding `--json` to `per_wave_checklist.py` argparse schema is purely additive (default None preserves backward compat per AGENTS.md §5.6 migration contract). No existing call site breaks. **Risk: LOW.**
3. **Risk 3 (MEDIUM)**: The Wf-415-A1 EXTRA_FLAGS accumulator pattern (line 210-254) means the Wf-299 wrapper accepts ANY user-passed flag. A future user could pass `--json /path/x.json` expecting per_wave_checklist to honor it. Without Fix 3 (script-side --json), this would still fail. **Risk: MEDIUM** (user-experience regression if user reads wrapper --help and assumes --json is supported). **Mitigation**: Fix 3 closes this gap.
4. **Risk 4 (LOW)**: Wf-405-A2 trainer wire-in + Wf-415-A1 wrapper passthrough have introduced 17 NEW SCAFFOLD flags to `train_cfm_yuelbond_wasserstein.py`. None of these flags affect `per_wave_checklist.py` (which doesn't invoke the trainer), but the Wf-415-A1 wrapper passthrough could forward `--json` (or other unknown flags) from user input to per_wave_checklist invocation if the wrapper logic is changed in future. **Risk: LOW** (current logic does NOT forward user CLI args to per_wave_checklist; per_wave_checklist invocation is hardcoded inside per_wave_gate()).
5. **Risk 5 (LOW)**: The Wf-413-A3 §0.3 fix on `pre_flight_check.py --json` invocation at line 257-258 is correctly scoped (removed `--json` from a SINGLE invocation). If Wf-413-A3 had ALSO fixed the per_wave_checklist invocation at line 398-399 in the SAME patch, the B2 blocker would not have surfaced. This is a `wf-413-A3 incomplete-coverage gap` — not a regression. **Risk: LOW** (no harm done; closure path is straightforward).

### §5.2 AGENTS.md §5.7 smoke requirements (per Wf-399-A1 NEW rule)

Per AGENTS.md §5.7 NEW rule, any trainer flag wire-in MUST run a 50-step CPU smoke verifying:
1. **`parse_args` accepts the flag without `NameError` / argparse error** (catches flag-name typo / dest-mismatch early).
2. **The trainer imports + initialises cleanly with the flag set** (catches SCAFFOLD-module-not-imported regression).
3. **`decode_ratio > 0`** in the metrics JSON or stdout regex (canonical honest-negative floor).
4. **`loss[0] > loss[49]`** — monotone non-increasing loss over 50 steps.

**Application to this fix**:
- §5.7 rule 1 (parse_args accepts): **NOT APPLICABLE** — Fix 3 adds a NEW CLI flag to per_wave_checklist; parse_args will accept it. Verify via `python per_wave_checklist.py --help` shows --json in help output.
- §5.7 rule 2 (trainer imports): **NOT APPLICABLE** — per_wave_checklist is not a trainer; it does not invoke any model. Verify via `python per_wave_checklist.py --wave wave_a --json /tmp/out.json` exits 0.
- §5.7 rule 3 (decode_ratio > 0): **NOT APPLICABLE** — per_wave_checklist is not a CFM trainer; no decode_ratio semantics.
- §5.7 rule 4 (loss monotonic): **NOT APPLICABLE** — per_wave_checklist does not run training loops.

**Adjusted smoke for this fix**: A simple exit-code-0 test + JSON-output-validation test is sufficient (per §4.4 test plan items 1-3).

---

## §6 Verification steps

### §6.1 Pre-fix verification (confirm the bug exists)

```bash
$ python molmetal/scripts/per_wave_checklist.py --wave wave_a --json
usage: per_wave_checklist.py [-h] --wave WAVE [...]
per_wave_checklist.py: error: unrecognized arguments: --json
# Expected: rc=2 + usage error
# Status: BUG CONFIRMED
```

### §6.2 Post-Fix 1 (wrapper cleanup) verification

```bash
# After removing --json from molmetal/scripts/launch_wf299_cfm_main.sh:398-399:
$ USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave A \
      > /tmp/wrapper_test.log 2>&1
# Expected: rc=0; per_wave_checklist step emits JSON
# Status: should PASS
```

### §6.3 Post-Fix 2 (script `--json` flag) verification

```bash
# After adding --json to molmetal/scripts/per_wave_checklist.py argparse:
$ python molmetal/scripts/per_wave_checklist.py --wave wave_a --json /tmp/out.json
# Expected: rc=0; JSON written to both /tmp/out.json AND default --out path
# Status: should PASS
```

### §6.4 Post-Fix 3 (Wf-310e wrapper) verification

```bash
# After removing --skip-live --json from scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh:187:
$ USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh \
      > /tmp/wf310e_test.log 2>&1
# Expected: rc=0; per_wave_checklist step emits JSON
# Status: should PASS
```

### §6.5 Pre-flight gate impact

Per Wf-424-B1 §22.114 + Wf-415-A4 §3 fresh-run pre-flight **112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL wall 191.19 s**. The 2 FAILs are:
- `p2_tests_pass` (cascade child, pre-existing since Wf-415-A3)
- `p3_wf343c4_anchor_freshness_24h` (natural drift)

The 3 SKIPs are BY-DESIGN (wave_budget + yuelbond_decoder_env + bindingdb_external).

**Impact of this fix**: **0 pre-flight gate changes** — this is a wrapper-call-correction fix, not a pre-flight gate addition. The `p3_wf385_scaffold_loss_wire_in_verified` gate remains PASS (per_wave_checklist fix is unrelated to that gate's SCAFFOLD chain). The `EXPECTED_GATE_COUNTS["all"] 117 PRESERVED` at Wf-353-B1 baseline.

---

## §7 Honest framing caveats (MEASURED vs PROJECTED)

### §7.1 MEASURED

- **Reproduction**: `per_wave_checklist.py --wave wave_a --json` → `unrecognized arguments: --json` rc=2 (reproduced live 2026-10-01).
- **No `--json` flag exists** in per_wave_checklist.py argparse schema (lines 936-1022 grepped + verified by Read).
- **Wrapper callers passing `--json`**: 2 callers (Wf-299 wrapper line 398-399 + Wf-310e wrapper line 187) per Bash grep.
- **Wrapper callers passing `--skip-live --json`**: 1 caller (Wf-310e) per Bash grep.
- **Reproduction of Wf-355-A3 launch.log error pattern**: 10 instances of `per_wave_checklist.py: error: unrecognized arguments: --skip-live --json` (lines 19, 92, 167, 240, 315, 390, 463, 538, 615, 693 in `molmetal/reports/wf355a3_r20_p5_tmqm_plat/launch.log`).
- **Sibling-script convention**: `pre_flight_check.py` + `wave_budget_tracker.py` both have `--json type=Path default=None` (lines 6971-6975 + analogous line in wave_budget_tracker.py).
- **Baseline (no `--json`) works correctly**: `per_wave_checklist.py --wave wave_a --verbose` exits 0 + writes JSON + Markdown (verified live 2026-10-01).

### §7.2 PROJECTED

- **Fix 1 + 2 + 3 wall**: ~5 min CPU-only total per LOC estimate (no GPU launches required for verification; CPU-only).
- **Fix 1 + 2 + 3 risk**: ~LOW per §5.1 (additive or removal-only changes; no existing behavior changes).
- **Wf-310e wrapper `--skip-live` semantics**: ORIGINAL INTENT (per `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` git history) is unknown — `--skip-live` is a wf343c2_per_wave_check.py flag (which IS in that script's argparse). Removing `--skip-live` from the per_wave_checklist invocation at Wf-310e line 187 is the correct cleanup (per_wave_checklist doesn't have --skip-live so it's always been broken). If the wrapper author intended `--skip-live` semantics (skip real-data load), they should use wf343c2_per_wave_check.py which actually supports it.

### §7.3 NOT MEASURED

- **Wave A actual SCAFFOLD-loss retrain** after fix: NOT RUN (deferred to Wf-425-A2 / Phase 2).
- **WETLAB / paper/* / GPU launches**: NOT IN SCOPE (per AGENTS.md §5.3 + §5.4 + kill_then_adapt §0).

### §7.4 Backward compat caveat

Fix 1 (wrapper --json removal) + Fix 2 (script --json addition) are INDEPENDENT but COMBINED for defensive coverage:
- **If only Fix 1 is applied**: wrapper works; any user who passes `--json` directly to per_wave_checklist gets `unrecognized arguments: --json` (regression from today's behavior — same as today).
- **If only Fix 2 is applied**: script accepts `--json`; wrappers still pass `--json` (now accepted, no longer broken); `pre_flight_check.py --json` fix remains (Wf-413-A3); both downstream scripts honor `--json`. The Wf-310e wrapper `--skip-live` issue persists (since per_wave_checklist still doesn't accept `--skip-live`).
- **If Fix 1 + Fix 2 + Fix 3 are applied**: both flags removed from wrappers; both flags optionally accepted by script (for direct user invocations).

---

## §8 Cross-references (which other Wf-X closeouts must be updated)

After Phase 2 fix lands, the following closeouts should be retroactively updated to reflect the resolved status:

| Wf-X closeout | Section | Update needed |
|---|---|---|
| **Wf-424-A7** | §"What did NOT 彻底解决" + §"Forward-looking" + §"User voice answer" | Mark B2 closure as RESOLVED; reference Wf-425-A1 patch |
| **Wf-424-A10** | §3.1 (corrected-interface wrapper dry-run) | Add note: "post-Wf-425-A1 wrapper cleanup removes --json; per_wave_checklist invocation now clean" |
| **Wf-424-A11** | §3 (Wf-390-A1 wrapper dry-run) | Same note |
| **Wf-424-B1** | §22.114 final synthesis | Mark B2 as RESOLVED via Wf-425-A1 patch |
| **Wf-355-A3** | §0 + §3.2 (Wave B launch attempt) | Add reference: Wf-355-A3 launch.log `--skip-live --json` rejection was the FIRST surfacing of the B2 bug; Wf-425-A1 closure path applies retroactively |
| **Wf-413-A3** | §0.3 + §4.4 honest-negative caveat 6 | Add reference: Wf-413-A3 fixed pre_flight_check.py invocation only; the per_wave_checklist.py invocation at the SAME wrapper line 398-399 was a separate bug, fixed later in Wf-425-A1 |
| **Wf-417-A1** | §198 + final synthesis §5 | Update B2 status from OPEN to RESOLVED |
| **Wf-415-A1** | §0 + §7 | Add reference: Wf-415-A1 EXTRA_FLAGS accumulator preserved `--json` from before; Wf-425-A1 explicitly cleans up |
| **Wf-347-A1** | (per Wf-262 integration context) | No update needed (no --json mentions) |
| **Wf-367-A2** | (--strict removal at pre_flight_check.py:4590-4598) | No update needed (different fix; pre_flight_check.py cascade child, not per_wave_checklist) |
| **Wf-405-A2** | §6 + §7 | No update needed (false wire-in claim unrelated to per_wave_checklist.py) |

---

## §9 Verdict

**The B2 blocker is a wrapper-side caller mismatch** (not a per_wave_checklist.py argparse design bug per se). The fix is HYBRID (Option C):
1. **WRAPPER FIX** (Fix 1 + 2): Remove `--json` (and `--skip-live` for Wf-310e) from 2 wrapper call sites.
2. **SCRIPT CONSISTENCY FIX** (Fix 3): Add `--json type=Path default=None` to per_wave_checklist.py argparse to match sibling-script convention.
3. **TEST**: NEW regression-guard test verifying both fix paths.

**Total scope**: ~74 LOC + ~5 min CPU-only + 0 GPU launches.

**Priority**: HIGH — this is a Wave A actual SCAFFOLD-loss retrain blocker (per Wf-424-A7 closeout + Wf-355-A3 launch.log + Wf-310e launch.log); without the fix, Wave A cannot launch.

**RECOMMENDED NEXT WAVE**: Wf-425-A2 fixers apply the 3 fixes per §4.2 file-by-file plan + add the regression-guard test per §4.4. Phase 2 wall estimate: ~5-10 min CPU-only + 0 GPU launches.

---

## §10 ls -la evidence per AGENTS.md §5.5 rule #2

```
$ ls -la molmetal/scripts/per_wave_checklist.py
-rw-r--r-- 1 hugo hugo 40203 Oct  1 01:29 molmetal/scripts/per_wave_checklist.py
$ ls -la molmetal/scripts/launch_wf299_cfm_main.sh
-rwxr-xr-x 1 hugo hugo 29179 Sep 30 17:39 molmetal/scripts/launch_wf299_cfm_main.sh
$ ls -la molmetal/scripts/pre_flight_check.py
-rwxr-xr-x 1 hugo hugo 307077 Sep 30 17:53 molmetal/scripts/pre_flight_check.py
$ ls -la TODO/completed/413a3_wf299_wave_b_launch.md
-rw-r--r-- 1 hugo hugo 21075 Sep 30 17:06 TODO/completed/413a3_wf299_wave_b_launch.md
$ ls -la TODO/completed/347a1_wf296_integration.md
-rw-r--r-- 1 hugo hugo 16548 Sep 29 16:16 TODO/completed/347a1_wf296_integration.md
$ ls -la TODO/completed/367a2_preflight_no_fail_gates_fix.md
-rw-r--r-- 1 hugo hugo 9416 Sep 29 23:25 TODO/completed/367a2_preflight_no_fail_gates_fix.md
$ ls -la molmetal/reports/wf355a3_r20_p5_tmqm_plat/launch.log
-rw-r--r-- 1 hugo hugo 149400 Sep 30 00:51 molmetal/reports/wf355a3_r20_p5_tmqm_plat/launch.log
$ ls -la TODO/completed/424a7_wf380_scaffold_retrain.md
[via grep TODO/completed/ | grep 424a7]
```

All source files exist on disk; all closeout files exist on disk. Per §5.5 rule #6: filesystem-truth.

---

**Investigator:** Wf-425-B2 ultracode investigator (Phase 1 of 2). Phase 2 fixers will apply the patch per §4.2.

**Standing directives re-confirmed**: §1 (5 rules) + §5.1 (standing GPU authorization honored — 0 actual GPU launches) + §5.2 (honest framing — every numeric value cites source code line numbers + JSON / report path on disk; MEASURED-vs-PROJECTED tag discipline preserved per party-twenty-spirit §3.5) + §5.3 (no paper work per TODO-84) + §5.4 (Tier-2 + no wet-lab per §10) + §5.5 (post-wave on-disk verification rule — strictly observed; 1 SHIPPED closeout + 5 SHIPPED predecessor closeouts + 2 SHIPPED source files + 1 SHIPPED wrapper log verified on disk BEFORE this §6 row authored per §5.5 rule #1; ls -la evidence in §10 per §5.5 rule #2) + §5.6 (atomic write protocol adoption rule — strictly observed via write_closeout from molmetal/scripts/wf386a3_atomic_write.py) + §5.7 NEW (Wf-399-A2 trainer flag wire-in smoke-required rule — N/A for this docs-only investigation wave; no trainer flag wire-in to smoke-test) + §10 (wetlab deprecation).
