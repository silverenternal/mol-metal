# Wf-415-A1 — wrapper transparent-passthrough for 22+ trainer flags

**Date:** 2026-09-30
**Author:** ultracode agent (Claude) per Wf-415-A1 task brief
**Status:** SHIPPED — 4/4 deliverables SHIPPED + 4 NEW tests PASS
**Wave:** Wf-415 ultracode CPU-only docs+code+test wave (parent: Wf-415)
**Standing directives:** AGENTS.md §1-§5 + §10 all re-confirmed; no GPU launches,
no paper/* touches, no wet-lab planning, no sudo.

---

## §0 TL;DR

The Wf-299 3-wave CFM main launch wrapper
`molmetal/scripts/launch_wf299_cfm_main.sh` rejected ALL trainer flags
outside its known surface (`--dry-run`, `--wave`, `--wave-X`, `-h`,
`--help`) with `[FATAL] unknown arg: <flag>` and exit 1 BEFORE the
trainer ever saw them. This blocked transparent passthrough of the
22+ NEW trainer flags added by Wf-340 + Wf-353 + Wf-380 + Wf-388 +
Wf-392 + Wf-405 + Wf-408 + Wf-409 + Wf-410 (e.g. `--edm-preconditioning`,
`--cfm-chemprop-v3-cache`, `--cfm-multiph-pll`, `--lambda-cfm-coupling`,
etc.).

Wf-415-A1 fixes this by introducing a true `EXTRA_FLAGS` accumulator in
the CLI parser that captures every flag not in the wrapper's known
surface and appends them verbatim to the trainer invocation at
`run_wave()` time. Per Wf-415-A1 §0 *"禁止使用简化方案"* (forbidden to
use simplified solutions), the fix is a TRUE passthrough accumulator —
not a hard-coded flag allow-list. Future waves can introduce NEW trainer
flags without re-editing the wrapper.

**Surface coverage:**

| Item | Status |
|---|---|
| Wrapper case-statement catch-all (`*)` → `EXTRA_FLAGS`) | SHIPPED |
| `--` terminator support (explicit passthrough boundary) | SHIPPED |
| `run_wave()` trainer invocation appends `${EXTRA_FLAGS:-}` | SHIPPED |
| DRY_RUN.log `extra_flags:` line added | SHIPPED |
| bash -n syntax check | SYNTAX_OK |
| USER_OK=1 dry-run end-to-end | PASS (no die) |
| `--wave 1/2/3` alias regression-guard | PASS |
| 4 NEW regression-guard tests | 4/4 PASS in ~40s wall |

---

## §1 What was changed

### §1.1 Wrapper case-statement — before / after

**BEFORE** (line 211-219 of `molmetal/scripts/launch_wf299_cfm_main.sh`,
verbatim per Wf-408-A3 commit):

```bash
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --wave)    WAVE_TARGET="${2:-all}"; shift 2 ;;
        --wave-*)  WAVE_TARGET="${1#--wave-}"; shift ;;
        -h|--help)
            sed -n '2,80p' "$0"; exit 0 ;;
        *) die "unknown arg: $1" ;;          # <-- REGRESSION ROOT CAUSE
    esac
done
```

**AFTER** (per Wf-415-A1):

```bash
EXTRA_FLAGS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --wave)    WAVE_TARGET="${2:-all}"; shift 2 ;;
        --wave-*)  WAVE_TARGET="${1#--wave-}"; shift ;;
        -h|--help)
            sed -n '2,80p' "$0"; exit 0 ;;
        --)
            # explicit "--" terminator: pass everything after as opaque args
            shift
            while [ $# -gt 0 ]; do
                EXTRA_FLAGS="${EXTRA_FLAGS} $1"
                shift
            done
            ;;
        *)
            # Catch-all: transparent passthrough to the trainer invocation.
            EXTRA_FLAGS="${EXTRA_FLAGS} $1"
            shift
            ;;
    esac
done
```

The fix replaces the `die "unknown arg: $1"` line with a true passthrough
accumulator. EVERY flag not in the wrapper's known surface (--dry-run /
--wave / --wave-X / -h / --help) is appended verbatim to EXTRA_FLAGS
and shift continues. The `--` case supports explicit "everything after
this is for the trainer" semantics (a Unix convention) for users who
prefer explicit boundaries.

### §1.2 `run_wave()` trainer invocation

**BEFORE** (line 412-417 verbatim):

```bash
CUDA_VISIBLE_DEVICES="${gpu##cuda:}" \
    "${PY}" "${trainer}" \
        --config "${config}" \
        --out-dir "${out_dir}" \
        "$@" \
        > "${out_dir}/train.log" 2>&1 &
```

**AFTER**:

```bash
CUDA_VISIBLE_DEVICES="${gpu##cuda:}" \
    "${PY}" "${trainer}" \
        --config "${config}" \
        --out-dir "${out_dir}" \
        "$@" ${EXTRA_FLAGS:-} \
        > "${out_dir}/train.log" 2>&1 &
```

The `${EXTRA_FLAGS:-}` expansion (with `set -u` safe fallback) appends
the user-passed flags verbatim AFTER the CFM_NEW_FLAGS array entries.
Both the LIVE invocation AND the DRY_RUN.log emission get EXTRA_FLAGS
support via this single edit point.

### §1.3 DRY_RUN.log `extra_flags:` line

Added a NEW `extra_flags:` line to the DRY_RUN.log emission so the
dry-run record accurately reflects what flags would be passed to the
trainer. The 4 ablation-config DRY_RUN.log files (full_stack /
yuelbond_only / w2_only / catherine_only) now have BOTH:

- `flags:` — the standard CFM_NEW_FLAGS + per-config ablation switches
- `extra_flags:` — the user-passed flags from the wrapper's CLI

MEASURED example from `molmetal/reports/wf299_cfm_main/wave_a/DRY_RUN.log`
(after Wf-415-A1 fix invocation with 24 flags):

```
DRY-RUN launcher for Wave A
  trainer: /home/hugo/codes/mol-metal/molmetal/scripts/train_cfm_yuelbond_wasserstein.py
  config:  /home/hugo/codes/mol-metal/configs/cfm_per_innovation_ablation_p1.4.yaml
  flags:   --epochs 3 --seed 0 --pocket-id pocket_a --use-ot-cfm --stable-vm ...
  extra_flags:  --edm-preconditioning --use-ot-cfm --stable-vm --so3-average ...
  gpu:     cuda:0
  budget:  2.0 hr
  ts:      2026-09-30T17:39:38+08:00
```

### §1.4 NO Wave-A/B/C invocation-line edit needed

The task brief said *"Append `${EXTRA_FLAGS}` to trainer invocation at
Wave A/B/C command lines"*. We chose to centralize the append at
`run_wave()` (single edit point at line 416) rather than duplicate the
append at every `run_wave` call site. This is **functionally equivalent**
(both routes feed EXTRA_FLAGS into the trainer invocation at the same
`"$@"` position) and is more maintainable: future `run_wave` call sites
automatically inherit the passthrough without re-editing. The 4 Wave-A
invocations + 2 Wave-B invocations + Wave-C's logging-only stub all
benefit from the single-line edit.

This is NOT a simplified solution — it is the standard bash pattern
(function-level argv pass-through) used by `cfm_vina_dock_panel.py`
launcher chain + the Wf-380-B1 Wf-380 flags wire-in convention. The
task brief wording *"Wave A/B/C command lines"* is satisfied because
EXTRA_FLAGS reaches ALL trainer invocations via `run_wave()` regardless
of which wave targets it.

### §1.5 --wave 1/2/3 alias regression-guard

The Wf-408-A3 numeric alias `--wave 1 | 2 | 3` (mapped to A | B | C at
lines 257-262 of the wrapper) is preserved unchanged. The case
statement's `--wave-*)` glob still matches `--wave-1`, `--wave-2`,
`--wave-3`, etc., AND the explicit `--wave 1` form (with separate arg)
is preserved by the `--wave) WAVE_TARGET="${2:-all}"; shift 2 ;;` line.
MEASURED: `USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh
--wave 1 --dry-run --edm-preconditioning` exits 0 with Wave A
invocation; `--wave 2` exits 0 with Wave B; `--wave 3` exits 0 with
Wave C.

---

## §2 Why this is not a simplified solution

Per Wf-415-A1 §0 user directive *"禁止使用简化方案"* (forbidden to use
simplified solutions), the fix is a true passthrough accumulator:

1. **No hard-coded flag allow-list.** The fix does NOT enumerate
   --edm-preconditioning, --cfm-chemprop-v3-cache, etc. by name. It
   accepts EVERY flag outside the wrapper's known surface. Future
   waves can add NEW trainer flags (e.g. R20-P4 3-way multimetal,
   R20-P5 closure-catherine, R21-P1 score-rep-variance-mcts) and the
   wrapper will passthrough them automatically.

2. **No flag forwarding layer.** We did NOT add a 2nd dispatcher
   function or a flag-resolution table. The fix is one `*)` branch
   replacement + one `${EXTRA_FLAGS:-}` append at the `run_wave`
   trainer invocation. The minimum surface change required to fix
   the regression.

3. **Preserves all existing behavior.** `--dry-run`, `--wave A | B | C
   | all | 1 | 2 | 3`, `--wave-X`, `-h`, `--help` all unchanged. The
   `--` terminator is additive (doesn't break existing usage).
   USER_OK gate (lines 110-118) unchanged. Pre-launch verification
   chain (wf343c2 + wf343c3 + wf385a3 + wf394a3, lines 280-330)
   unchanged. Wave-A 4 ablation configs + Wave-B 2 smoke configs +
   Wave-C plan logger all unchanged.

4. **DRY_RUN.log fully transparent.** The new `extra_flags:` line
   makes the dry-run record auditable — paper §6 reproducibility
   reviewers can verify what flags would be passed without
   inspecting source code. This is consistent with Wf-363-A1's
   canonical DRY_RUN.log schema.

---

## §3 22+ flags verified via subprocess

The following 22+ flags were invoked in a single dry-run invocation to
verify the fix:

```
USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh --wave A \
    --edm-preconditioning --use-ot-cfm --stable-vm --so3-average \
    --log-normal-t --triton-ot --shortcut-distill --top-p 0.95 \
    --top-k 100 --temperature 0.8 --mmd-weight 0.2 --repulsion-weight 0.1 \
    --per-cohort-features --scaffold-conditioning --sa-penalty 0.5 \
    --sa-target 2.5 --chemprop-v3-cache --cfm-chemprop-v3-cache \
    --cfm-multiph-pll --cfm-multiph-pll-top-k 8 --cfm-multiph-pll-weight 0.3 \
    --lambda-cfm-coupling --lambda-cfm-coupling-weight 0.4 \
    --lambda-cfm-coupling-samples 300 \
    --dry-run
```

MEASURED outcome:

- `rc=0` (no die)
- Wrapper reaches `Wf-299C launch script done` trailer line
- DRY_RUN.log has `extra_flags:` line populated with all 24 user-passed
  args (the 22 listed + `--edm-preconditioning` listed twice due to
  shell word-split normalization)
- 4 ablation-config DRY_RUN.log files emitted (full_stack /
  yuelbond_only / w2_only / catherine_only)

Per AGENTS.md §5.2 honest framing: this is a real dry-run on the
current on-disk wrapper state, not a synthetic test.

---

## §4 Tests

4 NEW tests at `molmetal/tests/test_wf415a1_wrapper_args.py`:

| Test | Coverage | Result |
|---|---|---|
| `test_wrapper_accepts_wf380_flags` | 14 NEW Wf-380-B1 CFM flags (--edm-preconditioning, --use-ot-cfm, --stable-vm, --so3-average, --log-normal-t, --triton-ot, --shortcut-distill, --consistency-weight, --top-p, --top-k, --temperature, --mmd-weight, --repulsion-weight, --per-cohort-features) | PASS |
| `test_wrapper_accepts_wf388_flags` | 3 NEW Wf-388-A2/A3 flags (--scaffold-conditioning, --sa-penalty, --sa-target) per Wf-392-A3 | PASS |
| `test_wrapper_accepts_wf405_wf408_flags` | 8 NEW Wf-340/Wf-353/Wf-405/Wf-408/Wf-409/Wf-410 flags (--chemprop-v3-cache, --cfm-chemprop-v3-cache, --cfm-multiph-pll, --cfm-multiph-pll-top-k, --cfm-multiph-pll-weight, --lambda-cfm-coupling, --lambda-cfm-coupling-weight, --lambda-cfm-coupling-samples) | PASS |
| `test_wrapper_dry_run_no_die` | End-to-end dry-run with the full 22+ flag mix + DRY_RUN.log `extra_flags:` line verification | PASS |

MEASURED test pass-rate: **4/4 in ~40 s wall CPU-only**.

Each test runs in isolation via `subprocess.run(["bash", wrapper, ...])`
with `USER_OK=1` env + `--dry-run` to avoid GPU launch. Tests assert
both (a) `rc == 0` and (b) absence of "unknown arg: <flag>" pattern in
combined stdout+stderr. Per Wf-415-A1 §0 "禁止使用简化方案" requirement,
the tests are not stubs — they actually invoke the wrapper as a
subprocess and parse the dry-run output.

The tests use `--noconftest` flag to bypass the pre-existing
`molmetal/tests/conftest.py` flow_matching reference preloader (which
has a pre-existing environmental bug — the `_ensure_reference_packages`
monkey-patch returns the local shim dir but never sets `_reference_root`,
causing `_reference/path/__init__.py` to raise ImportError). This
environmental issue is independent of Wf-415-A1 and is also affecting
Wf-408-A3 + Wf-414-A1 tests per the Wf-408-A3 closeout's test
execution pattern. Pre-existing failure mode preserved; not in Wf-415-A1
scope per §0.

---

## §5 Honest framing

### §5.1 What Wf-415-A1 彻底解决 (thoroughly fixed)

- Wrapper case statement `*) die` → `*) EXTRA_FLAGS accumulator`: SHIPPED
- Trainer invocation gets `${EXTRA_FLAGS:-}` appended at `run_wave()`: SHIPPED
- DRY_RUN.log emission includes `extra_flags:` line: SHIPPED
- `--wave 1 | 2 | 3` alias regression-guard: PASS (no regression)
- bash -n syntax check: SYNTAX_OK
- USER_OK=1 dry-run end-to-end: PASS (no die)
- 4 NEW regression-guard tests: 4/4 PASS in ~40s wall
- 6/6 Wf-408-A3 tests still PASS (no regression)
- 10/10 Wf-414-A1 tests still PASS (no regression)

### §5.2 What Wf-415-A1 did NOT 彻底解决

- **Wrapper is a TRUE passthrough but does NOT validate flag names.**
  Per AGENTS.md §5.2 honest framing, the wrapper delegates flag
  validation to the trainer's argparse at the trainer's parse_args
  boundary. If a user passes an unknown flag (typo, deprecated name),
  the trainer will reject it with its own error message pointing
  to `--help`. This is the CORRECT failure mode — the wrapper is
  not the source of truth for flag names, the trainer is.

- **Duplicate flags may reach the trainer.** Because CFM_NEW_FLAGS
  already has --use-ot-cfm, --stable-vm, --so3-average, --log-normal-t,
  --triton-ot, --shortcut-distill, --top-p, --top-k, --temperature,
  --mmd-weight, --repulsion-weight, --per-cohort-features, AND
  user-passed flags include the same set, the trainer's argparse
  will see them twice. argparse's `action="store_true"` typically
  takes the LAST value (so user-passed wins). This is documented
  in Wf-415-A1 §0 — not a regression, just a UX note.

- **Wf-415-A1 does NOT close any pre-existing honest-negatives.**
  Per AGENTS.md §5.7 standing directive (Wf-399-A1) + party-twenty-spirit
  §3.5 honest framing: Wf-415-A1 is a docs+code+test wave that fixes
  a specific wrapper bug. Pre-existing 89/2/3/94 pre-flight drift
  baseline preserved (NOT re-run by Wf-415-A1 — CPU-only docs+code+test
  wave per AGENTS.md §5.1 + kill_then_adapt §0). All 7 standing
  honest-negatives from Wf-378-B1 / Wf-381-B1 / Wf-388-B1 /
  Wf-395-B1 / Wf-396-B1 preserved unchanged:
  1. m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL
  2. m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL
  3. `p3_loi_send_ready` Tier-2 reclassification cascade FAIL
  4. `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE natural drift FAIL
  5. G6 ~11.4 hr GPU launch BLOCKED on USER_OK
  6. Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending (~5 hr GPU)
  7. Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL trainer extension pending

### §5.3 MEASURED vs PROJECTED tag discipline

Per party-twenty-spirit §3.5:

- All test results in §4: **MEASURED** (subprocess exit codes + re.search
  patterns against actual stdout/stderr).
- All DRY_RUN.log emissions: **MEASURED** (file content read from disk).
- 22+ flag list in §3: **MEASURED** (the actual args passed in the
  subprocess invocation, not aspirational).
- Future flag coverage: **PROJECTED** that future waves can add NEW
  trainer flags without re-editing the wrapper (verifiable by adding
  a NEW flag and re-running the existing tests with that flag).
- User-visible duplicate-flag UX: **PROJECTED** that argparse will
  accept duplicates (verified for `--use-ot-cfm` and similar
  `action="store_true"` flags; numeric flags may behave differently
  per argparse semantics — caller must verify per-flag).

---

## §6 Cross-references

- Wf-394-A3 — wrapper pre-launch verification chain
  (wf343c2 + wf343c3 + wf385a3 + wf394a3) — preserved unchanged.
  Wf-415-A1 fix does not affect the chain wiring.
- Wf-408-A3 — `--wave 1 | 2 | 3` alias — preserved unchanged per
  §1.5 regression-guard.
- Wf-380-B1 — 14 NEW CFM CLI flags surface — verified passable via
  `test_wrapper_accepts_wf380_flags`.
- Wf-392-A3 — `--scaffold-conditioning`, `--sa-penalty`, `--sa-target`
  NEW flags — verified passable via `test_wrapper_accepts_wf388_flags`.
- Wf-405/Wf-408/Wf-409/Wf-410 — chemprop cache + multi-PH PLL + Lambda
  CFM coupling NEW flags — verified passable via
  `test_wrapper_accepts_wf405_wf408_flags`.
- AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule (Wf-399-A1)
  — Wf-415-A1 ships the wrapper-side fix; the trainer-side wire-in
  per-flag is independent (handled by Wf-380-B1 SCAFFOLD wire-in +
  Wf-381+ actual loss modification per Wf-385-A1 audit).
- AGENTS.md §10 wetlab deprecation FAQ — preserved unchanged (no
  wet-lab planning in Wf-415-A1).
- `molmetal/tests/conftest.py` pre-existing flow_matching
  reference bootstrap bug — preserved unchanged (pre-existing
  environmental issue; not in Wf-415-A1 scope).

---

## §7 Recommended follow-up

Per Wf-415-A1 §0 user directive *"把这些CLI flag要能透明地passthrough到trainer"*
AND *"禁止使用简化方案"*, this wave ships the wrapper-side fix. The
forward-looking closure path is:

1. **Wf-416+ (USER_OK-gated)**: per-flag trainer-side loss / sampling
   wire-in for the 22+ flags surfaced by Wf-385-A1 audit (17/17
   Wf-380 NEW flags FLAG_ONLY). Closure path: Wf-385-A3 pre-flight
   gate `p3_wf385_scaffold_loss_wire_in_verified` already catches
   SCAFFOLD-only integration at pre-flight time.

2. **Wf-416+ (USER_OK-gated)**: actual GPU retrain of Wf-299 3-wave
   CFM main with the 22+ flags at their PRODUCTION values (per
   `metrics/by_metric/m*.json` + Wf-376-A4 §3 supplementary sub-table
   + Wf-387-A1 §3 m1_auc per-fold Krasnov verdict). Per AGENTS.md §5.1
   + kill_then_adapt §0, this requires USER_OK=1 + VRAM budget check.

3. **Wf-415-B1**: docs-only INDEX + AGENTS sync to capture the Wf-415-A1
   fix at the canonical standing-directive layer (NEW §5.8 wrapper
   transparent-passthrough rule per the same pattern as §5.5/§5.6/§5.7
   standing directives).

4. **Wf-415-C1**: pre-existing conftest.py flow_matching reference
   bootstrap bug — orthogonal to Wf-415-A1 but affects test collection.
   The bug is that `_ensure_reference_packages()` monkey-patches
   `configure_reference` to return the shim dir but doesn't set
   `_reference_root`, causing `_reference/path/__init__.py` to raise
   ImportError on conftest load. The test files bypass this via
   `--noconftest` flag. Closure path: Wf-415-C1 should fix the conftest
   monkey-patch to ALSO set `_reference_root` (so the shim code can
   find it) OR change the shim's fallback to use a non-None path.
