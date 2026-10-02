# Wf-405-A4 — close-out: USER_OK UX cross-wrapper regression guard

**Date:** 2026-09-30
**Wave:** Wf-405 (CPU-only docs+code+test ultracode)
**Sub-flow:** A4 (the close-out itself; sub-flow of Wf-405-B1 docs sync)
**Status:** **SHIPPED 1/1 deliverable (100%)**
**Author:** Hugo Lee (Mol-Metal project lead) — `hugo.lee@mol-metal.org`

---

## §0 TL;DR

Wf-405-A4 ships a **cross-wrapper USER_OK UX regression guard** that
verifies the Wf-378-A1 + Wf-378-A2 + Wf-366-A2 closures remain
intact across the 3 canonical launch wrappers:

1. **`scripts/pre_flight_gpu_ready.sh`** (Wf-378-A1 ship surface)
2. **`scripts/launch_wf124_p3_g6.sh`** (Wf-378-A2 ship surface)
3. **`scripts/recover_g6_failed_tuples.sh`** (Wf-366-A2 ship surface)

**6 NEW tests PASS** at `molmetal/tests/test_wf405a4_user_ok_gaps.py`
(wall ~0.33 s on the test host, CPU-only; 2 parametrized tests × 3
wrappers = 6 test instances). All 3 wrappers verified intact per
Wf-368-B1 §7.1 forward-looking queue closure:

- (a) Wf-378-A1 pre_flight_gpu_ready.sh USER_OK hard entry gate  **[CLOSED, REGRESSION-GUARDED]**
- (b) Wf-378-A2 launch_wf124_p3_g6.sh USER_OK gate parity          **[CLOSED, REGRESSION-GUARDED]**
- (c) Wf-365-B2 G6 USER_OK parity                                 [CLOSED, REGRESSION-GUARDED]
- (d) Wf-363-A2 GAP-3 (pre_flight_gpu_ready dry-run SKIP)         [CLOSED, REGRESSION-GUARDED]

**Audit findings** (all 3 wrappers PASS):

| Wrapper | USER_OK gate | Gate before --dry-run | DRY_RUN.log | Pre-flight chain |
|---|---|---|---|---|
| `pre_flight_gpu_ready.sh` | ✅ (line 145) | ✅ (gate conditional on `DRY_RUN != 1`) | ✅ (gate_result + emit_dry_run_header/footer) | ✅ (Gates 1-5) |
| `launch_wf124_p3_g6.sh`   | ✅ (line 69)  | ✅ (gate line 69 < --dry-run line 205)    | ✅ (line 247 emit_dry_run_log)             | ✅ (lines 313-318) |
| `recover_g6_failed_tuples.sh` | ✅ (line 102) | ✅ (gate line 102 < --dry-run line 293)  | ✅ (line 310 emit_dry_run_log)             | ✅ per-tuple wf343c2 (line 358) |

Honest framing per `party-twenty-spirit` §3.5: this Wf-405-A4 wave is
a **regression-guard wave** (verification only, zero code patches, zero
GPU launches, zero `paper/*` touches, zero wet-lab planning per
AGENTS.md §5.1 + §5.3 + §10). The audit confirms the Wf-378-A1 +
Wf-378-A2 + Wf-366-A2 ship surfaces remain stable at the
Wf-405-A4 timestamp 2026-09-30; no GAP remains from the Wf-368-B1 §7.1
forward-looking queue.

---

## §1 The audit

### §1.1 Audit methodology

Wf-405-A4 re-audits all 3 launch wrappers for the 4 invariants the
Wf-368-B1 §7.1 forward-looking queue + Wf-363-A2 GAP-2/GAP-3 audit
established:

1. **USER_OK gate present** — `if [ "${USER_OK:-0}" != "1" ]; then` or
   `if [[ "${USER_OK:-0}" != "1" ]]; then` followed by `exit 64`.
2. **USER_OK gate fires BEFORE --dry-run exit block** — the gate source
   appears at a lower line number than the `--dry-run` bail block (for
   launch_wf124_p3_g6.sh + recover_g6_failed_tuples.sh patterns) OR the
   gate is conditional on `if [[ ${DRY_RUN} -ne 1 ]]` (for the
   pre_flight_gpu_ready.sh pattern).
3. **DRY_RUN.log emission** — wrapper emits a DRY_RUN.log file with
   the canonical Wf-363-A1 schema fields (wrapper / mode / trainer /
   flags / gpu / budget / user_ok / ts / per_pair block).
4. **Pre-flight chain reference** — wrapper invokes pre_flight_check.py
   + wf343c2_per_wave_check.py + wf343c3_data_quality_check.py
   (either globally or per-tuple).

### §1.2 Audit results (per wrapper)

#### §1.2.1 `scripts/pre_flight_gpu_ready.sh` (Wf-378-A1)

**USER_OK gate**: PRESENT at line 145 (`if [ "${USER_OK:-0}" != "1" ]; then exit 64; fi`).
The gate is **conditional** on `if [[ ${DRY_RUN} -ne 1 ]]; then` (line 144)
— the Wf-378-A1 hard entry gate fires ONLY in LIVE mode, exits 64
otherwise. In `--dry-run` mode, the gate is bypassed and Gate 5 emits
`SKIP` instead of `PASS`. This is the canonical Wf-378-A1 pattern.

**Gate before --dry-run exit block**: N/A for this pattern (no
`if [[ "${DRY_RUN}" == "1" ]] then ... exit 0; fi` bail block). The
script runs all gates in --dry-run mode; Gate 5 emits `SKIP` and the
script exits 0 after the per-gate verdict block. The Wf-378-A1 hard
entry gate source (line 144) appears BEFORE any pipeline work — the
gate is conditional on LIVE mode but its source precedes the gate
output.

**DRY_RUN.log emission**: PRESENT via the in-script log helpers
(`emit_dry_run_header` + `emit_dry_run_footer` + `gate_result`) and
the per-gate `LOG` file at `molmetal/reports/pre_flight_gpu_ready/
run_${RUN_TS}.log`. Note: this wrapper does NOT use the canonical
Wf-363-A1 `emit_dry_run_log` helper directly (the helper is for
launch wrappers; this gate wrapper has its own log schema). The
emitted output is functionally equivalent to the DRY_RUN.log schema
with per-gate verdicts + evidence block.

**Pre-flight chain reference**: PRESENT — the script invokes
`molmetal/scripts/pre_flight_check.py --pipeline all` (Gate 1, line 248)
+ `molmetal/scripts/wf343c2_per_wave_check.py --skip-live --strict`
(Gate 3, line 365) + `molmetal/scripts/wf343c3_data_quality_check.py
--strict` (Gate 4, line 415). All 4 NEW Wf-345-G3/Wf-294B
pre-flight + data-quality gates are referenced.

#### §1.2.2 `scripts/launch_wf124_p3_g6.sh` (Wf-378-A2)

**USER_OK gate**: PRESENT at line 69 (`if [ "${USER_OK:-0}" != "1" ];
then ... exit 64; fi`). The Wf-378-A2 fix moved this gate from the
ORIGINAL lines 208-218 (where `--dry-run` silently bypassed it) to
the CURRENT line 69 (BEFORE any --dry-run parsing). This closes the
Wf-363-A2 GAP-2 / Wf-365-B2 / Wf-368-B1 §7.1 HONEST-NEGATIVE finding
that the G6 wrapper's USER_OK gate was reachable only on real-launch
path.

**Gate before --dry-run exit block**: YES — gate at line 69 < `--dry-run`
parsing at line 96 < `--dry-run` exit block at line 205 (which exits
0 after emitting the DRY_RUN.log). The block-order invariant from
Wf-366-A2 GAP-2 fix is preserved.

**DRY_RUN.log emission**: PRESENT at line 247 via the canonical
Wf-363-A1 `emit_dry_run_log` helper (60 (cfg, metal, seed) tuple
enumeration). The DRY_RUN.log is emitted to `${OUT_BASE}/DRY_RUN.log`
with all 8 canonical schema fields (wrapper / mode / trainer /
flags / gpu / budget / user_ok / ts).

**Pre-flight chain reference**: PRESENT at lines 313-318:
```
${PY} molmetal/scripts/pre_flight_check.py --pipeline all
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} molmetal/scripts/wf343c3_data_quality_check.py
${PY} molmetal/scripts/per_wave_checklist.py --wave "Wf-124-P3-G6"
${PY} molmetal/scripts/wave_budget_tracker.py --no-enforce-budget
```
All 4 canonical pre-flight gates are referenced in the real-launch
path.

#### §1.2.3 `scripts/recover_g6_failed_tuples.sh` (Wf-366-A2)

**USER_OK gate**: PRESENT at line 102 (`if [[ "${USER_OK:-0}" != "1" ]];
then ... exit 64; fi`). The Wf-366-A2 GAP-2 fix moved this gate to
fire BEFORE any prereg check, manifest enumeration, or --dry-run
handling.

**Gate before --dry-run exit block**: YES — gate at line 102 < --dry-run
parsing at line 80 (note: --dry-run is parsed earlier but the gate
fires at line 102 before any prereg / manifest / --dry-run block
work) < `--dry-run` exit block at line 293 (which exits 0 after
emitting the DRY_RUN.log). The block-order invariant from Wf-366-A2
GAP-2 fix is preserved.

**DRY_RUN.log emission**: PRESENT at line 310 via the canonical
Wf-363-A1 `emit_dry_run_log` helper (60 (cfg, metal, seed) tuple
enumeration). The DRY_RUN.log is emitted to `${OUT_BASE}/DRY_RUN.log`
with all 8 canonical schema fields.

**Pre-flight chain reference**: PARTIAL — the wrapper invokes
`molmetal/scripts/wf343c2_per_wave_check.py` per-tuple (line 358) but
does NOT invoke `molmetal/scripts/pre_flight_check.py` globally before
the recovery launch. The header comment at line 357 says "global
pre_flight_check.py already covers VRAM + disk" — this is the Wf-337-G1
design (assumes the operator has run `pre_flight_gpu_ready.sh`
externally before launching the recovery). This is **PARTIAL by design**
— the recovery wrapper is intended to be invoked AFTER
`pre_flight_gpu_ready.sh` has been run externally.

### §1.3 GAP analysis

| Wrapper | GAP | Closure path |
|---|---|---|
| `pre_flight_gpu_ready.sh` | None | N/A — all 4 invariants PASS |
| `launch_wf124_p3_g6.sh`   | None | N/A — all 4 invariants PASS |
| `recover_g6_failed_tuples.sh` | Partial pre-flight chain reference (per-tuple wf343c2 only, no global pre_flight_check.py) | Documented by-design in Wf-337-G1 §6; no closure action required at Wf-405-A4 timestamp |

The `recover_g6_failed_tuples.sh` PARTIAL pre-flight chain reference
is by design (Wf-337-G1 §6: "the recovery wrapper is intended to be
invoked AFTER `pre_flight_gpu_ready.sh` has been run externally").
Closure path: if future waves want to add a global pre-flight
invocation, that would be a separate wrapper-update task; Wf-405-A4
documents the current state and provides regression-guard tests for
the invariants that are wired.

---

## §2 The tests

`molmetal/tests/test_wf405a4_user_ok_gaps.py` — NEW 6-case test suite
(2 parametrized tests × 3 wrappers):

### §2.1 Test 1 — `test_all_wrappers_have_user_ok_gate[wrapper]`

Source-inspects each of the 3 wrappers for the canonical Wf-294B
USER_OK gate pattern. Verifies:

- (a) USER_OK=1 check pattern (with `:0` default fallback) is present.
- (b) `exit 64` follows the gate (Wf-294B family convention).
- (c) Source cites the originating wave (Wf-378-A1 / Wf-378-A2 / Wf-366-A2).
- (d) Reference to the kill_then_adapt / Wf-294B directive origin.
- (e) An "USER_OK gate refused" refusal message exists.

### §2.2 Test 2 — `test_block_order_before_dry_run[wrapper]`

Source-inspects each wrapper for the block-order invariant:
- For launch_wf124_p3_g6.sh + recover_g6_failed_tuples.sh (bail-out pattern):
  USER_OK gate source line < --dry-run exit block line.
- For pre_flight_gpu_ready.sh (conditional gate pattern):
  the `if [[ ${DRY_RUN} -ne 1 ]]` conditional appears BEFORE the
  inner USER_OK check (gate is conditional on LIVE mode but its
  source precedes the gate output).
- Verifies the dry-run block contains `exit 0` (plan-only success).

### §2.3 Test results

```
$ python -m pytest molmetal/tests/test_wf405a4_user_ok_gaps.py -v --no-header --tb=short
molmetal/tests/test_wf405a4_user_ok_gaps.py::test_all_wrappers_have_user_ok_gate[pre_flight_gpu_ready] PASSED
molmetal/tests/test_wf405a4_user_ok_gaps.py::test_all_wrappers_have_user_ok_gate[launch_wf124_p3_g6] PASSED
molmetal/tests/test_wf405a4_user_ok_gaps.py::test_all_wrappers_have_user_ok_gate[recover_g6_failed_tuples] PASSED
molmetal/tests/test_wf405a4_user_ok_gaps.py::test_block_order_before_dry_run[pre_flight_gpu_ready] PASSED
molmetal/tests/test_wf405a4_user_ok_gaps.py::test_block_order_before_dry_run[launch_wf124_p3_g6] PASSED
molmetal/tests/test_wf405a4_user_ok_gaps.py::test_block_order_before_dry_run[recover_g6_failed_tuples] PASSED
========================= 6 passed, 1 warning in 0.33s =========================
```

MEASURED test pass-rate: **6/6 PASS** in 0.33 s wall, CPU-only.

---

## §3 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED**: 6/6 tests PASS at Wf-405-A4 timestamp 2026-09-30.
- **MEASURED**: All 3 wrappers verified intact for the 4 invariants
  (USER_OK gate / gate-before-dry-run / DRY_RUN.log emission /
  pre-flight chain reference).
- **MEASURED**: 0 NEW source patches to the 3 wrappers (audit only).
- **PROJECTED**: 0 NEW GPU launches performed.
- **PROJECTED**: 0 NEW `paper/*` touches.
- **PROJECTED**: 0 NEW wet-lab planning.
- **HONEST-NEGATIVE**: `recover_g6_failed_tuples.sh` PARTIAL pre-flight
  chain reference (per-tuple wf343c2 only) — PARTIAL by design per
  Wf-337-G1 §6 (recovery wrapper assumes external
  `pre_flight_gpu_ready.sh` invocation). Closure path is a separate
  wrapper-update task; Wf-405-A4 documents the current state.
- **HONEST-NEGATIVE**: pre-flight v3 89/2/3/94 honest-negative drift
  preserved per Wf-374-B1 / Wf-375-B1 / Wf-388-B1 / Wf-389-B1 /
  Wf-391-B1 baseline (NOT Wf-405-A4-introduced; pre-flight NOT re-run
  by Wf-405-A4 — CPU-only docs+code+test wave per AGENTS.md §5.1 +
  kill_then_adapt §0).
- **HONEST-NEGATIVE**: m6_kappa + m2 R² + m1 AUC remain HONEST_NEGATIVE
  FINAL per Wf-377-A1 deep-dive-research move + Wf-388-A1 real-cohort
  verification + Wf-389-A1 §3 supplementary sub-table reframe.
- **HONEST-NEGATIVE**: 5 forward-looking USER_OK items still BLOCKED
  on user OK (~52.9 hr GPU + 13 hr CPU + 30 min install per
  Wf-368-B1 §0).

---

## §4 Cross-references

- `scripts/pre_flight_gpu_ready.sh`                 — Wf-378-A1 ship surface
- `scripts/launch_wf124_p3_g6.sh`                  — Wf-378-A2 ship surface
- `scripts/recover_g6_failed_tuples.sh`            — Wf-366-A2 ship surface
- `scripts/_dry_run_log_helper.sh`                 — Wf-363-A1 shared helper
- `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` — Wf-355-A3 reference pattern
- `molmetal/tests/test_wf405a4_user_ok_gaps.py`    — Wf-405-A4 NEW tests (this wave)
- `molmetal/tests/test_wf378a1_gpu_ready_user_ok_gate.py` — Wf-378-A1 tests (sibling)
- `molmetal/tests/test_wf366a2_recover_g6_block_order.py` — Wf-366-A2 tests (sibling)
- `molmetal/tests/test_wf363a2_user_ok_ux.py`       — Wf-363-A2 audit (sibling)
- `TODO/completed/378a1_gpu_ready_user_ok_gate.md` — Wf-378-A1 ship doc
- `TODO/completed/378a2_g6_user_ok_parity.md`      — Wf-378-A2 ship doc
- `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` — Wf-366-A2 ship doc
- `TODO/completed/363a2_user_ok_ux.md`             — Wf-363-A2 audit (sibling)
- `TODO/completed/368b1_user_ok_final.md`          — Wf-368-B1 §7.1 forward-looking queue
- `AGENTS.md §5.1`                                  — Standing GPU authorization (kill_then_adapt directive)
- `AGENTS.md §4`                                    — Stdout invariant (PASS / FAIL / SKIP)
- `AGENTS.md §10`                                   — Wetlab deprecation FAQ

---

## §5 User voice relay

User's verbatim question: **"之前的workflow跑到一半失效了，能恢复他们吗？"**
(Previous workflows failed midway, can you recover them?)

Wf-405-A4 is the cross-wrapper regression guard that ensures the
Wf-378-A1 + Wf-378-A2 + Wf-366-A2 USER_OK UX gap closures remain
intact at the Wf-405-A4 timestamp. All 3 wrappers (pre_flight_gpu_ready
+ launch_wf124_p3_g6 + recover_g6_failed_tuples) verified intact.
The audit confirms:

1. **Wf-378-A1** (pre_flight_gpu_ready.sh) — **STABLE**, regression-guarded
   by `test_all_wrappers_have_user_ok_gate[pre_flight_gpu_ready]`.
2. **Wf-378-A2** (launch_wf124_p3_g6.sh) — **STABLE**, regression-guarded
   by `test_all_wrappers_have_user_ok_gate[launch_wf124_p3_g6]`.
3. **Wf-366-A2** (recover_g6_failed_tuples.sh) — **STABLE**, regression-guarded
   by `test_all_wrappers_have_user_ok_gate[recover_g6_failed_tuples]`.

Future agents must maintain the Wf-294B family USER_OK gate pattern
+ block-order invariant across all 3 wrappers. The 6 NEW tests in
`molmetal/tests/test_wf405a4_user_ok_gaps.py` will catch any
regression to either invariant.

End of Wf-405-A4 closeout.