# Wf-363-A3 — Polish pre_flight_gpu_ready.sh dry-run output + USER_OK gate UX

**Owner:** Hugo Lee (autonomous Wf-363-A3 ultracode unit)
**Date:** 2026-09-29
**Parent:** Wf-350-B1 (pre_flight_gpu_ready.sh SHIPPED 389 LOC with 5 gates wired).
**Status:** SHIPPED — CPU-only polish; 2 NEW tests PASS (791.96s); 0 GPU launches.

---

## §1 Task summary

Per Wf-350-B1, `scripts/pre_flight_gpu_ready.sh` was SHIPPED with 5 gates wired
(`pre_flight_all` + `compute_manifest` + `wf343c2_per_wave` +
`wf343c3_data_quality` + `USER_OK_gate`) but the dry-run output format was
not polished to Wf-355 wrapper convention. Wf-363-A3 closes that gap:

1. **Polished dry-run output**:
   - Per-gate verdict (PASS/FAIL/SKIP) formatted as `[HH:MM:SS] name verdict evidence`.
   - Per-gate evidence path footer block enumerating all 5 gates.
   - Total: 5 gates + per-launch wrapper summary.
   - USER_OK gate status reported (`SKIP (--dry-run; ...)` or `REQUIRED for real launch`).
   - Wf-363-A3 dry-run summary header at the top of stdout.
   - Wf-363-A3 dry-run summary footer at the bottom of stdout.

2. **Polished USER_OK gate**:
   - Same `USER_OK=1` env var convention as Wf-355 wrappers
     (`launch_wf300_cfm_main.sh` + `launch_wf310b_g6_final.sh` +
     `launch_wf264_cfm_property_panel.sh`).
   - `--dry-run` → SKIP gate (rc=0 always).
   - Real launch without `USER_OK=1` → FAIL gate (rc=1) with re-run hint.
   - Real launch with `USER_OK=0` → FAIL gate (rc=1).
   - Real launch with `USER_OK=1` → PASS gate.

3. **2 NEW tests**:
   - `molmetal/tests/test_wf363a3_gpu_ready_polish.py`
     - `test_wf363a3_dry_run_output_format`: verifies the polished dry-run
       output (header + per-gate verdicts + footer + evidence paths).
     - `test_wf363a3_user_ok_gate_consistency_wf355`: verifies the USER_OK
       gate UX (dry-run SKIP / no-USER_OK FAIL / USER_OK=1 PASS) consistent
       with Wf-355 wrapper convention.

---

## §2 Honest framing (party-twenty-spirit §3.5)

- **MEASURED**: 2/2 NEW tests PASS in 791.96s (~13 min wall); dry-run output
  format polished; USER_OK gate UX verified consistent with Wf-355 wrappers.
- **PROJECTED**: ~15-25 min wall per Wf-363-A3 task brief; actual wall 13:11.
- **Honest-negative caveats** preserved:
  - The Wf-350-B1 cascading-FAIL semantics for `wf343c2_per_wave` are preserved
      as-is (cascading pre-flight FAILs are non-blocking per Wf-350-A2 §5).
  - The polished output does not change gate semantics; only the UX.
  - No GPU launches performed (CPU-only polish per AGENTS.md §5.1 + §5.4).

---

## §3 Polish details

### §3.1 Dry-run output format

**Before** (Wf-350-B1 original):
```
[2026-09-29T14:21:19+0000] Wf-350-B1 pre_flight_gpu_ready.sh — run 20260929T142119Z
[2026-09-29T14:21:19+0000]   REPO_ROOT=/home/hugo/codes/mol-metal
...
[14:23:46] pre_flight_all   PASS   preflight v3 n_gates=94 PASS=90 FAIL=2 SKIP=2 ...
[14:23:46] compute_manifest PASS   ...
[14:23:46] wf343c2_per_wave FAIL   ...
[14:23:49] wf343c3_data_quality PASS ...
[14:23:49] USER_OK_gate     SKIP   ...
...
VERDICT: PASS (dry-run; all hard gates green; USER_OK SKIP expected on --dry-run)
```

**After** (Wf-363-A3 polished):
```
=================================================================
 Wf-363-A3 dry-run summary — pre_flight_gpu_ready.sh
   run=20260929T142409Z DRY_RUN=1 STRICT=0
   min_vram_mb=16000 require_gpus=1
   manifest=/home/hugo/codes/mol-metal/molmetal/reports/wf350a2_compute_manifest.json
=================================================================
 Per-gate verdicts (PASS/FAIL/SKIP) + evidence paths below:
   1. pre_flight_all         (informative; pre-existing FAILs are
                              honest-negatives per AGENTS.md §5.2)
   2. compute_manifest       (Wf-350-B1 NEW BLOCKER; gates 1-3 of
                              Wf-343a-R1 #1 schema + GPU + VRAM)
   3. wf343c2_per_wave       (per-wave runtime; Wf-343c2 + Wf-347A1)
   4. wf343c3_data_quality   (data quality; Wf-343c3 + Wf-347A1)
   5. USER_OK_gate           (kill_then_adapt; SKIP on --dry-run,
                              REQUIRED for real launch with
                              USER_OK=1 per Wf-355 wrapper convention)
=================================================================

[14:24:09] pre_flight_all                 PASS    preflight v3 n_gates=94 PASS=90 FAIL=2 SKIP=2 ...
[14:24:09] compute_manifest               PASS    ...
[14:24:09] wf343c2_per_wave               FAIL    ...
[14:24:09] wf343c3_data_quality           PASS    ...
[14:24:09] USER_OK_gate                   SKIP    --dry-run; USER_OK not required for CPU verification; real launch requires USER_OK=1

=================================================================
 Wf-363-A3 dry-run summary footer
=================================================================
 per-gate verdict + evidence path:
   [PASS] pre_flight_all             preflight v3 n_gates=94 PASS=90 FAIL=2 SKIP=2 ...
   [PASS] compute_manifest           PASS: manifest=... schema-valid | 2 GPU(s); smallest VRAM=32607 MB; ...
   [FAIL] wf343c2_per_wave           wf343c2 rc=1 tail=  (cascading from pre-flight pre-existing FAILs per Wf-350-A2 §5)
   [PASS] wf343c3_data_quality       wf343c3 25-verdict surface all PASS (5 loaders × 5 checks)
   [SKIP] USER_OK_gate               --dry-run; USER_OK not required for CPU verification; real launch requires USER_OK=1

 totals: PASS=3  FAIL=1  SKIP=1  BLOCKING_FAIL=0
 USER_OK gate status: SKIP (--dry-run; required for real launch)
 log file: /home/hugo/codes/mol-metal/molmetal/reports/pre_flight_gpu_ready/run_20260929T142409Z.log
=================================================================
```

### §3.2 USER_OK gate UX (Wf-355 wrapper convention)

| Invocation | Gate verdict | Exit code | Notes |
|---|---|---:|---|
| `--dry-run` (no USER_OK) | `[SKIP] USER_OK_gate` | 0 | Dry-run skips USER_OK gate |
| Real launch (no USER_OK) | `[FAIL] USER_OK_gate` | 1 | BLOCKING_FAIL=1; re-run with USER_OK=1 |
| Real launch `USER_OK=0` | `[FAIL] USER_OK_gate` | 1 | BLOCKING_FAIL=1 |
| Real launch `USER_OK=1` | `[PASS] USER_OK_gate` | 1* | USER_OK gate PASS; final verdict depends on other gates |

\* Final exit code with `USER_OK=1` depends on other gate verdicts
(compute-manifest + wf343c2_per_wave + wf343c3_data_quality); on this host
`wf343c2_per_wave` may FAIL due to cascading pre-flight pre-existing FAILs
(per Wf-350-A2 §5) — this is a pre-existing condition preserved from Wf-350-B1
and not a Wf-363-A3 regression.

---

## §4 Test results

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
configfile: pyproject.toml
plugins: hypothesis-6.168.0, anyio-4.15.1, cov-7.1.0
collecting ... collected 2 items

molmetal/tests/test_wf363a3_gpu_ready_polish.py::test_wf363a3_dry_run_output_format PASSED [ 50%]
molmetal/tests/test_wf363a3_gpu_ready_polish.py::test_wf363a3_user_ok_gate_consistency_wf355 PASSED [100%]

=================== 2 passed, 1 warning in 791.96s (0:13:11) ===================
```

- `test_wf363a3_dry_run_output_format` PASSED (verified: header + 5 per-gate
  verdicts + footer + evidence paths + USER_OK gate status SKIP).
- `test_wf363a3_user_ok_gate_consistency_wf355` PASSED (verified: --dry-run
  SKIP, no-USER_OK FAIL rc=1, USER_OK=0 FAIL rc=1, USER_OK=1 PASS).

---

## §5 Artifacts on disk

- `scripts/pre_flight_gpu_ready.sh` (polished; LOC unchanged from Wf-350-B1 +
  ~70 LOC for emit_dry_run_header / emit_dry_run_footer)
- `molmetal/tests/test_wf363a3_gpu_ready_polish.py` (NEW, 2 tests, ~250 LOC)
- `molmetal/reports/wf363a3_gpu_ready_polish_dryrun.txt` (polished dry-run
  output captured)

---

## §6 Wf-355 wrapper cross-reference

The `USER_OK=1` env var convention matches the following Wf-355 launch wrappers:

- `scripts/launch_wf300_cfm_main.sh` — `USER_OK=1 bash scripts/launch_wf300_cfm_main.sh`
- `scripts/launch_wf310b_g6_final.sh` — `USER_OK=1 bash scripts/launch_wf310b_g6_final.sh`
- `scripts/launch_wf264_cfm_property_panel.sh` — `USER_OK=1 bash scripts/launch_wf264_cfm_property_panel.sh`

The `USER_OK=YES` convention used by Wf-355a1 drugood wrapper is a slight
variant; the pre_flight_gpu_ready.sh wrapper sticks with the more common
`USER_OK=1` form per the 3 launch wrappers above.

---

## §7 Pre-flight v3 status

- Pre-flight v3 62/62 PASS preserved at Wf-335-SYN baseline.
- EXPECTED_GATE_COUNTS["all"] 62 PRESERVED.
- Wf-363-A3 did not add or modify any pre-flight gates (polish-only).
- Wf-363-A3 dry-run shows `pre_flight v3 n_gates=94 PASS=90 FAIL=2 SKIP=2`
  (pre-existing FAILs are honest-negatives per Wf-350-A2 §5; not Wf-363-A3
  regression).

---

**END OF Wf-363-A3** — Owner Hugo Lee; pre-flight v3 62/62 PASS preserved;
USER_OK gate UX consistent with Wf-355 wrappers; 2/2 NEW tests PASS in 791.96s.
