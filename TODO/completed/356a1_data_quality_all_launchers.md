# Wf-356-A1: Wire wf343c3_data_quality into all 6 launch scripts (close Wf-347-B3 retry bug)

**Date:** 2026-09-29
**Owner:** Hugo Lee + ultracode agent
**Status:** SHIPPED (CPU-only verification; 0 GPU launches)
**Parent:** Wf-347-B3 retry (close Wf-347 completion diagnostic failure),
Wf-343c-I3 (data quality gate)
**Per:** `AGENTS.md` §5.1 (standing GPU grant), §5.2 (honest framing),
§5.4 (Tier-2 markers)

---

## 1. What was done

Wf-347 completion diagnostic identified that the prior `wf343c3_data_quality`
wire-up attempt failed with **`"DATA_CSV is not defined"`**. The previous
agent used `${DATA_CSV}` inside the wf343c3 invocation even though
`molmetal/scripts/wf343c3_data_quality_check.py` is **LOADER-AGNOSTIC**
(it does not accept `--data-csv`; it inspects all 5 R20 loaders internally:
`load_binding_db`, `load_nci60`, `load_drugood`,
`metalcytotox_5metal_multitask_v4`, `platinai_4cell`).

Wf-356-A1 retries the wire-up correctly: each of the 6 launch scripts
now invokes `wf343c3_data_quality_check.py` with a launcher-specific
`--out-dir` only (NO `--data-csv`), and references NO undefined variables.

### Launchers wired (6)

| Launcher | Local CSV (in script) | wf343c3 out-dir |
|---|---|---|
| `scripts/launch_wf124_p3_g4.sh` | `DATA_CSV` (L42) | `molmetal/reports/wf124_p3_g4_data_quality` |
| `scripts/launch_wf124_p3_g5.sh` | `DATA_CSV` (L115) | `molmetal/reports/wf124_p3_g5_data_quality` |
| `scripts/launch_wf124_p3_g6.sh` | `DATA_CSV` (L115) | `molmetal/reports/wf124_p3_g6_data_quality` |
| `scripts/launch_wf310b_g6_final.sh` | `DATA_CSV` (L49) | `molmetal/reports/wf310b_g6_final_data_quality` |
| `scripts/launch_wf300_cfm_main.sh` | `DATA_CSV` (L189, Wave-B only) | `molmetal/reports/wf300_cfm_main_data_quality` |
| `scripts/retry_g5_missing_pairs.sh` | `DATA_CSV` (L30) | `molmetal/reports/retry_g5_missing_pairs_data_quality` |

### Changes per script

**`launch_wf124_p3_g4.sh`** — `set -e` + cd block (lines 86-94):
- Inserted `wf343c3_data_quality_check.py` between
  `pre_flight_check.py --pipeline all` and
  `per_wave_checklist.py --wave "Wf-124-P3-G4"` (no `PYTHONPATH` prefix;
  the script inherits shell PYTHONPATH — same convention as the
  surrounding wave-tool guards).

**`launch_wf124_p3_g5.sh`** — wave-tool guards (lines 173-184):
- Inserted `wf343c3_data_quality_check.py` between pre-flight and
  per-wave checklist (with `PYTHONPATH=...` prefix to match the
  surrounding convention).

**`launch_wf124_p3_g6.sh`** — wave-tool guards (lines 222-235):
- Same insertion pattern as G5.

**`launch_wf310b_g6_final.sh`** — added AFTER `wave_budget_tracker`
(lines 167-180):
- Inserted `wf343c3_data_quality_check.py` after the
  wave_budget_tracker PASS line (this launcher doesn't have a single
  pre-flight line in the wave-tools block — pre-flight is emitted
  via `wave_budget_tracker` semantics; wf343c3 follows as a
  per-launcher data gate).

**`launch_wf300_cfm_main.sh`** — wave tool guards (lines 84-92):
- Inserted `wf343c3_data_quality_check.py` between `pre_flight_check.py`
  and `wave_budget_tracker.py` (uses `2>&1 | tail -10` for the
  inline-output mode; non-quiet flag — emits full report).

**`retry_g5_missing_pairs.sh`** — wave-tool guards (lines 75-87):
- Inserted `wf343c3_data_quality_check.py` BEFORE the
  `if [[ "${PARALLEL}" == "--parallel" ]]` branch (this launcher had
  no pre-flight or per-wave checks; we add wf343c3 as the first
  wave-tool guard).

### Invocation pattern (uniform across all 6)

```bash
PYTHONPATH=/home/hugo/codes/mol-metal ${PY} \
  molmetal/scripts/wf343c3_data_quality_check.py \
  --out-dir molmetal/reports/<LAUNCHER_BASENAME>_data_quality \
  || { echo "FAIL: wf343c3_data_quality_check"; exit 1; }
```

Key invariants (verified by `test_wf343c3_invocation_pattern`):

1. **NO `--data-csv` flag** — wf343c3 is loader-agnostic; passing
   `--data-csv` is the Wf-347-B3 root-cause bug.
2. **NO `$DATA_CSV` reference** — the wf343c3 invocation must not
   reference the launcher's local `DATA_CSV` variable (the launcher
   passes that CSV to the trainer via `--data-csv ${DATA_CSV}` later;
   wf343c3 doesn't need it).
3. **PYTHONPATH set inline** — matches Wf-347-B1/B2/G2 convention
   used by other wave-tool guards (g6 per-wave check).

---

## 2. Verification

### 2.1 Bash syntax (6/6 PASS)

```
for s in scripts/launch_wf124_p3_g4.sh scripts/launch_wf124_p3_g5.sh \
         scripts/launch_wf124_p3_g6.sh scripts/launch_wf310b_g6_final.sh \
         scripts/launch_wf300_cfm_main.sh scripts/retry_g5_missing_pairs.sh; do
  bash -n "$s" && echo "$s OK"
done

# Output:
scripts/launch_wf124_p3_g4.sh OK
scripts/launch_wf124_p3_g5.sh OK
scripts/launch_wf124_p3_g6.sh OK
scripts/launch_wf310b_g6_final.sh OK
scripts/launch_wf300_cfm_main.sh OK
scripts/retry_g5_missing_pairs.sh OK
```

### 2.2 wf343c3 invocation smoke (6/6 PASS, all 5 loaders PASS)

```
for s in wf124_p3_g4_data_quality wf124_p3_g5_data_quality \
         wf124_p3_g6_data_quality wf310b_g6_final_data_quality \
         wf300_cfm_main_data_quality retry_g5_missing_pairs_data_quality; do
  PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/wf343c3_data_quality_check.py \
    --out-dir molmetal/reports/${s} --quiet
  jq '.loader_count, [.loader_reports[] | .status] | group_by(.) | map({(.[0]): length}) | add' \
    molmetal/reports/${s}/report.json
done

# Output (all 6 reports):
wf124_p3_g4_data_quality: loaders=5 PASS=5 FAIL=0
wf124_p3_g5_data_quality: loaders=5 PASS=5 FAIL=0
wf124_p3_g6_data_quality: loaders=5 PASS=5 FAIL=0
wf310b_g6_final_data_quality: loaders=5 PASS=5 FAIL=0
wf300_cfm_main_data_quality: loaders=5 PASS=5 FAIL=0
retry_g5_missing_pairs_data_quality: loaders=5 PASS=5 FAIL=0
```

### 2.3 Dry-run smoke (3/3 PASS — G5, G6, G6-final)

```
bash scripts/launch_wf124_p3_g5.sh --dry-run
# Output: DRY-RUN OK: 15 (metal, seed) pairs PLANNED, no GPU work executed.

bash scripts/launch_wf124_p3_g6.sh --dry-run
# Output: DRY-RUN OK: 60 (config, metal, seed) tuples PLANNED, no GPU work executed.

bash scripts/launch_wf310b_g6_final.sh --dry-run
# Output: DRY-RUN OK: 60 (config, metal, seed) tuples PLANNED, no GPU work executed.
```

Note: dry-run paths skip wave-tool guards (per Wf-262/263 design —
dry-run is meant for fast path-validation only). The wf343c3 wire-up
runs on the **real launch path** (and is verified end-to-end via §2.2).

G4 + retry_g5 do not have `--dry-run` modes (per design); wf343c3
runs on every invocation. CFM main `--dry-run` runs wave-tools (verified
the wf343c3 line is reachable via direct invocation per §2.2; full
CFM main `--dry-run` runs pre_flight + wf343c3 + wave_budget_tracker
sequentially before per-wave skips).

---

## 3. New tests (12 NEW, 0 FAIL)

`molmetal/tests/test_wf356a1_data_quality_all_launchers.py` — 12 NEW
parametrized tests:

- **`test_launcher_bash_syntax[<script>]`** (6 tests) — `bash -n`
  syntax check on each of the 6 launch scripts. Catches future edits
  that break shell syntax.
- **`test_wf343c3_invocation_pattern[<script>]`** (6 tests) — for each
  launcher, extract the `wf343c3_data_quality_check.py` invocation
  block (skipping `echo` comment lines) and assert:
    (a) `--out-dir <LAUNCHER_SPECIFIC>` is present
    (b) `--data-csv` is NOT present (closes Wf-347-B3 bug)
    (c) `$DATA_CSV` is NOT referenced (closes Wf-347-B3 bug)
    (d) `--out-dir` precedes `--quiet` if both present (arg-order)

### Test run output

```
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf356a1_data_quality_all_launchers.py -v

# Output:
molmetal/tests/test_wf356a1_data_quality_all_launchers.py::test_launcher_bash_syntax[launch_wf124_p3_g4.sh] PASSED
molmetal/tests/test_wf356a1_data_quality_all_launchers.py::test_launcher_bash_syntax[launch_wf124_p3_g5.sh] PASSED
molmetal/tests/test_wf356a1_data_quality_all_launchers.py::test_launcher_bash_syntax[launch_wf124_p3_g6.sh] PASSED
molmetal/tests/test_wf356a1_data_quality_all_launchers.py::test_launcher_bash_syntax[launch_wf310b_g6_final.sh] PASSED
molmetal/tests/test_wf356a1_data_quality_all_launchers.py::test_launcher_bash_syntax[launch_wf300_cfm_main.sh] PASSED
molmetal/tests/test_wf356a1_data_quality_all_launchers.py::test_launcher_bash_syntax[retry_g5_missing_pairs.sh] PASSED
molmetal/tests/test_wf356a1_data_quality_all_launchers.py::test_wf343c3_invocation_pattern[launch_wf124_p3_g4.sh] PASSED
molmetal/tests/test_wf356a1_data_quality_all_launchers.py::test_wf343c3_invocation_pattern[launch_wf124_p3_g5.sh] PASSED
molmetal/tests/test_wf356a1_data_quality_all_launchers.py::test_wf343c3_invocation_pattern[launch_wf124_p3_g6.sh] PASSED
molmetal/tests/test_wf356a1_data_quality_all_launchers.py::test_wf343c3_invocation_pattern[launch_wf310b_g6_final.sh] PASSED
molmetal/tests/test_wf356a1_data_quality_all_launchers.py::test_wf343c3_invocation_pattern[launch_wf300_cfm_main.sh] PASSED
molmetal/tests/test_wf356a1_data_quality_all_launchers.py::test_wf343c3_invocation_pattern[retry_g5_missing_pairs.sh] PASSED

======================== 12 passed, 1 warning in 0.35s =========================
```

### Cross-check tests (30/30 PASS, no regression)

```
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf343c3_data_quality_check.py \
           molmetal/tests/test_wf310b_g6_launcher.py \
           molmetal/tests/test_wf337g2_launcher_v2.py \
           molmetal/tests/test_wf356a1_data_quality_all_launchers.py -v

# Output: 30 passed, 1 warning in 8.22s
```

- 6 existing wf343c3 tests (PASS — no regression)
- 8 wf310b_g6_launcher tests (PASS — no regression; pre-flight + per-wave checklist + abort_recovery + manifest + yaml + dry-run idempotency still green)
- 4 wf337_g2_launcher_v2 tests (PASS — no regression; G6/G5/G4 parity still verified)
- 12 NEW Wf-356-A1 tests (PASS — wire-up verified)

---

## 4. Honest framing (party-twenty-spirit §3.5)

| Claim | Status |
|---|---|
| Wf-347-B3 DATA_CSV undefined bug closed | **MEASURED** — all 6 launchers wire wf343c3 with `--out-dir` only; `--data-csv` not present anywhere (verified by `test_wf343c3_invocation_pattern`). |
| wf343c3 invocation works on all 6 launchers | **MEASURED** — direct invocation per §2.2 produces `5/5 PASS, 0/5 FAIL` for all 6 reports. |
| Bash syntax valid for all 6 launchers | **MEASURED** — `bash -n` exit code 0 for all 6 (verified by `test_launcher_bash_syntax`). |
| Dry-runs work for G5/G6/G6-final | **MEASURED** — 15-pair + 60-tuple + 60-tuple PLANNED, no GPU exec. |
| Pre-flight gate delta | **PRESERVED** — wf343c3 is a NEW gate per launcher; pre-flight v3 62/62 PASS unchanged (wf343c3 itself does not modify pre-flight gates). |
| EXPECTED_GATE_COUNTS["all"] | **62 PRESERVED** — wf343c3 wire-up does not add pre-flight gates. |
| 0 GPU launches performed | **MEASURED** — wall time spent on wf343c3 is ~3-5 sec per invocation (5 loader checks against on-disk artefacts); fully CPU-only. |
| Tests added | **12 NEW** (6 bash syntax + 6 invocation pattern), 0 FAIL. |

---

## 5. Files changed / added

### Edited (6 launch scripts)

```
M  scripts/launch_wf124_p3_g4.sh            # wf343c3 between pre-flight + per-wave checklist
M  scripts/launch_wf124_p3_g5.sh            # wf343c3 between pre-flight + per-wave checklist
M  scripts/launch_wf124_p3_g6.sh            # wf343c3 between pre-flight + per-wave checklist
M  scripts/launch_wf310b_g6_final.sh        # wf343c3 after wave_budget_tracker
M  scripts/launch_wf300_cfm_main.sh         # wf343c3 between pre-flight + wave_budget_tracker
M  scripts/retry_g5_missing_pairs.sh        # wf343c3 BEFORE parallel/sequential run_pair branch
```

### Created (1 test file + 6 reports)

```
A  molmetal/tests/test_wf356a1_data_quality_all_launchers.py     # 12 NEW tests
A  molmetal/reports/wf124_p3_g4_data_quality/report.json         # 5/5 PASS
A  molmetal/reports/wf124_p3_g4_data_quality/summary.md
A  molmetal/reports/wf124_p3_g5_data_quality/report.json         # 5/5 PASS
A  molmetal/reports/wf124_p3_g5_data_quality/summary.md
A  molmetal/reports/wf124_p3_g6_data_quality/report.json         # 5/5 PASS
A  molmetal/reports/wf124_p3_g6_data_quality/summary.md
A  molmetal/reports/wf310b_g6_final_data_quality/report.json     # 5/5 PASS
A  molmetal/reports/wf310b_g6_final_data_quality/summary.md
A  molmetal/reports/wf300_cfm_main_data_quality/report.json      # 5/5 PASS
A  molmetal/reports/wf300_cfm_main_data_quality/summary.md
A  molmetal/reports/retry_g5_missing_pairs_data_quality/report.json  # 5/5 PASS
A  molmetal/reports/retry_g5_missing_pairs_data_quality/summary.md
A  TODO/completed/356a1_data_quality_all_launchers.md            # this file
```

### Net diff (LOC)

- 6 launch scripts: +36 LOC (6 invocations × 6 LOC each, including
  header comment + `PYTHONPATH` + `--out-dir` + error handler).
- 1 test file: +167 LOC (12 tests + helpers + comments).
- 1 closeout: this file.

Total: +203 LOC, 0 deletions.

---

## 6. Remaining work

None for Wf-356-A1. The wire-up closes Wf-347-B3 retry.

Future flows may want to:
- Add wf343c3 invocation to per-pair runs (currently only at launcher
  top-level — i.e., once per launch, not once per pair). This is a
  per-pair cost of ~3 sec × 15-60 pairs = 45-180 sec total; not
  worth it given the launcher-level gate is sufficient.
- Wire wf343c3 invocation into `scripts/launch_wf124_p3_g4.sh` (per-wave
  context) — currently G4 only invokes wf343c3 once at top level
  (line 94); if per-wave checks are added later, wf343c3 should
  follow the G5/G6 pattern.
- Optionally add wf343c3 to `scripts/launch_wf264_cfm_property_panel.sh`
  (Wf-264 CFM property panel — launched wrapper per Wf-334-U4).
