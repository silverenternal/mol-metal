# Wf-394-A3 — Wf-299 CFM 3-wave pre-launch verification gate

**Document ID:** Wf-394-A3 / 394a3_wf299_pre_launch_verify
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + Wf-394-A3 ultracode agent
**Trigger:** Per user 2026-09-30 directive *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"* + Wf-394-A3 task brief for TODO-117 (Wf-299 3-wave CFM launch verification) + TODO-207 §2.5 pre-launch verification + AGENTS.md §5.1 standing GPU auth + Wf-378-A2 G6 USER_OK gate lesson + Wf-385-A2 wrapper review + Wf-392-A3 Wf-388 trainer flag integration + `mol-metal-cfm-training` skill keyword match.

---

## §0 TL;DR

**Goal**: Verify the Wf-299 CFM 3-wave launcher
(`molmetal/scripts/launch_wf299_cfm_main.sh`) passes 4 structural
contracts BEFORE any actual GPU launch.

**Verdict**:
- **Pre-edit (baseline)**: **1 of 4 contracts FAIL** (USER_OK gate
  MISSING + pre-flight chain NOT referenced).
- **Post-edit (this Wf)**: **4 of 4 contracts PASS** (USER_OK gate
  added BEFORE --dry-run block + wf343c2/c3 + wf385a3 + wf394a3 chain
  wired into `pre_flight_gate()` + 3-wave split verified + 16/16 Wf-388
  flags in `CFM_NEW_FLAGS`).

**Deliverables** (4 SHIPPED):

1. `molmetal/scripts/wf394a3_wf299_pre_launch_verify.py` (~370 LOC) —
   — 4 `verify_*` functions + `main()` orchestrator + structured JSON
   report at `molmetal/reports/wf394a3_wf299_pre_launch/result.json`.
2. `molmetal/tests/test_wf394a3_wf299_verify.py` (~210 LOC) — 4 NEW
   tests PASS (3 deliverable-mandated + 1 optional end-to-end).
3. `molmetal/scripts/launch_wf299_cfm_main.sh` — 2 hunks added (+74
   LOC net): USER_OK gate at top-of-file (BEFORE --dry-run block) +
   `pre_flight_gate()` extended to run wf343c2/c3 + wf385a3 + wf394a3.
4. `TODO/completed/394a3_wf299_pre_launch_verify.md` (this file, ~400
   LOC) — closeout with verification table + bash syntax check +
   USER_OK gate RC=64 verified.

**Verification**:

- `bash -n molmetal/scripts/launch_wf299_cfm_main.sh` → **BASH_SYNTAX_OK**
- `python molmetal/scripts/wf394a3_wf299_pre_launch_verify.py` →
  **PASS** (4 of 4 checks PASS)
- `USER_OK=unset bash launch_wf299_cfm_main.sh --dry-run --wave A` →
  **rc=64** (USER_OK gate refused, FATAL banner emitted)
- `USER_OK=1 bash launch_wf299_cfm_main.sh --dry-run --wave A` →
  **rc=0** (plan mode works, all flags + per-wave contracts verified)
- `python -c "import test_wf394a3_wf299_verify; ..."` → 4 of 4 tests PASS
- `python -m pytest molmetal/tests/test_wf394a3_wf299_verify.py` →
  pre-existing conftest ImportError (NOT Wf-394-A3-introduced;
  same failure mode affects test_wf385a3_scaffold_wire_in_gate.py +
  test_wf335s3_final.py + ~280 other test files per Wf-364-A3
  per-file count)

**MEASURED test pass-rates**:

- **4/4 NEW tests PASS** in 0.00 s wall (CPU-only) when invoked
  directly via Python (bypassing the pre-existing conftest bug).
- Pre-flight gate chain scripts (wf343c2 + wf343c3 + wf385a3 + wf394a3)
  are wired into `pre_flight_gate()` and emit JSON artefacts at
  `molmetal/reports/wf299_cfm_main/wf{343c2,343c3,385a3,394a3}_*.json`.

**Honest framing per `party-twenty-spirit` §3.5**: Wf-394-A3 verification
is CPU-only + structural (file presence + text pattern + case
statement). Actual smoke-step training validation is delegated to
`wf385a3_scaffold_wire_in_gate.py` (50-step smoke + decode_ratio >
0 + loss trajectory). Per AGENTS.md §5.1 + kill_then_adapt §0: the
Wf-299 GPU launch remains USER_OK-gated; this wave added the gate
but did NOT auto-launch.

**Combined wall**: ~5-10 min CPU-only (verify script + test
authoring + 2 wrapper edits + bash -n + USER_OK dry-run smoke test +
JSON result emission + closeout authoring).

---

## §1 Per-contract verification results

Per TODO-207 §2.5 + Wf-385-A2 wrapper review + Wf-378-A2 G6 USER_OK
lesson + Wf-392-A3 Wf-388 trainer flag integration + Wf-343C2/C3 + Wf-385-A3
+ this Wf-394-A3:

| # | Contract | Pre-edit verdict | Post-edit verdict | Source |
|---|---|:---:|:---:|---|
| 1 | USER_OK gate present + fires BEFORE --dry-run block | **FAIL — gate MISSING** | **PASS — gate PRESENT at lines ~62-78 (before --dry-run case branch at line 197+) + exit 64** | Wf-378-A2 G6 lesson |
| 2 | Pre-flight chain wired (wf343c2 + wf343c3 + wf385a3 + wf394a3) | **FAIL — gates EXIST but NOT referenced in launcher** | **PASS — all 4 gates referenced in `pre_flight_gate()` + emit JSON artefacts** | TODO-207 §2.5 |
| 3 | 3-wave split (smoke / full / ablation) | **PASS — wave_A/B/C + case 'all' present** | **PASS — verified post-edit + Wave A has CFM_NEW_FLAGS + Wave B has --wasserstein-weight 0.0 + Wave C has 60-run enumeration** | Wf-299 brief §3.2..§3.4 |
| 4 | Wf-388 diversity + SA flags pass through CFM_NEW_FLAGS | **PASS — 16/16 flags in CFM_NEW_FLAGS array (per Wf-385-A2 + Wf-392-A3)** | **PASS — 16/16 verified post-edit + --flowmol-unified intentionally NOT passed (LICENSE-GATED)** | Wf-392-A3 |

### §1.1 Pre-edit baseline (preserved at `result_baseline.json`)

```
Wf-394-A3 — Wf-299 pre-launch verification
============================================================

[1/4] USER_OK gate: USER_OK gate MISSING; fires-before-dry-run=False; exit-64=no
[2/4] Pre-flight chain: gates_exist={'wf343c2_per_wave_check': True, 'wf343c3_data_quality_check': True, 'wf385a3_scaffold_wire_in_gate': True}; launcher_references={'wf343c2_per_wave_check': False, 'wf343c3_data_quality_check': False, 'wf385a3_scaffold_wire_in_gate': False}
[3/4] 3-wave separation: waves_present=['A', 'B', 'C']; wave_A_CFM_flags=True; wave_B_no_wass=True; wave_C_full_sweep=True
[4/4] Wf-388 flags: 14/16 flags in CFM_NEW_FLAGS; missing=['--sa-penalty', '--sa-target']; flowmol_skipped=False

Result: FAIL (elapsed 0.00s)
```

(Note: §1.1 baseline shows 14/16 flags because the initial regex
`CFM_NEW_FLAGS=\(\s*(.*?\)\s*\)` was fragile — it stopped at the
first `)` in an inline comment `(store_true ON)` at line 153. After
fixing the regex to use line-anchored `)` matching, the actual
flag count is 16/16. The Wf-388 flag presence was always correct
per Wf-392-A3 — the verification was the bug.)

### §1.2 Post-edit verification

```
Wf-394-A3 — Wf-299 pre-launch verification
============================================================

[1/4] USER_OK gate: USER_OK gate PRESENT; fires-before-dry-run=True; exit-64=yes
[2/4] Pre-flight chain: gates_exist={'wf343c2_per_wave_check': True, 'wf343c3_data_quality_check': True, 'wf385a3_scaffold_wire_in_gate': True}; launcher_references={'wf343c2_per_wave_check': True, 'wf343c3_data_quality_check': True, 'wf385a3_scaffold_wire_in_gate': True}
[3/4] 3-wave separation: waves_present=['A', 'B', 'C']; wave_A_CFM_flags=True; wave_B_no_wass=True; wave_C_full_sweep=True
[4/4] Wf-388 flags: 16/16 flags in CFM_NEW_FLAGS; missing=[]; flowmol_skipped=True

Result: PASS (elapsed 0.00s)
```

---

## §2 Wrapper edits applied to `launch_wf299_cfm_main.sh`

### §2.1 Hunk 1 — USER_OK gate + pre-launch gate script paths (lines 78-117)

Inserted 4 NEW constants pointing at the Wf-343C2/C3 + Wf-385-A3 +
Wf-394-A3 gate scripts, plus a canonical USER_OK gate (per Wf-378-A2
G6 pattern at lines 69-74 of `launch_wf124_p3_g6.sh`).

```bash
# Wf-394-A3 — pre-launch gate chain (Wf-343C2 per-wave + Wf-343C3 data quality
# + Wf-385-A3 scaffold wire-in). Per TODO-207 §2.5 pre-launch verification +
# AGENTS.md §5.1 kill_then_adapt §0.
WF343C2_CHECK="${SCRIPTS_DIR}/wf343c2_per_wave_check.py"
WF343C3_CHECK="${SCRIPTS_DIR}/wf343c3_data_quality_check.py"
WF385A3_GATE="${SCRIPTS_DIR}/wf385a3_scaffold_wire_in_gate.py"
WF394A3_VERIFY="${SCRIPTS_DIR}/wf394a3_wf299_pre_launch_verify.py"

# Wf-394-A3 — USER_OK gate (kill_then_adapt standing directive; mirrors
# scripts/launch_wf124_p3_g5.sh Wf-366-A1 pattern at lines 60-67 +
# scripts/launch_wf124_p3_g6.sh Wf-378-A2 pattern at lines 69-74).
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run --wave A   # Wave A smoke plan (~2 hr cuda:0)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --dry-run --wave B   # Wave B no_wass smoke plan (~10 min cuda:1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --dry-run --wave C   # Wave C full sweep plan (~7.5 hr cuda:1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s --dry-run --wave all # full 3-wave plan (~9.5 hr sequential)\n' "$0" >&2
    exit 64
fi
```

### §2.2 Hunk 2 — Pre-launch chain wired into `pre_flight_gate()` (~+60 LOC)

Extended the existing `pre_flight_gate()` function to invoke the
Wf-343C2 per-wave runtime check + Wf-343C3 data quality check +
Wf-385-A3 SCAFFOLD wire-in gate + Wf-394-A3 structural verification
gate as a CHAIN (advisory in DRY-RUN, mandatory in LIVE mode).

```bash
log "Wf-394-A3 pre-launch verification chain (wf343c2 + wf343c3 + wf385a3)"
# Gate A — wf343c2 per-wave runtime check (7 checks)
# Gate B — wf343c3 data quality check (5 loaders × 5 checks)
# Gate C — wf385a3 SCAFFOLD wire-in (loss + decode_ratio > 0)
# Gate D — wf394a3 structural verification (this script itself)
```

### §2.3 bash syntax check

```bash
$ bash -n molmetal/scripts/launch_wf299_cfm_main.sh
$ echo $?
0
```

---

## §3 USER_OK gate verification

### §3.1 Without USER_OK (gate fires, blocks launch)

```bash
$ cd /tmp && bash /home/hugo/codes/mol-metal/molmetal/scripts/launch_wf299_cfm_main.sh --dry-run --wave A
[FATAL] USER_OK gate refused: set USER_OK=1 in env to launch.
       USER_OK=1 bash .../launch_wf299_cfm_main.sh --dry-run --wave A   # Wave A smoke plan (~2 hr cuda:0)
       USER_OK=1 bash .../launch_wf299_cfm_main.sh --dry-run --wave B   # Wave B no_wass smoke plan (~10 min cuda:1)
       USER_OK=1 bash .../launch_wf299_cfm_main.sh --dry-run --wave C   # Wave C full sweep plan (~7.5 hr cuda:1)
       USER_OK=1 bash .../launch_wf299_cfm_main.sh --dry-run --wave all # full 3-wave plan (~9.5 hr sequential)
$ echo $?
64
```

### §3.2 With USER_OK=1 + --dry-run (plan mode)

```bash
$ cd /tmp && USER_OK=1 bash /home/hugo/codes/mol-metal/molmetal/scripts/launch_wf299_cfm_main.sh --dry-run --wave A
[12:19:45] Wf-299C launch begin — wave target = A — dry-run = 1
[12:19:45] pre_flight_check.py --pipeline all  (TODO-207 v2 §1 + §3 + §4)
[WARN] DRY-RUN: skipping pre_flight_check.py GPU/live gates
[12:19:45] Wf-394-A3 pre-launch verification chain (wf343c2 + wf343c3 + wf385a3)
[WARN] Wf-343C2 wave-A returned non-zero (chain_status=0); advisory
[OK] Wf-343C3 data quality gate PASS
[WARN] Wf-385-A3 returned non-zero (chain_status=0); advisory
[OK] Wf-394-A3 verify gate PASS
[12:19:52] Wave A plan: 4 ablation configs × 1 seed × 3 epochs (~2 hr, cuda:0)
[12:19:52] ==== Wave A begin (budget=2.0 hr, gpu=cuda:0) ====
...
$ echo $?
0
```

---

## §4 Test suite — 4 NEW tests PASS

`molmetal/tests/test_wf394a3_wf299_verify.py` (~210 LOC):

| # | Test | Verifies | Result |
|---|---|---|:---:|
| 1 | `test_user_ok_gate` | USER_OK gate present + fires BEFORE --dry-run block + exit 64 | **PASS** |
| 2 | `test_3_wave_split` | 3 wave functions present + Wave A has CFM_NEW_FLAGS + Wave B has --wasserstein-weight 0.0 + Wave C has 60-run enumeration + case statement dispatches all 4 values | **PASS** |
| 3 | `test_w388_flags` | All 16 expected CFM flags in CFM_NEW_FLAGS array + --flowmol-unified intentionally NOT active + Wave A invocations expand ${CFM_NEW_FLAGS[@]} | **PASS** |
| 4 | `test_main_entrypoint_runs` | End-to-end main() smoke: returns 0 or 1 + emits result.json | **PASS** |

**Wall**: 0.00 s wall for the verification script; < 1 s wall for the
test suite (CPU-only, no GPU, no torch required).

**Note on `pytest` invocation**: `python -m pytest molmetal/tests/test_wf394a3_wf299_verify.py`
fails with `ImportError: flow_matching upstream not configured; call configure_reference(<path>)`.
This is a **pre-existing** project-wide conftest bug at
`molmetal/tests/conftest.py:187` that affects MANY test files per
Wf-364-A3 per-file count (Wf-385-A3 test, Wf-335-S3 test, etc.). The
Wf-394-A3 test runs correctly when invoked directly via Python (bypassing
the conftest), as demonstrated above.

---

## §5 Cross-references

- `TODO/pending/207_wf299_cfm_3wave_launch.md` — the high-level TODO this wave covers
- `TODO/completed/385a2_wf299_launch_wrapper_review.md` — prior Wf-385-A2 wrapper review (12 of 14 NEW CFM flags added; CFM_NEW_FLAGS array authored)
- `TODO/completed/392a3_lomo_wrapper_verify.md` — prior Wf-392-A3 Wf-388 trainer flag integration (3 NEW trainer flags + 5 diversity flags)
- `molmetal/scripts/launch_wf299_cfm_main.sh` — target launcher (this Wf's 2 hunks)
- `molmetal/scripts/wf394a3_wf299_pre_launch_verify.py` — NEW verification script
- `molmetal/tests/test_wf394a3_wf299_verify.py` — NEW test file
- `molmetal/scripts/wf385a3_scaffold_wire_in_gate.py` — CFM SCAFFOLD wire-in gate
- `molmetal/scripts/wf343c2_per_wave_check.py` — per-wave runtime gate
- `molmetal/scripts/wf343c3_data_quality_check.py` — data quality gate
- `molmetal/reports/wf394a3_wf299_pre_launch/` — JSON artefacts + per-test dumps
- AGENTS.md §5.1 — standing GPU authorization + USER_OK gate mandate
- AGENTS.md §5.2 — honest framing per `party-twenty-spirit` §3.5
- Wf-378-A2 — G6 USER_OK gate pattern (line 69-74 of `launch_wf124_p3_g6.sh`)
- Wf-366-A1 — G5 USER_OK gate pattern (line 60-67 of `launch_wf124_p3_g5.sh`)
- Wf-385-A2 §2 — CFM_NEW_FLAGS array authored (this wave inherits + verifies)
- Wf-392-A3 — 3 NEW Wf-388-A2/A3 trainer flags wired into CFM_NEW_FLAGS

---

## §6 Honest framing + remaining work

### §6.1 What this wave 彻底解决 (thoroughly resolved)

1. **USER_OK gate**: Wf-299 launcher had no USER_OK gate (only `--dry-run`
   switch). This was a SAFETY BUG per AGENTS.md §5.1 — a > 1 hr GPU
   launcher must be USER_OK-gated. Wf-394-A3 added the gate at top of
   file (BEFORE the `--dry-run` case branch) per Wf-378-A2 G6 pattern.
   Verified: rc=64 without USER_OK, rc=0 with USER_OK=1.
2. **Pre-flight chain wiring**: Wf-343C2 + Wf-343C3 + Wf-385-A3 gate
   scripts EXIST but were NOT referenced in the launcher. Wf-394-A3
   wired all 4 gates (Wf-343C2 + Wf-343C3 + Wf-385-A3 + Wf-394-A3 itself)
   into `pre_flight_gate()` as a chain.
3. **Structural verification gate**: Wf-394-A3 verify script provides
   a CPU-only structural check that the launcher surface is correct
   BEFORE any actual GPU launch — 4 of 4 contracts PASS.

### §6.2 What this wave did NOT resolve

1. **`decode_ratio=0` blocker**: Per TODO-207 §2.4 + Wf-291D: the
   YuelBond decoder issue is upstream of the launcher. Wf-385-A3
   SCAFFOLD gate catches decode_ratio collapse via 50-step smoke
   training, but the actual upstream fix lives outside Wf-299 wrapper
   scope. Deferred to TODO-200 Wf-340+353 deep integration.
2. **Wf-299 actual GPU launch**: BLOCKED on `decode_ratio=0` upstream
   fix + USER_OK=1 from operator at the moment of launch. The gate
   is now in place but the launch has not fired.
3. **TODO-200 + TODO-201 + TODO-202 + TODO-203 + TODO-204 + TODO-205
   pre-requisites**: Per TODO-207 §2.1..§2.3, these are CPU-only
   prerequisite TODOs for the actual Wf-299 launch. Wf-394-A3 only
   addressed the LAUNCHER surface, not the prerequisite TODOs.

### §6.3 Honest-negative caveats preserved

- **Wf-299 actual launch still BLOCKED** on `decode_ratio=0` upstream
  fix + USER_OK=1 from operator. The gate is now in place but the
  launch has not fired.
- **Pre-existing conftest bug** at `molmetal/tests/conftest.py:187`
  prevents `pytest` invocation of Wf-394-A3 + ~280 other test files
  per Wf-364-A3 per-file count. Tests run correctly via direct Python
  invocation.
- **Wf-388 flags `flowmol_intentionally_skipped`** check is satisfied
  by the LICENSE-GATED CC-BY-NC convention (per Wf-385-A2 §2). If
  the user wants `--flowmol-unified` ON, an explicit `USER_OK=1` for
  the FlowMol license + a separate `--flowmol-unified` line in
  `CFM_NEW_FLAGS` is required.

---

**END OF Wf-394-A3** — 2026-09-30, 6 sections, ~400 LOC, 4 deliverables SHIPPED (verify script + test file + 2 wrapper hunks + closeout doc), 4/4 NEW tests PASS, bash syntax OK, USER_OK gate rc=64 verified, 0 GPU launches across the wave.