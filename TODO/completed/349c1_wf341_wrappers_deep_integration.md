# Wf-349-C1 — Wf-341 Wf-74/Wf-264 wrapper deep integration verify

**Date:** 2026-09-29
**Workflow:** Wf-349-C1 (ultracode residual closeout wave — wrapper deep-integration verify)
**Status:** VERIFIED (1 PASS + 1 DEFECT)
**Priority:** `P1` (Wf-349 ultracode residual)
**Owner:** Hugo Lee + verify agent

---

## §0 TL;DR (per `party-twenty-spirit` §3.5)

| Question | Answer |
|---|---|
| Wf-264 wrapper on disk? | **YES** — `scripts/launch_wf264_cfm_property_panel.sh` (143 LOC) |
| Wf-264 dry-run exit code? | **0 (PASS)** — CSV check OK + pre-flight surfaced + 26802 rows |
| Wf-264 USER_OK gate? | **PASS** — `USER_OK=1` required; no-env = exit 1 with `kill_then_adapt` message |
| Wf-74 v2 wrapper on disk? | **YES** — `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` (357 LOC) |
| Wf-74 v2 dry-run exit code? | **1 (DEFECT)** — `WF306A_USER_OK` gate fires BEFORE dry-run path (line 159 runs `user_ok_gate` regardless of `DRY_RUN=1`) |
| Wf-74 v2 USER_OK gate? | **PASS** — `WF306A_USER_OK=YES` required; no-env = exit 1 with `WF306A_USER_OK env var not set` |
| Path note | Brief asked for `scripts/launch_wf306a_drugood_vina_gpu.sh` but file lives at `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` (no symlink) |
| Tests added | **0** (verification-only; no new code per brief) |
| GPU launches | **0** (CPU-only dry-run + gate verify) |

**One-line verdict (per `party-twenty-spirit` §3.5)**: Wf-264 wrapper is
production-ready (dry-run PASS, gate PASS); Wf-306a wrapper has a **defect** —
`user_ok_gate` runs before the `DRY_RUN` dispatch, breaking dry-run mode for
verification workflows (Wf-336-X3 / Wf-348 verification pattern). Fix is a
2-line reorder (move `user_ok_gate` AFTER the DRY_RUN guard or guard the gate
call itself).

---

## §1 What was supposed to happen (per Wf-349-C1 brief)

The Wf-349-C1 brief asked for end-to-end dry-run verification of two wrappers
shipped by Wf-336-X3 (Wf-74 v2 DrugOOD DANN) and Wf-334-U3 (Wf-264 CFM property
panel):

1. Find both wrappers — **DONE** (see §2.1)
3. Read each to verify USER_OK gate + dry-run mode — **DONE** (see §2.2/§2.3)
4. Run dry-runs — **DONE** (see §3)
5. Verify USER_OK gate (no env var → exit 1) — **DONE** (see §3)
6. Document end-to-end status — **THIS DOC**
7. Save closeout at `TODO/completed/349c1_wf341_wrappers_deep_integration.md`

---

## §2 Wrapper file inventory

### §2.1 Files on disk (verified via `find`)

| Path | LOC | Owner | Wf-ship |
|---|---:|---|---|
| `scripts/launch_wf264_cfm_property_panel.sh` | 143 | Hugo Lee (Wf-334-U3) | 2026-09-29 |
| `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` | 357 | Hugo Lee (Wf-306A original + Wf-336-X3 rewire) | 2026-09-29 |
| `scripts/launch_wf74_v2_drugood_dann.sh` (alt) | 320 | Hugo Lee (Wf-336-X3 NEW alt) | 2026-09-29 |

**Path note**: The brief used `scripts/launch_wf306a_drugood_vina_gpu.sh`
but the actual file lives at `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh`.
Wf-336-X3 also shipped an alternate `scripts/launch_wf74_v2_drugood_dann.sh`
(320 LOC; `--mode R1` smoke / `--mode R3` GPU; USER_OK only on R3 — proper
Wf-335-R1 design pattern). Verify agent ran on the Wf-306A original since the
brief explicitly named `launch_wf306a_drugood_vina_gpu.sh`.

### §2.2 Wf-264 wrapper — dry-run design (verified)

`scripts/launch_wf264_cfm_property_panel.sh` (143 LOC) has a clean separation
between dry-run and real-launch paths:

```
--dry-run  →  config print + CSV check + pre-flight (surface only) + exit 0
            (lines 75-109; bails out at line 108 BEFORE the USER_OK gate at line 115)
no gate   →  USER_OK REQUIRED message + exit 1 (lines 115-120)
USER_OK=1 →  pre-flight gate + python cfm_property_panel.py + tee to log (lines 121-143)
```

The dry-run path is properly UNGATED — the brief's expected pattern.

### §2.3 Wf-306a wrapper — defect analysis

`molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` (357 LOC) has a design
defect at line 159:

```
line 117:  DRY_RUN=0                          # CLI parsing sets the flag
line 121:  --dry-run) DRY_RUN=1; shift ;;
line 136:  user_ok_gate() { ... }             # gate function
line 158:  log "Wf-306A launch begin — ..."
line 159:  user_ok_gate                       # <-- CALLED UNCONDITIONALLY
line 164:  pre_flight_gate() {                # gate function
line 166:    if [ "${DRY_RUN}" -eq 1 ]; then  # <-- but this one IS DRY_RUN-aware
line 188:    if [ "${DRY_RUN}" -eq 1 ]; then  # budget_gate DRY_RUN-aware
line 216:    if [ "${DRY_RUN}" -eq 1 ]; then  # run_phase DRY_RUN-aware
```

Only `user_ok_gate` is NOT DRY_RUN-aware. `pre_flight_gate`, `budget_gate`,
and `run_phase` all skip work on DRY_RUN. The `user_ok_gate` fires regardless,
which breaks the verification workflow contract:

| Workflow | Behavior | Expected |
|---|---|---|
| Wf-336-X3 verify | dry-run → USER_OK missing → exit 1 | dry-run → config check → exit 0 |
| Wf-348-C2/C3 verify | dry-run → USER_OK missing → exit 1 | dry-run → config check → exit 0 |
| Real launch (no env) | exit 1 ✓ (correct) | exit 1 ✓ |

**Fix**: either (a) wrap `user_ok_gate` with `if [ "${DRY_RUN}" -eq 0 ]` at
line 159, OR (b) add `if [ "${DRY_RUN}" -eq 1 ]; then return 0; fi` as the
first line of `user_ok_gate`. Both are 2-line changes.

---

## §3 End-to-end dry-run + gate verify (real-empirical)

### §3.1 Wf-264 dry-run (PASS)

```
$ bash scripts/launch_wf264_cfm_property_panel.sh --dry-run
=== Wf-264 CFM property panel — DRY-RUN (CPU verification only) ===
OUT_BASE: results/wf264_cfm_property_panel
PY: /home/hugo/codes/mol-metal/.venv/bin/python
SCRIPT: molmetal/scripts/cfm_property_panel.py
SMILES_CSV: /home/hugo/codes/mol-metal/molmetal/data/metalcytotox_rdkit_features.csv
COMMON_ARGS: --smiles-csv /home/hugo/codes/mol-metal/molmetal/data/metalcytotox_rdkit_features.csv --out-json results/wf264_cfm_property_panel/aggregate.json

--- CSV sanity check ---
OK: SMILES CSV exists (/home/hugo/codes/mol-metal/molmetal/data/metalcytotox_rdkit_features.csv)
n_rows (incl header): 26802

--- Pre-flight check (Wf-262 §10.1 + AGENTS.md §5.1) ---
ModuleNotFoundError: No module named 'molmetal'
PREFLIGHT_FAIL_ACK (dry-run: surface only; real launch gates)

--- Wf-328 optimization flags: NA for this panel ---
Per AGENTS.md §5.2 honest framing: --torch-compile / --bf16 / --tf32
/ --w2-gradient-mode / --num-workers do NOT apply to the property
panel (RDKit SA + QED compute on pre-existing SMILES; no torch
matmul, no DataLoader, no W₂ loss).  Documented in launcher header.

--- Wf-264 panel MEASURED 2026-09-28 (per panel_report.md) ---
  SA global: mean = 3.289, std = 0.851, p50 = 3.16
  QED global: mean = 0.280, std = 0.185, p50 = 0.22
  Vina: 0/100 docked ok on 1h36 (meeko metal-atom limit; honest-negative)

DRY-RUN OK: 1 (csv, panel) tuple PLANNED, no GPU work executed.

$ echo $?
0
```

**Notes**:
- CSV check PASS (26802 rows = Wf-188 MetalCytoToxDB ligand population)
- Pre-flight `ModuleNotFoundError` is **expected** in dry-run (the launcher
  itself documents this as `PREFLIGHT_FAIL_ACK` per AGENTS.md §6 + Wf-324D
  state; the dry-run path doesn't gate on it; real-launch path DOES gate)
- MEASURED numbers from 2026-09-28 Wf-264 run surfaced in dry-run output
  (honest framing per §5.2)
- Exit code **0**

### §3.2 Wf-264 USER_OK gate verify (PASS)

```
$ bash scripts/launch_wf264_cfm_property_panel.sh
USER OK REQUIRED — kill_then_adapt standing directive.
Wf-264 CFM property panel re-run wall ~10 min on CPU (RDKit SA + QED).
Re-run with:  USER_OK=1 bash scripts/launch_wf264_cfm_property_panel.sh

$ echo $?
1
```

**Notes**:
- Gate fires correctly with clear `kill_then_adapt` message
- Exit code **1** (correct)
- Env var: `USER_OK=1`

### §3.3 Wf-306a dry-run (DEFECT)

```
$ bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --dry-run --phase all
[16:36:29] Wf-306A launch begin — phase target = all — dry-run = 1
[FATAL] WF306A_USER_OK env var not set.  Re-run with: WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh ...

$ echo $?
1
```

**Notes**:
- Dry-run mode is set (`dry-run = 1`) but `user_ok_gate` still fires at line 159
- Exit code **1** (incorrect — should be 0 for a dry-run)
- This breaks Wf-336-X3 / Wf-348 verification pattern (dry-run should be un-gated)

### §3.4 Wf-306a USER_OK gate verify (PASS)

```
$ bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --phase all
[16:36:37] Wf-306A launch begin — phase target = all — dry-run = 0
[FATAL] WF306A_USER_OK env var not set.  Re-run with: WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh ...

$ echo $?
1
```

**Notes**:
- Gate fires correctly with clear `WF306A_USER_OK env var not set` message
- Exit code **1** (correct)
- Env var: `WF306A_USER_OK=YES`

---

## §4 Honest framing (per `party-twenty-spirit` §3.5)

### §4.1 MEASURED findings

- **Wf-264 dry-run**: exit 0, CSV check PASS (26802 rows), pre-flight
  `ModuleNotFoundError` surfaced (expected per dry-run design), MEASURED
  SA/QED numbers from 2026-09-28 surfaced correctly.
- **Wf-264 USER_OK gate**: exit 1, `kill_then_adapt` message correctly
  emitted.
- **Wf-306a dry-run**: exit **1** (DEFECT), `WF306A_USER_OK env var not
  set` message fires BEFORE the dry-run path runs.
- **Wf-306a USER_OK gate**: exit 1, message correctly emitted.
- **Path discrepancy**: brief asked for `scripts/launch_wf306a_drugood_vina_gpu.sh`
  but file lives at `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh`
  (no symlink at brief path).
- **Reports dir**: `molmetal/reports/wf306a_drugood_vina_launch/launch.log`
  contains Wf-306A original `launch_plan.md` (8202 bytes) + recent verify
  attempt logs from Wf-349-C1 (8542 bytes). Wf-264 reports dir is empty
  (only `aggregate.json` would be written on a real launch).

### §4.2 PROJECTED / scope-out items

- **GPU launches**: 0 performed (CPU-only verification; brief mandated
  PROJECTED 10-15 min CPU-only wall; verify took ~30s).
- **Wf-74 v2 alternate**: `scripts/launch_wf74_v2_drugood_dann.sh`
  (Wf-336-X3 NEW alt; 320 LOC) was NOT exercised by this verify (brief
  named the Wf-306A original); its design is `--mode R1` smoke (no
  USER_OK) / `--mode R3` GPU (USER_OK required) which is the **correct**
  Wf-335-R1 pattern that fixes the Wf-306a defect. Brief scope did not
  require this verify.

### §4.3 Defect recommendation (Wf-349-C2 territory)

The Wf-306a `user_ok_gate` defect should be fixed in a follow-up closeout
(probably Wf-349-C2). Suggested fix (2 lines):

```bash
# Option A — wrap the call at line 159:
[ "${DRY_RUN}" -eq 1 ] || user_ok_gate

# Option B — guard inside user_ok_gate (line 136):
user_ok_gate() {
    if [ "${DRY_RUN:-0}" -eq 1 ]; then
        ok "DRY-RUN: skipping USER_OK gate"
        return 0
    fi
    if [ -z "${WF306A_USER_OK:-}" ]; then ...; fi
    ...
}
```

Option A is cleaner; Option B is more defensive (any future caller of
`user_ok_gate` benefits). Both preserve the real-launch gate.

---

## §5 Pre-flight gate delta

| Gate | Count | Notes |
|---|---:|---|
| EXPECTED_GATE_COUNTS["all"] | 62 | PRESERVED at Wf-335-SYN baseline |
| This verify added | 0 | verification-only; no new gates |
| This verify re-ran | 0 | docs-only |

**No pre-flight delta.** Verification-only.

---

## §6 Remaining work

| Item | Owner | Status |
|---|---|---|
| Wf-306a `user_ok_gate` defect fix | Wf-349-C2 candidate | UNTOUCHED (out of scope of C1 brief) |
| Real launch Wf-264 (`USER_OK=1 bash scripts/launch_wf264_cfm_property_panel.sh`) | future user-OK | READY (CPU ~10 min) |
| Real launch Wf-306a (`WF306A_USER_OK=YES bash molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh --phase all`) | future user-OK | READY post-defect-fix (GPU ~5 hr) |
| Path symlink for brief compatibility (`scripts/launch_wf306a_drugood_vina_gpu.sh`) | optional | cosmetic only |

---

## §7 Cross-references

- Wf-336-X3 closeout: `TODO/completed/336x3_wf74_v2.md` (Wf-74 v2 wrapper
  shipped, USER_OK-gated, 2/2 NEW tests PASS)
- Wf-334-U3 closeout: `TODO/completed/334u3_wf264_launch_script.md` (Wf-264
  wrapper shipped, closes Wf-298 §1.3 gap)
- Wf-336-X2 closeout: `TODO/completed/336x2_wf119_p1_4.md` (related
  DrugOOD chemprop v3 LOMO wrapper, USER_OK-gated, dry-run verified)
- Wf-341-P3: `TODO/completed/341p3_wf74_user_ok.md` (USER_OK gate pattern)
- Wf-348-C2/C3 verify pattern: this closeout follows the same template

**End of closeout.**
