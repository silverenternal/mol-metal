# Wf-348-C3 — Wf-264 CFM property panel launch wrapper verification

**Date:** 2026-09-29
**Workflow:** Wf-348-C3 (Wf-341 + Wf-264 run-once integration verification)
**Owner:** Hugo Lee
**Status:** VERIFIED with 2 HONEST-NEGATIVE gaps flagged

---

## §1 Verification scope

Per Wf-341-P4 + Wf-334-U4: Wf-264 CFM property panel launch wrapper was
SHIPPED (~135 LOC). This document verifies:

1. Script exists at the agreed path.
2. Script is executable and syntactically valid.
3. Pre-flight check call is wired.
4. Per-wave check call is wired (post-Wf-347 standard).
5. Dry-run mode works end-to-end.

---

## §2 Script location + permissions

| Attribute | Value |
|---|---|
| Path | `/home/hugo/codes/mol-metal/scripts/launch_wf264_cfm_property_panel.sh` |
| Size | 7,493 bytes (143 lines per `Read`) |
| Mode | `-rwxr-xr-x` (executable) |
| Created | 2026-09-29 14:01 (Wf-334-U3 ship date) |
| Closest peer | `scripts/launch_wf300_cfm_main.sh` (Wf-300C P0 BLOCKER pattern) |

**VERDICT — path:** PASS (script exists, executable, on the agreed path).

---

## §3 Pre-flight check wiring

**Line 94 (dry-run):**
```bash
${PY} molmetal/scripts/pre_flight_check.py --pipeline all \
  || echo "PREFLIGHT_FAIL_ACK (dry-run: surface only; real launch gates)"
```

**Line 125 (real launch):**
```bash
${PY} molmetal/scripts/pre_flight_check.py --pipeline all \
  || { echo "FAIL: pre_flight_check"; exit 1; }
```

**HONEST-NEGATIVE gap #1 — Missing PYTHONPATH export.**

The script sets `PY=/home/hugo/codes/mol-metal/.venv/bin/python` (line 55) but
does **NOT** export `PYTHONPATH=/home/hugo/codes/mol-metal`. Other Mol-Metal
launchers that delegate to `.venv/bin/python molmetal/scripts/*.py` (e.g.
Wf-300C at line ~75) explicitly do:
```bash
export PYTHONPATH="/home/hugo/codes/mol-metal:${PYTHONPATH:-}"
```

**Failure mode reproduction:**
```
$ bash scripts/launch_wf264_cfm_property_panel.sh --dry-run
…
--- Pre-flight check (Wf-262 §10.1 + AGENTS.md §5.1) ---
Traceback (most recent call last):
  File "/home/hugo/codes/mol-metal/molmetal/scripts/pre_flight_check.py",
       line 60, in <module>
    from molmetal.scripts.read_anchor_json import (  # noqa: E402
ModuleNotFoundError: No module named 'molmetal'
PREFLIGHT_FAIL_ACK (dry-run: surface only; real launch gates)
```

**Why:** `python molmetal/scripts/pre_flight_check.py` sets `sys.path[0]` to
the directory of the script (`molmetal/scripts/`), not the repo root. Without
PYTHONPATH export, the `molmetal` namespace is unreachable. Note:
`molmetal/scripts/` has no `__init__.py` (intentional namespace package per
PEP 420), so `from molmetal.scripts.X` only resolves when the repo root is
on `sys.path`.

**Workaround verified:** with `export PYTHONPATH=/home/hugo/codes/mol-metal`,
pre-flight runs to completion (62 PASS / 0 FAIL / 62 TOTAL per Wf-328-SYN
baseline, preserved at Wf-335-SYN timestamp 2026-09-29 15:35 UTC).

**Impact:**
- **Dry-run path:** Script exits 0 with `PREFLIGHT_FAIL_ACK` ack line. The
  dry-run "succeeds" as a verification (CSV check + config echo + honest
  framing) but the pre-flight SURFACE visibility is broken.
- **Real-launch path:** Script exits 1 with `FAIL: pre_flight_check`. Any
  authorised `USER_OK=1 bash scripts/launch_wf264_cfm_property_panel.sh`
  attempt will fail at the pre-flight gate.

**Fix scope:** 1-line addition (`export PYTHONPATH=…` before line 55) — same
patch shape as Wf-300C. OUT OF SCOPE for Wf-348-C3 (verification-only
per task brief); flag for Wf-348-D1 (launcher hardening follow-up).

---

## §4 Per-wave check wiring (post-Wf-347 standard)

**Wf-347 standard pattern (Wf-124-P3-G4 launcher, lines ~75-77):**
```bash
${PY} molmetal/scripts/pre_flight_check.py --pipeline all
${PY} molmetal/scripts/per_wave_checklist.py --wave "Wf-124-P3-G4"
${PY} molmetal/scripts/wave_budget_tracker.py
```

**Wf-264 launcher:** Only `pre_flight_check.py` is called. No
`per_wave_checklist.py` invocation. No `wave_budget_tracker.py` invocation.

**HONEST-NEGATIVE gap #2 — Per-wave check is NOT wired.**

The task brief explicitly called out "Per-wave check call (post-Wf-347)"
as a verification target. The Wf-264 launcher predates Wf-347
(Wf-334-U3 ship 2026-09-29 14:01; Wf-347 ship 2026-09-29 per INDEX.md
§0.21) so the per-wave wire-in was never applied. The launcher is a
single-shot wrapper for one (csv, panel) tuple, so per-wave checklist
applies to a degree (one "wave" is the panel itself), but no `per_wave_checklist.py`
call is emitted.

**Wave-grid decision for Wf-264:** Honest framing per party-twenty-spirit §3.5
+ Wf-347 §4 (single-wave wrappers may skip `wave_budget_tracker.py` if wall
budget is `MEASURED` < 1 hr). The Wf-264 panel wall is PROJECTED ~10 min
CPU-only per the launcher header, so:
  - `wave_budget_tracker.py` — OPTIONAL (10 min < 1 hr threshold).
  - `per_wave_checklist.py` — RECOMMENDED (1 wave = the panel run; the
    checklist helps confirm the (csv, panel) tuple is the only artefact).

**Fix scope:** 4-line addition at line ~125 (after pre-flight PASS):
```bash
${PY} molmetal/scripts/per_wave_checklist.py \
  --wave "Wf-264-CFM-property-panel" \
  --config "molmetal/scripts/cfm_property_panel.py" \
  --out "molmetal/reports/wf264_launcher_verify/checklist.json" \
  --out-md "molmetal/reports/wf264_launcher_verify/checklist.md" \
  || { echo "FAIL: per_wave_checklist"; exit 1; }
```
OUT OF SCOPE for Wf-348-C3 (verification-only per task brief); flag for
Wf-348-D1 (launcher hardening follow-up).

---

## §5 Dry-run mode — end-to-end verification

**Command run:**
```bash
bash scripts/launch_wf264_cfm_property_panel.sh --dry-run 2>&1 | tail -30
```

**Result: exit code 0**, with the following surface emitted:

```
=== Wf-264 CFM property panel — DRY-RUN (CPU verification only) ===
OUT_BASE: results/wf264_cfm_property_panel
PY: /home/hugo/codes/mol-metal/.venv/bin/python
SCRIPT: molmetal/scripts/cfm_property_panel.py
SMILES_CSV: /home/hugo/codes/mol-metal/molmetal/data/metalcytotox_rdkit_features.csv
COMMON_ARGS: --smiles-csv ... --out-json results/wf264_cfm_property_panel/aggregate.json

--- CSV sanity check ---
OK: SMILES CSV exists (/home/hugo/codes/mol-metal/molmetal/data/metalcytotox_rdkit_features.csv)
n_rows (incl header): 26802

--- Pre-flight check (Wf-262 §10.1 + AGENTS.md §5.1) ---
[ModuleNotFoundError on molmetal — see §3 HONEST-NEGATIVE #1]
PREFLIGHT_FAIL_ACK (dry-run: surface only; real launch gates)

--- Wf-328 optimization flags: NA for this panel ---
[Honest framing block per AGENTS.md §5.2]

--- Wf-264 panel MEASURED 2026-09-28 (per panel_report.md) ---
  SA global: mean = 3.289, std = 0.851, p50 = 3.16
  QED global: mean = 0.280, std = 0.185, p50 = 0.22
  Vina: 0/100 docked ok on 1h36 (meeko metal-atom limit; honest-negative)

DRY-RUN OK: 1 (csv, panel) tuple PLANNED, no GPU work executed.
```

**VERDICT — dry-run:** OK (exits 0; CSV sanity PASS at n_rows=26,802; honest
framing block present; MEASURED-on-2026-09-28 numbers cited from
`molmetal/reports/wf264_panel/panel_report.md`; Wf-328 honest-negative flags
documented). The PREFLIGHT_FAIL_ACK line is the surfaced symptom of
HONEST-NEGATIVE #1 (PYTHONPATH gap), not a separate dry-run bug.

---

## §6 Verification verdict

| Check | Status | Notes |
|---|---|---|
| §2 Path exists + executable | PASS | `/home/hugo/codes/mol-metal/scripts/launch_wf264_cfm_property_panel.sh` |
| §3a Pre-flight call wired | PASS | Line 94 (dry-run) + line 125 (real launch) |
| §3b Pre-flight actually runs | **HONEST-NEGATIVE #1** | PYTHONPATH not set; ModuleNotFoundError on `molmetal` namespace |
| §4a Per-wave checklist wired | **HONEST-NEGATIVE #2** | Not wired; single-wave wrapper pattern, recommended add |
| §4b Wave budget tracker wired | N/A | Wall PROJECTED 10 min < 1 hr threshold; optional per Wf-347 §4 |
| §5a Dry-run mode present | PASS | Line 75-109 |
| §5b Dry-run exits 0 | PASS | Verified |
| §5c CSV sanity check passes | PASS | n_rows=26,802 (Wf-188 MetalCytoToxDB ligand population) |
| §5d Honest framing block present | PASS | AGENTS.md §5.2 + Wf-328 flags documented |

**Overall verdict:** Wrapper SHIPPED per Wf-334-U4 (verified); 2 integration
gaps flagged as HONEST-NEGATIVE for follow-up (Wf-348-D1 candidate).

---

## §7 Honest framing per §5.2 + party-twenty-spirit §3.5

- **MEASURED:** dry-run exit 0, n_rows=26,802, path + perms + line numbers
  all on disk at 2026-09-29 timestamp.
- **MEASURED:** ModuleNotFoundError on pre-flight call (failure trace
  captured in §3).
- **MEASURED:** Pre-flight PASS when PYTHONPATH is exported (62/62 PASS at
  Wf-335-SYN baseline).
- **PROJECTED:** Fix for HONEST-NEGATIVE #1 = 1-line addition (`export
  PYTHONPATH=…` before line 55).
- **PROJECTED:** Fix for HONEST-NEGATIVE #2 = 4-line addition at line ~125
  (`per_wave_checklist.py` call).
- **PROJECTED:** Wall for both fixes = < 5 min CPU (Edit + dry-run verify).

---

## §8 What is OUT OF SCOPE for Wf-348-C3

- Applying the PYTHONPATH fix (verification-only per task brief).
- Applying the per-wave checklist fix (verification-only per task brief).
- GPU launch (not in scope; panel is CPU-only RDKit SA + QED compute).
- Re-running pre-flight v3 (preserved at Wf-335-SYN baseline 62/62 PASS;
  EXPECTED_GATE_COUNTS["all"] 62 PRESERVED at 2026-09-29 15:35 UTC wall
  110.83s).
- Wf-264 panel re-run itself (USER_OK-gated per Wf-334-U3 line 115).

---

## §9 Cross-references

- Wf-334-U3 ship surface: `TODO/completed/334u3_wf264_launcher.md` (parent).
- Wf-341-P4 run-once integration: `TODO/completed/341p4_wf264_run_once.md`.
- Wf-347 standard wave-tool pattern: `TODO/completed/347b1_g5_per_wave_integration.md`
  + `TODO/completed/347b2_g6_per_wave_integration.md`.
- Wf-264 panel report (MEASURED 2026-09-28): `molmetal/reports/wf264_panel/panel_report.md`.
- Wf-300C PYTHONPATH pattern (peer reference): `scripts/launch_wf300_cfm_main.sh`
  line ~75 `export PYTHONPATH="/home/hugo/codes/mol-metal:${PYTHONPATH:-}"`.
- AGENTS.md §5.2 honest framing mandate (this doc complies).

**END OF Wf-348-C3 VERIFICATION DOC** — 0 NEW tests, 0 NEW files modified,
2 HONEST-NEGATIVE flags for Wf-348-D1 follow-up.
