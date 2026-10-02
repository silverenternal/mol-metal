# Wf-262 Audit — Honest Gap / Follow-up Inventory in v2 Pre-Flight Checklist

**Date**: 2026-09-28
**Source doc**: `TODO/completed/207_unified_pre_flight_checklist_v2.md` (531 lines)
**Method**: grep markers `honest gap`, `TODO-208`, `follow-up`, `TBD`, `does NOT yet exist`, `implementation gap`, `abort_recovery.py`, `per_wave_checklist`, `wave_budget_tracker`
**Scope**: READ ONLY — audit only, no implementation launched
**Verifier**: Wf-262 subagent

---

## §1 Summary (FLAT acceptance gates)

| Gate | Value | Evidence |
|---|---|---|
| **G1_n_honest_gaps_found** | **6** | Marker occurrences in v2 doc (3 explicit "Honest gap" + 2 "TBD" + 1 "implementation gap acknowledged") |
| **G2_n_still_missing** | **2** | Both reference `abort_recovery.py` (lines 262 + 404) |
| **G3_n_already_filled** | **4** | `pre_flight_check.py` × 2 + YAML schema validator × 2 |
| **G4_audit_path** | `molmetal/reports/wf262_audit/gap_list.md` | this file |

---

## §2 Honest-framed gap markers (6 occurrences, 3 unique items)

| # | Doc line | Marker | Item | Verdict | Evidence |
|---:|---:|---|---|---|---|
| G-1 | L103 | "Honest gap (§3.5)" | YAML schema validator (`pre_flight_check.py --gate 1.3`) | **ALREADY_FILLED** | `molmetal/scripts/pre_flight_check.py:401` defines `gate_yaml_schema()`; PyYAML parseability check + 7 active YAMLs verified |
| G-2 | L262 | "Honest gap (§3.5)" | `molmetal/scripts/abort_recovery.py` | **STILL_MISSING** | No file at `molmetal/scripts/abort_recovery.py` (confirmed via `ls`); referenced in §7.2 7.2.1/7.2.2/7.2.5 + §7.3 contract |
| G-3 | L304 | "Honest gap (§3.5)" | `molmetal/scripts/pre_flight_check.py` | **ALREADY_FILLED** | `molmetal/scripts/pre_flight_check.py` shipped, 881 LOC, Wf-260 25/25 PASS at 2026-09-28 19:55 CST |
| G-4 | L403 | "implementation TBD as TODO-208" | `pre_flight_check.py` | **ALREADY_FILLED** | Same as G-3 |
| G-5 | L404 | "implementation TBD as TODO-208" | `abort_recovery.py` | **STILL_MISSING** | Same as G-2 |
| G-6 | L527 | "script implementation gap is acknowledged" | runnable schema validator (gate_yaml_schema) | **ALREADY_FILLED** | Same as G-1 |

**Deduplicated unique items**: 3 (YAML schema validator / `abort_recovery.py` / `pre_flight_check.py`)

---

## §3 STILL_MISSING detail

### G-2 + G-5: `molmetal/scripts/abort_recovery.py`

**Doc reference (L262)**: "*Honest gap (§3.5)*: `abort_recovery.py` does NOT yet exist; §7.2 / §7.3 reference it as the contract, implementation is a follow-up Wf item (TODO-208 candidate, same as `pre_flight_check.py`)."

**Doc reference (L404)**: "*Recovery script contract*: `molmetal/scripts/abort_recovery.py` (referenced §7.2; implementation TBD as TODO-208)"

**Required contract per §7.3**:
1. Accept `--pattern {NaN, OOM, IndexError, cfm-collapse, ...}` and `--metal {Ru, Ir, Os, Rh, Re}` + `--seed` flags
2. Identify last-good `(metal, seed)` ckpt via `ls -t molmetal/reports/wf*/ckpt_*.pt | head -5`
3. Verify resume by re-loading ckpt + running 1 epoch smoke (loss finite, no NaN)
4. Emit `molmetal/reports/wf*/recovery_report.json` with `pattern`, `recovered`, `n_partial_completed` fields
5. Exit 0 on successful recovery; exit 1 on hard-fail

**Status**: NOT IMPLEMENTED. No file at `molmetal/scripts/abort_recovery.py`. No stub, no TODO comment.

**Action**: ultracode should implement per §7.3 contract; ship as TODO-208 follow-up.

---

## §4 ALREADY_FILLED detail

### G-1 + G-6: YAML schema validator

**Doc reference (L103)**: "*Honest gap (§3.5)*: YAML schema definition does NOT yet exist as a runnable validator; §1.3 references `molmetal/scripts/pre_flight_check.py --gate 1.3` which must be implemented as a follow-up Wf item (TODO-208 candidate)."

**Doc reference (L527)**: "*Honest framing note (§3.5)*: G11 is `true` for the schema-contract definition (this doc §2.1), NOT for a runnable schema validator (which is TODO-208 follow-up)."

**Status**: IMPLEMENTED.
- `molmetal/scripts/pre_flight_check.py:401` defines `gate_yaml_schema()` — passes if every YAML is parseable via `yaml.safe_load`
- v2 doc §2.1 schema-contract keys (data/model/flow/train/protocol/eval/output/compute/honest_framing) are lowercased by design; parseability is the actual launch-blocker (per L411-414 docstring)
- Wf-260 auto-verify ran 24/24 PASS for `§1.3 P0 config schema (7 active YAMLs all parse OK)`

### G-3 + G-4: `molmetal/scripts/pre_flight_check.py`

**Doc reference (L304)**: "*Honest gap (§3.5)*: `molmetal/scripts/pre_flight_check.py` does NOT yet exist as a runnable script — it is referenced as the auto-verify contract; implementation is a follow-up Wf item."

**Doc reference (L403)**: "*Auto-verify script contract*: `molmetal/scripts/pre_flight_check.py` (referenced §5; implementation TBD as TODO-208)"

**Status**: IMPLEMENTED + SHIPPED.
- `molmetal/scripts/pre_flight_check.py` (881 LOC) shipped via Wf-259 + Wf-260
- Wf-260 auto-verify: 25/25 PASS, 0 FAIL, wall 69.7s
- 7 gate functions: `gate_grep`, `gate_help`, `gate_file_exists`, `gate_min_lines`, `gate_json_threshold`, `gate_pytest`, `gate_which_or_apt`, `gate_yaml_schema`, `gate_csv_hash`
- STALE flag logic per §0.2 (24 h auto-mark rule)

---

## §5 Items NOT labeled as honest gaps in v2 doc

The computed task Wf-262 expected 3 STILL_MISSING items including `per_wave_checklist.py` and `wave_budget_tracker.py`. The audit reveals these are **NOT labeled as honest gaps** in v2 doc — they exist as **doc-internal templates** (not separate scripts):

### `wave_budget_tracker.py`
- Doc has §7.1 "Wave budget tracker (template)" as an inline markdown table
- Doc has G8_wave_budget_tracker_present = true (L521), referring to §7.1 + §C worked example
- No `molmetal/scripts/wave_budget_tracker.py` exists, but doc does NOT label this as a missing-script honest gap
- Verdict: **NOT a flagged honest gap** — it's a doc-template gate value

### `per_wave_checklist.py`
- Doc has §10.1 "Per-wave sub-checklist template (3 fields: gate, command, evidence)" as an inline markdown table
- No `molmetal/scripts/per_wave_checklist.py` exists, but doc does NOT label this as a missing-script honest gap
- Verdict: **NOT a flagged honest gap** — it's a doc-template

### Other minor markers (NOT honest gaps)
- L39 "chemprop v3 follow-ups" — refers to budget (1.8h + 5h + 7.5h = 14.3h), not implementation gap
- L241/L242/L245 "Per-wave timeout / Failure pattern / Decode_ratio collapse" — §7.2 row labels (criteria names), not implementation gaps
- L521 "G8_wave_budget_tracker_present" — gate value = true (satisfied)
- L524 "G11_yaml_schema_valid" — gate value = true (contract level)

---

## §6 Honest framing (party-twenty-spirit §3.5)

The v2 doc's strict honest-gap discipline (§3.5) means:
1. Every "Honest gap (§3.5)" label is a real, scoped implementation gap (3 occurrences)
2. Every "implementation TBD as TODO-208" is a forward-reference to the same TODO-208 batch (2 occurrences)
3. Every "implementation gap acknowledged" is a meta-statement about an earlier gap (1 occurrence)

Of the 6 honest-framed markers, **2/6 (33%) are still missing** (both pointing at `abort_recovery.py`). The remaining **4/6 (67%) were filled by Wf-259 + Wf-260**.

The 33% remaining gap is **single-script** in scope (`abort_recovery.py`) and **single-purpose** (§7.2 7.2.1 / 7.2.2 / 7.2.5 + §7.3 recovery contract). This is a small, well-scoped follow-up for ultracode.

---

## §7 Recommended action for ultracode

Per user request "如果有就启动ultracode去修" (if there are gaps, kick off ultracode to fix):

**Single recommended Wf**: Wf-263-Build abort_recovery.py per §7.3 contract
- Scope: implement 5-step recovery contract (L256-260 of v2 doc)
- Acceptance: 5 unit tests in `molmetal/tests/test_abort_recovery.py` (one per contract step)
- Update v2 doc L262 + L404 from "Honest gap" to "✅ SHIPPED via Wf-263"
- Mark TODO-208 complete

No other v2 doc honest gaps remain after Wf-256..261 ship cycle. The 4/6 already-filled markers (pre_flight_check.py + gate_yaml_schema) were shipped by Wf-259 + Wf-260.

---

**End of audit. Verified by**: Wf-262 subagent
**Audit path**: `molmetal/reports/wf262_audit/gap_list.md`
