# Wf-284D — §10 Cross-Reference Audit (Retry of Wf-281B §10 audit)

**Date**: 2026-09-29
**Source doc**: `TODO/completed/207_unified_pre_flight_checklist_v2.md` §10
**Audit method**: Read + Grep (no Edit on cross-ref targets, only on §10 itself)

---

## §10.1 Per-wave sub-checklist template

**Cross-references audited**: §1.1, §1.2, §1.3, §1.4, §3.1.7, §5.1, §4.2.1-§4.2.5, §4.2.10, §4.1, §9.2, `TODO/completed/157_wf146_pre_flight_checklist.md` §9.1, `molmetal/reports/wf119_cfm_retrain/`, `molmetal/reports/wf146_smoke_3ep_chemprop/per_metal_summary.json`

**Verification result**:
- §1.1..§1.4, §3.1.7, §5.1, §4.2.1-§4.2.5, §4.2.10, §4.1, §9.2 — ALL exist in current 207 doc at expected line ranges
- `TODO/completed/157_wf146_pre_flight_checklist.md` §9.1 exists at line 330 with both `wf146_smoke_3ep/` and `wf146_smoke_3ep_chemprop/per_metal_summary.json` output paths cited (line 346 contains `wf146_smoke_3ep_chemprop/per_metal_summary.json`)
- 157 §12 row 1 "LIFTED" token referenced in §10.2 — verified present at line 552 of `157_wf146_pre_flight_checklist.md` (both `completed/` and `archive/` copies)
- `molmetal/reports/wf119_cfm_retrain/` — directory referenced as future emission target; NOT yet on disk (TODO-119 P1.3 PENDING GPU per §4.1) — **EXPECTED** (not a broken ref)
- `molmetal/reports/wf146_smoke_3ep_chemprop/` — directory referenced as historical canonical 157 §9.1 output path; on-disk status: NOT yet verified, but the path matches 157 §9.1 verbatim (line 346), so cross-ref to source is INTACT

**Verdict**: §10.1 has NO broken cross-refs. **CONSISTENT** (10 cross-refs audited, 10 PASS).

---

## §10.2 P0 BLOCKER user OK gate (re-confirmation every wave)

**Cross-references audited**: `TODO/INDEX.md` "kill then adapt" directive, BD-4 + BD-5, lines 104, 1046, 1080; §1.1; `TODO/completed/157_wf146_pre_flight_checklist.md` §12 row 1; `TODO/archive/157_wf146_pre_flight_checklist.md` §12 row 1

**Verification result**:
- `INDEX.md` line 104: contains "kill then adapt" content (BD-5 row in BLOCKERS table: "Standing 'kill then adapt' GPU directive") ✓
- `INDEX.md` line 1046: contains "kill then adapt" content (BD-4 row in owners/decisions table: "Standing 'kill then adapt' GPU directive — explicit fresh user OK") ✓
- `INDEX.md` line 1080: contains "kill then adapt" content ("All R19/R20 GPU retrain DEFERRED per standing 'kill then adapt' directive") ✓
- `157_wf146_pre_flight_checklist.md` §12 row 1 at line 552 — "LIFTED" token present ✓
- `archive/157_wf146_pre_flight_checklist.md` §12 row 1 at line 552 — "LIFTED" token present ✓
- §1.1 reference to "INDEX.md LIFTED" — verified in §1.1 row of §1 P0 BLOCKER table at line 79 ✓

**NOTE (informational, NOT a broken ref)**: INDEX.md has TWO overlapping BD-4/BD-5 numbering schemes across two tables:
- BLOCKERS table (line 103-104): BD-4 = SMILES featurization, BD-5 = "kill then adapt"
- OWNERS/decisions table (line 1045-1047): BD-4 = "kill then adapt", BD-5 = Vina/meko install
The §10.2 cite "BD-4 + BD-5, lines 104, 1046, 1080" is CORRECT in that ALL three lines DO contain "kill then adapt" content (BD-5 at line 104, BD-4 at line 1046, general statement at line 1080). The BD numbering inconsistency is upstream in INDEX.md, not a 207 §10.2 broken ref.

**Verdict**: §10.2 has NO broken cross-refs. **CONSISTENT** (6 cross-refs audited, 6 PASS).

---

## §10.3 Auto-verify JSON hash + report path (every wave)

**Cross-references audited**: 3×2 = 6 emission paths; `molmetal/scripts/per_wave_checklist.py`; §11 line reference (was 898, now 929); Wf-262

**Verification result**:
- 6 emission paths `molmetal/reports/wf207_pre_flight/wave_{1,2,n}.{json,md}` — paths are ILLUSTRATIVE naming convention (per §10.3 table), NOT literal default outputs of `per_wave_checklist.py`
- `molmetal/scripts/per_wave_checklist.py` — exists at canonical path ✓ (36,180 bytes per Wf-262 ship)
- **INCORRECT LINE NUMBER** (FIXED): §10.3 previously stated "per §11 line 898" but line 898 is "Wf-243..252 fixes: TODO/completed/119_closeout_v2_loi_sop.md §3" — NOT `per_wave_checklist.py`. The actual `per_wave_checklist.py` reference is at §11 line 929 ("Per-wave sub-checklist script (SHIPPED Wf-262): `molmetal/scripts/per_wave_checklist.py` (36,180 bytes; drives §10.1 per-wave sub-checklist emission)").
- **PATH-NAMING INCONSISTENCY** (DOCUMENTED): §10.3 table paths (`wf207_pre_flight/wave_N.{json,md}`) differ from `per_wave_checklist.py` default (`wf262_checklist/per_wave.{json,md}`); the `--out` / `--out-md` CLI flags allow operator override, so the §10.3 illustrative paths ARE reachable in principle, just not the script default.

**FIX APPLIED (line 878)**: Updated §10.3 narrative to:
1. Correct the line reference from "§11 line 898" → "§11 line 929"
2. Add explicit mention that `per_wave_checklist.py` default output is `wf262_checklist/per_wave.{json,md}` (overridable via `--out` / `--out-md`)
3. Clarify that the §10.3 table paths are illustrative of the per-wave naming convention, not literal default outputs

**Verdict**: §10.3 had 2 inconsistencies (wrong line number, undocumented path-naming mismatch). **BOTH FIXED**. (3 cross-refs audited, 3 PASS post-fix).

---

## Summary

| §10 entry | Cross-refs audited | Consistent | Inconsistencies | Fixes applied |
|---|---:|---:|---:|---:|
| §10.1 | 10 | 10 | 0 | 0 |
| §10.2 | 6 | 6 | 0 | 0 (1 upstream INDEX.md note documented) |
| §10.3 | 3 | 1 → 3 | 2 | 2 (line-number + path-naming mismatch) |
| **TOTAL** | **19** | **19** | **2** | **2** |

**MEASURED labels verified**:
- §10.1 evidence column uses honest framing (e.g. "mtime delta OR chat-session user verbatim" for §1.1, "selected GPU ≤ 5 GB" for §1.2) — NO PROJECTED-only claims presented as MEASURED
- §10.2 explicitly disambiguates auto-verify is MANUAL, not bash-token-grepped (corrects a prior V1-style "grep LIFTED" expectation)
- §10.3 JSON SHA-256 immutability note is a forward-looking contract, NOT a MEASURED claim — honest framing preserved

**Edits made**:
1. `TODO/completed/207_unified_pre_flight_checklist_v2.md` line 878 (now 908): replaced incorrect "per §11 line 898" with correct "per §11 line 929" + added path-naming clarification + default-output reference

**No content deleted**; all 3 §10 sub-sections preserved verbatim aside from the §10.3 fix.