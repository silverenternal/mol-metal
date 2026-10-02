# Wf-287B: §3.1.7 + §3.1.14 cross-ref verify (post-Wf-282B edit)

**Date**: 2026-09-29
**Trigger**: TODO-188 §3.1.7 + §3.1.14 cross-ref consistency after Wf-282B updated §3.1.14 gate text.
**Target**: `TODO/completed/207_unified_pre_flight_checklist_v2.md`

## Verification

### §3.1.14 (line 147) — P1 VERIFY gate test count

- Reads: **"184 tests PASS"** with breakdown `145 prior + +39 new per Wf-270D` (test_cfm_panel_v270d.py: 15 SA/QED + 10 Vina dock + 14 STUB_FALLBACK).
- Wf-282B edit applied correctly. **CONSISTENT**.

### §3.1.7 (line 140) — P1 VERIFY ablation flags

- Reads: `--no-multi-task`, `--no-wass`, `--no-kendall`, `--no-dann`, `--single-metal`, `--n-seeds-min=3`.
- Independent of §3.1.14 (gate on flags, not test count). No direct cross-ref to §3.1.14 needed.
- §5.4 (line 199) cross-refs §3.1.7 + §4.2.1 (ablation flag parallel) — also unaffected.
- **CONSISTENT** (no cross-ref dependency between §3.1.7 and §3.1.14; they are orthogonal P1 VERIFY gates).

### Stale "145 tests" references (3 found, 3 fixed)

| Line | Original | Fixed |
|---:|---|---|
| 737 | "§3.1.14 (P1 VERIFY gate '145 tests PASS' — needs update to 184)" | "§3.1.14 (P1 VERIFY gate '184 tests PASS' — updated per Wf-282B post-Wf-270D)" |
| 750 | "207 §3.1.14 P1 VERIFY gate '145 tests PASS' now needs update to '184 tests PASS' (post-Wf-270D); follow-up edit per pre_flight_check.py --gate 3.1.14 re-verify." | "207 §3.1.14 P1 VERIFY gate '184 tests PASS' (post-Wf-270D + Wf-282B edit; edit applied, re-verify pending via pre_flight_check.py --gate 3.1.14)." |
| 935 | "145 tests: molmetal/tests/ (full chemprop v2 v3 v4 suite, last refresh per Wf-256)" | "184 tests: molmetal/tests/ (full chemprop v2 v3 v4 + CFM panel + STUB_FALLBACK suite, post-Wf-270D + Wf-282B edit; refresh log per Wf-256)" |

### Preserved (legitimate) "145 prior" references

- Line 147 (§3.1.14): "145 prior + +39 new per Wf-270D" — historical breakdown that derives the 184 count.
- Line 742: "Total test suite = 184 PASS (145 prior per §3.1.14 + 39 new per Wf-270D)" — same breakdown in §9.AG narrative.

Both kept verbatim as they are CONSISTENT with the post-Wf-282B gate.

## Fixes applied

1. §3.1.14 cross-ref note (line 737) — removed "needs update to 184" claim; reflects current state.
2. §9.AG TODO follow-up hook (line 750) — removed obsolete "now needs update" TODO; reflects Wf-282B completion.
3. Chemprop v2 v3 reference (line 935) — bumped 145 → 184 in the bottom reference roll-up.

## Re-verify pending

- `pre_flight_check.py --gate 3.1.14` to confirm MEASURED "184 passed" on actual pytest run.
- pytest command per §3.1.14 column 4 (auto-verify):
  ```
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/ -q 2>&1 | tail -1
  ```
  (must report "184 passed" or higher).

## Honest framing (party-twenty-spirit §3.5)

- Wf-282B edit was correctly applied to the gate row (§3.1.14, line 147).
- 3 stale "145 tests PASS" mentions existed in narrative/cross-ref lines (737, 750, 935), all driven by the Wf-270D follow-up hook text written before Wf-282B shipped.
- All 3 stale refs now consistent with post-Wf-282B state.
- No §3.1.7 cross-ref inconsistency detected (§3.1.7 is orthogonal gate on ablation flags, not test count).
- Total fix count: **3 line edits**, **0 structural changes**, **0 new sections added**.