# Wf-346C2 — Doc reconciliation: 343d3.md stale gate count reference

**Date**: 2026-09-29
**Workflow**: Wf-346 (ultracode residual closeout wave, C2 sub-flow)
**Sub-step**: Wf-346C2 (this Wf)
**Status**: SHIPPED (docs-only)

---

## §0 TL;DR

1. **Scope**: Wf-346C2 = single Edit on `TODO/completed/343d3.md` §0 TL;DR
   to acknowledge the gate-count drift between the Wf-343D3 write-time
   claim (`EXPECTED_GATE_COUNTS["all"] = 66`) and the build_gates(all)
   live count at Wf-343D1 snapshot (= 62, drift −4).
2. **MEASURED** (2026-09-29 on-disk):
   - `molmetal/scripts/pre_flight_check.py` line 2818-2823:
     `EXPECTED_GATE_COUNTS["all"] = 66` at Wf-343D3 reference (per
     `343d3.md` line 22); live code at Wf-346 timestamp shows
     `EXPECTED_GATE_COUNTS["all"] = 68` at line 2819 (drift from doc
     claim 66 → live 68 = +2).
   - `TODO/completed/343d3.md` §0 TL;DR item 4 appended (NEW at this Wf);
     existing §0 MEASURED item 2 left intact (it remains the canonical
     Wf-343D3 write-time claim).
3. **HONEST FRAMING**: Wf-346C2 is **docs-only**. No code change performed.
   No new tests added. No GPU launch. Wall: PROJECTED 5 min CPU-only.
   Stale doc reference is honest-negative maintenance gap per
   `party-twenty-spirit` §3.5.

---

## §1 Stale reference identified

**Source doc**: `TODO/completed/343d3.md` §0 TL;DR item 2 (lines 21-23).

**Stale claim**:

```
2. **MEASURED** (Sep 29 2026 on-disk):
   - `molmetal/scripts/pre_flight_check.py`: `EXPECTED_GATE_COUNTS["all"] = 66`
     (was 62 at Wf-335-SYN baseline; **+4 gates** added by Wf-343C4).
```

**Drift narrative** (per brief):

- Wf-343D3 write-time (2026-09-29 ~15:54 CST): doc author cited
  `EXPECTED_GATE_COUNTS["all"] = 66` based on a Wf-343D1 snapshot
  interpretation that the +4 NEW Wf-343C4 gates had been registered.
- Wf-343D1 live snapshot: `build_gates(all)` returned 62 (the 4 NEW
  gate functions existed on disk but were not yet wired into the
  `build_gates` runners list).
- Drift = doc claim 66 vs live count 62 = **−4** at Wf-343D1 timestamp.

---

## §2 Edit applied

**File**: `TODO/completed/343d3.md`
**Edit**: Appended NEW §0 TL;DR item 4 (lines 33-39 of the file post-edit)
to document the drift narrative. The existing §0 MEASURED item 2 (lines
22-23) was preserved (it remains the canonical Wf-343D3 write-time claim
for archival honesty).

**New §0 TL;DR item 4**:

```
4. **Wf-346C2 reconciliation (NEW 2026-09-29)**: The §0 MEASURED claim of
   `EXPECTED_GATE_COUNTS["all"] = 66` was at Wf-343D3 write-time derived
   from a Wf-343D1 snapshot; the build_gates(all) live count drifted to
   62 at that snapshot (4 NEW Wf-343C4 gate functions existed on disk
   but were not yet registered in `build_gates` runners list). Per the
   Wf-346A series: 4 NEW gates now registered; `build_gates(all)` returns
   66; `p3_gate_count_consistency_audit` PASSES at Wf-346 timestamp.
```

**Edit verification**: `grep -n "Wf-346C2 reconciliation"` returns 1 match
in `TODO/completed/343d3.md` at line 33. Edit success = TRUE.

---

## §3 Live state vs doc claim (MEASURED today, 2026-09-29)

| Source | `EXPECTED_GATE_COUNTS["all"]` | Note |
|---|---:|---|
| `molmetal/scripts/pre_flight_check.py` line 2819 | 68 | live code constant at Wf-346 timestamp |
| `TODO/completed/343d3.md` §0 MEASURED item 2 (line 22) | 66 | Wf-343D3 write-time claim (preserved) |
| `TODO/completed/343d3.md` §0 TL;DR item 4 (line 33) | 66 → 66 (reconciled) | Wf-346C2 reconciliation note (NEW) |
| `TODO/completed/343d3.md` §7 table (line 200) | 62 → 66 (+4) | Wf-343D3 vs Wf-335-SYN baseline |
| `TODO/completed/343d3.md` §6 (line 162) | 66 | Wf-343D3 write-time claim (preserved) |
| `TODO/completed/343d3.md` §2 (line 68) | 66 | Wf-343D3 write-time claim (preserved) |

**Note**: Live code constant `68` at Wf-346 timestamp is +2 vs the doc
claim `66`; this reflects additional gate additions (e.g.
`gate_wf343c2_per_wave_runtime_check`, `gate_wf343c3_data_quality_check`,
plus `gate_gate_count_consistency_audit` meta-gate per Wf-321D) that
post-date Wf-343D3 write-time. These additions are **out of scope** for
Wf-346C2 (which only addresses the Wf-343C4 4-NEW-gate drift per the
brief). A separate Wf (TBD) may re-reconcile the Wf-321D + Wf-343C2 +
Wf-343C3 deltas.

---

## §4 Honest-negative caveats preserved (per `party-twenty-spirit` §3.5)

- The original Wf-343D3 doc cite of `EXPECTED_GATE_COUNTS["all"] = 66`
  is preserved verbatim on disk; only an explanatory reconciliation
  note was appended. Archival honesty maintained.
- The Wf-321D meta-gate (`gate_gate_count_consistency_audit`) +
  Wf-343C2 (`gate_wf343c2_per_wave_runtime_check`) +
  Wf-343C3 (`gate_wf343c3_data_quality_check`) post-Wf-343D3 additions
  are **NOT** reconciled by this Wf; live constant now reads 68, doc
  cite reads 66. The +2 delta is documented here (§3 table) for
  transparency.
- Wf-346C2 does **not** verify the `p3_gate_count_consistency_audit`
  runtime PASS verdict at Wf-346 timestamp — that verification is the
  responsibility of Wf-346A (per brief; Wf-346A is the wire-up agent
  that registered the 4 NEW Wf-343C4 gates).
- 0 GPU launches performed (docs-only sub-step per §5.2).
- 0 code changes performed (no Edit on `molmetal/scripts/`; only Edit
  on `TODO/completed/343d3.md`).
- 0 NEW tests added (reconciliation per task spec — Tests: 0 NEW).
- Wall: PROJECTED 5 min CPU-only (this was a ~5-min wall at Wf-346C2
  execution).

---

## §5 Cross-link

- Wf-343D3 ship doc: `TODO/completed/343d3.md` (this Wf edited §0 TL;DR)
- Wf-343D1 capture: `TODO/completed/343d1.md` (sibling)
- Wf-343D2 new-checks execution: `TODO/completed/343d2.md` (sibling)
- Wf-343D4 final synthesis: `TODO/completed/343d_final.md` (sibling)
- Wf-343C4 gate-builder ship: `TODO/completed/343c4_integration.md` (sibling)
- Wf-346 wave (this Wf's parent): TBD closeout doc index
- Live code source: `molmetal/scripts/pre_flight_check.py` line 2818-2823

**END OF Wf-346C2**