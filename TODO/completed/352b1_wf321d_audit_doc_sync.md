# Wf-352-B1 — Wf-321D audit doc sync (gate-count references → 71)

**Date:** 2026-09-29
**Workflow:** Wf-352 ultracode residual wave (continued from Wf-350 ship + Wf-352-A1)
**Task:** Wf-352-B1 — Sync `TODO/completed/321d_preflight_gate_audit.md` §4 references
from stale 55/56 → 71 to match §2.1 + §6 (which already reference 71)
**Status:** SHIPPED (CPU-only, no GPU launches)

---

## §1 Outcome

**Pre-state (per Wf-352-A1 sync):** `TODO/completed/321d_preflight_gate_audit.md`
§2.1 + §5 + §6 already referenced `71` gates for the `all` pipeline
(Wf-350-A2 final bump 70 → 71 per Wf-352-A1 §1). However, §4
(MEASURED-only verification table) still carried the **stale** 55/56
counts from the original Wf-321D audit (when the count was 56). This
was a §4-vs-§2.1/§6 doc-internal drift that contradicted the
Wf-352-A1-bumped `EXPECTED_GATE_COUNTS["all"] = 71` constant in
`pre_flight_check.py:3034`.

**Post-state (Wf-352-B1):** §4 references all updated to 71. The doc is
now internally consistent (§2.1 = §4 = §5 = §6 = 71).

**Net delta:** 3 Edit operations touching 6 numeric tokens (3 → 71).

---

## §2 Stale references found

Per task brief: search for `66 | 66 gate | 66 expected`. Direct grep
yielded no gate-count 66 hits in the doc (only `26633` / `26800` CSV
row-count thresholds in §3.2, which are NOT gate counts and were left
untouched per task brief §4 "NOT in dates or other contexts").

Inspection revealed the §4 section was the lagging sub-section:
§2.1, §5.2, §6 all already referenced 71, while §4 still said 55/56.

| Line | Old | New |
|---:|---|---|
| 189 | `results for the 55 gates:` | `results for the 71 gates:` |
| 218 | `**Total** | **56** | | **56 / 56 MEASURED-only** |` | `**Total** | **71** | | **71 / 71 MEASURED-only** |` |
| 220 | `**Result: 55/55 gates are MEASURED-only.**` | `**Result: 71/71 gates are MEASURED-only.**` |

**Note on task-brief expectation:** Brief expected "5 string replacements";
actual count was 3 Edit operations touching 6 numeric tokens (3 string
ops because the §4 table row Total/Result columns were a single Edit
each). The §3.2 CSV row counts `26633`/`26800` were intentionally NOT
changed (not gate-count contexts).

---

## §3 Per-section consistency (post-sync)

| Section | Pre-sync value | Post-sync value |
|---|---:|---:|
| §2.1 (`all` pipeline total) | 71 | 71 (unchanged) |
| §4 (MEASURED-only count) | 55 / 56 | **71 / 71** |
| §5.2 (EXPECTED_GATE_COUNTS["all"]) | 71 | 71 (unchanged) |
| §6 (n_gates_total, n_measured_only) | 71 / 71 | 71 / 71 (unchanged) |

All sections now agree on `all` = 71.

---

## §4 Files changed

| File | Change |
|---|---|
| `TODO/completed/321d_preflight_gate_audit.md` | 3 Edit ops (§4 stale 55/56 → 71) |
| `TODO/completed/352b1_wf321d_audit_doc_sync.md` | NEW (this file) |

---

## §5 Tests added

0 NEW tests. This is a docs-only sync; the
`molmetal/tests/test_wf352a1_expected_gate_counts.py` (2 tests
SHIPPED per Wf-352-A1) already pins the runtime invariant
(`EXPECTED_GATE_COUNTS["all"] == len(build_gates("all")) == 71`),
which is the load-bearing correctness check. The §4 doc update is a
documentation-side cosmetic.

---

## §6 Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

- **Measured:** 3 Edit operations applied to
  `TODO/completed/321d_preflight_gate_audit.md` lines 189/218/220.
  Numeric tokens changed: 6 (1 × "55→71" + 3 × "56→71" + 2 × "55→71").
  No other tokens changed.
- **Projected:** None.
- **Source:** Direct Edit tool calls; verified via re-grep (no stale
  55/56 gate-count tokens remain).
- **Honest-negative caveats:** §2.1 P3 row shows `40` but the
  parenthesized sub-list enumerates 46 individual gate entries
  (Wf-306C 9 + Wf-308A 1 + Wf-308B/Wf-312C/Wf-315A 8 + Wf-312B 1 +
  Wf-312E 1 + Wf-313C 1 + Wf-313H/Wf-320F/Wf-322E 3 + Wf-314E 1 +
  Wf-315D/Wf-319C 2 + Wf-319B 1 + Wf-319A 2 + Wf-321A 1 + Wf-321E 1
  + Wf-321D 1 + Wf-256 1 + Wf-323F 1 + Wf-325E 1 + Wf-325F 1 +
  Wf-327 1 + Wf-343C4 4 + Wf-347A1 2 + Wf-350-A3 1 + Wf-350-A2 1 =
  46). This is a separate doc-internal drift (P3 row count vs
  enumerated list) NOT in scope for Wf-352-B1 (which targets §4 only
  per task brief). Flagged here for downstream fix in a future Wf.

---

## §7 Audit trail

Per Wf-321D §5.2 design contract: bumping EXPECTED_GATE_COUNTS requires
updating the constant AND the closeout doc. Wf-352-A1 closed the
constant bump (pre_flight_check.py:3034, 66 → 71). Wf-352-B1 closes the
doc-side §4 sub-section sync.

The §4 stale 55/56 was not surfaced by the Wf-321D gate-count
consistency audit (which only checks `EXPECTED_GATE_COUNTS["all"] ==
len(build_gates("all"))`); it required a manual cross-section
inspection to catch. This is a known blind-spot of the audit design —
it does not catch §-vs-§ doc drift inside `321d_preflight_gate_audit.md`
itself.

---

**END OF Wf-352-B1 CLOSEOUT**
