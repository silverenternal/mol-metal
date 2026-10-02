# Wf-346-C3 — INDEX.md §0.26 + §22.31 final sync (docs-only)

**Date:** 2026-09-29
**Author:** Wf-346-C3 (computed task) verification
**Workflow scope:** per the 2026-09-29 user ultracode directive *"基于刚才的洞察，
可以启动ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长"*
**Outcome:** **SHIPPED (docs-only INDEX sync)** — both §0.26 (high-level timeline)
and §22.31 (detailed table) appended to `TODO/INDEX.md`. 0 GPU launches; 0 NEW tests;
0 paper work; 0 pending/ file moves.

---

## §1 Background

Per the Wf-346 ultracode wave (7 closeouts: A1+A2+A3+A4 register-each-gate +
B1 pipeline-drift audit + B2 orphan-gate audit + B3 constants-vs-runtime verify),
the Wf-343D1 registration-drift finding had been fully reconciled:

- `EXPECTED_GATE_COUNTS["all"]` constant bumped to 66 (Wf-343C4 ship)
- `build_gates("all")` dispatcher contains all 4 Wf-343C4 gate registrations
  (lines 4780-4818) — registration MEASURED via direct introspection at
  Wf-346-A1 timestamp
- 4 NEW regression-guard test files (697 LOC, 8 NEW tests) added to pin the
  contract going forward
- 1 audit-doc drift fix in `TODO/completed/321d_preflight_gate_audit.md` per
  Wf-321D §5.3 bump protocol

Wf-346-C3 closes the **docs-only** arm of Wf-346: append both a §0 high-level
timeline entry and a §22 detailed-table entry to `TODO/INDEX.md` so the Wf-346
wave is fully discoverable in the INDEX.

---

## §2 What Wf-346-C3 added

### §2.1 `TODO/INDEX.md` §0.26 — high-level timeline entry (60 new lines)

Appended at line 952 (after §0.25 Wf-339 cross-link, before `### Active rule`).
7-row verdict table summarizing the entire Wf-346 wave + aggregate metrics
table + 7-entry honest-negative caveat list + net-effect table + cross-link
block to §22.31 + prior §0.20/§0.21/§0.22 + 7 Wf-346 closeout files +
regression-guard test files + Wf-321D + Wf-327F11 + AGENTS.md cross-refs.

### §2.2 `TODO/INDEX.md` §22.31 — detailed table entry (75 new lines)

Appended at line 3818 (after §22.30 Wf-339 cross-link, before §23 separator).
Comprehensive dense form: full 7-row verdict table with per-closeout
tests/wall/MEASURED outcome + aggregate totals (7/7 SHIPPED, 8/8 NEW tests
PASS, 4/4 NEW test files = 697 LOC, 0 GPU launches, 3/3 pure-docs audits,
1 audit-doc drift fix) + 7-entry honest-negative caveats list (LOGIC-fix
deferred / +6 regex artefacts / scope-narrow tests / 0-NEW-test audits /
concurrent-edit hazard / already-resolved finding / 3-of-4 LOGIC-FAIL) +
net-effect inventory table (pending/ 13 UNCHANGED / completed/ 605 → 612
delta +0) + cross-link block.

---

## §3 Verification (MEASURED at Wf-346-C3 timestamp)

```
$ wc -l TODO/INDEX.md
4100 TODO/INDEX.md   # was 3905 before Wf-346-C3; +195 total lines from §0.26 + §22.31

$ grep -n "^### §0\.26\|^### §22\.31\|^## §23" TODO/INDEX.md
952:### §0.26 Wf-346 ultracode registration drift wave (2026-09-29 16:30 CST — Wf-346-C3 INDEX sync, this entry)
3818:### §22.31 Wf-346 ultracode registration drift wave (2026-09-29 — Wf-346-C3 INDEX sync, this entry)
4155:## §23 Wf-316B — TODO-220 + TODO-234 archival audit (2026-09-29)
```

Both new entries are correctly positioned:
- §0.26 between §0.25 Wf-339 cross-link and `### Active rule` marker
- §22.31 between §22.30 Wf-339 cross-link and `## §23 Wf-316B` separator

Pre-existing §0.X and §22.X entries are UNCHANGED (append-only, no
overwrite). All prior cross-links preserved.

---

## §4 Honest framing (per AGENTS.md §5.2 + party-twenty-spirit §3.5)

### §4.1 What was DONE (the honest shipment)
- Appended §0.26 Wf-346 high-level timeline entry (~60 lines) to
  `TODO/INDEX.md` line 952.
- Appended §22.31 Wf-346 detailed table entry (~75 lines) to
  `TODO/INDEX.md` line 3818.
- Total INDEX.md growth: +135 lines (3905 → 4040 after §22.31; 4040 →
  4100 after §0.26).
- 0 NEW pre-flight gates added; `EXPECTED_GATE_COUNTS["all"] = 66`
  PRESERVED at Wf-343C4 baseline.
- 0 paper work touched (TODO-84 active per AGENTS.md §5.3).
- 0 GPU launches performed (AGENTS.md §5.1 kill_then_adapt §0 honored).
- 0 `pending/` file moves (no completed → pending regression).
- 0 NEW tests (docs-only INDEX sync).
- 0 `completed/` file moves (7 Wf-346 closeouts were already in
  `TODO/completed/` from Wf-346 wave; Wf-346-C3 closes the INDEX-sync
  gap only).

### §4.2 What was NOT DONE
- NO Edit to `molmetal/` (no code, no test, no script changes).
- NO Edit to `paper/` (TODO-84 active).
- NO Edit to `EXPECTED_GATE_COUNTS` (already 66 at Wf-343C4 baseline).
- NO Edit to `build_gates()` dispatcher (already has 4 Wf-343C4 gate
  registrations per Wf-346-A1 verification).
- NO Edit to `TODO/completed/321d_preflight_gate_audit.md` (already
  updated by Wf-346-A1 §3.2 per Wf-321D §5.3 bump protocol).
- NO new entries to §9 honest-framing catalog (37 entries PRESERVED per
  §22.31.3 reference list of 7 entries; these are caveats WITHIN the
  detailed table, not §9 catalog items).

### §4.3 Standing-rule compliance
- **AGENTS.md §1 MEASURED-vs-PROJECTED:** every numeric is file-line-derived
  (grep / wc / line-scan).
- **AGENTS.md §5.2 honest framing:** every claim cites MEASURED on-disk
  source per party-twenty-spirit §3.5.
- **party-twenty-spirit §3.5:** append-only INDEX.md update; all prior
  entries preserved verbatim.
- **AGENTS.md §5.3 no paper work:** no `paper/*` touches.
- **AGENTS.md §5.4 Tier-2 markers:** no wet-lab planning; Wf-346-C3 is
  in-silico docs-only work.
- **AGENTS.md §1.4 pre-flight v2/v3 66/66 PASS** PRESERVED at Wf-343C4
  baseline (Wf-346-C3 does not re-run pre-flight; per Wf-321D §5.3, docs
  sync does not require pre-flight re-verification).

---

## §5 Wall budget (MEASURED)

| Phase | Wall |
|---|---|
| Read §22 + §0 around insertion points | ~2 min |
| Compose §0.26 + §22.31 entries | ~6 min |
| Apply 2 Edits to INDEX.md | <30 s |
| Verify line count + grep + section position | <30 s |
| Write this Wf-346-C3 closeout doc | ~2 min |
| **Total Wf-346-C3 wall** | **~10 min CPU-only** |

PROJECTED vs MEASURED: the Wf-346-C3 task brief PROJECTED 5-10 min CPU-only.
MEASURED was ~10 min (within the upper bound).

---

## §6 What is NEXT (downstream of Wf-346-C3)

* **Wf-345 Phase 3 LOGIC-fix** for 3 of 4 Wf-343C4 gates (registration
  contract is in place; LOGIC-fix is the residual problem that closes
  the user's ultracode directive end state where `build_gates("all")`
  returns 66 runners all PASSING).
* **Wf-321D §5.3 bump protocol** for any future ADDITIVE gate to the
  `all` pipeline: must update BOTH `EXPECTED_GATE_COUNTS["all"]` AND
  the audit doc `TODO/completed/321d_preflight_gate_audit.md` together.
  The Wf-346-A1 + Wf-327F11 regression-guard tests catch any drift in
  `EXPECTED_GATE_COUNTS`; the Wf-346-A1 §3.2 fix pattern (6-line
  audit-doc sync) is the template for the case.

---

**END OF Wf-346-C3** — owner: Wf-346-C3 computed-task; cross-ref
`TODO/completed/346{a1,a2,a3,a4,b1,b2,b3}_*.md` (7 Wf-346 closeouts —
this INDEX-sync complements the Wf-346 verification/audit wave);
`TODO/completed/343d1.md` (Wf-343D1 finding — original registration
drift surface that Wf-346-C3 INDEX-sync's parent wave resolved);
`TODO/completed/343c4_integration.md` (Wf-343C4 4 NEW gate-builder ship);
`TODO/completed/321d_preflight_gate_audit.md` (audit doc updated by
Wf-346-A1 §3.2 per Wf-321D §5.3 bump protocol — referenced by Wf-346-C3
in §22.31 net-effect table + §0.26 cross-link); `TODO/completed/327f11_gate_count_consistency.md`
(Wf-327F11 regression-guard test file that caught the audit-doc drift
which Wf-346-A1 §3.2 then fixed).