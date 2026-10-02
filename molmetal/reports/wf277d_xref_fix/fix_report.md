# Wf-277D — fix 1 inconsistency + 2 implicit gaps in 207 §9 / 188 §4 cross-refs

**Date**: 2026-09-29
**Workflow**: Wf-277D (post-Wf-275D fix)
**Owner**: Hugo Lee + Wf-277D ultracode agent
**Skill alignment**: `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-PROJECTED) + `mol-metal-paper-writing` §4 cross-ref contract + TODO-84 ACTIVE rule (no paper/*.tex writes without authorization)

---

## §1 Scope

Wf-275D audit found:
- 19/20 207 §9 entries CONSISTENT with 188 §4 cite targets
- **1 inconsistency**: 207 §9.4 cite-target-section "§6" but the m17 v3 BELOW_RANDOM 0.4544 honest-negative lives in 188 §5.2 (m17 DO NOT claim) + 188 §4.5.2 (Wf-266 v3 OOD fix), not in 188 §6 (which is the Weak-metrics ledger with an unrelated m17 iMoLD row)
- **6/8 explicit cite (188 §4 → 207 §9), 2 implicit gaps**: 188 §4.1 (m14 CFM decode_ratio) and 188 §4.4 (m7 Tanimoto diversity) content aligned but did not explicitly cite 207 §9.6.3 and 207 §9.6.2 respectively

Wf-277D fixes:
1. Added clarifying cite-target note in 207 §9.4 (canonical 188 §5.2 + §4.5.2 cite targets)
2. Added "see also 188 §4.1" reverse-cite in 207 §9.6.3
3. Added "see also 188 §4.4" reverse-cite in 207 §9.6.2

188 paper edits DEFERRED per TODO-84 ACTIVE rule — cross-refs preserved on the 207 side as reverse-cites.

## §2 Inconsistency fix (207 §9.4)

**Before**: 207 §9.4 referenced "§9.1 + §9.3 point #3 cite the same Wf-253d m17 BELOW_RANDOM 0.4544 figure" without explicitly identifying the 188 cite target. §9.1 + §9.3 point #3 had imprecise "paper §6 honest-negative" / "§6 honest-negative" phrasing.

**After** (207 §9.4 line 381): Added explicit "Cite target in 188 (Wf-277D fix, 2026-09-29)" bullet that names the canonical 188 cite targets:
- **188 §5.2 m17 DO NOT claim** (last bullet: "We DO NOT claim m17 DrugOOD v3 forward-pass reeval ≥ Wf-185 v2 baseline 0.6707...")
- **188 §4.5.2 Wf-266 v3 OOD fix (Wf-263A + Wf-266 LOMO-fix)** (structural-pass + empirical-NOT-yet-MEASURED status)

The bullet also documents WHY 188 §6 was the wrong target: 188 §6 is the Weak-metrics ledger (m17 iMoLD row about iMoLD head-to-head, NOT v3 BELOW_RANDOM). The prior imprecise "paper §6" / "§6 honest-negative" phrasing in §9.1 / §9.3 point #3 is acknowledged as imprecise; precise targets are §5.2 + §4.5.2.

**Honest framing (party-twenty-spirit §3.5)**: the original cite-target-section error was MINOR (no content error, only imprecise section number); numbers and rationale are consistent across 207 §9.4 + 188 §5.2 + 188 §4.5.2; the fix clarifies the cite path without introducing new claims.

## §3 Implicit gap fix #1 (188 §4.1 → 207 §9.6.3)

**Before**: 188 §4.1 line 178 + 192 cited "synthetic upper bound 1.000, gap = −0.0042 — expected per `party-twenty-spirit` §3.5" but did NOT cross-reference 207 §9.6.3 (which is the canonical upstream entry for the 0.9958 vs 1.000 expected gap framing).

**After** (207 §9.6.3 line 438): Added "See also 188 §4.1 (Wf-277D cross-ref fix, 2026-09-29)" bullet that names 188 §4.1 as the paper-side anchor:
- 188 §4.1 m14 real CFM decode_ratio MEASURED numbers (Wf-188 0.9958 on n=26,801, synthetic gap −0.0042 expected per `party-twenty-spirit` §3.5)

188 §4.1 paper-edit DEFERRED per TODO-84 ACTIVE rule — cross-ref preserved on the 207 side as reverse-cite.

## §4 Implicit gap fix #2 (188 §4.4 → 207 §9.6.2)

**Before**: 188 §4.4 line 245 cited the m7 Tanimoto +22.5% lift + +0.820 gap to TargetDiff per TODO-180 §1.3; 207 §9.6.2 is the upstream honest-negative catalog for the same +0.820 gap. 188 §4.4 did not explicitly cite 207 §9.6.2.

**After** (207 §9.6.2 line 429): Added "See also 188 §4.4 (Wf-277D cross-ref fix, 2026-09-29)" bullet that names 188 §4.4 as the paper-side anchor:
- 188 §4.4 m7 diversity Tanimoto 0.0397 with Tanimoto reward (+22.5% lift) + +0.820 gap to TargetDiff 0.860 ceiling (TODO-180 §1.3)

188 §4.4 paper-edit DEFERRED per TODO-84 ACTIVE rule — cross-ref preserved on the 207 side as reverse-cite.

## §5 Discrepancy note (3 implicit gaps vs 2 found)

Wf-275D audit table in `molmetal/reports/wf275d_xref_audit/audit_report.md` §3 explicitly marks 2 ⚠️ MINOR GAP rows (188 §4.1 + §4.4); the original Wf-277D task description mentions "3 implicit gaps". This discrepancy is not a substantive finding — the Wf-275D audit enumerates only 2 rows that are ⚠️ MINOR GAP; the rest are ✅ CONSISTENT. Wf-277D treats this as a counting discrepancy (2 vs 3) rather than a missed gap, since re-auditing the 188 §4.5.x entries confirms they explicitly cite 207 §9.AB (G5 BLOCK) + 207 §9.5 + 207 §9.7 + 207 §9.AC (LOMO-fix verify) + 207 §9.AH (Wf-269 catalog) — all marked ✅ CONSISTENT in the Wf-275D audit table.

## §6 TODO follow-up hooks

- **TODO-84 ACTIVE rule**: 188 paper edits DEFERRED. The reverse-cites on the 207 side (added in this Wf-277D fix) preserve the cross-reference path so reviewers can locate the 188 anchor without needing a 188-side edit. When TODO-84 is lifted (user authorization for paper edits), the corresponding 188 §4.1 + 188 §4.4 edits can be applied to make the cite explicit on BOTH sides.
- **Wf-268D-Xref timeline entry** (per `TODO/completed/207_unified_pre_flight_checklist_v2.md` §11.1): add a new entry "Wf-277D 2026-09-29 paper §4 cross-ref fix (207 §9.4 cite-target + §9.6.2 + §9.6.3 reverse-cites)" to keep the Wf-256..277 timeline current.
- **Cross-link**: this fix_report.md → `molmetal/reports/wf275d_xref_audit/audit_report.md` (the upstream Wf-275D audit that found the issues) + 207 §9.4 cite-target bullet (line 381) + 207 §9.6.2 reverse-cite (line 429) + 207 §9.6.3 reverse-cite (line 438).

## §7 MEASURED-vs-PROJECTED label discipline audit

| Edit | Status | Verdict |
|---|---|:---:|
| 207 §9.4 cite-target fix | Cite-target clarification only; no new claims introduced; numbers + rationale unchanged | ✅ DISCIPLINED |
| 207 §9.6.2 reverse-cite | Reverse-cite pointing to existing 188 §4.4 MEASURED numbers; no fabrication | ✅ DISCIPLINED |
| 207 §9.6.3 reverse-cite | Reverse-cite pointing to existing 188 §4.1 MEASURED numbers; no fabrication | ✅ DISCIPLINED |

All edits preserve party-twenty-spirit §3.5 0 容忍 不掩盖 discipline: no aspirational numbers introduced; cross-refs cite MEASURED evidence paths only.

---

**End of fix_report.md. Wf-277D SHIPPED**: 1 inconsistency fixed (207 §9.4 cite-target) + 2 implicit gaps closed (207 §9.6.2 + §9.6.3 reverse-cites); 188 paper edits deferred per TODO-84 ACTIVE rule.