# 188 Paper-Edit Awareness (Wf-281D, 2026-09-29)

**Date**: 2026-09-29
**Reason**: Wf-281D — formal awareness doc for the **PROPOSED (NOT APPLIED)** `paper/sections/04_results.tex` + `paper/sections/05_ablation.tex` edits that surfaced from Wf-277D, Wf-278D, and the Wf-281 wave. Wf-278E synthesis (`molmetal/reports/wf278_synthesis/followup_summary.md` §4 item 6) recommended **Wf-284 "188 paper-edit (gated on TODO-84 wet-lab LOI)"** as a follow-up; this doc records that recommendation as **awareness only**, NOT action — every entry below is BLOCKED per **TODO-84 ACTIVE rule** (`TODO/pending/84_no_paper_task_without_authorization.md`).
**Status**: 🟡 **AWARENESS DRAFT (NO `paper/*.tex` writes performed)** — ships to `TODO/completed/188_paper_edit_awareness_2026-09-29.md` ONLY.
**Owner**: Hugo Lee + Wf-281D ultracode
**Constraint**: NO `paper/` directory writes. PROPOSED edits tracked here until wet-lab PI LOI lands + user authorization lifts TODO-84.
**Skill alignment**: `party-twenty-spirit` §3.5 (0 容忍 不掩盖; honest framing: PROPOSED ≠ APPLIED) + `mol-metal-paper-writing` §4 verbatim template + TODO-84 ACTIVE rule.

---

## §0 TL;DR

Three recent waves — **Wf-277D** (`molmetal/reports/wf277d_xref_fix/fix_report.md`), **Wf-278D** (`molmetal/reports/wf278_synthesis/followup_summary.md` §1 D), and the **Wf-281 wave** (Wf-281A Ru seed0 recovery + Wf-281C G6 readiness — Wf-281B did not launch in this session) — surfaced **5 PROPOSED `paper/sections/04_results.tex` edits** (1 from Wf-277D row 1, 1 from Wf-277D row 2, 1 from Wf-281A contingent on G5 Ru seed0 recovery, 1 from Wf-281C contingent on G6 launch, plus 1 from the Wf-278E recommendation Wf-284 catch-all). **None are applied** — TODO-84 ACTIVE rule (issued by user on 2026-09-26 13:00 CST) forbids `paper/*.tex` writes without explicit user OK. Unblock requires: (a) wet-lab PI LOI signed (any of ORNL/PKU/协和) per `TODO/pending/wetlab_loi_ornl_protein_metal.md` + `wetlab_loi_pku_cisplatin_ic50.md` + `wetlab_loi_xiehe_nci60.md` (all currently DRAFT, not sent), (b) user authorization to lift TODO-84 per the rule memo §2.2 protocol. Until then, every entry below stays in PROPOSED state; do NOT cite as APPLIED.

---

## §1 Candidate from Wf-277D (`molmetal/reports/wf277d_xref_fix/fix_report.md`)

**C1 — 188 §4.1 (m14 real CFM decode_ratio 0.9958) reverse-cite → 207 §9.6.3.** The current 188 §4.1 cites the synthetic 1.000 expected-gap framing but does NOT name §9.6.3 as the upstream honest-negative catalog. **Source**: Wf-277D §3 fix_report.md (added reverse-cite on 207 side; 188 side deferred).

**C2 — 188 §4.4 (m7 Tanimoto diversity 0.0397 + +0.820 gap to TargetDiff) reverse-cite → 207 §9.6.2.** The current 188 §4.4 cites TODO-180 §1.3 but does NOT name §9.6.2 as the upstream honest-negative catalog. **Source**: Wf-277D §4 fix_report.md (added reverse-cite on 207 side; 188 side deferred).

---

## §2 Candidate from Wf-278D (`molmetal/reports/wf278_synthesis/followup_summary.md` §1 D)

**C3 — 188 §5.2 m17 DO NOT claim row wording tighten.** Refine the cite target from imprecise "§6 honest-negative" / "§6 honest-negative" phrasing to the precise canonical targets `188 §5.2 + §4.5.2` (already corrected on 207 §9.4 side via Wf-277D fix; 188 §5.2 row also needs the m17 v3 BELOW_RANDOM 0.4544 wording tightened). **Source**: Wf-277D §2 + Wf-278D §1 D — pre-existing imprecise phrasing; §9.4 cite-target fix is already applied (207 side), paper-side wording is the remaining gap.

---

## §3 Candidate from Wf-281A (`molmetal/reports/wf281a_ru_seed0_recovery/recovery_report.md`)

**C4 — 188 §4.5.1 (Wf-266 G4 15-pair aggregate) update on G5 completion.** When G5 Ru seed=0 finishes (~2 epochs remaining from this report's write time per Wf-281A ETA) and `wf278a_g4_aggregate.py` re-runs producing n=15 with Ru 3-seed mean replacing 2-seed mean, update 188 §4.5.1 honest framing to disclose the G5 recovery. **Contingent**: G5 run must complete + per_seed_metrics_seed0.json emitted + re-aggregation successful. **Source**: Wf-281A §3 + §recommended next-step. **Note**: this is a §4.5.1 update, not a fresh paper edit — but it touches `paper/sections/04_results.tex` §4.5.1 wording.

---

## §4 Candidate from Wf-281C (`molmetal/reports/wf281c_g6_readiness/readiness_report.md`)

**C5 — 188 §5 ablation paragraph for Wf-124 P3 G6 per-innovation ablation.** When G6 launches (after G5 frees GPU-1; ~4.5 hr wall remaining for G5 per Wf-281C §4) and produces per-lever `--single-task` / `--wasserstein-weight 0.0` ablation numbers, add a §5 ablation paragraph comparing per-innovation lift. **Contingent**: G6 launch + 30 CONFIRMED runs execute + per-lever JSON emitted. **Source**: Wf-278B §1 G6 dry-run (60 PLANNED tuples, 30 CONFIRMED per Wf-281C §3) + Wf-281C §4 launch-after-G5 verdict.

---

## §5 Cross-cutting Wf-284 umbrella (`molmetal/reports/wf278_synthesis/followup_summary.md` §4 item 6)

> "Wf-284: 188 paper-edit gated by TODO-84 — once wet-lab LOI lands (Wf-58 / TODO-97), apply all 207 §9.x 'See also 188 §x.y' reverse-cites as forward paper edits in `paper/sections/04_results.tex` + `paper/sections/05_ablation.tex`."

**C6 — Wf-284 batch-apply ALL `207 §9.x → 188 §x.y` reverse-cites.** Once wet-lab LOI lands, apply every cross-ref that currently exists only as a 207-side reverse-cite (per Wf-277D + Wf-278D work) as a forward cite on the paper side. This is the umbrella for C1..C5 above plus any unlisted reverse-cites surfaced by the Wf-283 §10 cross-ref audit extension (recommended Wf-283 item in Wf-278E §4 item 5).

---

## §6 TODO-84 gate (no edits without wet-lab LOI)

Per `TODO/pending/84_no_paper_task_without_authorization.md` §1.1 + §2.1, every entry C1..C6 is a **paper-type task** that touches `paper/sections/04_results.tex` (or §5/§6) and therefore falls under the **forbidden-without-authorization** category:

| § entry | Forbidden category | Files touched | Authorization needed |
|---|---|---|---|
| C1 (188 §4.1 reverse-cite) | Paper section modification | `paper/sections/04_results.tex` | TODO-84 §2.2 user OK |
| C2 (188 §4.4 reverse-cite) | Paper section modification | `paper/sections/04_results.tex` | TODO-84 §2.2 user OK |
| C3 (188 §5.2 m17 wording tighten) | Paper section modification | `paper/sections/04_results.tex` | TODO-84 §2.2 user OK |
| C4 (188 §4.5.1 G4→n=15 update) | Paper section modification | `paper/sections/04_results.tex` | TODO-84 §2.2 user OK + G5 run completion |
| C5 (188 §5 G6 ablation paragraph) | Paper section modification | `paper/sections/05_ablation.tex` | TODO-84 §2.2 user OK + G6 run completion |
| C6 (Wf-284 batch-apply) | Paper section modification (multiple) | `paper/sections/04_results.tex` + `paper/sections/05_ablation.tex` | TODO-84 §2.2 user OK + wet-lab PI LOI |

Per the rule memo §7, this gate is in effect **indefinitely from 2026-09-26 13:00 CST until user revokes**. The user's verbatim instruction: "所有 paper 类型的任务在我授权之前禁止执行" (all paper-type tasks are forbidden from execution without my authorization).

---

## §7 Tracking table — candidates × LOI status × blocked-until

| ID | Candidate edit | Source wave | LOI status | Blocked until |
|---|---|---|---|---|
| **C1** | 188 §4.1 (m14 CFM decode_ratio) reverse-cite → 207 §9.6.3 | Wf-277D §3 | ⏸️ All 3 LOIs DRAFT (not sent) | ORNL/PKU/协和 PI LOI signed + TODO-84 lifted |
| **C2** | 188 §4.4 (m7 Tanimoto diversity) reverse-cite → 207 §9.6.2 | Wf-277D §4 | ⏸️ All 3 LOIs DRAFT (not sent) | ORNL/PKU/协和 PI LOI signed + TODO-84 lifted |
| **C3** | 188 §5.2 m17 DO NOT claim wording tighten | Wf-278D §1 D | ⏸️ All 3 LOIs DRAFT (not sent) | ORNL/PKU/协和 PI LOI signed + TODO-84 lifted |
| **C4** | 188 §4.5.1 G4 n=14→n=15 honest framing update | Wf-281A §3 | ⏸️ All 3 LOIs DRAFT + G5 in-flight | G5 Ru seed0 epoch 50 + TODO-84 lifted |
| **C5** | 188 §5 G6 per-innovation ablation paragraph | Wf-281C §4 + Wf-278B §1 | ⏸️ All 3 LOIs DRAFT + G6 not launched | G6 launch + 30 CONFIRMED runs complete + TODO-84 lifted |
| **C6** | Wf-284 batch-apply ALL `207 §9.x → 188 §x.y` reverse-cites | Wf-278E §4 item 6 | ⏸️ All 3 LOIs DRAFT (not sent) | ORNL/PKU/协和 PI LOI signed + TODO-84 lifted (umbrella over C1..C5) |

**LOI status detail** (per `TODO/pending/wetlab_loi_*.md` headers):
- **LOI-ORNL-2026-09** (`wetlab_loi_ornl_protein_metal.md`): v1.0 DRAFT, ready for review, **not sent at this revision** (Hugo Lee coordinates cold-email per `TODO/INDEX.md` §12.3 P3 mitigation)
- **LOI-PKU-2026-09** (`wetlab_loi_pku_cisplatin_ic50.md`): v1.0 DRAFT, ready for review, **not sent at this revision**
- **LOI-协和-2026-09** (`wetlab_loi_xiehe_nci60.md`): v1.0 DRAFT, ready for review, **not sent at this revision**

All 3 LOIs are at the same status: drafted, internally reviewable, but no outbound send + no PI signature + no wet-lab measurement. The wet-lab gate (TODO-97 R20-P6 ship target + `TODO/completed/97_r20_p6_wetlab_ic50.md`) is itself BLOCKED on LOI send + signature.

---

## §8 How to unblock (do NOT act unilaterally)

Per TODO-84 §2.1 + §2.2 protocol, unblocking C1..C6 requires **user authorization** (not autonomous execution). The path is:

1. **Step 1 — Wet-lab LOI send**: Hugo Lee coordinates the 3 LOI cold-emails (ORNL/PKU/协和) per `TODO/INDEX.md` §12.3 P3 mitigation. **Status**: cold-emails NOT YET SENT (per LOI doc headers).
2. **Step 2 — Wet-lab PI response + LOI signature**: PI counter-signs one or more LOIs. **Status**: no response yet (LOI not sent).
3. **Step 3 — Wet-lab execution**: PI lab executes MTT / SRB / CellTiter-Glo / SPR / ITC per the agreed LOI scope. **Status**: not started.
4. **Step 4 — TODO-84 lift**: User explicitly authorizes paper edits per the §2.2 template:
   > "User, can I [specific paper action]? Touches: [list files]. Based on data: [list Wf reports]. Risk: [honest-negative findings visible to reviewer]."
5. **Step 5 — Apply C1..C6**: With TODO-84 lifted, apply the 5 candidate edits + Wf-284 batch. Per `mol-metal-paper-writing` §4 verbatim template + `party-twenty-spirit` §3.5 honest framing.

**Autonomous unblock is FORBIDDEN**. Even if wet-lab LOI lands and user wants to lift TODO-84, the agent MUST ask user first per the rule memo §2.1 ("Agent MUST ask user first: 'Can I launch paper-type wf X?'"). Acting on C1..C6 without this step is a TODO-84 violation.

---

## §9 Honest framing — PROPOSED edits, NOT APPLIED

Per `party-twenty-spirit` §3.5 (0 容忍 不掩盖), the strict honest framing for this awareness doc is:

- **PROPOSED**: C1..C6 are documented candidates sourced from Wf-277D + Wf-278D + Wf-281A/C + Wf-278E recommendations. They are real suggestions grounded in real upstream reports (`molmetal/reports/wf277d_xref_fix/fix_report.md` + `molmetal/reports/wf278_synthesis/followup_summary.md` + `molmetal/reports/wf281a_ru_seed0_recovery/recovery_report.md` + `molmetal/reports/wf281c_g6_readiness/readiness_report.md`).
- **NOT APPLIED**: No `paper/sections/04_results.tex` or `paper/sections/05_ablation.tex` writes have been performed in Wf-281D or any prior wave. The 207-side reverse-cites added by Wf-277D + Wf-278D live in `TODO/completed/207_unified_pre_flight_checklist_v2.md`, NOT in `paper/`.
- **BLOCKED on TODO-84**: Every C1..C6 entry is BLOCKED per the rule memo §1.1 forbidden-without-authorization table. User's verbatim instruction (2026-09-26 13:00 CST) is the gate.
- **BLOCKED on wet-lab LOI**: Even if TODO-84 were lifted, the wet-lab gate (TODO-97 + R20-P6) would still need PI signature + execution per the §8 path.
- **NO fabrication**: This doc does NOT claim that any of the C1..C6 edits have been applied to `paper/`. It documents the candidates as candidates only, per the party discipline of MEASURED vs PROPOSED transparency.
- **DO claim**: "Wf-277D + Wf-278D + Wf-281A/C surfaced 5 + 1 (Wf-284 umbrella) PROPOSED `paper/sections/04_results.tex` + `paper/sections/05_ablation.tex` edits; all 6 are BLOCKED per TODO-84 ACTIVE rule + wet-lab LOI gate."
- **DO NOT claim**: "Wf-281D has applied any of these edits to `paper/`." (no writes performed).
- **DO NOT claim**: "Wf-284 has been launched." (Wf-284 is a recommended future wave, not yet launched).

---

## §10 Cross-references

- `TODO/pending/84_no_paper_task_without_authorization.md` — TODO-84 ACTIVE rule (gate memo)
- `TODO/pending/wetlab_loi_ornl_protein_metal.md` + `wetlab_loi_pku_cisplatin_ic50.md` + `wetlab_loi_xiehe_nci60.md` — 3 wet-lab LOIs (all DRAFT, not sent)
- `TODO/completed/97_r20_p6_wetlab_ic50.md` + `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` — wet-lab ship target (R20-P6)
- `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` — the paper §4 DRAFT that C1..C6 would modify (when TODO-84 lifted)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 — the 207-side reverse-cite catalog that C1..C6 + Wf-284 would mirror on the paper side
- `TODO/completed/m14_lead_deployment_precondition.md` — Wf-275C formal spec for m14 LEAD YuelBond precondition (related but separate; not a paper-edit)
- `molmetal/reports/wf277d_xref_fix/fix_report.md` — Wf-277D source (C1 + C2)
- `molmetal/reports/wf278_synthesis/followup_summary.md` — Wf-278D + Wf-278E source (C3 + C6)
- `molmetal/reports/wf281a_ru_seed0_recovery/recovery_report.md` — Wf-281A source (C4)
- `molmetal/reports/wf281c_g6_readiness/readiness_report.md` — Wf-281C source (C5)
- `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 — 0 容忍 不掩盖 (always apply)
- `.claude/skills/mol-metal-paper-writing/SKILL.md` — paper §1-§7 + appendix writing helper (USE WITH AUTHORIZATION per TODO-84)

---

## §11 Final verdict

- **Status**: AWARENESS doc only; no `paper/*.tex` writes.
- **Candidates tracked**: 5 + 1 umbrella (C1..C6).
- **Gate**: TODO-84 ACTIVE + wet-lab LOI.
- **Action**: NONE required from Wf-281D. Next actionable steps live in §8 (LOI send + user OK to lift TODO-84), which are user-coordinated and OUT of scope for Wf-281D autonomous execution.

**Date**: 2026-09-29 (Wf-281D SHIPPED)

---

Co-Authored-By: Claude Code <noreply@anthropic.com>
