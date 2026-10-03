# Wf-317D — IMPROVEMENT_ENUM_2026_09_27.md Tier-2 strategy update

**Date**: 2026-09-29
**Owner**: Hugo Lee + Claude agent (party-twenty-spirit §3.5 honest framing)
**Status**: ✅ SHIP — 6 sections updated; 1 new §2.5 added; pre-flight 20/20 PASS
**Trigger**: User directive 2026-09-29 — "我们发二区就可以了，不用规划湿实验"
(Tier-2 journal is sufficient; do NOT plan wet-lab experiments). Per AGENTS.md §5.4.

---

## §1 Scope

Update `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` to reflect Tier-2 journal target
+ wet-lab deprecation cascade. This is the Wf-317D slice of the Wf-317 8-agent parallel
strategy update (Wf-317A INDEX / Wf-317B README / Wf-317C WEAK_TO_STRONG / **Wf-317D
IMPROVEMENT_ENUM** / Wf-317E LAMBDA_ROADMAP / Wf-317F 4 LOIs / Wf-317G 3 SOPs / Wf-317H
IMPROVEMENT_QUEUE).

---

## §2 Sections updated (6 modifications)

| # | Section | Change |
|---:|---|---|
| 1 | Header | Added `Updated: 2026-09-29` line + Wf-317D / Wf-316 cross-ref + §2.5 mention. Status adjusted to "4 个 P0 任务 (1 DONE 2026-09-27; 3 待做)" to match §0 enumeration reality (4 P0 listed, 1 already DONE) |
| 2 | §0 enumeration table | Task #1 marked `P0 ✅ DONE` (Wf-139 P7 was completed 2026-09-27); Task #2 Wf-134 annotated "CPU + GPU in-silico, keep"; Task #4 TODO-138 re-scoped from "JCIM" to "Tier-2 journal target" |
| 3 | §0 cross-ref block (NEW) | Added Wf-316 (TODO audit) cross-ref + Wf-317 (Tier-2 strategy) cross-ref, listing 8 parallel agents and noting pure text / no new measurements / no GPU launches |
| 4 | §1.5 (Task #4 TODO-138) | Re-scoped from "Paper v2 + JCIM submission package" to "Paper v2 + Tier-2 journal submission package"; added target journal list (JCAMD / J. Mol. Graph. Model. / WIREs Comput. Mol. Sci / Mol. Divers. / Comput. Biol. Chem.) |
| 5 | §2 Step 4 (TODO-138) | Inline annotation "(Tier-2 journal target per §2.5)" for navigation |
| **6** | **§2.5 NEW** | New section "Tier-2 strategy + wet-lab deprecation (Wf-317D, 2026-09-29)" with 3 subsections (deprecation catalog 16 rows, in-silico alternatives table, honest framing) |

---

## §3 §2.5 structure

### §3.1 §2.5.1 Deprecation catalog (16 rows)

| Row range | Items |
|---|---|
| 1-4 | 4 LOI drafts (Rusanov / PKU / 协和 / ORNL) → `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 5-7 | 3 wetlab SOPs (cytotoxicity / cellular uptake / binding affinity) → `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 8 | IC50 analysis pipeline → `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 9 | LOI consolidation review (Wf-308D) → stays active while LOIs DRAFT |
| 10 (reserved) | wetlab_loi_send_ready_package_v1.md (already moved to completed/ by Wf-316C) |
| 11 | wetlab cold-email template (Wf-310G) → `TEMPLATE_NOT_NEEDED_FOR_TIER2` |
| 12 | R20-P6 wet-lab IC50 validation (TODO-97 / Wf-58) → **DEPRECATED**, re-scope to in-silico |
| 13-16 | LAMBDA_ROADMAP R22-P1 (R23 / R24 / R25 clinical phases) → DEPRECATED or re-scope per Wf-317E |

### §3.2 §2.5.2 In-silico alternatives (5-row table)

Maps each Tier-2 deliverable to its in-silico replacement (GPU retrain / paper benchmarking
/ computational m18 / Tier-2 journal package). All paths are computable today without
external wet-lab dependency.

### §3.3 §2.5.3 Honest framing (party-twenty-spirit §3.5)

- MEASURED today: text-only, **0 new measurements**.
- NOT MEASURED / NOT auto-shippable: 4 LOIs + 3 SOPs + cold-email + IC50 pipeline
  are NOT auto-shippable per AGENTS.md §5.4 + TODO-84.
- 0 容忍 fabricate: every deprecation row cites a specific file path on disk; no
  aspirational wet-lab timing.
- Cross-cutting with Wf-316: 10 wetlab items in `pending/` retained with same
  `TEMPLATE_NOT_NEEDED_FOR_TIER2` rationale.

---

## §4 Pre-flight verification

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    /home/hugo/codes/mol-metal/molmetal/scripts/pre_flight_check.py --pipeline frontier

... summary ...
"summary": {
  "pass": 20,
  "fail": 0,
  "skip": 0,
  ...
}
```

**Pre-flight (frontier pipeline)**: 20/20 PASS, 0 FAIL, 0 SKIP. **Stale flag**: clean.
**No GPU launches** triggered. **No new code** added (text-only doc update).

---

## §5 Cross-references

- `AGENTS.md` §5.4 — Tier-2 journal target + NO wet-lab planning (user directive)
- `AGENTS.md` §5.3 — TODO-84 paper rule (still applies; paper work is user-authorization gated)
- `AGENTS.md` §5.1 — "Kill then adapt" GPU directive (still applies for any future retrain)
- `molmetal/reports/wf316c_wetlab_loi_audit.md` — Wf-316C TODO directory audit (4 moved,
  14 retained; 10 wetlab items retained)
- `tmp/wf317_tier2_strategy_update.js` — Wf-317 8-agent parallel orchestration script
- `TODO/completed/317d_improvement_enum_tier2_update.md` — this file (Wf-317D closeout)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 — honest-negatives catalog
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` — the updated doc (header + §0 + §1.5 +
  §2 Step 4 + §2.5 NEW)

---

## §6 Honest framing summary (party-twenty-spirit §3.5)

**MEASURED today**:
- 6 sections of IMPROVEMENT_ENUM_2026_09_27.md updated (above)
- 1 new §2.5 added with 3 subsections (deprecation catalog, alternatives, honest framing)
- Pre-flight frontier 20/20 PASS (0 FAIL, 0 SKIP)
- Cross-refs to AGENTS.md §5.4 + Wf-316C + Wf-317D explicitly stated

**NOT MEASURED / PROJECTED**:
- No new metric measurements (text-only doc update; 0 GPU launches)
- 16 deprecation rows cite on-disk file paths only — no projected timelines for when
  wet-lab might be re-enabled (re-enable requires explicit user directive reversal)
- §2.5.2 in-silico alternatives reference GPU retrain jobs that are still in flight per
  AGENTS.md §5.1 (TODO-181 G5 / TODO-182 G6 / TODO-119 P1.4 / Wf-299 CFM 3-wave) — none
  are claimed as DONE

**0 容忍 fabricate**:
- Every deprecation row cites a specific on-disk file path
- Every Tier-2 deliverable row cites a specific on-disk replacement
- §2.5.3 explicitly states "0 new measurements" and "NOT auto-shippable"
- No aspirational timing for wet-lab future-work

---

**End of Wf-317D ship report.** Owner: Hugo Lee; cross-ref `party-twenty-spirit` §3.5
honest framing; last update 2026-09-29.
