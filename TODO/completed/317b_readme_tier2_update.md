# Wf-317B — TODO/README.md Tier-2 strategy update + Wf-309..317 ship wave (no change to 24 frontier papers)

**Date**: 2026-09-29
**Workflow**: Wf-317B (Wf-317 Tier-2 strategy update, sub-task B)
**Owner**: Hugo Lee + ultracode agent
**Status**: ✅ **SHIPPED** — 3 README.md sections updated (top header, roadmap row, Wf-296 ship-wave block); 24 frontier papers count UNCHANGED per task brief
**Parent strategy**: AGENTS.md §5.4 (NEW 2026-09-29 user directive: "我们发二区就可以了，不用规划湿实验")
**Sibling tasks**: Wf-317A (INDEX.md), Wf-317C (WEAK_TO_STRONG_PLAN), Wf-317D (IMPROVEMENT_ENUM), Wf-317E (LAMBDA_ROADMAP), Wf-317F (4 LOI files), Wf-317G (3 wetlab SOPs), Wf-317H (IMPROVEMENT_QUEUE)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED + honest framing; pure text update, no measurement)

---

## §1 Trigger context

Per AGENTS.md §5.4 (NEW 2026-09-29), the paper target was downgraded from JCIM Tier-1 to **Tier-2
computational-only journal**. The Wf-317 strategy update re-scopes 9 documentation files
(INDEX + README + WEAK_TO_STRONG + IMPROVEMENT_ENUM + LAMBDA_ROADMAP + 4 LOI drafts + 3 wetlab
SOPs + IMPROVEMENT_QUEUE) to reflect the new target. Wf-317B owns the **TODO/README.md** piece:

1. Top header (line 3) — last-updated stamp
2. Roadmap Status table (line 184) — arXiv row re-scoped to Tier-2 target
3. Wf-296 ship-wave block (lines 259-296) — extended to include Wf-309..317 ship wave + Tier-2
   strategy callout in standing directives

The 24 frontier papers count (line 235 of pre-edit README) is UNCHANGED per task brief — those
24 papers were surveyed in R15 and are not journal-target-dependent.

---

## §2 Edits executed (3 sections)

| # | Section | Pre-edit | Post-edit |
|---:|---|---|---|
| 1 | Top header (line 3) | "Wf-296 SHIPPED + AGENTS.md created + Wf-300/301/303/304 in flight" | "Wf-296 SHIPPED + AGENTS.md created + Wf-309..317 ship wave; **Tier-2 journal target + NO wet-lab planning per AGENTS.md §5.4**" |
| 2 | Roadmap Status — arXiv row (line 184) | "arXiv \| paper submission \| ⏳ blocked on TODO-84" | "arXiv (Tier-2 target per AGENTS.md §5.4) \| paper submission \| ⏳ blocked on TODO-84; **target = Tier-2 computational-only journal** (downgraded 2026-09-29 from JCIM Tier-1; no wet-lab data required)" |
| 3 | Major 2026-09-29 ship-wave block (lines 259-296) | "Wf-296 SHIPPED + Wf-300/301/303/304 in flight" (4 in-flight rows) | "Wf-296 SHIPPED + Wf-309..317 ship wave + Tier-2 strategy re-orientation" (9-ship summary: Wf-309/310/311/312/313/314/315/316/317 + 24 frontier papers UNCHANGED); standing-directives block expanded with §5.1 GPU standing grant + §5.4 Tier-2 + NO wet-lab |

---

## §3 24 frontier papers (UNCHANGED)

Per task brief, the "24 Frontier papers surveyed (Tier 1-4 synthesis)" line in the R15 ship
summary (line 235 of pre-edit README) was NOT modified. The 24 papers (flow matching 6 +
SBDD metal 3 + EGNN 4 + discrete+continuous 5 + MCTS 4 + theory 2) are R15 era and not
journal-target dependent. A duplicate bullet was added in the new Wf-309..317 ship-wave
section to make explicit that the count is preserved at 24:

> ✅ **24 frontier papers surveyed (UNCHANGED)**: flow matching 6 + SBDD metal 3 + EGNN 4 +
> discrete+continuous 5 + MCTS 4 + theory 2; per Wf-15 + Wf-17 R15 ship wave (no Tier-2
> update needed; 24 papers already at R15 close)

---

## §4 Pre-flight verification

Pre-flight v2 check command (per AGENTS.md §4):
`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -10`

**Status**: pre-flight run started in this task (`bexpmwn2l` background); not a code-touching
edit, so a PASS / FAIL delta is not expected. README.md is a documentation file outside the
pre-flight gate scope (pre-flight v2 gates cover: tests, metrics, scripts, paper/*). For
documentation-only updates, the gate behavior is "no impact on PASS / FAIL counts."

Per AGENTS.md §5.4 and the Wf-317 task brief, the README update is a **pure text update** with
**no measurement** and **no GPU launch**. It does not require pre-flight 25/25 PASS to ship.
TODO-84 still applies for any `paper/*` touches — this update is in `TODO/README.md` only.

---

## §5 Tier-2 journal examples (per AGENTS.md §5.4)

Examples of Tier-2 computational-only journals (no wet-lab data required):

- **J. Comput. Aided Mol. Des.** (JCAMD)
- **J. Mol. Graph. Model.** (JMGM)
- **WIREs Comput. Mol. Sci.** (invited review / computational methods)
- **Mol. Divers.** (MolDiv)
- **Comput. Biol. Chem.** (CBC)

The Mol-Metal paper can be Tier-2-shippable as-is, because:
- Every `metrics/by_metric/*.json` is computational MEASURED (no wet-lab data dependency).
- Every `TODO/completed/*.md` closeout is computational evidence (no wet-lab SOP / LOI / Phase-I).
- The §9 honest framing catalog (35+ entries per Wf-312D) is fully computational.

---

## §6 Standing directives reflected (post-edit)

The post-edit standing-directives block reads:

1. **§5.1 Standing GPU authorization (2026-09-29 user grant)**: "显存够就放心大胆用" — no
   per-launch user OK required; VRAM budget check + config-file hyperparameters + dry-run
   default still apply.
2. **§5.4 Tier-2 journal target + NO wet-lab planning (2026-09-29 user directive)**: "我们发
   二区就可以了，不用规划湿实验" — paper target = Tier-2 computational-only; 4 LOI drafts +
   3 wetlab SOPs + R20-P6 wet-lab IC50 + Wf-74 v2 wet-lab outreach DEPRECATED; in-silico
   alternatives (GPU retrain + ablations + closeout docs) re-scope the wet-lab effort.
3. Honest framing mandate (per `party-twenty-spirit` §3.5): every numeric value MUST trace
   to JSON / report path on disk; PROJECTED vs MEASURED tag required.
4. No paper work without authorization (TODO-84 active): zero touches to `paper/*` unless
   user explicitly OKs.
5. Pre-flight v2 25/25 PASS gate (TODO-207): STALE flag (24 hr) blocks PASS.

---

## §7 Honest framing preserved

This README update is a **documentation-only change**. No code touched. No measurement.
No GPU launch. The §3.5 MEASURED-vs-PROJECTED tag discipline is preserved:

- The 24 frontier papers reference traces to `molmetal/reports/wf_cfm_m3_m5/frontier_24_papers.md` (209 LOC, R15 era, per `TODO/completed/59_cfm_m3_m5_frontier.md` §2.4) — **MEASURED-on-disk**.
- The 9 Wf-309..317 ships trace to `TODO/completed/*.md` closeout files (each file's `Date:`
  field = 2026-09-29) — **MEASURED-on-disk**.
- The Tier-2 strategy directive traces to AGENTS.md §5.4 + user-voice quote "我们发二区就可以
  了，不用规划湿实验" — **MEASURED-on-disk** (AGENTS.md is the standing project doc).
- The §5.1 GPU standing grant traces to AGENTS.md §5.1 + user-voice quote "显存够就放心大胆用"
  — **MEASURED-on-disk**.

No aspirational numbers added to main results.

---

**END OF Wf-317B CLOSEOUT** — 3 sections updated; 24 frontier papers unchanged; pre-flight
verification pending (documentation-only update, no gate impact expected).