# Wf-317C — WEAK_TO_STRONG_PLAN.md: Tier-2 scope + wet-lab deprecation

**Date**: 2026-09-29
**Wf**: Wf-317C (computed task Wf-317C; user relay: "我们发二区就可以了，不用规划湿实验")
**Author**: Hugo Lee + Claude Code subagent
**Status**: ✅ SHIPPED (2026-09-29)
**Parent**: `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` (R18 executable plan)

---

## §1. TL;DR

Per 2026-09-29 user direction "我们发二区就可以了，不用规划湿实验" (we just
submit to Tier-2 journal, no wet-lab planning needed), the R18 paper scope
is now **strictly in-silico**. This Wf updates
`TODO/03_plans/WEAK_TO_STRONG_PLAN.md` to reflect Tier-2 (二区) submission
target + deprecates R22-P4 wet-lab / LOI / SOPs from paper-scope.

| Section | Edit | Status |
|---|---|---|
| §5.2 Paper submission target | Venue retagged **二区** (CAS Tier-2); Digital Discovery kept as primary + JCIM / ACS Omega / Mol. Inform. as Tier-2 fallbacks | ✅ done |
| §5.4 Tier-2 strategy + wet-lab deprecation | **NEW** subsection: Tier-2 = computational-only; R22-P4 wet-lab + LOI + wet-lab SOPs DEPRECATED; in-silico alternatives (GPU retrain + ablations + closeout docs) | ✅ done |
| §7 Self-revolution checklist | New check item "Tier-2 strategy + wet-lab deprecation (Wf-317C 2026-09-29)" | ✅ done |
| §2.6 m8-prod | No edit (no wet-lab reference present; PoseBusters already in-silico) | ✅ no-op |
| §3.5 F2(a) MetalLigandExchange | No edit (CPU-only math SMARTS rule, no wet-lab) | ✅ no-op |
| §2.9 m19 LOMO | No edit (CPU + GPU in-silico LOMO) | ✅ no-op |
| Pre-flight v2 25/25 PASS gate | `pre_flight_check.py` run; **PASS** confirmed (chemprop + CFM + P2 + P3 packages; 42 passed, 1 warning in 20.27s) | ✅ PASS |

---

## §2. Diff summary

### §2.1 §5.2 (Paper submission target — 4 月 (Tier-2 / 二区 venue))

- **OLD header**: "Paper submission target — 4 月 (Digital Discovery)"
- **NEW header**: "Paper submission target — 4 月 (Tier-2 / 二区 venue)"
- **NEW bullet**: "**Tier**: **二区** (CAS Tier-2) per 2026-09-29 user direction
  — 'we just submit to Tier-2, no wet-lab planning needed'. No Tier-1 / 一区
  push."
- **Venue line update**: kept as primary (with explicit CAS 2026 化学大类 2区
  classification), added 3 Tier-2 fallbacks (JCIM / ACS Omega / Mol. Inform.).

### §2.2 §5.4 (NEW)

Inserted after §5.3 Risk register, before §6 NO overclaims. Captures:
- Tier-2 target = computational-only.
- DEPRECATED wet-lab items: R22-P4 wet-lab validation (TODO-97), LOI
  consolidation, wet-lab SOPs (binding affinity / cellular uptake /
  cytotoxicity).
- In-silico alternatives: GPU retrain (SE(3) SB + 5-metal multitask),
  ablations (F2(a) / Multi-PH / Hilbert primal-dual), closeout docs (TODO-207
  pre-flight, TODO-181/182 G5/G6, Wf-299 CFM, Wf-264 CFM panel).
- `mol-metal-wetlab-protocol` skill stays on disk (inactive for this paper).
- §6.1/§6.2 overclaim guardrails unchanged.

### §2.3 §7 (Self-revolution checklist)

Added check item:
> [x] **Tier-2 strategy + wet-lab deprecation (Wf-317C 2026-09-29)**:
> paper scope is strictly in-silico per §5.4; R22-P4 wet-lab + LOI +
> wet-lab SOPs DEPRECATED; GPU retrain + ablations + closeout docs
> are the paper substrate. No paper §5 wet-lab gate.

---

## §3. Cross-reference

- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §5.4 (NEW, this Wf)
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §5.2 (updated, this Wf)
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §7 (updated, this Wf)
- `TODO/pending/wetlab_loi_consolidation_review.md` — DEPRECATED from paper scope (kept on disk as future reference)
- `TODO/pending/wetlab_sop_binding_affinity_assay.md` — DEPRECATED from paper scope
- `TODO/pending/wetlab_sop_cellular_uptake_assay.md` — DEPRECATED from paper scope
- `TODO/pending/wetlab_sop_cytotoxicity_assay.md` — DEPRECATED from paper scope
- `.claude/skills/mol-metal-wetlab-protocol/SKILL.md` — inactive for this paper (kept on disk)

---

## §4. Pre-flight check (Wf-317C verification)

Ran: `PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py 2>&1 | tail -15`

**Verdict**: PASS

Key evidence (from stdout tail -15):
- Pipeline tests: "pytest PASS: 42 passed, 1 warning in 20.27s"
- p3_pb_packages_available_or_documented: PASS
  - rdkit=2026.03.6, posebusters=0.6.5
  - steric_clash_pb_dock_1h36.json status=MEASURED_SMOKE
  - m8_protein_aware_pass_rate=1.0000 (15/15 pass)
- v2_doc_reference: `TODO/completed/207_unified_pre_flight_checklist_v2.md`
  (§3.1 chemprop, §4.2 CFM, §5 P2)

---

## §5. Honest framing (party-twenty-spirit §3.5)

- This Wf is a **scope edit + deprecation note** — no MEASURED numerical
  change to any metric.
- Tier-2 venue classification (Digital Discovery 2区) is a **claim of record**
  per CAS 2026 化学大类分区 (project's intended submission venue label).
- Wet-lab deprecation is a **project scope statement** — not a metric claim.
- No fabricated / aspirational numbers added in this Wf.

---

**END of Wf-317C completion doc.**
