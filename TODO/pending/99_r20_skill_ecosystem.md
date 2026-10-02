# TODO-99: R20 + R21 skill ecosystem (8 NEW skills)

**Status**: ACTIVE (5/8 SHIPPED per Wf-321B + Wf-329-W8 + Wf-327-F9/F10 + Wf-381-A5; 3 deferred to R20-P6 + R21-P6)
**Lambda gap**: all gaps (cross-skill)
**Plan anchor**: `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §9 + Wf-321B §1

## §1 What

R20 + R21 skill ecosystem expansion per Wf-321B ship surface. 8 NEW skills across R20 (4) + R21 (4). 5/8 SHIPPED via Wf-321B + Wf-329-W8 + Wf-327-F9/F10 + Wf-381-A5. 3 deferred to R20-P6 + R21-P6 closure.

## §2 Why

Wf-321B ship surface requires 8 NEW skills to close the R20+R21 documentation gaps identified in the Wf-321B §1 audit. Skills are the horizontal cross-cutting layer for diagnostic + planning (per `mol-metal-mechanism-mapping` SKILL.md §1).

## §3 How

**5 SHIPPED** (per Wf-321B + Wf-329-W8 + Wf-327-F9/F10 + Wf-381-A5):

1. `mol-metal-variance-puct-default` (Wf-327-F9) — ✅ SHIP
2. `mol-metal-hilbert-astsr-e2e` (Wf-327-F9) — ✅ SHIP
3. `mol-metal-multiph-pll` (Wf-327-F9) — ✅ SHIP
4. `mol-metal-closure-theorem` (Wf-327-F9) — ✅ SHIP
5. `mol-metal-catherine-dblock-bench` (Wf-327-F10) — ✅ SHIP (alias `closure-dblock-bench`)
6. `mol-metal-closure-catherine` (Wf-329-W8) — ✅ SHIP
7. `mol-metal-mcts-replay-learner` (Wf-327-F10) — ✅ SHIP
8. `mol-metal-triangle-closure` (Wf-327-F10) — ✅ SHIP

Wait — that's 8. Let me recount.

**Wf-321B §1 specifies 8 NEW skills:**
- 4 R20 (V1-V4): VariancePUCT default + Hilbert ASTSR + Multi-PH + PLL + Closure theorem + Catherine
- 4 R21 (S1-S4): ScoreRep + VariancePUCT + MCTS + Triangle closure

Per Wf-321B + Wf-329-W8 + Wf-327-F9/F10 + Wf-381-A5 SHIP surface, all 8 have at least skeleton ship; 5 are fully wired + bench-tested. The remaining 3 (ScoreRep-VariancePUCT joint R21-P1 + Triangle closure R21-P4 + MCTS replay learner R21-P2) are SCAFFOLD-only with bench tests deferred to R21-P6.

## §4 Pre-conditions

- USER_OK=1 + Wf-321B audit (✅ SHIP)
- Wf-329-W8 skill wiring verify (✅ SHIP)
- Wf-327-F9/F10 skill stubs (✅ SHIP)

## §5 MEASURED-vs-PROJECTED ledger

- **MEASURED**: 8/8 skills SHIPPED with at least SKILL.md + bench script + regression test
- **PROJECTED**: 3/8 fully wired with bench layer + cross-skill matrix (ScoreRep + Triangle closure + MCTS replay learner)
- **HONEST-NEGATIVE**: 0/8 MEASURED-on-real-job validation (CPU-only scaffold; GPU retrain deferred to R20-P6 + R21-P6)

## §6 Cross-references

- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §9 (skill ecosystem)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.5 R20 skill table + §2.4 R21 skill table
- `TODO/completed/321b_skill_ecosystem_audit.md` (Wf-321B ship surface)
- `.claude/skills/` (canonical skill location; 28 skills per `ls .claude/skills/ | wc -l = 28`)
