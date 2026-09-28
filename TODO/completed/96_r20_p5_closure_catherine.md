# TODO-96 — R20-P5 Closure theorem Catherine properties (6/6 PASS)

**Date**: 2026-09-26
**Status**: 🚀 **Active — queued for R20 (CPU + GPU wall-clock, ~2 weeks CPU)**
**Priority**: **HIGH** (L1 Closure verifier + L6 Closure d-block ≥ 0.95 + L12 Closure OOD cross-domain)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 + `TODO/completed/83j_lambda_closeout.md` §5.1
**Lambda gap(s)**: L1 (Closure verifier) + L6 (Closure d-block) + L12 (Closure OOD Phase 2)
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` (T1 SE3 SB + T7 SB-Bass)

---

## §0 Why this plan exists

Wf-36 P1 + Wf-37 P2 + Wf-41 P2 SHIP Closure theorem at multiple scales: L1 reachability 0.5108 (836 samples × 9 metals) + L6 reachability 0.5891 (2244 × 30 metals) + L12 OOD reachability 0.5581 (PlatinAI 5K OOD) / 0.0300 (MetalCytoToxDB). **R20-P5 lifts to reachability ≥ 0.95 on ≥ 25 metals with all 6 Catherine properties** (Catherine 2024 J. Algebra §3) PASS — the theorem-level claim required for paper §3.4 Closure theorem main result.

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU-only (closure verifier + 6 Catherine property tests) + GPU for 25-metal × 1000-sample sweep |
| Wall-time | 2 weeks CPU + 1 week GPU-0 |
| Output | `molmetal/reports/wf54_r20_p5/{closure_verifier_r20,closure_dblock_r20,cross_domain_ood}.json` |
| Tests | ~25 new tests: per-metal reachability + 6/6 Catherine properties + OOD cross-domain |

## §2 Cross-skill strategy

| Sub-task | Primary skill | Secondary skill |
|---|---|---|
| Catherine 6 properties | mol-metal-r18-math (T1 SE3 SB + T7 SB-Bass) | mol-metal-lambda-calculus |
| 25-metal × 1000-sample | mol-metal-lambda-calculus | mol-metal-click-chemistry (d-block SMARTS) |
| Cross-domain OOD | mol-metal-lambda-calculus | mol-metal-mechanism-mapping |

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P5.1** Catherine 6 properties | 1 wk CPU | `molmetal/reports/wf54_r20_p5/catherine_6_props.json` (6/6 PASS) | 6/6 PASS |
| **P5.2** Closure reachability ≥ 0.95 | 1 wk GPU-0 | `closure_reachability_r20.json` (25 metals × 1000 samples) | reachability ≥ 0.95 |
| **P5.3** Cross-domain OOD end-to-end | 1 wk CPU | `cross_domain_ood.json` (PlatinAI 5K + MetalCytoToxDB + tmQM d-block) | OOD reachability ≥ 0.85 |
| **P5.4** Wf-54 close-out | 2 days CPU | `TODO/completed/96_r20_p5_closure_catherine.md` | INDEX §17 cross-ref |

## §4 MEASURED-vs-PROJECTED

按 party-twenty-spirit §3.5:
- MEASURED = Wf-54 runs + Catherine 6/6 + reachability JSON
- PROJECTED = reachability ≥ 0.95 on ≥ 25 metals

## §5 Cross-references

- `TODO/completed/83j_lambda_closeout.md` §3.1 (L1/L6/L12 MEASURED)
- `TODO/completed/92_wf41_phase2_closure_ood.md` (Wf-41 P2 L12 bench layer)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L1/L6/L12 trajectory column)
- `TODO/pending/98_wf44_closure_ood_phase2.md` (L12 Phase 2 in-flight Q4 2026)
- `TODO/pending/99_r20_skill_ecosystem.md` (4 NEW R20 skills: `closure-theorem`, `catherine-properties`)