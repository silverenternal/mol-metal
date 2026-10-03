# TODO-98 — Wf-44 P2 Closure OOD Phase 2 (in-flight Q4 2026)

**Date**: 2026-09-26
**Status**: 🚀 **Active — in flight under Wf-44 P2 (Q4 2026)**
**Priority**: **HIGH** (L12 Phase 2 full closure OOD end-to-end)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/completed/92_wf41_phase2_closure_ood.md` (Wf-41 P2 Phase 1 catalog) + `TODO/completed/87_r2_debug.md` (Wf-44 8 root causes)
**Lambda gap(s)**: L12 (Closure OOD cross-domain)
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` (T1 SE3 SB)

---

## §0 Why this plan exists

Wf-41 P2 SHIP L12 Phase 1 catalog (8 root causes ranked) + Phase 2 in flight under Wf-44. Wf-44 P1 SHIP root cause catalog (2 CRITICAL + 4 MAJOR + 2 MINOR); Wf-44 P2 SHIP predict-the-mean baseline check. **Wf-44 P2 finalizes the closure OOD end-to-end** with Tanimoto<0.4 OOD protocol + reachability metrics + 6 Catherine properties on ≥ 25 metals + tmQM d-block OOD pool. In flight Q4 2026.

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU-only + GPU-0 for 25-metal × 1000-sample sweep |
| Wall-time | 2 weeks CPU + 1 week GPU-0 |
| Output | `molmetal/reports/wf44_r2_debug/closure_ood_phase2.json` (per-domain reachability + 6/6 Catherine) |
| Tests | ~15 new tests: per-domain OOD + per-metal reachability + 6 Catherine properties |

## §2 Cross-skill strategy

| Sub-task | Primary skill | Secondary skill |
|---|---|---|
| Tanimoto<0.4 OOD protocol | mol-metal-lambda-calculus | mol-metal-r18-math (T1) |
| 6 Catherine properties | mol-metal-r18-math (T7 SB-Bass) | mol-metal-lambda-calculus |
| Per-domain reachability | mol-metal-lambda-calculus | mol-metal-mechanism-mapping |

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **Wf-44 P2.1** Tanimoto<0.4 OOD protocol on tmQM d-block | 1 wk CPU | `molmetal/reports/wf44_r2_debug/ood_tanimoto_protocol.json` | Tanimoto<0.4 confirmed |
| **Wf-44 P2.2** Per-domain closure reachability | 1 wk CPU | `closure_ood_phase2.json` (PlatinAI 5K + MetalCytoToxDB + tmQM d-block) | per-domain reachability ≥ 0.85 |
| **Wf-44 P2.3** 6 Catherine properties | 1 wk GPU-0 | `catherine_6_props.json` (≥ 25 metals × 6 properties) | 6/6 PASS on ≥ 25 metals |
| **Wf-44 P2.4** Wf-44 P2 close-out | 2 days CPU | `TODO/completed/98_wf44_closure_ood_phase2.md` | INDEX §17 cross-ref |

## §4 MEASURED-vs-PROJECTED

按 party-twenty-spirit §3.5:
- MEASURED = Wf-44 P2 GPU/CPU runs + per-domain reachability JSON
- PROJECTED = reachability ≥ 0.85 + 6/6 Catherine on ≥ 25 metals

## §5 Cross-references

- `TODO/completed/92_wf41_phase2_closure_ood.md` (Wf-41 P2 L12 Phase 1 catalog)
- `TODO/completed/87_r2_debug.md` (Wf-44 8 root causes)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L12 trajectory column)
- `TODO/pending/96_r20_p5_closure_catherine.md` (R20-P5 closure + Catherine 6/6)
- `TODO/pending/99_r20_skill_ecosystem.md` (4 NEW R20 skills: `closure-theorem`)
