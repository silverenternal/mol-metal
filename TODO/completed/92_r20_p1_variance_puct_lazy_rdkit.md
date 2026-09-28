# TODO-92 — R20-P1 VariancePUCT default + Lazy RDKit import

**Date**: 2026-09-26
**Status**: 🚀 **Active — queued for R20 (GPU wall-clock, ~2-3 wk)**
**Priority**: **HIGH** (L2 + vuln_a fix; unblocks real-data Lambda-MCTS trajectory benchmarks)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 + `TODO/completed/83j_lambda_closeout.md` §5.1
**Lambda gap(s)**: L2 (Variance MCTS ↔ proof_search)
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` (M26 Score-Rep)

---

## §0 Why this plan exists

Wf-36 Phase 2 + Wf-39 Phase 1 + Wf-46 Phase 4 (integration T4) SHIP VarianceMCTSWrapper + auto-built selector. **L2 ships in bench layer with honest-negative framing**: `variance_reduction -0.185` on mock CCO + `vuln_a CRITICAL` (RDKit missing on snapshot). The VariancePUCT scoring sign-flip + RDKit-import order are the two blockers for **real-data** trajectory benchmarks on PlatinAI 5K + MetalCytoToxDB 5-metal. R20-P1 fixes both.

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | Blackwell sm_120 + RTX 5090 dual-GPU |
| Wall-time | 2-3 weeks GPU-0 |
| Output | `molmetal/reports/wf50_r20_p1/{trajectory,variance_sign,rdkit_lazy_import}.json` |
| Tests | ~15 new tests: sign-flip regression + lazy import order + real-data trajectories |

## §2 Cross-skill strategy

| Sub-task | Primary skill | Secondary skill |
|---|---|---|
| VariancePUCT sign-bug fix | mol-metal-lambda-calculus | mol-metal-r18-math (M26) |
| Lazy RDKit import | mol-metal-lambda-calculus | mol-metal-cfm-training |

## §3 Phases (per LAMBDA_ROADMAP §1.2)

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** VariancePUCT scoring rule audit | 3 days CPU | `molmetal/molmetal_lam/search_alg/variance_puct_integration.py` patched; 5 new tests | pytest PASS + sign-flip regression test |
| **P1.2** Lazy RDKit import (deferred) | 1 week CPU | `molmetal/molmetal_lam/reward/learned_shaping.py` + lambda_combinators.py patched; 10 new tests | pytest PASS + import-order test |
| **P1.3** Real-data trajectory benchmark | 1 week GPU-0 | `molmetal/reports/wf50_r20_p1/trajectory_{platinai_5k,metalcytotox}.json` (per-rule score + per-arm visits) | variance_reduction ≥ 0 on real data |
| **P1.4** Wf-50 close-out | 2 days CPU | `TODO/completed/92_r20_p1_variance_puct_lazy_rdkit.md` (≤ 250 lines) | INDEX §17 cross-ref update |

## §4 MEASURED-vs-PROJECTED

按 party-twenty-spirit §3.5:
- MEASURED = Wf-50 GPU runs on disk + trajectory JSON
- PROJECTED = scripts ship + CPU smoke passes + variance_reduction ≥ 0 expected

## §5 Cross-references

- `TODO/completed/83j_lambda_closeout.md` §3.1 (L2 bench layer ship)
- `TODO/completed/85_wf39_mcts_real_trajectory.md` (Wf-39 P1 vuln_a CRITICAL)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L2 trajectory column)
- `TODO/pending/99_r20_skill_ecosystem.md` (4 NEW R20 skills includes `mol-metal-variance-puct`)