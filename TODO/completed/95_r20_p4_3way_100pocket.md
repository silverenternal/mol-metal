# TODO-95 — R20-P4 3-way on MetalCytoToxDB + 100-pocket × 10-seed

**Date**: 2026-09-26
**Status**: 🚀 **Active — queued for R20 (GPU wall-clock, ~4 weeks GPU-1)**
**Priority**: **HIGH** (L6 Closure d-block + L10 3-way on real pIC50 + G7 10-pockets × 10-seed expansion)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 + `TODO/completed/83j_lambda_closeout.md` §5.1
**Lambda gap(s)**: L6 (Closure d-block) + L10 (3-way integration on real pIC50)
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-cfm-training` + `mol-metal-click-chemistry`

---

## §0 Why this plan exists

Wf-41 P1 SHIP 3-way integration (Deflexpressor × Multi-PH × ASTSR) at CPU bench layer: `3way R²=-0.3595 ± 0.2806` on PlatinAI 5K synthetic `qed*qed` + honest `+0.7524` lift over AST baseline. Wf-37 P2 SHIP Closure d-block: `reachability 0.5891` over 2244 samples × 30 metals. Wf-30 SHIP 10/10 pocket staging but only 2/10 actually docked (Vina binary MISSING_DEP). **R20-P4 retrains 3-way on MetalCytoToxDB 5-metal real pIC50** (target R² ≥ +0.30), and **expands Wf-30 10/10 to 100/100 pockets × 10 seeds** (CrossDocked-2024 + PDBBind-2025).

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | RTX 5090 GPU-1 (32 GB) + Blackwell sm_120 GPU-0 for 100-pocket sweep |
| Wall-time | 4 weeks GPU-1 (3-way retrain) + 2 weeks GPU-0 (100-pocket sweep) |
| Output | `molmetal/reports/wf53_r20_p4/{3way_real_r2,100pocket_dock_results}.json` |
| Tests | ~30 new tests: per-metal R² + per-pocket dock score + per-cand ranking |

## §2 Cross-skill strategy

| Sub-task | Primary skill | Secondary skill |
|---|---|---|
| 3-way on MetalCytoToxDB | mol-metal-lambda-calculus | mol-metal-r18-math (T5 Multi-PH) |
| 100-pocket docking sweep | mol-metal-cfm-training | mol-metal-click-chemistry |
| Closure d-block ≥ 0.95 | mol-metal-lambda-calculus | mol-metal-mechanism-mapping |

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P4.1** 3-way retrain on MetalCytoToxDB 5-metal | 2 wk GPU-1 | `molmetal/reports/wf53_r20_p4/3way_real_r2.json` (per-metal R² + 5-seed mean ± std) | R² ≥ +0.30 |
| **P4.2** Closure d-block reachability ≥ 0.95 | 1 wk CPU | `closure_dblock_r20.json` (30 metals × 1000 samples) | reachability ≥ 0.95 |
| **P4.3** 100-pocket × 10-seed docking sweep | 2 wk GPU-0 | `100pocket_dock_results.json` (per-pocket dock score + per-cand ranking) | full sweep exit 0 |
| **P4.4** Wf-53 close-out | 2 days CPU | `TODO/completed/95_r20_p4_3way_100pocket.md` | INDEX §17 cross-ref |

## §4 MEASURED-vs-PROJECTED

按 party-twenty-spirit §3.5:
- MEASURED = Wf-53 GPU runs + per-metal R² + 100-pocket dock JSON
- PROJECTED = R² ≥ +0.30 on MetalCytoToxDB + reachability ≥ 0.95 + 100-pocket sweep exit 0

## §5 Cross-references

- `TODO/completed/83j_lambda_closeout.md` §3.2 (L6/L10 honest-negative)
- `TODO/completed/83i_lambda_combined.md` (Wf-41 L10 + L12)
- `TODO/completed/82b_vina_dock_retrain.md` (Wf-30 honest-negative stub)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L6/L10 trajectory column)
- `TODO/pending/99_r20_skill_ecosystem.md` (4 NEW R20 skills: `3way-integration`, `closure-dblock-30`)
