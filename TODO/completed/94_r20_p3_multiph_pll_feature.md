# TODO-94 — R20-P3 Score-Rep MC + Multi-PH Fourier feature extraction

**Date**: 2026-09-26
**Status**: 🚀 **Active — queued for R20 (GPU wall-clock, ~3 weeks GPU-1)**
**Priority**: **HIGH** (L7 Multi-PH + PLL feature extraction + Score-Rep MC variance reduction)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 + `TODO/completed/83j_lambda_closeout.md` §5.1
**Lambda gap(s)**: L7 (Multi-PH + PLL end-to-end)
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` (T5 Multi-PH, M26 Score-Rep)

---

## §0 Why this plan exists

Wf-37 P3 SHIPS Multi-PH + PLL end-to-end at CPU bench layer with honest-negative: `mean R² +0.087` (vs target +0.30) on n_target=200 + 73-d feature finite + seeds 3/4 NOT run. Wf-25 R² instability fixed by Wf-44 numerical fix. **R20-P3 completes the seeds 3/4 + n_target ≥ 1000** to lift R² ≥ 0.70. Additionally integrates **Score-Rep MC variance reduction** (Burch-Bardsley 2022 + Hu et al. 2026 ICML) into the Multi-PH pipeline for variance-reduced feature extraction.

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | RTX 5090 GPU-1 (32 GB) |
| Wall-time | 3 weeks GPU-1 |
| Output | `molmetal/reports/wf52_r20_p3/{multiph_pll_r2_extended,score_rep_mc_variance}.json` |
| Tests | ~20 new tests: per-seed R² + Score-Rep MC convergence + per-cohort breakdown |

## §2 Cross-skill strategy

| Sub-task | Primary skill | Secondary skill |
|---|---|---|
| Multi-PH Fourier 32-bin | mol-metal-r18-math (T5) | mol-metal-lambda-calculus |
| PLL signature 128-d | mol-metal-lambda-calculus | mol-metal-r18-math (M26) |
| Score-Rep MC variance | mol-metal-lambda-calculus | mol-metal-r18-math (M26) |

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P3.1** Multi-PH Fourier 32-bin feature | 1 wk GPU-1 | `molmetal/reports/wf52_r20_p3/multiph_fourier.json` | per-cohort Pearson r |
| **P3.2** PLL signature 128-d feature | 1 wk GPU-1 | `pll_signature_128d.json` | per-cohort Pearson r |
| **P3.3** Score-Rep MC variance reduction | 1 wk GPU-1 | `score_rep_mc_variance.json` (variance_reduction ≥ 0.2 on PlatinAI 4-cell) | convergence test |
| **P3.4** Wf-52 close-out | 2 days CPU | `TODO/completed/94_r20_p3_multiph_pll_feature.md` | INDEX §17 cross-ref |

## §4 MEASURED-vs-PROJECTED

按 party-twenty-spirit §3.5:
- MEASURED = Wf-52 GPU runs + per-cohort R² JSON
- PROJECTED = R² ≥ 0.70 on PlatinAI 4-cell + variance_reduction ≥ 0.2

## §5 Cross-references

- `TODO/completed/83j_lambda_closeout.md` §3.2 (L7 honest-negative)
- `TODO/completed/83b_lambda_triangle.md` (Wf-37 L5-L8 close)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L7 trajectory column)
- `TODO/pending/99_r20_skill_ecosystem.md` (4 NEW R20 skills: `multiph-pll`, `score-rep-mc`)