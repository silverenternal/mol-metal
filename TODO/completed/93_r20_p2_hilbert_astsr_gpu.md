# TODO-93 — R20-P2 Hilbert ASTSR GPU joint retrain

**Date**: 2026-09-26
**Status**: 🚀 **Active — queued for R20 (GPU wall-clock, ~4 weeks GPU-0)**
**Priority**: **CRITICAL** (L4+L5+L7+L8 joint retrain; unblocks paper §3.6 R² ≥ 0.85 claim)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 + `TODO/completed/83j_lambda_closeout.md` §5.1
**Lambda gap(s)**: L4 (Replay learner) + L5 (Hilbert-ASTSR e2e) + L7 (Multi-PH + PLL) + L8 (Triangle closure) + L11 (MCTS tree stats)
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` (T5 Multi-PH, P1-4 ML methods)

---

## §0 Why this plan exists

Wf-37 P1 + Wf-39 P2 + Wf-41 P1 SHIP Hilbert-ASTSR end-to-end at CPU bench layer: `Hilbert R²=+0.1631 ± 0.006` (vs Wf-25 -0.443) + composite R²=+0.186 + **3-way R² +0.7524 lift** on synthetic `qed*qed` target. Wf-44 numerical fix (grad_clip + step_decay + fp64) prevents collapse but does not lift R² ≥ 0 in absolute terms. Wf-49 Phase 2 swapped ASTSR features to F5_RDKIT_2D (real signal |ρ| up to +0.204 on MetalCytoTox). **R20-P2 retrains Hilbert+ASTSR on PlatinAI 4-cell labelled pIC50 with RDKit 2D features at GPU scale (5-seed × 50-epoch × dual-init {hilbert, tmqm})**, targeting R² ≥ 0.85 on real pIC50.

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | RTX 5090 GPU-0 (32 GB free) — Hilbert+ASTSR + Multi-PH joint retrain |
| Wall-time | 4 weeks GPU-0 (≈ 8 days compute + 2 weeks analysis) |
| Output | `molmetal/reports/wf51_r20_p2/{hilbert_astsr_r2,multiph_pll_r2,triangle_closure_r2}.json` |
| Tests | ~30 new tests: per-cohort R² + per-cell R² + per-fold R² + ablation table |

## §2 Cross-skill strategy

| Sub-task | Primary skill | Secondary skill |
|---|---|---|
| Hilbert-ASTSR retrain | mol-metal-lambda-calculus | mol-metal-r18-math (T5 Multi-PH) |
| Multi-PH + PLL feature extraction | mol-metal-lambda-calculus | mol-metal-cfm-training |
| Triangle closure end-to-end | mol-metal-lambda-calculus | mol-metal-r18-math (M26 Score-Rep) |
| MCTS tree vuln_f+g retrain | mol-metal-lambda-calculus | mol-metal-mechanism-mapping |

## §3 Phases (per LAMBDA_ROADMAP §1.2)

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P2.1** Hilbert-ASTSR 5-seed × 50-epoch on PlatinAI 4-cell | 1 wk GPU-0 | `molmetal/reports/wf51_r20_p2/hilbert_astsr_r2.json` (per-cell R² + 5-seed mean ± std) | R² ≥ 0.85 |
| **P2.2** Multi-PH + PLL on PlatinAI 4-cell + MetalCytoToxDB 5-metal | 1 wk GPU-0 | `multiph_pll_r2.json` (32-bin Fourier + 128-d PLL signature) | R² ≥ 0.70 |
| **P2.3** Triangle closure real pIC50 | 1 wk GPU-0 | `triangle_closure_r2.json` (Hilbert R² + evolutionary R² + variance reduction) | mean_hilbert_r2 ≥ 0.6 on real data |
| **P2.4** MCTS tree vuln_f+g retrain (full replay + variance penalty) | 1 wk GPU-0 | `molmetal/reports/wf51_r20_p2/mcts_tree_stats.json` | hit_rate ≥ 0.95 |
| **P2.5** Wf-51 close-out | 2 days CPU | `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` | INDEX §17 cross-ref update |

## §4 MEASURED-vs-PROJECTED

按 party-twenty-spirit §3.5:
- MEASURED = Wf-51 GPU runs on disk + per-cohort R² JSON
- PROJECTED = R² ≥ 0.85 on PlatinAI 4-cell per-cell breakdown

## §5 Cross-references

- `TODO/completed/83j_lambda_closeout.md` §3.2 (L4/L5/L7/L10/L11 honest-negative)
- `TODO/completed/87_r2_debug.md` (Wf-44 numerical fix + 8 root causes)
- `TODO/completed/91_wf49_r2_root_fix.md` (Wf-49 Phase 1+2 ship; F5 → F5_RDKIT_2D)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L4/L5/L7/L8/L11 trajectory column)
- `TODO/pending/99_r20_skill_ecosystem.md` (4 NEW R20 skills: `hilbert-astsr-e2e`, `multiph-pll`, `triangle-closure`, `mcts-replay-learner`)
