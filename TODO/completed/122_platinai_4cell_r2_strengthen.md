# TODO-122 — PlatinAI 4-cell R² strengthen via feature stack + multi-task + Wasserstein

**Date**: 2026-09-27
**Status**: 🚀 **Active — closes PlatinAI 4-cell R² = -6.71 (below constant baseline)**
**Priority**: **P0 CRITICAL** (the strongest honest-negative in the system — L4/L5/L7/L8/L11 central claim)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Honest model-state assessment (2026-09-27) + Wf-85 root cause analysis (2026-09-27)
**Lambda gap(s)**: L4 (Hilbert primal-dual) + L5 (ASTSR) + L7 (Multi-PH) + L8 (Coupling)
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` + `mol-metal-cfm-training`

---

## §0 Why this plan exists

**Current state** (Wf-83 P1 / Wf-85 P3):
- PlatinAI 4-cell combined R² = **−6.71** (5-fold random CV, 10-seed) — Agent 2 report
- Wf-85 P3 strict-scaffold R² = **−1.50** (5-seed)
- Both are **worse than predicting y.mean() baseline**
- 12-d RDKit features have |ρ| < 0.02 on PlatinAI pIC50 (signal floor)

**Why**:
- **Data signal floor**: PlatinAI pIC50 std = 0.45, all 12-d features have near-zero Pearson
- **Hilbert primal-dual diverges** (no L2 regularization, plain MSE on a feature floor)
- **ASTSR collapses to constant** (mode A, R² ≈ −0.014) or **explodes** (mode B, R² = −8 to −27 on OOD)
- Scoring/pipeline/splits are OK — bug is upstream

**Wf-122 in flight** addresses L2 + scale guard (P1.1 + P1.2). This TODO-122 adds **3 more innovations** to stack on top.

## §1 Scope

| Item | Detail |
|---|---|
| Hardware | CPU + GPU-0 Blackwell |
| Wall-time | 1 week GPU-0 |
| Output | feature stack + multi-task + Wasserstein regularization |
| Tests | ~30 new tests |

## §2 Solution path (3 innovations stacked on Wf-122 fixes)

1. **Multi-PH + PLL feature augmentation** (already shipped Wf-66 R20-P3)
   - 12-d RDKit → 12 + 32 Multi-PH Fourier + 128 PLL signature = **172-d feature stack**
   - Use as input to Hilbert + ASTSR branches
2. **Multi-task learning across 4 cell lines** (MCF7 / A549 / A2780 / A2780cis)
   - Shared backbone + per-cell-line heads
   - Per-cell-line signal boosts via cross-cell-line transfer
3. **Wasserstein gradient flow regularization** (theorem T1)
   - Penalize off-manifold pIC50 predictions
   - Force Hilbert-ASTR flow to stay on the chemistry manifold

## §3 Phases

| Phase | Wall | Output | Verification |
|---|---|---|---|
| **P1.1** 172-d feature stack (RDKit + Multi-PH + PLL) | 1 day GPU-0 | `molmetal/scripts/bench_hilbert_astr_172d.py` (NEW) + 8 tests | per-cohort R² with 172-d stack |
| **P1.2** Multi-task 4-cell backbone | 2 days GPU-0 | `molmetal/scripts/bench_multitask_4cell.py` (NEW) + 8 tests | per-cell-line R² + cross-cell-line transfer |
| **P1.3** Wasserstein regularization on pIC50 manifold | 1 day GPU-0 | `molmetal/molmetal_lam/reward/wasserstein_pic50.py` (NEW) + 6 tests | Wasserstein distance decay + R² lift |
| **P1.4** Full re-bench with all 3 innovations stacked + Wf-122 L2/guard | 1 day GPU-0 | `molmetal/reports/wf122_platinai_full_stack/` | combined R² per cell line ≥ 0 for at least 1 of 4 |

## §4 MEASURED-vs-PROJECTED

Per party-twenty-spirit §3.5:
- **MEASURED**: per-feature-stack R² + per-cell-line multi-task R² + Wasserstein penalty
- **PROJECTED**: combined R² ≥ 0 for at least 1 of 4 cell lines (vs current all 4 negative)

## §5 Honest framing

This is the **biggest honest-negative** in the system. Stacking **3 innovations** on top of Wf-122 L2/guard may lift R² to 0 for at least 1 cell line, but **the structural 12-d signal floor may not be breakable** with any amount of regularization. The honest path is to ship all 3 innovations, report per-cell-line results, and accept that the L4/L5/L7/L8/L11 central Lambda claim may need a different feature regime (e.g., reaction-product features from the 5-click productive space, not raw molecular descriptors).

## §6 Cross-references

- `metrics/by_metric/hilbert_4cell_r2.json` (current -6.71)
- `metrics/by_metric/platinai_combined_r2.json`
- Agent 2 Wf-85 root cause report (2026-09-27)
- TODO-111 / Wf-122 (Hilbert L2 + ASTSR scale guard in flight)
- TODO-94 / Wf-66 (Multi-PH + PLL feature augmentation shipped)
- `molmetal/math_skills/wasserstein_gradient_flow.py` (theorem)
- TODO-119 / TODO-120 / TODO-121 (Wasserstein regularization reuse)
- TODO-84 paper rule (paper §3.4 + §4 not blocked)
- LAMBDA_ROADMAP §4.2 L4/L5/L7/L8/L11 columns
