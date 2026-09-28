# TODO-127 — Wf-127 L5 Hilbert+ASTSR diagnostic close-out

**Date**: 2026-09-27
**Workflow**: Wf-127 (Hilbert+ASTSR parameter-sweep diagnostic + L2 guard — task IDs `wf127-p1`, `wf127-p2`)
**Status**: ✅ **P1 SHIP** (sweep cells MEASURED on synthetic; root cause identified); **✅ P2 SHIP** (L2 weight-decay guard patched + 15/15 tests PASS); **🔄 P3 in-flight** (joint sweep on PlatinAI 4-cell — depends on Wf-56 R20-P2 GPU retrain)
**Owner**: Hugo Lee + ultracode agents
**Parent**:
- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (R20-P2 Hilbert+ASTSR joint retrain — the GPU owner of any lift above the architectural ceiling)
- `TODO/completed/104_wf83_phase1_hilbert_astr_phase2_4cell.md` (Wf-83 P1 combined R² = -6.71 honest anchor on PlatinAI 4-cell)
- `TODO/completed/122_hilbert_l2_astsr_guard_closeout.md` (Wf-122 P1 L2 ridge + P2 ASTR scale-guard — the Wf-127 P2 L2-weight extension operates on top of this)
- `TODO/completed/87_r2_debug.md` (Wf-44 5-phase R² debug — 8 root causes)
- `TODO/completed/91_wf49_r2_root_fix.md` Cause 2 (12-d RDKit signal floor is the architectural ceiling)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 (R20 R² ≥ 0.85 target)
**Constraint**: CPU-only diagnostic + CPU+GPU-1 numerical-stability fix; paper-gate honored (no paper work touched); **honest framing per `party-twenty-spirit §3.5`**.
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` (M25 Hilbert PD) + `mol-metal-hilbert-astsr-e2e`.

---

## §0 TL;DR

Wf-127 ships a **two-pronged CPU diagnostic + numerical-stability fix** for the Wf-83 P1 combined R² = -6.71 honest anchor on PlatinAI 4-cell: **(P1) parameter-sweep diagnostic** identifying the stable region on synthetic, and **(P2) L2 weight-decay guard** on the proximal primal-dual step.

| Phase | Deliverable | Status |
|---|---|---|
| **P1** | (L2 weight, ASTR loss_scale, init mode) parameter sweep on synthetic (n=800, dim=12, snr=1.0) — stable region identified | ✅ **SHIP** |
| **P2** | PrimalDualOptimizer `l2_weight` proximal shrinkage on BOTH primal AND dual variables + composite default + 15 tests | ✅ **SHIP** |
| **P3** | Joint sweep across (l2_weight, l2_lambda, loss_scale, init_mode) with per-fold validation on PlatinAI 4-cell real pIC50 | 🔄 **IN-FLIGHT** (depends on Wf-56 R20-P2 GPU retrain) |

**Root cause (party-twenty-spirit §3.5 honest framing)**: the Wf-83 P1 combined R² = -6.71 is a **12-d RDKit signal floor 架构性** — the F5_RDKIT_2D descriptor does NOT contain coordination-chemistry-aware features. The Wf-127 P1 sweep guard + Wf-127 P2 L2 weight-decay are **defensive stabilizers** (prevent divergence / explosion); they do NOT lift R² above the structural ceiling. **The architectural lever for lift is Multi-PH + PLL feature augmentation** (Wf-79 P2 / Wf-94 R20-P3 SHIP, the modules that add 32-bin Fourier + 128-d PLL signature).

---

## §1 Deliverables (on disk)

### §1.1 P1 — Parameter sweep diagnostic (ship)

| File | Status | Path |
|---|---|---|
| L2 sweep JSON | ✅ | `molmetal/reports/wf127_p1_diagnostics/l2_sweep.json` (6 l2_lambda values × 3 seeds × train/test R²) |
| ASTSR sweep JSON | ✅ | `molmetal/reports/wf127_p1_diagnostics/astr_sweep.json` (4 loss_scale values × 3 seeds × train/test R²) |
| Init sweep JSON | ✅ | `molmetal/reports/wf127_p1_diagnostics/init_sweep.json` (3 init modes × 3 seeds × train/test R²) |
| P1 synthesis.md | ✅ | `molmetal/reports/wf127_p1_diagnostics/synthesis.md` |
| Bench script | ✅ | `molmetal/scripts/bench_hilbert_astsr_sweep.py` (NEW, ~280 LOC) |
| Bench tests | ✅ | `molmetal/tests/test_hilbert_astsr_sweep.py` |

### §1.2 P2 — L2 weight-decay guard (ship)

| File | Status | Path |
|---|---|---|
| Optimizer patch | ✅ | `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` — `l2_weight` dataclass field + proximal shrinkage at every PD step |
| Composite integration | ✅ | `molmetal/molmetal_lam/reward/hilbert_astsr_integration.py` — `LambdaASTSRHilbert.l2_weight=0.01` default |
| L2 guard tests | ✅ | `molmetal/tests/test_hilbert_l2_guard.py` — **15/15 tests PASS** |
| Guard test log | ✅ | `molmetal/reports/wf127_p2_l2_guard/guard_test_log.json` |

### §1.3 P3 — Joint sweep on PlatinAI 4-cell (in-flight)

| Component | Status | Path |
|---|---|---|
| Joint sweep script | 🔄 in-flight | `molmetal/scripts/bench_hilbert_astsr_joint_sweep.py` (planned ~400 LOC) |
| Per-fold validation probe | 🔄 in-flight | uses `bench_hilbert_astr_phase2_platinai_4cell.py` (Wf-83 P1 script) as the engine |
| Joint sweep JSON | 🔄 in-flight | `molmetal/reports/wf127_p3_joint_sweep/per_cohort_r2.json` |

---

## §2 MEASURED (party-twenty-spirit §3.5)

### §2.1 P1 sweep — L2 weight axis (synthetic n=800, dim=12, snr=1.0)

| `l2_lambda` | test R² (mean ± std) | stable? |
|---:|---:|:---:|
| 0e+00 | +0.4888 ± 0.0196 | YES |
| 1e-04 | +0.4888 ± 0.0196 | YES |
| 1e-03 | +0.4888 ± 0.0196 | YES |
| 1e-02 | +0.4887 ± 0.0199 | YES |
| 1e-01 | +0.4860 ± 0.0221 | YES |
| 1e+00 | +0.3832 ± 0.0237 | YES (degraded but stable) |

### §2.2 P1 sweep — ASTSR loss_scale axis

| `loss_scale` | test R² (mean ± std) | stable? |
|---:|---:|:---:|
| 0.1 | +0.3309 ± 0.0222 | YES |
| 1.0 | +0.3309 ± 0.0222 | YES |
| 10.0 | **-4.5768 ± 0.7286** | **NO** |
| 100.0 | **-828.0711 ± 100.0330** | **NO** (explosion) |

### §2.3 P1 sweep — Init mode axis

| `init_mode` | test R² (mean ± std) | stable? |
|---:|---:|:---:|
| `zeros` | +0.4888 ± 0.0203 | YES |
| `random_normal` | +0.4887 ± 0.0199 | YES |
| `xavier` | +0.4716 ± 0.0122 | YES (slightly lower; 5–8 grad-clip events per seed) |

### §2.4 P1 stable region (synthetic R² ≥ 0)

- L2 stable: `[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]` (the full grid is stable; degradation at l2_lambda=1.0 is graceful, not collapse)
- ASTSR loss_scale stable: `[0.1, 1.0]` (loss_scale ≥ 10 collapses to negative-R² explosion mode)
- Init mode stable: `['zeros', 'random_normal', 'xavier']` (all 3 stable; `xavier` triggers occasional grad-clip but does not collapse)

### §2.5 P2 L2 weight-decay guard (synthetic n=800, dim=12, snr=1.0)

| `l2_weight` | val R² (mean ± std) | last L2 penalty (mean) | finite? |
|---:|---:|---:|---:|
| 0.0 | +0.2681 ± 0.0029 | 0.0 | YES |
| 0.001 | +0.2681 ± 0.0029 | 0.000125 | YES |
| 0.01 (default) | +0.2675 ± 0.0029 | 0.001240 | YES |
| 0.05 | +0.2650 ± 0.0029 | 0.006074 | YES |
| 0.1 | +0.2619 ± 0.0028 | 0.011845 | YES |
| 0.5 | +0.2388 ± 0.0023 | 0.048512 | YES |

**Phase-1 stable match**: `l2_weight=0.01` gives val R² ≈ 0.268, identical to the no-guard baseline within 0.001 — confirms the new default is conservative on synthetic. Stable region `[0.0, 0.5]` — all values produce finite R² and non-negative L2 penalty.

### §2.6 P2 guard test coverage (15/15 PASS)

| Category | Test IDs | Count |
|---|---|---|
| Validation | `test_primal_dual_optimizer_rejects_negative_l2_weight`, `test_lambda_astsr_hilbert_rejects_negative_l2_weight` | 2 |
| Default values | `test_optimizer_default_l2_weight_is_zero`, `test_lambda_astsr_hilbert_default_l2_weight_is_phase1_stable` | 2 |
| Backward compat | `test_l2_weight_zero_matches_baseline_iterates`, `test_optimizer_last_l2_penalty_zero_when_l2_weight_zero` | 2 |
| Active guard | `test_l2_weight_positive_shrinks_primal_and_dual`, `test_l2_guard_prevents_dual_divergence_on_stress_probe` | 2 |
| Invariants | `test_l2_penalty_is_nonnegative_across_iters`, `test_optimizer_surfaces_last_l2_penalty_after_step` | 2 |
| Logging | `test_l2_weight_history_records_penalty_at_every_iter`, `test_history_dict_records_l2_penalty_per_iter` | 2 |
| Integration | `test_composite_l2_weight_propagates_to_inner_optimizer`, `test_composite_pre_built_optimizer_keeps_its_l2_weight`, `test_composite_fit_returns_l2_weight_and_penalty` | 3 |

**Regression suite** (Wf-122 + R18 P1-4 ship tests stay green): `test_hilbert_l2_regularization.py` 16/16, `test_hilbert_primal_dual.py` 29/29, `test_hilbert_astsr_integration.py` 25/25, `test_astr_scale_guard.py` 14/14, `test_astsr_multiph_hilbert.py` 28/28 — **112/112 PASS** total.

---

## §3 NOT SHIP / deferred

| Item | Why deferred | Where |
|---|---|---|
| **P3 joint sweep on PlatinAI 4-cell** | Per-fold validation probe requires Wf-83 P1 10-seed × 5-fold × 4-cell-line bench (Wf-83 P1 SHIP, reusable) but joint sweep over (l2_weight, l2_lambda, loss_scale, init_mode) × 4 cells × 5 folds is 4×4×4×3×4×5 = 3840 cells — CPU-only runtime exceeds 1 day; GPU retrain via Wf-56 R20-P2 is the proper venue | TODO-93 P2.1 |
| **Wf-122 P2 adaptive L2 schedule** | Finer doubling (1.5x instead of 2x) on top of the new l2_weight — engages the two regularisers jointly on weak-signal regimes | TODO-122 P2 follow-on |
| **Real-data lift above R² = -6.71** | The 12-d RDKit signal floor is architectural; the Wf-127 P1 sweep + Wf-127 P2 L2 weight are defensive stabilizers that **do not lift the ceiling**. The lift lever is **Multi-PH + PLL feature augmentation (Wf-79 P2 / Wf-94 R20-P3 SHIP)** — these add 32-bin Fourier + 128-d PLL signature that the 12-d descriptor lacks | TODO-94 P3.x |
| **Wf-128 / Wf-129 reuse of sweep recommendations** | The P1 stable weight recommendations (`l2_lambda ≤ 0.01`, `loss_scale ≤ 1.0`, all 3 init modes OK) are NOT YET consumed by downstream workflows (Wf-128 closure calibration + Wf-129 variance MCTS) — reuse is a follow-on | TODO-128 / TODO-129 P3 |

---

## §4 Cross-references

- **TODO-93** (R20-P2 Hilbert+ASTSR GPU retrain — parent plan, the GPU owner of any real-data lift) → `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` §3 (P2.1 Hilbert-ASTSR 5-seed × 50-epoch on PlatinAI 4-cell, R² ≥ 0.85 PROJECTED)
- **Wf-122 P1 L2 ridge + P2 ASTR scale-guard** (the prior defensive guard layer that Wf-127 P2 builds on) → `TODO/completed/122_hilbert_l2_astsr_guard_closeout.md` §1.1 (P1 L2 patch) + §1.2 (P2 ASTR guard)
- **Wf-83 P1 honest anchor** (combined R² = -6.71 on PlatinAI 4-cell — the structural ceiling this sweep diagnoses) → `TODO/completed/104_wf83_phase1_hilbert_astr_phase2_4cell.md` §0 (combined R² -6.71 MEASURED)
- **Wf-49 Cause 2** (12-d RDKit signal floor — the architectural ceiling identified here) → `TODO/completed/91_wf49_r2_root_fix.md` Cause 2
- **Wf-79 P2 / Wf-94 R20-P3** (Multi-PH + PLL feature augmentation — the architectural lift lever) → `TODO/pending/94_r20_p3_multiph_pll_feature.md` §3 (P3.1 32-bin Fourier + P3.2 128-d PLL signature)
- **Wf-128** (closure calibration — downstream consumer of Wf-127 P1 recommendations) → `TODO/pending/128_pt_vdw_calibration.md`
- **Wf-129** (variance MCTS — downstream consumer of Wf-127 P1 recommendations + Wf-123 M1 Wasserstein) → `TODO/pending/129_cisplatin_covalent_dock_followup.md` and `TODO/completed/129_l2_l7_variance_wasserstein_closeout.md`
- **LAMBDA_ROADMAP §1.3** (R20 R² ≥ 0.85 target — the ship bar that the joint sweep + GPU retrain target) → `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3
- **Wf-44 numerical fix** (grad_clip + step_decay + fp64 — the prior divergence-prevention layer) → `TODO/completed/87_r2_debug.md`

---

## §5 Honest framing (party-twenty-spirit §3.5)

- **MEASURED today**: P1 sweep cells (all axes) MEASURED on synthetic (n=800, dim=12, snr=1.0) with stable region identified; P2 L2 weight-decay guard patched + 15/15 tests PASS + 112/112 regression suite green.
- **NOT MEASURED today**: real-data lift on PlatinAI 4-cell (depends on Wf-56 R20-P2 GPU retrain, scheduled for RTX 5090 ~4 weeks); Wf-128 / Wf-129 downstream reuse of sweep recommendations (deferred).
- **Architectural ceiling (honest cause)**: the Wf-83 P1 combined R² = -6.71 on PlatinAI 4-cell is a **12-d RDKit signal floor 架构性** — the F5_RDKIT_2D descriptor does NOT contain coordination-chemistry-aware features. The Wf-127 P1 sweep + Wf-127 P2 L2 weight-decay guard are **defensive stabilizers** that prevent divergence / explosion modes; they do **NOT** lift R² above the structural ceiling.
- **The real lift lever**: **Multi-PH + PLL feature augmentation** (Wf-79 P2 / Wf-94 R20-P3 SHIP modules). These add 32-bin Fourier + 128-d PLL signature — the coordination-chemistry-aware features the 12-d descriptor lacks. Wf-94 R20-P3 is queued for GPU retrain on RTX 5090.
- **Wf-127 P3 honest deferral**: the joint sweep across (l2_weight, l2_lambda, loss_scale, init_mode) × 4 cells × 5 folds is the proper venue for finding the real-data optimum, but the cell×fold×seed combinatorial is 3840 cells — CPU-only runtime exceeds 1 day. GPU retrain via Wf-56 R20-P2 (TODO-93 P2.1) is the honest scheduling choice.
- **Backward compat**: Wf-127 P2 `l2_weight=0` reproduces the Wf-122 P1 path exactly (verified by `test_l2_weight_zero_matches_baseline_iterates` — dual-seed deterministic match within 1e-12). The default `l2_weight=0.01` is the P1 stable weight on synthetic, NOT claimed as optimal on PlatinAI 4-cell (Wf-127 P3 territory).
- **Net effect**: Wf-127 ships the **CPU-only sweep diagnostic + dual-side L2 weight-decay guard**; lift above R² = -6.71 requires Wf-94 R20-P3 Multi-PH+PLL augmentation on RTX 5090 GPU. The two follow-on consumers (Wf-128 + Wf-129) inherit the P1 stable region as their starting prior.

**Verdict**: ✅ **HONEST PARTIAL SHIP** — P1+P2 ship per spec with sweep + guard; P3 deferred to GPU retrain + joint sweep; root cause identified as architectural 12-d RDKit signal floor; honest pointer to the Wf-79/Wf-94 Multi-PH+PLL lever; party-twenty-spirit §3.5 honored.
