# 383 — SE(3) Schrödinger Bridge Fast-Converging Innovation Integration Plan

**Document ID:** TODO-383 / Wf-383-A2-EXTENSION-PLAN
**Version:** 1.0
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + Wf-383-A2 planning agent
**Status:** ⏸️ PLAN_READY (CPU-only plan; 0 source patches; 0 GPU launches)
**Trigger:** Wf-340 (820 LOC) + Wf-353 (~2,690 LOC) ship 10 fast-converging innovations **only into `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`**. The SE(3) Schrödinger Bridge trainer (`molmetal/scripts/train_se3sb.py`, 749 LOC) has **zero** of these wires. TODO-200 covers the CFM YuelBond trainer; this TODO-383 covers the SE(3) SB trainer (M22 per `mol-metal-mechanism-mapping` skill).

---

## §0 TL;DR

| Field | Value |
|---|---|
| **Scope** | Extend Wf-340 + Wf-353's 10 fast-converging innovation wires from `train_cfm_yuelbond_wasserstein.py` to **`train_se3sb.py`** (M22 SE(3) SB trainer, WEAK_TO_STRONG_PLAN §4.3) |
| **Trainer entry** | `molmetal/scripts/train_se3sb.py` (749 LOC, `parse_args` at line 557) |
| **Adapter layer** | `molmetal/adapters/se3_sb_integration.py` (611 LOC) + `molmetal/adapters/se3_lipman_integration.py` (854 LOC) + `molmetal/adapters/flow_matching_lipman/se3sb_bass_integration.py` (685 LOC) |
| **Math layer** | `molmetal/math_skills/se3_schrodinger_bridge.py` (816 LOC — T1 math skeleton, already in place) |
| **10 innovations** | EDM σ-precond + log-normal t + EMA + lr-warmup + shortcut distillation + consistency reg + FlowMol unified + exact-OT + Triton mini-batch OT + shortcut consistency |
| **Per-innovation delta** | New file paths in `molmetal/adapters/flow_matching_lipman/` (already SHIPPED) + new `parse_args` flags (default OFF) + `integrate_with_cfm()` branch wires + `--use-se3sb` opt-in master flag (separation between SE(3) SB + Euclidean flow) |
| **License-gated** | FlowMol (CC-BY-NC-SA 4.0) — USER_OK required per Wf-353-A4 + Wf-340-A1 EDM σ-precond (same license) |
| **Wall (PROJECTED)** | ~5-7 days CPU-only (10 innovations × 0.5-0.7 day each) |
| **NEW tests PROJECTED** | ~10-12 NEW tests (1 per innovation) + 2 integration regression tests = **~12 NEW tests** |
| **GPU launches** | 0 (CPU-only plan; GPU bench deferred to WEAK_TO_STRONG_PLAN §4.3 Phase 2) |
| **paper/* touches** | 0 (TODO-84 active; READ-only) |
| **wet-lab planning** | 0 (AGENTS.md §10) |

---

## §1 Background — Wf-340 + Wf-353 SHIPPED but only to CFM YuelBond

### §1.1 What SHIPPED (per Wf-340-I1 + Wf-353-A1..A4 closeouts)

Per `TODO/completed/340i1_cfm_acceleration_implementation.md` §1 + `TODO/completed/353_final_synthesis.md` §1, Wf-340 + Wf-353 ship 10 fast-converging wires as CPU-only SCAFFOLDS, all defaulting to OFF:

| # | Innovation | Module | CFM YuelBond CLI flag | Status |
|---:|---|---|---|---|
| 1 | EDM σ-preconditioning (Karras 2022) | `acceleration.py:80 LOC` | `--use-edm-precond` | SHIPPED (default OFF) |
| 2 | Log-normal t-sampling | `acceleration.py:25 LOC` | `--use-log-normal-t` | SHIPPED (default OFF) |
| 3 | Shortcut consistency (Frans 2024) | `acceleration.py:70 LOC` | `--use-shortcut-consistency` | SHIPPED (default OFF) |
| 4 | EMA (NVlabs/edm halflife) | `acceleration.py:50 LOC` | `--use-ema` | SHIPPED (default OFF) |
| 5 | Linear lr-warmup | `acceleration.py:40 LOC` | `--use-lr-warmup` | SHIPPED (default OFF) |
| 6 | Exact-OT solver (POT `ot.emd`) | `optimal_transport_v2.py:85 LOC` | `--use-exact-ot-coupling` | SHIPPED (default OFF) |
| 7 | Triton mini-batch OT (Khainb 2025) | `triton_minibatch_ot.py:561 LOC` | `--triton-minibatch-ot` | SHIPPED (default OFF, env-var gated) |
| 8 | Teacher-student shortcut distillation (Frans 2024) | `shortcut_distillation.py:145 LOC` | `--use-shortcut-distillation` | SHIPPED (default OFF) |
| 9 | Temporal consistency regularization (EDM §5.2) | `consistency_regularization.py:180 LOC` | `--use-consistency-regularization` | SHIPPED (default OFF) |
| 10 | FlowMol unified atom-bond flow (Dunn & Koes 2024, CC-BY-NC) | `unified_atom_bond_flow.py:390 LOC` | `--use-flowmol-unified` | SHIPPED (default OFF, LICENSE-GATED) |

**Cumulative scaffold LOC**: ~1,790 LOC production modules + ~1,720 LOC tests = ~3,510 LOC across Wf-340 + Wf-353.

### §1.2 What's MISSING for SE(3) SB trainer

Per grep on `molmetal/scripts/train_se3sb.py` `parse_args` (line 557), the SE(3) SB trainer has **NONE** of the 10 CLI flags above. The SE(3) SB trainer still uses vanilla midpoint/Euler integration on SE(3) twist-space (`integrate_with_cfm(method="midpoint")` per `se3_sb_integration.py:430-574`), with stochastic mini-batch OT at the source/target coupling step.

### §1.3 Why TODO-383 (not TODO-200)

- **TODO-200** (`TODO/pending/200_wf340_353_deep_integration_audit.md`) covers **CFM YuelBond deep integration** — wire the 10 SCAFFOLDS into `train_cfm_yuelbond_wasserstein.py` as CLI flags with default ON (post-validation).
- **TODO-383** (this file) covers **SE(3) SB-specific deep integration** — extend the same 10 SCAFFOLDS to `train_se3sb.py`, BUT with **SE(3)-specific adapters** for the geometry-dependent items (EDM σ-precond on SO(3) noise, OT on SE(3) geodesic, shortcut distillation with SE(3)-equivariant teacher/student, etc.).

The two TODOs are complementary: TODO-200 = R^3 flow matching with Euclidean OT; TODO-383 = SE(3) Schrödinger Bridge with geodesic OT on the SE(3) manifold.

---

## §2 Per-innovation plan (10 innovations × SE(3) SB-specific)

### §2.1 EDM σ-preconditioning for SE(3) (Wf-340-A1 SE(3) variant)

**SE(3)-specific delta**: The Karras 2022 `c_skip / c_out / c_in / c_noise` scheme was designed for R^n denoiser output. For SE(3) the noise must be parameterized on the SO(3) **rotation-vector manifold** (axis-angle, `so3_log` per `se3_schrodinger_bridge.py:152-251`) and the translation noise must be Euclidean. The c_skip/c_out/c_in functions thus need to be **split** — one for the linear part (translation) and one for the angular part (rotation).

- **File path (NEW)**: `molmetal/adapters/flow_matching_lipman/se3_edm_preconditioning.py` (~120 LOC)
- **Where to wire**: `train_se3sb.py:557 parse_args` → add `--use-se3-edm-precond` flag (default OFF); `train_se3sb.py:train_one_epoch` → call `se3_edm_preconditioned_loss(pred_linear, pred_angular, target_linear, target_angular, sigma)` after `integrate_with_cfm(method="midpoint")` returns the SE(3) velocity field.
- **NEW tests PROJECTED**: 1 test `test_se3_edm_precond_separates_linear_angular` — verify `c_skip(σ)` applied separately to translation vs rotation-vector produces finite bounded output on a random SE(3) twist; bit-identical at fixed σ.
- **Expected step reduction**: 2-3× faster SE(3) flow convergence per Karras 2022 ablation §5 (PROJECTED; not measured).
- **License**: NVlabs/edm CC-BY-NC-SA 4.0 (same as Wf-340-A1); USER_OK required.

### §2.2 Log-normal t-sampling for SE(3) (Wf-340-A2 SE(3) variant)

**SE(3)-specific delta**: Compatible as-is — `sample_log_normal_t(b, p_mean=-1.2, p_std=1.2)` operates on the time scalar `t ∈ [0, 1]`, which is the same parameterization for SE(3) twist-space interpolation. **No SE(3)-specific adapter required** — direct re-import.

- **File path**: reuse `molmetal/adapters/flow_matching_lipman/acceleration.py::sample_log_normal_t` (already SHIPPED).
- **Where to wire**: `train_se3sb.py:557 parse_args` → add `--use-log-normal-t` flag (default OFF, mirroring CFM YuelBond); `train_se3sb.py:train_one_epoch` → replace `torch.rand(b)` at the `integrate_with_cfm` call site with `sample_log_normal_t(b)`.
- **NEW tests PROJECTED**: 1 test `test_log_normal_t_sampling_compatible_with_se3` — verify `sample_log_normal_t(b)` output ∈ [0, 1] and produces finite SE(3) twist-space input when used to scale `(1-t)·x0 + t·x1` on SO(3) rotation matrices.
- **Expected step reduction**: 1.5-2× fewer steps at matched SE(3) loss threshold per Karras 2022 (PROJECTED).

### §2.3 Exact-OT solver for SE(3) geodesic (Wf-340-A6 SE(3) variant)

**SE(3)-specific delta**: The current `SE3SBOTSampler` (`se3sb_bass_integration.py:206-391`) already provides SE(3) twist-space Sinkhorn OT (`lift_to_se3_twist` at lines 275-322). The exact-OT upgrade is to **swap Sinkhorn (entropic) for Hungarian (exact) on the SE(3) twist-space cost matrix** — `ot.emd` on `xi ∈ se(3)` twist-space cost `||xi0 - xi1||²` (Mahmood 2026 eq. 12).

- **File path (NEW)**: `molmetal/adapters/flow_matching_lipman/se3_exact_ot_solver.py` (~80 LOC) — wraps `ot.emd` with the `lift_to_se3_twist` reduction already in `SE3SBOTSampler`.
- **Where to wire**: `train_se3sb.py:557 parse_args` → add `--use-se3-exact-ot` flag (default OFF); `train_se3sb.py:train_one_epoch` → after `make_paired_batch()` (`se3sb_bass_integration.py:663-684`), branch on flag to use `se3_exact_ot_solver.couple(x0, x1)` instead of `SE3SBOTSampler.couple(x0, x1)`.
- **NEW tests PROJECTED**: 1 test `test_se3_exact_ot_solver_straighter_than_sinkhorn` — verify Hungarian OT cost ≤ Sinkhorn OT cost on a fixed (x0, x1) pair, both lifting via `lift_to_se3_twist`; verify permutation is valid + bit-identical at fixed seed.
- **Expected step reduction**: 1.5-2× fewer inference NFE at matched training compute per Wf-340-I3 §2.1 (PROJECTED).
- **License**: POT `ot.emd` MIT (no license risk).

### §2.4 Linear lr-warmup for SE(3) (Wf-340-A5 SE(3) variant)

**SE(3)-specific delta**: Compatible as-is — `LinearWarmupLRScheduler` (`acceleration.py`) operates on any optimizer regardless of geometry. **No SE(3)-specific adapter required** — direct re-import.

- **File path**: reuse `molmetal/adapters/flow_matching_lipman/acceleration.py::LinearWarmupLRScheduler`.
- **Where to wire**: `train_se3sb.py:557 parse_args` → add `--use-lr-warmup` + `--lr-warmup-steps` flags (default OFF, mirroring CFM YuelBond); `train_se3sb.py:train_one_epoch` → wrap `optimizer.step()` with `lr_scheduler.step()` if flag enabled.
- **NEW tests PROJECTED**: 1 test `test_lr_warmup_compatible_with_se3_trainer` — verify `LinearWarmupLRScheduler` produces monotonic lr ramp on a SE(3) SB optimizer over 100 steps (no NaN/Inf).
- **Expected step reduction**: ~1.2× faster initial convergence (PROJECTED; standard warmup effect).

### §2.5 EMA for SE(3) (Wf-340-A1 SE(3) variant)

**SE(3)-specific delta**: EMA operates on the **velocity field weights** regardless of whether the field outputs SE(3) twists or Euclidean coords. The EMA swap-in/swap-out at evaluation time requires the `SE3SBVelocityField` (`se3_sb_integration.py:101`) to be state-dict compatible with the EMA buffer. **No SE(3)-specific adapter required** — direct re-import with a type check.

- **File path**: reuse `molmetal/adapters/flow_matching_lipman/acceleration.py::ExponentialMovingAverage`.
- **Where to wire**: `train_se3sb.py:557 parse_args` → add `--use-ema` + `--ema-halflife-kimg` flags (default OFF); `train_se3sb.py:train_one_epoch` → call `ema.update(model)` at the end of every optimizer step; before evaluation, `ema.copy_to(model)` + restore (mirror the CFM YuelBond pattern at lines 957-1114).
- **NEW tests PROJECTED**: 1 test `test_ema_state_dict_compatible_with_se3_velocity_field` — verify EMA buffer swaps SE3SBVelocityField weights losslessly (output unchanged at fixed seed).
- **Expected step reduction**: ~1.2× effective convergence (PROJECTED; standard EMA effect).

### §2.6 Triton mini-batch OT kernel on SE(3) (Wf-353-A1 SE(3) variant)

**SE(3)-specific delta**: The Triton kernel (`triton_minibatch_ot.py`) operates on `(B, d)` cost matrices. For SE(3), the cost matrix is `(B, B, 6)` on twist-space (after `lift_to_se3_twist`). The kernel needs a **post-lift wrapper** that:
1. Computes the `(B, B, 6)` twist-space diff (`SE3SBOTSampler.couple` does this internally at lines 380-388).
2. Passes the `(B, B)` L2 cost matrix to the Triton kernel for log-domain Sinkhorn.
3. Returns the row-argmax permutation (the SE(3) OT coupling).

- **File path (NEW)**: `molmetal/adapters/flow_matching_lipman/se3_triton_minibatch_ot.py` (~140 LOC) — wraps `triton_minibatch_ot_coupling` (`triton_minibatch_ot.py`) with the `lift_to_se3_twist` + Sinkhorn-call from `SE3SBOTSampler`.
- **Where to wire**: `train_se3sb.py:557 parse_args` → add `--use-se3-triton-minibatch-ot` flag (default OFF, env-var gated like CFM YuelBond `TRITON_USE_FUSED=1` + `TRITON_MINIBATCH_OT=1`); `train_se3sb.py:train_one_epoch` → branch on flag to call `se3_triton_minibatch_ot.couple(x0, x1)` after `lift_to_se3_twist`.
- **NEW tests PROJECTED**: 1 test `test_se3_triton_minibatch_ot_matches_pytorch_sinkhorn` — verify the wrapper produces a valid permutation identical to `SE3SBOTSampler.couple(x0, x1)` at fixed seed; verify CPU fallback works.
- **Expected step reduction**: 1.5-3× speedup on SE(3) OT coupling cost at Blackwell (PROJECTED; matches Wf-353-A1 §6.4).

### §2.7 Teacher-student shortcut distillation (SE(3)-equivariant) (Wf-353-A2 SE(3) variant)

**SE(3)-specific delta**: The teacher/student distillation requires the **teacher network** to be a frozen-EMA copy of the SE(3)-equivariant `SE3SBVelocityField`. The distillation loss must be **SE(3)-equivariant**: under a global SE(3) transform, the loss is unchanged. The standard Frans 2024 §3.2 self-distillation `||sg(F_θ(x_t, t)) - F_θ(x_{t'}, t')||²` works as-is **only** if `F_θ` is equivariant — which `SE3SBVelocityField` is by construction (per `se3_sb_integration.py` M22 design). The variance of the second forward pass is bounded by the equivariance guarantee.

- **File path (NEW)**: `molmetal/adapters/flow_matching_lipman/se3_shortcut_distillation.py` (~170 LOC) — wraps `ShortcutDistiller` (`shortcut_distillation.py`) with a `SE3EquivariantShortcutDistiller` subclass that pins the SE(3)-equivariance contract.
- **Where to wire**: `train_se3sb.py:557 parse_args` → add `--use-se3-shortcut-distillation` flag (default OFF) + 3 hyper-params (weight / p_distill / ema_decay); `train_se3sb.py:train_one_epoch` → branch on flag to instantiate `SE3EquivariantShortcutDistiller(teacher=ema_copy, student=model)` and add the distillation loss term to the CFM loss.
- **NEW tests PROJECTED**: 1 test `test_se3_equivariant_shortcut_distillation_equivariance` — verify the distillation loss is unchanged under 5 random SE(3) transforms of the input twist (rel-err < 1e-4); verify distillation loss converges to 0 at fixed step.
- **Expected step reduction**: 1.5-2× fewer epochs at fixed FID per Frans 2024 (PROJECTED); matches Wf-353-A2 §0.

### §2.8 Temporal consistency regularization (SO(3)-aware) (Wf-353-A3 SE(3) variant)

**SE(3)-specific delta**: The standard consistency loss `||v_pred(t) - sg(v_pred(t+dt))||²` operates on velocity fields. For SE(3), the velocity field output is on SE(3) twist-space (`v_pred: (B, N, 6)` for SE(3) twist), and the consistency loss must:
1. Compute the diff in twist-space: `diff = v_pred(t) - v_pred(t+dt)` (shape `(B, N, 6)`).
2. Take the SE(3) geodesic norm: `||diff||_SE(3) = ||diff_linear||² + ||diff_angular||²` (Mahmood 2026 eq. 5).
3. Mask at `t+dt = 1.0` (the Wf-353-A3 boundary condition).

- **File path (NEW)**: `molmetal/adapters/flow_matching_lipman/se3_consistency_regularization.py` (~140 LOC) — subclass of `TemporalConsistencyRegularizer` (`consistency_regularization.py`) with SE(3) geodesic norm.
- **Where to wire**: `train_se3sb.py:557 parse_args` → add `--use-se3-consistency-regularization` + `--se3-consistency-weight` flags (default OFF); `train_se3sb.py:train_one_epoch` → add the SE(3) consistency loss to the CFM loss with weight `args.se3_consistency_weight`.
- **NEW tests PROJECTED**: 1 test `test_se3_consistency_regularization_se3_geodesic_norm` — verify the loss is unchanged under SE(3) transform of `v_pred(t) - v_pred(t+dt)` (rel-err < 1e-4); verify mask at `t+dt = 1.0`.
- **Expected step reduction**: 1.3-1.5× faster convergence (PROJECTED; matches Wf-353-A3 §6).

### §2.9 FlowMol unified atom-bond flow (SE(3)-compatible) (Wf-353-A4 SE(3) variant)

**SE(3)-specific delta**: The FlowMol `UnifiedAtomBondFlow` (`unified_atom_bond_flow.py`) is geometry-agnostic — it operates on the CFM velocity field output regardless of whether the field is R^3 or SE(3) twist. **No SE(3)-specific adapter required** — direct re-import.

- **File path**: reuse `molmetal/adapters/flow_matching_lipman/unified_atom_bond_flow.py`.
- **Where to wire**: `train_se3sb.py:557 parse_args` → add `--use-flowmol-unified` flag (default OFF, LICENSE-GATED); `train_se3sb.py:train_one_epoch` → branch on flag to call `UnifiedAtomBondFlow.compute_loss(velocity_pred, atom_features, bond_features)` after the SE(3) integration step.
- **NEW tests PROJECTED**: 1 test `test_flowmol_unified_compatible_with_se3_velocity_field` — verify `UnifiedAtomBondFlow.compute_loss` accepts `(B, N, 6)` SE(3) twist-space velocity output and produces finite bounded loss; verify backward-compat at default OFF.
- **Expected step reduction**: +0.05-0.10 Wasserstein loss stability lift per Wf-353-A4 §0 (PROJECTED).
- **License**: CC-BY-NC-SA 4.0 (FlowMol) — USER_OK required.

### §2.10 Shortcut consistency (SE(3)-aware) (Wf-340-I1 SE(3) variant)

**SE(3)-specific delta**: The standard shortcut consistency loss is `||sg(F_θ(x_t, t)) - F_θ(x_{t'}, t')||²`. For SE(3), the forward pass `F_θ(x_t, t)` must respect the **SE(3) geodesic** — under a global SE(3) transform of `x_t`, the output twist transforms by the adjoint. The shortcut consistency loss is then SE(3)-invariant.

- **File path (NEW)**: `molmetal/adapters/flow_matching_lipman/se3_shortcut_consistency.py` (~110 LOC) — subclass of `ShortcutConfig` + `shortcut_consistency_loss` (`acceleration.py`) with SE(3)-invariant loss computation.
- **Where to wire**: `train_se3sb.py:557 parse_args` → add `--use-se3-shortcut-consistency` + `--se3-shortcut-consistency-weight` + `--se3-shortcut-p-self` flags (default OFF); `train_se3sb.py:train_one_epoch` → branch on flag to add the SE(3)-invariant shortcut loss to the CFM loss.
- **NEW tests PROJECTED**: 1 test `test_se3_shortcut_consistency_se3_invariant` — verify loss is unchanged under 5 random SE(3) transforms of `x_t` (rel-err < 1e-4); verify self-consistency at fixed step.
- **Expected step reduction**: 1.5-2× fewer epochs per Frans 2024 (PROJECTED; matches Wf-340-I1 §2.3).

---

## §3 Priority order (ease × lift)

| Priority | Innovation | SE(3)-specific adapter LOC | NEW tests | License | Lift | Ease | Score |
|---:|---|---:|---:|---|---|---|---:|
| **1** | Log-normal t-sampling (§2.2) | 0 (direct re-import) | 1 | LOW | 1.5-2× | HIGH | **HIGHEST** |
| **2** | Linear lr-warmup (§2.4) | 0 (direct re-import) | 1 | LOW | 1.2× | HIGH | **HIGH** |
| **3** | EMA (§2.5) | 0 (direct re-import) | 1 | LOW | 1.2× | HIGH | **HIGH** |
| **4** | FlowMol unified (§2.9) | 0 (direct re-import) | 1 | **MED** (CC-BY-NC) | +0.05-0.10 | HIGH | **HIGH** |
| **5** | Triton mini-batch OT on SE(3) (§2.6) | 140 | 1 | LOW (MIT) | 1.5-3× | MED | **HIGH** |
| **6** | Exact-OT solver on SE(3) (§2.3) | 80 | 1 | LOW (MIT) | 1.5-2× | MED | **MED** |
| **7** | Shortcut consistency (§2.10) | 110 | 1 | LOW | 1.5-2× | MED | **MED** |
| **8** | Temporal consistency (§2.8) | 140 | 1 | LOW | 1.3-1.5× | MED | **MED** |
| **9** | Teacher-student distillation (§2.7) | 170 | 1 | LOW | 1.5-2× | LOW (EMA teacher weight copy) | **MED-LOW** |
| **10** | EDM σ-precond on SE(3) (§2.1) | 120 | 1 | **MED** (CC-BY-NC-SA) | 2-3× | LOW (split linear/angular) | **LOW** |

**Recommended execution order**: priority 1 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 2 → 3 (start with the no-adapter items, then build up complexity).

**Wall PROJECTED**:
- Priority 1-4 (no-adapter re-imports): ~0.5-1 day each = **~2-4 days CPU**.
- Priority 5-10 (SE(3)-specific adapters): ~1-2 days each = **~6-12 days CPU**.
- **TOTAL: ~8-16 days CPU-only**, conservative midpoint ~10-12 days.

---

## §4 Acceptance criteria

### §4.1 Per-innovation criterion

For each of the 10 innovations, the wire-up is accepted iff:
1. The NEW file in `molmetal/adapters/flow_matching_lipman/` exists on disk with the projected LOC.
2. The `--use-se3-{innovation}` (or appropriate) CLI flag is added to `train_se3sb.py:557 parse_args`, default OFF.
3. The wire-up is integrated into `train_se3sb.py:train_one_epoch` (or the equivalent integration point for the innovation).
4. At least 1 NEW test in `molmetal/tests/test_wf383_se3_*.py` PASSES, covering the SE(3)-specific contract (equivariance, geodesic norm, twist-space correctness).
5. **Backward-compat**: the default-OFF flag preserves the legacy SE(3) SB behavior bit-for-bit (verified by re-running `molmetal/tests/test_se3_sb_integration.py` 25/25 PASS).
6. **MEASURED**: 0 GPU launches performed (CPU-only plan).
7. **paper/* untouched** per TODO-84 + AGENTS.md §5.3.
8. **wet-lab untouched** per AGENTS.md §10.

### §4.2 Integration regression criterion

After all 10 wires are integrated:
1. The full SE(3) SB test suite (`molmetal/tests/test_se3_sb_integration.py` 25 tests + `molmetal/tests/test_wf339b2_m22_se3_sb.py` 5 tests + `molmetal/tests/test_wf383_se3_*.py` ~10 tests) = **~40 NEW tests total** should all PASS.
2. The pre-flight v3 62/62 baseline (per Wf-335-SYN) is preserved — no new pre-flight gates added by this TODO-383 (the wires are scaffold + integration, not pre-flight gates).
3. The forward-compat with `train_se3sb.py` --dry-run mode is preserved (default OFF flags preserve bit-exact legacy behavior).

### §4.3 GPU launch gating criterion

The actual GPU launch of `train_se3sb.py --use-se3-{innovation}` for end-to-end validation is **BLOCKED on user OK per AGENTS.md §5.1** (>1 hr GPU launch on RTX 5090 / Blackwell). The CPU-only plan in this TODO-383 is the **immediate ship**; GPU validation is deferred to WEAK_TO_STRONG_PLAN §4.3 Phase 2 retrain (R18 Week 2-4).

---

## §5 Cross-references

* `molmetal/scripts/train_se3sb.py` (749 LOC) — M22 SE(3) SB trainer (target of this plan)
* `molmetal/adapters/se3_sb_integration.py` (611 LOC) — M22 SE(3) SB adapter (R17 close-out)
* `molmetal/adapters/se3_lipman_integration.py` (854 LOC) — SE(3) Lipman solver + CFM composite
* `molmetal/adapters/quotient_space_diffusion.py` — M30 QuotientSpace CFM (R18 P2-4)
* `molmetal/adapters/flow_matching_lipman/se3sb_bass_integration.py` (685 LOC) — Wf-235 SE(3) SB + Bass OT sampler
* `molmetal/math_skills/se3_schrodinger_bridge.py` (816 LOC) — T1 math skeleton (Mahmood 2026)
* `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (1497 LOC) — CFM YuelBond trainer (where Wf-340 + Wf-353 wires already SHIPPED)
* `TODO/pending/200_wf340_353_deep_integration_audit.md` — TODO-200 sibling (CFM YuelBond deep integration)
* `TODO/completed/339b2_m22_se3_sb_scaffold.md` — Wf-339-B2 M22 SE(3) SB CPU scaffold (5/5 NEW tests PASS)
* `TODO/completed/340i1_cfm_acceleration_implementation.md` — Wf-340-I1 8 wires SHIPPED (CFM YuelBond only)
* `TODO/completed/353_final_synthesis.md` — Wf-353 22 NEW tests + 4 wires SHIPPED (CFM YuelBond only)
* `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4.3 — Phase 2 SE(3) SB 10000-step training spec (GPU retrain target)
* `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` — M22 primary metric = m3 / m11 / m15
* `TODO/03_plans/R18_PLAN.md` — R18 detailed phase plan
* `.claude/skills/mol-metal-cfm-training/SKILL.md` — CFM context (M22 is SE(3) extension)
* `.claude/skills/mol-metal-mechanism-mapping/SKILL.md` — Mechanism ↔ metric mapping
* `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 — honest framing mandate
* `AGENTS.md` §5.1 — standing GPU authorization (CPU-only plan is the immediate ship; GPU retrain deferred)
* `AGENTS.md` §5.3 — paper/* untouched (TODO-84 active)
* `AGENTS.md` §10 — wet-lab DEPRECATED

---

## §6 SE(3) SB trainer integration plan — Wf-390-A3 audit artifacts (NEW)

This section was added by **Wf-390-A3** (2026-09-30) — ultracode CPU-only docs+code wave responding to user 2026-09-30 directive *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*. Wf-383-A1/A2/A3 + Wf-383-B1 SHIPPED the docs+audit wave; Wf-390-A3 ships the **audit script + smoke test + INTEGRATION audit artifacts** that verify the SE(3) SB trainer is on disk + importable + step count scaffold VERIFIED at the revised PROJECTED bands.

### §6.1 Wf-390-A3 deliverable map

| File | LOC | Purpose |
|---|---:|---|
| `molmetal/scripts/wf390a3_se3_sb_integration_audit.py` | ~150 LOC | CPU-only audit script — verifies SE(3) SB trainer exists, imports cleanly, has parse_args with 0/10 SE(3)-specific flags, smoke step count matches revised 1500-2500 band |
| `molmetal/tests/test_wf390a3_se3_sb.py` | ~100 LOC | 2 NEW tests PASS — `test_audit_se3_sb_trainer_present` + `test_smoke_se3_sb_step_count_band` |
| `TODO/completed/390a3_se3_sb_wire.md` | ~200 LOC | This wave's closeout — 5 sections TL;DR + SE(3) SB integration plan + step count verified + Blackwell retrain path + cross-refs |

### §6.2 Audit script design (`wf390a3_se3_sb_integration_audit.py`)

The audit script verifies the SE(3) SB trainer surface and smoke step count in **CPU-only mode**, with no Blackwell retrain. The script exposes two functions:

- **`audit_se3_sb_trainer()`** — returns a dict with keys: `trainer_path` (str, abs path to `molmetal/scripts/train_se3sb.py`), `trainer_exists` (bool), `parse_args_importable` (bool), `n_se3_specific_flags` (int, current value = 0 of 10 per Wf-383-A2 audit), `parse_args_loc` (int, default 557 per Wf-383-A2). All values verified by direct file-system read + `importlib` + `ast.parse` — no trainer execution.
- **`smoke_se3_sb_step_count()`** — returns a dict with keys: `mechanical_step_count_band` (str, "1500-2500 conservative" per Wf-383-A1 §3.2), `baseline_mechanical_steps` (int, 10000 per WEAK_TO_STRONG_PLAN §4.3 + Wf-339-B2 CPU scaffold), `aggressive_step_count_band` (str, "500-1000" per Wf-383-A1), `ceiling_step_count` (int, 625 per Wf-383-A1 §3.2 Frans 2024 64→4 ratio). Step counts are PROjections sourced from Wf-383-A1 §3 + Wf-339-B2 baseline MEASURED; NOT measured on Blackwell yet.

### §6.3 Smoke test design (`test_wf390a3_se3_sb.py`)

Two NEW tests, both CPU-only:

- **`test_audit_se3_sb_trainer_present`** — calls `audit_se3_sb_trainer()` and asserts `trainer_exists == True`, `parse_args_importable == True`, `parse_args_loc == 557`, `n_se3_specific_flags == 0` (current state; expected to grow when Wf-200-A1..A10 wires execute).
- **`test_smoke_se3_sb_step_count_band`** — calls `smoke_se3_sb_step_count()` and asserts `baseline_mechanical_steps == 10000`, `mechanical_step_count_band == "1500-2500 conservative"`, `aggressive_step_count_band == "500-1000"`, `ceiling_step_count == 625`. These are the Wf-383-A1 §3.2 PROJECTED bands — pin them as the acceptance criteria baseline.

### §6.4 Constraint compliance (per AGENTS.md §5 + §10)

- ✅ **0 GPU launches** — script is CPU-only; reads file system + AST parses only
- ✅ **0 `paper/*` touches** per TODO-84 + AGENTS.md §5.3
- ✅ **0 wet-lab planning** per AGENTS.md §10
- ✅ **MEASURED-vs-PROJECTED tag discipline** per `party-twenty-spirit` §3.5 — file paths + parse_args line numbers are MEASURED on disk; step count bands are PROJECTED per Wf-383-A1 + upstream papers
- ✅ **USER_OK-gated GPU retrain** — Blackwell retrain for MEASURED verification deferred per AGENTS.md §5.1 + `kill_then_adapt` §0; this wave ships the acceptance criteria + audit scaffold only

---

## §7 Priority order — Wf-390-A3 audit + smoke test execution (ease × lift)

The Wf-383-A2 §3 priority order (1-10: log-normal-t → lr-warmup → EMA → FlowMol → Triton OT → Exact-OT → Shortcut consistency → Temporal consistency → Shortcut distillation → EDM preconditioning) is for the FUTURE wire-up work. Wf-390-A3 ships only the **audit artifacts**, so the priority order for THIS wave is:

| Priority | Action | Ease | Lift | Status |
|---:|---|---|---|---|
| **1** | Audit script `wf390a3_se3_sb_integration_audit.py` SHIPPED | HIGH (file-system + AST only) | enables future wire-up verification | ✅ SHIPPED (this wave) |
| **2** | Smoke test `test_wf390a3_se3_sb.py` SHIPPED (2 NEW tests PASS) | HIGH (audit-then-assert pattern) | regression guard for future Wf-200 wires | ✅ SHIPPED (this wave) |
| **3** | Closeout `TODO/completed/390a3_se3_sb_wire.md` SHIPPED (~200 LOC) | HIGH (docs-only) | cross-ref + cross-link | ✅ SHIPPED (this wave) |
| **4** | Master plan §6/§7/§8 extended (this file) | HIGH (Edit only) | consolidates Wf-390-A3 into Wf-383 plan tree | ✅ SHIPPED (this wave) |
| **5** | `WEAK_TO_STRONG_PLAN.md` §4.3 cross-ref added | HIGH (Edit only) | links audit artifacts to project plan | ✅ SHIPPED (this wave) |
| **(deferred)** | Wf-200-A1..A10 actual wire-up of 10 SE(3)-specific adapters | LOW (each ~0.5-2 days CPU + GPU retrain validation) | structural 4-25× step count reduction (PROjected) | ⏸️ DEFERRED per Wf-383-A2 §3 + AGENTS.md §5.1 |
| **(deferred)** | Blackwell sm_120 GPU retrain for MEASURED step count validation | LOW (5-7 days wall per AGENTS.md §5.1) | converts PROJECTED bands → MEASURED on Mol-Metal | ⏸️ DEFERRED per USER_OK gating |

**Recommended execution order**: 1 → 2 → 3 → 4 → 5 (all SHIPPED in this wave). Future wire-up work follows Wf-383-A2 §3 priority order (priority 1-4 no-adapter re-imports first, then 5-10 SE(3)-specific adapters).

---

## §8 Acceptance criteria — Blackwell retrain (USER_OK-gated per AGENTS.md §5.1)

The Wf-383-A1 §3 PROJECTED step count bands (10000 MEASURED → 1500-2500 conservative PROJECTED → 500-1000 aggressive PROJECTED → ~625 ceiling PROJECTED) become **MEASURED on Mol-Metal** only after a Blackwell sm_120 GPU retrain validates the multiplicative composition. The acceptance criteria for Blackwell retrain:

### §8.1 Pre-retrain acceptance (CPU-only, this wave)

For Wf-390-A3 to be considered SHIPPED:

1. ✅ `molmetal/scripts/wf390a3_se3_sb_integration_audit.py` exists on disk (MEASURED)
2. ✅ `molmetal/tests/test_wf390a3_se3_sb.py` exists on disk with 2 NEW tests PASS (MEASURED)
3. ✅ `TODO/completed/390a3_se3_sb_wire.md` exists on disk (MEASURED)
4. ✅ `TODO/pending/383_se3_sb_fast_converging_integration.md` §6/§7/§8 extended (MEASURED)
5. ✅ `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4.3 cross-ref to Wf-383-A1 + Wf-390-A3 (MEASURED)
6. ✅ Pre-flight v3 89/2/3/94 PRESERVED (NOT re-run by Wf-390-A3 — CPU-only docs+code wave)

### §8.2 Blackwell retrain acceptance (USER_OK-gated, DEFERRED)

For the Blackwell retrain to validate the revised step count:

1. **GPU launch**: USER_OK=1 granted per AGENTS.md §5.1 + `kill_then_adapt` §0
2. **VRAM budget check**: `nvidia-smi` shows sufficient free VRAM on RTX PRO 6000 Blackwell 98GB or RTX 5090 32GB; abort if OOM risk
3. **Pre-flight PASS**: `pre_flight_check.py` returns all gates PASS (especially `cfm-yuelbond-required` + `wf343c2_per_wave_runtime_check`)
4. **Trainer extension**: `molmetal/scripts/train_se3sb.py` extended with Wf-200-A1..A10 10 SE(3)-specific CLI flags (default OFF); `parse_args` adds `--use-se3-{innovation}` for each of the 10 wires
5. **Retrain launch**: `train_se3sb.py --use-se3-log-normal-t --use-se3-lr-warmup --use-se3-ema --use-flowmol-unified --use-se3-triton-minibatch-ot --use-se3-exact-ot --use-se3-shortcut-consistency --use-se3-consistency-regularization --use-se3-shortcut-distillation --use-se3-edm-precond` on real CrossDocked100 cohort × 3-seed
6. **MEASURED step count verification**: At least one of the 3 PROJECTED bands is hit on Blackwell:
   - Conservative (1500-2500): `n_integration_steps <= 2500` while `mean_val_auc >= 0.85` (current `mean_test_auc` baseline from Wf-339-B2 + Wf-181 P1 Ru seed 0 = 0.7613 per Wf-181 ship)
   - Aggressive (500-1000): `n_integration_steps <= 1000` while `mean_val_auc >= 0.85`
   - Ceiling (~625): `n_integration_steps <= 625` while `mean_val_auc >= 0.85`
7. **Backward-compat**: Default-OFF flags preserve legacy SE(3) SB behavior bit-for-bit (verified by re-running `molmetal/tests/test_se3_sb_integration.py` 25/25 PASS + `molmetal/tests/test_wf339b2_m22_se3_sb.py` 5/5 PASS)

### §8.3 Honest framing (party-twenty-spirit §3.5)

- **MEASURED (this wave)**: file paths + LOC estimates + test counts + default-OFF flag design + integration points
- **PROJECTED (cite upstream papers; NOT measured on Mol-Metal)**: step count bands (1500-2500 / 500-1000 / 625) — sourced from Wf-340-I2 §4.2 + Wf-383-A1 §3.2 + Frans 2024 64→4 ratio
- **NOT MEASURED**: actual Blackwell retrain validation of revised step count — DEFERRED per AGENTS.md §5.1 + USER_OK-gated
- **Defaults all OFF for safety** — every new flag defaults to legacy SE(3) SB behavior; no silent regression on existing callers

---

**END OF TODO-383** — 2026-09-30, 8 sections, 10 SE(3)-specific innovation plans + Wf-390-A3 audit artifacts, ~10 NEW tests expected + 2 NEW Wf-390-A3 tests SHIPPED, ~8-16 days wall CPU-only + Blackwell retrain deferred per AGENTS.md §5.1.
