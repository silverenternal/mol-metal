# Wf-383-A2 — SE(3) Schrödinger Bridge Fast-Converging Integration Plan (CPU-only docs)

**Date:** 2026-09-30
**Wf:** Wf-383 (CPU-only docs+plan wave; 0 GPU launches; 0 source patches)
**Phase:** A2 — SE(3) SB-specific integration plan for 10 Wf-340 + Wf-353 fast-converging innovations
**Status:** ✅ SHIPPED — CPU-only plan complete; 10 NEW integration plans documented; 0 NEW tests (plan only); 0 GPU launches
**Owner:** Hugo Lee + Wf-383-A2 planning agent
**Mode:** Plain-text OUTPUT (per AGENTS.md §8 + Wf-291C workaround)
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline preserved); `mol-metal-mechanism-mapping` (M22 primary metric = m3 / m11 / m15); `mol-metal-r18-math` (Mahmood 2026 + Karras 2022 + Frans 2024)

---

## §0 TL;DR

| Field | Value |
|---|---|
| **Task** | CPU-only PLAN (file paths + LOC + NEW test count + expected step reduction) for SE(3) SB-specific integration of 10 fast-converging innovations from Wf-340 + Wf-353 |
| **Target trainer** | `molmetal/scripts/train_se3sb.py` (749 LOC, `parse_args` at line 557) |
| **Wired-shim status** | 0/10 wires currently in `train_se3sb.py` (10/10 SHIPPED only into `train_cfm_yuelbond_wasserstein.py` per Wf-340 + Wf-353) |
| **Total NEW modules** | 7 NEW files in `molmetal/adapters/flow_matching_lipman/` + 3 NO-NEW-FILE re-imports |
| **Total NEW LOC** | ~760 LOC production + ~150 LOC tests = **~910 LOC** |
| **Total NEW tests** | **10 NEW tests** (1 per innovation) + 2 integration regression tests = **~12 NEW tests** |
| **Wall PROJECTED** | ~8-16 days CPU-only (conservative midpoint ~10-12 days) |
| **GPU launches performed** | **0** (CPU-only plan; GPU bench deferred to WEAK_TO_STRONG_PLAN §4.3 Phase 2) |
| **MEASURED/PROJECTED discipline** | All step-reduction claims tagged PROJECTED (per upstream paper benchmarks); MEASURED only for the on-disk plan artifacts |

**Honest framing (`party-twenty-spirit` §3.5):**

- **MEASURED (this plan)**: file paths, LOC estimates, NEW test counts, default-OFF flag design, integration points, priority order, license classification — all on disk at `TODO/pending/383_se3_sb_fast_converging_integration.md` (NEW, ~520 LOC).
- **PROJECTED**: per-technique step reduction estimates — sourced from upstream papers (Karras 2022 / Frans 2024 / Mahmood 2026 / Khainb 2025 / Dunn & Koes 2024), NOT measured on Mol-Metal A/B.
- **NOT MEASURED**: actual wall-time impact on SE(3) SB retrain (deferred per AGENTS.md §5.1 until user OK).
- **Defaults all OFF for safety** — every new flag defaults to legacy SE(3) SB behavior; no silent regression on existing callers.

---

## §1 Background — Wf-340 + Wf-353 SHIPPED but only to CFM YuelBond

### §1.1 What was shipped (10 wires, ~3,510 LOC total)

Per `TODO/completed/340i1_cfm_acceleration_implementation.md` §1 + `TODO/completed/353_final_synthesis.md` §1.3, Wf-340 + Wf-353 ship 10 fast-converging wires as CPU-only SCAFFOLDS in `molmetal/adapters/flow_matching_lipman/` + `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`:

| # | Innovation | Module (LOC) | CFM YuelBond CLI flag (default) | Source |
|---:|---|---|---|---|
| 1 | EDM σ-preconditioning (Karras 2022) | `acceleration.py` (80) | `--use-edm-precond` (False) | Wf-340-A1 |
| 2 | Log-normal t-sampling | `acceleration.py` (25) | `--use-log-normal-t` (False) | Wf-340-A2 |
| 3 | Shortcut consistency (Frans 2024) | `acceleration.py` (70) | `--use-shortcut-consistency` (False) | Wf-340-I1 |
| 4 | EMA (NVlabs/edm halflife) | `acceleration.py` (50) | `--use-ema` (False) | Wf-340-I1 |
| 5 | Linear lr-warmup | `acceleration.py` (40) | `--use-lr-warmup` (False) | Wf-340-I1 |
| 6 | Exact-OT solver (POT `ot.emd`) | `optimal_transport_v2.py` (85) | `--use-exact-ot-coupling` (False) | Wf-340-I1 |
| 7 | Triton mini-batch OT (Khainb 2025) | `triton_minibatch_ot.py` (561) | `--triton-minibatch-ot` (False, env-gated) | Wf-353-A1 |
| 8 | Teacher-student shortcut distillation (Frans 2024) | `shortcut_distillation.py` (145) | `--use-shortcut-distillation` (False) | Wf-353-A2 |
| 9 | Temporal consistency regularization (EDM §5.2) | `consistency_regularization.py` (180) | `--use-consistency-regularization` (False) | Wf-353-A3 |
| 10 | FlowMol unified atom-bond flow (Dunn & Koes 2024, CC-BY-NC) | `unified_atom_bond_flow.py` (390) | `--use-flowmol-unified` (False, LICENSE-GATED) | Wf-353-A4 |

**Cumulative LOC**: ~1,790 LOC production modules + ~1,720 LOC tests = **~3,510 LOC** across Wf-340 + Wf-353 (all default OFF for backward compat).

### §1.2 Why SE(3) SB trainer is NOT yet wired

Per grep on `molmetal/scripts/train_se3sb.py` `parse_args` (line 557), the SE(3) SB trainer has **0/10** of the above CLI flags. The trainer still uses vanilla midpoint/Euler integration on SE(3) twist-space (`integrate_with_cfm(method="midpoint")` per `se3_sb_integration.py:430-574`) with stochastic mini-batch OT at the source/target coupling step (`SE3SBOTSampler.couple` per `se3sb_bass_integration.py:206-391`).

### §1.3 Why this matters (M22 mechanism-mapping)

Per `mol-metal-mechanism-mapping` skill + `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`, **M22 (SE(3) Schrödinger Bridge) is the primary mechanism for m3 Vina Dock lift** (PROJECTED −1.35 → −8.5 kcal/mol per WEAK_TO_STRONG_PLAN §2.3). The M22 SE(3) SB trainer is the **GPU retrain target** for R18 Phase 2 (WEAK_TO_STRONG_PLAN §4.3, ~2 weeks Blackwell single-card).

Without the 10 fast-converging wires extended to the SE(3) SB trainer:
1. The GPU retrain will run at legacy speed — no compounding of Wf-326/328/353 MEASURED 3-4× speedup on the SE(3) axis.
2. m3 Vina Dock lift will be smaller in wall-time-equivalent terms (PROJECTED 8 weeks → 4 weeks with Wf-353 wires extended; vs 8 weeks without).
3. The TODO-200 deep integration scope covers only `train_cfm_yuelbond_wasserstein.py`; TODO-383 covers the SE(3) SB trainer (M22).

---

## §2 SE(3) SB trainer stack (file paths + LOC)

### §2.1 Math layer (T1 — SHIPPED)

| File | LOC | Purpose |
|---|---:|---|
| `molmetal/math_skills/se3_schrodinger_bridge.py` | 816 | T1 math skeleton — `se3_exp`, `se3_log`, `so3_exp`, `so3_log`, `so3_slerp`, `SE3Geodesic`, `LogSinkhornLoss`, `SchrodingerBridgeSE3` (per Wf-339-B2 + R18 W1) |

### §2.2 Adapter layer (R17 + R18 close-out SHIPPED)

| File | LOC | Purpose |
|---|---:|---|
| `molmetal/adapters/se3_sb_integration.py` | 611 | M22 SE(3) SB adapter — `SE3SBVelocityField`, `SE3SBDecoder`, `integrate_with_cfm(method="midpoint"\|"euler")` (R17 close-out) |
| `molmetal/adapters/se3_lipman_integration.py` | 854 | SE(3) Lipman solver + CFM composite — `SE3LipmanSolver.solve`, `compare_se3_lipman_vs_base`, `SE3CFMComposite` (R18 P2-4) |
| `molmetal/adapters/quotient_space_diffusion.py` | 800+ | M30 QuotientSpace CFM — `QuotientSpace`, `QuotientDiffusionScheduler`, `QuotientSpaceCFM` (R18 P2-4) |
| `molmetal/adapters/flow_matching_lipman/se3sb_bass_integration.py` | 685 | Wf-235 SE(3) SB + Bass OT sampler — `SE3SBOTSampler`, `BassMartingaleOTSampler`, `SE3SBBassComposite` |

### §2.3 Trainer script (target)

| File | LOC | Purpose |
|---|---:|---|
| `molmetal/scripts/train_se3sb.py` | 749 | M22 SE(3) SB trainer — `parse_args` at line 557, imports from `se3_sb_integration.py` (R17 + Wf-339-B2 + WEAK_TO_STRONG_PLAN §4.3) |
| `molmetal/scripts/train_ru_5metal_se3sb.py` | (related) | Ru 5-metal SE(3) SB trainer variant (per Wf-130) |

### §2.4 Existing test suite

| Test file | Tests | Status |
|---|---:|---|
| `molmetal/tests/test_se3_sb_integration.py` | 25 | 25/25 PASS (R17 + Wf-339-B2 baseline) |
| `molmetal/tests/test_wf339b2_m22_se3_sb.py` | 5 | 5/5 PASS (Wf-339-B2 10000-step scaffold) |
| **Total existing SE(3) SB tests** | **30** | **30/30 PASS** |

---

## §3 Per-innovation plan (10 SE(3)-specific integration designs)

Each sub-section covers one innovation; the full detail is at `TODO/pending/383_se3_sb_fast_converging_integration.md` §2.1-§2.10.

### §3.1 EDM σ-preconditioning for SE(3) (§2.1)

- **SE(3) delta**: split `c_skip / c_out / c_in / c_noise` into **linear (translation) + angular (rotation)** branches since SO(3) noise is on rotation-vector manifold per `so3_log`.
- **NEW file**: `molmetal/adapters/flow_matching_lipman/se3_edm_preconditioning.py` (~120 LOC)
- **Wire-in**: `train_se3sb.py:557 parse_args` → `--use-se3-edm-precond` (OFF); `train_se3sb.py:train_one_epoch` → `se3_edm_preconditioned_loss(pred_linear, pred_angular, target_linear, target_angular, sigma)`.
- **NEW tests PROJECTED**: 1 test — verify `c_skip(σ)` applied separately to translation vs rotation-vector produces finite bounded output on random SE(3) twist; bit-identical at fixed σ.
- **Step reduction PROJECTED**: 2-3× faster SE(3) flow convergence per Karras 2022 §5.
- **License**: NVlabs/edm CC-BY-NC-SA 4.0 — **USER_OK required**.

### §3.2 Log-normal t-sampling (§2.2)

- **SE(3) delta**: compatible as-is — `sample_log_normal_t(b)` operates on time scalar `t ∈ [0, 1]`, identical parameterization for SE(3) twist-space interpolation. **No SE(3)-specific adapter required**.
- **NEW file**: 0 (direct re-import from `acceleration.py`).
- **Wire-in**: `train_se3sb.py:557 parse_args` → `--use-log-normal-t` (OFF); `train_se3sb.py:train_one_epoch` → replace `torch.rand(b)` at `integrate_with_cfm` call site with `sample_log_normal_t(b)`.
- **NEW tests PROJECTED**: 1 test — verify `sample_log_normal_t(b)` output ∈ [0, 1] and produces finite SE(3) twist-space input when used to scale `(1-t)·x0 + t·x1` on SO(3) rotation matrices.
- **Step reduction PROJECTED**: 1.5-2× fewer steps at matched SE(3) loss threshold per Karras 2022.
- **License**: LOW (no source port).

### §3.3 Exact-OT solver for SE(3) geodesic (§2.3)

- **SE(3) delta**: swap **Sinkhorn (entropic) for Hungarian (exact)** on the SE(3) twist-space cost matrix (`||xi0 - xi1||²` after `lift_to_se3_twist`).
- **NEW file**: `molmetal/adapters/flow_matching_lipman/se3_exact_ot_solver.py` (~80 LOC) — wraps `ot.emd` with `lift_to_se3_twist`.
- **Wire-in**: `train_se3sb.py:557 parse_args` → `--use-se3-exact-ot` (OFF); `train_se3sb.py:train_one_epoch` → branch on flag to call `se3_exact_ot_solver.couple(x0, x1)` instead of `SE3SBOTSampler.couple(x0, x1)`.
- **NEW tests PROJECTED**: 1 test — verify Hungarian OT cost ≤ Sinkhorn OT cost on a fixed (x0, x1) pair; valid permutation + bit-identical at fixed seed.
- **Step reduction PROJECTED**: 1.5-2× fewer inference NFE per Wf-340-I3 §2.1.
- **License**: POT MIT (no risk).

### §3.4 Linear lr-warmup (§2.4)

- **SE(3) delta**: compatible as-is — `LinearWarmupLRScheduler` operates on any optimizer regardless of geometry. **No SE(3)-specific adapter required**.
- **NEW file**: 0 (direct re-import from `acceleration.py`).
- **Wire-in**: `train_se3sb.py:557 parse_args` → `--use-lr-warmup` + `--lr-warmup-steps` (OFF); wrap `optimizer.step()` with `lr_scheduler.step()`.
- **NEW tests PROJECTED**: 1 test — verify `LinearWarmupLRScheduler` produces monotonic lr ramp on a SE(3) SB optimizer over 100 steps.
- **Step reduction PROJECTED**: ~1.2× faster initial convergence (standard warmup effect).
- **License**: LOW (no source port).

### §3.5 EMA (§2.5)

- **SE(3) delta**: EMA operates on velocity-field weights regardless of SE(3) vs Euclidean. **No SE(3)-specific adapter required** — type check at instantiation.
- **NEW file**: 0 (direct re-import from `acceleration.py`).
- **Wire-in**: `train_se3sb.py:557 parse_args` → `--use-ema` + `--ema-halflife-kimg` (OFF); call `ema.update(model)` at end of every step; `ema.copy_to(model)` before evaluation + restore (mirror CFM YuelBond pattern at lines 957-1114).
- **NEW tests PROJECTED**: 1 test — verify EMA buffer swaps SE3SBVelocityField weights losslessly (output unchanged at fixed seed).
- **Step reduction PROJECTED**: ~1.2× effective convergence (standard EMA effect).
- **License**: LOW (no source port).

### §3.6 Triton mini-batch OT on SE(3) (§2.6)

- **SE(3) delta**: wrap the Triton kernel (`triton_minibatch_ot.py`) with `lift_to_se3_twist` reduction → `(B, B, 6)` twist-space cost → Triton Sinkhorn → row-argmax permutation.
- **NEW file**: `molmetal/adapters/flow_matching_lipman/se3_triton_minibatch_ot.py` (~140 LOC).
- **Wire-in**: `train_se3sb.py:557 parse_args` → `--use-se3-triton-minibatch-ot` (OFF, env-var gated like CFM YuelBond); `train_se3sb.py:train_one_epoch` → branch on flag to call `se3_triton_minibatch_ot.couple(x0, x1)`.
- **NEW tests PROJECTED**: 1 test — verify wrapper produces valid permutation identical to `SE3SBOTSampler.couple(x0, x1)` at fixed seed; CPU fallback works.
- **Step reduction PROJECTED**: 1.5-3× speedup at Blackwell per Wf-353-A1 §6.4.
- **License**: LOW (MIT, Triton kernel already SHIPPED).

### §3.7 Teacher-student shortcut distillation (SE(3)-equivariant) (§2.7)

- **SE(3) delta**: the teacher network is a frozen-EMA copy of `SE3SBVelocityField` (equivariant by M22 construction). The distillation loss is SE(3)-equivariant by construction.
- **NEW file**: `molmetal/adapters/flow_matching_lipman/se3_shortcut_distillation.py` (~170 LOC) — wraps `ShortcutDistiller` with SE(3)-equivariance contract pin.
- **Wire-in**: `train_se3sb.py:557 parse_args` → `--use-se3-shortcut-distillation` + 3 hyper-params (OFF); `train_se3sb.py:train_one_epoch` → branch on flag to instantiate `SE3EquivariantShortcutDistiller(teacher=ema_copy, student=model)` + add distillation loss term.
- **NEW tests PROJECTED**: 1 test — verify distillation loss unchanged under 5 random SE(3) transforms (rel-err < 1e-4); converges to 0 at fixed step.
- **Step reduction PROJECTED**: 1.5-2× fewer epochs per Frans 2024.
- **License**: LOW.

### §3.8 Temporal consistency regularization (SO(3)-aware) (§2.8)

- **SE(3) delta**: replace L2 norm with **SE(3) geodesic norm** `||diff||_SE(3) = ||diff_linear||² + ||diff_angular||²` (Mahmood 2026 eq. 5) for the velocity-field diff `v_pred(t) - v_pred(t+dt)`.
- **NEW file**: `molmetal/adapters/flow_matching_lipman/se3_consistency_regularization.py` (~140 LOC) — subclass of `TemporalConsistencyRegularizer`.
- **Wire-in**: `train_se3sb.py:557 parse_args` → `--use-se3-consistency-regularization` + `--se3-consistency-weight` (OFF); `train_se3sb.py:train_one_epoch` → add SE(3) consistency loss with weight `args.se3_consistency_weight`.
- **NEW tests PROJECTED**: 1 test — verify loss unchanged under SE(3) transform (rel-err < 1e-4); mask at `t+dt = 1.0`.
- **Step reduction PROJECTED**: 1.3-1.5× faster convergence per Wf-353-A3 §6.
- **License**: LOW.

### §3.9 FlowMol unified atom-bond flow (SE(3)-compatible) (§2.9)

- **SE(3) delta**: FlowMol is geometry-agnostic — operates on velocity-field output regardless of R^3 or SE(3). **No SE(3)-specific adapter required**.
- **NEW file**: 0 (direct re-import from `unified_atom_bond_flow.py`).
- **Wire-in**: `train_se3sb.py:557 parse_args` → `--use-flowmol-unified` (OFF, LICENSE-GATED); `train_se3sb.py:train_one_epoch` → branch on flag to call `UnifiedAtomBondFlow.compute_loss(velocity_pred, atom_features, bond_features)`.
- **NEW tests PROJECTED**: 1 test — verify `UnifiedAtomBondFlow.compute_loss` accepts `(B, N, 6)` SE(3) twist-space velocity output and produces finite bounded loss.
- **Step reduction PROJECTED**: +0.05-0.10 Wasserstein loss stability lift per Wf-353-A4 §0.
- **License**: CC-BY-NC-SA 4.0 — **USER_OK required**.

### §3.10 Shortcut consistency (SE(3)-aware) (§2.10)

- **SE(3) delta**: the shortcut consistency loss is **SE(3)-invariant** under global SE(3) transform of `x_t` — the output twist transforms by the adjoint, so the diff norm is unchanged.
- **NEW file**: `molmetal/adapters/flow_matching_lipman/se3_shortcut_consistency.py` (~110 LOC) — subclass of `ShortcutConfig` + `shortcut_consistency_loss`.
- **Wire-in**: `train_se3sb.py:557 parse_args` → `--use-se3-shortcut-consistency` + `--se3-shortcut-consistency-weight` + `--se3-shortcut-p-self` (OFF); `train_se3sb.py:train_one_epoch` → branch on flag to add SE(3)-invariant shortcut loss.
- **NEW tests PROJECTED**: 1 test — verify loss unchanged under 5 random SE(3) transforms (rel-err < 1e-4); self-consistency at fixed step.
- **Step reduction PROJECTED**: 1.5-2× fewer epochs per Frans 2024.
- **License**: LOW.

---

## §4 Priority order (ease × lift)

Per `TODO/pending/383_se3_sb_fast_converging_integration.md` §3:

| Priority | Innovation | SE(3)-specific adapter LOC | NEW tests | License | Lift | Ease | Score |
|---:|---|---:|---:|---|---|---|---:|
| **1** | Log-normal t-sampling (§2.2) | 0 | 1 | LOW | 1.5-2× | HIGH | **HIGHEST** |
| **2** | Linear lr-warmup (§2.4) | 0 | 1 | LOW | 1.2× | HIGH | **HIGH** |
| **3** | EMA (§2.5) | 0 | 1 | LOW | 1.2× | HIGH | **HIGH** |
| **4** | FlowMol unified (§2.9) | 0 | 1 | **MED** | +0.05-0.10 | HIGH | **HIGH** |
| **5** | Triton mini-batch OT on SE(3) (§2.6) | 140 | 1 | LOW | 1.5-3× | MED | **HIGH** |
| **6** | Exact-OT solver on SE(3) (§2.3) | 80 | 1 | LOW | 1.5-2× | MED | **MED** |
| **7** | Shortcut consistency (§2.10) | 110 | 1 | LOW | 1.5-2× | MED | **MED** |
| **8** | Temporal consistency (§2.8) | 140 | 1 | LOW | 1.3-1.5× | MED | **MED** |
| **9** | Teacher-student distillation (§2.7) | 170 | 1 | LOW | 1.5-2× | LOW | **MED-LOW** |
| **10** | EDM σ-precond on SE(3) (§2.1) | 120 | 1 | **MED** | 2-3× | LOW | **LOW** |

**Recommended execution order**: priority 1 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 2 → 3 (start with no-adapter re-imports, then build complexity).

**Wall PROJECTED**: ~8-16 days CPU-only (conservative midpoint ~10-12 days), with all 10/10 wires integrated + 10/10 NEW tests PASS + 2/2 integration regression tests PASS + 30/30 existing SE(3) SB tests still PASS (no regression).

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
* `TODO/pending/383_se3_sb_fast_converging_integration.md` — TODO-383 plan (this file's master document)
* `TODO/completed/339b2_m22_se3_sb_scaffold.md` — Wf-339-B2 M22 SE(3) SB CPU scaffold (5/5 NEW tests PASS)
* `TODO/completed/340i1_cfm_acceleration_implementation.md` — Wf-340-I1 8 wires SHIPPED (CFM YuelBond only)
* `TODO/completed/353_final_synthesis.md` — Wf-353 22 NEW tests + 4 wires SHIPPED (CFM YuelBond only)
* `TODO/completed/353a1_triton_minibatch_ot.md` — Wf-353-A1 Triton kernel details
* `TODO/completed/353a2_shortcut_distillation.md` — Wf-353-A2 teacher-student shortcut distillation details
* `TODO/completed/353a3_consistency_regularization.md` — Wf-353-A3 temporal consistency regularization
* `TODO/completed/353a4_unified_atom_bond_flow.md` — Wf-353-A4 FlowMol unified atom-bond flow
* `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4.3 — Phase 2 SE(3) SB 10000-step training spec (GPU retrain target)
* `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` — M22 primary metric = m3 / m11 / m15
* `TODO/03_plans/R18_PLAN.md` — R18 detailed phase plan
* `.claude/skills/mol-metal-cfm-training/SKILL.md` — CFM context (M22 is SE(3) extension)
* `.claude/skills/mol-metal-mechanism-mapping/SKILL.md` — Mechanism ↔ metric mapping
* `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 — honest framing mandate
* `.claude/skills/mol-metal-r18-math/SKILL.md` — R18 mathematical theorem integration
* `AGENTS.md` §5.1 — standing GPU authorization (CPU-only plan is the immediate ship; GPU retrain deferred)
* `AGENTS.md` §5.3 — paper/* untouched (TODO-84 active)
* `AGENTS.md` §10 — wet-lab DEPRECATED

---

**END OF Wf-383-A2** — Owner Hugo Lee + Wf-383-A2 planning agent; SHIPPED; CPU-only PLAN-only (0 source patches, 0 NEW tests, 0 GPU launches); 10 SE(3) SB-specific integration designs documented at `TODO/pending/383_se3_sb_fast_converging_integration.md`; ~910 LOC PROJECTED + ~12 NEW tests PROJECTED + ~10-12 days wall CPU-only; all PROJECTED step-reduction claims cite upstream papers (Karras 2022 / Frans 2024 / Mahmood 2026 / Khainb 2025 / Dunn & Koes 2024) — NOT measured on Mol-Metal; standing directives §5.1 + §5.3 + §10 re-confirmed; ready for future workflow to execute the actual wire-up.
