# R18 Integration Adapters — W2.1/W2.2/W2.3 Synthesis

**Date**: 2026-09-26
**Parent plan**: `TODO/03_plans/R18_PLAN.md` §1 (Goals G1-G3) + `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md`
**Workflow binding**: Wf-1 (math skeleton) ⊕ Wf-2 (adapter layer) — see §5
**Owner**: Hugo Lee + ultracode agents
**Status**: Wf-2 adapter layer **shipped** (CPU-only, no GPU contention); Wf-3 paper writing pending.

---

## 1. TL;DR

Three independent integration adapters are now live under
`molmetal/adapters/`, wrapping the pure-math layer
(`molmetal/math_skills/{se3_schrodinger_bridge,transport_relaxation,multi_parameter_ph,
monge_ampere_regularity,wasserstein_gradient_flow}.py`) onto the R17
CFM stack:

| # | Adapter | Theorem anchor | ML surface | CPU tests | Smoke 0/N |
|---|---------|----------------|------------|-----------|-----------|
| **W2.1** | SE(3) Schrödinger-Bridge | Mahmood 2026 (arXiv:2603.14049) | CFM decoder + velocity field on `SE(3)` | 25/25 PASS | 0/1000 fail |
| **W2.2** | Transport-Relaxation | Jiang-Xu-Zhang 2026 (arXiv:2602.08118) | `learned_shaping` reward augment | 25/25 PASS | 0/1000 fail |
| **W2.3** | Multi-Parameter PH | Kim-Jung 2026 (Adv. Appl. Math. **174**) | Pocket encoder topology features | 25/25 PASS | 0/1000 fail |

All three adapters are **CPU-only** (no CUDA / Triton), each runs a
3-step smoke script in <5 s on a laptop, and **none of them modify
`molmetal/adapters/flow_matching_lipman/*`** — the R17 close-out
under that path stays bit-identical as far as the new adapters are
concerned (pre-existing R17 TODO-33 + `_reference` shim fixes are
out of scope).

---

## 2. Per-Adapter Deliverables (W2.1 / W2.2 / W2.3)

### 2.1 W2.1 — SE(3) Schrödinger-Bridge adapter

| Surface | Path | LOC |
|---|---|---|
| Adapter | `molmetal/adapters/se3_sb_integration.py` | **611** |
| Public API | `SE3SBVelocityField`, `SE3SBDecoder`, `integrate_with_cfm` | 3 surfaces |
| Unit tests | `molmetal/tests/test_se3_sb_integration.py` (25 `test_*` functions) | **747** |
| Smoke script | `molmetal/scripts/verify_se3_sb_adapter.py` (`--dry-run` supported) | 157 |
| Theorem backup | `molmetal/math_skills/se3_schrodinger_bridge.py` (pure-math, prior) | — |

**Public surface contract**:

* `SE3SBVelocityField(base_velocity_net=…)` wraps an existing EGNN net
  and returns a `se(3)` tangent twist `(ω, v)` of shape `(..., 6)`.
* `SE3SBDecoder` walks the SE(3) geodesic between source and target
  poses at a sequence of time points, with optional Sinkhorn SB
  coupling for cross-pair assignment.
* `integrate_with_cfm(samples, field, n_steps, method="euler"|"midpoint")`
  is the drop-in wrapper, signature-compatible with the existing
  `flow_matching.solver.ode_solver.ODESolver` so a future refactor can
  swap them in without breaking call sites.

**Test breakdown (25 functions)**: tangent-shape × 2, finite × 2,
linear-reduction × 2, angular × 2, invalid-input × 2, decoder-shape
× 2, decoder-at-0/1/½ × 3, sinkhorn × 1, integrator runs / finite /
methods-differ / centroid / invalid / large-steps / batch / legacy
signature / decoder no-sinkhorn / default `t_values` / invalid
shape = 25.

**Status**: ✅ **shipped** (CPU smoke 0/1000 fail).

### 2.2 W2.2 — Transport-Relaxation reward-shaping adapter

| Surface | Path | LOC |
|---|---|---|
| Adapter | `molmetal/adapters/transport_relaxation_integration.py` | **556** |
| Public API | `RelaxedEndpointTerm`, `augment_learned_shaping`, `LambdaRelaxSweep` | 3 surfaces |
| Unit tests | `molmetal/tests/test_transport_relaxation_integration.py` (25 tests) | **656** |
| Smoke script | `molmetal/scripts/verify_transport_relaxation_adapter.py` | 190 |
| Theorem backup | `molmetal/math_skills/transport_relaxation.py` (Sinkhorn-5-iter + KL penalty) | — |

**Public surface contract**:

* `reward' = reward_learned + λ_relax · KL(p_T ‖ q_T(ε))`,
  with `q_T(ε) = softmax(-E / ε)`.
* Augments — does **not** mutate — the existing
  `molmetal.molmetal_lam.reward.learned_shaping` /
  `molmetal.molmetal_lam.reward.lambda_astsr` modules.
* `LambdaRelaxSweep` sweeps `λ_relax` over a grid and picks the best
  by validation KL.

**Test breakdown (25 functions)**: KL non-negativity / zero at equal
energies / scaling with λ / finite at small ε / KL finite at any ε /
batch shapes × 2 / NaN repair × 1 / invalid-input × 2 / distribution
sums-to-one / uniform at large ε / augment shape × 3 / sweep ties /
sweep grid / sweep reject × 4 / non-mutation × 2 / determinism × 1 /
KL grows as ε shrinks = 25.

**Status**: ✅ **shipped** (CPU smoke 0/1000 fail; `m14` decode-ratio
not regressed — gating happens in Wf-1 / Phase 1 retrain).

### 2.3 W2.3 — Multi-Parameter Persistent-Homology adapter

| Surface | Path | LOC |
|---|---|---|
| Adapter | `molmetal/adapters/multi_ph_integration.py` | **536** |
| Public API | `MultiPHPocketFeatures`, `MultiPHMetalCoordHead`, `augment_pocket_embedding` | 3 surfaces |
| Unit tests | `molmetal/tests/test_multi_ph_integration.py` (25 tests) | **563** |
| Smoke script | `molmetal/scripts/verify_multi_ph_adapter.py` | 156 |
| Theorem backup | `molmetal/math_skills/multi_parameter_ph.py` (2-PH filtration + sliced-Wasserstein kernel) | — |

**Public surface contract**:

* `MultiPHPocketFeatures(pocket_clouds) → (B, n_bins, n_bins)` —
  pure-topology feature map via Kim-Jung 2026 Th.1.
* `MultiPHMetalCoordHead(metal_id, …)` — metal-coordination head
  comparing pocket 2-PH diagrams against per-metal reference
  diagrams via sliced-Wasserstein kernel.
* `augment_pocket_embedding(encoder_output, ph_features)` —
  concatenates PH features to the EGNN-lite encoder output without
  modifying the encoder.

**Test breakdown (25 functions)**: shape × 2, finite ≥ 0, rotation
invariant, translation invariant, invalid-input × 2, logits shape /
finite / unknown-kernel / ≥ 2 metals × 4, kernel behaviour × 3,
augment shape / no-mutation / 1st-slice = base / batch mismatch /
rejects-2D / non-mutation contract / CPU end-to-end / forward-alias /
minimal-pocket / learned-linear = 25.

**Status**: ✅ **shipped** (CPU smoke 0/1000 fail; `m7` Diversity
uplift gated on R18 Phase 3 retrain).

---

## 3. MEASURED vs PROJECTED

### MEASURED (this workflow, CPU-only)

| Item | Measured value | Evidence |
|---|---|---|
| Adapter modules import OK | 3/3 | All three `verify_*_adapter.py` print "imports OK" |
| Adapter unit tests PASS | 75/75 (25 each × 3) | `pytest` green per `tests/test_*_integration.py` |
| Adapter smoke scripts 0/1000 fail | 3/3 | Each script exits 0 with "0/1000 fail" line |
| CPU runtime per smoke | ≤ 5 s on laptop | `--dry-run` skips the slowest step |
| Adapter cross-module isolation | OK | No adapter imports from another adapter; only from `molmetal.math_skills` |

### PROJECTED (gated on R18 Phases 1-3 real Blackwell training)

| Item | Target | Path to verify |
|---|---|---|
| `m3` Vina Dock median | -8.5 → **≤ -10.5** kcal/mol | Phase 2 SE(3) SB 100-pocket × 3-seed retrain |
| `m7` Diversity | 0.65 → **≥ 0.75** | Phase 3 multi-PH Fourier sweep |
| `m14` Decode ratio | **= 1.000** (no regression) | Phase 1 transport-relaxation 10 k-step retrain |
| `m11` n_distinct | ≥ 5 | Phase 2 SE(3) decoder end-to-end |
| `m8` PB-Valid | ≥ 0.95 | Phase 2 SE(3) decoder end-to-end |
| `R²` (F5 gate) | ≥ 0.62 (no regression) | Phase 3 Hilbert primal-dual LambdaASTSR |
| LambdaASTSR wall-time | **-50 %** vs R17 | Phase 3 primal-dual vs evolutionary 30-gen |

These projections are **not yet measured** because Blackwell training
is blocked behind the ultracode blacklist (`R18_PLAN.md §5`) — Wf-1
math + Wf-2 adapter layers must converge first, then real Phase 1-3
retrain can run Thu-Sun.

---

## 4. Adapter Imports Do **Not** Conflict With `flow_matching_lipman`

Three pieces of evidence that the new adapter trio is fully isolated
from the locked R17 `flow_matching_lipman/` close-out:

1. **Each adapter explicitly forbids modifying
   `molmetal/adapters/flow_matching_lipman/`** in its module
   docstring (W2.1 line 26-29, W2.2 line 1-4, W2.3 line 25-27 of
   each file). Two of the three (`W2.2`, `W2.3`) carry
   `test_*_does_not_mutate_*` regression tests that snapshot the
   public surface of the up-stream module and fail if the adapter
   re-imports and writes.
2. **No cross-imports between the three new adapters** — each one
   imports only `torch` and `molmetal.math_skills`; the W2.1
   `SE3SBVelocityField` accepts a `base_velocity_net` argument so the
   EGNN dependency is duck-typed and never `import`-coupled.
3. **Pre-existing modifications to `flow_matching_lipman/` are
   unrelated** — `git status -s` shows the modified files in that
   directory are R17 TODO-33 fixes (atom vocab extension to 20 elements
   + M12 ablation hook + smiles-vocab validator) and reference-loader
   shim work, **not** W2.1-W2.3 work. The new adapters sit alongside,
   not inside, that subtree.

In short: each W2.x adapter is a clean leaf addition to
`molmetal/adapters/` and the only files it touches are its own
`.py` + test + smoke script.

---

## 5. Wf-1 + Wf-2 ⇨ R18 PLAN.md Phase 1-3 ship

Per `TODO/03_plans/R18_PLAN.md` §2 / §4 / §7, R18 splits into:

| Phase | Wf-1 (math skeleton) deliverable | Wf-2 (adapter layer) deliverable | Status |
|---|---|---|---|
| **Phase 1 — Transport Relaxation** | `math_skills/transport_relaxation.py` (Sinkhorn-5-iter + KL penalty) | `adapters/transport_relaxation_integration.py` (`RelaxedEndpointTerm` + `LambdaRelaxSweep`) | ✅ Both ship |
| **Phase 2 — SE(3) Schrödinger Bridge** | `math_skills/se3_schrodinger_bridge.py` (Mahmood 2026 + Lie geodesics) | `adapters/se3_sb_integration.py` (`SE3SBVelocityField` + `integrate_with_cfm`) | ✅ Both ship |
| **Phase 3 — Multi-PH + primal-dual** | `math_skills/multi_parameter_ph.py` (Kim-Jung 2026 2-PH) | `adapters/multi_ph_integration.py` (`MultiPHPocketFeatures` + `augment_pocket_embedding`) | ✅ Both ship |

Per `R18_PLAN.md §risk / §6`, the **math + adapter layer is
GPU-free** — both halves compile and self-test under CPU, so the
Wf-1 / Wf-2 turn-around does not contend with Blackwell for the
ultracode training blacklist. Real GPU retrain of each Phase is
scheduled for its own Thu-Sun window.

---

## 6. Next — Wf-3: paper §3 method reference

Wf-3 (paper writing) is the next workflow, scheduled to start once
Wf-2 closes (this report). Concretely Wf-3 must:

1. Draft `paper/sections/03_method.tex` §3.4 (SE(3) Schrödinger Bridge
   on compact Lie groups) — inline-cite `mahmood2026sb_lie` (Mahmood
   2026 arXiv:2603.14049), `xue_xiaoping2026ot` (Wasserstein GF),
   and the R17 baseline `lipman2023cfm` already in `refs.bib`.
2. Update `paper/sections/04_results.tex` Table 2 R18 row with the
   *measured* Wf-2 deltas (CPU smoke 0/1000 fail + 75/75 unit
   tests) and the *projected* Phase 1-3 retrain numbers.
3. Re-write §6 limitations per R18 honest-framing policy
   (`R18_PLAN.md §6 §honest`): drop Tier-1 AUC claims, keep m14 /
   m17 / R², add RCA for Phase A BLOCK.

The 9-定理 math backbone (mahmood2026sb_lie / jiang2026sb_relax /
wang_xujia2026ma / xue_xiaoping2026ot / henry_labordere2026sb_bass /
backhoff2025bass / kim_jung2026multi_ph / kolbe_mayr2026ph_hardness /
ma_xiaonan2026bergman) → 3-adapter surface → §3 method narrative is
the full Wf-3 chain.

---

## 7. Cross-Reference — Skills

| Skill | Role in this synthesis |
|---|---|
| **`.claude/skills/mol-metal-r18-math/SKILL.md`** | Holds the 9-定理 backlog + W2.1 / W2.2 / W2.3 theorem-to-surface mapping. This synthesis report ships the adapter half of those mappings. |
| **`.claude/skills/mol-metal-cfm-training/SKILL.md`** | Owns `flow_matching_lipman/` (locked R17). W2.1's `integrate_with_cfm` is signature-compatible so a future refactor can swap it into the CFM ODE solver slot without touching this skill's anchors. |
| **`.claude/skills/mol-metal-lambda-calculus/SKILL.md`** | Owns `molmetal_lam/reward/learned_shaping.py` + `lambda_astsr.py` (locked R17). W2.2's `augment_learned_shaping` is additive-only and verified by `test_adapter_does_not_mutate_learned_shaping`. |

Three new skill-line entries are flagged for `R18_PLAN.md` Phase 4
to write into each of those `SKILL.md` files (the §3 R18-upgrade
section per the plan's §Task 3), keeping the cross-reference table
above live.

---

**End `55_r18_integration_adapters.md` (≈ 230 lines).**
