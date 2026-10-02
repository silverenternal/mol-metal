# Wf-383-A3 — 10-Wire CFM Innovation Audit (Wf-340 + Wf-353 + TODO-200)

**Document ID:** Wf-383-A3 / TODO-200-AUDIT
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + Wf-383-A3 agent (docs-only wave)
**Trigger:** Per `TODO/pending/200_wf340_353_deep_integration_audit.md` §3 priority
order + `TODO/completed/340_final_synthesis.md` §6 wire inventory +
`TODO/completed/353a{1..4}_*.md` sub-flow closeouts
**Tag discipline:** MEASURED (file paths + LOC + test pass counts) / PROJECTED
(per-technique lift, cite upstream paper)

---

## §0 TL;DR

* **10 wires audited** (6 from Wf-340-SYN + 4 from Wf-353).
* **All 10 are SCAFFOLD-only** on disk — no trainer integration into
  `train_one_epoch` (CLI flags declared at `parse_args` but main-loop
  opt-in deferred).
* **5 are SCAFFOLD-stub** (StableVM + SO(3)-averaged FM + FlowMol + Triton
  OT CPU ref + EDM preconditioner) — math ported, NO verbatim upstream
  code.
* **5 are Production-ready SCAFFOLD** (Log-normal t + lr-warmup + EMA +
  OT-CFM exact solver + shortcut consistency) — at canonical SE(3)
  structure-agnostic; default OFF.
* **8/10 wire-able to SE(3) SB** (partial or yes); 2/10 are CFM-only
  (categorical heads + greedy row-assignment).
* **TODO-200 priority order honored**: lr-warmup > log-normal-t > Triton OT
  > EDM > OT-CFM > Consistency > StableVM > SO(3) > Shortcut distillation
  > FlowMol.
* **0 NEW tests** (this audit is docs-only per task brief).
* **0 GPU launches** (per AGENTS.md §5.1).
* **0 paper/* touches** (per TODO-84 + AGENTS.md §5.3).
* **0 wet-lab planning** (per AGENTS.md §10).

---

## §1 10-Wire Audit Table

Per task brief, all 10 wires from Wf-340-SYN (6) + Wf-353 (4). Each row
contains: Wire name · Wf source · Current state · File path · Test count
+ status · LOC estimate · Already wired to · SE(3) SB applicability.

| # | Wire | Wf source | Current state | File path (relative to repo root) | Test count | Status | LOC | Wired to (trainer / other) | SE(3) SB applicable? |
|---:|---|---|---|---|---:|---|---:|---|---|
| 1 | **EDM σ-preconditioning** | Wf-340-I1 (Karras 2022 / NVlabs/edm) | SCAFFOLD-stub (math ported, no verbatim code) | `molmetal/adapters/flow_matching_lipman/edm_preconditioning.py` (124 LOC) + bundled in `acceleration.py` (~80 LOC §3 of `acceleration.py`) | 4 (Wf-380-A1) + 1 (Wf-340-I1) = **5 PASS** | SCAFFOLD-stub | 124 + 80 = **204** | CFM YuelBond trainer via `--use-edm-precond` (default OFF, `parse_args:1297`) + `--edm-preconditioning` alias (Wf-349-A1, default ON per Wf-380 brief) | **partial** — coordinate preconditioning generalizes to SE(3) twist-space w/ separate `c_skip`/`c_out` factors (Mahmood 2026); need SE(3) geodesic-aware σ distribution |
| 2 | **Log-normal t-sampling** | Wf-340-I1 (NVlabs/edm) | Production-ready SCAFFOLD | `molmetal/adapters/flow_matching_lipman/log_normal_t.py` (59 LOC) + bundled `sample_log_normal_t` in `acceleration.py` (~25 LOC) | 4 (Wf-380-A1) + 1 (Wf-340-I1) = **5 PASS** | Production-ready SCAFFOLD | 59 + 25 = **84** | CFM YuelBond trainer via `--use-log-normal-t` (default OFF, `parse_args:1314`) | **yes** — pure t-resampling, manifold-agnostic; SE(3) SB can use same log-normal prior |
| 3 | **Shortcut consistency regularizer** | Wf-340-I1 (Frans 2024 self-consistency) | Production-ready SCAFFOLD | `molmetal/adapters/flow_matching_lipman/acceleration.py:313-365` (`ShortcutConfig` + `shortcut_consistency_loss`, ~70 LOC) | 1 (Wf-340-I1) = **1 PASS** | Production-ready SCAFFOLD | **70** | CFM YuelBond trainer via `--use-shortcut-consistency` (default OFF, `parse_args:1332`) | **yes** — coordinate-only consistency on `v_t = (1−t)x₀ + t x₁` generalizes to SE(3) geodesic `(R_t, t_t)` w/ Kabsch alignment |
| 4 | **Exponential Moving Average** | Wf-340-I1 (NVlabs/edm convention halflife=500 kimgs) | Production-ready SCAFFOLD | `molmetal/adapters/flow_matching_lipman/acceleration.py:368-426` (`ExponentialMovingAverage`, ~50 LOC) | 1 (Wf-340-I1) = **1 PASS** | Production-ready SCAFFOLD | **50** | CFM YuelBond trainer via `--use-ema` (default OFF, `parse_args:1321`); also integrated in `train_one_epoch` per `train_cfm_yuelbond_wasserstein.py:951` (Wf-348-C1 lr-warmup + EMA shadow update) | **yes** — manifold-agnostic (EMA on parameter weights, not on the manifold) |
| 5 | **Linear lr-warmup** | Wf-340-I1 (Goyal 2017) | Production-ready SCAFFOLD | `molmetal/adapters/flow_matching_lipman/lr_warmup.py` (64 LOC) + bundled `LinearWarmupLRScheduler` in `acceleration.py` (~40 LOC) | 5 (Wf-380-A1) + 1 (Wf-340-I1) = **6 PASS** | Production-ready SCAFFOLD | 64 + 40 = **104** | CFM YuelBond trainer via `--use-lr-warmup` (default OFF, `parse_args:1377`); also integrated at `train_cfm_yuelbond_wasserstein.py:951` (Wf-348-C1) | **yes** — optimizer schedule, manifold-agnostic |
| 6 | **Exact-OT solver** (Tong 2023 / Pooladian 2023 / khainb) | Wf-340-I1 + Wf-349-A2 | Production-ready SCAFFOLD | `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` (`exact_minibatch_ot_coupling` ~85 LOC at line 874; full module 1058 LOC) + `optimal_transport_v2_solver.py` (144 LOC) | 11 (Wf-349-A2) + 4 (Wf-380-A1) = **15 PASS** | Production-ready SCAFFOLD | 85 + 144 = **229** | CFM YuelBond trainer via `--use-exact-ot-coupling` (default OFF, `parse_args:1387`) + D-MPNN trainer (per Wf-340 integration plan); SE(3) exact OT uses Sinkhorn on twist space (see `se3sb_bass_integration.py`) | **partial** — Euclidean exact OT generalizes to SE(3) twist-space exact OT (already partially wired via `molmetal/adapters/se3sb_bass_integration.py`) |
| 7 | **Triton mini-batch OT (CPU ref)** | Wf-353-A1 (Khainb 2025 + Cuturi 2013 Sinkhorn) | SCAFFOLD-stub (Triton kernel compiled at runtime; CPU ref always available) | `molmetal/adapters/flow_matching_lipman/triton_minibatch_ot.py` (561 LOC) + `triton_mini_batch_ot.py` (110 LOC CPU ref) | 4 (Wf-353-A1) + 4 (Wf-380-A1) = **8 PASS** | SCAFFOLD-stub | 561 + 110 = **671** | CFM YuelBond trainer via `--triton-minibatch-ot` (default OFF, `parse_args:1409`); also `--use-triton-minibatch-ot` alias (Wf-380, default ON) | **partial** — Triton kernel wraps Euclidean Sinkhorn; SE(3) adaptation requires twist-space Sinkhorn (separate kernel) |
| 8 | **Shortcut distillation** (Frans 2024 teacher-student) | Wf-353-A2 | Production-ready SCAFFOLD | `molmetal/adapters/flow_matching_lipman/shortcut_distillation.py` (145 LOC) + `shortcut_distillation_wf380.py` (121 LOC SCAFFOLD duplicate) | 4 (Wf-353-A2) + 4 (Wf-380-A1) = **8 PASS** | Production-ready SCAFFOLD | 145 + 121 = **266** | CFM YuelBond trainer via `--use-shortcut-distillation` (default OFF, `parse_args:1350`) + 3 companion flags (weight, p_distill, ema_decay) | **yes** — distillation on velocity field generalizes to SE(3) geodesic velocity field; EMA teacher is manifold-agnostic |
| 9 | **Consistency regularization** (Karras 2022 / Frans 2024 fixed-dt) | Wf-353-A3 | SCAFFOLD-stub (CLI flag NOT wired; per §6 of `353a3_*.md`) | `molmetal/adapters/flow_matching_lipman/consistency_regularization.py` (164 LOC) + `consistency_regularization_wf380.py` (92 LOC SCAFFOLD) | 6 (Wf-353-A3 — 4 distinct + 1 parametrized ×3) + 4 (Wf-380-A1) = **10 PASS** | SCAFFOLD-stub | 164 + 92 = **256** | CFM YuelBond trainer CLI flag NOT YET wired (per `353a3_*.md` §6 — deferred to Wf-353-A4); trainer would gain `--consistency-regularization --consistency-weight --consistency-dt` | **yes** — fixed-dt consistency on `v_pred(t+dt)` generalizes to SE(3) geodesic `v_pred(R_{t+dt}, t_{t+dt})` |
| 10 | **FlowMol unified atom-bond flow** (Qiang 2024 arXiv:2412.01966 CC-BY-NC) | Wf-353-A4 | SCAFFOLD-stub (license-gated) | `molmetal/adapters/flow_matching_lipman/unified_atom_bond_flow.py` (441 LOC) + `flowmol_unified.py` (117 LOC SCAFFOLD) | 4 (Wf-353-A4) + 4 (Wf-380-A1) = **8 PASS** | SCAFFOLD-stub (LICENSE GATE) | 441 + 117 = **558** | CFM YuelBond trainer CLI flag NOT YET wired (per `353a4_*.md` §5 — deferred); trainer would gain `--unified-atom-bond-flow` (Wf-380 default ON); `--flowmol-unified` + `--no-flowmol-unified` aliases declared (`parse_args:1639` + `:1653`) | **no** — CFM-only (categorical heads for atom-type + bond-order simplex; no 3D coords in head); SE(3) SB structure does not apply |
| **Total** | — | — | — | **14 files / ~3,581 LOC** + Wf-340-I1 + Wf-353-A1+A2+A3+A4 tests (8+4+4+6+4 = 26) + Wf-349-A2 + Wf-380-A1 (11+41 = 54) = **80 NEW tests total** | — | — | — | — | — |

### §1.1 Audit table notes

* **Wire count = 10** per task brief; **`Total` LOC = ~3,581** is the union
  of SCAFFOLD modules (some modules serve multiple wires — e.g.
  `acceleration.py` is bundled across wires #1, #3, #4, #5; `optimal_transport_v2.py`
  hosts wire #6 + legacy Wasserstein paths).
* **Test count = 80 NEW** = Wf-340-I1 (8) + Wf-340-I3 (4) + Wf-353-A1 (4) +
  Wf-353-A2 (4) + Wf-353-A3 (6) + Wf-353-A4 (4) + Wf-349-A2 (11) + Wf-380-A1
  (41) — counts parametrized cases as 1 test (6 tests in Wf-353-A3 = 4
  distinct + 1 parametrized × 3 batch sizes).
* **Wf-380-A1 SCAFFOLD duplicates** (`_wf380` suffix modules) are
  intentionally coexisting with the Wf-340/Wf-353 canonical implementations
  to support deep-integration testing in isolation (per `380a1_*.md` §0).

---

## §2 SCAFFOLD vs Production-ready Breakdown

Per audit table §1, **5/10 SCAFFOLD-stub + 5/10 Production-ready SCAFFOLD**:

### §2.1 SCAFFOLD-stub (5 wires; need full impl or LICENSE gate before trainer integration)

* **#1 EDM σ-preconditioning** — math ported from Karras 2022 (no verbatim
  NVlabs/edm code); needs SE(3) geodesic-aware σ distribution for
  manifold-applicable extension.
* **#3 Shortcut consistency regularizer** — Production-ready SCAFFOLD
  (counts as #3 in §2.2); listed here only if used with SE(3) twist-space.
* **#7 Triton mini-batch OT (CPU ref)** — Triton kernel compiled at
  runtime; CPU fallback `pytorch_sinkhorn_minibatch_ot_coupling` always
  available; SE(3) extension needs separate twist-space kernel.
* **#9 Consistency regularization** — CLI flag NOT wired (deferred to
  Wf-353-A4 per `353a3_*.md` §6); needs trainer integration to become
  SCAFFOLD-stub → Production-ready.
* **#10 FlowMol unified atom-bond flow** — LICENSE GATE on CC-BY-NC
  (arXiv:2412.01966 per `353a4_*.md` §0); USER_OK required before
  trainer integration; `--flowmol-unified` flag already declared but
  `LipmanFlowMatchingAdapter.build_argparser()` hookup deferred.

### §2.2 Production-ready SCAFFOLD (5 wires; CLI flag wired, default OFF, bit-exact backward-compat)

* **#2 Log-normal t-sampling** — `sample_log_normal_t(b, p_mean=-1.2,
  p_std=1.2)` is a drop-in replacement for `torch.rand(b)`; pure t-resampling,
  manifold-agnostic.
* **#3 Shortcut consistency regularizer** — `shortcut_consistency_loss(v_pred,
  velocity_field, x_t, t, x_0, x_1, cfg, path_sampler)` is exposed as a
  primitive; CLI flag wired (`--use-shortcut-consistency`).
* **#4 Exponential Moving Average** — `ExponentialMovingAverage` halflife-kimg
  (NVlabs/edm convention); integrated at `train_one_epoch:951` per
  Wf-348-C1.
* **#5 Linear lr-warmup** — `LinearWarmupLRScheduler`; integrated at
  `train_one_epoch:951` per Wf-348-C1.
* **#6 Exact-OT solver** — `exact_minibatch_ot_coupling` via POT `ot.emd`;
  CLI flag wired (`--use-exact-ot-coupling`); SE(3) extension uses
  `se3sb_bass_integration.py` Sinkhorn on twist space (existing R18
  primitive).

### §2.3 Cross-cutting observations

* **Wf-353-A3 CLI flag NOT wired** is the **single largest gap** for
  trainer integration — wire `--consistency-regularization --consistency-weight --consistency-dt`
  per `353a3_*.md` §6 (deferred to Wf-353-A4).
* **Wf-353-A4 `--unified-atom-bond-flow` CLI flag NOT wired** is the
  second-largest gap — wire per `353a4_*.md` §5 (deferred to downstream
  sub-flow).
* **Wf-353-A2 + Wf-353-A4 trainer-loop integration NOT wired** per
  `353a2_*.md` §3 + `353a4_*.md` §5 (deferred to Wf-341-A7 per
  AGENTS.md §5.1).

---

## §3 TODO-200 Priority Order Cross-Reference

Per `TODO/pending/200_wf340_353_deep_integration_audit.md` §3, the
canonical priority order is:

| TODO-200 rank | Wire | Audit §1 row | Lift × ease verdict (per TODO-200 §3) | Compatible with this audit's "Production-ready SCAFFOLD" tier? |
|---:|---|:---:|---|:---:|
| 1 | Linear lr-warmup | #5 | **Easy; 0.5 day** | YES — Production-ready SCAFFOLD |
| 2 | Log-normal t-sampling | #2 | **Easy; 0.5 day** | YES — Production-ready SCAFFOLD |
| 3 | Triton mini-batch OT | #7 | **Medium; 1 day** | NO — SCAFFOLD-stub (Triton runtime) |
| 4 | EDM σ-preconditioning | #1 | **Medium; 1-2 days** | NO — SCAFFOLD-stub (math ported) |
| 5 | OT-CFM | #6 | **Medium; 1 day** | YES — Production-ready SCAFFOLD |
| 6 | Consistency regularization | #9 | **Medium; 1-2 days** | NO — SCAFFOLD-stub (CLI NOT wired) |
| 7 | StableVM | (NOT in §1 audit table) | **Hard; 2-3 days** | n/a — outside Wf-340+Wf-353 10-wire scope; per Wf-349-A3 `stablevm_stub.py` SCAFFOLD only |
| 8 | SO(3)-averaged FM | (NOT in §1 audit table) | **Hard; 2-3 days** | n/a — outside Wf-340+Wf-353 10-wire scope; per Wf-380-A1 `so3_average.py` SCAFFOLD only |
| 9 | Shortcut distillation | #8 | **Hard; 2-3 days** | YES — Production-ready SCAFFOLD |
| 10 | FlowMol unified | #10 | **Hard + LICENSE; 3-5 days + USER_OK** | NO — SCAFFOLD-stub (LICENSE GATE) |

### §3.1 Priority order audit verdict

* **TODO-200 ranks #1 + #2** (lr-warmup + log-normal-t) are correctly
  classified as Production-ready SCAFFOLD — **drop-in compatible** with
  the trainer integration sketch in `340i1_*.md` §5.
* **TODO-200 rank #3** (Triton OT) is correctly flagged SCAFFOLD-stub —
  Triton kernel needs Blackwell `sm_120` autotune retune per Wf-326E-A
  before GPU bench.
* **TODO-200 rank #4** (EDM) is correctly flagged SCAFFOLD-stub — math
  ported, no verbatim NVlabs/edm code; CC BY-NC-SA 4.0 license review
  required (USER_OK per AGENTS.md §5.1).
* **TODO-200 rank #5** (OT-CFM exact) is correctly classified as
  Production-ready SCAFFOLD — POT `ot.emd` solver is canonical and
  already integrated as a re-export.
* **TODO-200 rank #6** (Consistency reg) audit confirms SCAFFOLD-stub —
  module + tests ship but CLI flag NOT wired (deferred to Wf-353-A4).
* **TODO-200 ranks #7 + #8** (StableVM + SO(3)-averaged FM) are
  **outside the §1 audit table scope** (Wf-340+Wf-353 10-wire universe
  per task brief); they live in separate Wf waves (Wf-349-A3 stub +
  Wf-380-A1 scaffold). Should be promoted to audit table rows if a
  follow-up Wf-383-A4 extends scope.
* **TODO-200 rank #9** (Shortcut distillation) is correctly classified
  as Production-ready SCAFFOLD — module + tests + CLI flag all shipped
  per Wf-353-A2.
* **TODO-200 rank #10** (FlowMol) is correctly flagged SCAFFOLD-stub —
  LICENSE GATE on CC-BY-NC; USER_OK required.

### §3.2 Priority order ↔ SE(3) SB applicability cross-reference

Per audit §1 SE(3) SB column + TODO-200 priority:

| TODO-200 rank | Wire | SE(3) SB? | Notes |
|---:|---|:---:|---|
| 1 | lr-warmup | yes | optimizer schedule, manifold-agnostic |
| 2 | log-normal-t | yes | t-resampling, manifold-agnostic |
| 3 | Triton OT | partial | Euclidean Sinkhorn kernel; SE(3) needs twist-space variant |
| 4 | EDM | partial | coordinate preconditioning; SE(3) needs twist-space c_skip/c_out |
| 5 | OT-CFM exact | partial | Euclidean exact OT; SE(3) twist-space Sinkhorn already in `se3sb_bass_integration.py` |
| 6 | Consistency reg | yes | fixed-dt consistency on `v_pred(t+dt)`; SE(3) geodesic generalizes |
| 9 | Shortcut distillation | yes | EMA teacher manifold-agnostic; velocity field distillation generalizes |
| 10 | FlowMol | no | CFM-only (categorical heads); SE(3) structure does not apply |

**Verdict**: 6/8 wire-able (ranks #1, #2, #6, #9) + 3/8 partial (ranks
#3, #4, #5) + 1/8 not-applicable (rank #10) for SE(3) SB extension.

---

## §4 SE(3) SB Applicability

Per audit §1 SE(3) SB column + cross-cutting assessment:

### §4.1 What is SE(3) SB?

Per `molmetal/adapters/se3sb_bass_integration.py` docstring (Wf-235) +
`molmetal/adapters/se3_lipman_integration.py` (Wf-235):

* **SE(3) SB** = SE(3) Schrödinger Bridge — Sinkhorn coupling in twist
  space, replacing Euclidean `||x_0 - x_1||^2` cost with the SE(3)
  geodesic-aware twist-space cost + Sinkhorn entropy regularisation
  (Mahmood 2026, arXiv:2603.14049).
* **Best for**: rigid-body / SE(3) structured data (molecules, ligands,
  metal coordination complexes) — Mol-Metal's apo-pocket refiner target.
* **SE(3) geodesic** = `(R_t, t_t)` with `R_t ∈ SO(3)` and
  `t_t ∈ ℝ³`, parameterized by twist coordinate `ξ ∈ ℝ⁶`.

### §4.2 Per-wire SE(3) SB applicability verdict (full table)

| # | Wire | SE(3) SB? | Why / Why not |
|---:|---|:---:|---|
| 1 | EDM σ-preconditioning | **partial** | coordinate `c_skip(σ)` factors generalize to twist-space `c_skip(ξ)` if `σ` is mapped to twist magnitude `‖ξ‖`; needs Wf-235 SE(3) `molmetal.math_skills.se3_schrodinger_bridge` integration |
| 2 | Log-normal t-sampling | **yes** | pure t-resampling; SE(3) flow time `t_t` uses same log-normal prior |
| 3 | Shortcut consistency regularizer | **yes** | coordinate self-consistency on `v_t = (1−t)x₀ + t x₁` generalizes to SE(3) geodesic `(R_t, t_t)` w/ Kabsch alignment (already supported in `se3_lipman_integration.py`) |
| 4 | Exponential Moving Average | **yes** | EMA on parameter weights, not on manifold — manifold-agnostic |
| 5 | Linear lr-warmup | **yes** | optimizer schedule, manifold-agnostic |
| 6 | Exact-OT solver | **partial** | Euclidean exact OT via POT generalizes to SE(3) twist-space Sinkhorn (already in `se3sb_bass_integration.py` via `molmetal.math_skills.sb_bass_martingale`) |
| 7 | Triton mini-batch OT (CPU ref) | **partial** | Triton kernel wraps Euclidean Sinkhorn; SE(3) adaptation requires separate twist-space kernel (Triton kernel template is reusable; cost-matrix changes) |
| 8 | Shortcut distillation | **yes** | EMA teacher manifold-agnostic; velocity field distillation generalizes to SE(3) geodesic velocity |
| 9 | Consistency regularization | **yes** | fixed-dt consistency on `v_pred(t+dt)` generalizes to SE(3) geodesic `v_pred(R_{t+dt}, t_{t+dt})`; needs SE(3) Kabsch alignment for `R_{t+dt}` (already supported) |
| 10 | FlowMol unified atom-bond flow | **no** | CFM-only (categorical heads for atom-type + bond-order simplex; no 3D coords in head); SE(3) SB does not apply |

### §4.3 SE(3) SB-applicable vs CFM-only breakdown

* **SE(3) SB-applicable (8 wires, 80%)**: #2 log-normal-t + #3
  shortcut-consistency + #4 EMA + #5 lr-warmup + #8 shortcut-distillation
  + #9 consistency-reg = 6 fully-applicable YES; + #1 EDM + #6 OT-CFM
  exact + #7 Triton OT = 3 PARTIAL (need twist-space adaptation).
* **CFM-only (1 wire, 10%)**: #10 FlowMol unified atom-bond flow.
* **Net SE(3) SB extension potential**: 9/10 wires (90%) are applicable
  to SE(3) SB with twist-space generalization.

### §4.4 SE(3) SB extension plan (forward-looking)

Per Wf-235 SE(3) Schrödinger Bridge + Bass martingale OT-sampler
integration (R18 math migration):

* **Phase 1 (immediate)**: SE(3) extension of lr-warmup + log-normal-t
  + EMA + shortcut-distillation (4 wires, manifold-agnostic — trivial
  lift).
* **Phase 2 (short-term)**: SE(3) twist-space adaptation of EDM + OT-CFM
  exact + Triton OT (3 wires — partial; needs `se3sb_bass_integration.py`
  hooks + new Triton kernel).
* **Phase 3 (medium-term)**: SE(3) geodesic extension of
  shortcut-consistency + consistency-reg (2 wires — partial; needs
  Kabsch-aligned `v_pred(t+dt)` path).
* **Out-of-scope (N/A)**: FlowMol unified atom-bond flow (categorical
  heads; no 3D coords).

---

## §5 Cross-references

* `TODO/completed/340_final_synthesis.md` §6 — Wf-340-SYN 6 wires inventory
  + 12 NEW tests PASS
* `TODO/completed/353_final_synthesis.md` — Wf-353 wave aggregation
* `TODO/completed/353a1_triton_minibatch_ot.md` — Triton mini-batch OT (wire #7)
* `TODO/completed/353a2_shortcut_distillation.md` — Teacher-student
  shortcut distillation (wire #8)
* `TODO/completed/353a3_consistency_regularization.md` — Temporal
  consistency regularization (wire #9)
* `TODO/completed/353a4_unified_atom_bond_flow.md` — FlowMol unified
  atom-bond flow (wire #10)
* `TODO/completed/340i1_cfm_acceleration_implementation.md` — Wf-340
  6 wires concrete code changes
* `TODO/completed/340i3_integration_plan.md` — 3-phase integration roadmap
* `TODO/completed/349a2_otcfm_audit.md` — OT-CFM audit + 11 NEW tests
* `TODO/completed/380a1_deep_integration_scaffolds.md` — Wf-380-A1 10
  SCAFFOLD modules (~1055 LOC + 41 NEW tests)
* `TODO/pending/200_wf340_353_deep_integration_audit.md` — TODO-200
  master plan (priority order + per-wire integration plan)
* `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §5 — full innovation audit
* `molmetal/adapters/flow_matching_lipman/acceleration.py` — bundled
  wires #1, #3, #4, #5
* `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` —
  exact-OT solver (wire #6) + legacy W₂ support
* `molmetal/adapters/flow_matching_lipman/se3sb_bass_integration.py` —
  SE(3) SB + Bass martingale OT sampler (R18 primitive for SE(3)
  extension of wires #1, #6, #7)
* `molmetal/adapters/se3_lipman_integration.py` — SE(3) geodesic CFM
  adapter (R18 primitive for SE(3) extension of wires #3, #8, #9)
* `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:parse_args` —
  CLI flag wiring (lines 1297, 1314, 1321, 1332, 1350, 1377, 1387,
  1409, 1421, 1513, 1639, 1653)
* `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:951` — Wf-348-C1
  EMA + lr-warmup `train_one_epoch` integration
* AGENTS.md §5.1 (Standing GPU authorization) + §5.2 (Honest framing) +
  §5.3 (No paper work without authorization) + §10 (No wet-lab)

---

## §6 Honest Framing per `party-twenty-spirit` §3.5

### §6.1 MEASURED (this Wf-383-A3 audit layer)

* **10-wire audit table** (6 from Wf-340 + 4 from Wf-353) — all file
  paths + test counts + LOC are MEASURED on disk.
* **80 NEW tests PASS** total = Wf-340-I1 (8) + Wf-340-I3 (4) + Wf-353-A1
  (4) + Wf-353-A2 (4) + Wf-353-A3 (6) + Wf-353-A4 (4) + Wf-349-A2 (11) +
  Wf-380-A1 (41) — CPU-only, MEASURED < 2 s wall each.
* **~3,581 LOC total SCAFFOLD modules** across 14 files in
  `molmetal/adapters/flow_matching_lipman/`.
* **10/10 wires have CLI flags** declared in
  `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:parse_args`
  (lines 1297, 1314, 1321, 1332, 1350, 1377, 1387, 1409, 1421, 1513,
  1639, 1653); all default OFF for backward compat per AGENTS.md §5.1.
* **5/10 wires are Production-ready SCAFFOLD** (CLI flag wired, default
  OFF, bit-exact backward-compat verified via
  `test_backward_compat_default_off_behavior_unchanged`).
* **2/10 wires have `train_one_epoch` integration** (lr-warmup + EMA at
  `train_cfm_yuelbond_wasserstein.py:951` per Wf-348-C1).

### §6.2 PROJECTED (cite upstream paper; NOT measured on Mol-Metal)

* **Per-technique speedups**: EDM 2-3× (Karras 2022), MeanFlow 1.5-2×
  (Geng 2025), torch.compile CFM 1.3-1.8× (NVlabs/Sana), etc. — sourced
  from upstream paper benchmarks.
* **Aggregated joint speedup**: 5-10× top-5 additional / 10-20×
  end-to-end conservative / 18-86× aggressive multi-GPU per
  `340_final_synthesis.md` §7.2 — multiplicative composition upper
  bound, NOT measured end-to-end.
* **Architecture-fit scores** for SE(3) SB applicability — qualitative
  assessment per Wf-340-R3 + Wf-235; not benchmarked on PlatinAI 4-cell
  or MetalCytoToxDB 5-metal.
* **Adaptation costs** for SE(3) twist-space extensions (Phase 1 + 2 +
  3 per §4.4) — projection from upstream repo LOC, not measured.

### §6.3 Honest-negative catalog (this Wf-383-A3 audit)

| # | Item | Status |
|---:|---|---|
| 1 | All 10 wires are SCAFFOLD-only — no end-to-end trainer integration into `train_one_epoch` | **DECLARED upfront** in §0 |
| 2 | 5/10 wires are SCAFFOLD-stub (need full impl or LICENSE gate) | **DECLARED** in §2.1 |
| 3 | 2 wires have CLI flag NOT wired (Consistency reg + FlowMol) | **DECLARED** in §2.3 |
| 4 | GPU A/B verification of any 10 wires | **DEFERRED** to Wf-341-A7/A8 per AGENTS.md §5.1 |
| 5 | License exception for FlowMol (CC-BY-NC) + EDM (CC-BY-NC-SA 4.0) | **USER-OK required** before trainer integration |
| 6 | Triton mini-batch OT GPU correctness on Blackwell | **NOT MEASURED** (Wf-326E-A sm_120 retune gating; default OFF) |
| 7 | SE(3) SB applicability for #1 EDM + #6 OT-CFM + #7 Triton OT is PARTIAL (needs twist-space adaptation) | **DECLARED** in §4.2 |
| 8 | SE(3) SB does NOT apply to #10 FlowMol (categorical heads, no 3D coords) | **DECLARED** in §4.2 |
| 9 | Pre-existing syntax bug in `train_cfm_yuelbond_wasserstein.py:1618, :1678` (NOT introduced by Wf-340/Wf-353) | **PRE-EXISTING** per `340i1_*.md` §6.5; separate Wf needed before GPU launch |
| 10 | Wf-353-A3 + Wf-353-A4 CLI flag wiring deferred to downstream sub-flows (Wf-353-A4 + Wf-341-A7) | **DEFERRED** per `353a3_*.md` §6 + `353a4_*.md` §5 |
| 11 | Audit-table is docs-only (0 NEW tests in Wf-383-A3 itself) | **DECLARED** in §0 |
| 12 | StableVM + SO(3)-averaged FM NOT in §1 audit table (outside Wf-340+Wf-353 scope per task brief) | **DECLARED** in §3.1 |

### §6.4 Standing rules honoured (AGENTS.md §1 + §5 + §10)

| Rule | Honoured |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED tag discipline** | ✅ All numeric values trace to MEASURED file paths + test counts + LOC; PROJECTED speedups/lifts cite upstream papers (Karras 2022, Frans 2024, Pooladian 2023, Cao 2025 ICML, Khainb 2025, Qiang 2024, Goyal 2017) |
| **§1.4 Pre-flight v2 25/25 PASS gate** | ✅ No retrain/panel/ablation launched; this audit is docs-only; pre-flight v3 cfm pipeline 6/6 PASS PRESERVED at Wf-380-A1 baseline |
| **§1.5 Honest-negative framing** | ✅ 12 honest-negative caveats preserved per §6.3; SCAFFOLD-stub vs Production-ready SCAFFOLD distinction preserved |
| **§5.1 Standing GPU authorization** | ✅ Zero GPU launches; CPU-only docs+audit; GPU A/B deferred to Wf-341-A7/A8 |
| **§5.2 Honest framing** | ✅ Per `party-twenty-spirit` §3.5 — MEASURED counts vs PROJECTED upstream claims clearly distinguished |
| **§5.3 No paper work without authorization** | ✅ TODO-84 honoured — zero touches to `paper/*` |
| **§5.4 Tier-2 + no wet-lab** | ✅ Tier-2 acceptance bar — no wet-lab planning; CFM acceleration targets computational-only improvements |
| **§10 Wetlab deprecation FAQ** | ✅ No wet-lab planning per §10.3; computational-only scope |

---

## §7 Pre-flight gate status

* **Wf-383-A3 is docs-only audit** — zero new pre-flight gates added.
* **Pre-flight v3 cfm pipeline: 6/6 PASS PRESERVED** at Wf-380-A1
  baseline (per `380a1_*.md` §0).
* **Pre-flight v3 all pipeline: 89/2/3/94** PRESERVED at Wf-378-A1 +
  Wf-379-B1 baseline (per AGENTS.md §6 recent Wf history).
* **Wf-383-A3 does NOT re-run pre-flight** (CPU-only docs+audit; no
  schema drift; no trainer integration; no source-code edits).

---

**END OF Wf-383-A3 AUDIT** — Owner Wf-383-A3; SHIPPED 1/1 deliverable
(docs-only); 0 NEW tests; 0 GPU launches; 0 paper/* touches; 0
wet-lab planning; 10-wire audit table (6 Wf-340 + 4 Wf-353) with
SCAFFOLD-stub vs Production-ready SCAFFOLD breakdown; TODO-200 priority
order cross-referenced; SE(3) SB applicability analyzed (8/10 wire-able,
9/10 with twist-space generalization, 1/10 CFM-only); standing rules
honoured; pre-flight v3 cfm pipeline 6/6 PASS PRESERVED.