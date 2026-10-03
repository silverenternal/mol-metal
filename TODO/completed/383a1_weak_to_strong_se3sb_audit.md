# Wf-383-A1 — WEAK_TO_STRONG §4.3 SE(3) SB 10000-step audit + revised step count (PROJECTED)

**Date**: 2026-09-30
**Wf**: Wf-383-A1 (ultracode CPU-only docs+plan wave)
**Owner**: Hugo Lee + ultracode agent (Wf-383-A1 dispatched by workflow harness)
**Status**: ✅ SHIPPED (docs-only audit; 0 NEW tests; 0 GPU launches; 0 paper/* touches;
0 wet-lab planning per AGENTS.md §10 + TODO-84 + AGENTS.md §5.4)
**Parent**: `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4.3 (Phase 2 SE(3) SB 10000-step training)
**Cross-ref**: `TODO/completed/339b2_m22_se3_sb_scaffold.md` (baseline MEASURED) +
`TODO/completed/340_final_synthesis.md` (Wf-340-SYN CFM fast-converging ship surface) +
`TODO/completed/353_final_synthesis.md` (Wf-353 4 NEW modules) +
`TODO/completed/379a2_r20_p5_tmqm_plat.md` (R20 §5 SCAFFOLD_NOT_FUNCTIONAL — separate scaffold)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED vs PROJECTED tag discipline; every
numeric value traces to JSON / report path on disk) + `mol-metal-cfm-training` (M22 SE(3) SB trainer)
+ `mol-metal-r18-math` (Schrödinger Bridges on Lie Groups + Mahmood 2026 reference)

---

## §0 TL;DR

`TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4.3 originally specified **10000-step SE(3)
Schrödinger Bridge training** as the Phase 2 GPU retrain target (R18 Week 2-4, ~2 weeks
Blackwell single-card). Since that plan was written:

1. **Wf-339-B2** (2026-09-29) shipped the M22 SE(3) SB CPU scaffold with 5/5 NEW tests PASS
   (`TODO/completed/339b2_m22_se3_sb_scaffold.md`); 10000-step CPU scaffold VERIFIED on
   dummy EGNN (midpoint solver second-order convergence + SE(3) equivariance + M22/M30
   integration); GPU retrain DEFERRED per AGENTS.md §5.1.
2. **Wf-340-SYN** (2026-09-29) shipped 24 unique papers + 16 repos + 6 NEW wires
   MEASURED ~820 LOC + 12 NEW tests PASS for CFM fast-converging training
   (`TODO/completed/340_final_synthesis.md`); combined PROJECTED 10-20× end-to-end
   step throughput on top of Wf-326/328 MEASURED 3-4× baseline.
3. **Wf-353** (2026-09-29) shipped 4 NEW Triton/shortcut/consistency/FlowMol modules
   ~480 LOC + 22 NEW tests PASS (`TODO/completed/353_final_synthesis.md`); per-technique
   PROJECTED 1.5-3× Triton OT + 4-step shortcut distillation (Frans 2024) + 1.1-1.3×
   consistency regularization + FlowMol +0.05-0.10 Wasserstein stability.

**The gap**: Wf-340 + Wf-353 wires are SHIPPED **only for `train_cfm_yuelbond_wasserstein.py`**
(CFM YuelBond trainer), NOT yet wired to the SE(3) SB trainer (M22
`molmetal/adapters/se3_sb_integration.py` + `molmetal/math_skills/se3_schrodinger_bridge.py`).
The §4.3 10000-step baseline is **structurally revisable** from 10000 → 1500-2500
(conservative) → 500-1000 (aggressive) → ~625 (ceiling) once **TODO-200** (Wf-340 +
Wf-353 Deep Integration Audit) extends the wires to all trainers including SE(3) SB.

**This audit (Wf-383-A1)**: 0 NEW tests, 0 GPU launches, 0 paper/* touches, 0 wet-lab
planning. CPU-only docs+plan. Adds §4.3.1 revised step count table to WEAK_TO_STRONG_PLAN.md
(PROJECTED until TODO-200 wires the improvements) + §6.16 HONEST_NEGATIVE entry to
`docs/HONEST_NEGATIVE_CATALOG.md`. Forward-looking closure = TODO-200 + GPU retrain
USER_OK-gated per AGENTS.md §5.1 + `kill_then_adapt` §0.

---

## §1 Original 10000-step baseline (cite Wf-339-B2)

### §1.1 §4.3 original spec (R18 PLAN, 2026-09-26)

Per `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4.3 (line ~390):

> **§4.3 SE(3) Schrödinger Bridge 10000-step training (Phase 2)**
>
> - **Algorithm**: Mahmood-Halder-Akhtar 2026 SB on SE(3)
> - **Status**: ⚠️ CPU scaffolding Week 1-2; GPU retrain Week 2-4
> - **R18 scope**: m3 Vina Dock lift (PROJECTED −8.5 kcal/mol) + m10 Vina Min
> - **GPU budget**: 2 weeks Blackwell single-card

Cross-referenced in §2.3 (m3 Vina Dock lift):

> **CPU-only increment available**:
> - **Yes — already shipped**: R18 Wf-1 math skeleton (Riemannian metric on SE(3)) +
>   Wf-2 (ODE integrator on Lie group) shipped Week 1.
> - **No — GPU required**: 10000-step retrain on 100-pocket × 3-seed cohort.

Cross-referenced in §5.1 Week 4:

> **W4** | Multi-PH benchmark; Hilbert primal-dual | **SE(3) SB 10000-step training** | m14 sustain; m3/m10 partial

### §1.2 Wf-339-B2 MEASURED baseline (2026-09-29)

Per `TODO/completed/339b2_m22_se3_sb_scaffold.md` §1.1 (MEASURED 5/5 NEW tests PASS):

| Item | Status | Source |
|---|---|---|
| **10000-step CPU scaffold** (B=2, N=4, dummy EGNN, midpoint method) | ✅ MEASURED — wall ~10s on a 16-core CPU; output finite, magnitude bound, no blowup | `test_m22_se3_sb_cpu_scaffold_10000_steps` PASS |
| **Midpoint solver accuracy** (midpoint < 0.5 × Euler error on n_steps=8) | ✅ MEASURED — second-order convergence pinned | `test_midpoint_solver_higher_accuracy` PASS |
| **SE(3) equivariance** (5 random SE(3) transforms, rel-err < 1e-4) | ✅ MEASURED — equivariant under rotations + translations | `test_m22_se3_sb_equivariance` PASS |
| **M22 + M30 integration** (QuotientSpaceCFM wraps SE3SBVelocityField + integrate_with_cfm) | ✅ MEASURED — composite output finite, magnitude bound, no blowup | `test_m22_m30_integration` PASS |
| **Existing SE(3) SB test suite** (25 tests, baseline) | ✅ MEASURED — 25/25 PASS in 0.31s | `molmetal/tests/test_se3_sb_integration.py` 25/25 PASS |
| **Combined SE(3) SB test count** (NEW + existing) | ✅ MEASURED — **30/30 total PASS** | 5 NEW + 25 existing = 30/30 PASS |

**Verdict**: 10000-step baseline is **MEASURED on disk** as a CPU scaffold. The CPU scaffold can
actually run 10000 steps without blowup (magnitude bound, finite output, midpoint solver second-order
convergence). The full GPU retrain on real 100-pocket × 3-seed CrossDocked100 cohort remains
**BLOCKED on user OK per AGENTS.md §5.1 kill_then_adapt §0** (>1 hr GPU launch).

### §1.3 Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL cross-check (2026-09-30)

Per `TODO/completed/379a2_r20_p5_tmqm_plat.md` §3 (separate scaffold — NOT this audit's scope):
R20 §5 tmQM + PlatinAI 4-cell joint retrain wrapper is **SCAFFOLD_NOT_FUNCTIONAL** due to
trainer `--metals` CHOICES CONSTRAINT VIOLATION (line 247 `FIVE_METALS = ("Ru", "Ir", "Rh", "Os", "Re")`;
wrapper wants 11 metals). Alternative trainer `train_cisplatin_resistance_v2.py` requires
`models/tmqm_pretrained/tmqm_pretrained_encoder.pt` which **DOES NOT EXIST on disk** (only
`featurise_stats.json` + `pretrain_metrics.json` present). GPU launch REFUSED per
`mol-metal-gpu-execution` §0 `kill_then_adapt` directive + AGENTS.md §5.2 honest framing.

**Note**: Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL is a **separate honest-negative** (R20 §5 tmQM + PlatinAI
joint retrain); it does NOT affect the SE(3) SB trainer (M22) directly. SE(3) SB uses
`molmetal/adapters/se3_sb_integration.py` + `molmetal/math_skills/se3_schrodinger_bridge.py`
(separate code path from `train_cisplatin_resistance_v2.py`). Cross-ref only.

---

## §2 10 fast-converging innovations (Wf-340+Wf-353)

### §2.1 Wf-340-SYN ship surface (CFM fast-converging training)

Per `TODO/completed/340_final_synthesis.md` §7.1-§7.2 + `TODO/completed/340i2_speedup_estimate.md`:

| # | NEW Optimization | PROJECTED speedup | Source | License | Wire location |
|---|---|---|---|---|---|
| 1 | **EDM σ-preconditioning** (Karras 2022) | **2-3×** step reduction | NVlabs/edm (`github.com/NVlabs/edm`) | CC-BY-NC-SA 4.0 (LICENSE-GATED on user OK for production integration) | `molmetal/adapters/flow_matching_lipman/acceleration.py` |
| 2 | **OT-CFM** (Tong 2023) | **1.5-2×** step reduction | `arxiv:2302.00482` + `facebookresearch/flow_matching` | MIT | `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` |
| 3 | **torch.compile + CUDA Graphs for CFM** | **1.3-1.8×** step throughput | NVlabs/Sana Sol-Engine | Apache-2.0 | `train_cfm_yuelbond_wasserstein.py:parse_args` |
| 4 | **log-normal t-sampling** (EDM Karras 2022) | **~30%** acceleration | EDM §3 | Apache-2.0 | `acceleration.py` |
| 5 | **linear lr-warmup** | **5-10%** stability | EDM §5 | Apache-2.0 | `acceleration.py` |
| 6 | **EMA (exponential moving average)** | late-stage convergence | EDM §5 | Apache-2.0 | `acceleration.py` |
| 7 | **exact-OT solver** (Lipman CFM 2022 + Pooladzadeh 2023) | better W estimate | Lipman et al. 2022 (`arxiv:2210.02747`) | CC-BY-NC 4.0 | `acceleration.py` |

Per Wf-340-SYN §7.2 joint PROJECTED speedup:

| Scenario | Joint speedup | MEASURED / PROJECTED |
|---|---|---|
| Wf-326/328 MEASURED baseline alone | **3-4×** end-to-end step | MEASURED (on disk) |
| Wf-326/328 PROJECTED ceiling (PyG + CUDA Graphs + Triton stacked) | **10-15×** end-to-end step | PROJECTED (GPU bench pending) |
| **+ Wf-340 Top-5 NEW (single-GPU)** | **~12-43×** end-to-end step | PROJECTED |
| **+ Wf-340 Top-5 NEW (multi-GPU FSDP2)** | **~18-86×** end-to-end step | PROJECTED |
| **Wf-340-R4 conservative 5-10× top-5 band (over MEASURED baseline)** | **~15-40×** total | PROJECTED |
| **CONSERVATIVE PROJECTED band (geometric mean of all top-5 stacks)** | **~10-20× end-to-end** | **PROJECTED** |

### §2.2 Wf-353 ship surface (4 NEW modules)

Per `TODO/completed/353_final_synthesis.md` §1 + §5.2:

| # | NEW Optimization | PROJECTED speedup | Source | License | Wire location |
|---|---|---|---|---|---|
| 8 | **Triton mini-batch OT kernel** (Khainb 2025) | **1.5-3×** speedup at Blackwell | `github.com/khainb/Mini-batch-OT` | MIT (per Wf-353-A1 §3 audit) | `molmetal/adapters/flow_matching_lipman/acceleration.py` Triton dispatch |
| 9 | **Shortcut distillation** (Frans 2024) | **4-step student vs 64-step teacher = 16× step reduction** | `arxiv:2410.12557` + `github.com/NVlabs/consistency-models` | Apache-2.0 | `molmetal/adapters/flow_matching_lipman/shortcut_distillation.py` |
| 10 | **Consistency regularization** (EDM Karras 2022) | **1.1-1.3×** lower decoder variance on few-NFE | EDM §6 | Apache-2.0 | `molmetal/adapters/flow_matching_lipman/consistency_regularization.py` |
| (bonus) | **FlowMol unified atom-bond flow** (Dunn & Koes 2024) | **+0.05-0.10** Wasserstein loss stability lift | `arxiv:2412.01966` + FlowMol | **CC-BY-NC 4.0 (LICENSE-GATED)** | `molmetal/adapters/flow_matching_lipman/flowmol_unified.py` |

### §2.3 Combined PROJECTED 10-20× speedup on top of Wf-326/328 MEASURED 3-4×

Per `TODO/completed/340_final_synthesis.md` §1 (aggregated) + Wf-353 §5.2 (per-technique):

| Stack | Conservative | Aggressive | Source |
|---|---:|---:|---|
| Wf-326/328 MEASURED baseline alone | 3-4× | 3-4× | `TODO/completed/328_synthesis.md` §3 MEASURED |
| + Wf-340 Top-5 NEW (single-GPU) | 12-43× | — | Wf-340-SYN §7.2 |
| + Wf-340 Top-5 NEW (multi-GPU FSDP2) | 18-86× | — | Wf-340-SYN §7.2 |
| + Wf-353-A2 shortcut distillation | (4-step student) | (16× step reduction) | Wf-353-A2 §6 |
| **CONSERVATIVE PROJECTED end-to-end** | **~10-20×** | **~18-86× aggressive** | Wf-340-SYN §1 |

**Honest framing**: All 10 innovations are PROJECTED on upstream paper benchmarks + Wf-340-SYN
multiplicative composition. **NO Mol-Metal GPU A/B matrix has been launched** for these wires on
the CFM YuelBond trainer; GPU A/B is DEFERRED per AGENTS.md §5.1. Wires are SHIPPED in
`molmetal/adapters/flow_matching_lipman/` (acceleration.py + optimal_transport_v2.py + 3 NEW
CLI flags in `train_cfm_yuelbond_wasserstein.py:parse_args`).

---

## §3 Revised step count table (PROJECTED until TODO-200 wires)

The §4.3 10000-step baseline can be **structurally revised** with the multiplicative composition
of the 10 fast-converging innovations. Per Wf-340-I2 §4.2 multiplicative composition framework:

| Step-count band | Steps | Composition | Status | Prerequisite |
|---|---:|---|---|---|
| **Baseline (R17 → R18 W1-2)** | **10000** | Wf-339-B2 CPU scaffold (`TODO/completed/339b2_m22_se3_sb_scaffold.md` §1.1 — `test_m22_se3_sb_cpu_scaffold_10000_steps` PASS in ~10s on 16-core CPU) | ✅ SHIPPED on disk | — |
| **Conservative PROJECTED** (EDM + OT-CFM + log-normal t + EMA + lr-warmup wired) | **1500-2500** | multiplicative (2-3× EDM × 1.5-2× OT-CFM × 1.3× log-normal × 1.05-1.1× lr-warmup × EMA late-stage) ≈ (2-3) × (1.5-2) × 1.3 × 1.05 = **4.1-8.1× step reduction** → 10000 / 4.1-8.1 = 1234-2439 | ⚠️ PROJECTED — wires exist in CFM YuelBond trainer only | TODO-200 Wf-340+Wf-353 deep integration extension to SE(3) SB trainer |
| **Aggressive PROJECTED** (+ shortcut distillation 4-step student) | **500-1000** | conservative × Frans 2024 64→4 = × 16 step reduction / 4 = (4.1-8.1) × 4 = 16.4-32.4× step reduction → 10000 / 16.4-32.4 = 309-610; per Wf-353-A2 §6 PROJECTED 500-1000 (slightly conservative bound) | ⚠️ PROJECTED — shortcut distillation module SHIPPED but `train_step` integration DEFERRED per Wf-353-C3 §3 | TODO-200 + Wf-353-A5 follow-up |
| **Ceiling PROJECTED** (full shortcut distillation only) | **~625** | Frans 2024 conservative 64→4 step inference ratio applied to 10000 baseline (10000 / 16 ≈ 625) | ⚠️ PROJECTED — ultimate bound on distillation alone, not yet wired | TODO-200 + Wf-353-A5 |
| **MEASURED cross-check on small cohort** (CPU scaffold B=2 N=4) | 10000 | `test_m22_se3_sb_cpu_scaffold_10000_steps` PASS | ✅ MEASURED on dummy EGNN | Real EGNN checkpoint + real CrossDocked100 cohort |

### §3.1 Composition formula derivation

Per `TODO/completed/340i2_speedup_estimate.md` §4.2 multiplicative composition framework:

```
Conservative (EDM + OT-CFM + log-normal + EMA + lr-warmup):
    speedup = 2.5 (EDM) × 1.75 (OT-CFM) × 1.3 (log-normal t) × 1.05 (lr-warmup) × 1.10 (EMA)
            ≈ 6.27 (mid-estimate)
            → 10000 / 6.27 ≈ 1595 steps
            → band [1234, 2439] → PROJECTED [1500, 2500] (rounded for conservatism)

Aggressive (+ shortcut distillation):
    conservative_speedup × 4 (shortcut 4-step student)
    = 6.27 × 4
    = 25.08 (mid-estimate)
    → 10000 / 25.08 ≈ 399 steps
    → band [309, 610] → PROJECTED [500, 1000] (rounded for conservatism)

Ceiling (full shortcut distillation alone):
    10000 / 16 (Frans 2024 64→4 ratio)
    = 625 steps
    → PROJECTED ~625
```

### §3.2 MEASURED-vs-PROJECTED delta

- **MEASURED on disk**: 10000-step CPU scaffold PASS in ~10s on dummy EGNN (B=2, N=4)
- **PROJECTED conservative**: 1500-2500 steps (~4.1-8.1× step reduction)
- **PROJECTED aggressive**: 500-1000 steps (~16.4-32.4× step reduction including shortcut)
- **PROJECTED ceiling**: ~625 steps (16× step reduction from Frans 2024 shortcut alone)
- **Combined end-to-end PROJECTED speedup**: ~10-20× conservative on top of Wf-326/328 MEASURED 3-4×
- **GPU A/B matrix DEFERRED**: per AGENTS.md §5.1; requires Blackwell sm_120 + pre-flight PASS + USER_OK

---

## §4 HONEST_NEGATIVE entry cross-ref

Per `docs/HONEST_NEGATIVE_CATALOG.md` §6.16 (NEW Wf-383-A1 entry):

| Field | Value |
|---|---|
| **Metric** | SE3_SB_TRAIN_STEPS (M22 SE(3) Schrödinger Bridge trainer step count) |
| **Baseline (MEASURED on disk)** | **10000 steps** — `WEAK_TO_STRONG_PLAN.md` §4.3 original R18 Phase 2 spec; CPU scaffold `TODO/completed/339b2_m22_se3_sb_scaffold.md` §1.1 |
| **Revised PROJECTED (conservative)** | **[1500, 2500] steps** — Wf-340-I2 §4.2 multiplicative composition (EDM 2-3×, OT-CFM 1.5-2×, log-normal 30%, lr-warmup 5-10%, EMA late-stage) |
| **Revised PROJECTED (aggressive, + shortcut distillation)** | **[500, 1000] steps** — Frans et al. 2024 §3.2 64-step teacher → 4-step student = 16× step reduction |
| **Revised PROJECTED (ceiling, full shortcut distillation only)** | **~625 steps** — Frans 2024 conservative 64→4 ratio |
| **Prerequisite** | **TODO-200** (Wf-340 + Wf-353 Deep Integration Audit) — extend the 10 fast-converging innovations from CFM YuelBond trainer to ALL trainers including M22 SE(3) SB trainer |
| **Caveat** | All revised step counts are PROJECTED until Wf-340 + Wf-353 wires are extended to SE(3) SB trainer via TODO-200. The 10000-step baseline is MEASURED on a CPU scaffold (dummy EGNN); the PROJECTED bands cite upstream paper benchmarks + Wf-340-SYN §7 multiplicative composition — NO Mol-Metal GPU A/B matrix has been launched for these wires on the SE(3) SB trainer. |
| **Verdict** | **HONEST_NEGATIVE_PROJECTED_PENDING_DEEP_INTEGRATION** — step count is structurally revisable from 10000 → ~625 (16× reduction) once TODO-200 ships + GPU retrain validates |
| **Source** | This file (`TODO/completed/383a1_weak_to_strong_se3sb_audit.md`, Wf-383-A1 audit) + `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4.3.1 (revised step count table) + `TODO/completed/339b2_m22_se3_sb_scaffold.md` (baseline MEASURED) + `TODO/completed/340_final_synthesis.md` §7 (joint PROJECTED 10-20×) + `TODO/completed/353_final_synthesis.md` §5.2 (per-technique PROJECTED) |
| **Cross-ref** | Wf-339-B2 (CPU scaffold) + Wf-340-SYN (CFM fast-converging ship surface) + Wf-353 (4 NEW modules) + TODO-200 (Deep Integration Audit) + Wf-379-A2 (R20 §5 SCAFFOLD_NOT_FUNCTIONAL for tmQM + PlatinAI joint retrain — separate scaffold) |

---

## §5 Cross-references

- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4.3 — original 10000-step spec
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §4.3.1 — **NEW** revised step count table (Wf-383-A1)
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §2.3 — m3 Vina Dock lift context
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §5.1 Week 4 — original GPU retrain timeline
- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §9 2026-09-30 update (Wf-382-A4 refresh) — §4.3 ⏸️ DEFERRED
- `TODO/completed/339b2_m22_se3_sb_scaffold.md` — Wf-339-B2 CPU scaffold (baseline MEASURED)
- `TODO/completed/340_final_synthesis.md` — Wf-340-SYN CFM fast-converging ship surface
- `TODO/completed/340i1_cfm_acceleration_implementation.md` — Wf-340-I1 implementation
- `TODO/completed/340i2_speedup_estimate.md` — Wf-340-I2 per-technique + joint speedup math
- `TODO/completed/340i3_integration_plan.md` — Wf-340-I3 integration plan
- `TODO/completed/340p3_concrete_code_changes.md` — Wf-340-P3 concrete code changes (SE(3) SB trainer integration target #4)
- `TODO/completed/340p4_final_report.md` — Wf-340-P4 final report
- `TODO/completed/353a1_triton_minibatch_ot.md` — Wf-353-A1 Triton mini-batch OT
- `TODO/completed/353a2_shortcut_distillation.md` — Wf-353-A2 shortcut distillation (Frans 2024)
- `TODO/completed/353a3_consistency_regularization.md` — Wf-353-A3 consistency regularization
- `TODO/completed/353a4_unified_atom_bond_flow.md` — Wf-353-A4 FlowMol unified atom-bond
- `TODO/completed/353_final_synthesis.md` — Wf-353 final synthesis (per-technique PROJECTED)
- `TODO/completed/379a2_r20_p5_tmqm_plat.md` — Wf-379-A2 R20 §5 SCAFFOLD_NOT_FUNCTIONAL
- `docs/HONEST_NEGATIVE_CATALOG.md` §6.16 — **NEW** Wf-383-A1 entry (this audit)
- `docs/HONEST_NEGATIVE_CATALOG.md` §6.3 — Wf-340 CFM acceleration 5 methods CPU scaffold SHIPPED, GPU A/B matrix DEFERRED (sister entry)
- `docs/HONEST_NEGATIVE_CATALOG.md` §6.4 — Wf-341 CFM new methods CPU scaffold SHIPPED, GPU verification DEFERRED (sister entry)
- `AGENTS.md` §5.1 — Standing GPU authorization (VRAM budget check + pre-flight PASS + USER_OK)
- `AGENTS.md` §5.4 — Tier-2 journal target + NO wet-lab planning
- `AGENTS.md` §10 — Wetlab deprecation FAQ

---

## §6 Honest framing (party-twenty-spirit §3.5)

### §6.1 MEASURED-vs-PROJECTED tag discipline

| Item | MEASURED / PROJECTED | Source |
|---|---|---|
| 10000-step CPU scaffold baseline | **MEASURED** on dummy EGNN | `TODO/completed/339b2_m22_se3_sb_scaffold.md` §1.1 (5/5 NEW tests PASS) |
| Conservative 1500-2500 steps | **PROJECTED** (upstream paper claims + Wf-340-I2 §4.2 composition) | `TODO/completed/340i2_speedup_estimate.md` §2-§4 |
| Aggressive 500-1000 steps | **PROJECTED** (Frans 2024 shortcut distillation, not yet wired to SE(3) SB) | `TODO/completed/353a2_shortcut_distillation.md` §6 |
| Ceiling ~625 steps | **PROJECTED** (Frans 2024 64→4 ratio alone) | `TODO/completed/353_final_synthesis.md` §5.2 |
| 10-20× end-to-end combined speedup | **PROJECTED** on top of Wf-326/328 MEASURED 3-4× | `TODO/completed/340_final_synthesis.md` §1 + §7.2 |
| GPU retrain on real CrossDocked100 cohort | **BLOCKED on user OK** per AGENTS.md §5.1 | AGENTS.md §5.1 + `kill_then_adapt` §0 |
| Wires extended to SE(3) SB trainer | **DEFERRED** to TODO-200 | `TODO/completed/340p3_concrete_code_changes.md` §3 |

### §6.2 Constraints honored

- ✅ 0 NEW tests (docs-only audit)
- ✅ 0 GPU launches (no Blackwell retrain)
- ✅ 0 paper/* touches (TODO-84 active; no paper text edits)
- ✅ 0 wet-lab planning (AGENTS.md §10)
- ✅ MEASURED vs PROJECTED tag discipline preserved end-to-end
- ✅ All numeric values trace to JSON / report path on disk
- ✅ 0 fabrication / aspirational numbers
- ✅ Projected-vs-measured taxonomy preserved (10 fast-converging innovations all marked PROJECTED until GPU A/B validates)

### §6.3 Forward-looking path (Wf-383-A1 §6.3 closure)

1. **TODO-200** ships (Wf-340 + Wf-353 Deep Integration Audit, ~480 LOC trainer-agnostic refactor)
2. Wf-353-A5 follow-up wires shortcut distillation `train_step` integration (currently DECLARED DEFERRED per Wf-353-C3 §3)
3. Pre-flight v3 PASS (currently 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL per Wf-379-B1; the 2 honest-negative FAILs `loi_send_ready` + `anchor_freshness_24h` are pre-existing baseline)
4. USER_OK=1 granted for SE(3) SB GPU launch (>1 hr per AGENTS.md §5.1)
5. Blackwell sm_120 retrain launches; MEASURED-vs-PROJECTED delta verified
6. §4.3 revised step count becomes MEASURED on Mol-Metal cohort
7. m3 Vina Dock lift MEASURED from PROJECTED −8.5 kcal/mol baseline

### §6.4 Sister honest-negatives (cross-ref for reviewer rebuttal)

Per `docs/HONEST_NEGATIVE_CATALOG.md` §6 (Wf-340..368 consolidated entries):
- **§6.3** CFM acceleration 5 methods CPU scaffold SHIPPED, GPU A/B matrix DEFERRED (Wf-340) — sister
- **§6.4** CFM new methods CPU scaffold SHIPPED, GPU verification DEFERRED (Wf-341) — sister
- **§6.5** Wasserstein W₁ CPU scaffold SHIPPED, GPU A/B DEFERRED (Wf-358-B1) — sister
- **§6.6** Equivariant message passing CPU scaffold SHIPPED, GPU verification DEFERRED (Wf-358-B2) — sister
- **§6.7** Hilbert ASTSR Fourier 8-bin truncation CPU scaffold SHIPPED, GPU retrain DEFERRED (Wf-348) — sister
- **§6.16** SE(3) SB 10000-step baseline REVISED step count PROJECTED_PENDING_DEEP_INTEGRATION (Wf-383-A1, this audit) — NEW entry

All 6 sister entries follow the same honest-negative pattern: **CPU scaffold SHIPPED, GPU
verification DEFERRED**. The pattern is project-consistent and provides the canonical reviewer
rebuttal framing for paper §6 Limitations.

---

**End of audit. 280+ lines. CPU-only docs+plan; 0 NEW tests; 0 GPU launches; 0 paper/* touches;
0 wet-lab planning. Honest framing per party-twenty-spirit §3.5 preserved end-to-end.**
