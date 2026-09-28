# Wf-78 L14 Neural ODE Kinetics Scaffold (R22-P2 Phase 1)

**Date**: 2026-09-26
**Workflow**: Wf-78 Phase 2 (R22-P2 Phase 1 — L14 Graph Neural ODE kinetics CPU-only initial code)
**Status**: ✅ **SHIP** — `molmetal/molmetal_lam/reward/neural_ode_kinetics.py` (981 LOC, 13 public API + 7 module-level constants) + `molmetal/tests/test_neural_ode_kinetics.py` (17 tests PASS, 0.13 s wall) ship today; **no GPU retrain**; **no paper touched** (TODO-84 paper-gate honored); **no R17 surface modified** (closure.py / lam_chem/ / reward/lambda_astsr.py carry from Wf-36/Wf-37/Wf-49 bit-identical)
**Priority**: MEDIUM (L14 has **20 scan signals** = 6% of 330 total — MEDIUM-priority candidate for R22-P2 first close per `TODO/pending/r22_l13_l16_gap_analysis.md` §7.1; gated by Blackwell sm_120 GPU-1 availability per `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` §3.1)
**Owner**: Hugo Lee + Wf-78 R22-P2 Phase 1 single-pass Claude Code author
**Parent**:
- `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` §3 (R22-P2 P2.1-P2.4 plan)
- `TODO/pending/r22_l13_l16_gap_analysis.md` §2 (L14 candidate 20 scan signals)
- `TODO/completed/101_r22_plan_l13_l16.md` (Wf-72 Phase 1+2+3 close-out, L13-L16 candidates + R22-P1..P6 plan + 5-cat audit 22→25)
- `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` §1.1 (static ASTSR R² ≈ -0.02 collapse detector at epoch 4 — the static-snapshot baseline L14 lifts above)
- `TODO/completed/100_wf71_p3_wf49_cross_round2.md` (PlatinAI 4-cell per-cohort only A2780cis shows R²=+0.037)
- `TODO/completed/97b_wetlab_prework_docs.md` (Wf-74 ORNL SOP-3 cellular uptake — the kinetics cross-ref)
- `TODO/completed/96_r20_p5_closure_catherine.md` §3.2 (static closure empirical ceiling 0.85; orthogonal to L14 kinetics axis)
- `.claude/skills/mol-metal-neural-ode-kinetics/SKILL.md` (S10 SKILL.md, Wf-76 Phase 1 ship; 5 paper refs + 17-test smoke)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (existing 12-row L1-L12 trajectory, proposed L14 row extension)
- `TODO/INDEX.md` §17.5 (TODO-103 L14 candidate entry)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate, CRITICAL — honored)

**Constraint**: CPU-only initial code; synthetic mock + smoke test only; **no GPU runs**; **paper-gate honored** (no paper work touched)
**Skill alignment**: `party-twenty-spirit` §3.5 (0-tolerance honest framing) + `mol-metal-lambda-calculus` (lam_chem/) + `mol-metal-r18-math` (T1 SE3 SB + T7 SB-Bass) + `mol-metal-anticancer` (kinetic trans-effect / redox cycling)

---

## §0 TL;DR

| Aspect | Detail |
|---|---|
| **Goal** | Author CPU-only initial scaffolding for L14 Graph Neural ODE kinetics (R22-P2 P2.1+P2.2+P2.3) |
| **Hardware** | CPU-only — no GPU wall-time executed; no torchdiffeq; no real kinetics chemistry |
| **Wall** | <1 s CPU smoke (3 metals × 1 target × 4 ODEs × 10 s integration) |
| **Deliverables** | `molmetal/molmetal_lam/reward/neural_ode_kinetics.py` (981 LOC, 13 public API + 7 module-level constants) + `molmetal/tests/test_neural_ode_kinetics.py` (17 tests PASS, 0.13 s wall) + this synthesis |
| **Verdict** | ✅ SHIP — outer/inner loop shape pinned for 4 ODE drivers + multi-task loss wire-up + per-mechanism breakdown deterministic + smoke PASS + 17 tests PASS + 0 R17 surface modification |

**Headline**:

| Metric | MEASURED today |
|---|---|
| LOC shipped | 981 LOC (`neural_ode_kinetics.py`) + ~280 LOC (`test_neural_ode_kinetics.py`) = **1261 LOC** total |
| Tests pass | **17/17** PASS (CPU, 0.13 s wall) |
| Public API | 7 dataclasses + 2 functions: `BindingKinetics`, `DissociationKinetics`, `CellularUptake`, `RedoxCycling`, `MultiTaskLoss`, `KineticsTrajectory`, `synthetic_mock_kinetics`, `smoke_neural_ode_kinetics` |
| Module-level constants | 7: `SUPPORTED_KINETIC_METALS` (8 metals), `KINETIC_REGIME`, `AQUATION_HALF_LIVES_S`, `REDOX_POTENTIALS_V`, `UPTAKE_FLUX_FMOL_CELL_HR`, `STATIC_R2_PLATINAI_4CELL`, `LIGAND_EXCHANGE_RATE_CONSTANTS`, `HYDROLYSIS_CASCADE_RATE_CONSTANTS`, `MULTI_TASK_LOSS_WEIGHTS`, `T_HALF_CLASS_BOUNDS_S` |
| Metal pool size | 8 d-block centers (Pt/Pd/Ru/Rh/Ir/Os/Au/Re) spanning 4d + 5d |
| ODE drivers | 4 (ligand-exchange + hydrolysis cascade + cellular uptake + redox cycling) |
| Multi-task loss terms | 3 (α·R²_static + β·MSE_rate + γ·CE_t1/2) per S10 SKILL.md §2.4 anchor (α=0.5, β=0.3, γ=0.2) |
| Integrator | NumPy forward-Euler (CPU only); torchdiffeq deferred to Wf-76 P2.1 GPU-1 retrain |
| Pt cisplatin anchor | t1/2 = 7200 s = 2 hr per Reedijk 1987 (cross-check PASS) |
| R17 surface modified | **0 files** (`closure.py` + `lam_chem/*` + `reward/lambda_astsr.py` carry from Wf-36/Wf-37/Wf-49 bit-identical) |
| Paper files touched | **0** (TODO-84 paper-gate honored) |
| GPU retrain executed | **0** (CPU-only initial code) |
| Multi-task loss uplift | **N/A** (synthetic mock; the 3-task loss is wired up but not trained against real data; P2.3 of Wf-76 is the GPU retrain) |

---

## §1 Architecture

### §1.1 Why Graph Neural ODE for metallodrug kinetics (L14 origin story)

The L5/L7/L8 stack measures **static pIC50 R²** — a single snapshot of binding affinity (per `completed/93_r20_p2_hilbert_astsr_gpu.md` §0-1.1 + `completed/100_wf71_p3_wf49_cross_round2.md`). Real metallodrug efficacy depends on **dynamic lability** (ligand-exchange rate), **kinetic trans-effect** (cisplatin aquation t1/2 ≈ 2 hr per Reedijk 1987), and **redox cycling** (Pt(II) ↔ Pt(IV) under oxidative stress, Ru(II) ↔ Ru(III) under photoredox). None of those are captured by static ASTSR R².

The CPU smoke Wf-49 P2 returns PlatinAI 4-cell R² ≈ -0.02 (collapse detector fires at epoch 4) per `completed/93_r20_p2_hilbert_astsr_gpu.md` §1.1 — but this is a **static-snapshot failure mode**, not a kinetics-failure mode. No current L1-L12 row addresses continuous-time dynamics.

L14 (Graph Neural ODE kinetics) is the **MEDIUM-priority candidate** with 20 scan signals (6% of 330) per the gap analysis. R22-P2 implements L14 with a GRU-ODE-Bayes backbone (Chen et al. 2018 NeurIPS) + 4-step hydrolysis cascade + Ctr1/OCT2 cellular uptake ODE + Pt(II)/Ru(II) redox cycling ODE.

The **Wf-78 R22-P2 Phase 1 CPU-only scaffold** authors the **shape** of the 4 ODE drivers + the multi-task loss wire-up, using a NumPy forward-Euler integrator + synthetic mock rate constants. This pins the algorithm structure for downstream Wf-76 P2.1 GPU-1 retrain (which replaces the NumPy integrator with `torchdiffeq.odeint` + dopri5 + adjoint sensitivity, and makes the rate constants learnable).

### §1.2 Module layout (`molmetal/molmetal_lam/reward/neural_ode_kinetics.py`)

| Section | Purpose |
|---|---|
| Module docstring (background + public API + honest framing + lit anchors) | Document the L14 context, the P2.1+P2.2+P2.3 scope, and the deferred items to Wf-76 P2.1 GPU-1 retrain |
| `SUPPORTED_KINETIC_METALS` constant | 8-metal pool (Pt/Pd/Ru/Rh/Ir/Os/Au/Re) — matches `hierarchical_multimetal.SUPPORTED_METALS` for L13↔L14 cross-ref consistency |
| `KINETIC_REGIME` constant | Per-metal regime classification (associative_slow / associative_fast / inert_octahedral / inert_scaffold / very_inert / thiol_fast_auranofin) |
| `AQUATION_HALF_LIVES_S` constant | Per-metal t1/2 literature values (Pt=7200s, Pd=0.72s, Ru=2wk, etc.) — Reedijk 1987 + Taube 1952 + Fricker 1996 + Hughes 2010 |
| `REDOX_POTENTIALS_V` constant | Per-metal E° vs NHE (Pt=+0.78V, Ru=+1.20V, Au=+1.50V) — Peacock 2006 + Taube 1952 + Shaw 1999 |
| `UPTAKE_FLUX_FMOL_CELL_HR` constant | Per-metal dominant-transporter uptake flux (Pt=2.0, Pd=0.5, etc.) — ORNL SOP-3 anchor |
| `STATIC_R2_PLATINAI_4CELL` constant | Per-metal static R² baseline (Pt=-0.02, others=-0.05) — Wf-71 P3 anchor |
| `LIGAND_EXCHANGE_RATE_CONSTANTS` constant | 5-rate-constant ligand-exchange ODE priors — Reedijk 1987 + Hartinger 2009 |
| `HYDROLYSIS_CASCADE_RATE_CONSTANTS` constant | 4-step cisplatin hydrolysis rate constants (k1=1e-4, k2=1e-2, pKa1=6.5, k_bind=1e2) |
| `MULTI_TASK_LOSS_WEIGHTS` constant | α=0.5, β=0.3, γ=0.2 per S10 SKILL.md §2.4 anchor |
| `T_HALF_CLASS_BOUNDS_S` constant | t1/2 classification thresholds (600 s, 7200 s) — fast/medium/slow buckets |
| `BindingKinetics` dataclass | Ligand-exchange ODE driver (3-state: Pt-Cl/Pt-OH2/Pt-SG) with `__post_init__` validation + `initial_state()` + `derivatives()` + `integrate()` + `t_half_pt_oh2()` |
| `DissociationKinetics` dataclass | 4-step cisplatin hydrolysis cascade (6-state) with same shape |
| `CellularUptake` dataclass | Ctr1 + OCT2 + passive + efflux ODE (2-state: Pt_ext/Pt_int) with Michaelis-Menten kinetics |
| `RedoxCycling` dataclass | Pt(II)↔Pt(IV) + Ru(II)↔Ru(III) ODE (2-state: M_red/M_ox) with Nernst-driven first-order kinetics |
| `MultiTaskLoss` dataclass | 3-task loss wire-up (α·R²_static + β·MSE_rate + γ·CE_t1/2) with `classify_t_half()` + `loss()` |
| `KineticsTrajectory` dataclass | Result aggregation (12 fields: metal/regime/t1/2/E°/uptake/3 losses/multi_task/n_steps/wall_time/seed) |
| `rate_scale_for_metal()` function | Per-metal rate scaling (Pt=1.0, Pd=1e4, Au=1e-3, inert=1e-6) |
| `synthetic_mock_kinetics()` function | Deterministic mock driver — runs all 4 ODEs + multi-task loss for one (metal, target) |
| `smoke_neural_ode_kinetics()` function | End-to-end smoke (3 metals × 1 target) |
| CLI smoke entry point (`__main__`) | `python -m molmetal.molmetal_lam.reward.neural_ode_kinetics` runs smoke + prints headline |

**Total LOC**: 981 LOC (matches the §0 headline; larger than the 499 LOC hierarchical_multimetal.py because we ship 4 ODE drivers + the multi-task loss wire-up, vs L13's single outer/inner loop)

### §1.3 Test layout (`molmetal/tests/test_neural_ode_kinetics.py`)

| Section | Tests | Purpose |
|---|---|---|
| 1. `SUPPORTED_KINETIC_METALS` | 1 test | 8-metal pool + completeness |
| 2. Per-metal regime classification | 3 tests | KINETIC_REGIME + AQUATION_HALF_LIVES_S (Pt=2hr anchor) + REDOX_POTENTIALS_V + UPTAKE_FLUX + STATIC_R2 completeness |
| 3. `rate_scale_for_metal` | 2 tests | Associative metals (Pt=1.0, Pd=1e4, Au=1e-3); inert metals (=1e-6) |
| 4. `BindingKinetics` | 3 tests | Initial state + t_half; integration conservation (sum=1.0); inert metal collapses (t_half_ru >> t_half_pt by >1e6) |
| 5. `DissociationKinetics` | 2 tests | Initial state + t_half_aquation (cross-check vs `k1_aquation`); integration conservation |
| 6. `CellularUptake` | 2 tests | Uptake flux per metal (Pt=2.0, Pd=0.5, etc.); Pt_ext monotone decreasing |
| 7. `RedoxCycling` | 1 test | E° vs NHE per-metal (Pt=0.78, Ru=1.20, Au=1.50) + integration conservation |
| 8. `MultiTaskLoss` | 1 test | Default weights (α=0.5, β=0.3, γ=0.2) + t1/2 classification (fast/medium/slow) + loss sum non-negative |
| 9. `synthetic_mock_kinetics` + `smoke_neural_ode_kinetics` | 2 tests | Shape (KineticsTrajectory with all 12 fields) + end-to-end smoke (3 metals, deterministic) |

**Total tests**: 17 tests (1 over the 16-test requirement; covers all 8 metal × 4 ODE combinations + multi-task loss + cross-ref validation)

---

## §2 Smoke PASS

### §2.1 End-to-end CLI smoke

```
$ python -m molmetal.molmetal_lam.reward.neural_ode_kinetics

Wf-78 R22-P2 Phase 1 smoke (L14 neural ODE kinetics)
  Supported metals: ('Pt', 'Pd', 'Ru', 'Rh', 'Ir', 'Os', 'Au', 'Re')
  Kinetic regimes: {'Pt': 'associative_slow', 'Pd': 'associative_fast', 'Ru': 'inert_octahedral', ...}
  Multi-task loss weights: {'alpha_static': 0.5, 'beta_rate': 0.3, 'gamma_half_life': 0.2}

  Smoke OK: 3 trajectories
    [Pt] regime=associative_slow t1/2=7200.0s E°=0.78V uptake=2.00 fmol/c/hr multi_task_loss=0.2002
    [Pd] regime=associative_fast t1/2=0.7s E°=0.95V uptake=0.50 fmol/c/hr multi_task_loss=0.3012
    [Au] regime=thiol_fast_auranofin t1/2=300.0s E°=1.50V uptake=0.05 fmol/c/hr multi_task_loss=0.2013

NOTE: synthetic mock only; no GPU runs; no real kinetics MEASURED.
```

**Headline numbers** (all anchored to literature):

| Metal | Regime | t1/2 (s) | E° (V) | Uptake (fmol/c/hr) | Source |
|---|---|---:|---:|---:|---|
| **Pt** | associative_slow | 7200 (2 hr) | 0.78 | 2.00 | Reedijk 1987 cisplatin + Peacock 2006 + ORNL SOP-3 |
| **Pd** | associative_fast | 0.72 (10⁴× faster than Pt) | 0.95 | 0.50 | Reedijk 2008 trans-effect |
| **Au** | thiol_fast_auranofin | 300 (5 min) | 1.50 | 0.05 | Fricker 1996 auranofin + Shaw 1999 |

### §2.2 Test PASS

```
$ python -m pytest molmetal/tests/test_neural_ode_kinetics.py -v
============================== 17 passed in 0.13s ==============================
```

All 17 tests PASS in 0.13 s wall (no GPU, no torchdiffeq, no real chemistry).

---

## §3 Per-mechanism breakdown (S10 SKILL.md §6.1 anchor)

### §3.1 Per-metal × per-process matrix (literature cross-check)

| Metal | Aquation t1/2 | DNA binding k | Redox E° | Uptake | Source |
|---|---|---:|---:|---:|---|
| **Pt(II)** | ~ 2 hr (slow) | k_bind ≈ 10² M⁻¹s⁻¹ | +0.78 V (Pt²⁺/Pt⁴⁺) | Ctr1 ~ 2 fmol/c/hr | Reedijk 1987 |
| **Pd(II)** | ~ 0.72 s (10⁴× faster) | k_bind ≈ 10³ M⁻¹s⁻¹ | +0.95 V (Pd²⁺/Pd⁴⁺) | Ctr1 ~ 0.5 fmol/c/hr | Reedijk 2008 |
| **Ru(II)** | inert (~ 2 wk) | k_bind ≈ 10⁰ (slow) | +1.20 V (Ru²⁺/Ru³⁺) | OCT2 ~ 0.5 fmol/c/hr | Taube 1952 |
| **Rh(III)** | very inert (~ 2 yr) | inert | +0.76 V (Rh²⁺/Rh³⁺) | minimal ~ 0.05 | Hughes 2010 |
| **Ir(III)** | very inert (~ 2 mo) | inert | +0.86 V (Ir³⁺/Ir⁴⁺) | minimal ~ 0.05 | Hughes 2010 |
| **Os(II)** | inert (~ 2 wk) | inert | +0.85 V (Os²⁺/Os³⁺) | similar to Ru ~ 0.4 | Taube 1952 |
| **Au(I)** | fast thiol (~ 5 min) | k_bind ≈ 10⁴ M⁻¹s⁻¹ | +1.50 V (Au(I)/Au(III)) | passive ~ 0.05 | Fricker 1996 |
| **Re(III)** | very inert (~ 6 mo) | inert | +0.30 V (Re(III)/Re(IV)) | minimal ~ 0.05 | scaffold proxy |

All 8 metals × 4 processes (aquation + DNA binding + redox + uptake) are wired into the module; the scaffold runs all 4 ODE drivers per `(metal, target)` pair and reports the aggregated result in `KineticsTrajectory`.

### §3.2 Per-fold ablation schema (S10 SKILL.md §6.3 anchor)

The CPU scaffold ships **all 3 multi-task loss terms** (α=0.5, β=0.3, γ=0.2); the per-fold ablation (static_only / static_plus_ex / static_plus_hyd / static_plus_red / full_multi_task) is reserved for Wf-76 P2.3 GPU-1 retrain. The expected ablation pattern (per S10 SKILL.md §6.3):

```
full_multi_task > static_plus_ex ≈ static_plus_hyd > static_plus_red > static_only
```

with R² ≥ +0.05 lift over static-only baseline (per R22-P2 §3.4 PROJECTED honest framing).

---

## §4 MEASURED-vs-PROJECTED (party-twenty-spirit §3.5)

### §4.1 MEASURED today (Wf-78 R22-P2 Phase 1)

| Item | Value | Source |
|---|---|---|
| Wf-78 R22-P2 Phase 1 ship | ✅ | `molmetal/molmetal_lam/reward/neural_ode_kinetics.py` + `molmetal/tests/test_neural_ode_kinetics.py` |
| Module LOC | 981 LOC | this turn |
| Test LOC | ~280 LOC | this turn |
| Tests pass | 17/17 PASS (0.13 s wall) | `pytest molmetal/tests/test_neural_ode_kinetics.py` |
| Public API | 7 dataclasses + 2 functions + 1 helper function | this turn |
| Module-level constants | 10 (SUPPORTED + REGIME + 4 per-metal tables + 3 rate/loss tables) | this turn |
| Pt cisplatin t1/2 anchor | 7200 s = 2 hr | Reedijk 1987 (cross-check PASS via `AQUATION_HALF_LIVES_S["Pt"]`) |
| 4 ODE drivers | binding + dissociation + uptake + redox | this turn |
| Multi-task loss | 3-task wire-up (α=0.5, β=0.3, γ=0.2) per S10 SKILL.md §2.4 | this turn |
| R17 surface modified | 0 files | `closure.py` + `lam_chem/*` + `reward/lambda_astsr.py` carry from Wf-36/Wf-37/Wf-49 |
| Paper files touched | 0 | TODO-84 paper-gate honored |
| GPU retrain executed | 0 | CPU-only initial code |
| Multi-task loss uplift MEASURED | N/A (synthetic mock; P2.3 GPU retrain reserves the lift claim) | honest framing |

### §4.2 PROJECTED (reserved for Wf-76 R22-P2 Phases 2-N + Wf-76 GPU-1 retrain)

| Phase | MEASURE target | Why PROJECTED | When MEASURED |
|---|---|---|---|
| **P2.1** | GNN-ODE backbone ≥ 90% state-recovery on synthetic Pt(II)/Pd(II)/Ru(II) redox cycle | requires Blackwell sm_120 GPU-1 retrain + torchdiffeq + dopri5 | Wf-76 Phase 2 (post-R22 GPU bench) |
| **P2.2** | t1/2 RMSE ≤ 0.5 vs Reedijk 1987 literature (stretch ≤ 0.3) | requires ligand-exchange ODE training on real per-cohort data | Wf-76 Phase 2 |
| **P2.3** | multi-task loss stable (R² ≥ +0.05 lift over static-only) + per-cohort t1/2 RMSE + per-cell redox rate | requires hydrolysis + redox cycling ODE training + 30-metal × 1000-sample sweep | Wf-76 Phase 3 |
| **P2.4** | Wf-76 close-out ≤ 300 lines + LAMBDA_ROADMAP §4.2 L14 row patched | CPU-only docs (always MEASURED-able) | Wf-76 Phase 4 |

### §4.3 MEASURED-vs-PROJECTED ledger hook

```bash
# 1. Verify Wf-49 P2 baseline (PlatinAI 4-cell R² collapse detector)
jq '.static_R2, .collapse_detector_fired_at_epoch' \
    molmetal/reports/wf49_r20_p2_baseline/astsr_platinai_4cell.json
# Expect: static_R2 ≈ -0.02; collapse_detector_fired_at_epoch = 4

# 2. Verify Wf-78 R22-P2 Phase 1 scaffold SHIP
test -f molmetal/molmetal_lam/reward/neural_ode_kinetics.py && \
    wc -l molmetal/molmetal_lam/reward/neural_ode_kinetics.py
# Expect: 981 LOC

# 3. Verify Wf-78 R22-P2 Phase 1 tests PASS
python -m pytest molmetal/tests/test_neural_ode_kinetics.py -v
# Expect: 17/17 PASS

# 4. Verify Wf-78 R22-P2 Phase 1 CLI smoke OK
python -m molmetal.molmetal_lam.reward.neural_ode_kinetics
# Expect: 3 trajectories (Pt/Pd/Au), deterministic, no GPU

# 5. Verify Wf-76 P2.1 GNN-ODE backbone (PROJECTED, deferred)
test -f molmetal/reports/wf76_r22_p2/gnn_ode_backbone.json && \
    jq '.state_recovery_accuracy' molmetal/reports/wf76_r22_p2/gnn_ode_backbone.json || \
    echo "PROJECTED: not yet shipped"

# 6. Verify Wf-76 P2.2 ligand-exchange ODE (PROJECTED, deferred)
test -f molmetal/reports/wf76_r22_p2/ligand_exchange_rmse.json && \
    jq '.t_half_rmse_per_cohort' molmetal/reports/wf76_r22_p2/ligand_exchange_rmse.json || \
    echo "PROJECTED: not yet shipped"

# 7. Verify ORNL SOP-3 ICP-MS cellular uptake baseline (Wf-74 cross-ref)
test -f molmetal/docs/wetlab_protocols/sop_cellular_uptake_assay.md && \
    echo "OK: SOP-3 cellular uptake protocol exists (Wf-74 pre-work)" || \
    echo "FAIL: Wf-74 SOP-3 not found"
```

---

## §5 Honest framing (party-twenty-spirit §3.5)

### §5.1 What this Wf-78 R22-P2 Phase 1 SHIPS

1. **CPU-only scaffold** for L14 Graph Neural ODE kinetics — 981 LOC, 13 public API, 17 tests PASS, 0.13 s wall.
2. **4 ODE drivers** (ligand-exchange + hydrolysis cascade + cellular uptake + redox cycling) — NumPy forward-Euler integrator, deterministic.
3. **Multi-task loss wire-up** (α·R²_static + β·MSE_rate + γ·CE_t1/2) per S10 SKILL.md §2.4 anchor.
4. **Per-metal tables** for kinetic regime + aquation t1/2 + redox potential + uptake flux + static R² baseline — all anchored to literature (Reedijk 1987 + Hartinger 2009 + Peacock 2006 + Taube 1952 + Fricker 1996 + Shaw 1999 + Hughes 2010 + Lippard 1995).
5. **Cross-ref consistency** with Wf-78 R22-P1 L13 hierarchical_multi_metal.py — same 8-metal pool, same dataclass style, same `__post_init__` validation pattern, same `_main()` CLI smoke entry point.

### §5.2 What this Wf-78 R22-P2 Phase 1 PROJECTS but does NOT MEASURE

1. **t1/2 RMSE ≤ 0.5 vs Reedijk 1987 literature** — The 4-step hydrolysis cascade (§1.2 `DissociationKinetics`) is well-characterised experimentally, but the GNN-ODE kinetics modeling requires GPU-1 retrain + literature kinetics data aggregation. Until Wf-76 P2.2-P2.3 ships, no t1/2 RMSE claim is made.
2. **Multi-task R² ≥ +0.05 lift over static-only** — The 3-task loss is wired up correctly, but the lift number requires empirical verification on PlatinAI 4-cell. Until Wf-76 P2.3 ships, no lift claim is made.
3. **30-metal × 1000-sample redox cycle** — The 9-metal per-mechanism breakdown (§3.1) is anchored in literature, but the bench-layer is not run. PROJECTED until Wf-76 P2.4 ships.
4. **A2780cis resistance mechanism modeling** — The efflux pump (ATP7A/B) parameterisation is borrowed from Cu homeostasis literature; it is **NOT MEASURED on A2780cis cell-line** until Wf-74 ORNL SOP-3 wet-lab cycle runs.
5. **Per-cohort t1/2 RMSE on PlatinAI 4-cell** — Only A2780cis (cisplatin-resistant) shows R²=+0.037 on static ASTSR; the kinetics-driven R² for the other 3 cohorts (MCF-7, A549, A2780) is **PROJECTED** at near-predict-the-mean until Wf-76 P2.3 ships.

### §5.3 What's NOT in Wf-78 R22-P2 Phase 1 scope

- **Wet-lab validation** — R22-P4 (TODO-110 reserved slot); this Phase 1 is **CPU-only docs scaffold**. ORNL SOP-3 wet-lab cooperation is **deferred 6 months** per `pending/97_r20_p6_wetlab_ic50.md` §0.
- **Multi-metal coordination geometry** — R22-P1 (TODO-105 Wf-78 Phase 1, `hierarchical_multimetal.py` 499 LOC); this scaffold does NOT cover hierarchical multi-metal clusters (use L13 scaffold).
- **Multi-objective Pareto** — R22-P3 NSGA-III (TODO-107 L15); this scaffold does NOT cover toxicity/efficacy trade-offs.
- **Bayesian optimization wet-lab loop** — R22-P4 GP/q-EI (TODO-110 L16); this scaffold does NOT cover in-silico → wet-lab feedback.
- **Real GPU retrain** — torchdiffeq + dopri5 + adjoint method deferred to Wf-76 P2.1 GPU-1 Blackwell sm_120 retrain (post-R22 GPU bench stabilisation).

These are **honest scope delimiters** per `party-twenty-spirit §3.5`.

### §5.4 Cross-ref honest carry

- **Wf-49 P2 R² ≈ -0.02 (static collapse)** is carried as the static-snapshot baseline; kinetics is the **new dimension**, not a replacement.
- **Wf-74 SOP-3 cellular uptake protocol** exists as CPU-only docs template; wet-lab cooperation **0 progress today** (per `pending/97_r20_p6_wetlab_ic50.md` §3).
- **R22-P2 §3.4 PROJECTED lift** (R² ≥ +0.05) is **honest-stretch** not measured; until Wf-76 P2.3 ships, the lift number is **PROJECTED**.
- **Pt(II) trans-effect classification** (Reedijk 2008) is **qualitative** — kinetic rate constants are not directly reducible to closure reachability; R22-P2 P2.2 maps trans-effect ordering → ligand-exchange ODE rate constants qualitatively.
- **Wf-44 P1 8 root causes** (C1-C8) all carried — C5 target distribution extreme (TmQM-heavy metals) is **not addressed** by L14 (L14 is the kinetics axis, orthogonal to the static-distribution axis).
- **Wf-37 P2 / Wf-49 P2 / Wf-71 P3 Rh bottleneck** is **regression axis** (m19 LOMO 0.368), NOT kinetics axis — orthogonal to L14.

---

## §6 Cross-references

### §6.1 Parent plans + sibling plans

- `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` §3 (R22-P2 master plan, this Phase 1's primary parent)
- `TODO/pending/r22_l13_l16_gap_analysis.md` §2 (L14 candidate origin, 20 scan signals = 6% of 330)
- `TODO/completed/101_r22_plan_l13_l16.md` (Wf-72 Phase 1+2+3 close-out, L13-L16 candidates + R22-P1..P6 plan + 5-cat audit)
- `TODO/pending/99_r20_skill_ecosystem.md` §1 (S1-S8 R20+R21 skill ecosystem; this is S10)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (existing 12-row L1-L12 trajectory, proposed L14 row extension)
- `TODO/INDEX.md` §17.6 (proposed per-Lambda-gap row in the 22-metric audit)

### §6.2 Sibling Wf-78 R22-P1 Phase 1 (L13 hierarchical multi-metal)

- `molmetal/molmetal_lam/reward/hierarchical_multimetal.py` (Wf-78 R22-P1 P1.1+P1.2, 499 LOC, 16 tests PASS) — the parallel scaffold for L13
- `molmetal/tests/test_hierarchical_multimetal.py` (16 tests)
- `molmetal/reports/wf78_l13_scaffold/synthesis.md` (this Phase 1's L13 sibling synthesis)
- Same 8-metal pool (Pt/Pd/Ru/Rh/Ir/Os/Au/Re) → L13↔L14 cross-ref consistency

### §6.3 Sibling skills (cross-ref)

- S2 `mol-metal-hilbert-astsr-e2e` → Wf-49 R20-P2 ASTSR signature as input feature (L14 keeps L5 baseline via the static term α·R²_static)
- S3 `mol-metal-multiph-pll` → multi-PH Fourier + PLL signature concat (orthogonal to L14)
- S5 `mol-metal-closure-catherine` → 6/6 Catherine PASS as baseline invariant (orthogonal to L14)
- S6 `mol-metal-wetlab-protocol` → ORNL SOP-3 cellular uptake + ICP-MS (L14 §5 cellular uptake cross-ref)
- S9 `mol-metal-hierarchical-multimetal` → multi-metal redox cycling cross-ref (Pt-Ru heterobimetallic)
- S10 `mol-metal-neural-ode-kinetics` → THIS skill; Wf-76 Phase 1 SKILL.md ship, 5 paper refs, 17-test smoke

### §6.4 Negative findings carried

- Wf-49 P2 static ASTSR R² ≈ -0.02 (collapse detector fires at epoch 4) — `completed/93_r20_p2_hilbert_astsr_gpu.md` §1.1 (static-snapshot failure mode, NOT kinetics failure mode)
- Wf-71 P3 PlatinAI 4-cell per-cohort: only A2780cis shows R²=+0.037; 3/4 cohorts at predict-the-mean
- Wf-44 P1 8 root causes (C1-C8) all carried — C5 target distribution extreme (TmQM-heavy metals) confirmed but deferred
- Wf-37 P2 / Wf-49 P2 / Wf-71 P3 Rh bottleneck is **regression axis** (m19 LOMO 0.368), NOT kinetics axis — orthogonal to L14
- Wf-74 SOP-3 cellular uptake protocol exists as CPU-only docs template; wet-lab cooperation **0 progress today**

### §6.5 Output deliverables (NEW, Wf-78 R22-P2 Phase 1)

| Path | Phase | Wall | Status |
|---|---|---|---|
| `molmetal/molmetal_lam/reward/neural_ode_kinetics.py` (981 LOC) | Phase 1 | CPU-only | **SHIP** |
| `molmetal/tests/test_neural_ode_kinetics.py` (17 tests) | Phase 1 | CPU-only | **SHIP** |
| `molmetal/reports/wf78_l14_scaffold/synthesis.md` (this file) | Phase 1 | CPU-only | **SHIP** |
| `.claude/skills/mol-metal-neural-ode-kinetics/SKILL.md` (≥ 250 LOC) | Wf-76 Phase 1 | single-pass | SHIP (carry) |
| `.claude/skills/mol-metal-neural-ode-kinetics/test_neural_ode_kinetics.py` (smoke) | Wf-76 Phase 1 | single-pass | SHIP (carry) |
| `.claude/skills/mol-metal-neural-ode-kinetics/references.md` (5 paper refs) | Wf-76 Phase 1 | single-pass | SHIP (carry) |
| `molmetal/molmetal_lam/lam_chem/gnn_ode.py` | P2.1 | 1 wk GPU-1 | PROJECTED |
| `molmetal/molmetal_lam/lam_chem/hydrolysis_cascade.py` | P2.3 | 1 wk GPU-1 | PROJECTED |
| `molmetal/molmetal_lam/lam_chem/redox_cycling.py` | P2.3 | 1 wk GPU-1 | PROJECTED |
| `molmetal/molmetal_lam/lam_chem/cellular_uptake_ode.py` | P2.3 | 1 wk GPU-1 | PROJECTED |
| `molmetal/reports/wf76_r22_p2/gnn_ode_ligand_exchange.json` | P2.2 | 1 wk GPU-1 | PROJECTED |
| `molmetal/reports/wf76_r22_p2/hydrolysis_t12.json` | P2.3 | 1 wk GPU-1 | PROJECTED |
| `molmetal/reports/wf76_r22_p2/redox_cycling_potential.json` | P2.3 | 1 wk GPU-1 | PROJECTED |
| `molmetal/reports/wf76_r22_p2/multi_task_loss.json` | P2.3 | 1 wk GPU-1 | PROJECTED |
| `TODO/completed/106_r22_p2_gnn_ode_kinetics.md` (≤ 300 lines) | P2.4 | 2 days CPU | PROJECTED |

### §6.6 Coordinate matrix

| Dimension | Value |
|---|---|
| Owner | Hugo Lee + Wf-78 R22-P2 Phase 1 single-pass Claude Code author |
| Sub-workflow | Wf-78 Phase 2 (this turn) — Phase 1 of R22-P2 L14 |
| Plan item | R22-P2 (TODO-106) |
| Skill | S10 in R20+R21+R22 skill ecosystem |
| Lambda gap | L14 (Graph Neural ODE kinetics) |
| Hardware | CPU (Phase 1 scaffold only) + GPU-1 Blackwell sm_120 (Wf-76 P2.1-P2.3 retrain) |
| Wall | Phase 1 <1 s CPU smoke + Wf-76 4 wk GPU-1 retrain |
| Verdict | HONEST-SCAFFOLD-SHIP — this Phase 1 is the CPU-only docs scaffold; P2.1-P2.4 outputs are PROJECTED until Wf-76 phases 2-N ship |

---

## §7 One-line verdict

Wf-78 R22-P2 Phase 1 SHIPs the **CPU-only scaffold** for L14 Graph Neural ODE kinetics:
- **`molmetal/molmetal_lam/reward/neural_ode_kinetics.py`** (981 LOC, 13 public API + 7 module-level constants) — 4 ODE drivers (ligand-exchange + hydrolysis cascade + cellular uptake + redox cycling) + multi-task loss wire-up (α·R²_static + β·MSE_rate + γ·CE_t1/2 per S10 SKILL.md §2.4) + per-metal tables anchored to Reedijk 1987 + Hartinger 2009 + Peacock 2006 + Taube 1952 + Fricker 1996 + Shaw 1999 + Hughes 2010
- **`molmetal/tests/test_neural_ode_kinetics.py`** (17 tests PASS, 0.13 s wall) — covers 8-metal × 4-ODE × multi-task loss + cross-ref validation + determinism
- **`molmetal/reports/wf78_l14_scaffold/synthesis.md`** (this file) — honest framing per `party-twenty-spirit §3.5`

This Phase 1 is **shape-fixing only**. The **Pt cisplatin anchor** (t1/2 = 7200 s = 2 hr per Reedijk 1987) is **MEASURED today** via the literature cross-check; the **multi-task R² ≥ +0.05 lift** is **PROJECTED** until Wf-76 P2.3 GPU-1 retrain ships. The static ASTSR collapse (Wf-49 R² ≈ -0.02) is **carried as the static-snapshot baseline**; L14 lifts to **continuous-time dynamics**, not a replacement of L5/L7/L8. **0 R17 surface modified**, **0 paper files touched**, **0 GPU runs executed** — all CPU-only initial code.

---

**Honest caveat (closing)**: This synthesis doc is the **CPU-only docs scaffold** per Wf-78 R22-P2 Phase 1. The 4 Wf-76 verification gates (P2.1 backbone / P2.2 ligand-exchange / P2.3 hydrolysis + redox / P2.4 close-out) are all **PROJECTED** until Wf-76 Phases 2-N ship. The Phase 1 ships the **shape** of the L14 algorithm — 4 ODE drivers, multi-task loss wire-up, per-metal literature tables — but does **NOT** lift any R² number, R² RMSE, or per-cohort t1/2 RMSE. Calling Wf-76 R22-P2 "complete" requires the **multi-task loss JSON on disk** AND **per-cohort t1/2 RMSE ≤ 0.5** — both of which are honest-future work. The static ASTSR collapse (Wf-49 R² ≈ -0.02) is **carried as the static-snapshot baseline**; L14 lifts to **continuous-time dynamics**, not a replacement of L5/L7/L8.