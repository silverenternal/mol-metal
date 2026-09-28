# Wf-78 L13 Hierarchical Multi-Metal Scaffold (R22-P1 Phase 1)

**Date**: 2026-09-26
**Workflow**: Wf-78 (R22-P1 Phase 1 — L13 hierarchical multi-metal coord CPU-only initial code)
**Status**: ✅ **SHIP** — `molmetal/molmetal_lam/reward/hierarchical_multimetal.py` (499 LOC, 8 public API) + `molmetal/tests/test_hierarchical_multimetal.py` (16 tests PASS) ship today; **no GPU retrain**; **no paper touched** (TODO-84 paper-gate honored); **no R17 surface modified** (closure.py / lam_chem/ carry from Wf-36/Wf-37)
**Priority**: HIGH (L13 has **188 scan signals** = 57% of 330 total — top candidate for R22-P1 first close per `TODO/pending/r22_l13_l16_gap_analysis.md` §7.1)
**Owner**: Hugo Lee + Wf-78 single-pass Claude Code author
**Parent**:
- `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` §2 (R22-P1 P1.1-P1.5 plan)
- `TODO/pending/r22_l13_l16_gap_analysis.md` §1 (L13 candidate 188 scan signals)
- `TODO/completed/101_r22_plan_l13_l16.md` (Wf-72 Phase 1+2+3 close-out, L13-L16 candidates + R22-P1..P6 plan + 5-cat audit 22→25)
- `TODO/completed/96_r20_p5_closure_catherine.md` §3.2 (reachability 0.5962 NOT gate 0.95; structural blockers)
- `TODO/completed/95_r20_p4_3way_100pocket.md` (3-way + 100-pocket L10 baseline)
- `TODO/completed/91_wf41_3way_integration.md` (L10 3-way R²=-0.3595 ± 0.2806 baseline)
- `TODO/completed/87j_wf44_root_cause_fixes.md` C5 (target distribution extreme TmQM-heavy)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (existing 12-row L1-L12 trajectory, proposed L13 row extension)
- `TODO/INDEX.md` §17.5 (TODO-103 L13 candidate entry)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate, CRITICAL — honored)

**Constraint**: CPU-only initial code; synthetic mock + smoke test only; **no GPU runs**; **paper-gate honored** (no paper work touched)
**Skill alignment**: `party-twenty-spirit` §3.5 (0-tolerance honest framing) + `mol-metal-lambda-calculus` (closure.py / lam_chem/) + `mol-metal-r18-math` (M26 Score-Rep) + `mol-metal-click-chemistry` (P1.1 dative-bond SMARTS deferred to Wf-79)

---

## §0 TL;DR

| Aspect | Detail |
|---|---|
| **Goal** | Author CPU-only initial scaffolding for L13 hierarchical multi-metal coordination (R22-P1 P1.1+P1.2) |
| **Hardware** | CPU-only — no GPU wall-time executed; no real `closure_test` invocation; no real metal-coord chemistry |
| **Wall** | <1 s CPU smoke (synthetic mock scorer is SHA-256 hash over (target_id, metal, seed) → [0, 1)) |
| **Deliverables** | `molmetal/molmetal_lam/reward/hierarchical_multimetal.py` (499 LOC, 8 public API) + `molmetal/tests/test_hierarchical_multimetal.py` (16 tests PASS) + this synthesis |
| **Verdict** | ✅ SHIP — outer/inner loop shape pinned + per-metal + per-target breakdown deterministic + smoke PASS + 16 tests PASS + 0 R17 surface modification |

**Headline**:

| Metric | MEASURED today |
|---|---|
| LOC shipped | 499 LOC (`hierarchical_multimetal.py`) + 245 LOC (`test_hierarchical_multimetal.py`) = **744 LOC** total |
| Tests pass | **16/16** PASS (CPU, 0.12 s wall) |
| Public API | 8 classes/funcs: `MetalPool`, `TargetSpec`, `HierarchicalCoordConfig`, `HierarchicalCoordResult`, `HierarchicalCoordinator`, `synthetic_mock_score`, `smoke_hierarchical`, `SUPPORTED_METALS` |
| Metal pool size | 8 d-block centers (Pt/Pd/Ru/Rh/Ir/Os/Au/Re) spanning 4d + 5d |
| Outer depth default | 2 (matches R22-P1 P1.2 hierarchical BFS depth-≥2 spec) |
| Inner-loop truncation | 8 (full pool) — `inner_per_target` config allows truncation |
| R17 surface modified | **0 files** (`closure.py` + `lam_chem/*` carry from Wf-36/Wf-37 bit-identical) |
| Paper files touched | **0** (TODO-84 paper-gate honored) |
| GPU retrain executed | **0** (CPU-only initial code) |
| Reachability gate lifted | **N/A** (synthetic mock; structural blockers unchanged — see §3 for the Wf-79 R22-P1 P1.3 GPU plan) |

---

## §1 Architecture

### §1.1 Why hierarchical (outer × inner loop)

L13 (hierarchical multi-metal coord, 188 scan signals) addresses the **structural reachability ceiling** (0.5744-0.5962 ≪ gate 0.95) on the L1/L6/L12 closure bench by replacing the **single-metal-center closure** with a **two-level hierarchical loop**:

  * **Outer loop** — iterate over :class:`TargetSpec` instances (pocket / scaffold index).  In R22-P1 P1.1 this is **abstract** (just a string `target_id`); Wf-79 P1.3 will plug in a real `(pocket_id, scaffold_smiles)` pair sourced from the 100-pocket CrossDocked bench staged by Wf-30 (`TODO/completed/82_metric_gap_filling.md` §13.6).
  * **Inner loop** — iterate over a configurable :class:`MetalPool` of d-block centers.  Default pool = `SUPPORTED_METALS = (Pt, Pd, Ru, Rh, Ir, Os, Au, Re)` (8 centers spanning 4d + 5d); this matches the closure bench metal subset per `completed/95_r20_p4_3way_100pocket.md` (Ru/Rh/Pd/Os/Ir/Re/Pt/Au).

The P1.1 + P1.2 scaffold runs the outer × inner loop for ``max_outer_depth=2`` iterations (default), emitting a deterministic mock reward per `(target, metal)` pair and aggregating per-metal + per-target breakdowns.  This pins the **shape** of the L13 algorithm without lifting the reachability gate (the structural blockers — valence-BNF dative-bond rejection, tautomerisation redex exhaustion, sample sparsity — remain; P1.3 + P1.4 of Wf-79 R22-P1 are the GPU phases that actually attempt the lift).

### §1.2 Module layout (`molmetal/molmetal_lam/reward/hierarchical_multimetal.py`)

| Section | Lines | Purpose |
|---|---|---|
| Module docstring (background + public API + honest framing + lit anchors) | 1-72 | Document the L13 context, the P1.1+P1.2 scope, and the deferred items to Wf-79 |
| `SUPPORTED_METALS` constant | 78-89 | 8-metal pool constant |
| `MetalPool` dataclass | 92-117 | Immutable d-block metal pool with validation + membership + iteration |
| `TargetSpec` dataclass | 121-138 | Target scaffold spec (abstract for P1.1; Wf-79 P1.3 will replace with `(pocket_id, scaffold_smiles)`) |
| `HierarchicalCoordConfig` dataclass | 142-178 | Config dataclass (`max_outer_depth` / `inner_per_target` / `seed` / `scorer`) with validation |
| `HierarchicalCoordResult` dataclass | 181-241 | Result dataclass (`n_targets` / `n_metals` / `n_pairs` / `mean_inner_reward` / per-metal breakdown / per-target breakdown / wall_time_s / seed / max_outer_depth` + `is_balanced` + `samples_per_pair` accessors) |
| `synthetic_mock_score` function | 245-273 | SHA-256 hash-based deterministic mock scorer (no real chemistry, no GPU) |
| `HierarchicalCoordinator` dataclass | 277-413 | Outer/inner loop runner with `__post_init__` validation + `run()` method that aggregates per-metal + per-target means |
| `smoke_hierarchical` function | 417-440 | End-to-end smoke (3 targets × 8 metals × depth 2 = 48 pairs) |
| CLI smoke entry point (`__main__`) | 445-451 | `python -m molmetal.molmetal_lam.reward.hierarchical_multimetal` runs smoke + prints headline |

**Total LOC**: 499 LOC (matches the §0 headline; lighter than the 669 LOC Hilbert R18 ship because we deferred P1.3 + P1.4 GPU phases to Wf-79)

### §1.3 Test layout (`molmetal/tests/test_hierarchical_multimetal.py`)

| Section | Tests | Purpose |
|---|---|---|
| 1. `MetalPool` | 3 tests | Default contains 8 supported metals; custom subset; validation (empty / non-string rejected) |
| 2. `TargetSpec` | 2 tests | Basic construction; empty `target_id` rejected |
| 3. `HierarchicalCoordConfig` | 3 tests | Defaults; rejects non-positive; custom scorer round-trip |
| 4. `synthetic_mock_score` | 3 tests | Determinism; seed sensitivity (loose — collision 1 in 2**32); validation |
| 5. `HierarchicalCoordinator.run()` | 4 tests | Shape (`n_pairs = n_targets * n_metals * max_outer_depth`); per-metal balanced; custom scorer overrides mock; seed determinism across runs |
| 6. `smoke_hierarchical` | 1 test | End-to-end smoke (3 × 8 × 2 = 48 pairs, deterministic, <5 s) |

**Total tests**: 16 tests (matches the §0 headline)

---

## §2 Smoke PASS

### §2.1 End-to-end CLI smoke

```
$ python -m molmetal.molmetal_lam.reward.hierarchical_multimetal
hierarchical_multimetal smoke: n_targets=3 n_metals=8 n_pairs=48 mean_inner_reward=0.482993 wall_time_s=0.0001
```

**Headline**: 3 targets × 8 metals × depth-2 outer loop = 48 pairs scored deterministically in 0.0001 s wall; mean reward in [0, 1) bounded by the SHA-256 hash normalization.

### §2.2 pytest run

```
$ pytest molmetal/tests/test_hierarchical_multimetal.py -v
molmetal/tests/test_hierarchical_multimetal.py::test_metal_pool_default_contains_supported_set PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_metal_pool_custom_subset PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_metal_pool_rejects_invalid_inputs PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_target_spec_basic_construction PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_target_spec_rejects_empty_id PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_hierarchical_coord_config_defaults PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_hierarchical_coord_config_rejects_nonpositive PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_hierarchical_coord_config_custom_scorer_round_trip PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_synthetic_mock_score_is_deterministic PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_synthetic_mock_score_seed_sensitivity PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_synthetic_mock_score_validation PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_hierarchical_coordinator_run_shape PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_hierarchical_coordinator_per_metal_breakdown_balanced PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_hierarchical_coordinator_custom_scorer_overrides_mock PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_hierarchical_coordinator_seed_determinism PASSED
molmetal/tests/test_hierarchical_multimetal.py::test_smoke_hierarchical_end_to_end PASSED
16 passed in 0.12s
```

**Headline**: 16/16 PASS, 0.12 s wall — all assertions on shape / validation / per-metal breakdown / custom scorer / seed determinism / smoke hold.

### §2.3 Manual sanity check

```
$ python -c "from molmetal.molmetal_lam.reward.hierarchical_multimetal import smoke_hierarchical; \
             r = smoke_hierarchical(n_targets=4, seed=42); \
             print(r.n_targets, r.n_metals, r.n_pairs, r.mean_inner_reward, r.is_balanced, r.wall_time_s)"
4 8 64 0.565434... True 0.000157...
```

**Headline**: 4 targets × 8 metals × depth 2 = 64 pairs (matches `n_pairs = n_targets * n_metals * max_outer_depth`); `is_balanced = True`; wall_time < 1 ms; mean reward in [0, 1).

---

## §3 GPU R22-P1 plan (Wf-79)

### §3.1 What P1.1 + P1.2 ship vs defer

| R22-P1 phase | Status | Detail |
|---|---|---|
| **P1.1 Chemistry-aware dative-bond parser** | ⚠️ DEFERRED to Wf-79 R22-P1 P1.1 | The R22-P1 P1.1 plan item ships `molmetal/molmetal_lam/lam_chem/dative_bond_parser.py` + `tests/test_dative_bond_parser.py` (≥10 tests).  Today's Wf-78 ships the **outer/inner loop shape** that P1.1 will plug into; the parser itself is a separate module with separate tests. |
| **P1.2 Hierarchical M-M closure BFS (depth ≥ 2)** | ✅ SHIPPED (CPU scaffold only) | Today's Wf-78 ships `HierarchicalCoordinator` with `max_outer_depth=2` (matches the depth-≥2 spec).  The mock scorer is a stand-in; Wf-79 P1.2 will replace it with the real `closure_test` from `molmetal/molmetal_lam/lam_chem/closure.py` (R17 ship, Wf-36 P1 verified). |
| **P1.3 Multi-metal cluster centers (30-metal sweep)** | 🔵 DEFERRED R22-P1 Wf-79 (1 wk GPU-0) | 30 metals × 1000 samples per-metal reachability; per-metal reachability ≥ 0.85 on ≥ 25/30 metals (vs current 0/30 at gate 0.95 strict, 1/30 at 0.85).  Blocked on Blackwell sm_120 GPU-0 + tmQM 30-metal subset pre-stage. |
| **P1.4 Trans-effect propagation** | 🔵 DEFERRED R22-P1 Wf-79 (2-3 days CPU) | Pt(II) series: Cl⁻ > NH₃ > H₂O kinetic trans-effect ordering vs Reedijk 2008 experimental literature.  Blocked on lit kinetics data ingestion + per-metal rate-constant regression. |
| **P1.5 Wf-75 close-out** | 🔵 DEFERRED R22-P1 Wf-79 (2 days CPU) | `TODO/completed/105_r22_p1_hierarchical_multimetal.md` (≤ 300 lines) + LAMBDA_ROADMAP §4.2 L13 row patched + INDEX §17.6 cross-ref update.  Today's Wf-78 is the **Wf-78 Phase 1** ship; Wf-79 will author the Wf-75 close-out after P1.3 + P1.4 GPU phases ship. |

### §3.2 Wf-79 R22-P1 launch readiness

This Wf-78 Phase 1 ship is the **launch gate** for Wf-79 R22-P1 GPU phase.  Wf-79 launch requires:

1. ✅ Outer/inner loop shape pinned (today's Wf-78 ship)
2. ✅ Per-metal + per-target breakdown deterministic (today's Wf-78 ship)
3. ✅ Synthetic mock scorer + smoke test PASS (today's Wf-78 ship, 16/16 tests)
4. ⏳ Blackwell sm_120 GPU-0 retrain queue slot (per `molmetal/reports/wf_blackwell_retrain/gpu_env_check.json`; Wf-23 in-flight; Wf-79 launch queue slot TBD)
5. ⏳ tmQM 30-metal subset pre-stage (per R22-P1 P1.3 plan; Wf-79 Phase 1 will stage)
6. ⏳ Reedijk 2008 kinetics data ingestion (per R22-P1 P1.4 plan; Wf-79 Phase 1 will ingest)

**Status**: Wf-78 Phase 1 ship unblocks items 1-3 (CPU scaffold complete); items 4-6 are Wf-79 launch dependencies (GPU + data prep).

### §3.3 Reachability gate NOT MET (honest framing)

Per `party-twenty-spirit §3.5` (0-tolerance honest framing), today's Wf-78 Phase 1 ship does **NOT** lift the reachability gate (0.95).  The structural blockers catalogued in `completed/96_r20_p5_closure_catherine.md §3.2` remain unchanged:

  1. **Valence-BNF rejection** of `[M+]<-X` dative bonds — requires Wf-79 P1.1 dative-bond parser
  2. **Tautomerisation redex exhaustion** (~30% tmQM samples) — requires Wf-79 P1.3 30-metal sweep + Yeo-Johnson power transform per `87j_wf44_root_cause_fixes.md` C5
  3. **Sample sparsity** for 7 metals (Hf=20, Ta=30, La=3, Nb=19, Tc=15, Sc=20, Zr=53) — requires Wf-79 P1.3 multi-metal cluster centers

**Today's MEASURED**: Wf-78 Phase 1 ships the outer/inner loop shape + per-metal + per-target breakdown on synthetic mock.  Reachability lift = 0 (mock scorer does not invoke `closure_test`).

**PROJECTED** (Wf-79 R22-P1 P1.3 GPU): per-metal reachability ≥ 0.85 on ≥ 25/30 metals (realistic); ≥ 0.95 on ≥ 25/30 metals (stretch).

---

## §4 Honest framing

### §4.1 What ships today (on-disk evidence)

| Deliverable | Path | Size | Status |
|---|---|---|:---:|
| Module | `molmetal/molmetal_lam/reward/hierarchical_multimetal.py` | 499 LOC | ✅ SHIP |
| Tests | `molmetal/tests/test_hierarchical_multimetal.py` | 245 LOC, 16 tests | ✅ SHIP (16/16 PASS) |
| CLI smoke | `python -m molmetal.molmetal_lam.reward.hierarchical_multimetal` | runs in 0.0001 s | ✅ SHIP |
| Synthesis (this file) | `molmetal/reports/wf78_l13_scaffold/synthesis.md` | 250+ lines | ✅ SHIP |
| Public API | 8 classes/funcs (`MetalPool` / `TargetSpec` / `HierarchicalCoordConfig` / `HierarchicalCoordResult` / `HierarchicalCoordinator` / `synthetic_mock_score` / `smoke_hierarchical` / `SUPPORTED_METALS`) | — | ✅ SHIP |

### §4.2 What does NOT ship today (honest-negative)

| Item | Status | Reason |
|---|---|---|
| **P1.1 dative-bond parser** | ❌ NOT ship | Deferred to Wf-79 R22-P1 P1.1 (separate module `lam_chem/dative_bond_parser.py` with ≥10 tests) |
| **P1.3 multi-metal cluster centers (30-metal sweep)** | ❌ NOT ship | Requires Blackwell sm_120 GPU-0 retrain; deferred to Wf-79 R22-P1 P1.3 (1 wk GPU-0) |
| **P1.4 trans-effect propagation** | ❌ NOT ship | Requires Reedijk 2008 kinetics data ingestion + per-metal rate regression; deferred to Wf-79 R22-P1 P1.4 (2-3 days CPU) |
| **P1.5 Wf-75 close-out** | ❌ NOT ship | Requires P1.3 + P1.4 GPU phases to ship first; deferred to Wf-79 (2 days CPU) |
| **Reachability gate 0.95 lifted** | ❌ NOT ship | Today's Wf-78 Phase 1 is CPU scaffold only; structural blockers (valence-BNF dative / tautomerisation redex / sample sparsity) unchanged |
| **Paper §3.6 L13 row** | ❌ NOT ship | TODO-84 paper-gate in effect (no paper work without authorization); Wf-79 P1.5 close-out will update INDEX §17.6 cross-ref but not paper |
| **LAMBDA_ROADMAP §4.2 L13 row patch** | ❌ NOT ship | Deferred to Wf-79 P1.5 close-out |

### §4.3 R17 surface unmodified (no-regression guarantee)

| R17/R18 surface | Status |
|---|---|
| `molmetal/molmetal_lam/lam_chem/closure.py` (Wf-36 P1 / Wf-37 P2 ship) | **unmodified** |
| `molmetal/molmetal_lam/lam_chem/lambda_combinators.py` (R17) | **unmodified** |
| `molmetal/molmetal_lam/reward/{hilbert_primal_dual,lambda_astsr,learned_shaping,symbolic_regression}.py` (R17/R18 ship) | **unmodified** |
| `molmetal/molmetal_lam/reward/__init__.py` (empty) | **unmodified** (today's Wf-78 does NOT add `__init__.py` imports; callers import `from molmetal.molmetal_lam.reward.hierarchical_multimetal import ...` directly per Wf-36 P1 / Wf-37 P2 convention) |
| `molmetal/scripts/*` R17/R18 scripts | **unmodified** |
| `molmetal/proof_search.py` (6217 LOC R17) | **unmodified** |
| `molmetal/lambda_astsr.py` (1059 LOC R17) | **unmodified** |
| `molmetal/coupling_adapter.py` (730 LOC R17) | **unmodified** |

**Verdict**: Wf-78 Phase 1 ships **0 modifications** to R17/R18 surface — `hierarchical_multimetal.py` is a **new** module under `molmetal_lam/reward/` that is independently testable and does NOT alter any existing import path.

### §4.4 GPU retrain executed today

**0** — today's Wf-78 Phase 1 is CPU-only initial code.  All 16 tests run on CPU in 0.12 s wall.  The synthetic mock scorer uses SHA-256 hashing (no numpy, no sklearn, no PyTorch, no GPU).  Blackwell sm_120 GPU-0 retrain is deferred to Wf-79 R22-P1 P1.3.

### §4.5 Paper-gate honored

**0 paper files touched** — `paper/main.tex` / `paper/sections/*.tex` / `paper/refs.bib` carry from R18 close-out (132+ pages v1.7).  TODO-84 paper-gate is honored: this Wf-78 Phase 1 ship is **code-only**, no paper work.  Paper §3.6 L13 row will be authored only after (a) TODO-84 user authorization + (b) Wf-79 P1.3 + P1.4 GPU phases ship MEASURED lift.

### §4.6 Final verdict

**Wf-78 Phase 1 ships today**:
- ✅ `molmetal/molmetal_lam/reward/hierarchical_multimetal.py` (499 LOC, 8 public API)
- ✅ `molmetal/tests/test_hierarchical_multimetal.py` (16 tests PASS, 0.12 s CPU wall)
- ✅ `molmetal/reports/wf78_l13_scaffold/synthesis.md` (this file)
- ✅ 0 R17/R18 surface modifications
- ✅ 0 paper work touched
- ✅ 0 GPU retrain executed
- ✅ Wf-79 R22-P1 P1.3 + P1.4 launch gate unblocked (CPU scaffold + smoke + per-metal breakdown deterministic)

**Wf-78 Phase 1 does NOT ship** (honest-negative):
- ❌ P1.1 dative-bond parser (deferred to Wf-79 P1.1)
- ❌ P1.3 multi-metal cluster centers 30-metal sweep (deferred to Wf-79 P1.3 GPU)
- ❌ P1.4 trans-effect propagation (deferred to Wf-79 P1.4 CPU)
- ❌ Reachability gate 0.95 lift (deferred to Wf-79 GPU)
- ❌ Wf-75 close-out (deferred to Wf-79 P1.5 after P1.3 + P1.4)
- ❌ LAMBDA_ROADMAP §4.2 L13 row patch (deferred to Wf-79 P1.5)
- ❌ INDEX §17.6 cross-ref update (deferred to Wf-79 P1.5)

**Date**: 2026-09-26 (Wf-78 Phase 1 — L13 hierarchical multi-metal coord CPU-only initial scaffold ship)

---

## §5 Cross-references

### §5.1 Source-of-truth documents

- `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` §2 (R22-P1 P1.1-P1.5 phases + cross-skill strategy + MEASURED-vs-PROJECTED)
- `TODO/pending/r22_l13_l16_gap_analysis.md` §1 (L13 candidate 188 scan signals)
- `TODO/completed/101_r22_plan_l13_l16.md` (Wf-72 Phase 1+2+3 close-out umbrella)
- `TODO/completed/96_r20_p5_closure_catherine.md` §3.2 (reachability 0.5962 NOT gate 0.95 structural blockers)
- `TODO/completed/95_r20_p4_3way_100pocket.md` (3-way + 100-pocket L10 baseline)
- `TODO/completed/91_wf41_3way_integration.md` (L10 3-way R²=-0.3595 ± 0.2806)
- `TODO/completed/87j_wf44_root_cause_fixes.md` C5 (target distribution extreme TmQM-heavy)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (existing 12-row L1-L12 trajectory; proposed L13 row extension)
- `TODO/INDEX.md` §17.5 (TODO-103 L13 candidate entry)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate CRITICAL — honored)

### §5.2 Today's ship surface (on-disk evidence)

| Deliverable | Path | Size | Status |
|---|---|---|:---:|
| Module | `molmetal/molmetal_lam/reward/hierarchical_multimetal.py` | 499 LOC | ✅ SHIP |
| Tests | `molmetal/tests/test_hierarchical_multimetal.py` | 245 LOC, 16 tests | ✅ SHIP (16/16 PASS) |
| CLI smoke entry | `python -m molmetal.molmetal_lam.reward.hierarchical_multimetal` | runs in 0.0001 s | ✅ SHIP |
| Synthesis (this file) | `molmetal/reports/wf78_l13_scaffold/synthesis.md` | 250+ lines | ✅ SHIP |

### §5.3 Downstream Wf-79 R22-P1 launch

- **Wf-79 R22-P1 P1.1**: ship `molmetal/molmetal_lam/lam_chem/dative_bond_parser.py` + `tests/test_dative_bond_parser.py` (≥10 tests); chemistry-aware `[M+]<-X` SMARTS for Pt/Pd/Au/Ru/Ir
- **Wf-79 R22-P1 P1.2**: replace today's mock scorer with the real `closure_test` from `molmetal/molmetal_lam/lam_chem/closure.py` (R17 ship, Wf-36 P1 verified)
- **Wf-79 R22-P1 P1.3**: Blackwell sm_120 GPU-0 retrain queue slot — 30 metals × 1000 samples per-metal reachability; per-metal reachability ≥ 0.85 on ≥ 25/30 metals (realistic) / ≥ 0.95 on ≥ 25/30 metals (stretch)
- **Wf-79 R22-P1 P1.4**: Reedijk 2008 kinetics data ingestion + Pt(II) trans-effect ordering per-metal rate regression
- **Wf-79 R22-P1 P1.5**: `TODO/completed/105_r22_p1_hierarchical_multimetal.md` (≤ 300 lines) + LAMBDA_ROADMAP §4.2 L13 row patch + INDEX §17.6 cross-ref update

### §5.4 Skill alignment

- **`party-twenty-spirit` §3.5** (0-tolerance honest framing) — this synthesis's §4 honest framing section follows the 0-tolerance discipline; no fabricated MEASURED numbers; reachability gate lift explicitly honest-negative.
- **`mol-metal-lambda-calculus`** (closure theorem / β-NF / KataGo-MCTS / PLL / 21-rule PRIMITIVE_RULES) — today's Wf-78 defers real `closure_test` call to Wf-79 P1.2; P1.1 dative-bond parser to Wf-79 P1.1.
- **`mol-metal-r18-math`** (T1 SE(3) SB / T5 Multi-PH) — referenced for Wf-79 P1.3 multi-metal cluster centers integration.
- **`mol-metal-click-chemistry`** (18 click rules + 84-MLE + 4 NEW SMARTS) — referenced for Wf-79 P1.1 dative-bond parser SMARTS pattern.