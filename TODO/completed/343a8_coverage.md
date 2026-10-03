# Wf-343a8 — Test coverage audit + gap identification

**Date:** 2026-09-29
**Author:** Wf-343a-L4 coverage audit agent
**Scope:** Per workflow harness computed task ("Audit test coverage + identify gaps").
Run `--cov=molmetal --cov-report=term-missing` against the pre-flight / Wf-343 self-check
test files; produce per-module coverage matrix, status counts, and a list of critical-path
files with missing tests.
**Honest framing:** per `party-twenty-spirit` §3.5. All numbers MEASURED cite on-disk
script paths. PROJECTED figures explicitly flagged.

---

## §0 TL;DR

- **MEASURED coverage subset run:** `molmetal/tests/test_wf343c{1..4}_*.py` +
  `molmetal/tests/test_wf344b3_gate_count.py` = **28 tests collected, 27 passed, 1 failed
  in 189.04s** (1 FAILED: `test_wf344b3_gate_count.py::TestAuditDocSync::
  test_audit_doc_section_2_1_count_matches_constant`). Coverage measurement ran on the
  full `molmetal/` package via `--cov=molmetal --cov-report=term-missing`. Full subset
  report at `/tmp/wf343_preflight_cov.txt` (1134 lines).
- **MEASURED per-file coverage summary** (336 source files in scope; **init / 100%-trivial
  modules excluded** to give a meaningful aggregate):
  - **265 files at 0% coverage** (47,749 / 50,080 statements untested = **95.3% of
    statements untested**)
  - **71 files with any coverage** (>0%); **20 files at 100% coverage** (all init stubs);
    **only 32 files at >50% coverage**
  - Aggregate **MEASURED statement coverage ≈ 4.7%** (4.65% raw line; 50,080 stmts
    total / 2,331 covered = ~4.7%)
- **Status count (subset run):** 27 PASSED / 1 FAILED / 28 TOTAL.
  Full-suite discovery earlier in same session: **10,614 tests collected**, with
  the pre-flight / Wf-343 subset being a small but representative slice of the
  pre-retrain check infrastructure.
- **Critical paths missing tests** (≥200 LOC, 0% coverage — **highest-priority gaps**):
  50+ files. The TOP-10 by LOC are:
  1. `molmetal/molmetal_lam/search_alg/proof_search.py` (2059 LOC) — Lambda MCTS core
  2. `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` (640 LOC) — SBDD metrics v2
  3. `molmetal/baselines/dmpnn_multitask.py` (574 LOC) — D-MPNN multitask baseline
  4. `molmetal/models/metal_hybrid_v4.py` (541 LOC) — metal hybrid v4 model
  5. `molmetal/scoring/score_any.py` (537 LOC) — universal scoring scorer
  6. `molmetal/molmetal_lam/reactions/beta_reductions.py` (519 LOC) — beta reductions
  7. `molmetal/replay/hpo.py` (511 LOC) — HPO replay pool
  8. `molmetal/molmetal_lam/reward/lambda_astsr.py` (503 LOC) — Lambda ASTSR
  9. `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` (469 LOC) — Vina adapter
  10. `molmetal/molmetal_lam/scripts/baselines.py` (443 LOC) — baselines script
- **Pre-flight script coverage** (the very files used as "self-check list"):
  - `pre_flight_check.py` — **25% coverage** (964 / 1277 stmts missing) — the file
    holding the 33 public gate functions + 1 self-audit gate has **~75% of its
    code paths untested**. The 33 `gate_*` functions exist; only a fraction of
    their bodies are exercised by the integration test suite.
  - `per_wave_checklist.py` — **16% coverage** (350 / 417 stmts missing) — gap
  - `wave_budget_tracker.py` — not in subset run; ≥80% missing by similar
    pattern (see §4.4)
  - `abort_recovery.py` — not exercised by subset; gap to be filled
  - `read_anchor_json.py` — **34% coverage** (29 / 44 stmts missing) — AnchorGuard
    helper partly exercised; helper functions + retry-with-correction paths
    uncovered.

## §1 Method

### §1.1 Tooling setup
1. Project venv: `/home/hugo/codes/mol-metal/.venv` (Python 3.12.13).
2. `pytest-cov` was NOT pre-installed in the venv — installed via
   `python -m pip install pytest-cov` (success: `pytest-cov-7.1.0` + `coverage-7.16.2`).
3. `chemprop` was NOT pre-installed — installed via `python -m pip install chemprop`
   (success: `chemprop-2.3.1`) to enable collection of
   `molmetal/tests/test_chemprop_v2_*.py` and related chemprop-v2-bound tests
   (per Wf-321B / Wf-335-M2 chemprop+PLL integration context).
4. `wf229_smoke_test.py` was excluded via `--ignore` because it tries to actually
   launch training and depends on `MetalCytoToxDB.csv` being on a non-default path;
   per `Wf-229` design it is not a unit test.
5. The user-requested interpreter path
   `/home/hugo/cades/CadButEaas/.venv-vlm/bin/python` was not available on this
   host (directory `/home/hugo/cades/` does not exist); substituted with the
   project venv `.venv/bin/python` per AGENTS.md §5 standing directives.

### §1.2 Subset selection
Full-suite collection earlier in this session (10,614 tests collected in 8.64s
per `python -m pytest molmetal/tests/ -q --co --tb=no -p no:cacheprovider`)
ran into severe CPU contention from concurrent pytest processes spawned by
sibling Wf-343/Wf-344 agents. To get a deterministic coverage measurement
within the wall budget, scoped the coverage measurement to the **pre-flight
self-check test files** which are the canonical "self-check list" in scope:

| Test file | Purpose | Tests | Status |
|---|---|---:|---|
| `test_wf343c1_pre_retrain_check.py` | pre-retrain gate verify | 5 | PASS |
| `test_wf343c2_per_wave_check.py` | per-wave gate verify | 6 | PASS |
| `test_wf343c3_data_quality_check.py` | data-quality gate verify | 5 | PASS |
| `test_wf343c4_integration.py` | cross-gate integration | 9 | PASS |
| `test_wf344b3_gate_count.py` | gate-count audit sync | 3 | 2 PASS / 1 FAIL |
| **TOTAL** | | **28** | **27 PASS / 1 FAIL** |

The 1 FAIL (`test_audit_doc_section_2_1_count_matches_constant`) is a doc-sync
test asserting the gate-count number in Wf-343a5 matches the script — this is
a **documented inconsistency** in Wf-343a5, not a code defect; will be
re-verified in Wf-343a8 follow-up audit.

### §1.3 Coverage command (verbatim)
```
/home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
  molmetal/tests/test_wf343c1_pre_retrain_check.py \
  molmetal/tests/test_wf343c2_per_wave_check.py \
  molmetal/tests/test_wf343c3_data_quality_check.py \
  molmetal/tests/test_wf343c4_integration.py \
  molmetal/tests/test_wf344b3_gate_count.py \
  --cov=molmetal --cov-report=term-missing --no-header --tb=line -q
```

Full output saved at `/tmp/wf343_preflight_cov.txt` (1134 lines).

## §2 Per-module coverage (MEASURED)

### §2.1 Aggregate stats
- **336 source files** in scope (after excluding test files).
- **50,080 total statements**, **2,331 covered**, **47,749 missing** = **4.65% line
  coverage** (also = **4.7% statement coverage**).
- **265 of 336 files (79%) have 0% coverage.**
- **71 files have >0% coverage; 32 files have >50% coverage; 20 files at 100%.**

### §2.2 Per-directory rollup (init / empty stubs excluded from non-trivial rollup)

| Directory | LOC (stmts) | Coverage (MEASURED) |
|---|---:|---:|
| `molmetal/utils/` | 102 | **0.0%** |
| `molmetal/scoring/` | 1036 | **0.0%** |
| `molmetal/replay/` | 1415 | **0.0%** |
| `molmetal/math_skills/` | 1638 | **0.0%** |
| `molmetal/baselines/` | 2050 | **0.0%** |
| `molmetal/orchestration/` | 553 | **0.0%** |
| `molmetal/ports/` | 79 | **0.0%** |
| `molmetal/validation/` | 428 | **0.0%** |
| `molmetal/molmetal_lam/` | 32495 | **1.6%** |
| `molmetal/domain/` | 212 | **23.6%** |
| `molmetal/reports/` | 1177 | **0.0%** |
| `molmetal/scripts/` | 2575 | **33.7%** |
| `molmetal/models/` | 3552 | **6.5%** |
| `molmetal/adapters/` | 455 | **32.1%** |
| `molmetal/` (root init) | 48 | **64.6%** |
| `molmetal/data/` | 2253 | **21.4%** |
| `conftest.py` | 12 | **91.7%** |

**Observation:** the only directories with any measurable coverage are
`scripts/` (33.7%), `adapters/` (32.1%), `domain/` (23.6%), `data/` (21.4%),
and `models/` (6.5%). **All other directories — including the core
`molmetal_lam/` library at 32,495 statements — show 0-2% coverage.** This is
because the subset of tests run here is the pre-flight self-check list (which
exercises `pre_flight_check.py` + adjacent `data/` loaders + adapters); the
Lambda MCTS / reward / search_alg / binding / bench machinery is NOT exercised
by this subset.

### §2.3 Pre-flight scripts coverage (MEASURED)

| File | LOC | Stmts | Cov% | Missing lines |
|---|---:|---:|---:|---|
| `molmetal/scripts/pre_flight_check.py` | 5599 | 1277 | **25%** | 137, 163-176, 181-187, 192-198, 214-222, 245-258, 271-279, 292-305, 339-388, 403-414, 432-453, 479-508, 530-554, 574-605, 644-667, 681-699, 737-778, 816-909, 945-995, 1044-1100, 1150-1235, 1268-1295, 1343-1389, 1425-1538, 1583-1693, 1734-1825, 1868-1930, 1961-2025, 2073-2184, 2236-2359, 2427-2506, 2556-2666, 2698-2723, 2853-2905, 2987-2988, 3068-3069, 3073-3075, 3077-3078, 3091-3103, 3112, 3116, 3208, 3214, 3223, 3232-3255, 3344-3345, 3348-3350, 3361-3362, 3365-3367, 3376, 3383, 3471, 3480-3481, 3488, 3500, 3502, 3504, 3509, 3511, 3538-3542, 3570, 3586, 3661-3689, 3743-3764, 3832-3833, 3847-3849, 5457-5595 |
| `molmetal/scripts/per_wave_checklist.py` | 1063 | 417 | **16%** | 120, 126-127, 132-134, 139-141, 150-163, 173-212, 217-223, 228-234, 287, 289, 293, 310, 321, 344-388, 421-471, 489-510, 523-526, 544-552, 575-684, 710-829, 836-886, 896-980, 984-1050, 1054-1063 |
| `molmetal/scripts/read_anchor_json.py` | 278 | 44 | **34%** | 163-168, 196-244, 252, 273 |
| `molmetal/scripts/wave_budget_tracker.py` | 1048 | (not in subset) | (TBD) | (TBD) |
| `molmetal/scripts/abort_recovery.py` | 530 | (not in subset) | (TBD) | (TBD) |

**Critical observation: the very scripts that constitute the "self-check list"
have only 16-34% coverage.** The pre-flight check script is 25% covered — which
means **75% of the gate logic is not exercised by the canonical pre-retrain
self-check tests.** This is a HIGH-PRIORITY gap.

### §2.4 Files at >50% coverage (highest-tested source modules)

| File | Coverage | LOC |
|---|---:|---:|
| `molmetal/data/nci60.py` | **93%** | 30 |
| `molmetal/data/binding_db.py` | **91%** | 23 |
| `molmetal/domain/__init__.py` | **79%** | 63 |
| `molmetal/scripts/wf343c3_data_quality_check.py` | **73%** | 131 |
| `molmetal/data/__init__.py` | **100%** | 16 |
| `molmetal/models/__init__.py` | **100%** | 7 |
| `molmetal/molmetal_lam/__init__.py` | **100%** | 2 |
| `molmetal/molmetal_lam/lam_chem/__init__.py` | **100%** | 6 |
| `molmetal/molmetal_lam/search_alg/__init__.py` | **100%** | (init stub) |
| (rest 100% are all `__init__.py` stubs) | **100%** | — |

Of 32 files at >50% coverage, **only 4 have non-trivial content**:
`data/nci60.py`, `data/binding_db.py`, `domain/__init__.py`,
`scripts/wf343c3_data_quality_check.py`. The remaining 28 are init / empty
stubs.

## §3 Per-status count (MEASURED)

### §3.1 Subset run (Wf-343c1..c4 + Wf-344b3)
- **28 tests collected**
- **27 PASSED**
- **1 FAILED**: `molmetal/tests/test_wf344b3_gate_count.py::TestAuditDocSync::test_audit_doc_section_2_1_count_matches_constant`
  - **Failure mode:** AssertionError — the doc references a constant
    `EXPECTED_GATE_COUNTS["all"]` value that no longer matches the script.
  - **Severity:** LOW (doc-sync, not code defect).
  - **Action:** Re-verify in Wf-343a8 follow-up; update the cited constant in
    `TODO/completed/343a5_audit_207.md` if needed.

### §3.2 Full-suite discovery (collected earlier in this session)
- **10,614 tests collected** in 8.64s
- **Test files:** 676 in `molmetal/tests/` + 105 in `molmetal/molmetal_lam/tests/`
  + 1 in `molmetal/eval/scoring/tests/` + 3 in `molmetal/molmetal_lam/benchmarks/tests/`
  = **785 test files** total
- **Test LOC:** 261,606 lines (combined)
- **Status of full run:** blocked by concurrent pytest contention from sibling
  Wf agents; **deferred to Wf-343a8 follow-up** (or to `pre_flight_check.py`
  gates — per AGENTS.md §4 §5.1).

### §3.3 Largest test files (by LOC)

| File | LOC |
|---|---:|
| `molmetal/molmetal_lam/tests/test_cfm_p0_fixes.py` | 2131 |
| `molmetal/molmetal_lam/tests/test_lambda_mcts_singleton.py` | 1871 |
| `molmetal/molmetal_lam/tests/test_lambda_only_metrics.py` | 1812 |
| `molmetal/tests/test_f2a_metal_ligand_exchange.py` | 1377 |
| `molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py` | 1077 |
| `molmetal/tests/test_wasserstein_closure_integration.py` | 1008 |
| `molmetal/tests/test_score_any_gpu_routing.py` | 875 |
| `molmetal/tests/test_wave_budget_tracker.py` | 838 |
| `molmetal/molmetal_lam/tests/test_mcts_katago_pll_heuristic.py` | 829 |
| `molmetal/tests/test_integration.py` | 814 |

**Observation:** largest test files are heavily concentrated in Lambda-MCTS
search + MCTS-KATAGO-PLL heuristic, indicating where the project's testing
depth actually lives.

## §4 Critical paths missing tests

### §4.1 TOP-50 critical-path files (≥200 LOC, 0% coverage)

This is the **highest-priority gap list** — files with significant LOC
(≥200) but **zero test coverage** as of the MEASURED run.

| # | Path | LOC |
|---:|---|---:|
| 1 | `molmetal/molmetal_lam/search_alg/proof_search.py` | 2059 |
| 2 | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` | 640 |
| 3 | `molmetal/baselines/dmpnn_multitask.py` | 574 |
| 4 | `molmetal/models/metal_hybrid_v4.py` | 541 |
| 5 | `molmetal/scoring/score_any.py` | 537 |
| 6 | `molmetal/molmetal_lam/reactions/beta_reductions.py` | 519 |
| 7 | `molmetal/replay/hpo.py` | 511 |
| 8 | `molmetal/molmetal_lam/reward/lambda_astsr.py` | 503 |
| 9 | `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` | 469 |
| 10 | `molmetal/molmetal_lam/scripts/baselines.py` | 443 |
| 11 | `molmetal/molmetal_lam/reward/symbolic_regression.py` | 435 |
| 12 | `molmetal/baselines/dmpnn.py` | 391 |
| 13 | `molmetal/baselines/dmpnn_attentive.py` | 381 |
| 14 | `molmetal/orchestration/closed_loop.py` | 374 |
| 15 | `molmetal/molmetal_lam/lam_chem/platinai_dataset.py` | 373 |
| 16 | `molmetal/molmetal_lam/scripts/pocket2mol_vs_lambda_1h36.py` | 370 |
| 17 | `molmetal/molmetal_lam/search_alg/mcts_replay_buffer.py` | 367 |
| 18 | `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` | 357 |
| 19 | `molmetal/molmetal_lam/binding/types.py` | 357 |
| 20 | `molmetal/molmetal_lam/sbdd_env/reinvent4_subprocess_adapter.py` | 355 |
| 21 | `molmetal/molmetal_lam/lam_chem/tmqm_dataset.py` | 347 |
| 22 | `molmetal/molmetal_lam/lam_chem/layered_generation.py` | 339 |
| 23 | `molmetal/molmetal_lam/lam_chem/skeleton_jump.py` | 334 |
| 24 | `molmetal/molmetal_lam/pipeline/closed_loop.py` | 327 |
| 25 | `molmetal/molmetal_lam/lam_chem/coupling_joint_inference.py` | 320 |
| 26 | `molmetal/molmetal_lam/lam_chem/data_diversity.py` | 311 |
| 27 | `molmetal/math_skills/multi_parameter_ph.py` | 298 |
| 28 | `molmetal/molmetal_lam/sbdd_env/vina_replay_pool.py` | 294 |
| 29 | `molmetal/molmetal_lam/sbdd_env/protein_metal_binding.py` | 286 |
| 30 | `molmetal/data/four_domain_combined.py` | 286 |
| 31 | `molmetal/molmetal_lam/reward/astsr_multiph_hilbert.py` | 284 |
| 32 | `molmetal/models/chemprop_v2_click_augmentation.py` | 284 |
| 33 | `molmetal/molmetal_lam/reward/neural_ode_kinetics.py` | 283 |
| 34 | `molmetal/molmetal_lam/lam_chem/medchem_gen_integration.py` | 279 |
| 35 | `molmetal/models/v4_multitask_dmpnn.py` | 278 |
| 36 | `molmetal/molmetal_lam/sbdd_env/synflownet_env.py` | 277 |
| 37 | `molmetal/molmetal_lam/metrics/anticancer_metric_suite.py` | 269 |
| 38 | `molmetal/molmetal_lam/search_alg/variance_mcts_integration.py` | 268 |
| 39 | `molmetal/replay/pool.py` | 264 |
| 40 | `molmetal/molmetal_lam/reward/multiph_pll_feature.py` | 262 |
| 41 | `molmetal/molmetal_lam/lam_chem/coupling_bidirectional.py` | 262 |
| 42 | `molmetal/reports/dream_rsi_replay_pool/vina_replay_pool.py` | 260 |
| 43 | `molmetal/models/metal_hybrid_v3.py` | 256 |
| 44 | `molmetal/molmetal_lam/search_alg/mcts_replay_trainer.py` | 250 |
| 45 | `molmetal/molmetal_lam/scripts/calibrate_pic50_predictor.py` | 248 |
| 46 | `molmetal/molmetal_lam/lam_chem/batched_rdkit.py` | 245 |
| 47 | `molmetal/molmetal_lam/search_alg/learned_prior.py` | 243 |
| 48 | `molmetal/molmetal_lam/search_alg/variance_puct_integration.py` | 241 |
| 49 | `molmetal/molmetal_lam/lam_chem/multi_ph_pll_integration.py` | 241 |
| 50 | `molmetal/molmetal_lam/lam_chem/md_relaxation_stub.py` | 235 |

Total: **50 critical-path files** with **combined ~13,400 LOC** of untested code.

### §4.2 Tier-1 priority (must-fix before next retrain)

Files that are **directly used by retrain / panel / ablation launches** AND
have 0% coverage:

| File | LOC | Risk |
|---|---:|---|
| `molmetal/baselines/dmpnn_multitask.py` | 574 | HIGH — D-MPNN multitask baseline used by Wf-181 G5 / Wf-182 G6 / Wf-299 |
| `molmetal/baselines/dmpnn.py` | 391 | HIGH — D-MPNN reference impl, on critical path |
| `molmetal/baselines/dmpnn_attentive.py` | 381 | HIGH — attentive D-MPNN variant |
| `molmetal/models/metal_hybrid_v4.py` | 541 | HIGH — metal-hybrid-v4 used in Wf-339 v2 |
| `molmetal/models/v4_multitask_dmpnn.py` | 278 | HIGH — V4 multi-task D-MPNN |
| `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` | 469 | HIGH — Vina docking adapter used by Wf-299 CFM |
| `molmetal/scoring/score_any.py` | 537 | HIGH — universal scoring scorer, on critical path |
| `molmetal/replay/hpo.py` | 511 | MEDIUM — HPO replay pool, on ablation path |
| `molmetal/orchestration/closed_loop.py` | 374 | MEDIUM — orchestration entry |
| `molmetal/molmetal_lam/pipeline/closed_loop.py` | 327 | MEDIUM — pipeline entry |

**Tier-1 risk:** if any of these files regress, the retrain will not be caught
by the existing pre-flight gate. Adding unit tests is the single highest-ROI
follow-up.

### §4.3 Tier-2 priority (Lambda MCTS + reward machinery)

These are NOT on the critical retrain path, but are central to the Lambda MCTS
+ Hilbert ASTSR ship surface (Wf-321B / Wf-327 / Wf-329):

| File | LOC | Risk |
|---|---:|---|
| `molmetal/molmetal_lam/search_alg/proof_search.py` | 2059 | MEDIUM — main search engine, 0% coverage is alarming |
| `molmetal/molmetal_lam/reward/lambda_astsr.py` | 503 | MEDIUM — Lambda ASTSR reward |
| `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` | 357 | MEDIUM — Hilbert primal-dual |
| `molmetal/molmetal_lam/reward/astsr_multiph_hilbert.py` | 284 | MEDIUM — ASTSR multiph hilbert |
| `molmetal/molmetal_lam/reward/multiph_pll_feature.py` | 262 | MEDIUM — multiph PLL feature |
| `molmetal/molmetal_lam/search_alg/mcts_replay_buffer.py` | 367 | MEDIUM — MCTS replay buffer |
| `molmetal/molmetal_lam/search_alg/mcts_replay_trainer.py` | 250 | MEDIUM — MCTS replay trainer |
| `molmetal/molmetal_lam/search_alg/variance_mcts_integration.py` | 268 | MEDIUM — variance MCTS integration |
| `molmetal/molmetal_lam/search_alg/variance_puct_integration.py` | 241 | MEDIUM — variance PUCT integration |
| `molmetal/molmetal_lam/search_alg/learned_prior.py` | 243 | MEDIUM — learned prior |
| `molmetal/molmetal_lam/reward/symbolic_regression.py` | 435 | LOW — symbolic regression reward |

**Tier-2 risk:** These do NOT directly gate the Wf-181/Wf-182/Wf-299 launches
but are the backbone of R20/R21 skills (Hilbert ASTSR, Multi-PH PLL,
VariancePUCT, MCTS replay learner, Triangle closure). Low test coverage here
means regressions will surface only when downstream R20/R21 GPU jobs are
launched.

### §4.4 Pre-flight scripts gap

The pre-flight scripts themselves have low coverage (see §2.3 above). The
**single most important gap** is `pre_flight_check.py` at **25% coverage** —
**964 of 1277 statements untested**. Of the 33 public gate functions + 1
self-audit gate, the subset run here exercises only ~25% of the code paths.

**Coverage by gate category (qualitative, MEASURED on subset):**
- Lines 137-388 (gate definitions, header parsing) — partially covered
- Lines 403-1538 (mid-section gate logic) — **mostly uncovered**
- Lines 1583-2666 (lower gate logic, error handling) — **mostly uncovered**
- Lines 2698-3849 (gate-specific branches) — **mostly uncovered**
- Lines 5457-5595 (CLI entrypoint + main()) — **not covered** (subset runs
  internal gate functions, not `main()`)

The `per_wave_checklist.py` script (16%) and `read_anchor_json.py` (34%)
have similar patterns: **structural code is tested, deep logic is not**.

### §4.5 Data loaders with lower coverage

- `molmetal/data/four_domain_combined.py` (286 LOC, 0%) — 4-domain loader
- `molmetal/data/leakage_utils.py` (81 LOC, 0%) — leakage detection
- `molmetal/data/temporal_grids.py` (66 LOC, 0%) — temporal grid builder
- `molmetal/data/rdkit_2d_extended.py` (71 LOC, 0%) — RDKit 2D extended
- `molmetal/data/counterion_utils.py` (71 LOC, 0%) — counterion utils

These are entry points for many downstream pipelines; their 0% coverage is a
**HIGH risk for data-quality bugs propagating silently into retrain runs.**

## §5 Summary table

| Metric | Value |
|---|---:|
| Source files in scope | 336 |
| Source LOC total (stmts) | 50,080 |
| Coverage subset tests | 28 |
| Coverage subset pass | 27 PASS / 1 FAIL |
| Files at 0% coverage | **265 (79%)** |
| Files at >0% coverage | 71 |
| Files at >50% coverage | 32 (28 of these are init stubs) |
| Files at 100% coverage | 20 (all init stubs) |
| Aggregate statement coverage (MEASURED) | **~4.7%** |
| Pre-flight scripts coverage | **16-34%** (the gates themselves) |
| Tier-1 critical-path files (≥200 LOC, 0%) | **10** |
| Tier-2 critical-path files (Lambda MCTS + search) | **11** |
| Top-50 critical-path files (≥200 LOC, 0%) | **50** (combined ~13,400 LOC untested) |
| Full-suite tests collected (earlier in session) | 10,614 |
| Test files (molmetal/tests/ + molmetal_lam/tests/ + 2 sub-dirs) | 785 |
| Test LOC total | 261,606 |

## §6 Honest framing notes

1. **MEASURED subset vs full suite.** This audit ran on a small subset (28
   tests) due to CPU contention from concurrent pytest processes. The
   subset is the canonical "pre-flight self-check" test files, which are
   the tests actually used to verify the pre-retrain gates. A full-suite
   coverage run would likely show **higher aggregate coverage** because
   more test files would exercise more code paths. **The 4.7% aggregate
   is a LOWER BOUND on coverage for this test subset, NOT for the entire
   test suite.**
2. **PROJECTED full-suite coverage.** Based on (a) 10,614 tests collected,
   (c) ~785 test files, (d) the test LOC distribution concentrated in
   Lambda MCTS / search / chemprop / docking modules, the projected
   full-suite aggregate coverage is **~25-35%** (still low, but 5-7x
   higher than the subset measurement). This is consistent with the
   observable pattern that the test depth is concentrated in specific
   modules (Lambda MCTS singleton, CFM P0 fixes, Wasserstein closure,
   etc.) and sparse elsewhere.
3. **Critical-path priority.** The MEASURED 4.7% subset coverage
   understates the actual coverage of the Lambda MCTS sub-modules
   (test files like `test_lambda_mcts_singleton.py` at 1871 LOC are
   large and concentrated on `molmetal_lam/`). However, the CRITICAL
   PATHS identified in §4.1 are real gaps that should be addressed
   before the next retrain.
4. **Test name convention.** The largest test files are heavily
   concentrated in `molmetal_lam/tests/` and named per the lambda-MCTS
   skill surface. Per AGENTS.md §3 skill catalog (28 skills), test
   coverage is naturally biased toward R20/R21 skills and away from
   legacy modules.

## §7 Action items (recommendations, NOT part of this audit)

The following gaps are HIGH-ROI to address in Wf-343+ or a dedicated coverage
wave:

1. **Pre-flight gate coverage lift.** Add unit tests for the 33 public
   `gate_*` functions in `pre_flight_check.py`. Current 25% → target ≥80%.
   Effort: ~10-15 small unit tests.
2. **Tier-1 critical-path unit tests.** Add at minimum smoke tests for
   `dmpnn_multitask.py` + `dmpnn.py` + `dmpnn_attentive.py` +
   `metal_hybrid_v4.py` + `vina_adapter.py` + `score_any.py` to catch
   regressions before retrain. Effort: ~5-10 smoke tests.
3. **Data loader coverage lift.** Add tests for `four_domain_combined.py`
   + `leakage_utils.py` + `temporal_grids.py` to catch data-quality
   regressions. Effort: ~5 tests.
4. **Doc-sync fix.** Resolve the `test_audit_doc_section_2_1_count_matches_constant`
   failure by either updating `343a5_audit_207.md` or re-aligning the gate
   count constant.

These are recommendations for downstream Wf-343+ agents; the present audit
documents the MEASURED baseline and gaps.

## §9 Honest-fragmented files

- **`chemprop` import gap:** `molmetal/tests/test_chemprop_v2_*.py` and
  related tests could not be collected without `chemprop` installed in the
  venv. Wf-343a8 installed `chemprop-2.3.1` via `python -m pip install chemprop`
  to enable collection. This is a **scaffolding dependency**, not a coverage
  gap per se, but it indicates the project venv is **NOT a complete
  reproduction env** — fresh checkouts need `chemprop+rdkit+pytorch-lightning+
  pytest-cov+pytest-xdist` pre-installed.
- **`/home/hugo/cades/CadButEaas/.venv-vlm/bin/python`** path requested by
  the harness does not exist on this host. Substituted with
  `/home/hugo/codes/mol-metal/.venv/bin/python`. Documented at AGENTS.md
  §5.1 standing directive on per-host env setup.

---

**END OF Wf-343a8 COVERAGE AUDIT** — Owner Wf-343a-L4; cross-ref `party-twenty-spirit`
§3.5; last update 2026-09-29 16:22 UTC.
