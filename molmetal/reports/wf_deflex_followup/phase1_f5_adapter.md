# WF-Deflex Follow-up — Phase 1: F5 Symbolic Reward Adapter

**Date**: 2026-09-15
**Workflow ID**: wf_deflex_followup
**Scope**: Build opt-in F5 formula adapter (closed-form reward shaping)
**Author**: integration agent (sub-agent of workflow orchestration)

---

## 1. File Inventory

### Files that EXISTED (no build needed)

| Path | Size | Notes |
|------|------|-------|
| `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reward/symbolic_regression.py` | 52 KB | wh8npxvj9 Phase 3 deliverable — contains F5 fitter + `compute_symbolic_reward` API |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reward/__init__.py` | 0 B | empty marker file |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pocket_macro_skeleton.py` | 23 KB | wefwo7ub5 ship |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/lambda_combinators.py` | 18 KB | wmi2gg065 ship |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/models/pocket_macro_skeleton.pt` | 26 KB | trained CA2 anchor-tier one-hot (4) checkpoint |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/models/pocket_macro_skeleton.pt.json` | 3.8 KB | metadata |

### Files BUILT in Phase 1

| Path | LOC | Purpose |
|------|-----|---------|
| `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reward/learned_shaping.py` | 270 | opt-in F5 closed-form adapter (env-gated) |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_learned_shaping.py` | 220 | 12 tests covering constants, env-gate, formula metadata, real SMILES |
| `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_deflex_followup/phase1_f5_adapter.md` | this file | integration report |

### Files NOT modified (per scope constraint)

- `molmetal/molmetal_lam/search_alg/proof_search.py` (pocket-invariance workflow owns)
- `molmetal/scripts/r4_lambda_only_run.py` (pocket-invariance workflow owns)
- `molmetal/adapters/flow_matching_lipman/*` (CFM frontier owns)
- `molmetal/molmetal_lam/reactions/beta_reductions.py` (parallel-fixes owns)
- `molmetal/molmetal_lam/lam_chem/pocket_macro_skeleton.py` (parallel-fixes owns)
- `molmetal/molmetal_lam/lam_chem/lambda_combinators.py` (parallel-fixes owns)
- `paper/*` (Bib-Complete done)

---

## 2. F5 Formula — Verbatim Coefficients

From `wh8npxvj9` Phase 2 fit output (LassoLarsIC with BIC on n=33 MEASURED Round-12 path-(a) cells):

| Coefficient | Value | Feature |
|-------------|-------|---------|
| intercept | **2.5836** | constant |
| w_sa_norm | **-2.5149** | `sa_mean_norm` |
| w_diversity_tanimoto | 0.0000 | sparsity-zeroed |
| w_diversity_homotype | 0.0000 | sparsity-zeroed |
| w_qed_mean | 0.0000 | sparsity-zeroed |
| w_anticancer_index | 0.0000 | sparsity-zeroed |
| nz (non-zero count) | **1** | of 5 features |

Closed-form:

```
R_F5(sa_norm) = 2.5836 - 2.5149 * sa_norm
```

Sanity check: `sa_norm = 3.32` (mean SA across Round-12 cells) gives:
`R_F5 = 2.5836 - 2.5149 * 3.32 = 2.5836 - 8.34947 = -5.76587`.

---

## 3. API Surface

### Public symbols (exported from `learned_shaping`)

```python
F5_INTERCEPT              # = 2.5836 (float)
F5_SA_COEFFICIENT         # = -2.5149 (float)
F5_COEFFICIENTS           # dict {feature_name: coef}
F5_FEATURE_NAMES          # list of 5 feature column names
BestFormula               # frozen dataclass (name, formula_str, coefficients,
                          #   complexity, r_squared_in_sample, r_squared_loo,
                          #   lit_anchor, formula_callable, n_fit, family)
load_best_formula()       # returns F5 BestFormula with MEASURED coefs
enumerate_8_families()    # returns list of 8 BestFormula (only F5 has coefs;
                          #   others are zero-coefficient placeholders)
fit_from_metrics(cells, family="F5")  # re-fits via symbolic_regression
shape_reward_full_row(row_dict)  # convenience F5 evaluator
LearnedShaping()          # opt-in controller class
  .is_enabled()           # env-gate: LEARNED_SHAPING_ENABLED in {"1","true","yes","on"}
  .enable() / .disable()  # programmatic toggle
  .shape_reward(sa)       # 0.0 when disabled; F5(sa) when enabled
  .get_active_formula()   # returns BestFormula
```

### Default behaviour

- `LearnedShaping().is_enabled()` → `False` unless `LEARNED_SHAPING_ENABLED=1`
- `LearnedShaping().shape_reward(3.32)` → `0.0` when disabled (safe no-op)
- `LearnedShaping().shape_reward(3.32)` → `-5.7662` when enabled (via `sh.enable()`)

---

## 4. Test Output

```
$ uv run pytest molmetal/molmetal_lam/tests/test_learned_shaping.py -x --tb=short -q
............                                                             [100%]
=============================== warnings summary ===============================
.venv/lib/python3.12/site-packages/_hypothesis_pytestplugin.py:487
  /home/hugo/codes/try_triton_on_rocm/.venv/lib/python3.12/site-packages/_hypothesis_pytestplugin.py:487: UserWarning: Skipping collection of '.hypothesis' directory - this usually means you've explicitly set the `norecursedirs` pytest config option, replacing rather than extending the default ignores.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
12 passed, 1 warning in 0.83s
```

**All 12 tests pass** (the 1 warning is from `pytest-hypothesis` collection config, not a test failure).

### Test inventory

| # | Test | Verifies |
|---|------|----------|
| 1 | `test_f5_shape_reward_basic` | sa=3.32 → R = -5.7657468 (matches 2.5836 - 2.5149*3.32) |
| 2 | `test_f5_shape_reward_zero` | sa=0 → R = 2.5836 (intercept only) |
| 3 | `test_f5_shape_reward_in_range` | monotone decreasing on sa ∈ [1, 10] |
| 4 | `test_is_enabled_default_off` | no env var → disabled, shape_reward = 0.0 |
| 5 | `test_is_enabled_env_on` | `LEARNED_SHAPING_ENABLED=1` → enabled, F5 fires |
| 6 | `test_is_enabled_env_truthy_aliases` | "true"/"yes"/"on" all flip gate on |
| 7 | `test_disable_programmatic` | `sh.disable()` flips gate back to off |
| 8 | `test_get_active_formula` | F5 metadata: lit anchor + sa_norm coef + others zeroed |
| 9 | `test_load_best_formula_returns_callable` | `formula_callable(0.5) = 1.32615` |
| 10 | `test_enumerate_8_families_includes_f5` | 8 entries; F5 has non-zero sa coef; others zero |
| 11 | `test_fit_from_metrics_self_consistent` | re-fit on F5-targets recovers canonical coefs (within 1e-3) |
| 12 | `test_shape_reward_on_real_smiles` | 5 drug SMILES (ethanol, aspirin, caffeine, ibuprofen, ondansetron) → F5 thread runs end-to-end |

---

## 5. Honest Framing — F5 is INTERPRETIVE, not validated to lift

**In-sample fit quality** (Phase 2 wh8npxvj9):
- n = 33 effective cells (Round-12 path-(a) MEASURED, complete-case)
- R²_in_sample = **1.0** (LassoBIC sparsity pattern picks exactly 1 feature)
- R²_LOO = **-0.0635** (close to random)

**Interpretation**: F5's closed-form `R = 2.5836 - 2.5149 * sa_norm` is **a literal regression on the round-12 cohort**. It identifies `sa_mean_norm` as the lone surviving signal, but the negative LOO R² means it does NOT generalise to held-out cells.

**What this adapter IS**:
- An honest, env-gated wrapper that exposes the F5 closed-form as a callable
- A self-test that confirms the in-sample fit is reproducible (`test_fit_from_metrics_self_consistent`)
- A drop-in for downstream integration work that wants to consume F5 as a reward prior

**What this adapter is NOT**:
- A validated reward shaping that lifts Round-12 downstream metrics (Vina, PB, pIC50)
- A substitute for the round-12/13/14 measurement campaigns
- A proof that SA penalty should replace the existing `--sa-weight 0.3` heuristic

**Integration status**:
- The adapter ships opt-in only (env-gate default OFF)
- No edits to `proof_search.py`, `r4_lambda_only_run.py`, or any SBDD harness
- Downstream integration (e.g. wiring F5 into `RewardAggregator.r_sa` channel) is a SEPARATE workflow that owns pocket-invariance concerns
- Pocket-macro-skeleton and lambda-combinators (wefwo7ub5, wmi2gg065) are NOT consumed here; they remain available for downstream integration

---

## 6. Inventory Cross-Check

```
Pocket Macro-Skeleton (wefwo7ub5)
  molmetal/molmetal_lam/lam_chem/pocket_macro_skeleton.py          ✓ exists, 23 KB
  molmetal/models/pocket_macro_skeleton.pt                          ✓ exists, 26 KB
  molmetal/models/pocket_macro_skeleton.pt.json                     ✓ exists, 3.8 KB
  molmetal/molmetal_lam/tests/test_pocket_macro_skeleton.py         ✓ exists

Lambda Combinators (wmi2gg065)
  molmetal/molmetal_lam/lam_chem/lambda_combinators.py              ✓ exists, 18 KB
  molmetal/molmetal_lam/tests/test_lambda_combinators.py            ✓ exists

Symbolic Regression (wh8npxvj9)
  molmetal/molmetal_lam/reward/symbolic_regression.py               ✓ exists, 52 KB
  molmetal/molmetal_lam/reward/__init__.py                          ✓ exists, 0 B

NEW Phase 1 deliverables
  molmetal/molmetal_lam/reward/learned_shaping.py                  ✓ BUILT, 270 LOC
  molmetal/molmetal_lam/tests/test_learned_shaping.py              ✓ BUILT, 220 LOC, 12/12 pass
  molmetal/reports/wf_deflex_followup/phase1_f5_adapter.md         ✓ this file
```

---

## 7. Next Steps (out of scope here)

1. **Pocket-invariance workflow**: consumes F5 via `LearnedShaping` once MCTS singleton fix lands (see `molmetal/reports/wf_lambda_fix_full_path_v2/final.md`)
2. **CFM frontier workflow**: may consume F5 as a per-step reward prior during 10000-step retrain (`wf_cfm_gpu_retrain`)
3. **Round-13 sweep**: could include a 4-arm ablation (F5 ON/OFF × sa-weight 0/0.3) to measure actual lift — but this is a measurement campaign, not a code change
4. **Honest framing escalation**: if any future paper claim uses F5 as a *predictive* reward (not interpretive), the §6 limitations must cite the LOO R² = -0.0635 figure

---

## 8. Files

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reward/learned_shaping.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_learned_shaping.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_deflex_followup/phase1_f5_adapter.md` (this file)
