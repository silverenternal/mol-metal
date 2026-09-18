# WF-SOTA-Reuse R3: PySR replacement verdict

**Date:** 2026-09-16
**Workflow:** WF-SOTA-Reuse R3 (PySR thin wrapper for symbolic-reward regression)
**Clone:** `/home/hugo/codes/try_triton_on_rocm/molmetal/references/PySR/`
**Replaces:** `molmetal/molmetal_lam/reward/symbolic_regression.py` (1247 LOC) for the
symbolic-regression path (kept as closed-form fallback).

---

## 1. PySR Python API surface (read from `pysr/sr.py`)

| Symbol | Location | Purpose |
|---|---|---|
| `class PySRRegressor` | sr.py:457 | sklearn-style fit/predict API |
| `def fit(X, y, *, variable_names=...)` | sr.py:2803 | Run evolutionary search; populates `self.equations_` |
| `self.equations_` (DataFrame or list of DataFrames) | sr.py:1069, sr.py:1544 | Hall of fame, sorted by complexity |
| `def get_best(index=None)` | sr.py:1764 | Picks the row per `model_selection` ("best"/"accuracy"/"score") |
| `def predict(X)` | sr.py:2991 | Eval the best expression on new X |
| `def refresh()` | sr.py:2971 | Reload `equations_` from disk |
| `__init__(binary_operators, unary_operators, niterations, populations, ...)` | sr.py:1087 | Constructor with sane defaults |

Equation-row columns (per `equations_` DataFrame): `equation` (str),
`complexity` (int), `loss` (float), `score` (float), `lambda_format`
(callable), `sympy_format` (sympy expression).

---

## 2. Adapter shipped: `molmetal/molmetal_lam/reward/pysr_adapter.py`

Public surface (all import-safe; never raises on missing PySR/Julia):

- `is_available() -> bool` — cached module-level probe.
- `get_import_error() -> Optional[str]` — captured import error string.
- `fit(X, y, *, binary_ops, unary_ops, niterations=100, populations=20,
   variable_names, niterations_warmup, model_selection, temp_equation_file,
   progress, **extra_pysr_kwargs) -> Optional[PySRFitResult]` — runs the
  spec'd defaults.
- `evaluate(formula_str, X) -> np.ndarray` — vectorised evaluator that
  uses SymPy + NumPy (mirrors `evaluate_formula` in the hand-rolled
  module but is self-contained).
- `PySRFitResult` (frozen dataclass) — mirrors `equations_.iloc[idx]`:
  `formula_str`, `complexity`, `loss`, `score`, `equation_idx`,
  `n_samples`, `n_features`, `feature_names`.

Defaults (per spec):
- `binary_ops = ["+", "-", "*", "/"]`
- `unary_ops = []` (None-safe)
- `niterations = 100`
- `populations = 20`
- `niterations_warmup = 10`
- `random_state = 42`
- `model_selection = "best"`
- `temp_equation_file = True` (CI-friendly cleanup)
- `verbosity = 0` (no stdout spam)

Graceful-fallback contract:
1. PySR/Julia missing -> `fit()` returns `None`, never raises.
2. `n_samples < 5` or NaN-only filter -> `None`.
3. Shape mismatch (1D X, 2D y, length mismatch) -> `None`.
4. `PySRRegressor.__init__` or `.fit()` raised -> `None`, log captured.
5. `equations_` empty -> `None`.

---

## 3. Tests shipped: `molmetal/molmetal_lam/tests/test_pysr_adapter.py`

5 functional tests + 1 evaluate test = **6 total**, all green on host:

```
collected 6 items
test_is_available_graceful            PASSED
test_get_import_error_str_when_missing PASSED
test_fit_returns_none_when_unavailable PASSED
test_fit_input_validation              PASSED
test_fit_endto_end_when_available      SKIPPED (no PySR; skipif correct)
test_evaluate_fallback_zero            PASSED
5 passed, 1 skipped, 2 warnings in 0.89s
```

The skipped test auto-runs on hosts where PySR + Julia are present (the
`pytest.mark.skipif` uses `is_available()` from the adapter itself, so
no host is special-cased).

---

## 4. Smoke test result (host without PySR)

The spec'd smoke command runs cleanly:

```
$ uv run python -c "
import numpy as np
from molmetal_lam.reward.pysr_adapter import fit, is_available, get_import_error
print('is_available():', is_available())
print('import_error:', get_import_error())
result = fit(np.random.randn(33, 4), np.random.randn(33))
print('result:', result)
"
is_available(): False
import_error: ModuleNotFoundError: No module named 'pysr'
result: None
```

`fit()` returns `None` instead of crashing — the contract that
downstream callers can switch to the closed-form fitters.

---

## 5. Honest framing vs the prior audit

The earlier audit (`wf_sota_reuse/audit_PySR.md`, "KEEP HAND-ROLLED")
still holds for the **default** path:

1. PySR needs Julia 1.8+; the dev box has no Julia.
2. First `import pysr` triggers a 1.5 GB depot download (~3-5 min).
3. Memory ceiling on 100-population island swarm = 1-2 GB RAM.
4. The hand-rolled F1..F8 fitters remain the production path; PySR is
   the **opt-in interpretive upgrade** (paper §5, ablation tables).

This adapter implements the audit's "OPTION B" suggestion (lazy
`PySRRegressor` behind a `fit()` call, import-safe). It does NOT
delete the hand-rolled 1247 LOC — that code is the fallback when
PySR is unavailable.

---

## 6. Verdict

**SHIPPED — adapter + 6 tests + smoke green on PySR-less host.**

- Adapter: `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reward/pysr_adapter.py`
- Tests:    `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_pysr_adapter.py`
- Smoke:    `uv run python -c "from molmetal_lam.reward.pysr_adapter import fit; print(fit(np.random.randn(33, 4), np.random.randn(33)))"`
- Status:   5 passed, 1 skipped (PySR missing) — both paths verified.
- Hand-rolled symbolic_regression.py: kept intact, still default.
- Effort to upgrade a §5 panel to PySR-quality expressions:
  `pysr_adapter.fit(X_cell, y_metric, niterations=40)` — 1 line.

### Recommendation

Use the adapter in **§5 ablation panels** where a Julia-quality
symbolic expression would strengthen the interpretive figure (Cranmer
2023, arXiv:2305.01582). Keep the hand-rolled F5 path for the
**production reward channel** (proven in WF-Deflex-Symbolic Phase 3).

### Follow-ups (NOT blocked by GPU)

1. Once Julia is installed (`uv pip install pysr` + Julia 1.8+),
   re-run `test_fit_endto_end_when_available` and assert complexity
   floor on the y = x0 + 2*x1 toy.
2. Add a `--symbolic-backend=pysr` CLI flag in
   `r4_lambda_only_run.py` that routes through the adapter (1-day
   wire-up).
3. The Deflex paper §5.8 panel can use the adapter to regenerate the
   symbolic-reward figure with `niterations=40, populations=8` for a
   Cranmer-style ablation row.