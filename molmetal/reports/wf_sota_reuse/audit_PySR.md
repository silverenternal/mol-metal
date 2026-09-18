# Audit: PySR (Cranmer et al. 2023, arXiv:2305.01582)

Path: `/home/hugo/codes/try_triton_on_rocm/molmetal/references/PySR/`
Our hand-rolled code: `molmetal/molmetal_lam/lam_symbolic/symbolic_regression.py`
(1247 LOC) — backend: Python AST + LASSO regression only.

## (1) What we already have hand-rolled that is redundant

- **Symbolic regression engine**: we use a greedy binary-tree grow + a
  tiny library of operators (+, -, *, /, exp, log, sqrt) plus LASSO
  for constant fitting. PySR (`pysr.PySRRegressor`) is a 10+ year mature
  Julia backend with evolutionary search, simulated annealing,
  parsimony, adaptive parsimony, adaptive mutation weights, constant
  optimisation, and many more operators.
- **SymPy export**: we generate LaTeX via SymPy round-trip; PySR exports
  via `sympy2jax`/`sympy2torch` (differentiable!) and `export_latex.py`
  with proper precedence.
- **Hold-out validation**: PySR has it built in (`denoising=True`,
  `loss_function=...`, `batched_evaluation=True`).
- **Hypothesis search across 100+ cells**: PySR's parallel island
  population is purpose-built for this; our 1247 LOC lacks parallelism.

## (2) What is actually different / better in our hand-rolled code

- **Zero external deps**: PySR needs Julia 1.8+ + first-import download
  of ~30 packages (`pip install pysr` triggers Julia depot install).
  Our hand-roll is Python-only.
- **No LLVM / binary JIT**: PySR uses Julia's LLVM-compiled expressions;
  our hand-roll is interpretable Python (useful for debugging inside the
  reward loop).
- **Honest framing**: per WF-Deflex-Symbolic Phase 3 verdict (task #788),
  our symbolic_regression on 147 MEASURED cells is a held-out validation
  exercise, NOT a production reward channel. PySR would be overkill for
  a "nice to have" interpretive figure.
- **Cite-only §5**: paper §5 uses PySR as a citation, not a code import.
- **Memory ceiling**: PySR's 100-population island swarm is 1-2 GB RAM;
  our CPU-only repo is GPU-out + recovery-constrained (memory pressure).

## (3) Concrete 3-line patch plan (file + lines + import)

```
# molmetal/molmetal_lam/lam_symbolic/symbolic_regression.py:1-30 (header)
# OPTION A: keep hand-roll; cite PySR in paper §5.8.
# OPTION B (if a fresh §5 panel needs Julia-quality expressions):
from pysr import PySRRegressor
model = PySRRegressor(niterations=40, binary_operators=["+","-","*","/"],
                       unary_operators=["exp","log","sqrt"], populations=8)
model.fit(X_cell_matrix, y_metric_vec); print(model.equations_)
```
Caveat: PySR's first `import pysr` downloads `SymbolicRegression.jl` —
~1.5 GB and 3-5 min on first run. Acceptable for paper figure
regeneration, not for the reward loop.

## Verdict

**KEEP HAND-ROLLED.** Symbolic regression is a one-shot interpretive
exercise (WF-Deflex-Symbolic task #783); PySR's heavyweight install +
Julia dependency does not earn its keep. Cite Cranmer 2023 in paper §5
as the SOTA symbolic-regression baseline; keep our 1247 LOC as a
lightweight fallback for the held-out validation cell. If a future
§5 panel truly needs Julia-quality expressions, we can PySR behind a
`--symbolic-backend=pysr` flag and lazy-import it.