# `lam_chem/` — Molecular Lambda Calculus atoms-as-combinators

Layer 1-5 of the MLC framework. Provides the *purely functional* core of the
Molecular Lambda Calculus: an untyped λ-calculus AST, β-reduction, capture-
avoiding substitution, and the "atoms-as-combinators" encoding (S, K, I, Pt_II,
…). PySR is wrapped with a graceful scikit-learn fallback so the module is
importable without the Julia backend.

## Key files

| File | Role |
|------|------|
| `ast.py` | `LamVar` / `LamAbs` / `LamApp` dataclasses (frozen, hashable). |
| `reducer.py` | `reduce_step`, `reduce_to_normal_form`, `pll_cost` (Pareto-Lévy-Lévy cost). |
| `well_formedness.py` | `check_closed_term`, `check_beta_normal_form`, `check_arity_conservation`. |
| `lambda_combinators.py` | Atom-as-combinator table (S, K, I + metal arities: Pt_II, Cu_II, …). |
| `pt_click_compat.py` | 84-rule MLE matrix + 18 click + 4 NEW SMARTS (M2/M10/M11). |
| `pt_metal_ligand_exchange.py` | Transmetalation / Cl-OH swap / ligand-shell MMA (M11). |
| `cisplatin_builder.py` | `build_cisplatin()` — canonical reference molecule. |
| `coupling_adapter.py` | Lambda × CFM 32→64 feature bridge (M9). |
| `pysr_wrapper.py` | `HeuristicRegressor` — PySR with sklearn fallback. |
| `pll_signature.py` | PLL (Persistent Local Laplacian) signature (M4). |
| `pharmacophore_filter.py`, `pharmacophore_reward_channel.py` | ADMET type predicates (Layer 7). |
| `wetlab_protocol.py`, `wetlab_reward_channel.py` | Wet-lab feasibility scoring. |
| `tmqm_dataset.py`, `platinai_dataset.py` | Pre-training / fine-tuning datasets. |

## Usage

```python
from molmetal_lam.lam_chem import var, lam, apply, reduce_to_normal_form

T = var("T")                          # LamVar('T')
F = lam("x", var("x"))                # identity combinator I
M = apply(F, T)                       # β-reduction step
reduce_to_normal_form(M, max_steps=10) # → LamVar('T')
```

```python
from molmetal_lam.lam_chem import HeuristicRegressor
h = HeuristicRegressor()                # auto-detects PySR vs sklearn
h.fit([[1.0, 2.0], [2.0, 3.0]], [1.0, 4.0])
h.predict([[1.5, 2.5]])
```

## Common pitfalls

* `LamNode` is **frozen** — mutate via `dataclasses.replace`, not assignment.
* `beta_reduce` is **single-step, leftmost**; use `to_normal_form` for full NF.
* Substitution is capture-avoiding — don't bypass it with string replace.
* Mechanism ↔ metric cross-refs: see `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`
  for **M1** (atoms-as-combinators), **M2/M10/M11** (rules),
  **M4** (PLL), **M9** (coupling).
* Tests: `pytest molmetal/tests/test_lambda_combinators_expansion.py -q`.
