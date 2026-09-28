# `reward/` — Coupling adapter + ASTSR + learned shaping

The "middle layer" between the **symbolic** λ-calculus (Layer 1-5) and the
**geometric** CFM/EGNN side. Provides: the LambdaASTSR (Deflexpressor-style
AST symbolic regression), the coupling adapter (Lambda × CFM 32→64 bridge),
learned reward shaping (RL-style value-function bootstrap), and BioLM /
PlatinAI oracle proxies.

## Key files

| File | Role |
|------|------|
| `lambda_astsr.py` | LambdaASTSR — AST-based symbolic regression (M8). |
| `coupling_adapter.py` | Lambda × CFM feature bridge 32→64 (M9). |
| `learned_shaping.py` | RL value-function bootstrap for reward shaping. |
| `symbolic_regression.py` | Generic SR wrapper around PySR. |
| `pysr_adapter.py` | PySR backend shim. |
| `biolm_score_adapter.py` | BioLM-based scoring proxy. |
| `platinai_oracle.py` | PlatinAI pIC50 oracle (Pt 4-cell). |
| `lambda_reward_dataset.py` | Dataset of (term, reward) for ASTSR training. |

## Usage

```python
from molmetal_lam.reward.lambda_astsr import LambdaASTSR
astsr = LambdaASTSR(complexity = 10)
astsr.fit(terms, rewards)            # symbolic regression on λ-term features
expr = astsr.best_expression_       # human-readable formula
```

```python
from molmetal_lam.reward.coupling_adapter import CouplingAdapter
adapter = CouplingAdapter(in_dim=32, out_dim=64)
z64 = adapter(z32_lambda_features, z32_cfm_features)
```

## Common pitfalls

* `LambdaASTSR` requires the PySR Julia backend; falls back to sklearn
  `RandomForestRegressor` automatically (slower convergence).
* `CouplingAdapter` is **frozen-feature** only — no gradient through AST.
* `learned_shaping` value targets need **discounted returns**, not raw reward.
* Mechanism ↔ metric cross-refs: `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md`
  for **M8** (LambdaASTSR), **M9** (coupling adapter).
* Tests: `pytest molmetal/tests/test_lambda_astsr.py test_coupling_adapter.py -q`.