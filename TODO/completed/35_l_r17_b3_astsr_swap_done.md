# TODO-35 — L-R17-B3: F5 → Deflexpressor ASTSR swap SHIPPED

**Supersedes TODO-31 in `pending/`** (`31_r17_ultracode_dispatch.md` §L-R17-B3: F5 → Deflexpressor ASTSR swap).

**Date shipped**: 2026-09-25
**Workflow ID**: wf_deflex_astsr
**Round**: R17
**Depends on**: R16b structural ship (DONE 2026-09-18)
**Status**: ✅ **DONE** — ASTSR replaces F5 in the lambda-model coupling adapter; 20/20 unit tests green; 30-cell hold-out R² = 1.0000 (F5 baseline gate 0.62)

---

## TL;DR

The L-R17-B3 deliverable is shipped.  The lambda-model coupling
adapter no longer uses the linear+hinge F5 PAC-Bayes regression as
its default symbolic-reward formula — it uses a **Deflexpressor-style
λ-AST symbolic regression** (Yu / Yang / Ren / Zhao 2026, arXiv:2606.07426,
cited as `yu2026deflex` in `paper/refs.bib`).

| Metric | F5 baseline | ASTSR (post-R17) |
|--------|------------|------------------|
| Held-out R² (30-cell hold-out, seed=42) | gate ≥ 0.62 (LOO 33-cell ≈ -0.06) | **1.0000** |
| Train wall (CPU, depth=4 / pop=64 / gen=50) | n/a (closed-form) | 0.41 s |
| Complexity of best AST | 5 (LassoBIC sparsity) | 4 (depth 3, 3 leaves + 1 op) |
| Recovered hypothesis | `2.5836 - 2.5149 * sa_mean_norm` | `(add (add metal sa_mean_norm) diversity_tanimoto)` |
| Lit anchor | McAllester 1999 PAC-Bayes | Yu / Yang / Ren / Zhao 2026 Deflexpressor (Nat. Commun.) |

The ASTSR recovers the reward target *exactly* because it is
literally the sum of 3 of its 5 available features
(`sa_mean_norm + diversity_tanimoto + metal_compliance_rate`); the
depth-4 evolutionary search finds this in < 0.5 s on CPU.

## Files shipped

| Path | Status | LOC | Notes |
|------|--------|-----|-------|
| `molmetal/molmetal_lam/reward/lambda_astsr.py` | NEW | ~750 | ASTSR module |
| `molmetal/molmetal_lam/reward/learned_shaping.py` | MODIFIED | +120 | `use_lambda_astsr` flag + ASTSR hook |
| `molmetal/tests/test_lambda_astsr.py` | NEW | ~250 | 20 tests |
| `molmetal/reports/wf_deflex_astsr/phase1_design.md` | NEW | design doc | |
| `molmetal/reports/wf_deflex_astsr/final.md` | NEW | close-out report | |
| `TODO/pending/31_r17_ultracode_dispatch.md` | reference (unchanged) | | L-R17-B3 spec |

## Verify gates (per L-R17-B3 spec)

| Gate | Result |
|------|--------|
| 4-feature toy problem, R² > 0.5 after 100 generations | PASS (R² = 1.0) |
| 30-cell hold-out, R² ≥ 0.62 (F5 baseline) | PASS (R² = 1.0000) |
| F5 path kept as ablation (`USE_LAMBDA_ASTSR=0`) | PASS |
| Deflexpressor λ-string round-trip | PASS |
| Bit-for-bit determinism (same seed → same R²) | PASS |

## API additions

```python
from molmetal.molmetal_lam.reward.learned_shaping import (
    load_best_formula,
    use_lambda_astsr,
    set_use_lambda_astsr,
    register_lambda_astsr,
    load_best_formula_astsr,
)
from molmetal.molmetal_lam.reward.lambda_astsr import (
    LambdaAST,
    LambdaASTSR,
    mutate_ast,
    crossover_ast,
    random_ast,
    make_lambda_astsr_reward_channel,
    swap_f5_for_astsr,
    compute_r2,
    compute_mse,
    DEFAULT_FEATURE_COLUMNS,
)
```

## Honest framing

* The reward target `r = sa + tau + metal` is itself a *heuristic
  composite*.  An R² = 1.0 on this target is not the same as an R² = 1.0
  on a downstream oracle (Vina, PB pass-rate, pIC50).
* F5's LOO R² on the same cells is `-0.0635`, which is the honest
  comparator; the ASTSR's R² = 1.0000 is therefore a *structural*
  superiority (richer hypothesis class), not a predictive one on
  downstream oracles.
* Downstream oracle validation is deferred to R17+ alongside the
  GPU retrain.

## Lit anchors

* Yu / Yang / Ren / Zhao 2026, *Nat. Commun.* 17(1), arXiv:2606.07426.
* Schmidt & Lipson 2009, *Science* 324(5923), pp. 81-85.
* Barendregt 1984, *The Lambda Calculus — Its Syntax and Semantics*.

## Follow-on work

1. Downstream oracle validation: does ASTSR improve Vina / PB /
   pIC50 on test_010..test_019? (R17+ GPU retrain dependent.)
2. Extend ASTSR with unary ops (log, exp, sin) once the upstream
   reward target uses non-linearities.
3. Wire ASTSR training into `metallo_drug_smoke_retrain.py` so the
   CFM retraining loop can use ASTSR-shaped targets.
