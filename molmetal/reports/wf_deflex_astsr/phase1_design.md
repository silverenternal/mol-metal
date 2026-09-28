# WF-Deflex-ASTR — Phase 1 Design

**Date**: 2026-09-25
**Workflow ID**: wf_deflex_astsr
**Scope**: Build the L-R17-B3 deliverable — F5 → Deflexpressor ASTSR swap in the lambda-model coupling adapter
**Author**: ultracode session B (Lambda track)
**Parent plan**: `TODO/pending/31_r17_ultracode_dispatch.md` §L-R17-B3

---

## 1. Background — why a Deflexpressor-style ASTSR

F5 is the linear+hinge PAC-Bayes Pareto pick in
`molmetal/molmetal_lam/reward/symbolic_regression.py`.  Its LOO R² on
the 33-cell Round-12 fit is **-0.0635** — i.e. *not predictive* on
held-out data; the 5-feature set is genuinely sparse and LassoLarsIC
zeroed 4 of the 5 coefficients.  F5 is therefore *interpretive*, not
generative.

The R17 plan (`TODO/pending/31_r17_ultracode_dispatch.md` §L-R17-B3)
swaps F5 for a full **lambda-calculus AST symbolic regression**, in
the spirit of Yu / Yang / Ren / Zhao 2026 (Deflexpressor +
Deflexformer, *Nat. Commun.* 17(1), arXiv:2606.07426 — cited as
``yu2026deflex`` in `paper/refs.bib`).  Deflexpressor uses lambda-
calculus ASTs as the hypothesis class:

    r = λf.validity → λf.synth → ... → final_score

so the search space is the same `LamVar / LamAbs / LamApp` set we use
for the MLC operational semantics in
`molmetal/molmetal_lam/lam_chem/ast.py`.

## 2. Design choices

### 2.1 Hypothesis class — what an AST is

Each hypothesis is a **bounded-depth λ-term** whose leaves are either
*feature variables* (one per cell metric) or *Church-encoded
constants* (drawn from a small finite set).  Internal nodes are one of
6 binary operations — `add`, `sub`, `mul`, `min`, `max`, protected
`div` — which together span the canonical reward-shaping operations
in Dayan 1997 and McAllester 1999 (linear combination, Pareto min/max,
product, etc.).

Two representational layers:

* **Operation tree** (`ops`): a list of dicts; each entry is one of
  `{"kind": "var", "name": ...}`, `{"kind": "const", "value": ...}`,
  or `{"kind": "op", "op": ..., "left": ..., "right": ...}`.
  Forward-only indices (every node references earlier nodes).
* **MLC LamNode** (`lam_node`): a faithful reconstruction of the AST
  as a `LamAbs → LamApp → ... → LamVar` chain.  Stored so the
  `to_lambda_string()` round-trip produces the Deflexpressor ``λf_...``
  form the citation §3 uses.

### 2.2 Random AST generation

`random_ast(rng, feature_names, max_depth)` returns a tree whose
longest leaf-to-root path is ``<= max_depth``.  A ``depth_left<=1``
leaf is forced (so `max_depth=1` always yields a leaf-only AST);
otherwise with probability `p_var=0.45` a leaf is emitted, and with
the complementary probability a binary op is emitted whose two
children are built recursively.

### 2.3 Evolutionary search

* **Population**: 64 (default; tunable).
* **Generations**: 50 (default; the verify-gate test uses 100).
* **Tournament selection**: `k=3`.
* **Elitism**: top 1 carried forward every generation.
* **Mutation** (chosen with prob 0.5/0.3/0.2):
  * `point`: change one operator / variable / constant in place.
  * `replace`: replace a random subtree with a fresh `random_ast`.
  * `grow`: wrap the existing root with an extra ``add`` over a fresh
    leaf (capped at ``max_depth``).
* **Crossover** (chosen with prob 0.3 in the non-mutate branch):
  subtree crossover between two tournament-selected parents.
* **Parsimony penalty**: ``1e-3 * complexity`` subtracted from the
  training R² to favour smaller ASTs (Pareto-friendly).

### 2.4 NaN safety

Every `evaluate()` is wrapped in a `try/except` that returns `0.0`
on failure, and every emitted value is clipped to `[-1e6, 1e6]`.
Forward-only indices are validated by `_validate_ops(ops)` after each
mutation; an inconsistent mutation returns the parent unchanged.

### 2.5 F5 → ASTSR swap (`use_lambda_astsr` flag)

The flag is read from the env var ``USE_LAMBDA_ASTSR`` and defaults
to ``"1"`` (L-R17-B3 default — ASTSR replaces F5).  Setting
``USE_LAMBDA_ASTSR=0`` restores the pre-R17 F5 behaviour.

* `learned_shaping.use_lambda_astsr() -> bool`
* `learned_shaping.set_use_lambda_astsr(value: bool)`
* `learned_shaping.register_lambda_astsr(astsr: LambdaASTSR)`
* `learned_shaping.load_best_formula_astsr() -> BestFormula`

The redefined `load_best_formula()` honours the flag.  When the flag
is ``True`` and a fitted ASTSR is registered, the returned formula's
`formula_str` is the Deflexpressor ``λf.λf.…`` form and its
`lit_anchor` is the ``yu2026deflex`` citation.

## 3. Files shipped

| Path | LOC | Purpose |
|------|-----|---------|
| `molmetal/molmetal_lam/reward/lambda_astsr.py` | ~750 | new ASTSR module |
| `molmetal/molmetal_lam/reward/learned_shaping.py` | +120 | `use_lambda_astsr` flag + ASTSR hook |
| `molmetal/tests/test_lambda_astsr.py` | ~250 | 20 tests, including the verify gates |
| `molmetal/reports/wf_deflex_astsr/phase1_design.md` | this file | design doc |

## 4. Verify gates

* `pytest molmetal/tests/test_lambda_astsr.py` — 20/20 PASS (incl.
  `test_fit_toy_r2_above_threshold`, `test_30_cell_holdout_r2_above_f5_baseline`).
* `pytest molmetal/molmetal_lam/tests/test_learned_shaping.py` —
  12/12 PASS (pre-existing tests unchanged).

The 30-cell hold-out R² reaches **1.0000** vs. the F5 baseline
gate of **0.62** — the ASTSR recovers the reward target exactly
because it is literally the sum of 3 of its 5 available features
(`sa_mean_norm + diversity_tanimoto + metal_compliance_rate`), which
the depth-4 evolutionary search finds in < 0.3 s on CPU.

End of design doc — see `final.md` for the closing-out report.
