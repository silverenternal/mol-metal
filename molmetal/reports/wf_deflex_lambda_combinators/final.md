# WF-Deflex Lambda Combinators — Final Report

**Status:** WF SHIPPED — Phases 1 + 2 + 3 complete; combinators live in
`lambda_combinators.py`, all 10 named tests pass, integration smoke on a
real CA2 pocket dict passes, no regression introduced.
**Date:** 2026-09-15
**Author:** WF-Deflex orchestrator (deflex sub-agent)
**Scope:** final verdict on the WF-Deflex-Lambda-Combinators workflow.

---

## 0. Honest framing (mandatory)

- **Combinators are abstractions, NOT yet wired into the hot search path.**
  `proof_search.py` is locked by the pocket-invariance combined workflow
  and was not edited in Phases 1-3. The swap-over is a Phase 4 concern
  gated on bit-for-bit output equivalence (N≥1000 random + N≥10 real
  pockets).
- The MLLC-shaped selectors (`ClickRuleCombinator`,
  `MoleculeGenerationCombinator`) are **duck-typed** — they accept any
  object with `name` / `applies_to(pf)` / `score(pf)` / `apply(scaffold,
  tile)` so the module does NOT import any READ-ONLY file.
- All 10 named tests pass in 0.14 s (CPU-only, no GPU/RX 7800 XT touched).
- The integration smoke uses a real CA2 active-site residue snapshot and
  4 duck-typed click-rule mocks; it exercises the full typed-λ pipeline
  on a realistic pocket_features frame shape.
- Pre-existing pocket-invariance test failures are NOT caused by this WF.
  They are tracked under the pocket-invariance combined workflow.
- Total wall for Phase 3 (full pytest + integration smoke): ~138-157 s
  (depending on whether `test_pocket_macro_skeleton.py` is bundled).

---

## 1. What shipped (Phases 1-3)

### 1.1 NEW files

| File | Purpose | LOC |
|---|---|---|
| `molmetal/reports/wf_deflex_lambda_combinators/phase1_design.md` | Phase 1 design — 5 imperative nested-loop sites catalogued | ~310 |
| `molmetal/reports/wf_deflex_lambda_combinators/phase2_combinators.md` | Phase 2 report — 10/10 tests pass, lit anchors | ~310 |
| `molmetal/reports/wf_deflex_lambda_combinators/phase3_integration.md` | Phase 3 report — full pytest + integration smoke | ~315 |
| `molmetal/reports/wf_deflex_lambda_combinators/final.md` | This file | ~280 |
| `molmetal/molmetal_lam/lam_chem/lambda_combinators.py` | Typed-λ combinator library | ~470 |
| `molmetal/molmetal_lam/tests/test_lambda_combinators.py` | 10 named tests | ~270 |

Total: 6 NEW files, ~1,955 lines (including report + code + tests).

### 1.2 Public surface of `lambda_combinators.py`

| Symbol | Lit anchor | Signature |
|---|---|---|
| `map_combinator(fn, xs)` | Reynolds 1972 | `(A→B, [A]) → Iterator[B]` |
| `fold_combinator(fn, init, xs)` | Bird 1988 + Pfenning 2001 | `(B×A→B, B, [A]) → B` |
| `fold_right_combinator(fn, init, xs)` | Bird 1988 | `(A×B→B, B, [A]) → B` |
| `filter_combinator(pred, xs)` | Reynolds 1972 | `(A→bool, [A]) → Iterator[A]` |
| `zip_with_combinator(fn, xs, ys)` | Meijer 1991 (lens) | `(A×B→C, [A], [B]) → Iterator[C]` |
| `compose(f, g)` | Reynolds 1972 | `(B→C, A→B) → (A→C)` |
| `nest(outer, inner)` | NEW (curried HOF) | `(λ×[A]→[B], A→B) → ([A] → [B])` |
| `ClickRuleCombinator.select_rules_for_pocket(pf, rules)` | Reynolds + Bird + Meijer | `(pf, [Rule]) → [Rule]` |
| `MoleculeGenerationCombinator.generate_molecule(scaffold, rules, tiles)` | nest(map, fold) | `(scaffold, [Rule], [Tile]) → [SMILES]` |

### 1.3 Test results

```
$ uv run pytest molmetal/molmetal_lam/tests/test_lambda_combinators.py -v
10 passed, 1 warning in 0.14s

$ uv run pytest molmetal/molmetal_lam/tests/ --tb=short -q
1 failed, 850 passed, 3 skipped, 1 xpassed in 138.05s   # 1 pre-existing pocket-invariance failure
```

### 1.4 Files NOT touched (READ-ONLY contract)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reactions/beta_reductions.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/warm_start.py`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/learned_prior.py`

---

## 2. Workflow contract (Phases 1 → 2 → 3)

### Phase 1 (Design)

- Inventoried 5 imperative nested-loop sites across the 4 READ-ONLY files.
- Chose typed-lambda-calculus abstractions over a Python monad library
  (Reynolds 1972 defunctionalisation is enough).
- Designed the `nest` combinator as curried `nest(outer, inner)(coll) =
  outer(inner, coll)` (note: this differs from `compose(f, g)(x) = f(g(x))`).
- Specified the duck-typing contract so the new module does NOT import any
  READ-ONLY file.

### Phase 2 (Combinator Implementation)

- Shipped `lambda_combinators.py` + `test_lambda_combinators.py`.
- 10 named tests pass in 0.17 s (now 0.14 s in Phase 3 re-verify).
- The `nest` semantics went through 2 iterations during Phase 2; final
  form is Draft 2 (curried HOF) — see phase2_combinators.md §3.2.
- 4 MLLC-shaped convenience selectors ship; duck-typed for backward-compat.

### Phase 3 (Integration)

- Full pytest on `test_lambda_combinators.py` — 10/10 pass.
- Full pytest on `molmetal_lam/tests/` — 850 pass + 3 skip + 1 xpass + 1 fail
  (pre-existing pocket-invariance failure, NOT caused by this WF).
- Integration smoke on a real CA2 pocket dict (HIS92/94/119 + VAL96/121
  + Zn²⁺ active centre) with 4 click-rule mocks — all 4 HOFs fire correctly.
- Phase 4 refactor candidates identified (10-16 h total effort, gated on
  bit-for-bit output equivalence).

---

## 3. Refactor candidates for Phase 4

**Honest framing**: swap-over is **GATED** on bit-for-bit output equivalence
(N≥1000 random + N≥10 real pockets). None of these edits happen in Phase 3.

### 3.1 `proof_search.py` (priority: HIGH)

| Site | Combinator | Effort |
|---|---|---|
| L2680-2759 (per-iteration history block in `search()`) | `fold_combinator` + State-monad wrapper | 4-6 h |
| L2745 (collision-rate scan) | `zip_with_combinator` | 0.5 h |
| L2885 (oracle top-k skip) | `filter_combinator` + `map_combinator` | 0.5 h |
| L2917 (top-k extract) | `map_combinator` | 0.25 h |
| L3063 (depth-indexed path scan) | `zip_with_combinator` | 0.5 h |
| L4594, L4685 (small list comprehensions) | `map_combinator` | 0.25 h |

Sub-total: 6-8 h for `proof_search.py`.

### 3.2 `beta_reductions.py` (priority: HIGH — most uniform)

| Site | Combinator | Effort |
|---|---|---|
| L116-120 (`_count_heavy_atoms`) | `flat_map` | 0.5 h |
| L481-511 (product_sets loop) | `flat_map` + `fold_combinator` | 1.0 h |
| L665, L731, L793, L851, L1074, L1140, L1236, L1323 (8× `for p in products`) | `fold_combinator` | 2.0 h |
| L902-922 (ThiolEne SMARTS nested atoms/bonds) | `flat_map` | 0.5 h |

Sub-total: 4 h for `beta_reductions.py`.

### 3.3 `warm_start.py` (priority: MEDIUM)

| Site | Combinator | Effort |
|---|---|---|
| L310 (`for r in pocket_residues:`) | `map_combinator` | 0.5 h |
| L493 (`for i, a in enumerate(actions):`) | `zip_with_combinator` | 0.5 h |

Sub-total: 1 h.

### 3.4 `learned_prior.py` (priority: MEDIUM)

| Site | Combinator | Effort |
|---|---|---|
| L183, L235 (vocab + rule indexing) | `map_combinator` | 0.5 h |
| L418, L461 (per-state scan) | `fold_combinator` + `filter_combinator` | 1.0 h |
| L470 (`for epoch in range(epochs):` training loop) | `fold_combinator` (per-epoch loss as accumulator state) | 2.0 h |

Sub-total: 3-5 h.

### 3.5 Phase 4 combined

- Total: 14-18 h
- Risk profile: LOW (pure refactor, no behaviour change)
- Gating: equivalence test on N≥1000 random + N≥10 real pockets
- Owner: separate workflow (would need to re-open `proof_search.py` edit
  permission after the pocket-invariance combined workflow closes)

---

## 4. Lit anchors per combinator (recap)

- **Reynolds 1972** — typed lambda calculus + defunctionalisation;
  justifies `map_combinator`, `filter_combinator`, `compose`.
  → Reynolds, J. C. *Definitional Interpreters for Higher-Order
    Programming Languages*. Proc. ACM National Conference 1972, pp.
    717-740.
- **Pfenning 2001** — higher-order judgment + frames;
  justifies `fold_combinator` as the judgment-level iteration.
  → Pfenning, F. & Davies, R. *A Judgmental Reconstruction of Modal
    Logic*. Mathematical Structures in Computer Science 11(4), 2001,
    pp. 511-540.
- **Bird 1988** — promotion theorem + fold fusion;
  justifies `fold_combinator` (strict left fold) vs
  `fold_right_combinator` (right fold) and the fusion law
  `foldr op e (map f xs) = foldr op' e' xs`.
  → Bird, R. S. *Introduction to the Theory of Lists*. Tech. report,
    Programming Research Group, Oxford, 1988.
- **Meijer 1991** — bananas / lenses / envelopes / barbed wire;
  justifies `zip_with_combinator` (lens), `concatMap` (banana),
  score-desc sort (envelope with max-monoid).
  → Meijer, E., Fokkinga, M., Paterson, R. *Functional Programming with
    Bananas, Lenses, Envelopes and Barbed Wire*. FPCA 1991, LNCS 523,
    pp. 124-144.
- **NEW**: `nest(outer, inner)(coll) = outer(inner, coll)` is the
  curried HOF nesting shape — not classical, but the natural reading of
  `nest(map_combinator, fold_combinator)` for cartesian apply+accumulate
  in a typed-λ setting where `map_combinator: (A→B, [A]) → [B]` and
  `fold_combinator: (B×A→B, B, [A]) → B`.

---

## 5. Verdict

- **Combinators ship**: 10/10 named tests pass, integration smoke passes
  on a real pocket dict, no regression introduced.
- **Combinators are NOT yet wired** into the hot search path (per the
  workflow contract — `proof_search.py` is locked by the pocket-invariance
  combined workflow). They live as **abstractions** that future work can
  swap into the imperative nested loops.
- **Phase 4 refactor candidates** are catalogued with file:line refs and
  effort estimates (14-18 h total). The swap-over is gated on a property
  test that proves bit-for-bit output equivalence on N≥1000 random +
  N≥10 real pockets.
- **No GPU / RX 7800 XT touched** during any phase. All work is CPU-only.
- **No new dependencies** introduced (Phase 2 + Phase 3).
- **No READ-ONLY file modified** in Phases 1-3.

---

## 6. File paths (absolute)

- NEW source: `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/lambda_combinators.py`
- NEW tests: `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_lambda_combinators.py`
- Phase 1 design: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_deflex_lambda_combinators/phase1_design.md`
- Phase 2 report: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_deflex_lambda_combinators/phase2_combinators.md`
- Phase 3 report: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_deflex_lambda_combinators/phase3_integration.md`
- Final report: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_deflex_lambda_combinators/final.md` (this file)
- READ-ONLY (NOT modified in Phases 1-3):
  - `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py`
  - `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reactions/beta_reductions.py`
  - `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/warm_start.py`
  - `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/learned_prior.py`

---

**End of WF-Deflex Lambda Combinators final report.**
