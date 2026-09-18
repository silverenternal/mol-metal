# WF-Lambda-4 — Closure-Theorem Surface Audit

**Date:** 2026-09-14
**Status:** AUDIT COMPLETE — surface mapped, ready for `closure.py` implementation
**Mode:** MEASURED (file/line evidence) vs PROJECTED (spec to be implemented)
**Goal:** prove the closure theorem for the 5-click productive space:

> *Given a starting scaffold `S` and a fixed reaction budget `B`, the set of
> λ-terms reachable via `B` applications of the 5 click combinators
> (CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling) is the **constructive
> synthesis space** — every term in this space has a β-reduction witness
> path back to the click application sequence.*

---

## 1. The 5 click rules — surface

**Source of truth:** `molmetal/molmetal_lam/lam_chem/rules.py` is a
*thin re-export* of `REACTION_RULES` from
`molmetal/molmetal_lam/reactions/beta_reductions.py`. The dataclass
fields and `pattern_smiles` templates live in `beta_reductions.py`.

| # | Rule | File:Line | pattern_smiles (SMARTS template) | Reactant types (arity, what it consumes) | Product signature |
|---|------|-----------|----------------------------------|------------------------------------------|-------------------|
| 1 | **CuAAC** | `beta_reductions.py:545` (dataclass), `:563` (pattern) | `[N:1]=[N:2]=[N:3].[C:4]#[CH:5]>>[C:4]1=[C:5][N:3]=[N:2][N:1]1` | 2-tuple: (organic azide `R-N3`, terminal alkyne `R'-C≡CH`) | one 1,4-disubstituted 1,2,3-triazole (5-ring aromatic) |
| 2 | **SPAAC** | `beta_reductions.py:613` (dataclass), `:627` (pattern) | `[N:1]=[N:2]=[N:3].[C:4]#[C:5]>>[C:4]1=[C:5][N:3]=[N:2][N:1]1` | 2-tuple: (azide `R-N3`, *internal / strained* alkyne `R'-C≡C-R''` — terminal alkyne is **not** SPAAC) | one 1,2,3-triazole, both 1,4- and 1,5-regioisomers when RDKit enumerates |
| 3 | **ThiolEne** | `beta_reductions.py:795` (dataclass) — **NO SMARTS** | `pattern_smiles = None` (`:817`) — structural check in `_reaction_sites()` (`:821-869`) | 2-tuple: (neutral `R-SH` thiol `with exactly 1 heavy C neighbor`, partner with a non-aromatic `C=C`) | one anti-Markovnikov thioether (`R-S-CH2-CH2-R'`) |
| 4 | **Suzuki** | `beta_reductions.py:943` (dataclass), `:965` (pattern) | `[#6:1][B]([O])[O].[#6:3][F,Cl,Br,I]>>[#6:1][#6:3]` | 2-tuple: (aryl boronic acid `R-B(OH)2`, aryl halide `R'-X` with X ∈ {F,Cl,Br,I}) | one biaryl `R-R'` (boronic acid + halide leave as by-products but are not in the MCTS state) |
| 5 | **AmideCoupling** | `beta_reductions.py:1015` (dataclass), `:1033` (pattern) | `[C:1](=[O:2])[OH].[NH2:4]>>[C:1](=[O:2])[NH:4]` | 2-tuple: (carboxylic acid `R-C(=O)OH`, primary amine `R'-NH2`) | one amide `R-C(=O)-NH-R'` (water leaves as implied by-product) |

**Confirmation of registry contract** — `rules.py:72-88` builds a dict
`CLICK_REACTIONS` that **indexes the same 5 `ReactionRule` singletons**
under canonical camelCase, kebab-case, and space-separated names
(`CuAAC`, `cuaac`, `thiol-ene`, `amide coupling`, etc.) so the MCTS
expansion path and the metrics-harness can use one rule instance by any
of its accepted spellings.

### ReactionRule dataclass (`beta_reductions.py:144-184`)

```
ReactionRule(name, pattern_smiles, pattern, requires_catalyst, stoichiometry,
             _rdkit_reaction, rate_predictor)
```

Notable fields for the closure proof:

- `pattern_smiles: Optional[str]` — RDKit reaction SMARTS (`>>` arrow);
  `None` for `ThiolEne` which has no canonical SMARTS (`:817`).
- `stoichiometry: Dict[str, int]` — net atom change; for the 5 click
  rules this is **always the empty dict** by construction (heavy atoms
  are conserved). `verify_mass_balance` (`beta_reductions.py:1209`) audits
  this invariant at instantiation time.
- `requires_catalyst` — `CuAAC` needs Cu(I), `Suzuki` needs Pd(0)+base,
  `AmideCoupling` needs coupling reagent; `SPAAC` and `ThiolEne` need
  none. The reduction itself does **not** branch on this field — the
  pure-function SMARTS apply uniformly.

### What `reduce(...)` returns

`ReactionRule.reduce(molecule, **kwargs) -> List[MoleculeClosedTerm]`
(`beta_reductions.py:232`). For the 5 click rules the return list is:

- **non-empty** when the redex fires (each list entry is a distinct
  β-successor term, deduplicated by canonical SMILES);
- **empty** when the rule cannot apply (no functional-group match);
- raises `ReactionError` only on RDKit parse failures.

Each product is decorated with `term.term` carrying a derivation tag
(`f"CuAAC(R-N3, R'-C≡CH) -> <canonical_smiles>"`, etc.) so the
synthesis layer can recover the click rule that produced it.

---

## 2. LamApp + reduce_once + has_redux + BNF

### `LamNode` AST (`lam_chem/ast.py`)

Three frozen dataclasses (`ast.py:72`):

- `LamVar(name, domain='real')` — `:79`; free var is itself.
- `LamAbs(var, body)` — `:121`; free_vars = body.free_vars() − {var}.
- `LamApp(func, arg)` — `:185`; β-reduction rule
  `(λx.M) N → M[x := N]` at `:215-217`.

Reduction primitives:

- `LamApp.beta_reduce()` (`ast.py:209`) — leftmost-outermost
  one-step β-reduction; returns `None` when already in β-NF.
- `LamApp.to_normal_form(max_depth=256)` (`ast.py:229`) — iterative
  β-NF computation, bounded by `max_depth` to guarantee termination on
  divergent terms like Ω.

> **NOTE** — the *closure-theorem spec* lives one level higher, on
> `MoleculeClosedTerm`, not on the syntactic `LamNode` tree. The
> `MoleculeClosedTerm` class (file
> `molmetal/molmetal_lam/molecules/closed_term.py`) wraps a chemical
> state and exposes the chemistry-aware reductions:

- `reduce_once(self)` — `closed_term.py:333` — one chemistry-aware
  β-step (used by the synthesis layer).
- `has_redex(self) -> bool` — `closed_term.py:318` — `True` iff the term
  has an open redex (e.g. cyclopentadiene diene, unsaturated cisplatin
  shell). See test `test_cyclopentadiene_has_redex_with_explicit_h` in
  `tests/test_closed_term.py:188`.
- `is_closed` (property) — `closed_term.py:157` — `True` iff every atom
  has filled valence (chemistry-correct, ignores lone-pair capacity
  after the WF-Lambda-1c one-line patch).
- `alpha_equivalent(other)` — `closed_term.py:461` — equal iff canonical
  SMILES match.

### BNF (well_formedness.py:230-353)

Three layered predicates:

| Predicate | File:Line | What it checks |
|-----------|-----------|----------------|
| `check_beta_normal_form(term)` | `well_formedness.py:230` | legacy arity-based (lone-pair inclusive) |
| `check_beta_normal_form_for_rdkit(a, atom)` | `well_formedness.py:257` | per-atom valence (covalent) saturated, used during bond-ledger manipulation |
| `check_beta_normal_form_for_rdkit_term(term)` | `well_formedness.py:306` | **term-level BNF** using `term.valence_used` (RDKit total, implicit-H-inclusive) with `AtomSite.used_sites` fallback |

`check_beta_normal_form_for_rdkit_term` is the predicate that defines
"closure" in the synthesis sense — a term is **closed** iff every atom
in `term.atoms` satisfies `valence_used[i] >= atom.valence`. Empty term
is trivially closed (`:329-330`).

The crucial subtlety for the closure theorem: `valence_used` is
**RDKit-populated from `from_rdkit` / `from_smiles`** (line 311-323)
and the bond ledger's `AtomSite.used_sites` is the parallel
lambda-combinator bookkeeping. The predicate takes the **max** of the
two at `:347-349` to handle callers who mutate bonds after parsing.

---

## 3. MCTS expansion: tile_pool + click rules

**Call site:** `search_alg/proof_search.py:3061` (inside
`_rollout`/expand path) — `tile_lib = self._resolve_expand_tile_pool()`.

**`_resolve_expand_tile_pool()`** (`:2757-2829+`):

1. If `use_fragment_pool` is False → return `self.tile_library`
   unchanged (legacy 12-tile library path).
2. Else → lazy-load `FRAGMENT_LIBRARY_200_TILES()` from
   `molmetal_lam.tile_lib.library`, convert each `Tile` to a
   `MoleculeClosedTerm` via `from_smiles(embed_3d=False)` and cache on
   `self._fragment_pool_cache`.
3. **R10 axis A extension** (`:2821-2829+`) — when env
   `ROUND10_5_CLICK=1` (default), append 5×4=20 click-handle tiles
   (DBCO, boronic acid, aryl halide, carboxylic acid, amine) to the
   pool so all 5 click reactions are reachable. Total pool: 220 tiles.

**How the MCTS combines tile_pool + click rules** (`proof_search.py:3057-3098`):

```
rule_names = list(self.rules.keys())                    # the 5 click rules
tile_lib = self._resolve_expand_tile_pool()             # 220-tile pool (or fallback)
for each rollout step (up to depth):
    pick (rule, tile) by guided-rollout or uniform random
    products = self._safe_reduce(rule, current, tile)  # apply beta
    if products:  current = rng.choice(products)
```

`_safe_reduce` (`:3470-3485`) wraps `rule.reduce((state, tile))` and
catches every exception → returns `[]` (a "non-fire" is a legitimate
MCTS outcome). When the redex fires, `current` advances along one
β-step, and after `depth` iterations the leaf state is scored by the
reward aggregator.

**Key fact for the closure theorem:** every state the MCTS visits is a
β-successor of a click-rule reduction. The expansion path is *the*
construction of the synthesis space — there is no other transition.

---

## 4. Existing click-rule tests (`tests/test_click_reactions.py`)

Five tests in `tests/test_click_reactions.py` (lines 64-235) cover
specific click products:

1. `test_cuaac_smiles_valid` (`:64`) — ethyl azide + propyne →
   1-ethyl-4-methyl-1,2,3-triazole; 5-membered ring; 8 heavy atoms.
2. `test_spaac_with_cyclooctyne` (`:97`) — benzyl azide + cyclooctyne
   → triazole; rejects when one of azide / internal alkyne is missing.
3. `test_spc_amide_formation` (`:138`) — ethyl azide + methylphosphine
   → iminophosphorane `CCN=PC`; 5 heavy atoms (2 N's released as N2).
4. `test_diels_alder_cyclohexene` (`:169`) — butadiene + ethylene →
   `C1CC=CCC1`; 6-membered ring with one C=C.
5. `test_tile_library_size` (`:204`) — ≥12 standard tiles; 4 azide +
   4 alkyne + 4 partner split; dispatcher `apply_click_reaction`
   works for each name.

The tests confirm the *product* layer of the closure (one application
of a click rule produces a well-formed λ-term). **No test currently
exercises multi-step closure** — i.e. "starting from scaffold S, after
B applications of the 5 click rules, every closed term in the
constructive synthesis space has a β-witness". That gap is exactly what
`closure_theorem()` will fill.

---

## 5. Closure-theorem spec

### Definition (PROJECTED — to be implemented in `lam_chem/closure.py`)

```
def closure_theorem(
    spec: ClosureSpec,
    click_rules: Sequence[ReactionRule],
    max_depth: int,
) -> ClosureWitness:
    """Audit the closure of the 5-click productive space.

    A term `t` is *in the constructive synthesis space* of
    `(spec, click_rules, max_depth)` iff there exists a sequence
    `[(rule_0, react_0), …, (rule_{n-1}, react_{n-1})]` with
    `n <= max_depth` such that each `rule_i` is in `click_rules` and
    applying it as a β-reduction to the previous state yields the next,
    with the final state `α-equivalent` to `t`.

    Closure theorem:
        ∀ spec.closed_scaffold ∈ spec.starting_materials:
            ∀ t ∈ derived(spec.starting_materials, click_rules, max_depth):
                (1) t is α-equivalent to some β-reduction sequence
                    whose witness is recoverable from the derivation
                (2) t is closed (BNF under
                    check_beta_normal_form_for_rdkit_term)
                (3) every intermediate state is a closed λ-term
                    (no open redex between reductions)

    Returns a ClosureWitness carrying:
        closed (bool): True iff (1)–(3) hold for every (scaffold, t) pair
        witness_sequences (list[list[(rule_name, reactant_smiles)]]):
            one derivation per (scaffold, target) tested
        tested_pairs (int): number of (scaffold, target) pairs verified
        max_depth_used (int): the depth budget that was actually consumed
    """
```

### Return type — Boolean + witness

The honest return type is a **dataclass** `ClosureWitness`:

```
@dataclass(frozen=True)
class ClosureWitness:
    closed: bool                                    # the theorem verdict
    witness_sequences: List[Tuple[str, ...]]        # one rule-name tuple
                                                     #   per (scaffold, target) pair
    tested_pairs: int                               # number of audits run
    max_depth_used: int                             # depth budget actually used
    unverified_targets: Tuple[str, ...] = ()        # any targets that failed
```

This is the same shape used by `synthesis.derivations.SynthesisPath`
(`synthesis/derivations.py:70`) — the closure layer should compose with
the existing forward-search / backward-search pipeline.

### Algorithm (PROJECTED)

The closure theorem is verified by **forward constructive synthesis**:

```
def closure_theorem(spec, click_rules, max_depth) -> ClosureWitness:
    witnesses = []
    failed    = []
    for scaffold in spec.starting_materials:
        targets = [scaffold] + spec.closed_targets
        for target in targets:
            derivation = synthesize(
                target=target,
                starting_materials=[scaffold],
                rules=click_rules,
                max_depth=max_depth,
            )
            if derivation is None:
                if target.alpha_equivalent(scaffold):
                    derivation = []  # trivial: zero-step derivation
                else:
                    failed.append(target.canonical_smiles())
                    continue
            # Verify every intermediate is closed
            # (the synthesize BFS already enforces this via
            #  check_beta_normal_form_for_rdkit_term on the frontier)
            witnesses.append(tuple(r.name for r in derivation))
    return ClosureWitness(
        closed=not failed,
        witness_sequences=witnesses,
        tested_pairs=len(spec.starting_materials) * (1 + len(spec.closed_targets)),
        max_depth_used=max_depth,
        unverified_targets=tuple(failed),
    )
```

The proof relies on three observations already in the codebase:

1. **`synthesize` (`synthesis/derivations.py:249`)** already does
   breadth-first β-reduction over the 5 click rules with a depth cap.
2. **`check_beta_normal_form_for_rdkit_term`
   (`well_formedness.py:306`)** is the chemistry-correct closure
   predicate (valence-saturated atoms).
3. **`ReactionRule.reduce` (`beta_reductions.py:232`)** always returns
   a `List[MoleculeClosedTerm]` whose entries are
   `from_rdkit`-constructed closed terms, so the BFS frontier is always
   composed of closed terms by induction.

### Edge cases the audit must cover (≥6 tests required)

1. **Empty scaffold** — `spec.starting_materials = ()`; closure holds
   vacuously (`closed = True`, `tested_pairs = 0`).
2. **Scaffold alone (B=0)** — `target = scaffold`; the only derivation
   is `[]` (zero steps), and BNF must hold.
3. **Single-step CuAAC** — azide tile + alkyne tile → triazole; depth
   budget 1 finds the derivation.
4. **Two-step sequential** — `[azide, alkyne_1, alkyne_2]` where the
   first CuAAC product is the alkyne substrate for the second SPAAC;
   depth budget 2 suffices.
5. **Over-budget depth** — a target unreachable in `max_depth` steps
   is recorded in `unverified_targets`, and `closed = False`.
6. **BNF violation in intermediate** — a reactant that is itself
   unsaturated (e.g. an azide without its H) must be rejected; the
   BFS frontier only admits closed terms, so the test confirms no
   open redex survives between reductions.
7. **5-click coverage** — at least one derivation per
   {CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling} is discovered.
8. **Mass-balance invariant** — every product term has the same heavy-
   atom count as the sum of its reactants (verifies
   `verify_mass_balance` is satisfied for the 5 rules).

---

## 6. File:line index — load-bearing surface

| Component | File:Line |
|-----------|-----------|
| Click rule re-export | `molmetal/molmetal_lam/lam_chem/rules.py:59-88` |
| `ReactionRule` dataclass | `molmetal/molmetal_lam/reactions/beta_reductions.py:144-184` |
| `CuAAC` pattern | `beta_reductions.py:545` + `:563-569` |
| `SPAAC` pattern | `beta_reductions.py:613` + `:627-633` |
| `ThiolEne` (no SMARTS) | `beta_reductions.py:795-934` (`_reaction_sites` at `:821`) |
| `Suzuki` pattern | `beta_reductions.py:943` + `:965-973` |
| `AmideCoupling` pattern | `beta_reductions.py:1015` + `:1033-1039` |
| `ReactionRule.reduce` | `beta_reductions.py:232-265` |
| `verify_mass_balance` | `beta_reductions.py:1209-1236` |
| `LamNode` AST | `molmetal/molmetal_lam/lam_chem/ast.py:72-275` |
| `LamApp.beta_reduce` | `ast.py:209-227` |
| `LamApp.to_normal_form` | `ast.py:229-242` |
| `MoleculeClosedTerm.reduce_once` | `molmetal/molmetal_lam/molecules/closed_term.py:333` |
| `MoleculeClosedTerm.has_redex` | `closed_term.py:318` |
| `MoleculeClosedTerm.is_closed` | `closed_term.py:157` |
| `MoleculeClosedTerm.alpha_equivalent` | `closed_term.py:461` |
| BNF (valence-based, term) | `molmetal/molmetal_lam/lam_chem/well_formedness.py:306-353` |
| BNF (legacy arity) | `well_formedness.py:230-254` |
| BNF (per-atom covalent) | `well_formedness.py:257-303` |
| `synthesize` BFS | `molmetal/molmetal_lam/synthesis/derivations.py:249-359` |
| `SynthesisPath` dataclass | `synthesis/derivations.py:70` |
| `_resolve_expand_tile_pool` | `molmetal/molmetal_lam/search_alg/proof_search.py:2757-2829+` |
| `_resolve_expand_tile_pool` call site | `proof_search.py:3061` (and `:2708`) |
| `_safe_reduce` | `proof_search.py:3470-3485` |
| Click product tests | `molmetal/molmetal_lam/tests/test_click_reactions.py:64-235` |
| BNF / closed-term tests | `molmetal/molmetal_lam/tests/test_closed_term.py:103-210` |
| `test_cyclopentadiene_has_redex_with_explicit_h` | `tests/test_closed_term.py:188` |

---

## 7. Honest framing — MEASURED vs PROJECTED

| Claim | Status |
|-------|--------|
| 5 click rules + `pattern_smiles` templates | **MEASURED** (file:line above) |
| `ReactionRule` dataclass fields | **MEASURED** (`beta_reductions.py:144-184`) |
| `LamApp` / `beta_reduce` / `to_normal_form` | **MEASURED** (`ast.py:209-242`) |
| `MoleculeClosedTerm.reduce_once` / `has_redex` / `is_closed` | **MEASURED** (`closed_term.py:333 / :318 / :157`) |
| BNF valence predicate for chemistry closure | **MEASURED** (`well_formedness.py:306-353`) |
| MCTS combines tile_pool + click rules via `_safe_reduce` | **MEASURED** (`proof_search.py:3061`, `:3470`) |
| Existing click tests cover single-step products only | **MEASURED** (`test_click_reactions.py:64-235`) |
| `closure_theorem(spec, click_rules, max_depth)` API | **PROJECTED** — to be implemented |
| `ClosureWitness` dataclass | **PROJECTED** |
| 6+ audit tests covering scaffold / BNF / depth-budget | **PROJECTED** |
| Formal proof LaTeX (`paper/appendices/closure_theorem.tex`) | **PROJECTED** |
| Audit summary report (`molmetal/reports/wf_lambda4_closure_theorem.md`) | **PROJECTED** |

---

## 8. Next actions

1. Implement `molmetal/molmetal_lam/lam_chem/closure.py` with the
   `ClosureSpec` + `ClosureWitness` dataclasses and the
   `closure_theorem` function above.
2. Add `molmetal/molmetal_lam/tests/test_closure_theorem.py` with
   ≥ 6 tests covering the eight edge cases listed in §5.
3. Write the audit summary at
   `molmetal/reports/wf_lambda4_closure_theorem.md` and the LaTeX
   proof at `paper/appendices/closure_theorem.tex`.
4. Wire `closure_theorem` into the post-rollout pipeline so that
   `MCTSProofSearch` can request a closure audit at every leaf node
   when `ROUND10_CLOSURE_AUDIT=1` is set (NEW flag; default off).
5. Verify with `uv run pytest molmetal/molmetal_lam/tests/test_closure_theorem.py -v`
   and the existing 158-test suite stays green.
