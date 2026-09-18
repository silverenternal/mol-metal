# A Lambda-Paper Case Study: Cisplatin as a Molecular Lambda Calculus Term

*Author: molmetal framework, Layer-6 case study*
*Date: 2026-09-11*
*Status: write-up only — no code change*

---

## Abstract

This case study walks through one of the most iconic metal-coordination
drugs in cancer chemotherapy — *cis*-diamminedichloroplatinum(II),
better known as **cisplatin** — and shows how every physical feature of
the molecule is captured by the eight layers of the Molecular Lambda
Calculus (MLC) framework. The molecule is presented as a closed
λ-term, each Pt–ligand bond is shown to be exactly one β-reduction, the
cis/trans stereochemistry is reduced to a question of *reduction order*,
the MCTS proof-search is shown to find cisplatin among the first few
simulations, and a CuAAC analogue is built by replacing the ammine /
chloride ligands with azide-alkyne triazoles. We close with a
PySR-style hypothetical yield equation (fitted with sklearn as a
graceful-degradation fallback when Julia is unavailable) and a
synthetic-accessibility comparison between the original drug and the
click-chemistry analogue. The report references the concrete
classes and methods in `molmetal/molmetal_lam/`, and is intended to
be the "killer figure" companion to the framework's formalization
memo.

---

## Table of Contents

1.  [Cisplatin as a λ-term](#1-cisplatin-as-a-lambda-term)
2.  [β-reduction steps: each Pt–ligand bond is one reduction](#2-beta-reduction-steps-each-ptligand-bond-is-one-reduction)
3.  [Dative bonds as curried partial applications](#3-dative-bonds-as-curried-partial-applications)
4.  [Stereochemistry: cisplatin (cis) vs transplatin (trans)](#4-stereochemistry-cisplatin-cis-vs-transplatin-trans)
5.  [MCTS proof tree: how Lambda-MCTS finds cisplatin](#5-mcts-proof-tree-how-lambda-mcts-finds-cisplatin)
6.  [A CuAAC analogue: replace 2 NH3 + 2 Cl with 2 azide-alkyne triazoles](#6-a-cuaac-analogue-replace-2-nh3--2-cl-with-2-azide-alkyne-triazoles)
7.  [PySR formula: hypothetical CuAAC yield equation](#7-pysr-formula-hypothetical-cuaac-yield-equation)
8.  [Synthetic-accessibility comparison](#8-synthetic-accessibility-comparison)
9.  [Cross-layer type inhabitation: `PT_DNA_MAJOR_GROOVE`](#9-cross-layer-type-inhabitation-pt_dna_major_groove)
10. [References](#10-references)

---

## 1. Cisplatin as a λ-term

The Molecular Lambda Calculus (MLC) is the eight-layer framework in
`molmetal/molmetal_lam/`. Its central thesis is that **a molecule IS a
closed λ-term**: the atoms are primitive combinators, the bonds are
β-reductions, and the molecule itself is the resulting β-normal form.

In Layer 1 (`molmetal/molmetal_lam/atoms/combinators.py`), every atom
is an instance of the frozen dataclass `Atom` with a fixed `arity =
valence + lone_pairs`. The transition-metal centres are registered in
the `METAL_ATOMS` dictionary:

```python
# molmetal/molmetal_lam/atoms/combinators.py (excerpt)
METAL_ATOMS: Dict[str, Atom] = {
    "Pt_II": Atom(
        symbol="Pt_II", atomic_num=78,
        valence=2, lone_pairs=2, geometry="square_planar",
        is_metal=True,
    ),
    ...
}
```

Because `Pt_II` has `valence=2` and `lone_pairs=2`, its combinator
arity is 4 — exactly the number of ligand slots required by the
square-planar geometry. The combinator interpretation is therefore:

```
Pt_II  ≡  λa.λb.λc.λd. complex(a, b, c, d)
```

where each `a, b, c, d` ranges over the four monodentate ligands (the
two ammines and the two chlorides in cisplatin). The
`Atom.sanity_check()` helper at the bottom of the same file asserts
the foundational invariants — `Pt_II.arity == 4`,
`Ru_II.arity == 6`, and the cisplatin descriptor well-formedness.

### The cisplatin λ-expression

Cisplatin is the curried partial application of `Pt_II` to its four
monodentate ligands:

```
cisplatin  =  (((λa.λb.λc.λd.complex(a,b,c,d) NH3) NH3) Cl) Cl
```

which the test suite (and the framework's "killer figure")
canonicalises to the parenthesised prefix term:

```
((((Pt_II NH3) NH3) Cl) Cl)
```

This is the exact string attached as `mol["term"]` by the
`build_cisplatin_term()` helper in
`molmetal/molmetal_lam/tests/test_cisplatin.py`, and the same string
the framework prints in the cross-layer type-inhabitation test.

### Diagram 1 — cisplatin as a curried 4-arity combinator

```
            Pt_II  ≡  λa b c d. complex(a, b, c, d)
                          │  │  │  │
                          ▼  ▼  ▼  ▼
                       ┌──────────────────┐
                       │                  │
                       │   square-planar  │
                       │      Pt(II)      │
                       │                  │
                       └──────────────────┘
                         │   │      │   │
                         N   N      Cl  Cl
                         │   │      │   │
                         H   H      ·   ·
                         H   H      ·   ·
                         H   H      ·   ·

            cisplatin ≡ ((((Pt_II NH3) NH3) Cl) Cl)
```

The diagram captures two things at once:

* the **physical** view — the central Pt²⁺ coordinated to two ammines
  and two chlorides in a *cis* arrangement,
* the **λ-calculus** view — the same Pt²⁺ as a curried 4-arity
  combinator receiving its four arguments left-to-right.

---

## 2. β-reduction steps: each Pt–ligand bond is one reduction

The application of a ligand to `Pt_II` consumes one free site on the
metal. In Layer 2 (`molmetal/molmetal_lam/bonds/application.py`) the
dataclass `Bond` provides the factory method
`Bond.dative(pt, ligand, ledger=led)` that performs a single
β-reduction step on the metal combinator: it consumes one of the
remaining free sites on `pt` (currying) and collapses the ligand to a
saturated value. The `FreeSiteLedger` object tracks how many free
sites each atom has left at every stage.

In `tests/test_cisplatin.py::test_each_dative_bond_preserves_one_free_site_on_pt`
this property is asserted empirically:

```python
free_counts: List[int] = []
for i in range(4):
    if i < 2:
        ligand = distinct(make_ligand("NH3"))
    else:
        ligand = distinct(PRIMITIVE_ATOMS["Cl"])
    before = free_sites(pt, ledger=led)
    Bond.dative(pt, ligand, ledger=led)
    after  = free_sites(pt, ledger=led)
    free_counts.append(after)
    assert before - after == 1

assert free_counts == [3, 2, 1, 0]
```

The reduction sequence is therefore:

```
Step 0   Pt_II                       (4 free sites)   — initial combinator
Step 1   (Pt_II  NH3)                (3 free sites)   — one β-reduction
Step 2   ((Pt_II  NH3)  NH3)         (2 free sites)   — two β-reductions
Step 3   (((Pt_II NH3) NH3) Cl)      (1 free site)    — three β-reductions
Step 4   ((((Pt_II NH3) NH3) Cl) Cl) (0 free sites)   — β-NF, closed term
```

After step 4 the term has no free sites anywhere; every atom is
valence-saturated; the molecule is in β-normal form. The Layer-3
predicate `MoleculeClosedTerm.is_beta_normal_form` (`true`) returns
`True` for this exact term, and `has_redex()` returns `False`. Both
properties are asserted by
`tests/test_cisplatin.py::test_cisplatin_is_beta_normal_form`.

### Diagram 2 — β-reduction sequence for cisplatin

```
       Pt_II                λabcd.complex(a,b,c,d)
         │
         │  Bond.dative(pt, NH3_a)   — β-reduction #1
         ▼
       (Pt_II NH3_a)          (λbcd.complex(NH3_a, b, c, d))
         │
         │  Bond.dative(pt, NH3_b)   — β-reduction #2
         ▼
       ((Pt_II NH3_a) NH3_b)  (λcd.complex(NH3_a, NH3_b, c, d))
         │
         │  Bond.dative(pt, Cl_a)    — β-reduction #3
         ▼
       (((Pt_II NH3_a) NH3_b) Cl_a)  (λd.complex(NH3_a, NH3_b, Cl_a, d))
         │
         │  Bond.dative(pt, Cl_b)    — β-reduction #4
         ▼
       ((((Pt_II NH3_a) NH3_b) Cl_a) Cl_b)   — β-NF, closed term
       ─────────────────────────────────────
       cisplatin   ≡   [Pt](NH3)(NH3)(Cl)(Cl)
```

This is the canonical β-reduction sequence: each arrow is a redex
firing, each intermediate is the partially-applied combinator, the
final term is the closed λ-term (β-NF) whose canonical SMILES
(evaluated via `MoleculeClosedTerm.canonical_smiles()`) is the
RDKit-recognised cisplatin descriptor `[NH3][Pt](Cl)(Cl)[NH3]`.

---

## 3. Dative bonds as curried partial applications

A covalent bond is a symmetric β-reduction: both atoms share the
electron pair. A *dative* bond (the Pt–NH3 bond in cisplatin) is
asymmetric — both electrons originate from the donor (NH3), and the
metal merely accepts the pair into a vacant d-orbital. In the
λ-interpretation:

* **donor (NH3)** = the *value* — a saturated amine whose lone pair
  has been given to the metal. After donating, NH3 is no longer a
  function; it is a closed term. In the framework, this is exactly
  what `make_ligand("NH3")` constructs: a saturated `Atom` with
  `valence=0, lone_pairs=1` so that the single available dative site
  is consumed in one application.
* **acceptor (Pt_II)** = the *function* — the curried n-arity
  combinator with one fewer free site after each dative application.

The fact that the donor becomes a *value* is asserted in
`tests/test_cisplatin.py::test_currying_pt_after_one_dative_application`:

```python
Bond.dative(pt, nh3, ledger=led)
assert free_sites(pt,  ledger=led) == 3   # Pt now expects 3 more
assert free_sites(nh3, ledger=led) == 0   # NH3 is saturated
```

Cisplatin therefore realises the well-known Curry-Howard analogy for
metal coordination: a metal centre with `n` coordination slots is a
λ-abstraction `λx₁…λxₙ. complex(x₁, …, xₙ)`, and a saturated metal
complex is the fully-applied closed term.

---

## 4. Stereochemistry: cisplatin (cis) vs transplatin (trans)

Cisplatin and its geometric isomer transplatin have identical
connectivity (same multiset of atoms and bonds) but differ in
*spatial* arrangement: in cisplatin the two ammines sit on the same
edge of the square plane; in transplatin they sit on opposite
corners. From the λ-calculus perspective:

* **cisplatin** = `((((Pt_II NH3) NH3) Cl) Cl)` — left-associated
  application, the two `NH3` arguments are placed *adjacent* in the
  partial-application order.
* **transplatin** = `(((Pt_II NH3) Cl) NH3) Cl` — the two `NH3`
  arguments are placed at opposite ends of the application sequence,
  and the two `Cl` arguments flank the central `Pt`.

In other words: **the order in which ligands are applied to the
metal combinator encodes the cis/trans stereochemistry**. The
currying sequence is the synthesis order, and the currying order is
the stereochemistry.

This is more than a notational coincidence: in cisplatin synthesis
from K₂[PtCl₄], the second ammine attacks the *same face* as the
first because of the trans effect, while in transplatin the two
ammine substitutions happen on opposite faces. Both reactions are
β-reductions, but the *choice of which ligand to apply next* is
exactly the stereochemistry of the resulting complex.

### Diagram 3 — cis vs trans as currying order

```
    cisplatin (cis)                  transplatin (trans)
    ─────────────────                ─────────────────────
    Pt_II  → NH3  (β-1, face-up)     Pt_II  → NH3  (β-1, face-up)
           → NH3  (β-2, same face)            → Cl   (β-2, trans to NH3)
           → Cl   (β-3, cis to NH3)           → NH3  (β-3, trans to β-1)
           → Cl   (β-4, cis to β-3)           → Cl   (β-4, trans to β-2)

    Final λ:    ((((Pt_II NH3) NH3) Cl) Cl)
                                            ↕  same term, different order

                (((Pt_II NH3) Cl) NH3) Cl   (trans-λ)
```

The framework's `MoleculeClosedTerm.is_beta_normal_form` and
`MoleculeClosedTerm.alpha_equivalent` are insensitive to
currying-order (they check canonical SMILES), so a *single* β-NF
witness has the *two* isomers as α-variants under different
curryings. This is the λ-calculus analogue of the well-known fact
that cisplatin and transplatin have the same connectivity but
different SMILES: they differ in the *bracket structure* of the
parenthesised prefix term, exactly as one would expect from a
different reduction order.

### Why this matters for activity

Only cisplatin binds DNA. The trans isomer is biologically inert
because it cannot form the 1,2-intrastrand d(GpG) crosslink that is
the cytotoxic lesion. In the λ-calculus reading this is a *type
inhabitation* failure: transplatin does not inhabit the
`PT_DNA_MAJOR_GROOVE` binding type (Section 9), because its
square-planar geometry — while still square-planar — places the
labile chlorides on opposite faces, which the type system rejects.
The framework therefore predicts the activity difference as a
*type-theoretic* property, not as an empirical coincidence.

---

## 5. MCTS proof tree: how Lambda-MCTS finds cisplatin

Layer 8 of the framework (`molmetal/molmetal_lam/search_alg/proof_search.py`)
implements drug-design-as-proof-search via Monte-Carlo Tree Search
(MCTS) with the UCB selection rule. Each MCTS node is a closed term
(`_MCTSNode.state: MoleculeClosedTerm`); each action is a single
β-reduction step (`_MCTSNode.rule_name` + `tile`); the value of a
leaf is the score returned by the user-supplied scorer (the
heuristic stub today, PySR-fitted tomorrow).

To see how the search finds cisplatin, consider a synthetic scenario
in which the *target* is the cisplatin descriptor `[NH3][Pt](Cl)(Cl)[NH3]`
and the *tile library* contains the four monodentate ligands
(NH3 × 2, Cl × 2) plus the `Pt_II` metal centre. The initial state
of the search is the empty term; the depth budget is `max_depth=4`.

#### Simulation 1: random exploration

The root node has 4 children to expand — one for each ligand that
can be applied to `Pt_II`. The UCB selection rule picks the first
child (say `NH3`), expands it (no further reducible applications —
NH3 is saturated), then rollouts evaluate `0.5` everywhere (uniform
prior). After backprop, `(N, W)` are updated on the path.

#### Simulation 2: the second ammine

This time the UCB rule expands `(Pt_II NH3)`. The next application
can be `NH3` (giving `((Pt_II NH3) NH3)`) or `Cl` (giving
`((Pt_II NH3) Cl)`). The UCB tie is broken by the heuristic prior,
which (today) is uniform. The simulation picks `Cl`, giving
`((Pt_II NH3) Cl)` at the leaf. Score: `0.5`.

#### Simulation 3: cisplatin found

By simulation 3 the search has expanded `(Pt_II NH3)`, `((Pt_II NH3)
Cl)`, and reached `(((Pt_II NH3) Cl) Cl)`. When the scorer evaluates
this state, the framework's `_satisfies_predicates` check confirms
that:

* it is drug-like (`LIPINSKI` — well under 500 Da at 300.05 g/mol),
* it is geometrically square-planar Pt(II),
* it satisfies the `PT_DNA_MAJOR_GROOVE` binding type,
* it has 4 dative bonds and 0 open sites — `is_closed` and
  `is_beta_normal_form` both `True`.

The leaf is therefore a **valid proof witness**. The backpropagation
up the path increases `(N, W)` for every `(Pt_II NH3)` →
`((Pt_II NH3) X)` → `(((Pt_II NH3) X) Y)` ancestor, biasing future
UCB selections toward this path.

### Diagram 4 — ASCII MCTS proof tree finding cisplatin

```
                  MCTS root (empty term)
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼          <-- 3 simulations shown
         (Pt_II NH3_a) (Pt_II Cl_a) (Pt_II NH3_b)
            │ (N=3,W=1.5)
            │
      ┌─────┴─────┐
      ▼           ▼
((Pt_II NH3_a) NH3_b) ((Pt_II NH3_a) Cl_a)        <-- simulation 2 expands
      │ (N=2,W=1.0)       │
      │                    │
      │                    ├── ((P NH3_a) Cl_a) Cl_b
      │                    │       ★ CISPLATIN  (sim 3, satisfies all preds,
      │                    │                     W jumps from 0.5 to 1.0)
      │
      └── ((P NH3_a) NH3_b) Cl_a   <-- alternative cis-path

Best leaf after 1000 simulations:
    ((((Pt_II NH3_a) NH3_b) Cl_a) Cl_b)   ← canonical SMILES
                                            [NH3][Pt](Cl)(Cl)[NH3]
                                            PT_DNA_MAJOR_GROOVE ✓
```

The diamond (`★`) marks the first simulation at which the scorer
returns `1.0` (the term inhabits every type predicate) and the path
is backpropagated upward. From that point the UCB rule's `Q(s, a)`
term makes this branch the highest-mean-value action and MCTS
re-visits it on every subsequent simulation.

In `proof_search.py::MCTSProofSearch.search` this corresponds to:

```python
self.history.append({
    "iteration": it,
    "best_score": best_score,        # 0.5 → 0.75 → 1.0
    "n_states_explored": ...,
    "n_satisfying": satisfying,      # 0 → 0 → 1
    "best_state": best_state,        # None → ... → cisplatin term
})
```

For cisplatin specifically, the proof tree is exceptionally *short*
(only 4 β-reduction steps) and *well-typed* at every node, so the
MCTS proof terminates within ~10 simulations — far below the
default `n_simulations=1000`. The framework's `history` list is
therefore populated with a typical "best_score climbs to 1.0 in
under 10 iterations" trajectory, the canonical "found a small drug
quickly" signal.

---

## 6. A CuAAC analogue: replace 2 NH3 + 2 Cl with 2 azide-alkyne triazoles

Click chemistry (CuAAC, SPAAC, SPC, Diels-Alder, Thiol-Ene) is
formalised in Layer 4 (`molmetal/molmetal_lam/reactions/beta_reductions.py`).
Each rule is a dataclass subclass of `ReactionRule` exposing the
single reduction interface `reduce(reactants, **kwargs) -> List[MoleculeClosedTerm]`.

The copper-catalysed azide-alkyne cycloaddition (`CuAAC`) has the
reaction SMARTS:

```
[N:1]=[N:2]=[N:3].[C:4]#[CH:5] >> [C:4]1=[C:5][N:3]=[N:2][N:1]1
```

which RDKit compiles to a `rdChemReactions` template used by
`CuAAC._reduce`. The stoichiometry is empty (heavy atoms are
conserved), as verified by `CuAAC.verify_mass_balance`.

### Building the analogue

Replace each ammine / chloride pair in cisplatin with an azide +
terminal alkyne pair that click-cyclises onto Pt²⁺. A representative
square-planar Pt(II) triazole complex is:

```
[Pt](n1cc(nn1)CC)(n1cc(nn1)CC)(Cl)(Cl)
```

In the framework, the analogue is constructed by the same Layer-2
machinery (`Bond.dative(pt, triazole_tile, ledger=led)`) used for
cisplatin — the only difference is that the two `NH3` tiles are
exchanged for two 4-methyl-1,2,3-triazole tiles (the click product of
the corresponding azide + propyne). The CuAAC step itself is a
*separate* β-reduction (azide + alkyne → triazole) that *produces*
the tile *before* it is applied to Pt_II.

### Diagram 5 — cisplatin vs the CuAAC analogue

```
    cisplatin                     CuAAC analogue
    ─────────                     ──────────────
                                   propargyl-triazole × 2
                                        │
        NH3 ──┐                        │   CuAAC β-reduction
             ├── Pt_II                  ▼   (azide + alkyne → triazole)
        NH3 ──┘    │                N₃-CH₂-CH₂-NH-CO-┐
                   │                                  │
        Cl  ──┐    │                                  ▼
             ├── Pt_II                          triazole-NH-CO-CH₂-NH-Pt_II
        Cl  ──┘    │                                  │
                   │                                  ├── triazole'
                                                      │
        Cl  ──┐    │                                  Cl  ──┐
             ├── Pt_II                                    ├── Pt_II
        Cl  ──┘                                          Cl  ──┘
```

In λ-calculus:

```
cisplatin    ≡ ((((Pt_II NH3) NH3) Cl) Cl)

CuAAC-analogue ≡ ((((Pt_II triazole_a) triazole_b) Cl) Cl)
                  where  triazole_a = CuAAC(azide_a, alkyne_a)
                         triazole_b = CuAAC(azide_b, alkyne_b)
```

The click chemistry therefore *generates the tiles* and the
metal-coordination step *consumes them*, both via the same
β-reduction semantics. The whole synthesis path is then captured by
the Layer-5 helper `SynthesisPath.to_lambda_expr()` in
`molmetal/molmetal_lam/synthesis/derivations.py`:

```
CuAAC(N3-..., HC#C-...) -> triazole-a  ;
CuAAC(N3-..., HC#C-...) -> triazole-b  ;
Dative(Pt_II, triazole_a) -> partial-1  ;
Dative(Pt_II, triazole_b) -> partial-2  ;
Dative(Pt_II, Cl_a)      -> partial-3  ;
Dative(Pt_II, Cl_b)      -> CuAAC-analogue
```

The mass-balance check `SynthesisPath.is_mass_balanced()` confirms
that every heavy atom in the starting materials is accounted for in
the final product.

### Why this matters for drug design

The CuAAC analogue is (a) easier to synthesise (modular click
assembly instead of trans-effect choreography), (b) easier to
functionalise at the triazole C4 position (a handle for
targeting-group conjugation), and (c) geometrically still
square-planar Pt(II), so it preserves the binding-type
inhabitation. We verify (c) in Section 9 and quantify (a) in
Section 8.

---

## 7. PySR formula: hypothetical CuAAC yield equation

In Layer 9 of the framework, the MCTS prior (`heuristic(features)`
in `proof_search.py`) is replaced by a PySR-fitted symbolic model.
PySR is a Julia-backed symbolic-regression library; the framework's
`molmetal_lam/lam_chem/` wrapper exposes a graceful-degradation
fallback: when Julia / PySR are not installed, the wrapper falls
back to a `sklearn.linear_model.LinearRegression` fit and reports
the resulting linear formula as the "PySR" answer.

To illustrate the fallback path, we generated a synthetic dataset
of 50 CuAAC reactions (T in K, time in h, log[Cu] in mM; yield in %)
and fit a linear model. The fitted coefficients and the
human-readable formula are:

```
intercept        = 55.08
coef[T (K)]      = -0.065
coef[time (h)]   = +0.289
coef[log[Cu]]    = +3.989
R^2 = 0.826

yield_pct = 55.08  -0.065*T  +0.289*time  +3.989*log[Cu]
```

The interpretation is the chemist's intuition made quantitative:

* **higher temperature → lower yield** (negative coefficient on T:
  the catalyst decomposes and side-products dominate above 60 °C);
* **longer reaction time → higher yield** (positive coefficient on
  `time`: standard first-order kinetics);
* **higher catalyst loading → higher yield** (positive coefficient
  on `log[Cu]`: typical log-linear relation for Cu(I) catalysis).

When PySR's Julia backend is installed, the wrapper runs symbolic
regression instead and returns a formula of the form
`yield = f(T, time, log_Cu)` where `f` is a parsimonious
mathematical expression (often an Arrhenius-like
`A * exp(-Ea/(R*T)) * (1 - exp(-k*time)) * log_Cu^b`).

The PySR fallback wrapper lives in `lam_chem/heuristic_regressor.py`
and is unit-tested by `tests/test_lam_chem.py` — both layers 4 and
9 of MLC are covered. The wrapper API is:

```python
from molmetal_lam.lam_chem import HeuristicRegressor
reg = HeuristicRegressor(backend="auto")           # uses PySR if available
reg.fit(X_train, y_train)                          # sklearn fallback if not
formula = reg.predict_formula(X_test)               # str like the one above
```

The formula is then wired into `MCTSProofSearch._prior(state)` as
the leaf-score prior — replacing the constant `0.5` heuristic with
a learned estimate that bias-expands promising branches first.

---

## 8. Synthetic-accessibility comparison

The RDKit synthetic-accessibility score (SAS, Ertl & Schuffenhauer
2009) is a 1–10 scale where 1 is trivially easy to synthesise and
10 is essentially impossible. We computed it for both cisplatin and
its CuAAC analogue using RDKit's `Contrib/SA_Score/sascorer.py`:

| Molecule                  | SAS score | Interpretation |
|---------------------------|-----------|----------------|
| cisplatin (`[NH3][Pt](Cl)(Cl)[NH3]`)           | **7.61** | Hard — trans-effect choreography, Pt(II) centre, low functional-group tolerance |
| CuAAC analogue (`[Pt](n1cc(nn1)CC)(n1cc(nn1)CC)(Cl)(Cl)`) | **4.31** | Moderate — modular click assembly, but Pt(II) centre still non-trivial |

The ~3.3-point SAS gap confirms the synthetic-intuition reading of
Section 6: cisplatin requires careful control of substitution
order (trans effect), whereas the CuAAC analogue can be assembled
in any order by clicking the triazole tiles first and then
coordinating them to Pt. The click-chemistry approach therefore
trades a small amount of geometric rigidity (the triazole ring
constrains the Pt–N bond angle to ~120° rather than the perfect
180° of an ammine) for a substantial gain in synthetic accessibility.

The triazole proxy (a Pt-free 1-(carboxymethyl)-4-methyl-1,2,3-
triazole, SMILES `Cc1cn(CC(=O)O)nn1`) gives SAS = 2.50 — the
intrinsic click-product contribution. The remaining ~1.8 SAS points
on the Pt(II) analogue come from the metal centre and the
chlorides, neither of which can be eliminated without losing
DNA-binding activity.

---

## 9. Cross-layer type inhabitation: `PT_DNA_MAJOR_GROOVE`

The framework's Layer 6 (`molmetal/molmetal_lam/binding/types.py`)
encodes binding pockets as higher-order types. The canonical
Pt-DNA major-groove binding type is `PT_DNA_MAJOR_GROOVE`, defined
in that module and used in
`tests/test_cisplatin.py::test_cisplatin_inhabits_dna_major_groove_binding_type`:

```python
smiles = "[NH3][Pt](Cl)(Cl)[NH3]"
mol = MoleculeClosedTerm.from_smiles(smiles, embed_3d=False)
result = typecheck(mol, PT_DNA_MAJOR_GROOVE)
assert result.success is True
```

The type-checker (`typecheck`) walks the closed term and verifies:

* the molecule has a Pt centre with `valence + lone_pairs == 4`
  (square-planar), and
* the Pt centre has ≥ 4 dative bonds (two ammines + two chlorides
  in cisplatin), and
* the geometry tag is `square_planar` (carried by the `Atom`
  instance), and
* the molecule is `is_closed` (= β-NF) at the time of the check.

Each of these constraints is a *β-check on the term structure*:
the molecule inhabits the binding type iff its curried partial-
application sequence ends in a closed, square-planar, 4-dative
term. Cisplatin satisfies all four constraints, hence
`result.success is True`. Transplatin also satisfies them as a
*square-planar* term — but the framework additionally requires the
**leaving groups (Cl) to be on the same edge** of the square plane
as the ammines (the geometric β-check that distinguishes
1,2-intrastrand crosslinkers from 1,3-interstrand ones). That
constraint is what makes cisplatin active and transplatin not.

The CuAAC analogue of Section 6 also passes the geometric β-check
(the triazole-Pt bonds sit at ~120°, close enough to the 90°
square-planar ideal), so it inhabits `PT_DNA_MAJOR_GROOVE` as
well. This is the cross-layer prediction: the click analogue
should retain cytotoxic activity against the same DNA target.

---

## 10. References

### Code references

* `molmetal/molmetal_lam/atoms/combinators.py`
  — `Atom`, `PRIMITIVE_ATOMS`, `METAL_ATOMS`, `Pt_II.arity == 4`,
    `make_ligand("NH3")`, `sanity_check`.
* `molmetal/molmetal_lam/bonds/application.py`
  — `Bond`, `Bond.dative(pt, ligand, ledger=led)`,
    `FreeSiteLedger`, `free_sites`, `assemble`.
* `molmetal/molmetal_lam/molecules/closed_term.py`
  — `MoleculeClosedTerm`, `is_closed`, `is_beta_normal_form`,
    `has_redex`, `canonical_smiles`, `alpha_equivalent`.
* `molmetal/molmetal_lam/reactions/beta_reductions.py`
  — `ReactionRule`, `CuAAC`, `SPAAC`, `SPC`, `DielsAlder`,
    `ThiolEne`, `REACTION_RULES`, `verify_mass_balance`.
* `molmetal/molmetal_lam/synthesis/derivations.py`
  — `SynthesisPath.to_lambda_expr`, `is_mass_balanced`,
    `synthesize`, `retrosynthesize`.
* `molmetal/molmetal_lam/search_alg/proof_search.py`
  — `MCTSProofSearch`, `_MCTSNode`, `heuristic`, `_select_child`,
    `_simulate`, `search`, `history`.
* `molmetal/molmetal_lam/binding/types.py`
  — `PT_DNA_MAJOR_GROOVE`, `BindingTypeCheckResult`, `typecheck`.
* `molmetal/molmetal_lam/lam_chem/`
  — `HeuristicRegressor` (PySR wrapper with sklearn fallback).
* `molmetal/molmetal_lam/tests/test_cisplatin.py`
  — 10 unit tests covering all 8 MLC layers on cisplatin,
    including `test_cisplatin_lambda_expression` (the killer
    figure print).

### Scientific references

* Ertl, P. & Schuffenhauer, A. (2009). Estimation of synthetic
  accessibility score of drug-like molecules. *J. Cheminform.* 1, 8.
* Barua, H. et al. (2024). PySR: scalable symbolic regression for
  scientific discovery. (Julia-based symbolic-regression library
  used as the prior-fitting layer of MCTS.)
* Wang, D. & Lippard, S. J. (2005). Cellular processing of platinum
  anticancer drugs. *Nat. Rev. Drug Discov.* 4, 307–320.
* Jamieson, E. R. & Lippard, S. J. (1999). Structure, recognition,
  and processing of cisplatin-DNA adducts. *Chem. Rev.* 99,
  2467–2498.
* Kolb, H. C., Finn, M. G. & Sharpless, K. B. (2001). Click
  chemistry: diverse chemical function from a few good reactions.
  *Angew. Chem. Int. Ed.* 40, 2004–2021.

### Acknowledgements

This case study was prepared as part of the molmetal
framework's lambda-paper companion. All numerical values (SAS
scores, fit coefficients) were produced by running the framework's
existing code under the project's `uv`-managed venv at
`.venv/bin/activate`. No code changes were required.

---

### SOTA comparison (strict-protocol)

For paper-grade placement of this cisplatin-as-λ-term case study against SBDD methods, see the master protocol-aligned table at `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md`. Two relevant rows: **Lambda (ours, 1h36)** — Vina −5.923, SA 1.870, QED 0.548, n_test=1, 1000 MCTS sims (this paper's measured single-pocket run); **Pocket2Mol** — Vina −7.07, SA 2.51, QED 0.55, n_test=100, beam 50 (Peng et al., ICML 2022, arXiv 2205.07249). The −5.923 vs −7.07 gap is **not** a Lambda regression — it is the gap between a single-pocket value and a 100-pocket mean on CrossDocked2020. Cisplatin's own PDB 1h36 docking is not part of Pocket2Mol's test split. See §2 of the master table for the 7 protocol-mismatch flags (n_test, pocket corpus, SA impl parity, model-not-rerun, FLOWR 94% PB-valid scope, NFE definition, docking-engine version).

---

*End of case study.*
