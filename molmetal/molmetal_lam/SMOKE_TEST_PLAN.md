# Smoke Test Plan — Molecular Lambda Calculus End-to-End

## Purpose

A single pytest that proves the MLC framework is wired correctly
*before* investing in any sophisticated feature work. The test must
exercise **every one of the 11 layers** of `molecular_lambda_calculus.md`
on a single canonical example: **cisplatin**.

Cisplatin is the minimal interesting drug in this framework because:
1. It contains a metal (Pt_II) — exercises arity-as-valence encoding.
2. It has 2 dative (coordination) bonds — exercises curried partial application.
3. It has 4 distinct ligands (2 NH3 + 2 Cl−) — exercises β-reduction.
4. The closed term has a non-trivial structure (Pt with 4 free sites, all bound).

---

## Canonical example: cisplatin as a closed λ-term

In MLC notation:

    Cisplatin = Pt_II(NH3)(NH3)(Cl)(Cl)
              = (((Pt_II NH3) NH3) Cl) Cl

Where:
- `Pt_II` is a 4-arity primitive combinator (square-planar, 2 covalent + 2 dative sites)
- `NH3` is a saturated 0-arity ligand (already in β-NF)
- `Cl` is a saturated 1-arity halide
- Each bond is a β-reduction step
- The fully applied term is in β-NF and closed (no free sites)

---

## Step-by-step test (`tests/test_cisplatin_smoke.py`)

### Step 0 — Layer 8 (package import)

```python
import molmetal_lam
assert molmetal_lam.__version__ == "0.0.1-setup"
```

Confirms: directory structure, `__init__.py`, package metadata.

### Step 1 — Layer 1: Atoms-as-combinators

```python
from molmetal_lam.atoms.combinators import Atom, PRIMITIVE_ATOMS

pt   = PRIMITIVE_ATOMS["Pt_II"]
nh3  = PRIMITIVE_ATOMS["N"]   # N has lone_pairs=1 so it can donate
cl   = PRIMITIVE_ATOMS["Cl"]

assert pt.arity == 4                  # 2 covalent + 2 lone-pair coordination sites
assert pt.geometry == "square_planar"
assert pt.is_metal is True
```

Confirms: every atom has an arity (= valence + lone pairs). Pt_II expects
exactly 4 ligands.

### Step 2 — Layer 2: Bonds-as-application

```python
from molmetal_lam.bonds.application import Bond

# First dative bond: Pt + NH3 → curried Pt with 3 free sites
pt_with_one = Bond.dative(pt, nh3)
assert pt_with_one.metal_free_sites == 3
assert pt_with_one.ligand_is_saturated(nh3) is True

# Second dative bond: Pt + NH3 → 2 free sites
pt_with_two = Bond.dative(pt_with_one, nh3)
assert pt_with_two.metal_free_sites == 2
```

Confirms: dative bonds are *partial* applications — Pt retains unapplied
sites. This is the key insight that coordination chemistry IS currying.

### Step 3 — Layer 3: Molecules-as-closed-terms

```python
from molmetal_lam.molecules.closed_term import Molecule

cisplatin = Molecule.from_atoms([pt, nh3, nh3, cl, cl])
# Apply 4 β-reductions (2 dative + 2 ionic) to close the term
for bond in cisplatin.bonds():
    cisplatin = cisplatin.reduce_once()   # one β-reduction per bond

assert cisplatin.is_closed is True
assert cisplatin.is_beta_normal_form is True
assert cisplatin.arity_remaining() == 0
```

Confirms: cisplatin is a closed λ-term in β-NF.

### Step 4 — Layer 4: Reactions-as-β-reduction

```python
from molmetal_lam.reactions.reduction import reduce_once

# Pick any single bond and reduce it
initial_bond_count = len(cisplatin.bonds)
reduced = reduce_once(cisplatin, redex_selector="leftmost")
assert len(reduced.bonds) == initial_bond_count     # bonds persist (history)
assert reduced.n_unsaturated_atoms < cisplatin.n_unsaturated_atoms
```

Confirms: a single β-reduction step corresponds to one chemical transformation.

### Step 5 — Layer 7: ADMET type predicates

```python
from molmetal_lam.types.predicates import LIPINSKI, QED_ACCEPTABLE, well_typed

# Cisplatin is small (MW ~300), so it satisfies Lipinski
assert LIPINSKI(cisplatin) is True
# It has decent drug-likeness
assert QED_ACCEPTABLE(cisplatin) is True
# It is "well-typed" under both predicates
assert well_typed(cisplatin, [LIPINSKI, QED_ACCEPTABLE]) is True
```

Confirms: ADMET properties are type predicates; satisfying them = well-typed.

### Step 6 — Layer 6: Binding type inhabitation

```python
from molmetal_lam.binding.types import BindingType
from molmetal_lam.binding.complex import dock

# DNA major groove has Pt-coordination + 2 H-bond acceptor sites
dna_major_groove = BindingType(
    name="DNA_Pt_crosslink",
    constraints=[
        HasAtom("Pt_II"),
        HasCoordinationNumber(4),
        IsSquarePlanar(),
    ],
)

result = dock(cisplatin, dna_major_groove)
assert result.success is True                  # cisplatin inhabits the type
assert result.affinity < -5.0                  # sub-uM by literature
```

Confirms: drug activity is *type inhabitation* — cisplatin is a witness for
the proposition "binds DNA via Pt-crosslink".

### Step 7 — Layer 5: Synthesis (forward = β-reduction sequence)

```python
from molmetal_lam.synthesis.forward import synthesize

path = synthesize(
    target=cisplatin,
    starting_materials=[K2PtCl4, NH3, NH3, Cl, Cl],   # 5 raw tiles
)
# Each reaction in the path is one β-reduction
assert all(r.is_beta_reduction() for r in path)
assert len(path) >= 2                                # Pt(NH3)2 formation + Cl exchange
```

Confirms: forward synthesis = a sequence of β-reductions from raw materials
to the target.

### Step 8 — Layer 5: Synthesis (retro = β-expansion)

```python
from molmetal_lam.synthesis.retro import retrosynthesize

expansions = retrosynthesize(cisplatin)
# Each expansion is one β-expansion → a precursor + the redex that formed the bond
assert len(expansions) == 4                          # 4 bonds = 4 expansions
assert all(p.has_redex_pattern() for p in expansions)
```

Confirms: retrosynthesis = β-expansion (the dual operation).

### Step 9 — Layer 9: Equivariance (α-conversion invariance)

```python
from molmetal_lam.molecules.closed_term import Molecule

# Re-label the atoms — same molecule, different atom IDs
relabeled = cisplatin.alpha_rename({0: "Pt_A", 1: "N_1", ...})
assert cisplatin.alpha_equivalent(relabeled) is True
```

Confirms: α-conversion (renaming) doesn't change the molecule — this is
exactly SE(3) coordinate-frame invariance.

### Step 10 — Curry-Howard: proof term IS synthesis path

```python
# The λ-term's β-reduction history IS the synthesis path
synthesis_steps = cisplatin.beta_reduction_history()
synthesis_steps_str = " → ".join(str(s) for s in synthesis_steps)
assert "Pt_II(NH3)" in synthesis_steps_str or "Pt_II NH3" in synthesis_steps_str
```

Confirms: the synthesis path is **literally the β-reduction trace** of the
λ-term. No separate "explanation" needed.

---

## Pass criteria

The smoke test passes iff ALL 10 assertions succeed. Failure modes:

| Failure | Likely root cause |
|---|---|
| Step 0 import fails | package not in `[tool.setuptools].packages` |
| Step 1 arity wrong | `Atom.arity` not = `valence + lone_pairs` |
| Step 2 metal_free_sites wrong | `Bond.dative` not currying |
| Step 3 not is_beta_normal_form | Molecule has redexes left |
| Step 4 reduce_once decreases atoms | reaction not a proper β-reduction |
| Step 5 LIPINSKI fails | MW / logp calculation wrong |
| Step 6 dock fails | binding type check incomplete |
| Step 7 synthesize empty | forward search broken |
| Step 8 retrosynth wrong count | β-expansion semantics broken |
| Step 9 alpha-eqv fails | α-conversion not properly implemented |
| Step 10 trace missing | reduction history not recorded |

---

## Status

- 2026-09-11 — TASK 0 SETUP complete: package skeleton, references cloned.
- Smoke test planned but NOT YET implemented (depends on Tasks 1..N layers).
- Will be implemented as `tests/test_cisplatin_smoke.py` once Layers 1–10
  are filled in (TASK 1 onwards).

