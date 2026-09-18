"""molmetal_lam — Molecular Lambda Calculus (MLC) framework.

============================================================
Thesis
============================================================
"De novo drug design is constructive proof search in the Molecular
Lambda Calculus. Every generated molecule is a witness (proof term)
for the proposition 'this molecule binds target T with acceptable
ADMET'."

Chemistry IS lambda calculus — every chemical concept has a direct
lambda-calculus counterpart:

    chemical concept       | lambda-calculus counterpart
    -----------------------|----------------------------------------
    atom                   | primitive combinator (S, K, I, Pt_II, ...)
    covalent bond          | beta-reduction (function application)
    dative (coord) bond    | curried partial application
    aromaticity            | eta-conversion class
    stereochemistry        | distinct beta-normal forms
    molecule               | closed lambda-term in beta-NF
    isomer                 | alpha-eqv but non-beta-eqv terms
    reaction               | one beta-reduction step
    forward synthesis      | beta-reduction sequence
    retrosynthesis         | beta-EXPANSION
    binding (protein+lig)  | beta-reduction + type check
    ADMET                  | type predicate
    drug activity          | constructive proof
    equivariance (SE(3))   | alpha-conversion invariance
    GNN expressivity       | eta-equivalence class

============================================================
Layered architecture (11 layers, see molecular_lambda_calculus.md)
============================================================

Layer 1  — atoms/       Atoms-as-combinators (SKI + metal arities)
Layer 2  — bonds/       Bonds-as-application (covalent, dative, aromatic)
Layer 3  — molecules/   Closed lambda-term in beta-NF
Layer 4  — reactions/   Reactions-as-beta-reduction (SN2, CuAAC, Diels-Alder)
Layer 5  — synthesis/   Forward (reduction) & retro (expansion) synthesis
Layer 6  — binding/     Protein + ligand = beta-reduction + type check
Layer 7  — types/       ADMET predicates (Lipinski, Veber, QED, ...)
Layer 8  — search_alg/  MCTS + PySR heuristic for proof search
Layer 9  — synthesis/   Tile library (click-chemistry building blocks)
Layer 10 — types/       Type inference via EGNN (binding-type inhabitation)
Layer 11 — pipeline/    End-to-end drug-design loop (built on the above)

============================================================
Curry-Howard isomorphism
============================================================
    type  = proposition (e.g. "Lipinski", "Binds MMP2")
    term  = proof (= a molecule)
    inhabitation = synthesizable drug candidate

A drug candidate is a "well-typed" molecule: it inhabits every ADMET
type predicate AND inhabits the binding type of its target.

============================================================
References (cloned into molmetal/references/)
============================================================
- PySR      Cranmer 2023 — Lambda-style symbolic regression (Julia+Py)
- REINVENT4 MolecularAI — SBDD sandbox w/ built-in scoring

============================================================
Status
============================================================
This is TASK 0 (SETUP): package skeleton only. See SMOKE_TEST_PLAN.md
for the eventual end-to-end smoke test that proves the framework.
"""

__version__ = "0.0.1-setup"
__all__ = ["__version__"]
