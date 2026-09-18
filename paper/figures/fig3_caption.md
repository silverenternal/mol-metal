# Figure 3 — The 5 click-chemistry reactions

**Caption.** The canonical click-chemistry rule set consumed by the
Molecular Lambda Calculus (MLC) reduction engine.  Each panel shows the
reaction name, the SMARTS template (verbatim from the source code), the
two reactant SMILES, the canonical product SMILES, and the catalyst
label rendered on the curved arrow.  Colour key: **CuAAC** = orange,
**SPAAC** = amber (strained variant of CuAAC, no Cu catalyst),
**Thiol-ene** = green, **Suzuki-Miyaura** = blue, **Amide coupling** =
red.

## Source provenance

The five rules are the *strict subset* of click reactions registered in
`molmetal/molmetal_lam/lam_chem/rules.py` (re-exporting the
`ReactionRule` singletons from
`molmetal/molmetal_lam/reactions/beta_reductions.py`).  Per-rule
`pattern_smiles` and `requires_catalyst` annotations:

| Rule             | SMARTS template (verbatim)                                                                              | Catalyst                 |
|------------------|----------------------------------------------------------------------------------------------------------|--------------------------|
| CuAAC            | `[N:1]=[N:2]=[N:3].[C:4]#[CH:5]>>[C:4]1=[C:5][N:3]=[N:2][N:1]1`                                          | `Cu(I)`                  |
| SPAAC            | `[N:1]=[N:2]=[N:3].[C:4]#[C:5]>>[C:4]1=[C:5][N:3]=[N:2][N:1]1`                                          | none (strain-promoted)   |
| ThiolEne         | `None` (functional-group rewrite, not SMARTS)                                                            | `hν / radical initiator` |
| Suzuki-Miyaura   | `[#6:1][B]([O])[O].[#6:3][F,Cl,Br,I]>>[#6:1][#6:3]`                                                      | `Pd(0) + base`           |
| AmideCoupling    | `[C:1](=[O:2])[OH].[NH2:4]>>[C:1](=[O:2])[NH:4]`                                                         | coupling reagent (EDC/HOBt, HATU, ...) |

Cite: `molmetal/molmetal_lam/lam_chem/rules.py` (the public-facing
registry) and `molmetal/molmetal_lam/reactions/beta_reductions.py` (the
concrete `ReactionRule` subclasses).

## Honest-framing note

- **MEASURED:** the *rule definitions* (SMARTS templates, catalyst
  labels, stoichiometry annotations) are imported verbatim from the
  source code at figure-render time.
- **PROJECTED (illustrative):** the example reactant/product SMILES in
  each panel are *literature-typical*, hand-picked, RDKit-parseable
  examples — **not** synthesised by the rule engine.  The SPAAC
  example product in particular is drawn in its chemically expected
  form (triazole attached to the cyclooctane ring via the original
  alkyne carbon), because the raw SMARTS product currently trips
  RDKit's kekulé sanitiser (see `lam_chem/rules.py` issue list,
  pending task T-future-3 in TODO/pending).

## Regeneration

```bash
uv run paper/figures/fig3_click_reactions.py
```

Outputs: `paper/figures/fig3_click_reactions.png` (600 dpi) +
`paper/figures/fig3_click_reactions.svg` (scalable).
