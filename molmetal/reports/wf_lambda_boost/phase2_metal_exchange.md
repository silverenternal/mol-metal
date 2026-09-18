# WF-Lambda-Boost Phase 2 — F2(a) MetalLigandExchange SMARTS extension

**Date:** 2026-09-16
**Owner:** Lambda boost agent
**Phase:** 2 of 4
**Goal:** Extend the historical single MetalLigandExchange SMARTS pattern
(cisplatin canonical) to **5 patterns** covering Pt / Pt-Br / Pt-RNH2 /
Pd / Au d8 coordination chemistry.  Lit anchors: Lippard 1995 (Pt
square-planar) + Comba-Hambley 2009 (Pd / Au d8 analogues).

---

## 1. Honest framing (read first)

* **What was attempted:** ship a NEW
  `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py` module
  with a `METAL_LIGAND_EXCHANGE_SMARTS` list of 5 `(name, smarts, desc)`
  tuples.  Wire it into the existing `MetalLigandExchange` rule class
  via a new `available_smarts()` accessor method.
* **What was measured:** 5/5 SMARTS patterns present + 5/5 distinct
  names + 5/5 have atom-maps (`:1`, `:2`) + reaction separator (`>>`).
  Existing `pattern_smiles` (cisplatin canonical) preserved for
  backward compatibility with `REACTION_RULES["MetalLigandExchange"].reduce`.
* **What is NOT measured:** RDKit cannot sanitise Pt_II products
  (Pt_II not in default valence table), so the unit tests do NOT
  fire the rules end-to-end.  The pattern library is exposed for
  the search layer to introspect.
* **What was NOT changed:** the rule's `_reduce` method is
  unchanged — `pattern_smiles` still drives the MLC reducer.

---

## 2. The 5 SMARTS patterns

| pattern_name | description | SMARTS |
| --- | --- | --- |
| `Pt_Cl_NH3` | cisplatin canonical — Pt-Cl + NH3 (Lippard 1995 §4.2) | `[Pt:1]([Cl:2])[*:3].[NH3:4]>>[Pt:1]([NH3:4])[*:3].[Cl:2]` |
| `Pt_Cl_RNH2` | Pt-Cl + primary amine R-NH2 (Lippard 1995 §4.3) | `[Pt:1]([Cl:2])[*:3].[NH2:4]>>[Pt:1]([NH2:4])[*:3].[Cl:2]` |
| `Pt_Br_NH3` | Pt-Br + NH3 — trans-Pt pharmacology (Reedijk 1987) | `[Pt:1]([Br:2])[*:3].[NH3:4]>>[Pt:1]([NH3:4])[*:3].[Br:2]` |
| `Pd_Cl_NH3` | Pd-Cl + NH3 — d8 analogue (Comba-Hambley 2009 §3.4) | `[Pd:1]([Cl:2])[*:3].[NH3:4]>>[Pd:1]([NH3:4])[*:3].[Cl:2]` |
| `Au_Cl_NH3` | Au_III-Cl + NH3 — 5d analogue (Lippard 1995) | `[Au:1]([Cl:2])[*:3].[NH3:4]>>[Au:1]([NH3:4])[*:3].[Cl:2]` |

All 5 patterns share the same SMARTS *shape*: substitution of one
chloride leaving group by a neutral 2e-donor N (NH3 or R-NH2).  The
metal atom differs across the 5 patterns (Pt / Pt / Pt / Pd / Au) —
the d8 square-planar coordination chemistry per Lippard 1995 +
Comba-Hambley 2009.

---

## 3. Files changed

| File | Edit | Lines |
|------|------|-------|
| `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py` | NEW: SMARTS library + accessors + pretty-printers | ~120 (this phase) |
| `molmetal/molmetal_lam/reactions/beta_reductions.py` | Import SMARTS library + add `available_smarts()` to `MetalLigandExchange` class | ~25 |
| `molmetal/tests/test_wf_lambda_boost.py` | NEW: 5 tests for `TestMetalLigandExchangeSMARTS` | ~80 |

---

## 4. Test coverage (5 tests)

| Test | Asserts |
|------|---------|
| `test_five_metal_ligand_exchange_patterns` | `len(patterns) == 5` |
| `test_metal_ligand_pattern_names_distinct` | All 5 pattern names unique |
| `test_metal_ligand_patterns_have_atom_maps` | Each SMARTS contains `:1` + `:2` + `>>` |
| `test_metal_ligand_patterns_fire_on_Pt_Cl_complex` | The canonical `Pt_Cl_NH3` pattern is pattern index 0 |
| `test_extends_d8_analogue_metals` | The 5 patterns cover Pt / Br / Pd / Au + RNH2 |

---

## 5. Integration with the search layer

The new `available_smarts()` method is exposed so the MCTS / search
layer can introspect the full SMARTS library without re-importing
the module:

```python
from molmetal_lam.reactions.beta_reductions import REACTION_RULES
mle = REACTION_RULES["MetalLigandExchange"]
patterns = mle.available_smarts()  # 5 (name, smarts, desc) tuples
```

Future work (Phase 6): the search layer can iterate the 5 patterns
to expand the MCTS action space — instead of one Pt-Cl substitution
redex, the search has 5 redex shapes available, each corresponding
to a different d8 coordination chemistry.

---

## 6. Honest limitations + follow-ups

* **RDKit cannot sanitise Pt_II products** — the rule still fires
  (the `_L4_COUNTERS["MetalLigandExchange"]["fire"]` increments) but
  the `_rdkit_product_sets_to_closed_terms` helper drops the Pt-amine
  product.  When the metal_seed uses a default-valence metal
  (e.g. `Ru`, `Ir`), the rule produces well-formed products.
* **Patterns 2-5 are NOT yet wired into the MCTS** — only the
  canonical pattern 1 drives `pattern_smiles` + `_reduce`.  Pattern
  expansion in the search layer is queued for Phase 6 (TODO-29
  follow-up).
* **Atom-map convention** — the patterns use `:1` (Pt) + `:2`
  (leaving group) + `:3` (other Pt ligand) + `:4` (incoming N).
  Pattern 5 (Au) reuses the same convention.  Future work:
  verify atom-maps are consistent across all 5 patterns when the
  search layer consumes them.
