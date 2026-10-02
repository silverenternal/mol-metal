# Wf-290F4 — Fix `test_metal_ligand_exchange_matrix`

## TL;DR
Added the missing `METAL_LIGAND_EXCHANGE_MATRIX` constant (12 metals × 7
ligands = 84 patterns) plus three accessors and one registration helper to
`molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py`.  All 659
collected test cases now pass (collection went from 1 collection-error to
659 passed in 0.85s).

## Error before
Collection aborted with `ImportError`:

```
molmetal/tests/test_metal_ligand_exchange_matrix.py:74: in <module>
    from molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
E   ImportError: cannot import name 'METAL_LIGAND_EXCHANGE_MATRIX' from
    'molmetal_lam.lam_chem.pt_metal_ligand_exchange'
    (molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py)
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!
no tests collected, 1 error in 0.65s
```

The test imports five public names from the module:
* `METAL_LIGAND_EXCHANGE_MATRIX`
* `get_metal_ligand_exchange_matrix_patterns`
* `get_metal_ligand_matrix_ligands`
* `get_metal_ligand_matrix_metals`
* `register_metal_ligand_exchange_rules`

None of these were defined — only the F2(a) curated
`METAL_LIGAND_EXCHANGE_SMARTS` (5 patterns) and `AQUA_EXCHANGE_SMARTS`
(2 patterns) were shipped.

## Diagnosis
The test file (`molmetal/tests/test_metal_ligand_exchange_matrix.py`)
implements the L-R17-B2 TODO-31 §2 (2026-09-25) requirement of a 12-metal
× 7-ligand = 84-pattern exhaustive matrix covering the d-block
substitution chemistry the MCTS can fire on metallodrug scaffolds.  The
test file ships 22 test functions (84 parametrized cells × multiple axes)
that pin:
1. Matrix cardinality = 84 + unique names
2. SMARTS parses via RDKit `ReactionFromSmarts`
3. Per-metal coverage (each of 12 metals has 7 patterns)
4. Per-ligand coverage (each of 7 ligands has 12 patterns)
5. `register_metal_ligand_exchange_rules()` returns >= 84 entries
6. The F2(a) 5-pattern library + 2-pattern AquaExchange library are
   unchanged
7. `Pt_II_Cl_NH3` matches F2(a) `Pt_Cl_NH3` shape-for-shape
8. `Pt_II_Cl_H2O` matches F2(a) `Pt_Cl_H2O_first` shape-for-shape
9. Canonical donor SMILES inventory + per-cell `RunReactants` smoke

But the L-R17-B2 matrix constants were never wired into the
implementation module — only the test was written.  This is a textbook
"test-first but no implementation" gap.

## Fix applied
Edited `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py` to
add the missing L-R17-B2 surface:

1. **`MATRIX_METALS`** — tuple of 12 `(symbol, ox_state, geometry)` tuples:
   ```
   (Pt, II, square_planar), (Pt, IV, octahedral),
   (Ru, II, octahedral),  (Ru, III, octahedral),
   (Ir, III, octahedral), (Ir, IV, octahedral),
   (Os, II, octahedral),  (Os, III, octahedral),
   (Rh, I, linear),       (Rh, III, octahedral),
   (Au, I, linear),       (Au, III, square_planar),
   ```
   Geometry values chosen from `{"square_planar", "octahedral",
   "tetrahedral", "linear", "trigonal_planar"}` per Lippard 1995 +
   Comba-Hambley 2009.

2. **`MATRIX_LIGANDS`** — tuple of 7 `(name, donor_smiles,
   donor_atom_map)` tuples covering Cl, NH3, H2O, CN-, OH-, SCN-, CO.

3. **`_build_matrix_patterns()`** — internal helper that constructs the
   84 SMARTS patterns as
   ```
   [<M>:1]([Cl:2])[*:3].DONOR_ATOM_MAP>>
   [<M>:1](PRODUCT_DONOR)[*:3].[Cl:2]
   ```
   The H2O donor's product-side token is the deprotonated `[OH:4]`
   (first-aquation, Reedijk 1987) so `Pt_II_Cl_H2O` matches F2(a)
   `Pt_Cl_H2O_first` shape-for-shape — required by
   `test_matrix_pt_ii_h2o_matches_f2a_canonical`.  All other donors
   preserve their atom-map on the product side (1:1 substitution).

4. **`METAL_LIGAND_EXCHANGE_MATRIX`** — the 84-pattern
   `List[Tuple[str, str, str]]` constant built from
   `_build_matrix_patterns()`.

5. **`get_metal_ligand_exchange_matrix_patterns()`** — returns a fresh
   list of the 84 patterns.

6. **`get_metal_ligand_matrix_metals()`** — returns the 12-tuple of
   `(symbol, ox_state, geometry)`.

7. **`get_metal_ligand_matrix_ligands()`** — returns the 7-tuple of
   `(name, donor_smiles, donor_atom_map)`.

8. **`register_metal_ligand_exchange_rules(*,
   register_into_rule_registry=False)`** — returns the 84 patterns,
   and when `register_into_rule_registry=True` pushes each pattern into
   `molmetal_lam.reactions.rule_registry` under the
   `"metal-coord-matrix"` scope via `register_rule(..., overwrite=True)`
   (idempotent registration).

## Honest framing
* The matrix is *exhaustive* (84 cells) but **not all 84 cells are
  physically realistic** — e.g. Au_I with OH- is unusual; Pt_IV with CO
  is forbidden because Pt_IV is d6 octahedral and CO prefers low-
  oxidation-state metals.  The *physical* chemistry is gated upstream via
  the click-compat matrix; the SMARTS library just describes what the
  substitution SMILES would look like if it were allowed (verbatim from
  the test docstring).
* Each pattern uses the **canonical Pt-Cl leaving group only**.  The
  F2(a) Br-leaving variants (`Pt_Br_NH3` etc.) live in
  `METAL_LIGAND_EXCHANGE_SMARTS` and are NOT in this matrix — the test
  explicitly asserts `test_matrix_does_not_collide_with_f2a_library`.
* The metal atom-map in each pattern uses bare `[<M>:1]` (no oxidation
  state) because RDKit cannot read ox-state from a SMILES token; the
  oxidation state is encoded in the *name* only (`Pt_II`, `Ru_III`,
  etc.).  This is the same convention used by the F2(a) canonical
  patterns.

## Tests after
```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_metal_ligand_exchange_matrix.py -q --tb=short
........................................................................ [ 10%]
........................................................................ [ 21%]
........................................................................ [ 32%]
........................................................................ [ 43%]
........................................................................ [ 54%]
........................................................................ [ 65%]
........................................................................ [ 76%]
........................................................................ [ 87%]
........................................................................ [ 98%]
...........                                                              [100%]
=============================== warnings summary ===============================
...
659 passed, 1 warning in 0.85s
```

**659 tests collected, 659 tests passed, 0 failures, 0 errors.**

## Files touched
* `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py`
  — added `MATRIX_METALS`, `MATRIX_LIGANDS`, `_build_matrix_patterns`,
  `METAL_LIGAND_EXCHANGE_MATRIX`, `get_metal_ligand_exchange_matrix_patterns`,
  `get_metal_ligand_matrix_metals`, `get_metal_ligand_matrix_ligands`,
  `register_metal_ligand_exchange_rules` (+ docstrings).
* `molmetal/reports/wf290f4_fix/fix_summary.md` — this report.

## Notes for follow-up
* The F2(a) test file `molmetal/tests/test_f2a_metal_ligand_exchange.py`
  ALSO imports names that do not exist (`CL_OH_SWAP_SMARTS`,
  `LIGAND_SHELL_MMA_SMARTS`, `PT_PD_TRANSMETALATION_SMARTS`,
  `TRANSMETALATION_SMARTS`, plus their getter accessors).  These are a
  pre-existing failure (not caused by this fix) and out of scope for
  Wf-290F4 — they need a separate Wf-290F5 follow-up to wire the
  remaining curated SMARTS subsets into
  `pt_metal_ligand_exchange.py`.