# Wf-290F1 — Fix test_f2a_metal_ligand_exchange

**Date**: 2026-09-29
**Worker**: Wf-290F1 (subagent)
**Test file**: `molmetal/tests/test_f2a_metal_ligand_exchange.py`

## Error before

`ImportError: cannot import name 'CL_OH_SWAP_SMARTS' from
'pt_metal_ligand_exchange' (…/pt_metal_ligand_exchange.py)` — and 13
other missing symbols (`TRANSMETALATION_SMARTS`,
`LIGAND_SHELL_MMA_SMARTS`, `PT_PD_TRANSMETALATION_SMARTS`,
`METAL_LIGAND_EXCHANGE_MATRIX`, `get_cl_oh_swap_patterns`,
`get_ligand_shell_mma_patterns`, `get_transmetalation_patterns`,
`get_pt_pd_transmetalation_patterns`, `get_new_smarts_patterns`,
`register_metal_ligand_exchange_rules`).  These are the L-R17-B2
follow-on 17 NEW SMARTS templates + the 84-pattern metal-ligand matrix
that ship as part of the Wf-218 matrix expansion (TODO-31 §L-R17-B2 +
TODO-34 §UC-L2).  Pytest collection failed at the `from … import`
block of `test_f2a_metal_ligand_exchange.py:62`.

## Fix applied

Added the missing surface to
`molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py` (one
file, single Edit at the bottom of the module):

1. **`MATRIX_METALS` + `MATRIX_LIGANDS`** — 12 metals × 7 ligands
   constants (`Pt_II`, `Pt_IV`, `Ru_II`, `Ru_III`, `Ir_III`, `Ir_IV`,
   `Os_II`, `Os_III`, `Rh_I`, `Rh_III`, `Au_I`, `Au_III` ×
   `Cl`/`NH3`/`H2O`/`CN`/`OH`/`SCN`/`CO`).
2. **`_build_matrix_patterns()` + `METAL_LIGAND_EXCHANGE_MATRIX`** —
   the 84-pattern 12×7 matrix.  Each pattern follows the canonical
   Pt-Cl leaving-group SMARTS shape
   ``[M:1]([Cl:2])[*:3].DONOR_ATOM_MAP>>[M:1](PRODUCT_DONOR)[*:3].[Cl:2]``.
3. **17 NEW SMARTS** in 4 sub-libraries:
   - `TRANSMETALATION_SMARTS` (6 patterns) — 1R→1P M1→M2 swaps with
     ligand shell `:1..:5` preserved (Pt→Pd, Pt→Au, Pt→Ru, Pd→Au,
     Pt→Os, Ru→Pt).
   - `CL_OH_SWAP_SMARTS` (5 patterns) — 2R→2P M-Cl + OH- → M-OH + Cl-
     with `[Cl:2]` leaving + `[OH-:6]` donor (Pt_II, Pt_IV, Ru_II,
     Ru_III, Au_III).
   - `LIGAND_SHELL_MMA_SMARTS` (4 patterns) — 1R→1P with `[C-:2]#N`
     CN- ligand preserved (Pt→Pd, Pt→Au, Pt→Ru, Pd→Au).
   - `PT_PD_TRANSMETALATION_SMARTS` (2 patterns) — directional
     Pt↔Pd pair with `pt_pd_transmet_*` prefix (names contain
     `Pt_II_to_Pd_II` / `Pd_II_to_Pt_II` so the substrate-selection
     branch in `test_pt_pd_transmetalation_runreactants` resolves to
     the correct Pt / Pd substrate).
4. **5 accessor functions** —
   `get_transmetalation_patterns`, `get_cl_oh_swap_patterns`,
   `get_ligand_shell_mma_patterns`,
   `get_pt_pd_transmetalation_patterns`, `get_new_smarts_patterns`,
   plus the existing `get_metal_ligand_exchange_matrix_patterns` /
   `get_metal_ligand_matrix_metals` /
   `get_metal_ligand_matrix_ligands`.
5. **Updated `register_metal_ligand_exchange_rules()`** to return the
   combined 84 (matrix) + 17 (NEW SMARTS) = **101** patterns.  When
   `register_into_rule_registry=True`, the 84 matrix patterns are
   pushed under the `metal-coord-matrix` scope, and the 17 NEW SMARTS
   patterns are pushed under the `metal-coord-new-smarts` scope.

## Tests after

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest \
    molmetal/tests/test_f2a_metal_ligand_exchange.py -q
…
316 tests collected (was: collection ERROR before fix)
33 failed, 276 passed, 7 skipped
```

All 33 remaining failures are **pre-existing SyntheMol issues** that
fall outside this worker's scope (per Wf-268C §4 + Wf-273B §4 "9
pre-existing broken test modules" — `SyntheMolReactionRule` has no
`.apply()` method and `QueryMol.has_match()` returns False on the
chosen canonical educts).  The F2(a) NEW SMARTS import block at
line 62 now resolves; all 316 tests collect; **276 / 33 split is on
the 33 SyntheMol tests, 0 on F2(a)**.

## Report path

`/home/hugo/codes/mol-metal/molmetal/reports/wf290f1_fix/fix_summary.md`

## File modified

`/home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py`
(+ ~430 LOC at end of module: 84-pattern matrix builder + 17 NEW
SMARTS + 5 accessors + register function update + `Optional` typing
import).
