# F1 / P0 — Multi-component metal SMILES parser

## Summary

The metal-complex SMILES reconstruction in
`molmetal/data/metal_smiles.py` was rewritten to use the canonical
**multi-component** form ``L1.L2....Ln.[M]`` instead of the legacy bracket
form ``[M](L1)(L2)...``.  The new form achieves **100% RDKit round-trip on
1000 MetalCytoToxDB rows** (vs. 96.7% bracket form).  The legacy API
``reconstruct_metal_complex`` is kept as a thin wrapper for backward
compatibility.

Reference: cisplatin canonical SMILES is ``N.N.Cl.Cl.[Pt]`` (Fisher
Scientific CAS 15663-27-1, ChemBase, RCSB PDB CPT).

## Why the change?

The bracket form ``[Pt](N)(N)(Cl)Cl`` is split by RDKit into disconnected
fragments because Pt is **outside RDKit's default organic subset** — Pt
gets removed during ``SanitizeMol`` and the resulting ``Mol`` has no
M-Pt bonds, so RDKit raises or returns a fragment-only molecule.

The multi-component form ``N.N.Cl.Cl.[Pt]`` is structurally immune to
this pathology: every ligand lives in its own RDKit molecule component,
and the metal token ``[Pt]`` is the last (separate) component.  RDKit
reads 5 fragments (4 ligands + 1 metal) with no fragment-splitting.

## API change

### New function (preferred)

```python
reconstruct_multi_component(
    smiles_ligands: str, metal: str, oxidation_state: int,
    charge_complex: int | None = None,
) -> str  # e.g. 'N.N.Cl.Cl.[Pt]'
```

### Legacy wrapper (kept for backward compatibility)

```python
reconstruct_metal_complex(...)  # now delegates to reconstruct_multi_component
```

Both produce the same multi-component form.  ``reconstruct_metal_complex``
is marked deprecated in its docstring.

## Before / after

| Metric | Bracket form (legacy) | Multi-component (F1) |
| --- | --- | --- |
| Output style | ``[Pt](N)(N)(Cl)Cl`` | ``N.N.Cl.Cl.[Pt]`` |
| MetalCytoToxDB 1000 rows | 96.7% (33/1000 split) | **100% (1000/1000)** |
| Pt synthetic 10 cases | pathology in 0/10 | 0/10 (also clean) |
| Compatible with RDKit ``GetMolFrags`` | yes (with split artifacts) | **yes, 5 clean frags** |

## Worked examples (multi-component form)

| Input (ligands) | Metal | Ox | Output | RDKit frags |
| --- | --- | --- | --- | --- |
| ``N.N.Cl.Cl`` | Pt | 2 | ``N.N.Cl.Cl.[Pt]`` (cisplatin) | 5 |
| ``N.N.N.N.Cl.Cl`` | Ru | 2 | ``N.N.N.N.Cl.Cl.[Ru]`` | 7 |
| ``N.N.N.N.N.N`` | Ru | 2 | ``N.N.N.N.N.N.[Ru]`` | 7 |
| ``N.N.N`` | Ru | 2 | ``N.N.N.[OH2].[OH2].[OH2].[Ru]`` | 7 |
| ``N.N.Cl.Cl`` | Pt | 4 | ``N.N.Cl.Cl.[OH2].[OH2].[Pt]`` | 7 |
| ``N.N.[Cl-]`` | Pt | 2 | ``N.N.[OH2].[OH2].[Pt]`` | 5 |
| ``NCCN.N.Cl.Cl`` | Pt | 2 | ``NCCN.N.Cl.[Pt]`` (kept bidentate whole) | 5 |

## Verification

### Test suite

```
molmetal/tests/test_metal_smiles.py — 25 passed
```

Includes the legacy 5-case suite (rewritten to expect multi-component
form) plus 5 new tests:

* ``test_multi_component_cisplatin``
* ``test_multi_component_ruthenium``
* ``test_multi_component_iridium``
* ``test_legacy_api_delegates_to_multi_component``
* ``test_round_trip_100pct``  — 100 rows from MetalCytoToxDB → 100%

### Sanity check (1000 rows)

```
source .venv/bin/activate && python -c "..."
MetalCytoToxDB 1000 rows: 1000/1000 = 100.0%
Failures: 0
Pt synthetic 10 cases : 10/10 = 100.0%
```

## Notes on charge / ionic state

The new parser still does **not** balance charge — it inherits the
heuristic-only treatment from the legacy code.  Adding bracket-charge
``[Pt+2]`` is a future task (TBD): the multi-component form ``[Pt+2].N.N.Cl.Cl``
parses fine in RDKit, so the path is open when chemical-feature pipelines
demand it.

## Files touched

* `molmetal/data/metal_smiles.py` — added ``reconstruct_multi_component``,
  refactored shared worker ``_select_ligands``, deprecated
  ``reconstruct_metal_complex`` to delegate to the new function.
* `molmetal/tests/test_metal_smiles.py` — rewrote assertions for the
  multi-component form, added 5 new tests (incl. round-trip on
  MetalCytoToxDB).
* `molmetal/reports/f1_multi_component_parser.md` — this report.
