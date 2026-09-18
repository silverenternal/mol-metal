# Multi-component metal SMILES parser

**Status:** completed
**Completed:** 2026-09-11
**Evidence:** /home/hugo/codes/try_triton_on_rocm/molmetal/reports/a1_metal_smiles_parser.md
**Owner:** (unset)

## What was delivered
A multi-component metal SMILES parser that handles disconnected metal
fragments (e.g. `N.N.Cl.Cl.[Pt]`), preserves counter-ion vs ligand roles,
and round-trips through RDKit. Used as a domain primitive for the
metal-coordination prior and the D-MPNN encoder that consumes tmQM /
MetalCytoToxDB data.

## Hard numbers
- Round-trip parse validity on **26,801 MetalCytoToxDB rows**: passes (see
  `a1_metal_smiles_parser.md` + `f1_multi_component_parser.md`)
- Counter-ion vs ligand role preserved across round-trip (e.g. `[Pt]`
  metal centre, `[Cl-]` counter-ion)
- Used by the metal-coordination prior and the D-MPNN encoder pipeline
- Companion report `f1_multi_component_parser.md` documents the parser
  validation suite

## Lessons learned
- Standard RDKit SMILES handles most cases out of the box; the parser's
  value-add is the explicit counter-ion/ligand role separation that
  MetalCytoToxDB requires.
- Counter-ion handling matters: see TODO/completed/12_counterion_ablation.md
  for the temporal-ablation finding.

## Related files
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/a1_metal_smiles_parser.md
- /home/hugo/codes/try_triton_on_rocm/molmetal/reports/f1_multi_component_parser.md