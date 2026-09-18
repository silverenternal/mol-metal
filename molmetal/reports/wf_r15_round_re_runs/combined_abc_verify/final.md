# Combined Sub-Fix A + B + C Verification (R15 Round Re-runs)

**Date:** 2026-09-16
**Output:** `molmetal/reports/wf_r15_round_re_runs/combined_abc_verify/`

## CLI invocation

```bash
.venv/bin/python molmetal/scripts/r4_lambda_only_run.py \
  --pockets 3 --seeds 0 --n-simulations 1000 --n-top-k 20 \
  --output-dir wf_r15_round_re_runs/combined_abc_verify \
  --manifest molmetal/reports/wf_r15_round_re_runs/combined_abc_verify/test_010_011_012_manifest.csv \
  --metal-seed cisplatin --click-rules all-5 \
  --use-pocket-conditioned-reference \
  --use-learned-prior \
  --pocket-boost-strength 2.0
```

## Sub-fixes enabled

| sub-fix | flag | value |
|---|---|---|
| A | `--pocket-boost-strength` | 2.0 (default 1.0) |
| B | `--use-learned-prior` | ON |
| C | `--use-pocket-conditioned-reference` | ON |

## Pockets used

Pockets: test_010, test_011, test_012

Source: filtered subset of `molmetal/data/crossdocked100_manifest.csv` (rows `test_010`, `test_011`, `test_012`).
Manifest written to `molmetal/reports/wf_r15_round_re_runs/combined_abc_verify/test_010_011_012_manifest.csv` WITH header row.

## Per-pocket metrics

| pocket_id | seed | n_distinct | diversity_tanimoto_mean | candidate_list first 3 |
|---|---|---|---|---|
| test_010 | 0 | 20 | 0.1065 | ['NC(Cn1c[c]([Pt])nn1)C(=O)O', 'CC(C(=O)O)n1c[c]([Pt])nn1', 'NC(Cn1nnc[c]1[Pt])C(=O)O'] |
| test_011 | 0 | 20 | 0.1065 | ['NC(Cn1c[c]([Pt])nn1)C(=O)O', 'CC(C(=O)O)n1c[c]([Pt])nn1', 'NC(Cn1nnc[c]1[Pt])C(=O)O'] |
| test_012 | 0 | 20 | 0.1065 | ['NC(Cn1c[c]([Pt])nn1)C(=O)O', 'CC(C(=O)O)n1c[c]([Pt])nn1', 'NC(Cn1nnc[c]1[Pt])C(=O)O'] |

## Full candidate_list (truncated to first 20 SMILES)

### pocket test_010

- `NC(Cn1c[c]([Pt])nn1)C(=O)O`
- `CC(C(=O)O)n1c[c]([Pt])nn1`
- `NC(Cn1nnc[c]1[Pt])C(=O)O`
- `CC(C(=O)O)n1nnc[c]1[Pt]`
- `OCC(O)n1c[c]([Pt])nn1`
- `CC(O)n1c[c]([Pt])nn1`
- `OCCOCCOCCn1c[c]([Pt])nn1`
- `ClC=Cc1ccc(-n2c[c]([Pt])nn2)cc1`
- `[Pt][c]1cn(-c2ccc3ccccc3c2)nn1`
- `[Pt][c]1cn(-c2ccc3ccccc3n2)nn1`
- `[Pt][c]1cn(Cc2ccno2)nn1`
- `[Pt][c]1cn(Cc2ccncc2)nn1`
- `[Pt][c]1cn(Cc2cnccn2)nn1`
- `[Pt][c]1cn(CC2CNCN2)nn1`
- `COc1ccccc1Cn1c[c]([Pt])nn1`
- `Fc1ccccc1Cn1c[c]([Pt])nn1`
- `Clc1ccccc1Cn1c[c]([Pt])nn1`
- `O=C(O)CCn1c[c]([Pt])nn1`
- `O=C(O)Cn1c[c]([Pt])nn1`
- `CC(=O)NCCn1c[c]([Pt])nn1`

### pocket test_011

- `NC(Cn1c[c]([Pt])nn1)C(=O)O`
- `CC(C(=O)O)n1c[c]([Pt])nn1`
- `NC(Cn1nnc[c]1[Pt])C(=O)O`
- `CC(C(=O)O)n1nnc[c]1[Pt]`
- `OCC(O)n1c[c]([Pt])nn1`
- `CC(O)n1c[c]([Pt])nn1`
- `OCCOCCOCCn1c[c]([Pt])nn1`
- `ClC=Cc1ccc(-n2c[c]([Pt])nn2)cc1`
- `[Pt][c]1cn(-c2ccc3ccccc3c2)nn1`
- `[Pt][c]1cn(-c2ccc3ccccc3n2)nn1`
- `[Pt][c]1cn(Cc2ccno2)nn1`
- `[Pt][c]1cn(Cc2ccncc2)nn1`
- `[Pt][c]1cn(Cc2cnccn2)nn1`
- `[Pt][c]1cn(CC2CNCN2)nn1`
- `COc1ccccc1Cn1c[c]([Pt])nn1`
- `Fc1ccccc1Cn1c[c]([Pt])nn1`
- `Clc1ccccc1Cn1c[c]([Pt])nn1`
- `O=C(O)CCn1c[c]([Pt])nn1`
- `O=C(O)Cn1c[c]([Pt])nn1`
- `CC(=O)NCCn1c[c]([Pt])nn1`

### pocket test_012

- `NC(Cn1c[c]([Pt])nn1)C(=O)O`
- `CC(C(=O)O)n1c[c]([Pt])nn1`
- `NC(Cn1nnc[c]1[Pt])C(=O)O`
- `CC(C(=O)O)n1nnc[c]1[Pt]`
- `OCC(O)n1c[c]([Pt])nn1`
- `CC(O)n1c[c]([Pt])nn1`
- `OCCOCCOCCn1c[c]([Pt])nn1`
- `ClC=Cc1ccc(-n2c[c]([Pt])nn2)cc1`
- `[Pt][c]1cn(-c2ccc3ccccc3c2)nn1`
- `[Pt][c]1cn(-c2ccc3ccccc3n2)nn1`
- `[Pt][c]1cn(Cc2ccno2)nn1`
- `[Pt][c]1cn(Cc2ccncc2)nn1`
- `[Pt][c]1cn(Cc2cnccn2)nn1`
- `[Pt][c]1cn(CC2CNCN2)nn1`
- `COc1ccccc1Cn1c[c]([Pt])nn1`
- `Fc1ccccc1Cn1c[c]([Pt])nn1`
- `Clc1ccccc1Cn1c[c]([Pt])nn1`
- `O=C(O)CCn1c[c]([Pt])nn1`
- `O=C(O)Cn1c[c]([Pt])nn1`
- `CC(=O)NCCn1c[c]([Pt])nn1`

## Pocket-invariance break test (pairwise Jaccard of candidate lists)

Target: pairwise Jaccard < 0.6 (was 1.0 pre-fix).

| pair | Jaccard | target | pass |
|---|---|---|---|
| test_010_vs_test_011 | 1.0000 | < 0.6 | FAIL |
| test_010_vs_test_012 | 1.0000 | < 0.6 | FAIL |
| test_011_vs_test_012 | 1.0000 | < 0.6 | FAIL |

**n_cells with distinct candidate list from cell-0:** 0 of 3

**Verdict:** `FAIL`

## Honest finding

All 3 pockets produced **identical candidate lists** with pairwise Jaccard = 1.0 (NOT < 0.6). The combined A+B+C fix did NOT break pocket-invariance on this run.

**Root cause** (per cell warnings):

```
reference_ligand_resolved: pocket_key='cisplatin_legacy' chemistry='bare_metal_alkyne' is_fallback=True fallback_reason='missing_pocket_features'
```

Sub-fix C's pocket-conditioned lookup is wired through `molmetal/molmetal_lam/lam_chem/reference_ligand_resolver.py`. The resolver needs per-pocket residue features (charge fractions, hydrophobic residue fractions, etc.) to pick a pocket-specific reference ligand. The CrossDocked100 manifest rows for test_010/011/012 carry **no residue-feature columns** — only `pocket_id, receptor_path, ligand_path, ref_path, metal_atoms, n_atoms, n_residues, source`. Without those features the resolver cannot key on per-pocket fingerprint and falls back to the legacy cisplatin_seed='[Pt]C#C'.

All 3 pockets therefore start MCTS from the same `[Pt]C#C` root, and the singleton attractor (cache hit on `_unreactive_states` + `transposition_table`) collapses every (pocket, seed) cell to the same first-state hit, regardless of sub-fix A or sub-fix B. Sub-fix A boosts per-pocket exploration, but the boost lands on the same first-state root, so the diversity gain cannot surface.

## Constraints satisfied

- No production file modified (only reading from `r4_lambda_only_run.py`, `reference_ligand_resolver.py`, manifest CSV).
- /mnt/storage was accessible for all 3 receptor PDBs.
- Output written to `molmetal/reports/wf_r15_round_re_runs/combined_abc_verify/final.md` + `final.json`.

## Recommendation for next round

Either:
- (a) **Augment manifest** with per-pocket residue-feature columns (positive_charge_fraction, negative_charge_fraction, hydrophobic_residue_fraction, total_residues) so `reference_ligand_resolver.py` can map pocket -> reference SMILES without falling back to cisplatin.
- (b) **Inline pocket_id->reference_ligand table** at the `r4_lambda_only_run.py` CLI layer (e.g., `--reference-ligand-table <csv>`) so the resolver is bypassed and the singleton attractor is broken at the root prior layer.
- (c) **Add Pt-II-eligible scaffolds per pocket** to a lookup table (similar to F2(a) MetalLigandExchange but indexed by pocket fingerprint, not metal_seed alone) and pass that table through `--metal-seed-from-pocket` + `--use-pocket-conditioned-reference` together.

Without one of those fixes, the combined A+B+C verification will always read Jaccard=1.0 regardless of `n_simulations` because the singleton attractor fires upstream of all three sub-fixes.
