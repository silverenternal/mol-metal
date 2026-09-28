# Wf-247-B — m18 Per-Residue REDESIGN Report

## TL;DR

- **Complex-level AUC (legacy reproduced):** 0.9945
- **Per-residue AUC (synthetic proxy):** 0.9627
- **Per-residue PR-AUC (synthetic proxy):** 0.8331
- **N complexes:** 5257
- **N residue samples (synthesized):** 66303

## Honest framing

Per-residue labels do **NOT** ship with MetalCytoToxDB.  This run treats per-residue labels as a synthetic proxy derived from the complex-level active flag. Until wet-lab per-residue affinity labels ship (TODO/pending/wetlab_loi_ornl_protein_metal.md), the per-residue AUC above is a **numerator-level upper bound**: if the per-complex classifier is perfect, the per-residue AUC also collapses to ≈1 by construction because each residue inherits the parent complex's active label.  See :mod:`molmetal.adapters.per_residue_protein_metal_auc` §Honest framing.

## Reconciliation table (top 25)

| idx | protein | metal | n_residues | n_pos_residues | mean_score | max_score | true_label | complex_score |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1256 | HCT-116 p53-/- | Ru | 21 | 5 | 0.5046 | 0.5944 | 1 | 0.5280 |
| 4083 | HeLa | Ir | 13 | 5 | 0.5226 | 0.6052 | 1 | 0.5401 |
| 5061 | HeLa | Re | 16 | 5 | 0.5230 | 0.6329 | 1 | 0.5469 |
| 4469 | HeLa | Ir | 13 | 5 | 0.5352 | 0.6411 | 1 | 0.5477 |
| 4009 | A549 | Ir | 18 | 10 | 0.5577 | 0.6291 | 1 | 0.5520 |
| 4337 | 16HBE | Ir | 18 | 7 | 0.5473 | 0.6330 | 1 | 0.5540 |
| 3020 | A549 | Ru | 13 | 5 | 0.5444 | 0.6102 | 1 | 0.5541 |
| 4737 | MCF-7 | Ir | 21 | 7 | 0.5365 | 0.6482 | 1 | 0.5579 |
| 1494 | CH-1 | Ru | 18 | 10 | 0.5599 | 0.6350 | 1 | 0.5583 |
| 20 | MDA-MB-231 | Ru | 27 | 11 | 0.5450 | 0.6242 | 1 | 0.5604 |
| 2507 | HT-29 | Ru | 13 | 5 | 0.5481 | 0.6172 | 1 | 0.5612 |
| 271 | HepG2 | Ru | 19 | 5 | 0.5362 | 0.6489 | 1 | 0.5627 |
| 3447 | A375 | Ru | 11 | 0 | 0.6521 | 0.6678 | 0 | 0.6908 |
| 3082 | LS-174 | Ru | 8 | 0 | 0.6497 | 0.6831 | 0 | 0.6896 |
| 1816 | HCT-116 | Ru | 13 | 0 | 0.6234 | 0.6586 | 0 | 0.6719 |
| 4559 | HaCaT | Ir | 14 | 0 | 0.6277 | 0.6614 | 0 | 0.6646 |
| 3184 | CNE-2 | Ru | 8 | 0 | 0.6064 | 0.6340 | 0 | 0.6614 |
| 1587 | SW480 | Ru | 14 | 0 | 0.6103 | 0.6442 | 0 | 0.6595 |
| 4696 | HeLa | Ir | 8 | 0 | 0.6032 | 0.6427 | 0 | 0.6584 |
| 3690 | A549 | Ir | 8 | 0 | 0.5985 | 0.6173 | 0 | 0.6550 |
| 3047 | MCF-7 | Ru | 14 | 0 | 0.6054 | 0.6291 | 0 | 0.6546 |
| 943 | ZR-75-30 | Ru | 10 | 0 | 0.6089 | 0.6358 | 0 | 0.6522 |
| 2294 | HeLa | Ru | 8 | 0 | 0.5917 | 0.6543 | 0 | 0.6486 |
| 2729 | HepG2 | Ru | 14 | 0 | 0.5889 | 0.6311 | 0 | 0.6479 |

## Acceptance gates

| Gate | Status |
|---|---|
| G1_data_located | True (loaded 5257 rows from /home/hugo/codes/mol-metal/molmetal/data/protein_metal_binding.csv) |
| G2_per_residue_labels | False (synthetic proxy only; MetalCytoToxDB does not ship real per-residue labels) |
| G3_adapter_implemented | True (`molmetal/adapters/per_residue_protein_metal_auc.py`) |
| G4_per_complex_auc | 0.9945 |
| G5_per_residue_auc | 0.9627 |
| G6_reconciliation_table | True (≥25 rows in JSON) |
| G7_test_added | True (`molmetal/tests/test_per_residue_protein_metal.py`) |
| G8_report_written | True (this file) |

## Elapsed: 0.88s
