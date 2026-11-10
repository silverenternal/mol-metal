# Wf-A163 branch V — adversarial re-derivation of the c7 ligand-only measurement

- tag: **MEASURED**  date: 2026-11-01  wave: Wf-A163 (branch V — adversarial re-derivation)
- pool: `molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`
- script: `molmetal/scripts/wfa163_c7_rederive.py`
- GPU: 0 (CPU-only, RDKit-only, no checkpoint, no decode)

## Independence contract

- **imports_tool_under_test**: False
- **connected_components**: own iterative BFS over a hand-built bond adjacency (no Chem.GetMolFrags on the primary path)
- **metal_predicate**: extracted with ast from molmetal/molmetal_lam/lam_chem/*.py at run time; the list is NOT written into this file
- **mcnemar**: own exact integer/rational implementation; scipy.stats.binomtest used only as a cross-check
- **csv_reader**: own csv.reader-based reader (header index map), not csv.DictReader

## Metal predicate

Primary predicate: **dblock_platinai** (`molmetal/molmetal_lam/lam_chem/platinai_dataset.py::TRANSITION_METALS`, 30 symbols), extracted at run time with `ast` — the list is NOT copied into the script.

Why: It is the repo's own 'is this atom a metal' vocabulary, read from its own source rather than copied, so this script cannot inherit a list that was tuned to make the headline come out.

## Reproduction

| quantity | published | re-derived | delta | verdict |
|---|---|---|---|---|
| metal_inclusive.single_fragment_rate (vs A162) | 0.08509630565203663 | 0.08509630565203663 | 0 | **MATCH** |
| ligand_only.single_fragment_rate (vs A162) | 0.02083991158825387 | 0.02083991158825387 | 0 | **MATCH** |
| metal_inclusive.n_single_fragment (vs A162) | 1078 | 1078 | 0 | **MATCH** |
| ligand_only.n_single_fragment (vs A162) | 264 | 264 | 0 | **MATCH** |
| contingency.a (vs A162:a_both_connected) | 220 | 220 | 0 | **MATCH** |
| contingency.b (vs A162:b_ligand_only_connected_metal_inclusive_not) | 44 | 44 | 0 | **MATCH** |
| contingency.c (vs A162:c_metal_inclusive_connected_ligand_only_not) | 858 | 858 | 0 | **MATCH** |
| contingency.d (vs A162:d_neither_connected) | 11546 | 11546 | 0 | **MATCH** |
| mcnemar.p_value_two_sided (vs A162) | 8.635435522167073e-197 | 8.635435522167073e-197 | 0 | **MATCH** |
| metal_inclusive.mean_n_frags (vs A162) | 10.244316387748658 | 10.244316387748658 | 0 | **MATCH** |
| ligand_only.mean_n_frags (vs A162) | 11.602778654878433 | 11.602778654878433 | 0 | **MATCH** |
| published_headline.raw_axis.single_fragment_rate_over_parsed (metrics/by_metric/cfm_fragmentation_postproc_arm.json) | 0.08509630565203663 | 0.08509630565203663 | 0 | **MATCH** |
| published_headline.raw_axis.n_single (metrics/by_metric/cfm_fragmentation_postproc_arm.json) | 1078 | 1078 | 0 | **MATCH** |

**n_match = 13, n_mismatch = 0.**

## Denominator probe

- rows in CSV: **12669**
- empty SMILES: **0**
- unparseable: **1**
- analysed denominator: **12668**
- the CSV holds 12669 data rows; 1 of them do not parse under RDKit (see unparseable_sites), so the analysed denominator is 12668
- rate if the denominator were 12669 instead: 0.08508958875996527
  - unparseable site: `molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv:9848` (epoch 13, len 244) — `C.CCC1C23C45OC6(C42C563)C23C45P6C78C69C6(C1%10C1(C%11(C)C%12(CCCCl)P(C(C(C)C%13C(C)C%13C(C)C)C%13(P%14CC%14C%14CC%14)C%14%15C%13%16C%14%13C42C%15%16%13)C%12%111)C%1076)C93C81C23N4C25[C]25C31[C]42[Ru]5`
    - RDKit's own reason: `ValueError: Maximum BFS search size exceeded.
This is likely due to a highly symmetric fused ring system.`

## Per-epoch vs the published metric

- per-epoch raw_single_rate vs `metrics/by_metric/cfm_fragmentation_postproc_arm.json:/per_epoch`: **18/18 MATCH**
- per-epoch n_input vs the published n_input: **18/18** MATCH

| epoch | n_input (rederived / published) | n_analyzed | raw_single_rate (rederived / published) | verdict | ligand-only n |
|---|---|---|---|---|---|
| 0 | 704 / 704 | 704 | 0.028409090909090908 / 0.028409090909090908 | **MATCH** | 9 |
| 1 | 704 / 704 | 704 | 0.022727272727272728 / 0.022727272727272728 | **MATCH** | 1 |
| 2 | 704 / 704 | 704 | 0.04403409090909091 / 0.04403409090909091 | **MATCH** | 7 |
| 3 | 704 / 704 | 704 | 0.08380681818181818 / 0.08380681818181818 | **MATCH** | 6 |
| 4 | 704 / 704 | 704 | 0.08948863636363637 / 0.08948863636363637 | **MATCH** | 13 |
| 5 | 704 / 704 | 704 | 0.09517045454545454 / 0.09517045454545454 | **MATCH** | 15 |
| 6 | 704 / 704 | 704 | 0.09517045454545454 / 0.09517045454545454 | **MATCH** | 24 |
| 7 | 704 / 704 | 704 | 0.07528409090909091 / 0.07528409090909091 | **MATCH** | 18 |
| 8 | 704 / 704 | 704 | 0.10227272727272728 / 0.10227272727272728 | **MATCH** | 17 |
| 9 | 704 / 704 | 704 | 0.08522727272727272 / 0.08522727272727272 | **MATCH** | 16 |
| 10 | 703 / 703 | 703 | 0.10668563300142248 / 0.10668563300142248 | **MATCH** | 17 |
| 11 | 704 / 704 | 704 | 0.10227272727272728 / 0.10227272727272728 | **MATCH** | 20 |
| 12 | 703 / 703 | 703 | 0.08819345661450925 / 0.08819345661450925 | **MATCH** | 12 |
| 13 | 703 / 703 | 702 | 0.08974358974358974 / 0.08974358974358974 | **MATCH** | 18 |
| 14 | 704 / 704 | 704 | 0.09517045454545454 / 0.09517045454545454 | **MATCH** | 17 |
| 15 | 704 / 704 | 704 | 0.09375 / 0.09375 | **MATCH** | 11 |
| 16 | 704 / 704 | 704 | 0.11221590909090909 / 0.11221590909090909 | **MATCH** | 22 |
| 17 | 704 / 704 | 704 | 0.12215909090909091 / 0.12215909090909091 | **MATCH** | 21 |

## Cross-check against the pool's own columns

- this file's own BFS vs the pool's precomputed `n_frags_raw` column: **0 mismatches** over 12668 rows
- largest `largest_frag_frac` absolute delta vs the pool's column: 5.000000000000143e-05 (the pool's column is rounded to 4 dp)

## Probe: metal predicate

| predicate | n symbols | metal-incl rate | ligand-only rate | ligand-only n |
|---|---|---|---|---|
| dblock_platinai | 30 | 0.08509630565203663 | 0.02083991158825387 | 264 |
| dblock_tmqm | 30 | 0.08509630565203663 | 0.02083991158825387 | 264 |
| mle82 | 82 | 0.08509630565203663 | 0.02083991158825387 | 264 |
| pool_derived | 6 | 0.08509630565203663 | 0.02083991158825387 | 264 |
| STRAWMAN_metalloid_widened **(strawman control)** | 37 | 0.08509630565203663 | 0.018866435112093464 | 239 |

## Probe: row order

- rows shuffled with random.Random(0).shuffle, then re-measured
- metal-inclusive rate after shuffle: 0.08509630565203663
- ligand-only rate after shuffle: 0.02083991158825387
- identical to original: **True**

## Probe: epoch column

- every row's epoch label cyclically shifted by one, then re-measured
- pooled metal-inclusive rate after the shift: 0.08509630565203663
- pooled ligand-only rate after the shift: 0.02083991158825387
- pooled rate moved: **False**
- per-epoch rates changed: **17 / 18**

## Probe: connected definition

- metal-inclusive, rule A (`one connected component AND >= 1 heavy atom (as published)`): 0.08509630565203663 (n=1078)
- metal-inclusive, rule B (`one connected component (component count only)`): 0.08509630565203663 (n=1078)
- ligand-only, rule A: 0.02083991158825387 (n=264)
- ligand-only, rule B: 0.02083991158825387 (n=264)
- the `>= 1 heavy atom` clause does hidden work: **False**
- metal-inclusive with `>= 2 heavy atoms`: 0.08509630565203663 (n=1078)
- rows with < 2 heavy atoms: 0

## Probe: free-halide rows excluded

- rows carrying a free (single-atom) halide counterion are removed
- free-halide rows found: **1903**
- analysed after exclusion: **10765**
- metal-inclusive rate: 0.10013934045517882 (n=1078)
- ligand-only rate: 0.024523920111472364 (n=264)
- contingency: {'a': 220, 'b': 44, 'c': 858, 'd': 9643}
- the NUMERATOR is unchanged (1078 metal-inclusive, 264 ligand-only): every free-halide row is by construction a MULTI-fragment row (the halide is its own component), so the 1903 halide rows contribute 0 to both numerators and 1903 to the denominator. Excluding them therefore RAISES both rates (the ligand-only rate by 17.7 % relative). They DILUTE the ligand-only arm, they do not inflate it.
- site list (1903 line numbers, one per row): `rederive.json` -> `/probes/free_halide_excluded/free_halide_line_numbers` (source `molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`; 12669 rows scanned).

## McNemar cross-check

- headline p-value implementation: this file's own exact rational implementation (fractions.Fraction + math.comb)
- scipy available: True
- own p: 8.635435522167073e-197
- scipy.stats.binomtest p: 8.635435522166859e-197
- own p vs the PUBLISHED p: **bit-for-bit identical** (the published value is the same exact-rational conversion, so the publisher used an exact rational computation, not scipy's floating-point one)
- own p vs scipy's binomtest: abs delta 2.1387280120820548e-210, relative delta 2.476688068125762e-14 (scipy's float summation is the less accurate of the two)

## Verdict

**The c7 headline REPRODUCES bit-for-bit under an independently written implementation.**

(13/13 quantities matched; 0 mismatched.)

### What could NOT refute the headline (each a bounded claim)

1. The metal-inclusive rate 0.08509630565203663 is not an artefact of the metal predicate: it is identical under all four repo/pool metal vocabularies (30-dblock from two loaders, the 82-metal MLE list, and the 6-symbol set derived from the pool itself).
2. The `>= 1 heavy atom` clause in the connected definition does **no** hidden work: dropping it (rule B) leaves both arms unchanged, and 0 rows in the pool have fewer than 2 heavy atoms, so even the stricter `>= 2 heavy atoms` variant gives the identical rate.
3. The rate is invariant to row order and to the epoch column: a shuffle and a cyclic epoch mislabelling both leave the pooled rates and the contingency table unchanged (the per-epoch breakdown does move, 17/18 epochs, which is what the epoch column is actually for).
4. The 1903 free-halide rows do not inflate the ligand-only arm; they DILUTE it (numerator unchanged, denominator +1903).
5. The denominator is 12668, not 12669, for a stated and independently confirmed reason: exactly one row (`.../generated_smiles.csv:9848`, epoch 13) fails RDKit sanitisation with 'Maximum BFS search size exceeded' (a highly symmetric fused ring system). Using 12669 instead would move the headline to 0.08508958875996527 (delta -6.7e-06).

### The one place the invariance WOULD break

The ligand-only arm is invariant to the choice among genuine METAL vocabularies, but NOT to a predicate widened to include the metalloids that occur in this pool (B, Si, As, Se): that strawman predicate drops the ligand-only rate to 0.018866435112093464 (n=239). That is a control, not a defect: the published rule (d-block metals) is a metal rule, and the pool's metalloid atoms are part of the ligand skeleton.
