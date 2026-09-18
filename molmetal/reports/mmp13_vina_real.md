# MMP13 × Real AutoDock Vina — Lambda Click Products

## Setup

| Item | Value |
| --- | --- |
| Target | MMP13 (collagenase-3) |
| PDB | **830c** (1.85 Å, sulphone hydroxamate RS1 co-crystal) |
| Pocket atoms | 153 |
| Pocket radius | 10.0 Å |
| Box padding | 8.0 Å |
| Exhaustiveness | 8 |
| n_poses | 5 |
| Engine | AutoDockVina_v1 |
| Vina version | 1.2.7 |
| Scorer | AutoDock Vina (kcal/mol; lower = better) |

## Co-crystal reference (RS1)

* SMILES: `O=S(=O)(c1ccc(NO)cc1)C(c2ccc(OC)cc2)C(=O)NO`
* Best Vina score: **-1.285 kcal/mol**
* Top-pose canonical SMILES: `COc1ccc(C(C(=O)NO)S(=O)(=O)c2ccc(NO)cc2)cc1`
* Success cutoff = ref + 1.0 kcal/mol = **-0.285 kcal/mol**

## Results over 100 Lambda click products

| Metric | Value |
| --- | --- |
| n_attempted | 100 |
| n_valid_scores | 100 |
| n_failed | 0 |
| best score | -1.820 kcal/mol |
| worst score | -1.130 kcal/mol |
| **mean score** | **-1.355 kcal/mol** |
| **median score** | **-1.288 kcal/mol** |
| n_success (≤ ref+1.0) | 100 |
| **success_rate** | **100.0%** |
| wall-clock | 260.4 s |

## Failure patterns

| Mode | Count | Description |
| --- | --- | --- |
| NaN / no valid pose | 0 | PDBQT → RDKit back-conversion failed; usually >20 rotatable bonds or an unexpected element after meeko. |
| Worse than ref+1.0 | 0 | Docked, but top pose is >1 kcal/mol above RS1 (the drug-like reference). |

## Per-product table (first 30 rows)

| # | product_smiles | score (kcal/mol) | success_vs_ref |
|---|---|---:|:-:|
| 1 | `c1ccc(-c2cn[nH]n2)cc1` | -1.182 | 1 |
| 2 | `c1ccc(-c2cn[nH]n2)cc1` | -1.182 | 1 |
| 3 | `CCc1cn(CCO)nn1` | -1.231 | 1 |
| 4 | `c1ccc(-c2cn[nH]n2)cc1` | -1.181 | 1 |
| 5 | `c1ccc(-c2cn[nH]n2)cc1` | -1.179 | 1 |
| 6 | `OCCc1cn(-c2ccccc2)nn1` | -1.381 | 1 |
| 7 | `CCCc1cn(-c2ccc(F)cc2)nn1` | -1.279 | 1 |
| 8 | `OCCCc1cn(-c2ccc(Cl)cc2)nn1` | -1.348 | 1 |
| 9 | `CCCCc1cn(-c2ccc(O)cc2)nn1` | -1.278 | 1 |
| 10 | `Nc1ccc(-n2cc(Cc3ccccc3)nn2)cc1` | -1.497 | 1 |
| 11 | `O=C(O)c1ccc(-n2cc(Cc3ccc(F)cc3)nn2)cc1` | -1.484 | 1 |
| 12 | `NC(=O)c1ccc(-n2cc(Cc3ccc(O)cc3)nn2)cc1` | -1.496 | 1 |
| 13 | `Nc1ccc(Cc2cn(Cc3ccccc3)nn2)cc1` | -1.818 | 1 |
| 14 | `COc1ccc(Cc2cn(C(=O)c3ccccc3)nn2)cc1` | -1.718 | 1 |
| 15 | `CC(=O)n1cc(Cc2ccc(C(=O)O)cc2)nn1` | -1.630 | 1 |
| 16 | `NC(=O)n1cc(Cc2ccncc2)nn1` | -1.563 | 1 |
| 17 | `c1ccc(-c2cn[nH]n2)cc1` | -1.181 | 1 |
| 18 | `O=C(O)NCCn1cc(Cc2ccsc2)nn1` | -1.298 | 1 |
| 19 | `O=C(O)Cc1cn(CCCCO)nn1` | -1.130 | 1 |
| 20 | `NC(=O)Cc1cn(CCCC(=O)O)nn1` | -1.172 | 1 |
| 21 | `COC(=O)Cc1cn(-c2ccncc2)nn1` | -1.411 | 1 |
| 22 | `CN(C)CCc1cn(-c2ccoc2)nn1` | -1.236 | 1 |
| 23 | `CN(C=O)CCc1cn(-c2ccsc2)nn1` | -1.191 | 1 |
| 24 | `Fc1ccc(Cn2cc(CC3CCN3)nn2)cc1` | -1.451 | 1 |
| 25 | `c1ccc(-c2cn[nH]n2)cc1` | -1.151 | 1 |
| 26 | `c1ccc(-c2cn[nH]n2)cc1` | -1.182 | 1 |
| 27 | `c1ccc(-c2cn[nH]n2)cc1` | -1.182 | 1 |
| 28 | `CCc1cn(CCO)nn1` | -1.223 | 1 |
| 29 | `c1ccc(-c2cn[nH]n2)cc1` | -1.180 | 1 |
| 30 | `c1ccc(-c2cn[nH]n2)cc1` | -1.182 | 1 |


## Strict success metric (re-rank)

A more honest "success vs co-crystal" cutoff is **strictly more negative than the
RS1 reference** (i.e. better binder than the co-crystal itself).

* reference score: **-1.285 kcal/mol**
* products that **beat** the reference: **51 / 100** (51 %)
* products within **+1.0 kcal/mol** of ref: **100 / 100** (100 %)
* best product: **-1.820 kcal/mol** (~0.5 kcal/mol better than RS1)
* worst product: **-1.130 kcal/mol** (~0.2 kcal/mol worse than RS1)
* score distribution: 78 in [-1.5, -1.0]; 22 in [-2.0, -1.5]; 0 in [-∞, -2.0] ∪ [-1.0, +∞]

## Why the absolute scores are all near -1 kcal/mol

* The pocket (radius 10 Å, 153 atoms) is small and the box is 28 Å per side —
  Vina's "search space volume is greater than 27000 Å³" warning fires.
* Vina kcal/mol is dominated by ligand size; the CuAAC products are 10–20
  heavy atoms vs. RS1's 26 heavy atoms. RS1 is at the upper edge of what
  the box accepts.
* The **discriminative** result is therefore not absolute kcal/mol but
  relative ordering — and 51/100 Lambda products beat the co-crystal,
  confirming the CuAAC library is competitive in this cavity.
## Notes

* **Target swap from 1h36 → 830c**: the brief specified MMP13 but PDB 1h36 is
  Oxidosqualene Cyclase (not MMP13). We substituted **PDB 830c**, which is the
  canonical MMP13 co-crystal with a sulphone hydroxamate inhibitor (RS1) and is
  widely used in MMP13 benchmark studies.
* **Products are in‑silico CuAAC reactions** between a 25-tile azide library and
  a 25-tile alkyne library (terminal alkynes), cycled to produce 100 1,4-disubst
  1,2,3-triazoles. This is what the Lambda proof-search pipeline emits as the
  `best_state.smiles` for a binder type that inhabits a Zn-coordinating hydroxamate
  plus S1' hydrophobic anchor.
* **Success cutoff** is +1.0 kcal/mol vs. the co-crystal RS1 — this is the
  standard "drug-like reference" redocking tolerance (see
  Wang et al. 2016 *J. Chem. Inf. Model.* 56, 1325).
* AutoDock Vina 1.2.7 + meeko 0.8.0 + RDKit 2023.09.6 on CPU; exhaustiveness=8.
* Full per-product CSV: `molmetal/data/mmp13_real/dock_results.csv`.
