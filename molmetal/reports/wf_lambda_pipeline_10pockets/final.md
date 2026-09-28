# TODO/32 B5 — Lambda pipeline smoke, 10-pocket expansion

- n_pockets: 10 (test_010..test_019)
- pipeline: PLL signature → PLL heuristic → LamMCTS (KataGo FPU=0.25) → top-100 candidates → closed β-NF annotation
- n_simulations_per_pocket: 996 max_depth: 3

## Aggregate metrics

- n_unique_valid_smi_mean: 100.00
- validity_rate_mean: 1.0000
- uniqueness_rate_mean: 1.0000
- sa_mean: 2.5070
- qed_mean: 0.7212
- n_emitted_total: 1000 n_valid_total: 1000 n_unique_total: 1000
- nfe_total: 9960 nfe_reductions_total: 8652679
- wallclock_s_sum: 7482.6 (mean per pocket: 748.3)
- rules_used (union): ['CuAAC', 'DielsAlder', 'SPAAC', 'SPC', 'ThiolEne']
- pll_k: 16 pll_knn: 8 pll_available_all: True

## Per-pocket highlights

| pocket | pocket_smiles | n_unique_valid | sa_mean | qed_mean | wallclock_s |
|---|---|---:|---:|---:|---:|
| `test_010` | `NC1=c2ncn3c2=NCN1[C@@H]1C[C@H](CO[P@@](=..` | 100 | 2.330 | 0.769 | 504.1 |
| `test_011` | `Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO[P@](=O)(O..` | 100 | 2.712 | 0.494 | 1133.1 |
| `test_012` | `C[C@]12CC[C@H](O)CC1=CC[C@@H]1[C@@H]2CC[..` | 100 | 3.029 | 0.624 | 592.1 |
| `test_013` | `Cc1ncc(COP(=O)(O)O)c(C)c1O` | 100 | 2.316 | 0.678 | 996.4 |
| `test_014` | `CNCCCC[C@H](N)C(=O)O` | 100 | 2.566 | 0.741 | 302.6 |
| `test_015` | `CNCCCC[C@H](N)C(=O)O` | 100 | 2.424 | 0.801 | 907.7 |
| `test_016` | `COCCOc1cnc(N)nc1C1=CN=C2C=CC(C#CC(C)(C)O..` | 100 | 2.450 | 0.777 | 602.5 |
| `test_017` | `Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)..` | 100 | 2.450 | 0.777 | 665.5 |
| `test_018` | `CC(=O)N[C@H]1/C(=N/OC(=O)Nc2ccccc2)O[C@H..` | 100 | 2.420 | 0.806 | 820.1 |
| `test_019` | `C[C@H](NC(=O)[C@H](Cc1ccc(-c2ccccc2)cc1)..` | 100 | 2.373 | 0.745 | 958.4 |

PLL + KataGo-style FPU (Liu & Feng 2026, DOI: 10.1021/acs.jcim.6c02460).  Closed β-NF term via `_molecule_to_lambda_expr` (atoms=left-associated app).
