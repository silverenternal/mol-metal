# Lambda paired-input search diagnostic

Requested pockets: 10; completed pocket/seed jobs: 30.
Jobs returning candidates: 18; failed or empty: 12.
Search budget: 10 simulations, depth 3.

This diagnostic uses the paired reference ligand as a search seed. It does not establish de novo SBDD performance.
Docking values are descriptor proxies with no kcal/mol interpretation. Physical docking, PoseBusters validity,
co-crystal thresholds, and learned-oracle measurements are unavailable in this runner.
Lipinski pass rate is a descriptor statistic, not the published SBDD success rate.
Requested configuration and executed search settings are recorded separately in JSON.

| Metric | Value |
|---|---:|
| Candidates | 18 |
| Lipinski pass rate | 1.000 |
| SA mean (backend recorded in search_config) | 3.363 |
| QED mean | 0.606 |
| Descriptor docking proxy, top-1 mean | -15.260 |

| pocket | seed | status | candidates | SMILES | seconds |
|---|---:|---|---:|---|---:|
| test_000 | 42 | no_candidates | 0 | `` | 2.70 |
| test_000 | 0 | no_candidates | 0 | `` | 2.51 |
| test_000 | 1234 | no_candidates | 0 | `` | 2.61 |
| test_001 | 42 | ok | 1 | `COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(C)=O` | 2.72 |
| test_001 | 0 | ok | 1 | `COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(C)=O` | 2.72 |
| test_001 | 1234 | ok | 1 | `COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(C)=O` | 2.72 |
| test_002 | 42 | no_candidates | 0 | `` | 2.54 |
| test_002 | 0 | no_candidates | 0 | `` | 2.52 |
| test_002 | 1234 | no_candidates | 0 | `` | 2.55 |
| test_003 | 42 | ok | 1 | `Nc1cc(S(O)(O)O)c(N)c2c1C(=O)c1ccccc1C2=O` | 2.72 |
| test_003 | 0 | ok | 1 | `Nc1cc(S(O)(O)O)c(N)c2c1C(=O)c1ccccc1C2=O` | 2.69 |
| test_003 | 1234 | ok | 1 | `Nc1cc(S(O)(O)O)c(N)c2c1C(=O)c1ccccc1C2=O` | 2.70 |
| test_004 | 42 | ok | 1 | `CC(C)NC[C@H](O)COc1cccc2ccccc12` | 2.69 |
| test_004 | 0 | ok | 1 | `CC(C)NC[C@H](O)COc1cccc2ccccc12` | 2.73 |
| test_004 | 1234 | ok | 1 | `CC(C)NC[C@H](O)COc1cccc2ccccc12` | 2.72 |
| test_005 | 42 | ok | 1 | `CC(C)[C@@H]1NC(=O)[C@]2(C)CSC(=N2)c2cccc(n2)CNC(=O)C[C@@H](/C=C/CCS)NC1=O` | 2.74 |
| test_005 | 0 | ok | 1 | `CC(C)[C@@H]1NC(=O)[C@]2(C)CSC(=N2)c2cccc(n2)CNC(=O)C[C@@H](/C=C/CCS)NC1=O` | 2.75 |
| test_005 | 1234 | ok | 1 | `CC(C)[C@@H]1NC(=O)[C@]2(C)CSC(=N2)c2cccc(n2)CNC(=O)C[C@@H](/C=C/CCS)NC1=O` | 2.76 |
| test_006 | 42 | ok | 1 | `C[C@H](CCC(N)=O)[C@H]1CC[C@H]2[C@@H]3C(=O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C` | 2.72 |
| test_006 | 0 | ok | 1 | `C[C@H](CCC(N)=O)[C@H]1CC[C@H]2[C@@H]3C(=O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C` | 2.74 |
| test_006 | 1234 | ok | 1 | `C[C@H](CCC(N)=O)[C@H]1CC[C@H]2[C@@H]3C(=O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C` | 2.71 |
| test_007 | 42 | no_candidates | 0 | `` | 2.52 |
| test_007 | 0 | no_candidates | 0 | `` | 2.51 |
| test_007 | 1234 | no_candidates | 0 | `` | 2.52 |
| test_008 | 42 | ok | 1 | `Nc1ccc2nccn2c1` | 2.65 |
| test_008 | 0 | ok | 1 | `Nc1ccc2nccn2c1` | 2.66 |
| test_008 | 1234 | ok | 1 | `Nc1ccc2nccn2c1` | 2.66 |
| test_009 | 42 | no_candidates | 0 | `` | 2.56 |
| test_009 | 0 | no_candidates | 0 | `` | 2.53 |
| test_009 | 1234 | no_candidates | 0 | `` | 2.57 |

Published baseline context is maintained in `lambda_vs_sbdd_protocol_aligned.md`.
These proxy values are not compared numerically or statistically to cited docking energies.
