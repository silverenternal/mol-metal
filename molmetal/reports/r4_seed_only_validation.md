# Lambda paired-input search diagnostic

Requested pockets: 1; completed pocket/seed jobs: 1.
Jobs returning generated candidates: 0; failed, empty or seed-only: 1.
Search budget: 10 simulations, depth 3.

This diagnostic uses the paired reference ligand as a search seed. It does not establish de novo SBDD performance.
Docking values are descriptor proxies with no kcal/mol interpretation. Physical docking, PoseBusters validity,
co-crystal thresholds, and learned-oracle measurements are unavailable in this runner.
Lipinski pass rate is a descriptor statistic, not the published SBDD success rate.
Requested configuration and executed search settings are recorded separately in JSON.

| Metric | Value |
|---|---:|
| Candidates | 0 |
| Newly generated candidates | 0 |
| Returned input-seed candidates | 1 |
| Lipinski pass rate | — |
| SA mean (backend recorded in search_config) | — |
| QED mean | — |
| Descriptor docking proxy, top-1 mean | — |

| pocket | seed | status | candidates | generated | SMILES | seconds |
|---|---:|---|---:|---:|---|---:|
| test_001 | 42 | seed_only | 1 | 0 | `COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(C)=O` | 3.00 |

Published baseline context is maintained in `lambda_vs_sbdd_protocol_aligned.md`.
These proxy values are not compared numerically or statistically to cited docking energies.
