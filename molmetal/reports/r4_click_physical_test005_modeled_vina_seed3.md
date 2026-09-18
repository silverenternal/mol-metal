# Lambda paired-input search diagnostic

Requested pockets: 1; completed pocket/seed jobs: 3.
Jobs returning generated candidates: 3; failed, empty or seed-only: 0.
Search budget: 4 simulations, depth 1.

Initialization strategies are recorded per job; reference initialization uses the paired ligand chemistry.
Click-tile initialization uses the declared tile library. Reference coordinates define physical docking boxes.
Search docking proxies have no kcal/mol interpretation. Actual docking/PoseBusters, when enabled, are in the physical records.
Reference thresholds are redocked-ligand scores, not measured co-crystal affinity. This is not a SOTA comparison.
Lipinski pass rate is a descriptor statistic, not the published SBDD success rate.
Requested configuration and executed search settings are recorded separately in JSON.

| Metric | Value |
|---|---:|
| Candidates | 7 |
| Newly generated candidates | 7 |
| Returned input-seed candidates | 0 |
| Physical jobs completed | 3 |
| Generated products physically docked | 7 |
| Docked products passing PoseBusters | 7 |
| Lipinski pass rate | 1.000 |
| SA mean (backend recorded in search_config) | 2.564 |
| QED mean | 0.678 |
| Descriptor docking proxy, top-1 mean | -10.189 |

| pocket | seed | status | candidates | generated | SMILES | seconds |
|---|---:|---|---:|---:|---|---:|
| test_005_modeled_ASP_B101 | 42 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 26.33 |
| test_005_modeled_ASP_B101 | 0 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 27.12 |
| test_005_modeled_ASP_B101 | 1234 | ok | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 25.03 |

Published baseline context is maintained in `lambda_vs_sbdd_protocol_aligned.md`.
These proxy values are not compared numerically or statistically to cited docking energies.
