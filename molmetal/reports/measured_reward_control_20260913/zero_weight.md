# Lambda paired-input search diagnostic

Requested pockets: 2; completed pocket/seed jobs: 6.
Jobs returning generated candidates: 0; failed, empty or seed-only: 6.
Search budget: 4 simulations, depth 1.

Initialization strategies are recorded per job; reference initialization uses the paired ligand chemistry.
Click-tile initialization uses the declared tile library. Reference coordinates define physical docking boxes.
Search docking proxies have no kcal/mol interpretation. Actual docking/PoseBusters, when enabled, are in the physical records.
Reference thresholds are redocked-ligand scores, not measured co-crystal affinity. This is not a SOTA comparison.
Lipinski pass rate is a descriptor statistic, not the published SBDD success rate.
Requested configuration and executed search settings are recorded separately in JSON.

| Metric | Value |
|---|---:|
| Candidates | 0 |
| Newly generated candidates | 14 |
| Returned input-seed candidates | 0 |
| Physical jobs completed | 0 |
| Generated products physically docked | 0 |
| Docked products passing PoseBusters | 0 |
| Lipinski pass rate | — |
| SA mean (backend recorded in search_config) | — |
| QED mean | — |
| Descriptor docking proxy, top-1 mean | — |

| pocket | seed | status | candidates | generated | SMILES | seconds |
|---|---:|---|---:|---:|---|---:|
| test_001 | 42 | docking_reward_unavailable | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 3.88 |
| test_001 | 0 | docking_reward_unavailable | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 3.84 |
| test_001 | 1234 | docking_reward_unavailable | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 3.85 |
| test_002 | 42 | docking_reward_unavailable | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 3.79 |
| test_002 | 0 | docking_reward_unavailable | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 3.73 |
| test_002 | 1234 | docking_reward_unavailable | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 3.71 |

Published baseline context is maintained in `lambda_vs_sbdd_protocol_aligned.md`.
These proxy values are not compared numerically or statistically to cited docking energies.
