# Lambda paired-input search diagnostic

Requested pockets: 1; completed pocket/seed jobs: 1.
Jobs returning generated candidates: 0; failed, empty or seed-only: 1.
Search budget: 1000 simulations, depth 3.

Initialization strategies are recorded per job; reference initialization uses the paired ligand chemistry.
Click-tile initialization uses the declared tile library. Reference coordinates define physical docking boxes.
Search docking proxies have no kcal/mol interpretation. Actual docking/PoseBusters, when enabled, are in the physical records.
Reference thresholds are redocked-ligand scores, not measured co-crystal affinity. This is not a SOTA comparison.
Lipinski pass rate is a descriptor statistic, not the published SBDD success rate.
Requested configuration and executed search settings are recorded separately in JSON.

| Metric | Value |
|---|---:|
| Candidates | 0 |
| Newly generated candidates | 0 |
| Returned input-seed candidates | 0 |
| Physical jobs completed | 0 |
| Generated products physically docked | 0 |
| Docked products passing PoseBusters | 0 |
| Candidates passing PoseBusters (chemistry, post-dock) | 0 |
| PoseBusters (chemistry) pass rate | — |
| Lipinski pass rate | — |
| SA mean (backend recorded in search_config) | — |
| QED mean | — |
| Descriptor docking proxy, top-1 mean | — |

| pocket | seed | status | candidates | generated | SMILES | seconds |
|---|---:|---|---:|---:|---|---:|
| p1 | 42 | seed_parse_fail | 0 | 0 | `` | 3.20 |

Published baseline context is maintained in `lambda_vs_sbdd_protocol_aligned.md`.
These proxy values are not compared numerically or statistically to cited docking energies.
