# Lambda paired-input search diagnostic

Requested pockets: 5; completed pocket/seed jobs: 2.
Jobs returning generated candidates: 2; failed, empty or seed-only: 0.
Search budget: 1000 simulations, depth 3.

Initialization strategies are recorded per job; reference initialization uses the paired ligand chemistry.
Click-tile initialization uses the declared tile library. Reference coordinates define physical docking boxes.
Search docking proxies have no kcal/mol interpretation. Actual docking/PoseBusters, when enabled, are in the physical records.
Reference thresholds are redocked-ligand scores, not measured co-crystal affinity. This is not a SOTA comparison.
Lipinski pass rate is a descriptor statistic, not the published SBDD success rate.
Requested configuration and executed search settings are recorded separately in JSON.

| Metric | Value |
|---|---:|
| Candidates | 40 |
| Newly generated candidates | 40 |
| Returned input-seed candidates | 0 |
| Physical jobs completed | 2 |
| Generated products physically docked | 21 |
| Docked products passing PoseBusters | 21 |
| Candidates passing PoseBusters (chemistry, post-dock) | 0 |
| PoseBusters (chemistry) pass rate | 0.000 |
| Lipinski pass rate | 1.000 |
| AiZynth synthesis success rate (mean, mode-aware) | — |
| AiZynth n_pockets_scored | 0 |
| AiZynth n_synthesis_route_total | 0 |
| BioLM-Score affinity score (mean across pockets) | — |
| BioLM-Score n_pockets_scored | 0 |
| BioLM-Score status counts | {'disabled': 2} |
| SA mean (backend recorded in search_config) | 2.309 |
| QED mean | 0.618 |
| Descriptor docking proxy, top-1 mean | -11.124 |

| pocket | seed | status | candidates | generated | SMILES | seconds |
|---|---:|---|---:|---:|---|---:|
| test_001 | 42 | ok | 1 | 1 | `Cc1ccc(-c2ccc(C(=O)CS)cc2)cc1` | 105.61 |
| test_001 | 0 | ok | 39 | 39 | `NCCSCCc1ccc(Br)cc1` | 355.47 |

Published baseline context is maintained in `lambda_vs_sbdd_protocol_aligned.md`.
These proxy values are not compared numerically or statistically to cited docking energies.
