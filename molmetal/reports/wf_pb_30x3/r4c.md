# Lambda paired-input search diagnostic

Requested pockets: 10; completed pocket/seed jobs: 30.
Jobs returning generated candidates: 0; failed, empty or seed-only: 30.
Search budget: 100 simulations, depth 3.

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
| Returned input-seed candidates | 15 |
| Physical jobs completed | 0 |
| Generated products physically docked | 0 |
| Docked products passing PoseBusters | 0 |
| Candidates passing PoseBusters (chemistry, post-dock) | 0 |
| PoseBusters (chemistry) pass rate | — |
| Lipinski pass rate | — |
| AiZynth synthesis success rate (mean, mode-aware) | — |
| AiZynth n_pockets_scored | 0 |
| AiZynth n_synthesis_route_total | 0 |
| BioLM-Score affinity score (mean across pockets) | — |
| BioLM-Score n_pockets_scored | 0 |
| BioLM-Score status counts | {'disabled': 30} |
| SA mean (backend recorded in search_config) | — |
| QED mean | — |
| Descriptor docking proxy, top-1 mean | — |

| pocket | seed | status | candidates | generated | SMILES | seconds |
|---|---:|---|---:|---:|---|---:|
| test_000 | 42 | no_candidates | 0 | 0 | `` | 4.14 |
| test_000 | 0 | no_candidates | 0 | 0 | `` | 30.34 |
| test_000 | 1234 | no_candidates | 0 | 0 | `` | 4.27 |
| test_001 | 42 | seed_only | 1 | 0 | `COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(C)=O` | 3.49 |
| test_001 | 0 | seed_only | 1 | 0 | `COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(C)=O` | 3.53 |
| test_001 | 1234 | seed_only | 1 | 0 | `COc1cc(OC)c(S(=O)(=O)NCc2ccccc2N2CCCCC2)cc1NC(C)=O` | 3.53 |
| test_002 | 42 | no_candidates | 0 | 0 | `` | 3.13 |
| test_002 | 0 | no_candidates | 0 | 0 | `` | 3.13 |
| test_002 | 1234 | no_candidates | 0 | 0 | `` | 3.13 |
| test_003 | 42 | seed_only | 1 | 0 | `Nc1cc(S(O)(O)O)c(N)c2c1C(=O)c1ccccc1C2=O` | 3.37 |
| test_003 | 0 | seed_only | 1 | 0 | `Nc1cc(S(O)(O)O)c(N)c2c1C(=O)c1ccccc1C2=O` | 3.39 |
| test_003 | 1234 | seed_only | 1 | 0 | `Nc1cc(S(O)(O)O)c(N)c2c1C(=O)c1ccccc1C2=O` | 3.36 |
| test_004 | 42 | seed_only | 1 | 0 | `CC(C)NC[C@H](O)COc1cccc2ccccc12` | 3.32 |
| test_004 | 0 | seed_only | 1 | 0 | `CC(C)NC[C@H](O)COc1cccc2ccccc12` | 3.32 |
| test_004 | 1234 | seed_only | 1 | 0 | `CC(C)NC[C@H](O)COc1cccc2ccccc12` | 3.32 |
| test_005 | 42 | no_candidates | 0 | 0 | `` | 6.96 |
| test_005 | 0 | no_candidates | 0 | 0 | `` | 7.14 |
| test_005 | 1234 | no_candidates | 0 | 0 | `` | 7.06 |
| test_006 | 42 | seed_only | 1 | 0 | `C[C@H](CCC(N)=O)[C@H]1CC[C@H]2[C@@H]3C(=O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C` | 3.60 |
| test_006 | 0 | seed_only | 1 | 0 | `C[C@H](CCC(N)=O)[C@H]1CC[C@H]2[C@@H]3C(=O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C` | 3.40 |
| test_006 | 1234 | seed_only | 1 | 0 | `C[C@H](CCC(N)=O)[C@H]1CC[C@H]2[C@@H]3C(=O)C[C@@H]4C[C@H](O)CC[C@]4(C)[C@H]3CC[C@]12C` | 3.39 |
| test_007 | 42 | no_candidates | 0 | 0 | `` | 2.99 |
| test_007 | 0 | no_candidates | 0 | 0 | `` | 3.01 |
| test_007 | 1234 | no_candidates | 0 | 0 | `` | 3.03 |
| test_008 | 42 | seed_only | 1 | 0 | `Nc1ccc2nccn2c1` | 3.12 |
| test_008 | 0 | seed_only | 1 | 0 | `Nc1ccc2nccn2c1` | 3.10 |
| test_008 | 1234 | seed_only | 1 | 0 | `Nc1ccc2nccn2c1` | 3.11 |
| test_009 | 42 | no_candidates | 0 | 0 | `` | 3.08 |
| test_009 | 0 | no_candidates | 0 | 0 | `` | 3.08 |
| test_009 | 1234 | no_candidates | 0 | 0 | `` | 3.12 |

Published baseline context is maintained in `lambda_vs_sbdd_protocol_aligned.md`.
These proxy values are not compared numerically or statistically to cited docking energies.
