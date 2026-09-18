# Lambda paired-input search diagnostic

Requested pockets: 10; completed pocket/seed jobs: 30.
Jobs returning generated candidates: 30; failed, empty or seed-only: 0.
Search budget: 4 simulations, depth 1.

Initialization strategies are recorded per job; reference initialization uses the paired ligand chemistry.
Click-tile initialization uses the declared tile library. Reference coordinates define physical docking boxes.
Search docking proxies have no kcal/mol interpretation. Actual docking/PoseBusters, when enabled, are in the physical records.
Reference thresholds are redocked-ligand scores, not measured co-crystal affinity. This is not a SOTA comparison.
Lipinski pass rate is a descriptor statistic, not the published SBDD success rate.
Requested configuration and executed search settings are recorded separately in JSON.

| Metric | Value |
|---|---:|
| Candidates | 70 |
| Newly generated candidates | 70 |
| Returned input-seed candidates | 0 |
| Physical jobs completed | 27 |
| Generated products physically docked | 63 |
| Docked products passing PoseBusters | 63 |
| Lipinski pass rate | 1.000 |
| SA mean (backend recorded in search_config) | 2.564 |
| QED mean | 0.678 |
| Descriptor docking proxy, top-1 mean | -10.189 |

| pocket | seed | status | candidates | generated | SMILES | seconds |
|---|---:|---|---:|---:|---|---:|
| test_000 | 42 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 15.24 |
| test_000 | 0 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 15.08 |
| test_000 | 1234 | ok | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 12.42 |
| test_001 | 42 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 11.95 |
| test_001 | 0 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 12.30 |
| test_001 | 1234 | ok | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 9.29 |
| test_002 | 42 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 12.78 |
| test_002 | 0 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 13.04 |
| test_002 | 1234 | ok | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 10.23 |
| test_003 | 42 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 10.39 |
| test_003 | 0 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 10.61 |
| test_003 | 1234 | ok | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 7.93 |
| test_004 | 42 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 12.23 |
| test_004 | 0 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 12.54 |
| test_004 | 1234 | ok | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 9.57 |
| test_005 | 42 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 4.89 |
| test_005 | 0 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 4.88 |
| test_005 | 1234 | ok | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 4.85 |
| test_006 | 42 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 11.54 |
| test_006 | 0 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 11.66 |
| test_006 | 1234 | ok | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 9.23 |
| test_007 | 42 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 16.81 |
| test_007 | 0 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 16.23 |
| test_007 | 1234 | ok | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 14.06 |
| test_008 | 42 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 9.83 |
| test_008 | 0 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 9.92 |
| test_008 | 1234 | ok | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 7.44 |
| test_009 | 42 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 14.35 |
| test_009 | 0 | ok | 3 | 3 | `OCCOCCn1nncc1Cc1ccccc1` | 14.09 |
| test_009 | 1234 | ok | 1 | 1 | `NCc1cnnn1Cc1ccccc1` | 11.71 |

Published baseline context is maintained in `lambda_vs_sbdd_protocol_aligned.md`.
These proxy values are not compared numerically or statistically to cited docking energies.
