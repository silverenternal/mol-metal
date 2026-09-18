# CrossDocked test-pair search diagnostic — 2026-09-13

All 30 bounded jobs completed without a timeout. No job returned a molecule different from its input reference ligand.
This is a negative generation result; it does not meet the round-12 scientific pilot criteria.

Input artifact SHA-256: `69de8e8970740adda158e1443db2ed055bc22ecfc64b137c0036109b18aa45af`.
Source JSON: `r4_test10_seed3_diagnostic.json`; each pair includes receptor/ligand SHA-256, seed and executed MCTS settings.

| Quantity | Observed |
|---|---:|
| Test pairs | 10 |
| Seeds | [42, 0, 1234] |
| Completed jobs | 30 |
| Sum of worker search times, seconds | 79.44 |
| Median worker time, seconds | 2.69 |
| Jobs returning only the input seed | 18 |
| Jobs returning no candidate | 12 |
| New molecules returned | 0 |

| Test pair | Returned candidates across 3 seeds | New molecules |
|---|---:|---:|
| test_000 | 0 | 0 |
| test_001 | 3 | 0 |
| test_002 | 0 | 0 |
| test_003 | 3 | 0 |
| test_004 | 3 | 0 |
| test_005 | 3 | 0 |
| test_006 | 3 | 0 |
| test_007 | 0 | 0 |
| test_008 | 3 | 0 |
| test_009 | 0 | 0 |

## Protocol and interpretation

Exact command:
```sh
HIP_VISIBLE_DEVICES=0 uv run python molmetal/scripts/r4_c_full_sweep.py --n-pockets 10 --n-simulations 10 --branching-target 60 --seeds 42 0 1234 --job-timeout 30 --output-prefix molmetal/reports/r4_test10_seed3_diagnostic
```

Each search used 12 actual tiles and five click reaction families, with depth 3 and top_k=100. These budgets are diagnostic overrides, not the planned full search budget.
The raw artifact predates the new `seed_only` status: its 18 `ok` records mean search execution returned something, not successful generation. The updated wrapper explicitly marks these as `seed_only` failures for generation coverage.
SA reporting uses RDKit Contrib Ertl. Docking remains a descriptor proxy; PoseBusters and the requested symbolic refit/synthesis oracle were not executed. No significance test against cited Vina energies is warranted.
The current runner seeds from complete reference ligands and uses a generic protease filter, not physical conditioning on each receptor. Next work is to implement/validate a declared click-tile initialization strategy and actual receptor-based docking/validation without treating copied ligands as generated molecules.
QuickVina 2 availability and actual adapter docking are independently established in `docking_adapter_smoke.json` and `quickvina2_binary_identity.md`; this diagnostic failure is not an external binary blocker.
