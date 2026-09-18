# Measured GPU reward control

2 pockets / 6 paired seed jobs.
Actual oracle attempt counts equal in every pair: True.
Generated sets equal in every pair: True.
Top-1 changes: 0; mean top-1 score delta (feedback minus zero): 0.000000 kcal/mol.
Mean all-pose score delta: 0.000000 kcal/mol.

| Pocket | Seed | Zero top-1 | Feedback top-1 | Delta | Same generated set |
|---|---:|---:|---:|---:|---|
| test_001 | 42 | -6.2 | -6.2 | 0.000 | True |
| test_001 | 0 | -5.3 | -5.3 | 0.000 | True |
| test_001 | 1234 | -5.2 | -5.2 | 0.000 | True |
| test_002 | 42 | -5.2 | -5.2 | 0.000 | True |
| test_002 | 0 | -4.9 | -4.9 | 0.000 | True |
| test_002 | 1234 | -5.2 | -5.2 | 0.000 | True |

## Limits
- Only two pockets and three nested seeds; no independent-six-sample significance claim or generalization conclusion.
- Weight zero still measures the oracle: same configured limits, chemistry, seeds and post-search protocol. Actual attempts are checked separately.
- Changed top-1 selection is distinct from generation diversity; unchanged generated sets do not prove expanded chemical exploration.
- Evaluation repeats docking separately from the in-search cached score, on the same engine; no independent physical affinity confirmation.
- Wall time includes evaluation and reflects concurrent host activity; no hardware speedup benchmark.
- Shared references/environment are listed in the snapshot manifest; this experiment does not retrain or load a learned generator.
