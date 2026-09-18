# Measured click-product physical evaluation

## Goal
Validate generated products on exact CrossDocked pairs with saved docking poses and strict PoseBusters checks.

## Outcome
30 jobs; 70 generated products; 63 docked; 63 pass PoseBusters.
Physical job status counts: {'completed': 27, 'receptor_preparation_failed': 3}
Products beating a redocked reference: 4; triple threshold successes: 4.
Reference PoseBusters statuses: {'failed': 9, 'passed': 18, 'unavailable': 3}; reference comparisons unavailable: 7.
Reference-PB-passing subset: {'n_jobs': 18, 'n_generated': 42, 'n_docked': 42, 'n_beats_redocked_reference': 4, 'n_triple_threshold_redocked_reference': 4}.

| pocket | seed | physical status | docked/generated | Vina mean | PB pass/generated | reference PB |
|---|---:|---|---:|---:|---:|---|
| test_000 | 42 | completed | 3/3 | -5.533 | 3/3 | failed |
| test_000 | 0 | completed | 3/3 | -4.500 | 3/3 | failed |
| test_000 | 1234 | completed | 1/1 | -5.300 | 1/1 | failed |
| test_001 | 42 | completed | 3/3 | -5.067 | 3/3 | passed |
| test_001 | 0 | completed | 3/3 | -4.900 | 3/3 | passed |
| test_001 | 1234 | completed | 1/1 | -5.300 | 1/1 | passed |
| test_002 | 42 | completed | 3/3 | -4.633 | 3/3 | passed |
| test_002 | 0 | completed | 3/3 | -4.400 | 3/3 | passed |
| test_002 | 1234 | completed | 1/1 | -5.400 | 1/1 | passed |
| test_003 | 42 | completed | 3/3 | -4.733 | 3/3 | failed |
| test_003 | 0 | completed | 3/3 | -4.767 | 3/3 | failed |
| test_003 | 1234 | completed | 1/1 | -6.100 | 1/1 | failed |
| test_004 | 42 | completed | 3/3 | -6.167 | 3/3 | passed |
| test_004 | 0 | completed | 3/3 | -6.233 | 3/3 | passed |
| test_004 | 1234 | completed | 1/1 | -7.000 | 1/1 | passed |
| test_005 | 42 | receptor_preparation_failed | 0/3 | unavailable | 0/3 | unavailable |
| test_005 | 0 | receptor_preparation_failed | 0/3 | unavailable | 0/3 | unavailable |
| test_005 | 1234 | receptor_preparation_failed | 0/1 | unavailable | 0/1 | unavailable |
| test_006 | 42 | completed | 3/3 | -5.167 | 3/3 | passed |
| test_006 | 0 | completed | 3/3 | -5.667 | 3/3 | passed |
| test_006 | 1234 | completed | 1/1 | -6.300 | 1/1 | passed |
| test_007 | 42 | completed | 3/3 | -5.300 | 3/3 | failed |
| test_007 | 0 | completed | 3/3 | -5.133 | 3/3 | failed |
| test_007 | 1234 | completed | 1/1 | -5.800 | 1/1 | failed |
| test_008 | 42 | completed | 3/3 | -5.567 | 3/3 | passed |
| test_008 | 0 | completed | 3/3 | -5.733 | 3/3 | passed |
| test_008 | 1234 | completed | 1/1 | -6.300 | 1/1 | passed |
| test_009 | 42 | completed | 3/3 | -4.467 | 3/3 | passed |
| test_009 | 0 | completed | 3/3 | -4.333 | 3/3 | passed |
| test_009 | 1234 | completed | 1/1 | -5.500 | 1/1 | passed |

## Caveats
pocket-independent click generation, then per-pocket docking; no pocket-conditioned search reward.
bounded physical integration experiment, not SOTA protocol completion.
Training fingerprints not supplied; new-to-seed does not prove training novelty.
QuickVina/native Vina and PoseBusters retain their native CPU chemistry implementation. Pairwise fingerprint similarity executes on the recorded torch device.
Unprepared receptors and missing poses remain in all-generated denominators. Descriptor/Vina means include only measured finite values and report n in JSON.
Reference thresholds use redocked ligand scores, not co-crystal measured affinity. No significance test against cite-only means is performed.
Some redocked reference poses can fail PB; the report retains their raw score comparisons and separately reports the reference-PB-passing subset. Unavailable comparisons are not observed failures.
Only the common pocket subset with measured Vina for every requested seed; coverage reported separately.

## Reproduce
Input: `molmetal/reports/r4_click_physical_test10_seed3_v2.json`; SHA256 `cdcda44dc4a80363c58ade655ef7f67a5921cb00f58a4457400e4e33dc218aed`.
The JSON includes exact generation/docking budgets, source/config hashes, per-job pose paths and GPU diversity backend.
