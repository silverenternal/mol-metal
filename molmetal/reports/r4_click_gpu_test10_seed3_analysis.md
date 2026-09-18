# Measured click-product physical evaluation

## Goal
Validate generated products on exact CrossDocked pairs with saved docking poses and strict PoseBusters checks.

## Outcome
30 jobs; 70 generated products; 63 docked; 63 pass PoseBusters.
Unique generated structures across jobs: 4. Candidate instances repeated across pockets/seeds remain separate docking observations.
Physical job status counts: {'completed': 27, 'receptor_preparation_failed': 3}
Products beating a redocked reference: 9; triple threshold successes: 9.
Reference PoseBusters statuses: {'failed': 9, 'passed': 18, 'unavailable': 3}; reference comparisons unavailable: 7.
Reference-PB-passing subset: {'n_jobs': 18, 'n_generated': 42, 'n_docked': 42, 'n_beats_redocked_reference': 8, 'n_triple_threshold_redocked_reference': 8}.
GPU execution evidence: {'records': 90, 'verified': 90, 'kernel_trace_verified': 90, 'devices': ['gfx1101'], 'timing_note': 'GPU event timestamps invalid on this driver; no speedup inferred from these experiments.'}.

| pocket | seed | physical status | docked/generated | Vina mean | PB pass/generated | reference PB |
|---|---:|---|---:|---:|---:|---|
| test_000 | 42 | completed | 3/3 | -5.433 | 3/3 | failed |
| test_000 | 0 | completed | 3/3 | -5.333 | 3/3 | failed |
| test_000 | 1234 | completed | 1/1 | -6.700 | 1/1 | failed |
| test_001 | 42 | completed | 3/3 | -4.900 | 3/3 | passed |
| test_001 | 0 | completed | 3/3 | -4.567 | 3/3 | passed |
| test_001 | 1234 | completed | 1/1 | -5.200 | 1/1 | passed |
| test_002 | 42 | completed | 3/3 | -4.333 | 3/3 | passed |
| test_002 | 0 | completed | 3/3 | -4.367 | 3/3 | passed |
| test_002 | 1234 | completed | 1/1 | -5.200 | 1/1 | passed |
| test_003 | 42 | completed | 3/3 | -4.967 | 3/3 | failed |
| test_003 | 0 | completed | 3/3 | -4.900 | 3/3 | failed |
| test_003 | 1234 | completed | 1/1 | -5.400 | 1/1 | failed |
| test_004 | 42 | completed | 3/3 | -6.000 | 3/3 | passed |
| test_004 | 0 | completed | 3/3 | -5.733 | 3/3 | passed |
| test_004 | 1234 | completed | 1/1 | -6.800 | 1/1 | passed |
| test_005 | 42 | receptor_preparation_failed | 0/3 | unavailable | 0/3 | unavailable |
| test_005 | 0 | receptor_preparation_failed | 0/3 | unavailable | 0/3 | unavailable |
| test_005 | 1234 | receptor_preparation_failed | 0/1 | unavailable | 0/1 | unavailable |
| test_006 | 42 | completed | 3/3 | -5.267 | 3/3 | passed |
| test_006 | 0 | completed | 3/3 | -5.267 | 3/3 | passed |
| test_006 | 1234 | completed | 1/1 | -6.100 | 1/1 | passed |
| test_007 | 42 | completed | 3/3 | -5.000 | 3/3 | failed |
| test_007 | 0 | completed | 3/3 | -4.767 | 3/3 | failed |
| test_007 | 1234 | completed | 1/1 | -5.200 | 1/1 | failed |
| test_008 | 42 | completed | 3/3 | -5.433 | 3/3 | passed |
| test_008 | 0 | completed | 3/3 | -5.500 | 3/3 | passed |
| test_008 | 1234 | completed | 1/1 | -6.100 | 1/1 | passed |
| test_009 | 42 | completed | 3/3 | -4.500 | 3/3 | passed |
| test_009 | 0 | completed | 3/3 | -4.333 | 3/3 | passed |
| test_009 | 1234 | completed | 1/1 | -5.100 | 1/1 | passed |

## Caveats
pocket-independent click generation, then per-pocket docking; no pocket-conditioned search reward.
bounded physical integration experiment, not SOTA protocol completion.
Training fingerprints not supplied; new-to-seed does not prove training novelty.
Native Vina/QuickVina are CPU engines. QuickVina2-GPU uses OpenCL grid/search when selected; preparation, final refinement and PoseBusters retain CPU stages. Pairwise fingerprint similarity uses the recorded torch device.
Unprepared receptors and missing poses remain in all-generated denominators. Descriptor/Vina means include only measured finite values and report n in JSON.
Reference thresholds use redocked ligand scores, not co-crystal measured affinity. No significance test against cite-only means is performed.
Some redocked reference poses can fail PB; the report retains their raw score comparisons and separately reports the reference-PB-passing subset. Unavailable comparisons are not observed failures.
Only the common pocket subset with measured Vina for every requested seed; coverage reported separately.

## Reproduce
Input: `molmetal/reports/r4_click_gpu_test10_seed3_trace_recovered.json`; SHA256 `523cef61ac6d99ea34542480b2a833bc313afa71006141ba889592b7d87e4b68`.
The JSON includes exact generation/docking budgets, source/config hashes, per-job pose paths and GPU diversity backend.
