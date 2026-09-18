# Measured click-product physical evaluation

## Goal
Validate generated products on exact CrossDocked pairs with saved docking poses and strict PoseBusters checks.

## Outcome
3 jobs; 7 generated products; 7 docked; 7 pass PoseBusters.
Physical job status counts: {'completed': 3}
Products beating a redocked reference: 7; triple threshold successes: 7.
Reference PoseBusters statuses: {'passed': 3}; reference comparisons unavailable: 0.
Reference-PB-passing subset: {'n_jobs': 3, 'n_generated': 7, 'n_docked': 7, 'n_beats_redocked_reference': 7, 'n_triple_threshold_redocked_reference': 7}.

| pocket | seed | physical status | docked/generated | Vina mean | PB pass/generated | reference PB |
|---|---:|---|---:|---:|---:|---|
| test_005_modeled_ASP_B101 | 42 | completed | 3/3 | -4.280 | 3/3 | passed |
| test_005_modeled_ASP_B101 | 0 | completed | 3/3 | -3.845 | 3/3 | passed |
| test_005_modeled_ASP_B101 | 1234 | completed | 1/1 | -4.510 | 1/1 | passed |

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
Input: `molmetal/reports/r4_click_physical_test005_modeled_vina_seed3.json`; SHA256 `46483fb6bb148a1958b31a34261c6d39da856c3c53a95ac3427aac9e7e8515b8`.
The JSON includes exact generation/docking budgets, source/config hashes, per-job pose paths and GPU diversity backend.
