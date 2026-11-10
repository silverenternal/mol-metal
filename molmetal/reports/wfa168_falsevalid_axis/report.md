# Wf-A168 -- OUR side of the false-valid / speck axis (c7 pool, POST-PROCESSED arm)

**0 GPU.** CPU-only, RDKit-only. No checkpoint, no decode, no training.

Source pool: `molmetal/reports/cfm_c7_fullgen_2026_10_20/generated_smiles.csv`
(12669 data rows, 12668 parsed, 1 unparseable).

Everything below is MEASURED by re-running the pipeline on the RAW pool.
`metrics/by_metric/cfm_fragmentation_postproc_arm.json` is a CROSS-CHECK, never a source.

## 1. The speck decomposition, re-derived from the pool

| bucket | measured | published | delta |
|---|---|---|---|
| survivors <= 2 heavy atoms | **2663** | 2663 | 0 |
| survivors 3-10 heavy atoms | **2** | 2 | 0 |
| survivors >= 11 heavy atoms | **10003** | 10003 | 0 |
| sum | **12668** | 12668 | 0 |

Denominator: 12668 parsed rows.

The published split REPRODUCES EXACTLY (delta 0 on every bucket).
Survivor heavy-atom mean 34.39161667192927, median 38.0, min 1, max 89 -- all equal to the published values.

Field-by-field cross-check against `metrics/by_metric/cfm_fragmentation_postproc_arm.json`: **17 / 17** fields agree (16 byte-identical).

### What a speck actually IS (measured, not assumed)

* Every speck is **exactly 1 heavy atom** (histogram: {1: 2663}).
* Element signatures: `{'C': 2602, 'Si': 61}` -- lone carbon and lone silicon only.
* **0** of the 2663 specks contain a metal.
* **0** of the 2663 speck rows were a single fragment in the RAW SMILES -- every one of them came out of a 21-to-89-fragment raw molecule.

## 1b. The false-valid rate is NOT stationary -- it is a time average

Across the 18 epochs of the c7 pool the post-processed speck rate falls from **0.5639** (epoch 0) to **0.0085** (epoch 17), against a pooled **0.2102**.

So the pooled 21.02 % is an average over a generator that was still
improving, NOT a property of the final model. The per-epoch table is in
the artifact under `per_epoch`; the pooled rate must never be quoted
without it.

## 2. The false-valid rate

**2663 / 12668 = 0.210215 (21.0215 %)**

That is the share of the 100 % 'connected' arm that is still a fragment.

## 3. The ligand-only analogue on the POST-PROCESSED arm

Strip every metal atom from the SURVIVOR and recount:

* survivors that become a <= 2-heavy speck: **2663 / 12668** = 21.0215 %
* **NEW** specks created by removing the metal: **0**
* survivors whose largest ligand fragment is <= 2 heavy: **2663**
* survivors that SPLIT into >1 fragment when the metal is removed: **7040**

The metal-free survivors: 3269 of 12668, of which 2663 are specks and 606 are drug-sized. The other 9399 survivors still carry the metal, and **none** of them is a speck.

So the false-valid arm is NOT a metal-coordination artifact: removing the metal
creates ZERO new specks. The specks are lone carbon and silicon atoms.

## 4. What the post-processing actually bought

* raw single-fragment count: **1078 / 12668** = 0.085096 (RAW, no post-processing)
* post-processed connected count: **12668 / 12668** = 1.000000
* heavy-atom mass retention: kept 435673 / raw 561976 = 0.775252 -> **discarded 0.224748** (22.47 %)
* per-row retention mean 0.7395022201805989, median 0.926829268292683

The measured discard fraction equals the published `mass_retention.fraction_of_raw_mass_discarded` to 15 significant digits (the two differ by 5.55e-17 in the last float ULP -- see section 6).

* UFF step: **12668 / 12668** raised (['ValueError: Bad Conformer Id']), so `uff_ok_rate = 0.0` -- the arm measures steps 1-2 only.

## 5. What a reader would conclude, and what must be printed beside it

A reader of 'post-processed connected rate = 100 %' alone would conclude that
every one of the 12,668 post-processed molecules is a single, chemically meaningful
molecule; in fact **2663 of them (21.02 %) are a SINGLE
heavy atom** (2,602 lone carbons, 61 lone silicons) and the post-processing
discarded 22.47 % of the raw heavy-atom mass to get there --
so the FALSE-VALID (speck) rate, the survivor-size histogram and the
discarded-mass fraction must be printed beside the 100 %, and the 100 % must
never be quoted alone.

And because the pooled speck rate is a time average over an improving
generator (epoch 0: 0.5639 -> epoch 17: 0.0085), the pooled figure must
be printed beside the PER-EPOCH axis as well -- otherwise a reader reads a
snapshot of an early epoch as a property of the final model.

## 6. Two findings about the published pipeline note

* `metrics/by_metric/cfm_fragmentation_postproc_arm.json` describes step 1 as
  "LargestFragmentChooser (rdMolStandardize, fallback: max heavy atoms)", which
  reads as if the two rules are interchangeable. They are not: RDKit's chooser
  maximises TOTAL atom count INCLUDING hydrogens (ties to the last fragment),
  not the heavy-atom count. An independent re-derivation of that rule agrees
  with RDKit on **12668 / 12668** rows.
* The two rules pick a DIFFERENT fragment on **261** of 12668 rows, but the survivor's heavy-atom SIZE differs on only **2**, so the survivor-size histogram -- and therefore every headline in this artifact --
  is invariant to the choice. The published numbers match the CHOOSER.
* The published artifact's `fraction_of_raw_mass_discarded` is computed as
  `1 - kept/raw`; this script computes `(raw-kept)/raw`. The two differ by
  5.55e-17 (2 ULP) -- a float-expression difference, not a measurement
  difference. Both are in the artifact under `mass_retention`.

## 7. Comparability (carried forward, not contradicted)

* The frontier band is POST-PROCESSED, and so is this arm -- the POST-PROCESSING
  break of HONEST_NEGATIVE_CATALOG.md 6.264 therefore does NOT apply between this
  number and the frontier's.
* The other three breaks DO still apply: different object, different denominator
  (12,668 rows of one post-hoc decode vs 10,000-15,000 sampled molecules),
  different chemistry (a metal complex, not an all-organic molecule).
* The pool carries implicit hydrogens; the frontier datasets carry explicit ones.
* This is ONE pool, ONE post-hoc decode -- not a rate over the generator's output
  distribution.

## 8. Files

* script: `molmetal/scripts/wfa168_falsevalid_axis.py`
* artifact: `molmetal/reports/wfa168_falsevalid_axis/falsevalid_axis.json`
* test: `molmetal/tests/test_wfa168_falsevalid_axis.py`
