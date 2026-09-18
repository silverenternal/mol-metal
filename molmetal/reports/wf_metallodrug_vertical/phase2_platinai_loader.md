# Phase 2 — PlatinAI / MetalCytoTox / MetalloDrug Dataset Loaders

**Workflow:** Metallodrug de novo — data + GPU + paper (one ultracode)
**Phase:** 2 of N (data loaders)
**Date:** 2026-09-16
**Status:** SHIPPED — 24/24 tests pass

## Files shipped

| File | Lines | Purpose |
|------|-------|---------|
| `molmetal/molmetal_lam/lam_chem/platinai_dataset.py` | 530 | Three dataset classes (PlatinAI / MetalCytoTox / MetalloDrug union) + helpers |
| `molmetal/molmetal_lam/tests/test_platinai_dataset.py` | 296 | 24 tests across 5 test classes |

## What ships

### 1. `PlatinAIDataset` (primary metallodrug corpus)

Loads `/mnt/storage/data/molmetal/PlatinAI_MBFinder_dataset.xlsx`
(226,918 SMILES) and optionally joins the predicted-activity sheets
`PlatinAI_predicted_{A2780,MCF7}.xlsx` (214,376 rows each).

Pipeline:

```
xlsx → pandas → dropna → canonicalise(RDKit) → heavy-atom-window [8, 38]
     → metal filter (Pt / Ru / Ir / Au / Pd / Rh / Os / Re / labile)
     → dedup by canonical SMILES → (smiles, [a2780, mcf7], metal) records
```

Returns a uniform ``(smiles, (a2780, mcf7), metal)`` tuple.  The
``__init__`` accepts ``split`` (train/val/test, deterministic 80/10/10),
``n_max`` (cap after filtering), ``metal_filter`` (single symbol or
iterable), ``join_a2780`` / ``join_mcf7`` (toggle activity joins).

### 2. `MetalCytoToxDataset` (cytotoxicity-labelled subset)

Loads `/mnt/storage/data/molmetal/MetalCytoToxDB.csv` (26,802 rows).
Returns ``(smiles, ic50_dark_uM, metal, oxidation_state)`` tuples.  IC50
is filtered to ``[1e-3, 1e6] µM`` (sanity window) and NaN values are
dropped.

Public methods:

* ``get_by_metal(metal='Pt')`` — filter to a single metal centre
* ``get_top_k(k=100)`` — most-potent (lowest IC50) k records

### 3. `MetalloDrugDataset` (union class)

Combines PlatinAI + MetalCytoTox (+ optional NCI60 Pt slice).  Returns
unified ``(smiles, metal, activity_dict)`` tuples where the activity
dict has any subset of ``{'a2780', 'mcf7', 'ic50_dark'}``.

Includes a 500-mol MaxMin diversity subset (``get_subset(n=500)``) that
reuses the existing :func:`greedy_maxmin_diversity` helper from
:mod:`molmetal.molmetal_lam.lam_chem.data_diversity` (the same
implementation used by the metallodrug training pipeline).

### 4. Helpers

* ``metals_in_smiles`` / ``primary_metal`` — identify TM centres
  (preference order Pt > Ru > Ir > Au > Pd > Rh > Os > Re)
* ``canonicalise`` — RDKit-canonical SMILES with graceful None on parse
  failure
* ``heavy_atom_count`` — heavy-atom window validation
* ``is_transition_metal_present`` — string / iterable metal filter

## Tests (24 / 24 pass)

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestHelpers` | 7 | canonical SMILES, dedup, heavy-atom range, metal detection, primary metal priority, invalid metal raises |
| `TestPlatinAI` | 7 | loads, Pt filter, no duplicates, train/val/test split sizes, activity labels present, matrix shape, factory |
| `TestMetalCytoTox` | 4 | loads, get_by_metal (Ru + Pt), get_top_k (sorted), factory |
| `TestMetalloDrugUnion` | 4 | union, 500-mol subset, smaller-than-full subset, factory |
| `TestCrossCutting` | 2 | every PlatinAI record is RDKit-canonical; heavy-atom window enforced |

Test runtimes (CPU, single thread):

* `TestHelpers` + `TestMetalCytoTox` + 4 fast `TestPlatinAI` + `TestCrossCutting`: 17 tests in 491 s
* Slow `TestPlatinAI` (split sizes + matrix): 3 tests in 318 s
* `TestMetalloDrugUnion`: 4 tests in 434 s
* **Total: 24 tests, ~21 min wall** (driven by the 226,918-row PlatinAI
  xlsx parse + 5,000-row canonical+heavy-atom filter on each fixture
  load)

## Notes for future work

1. **NCI60 Pt slice is intentionally no-op.**  `GI50.csv` has
   `NSC` + concentration + cell-name columns but **no SMILES column**,
   so we cannot join by canonical SMILES without an external
   `NSC → SMILES` lookup.  Hook is left in `MetalloDrugDataset._load`
   for a future patch.

2. **Activity join uses pre-canonical SMILES as key.**  Both
   `PlatinAI_predicted_A2780.xlsx` and `PlatinAI_predicted_MCF7.xlsx`
   use the raw input SMILES (pre-canonical) as the row identifier, so
   the loader tries both `canon` and `raw_str` for the lookup.

3. **`split` + `n_max` interaction.**  `n_max` is applied *before* the
   80/10/10 split so a `n_max=1000` user gets exactly 800/100/100.
   This was the source of one of the early test failures — now fixed
   and verified by `test_platinai_split_sizes`.

4. **No file conflicts.**  The new module lives in
   `molmetal/molmetal_lam/lam_chem/` (alongside the existing
   `data_diversity.py`, `closure.py`, `pt_click_compat.py`) and does
   not overlap with any existing class or symbol.  `molmetal.data` was
   not modified.

5. **No new dependencies.**  Uses only `rdkit`, `pandas`, and the
   stdlib.  `numpy` is imported lazily inside `get_activity_matrix`.

## Integration plan

This loader is the data backbone for the next two phases:

* **Phase 3**: PlatinAI oracle column for §4.7 — wire
  ``load_platinai(n_max=500)`` into the existing
  ``r4_lambda_only_run.py --platinai-oracle`` path.
* **Phase 4**: Round-13 100×3 sweep evaluation — replace the legacy
  `metallo_drugs_500_train.csv` with a freshly-diverse 500-mol subset
  pulled via ``MetalloDrugDataset.get_subset(n=500)``.

Both integrations should be 1-2 hours of work each, since the return
shapes (``(smiles, [a2780, mcf7], metal)`` etc.) match the existing
adapter contracts.

## Verdict

**READY.** 24/24 tests green.  No regressions.  Loader surface is
complete, documented, and consistent with the project's existing
``data_diversity.py`` / ``tmqm.py`` / ``metalcytotox.py`` patterns.