# Lambda Round 3 — L-3 Fragment Library Expansion Report

## Goal

Replace the 12-tile hardcoded Phase-0 click-chemistry library (which
caps MCTS branching at |rules| × |tiles| = 5 × 12 = 60) with a
SMARTS-diverse 200+ tile pool drawn from a ChEMBL reactive-handle /
ZINC click-chemistry subset (Sterling & Irwin, 2015).

## Files delivered

| Path | Purpose |
| --- | --- |
| `molmetal/molmetal_lam/tile_lib/fragment_pool.py` | New: hardcoded SMARTS-diverse pool + validation hook |
| `molmetal/molmetal_lam/tile_lib/library.py` | Updated: `build_tile_library(include_fragments=True)` + `FRAGMENT_LIBRARY_200_TILES` |
| `molmetal/molmetal_lam/tests/test_fragment_library.py` | New: 7 unit tests (size, backward-compat, validity, category coverage, metrics) |
| `molmetal/scripts/measure_fragment_library.py` | New: per-category embed-failure report runner |

The original 12-tile library (`STANDARD_12_TILES`, `STANDARD_14_TILES`)
remains intact and backward-compatible — `include_fragments=False` is
the default and returns the original 12 tiles.

## Tile count by category (post-validation)

| Category | Input SMILES | Valid tiles | Embed-fail | MW-fail | logP-fail | Fail % |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| Azide   | 51 | **50** | 1 | 0 | 0 | 2.0% |
| Alkyne  | 53 | **52** | 1 | 0 | 0 | 1.9% |
| Diene   | 52 | **52** | 0 | 0 | 0 | 0.0% |
| Thiol   | 52 | **50** | 2 | 0 | 0 | 3.8% |
| **TOTAL** | **208** | **204** | 4 | 0 | 0 | **1.9%** |

All four reactive-handle families (azide / alkyne / diene / thiol)
are populated. The pool exceeds the 200-tile target.

## Embed-failure rate per category

| Category | ETKDGv3 embed failures | Failure rate |
| --- | ---:| ---:|
| Azide   | 1 / 51 | 2.0% |
| Alkyne  | 1 / 53 | 1.9% |
| Diene   | 0 / 52 | 0.0% |
| Thiol   | 2 / 52 | 3.8% |
| **TOTAL** | **4 / 208** | **1.9%** |

No tile was rejected for MW or logP — all four failures were ETKDGv3
embed failures (RDKit failed to find a 3D conformer within the
default ETKDGv3 timeout). Failures were distributed across categories
with no systematic bias.

## Library validation summary

* All 204 emitted tiles pass:
  * `Chem.MolFromSmiles` parse,
  * `AllChem.EmbedMolecule` with ETKDGv3 (`randomSeed=0xC11C`) returns 0,
  * `40 <= MW <= 300` g/mol,
  * `-2 <= logP <= 5`,
  * `coords` populated (Tensor of shape `(N_atoms, 3)`).

* Branching factor growth: |rules| × |tiles| = 5 × 14 = 70 (today) →
  5 × 204 = **1020** (Phase-1).

## Test results

`pytest molmetal/tests/ molmetal/molmetal_lam/tests/ -v` (excluding the
pre-existing `test_clone_integration_adapters.py` import error and
the unrelated pre-existing `test_baselines.py` / `test_layer_metrics_*`
/ `test_proof_search_strengthened.py` / `test_3d_embed.py` failures
that are not caused by L-3):

* **All 7 new fragment-library tests pass** (`test_fragment_library.py`).
* All 9 existing `test_tile_lib.py` tests still pass.
* **molmetal_lam/test_*: 156 passed** (excluding pre-existing broken
  `test_baselines.py`).

Total new tests: 7 (all PASS).
Backward-compat: `STANDARD_12_TILES()` still returns 12 tiles (verified).

## Caveats / next steps

* L-3 did **not** wire the 200-tile library into the MCTS proof search —
  that requires the L-1 binding oracle to be in place so the additional
  branching can be guided.
* Fragment pool SMILES are curated from public knowledge of ChEMBL_29
  reactive-handle patterns + ZINC 15 click-chemistry subset; no network
  access is performed at runtime (all SMILES hardcoded as Python literals).
* `n_sims` should be raised from 1000 → 5000 when this pool is wired in
  (per `audit_lambda_upper_bound.md` Gap 2).
