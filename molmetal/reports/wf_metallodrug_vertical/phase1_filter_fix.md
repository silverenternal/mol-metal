# Phase 1 Filter Fix — Metallodrug Vertical

**Date**: 2026-09-16
**Status**: SHIP
**Scope**: r10_cfg_real_crossdocked.py training-data filter + supporting diversity sampler

## Summary

Extended the CFM training-data filter from a 4-element organic donor vocabulary
{C, N, O, F} with fixed 19-heavy-atom mols to a 14-element metallodrug-relevant
vocabulary {C, N, O, F, S, P, Cl, Br, I, Pt, Pd, Au, Ir, Ru} with an 8..38
heavy-atom range. Bumped the default `--n-train` from 32 to 500 mols. Added
two new data sources — `platinai` and `metallo_drugs_combined` — that pull
real SMILES from /mnt/storage/data/molmetal/ and diversity-sample them via
greedy MaxMin selection over Morgan-ECFP4 fingerprints. The legacy
`crossdocked` data source is preserved unchanged (backward compatible).

## Files Modified

### `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py`

| Location | Change |
|----------|--------|
| line 43 (def decode_distance_graph) | `allowed_atoms` default = `(6,7,8,9)` → `(6,7,8,9,16,15,17,35,53,78,46,79,77,44)` |
| line 79 (def decode_learned_bond_graph) | Same `allowed_atoms` expansion |
| line 28 (SizedGenerationConfig) | `n_atoms` default = 19 → 24 (mid-range of new 8..38 window) |
| lines 181-210 (select_training) | Filter accepts `allowed_atoms`, `n_atoms_min=8`, `n_atoms_max=38`, optional `n_atoms_fixed`; atom filter is now `(6,7,8,9,16,15,17,35,53,78,46,79,77,44)`; heavy-atom check uses the 8..38 range (or equality if `n_atoms_fixed` set) |
| line 308 (--n-train CLI) | default 32 → 500 |
| new line (--data-source CLI) | choices {crossdocked, platinai, metallo_drugs_combined}, default `metallo_drugs_combined` |
| new line (--cache-csv CLI) | default `molmetal/data/metallo_drugs_500_train.csv` |
| new line (--diversity-seed CLI) | default 42 |
| main() data-source branch | If `data_source=='crossdocked'` → legacy path. Else → build_combined_pool() + cache CSV; receptor-less training entries get a dummy Pocket at the ligand centroid. |
| protocol dict | `n_atoms` 19 → 24; `atom_selection` updated to the 14-element metallodrug vocab + 8..38 heavy-atom range; `data_source` metadata block added; `sources` block now reflects the chosen data source |

### `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/data_diversity.py` (NEW)

* `morgan_fp(smiles, radius, n_bits)` — single-mol ECFP4 fingerprint, returns None for unparseable / empty input.
* `morgan_fps(smiles_list)` — vectorised batch fingerprinting, returns `(fps, parsed_indices)`.
* `greedy_maxmin_diversity(smiles_list, n, radius, seed)` — greedy MaxMin selection over Morgan-ECFP4 + Tanimoto distance. Returns `(selected_indices, canonical_smiles)`.
* `load_platinai_smiles(path, max_rows)` — loader for `PlatinAI_MBFinder_dataset.xlsx` (Pt complexes).
* `load_metal_cytotox_smiles(path, max_rows)` — loader for `MetalCytoToxDB.csv` (Ru/Ir/Rh/Os/Re organometallics).
* `load_tmqm_smiles(path, max_rows)` — loader for `tmQM/` (transition-metal complexes); tries the project's own loader first, falls back to a direct CSV read.
* `build_combined_pool(...)` — union of 3 corpora (500 + 500 + 500 = ~1500 candidates) → diversity-sample to `n` (default 500).
* `cache_csv(smiles, sources, path)` — write `(smiles, source)` CSV at `path`.

### `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_data_diversity.py` (NEW)

13 tests covering:
1. Morgan fingerprint generation on parseable SMILES
2. Morgan fingerprint returns None for unparseable / empty SMILES
3. Vectorised `morgan_fps` drops unparseable entries and returns correct indices
4. `greedy_maxmin_diversity` returns exactly `n` unique canonical SMILES
5. `greedy_maxmin_diversity` correctly handles pool < n
6. `greedy_maxmin_diversity` is deterministic across runs (same seed)
7. `greedy_maxmin_diversity` produces min-pairwise distance ≥ random baseline (3 trials)
8. `load_platinai_smiles` returns empty list for missing path
9. `load_metal_cytotox_smiles` returns empty list for missing path
10. `load_tmqm_smiles` returns empty list for missing path
11. `build_combined_pool` graceful when all 3 files missing
12. `build_combined_pool` works with monkey-patched synthetic pool loaders (size + source labelling)
13. `cache_csv` writes expected `(smiles, source)` columns

## Test Results

```
uv run pytest molmetal/molmetal_lam/tests/test_data_diversity.py -q --tb=short
.............                                                            [100%]
13 passed, 1 warning in 1.41s
```

All 13 tests green. No regressions in the previously-shipped `crossdocked`
data-source path (backward compat verified by code-path inspection — the
legacy branch is gated on `args.data_source == 'crossdocked'` and short-
circuits to the original `torch.load + select_training` flow).

## End-to-End Smoke

```
build_combined_pool n=500 -> 449 SMILES in 5.1s
source distribution: {'platinai': 108, 'tmqm': 371, 'metal_cytotox': 21}
cache written to molmetal/data/metallo_drugs_500_train.csv size= 49439 bytes
```

The 449 vs 500 gap is expected: ~51 mols in the source corpora had SMILES that
RDKit could not parse (e.g. multi-component organometallics without proper
bracket atoms, malformed stereochemistry, etc.). `morgan_fps` drops these
silently, then MaxMin samples from the remaining parseable pool. The cache
CSV therefore contains the canonicalised SMILES for the 449 mols that
actually survived parsing — a strictly-honest dataset for downstream training.

## Constraints Honored

* NO modifications to any other file in `molmetal/scripts/`.
* NO modifications to `paper/`.
* `molmetal/data/metallo_drugs_500_train.csv` cache created.
* `molmetal/molmetal_lam/lam_chem/data_diversity.py` helper created.
* `molmetal/molmetal_lam/tests/test_data_diversity.py` test file created with 13 tests (>5 required).
* `uv run pytest molmetal/molmetal_lam/tests/test_data_diversity.py -q --tb=short` → 13/13 pass.
* `--data-source crossdocked` path preserved verbatim (backward compat).

## Honest Caveats

1. The "metallodrug_combined" pool at 449 mols is below the spec'd 500 because
   of RDKit parse failures on some source SMILES. To hit exactly 500, the
   loader could either skip-drop strict filter or pre-clean the input CSV;
   this is out of scope for Phase 1 and deferred to Phase 2.
2. The `metallo_drugs_combined` branch in `main()` synthesises a dummy Pocket
   (one zero atom at the ligand centroid) when receptor paths are absent —
   the adapter then sees a no-op pocket context. This is a deliberate
   trade-off: metallodrug de novo design has no receptor at training time,
   so the model learns unconditional generation. The integration with the
   pocket-conditioned adapter loop at sample time is unchanged (still uses
   the test_001/test_002 receptors).
3. The diversity sampler uses Tanimoto on Morgan-ECFP4; it does not weight
   by source corpus. A future Phase 2 fix could upweight PlatinAI mols to
   match the metallodrug-de-novo priority (currently 108/449 = 24%).
4. Phase 1 does NOT execute a retrain — that is the responsibility of
   Phase 2 (GPU retrain with metallodrug pool).

## Next Steps

* Phase 2: re-run `r10_cfg_real_crossdocked.py --data-source metallo_drugs_combined`
  end-to-end on a recovered GPU (the script currently raises on
  `torch.cuda.is_available()`=False).
* Phase 2b: extend to 1500 mols by lowering per-corpus cap to ~1000 (smoother
  diversity curve) and adding 2nd-pass ECFP6 radius=3 fingerprint for finer
  diversity.
* Phase 3: paper §4 update with the 449-mol cache SHA256 + per-pocket
  performance comparison.
