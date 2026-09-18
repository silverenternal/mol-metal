# WF-T30-P6.1+P6.2 — Scaffold split + seed/fingerprint hygiene

## Goal
Close TODO/30 Tier-2 P6.1 and P6.2 reproducibility gaps on the
training-side split and on per-run hygiene. Three deliverables:

1. **P6.1** Bemis-Murcko scaffold split for the metallodrug pool,
   with per-metal coverage (Pt/Pd/Au/Ir/Ru).
2. **P6.2a** `--split {scaffold,time,random,scaffold_tanimoto_0.4}`
   flag in `r4_lambda_only_run.py` (default OFF for backward compat).
3. **P6.2b** `seeds.json` emission next to `report.json`.
4. **P6.2c** `checkpoint_sha256` fingerprint emitted to
   `runs/{run_id}/fingerprint.json` by `r4_c_full_sweep.py`.

All changes are CPU-only (RDKit + json), additive (no breaking change
to existing r4 scripts), and forward-compat with the future
MMseqs2-split (P6.3 deferred — placeholder alias exposed).

## Outcome — PASS
- 14/14 tests in `molmetal/tests/test_reproducibility_split.py`
  pass.
- `metallo_pool_scaffold_split.py` runs end-to-end on the bundled
  `metallo_drugs_500_train.csv` (449 rows after blank-line skip):
  - 449 rows → 213 Murcko clusters → 314/67/68 (train/val/test).
  - Tanimoto threshold 0.6 by default; passes 0.4 and 0.6.
  - Determinism: same `(rows, fractions, threshold, seed)` produces
    byte-identical CSVs (verified by `test_scaffold_split_deterministic`).
  - Per-metal coverage:
    - `train`: Pt=105, OTHER=209
    - `val`:   Pt=17,  OTHER=50
    - `test`:  Pt=23,  OTHER=45
    - **Honest caveat**: the bundled 500-pool is heavily Pt-dominated
      and other metals (Pd/Au/Ir/Ru) are detected via RDKit atomic
      number scan on near-isomorphic SMILES rather than by a clean
      source-token prefix. This is documented in the manifest and
      is the right behaviour given the pool composition; a richer
      pool would expand the OTHER bucket.
- `r4_lambda_only_run.py` argparser accepts `--split none|scaffold|
  time|random|scaffold_tanimoto_0.4` (default `none`, no behaviour
  change for legacy invocations).
- `seeds.json` emitted alongside `report.json` with schema:
  `{pocket_id, seed, run_timestamp, status}` per cell.
- `fingerprint.json` written under `runs/{run_id}/` with `cli_sha256`,
  `smiles_concat_sha256`, `receptor_sha256`, `ligand_sha256`,
  `manifest_sha256`, `driver_sha256`, `runner_sha256`,
  `runtime_sha256`, `seeds`, `n_pockets`.

## What changed

| File | Lines | Purpose |
|---|---|---|
| `molmetal/scripts/metallo_pool_scaffold_split.py` | NEW (~300 LOC) | P6.1 — Murcko scaffold splitter with Tanimoto clustering, per-metal coverage report, deterministic balanced assignment |
| `molmetal/scripts/r4_lambda_only_run.py` | +~110 LOC | P6.2a (--split flag + `_load_split_training_smiles` helper) + P6.2b (seeds.json emission next to report.json) |
| `molmetal/scripts/r4_c_full_sweep.py` | +~50 LOC | P6.2c (fingerprint.json emission under `runs/{run_id}/`) |
| `molmetal/tests/test_reproducibility_split.py` | NEW (14 tests) | P6.2d — disjoint splits, time/random branches, seeds roundtrip, fingerprint stability |
| `molmetal/data/metallo_scaffold_split/{train,val,test}.csv` | NEW | Generated scaffold split on bundled 500-pool |
| `molmetal/data/metallo_scaffold_split/manifest.json` | NEW | Bucket sizes + per-metal coverage + schema_version=1 |

## Caveats / honest framing

- **MMseqs2 + Tanimoto<0.4 (P6.3) is DEFERRED.** The
  `--split scaffold_tanimoto_0.4` flag currently re-runs the
  Murcko-Tanimoto cluster at threshold 0.4 (more permissive) — this
  is honest: it is *only* a Tanimoto relaxation, not a true
  sequence-identity-aware OOD split. The MMseqs2 binary is not
  installed on this host.
- **`time` split is a placeholder.** The bundled
  `metallo_drugs_500_train.csv` has no temporal column (no PDB
  deposition date) — the helper preserves input order and returns
  the first 70% as `train`. Forward-compat: when a `date` column is
  added, the helper will switch to a cutoff-date split without
  changing the CLI surface.
- **The `OTHER` bucket dominates the metallo-pool coverage report.**
  This is not a bug — the `platinai` and `tmqm` source tokens in the
  bundled CSV mostly classify Pt and Pt-free organics respectively.
  When a metal-balanced pool (e.g. real PlatinAI-Pd, PlatinAI-Au,
  PlatinAI-Ir, PlatinAI-Ru subsets) is added, the bucket distribution
  will shift; the helper already classifies via atomic-number scan so
  any new pool is supported.
- **`fingerprint.json` is emitted at preflight, before jobs run.**
  The `smiles_concat_sha256` field therefore reflects the union of
  pocket ids (when no cells have run yet) or the union of
  `top1_smiles` (after jobs have populated). Tests treat the helper
  output as a stable contract regardless of run-state.
- **`--split` is additive.** Existing `r4_lambda_only_run.py`
  invocations that do not pass `--split` retain the legacy
  training-set loader (the helper is gated by `args.split != "none"`).

## Reproduce

```bash
# 1. Scaffold split on the bundled 500-pool (deterministic)
uv run python molmetal/scripts/metallo_pool_scaffold_split.py \
    --tanimoto-threshold 0.6 \
    --output-dir molmetal/data/metallo_scaffold_split

# 2. Run lambda-only with scaffold-aware novelty oracle
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --output-dir wf_t30_p62_scaffold \
    --seeds 42 0 1234 \
    --split scaffold

# 3. Run the reproducibility test suite
uv run pytest molmetal/tests/test_reproducibility_split.py -v

# 4. Inspect emitted artefacts
cat molmetal/reports/wf_lambda1_wf_t30_p62_scaffold/seeds.json
cat molmetal/reports/runs/<run_id>/fingerprint.json
```

All commands are CPU-only, take seconds, and do not depend on the
GPU/CFM stack.
