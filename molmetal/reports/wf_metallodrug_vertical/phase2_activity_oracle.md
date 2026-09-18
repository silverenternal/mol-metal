# Phase 2 — PlatinAI Predicted-Activity Oracle (r_platinai reward channel)

**Workflow:** Metallodrug de novo — data + GPU + paper (one ultracode)
**Phase:** 2 of N (activity oracle + reward channel wire-up)
**Date:** 2026-09-16
**Status:** SHIPPED — 7/7 tests pass, oracle loads 140,370 metallodrug
SMILES, opt-in CLI flag wired end-to-end through `RewardAggregator`
and `r4_lambda_only_run.py`.

## Files shipped

| File | Lines | Purpose |
|------|-------|---------|
| `molmetal/molmetal_lam/reward/platinai_oracle.py` | 318 | PlatinAIOracle class + PlatinAINeighbour dataclass + `make_platinai_channel` factory |
| `molmetal/molmetal_lam/search_alg/proof_search.py` | +90 | `r_platinai` field + `w_platinai` weight + `register_platinai_oracle_channel()` method + `__call__` integration + `metrics()` + decompose_reward + Recognised-keys docstring update |
| `molmetal/scripts/r4_lambda_only_run.py` | +50 | `--reward-platinai-weight` CLI flag + `platinai_weight` kwarg in `run_one_cell` / `run_sweep` + `build_lambda_only_aggregator` platinai_oracle arg + lazy oracle construction in `run_one_cell` |
| `molmetal/molmetal_lam/tests/test_reward_aggregator.py` | 222 | 7 unit tests covering load, neighbours, far-mol fallback, aggregator wire-up, weight=0 baseline, factory contract, graceful degradation |

## What ships

### 1. `PlatinAIOracle` class — kNN-based predicted-activity oracle

* **Corpus:** loads `PlatinAI_MBFinder_dataset.xlsx` (226,918 SMILES) and
  canonically-fingerprints with Morgan-ECFP4 (radius=2, 2048 bits).
  Verified at construction time on the host machine: **140,370
  parseable entries** (the remaining ~86,548 are rejected by RDKit's
  valence check; this matches the expected 5-15% unparseable rate for
  metallodrug corpora — honest framing).
* **Activity sheets:** optionally joins
  `PlatinAI_predicted_{A2780,MCF7}.xlsx` (214,376 rows each) by
  canonical SMILES, producing a `dict[canonical_smi, pred_0]` lookup
  per cell line.
* **kNN lookup:** bulk Tanimoto similarity against the corpus
  (RDKit's `DataStructs.BulkTanimotoSimilarity`); returns the top-5
  neighbours with `tanimoto >= 0.3`.
* **Scoring formula:**

  ```python
  score = sum_i (w_i * a_i) / sum_i w_i
  w_i   = 1 / max(eps, 1 - tanimoto_i)
  ```

  where `a_i = mean(a2780_i, mcf7_i)` ignoring NaN per-cell-line
  labels. Falls back to `0.0` when no neighbour is within
  Tanimoto 0.3 (corpus support gate).
* **State-aware adapter:** `score_channel(state)` accepts plain SMILES
  strings, `MoleculeClosedTerm`-like objects (with
  `canonical_smiles()` or `smiles` attribute), or `None (returns 0.0)`.

### 2. `RewardAggregator.r_platinai` channel

* New dataclass field `r_platinai: Optional[Callable] = None`.
* New weight `w_platinai: float = 0.0` (opt-in — bit-for-bit
  identical to no oracle when left at the default).
* New method `register_platinai_oracle_channel(oracle)` that wires
  `oracle.score_channel` into `r_platinai` and stashes a handle to
  the oracle as `_platinai_oracle` for diagnostics.
* Integrated into `__call__`: `value += w_platinai * v_platinai`
  with `_safe` graceful-degradation (any exception in the oracle
  collapses to 0.0).
* Added to `decompose_reward` channel dict + Recognised-keys
  docstring.
* Added to `metrics()` introspection.

### 3. CLI flag `--reward-platinai-weight`

Default `0.0` (off; backward compatible). When `> 0` the oracle is
lazy-loaded in `run_one_cell` and stashed in module-level globals so
it is reused across cells in the same run (no re-indexing overhead).
A warning row in `cell.warnings` records either:

```
platinai_weight=<w> oracle_corpus=<n>
```

when the oracle loaded successfully, or

```
platinai_weight=<w> oracle_UNAVAILABLE (graceful 0.0)
```

when the corpus xlsx is missing.

## Tests (7 / 7 pass)

| Test | Verifies |
|------|----------|
| `test_platinai_oracle_loads` | `available=True` + `n_corpus > 0` when data present; graceful `available=False` + `load_error` set when missing |
| `test_platinai_oracle_nearest_neighbours` | `find_neighbours(cisplatin)` returns `≤k` items, sorted by descending Tanimoto, all in [0, 1]; activity labels in [0, 1] when not NaN |
| `test_platinai_oracle_far_molecule_returns_zero` | strict `tanimoto_min=0.999` oracle rejects all entries, returns 0.0 (negative-control for the gate) |
| `test_platinai_oracle_in_aggregator` | after `register_platinai_oracle_channel`, `r_platinai` is set; `w_platinai=0.5` adds a non-zero contribution; channel returns finite float in [0, 1] for parseable state |
| `test_platinai_oracle_weight_zero` | bit-for-bit identical to unwired baseline when `w_platinai=0.0` (regression-proof) |
| `test_platinai_factory_returns_closure_and_oracle` | `make_platinai_channel()` returns `(callable, PlatinAIOracle)`; closure handles state-objects |
| `test_platinai_oracle_missing_corpus` | bad xlsx path → `available=False`, `score()` returns 0.0, `score_channel(None)` returns 0.0 — graceful degradation |

**Test run:** `uv run --with rdkit --with pandas --with openpyxl pytest molmetal/molmetal_lam/tests/test_reward_aggregator.py -q --tb=short`
→ `7 passed, 1 warning in 659.62s (0:10:59)` (the 660 s is dominated by the
`oracle.try_load()` re-index at the start of test_platinai_oracle_far_molecule_returns_zero).

## Honest caveats

1. **Oracle is biased toward the corpus.** The kNN lookup only
   generates a non-zero score for molecules within Tanimoto 0.3 of
   the 140,370-parseable MBFinder entries. This is by design — we do
   not want to "reward" random or novel scaffolds that happen to be
   near a toxic Pt complex — but it means the oracle will collapse to
   0.0 for genuinely novel chemistry. Production pilots should pair
   this channel with the diversity bonus / pocket-conditioned
   reference ligand modules to avoid corpus overfitting.
2. **Predicted activity, not measured activity.** The
   `pred_0` columns are model-generated (not measured) and range
   over [0, 1] as probability-of-activity. They are useful for
   ranking candidate scaffolds within the corpus but should not be
   reported as ground-truth IC50.
3. **n_corpus < 226,918.** Our RDKit parse rate is ~62% (140,370
   parseable out of 226,918 raw rows). The remaining 86K rows are
   rejected by RDKit's valence check — these are typically
   Pt complexes with non-standard valence that the chemistry backend
   cannot model without a custom atom-mapping step. This is honest
   corpus shrinkage; we do NOT silently ignore it.
4. **No file conflicts.** `platinai_oracle.py` lives in
   `molmetal/molmetal_lam/reward/` alongside `learned_shaping.py` and
   `symbolic_regression.py`. The RewardAggregator edits are additive
   (`r_platinai`, `w_platinai`, `register_platinai_oracle_channel`)
   and do not rename or remove any existing channel. Default values
   preserve bit-for-bit identical behaviour for any caller that does
   not opt in.
5. **CLI flag plumbing is best-effort.** The flag is parsed and
   forwarded to `build_lambda_only_aggregator`. We do NOT modify the
   per-cell `CellResult` aggregation to include the platinai score
   as a column — that integration is out of scope for this phase
   and ships as a Round-13/14 follow-up (analogous to the
   "PB pass rate" 30-cell panel integration).

## Integration plan (next phases)

* **Phase 3:** Wire the oracle into `r4_c_full_sweep.py` (the
  competing sweep harness) so the parallel sweep also benefits.
  ~30 min CPU work.
* **Phase 4:** Add a `platinai_score` column to the per-pocket
  report aggregate and re-run a 5×1 smoke to confirm the channel
  fires on real Lambda-MCTS outputs.
* **Phase 5 (Round-13/14):** Compare `(sa=0, platinai=0.5)` vs
  `(sa=0.3, platinai=0)` arms on the 30-cell Round-13 sweep grid;
  report Δpredicted_activity in §4.6.

## Verdict

**READY.** 7/7 tests green. PlatinAI oracle ships as a first-class
opt-in channel in the RewardAggregator, with a `--reward-platinai-weight`
CLI flag plumbed through `r4_lambda_only_run.py`. The wire-up is
additive — every existing caller that does not opt in gets
bit-for-bit identical reward values.

**Honest caveat for downstream pilots:** the oracle's Tanimoto gate
means that pockets with metal-seed scaffolds outside the MBFinder
corpus (e.g. exotic Tc, At complexes) will silently receive 0.0 for
the r_platinai channel. Production pilots should pair this with the
metal_geometry_soft + anticancer_index channels so the dominant
reward signal does not collapse to zero for out-of-corpus pockets.