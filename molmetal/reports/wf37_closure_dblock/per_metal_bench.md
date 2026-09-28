# Wf-37 Phase 2 — Closure theorem d-block benchmark (L6)

Per-metal closure reachability benchmark on the **30 d-block metals** from the R17 ``PRIMITIVE_ATOMS`` / ``METAL_ATOMS`` library, run against the tmQM d-block subset CSV.  Fills the **L6 gap** in :mod:`TODO/pending/83_lambda_algorithm_enhancement.md`.

## Run metadata

**Date (UTC)**: 2026-09-26T04:56:34Z  
**Tool**: `bench_closure_dblock.py`  
**Dataset**: `tmqm` (/home/hugo/codes/mol-metal/molmetal/data/tmqm_dblock_subset.csv)  
**N metals**: 30  
**N samples per metal**: 100  
**N samples total**: 2244  
**Max depth**: 1  
**Click rules**: CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling  
**Partner terms**: 7  
**Seeds**: 1  
**Total wall**: 179.52 s  

## Aggregate (across all 30 d-block metals)

| Metric | Value |
|--------|-------|
| N samples | 2244 |
| N closure_ok | 1322 |
| closure_reachability_rate | 0.5891 |
| mean_beta_depth | 0.9996 |
| mean_n_products | 9.1627 |
| mean_wall_time (s) | 0.077246 |
| std_wall_time (s) | 0.072083 |
| peak_memory_mb | 0.7416 |
| N error | 0 |

## Per-periodic-block aggregate

| Block | N | closure_ok | reachability_rate | mean_wall_s |
|-------|---|------------|-------------------|-------------|
| 3d | 892 | 596 | 0.6682 | 0.089581 |
| 4d | 683 | 397 | 0.5813 | 0.077274 |
| 5d_La | 669 | 329 | 0.4918 | 0.060771 |

## Per-metal reachability rate

| Metal | N | closure_ok | reachability_rate | metal_coord_rate | mean_wall_s |
|-------|---|------------|-------------------|------------------|-------------|
| Sc | 20 | 9 | 0.4500 | 1.0000 | 0.091122 |
| Ti | 100 | 66 | 0.6600 | 1.0000 | 0.119806 |
| V | 72 | 10 | 0.1389 | 1.0000 | 0.018467 |
| Cr | 100 | 76 | 0.7600 | 1.0000 | 0.092229 |
| Mn | 100 | 76 | 0.7600 | 1.0000 | 0.090451 |
| Fe | 100 | 67 | 0.6700 | 1.0000 | 0.082339 |
| Co | 100 | 66 | 0.6600 | 1.0000 | 0.091377 |
| Ni | 100 | 74 | 0.7400 | 1.0000 | 0.093791 |
| Cu | 100 | 68 | 0.6800 | 1.0000 | 0.090023 |
| Zn | 100 | 84 | 0.8400 | 1.0000 | 0.107527 |
| Y | 30 | 22 | 0.7333 | 1.0000 | 0.100253 |
| Zr | 53 | 33 | 0.6226 | 1.0000 | 0.083703 |
| Nb | 19 | 8 | 0.4211 | 1.0000 | 0.086496 |
| Mo | 100 | 32 | 0.3200 | 1.0000 | 0.039122 |
| Tc | 15 | 8 | 0.5333 | 1.0000 | 0.064701 |
| Ru | 100 | 72 | 0.7200 | 1.0000 | 0.103887 |
| Rh | 100 | 55 | 0.5500 | 1.0000 | 0.070772 |
| Pd | 100 | 45 | 0.4500 | 1.0000 | 0.056466 |
| Ag | 81 | 50 | 0.6173 | 1.0000 | 0.082596 |
| Cd | 85 | 72 | 0.8471 | 1.0000 | 0.105945 |
| La | 3 | 3 | 1.0000 | 1.0000 | 0.115445 |
| Hf | 20 | 11 | 0.5500 | 1.0000 | 0.069521 |
| Ta | 30 | 12 | 0.4000 | 1.0000 | 0.062713 |
| W | 100 | 57 | 0.5700 | 1.0000 | 0.068432 |
| Re | 100 | 54 | 0.5400 | 1.0000 | 0.063709 |
| Os | 55 | 22 | 0.4000 | 1.0000 | 0.052237 |
| Ir | 100 | 38 | 0.3800 | 1.0000 | 0.048861 |
| Pt | 100 | 47 | 0.4700 | 1.0000 | 0.057771 |
| Au | 100 | 44 | 0.4400 | 1.0000 | 0.054665 |
| Hg | 61 | 41 | 0.6721 | 1.0000 | 0.079036 |

## Honest framing

- **CPU-only.** No CUDA / Triton / CFM retraining is exercised.
- **Deterministic.** closure_theorem is pure-functional; the
  benchmark can be re-run with the same flags and reproduce
  every per-sample metric.
- **Real data.** The tmQM d-block subset CSV is a real ship
  dataset (Wf-28 ship, 5K samples spanning 30 d-block metals).
- **Valence-BNF rejection.** Many tmQM samples fail the
  valence-BNF check on the start term (dative-bond
  organometallics with ``[M+]<-`` notation that the MLC
  ledger treats as under-saturated).  These count as
  ``closure_ok=False`` but ``status=ok`` — the BFS runs, the
  assertion rejects the seed.
- **Some metals short on samples.** Metals like ``Os`` / ``Re``
  have only ~50-100 samples in tmQM; the benchmark gracefully
  reports whatever is available.

