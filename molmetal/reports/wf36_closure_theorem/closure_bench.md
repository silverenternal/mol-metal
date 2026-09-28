# Wf-36 Phase 1 — Closure theorem end-to-end verifier (L1)

Per-metal closure reachability benchmark on the tmQM d-block
subset + PlatinAI 5K datasets.  Fills the **L1 gap** in
:mod:`TODO/pending/83_lambda_algorithm_enhancement.md`.


## Run metadata

**Date (UTC)**: 2026-09-26T04:45:07Z  
**Tool**: `verify_closure_theorem.py`  
**Dataset**: `tmqm` (/home/hugo/codes/mol-metal/molmetal/data/tmqm_dblock_subset.csv)  
**N metals**: 9  
**N samples per metal**: 100  
**N samples total**: 836  
**Max depth**: 1  
**Click rules**: CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling  
**Partner terms**: 7  
**Total wall**: 72.60 s  

## Aggregate

| Metric | Value |
|--------|-------|
| N samples | 836 |
| N closure_ok | 427 |
| closure_reachability_rate | 0.5108 |
| mean_beta_depth | 0.9988 |
| mean_n_products | 9.1914 |
| mean_wall_time (s) | 0.083948 |
| std_wall_time (s) | 0.100952 |
| peak_memory_mb | 0.7324 |
| N error | 0 |

## Per-metal reachability rate

| Metal | N | closure_ok | reachability_rate | mean_wall_s |
|-------|---|------------|-------------------|-------------|
| Pt | 100 | 47 | 0.4700 | 0.088227 |
| Ru | 100 | 72 | 0.7200 | 0.147790 |
| Ir | 100 | 38 | 0.3800 | 0.074426 |
| Rh | 100 | 55 | 0.5500 | 0.106225 |
| Re | 100 | 54 | 0.5400 | 0.080962 |
| Os | 55 | 22 | 0.4000 | 0.051778 |
| Au | 100 | 44 | 0.4400 | 0.053465 |
| Pd | 100 | 45 | 0.4500 | 0.057048 |
| Ag | 81 | 50 | 0.6173 | 0.080477 |

## Honest framing

- **CPU-only.** No CUDA / Triton / CFM retraining is exercised.
- **Deterministic.** closure_theorem is pure-functional.
- **Real data.** Both CSV inputs are real ship datasets.
- **Valence-BNF rejection.** Many tmQM samples fail the
  valence-BNF check on the start term (dative-bond organometallics
  with ``[M+]<-`` notation that the MLC ledger treats as
  under-saturated).  These count as ``closure_ok=False`` but
  ``status=ok`` — the BFS runs, the assertion rejects the seed.

