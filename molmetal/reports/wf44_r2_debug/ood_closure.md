# Wf-41 Phase 2 — Closure theorem cross-domain OOD (L12)

Cross-domain closure reachability benchmark on **PlatinAI 5K** + **MetalCytoToxDB 5-metal**, filtered by **Tanimoto<0.4 (Morgan-ECFP4) against the tmQM d-block training pool**.  Fills the **L12 gap** in :mod:`TODO/pending/83_lambda_algorithm_enhancement.md`.

## Run metadata

**Date (UTC)**: 2026-09-29T05:48:40Z  
**Tool**: `bench_closure_ood.py`  
**Method**: `closure+ood_filter`  
**Tanimoto threshold**: 0.40  
**Train pool size (tmQM)**: 5000  
**Domains**: platinai_5k, metalcytotox_5metal, tmqm_dblock  
**N samples per metal**: 10  
**Max depth**: 1  
**Click rules**: CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling  
**Partner terms**: 7  
**Seeds**: 1  
**Total wall**: 296.60 s  

## Aggregate (across all domains)

| Metric | Value |
|--------|-------|
| N input (before OOD) | 413 |
| N OOD (after filter) | 94 |
| N samples run | 94 |
| N closure_ok | 27 |
| closure_reachability_rate | 0.2872 |
| mean_wall_time (s) | 0.043564 |
| std_wall_time (s) | 0.071062 |
| peak_memory_mb | 0.1494 |
| N error | 0 |

## Per-domain reachability rate

| Domain | N input | N OOD | N samples | closure_ok | reachability_rate (mean ± std) | mean_wall_s |
|--------|---------|-------|-----------|------------|------------------------------|-------------|
| platinai_5k | 70 | 44 | 44 | 27 | 0.6310 ± 0.1854 | 0.092061 |
| metalcytotox_5metal | 50 | 50 | 50 | 0 | 0.0000 ± 0.0000 | 0.000887 |
| tmqm_dblock | 293 | 0 | 0 | 0 | 0.0000 ± 0.0000 | 0.000000 |

## Per-(domain, metal) reachability rate

| Domain | Metal | N input | N OOD | closure_ok | reachability_rate | ood_rate |
|-------|-------|---------|-------|------------|-------------------|----------|
| platinai_5k | Pt | 10 | 6 | 5 | 0.8333 | 0.6000 |
| platinai_5k | Pd | 10 | 6 | 3 | 0.5000 | 0.6000 |
| platinai_5k | Ru | 10 | 4 | 3 | 0.7500 | 0.4000 |
| platinai_5k | Rh | 10 | 8 | 6 | 0.7500 | 0.8000 |
| platinai_5k | Ir | 10 | 4 | 3 | 0.7500 | 0.4000 |
| platinai_5k | Os | 10 | 10 | 5 | 0.5000 | 1.0000 |
| platinai_5k | Au | 10 | 6 | 2 | 0.3333 | 0.6000 |
| metalcytotox_5metal | Ru | 10 | 10 | 0 | 0.0000 | 1.0000 |
| metalcytotox_5metal | Ir | 10 | 10 | 0 | 0.0000 | 1.0000 |
| metalcytotox_5metal | Rh | 10 | 10 | 0 | 0.0000 | 1.0000 |
| metalcytotox_5metal | Os | 10 | 10 | 0 | 0.0000 | 1.0000 |
| metalcytotox_5metal | Re | 10 | 10 | 0 | 0.0000 | 1.0000 |
| tmqm_dblock | Ag | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Au | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Cd | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Co | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Cr | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Cu | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Fe | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Hf | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Hg | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Ir | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | La | 3 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Mn | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Mo | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Nb | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Ni | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Os | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Pd | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Pt | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Re | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Rh | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Ru | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Sc | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Ta | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Tc | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Ti | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | V | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | W | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Y | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Zn | 10 | 0 | 0 | 0.0000 | 0.0000 |
| tmqm_dblock | Zr | 10 | 0 | 0 | 0.0000 | 0.0000 |

## Honest framing

- **CPU-only.** No CUDA / Triton / CFM retraining is exercised.
- **Deterministic.** closure_theorem is pure-functional; the
  benchmark can be re-run with the same flags and reproduce
  every per-sample metric.
- **Real cross-domain data.** PlatinAI 5K (R17) and
  MetalCytoToxDB 5-metal (Wf-21) are both real ship
  datasets; the OOD filter is the Wf-7 Tanimoto<0.4
  protocol (Morgan-ECFP4).
- **Valence-BNF rejection.** Many PlatinAI / MetalCytoTox
  samples fail the valence-BNF check on the start term
  (dative-bond organometallics with ``[M+]<-`` notation
  that the MLC ledger treats as under-saturated).
  These count as ``closure_ok=False`` but ``status=ok``
  — the BFS runs, the assertion rejects the seed.
- **Smoke-friendly.** The default ``--n-samples 50`` keeps
  the OOD-filtered cross-domain run under 5 min on CPU;
  the smoke tests in :mod:`test_closure_ood` use
  ``--n-samples 3`` for sub-second CI runs.

