# Wf-41 Phase 2 — Closure theorem cross-domain OOD (L12)

Cross-domain closure reachability benchmark on **PlatinAI 5K** + **MetalCytoToxDB 5-metal**, filtered by **Tanimoto<0.4 (Morgan-ECFP4) against the tmQM d-block training pool**.  Fills the **L12 gap** in :mod:`TODO/pending/83_lambda_algorithm_enhancement.md`.

## Run metadata

**Date (UTC)**: 2026-09-26T07:17:42Z  
**Tool**: `bench_closure_ood.py`  
**Method**: `closure+ood_filter`  
**Tanimoto threshold**: 0.40  
**Train pool size (tmQM)**: 5000  
**Domains**: platinai_5k, metalcytotox_5metal  
**N samples per metal**: 20  
**Max depth**: 1  
**Click rules**: CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling  
**Partner terms**: 7  
**Seeds**: 1  
**Total wall**: 221.33 s  

## Aggregate (across all domains)

| Metric | Value |
|--------|-------|
| N input (before OOD) | 240 |
| N OOD (after filter) | 186 |
| N samples run | 186 |
| N closure_ok | 51 |
| closure_reachability_rate | 0.2742 |
| mean_wall_time (s) | 0.053426 |
| std_wall_time (s) | 0.089696 |
| peak_memory_mb | 0.2614 |
| N error | 0 |

## Per-domain reachability rate

| Domain | N input | N OOD | N samples | closure_ok | reachability_rate | mean_wall_s |
|--------|---------|-------|-----------|------------|-------------------|-------------|
| platinai_5k | 140 | 86 | 86 | 48 | 0.5581 | 0.108935 |
| metalcytotox_5metal | 100 | 100 | 100 | 3 | 0.0300 | 0.005689 |

## Per-(domain, metal) reachability rate

| Domain | Metal | N input | N OOD | closure_ok | reachability_rate | ood_rate |
|-------|-------|---------|-------|------------|-------------------|----------|
| platinai_5k | Pt | 20 | 12 | 7 | 0.5833 | 0.6000 |
| platinai_5k | Pd | 20 | 9 | 3 | 0.3333 | 0.4500 |
| platinai_5k | Ru | 20 | 10 | 7 | 0.7000 | 0.5000 |
| platinai_5k | Rh | 20 | 17 | 11 | 0.6471 | 0.8500 |
| platinai_5k | Ir | 20 | 10 | 6 | 0.6000 | 0.5000 |
| platinai_5k | Os | 20 | 17 | 9 | 0.5294 | 0.8500 |
| platinai_5k | Au | 20 | 11 | 5 | 0.4545 | 0.5500 |
| metalcytotox_5metal | Ru | 20 | 20 | 3 | 0.1500 | 1.0000 |
| metalcytotox_5metal | Ir | 20 | 20 | 0 | 0.0000 | 1.0000 |
| metalcytotox_5metal | Rh | 20 | 20 | 0 | 0.0000 | 1.0000 |
| metalcytotox_5metal | Os | 20 | 20 | 0 | 0.0000 | 1.0000 |
| metalcytotox_5metal | Re | 20 | 20 | 0 | 0.0000 | 1.0000 |

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

