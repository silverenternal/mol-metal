# Wf-17 P4.2 — closure theorem REAL benchmark on PlatinAI 5K

Comparison anchor between the Wf-16 P4.1 mock skeleton and the
real closure-theorem BFS shipped by R17 in
:mod:`molmetal_lam.lam_chem.closure`.

## Run metadata

**Date (UTC)**: 2026-09-25T17:46:48Z  
**Tool**: `closure_real_benchmark.py`  
**Dataset**: `/home/hugo/codes/mol-metal/molmetal/data/platinai_5k.csv` (platinai_5k)  
**N samples**: 500  
**Max depth**: 1  
**Click rules**: CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling  
**Partner terms**: 7  
**Smoke mode**: False  

## Contract-at-a-glance

The benchmark reports the six documented metrics that the
Wf-17 P4.2 brief pins as the comparison contract vs the
Wf-16 P4.1 mock:


1. `closure_success_rate` — fraction of PlatinAI samples on
   which `closure_theorem()` returns True.
2. `mean_beta_depth` — mean β-reduction depth reached in the
   BFS (max_depth_seen averaged over samples).
3. `mean_wall_time` — mean per-sample wall-clock seconds.
4. `std_wall_time` — sample stdev of per-sample wall-time.
5. `peak_memory_mb` — peak per-sample memory delta (tracemalloc).
6. `total_wall_time` — sum of per-sample wall-times.


## Aggregate metrics

| Metric | Value |
|--------|-------|
| N samples | 500 |
| N parsed | 500 |
| N closure_ok | 255 |
| closure_success_rate | 0.5100 |
| mean_beta_depth | 0.9880 |
| mean_wall_time (s) | 0.101083 |
| std_wall_time (s) | 0.117091 |
| median_wall_time (s) | 0.142137 |
| peak_memory_mb | 0.6496 |
| total_wall_time (s) | 50.5417 |
| N error | 0 |
| N status ok | 500 |

## Status counts

| Status | Count |
|--------|-------|
| ok | 500 |
| skipped | 0 |
| error | 0 |

## Per-sample depth distribution

| Depth (β) | Count |
|-----------|-------|
| 0 | 6 |
| 1 | 494 |

## Wf-16 P4.1 mock vs Wf-17 P4.2 real

| Aspect | Wf-16 P4.1 (mock) | Wf-17 P4.2 (real) |
|--------|-------------------|-------------------|
| BFS engine | SHA-256 deterministic mock | `ProductiveSpace` BFS |
| Source | `lambda_100x3_cpu_mock.py` | `closure_real_benchmark.py` |
| Dataset | synthetic pocket pool | PlatinAI 5K real |
| Reachability metric | `mock_mean` projection | `closure_success_rate` |
| Wall-time | 2 s/cell deterministic | ~0.07 s/sample (CPU) |
| Memory | 0.0 MB (no allocation) | ~0.65 MB peak |
| Determinism | deterministic | deterministic (closure is pure) |


The Wf-16 P4.1 mock produces a **projection** of what the full
100x3 sweep would look like; the Wf-17 P4.2 real benchmark
measures what the closure-theorem BFS **actually** returns on
500 real PlatinAI samples.  These two outputs are complementary:
the mock is a feasibility check, the real run is the ground
truth on the available data subset.


## Honest framing

- **CPU-only.** No CUDA / Triton / CFM retraining is exercised.
- **Depth-1 by default.** Mirrors the Wf-16 P4.1 mock budget.
- **Real data.** PlatinAI 5K CSV is the canonical R17 ship dataset.
- **No synthetic fallback.** When the CSV is unreachable the script aborts.
- **Valence-BNF rejection.** Many PlatinAI samples fail the
  `closure_theorem` valence-BNF check on the start term
  (the dative-bond organometallics carry [CH-] / [M+] charges
  that the MLC ledger treats as under-saturated).  These
  samples count as `closure_ok=False` but `status=ok` (the
  BFS runs, the assertion rejects the seed).


## How to re-run

```bash
uv run --no-sync python molmetal/scripts/closure_real_benchmark.py --n-samples 500
```
