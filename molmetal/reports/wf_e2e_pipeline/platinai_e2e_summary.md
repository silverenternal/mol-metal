# R18 end-to-end pipeline benchmark on PlatinAI 5K

**Date**: 2026-09-26 (workflow `wf_e2e_pipeline`)
**Script**: `molmetal/scripts/r18_pipeline_orchestrator.py --n-samples 500 --dataset platinai`
**Output**: `molmetal/reports/wf_e2e_pipeline/platinai_e2e.json`

## Pipeline overview

The R18 orchestrator chains five ship-modules into a single
end-to-end pipeline:

| # | Step                              | Module                                                                 | What it does                                              |
|---|-----------------------------------|------------------------------------------------------------------------|-----------------------------------------------------------|
| 1 | `step1_closure_theorem`           | `molmetal_lam.lam_chem.closure.ProductiveSpace` + `CuAAC`              | BFS-closure over the 5-click chemistry, depth 1            |
| 2 | `step2_se3_sb_ode`                | `molmetal.adapters.se3_sb_integration` + `se3_lipman_integration`      | SE(3) Schrödinger-Bridge ODE integration, 10 sub-steps     |
| 3 | `step3_multiph_features`          | `molmetal.molmetal_lam.lam_chem.multi_ph_pll_integration`              | Multi-PH + PLL signature features, 8×8 bin grid           |
| 4 | `step4_variance_mcts`             | `molmetal.molmetal_lam.search_alg.variance_mcts_integration`           | Variance-controlled PUCT, 4-action selector               |
| 5 | `step5_hilbert_reward`            | `molmetal.molmetal_lam.reward.hilbert_astsr_integration`               | Hilbert-ASTSR composite (Hilbert + evolutionary branches) |

The orchestrator is **read-mostly** on every inner module — it
constructs them once, calls the forward contract, and never
mutates their state machines.

## Honest framing

This is a **CPU-only** end-to-end benchmark. Every module runs on
the CPU backend (no CUDA / Triton kernels are exercised). The
purpose is to prove the 5-step integration contract — that the
orchestrator can chain all five modules without import failures,
shape mismatches, or state-machine errors on real PlatinAI 5K
data.

It is **not** the same as a GPU training lift. A GPU run of
`step2_se3_sb_ode` would train a velocity-field EGNN; here we run
the integrator on a zero-velocity stub. The benchmark measures
*integration throughput*, not *model accuracy lift*.

## Per-step wall-time breakdown (PlatinAI 5K, n=500)

Actual numbers from the 2026-09-26 benchmark run
(`molmetal/reports/wf_e2e_pipeline/platinai_e2e.json`):

| Step                     | Wall-time / sample | Status mix                           |
|--------------------------|--------------------|---------------------------------------|
| step1_closure            | 0.004 ms           | ok: 500 / skipped: 0 / error: 0      |
| step2_se3                | 31.43 ms           | ok: 500 / skipped: 0 / error: 0      |
| step3_multiph            | 271.09 ms          | ok: 371 / skipped: 0 / error: 129    |
| step4_mcts               | 0.05 ms            | ok: 500 / skipped: 0 / error: 0      |
| step5_hilbert            | 6.02 ms            | ok: 500 / skipped: 0 / error: 0      |
| **total / sample**       | **~308 ms**        |                                       |

Total wall-clock for the 500-sample benchmark: **154.4 s**.

Success rate: **74.20 %** (371/500 fully OK across all five steps).
The 26 % failure rate is concentrated entirely in step 3
(multi-PH), where RDKit's UFFTYPER fails to embed 129 of the
500 PlatinAI SMILES due to exotic metal-coordinated topologies
(Pd6+2, Au6+3, Ir6, Ru5, etc.).  Steps 1, 2, 4, and 5 succeed on
all 500 samples.

The dominant cost is `step3_multiph_features` (RDKit conformer
embedding + multi-PH filter grid + PLL spectrum). On
metal-coordinated SMILES with exotic oxidation states (Pd6+2,
Au6+3, etc.) RDKit's UFFTYPER sometimes fails to embed the
molecule — those rows are recorded as `status="error"` rather
than skipped, and the benchmark counts them in the failure rate.

## Success criteria

* **PASS** — every step is `status="ok"` or `status="skipped"`.
  The benchmark reports `success=True` for the SMILES whose
  step-chain returns finite dicts.
* **FAIL** — any step raises an unhandled exception. The bench
  records `status="error"` for that step on that sample and the
  per-sample result is `success=False`.

## Files emitted by this benchmark

* `molmetal/reports/wf_e2e_pipeline/platinai_e2e.json` — full
  benchmark report (per-sample results, per-step wall-time /
  memory, status counts, dataset metadata).
* `molmetal/reports/wf_e2e_pipeline/verify_smoke.json` — 5-sample
  smoke verification (written by `verify_r18_pipeline.py`).
* `molmetal/reports/wf_e2e_pipeline/platinai_e2e_summary.md` —
  this summary.

## How to re-run

```bash
# Smoke (5 samples, ~0.5s):
uv run --no-sync python molmetal/scripts/verify_r18_pipeline.py --smoke

# Full 500-sample benchmark (~60-90s on a CPU host):
uv run --no-sync python molmetal/scripts/r18_pipeline_orchestrator.py \
    --n-samples 500 --dataset platinai \
    --out molmetal/reports/wf_e2e_pipeline/platinai_e2e.json

# MetalCytoTox subset (Ru/Ir/Rh/Re/Os):
uv run --no-sync python molmetal/scripts/r18_pipeline_orchestrator.py \
    --n-samples 100 --dataset metalcytotox
```

## Verification

```bash
uv run --no-sync python -m pytest molmetal/tests/test_r18_pipeline_orchestrator.py -v
```

The test suite (39 tests) pins the orchestrator's contract end to
end: construction, each step's dict shape, the full-pipeline
runner, the benchmark aggregator, and the PlatinAI / MetalCytoTox
e2e runners.

## Limitations

* The orchestrator's `step2_se3_sb_ode` uses a *zero-velocity
  stub* EGNN — the SE(3) integrator walks the trajectory under
  a no-op velocity. A trained EGNN would change the wall-time
  by an additive constant (the forward pass) but the
  integration-loop cost is the same.
* `step3_multiph_features` requires RDKit to embed the
  SMILES. Metal-coordinated complexes with unusual oxidation
  states occasionally fail to embed; those samples are
  reported as `error` rather than silently dropped.
* The orchestrator caches the inner modules in `__post_init__`,
  so the benchmark's first sample pays the import cost;
  subsequent samples reuse the cached modules.