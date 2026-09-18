# Round-10 axis B — measured ROCm OT micro-benchmark

This is a synthetic CFM training comparison on the local gfx1101 GPU. No
ligand or receptor file was loaded; the historical script name `1h36` does
not make this a real-pocket evaluation. The existing scientific acceptance
criteria for real-pocket evaluation, full-batch OT parity, CFG and Pt prior
benefit remain unproven by this run.

Reproduce from the repository root:

```sh
uv run python molmetal/scripts/r10_ot_ablation_1h36.py --epochs 50 --b 8 --n 32 --hidden-dim 16 --seed 0 --device cuda:0 --output-prefix molmetal/reports/r10_ot_rocm_sinkhorn_seed0
```

## Protocol

- Seed 0; two independently initialized copies with the same initial seed.
- 50 optimizer updates per branch, B=8, N=32, H=16, two EGNN layers.
- AdamW lr=0.001, gradient clipping norm 1.0, float32; fresh Gaussian source
  and target coordinates each step, with matched RNG seeds across branches.
- Baseline: `use_minibatch_ot=False`, independent ordering, not full-batch OT.
- Treatment: four groups of two, entropic OT reg=0.05, maximum 200 iterations,
  log-domain Sinkhorn followed by the existing greedy permutation projection.
- Python 3.12.13, torch 2.14.0+rocm7.2, HIP
  7.2.53211, triton-rocm 3.8.0,
  POT 0.9.7.post1, scipy 1.18.1.
- GPU: AMD Radeon Graphics, gfx1101, cuda:0.

## Measurements

| Quantity | Independent | Mini-batch OT |
|---|---:|---:|
| First-step loss | 17.709328 | 18.376522 |
| Final-step loss | 6.613095 | 6.193394 |
| Mean loss across steps | 7.373599 | 7.131063 |
| Population std across steps | 2.253787 | 2.241872 |
| Last-10 mean loss | 6.074988 | 6.055853 |
| Re-paired sample fraction | 0.000 | 0.460 |
| Mean target velocity squared | 2.007865 | 1.934749 |
| Training wall seconds | 0.825 | 5.128 |

All 200 coupled groups used
`pot_sinkhorn_log` with plan tensors on cuda:0. Fallback count: 0.
Maximum marginal absolute error after the finite iteration budget:
0.001386285. This is recorded as a numerical residual;
it does not certify convergence to the solver's tight default tolerance.

The final loss is lower for this seed, but last-10 means differ by only
-0.019135. Step-to-step std is not a
confidence interval or multi-seed uncertainty estimate. These training-only
numbers cannot establish molecular validity, Vina improvements, or general
scientific superiority. GPU atomic accumulation may cause small roundoff
variation across repetitions. Neither the stopping condition nor the
original TODO acceptance criterion was changed to force success.

## Backend correction and CPU boundaries

The old implementation called `_pot.ot.sinkhorn`; installed POT exports
`ot.sinkhorn` and has no `ot.ot`. Its caught AttributeError silently selected
CPU SciPy Hungarian. The historical `r10_ot_rocm_measured_seed0.*` files
preserve that run and explicitly label it as pre-fix.

The corrected implementation uses the public POT call with torch GPU costs
and marginals. `sinkhorn_log` solves the same entropic problem without the
exp(-cost/reg) underflow of a direct float32 kernel. This changes the
numerical implementation, not the entropic objective. The original greedy
rounding is retained; the hard permutation is not relabeled exact OT.

Group bookkeeping and greedy soft-plan rounding still run on CPU; the small
plan is copied to CPU for that projection and indices return to the GPU.
SciPy Hungarian is used only when explicitly requested or when POT is
unavailable/fails, with a RuntimeWarning and a recorded reason. Hence this
is a GPU Sinkhorn solve, not an entirely GPU-resident evaluation. For these
tiny groups, its 5.13s training wall is slower than the
historical Hungarian path; no speedup claim is made.

## Validation

`uv run pytest -q molmetal/tests/test_ot_effective_backend.py molmetal/tests/test_minibatch_ot.py`
exercises real POT on CPU/ROCm, nonzero soft transport mass, large-cost
numerical stability, explicit Hungarian selection, warning/diagnostics for
missing POT, and unchanged loss when diagnostics are enabled. No mocked
transport solution can satisfy the real-backend tests.
