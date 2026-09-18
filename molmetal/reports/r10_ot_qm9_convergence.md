# Real-QM9 OT: numerical convergence and three-seed control

The original real-data experiment is retained as negative numerical evidence: its finite soft transport plans had marginal residuals near 0.125. A valid hard permutation did not imply a valid marginally converged soft plan.

## Solver calibration without validation-based tuning

`uv run python molmetal/scripts/r10_ot_qm9_convergence.py` verifies the raw-data hashes in the original manifest, then tests the first eight training groups under each of the three fixed noise seeds. No validation loss or model-training outcome chooses the numerical setting.

| Entropic reg | Max iterations | Max marginal residual | Groups satisfying ≤0.001 | Wall seconds | Objective change |
|---:|---:|---:|---:|---:|---|
| 0.05 | 200 | 0.125008702 | 5/24 | 1.383 | Same cost and entropy coefficient |
| 0.05 | 2000 | 0.000216715 | 24/24 | 4.727 | Same cost and entropy coefficient |
| 1.0 | 2000 | 0.000111349 | 24/24 | 4.822 | Entropy coefficient increased 20× |

**Selected setting: reg=0.05, max_iter=2000.** This preserves the original raw sum-of-squared-coordinate cost and entropy coefficient; only the numerical iteration budget changes. No cost normalization is applied. The tested reg=1.0 alternative deliberately changes the entropic objective and was not used in the final training comparison.

## Fixed paired training experiment

```sh
uv run python molmetal/scripts/r10_ot_qm9_3seed.py --ot-reg 0.05 --ot-max-iter 2000 --max-marginal-error 0.001 --output-prefix molmetal/reports/r10_ot_qm9_3seed_converged
```

Both historical and new experiments use the same 64 real-QM9 training molecules, 32 held-out validation molecules, seeds (42, 0, 1234), model/encoder initialization, source-noise/time seeds, 30 updates, AdamW lr=0.001, and train-only normalization. The three source SHA256 values and selected rows are identical. Coordinates and atomic numbers for every selected row match the original DFT-optimized SDF, and canonical training/validation SMILES overlap is zero.

The new OT branch rejects every soft plan whose marginal error exceeds 0.001 before loss evaluation and before an optimizer update. Settings stay fixed throughout training; no invalid coupling is silently replaced or accepted merely because rounding yields a bijection. Both branches use the same independent-pairing validation objective before and after training.

| Seed | OT-off validation after | OT-on validation after | OT minus off | Max OT marginal residual | GPU POT solves | Gate passed |
|---:|---:|---:|---:|---:|---:|---|
| 42 | 6.479826 | 6.515982 | +0.036156 | 0.000247113 | 30 | True |
| 0 | 5.892367 | 5.810295 | -0.082072 | 0.000231944 | 30 | True |
| 1234 | 5.455019 | 5.423133 | -0.031886 | 0.000186250 | 30 | True |

Mean paired validation delta ± sample std (n=3, ddof=1): **-0.025934 ± 0.059338**.
Negative delta is lower validation loss. Seed 42 worsens; seeds 0 and 1234 improve. The mixed results do not establish consistent improvement, and all three seeds are reported. No acceptance criterion was redefined.

Total wall: 38.44s, below the 240s runtime budget. All 90 training couplings use real POT sinkhorn_log tensors on cuda:0 with no fallback. Runtime failures: [].
GPU: gfx1101, PyTorch 2.14.0+rocm7.2 / HIP 7.2.53211 / triton-rocm 3.8.0. CPU boundaries remain data parsing, group bookkeeping, and greedy hard-permutation projection. Meeting marginal tolerance is not a guarantee of exact transport optimality or convergence of the learned molecular model.

## Retained negative evidence and scope limits

The original max_iter=200 result reported mean paired delta -0.083671 ± 0.075525, but used substantially under-converged soft plans. It remains in `r10_ot_qm9_3seed.json/.csv/.md`; it is not overwritten or presented as the converged result.
QM9 is a small organic-molecule dataset, and this subset deliberately selects compatible ordered atom signatures to keep molecular metadata invariant under coordinate reassignment. It is not a metal-complex, MMP13, or representative full-QM9 benchmark. Thirty updates and three seeds do not establish large-scale convergence, general superiority, generated-molecule quality, or Vina improvements. The original TODO10 MMP13/Vina criterion remains unmeasured.

## Artifacts and checks

- `r10_ot_qm9_convergence.json`: numerical calibration and per-group effective backends/residuals.
- `r10_ot_qm9_3seed_converged.json`: protocol, split-row and SDF IDs, data/code SHA256, all losses, GPU/memory/backend/failure/budget records.
- `r10_ot_qm9_3seed_converged.csv`: every training step and validation batch, both treatments and all seeds.
- `r10_ot_qm9_3seed_converged.md`: runtime table and limitations.
- `uv run pytest -q molmetal/tests/test_real_qm9_ot_protocol.py molmetal/tests/test_ot_effective_backend.py molmetal/tests/test_minibatch_ot.py`: **20 passed**, including real-POT solver paths and rejection of unconverged soft plans before model loss evaluation.
