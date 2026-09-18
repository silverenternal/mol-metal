# Real QM9 three-seed OT training experiment

Execution status: **complete**. Total wall: 37.10s.

```sh
uv run python molmetal/scripts/r10_ot_qm9_3seed.py --ot-reg 0.05 --ot-max-iter 2000 --max-marginal-error 0.001 --output-prefix molmetal/reports/r10_ot_qm9_graph_independent
```

This protocol uses real QM9 (DFT-optimized) SDF coordinates, separate from the Gaussian micro-benchmark. No Vina, molecular generation, or MMP13 experiment was performed.

## Fixed protocol

```json
{
  "dataset": "QM9 real 3D SDF coordinates; no generated targets",
  "split": "project QM9Dataset 80/10/10 permutation using numpy RNG seed 42; not official benchmark split",
  "train_full_size": 105576,
  "val_full_size": 13197,
  "train_selected": 64,
  "val_selected": 32,
  "selection": "first 8 shared ordered-atom signatures (4..16 atoms), sorted by length then values; first 8 train/4 val split rows per signature",
  "canonical_smiles_train_val_overlap": 0,
  "preprocess": "train-only median coordinate std, per-molecule centering; complete graphs; 29-atom padding",
  "position_scale": 1.237749457359314,
  "steps_per_run": 30,
  "seeds": [
    42,
    0,
    1234
  ],
  "model": "VelocityNet 2 layers + MolEncoder 1 layer, scalar-property conditioning disabled",
  "hidden_dim": 16,
  "optimizer": "AdamW",
  "lr": 0.001,
  "gradient_clip_norm": 1.0,
  "dtype": "float32",
  "ot": "POT sinkhorn_log, reg=0.05,max_iter=2000, greedy hard projection; one compatible group of 8 per training batch",
  "marginal_error_gate": 0.001,
  "ot_objective_note": "raw sum-of-squared-coordinate cost, without normalization; changing reg changes entropic objective; changing iteration count only changes numerical budget",
  "eval": "common independent-pairing CFM objective, fixed noise/times per seed; 8 compatible batches of 4; equal batch weighting",
  "initialization": "matched model/encoder state hash within each seed; fresh training noise seeded identically across branches",
  "budget_seconds": 240
}
```

## Runtime

```json
{
  "python": "3.12.13",
  "torch": "2.14.0+rocm7.2",
  "hip": "7.2.53211",
  "pot": "0.9.7.post1",
  "scipy": "1.18.1",
  "triton_rocm": "3.8.0",
  "device": "cuda:0",
  "gpu": "AMD Radeon Graphics",
  "arch": "gfx1101"
}
```

## Results

| Seed | OT | First train loss | Last train loss | Common val before | Common val after | Train seconds |
|---|---|---:|---:|---:|---:|---:|
| 42 | False | 12.854166 | 5.021603 | 15.192387 | 5.990429 | 0.451 |
| 42 | True | 13.143356 | 4.438663 | 15.192387 | 5.932154 | 5.986 |
| 0 | False | 5.760997 | 5.949902 | 7.193752 | 5.883198 | 0.201 |
| 0 | True | 5.108571 | 4.309663 | 7.193752 | 5.789574 | 6.310 |
| 1234 | False | 8.733059 | 4.907282 | 11.046428 | 5.499146 | 0.211 |
| 1234 | True | 7.661082 | 4.713912 | 11.046428 | 5.443136 | 6.308 |

Paired common-validation differences (OT minus off; negative is lower): -0.058275, -0.093624, -0.056010
Mean ± sample std across 3 seeds: **-0.069303 ± 0.021093** (ddof=1).
Training losses use different pairing objectives and are not an unbiased between-method evaluation. The validation rows use identical held-out samples/noise/times within each seed, with OT disabled for both treatments.

## Effective backend and numerical residual

| Seed | Solver counts | Device | Fallbacks | Maximum marginal absolute error | Peak allocated GPU bytes |
|---|---|---|---|---:|---:|
| 42 | {'pot_sinkhorn_log': 30} | ['cuda:0'] | [] | 0.000247113 | 77577216 |
| 0 | {'pot_sinkhorn_log': 30} | ['cuda:0'] | [] | 0.000231944 | 77577216 |
| 1234 | {'pot_sinkhorn_log': 30} | ['cuda:0'] | [] | 0.000186250 | 77577216 |

POT soft plans and the model execute on GPU. Dataset parsing, group bookkeeping, and greedy hard-permutation rounding run on CPU. The hard pairing algorithm is unchanged.

**Finite soft plans alone do not certify convergence.** The fixed solver settings must be interpreted alongside the reported marginal residuals. These losses are descriptive micro-experiment results, not evidence for OT optimality, large-scale model convergence, general superiority, molecular quality, or better Vina scores. The subset is small and selected for compatible atom signatures.

## Provenance and split

Every selected cached atom/coordinate tensor was matched byte-for-byte to an original SDF record. The JSON contains 64 training and 32 validation split-row indices, sample SHA256s, SDF IDs, and canonical SMILES. Training/validation canonical SMILES overlap is zero. The project loader's seed-42 80/10/10 split is not described as an official benchmark split.

| Source | SHA256 | Bytes |
|---|---|---:|
| /home/hugo/codes/try_triton_on_rocm/data/qm9_processed.pt | bf1108b04f7554670e237c1fb46180a662f5ebc3a58c5b66e3430600edc7cec5 | 102459665 |
| /home/hugo/codes/try_triton_on_rocm/data/qm9_raw/gdb9.sdf | 98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e | 235434378 |
| /home/hugo/codes/try_triton_on_rocm/data/qm9_raw/gdb9.sdf.csv | 73a67793e3cfa9660f001278bd019c143f57e4785db537a01811cf2ce72aa7eb | 27685401 |

## Failures and budget

[]

An empty failures list means no runtime failure; it does not certify scientific acceptance or solver convergence. The runner enforces its wall budget and saves each completed branch to JSON. Hardware/toolchain-dependent GPU reductions may produce small floating-point variation on reproduction.

This run rejects every soft plan with marginal error >0.001 before evaluating its training loss or updating model weights. Completion of all OT branches therefore certifies this marginal tolerance for all used training couplings. Regularization and iteration budget are fixed throughout each branch; validation before and after training uses the same common independent-pairing objective.
