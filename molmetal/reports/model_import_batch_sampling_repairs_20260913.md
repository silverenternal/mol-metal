# Model import, graph batching and Lipman sampling repairs

2026-09-13. This note records production defects found during combined model
and Lambda regression, their fixes, and the measurements affected by them.
Historical reports are preserved; their model numbers describe the older
runtime and must not be used as evidence for the repaired implementation.

## Independent official and local flow-matching imports

`_import_fm_lib` previously removed the public `flow_matching*` modules and
prepended the official clone to `sys.path`. Later local-loss imports then
resolved to the official package, which has a different loss API. Temporarily
restoring those names would still leave class identity and deferred imports
ambiguous.

The clone now loads under the stable private package
`molmetal.adapters.flow_matching_lipman._reference`. A finder matches only this
private prefix, and each upstream module gets its own import function that
redirects upstream absolute `flow_matching` imports. Source files and global
`builtins.__import__` are unchanged. Public MolFlow module objects and
`sys.path` are preserved. Repeated adapter setup returns the same upstream
class objects; official Affine paths and ODE solves execute correctly between
real local-loss forward/backward calls.

The real private-package entry point supports default-clone pickle loading in
a fresh interpreter without calling adapter setup first. An alternate clone
can be configured before its first import (and before unpickling there); the
loader rejects switching to a different clone within one process rather than
reusing one module identity for incompatible source. It does not claim that
old pickle files whose classes were globally named `flow_matching.*` can be
unambiguously migrated. State-dict checkpoints keep their existing keys.

## Graph independence and singleton batches

`scatter_sum` intentionally retains its historical `(N,F)` result for
`batch_size=None` or `1`. `EGNNLayer` treated that result as `(B,N,F)`, causing
singleton loss computation to fail when combining vector norms with batched
node features. The caller now restores its explicit batch dimension.

The scalar paths in `EGNNLayer` and `MolEncoder` also reshaped `(B,2,E)` directly
to `(2,B*E)`, mixing index channels, and summed all molecules into one graph
before broadcasting the result. They now flatten source/destination channels
separately and pass the actual batch size for independent graph accumulation.
No parameter shapes or initialization of the local MolFlow model changed.

CPU and real ROCm tests compare batched outputs with separate single-graph
executions using different atom types, geometries, topologies and padding
masks. Coordinate gradients and every parameter gradient agree within
float32 tolerances. Perturbing only molecule two leaves molecule one's output
unchanged. Both singleton loss forward/backward and the legacy scatter shape,
masked values and gradients are tested.

## Pocket-conditioned sampling instability

The controlled diagnostic `pocket_batch_crosstalk_diagnostic.json` retains the
negative evidence: with two toy training updates and four Euler steps, the old
scalar aggregation produces NaNs at initialization seed 0. Correcting graph
independence alone still gives NaNs at seed 1234, and some finite coordinates
grow to hundreds of thousands. Passing a finiteness check alone did not
establish a stable sampler.

Three additional implementation defects were corrected:

- The advertised zero-initialized velocity head was bypassed by a random
  final equivariant vector projection. That projection now also starts at
  zero, so initial velocity is actually zero and the learned vector head can
  acquire nonzero gradients during training.
- `pocket_embed_scale=0.1` was accepted but ignored. The declared scale now
  multiplies the encoded pocket; invalid scales are rejected and the effective
  scale is recorded in metadata.
- `generate` ignored `GenerationConfig.seed` and left context dropout active
  inside the ODE. It now uses a device-local generator for noise and atom
  sampling, temporarily evaluates both networks in inference mode, and
  restores all original submodule modes even when generation fails. Global
  training RNG state is unchanged. Non-finite coordinates/probabilities fail
  explicitly; no coordinate clipping or NaN replacement hides divergence.

The final controlled check `pocket_sampling_contract_fixed.json` uses the same
toy input molecules and initialization seeds 0, 42, 1234, two training updates,
two generated samples and four Euler steps. All three produce finite outputs;
maximum absolute coordinates are respectively **4.2883, 2.2519, 2.7371**. The
new regression verifies exact zero initial velocity, those three seeded cases,
same-seed atom identity and coordinate agreement, different-seed variation,
global RNG preservation, scale application, and inference-mode restoration.
ROCm atomic summation can differ by a few float32 ULPs; same-seed coordinate
agreement is checked at `atol=rtol=1e-6`, not claimed bitwise deterministic.
These are synthetic optimizer/sampler checks, not molecular-quality evidence.

The old pocket learning test compared the first/last five different noisy
minibatches. Its result depended on test order and changed targets as well as
the trained model. The test now isolates initialization RNG, trains on the
fixed toy molecules with new training noise each update, and evaluates the
same bank of 16 independent noise/time settings before and after training.
The **5% loss-decrease threshold is unchanged**, and pocket-encoder parameter
updates are checked directly. It measures tiny-data optimization and makes no
held-out molecular generalization claim.

## Real QM9 remeasurement after graph independence

The affected converged-OT control was rerun with identical real QM9 sources,
selected 64 training/32 validation molecules, ordered-atom compatibility groups,
seeds 42/0/1234, 30 updates per treatment, initialization protocol, normalization,
common independent-pairing validation objective, and optimizer settings.
Source and selected-row records and the entire experiment protocol match
`r10_ot_qm9_3seed_converged.json`. Only the recorded source hashes for
`models/encoder.py` and `models/velocity_net.py` differ. The Lipman-only sampling
and import changes do not participate in this local-model experiment.

| Seed | OT-off validation | OT-on validation | OT minus off |
|---:|---:|---:|---:|
| 42 | 5.990429 | 5.932154 | -0.058275 |
| 0 | 5.883198 | 5.789574 | -0.093624 |
| 1234 | 5.499146 | 5.443136 | -0.056010 |

Mean paired delta ± sample SD: **−0.069303 ± 0.021093** (three seeds, ddof=1).
All 90 OT couplings use GPU POT log-domain Sinkhorn with no fallback and pass
the unchanged marginal-error gate `<=0.001`; maximum residual is
`0.0002471134066581726`. The entropic objective remains `reg=0.05`, with
`max_iter=2000`. Total wall time was **37.103 s**.

All three observed deltas favor OT in this small rerun. Thirty updates on a
deliberately selected compatible subset do not establish full-QM9 convergence,
general OT superiority, metal-complex performance or improved docking. Earlier
underconverged and graph-mixing reports remain historical evidence; numerical
OT convergence measurements remain useful independently of model batching.

```sh
uv run python molmetal/scripts/r10_ot_qm9_3seed.py \
  --ot-reg 0.05 --ot-max-iter 2000 --max-marginal-error 0.001 \
  --output-prefix molmetal/reports/r10_ot_qm9_graph_independent
```

`r10_ot_qm9_graph_independent.json/.csv/.md` retain all data/runtime hashes,
split/SDF identifiers, losses, backend/device diagnostics and denominators.

## Focused validation

**42 passed, 1 existing skipped** on Python 3.12 / real gfx1101 ROCm:

```sh
uv run pytest -q \
  molmetal/tests/test_lipman_sampling_contract.py \
  molmetal/tests/test_pocket_conditioned_lipman.py \
  molmetal/tests/test_graph_batch_independence.py \
  molmetal/tests/test_velocity_singleton_batch.py \
  molmetal/tests/test_lipman_import_isolation.py \
  molmetal/tests/test_minibatch_ot.py \
  molmetal/tests/test_egnn_velocity_cfg.py \
  molmetal/tests/test_rocm_lipman.py
```

The skipped test is the existing CPU/GPU timing benchmark in the ROCm Lipman
module; no new regression was skipped. Repository-wide regression is managed
separately by the parent agent after this runtime was frozen.
