# Round-10 axis B micro-ablation — synthetic coordinates

Historical command (pre-fix source): `uv run python molmetal/scripts/r10_ot_ablation_1h36.py --epochs 50 --b 8 --n 32 --hidden-dim 16 --seed 0 --device cuda:0 --output-prefix molmetal/reports/r10_ot_rocm_measured_seed0`

- Epochs: **50**
- Batch size: **8**, atoms/ligand: **32**, hidden_dim: **16**
- Device: **cuda:0**, lr: 0.001
- Seed: **0**, environment: {'python': '3.12.13', 'torch': '2.14.0+rocm7.2', 'hip': '7.2.53211', 'triton_rocm': '3.8.0', 'pot': '0.9.7.post1', 'scipy': '1.18.1', 'gpu': 'AMD Radeon Graphics', 'gpu_arch': 'gfx1101'}
- Data: synthetic Gaussian coordinates; no real 1h36 data loaded.
- The legacy `vanilla_OT` label means independent ordering (OT disabled), not full-batch optimal transport.
- OT groups: **pairs of samples** (batch_idx = arange(B) // 2)

## Results

- vanilla_OT wall: 0.78s, first=17.7093, last=6.6131, delta=+11.0962
- minibatch_OT wall: 1.19s, first=18.3765, last=6.1934, delta=+12.1831

- independent: loss mean=7.373600, std across steps=2.253787, last-10 mean=6.074992, re-paired fraction=0.000, target velocity MSE=2.007865
- minibatch: loss mean=7.131064, std across steps=2.241873, last-10 mean=6.055853, re-paired fraction=0.460, target velocity MSE=1.934749

Single-seed training smoke only. Step-to-step std is not uncertainty across seeds. Loss/transport-cost changes do not establish a molecular quality or Vina improvement, nor the full-batch OT parity criterion.

## Success criteria

- vanilla loss decreased: **True**
- minibatch loss decreased: **True**
- total wall under 5 min: **True** (total 1.97s)

## Historical backend correction

This run preceded the POT API fix. POT 0.9.7.post1 has no `ot.ot` attribute, so the old `_pot.ot.sinkhorn` call raised `AttributeError` and used SciPy Hungarian on CPU. This is not evidence for a GPU Sinkhorn run. See `r10_ot_rocm_sinkhorn_seed0.md` for the measured run after the fix.
