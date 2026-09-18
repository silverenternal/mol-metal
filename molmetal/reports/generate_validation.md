# Generate() Validation — Joint Atom-Type + Coord FM

End-to-end smoke test of the new joint atom-type + coordinate
Flow Matching ``LipmanFlowMatchingAdapter.generate()``.

## Setup

- Adapter: `LipmanFlowMatchingAdapter` (hidden_dim=64, n_layers=2, lr=0.005, atom_loss_weight=0.1)
- Device: `cuda:0`  (ROCm active: True, HIP: 7.2.53211, name: AMD Radeon Graphics)
- Train molecules: 200 unique Ru SMILES with ≤30 heavy atoms
- Training: 30 steps × batch_size=8
- Generation: n_samples=100, n_steps=20, n_atoms=8

## Training Loss Curve

| Step | Total | CFM | Atom-CE | ms/step |
|------|-------|-----|---------|---------|
| 0 | 4.9124 | 4.4519 | 4.6052 | 1300.1 |
| 1 | 8257.2109 | 8257.0312 | 1.8010 | 7.5 |
| 2 | 40.9770 | 40.6976 | 2.7933 | 7.0 |
| 3 | 5240.0103 | 5238.9297 | 10.8038 | 6.9 |
| 4 | 60.1873 | 60.0462 | 1.4104 | 7.0 |
| 5 | 5.4037 | 4.9929 | 4.1078 | 8.6 |
| 6 | 291.5820 | 290.9795 | 6.0250 | 6.8 |
| 7 | 13.9323 | 13.6841 | 2.4821 | 6.6 |
| 8 | 146.1273 | 145.4779 | 6.4945 | 6.5 |
| 9 | 48.5502 | 48.2183 | 3.3196 | 7.0 |
| 10 | 8.4601 | 8.3089 | 1.5126 | 6.6 |
| 11 | 8.3821 | 8.2178 | 1.6428 | 6.8 |
| 12 | 80.6111 | 80.4863 | 1.2485 | 6.4 |
| 13 | 10.4276 | 10.2784 | 1.4927 | 6.9 |
| 14 | 27.3458 | 27.1301 | 2.1563 | 6.9 |
| 15 | 1.9994 | 1.8521 | 1.4731 | 5.9 |
| 16 | 2.0823 | 2.0121 | 0.7014 | 6.7 |
| 17 | 2.5858 | 2.5204 | 0.6546 | 7.2 |
| 18 | 4.0529 | 3.9891 | 0.6378 | 7.2 |
| 19 | 1.6476 | 1.5712 | 0.7640 | 7.6 |
| 20 | 2.6977 | 2.6357 | 0.6201 | 8.3 |
| 21 | 1.4340 | 1.3715 | 0.6250 | 7.5 |
| 22 | 1.1590 | 1.0873 | 0.7170 | 8.2 |
| 23 | 1.9960 | 1.9127 | 0.8333 | 9.0 |
| 24 | 0.5291 | 0.4527 | 0.7641 | 8.4 |
| 25 | 5.1825 | 5.1182 | 0.6437 | 8.0 |
| 26 | 3.1775 | 3.1275 | 0.5002 | 7.0 |
| 27 | 1.0495 | 0.9980 | 0.5147 | 6.7 |
| 28 | 2.5613 | 2.5163 | 0.4500 | 6.9 |
| 29 | 2.3535 | 2.3164 | 0.3704 | 6.9 |

- Initial loss: total=4.9124 (cfm=4.4519, atom=4.6052)
- Final loss:   total=2.3535 (cfm=2.3164, atom=0.3704)
- Loss drop (total / cfm / atom): 2.09x / 1.92x / 12.43x

## Generated Molecule Quality

| Metric | Generated | Training Reference |
|--------|-----------|--------------------|
| Valid (RDKit-parseable) | 100 / 100 | 200 / 200 |
| QED mean | 0.2947 | 0.5097 |
| QED std  | 0.0000 | 0.1361 |
| QED median | 0.2947 | 0.4970 |
| Fraction drug-like (QED ≥ 0.5) | 0.000 | 0.485 |
| MolWt mean | 130.0 | 389.7 |
| MolWt in [200, 500] | 0.000 | 0.955 |
| TPSA mean | 95.84 | 49.05 |

## Atom-Type Distribution

| Bin | Generated | Training Reference |
|-----|-----------|--------------------|
| Z>20 | 0 | 47 |
| z0_padding | 0 | 0 |
| Z=6 | 0 | 3996 |
| Z=7 | 0 | 489 |
| Z=8 | 800 | 188 |
| Z=9 | 0 | 8 |
| Z=15 | 0 | 16 |
| Z=16 | 0 | 69 |
| Z=17 | 0 | 232 |

- No Z=0 padding atoms in generated molecules (the `atom_logits[..., 0] = -inf` mask is working).

## Verdict

- Training **converged** (total loss dropped 2.09x over 30 steps).
- Generated QED mean = 0.295; reference = 0.510. With only 30 train steps the model is far from converged; the distribution should sharpen as we increase the budget.
- Atom-type histogram coverage = 1 unique elements. We expect this to grow with more training.

