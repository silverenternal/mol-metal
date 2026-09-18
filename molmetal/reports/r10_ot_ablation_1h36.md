# Round-10 axis B micro-ablation — 1h36 pocket

- Epochs: **50**
- Batch size: **8**, atoms/ligand: **32**, hidden_dim: **16**
- Device: **cpu**, lr: 0.001

## Results

- vanilla_OT wall: 0.25s, first=16.0826, last=5.7896, delta=+10.2930
- minibatch_OT wall: 1.07s, first=16.0826, last=5.7896, delta=+10.2930

## Success criteria

- vanilla loss decreased: **True**
- minibatch loss decreased: **True**
- total wall under 5 min: **True** (total 1.32s)
