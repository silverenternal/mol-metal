# R1 — Cross-attention fusion V3 fix

V2 (plain cross-attention) collapsed on the Ru temporal test set (test AUC 0.3764 vs V1 concat's 0.6617). V3 adds dropout + residual scaling + equivariant init to the same backbone. All three variants trained on the identical Ru temporal split.

- Backbone: D-MPNN (3 layers, hidden=128) + EGNN (3 layers, hidden=128)
- Dual head: pIC50 regression + active classification (alpha=0.5)
- Optimiser: Adam, lr=1e-3, cosine schedule over --epochs
- Filter: max_atoms=50 (post-filter n same as W2 ablation)

## Results

| variant | params | Ru val AUC | Ru val PR | Ru test AUC | Ru test PR | test pIC50 MAE | avg epoch time | wall-clock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v1_concat | 943,302 | 0.6504 | 0.3566 | 0.6239 | 0.3477 | 0.7317 | 49.1s | 736.8s |
| v3_crossattn | 968,135 | 0.7234 | 0.4085 | 0.5582 | 0.5132 | 0.7317 | 39.3s | 590.0s |

## Per-epoch validation curves


### v1_concat (params=943,302)

| epoch | train_loss | val_AUC | val_PR |
| ---: | ---: | ---: | ---: |
| 1 | 0.6788 | 0.5642 | 0.2398 |
| 2 | 0.6697 | 0.6114 | 0.3376 |
| 3 | 0.6611 | 0.6504 | 0.3566 |
| 4 | 0.6513 | 0.6371 | 0.3478 |
| 5 | 0.6479 | 0.6483 | 0.3625 |
| 6 | 0.6451 | 0.6503 | 0.3677 |
| 7 | 0.6431 | 0.6457 | 0.3502 |
| 8 | 0.6375 | 0.6171 | 0.3324 |
| 9 | 0.6371 | 0.6268 | 0.3460 |
| 10 | 0.6340 | 0.6189 | 0.3409 |
| 11 | 0.6294 | 0.6024 | 0.3225 |
| 12 | 0.6264 | 0.6075 | 0.3347 |
| 13 | 0.6230 | 0.6134 | 0.3443 |
| 14 | 0.6223 | 0.6135 | 0.3406 |
| 15 | 0.6201 | 0.6114 | 0.3401 |

### v3_crossattn (params=968,135)

| epoch | train_loss | val_AUC | val_PR |
| ---: | ---: | ---: | ---: |
| 1 | 6.8743 | 0.4928 | 0.2068 |
| 2 | 0.8479 | 0.5518 | 0.2594 |
| 3 | 0.9909 | 0.6599 | 0.3394 |
| 4 | 0.6498 | 0.6688 | 0.3382 |
| 5 | 0.7182 | 0.6469 | 0.2853 |
| 6 | 0.6253 | 0.7051 | 0.3629 |
| 7 | 0.6154 | 0.7234 | 0.4085 |
| 8 | 0.6085 | 0.6906 | 0.3573 |
| 9 | 0.6028 | 0.6695 | 0.4020 |
| 10 | 0.5981 | 0.6909 | 0.4027 |
| 11 | 0.5929 | 0.7096 | 0.3954 |
| 12 | 0.5872 | 0.7029 | 0.3991 |
| 13 | 0.5825 | 0.7113 | 0.4032 |
| 14 | 0.5788 | 0.7138 | 0.4230 |
| 15 | 0.5749 | 0.7163 | 0.4265 |

## Conclusion

**Winner on Ru temporal test set: `v1_concat` (test AUC=0.6239)**

V3 vs V1 (concat): -0.0658 test AUC (-10.5% relative). V3 closes the V2 gap of +0.6239 by +0.0658.

Margin over `v3_crossattn` (test AUC=0.5582): +0.0658 (+11.8% relative).

- `v1_concat`: 943,302 params, 49.1s/epoch, test AUC=0.6239, test PR=0.3477
- `v3_crossattn`: 968,135 params, 39.3s/epoch, test AUC=0.5582, test PR=0.5132
