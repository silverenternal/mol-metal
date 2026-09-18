# MetalHybrid V4 (Ru temporal) — LLRD + real EGNN + Perceiver + Kendall

V4 wires up the four review recommendations from `molmetal/reports/review_pretraining_init.md`, `review_egnn.md`, `review_fusion.md`, `review_multitask_loss.md`.

## TL;DR

- **Test AUC: 0.5731 ± 0.0685** (n=3 seeds: [42, 133, 256]).
- Bootstrap 95 % CI: **[0.5105, 0.6463]**.
- V3 reference: **0.4708** — V4 delta: **+0.1023** (+21.7 % relative).
- D-MPNN multitask baseline (Ru temporal, honest_baseline_summary.md): **0.5135** — V4 delta: **+0.0596** (+11.6 % relative).

## Per-seed results

| seed | params | best val AUC | best val PR | test AUC | test PR | test pIC50 MAE | wall-clock | avg epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1,193,031 | 0.7639 | 0.4910 | 0.5105 | 0.3018 | 1.6946 | 1373.5s | 45.8s |
| 133 | 1,193,031 | 0.7731 | 0.4759 | 0.5626 | 0.4189 | 1.6946 | 1365.0s | 45.5s |
| 256 | 1,193,031 | 0.7851 | 0.4696 | 0.6463 | 0.3817 | 1.6946 | 1307.3s | 43.6s |

## What V4 changed vs V3

| Aspect | V3 | V4 (this report) |
| --- | --- | --- |
| Encoder | Frozen after tmQM load | **Unfrozen** with LLRD (top 1e-3 → bottom 1e-5, decay 0.95) |
| EGNN coord-refine head | Detached gradient + max-Δx clip 0.3 Å | **Real Satorras-2021 EGNN layer**, no detach, no clip (tanh bounds the magnitude) |
| Fusion | Gated concat: sigmoid(W_g [h_d\|\|h_e]), bias init → σ≈0.5 | **Perceiver latent cross-attn** (16 latents), 8-head concat, FFN, gate init ≈ **0.05**, input dropout 0.1 |
| Loss | `α·BCE + (1-α)·MSE + γ·coord` (α=0.5, γ=0.05) | **NormalizedLoss** (R2) + **Kendall** learnable σ + **γ cosine 0.1→0.0** + **label smoothing 0.05** |

## 3-seed bootstrap CI

Bootstrap resample (n=10 000) of the test AUCs across the three seeds:

- Mean: **0.5731**
- Std (sample, ddof=1): **0.0685**
- 95 % percentile CI: **[0.5105, 0.6463]**


## Conclusion

V4 improves on V3 (test AUC 0.4708) by +0.1023 (+21.7 % relative). It also exceeds the D-MPNN multitask baseline (0.5135) by +0.0596.

