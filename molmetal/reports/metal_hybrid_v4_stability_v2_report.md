# MetalHybrid V4 (Ru temporal) — LLRD + real EGNN + Perceiver + Kendall

V4 wires up the four review recommendations from `molmetal/reports/review_pretraining_init.md`, `review_egnn.md`, `review_fusion.md`, `review_multitask_loss.md`.

## TL;DR

- **Test AUC: 0.4857 ± 0.0556** (n=3 seeds: [42, 133, 256]).
- Bootstrap 95 % CI: **[0.4327, 0.5436]**.
- V3 reference: **0.4708** — V4 delta: **+0.0149** (+3.2 % relative).
- D-MPNN multitask baseline (Ru temporal, honest_baseline_summary.md): **0.5135** — V4 delta: **-0.0278** (-5.4 % relative).

## Per-seed results

| seed | params | best val AUC | best val PR | test AUC | test PR | test pIC50 MAE | wall-clock | avg epoch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1,193,031 | 0.7083 | 0.3861 | 0.5436 | 0.3047 | 0.7317 | 1362.0s | 45.4s |
| 133 | 1,193,031 | 0.7212 | 0.4034 | 0.4808 | 0.2931 | 0.7317 | 1614.9s | 53.8s |
| 256 | 1,193,031 | 0.7162 | 0.3733 | 0.4327 | 0.2788 | 0.7317 | 1113.4s | 37.1s |

## What V4 changed vs V3

| Aspect | V3 | V4 (this report) |
| --- | --- | --- |
| Encoder | Frozen after tmQM load | **Unfrozen** with LLRD (top 1e-3 → bottom 1e-5, decay 0.95) |
| EGNN coord-refine head | Detached gradient + max-Δx clip 0.3 Å | **Real Satorras-2021 EGNN layer**, no detach, no clip (tanh bounds the magnitude) |
| Fusion | Gated concat: sigmoid(W_g [h_d\|\|h_e]), bias init → σ≈0.5 | **Perceiver latent cross-attn** (16 latents), 8-head concat, FFN, gate init ≈ **0.05**, input dropout 0.1 |
| Loss | `α·BCE + (1-α)·MSE + γ·coord` (α=0.5, γ=0.05) | **NormalizedLoss** (R2) + **Kendall** learnable σ + **γ cosine 0.1→0.0** + **label smoothing 0.05** |

## 3-seed bootstrap CI

Bootstrap resample (n=10 000) of the test AUCs across the three seeds:

- Mean: **0.4857**
- Std (sample, ddof=1): **0.0556**
- 95 % percentile CI: **[0.4327, 0.5436]**


## Conclusion

V4 improves on V3 (test AUC 0.4708) by +0.0149 (+3.2 % relative). It also falls below the D-MPNN multitask baseline (0.5135) by -0.0278.

