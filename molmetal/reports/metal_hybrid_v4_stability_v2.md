# Metal-Hybrid V4 Stability — Round 2 (M-B1 + M-B2 combined)

Master stability report for `molmetal/scripts/train_metal_hybrid_v4.py`
with **all M-B1 (decoupled loss + multi-fidelity pIC50) and M-B2 (EMA +
top-K val-epoch checkpoint ensemble + mixup + SWA) hooks enabled**
(CLI defaults: `--use-ema --use-ensemble --use-mixup --use-swa`).

Seeds: **[42, 133, 256]**, 30 epochs each, batch=8, Ru temporal split,
D-MPNN + real EGNN + Perceiver + LLRD + Kendall (V4 architecture).

Source data: `molmetal/reports/metal_hybrid_v4_stability_v2_results.json`
(per-seed) + auto-generated `metal_hybrid_v4_stability_v2_report.md`.

## TL;DR

- **Test AUC: 0.4857 ± 0.0556** (n=3 seeds: [42, 133, 256]).
- **Test pIC50 MAE: 0.7317** — still **constant** across seeds (reg head
  collapses to `pic50_min=4.0` clamp; `pic50_test_mean ≈ 4.73`).
- Bootstrap 95 % CI: **[0.4327, 0.5436]**.
- **std < 0.06 target HIT** (0.0556 < 0.06); also < 0.10 target.
- V4 baseline std was **0.1346** → new std **0.0556** — a **0.0790
  absolute reduction (58.7 % relative)**.
- D-MPNN multitask baseline (Ru temporal): **0.5135** — V4 delta
  **−0.0278** (−5.4 % relative) — still **below baseline**.

## Per-seed results

| seed | params | best val AUC | best val PR | test AUC | test PR | test pIC50 MAE | wall-clock | avg epoch | test AUC EMA-only | strategy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42  | 1,193,031 | 0.7083 | 0.3861 | **0.5436** | 0.3047 | 0.7317 | 1362.0s | 45.4s | 0.4862 | ensemble |
| 133 | 1,193,031 | 0.7212 | 0.4034 | **0.4808** | 0.2931 | 0.7317 | 1614.9s | 53.8s | 0.4689 | ensemble |
| 256 | 1,193,031 | 0.7162 | 0.3733 | **0.4327** | 0.2788 | 0.7317 | 1113.4s | 37.1s | 0.4327 | ema |

Best-val AUCs are tight (0.708–0.721, range 0.013), confirming
convergence stability across seeds. Test AUCs diverge more (0.43–0.54,
range 0.11), which is the OOD test-set variance — but the std is still
below 0.06.

### 3-seed mean ± std

| metric | mean | std (ddof=1) | previous V4 std | Δ vs. V4 |
| --- | ---: | ---: | ---: | ---: |
| **test AUC** | **0.4857** | **0.0556** | 0.1346 | **−0.0790 (58.7 %)** |
| test PR-AUC | 0.2922 | 0.0130 | — | — |
| best val AUC | 0.7154 | 0.0065 | — | — |
| test pIC50 MAE | 0.7317 | 0.0000 | 0.0000 (constant) | 0 (still constant) |
| wall-clock | 1363.4s | 252.0s | — | — |

### Bootstrap 95 % CI

Resample (n=10 000) of the test AUCs across the three seeds:

- Mean: **0.4857**
- Std (sample, ddof=1): **0.0556**
- 95 % percentile CI: **[0.4327, 0.5436]**

## Comparison vs previous rounds

| run | test AUC mean | std | vs target < 0.06 |
| --- | ---: | ---: | ---: |
| V4 baseline (Ru temporal, no M-B1/M-B2) | 0.5268 | 0.1346 | FAIL |
| M-B1 only (decoupled loss + multi-fidelity) | 0.5327 | 0.0432 | PASS |
| M-B2 only (EMA + ensemble + mixup + SWA) | 0.5035 | 0.0363 | PASS |
| **M-B1 + M-B2 combined (this run)** | **0.4857** | **0.0556** | **PASS** |

Both single-shot fixes achieved std < 0.06; the combined stack is also
below the 0.06 target (0.0556 < 0.06) but lands ~0.04 above either
single fix.  In other words, **M-B1 + M-B2 stack lower mean test AUC
than either alone** — both fixes individually nudge the val selection
toward more aggressive ensembles; stacking them amplifies the
ensemble-driven distribution shift and costs ~0.05 mean AUC.

## Conclusion (vs D-MPNN baseline 0.5135)

The combined M-B1 + M-B2 stack delivers **stable** V4 behaviour
(std 0.0556 < 0.06) but at a lower mean test AUC (0.4857) than the
D-MPNN multitask baseline (0.5135).  We are **NOT** claiming
"stable + above-baseline" yet — the headline metric is the *spread*,
not the *level*.  The Kendall-regularised regression head still
collapses to `pic50_min=4.0` (constant 0.7317 pIC50 MAE across
seeds), and the OOD test set's natural variance pulls ensemble
predictions toward the majority class.

Path to "stable AND above-baseline":
1. Loosen `pic50_min` clamp to 3.0 (currently 4.0) so the reg head
   can express the test distribution.
2. Add active-sample weight 5× (currently 3×) inside M-B1.
3. Re-enable `LossV4MB1.log_s_*` registration in this V4 path
   (currently only registered in the M-B1 branch's training script).
4. Optional: drop SWA tail when ensemble_k=3 + EMA already on —
   the three regularisers may be double-counting weight smoothing.

These are the M-B3 / M-C1 candidates.

## Provenance

- Script: `molmetal/scripts/train_metal_hybrid_v4.py`
  (defaults: `--use-ema --use-ensemble --use-mixup --use-swa`).
- Per-seed JSON: `molmetal/reports/metal_hybrid_v4_stability_v2_results.json`.
- Auto-generated per-run report:
  `molmetal/reports/metal_hybrid_v4_stability_v2_report.md`.
- Checkpoints: `molmetal/checkpoints/metal_hybrid_v4_ru_temporal_seed{42,133,256}.pt`.
- v1 baselines: `molmetal/reports/metal_hybrid_v4_stability_v2_MB1.md`,
  `molmetal/reports/metal_hybrid_v4_stability_v2_MB2.md`.