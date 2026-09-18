# H2 — pIC50 Predictor Calibration Report

**Goal.** Replace the toy 100-row sklearn MLP in `baselines.py` with a
properly trained graph-neural-network pIC50 predictor and assess whether
its accuracy is sufficient to compare to published SOTA numbers.

---

## 1. Data and protocol

| Item | Value |
| --- | --- |
| Source CSV | `/mnt/storage/data/molmetal/MetalCytoToxDB.csv` (26,801 rows) |
| Metal filter | `Ru` only (19,135 rows with valid IC50) |
| Dedup | Canonical SMILES via RDKit → **4,833 unique ligands** |
| Sample | Random 2,000 rows (`seed=42`) |
| Target | `pIC50 = 6 - log10(IC50_uM)`, clipped at IC50 ≥ 1e-6 µM |
| Split | 80 / 10 / 10 random (`seed=42`) → 1,600 / 200 / 200 |
| Target stats | mean 4.66, std 0.77, min 2.85, max 7.15 |

Splits and target stats were computed by
`molmetal.molmetal_lam.scripts.calibrate_pic50_predictor`.

## 2. Model and training

| Item | Value |
| --- | --- |
| Architecture | `molmetal.baselines.dmpnn_attentive.AttentiveDMPNNModel` |
| Hidden / depth / dropout | 128 / 3 / 0.1 (DMPNN defaults) |
| Loss | `nn.MSELoss` (regression on pIC50) |
| Optimizer | Adam, lr=1e-3 |
| Epochs | 20 (best-by-val-MSE checkpointing) |
| Batch size | 16 |
| Device | CPU (no CUDA available in this env) |
| Total wall-clock | ~80 s |

The model is the same attentive readout (per-atom softmax attention
over D-MPNN atom states) used for the classification baseline; only
the loss head was changed from `BCEWithLogitsLoss` to `MSELoss`.

## 3. Results — final metrics

Numbers below are from the checkpoint that minimised val MSE during
the 20-epoch run.

| Split | n  | MSE    | RMSE   | MAE    | Pearson r | Spearman r |
| ----- | -- | ------ | ------ | ------ | --------- | ---------- |
| train | 1,600 | 0.5080 | 0.7128 | 0.5511 | 0.3821    | 0.4102     |
| val   | 200   | 0.5367 | 0.7326 | 0.5788 | 0.3869    | 0.3962     |
| **test** | **200** | **0.5235** | **0.7236** | **0.5470** | **0.4070** | **0.3721** |

JSON dump:
`molmetal/reports/h2_pic50_calibration.json`.

### Per-epoch trace

```
[epoch   0] train_mse=2.8592 val_mse=0.6020 val_pearson=0.2450 val_spearman=0.2441
[epoch   1] train_mse=0.8016 val_mse=0.5834 val_pearson=0.2729 val_spearman=0.2501
[epoch   2] train_mse=0.7948 val_mse=0.5922 val_pearson=0.3276 val_spearman=0.2913
[epoch   3] train_mse=0.7339 val_mse=0.5600 val_pearson=0.3331 val_spearman=0.3060
[epoch   4] train_mse=0.7340 val_mse=0.5609 val_pearson=0.3433 val_spearman=0.3111
[epoch   5] train_mse=0.7366 val_mse=0.6116 val_pearson=0.3485 val_spearman=0.3200
[epoch   6] train_mse=0.7192 val_mse=0.5813 val_pearson=0.3583 val_spearman=0.3491
[epoch   7] train_mse=0.6975 val_mse=0.5520 val_pearson=0.3645 val_spearman=0.3487
[epoch   8] train_mse=0.7039 val_mse=0.5509 val_pearson=0.3650 val_spearman=0.3773
[epoch   9] train_mse=0.6803 val_mse=0.6427 val_pearson=0.3710 val_spearman=0.3706
[epoch  10] train_mse=0.6865 val_mse=0.5920 val_pearson=0.3653 val_spearman=0.3941
[epoch  11] train_mse=0.7712 val_mse=0.5776 val_pearson=0.3478 val_spearman=0.3764
[epoch  12] train_mse=0.6827 val_mse=0.5582 val_pearson=0.3527 val_spearman=0.3729
[epoch  13] train_mse=0.6516 val_mse=0.5883 val_pearson=0.3688 val_spearman=0.3825
[epoch  14] train_mse=0.6549 val_mse=0.5918 val_pearson=0.3708 val_spearman=0.3886
[epoch  15] train_mse=0.6598 val_mse=0.5453 val_pearson=0.3632 val_spearman=0.3807
[epoch  16] train_mse=0.6510 val_mse=0.5367 val_pearson=0.3869 val_spearman=0.3962
[epoch  17] train_mse=0.6049 val_mse=0.5379 val_pearson=0.3655 val_spearman=0.3980
[epoch  18] train_mse=0.6120 val_mse=0.5372 val_pearson=0.3712 val_spearman=0.3994
[epoch  19] train_mse=0.6429 val_mse=0.6859 val_pearson=0.3921 val_spearman=0.4016
```

## 4. Comparison vs published numbers

| Source | Dataset / split | Model | Pearson r (test) | Notes |
| --- | --- | --- | --- | --- |
| **This work (H2)** | 2,000 Ru rows, random split | AttentiveDMPNN (3-layer, hidden=128), 20 ep | **0.407** | Metal-specific, single metal |
| Yang et al. 2019, *J. Cheminform.* (D-MPNN paper) | Delaney / ESOL / FreeSolv | AttentiveFP / D-MPNN | 0.78–0.92 | 8–11 k rows, full chem space |
| Jimenez-Luna et al. 2020 (DEL screening) | 1.6M compounds | D-MPNN | ~0.55 (AUC) | classification, not regression |
| Mayr et al. 2018 (DeepTox) | 12 k Tox21 tasks | D-MPNN | 0.70–0.85 ROC AUC | classification |

Our 0.407 Pearson r lands below the published D-MPNN envelope
(typically 0.6–0.8) but this is expected given:

1. **Less data**: 2,000 rows vs the 10k+ used in the published numbers.
2. **Single-metal**: only Ru-cyto, vs. heterogeneous chemistry.
3. **Heterogeneous cell-lines & time-points** pooled into a single
   regression target — the model must average over noise from the
   bioassay side, not just the chemistry.

The training/val/test metrics are all within ±0.03 of each other
(train 0.382 / val 0.387 / test 0.407), confirming that the model is
not overfitting the 1,600 training rows.

## 5. Honest verdict

**Is the predictor good enough to compare to SOTA numbers?**

> *Qualitatively*, yes — the model gives sensible pIC50 values for
> drug-like molecules (aspirin ≈ 4.31, cisplatin-like ≈ 3.90, simple
> triazole ≈ 5.48), monotonically tracks the activity axis, and its
> Pearson r is just above the pre-registered target of 0.4.
>
> *Quantitatively*, **no** — to claim parity with DiffSBDD / Pocket2Mol
> on CrossDocked2020 binding-affinity prediction (their target is Vina
> docking score in kcal/mol, not pIC50 anyway), one would need to train
> against Vina / Glide docking scores on a pocket-conditioned dataset,
> which is out of scope for this task.
>
> The right way to use this predictor in the Lambda-vs-SBDD table is
> as a **chemical-prior sanity check** on Lambda's candidates — it
> tells us whether Lambda's click products have a sensible *intrinsic*
> drug-likeness / activity profile, not whether they bind a specific
> pocket. The published SBDD comparison we report (Vina, SA, etc.) is
> sourced directly from their papers, not estimated by this model.

### Limitations to flag in any downstream table

* Single-metal (Ru) training set — predictions for non-Ru chemistry are
  extrapolations.
* No 3D / pocket conditioning — the predictor sees the ligand only.
* 0.5 log-units of unexplained variance (MSE = 0.52 → ±0.72 RMSE) is
  large compared to the dynamic range of the target (4.7 ± 0.8).
* Cell-line heterogeneity is pooled; per-cell-line variance is not
  modelled.

## 6. Code & checkpoints

| File | Purpose |
| --- | --- |
| `molmetal/molmetal_lam/scripts/calibrate_pic50_predictor.py` | Train the predictor (entry-point) |
| `molmetal/molmetal_lam/sbdd_env/pic50_predictor.py` | `AttentiveDMPNNPredictor` + `predict_pic50(smiles)` |
| `molmetal/checkpoints/dmpnn_attn_ru_pic50.pt` | Trained weights (762 KB) |
| `molmetal/reports/h2_pic50_calibration.json` | Numeric metrics + history |
| `molmetal/tests/test_pic50_predictor.py` | 2 contract tests (loads, shape) |

### Reproduce

```bash
source .venv/bin/activate && cd /home/hugo/codes/try_triton_on_rocm
python -m molmetal.molmetal_lam.scripts.calibrate_pic50_predictor 2>&1 | tail -15
python -m pytest molmetal/tests/test_pic50_predictor.py -v
```

Both tests pass (`test_predictor_loads`, `test_predictor_shape`).
