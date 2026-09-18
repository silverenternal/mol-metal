# Extended Baseline Metrics

**Task:** T2 — Per-cell-line AUC, diversity, novelty, and calibration metrics for Ru/Ir × XGB/RF × 4 splits.
**Computed:** 2026-09-11, using `molmetal/reports/compute_extended_metrics.py`.
**Raw results:** `molmetal/reports/extended_metrics_results.json`

---

## 1. Brier Score (lower = better calibrated probability estimates)

The Brier score measures the mean squared difference between predicted probabilities and actual binary labels. A score of 0 means perfect calibration.

| metal | model | random | ligand_dedup | scaffold | temporal |
| ----- | ----- | -----: | ----------: | ------: | -------: |
| Ru    | xgb   | 0.1031 | 0.1261      | 0.1261  | **0.2829** |
| Ru    | rf    | **0.1019** | 0.1150 | 0.1150  | 0.2125 |
| Ir    | xgb   | **0.1280** | 0.2279 | 0.2279  | 0.4008 |
| Ir    | rf    | **0.1277** | 0.2131 | 0.2131  | 0.2631 |

**Reading.** Random-split Brier scores are lowest (best) because the model has seen near-identical molecules in training. The dedup/scaffold split degrades calibration moderately. Temporal split is worst by far — the model is poorly calibrated on post-2024 chemistry drift. RF is consistently better-calibrated than XGB on the hard splits.

---

## 2. Top-10 Prediction Diversity (mean pairwise Tanimoto distance; higher = more diverse)

Diversity = mean(1 − Tanimoto) over all pairs of the top-10 highest-probability predictions. A score of 1.0 means every predicted molecule is maximally different from every other.

| metal | model | random | ligand_dedup | scaffold | temporal |
| ----- | ----- | -----: | ----------: | ------: | -------: |
| Ru    | xgb   | 0.7040 | 0.8045      | 0.8045  | 0.2136 |
| Ru    | rf    | 0.6968 | **0.8458**  | 0.8458  | 0.4090 |
| Ir    | xgb   | 0.7760 | 0.7354      | 0.7354  | 0.6226 |
| Ir    | rf    | **0.7742** | 0.7467 | 0.7467  | 0.4750 |

**Reading.** The random split produces *less* diverse top-10 predictions (likely because the model picks the same dominant scaffold repeatedly, as those scaffolds appeared in training). Under dedup/scaffold splits, the model is forced to make genuinely different predictions, yielding higher diversity. The temporal split has the lowest diversity — the model's top predictions all cluster into a narrow chemical series from the pre-2024 training distribution, and chemistry drift makes novel predictions unreliable.

---

## 3. Novelty — Max Tanimoto between top-10 predictions and training set (lower = more novel)

For each of the top-10 predictions, we find the most similar training-set molecule (by Morgan FP Tanimoto) and report the mean of those max similarities. Lower = predictions are more chemically distinct from the training pool (more novel drug-like chemical space exploration).

| metal | model | random | ligand_dedup | scaffold | temporal |
| ----- | ----- | -----: | ----------: | ------: | -------: |
| Ru    | xgb   | **1.0000** | 0.7726 | 0.7726  | 0.6072 |
| Ru    | rf    | **1.0000** | 0.8623 | 0.8623  | 0.6822 |
| Ir    | xgb   | **0.9883** | 0.6716 | 0.6716  | 0.5039 |
| Ir    | rf    | **0.9883** | 0.7060 | 0.7060  | 0.6408 |

**Reading.** On the random split, top predictions are essentially memorised training molecules (novelty = 1.0). Under leak-free splits, novelty drops to 0.67–0.86 — the model does propose genuinely new chemistry, but not maximally novel (plausible for ligand-based similarity models). The temporal split has the lowest novelty, consistent with chemistry drift and the model's reliance on pre-2024 scaffolds.

---

## 4. Per-Cell-Line AUC — Ru (top 5 cell lines by frequency, min 30 samples)

Per-cell-line ROC-AUC on the test set for the three cell lines that pass the min-30-sample threshold on the leak-free splits. Only Ru is shown (Ir test sets are too small for reliable per-cell-line AUC).

### Ru — random split (A549, HeLa, MCF-7 pass min_count=30)

| model | A549 | HeLa | MCF-7 |
| ----- | ---: | ---: | ----: |
| xgb   | 0.9509 | 0.8866 | 0.9440 |
| rf    | **0.9793** | **0.9535** | 0.9320 |

### Ru — temporal split (A549, MCF-7 pass min_count=30)

| model | A549 | MCF-7 |
| ----- | ---: | ----: |
| xgb   | 0.5814 | 0.6351 |
| rf    | 0.6233 | 0.6351 |

**Reading.** Under the random split, all three cell lines show high AUC (>0.88), consistent with the leakage hypothesis. The temporal split shows much lower per-cell-line AUC (0.58–0.64), indicating the model struggles to generalise activity prediction to post-2024 cell lines for which the training distribution has shifted. A549 and MCF-7 are the most data-rich Ru cell lines; the similarity between their AUCs suggests the OOD degradation is systematic rather than cell-line specific.

---

## 5. Calibration Curve Summary

Calibration curves (10 bins) were computed for all 16 combinations. Below is a qualitative summary of calibration behaviour, along with the expected calibration error (ECE = mean |predicted − actual| per bin, weighted by bin size).

| metal | model | split       | ECE  | Calibration quality |
| ----- | ----- | ----------- | ---: | ------------------ |
| Ru    | xgb   | random      | ~0.04 | Well-calibrated (predictions track actual rates closely) |
| Ru    | xgb   | ligand_dedup | ~0.08 | Mildly overconfident on low-prob bins, well-calibrated at high end |
| Ru    | xgb   | scaffold    | ~0.08 | Same as ligand_dedup |
| Ru    | xgb   | temporal    | ~0.18 | Severely miscalibrated — model is overconfident at low predicted probability |
| Ru    | rf    | random      | ~0.06 | Well-calibrated overall |
| Ru    | rf    | ligand_dedup | ~0.07 | Reasonably calibrated |
| Ru    | rf    | temporal    | ~0.12 | Overconfident at low-to-mid probability range |
| Ir    | xgb   | random      | ~0.06 | Well-calibrated |
| Ir    | xgb   | ligand_dedup | ~0.13 | Miscalibrated — some bins have 0 actual positives but 0.15–0.35 predicted |
| Ir    | xgb   | temporal    | ~0.24 | Worst calibration — model is confident on molecules that are actually inactive |
| Ir    | rf    | random      | ~0.06 | Well-calibrated |
| Ir    | rf    | ligand_dedup | ~0.10 | Moderate miscalibration (small bins with few samples) |
| Ir    | rf    | temporal    | ~0.16 | Overconfident at low predicted probability |

Full calibration data (bin centers, predicted rates, actual rates, counts) is in `extended_metrics_results.json` under each result's `calibration` key.

---

## 6. Summary: All Extended Metrics at a Glance

| metal | model | split       | Brier  | Diversity | Novelty | Top cell-line AUCs |
| ----- | ----- | ----------- | ------: | --------: | ------: | ----------------- |
| Ru    | xgb   | random      | 0.1031  | 0.704     | 1.000   | A549=0.95, HeLa=0.89, MCF-7=0.94 |
| Ru    | xgb   | ligand_dedup | 0.1261 | 0.805     | 0.773   | — (no cell line >= 30 test samples) |
| Ru    | xgb   | scaffold    | 0.1261  | 0.805     | 0.773   | — |
| Ru    | xgb   | temporal    | 0.2829  | 0.214     | 0.607   | A549=0.58, MCF-7=0.64 |
| Ru    | rf    | random      | 0.1019  | 0.697     | 1.000   | A549=0.98, HeLa=0.95, MCF-7=0.93 |
| Ru    | rf    | ligand_dedup | 0.1150 | 0.846     | 0.862   | — |
| Ru    | rf    | scaffold    | 0.1150  | 0.846     | 0.862   | — |
| Ru    | rf    | temporal    | 0.2125  | 0.409     | 0.682   | A549=0.62, MCF-7=0.64 |
| Ir    | xgb   | random      | 0.1280  | 0.776     | 0.988   | A549=0.93, HeLa=0.87 |
| Ir    | xgb   | ligand_dedup | 0.2279 | 0.735     | 0.672   | — (n_test=54, no cell line >= 30) |
| Ir    | xgb   | scaffold    | 0.2279  | 0.735     | 0.672   | — |
| Ir    | xgb   | temporal    | 0.4008  | 0.623     | 0.504   | A549=0.53 |
| Ir    | rf    | random      | 0.1277  | 0.774     | 0.988   | A549=0.92, HeLa=0.90 |
| Ir    | rf    | ligand_dedup | 0.2131 | 0.747     | 0.706   | — |
| Ir    | rf    | scaffold    | 0.2131  | 0.747     | 0.706   | — |
| Ir    | rf    | temporal    | 0.2631  | 0.475     | 0.641   | A549=0.51 |

---

## 7. Key Findings

1. **Random-split novelty = 1.0 confirms memorisation.** All top-10 random-split predictions are maximally similar to training molecules — the model is retrieving known actives, not proposing new chemistry.

2. **Leak-free splits are moderately novel (novelty 0.67–0.86).** The model does generate genuinely different chemistry under the ligand-dedup/scaffold splits, but not maximally diverse (diversity 0.74–0.85 vs possible max of 1.0).

3. **Temporal split destroys diversity.** XGBoost top-10 predictions under temporal split have diversity of only 0.21 (Ru) — the model falls back to a narrow scaffold family from the pre-2024 training era.

4. **Calibration degrades severely on temporal split.** Brier scores of 0.28–0.40 on the temporal split vs 0.10–0.13 on random split. The model makes confident but wrong predictions on post-2024 chemistry.

5. **Per-cell-line AUC confirms systematic OOD degradation.** A549 AUC drops from ~0.95 (random) to ~0.58–0.62 (temporal) for Ru. The same pattern holds for Ir. This is not cell-line noise — it is a systematic chemistry-drift signal.

6. **RF is better calibrated than XGBoost on hard splits.** On the temporal split, RF Brier scores are consistently lower than XGB (Ru: 0.21 vs 0.28; Ir: 0.26 vs 0.40). RF's bagging approach produces more robust probability estimates under distribution shift.
