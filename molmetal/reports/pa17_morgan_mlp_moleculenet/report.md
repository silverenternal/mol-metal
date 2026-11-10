# Metric Tier 2 — MoleculeNet cross-benchmark (BBBP / BACE / HIV / ClinTox)

**Split:** Bemis-Murcko scaffold (70 / 10 / 20) per Chemprop convention
**Split key:** Bemis-Murcko, stereo-blind (`includeChirality=False`) — the `astartes`/chemprop reference
**Featurization:** Morgan FP (radius=2, n_bits=2048) — ECFP4
**Model:** small MLP (2048 → 256 → 128 → 1) with class-weighted BCE
**Backbone:** morgan_mlp
**Seeds:** 0, 1, 2, 3, 4  

## Chemprop baselines (Tier 3 reference, Wu 2018)

| Task | Chemprop D-MPNN (atom+bond) | Chemprop RF + Morgan FP | **Mol-Metal (this run)** |
|---|---|---|---|
| bbbp | 0.940 | 0.728 | **0.8610 ± 0.0527** (n=5) |
| bace | 0.876 | 0.868 | **0.8500 ± 0.0334** (n=5) |
| hiv | 0.823 | 0.761 | **0.7600 ± 0.0368** (n=5) |
| clintox | 0.894 | 0.832 | **0.7250 ± 0.0471** (n=5) |

**Mean across 4 tasks: 0.7990** (gate ≥ 0.80, Chemprop D-MPNN mean ≈ 0.882).

**Verdict:** ⚠️ BELOW GATE — measured honestly; not paper-promotable as headline Tier-2 metric

## Per-seed detail

### bbbp

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1434 | 412 | 0.779 | 0.9341 | 0.9748 | 1.000 |
| 1 | 1472 | 401 | 0.748 | 0.9008 | 0.9584 | 1.000 |
| 2 | 1450 | 403 | 0.824 | 0.7811 | 0.9241 | 0.900 |
| 3 | 1435 | 409 | 0.812 | 0.8402 | 0.9481 | 0.950 |
| 4 | 1403 | 474 | 0.795 | 0.8486 | 0.9514 | 1.000 |

### bace

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1066 | 299 | 0.485 | 0.8581 | 0.8309 | 0.933 |
| 1 | 1058 | 305 | 0.475 | 0.8331 | 0.8094 | 1.000 |
| 2 | 1058 | 304 | 0.474 | 0.8849 | 0.8739 | 1.000 |
| 3 | 1059 | 303 | 0.419 | 0.7942 | 0.6756 | 0.733 |
| 4 | 1057 | 308 | 0.529 | 0.8798 | 0.8811 | 0.933 |

### hiv

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 28753 | 8296 | 0.040 | 0.7819 | 0.4009 | 0.395 |
| 1 | 28790 | 8226 | 0.031 | 0.6985 | 0.2536 | 0.226 |
| 2 | 28786 | 8225 | 0.040 | 0.7722 | 0.3247 | 0.355 |
| 3 | 28789 | 8226 | 0.040 | 0.7420 | 0.3150 | 0.307 |
| 4 | 28788 | 8227 | 0.031 | 0.8053 | 0.4533 | 0.338 |

### clintox

| Seed | n_train | n_test | pos_rate_test | ROC-AUC | PR-AUC | Hit@5% |
|---|---|---|---|---|---|---|
| 0 | 1039 | 297 | 0.067 | 0.7127 | 0.1883 | 0.133 |
| 1 | 1041 | 296 | 0.061 | 0.6568 | 0.1476 | 0.200 |
| 2 | 1038 | 298 | 0.064 | 0.7270 | 0.2310 | 0.267 |
| 3 | 1035 | 295 | 0.102 | 0.8043 | 0.3851 | 0.600 |
| 4 | 1039 | 297 | 0.081 | 0.7240 | 0.1932 | 0.333 |

## Honest caveats

1. **Model capacity**: a 2048 -> 256 -> 128 -> 1 MLP on Morgan FP has far
   fewer parameters than Chemprop's D-MPNN. Use `--backbone dmpnn` for the
   apples-to-apples architecture comparison on the same split.
2. **Class-imbalance handling**: HIV and ClinTox are severely imbalanced;
   pos_weight alone gives a high-recall bias that depresses PR-AUC.
3. **Scaffold split is OOD**: Chemprop baselines above are scaffold-split
   (Wu 2018 Table 3), so the comparison is protocol-aligned.
4. **Per-task variance**: small tasks (ClinTox, BACE) have wide per-seed
   spread; read the per-seed table above, not the mean alone.

## Files

* `molmetal/scripts/train_moleculenet.py` — driver.
* `molmetal/reports/pa17_morgan_mlp_moleculenet/report.md` — this report.
* `molmetal/reports/pa17_morgan_mlp_moleculenet/metrics.json` — aggregate JSON (per-task x per-seed).

## Baseline provenance (literature-sourced, protocol-differs)

The Chemprop numbers in the table above are **literature-sourced** (`CHEMPROP_DMPNN_BASELINE` / `CHEMPROP_RF_BASELINE`, constant block) and are **NOT reproduced under our protocol**. They are best-of-hyperparameter-search single-task runs; our runs use a shared-trunk multi-task head at fixed hyperparameters. Where our AUC trails the literature value, part of the gap is **tuning, not architecture**. Do NOT present the delta as a pure architecture result, and do NOT adjust our number to match the literature.
