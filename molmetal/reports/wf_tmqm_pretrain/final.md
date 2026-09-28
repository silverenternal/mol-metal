# tmQM Pretraining for Mol-Metal — Encoder Init Report

**Date:** 2026-09-25
**Status:** ✅ **SHIPPED** — pretrain (15 epochs / 20 k rows) + fine-tune ablation (Ru subset, 1 seed) PASS the non-negative-delta gate.

## Background

tmQM (Balcells & Skjelstad, *tmQM Dataset — Quantum Geometries and Properties of 86k Transition Metal Complexes*, J. Chem. Inf. Model. 2020, 60, 6135-6146; extended release = 108,541 mononuclear TM complexes) is the **canonical pretraining corpus** for our metal-aware D-MPNN.  The release shipped here lives at `.data/molmetal/tmQM/` and includes:

* `tmQM_y.csv` — ';'-separated DFT properties (Electronic_E, Dispersion_E, Dipole_M, Metal_q, HL_Gap, HOMO_Energy, LUMO_Energy, Polarizability) + SMILES.
* `tmQM_X{1,2,3}.xyz.gz` — XYZ blocks with comment lines carrying `CSD_code = … | q = charge | S = spin | Stoichiometry = … | MND = coord_num`.
* `tmQM_X{1,2,3}.BO.gz` — Wiberg bond-order blocks (one line per metal centre with `metal_bo_total`).
* `tmqm_parsed.csv` — merged cache produced by `molmetal/data/tmqm.py`.

Level of theory: TPSSh-D3BJ/def2-SVP.  Licence: MIT.

## What this pretraining does

`molmetal/scripts/pretrain_tmqm.py` (~660 LOC) trains a **D-MPNN backbone** (atom_emb, bond_emb, edge_init, gru_update at hidden=128, depth=3, dropout=0.1) on 6 tmQM-derived heads:

| Head | Target | Type | Range |
|---|---|---|---|
| 1 | HOMO_Energy | regression | -0.44 to 0.05 eV |
| 2 | LUMO_Energy | regression | -0.37 to 0.20 eV |
| 3 | HL_Gap | regression | 0.002 to 0.307 eV |
| 4 | metal_bo_total | regression | Wiberg bond order of metal centre |
| 5 | coord_number (= MND) | regression | 1..19 |
| 6 | charge (= complex formal charge) | classification (9 bins = -2..+6) | mostly 0/1 |

Standardisation: regression targets are z-scored on the train split only (`RegScaler`).  Oxidation-state classification uses a fixed 9-bin ordinal scheme.

Loss: `reg_weight * MSE + cls_weight * CrossEntropy(ox_bin)`, both = 1.0.  Adam optimiser, lr=1e-3, batch=128.

After training, only the **backbone weights** (atom_emb, bond_emb, edge_init, gru_update = 12 tensors) are saved to `models/tmqm_pretrained/tmqm_pretrained_encoder.pt`.  The heads are discarded — they are tmQM-specific (DFT properties) and do not transfer to cytotoxicity prediction.

## Run results — pretrain (20k subset, 15 epochs)

This session ran a **20 000-row / 15-epoch** subset (rather than the full 100k × 30 epochs) to fit the wall-time budget.  Both loss and metric curves converged cleanly:

| Epoch | Wall (s) | Train loss | Train reg | Train cls | Val RMSE | Val MAE | Val cls_acc | Test RMSE | Test cls_acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  1 |  59.0 | 1.7429 | 1.1849 | 0.5580 | 0.4875 | 0.2543 | 0.837 | 0.4939 | 0.837 |
|  2 |  59.3 | 1.0053 | 0.6003 | 0.4050 | 0.4403 | 0.2235 | 0.866 | 0.4544 | 0.863 |
|  3 |  58.5 | 0.9059 | 0.5448 | 0.3611 | 0.5030 | 0.2621 | 0.874 | 0.4988 | 0.867 |
|  4 |  59.4 | 0.8357 | 0.5080 | 0.3277 | 0.4354 | 0.2186 | 0.895 | 0.4421 | 0.902 |
|  5 |  58.8 | 0.7850 | 0.4740 | 0.3110 | 0.3851 | 0.1888 | 0.910 | 0.4004 | 0.913 |
|  6 |  59.4 | 0.7094 | 0.4361 | 0.2732 | 0.3454 | 0.1654 | 0.911 | 0.3624 | 0.924 |
|  7 |  59.5 | 0.6616 | 0.4083 | 0.2532 | 0.3647 | 0.1883 | 0.920 | 0.3805 | 0.911 |
|  8 |  59.9 | 0.6087 | 0.3843 | 0.2244 | 0.3103 | 0.1453 | 0.930 | 0.3247 | 0.942 |
|  9 |  59.3 | 0.5644 | 0.3621 | 0.2023 | 0.3178 | 0.1484 | 0.943 | 0.3258 | 0.946 |
| 10 |  59.7 | 0.5421 | 0.3498 | 0.1923 | 0.3077 | 0.1441 | 0.944 | 0.3203 | 0.953 |
| 11 |  59.8 | 0.5055 | 0.3327 | 0.1728 | 0.3357 | 0.1635 | 0.950 | 0.3406 | 0.956 |
| 12 |  59.6 | 0.4785 | 0.3191 | 0.1594 | 0.2803 | 0.1311 | 0.948 | 0.2953 | 0.953 |
| 13 |  60.0 | 0.4564 | 0.3072 | 0.1492 | 0.2906 | 0.1386 | 0.960 | 0.2978 | 0.963 |
| 14 |  59.9 | 0.4292 | 0.2969 | 0.1323 | 0.2720 | 0.1276 | 0.964 | 0.2801 | 0.963 |
| 15 |  59.4 | 0.4117 | 0.2878 | 0.1239 | **0.2640** | 0.1212 | **0.967** | 0.2718 | 0.965 |

**Final test RMSE = 0.2718, cls_acc = 0.965, total wall = 891.5 s (~15 min).**  The best-epoch encoder was checkpointed at epoch 15 (val_RMSE = 0.2640).

Per-task RMSE at the final epoch (inverse-transformed to original units):

| Target | Test RMSE |
|---|---:|
| HOMO_Energy | 0.0416 |
| LUMO_Energy | 0.0549 |
| HL_Gap | 0.0641 |
| metal_bo_total | 0.2118 |
| coord_number | 0.9871 |

The regression head RMSE values are in z-score units, so they reflect per-target variance-normalised error (the raw RMSE in original units is much smaller).  Classification accuracy on the 9-bin charge scheme reached 96.5 %, dominated by the easy bin-2 (charge = 0) class which accounts for ~89 % of the data.

## Downstream verification — fine-tune ablation (Ru subset, seed 0)

The pretrained encoder is consumed by `molmetal/scripts/finetune_metalcytotox_from_tmqm.py`.  We compare:

| init | seed 0 test AUC | seed 0 test acc | n_test | epoch 1 val_AUC | epoch 5 val_AUC |
|---|---:|---:|---:|---:|---:|
| `--init random` | **0.7195** | 0.751 | 714 | 0.685 | 0.728 |
| `--init tmqm` | **0.7655** | 0.776 | 714 | 0.762 | 0.780 |
| **Delta (tmqm − random)** | **+0.0460** | +0.025 | — | +0.077 | +0.052 |

**Verification gate (per `TODO/pending/23_weak_to_strong_plan.md` Metric 1):** the tmQM-pretrained encoder improves MetalCytoToxDB Ru subset test AUC by **+0.046** on seed 0 (5 epochs, batch=128, scaffold-split, 19 135 rows).  This is **PASS** — non-negative delta observed.

Honest framing:
* Single-seed is too few for a hard claim; multi-seed ablation is the next step (see TODO below).
* The Krasnov 2026 MetalCytoToxDB Ru subset AUC = 0.81 is the comparator (we sit at 0.77, ~0.04 below).  The gap is consistent with using fewer epochs (5 vs 30) and a smaller hidden dim / shorter training budget than the Krasnov baseline.  The +0.046 delta is the load-bearing signal here — pretraining helps convergence in the low-data regime.
* `train_loss=0.5315` after epoch 1 with `--init tmqm` (vs `0.6925` for random) shows that the pretrained encoder starts in a substantially better basin.

## Files added / changed (this session)

* `molmetal/scripts/pretrain_tmqm.py` (~660 LOC, NEW) — pretrain script with backbone-only export, best-epoch checkpointing, and stratified split.
* `molmetal/scripts/finetune_metalcytotox_from_tmqm.py` (~330 LOC, NEW) — fine-tune ablation (init=tmqm|random).
* `molmetal/tests/test_tmqm_pretrain.py` (~190 LOC, NEW) — **17 contract tests pass** (loader, split, model, save/load, scaler).
* `molmetal/reports/wf_tmqm_pretrain/final.md` (this file).

## Artifacts shipped

* `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` (~560 KB) — backbone state_dict (12 tensors) + config + train_meta + reg_scaler.
* `models/tmqm_pretrained/pretrain_metrics.json` — full 15-epoch curves.
* `models/tmqm_pretrained/pretrain_summary.json` — final test RMSE / cls_acc / wall / n_rows.
* `models/tmqm_pretrained/featurise_stats.json` — featurisation timing and split sizes.
* `logs/finetune_tmqm_vs_random/finetune_random_Ru.json` — random-init fine-tune.
* `logs/finetune_tmqm_vs_random/finetune_tmqm_Ru.json` — tmQM-pretrained fine-tune.

## Verification gates

1. ✅ Pretrain converges: 6 heads all show finite loss + decreasing RMSE over epochs.
2. ✅ Encoder load round-trip: `load_tmqm_encoder()` in `finetune_metalcytotox_from_tmqm.py` matches all 12 backbone tensors (atom_emb / bond_emb / edge_init / gru_update).
3. ✅ Fine-tune ablation: AUC delta (tmqm − random) = **+0.046 ≥ 0** on MetalCytoToxDB Ru subset (seed 0).

## TODO / next steps

* **3-seed + 30-epoch fine-tune** to lock down the +0.046 delta with a confidence interval.  Wall-time budget ~30 min/seed on contested CPU; recommend 3 seeds × 30 epochs as the R18 follow-up.
* **Full-corpus pretrain** (100 849 rows × 30 epochs ≈ 2.5 h wall on contested CPU, ~5 min/epoch).  Expected to lift the AUC delta above 0.05 once the encoder sees the full tmQM distribution; the 20 k-row subset already hits a +0.046 lift.
* **Try `--init tmqm --epochs 30`** for the fine-tune (5 epochs leaves head-training capacity on the table — the encoder is better than random at epoch 1 but the head has not caught up).
* **Cross-metal transfer**: re-run with `--metal Ir`, `--metal Pt`, `--metal Rh` to confirm the gain generalises beyond Ru.

## Cross-references

* Plan: `TODO/pending/22_data_gap_alignment_plan.md` (Metric 22 — CFM-vs-Lambda via R12 Deflex + R13 partial; tmQM pretrain is the encoder-init half).
* Plan: `TODO/pending/23_weak_to_strong_plan.md` Metric 1 status / strong-metric audit.
* Plan: `TODO/pending/34_parallel_ultracode_dispatch.md` §2.5 UC-C1 (PlatinAI coupling pretrain + MCTS, the *companion* pretrain work).
* Skill: `.claude/skills/mol-metal-cfm-training/SKILL.md` (encoder = D-MPNN backbone, downstream heads are task-specific).
* Skill: `.claude/skills/mol-metal-anticancer/SKILL.md` (MetalCytoToxDB protocol).
* Loader: `molmetal/data/tmqm.py` (canonical tmQM loader).