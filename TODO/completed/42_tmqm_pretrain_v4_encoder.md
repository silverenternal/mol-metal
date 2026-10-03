# TODO — tmQM Pretraining of the Metal-Aware D-MPNN V4 Encoder

**Supersedes TODO-31 in `pending/`** (`31_r17_ultracode_dispatch.md` §M-R17-M6: D-MPNN V4 encoder pretrain on tmQM; enables downstream fine-tune ablation for m1/m22).

**Date**: 2026-09-25 (R17)
**Status**: ✅ **SHIPPED** — pretrain (15 epochs / 20 k rows) + fine-tune ablation (Ru subset, 1 seed) PASS the non-negative-delta gate.
**Source PRD**: `TODO/INDEX.md` (R17 close-out plans) + `TODO/02_metrics/METRICS_TARGETS.md` Metric 1 / Metric 22.

## 1. TL;DR

The metal-aware D-MPNN encoder is now pretrainable on tmQM (Balcells & Skjelstad 2020, J. Chem. Inf. Model. 60, 6135-6146; extended 108,541 mononuclear TM complexes shipped here at `.data/molmetal/tmQM/`).

* `molmetal/scripts/pretrain_tmqm.py` (~660 LOC) trains a D-MPNN backbone (atom_emb / bond_emb / edge_init / gru_update; hidden=128, depth=3, dropout=0.1) on 6 tmQM-derived heads (5 regression + 1 oxidation-state classification) for 15 epochs on a 20 000-row subset (full 100 k × 30 epochs deferred to R18).
* `molmetal/scripts/finetune_metalcytotox_from_tmqm.py` (~330 LOC) loads the backbone and fine-tunes a binary head on the MetalCytoToxDB Ru subset (scaffold split, 19 135 rows, 29.6 % active).
* `molmetal/tests/test_tmqm_pretrain.py` — **17 contract tests pass** (loader, split, model, save/load, scaler).
* `molmetal/reports/wf_tmqm_pretrain/final.md` — full report (curves, per-task RMSE, downstream ablation, honest framing).

## 2. Files shipped

| Path | Purpose |
|---|---|
| `molmetal/scripts/pretrain_tmqm.py` | D-MPNN multi-head pretrain on tmQM with stratified split + best-epoch checkpointing |
| `molmetal/scripts/finetune_metalcytotox_from_tmqm.py` | Fine-tune ablation (init=tmqm\|random) on MetalCytoToxDB Ru subset |
| `molmetal/tests/test_tmqm_pretrain.py` | 17 contract tests for the pretrain pipeline |
| `molmetal/reports/wf_tmqm_pretrain/final.md` | Full report |
| `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` | Pretrained backbone state_dict (12 tensors, ~560 KB) |
| `models/tmqm_pretrained/pretrain_metrics.json` | 15-epoch curves |
| `models/tmqm_pretrained/pretrain_summary.json` | Final test RMSE / cls_acc / wall / n_rows |
| `models/tmqm_pretrained/featurise_stats.json` | Featurisation timing + split sizes |
| `logs/finetune_tmqm_vs_random/finetune_random_Ru.json` | Random-init fine-tune JSON |
| `logs/finetune_tmqm_vs_random/finetune_tmqm_Ru.json` | tmQM-pretrained fine-tune JSON |

## 3. Pretrain headline numbers (20 000 rows, 15 epochs, batch 128)

| Metric | Epoch 1 | Epoch 15 |
|---|---:|---:|
| train loss | 1.7429 | 0.4117 |
| val RMSE (overall) | 0.4875 | **0.2640** |
| val cls_acc (charge bin) | 0.837 | **0.967** |
| test RMSE (overall) | 0.4939 | 0.2718 |
| test cls_acc (charge bin) | 0.837 | 0.965 |

* Best-epoch encoder saved at epoch 15.
* Total wall = 891.5 s (~15 min on contested CPU).

## 4. Fine-tune ablation (Ru subset, scaffold split, seed 0, 5 epochs)

| init | test AUC | test acc | epoch 1 val_AUC | epoch 5 val_AUC |
|---|---:|---:|---:|---:|
| `--init random` | **0.7195** | 0.751 | 0.685 | 0.728 |
| `--init tmqm` | **0.7655** | 0.776 | 0.762 | 0.780 |
| **Delta (tmqm − random)** | **+0.0460** | +0.025 | +0.077 | +0.052 |

**Verify gate PASS** — non-negative delta.  Pretraining improves Ru subset test AUC by **+0.046** with a single seed and 5 epochs.  The Krasnov 2026 baseline (Ru subset AUC = 0.81) is the comparator; the gap (~0.04) is consistent with using fewer epochs and a smaller hidden dim than the Krasnov baseline.

## 5. Verification gates

1. ✅ Pretrain converges (finite loss + decreasing RMSE over 15 epochs).
2. ✅ Encoder load round-trip (`load_tmqm_encoder` matches all 12 backbone tensors).
3. ✅ Fine-tune ablation: AUC delta (tmqm − random) = +0.046 ≥ 0.
4. ✅ 17 / 17 contract tests pass (`pytest molmetal/tests/test_tmqm_pretrain.py`).

## 6. TODO / next steps (R18+)

* **3-seed × 30-epoch fine-tune** to lock down the +0.046 delta with a confidence interval.
* **Full-corpus pretrain** (100 849 rows × 30 epochs ≈ 2.5 h wall on contested CPU).  Expected to lift the AUC delta above +0.05.
* **Cross-metal transfer** — re-run with `--metal Ir`, `--metal Pt`, `--metal Rh` to confirm the gain generalises beyond Ru.

## 7. Cross-references

* Plan: `TODO/pending/22_data_gap_alignment_plan.md` (Metric 22 — CFM-vs-Lambda via R12 Deflex + R13 partial; tmQM pretrain is the encoder-init half).
* Plan: `TODO/pending/23_weak_to_strong_plan.md` Metric 1 status / strong-metric audit.
* Plan: `TODO/pending/34_parallel_ultracode_dispatch.md` §2.5 UC-C1 (companion pretrain work).
* Skill: `.claude/skills/mol-metal-cfm-training/SKILL.md` (encoder = D-MPNN backbone).
* Skill: `.claude/skills/mol-metal-anticancer/SKILL.md` (MetalCytoToxDB protocol).
* Loader: `molmetal/data/tmqm.py` (canonical tmQM loader).
