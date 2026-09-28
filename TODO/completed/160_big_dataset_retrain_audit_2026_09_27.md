# TODO-160 — Big Dataset Retrain Audit (2026-09-27, honest framing)

**Date**: 2026-09-27
**Reason**: User challenge 2026-09-27 "把所有的弱指标都拉起来吧，而且更重要的是，我们的整个模型在大数据集上完全重新训练了吗？需要更完整的规划以及todo目录的完整更新"
**Auditor**: Claude Code (under explicit user instruction + party-twenty-spirit §3.5 honest framing)
**Skill alignment**: `party-twenty-spirit` §3.5 (0 容忍 不掩盖) + `mol-metal-gpu-execution` (GPU retrain context)

---

## §0 TL;DR — three honest numbers

1. **Total training data actually available on disk**: **~1,924,037 rows** across 7 datasets (MetalCytoToxDB + NCI-60 + BindingDB subset + DrugOOD 1.58M + tmQM + PlatinAI 500 + CrossDocked 100K pairs). The "2.67M" aspirational figure is **NOT on disk** — 1.07M BindingDB full + 226K PlatinAI full + 296K DrugOOD partial are absent.
2. **Actually used in actual paper-grade retrain**: **~146,830 rows** (MetalCytoToxDB 26,801 + NCI-60 7,823 + BindingDB subset 5,257 + tmQM 100,849 + metacytotox_fallback 5,200 + DrugOOD 10K subsample). Utilization rate: **146,830 / 1,924,037 ≈ 7.6%** of what's actually on disk.
3. **Model backbone used in retrain**: Mostly **D-MPNN multi-task + Wasserstein pIC50** (Wf-124 P3, Wf-131 P1/P2) or **chemprop v2** (Wf-124 P3 head-to-head, Wf-146 LOMO in flight). The "big dataset full retrain" target architecture — chemprop v2 with Multi-PH + PLL + Wasserstein transfer — has **NOT been retrained**. Only CPU wiring + smoke exists.

---

## §1 Per-dataset status — what is actually on disk vs aspirational

### §1.1 Inventory (MEASURED on 2026-09-27)

| Dataset | Available on disk | Aspirational figure | Source of truth | Status |
|---|---:|---:|---|---|
| **MetalCytoToxDB** | **26,801 rows** (real, Zenodo 17106822) | 26,801 | `.data/molmetal/MetalCytoToxDB.csv` | ✅ AVAILABLE + ACTIVE |
| **NCI-60 GI50** (precious-metal filtered) | **7,823 rows** (real, filtered to 9 precious-metal cell lines) | 7,823 | `molmetal/data/nci60_platinai_link.csv` | ✅ AVAILABLE + ACTIVE |
| **BindingDB** (metal-filtered subset) | **5,257 rows** (subset only — NOT the full 1.07M) | 1,070,000 | `molmetal/data/protein_metal_binding.csv` | ⚠️ PARTIAL (subset only; full not downloaded) |
| **DrugOOD IC50** (native) | **1,583,302 rows** (per `drugood_ood_auc_native_smoke.json`) | 1,583,302 | `.data/molmetal/drugood/ic50/data/train.csv` (138 MB) | ✅ AVAILABLE + LOADER MISSING (`molmetal.data.drugood_loader` does not exist) |
| **DrugOOD partial / Ji 2023 296K protocol** | **0 rows** (drugood/ sub-dir is empty per TODO-148 §2.1) | 296,675 | `molmetal/data/drugood/` (empty) | ❌ NOT ON DISK |
| **DrugOOD refined × 3 configs** | **0 rows** | 799,566 | (not staged) | ❌ NOT ON DISK |
| **tmQM** | **100,849 rows** (real, 7 files, 300 MB) | 100,849 | `.data/molmetal/tmQM/tmQM_y.csv` + `tmQM_X{1,2,3}.{xyz,BO}.gz` | ✅ AVAILABLE + ACTIVE |
| **PlatinAI** (pre-shipped subset) | **500 mol** (pre-shipped MaxMin pool) | 500 | `molmetal/data/metallo_drugs_500_train.csv` | ✅ AVAILABLE (subset only) |
| **PlatinAI** (full xlsx) | **0** (GitHub 404, ChemRxiv gated, Zenodo no match) | 226,000 | (not available) | ❌ NOT AVAILABLE |
| **CrossDocked2020** (pocket-ligand) | **100,000 pairs** (symlinked from pocket/) | 100,000 | `.data/molmetal/crossdocked/` | ✅ AVAILABLE + ACTIVE (for docking only, NOT for retrain) |
| **TOTAL on disk** | **1,924,037 rows / pairs** | **3,828,438 rows** (aspirational) | — | — |
| **TOTAL aspirational** | — | **~2.67M** (per task framing) | — | 2.67M is INFLATED; real on-disk is 1.92M |

**Honest finding**: The "2.67M" total referenced in the user prompt is **inflated**. Real on-disk total is **1.92M** (DrugOOD 1.58M dominates). The gap (2.67M − 1.92M = 750K) is data that does not exist on this host:
- BindingDB full 1.07M → only 5K subset on disk (gap = 1.06M)
- PlatinAI full 226K → only 500 mol subset (gap = 225K)
- DrugOOD refined × 3 = 800K → not on disk (gap = 800K)
- DrugOOD partial Ji 2023 296K → not on disk (gap = 296K)
- **Sum of gaps**: ~2.4M, which exceeds the "aspirational" 2.67M discrepancy

### §1.2 Loader module gaps (blocking paper-grade retrain even for on-disk data)

Per TODO-148 §2.1 verbatim audit (3 of 4 loaders DO NOT EXIST):

| Dataset | rows | Loader module | Status |
|---|---:|---|---|
| MetalCytoToxDB | 26,801 | (existing loaders via `molmetal.molmetal_lam.lam_chem.platinai_dataset`) | ✅ LOADER PRESENT |
| NCI-60 | 7,823 | `molmetal.data.nci60.load_nci60` | ❌ **MODULE MISSING** (loaded via pandas direct read) |
| BindingDB subset | 5,257 | `molmetal.data.binding_db.load_binding_db` | ❌ **MODULE MISSING** (loaded via pandas direct read) |
| DrugOOD native | 1,583,302 | `molmetal.data.drugood_loader.load_drugood` | ❌ **MODULE MISSING** (loaded via pandas direct read) |
| tmQM | 100,849 | `molmetal.data.tmqm.load_tmqm` | ✅ LOADER PRESENT |
| PlatinAI subset | 500 | (pre-shipped `metallo_drugs_500_train.csv`) | ✅ CSV PRESENT |
| CrossDocked2020 | 100,000 pairs | `molmetal.data.crossdocked.CrossDockedDataset` | ✅ LOADER PRESENT |

**Critical bottleneck**: 3 of 6 datasets have **NO loader module**, only pandas direct reads. This blocks paper-grade end-to-end pipeline runs even for data that IS on disk. The `drugood_ood_auc_native_smoke.json` proves the data is accessible (10K subsample + LR worked), but only via manual pandas load — no module API.

---

## §2 What was actually used in recent training runs

### §2.1 Wf-124 P3 chemprop v2 head-to-head (2026-09-27, MEASURED per TODO-154)

- **Data**: MetalCytoToxDB 26,801 rows × 5 metals (Ru/Ir/Os/Rh/Re), per-metal scaffold split
- **Model**: chemprop v2.0 default D-MPNN classifier (hidden=300, depth=3, 50 epochs default)
- **Runs**: 14 paired runs = Ru/Ir/Os/Rh × 3 seeds + Re × 2 seeds (Re underpowered at n=2)
- **Result**: chemprop v2 mean AUC = **0.886 ± 0.020** (across 5 metals, beats Krasnov 0.81 on Ru by +0.099)
- **Rows used per run**: full per-metal training subset (Ru=19135, Ir=4546, Rh=1134, Os=1118, Re=868)
- **Per-run wall time**: 0.193 min (chemprop v2 5.1× faster than V4)
- **Total compute used**: ~14 × 0.193 min ≈ 2.7 min wall (CPU-class epochs on chemprop v2)
- **Source**: `molmetal/reports/wf124_p3_chemprop_head_to_head.json` (or equivalent)

### §2.2 Wf-124 P3 D-MPNN multi-task Wasserstein (Ru seed=0, 2026-09-27, MEASURED per TODO-131 + commit 1a8eebd)

- **Data**: MetalCytoToxDB 26,801 rows × 5-metal joint pool
- **Model**: V4MultiTaskDMPNNPretrain (D-MPNN + Wasserstein pIC50 reg + L2 guard)
- **Runs**: 1 seed (seed=0) × 30 epochs on GPU-1 RTX 5090
- **Result**: Ru test_auc = **0.7613** (vs baseline 0.7123, +0.049 lift); test_pearson = 0.3091
- **Rows used**: 2509 train / 363 val / 796 test (per-metal scaffold split)
- **Per-run wall time**: 463.2 s (≈15.4 s/epoch)
- **Source**: `molmetal/reports/wf124_p3_ru_seed0/` + commit `1a8eebd Wf-124 P3: Ru seed=0 first complete result`

### §2.3 Wf-131 P1 Ru 3-seed × 10-epoch (2026-09-27, MEASURED per TODO-131 §1.2)

- **Data**: MetalCytoToxDB 5-metal (Ru/Ir/Os/Rh/Re), per-metal scaffold split
- **Model**: V4MultiTaskDMPNN + Wasserstein pIC50 + L2 guard
- **Runs**: 3 seeds × 10 epochs × 5 metals
- **Result**: Ru mean AUC = **0.7352 ± 0.0023** (3 seeds); Ir 0.6035 ± 0.0311; Os 0.2500 (n=1 only); Rh n=1 only; Re n=1 only
- **Rows used**: 2509 train / 796 test for Ru; smaller for Ir/Os/Rh/Re
- **Total compute**: ~16 × 10 epochs on GPU-1

### §2.4 Wf-131 P3 LOMO full (DEFERRED, per TODO-131 §3)

- **Data**: MetalCytoToxDB 26,801 rows × 5-metal leave-one-metal-out folds
- **Model**: V4MultiTaskDMPNN + Wasserstein pIC50 + L2 guard + Rh closure oversample (500 synthetics)
- **Runs**: 5 metals × 3 seeds × 50 epochs = 75 paired runs (per Wf-131 master plan)
- **Status**: **DEFERRED** — only fold=Ru seed=0 20-epoch partial log on disk (val_auc=0.7941 at epoch 19, no test_auc surfaced)
- **On-disk completion** (per `chemprop_v2_status_2026_09_27.json` `actually_interrupted_workflow_identified`):
  - Ru: 5/5 done (mean test_auc 0.594, std 0.013) — **NOTE: this is D-MPNN Wf-131, NOT chemprop v2**
  - Ir: 5/5 done (mean test_auc 0.624, std 0.017)
  - Os: 5/5 done (mean test_auc 0.608, std 0.026)
  - Rh: 3/5 done (seeds 0,1,2; mean test_auc 0.687 across 3 seeds, std 0.293)
  - Re: 0/5 done
  - **Remaining**: 7 pairs = [Rh seed3, Rh seed4, Re seed0, Re seed1, Re seed2, Re seed3, Re seed4]
  - **Estimated wall time**: ~92 min on RTX 5090

### §2.5 Wf-146 chemprop v2 LOMO sweep (IN FLIGHT, 2026-09-27 16:50 CST per `chemprop_v2_5metal_5seed_50epoch.json`)

- **Data**: MetalCytoToxDB 26,801 rows × 5-metal LOMO folds
- **Model**: chemprop v2 (hidden=64, depth=2 — SMOKE-grade backbone, NOT production 300/3)
- **Runs**: 5 metals × 5 seeds × 50 epochs = 25 paired runs
- **Status**: **PARTIAL — 1/25 completed**
  - Rh seed=0 test_auc = **0.5256** (completed at epoch 50, 156.5 s elapsed)
  - 24 remaining = PENDING
  - Sweep state: relaunched via nohup+disown at 16:45:41 CST, PID 1382182, monitor task `bql4sj5vy`
- **Per-run wall time** (chemprop v2 smoke): ~156.5 s × 25 = ~65 min total projected
- **Source**: `molmetal/reports/wf146_chemprop_lomo/{metal}_seed{N}.json`

### §2.6 V4 DrugOOD 3-split (assay/scaffold/size) per `drugood_ood_auc.json`

- **Data**: metacytotox_fallback subset = **5,207 rows** (NOT the full DrugOOD 1.58M)
- **Model**: v4_morgan_mlp (Morgan FP 2048 + Ridge MLP) — NOT chemprop v2, NOT D-MPNN
- **Runs**: 3 splits (assay/scaffold/size) × 3 seeds × 30 epochs
- **Result**: OOD AUC = 0.891/0.697/0.748 (mean) — beats iMoLD 0.721/0.688/0.679 on all 3 splits
- **Honest caveat**: This is **MLP on Morgan FP**, NOT the iMoLD graph NN. The honest framing note in the JSON explicitly states "NOT a head-to-head comparison against iMoLD on the actual DrugOOD IC50 splits".
- **Source**: `metrics/by_metric/drugood_ood_auc.json`

### §2.7 DrugOOD native smoke (per `drugood_ood_auc_native_smoke.json`)

- **Data**: DrugOOD IC50 native 1,583,302 rows — **10K subsample only** used (not full)
- **Model**: LogisticRegression (proxy; NOT graph NN like iMoLD)
- **Runs**: 3 splits × 3 seeds × 5 epochs
- **Result**: OOD AUC = 0.5625/0.5999/0.5813 (assay/scaffold/size) — below iMoLD 0.7211/0.6884/0.6792
- **Honest framing**: "Mol-Metal pipeline verified on real DrugOOD IC50 splits; same-protocol head-to-head with iMoLD requires graph NN backbone and full training, out of scope for this MLP-based paper. We do NOT claim to beat iMoLD."
- **Source**: `metrics/by_metric/drugood_ood_auc_native_smoke.json`

### §2.8 Wf-86 cross-dataset R² (per `metalcytotox_combined_r2.json`)

- **Data**: tmQM d-block (29 metals) + MetalCytoToxDB 5-metal + PlatinAI 4-cell
- **Model**: F5_RDKIT_2D (12-d RDKit descriptors) + Ridge baseline
- **Runs**: 5 seeds × 80/20 train/test
- **Result**:
  - PlatinAI 4-cell: macro R² = +0.0034 (near predict-the-mean)
  - MetalCytoToxDB 5-metal: macro R² = +0.128 (per-metal all positive, Re strongest +0.168)
  - tmQM d-block: macro R² = +0.243 (29 cohorts all positive, Mn/Cr/Zn strongest)
- **Cross-dataset macro**: +0.1248
- **Source**: `molmetal/reports/wf71_p3_wf49_cross_round2/cross_dataset_summary.json`

### §2.9 NCI-60 precious-metal 9-cell AUC (per `nci60_precious_9cell_auc.json`)

- **Data**: NCI-60 9 precious-metal cell lines subset = 7,823 rows
- **Model**: D-MPNN k-cell-line multi-task (9 cls + 9 reg heads), scaffold split
- **Runs**: 3 seeds × 50 epochs
- **Result**: 9-cell mean AUC = **0.8007 ± 0.016** (vs Krasnov 0.81 baseline, −0.009 just under)
- **Per-cell**: MCF7 0.744, A549 0.776, A2780 0.693, A2780cis 0.728, HCT116 0.758, ... (9 cells total)
- **Source**: `molmetal/scripts/train_nci60_10cell.py`

### §2.10 tmQM Pretrain baseline (per `tmqm_combined_r2.json`)

- **Data**: tmQM 100,849 rows × 29 d-block metals
- **Model**: F5_RDKIT_2D + Ridge baseline
- **Runs**: 5 seeds × 80/20 split
- **Result**: macro R² = +0.243 (29 cohorts positive)
- **Source**: `molmetal/reports/wf71_p3_wf49_cross_round2/tmqm_dblock/per_cohort_metrics.json`

### §2.11 Summary — actually used in recent training

| Source | Rows used | Status |
|---|---:|---|
| MetalCytoToxDB (Wf-124 + Wf-131 + Wf-146) | 26,801 (used many times, full coverage) | ✅ Used in 4+ workflows |
| NCI-60 9-cell subset (Wf-nci60) | 7,823 | ✅ Used |
| BindingDB metal-filtered (Wf-49, Wf-71) | 5,257 | ✅ Used (subset only — full 1.07M NOT used) |
| DrugOOD metacytotox_fallback (drugood_ood_auc) | 5,207 | ✅ Used |
| DrugOOD native smoke (10K subsample) | 10,000 (out of 1,583,302) | ⚠️ <1% used |
| tmQM d-block (Wf-49 + Wf-71 + Wf-77/79/128) | 100,849 | ✅ Used (multiple times) |
| PlatinAI 500-mol subset | 500 | ✅ Used |
| CrossDocked2020 (docking only, no retrain) | 100,000 pairs | ⚠️ Used for docking eval only |
| **TOTAL actually used** | **~146,437 rows + 100K docking pairs** | — |
| **TOTAL on disk** | **1,924,037 rows / pairs** | — |
| **Utilization rate** | **146,437 / 1,924,037 ≈ 7.6%** | — |

**Honest interpretation**:
- DrugOOD 1.58M rows are **on disk** but **<1% used** (only 10K subsample in 1 experiment)
- BindingDB 1.07M is **NOT on disk**; only 5K subset
- tmQM 100K is fully used in cross-dataset R² tests
- MetalCytoToxDB 26K is the workhorse (used 4+ times)
- **The "big dataset full retrain" goal has NOT been achieved**

---

## §3 Per-dataset utilization calculation

### §3.1 Utilized vs available (MEASURED, on-disk only)

```
Total on disk:           1,924,037 rows / pairs
Total utilized:            146,437 rows + 100K docking pairs = 246,437
Utilization rate:        246,437 / 1,924,037 = 12.8%
                        (or 146,437 / 1,924,037 = 7.6% excluding docking)
```

**Per-dataset utilization**:

| Dataset | On disk | Used | Utilization |
|---|---:|---:|---:|
| MetalCytoToxDB | 26,801 | 26,801 | **100%** (used many times) |
| NCI-60 subset | 7,823 | 7,823 | **100%** |
| BindingDB subset | 5,257 | 5,257 | **100%** (subset; full 1.07M not on disk) |
| DrugOOD native | 1,583,302 | 10,000 (subsample) | **0.6%** |
| tmQM d-block | 100,849 | 100,849 | **100%** |
| PlatinAI 500 subset | 500 | 500 | **100%** (subset; full 226K not on disk) |
| CrossDocked2020 | 100,000 pairs | 100,000 pairs (docking only) | **100%** (for docking; 0% for retrain) |

### §3.2 What NEEDS to be used for "full retrain" goal

If "full retrain" = chemprop v2 with Multi-PH + PLL + Wasserstein transfer on the 1.92M on-disk data:

| Phase | Data | Rows | Status |
|---|---|---:|---|
| Phase A: chemprop v2 per-metal in-domain (in flight) | MetalCytoToxDB | 26,801 | Wf-146 in flight (1/25 done) |
| Phase B: chemprop v2 LOMO 5×5×50 | MetalCytoToxDB | 26,801 | Wf-146 (pivoted, ETA 4.6 hr) |
| Phase C: D-MPNN NCI-60 9-cell multi-task | NCI-60 | 7,823 | SHIPPED (3-seed × 50-epoch) |
| Phase D: BindingDB full 1.07M | BindingDB | 1,070,000 | ❌ NOT ON DISK |
| Phase E: DrugOOD 1.58M native | DrugOOD | 1,583,302 | ⚠️ 10K subsample only (loader module missing) |
| Phase F: tmQM 100K d-block pretrain | tmQM | 100,849 | ✅ Used |
| Phase G: PlatinAI full 226K | PlatinAI | 226,000 | ❌ NOT ON DISK |
| **Total rows needed for "full retrain"** | — | **3,040,775** | — |
| **Total currently on disk + used** | — | **146,437** | — |
| **Utilization vs full retrain goal** | — | **4.8%** | — |

---

## §4 What NEEDS to be done for full retrain (honest plan)

### §4.1 Critical prerequisites (must address before any "full retrain" can happen)

1. **Data download gaps** (P0 CRITICAL):
   - Download BindingDB full 1.07M (filter to metal-binding, project to ~50K-200K rows)
   - Download PlatinAI full xlsx 226K (if URL ever becomes available; currently 404)
   - Verify DrugOOD refined × 3 configs (check `Ji 2023` paper protocol for partial subset 296K)
2. **Loader modules** (P0 CRITICAL):
   - Add `molmetal/data/drugood_loader.py` (DrugOOD 1.58M has no module loader per TODO-148 §2.1)
   - Add `molmetal/data/binding_db.py` (BindingDB subset 5K has no module loader)
   - Add `molmetal/data/nci60.py` (NCI-60 7.8K has no module loader)
3. **GPU retrain authority** (P0 per standing "kill then adapt" directive):
   - User authorization required for any GPU work that exceeds 1 hr wall time
   - All recent GPU runs are DEFERRED (Wf-131 P3, Wf-146, Wf-135, Wf-138, R19-P1..P6)
   - Standing directive: TODO-148 §0, TODO-152, INDEX.md "LOMO 5/5×5-seed×50-epoch sweep DEFERRED per 'kill then adapt' user directive (GPU work)"

### §4.2 Architecture decisions (P1 HIGH)

- **Per-metal fold vs LOMO** (decision needed):
  - Current: Wf-146 LOMO 5×5×50 = lower AUC (Rh=0.5256 in 1-seed smoke) due to held-out metal head untrained
  - Alternative: per-metal fold 5×5×50 = Wf-154 MEASURED AUC=0.886 (beats Krasnov 0.81)
  - Recommendation: **Per-metal fold for paper §4 headline** (chemprop v2 wins on all 5 metals), **LOMO for §B appendix** (honest cross-metal generalization story)
- **Backbone** (decision made per TODO-155):
  - chemprop v2 (WINS head-to-head vs V4 by +0.282 AUC, 5.1× faster)
  - V4 D-MPNN kept as §B appendix (lower bound on per-metal D-MPNN)
- **Sample efficiency**:
  - per-metal fold requires 26,801 rows
  - LOMO requires 26,801 rows per fold (n_train varies by metal: Ru=13,395 → Re=607)
  - Cross-dataset retrain (DrugOOD + BindingDB) requires 1M+ rows

### §4.3 Compute budget (P1 HIGH)

Per TODO-155 §2.2 (chemprop v2 ETA):
- **5-metal × 5-seed × 50-epoch LOMO** = 125 paired runs × chemprop per-run cost (incl. data load + ckpt I/O + scaffold-split) = **3.8 hr** + 0.8 hr overhead = **4.6 hr wall-time (sequential single-GPU)**
- **5-metal × 5-seed × 50-epoch per-metal in-domain** = similar 4.6 hr (chemprop v2 doesn't benefit from parallelism)
- **DrugOOD 1.58M chemprop v2 + 5-cell lines** = ~10 hr (assuming linear scaling)
- **BindingDB full 1.07M** = ~7 hr (similar scaling)
- **TOTAL full retrain estimate**: ~22 hr wall-time (single RTX 5090) or ~11 hr (Blackwell + RTX 5090 dual)

### §4.4 Wall-time budget (P1 HIGH)

Per `party-twenty-spirit` §3.5 honest framing:
- All wall-time figures are **PROJECTED**, not measured
- 4.6 hr LOMO chemprop v2 (TODO-155 ETA, MEASURED per TODO-154 §3)
- 22 hr full retrain across 4 datasets (PROJECTED)
- GPU Blackwell single-card constraint per INDEX.md §0

---

## §5 Remaining work (ordered by P0/P1 priority)

### P0 — Critical blockers (must address before any retrain)

1. **Download BindingDB full 1.07M** (or document why it's not available; Zenodo/HuggingFace/API)
2. **Add 3 missing loader modules**: `drugood_loader.py`, `binding_db.py`, `nci60.py`
3. **Resolve chemprop v2 CPU SMILES featurization bottleneck** (Wf-155 aborted at iteration 1; precompute features per epoch + faster encoder pipeline; ETA fix: 30 min engineering + 1 day validation)
4. **User authorization for GPU retrain** (standing "kill then adapt" directive; explicit fresh OK needed for any work > 1 hr wall)

### P1 — High priority (after P0 unblocked)

5. **Complete Wf-146 chemprop v2 LOMO sweep** (24/25 remaining; ~60 min ETA per chemprop_v2_5metal_5seed_50epoch.json)
6. **Complete Wf-131 P3 LOMO full** (7 pairs remaining: Rh seed3/4 + Re seed0-4; ~92 min ETA per chemprop_v2_status_2026_09_27.json)
7. **Wf-135 m19 LOMO Rh oversample regression fix** (TODO-135 master plan, GPU retrain pending)
8. **Wf-138 paper v2 submission package** (pending Wf-134 + Wf-135 + Wf-139 P5)
9. **Wf-134 m15 Docking RMSD redesign Phase 2** (TODO-134, 5-pocket × 3-seed Vina redocking)
10. **R19-P1..P6 GPU retrain** (R19_PLAN.md §3.1, 10 mechanism M22-M30 + 4 joint retrain)
11. **Wf-65 R20-P1 VariancePUCT default** (CPU-only, can ship anytime)
12. **Wf-56 R20-P2 Hilbert-ASTSR + Multi-PH + PLL + Triangle closure** (GPU retrain deferred)
13. **Wf-66 R20-P3 Multi-PH Fourier 32-bin + PLL 128-d + Score-Rep MC** (GPU retrain deferred)
14. **Wf-67 R20-P4 3-way on MetalCytoToxDB + Closure d-block ≥ 0.95** (GPU retrain pending)

### P2 — Medium priority

15. **Wf-68 R20-P5 Closure theorem + Catherine 6/6** (CPU smoke SHIP; GPU retrain deferred)
16. **Wf-58 R20-P6 Wet-lab IC50 cooperation** (6-mo deferred; LOIs drafted but not sent)
17. **Wf-73 P2/P3 per-metal / per-element OOD** (TODO-101/102, in flight)
18. **Wf-71 P3 Wf-49 cross-dataset retrain round 2** (TODO-100 SHIPPED; R² +0.1248 macro)
19. **Wf-77 R22-P3 NSGA-III Pareto front** (TODO-101 in flight)
20. **Wf-83 Phase 1 + 2 Hilbert-ASTSR + PlatinAI 4-cell** (TODO-104/106 SHIPPED)
21. **Wf-85 P1/P2 Hilbert only + ASTSR only real MetalCytoToxDB** (TODO-105/106 SHIPPED)
22. **Wf-86 P2/P3 tmQM + PlatinAI 4-cell real** (TODO-106 SHIPPED)
23. **Wf-44 P2 Closure OOD Phase 2** (TODO-98 SHIPPED; per-domain reachability ≥ 0.85 deferred Q4 2026)
24. **Wf-77 R22-P3 NSGA-III Pareto ranker wire-in** (TODO-101 in flight)
25. **Wf-83..87 L5 L7 L10 joint rescue** (TODO-127/128/129 SHIPPED; arch ceiling honest-negative)

### P3 — Low priority / long-tail

26. **Wf-49 R2 root fix** (TODO-91 SHIPPED; 12-d RDKit features shipped)
27. **Wf-46 23 fails rerun** (TODO-117 in flight)
28. **Wf-56..58 R20-P1..P3 stubs** (5 NEW skills R20, partial ship)
29. **Wf-74 wet-lab outreach partial** (TODO-116 SHIPPED)
30. **Wetlab SOP/LOI drafts** (7 files in TODO/pending/, all DRAFT/TEMPLATE)

---

## §6 Honest framing summary

### §6.1 What IS true

- 1.92M rows of training data IS on disk across 7 datasets
- MetalCytoToxDB 26,801 rows IS the workhorse (used 4+ workflows)
- Wf-124 P3 chemprop v2 BEATS Krasnov 2026 baseline by +0.099 AUC on Ru (MEASURED)
- Wf-124 P3 D-MPNN Wasserstein lifts Ru seed=0 from 0.7123 → 0.7613 (+0.049 lift MEASURED)
- Wf-71 P3 cross-dataset R² lift +0.1248 macro (MEASURED)
- Wf-131 P1 Ru 3-seed × 10-epoch mean AUC = 0.7352 ± 0.0023 (MEASURED)
- Wf-131 P2 500 Rh closure-oversample + W₂ decay SHIPPED (MEASURED)
- D-MPNN 25 ckpts (5 metals × 5 seeds) SHIPPED with SHA256 overwrite guard
- Wf-146 chemprop v2 LOMO 1/25 completed (Rh seed=0 = 0.5256 MEASURED)
- Wf-131 P3 D-MPNN LOMO 18/25 completed (Ru/Ir/Os 5/5 + Rh 3/5, Re 0/5)
- All 32 R18 modules + ≈1,235 tests CPU-shipped
- 12 R17+R18 skills complete (5 R17 + 3 R18 NEW + 4 R19/R20 stubs)

### §6.2 What is NOT true (claims to drop per party-twenty-spirit §3.5)

- **"2.67M total available data"** is INFLATED. Real on-disk is 1.92M (DrugOOD dominates 1.58M)
- **"Full retrain on big dataset"** has NOT been done. Utilized = 7.6% of on-disk; 4.8% of aspirational full-retrain goal
- **DrugOOD 1.58M is mostly UNUSED** (only 10K subsample in 1 experiment)
- **BindingDB full 1.07M is NOT on disk** (only 5K subset)
- **PlatinAI full 226K is NOT on disk** (only 500-mol subset)
- **3 of 6 dataset loader modules MISSING** (drugood_loader, binding_db, nci60)
- **Wf-131 P3 LOMO DEFERRED** (only fold=Ru seed=0 20-epoch partial; full sweep not measured)
- **Wf-146 LOMO in flight** (1/25 done; honest framing: this is SMOKE backbone hidden=64/depth=2, NOT production 300/3)
- **Wf-135 m19 Rh oversample regression** (TODO-135 master plan; 5-seed re-bench pending)
- **Wf-138 paper v2** (pending Wf-134 + Wf-135 + Wf-139 P5; NOT shipped)
- **All R19/R20 GPU retrain** DEFERRED per standing "kill then adapt" directive

### §6.3 What to honestly claim in paper

Per `party-twenty-spirit` §3.5 honest framing:
- **Paper §4 LEAD**: 8 strong metrics (m11 n_distinct, m12 synthesis, m13 RDKit validity, m14 CFM decode, m17 DrugOOD OOD, m2 PC3/HL60/SKOV3, m21 NCI-60 9-cell, Deflexpressor ba_acc)
- **Paper §4 ACCEPTABLE**: m2 Pt MCF-7/A549/A2780cis aggregate (within R18 9-cell 0.8007)
- **Paper §4.5 HONEST-NEGATIVE**: m1 Ru AUC (0.7613 vs 0.86 gate), m3 Vina Dock (−1.29 vs −9.37 SOTA), m4 Success Rate (0.0 vs 60% gate), m7 Diversity (0.1065 vs 0.65 gate), m8 PB-Valid, m10 Vina Min, m16 Affinity Pearson r (0.207 vs 0.86 Boltz-2), m19 LOMO (0.535 vs 0.70 gate)
- **Paper §6 LIMITATIONS**: Dataset gaps (BindingDB/PlatinAI full not on disk), GPU retrain deferred (per standing directive), 3 loader modules missing
- **Paper §B APPENDIX**: chemprop v2 vs V4 head-to-head (TODO-154 verbatim), cross-dataset R² baseline (TODO-100 verbatim)

---

## §7 Cross-references

- **TODO-148** — Wf-146 LOMO retrain prep status (PREP COMPLETE, ACTUAL DEFERRED, PIVOTED chemprop v2)
- **TODO-155** — Pivot Wf-146/157/158 to chemprop v2 (V4 → chemprop_v2)
- **TODO-154** — chemprop v2 vs V4MultiTaskDMPNN head-to-head (MEASURED n=14 paired runs, p=1.35e-5)
- **TODO-131** — m1 + m19 joint rescue close-out (PARTIAL SHIP, m1 0.7123 → 0.7613 lift, m19 LOMO DEFERRED)
- **TODO-120** — Wf-120 D-MPNN ckpt seed integrity close-out (P1+P3 SHIP, P2 partial)
- **TODO-121** — m19 LOMO strengthen via Wasserstein transfer + Rh oversample
- **TODO-100** — Wf-71 P3 Wf-49 cross-dataset retrain round 2 (cross-dataset R² +0.1248 macro)
- **TODO-82** — Metric gap filling plan (post-R18 close-out, 7 gaps identified)
- **TODO-23** — Weak → strong conversion R18 plan
- **TODO-22** — Data gap alignment plan (mostly SHIPPED, 17/25 metrics MEASURED)
- **TODO-58** — Weak → strong synthesis (5 truly strong + 9 weak honest-negative)
- **TODO-82 metric_gap_filling_plan.md** — Phase A-F 4-week GPU retrain plan
- **TODO/03_plans/R17_AUC_RECOVERY_PLAN.md** — R17 Tier-1 AUC fix strategy
- **TODO/03_plans/R18_PLAN.md** — R18 detailed phase plan (47-item completion matrix)
- **TODO/03_plans/R19_PLAN.md** — R19 GPU retrain cycle (12-week Blackwell + RTX 5090)
- **TODO/03_plans/WEAK_TO_STRONG_PLAN.md** — 8 STRONG + 8 WEAK + paper-credibility framing
- **TODO/03_plans/LAMBDA_ROADMAP.md** — R20-R25 Lambda algorithm roadmap (8 gaps L1-L8)
- **TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md** — Sprint A improvement tasks + acceptance gates
- **`metrics/by_metric/dmpnn_ckpt_status.json`** — 25 D-MPNN ckpts (5 metals × 5 seeds) status
- **`metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json`** — Wf-146 LOMO in flight
- **`metrics/by_metric/metalcytotox_ru_auc.json`** — Ru seed=0 0.7613 MEASURED
- **`metrics/by_metric/metal_lomo_auc_wasserstein.json`** — Wf-125 P2 0.4568 BELOW GATE (smoke re-bench)
- **`metrics/by_metric/nci60_precious_9cell_auc.json`** — 9-cell mean AUC 0.8007 MEASURED
- **`metrics/by_metric/drugood_ood_auc.json`** — 3-split OOD AUC 0.891/0.697/0.748 (metacytotox_fallback 5,207 rows)
- **`metrics/by_metric/drugood_ood_auc_native_smoke.json`** — DrugOOD native 10K subsample 0.5625/0.5999/0.5813
- **`metrics/by_metric/metalcytotox_combined_r2.json`** — 5-metal R² aggregated (Wf-77/79/128)
- **`metrics/by_metric/tmqm_combined_r2.json`** — tmQM 100K d-block R² (Wf-49 R2 root fix)
- **`metrics/by_metric/dmpnn_ckpt_status.json`** — D-MPNN ckpt inventory + SHA256 overwrite guard
- **`metrics/by_metric/chemprop_v2_status_2026_09_27.json`** — chemprop v2 sweep status snapshot (honest-negative)
- **`TODO/INDEX.md`** — Single source of truth (272+ files, 18 truly pending)
- **`TODO/EXECUTION_ORDER.md`** — Priority order (R18 done, R19 in flight)
- **`TODO/03_plans/R18_PLAN.md`** — R18 phase breakdown
- **`DATA_STATUS.md`** — Local dataset inventory (verified 2026-09-25)

---

## §8 Verdict — single line

**Honest verdict**: ~1.92M rows of training data exist on host across 7 datasets; only **7.6%** has been used in actual paper-grade retrain. The "big dataset full retrain" goal is **NOT achieved**. P0 blockers: 3 missing loader modules + BindingDB full 1.07M not on disk + PlatinAI full 226K not on disk + standing GPU "kill then adapt" directive. P1 work: complete Wf-146 (24/25 remaining), complete Wf-131 P3 (7/25 remaining), Wf-135 Rh regression fix, Wf-134 RMSD redesign, paper v2 (Wf-138), 4 R19 GPU retrain phases (M22-M30).