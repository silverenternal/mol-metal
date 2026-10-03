# R20 Big-Dataset Full Retrain Plan (2026-09-27, post-audit)

**Date**: 2026-09-27
**Reason**: User challenge 2026-09-27 "把所有的弱指标都拉起来吧，而且更重要的是，我们的整个模型在大数据集上完全重新训练了吗？需要更完整的规划以及todo目录的完整更新"
**Status**: 🆕 **NEW pending** — comprehensive retrain plan across all available datasets
**Priority**: **P0 CRITICAL** (paper §4 metric lifts + paper §6 honest framing)
**Owner**: Hugo Lee + ultracode agents (Wf-65..76 + Wf-124/131/146 + Wf-49/71/77/79/86)
**Parent**: `TODO/completed/160_big_dataset_retrain_audit_2026_09_27.md` (this audit) + `TODO/completed/82_metric_gap_filling_plan.md` (R18 metric gap) + `TODO/03_plans/R19_PLAN.md` (R19 GPU retrain)
**Skill alignment**: `party-twenty-spirit` §3.5 (honest framing — no aspirational numbers, every number traces to a JSON path) + `mol-metal-gpu-execution` (Blackwell + RTX 5090 dual-card) + `mol-metal-anticancer` (Tier-1 SOTA comparison)

---

## §0 TL;DR — three honest numbers + 6 phases

1. **Total on disk**: 1.92M rows / pairs (7 datasets: MetalCytoToxDB 26.8K + NCI-60 7.8K + BindingDB 5K + DrugOOD 1.58M + tmQM 100K + PlatinAI 500 + CrossDocked 100K)
2. **Currently utilized**: 7.6% (146K rows in actual retrain); DrugOOD 1.58M mostly UNUSED; BindingDB full 1.07M NOT on disk
3. **Architecture pivot**: chemprop v2 (NOT V4) per TODO-155 — MEASURED +0.282 AUC vs V4 (paired t-test p=1.35e-5) + 5.1× faster

**6-phase plan (12 weeks wall-time)**:

| Phase | Wall | Data | Goal | Gate |
|---|---|---|---|---|
| **Phase 0** (P0 CRITICAL) | 1 week CPU | n/a | P0 blockers: 3 loader modules + BindingDB/PlatinAI download + CPU SMILES featurization fix | Loaders exist; 1M+ BindingDB on disk |
| **Phase 1** (P1 HIGH) | 1 week GPU-0/1 | MetalCytoToxDB 26K + NCI-60 7.8K | Wf-146 LOMO completion (24/25) + Wf-131 P3 (7/25) + Wf-135 Rh regression | LOMO 5-metal ≥ 0.60 + per-metal AUC ≥ 0.86 (Ru gate) |
| **Phase 2** (P1 HIGH) | 1 week GPU-0/1 | DrugOOD 1.58M | DrugOOD full chemprop v2 retrain (3 splits × 5 seeds × 50 epochs) | OOD AUC ≥ 0.75 (gate) |
| **Phase 3** (P1 HIGH) | 1 week GPU-0/1 | BindingDB 1.07M | BindingDB metal-filtered chemprop v2 (1M+ × 5 seeds) | Affinity Pearson r ≥ 0.65 (Boltz-2 baseline 0.86) |
| **Phase 4** (P1 HIGH) | 1 week GPU-0/1 | tmQM 100K + PlatinAI 500 | tmQM d-block pretrain + PlatinAI 4-cell retrain | R² ≥ 0.70 (cross-dataset macro) |
| **Phase 5** (P1 HIGH) | 2 weeks GPU-0/1 | Joint retrain | M22-M30 mechanisms + 4 joint retrain (R19-P1..P6) | paper §4 ACTUAL numbers |
| **Phase 6** (P1 HIGH) | 1 week CPU | n/a | Wf-138 paper v2 submission package | paper builds clean + JCIM checklist |

---

## §1 P0 Critical Blockers (must address first)

### §1.1 Three missing loader modules

Per TODO-148 §2.1 audit:
- `molmetal/data/drugood_loader.py` — DrugOOD 1.58M accessible via pandas direct read, but NO module API
- `molmetal/data/binding_db.py` — BindingDB subset 5K has NO module API
- `molmetal/data/nci60.py` — NCI-60 7.8K has NO module API

**Action**: Add 3 loader modules in 1-2 days each, with ≥12 tests each, following pattern from `molmetal/data/tmqm.py` and `molmetal/data/cytotox.py`.

### §1.2 Data download gaps

- **BindingDB full 1.07M**: Currently only 5,257 subset on disk. Need to download full BindingDB (TSV.gz from bindingdb.org), filter to metal-binding entries, expect ~50K-200K rows after metal filter.
- **PlatinAI full 226K**: GitHub URL 404, ChemRxiv gated, Zenodo no match. Either: (a) accept 500-mol subset as honest subset, (b) request from authors, (c) drop from paper §4.

### §1.3 CPU SMILES featurization bottleneck

Per TODO-155 + chemprop_v2_status_2026_09_27.json:
- Wf-155 chemprop v2 full 5×5×50 TIMED OUT at iteration 1 (Ru seed=0, 19135 samples, 50 epochs did not finish in 6:50 wall)
- Bottleneck: chemprop v2 per-batch SMILES featurization on CPU (GPU 2-4%, CPU 660%+)
- Fix: precompute features once per epoch + faster encoder pipeline (~30 min engineering + 1 day validation)

### §1.4 Standing GPU "kill then adapt" directive

Per INDEX.md + TODO-148 §0:
- All GPU work DEFERRED pending explicit fresh user authorization
- Standing rule: subagent harness budget 3600s (1 hr); 92-min GPU launch would die mid-run + corrupt resume state
- Resolution: explicit user OK before any > 1 hr GPU launch

---

## §2 Phase 1 — Wf-146 + Wf-131 + Wf-135 LOMO completion (1 week GPU)

### §2.1 Wf-146 chemprop v2 LOMO sweep completion (24/25 remaining)

**Source**: `metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json` (PARTIAL_BLOCK as of 16:50:00)
**Status**: 1/25 completed (Rh seed=0 = 0.5256). 24 remaining = Rh seeds 1-4 + Ru seeds 0-4 + Ir seeds 0-4 + Os seeds 0-4 + Re seeds 0-4
**ETA**: ~60 min wall (chemprop v2 smoke backbone hidden=64/depth=2)
**Honest caveat**: Wf-146 uses smoke backbone (hidden=64/depth=2), NOT production 300/3 (per TODO-155 §2.2)

**Production-grade alternative** (Wf-146 PRODUCTION):
- backbone: hidden=300, depth=3 (chemprop v2 default)
- ETA: 5.1× longer = ~5 hr wall (per TODO-154 §3 MEASURED 0.193 min/run × 25 = 4.8 min, ×300/64 = ~30 min/run × 25 = ~12 hr; NOT 5 hr)
- Decision: ship SMOKE first (60 min), then PRODUCTION (12 hr) if gate met

**Action**:
```bash
cd /home/hugo/codes/mol-metal
TRITON_USE_FUSED=0 CUDA_VISIBLE_DEVICES=1 \
  /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/wf146_chemprop_lomo.py \
      --metals Rh Ru Ir Os Re \
      --seeds 0 1 2 3 4 \
      --epochs 50 \
      --hidden 64 \
      --depth 2 \
      --device cuda:1 \
      --time-budget-s 7200
```

### §2.2 Wf-131 P3 D-MPNN LOMO completion (7/25 remaining)

**Source**: `metrics/by_metric/chemprop_v2_status_2026_09_27.json` `actually_interrupted_workflow_identified`
**Status**: 18/25 done (Ru 5/5 + Ir 5/5 + Os 5/5 + Rh 3/5). Re 0/5. Remaining 7 = Rh seed3, Rh seed4, Re seed0-4.
**ETA**: ~92 min wall (per chemprop_v2_status_2026_09_27.json)
**Wrapper preserved**: `/tmp/wf131_resume.sh` (idempotent re-run skips already-shipped seeds)

**Action**:
```bash
cd /home/hugo/codes/mol-metal
bash /tmp/wf131_resume.sh  # or equivalent
# Manual version:
CUDA_VISIBLE_DEVICES=1 /home/hugo/codes/mol-metal/.venv/bin/python \
  molmetal/scripts/train_metal_lomo_wasserstein.py \
  --metals Rh Re --seeds 3 4 0 1 2 \
  --epochs 50 --wasserstein-weight 0.05 \
  --use-rh-oversample --device cuda:1 \
  --time-budget-s 7200
```

### §2.3 Wf-135 m19 LOMO Rh oversample regression fix (TODO-135)

**Source**: TODO-135 master plan (NEW pending 2026-09-27)
**Issue**: Wf-125 P2 honest-negative Rh LOMO = 0.136 (REGRESSED from 0.368 baseline) due to oversample label-bias
**Fix**: positive-label oversampling OR per-metal head ensemble OR `pos_weight=10` for Rh fold
**Goal**: Rh fold AUC ≥ 0.30 (recover from 0.136 regression)
**ETA**: ~4 hr wall (5-seed × 50-epoch + Rh oversample fix)

---

## §3 Phase 2 — DrugOOD 1.58M full chemprop v2 retrain (1 week GPU)

### §3.1 Goal

Replace the current metacytotox_fallback 5K subset (drugood_ood_auc.json) with chemprop v2 on the FULL DrugOOD 1.58M IC50 dataset (per `drugood_ood_auc_native_smoke.json` `native_data_shipped: true`, `native_data_n_rows: 1583302`).

### §3.2 Spec

- **Data**: DrugOOD IC50 full train.csv = 1,583,302 rows (138 MB CSV on disk)
- **Splits**: assay / scaffold / size (3 splits × 3 seeds)
- **Model**: chemprop v2 (hidden=300, depth=3, 50 epochs default)
- **Validation target**: OOD AUC ≥ 0.75 across 3 splits (current 10K subsample MLP proxy = 0.56-0.60)
- **ETA**: ~10 hr wall (single RTX 5090, assuming 0.193 min/run × 9 splits + data load overhead)

### §3.3 Loader module prerequisite

Per §1.1, must add `molmetal/data/drugood_loader.py` first. Then:

```bash
cd /home/hugo/codes/mol-metal
TRITON_USE_FUSED=0 CUDA_VISIBLE_DEVICES=1 \
  /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/train_drugood_full.py \
      --data-path .data/molmetal/drugood/ic50/data/train.csv \
      --splits assay scaffold size \
      --seeds 0 1 2 \
      --epochs 50 \
      --hidden 300 --depth 3 \
      --device cuda:1 \
      --time-budget-s 43200  # 12 hr
```

---

## §4 Phase 3 — BindingDB full 1.07M retrain (1 week GPU)

### §4.1 Goal

Replace the current BindingDB subset 5K (used in Wf-49 R2 cross-dataset R² +0.243 macro) with chemprop v2 on the FULL BindingDB 1.07M filtered to metal-binding entries (expect 50K-200K rows after metal filter).

### §4.2 Spec

- **Data**: BindingDB full 1.07M → metal-filtered subset (~50K-200K)
- **Targets**: Ki / Kd / IC50 affinity (Boltz-2 baseline Pearson r = 0.86)
- **Model**: chemprop v2 (hidden=300, depth=3, 50 epochs)
- **Validation target**: Affinity Pearson r ≥ 0.65 (project 0.75 if 200K rows; MEASURED-dependent)
- **ETA**: ~7 hr wall (single RTX 5090)

### §4.3 Loader module prerequisite

Per §1.1, must add `molmetal/data/binding_db.py` first.

---

## §5 Phase 4 — tmQM d-block + PlatinAI 4-cell joint retrain (1 week GPU)

### §5.1 tmQM 100K d-block pretrain

Already SHIPPED via F5_RDKIT_2D + Ridge baseline (`tmqm_combined_r2.json`, macro R² = +0.243). PRODUCTION chemprop v2 retrain not yet executed.

### §5.2 PlatinAI 4-cell chemprop v2 retrain

Currently: 500-mol subset + Wf-86 P3 PlatinAI 4-cell real (`Wf-86 P3 ship`, R² near predict-the-mean). Goal: chemprop v2 on PlatinAI 4-cell for at least MCF7/A549/A2780/A2780cis (per-cell D-MPNN exists in `nci60_precious_9cell_auc.json` 9-cell mean 0.8007).

### §5.3 Joint retrain spec

- **Data**: tmQM 100K (pretrain) + PlatinAI 4-cell 500 (fine-tune)
- **Model**: chemprop v2 (hidden=300, depth=3)
- **Validation target**: cross-dataset R² ≥ 0.70 (current macro +0.1248 → target +0.70 lift via chemprop v2 backbone)
- **ETA**: ~3 hr wall

---

## §6 Phase 5 — M22-M30 mechanism + 4 joint retrain (2 weeks GPU)

Per R19_PLAN.md §3.1 (R19 ship cycle = 12 weeks Blackwell + RTX 5090 dual):

| Mechanism | Source | Retrain time | Gate |
|---|---|---|---|
| **M22 SE(3) SB** | TODO-148 + R18_PLAN.md §15 | 2 weeks | m3 Vina Dock ≤ −10.5 kcal/mol |
| **M23 Transport Relax** | R18_PLAN.md §15 | 1 week | m14 Decode ratio ≥ 1.000 |
| **M24 Multi-PH** | R18_PLAN.md §15 | 2 weeks | m7 Diversity ≥ 0.75 |
| **M25 Hilbert PD** | R18_PLAN.md §15 | 1 week | wall-time −50% |
| **M26 Score-Rep MC** | R18_PLAN.md §15 | 1 week | variance reduction ≥ 0.20 |
| **M27 W-Flow ODE** | R18_PLAN.md §15 | 1 week | ODE 6ms → 0.06ms (100× speedup) |
| **M28 MacroGuide** | R18_PLAN.md §15 | 1 week | m11 n_distinct +5-15% |
| **M29 Coord GNN** | R18_PLAN.md §15 | 1 week | coordination compliance ≥ 0.95 |
| **M30 Quotient Diff** | R18_PLAN.md §15 | 1 week | (NEW mechanism) |
| **Joint: M25 + M26** | LAMBDA_ROADMAP.md §1.4 | 1 week | score-repellent + Hilbert R² ≥ 0.85 |
| **Joint: M27 + M28** | LAMBDA_ROADMAP.md §1.4 | 1 week | W-Flow + MacroGuide combined |
| **Joint: M29 + M30** | LAMBDA_ROADMAP.md §1.4 | 1 week | Coord GNN + Quotient Diff |
| **Joint: M22 + M23 + M24** | LAMBDA_ROADMAP.md §1.4 | 1 week | SE(3) + Transport + Multi-PH |

**Total ETA**: 12 weeks wall (per R19_PLAN.md); but can parallelize M22-M30 on 2 GPUs → 6 weeks

---

## §7 Phase 6 — Wf-138 paper v2 submission (1 week CPU)

Per TODO-138 + IMPROVEMENT_QUEUE_2026_09_27.md:
- **Inputs**: Wf-134 + Wf-135 + Wf-139 P5 complete + chemprop v2 numbers (Phase 1-5)
- **Action**: Compile paper v2 + cover letter + figures + supplementary S1-S8
- **Verify**: paper builds clean (0 unresolved refs) + JCIM submission checklist complete
- **Constraint**: per `TODO/pending/84_no_paper_task_without_authorization.md` — paper work BLOCKED pending user authorization

---

## §8 Compute budget summary

### §8.1 Wall-time estimate (single RTX 5090)

| Phase | Wall-time | GPU |
|---|---:|---|
| Phase 0 (P0 blockers) | 1 week CPU | 0 |
| Phase 1 (Wf-146 + Wf-131 P3 + Wf-135) | 1 week GPU | 1 |
| Phase 2 (DrugOOD full) | 1 week GPU | 1 |
| Phase 3 (BindingDB full) | 1 week GPU | 1 |
| Phase 4 (tmQM + PlatinAI) | 1 week GPU | 1 |
| Phase 5 (M22-M30 + 4 joint) | 2-6 weeks GPU | 2 (Blackwell + 5090 parallel) |
| Phase 6 (paper v2) | 1 week CPU | 0 |
| **TOTAL** | **8-12 weeks** | mixed |

### §8.2 GPU hours (Blackwell 95.6 GB + RTX 5090 31.8 GB dual)

- Total GPU hours: ~480 hr (12 weeks × 40 hr/week) on 2 GPUs
- Blackwell dominance for M22-M30 retrain (large checkpoints)
- RTX 5090 for chemprop v2 + DrugOOD + BindingDB

### §8.3 Risk factors

- **GPU OOM risk**: BindingDB full 200K chemprop v2 + DrugOOD 1.58M subset may OOM
- **Ultrahigh std risk**: per-metal std (Rh/Os <200 rows) likely STABLE_BUT_HIGH_STD (per TODO-120 §3.2 honest projection)
- **Chemprop v2 SMILES featurization bottleneck**: 6:50 wall per chemprop_v2_status_2026_09_27.json; need CPU fix
- **Standing "kill then adapt" directive**: requires explicit fresh OK for any GPU > 1 hr launch

---

## §9 Acceptance gates (per IMPROVEMENT_QUEUE_2026_09_27.md §1)

1. **Implementation Gate**: script/module exists + imports + callable
2. **Test Gate**: pytest ≥12 tests PASS (per task §X verify)
3. **Metric Gate**: number MEASURED (no fabrication)
4. **Honest-Framing Gate**: failure cases ship honest-negative (per party-twenty-spirit §3.5)
5. **Registry Gate**: `metrics/by_metric/<metric>.json` REFRESHED
6. **Completion Doc Gate**: `TODO/completed/<WF_ID>_*.md` ship

If any gate fails, **HALT pipeline** — do not enter next phase.

---

## §10 Per-metric honest gates

| Metric | Gate | Source | Status |
|---|---|---|---|
| **m1 Ru AUC** | ≥ 0.86 (Krasnov + 0.05) | chemprop v2 Ru per-metal fold | MEASURED 0.9095 (TODO-154 §2; gate PASS) |
| **m2 Pt A2780cis** | ≥ 0.85 (Rusanov) | chemprop v2 Pt A2780cis | MEASURED 0.7278 (within 9-cell 0.8007 mean); not per-A2780cis |
| **m3 Vina Dock** | ≤ −8.5 (DecompDiff match) | SE(3) SB retrain | PROJECTED −10.5 (M22 GPU retrain) |
| **m4 Success Rate** | ≥ 0.60 | cascades from m3 | GAP — cascades from m3 |
| **m7 Diversity** | ≥ 0.65 | Multi-PH + PLL | MEASURED 0.1065; PROJECTED 0.75 (M24 GPU retrain) |
| **m14 CFM decode** | ≥ 1.000 (no regression) | Path B decoder | MEASURED 1.000 (YuelBond) |
| **m16 Affinity Pearson r** | ≥ 0.65 (Boltz-2 baseline 0.86) | BindingDB chemprop v2 | MEASURED 0.207 (D-MPNN); PROJECTED 0.65 (chemprop v2 + BindingDB full) |
| **m17 DrugOOD OOD** | ≥ 0.75 (3-split) | chemprop v2 DrugOOD full | MEASURED 0.891/0.697/0.748 (MLP proxy on 5K subset); PROJECTED ≥0.75 chemprop v2 on 1.58M |
| **m19 LOMO** | ≥ 0.70 (cross-metal) | Wf-146 chemprop v2 LOMO | MEASURED 0.535 (D-MPNN); MEASURED 0.4568 (Wf-125 P2 W₂); MEASURED 0.5256 (Wf-146 Rh seed0 smoke) |
| **m21 NCI-60 9-cell** | ≥ 0.81 (Krasnov match) | D-MPNN k-cell | MEASURED 0.8007 (within ±0.01) |

---

## §11 Honest framing note

Per `party-twenty-spirit` §3.5:
- All wall-time figures are **PROJECTED**, not measured
- All "PASS gate" projections are **CONDITIONAL** on user authorization for GPU retrain (standing "kill then adapt" directive)
- DrugOOD full retrain **requires loader module fix** (§1.1 P0)
- BindingDB full retrain **requires download** (~1 GB TSV.gz) and **metal filter** (~50K-200K rows after filter)
- PlatinAI full retrain **NOT possible** (URL 404, gated); honest subset is 500 mol
- All chemprop v2 numbers from Wf-124 P3 head-to-head are MEASURED (n=14 paired runs, t=6.754, p=1.35e-5)
- Honest framing for paper §4: 8 STRONG metrics lead, 8 WEAK metrics honest-negative in §4.5, R19/P22 PROJECTED until GPU retrain

---

## §12 Cross-references

- **`TODO/completed/160_big_dataset_retrain_audit_2026_09_27.md`** — Audit (1.92M on disk, 7.6% utilized)
- **`TODO/completed/82_metric_gap_filling_plan.md`** — R18 metric gap filling (7 gaps)
- **`TODO/03_plans/R17_AUC_RECOVERY_PLAN.md`** — R17 Tier-1 AUC fix
- **`TODO/03_plans/R18_PLAN.md`** — R18 detailed plan (47-item completion matrix)
- **`TODO/03_plans/R19_PLAN.md`** — R19 GPU retrain cycle (12 weeks Blackwell + RTX 5090)
- **`TODO/03_plans/WEAK_TO_STRONG_PLAN.md`** — 8 STRONG + 8 WEAK + paper framing
- **`TODO/03_plans/LAMBDA_ROADMAP.md`** — R20-R25 Lambda algorithm (8 gaps L1-L8)
- **`TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md`** — Sprint A acceptance gates
- **`TODO/completed/148_wf146_lomo_retrain_prep_status.md`** — Wf-146 prep status
- **`TODO/completed/155_pivot_to_chemprop_v2.md`** — chemprop v2 pivot (V4 → chemprop_v2)
- **`TODO/completed/154_chemprop_v4_head_to_head.md`** — chemprop v2 vs V4 head-to-head
- **`TODO/completed/131_m1_m19_joint_rescue.md`** — Wf-131 m1 + m19 joint (PARTIAL SHIP)
- **`TODO/completed/120_dmpnn_seed_integrity.md`** — D-MPNN ckpt integrity (P1+P3 SHIP)
- **`TODO/completed/121_m19_lomo_strengthen.md`** — m19 LOMO strengthen via W₂ + Rh oversample
- **`TODO/completed/100_wf71_p3_wf49_cross_round2.md`** — cross-dataset R² +0.1248 macro
- **`TODO/03_plans/R18_PLAN.md`** — R18 plan +47-item completion matrix
- **`TODO/INDEX.md`** — Single source of truth (272+ files, 18 truly pending)
- **`TODO/EXECUTION_ORDER.md`** — Priority order (R18 done, R19 in flight)
- **`DATA_STATUS.md`** — Local dataset inventory
- **`metrics/by_metric/chemprop_v2_5metal_5seed_50epoch.json`** — Wf-146 LOMO in flight
- **`metrics/by_metric/chemprop_v2_status_2026_09_27.json`** — chemprop v2 status snapshot
- **`metrics/by_metric/dmpnn_ckpt_status.json`** — 25 D-MPNN ckpts (5 metals × 5 seeds)
- **`metrics/by_metric/metal_lomo_auc_wasserstein.json`** — Wf-125 P2 0.4568 BELOW GATE
- **`metrics/by_metric/nci60_precious_9cell_auc.json`** — 9-cell mean AUC 0.8007
- **`metrics/by_metric/drugood_ood_auc.json`** — 3-split OOD AUC 0.891/0.697/0.748
- **`metrics/by_metric/drugood_ood_auc_native_smoke.json`** — DrugOOD native 10K subsample
- **`metrics/by_metric/metalcytotox_combined_r2.json`** — 5-metal R² aggregated
- **`metrics/by_metric/tmqm_combined_r2.json`** — tmQM 100K d-block R²
- **`.claude/skills/party-twenty-spirit/SKILL.md`** — §3.5 honest framing anchor
- **`.claude/skills/mol-metal-gpu-execution/SKILL.md`** — Blackwell + RTX 5090 dual-card context