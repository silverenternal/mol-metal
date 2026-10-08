# COMPARISON_BASELINES — Selected Comparison Baselines + Protocol Alignment Matrix

**Date**: 2026-10-01 (Wf-431 docs-only refactor)
**Status**: Active — single source of truth for which papers we compare against
**Purpose**: Formalize selected baselines + per-metric protocol alignment matrix
**Owner**: Mol-Metal docs agents (party-twenty-spirit §3.5 honest framing)
**Cross-references**: `TODO/02_metrics/METRICS_TARGETS.md` (per-metric target values) + `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` (quick reference)

---

## §0 TL;DR — principle + 3 selected baselines

### Principle (per user directive 2026-10-01)

> **我们不需要 reproduce 别人的数字 — 我们需要 same dataset + same protocol + same metric 来 make results comparable.**

Numerical comparison 不要求 bit-identical reproduction,只要:
- Dataset 相同 (e.g. MetalCytoToxDB, PlatinAI, DrugOOD)
- Split protocol 相同 OR 同方法学 (e.g. scaffold-split)
- Metric 相同 (e.g. binary classification AUC, R², median kcal/mol)

Protocol difference (5-fold CV vs scaffold-split) is OK — document it in §6 honest limitations.

### Selected baselines (Tier-1 direct comparison)

| # | Baseline | Dataset | Why selected |
|---|---|---|---|
| 1 | **Krasnov 2026** (J. Med. Chem.) | MetalCytoToxDB | 同领域 (precious-metal anticancer) + 同任务 (binary AUC) + 同 5-metal subset (Ru/Ir/Rh/Re/Os) |
| 2 | **Rusanov 2025** (ChemRxiv) | PlatinAI | 同金属 (Pt) + 同目标 (cisplatin-resistance) + 4 cell-line |
| 3 | **DrugOOD 2022** (ICLR) | DrugOOD | SBDD OOD generalization baseline + iMoLD uses this dataset |

### Out-of-scope baselines (explicitly NOT compared against)

See §2 for full list + rationale.

---

## §1 Selected comparison baselines

### 1.1 Krasnov 2026 (J. Med. Chem.) — MetalCytoToxDB

**Citation**: Krasnov et al., *J. Med. Chem.* 2026-04 (Tier 1)
**Task**: Ru/Ir/Rh/Re/Os 抗癌活性分类 (binary classification + pIC50 regression)
**Why selected**: 同领域 + 同任务 + 同 5-metal subset (Ru/Ir/Rh/Re/Os)
**Dataset**: MetalCytoToxDB (公开; per `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` schema reference)
**Paper split**: DOI-split 10-fold CV (primary; Krasnov paper Table 4 — see §6.2, the earlier "5-fold CV" reading was superseded 2026-10-04)
**Paper metric**: binary AUC + pIC50 MAE

**Our alignment**:
- Dataset: **IDENTICAL** (MetalCytoToxDB) ✓
- Split: **DIFFERENT** in the m1 replication (we adopted the paper's DOI-split 10-fold CV for m1 per §6.2; the m21 NCI-60 line still uses scaffold-split 70/10/20 per Wf-261) — note in §6
- Metric: **IDENTICAL** (binary AUC) ✓

**Applicable metrics**:
- **m1 Ru subset binary AUC**: 0.7717 ± 0.0380 vs Krasnov 0.81 (DOI-split 10-fold CV) — MEASURED, see §6.2
- **m21 NCI-60 9-cell AUC**: target ≥ 0.70 (Ru fold) vs Krasnov 0.81 (single-cell-line avg)
- **m15 Docking RMSD**: NOT directly comparable (different task) — for internal sanity check only

### 1.2 Rusanov 2025 (ChemRxiv) — PlatinAI

**Citation**: Rusanov et al., ChemRxiv 2025-06
**Task**: Pt 配合物 de novo + 4 cell-line (MCF-7, A549, A2780, A2780cis)
**Why selected**: 同金属 (Pt) + 同目标 (cisplatin-resistance, A2780cis)
**Dataset**: PlatinAI (公开; 4 cell-line subset)
**Paper split**: per-cell-line split
**Paper metric**: per-cell-line binary AUC

**Our alignment**:
- Dataset: **IDENTICAL** (PlatinAI) ✓
- Split: **APPROXIMATELY ALIGN** (paper: per-cell-line; us: cohort-split 70/10/20 per Wf-388-A2 + per-cell-line evaluation)
- Metric: **IDENTICAL** (per-cell-line binary AUC) ✓

**Applicable metrics**:
- **m2 R² per-cell-line**: target ≥ per-line baseline per Rusanov paper Table S4-S5
- **m19 LOMO multi-metal transfer**: target ≥ 0.70 (Pt fold primary)

### 1.3 DrugOOD 2022 (ICLR) — OOD generalization

**Citation**: Ji et al., ICLR 2022 — DrugOOD
**Task**: OOD generalization across 3 splits (assay / scaffold / size)
**Why selected**: SBDD OOD generalization baseline + iMoLD uses this dataset
**Dataset**: DrugOOD (公开; 1.58M rows per `molmetal/reports/wf265_collect/all_results.json`)
**Paper split**: assay / scaffold / size splits (官方)
**Paper metric**: OOD binary AUC (mean over 3 splits)

**Our alignment**:
- Dataset: **NOT ALIGN** — our shipped run used the `metacytotox_fallback` cohort (MetalCytoToxDB-derived, 5,207 rows), NOT the official DrugOOD IC50 splits. The official DrugOOD CSVs are on disk at `.data/molmetal/drugood/ic50/csv/` (assay/scaffold/size × lbap_core / lbap_refined / ec50 / ic50 / ki / potency); m17 regeneration on them is **pending wf-R34-B2**.
- Split: **NOT ALIGN** — the assay/scaffold/size split labels were reproduced on the fallback cohort, so they are DrugOOD-*style* splits over the wrong cohort, not DrugOOD's own splits.
- Metric: **IDENTICAL** (binary AUC) ✓

**Applicable metrics**:
- **m17 OOD generalization AUC**: target ≥ iMoLD - 0.02 = 0.6909 (assay) per `metrics/by_metric/drugood_ood_auc.json` — **gate arithmetic only, NOT a protocol-aligned comparison**. The metric matches iMoLD; the cohort does not. Per the same JSON's `honest_framing` and `native_data_imold_proxy_results` (proxy-MLP on a 10K subsample scored 0.5625 / 0.5999 / 0.5813 vs iMoLD 0.7211 / 0.6884 / 0.6792 = −0.159 / −0.089 / −0.098), we explicitly do NOT claim to beat iMoLD. See §6.6.

---

## §2 Out-of-scope baselines (explicitly NOT compared against)

Per user directive 2026-10-01, we do NOT need to compare against:

### 2.1 Different task (SBDD 3D pose generation vs binding affinity)
- **DiffDock** (2022), **DeltaDock** (2023), **EquiBind** (2022), **TransDiffSBDD** (2025) — 3D pose generation, not 2D-3D binding affinity prediction

### 2.2 Different dataset
- **Multimodal Protein-Metal Binding 2025** (newx.sg) — protein-level binding, not small-molecule anticancer
- **PDBbind** — protein-ligand binding, not precious-metal coordination
- **Boltz-2 2025** — protein-ligand complex prediction, different method class

### 2.3 Different method class
- **Chemprop 2019 / D-MPNN** — general MoleculeNet baseline, no precious-metal specialization (we use D-MPNN as our method, not as baseline)
- **REINVENT4 / MOSES / GuacaMol** — language-model-style molecule generation, different evaluation paradigm

### 2.4 General SBDD methods (not precious-metal specific)
- **Pocket2Mol** (2022), **TargetDiff** (2023), **MolCRAFT** (2024), **DecompDiff** (2024) — general SBDD methods for pocket-conditional generation; not in scope of Mol-Metal's precious-metal anticancer focus

These are NOT in scope of the Mol-Metal Tier-2 paper. We do NOT include them in §4 comparison tables.

---

## §3 Per-metric protocol alignment matrix

| Metric | Selected baseline | Paper dataset | Paper split | Paper metric | Our dataset | Our split | Our metric | Alignment level |
|---|---|---|---|---|---|---|---|---|
| **m1** Ru AUC | Krasnov 2026 | MetalCytoToxDB (v2) | DOI-split 10-fold CV (primary; + SMILES-split 10-fold CV alt) | binary AUC | MetalCytoToxDB v2 Ru subset — paper criteria adopted 2026-10-04: no exposure-time filter, no IC50 noise floor, measurement-level dedup (SMILES_Ligands, Cell_line, Time(h)) keep-first, 76-name non-cancer exclusion = 15,799 rows / 4,793 complexes / 591 cell lines; release md5 f270dc90… = Zenodo v2 | DOI-split 10-fold CV (replicated; folds DOI-disjoint) | binary AUC | **DATASET + METRIC + SPLIT ALIGN — cohort honest-negative** (MEASURED D-MPNN 0.7717 ± 0.0380 vs paper LightGBM 0.810 ± 0.024 = 0.038 below; paper cohort 11,294 / 4,399 / 23 not replicated; see §6) |
| **m2** R² per-cell | Rusanov 2025 | PlatinAI | per-cell-line | binary AUC + R² | PlatinAI + MetalCytoTox | cohort-split 70/10/20 + per-cell-line eval | per-cell-line AUC + R² | **DATASET + METRIC + SPLIT APPROX ALIGN** |
| **m3** Vina Dock | — none (see §2 out-of-scope) | CrossDocked100 | pocket-split | median kcal/mol | CrossDocked100 (5-pocket subset) | pocket-split | median kcal/mol | **NO FORMAL PAPER COMPARISON — Tier-2 supplementary** (no external baseline; our 5-pocket CrossDocked100 numbers are reported as absolute values only) |
| **m7** Diversity | — none (see §2 out-of-scope) | CrossDocked100 | pocket-split | Morgan ECFP4 Tanimoto | CrossDocked100 | pocket-split | Morgan ECFP4 Tanimoto | **NO FORMAL PAPER COMPARISON — Tier-2 supplementary** (no external baseline; Morgan ECFP4 Tanimoto reported as an absolute value only) |
| **m15** RMSD | — none (see §2 out-of-scope) | CrossDocked100 (PDBbind) | pocket-split | median Å | CrossDocked100 (5-pocket) | pocket-split | median Å | **NO FORMAL PAPER COMPARISON — Tier-2 supplementary** (no external baseline; median Å reported as an absolute value only) |
| **m17** OOD | iMoLD / DrugOOD 2022 | DrugOOD | assay / scaffold / size | binary AUC | **NOT ALIGN — fallback cohort** (MEASURED: our 5,207 rows are `metacytotox_fallback` = MetalCytoToxDB-derived, NOT the official DrugOOD IC50 splits; the official DrugOOD CSVs are on disk at `.data/molmetal/drugood/ic50/csv/` (assay/scaffold/size × 6 tasks) but the m17 regeneration is still pending (wf-R34-B2). Per `metrics/by_metric/drugood_ood_auc.json::honest_framing`, the current numbers are NOT a head-to-head comparison against iMoLD. Excluded from §4 comparison tables; see §6.6) |
| **m21** NCI-60 9-cell | Krasnov 2026 | MetalCytoToxDB | 5-fold CV per-cell | per-cell AUC | MetalCytoToxDB + NCI-60 panel | scaffold-split per-cell | per-cell AUC | **DATASET ALIGN** (split differs) |

### Alignment level legend

- **FULL ALIGN**: same dataset + same split protocol + same metric — directly comparable numerically
- **DATASET + METRIC ALIGN** (split differs): numerical values not directly comparable but per-seed/per-split statistics are; document split difference in §6
- **DATASET ALIGN** (split + metric differ): values comparable as sample-size-scaled estimates; document both in §6
- **NOT ALIGN**: do not include in §4 comparison tables
- **NOT ALIGN — fallback cohort**: only the metric definition matches; the dataset is a locally-derived stand-in for the paper's dataset (here: `metacytotox_fallback`, 5,207 MetalCytoToxDB-derived rows, used in place of the official DrugOOD IC50 splits). Numeric gates computed on it are internal sanity checks, never head-to-head comparisons. Report as an absolute standalone number + §6 limitation; regenerate on the official cohort before any comparison table (m17: pending wf-R34-B2)
- **NO FORMAL PAPER COMPARISON — Tier-2 supplementary**: no external baseline exists for this metric; report our value as an absolute measured number in paper §4 supplementary only. Do NOT place it in a comparison table next to a literature value.
- **DATASET + METRIC + SPLIT ALIGN — cohort honest-negative**: dataset + metric + split protocol aligned, but our cohort scope differs from the paper's (stated in §6) — the numeric comparison is a cross-cohort estimate, not a reproduction

---

## §4 Key distinction: protocol replication vs protocol alignment

### Protocol REPLICATION (NOT our goal)

- Exactly reproduce paper's protocol (same k-fold seed, same hyperparameter grid, same training script)
- We are NOT benchmarking whether we can reproduce others' numbers
- We do NOT need this for Tier-2 acceptance per AGENTS.md §5.4 + user directive 2026-09-29 "我们发二区就可以了"

### Protocol ALIGNMENT (OUR goal)

- Use same dataset + same metric + (may use different split protocol)
- Numerical comparison is meaningful IF:
  1. We document the protocol difference (paper used 5-fold CV, we used scaffold-split → note this in §6)
  2. We report per-seed/per-split statistics not just headline mean
  3. We acknowledge the protocol mismatch in §6 honest limitations

### What this means for paper writing

- §3 main results: report our MEASURED + comparable to baseline numbers (cite baseline per protocol alignment matrix)
- §6 honest limitations: explicitly note protocol differences (e.g. "Krasnov uses 5-fold CV; we use scaffold-split; both report Ru AUC on same MetalCytoToxDB subset")
- §4 ablations: split = protocol axis + dataset axis; can isolate the impact of each

---

## §5 Cross-references

- `TODO/02_metrics/METRICS_TARGETS.md` — per-metric target values + verdict gates
- `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` — quick reference table
- `TODO/03_plans/R19_PLAN.md` — actual GPU retrain plan (R19 phase)
- `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` — metric lift strategy
- `TODO/decisions.md` D-Wf-431 — this refactor decision
- `TODO/archive/reproduce_baselines.md.archived_2026_10_01` — previous "reproduce baseline" approach (DEPRECATED per new principle)

---

## §6 Honest limitations (m1 cohort alignment state)

### 6.1 m1 (Ru AUC) — current measured state (paper criteria adopted, 2026-10-04)

- **Decision (user, 2026-10-04)**: adopt the paper's criteria — drop and record the filters, replicate the paper's funnel, re-evaluate under DOI-split 10-fold CV (the former §6.2 Option A).
- **Cohort (MEASURED)**: `molmetal/data/cytotox.py::build_paper_cohort` on the release-identical v2 file (`.data/molmetal/MetalCytoToxDB.csv`, md5 `f270dc907dbdfede953063a14f0825a7` = Zenodo v2 10.5281/zenodo.17106822, 26,801 rows): IC50-present → Ru → measurement-level dedup `(SMILES_Ligands, Cell_line, Time(h))` keep-first → 76-name non-cancer exclusion (exact strip().upper() match) = **15,799 rows / 4,793 complexes / 591 cell lines** (active_rate 0.3123). Paper: 11,294 rows / 4,399 complexes / 23 cancer cell lines (from 18,895 raw; no exposure-time filter — 24–144 h kept as feature; no IC50 noise floor; binary cut 10 uM).
- **Split (MEASURED)**: DOI-split 10-fold CV — the paper's primary scheme; all rows sharing a DOI in one fold, DOI-disjoint verified; fold indices written by `molmetal/scripts/doi_split_cv.py` (`folds/fold_*.json`); no trainer or filter default changed.
- **Result (MEASURED, 10/10 folds)**: D-MPNN multi-task, 10 epochs, seed 0, batch 64, hidden 128, depth 3, lr 1e-3, val=none (no checkpoint selection) → mean AUC **0.7717 ± 0.0380**; per-fold 0.8007, 0.7691, 0.8141, 0.7652, 0.7203, 0.7424, 0.8174, 0.7043, 0.7723, 0.8115 (3/10 folds ≥ 0.81). vs paper Ru binary ROC-AUC **0.810 ± 0.024** (LightGBM, DOI-split 10CV, 23 cell lines) → **0.038 below**; model class differs (D-MPNN vs LightGBM); the paper's SMILES-split 0.894 was NOT run.
- **What still differs (honest-negative)**: the paper's cohort scope is not replicated — rows +4,505 (+39.9%), complexes +394 (+9.0%), cell lines 591 vs 23 (25.7×); 95% of the residual row gap (4,282 of 4,505) sits in the 568 cell lines outside our top-23-by-volume subset (that subset holds 11,517 of 15,799 rows, within 2% of the paper's 11,294) — neither the dedup-key lever (moves 412 rows = 2.2% of raw) nor exclusion breadth (2,446 rows) reproduces the paper's ~23-cell-line scope; unexplained, not tuned. Our non-cancer list (76 names) is our curation; the paper's inclusion criteria are documented in no Zenodo release (measured: zero matches for excluded/filter/dropped/dedup/criteria). The numeric comparison is a cross-cohort estimate, not a reproduction.
- **Shipped record (unchanged)**: the shipped 0.7613 (scaffold split; no recorded row-level filter; n_train_avg 2509 / n_test_avg 796 matching no measured rerun) stays untraceable; the new measurement is an additive entry (`paper_protocol_measurement`) in `metrics/by_metric/metalcytotox_ru_auc.json`.
- **Evidence**: `/tmp/wf_m1_paper_cv/summary.json` + `folds/fold_*.json` + `run{,a,b,c,a2,b2,c2}.log`; contaminated first pass excluded (15 resume-artifact rows in `summary.excluded_rows`), all 10 accepted folds cold-start (cold=10 / resumed=0); tests `molmetal/tests/test_paper_cohort_doi_cv.py` 5/5 PASS.

### 6.2 m1 decision record (closed) + retained default-path facts

- **Former open decision is closed**: Option A adopted and executed 2026-10-04; Option B (keep cohort, declare) superseded. Resolved vs the old flags: filters dropped + recorded; release identity established (v2, md5-identical); split scheme = paper's DOI-split 10-fold CV; dedup + non-cancer funnel replicated. No open protocol item remains; the cohort-size gap above is the standing honest-negative for paper §6.
- **Retained default-path facts (NOT the m1 comparison numbers)**: default `CytotoxFilter` (time_threshold=24.0, ic50_min=0.01; `train_dmpnn_multitask_wasserstein.py:3449`) keeps 3,650/18,987 v4 Ru rows = 19.2%; measured reruns 2,262/394/994 AUC 0.5727 (10 ep) / 0.5901 (30 ep); loose 12,471/2,166/4,350 AUC 0.7622 (10 ep).
- **Flag (not edited here)**: §1.1's "Paper split: 5-fold CV (Krasnov paper Table 4)" line and the "(5-fold CV)" parenthetical at the §1.1 m1 bullet are superseded by the verified ChemRxiv v2 reading (10-fold CV; DOI-split primary + SMILES-split alt).

### 6.6 m17 (OOD generalization) — fallback cohort, regen pending

- **What we measured (MEASURED)**: `metrics/by_metric/drugood_ood_auc.json` (dated 2026-09-25, v4_morgan_mlp + Morgan FP 2048 r2, 30 epochs, 3 seeds) on `data_source = metacytotox_fallback` — MetalCytoToxDB-derived, 5,207 rows — using DrugOOD-*style* assay/scaffold/size splits. Gate pass 3/3 (assay / scaffold / size), `overall_pass = true` against the `iMoLD - 0.02` margin.
- **Why that is not a comparison**: the cohort is a stand-in, not DrugOOD. The same JSON's `honest_framing` states plainly that these are NOT head-to-head numbers against iMoLD on the actual DrugOOD IC50 splits. The `native_data_imold_proxy_results` block (10K subsample + LogisticRegression proxy, not the graph NN iMoLD uses) scores 0.5625 / 0.5999 / 0.5813 vs iMoLD 0.7211 / 0.6884 / 0.6792 = −0.1586 / −0.0885 / −0.0979. We do NOT claim to beat iMoLD.
- **What unblocks it**: the official DrugOOD IC50 CSVs are on disk at `.data/molmetal/drugood/ic50/csv/` (`lbap_core_ic50_{assay,scaffold,size}.csv`, `lbap_refined_ic50_*`, plus ec50 / ki / potency variants; native consolidated `train.csv` = 1,583,302 rows). Regenerating m17 on that cohort with a graph-NN backbone is **pending wf-R34-B2**.
- **Paper consequence**: until that lands, m17 is reported as an absolute measured OOD AUC on the fallback cohort, with the cohort substitution stated here — it is excluded from the §4 comparison table.
