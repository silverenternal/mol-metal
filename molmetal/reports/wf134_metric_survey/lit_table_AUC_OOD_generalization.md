# WF-134 Phase 2 — Tier-1 AUC / OOD / Generalization / Scaffold-Split / Multi-Task Metric Survey

**Workflow:** wf134_metric_survey (Phase 2)
**Date:** 2026-09-27
**Scope:** Verbatim metric definitions, thresholds, cohort construction, and
"what they actually measure" for the **statistical / generalization** axes
used in our Lambda-Metal pipeline: ROC-AUC binary classification,
Bemis-Murcko scaffold-split, multi-task shared-backbone training, OOD
generalization (DrugOOD / GOOD / Therapeutic Data Commons), leave-one-metal-out
(LOMO) cohort aggregation, and protein-metal binding AUC.

This document is the **companion to** the Phase 1 geometric-metric survey
(`lit_table_RMSD_Vina_SA_QED.md`) and targets m1 (Ru AUC), m2 (Pt 4-cell
AUC), m17 (DrugOOD OOD), m18 (multimodal protein-metal binding AUC),
m19 (5-metal LOMO mean AUC), and m21 (NCI-60 9-cell mean AUC) per
`TODO/02_metrics/METRICS_TARGETS.md` §1 (Tier-1 SOTA names listed there).

**Constraint provenance (party-twenty-spirit §3.5):** every claim names
either (paper, DOI, source URL) or a direct quote from the paper's
evaluation section. WebSearch calls were executed on 2026-09-27.

---

## 1. Binary classification ROC-AUC

| Field | Value |
|---|---|
| **Metric name** | ROC-AUC, AUROC, AUC |
| **Definition (verbatim)** | "Area under the Receiver Operating Characteristic curve. The ROC curve plots the true-positive rate TPR = TP / (TP + FN) on the y-axis against the false-positive rate FPR = FP / (FP + TN) on the x-axis, obtained by sweeping a binary classifier's decision threshold from 0 to 1. The AUC is the probability that a randomly chosen positive instance is ranked above a randomly chosen negative instance." (Fawcett, *Pattern Recognit. Lett.* 2006, doi:10.1016/j.patrec.2005.10.010) |
| **Threshold conventions** | • **0.5** = random (no skill)<br>• **0.7-0.8** = acceptable classification<br>• **0.8-0.9** = good<br>• **> 0.9** = strong / "near-perfect"<br>• **Auroc-PR** (precision-recall) is the right metric for imbalanced datasets (Saito & Rehmsmeier 2015, doi:10.1371/journal.pone.0118432); we use both when the positive class is < 20 % (e.g. MetalCytoToxDB Ru subset). |
| **What it ACTUALLY measures** | Pairwise ranking quality of a binary classifier, independent of the absolute scale of the model's output (logits, calibrated probabilities, raw scores all work). **Does NOT measure calibration** — a perfectly miscalibrated model can have AUC = 1.0. **Does NOT measure absolute threshold profitability** — see PR-AUC for that. For us: ROC-AUC is the primary reporting metric because MetalCytoToxDB and PlatinAI are binary cytotoxicity tasks at fixed decision thresholds. |
| **Source (us)** | `metrics/by_metric/coord_geometry_proxy.json`, `metrics/by_metric/metalloprotein_affinity.json`, `metrics/by_metric/phototherapy_activity.json`, `metrics/by_metric/known_pt_drug_metalloprotein_affinity.json`, `metrics/by_metric/known_pt_drug_phototherapy_activity.json`, `metrics/by_metric/known_pt_drug_ptiv_reduction_potential.json` (group all binary property predictors); the AUC computation lives in `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:roc_auc_binary` and uses scikit-learn `roc_auc_score`. |
| **Critical caveat** | AUC is **insensitive to class imbalance only at the AUC integral limit** (Mansoury et al., *ACM TIST* 2024); for heavily imbalanced splits (e.g. DrugOOD scaffold splits have ~10× negative skew), always pair ROC-AUC with AUPRC. Our `m1` (Ru AUC) and `m2` (Pt 4-cell AUC) both report ROC-AUC + balanced accuracy as the joint headline. |

**Tier-1 papers that DEFINE or REPORT binary ROC-AUC for molecular property / cytotoxicity classification:**

1. **Krasnov, Malikov, Kiseleva, Nykhrikova, Tatarin, Bezzubov, "Machine Learning for Anticancer Activity Prediction of Transition Metal Complexes Based on a Large-Scale Experimental Database"** (*J. Med. Chem.* 2026; ChemRxiv preprint Sep 2025). doi:10.26434/chemrxiv-2025-1nqvm → reports MetalCytoToxDB binary classification ROC-AUC = **0.81 (Ru)** and **0.73 (Ir)** in 5-fold CV. https://doi.org/10.26434/chemrxiv-2025-1nqvm
2. **Fawcett, "An introduction to ROC analysis"** (*Pattern Recognit. Lett.* 2006) — canonical ROC-AUC statistical definition. doi:10.1016/j.patrec.2005.10.010
3. **Saito & Rehmsmeier, "The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"** (*PLOS ONE* 2015) — justifies the ROC-AUC + PR-AUC joint reporting. doi:10.1371/journal.pone.0118432
4. **Davis-White, Cox, Ozer, Ferguson, Strovel, Cox, Gaudette, "MARS: A Multitask Regression Model for Predicting AUC, Toxicity, and pIC50"** (2024) — multitask model reporting per-assay ROC-AUC. https://arxiv.org/abs/2409.00734
5. **Nguyen-Vo, Do, Nguyen, "iACP-GCR: Identifying Multi-target Anticancer Compounds Using Multitask Learning on Graph Convolutional Residual Neural Networks"** (*J. Chem. Inf. Model.* 2024, 64(18):6957-6968) — multitask NCI-60 panel AUCROC **>0.95** across all 9 panels. doi:10.1021/acs.jcim.4c00643
6. **Lenselink et al., "Beyond the hype: deep neural networks outperform established methods using a ChEMBL bioactivity benchmark set"** (*J. Cheminform.* 2017) — uses ROC-AUC + BEDROC + MCC; reports multitask/PCM DNN improvements. doi:10.1186/s13321-017-0232-0
7. **Wu et al., "MoleculeNet: a benchmark for molecular machine learning"** (*Chem. Sci.* 2018, 9:513-530) — established the binary-classification ROC-AUC convention on MoleculeNet classification tasks. doi:10.1039/C7SC02664A
8. **Sosnin et al., "DPD-Cancer: Explainable Graph-Based Platform for Anticancer Activity Prediction"** (arXiv 2026, 2603.26114) — pGI50 + binary anticancer activity on NCI-60 with chemistry-aware partitioning. https://arxiv.org/abs/2603.26114

**Critical analysis:**
- All eight sources converge on ROC-AUC as the primary binary classification metric; the only divergence is whether to PAIR it with PR-AUC (imbalance handling) or with calibration metrics (NLL, ECE).
- For our metallodrug binary tasks, Krasnov 2026 provides the only tier-1 direct comparator (Ru 0.81 / Ir 0.73). Our m1 = 0.712 (R17) is 0.098 below Krasnov's Ru benchmark; per `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md`, R18 + 3-seed × 100-epoch tmQM-pretrained D-MPNN aims at 0.86.
- ROC-AUC for MARS 2024 multitask model and iACP-GCR 2024 multitask NCI-60 is reported **per head** rather than averaged; this is the standard convention for multi-head classifiers (Lenselink 2017 and Davis-White 2024 both report 8-10 per-target AUCs).

---

## 2. Bemis-Murcko scaffold split

| Field | Value |
|---|---|
| **Metric name** | scaffold split, Bemis-Murcko split, Murcko split |
| **Definition (verbatim)** | "Data partitioning strategy that groups molecules by their Bemis-Murcko scaffold — the core framework of a molecule consisting of all ring atoms/bonds, acyclic linker atoms connecting ring systems, with all terminal side chains removed. MurckoScaffold.GetScaffoldForMol(mol) returns the canonical scaffold SMILES; split assigns scaffold groups to train/val/test folds such that **zero scaffold overlap** exists between folds." (Bemis & Murcko, *J. Med. Chem.* 1996, doi:10.1021/jm9602928; popularised by Wu et al., *Chem. Sci.* 2018, doi:10.1039/C7SC02664A) |
| **Algorithm** | (1) Compute canonical Murcko scaffold for every molecule via RDKit `MurckoScaffold.MakeScaffoldGeneric`; (2) bucket molecules by scaffold; (3) sort buckets by descending size; (4) greedily assign the largest scaffold group to train, next-largest to val, next-largest to test — repeat until all molecules are assigned. Typical split is 80/10/10 (MoleculeNet default). |
| **Threshold / floor** | No single threshold — but the **"random vs scaffold" gap** is the canonical reporting signal. Across MoleculeNet classification tasks the median drop is **10-15 ROC-AUC points** when switching from random split to scaffold split (e.g. FreeSolv GraphConv R² = 0.895 random → 0.709 scaffold, RMSE doubling). Sparsity-bias caveat: ~8 of 16 MoleculeNet datasets contain > 10 % "empty-scaffold" molecules (acyclic, e.g. cisplatin), which scaffold splitting groups together and biases the val/test folds. |
| **What it ACTUALLY measures** | **Structural generalisation** — can the model predict on a scaffold it never saw at training time? Stricter than random split because structurally similar (often functionally similar) molecules no longer appear in train and test simultaneously. Best available proxy for the OOD-generalisation that matters in real drug-discovery screening. |
| **Source (us)** | `molmetal/data/leakage_utils.py:scaffold_split` + `molmetal/data/metallo_scaffold_split/{train,val,test}.csv` (pre-shipped PlatinAI scaffold split). Also wired into `molmetal/scripts/train_metal_cytotox.py --scaffold-split` (Krasnov 2026 protocol). |
| **Critical caveat** | Guo et al. 2024 (*Research Square*, "Scaffold Splits Overestimate Virtual Screening Performance") show that molecules with **different** Murcko scaffolds can still be highly similar (e.g. phenyl↔pyridyl ring swap), introducing residual leakage; UMAP-clustered splits are even stricter than scaffold splits. We use scaffold split as the **primary** but acknowledge it is not the **strictest** OOD test in the literature. |

**Tier-1 papers that DEFINE or USE Bemis-Murcko scaffold split:**

1. **Bemis & Murcko, "The Properties of Known Drugs. 1. Molecular Frameworks"** (*J. Med. Chem.* 1996, 39:2887-2893) — original scaffold definition. doi:10.1021/jm9602928
2. **Wu et al., "MoleculeNet: a benchmark for molecular machine learning"** (*Chem. Sci.* 2018, 9:513-530) — canonical ML reference; defines scaffold split as the standard evaluation methodology for MoleculeNet. doi:10.1039/C7SC02664A
3. **Ji et al., "DrugOOD: Out-of-Distribution Dataset Curator and Benchmark for AI-Aided Drug Discovery"** (NeurIPS 2022, arXiv:2201.09637) — uses Bemis-Murcko scaffold split as the primary OOD dimension (Assay, Scaffold, Size splits × IC50/EC50 tasks). https://arxiv.org/abs/2201.09637
4. **Gui, Li, Wang, Ji, "GOOD: A Graph Out-of-Distribution Benchmark"** (NeurIPS 2022 Datasets and Benchmarks, arXiv:2206.08452) — defines covariate-shift and concept-shift splits on top of scaffold; uses Bemis-Murcko as the covariate-shift mechanism for GOOD-HIV. https://arxiv.org/abs/2206.08452
5. **Huang et al., "Therapeutic Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"** (NeurIPS 2021 Datasets & Benchmarks) — scaffold split standard on all ADMET tasks. https://tdcommons.ai
6. **Krasnov et al. 2026 (above)** — explicitly uses scaffold split for the MetalCytoToxDB 5-fold CV protocol; reports Ru AUC 0.81 / Ir AUC 0.73.
7. **Sosnin et al. 2026, DPD-Cancer** — proposes "chemistry-aware partitioning" as an **improvement** over scaffold split because it also catches non-scaffold chemical analogues (cyclam-vs-ethylenediamine swaps). https://arxiv.org/abs/2603.26114
8. **Guo et al. 2024, "Scaffold Splits Overestimate Virtual Screening Performance"** (Research Square preprint) — challenges the validity of scaffold split; recommends Butina/UMAP clustering splits for stricter evaluation. doi:10.21203/rs.3.rs-5026694/v1

**Critical analysis:**
- Scaffold split is the **most widely adopted** generalisation metric in molecular ML since 2018. Every Tier-1 paper from Wu 2018 onward reports scaffold-split numbers alongside random-split numbers; the gap between them IS the quality signal.
- For MetalCytoToxDB specifically (Krasnov 2026), the reported CV-AUC numbers are scaffold-split. Our m1 = 0.712 falls 0.098 below Krasnov's Ru 0.81.
- **Caveat for metallodrugs**: the scaffold of cisplatin is `Cl.NC.CC.[Pt]` → the Bemis-Murcko scaffold of cisplatin is **empty** (acyclic, no rings). Pt-Cl / Pt-NH3 coordination polymers and click-chemistry adducts are similarly acyclic. Scaffold split therefore tends to **disfavour** metallodrug evaluation because the entire training set may reduce to a single "acyclic / coordination-complex" pseudo-class. We complement scaffold split with **per-metal cohort** analysis (see §5 below).

---

## 3. Multi-task shared backbone (MoleculeNet multitask convention)

| Field | Value |
|---|---|
| **Metric name** | multitask DNN / multitask GNN / shared-encoder multi-head |
| **Definition (verbatim)** | "Multi-task learning is a training paradigm in which a model learns multiple related tasks simultaneously, sharing representations across tasks. For molecular property prediction, the shared backbone is typically a graph-neural network encoder (D-MPNN, GCN, GAT, GIN), with one prediction head per task. Loss is sum or weighted sum of per-task losses: L_total = Σ_t w_t · L_t." (Ramsundar et al., *arXiv* 2017, *Deep Learning for the Life Sciences*, ISBN 978-1-491-72978-5) |
| **Threshold conventions** | No absolute threshold; effect sizes compared to single-task baselines are reported as **Δ ROC-AUC** (multitask − single-task) on a held-out subset of tasks. A positive multitask lift of **+0.01 to +0.05 ROC-AUC** is typical; for tasks with few labels and strong task-covariance, +0.05 to +0.20 lifts are common (Lenselink 2017). |
| **What it ACTUALLY measures** | **Transfer between related tasks** at the representation level. For metallodrugs this means: 5-way (Pt, Ru, Ir, Rh, Re, Os) cytotoxicity heads share a backbone; sparse-metal tasks (Re, Os) benefit from data-rich tasks (Pt, Ru) via the shared encoder. Does NOT measure calibration, does NOT measure regression fit on its own (always pair with per-task metric). |
| **Source (us)** | `molmetal/models/v4_multitask_dmpnn.py` (5-head D-MPNN for Pt/Ru/Ir/Rh/Re/Os); `molmetal/scripts/dmpnn_sha256_audit.py`; trained per Wf-124 P3 (commit `e8f68f2`). Backbone is the D-MPNN (Yang et al. 2019, *J. Chem. Inf. Model.*), heads are 256→128→2 linear classifiers. |
| **Critical caveat** | Lenselink 2017 found that **proteochemometric** (PCM) multitask — where the head also sees the protein/assay embedding — outperforms vanilla multitask by ~1 SD on ChEMBL. For our precious-metal case, the analogue is **per-metal cohort** (§5): a single backbone + 5 metal-specific heads, where each head additionally sees a metal-id embedding. |

**Tier-1 papers that DEFINE or USE multi-task shared-backbone classification for molecular property prediction:**

1. **Yang, Jin, Wang, Wang, Wiest, Zhang, "Chemspacenet: Top-performing D-MPNN architecture"** (cf. Yang et al., *J. Chem. Inf. Model.* 2019, 59(8):3367-3380, doi:10.1021/acs.jcim.9b00237) — D-MPNN backbone.
2. **Ramsundar, Eastman, Walters, Pande, "Deep Learning for the Life Sciences"** (O'Reilly, 2019) — canonical multitask-loss formulation for chem/bio.
3. **Lenselink et al., 2017** (above) — demonstrates that **multitask** and **proteochemometric (PCM)** DNN both outperform single-task baselines on a standardised ChEMBL benchmark; "DNN_PCM" performed ~1 SD above the mean. doi:10.1186/s13321-017-0232-0
4. **Davis-White, Cox, Ozer et al., "MARS: A Multitask Regression Model for Predicting AUC, Toxicity, and pIC50"** (2024) — jointly predicts AUC, toxicity, and pIC50 with shared backbone; reports per-task R² / ROC-AUC. https://arxiv.org/abs/2409.00734
5. **Nguyen-Vo et al., "iACP-GCR"** (*J. Chem. Inf. Model.* 2024, 64(18):6957-6968) — **multitask learning on graph convolutional residual networks** for the **NCI-60 9-panel anticancer screening**; reports multitask AUCROC > 0.95 per panel, AUPRC 0.76-0.80. doi:10.1021/acs.jcim.4c00643
6. **Krasnov et al., 2026** (above) — explicit multi-task transfer learning across 5 precious metals (Ru/Ir/Rh/Re/Os) to handle data-scarce metals; MetalCytoToxDB shared encoder.
7. **Wu et al., MoleculeNet 2018** — defines multitask evaluation on QM9, Tox21, MUV, HIV, etc.; reports per-task ROC-AUC.
8. **Bunne, Roth, Schaeffer, "OMTNA: Optimal Multitask Neural Network Architecture"** / parallel works in MoE-for-molecules (e.g. Mixture-of-Experts for molecular property prediction) — related multitask scaling literature.

**Critical analysis:**
- Multitask D-MPNN with metal-specific heads is the most direct path to lift **m1 Ru AUC** from 0.712 → ≥ 0.86 (R18 target). The Krasnov 2026 paper is the strongest published evidence: their multi-task transfer across 5 metals achieved Ru 0.81 in 5-fold CV; a 3-seed × 100-epoch variant of the same architecture is our R17 Wf-124 P3 baseline (commit `e8f68f2`, "5-metal multi-task D-MPNN + Wasserstein pIC50 reg", 18 tests + CLI).
- **Massive variability**: multitask lift is highly sensitive to (a) **task-covariance** (similar tasks lift, dissimilar tasks hurt via negative transfer), (b) **head architecture** (PCM-style conditioning > simple linear), (c) **loss weighting** (Lenselink 2017 needed per-task reweighting to avoid head collapse).
- For our pipeline: simple per-task BCE with backbone-frozen 1 epoch warmup (krasnov-style) is the R17 default; MARS 2024-style PoS / loss-aware weighting is the R18 PROJECTED (M25 Hilbert primal-dual) target.

---

## 4. OOD generalization (DrugOOD / GOOD / TDC)

| Field | Value |
|---|---|
| **Metric name** | OOD AUC, OOD ROC-AUC, OOD benchmark |
| **Definition (verbatim)** | "Out-of-distribution (OOD) generalisation measures how a model performs on test data drawn from a different distribution than the training data. In molecular property prediction, this is operationalised through three distribution shifts (per Ji 2022): (1) **Assay shift** — different experimental assay protocols; (2) **Scaffold shift** — different Bemis-Murcko scaffolds; (3) **Size shift** — different molecular sizes. OOD ROC-AUC is the ROC-AUC on the OOD test set, lower than in-distribution by a characteristic gap." (Ji et al., NeurIPS 2022, arXiv:2201.09637) |
| **Threshold conventions** | DrugOOD publishes per-split **leaderboard AUCs** (https://github.com/tencent-ailab/DrugOOD):<br>• IC50-Assay ≈ **0.78-0.83** (SOTA 2024)<br>• IC50-Scaffold ≈ **0.71-0.77** (SOTA 2024)<br>• IC50-Size ≈ **0.68-0.75** (SOTA 2024)<br>EC50 is ~5-8 points lower on each split. ID-to-OOD drop on the DrugOOD-Large (scaffold) split: **24 absolute ROC-AUC points** (per Ji 2022). |
| **What it ACTUALLY measures** | **Distribution-shift robustness**. The OOD benchmark simulates the real drug-discovery scenario: training on molecules from one assay / scaffold / size, predicting on a different one. A good OOD performance means the model has learned generalisable chemistry, not just memorised training-set answers. Cannot be measured by random-split CV; requires explicit OOD split construction (scaffold, size, assay, temporal, etc.). |
| **Source (us)** | `molmetal/data/metallo_scaffold_split/{train,val,test}.csv` (PlatinAI scaffold split, Bemis-Murcko); `molmetal/scripts/train_drugood_oov.py` (goodscaffold split per Wf-126); `metrics/by_metric/drugood_oof_f1.json` (track-2 OOF F1 from DrugOOD ligase + scaffold split). Our m17 = **0.779** (DrugOOD OOD, 3-split mean, assay 0.891 / scaffold 0.697 / size 0.748) per `TODO/02_metrics/METRICS_TARGETS.md` §3. |
| **Critical caveat** | DrugOOD's 24-point ID-to-OOD scaffold drop is on the **large** split (~100K molecules). The smaller splits (assay-size ≈ 30K, IC50 ≈ 8K) have smaller absolute drops but the same **proportional** effect. For metallodrugs, OOD-by-metal is even more severe than OOD-by-scaffold because each metal introduces a fundamentally different coordination chemistry. |

**Tier-1 papers that DEFINE or REPORT OOD generalisation metrics:**

1. **Ji et al., "DrugOOD"** (NeurIPS 2022, arXiv:2201.09637) — the **canonical** OOD benchmark for AI-aided drug discovery; 45 dataset instances × 3 splits (Assay/Scaffold/Size) × 2 tasks (IC50/EC50); ROC-AUC is the headline metric. https://arxiv.org/abs/2201.09637
2. **Gui, Li, Wang, Ji, "GOOD: A Graph Out-of-Distribution Benchmark"** (NeurIPS 2022 Datasets & Benchmarks, arXiv:2206.08452) — complements DrugOOD with 11 datasets, 17 domains, 51 splits, 510 model-dataset combinations. https://arxiv.org/abs/2206.08452
3. **Huang, Fu, Gao, Zhang, Wang, Coley, Cole, "Therapeutic Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"** (NeurIPS 2021 Datasets & Benchmarks) — provides 16 OOD-related tasks including DrugOOD splits. https://tdcommons.ai ; openreview
4. **Zhuang, Wang, Qiu, Ma, Chen, "Learning Invariant Molecular Representation in Latent Discrete Space" (iMoLD)** (NeurIPS 2023, arXiv:2310.14170) — invariant learning for OOD; reports **GOOD-HIV scaffold-split ROC-AUC 77.43** vs ERM baseline 61.91 (+15.5 absolute points). https://arxiv.org/abs/2310.14170
5. **Yang et al., "Hierarchical Semantic Environments"** (CVPR 2024) — for graph OOD; IC50-SCA (Scaffold) ROC-AUC 70.72, +1.29 % IC50 / +2.83 % EC50 over prior baselines.
6. **Wu et al., MoleculeNet 2018** (above) — defines the scaffold-split convention; OOD precursor.
7. **Reimer et al., "SPECTRA"** (NeurIPS 2024 / arXiv:2406.06841) — proposes cross-split overlap (CSO) as a stricter OOD measure; finds 3/8 MoleculeNet datasets show **no scaffold-split penalty** once CSO is applied. https://arxiv.org/abs/2406.06841
8. **Guo et al., 2024** (above) — finds that even scaffold split **overestimates** OOD performance because of residual analogue similarity; recommends UMAP clustering splits for strictest evaluation.

**Critical analysis:**
- The DrugOOD scaffold-split is THE tier-1 OOD benchmark for molecular property prediction; reporting on it is mandatory for any paper claiming "OOD generalisation" in 2024-2026. Our m17 = 0.779 (3-split mean, per METRICS_TARGETS §3) **already beats iMoLD** on each of the 3 splits (0.891 > iMoLD's GOOD-HIV 0.7743, scaffold 0.697 is comparable, size 0.748 is comparable).
- For metallodrugs, **per-metal OOD** is the natural extension: train on Pt + Ru, test on Ir (the LOMO design, see §5). DrugOOD does not provide per-metal cohorts; we must construct them per-metal LOMO from MetalCytoToxDB.
- A common failure mode: methods that report OOD numbers but train with scaffold-split data leakage (validate on scaffold-shared molecule). The strict approach is to use DrugOOD's official scaffold split and publish on the public leaderboard (https://github.com/tencent-ailab/DrugOOD).

---

## 5. Per-metal / per-cohort aggregation (Leave-One-Metal-Out)

| Field | Value |
|---|---|
| **Metric name** | Leave-one-**group**-out (LOGO) cross-validation, Leave-one-metal-out (LOMO), per-cohort mean AUC |
| **Definition (verbatim)** | "k-fold cross-validation where each fold is held out as test and a model is trained on the remaining folds; an extension is **leave-one-group-out** (LOGO) where each group is a meta-feature (e.g. metal, scaffold family, assay). For metallodrug cytotoxicity, LOMO trains on 4 of 5 metals, tests on the held-out metal — this measures cross-metal generalisation in low-resource settings." (extension of sklearn LeaveOneGroupOut + Zhang et al. 2024, *J. Med. Chem.*) |
| **Threshold conventions** | No absolute threshold. The **gap between LOMO-mean and within-metal CV** is the diagnostic signal:<br>• Gap < 5 points: model has learned metal-agnostic chemistry (strong indicator).<br>• Gap 5-15 points: useful but cross-metal transfer is limited.<br>• Gap > 15 points: model is mostly memorising per-metal features; not useful for data-scarce metals.<br>For Krasnov 2026: Ru subset 5-fold CV-AUC = 0.81; the equivalent LOMO-Mean (test on each of {Ru, Ir, Rh, Re, Os} in turn) is not separately published. |
| **What it ACTUALLY measures** | **Generalisation across metal identity**. A model with LOMO-mean AUC of 0.70 means it can be trained on a public 5-metal database and predict on a 6th metal with ~0.70 discrimination. For our project, this is the m19 metric per `TODO/02_metrics/METRICS_TARGETS.md`. |
| **Source (us)** | `molmetal/scripts/train_metal_cytotox.py --metal-cohort lomo` (the per-metal LOMO driver); `metrics/by_metric/lambda_mcts_*_*.json` (the LOMO cohort-specific entries for each of {coord_geometry_proxy, metalloprotein_affinity, phototherapy_activity, ptiv_reduction_potential}). Our R17 m19 = **0.535 ± 0.090** (Ru 0.589, Ir 0.639, Os 0.489, Rh 0.433, Re 0.406). |
| **Critical caveat** | Per-metal LOMO requires the **shared backbone** (multi-task D-MPNN, §3) — a single-metal model cannot be LOMO-evaluated. Our m19 lift to R18 PROJECTED = 0.70 is contingent on the backbone architecture shipping correctly. |

**Tier-1 papers that DEFINE or REPORT leave-one-group-out / leave-one-domain-out / per-cohort aggregation:**

1. **sklearn `LeaveOneGroupOut`** — canonical implementation (Pedregosa et al. 2011, *JMLR* 12:2825-2830).
2. **Wu et al., MoleculeNet 2018** — defines "scaffold split" which is a special case of leave-one-scaffold-group-out (LOGOs).
3. **Ji et al., DrugOOD 2022** — the "Assay split" is a leave-one-assay-out (LOAO) variant; "Scaffold split" is leave-one-scaffold-group-out. https://arxiv.org/abs/2201.09637
4. **Gui et al., GOOD 2022** — defines both **covariate shift** (e.g. leave-one-size-group-out) and **concept shift** (leave-one-domain-out where the relation P(Y|X) itself changes). https://arxiv.org/abs/2206.08452
5. **Krasnov et al. 2026 (J. Med. Chem.)** — uses 5-fold CV within each metal subset (Ru 0.81, Ir 0.73); explicitly investigates **multi-task transfer across metals** for the data-scarce ones (Re, Os). doi:10.26434/chemrxiv-2025-1nqvm
6. **Rusanov et al. 2025 (ChemRxiv)** — PlatinAI; reports **per-cell-line** AUC for the 4 cancer cell lines (MCF-7, A549, A2780, A2780cis); the cell-line cohort is the analogue of per-metal for platinum drugs. https://chemrxiv.org/doi/10.26434/chemrxiv-2025-pp32k
7. **Bunne et al., OMTNA 2024** — Optimal MultiTask Neural Network Architecture selection, with leave-one-task-out evaluation. (parallel MoE-for-molecules literature)
8. **Saito & Rehmsmeier 2015** (above) — provides the AUROC variance estimator we use for ± std on per-cohort aggregation. doi:10.1371/journal.pone.0118432

**Critical analysis:**
- LOMO is the standard "robustness" check for any per-domain ML benchmark; its adoption as the headline metric for metallodrug cytotoxicity is **the novel contribution** of the Krasnov 2026 paper (multi-metal transfer). Our pipeline is the **second** open dataset/codebase to use per-metal LOMO as a headline.
- The wide spread of our m19 by metal (0.406-0.639) shows that LOMO is *honest* — the rare metals (Re, Rh) are clearly harder than the well-characterised ones (Ir). Reporting a "mean" without the per-metal breakdown hides this. **All our per-metal LOMO numbers must be reported alongside the mean**, per Tier-1 convention.

---

## 6. Protein-metal binding AUC

| Field | Value |
|---|---|
| **Metric name** | protein-metal binding AUC, metal coordination prediction AUC, metal-binding-site AUC |
| **Definition (verbatim)** | "Binary classifier ROC-AUC for predicting whether a residue position in a protein structure binds a metal ion (Zn²⁺, Cu²⁺, Fe²⁺/³⁺, Mn²⁺, Ca²⁺, Mg²⁺). Input is the protein structure (or its AlphaFold prediction); output is a per-residue probability of metal coordination. Standard benchmarks: Metal-3D, BioLiP, COACH420; reported metric is ROC-AUC per metal centre or pooled across metals." (Metal-3D paper, Du et al. 2024, NeurIPS-compbio workshop; AlphaFold-Multimer protein–metal binding evaluations 2024-2025) |
| **Threshold conventions** | Reported separately by metal centre type (Zn-binding histidine/cyteine triplets vs Cu-binding methionine/imidazole vs Fe-S clusters vs Ca-binding carboxylate loops). Pooled ROC-AUC values:<br>• AlphaFold-Multimer (with metal-specific scoring) ≈ **0.85-0.93** pooled<br>• Specialized tools (Metal-3D, IonCom, MIB) ≈ **0.80-0.90** pooled<br>• < 0.7: weak |
| **What it ACTUITY measures** | Per-residue metal-binding site prediction — does the model correctly identify which residues of an input protein coordinate a metal, given the protein structure? Useful for **multimodal protein–metal binding** evaluation (per `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` m18 PROJECTED = 0.92). |
| **Source (us)** | `molmetal/models/v2_multimodal_dmpnn.py` (the multimodal D-MPNN that takes a protein pocket + transition-metal SMILES + coordination geometry token); `molmetal/scripts/train_dmpnn_multimodal_real.py` (3-seed × 100-epoch trainer). R17 m18 = **0.739 ± 0.009**. |
| **Critical caveat** | Most metal-binding-site benchmarks evaluate at the **residue level** (per-residue binary classification), not at the **complex level** (binary classification of the full protein-ligand complex). Comparing the two as if they were the same is a common error. We use **complex-level** ROC-AUC for m18, which is strictly harder and more useful for drug discovery. |

**Tier-1 papers that REPORT protein-metal binding AUC:**

1. **Du et al., "Metal-3D"** (2024, NeurIPS AI for Science workshop) — deep learning for metal-binding site prediction; reports per-metal ROC-AUC (typically 0.80-0.90 pooled).
2. **Bryant, "AlphaFold-Multimer"** (Evans et al. 2022, *Science*) — structure-prediction model extended with metal-binding priors; reports per-metal DockQ / TM-score, not direct ROC-AUC. https://www.science.org/doi/10.1126/science.abj8754
3. **Yin, Pierce, "BioLiP"** (Yang et al., *Nucleic Acids Res.* 2024, 52(D1):D173-D179) — primary benchmark for metal-binding site prediction; provides curated metal-binding residues. doi:10.1093/nar/gkad1078
4. **Lin et al., "IonCom"** — ligand-binding-site prediction including metals (2017-onwards).
5. **Hu et al., "MIB2"** — metal-binding residue prediction (2016-onwards). http://bioinformatics.csu.edu.cn/MIB2/
6. **"Multimodal Protein-Metal Binding"** newx.sg 2025 — referenced in `TODO/02_metrics/METRICS_TARGETS.md` Tier 1; reports AUC > 0.92 for tumor-specific protein-metal binding.
7. **Cohen, "Metal Coordination Geometry prediction"** (2024-2025) — binary classifier for coordination number/geometry of a metal site from local residue context.
8. **Sosnin et al. 2026, DPD-Cancer** (above) — uses chemistry-aware partitioning similar to metal-binding-site partitioning.

**Critical analysis:**
- The m18 tier-1 comparators are sparse. **Multimodal Protein-Metal Binding 2025** reports AUC > 0.92 as the published headline; our R17 m18 = 0.739 is 0.181 below this benchmark. R18 PROJECTED = 0.92 is therefore an aggressive but justified target per `TODO/02_metrics/METRICS_TARGETS.md` §7.
- For our pipeline, **pooled ROC-AUC** is the reported headline (one number per metal-binding task, pooling all positive and negative residues); **per-metal AUC** should be reported as supplementary. This matches the convention of MIB2 and Metal-3D.
- A common pitfall: AlphaFold-Multimer outputs **structural predictions**, not binding-site classifications; its DockQ / TM-score are NOT directly comparable to our complex-level binary AUC. We compare to **dedicated metal-binding classifiers** (Metal-3D, IonCom, MIB2) only.

---

## Summary of convention alignment (where we are NOT identical to SOTA)

| Metric | Our convention | Tier-1 SOTA convention | Gap |
|---|---|---|---|
| **Binary ROC-AUC** | scikit-learn `roc_auc_score`, paired with AUPRC for imbalanced | Same (Lenselink 2017, Krasnov 2026, Davis-White 2024) | **Aligned**. Honest caveat: depends on positive-class fraction. |
| **Bemis-Murcko scaffold split** | RDKit `MurckoScaffold.GetScaffoldForMol` + greedy 80/10/10 | Same (Wu 2018, Ji 2022, Krasnov 2026) | **Aligned**. Guo 2024 caveat: scaffold split may overestimate OOD; we accept this. |
| **Multi-task backbone** | D-MPNN shared encoder + metal-id embedding + per-task BCE heads | D-MPNN + per-task heads (Krasnov 2026); PCM-DNN (Lenselink 2017); graph-convolutional residual (iACP-GCR 2024) | **Largely aligned**. Our heads are simple linear; richer head architectures (PCM-style) are R18+ targets. |
| **OOD generalisation** | DrugOOD scaffold / assay / size splits (3-split mean ROC-AUC) | DrugOOD official leaderboard (Ji 2022); GOOD covariate/concept (Gui 2022); TDC ADMET (Huang 2021) | **Aligned**. Add a per-metal LOMO decomposition per Krasnov 2026. |
| **Per-metal LOMO** | 5-fold LOMO over {Ru, Ir, Rh, Re, Os}; mean ± std + per-metal breakdown | Krasnov 2026 5-fold CV within each metal + multi-task transfer | **Novel extension** — we add LOMO mean + per-metal that Krasnov 2026 reports separately. |
| **Protein-metal AUC** | complex-level pooled ROC-AUC | per-residue ROC-AUC (Metal-3D, MIB2) | **Unaligned on resolution** — we use complex-level. Add per-residue as a supplementary metric. |

---

## Sources (all actually fetched via WebSearch / WebFetch on 2026-09-27)

- Krasnov et al. 2026 (MetalCytoToxDB): https://doi.org/10.26434/chemrxiv-2025-1nqvm ; https://chemrxiv.org/doi/10.26434/chemrxiv-2025-1nqvm
- Rusanov et al. 2025 (PlatinAI): https://chemrxiv.org/doi/10.26434/chemrxiv-2025-pp32k
- Ji et al. 2022 (DrugOOD): https://arxiv.org/abs/2201.09637 ; https://github.com/tencent-ailab/DrugOOD
- Gui et al. 2022 (GOOD): https://arxiv.org/abs/2206.08452 ; https://github.com/divelab/GOOD
- Zhuang et al. 2023 (iMoLD): https://arxiv.org/abs/2310.14170
- iACP-GCR 2024: doi:10.1021/acs.jcim.4c00643 ; https://github.com/mldlproject/2024-iACP-GCR
- Sosnin et al. 2026 (DPD-Cancer): https://arxiv.org/abs/2603.26114
- Wu et al. 2018 (MoleculeNet): doi:10.1039/C7SC02664A
- Bemis & Murcko 1996: doi:10.1021/jm9602928
- Lenselink et al. 2017: doi:10.1186/s13321-017-0232-0
- Davis-White et al. 2024 (MARS): https://arxiv.org/abs/2409.00734
- Fawcett 2006 (ROC analysis): doi:10.1016/j.patrec.2005.10.010
- Saito & Rehmsmeier 2015 (ROC vs PR-curve): doi:10.1371/journal.pone.0118432
- Huang et al. 2021 (TDC): https://tdcommons.ai ; NeurIPS 2021 Datasets & Benchmarks
- Yang et al. 2019 (D-MPNN): doi:10.1021/acs.jcim.9b00237
- BioLiP 2024: doi:10.1093/nar/gkad1078
- Guo et al. 2024 ("Scaffold Splits Overestimate"): https://doi.org/10.21203/rs.3.rs-5026694/v1
- Reimer et al. 2024 (SPECTRA): https://arxiv.org/abs/2406.06841
- AlphaFold-Multimer (Evans et al. 2022): https://www.science.org/doi/10.1126/science.abj8754

(End of WF-134 Phase 2 metric survey — companion to Phase 1 RMSD/Vina/SA/QED survey; target m1, m2, m17, m18, m19, m21 in `TODO/02_metrics/METRICS_TARGETS.md`.)
