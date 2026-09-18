# SOTA Papers — Detailed Metrics for Precious-Metal Anticancer Drug Design (2025-2026)

> **Purpose:** Track exact headline metrics from the 2025-2026 SOTA papers that are the most direct comparison targets for molmetal / MetalHybrid V4. Every cell in this table is either a number from the cited paper or an explicit "NOT REPORTED". Use this as the source of truth for what we need to beat or match.
>
> **Companion:** `TODO/01_research/sota_landscape.md` (broader 25-method landscape); `TODO/06_milestones/milestones.md` (Phase plan).
>
> **Date compiled:** 2026-09-12

---

## §1 — Direct Anchor: Metal-Anticancer ML (the two most relevant SOTA papers)

### 1.1 Krasnov 2026 — MetalCytoToxDB (Ru / Ir / Rh / Re / Os)

| Field | Value |
|---|---|
| **Paper** | Krasnov, Malikov, Kiseleva, Nykhrikova, Tatarin, Bezzubov. "Machine Learning for Anticancer Activity Prediction of Transition Metal Complexes." *J. Med. Chem.* (version of record 2026-04-13; ChemRxiv 2025-09-16 v2). DOI: 10.26434/chemrxiv-2025-1nqvm-v2 |
| **Dataset** | **MetalCytoToxDB** — 26,500 IC50 / 7,050 complexes / 754 cell lines / 5 metals (Ru/Ir/Rh/Re/Os) / 1,921 papers (2001–2025). Manual curation. |
| **App** | https://biometaldb.streamlit.app/ |
| **Task** | Binary classification (active / inactive), IC50 < threshold → active |
| **Split** | Time-based: pre-2024 train / 2025 test |
| **Ru subset ROC-AUC** | **0.81** (best binary classification model on cross-validation) |
| **Ir subset ROC-AUC** | **0.73** |
| **External validation (pre-2024 → 2025)** | 9/10 active predictions confirmed → **2× hit rate vs random sampling** |
| **Multi-metal transfer** | Single model across all 5 metals (Ru/Ir/Rh/Re/Os) demonstrated; per-metal evaluation available |
| **Architecture** | Gradient boosting on (metal + ligands composition) features. NOT a GNN. |
| **Code** | App at biometaldb.streamlit.app; GitHub repo not published in preprint |
| **Limitations explicitly stated by authors** | (a) no complex geometry features, (b) no counterion influence, (c) no healthy-cell selectivity |

**Our target:** V4 on Ru subset ≥ 0.81 (match Morgan+GBDT) → 0.86+ (beat it with GNN).

### 1.2 Rusanov 2025 — PlatinAI (Pt-specific de novo + activity prediction)

| Field | Value |
|---|---|
| **Paper** | Rusanov, Brezgunov, Matnurov, Pashaliev, Vetrova, Babak. "Overcoming cisplatin resistance via deep learning-assisted de novo design of platinum complexes." ChemRxiv 2025-06-20 v2. DOI: 10.26434/chemrxiv-2025-pp32k |
| **Dataset** | **4,078 platinum complexes / 18,995 IC50 values**, 4 cell lines: MCF-7, A549, A2780, A2780cis. Scaffold-split protocol. |
| **External validation** | Scored 214,373 published Pt complexes across the 4 cell lines (scaffold-split) |
| **Synthesis validation** | 20/20 predictions synthesized → strong agreement predicted vs observed |
| **De novo design** | Designed **PlatinAI** (a bis-carbene Pt(II) complex), synthesized |
| **PlatinAI experimental IC50** | Tested in MCF-7, A549, A2780, A2780cis — **superior to cisplatin in cisplatin-resistant cell lines** |
| **Mechanism** | NOT relying on DNA adducts (distinct from cisplatin) |
| **Architecture** | GNN + fragment assembly approach, scaffold-based splitting |
| **Code** | GitHub repo in supplementary (collection of algorithms used) |
| **Target cell line performance** | AUC reported per cell line (paper Table S4–S5) — values for MCF-7, A549, A2780, A2780cis in supplementary xlsx tables |

**Our target:** V4 on PlatinAI's 4,078 Pt dataset → match or beat per-cell-line AUC (especially **A2780cis** to demonstrate resistance overcoming).

---

## §2 — SBDD 2025-2026 SOTA (most relevant for Lambda line)

### 2.1 TransDiffSBDD (Tsinghua + Microsoft Research, March 2025)

| Field | Value |
|---|---|
| **Paper** | Tsinghua + Microsoft Research AI for Science + McGill + Mila. March 2025. |
| **Training data** | Uni-Mol 209M molecules + CrossDocked2020 100,000 complexes |
| **Eval protocol** | CrossDocked2020 100 held-out pockets, 100 mols each, AutoDock Vina |
| **Vina Dock (median)** | **−9.37 kcal/mol** |
| **SA** | 0.75 |
| **Diversity** | 0.81 |
| **QED** | 0.48 |
| **Vina Score (calibrated)** | −6.02 kcal/mol |
| **Success rate** (Vina Dock < −8.18, QED > 0.25, SA > 0.59) | **83.9%** |

### 2.2 MolCRAFT

| Field | Value |
|---|---|
| **Vina Dock** | −9.25 kcal/mol |
| **Success rate** | 36.1% |

### 2.3 AlphaDrug

| Field | Value |
|---|---|
| **# molecules generated** | 9,617 |
| **Median Vina** | **−9.77 kcal/mol** (lowest median) |
| **QED** | 0.42 |
| **SA** | 0.80 |

### 2.4 MolChord (Oct 2025) — NatureLM-based + DPO

| Field | Value |
|---|---|
| **Success rate** | **33.2%** |
| **Vina Dock (median)** | −7.62 kcal/mol |
| **High-affinity rate** | 55.1% |
| **QED** | 0.56 |
| **SA** | 0.77 |
| **Diversity** | 0.76 |

### 2.5 ReInvent + Vina (baseline in TransDiffSBDD paper)

| Field | Value |
|---|---|
| **Vina Dock** | −9.18 kcal/mol |
| **Success rate** | 76.7% |

### 2.6 SBDD SOTA baseline numbers (CrossDocked2020)

| Method | Year | Vina Dock (median) | Success Rate | PB-Valid | Source |
|---|---|---:|---:|---:|---|
| Pocket2Mol | 2022 | −7.15 | 24.4% | — | TransDiffSBDD Table |
| TargetDiff | 2023 | −7.80 | 10.5% | — | TransDiffSBDD Table |
| DecompDiff | 2024 | −8.39 | 24.5% | — | TransDiffSBDD Table |
| DiffSBDD | 2024 | −7.62 | — | 88% | DiffSBDD / sota2 |
| MolCRAFT | 2024 | −9.25 | 36.1% | — | TransDiffSBDD Table |
| ReInvent+Vina | 2024 | −9.18 | 76.7% | — | TransDiffSBDD Table |
| AlphaDrug | 2025 | −9.77 | — | — | DrugGen 2024 |
| MolChord | 2025 | −7.62 | 33.2% | — | MolChord paper |
| **TransDiffSBDD** | 2025 | **−9.37** | **83.9%** | — | TransDiffSBDD paper |
| **OMTRA** (2025-12 SOTA2 leaderboard) | 2025 | −6.8 | 27% | **97.7** | SOTA2 leaderboard |
| DrugFlow | 2025 | −5.9 | 17% | 73.4 | SOTA2 leaderboard |

### 2.7 SOTA2 leaderboard snapshot (2025-12-04, CrossDocked)

| Method | PB-Valid | E_strain | Vina (kcal/mol) | Interaction Parity |
|---|---:|---:|---:|---:|
| OMTRA | **97.7** | 21.3 | −7.1 | 100 |
| Pocket2Mol | 86.6 | 24 | −4.5 | 22 |
| DrugFlow | 73.4 | 72.5 | −5.9 | 21 |
| TargetDiff | 47.7 | 446 | −6.8 | 20 |
| DiffSBDD | 38.2 | 546 | −2.8 | 18 |

### 2.8 ShallowBench (harder, flat-pocket subset)

| Method | Vina (kcal/mol) | QED | SA | Sc |
|---|---:|---:|---:|---:|
| DiffSBDD | −4.75 | 0.25 | 0.00 | −0.003 |
| SimpleSBDD | −6.47 | 0.62 | 0.04 | 0.037 |
| TargetDiff | −5.26 | 0.49 | 0.64 | 0.609 |

---

## §3 — Protein-Metal Binding (the gap for molmetal)

### 3.1 Multimodal Protein-Metal Binding (2025, *Interpretable Multimodal Learning for Tumor Protein-Metal Binding*)

| Field | Value |
|---|---|
| **Task** | Tumor-specific protein-metal binding prediction |
| **AUC** | **> 0.92** |
| **Methods** | Multimodal fusion of sequence + 3D structure + PPI priors + metal-induced conformational change modeling + LRP for interpretability |
| **Application** | Guided rational optimization of 2 platinum + 2 ruthenium anticancer complexes |
| **Key contribution** | First tumor-specific, interpretable multimodal ML framework for protein-metal binding |

**Our gap:** Currently we have NO protein-metal binding prediction head. Should add one in V5.

---

## §4 — DrugOOD OOD Generalization (relevant for metal multi-cell-line transfer)

### 4.1 iMoLD (2024 — current SOTA on DrugOOD)

| Split | IC50-assay | IC50-scaffold | IC50-size |
|---|---:|---:|---:|
| iMoLD | **72.11 ± 0.51** | **68.84 ± 0.58** | **67.92 ± 0.43** |
| ERM baseline | 71.63 ± 0.76 | 68.79 ± 0.47 | 67.50 ± 0.38 |
| CIGA | 71.86 ± 1.37 | 69.14 ± 0.70 | 66.92 ± 0.54 |
| MoleOOD | 71.62 ± 0.52 | 68.58 ± 1.14 | 65.62 ± 0.77 |

Source: arXiv:2310.14170 (iMoLD paper).

### 4.2 DrugOOD Dataset Details (lbap-core-ic50)

| Split | Train | Val (ID) | Test (ID) | Val (OOD) | Test (OOD) |
|---|---:|---:|---:|---:|---:|
| **assay** | 34,179 (311 largest assays) | 11,314 | 11,683 | 19,028 (314 mid) | 19,302 (314 smallest) |
| **scaffold** | 21,519 (6,881 largest scaffolds) | 4,920 (1,912) | 30,708 (24,112) | 11,683 (6,345) | 19,048 (4,350) |

Source: arXiv:2201.09637 (DrugOOD original paper).

---

## §5 — Property Prediction (MoleculeNet anchors)

### 5.1 ROC-AUC on MoleculeNet (scaffold split)

| Method | BBBP | BACE | ClinTox | Tox21 | HIV | Avg |
|---|---:|---:|---:|---:|---:|---:|
| Chemprop D-MPNN | 0.94 | 0.87 | 0.89 | 0.84 | 0.82 | ~0.87 |
| AttentiveFP | 0.93 | 0.78 | 0.94 | 0.81 | 0.76 | ~0.84 |
| GEM | 0.72 | 0.86 | 0.90 | 0.83 | 0.77 | ~0.82 |
| GraphMVP | 0.72 | 0.72 | 0.79 | 0.80 | 0.77 | ~0.76 |
| MolCLR | 0.74 | 0.82 | 0.91 | 0.82 | 0.76 | ~0.81 |
| GROVER | 0.94 | 0.83 | 0.93 | 0.83 | 0.69 | ~0.84 |
| Uni-Mol | 0.73 | 0.86 | 0.92 | 0.82 | 0.77 | ~0.82 |
| Morgan+FINGPT (Krasnov baseline) | ~0.75 | ~0.70 | — | — | — | ~0.72 |

### 5.2 Docking RMSD on PDBBind (top-1 % < 2Å)

| Method | Top-1 | Top-5 | Speed |
|---|---:|---:|---|
| DiffDock | 38.2% | 44.7% | ~10s/A100 |
| **DeltaDock** | **47.4%** | — | faster than DiffDock |
| EquiBind | 38.0% | — | 10× faster than classical |
| TankBind | 20.4% | — | — |
| GNINA | 22.9% | — | 146s/complex |
| GLIDE | 21.8% | — | slow |
| Surflex-Dock | 68% | 81% | slow |

### 5.3 Pocket-conditioned ligand generation (SPINDR test set)

| Method | RDKit-valid | PB-valid | Vina (kcal/mol) | Strain energy |
|---|---:|---:|---:|---:|
| **FLOWR (100 steps)** | 0.94 | **0.88** | **−6.93** | 90.05 |
| PILOT | 0.79 | 0.71 | −6.30 | 120.0 |

### 5.4 Affinity prediction (Pearson r vs experimental)

| Method | Affinity | Pearson r | Speed |
|---|---|---:|---|
| **Boltz-2** | pIC50 | **0.86** | ~20s/complex |
| FEP+ (physics) | pIC50 | ~0.90 | hours-days |
| FLOWR.ROOT | pIC50 | r=0.86 | fast |

---

## §6 — What we have vs what we don't

### 6.1 Currently evaluated (molmetal / MetalHybrid V4)

| Task | Dataset | Metric | Value | Where |
|---|---|---|---:|---|
| Binary classification | Ru temporal (baseline) | test AUC | 0.5268 (V4 round-2 mean across 3 seeds) | reports/metal_hybrid_v4_ru_temporal_report.md |
| Multi-task | Ru temporal | test AUC | 0.5135 (D-MPNN baseline) | reports/honest_baseline_summary.md |
| OOD (preliminary) | DrugOOD scaffold | test AUC | 0.6148 (V3 reference) | reports/r3_drugood_benchmark.md |
| OOD | DrugOOD OOD assay | delta vs ID | −0.012 | reports/r3_drugood_benchmark.md |
| OOD | DrugOOD OOD resistance | delta vs ID | −0.068 | reports/r3_drugood_benchmark.md |
| CrossDocked (Lambda) | 1h36 single pocket | Vina mean | −5.923 | reports/lambda_vs_sbdd_protocol_aligned.md |
| CrossDocked (Lambda) | 1h36 single pocket | SA / QED | 1.870 / 0.548 | reports/lambda_vs_sbdd_protocol_aligned.md |

### 6.2 Gaps we have NOT evaluated on

| Task | Dataset | Current SOTA | What we need |
|---|---|---|---|
| Binary classification | MetalCytoToxDB (26,500 IC50, 5 metals, 754 cell lines) | Ru 0.81, Ir 0.73 (Krasnov 2026) | Train V4 on MetalCytoToxDB with scaffold split |
| Binary classification | PlatinAI Pt dataset (4,078 Pt / 18,995 IC50, 4 cell lines) | AUC per cell line (Rusanov 2025) | Train V4 on PlatinAI 4 cell lines |
| Selectivity / TI | MCF-7 vs A2780cis | PlatinAI experimental — PlatinAI > cisplatin in A2780cis | V4 4-cell-line multi-task head |
| Protein-metal binding | Tumor protein-metal (2025 paper) | AUC > 0.92 | New head — currently we have NO protein-metal binding predictor |
| CrossDocked | 100-pocket test | Vina −9.37, success 83.9% (TransDiffSBDD) | Lambda L-A3 100-pocket sweep |
| DrugOOD | IC50-assay / scaffold / size | iMoLD 72.11 / 68.84 / 67.92 | Train V4 on DrugOOD 3 splits |
| MoleculeNet (generalization) | BBBP / BACE / HIV / ClinTox | Chemprop 0.94 / 0.87 / 0.82 / 0.89 | V4 transfer learning evaluation |

### 6.3 Datasets we should track but currently don't

| Dataset | Why | URL |
|---|---|---|
| MetalCytoToxDB | Direct SOTA anchor for Ru/Ir/Os/Re/Rh | https://biometaldb.streamlit.app |
| PlatinAI Pt dataset | Direct SOTA anchor for Pt de novo | ChemRxiv supplementary xlsx |
| DrugOOD lbap-core-ic50 | OOD generalization benchmark | https://github.com/tencent-ailab/DrugOOD |
| NCI-60 | Gold standard multi-cell-line cytotoxicity | https://dtp.cancer.gov/discovery_development/nci-60/ |
| GDSC | Drug sensitivity in cancer (larger than NCI-60) | https://www.cancerrxgene.org/ |
| tmQM | Transition-metal quantum mechanics (108k complexes) | https://github.com/chemspacelab/tmqm |
| ChEMBL 29+ | General bioactivity database | https://www.ebi.ac.uk/chembl/ |
| BindingDB | Binding affinity database | https://www.bindingdb.org/ |
| CCLE | Cancer cell line encyclopedia | https://depmap.org/portal/ |
| MultiModal Protein-Metal 2025 | First tumor protein-metal binding dataset | per paper supplementary |

---

## §7 — Honest assessment of "SOTA" gaps for molmetal

| Gap | Severity | Path to close | Estimated effort |
|---|---|---|---|
| **No MetalCytoToxDB evaluation** | **HIGH** — single most direct SOTA anchor | Download from biometaldb.streamlit.app; reuse V4 with scaffold split | 1 day |
| **No PlatinAI dataset evaluation** | **HIGH** — single most direct Pt anchor | Use ChemRxiv supplementary xlsx; per-cell-line AUC | 1 day |
| **Lambda single-pocket only** | **HIGH** — paper-grade requires 100 | BatchPocketRunner + 100-pocket sweep | 24 h (compute-bound) |
| **No DrugOOD evaluation** | MEDIUM | Reuse V4 + 3 splits × 3 seeds | 6 h |
| **No protein-metal binding head** | MEDIUM | Add multi-modal head; pretrain on TM-align datasets | 2 days |
| **No multi-metal transfer test** | MEDIUM | LOMO (Leave-One-Metal-Out) eval | 1 day |
| **No external prospective validation** | LOW (out of ML scope) | Requires wet lab | months |
| **No NCI-60 multi-cell-line** | MEDIUM | Download + 60-task multi-head | 2 days |
| **No tmQM pretraining** | MEDIUM | Pretrain encoder on tmQM 108k, fine-tune downstream | 1 day |
| **No MoleculeNet cross-benchmark** | LOW | Add BBBP / BACE / HIV / ClinTox evaluations | 4 h |

---

## §7.5 — Hardware / Compute used by SOTA labs (verified)

| Lab / Paper | GPU cluster | Where | Source / URL | Notes |
|---|---|---|---|---|
| **CityU Hong Kong — Institute of AI for Science** (hosts PlatinAI / Babak Lab) | **64× NVIDIA A100 Tensor Core GPUs** + 256 nodes with dual AMD EPYC + 512GB RAM each + 4PB storage + 200 Gb/s InfiniBand | Hong Kong | https://www.hkguides.com/en/hot-topic/516301.html | PlatinAI (Rusanov 2025) is published from this lab; almost certainly used this cluster |
| **Tsinghua University** (TransDiffSBDD co-author) | Tsinghua has multi-thousand GPU cluster for AI for Science | Beijing | (institutional) | TransDiffSBDD trained on Uni-Mol 209M + CrossDocked 100k; would need ≥8× A100 / H100 for days |
| **Kurnakov Institute (Russia)** (Krasnov 2026 MetalCytoToxDB) | **Not reported** in ChemRxiv paper or GxP News | Moscow | https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-1nqvm-v2 | Krasnov used Morgan FP + GBDT (sklearn) — does not need GPU |
| **DeepMind / Isomorphic Labs** (Boltz-1 / Boltz-2) | TPU v4/v5 + internal clusters (not disclosed) | London | (not disclosed) | Boltz-2 affinity Pearson r=0.86 with ~20s/complex inference |
| **Baker Lab** (RFAA / RoseTTAFold-AA) | NVIDIA cluster + AlexTravinsky nodes (specifics not disclosed) | Seattle | (not disclosed) | RFAA matches ~90% AF3 ligand accuracy |
| **Chai Discovery** (Chai-1 / Chai-2) | NVIDIA H100 cluster | San Francisco | (not disclosed) | Chai-2 16% antibody hit rate |
| **Cremer et al.** (FLOWR / FlowDock) | AstraZeneca internal (computational chemistry group) | Cambridge UK | (AstraZeneca confidential) | FLOWR 70× faster than PILOT |
| **NVIDIA** (DiffDock / DiffDock-L) | NVIDIA A100 80GB for inference (10s/A100) | US | https://github.com/gcorso/DiffDock | DiffDock-L uses PLINDER 486K complexes |
| **Tencent AI Lab** (DrugOOD) | Multi-GPU DGX station | Shenzhen | https://github.com/tencent-ailab/DrugOOD | DrugOOD dataset + benchmarks |

**Key observations for our position (ROCm 7.2 + Triton 3.8 on RX 7800 XT gfx1101):**

1. **We CANNOT directly run most SOTA training** — Uni-Mol pretraining (TransDiffSBDD's backbone, 209M molecules + 100k complexes) requires 8+ A100 days; we have a single RX 7800 XT (gfx1101) with 16GB VRAM.
2. **PlatinAI's 4,078 Pt / 18,995 IC50 dataset is small enough for our hardware** — fine-tuning with D-MPNN on 4k complexes can fit comfortably on 16GB VRAM.
3. **MetalCytoToxDB (Krasnov) is even smaller per-metal** (Ru/Ir ~2-3k complexes each) — definitely fits.
4. **Lambda 100-pocket CrossDocked** is CPU-bound on Vina docking (not GPU-bound) — we already have this pipeline.
5. **Direct comparison with TransDiffSBDD / MolCRAFT is INFEASIBLE on our hardware** for training, but protocol-aligned INFERENCE comparison (run published checkpoints on the same 100 pockets) is feasible if the checkpoints can be loaded on ROCm.

**Strategy implication:** Our SOTA path is **inference-time protocol alignment + small-data fine-tuning** (PlatinAI / MetalCytoToxDB / DrugOOD), NOT direct training-time SOTA reproduction. Document this clearly in the paper to manage reviewer expectations.

### Compute-cost reference points

| Model / scale | GPU type | GPU-hours | Cloud cost est. |
|---|---|---:|---:|
| 7B model full pretrain, 1T tokens | 8× A100 80GB | ~24h | ~$267 |
| 70B model full pretrain, 10B tokens | 8× H100 80GB | ~7-12h | ~$259 |
| D-MPNN fine-tune on 4k complexes | 1× A100 80GB | ~30 min | ~$2 |
| CrossDocked 100k pocket pre-training | 8× A100 80GB | ~3-7 days | ~$3k-$7k |
| tmQM 108k complex pre-training | 4× A100 80GB | ~1-2 days | ~$1k-$2k |

Our RX 7800 XT (gfx1101) is roughly **equivalent to a 3090 / 4090 tier** in raw TFLOPS, but lacks FP8/TF32 tensor core support that A100/H100 have for transformer-style workloads.

---



| # | Paper | Year | Venue | arXiv | DOI | bibtex key |
|--:|---|---|---|---|---|---|
| 1 | Krasnov et al., MetalCytoToxDB | 2026 | J. Med. Chem. | (ChemRxiv) | 10.26434/chemrxiv-2025-1nqvm-v2 | krasnov2026metalcytotoxdb |
| 2 | Rusanov et al., PlatinAI | 2025 | ChemRxiv | (ChemRxiv) | 10.26434/chemrxiv-2025-pp32k | rusanov2025platinai |
| 3 | TransDiffSBDD (Tsinghua + MSR) | 2025 | preprint | (per bio.rodeo) | — | transdiffsbdd2025 |
| 4 | MolChord (Zhongguancun + USTC) | 2025 | preprint | — | — | molchord2025 |
| 5 | iMoLD | 2024 | arXiv | 2310.14170 | — | imold2024 |
| 6 | DrugOOD original | 2022 | arXiv | 2201.09637 | — | drugood2022 |
| 7 | Multimodal Protein-Metal Binding | 2025 | newx.sg | — | — | multimodal_protein_metal_2025 |
| 8 | AlphaDrug (DrugGen) | 2024 | Database (Oxford) | — | 10.1093/database/baad090 | alphadrug2024 |
| 9 | MolCRAFT | 2024 | — | — | — | molcraft2024 |
| 10 | Boltz-2 | 2025 | — | — | — | boltz2_2025 |
| 11 | FLOWR | 2025 | Nat. Comput. Sci. | 2504.10564 | — | flowr2025 |
| 12 | FlowDock | 2025 | Oxford Bioinf ISMB | 2412.10966 | — | flowdock2025 |
| 13 | Pocket2Mol | 2022 | ICML | 2205.07249 | — | peng2022pocket2mol |
| 14 | TargetDiff | 2023 | ICLR | 2303.03543 | — | guan2023targetdiff |
| 15 | DiffSBDD | 2024 | Nat. Comput. Sci. | 2210.13695 | 10.1038/s43588-024-00737-x | schneuing2024diffsbdd |
| 16 | DecompDiff | 2024 | ICLR | 2303.10120 | — | guan2024decompdiff |
| 17 | DeltaDock | 2024 | — | — | — | deltadock2024 |

---

*Last updated: 2026-09-12. Cite external papers via arXiv/DOI; do NOT run other people's pretrained weights.*
