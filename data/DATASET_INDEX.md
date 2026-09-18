# Dataset Inventory — molmetal / Round-4 SOTA evaluation

**Date:** 2026-09-12
**Storage root:** `/mnt/storage/data/molmetal/`
**Project local:** `/home/hugo/codes/try_triton_on_rocm/data/`

## Status legend
- ✅ ON DISK — downloaded, ready
- ⚠️ PARTIAL — partial download, sub-pieces missing
- ❌ MISSING — not yet downloaded

---

## SOTA-anchor datasets (P0/P1)

### MetalCytoToxDB — Krasnov 2026
- **Path:** `/mnt/storage/data/molmetal/MetalCytoToxDB.csv` (4.8 MB)
- **Project copy:** `/home/hugo/codes/try_triton_on_rocm/data/MetalCytoToxDB.csv` (96 KB — header + 600 rows, **subset**)
- **Source:** https://zenodo.org/records/17106822 (full 26,800 rows, 6.4 MB)
- **Status:** ✅ ON DISK at /mnt/storage, ⚠️ project copy is partial
- **Used by:** R4-A (Ru/Ir/Os/Re/Rh per-metal binary classification)
- **SOTA target:** Ru ROC-AUC ≥ 0.81, Ir ≥ 0.73

### PlatinAI — Rusanov 2025
- **Path:** `/mnt/storage/data/molmetal/PlatinAI_*.xlsx`
  - `PlatinAI_MBFinder_dataset.xlsx` (5.9 MB) — **Unlabelled pretraining only** (226k SMILES)
  - `PlatinAI_predicted_A2780.xlsx` (8.7 MB) — predictions, not training labels
  - `PlatinAI_predicted_MCF7.xlsx` (7.2 MB) — predictions, not training labels
- **Missing:** the **4,078 Pt / 18,995 IC50 labelled training set** (scaffold-split, 4 cell lines MCF-7/A549/A2780/A2780cis)
- **Source:** https://chemrxiv.org/doi/suppl/10.26434/chemrxiv-2025-pp32k
  - Looking for `table_s4_*.xlsx`, `table_s5_*.xlsx`, `table_s7.xlsx` referenced in the paper
- **Status:** ⚠️ PARTIAL — pretraining + predictions OK, **labelled training set MISSING**
- **Used by:** R4-B (4-cell-line multi-task)
- **Need to download:** ~50 MB more from ChemRxiv supplementary

### tmQM — UiO Computational Catalysis
- **Path:** `/mnt/storage/data/molmetal/tmQM/` (419 MB)
- **Files:** `tmqm_parsed.csv`, `tmQM_y.csv`, `tmQM_X1.xyz.gz` ... `tmQM_X3.xyz.gz`, `tmQM_X1.BO.gz` ... `tmQM_X3.BO.gz`, `tmQM_X.q`
- **Source:** https://github.com/uiocompcat/tmQM
- **Status:** ✅ ON DISK, full set
- **Used by:** R4-G (D-MPNN encoder pretrain on metal coordination chemistry)
- **Note:** 108,703 transition-metal complexes with DFT properties (HOMO/LUMO/charge/spin)

### NCI-60 — DTP Growth Inhibition
- **Path:** `/mnt/storage/data/molmetal/NCI60_GI50/GI50.csv` (378 MB)
- **Source:** https://wiki.nci.nih.gov/display/NCIDTPdata/NCI-60+Growth+Inhibition+Data
- **Status:** ✅ ON DISK (official DTP file)
- **Plus:** `/mnt/storage/data/molmetal/NCI60_pharmacoset.rds` (312 MB, ORCESTRA processed)
- **Used by:** generalization validation (not in Round-4 P0-P3 but available)

### CrossDocked2020 (CascadeDiff processed)
- **Path:** `/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/` (8.3 GB, 2,464 pockets)
- **Files in each pocket:** `*_lig_tt_min_0.sdf`, `*_pocket10.pdb`
- **Source:** https://zenodo.org/records/20703074 (original)
- **Status:** ✅ ON DISK, full extracted dataset
- **Used by:** R4-C (100-pocket SBDD sweep)
- **Note:** 100 held-out test pockets need to be carved out per CrossDocked100 split convention (Luo et al. 2021)

### SPINDR — FLOWR training data
- **Path:** `/mnt/storage/data/molmetal/SPINDR/smol_data/` (11 GB)
- **Files:** `train.smol`, `val.smol`, `test.smol`
- **Source:** https://zenodo.org/records/15257565
- **Status:** ✅ ON DISK
- **Used by:** Optional — for Lambda pretraining if needed

### PDBbind (MBD subset only)
- **Path:** `/mnt/storage/data/molmetal/pdbbind/mbd_15_family_table.txt` (8 KB)
- **Source:** Metz et al. 2024, DOI 10.1021/acs.jcim.3c01568
- **Status:** ⚠️ PARTIAL — only MBD 15-family table, NOT full PDBbind time-split 2019 (363 complexes)
- **Used by:** not currently in Round-4
- **Need to download:** full PDBbind 2019 from http://www.pdbbind.org.cn/

---

## Missing datasets for Round-4

### ❌ DrugOOD — Tencent AI Lab (P1: R4-D)
- **What:** LBAP-core-IC50 with assay / scaffold / size splits
- **Size:** scaffold split train 21,519 mol / test OOD 19,048 mol; ~50 MB JSON
- **Source:** https://github.com/tencent-ailab/DrugOOD
- **Needs:** ChEMBL 29 SQLite database as upstream (or pre-curated JSON from DrugOOD repo releases)
- **Storage path:** `/mnt/storage/data/molmetal/DrugOOD/`

### ❌ MoleculeNet 4 tasks (P3: R4-H)
- **What:** BBBP / BACE / HIV / ClinTox
- **Size:** BBBP ~2k / BACE ~1.5k / HIV ~40k / ClinTox ~1.5k
- **Source:** https://moleculenet.org/datasets-1 or https://github.com/deepchem/deepchem
- **Storage path:** `/mnt/storage/data/molmetal/MoleculeNet/`

### ❌ Multimodal Protein-Metal Binding 2025 (P2: R4-F)
- **What:** tumor-specific protein-metal binding benchmark, AUC > 0.92 SOTA
- **Source:** newx.sg paper supplementary (per sota_papers_2025_2026.md §3.1)
- **Storage path:** `/mnt/storage/data/molmetal/ProteinMetalBinding/`

### ❌ Full PDBbind time-split 2019 (deferred — not in Round-4)
- **Source:** http://www.pdbbind.org.cn/

### ❌ PlatinAI labelled training set (P0: R4-B blocker)
- **What:** 4,078 Pt complexes with IC50 labels for 4 cell lines
- **Source:** ChemRxiv supplementary — `table_s4_*.xlsx`, `table_s5_*.xlsx` from paper
- **URL:** https://chemrxiv.org/doi/suppl/10.26434/chemrxiv-2025-pp32k
- **Storage path:** `/mnt/storage/data/molmetal/PlatinAI_labelled/`

---

## Datasets NOT needed (out of Round-4 scope)

- GDSC (Genomics of Drug Sensitivity in Cancer) — could add as P3 but not in plan
- CCLE — same
- ChEMBL 29 (only needed if we re-curate DrugOOD from scratch; otherwise use pre-curated JSON)
- BindingDB — same
- QM9 (already preprocessed at `/home/hugo/codes/try_triton_on_rocm/data/qm9_processed.pt`, 102 MB)

---

## Storage usage

| Item | Size |
|---|---:|
| /mnt/storage total free | 421 GB |
| /mnt/storage/data/molmetal/ used | ~22 GB |
| All current SOTA datasets + CrossDocked | ~22 GB |
| Headroom for DrugOOD + MoleculeNet + PlatinAI labels | < 1 GB |
| tmQM pretrain checkpoint (estimated) | ~500 MB |
| NCI-60 / GDSC / CCLE (if needed later) | ~5-10 GB |

Plenty of room.

---

## Action items (download priority)

1. **🔴 PlatinAI labelled training set** — fetch ChemRxiv supplementary (blocks R4-B)
2. **🟡 DrugOOD pre-curated JSON** — clone Tencent repo releases page
3. **🟡 MoleculeNet 4 tasks** — deepchem datasets or MoleculeNet mirror
4. **🟢 Multimodal Protein-Metal Binding 2025** — fetch when R4-F scheduled
5. **🟢 Full PDBbind** — only if needed later

---

*Last updated: 2026-09-12. Status legend: ✅ on disk / ⚠️ partial / ❌ missing.*
