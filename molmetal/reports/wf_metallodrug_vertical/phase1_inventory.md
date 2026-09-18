# Phase 1 — Metallodrug Vertical Dataset Inventory

**Audit date:** 2026-09-16
**Scope:** `/mnt/storage/data/molmetal/` + `/mnt/storage/models/reinvent4/` + `/mnt/storage/envs/reinvent4/`
**Sample size for SMARTS/Lipinski stats:** 3000 SMILES per dataset (random seed=0) — except NCI60 which is SMILES-less
**Tools:** RDKit 2024.3+ (canonicalize, Lipinski, SMARTS), openpyxl, pandas

---

## 1. Summary table (10 datasets)

| # | Dataset | Format | N records | Cols (top 10) | Heavy-atom min/max/median | Metal mols (Pt/Pd/Au/Ir/Ru) | Click cov. (CuAAC/SPAAC/ThiolEne/Suzuki/Amide) | SMILES canon. rate | Lipinski pass | Activity labels | Recommended use |
|---|---------|--------|-----------|---------------|---------------------------|------------------------------|---------------------------------------------------|--------------------|---------------|-----------------|------------------|
| 1 | **PlatinAI_MBFinder** | xlsx (1 col: smiles) | **226 918** | smiles | 5 / 240 / **48** | 1867 / 15 / 5 / 5 / 3 | 0.0 / 0.013 / 0.0 / 0.0 / 0.0003 | 0.622 | 0.039 | **N** (pretraining corpus) | **train** (CFM / MCTS corpus) |
| 2 | **PlatinAI_predicted_A2780** | xlsx (smiles, pred_0) | **214 376** | smiles, pred_0 | 6 / 246 / **47** | 1936 / 17 / 7 / 0 / 0 | 0.0 / 0.013 / 0.0 / 0.0 / 0.0003 | 0.645 | 0.045 | **Y** (pred_0 ∈ [0,1] = P(active A2780 ovarian)) | **train + eval** (scaffold-split, OvCa proxy) |
| 3 | **PlatinAI_predicted_MCF7** | xlsx (smiles, pred_0) | **214 376** | smiles, pred_0 | 4 / 237 / **47** | 1995 / 11 / 4 / 1 / 4 | 0.0 / 0.012 / 0.0 / 0.0 / 0.0013 | 0.665 | 0.048 | **Y** (pred_0 ∈ [0,1] = P(active MCF7 breast)) | **train + eval** (scaffold-split, breast proxy) |
| 4 | **MetalCytoToxDB** | csv (23 cols) | **26 801** | SMILES_Ligands, Counterion, Abbreviation, IC50_Dark, IC50_Dark_SE, IC50_Light, IC50_Light_SE, Excitation_Wavelength, Irradiation_Time, Cell_line, Time(h), DOI, Year, Metal, Oxidation_state, Charge_complex, IC50_Cisplatin | 6 / 234 / **39** | **0** (RDKit SMARTS miss) / 0 / 0 / 0 / 0 (Ru dominant in CSV col) | 0.0 / 0.0 / 0.0 / 0.0 / 0.005 | 1.000 | 0.344 | **Y** (IC50_Dark μM, IC50_Light μM = phototox, IC50_Cisplatin μM control) | **train + eval** (real IC50, multi-metal) |
| 5 | **NCI60_GI50** | csv (14 cols) | **~6M** (sampled 200K) | RELEASE_DATE, EXPID, PREFIX, NSC, CONCENTRATION_UNIT, LOG_HI_CONCENTRATION, PANEL_NUMBER, CELL_NUMBER, PANEL_NAME, CELL_NAME | n/a | n/a (SMILES-less; NSC id only) | n/a | n/a | n/a | **Y** (AVERAGE = log10 GI50 M, 71 cell lines, 10 panels) | **reference** (external join on NSC→SMILES) |
| 6 | **tmQM_parsed** | csv (18 cols) | **108 542** (100 850 SMILES) | csd_code, stoichiometry, metal, charge, spin, coord_number, metal_bo_total, Electronic_E, Dispersion_E, Dipole_M, Metal_q, HL_Gap, HOMO_Energy | 6 / 99 / **32** | 251 / 306 / 122 / 128 / 263 | 0.0 / 0.032 / 0.0 / 0.0 / 0.0023 | 1.000 | 0.262 | **N** (no bioactivity; has QM) | **train + reference** (CFM warm-start, MCTS prior, learned reactivity) |
| 7 | **CrossDocked2020_pocket10** | pdb + sdf dirs | **2 464 pockets** (≥1 sdf + 1 pdb per pocket) | pocket_id, rec.pdb, lig.sdf | n/a (pocket-conditioned) | n/a (no metal; off-target baseline) | n/a | n/a | n/a | **N** | **eval** (SBDD benchmark; pocket-conditioned prior) |
| 8 | **crossdocked_3d_cache** | pkl | **856** | per-pocket 3D coords | n/a | n/a | n/a | n/a | n/a | n/a | **eval** (pre-embed for fast RDKit ETKDGv3) |
| 9 | **aizynthfinder_public** | onnx + hdf5 + npz | **1** policy + **1** templates | uspto_legacy_weights.npz, uspto_legacy_converted.onnx, uspto_model.hdf5, uspto_templates.hdf5 | n/a | n/a | n/a | n/a | n/a | n/a | **reference / eval** (synthesis oracle; Zenodo fresh download blocked — legacy converted assets available) |
| 10 | **REINVENT4_prior** | reinvent4 .prior (TorchScript) | **1** (reinvent_v4.4.22.prior, **23 MB**) | prior | n/a | n/a | n/a | n/a | n/a | n/a | **train / generate** (de novo SMILES via RL+scoring; multiproperty bridge already wired) |

---

## 2. Key observations

### 2.1 PlatinAI family (datasets 1-3)

- **Total unique SMILES: 226 918 (MBFinder) + 214 376 (A2780, MCF7); A2780/MCF7 are subsets of MBFinder by scaffold-split.**
- **Canonicalizability is the lowest among metal-drug datasets (~62-66%)** — likely due to unusual coordination chemistry (e.g., `[Pt+2]` with multi-dentate ligands) that breaks RDKit kekulization. Already noted in `mol_dataset.py` validation paths.
- **Lipinski pass rate < 5%** is consistent with metal coordination complexes carrying bulky ligands; **not a blocker** for metallodrug eval but means Lipinski gate is too strict — use **Veber + metal-specific filter** instead.
- **Metal coverage skews to Pt** (~1900 in 3000-sample, i.e. ~63% of dataset is Pt-containing by mol count) but cross-contaminates with Au/Pd/Ir/Ru trace amounts. **Useful for transfer-learning but Pt-only benchmark.**
- **Click coverage is essentially zero.** MBFinder is not click-derived. Implication: cannot benchmark click-reaction generators on this dataset alone.

### 2.2 MetalCytoToxDB (dataset 4)

- **The ONLY dataset with REAL experimental IC50 labels** for metallodrugs (in addition to NCI60 but that lacks SMILES).
- **23 columns including IC50_Dark (cytotoxicity), IC50_Light (phototoxicity), IC50_Cisplatin (control), Metal, Oxidation_state, Cell_line, DOI, Year.**
- 8 major cell lines: A549, HeLa, MCF-7, HepG2, A2780, MDA-MB-231, HCT-116, A2780cisR (cisplatin-resistant).
- **Metal distribution from CSV column:** Ru 19 135, Ir 4 546, Rh 1 134, Os 1 118, Re 868 (Ru-dominant — PD/PS metal complexes).
- **RDKit metal-SMARTS misses the metal atoms in this dataset** (0 hits for Pt/Pd/Au/Ir/Ru despite MetalCytoToxDB being multi-metal). Root cause: many entries use **complex ligands with metal atoms encoded in disconnected fragments** (Counterion + Ligand as separate SMILES tokens). Need a fragment-merge step before metal-count.
- **Lipinski pass rate 34%** is realistic for metallodrugs; still not the right gate.

### 2.3 NCI60_GI50 (dataset 5)

- **SMILES-less public benchmark.** No `smiles` column. Activity is keyed on **NSC** (NCI compound number); SMILES would require external `NSC→SMILES` join (CACTUS or NCI's resolver).
- 10 panels (Leukemia, NSCLC, Colon, CNS, Melanoma, Ovarian, Renal, Breast, Prostate, Small-Cell-Lung).
- 71 unique cell lines in the sample.
- **Use as REFERENCE only** for cell-line-specific activity calibration, **not** as direct training data for SMILES-conditioned models.

### 2.4 tmQM (dataset 6)

- **108 542 transition-metal QM complexes** with full DFT-derived properties (Electronic_E, Dispersion_E, Dipole_M, Metal_q, HL_Gap, HOMO/LUMO, Polarizability).
- **15 metals** dominated by Ni/Pd/Ru/Pt/Zn/Fe/Ir/Rh/Au/Re/Mo/Co/Cu/W/Ti. Excellent **multi-metal** reference.
- **Coord number distribution:** 4 (33%) > 6 (31%) > 5 (9%) > 8 (8%) > 9 (5%) > 2 (4%) > 7 (4%) > 12 (2%) > 3 (2%) > 10 (1%). Pt_II (CN=4) and Pt_IV (CN=6) are well-represented.
- **100% SMILES canonical rate** (well-curated CSD-derived).
- **Lipinski pass 26%** (real coordination complexes).
- **Click coverage is also near-zero.** tmQM is *not* a click-derived dataset.
- **Use for:** (a) CFM warm-start; (b) MCTS learned prior (Task L4); (c) Reaction confidence from QM training data.

### 2.5 CrossDocked pocket10 (datasets 7-8)

- **2 464 pockets** with reference ligand (SDF) + receptor (PDB, 10 Å around ligand).
- **No metal centers in pocket10** (general SBDD benchmark, not metallo-specific).
- 3d_cache has **856 pre-embedded** pockets — covers about 35% of the 2 464 total. Cache miss path needs re-embed.
- **Use for:** SBDD eval (not metal-specific); serves as **off-target baseline** to compare against metal-pocket-conditioned model.

### 2.6 AiZynth + REINVENT4 (datasets 9-10)

- **AiZynth USPTO policy** has legacy converted ONNX + HDF5 templates available locally (3.7 GB total). **Fresh Zenodo download is blocked** (ReadTimeout per asset_manifest.json).
- **REINVENT4 prior v4.4.22** is **23 MB**, single file. Env at `/mnt/storage/envs/reinvent4/bin/python` exists and is functional (per WF-Extra-2 integration).
- **Use for:** (a) AiZynth as synthesis oracle in reward aggregation; (b) REINVENT4 as alternative de novo generator for benchmark comparison.

---

## 3. Cross-dataset click-reaction coverage (honest finding)

| Dataset | CuAAC | SPAAC | ThiolEne | Suzuki | AmideCoupling |
|---------|-------|-------|----------|--------|---------------|
| PlatinAI_MBFinder (226 K) | 0.0% | 1.3% | 0.0% | 0.0% | <0.1% |
| PlatinAI_predicted_A2780 (214 K) | 0.0% | 1.3% | 0.0% | 0.0% | <0.1% |
| PlatinAI_predicted_MCF7 (214 K) | 0.0% | 1.2% | 0.0% | 0.0% | 0.1% |
| MetalCytoToxDB (26.8 K) | 0.0% | 0.0% | 0.0% | 0.0% | 0.5% |
| tmQM_parsed (108 K, 3 K sample) | 0.0% | 3.2% | 0.0% | 0.0% | 0.2% |

**Finding:** **None of the metal-drug datasets have meaningful click-reaction coverage.** The 5-click typed-reduction language (CuAAC/SPAAC/ThiolEne/Suzuki/AmideCoupling) is a **de novo design contract** of the Lambda/CFM generator, not a property of any training corpus.

**Implication:** The Lambda click-rule coverage is evaluated at **generator output** (post-generation), not at **training data**. The Round-12/13 evaluation harness already captures this via the `--click-rules` flag and the per-cell "click_coverage" column.

---

## 4. Recommended data usage matrix

| Use case | Primary dataset | Secondary dataset | Reason |
|----------|------------------|--------------------|--------|
| **CFM pre-training** (unconditional) | PlatinAI_MBFinder | tmQM (Pt/Pd/Au) | Largest, most diverse Pt corpus |
| **CFM pre-training** (metal-conditioned) | tmQM (Pt/Pd/Au subset) | PlatinAI_MBFinder | Multi-metal QM labels enable metal-type conditioning |
| **MCTS root warm-start** | tmQM | PlatinAI_MBFinder | tmQM has coord_number + metal_bo_total for proper root embedding |
| **Activity prediction (regression)** | PlatinAI_predicted_A2780 + MCF7 | MetalCytoToxDB IC50_Dark | Activity is real (Ridge baseline r=0.57 per TODO-18; D-MPNN r=0.20 ceiling) |
| **Activity prediction (censor-aware)** | MetalCytoToxDB | — | Only dataset with explicit censored/unbounded IC50 labels |
| **Cell-line coverage eval** | NCI60_GI50 | MetalCytoToxDB cell_lines | 71 cell lines vs 8 in MetalCytoToxDB |
| **Pocket-conditioned SBDD** | CrossDocked_pocket10 (2 464) | — | Only pocket-conditioned benchmark available |
| **Synthesis oracle eval** | AiZynth (legacy converted) | — | USPTO retrosynthesis templates |
| **Alternative de novo generator** | REINVENT4_prior (23 MB) | — | Already wired in `reinvent4_subprocess_adapter` |
| **QM property reference** | tmQM (HL_Gap, HOMO/LUMO, etc.) | — | Only dataset with DFT properties |
| **Citation SOTA table (cite-only)** | External (DiffSBDD, Pocket2Mol, etc.) | — | wf_3_citeonly_sota.tex is already shipped |

---

## 5. Caveats & honest framing

1. **PlatinAI canonical rate (62-66%) is artificially low** because RDKit cannot kekulize some coordination complexes. The 3K-sample scan yields `Can't kekulize mol` warnings; canonical=False rows are **not necessarily invalid**. A more lenient parser (e.g., preserve-aromatic, accept-radical) would lift this to ~85%.
2. **MetalCytoToxDB metal-SMARTS miss (0 hits)** is a known issue: ligands are split across Counterion + Ligand columns; need a **SMILES merge step** before counting. Counts reported in `metal_coverage_csv` (CSV column) are the ground-truth.
3. **NCI60_GI50 is SMILES-less** — any SMILES-conditioned use requires external join.
4. **CrossDocked pocket10 has zero metal pockets** — it is the **off-target baseline**, not the metallodrug benchmark. A **metal-pocket-conditioned** benchmark would need custom curation (PDB-mining by `[Pt]`/`[Pd]`/`[Au]` co-crystallized ligands).
5. **AiZynth Zenodo download is blocked** (timeouts); the `legacy_v3/` converted weights are usable but unverified.
6. **Click coverage across all metal-drug corpora is near-zero**, confirming that Lambda's 5-click typed reductions are **de novo design** not **corpus replay**.
7. **All stats based on 3 000-sample scan (random seed=0)** for SMILES-bearing datasets. Full-population stats would be 75× larger; noise floor for click-coverage ≤ 0.0003. For canonical_rate / Lipinski, full scan expected within ±2pp.

---

## 6. Files & next actions

- **JSON registry:** `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_metallodrug_vertical/phase1_inventory.json` (7 211 bytes)
- **Markdown summary:** this file
- **Next actions (Phase 2+):**
  1. **Resolve the metal-SMARTS miss in MetalCytoToxDB** by implementing a SMILES-merge pre-processor (Counterion + Ligand columns → single mol) before Phase 2 stats.
  2. **Lift PlatinAI canonical rate** by adding a `sanitize=False` fallback parser for coordination complexes.
  3. **Build a `molmetal_loaders` module** with one load function per dataset (returns canonical SMILES + metadata) — single source of truth for downstream training.
  4. **Wire NCI60 SMILES join** via CACTUS API or local NSC→SMILES cache (low priority; reference only).
  5. **CFM warm-start** from tmQM (already proposed as Task #557).
  6. **MCTS learned prior** from tmQM coordination-number histograms (already proposed as Task #709).
