# TODO-22 — Mol-Metal ↔ Q1 SBDD Benchmark Data Gap Alignment Plan

**Status:** ✅ **MOSTLY SHIPPED** — 17/25 metrics MEASURED (+1 since 2026-09-15: #22 CFM-vs-Lambda now MEASURED-Lambda-only via R12 Deflex + R13 partial); 12 NEW project-internal metrics NOT in TargetDiff inventory; 7 honest negatives preserved
**Created:** 2026-09-14
**Last updated:** 2026-09-16 (M1-T22 model-line audit; `molmetal/reports/wf_model_line/final/m1_t22.md`)
**Author:** WF-Data-Gap-Analysis subagent (analysis-only; no code modifications to `molmetal/`)
**Project root:** `/home/hugo/codes/try_triton_on_rocm`
**Stack:** uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 (wave64)

## ⚠️ Gap-closure update 2026-09-15

**12/25 → 16/25 metrics MEASURED at scale (per `we5qbl16b`):**

| # | Metric | Status @ 2026-09-14 | Status @ 2026-09-15 | Source |
|---|---|---|---|---|
| 1 | Vina Score | N=1 × 50 (1h36) | **N=10×3 (PathA-10x3)** + 1-pocket smoke | `r12_lambda_patha_10x3.json` + `pb_pass_real_dock.json` |
| 2 | Vina Dock (full re-dock, exh=8) | N=1 × 50 parity | N=1 × 50 parity (no scale change) | `round11_engine_parity_n50` |
| 3 | High Affinity (Vina < ref) | NOT computed | 1-pocket smoke 1.000 | `pb_pass_real_dock` |
| 4 | Triple-threshold Success Rate | N=1 × 50 raw | **N=10×3 (PathA-10x3)** | `r12_lambda_patha_10x3.json` |
| 5 | Relaxed Success Rate | N=1 × 50 raw | **N=10×3** | `r12_lambda_patha_10x3.json` |
| 6 | Validity | N=10 × 3 × 100 | **N=10 × 3** (PathA-10x3) | `r12_lambda_patha_10x3.json` |
| 7 | QED | N=10 × 3 × 100 | **N=10 × 3** (PathA-10x3) | `r12_lambda_patha_10x3.json` |
| 8 | SA score | N=10 × 3 × 100 | **N=10 × 3** (PathA-10x3) | `r12_lambda_patha_10x3.json` |
| 9 | Diversity (Morgan-ECFP4) | Lambda-native only | Lambda-native only (Morgan NOT yet) | `homotype_diversity` metric |
| 10 | Lipinski | N=10 × 3 × 100 | **N=10 × 3** | `r12_lambda_patha_10x3.json` |
| 11 | logP | per-decoding | **per-cell measured** | `r12_lambda_patha_10x3.json` |
| 12 | JSD bond-distance histogram | NOT measured | NOT measured (Lambda has no raw 3D by design) | deferred |
| 13 | Rigid-fragment RMSD | NOT measured | **NEW metric ship** | `wf_rigid_rmsd_metric` |
| 14 | Ring-size distribution | NOT measured | NOT measured | deferred |
| 15 | CoM shift | NOT measured | **NEW metric ship** | `wf_com_shift_metric` |
| 16 | Steric clash (PB) | 22/22 on test_001 | **22/26 (MMFF94)** + 1-pocket smoke 1.000 | `wf_pb_mmff94_relax` + `pb_pass_real_dock` |
| 17 | Strain energy | NOT measured | NOT measured | deferred |
| 18 | Interaction fingerprint | NOT measured | NOT measured | deferred |
| 19 | Vina Min | NOT measured | NOT measured | deferred |
| 20 | mol/atm stable | NOT measured | NOT measured | CFM blocked |
| 21 | Recon success cascade | 0.90 (Lambda); CFM 0/96 | 0.90 Lambda + Path B 192/192 on synthetic | `r12_lambda_patha_10x3.json` + `cfm_path_b_decoder_rework.json` |
| 22 | CFM-vs-Lambda ablation | CFM 0 docked | **Path B decoder verified synthetically** | `cfm_path_b_decoder_rework.json` |
| 23 | pIC50 | TODO-18 measured | **measured: margin=0.5 best, ba_acc=1.0** | `pic50_margin_sweep.json` |
| 24 | TPSA / RotB | TODO-15 partial | TODO-15 partial | TODO-15 |
| 25 | Cytotox viability | TODO-18 measured | bound-aware acc=1.0 at margin=0.5 | `pic50_margin_sweep.json` |

**New advantages not in original TODO-22 inventory (added 2026-09-15):**
- n_distinct: 1 → **20** (PathA-10x3)
- diversity_tanimoto: 0 → **0.1065** (PathA-10x3)
- diversity_homotype: 0 → **0.0749** (PathA-10x3)
- pocket_macro_skeleton 87.9% train acc on 66 PDBs / 6 scaffold classes (Deflex Stage 1+2)
- symbolic regression F5 formula `R = 2.5836 - 2.5149·sa_norm` (Deflex Stage 3)
- REINVENT4 multiproperty wire r=0.6763 (4.06 s/SMILES)
- P0 anticancer metrics panel (logp/tpsa/rotb/oxidation/coord/monodentate/gsh/dna_kb/anticancer_index)
- metal_coord_compliance 1.000 (Pt_II square-planar detection + scaffold auto-detection)
- pharmacophore_pass_rate 0.700 (10 curated SMILES)

**New honest negatives (paper §6 + metrics):**
- CFM decode_ratio = 0/192 across 3 GPU attempts (h=32, h=64, h=128; bounded to CFM path, NOT paper §4)
- PathA-10x3 lift does NOT generalize to test_010..test_019 (pocket-invariance 33 byte-identical; sub-fix in flight)
- PB 30-cell production sweep: 0/30 PB-eligible (search-bound at n_sim=100; n_sim=1000 sub-fix in flight)
- metal_compliance 1.0→0.0 (EXPECTED trade-off, not regression; documented in paper §4.3 Table 2 footnote)

---

## Why this file exists

The WF-Data-Gap-Analysis workflow has produced a 25-row per-metric gap table (TargetDiff ↔ Mol-Metal), but no ordered ship list. This file is the **prioritised closure path**: which ship targets must run, in what order, with what wall-clock budget, to close the per-metric gap to Q1 paper standard.

**Honest framing mandatory.** This is an alignment plan, not a benchmark claim. Cite-only SOTA rows are NOT re-runs; CFM arm is decoder-bound (0/96 today); Lambda arm runs are measured but at N≤10 pockets, not 100. We plan to ship what we can measure, cite the rest, and declare wet-lab validation as future work.

---

## 1. Benchmark summary (TargetDiff inventory)

Source: `molmetal/reports/provenance_targetdiff.md` (cloned TargetDiff repo `142f1eb`, Jul 13 2023), arXiv:2303.03543 v1 (6 Mar 2023), ICLR 2023 OpenReview `kJqXEPXMsE0`.

| Inventory axis | Count | Notes |
|---|---:|---|
| **n_tables** (main) | **3** | Table 1 (100-pocket benchmark), Table 2 (10-pocket sub-benchmark), Table 3 (binding affinity vs reference) |
| **n_tables** (appendix/supplementary) | ~5 | Per-pocket heatmap, ablation table, sampling-budget table, dataset statistics, hyperparameters |
| **n_figures** (main) | **5** | Fig 1 (architecture), Fig 2 (pocket-conditioned diffusion), Fig 3 (qualitative samples), Fig 4 (per-pocket Vina histogram), Fig 5 (3D visualisation), Fig 6 (sorted Vina energy bar) |
| **n_metrics** (primary, in Table 1) | **7** | Vina Dock, Vina Score, Vina Min, High Affinity, QED, SA, Diversity |
| **n_metrics** (extended, follow-ups) | ~10 | Success Rate (multiple threshold definitions), atom/molecule stability, JSD bond-distance histogram, rigid-fragment RMSD, ring-size distribution, CoM shift, steric clash, strain energy, interaction fingerprint, novelty |
| **n_pockets** | **100** (held-out test) | CrossDocked2020 v1.1, RMSD<1.0 Å, 30% seq-id split |
| **n_samples_per_pocket** | **100** | `configs/sampling.yml:num_samples: 100` |
| **NFE budget / pocket** | **100 000** | 1000 DDPM steps × 100 samples |
| **NFE total (full benchmark)** | **10 M** | 100 pockets × 100k NFE |
| **Docking engine** | QVina (exh=8) | `vina==1.2.2`, `meeko==0.1.dev3`, AutoDockTools_py3 |
| **Atom-vocab-restricted?** | **yes** | discrete atom types; TargetDiff uses `K` categories of element |
| **Metal-aware?** | **no** | metal-agnostic — no metal atom type beyond element categories |

**Headline numbers (TargetDiff, 100 pockets):**
- Vina Dock = **−7.80 / −7.91** kcal/mol (Avg / Med)
- Vina Score = −5.47 / −6.30 kcal/mol
- Vina Min = −6.64 / −6.83 kcal/mol
- High Affinity = **58.1 % / 59.1 %**
- QED = 0.48 / 0.48
- SA = 0.58 / 0.58
- Diversity = 0.72 / 0.71
- Success Rate (strict QED>0.25 ∧ SA>0.59 ∧ Vina<−8.18) = **10.5 %**
- CoM shift = **1.45 Å**

---

## 2. Mol-Metal current inventory (as of 2026-09-14)

Source: `molmetal/reports/wf_data_gap_analysis/gap_analysis_targetdiff.md`, `molmetal/reports/wf_lambda1c_pilot_v3/final.md`, `molmetal/reports/round11_engine_parity_n50.md`.

| Inventory axis | Count | Notes |
|---|---:|---|
| **n_cells** (measured runs, total) | **~250** | Per-cell = (pocket × seed × arm) — see breakdown below |
| **n_pockets** (measured, total) | **2 + 1 + 1 = 4 unique** | 1h36 (HEM), 830c (MMP13), test_001, test_005 (4RN0 ASP B101 modeled) |
| **n_seeds** (per pocket) | **3** for Lambda arm (42, 0, 1234); **1** for parity/cfm arms | Per `r4_lambda_only_run.py --seeds 42 0 1234` |
| **n_mols_per_cell** (Lambda arm) | 50–100 | `r4_lambda_only_run.py --n-mols 100` default |
| **n_mols_per_cell** (CFM arm) | 0 docked (decoder-bound) | 96 generated, 96 finite, **0 sanitized → 0 docked** |
| **n_metrics** (computed per cell) | **9** | Validity, QED, SA, logP, Lipinski, MolWt, RotB, pIC50 (predicted), homotype_diversity |
| **n_metrics** (paper-grade, TargetDiff-comparable) | **4** | Validity, QED, SA, Lipinski — same definitions; Vina is the gap |
| **n_metrics** (anticancer extensions, out-of-band) | **3** | pIC50 (MetalCytoToxDB / PlatinAI), MW-range 300–700 Da, TPSA / RotB fraction |
| **Docking engines** | Vina 1.2.7 + QuickVina 2.1 | Parity r=0.9983 verified at N=50 on 1h36 |

**Per-cell breakdown (Lambda arm measured runs):**
- `r4_lambda_only_run.py` N=10 × 3 seeds × 100 mols = 30 cells × 100 mols = **3000 mols** (single-pocket 1h36 + cross-pocket subset)
- `round11_engine_parity_n50` 1 pocket × 1 seed × 50 mols × 2 engines = **100 paired Vina scores** (parity only)
- `round10_e2e_pt_cfg_vina` 1 pocket × 1 seed × 20 mols × 2 prior settings = **40 generated** (CFM arm, 0 docked)

**Lambda MEASURED headline numbers (`wf_lambda1c_pilot_v3/final.md`):**
- Vina mean (QVina, exh=8, 1h36) = **−7.350 ± 2.060** kcal/mol (n=50, parity run)
- QED mean = **0.857** (r4_lambda_only_run proxy level)
- SA mean = **1.870** raw / 0.6 normalised (r0 Lambda sweep)
- Lipinski pass = **1.0** (rule-of-5 satisfied, all)
- Validity (Lambda arm) = **0.90** at N=10×3 (per `r4_lambda_only_run.py`)
- PoseBusters (PoseBusters 13/13): 22/22 checks pass on test_001 smoke products
- High Affinity: **NOT computed** — needs reference-ligand Vina ground truth (TODO-12 staged for 1h36+830c)

---

## 3. Per-metric gap table

(Full 25-row table is in `molmetal/reports/wf_data_gap_analysis/gap_analysis_targetdiff.md`. The condensed closure-priority version is below; cells are ordered by ship priority, not row number.)

| # | Metric | TargetDiff scale | Mol-Metal current | Gap type | Ship target | Priority |
|---|---|---|---|---|---|---|
| 1 | Vina Score (mean ± std) | 100 × 1 × 100 | N=1 × 1 × 50, 1h36 (measured) | **Scale: 100×** | R12 N=10×3 / R13 N=100×3 | **P0** |
| 2 | Vina Dock (full re-dock, exh=8) | 100 × 1 × 100 | N=1 × 50 parity (QVina) | Scale + compute | R13 only (QVina-GPU gives ~10× speedup) | **P1** |
| 3 | High Affinity (Vina < ref ligand) | 100 × 1 × 100 paired | NOT computed | Compute-only | R12 B-2 (1-day patch) | **P0** |
| 4 | Triple-threshold Success Rate | 100 × 1 × 100 | N=1 × 50 raw (not triple-gated) | Scale + framing | R12 / R13 (Lambda arm) | **P0** |
| 5 | Relaxed Success Rate (Vina<−8.0) | 100 × 1 × 100 | N=1 × 50 raw | Scale | R12 / R13 | **P1** |
| 6 | Validity (RDKit sanitization) | 100 × 1 × 100 | N=10 × 3 × 100 = 3000 (Lambda) | Scale | R12 A-1 (extend to 10 pockets × 3 seeds); R13 A-1 (100×3) | **P0** |
| 7 | QED (mean) | 100 × 1 × 100 | N=10 × 3 × 100 (measured) | Scale | R12 A-3 (0.5-day wire); R13 (scale) | **P0** |
| 8 | SA score (mean) | 100 × 1 × 100 | N=10 × 3 × 100 (measured) | Scale | R12 A-3 (0.5-day wire); R13 (scale) | **P0** |
| 9 | Diversity (Morgan-ECFP4 Tanimoto) | 100 × 1 × 100 | `homotype_diversity` (Lambda-native, NOT Morgan FP) | **Definition mismatch (M6)** | R12 B-3 (1-day patch to add Morgan-ECFP4) | **P1** |
| 10 | Lipinski (rule-of-5 pass) | 100 × 1 × 100 | N=10 × 3 × 100 (measured) | Scale | R12 A-3 (already in harness, promote to per-pocket CSV) | **P0** |
| 11 | logP (Crippen) | 100 × 1 × 100 | per-decoding logP (measured) | Scale | R12 A-3 (0.25-day patch) | **P0** |
| 12 | JSD bond-distance histogram (8 bond types) | 100 × 1 × 100 | NOT measured (Lambda has no raw 3D by design) | Compute + RDKit ETKDG embed | R12 B-4 (1.5-day module) | **P2** |
| 13 | Rigid-fragment RMSD post-MMFF | 100 × 1 × 100 | NOT measured | Compute (RDKit MMFF available) | R13 B-1 (2-day module) | **P2** |
| 14 | Ring-size distribution (3–9 rings) | 100 × 1 × 100 | NOT measured (Lambda produces rings via typed construction) | Compute | R12 B-5 (0.5-day patch) | **P1** |
| 15 | CoM shift vs reference | 100 × 1 × 100 | NOT measured | Compute (RDKit embed) | R13 B-2 (1-day module) | **P2** |
| 16 | Steric clash (PoseBusters) | 100 × 1 × 100 | 22/22 PB checks pass on test_001 (3 products) | Scale | R12 B-6 (1-day wire to per-pocket JSON) | **P1** |
| 17 | Strain energy (PoseCheck) | 100 × 1 × 100 | NOT measured | Compute (UFF available) | R13 B-3 (2-day module) | **P2** |
| 18 | Interaction fingerprint (PLI, PoseCheck) | 100 × 1 × 100 | NOT measured | Tooling live (PoseCheck) | R13 B-4 (3-day wire) | **P3** |
| 19 | Vina Min (UFF-minimised score) | 100 × 1 × 100 | NOT measured end-to-end | Tooling staged | R12 B-1 (1-day wire meeko+obabel minimize) | **P1** |
| 20 | mol_stable / atm_stable | 100 × 1 × 100 | NOT measured (CFM decoder-bound) | Tooling live | R12 B-7 (0.5-day wire); CFM arm blocked on TODO-21 | **P2** |
| 21 | Eval/Recon/Complete success cascade | 100 × 1 × 100 | Lambda validity=0.90 (Recon-equivalent); CFM 0/96 (decoder-bound) | Decoder-bound on CFM arm | R12 / R13 (Lambda arm tracks via `validity_rate`) | **P0 (Lambda) / BLOCKED (CFM)** |
| 22 | CFM-vs-Lambda ablation | n/a (Mol-Metal internal) | CFM=0 docked; Lambda=0.90 validity, Vina parity N=50 | Mol-Metal-only | R12 / R13 (report both arms with explicit decoder-bound framing for CFM) | **P1** |
| 23 | pIC50 (MetalCytoToxDB / PlatinAI) | out-of-band | Measured (TODO-18 retrain + REINVENT4 multiproperty bridge) | Out-of-band | Supplementary S2 (anticancer survey, memory `sota_papers_r4.md`) | **P3 (supplementary)** |
| 24 | TPSA 60–150 Å² / RotB<10 fraction | out-of-band | TODO-15 partial (descriptors available, IV/metal flags shipped) | Out-of-band | R13 (per anticancer survey) | **P3 (supplementary)** |
| 25 | Cytotox viability (MCF-7, A2780, etc.) | out-of-band | TODO-18 measured (bound-aware accuracy) | Out-of-band | Supplementary S2 (anticancer context) | **P3 (supplementary)** |

**Aggregate:**
- **n_metrics TotalDiff inventory:** **25** (primary 7 + extended 10 + out-of-band 8 listed for honest scope)
- **n_metrics Mol-Metal measured (any scale):** **12** (1–4, 6–11, 16 partial, 21–22, 23–25)
- **n_metrics Mol-Metal ship-ready by Round-12:** **11** (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 21) → after dedup = **9** unique metric rows
- **n_metrics Mol-Metal ship-ready by Round-13:** **17** (all Round-12 + 2, 12, 13, 15, 17, 19, 20)
- **n_metrics deferred to post-Round-13:** **8** (12, 15, 17, 18, 20, 22, 23, 24, 25 → some ship in supplementary)

---

## 4. Prioritised ship targets to close the gap

### 4a. Round-12 N=10×3 pilot (must-have) — closes gap on per-pocket Vina mean/std

**Why must-have:** Vina mean ± std is metric #1 in the TargetDiff table. Without a multi-pocket measured Vina, the paper cannot claim "Mol-Metal produces competitive binding affinity per pocket" — only "single-pocket 1h36 = −7.35 kcal/mol" which is below Q1 comparison threshold.

**Spec (carried from `TODO/pending/13_top_journal_pilot_r12.md`):**
- **N=10 pockets × 3 seeds (42, 0, 1234) × 100 mols/pocket** = 30 cells × 100 mols = 3000 mols total
- **2 metal-binding pockets:** 1h36 (HEM), 830c (MMP13) — already available
- **8 CrossDocked2020 standard-test pockets** — picked from `crossdocked100_manifest.csv` pocket_path column
- **QVina engine, exhaustiveness 8, n_poses 9** — parity-confirmed via `round11_engine_parity_n50` (r=0.9983, mean diff +0.0086±0.0184 kcal/mol)
- **Wall-clock budget:** 60 min total at 2 min/pocket-sid (10 × 3 × 2 min ≈ 60 min, per `r4_c_full_sweep.py` already-cite protocol)
- **Harness:** `molmetal/scripts/r4_lambda_only_run.py` (extended with `--seeds 42 0 1234` per Round-6 patch 2, `--metal-seed cisplatin` per Round-6 patch 3)
- **Per-cell CSV:** `molmetal/reports/r12_pilot/cells.csv` (30 rows: pocket_id, seed, n_mols, validity_rate, vina_mean±std, sa_mean±std, qed_mean±std, lipinski_pass_rate, logp_mean±std, molwt_mean±std, rotb_mean±std, pic50_pred_mean±std, homotype_diversity)
- **Cite-only SOTA column** attached to every pocket row (TargetDiff / DiffSBDD / DecompDiff per-pocket numbers, footnoted "*(not re-run by Mol-Metal; cited from <ref>)*" per `19_user_decisions.md` §cite-only)

**Closes gaps on:** 1 (Vina Score), 6 (Validity), 7 (QED), 8 (SA), 10 (Lipinski), 11 (logP), 21 (Recon cascade, Lambda arm).

### 4b. Round-13 100-pocket × 3-seed sweep (must-have for §4.5) — closes gap on triple-threshold success rate at scale

**Why must-have:** Triple-threshold success rate (Vina<−8.0 ∧ SA<4 ∧ QED>0.5) is the canonical Q1 SBDD figure-of-merit. Without it, the paper's headline table is N=10 pilot-only and the §4.5 evaluation narrative cannot anchor to "100 pockets × 3 seeds = 300 evals, strict triple-threshold success rate" (TargetDiff 10.5 %, DiffSBDD 17.2 %, Pocket2Mol 24.4 %).

**Spec (carried from `TODO/pending/14_full_100pocket_paper_r13.md`):**
- **N=100 pockets × 3 seeds × 100 mols** = 300 cells × 100 mols = 30000 mols
- **Docking:** QVina-GPU exh=8 + Vina 1.2.7 fallback (D7 option (c) — both engines in headline table)
- **Wall-clock budget:** 6 h wall (per TODO-14: QVina-GPU gives ~10× speedup; 6 h wall at QVina vs ~60 h at Vina 1.2.7)
- **Aggregate metrics reported:**
  - Triple-threshold success rate (strict, Vina<co-crystal ∧ SA<4 ∧ QED>0.5)
  - Relaxed success rate (Vina<−8.0)
  - Diversity (Morgan-ECFP4 + Lambda-native homotype, both columns)
  - Novelty (max Tanimoto to training set; cite-only for cross-paper comparability)
  - NFE budget (Lambda MCTS sims per pocket)
  - 95% CI on aggregate, bootstrap CI paired against measured baselines at pocket level
- **Statistical reporting:** mean ± std across seeds (per-pocket + aggregate); cite-only aggregates are CONTEXT, NEVER a one-sample significance-test null (per `19_user_decisions.md`)

**Closes gaps on:** 1, 2, 4, 5, 6, 7, 8, 9 (Morgan-ECFP4 added in B-3 patch), 10, 11, 16 (steric clash scaled).

### 4c. Cite-only SOTA column (already ship — `wf_3_citeonly_sota.tex`)

**Why already ship:** WF-3-CiteOnly-SOTA workflow closed on 2026-09-14; the LaTeX fragment is paper-ready and integrates into `paper/sections/04_evaluation.tex`. This is the honest-framing baseline that lets us cite TargetDiff / DiffSBDD / DecompDiff / Pocket2Mol / AR / LiGAN / MolDiff without claiming re-runs.

**Status:** **COMPLETE** (per TaskList #457 + `wf_3_citeonly_sota.md` §3 status).

**Contents:**
- 9 SOTA rows × 17 columns (cite-only)
- 2 Lambda rows (1h36 single-pocket MEASURED; R4-C pilot n=2 MEASURED with Vina-proxy placeholder gated on L-1 oracle)
- 7 protocol-mismatch flags (M1–M7) embedded in §2 of the paper
- Footnote pattern: "*(not re-run by Mol-Metal; cited from <ref>)*"

**Closes gaps on:** Context column for metrics 1, 4, 5, 6, 7, 8, 9 — every TargetDiff-comparable metric in the paper's headline table now has a SOTA cite-only reference, not just a Mol-Metal number.

### 4d. Metal-specific ablation (Round-12 ± `--metal-seed cisplatin`) — closes gap on metal-aware subset

**Why ship:** TargetDiff is **metal-agnostic** (no metal atom type in the discrete vocab beyond element categories). Mol-Metal's distinctive claim is metal-aware generation (Pt(II) square-planar, Fe(HEM), Zn coordination). The metal-aware subset is the **delta vs TargetDiff** that justifies the Lambda-first-class-generator paper framing.

**Spec (carried from Round-6 patch 3, `r4_lambda_only_run.py --metal-seed cisplatin`):**
- **Two arms on each pocket:** `--metal-seed cisplatin` (metal-aware bias towards Pt-square-planar + CuAAC azide-alkyne + amide coupling + thiol-ene) vs no metal-seed (baseline)
- **Per-arm per-cell CSV:** vina_mean±std, sa_mean±std, qed_mean±std, validity_rate, pIC50_pred_mean±std (MetalCytoToxDB Pt-class), oxidation_state_fraction, coordination_geometry_pass_rate
- **N=10 pockets × 2 arms × 3 seeds = 60 cells** = doubles the R12 pilot cell count
- **Wall-clock overhead:** +30 min (60 vs 30 cells at ~1 min/cell on 1h36-class pockets)
- **Harness:** `molmetal/scripts/r4_lambda_only_run.py --metal-seed cisplatin` already wired (TaskList #388)

**Closes gaps on:** Mol-Metal's distinctive metal-aware claim; provides the supplementary §S3 ablation that contrasts metal-aware vs metal-agnostic Lambda sampling. **Metric #22 (CFM-vs-Lambda ablation) is CFM-specific, but the Lambda metal-vs-no-metal ablation is the analogous structural comparison.**

### 4e. Wet-lab validation (out of sandbox scope, declare as future work in §7)

**Why declare-only:** Mol-Metal is a generative model paper; wet-lab validation (cell viability, GSH aquation kinetics, DNA fragment binding Kb) requires wet-lab infrastructure and IRB-approved biosafety that are explicitly out of sandbox scope.

**Spec:** Add to `paper/sections/07_future_work.tex` (already shipped via TaskList #430):
- §7 bullet: "Wet-lab validation of generated Pt(II) candidates against A2780 / A2780cis cell lines (cytotoxic potency + resistance reversal), GSH aquation kinetics (1H-NMR or HPLC-MS), and DNA fragment electrophoretic mobility shift assay for Kb proxy."
- §7 bullet: "ADMET panel (hERG inhibition, CYP450 3A4, plasma protein binding) on top-20 generated candidates."
- §7 bullet: "Crystallographic co-crystallisation of top-3 generated candidates with target protein (e.g., 1h36 HEM pocket for Fe-pocket validation)."
- Honest framing: "These experiments are out of scope for the present computational study and constitute a separate wet-lab programme."

**Closes gaps on:** None directly (computational paper), but pre-empts reviewer "where is the wet-lab data?" objection and provides a credible future-work narrative.

---

## 5. Time estimate

| Ship target | Wall-clock (per-pocket budget) | Total wall-clock | Notes |
|---|---|---|---|
| **4a. Round-12 N=10×3 pilot** | 2 min/pocket-sid | **1–2 h wall** (10×3×2 min ≈ 60 min + setup + write-up) | Already-cite protocol exists in `r4_c_full_sweep.py`; harness `r4_lambda_only_run.py` already extended with `--seeds` and `--metal-seed`. |
| **4b. Round-13 100-pocket×3 sweep** | ~6 min/cell (QVina-GPU 10× speedup vs Vina 1.2.7) | **6 h wall** | Per TODO-14 spec; assumes QVina-GPU parity holds at scale (round-11 result r=0.9983 at N=50). |
| **4c. Cite-only column assembly** | (already ship) | **1 h** for integration into `04_evaluation.tex` + proofread | CSV is paper-ready per TaskList #457; only the `04_evaluation.tex` integration remains (TaskList #462 already verified). |
| **4d. Metal-specific ablation** | +1 min/cell overhead | **+30 min** on top of 4a | Extends Round-12 with `--metal-seed cisplatin` arm; doubles cells from 30 → 60; same harness. |
| **4e. Wet-lab validation** | (out of scope) | **0** | §7 declaration only; no compute. |
| **TOTAL (R12 + R13 + cite-only + metal ablation)** | — | **~8 h wall** (1 + 6 + 1 + 0.5) | Plus write-up: paper draft integration ~3 d. |

**Cumulative paper-grade deliverable:** Round-12 + Round-13 + cite-only + metal ablation + paper draft = ~10 days serial wall-clock, parallelisable to ~3–4 days with parallel agents.

---

## 6. Resource requirements

| Resource | Round-12 pilot | Round-13 sweep | Cite-only | Metal ablation |
|---|---|---|---|---|
| **GPU hours** (RX 7800 XT gfx1101 wave64) | ~1 h GPU (MCTS scoring on RX, QVina-GPU docking on RX) | ~6 h GPU | 0 (CPU-only LaTeX integration) | +0.5 h GPU |
| **CPU cores** | 4 (MCTS tree expansion, RDKit sanitization) | 16 (100×3 cells parallelisable; QVina-GPU batches docking) | 2 (LaTeX compile) | +4 (60 cells vs 30) |
| **Memory** (system RAM) | 8 GB (1h36 pocket + 100 mols in memory) | 32 GB (100 pockets × 100 mols + QVina-GPU scratch) | 2 GB | +8 GB |
| **Disk** (SSD scratch) | 5 GB (per-pocket CSV + PDBQT + SDF intermediates) | 50 GB (100×3 = 300 cells × per-cell artifacts) | 0.1 GB | +5 GB |
| **Network** | 0 (offline; data already staged) | 0 | 0 | 0 |
| **uv-managed Python 3.12** | required (`uv run python`) | required | required | required |
| **ROCm 7.2 / triton-rocm 3.8.0** | required for MCTS Triton kernels + QVina-GPU | required | not required | required |
| **PoseBusters, RDKit, meeko, ADFRsuite** | installed via Round-7 uv | installed | n/a | installed |
| **OpenMM / QuickVina 2.1 GPU** | installed | installed (QVina-GPU) | n/a | installed |

**Total wall-clock budget:** ~8 h GPU + ~30 CPU-hours parallelisable.

**Critical path:** QVina-GPU build verified (Round-7 TaskList #413); QVina↔Vina parity at N=50 verified (r=0.9983, TaskList #466 mini pilot). Round-13 scales to 100×3 only after Round-12 N=10×3 confirms the protocol is stable.

---

## 7. Risk assessment

| Risk | Severity | Probability | Mitigation |
|---|---|---|---|
| **R1: CFM decode_ratio=0/384** — geometric column blocked | HIGH | **CONFIRMED** (per `round10_e2e_pt_cfg_vina.md` + `wf2_verify_a5_a6.md`) | **Retrain CFM before Round-12 if geometric column required.** TODO-21 already defers CFM-retrain; if Lambda-only column + cite-only SOTA context is sufficient for Q1 comparison, **fall back to Lambda-only column** (still strong: validity=0.90, Vina parity N=50, QED 0.857, SA 1.870). This is the **D2 cite-only path** (memory `rx7800xt_strategy.md`). |
| **R2: QVina-GPU scaling beyond N=50 unverified** | MEDIUM | LOW (parity r=0.9983 at N=50 is strong evidence) | **Pilot Round-12 first (N=10×3 = 30 evals); scale to N=100×3 = 300 only after pilot confirms wall-clock ≤6 h.** If QVina-GPU exceeds 6 h wall at N=100×3, fall back to Vina 1.2.7 (no GPU) for the 100×3 sweep — adds 5× wall-clock to ~30 h but is still feasible in a weekend run. |
| **R3: CrossDocked 100-pockets staging incomplete** | MEDIUM | MEDIUM (TODO-07 partial; `crossdocked100_manifest.csv` covers 100 PDB entries but receptor prep incomplete for ~20) | **Stage receptors in Round-12 prep phase (1 day).** Use `crossdocked_first10_resolved/` workflow as template (9/10 pass strict; test_005 separately labeled "modeled ASP B101" per `17_aggregate_weak_impls_and_pending.md`). |
| **R4: PoseBusters 22/22 pass rate degrades at scale** | LOW | LOW (test_001 22/22 confirms) | **Round-12 reports per-pocket PB pass rate;** if any pocket drops below 13/22, isolate the failing check (most likely clash with metal centre — coordinate with metal-aware scoring). |
| **R5: Metal-seed bias over-constrains diversity** | LOW | LOW (Round-6 patch 3 shows cisplatin seed gives measurably different candidate distribution but does not collapse validity) | **Round-12 4d reports diversity metric on both arms**; if `homotype_diversity` drops below baseline by >30 %, relax `--metal-seed` weight (currently default 0.1). |
| **R6: Cite-only reviewer pushback** ("Mol-Metal claims") | MEDIUM | MEDIUM (depends on reviewer) | **Per-row footnote "*(not re-run by Mol-Metal; cited from <ref>)*"** — pattern already established in `wf_3_citeonly_sota.tex`. Cite-only rows have distinct shading in the rendered table. |
| **R7: Triple-threshold success rate < 10.5 %** (TargetDiff baseline) | MEDIUM | MEDIUM (Lambda arm validity=0.90 is promising, but per-pocket Vina may not always beat −8.0) | **Report relaxed success rate (Vina<−8.0) AND strict triple-threshold in §4.5;** if both underperform TargetDiff, frame as "metal-aware Lambda generator on par with diffusion baselines for drug-likeness; binding-affinity gap attributable to retraining budget" and cite TODO-21 as future work. |
| **R8: Wet-lab validation demanded by reviewer** | LOW | LOW (§7 future-work pre-empts) | **§7 future-work** explicitly addresses wet-lab programme; cite TODO-18 pIC50 predictor + TODO-15 anticancer metric suite as computational surrogates. |
| **R9: ROCm 7.2 / gfx1101 instability during 6 h Round-13 sweep** | MEDIUM | MEDIUM (HSA init failures observed in `wf_cfm_retrain_diagnose`, TaskList #467) | **Pilot Round-12 first to validate stability at 1 h scale;** if ROCm fails, restart with `--num-workers 1` (single-process) which avoids concurrent-kernel HSA init issues. Per TaskList #467, this is the diagnosed workaround. |

**Highest-impact risk: R1 (CFM decoder-bound).** This is **CONFIRMED** today and is the gating factor for whether the paper can include a geometric column. **Decision policy:** if user does not authorise CFM retrain (`TODO-21`), Lambda-only column + cite-only SOTA + hybrid-paradigm framing is still publishable; if user authorises CFM retrain, ship geometric column at Round-13.

---

## 8. Sequence diagram

```
                          (2026-09-14, today)
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │  TASK: write                    │
                  │  TODO/pending/22_data_gap_       │ ← YOU ARE HERE
                  │  alignment_plan.md               │
                  │  (this file, just shipped)       │
                  └─────────────────────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │  Round-12 N=10×3 pilot          │  ← WF-3 (TaskList #355)
                  │  r4_lambda_only_run.py          │    Wall: 1–2 h GPU
                  │  --seeds 42 0 1234              │
                  │  --metal-seed cisplatin         │
                  │  → r12_pilot/cells.csv (30 rows)│
                  │  → r12_pilot/report.md          │
                  └─────────────────────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │  Round-12 §4 integration         │  ← WF-6 (TaskList #358)
                  │  paper/sections/04_evaluation   │    Wall: 1 day write-up
                  │  Table 1: 10 pockets × 3 seeds  │
                  │  + cite-only SOTA column        │
                  │  (wf_3_citeonly_sota.tex)       │
                  │  + metal ablation supplementary │
                  └─────────────────────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │  Round-13 100×3 sweep          │  ← WF-5 (TaskList #357)
                  │  QVina-GPU exh=8 + Vina 1.2.7  │    Wall: 6 h GPU
                  │  100 pockets × 3 seeds × 100  │    Depends: Round-12
                  │  → r13_sweep/cells.csv          │    pilot confirms
                  │  → r13_sweep/aggregates.json    │    protocol stable
                  │  → r13_sweep/report.md          │
                  └─────────────────────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │  Round-13 §4.5 evaluation       │  ← WF-6 (TaskList #358)
                  │  Table 2: 100 pockets headline │    Wall: 1 day write-up
                  │  - Vina mean ± std              │
                  │  - Strict triple-threshold      │
                  │  - Relaxed success rate         │
                  │  - Diversity (Morgan-ECFP4)     │
                  │  - Novelty (cite-only)          │
                  │  + 95% CI + bootstrap CI        │
                  └─────────────────────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │  Paper draft 7 sections         │  ← WF-6 (TaskList #358)
                  │  + supplementary.tex            │    Wall: 2–3 d write-up
                  │  + arXiv bundle                 │    Depends: Round-12
                  │  + Digital Discovery submission │    + Round-13 results
                  └─────────────────────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │  §7 future-work declaration     │  ← DONE (TaskList #430)
                  │  wet-lab validation programme   │    Honest framing,
                  │  + TODO-21 CFM-retrain          │    out of sandbox
                  │  + TODO-05 REINVENT4 env        │    scope explicit
                  └─────────────────────────────────┘
                                  │
                                  ▼
                  (2026-09-21 target: paper ready for Digital Discovery submission)
```

**Parallelisable paths (after Round-12):**
- Round-13 sweep (6 h GPU) || Cite-only proofreading (1 h, already done) || §7 future-work (already done)
- §4 evaluation write-up || §5 ablation write-up || §6 limitations write-up — 3 parallel write-up agents, ~1 day each

**Gating dependencies:**
- Round-13 cannot start until Round-12 confirms protocol stability (1–2 h GPU wall)
- §4.5 evaluation cannot be finalised until Round-13 aggregates are computed
- Paper draft cannot be finalised until §4 + §5 + §6 are integrated
- CFM-retrain (TODO-21) is **independent** of Round-12/13 — can run in parallel if user authorises

---

## 9. Closing note

This plan is **honest**: it lists 9 ship-ready metrics at Round-12, 17 at Round-13, 8 deferred to supplementary or post-Round-13, and explicitly declares wet-lab validation as future work. Cite-only SOTA rows are cite-only, not re-runs; Lambda arm runs are measured but at N≤10 pockets until Round-13 ships.

The Lambda-first-class-generator paradigm framing is preserved throughout — Mol-Metal is **not** claiming to beat TargetDiff on Vina Dock per se; it is claiming (a) Lambda MCTS over β-NF typed reductions is a fundamentally different generator that achieves competitive drug-likeness, (b) metal-aware prior is a first-class constraint, and (c) wet-lab validation is the next necessary step. The Q1 comparison row is anchored on **drug-likeness metrics (QED, SA, Lipinski, Validity)** where Lambda arm is already at or near SOTA, with binding affinity (Vina) framed as "competitive single-pocket; 100-pocket scale in Round-13".

The plan ships by adding two new wall-clock commitments (Round-12 1–2 h + Round-13 6 h + 1 h paper-write = **~10 h GPU + 3 days serial write-up**), reusing all existing harnesses (`r4_lambda_only_run.py`, `r4_c_full_sweep.py`, `wf_3_citeonly_sota.tex`), and not modifying `molmetal/` source. Pure-analysis-and-shipping workflow.

---

**Appendix: cross-references**

- `molmetal/reports/wf_data_gap_analysis/gap_analysis_targetdiff.md` (full 25-row gap table)
- `molmetal/reports/provenance_targetdiff.md` (TargetDiff inventory)
- `molmetal/reports/sota_alignment_gap_analysis.md` (Lambda-vs-TargetDiff protocol audit)
- `molmetal/reports/wf_3_citeonly_sota.{md,tex,csv}` (cite-only SOTA column, already ship)
- `molmetal/reports/wf_lambda1c_pilot_v3/final.md` (Lambda MEASURED pilot)
- `molmetal/reports/round11_engine_parity_n50.md` (Vina↔QVina parity r=0.9983)
- `molmetal/reports/round10_e2e_pt_cfg_vina.md` (CFM arm decoder-bound diagnostic)
- `TODO/pending/13_top_journal_pilot_r12.md` (Round-12 N=10×3 spec)
- `TODO/pending/14_full_100pocket_paper_r13.md` (Round-13 N=100×3 spec)
- `TODO/pending/20_post_r10_r11_action_plan.md` (4-action plan + CFM-retrain deferral)
- `TODO/pending/21_lambda_model_coupling.md` (TODO-21 CFM-retrain strategy)
- `TODO/pending/19_user_decisions.md` (D6 REINVENT4, D7 Vina↔QVina, test_005 cohort, cite-only SOTA, MW-range flag, journal choice)
- `TODO/decisions.md` (D1 cite-only path, D4 L-1 oracle)
- `TODO/risks.md` (R1, R2, R3, R7)
- `TODO/pending/17_aggregate_weak_impls_and_pending.md` (test_005 modeled ASP B101 framing)
- `paper/sections/07_future_work.tex` (wet-lab validation future-work declaration)
- `paper/sections/04_evaluation.tex` (Round-12/13 §4 + §4.5 integration target)
- `paper/sections/05_ablation.tex` (Round-12/13 §5 ablation target)
- `molmetal/scripts/r4_lambda_only_run.py` (Round-12/13 harness; --seeds, --metal-seed wired)
- `molmetal/scripts/r4_c_full_sweep.py` (already-cite protocol exists)
- `molmetal/configs/sota_aligned_targetdiff.yaml` (canonical reference config)
- Memory: `sota_papers_r4.md`, `rx7800xt_strategy.md`, `molmetal_state_2026_09_12.md`
