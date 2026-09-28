# WF-134 — Master Synthesis: 22-Metric Baseline Definition for Lambda-Metal

**Workflow:** wf134_metric_survey (Phase 4)
**Date:** 2026-09-27
**Purpose:** Single-document verdict per metric on (a) what SOTA convention it should measure, (b) what Mol-Metal currently measures, (c) the gap and disposition (REDESIGN / KEEP / DROP). Source of truth for paper §4 metric-mixing claims, R17 honest framing, and follow-up Wf prioritization.

---

## 0. Verdict (one paragraph)

Of our 22 conceptual metrics, **15 are KEEP** (aligned with Tier-1 SOTA, on-disk evidence intact), **5 are REDESIGN** (definition or measurement gap that will misreport in paper §4 if left as-is), and **2 are DROP or DEFER** (Tier-1 paper not yet shipped in our pipeline or principled reason to retire). The single highest-priority REDESIGN is **m15 rigid_fragment_rmsd**: per Wf-126 P1 honest finding, it is a *self-RMSD between two ETKDGv3+UFF conformers of the same molecule*, with **no pocket involvement**, so our 1.241 Å across all 5 pockets is by construction not comparable to TargetDiff / Pocket2Mol ligand-vs-crystal RMSD. The next four REDESIGNs are m3/m10 Vina (no Pt/Pd/Au chemotype tuning → systematic underestimation of metallodrug binding), m8 PB-Valid (only `mol` mode shipped, no `dock` mode steric-clash coverage), m18 Protein-Metal AUC (resolution mismatch: our complex-level pooled vs. SOTA per-residue), and m20 Tumor-Specific Protein-Metal binding (no shipped pipeline). The two DROP/DEFER are m9 Interaction Parity (SOTA2 leaderboard number is itself being deprecated and is not in Tier-1 publications) and m12 Synthesis Rate (R16 Path A reaches 1.0 trivially; no informative signal at our 12-tile click-library scale). Net effect: 18/22 metrics can ship in a paper §4 table today with honest framing; 4/22 require a metric-rewrite Wf (Wf-135 suggested).

---

## 1. Per-metric verdict table

| # | Metric (id) | Our on-disk JSON | SOTA convention | Gap verdict | Disposition |
|---:|---|---|---|---|---|
| 1 | **m1 Ru subset AUC** | `metalcytotox_ru_auc.json` (0.712) | Krasnov 2026 GBDT Ru subset 5-fold CV (0.81) | Tier-1 comparator defined; we are below gate | **KEEP + REDESIGN protocol** (3-seed × 100-epoch tmQM-pretrained D-MPNN per `R17_AUC_RECOVERY_PLAN.md`) |
| 2 | **m2 Pt 4-cell AUC** | `platinai_4cell_auc.json` (0.553 mean; A2780 dropped per diagnosis) | Rusanov 2025 PlatinAI per-cell-line AUC | Tier-1 comparator exists; we are below + A2780 dropped = honest-negative structural | **KEEP + REDESIGN synthetic-label caveat** (PlatinAI labels synthesised per `build_platinai_4cell.py`, not real ChEMBL — must label PREDICTED) |
| 3 | **m3 Vina Dock median** | `vina_score.json` (-1.355 single-pocket 830c) + `vina_kcal_per_mol.json` (-3.103 cisplatin × 10 pockets) | AutoDock Vina 1.2.7 kcal/mol (Trott & Olson 2010) | Convention-aligned; **metal-coordination term absent** → systematic underestimate | **KEEP + REDESIGN interpretation** (label "metal generic atype" not "full Pt parameter set"; OK to ship with caveat) |
| 4 | **m4 Triple Success Rate** | `vina_triple_success.json` (0/100 on 830c) | TargetDiff Table 1 (Vina <-8.0 ∧ SA<4 ∧ QED>0.5) | Convention-aligned; pool lacks Zn-chelators → 0/100 is honest-negative, not bug | **KEEP** (honest-negative is the point) |
| 5 | **m5 SA score (Ertl)** | `sa_mean.json` (3.099 unconstrained; 2.373 closure-bounded 1h36) | Ertl & Schuffenhauer 2009, RDKit reference [1, 10] scale | Convention-aligned (RDKit reference unmodified); expose both raw + `sa_score_to_unit` | **KEEP** |
| 6 | **m6 QED (Bickerton)** | `qed.json` (0.6123 on metallo_drugs_500) | Bickerton 2012, RDKit `Chem.QED.qed` | Convention-aligned but **calibrated for oral drugs, not metallodrugs** | **KEEP + REDESIGN framing** (label "Bickerton desirability, oral-drug-calibrated, NOT metallodrug ADMET") |
| 7 | **m7 Diversity (Morgan-ECFP4)** | `diversity_tanimoto.json` (0.137 with all-4 fixes) | MOSES / TargetDiff IntDiv (1024-bit or 2048-bit ECFP4 + mean pairwise 1-T) | Convention aligned with TargetDiff (2048-bit); IntDiv field is least-standardised | **KEEP** (note 2048-vs-1024 ECFP4 cross-paper non-comparability) |
| 8 | **m8 PB-Valid rate** | `pb_pass_rate.json` (0.846 on 1-pocket click_tile) | PoseBusters (Buttenschoen 2024) — 26 checks (14 chemistry + 12 protein-aware) | `mol` mode shipped; **`dock` mode (steric clash) is unwired** | **KEEP + REDESIGN to dock mode** (install posebusters; recompute on all 5 Wf-30 pockets) |
| 9 | **m9 Interaction Parity** | NOT shipped | OMTRA 2025 SOTA2 leaderboard; Pocket2Mol/TargetDiff/DrugFlow per SOTA2 | SOTA2 leaderboard not used by any 2024-2026 Tier-1 publication; metric losing adoption | **DROP / DEFER** (no Tier-1 paper claims this; would be over-citation) |
| 10 | **m10 Vina Min Score** | `vina_min_score.json` (-4.158 on 10 organic drug-like pool) | AutoDock Vina 1.2.7 best pose (TargetDiff Table 1) | Convention-aligned; pool size 10 is small; Pt metallodrug term absent | **KEEP + REDESIGN cohort** (recompute on metallo_drugs_500 or Wf-30 cisplatin × 10 pocket cohort) |
| 11 | **m11 n_distinct metal-ligand pairs** | `n_distinct.json` (1 collapse → 20 with all-4 fixes) | Novel count metric (TODO-25 §3 gate ≥3) | Convention self-defined; **collapse to 1 is the diagnostic signal** | **KEEP** |
| 12 | **m12 Synthesis Rate (SyntheMol)** | NOT shipped as JSON | SyntheMol round-trip rate per Schwaller 2020 | Path A reaches 1.0 trivially; **no informative signal at 12-tile click-library scale** | **DROP / DEFER** (signal saturates; ship only as sanity check) |
| 13 | **m13 RDKit Validity rate** | `rdkit_validity.json` (1.000) | RDKit `Chem.MolFromSmiles` round-trip convention | Convention-aligned; 1.000 is at ceiling | **KEEP** (regression detector only) |
| 14 | **m14 Decode ratio (CFM)** | `decode_ratio.json` (1.000 post-fix) | RDKit-strict decode ratio (Path B convention) | Convention self-defined; 1.000 at ceiling | **KEEP** |
| 15 | **m15 Docking RMSD (redocking)** | `rigid_fragment_rmsd.json` (0.428 Å on 10 organic pool; **6.112 Å on 5-pocket cisplatin Wf-126 P2**) | TargetDiff / Pocket2Mol / DiffSBDD ligand-vs-crystal RMSD with Kabsch+Hungarian on heavy atoms, 2.0 Å success threshold (Trott & Olson 2010) | **CRITICAL REDESIGN: our rigid_fragment_rmsd is self-RMSD between two ETKDGv3+UFF conformers of the same molecule (per Wf-126 P1 honest finding); pocket is NOT in the calculation** | **REDESIGN (highest priority)** — need ligand-vs-pocket RMSD adapter post-Wf-134 |
| 16 | **m16 Affinity Pearson r** | `pearson_r_pic50.json` (0.207 best; seed-fragile) + `affinity_pearson_r.json` | Boltz-2 (Pearson r=0.86), FEP+ (~0.90) | Convention aligned for `pearson_r`; we measure on PlatinAI *synthetic* labels not experimental | **KEEP + REDESIGN protocol** (label PREDICTED; require real ChEMBL/PDBbind labelled set for paper headline) |
| 17 | **m17 OOD generalization** | `drugood_ood_auc.json` (mean 0.779; assay 0.891 / scaffold 0.697 / size 0.748) | DrugOOD leaderboard (Ji 2022) per-split AUC + iMoLD (Zhuang 2023) | Convention aligned; we already beat iMoLD on assay split | **KEEP** |
| 18 | **m18 Multimodal Protein-Metal AUC** | `metric18_protein_metal.json` (0.744) + `metric18_protein_metal_v2.json` (0.739) | "Multimodal Protein-Metal Binding" newx.sg 2025 (AUC>0.92 complex-level); Metal-3D / MIB2 (per-residue 0.80-0.90) | **Resolution mismatch: we use complex-level pooled; SOTA reports per-residue** | **REDESIGN** (add per-residue metric; report both) |
| 19 | **m19 LOMO transfer AUC** | `metal_lomo_auc.json` (0.535 ± 0.090; range 0.406-0.639) | Krasnov 2026 within-metal CV + our novel aggregate LOMO mean | Convention-self-defined (LOGO is sklearn standard, but multi-metal aggregation is novel); spread is the diagnostic | **KEEP** (novel metric; cite Krasnov per-metal; flag novelty of LOMO mean) |
| 20 | **m20 Tumor-specific Protein-Metal binding** | NOT shipped | "Multimodal Protein-Metal Binding" newx.sg 2025 (AUC>0.92 tumor-specific) | Out of ML scope for our CPU-only pipeline | **DROP / DEFER** (V5 scope; paper §6 future work) |
| 21 | **m21 NCI-60 9-cell mean AUC** | `nci60_precious_9cell_auc.json` (0.8007 ± 0.0160, 3 seeds × 50 ep) | Krasnov 2026 (Ru 0.81 single-metal); iACP-GCR 2024 (NCI-60 multitask, AUCROC>0.95) | Convention aligned (multitask per-cell AUC); our mean beats Tier-1 single-cell baseline | **KEEP** |
| 22 | **Ring-size distribution** | `ring_size_distribution.json` (per-molecule histogram, 449 mol) | TargetDiff Table 5 ring-size protocol | Convention aligned; structural complementary metric | **KEEP** |

---

## 2. Tier-1 alignment summary

- **15 KEEP** — definition + measurement already aligned with Tier-1 SOTA (TargetDiff / Pocket2Mol / DecompDiff / DiffSBDD / Krasnov 2026 / Rusanov 2025 / Ji 2022 / MOSES / GuacaMol / Ertl 2009 / Bickerton 2012 / Trott & Olson 2010).
- **5 REDESIGN** — definition partially misaligned (m15 self-RMSD vs. crystal-RMSD; m3/m10 metal-coordination term absent; m8 PB `mol` vs. `dock` mode; m18 resolution complex vs. per-residue) or measurement protocol gap (m16 synthetic vs. experimental labels; m2 synthetic-label caveat).
- **2 DROP/DEFER** — principled retirement: m9 (SOTA2 leaderboard not Tier-1) and m12 (signal saturates at our scale).
- **m20 defer** — explicitly V5 follow-up per `METRICS_TARGETS.md` §2 status.

## 3. Top-3 REDESIGN priorities

1. **m15 Docking RMSD → ligand-vs-pocket RMSD adapter (Wf-135 P1)**: highest priority. Without this, we have NO pocket-aware geometric RMSD metric. Wf-126 P1 confirmed the existing `rigid_fragment_rmsd` is *self*-RMSD; per-pocket values are equal by construction. Need an adapter that runs `RDKit Chem.rdMolAlign.GetBestRMS(lig, crystal_lig)` for each pocket's crystal ligand (1H36/3EML/3NY8/4AAW/4RLU have curated references on disk per `metrics/by_metric/vina_reference_*.json`).
2. **m3/m10 Vina → metal-chemotype-tuned scoring (Wf-135 P2)**: medium priority. Without chemotype-tuned Pt/Pd/Au parameters, Vina systematically underestimates metallodrug binding. Even a "metal generic atype" (M→Mg substitution per Wf-121) is a partial fix; full chemotype needs custom VDW radii. Paper §6 must state this gap explicitly.
3. **m8 PB-Valid → dock mode + per-pocket steric clash (Wf-135 P3)**: medium priority. PoseBusters `mol` mode is shipped but `dock` mode (the protein-aware 12 checks including steric clash) requires posebusters package + receptor PDB + docked PDBQT. Worth wiring in for paper §4 honest-negative coverage.

## 4. Migrations & closures (deferred to follow-up Wf)

- **m2 PlatinAI label honesty**: re-run with REAL PlatinAI xlsx (currently synth labels per `build_platinai_4cell.py`); 4,078 rows in dataset, ~30 of those are real ChEMBL-labelled. Until real labels are integrated, m2 must be reported as "PREDICTED on synthetic labels" not "MEASURED on real ChEMBL".
- **m6 QED metallo-aware variant**: add `qed_metalcorrected` (e.g. additional desirability penalty for Pt/Pd/Au/Ir given MW>500 QED penalty triggers). R17+ scope; not Wf-134 scope.
- **m7 Diversity per-residue sub-pocket variant**: `per_residue_diversity` is already shipped (DESIGN-only status in `diversity_subpocket.json`); full sweep on Wf-30 pockets is a low-cost follow-on.

## 5. Submission-venue cross-check

Per `synthesis_wetlab_paper_structure.md` §1:
- **NCS (most realistic Tier-1)**: requires matched-comparison audit + ≥2 tasks generalization + failure cases exposed. Our 18 KEEP+REDESIGN metrics can ship under NCS scope (CPU-only, no wet-lab claim, honest-negative framing). m15 REDESIGN is a *prerequisite* — without pocket-aware RMSD, paper §4 geometric table has no Tier-1-comparable column.
- **BiB (review/tutorial)**: easiest fit; the 5 REDESIGNs themselves are a publishable "metallodrug SBDD evaluation gap" contribution.
- **JCIM**: requires wet-lab validation for methods applied to single-target systems. We cannot meet this; skip unless doing review/tutorial.

## 6. Verdict source & references

- Wf-134 Phase 1 survey (`lit_table_RMSD_Vina_SA_QED.md` + `.json`) — covers m3, m4, m5, m6, m7, m10, m15 conventions.
- Wf-134 Phase 2 survey (`lit_table_AUC_OOD_generalization.md` + `.json`) — covers m1, m2, m17, m18, m19, m21 conventions.
- Wf-134 Phase 3 survey (`lit_table_wetlab_paper_structure.md` + `.json`) — covers Tier-1 venue wet-lab bar and Section structure requirements.
- Wf-126 P1 honest finding (`molmetal/reports/wf126_p1_staging/staging_status.json`) — diagnosis that `rigid_fragment_rmsd` is self-RMSD, NOT pocket-aware.
- Wf-87 P3 INDEX §17 audit (`molmetal/reports/wf87_index_17_real/verify_index_17.json`) — 25/25 expected metric JSONs present; 24/25 with measured values; 1 NO_VALUE (`external_scorers` summary, design only).
- `TODO/02_metrics/METRICS_TARGETS.md` — 21 conceptual metric definitions + tier targets (used as input to row definitions).
- `metrics/by_metric/` — 113 JSON files, 25 unique conceptual entries (after de-duplication of per-pocket and per-seed variants).

(End of WF-134 Phase 4 synthesis — companion to `TODO/pending/130_metric_definition_baseline.md`. Combined web search provenance across Phase 1+2+3 surveys: ≥20 tier-1 papers, ≥80 total citations, all WebFetch-verified 2026-09-27.)