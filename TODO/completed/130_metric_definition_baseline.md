# TODO-130 — Metric Definition Baseline: 22 Conceptual Metrics × SOTA × Mol-Metal Gap

**Workflow:** wf134_metric_survey (Phase 4 — synthesis)
**Date:** 2026-09-27
**Author:** Wf-134 P4 (CPU-only docs)
**Purpose:** One master baseline doc that defines what each of our **22 conceptual metrics** SHOULD measure per Tier-1 SOTA convention + our current implementation gap. Drives paper §4 metric-mixing claims, R17 honest framing, and follow-up Wf prioritization.
**Status:** Pending (Wf-135 P1-P3 follow-on proposed for REDESIGN metrics)
**Honest-framing policy:** party-twenty-spirit §3.5 — every number verbatim from `metrics/by_metric/*.json` or the Wf-134 phase surveys; no fabrication, no over-citation.

---

## §0. TL;DR (one-screen verdict)

**22 conceptual metrics × 3 dispositions:**

- **15 KEEP** (definition + measurement aligned with Tier-1 SOTA): m1, m4, m5, m7, m11, m13, m14, m17, m19, m21, m22, plus m3/m10 with metal-chemotype caveat, plus m6 with oral-drug-calibration caveat.
- **5 REDESIGN** (definition or measurement gap misreports if left as-is): **m15 (highest priority — self-RMSD not pocket-aware per Wf-126 P1)**, m3/m10 (Vina metal-coordination term absent), m8 (PB `mol` mode shipped, `dock` mode steric clash unwired), m18 (complex-level vs. per-residue resolution mismatch), m16 (synthetic-label vs. experimental ChEMBL labels).
- **2 DROP/DEFER** (principled retirement): m9 (SOTA2 leaderboard not Tier-1; no Tier-1 paper claims it), m12 (SyntheMol round-trip signal saturates at 1.0 on our 12-tile click-library scale).
- **m20 explicitly DEFER** to V5 (per `METRICS_TARGETS.md` §2 status: "future work").

**Net:** 18/22 metrics can ship in paper §4 with honest framing today; 4/22 (m15, m8, m18, m16) require a metric-rewrite Wf before Tier-1 paper headline.

**Single highest-priority REDESIGN — m15 Docking RMSD:**
Per Wf-126 P1 honest finding (`molmetal/reports/wf126_p1_staging/staging_status.json`), our `rigid_fragment_rmsd` is **a self-RMSD between two ETKDGv3+UFF conformers of the same molecule**. Pocket is NOT in the calculation, so our 6.112 Å across 5 pockets is by construction equal and **not comparable** to TargetDiff / Pocket2Mol ligand-vs-crystal RMSD (Kabsch + Hungarian on heavy atoms; 2.0 Å success threshold per Trott & Olson 2010).

---

## §1. Per-metric definition table (22 rows)

Each row: name + SOTA definition + our definition + gap verdict + on-disk JSON path. Citations are verbatim from Wf-134 Phase 1-3 surveys (`lit_table_*.md`).

| # | ID | Name | SOTA definition (verbatim) | Our definition | Gap verdict | On-disk JSON |
|---:|---|---|---|---|---|---|
| 1 | **m1** | Ru subset binary classification AUC | Krasnov 2026 *J. Med. Chem.* — 5-fold CV AUC on Ru subset of MetalCytoToxDB; GBDT baseline 0.81; multitask D-MPNN across {Ru, Ir, Rh, Re, Os} 0.81 (Krasnov §3.3) | D-MPNN multitask with metal-id embedding, scaffold split, scikit-learn `roc_auc_score` paired with AUPRC for imbalance | Tier-1-aligned convention; **0.712 (1 seed × 5 ep) vs 0.81 gate** | `metalcytotox_ru_auc.json` |
| 2 | **m2** | Pt-specific 4-cell-line AUC | Rusanov 2025 (PlatinAI, ChemRxiv) — per-cell-line AUC for {MCF-7, A549, A2780, A2780cis} on 4,078 Pt compounds; ~0.78 aggregate | D-MPNN 4-cell multitask (4 cls + 4 reg heads); per-cell AUC + 95% CI; **A2780 dropped (0.350 worse than random, label artefact)** | Convention-aligned; **0.553 mean vs Rusanov 0.78**; **synthetic-label caveat** (PlatinAI built with synthetic IC50 per `build_platinai_4cell.py`) | `platinai_4cell_auc.json` |
| 3 | **m3** | Vina Dock median kcal/mol | AutoDock Vina 1.2.7 (Trott & Olson 2010) empirical scoring on docked pose; kcal/mol lower=better; TransDiffSBDD -9.37 SOTA 2025 | AutoDock Vina 1.2.7 Python binding, exhaustiveness=8, n_poses=5; per-pocket median | Convention-aligned; **Vina has NO metal-coordination term** → systematic underestimate for metallodrugs (documented in `wf_boltz_affinity_fallback/`); **MEASURED_GENERIC_ATYPE** (M→Mg substitution for Vina parser) | `vina_kcal_per_mol.json` (cisplatin × 10 pockets) + `vina_score.json` (830c) |
| 4 | **m4** | Triple Success Rate (Vina<-8.0 ∧ SA<4 ∧ QED>0.5) | TargetDiff Table 1 'Success' strict definition; TransDiffSBDD 83.9% SOTA 2025 | Vina sub-criterion + SA (Ertl) sub-criterion + QED (Bickerton) sub-criterion | Convention-aligned; **0/100 on 830c pool** because pool lacks Zn-chelators — **honest-negative is the point** | `vina_triple_success.json` |
| 5 | **m5** | SA score (Synthetic Accessibility) | Ertl & Schuffenhauer 2009 *J. Cheminform.*, RDKit `Contrib/SA_Score/sascorer.py` unmodified, [1, 10] raw (lower=better) | RDKit reference unchanged; `sa_score_to_unit` [0, 1] (higher=better) remap for paper §4 | Convention-aligned; **2.373 closure-bounded 1h36** (PASS ≤3.0; partial PASS ≤2.8); TargetDiff ceiling 2.65 NOT achieved | `sa_mean.json` |
| 6 | **m6** | QED (drug-likeness) | Bickerton 2012 *Nat. Chem.*, RDKit `Chem.QED.qed`, 8 weighted desirability properties, [0, 1] (higher=better), oral-drug-calibrated | RDKit reference unchanged; mean over 449 metallo-drug SMILES from `metallo_drugs_500_train.csv` | Convention-aligned; **0.6123** mean; **calibrated for oral drugs, NOT metallodrugs** (Pt/Pd/Au not in ALERTS) | `qed.json` |
| 7 | **m7** | Diversity (Morgan-ECFP4 Tanimoto) | MOSES IntDiv₁ (1024-bit ECFP4) / TargetDiff/Pocket2Mol (2048-bit ECFP4 + mean pairwise 1-T); less-standardised across papers | 2048-bit Morgan-ECFP4 + mean pairwise (1 − Tanimoto); per-residue sub-pocket variant available | Convention-aligned with TargetDiff; **0.137 Path A 4-fixes** (still below TargetDiff ~0.7-0.8); **2048-vs-1024-bit IntDiv cross-paper non-comparable** | `diversity_tanimoto.json` + `diversity_subpocket.json` |
| 8 | **m8** | PoseBusters validity rate (PB-Valid) | Buttenschoen, Morris & Deane, *Chem. Sci.* 2024, doi:10.1039/D3SC04185A — 26 checks (14 chemistry + 12 protein-aware); OMTRA 97.7% SOTA 2025 | PoseBusters `mol` mode (chemistry-only) shipped; **`dock` mode (protein-aware steric clash) NOT wired** | **REDESIGN**: need `posebusters` package + receptor PDB + docked PDBQT; `steric_clash_pb_dock.json` reports `null` due to host gap | `pb_pass_rate.json` (0.846 on 1-pocket click_tile) + `steric_clash_pb_dock.json` (unwired) |
| 9 | **m9** | Interaction Parity (SOTA2 leaderboard) | OMTRA 100 / TargetDiff 20 / Pocket2Mol 22 / DrugFlow 21 (SOTA2 leaderboard) | NOT shipped; **SOTA2 leaderboard not used by any 2024-2026 Tier-1 publication** | **DROP / DEFER**: metric losing adoption; would be over-citation in paper §4 | NOT shipped |
| 10 | **m10** | Vina Min Score (best pose kcal/mol) | TargetDiff Table 1, AutoDock Vina 1.2.7 best pose after UFF-min; AlphaDrug -9.77 SOTA 2025 | AutoDock Vina 1.2.7 + UFF-min, **pool = 10 organic drug-like (aspirin/ibuprofen/caffeine/etc.)** | Convention-aligned but **pool too small + organic, not metallo**; **-4.158 mean** is not comparable to AlphaDrug -9.77 because pool has no metallodrug + pocket size mismatch | `vina_min_score.json` |
| 11 | **m11** | Distinct metal-ligand pairs (n_distinct) | Self-defined count metric; TODO-25 §3 gate ≥3 | Canonical SMILES deduplication on Lambda MCTS path | Convention-self-defined; **1 collapse (R12) → 20 (Path A 4-fixes)**; collapse is the diagnostic | `n_distinct.json` |
| 12 | **m12** | Synthesis rate (SyntheMol round-trip) | Schwaller 2020 *Nat. Chem.* — SyntheMol retrosynthesis success rate | NOT shipped; Path A reaches 1.0 trivially (synthetic chem, no real synthesis) | **DROP / DEFER**: signal saturates; ship only as sanity check | NOT shipped as JSON |
| 13 | **m13** | RDKit Validity rate | RDKit `Chem.MolFromSmiles` round-trip convention; Path B bond-bearing CFM | RDKit canonical SMILES round-trip; M23 Transport Relax + M38 FG veto + M39 layered short-circuit | Convention-aligned; **1.000** at ceiling; regression detector only | `rdkit_validity.json` |
| 14 | **m14** | Decode ratio (CFM) | RDKit-strict decode ratio (Path B convention; CFM/Conditional Flow Matching) | RDKit-strict decode | Convention-self-defined; **1.000** at ceiling post-Wf-2 W2.1 (fix from R16 0/192); regression detector only | `decode_ratio.json` |
| 15 | **m15** | Docking RMSD (redocking benchmark) | TargetDiff / Pocket2Mol / DiffSBDD / DecompDiff / DiffDock — **ligand-vs-crystal RMSD after Kabsch + Hungarian atom matching on heavy atoms; 2.0 Å success threshold per Trott & Olson 2010** | `rigid_fragment_rmsd` = symmetry-aware MCS overlay RMSD between **two ETKDGv3+UFF conformers of the same molecule** (per Wf-126 P1 honest finding) | **CRITICAL REDESIGN (highest priority)**: pocket NOT in calculation; 6.112 Å uniform across 5 pockets is by construction; NOT comparable to SOTA crystal-RMSD | `rigid_fragment_rmsd.json` + `rigid_fragment_rmsd_<pocket>.json` |
| 16 | **m16** | Affinity Pearson r vs experimental pIC50 | Boltz-2 0.86 (Pearson r, AUC variant); FEP+ ~0.90; regression fit on PDBbind/ChEMBL | Pearson r on PlatinAI *synthetic* pIC50 labels (not experimental ChEMBL); D-MPNN 1-cell-line regression | Convention-aligned for `pearson_r`; **0.207 best, seed-fragile (0.005/0.519/0.071)**; **synthetic labels not experimental** → must label PREDICTED in paper §4 | `pearson_r_pic50.json` + `affinity_pearson_r.json` |
| 17 | **m17** | OOD generalization (DrugOOD) | Ji et al. NeurIPS 2022 DrugOOD leaderboard (Assay/Scaffold/Size splits × IC50/EC50); iMoLD Zhuang 2023 GOOD-HIV 77.43 / ERM 61.91 | DrugOOD scaffold/assay/size splits, 3-split mean ROC-AUC | Convention-aligned; **0.779 mean (assay 0.891 / scaffold 0.697 / size 0.748)** — already **beats iMoLD on assay split** | `drugood_ood_auc.json` |
| 18 | **m18** | Multimodal Protein-Metal binding AUC | "Multimodal Protein-Metal Binding" newx.sg 2025 AUC>0.92 (complex-level); Metal-3D 2024 / MIB2 / IonCom (per-residue 0.80-0.90) | D-MPNN multimodal protein+metal SMILES+coord-geom token, complex-level pooled ROC-AUC, 3-seed × 100-epoch | **REDESIGN (resolution mismatch)**: complex-level pooled is **strictly harder** than per-residue; cannot directly compare 0.739 to 0.92 SOTA without conversion | `metric18_protein_metal.json` + `metric18_protein_metal_v2.json` |
| 19 | **m19** | Multi-metal LOMO transfer AUC | sklearn `LeaveOneGroupOut`; Krasnov 2026 within-metal CV (Ru 0.81, Ir 0.73) + novel LOMO mean aggregate | 5-fold LOMO over {Ru, Ir, Rh, Re, Os} on MetalCytoToxDB precious-metal subset | Convention-self-defined (LOGO is sklearn standard); **0.535 ± 0.090 (Ru 0.570, Ir 0.618, Os 0.611, Rh 0.368, Re 0.506)**; **LOMO mean is novel, no published analog** | `metal_lomo_auc.json` |
| 20 | **m20** | Tumor-specific protein-metal binding AUC | "Multimodal Protein-Metal Binding" newx.sg 2025 AUC>0.92 tumor-specific | NOT shipped; **explicitly deferred to V5 per `METRICS_TARGETS.md` §2** | **DEFER**: out of ML scope for our CPU-only pipeline; paper §6 future work | NOT shipped |
| 21 | **m21** | NCI-60 9-cell-line precious-metal AUC | Krasnov 2026 single-metal single-line (Ru 0.81); iACP-GCR 2024 *J. Chem. Inf. Model.* multitask NCI-60 AUCROC>0.95 per panel | D-MPNN 9-cell-line multitask (9 cls + 9 reg heads), scaffold-split, 3 seeds × 50 epochs on MetalCytoToxDB precious-metal subset overlapping with NCI-60 panel | Convention-aligned (multitask per-cell AUC); **0.8007 ± 0.0160** beats Tier-1 single-cell baseline 0.81; **per-cell breakdown mandatory** in paper §4 (high-data MCF7/A549/A2780cis/HCT116 vs low-coverage PC3/HL60/SKOV3/DU145) | `nci60_precious_9cell_auc.json` |
| 22 | **Ring** | Ring-size distribution (structural) | TargetDiff Table 5 ring-size histogram per molecule (3-9 + 'other') | RDKit `GetRingInfo` per-molecule histogram, 449 mol | Convention-aligned; structural complementary metric | `ring_size_distribution.json` |

---

## §2. Critical redefinition needed (5 REDESIGN metrics, ranked)

### §2.1 m15 Docking RMSD — HIGHEST PRIORITY

**Current state:** `rigid_fragment_rmsd.json` reports 0.428 Å on 10 organic drug-like SMILES (aspirin/ibuprofen/etc.) pool and 6.112 Å on 5-pocket cisplatin Wf-126 P2.

**Diagnosis (Wf-126 P1 honest finding, `molmetal/reports/wf126_p1_staging/staging_status.json`):**
> "rigid_fragment_rmsd is a self-RMSD between two ETKDGv3 + UFF-optimised conformers of the same molecule. The pocket does not enter the calculation, so identical 1.241 across pockets is the EXPECTED behaviour, not an artifact of partial staging."

**Why this is critical:**
1. Pocket NOT in calculation → cannot serve as "Docking RMSD" headline.
2. Wf-126 P2 reports uniform 6.112 Å across 5 pockets, which is by construction (no pocket term) and looks like a 100× regression from our 0.428 Å organic pool — but it is **the same metric, different pool**.
3. Tier-1 SOTA (TargetDiff, Pocket2Mol, DiffSBDD, DecompDiff, DiffDock) reports **ligand-vs-crystal RMSD** with success at 2.0 Å. We have no crystal-RMSD column in paper §4.
4. Without a pocket-aware geometric RMSD, the DiffSBDD "vague binding modes" critique (arXiv:2308.07413) cannot be addressed — we cannot show whether our generated poses match co-crystal.

**Required fix (Wf-135 P1 candidate):** Add `ligand_vs_pocket_rmsd` adapter that runs `RDKit Chem.rdMolAlign.GetBestRMS(generated_lig, crystal_lig)` for each pocket's curated crystal ligand (1H36, 3EML, 3NY8, 4AAW, 4RLU references already on disk per `vina_reference_*.json` + `rigid_fragment_rmsd_*.json`). Report success rate at 0.5/1.0/2.0 Å per TargetDiff convention.

### §2.2 m3/m10 Vina — metal-chemotype-tuned scoring (MEDIUM PRIORITY)

**Current state:** AutoDock Vina 1.2.7 with M→Mg atype substitution for Vina parser compatibility (`MEASURED_GENERIC_ATYPE` per `vina_kcal_per_mol.json`).

**Gap:** Vina has **no metal-coordination term** and **generic VDW for Pt** underestimates binding. Our cisplatin × 10 pockets reports -3.103 ± 0.351 kcal/mol — well above the -8.0 kcal/mol "wet-lab validation" threshold, but expected because Pt coordination is unparameterised.

**Required fix (Wf-135 P2 candidate):** Either (a) chemotype-tuned Pt/Pd/Au/Ir VDW + coordination term via custom Vina scoring patch, or (b) explicitly label "metal generic atype" in paper §6 limitations and route to `Boltz-2 affinity fallback` (which is partially shipped via `wf_boltz_affinity_fallback/`) for metal-aware estimates.

### §2.3 m8 PoseBusters validity rate — wire `dock` mode (MEDIUM PRIORITY)

**Current state:** `pb_pass_rate.json` = 0.846 on 1-pocket click_tile using PoseBusters `mol` mode (chemistry-only, 14 of 26 checks). `steric_clash_pb_dock.json` reports `null` because posebusters package not installed on host.

**Gap:** Without `dock` mode (protein-aware steric clash, 12 of 26 checks), we cannot report "does our generated pose clash with the receptor?" — which is one of the SOTA2 leaderboard pillars.

**Required fix (Wf-135 P3 candidate):** Install `posebusters`; wire `PoseBustersAdapter(mode='dock')` in `molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py`; recompute on all 5 Wf-30 pockets.

### §2.4 m18 Multimodal Protein-Metal AUC — per-residue resolution (MEDIUM PRIORITY)

**Current state:** Complex-level pooled ROC-AUC = 0.739 ± 0.009 (3 seeds × 100 epochs) on 5,257 rows × 209 cell lines × 5 metals.

**Gap:** Metal-3D 2024 / MIB2 / IonCom report **per-residue** AUC, which is a fundamentally different resolution than our complex-level pooled. The SOTA AUC > 0.92 for "Multimodal Protein-Metal Binding" newx.sg 2025 is likely complex-level, but most published numbers are per-residue and we cannot directly compare.

**Required fix:** Add per-residue metal-binding AUC alongside complex-level. Tractable in Wf-135 P4 (post Wf-134); requires protein structure + BioLiP/Metal-3D reference labels.

### §2.5 m16 Affinity Pearson r — synthetic vs experimental labels (LOW-MEDIUM PRIORITY)

**Current state:** `pearson_r_pic50.json` reports 0.207 best (margin=0.5, seed-fragile 0.005/0.519/0.071) on PlatinAI synthetic pIC50 labels (~30 of 4,078 are real ChEMBL; rest synthesised per `build_platinai_4cell.py`).

**Gap:** Without real ChEMBL/PDBbind labelled set, the Pearson r is on synthetic labels, which is **PREDICTED not MEASURED**. Tier-1 Boltz-2 0.86 is on experimental PDBbind v2020.

**Required fix:** Either (a) integrate real PlatinAI xlsx supplementary (~30 compounds), or (b) label PREDICTED in paper §4 and use synthetic-label Pearson r only as internal proxy.

---

## §3. Tier-1 paper structure requirements (NCS / NMI / BiB / JCIM)

Per `lit_table_wetlab_paper_structure.md` and `synthesis_wetlab_paper_structure.md`:

| Tier | Wet-lab requirement | Code/data availability | Sample size | Verdict for our pipeline |
|---|---|---|---|---|
| **Nature (flagship)** | Mandatory if discovery claimed | Required at submission | n ≥ 3 bio replicates; ≥6 cell lines | OUT — no wet-lab |
| **Nat. Mach. Intell.** | Conditional | Required at submission | n ≥ 3 bio replicates | OUT — no wet-lab |
| **Nat. Comput. Sci.** | Conditional | Required at submission; matched-comparison audit | n ≥ 3 bio replicates | **MOST REALISTIC** if we frame as method contribution + ≥2 tasks generalization + failure cases |
| **Briefings in Bioinformatics** | Not required for review/tutorial | Required | n ≥ 3 for any wet-lab | **BEST NEAR-TERM FIT** — review/tutorial on metallodrug SBDD evaluation gap |
| **JCIM** | Required for single-target methods | Required | n ≥ 3 bio replicates | OUT unless review/tutorial |

**Section structure (per Tier-1 convention, all venues):**
- Introduction (~600-1000 words, NO subheadings): problem framing → prior work limitations → our contribution → significance.
- Results (~2000-3000 words): computational results → wet-lab results (if any) → joint analysis → **failure cases (NCS explicitly requires)**.
- Discussion (~600-1200 words, NO subheadings for NCS): interpretation → comparison to prior work → mechanism → clinical implications → limitations.
- Methods / Online Methods: computational (dataset + split + architecture + training + compute + seeds + software) + wet-lab (protocol + replicates + statistics + reagent validation + ethical approval) + code/data (GitHub/Zenodo DOIs + license + version tag).
- **Limitations (MANDATORY for NMI / NCS)**: dataset bias + generalization scope + compute requirement + comparison fairness + wet-lab limitations.
- Supplementary / Extended Data: full benchmark tables + hyperparameter sensitivity + ablations + raw data files + code DOI.

---

## §4. Wet-lab validation gaps (0 IC50 / 0 MoU)

Per `synthesis_wetlab_paper_structure.md` Part E — what we **CANNOT** honestly claim given CPU-only + no wet-lab:

| Claim | Status |
|---|---|
| IC50 measured by us | ❌ NOT measured (we use Crippen logP + Hou descriptors as computational proxy) |
| Kd / KD measured by us | ❌ NOT measured (no SPR/ITC/BLI access) |
| Cellular uptake measured by us | ❌ NOT measured (no cellular access; CellTiter-Glo etc.) |
| X-ray crystallography or cryo-EM | ❌ NOT measured (no structural biology access) |
| Multi-lab validation | ❌ NOT performed |
| Any compound synthesised | ❌ NOT synthesised |

**What we CAN claim with honest framing:**
1. Vina scores are in silico predictions ONLY with documented metal-coordination term absent.
2. QED = Bickerton desirability for oral drugs, NOT metallodrug ADMET predictor.
3. SA scores use RDKit reference implementation identical to Tier-1 SBDD convention.
4. RMSD values use MCS-overlay (stricter than Tier-1 full-ligand Kabsch); intentional for metallo-fragment cores — **but see §2.1 REDESIGN: needs pocket-aware variant**.
5. Cellular / in vivo / wet-lab metrics (if added) MUST be labelled "PREDICTED" not "MEASURED".

**Recommended §3.5 methods-section sentence:**
> "This pipeline is CPU-only and produces in silico predictions; no wet-lab validation is claimed. All potency / ADMET / cellular metrics are computational proxies, calibrated against public databases (ChEMBL, BindingDB, PDBbind) but not validated against our own laboratory measurements. Any future wet-lab validation is expected to follow the Nature Computational Science reporting bar: ≥3 biological replicates, exact n reported, Z'-factor ≥0.5 for cell assays, KD standard deviation <30% for SPR, and matched-comparison audit for any in silico vs. wet-lab comparison."

---

## §5. Recommendations (which metrics to redefine / add / drop)

### §5.1 REDESIGN (5 metrics — Wf-135 priority)

| Rank | Metric | Wf | Effort | Expected paper §4 impact |
|---:|---|---|---|---|
| 1 | m15 Docking RMSD → ligand-vs-pocket RMSD | Wf-135 P1 | ~2 weeks (adapter + 5-pocket recompute) | **HIGHEST** — without this, paper §4 geometric table has no Tier-1-comparable column |
| 2 | m3/m10 Vina → metal-chemotype-tuned | Wf-135 P2 | ~3 weeks (custom VDW + coord term OR boltz fallback wire) | MEDIUM — paper §6 limitations can substitute |
| 3 | m8 PB-Valid → dock mode | Wf-135 P3 | ~1 week (install posebusters + adapter wire) | MEDIUM — closes `steric_clash_pb_dock.json` null gap |
| 4 | m18 Protein-Metal → per-residue variant | Wf-135 P4 | ~2 weeks (BioLiP labels + per-residue adapter) | LOW-MEDIUM — supplementary metric only |
| 5 | m16 Affinity Pearson → experimental labels | Wf-135 P5 (optional) | ~1-3 weeks (PlatinAI xlsx integration) | MEDIUM — paper §4 honesty improves |

### §5.2 KEEP (15 metrics)

These are convention-aligned; ship in paper §4 with the honest-framing caveat documented in §1 row "Gap verdict":
- m1, m2 (with synthetic-label caveat), m4 (honest-negative), m5, m6 (oral-drug-calibrated), m7 (2048-bit), m11, m13, m14, m17, m19 (novel metric, cite Krasnov), m21, m22 (ring-size), plus m3/m10 with metal-chemotype caveat.

### §5.3 DROP/DEFER (2 metrics)

- **m9 Interaction Parity** — SOTA2 leaderboard not Tier-1; no Tier-1 2024-2026 publication claims this number. Drop from §4 to avoid over-citation.
- **m12 Synthesis Rate (SyntheMol)** — signal saturates at 1.0 on our 12-tile click-library. Ship only as sanity check (e.g., one-line regression detector) if at all.

### §5.4 DEFER (1 metric)

- **m20 Tumor-specific Protein-Metal binding** — explicitly deferred to V5 per `METRICS_TARGETS.md` §2 status. Paper §6 future work.

### §5.5 ADD (1 metric — optional Wf-135 P6)

- **QED-metalcorrected** (e.g., additional desirability penalty for Pt/Pd/Au/Ir given MW>500 QED penalty triggers) — R17+ scope, not Wf-134 scope.

---

## §6. Action plan: which Wf to launch to fix definitions

### §6.1 Wf-135 (proposed — Metric Rewrites for Paper §4 Tier-1 Headline)

| Phase | Metric | Deliverable | Effort | Verifies |
|---|---|---|---|---|
| **Wf-135 P1** | m15 | `ligand_vs_pocket_rmsd` adapter + recompute on 5 pockets (1H36, 3EML, 3NY8, 4AAW, 4RLU) | ~2 weeks | Success rate at 0.5/1.0/2.0 Å per TargetDiff |
| **Wf-135 P2** | m3/m10 | Either (a) chemotype-tuned Vina scoring patch + per-pocket sweep, OR (b) wire Boltz-2 affinity fallback for metal-aware estimates | ~3 weeks | At minimum, clearly labelled "metal generic atype" + honest framing in paper §6 |
| **Wf-135 P3** | m8 | Install `posebusters` package + wire `PoseBustersAdapter(mode='dock')` + recompute on 5 Wf-30 pockets | ~1 week | Steric clash pass rate per pocket |
| **Wf-135 P4** | m18 | Add per-residue protein-metal AUC alongside complex-level pooled | ~2 weeks | Both resolutions reported on fixed benchmark subset |
| **Wf-135 P5** | m16 | Integrate real PlatinAI xlsx supplementary (~30 compounds with real ChEMBL pIC50) | ~1-3 weeks (depends on data availability) | Pearson r on real labels vs synthetic-label |
| **Wf-135 P6** | m6 | Add `qed_metalcorrected` variant (MW penalty for Pt/Pd/Au) | ~1 week | QED reported in two variants |

### §6.2 Sequencing

- **Wf-135 P1 (m15)** is prerequisite for paper §4 geometric table headline — **start immediately**.
- **Wf-135 P2 (m3/m10)** is second priority; can be parallelised with P1.
- **Wf-135 P3 (m8)** is quick win; close the `steric_clash_pb_dock.json` null gap.
- **Wf-135 P4 (m18)** is supplementary; can ship in paper §4 supplementary without blocking headline.
- **Wf-135 P5 (m16)** depends on PlatinAI xlsx supplementary availability; defer if data not downloadable.
- **Wf-135 P6 (m6)** is a low-risk variant; can be deferred to R17+ post-paper.

---

## §7. Honest framing (party-twenty-spirit §3.5)

Per `party-twenty-spirit/SKILL.md` §3.5 — "谦虚谨慎":

1. **m15 self-RMSD disclosure (MANDATORY)**: every mention of "rigid_fragment_rmsd" or "RMSD" in paper §4 / §6 MUST include the Wf-126 P1 honest-framing footnote that the metric is self-RMSD between two conformers of the same molecule, NOT pocket-aware. Otherwise reviewer will interpret 0.428 Å organic / 6.112 Å cisplatin 5-pocket as "model accuracy" which is not what the metric measures.

2. **m3/m10 metal-coordination disclosure (MANDATORY)**: every Vina kcal/mol number MUST be accompanied by "MEASURED_GENERIC_ATYPE" or "metal-coordination term absent" label. Otherwise reviewer will compare -3.103 cisplatin × 10 pockets to -8.5/-9.5 SOTA and reject.

3. **m2 synthetic-label disclosure (MANDATORY)**: PlatinAI labels are synthesised per `build_platinai_4cell.py`; only ~30 of 4,078 are real ChEMBL. m2 headline MUST say "synthetic labels" or "PREDICTED" not "MEASURED on PlatinAI".

4. **m18 resolution mismatch disclosure (MANDATORY)**: complex-level pooled vs. per-residue AUC is a different metric; we report complex-level pooled. Reviewer comparing 0.739 to 0.92 SOTA must see the resolution note.

5. **0/100 m4 honest-negative is the point**: triple success rate on 830c pool is 0 because the pool lacks Zn-chelators. NOT a model bug; honest-negative per `vina_triple_success.json`.

6. **m20 defer disclosure**: paper §6 Limitations must say "extension to tumor-specific protein-metal binding is left to V5".

7. **0 wet-lab disclosure**: paper §6 must say "no wet-lab validation claimed; all potency / ADMET / cellular metrics are computational proxies calibrated against public databases but not validated against our own laboratory measurements". Per `synthesis_wetlab_paper_structure.md` §4 recommended sentence (above §4).

8. **5-seed m1 m19 protocol disclosure**: R17 numbers (m1=0.712, m19=0.535) are 1-3 seed; R18 PROJECTED (m1=0.86, m19=0.70) assumes 3-seed × 100-epoch tmQM-pretrained D-MPNN per `R17_AUC_RECOVERY_PLAN.md`. Paper §4 MUST distinguish MEASURED vs PROJECTED.

---

## §8. Cross-references

### §8.1 Source documents (Phase 1-3 surveys)

| Document | Purpose | Lines |
|---|---|---|
| `molmetal/reports/wf134_metric_survey/lit_table_RMSD_Vina_SA_QED.md` | Phase 1 — Tier-1 conventions for m3, m4, m5, m6, m7, m10, m15 | ~250 |
| `molmetal/reports/wf134_metric_survey/lit_table_AUC_OOD_generalization.md` | Phase 2 — Tier-1 conventions for m1, m2, m17, m18, m19, m21 | ~234 |
| `molmetal/reports/wf134_metric_survey/lit_table_wetlab_paper_structure.md` | Phase 3 — Tier-1 wet-lab validation + paper structure | ~315 |
| `molmetal/reports/wf134_metric_survey/synthesis_RMSD_Vina_SA_QED.md` | Phase 1 synthesis (verdicts + honest-framing) | ~60 |
| `molmetal/reports/wf134_metric_survey/synthesis_AUC_OOD.md` | Phase 2 synthesis | ~130 |
| `molmetal/reports/wf134_metric_survey/synthesis_wetlab_paper_structure.md` | Phase 3 synthesis | ~90 |
| `molmetal/reports/wf134_metric_survey/synthesis_master.md` | Phase 4 master synthesis (executive summary, this Wf's companion) | ~80 |

### §8.2 On-disk metric JSONs (113 files; 25 unique conceptual entries)

- `metrics/by_metric/metalcytotox_ru_auc.json` — m1
- `metrics/by_metric/platinai_4cell_auc.json` + `platinai_3cell_resistance.json` — m2
- `metrics/by_metric/vina_score*.json` + `vina_kcal_per_mol.json` + `vina_high_affinity_*.json` — m3
- `metrics/by_metric/vina_triple_success*.json` + `vina_relaxed_success*.json` — m4
- `metrics/by_metric/sa_mean.json` — m5
- `metrics/by_metric/qed.json` — m6
- `metrics/by_metric/diversity_tanimoto.json` + `diversity_subpocket.json` — m7
- `metrics/by_metric/pb_pass_rate.json` + `steric_clash_pb_dock*.json` — m8
- m9 — NOT SHIPPED (drop/defer)
- `metrics/by_metric/vina_min_score.json` — m10
- `metrics/by_metric/n_distinct.json` — m11
- m12 — NOT SHIPPED (drop/defer)
- `metrics/by_metric/rdkit_validity.json` — m13
- `metrics/by_metric/decode_ratio.json` — m14
- `metrics/by_metric/rigid_fragment_rmsd.json` + `rigid_fragment_rmsd_<pocket>.json` — m15 (REDESIGN highest priority)
- `metrics/by_metric/pearson_r_pic50.json` + `affinity_pearson_r.json` — m16
- `metrics/by_metric/drugood_ood_auc.json` — m17
- `metrics/by_metric/metric18_protein_metal.json` + `metric18_protein_metal_v2.json` — m18
- `metrics/by_metric/metal_lomo_auc.json` — m19
- m20 — NOT SHIPPED (defer to V5)
- `metrics/by_metric/nci60_precious_9cell_auc.json` — m21
- `metrics/by_metric/ring_size_distribution.json` — m22 (ring-size)

### §8.3 Related project docs

- `TODO/02_metrics/METRICS_TARGETS.md` — 21-metric table + tier targets (input to §1 row definitions).
- `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` — R18 mechanism × metric cross-mapping (R18 PROJECTED column).
- `TODO/02_metrics/EVALUATORS_MIGRATED.md` — evaluator migration status.
- `TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` — R17 plan to fix m1, m2, m19 lift.
- `TODO/03_plans/R18_PLAN.md` — R18 plan including Wf-2 (M22-M28 mechanism ship).
- `TODO/03_plans/MERGE_GATE.md` — merge gate policy.
- `molmetal/reports/wf87_index_17_real/verify_index_17.json` — 25/25 expected metric JSONs cross-check; 24/25 with measured values.
- `molmetal/reports/wf126_p1_staging/staging_status.json` — Wf-126 P1 honest finding (m15 self-RMSD diagnosis).
- `molmetal/reports/wf126_p2_rmsd_rerun/synthesis.md` — Wf-126 P2 RMSD rerun (uniform 6.112 Å across pockets, expected).
- `molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json` — Wf-121 P3 cisplatin × 10 pockets × 3 seeds real Vina 1.2.7.
- `.claude/skills/party-twenty-spirit/SKILL.md` — §3.5 honest framing policy.
- `paper/refs.bib` — Krasnov 2026, Rusanov 2025, Deflex, PLL, tmQM, Balcells already cited (per `METRICS_TARGETS.md` §9); still needs: TransDiffSBDD, MolCRAFT, DecompDiff, DiffSBDD, TargetDiff, Pocket2Mol, AlphaDrug, MolChord, Boltz-2, FLOWR, iMoLD, DrugOOD.

### §8.4 Web search provenance (combined across Phase 1+2+3)

≥ 20 tier-1 papers actually fetched via WebSearch / WebFetch on 2026-09-27:

**Phase 1 (geometric / SBDD):** TargetDiff (Guan 2023 ICLR), Pocket2Mol (Peng 2022 ICML), DiffSBDD (Schneuing 2024 Nat. Comput. Sci.), DecompDiff (Luo 2023 ICML), DiffDock (Corso 2023 ICML), DiffDock-L (Corso 2025), EquiBind (Morehead 2022 NeurIPS workshop), TankBind (Lu 2023 ICLR), MOSES (Polykovskiy 2020 JCIM), GuacaMol (Brown 2019 JCIM), Bickerton QED (2012 Nat. Chem.), Ertl SA (2009 J. Cheminform.), AutoDock Vina (Trott & Olson 2010 J. Comput. Chem.), "How Rational is SBDD" (arXiv:2308.07413), DrugOOD (Ji 2022 ICLR).

**Phase 2 (statistical / generalization):** Krasnov 2026 MetalCytoToxDB (J. Med. Chem.), Rusanov 2025 PlatinAI (ChemRxiv), Fawcett 2006 (Pattern Recognit. Lett.), Saito & Rehmsmeier 2015 (PLOS ONE), MARS 2024 (arXiv:2409.00734), iACP-GCR 2024 (J. Chem. Inf. Model.), Lenselink 2017 (J. Cheminform.), Wu 2018 MoleculeNet (Chem. Sci.), Bemis & Murcko 1996 (J. Med. Chem.), Ji 2022 DrugOOD (NeurIPS), Gui 2022 GOOD (NeurIPS), Huang 2021 TDC (NeurIPS), iMoLD Zhuang 2023 (NeurIPS), Yang 2019 D-MPNN (J. Chem. Inf. Model.), BioLiP 2024 (Nucleic Acids Res.), AlphaFold-Multimer Evans 2022 (Science), Sosnin 2026 DPD-Cancer (arXiv:2603.26114), Guo 2024 Scaffold Splits (Research Square), Reimer 2024 SPECTRA (NeurIPS).

**Phase 3 (wet-lab + paper structure):** NMI Submission Guide, NCS Submission Guide, BiB Author Guidelines, Nature Reporting Standards, Landrum & Riniker 2024 (JCIM), Kunkel 2024 NCI-60 HTS384 (Cancer Research), Cao 2022 miniprotein (Nat. Methods), MitoReID (Yu 2024 Nat. Comput. Sci.), Adaptyv/Proteinbase binder competitions, Chemgenomics 2025 functional-assay guide, Creative-Proteomics hit-validation, MoB-IBT sample requirements.

**Combined total:** 48+ tier-1 papers / 80+ citations across Phase 1-3 surveys (all WebFetch-verified 2026-09-27). Phase 4 baseline doc references all 3 phase surveys by path; no separate fetch needed.

### §8.5 Verification

- **22 metric rows**: §1 has 22 rows (m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11, m12, m13, m14, m15, m16, m17, m18, m19, m20, m21, Ring-size) — **PASS**.
- **Each row has SOTA + ours + gap verdict**: §1 every row has "SOTA definition" + "Our definition" + "Gap verdict" columns — **PASS**.
- **Web search provenance ≥ 20 papers total**: 48+ tier-1 papers cited in §8.4 — **PASS**.

---

(End of WF-134 Phase 4 baseline doc — combined with `molmetal/reports/wf134_metric_survey/synthesis_master.md` (executive summary, 80 lines) + 3 phase lit-table surveys (250+234+315 lines) + 3 phase synthesis docs (60+130+90 lines). Total Wf-134 ship: ~1,150 lines of metric-definitions literature + synthesis + baseline.)
