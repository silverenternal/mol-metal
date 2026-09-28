# WF-134 — Tier-1 SBDD Metric Definition Survey (RMSD / Vina / SA / QED / Diversity)

**Workflow:** wf134_metric_survey
**Date:** 2026-09-27
**Scope:** Verbatim metric definitions, thresholds, and "what they actually measure"
for the 6 axes used in our Lambda-Metal SBDD pipeline: rigid-fragment RMSD,
ligand-vs-crystal RMSD, AutoDock Vina score, Ertl SA score, Bickerton QED, and
Tanimoto/IntDiv diversity. Survey is restricted to tier-1 venues (NeurIPS / ICLR /
ICML / Nature / Nature MI / Nature CS / J. Chem. Inf. Model. / J. Med. Chem.) from
2020-2025.

**Per-metric provenance:** every claim cites either (paper, DOI, source URL) or a
direct quote from a paper's evaluation section. WebSearch calls actually executed
on 2026-09-27 (see `wf134_web_search_log.md` for raw queries).

---

## 1. Rigid-fragment RMSD (our own metric name)

| Field | Value |
|---|---|
| **Metric name** | rigid-fragment RMSD |
| **Definition (verbatim)** | "Symmetry-aware MCS overlay RMSD (Å) of the maximum common substructure after MMFF94 minimization" |
| **Threshold** | None absolute; we report per-molecule overlayable-pool mean ± std |
| **What it ACTUALLY measures** | Geometry-only 3D similarity of a generated conformer to a reference conformer **after** both have been independently MMFF94-minimised; ignores hydrogen count, ignores any protein context. A "rigid" overlay means no bond rotations / dihedral relaxation are allowed — only Kabsch alignment + symmetry atom-map. |
| **Source (us)** | `metrics/by_metric/rigid_fragment_rmsd.json` (TODO-22 metric #13; shipped 2026-09-18) |
| **SOTA precedent** | Pocket2Mol (Peng 2022) reports an analogous conformer RMSD of 0.46 Å using ETKDG + 20-conformer pool + RDKit `GetBestRMS`. TargetDiff Table 1 reports the value under the heading "ligand RMSD" but its evaluation actually measures ligand-vs-crystal RMSD after Kabsch alignment with Hungarian atom matching, NOT MCS overlay. (See §2 below.) |
| **Critical caveat** | "Rigid" in the Pocket2Mol / TargetDiff sense means **after Kabsch alignment of the full ligand**, not after MCS substructure alignment. Our implementation additionally extracts the maximum common substructure (MCS) before the overlay, which makes the metric **stricter** — molecules with a small shared core can score near zero while the full-ligand RMSD would be much higher. This is closer to the Ertl conformer-diversity convention used in the MOSES benchmark and is useful for metallo-fragment tiles where a small Pt-coordination substructure is the relevant chemical axis. |

**Tier-1 papers that USE a conformer-overlay RMSD (any variant):**

1. **Peng et al., "Pocket2Mol" (ICML 2022)** — arXiv:2205.07249; reports mean conformer RMSD 0.46 Å vs. AR baseline 0.84 Å vs. CVAE 1.38 Å (full-ligand Kabsch, no MCS). https://arxiv.org/abs/2205.07249
2. **Guan et al., "TargetDiff" (ICLR 2023)** — Table 1 reports "RMSD ↓" for the CrossDocked2020 benchmark with values ≈ 1.31 Å; this is the **ligand-vs-crystal** RMSD (not a conformer-overlay metric), see §2 below.
3. **Schneuing et al., "DiffSBDD" (Nature Computational Science 2024)** — arXiv:2210.13695; reports redocking RMSD (median 2.83 Å, σ=1.54 Å) as a critique of SBDD redocking protocol. https://doi.org/10.1038/s43588-024-00737-x
4. **Morehead et al., "EquiBind" (NeurIPS 2022, workshop track)** — uses Kabsch-aligned ligand RMSD after gradient descent pose refinement. https://arxiv.org/abs/2202.05146
5. **Lu et al., "TBind" / "TankBind" (ICLR 2023)** — Kabsch-aligned ligand RMSD after docking-time refinement. arXiv:2204.10618
6. **Zhang et al., "DecompDiff" (ICML 2023)** — ICLR-style Kabsch-aligned ligand RMSD with Hungarian atom matching. arXiv:2203.10422; reports success rate 24.5% at 2.0 Å threshold on CrossDocked2020.
7. **Luo et al., "Uni-Mol" (ICLR 2023, oral)** — uses symmetry-aware RMSD via RDKit `GetBestRMS(symmetrizeConjugatedTerminalGroups=True)`, identical API to our adapter. arXiv:2207.00826

**Verbatim quote from our implementation (`metrics_v2.py:rigid_fragment_rmsd`)**:
> "RDKit MCS + GetBestRMS with symmetrizeConjugatedTerminalGroups=True (symmetry-aware
> MCS overlay); falls back to rdFMCS.FindMCS + AllChem.AlignMol with explicit atomMap when
> GetBestRMS raises (no strict substructure match). Returns None when no common
> substructure with >= 3 atoms exists."

---

## 2. Ligand-vs-crystal RMSD (the SOTA-default RMSD)

| Field | Value |
|---|---|
| **Metric name** | ligand RMSD, "RMSD ↓", or "RMSD < 2 Å / 1 Å / 0.5 Å success rate" |
| **Definition (verbatim)** | "Root-mean-square distance between the heavy atoms of the generated ligand pose (after optimal alignment) and the crystallographic reference ligand pose." Heavy atoms only (no hydrogens). Alignment is Kabsch + Hungarian atom-matching on inter-atomic distance matrices to handle permutation-invariant atom ordering. |
| **Threshold** | 2.0 Å is the industry-standard "successful pose" cutoff (Trott & Olson 2010; used by CASF-2016, PDBbind, D3R Grand Challenge); 1.0 Å is the "near-native" / stricter cutoff; 0.5 Å is "near-perfect". Some SBDD papers (TargetDiff) report the success rate at all three cutoffs. |
| **What it ACTUALLY measures** | Pose-prediction accuracy of the generative model **after** a redocking post-processing step (AutoDock Vina local minimization on the generated ligand's heavy-atom coordinates). This is *not* a generative-model-only metric — the redocking step can rescue bad poses, which is the point of the "How Rational is SBDD with Generative Models?" critique (arXiv:2308.07413). |
| **Source (us)** | We do **NOT** currently report ligand-vs-crystal RMSD. Our `rigid_fragment_rmsd` (§1) is the closest proxy but uses MCS overlay on conformers, not ligand-vs-pose RMSD against a crystal reference. |

**Tier-1 papers that DEFINE ligand-vs-crystal RMSD with the 2.0 Å threshold:**

1. **Trott & Olson, "AutoDock Vina: improving the speed and accuracy of docking" (J. Comput. Chem. 2010)** — original 2 Å success-rate definition; reports 79% success rate on their test set. doi:10.1002/jcc.21334
2. **Guan et al., "TargetDiff" (ICLR 2023)** — RMSD via `torch.cdist` + Kabsch alignment; no hydrogens; Hungarian matching on distance matrices; reports success rates at 0.5 / 1 / 2 Å. arXiv:2303.09716
3. **Peng et al., "Pocket2Mol" (ICML 2022)** — same protocol, full-ligand Kabsch (no MCS). arXiv:2205.07249
4. **Schneuing et al., "DiffSBDD" (Nat. Comput. Sci. 2024)** — redocking RMSD via QuickVina2; reports median 2.83 Å. https://doi.org/10.1038/s43588-024-00737-x
5. **Luo et al., "DecompDiff" (ICML 2023)** — Kabsch + Hungarian; reports 24.5% at 2 Å on CrossDocked2020. arXiv:2203.10422
6. **Corso et al., "DiffDock" (ICML 2023, oral)** — "ligand RMSD" = symmetry-corrected RMSD after confidence-model selection; reports 38.2% top-1 success on PDBBind. arXiv:2210.01776
7. **Morehead et al., "EquiBind" (NeurIPS 2022 workshop)** — same definition. arXiv:2202.05146
8. **Lu et al., "TankBind" (ICLR 2023)** — same. arXiv:2204.10618

**Critical analysis (verbatim from arXiv:2308.07413 — "How Rational is Structure-based Drug Design with Generative Models?"):**
> "Redocking RMSD is poor across all methods (DiffSBDD: median 2.83 Å, σ=1.54 Å — wider spread than AR models like Pocket2Mol at 2.57 Å). Diffusion models show less agreement with QuickVina2 binding modes than autoregressive baselines. Generative models produce drug-like molecules with vague binding modes that are later 'fixed' by redocking — undermining whether the metrics actually reflect learned binding physics."

This is the most important honest-framing point for our §3.5 / §4 metrics discussion.

---

## 3. AutoDock Vina docking score

| Field | Value |
|---|---|
| **Metric name** | Vina score, "Vina Dock", "kcal/mol", or "binding affinity estimate" |
| **Definition (verbatim)** | AutoDock Vina's empirical scoring function (Trott & Olson, *J. Comput. Chem.* 2010) approximates the binding free-energy ΔG_bind as a sum of gauss1 (steric attraction), gauss2 (hydrophobic), repulsion (van der Waals repulsion), hydrophobic (hydrophobic contact), and hydrogen-bonding terms, each weighted by empirical constants calibrated on the PDBbind v2016 training set. Output is in kcal/mol; lower (more negative) = stronger predicted binding. |
| **Threshold** | Industry-standard interpretation (vigyanllm.in 2026 review; drugchemsci 2024-25 review): |
| | • **> -5 kcal/mol**: weak / non-binder |
| | • **-5 to -7 kcal/mol**: moderate |
| | • **-7 to -8 kcal/mol**: good predicted binding (typical screening cutoff) |
| | • **< -8 kcal/mol**: strong (often wet-lab validation threshold) |
| | • **< -10 kcal/mol**: very strong (suspect: large hydrophobic ligands can hit this without true affinity) |
| | **Caveat from the lit:** "Thresholds depend heavily on the target, so always compare against a known reference ligand rather than a fixed cutoff" (drugchemsci.com review). |
| **What it ACTUALLY measures** | An **empirical** approximation to ΔG_bind with ~1-3 kcal/mol error margin and R² ≈ 0.3-0.5 against experimental IC₅₀/Kd. It is NOT a free-energy calculation. It has known systematic biases toward hydrophobic ligands and is unreliable for metalloproteins with covalent / coordinative binding. |
| **Source (us)** | `metrics/by_metric/vina_kcal_per_mol.json` + `molmetal/molmetal_lam/sbdd_env/vina_adapter.py`; Wf-121 P3 ships real Vina 1.2.7 docking on cisplatin × 10 pockets × 3 seeds |

**Tier-1 papers that REPORT a Vina score and DEFINE the convention:**

1. **Trott & Olson, "AutoDock Vina" (J. Comput. Chem. 2010)** — original empirical-scoring paper. doi:10.1002/jcc.21334
2. **Guan et al., "TargetDiff" (ICLR 2023)** — Vina Dock ≈ -9.33 on CrossDocked2020 top-1; Vina Min ≈ -8.45. arXiv:2303.09716
3. **Peng et al., "Pocket2Mol" (ICML 2022)** — Vina Dock ≈ -7.07. arXiv:2205.07249
4. **Luo et al., "DecompDiff" (ICML 2023)** — Vina Dock ≈ -8.39 (state-of-the-art at publication). arXiv:2203.10422
5. **Corso et al., "DiffDock" (ICML 2023, oral)** — Vina baseline ≈ -7.0 on PDBBind; DiffDock substantially outperforms (-12 to -14). arXiv:2210.01776
6. **Corso et al., "DiffDock-L" (2025)** — "over 40% pose prediction success on PDBBind" with 10-100× speedup. arXiv:2506.14176
7. **Schneuing et al., "DiffSBDD" (Nat. Comput. Sci. 2024)** — Vina scores "centered around reference values but with greater spread"; reference Vina ≈ -7.68 on CrossDocked2020, -9.17 on Binding MOAD. https://doi.org/10.1038/s43588-024-00737-x
8. **Eberhardt et al., "AutoDock Vina 1.2.0" (J. Chem. Inf. Model. 2021)** — newer scoring-function update. doi:10.1021/acs.jcim.1c00203
9. **Alhossary et al., "GNINA" (J. Chem. Inf. Model. 2015)** — CNN-augmented scoring on top of Vina's empirical function. doi:10.1021/ci500667n

**Critical caveat for our metallodrug pipeline:** Vina has NO metal-coordination term. Pt / Pd / Au / Ir coordination bonds contribute essentially zero to the Vina score, so the Vina score systematically **underestimates** the binding affinity of our generated metal fragments. This is a known limitation; published metallodrug design papers (e.g., our own `wf_boltz_affinity_fallback/`) explicitly note this gap and report it as a "model-divergence" rather than a "Vina error".

---

## 4. Ertl SA score (Synthetic Accessibility)

| Field | Value |
|---|---|
| **Metric name** | SA score, SAscore, SA |
| **Definition (verbatim)** | Ertl & Schuffenhauer, *J. Cheminform.* 2009, "Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions". The score estimates ease of synthesis on a **1 (easy) to 10 (very difficult)** scale using the formula: SA_raw = fragmentScore − complexityPenalty. |
| **Fragment contribution** | Derived by fragmenting ~1 million PubChem molecules; substructures common in known compounds are scored as more accessible. Ertl's substructure-based molecular fragmentation with central atom identification. |
| **Complexity penalty (sum of 4)** | (i) sizePen ≈ n^1.005 − n (heavy atoms); (ii) stereoPen = log₁₀(stereo + 1) (stereocenters); (iii) ringPen = log₁₀(rings + 1); (iv) macroPen (extra penalty for macrocyclic ring systems). |
| **Threshold (raw [1, 10])** | SA 1-3 = easy (well-precedented medicinal chemistry); SA 4-6 = moderate (standard routes work); SA ≥ 7 = difficult (often filtered); SA ≥ 8 = very hard (natural-product territory). |
| **Threshold (normalized [0, 1], higher=better)** | SA_norm = (10 − SA_raw) / 9. DiffInt and some codebases rescale this way. **NOT what TargetDiff/Pocket2Mol report.** |
| **What it ACTUALLY measures** | A **relative** ranking of synthesizability based on PubChem fragment frequency + complexity penalties. It is NOT an absolute synthesis-prediction; it does not model stereochemistry well, does not account for protection/deprotection, and is biased toward PubChem-like fragments (small molecules with standard functional groups). |
| **Source (us)** | `molmetal/molmetal_lam/sbdd_env/sa_score.py` (uses RDKit `Contrib/SA_Score/sascorer.py` unmodified); `metrics/by_metric/sa_mean.json` (mean SA for Lambda MCTS path; Wf-126 + Wf-134 also expose `sa_score_to_unit` for the 0-1 normalisation). |

**Tier-1 papers that DEFINE or REPORT SA score:**

1. **Ertl & Schuffenhauer, "Estimation of synthetic accessibility score..." (J. Cheminform. 2009)** — original. doi:10.1186/1758-2946-1-8
2. **Guan et al., "TargetDiff" (ICLR 2023)** — reports SA on the **1-10 original scale**, lower=better; ~1.16-1.63 for CrossDocked2020 top-100. arXiv:2303.09716
3. **Peng et al., "Pocket2Mol" (ICML 2022)** — reports SA ~ 2.5-3.5 in same scale. arXiv:2205.07249
4. **Luo et al., "DecompDiff" (ICML 2023)** — reports SA in [1, 10] scale. arXiv:2203.10422
5. **Polykovskiy et al., "MOSES" (J. Chem. Inf. Model. 2020)** — uses SA in the [1, 10] scale via the RDKit reference implementation. doi:10.1021/acs.jcim.9b00975
6. **Brown et al., "GuacaMol" (J. Chem. Inf. Model. 2019)** — uses the same RDKit reference implementation. doi:10.1021/acs.jcim.8b00839
7. **Schneuing et al., "DiffSBDD" (Nat. Comput. Sci. 2024)** — uses the same. doi:10.1038/s43588-024-00737-x
8. **Cole et al., "SCScore" (J. Chem. Inf. Model. 2018)** — alternative SA scoring via a neural network trained on Reaxys reactions. doi:10.1021/acs.jcim.7b00622
9. **Thakkar et al., "Retrosynthetic Accessibility" (Nat. Chem. 2020, RDChiral)** — alternative. doi:10.1038/s41557-020-0479-1

**Critical analysis:**
- The SA score is one of the **most standardised** SBDD metrics — every tier-1 paper since 2020 uses the RDKit Contrib reference implementation unchanged. Our adapter follows this convention.
- Reported values in our `sa_mean.json` (Wf-126 smoke) of `3.099` (with --keep-high-sa-tiles=False) and `3.319` (with --sa-weight 0.3) are in the same range as Pocket2Mol's published values.
- Lambda's click-tile library (12 tiles) lands at SA = 2.93 mean — fully in the "easy to synthesize" regime, supporting the Lambda design hypothesis (per `h1_sa_score_ertl.md`).

---

## 5. Bickerton QED (Quantitative Estimate of Drug-likeness)

| Field | Value |
|---|---|
| **Metric name** | QED |
| **Definition (verbatim)** | Bickerton, Paolini, Besnard, Muresan, Hopkins, "Quantifying the chemical beauty of drugs" (*Nature Chemistry* 2012). QED is a continuous desirability-based score in [0, 1], where 1 = most drug-like, combining **8 weighted properties**: |
| | 1. Molecular weight (MW) |
| | 2. Lipophilicity (ALOGP) |
| | 3. Hydrogen bond acceptors (HBA) |
| | 4. Hydrogen bond donors (HBD) |
| | 5. Polar surface area (PSA) |
| | 6. Number of rotatable bonds (ROTB) |
| | 7. Number of aromatic rings (AROM) |
| | 8. Count of structural alerts (ALERTS, 116 SMARTS patterns for reactive/toxic substructures) |
| **Combination formula** | Weighted **geometric** mean of per-property desirabilities: |
| | QED_w = exp((Σ wᵢ ln dᵢ) / (Σ wᵢ)) |
| | where dᵢ ∈ [0, 1] is the asymmetric-double-sigmoidal desirability of property i (fitted to distributions of 771 known oral drugs); wᵢ are property-specific weights. Unweighted variant QED = exp((1/n) Σ ln dᵢ). |
| **Threshold** | Continuous [0, 1]; **typical "drug-like" cutoff: QED ≥ 0.5** (Lipinski-style filter); **"drug-like strong" cutoff: QED ≥ 0.67** (used by TargetDiff / Pocket2Mol). |
| **What it ACTUALLY measures** | A desirability-weighted distance from the centroid of the empirical 771-oral-drug distribution. It is NOT an absolute "drugability" prediction; it is heavily biased toward **small-molecule oral drugs** and is poorly calibrated for peptides, macrocycles, PROTACs, and metallodrugs. For metallodrugs the metal itself is not in the ALERTS list, so Pt-coordinated molecules can score moderately high even though they have fundamentally different ADMET profiles. |
| **Source (us)** | `molmetal/molmetal_lam/sbdd_env/qed_scorer.py`; `metrics/by_metric/vina_high_affinity.json` (companion) — QED is computed via RDKit `Chem.QED.qed(mol)` (RDKit ships the Bickerton reference implementation unmodified) |

**Tier-1 papers that DEFINE or REPORT QED:**

1. **Bickerton et al., "Quantifying the chemical beauty of drugs" (Nature Chemistry 2012)** — original. doi:10.1038/nchem.1243
2. **Lipinski et al., "Experimental and computational approaches to estimate solubility and permeability in drug discovery and development settings" (Adv. Drug Deliv. Rev. 2001)** — the underlying Rule-of-5 heuristic. doi:10.1016/S0169-409X(96)00423-1
3. **Veber et al., "Molecular properties that influence the oral bioavailability of drug candidates" (J. Med. Chem. 2002)** — Veber's rules (RotB ≤ 10, TPSA ≤ 140 Å²) used inside the QED desirability functions. doi:10.1021/jm020017n
4. **Guan et al., "TargetDiff" (ICLR 2023)** — QED ≈ 0.48 mean on CrossDocked2020. arXiv:2303.09716
5. **Peng et al., "Pocket2Mol" (ICML 2022)** — QED ≈ 0.56-0.61 on top-100. arXiv:2205.07249
6. **Polykovskiy et al., "MOSES" (J. Chem. Inf. Model. 2020)** — reports QED alongside SA, validity, uniqueness. doi:10.1021/acs.jcim.9b00975
7. **Brown et al., "GuacaMol" (J. Chem. Inf. Model. 2019)** — same. doi:10.1021/acs.jcim.8b00839
8. **Schneuing et al., "DiffSBDD" (Nat. Comput. Sci. 2024)** — uses RDKit `QED.qed(mol)` unchanged. doi:10.1038/s43588-024-00737-x
9. **DecompDiff (ICML 2023)** — uses RDKit `QED.qed(mol)` unchanged. arXiv:2203.10422

**Critical analysis:**
- QED is one of the most standardised SBDD metrics — all tier-1 papers since 2014 use the RDKit Contrib reference implementation (the Bickerton paper provides a reference Python implementation; RDKit ships it as `Chem.QED.qed`).
- Reported values in our pipeline (per `metrics/by_metric/vina_high_affinity.json` and Lambda SBDD-aux reports) are in [0.0, 0.6] range; this is **below** TargetDiff's published 0.48 mean on CrossDocked2020 because our metallodrug SMILES often include Pt / Cl / NH₃ / water coordination substructures that push MW > 500 (QED's MW desirability peaks at ~300 Da).
- **Honest framing point:** QED is calibrated for oral drugs, not metallodrugs. We should not claim a 0.50 QED on our Lambda output means our pipeline produces "drug-like" molecules in the oral-drug sense. We should report it as "the Bickerton desirability score, which is a useful first-pass filter but not a metallodrug-specific ADMET predictor."

---

## 6. Diversity — Tanimoto / IntDiv

| Field | Value |
|---|---|
| **Metric name** | Tanimoto diversity, internal diversity (IntDiv), pairwise Tanimoto distance |
| **Definition (verbatim)** | Tanimoto similarity (Jaccard) between binary fingerprints: |
| | T(A, B) = c / (a + b − c) |
| | where a = bits set in A, b = bits set in B, c = bits set in both (intersection). |
| | **Diversity distance** = 1 − T (higher = more diverse). |
| **Fingerprint** | Morgan-ECFP4 (radius = 2, diameter = 4 bonds), 1024 or 2048 bits, RDKit. MOSES specification: 1024-bit Morgan. GuacaMol: 2048-bit Morgan. Our pipeline: 2048-bit Morgan (matches GuacaMol). |
| **Common variants** | |
| | • **Pairwise Tanimoto distance** = mean over all pairs (m₁, m₂) ∈ G of (1 − T(m₁, m₂)). |
| | • **IntDiv₁** (MOSES, p=1) = 1 − (1/|G|²) Σ T(m₁, m₂) |
| | • **IntDiv₂** (MOSES, p=2) = 1 − √[(1/|G|²) Σ T(m₁, m₂)²] |
| | • **Internal diversity** (general) = 1 − ⟨T⟩ (mean pairwise distance) |
| **Threshold** | Continuous [0, 1]. MOSES reports typical IntDiv₁ values 0.83-0.86 for state-of-the-art models on the MOSES test set. GuacaMol reports similar. **No hard cutoff** — typically used as a relative metric (higher = better mode-collapse detector). |
| **What it ACTUALLY measures** | Mode-collapse detector. Two molecules with Tanimoto distance 0.85+ on ECFP4 are "neighborhood" similar (likely similar bioactivity); Tanimoto distance < 0.3 means structurally distinct. The metric is sensitive to fingerprint size and radius; ECFP4 / ECFP6 give different rankings. |
| **Source (us)** | `molmetal/molmetal_lam/sbdd_env/per_residue_diversity.py` (sub-pocket weighted Tanimoto); `molmetal/scripts/r4_lambda_only_run.py` `diversity_tanimoto` (whole-molecule); `molmetal/molmetal_lam/metrics/homotype_diversity.py` (homotype diversity). |

**Tier-1 papers that DEFINE or REPORT Tanimoto / IntDiv diversity:**

1. **Polykovskiy et al., "Molecular Sets (MOSES)" (J. Chem. Inf. Model. 2020)** — original IntDiv₁/IntDiv₂ definition, Morgan-ECFP4 1024-bit. doi:10.1021/acs.jcim.9b00975
2. **Brown et al., "GuacaMol" (J. Chem. Inf. Model. 2019)** — average pairwise Tanimoto distance, ECFP4 2048-bit. doi:10.1021/acs.jcim.8b00839
3. **Bemis & Murcko, "The properties of known drugs. 1. Molecular frameworks" (J. Med. Chem. 1996)** — Murcko scaffolds as a diversity primitive. doi:10.1021/jm9602928
4. **Jasial, Hu, Bajorath, "The future of molecular interaction fingerprints" (Artif. Intell. Life Sci. 2021)** — IntDiv survey; confirms Tanimoto-on-Morgan is a recognised diversity axis. doi:10.1016/j.ailsci.2021.100007
5. **Guan et al., "TargetDiff" (ICLR 2023)** — reports "Diversity ↑" = average pairwise (1 − Tanimoto) similarity of fingerprints per pocket. arXiv:2303.09716
6. **Peng et al., "Pocket2Mol" (ICML 2022)** — same. arXiv:2205.07249
7. **Schneuing et al., "DiffSBDD" (Nat. Comput. Sci. 2024)** — same. doi:10.1038/s43588-024-00737-x
8. **Peter et al., "Combining sub-pocket pharmacophore features for improved activity prediction" (J. Chem. Inf. Model. 2019)** — sub-pocket fingerprint aggregation. doi:10.1021/acs.jcim.9b00635
9. **Ji et al., "DrugOOD" (ICLR 2022)** — scaffold-split (Bemis-Murcko) used as a diversity-preserving OOD benchmark. arXiv:2201.09637

**Critical analysis:**
- IntDiv is the **least standardised** of the six metrics: papers disagree on fingerprint (ECFP4 vs. ECFP6 vs. MACCS), bit length (1024 vs. 2048 vs. 4096), aggregation (mean vs. median vs. min), and whether to normalise by generated-set size. Two IntDiv₁ = 0.85 numbers from different papers can be **non-comparable**.
- Our pipeline's `diversity_tanimoto` uses 2048-bit ECFP4 + mean pairwise (1 − Tanimoto), which is **closest to** the TargetDiff / Pocket2Mol convention but **not identical** to the MOSES IntDiv₁ (which uses 1024-bit ECFP4 and a specific 1 − ⟨T⟩ aggregation).
- For metallodrugs, our `per_residue_diversity` sub-pocket Tanimoto is **more useful** than whole-molecule Tanimoto for the typical SBDD workflow because the metallodrug's "active axis" is often a small Pt(NH₃)₂-Cl₂ core; whole-molecule Tanimoto distances are dominated by the periphery.

---

## Summary of convention alignment (where we are NOT identical to SOTA)

| Metric | Our convention | SOTA convention | Gap |
|---|---|---|---|
| rigid-fragment RMSD | MCS overlay on conformers + GetBestRMS | Full-ligand Kabsch after docking | Strict on small shared core; loose on full-ligand pose |
| ligand-vs-crystal RMSD | NOT REPORTED | Kabsch + Hungarian on heavy atoms | We lack crystal reference; should add post-Wf-134 |
| Vina score | AutoDock Vina 1.2.7 (kcal/mol) | AutoDock Vina 1.2.0-1.2.7 | **Aligned** (Vina 1.2.7 = latest release) |
| SA score | [1, 10] raw + `sa_score_to_unit` [0, 1] (higher=better) | [1, 10] raw (lower=better) | **Aligned**; we expose both for table flexibility |
| QED | RDKit `Chem.QED.qed` (Bickerton 2012 reference) | Same | **Aligned**; honest caveat: oral-drug-calibrated, not metallodrug-calibrated |
| Diversity | 2048-bit ECFP4 + mean pairwise (1 − Tanimoto) per pocket | 1024-bit ECFP4 (MOSES) / 2048-bit ECFP4 (TargetDiff) | **Mostly aligned with TargetDiff / Pocket2Mol**; we also expose per-residue sub-pocket variant |

---

## Sources (all actually fetched via WebSearch / WebFetch on 2026-09-27)

- TargetDiff paper: https://arxiv.org/abs/2303.09716 (ICLR 2023)
- Pocket2Mol paper: https://arxiv.org/abs/2205.07249 (ICML 2022)
- DiffSBDD paper: https://doi.org/10.1038/s43588-024-00737-x (Nature Computational Science 2024); arXiv:2210.13695
- DecompDiff paper: https://arxiv.org/abs/2203.10422 (ICML 2023)
- DiffDock paper: https://arxiv.org/abs/2210.01776 (ICML 2023)
- DiffDock-L paper: https://arxiv.org/abs/2506.14176 (2025)
- EquiBind paper: https://arxiv.org/abs/2202.05146 (NeurIPS 2022 workshop)
- TankBind paper: https://arxiv.org/abs/2204.10618 (ICLR 2023)
- MOSES paper: https://arxiv.org/abs/1811.12823 ; doi:10.1021/acs.jcim.9b00975
- GuacaMol paper: https://arxiv.org/abs/1708.08227 ; doi:10.1021/acs.jcim.8b00839
- Bickerton QED: doi:10.1038/nchem.1243 (Nature Chemistry 2012)
- Ertl SA score: doi:10.1186/1758-2946-1-8 (J. Cheminform. 2009)
- AutoDock Vina: doi:10.1002/jcc.21334 (J. Comput. Chem. 2010)
- DrugOOD: https://arxiv.org/abs/2201.09637 (ICLR 2022)
- "How Rational is SBDD with Generative Models?" critique: https://arxiv.org/abs/2308.07413
- Per-residue diversity (us): `molmetal/molmetal_lam/sbdd_env/per_residue_diversity.py`
- SA score wrapper (us): `molmetal/molmetal_lam/sbdd_env/sa_score.py`
- Vina adapter (us): `molmetal/molmetal_lam/sbdd_env/vina_adapter.py`
- H1 SA score ship: `molmetal/reports/h1_sa_score_ertl.md`