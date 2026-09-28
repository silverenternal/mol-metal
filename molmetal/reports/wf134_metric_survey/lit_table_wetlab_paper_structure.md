# WF-134 — Tier-1 SBDD Wet-Lab Validation & Paper Structure Survey

**Workflow:** wf134_metric_survey (Phase 3 — wet-lab validation)
**Date:** 2026-09-27
**Scope:** Per-journal (Nature MI / Nat. Comput. Sci. / Briefings in Bioinformatics) wet-lab
requirements for SBDD-class papers; per-assay (IC50 / Kd / cellular uptake / binding) minimum
sample size + replicates + statistical bar; per-section (intro / methods / results / discussion /
limitations / supplementary) what content is conventionally placed.
**Constraint:** party-twenty-spirit §3.5 (honest framing of what tier-1 journals actually require
vs. what we as a CPU-only computational lab can claim).

---

## Part A — Per-journal standards (4 tier-1 venues)

### A.1 Nature Machine Intelligence (Nat. Mach. Intell., NMI)

| Field | Value |
|---|---|
| **Publisher / venue** | Nature Portfolio (Springer Nature), ISSN 2522-5839 |
| **Scope** | Original research in ML/AI with conceptual significance OR real-world relevance. Drug-discovery papers must have an AI component that is NOT a "black-box accelerator" — interpretive insight is required. |
| **Wet-lab requirement** | Not strictly mandatory, but if the paper claims a scientific discovery (e.g. "we designed a binder to X"), NMI expects **experimental validation that the AI method enables a result inaccessible by prior methods**. Pure benchmark superiority without scientific consequence is rejected at the desk. |
| **Paper structure (Article)** | Introduction (no heading) → Results → Discussion → Online Methods; ≤5000 words main text, ≤150-word abstract, ≤60 references, ≤6 display items |
| **Methods / Online Methods** | Divided by topical subheadings. Must include: dataset provenance, train/val/test separation, hyperparameters, compute resources, statistical analysis method, human-participant oversight if applicable, random seeds (e.g. `torch.manual_seed(42)`) |
| **Statistics bar (Methods section)** | Statistical testing of performance differences required; multiple datasets / settings expected; appropriate baselines with matched compute/budget; ablation studies; random seeds fixed; resource budgets reported |
| **Reproducibility** | Complete, raw, unprocessed datasets OR clear access documentation; code open-sourced at submission (GitHub/GitLab with README, dependency lists, run instructions); Docker / Conda env recommended; promises to release materials "after acceptance" are discouraged |
| **Limitations** | Required (separate section) |
| **APC** | €9,500 (~USD 11,690); 100% waiver for low-income, 50% for middle-income |
| **Key references** | AlphaFold 3 (Nature 2024; s41586-024-07415-w — sister Nature venue); Cao 2022 miniprotein binder paper (Nature Methods 2022; s41587-022-01557-w — wet-lab FACS + binding assay on 52 designed miniproteins) |
| **Source URLs** | https://mts-natmachintell.nature.com/ (submission system); https://manusights.com/blog/nature-machine-intelligence-submission-guide (third-party synthesis of NMI requirements) |

**What NMI actually scrutinises for SBDD-style submissions (verified from public editor letters and Nature Portfolio Reporting Standards):**
- Dataset provenance must be auditable (FAIR: Findable, Accessible, Interoperable, Reusable).
- Train/val/test separation MUST be done at the **molecule** level (not per-conformer or per-frame); scaffold split expected for generalization claims.
- AI-use disclosure: NO LLM authorship; document LLM use in Methods.
- CRediT author contribution statement required; Conflict of Interest disclosure required.
- Common rejection reasons: (1) benchmark gain without general insight; (2) generality claimed from narrow evaluation; (3) baseline/compute asymmetry; (4) transparency arriving too late (code/data only promised post-acceptance); (5) missing resource accounting.

---

### A.2 Nature Computational Science (Nat. Comput. Sci., NCS)

| Field | Value |
|---|---|
| **Publisher / venue** | Nature Portfolio, ISSN 2662-8457 |
| **Scope** | Computational methods papers where "computation is the *reusable scientific advance*", not merely the tool. Methods that interface with biology must demonstrate that computation changes what researchers can infer, predict, simulate, or discover across a meaningful domain. |
| **Wet-lab requirement** | **Conditional** — NCS does not require wet-lab work for a pure-method paper, but if the paper claims a "scientific discovery" (e.g. identifying a drug target, predicting off-target effects, designing a ligand), NCS expects validation against at least one external dataset and ideally orthogonal experimental evidence. The Oct 2024 MitoReID paper (Yu et al., s43588-024-00682-9) is the in-house example: it included in vitro COX-2 inhibition validation. |
| **Paper structure (Article)** | Introduction (no heading) → Results → Discussion → Online Methods; ≤3500 words main text, ≤150-word abstract, ≤50 references, ≤6 display items |
| **Methods / Online Methods** | Should expose failure cases; provide reproducible paths; generalization tested across multiple tasks, systems, regimes, or external datasets |
| **Statistics bar** | Matched-comparison audit expected: exact training/evaluation datasets, versions, splits, access dates; compute and tuning budget for each baseline; performance-cost tradeoffs reported; code, parameters, seeds, environment, and output commands documented |
| **Reproducibility** | Public datasets required in repositories (figshare, Zenodo, Dryad) with DOI, per FAIR principles; GitHub link or supplementary information for algorithms; preprint policy accepts arXiv/ChemRxiv (must be declared at submission) |
| **Limitations** | Required; Discussion has no subheadings but Limitations is a separate section |
| **APC** | €9,500; same waiver structure as NMI |
| **Key references (recent)** | DiffSBDD (Schneuing 2024, s43588-024-00737-x — redocking protocol paper); DeepBlock (Li 2024, s43588-024-00718-0 — ligand generation with reactive building blocks); MitoReID (Yu 2024, s43588-024-00682-9 — wet-lab COX-2 inhibition); Durant 2024 ML data commentary (s43588-024-00699-0) |
| **Source URLs** | https://www.nature.com/natcomputsci (journal home); https://manusights.com/blog/nature-computational-science-submission-guide |

**What NCS scrutinises for SBDD-style submissions (in-house editor priority, Fernando Chirigati, chief editor):**
1. **Scientific data management, provenance, analytics** = first-screen evidence.
2. **Computational reproducibility** must be demonstrable (NOT just promised).
3. **Matched-comparison audit** = the most-common desk-reject trigger (e.g. Vina score vs. our method without matched compute budget).
4. **Generalization claims** must be tested on ≥2 tasks OR ≥2 domains; cross-pocket or cross-target validation expected.

---

### A.3 Briefings in Bioinformatics (BiB)

| Field | Value |
|---|---|
| **Publisher / venue** | Oxford University Press, ISSN 1467-5463 (print) / 1473-7104 (online) |
| **Scope** | Reviews and tutorials in bioinformatics that aim to provide cutting-edge overviews of current research. **Original research articles are NOT generally considered** unless they present a significant methodological advance. Considers: Review Articles, Tutorials, Brief Communications, Application Notes. |
| **Wet-lab requirement** | **Variable** — BiB accepts both pure-computational reviews AND mixed wet-lab/computational papers. For a methods paper with experimental validation, BiB expects the wet-lab component to be **supporting evidence** for the computational contribution, NOT the primary advance. |
| **Paper structure** | Abstract → Keywords → Introduction → Methods (if applicable) → Results → Discussion → Conclusions. Times New Roman 11pt; double-spacing. |
| **Manuscript length** | ≤50 pages including references; Brief Communications ≤8 typeset pages |
| **Figures** | TIFF / EPS / high-resolution PDF, ≥300 dpi, submitted as separate files |
| **Supplementary material** | Encouraged, submitted as separate files with clear reference in main manuscript |
| **Reproducibility** | Code/data availability statement required; DOI citation for deposited code/data expected; open access available via Oxford Open (APC applies) |
| **Limits section** | Conventional but not mandated; reviewers regularly ask for one |
| **Peer review** | Double-blind |
| **Key references (recent SBDD)** | "Structure-based drug design with graph learning, molecular generation and protein structure prediction" (Wang 2024, bbae258, 25(4)); "Protein language models are performant in structure-free virtual screening" (Lam 2024, bbae480, 25(6)); "Rag2Mol: Structure-based drug design based on retrieval augmented generation" (bbaf265) |
| **Source URLs** | https://academic.oup.com/bib/pages/general-instructions |

**What BiB scrutinises for SBDD-style submissions (2024-2026 cohort patterns):**
- All three 2024 BiB SBDD reviews converge on the same critique: **computational SBDD outputs frequently fail wet-lab validation** due to (1) poor synthetic accessibility, (2) training-data bias (memorizing known ligands), (3) over-reliance on Vina docking scores.
- BiB reviewers explicitly call for: similarity-based metrics (to known actives), virtual-screening-based metrics (actives vs. decoys), multi-level evaluation frameworks.
- BiB does NOT require a wet-lab section in a review/tutorial, but if present, reviewers expect the wet-lab to validate the framework, not just demonstrate it.

---

### A.4 Journal of Chemical Information and Modeling (JCIM) — comparator tier-2 venue

| Field | Value |
|---|---|
| **Publisher / venue** | American Chemical Society, ISSN 1549-9596 |
| **Scope** | Chemical informatics, computational chemistry, molecular modeling |
| **Wet-lab requirement** | **REJECTED without wet-lab validation** for methods applied to single target systems — per JCIM editorial policy, a docking method or ML prediction method without sufficient experimental validation is rejected. Landrum & Riniker 2024 (64(5):1560-1567, DOI 10.1021/acs.jcim.4c00049) documents how combining IC50 data from different assays introduces ~65% noise >0.3 log units without curation, making validation bar explicit. |
| **APC** | ~$4,000 open access |
| **Key reference** | Landrum & Riniker 2024 "Combining IC50 or Ki Values from Different Sources Is a Source of Significant Noise" — JCIM 64(5):1560-1567 |
| **Source URLs** | https://pubs.acs.org/jcisd8 |

**Why JCIM is in this table:** JCIM is the most relevant tier-2 venue for our pipeline and its bar ("docking methods without sufficient experimental validation are rejected") is the **most explicit wet-lab requirement** of any venue surveyed. NCS, NMI, and BiB all have weaker wet-lab requirements than JCIM.

---

## Part B — Per-assay minimums (4 wet-lab assay types)

### B.1 IC50 (half-maximal inhibitory concentration)

| Aspect | Convention | Source |
|---|---|---|
| **Replicates (technical)** | ≥3 technical replicates per concentration point; ≥9 concentration points per curve (typically half-log dilutions, 10 nM – 100 µM or 100 pM – 1 µM depending on expected potency) | NCI-60 HTS384 (Kunkel 2024, Cancer Res 84(15):2403); standard pharmacology convention |
| **Replicates (biological)** | ≥3 independent experiments (different days, different cell passages / different protein batches) | Nature Methods statistical guidelines; Landrum & Riniker 2024 (JCIM) |
| **Curve fit** | 4-parameter logistic (Hill equation) with bottom, top, Hill slope, EC50/IC50; report Hill slope (typical 0.8–1.5) and R² | GraphPad Prism convention; reported in Nat. Mach. Intell. + Nat. Chem. Biol. methods sections |
| **Statistical bar** | 95% CI on log(IC50); report SE or 95% CI; report individual data points (Nature encourages dot plots with mean overlay over bar charts) | Nature Reporting Standards |
| **Quality control** | Z'-factor ≥0.5 for the assay platform; signal:background ≥5; CV of vehicle controls ≤10% | Kunkel 2024 NCI-60 HTS384 (Z' avg 0.82, CV 5.2%) |
| **Reference compound** | At least one known inhibitor with literature IC50 run in parallel; ±0.5 log units agreement expected | Landrum & Riniker 2024 |
| **Acceptance criterion for "validated hit"** | IC50 ≤10 µM (typical screening hit threshold); confirmation in orthogonal assay (SPR or ITC for binding, cell viability for functional) | Industry convention; reported in Isomorphic Labs / Insilico Medicine public disclosures |
| **Common pitfalls** | Curve fit converging on a single replicate; combining data from different assays without normalisation (Landrum 2024 quantifies: 65% of points differ >0.3 log units pre-curation) | Landrum & Riniker 2024 |

---

### B.2 Kd / KD (equilibrium dissociation constant — SPR / ITC / BLI)

| Aspect | Convention | Source |
|---|---|---|
| **Replicates (technical)** | ≥3 analyte concentration series (typically 5–8 concentrations, 2-fold or 3-fold serial dilution); 2–3 replicate injections per concentration | Biacore / Cytiva application notes; Creative-Proteomics SPR/BLI SOP |
| **Replicates (biological)** | ≥2 independently prepared protein batches, ideally ≥3; report inter-batch variability | Biacore method guide |
| **Buffer requirement** | PBS or HBS (10 mM HEPES pH 7.4, 150 mM NaCl, 0.05% Tween-20, ±3 mM EDTA); DMSO ≤5% final for compound stocks; avoid glycerol for HDX-MS | Creative-Proteomics hit-validation |
| **Affinity range** | Working range 100 µM (small molecules) to nM (protein binders); analyte at 0.1–10× Kd | MoB-IBT sample requirements (mosbri.eu) |
| **Curve fit** | 1:1 Langmuir binding model for single-site binding; report ka (M⁻¹s⁻¹), kd (s⁻¹), KD (M); report χ² and Rmax; residuals should be random with no systematic deviation | Biacore evaluation software; standard in Cao 2022 miniprotein paper (Nature Methods) |
| **Acceptance criteria** | KD standard deviation <20–30%; χ²/Rmax fit quality acceptable; residuals random; 1:1 fitting model appropriate | Biacore / SLAS Discovery 2024 guidance |
| **Orthogonal confirmation** | ITC (MicroCal iTC200: protein at 10× Kd, ligand at 20× protein, 50–600 µM) OR BLI OR DSF/CETSA | MoB-IBT; Creative-Proteomics |
| **Tier-1 paper examples** | Cao 2022 (Nature Methods, s41587-022-01557-w): SPR-based affinity measurement for 52 designed miniprotein binders; Adaptyv/Proteinbase 2024-2026 binder competitions: standardised SPR protocol, e.g. EGFR best 1.21 nM, RBX1 best 23.7 nM | Proteinbase.com competition reports |

---

### B.3 Cellular uptake / functional cellular assay

| Aspect | Convention | Source |
|---|---|---|
| **Replicates (technical)** | ≥3 wells per condition; ≥3 concentration points per curve (typical dose-response) | Standard pharmacology |
| **Replicates (biological)** | ≥3 independent experiments; different cell passages; report cell-line authentication (STR profiling) | Nature Reporting Standards |
| **Cell-line panel** | For anti-cancer claims, ≥6 cell lines minimum; ideally NCI-60 panel (or sub-panel covering 9 cancer types: lung, colon, breast, prostate, melanoma, renal, ovarian, brain, leukemia) | Kunkel 2024 NCI-60 HTS384 |
| **Detection method** | CellTiter-Glo luminescence (ATP-based, modernized 2024) replacing SRB absorbance; exposure time 72h for HTS384 (was 48h for classic NCI-60) | Kunkel 2024 |
| **Quality metrics** | Z'-factor ≥0.5 (HTS384 average 0.82); signal:background ≥5 (HTS384 average 89); CV ≤10% (HTS384 average 5.2%) | Kunkel 2024 |
| **Statistical bar** | GI50 / TGI / LC50 reported with 95% CI; dose-response curves with ≥6 concentration points; replicate plates on different days | NCI-60 standard |
| **Controls** | Vehicle (DMSO ≤0.1% v/v), positive control (doxorubicin / cisplatin), blank (medium only); reference compound with known GI50 run in parallel | NCI-60 SOP |
| **Tier-1 paper examples** | MitoReID (Nat. Comput. Sci. 2024): epicatechin binds COX-2 and inhibits activity, validated by in vitro cellular COX-2 assay; MAP (Nat. Mach. Intell. 2026): knowledge-driven framework for single-cell response prediction, validated in A-549 cells | Yu 2024; MAP 2026 |

---

### B.4 Direct binding assay (X-ray crystallography, cryo-EM, HDX-MS)

| Aspect | Convention | Source |
|---|---|---|
| **Replicates (technical)** | Single structure is the norm; resolution (Å), R-free, density map quality are the QC metrics | Standard structural biology |
| **Resolution bar** | ≤2.5 Å for unambiguous ligand placement; ≤2.0 Å for detailed interaction analysis | PDB deposition standards |
| **Validation** | MolProbity Ramachandran, rotamer outliers, clashscore; PDB validation report required for deposition | wwPDB |
| **Orthogonal confirmation** | HDX-MS (100–500 µg, PBS without glycerol); Native MS (10–50 µg, 50–200 mM ammonium acetate); mutagenesis of key contact residues + functional assay | Creative-Proteomics; standard in medicinal chemistry |
| **Tier-1 paper examples** | Chemgenomics 2025 cascade: X-ray crystallography as "gold standard" for binding mode confirmation; AlphaFold 3 (Nature 2024): validated against PDBbind + bespoke designed-binder X-ray structures | Chemgenomics.com; AlphaFold 3 paper |
| **Acceptance criterion** | Designed pose reproduced within RMSD ≤1.0 Å of design target; key H-bonds / salt bridges observed in 2Fo-Fc map at ≥1.0 σ | Standard for designed-binder papers |

---

## Part C — Per-section content (what each section of a tier-1 paper typically carries)

### C.1 Introduction (~600–1000 words)

**Typical content** (verified across NMI, NCS, BiB):
- **Problem framing** — what is the unmet need (e.g. "no metallodrug design exists for Pt family resistance in TNBC")
- **Prior work limitations** — what existing computational or wet-lab methods cannot do (2–3 paragraphs)
- **Our contribution** — explicit statement of the AI/computational advance + the experimental advance (if any)
- **Significance statement** — why this matters for the field (broad audience framing for NMI; method-specific framing for NCS; field-overview framing for BiB)

**Tier-1 convention**: NO subheadings in Introduction (Nature style); do NOT mix results into the intro; explicitly state limitations of prior approaches.

---

### C.2 Results (~2000–3000 words; ≤3500 for NCS Article)

**Typical structure**:
1. **Computational results** — benchmark tables (compared to SOTA), generalization tests (cross-pocket, cross-target), ablation studies, runtime / cost
2. **Wet-lab results (if applicable)** — IC50 dose-response curves, Kd sensorgrams, cellular viability curves, X-ray structures
3. **Joint analysis** — does the computational prediction match the experimental observation? Where does it diverge?
4. **Failure cases** — explicitly shown (NCS editor requirement)

**For a tier-1 SBDD paper with wet-lab validation**, the Results section is expected to:
- Show matched-comparison tables (your method vs. ≥3 baselines with matched compute budget)
- Report exact n values per experiment (Nature convention)
- Use dot plots with mean overlay rather than bar charts (Nature Reporting Standards 2023+)
- Provide exact p-values (not just p<0.05) for each comparison

---

### C.3 Discussion (~600–1200 words; no subheadings for NCS)

**Typical content**:
- **Interpretation** of results in biological/chemical context
- **Comparison to prior work** — explicit numerical comparison to published SOTA
- **Mechanism** — proposed molecular mechanism (especially for IC50 / Kd / cellular data)
- **Clinical/translational implications** — for SBDD papers, this is usually brief and explicit about being speculative
- **Limitations** — separate "Limitations" section required for NMI, NCS

---

### C.4 Methods / Online Methods

**Structure** (verified from Nature Portfolio Reporting Standards + NCS + BiB):
1. **Computational methods**:
   - Dataset description (provenance, license, version, access date)
   - Train/val/test split (with split strategy: random / scaffold / temporal)
   - Model architecture (with diagram reference)
   - Training procedure (hyperparameters, optimizer, learning rate, batch size, epochs, early stopping criterion)
   - Compute resources (GPU type, count, hours; CPU for Vina / RDKit operations)
   - Random seeds (fixed and reported)
   - Software versions (Python, RDKit, PyTorch, etc.)
2. **Wet-lab methods** (if applicable):
   - **Assay protocol** — cell line / protein / construct details, buffer composition, temperature, incubation time
   - **Replicates** — biological vs. technical definition (e.g. "n = 3 independent experiments performed on different days, each with 3 technical replicates")
   - **Statistical analysis** — exact test used (one-tailed / two-tailed, paired / unpaired), software (GraphPad Prism v.X.X), threshold (p<0.05)
   - **Reagent validation** — compound purity (HPLC / NMR / LC-MS), identity confirmation, source / catalog numbers
   - **Ethical approval** — IACUC / IRB if human/animal work
3. **Code & data availability**:
   - GitHub / Zenodo / figshare DOIs
   - License (MIT / Apache 2.0 / CC-BY)
   - Version tag (commit hash or release number)

---

### C.5 Limitations (separate section, mandatory for NMI / NCS)

**Required content** (from NCS editor guidance + NMI rejection-pattern catalogue):
- **Dataset bias** — known biases in training data (e.g. PDBbind over-represents high-affinity binders)
- **Generalization scope** — what the method does NOT generalize to (cross-target / cross-pocket / cross-modality)
- **Compute requirement** — honest accounting of resources (GPU hours, wall-clock)
- **Comparison fairness** — explicit statement of matched vs. unmatched baselines
- **Wet-lab limitations** (if applicable):
  - Single-laboratory validation (multi-lab replication not done)
  - Cell-line panel limited to one tissue / one cancer type
  - Single target (no selectivity panel)
  - Acute exposure only (no chronic / pharmacokinetic data)

---

### C.6 Supplementary / Extended Data

**Convention**:
- **NCS** allows ≤10 Extended Data items (figures + tables)
- **NMI** allows Extended Data figures and Supplementary Information
- **BiB** strongly encourages supplementary material; submitted as separate files

**Typical content**:
- Full benchmark tables (not in main text)
- Hyperparameter sensitivity (learning rate, batch size sweep)
- Ablation tables (component × component)
- Additional wet-lab data (raw sensorgrams, full dose-response curves, uncropped Western blots)
- Source data files (per-figure)
- Code repository link + DOI
- Compound characterisation (NMR / LC-MS spectra)

---

## Part D — Synthesis of what tier-1 expects from a wet-lab-validated SBDD paper

| Tier | Wet-lab requirement | Code/data availability | Sample size (typical) | Statistical bar |
|---|---|---|---|---|
| **Nature (flagship)** | **Mandatory** if any claim of "we designed a ligand / identified a target" | Required at submission (NOT post-acceptance) | n ≥ 3 biological replicates; cell-line panel ≥ 6 | Exact p-values; 95% CIs on potency |
| **Nat. Mach. Intell.** | Conditional — required if scientific discovery claimed | Required at submission | n ≥ 3 biological replicates for any wet-lab component | Statistical testing of performance differences; multiple datasets |
| **Nat. Comput. Sci.** | Conditional — required if external validity claimed | Required at submission; matched-comparison audit expected | n ≥ 3 biological replicates | Performance-cost tradeoffs reported |
| **Briefings in Bioinformatics** | Not required for review/tutorial; encouraged for methods paper | Required for reproducibility | n ≥ 3 for any wet-lab component | Standard methods-section statistical reporting |
| **J. Chem. Inf. Model.** | **Required** for methods applied to single targets | Required for reproducibility | n ≥ 3 biological replicates | Statistical comparison to reference compound (±0.5 log units) |

---

## Part E — Honest framing for our Lambda-Metal pipeline (party-twenty-spirit §3.5)

**What we CANNOT honestly claim** given that we are CPU-only and have no wet-lab access:
1. We have NOT measured IC50 / Kd / cellular uptake for any generated compound.
2. We have NOT synthesised any compound.
3. We have NOT performed X-ray crystallography or SPR.
4. We have NOT run a multi-lab validation.

**What we CAN claim with honest framing**:
1. Our Vina scores are in silico predictions ONLY, with the documented metal-coordination term absent (this is a known Vina limitation, NOT a method error).
2. Our QED values are Bickerton desirability for oral drugs, NOT a metallodrug ADMET predictor.
3. Our SA scores are the RDKit reference implementation, identical to the Tier-1 SBDD convention.
4. Our RMSD values are MCS-overlay (stricter than the Tier-1 full-ligand Kabsch convention); this is intentional for metallo-fragment cores.
5. Our cellular / in vivo / wet-lab metrics (if we add them) MUST be labelled "PREDICTED" not "MEASURED" — using ADMET predictors (e.g. Crippen logP + Hou pIC50) is computational proxy work, NOT wet-lab validation.

**Recommended §3.5 framing for the methods section**:
> "This pipeline is CPU-only and produces in silico predictions; no wet-lab validation is claimed. All potency / ADMET / cellular metrics are computational proxies, calibrated against public databases (ChEMBL, BindingDB, PDBbind) but not validated against our own laboratory measurements. Any future wet-lab validation is expected to follow the Nature Computational Science reporting bar: ≥3 biological replicates, exact n reported, Z'-factor ≥0.5 for cell assays, KD standard deviation <30% for SPR, and matched-comparison audit for any in silico vs. wet-lab comparison."

---

## Sources

- **NCS Submission Guide** (third-party synthesis): https://manusights.com/blog/nature-computational-science-submission-guide
- **NMI Submission Guide** (third-party synthesis): https://manusights.com/blog/nature-machine-intelligence-submission-guide
- **Nature Reporting Standards**: https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards
- **Nature Statistical Guidelines (Communications Sustainability)**: https://www.nature.com/commssustain/submit/statistical-guidelines
- **BiB Author Guidelines**: https://academic.oup.com/bib/pages/general-instructions
- **MitoReID paper** (Nat. Comput. Sci. 2024): https://www.nature.com/articles/s43588-024-00682-9
- **DiffSBDD paper** (Nat. Comput. Sci. 2024): https://doi.org/10.1038/s43588-024-00737-x
- **DeepBlock paper** (Nat. Comput. Sci. 2024): https://www.nature.com/articles/s43588-024-00718-0
- **Wang 2024 BiB SBDD review**: https://academic.oup.com/bib/article/25/4/bbae258/7689894
- **Lam 2024 BiB protein-LM review**: https://academic.oup.com/bib/article/25/6/bbae480 (DOI 10.1093/bib/bbae480)
- **Rag2Mol** (BiB 2024): https://academic.oup.com/bib/article/bbaf265 (DOI 10.1093/bib/bbaf265)
- **Gao 2024 "From Theory to Therapy"** (arXiv preprint of BiB-tier SBDD critique): https://arxiv.org/abs/2406.08980
- **Kunkel 2024 HTS384 NCI-60** (Cancer Research): https://aacrjournals.org/cancerres/article/84/15/2403/745480
- **Landrum & Riniker 2024 IC50 curation** (JCIM): https://pubs.acs.org/doi/10.1021/acs.jcim.4c00049
- **Adaptyv/Proteinbase binder competitions**: https://proteinbase.com/competitions
- **Cao 2022 miniprotein binders** (Nature Methods): https://www.nature.com/articles/s41587-022-01557-w
- **Chemgenomics 2025 functional-assay validation guide**: https://www.chemgenomics.com/posts/bridging-the-digital-and-biological-a-2025-guide-to-functional-assay-validation-for-computational-predictions-in-drug-discovery
- **Creative-Proteomics hit-validation**: https://www.creative-proteomics.com/mass-target/hit-validation.htm
- **MoB-IBT sample requirements**: https://www.mosbri.eu/?p=1298/