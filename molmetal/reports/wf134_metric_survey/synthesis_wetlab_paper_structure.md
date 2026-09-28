# WF-134 — Synthesis: Tier-1 Wet-Lab Validation Standards for Lambda-Metal SBDD

**Date:** 2026-09-27
**Source data:** `lit_table_wetlab_paper_structure.md` + `.json`
**Scope:** 1-paragraph verdict per journal + 1-paragraph verdict per assay + actionable
honest-framing for §3.5 / §4 (party-twenty-spirit).

---

## 1. Per-journal verdicts

### 1.1 Nature Machine Intelligence (NMI)

**Verdict:** NMI requires wet-lab validation **only if** the paper claims a scientific discovery (e.g. "we designed a binder", "we identified a target"); pure-benchmark superiority without scientific consequence is desk-rejected. The reproducibility bar is strict: code/data MUST be open-sourced at submission (NOT post-acceptance), random seeds fixed, compute budgets reported. For our pipeline, NMI is **out of scope** — we have no wet-lab access and our computational contribution is incremental, not the kind of "interpretive insight" NMI expects. **Action:** mark NMI as "submission bar exists but our pipeline does not meet it; R20+ if we ever do wet-lab validation".

### 1.2 Nature Computational Science (NCS)

**Verdict:** NCS is the most realistic tier-1 venue for our pipeline, **but** with caveats. NCS does not strictly require wet-lab work for a pure-method paper; it does require (a) computation as the reusable scientific advance (not a black-box), (b) matched-comparison audit, (c) generalization tested across ≥2 tasks/domains, (d) failure cases exposed. The 2024 MitoReID paper (s43588-024-00682-9) is the in-house example of a paper that combined computational method with in vitro COX-2 inhibition validation — that paper passed NCS bar. **Action:** if we ever aim for NCS, the cleanest path is (a) claim a method contribution (not a discovery), (b) report matched-comparison audit against ≥3 baselines with compute budgets, (c) show generalization across ≥2 metal targets / ≥2 pockets, (d) explicitly label failure cases. Our current Wf-124 P3 multi-task D-MPNN is a step in this direction.

### 1.3 Briefings in Bioinformatics (BiB)

**Verdict:** BiB is the most accessible tier-1 venue for our pipeline. BiB accepts both reviews and methodological-advance articles; original wet-lab validation is NOT required for review/tutorial submissions. The 2024 BiB SBDD reviews (Wang bbae258, Lam bbae480, Rag2Mol bbaf265) all critique the field for over-reliance on Vina and lack of wet-lab — which is itself a publishable critique for us. **Action:** a BiB review/tutorial on "the metallodrug SBDD evaluation gap" (Vina underestimates Pt-coordination, QED mis-calibrated for metals, Bickerton desirability ≠ metallodrug ADMET) fits our Wf-134 scope and our honest-framing constraints directly.

### 1.4 Journal of Chemical Information and Modeling (JCIM)

**Verdict:** JCIM is the tier-2 venue with the **MOST EXPLICIT wet-lab requirement**: docking methods without sufficient experimental validation are rejected. Landrum & Riniker 2024 (64(5):1560-1567, DOI 10.1021/acs.jcim.4c00049) makes the validation bar explicit by quantifying how IC50 noise accumulates from different assays (~65% of points differ >0.3 log units pre-curation). **Action:** if we publish in JCIM, we MUST either (a) include our own wet-lab validation (which we cannot) OR (b) submit only review/tutorial articles (which BiB already covers) OR (c) explicitly limit our computational claim to "in silico metric framework, not validated against our own wet-lab".

---

## 2. Per-assay verdicts

### 2.1 IC50

**Tier-1 minimum:** ≥3 biological replicates, ≥3 technical replicates per concentration point, 4-parameter logistic (Hill) curve fit with bottom/top/Hill slope/IC50 reported, 95% CI on log(IC50), Z'-factor ≥0.5, signal:background ≥5, CV ≤10%. Reference compound must be run in parallel with agreement to ±0.5 log units (Landrum 2024). **For us:** our `pic50_predictor.py` (Hou descriptors + Crippen logP) is a computational proxy, NOT an IC50 measurement. We cannot honestly report IC50 in any tier-1 venue.

### 2.2 Kd / KD (SPR / ITC / BLI)

**Tier-1 minimum:** ≥2 independently prepared protein batches, 1:1 Langmuir fit with ka/kd/KD + χ² + Rmax, residuals random, KD SD <30%, orthogonal confirmation (ITC or DSF/CETSA). Adaptyv/Proteinbase 2024-2026 standardised SPR protocol is the modern benchmark (EGFR best 1.21 nM, RBX1 best 23.7 nM). **For us:** we have no protein expression or SPR access. Any future "binding affinity" claims must be labelled PREDICTED (e.g. via Boltz affinity or ΔG from MM-GBSA) and never MEASURED.

### 2.3 Cellular uptake / functional cellular assay

**Tier-1 minimum:** ≥3 biological replicates, ≥6 cell lines (ideally NCI-60 sub-panel), CellTiter-Glo luminescence (modern 2024 standard), Z'-factor ≥0.5 (HTS384 average 0.82), signal:background ≥5 (HTS384 average 89), CV ≤10% (HTS384 average 5.2%), 72h exposure. Reference compound run in parallel. **For us:** no cellular access. Any future cellular claims MUST be labelled PREDICTED via ADMET/QSAR models.

### 2.4 Direct binding (X-ray crystallography / cryo-EM / HDX-MS)

**Tier-1 minimum:** ≤2.5 Å resolution, MolProbity validation (Ramachandran, rotamers, clashscore), PDB validation report. Designed pose reproduced within RMSD ≤1.0 Å; key H-bonds observed in 2Fo-Fc at ≥1.0 σ. **For us:** no crystallography access. We can only report **predicted** binding modes (Vina, Boltz), never experimental confirmation.

---

## 3. Per-section conventions (what goes where)

- **Introduction (~600-1000 words, NO subheadings):** problem framing → prior work limitations → our contribution → significance. Do NOT mix results into the intro.
- **Results (~2000-3000 words):** computational results → wet-lab results (if any) → joint analysis → failure cases (NCS explicitly requires failure cases). Use dot plots with mean overlay, NOT bar charts (Nature 2023+ convention). Report exact n per experiment. Report exact p-values.
- **Discussion (~600-1200 words, NO subheadings for NCS):** interpretation → comparison to prior work → mechanism → clinical/translational implications → limitations.
- **Methods / Online Methods:** computational (dataset + split + architecture + training + compute + seeds + software) + wet-lab (protocol + replicates + statistics + reagent validation + ethical approval) + code/data (GitHub/Zenodo DOIs + license + version tag).
- **Limitations (MANDATORY for NMI / NCS):** dataset bias + generalization scope + compute requirement + comparison fairness + wet-lab limitations.
- **Supplementary / Extended Data:** full benchmark tables + hyperparameter sensitivity + ablations + raw wet-lab (sensorgrams, dose-response curves, uncropped Westerns) + source data files + code DOI + compound characterisation (NMR/LC-MS).

---

## 4. Honest-framing punch-list for §3.5 / §4 (party-twenty-spirit)

1. **Label everything honestly.** All potency / ADMET / cellular metrics in our pipeline are **PREDICTED**, not measured. Crippen logP + Hou pIC50 descriptors, Bickerton QED, Ertl SA, AutoDock Vina scores, Boltz affinity — all computational proxies. We have NOT synthesised any compound, run any wet-lab assay, or obtained any experimental binding affinity.

2. **Calibration honesty.** QED is calibrated for **oral drugs**, not metallodrugs (Pt coordination is not in the ALERTS list). Vina has no metal-coordination term and **systematically underestimates** our metallodrug binding. SA scores use the RDKit reference implementation unchanged (identical to Tier-1 convention). MCS-overlay RMSD is **stricter** than the Tier-1 full-ligand Kabsch convention; intentional for metallo-fragment cores.

3. **What tier-1 actually requires vs. what we can claim.**
   - NMI: requires wet-lab if scientific discovery claimed → we have no wet-lab → NMI out of scope.
   - NCS: requires matched-comparison audit + generalization ≥2 tasks + failure cases → we CAN meet this for our method contribution if we frame honestly.
   - BiB: requires reviews/tutorials with cutting-edge overviews → we CAN submit a review on the metallodrug SBDD evaluation gap.
   - JCIM: requires wet-lab for single-target methods → we cannot meet this; JCIM out of scope unless we do review/tutorial.

4. **Recommended §3.5 methods-section sentence:** "This pipeline is CPU-only and produces in silico predictions; no wet-lab validation is claimed. All potency / ADMET / cellular metrics are computational proxies, calibrated against public databases (ChEMBL, BindingDB, PDBbind) but not validated against our own laboratory measurements. Any future wet-lab validation is expected to follow the Nature Computational Science reporting bar: ≥3 biological replicates, exact n reported, Z'-factor ≥0.5 for cell assays, KD standard deviation <30% for SPR, and matched-comparison audit for any in silico vs. wet-lab comparison."

5. **R17+ scope markers (not R15/Wf-134 scope):** wet-lab validation of any compound is out of scope for our current CPU-only computational pipeline. If we ever do wet-lab (multi-lab collaboration, contract research org), the standards documented here are the bar — IC50 / Kd / cellular / X-ray all have explicit sample-size and replicate requirements that would need to be budgeted and resourced.

---

## 5. Submission-venue recommendation (actionable)

| Venue | Submit now? | If yes, what to claim | If later, what's needed |
|---|---|---|---|
| **NMI** | NO | n/a | Wet-lab validation; interpretive insight (not black-box accelerator) |
| **NCS** | CONDITIONAL | Method contribution (not discovery); matched-comparison audit; ≥2 tasks generalization | Code/data open-sourced; failure cases; performance-cost tradeoffs |
| **BiB** | YES (best near-term fit) | Review/tutorial on "the metallodrug SBDD evaluation gap" (Vina, QED, Bickerton, MCS-vs-Kabsch) | n/a — review/tutorial format fits |
| **JCIM** | NO (without wet-lab) | n/a | Wet-lab for single-target methods OR limit to review/tutorial |

**Top recommendation:** BiB review/tutorial submission is the cleanest near-term fit. The field critique documented in this survey (Vina underestimates metallodrugs, QED mis-calibrated for metals, etc.) is itself a publishable contribution.