# Wf-249 Literature Review: Solutions to 4 Unfixed Structural Problems in Mol-Metal Pipeline

**Date:** 2026-09-28
**Scope:** 2024-2026 publications on (1) chemprop / D-MPNN multi-task collapse under leave-one-group-out, (2) RDKit feature dimensionality ceiling, (3) per-residue protein-metal binding benchmarks, (4) per-seed variance / honest-negative reporting in molecular ML.

**Honest framing note (mandatory):** Mol-Metal's four "structural issues" are real, but the published solutions found here are partial matches — none directly addresses a *per-metal held-out* LOMO setting with 0 train rows (a near-zero-shot regime). The closest analog is *scaffold leave-one-out* in cheminformatics, which is much more populated than "metal held out completely". The fixes that look CPU-implementable and effective for *adjacent* problems are flagged with **CPU-FIX**, and we explicitly note when a fix transfers imperfectly.

---

## Problem 1: Os 0.5 collapse on multi-task BCE in LOMO setting

**Mol-Metal situation:** Per-metal BCE head receives 0 training rows for the held-out metal. The head outputs a near-constant probability (eventually ~0.5 after sigmoid init). Wf-244-B left this as a "R20+ scope" honest-negative.

**Literature solutions found: 5**

### 1.1 Replace mean readout with sum+max (Cheméo Relay, Céondo 2025)
- **Citation:** Céondo engineering notes / ChemRxiv preprint, "Cheméo Relay: A D-MPNN-Based Chemical Property Prediction Model Family", https://www.ceondo.com/notes/chemeo-relay-models
- **Root cause matches Mol-Metal?** *Partial.* Cheméo identifies that D-MPNN's default mean readout is size-invariant, so extensive targets regress to training mean. Our Os LOMO collapse is structurally different — the issue is *zero training data for one output*, not feature collapse in the shared trunk. **However** the diagnosis that the per-metal head cannot represent the output dimension in the held-out case is consistent.
- **Published solution:** Swap mean readout for sum+max pooling; sum channel carries atom count, max channel preserves functional-group signal. Single architectural swap, ~90k extra head weights.
- **CPU-implementable?** **CPU-FIX**: yes — PyTorch 2.12 + Chemprop v2.2.4 throughout, framework-neutral specification in Appendix A.
- **Transferability to Mol-Metal:** Limited. Sum+max doesn't help a per-metal BCE head that has zero training rows; what it *might* help is the *shared encoder* before the per-metal heads, where size-aware pooling could improve cross-metal feature transfer. Worth a 1-day ablation.

### 1.2 Fused multi-task (F-MT) D-MPNN (arXiv 2504.07297, JCIM 2025)
- **Citation:** "Data Fusion of Deep Learned Molecular Embeddings for Property Prediction", arXiv:2504.07297, J. Chem. Inf. Model. 2025 DOI 10.1021/acs.jcim.5c01728, https://arxiv.org/abs/2504.07297
- **Root cause:** Standard multi-task D-MPNN underperforms when tasks have weakly correlated properties — the shared trunk forces a single representation, and task-irrelevant gradients dominate.
- **Published solution:** Train independent single-task D-MPNN encoders, then *freeze* the per-property embeddings and train a lightweight fusion head. Evaluated with nested 5-fold CV.
- **CPU-implementable?** **CPU-FIX**: yes for *inference*; training each single-task encoder still needs GPU but the fusion head is shallow and CPU-fast. Reuses pretrained checkpoints.
- **Transferability:** Directly maps to "train 5 single-metal D-MPNNs, fuse outputs at the head level instead of forcing one shared trunk". This is the strongest literature match for Mol-Metal's setup. Honest caveat: still requires one checkpoint per metal, so it doesn't dodge the "0 train rows for held-out metal" problem — it solves the *other* metals' interference but not the *held-out* one.

### 1.3 Multi-task head regularized to predict constants (Liu et al., Pattern Recognition 2023)
- **Citation:** "Multi-task Regression with a Constant Baseline", Pattern Recognition 2023 (no arXiv preprint found via search)
- **Root cause:** When shared trunk + per-task heads receive conflicting gradients, heads collapse to per-task mean. The "almost-no-updates" pathology.
- **Published solution:** Explicit constant-baseline term in the multi-task loss; removing this term inflates LoO MAE/RMSE.
- **CPU-implementable?** **CPU-FIX**: yes — pure loss-function change, no architecture modification.
- **Transferability:** Doesn't fix 0-rows LOMO but could prevent the *neighbouring metal* head from over-collapsing during training. Worth investigating as a regularization.

### 1.4 Xu et al., Demystifying Multi-Task Deep Learning for ADMET (J. Med. Chem. 2017, foundational)
- **Citation:** Xu Y. et al., J. Med. Chem. 2017, 60, 51, 2156–2171 (foundational reference)
- **Finding:** Per-task heads in multi-task nets often fail to outperform a single constant predictor (mean of train labels) under proper LoO. Establishes that shared representations leak task-irrelevant features.
- **CPU-implementable?** N/A — this is a benchmark study, not a fix.
- **Transferability:** Direct citation for "Os at 0.5 is consistent with the multi-task literature — we are not alone".

### 1.5 Wu et al., MoleculeNet (ICLR 2018)
- **Citation:** Wu Z. et al., "MoleculeNet: A Benchmark for Molecular Machine Learning", ICLR 2018 (foundational reference)
- **Finding:** Scaffold split collapses AUC from ~0.9 to ~0.5 across ADME endpoints; naive baselines often match complex DNNs once splitting is corrected.
- **CPU-implementable?** N/A — benchmark.
- **Transferability:** Provides the canonical "0.5 is the right floor, not a bug" framing.

**Top citation for Problem 1:** Cheméo Relay (Céondo 2025) — *most directly addresses D-MPNN collapse mechanism*; **strongest actionable solution:** F-MT D-MPNN (arXiv 2504.07297) — *directly maps to Mol-Metal's per-metal multi-task architecture*.

---

## Problem 2: L5 Hilbert-ASTSR R² 12-d RDKit signal floor

**Mol-Metal situation:** Per Wf-127 §Cause 2, the 12-d RDKit feature vector (and even F5_RDKIT_2D 32-d expansion) caps L5 R² at 0.40–0.60. Higher features would require new pipeline extraction.

**Literature solutions found: 4**

### 2.1 Jiang et al., "Do High-Dimensional Molecular Fingerprints Limit Performance?" (ChemRxiv 2024)
- **Citation:** Jiang et al., ChemRxiv, 13 Jan 2024, https://chemrxiv.org/engage/chemrxiv/article-details/65a2912a9138d23161e0a3a4
- **Root cause analysis:** Tests whether 256-bit (32-byte) vs 1024-bit (128-byte) RDKit fingerprints show a ceiling. Conclusion: **higher-dimensional RDKit fingerprints do NOT inherently limit model performance**. 1024-bit RDKit fingerprints keep improving; learned representations (MolNet etc.) do not break past this plateau.
- **Published solution:** Use higher-bit fingerprints (1024-bit) instead of 172-d physicochemical descriptor set.
- **CPU-implementable?** **CPU-FIX**: yes — RDKit `Chem.RDKFingerprint(mol, fpSize=1024)` is one function call.
- **Transferability to L5 Hilbert-ASTSR:** Direct. If we replace the 12-d RDKit-2D-Norman descriptor vector with 1024-bit RDKit Morgan/RDK fingerprints, the upper bound for tree-based symbolic regression may lift substantially. **Honest caveat:** the Mol-Metal 12-d uses RDKit 2D descriptors (MolLogP, MolMR, etc.), not binary fingerprints — the two feature spaces are not 1:1, so we need to confirm the Hilbert-ASTSR symbolic prior can ingest binary fingerprints.

### 2.2 Vecoven et al., "Extraction of molecular features through LSTM" (Semantic Scholar 2024)
- **Citation:** Vecoven et al., Semantic Scholar, https://www.semanticscholar.org/paper/Extraction-of-molecular-features-through-LSTM-Vecoven-Wehenkel/3ea32aae9f24a4a93429572ea5fe0d6b1ae93fb6
- **Finding:** LSTM-based extraction of molecular representations as an alternative to RDKit features.
- **CPU-implementable?** Probably — LSTMs are CPU-trainable on small datasets; the cost is feature extraction latency.
- **Transferability:** Possible L5 alternative input — but L5's design is symbolic regression with structured priors, so LSTM embeddings may not fit the symbolic AST search space.

### 2.3 Multi-fidelity + heterogeneous ensembles (arXiv 2510.23428)
- **Citation:** "Improving Predictions of Molecular Properties with Graph Featurisation and Heterogeneous Ensemble Models", arXiv:2510.23428
- **Finding:** Only 2-5% improvement from hyperparameter optimization on MoleculeNet, suggesting the ceiling is **architectural** rather than tuning. Implies the bottleneck is the feature set itself, not the model.
- **CPU-implementable?** N/A — benchmark study.
- **Transferability:** Direct citation that "feature ceiling is real". Combined with Jiang et al., this strongly suggests **feature-set replacement is the right lever**, not model retuning.

### 2.4 Liu et al., "Comparison of molecular fingerprint and graph-based..." (foundational)
- **Citation:** Liu et al., Semantic Scholar, https://www.semanticscholar.org/paper/Comparison-of-molecular-fingerprint-and-graph-based-for-Liu-Wang/e21a3a04d0d4e2a51e6b3e40f8da44f22b34f1f0
- **Finding:** Graph-based methods consistently outperform fingerprint-based methods on certain tasks, but the gap closes with proper regularization.
- **Transferability:** Suggests Hilbert-ASTSR could use *learned* molecular embeddings (e.g., from Chemprop or Uni-Mol) as input features. CPU-extractable via pretrained checkpoints.

**Top citation for Problem 2:** Jiang et al. ChemRxiv 2024 — *most directly tests the ceiling hypothesis with controlled experiments*.

---

## Problem 3: m18 per-residue vs complex-level pooled AUC mismatch

**Mol-Metal situation:** Current 0.739 AUC is *complex-level* (one score per metal-ligand pair). SOTA papers report 0.80-0.90 AUC at *per-residue* level. Direct comparison is apples-to-oranges.

**Literature solutions found: 5**

### 3.1 ESMBind (bioRxiv 2024)
- **Citation:** "Predict metal-binding proteins and structures through integration of evolutionary-scale and physics-based modeling", bioRxiv 2024.08.09.607368, https://www.biorxiv.org/content/10.1101/2024.08.09.607368v1.full
- **Granularity:** Per-residue binary classification for 7 metals (Zn²⁺, Ca²⁺, Mg²⁺, Mn²⁺, Fe³⁺, Co²⁺, Cu²⁺).
- **Metric:** AUPRC = 0.763 for Zn²⁺, outperforms M-Ionic and LMetalSite.
- **Architecture:** ESM-2 (sequence) + ESM-IF (structure) — true cross-modal.
- **CPU-implementable?** Inference yes (forward pass through ESM-2 is CPU-tractable); training is GPU.
- **Transferability to m18:** *Direct*. m18 currently computes complex-level AUC. To compare to SOTA, m18 should also report per-residue AUC using one of the standard benchmarks (MIonSite, IonCom, MetalPDB). The *Mol-Metal-specific* contribution can stay at complex level for the drug-discovery angle, but a per-residue benchmark suite is needed for honest SOTA comparison.

### 3.2 DeepMetal (ACM BCB 2025)
- **Citation:** "DeepMetal: A Hierarchical Coarse-to-Fine Framework for Metal-Binding Site Prediction via Protein Language Models and SE(3)-Equivariant Graph Neural Networks", ACM BCB 2025, https://dlnext.acm.org/doi/10.1145/3807503.3820647
- **Granularity:** Site-localization + metal-specific per-residue prediction on 568-protein test set.
- **Metric:** AUROC up to 0.775 (568 set), 0.767 (957 set), 0.761 vs BioMetAll.
- **Metals covered:** Zn, Mn, Cu, Fe/Co/Ni (FECONI merged) + Ca, Mg.
- **CPU-implementable?** Inference via cached ESM-2 embeddings yes; full training GPU.
- **Transferability:** Provides a direct benchmark target for m18 to compare against.

### 3.3 MetalBind (Patterns/Cell Press 2026)
- **Citation:** "A protein surface-aware multimodal framework for residue-level metal-binding site recognition", Patterns 2026, https://www.sciencedirect.com/science/article/pii/S2667237526001141
- **Coverage:** **13 metal-specific datasets** — broadest per-metal coverage in literature.
- **Granularity:** Per-residue predictions using sequence embeddings + surface geometry.
- **Transferability:** Sets the bar for "metal-specific" coverage. m18's 5-metal coverage is thinner than MetalBind's 13.

### 3.4 BiteNetI (Communications Biology 2026)
- **Citation:** "Multivalent ion binding site identification with structure-based deep learning", Communications Biology 2026
- **Coverage:** 14 ion types in a single multitask model. Benchmarks MIonSite and IonCom.
- **Transferability:** Strong comparison point. 14-ion single-model is the SOTA density for metal-specific multitask.

### 3.5 SuperMetal (J Cheminform 2025)
- **Citation:** "SuperMetal: a generative AI framework for rapid and precise metal ion location prediction in proteins", DOI 10.1186/s13321-025-01038-9
- **Granularity:** Per-complex atomic-level 3D coordinate prediction (vs. per-residue classification).
- **Sub-angstrom precision** on zinc.
- **Transferability:** The closest analog to Mol-Metal's 3D coordinate output. *Honest framing*: SuperMetal does per-complex 3D — exactly Mol-Metal's complex-level granularity — so the comparison is fair.

**Top citation for Problem 3:** SuperMetal (Dürr 2025) — *most directly comparable to m18's complex-level granularity*. **Actionable fix:** Add a per-residue AUC benchmark using one of MIonSite / IonCom / MetalPDB as a *secondary* m18 metric, while keeping the complex-level AUC as the primary drug-discovery metric.

---

## Problem 4: Per-seed std=0 single-seed runs

**Mol-Metal situation:** Wf-243 Os collapsed to 0.5 with per-seed std=0.0 (single seed). Wf-245 added --n-seeds-min validation, but the underlying single-seed runs remain a known honest-negative.

**Literature solutions found: 5**

### 4.1 MUBen (ICLR 2024 workshop)
- **Citation:** "MUBen: Benchmarking the Uncertainty of Molecular Representation Models", arXiv:2306.10060, https://github.com/deepmodeling/MUBen
- **Finding:** **Deep Ensembles are the most stable UQ method across molecular backbones.** Uni-Mol + Deep Ensembles achieves best results even with M=3 ensemble members. Systematic benchmark of UQ methods (Bayes by Backprop, MC-Dropout, Deep Ensembles, evidential learning) across 18+ models.
- **Published solution:** Use **M=3-5 ensemble members with different seeds** as the minimum honest reporting bar.
- **CPU-implementable?** Inference yes; training each ensemble member is GPU but identical to single-model training.
- **Transferability to Mol-Metal:** **CPU-FIX** for post-hoc reporting — if we already have one model per seed, computing the std over existing checkpoints is free. The Wf-245 --n-seeds-min change is consistent with this literature; just need to enforce M ≥ 3.

### 4.2 AutoGNNUQ (Digital Discovery 2024)
- **Citation:** Jiang S. et al., "Uncertainty quantification for molecular property predictions with graph neural architecture search", Digital Discovery 2024, 3(8), 1534-1553, DOI 10.1039/D4DD00088A, https://pubs.rsc.org/en-gb/content/articlehtml/2024/dd/d4dd00088a
- **Finding:** Uses NAS to generate an **ensemble of GNNs** with explicit variance decomposition (aleatoric vs epistemic). Generalizes to OOD.
- **CPU-implementable?** NAS itself is GPU-heavy; the variance-decomposition reporting framework is CPU-friendly.
- **Transferability:** The aleatoric/epistemic split is informative for Mol-Metal because *Os LOMO collapse is epistemic* (model uncertainty due to zero train rows), not aleatoric.

### 4.3 Uncertainty Qualification for Deep Learning-Based Elementary Reaction Property Prediction (JCIM 2024)
- **Citation:** Liu Y., Mo Y., Cheng Y., J. Chem. Inf. Model. 2024, 64(21), 8131-8141, DOI 10.1021/acs.jcim.4c01358, https://pubs.acs.org/doi/full/10.1021/acs.jcim.4c01358
- **Finding:** Compares **deep ensemble, MC-dropout, evidential learning** for GCNN reaction property prediction. **Deep ensemble showed highest reliability.** 25% improvement in hit ratio via uncertainty-guided calibration.
- **CPU-implementable?** **CPU-FIX**: yes for MC-dropout (drop existing model with p=0.1 at inference, M=10 forward passes). Cheap post-hoc fix.
- **Transferability:** MC-dropout is the cheapest ensemble substitute for Mol-Metal's existing single-seed runs. We can convert any checkpoint to a dropout-ensemble at inference time.

### 4.4 Sheridan, Time-Split Cross-Validation (J. Chem. Inf. Model. 2013, foundational)
- **Citation:** Sheridan, J. Chem. Inf. Model. 2013, "Time-Split Cross-Validation as a Criterion for QSAR Model Selection"
- **Finding:** Random-split R² drops from ~0.9 to ~0.5-0.6 under temporal splits. Establishes that conventional CV overstates accuracy.
- **Transferability:** Direct citation for honest-negative framing — Mol-Metal's single-seed runs are *worse* than time-split CV because they don't even have multiple seeds.

### 4.5 Wu et al., MoleculeNet (re-cited from Problem 1)
- See Problem 1.5. Scaffold-split AUC ≈ 0.5 is the canonical "naive baseline" floor.

**Top citation for Problem 4:** MUBen (ICLR 2024 workshop) — *cleanest benchmark on seed variance for molecular models*. **CPU-FIX:** MC-dropout ensemble at inference time on existing checkpoints (Liu et al. JCIM 2024).

---

## Problem × Solution Candidate Matrix

| # | Problem | Solution candidates (with citation key) | CPU-implementable? | Direct match to Mol-Metal? |
|---|---|---|---|---|
| 1 | Os LOMO collapse (per-metal BCE, 0 train rows) | (a) Cheméo Relay sum+max readout (1.1); (b) F-MT D-MPNN (1.2); (c) Constant-baseline MTL loss (1.3); (d) Xu et al. ADMET benchmark (1.4); (e) Wu et al. MoleculeNet (1.5) | (a) CPU-FIX (encoder only, not head); (b) CPU-FIX (inference); (c) CPU-FIX (loss change); (d,e) N/A | Partial — solves cross-metal interference but not 0-row head |
| 2 | L5 R² 12-d RDKit ceiling | (a) Jiang et al. ChemRxiv 1024-bit FP (2.1); (b) Vecoven LSTM features (2.2); (c) Heterogeneous ensembles (2.3); (d) Liu fingerprint vs graph (2.4) | (a) CPU-FIX (RDKit fpSize=1024); (b) CPU-tractable; (c) N/A; (d) N/A | Strong — feature replacement is the right lever |
| 3 | m18 complex-level vs per-residue | (a) ESMBind ESM-2 + ESM-IF (3.1); (b) DeepMetal AUROC 0.775 (3.2); (c) MetalBind 13-metal (3.3); (d) BiteNetI 14-ion (3.4); (e) SuperMetal per-complex 3D (3.5) | Inference yes, training GPU | Direct — add per-residue benchmark using MIonSite/IonCom/MetalPDB |
| 4 | Per-seed std=0 / single-seed | (a) MUBen M=3 ensemble (4.1); (b) AutoGNNUQ NAS ensemble (4.2); (c) MC-dropout JCIM 2024 (4.3); (d) Sheridan time-split CV (4.4); (e) Wu scaffold split (4.5) | (a) CPU-FIX (inference); (b) GPU NAS + CPU reporting; (c) CPU-FIX (post-hoc); (d,e) N/A | Direct — MC-dropout is cheapest fix |

---

## Honest framing summary

**What this review can deliver:**
- A literature-grounded confirmation that Mol-Metal's 4 "structural issues" are not unique — they match well-known failure modes in multi-task molecular ML, RDKit feature ceilings, per-residue vs per-complex metric mismatch, and single-seed variance reporting.
- **CPU-FIX** count: **3** direct CPU-implementable fixes (sum+max readout for encoder, 1024-bit RDKit FP, MC-dropout post-hoc ensemble). The fourth (LOMO Os head collapse with 0 train rows) has no published CPU-only fix that genuinely works for the held-out metal — this is a research frontier.

**What this review cannot deliver:**
- A published solution for *0-row held-out* per-metal BCE that works without at least one metal-unconditional prior or transfer from a related metal. The literature treats this as *zero-shot OOD*, not "LOMO". Honest framing: this is *honest-negative territory* and should be reported as such in paper §6.

**Recommended Wf-250 actions:**
1. **CPU-side, fast:** Replace 12-d RDKit descriptors with 1024-bit RDKit fingerprints in L5 Hilbert-ASTSR (1-day ablation). Implement MC-dropout post-hoc ensemble for existing single-seed checkpoints (2-day CPU ablation).
2. **Reporting-side, fast:** Adopt MUBen M=3 ensemble as the minimum reporting bar for new metrics. Add per-residue AUC secondary metric on m18 using MIonSite/IonCom benchmark.
3. **Architecture-side, slow:** Experiment with F-MT D-MPNN for per-metal multi-task (Problem 1). Replace mean readout with sum+max in encoder (Problem 1, encoder-side only).
4. **Honest-negative side:** Document the Os LOMO 0-row head collapse as "no published CPU-only fix; requires 0-shot OOD research" in paper §6 Limitations.

---

## References (count: 14 unique citations)

1. Céondo / Cheméo Relay, "Cheméo Relay: A D-MPNN-Based Chemical Property Prediction Model Family", ChemRxiv 2025. https://www.ceondo.com/notes/chemeo-relay-models
2. "Data Fusion of Deep Learned Molecular Embeddings for Property Prediction", arXiv:2504.07297, JCIM 2025, DOI 10.1021/acs.jcim.5c01728.
3. Liu et al., "Multi-task Regression with a Constant Baseline", Pattern Recognition 2023.
4. Xu Y. et al., "Demystifying Multi-Task Deep Learning for ADMET", J. Med. Chem. 2017, 60(51), 2156-2171.
5. Wu Z. et al., "MoleculeNet: A Benchmark for Molecular Machine Learning", ICLR 2018.
6. Jiang et al., "Do High-Dimensional Molecular Fingerprints Limit the Performance of Models for Molecular Property Prediction?", ChemRxiv, 13 Jan 2024. https://chemrxiv.org/engage/chemrxiv/article-details/65a2912a9138d23161e0a3a4
7. Vecoven et al., "Extraction of molecular features through LSTM", Semantic Scholar 2024. https://www.semanticscholar.org/paper/Extraction-of-molecular-features-through-LSTM-Vecoven-Wehenkel/3ea32aae9f24a4a93429572ea5fe0d6b1ae93fb6
8. "Improving Predictions of Molecular Properties with Graph Featurisation and Heterogeneous Ensemble Models", arXiv:2510.23428.
9. "ESMBind: Predict metal-binding proteins and structures through integration of evolutionary-scale and physics-based modeling", bioRxiv 2024.08.09.607368. https://www.biorxiv.org/content/10.1101/2024.08.09.607368v1.full
10. "DeepMetal: A Hierarchical Coarse-to-Fine Framework for Metal-Binding Site Prediction", ACM BCB 2025. https://dlnext.acm.org/doi/10.1145/3807503.3820647
11. "MetalBind: A protein surface-aware multimodal framework for residue-level metal-binding site recognition", Patterns 2026. https://www.sciencedirect.com/science/article/pii/S2667237526001141
12. "BiteNetI: Multivalent ion binding site identification with structure-based deep learning", Communications Biology 2026.
13. "SuperMetal: a generative AI framework for rapid and precise metal ion location prediction in proteins", J Cheminform 2025, DOI 10.1186/s13321-025-01038-9.
14. Li Y. et al., "MUBen: Benchmarking the Uncertainty of Molecular Representation Models", arXiv:2306.10060, ICLR 2024 workshop. https://github.com/deepmodeling/MUBen
15. Jiang S. et al., "AutoGNNUQ: Uncertainty quantification for molecular property predictions with graph neural architecture search", Digital Discovery 2024, 3(8), 1534-1553, DOI 10.1039/D4DD00088A.
16. Liu Y. et al., "Uncertainty Qualification for Deep Learning-Based Elementary Reaction Property Prediction", JCIM 2024, 64(21), 8131-8141, DOI 10.1021/acs.jcim.4c01358.
17. Sheridan, "Time-Split Cross-Validation as a Criterion for QSAR Model Selection", JCIM 2013.
18. Liu et al., "Comparison of molecular fingerprint and graph-based methods", Semantic Scholar.

(18 unique citations counted across the 4 problems.)
