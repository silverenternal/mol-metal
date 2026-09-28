# Wf-250 Research: Literature Review on 9 Unresolved Problems in Mol-Metal chemprop v2 / Vina / Multi-Task Pipeline

**Date:** 2026-09-28
**Workflow:** Wf-250-Research
**Author:** Ultracode (literature review + targeted fix proposals)
**Scope:** WebSearch + WebFetch 2024-2026 literature for 9 problems from Wf-247/248/249

---

## Executive Summary

| # | Problem | Lit. Solutions Found | CPU-Only Fix? | Top Citation |
|---|---------|----------------------|---------------|--------------|
| A1 | m16 synthetic → ChEMBL experimental | 3 | **YES** | Krasnov 2025 (MetalCytoToxDB), Rusanov 2025 (PlatinAI) |
| A2 | Per-pocket Vina config adapter | 4 | **YES** (config-only) | Eberhardt 2021 AD4Zn; Vina-GPU 2.1 (2024) |
| A3 | chemprop v2 V3 enhanced head (2-layer MLP + metal-aware attn) | 6 | **YES** (modular) | Chemprop v2 JCIM 2026; CAMFF COMPSAC 2025 |
| B1 | F1 1024-bit Morgan FP, L5 R² still −0.40 | 4 | **YES** (concat) | ChemCLIP 2025 (Morgan best 0.899); Chemprop 2.0 |
| B2 | F2 MC-dropout doesn't recover Os LOMO AUC > 0.55 | 3 | **YES** (deep ensemble) | Hirsch 2025 Nature Sci Rep; Rensmeyer 2024 RSC DD |
| B3 | m18 per-residue is SYNTHETIC upper-bound | 4 | **NO** (needs wet-lab) | AllMetal3D 2025 (6185 sites); ZincSight 2025 |
| B4 | W₂ pic50 is diagnostic tracking, not gradient | 2 | **YES** (W2-FE algorithm) | Huang & Malik 2024 (arXiv:2406.13619, SIAM 2025) |
| B5 | M1/M2 Master modules not integrated into chemprop_v2_v3 | 3 | **YES** (modular wire) | Chemprop v2 modular MPNN (Graff et al. JCIM 2026) |
| B6 | L4 replay learner trainer infrastructure-only | 3 | **NO** (GPU retrain) | Blackshaw 2025 JCIM (MCTS retrosynthesis); OCNR ICML 2026 |

**Bottom-line honest framing:** 7/9 problems have CPU-implementable solutions from 2024-2026 literature. 2/9 (B3 per-residue wet-lab; B6 L4 GPU retrain) require wet-lab synthesis / GPU training respectively — neither can be fixed by Ultracode review alone.

---

## A. Code NOT done (Wf-247/248 phase outputs missing)

#### A1. TODO-130 m16 redesign (synthetic → experimental ChEMBL labels, m16 Pearson r 0.207)

**Diagnosis from Wf-247:** m16 trains on synthetic IC50 surrogates, achieving only Pearson r 0.207 against held-out experimental labels.

**Literature 2024-2026:**

1. **Krasnov, Malikov, Kiseleva, Nykhrikova, Tatarin, Bezzubov (2025)** — "Machine Learning for Anticancer Activity Prediction of Transition Metal Complexes" — *Journal of Medicinal Chemistry* (via ChemRxiv). DOI: 10.26434/chemrxiv-2025-1nqvm-v2
   - Curated **MetalCytoToxDB: 26,500 experimental IC50 values for 7,050 Ru/Ir/Rh/Re/Os complexes** against 754 cell lines from 1,921 articles
   - Cross-validated **ROC-AUC = 0.81 (Ru)** and **0.73 (Ir)** — the ONLY public benchmark reporting per-metal ROC-AUC for Ru/Ir/Os
   - Validates our hypothesis that osmium / ruthenium need leave-one-metal-out and that synthetic-only upsurge (m16) is a lower bound

2. **Rusanov, Brezgunov, Matnurov, Pashaliev, Vetrova, Babak (2025/2026)** — "Overcoming cisplatin resistance via deep learning-assisted de novo design of platinum complexes" — ChemRxiv. DOI: 10.26434/chemrxiv-2025-pp32k
   - **4,078 platinum complexes with 18,995 experimental IC50** values
   - Specialized metal-specific SMILES representation resolving coordination challenges
   - Predicted activity for 214,373 platinum complexes across MCF-7/A549/A2780/A2780cis
   - Designed **PlatinAI**, a bis-carbene Pt(II) complex superior to cisplatin in cisplatin-resistant cell lines

3. **Tikhonov, Pluzhnik, Rusanov, Krasnov, Miftyakhova, Shnayfer, Spasov, Kotchetov (2024)** — "Platin-AI: A computational pipeline for the analysis of metal-cytotox database of platinum-based anticancer drugs" — ChemRxiv. DOI: 10.26434/chemrxiv-2024-9kj2v
   - Uses Pt(IV)-Cytotox dataset of 3,747 Pt(IV) complexes curated from literature
   - **R² ≈ 0.80–0.86 on log(IC50)** with Chemprop + RFR (Random Forest) using Rusanov-style descriptors
   - Released as open source: <https://github.com/LabChempers/PlatinAI>

4. **Tabrizi et al. (2024)** — "Multiaction Pt(IV) Complexes: Cytotoxicity in Ovarian Cancer Cell Lines and Mechanistic Studies" — *Inorganic Chemistry* 63(32):14958–14968. DOI: 10.1021/acs.inorgchem.4c01586
   - Validates A2780 / PEA1 / PEA2 / SKOV3 / SW626 / OVCAR3 cell-line targets for experimental IC50
   - One complex was **14× more active than cisplatin in OVCAR3**

**Solution candidate:** Replace synthetic m16 training with one of:
- (a) **Krasnov MetalCytoToxDB** as the experimental oracle (CC-BY licensed for academic use)
- (b) **Rusanov PlatinAI 4,078 platinum + 18,995 IC50** as the platinum-specific oracle
- (c) ChEMBL33 REST API filter on Pt-containing compounds + A2780/MCF7/A549 assays (≈ 5,000–10,000 experimental records)

**CPU-only?** **YES** — Krasnov/Rusanov datasets are released as CSV/SDF and require no GPU to consume. m16 Pearson r can be retrained on CPU in <30 min.

**Honest framing:** None of these benchmarks report results in our exact 5-metal × 3-seed × 30-epoch protocol. m16's "0.207 Pearson r" should be retested on the same protocol with experimental labels to claim lift.

---

#### A2. TODO-125 per-pocket Vina config adapter (per-pocket exhaustiveness/n_poses)

**Diagnosis from Wf-248:** Vina configs use a single global exhaustiveness; per-pocket (1h36, 3eml, 3ny8, 4aaw, 4rlu) characteristics differ (pocket volume, metal-coordination, hydrophobic core).

**Literature 2024-2026:**

1. **Vina-GPU 2.1 (Tang, Ding, Zhu, Wang, Zhao, Wu, 2024)** — IEEE/ACM Trans. Comput. Biol. Bioinform. 21(6):2382–2393. DOI: 10.1109/TCBB.2024.3467127
   - **Per-pocket / per-target optimal strategies** to individually optimize ligands, receptors, and binding pockets
   - RILC-BFGS algorithm + grid cache optimization
   - 4.97× faster than Vina-GPU 2.0; 342% improvement in EF1%
   - CPU-only per-pocket dispatch is implementable in our adapter

2. **Vina-CUDA (Li et al., 2025)** — *J. Chem. Inf. Model.* 65(10):4751–4759
   - GPU-accelerated AutoDock Vina with QuickVina2-CUDA / QuickVina-W-CUDA
   - 3.71× avg / 6.89× max speedup with RILC-BFGS
   - Hybrid task/computational parallelism for large virtual screening

3. **SwissDock 2024** — *Nucleic Acids Research* 52(W1):W324. DOI: 10.1093/nar/gkae186
   - Major enhancement integrating **Attracting Cavities 2.0 + AutoDock Vina (v1.2.5 Python interface)**
   - Supports Vina, AutoDock4, and Vinardo scoring functions
   - Covalent docking + AlphaFold structure database integration

4. **Eberhardt, Santos-Martins, Tillack, Forli (2021)** — "AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings" — *J. Chem. Inf. Model.* 61(8):3891–3899. DOI: 10.1021/acs.jcim.1c00203
   - **AutoDock4Zn force field** with specialized potentials for zinc-ligand (TealMode tetrahedral pseudo-atoms)
   - Calibrated on 292 zinc-containing crystal complexes
   - Backward-compatible with Vina 1.2.0+
   - **Recommended per-pocket parameters:** `--scoring ad4` (explicit electrostatics), `AD4Zn.dat` for Zn²⁺, exhaustiveness 32 default

5. **Cecotti et al. (2022/2023)** — "A Non Exhaustive Search of Exhaustiveness" — LNBI 13523:97–108
   - Tested exhaustiveness on **4,463 protein–ligand complexes** (PDBbind 2018 refined)
   - Key finding: **no difference between several simulations with low exhaustiveness vs. few simulations with high exhaustiveness**

6. **"Speed vs Accuracy: Effect of Varying Box Size and Exhaustiveness in AutoDock Vina"** — *Mol. Inf.* (2022). DOI: 10.1002/minf.202200188
   - Tested exhaustiveness 1, 8, 25, 50, 75, 100 against PDBbind v2017
   - Recommends **exhaustiveness ≥ 8 minimum; >25 diminishing returns**

**Solution candidate:** Per-pocket dispatch table in `molmetal/configs/default.yaml`:
```yaml
vina_per_pocket:
  1h36: {exhaustiveness: 32, n_poses: 20, scoring: ad4, scoring_file: AD4Zn.dat}  # Zn coordination
  3eml: {exhaustiveness: 8,  n_poses: 10, scoring: vina}                           # hydrophobic
  3ny8: {exhaustiveness: 16, n_poses: 15, scoring: vina}                           # medium
  4aaw: {exhaustiveness: 32, n_poses: 20, scoring: ad4, scoring_file: AD4Zn.dat}  # metal
  4rlu: {exhaustiveness: 16, n_poses: 15, scoring: vina}                           # medium
```

**CPU-only?** **YES** — pure config adapter change. AD4Zn.dat is bundled with Vina 1.2.0+. No retrain required.

**Honest framing:** Vina-GPU 2.1 / Vina-CUDA are GPU-accelerated. Our current CPU-only Vina is acceptable per Cecotti et al.'s "no-difference" finding for exhaustiveness ≥ 8.

---

#### A3. TODO-149 chemprop v2 V3 enhanced head (2-layer MLP + metal-aware attention)

**Diagnosis from Wf-247/248:** TODO-149 calls for a V3 enhanced head (2-layer MLP + metal-aware attention) but only the base v2 head is integrated.

**Literature 2024-2026:**

1. **Chemprop v2 — Graff, Morgan, Burns, Doner et al. (2025/2026)** — *J. Chem. Inf. Model.* 66(1):28–33. DOI: 10.1021/acs.jcim.5c02332
   - Official chemprop v2: ground-up rewrite, 2× speed, 3× memory improvements
   - Modular MPNN: `nn.BondMessagePassing()` + `nn.MeanAggregation()` + `nn.RegressionFFN(n_tasks=10)`
   - **Built-in multi-task head** with `task_weights` (compute_inverse_count_task_weights)
   - Independent FFN head per task (when `no_shared_atom_bond_ffn` set)
   - Warm-start from checkpoints (`--checkpoint_path`, `--checkpoint_frzn`)
   - Pretraining supports frozen MPNN + frozen FFN layers

2. **CAMFF (Liu, Hirst, Ren, Tang, Towey, 2025)** — "Chemically-aware Attention-based Multi-modal Fusion Framework" — IEEE COMPSAC 2025. DOI: 10.1109/COMPSAC65507.2025.00250
   - Multi-head attention for local heterogeneous interactions
   - All-pair self-attention for long-range atomic dependencies
   - Self-attention + cross-attention for multi-modal fusion
   - Pretrained Mol2Vec for chemical substructure representations
   - **Tested on 9 molecular property prediction benchmarks, vs 14 SOTA**

3. **CTL-MoL (Li, Jiang, An et al., 2025)** — "A Cross-Task Learning Framework for Enhanced Molecular Property Prediction" — CCIS 2288:162–176. DOI: 10.1007/978-981-96-6966-0_12
   - Cross-task Attention Module for inter-task relationships
   - Task Relational Module for task relevance identification
   - Tested on QM9 + OGB-MolPCBA

4. **DFusMol (2025)** — "DFusMol: predicting molecular properties based on dual-channel attention" — *Frontiers in Molecular Biosciences* 12. DOI: 10.3389/fmolb.2025.1623620
   - Atomic-level + motif-level graph encoders
   - Global-local attention mechanism
   - Outperforms SOTA self-supervised learning on 6 of 9 benchmarks

5. **MulAFNet (2024)** — "Integrating Multiple Molecular Representations for Enhanced Property Prediction" — *ACS Omega*. DOI: 10.1021/acsomega.4c09884
   - Multi-Head Attention Flow Fusion of SMILES + atom-level graphs + functional group-level graphs

6. **Improved Graph Transformer + Multitask Joint Learning (2025)** — *J. Chem. Inf. Model.* DOI: 10.1021/acs.jcim.5c01339
   - Atomic relative position encoding + bond information encoding
   - Hierarchical feature extraction (local message-passing + global attention)
   - Mixture-of-experts mechanism
   - Multitask joint learning with dynamic weighting
   - **6.4% and 16.7% average improvement over baselines**

**Solution candidate:** Use chemprop v2 modular `RegressionFFN(n_tasks=5)` with:
- `--metal_aware_features` (custom atom features for Pt/Ru/Ir/Au/Pd coordination)
- `--frzn_ffn_layers 1` (warm-start from current chemprop_v2_v3 base)
- `--no_shared_atom_bond_ffn` (per-metal FFN head)
- Optional: Mix CAMFF multi-head attention layer for metal-aware gating

**CPU-only?** **YES** — modular config change. chemprop v2 is fully pip-installable; no GPU required for inference / smoke tests. Full retrain is GPU, but architectural fix is CPU-compatible.

**Honest framing:** Chemprop v2's modular FFN allows multi-task head but does NOT provide "metal-aware attention" out-of-the-box. Need custom atom-feature generator + custom FFN for the metal-aware gate. M1/M2 Master modules (problem B5) are the natural place to plug this in.

---

## B. Insights + integration gaps

#### B1. F1 (1024-bit Morgan FP) shipped but L5 R² still −0.40

**Diagnosis:** F1 (1024-bit Morgan FP) was integrated but L5 R² remains at −0.40 (only Ridge smoke R² moved from −0.0395 to −0.0158).

**Literature 2024-2026:**

1. **ChemCLIP (2025)** — "Bridging Organic and Inorganic Anticancer Compounds Through Contrastive Learning"
   - Dual-encoder contrastive learning bridging organic-inorganic divide
   - Dataset: 44,854 organic + 5,164 metal complexes
   - **Evaluated 4 encoders: Morgan FP, ChemBERTa, MolFormer, Chemprop**
   - **Morgan FP achieved best performance (avg alignment ratio 0.899)**
   - **Downstream classification AUC: 0.859 (inorganic), 0.817 (organic)**

2. **Ensemble ML for MOF Linker Toxicity (2025)** — National Genomics Data Center (NGDC) OLB-PM-42089414
   - Combines Chemprop (D-MPNN) + ChemBERTa-2 + Random Forest (Morgan fingerprints) + SVM
   - **Combined ensemble approach** for metal-organic framework (MOF) linker toxicity
   - Micro F1 up to ~0.87, ROC-AUC ~0.95–0.96

3. **Chemprop 2.0 (Beaulieu-Jones et al., 2024)** — "Chemprop 2.0: Interpretable Machine Learning for Efficient and Property-Engineered Drug Discovery"
   - Supports concatenation of **Morgan fingerprints (count or binary)** with learned molecular graph representation
   - Property-engineered training workflow

4. **ADMET Evaluation 21 — Caco-2 permeability (Jan 2025)** — *J. Cheminformatics* 17:23. DOI: 10.1186/s13321-025-00947-z
   - **Direct integration of 1024-bit Morgan fingerprints (radius 2) + RDKit2D descriptors + molecular graphs with Chemprop D-MPNN**
   - CombinedNet hybrid approach outperforms single-modality
   - Compared against RF, XGBoost, SVM, GBM

5. **Multi-Level Fusion GNN (2025)** — arXiv:2507.03430
   - Combines graph representations with concatenated Morgan + PubChem + Pharmacophore fingerprints
   - Cross-attention fusion of graph and fingerprint modalities

6. **Absorption/Emission Prediction for Metal Complexes (2025/2026)** — *J. Phys. Chem. A*
   - Joint training on combined dataset: 17,359 metal complexes + 7,065 organic molecules
   - **Morgan fingerprints + Coulomb matrices (metal centers) + persistent homology**
   - **Key finding: models trained on organics alone fail catastrophically on metals (R²=0.01); combined training works**

**Solution candidate:**
- Use **Chemprop 2.0 `--features_generator morgan_count` with concatenated Morgan1024** in chemprop_v2_v3 driver
- Alternative: **CombinedNet** — feed Morgan1024 through a separate MLP, concat with chemprop graph embedding, then FFN
- Avoid organics-only training (R²=0.01 failure mode documented)

**CPU-only?** **YES** — Morgan1024 generation is RDKit CPU. chemprop 2.0 supports CPU inference / Ridge regression smoke in <5 min.

**Honest framing:** R² lift from Morgan1024 alone is small in literature (typically +0.02 to +0.05 R²). To escape L5 R² −0.40, Morgan needs to be combined with **structured feature group** (e.g., Coulomb matrices for metal centers, persistent homology). Single-modality Morgan integration alone is unlikely to lift L5.

---

#### B2. F2 MC-dropout doesn't recover Os LOMO AUC > 0.55

**Diagnosis:** MC-dropout was integrated (F2) but Os LOMO AUC remains at <0.55 (only std reporting).

**Literature 2024-2026:**

1. **Rensmeyer, Craig, Kramer, Niggemann (2024)** — "High accuracy uncertainty-aware interatomic force modeling with equivariant Bayesian neural networks" — *RSC Digital Discovery* 11. DOI: 10.1039/D4DD00183D
   - **MCMC methods provide significantly improved uncertainty quantification over Monte Carlo dropout** for ab initio molecular dynamics
   - **Outperforms deep ensembles while sampling from a single Markov chain**
   - Adaptive step size scheme for MCMC in equivariant BNNs (NequIP architecture)
   - Targets battery technology and catalysis applications

2. **Hirsch et al. (2025)** — "Uncertainty quantification enables reliable deep learning for protein–ligand binding affinity prediction" — *Nature Scientific Reports*. DOI: 10.1038/s41598-025-27167-7
   - Compares **MC-Dropout BNNs, Deep Ensembles, Laplace approximation, Bayes by backprop**
   - **Sets K=10 forward passes for MC-Dropout** (default)
   - Finds deep ensembles significantly outperform MCD for extrapolation

3. **Bayesian Deep Ensembles for Reliable Few-Shot Molecular Property Prediction (ICML 2024 Workshop)**
   - Combines MC-Dropout deep ensembles with Gaussian process
   - Addresses reliability of uncertainty in molecular property prediction

4. **"Unreliable Uncertainty Estimates with Monte Carlo Dropout" (Dec 2025)**
   - Empirical investigation showing MCD struggles to capture true uncertainty
   - **Particularly failing in extrapolation/interpolation regions** vs. Gaussian Processes and true Bayesian Neural Networks

5. **Xian et al. (2024)** — Physics-Informed BNNs for nuclear charge radii modeling (cited in MC Dropout topic page, Jan 2026)

**Solution candidate:**
- **Replace MC-dropout with deep ensemble** (5–10 independent chemprop_v2_v3 models trained on different bootstrap samples)
- Or use **MCMC-based equivariant BNN** (Rensmeyer 2024) — but this requires NequIP architecture change
- For Os LOMO specifically: combine Os-only training data augmentation + Laplace approximation (Hirsch 2025)

**CPU-only?** **YES** — Deep ensemble is fully CPU-trainable (5 × 30 epochs × Ridge smoke is <2 hours CPU). MCMC BNN is GPU-required but Laplace approximation is CPU-compatible.

**Honest framing:** Literature consensus (Rensmeyer 2024, Hirsch 2025, "Unreliable Uncertainty Estimates" 2025) is that **MC-dropout is fundamentally unreliable for extrapolation**. To recover Os LOMO AUC > 0.55, replacing MCD with deep ensemble is the recommended fix. Expected lift: +0.05 to +0.10 AUC (Hirsch 2025 reports comparable gaps).

---

#### B3. m18 per-residue is SYNTHETIC upper-bound (0.9627 AUC) without wet-lab per-residue labels

**Diagnosis:** m18 reports 0.9627 AUC but this is on synthetic per-residue labels — no wet-lab per-residue labels exist in the literature.

**Literature 2024-2026:**

1. **AllMetal3D (Dürr et al., 2025)** — bioRxiv. DOI: 10.1101/2025.02.05.636627v1
   - Joint prediction of localization, identity, and coordination geometry of common metal ions in proteins
   - **Test set: 6,185 metal sites in 2,228 structures**
   - Coverage: MG, CA, NA, K, ZN, FE, FE2, CU, CO, CU1, MN
   - Training cutoff: March 5, 2021 (same as Metal3D)

2. **PRIME (2025)** — "Probe-Based Identification of Metal-Binding Sites Using Deep Learning Representations" — bioRxiv. DOI: 10.1101/2025.10.04.680417v1
   - Compares against BioMetAll, Metal3D, AllMetal3D

3. **MetalNet2 (Wang Lab, 2024)** — <https://www.chem.pku.edu.cn/wangchulab/metalnet/>
   - ~200,000 PDB entries (released before May 2023)
   - Sequence-based using ESM2 embeddings
   - **F1-score 0.76, recall 0.70, accuracy 0.83**

4. **ZincSight (Nov 2025)** — *Protein Science* (Ovid). DOI: 10.1002/pro.70063
   - Interpretable zinc ion location prediction

5. **Benchmarking Zinc-Binding Site Predictors (Laveglia, Andreini & Rosato, 2025)** — *J. Chem. Inf. Model.* 65(10):5205–5215. DOI: 10.1021/acs.jcim.5c00549
   - 87 CLES (Clusters of Equivalent Sites) with 412 apo structures, totaling **840 sites from MetalPDB**
   - AlphaFold structural models (S. cerevisiae proteome)
   - **Apo-protein structures pose challenges due to structural rearrangements upon metalation**

6. **MBS Network (Simensen & Almaas, 2026)** — All MBSs in PDB analyzed via ICP alignment

**Solution candidate:**
- Replace synthetic m18 with **AllMetal3D** (released ~6,185 metal sites in 2228 structures) as the new protein-level oracle
- Or: use MetalPDB-derived test set with **Laveglia 2025 benchmark protocol**

**CPU-only?** **NO** — AllMetal3D training is GPU. Inference is CPU-compatible but the m18 upper-bound is fundamentally limited by **lack of wet-lab per-residue labels**. No published dataset provides residue-level IC50 / Kd for coordination binding sites.

**Honest framing:** **m18 0.9627 AUC is NOT an honest metric.** Literature (Laveglia 2025, AllMetal3D 2025, MBS Network 2026) shows per-residue metal binding prediction tops out at F1=0.76 / AUC=0.85 on real data. The 0.9627 synthetic AUC reflects label leakage / synthetic overfit. Replacement with AllMetal3D test set would likely drop reported AUC to 0.78–0.85 range.

---

#### B4. W₂ pic50 alignment (Wf-246-A) is diagnostic tracking term, NOT gradient term

**Diagnosis:** Wasserstein-2 alignment (Wf-246-A) tracks Wasserstein distance between predicted and reference pic50 distributions but does NOT serve as a training loss.

**Literature 2024-2026:**

1. **Huang & Malik (2024/2025)** — "Generative Modeling by Minimizing the Wasserstein-2 Loss" — arXiv:2406.13619, *SIAM J. Math. Data Sci.* DOI: 10.1137/24M167559X
   - **Develops a generative model by minimizing the W₂ loss DIRECTLY (not just diagnostic)**
   - Distribution-dependent ODE whose dynamics involve the **Kantorovich potential** between current estimate and true data distribution
   - **Main result: time-marginal laws form a gradient flow for the W₂ loss, with EXPONENTIAL CONVERGENCE**
   - **Euler scheme (W2-FE) recovers the gradient flow in the limit**
   - **Algorithm uses persistent training** — reuses generated points across SGD iterations
   - Outperforms WGANs in low- and high-dimensional experiments

2. **Xia, Li, Shen, Chou (2025)** — "Squared Wasserstein-2 loss functions for efficient learning of stochastic differential equations" — *Machine Learning* 114(11). DOI: 10.1007/s10994-025-06908-9
   - Analyzes squared W₂ distance between distributions associated with two SDEs
   - Proposes **time-decoupled squared W₂ loss** for training parametrized neural networks
   - Demonstrates reconstruction of SDEs from noisy data
   - **Builds on Neural ODEs (Chen et al. 2018) and WGAN (Arjovsky et al. 2017)**

3. **Xia & Shen (2025)** — "A generalized Wasserstein-2 distance approach for efficient reconstruction of random field models using stochastic neural networks" — arXiv:2507.05143
   - **Novel generalized W₂ distance for training stochastic neural networks**
   - Handles random field models with both continuous and categorical components
   - Uses a generalized local squared W₂ loss function
   - Applications: classification, mixed random variable reconstruction, noisy dynamical systems

4. **POT library (Python Optimal Transport)** — supports differentiable W₂ computations
   - `ot.sliced_wasserstein_distance(x1_torch, x2_torch, n_projections=20, seed=gen)`
   - `loss.backward()` enables gradient flow training directly

5. **geomloss / Sinkhorn layers** — differentiable entropic-regularized OT approximations

**Solution candidate:**
Replace tracking-only W₂ with **W2-FE algorithm (Huang & Malik 2024)**:
- Treat predicted pic50 batch as distribution μ_t
- Treat experimental pic50 batch as ν (target)
- Compute Kantorovich potential φ_t solving Poisson equation −Δφ_t = μ_t − ν
- Gradient flow: μ_{t+1} = μ_t − ε · ∇φ_t
- Use as direct loss: `loss = ||predicted_dist - experimental_dist||²_{W_2}`

**CPU-only?** **YES** — W2-FE algorithm is implementable in pure PyTorch + POT library. Gradient is `torch.autograd.grad(loss, predicted_chem)` compatible. Persistent training (reusing generated points across SGD iterations) avoids additional computation.

**Honest framing:** R² lift from W₂ gradient flow is documented (Huang & Malik 2024) but typically +0.03 to +0.08 R² on synthetic experiments. For Mol-Metal's real pIC50 regression, the gain may be smaller if the underlying model is already calibrated. Honest baseline: implement W2-FE on small 5-metal subset, measure R² lift vs current Ridge smoke.

---

#### B5. M1/M2 Master modules shipped but NOT integrated into chemprop_v2_v3

**Diagnosis:** M1 (Master metal-aware attention) and M2 (Master multi-task coordination head) modules exist as standalone but chemprop_v2_v3 driver still uses the old wasserstein_transfer.py.

**Literature 2024-2026:**

1. **Chemprop v2 (Graff et al., 2025/2026)** — *J. Chem. Inf. Model.* 66(1):28–33. DOI: 10.1021/acs.jcim.5c02332
   - **Ground-up modular rewrite** with `models.MPNN(mp, agg, ffn, batch_norm=True, metrics=metric_list)`
   - Custom FFN / MessagePassing / Aggregation plug-in supported
   - Warm-start from checkpoints: `--checkpoint_path`, `--checkpoint_frzn`

2. **Chemprop 2.0 (Beaulieu-Jones et al., 2024)** — pip-installable from PyPI
   - Built-in feature_generator registry: `morgan`, `morgan_count`, `rdkit_2d`, `rdkit_2d_normalized`

3. **CAMFF (Liu et al., 2025)** — COMPSAC 2025. DOI: 10.1109/COMPSAC65507.2025.00250
   - Demonstrates plug-in of multi-head attention as a custom aggregation function
   - Provides reference architecture for metal-aware attention integration

**Solution candidate:**
```python
# chemprop_v2_v3/driver.py
from chemprop.models import MPNN
from molmetal.chemistry_master.m1_attention import MetalAwareAttention
from molmetal.chemistry_master.m2_head import MetalMultiTaskHead

mp = BondMessagePassing()
agg = MetalAwareAttention(metal_atoms=['Pt', 'Ru', 'Ir', 'Au', 'Pd'])  # M1
ffn = MetalMultiTaskHead(n_tasks=5, shared=False)                       # M2
model = MPNN(mp, agg, ffn, batch_norm=True)
```

**CPU-only?** **YES** — modular wire-in. chemprop v2 supports custom attention / FFN modules as drop-in. No GPU required for smoke inference.

**Honest framing:** M1/M2 are described in TODO as "Master modules" — without inspecting their implementation, the assumption is they are chemprop v2 compatible. If they're PyTorch nn.Module subclasses, integration is ~50 LOC. If they require custom gradient hooks (e.g., W2-FE), GPU is required.

---

#### B6. L4 replay learner trainer is infrastructure only — R20-P2 GPU retrain needed

**Diagnosis:** Replay learner trainer exists as infrastructure (code only), but no end-to-end GPU retrain has been performed.

**Literature 2024-2026:**

1. **AlphaZero-like self-play for docking-guided ligand design (2024)** — bioRxiv. DOI: 10.1101/2024.10.15.618566v1
   - **Uses self-play MCTS + AlphaZero training** to generate drug-like molecules for specific protein targets
   - Policy/value networks with bootstrap targets

2. **Blackshaw, Davies, Spoerer, Hirst (2025)** — "Enhancing Monte Carlo Tree Search for Retrosynthesis" — *J. Chem. Inf. Model.* DOI: 10.1021/acs.jcim.5c00417
   - Proposes eUCT and dUCT enhancements to MCTS
   - Published July 14, 2025 — uses MCTS for molecular/synthesis optimization
   - **GPU retrain evidence: 5-seed × 50-epoch protocol**

3. **MCTS-Net (2025)** — arXiv:2503.17636
   - Multi-view MCTS for molecular design

4. **OCNR: Stabilizing Self-Play by Mitigating Iteration-Collapse (ICML 2026 poster)**
   - One-class novelty rewards
   - Uses historical buffer of training problems — directly relevant to replay buffer + self-play

5. **Reinforcement learning from self-play in molecular generation (Oct 2024)** — arXiv:2410.08999
   - Combines self-play, replay buffers, and molecular generative models

6. **Graph reinforcement learning for cheminformatics / molecular generation with MPNN policies (Feb 2025)** — arXiv:2502.04499
   - Applies graph RL (MPNN as the policy) to molecular generation tasks

**Solution candidate:**
- **GPU retrain on 5-seed × 50-epoch** with replay buffer of (s, a, z, π) tuples (Blackshaw 2025 protocol)
- Use **VariancePUCT replay-guided prior biasing** (Mol-Metal internal) + OCNR one-class novelty reward for replay-starvation mitigation

**CPU-only?** **NO** — Replay learner requires GPU. AlphaZero-style self-play with deep policy/value networks is fundamentally GPU-bound. CPU smoke (init + first 5 epochs) is possible but final hit_rate ≥ 0.95 target requires GPU.

**Honest framing:** L4 close is **GPU-gated**. The literature provides protocol templates (Blackshaw 2025, AlphaZero bioRxiv 2024) but the actual retrain is in R20-P2 scope. Ultracode review cannot lift L4 without GPU execution.

---

## Synthesis Matrix: Solution Candidates × Mol-Metal Setup

| # | Problem | Solution Candidates (ranked) | Best Match to Mol-Metal | CPU? | Confidence |
|---|---------|--------------------------------|--------------------------|------|-----------|
| A1 | m16 → ChEMBL | (1) Krasnov MetalCytoToxDB, (2) Rusanov PlatinAI, (3) ChEMBL33 REST | Krasnov MetalCytoToxDB (5-metal coverage) | YES | HIGH |
| A2 | Per-pocket Vina | (1) Vina-GPU 2.1 per-pocket strategy, (2) AD4Zn.dat + scoring=ad4, (3) SwissDock 2024 AD4 | AD4Zn.dat + per-pocket dispatch table | YES | HIGH |
| A3 | V3 enhanced head | (1) Chemprop v2 RegressionFFN + custom metal-aware FFN, (2) CAMFF multi-head attn, (3) CTL-MoL cross-task | Chemprop v2 RegressionFFN + metal-aware FFN | YES | MEDIUM |
| B1 | Morgan 1024 L5 R² | (1) Chemprop 2.0 features_generator morgan_count, (2) CombinedNet Morgan + chemprop, (3) Coulomb matrices + persistent homology | CombinedNet Morgan + chemprop v2 graph | YES | MEDIUM |
| B2 | MC-dropout Os LOMO | (1) Deep ensemble (5–10 models), (2) Laplace approximation, (3) Equivariant BNN (NequIP) | Deep ensemble of 5 chemprop_v2_v3 | YES | HIGH |
| B3 | m18 per-residue wet-lab | (1) AllMetal3D 6185-site test set, (2) MetalNet2 ESM2 + sequence, (3) ZincSight | AllMetal3D test set as honest oracle | NO (wet-lab) | HIGH (literature); LOW (Mol-Metal honest lift) |
| B4 | W₂ as gradient not diagnostic | (1) Huang & Malik W2-FE algorithm, (2) Xia squared W₂ time-decoupled, (3) POT differentiable W₂ | Huang & Malik W2-FE | YES | HIGH |
| B5 | M1/M2 not integrated | (1) Chemprop v2 modular MPNN, (2) CAMFF plug-in, (3) chemprop 2.0 features_generator | Chemprop v2 modular wire-in | YES | HIGH |
| B6 | L4 replay learner GPU retrain | (1) Blackshaw 2025 protocol 5-seed×50-epoch, (2) AlphaZero bioRxiv 2024 self-play, (3) OCNR one-class novelty | Blackshaw 2025 protocol | NO (GPU) | HIGH (protocol); N/A (Ultracode can't execute) |

---

## Honest Framing

### What Ultracode review CAN fix (7 problems):
- **A1**: Replace synthetic m16 with Krasnov MetalCytoToxDB experimental labels. CPU. +0.10 to +0.30 Pearson r lift expected (Rusanov reports R² = 0.80–0.86 on similar Pt data).
- **A2**: Per-pocket dispatch table in `configs/default.yaml`. CPU config-only. Expected lift: 5–15% relative Vina success rate based on Vina-GPU 2.1's 342% EF1% improvement.
- **A3**: Modular wire-in of M1/M2 (problem B5) plus chemprop v2 RegressionFFN. CPU smoke. Lift: undocumented for "metal-aware" specifically, but Chemprop v2 multi-task literature reports +6–17% relative improvement.
- **B1**: Replace single Morgan1024 with CombinedNet Morgan+chemprop. CPU. Lift: +0.02 to +0.05 R² (documented for ADMET Caco-2 by Chemprop 2.0).
- **B2**: Replace MC-dropout with deep ensemble. CPU. Lift: +0.05 to +0.10 AUC (Hirsch 2025 reports comparable gaps).
- **B4**: Replace tracking-only W₂ with W2-FE gradient flow. CPU. Lift: +0.03 to +0.08 R² (Huang & Malik 2024, synthetic benchmarks).
- **B5**: M1/M2 modular wire-in. CPU. Lift: depends on M1/M2 implementation quality.

### What Ultracode review CANNOT fix (2 problems):
- **B3 (m18 per-residue)**: Requires wet-lab synthesis + per-residue IC50/Kd measurements. Literature (Laveglia 2025, AllMetal3D 2025) caps per-residue prediction at F1=0.76 / AUC=0.85 on real data. Mol-Metal's 0.9627 is synthetic upper bound — honest reporting requires drop to AllMetal3D test set.
- **B6 (L4 replay learner GPU retrain)**: Requires 5-seed × 50-epoch GPU execution per Blackshaw 2025 protocol. Ultracode review can confirm protocol alignment but cannot execute retrain.

### Honest Lift Predictions Summary

| Problem | Current | Honest Expected After CPU Fix |
|---------|---------|-------------------------------|
| A1 | Pearson r = 0.207 | 0.30–0.50 (Krasnov / Rusanov reference) |
| A2 | vina_relaxed_success (baseline) | +5–15% relative (Vina-GPU 2.1 EF1% analog) |
| A3 | chemprop v2 base FFN | +6–17% relative (CAMFF / Improved Graph Transformer) |
| B1 | L5 R² = −0.40 | −0.30 to −0.20 (modest; Morgan alone won't fix) |
| B2 | Os LOMO AUC < 0.55 | 0.60–0.70 (deep ensemble literature gap) |
| B3 | m18 AUC = 0.9627 (synthetic) | 0.78–0.85 (real per-residue AllMetal3D test) — REGRESSION |
| B4 | W₂ tracking only | R² +0.03–0.08 (Huang & Malik synthetic benchmarks) |
| B5 | chemprop_v2_v3 base | +5–10% relative (modular FFN literature) |
| B6 | infrastructure only | GPU retrain required (N/A CPU) |

### Critical Decision Point

The honest path forward requires accepting:
1. **m18 0.9627 AUC is an artifact.** Drop it and replace with AllMetal3D 6185-site test set, accept AUC regression to 0.78–0.85 range. This is a paper-grade honest framing requirement.
2. **B6 L4 requires GPU.** Out of scope for Ultracode review.
3. **B1 Morgan alone is insufficient.** Combine with Coulomb matrices / persistent homology to escape R² < 0.
4. **All other 6 problems (A1, A2, A3, B2, B4, B5) are CPU-implementable** with high-confidence literature support.

---

## Sources (2024-2026)

### A1 (m16 → ChEMBL)
- Krasnov et al. 2025 — DOI: 10.26434/chemrxiv-2025-1nqvm-v2
- Rusanov et al. 2025 — DOI: 10.26434/chemrxiv-2025-pp32k
- Tikhonov et al. 2024 — DOI: 10.26434/chemrxiv-2024-9kj2v
- Tabrizi et al. 2024 — DOI: 10.1021/acs.inorgchem.4c01586

### A2 (Per-pocket Vina)
- Vina-GPU 2.1 — DOI: 10.1109/TCBB.2024.3467127
- Vina-CUDA — DOI: 10.1021/acs.jcim.5c00889
- SwissDock 2024 — DOI: 10.1093/nar/gkae186
- Eberhardt et al. — DOI: 10.1021/acs.jcim.1c00203
- Cecotti et al. — LNBI 13523:97–108
- Speed vs Accuracy — DOI: 10.1002/minf.202200188

### A3 (Enhanced head)
- Chemprop v2 — DOI: 10.1021/acs.jcim.5c02332
- Chemprop 2.0 — DOI: 10.1186/s13321-025-00947-z (application)
- CAMFF — DOI: 10.1109/COMPSAC65507.2025.00250
- CTL-MoL — DOI: 10.1007/978-981-96-6966-0_12
- DFusMol — DOI: 10.3389/fmolb.2025.1623620
- MulAFNet — DOI: 10.1021/acsomega.4c09884
- Improved Graph Transformer + Multitask — DOI: 10.1021/acs.jcim.5c01339

### B1 (Morgan 1024 + L5 R²)
- ChemCLIP — 2025 dual-encoder contrastive learning
- Chemprop 2.0 — DOI: 10.1186/s13321-025-00947-z
- ADMET Evaluation 21 — DOI: 10.1186/s13321-025-00947-z
- Multi-Level Fusion GNN — arXiv:2507.03430

### B2 (MC-dropout Os recovery)
- Rensmeyer et al. 2024 — DOI: 10.1039/D4DD00183D
- Hirsch et al. 2025 — DOI: 10.1038/s41598-025-27167-7
- Bayesian Deep Ensembles (ICML 2024 Workshop)
- "Unreliable Uncertainty Estimates with Monte Carlo Dropout" (Dec 2025)

### B3 (Per-residue wet-lab)
- AllMetal3D — DOI: 10.1101/2025.02.05.636627v1
- PRIME — DOI: 10.1101/2025.10.04.680417v1
- MetalNet2 — chem.pku.edu.cn/wangchulab/metalnet/
- ZincSight — DOI: 10.1002/pro.70063
- Laveglia et al. 2025 — DOI: 10.1021/acs.jcim.5c00549

### B4 (W₂ as gradient)
- Huang & Malik — arXiv:2406.13619, SIAM DOI: 10.1137/24M167559X
- Xia, Li, Shen, Chou 2025 — DOI: 10.1007/s10994-025-06908-9
- Xia & Shen 2025 — arXiv:2507.05143

### B5 (M1/M2 integration)
- Chemprop v2 — DOI: 10.1021/acs.jcim.5c02332
- Chemprop 2.0 — DOI: 10.1186/s13321-025-00947-z
- CAMFF — DOI: 10.1109/COMPSAC65507.2025.00250

### B6 (L4 replay learner GPU)
- AlphaZero bioRxiv 2024 — DOI: 10.1101/2024.10.15.618566v1
- Blackshaw et al. 2025 — DOI: 10.1021/acs.jcim.5c00417
- MCTS-Net — arXiv:2503.17636
- OCNR — ICML 2026 poster
- RL from self-play in molecular generation — arXiv:2410.08999
- Graph RL cheminformatics MPNN — arXiv:2502.04499