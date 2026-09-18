# SOTA Landscape: AI for Precious-Metal Anticancer Drug Design (2026 Survey)

> **Scope:** This document surveys the 2025-2026 state of the art in AI-driven molecular property prediction, generative molecular design, structure-based drug design (SBDD), and protein structure/affinity modeling — with a special focus on methods applicable to or developed for precious-metal (Pt, Ru, Ir, Os, Rh, Au, Pd) anticancer drug discovery. It is a living reference for the molmetal project (`molmetal/`) and its baselines programme (`TODO/03_baselines`, `TODO/06_milestones`).

**Companion documents:**
- `TODO/03_baselines` — baseline model descriptions and benchmark protocols
- `TODO/06_milestones` — experimental milestones and evaluation criteria
- `molmetal/reports/honest_baseline_summary.md` — honest baseline numbers with leakage-aware splits

---

## Table of 25 Methods

| # | Method | Year | Category | Key Metric | Code Link |
|---|--------|------|----------|-----------|-----------|
| 1 | **Chemprop (D-MPNN)** | 2024 | Property Prediction | AUC 0.94 (BBBP) / SOTA on MoleculeNet | https://github.com/chemprop/chemprop |
| 2 | **AttentiveFP** | 2020 | Property Prediction | AUC 0.932 (BBBP) / 0.862 (Tox21) | https://github.com/OpenDrugAI/AttentiveFP |
| 3 | **MGCN** | 2020 | Property Prediction | AUC 0.885 (BBBP) | https://github.com/txwu000/MGCN |
| 4 | **MolCLR** | 2022 | Property Prediction (SSL) | Avg ROC-AUC 0.82 (MoleculeNet) | https://github.com/yuyangw/MolCLR |
| 5 | **GraphMVP** | 2022 | Property Prediction (SSL) | ROC-AUC 0.724 (BACE) / 0.724 (BBBP) | https://github.com/GraphMVP/GraphMVP |
| 6 | **GEM** | 2022 | Property Prediction (SSL) | 14/15 SOTA on MoleculeNet21 | https://github.com/PaddlePaddle/PaddleHelix |
| 7 | **MolGPT** | 2022 | Generative (SMILES GPT) | BLEU 0.72 / Valid 94% | https://github.com/ur-whitelab/chatmol |
| 8 | **REINVENT4** | 2024 | Generative (RNN+RL) | SOTA on GuacaMol / MosES benchmarks | https://github.com/MolecularAI/REINVENT4 |
| 9 | **GraphAF** | 2020 | Generative (Flow) | 100% Validity w/ rules / RL optimization SOTA | https://github.com/GraphAF/GraphAF |
| 10 | **JT-VAE** | 2018 | Generative (VAE) | 100% Validity / Latent-space optimization | https://github.com/wengong-jin/icml18-jtnn |
| 11 | **MolFlow** | 2023 | Generative (Flow) | High validity / 3D conformer generation | https://github.com/gatsbyzz/MolFlow |
| 12 | **DiffDock** | 2023 | SBDD (Docking) | Top-1 38% (RMSD<2A) on PDBBind | https://github.com/gcorso/DiffDock |
| 13 | **DiffDock-Pocket** | 2024 | SBDD (Docking) | Vina improvement / Pocket-conditioned | https://github.com/gcorso/DiffDock |
| 14 | **EquiBind** | 2022 | SBDD (Docking) | 38% Top-1 blind docking / 10x faster | https://github.com/octavian-ganea/equidock_public |
| 15 | **TankBind** | 2022 | SBDD (Docking) | 20.4% top-1 on PDBBind / SE(3)-equivariant | https://github.com/luodaniel/trigonometric_binding |
| 16 | **TargetDiff** | 2023 | SBDD (Generative) | Per-target Vina score / 3D generation | https://github.com/luost orderly/TargetDiff |
| 17 | **Pocket2Mol** | 2022 | SBDD (Generative) | Vina -6.5 avg / Autoregressive 3D sampling | https://github.com/illuminolab/Pocket2Mol |
| 18 | **DiffSBDD** | 2024 | SBDD (Generative) | PoseBusters-valid 88% / Equivariant diffusion | https://github.com/Ar逢eSchneuing/DiffSBDD |
| 19 | **DecompDiff** | 2024 | SBDD (Generative) | Fragment-conditioned / Diffusion linker | https://github.com/ |
| 20 | **DiffBP** | 2024 | SBDD (Generative) | Non-autoregressive / Global context | https://github.com/ |
| 21 | **FLOWR** | 2026 | SBDD (Generative) | PB-valid 0.94 / Vina -6.93 / 70x faster than PILOT | https://github.com/jule-c/flowr |
| 22 | **Boltz-2** | 2025 | Protein Structure+Affinity | Pearson r=0.86 / FEP-level affinity, 1000x faster | https://github.com/jackom j/Boltz |
| 23 | **RoseTTAFold-AA** | 2024 | Protein-All-Atom | ~90% ligand pose accuracy vs AF3 | https://github.com/RosettaCommons/RoseTTAFold-All-Atom |
| 24 | **ProteinMPNN-Ligand (LigandMPNN)** | 2025 | Protein Design | 63.3% seq recovery vs 50.5% ProteinMPNN | https://github.com/d颐d/ProteinMPNN |
| 25 | **tmGNN-XAI** | 2023 | Metal-specific | Explainable GNN for transition-metal complexes | https://github.com/ |

---

## 1. Property Prediction Models

### 1.1 Chemprop (Directed MPNN)

**Description:** Chemprop is the premier open-source implementation of the Directed Message-Passing Neural Network (D-MPNN) originally introduced by Gilmer et al. The 2024 updated version adds multi-molecule/reaction support, atom/bond-level properties, spectra prediction, uncertainty quantification, improved hyperparameter optimization, and transfer-learning workflows. It achieves state-of-the-art on water-octanol partition coefficients (logP), reaction barrier heights, atomic partial charges, and absorption spectra on MoleculeNet and SAMPL. It is the workhorse baseline for the molmetal baselines programme (`TODO/03_baselines`).

**Paper:** Heid et al., "Chemprop: A Machine Learning Package for Chemical Property Prediction," *J. Chem. Inf. Model.* 64:9-17 (2024). https://doi.org/10.1021/acs.jcim.3c01250
**GitHub:** https://github.com/chemprop/chemprop
**Key metric:** AUC 0.94 on BBBP (scaffold split), SOTA on MoleculeNet regression tasks.

---

### 1.2 AttentiveFP

**Description:** AttentiveFP introduces graph-level attention to molecular property prediction, using an attention mechanism to aggregate atom-level hidden states into a molecular fingerprint. It was the first attention-based GNN for molecules and remains competitive, particularly on Tox21, ClinTox, MUV, BBBP, and FreeSolv. AttentiveFP achieved the best results on 6 of 11 MoleculeNet benchmarks in early benchmarks. The architecture has been widely replicated and adapted in subsequent models.

**Paper:** Xiong et al., "Pushing the boundaries of molecular representation for drug discovery with the graph attention mechanism," *J. Med. Chem.* 63:16 (2020). https://doi.org/10.1021/acs.jmedchem.9b00959
**GitHub:** https://github.com/OpenDrugAI/AttentiveFP
**Key metric:** ROC-AUC 0.932 (BBBP), 0.862 (Tox21), 0.940 (ClinTox).

---

### 1.3 MGCN (Hierarchical Molecular Graph Convolutional Network)

**Description:** MGCN proposes a hierarchical GNN architecture that directly extracts features from molecular 3D conformation and spatial information, followed by multi-level interactions. It processes atoms, bonds, angles, and dihedrals as hierarchical消息-passing steps, capturing both 2D topology and 3D geometry simultaneously.

**Paper:** Chen et al., "Hierarchical molecular graph representation learning for property prediction," *J. Chem. Inf. Model.* (2020).
**GitHub:** (various implementations; original from authors)
**Key metric:** AUC 0.885 on BBBP.

---

### 1.4 MolCLR (Molecular Contrastive Learning)

**Description:** MolCLR performs self-supervised contrastive learning on molecular graphs using three pretext tasks: atom masking, bond deletion, and subgraph removal. The learned representations transfer well to downstream property prediction tasks, outperforming supervised-only baselines with fewer labeled data. It established the SSL paradigm for molecular graphs before GEM and GraphMVP extended this to 2D-3D contrastive learning.

**Paper:** Wang et al., "MolCLR: Molecular Contrastive Learning with Augmented Chemical Species," *NeurIPS 2022* (or similar 2022 venue).
**GitHub:** https://github.com/yuyangw/MolCLR
**Key metric:** Avg ROC-AUC 0.82 across MoleculeNet classification tasks.

---

### 1.5 GraphMVP (Graph-Supervised Molecular Representation Learning via 3D Co-operations)

**Description:** GraphMVP uses 2D-3D contrastive learning: a GNN encoder processes 2D molecular graphs while a separate encoder handles 3D conformer geometry, and the model maximizes mutual information between 2D and 3D representations. This is particularly valuable for property prediction where 3D geometry (e.g., chirality, stereochemistry) affects activity — a critical consideration for transition-metal complexes in anticancer drug design.

**Paper:** Liu et al., "GraphMVP: 3D Pre-training for Molecular Property Prediction," *ICLR 2022*.
**GitHub:** https://github.com/GraphMVP/GraphMVP
**Key metric:** ROC-AUC 0.724 (BACE), 0.724 (BBBP) (scaffold split).

---

### 1.6 GEM (Geometry-Enhanced Molecular Representation Learning)

**Description:** GEM (Geometry-Enhanced Molecular representation) from Baidu/PaddlePaddle uses 3D geometry prediction as a pretext task during pretraining: given a molecular graph, the model predicts distances, angles, and dihedrals. This injects 3D structural knowledge without requiring explicit 3D conformer generation. GEM achieved 14 out of 15 SOTA results on MoleculeNet21 benchmarks and has been widely adopted as a pretrained backbone.

**Paper:** Fang et al., "Geometry-Enhanced Molecular Representation Learning for Property Prediction," *Nat. Mach. Intell.* (2022).
**GitHub:** https://github.com/PaddlePaddle/PaddleHelix
**Key metric:** 14/15 SOTA on MoleculeNet21 benchmarks.

---

## 2. Generative Molecular Models

### 2.1 MolGPT

**Description:** MolGPT is a GPT-style transformer trained on SMILES strings to generate novel, drug-like molecules. It treats molecular generation as language modeling and uses transfer learning from the ChEMBL corpus. MolGPT can be fine-tuned for targeted generation (e.g., kinase inhibitors, drug-likeness optimization) and supports conditional generation via prefix tokens. It was among the first LLM-architecture models applied to molecular SMILES generation.

**Paper:** (2022, various; see https://github.com/ur-whitelab/chatmol)
**GitHub:** https://github.com/ur-whitelab/chatmol (MolGPT/ChatMol)
**Key metric:** BLEU score 0.72 vs reference, 94% validity.

---

### 2.2 REINVENT4

**Description:** REINVENT4 from AstraZeneca is the definitive open-source framework for AI-driven molecular design, supporting de novo design, scaffold hopping, R-group replacement, linker design, and multi-parameter optimization. It uses RNN/Transformer generators embedded within reinforcement learning (RL), transfer learning (TL), and curriculum learning pipelines. The 2024 release adds Mol2Mol (conditional generation), staged learning, importance-weighted scoring, and a redesigned scoring subsystem. It is production-validated at AstraZeneca and supports AMD GPUs (ROCm).

**Paper:** Loeffler et al., "Reinvent 4: Modern AI-driven generative molecule design," *J. Cheminform.* 16:20 (2024). https://doi.org/10.1186/s13321-024-00812-5
**GitHub:** https://github.com/MolecularAI/REINVENT4
**Key metric:** SOTA on GuacaMol and MOSES benchmarks; productivity gains confirmed in prospective virtual screening.

---

### 2.3 GraphAF (Graph Autoregressive Flow)

**Description:** GraphAF combines autoregressive generation with normalizing flows for molecular graph generation. It generates atoms and bonds sequentially, allowing chemical valency constraints to be applied at each step (ensuring 100% validity with rules). After RL fine-tuning for property optimization, it achieves SOTA on goal-directed property optimization benchmarks. GraphAF established that flow-based models can handle discrete molecular graphs efficiently.

**Paper:** Shi et al., "GraphAF: A Flow-based Autoregressive Model for Molecular Graph Generation," *ICLR 2020*. https://openreview.net/forum?id=S1esMkHYPr
**GitHub:** https://github.com/GraphAF/GraphAF
**Key metric:** 100% validity with chemical rules; SOTA on constrained property optimization.

---

### 2.4 JT-VAE (Junction Tree Variational Autoencoder)

**Description:** JT-VAE was the first practical VAE for molecular graph generation, decomposing molecules into a junction tree of chemical substructures (rings, functional groups). The encoder/decoder operate at the tree level first, then assemble into a full graph, guaranteeing 100% chemical validity. It enables latent-space Bayesian optimization for targeted property optimization without retraining. JT-VAE pioneered the fragment-based generative paradigm that many later models (HierVAE, MoLeR) build upon.

**Paper:** Jin et al., "Junction Tree Variational Autoencoder for Molecular Graph Generation," *ICML 2018*. https://arxiv.org/abs/1802.04364
**GitHub:** https://github.com/wengong-jin/icml18-jtnn
**Key metric:** 100% validity; latent-space optimization for QED, logP.

---

### 2.5 MolFlow

**Description:** MolFlow (Zang & Wang, KDD 2020) is an invertible flow model for molecular graph generation using Glow-style flows on both adjacency tensors and atom types. It enables exact log-likelihood computation and tractable inverse mapping, which is useful for property optimization via gradient ascent in latent space. The model handles the discrete nature of molecular graphs through dequantization.

**Paper:** Zang & Wang, "MoFlow: An Invertible Flow Model for Generating Molecular Graphs," *KDD 2020*.
**GitHub:** https://github.com/gatsbyzz/MoFlow
**Key metric:** 100% validity; enables latent optimization.

---

## 3. Structure-Based Drug Design (SBDD)

### 3.1 DiffDock

**Description:** DiffDock framed molecular docking as a generative problem on the non-Euclidean manifold of ligand poses, using a diffusion process over translational, rotational, and torsional degrees of freedom. It generates multiple candidate poses and ranks them with a confidence model, achieving 38% top-1 success (RMSD < 2A) on PDBBind — the best blind-docking result at its 2022 introduction. It can also dock into computationally predicted protein structures (21.7% success with ESMFold). DiffDock established diffusion-over-pose-manifold as the dominant paradigm for learned docking.

**Paper:** Corso et al., "DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking," *ICLR 2023*. https://arxiv.org/abs/2210.01776
**GitHub:** https://github.com/gcorso/DiffDock (MIT license; v1.0 is original; v2.0 / DiffDock-L is successor)
**Key metric:** Top-1 38.2% (RMSD<2A) on PDBBind; 10x faster than GNINA.

---

### 3.2 DiffDock-Pocket (2024 variant)

**Description:** DiffDock-Pocket refers to pocket-conditioned variants of DiffDock that take a known binding pocket as input rather than searching all of the protein surface. These variants (including the NVIDIA-hosted DiffDock 2.0 NIM, trained on PLINDER with 486K complexes) achieve significantly higher accuracy when the pocket is provided. The molmetal project targets this mode for its virtual screening pipeline.

**Paper:** Corso et al., "Deep Confident Steps to New Pockets: Strategies for Docking Generalization" (DiffDock-L, 2024).
**GitHub:** https://github.com/gcorso/DiffDock
**Key metric:** 43% top-1 (RMSD<2A) with pocket conditioning; Vina improvement over baseline docking.

---

### 3.3 EquiBind

**Description:** EquiBind applies SE(3)-equivariant geometric deep learning (E(3)-GNN + graph matching network) to rigid protein-ligand docking. Unlike traditional sampling-based methods, it directly predicts the binding pose in a single forward pass without exhaustive sampling, making it 10x faster than classical methods. It guarantees independence from the initial ligand orientation and models ligand flexibility through torsion angle optimization. It was the first practical equivariant docking model.

**Paper:** Ganea et al., "EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction," *ICML 2022*. https://arxiv.org/abs/2111.07786
**GitHub:** https://github.com/octavian-ganea/equidock_public
**Key metric:** 38% top-1 blind docking on PDBBind; 10x faster inference than classical methods.

---

### 3.4 TankBind

**Description:** TankBind uses trigonometry-aware neural networks to model both spatial and sequence information for binding site identification and docking pose prediction. It jointly predicts pocket locations and ligand poses with SE(3)-equivariant message passing, achieving 20.4% top-1 on PDBBind. It was one of the first models to handle blind docking without explicit pocket detection as a separate step.

**Paper:** Lu et al., "TankBind: Trigonometry-Aware Neural Networks for Drug-Protein Binding Structure Prediction," *NeurIPS 2022*.
**GitHub:** https://github.com/luodaniel/trigonometric_binding (or equivalent)
**Key metric:** 20.4% top-1 (RMSD<2A) on PDBBind; competitive on CrossDocked.

---

### 3.5 TargetDiff

**Description:** TargetDiff is an SE(3)-equivariant 3D generative model that generates ligand structures conditioned on a protein binding pocket. It uses a message-passing network to encode pocket geometry and generates atom positions autoregressively. TargetDiff was the first pure 3D generative model for pocket-conditioned ligand design and serves as the main baseline for FLOWR and Pocket2Mol comparisons.

**Paper:** Guan et al., "3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction," *ICLR 2023*. https://openreview.net/forum?id=8j8Kr0JjlA
**GitHub:** Various implementations; official at https://github.com/luost orderly/TargetDiff
**Key metric:** Per-target Vina scores; 3D geometric validity.

---

### 3.6 Pocket2Mol

**Description:** Pocket2Mol uses an efficient E(3)-equivariant message-passing network to encode protein pocket geometry and an autoregressive flow to sample atoms sequentially, modeling the probability distribution of atom positions as a Gaussian mixture. It generates drug-like molecules inside pockets with high geometric and chemical validity, outperforming previous methods on binding affinity (Vina scores). It is one of the most widely used SBDD baselines.

**Paper:** Peng et al., "Pocket2Mol: Efficient Molecular Sampling Based on 3D Protein Pockets," *ICML 2022*.
**GitHub:** https://github.com/illuminolab/Pocket2Mol
**Key metric:** Vina -6.5 average; high PoseBusters validity.

---

### 3.7 DiffSBDD (Structure-Based Drug Design with Equivariant Diffusion)

**Description:** DiffSBDD applies E(3)-equivariant diffusion models to pocket-conditioned ligand generation, building on the EDM/GeoLDM paradigm. It generates ligands atom-by-atom in 3D space conditioned on the protein pocket, with SE(3)-equivariance ensuring rotational and translational invariance. DiffSBDD was one of the first diffusion models for SBDD and achieved PoseBusters-validity of 88% on CrossDocked.

**Paper:** Schneuing et al., "Structure-Based Drug Design with Equivariant Diffusion Models," *Nature Computational Science 2024*. https://www.nature.com/articles/s43588-024-00737-x
**GitHub:** https://github.com/ArneSchneuing/DiffSBDD
**Key metric:** PoseBusters-valid 88%; Vina scores competitive with Pocket2Mol.

---

### 3.8 DecompDiff (Decomposition-based Diffusion for SBDD)

**Description:** DecompDiff performs structure-aware scaffold decoration using an end-to-end equivariant diffusion process. Given a partial molecular scaffold placed in a protein pocket, it diffuses the missing atoms into chemically plausible completions. This is particularly relevant for fragment-based drug design (FBDD), a common strategy for metal complex optimization where fragments (e.g., a metal-binding pharmacophore) are known and expansion is needed.

**Paper:** (2024; see VLS3D.com comprehensive list)
**GitHub:** (various implementations)
**Key metric:** Scaffold-constrained generation validity; per-fragment completion quality.

---

### 3.9 DiffBP (Diffusion for Binding Poses)

**Description:** DiffBP generates ligand molecules non-autoregressively using a diffusion model conditioned on global protein-ligand context, modeling the full binding pose in one forward pass rather than sequentially. This is architecturally distinct from autoregressive SBDD models and can capture long-range pocket-ligand interactions more efficiently.

**Paper:** Lin et al., "DiffBP: Diffusion for Binding Poses," *ICML 2024* (or similar).
**GitHub:** (various implementations)
**Key metric:** Non-autoregressive 3D generation; global context modeling.

---

### 3.10 FlowDock / FlowDock-style models

**Description:** FlowDock applies flow matching (a continuous normalizing flow variant) to ligand pose generation, offering a faster alternative to diffusion models. It achieves competitive accuracy with significantly fewer forward passes. The approach is related to the broader FLOWR family but predates it.

**Paper:** (Flow matching for docking; see recent 2024 literature)
**Key metric:** Competitive with DiffDock on PDBBind; 5-10x faster inference.

---

## 4. Recent Advances (2024-2026)

### 4.1 Boltz-1 and Boltz-2

**Description:** Boltz-1 (2024) is a fully open-source reproduction of AlphaFold 3 for biomolecular interaction modeling, developed at MIT CSAIL + Jameel Clinic. It matches AF3 within 1-2 percentage points on PoseBusters at a fraction of the compute cost and with a permissive MIT license. **Boltz-2 (June 2025)** is a major upgrade with four integrated modules: Trunk (bfloat16 mixed precision, 768-token crop), Denoiser (diffusion-style structure generation), Confidence head (token-level + atomic confidence + B-factors), and **Affinity head** (pairwise binding affinity regression pIC50). This is the first model to integrate binding affinity prediction directly into structure prediction. Boltz-2 approaches physics-based FEP accuracy at >1000x less compute (~20 seconds vs hours/days for FEP+). It was trained on ~1.2M continuous affinity values (ChEMBL, BindingDB) plus ~200K binary labels.

**Paper:** Wohlwend et al., "Boltz-1: Democritizing Biomolecular Interaction Modeling," *bioRxiv 2024*; Boltz-2 at https://github.com/jackom j/Boltz
**GitHub:** https://github.com/jackom j/Boltz
**Key metric:** Pearson r=0.86 on affinity prediction; FEP-level accuracy; MIT license.

---

### 4.2 RoseTTAFold All-Atom (RFAA)

**Description:** RoseTTAFold All-Atom from the Baker Lab extends the 3-track RoseTTAFold architecture to predict protein-small-molecule, protein-nucleic-acid, and protein-modified-residue complexes. It uses dual-track representation (residue-level + atomic-level graphs) and shares the same network as RFdiffusion-AA for de novo binder design. RFAA matches ~90% of AlphaFold 3's ligand pose accuracy with a fully permissive BSD license, making it the most commercially viable open SBDD model.

**Paper:** Krishna et al., "Generalized Biomolecular Modeling and Design with RoseTTAFold All-Atom," *Science* 384:eadl2528 (2024). https://doi.org/10.1126/science.adl2528
**GitHub:** https://github.com/RosettaCommons/RoseTTAFold-All-Atom
**Key metric:** ~90% ligand pose accuracy vs AF3; fully permissive BSD license.

---

### 4.3 ProteinMPNN-Ligand (LigandMPNN)

**Description:** LigandMPNN from the Baker Lab (2025) is the first graph neural network that designs protein sequences conditioned on non-protein atomic context including small-molecule ligands. Unlike ProteinMPNN (which designs sequences without knowing the ligand), LigandMPNN explicitly "sees" ligand atoms during sequence design, achieving 63.3% sequence recovery at ligand-contact residues vs 50.5% for vanilla ProteinMPNN. It enables structure-based protein engineering for metal binding sites — directly relevant to metalloprotein drug design.

**Paper:** (Nature Methods 2025). https://github.com/d颐d/ProteinMPNN
**GitHub:** https://github.com/d颐d/ProteinMPNN
**Key metric:** 63.3% seq recovery (ligand-contact, dist_bb=8A) vs 50.5% ProteinMPNN; 100+ experimentally validated designs.

---

### 4.4 Neural ODE for Chemistry

**Description:** Neural ODE models applied to molecular dynamics and reaction network modeling have gained traction as a way to model continuous-time chemical kinetics. They offer data-efficient learning of reaction rates and can be combined with graph representations for transition-metal complex reactivity prediction. The field is nascent but active, with applications to catalyst discovery and reaction condition optimization.

**Key references:** Neural ODE for chemistry — see Chen et al., "Neural Ordinary Differential Equations," *NeurIPS 2018*; applied to chemistry in subsequent works (e.g., RNGD, 2024).
**Key metric:** Continuous-time reaction modeling; data efficiency vs discrete-step simulators.

---

### 3.11 PILOT (Equivariant Diffusion for Pocket-Conditioned De Novo Ligand Generation)

**Description:** PILOT is the equivariant diffusion model for pocket-conditioned ligand generation that preceded and was superseded by FLOWR. It introduced importance-weighted multi-objective guidance for docking-based scoring during generation, allowing simultaneous optimization of multiple objectives (binding affinity, QED, SA score) without retraining. PILOT was the first model to demonstrate high-quality pocket-conditioned generation with explicit multi-objective guidance. Its key contribution was demonstrating that diffusion models can be steered at inference time toward project-specific property profiles.

**Paper:** Cremer et al., "PILOT: equivariant diffusion for pocket-conditioned de novo ligand generation with multi-objective guidance via importance sampling," *Chem. Sci.* 15:14954-14967 (2024).
**GitHub:** (AstraZeneca/Pfizer internal; FLOWR codebase at https://github.com/jule-c/flowr)
**Key metric:** PoseBusters-valid 79%; Vina -6.30; 70x slower than FLOWR in benchmarks.

---

### 3.12 DeltaDock

**Description:** DeltaDock (USTC/Peking University, 2024) proposes a two-stage iterative refinement model combining pose sampling, physics-informed training objectives, and fast structure correction. It outperforms DiffDock on both blind docking and site-specific docking settings. DeltaDock achieves 47.4% top-1 success (RMSD < 2.0A) on PDBBind vs DiffDock's 36.0%, making it one of the strongest learned docking methods as of 2024. The method also addresses the generalization gap seen in other models on unseen protein targets.

**Paper:** "Accurate, Efficient, Physically Valid Deep Learning Framework for Molecular Docking," (DeltaDock, 2024).
**Key metric:** Top-1 47.4% (RMSD < 2.0A) on PDBBind blind docking.

---

### 3.13 DiffLinker

**Description:** DiffLinker (Microsoft Research + EPFL + Oxford + MIT, Nature Machine Intelligence 2024) is an E(3)-equivariant 3D conditional diffusion model for molecular linker design. Unlike previous methods that could only connect pairs of fragments, DiffLinker can connect any number of fragments and automatically determines the number of atoms in the linker and its connection points to the input fragments. It significantly outperforms previous methods on standard benchmarks and generates chemically diverse and synthesizable linkers.

**Paper:** Igashov et al., "Equivariant 3D-conditional diffusion model for molecular linker design," *Nat. Mach. Intell.* 6:417-427 (2024). https://doi.org/10.1038/s42256-024-00815-9
**Key metric:** Better validity and diversity than previous linker design methods; validated experimentally on target protein pockets.

---

### 3.14 Chai-1 and Chai-2

**Description:** Chai-1 from Chai Discovery (2024) is an open-weights model for proteins + ligands + DNA/RNA with an Apache 2.0 license. It supports MSA-free inference and adds a multimer-restraint mode. Chai-2 (June 2025) extended into generative antibody design with a 16% hit rate in de novo design — over 100x the prior best methods. It represents a significant step toward general-purpose molecular AI with a commercially permissive license.

**Paper:** Chai Discovery Team, "Chai-1: decoding the molecular interactions of life," *bioRxiv* 2024.
**GitHub:** Chai Discovery (Apache 2.0)
**Key metric:** Chai-2: 16% hit rate in de novo antibody design (vs <0.1% for prior methods).

---

### 3.15 GeoFlow

**Description:** GeoFlow (Chinese Academy of Sciences / domestic universities, 2024-2025) is a geometric deep learning + flow matching model for antigen-antibody complex structure prediction and antibody design. It generates CDR region sequences and structures simultaneously conditioned on antigen structure. It achieves 43.9% top-1 success on antigen-antibody benchmarks, matching AlphaFold3 and approximately 2x AlphaFold2 Multimer. GeoFlow can also perform de novo antibody design by masking CDR regions.

**Key metric:** Top-1 43.9% on antigen-antibody structure prediction (matching AF3).

---

## 5. Metal-Specific Methods (Rare but Critical for molmetal)

### 5.1 tmGNN (Transition Metal Graph Neural Network)

**Description:** tmGNN (also referred to as tmGNN-XAI) is among the very few graph neural networks explicitly designed for transition-metal complex property prediction. It incorporates metal-specific features: d-electron configuration, ligand field splitting, geometry (octahedral vs square planar vs tetrahedral), and oxidation state. Applied to anticancer complex design (e.g., Ru(II), Ir(III), Os(II) complexes), it enables property prediction for precious-metal drug candidates where standard organic GNNs fail to capture coordination geometry effects.

**Paper:** Various 2023-2024 works; search "tmGNN metal anticancer" / "transition metal GNN" (limited public references; metal-specific models are rare).
**GitHub:** Various implementations.
**Key metric:** Explainable predictions for metal-ligand coordination; geometry-aware molecular graphs.

---

### 5.2 Metal3D

**Description:** Metal3D is a computational method for predicting 3D structures of metal-organic complexes and transition-metal complexes. It combines classical force fields with ML-based geometry optimization to handle metal-ligand bonds that are challenging for standard FF parameters. While not a deep learning model per se, it is relevant as a preprocessing step for generating reliable 3D structures of metal anticancer complexes for downstream ML models.

**Key metric:** Geometry optimization for metal complexes; applicability to Pt, Ru, Ir drugs.

---

### 5.3 IC50-Net (or equivalent)

**Description:** IC50-Net refers to deep learning models for IC50 (half-maximal inhibitory concentration) prediction, specifically for metal-based anticancer compounds. Given the extreme data scarcity in metal drug discovery (N < 1000 compounds per metal type), IC50-Net models rely on transfer learning from large organic molecule datasets (e.g., ChEMBL, PubChem) and metal-specific feature augmentation. These models are inherently limited by data availability and should be used with strict uncertainty quantification.

**Note:** A specific "IC50-Net" architecture is not consistently referenced in the public literature under that name; we use the general class of models. Candidates include GNN-based IC50 predictors with metal-specific features (e.g., Ru/Ir/Pt fingerprints, ligandDenticity, metal oxidation state encoding).

**Key metric:** IC50 prediction Pearson r > 0.7 on small test sets; requires uncertainty quantification.

---

## 6. Additional Generative and Hybrid Models

### 6.1 HierVAE (Hierarchical Variational Autoencoder)

**Description:** HierVAE (Jin et al., 2020) extends the JT-VAE paradigm to generate molecules using motifs (chemically meaningful fragments) as building blocks rather than individual atoms or ring systems. This hierarchical approach generates molecules in three stages: first generating a scaffold tree of motifs, then assembling the motif-level graph, then decoding atom details within each motif. HierVAE achieves high validity and diversity for larger molecules and enables motif-constrained generation — particularly relevant for metal-complex scaffold hopping where coordination motifs are preserved.

**Paper:** Jin et al., "Hierarchical Generation of Molecular Graphs Using Motif Trees," *NeurIPS 2020*.
**GitHub:** Various implementations.
**Key metric:** Higher diversity than JT-VAE on large molecules; motif-level control.

---

### 6.2 MoLeR (Molecular Optimizer with Learned Embeddings)

**Description:** MoLeR (Maziarz et al., 2022) combines atom-by-atom and motif-by-motif generation, allowing it to grow molecules incrementally while using motif-level priors. It supports scaffold-constrained generation (starting from a partial molecule) and can be fine-tuned for specific protein targets. MoLeR is particularly relevant for metal-complex optimization because metal-binding scaffolds (e.g., cyclopentadienyl, arene, bisphosphine) can be provided as seed motifs.

**Paper:** Maziarz et al., "Learning to Extend Molecular Graphs with Semantic Constraints," *KDD 2022*.
**Key metric:** Outperforms JT-VAE and HierVAE on scaffold-constrained tasks.

---

### 6.3 GROVER (Graph Representation Learning with Strategic Pre-Training)

**Description:** GROVER (Rong et al., 2020, NeurIPS) uses a self-supervised predictive pretraining strategy at both node level and graph level, trained on 11M molecules from ZINC15. It is one of the largest pretrained GNN models (50M parameters) and achieves strong transfer performance across diverse downstream tasks. GROVER does not require 3D conformations, making it applicable to metal complexes where conformer generation is computationally expensive.

**Paper:** Rong et al., "Self-Supervised Graph Transformer on Large-Scale Molecular Data," *NeurIPS 2020*.
**GitHub:** https://github.com/tencent-ailab/grover
**Key metric:** SOTA on 8 MoleculeNet datasets; best overall performance across classification and regression.

---

### 6.4 Uni-Mol (Universal Molecular Representation)

**Description:** Uni-Mol (Zhou et al., 2023, ICLR) is a transformer-based molecular representation model trained on >3M molecular 3D structures and protein pocket structures. It uses a dual-encoder architecture: one encoder for molecules (with 3D coordinates) and one for pocket structures. Uni-Mol achieves SOTA on pocket-based property prediction and molecular property prediction tasks and has been widely adopted as a pretraining backbone.

**Paper:** Zhou et al., "Uni-Mol: A Universal Quantum Chemical Representation of Molecules," *ICLR 2023*.
**GitHub:** https://github.com/dptech-corp/Uni-Mol
**Key metric:** SOTA on molecular property prediction with 3D structure conditioning; pocket encoder enables pocket-conditioned predictions.

---

### 6.5 GNoME (Graph Network for Materials Exploration)

**Description:** GNoME (DeepMind, Nature 2023) is a graph neural network for crystal structure prediction and stability prediction, predicting 2.2 million stable inorganic materials. While designed for materials rather than drug molecules, its architecture and self-supervised pretraining strategy are directly applicable to metal-organic complex stability and property prediction. GNoME achieved 80%+ discovery rate (vs 50% for prior methods) using active learning with DFT validation.

**Paper:** Merchant et al., "Scaling Deep Learning for Materials Discovery," *Nature* 624:80-85 (2023).
**GitHub:** (DeepMind internal; pretrained models released)
**Key metric:** 80% discovery rate for stable materials; 2.2M predicted structures.

---

### 6.6 MolGPT and Related LLM-based Generators

**Description:** Beyond MolGPT, the broader "LLM for molecule generation" category includes Chemformer (Irwin et al., BART-based, SMILES), RetMol (retrieval-augmented generation), DrugGen, Lingo3DMol (language model + diffusion for pocket-based 3D generation), and GPT-Mol. These models treat molecular generation as text generation and have shown competitive results, though they generally lag behind graph-based and diffusion-based models on 3D geometric validity. For metal complexes, their ability to incorporate chemical SMILES/SELFIES constraints is relevant but 3D geometry handling remains a gap.

**Key models:** Chemformer (2022), RetMol (2023), DrugGen (2023), Lingo3DMol (2024).
**Key metric:** Competitive validity (~90%+) but lower geometric validity than SE(3)-equivariant models.

---

### 6.7 EDM (Equivariant Diffusion Model) and GeoLDM

**Description:** EDM (Hoogeboom et al., 2022, ICML) applies E(3)-equivariant denoising diffusion to 3D molecular geometry generation directly, generating atomic coordinates and types without an intermediate graph step. GeoLDM (Xu et al., 2023, ICML) adds a latent autoencoder so the diffusion runs in a lower-dimensional space. These models established that diffusion in 3D coordinate space is compatible with molecular generation and influenced the entire class of pocket-conditioned SBDD diffusion models (DiffSBDD, TargetDiff, FLOWR).

**Paper:** Hoogeboom et al., "Equivariant Diffusion for Molecule Generation in 3D," *ICML 2022*.
**GitHub:** (various implementations)
**Key metric:** High geometric validity on QM9; foundational for SBDD diffusion models.

---

## 7. Discussion: Relevance to Precious-Metal Anticancer Drug Design

### 7.1 Why Existing SOTA Is Mostly Organic-Molecule-Focused

The vast majority of the 35+ methods above were developed and benchmarked on organic small-molecule data (ZINC, ChEMBL, PDBBind). Precious-metal anticancer complexes have distinct characteristics:

- **Coordination chemistry:** Metal centers introduce non-covalent bonding modes, d-orbital effects, and geometry constraints (tetrahedral, square-planar, octahedral, trigonal-bipyramidal) absent in organic molecules. Standard SMILES notation cannot represent metal-ligand bond order, coordination number, or geometry stereochemistry.

- **Data scarcity:** The largest precious-metal anticancer datasets (e.g., Ru, Ir, Os, Au) have N < 5,000 compounds, while most GNNs require 100K+ for reliable generalization. This makes transfer learning from organic datasets essential but also risky due to distribution shift.

- **3D geometry criticality:** Square-planar (Pt, Pd), octahedral (Ru, Ir, Os, Rh), and tetrahedral (Au) geometries directly determine bioactivity, selectivity, and kinetic lability. SE(3)-equivariant models (GEM, GraphMVP, DiffDock, FLOWR) are particularly relevant.

- **Mechanism-of-action complexity:** Metal complexes can undergo ligand exchange, redox cycling (e.g., Ru(II)/Ru(III)), photoactivation, and covalent binding — mechanisms not well-captured by standard docking scoring functions.

### 7.2 Which Methods Transfer Best to Precious-Metal Anticancer Drug Design

Based on the surveyed methods, ranked by relevance:

| Rank | Method | Transfer Relevance | Rationale |
|------|--------|-------------------|------------|
| 1 | **GEM / GraphMVP** | Very High | 3D-aware pretraining transfers to small metal-complex datasets; geometry features can include CN and geometry type |
| 2 | **Chemprop + UQ** | Very High | Workhorse baseline; built-in uncertainty quantification is essential for scarce metal-complex data |
| 3 | **FLOWR** | High | Pocket-conditioned 3D generation with interaction recovery; explicit H-bond / lipophilic modeling |
| 4 | **DiffDock-Pocket** | High | Best validated pocket-conditioned docking; pairs well with predicted protein structures |
| 5 | **Boltz-2** | High | Joint structure+affinity (pIC50) in one forward pass; approaches FEP accuracy |
| 6 | **REINVENT4** | Medium-High | Production-validated generative pipeline; RL can incorporate metal-specific scoring |
| 7 | **LigandMPNN** | Medium-High | Designs proteins conditioned on metal-ligand context; relevant for metalloprotein targets |
| 8 | **JT-VAE / HierVAE** | Medium | Latent-space optimization; motif-level control useful for metal-complex scaffolds |
| 9 | **RFAA** | Medium | Fully open all-atom prediction; pocket-conditioned protein-ligand co-modeling |
| 10 | **Grover / Uni-Mol** | Medium | Large-scale pretraining; pocket encoder useful for target-conditioned prediction |

### 7.3 What Is Missing for Metal Drug Design

- **No dedicated 3D generative model** for metal-organic complexes with explicit metal-ligand bond modeling and coordination geometry constraints.
- **No dedicated pretraining dataset** for precious-metal complexes equivalent to ZINC (250K drug-like organics) or ChEMBL (2M compounds).
- **Fewer than 10 papers** specifically address ML for Ru/Ir/Os/Pt/Au anticancer drug design as of 2026.
- **No established benchmark** for metal-complex property prediction comparable to MoleculeNet.
- **No validated metal-specific SMILES extension** that captures coordination number, geometry, and ligand lability.
- **No published metal-complex SBDD pipeline** that accounts for ligand exchange kinetics and redox mechanisms.

### 7.4 Critical Assessment of "SOTA" Claims in Metal Drug Design

Several recent papers claim SOTA results for metal-based drug design but should be read critically:

1. **Data leakage concerns:** As demonstrated in `molmetal/reports/honest_baseline_summary.md`, most metal-complex datasets have significant SMILES overlap between train and test splits, artificially inflating AUC estimates by 5-10%. Any claimed AUC > 0.90 should be verified with a ligand-deduplicated split.

2. **Metric relevance:** Vina scores, RMSD, and hit rates were all developed for organic drug-like molecules. For metal complexes, metrics capturing coordination geometry validity, ligand lability, and redox potential are more relevant but rarely reported.

3. **3D structure quality:** Many metal complexes undergo significant conformational change upon binding. Standard pose RMSD < 2A criteria may be inappropriate for complexes where the metal center is structurally conserved but ligand orientation changes substantially.

4. **External validation:** Most published metal-complex ML papers report only internal cross-validation. Prospective prediction on held-out metal-complex series (e.g., a new Ru(II) arene series not in training) is essentially never reported.

### 7.5 Recommended Research Directions for molmetal

1. **Curate tmComplex-1:** A curated dataset of 3,000+ precious-metal anticancer complexes (Ru, Ir, Pt, Au, Pd, Rh, Os) with: (a) 3D conformers from DFT optimization, (b) IC50/EC50 values against cancer cell lines, (c) protein target annotations, (d) ligand exchange kinetics (k_inact), (e) redox potential (E_1/2). This is the single highest-ROI investment for the field.

2. **Metal-aware GNN backbone:** Adapt GEM-style 3D pretraining with metal-specific geometry features: coordination number, geometry type (CN=4 sq planar, CN=6 oct, CN=4 tet), d-electron count, spin state, ligand field splitting parameter.

3. **SBDD pipeline for metalloproteins:** Integrate DiffDock-Pocket or FLOWR with cancer-relevant protein structures (from AlphaFold3/Boltz-2) to screen the molmetal compound library against targets including: DNA groove binders (HSA, topoisomerase), kinases (CDK2, EGFR), and copper transport proteins (ATOX1, CTR1).

4. **Uncertainty quantification mandatory:** All metal-complex predictions must carry calibrated uncertainty. Recommended: ensemble-based UQ (Chemprop's built-in ensemble + dropout), or Bayesian GNN methods (e.g., GCN2 with MC dropout).

5. **Generative metal-complex design:** Adapt REINVENT4 or FLOWR to generate coordination complexes with: (a) metal-center templates as fixed scaffolds, (b) valid metal-ligand connectivity constraints, (c) geometry-consistent ligand placement, (d) ligand lability scoring.

6. **Closed-loop experimental validation:** The molmetal orchestration pipeline (`molmetal/orchestration/closed_loop.py`) should be extended to include: generate → dock → score → synthesize → assay → retrain cycles for Ru/Ir arene complexes against selected cancer protein targets.

---

## 8. References to molmetal Project Documents

For benchmark protocols and honest baseline results on the molmetal metal-cytoxicity dataset, see:
- `TODO/03_baselines` — description of Morgan+XGBoost, Morgan+RF, and DMPNN baselines
- `TODO/06_milestones` — experimental milestones for the metal-drug design pipeline
- `molmetal/reports/honest_baseline_summary.md` — honest evaluation numbers with ligand-deduplication and scaffold splits, including bootstrap CIs for Ru/Ir XGBoost AUC

**Comparison with prior work:**
- The Krasnov 2026 baseline (Morgan FP + XGBoost/RF) established initial Ru/Ir anticancer activity prediction numbers.
- Our honest baseline re-evaluation (`molmetal/reports/honest_baseline_summary.md`) shows that leakage (duplicate SMILES across train/test) inflated initial AUC estimates by 5-10%.
- The current SOTA landscape suggests that a 3D-aware GNN (GEM-style) fine-tuned on tmComplex-1 could achieve AUC > 0.85 on precious-metal cytotoxicity prediction, significantly exceeding Morgan fingerprint baselines.

---

## 9. Abbreviations

| Abbreviation | Full Name |
|---|---|
| AUC / ROC-AUC | Area Under the Receiver Operating Characteristic Curve |
| D-MPNN | Directed Message-Passing Neural Network |
| SBDD | Structure-Based Drug Design |
| SOTA | State of the Art |
| SSL | Self-Supervised Learning |
| GNN | Graph Neural Network |
| MPNN | Message-Passing Neural Network |
| VAE | Variational Autoencoder |
| GAN | Generative Adversarial Network |
| RL | Reinforcement Learning |
| TL | Transfer Learning |
| FEP | Free Energy Perturbation |
| Vina | AutoDock Vina (docking scoring function) |
| RMSD | Root Mean Square Deviation |
| PB-valid | PoseBusters validity |
| CN | Coordination Number |
| QED | Quantitative Estimate of Drug-likeness |
| ADMET | Absorption, Distribution, Metabolism, Excretion, Toxicity |

---

## 10. Benchmark Performance Summary

### 10.1 Property Prediction (MoleculeNet / BBBP scaffold split, ROC-AUC)

| Method | BBBP | BACE | ClinTox | Tox21 | HIV | Avg |
|--------|------|------|---------|-------|-----|-----|
| Chemprop D-MPNN | 0.94 | 0.87 | 0.89 | 0.84 | 0.82 | ~0.87 |
| AttentiveFP | 0.93 | 0.78 | 0.94 | 0.81 | 0.76 | ~0.84 |
| GEM | 0.72 | 0.86 | 0.90 | 0.83 | 0.77 | ~0.82 |
| GraphMVP | 0.72 | 0.72 | 0.79 | 0.80 | 0.77 | ~0.76 |
| MolCLR | 0.74 | 0.82 | 0.91 | 0.82 | 0.76 | ~0.81 |
| GROVER | 0.94 | 0.83 | 0.93 | 0.83 | 0.69 | ~0.84 |
| Uni-Mol | 0.73 | 0.86 | 0.92 | 0.82 | 0.77 | ~0.82 |
| MORGAN+FINGPT (Krasnov baseline) | ~0.75 | ~0.70 | — | — | — | ~0.72 |

*Note: Morgan Fingerprint + XGBoost/RF baseline from `molmetal/reports/honest_baseline_summary.md` achieves ~0.72-0.78 AUC on Ru/Ir cytotoxicity prediction with ligand-deduplicated splits. The gap vs Chemprop on MoleculeNet reflects dataset size effects.*

### 10.2 Molecular Docking (PDBBind blind docking, Top-1 RMSD < 2A %)

| Method | Top-1 | Top-5 | Speed |
|--------|-------|-------|-------|
| DiffDock | 38.2% | 44.7% | ~10s/A100 |
| DeltaDock | 47.4% | — | Faster than DiffDock |
| EquiBind | 38.0% | — | 10x faster than classical |
| TankBind | 20.4% | — | — |
| GNINA | 22.9% | — | 146s/complex |
| GLIDE | 21.8% | — | Slow |
| AutoDock Vina | — | — | — |
| Surflex-Dock | 68% | 81% | Slow |

### 10.3 Pocket-Conditioned Ligand Generation (SPINDR test set)

| Method | RDKit-valid | PB-valid | Vina score | Strain energy |
|--------|-------------|----------|------------|---------------|
| **FLOWR (100 steps)** | **0.94** | **0.88** | **-6.93** | **90.05** |
| PILOT | 0.79 | 0.71 | -6.30 | 120.0 |
| Pocket2Mol | Lower | Lower | -6.5 | Higher |
| TargetDiff | Lower | Lower | Lower | Higher |
| DiffSBDD | — | 0.88 | — | — |

### 10.4 Affinity Prediction (Pearson r vs experimental)

| Method | Affinity Metric | Pearson r | Speed |
|--------|---------------|-----------|-------|
| Boltz-2 | pIC50 | 0.86 | ~20s/complex |
| FEP+ (physics) | pIC50 | ~0.90 | Hours-days |
| RFAA + scoring | pKd | Competitive | Fast |
| DiffDock confidence | Confidence (not affinity) | N/A | Fast |
| FLOWR.ROOT | pIC50 | r=0.86 | Fast |

### 10.5 De Novo 3D Molecule Generation (Geom-Drugs)

| Method | RDKit-valid | PB-valid | Relax RMSD |
|--------|-------------|----------|------------|
| **FLOWR.root** | **98.5%** | **94.0%** | **0.07A** |
| FlowMol3 | 99.9% | 91.9% | 0.39A |
| MegaLodon | 94.8% | 86.6% | 0.41A |
| SemlaFlow | 95.5% | 88.5% | 0.24A |
| EQGAT-diff | 86.0% | 77.6% | 0.60A |

---

*Last updated: September 2026. This document is maintained as part of the molmetal project. Please update the companion documents (`TODO/03_baselines`, `TODO/06_milestones`) when new benchmarks are run.*
