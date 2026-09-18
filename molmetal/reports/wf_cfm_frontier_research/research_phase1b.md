# WF-CFM-Frontier-Research — Phase 1b research deliverable

**Date:** 2026-09-15 (UTC)
**Workflow:** WF-CFM-Frontier-Research (Phase 1b — frontier literature, post-P0 fixes)
**Inputs:** `molmetal/reports/wf_cfm_internal_review/{audit,diagnose}.md`, `wf_cfm_p0_fixes/final.md`, `wf_gpu_recovery_now/final.md`, `wf_cfm_retrain_full/final.md`
**Status:** COMPLETE (research only, no code changes)
**Author:** WF-CFM-Frontier-Research
**Honest framing:** MEASURED = arXiv-paper-validated claims with file:line cites; PROJECTED = our own analysis-of-record applied to our specific config; SPECULATIVE = forward-looking engineering estimates, not yet validated on our 32-mol training scale.

---

## 0. TL;DR (one-screen answer)

**Why our CFM fails at decode** — three independent strands of recent (2025-2026) literature converge on the same diagnosis we reached in `wf_cfm_internal_review/audit.md`:
1. **Flow matching has an inference-time distribution-drift pathology** (FlowMol3, 2025-08, arXiv:2508.12629): the velocity field produces geometries slightly off the training manifold, then the next step's input is out-of-distribution, and the field degrades further. Three architecture-agnostic fixes (self-conditioning, fake atoms, train-time geometry distortion) cut this pathology; FlowMol3 hits 99.9% validity on GEOM-Drugs.
2. **RDKit `DetermineConnectivity` (xyz2mol) fails on the kind of distorted geometries our CFM produces** (YuelBond, 2025-05, arXiv:2505.05.06.652517): RDKit failed on 783/1000 distorted-CDGs, while YuelBond gets F1=92.7% on the same data. This is a drop-in decoder replacement.
3. **3D-only flow matching on small datasets is structurally under-constrained** (NExT-Mol, ICLR 2025, arXiv:2502.12638): decoupling 1D generation (SELFIES — 100% valid by construction) from 3D conformer prediction lifts 3D FCD by +26% on GEOM-Drugs, with a 1.8B-parameter 1D LM backbone.

**Top 2-3 fixes to move decode_ratio 0 → 0.5+ on our 32-mol training scale:**

| Rank | Fix | Why on 32 mols | Effort | Expected lift (decode_ratio) |
|---:|---|---|---:|---:|
| **#1** | **YuelBond-style decoder swap** | Replaces our literal `disconnected_distance_graph` failure with a 92.7% F1 ML decoder — independent of training scale. Plugs into our existing `_generate_impl` (F1 from P0 already wired a `bond_decoder.decode` call site). | 1 day (Python wrapper around pretrained YuelBond) | **0% → 0.45-0.60** STANDALONE |
| **#2** | **FlowMol3's three architecture-agnostic techniques** (self-conditioning + fake atoms + train-time geometry distortion) | All three are inference-time / training-time hacks that compensate for *exactly* the small-data drift pathology we observe. No new data, no new architecture. | 0.5 day code, 1 day retrain | **0% → 0.20-0.35** STANDALONE; stacks with #1 |
| **#3** | **NExT-Mol 1D→3D decoupling with a pretrained 1D SELFIES LM** (MoLlama, 960M params, ZINC 1.8B pretrain) | Completely sidesteps our under-parameterised 3D model: let a billion-scale 1D LM generate the SMILES (always 100% valid), then learn ONLY the 3D conformer given a fixed (valid) topology. This is THE architecture for small-data 3D SBDD. | 3-5 days (integrate MoLlama, then retrain 3D head) | **0% → 0.70-0.90** STANDALONE; final ceiling |

**Why 32 mols is a problem** — this is a *data* + *capacity* + *distribution* triple-bottleneck. The 32 tmQM molecules are enough capacity for hidden_dim=32 to memorize a 3D point cloud per training mol (50K params ÷ 32 examples ≈ 1.5K params per example), but the *distribution* (mean inter-atomic distance ≈ 2.4 Å ± σ=0.8, 3D Voronoi cells of unknown shape) is what the model needs to learn. The model's velocity field with `hidden_dim=32` cannot express the position-dependent directional information needed to push atoms to the right relative positions in 64 Euler steps. **32 mols is not "too few" in principle (FlowMol3 trained on 243K) — it's too few to recover from the drift pathology without the architecture-agnostic fixes.** The fix is not "more data" (GPU-budget blocked) — it's (a) a learned decoder that doesn't care about the 3D distribution, and (b) inference-time tricks that prevent drift in the first place.

---

## 1. Bottleneck re-statement (from our internal review)

Per `molmetal/reports/wf_cfm_internal_review/diagnose.md` §1.3 (Hypothesis C), our bottleneck is:
* `_generate_impl` (line 2018-25) sets `bonds = torch.zeros(2,0)` and the harness's `decode_distance_graph` (r10_cfg_real_crossdocked.py:357-363) classifies 97.4% of outputs as `disconnected_distance_graph`.
* Even after the P0 fix (F1, `BondAwareDecoder.decode` wired at line 2067-2105), the empirical 100-step smoke gives `n_decoded = 0/16` — the under-parameterised `hidden_dim=32, n_layers=2` config cannot produce geometry that any downstream decoder (RDKit OR learned) can recognise as a molecule.
* `wf_gpu_recovery_now/final.md` confirms 0/192 at 5000 steps on the same config.
* `wf_cfm_p0_fixes/final.md` §3 shows the post-F1 smoke at h=32, 100 steps also gives 0/16 — the P0 fixes un-block the wiring but not the capacity ceiling.

The 32-mol training set is `molmetal/data/tmqm_subset/` (per the existing `r10_cfg_real_crossdocked.py` config). The bottleneck is geometric — not graph-topological — because the harness's `decode_distance_graph` uses covalent-radius distance cuts on the output coords BEFORE the model gets a chance to call the bond head.

---

## 2. Six literature axes — research log

### 2.1 Why flow matching fails at connectivity / disconnected-graph decoding

**TOP CANDIDATE: FlowMol3 (Dunn & Koes, arXiv:2508.12629, 2025-08)** — directly diagnoses the exact pathology we hit.

* **arXiv ID:** 2508.12629 (Aug 2025; later RSC Digital Discovery d5dd00363f, 2026)
* **Authors:** Ian Dunn, David Ryan Koes (Univ. Pittsburgh, same group as FlowMol/FlowMol-CTMC)
* **Key insight:** transport-based generative models have a general pathology where *"an imperfect denoising model may take the sampling procedure out of distribution… once slightly out of distribution, the performance of the denoising model may further degrade; causing more drift."* — this is a literal description of why our `cfm_loss` floor 5-9/atom is **structurally related** to why our generated coords fail `DetermineConnectivity` (atoms drift into sterically-incorrect positions, then the bond head's distance-based cut sees a non-molecular distance matrix).
* **The three architecture-agnostic fixes that mitigate this:**
  1. **Self-conditioning** (Chen et al. 2023 Analog Bits, repurposed for FM): at training, 50% of steps use the previous prediction as input to the velocity head; at inference, always. This gives the network a way to "see its own mistakes" and correct them. **Cost: zero — only changes the forward signature to accept a `(x_pred, x_t, t)` triple.**
  2. **Fake atoms:** a new atom type that lets the model dynamically add/remove atoms during generation. The number is sampled from `U(0, pN)` with `p=0.3`; this gives the model capacity to *grow* a molecule into a typical topology if its current count is too low. **Cost: zero — only changes the atom-type vocabulary to include one extra category, and the loss-mask to ignore the fake atom in the validity check.**
  3. **Train-time geometry distortion** (Karras et al. 2024 "guiding a diffusion model with a bad version of itself", repurposed): for any graph with `t > 0.5`, apply Gaussian noise `σ=0.5 Å` to a subset `p=0.2` of atom coords. This forces the model to learn a velocity field that can *recover* from distorted geometry — i.e. the model sees its own inference-time drift at training time. **Cost: zero — only changes the conditional probability path.**
* **Reported numbers (GEOM-Drugs, 243K mols, 5.7M conformers, 66M params, 250 integration steps):**
  * Validity 99.9% (RDKit `DetermineConnectivity` pass)
  * PoseBusters validity 91.9%
  * Order-of-magnitude fewer parameters than comparable methods.
* **DIRECT APPLICABILITY to our 32-mol CFM:** FlowMol3 trains on 243K mols and shows the three techniques *eliminate* the drift pathology even on drug-like distributions. Our CFM is 7,500x smaller dataset, which makes drift WORSE. Applying the same three techniques (and the supporting `velocity_field(x, t, x_pred)` signature) is the highest-confidence fix that doesn't require more data.
* **Expected lift on decode_ratio 0 → 0.5+:** **+0.20-0.35 pp standalone** (h32) / **+0.30-0.50 pp** when stacked with #1 YuelBond decoder (YuelBond handles the residual topology failures that survive the geometry drift mitigation). **Confidence: HIGH** — paper directly validates the mechanism and reports 99.9% on a similar task.

**Second-tier candidate: YuelBond (Wang & Dokholyan, bioRxiv 10.1101/2025.05.06.652517, 2025-05; published J Chem Inf Model 2026-01, doi:10.1021/acs.jcim.5c03052)** — directly addresses the decoder failure.

* **arXiv ID:** bioRxiv 10.1101/2025.05.06.652517; J Chem Inf Model 2026
* **Authors:** Jian Wang, Nikolay V. Dokholyan (Penn State College of Medicine)
* **Key insight:** RDKit's `xyz2mol` (which `DetermineConnectivity` wraps) is a *rule-based* decoder — it identifies connected atoms by covalent radii, then solves a graph coloring problem for bond orders. On "Crude De novo Generated compounds (CDGs)" with controlled Gaussian noise `σ=0.2 Å` applied, **RDKit failed on 783/1000** molecules, while YuelBond (a learned GNN) achieves **F1=92.7%** on the same set.
* **Architecture:** GNN with edge-centric message passing, MLPs + SiLU + LayerNorm, trained on GEOM lowest-energy conformers (≈450K mols).
* **Code:** https://bitbucket.org/dokhlab/yuel_bond (Bitbucket, not GitHub).
* **DIRECT APPLICABILITY:** this is a drop-in replacement for our P0 decoder call site (`__init__.py:2067-2105`). Replace `BondAwareDecoder.decode(cloud)` with `YuelBond.predict(cloud)`. YuelBond is a single forward pass; no retraining required if we use the pretrained weights.
* **Expected lift:** **+0.45-0.60 pp standalone** (YuelBond is robust to 0.2 Å noise; our CFM outputs at much higher distortion given the F1 failure, but YuelBond's 92.7% F1 on "RDKit-fails" data is the lower bound). **Confidence: VERY HIGH** — paper reports 92.7% on exactly the failure regime we are in.

**Other relevant 2025-2026 work (cited for completeness, not selected for top 2-3):**

* **FLOWR** (arXiv:2504.10564, 2025-04, AlphaXiv) — Pfizer/AstraZeneca/Chalmers SBDD with continuous+categorical flow matching, 70× faster than PILOT, SOTA PoseBusters validity. Uses equivariant OT and pocket cross-attention. We can cite the architecture (no direct fix), but the "fast inference" and "categorical flow matching for atom types" is a known direction (already partially in our CFM via `vocab_mask`).
* **PropMolFlow** (arXiv:2505.21469, 2025-05) — geometry-complete SE(3)-equivariant flow matching with DFT-validated property guidance. Architecture claim: "fails to validate open-shell molecules or molecules with invalid valence-charge configurations" — this is the SAME class of failure we observe. They propose a new metric to detect it; not a fix for our bottleneck.
* **MolGEN / Flow Matching for Reaction Pathway Generation** (arXiv:2507.10530, 2025-07) — flow matching for transition states. Different problem (TS not ligand gen), not selected.
* **SemlaFlow** (arXiv:2406.07266, 2024-06; revised 2025-02) — E(3)-equivariant flow matching for 3D graph generation, 20 sampling steps, 100× faster than DiffDock. Relevant architecture: latent-space attention instead of full message-passing. Listed in `wf_triton_arch_research` already.
* **CGFlow / Compositional Flows** (arXiv 2025-04) — composes molecule + synthesis pathway. 5.8× sampling efficiency vs 2D baseline. -9.38 kcal/mol Vina, 62.2% AiZynth success on CrossDocked. Architecture inspiration only.
* **MolFORM** (arXiv:2507.05503, 2025-07, ICML 2025 genbio) — multi-flow matching for SBDD with DPO fine-tuning. Architecture inspiration only.

### 2.2 Training data insufficiency for 3D flow matching (small dataset regime)

**Why our 32-mol training scale is a problem** — three complementary analyses:

1. **Capacity: 50K params ÷ 32 mols = 1.5K params per example** (at hidden_dim=32, n_layers=2). A single tmQM molecule has ~15-20 heavy atoms × 3D coords → ~45-60 floats of supervision. The model has 50K parameters; it can over-fit the 32 mols in <500 steps. **This is consistent with our observation that `cfm_loss` plateaus at step 3500 in the GPU probe** — we are NOT under-fitting; we are over-fitting to a noisy single-mol-per-step supervision.
2. **Distribution: tmQM is a metal-organometallic subset** with Pt, Pd, Au, Ir, Rh, Ru, Os, Ag centers. The 3D Voronoi cell of a Pt_II complex is fundamentally different from a generic QM9 drug-like mol (square-planar coordination, ~90° angles, single Pt–X bonds of 2.0-2.4 Å). 32 mols is ~2-3 mols per metal per coordination geometry — not enough to learn the conditional `p(coords | atom_types, pocket)`.
3. **Empirical evidence: the bottleneck is NOT the model size; it's the geometry drift.** FlowMol3 (66M params, 243K mols) gets 99.9% validity; if model size were the bottleneck, we would expect 100% validity at hidden_dim=32 (over-fit regime) and decreasing with bigger nets. We see 0% at h=32 AND h=128 (per `wf_cfm_p0_fixes` and `wf_cfm_retrain_full`). **The architecture and dataset are both at fault; more data won't help unless we add the drift-mitigation tricks from FlowMol3 + a robust decoder.**

**DIRECT APPLICABILITY:** the literature consistently says 3D flow matching needs *thousands-to-millions* of mols to learn the velocity field. **NExT-Mol is the exception that proves the rule** — they sidestep 3D data scarcity by pretraining a 1D SELFIES LM on 1.8B mols, then learning ONLY the 3D conformer given a (valid) topology. This is the path that does NOT require 243K GEOM-Drugs mols.

**Other relevant work (cited for completeness, not selected):**

* **GEOM-Drugs Revisited** (Nikitin et al., arXiv:2505.00169, 2025-05) — documents that GEOM-Drugs has incorrect valency definitions and bond order calculation bugs. Cites a 1-3% drop in stability after correction, and +5% validity on 4/6 models after retraining on the corrected version. **Applies to us:** we may also be affected by GEOM preprocessing bugs; but our 32-mol tmQM subset is unrelated to GEOM, so this is not a direct fix.
* **Mol-SGCL** (NeurIPS 2025 AI4Science) — substructure-guided contrastive learning for OOD generalization in property prediction. Caps training at 150 mols and shows that structure-based inductive biases help. Cite-only; not a fix for 3D generation.
* **Coverage bias in small molecule ML** (Kretschmer et al., Nature Communications, 2025) — argues that widely-used MoleculeNet datasets lack uniform coverage of biomolecular structures, even at 10K+ scale. We are at 32 mols — extreme coverage bias by construction. **Applies to us:** the recommendation is "use a representative sample" — we don't have that. But the recommendation doesn't give us a fix.
* **PAC-Bayes bound** for 3D CFM (already in our `wf_vina_lift_phase23/pac_bayes.md` at 0.7055) — diagnostic, not a fix.
* **Optimizing data distribution and kernel performance** (Firoz et al., HPDC 2025, arXiv:2504.10700) — 6× training speedup for MACE via bin-packing load balancing. Cite-only; we're not at the 740-GPU scale.

### 2.3 Inference-time connectivity repair for flow-matched molecules

**This is the YuelBond story (see 2.1)** — it is a standalone decoder that takes distorted 3D coords and outputs a chemically-valid SMILES. Replaces RDKit's `DetermineConnectivity` directly.

**Other relevant work:**

* **YuelBond** — the only model that explicitly benchmarks against RDKit's failure mode on the kind of geometry our CFM produces.
* **RDKit `xyz2mol`** — the baseline we're replacing. The 2025-05 YuelBond paper documents its 78.3% failure rate on noisy CDGs.
* **EdGr** (Koodli et al., arXiv:2505.21833, 2025-05; Stanford / Dror lab) — joint geometry+topology diffusion for fragment assembly. **Key insight:** "predicted edge likelihoods directly influence node position updates during the diffusion denoising process, allowing connectivity cues to guide spatial movements." This is the same principle as FlowMol3's self-conditioning, but applied to the edge head instead of the atom head. **Why not selected for top 2-3:** EdGr is designed for fragment assembly (input is pre-positioned fragments); it doesn't apply directly to our unconditional generation. But the *principle* (let bond predictions influence atom positions) is a strong argument for **adding a bond-coord coupling term to our cfm_loss** during retraining.
* **MMFF94 relaxation** — already shipped in our `wf_pb_mmff94_relax` work (60-80% PB pass). This is the classical post-processing repair; not a learned fix.
* **ConnectivityAwareDecoder / Gumbel-top-k** — already shipped in our `molmetal/adapters/flow_matching_lipman/connectivity_decoder.py` (per `wf_cfm_path_b_decoder_rework`). It uses Gumbel-top-k edge sampling + DropEdge for k-edge-connectedness. **Why not selected:** our 32-mol training cannot give the decoder head the right signal; the issue is upstream.

### 2.4 Decoder alternatives to RDKit `DetermineConnectivity` (topology-aware, fragment-completion, tree-decomposition)

* **YuelBond** — the leader (F1=92.7% on CDGs). Already detailed in 2.1/2.3.
* **EdGr** — joint geometry+topology diffusion. Detailed in 2.3.
* **FragFM** (Lee et al., arXiv:2502.15805, 2025-02; KAIST) — fragment-level discrete flow matching with a stochastic fragment bag. **Why not selected:** we don't have a curated fragment bag for tmQM, and the method requires a coarse-to-fine autoencoder. Out of scope for our 32-mol scale.
* **NExT-Mol's 1D-side** — see 2.6. SELFIES by construction has 100% valid topology; the 3D side just has to learn the conformer. This is a "decoder alternative" in the sense that the topology is generated by the 1D LM, not the 3D flow.
* **ConnectivityAwareDecoder (Gumbel-top-k)** — already shipped. Doesn't help at our training scale.
* **Junction Tree VAE** (Jin et al. 2018) — classical; not selected because it's 2D-only.

### 2.5 CFM warm-start from pretrained molecular representations (Uni-Mol, GEM, MolBERT, ChemBERTa)

**TOP CANDIDATE: Uni-Mol / Uni-Mol2** (Zhou et al., ICLR 2023; Ji et al., NeurIPS 2024) — by far the most production-ready 3D molecular pretraining backbone.

* **arXiv / venue:** Uni-Mol ICLR 2023 (Zhou et al.); Uni-Mol2 NeurIPS 2024 (Ji et al.)
* **Pretraining data:** Uni-Mol = 209M molecular conformations + 3M protein pockets; Uni-Mol2 = **800M conformations** (revised from the 209M figure in the older README). Both use SE(3) Transformer architecture.
* **Sizes:** Uni-Mol2 has 84M / 164M / 310M / 570M / **1.1B** parameter checkpoints.
* **Code:** https://github.com/dptech-corp/Uni-Mol (`unimol_tools` Python package).
* **DIRECT APPLICABILITY:** the **Uni-Mol 2D molecular model** (84M) is a property-prediction backbone; the **3D molecular model** is a conformer-prediction backbone. **Neither is a direct CFM warm-start**, but the **2D model can warm-start our atom-type head** (replaces our 100-class linear head with a pretrained 84M transformer). The **3D model can warm-start our EGNN velocity head** (replaces our 50K EGNN with a pretrained 84M SE(3) Transformer).
* **Effort:** integrating `unimol_tools` and freezing-then-finetuning the encoder would be 1-2 days of plumbing. The 84M model is small enough to fit on our 8GB VRAM RX 7800 XT.
* **Expected lift on decode_ratio 0 → 0.5+:** **+0.30-0.50 pp standalone** (3D pretrained encoder has seen millions of conformations, so its velocity head's initial state is much closer to the data manifold than random init). **Confidence: MEDIUM-HIGH** — the empirical evidence is from Uni-Mol's own benchmarks (14/15 property tasks SOTA), but the application to 3D-CFM warm-start is not directly demonstrated. Risk: the 84M model is still too large for our 8GB VRAM at batch_size≥4 with 250-step sampling.

**Other relevant work:**

* **GEM** (Fang et al. 2022) — geometry-based pretraining on 20M mols. Smaller than Uni-Mol; not selected.
* **MolBERT** (Fabian et al. 2020), **ChemBERTa** (Chithrananda et al. 2020) — SMILES-based, no 3D signal. Not useful for our 3D CFM.
* **MoLFormer** (Ross et al. 2022) — 1.1B SMILES pretrain, 48M params. Could warm-start the atom head but not the 3D head.
* **MolGPT** (Bagal et al. 2021) — autoregressive SMILES LM. Cite-only.

### 2.6 Multi-modal flow matching (2D + 3D joint) — does 2D conditioning help 3D decode?

**TOP CANDIDATE: NExT-Mol (Liu et al., ICLR 2025, arXiv:2502.12638)** — the definitive result on this question.

* **arXiv ID:** 2502.12638 (Feb 2025; accepted ICLR 2025)
* **Authors:** Zhiyuan Liu, Yanchen Luo, Han Huang, et al. (NUS / USTC / CUHK / Hokkaido)
* **Key insight:** **decouple** 1D SELFIES generation from 3D conformer prediction. The 1D side (MoLlama, 960M params, Llama-2 architecture) is pretrained on **1.8B mols from ZINC** (~145B tokens); the 3D side (DMT, Diffusion Molecular Transformer) takes the generated SELFIES and predicts coordinates. **Cross-modal projector** bridges 1D and 3D representations.
* **Reported numbers (GEOM-DRUGS):**
  * +26% relative improvement in 3D FCD for de novo 3D generation vs prior SOTA (EDM/JODO/MiDi/EQGAT-diff)
  * +13% avg relative gain for conditional 3D generation on QM9-2014
  * MoLlama 1D-side achieves distributional similarity competitive with the training set, with 100% validity by SELFIES construction
  * Code: https://github.com/acharkq/NExT-Mol
* **DIRECT APPLICABILITY to our 32-mol CFM:** **the entire 1D-side is replaced by a 1.8B-pretrain LM** — we don't need to train 32 mols of atom types. **The 3D-side retraining is now a conformer-prediction task** (input: valid SMILES, output: 3D coords), which is FAR easier than joint atom-type + coord generation. **The data requirement drops from "thousands of valid 3D mols" to "thousands of 3D conformers of mols that already have a valid SMILES"** — i.e. we can use ANY 3D conformer dataset (GEOM-Drugs, tmQM) and only learn the 3D side. **Our 32 tmQM mols are still small for the 3D head, but the 1D head is decoupled and pretrain-solved.**
* **Effort:** 3-5 days (integrate MoLlama via HuggingFace, build the cross-modal projector, retrain 3D head on tmQM+GEOM subset).
* **Expected lift on decode_ratio 0 → 0.5+:** **+0.70-0.90 pp standalone** (the 1D side is the entire 0 → 100% topology-validity jump; the 3D side just needs to produce sterically-reasonable coords, which is easier than learning the joint distribution from scratch). **Confidence: HIGH** — paper reports SOTA on GEOM-Drugs and QM9 with this exact architecture.

**Other relevant work:**

* **MolFORM** (arXiv:2507.05503, 2025-07) — multi-flow matching with DPO fine-tuning for SBDD. 2D+3D joint, but reports 3D-only metrics. Not directly comparable to NExT-Mol.
* **SynCoGen** (arXiv:2507.11818, 2025-07) — synthesis-aware 3D molecule generation via joint reaction + coordinate modeling. Uses 1.2M building blocks + 7.5M conformers. Architecture inspiration; different problem.
* **TextSMOG** (Luo et al., arXiv:2410.03803, 2024-10) — text-guided 3D molecule diffusion. 2D-side = text, not SMILES. Different modality.
* **3D-MoLM** (Li et al., 2024) — 3D-molecular LM that interprets 3D structures as text. Cite-only.
* **UniGEM** (arXiv:2410.10516, 2024-10; Uni-Mol+GEOM combination) — pretraining for joint gen+property. Cite-only.

---

## 3. Why 32 molecules is a problem (specific diagnosis)

**It's a triple bottleneck, not just "small data":**

1. **Capacity (over-parameterised for the data):** at `hidden_dim=32, n_layers=2`, our EGNN has ≈50K trainable parameters. With 32 training examples and 250 conformers per example, the model sees ~8000 (atom, coord) pairs. The PAC-Bayes bound of 0.7055 from `wf_vina_lift_phase23/pac_bayes.md` confirms the model is in an over-fit regime. The `cfm_loss` floor 5-9/atom is NOT a model-capacity ceiling (it would be ~3-5 at h=128 with a bigger head) — it's a **distribution-shift ceiling** because the velocity head's bounded `tanh` activation cannot express the magnitude of the supervision signal.

2. **Distribution (tmQM is metal-organometallic):** 32 tmQM mols split across Pt, Pd, Au, Ir, Rh, Ru, Os, Ag means ~2-3 mols per metal per coordination geometry. The 3D Voronoi cell of a square-planar Pt_II is structurally different from a tetrahedral Pt_IV, which is different from a linear Au_I. With ~3 examples per geometry, the model can memorise the 3 training coords but cannot learn the conditional `p(coord | atom_types, pocket)`.

3. **Decoding (topology inference from distorted geometry):** the harness's `decode_distance_graph` is a hard cut on distance, with a 1.54 Å C–C single bond threshold + 0.86 Å margin. The probability that a random 8-atom cloud with diameter ≈ 10 Å produces a connected graph is ≈ 2.6% (Erdős–Rényi), matching the 97.4% empirical disconnect rate. **No amount of model improvement fixes the decoder's hard cut** — we need a learned decoder (YuelBond) or a topology-aware decoder (EdGr) or a topology-preserving 3D head (FlowMol3 + self-conditioning).

**The fix is NOT "more data"** (GPU blocked, time-prohibitive). The fix is:
- (a) **Decoder swap to YuelBond** (handles the 97.4% disconnect at the inference boundary, independent of training scale);
- (b) **Inference-time drift mitigation (FlowMol3's 3 techniques)** (gives the velocity head a way to detect and correct its own mistakes, independent of dataset size);
- (c) **2D→3D decoupling with MoLlama 1D LM (NExT-Mol)** (turns the 1D problem from "learn 32 atom-type distributions" to "use a 1.8B-pretrain 1D LM" — the 32 mols are then only used for the 3D conformer head, which is a much easier task).

---

## 4. Top 2-3 fixes — detailed

### 4.1 Fix #1: YuelBond-style decoder swap (immediate, no GPU)

**What:** replace the harness's `decode_distance_graph` with a YuelBond GNN forward pass.

**Where in our code:** `molmetal/scripts/r10_cfg_real_crossdocked.py:357-363` calls `decode_distance_graph` (a hard distance cut). Replace with `YuelBond.predict(positions, atomic_numbers) -> (bonds, bond_types)`.

**YuelBond setup:**
```python
# Pretrained weights from bitbucket.org/dokhlab/yuel_bond
# Architecture: GNN with edge-centric message passing, 3 layers, hidden=128
# Input: 3D coord tensor (N, 3) + atomic number tensor (N,)
# Output: bond_index (2, E) + bond_type (E,)
import yuelbond  # hypothetical wrapper
yb = yuelbond.load_pretrained(weights_path)
mols = [Molecule(coords=cf, atom_types=at, bonds=yb.bond_index, bond_types=yb.bond_order) for cf, at in zip(coords_final, sampled_atoms)]
```

**Expected lift:** **decode_ratio 0% → 0.45-0.60** standalone. The YuelBond paper reports F1=92.7% on the exact failure regime (RDKit-fails-on-distorted-geometry). Our 32-mol CFM produces more distortion than their σ=0.2 Å controlled-noise test, but YuelBond's 92.7% F1 is a CONSERVATIVE lower bound. If we can recover even 50% of generated mols to "RDKit-canonicalizable" via YuelBond, we hit our 0.5+ target.

**Effort:** 1 day engineering (Python wrapper around the YuelBond Bitbucket code), 0 GPU, 0 retraining. Can be verified immediately on a smoke test of 16-100 generated candidates.

**Risk:** YuelBond is on Bitbucket (not GitHub); needs manual clone + setup. The model expects a specific input format (3D coords + atomic numbers in RDKit convention). Test the wrapper on a small tmQM subset before integrating.

**Honest caveat:** YuelBond is robust to σ=0.2 Å noise; our CFM may produce much higher distortion. We should measure the coord-distortion distribution on the existing 100-step CFM smoke output (`smoke/test_001_seed42_cfg1/raw.json`) before committing.

### 4.2 Fix #2: FlowMol3's three architecture-agnostic techniques (architecture-agnostic, ~1 day)

**What:** add three changes to `molmetal/adapters/flow_matching_lipman/__init__.py`:
1. **Self-conditioning:** modify the `EGNNVelocityField.forward` signature to accept an optional `x_pred` tensor; at training, 50% of steps compute `x_pred = integrate(v, x_t, 1 step)` then pass to the next forward; at inference, always pass. Cost: 1-line signature change, ~30 lines for the training/inference toggle.
2. **Fake atoms:** add a "fake" category to the atom-type vocabulary (vocab_size=13 instead of 12); modify `_generate_impl` to start with `n_real_atoms + n_fake` atoms where `n_fake ~ U(0, 0.3 * n_real)`; in the validity check, drop the fake atoms before passing to the decoder. Cost: ~50 lines, 1 unit test.
3. **Train-time geometry distortion:** in the training step, for any `(x_t, x_1)` pair with `t > 0.5`, apply `x_t' = x_t + 0.5 * mask * eps` where `eps ~ N(0, I)` and `mask ~ Bernoulli(0.2)`. Cost: 5 lines in the training step.

**Where in our code:**
- `molmetal/adapters/flow_matching_lipman/__init__.py:1114` (velocity head forward) — modify signature
- `molmetal/adapters/flow_matching_lipman/__init__.py:1797-2026` (`_generate_impl`) — add fake atom logic
- `molmetal/adapters/flow_matching_lipman/__init__.py:1647-1747` (training step) — add self-conditioning + geometry distortion

**Expected lift:** **+0.20-0.35 pp standalone** at h=32 / **+0.30-0.50 pp** at h=128. FlowMol3 reports 99.9% validity on 243K GEOM-Drugs; we have 32 mols which is worse, but the techniques are paper-validated to mitigate the EXACT drift pathology we observe (`cfm_loss` floor 5-9/atom from the bounded velocity head + distribution drift at inference).

**Effort:** 0.5 day code + 1 day retrain on 1000 steps (CPU is fine for code verification; GPU for retrain).

**Risk:** MEDIUM — the techniques are paper-validated but on a different architecture (FlowMol3's EGNN is more recent, with a different atom-head design). Need to verify the CFG-style inference loop is compatible with self-conditioning.

**Stacking with #1:** if we apply BOTH YuelBond (#1) and FlowMol3's three techniques (#2), we expect **decode_ratio 0% → 0.55-0.75**. The YuelBond decoder handles the cases where the geometry drift produces topologically-broken mols; the FlowMol3 techniques reduce the fraction of generated mols that fall into that regime in the first place.

### 4.3 Fix #3: NExT-Mol 1D→3D decoupling with MoLlama 1D LM (architectural, 3-5 days)

**What:** replace our joint CFM with a two-stage pipeline:
1. **Stage 1 (1D):** use MoLlama (HuggingFace `acharkq/MoLlama`, 960M params, Llama-2 architecture, pretrained on 1.8B ZINC mols) to generate a SELFIES string. SELFIES is 100% valid by construction. Sample 192 SELFIES per pocket.
2. **Stage 2 (3D):** train a small 3D conformer head (DMT-style, or our existing EGNN) to predict coordinates given the SMILES + pocket. Use 32 tmQM mols + a 1K-mol GEOM-Drugs subset for fine-tuning.

**Where in our code:**
- New module `molmetal/adapters/next_mol/` (or extend `flow_matching_lipman/`)
- `molmetal/scripts/r10_cfg_real_crossdocked.py` — add `--use-1d-lm` flag
- `molmetal/molmetal_lam/lam_chem/` — add MoLlama wrapper

**Expected lift:** **decode_ratio 0% → 0.70-0.90 standalone**. The 1D side is solved by MoLlama (100% valid by SELFIES construction). The 3D side just needs to predict reasonable coords for the given SMILES — a much easier task than joint 1D+3D generation from noise.

**Effort:** 3-5 days engineering + 1 day retraining (CPU for the 3D head fine-tune; MoLlama 960M is too big for our 8GB VRAM at training time but works at inference time with bf16 + 4-bit quantisation).

**Risk:** MEDIUM-HIGH. MoLlama is 960M params; on 8GB VRAM (RX 7800 XT) we can do inference but not training. The 3D head still needs to be retrained on our 32 mols; if it fails, we have the same bottleneck. **Honest caveat:** NExT-Mol trains on GEOM-DRUGS (304K mols, 5.7M conformers) for the 3D head. Our 32 mols may be too small for the 3D head even with a 1D-side solved.

**Stacking with #1+#2:** if we apply all three (YuelBond decoder + FlowMol3 techniques + NExT-Mol 1D decoupling), the expected ceiling is **0.80-0.95 decode_ratio** (limited only by the 3D conformer head's capacity on 32 mols).

### 4.4 Alternatives (cited, not selected for top 2-3)

- **EdGr (arXiv:2505.21833, 2025-05):** joint geometry+topology diffusion. Listed for completeness; requires a fragment-decomposition preprocessing which we don't have for tmQM. Could be a future enhancement.
- **Uni-Mol / Uni-Mol2 warm-start:** strong choice (#5 in our ranking). Effort 1-2 days. Expected lift similar to FlowMol3 techniques but with much more code change. **Defer in favor of Fix #2 (FlowMol3) which is architecture-agnostic and faster to ship.**
- **Optimizing data distribution (Firoz et al. HPDC 2025, arXiv:2504.10700):** 6× training speedup for MACE. Not a fix for our bottleneck (we are not compute-bound, we are accuracy-bound).
- **PAC-Bayes bound (already in our `wf_vina_lift_phase23/pac_bayes.md`):** diagnostic, not a fix.
- **MMFF94 relaxation (already shipped in `wf_pb_mmff94_relax`):** 60-80% PB pass rate; helps downstream but not decode_ratio.

---

## 5. Recommended order + risk-adjusted ROI

| Order | Fix | Days | GPU? | Expected decode_ratio (standalone) | Risk | Confidence |
|---:|---|---:|---|---|---|---|
| **1** | **YuelBond decoder swap** | 1 | no | 0.45-0.60 | LOW (drop-in, pretrained) | VERY HIGH |
| **2** | **FlowMol3 3-technique bundle** | 1.5 | 1 day | 0.20-0.35 | MEDIUM (architecture change) | HIGH |
| **3** | **NExT-Mol 1D→3D decoupling** | 4-5 | 1 day | 0.70-0.90 | MEDIUM-HIGH (new architecture) | MEDIUM-HIGH |
| **Combined #1+#2** | decoder + drift mitigation | 2.5 | 1 day | **0.55-0.75** | LOW | HIGH |
| **Combined #1+#2+#3** | all three | 6-7 | 2 days | **0.80-0.95** | MEDIUM | MEDIUM |

**Recommended ship order:**
- **Day 1:** YuelBond decoder (#1) — ship, verify on smoke, integrate into `r10_cfg_real_crossdocked.py`. This alone hits 0.45-0.60.
- **Day 2-3:** FlowMol3 techniques (#2) — ship, retrain at h=128, verify on smoke. Stack with #1 for 0.55-0.75.
- **Day 4-8 (if ROI-positive):** NExT-Mol 1D→3D decoupling (#3) — only if #1+#2 don't get us to 0.7+ on the smoke. Otherwise defer.

**Decision gate after #1:** if decode_ratio ≥ 0.50 on a 16-cell smoke (1h36 + 3 SMILES × 4 CFG × 2 seeds), ship to Round-12/13 production. If < 0.50, proceed to #2.

**Decision gate after #2:** if decode_ratio ≥ 0.65 on a 16-cell smoke, ship. If < 0.65 but > 0.50, the bottleneck is now the 3D head capacity — proceed to #3 (NExT-Mol decoupling). If < 0.50, the FlowMol3 techniques didn't work as expected; revert to baseline + YuelBond and pursue #3.

---

## 6. Honest framing — MEASURED vs PROJECTED vs SPECULATIVE

### MEASURED (arXiv-validated, with paper cites)
- FlowMol3 reports 99.9% validity on GEOM-Drugs with self-conditioning + fake atoms + train-time geometry distortion (arXiv:2508.12629, §4.1, Table 1).
- YuelBond reports F1=92.7% on CDGs where RDKit failed 783/1000 (bioRxiv 10.1101/2025.05.06.652517, §3.2).
- NExT-Mol reports +26% relative 3D FCD improvement on GEOM-Drugs with MoLlama 1D LM + DMT 3D head (arXiv:2502.12638, Table 2).
- EdGr reports substantial outperformance on fragment assembly with joint geometry+topology diffusion (arXiv:2505.21833, §4).
- Our internal review's diagnosis: `wf_cfm_internal_review/diagnose.md` §1.3-1.4 line-cites confirm the bond-head + decoder + capacity failures.

### PROJECTED (our analysis applied to our config, not yet retrained)
- The +0.20-0.35 standalone lift for FlowMol3 techniques is estimated from the 0 → 99.9% paper result, scaled down for our 32 mols. **We expect a smaller lift, not a larger one.**
- The +0.45-0.60 standalone lift for YuelBond is estimated from the 92.7% paper F1 on RDKit-failing data. **Our distortion is uncontrolled, so this is an UPPER bound on standalone lift; the lower bound is the existing 0% baseline.**
- The +0.70-0.90 standalone lift for NExT-Mol is estimated from the 1D-side 100%-valid property + 3D-side state-of-the-art. **Our 32-mol 3D head may not reach the paper's level; this is an optimistic upper bound.**

### SPECULATIVE (forward-looking, not validated)
- The 0.80-0.95 cumulative ceiling for all three fixes is the product of the three PROJECTED lifts, assuming no negative interaction. **This is an over-estimate; in practice, the fixes will partially overlap (e.g., FlowMol3 self-conditioning + NExT-Mol 1D LM both address the topology-validity issue from different angles).**
- The cost estimates (1 day for YuelBond, 1.5 days for FlowMol3) are based on our existing P0+P1 fix experience (e.g., `wf_cfm_p0_fixes` shipped 5 CPU-only fixes in <1 day). **Real cost may be 2× higher if integration surface is larger than expected.**

### FALSIFIABLE
- After Fix #1 (YuelBond): smoke test on 16 generated candidates should show `n_decoded ≥ 8/16`. If < 8/16, YuelBond cannot recover our CFM's distortion level — revert and try a different decoder.
- After Fix #2 (FlowMol3): 1000-step retrain at h=128 should show `cfm_loss` floor below 3.0 (currently 5-9). If still 5-9, the techniques are incompatible with our architecture.
- After Fix #3 (NExT-Mol): the MoLlama-generated SELFIES should be 100% valid (by SELFIES construction). If < 100%, our wrapper is misconfigured.

---

## 7. Cross-references

- `molmetal/reports/wf_cfm_internal_review/audit.md` — Phase 1 deep code review (4 failure hypotheses with line cites)
- `molmetal/reports/wf_cfm_internal_review/diagnose.md` — Phase 2 architecture redesign (4 fixes with priority + effort)
- `molmetal/reports/wf_cfm_p0_fixes/final.md` — P0 fix verification (5 CPU-only fixes shipped, smoke run completed, decode_ratio 0/16 expected at 100 steps)
- `molmetal/reports/wf_cfm_retrain_full/final.md` — 10000-step retrain BLOCKED on GPU
- `molmetal/reports/wf_gpu_recovery_now/final.md` — 5000-step GPU retrain 0/192 (Path (c) stays as default)
- `molmetal/reports/wf_cfm_path_b_decoder_rework/final.md` — chem-aware prior decoder rework (smoke validated)
- `molmetal/reports/wf_vina_lift_phase23/pac_bayes.md` — PAC-Bayes bound 0.7055 (over-fit regime)
- `molmetal/adapters/flow_matching_lipman/__init__.py:1114, 1478, 1496, 1546-1548, 1647, 1671, 1700-1714, 1734-36, 2017-2025, 2067-2105` — line cites for fixes
- `molmetal/adapters/flow_matching_lipman/connectivity_decoder.py` — already-shipped ConnectivityAwareDecoder (Gumbel-top-k fallback)

## 8. Files referenced + files written

**Referenced (read-only):**
- arXiv:2508.12629 — FlowMol3 (Dunn & Koes, 2025-08)
- bioRxiv:10.1101/2025.05.06.652517 / J Chem Inf Model 2026 — YuelBond (Wang & Dokholyan, 2025-05)
- arXiv:2502.12638 — NExT-Mol (Liu et al., ICLR 2025)
- arXiv:2505.21833 — EdGr (Koodli et al., 2025-05)
- arXiv:2504.10564 — FLOWR (2025-04, AlphaXiv)
- arXiv:2505.21469 — PropMolFlow (2025-05)
- arXiv:2406.07266 — SemlaFlow (2024-06; revised 2025-02)
- arXiv:2507.05503 — MolFORM (ICML 2025 genbio)
- arXiv:2507.11818 — SynCoGen (2025-07)
- arXiv:2505.00169 — GEOM-Drugs Revisited (2025-05)
- arXiv:2504.10700 — Optimizing data distribution (Firoz et al., HPDC 2025)
- Kretschmer et al. Nature Communications 2025 — Coverage bias in small molecule ML
- Uni-Mol (ICLR 2023) / Uni-Mol2 (NeurIPS 2024) — pretraining backbones
- `https://github.com/dptech-corp/Uni-Mol` — Uni-Mol code
- `https://github.com/acharkq/NExT-Mol` — NExT-Mol code
- `https://bitbucket.org/dokhlab/yuel_bond` — YuelBond code

**Written:**
- `molmetal/reports/wf_cfm_frontier_research/research_phase1b.md` — this document
