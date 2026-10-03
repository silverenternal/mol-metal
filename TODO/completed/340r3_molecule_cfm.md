# Wf-340-R3: Molecule-Specific CFM Acceleration Survey

**Wf ID:** Wf-340-R3
**Phase:** Survey (WebSearch + WebFetch only; READ-only on `paper/*`)
**Date:** 2026-09-29
**Author:** Mol-Metal R3 agent (Ultracode sub-task per user request)
**Tag discipline:** MEASURED (papers + GitHub URLs) / PROJECTED (architecture-fit scoring)

---

## §0 Context (user request, in their own words)

> "你看一下模型的训练框架能不能再提升，CFM那块，我们模型的本质还是flow matching，
> 应该是能找到那种快速收敛训练的论文和对应的开源代码的，额外启动一个ultracode去做，
> 调研完了得做对应的改进"

Translation: "See if the training framework can be improved further, especially the CFM part.
Our model is fundamentally flow matching, so there should be papers on fast-convergence training
and the corresponding open-source code. Spin up a separate ultracode to do the survey, then make
the corresponding improvements after the survey is done."

**Task scope (from workflow script):**
1. WebSearch molecule-specific CFM works (FlowMM / Equinv / GeoDiff / TargetDiff / DecompDiff /
   MolDiff / DiffDock / CDVAE and adjacent).
2. WebFetch key papers for full metadata.
3. Extract: title + authors + year + architecture + training-acceleration tricks + code URL +
   applicability to Mol-Metal (D-MPNN + Wasserstein + EGNN/3D).
4. Output: `TODO/completed/340r3_molecule_cfm.md` (this file).
5. Honest framing per `party-twenty-spirit` §3.5: MEASURED for what is in the papers, PROJECTED
   for architecture-fit scoring.

---

## §1 Honest framing header

- **MEASURED:** 10 molecule-specific CFM / diffusion papers (this survey); GitHub URLs
  verified via WebSearch result snippets (full WebFetch verification on arXiv was blocked by
  network policy; abstracts read via WebSearch snippets + emergentmind + project pages).
- **PROJECTED:** Architecture-fit scoring in §4 is **subjective ranking**, not benchmarked on
  Mol-Metal data. Each "PROJECTED" score reflects the closest architectural match to Mol-Metal's
  D-MPNN + YuelBond + Wasserstein setup, not measured convergence lift.

---

## §2 Survey table (n=10 molecule-specific CFM / diffusion papers)

| # | Paper | Year | Venue | Arch | Code |
|---|-------|------|-------|------|------|
| 1 | FlowMol (Mixed Continuous + Categorical FM, 3D de novo) | 2024-04 | arXiv:2404.19739 | EGNN-based + categorical SimplexFlow | github.com/dunni3/FlowMol |
| 2 | FlowMol-CTMC (Discrete FM for 3D molecules) | 2024-11 | arXiv:2411.16644 | EGNN-based + discrete FM | (FlowMol repo, extended) |
| 3 | FlowMM (Riemannian FM for crystalline materials) | 2024-06 | ICML 2024 | EGNN + Riemannian flow on lattices (manifold from CDVAE/DiffCSP) | github.com/facebookresearch/flowmm |
| 4 | GeoDiff (Geometric Diffusion for conformers) | 2022-03 | ICLR 2022 | EGNN + equivariant Markov kernels | github.com/MinkaiXu/GeoDiff |
| 5 | GEOLDM (Geometric Latent Diffusion for 3D) | 2023-05 | ICML 2023 | EGNN encoder + latent diffusion | github.com/MinkaiXu/GeoLDM |
| 6 | Equivariant Flow Matching (atomistic) | 2023-06 | NeurIPS 2023 (arXiv:2306.07573) | EGNN / Equivariant transformer + flow matching | (paper repo; project page equinn.github.io) |
| 7 | EqFM (EqFlow, equivariant flow + force-field correction) | 2024 | (Rice + TUM) | EGNN + denoising task + force correction | (Per joint training paper) |
| 8 | GOAT (Jointly Geometric OT for 3D mol) | 2024-05 | ICLR 2025 (arXiv:2405.15252) | EGNN + equivariant latent space + OT coupling purification (~2× sampling speedup vs EquiFM) | github.com/WanyuGroup/ICLR2025-GOAT |
| 9 | SO(3)-Averaged FM + Reflow (Cao et al., NVIDIA) | 2025-07 | ICML 2025 (arXiv:2507.09785) | EGNN + SO(3)-averaged flow objective + Reflow/distillation → 1-step inference | (NVIDIA; verify before chase) |
| 10 | DiffDock (Diffusion for molecular docking pose) | 2022-10 | ICLR 2023 (arXiv:2210.01776) | SE(3)-equivariant + diffusion (not flow matching, but adjacent SBDD refiner) | github.com/gcorso/DiffDock |
| 11 | ConfDiff (Latent diffusion for conformations) | 2025-05 | arXiv:2505.07587 | Latent diffusion (drift) + equivariant backbone | github.com/anchen1011/ConfDiff |
| 12 | MolDiff (Diffusion for conformations) | 2023 | ICML 2023 | EGNN + bond/angle auxiliary supervision + diffusion bridge | (openreview.net/forum?id=...) |
| 13 | CDVAE (Crystal Diffusion VAE) | 2022 | ICLR 2022 | VAE + score-based diffusion in latent space | (referenced inside FlowMM as sub-module) |

(N=13 if you include the related diffusion-only works; n=10 if you count only flow-matching +
equivariant molecule-generation papers.)

---

## §3 Per-paper summary (architecture + training-acceleration tricks)

### 3.1 FlowMol (Dunn & Koes, 2024) — closest architectural neighbour

- **arXiv:** 2404.19739 (Apr 2024)
- **Architecture:** EGNN-style backbone for 3D coordinates + **SimplexFlow** for categorical
  atom types. The model is split into continuous FM (over Cartesian atom coords + atom type
  simplices). Both coordinates and atom types are generated end-to-end.
- **Training acceleration tricks:**
  - **Mixed continuous + categorical FM** — the categorical side uses **SimplexFlow** (a
    simplified flow on the atom-type simplex), which lets the model avoid expensive discrete
    diffusion samplers (cf. DiffDock's discrete-loss treatment).
  - **10× inference speedup vs diffusion** (per FlowMol paper, "more than a tenfold" inference
    speedup vs diffusion-based methods). Inference uses standard ODE integration over the FM
    path — much shorter than 1000-step DDPM-style sampling.
  - **Flexible prior** — uniform or Gaussian on the source; no second hard sampler is needed.
- **Code:** github.com/dunni3/FlowMol (MIT-style; check licence before reuse).
- **MEASURED applicability:** directly applicable to Mol-Metal's 3D structure + atom-type
  generation. Mol-Metal's 3D refiner (YuelBond + bond head + EGNN/YuelBlock) is closer to
  FlowMol's continuous branch than to GeoDiff's pure-conformer diffusion. (PROJECTED)
- **PROJECTED lift estimate:** if Mol-Metal adopts SimplexFlow-style handling for atom-type
  categorical noise (vs the current one-hot CE), the bond-head training signal may become
  more stable; expected lift on decode_ratio + bond-head accuracy is **+0.05–0.10** (PROJECTED).

### 3.2 FlowMol-CTMC (Dunn & Koes, Nov 2024) — discrete FM extension

- **arXiv:** 2411.16644
- **Architecture:** EGNN + **continuous-time Markov chain (CTMC) flow** on atom-type simplices.
  This is the discrete-flow analogue of FM (Lipman et al.).
- **Training acceleration tricks:**
  - **CTMC rate network** replaces SimplexFlow with a continuous-time Markov chain — keeps
    the "convergent" property of FM while handling discrete variables properly.
  - Direct multi-modal coupling between continuous coordinates and categorical types.
- **Code:** extension of FlowMol repo (github.com/dunni3/FlowMol).
- **PROJECTED applicability:** relevant if Mol-Metal wants to add a fully-discrete category
  flow alongside the existing continuous FM. (PROJECTED, no immediate plan.)

### 3.3 FlowMM (Miller et al., Meta FAIR, ICML 2024) — closest Riemannian-FM neighbour

- **arXiv:** 2406.04713
- **Architecture:** Riemannian Flow Matching over **atom-type simplex + periodic-lattice
  parameters** (rotation, permutation, PBC). The manifold-aware flow handles crystal
  symmetries; submodule geometry is borrowed from CDVAE and DiffCSP. EGNN-style backbone.
- **Training acceleration tricks:**
  - **Manifold-aware flow base distributions** (different choices: `abits`+`nonsym`,
    `abits`+`params`, `simplex`+`params_normal_base`). Each combo produces a different
    manifold geometry; ablation in paper.
  - **~3× more efficient in finding stable materials** vs prior open methods (per paper
    abstract). This is a finding-efficiency metric, not wall-clock.
  - **Reflow + distillation** support for one-step generation.
- **Code:** github.com/facebookresearch/flowmm (archived Sep 18 2025; use `bkmi/flowmm`
  mirror which is the canonical author repo).
- **PROJECTED applicability:** FlowMM is for **periodic crystals**, not molecules. The
  technique (Riemannian FM on atom-type manifold + PBC) does NOT transfer to Mol-Metal's
  apo-pocket molecular setting. However, the **ablation-of-base-distribution methodology**
  (which choice of base manifold converges fastest) is directly applicable to Mol-Metal's
  flow base distribution choice (currently uniform-Gaussian hybrid for coords + one-hot CE
  for atom types). (PROJECTED)

### 3.4 GeoDiff (Xu et al., ICLR 2022) — foundational 3D-mol diffusion

- **arXiv:** 2203.02923
- **Architecture:** EGNN + equivariant Markov kernels; the model is a **conformer diffusion**
  (predicts 3D conformers given a fixed 2D molecular graph).
- **Training acceleration tricks:**
  - **Equivariant Markov kernel** — by construction, roto-translationally invariant. This
    removes the need for explicit data augmentation (rotating every conformer in mini-batch).
  - **Weighted variational lower bound** on the conditional likelihood; smooth loss surface.
- **Code:** github.com/MinkaiXu/GeoDiff (Apache-2.0; widely re-used as basis for TargetDiff, etc.).
- **PROJECTED applicability:** GeoDiff is **conformer-only** (fixed 2D graph), not
  de novo. But the **equivariant-Markov-kernel design** is a relevant architectural insight
  for any flow-matching model that needs SE(3) invariance — Mol-Metal's apo-pocket coord
  flow currently uses centering + random rotation as data augmentation; equivariant kernel
  could remove the need for augmentation and produce faster convergence. (PROJECTED)

### 3.5 GEOLDM (Xu et al., ICML 2023) — latent FM/diffusion

- **arXiv:** 2305.01140
- **Architecture:** **EGNN-based geometric autoencoder** (compresses 3D molecule to
  point-structured latent `(z_h, z_x)`) + **diffusion model in latent space**. The latent
  diffusion is the inner training loop.
- **Training acceleration tricks:**
  - **Latent space smoothing** — diffusion operates in a smoother, lower-dimensional latent,
    which is empirically faster to converge than operating directly on atom coordinates.
  - **Early-stopping regularization (ES-reg)** instead of KL regularization in VAE — ES-reg
    outperforms KL on downstream generation quality. Specifically: stop the autoencoder
    training before it overfits the reconstruction.
  - **Point-structured latent** — `(z_h, z_x)` separates invariant (chemical) from
    equivariant (geometric) features, allowing the diffusion model to learn them with
    different objectives.
- **Code:** github.com/MinkaiXu/GeoLDM
- **MEASURED results (from paper):** 99.3% validity on GEOM-DRUG; 89.4% molecule stability
  on QM9 (vs prior 84.6%); up to 7% validity lift on large biomolecules.
- **PROJECTED applicability:** ES-reg is the most directly portable trick. Mol-Metal's
  refiner encoder (YuelBlock) could be ES-reg'd to avoid overfitting of the bond head.
  Latent-space FM is NOT directly applicable to Mol-Metal — Mol-Metal operates on real
  3D coords + a discrete bond head, not on a learned latent. (PROJECTED)

### 3.6 Equivariant Flow Matching (Köhler et al., NeurIPS 2023) — foundational equivariant FM

- **arXiv:** 2306.07573
- **Architecture:** EGNN + flow matching with **equivariant construction** — the velocity
  field is rotation/translation equivariant by design.
- **Training acceleration tricks:**
  - **Equivariant flow construction** (no augmentation needed at training time) — converges
    faster than non-equivariant FM + augmentation.
  - **Direct FM on SE(3)-invariant** target — no need for roto-translation alignment tricks.
- **Code:** paper repo (check arXiv page); project page: equinn.github.io/equivariant_flow_matching
- **PROJECTED applicability:** Mol-Metal's apo-pocket refiner could adopt **equivariant
  velocity net construction** to remove the random-rotation augmentation step in training.
  This is a **direct code-borrow opportunity** since YuelBlock's local-frame message
  passing is already equivariant — wiring an explicit equivariant FM head should be a
  1-2 week code change). (PROJECTED)

### 3.7 EqFM (EqFlow, Plainer et al., Rice + TUM, 2024) — equivariant FM with predictive head

- **Architecture:** EGNN + **joint generative + predictive** training: equivariant flow
  generates per-atom forces given molecular structure, AND a denoising task runs in parallel.
- **Training acceleration tricks:**
  - **Joint denoising + force prediction** — multi-task learning; the denoising task acts
    as an auxiliary loss for the flow model.
  - **Single-function-evaluation sampling** — generates trajectories in one function eval
    (vs many-step ODE).
  - **Sample efficiency improvement** via the predictive correction.
- **Code:** (verify before chase; per paper).
- **PROJECTED applicability:** Mol-Metal's YuelBond + bond-head already uses an auxiliary
  classifier (bond-existence head). EqFM's pattern of **joint generative + denoising
  auxiliary task** could be ported as a multi-task loss (current Wasserstein regression +
  bond classifier + denoising MSE on patch-derived objects). (PROJECTED; MEASURED
  comparison would be in Wf-340 implementation wave.)

### 3.8 GOAT (Hong, Lin, Tan, ICLR 2025) — geometric OT for 2× sampling speedup

- **arXiv:** 2405.15252
- **Architecture:** EGNN + **equivariant latent space + OT coupling estimation + coupling
  purification**.
- **Training acceleration tricks:**
  - **Geometric Transport Formula** — jointly measures cost on continuous coords +
    categorical atom types in an equivariant latent space.
  - **Optimal Coupling Estimation + Purification** — converts arbitrary distribution
    couplings into deterministic ones, with theoretical non-increasing transport cost.
  - **2× sampling speedup** vs sub-optimal FM (EquiFM) — 90 sampling steps vs 200 on QM9.
  - **3× improvement in S-Time** (0.12s vs 0.37s on QM9) — single-sample wall-time.
  - **~89.65% reduction** in geometric transport cost.
- **MEASURED results:** QM9 92.9% validity / 78.6% novelty / 72.3% significance;
  GEOM-DRUG 96.2% validity / 99.9% uniqueness / 0.94s S-Time (vs GeoLDM 12.84s).
- **Code:** github.com/WanyuGroup/ICLR2025-GOAT
- **PROJECTED applicability:** GOAT's **coupling-purification mechanism** is the most
  novel training-time trick in the survey. It estimates an OT plan between noise and data,
  then prunes sub-optimal pairings — equivalent to a **curriculum** that starts with the
  easiest training pairs. Mol-Metal's current training randomly pairs (pocket, ligand) and
  runs Wasserstein regression. **If GOAT-style coupling purification is added to the
  Wasserstein pairing step, expected decoding-sample speedup: ~2×; expected
  Wasserstein-loss stability: +0.05-0.10.** (PROJECTED)

### 3.9 SO(3)-Averaged FM + Reflow (Cao et al., NVIDIA, ICML 2025)

- **arXiv:** 2507.09785
- **Architecture:** EGNN-based 3D conformer generator + **SO(3)-averaged flow objective**.
- **Training acceleration tricks:**
  - **SO(3)-averaged flow objective** — averages the loss over the SO(3) rotation group
    rather than relying on Kabsch alignment or conditional OT. **Faster training
    convergence** vs conditional-OT flow or Kabsch-aligned flow.
  - **Reflow** — straightens trajectories, enabling few-step sampling.
  - **Distillation** — compresses multi-step flow into a single-step generator.
- **MEASURED results:** SOTA conformer generation quality at ICML 2025; few-step/one-step
  inference available.
- **Code:** (NVIDIA; verify before chase.)
- **PROJECTED applicability:** **SO(3)-averaging is the single most directly portable
  trick** in this survey. Mol-Metal's apo-pocket flow currently uses Kabsch-aligned
  coordinates as the FM target. Replacing Kabsch alignment with **SO(3)-averaged loss**
  (which integrates out rotation in the loss itself) should give faster training convergence
  and more stable Wasserstein-gradient signal. (PROJECTED)

### 3.10 DiffDock (Corso et al., ICLR 2023) — adjacent docking diffusion

- **arXiv:** 2210.01776 (Oct 2022)
- **Architecture:** SE(3)-equivariant diffusion over ligand pose (rotation, translation,
  torsion angles).
- **Training acceleration tricks:**
  - **Confidence model** — a separately-trained model ranks generated poses by predicted
    RMSD to ground truth. This is an inference-time trick (sample-best-of-N), not a
    training-acceleration trick. But it gives SOTA benchmark numbers without retraining.
  - **Diffusion over SE(3) —** the random-torsion + random-translation is a useful prior.
- **Code:** github.com/gcorso/DiffDock (34k+ stars; actively maintained).
- **PROJECTED applicability:** DiffDock is **diffusion, not flow matching**, so the
  training-tricks transfer is limited. The **confidence model** pattern (separate model
  ranks generations) is directly applicable to Mol-Metal's Vina-dock refinement — a
  lightweight "decoder-quality" model could rank generated binders by predicted Vina
  score without rerunning docking. (PROJECTED)

### 3.11 ConfDiff (arXiv:2505.07587, 2025)

- **Architecture:** Latent diffusion for molecular conformations.
- **Training acceleration tricks:** latent-space drift.
- **Code:** github.com/anchen1011/ConfDiff
- **PROJECTED applicability:** same as GEOLDM — latent-space idea is not directly
  applicable to Mol-Metal's real-coord refiner. (PROJECTED)

### 3.12 MolDiff (Peng et al., ICML 2023)

- **Architecture:** EGNN + auxiliary bond/angle supervision + diffusion bridge.
- **Training acceleration tricks:** auxiliary geometric losses (bond length, bond angle)
  stabilise the diffusion training.
- **PROJECTED applicability:** **Mol-Metal's YuelBond bond head already provides
  auxiliary bond supervision.** The pattern of **auxiliary geometric supervision on
  bonds + angles + dihedrals** is already in place. (PROJECTED, no new code needed.)

### 3.13 CDVAE (Xie et al., ICLR 2022)

- **Architecture:** VAE + score-based diffusion in latent space, for crystal inverse design.
- **PROJECTED applicability:** as §3.3 — crystals, not molecules. Sub-module of FlowMM.

---

## §4 Architecture-fit scoring (PROJECTED)

The Mol-Metal 3D refiner is built around:
- **YuelBond + bond head** (discrete bond prediction),
- **YuelBlock** (D-MPNN-style local-frame message passing on apo-pocket),
- **Wasserstein regression** (pIC50-style metric regression),
- **Conditional FM** (Lipman-style FM with pocket conditioning).

Architectural fit, ranked by how directly each paper's tricks apply:

| Rank | Paper | Fit score (PROJECTED) | Why |
|------|-------|----------------------|-----|
| 1 | **FlowMol** | 9/10 | Same task (3D de novo mol), mixed continuous+categorical FM, EGNN-style backbone |
| 2 | **SO(3)-Averaged FM (Cao 2025)** | 9/10 | Trick is directly portable: SO(3)-averaged loss replaces Kabsch-aligned loss |
| 3 | **Equivariant Flow Matching (Köhler 2023)** | 8/10 | EGNN + equivariant velocity construction is direct code-borrow |
| 4 | **GOAT** | 8/10 | OT coupling purification → curriculum on (pocket, ligand) pairs |
| 5 | **EqFM** | 7/10 | Joint denoising + force prediction; multi-task auxiliary loss pattern |
| 6 | **GEOLDM** | 5/10 | Latent-space idea not directly applicable; ES-reg trick is portable |
| 7 | **GeoDiff** | 5/10 | Conformer-only; equivariant kernel design is portable but additive |
| 8 | **FlowMol-CTMC** | 5/10 | Discrete-FM only; not immediate priority |
| 9 | **MolDiff** | 4/10 | Auxiliary geometric supervision already in place via YuelBond |
| 10 | **ConfDiff** | 3/10 | Latent-space drift, not directly applicable |
| 11 | **FlowMM** | 2/10 | Crystals, not molecules |
| 12 | **CDVAE** | 2/10 | Crystals, not molecules |
| 13 | **DiffDock** | 3/10 | Adjacent SBDD refiner; confidence-model pattern portable |

**PROJECTED winner combo for the Wf-340 implementation wave (per user "调研完了得做对应的改进"):

- **Primary:** SO(3)-Averaged FM (replaces Kabsch alignment in current loss).
- **Secondary:** Equivariant flow velocity construction (removes rotation augmentation).
- **Third:** GOAT-style OT coupling for Wasserstein (pocket, ligand) pair selection.
- **Quaternary:** FlowMol's SimplexFlow for atom-type categorical noise (orthogonal).
- **Background:** ES-reg on YuelBlock autoencoder (per GEOLDM trick).

This 5-trick combo targets a PROJECTED **+30% wall-clock reduction** on training the CFM
refiner end-to-end, plus PROJECTED **+5–10% lift on Wasserstein loss stability**. (PROJECTED.)

---

## §5 Code-availability table (verify-before-chase list)

| Paper | GitHub URL | License | Status (2026-09-29) |
|-------|-----------|---------|---------------------|
| FlowMol | github.com/dunni3/FlowMol | (check) | active |
| FlowMM | github.com/facebookresearch/flowmm (mirror: github.com/bkmi/flowmm) | MIT | archived Sep 18 2025 (mirror still active) |
| GeoDiff | github.com/MinkaiXu/GeoDiff | Apache-2.0 | active |
| GEOLDM | github.com/MinkaiXu/GeoLDM | (check) | active |
| Equivariant Flow Matching | (arXiv page links) | (check) | active |
| EqFM | (arXiv page links) | (check) | verify before chase |
| GOAT | github.com/WanyuGroup/ICLR2025-GOAT | (check) | active (ICLR 2025) |
| SO(3)-Averaged FM | (NVIDIA; verify) | (likely internal) | verify before chase |
| DiffDock | github.com/gcorso/DiffDock | MIT | active (34k+ stars) |
| ConfDiff | github.com/anchen1011/ConfDiff | (check) | active |
| MolDiff | (openreview.net/forum?id=...) | (check) | verify before chase |
| CDVAE | (sub-module of FlowMM) | MIT | active |

---

## §6 Open questions / next-step TODOs

1. **Wf-340-A1:** Wire SO(3)-averaged FM loss into Mol-Metal's refiner. Replace current
   Kabsch-aligned target with the SO(3)-averaged loss. PROJECTED wall-clock saving:
   30% (sample budget). MEASURED comparison would be in Wf-340 implementation wave.
2. **Wf-340-A2:** Wire equivariant velocity net construction (no augmentation). PROJECTED
   10-20% wall-clock saving (data-loader faster).
4. **Wf-340-A3:** Wire OT coupling purification on (pocket, ligand) pair selection for
   Wasserstein regression. PROJECTED +5-10% Wasserstein loss stability.
5. **Wf-340-A4 (background):** ES-reg on YuelBlock autoencoder (per GEOLDM §3.4).
   PROJECTED +0.05 bond-head accuracy.
7. **Wf-340-A5 (low-priority):** DiffDock-style decoder-quality predictor for generated binders.
   Useful for Mol-Metal's Vina-dock refinement pipeline (not directly for CFM training).

---

## §7 Honest-negative items

- **DiffDock is diffusion, not flow matching** — its tricks transfer only at the
  inference-time confidence-model level, not at the training-acceleration level. (HONEST
  note per `party-twenty-spirit` §3.5.)
- **CDVAE + FlowMM are for crystals**, not molecules. Borrowed-tricks are limited to
  manifold-base-distribution ablation methodology. (HONEST.)
- **arXiv WebFetch was blocked** by network policy in this run. Paper abstracts were read
  via WebSearch snippets + emergentmind summaries + project pages. Full-paper claims should
  be re-verified by reading PDFs before implementation. (HONEST.)
- **All architecture-fit scores are PROJECTED**, not benchmarked on Mol-Metal data. The
  30% wall-clock saving claim is a target, not a measured result. (HONEST.)

---

## §8 Source list (WebSearch + WebFetch result URLs)

- FlowMol: emergentmind.com/papers/2404.19739 + github.com/dunni3/FlowMol
- FlowMM: emergentmind.com/papers/2406.04713 + github.com/facebookresearch/flowmm
- GeoDiff: emergentmind.com/papers/2203.02923 + github.com/MinkaiXu/GeoDiff
- GEOLDM: emergentmind.com/papers/2305.01140 + github.com/MinkaiXu/GeoLDM
- Equivariant Flow Matching: arXiv:2306.07573 + equinn.github.io
- EqFM: arXiv via WebSearch snippet (Plainer, Wang, Günnemann — Rice + TUM)
- GOAT: arXiv:2405.15252 + github.com/WanyuGroup/ICLR2025-GOAT + ICLR 2025
- SO(3)-Averaged FM: arXiv:2507.09785 (Cao, Geiger, ..., Kucukbenli — NVIDIA + MIT) + ICML 2025
- DiffDock: arXiv:2210.01776 + github.com/gcorso/DiffDock + ICLR 2023
- ConfDiff: arXiv:2505.07587 + github.com/anchen1011/ConfDiff
- MolDiff: openreview.net ICML 2023 entry (Peng et al.)
- CDVAE: ICLR 2022 (Xie, Fu, Ganea, Barzilay, Jaakkola — MIT)

---

**END OF Wf-340-R3 SURVEY** — Owner Mol-Metal R3 agent; cross-ref `party-twenty-spirit` §3.5.
Last updated 2026-09-29. Survey is READ-only deliverable; **implementation wave is Wf-340-A1..A5
per §6**, gated on user OK per AGENTS.md §5.1 (no GPU launch without pre-flight PASS).
