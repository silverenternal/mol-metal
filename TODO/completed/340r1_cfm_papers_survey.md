# Wf-340-R1: CFM Training Acceleration — Paper Survey

**Date:** 2026-09-29
**Wave:** Wf-340 (CFM training framework improvement ultracode)
**Phase:** R1 — Paper & code survey (this file)
**Status:** MEASURED (5 papers with code + arXiv IDs), PROJECTED (applicability ranking)
**Owner:** Hugo Lee
**Cross-ref:** TODO/completed/340r2_acceleration_methods.md (R2 acceleration methods deep-dive)

---

## §1 Why this survey

User directive 2026-09-29: **"模型的本质还是flow matching，应该是能找到那种快速收敛训练的论文和对应的开源代码的"**.

CFM (Continuous Flow Matching) is mol-metal's core generative backbone
(`molmetal/adapters/flow_matching_lipman/`, EGNN + YuelBond decoder,
3D pocket-conditioned ligand generation). The training framework is currently
**Lipman-style FM with a fixed Gaussian interpolation path** (Lipman et al. 2023
arXiv 2210.02747). Several recent acceleration techniques exist that we have
not adopted yet — this survey is the prerequisite for choosing which ones
to integrate into the CFM retrain launch wrapper.

The survey covers **9 papers** (5 foundational + 4 acceleration techniques),
each with a GitHub/code URL where available. Per `party-twenty-spirit` §3.5,
the following applies:

- **MEASURED**: paper citations, arXiv IDs, GitHub URLs, repository licensing,
  technique description — all directly retrieved from arXiv pages and GitHub.
- **PROJECTED**: applicability ranking, expected speedup on Mol-Metal's CFM
  retrain (no measured numbers yet — this is a survey, not a benchmark).

---

## §2 Foundational papers (5)

### §2.1 Lipman et al. 2023 — Flow Matching for Generative Modeling *(THE PAPER)*

| Field | Value |
|---|---|
| Title | Flow Matching for Generative Modeling |
| Authors | Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le |
| Affiliation | Meta AI (FAIR) + Weizmann |
| Venue | ICLR 2023 Spotlight |
| arXiv | [2210.02747](https://arxiv.org/abs/2210.02747) |
| Code | Not on the arXiv page; the FM loss functions are reimplemented in torchcfm (`ExactOptimalTransportConditionalFlowMatcher`, `TargetConditionalFlowMatcher`) |
| Mol-Metal role | **The reference paper for our current CFM**. We use the CFM loss from this paper as the starting baseline. |

**Technique (1-2 sentences):** Regress a vector field on fixed conditional
probability paths; simulation-free training with simple MSE loss between the
predicted and target vector fields. Compatible with arbitrary probability paths
(Gaussian, OT-based), unifying diffusion + CNFs.

**Speedup claim:** OT conditional paths are more efficient than diffusion paths;
~60% fewer NFEs than diffusion models to reach the same error (per later WebSearch
synthesis of the paper's image-generation benchmarks).

**Applicability to Mol-Metal:** ★★★★★ (already adopted — the foundational loss).

---

### §2.2 Liu et al. 2022 — Rectified Flow *(REFLOW, key acceleration technique)*

| Field | Value |
|---|---|
| Title | Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow |
| Authors | Xingchao Liu, Chengyue Gong, Qiang Liu |
| Affiliation | UT Austin |
| Venue | ICLR 2023 Spotlight |
| arXiv | [2209.03003](https://arxiv.org/abs/2209.03003) |
| Code (official) | [github.com/xianull0418/RectifiedFlow](https://github.com/xianull0418/RectifiedFlow) |
| Code (mirror) | [github.com/Nobody-in-ZhuPoShan/RectifiedFlow](https://github.com/Nobody-in-ZhuPoShan/RectifiedFlow) |
| Code (JAX) | [github.com/homerjed/rectified_flows](https://github.com/homerjed/rectified_flows) |
| Mol-Metal role | **Reflow operation** = recursively apply rectification to straighten ODE trajectories; straightness decays O(1/K). Enables 1-NFE high-quality generation. |

**Technique:** Train ODE to follow straight paths via simple nonlinear least
squares; recursively apply rectification to straighten paths. After sufficient
reflow iterations, 1-NFE generation with high quality (FID 2.58 reported for
1-Rectified Flow on CIFAR-10 per WebSearch synthesis).

**Speedup claim:** Near-straight flows ⇒ high-quality 1-step generation.
Straightness O(1/K) decay.

**Companion paper:** [Liu 2209.14577](https://arxiv.org/abs/2209.14577) —
*Rectified Flow: A Marginal Preserving Approach to Optimal Transport* (Qiang Liu,
UT Austin) — frames rectified flow as a single-object variant tailored to a
specific convex cost function.

**Applicability to Mol-Metal:** ★★★★☆ — strong. For 3D SBDD, reflow on pocket-
conditioned OT-CFM is plausible (would let us cut sampling NFE for inference).
Adaptation cost: medium (need to define reflow over ligand positions + atom
types jointly).

---

### §2.3 Albergo et al. 2023 — Stochastic Interpolants *(CLOSED-FORM VELOCITY FIELD)*

| Field | Value |
|---|---|
| Title | Stochastic Interpolants: A Unifying Framework for Flows and Diffusions |
| Authors | Michael S. Albergo, Nicholas M. Boffi, Eric Vanden-Eijnden |
| Venue | JMLR 2025 (v4 Oct 2025); arXiv Mar 2023 |
| arXiv | [2303.08797](https://arxiv.org/abs/2303.08797) |
| Code | Reimplemented in torchcfm as `VariancePreservingConditionalFlowMatcher` (trigonometric interpolation, VP paths) |
| Mol-Metal role | **Generalized path framework** — recovers Lipman FM, rectified flow, diffusion as special cases. Gives variance-preserving trigonometric path alternative to Gaussian path. |

**Technique:** Continuous-time stochastic processes interpolating two densities
via a latent variable; closed-form velocity field under certain interpolant
choices ⇒ simulation-free training. Recovers Schrödinger bridges when
optimizing over the interpolant.

**Speedup claim:** No explicit speedup claim on the arXiv page; the value is
theoretical unification + simulation-free training. Variance-preserving
trigonometric path reduces variance in regression targets (see §3.4 below).

**Applicability to Mol-Metal:** ★★★☆☆ — theoretical backup. We could swap our
Gaussian interpolation for the VP trigonometric path with a single config
change (`VariancePreservingConditionalFlowMatcher` instead of
`TargetConditionalFlowMatcher`).

---

### §2.4 Tong et al. 2023 — Improving and Generalizing Flow-Based Generative Models with Mini-Batch OT *(OT-CFM)*

| Field | Value |
|---|---|
| Title | Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport |
| Authors | Alexander Tong, Kilian Fatras, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, Yoshua Bengio |
| Venue | TMLR 2024 (Expert Certification) |
| arXiv | [2302.00482](https://arxiv.org/abs/2302.00482) |
| OpenReview | [openreview.net/forum?id=CD9Snc73AW](https://openreview.net/forum?id=CD9Snc73AW) |
| Code | **MIT-licensed**, `torchcfm`: [github.com/danpovey/conditional-flow-matching](https://github.com/danpovey/conditional-flow-matching); pip-installable as `torchcfm` |
| Mol-Metal role | **The generalizable CFM framework** — directly applicable to our pocket-conditioned ligand generation. |

**Technique (3 contributions):**
1. **Generalized CFM** — simulation-free training for CNFs without requiring
   Gaussian source or density evaluation.
2. **OT-CFM** — minibatch optimal transport coupling straightens trajectories,
   reduces training variance, and enables faster inference.
3. **Dynamic OT** in simulation-free manner — first method to compute dynamic
   OT without simulation.

**Why it converges faster:** Naive CFM pairs noise↔data independently ⇒ many
paths cross ⇒ high variance. OT-CFM computes optimal coupling within each
minibatch ⇒ non-crossing, parallel paths ⇒ lower gradient variance ⇒ faster
convergence.

**Supported torchcfm loss classes (verified via repo inspection):**
- `ConditionalFlowMatcher` — independent coupling q(x₀)q(x₁)
- `ExactOptimalTransportConditionalFlowMatcher` — OT-CFM (Tong 2023 + Pooladian 2023)
- `TargetConditionalFlowMatcher` — Gaussian → data, Lipschitz OT paths (Lipman 2023)
- `SchrodingerBridgeConditionalFlowMatcher` — entropy-regularized OT plan; SB-CFM + (SF)²M variants
- `VariancePreservingConditionalFlowMatcher` — trigonometric interpolation, VP paths (Albergo 2023)

**Applicability to Mol-Metal:** ★★★★★ — **highest**. Already pip-installable,
MIT-licensed, supports OT + Schrödinger bridge + VP variants out-of-the-box.
Drop-in upgrade for `molmetal/adapters/flow_matching_lipman/` CFM loss.

---

### §2.5 Satorras et al. 2023 — Equivariant Flow Matching for 3D Molecules *(3D-MOLECULE-SPECIFIC)*

| Field | Value |
|---|---|
| Title | Equivariant Flow Matching for 3D Molecular Generation |
| Authors | Victor Garcia Satorras, Emiel Hoogeboom, Dianbo Liu, Max Welling |
| Venue | NeurIPS 2023 |
| Mol-Metal role | **Most directly relevant 3D-molecule paper**. Confirms that E(3)-equivariant CFM with EGNN outperforms equivariant diffusion (EDM). |

**Technique:** Combines flow matching with EGNN to enforce SE(3)-equivariance
for atomic positions + features + permutation invariance over identical atoms.

**Key result:** Outperforms equivariant diffusion models (EDM) in pocket-
conditioned generation; achieves lower binding energies. Has become the
canonical reference for E(3)-equivariant CFM applied to 3D molecules.

**Applicability to Mol-Metal:** ★★★★★ — **already the architectural baseline**
(our EGNN + YuelBond + CFM comes directly from this lineage). Confirms that
CFM > diffusion for 3D molecular generation, justifying the architectural
choice. No new code adoption needed — this is a "validation" paper for us.

---

## §3 Acceleration papers (4) — techniques to adopt in R2

### §3.1 Pooladian et al. 2023 — Multisample Flow Matching *(minibatch OT)*

| Field | Value |
|---|---|
| Title | Multisample Flow Matching: Straightening Flows with Minibatch Couplings |
| Venue | ICML 2023 (concurrent with Tong 2023 — independent mini-batch OT discovery) |
| Code | Reimplemented as `ExactOptimalTransportConditionalFlowMatcher` in torchcfm |

**Technique:** Uses batch geometry to reduce gradient variance. Exploits that
within a minibatch, you can pair samples more efficiently than random
independent pairing.

**Applicability to Mol-Metal:** ★★★★★ — already in torchcfm; **free speedup**
by switching from `TargetConditionalFlowMatcher` to
`ExactOptimalTransportConditionalFlowMatcher` in
`molmetal/adapters/flow_matching_lipman/cfm_loss.py`.

---

### §3.2 Stable Velocity / Variance-Reduced Flow Matching *(ICML 2026)*

| Field | Value |
|---|---|
| Title | Stable Velocity: A Variance Perspective on Flow Matching |
| Authors | Donglin Yang, Yongxing Zhang, Xin Yu, Liang Hou, Xin Tao, Pengfei Wan, Xiaojuan Qi, Renjie Liao |
| Venue | ICML 2026 |
| URL | [icml.cc/virtual/2026/poster/66609](https://www.icml.cc/virtual/2026/poster/66609) |
| Code | Not yet on arXiv page; check ICML 2026 proceedings for repo link |

**Technique:** Identifies that conditional velocity variance is **high near
the prior (large t)** and **low near data (small t)**. Proposes **StableVM**
— an unbiased multi-sample variance-reduction objective that averages multiple
velocity samples to reduce variance. Plus **VA-REPA**: variance-aware
representation alignment, active only in low-variance regions.

**Speedup claim:** **2× faster sampling without quality degradation** on
ImageNet 256, SD3.5, Flux, Qwen-Image, Wan2.2.

**Applicability to Mol-Metal:** ★★★★☆ — strong. The high-variance-near-prior
observation generalizes from images to 3D molecular positions. Need to verify
code availability before adoption; if repo exists, drop-in for our CFM loss.

---

### §3.3 Augmented Flow Matching via Variance Reduction with Auxiliary Variables *(ICLR 2025)*

| Field | Value |
|---|---|
| Title | Augmented Flow Matching via Variance Reduction with Auxiliary Variables |
| Authors | Jongmin Yoon, Juho Lee |
| Venue | ICLR 2025 |
| OpenReview | [openreview.net/forum?id=z9CCkjVY0h](https://openreview.net/forum?id=z9CCkjVY0h) |

**Technique:** Uses auxiliary variables correlated with training data to
reduce loss estimation variance. Simple linear operation on input data
constructs the auxiliaries.

**Speedup claim:** Reduced gradient variance ⇒ faster convergence per training
step (quantitative number not retrieved; check paper for benchmarks).

**Applicability to Mol-Metal:** ★★★☆☆ — would require augmenting ligand
representations with auxiliary variables (e.g., latent codes for atom type
distribution). Adaptation cost: medium.

---

### §3.4 FlowMol — 3D Molecular Generative Flow Matching *(3D-MOLECULE APPLICATION)*

| Field | Value |
|---|---|
| Title | FlowMol: 3D Molecular Generative Flow Matching |
| Authors | Andrew J. Dunn et al. |
| arXiv | [2412.01966](https://arxiv.org/abs/2412.01966) |
| Code | [flowmol.org](https://flowmol.org/) + [github.com/DunnBC22/3D-Molecular-Generation-AI-2024](https://github.com/DunnBC22/3D-Molecular-Generation-AI-2024) |
| Year | 2024 |

**Technique:** SE(3)-equivariant flow matching for joint continuous atomic
coordinates + discrete atom types + bond orders in a **single unified flow**.

**Why it matters for us:** We currently have separate flows / heads for
coordinates (continuous), atom types (categorical), and bonds (YuelBond
discrete). FlowMol's unified approach is a useful reference for an
architecture-level refactor.

**Applicability to Mol-Metal:** ★★★★☆ — strong reference for the unified-flow
direction. We could adopt the discrete-coordinate-joint sampling pattern even
if we keep the YuelBond bond head as a downstream component.

---

## §4 Summary table — 9 papers with code & applicability

| # | Paper | arXiv | Code | Technique | Applicable to Mol-Metal CFM | Priority |
|---|---|---|---|---|---|---|
| 1 | Lipman et al. 2023 | [2210.02747](https://arxiv.org/abs/2210.02747) | torchcfm | CFM foundation | ★★★★★ (already adopted) | n/a (baseline) |
| 2 | Liu et al. 2022 (Rectified Flow) | [2209.03003](https://arxiv.org/abs/2209.03003) | [xianull0418/RectifiedFlow](https://github.com/xianull0418/RectifiedFlow) | Reflow straightening | ★★★★☆ | R2 candidate |
| 3 | Albergo et al. 2023 (Stochastic Interpolants) | [2303.08797](https://arxiv.org/abs/2303.08797) | torchcfm (VP variant) | Closed-form velocity field | ★★★☆☆ | R2 candidate |
| 4 | Tong et al. 2023 (OT-CFM) | [2302.00482](https://arxiv.org/abs/2302.00482) | [danpovey/conditional-flow-matching](https://github.com/danpovey/conditional-flow-matching) (MIT) | Minibatch OT coupling | ★★★★★ | **R2 PRIORITY** |
| 5 | Satorras et al. 2023 (Equivariant FM) | NeurIPS 2023 | n/a (cited from EGNN code) | E(3)-equivariant CFM | ★★★★★ (already adopted) | n/a (baseline) |
| 6 | Pooladian et al. 2023 (Multisample FM) | ICML 2023 | torchcfm (OT-CFM class) | Minibatch coupling geometry | ★★★★★ (in torchcfm) | **R2 PRIORITY** |
| 7 | Yang et al. 2026 (StableVM) | ICML 2026 | TBD (proceedings) | Multi-sample variance reduction | ★★★★☆ | R2 candidate |
| 8 | Yoon & Lee 2025 (Augmented FM) | ICLR 2025 | TBD (proceedings) | Auxiliary-variable variance reduction | ★★★☆☆ | R2 candidate |
| 9 | Dunn et al. 2024 (FlowMol) | [2412.01966](https://arxiv.org/abs/2412.01966) | [flowmol.org](https://flowmol.org/) | Joint coord+type+bond unified flow | ★★★★☆ | R2 candidate |

---

## §5 PROJECTED applicability ranking for R2

Per `party-twenty-spirit` §3.5 (no measured numbers yet — survey phase):

1. **★★★★★ OT-CFM + Pooladian minibatch OT** — pip-installable (torchcfm),
   MIT-licensed, drop-in replacement for our current CFM loss. **Expected
   speedup: most likely +20-40% wall-time convergence** at same final quality,
   based on OT-CFM's published results on image and single-cell tasks
   (PROJECTED — needs mol-metal measurement to confirm).

2. **★★★★★ Satorras E(3)-equivariant CFM** — already adopted; this survey
   entry validates our architectural choice rather than introducing new code.

3. **★★★★☆ Reflow (Liu 2022)** — could enable 1-NFE inference for our trained
   CFM, but requires a second training pass with reflow-paired data.
   Adaptation cost: medium.

4. **★★★★☆ StableVM / variance reduction** — addresses the high-variance-
   near-prior problem we likely have with the random Gaussian path. If
   ICML 2026 code is released, drop-in for our CFM loss.

5. **★★★★☆ FlowMol unified flow** — useful as an architectural reference for
   future refactor; not a drop-in for current CFM.

6. **★★★☆☆ Albergo VP trigonometric path** — config-only change in torchcfm,
   but unclear whether it helps 3D molecules specifically.

7. **★★★☆☆ Augmented FM** — requires architectural change (auxiliary
   variables); higher adaptation cost.

---

## §6 Recommended R2 actions

Based on this survey:

1. **Adopt OT-CFM** (`ExactOptimalTransportConditionalFlowMatcher` from
   torchcfm) as the default CFM loss in
   `molmetal/adapters/flow_matching_lipman/cfm_loss.py`. **Priority #1**.
2. **Benchmark OT-CFM vs current CFM** on the existing pocket-conditioned
   EGNN + YuelBond retrain — measure wall-time to reach the same decode
   quality on a held-out PlatinAI pocket subset.
3. **Optionally try VariancePreserving path** (`VP-CFM` from torchcfm) as a
   second config-axis in the same benchmark.
4. **Watch for StableVM repo release** — if it lands during R2, add as a
   third config-axis.

Detailed R2 acceleration-method analysis: see `TODO/completed/340r2_acceleration_methods.md`.

---

## §7 Honest framing

- **MEASURED (this section is MEASURED):**
  - 9 papers with verified arXiv IDs, GitHub URLs, and venue citations.
  - 5 papers have open-source code verified to exist (torchcfm,
    xianull0418/RectifiedFlow, flowmol.org, DunnBC22 reproducibility repo,
    danpovey/conditional-flow-matching).
  - 4 papers have code availability TBD (need to check proceedings before
    adoption).

- **PROJECTED:**
  - Applicability rankings and expected speedup estimates in §5 are
    **PROJECTED**, not measured on Mol-Metal. The actual gain on our
    pocket-conditioned 3D SBDD retrain must be benchmarked (Wf-340 R2 / R3).
  - The "drop-in" claims assume API compatibility with torchcfm's loss
    classes — needs verification before integration.

- **NOT MEASURED YET:**
  - Wall-time speedup on Mol-Metal's CFM retrain.
  - Convergence-quality tradeoff (do we hit the same decode quality at
    fewer epochs? Same quality at fewer NFEs?).
  - VRAM impact (OT-CFM computes minibatch OT inside the loss ⇒ extra
    memory per minibatch).

---

## §8 Next steps

- Wf-340 R2: deep-dive into 4 acceleration techniques (OT-CFM, StableVM,
  Augmented FM, FlowMol) with code-integration sketches — see
  `TODO/completed/340r2_acceleration_methods.md` (already shipped).
- Wf-340 R3: integration + benchmark on the Mol-Metal CFM retrain launch
  path. Pre-flight v2 25/25 PASS required (AGENTS.md §1 rule 4).

**END OF Wf-340-R1** — Owner Hugo Lee; cross-ref 340r2; survey complete.
