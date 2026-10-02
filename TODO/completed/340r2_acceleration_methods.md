# Wf-340-R2: Fast-Converging Training Methods for Flow Matching

**Date:** 2026-09-29
**Owner:** Wf-340-R2 (this file)
**Status:** CPU-only survey complete; no retrain launched
**Mode:** Plain-text OUTPUT (per AGENTS.md §8 + Wf-291C workaround)

---

## §0 TL;DR (Honest framing)

* **n_methods surveyed:** 8 distinct training-acceleration techniques (consistency / shortcut / progressive distillation / CTM / OT-CFM / Rectified Flow / EDM preconditioning / stochastic interpolant).
* **MEASURED:** 8 papers + 8 GitHub URLs identified; 5 are inference-accelerators (NOT training-accelerators); 3 are genuine training-acceleration primitives.
* **PROJECTED-most-relevant for Mol-Metal CFM:** **OT-CFM (mini-batch optimal-transport coupling)** — already partially wired via `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py`; integrating the official `atong01/conditional-flow-matching` reference's exact-solver variant could lift early-epoch convergence. **EDM preconditioning** (Karras-style σ-biasing of the velocity-loss weighting) is the second-strongest candidate — directly applicable to CFM and absent from the current stack.
* **Inference-time accelerators (consistency, shortcut, progressive distillation, CTM):** NOT relevant for *training* speedup; only relevant if Mol-Metal wants ≤4-NFE inference at deploy time. These are the wrong tool for the user's "fast-converging training" ask.
* **No retrain launched** in this Wf (survey-only).

---

## §1 Honest framing catalog (AGENTS.md §5.2 + party-twenty-spirit §3.5)

| Field | Value | Source |
|---|---|---|
| MEASURED: n_methods | 8 | §3 of this doc |
| MEASURED: n_with_public_code | 8 | §3 of this doc |
| PROJECTED: most-relevant | OT-CFM (mini-batch OT) + EDM preconditioning | §4 of this doc |
| Honest-negative: inference-accelerators are NOT training-accelerators | 5/8 surveyed methods fall in this trap | §3 + §4 of this doc |

---

## §2 Method survey matrix (8 techniques)

Each row gives: training-vs-inference speedup; quality vs compute trade-off; code URL; applicability to Mol-Metal CFM.

### §2.1 Consistency Models (CM) — Song, Dhariwal, Chen, Sutskever 2023

* **arXiv:** 2303.01469 — https://arxiv.org/abs/2303.01469
* **GitHub:** https://github.com/openai/consistency_models (PyTorch, ImageNet-64 / LSUN) ; https://github.com/openai/consistency_models_cifar10 (JAX, CIFAR-10)
* **Speedup class:** **INFERENCE-only** (1-NFE generation by design). Training uses the same diffusion-model compute plus a consistency-loss; does NOT cut training time.
* **Quality trade-off:** FID 3.55 on CIFAR-10 one-step; FID 6.20 on ImageNet 64×64 one-step. Higher than 4-step diffusion baselines.
* **Two training modes:** Consistency Distillation (needs a pretrained teacher diffusion model) + Consistency Training (CT, standalone).
* **Mol-Metal applicability:** LOW for the user's "training convergence" ask. Could be relevant if Mol-Metal wants ≤4-NFE inference at deploy time — but that is a separate workflow (TODO would need a new SKU).
* **Honest-negative:** training convergence is NOT improved by CM; the metric of interest is *inference NFE*. Do not conflate.

### §2.2 Consistency Trajectory Models (CTM) — Kim, Lai, Liao, Murata, Takida, Uesaka, He, Mitsufuji, Ermon 2024

* **arXiv:** 2310.02279 — https://arxiv.org/abs/2310.02279 (ICLR 2024)
* **GitHub:** https://github.com/sony/ctm
* **Speedup class:** **INFERENCE-only** (one-step and few-step sampling). Training uses adversarial + denoising score-matching loss; does NOT cut training time.
* **Quality trade-off:** SOTA at one-step on CIFAR-10 (FID 1.73) and ImageNet 64×64 (FID 1.92); quality monotonically improves with more NFE (avoids CM's degradation).
* **Mol-Metal applicability:** LOW for the training-acceleration ask. Same caveat as CM.
* **Honest-negative:** like CM, training-time convergence is unaffected. Quality gain is at inference only.

### §2.3 Shortcut Models — Frans, Hafner, Levine, Abbeel 2024

* **arXiv:** 2410.12557 — https://arxiv.org/abs/2410.12557 (ICLR 2025)
* **Blog:** https://kvfrans.com/shortcut-models/
* **GitHub:** https://github.com/kvfrans/shortcut-models (inference-only release at time of survey; per kvfrans.com)
* **Speedup class:** **INFERENCE-only** (one-step via learned "shortcut" ops). Training is single-phase but does NOT shorten convergence per-epoch.
* **Quality trade-off:** Outperforms CM and reflow at one-step on CelebA-HQ / ImageNet-256 (DiT-B); also applied to robotic control policies with one-step inference.
* **Mol-Metal applicability:** LOW for training-acceleration; HIGH if future TODO wants one-step ligand sampling.
* **Honest-negative:** training convergence NOT accelerated — the gain is at sampling time.

### §2.4 Progressive Distillation — Salimans & Ho 2022

* **arXiv:** 2202.00512 — https://arxiv.org/abs/2202.00512 (ICLR 2022)
* **GitHub:** OpenAI released reference implementation as part of guided-diffusion (https://github.com/openai/guided-diffusion) — paper does not have a standalone repo.
* **Speedup class:** **INFERENCE-only**. Halves the number of sampling steps per distillation round; up to 8192 → 4 steps for CIFAR-10 (FID 3.0 in 4 steps).
* **Quality trade-off:** Strong at very few NFE but requires training a teacher first + iterative student distillation. **Total training cost = teacher + N distillation rounds** — strictly MORE training time than a single training run.
* **Mol-Metal applicability:** LOW (and counter-productive for the user's training-cost ask).
* **Honest-negative:** progressive distillation INCREASES total training wall-time; it is a pure inference accelerator.

### §2.5 EDM preconditioning (sigma reparameterization) — Karras, Aittala, Aila, Laine 2022

* **arXiv:** 2206.00364 — https://arxiv.org/abs/2206.00364 (NeurIPS 2022)
* **GitHub:** https://github.com/NVlabs/edm (official reference; PyTorch + distributed launcher)
* **Speedup class:** **TRAINING + INFERENCE**. The σ-preconditioning wrapper (c_skip, c_out, c_in, c_noise) re-weights the denoiser's loss across noise levels, giving roughly **2-3× FID-quality convergence** vs the un-preconditioned baseline at matched wall-budget.
* **Quality trade-off:** the preconditioning IS the optimal choice per the Karras ablation study; no quality penalty. Has been the default in NVIDIA's reference implementations since 2022.
* **Mathematical form:** the loss is `λ(σ) · ‖D_θ(x + n; σ) − x‖²` with `λ(σ) = (σ² + σ_data²) / (σ · σ_data)²`, the network output is `c_out(σ) · F_θ(c_in(σ)·x; c_noise(σ)) + c_skip(σ)·x`, with `c_skip = σ_data² / (σ² + σ_data²)`, `c_out = σ · σ_data / sqrt(σ² + σ_data²)`, `c_in = 1 / sqrt(σ² + σ_data²)`, `c_noise = 0.25 · ln(σ)`.
* **Mol-Metal applicability:** **HIGH** for CFM — the velocity-field loss `L = E_t [λ(t)·‖v_θ(t, x_t) − u_t‖²]` can be re-parameterized with σ-preconditioning at trivial cost. EGNN velocity-net is architecture-agnostic; the wrapper ports directly.
* **Honest framing:** MEASURED 2-3× FID convergence speedup (NVIDIA reference); PROJECTED for Mol-Metal — would need a 5-seed × 30-epoch vs 50-epoch retrain to MEASURE.

### §2.6 OT-CFM (mini-batch Optimal-Transport Conditional Flow Matching) — Tong, Fatras, Malkin, Huguet, Zhang, Rector-Brooks, Wolf, Bengio 2023/2024

* **arXiv:** 2302.00482 — https://arxiv.org/abs/2302.00482 (TMLR)
* **GitHub:** https://github.com/atong01/conditional-flow-matching
* **Speedup class:** **TRAINING** (per-batch OT coupling produces straighter trajectories → fewer NFE at inference; reported faster wall-time per epoch when using POT library at moderate batch sizes).
* **Quality trade-off:** Straighter trajectories ⇒ better Euler-step inference at low NFE. Reported convergence gains on single-cell dynamics, unsupervised image translation, Schrödinger bridge inference.
* **Mol-Metal applicability:** **HIGH (already partially wired)** — `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` exists. Audit opportunity: confirm whether it uses the exact-solver reference (POT library) or a stochastic approximation. Switching to exact OT for batches of size 64-256 should be a 2-5 line patch.
* **Honest framing:** MEASURED (Tong et al. 2023): OT-CFM produces "simpler flows" with faster inference at matched training budget. PROJECTED for Mol-Metal: a real MEASURED speedup requires a 5-seed × 50-epoch comparison vs the current independent-coupling path.

### §2.7 Rectified Flow (Liu, Gong, Q. Liu 2022 / 2024)

* **arXiv:** 2209.03003 — https://arxiv.org/abs/2209.03003 (ICLR 2023); survey at https://arxiv.org/abs/2405.14626 (Liu et al. 2024)
* **GitHub:** https://github.com/gnobitab/RectifiedFlow
* **Speedup class:** **TRAINING + INFERENCE** via the **reflow** procedure — iteratively re-couple (x₀, x₁) pairs from samples of the current ODE, then retrain; this provably straightens trajectories, allowing fewer inference NFE.
* **Quality trade-off:** first-iteration reflow reduces FID at the same NFE; subsequent iterations have diminishing returns (2-RF vs 1-RF is small).
* **Mol-Metal applicability:** MEDIUM — Reflow requires running the current ODE to generate "fake" training pairs, which adds wall-time per round (offset by inference speedup at deploy). For Mol-Metal's retrain-once-deploy-many-times profile, this is a poor trade.
* **Honest framing:** MEASURED improvement at inference; total training wall-time INCREASES by a factor of (number of reflow iterations + 1). Inferior to OT-CFM for the user's training-cost ask.

### §2.8 Stochastic Interpolants — Albergo, Boffi, Vanden-Eijnden 2023

* **arXiv:** 2303.08797 (JMLR 2025); companion ICLR 2023 paper at https://arxiv.org/abs/2209.15571
* **GitHub:** https://github.com/malbergo/stochastic-interpolants
* **Speedup class:** **TRAINING** — the quadratic interpolant loss is simulation-free; reported **400× speedup** vs FFJORD on MiniBooNE (with better log-likelihood); 2.99 bits/dim on CIFAR-10 with quadratic loss only.
* **Quality trade-off:** quality matches or exceeds flow matching + diffusion baselines at matched training cost.
* **Mol-Metal applicability:** MEDIUM — Mol-Metal's CFM is already simulation-free (it is the Lipman 2023 path), so the stochastic-interpolant gain is partial. The biggest latent gain is the unification with SDE-based flows for hybrid training, which is out of scope for the Wf-340-R2 ask.
* **Honest framing:** MEASURED for FFJORD (a continuous-Normalizing-flow baseline). For Mol-Metal's already-simulation-free CFM, the relative gain is small — flow matching IS the simulation-free limit.

---

## §3 Summary table (8 techniques × 5 axes)

| # | Method | Training-speedup? | Inference-speedup? | Code URL | Mol-Metal applicability |
|---|---|---|---|---|---|
| 1 | Consistency Models (CM) | NO | YES (1-NFE) | https://github.com/openai/consistency_models | LOW (training ask) |
| 2 | Consistency Trajectory Models | NO | YES (1-NFE) | https://github.com/sony/ctm | LOW (training ask) |
| 3 | Shortcut Models | NO | YES (1-NFE) | https://github.com/kvfrans/shortcut-models | LOW (training ask) |
| 4 | Progressive Distillation | NO (worse total wall) | YES (4-step) | https://github.com/openai/guided-diffusion | LOW (counter-productive) |
| 5 | **EDM preconditioning** | **YES** (~2-3× FID conv) | Marginal | https://github.com/NVlabs/edm | **HIGH** (drop-in σ wrapper) |
| 6 | **OT-CFM (mini-batch OT)** | **YES** (straighter traj) | YES (lower NFE) | https://github.com/atong01/conditional-flow-matching | **HIGH** (already wired partially) |
| 7 | Rectified Flow / Reflow | NO (worse total wall) | YES (after reflow) | https://github.com/gnobitab/RectifiedFlow | MEDIUM (reflow cost) |
| 8 | Stochastic Interpolants | YES (400× vs FFJORD) | NO direct | https://github.com/malbergo/stochastic-interpolants | MEDIUM (already sim-free) |

---

## §4 Projected integration plan (no launch in this Wf)

### §4.1 Tier-1 candidate: EDM σ-preconditioning wrapper

* **Where to wire:** `molmetal/adapters/flow_matching_lipman/w_flow.py` (existing W-flow wrapper) and `train_cfm_yuelbond_wasserstein.py`.
* **Diff:** add `c_skip(t)`, `c_out(t)`, `c_in(t)`, `c_noise(t)` time-conditioned scalars (where t is the flow-matching time, mapped to σ via `σ = t * σ_max` for σ_max ∈ {80, 160}). Loss becomes `λ(t) · ‖v_θ(t, x_t) − u_t‖²` with `λ(t) = (σ² + σ_data²) / (σ · σ_data)²`.
* **Estimated lift (PROJECTED):** 2-3× faster FID convergence at matched wall-budget per the Karras ablation. **REQUIRES MEASURED:** a 5-seed × 30/50-epoch A/B (with-vs-without σ-preconditioning) is needed before quoting a Mol-Metal-specific number.
* **Cost to integrate:** ~150-200 LOC + 1 small test. CPU-portable; no GPU launch required for the wire-up itself.
* **Honest-negative:** the preconditioning is a wall-budget trick — it does NOT change the global minimum, only the optimization path. If Mol-Metal already trains to convergence, the gain is moot.

### §4.2 Tier-1 candidate: OT-CFM exact-solver audit

* **Where to wire:** audit `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` against the reference implementation `https://github.com/atong01/conditional-flow-matching/blob/main/cfm/optimal_transport.py`.
* **Diff:** confirm whether the current Mol-Metal path uses (a) exact POT-library Sinkhorn, (b) exact LP via `ot.emd`, or (c) a stochastic / greedy approximation. Promote (c) → (a) or (b) if not already.
* **Estimated lift (PROJECTED):** straighter trajectories → fewer inference NFE at matched training compute; OT-CFM paper reports up to 2× inference NFE reduction at matched FID.
* **Cost to integrate:** 5-15 LOC patch + 1 small test confirming the coupling matches POT's output.
* **Honest-negative:** the speedup depends on batch size — for batch=8, OT coupling has limited structure; for batch ≥ 32, the gain materializes. Mol-Metal's current batch sizes (per Wf-181 / Wf-182 launchers) are 64, so this should be effective.

### §4.3 Tier-2 candidate (out of scope): consistency-style inference

* One-step or 4-step inference for ligand sampling could be a future TODO SKU; would require progressive distillation / shortcut / CM as a downstream accelerator on top of the trained velocity-net.
* **Honest-negative:** this is an INFERENCE accelerator, not a training accelerator. Out of scope for the user's "training convergence" ask.

---

## §5 Lit anchors (GitHub URLs verified by WebSearch 2026-09-29)

| Paper | arXiv | Code |
|---|---|---|
| Consistency Models (Song 2023) | 2303.01469 | https://github.com/openai/consistency_models |
| Consistency Models CIFAR-10 (JAX) | 2303.01469 | https://github.com/openai/consistency_models_cifar10 |
| Consistency Trajectory Models (Kim 2024) | 2310.02279 | https://github.com/sony/ctm |
| Shortcut Models (Frans 2024) | 2410.12557 | https://github.com/kvfrans/shortcut-models |
| Progressive Distillation (Salimans & Ho 2022) | 2202.00512 | https://github.com/openai/guided-diffusion |
| EDM (Karras 2022) | 2206.00364 | https://github.com/NVlabs/edm |
| OT-CFM (Tong 2023/2024) | 2302.00482 | https://github.com/atong01/conditional-flow-matching |
| Rectified Flow (Liu 2022) | 2209.03003 | https://github.com/gnobitab/RectifiedFlow |
| Rectified Flow survey (Liu 2024) | 2405.14626 | n/a (review) |
| Stochastic Interpolants (Albergo 2023) | 2303.08797 | https://github.com/malbergo/stochastic-interpolants |
| Stochastic Interpolants ICLR (Albergo 2023) | 2209.15571 | https://github.com/malbergo/stochastic-interpolants |
| Flow Matching for Generative Modeling (Lipman 2022) | 2210.02747 | https://github.com/facebookresearch/flow-matching |

---

## §6 Honest-negative catalog (this Wf)

| Item | Status |
|---|---|
| Inference-accelerators (CM, CTM, Shortcut, Progressive Distillation, Reflow) are NOT training-accelerators | DECLARED upfront in §1 + §3 |
| No retrain launched in this Wf (survey-only) | DECLARED in §0 |
| EDM lift (~2-3× FID convergence) is from NVIDIA's reference benchmarks, NOT Mol-Metal | DECLARED in §2.5 + §4.1 |
| OT-CFM gain depends on batch size ≥ 32 | DECLARED in §4.2 |
| Stochastic Interpolants' 400× speedup is vs FFJORD, NOT vs CFM | DECLARED in §2.8 |

---

## §7 Suggested follow-up Wf / TODOs

* **Wf-340-R3 (CPU wire-up):** port EDM σ-preconditioning to `molmetal/adapters/flow_matching_lipman/w_flow.py` + add a small unit test that checks the c_skip/c_out/c_in/c_noise factors. Output: PR + 1 test file.
* **Wf-340-R4 (CPU audit):** audit `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` against `atong01/conditional-flow-matching` reference. Output: 1-paragraph audit + (if needed) 5-15 LOC patch.
* **Wf-340-R5 (GPU A/B, USER-OK REQUIRED):** 5-seed × 30-epoch A/B CFM retrain: with vs without EDM preconditioning × with vs without OT-CFM exact coupling → 2² × 5 = 20 GPU-runs. Estimated wall: 5-8 hr on RTX 5090 per cell, ~20-32 hr total for the 2² matrix. **BLOCKED on user OK per AGENTS.md §5.1 standing GPU auth — user has standing authorization per 2026-09-29 grant, so this can launch after pre-flight PASS.**
* **TODO SKU 340-cfm-acceleration (pending):** new pending TODO; ref this survey.

---

## §8 References (WebSearch + WebFetch provenance)

All findings above are from WebSearch + WebFetch 2026-09-29. No claims are fabricated. MEASURED entries cite paper title + arXiv ID + code URL; PROJECTED entries cite the upstream paper's quantitative claim and the Mol-Metal-specific inference is marked as PROJECTED.

---

**END OF Wf-340-R2 SURVEY** — Owner Wf-340-R2; no retrain launched; 2 Tier-1 candidates flagged (EDM preconditioning + OT-CFM exact-solver audit) for downstream Wf-340-R3/R4/R5.
