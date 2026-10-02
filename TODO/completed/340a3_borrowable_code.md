# Wf-340-A3: Borrowable Code Inventory from CFM-Acceleration Repos

**Wf ID:** Wf-340-A3
**Phase:** Adaptation analysis (READ upstream repos + quantify per-repo adaptation cost)
**Date:** 2026-09-29
**Author:** Mol-Metal A3 agent (Ultracode sub-task per user request — see Wf-340 §0)
**Tag discipline:** MEASURED (LOC + licenses per Wf-340-R4 GitHub API survey) / PROJECTED (adaptation
cost + speedup are not benchmarked on Mol-Metal EGNN).

---

## §0 Context (user request)

> "你看一下模型的训练框架能不能再提升，CFM那块，我们模型的本质还是flow matching，
> 应该是能找到那种快速收敛训练的论文和对应的开源代码的，额外启动一个ultracode去做，
> 调研完了得做对应的改进"

Translation: see if CFM training can be improved — find fast-convergence papers + their open-source
code; spin up a separate ultracode to survey; then make improvements.

**This doc's task scope (Wf-340-A3 from workflow script):**
1. Read Wf-340-R4 repo survey (DONE; 15 repos catalogued)
2. For each repo: identify **which code is directly portable** (Triton kernels + PyTorch idioms),
   quantify **adaptation cost** in LOC, list **integration risks**
3. **Rank top 5 repos by reusability** with concrete per-repo adaptation plan
4. Document at `TODO/completed/340a3_borrowable_code.md` (this file)

**Cross-refs:**
- `TODO/completed/340r4_code_repos.md` (R4 survey — 15 repos + license + stars)
- `TODO/completed/340r2_acceleration_methods.md` (R2 acceleration deep-dive — 8 techniques)
- `TODO/completed/340r1_cfm_papers_survey.md` (R1 papers)
- `TODO/completed/340r3_molecule_cfm.md` (R3 molecule-specific CFM)
- `molmetal/adapters/flow_matching_lipman/_reference/` (vendored facebookresearch/flow_matching, CC-BY-NC)
- `molmetal/adapters/flow_matching_lipman/w_flow_cfm_integration.py` (current CFM main, 837 LOC)
- `molmetal/adapters/flow_matching_lipman/w_flow.py` (current W-Flow wrapper, 1066 LOC)
- `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` (current OT, 844 LOC — replace candidate)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (current CFM training entry)

---

## §1 Honest framing header (AGENTS.md §5.2 + party-twenty-spirit §3.5)

- **MEASURED**: per-repo source LOC + license + GitHub stats pulled via Wf-340-R4 (api.github.com +
  raw.githubusercontent.com READMEs). Per-file grep for `def`/`class` boundaries on the locally
  vendored facebookresearch/flow_matching copy + cross-checked against Wf-340-R4 §3 verbatim.
- **MEASURED gap**: `grep -nE "torch\.compile|cudagraphs|CUDAGraph|reduce-overhead|cudnn\.benchmark"`
  against `train_cfm_yuelbond_wasserstein.py` + `w_flow_cfm_integration.py` + `w_flow.py` returns
  **NO matches** — Mol-Metal CFM currently lacks `torch.compile` wrap, `cudnn.benchmark`, and
  CUDA Graphs. Wf-340-R4 §6 GAP claim CONFIRMED.
- **PROJECTED**: adaptation cost + integration speedup are not benchmarked; subjective ranking
  based on architecture match to `EGNNVelocityField` + `YuelBond` + `WFlowCFM` + the
  5-click-Pt/Ir/Ru productivity constraint.
- **License-checked**: every repo row carries a license annotation + commercial-use verdict
  per Wf-340-R4 §7.4.

---

## §2 Per-repo borrowable-code inventory (15 repos)

For each repo: (a) what code is directly portable, (b) adaptation cost in LOC, (c) integration
risks, (d) license verdict.

### §2.1 `NVlabs/edm` — HIGHEST REUSABILITY (1st of top 5)

- **License**: NOASSERTION (CC BY-NC-SA 4.0 per source headers — **verify commercial compatibility
  before adoption**).
- **What code is directly portable**:
  - `training/networks.py::EDMPreconditioner` (~150 LOC) — the σ-preconditioning wrapper:
    `c_skip(σ) = σ_data² / (σ² + σ_data²)`, `c_out(σ) = σ·σ_data / √(σ² + σ_data²)`,
    `c_in(σ) = 1 / √(σ² + σ_data²)`, `c_noise(σ) = 0.25·ln(σ)`. Ports directly to CFM via a
    `CFM_Preconditioner(t)` wrapper that scales velocity-field inputs/outputs instead of
    denoiser inputs/outputs.
  - `training/loss.py::EDMLoss` (~10 LOC) — log-normal σ sampling `P_mean=-1.2, P_std=1.2`.
    Drop-in replacement for Mol-Metal's uniform `t ∈ [0, 1]` sampler.
  - `training/training_loop.py::EMA` (~30 LOC) — exponential moving average with halflife
    ramping. Already partially present in Mol-Metal — verify + complete.
  - `cudnn.benchmark=True` + `torch.nan_to_num(grad, nan=0, posinf=1e5, neginf=-1e5)` +
    `optimizer.zero_grad(set_to_none=True)` (~5 LOC) — PyTorch idioms that cost nothing to add.
  - `training/augment.py::AugmentPipe` (~400 LOC) — AugmentPipe class for geometric + color
    augmentation. **Adapts to 3D molecular coordinates** by replacing pixel ops with SE(3)
    random rotations + translations.
  - `lr_rampup` linear warmup (~5 LOC).
- **Adaptation cost (LOC)**: ~150 LOC (preconditioning) + ~30 LOC (EMA) + ~5 LOC (cudnn +
  nan_to_num) + ~5 LOC (log-normal t-sampling) + ~5 LOC (lr_rampup) = **~195 LOC of new code**
  in `molmetal/adapters/flow_matching_lipman/w_flow_cfm_integration.py` +
  `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`. The 400-LOC AugmentPipe adaptation is
  **separately costed** as R&D (3D-specific) and listed in §2.3 below as Tier-2.
- **Integration risks**:
  - **License**: CC BY-NC-SA 4.0 is share-alike + non-commercial. If Mol-Metal commercializes,
    need a written exception OR a permissively-licensed substitute (none found at this
    fidelity).
  - **Equivariance**: EDM's preconditioner is for image U-Nets; EGNN output is SE(3)-equivariant.
    Need to verify the preconditioner scaling preserves equivariance (the c_in/c_out scalars
    commute with rotation, so it should — but needs a unit test).
  - **Numerical stability**: `c_noise(σ) = 0.25·ln(σ)` requires σ > 0; need σ-floor at ~1e-3.
- **Verdict**: **TOP PRIORITY**. Single biggest convergence speedup (~2-3×) per Wf-340-R2 §2.5
  + Wf-340-R4 §7.5. Wall budget: 2-3 days implementation + 2-day 5-seed × 30-epoch retrain
  verification.

### §2.2 `lucidrains/rectified-flow-pytorch` — HIGH REUSABILITY (2nd of top 5)

- **License**: MIT (permissive — borrowable, no commercial caveat).
- **What code is directly portable**:
  - `rectified_flow_pytorch/mean_flow.py` (275 LOC) — **MeanFlow** (Geng et al. 2025,
    arXiv:2505.13447). The single-most-promising training accelerator per Wf-340-R4 §3.2.
    Uses `torch.func.jvp` (Jacobian-vector product) to share forward passes between two
    objectives, halving wall-time per step. Built-in `use_adaptive_loss_weight` (pseudo-Huber
    p=0.5) + `use_logit_normal_sampler` (P_mean=-0.4, P_std=1.0) + `prob_default_flow_obj=0.5`
    (50% standard FM obj, 50% mean-flow obj).
  - `rectified_flow_pytorch/reflow.py` (~150 LOC) — iterative reflow (Liu et al. 2022) for
    straighter trajectories. Lower priority than MeanFlow.
- **Adaptation cost (LOC)**: MeanFlow is a self-contained loss class — port verbatim into
  `molmetal/adapters/flow_matching_lipman/_reference/` as `mean_flow_loss.py` (~275 LOC
  wholesale copy + ~20 LOC adapter to EGNN's `velocity_fn` signature). Reflow is +~150 LOC
  but with low expected ROI. **Total: ~295 LOC** of new code if both ported, or ~295 LOC
  for MeanFlow alone.
- **Integration risks**:
  - **`torch.func.jvp` on EGNN**: untested. EGNN's message-passing loop may not compose
    cleanly with `jvp`; if it doesn't, requires a custom `autograd.Function` wrapper
    (~+50 LOC). Pilot with a 5-line `jvp` smoke test on EGNN before committing to the port.
  - **MeanFlow vs CFM math**: Geng et al. 2025 targets "mean flow" — average velocity over
    a sub-interval. Lipman CFM targets instantaneous velocity at a single t. Different loss
    targets; the `prob_default_flow_obj=0.5` mixing means we're effectively doing a 50/50
    blend, not a full swap. This is acceptable but should be documented.
  - **Equivariance**: MeanFlow loss weight is a scalar (pseudo-Huber), so equivariance
    should compose — same caveat as EDM, needs a unit test.
- **Verdict**: **HIGH PRIORITY** (best new repo to borrow from per Wf-340-R4 §3.2).
  Wall budget: 3-5 days (1 day port + 1-2 day EGNN-jvp smoke + 1-2 day 5-seed verification).

### §2.3 `NVlabs/Sana` — HIGH REUSABILITY (3rd of top 5)

- **License**: Apache-2.0 (permissive — borrowable, no commercial caveat).
- **What code is directly portable**:
  - `train_scripts/train.py` (~1000 LOC) — modern BF16 + FSDP2 (Fully Sharded Data Parallel)
    + Accelerate launch config. This is **the most current best-practice** training
    infrastructure (actively maintained 2026-09-27).
  - `sana/cli/` CAME-8bit optimizer YAML config — Yong et al. 2023 optimizer that saves
    GPU memory vs AdamW at same convergence. Borrow as a YAML drop-in.
  - `sana/diffusion/` linear-DiT architecture (~500 LOC) — linear attention that gives
    4-8× speedup at long sequences. **Significant architecture change** for EGNN; needs
    ablation.
  - `sana/tools/Sol-Engine/` (~2000+ LOC Triton) — Blackwell-specific kernels. **R&D
    only**; not immediate borrow.
- **Adaptation cost (LOC)**: BF16 + FSDP2 is a config copy (~50 LOC, mostly YAML +
  Accelerate launcher) + training-loop patches (~100 LOC) = **~150 LOC total** to
  `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`. CAME-8bit is +~5 LOC YAML.
  Linear-DiT replacement is **separate cost** (R&D, ~3-week architectural refactor — listed
  in §3.3 as deferred).
- **Integration risks**:
  - **FSDP2 + Blackwell sm_120**: untested on Mol-Metal's current 1-GPU or 2-GPU config.
    Sana's FSDP2 is multi-GPU; Mol-Metal's `train_cfm_yuelbond_wasserstein.py` is
    single-GPU. Need to verify FSDP2 wrapping doesn't break the EGNN encoder + YuelBond
    decoder + WFlowCFM solver stack.
  - **BF16 vs FP32**: Mol-Metal's CFM trainer currently uses mixed BF16 (per Wf-335 R1 fix
    at `train_dmpnn_multitask_wasserstein.py`) but the CFM path may still be FP32 for
    the ODE solver. Need to audit and align.
  - **CAME-8bit optimizer**: needs verification that it converges on Mol-Metal's loss
    surface. AdamW → CAME-8bit is not a drop-in for all tasks.
- **Verdict**: **HIGH PRIORITY** for BF16 + FSDP2 (1.5-2× throughput). Linear-DiT
  **DEFERRED** to a future workflow. Wall budget: 1-2 days (BF16+FSDP2) + 0.5 day (CAME-8bit).

### §2.4 `bkmi/flowmm` — HIGH REUSABILITY (4th of top 5)

- **License**: NOASSERTION (CC-BY-NC per README — **verify commercial compatibility
  before adoption**).
- **What code is directly portable**:
  - `src/flowmm/rfm/vmap.py` (~14 KB) — vectorized map utilities for Riemannian FM
    math.
  - `src/flowmm/rfm/manifold_getter.py` (~21 KB) — manifold construction for flat torus
    lattice parameters.
  - `src/flowmm/model/model_pl.py` (~38 KB) — full PyTorch Lightning training module for
    FM. Reference pattern, not direct port.
  - **Analog bits for atom type** (~100 LOC) — Chen et al. 2022. Reduces 100-d one-hot
    atom-type head to 7-d binary analog bits. **14× compression on atom-type head** →
    measurable speedup on the YuelBond bond-head loss.
- **Adaptation cost (LOC)**: Analog bits ~100 LOC new file
  `molmetal/adapters/flow_matching_lipman/analog_bits.py` + ~30 LOC patch in
  `YuelBondDecoder` to use 7-d analog bits instead of 100-d one-hot. Riemannian FM math
  ~400 LOC but is **R&D only** — not immediate borrow. **Total immediate: ~130 LOC**.
- **Integration risks**:
  - **License**: CC-BY-NC is non-commercial. Same caveat as NVlabs/edm.
  - **Analog bits vs one-hot quality**: the analog bits trick assumes smooth interpolation
    between atom types — Pt(II)/Pt(IV) borderline cases may suffer. Need a small ablation
    on 5-metal × 3-seed to verify.
  - **PyTorch Lightning dependency**: model_pl.py uses Lightning; Mol-Metal uses raw
    PyTorch training loops. Direct port requires rewriting the Lightning Trainer hooks.
- **Verdict**: **HIGH PRIORITY** for analog bits (atom-type compression). Riemannian FM
  math **DEFERRED** to R&D workflow. Wall budget: 1-2 days for analog bits.

### §2.5 `khainb/Mini-batch-OT` — HIGH REUSABILITY (5th of top 5)

- **License**: MIT (permissive — borrowable, no commercial caveat).
- **What code is directly portable**:
  - `BoMb-OT` solver (~500 LOC) — mini-batch Optimal Transport with **partial
    transportation** (drops the most expensive pairings for stability) and **sliced OT**
    (1D projections for cheaper OT approximation).
  - Replaces Mol-Metal's `optimal_transport_v2.py::MeanFieldW2Plan` + `mean_field_*`
    functions for the OT coupling step in CFM training.
- **Adaptation cost (LOC)**: ~500 LOC wholesale port into
  `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` (the existing 844-LOC
  file would be the host for `BoMb-OT` integration) + ~50 LOC adapter to Mol-Metal's
  `velocity_fn` signature. **Total: ~550 LOC**.
- **Integration risks**:
  - **Numerical stability**: partial-transportation drops the most expensive pairings —
    for very-small-batch (n < 16) or pathological pocket-ligand distributions, the dropped
    fraction may include chemically-important pairs. Needs a configurable `drop_fraction`
    (default 0.1 per ICML 2022 paper).
  - **Sliced OT quality**: 1D projections are cheaper but lossy. Need to verify
    Mol-Metal's Wasserstein-pIC50 regression (TODO-179) doesn't degrade.
  - **Performance**: BoMb-OT is a Python solver; Mol-Metal's current
    `optimal_transport_v2.py` uses `mean_field_*` + `ma_regularized_ot_plan` which
    are already efficient. May not yield a wall-time speedup — primarily a
    stability/quality improvement.
- **Verdict**: **MED-HIGH PRIORITY** (replaces an existing module with a more robust
  solver). Wall budget: 1-2 days for port + 1 day for validation.

### §2.6 `facebookresearch/flow_matching` — REFERENCE (already vendored)

- **License**: CC BY-NC 4.0 (NON-COMMERCIAL — affects commercial reuse).
- **What code is already vendored**: full `flow_matching/{loss,path,scheduler,solver,utils}/`
  at `molmetal/adapters/flow_matching_lipman/_reference/`. ~3,500 LOC vendored. 6 R17
  shim fixes applied.
- **Adaptation cost**: **0 LOC** — already integrated. Future updates: pull upstream
  `path/AffineProbPath` periodically for new schedule variants.
- **Integration risks**: license (CC-BY-NC) — same as above.
- **Verdict**: REFERENCE. Already integrated; no new code to borrow. Verify against
  upstream on each Mol-Metal CFM release.

### §2.7 `openai/consistency_models` — MED reusability (deferred)

- **License**: MIT (permissive).
- **What code is portable**: consistency distillation trainer (~2000 LOC) for 1-step
  inference. Needs a pre-trained teacher (we have `fm_pocket` checkpoint).
- **Adaptation cost**: ~2000 LOC port + ~500 LOC adapter. Heavy.
- **Integration risks**: consistency models target denoising, not flow matching —
  adapting requires re-deriving the consistency target for velocity-field outputs
  (~+200 LOC math derivation, not implementation). **Significant R&D risk**.
- **Verdict**: **DEFERRED** to a future workflow (separate SKU). Not on the top-5 list.

### §2.8 `yang-song/score_sde` — MED reusability (reference)

- **License**: Apache-2.0.
- **What code is portable**: Predictor-Corrector (PC) sampler (~150 LOC) for ODE
  solve quality improvement at same step count. VE/VP SDE discretization (~200 LOC)
  for cross-verification.
- **Adaptation cost**: ~150 LOC PC sampler adapter into `w_flow.py::WaveformRelaxationSolver`.
- **Integration risks**: PC adds compute (Langevin dynamics corrector steps) — not a
  speedup, a quality improvement. Trade-off.
- **Verdict**: **MED** priority, optional. Not on top-5.

### §2.9 `mlfoundations/shortcut-models` — NOT FOUND; substitute `kvfrans/shortcut-models`

- **Status**: 404 — substitute is JAX-only, not PyTorch.
- **License**: MIT.
- **Adaptation cost**: full port from JAX to PyTorch (~1000+ LOC).
- **Verdict**: **CONCEPTUAL reference only**. Not on top-5.

### §2.10 `gnobitab/miniOT` — NOT FOUND; see §2.5

- Replaced by `khainb/Mini-batch-OT` (§2.5).

### §2.11 `CompVis/stable-diffusion` — LOW reusability (skip)

- License: CreativeML OpenRAIL-M (use-based restrictions).
- 2D image scope mismatch. Skip.

### §2.12 `CompVis/latent-diffusion` — LOW reusability (skip)

- License: MIT.
- Latent space unrelated to 3D coords. Skip.

### §2.13 `facebookresearch/DiT` — MED reusability (conceptual only)

- License: NOASSERTION.
- DiT architecture incompatible with SE(3) EGNN. AdaLayerNorm-zero trick is portable
  (~10 LOC).
- **Verdict**: not on top-5.

### §2.14 `lucidrains/denoising-diffusion-pytorch` — MED reusability (skip)

- License: MIT.
- DDPM canonical, but Mol-Metal uses FM, not DDPM. Skip.

### §2.15 `lucidrains/rectified-flow-pytorch` soflow / split_mean_flow / recursive_flow

- These are sub-modules of §2.2; same repo, same license (MIT). SoFlow ~150 LOC,
  SplitMeanFlow ~100 LOC, recursive_flow ~100 LOC. Each is a self-contained variant
  with similar adaptation cost as MeanFlow. **Not on top-5** (MeanFlow + Reflow cover
  the expected ROI).

---

## §3 Top-5 reusability ranking (with concrete adaptation plans)

| Rank | Repo | What to borrow | LOC | License | Wall | Speedup (PROJECTED) | Risk |
|------|------|----------------|----:|---------|------|---------------------|------|
| 1 | NVlabs/edm | EDM Preconditioning + Log-normal t-sampling + cudnn.benchmark + EMA | ~195 | CC BY-NC-SA 4.0 (verify) | 2-3 days | 2-3× faster convergence | LOW |
| 2 | lucidrains/rectified-flow-pytorch | MeanFlow (275 LOC) + jvp smoke test | ~295 (+50 if jvp custom autograd) | MIT | 3-5 days | 1.5-2× faster training | MED (jvp on EGNN) |
| 3 | NVlabs/Sana | FSDP2 + BF16 training infrastructure + CAME-8bit | ~155 | Apache-2.0 | 1-2 days | 1.5-2× throughput | LOW |
| 4 | bkmi/flowmm | Analog bits for atom-type head | ~130 | CC-BY-NC (verify) | 1-2 days | 1.2-1.5× atom-type head speedup | MED (quality ablation) |
| 5 | khainb/Mini-batch-OT | BoMb-OT solver replacement | ~550 | MIT | 1-2 days | Better large-batch stability | LOW |

**Aggregated immediate-borrow cost**: ~1,325 LOC of new code (top 5).
**Aggregated immediate-borrow speedup**: **PROJECTED 5-10× faster CFM training** (combining
preconditioning + MeanFlow + FSDP/BF16 + analog bits + mini-batch OT; assuming 2-GPU Blackwell).

### §3.1 Recommended adaptation order

1. **#1 NVlabs/edm preconditioning + log-normal t-sampling** (lowest risk, highest
   expected ROI). 2-3 days. Verify license compatibility first.
2. **#3 NVlabs/Sana BF16 + FSDP2** (Apache-2.0, no license risk). 1-2 days. Aligns
   with Wf-335 R1 `reduce-overhead` torch.compile mode.
3. **#5 khainb/Mini-batch-OT** (MIT, no license risk). 1-2 days. Drop-in for
   `optimal_transport_v2.py`.
4. **#4 bkmi/flowmm analog bits** (CC-BY-NC, verify license). 1-2 days.
5. **#2 lucidrains/rectified-flow-pytorch MeanFlow** (MIT, no license risk). 3-5 days
   including EGNN-jvp smoke test.

**Total wall budget for top-5 implementation**: 8-14 days (~2-3 weeks) including
verification.

### §3.2 Items NOT on top-5 but worth tracking

- **`NVlabs/Sana` Linear-DiT architecture** (~500 LOC, but ~3-week refactor). Deferred
  to R&D workflow after EGNN benchmarks saturate.
- **`bkmi/flowmm` Riemannian FM math** (~400 LOC + 200 LOC math derivation). Deferred
  to R&D workflow.
- **`NVlabs/Sana` Sol-Engine Triton kernels** (~2000+ LOC, Blackwell-specific). Deferred
  to R&D workflow; not aligned with Wf-335 P3 fused Triton kernel work.
- **`openai/consistency_models` consistency distillation** (~2000 LOC). Deferred —
  needs re-derivation for velocity-field outputs.

---

## §4 License verdict matrix (commercial reuse)

| License | Repos | Mol-Metal commercial OK? |
|---------|-------|---------------------------|
| MIT | lucidrains/rectified-flow-pytorch, khainb/Mini-batch-OT, openai/consistency_models, lucidrains/denoising-diffusion-pytorch, CompVis/latent-diffusion, kvfrans/shortcut-models | YES (no attribution beyond copyright notice) |
| Apache-2.0 | yang-song/score_sde, NVlabs/Sana | YES (attribution + NOTICE file) |
| CC BY-NC 4.0 | facebookresearch/flow_matching (already vendored) | **NO** without written exception |
| CC BY-NC-SA 4.0 | NVlabs/edm | **NO** without written exception (share-alike + non-commercial) |
| CC-BY-NC | bkmi/flowmm | **NO** without written exception |
| NOASSERTION | CompVis/stable-diffusion, facebookresearch/DiT | UNCLEAR — defer until clarified |
| CreativeML OpenRAIL-M | CompVis/stable-diffusion | Use-based restrictions — verify |

**Action items**:
- Decide whether Mol-Metal commercializes (Tier-2 paper target per AGENTS.md §5.4 does
  NOT require commercial license). If non-commercial academic use only, all licenses are
  compatible.
- If any of NVlabs/edm / bkmi/flowmm / facebookresearch/flow_matching code is used in a
  commercial product, obtain a written license exception.
- Apache-2.0 + MIT code is safe to use unconditionally with attribution.

---

## §5 Cross-references

- `TODO/completed/340r4_code_repos.md` — R4 survey (15 repos + license + stats)
- `TODO/completed/340r2_acceleration_methods.md` — R2 8 acceleration techniques
- `TODO/completed/340r1_cfm_papers_survey.md` — R1 5 foundational + 4 acceleration papers
- `TODO/completed/340r3_molecule_cfm.md` — R3 10 molecule-specific CFM papers
- `molmetal/adapters/flow_matching_lipman/w_flow_cfm_integration.py` — current CFM main
- `molmetal/adapters/flow_matching_lipman/w_flow.py` — current W-Flow wrapper
- `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` — current OT (BoMb-OT replacement candidate)
- `molmetal/adapters/flow_matching_lipman/_reference/` — vendored facebookresearch/flow_matching
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — current CFM training entry

---

## §6 Per the workflow brief

> "OUTPUT (PLAIN TEXT): n_borrowable + per_repo_adaptation_cost + integration_risks. Do NOT call StructuredOutput."

### §6.1 Plain-text OUTPUT

```
BORROWABLE CODE INVENTORY: 15 repos surveyed, 5 with concrete immediate-borrow plans

TOP 5 REPOS BY REUSABILITY (per Wf-340-A3 ranking):
1. NVlabs/edm (CC BY-NC-SA 4.0 — verify commercial)
   - Borrow: EDMPreconditioner (~150 LOC) + EDMLoss log-normal σ sampling (~10 LOC)
     + cudnn.benchmark + nan_to_num + zero_grad(set_to_none) + EMA + lr_rampup (~40 LOC)
   - Adapt: ~195 LOC new code in w_flow_cfm_integration.py + train_cfm_yuelbond_wasserstein.py
   - Speedup: PROJECTED 2-3x faster convergence (Wf-340-R2 §2.5 MEASURED for EDM, PROJECTED for Mol-Metal)
   - Wall: 2-3 days
   - Risks: license (CC BY-NC-SA), equivariance preservation under preconditioning,
     numerical stability of c_noise = 0.25*ln(sigma) at sigma -> 0
2. lucidrains/rectified-flow-pytorch (MIT)
   - Borrow: MeanFlow (275 LOC, torch.func.jvp-based, arXiv:2505.13447 Geng et al. 2025)
   - Adapt: ~295 LOC (+ 50 LOC if custom autograd.Function needed)
   - Speedup: PROJECTED 1.5-2x faster training
   - Wall: 3-5 days (including EGNN-jvp smoke test)
   - Risks: torch.func.jvp on EGNN is untested; MeanFlow mixes 50/50 standard FM + mean-flow obj
3. NVlabs/Sana (Apache-2.0)
   - Borrow: BF16 + FSDP2 + Accelerate training infra (~1000 LOC patterns, ~155 LOC adapter)
     + CAME-8bit optimizer YAML (~5 LOC)
   - Adapt: ~155 LOC + YAML configs
   - Speedup: PROJECTED 1.5-2x throughput (multi-GPU)
   - Wall: 1-2 days
   - Risks: FSDP2 wrapping may break EGNN encoder + YuelBond decoder + WFlowCFM solver stack;
     BF16 vs FP32 alignment with ODE solver needed
4. bkmi/flowmm (CC-BY-NC — verify commercial)
   - Borrow: Analog bits for atom-type head (~100 LOC, Chen et al. 2022)
     — 100-d one-hot -> 7-d binary analog bits = 14x compression
   - Adapt: ~130 LOC (new analog_bits.py + YuelBondDecoder patch)
   - Speedup: PROJECTED 1.2-1.5x atom-type head speedup
   - Wall: 1-2 days
   - Risks: license (CC-BY-NC), atom-type borderline cases (Pt II/IV) may suffer
5. khainb/Mini-batch-OT (MIT)
   - Borrow: BoMb-OT solver with partial transportation + sliced OT (~500 LOC)
   - Adapt: ~550 LOC replacement for optimal_transport_v2.py
   - Speedup: PROJECTED stability improvement at large batch (>512); not necessarily wall-time
   - Wall: 1-2 days
   - Risks: partial-transportation may drop chemically-important pairs at n < 16;
     sliced OT 1D projections lossy

AGGREGATED IMMEDIATE-BORROW: ~1,325 LOC of new code (top 5).
PROJECTED AGGREGATED SPEEDUP: 5-10x faster CFM training (per Wf-340-R4 §7.5).
TOTAL WALL BUDGET: 8-14 days (~2-3 weeks) including 5-seed verification.

LICENSE VERDICT:
- MIT (lucidrains, khainb, openai): borrowable without issue
- Apache-2.0 (Sana, score_sde): borrowable with attribution
- CC BY-NC / CC BY-NC-SA (edm, flowmm, flow_matching): non-commercial only
- NOASSERTION (DiT, SD): defer until clarified

INTEGRATION RISKS (cross-cutting):
- SE(3) equivariance preservation under preconditioning (NVlabs/edm, MeanFlow) — needs unit test
- BF16 vs FP32 alignment with CFM ODE solver (NVlabs/Sana)
- FSDP2 wrapping compatibility with EGNN encoder + YuelBond decoder (NVlabs/Sana)
- torch.func.jvp compatibility with EGNN message passing (lucidrains)
- Atom-type borderline cases under analog-bits compression (bkmi/flowmm)
- Partial-transportation drop-fraction at small batch (khainb)

DEFERRED (not on top-5):
- NVlabs/Sana Linear-DiT (~500 LOC, 3-week refactor) — R&D workflow
- bkmi/flowmm Riemannian FM math (~400 LOC + 200 LOC math derivation) — R&D workflow
- NVlabs/Sana Sol-Engine Triton kernels (~2000+ LOC, Blackwell-specific) — R&D workflow
- openai/consistency_models consistency distillation (~2000 LOC + math re-derivation) — separate SKU
```

---

**END OF 340a3_borrowable_code.md** — MEASURED 15 repos × license + LOC from Wf-340-R4;
PROJECTED top-5 ranking + ~1,325 LOC immediate-borrow cost + 5-10× speedup.
License review needed for CC-BY-NC / CC-BY-NC-SA sources (edm + flowmm + vendored flow_matching).
