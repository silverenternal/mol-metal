# Wf-340-A2: NEW Acceleration Opportunities NOT in Our Code

**Date:** 2026-09-29
**Wave:** Wf-340 (CFM training framework improvement ultracode)
**Phase:** A2 — Coverage matrix + opportunity ranking
**Cross-refs:**
- `TODO/completed/340r1_cfm_papers_survey.md` (R1 — 9 papers)
- `TODO/completed/340r2_acceleration_methods.md` (R2 — 8 acceleration methods deep-dive)
- `TODO/completed/340r3_molecule_cfm.md` (R3 — 10 molecule-specific CFM papers)
- `TODO/completed/340r4_code_repos.md` (R4 — 15 code repos)
**Tag discipline:** MEASURED (coverage status by reading code) / PROJECTED (speedup estimates, cost)
**Owner:** Wf-340-A2 agent

---

## §0 TL;DR (Honest framing)

* **Coverage matrix:** 26 acceleration techniques evaluated; 8 already in our code; 18 MISSING or partial.
* **Top-5 highest speedup/cost ratio opportunities (ranked):**
  1. **EDM σ-preconditioning** — 2-3× faster convergence, ~150 LOC, LOW risk, license CC-BY-NC-SA 4.0 (verify).
  2. **Log-normal t-sampling** — drop-in replacement, ~5 LOC, LOW risk, no license issue (formula only).
  3. **torch.compile + CUDA Graphs for CFM velocity field** — 1.3-1.8× training step, ~30 LOC, LOW risk (already done for D-MPNN).
  4. **MeanFlow (jvp-based)** — 1.5-2× faster training, ~200 LOC, MED risk (untested on EGNN), MIT license.
  5. **FSDP2 + CAME-8bit multi-GPU** — 1.5-2× multi-GPU throughput, 0.7× memory, ~100 LOC, LOW risk, Apache-2.0.
* **Total PROJECTED aggregated speedup (top-5 implemented): 5-10× faster CFM training** (per R4 §7.5; cross-validated).
* **Honest-negative:** every speedup claim below is **PROJECTED** from upstream papers' benchmarks, NOT measured on Mol-Metal. The license on #1 (NVlabs/edm) is CC-BY-NC-SA 4.0 which has commercial reuse implications for Tier-2 paper work; legal review needed.
* **No retrain launched** in this Wf — coverage matrix + ranking only.

---

## §1 Coverage matrix — what we have vs what's missing

This section is **MEASURED** by reading the code (Bash grep over `molmetal/adapters/flow_matching_lipman/`, `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`, `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`).

### §1.1 Already in our code (8 techniques)

| # | Technique | Where | Status |
|---|-----------|-------|--------|
| 1 | Lipman CFM (baseline FM loss) | `molmetal/adapters/flow_matching_lipman/__init__.py:2877` (`LipmanFlowMatching_v2_equivariant`) | SHIPPED |
| 2 | EGNN-based equivariant velocity field | `__init__.py:1049` ("SE(3) equivariant by construction") | SHIPPED |
| 3 | BF16 autocast (D-MPNN trainer) | `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:271` `DEFAULT_BF16=True` + `molmetal/adapters/flow_matching_lipman/amp.py` | SHIPPED |
| 4 | Wasserstein pIC50 regression (W₂ closed form) | `train_dmpnn_multitask_wasserstein.py:207` (`torch.compile(wasserstein_pic50_gradient)`) | SHIPPED |
| 5 | Optimal Transport coupling (mean-field upgrade) | `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py:80` (re-exports upstream `mini_batch_ot_coupling`) | SHIPPED — but uses **stochastic approx** upstream, NOT exact POT (R2 §2.6 + R4 §3.7 audit) |
| 6 | torch.compile wrap for D-MPNN | `train_dmpnn_multitask_wasserstein.py:163-195` (`compile _forward_one_mol`) | SHIPPED (D-MPNN only) |
| 7 | Lipman reference (vendored) | `molmetal/adapters/flow_matching_lipman/_reference/` (CC-BY-NC, 6 R17 shim fixes) | SHIPPED |
| 8 | Triton fused ops (4 kernels) | `_reference/dmpnn_triton_kernel.py` + `_reference/fused_*.py` per AGENTS.md §6 Wf-333-T2 | SHIPPED (D-MPNN only) |

### §1.2 MISSING / partial (18 techniques — these are the opportunities)

Group A — **EDM / Karras recipe** (highest speedup/cost):

| # | Technique | Status | Source repo |
|---|-----------|--------|-------------|
| A1 | **EDM σ-preconditioning** (`c_skip`/`c_out`/`c_in`/`c_noise`) | MISSING (absent from CFM trainer) | NVlabs/edm |
| A2 | **Log-normal t-sampling** (`P_mean=-1.2, P_std=1.2`) | MISSING (uses uniform `t ∈ [0, 1]`) | NVlabs/edm |
| A3 | **AugmentPipe for 3D SE(3)** | MISSING (no augmentation in CFM loop) | NVlabs/edm |
| A4 | **lr_rampup** (linear warmup) | MISSING (no warmup, plain step) | NVlabs/edm |
| A5 | **EMA decay with halflife** | MISSING (no EMA tracking) | NVlabs/edm |
| A6 | **cudnn.benchmark = True** | MISSING (not set anywhere) | NVlabs/edm |
| A7 | **nan_to_num** gradient guard | MISSING (no numerical safety) | NVlabs/edm |

Group B — **MeanFlow / Reflow** (training-acceleration primitives):

| # | Technique | Status | Source repo |
|---|-----------|--------|-------------|
| B1 | **MeanFlow** (`torch.func.jvp`-based mean-velocity objective) | MISSING (Geng et al. 2025 arXiv:2505.13447) | lucidrains/rectified-flow-pytorch |
| B2 | **Reflow** (iterative trajectory straightening) | MISSING (Liu et al. 2022 / ICLR 2023) | gnobitab/RectifiedFlow |

Group C — **Path / variance-reduction variants**:

| # | Technique | Status | Source repo |
|---|-----------|--------|-------------|
| C1 | **VariancePreserving CFM** (trigonometric interpolation) | MISSING (uses Gaussian path) | torchcfm `VariancePreservingConditionalFlowMatcher` |
| C2 | **Schrödinger Bridge CFM** (entropy-regularized OT) | MISSING | torchcfm `SchrodingerBridgeConditionalFlowMatcher` |
| C3 | **StableVM** (multi-sample variance reduction, ICML 2026) | MISSING | Yang et al. 2026 (TBD repo) |
| C4 | **Augmented FM** (auxiliary variables, ICLR 2025) | MISSING | Yoon & Lee 2025 (TBD repo) |

Group D — **Molecule-specific (R3)**:

| # | Technique | Status | Source repo |
|---|-----------|--------|-------------|
| D1 | **SO(3)-averaged FM loss** (replaces Kabsch alignment) | MISSING (uses Kabsch alignment per R3 §3.9) | Cao et al. NVIDIA 2025 (arXiv:2507.09785) |
| D2 | **FlowMol SimplexFlow** (categorical atom-type) | MISSING (uses one-hot CE per R3 §3.1) | dunni3/FlowMol |
| D3 | **GOAT coupling purification** | MISSING (per R3 §3.8) | WanyuGroup/ICLR2025-GOAT |
| D4 | **ES-reg on YuelBlock** | MISSING (per R3 §3.5) | GEOLDM |

Group E — **Modern training infrastructure** (R4):

| # | Technique | Status | Source repo |
|---|-----------|--------|-------------|
| E1 | **torch.compile + CUDA Graphs (CFM velocity field)** | MISSING (only D-MPNN has it) | NVlabs/Sana (mode='reduce-overhead') |
| E2 | **FSDP2 multi-GPU** | MISSING (single-GPU only) | NVlabs/Sana |
| E3 | **CAME-8bit optimizer** | MISSING (uses `optim.Adam` per `train_cfm_yuelbond_wasserstein.py:1305`) | NVlabs/Sana |

---

## §2 Top-10 opportunities ranked by speedup / cost ratio

Each row: implementation cost (LOC), PROJECTED speedup, compatibility risk, license.

### §2.1 Rank #1: EDM σ-preconditioning

* **Implementation cost:** ~150 LOC (port `EDMPreconditioner` from NVlabs/edm `training/networks.py:35.8KB` to `molmetal/adapters/flow_matching_lipman/__init__.py`; adapt for velocity-field output instead of denoiser).
* **PROJECTED speedup:** 2-3× faster FID convergence at matched wall-budget (per Karras 2022 ablation §5). For our pocket-conditioned CFM retrain: MEASURED-claim-from-paper, needs Mol-Metal A/B to confirm.
* **Compatibility risk:** **LOW** — c_skip/c_out/c_in/c_noise are scalar time-conditioned wrappers, drop-in around `EGNNVelocityField`. License **CC-BY-NC-SA 4.0** (R4 §7.4) — verify before adoption.
* **Where to wire:** `molmetal/adapters/flow_matching_lipman/w_flow.py` (existing W-flow wrapper) + `train_cfm_yuelbond_wasserstein.py:1305` (Adam optimizer step).
* **Honest-negative:** the preconditioning is a wall-budget trick — it does NOT change the global minimum, only the optimization path. If CFM already trains to convergence, gain is moot. (R2 §4.1 HONEST note.)

### §2.2 Rank #2: Log-normal t-sampling

* **Implementation cost:** ~5 LOC — replace `torch.rand(batch) * 1.0` with `(torch.randn(batch) * 1.2 - 1.2).exp().clamp(0, 1)` in the loss wrapper.
* **PROJECTED speedup:** implicit in EDM preconditioning (R2 §3.5); ~1.5-2× faster convergence of the EDMPreconditioner path at matched step count.
* **Compatibility risk:** **LOW** — drop-in replacement of the time-sampling. License **CC-BY-NC-SA 4.0** (formula only, not code — license review trivially OK).
* **Where to wire:** `molmetal/adapters/flow_matching_lipman/w_flow_cfm_integration.py` (current CFM main flow).
* **Honest-negative:** must be paired with EDM preconditioning to realize full benefit; standalone benefit is small.

### §2.3 Rank #3: torch.compile + CUDA Graphs for CFM velocity field

* **Implementation cost:** ~30 LOC (mirroring `train_dmpnn_multitask_wasserstein.py:163-195` pattern).
* **PROJECTED speedup:** 1.3-1.8× training step throughput (per NVlabs/Sana benchmarks; same Blackwell / RTX 5090 hardware as our setup).
* **Compatibility risk:** **LOW** — D-MPNN already does this in our code; same approach extends to `EGNNVelocityField`. Need `--no-cudagraphs` safety lever per Wf-335-R3.
* **Where to wire:** `molmetal/adapters/flow_matching_lipman/__init__.py:1049` (EGNNVelocityField class).
* **Honest-negative:** Blackwell sm_120 has CUDA Graphs cudagraphs correctness gap (Wf-335-R3); must guard with env-var lever.

### §2.4 Rank #4: MeanFlow (jvp-based)

* **Implementation cost:** ~200 LOC (port `rectified_flow_pytorch/mean_flow.py:275 LOC` from lucidrains to `molmetal/adapters/flow_matching_lipman/mean_flow.py`).
* **PROJECTED speedup:** 1.5-2× faster training per the Wf-340-R2 §3.1 analysis (Geng et al. 2025). Single biggest published speedup in our R4 survey per repo.
* **Compatibility risk:** **MED** — `torch.func.jvp` untested on EGNN output (equivariance may need special handling). Requires custom autograd.Function fallback if jvp doesn't work on the equivariant net. License **MIT** (R4 §3.2) — borrowable.
* **Where to wire:** NEW file `molmetal/adapters/flow_matching_lipman/mean_flow.py`; called from `w_flow_cfm_integration.py` behind a `--use-mean-flow` CLI flag.
* **Honest-negative:** the 1.5-2× claim is from paper benchmarks on image tasks (CIFAR / ImageNet); needs Mol-Metal A/B on PlatinAI 4-cell to confirm 3D-molecular applicability. (R4 §7.2: R&D-tier work.)

### §2.5 Rank #5: FSDP2 + CAME-8bit multi-GPU

* **Implementation cost:** ~100 LOC (copy `train_scripts/train.py` patterns from NVlabs/Sana; CAME-8bit optimizer is YAML config change).
* **PROJECTED speedup:** 1.5-2× multi-GPU throughput (FSDP2 scales linearly up to 4 GPUs); 0.7× memory at same speed (CAME-8bit vs AdamW).
* **Compatibility risk:** **LOW** — NVlabs/Sana uses standard PyTorch + Accelerate + FSDP2; no custom kernels. License **Apache-2.0** — borrowable.
* **Where to wire:** `train_cfm_yuelbond_wasserstein.py:1305` (replace `optim.Adam` with `CAME_8bit`); add FSDP2 wrapping around `nn.Module`.
* **Honest-negative:** current Mol-Metal runs are single-GPU (per Wf-181/Wf-182 launchers); multi-GPU FSDP2 benefit only materializes on ≥2 GPU setup. RTX 5090 32GB likely OOMs at 5-metal batch size (need to verify).

### §2.6 Rank #6: Mini-batch OT (BoMb-OT solver replacement)

* **Implementation cost:** ~50 LOC (replace stochastic approximation in `optimal_transport_v2.py:80` re-export with `khainb/Mini-batch-OT`'s `BoMb-OT` solver).
* **PROJECTED speedup:** straighter trajectories → 1.5-2× fewer inference NFE at matched training compute (per R2 §2.6 Tong 2023); batch-size-dependent (effective at batch ≥ 32, current batch=64 is OK).
* **Compatibility risk:** **LOW** — R2 §4.2 audit confirms 5-15 LOC patch. License **MIT** (R4 §3.7).
* **Where to wire:** `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py:80`.
* **Honest-negative:** at batch=8 (which is NOT our setup) the OT coupling has limited structure and gain is small; at batch ≥ 32 (our setup) the gain materializes.

### §2.7 Rank #7: SO(3)-averaged FM loss

* **Implementation cost:** ~100 LOC (port from Cao et al. 2025 arXiv:2507.09785; replace Kabsch-aligned target with SO(3)-averaged loss).
* **PROJECTED speedup:** faster training convergence + more stable Wasserstein-gradient signal (per R3 §3.9 PROJECTED analysis).
* **Compatibility risk:** **MED** — SO(3)-averaging integrates out rotation in the loss itself; changes loss target. Need to verify `molmetal/adapters/flow_matching_lipman/__init__.py:1145` equivariant vector message handling.
* **Where to wire:** `molmetal/adapters/flow_matching_lipman/w_flow.py` (loss target computation).
* **Honest-negative:** code repo not yet verified (NVIDIA internal per R3 §3.9); needs upstream chase before implementation.

### §2.8 Rank #8: AugmentPipe for 3D SE(3) ligands

* **Implementation cost:** ~400 LOC (port `training/augment.py:19.4KB` from NVlabs/edm; adapt for 3D SE(3): random rotations + translations + ligand permutation).
* **PROJECTED speedup:** 1.2-1.5× convergence (less overfitting; per R2 §2.5 + R4 §5).
* **Compatibility risk:** **MED** — SE(3) augmentations are rotation + translation, not pixel-flip / color-jitter; not a direct port. License **CC-BY-NC-SA 4.0** (NVlabs/edm).
* **Where to wire:** `molmetal/adapters/flow_matching_lipman/__init__.py:363` (`CFMJointModel`) data-loader integration.
* **Honest-negative:** R&D-tier work; ~400 LOC is non-trivial; gain modest vs cost.

### §2.9 Rank #9: lr_rampup + EMA decay with halflife

* **Implementation cost:** ~30 LOC (linear warmup over 10k images + EMA with halflife=500 kimgs, both in 5-15 LOC each).
* **PROJECTED speedup:** 1.2× faster effective convergence (prevents early-epoch divergence + reduces overfitting).
* **Compatibility risk:** **LOW** — standard PyTorch optim.lr_scheduler pattern + EMA class. License **CC-BY-NC-SA 4.0** (NVlabs/edm).
* **Where to wire:** `train_cfm_yuelbond_wasserstein.py:1305` (optimizer creation + lr_scheduler).
* **Honest-negative:** small standalone gain; primarily useful in combination with #1 (EDM preconditioning) which also benefits from lr_rampup.

### §2.10 Rank #10: FlowMol SimplexFlow (atom-type categorical)

* **Implementation cost:** ~200 LOC (port categorical simplex flow from FlowMol `dunni3/FlowMol`; replace one-hot CE in atom-type head).
* **PROJECTED speedup:** +0.05-0.10 decode_ratio + bond-head accuracy improvement (per R3 §3.1 PROJECTED).
* **Compatibility risk:** **MED** — changes atom-type head from softmax+CE to SimplexFlow; affects YuelBond decoder integration. License: FlowMol is MIT-style (verify).
* **Where to wire:** `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py` (atom-type head).
* **Honest-negative:** orthogonal to most other accelerations; gain is on decode quality not training speed.

---

## §3 Aggregated speedup projection

| Scenario | Speedup vs current | MEASURED / PROJECTED | Source |
|----------|---------------------|----------------------|--------|
| Current state (Lipman + BF16 + OT) | 1× (baseline) | MEASURED | AGENTS.md §6 |
| + Rank #1 (EDM preconditioning) | 2-3× faster convergence | PROJECTED | Karras 2022 ablation |
| + Rank #2 (Log-normal sampling) | 1.5-2× (combined with #1) | PROJECTED | NVlabs/edm |
| + Rank #3 (torch.compile + CUDA Graphs) | 1.3-1.8× step throughput | PROJECTED | NVlabs/Sana |
| + Rank #4 (MeanFlow) | 1.5-2× training | PROJECTED | Geng 2025 |
| + Rank #5 (FSDP2 + CAME-8bit) | 1.5-2× multi-GPU | PROJECTED | NVlabs/Sana |
| **Top-5 combined** | **5-10× faster training** | PROJECTED | R4 §7.5 cross-validated |
| + Rank #6 (Mini-batch OT) | +1.5-2× inference NFE | PROJECTED | Tong 2023 |
| + Rank #7 (SO(3)-averaged FM) | +stable W₂ gradients | PROJECTED | Cao 2025 |
| + Rank #8 (AugmentPipe 3D) | +1.2-1.5× convergence | PROJECTED | NVlabs/edm |

**PROJECTED total wall-time savings for a 50-epoch CFM retrain:**
- Current: 50 epochs × 1× = 50 epoch-equivalents
- With Top-5: 50 epochs × ~0.1-0.2× = 5-10 epoch-equivalents → **40-45 epoch-equivalent savings**
- (PROJECTED; needs Mol-Metal A/B to confirm)

---

## §4 License review summary

| Source repo | License | Risk for Mol-Metal (Tier-2 paper) |
|-------------|---------|-----------------------------------|
| NVlabs/edm | CC BY-NC-SA 4.0 | **MED** — non-commercial + share-alike. Need written license exception OR find MIT equivalent. Affects #1, #2, #8, #9. |
| lucidrains/rectified-flow-pytorch | MIT | LOW — borrowable for #4 (MeanFlow) |
| NVlabs/Sana | Apache-2.0 | LOW — borrowable for #3, #5 |
| khainb/Mini-batch-OT | MIT | LOW — borrowable for #6 |
| dunni3/FlowMol | MIT-style | LOW — borrowable for #10 |
| Cao et al. 2025 (SO(3)-averaged) | (NVIDIA internal per R3 §3.9) | **HIGH** — repo not yet verified; needs upstream chase |
| facebookresearch/flow_matching | CC BY-NC 4.0 | already vendored (R17 R17 shim) |
| bkmi/flowmm | CC-BY-NC | **MED** — non-commercial |
| Yang et al. 2026 (StableVM) | TBD | TBD — needs ICML 2026 proceedings check |

**Honest-negative:** license review is a separate workflow (NOT covered in this Wf-340-A2); before adopting #1/#2/#8/#9 from NVlabs/edm, agent must surface this to user for a commercial-use decision.

---

## §5 Compatibility risk assessment

| Opportunity | Compatible with Lipman CFM? | Compatible with Wasserstein regression? | Compatible with D-MPNN backend? |
|-------------|------------------------------|------------------------------------------|----------------------------------|
| #1 EDM σ-precond | YES (drop-in wrap) | YES (loss is independent) | YES (D-MPNN already adopted) |
| #2 Log-normal t | YES (drop-in) | YES | YES |
| #3 torch.compile | YES (D-MPNN proven) | YES (D-MPNN proven) | YES (D-MPNN already adopted) |
| #4 MeanFlow | MED (untested on EGNN) | MED (loss shape changes) | n/a (D-MPNN doesn't use FM) |
| #5 FSDP2 + CAME-8bit | YES (NVlabs/Sana stack) | YES (CAME-8bit standard) | YES |
| #6 Mini-batch OT | YES (R2 §2.6 audit) | YES (OT is independent) | YES |
| #7 SO(3)-averaged | YES (drop-in loss) | MED (Kabsch-replaced) | n/a |
| #8 AugmentPipe 3D | YES (data-loader side) | YES | YES |
| #9 lr_rampup + EMA | YES (training infra) | YES | YES |
| #10 SimplexFlow | YES (atom-type head) | n/a | n/a |

**Cross-references already in our stack (per AGENTS.md §6 + Wf-333-T2 + Wf-335-R1):**
- torch.compile is shipped for D-MPNN (`train_dmpnn_multitask_wasserstein.py:163-195`) — extending to CFM is well-trodden.
- BF16 autocast is shipped — extending log-normal sampling + EDM preconditioning to BF16 is straightforward (fp32 loss + bf16 forward).
- 4 Triton fused ops are shipped (per Wf-333-T2) — additional CFM-specific kernels (e.g., fused EGNN message passing) could be added.

---

## §6 Recommended implementation order

Per speedup/cost ratio + license safety + risk minimization:

| Phase | Opportunities | Wall-budget | License risk |
|-------|---------------|-------------|--------------|
| **Phase 1 (LOW risk + low cost)** | #2 (log-normal) + #3 (torch.compile CFM) + #9 (lr_rampup + EMA) | 1-2 days | LOW |
| **Phase 2 (MED cost, HIGH gain)** | #1 (EDM preconditioning) + #6 (Mini-batch OT replacement) | 3-5 days | MED (verify CC-BY-NC-SA) |
| **Phase 3 (R&D-tier)** | #4 (MeanFlow) + #5 (FSDP2 + CAME-8bit) | 5-10 days | LOW |
| **Phase 4 (future, depending on user OK)** | #7 (SO(3)-averaged) + #8 (AugmentPipe 3D) + #10 (SimplexFlow) | 2-3 weeks | MIXED |

**Honest-negative:** Phase 2 requires commercial-use license decision from user (CC-BY-NC-SA 4.0 on NVlabs/edm). Phase 1 can launch without license gate.

---

## §7 Honest framing + risks

### §7.1 Speedup estimates are PROJECTED

* Every speedup number in §2-§3 is **PROJECTED from upstream paper benchmarks** (Karras 2022 on images, Geng 2025 on images, NVlabs/Sana on images+text).
* **NOT MEASURED on Mol-Metal data.** A 5-seed × 30/50-epoch A/B is the only honest verification.
* The aggregated 5-10× speedup assumes multiplicative composition, which is an upper bound — real wall-time savings may be lower (some optimizations share the same compute resources).

### §7.2 License is a real risk for Tier-2 paper work

* **CC-BY-NC-SA 4.0** (NVlabs/edm) is non-commercial + share-alike. If Mol-Metal is commercial / publishing for-fee, written license exception needed. If user wants pure open-source Tier-2 paper, MAY be acceptable.
* This decision is OUT OF SCOPE for Wf-340-A2 — surface to user before Phase 2 implementation.

### §7.3 R&D risk on Phase 3-4

* **MeanFlow on EGNN** (R4 §7.2) — `torch.func.jvp` may not work on equivariant net output; custom autograd.Function fallback required.
* **AugmentPipe for 3D** (R2 §7.2) — SE(3) augmentations are non-trivial; not a direct port.
* **SO(3)-averaged FM** (R3 §3.9) — repo not yet verified (NVIDIA internal); needs upstream chase.
* These R&D-tier opportunities have higher variance in their actual speedup.

### §7.4 No retrain launched

* This Wf is coverage matrix + ranking only. No GPU launches performed.
* Per AGENTS.md §5.1 standing GPU grant, Phase 1-2 CPU wire-ups can launch immediately; Phase 3-4 GPU A/B retrain launches require user OK per kill_then_adapt (and pre-flight PASS).

---

## §8 Next steps + downstream Wfs

* **Wf-340-A3 (Phase 1 CPU wire-up):** port log-normal t-sampling + torch.compile(CFM) + lr_rampup + EMA decay. ~1-2 day wall; ~35 LOC + 2 unit tests. LICENSE-FREE.
* **Wf-340-A4 (Phase 2 wire-up):** port EDM σ-preconditioning + replace OT solver with BoMb-OT. ~3-5 day wall; ~200 LOC + 3 unit tests. LICENSE GATE on NVlabs/edm.
* **Wf-340-A5 (Phase 3 R&D):** port MeanFlow + add FSDP2 + CAME-8bit. ~5-10 day wall; ~300 LOC + 4 unit tests. Requires RTX 5090 multi-GPU bench.
* **Wf-340-A6 (Phase 4 future):** SO(3)-averaged FM + AugmentPipe 3D + SimplexFlow. 2-3 week wall. LICENSE GATE + R&D risk.
* **Wf-340-A7 (verification):** 5-seed × 30-epoch A/B retrain (with vs without Top-5) → MEASURED speedup numbers. Pre-flight v2 25/25 PASS required per AGENTS.md §1 rule 4. GPU launch per §5.1 standing grant.

**Suggested launch order for the ultracode implementation wave:**
A3 → A4 → A5 → A7 (verification) → A6 (if A7 confirms ≥2× speedup).

---

**END OF Wf-340-A2** — Owner Wf-340-A2; 18 MISSING techniques identified; Top-10 ranked by speedup/cost ratio; license risk surfaced; downstream Wf-340-A3..A7 listed.
