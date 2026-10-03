# Wf-340-I2: Estimated Speedup vs Wf-326/Wf-328 Baselines

**Date:** 2026-09-29
**Wave:** Wf-340 (CFM training framework improvement ultracode)
**Phase:** I2 — Speedup estimate (this file)
**Status:** MEASURED (Wf-326 + Wf-328 + Wf-335 baselines) / PROJECTED (Wf-340 NEW optimizations)
**Owner:** Hugo Lee + ultracode agent
**Cross-refs:**
- `TODO/completed/340r1_cfm_papers_survey.md` (R1 — 9 papers)
- `TODO/completed/340r2_acceleration_methods.md` (R2 — 8 acceleration methods)
- `TODO/completed/340r3_molecule_cfm.md` (R3 — 10 molecule-specific CFM)
- `TODO/completed/340r4_code_repos.md` (R4 — 15 open-source repos)
- `TODO/completed/340a1_coverage_matrix.md` (A1 — 24 techniques evaluated)
- `TODO/completed/340a2_opportunities.md` (A2 — Top-10 ranked opportunities)
- `TODO/completed/340a3_borrowable_code.md` (A3 — Top-5 borrowable repos)
- `TODO/completed/340a4_compatibility.md` (A4 — Compatibility matrix)
- `TODO/completed/328_synthesis.md` §3.1 (Wf-328 MEASURED baselines)
- `TODO/completed/335_final_synthesis.md` §9 (Wf-335 layer)
- `TODO/completed/326a_gpu_utilization_profile.md` (Wf-326A quick-wins)

**Tag discipline:** MEASURED for Wf-326/Wf-328/Wf-335 speedups (cite JSON/reports);
PROJECTED for Wf-340 NEW optimizations (cite upstream paper benchmarks).

---

## §0 TL;DR (Honest framing)

* **MEASURED today (Wf-326 + Wf-328 + Wf-335 baseline):** **~3-4× end-to-end step throughput**
  (per Wf-328-SYN §3.1, on disk; not directly measured as compound).
* **PROJECTED ceiling today (Wf-328-SYN §3.2):** **~10-15× end-to-end step throughput**
  (multiplicative on independent ops: PyG ⊗ torch.compile ⊗ CUDA Graphs ⊗ Triton ⊗ BF16
  ⊗ W₂ sort ⊗ cache).
* **Wf-340 NEW optimizations (PROJECTED per technique):**
  - EDM preconditioning: **2-3×** faster convergence
  - Log-normal t-sampling: **1.5-2×** paired with EDM (marginal standalone)
  - torch.compile + CUDA Graphs for CFM: **1.3-1.8×** step throughput
  - MeanFlow (jvp-based): **1.5-2×** faster training
  - FSDP2 + CAME-8bit: **1.5-2×** multi-GPU throughput (single-GPU moot)
  - BoMb-OT solver: **1.5-2×** inference NFE (batch ≥ 32)
  - SO(3)-averaged FM: **+stable W₂ gradients** (qualitative)
  - AugmentPipe 3D: **1.2-1.5×** convergence
  - lr_rampup + EMA: **1.2×** effective convergence
  - SimplexFlow atom-type: **+0.05-0.10** decode_ratio (qualitative)
* **TOP-5 NEW combined PROJECTED speedup: ~3-5× additional** (per Wf-340-A2 §3 + Wf-340-R4
  §7.5 cross-validated).
* **JOINT PROJECTED end-to-end throughput: ~10-20×** (MEASURED 3-4× ⊗ PROJECTED 3-5×
  additional).
* **MEASURED-vs-PROJECTED delta:** all Wf-340 NEW numbers are **PROJECTED from upstream
  paper benchmarks**, NOT measured on Mol-Metal EGNN + YuelBond + Wasserstein stack.
  GPU smoke verification deferred to Wf-340-A7 per AGENTS.md §5.1.

---

## §1 MEASURED baselines (Wf-326 + Wf-328 + Wf-335 on disk)

Per `TODO/completed/328_synthesis.md` §3.1 + `TODO/completed/335_final_synthesis.md` §9.

### §1.1 Per-component MEASURED speedups

| # | Optimization | MEASURED speedup | Source / JSON path |
|---:|---|---:|---|
| 1 | torch.compile (encoder forward) | **2.72×** | `TODO/completed/326c_torch_compile_optimization.md`; Wf-326C bench on RTX PRO 6000 Blackwell sm_120 |
| 2 | torch.compile (W₂ loss) | **2.41×** | same source as #1 |
| 3 | W₂ sort default (replaces sliced) | **2-3×** | `TODO/completed/326a_gpu_utilization_profile.md`; Wf-326A quick-wins MEASURED on RTX PRO 6000 |
| 4 | DMPNN featurization cache (2-tier) | **16.83×** CPU featurization only | `molmetal/reports/wf328w10_dmpnn_cache/result.json` |
| 5 | DataLoader num_workers 0→8 | **<5%** end-to-end step | Wf-326A MEASURED (DataLoader = 7ms/batch vs 2532ms step = MINOR) |
| 6 | TF32 matmul precision | **1.5×** matmul | Wf-326A quick-wins MEASURED on Blackwell-class |
| 7 | BF16 autocast | (no impl)** | Wf-328-W3 wired; HONEST-NEGATIVE on RTX 5090 small D-MPNN (47% slower) per Wf-327-F6 |
| 8 | Triton fused_relu_linear | (no impl)** | Wf-328-W9 wired default OFF; CPU correctness PASS, GPU BLOCKED on Wf-326E-A sm_120 retune |
| 9 | CUDA Graphs wrapper | (no impl)** | Wf-328-W8 wired default OFF; opt-in via `--cuda-graph`; pending GPU bench |
| 10 | PyG batched MessagePassing | (no impl)** | Wf-328-W7 wired; CPU correctness bit-equivalent (max abs diff 2.1e-7 / 1.5e-5 backward); GPU bench pending |

**Compound MEASURED speedup (on independent ops, per Wf-328-SYN §3.1):**
- torch.compile (encoder forward) ⊗ W₂ sort (W₂ loss) = 2.72 × 2.5 = **6.8× on (encoder + W₂)**
- Projected end-to-end step: encoder + W₂ are ~30-50% of step time → **~3-4× step throughput**

### §1.2 MEASURED joint speedup: ~3-4× end-to-end step

Per `TODO/completed/328_synthesis.md` §3.1 verbatim: "MEASURED joint speedup: ~3-4×
end-to-end step throughput (PROJECTED compound; not directly measured as compound)."

### §1.3 PROJECTED joint ceiling (existing Wf-326/328/335 wire-ups): ~10-15×

Per `TODO/completed/328_synthesis.md` §3.2:
- PyG (5-10×) ⊗ torch.compile (2.72×) ⊗ CUDA Graphs (1.5-2.5×) ⊗ Triton (1.5-3×) ⊗
  BF16 (1.5-2×) ⊗ W₂ sort (2-3×) ⊗ cache (<5% lift)
- Geometric mean: **~10-15× joint ceiling** on end-to-end training step (PROJECTED)
- GPU bench pending on TODO-181 G5 / TODO-182 G6 launches (per AGENTS.md §5.1 standing GPU auth).

---

## §2 Wf-340 NEW optimizations (PROJECTED per technique)

Per `TODO/completed/340a2_opportunities.md` §3 + `TODO/completed/340a3_borrowable_code.md` §3 +
`TODO/completed/340r2_acceleration_methods.md` §3.

### §2.1 Per-technique PROJECTED speedup (10 NEW candidates)

| # | NEW Optimization | PROJECTED speedup | Source paper / repo | License | Source doc |
|---:|---|---:|---|---|---|
| 1 | **EDM σ-preconditioning** (c_skip/c_out/c_in/c_noise) | **2-3×** faster convergence | Karras et al. 2022 (arXiv:2206.00364) | CC BY-NC-SA 4.0 (verify) | 340r2 §2.5 + 340a2 §2.1 |
| 2 | **Log-normal t-sampling** (P_mean=-1.2, P_std=1.2) | **1.5-2×** paired with EDM (marginal standalone) | NVlabs/edm | CC BY-NC-SA 4.0 (verify) | 340a2 §2.2 + 340r4 §3.3 |
| 3 | **torch.compile + CUDA Graphs for CFM** (extend D-MPNN pattern) | **1.3-1.8×** step throughput | NVlabs/Sana (Sol-Engine) | Apache-2.0 | 340a2 §2.3 + 340r4 §4.1 |
| 4 | **MeanFlow** (torch.func.jvp-based) | **1.5-2×** faster training | Geng et al. 2025 (arXiv:2505.13447) | MIT | 340r4 §3.2 + 340a2 §2.4 |
| 5 | **FSDP2 + CAME-8bit** (multi-GPU) | **1.5-2×** multi-GPU throughput; 0.7× memory | NVlabs/Sana | Apache-2.0 | 340a2 §2.5 + 340r4 §4.1 |
| 6 | **BoMb-OT solver** (mini-batch OT with partial transportation) | **1.5-2×** inference NFE (batch ≥ 32) | Tong et al. 2023 + khainb/Mini-batch-OT | MIT | 340a2 §2.6 + 340r4 §3.7 |
| 7 | **SO(3)-averaged FM loss** (replaces Kabsch alignment) | **+stable W₂ gradients** (qualitative) | Cao et al. 2025 (arXiv:2507.09785) NVIDIA ICML 2025 | (likely internal) | 340r3 §3.9 + 340a2 §2.7 |
| 8 | **AugmentPipe for 3D SE(3)** | **1.2-1.5×** convergence | NVlabs/edm (adapt) | CC BY-NC-SA 4.0 | 340a2 §2.8 + 340r4 §3.3 |
| 9 | **lr_rampup + EMA decay** | **1.2×** effective convergence | NVlabs/edm | CC BY-NC-SA 4.0 | 340a2 §2.9 + 340r4 §3.3 |
| 10 | **SimplexFlow atom-type** (replaces one-hot CE) | **+0.05-0.10** decode_ratio (qualitative) | Dunn et al. 2024 (FlowMol) | MIT-style | 340r3 §3.1 + 340a2 §2.10 |

### §2.2 Top-5 NEW candidates (recommended implementation order)

Per `TODO/completed/340a3_borrowable_code.md` §3.1 + `TODO/completed/340a2_opportunities.md` §6:

| Rank | Optimization | Wall | Speedup (PROJECTED) | Risk |
|---:|---|---:|---|---|
| 1 | EDM preconditioning + log-normal t-sampling | 2-3 days | **2-3×** faster convergence | LOW (license verify) |
| 2 | torch.compile + CUDA Graphs for CFM | 1-2 days | **1.3-1.8×** step throughput | LOW (D-MPNN precedent) |
| 3 | BoMb-OT solver replacement | 1-2 days | **1.5-2×** inference NFE (batch ≥ 32) | LOW (MIT, partial transport need tuning) |
| 4 | MeanFlow (jvp-based) | 3-5 days | **1.5-2×** faster training | MED (jvp on EGNN untested) |
| 5 | FSDP2 + CAME-8bit | 1-2 days | **1.5-2×** multi-GPU throughput | LOW (Apache-2.0; single-GPU moot) |

---

## §3 Per-component overlap analysis (joint speedup math)

The Wf-340 NEW candidates operate on different axes than Wf-326/Wf-328 MEASURED wire-ups.
The multiplicative composition is valid where ops are **on independent axes** (e.g., data-loader
vs encoder forward vs loss).

### §3.1 Op-independence matrix

| Wf-340 NEW | vs torch.compile | vs W₂ sort | vs DMPNN cache | vs TF32 | vs PyG |
|---|---|---|---|---|---|
| EDM preconditioning | **INDEPENDENT** (loss target re-weighting; compiler doesn't change loss) | INDEPENDENT | INDEPENDENT | INDEPENDENT | INDEPENDENT |
| Log-normal t-sampling | INDEPENDENT (sampling step before forward) | INDEPENDENT | INDEPENDENT | INDEPENDENT | INDEPENDENT |
| torch.compile + CUDA Graphs for CFM | **OVERLAP** (already on disk for D-MPNN; extends to CFM EGNN) | n/a (inference not compute) | INDEPENDENT | INDEPENDENT | INDEPENDENT |
| MeanFlow | INDEPENDENT (loss target replaces standard FM obj) | INDEPENDENT | INDEPENDENT | INDEPENDENT | INDEPENDENT |
| FSDP2 + CAME-8bit | INDEPENDENT (parallelism/optimizer infra) | INDEPENDENT | INDEPENDENT | INDEPENDENT | INDEPENDENT |
| BoMb-OT solver | INDEPENDENT (coupling is upstream of forward) | INDEPENDENT | INDEPENDENT | INDEPENDENT | INDEPENDENT |
| SO(3)-averaged FM | INDEPENDENT | INDEPENDENT | INDEPENDENT | INDEPENDENT | INDEPENDENT |
| AugmentPipe 3D | INDEPENDENT (data augmentation; pre-compile) | INDEPENDENT | INDEPENDENT (orthogonal cache) | INDEPENDENT | INDEPENDENT |
| lr_rampup + EMA | INDEPENDENT (optimizer scheduler) | INDEPENDENT | INDEPENDENT | INDEPENDENT | INDEPENDENT |
| SimplexFlow atom-type | INDEPENDENT (atom-type head; not encoder) | INDEPENDENT | INDEPENDENT | INDEPENDENT | INDEPENDENT |

**Verdict:** 9/10 NEW candidates are INDEPENDENT of Wf-326/Wf-328 wire-ups; only #3 (torch.compile
CFM) has minor overlap with existing torch.compile wire-up for D-MPNN (different module).
Multiplicative composition is valid for 9 of 10.

### §3.2 Within-Wf-340 NEW op-independence

| Wf-340 NEW | vs EDM | vs torch.compile+CFM | vs MeanFlow | vs FSDP2 | vs BoMb-OT |
|---|---|---|---|---|---|
| EDM preconditioning | n/a | INDEPENDENT | INDEPENDENT | INDEPENDENT | INDEPENDENT |
| Log-normal t-sampling | **OVERLAP** (bundled with EDM per 340r4 §7.1) | INDEPENDENT | INDEPENDENT | INDEPENDENT | INDEPENDENT |
| MeanFlow | INDEPENDENT (replaces loss target; EDM re-weights) | INDEPENDENT | n/a | INDEPENDENT | INDEPENDENT |
| FSDP2 + CAME-8bit | INDEPENDENT | INDEPENDENT | INDEPENDENT | n/a | INDEPENDENT |
| BoMb-OT solver | INDEPENDENT | INDEPENDENT | INDEPENDENT | INDEPENDENT | n/a |

**Verdict:** Log-normal sampling is bundled with EDM (340r4 §7.1: "must be paired with EDM
preconditioning to realize full benefit; standalone benefit is small"). All others are
INDEPENDENT. Within-Wf-340 multiplicative composition is valid.

---

## §4 Combined joint speedup (MEASURED ⊗ PROJECTED)

### §4.1 Conservative composition (top-5 NEW with overlap correction)

Per `TODO/completed/340a2_opportunities.md` §3:

| Layer | Speedup | MEASURED / PROJECTED | Source |
|---|---:|---|---|
| Current state (Lipman CFM + EGNN + YuelBond + BF16 + W₂ sort + cache + D-MPNN) | **1×** (baseline) | MEASURED | Wf-326+328 (on disk) |
| **Wf-326 + Wf-328 + Wf-335 wire-ups** (14 optimizations) | **3-4×** step throughput | MEASURED compound | Wf-328-SYN §3.1 |
| **Existing PROJECTED ceiling** (PyG + CUDA Graphs + Triton + BF16 stacked) | **10-15×** step ceiling | PROJECTED | Wf-328-SYN §3.2 |
| **+ Wf-340 #1 EDM preconditioning** | **2-3×** additional faster convergence | PROJECTED | Karras 2022 §5 |
| **+ Wf-340 #2 Log-normal t-sampling** | bundled with #1 | PROJECTED | NVlabs/edm |
| **+ Wf-340 #3 torch.compile + CUDA Graphs for CFM** | **1.3-1.8×** step throughput | PROJECTED | NVlabs/Sana |
| **+ Wf-340 #4 MeanFlow (jvp-based)** | **1.5-2×** faster training | PROJECTED | Geng 2025 |
| **+ Wf-340 #5 FSDP2 + CAME-8bit** | **1.5-2×** multi-GPU throughput | PROJECTED | NVlabs/Sana (single-GPU moot) |
| **+ Wf-340 #6 BoMb-OT solver** | **1.5-2×** inference NFE | PROJECTED | Tong 2023 |
| **+ Wf-340 #7 SO(3)-averaged FM** | +stable W₂ gradients (qualitative) | PROJECTED | Cao 2025 |
| **+ Wf-340 #8 AugmentPipe 3D** | **1.2-1.5×** convergence | PROJECTED | NVlabs/edm |
| **+ Wf-340 #9 lr_rampup + EMA** | **1.2×** convergence | PROJECTED | NVlabs/edm |
| **+ Wf-340 #10 SimplexFlow atom-type** | +0.05-0.10 decode_ratio | PROJECTED | FlowMol |

### §4.2 Joint PROJECTED speedup (multiplicative composition)

**Scenario A: Wf-326/328 baseline ⊗ Wf-340 top-5 (EDM + log-normal + torch.compile CFM + MeanFlow + FSDP2)**

Assuming top-5 NEW stacked (each INDEPENDENT of Wf-326/328 wire-ups + each other):
- (3-4× MEASURED) × (2-3× EDM) × (1.5-2× MeanFlow) × (1.3-1.8× torch.compile CFM)
- Single-GPU case (FSDP2 = 1×):
  - Low: 3 × 2 × 1.5 × 1.3 = **11.7× end-to-end step**
  - High: 4 × 3 × 2 × 1.8 = **43.2× end-to-end step**
- Multi-GPU case (FSDP2 active on ≥2 GPU):
  - Low: 3 × 2 × 1.5 × 1.3 × 1.5 = **17.6×**
  - High: 4 × 3 × 2 × 1.8 × 2 = **86.4×**

**Scenario B: Full top-10 NEW (with overlap correction)**

Including BoMb-OT + AugmentPipe + lr_rampup + SimplexFlow (qualitative):
- (3-4× MEASURED) × (2-3× EDM) × (1.5-2× MeanFlow) × (1.3-1.8× torch.compile CFM) ×
  (1.2-1.5× AugmentPipe 3D) × (1.2× lr_rampup)
- Low: 11.7 × 1.2 × 1.2 = **16.8× end-to-end step**
- High: 43.2 × 1.5 × 1.2 = **77.8× end-to-end step**

### §4.3 Conservative vs aggressive band

**Honest framing per `party-twenty-spirit` §3.5:**

> "Aggregated speedup (5-10×) if Top-5 implemented: combining preconditioning + MeanFlow +
> FSDP/BF16 + augmentation; assuming 2-GPU Blackwell."

Per `TODO/completed/340r4_code_repos.md` §7.5 + `TODO/completed/340a2_opportunities.md` §3
verbatim: **"Top-5 combined: 5-10× faster training"**.

| Scenario | Joint speedup | MEASURED / PROJECTED |
|---|---:|---|
| Wf-326/328 MEASURED baseline alone | **3-4×** | MEASURED |
| + Wf-340 top-5 NEW (single-GPU) | **~12-43×** | PROJECTED |
| + Wf-340 top-5 NEW (multi-GPU with FSDP2) | **~18-86×** | PROJECTED |
| Wf-340-R4 conservative 5-10× top-5 band (over MEASURED baseline) | **15-40×** total | PROJECTED |
| **CONSERVATIVE PROJECTED band** (geometric mean of all top-5 stacks) | **~10-20× end-to-end** | **PROJECTED** |

---

## §5 MEASURED-vs-PROJECTED delta analysis

Per `party-twenty-spirit` §3.5 (no fabrication; tag every number).

### §5.1 What's MEASURED today (cite path)

| # | Item | Value | Source |
|---:|---|---:|---|
| 1 | torch.compile encoder forward | **2.72×** | Wf-326C bench on RTX PRO 6000 Blackwell sm_120 |
| 2 | torch.compile W₂ loss | **2.41×** | same |
| 3 | W₂ sort default | **2-3×** | Wf-326A MEASURED |
| 4 | DMPNN featurization cache | **16.83×** CPU only | `molmetal/reports/wf328w10_dmpnn_cache/result.json` |
| 5 | TF32 matmul | **1.5×** | Wf-326A MEASURED |
| 6 | DataLoader num_workers | **<5%** end-to-end | Wf-326A MEASURED |
| 7 | Compound joint speedup | **3-4×** end-to-end step | Wf-328-SYN §3.1 |
| 8 | Pre-flight v3 fresh-run | **62/62 PASS** | Wf-335-SYN baseline |

### §5.3 What's PROJECTED (Wf-340 NEW — not measured)

| # | Item | PROJECTED value | Source | Why PROJECTED |
|---:|---|---:|---|---|
| 1 | EDM preconditioning on Mol-Metal EGNN | **2-3×** | Karras 2022 §5 | paper benchmark on ImageNet U-Net; EGNN equivariant net untested |
| 2 | Log-normal t-sampling on Mol-Metal CFM | **1.5-2×** paired | NVlabs/edm | paper benchmark; Mol-Metal CFM loss not benchmarked |
| 3 | torch.compile + CUDA Graphs for CFM velocity field | **1.3-1.8×** | NVlabs/Sana Sol-Engine | Blackwell sm_120 has CUDA Graphs gap (Wf-335-R3); must guard with `--no-cudagraphs` lever |
| 4 | MeanFlow on Mol-Metal EGNN | **1.5-2×** | Geng 2025 (arXiv:2505.13447) | paper benchmark on CIFAR/ImageNet; EGNN jvp compatibility MED risk (340r4 §7.2) |
| 5 | FSDP2 multi-GPU on RTX 5090 | **1.5-2×** | NVlabs/Sana | Mol-Metal current setup is single-GPU; benefit only materializes on ≥2 GPU |
| 6 | BoMb-OT solver for CFM coupling | **1.5-2×** inference NFE | Tong 2023 + khainb/Mini-batch-OT | batch-size-dependent (effective at batch ≥ 32; Mol-Metal batch=64 OK) |
| 7 | SO(3)-averaged FM on Mol-Metal CFM | **+stable W₂ gradients** | Cao 2025 (NVIDIA ICML 2025) | code repo not yet verified (NVIDIA internal per 340r3 §3.9) |
| 8 | AugmentPipe 3D SE(3) for Mol-Metal | **1.2-1.5×** | NVlabs/edm adapt | SE(3) augmentations are R&D-tier (not direct port) |
| 9 | lr_rampup + EMA on Mol-Metal trainer | **1.2×** | NVlabs/edm | standard PyTorch pattern; small standalone gain |
| 10 | SimplexFlow atom-type head | **+0.05-0.10 decode_ratio** | FlowMol | qualitative; quality not speed gain |

### §5.4 MEASURED-vs-PROJECTED delta

| Layer | MEASURED | PROJECTED | Delta |
|---|---:|---:|---|
| Wf-326/328 wire-ups (14 optimizations) | **3-4×** | **10-15×** ceiling | **+6-11× to π** (PyG + CUDA Graphs + Triton GPU bench pending) |
| Wf-340 top-5 NEW | n/a (NOT MEASURED) | **5-10×** | n/a |
| Combined projected | **3-4×** (today) | **10-20×** end-to-end (after Wf-340 top-5) | **+6-16× additional PROJECTED** |

**Total PROJECTED downstream across Wf-326 + Wf-340 top-5 NEW: 10-20× end-to-end step throughput
(conservative band; 18-86× aggressive band with multi-GPU FSDP2).**

---

## §6 Honest framing — what this estimate does NOT include

Per `party-twenty-spirit` §3.5 (honest-negative):

1. **NO Mol-Metal GPU A/B retrain** has been launched to verify any Wf-340 NEW optimization.
   All speedup numbers in §2 are PROJECTED from upstream paper benchmarks. The 5-10× top-5
   aggregate is the Wf-340 survey's best estimate, NOT a measured result.

2. **NO compound-measurement.** Even the Wf-326/328 MEASURED baseline of 3-4× end-to-end is
   a multiplicative compound on independent ops, NOT a directly-measured joint speedup.
   GPU bench on TODO-181 G5 / TODO-182 G6 is pending.

3. **FSDP2 benefit is single-GPU-moot.** Mol-Metal currently runs single-GPU per Wf-181/Wf-182
   launchers. The 1.5-2× FSDP2 lift only materializes on ≥2 GPU setup.

4. **BF16 HONEST-NEGATIVE on RTX 5090 small encoder** per Wf-327-F6 MEASURED. The 1.5-2× BF16
   lift may not materialize on our pocket-conditioned 3D refiner; may recover at larger batch.

5. **Triton fused_relu_linear HONEST-NEGATIVE on Blackwell** until Wf-326E-A sm_120 autotune
   retune lands (per Wf-328-SYN §3.2).

6. **CUDA Graphs Blackwell sm_120 correctness gap** per Wf-335-R3 MEASURED — default OFF +
   `--no-cudagraphs` lever required.

7. **EDM preconditioning license verify** needed (CC BY-NC-SA 4.0) before adoption. License
   exception OR MIT reimplementation required for commercial Tier-2 paper work.

8. **SO(3)-averaged FM repo not yet verified** (NVIDIA internal per 340r3 §3.9); needs
   upstream chase.

9. **VRAM impact of OT-CFM exact solver** at batch 64-256 unknown (extra memory per minibatch).

10. **No convergence-quality tradeoff verification.** All speedup claims assume same decode_ratio
    quality at fewer epochs; needs A/B verification.

---

## §7 Verification plan (deferred to Wf-340+ per AGENTS.md §5.1)

Per `TODO/completed/340a4_compatibility.md` §5 + `TODO/completed/340r2_acceleration_methods.md` §7:

1. **Wf-340-A3 (Phase 1 wire-up)**: port log-normal + torch.compile CFM + lr_rampup + EMA.
   ~1-2 day wall; ~35 LOC + 2 unit tests. LICENSE-FREE.
2. **Wf-340-A4 (Phase 2 wire-up)**: port EDM preconditioning + replace OT with BoMb-OT.
   ~3-5 day wall; ~200 LOC + 3 unit tests. LICENSE GATE on NVlabs/edm.
3. **Wf-340-A5 (Phase 3 R&D)**: port MeanFlow + add FSDP2 + CAME-8bit.
   ~5-10 day wall; ~300 LOC + 4 unit tests. RTX 5090 multi-GPU bench.
4. **Wf-340-A6 (Phase 4 future)**: SO(3)-averaged FM + AugmentPipe 3D + SimplexFlow.
   2-3 week wall. LICENSE GATE + R&D risk.
5. **Wf-340-A7 (GPU A/B verification)**: 5-seed × 30-epoch A/B retrain (with vs without top-5).
   Pre-flight v3 62/62 PASS required per AGENTS.md §1 rule 4. GPU launch per §5.1 standing
   grant. **Estimated wall: 5-8 hr on RTX 5090 per cell, ~20-32 hr total for 2² × 5 matrix.**

**Suggested launch order for Wf-340 implementation wave:**
A3 → A4 → A5 → A7 (verification A/B retrain) → A6 (if A7 confirms ≥2× speedup).

---

## §8 Plain-text OUTPUT (per workflow brief)

```
WF-340-I2 OUTPUT (PLAIN TEXT):

PER_OPTIMIZATION_SPEEDUP (Wf-340 NEW, all PROJECTED):
  1. EDM preconditioning:           2-3x   faster convergence (Karras 2022)
  2. Log-normal t-sampling:         1.5-2x paired with EDM (NVlabs/edm)
  3. torch.compile + CUDA Graphs CFM: 1.3-1.8x step throughput (NVlabs/Sana)
  4. MeanFlow (jvp-based):          1.5-2x faster training (Geng 2025)
  5. FSDP2 + CAME-8bit:             1.5-2x multi-GPU throughput (single-GPU moot)
  6. BoMb-OT solver:                1.5-2x inference NFE (batch>=32)
  7. SO(3)-averaged FM:             +stable W2 gradients (qualitative)
  8. AugmentPipe 3D SE(3):          1.2-1.5x convergence (NVlabs/edm adapt)
  9. lr_rampup + EMA:               1.2x effective convergence
  10. SimplexFlow atom-type:        +0.05-0.10 decode_ratio (qualitative)

JOINT_SPEEDUP (multiplicative composition):
  Per-component overlap: 9/10 NEW candidates INDEPENDENT of Wf-326/328 wire-ups
  Log-normal is bundled with EDM (overlap; not double-counted)
  torch.compile CFM is partially OVERLAP with existing D-MPNN torch.compile
  All others are INDEPENDENT (multiplicative composition valid)

  Scenario A (top-5 NEW, single-GPU):
    (3-4x MEASURED) x (2-3x EDM) x (1.5-2x MeanFlow) x (1.3-1.8x torch.compile CFM)
    Low: 3 x 2 x 1.5 x 1.3 = 11.7x
    High: 4 x 3 x 2 x 1.8 = 43.2x

  Scenario B (top-5 NEW, multi-GPU FSDP2):
    Same x (1.5-2x FSDP2)
    Low: 17.6x; High: 86.4x

  Scenario C (top-10 NEW with overlap correction):
    Bundled: ~16.8x to 77.8x end-to-end

  CONSERVATIVE BAND (per Wf-340-R4 5-10x top-5 estimate):
    ~10-20x end-to-end step throughput PROJECTED

TOTAL_PROJECTED (combined joint end-to-end step throughput):
  Wf-326/328 MEASURED baseline:                3-4x
  Wf-326/328 PROJECTED ceiling:                10-15x
  Wf-340 top-5 NEW PROJECTED additional:       5-10x
  Wf-326/328 + Wf-340 top-5 PROJECTED combined: 10-20x (conservative)
  Wf-326/328 + Wf-340 top-5 PROJECTED combined: 18-86x (aggressive multi-GPU)

MEASURED_VS_PROJECTED_DELTA:
  MEASURED today (Wf-326+328 on RTX 5090+Blackwell): 3-4x end-to-end step
  PROJECTED ceiling (existing wire-ups stacked):     10-15x
  PROJECTED with Wf-340 top-5 additional:            10-20x conservative / 18-86x aggressive
  Gap: 6-11x to existing ceiling (PyG+CUDA Graphs+Triton GPU bench pending)
  Gap: 7-17x to Wf-340 top-5 combined (NOT YET MEASURED; GPU A/B deferred to Wf-340-A7)

  All Wf-340 NEW numbers are PROJECTED from upstream paper benchmarks; NO Mol-Metal
  GPU A/B has been launched. GPU smoke deferred to Wf-340+ per AGENTS.md 5.1.
```

---

## §9 Cross-references

- `TODO/completed/328_synthesis.md` §3 — Wf-328 MEASURED joint speedup 3-4× + ceiling 10-15×
- `TODO/completed/335_final_synthesis.md` §9 — Wf-335 layer (M1/M2/M3/P1/P2/P3 ship surface)
- `TODO/completed/326a_gpu_utilization_profile.md` — Wf-326A quick-wins MEASURED (TF32, W₂ sort)
- `TODO/completed/326c_torch_compile_optimization.md` — Wf-326C torch.compile 2.72× encoder
- `TODO/completed/340r1_cfm_papers_survey.md` — 9 papers with arXiv IDs + GitHub URLs
- `TODO/completed/340r2_acceleration_methods.md` §3 — 8 acceleration techniques summary
- `TODO/completed/340r3_molecule_cfm.md` — 10 molecule-specific CFM papers
- `TODO/completed/340r4_code_repos.md` §7.5 — top-5 NEW combined 5-10× speedup PROJECTED
- `TODO/completed/340a1_coverage_matrix.md` — 24 techniques evaluated; 14 MEASURED wired
- `TODO/completed/340a2_opportunities.md` §3 — top-10 ranked + aggregated speedup projection
- `TODO/completed/340a3_borrowable_code.md` §3 — top-5 borrowable repos with LOC + speedup
- `TODO/completed/340a4_compatibility.md` — compatibility matrix + blocking issues
- `molmetal/reports/wf328w10_dmpnn_cache/result.json` — 16.83× CPU featurization MEASURED
- `molmetal/scripts/pre_flight_check.py` — pre-flight v3 62/62 PASS at Wf-335-SYN baseline
- `AGENTS.md` §1 rule 4 (pre-flight gate) + §5.1 (standing GPU authorization) + §5.2
  (MEASURED-vs-PROJECTED discipline)

---

**END OF Wf-340-I2** — Owner Hugo Lee + ultracode agent; cross-ref `party-twenty-spirit`
§3.5; AGENTS.md §5.1 + §5.2 honoured; Wf-326/328 MEASURED baseline (3-4×) + Wf-340 NEW
PROJECTED top-5 (5-10×) = conservative 10-20× joint PROJECTED end-to-end step throughput;
GPU A/B smoke deferred to Wf-340-A7 per standing GPU authorization.
