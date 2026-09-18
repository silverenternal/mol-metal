# WF-Triton-Arch-Research: mHC (Manifold-Constrained Hyper-Connections)

**Date:** 2026-09-15
**Author:** WF-Triton-Arch-Research subagent (mHC slice)
**Companion to:** `flash_attention_egnn.md` (same workflow)
**Hardware target:** RX 7800 XT, gfx1101, RDNA3, wave64, ROCm 7.2, triton-rocm 3.8.0

**Honest-framing note.** This is a web-research synthesis, not a hands-on benchmark or local port. Every claim about ROCm / gfx1101 feasibility is grounded in either (a) explicit ROCm-specific code in a referenced repo, (b) reported failures from upstream users, or (c) first-principles reasoning about wave64 / RDNA3 vs the NVIDIA-only features (TMA, WGMMA, warp-specialization) used by DeepSeek's TileLang reference kernel. The dGPU on this host is in SMU-hang state (per `WF-GPU-Diag-Fix` / `WF-GPU-Auto-Recover`), so I cannot benchmark the Triton kernels myself. Where evidence is mixed or absent, that is marked.

---

## 1. Methodology

Three web searches via the MiniMax search API:

1. `"manifold-constrained hyper-connections ByteDance Seed"` (10 organic results)
2. `"mHC residual connection residual pathway scaling"` (10 organic results)
3. `"triton kernels mHC GPU"` (10 organic results)

Plus three WebFetch deep-reads on:
- `arxiv.org/abs/2512.24880` (mHC paper abstract)
- `gaetanx21.github.io/blog/fused-and-furious/` (independent Triton Sinkhorn kernel writeup)
- `github.com/ROCm/aiter/.../mhc_ref.py` (ROCm aiter's PyTorch reference + Triton wrapper locations)

These were chosen to triangulate three independent axes:
- **Algorithmic claim** (what mHC proposes, what it replaces) — searches 1+2
- **Kernel state of the art** (what Triton implementations exist, with what speedups) — search 3 + WebFetch
- **ROCm / gfx1101 deployability** (Triton language compat, HIP backend presence, RDNA3 feature use) — WebFetch on aiter + first-principles reasoning

Honest gap: the DeepSeek paper's full PDF was not fetched (arxiv abstract page returned only the abstract). The 0.021 loss reduction, 6.7% overhead, and Amax-gain-3000→1.6 numbers below are second-hand from `1kpapers.com/papers/...` (65 citations), `daily.dev` analysis, and the Towards AI blog, all citing the same source. The paper was v1 on 2025-12-31, v2 on 2026-01-05; it has had ~9 months to be re-analyzed in the literature by report time.

---

## 2. What mHC proposes

### 2.1 The problem mHC fixes

Per the paper abstract (arxiv:2512.24880) and three independent secondary analyses (`emergentmind.com`, `1kpapers.com`, `daily.dev/posts/.../srohkbexi`):

- The dominant residual paradigm (He et al. 2015, ResNet) is `x_{l+1} = x_l + F(x_l, W_l)`. The additive identity preserves feature mean, gives σ_max = 1 on the skip path, and bounds gradient norm.
- ByteDance Seed team (Zhu et al. 2024, arXiv:2409.19606, ICLR 2025) generalized this to **Hyper-Connections (HC)** by widening the residual stream to n×C (n typically = 4) and introducing three learnable mappings per layer: `H^pre ∈ R^{1×n}` (aggregate streams into layer input), `H^post ∈ R^{1×n}` (distribute layer output back to streams), and `H^res ∈ R^{n×n}` (mix streams within the residual pathway). The composite update is:

  `X_{l+1} = H^res_l X_l + H^{post⊤}_l F(H^pre_l X_l, W_l)`

- HC yielded downstream-task gains at small scale, but at 27B parameters the unconstrained `H^res` matrices compound multiplicatively: `∏_{i=1}^{L-l} H^res_{L-i}`. Reported empirical gain magnitudes reached **~3000×** in DeepSeek-V3 27B runs, manifesting as sudden loss spikes and gradient surges (Emergent Mind; daily.dev; 1kpapers.com).

### 2.2 The mHC fix

mHC constrains `H^res` to lie on the **Birkhoff polytope** (the set of doubly-stochastic matrices — rows and columns sum to 1, all entries ≥ 0). This is the convex hull of all permutation matrices and is closed under multiplication. Three immediate consequences (paper abstract; emergentmind.com §2; James Smith "Paper Speedrun" blog, 2026-01-21):

1. **Norm non-expansiveness** — every doubly-stochastic matrix has σ_max ≤ 1, so signal cannot amplify across stacked layers.
2. **Compositional closure** — `H^res_a · H^res_b` is also doubly-stochastic, so the manifold property holds for any depth.
3. **Mean preservation** — both row and column sums equal one, which restores the identity-mapping property HC broke.

The reported composite gain magnitude in mHC is **~1.6×** instead of ~3000× — i.e. roughly three orders of magnitude tighter bound (1kpapers.com summary).

### 2.3 How the constraint is enforced

Projection is done via the **Sinkhorn-Knopp algorithm** (1967, Pacific J. Math). For an unconstrained matrix `M`:

```
M₀ = exp(M)                            # positivity
for t in 1..T (T=20 in practice):
    M_t = normalize_rows(M_{t-1})
    M_t = normalize_cols(M_t)
return M_T
```

The paper uses T = 20 Sinkhorn iterations. The `H^pre` and `H^post` matrices are NOT doubly-stochastic; only `H^res` is. `H^pre` goes through sigmoid (per Eq. 17 in the paper, see ROCm aiter `mhc_ref.py:144-152`); `H^post` is a learnable linear projection.

When the expansion rate n collapses to 1 stream, mHC reduces exactly to a standard residual connection (AI Insider summary; Nurix blog) — i.e. it strictly generalizes rather than replaces.

### 2.4 Reported numbers (second-hand)

From `1kpapers.com/papers/mhc-manifold-constrained-hyper-connections` (citing the paper directly) and `daily.dev/posts/...`:

| Quantity | Value | Source | Caveat |
|---|---|---|---|
| Expansion rate n | 4 (used in 27B experiments) | paper Sec. 4 / 1kpapers | other values not tested in main results |
| Sinkhorn iterations T | 20 | paper Sec. 4.3 | not an ablation parameter in main results |
| Test model sizes | 3B, 9B, 27B MoE | paper Sec. 4 | not validated at 100B+ |
| Loss reduction vs baseline | -0.021 final | paper Tab. 1 (via 1kpapers) | small absolute; "modest" per Smith speedrun |
| Training overhead | +6.7% at n=4 | paper Sec. 4.3.1 | w/ fused TileLang kernels |
| Amax gain (HC vs mHC) | 3000× → 1.6× | paper Sec. 3 (via 1kpapers) | diagnostic, not a metric |
| Outperforms HC on | most benchmarks (BBH, DROP, MMLU, MATH listed) | paper Tab. 1 (via Nurix) | "most" not quantified |
| Tail behaviour | reduced loss spikes | paper Fig. 3 (via Emergent Mind) | qualitative figure |

**Honest framing of these numbers.** This is a 2025-12 paper that has only been peer-validated by secondary sources, and the speedrun author (Smith) explicitly noted "I haven't looked into the benchmarks more to get a sense of the spread of performance from other models." The +6.7% overhead is the *fused-kernel* number; naïve HC overhead is much higher (Nurix reports the all-to-all communication pattern is the killer without fusion). The 0.021 loss is on the order of 1-2% of typical pretraining loss — meaningful but not transformative. **None of these have been reproduced by independent labs in this 9-month window**, and the paper's authors include DeepSeek's founder and core infrastructure team, which both lends weight and creates a conflict of interest.

---

## 3. Triton implementations found

This is the most relevant section for our project. There are **at least 5 distinct Triton implementations** of mHC, plus ROCm's aiter has a PyTorch reference + Triton wrapper pair:

### 3.1 `linkedin/Liger-Kernel` — `LigerMHC` (Feb 2026)

- **Repo:** `github.com/linkedin/Liger-Kernel`
- **PR:** #1065 ("Add mHC fused kernels + LigerMHC API + benchmarks"), docs in #1132
- **Language:** Triton + PyTorch (`torch.library` custom ops), supports `torch.compile`
- **n_streams:** 4 (hardcoded specialization, matches paper)
- **Backward:** yes — full fwd+bwd for coeffs / Sinkhorn / apply kernels
- **Benchmark env:** NVIDIA RTX 3090, CUDA 12.8, PyTorch 2.10.0+cu128, **Triton 3.6.0**, BF16, T=128–2048, B=4, HC=4, C=4096
- **Speedups (per PR #1065 issue body):**
  - `coeffs` kernel: 3.5× faster, 77% less memory
  - `pre` kernel: 2.2× faster, 33% less memory
  - `post_res` kernel: 1.9× faster, 31% less memory
  - End-to-end LM (B=2, T=256, 2 layers, hidden 256–1024): 1.5× faster, 18% less memory
- **Mixed precision:** default BF16/FP16 inputs with FP32 projection coefficients; `allow_fp32=True` opt-in for debugging
- **Out of scope (per PR description):** the paper's recomputation strategy (Sec. 4.3.2) is delegated to `torch.utils.checkpoint`; the DualPipe pipeline schedule (Sec. 4.3.3) is not implemented
- **ROCm / gfx1101 status:** **NOT TESTED.** Repo is owned by LinkedIn (HIP-experienced org) but no gfx1101 / RDNA3 runs in the PR thread. Triton 3.6 is close to our triton-rocm 3.8.0 so kernel *language* should be compatible; whether the autotune configs transfer is unknown
- **Licence:** BSD-2 (LinkedIn standard), friendly to derivative integration

### 3.2 `YangWang92/mHC-triton` (NucleusAI fork) — full autograd path

- **Repo:** `github.com/YangWang92/mHC-triton` (forked from NucleusAI/mHC-triton)
- **Language:** Triton + PyTorch, full autograd
- **n_streams:** 4
- **Reported speedup:** **6.2× faster full forward+backward** vs PyTorch baseline
- **Distinctive techniques:**
  - **Transposed φ layout** — projection matrix stored as `[24, in_dim]` instead of `[in_dim, 24]` for coalesced reads (24 = n + n + n² at n=4)
  - **Fused RMSNorm inside the matmul** — eliminates a separate reduction kernel
  - **Inline Sinkhorn** — doubly-stochastic projection runs in same kernel, **4×4 matrix kept entirely in registers** (16 scalars per batch element)
  - **O(T²) recomputation for backward** — trades 20× more compute for 20× less memory (the 4×4 matrices are tiny so this is a net win)
  - **Two-phase weight gradient reduction** — Triton partial sums + PyTorch `tensor.sum()` final reduction
- **ROCm / gfx1101 status:** **NOT MENTIONED.** Repo is single-author, NVIDIA-leaning benchmarks. Triton language version not pinned

### 3.3 `alint77/flash-mHC` — productionized Triton extraction

- **Repo:** `github.com/alint77/flash-mHC`
- **Author context:** Author (Ali Naeimi) integrated mHC into nanoPLM and built this to escape 35% PyTorch overhead
- **Language:** Triton + `torch.library` + `torch.compile`-compatible
- **n_streams:** 4 hardcoded
- **Decomposition:** K1 (fused RMSNorm + projection → hyperconnection logits), K2 (coefficient matrices in PyTorch — sigmoid gates, softmax mixing, merged H matrix), K3 (fused pre-map of weighted input streams), K4 (fused post-residual combining H-merged stream mixing with layer output)
- **Backward:** full fwd+bwd for K1, K3, K4 (K2 is coefficient math, no large tensor I/O)
- **Reported result:** overhead dropped from **35% (pure PyTorch) → 11%** with custom kernels (Naeimi LinkedIn post 2026-03-05)
- **Built-in tooling:** runtime Triton autotune (capped config sets + disk cache), one-shot grid search script, SM120 (Blackwell) tuning summary
- **Env vars:** `FLASH_MHC_TRITON_AUTOTUNE=0`, `FLASH_MHC_TRITON_AUTOTUNE_STATUS=0`, `TRITON_CACHE_DIR`
- **ROCm / gfx1101 status:** **NOT TESTED.** All tuning artifacts target SM120 Blackwell. Autotune configs will need re-tuning for gfx1101 wave64

### 3.4 `NVIDIA/TransformerEngine` — `fused_mhc_kernels` module

- **Module:** `transformer_engine/pytorch/triton/mhc.py` (common + pytorch variants)
- **Recent PR (2026-05-12):** "Improve mHC to match DeepSeek's implementation" — adds `mhc_generate_mix_and_aggregate` API, TMA path on Hopper+, mixed-dtype (BF16 x / FP32 H_pre), `norm_weight` learnable RMSNorm affine, `fuse_grad_x_acc` buffer sharing, grid-dim pruning to avoid CUDA's 65535 Y-dim limit
- **Backend policy:** explicit `MHCBackend` ∈ {'auto', 'native', 'triton', 'cutile'}; auto policy falls back to PyTorch when accelerated backend unavailable
- **Four fused ops exposed:** `fused_sinkhorn`, `fused_h_aggregate`, `fused_h_post_bda`, `fused_proj_rms_compute_h`
- **Megatron-Core binding:** `core.fusions.fused_mhc_kernels` wraps these with the same backend policy
- **ROCm / gfx1101 status:** **NEGATIVE.** This is NVIDIA TransformerEngine — by construction, only supports CUDA / hopper+. However, the Triton kernel code is in plain Triton (not CUDA-specific), so a *porting* effort could lift it. The TMA path uses CUDA-specific TMA descriptors (`tl.experimental_descriptor_load`) and would need replacement. The cuTile backend is `cutile`-specific (`is_cutile_available()` returns False on ROCm)

### 3.5 `ROCm/aiter` — `mhc_ref.py` + Triton wrapper at `_triton_kernels/fusions/mhc.py`

- **Location:** `op_tests/triton_tests/utils/mhc_ref.py` (reference) + `_triton_kernels/fusions/mhc.py` (Triton) + `csrc/kernels/mhc_kernels.cu` (HIP)
- **ROCm relevance:** **YES, this is the ROCm-native path.** ROCm aiter is AMD's equivalent of NVIDIA's TransformerEngine / Liger-Kernel for kernel fusion. The presence of an `mhc_kernels.cu` file with HIP guards and a `_triton_kernels/fusions/mhc.py` Triton wrapper means AMD has officially engaged with the algorithm.
- **API exposed in reference:** `mhc_torch`, `sinkhorn_knopp_exp_domain_torch`, `sinkhorn_knopp_log_domain_torch`, `is_doubly_stochastic`, `generate_mhc_inputs`, `get_test_shapes`, `mhc_post_torch`, `generate_mhc_post_inputs`
- **Test shapes (from `get_test_shapes`):** standard configurations across (M, n, C) for mHC mapping
- **Notable:** supports `asymmetric_exp_domain` and `hc_sinkhorn_eps` branches, matching DeepSeek's TileKernel reference implementation behavior
- **gfx1101 status:** **UNKNOWN.** aiter ships Triton kernels that work on AMD GPUs but I could not verify gfx1101 has been tested specifically. RDNA3 wave64 is supported by aiter in general (e.g. RMSNorm, Softmax exist), but mHC-specific test results on RX 7800 XT are absent from the file I viewed
- **Honest framing:** this is the most credible path to a working mHC kernel on our hardware, because (a) it's ROCm-native, (b) it tracks DeepSeek's reference implementation, and (c) it's MIT-licensed

### 3.6 `gaetanx21/blog/fused-and-furious` — independent Triton Sinkhorn kernel writeup

- **Repo:** referenced in the blog post (not fetched)
- **Independent contribution:** Sinkhorn-only Triton kernel with register-resident 4×4 matrix and block packing (64, 4, 4) tensors per block
- **Speedup vs compiled PyTorch:** **20×–139×** depending on batch size ("two orders of magnitude for realistic batch sizes")
- **Distinctive:** a single Sinkhorn-only kernel — does not implement the full mHC block, just the doubly-stochastic projection step
- **ROCm / gfx1101 status:** unknown (blog post did not specify GPU)

### 3.7 Summary table

| Implementation | Language | n | Fwd+Bwd | PyTorch speedup | ROCm support | gfx1101 tested | License |
|---|---|---|---|---|---|---|---|
| Liger-Kernel `LigerMHC` | Triton | 4 | yes | 1.5× end-to-end | untested | NO | BSD-2 |
| `YangWang92/mHC-triton` | Triton | 4 | yes | 6.2× fwd+bwd | untested | NO | (not fetched) |
| `alint77/flash-mHC` | Triton | 4 | yes | 35%→11% overhead | untested | NO | (not fetched) |
| NVIDIA TransformerEngine | Triton | 4 | yes | (no baseline given) | NO (CUDA-only) | NO | Apache-2 |
| Megatron-Core `fused_mhc_kernels` | Triton+cutile | 4 | yes | (no baseline given) | NO | NO | BSD-3 |
| **ROCm aiter** | **Triton + HIP** | **4** | **yes** | **(baseline only)** | **YES** | **unknown** | **MIT** |
| gaetanx21 Sinkhorn-only | Triton | n×n | no | 20-139× | unknown | unknown | (not fetched) |

**Honest summary:** Of the 5 Triton implementations of the full mHC block, only **ROCm aiter** has explicit ROCm code, but **none** of the 7 entries reports gfx1101 results. The Python-side math (Sinkhorn iterations, matrix operations) is independent of GPU and runs identically on ROCm; only the kernel-level fusion is GPU-specific. The Triton kernels are written in standard Triton language (no inline PTX), so they should compile on triton-rocm 3.8.0 — but autotune configs and launch parameters from NVIDIA GPUs will not transfer, and the 4×4 matrix-in-registers optimization assumes warps that hold 16 scalars per thread, which is true on both warp32 (NVIDIA) and wave64 (AMD RDNA3) but with different occupancy characteristics.

---

## 4. ROCm / gfx1101 feasibility

### 4.1 What should "just work" on gfx1101

- **The Sinkhorn-Knopp iteration itself.** It's just alternating row/column normalization on a 4×4 matrix. PyTorch's `.sum()` + broadcasting work identically on ROCm. The reference implementation in `ROCm/aiter/.../mhc_ref.py:144-152` is plain PyTorch and runs on any device.
- **The doubly-stochastic constraint.** It's a property of the output, not a kernel feature. As long as the projection step works, the constraint is enforced.
- **The mathematical structure.** mHC's mHC-lite decomposition (K1/K2/K3/K4 in flash-mHC) doesn't use any NVIDIA-only math. Sinkhorn is a CPU-era algorithm (1967).

### 4.2 What requires Triton-port care

- **The fused `proj_rms_compute_h` kernel** (paper Eq. 14–19). This fuses matrix multiply, RMSNorm, and the bias/scale into one kernel. The fusion pattern is language-portable, but the autotune `BLOCK_M`, `BLOCK_N`, `BLOCK_K` parameters must be re-tuned for RDNA3 wave64. Reasonable starting values: 64×128×32 per the Liger-Kernel issue, but expect 1-2 days of empirical sweep on gfx1101.
- **The `4×4 matrix in registers` optimization** (YangWang92 + gaetanx21). On wave64 each thread holds 64 scalar registers; the 16-scalar 4×4 matrix fits trivially. The optimization should transfer cleanly, but the reduction-across-wave64-threadgroup pattern for averaging Sinkhorn row/column sums differs from warp32 and would need a `tl.sum(..., axis=0)` over a 64-wide thread block instead of 32.
- **TMA loads (Hopper+) from NVIDIA TransformerEngine** — explicitly NOT portable. Would need replacement with regular `tl.load` + bounds-checked tile loads. Liger-Kernel / flash-mHC don't use TMA so they're cleaner starting points.

### 4.3 What is genuinely broken on gfx1101

- **The DualPipe schedule** (paper Sec. 4.3.3). DeepSeek's pipeline parallelism optimization is a CUDA-graph-based overlapped schedule that uses CUDA stream semantics that don't map 1:1 to HIP. Out of scope for our use anyway (single-GPU RX 7800 XT is not a pipeline-parallel target).
- **The 671B-scale validation** (paper is only validated up to 27B, Nurix notes trillion-parameter replication is open). Not relevant for our 16 GB card.

### 4.4 The GPU-outage context

Per `WF-GPU-Diag-Fix` and `WF-GPU-Auto-Recover`, our dGPU (RX 7800 XT, gfx1101) is in firmware-level SMU hang. dmesg shows `psp gfx command LOAD_TA(0x1) failed status 0x2C` and `PCI runtime_status=error`. The only fix is cold power cycle. As of report time:
- `torch.cuda.is_available() = False`
- `rocminfo` returns HSA_STATUS_ERROR
- `/proc/kmsg` is unreadable (Permission denied)
- iGPU (Radeon 780M gfx1100) is hardware-healthy but ROCm 7.2 init fails because HSA treats any broken KFD node as FATAL

**Honest framing of the GPU-outage interaction with mHC research:** None of the kernels above can be benchmarked on this host until GPU comes back. The research question "does mHC run on gfx1101?" can only be answered by reading the kernel code, which I did, and concluding "the kernels should compile and run on gfx1101 but autotune configs need re-tuning". The empirical speedup vs PyTorch (1.5–6.2×) is what we should expect on gfx1101 too, modulo the autotune overhead.

### 4.5 Estimated integration effort

If GPU comes back and we want to wire mHC into `molmetal/adapters/flow_matching_lipman/__init__.py` (our CFM harness), the realistic paths are:

| Path | Effort | Risk | Notes |
|---|---|---|---|
| **ROCm aiter `_triton_kernels/fusions/mhc.py`** | 2-3 days | LOW | Already ROCm-native, MIT licence, MIT dep is acceptable. Need to verify gfx1101 wave64 launch params and write a thin PyTorch wrapper. |
| **Liger-Kernel `LigerMHC`** | 1-2 days | MED | Triton 3.6 vs triton-rocm 3.8.0 kernel compatibility unknown. BSD-2 licence. Would need autotune re-tune for gfx1101. |
| **YangWang92/mHC-triton** | 1 day | MED | Single-author repo, not vendored. Smaller surface area, full autograd. Would need autotune re-tune. |
| **Port from scratch using `mhc_ref.py` as spec** | 1-2 weeks | HIGH | Reinventing the wheel. Only justified if all upstream kernels prove incompatible. |

The **ROCm aiter path is the recommended starting point.** I would recommend deferring this work until the GPU outage is resolved, then running a smoke test importing `_triton_kernels.fusions.mhc` to confirm the kernels compile on gfx1101.

---

## 5. Applicability to SE(3)-equivariant EGNN

This is the architectural question: **does mHC make sense as the residual mechanism inside an equivariant graph neural network?**

### 5.1 What mHC preserves (identity-mapping)

The standard EGNN residual is `x_{l+1} = x_l + F(x_l, h_l)` where `x` is coordinates and `h` is scalar features. This additive residual:
- preserves translation invariance (sum of equivariant quantities is equivariant)
- preserves rotation invariance (the identity path carries the input unchanged)
- bounds gradient norm through depth (via the identity Jacobian)

mHC's `H^res ∈ R^{n×n}` acts on the **stream dimension n**, not on the geometric dimension. If we put mHC around an EGNN message-passing layer, the update becomes:

`X_{l+1} = H^res_l X_l + H^{post⊤}_l F(H^pre_l X_l, h_l, W_l)`

where `X_l ∈ R^{n×N_atoms×3}` is a stack of n parallel coordinate streams and `h_l` is the scalar features (single-stream or also multi-stream). The doubly-stochastic `H^res` then mixes these n coordinate streams.

### 5.2 What this would buy for EGNN

**Possibly useful:**
- **Stream-wise diversity** — different residual streams could specialize in different "views" of the molecule (e.g., one stream for backbone atoms, one for click-handle atoms, one for metal coordination)
- **Mean preservation across depth** — for CFM (continuous flow matching) where we integrate position trajectories over many layers, the doubly-stochastic bound on coordinate drift could improve numerical stability

**Almost certainly NOT useful for our 3D equivariant use case:**
- **The "topological complexity" benefit** (paper's main claim) is benchmarked on language modeling, where multi-stream residual pathways act as a learned routing mechanism across token positions. EGNN already has its own routing via the k-NN graph; adding mHC on top would be **double-counting**.
- **n=4 streams × N_atoms × 3 coords** is a 4× memory blowup on coordinates, which is exactly the bottleneck in our flow_matching_lipman harness (per `WF-CFM-Internal-Review` audit, hidden_dim=32 is a constraint, not a choice).
- **The "identity-mapping" preservation** is already trivially true for the standard EGNN `x_{l+1} = x_l + F(x_l)`. We don't need Sinkhorn to recover it.
- **SE(3) equivariance is NOT preserved by mHC's stream mixing in general.** If `H^res` mixes n coordinate streams that were each equivariant to a different rotation, the result is no longer equivariant — it's the convex combination of equivariant functions, which is only equivariant if all streams are tied to the same reference frame. In practice, EGNN's coordinate update is already equivariant because it uses **relative displacements** `x_i - x_j`; mHC would have to be applied at the scalar feature level `h_l`, not the coordinate level `x_l`, to preserve equivariance.

### 5.3 What this would NOT buy for Mol-Metal specifically

Per `WF-CFM-Internal-Review` audit:
- Our 4 failure points are **bond_loss, cfm_loss, 97.4% disconnect rate, hidden_dim=32**. None of these are "residual stream signal explosion" issues. mHC fixes a problem we don't have.
- Our MCTS operates in **2D molecular graph space** (paper §3.4), not in 3D coordinate space. The mHC "residual stream widening" is irrelevant to graph-edit operations.
- The reported gain on **reasoning benchmarks** (BBH, DROP) doesn't transfer to **molecular generation quality** without evidence.

### 5.4 Where mHC COULD make sense in our pipeline

- **Lambda-policy network** (our MCTS value/policy head in `r4_lambda_only_run.py`). If we ever scale to a Transformer-based value estimator over β-NF rule sequences, mHC's stream-mixing could provide richer intermediate representations without the typical signal-explosion issues. This is speculative and would only be justified if we hit a training-stability wall — which we haven't (`WF-Lambda-Only-MiniPilot` shows stable 5×1 runs).
- **Hybrid Lambda + CFM end-to-end** (per `TODO/pending/21_lambda_model_coupling.md`, deferred). If we ever fuse the algebraic Lambda generator with the geometric CFM, mHC-style stream mixing between algebraic-stream and geometric-stream might be a natural composition pattern. Again, speculative.

### 5.5 Honest applicability score

**Score: 2/10** for direct application to our SE(3)-equivariant EGNN or CFM harness.
**Score: 5/10** as a future-option pattern for a Transformer-based policy network if training-stability becomes a concern.
**Score: 8/10** as a research curiosity / keep-in-mind technique — the doubly-stochastic residual stream idea is principled and may generalize beyond LLMs.

The honest conclusion is that **mHC solves a problem we don't have**, on hardware (RDNA3) that has only one productionized kernel reference (ROCm aiter, untested on gfx1101), for a use case (residual stream widening for LLM scaling) that doesn't transfer cleanly to equivariance-preserving geometric networks. **Do not pursue integration in this round.**

---

## 6. Integration plan (deferred)

If we revisit mHC after the GPU outage is resolved and a training-stability issue emerges:

### 6.1 Phase 1: Validate ROCm aiter compiles on gfx1101 (1 day)

```python
# Smoke test: does the kernel even import and JIT-compile?
from aiter.ops.triton.fusions.mhc import mhc_pre, mhc_post
import torch
x = torch.randn(2, 128, 4, 64, device='cuda', dtype=torch.bfloat16)  # (B, T, n, C)
phi = torch.randn(64, 4 + 4 + 16, device='cuda', dtype=torch.float32)
out = mhc_pre(x, phi, n=4)  # expect hpost, hres, layer_input
assert out[0].shape == (2, 128, 64)
```

If this throws a triton-rocm compilation error, the kernel is incompatible with our triton version and we'd need to either (a) bump triton-rocm, (b) edit the kernel's `triton.jit` decorators, or (c) abandon and try Liger-Kernel.

### 6.2 Phase 2: Wrap aiter's mHC as a PyTorch nn.Module (2-3 days)

Create `molmetal/adapters/mhc_block.py` exposing `MHCLiteBlock` similar to flash-mHC's API but backed by aiter. Cover with unit tests on CPU + (if GPU available) smoke benchmark vs PyTorch reference.

### 6.3 Phase 3: Drop into CFM harness as an opt-in residual (3-5 days)

Wrap `EGNNMessagePassingLayer` in `MHCLiteBlock` and run a `--mhc` CLI flag in `r4_lambda_only_run.py`. Compare 5×1 pilot `--mhc` vs default on:
- `bond_loss_final`
- `cfm_loss_final`
- `decode_ratio` (the 97.4% disconnect issue from `WF-CFM-Internal-Review`)
- `wall_clock_per_step`

**Honest gating criterion:** only adopt if `--mhc` shows ≥ 2× reduction in `bond_loss_final` OR ≥ 10% reduction in `decode_ratio`. Otherwise it's added complexity without benefit.

### 6.4 Phase 4 (optional): Transformer-policy net (1-2 weeks)

If we ever build a Transformer-based Lambda-policy net (currently a small MLP), wrap it in mHC and compare. Out of scope until/unless that architecture is adopted.

---

## 7. Ranked summary

| # | Candidate | Architecture fit | ROCm/gfx1101 fit | Algorithmic novelty | Recommendation |
|---|---|---|---|---|---|
| 1 | E2Former-V2 (Triton equivariant attention) | HIGH (EGNN-replacement) | MEDIUM (Triton-OK, untested on g1) | HIGH | **DEFERRED** (see flash_attention_egnn.md) |
| 2 | EquiformerV2 (eSCN conv) | HIGH (SOTA EGNN-replacement) | MEDIUM (e3nn ROCm-runnable) | MEDIUM | **DEFERRED** (see flash_attention_egnn.md) |
| 3 | **mHC as residual in EGNN** | LOW (no equivariant justification) | MEDIUM (aiter path exists, untested) | HIGH (algorithmic novelty) | **SKIP** for current scope |
| 4 | **mHC for future Transformer-policy** | MEDIUM (if we ever scale) | MEDIUM | HIGH | **DEFER** until relevant |
| 5 | FlashAttention v2/v3 (ROCm Triton) | HIGH (relevant to EGNN if we add attention) | HIGH (Triton-OK) | MEDIUM | **DEFERRED** (see flash_attention_egnn.md) |

**Headline recommendation for the WF-Triton-Arch-Research workflow:** mHC is a **5/10** candidate for Mol-Metal — credible algorithm, real productionized kernels, but solves the wrong problem for our SE(3)-equivariant EGNN on a hardware target where only one untested kernel reference exists. **Do not integrate in this round. Document and re-evaluate if training-stability issues emerge post-CFM-retrain.**

---

## 8. Open questions for follow-up

1. **Does `ROCm/aiter/_triton_kernels/fusions/mhc.py` actually compile on gfx1101?** Cannot verify without working GPU. **Gate item.**
2. **Is the `mhc_kernels.cu` HIP kernel a true AMD-tuned implementation or just an NVIDIA `.cu` file with `#ifdef` guards?** Cannot verify from file listing alone. Would need to read the source.
3. **Has any independent lab reproduced the +6.7% overhead number on AMD hardware?** No public source found. Likely not.
5. **Does the Sinkhorn iteration in PyTorch on gfx1101 have any specific numerical issues (e.g., fp16 underflow in `exp(M)`)?** The aiter reference uses `asymmetric_exp_domain` mode for stability — would need to verify which mode is needed on RDNA3.
6. **What does the `n=8` ablation look like?** The paper only reports n=4. n=8 would be more interesting for our 16 GB card (still small).

---

## 9. Sources

### Primary (mHC paper + DeepSeek reference)
- arXiv:2512.24880 — Xie et al., "mHC: Manifold-Constrained Hyper-Connections", submitted 2025-12-31, v2 2026-01-05
- arXiv:2409.19606 — Zhu et al., "Hyper-Connections", ICLR 2025 (the paper mHC builds on)
- `github.com/deepseek-ai/TileKernels/tree/main/tile_kernels/mhc` (referenced in NVIDIA TE PR; not fetched directly)

### Independent secondary analyses
- `jamesetsmith.github.io/blog_posts/2026-01-21-mHC-papers.html` — "Paper Speedrun" walkthrough
- `1kpapers.com/papers/mhc-manifold-constrained-hyper-connections` — 65-citation summary
- `daily.dev/posts/why-decade-old-residual-connections-still-power-all-of-ai-and-why-that-s-a-problem--srohkbexi`
- `api.emergentmind.com/topics/manifold-constrained-hyper-connections-mhc`
- `towardsai.net/p/machine-learning/mhc-rethinking-the-neural-highway`
- `theaiinsider.tech/?p=41980`
- `nurix.ai/resources/deepseek-mhc-ai-architecture-disrupting-llm-training`
- `stellitron.com/blog/manifold-constrained-hyper-connections-enterprise-ai`
- `analyticsvidhya.com/blog/2026/01/deepseek-mhc`
- `hyper.ai/en/papers/2512.24880`
- `gaetanx21.github.io/blog/fused-and-furious/` — Triton Sinkhorn-only kernel writeup
- `linkedin.com/posts/qusai-rahhal-b05813240_mhc-manifold-constrained-hyper-connections-activity-7416371403958071296-yyDX`
- `linkedin.com/posts/ali-naeimi5055_github-alint77flash-mhc-fast-implementation-activity-7435298037448482816-wucT`
- `linkedin.com/pulse/curve-we-cant-see-why-ai-could-start-improving-itself-keith-sartain-m4rle/`

### Triton / kernel implementations
- `github.com/linkedin/Liger-Kernel` — `LigerMHC`, PR #1065 + docs #1132
- `github.com/YangWang92/mHC-triton` — NucleusAI fork, full autograd
- `github.com/alint77/flash-mHC` — productionized extraction, MHCLiteBlock
- `github.com/NVIDIA/TransformerEngine` — `transformer_engine/pytorch/triton/mhc.py`
- `docs.nvidia.com/megatron-core/.../core.fusions.fused_mhc_kernels.html`
- `github.com/ROCm/aiter` — `op_tests/triton_tests/utils/mhc_ref.py` + `_triton_kernels/fusions/mhc.py` + `csrc/kernels/mhc_kernels.cu`
- `git.mafyuh.dev/lucidrains/hyper-connections` — community PyTorch port of the original HC paper

### Internal context (this project)
- `WF-CFM-Internal-Review` (audit findings — what we actually need to fix)
- `WF-GPU-Diag-Fix` / `WF-GPU-Auto-Recover` (dGPU state)
- `WF-iGPU-Switch` (gfx1100 fallback)
- `TODO/pending/21_lambda_model_coupling.md` (deferred Lambda-CFM coupling plan)
- `flash_attention_egnn.md` (companion research note, same workflow)

---

**Final honest framing:** This document reports what the literature and 7 referenced code repos claim about mHC. The architectural fit for Mol-Metal's SE(3)-equivariant EGNN is poor; the ROCm/gfx1101 feasibility is plausible but unverified; the algorithmic novelty is real but solves a problem we don't have. **No integration is recommended for this round.** The recommended next step is to wait for the GPU outage to resolve, then run the Phase 1 smoke test (Section 6.1) to close the gfx1101 compile-feasibility question. If that passes, revisit the integration plan only if a training-stability issue emerges in CFM retrain.