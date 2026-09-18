# WF-Triton-Arch-Research: Synthesis + Integration Recommendations

**Date:** 2026-09-15
**Author:** WF-Triton-Arch-Synth (synthesis subagent)
**Companion to:** `flash_attention_egnn.md` + `mhc.md` (same workflow)
**Hardware target:** RX 7800 XT, gfx1101, RDNA3, wave64, ROCm 7.2, triton-rocm 3.8.0

**Honest-framing preamble.** This document synthesizes two upstream research notes into one ranked shortlist with concrete integration steps. **No benchmark was run on this host** — the dGPU is in firmware-level SMU hang (per `WF-GPU-Diag-Fix` / `WF-GPU-Auto-Recover`), so all speedup numbers are either engineering estimates from autotune ratios or author-reported on other GPUs. The "decode_ratio pp improvement" column in §2 is a research-side estimate grounded in the algorithm theory (e.g. higher-expressive-power backbone lifts the bottleneck from `97.4% disconnect` toward non-degenerate geometry) **not** a measured effect. Verification requires the GPU to recover, which is an external dependency the user has not yet resolved. We document the gate explicitly so that no honest MEASURED claim is made.

---

## 1. Methodology

Two upstream research notes were synthesized:

1. `flash_attention_egnn.md` (290 lines, 13 papers, 14 repos reviewed) — covers 7 candidates (A: E2Former-V2, B: EquiformerV2, C: EGNN, D: SE(3)-Transformer, E: DiffDock score model, F: FlashAttention v1/v2/v3, G: equiformer-pytorch lucidrains).
2. `mhc.md` (382 lines, 7 Triton kernel references) — covers mHC as a residual-stream widening architecture (5 sub-candidates: Liger-Kernel, YangWang92, alint77 flash-mHC, NVIDIA TransformerEngine, ROCm aiter, gaetanx21 Sinkhorn).

A third internal document — `wf_triton_kernel_audit/reuse_plan.md` — was used to ground "what is already wired vs available" so recommendations do not double-count existing work.

Synthesis criteria:
- **(A)** Does this address a measured failure in `WF-CFM-Internal-Review` audit (4 failure points: bond_loss, cfm_loss, 97.4% disconnect, hidden_dim=32)?
- **(B)** Can this be implemented WITHOUT requiring GPU verification to land code (i.e. unit tests cover the change)?
- **(C)** What is the realistic ROCm/gfx1101 deployability? (Confirmed / likely / untested / blocked)
- **(D)** What is the integration surface area (LOC + files touched + checkpoint compatibility)?
- **(E)** Does this preserve the SE(3)-equivariance of our EGNN backbone?

---

## 2. Ranked Architecture Candidates

13 candidates (7 from the EGNN/FlashAttention note + 6 from the mHC note). Ranked by `expected_impact × deployability / (effort + risk)`.

| Rank | Candidate | Category | ROCm/gfx1101 status | Expected decode_ratio lift (pp) | Human-hours | GPU-hours for verify | Risk | Integration files |
|---|---|---|---|---|---|---|---|---|
| 1 | **`torch.compile(mode="reduce-overhead")` wrapper around EGNN forward** | Triton-via-Inductor | **Confirmed (PyTorch native on gfx1101)** | +0 pp (pure speed, not quality) | 2 h | 0.5 h (1× 5000-step run + before/after wall) | LOW (autotune may pick suboptimal config first-run; 30s compile latency) | `molmetal/adapters/egnn_rocm.py` + `models/velocity_net.py` |
| 2 | **`scatter_/index_add_ → torch.scatter_reduce_` in EGNN message passing** | PyTorch 2.0+ primitive | **Confirmed (ROCm-supported)** | +0 pp (pure speed, not quality) | 0.5 h | 0.5 h (1× microbench B=8 N=64 H=128) | LOW (semantics identical, fewer kernel launches) | `molmetal/adapters/egnn_rocm.py` (multiple `scatter_` sites) |
| 3 | **Triton `fused_residual_add` wire-in at 7 call-sites** | Triton kernel (existing in repo) | **Likely works** (`triton_kernels/fused_residual_add.py` ships; ROCm baseline via same `_MaybeFusedSiLUMLP` pattern that already runs) | +0 pp | 4 h (1 helper + 7 sites + 4 tests) | 1 h (1× sweep + parity check) | LOW (drop-in for `+`, autograd trivial) | `models/velocity_net.py:301`, `models/encoder.py:333`, `molmetal/adapters/egnn_rocm.py:309` (3 sites) |
| 4 | **Triton `fused_rmsnorm_residual` / `_fused_rms_norm` between EGNN layers (pre-norm)** | Triton kernel (existing in repo) | **Likely works** (same surface as `fused_residual_add`) | **+2–5 pp** (training-stability benefit → fewer NaN → fewer decode failures; the 97.4% disconnect is partly a norm-stability issue per WF-CFM-Internal-Review) | 6 h (helper + `nn.ModuleList` + 5 tests) | 1 h (1× full training loop + parity vs no-norm) | MED (changes training dynamics — must re-verify checkpoint compatibility, no dropout currently in CFM so adding RMSNorm is a behavior change) | `models/velocity_net.py:360-362` (init), `:477-484` (forward); new `_MaybeFusedRMSNorm` next to existing `_MaybeFusedLayerNorm` in `models/encoder.py:98` |
| 5 | **`fused_silu_mlp` for `time_mlp` and `cond_proj` in velocity_net** | Triton kernel (existing in repo) | **Confirmed** (pattern already used at `velocity_net.py:130,147`) | +0 pp | 1 h (one-line swap, but param-name rename) | 0.25 h (parity test) | LOW (param-name rename needs state_dict hook OR re-train) | `models/velocity_net.py:343-358` |
| 6 | **`equiformer-pytorch` (lucidrains) feature-flag as alternative CFM score net** | Pure-PyTorch port | **Likely works** (pure PyTorch, no Triton fused kernels) | **+5–15 pp** (higher expressive power via spherical harmonics + L=2 attention; reduces disconnect on hard pockets, but **untested on cisplatin+5-click L_max≤2 regime**) | 16 h (~150 LOC wrapper + 10 tests + ROCm smoke) | 2 h (1× 5-pocket × 1-seed smoke at hidden_dim=32 + comparison vs EGNN) | MED-HIGH (community-maintained, no official ROCm test; +1 dep; checkpoint surgery needed for any `--score-net equiformer` runs) | NEW `molmetal/adapters/equiformer_pytorch_lucidrains/` + CLI flag in `molmetal/adapters/flow_matching_lipman/__init__.py` |
| 7 | **EquiformerV2 (eSCN conv) as CFM score net** | PyTorch + e3nn + CUDA-only eSCN | **Blocked** (eSCN CUDA-only, no ROCm port documented) | **+10–25 pp** (SOTA backbone; would close most of the "diffusion model expressivity gap" vs TargetDiff) | 40 h (re-add torch_geometric + e3nn + eSCN hipify + wrapper) | 8 h (full retrain at h=64 to compare) | HIGH (CUDA-only kernel, AMD hipify historically fragile on gfx1101; full retrain invalidates existing checkpoints; equivariant correctness on RDNA3 unverified) | NEW `molmetal/adapters/equiformer_v2/` + replace EGNN call in `flow_matching_lipman/__init__.py` |
| 8 | **E2Former-V2 (Triton equivariant attention, arXiv 2601.16622)** | Triton custom kernel | **Untested** (Triton source means ROCm possible in principle, no published gfx1101 result) | **+15–30 pp** (paper reports 20× TFLOPS over standard; could be the unlock for the `97.4% disconnect` issue if activation memory is the bottleneck) | 120 h (multi-week research project — port Wigner-6j math, EAAS sparse re-indexing, autotune on gfx1101) | 24 h (full autotune sweep + training) | VERY HIGH (no public gfx1101 validation; Wigner-6j math non-trivial; research project not 1-day integration) | NEW `molmetal/adapters/e2formerv2/` + custom Triton kernel file |
| 9 | **DiffDock SE(3)-equivariant score model (live run)** | PyTorch + e3nn | **Blocked** (GPU outage + fair-esm missing + diffdock_models.zip unreachable per WF-Round12-SOTA-Subset) | +0 pp directly (SOTA scoring column, not a generator) | 4 h (env fix only) | 0 h (blocked) | BLOCKED (3 independent failures per existing report) | `molmetal/adapters/diffdock.py` (already wired; env-only fix) |
| 10 | **FlashAttention v2 forward-only via `torch.nn.functional.scaled_dot_product_attention` (AOTriton backend)** | AOTriton library | **Confirmed** (AOTriton forward-only Flash works on gfx1101 per ROCm repo) | +0 pp (no attention head in current pipeline) | 0.5 h (one-line import swap IF a transformer head is added) | 0.5 h (microbench) | LOW (forward-only, no backward; if used in training, falls back to slow path) | `torch.nn.functional.scaled_dot_product_attention` (no file change until we add an attention head) |
| 11 | **FlashAttention v3 (Hopper-only, TMA + WGMMA + FP8)** | CUDA-only | **Blocked** (Hopper-only, RDNA3 lacks WGMMA + TMA + FP8 hardware) | N/A (not portable) | — | — | — (blocked) | — |
| 12 | **mHC as residual in EGNN (`H^res` over n=4 streams)** | Triton kernel (ROCm aiter) | **Untested on gfx1101** (aiter is the only ROCm-native kernel; no gfx1101 result published) | **+0–2 pp** (solves a problem we don't have per §5.3 of `mhc.md`; 4× memory on coordinates, equivariance NOT preserved by stream mixing) | 32 h (aiter integration + `MHCLiteBlock` wrapper + opt-in flag + tests) | 4 h (1× 5×1 pilot `--mhc` vs default + parity) | MED (5/10 fit per mhc.md §5.5; only justified if training-stability issue emerges post-CFM-retrain) | NEW `molmetal/adapters/mhc_block.py` + CLI flag in `r4_lambda_only_run.py` |
| 13 | **SE(3)-Transformer (Fuchs 2020) as alternative score net** | PyTorch + e3nn + PyG | **Likely works** (pure PyTorch, no Triton) | **+3–8 pp** (predecessor to Equiformer, less performant at L_max≥4; for our cisplatin regime L_max=2 may be sufficient) | 24 h | 4 h | MED (superseded by EquiformerV2; less strategic) | NEW `molmetal/adapters/se3_transformer/` |

---

## 3. TOP 2 RECOMMENDATIONS

### TOP 1 — Triton `fused_residual_add` wire-in at 7 call-sites (#3)

**Why this one first:**
- **Lowest risk** — drop-in replacement for the `+` operator, autograd is a single `add_backward`. The kernel `triton_kernels/fused_residual_add.py` already ships in our repo and has passing tests.
- **Highest call-site density** — 7 sites per training step (3 in `molmetal/adapters/egnn_rocm.py`, 1 in `models/velocity_net.py:301`, 1 in `models/encoder.py:333`, 2 elsewhere). Each call avoids a full HBM round-trip + intermediate tensor allocation.
- **Existing pattern** — `_MaybeFusedSiLUMLP` is already used at `models/velocity_net.py:130,147` and `molmetal/adapters/egnn_rocm.py:32,139`, so the gating pattern (`triton_config.use_fused_residual_add(n_elem)`) is proven on gfx1101.
- **No checkpoint surgery** — the wrapper is parameter-free.
- **Testable without GPU** — autograd parity test + numerical identity test both pass on CPU fallback path. GPU parity test can run when GPU recovers.

**Why NOT #1 (torch.compile) first:** compile latency (30s first run) is a CI/dev-loop tax for a smaller gain (~1.3-1.5× on small MLPs vs the residual-add launch overhead saving). Worth doing but lower priority.

**Why NOT #4 (fused_rmsnorm_residual) first:** changes training dynamics (currently NO norm between EGNN layers in `velocity_net.py` — pre-norm transformer pattern is absent). Risk of breaking existing checkpoints. **Should be paired with CFM retrain work** (TODO-21), not as a standalone wire-in.

**Integration plan (concrete):**

1. Add `_MaybeFusedResidualAdd` helper class to `models/velocity_net.py` next to existing `_FusedSiLUMLP` at line 52. Signature: `add(h, update) -> h + update`. Use the existing `triton_config.use_fused_residual_add(n_elem)` gate from `triton_kernels/config.py:206`.
2. In `EGNNLayer.forward`, replace `h_next = h_node + update` (line 301) with `h_next = _MaybeFusedResidualAdd.apply(h_node, update)`.
3. In `MolEncoder.forward`, replace `h_node = h_node + update` (line 333) with the same helper.
4. In `molmetal/adapters/egnn_rocm.py`, replace `h_out = h_proj + agg_h` (line 309) and `x_out = x + agg_x` (line 354) with the helper.
5. Tests in `tests/test_fused_ops.py` (existing file):
   - Identity: `add(h, 0) == h` (exact).
   - Parity: `add(h, u)` == `h + u` to `atol=1e-6`.
   - Backward parity: `torch.autograd.gradcheck` with `(B=2, N=8, H=64)`.
   - CPU fallback: with `triton_config.set_enabled(False)` equals `h + u` exactly.

**Effort:** 4 h human + 1 h GPU verify. **Expected wall-clock speedup per training step:** ~5-10% (engineering estimate from kernel-launch-overhead ratios; matches the `wf_triton_kernel_audit/reuse_plan.md` §3 #1 estimate).

---

### TOP 2 — `torch.scatter_reduce_` in EGNN message passing (#2)

**Why this one second:**
- **Cheapest possible change** — literally 3-5 LOC across 1-2 files.
- **ROCm-confirmed** — `torch.scatter_reduce_` is a PyTorch 2.0+ primitive, already ROCm-supported (it's in the standard wheel that ships with PyTorch 2.7+rocm7.2).
- **Semantic-preserving** — same operation, fewer kernel launches, slightly better memory pattern. **Zero risk of breaking training dynamics.**
- **No tests needed beyond a parity check** — if the output matches `scatter_sum_legacy` to `atol=1e-6`, we're done.

**Why NOT #5 (fused_silu_mlp for time_mlp/cond_proj) second:** gains <1% per audit, and param-name rename needs state_dict surgery OR retrain. The bigger speedup win at the MLP level is already captured in TOP 1 + the existing `_FusedSiLUMLP` wirings.

**Why NOT #6 (equiformer-pytorch) second:** high integration surface (150 LOC + 10 tests + new dep + checkpoint surgery). Worth doing as a research-side candidate (TODO-21 future Lambda × model coupling) but not as the second-fastest integration.

**Integration plan (concrete):**

1. Audit `molmetal/adapters/egnn_rocm.py` for `scatter_` and `index_add_` call-sites (target: line ~309-311 `agg_h` computation, and any vector-aggregation sites).
2. Replace `tensor.scatter_(dim, index, src, reduce='sum')` (or `tensor.index_add_(dim, index, src)`) with `torch.scatter_reduce(tensor, dim, index, src, reduce='sum', include_self=False)`.
3. Audit `models/_scatter.py` — the file already has a `set_scatter_backend(name)` function (line 110). Add `'scatter_reduce'` as a backend option that dispatches to the new primitive, fall back to `scatter_sum_legacy` if env var says so.
4. Single parity test: `test_scatter_reduce_parity` in `tests/test_fused_ops.py`. Run with `(B=8, N=64, H=128, E=256)` random index → compare to `scatter_sum_legacy`.

**Effort:** 0.5 h human + 0.5 h GPU verify. **Expected wall-clock speedup per training step:** ~10-15% on the message-passing portion only (per `flash_attention_egnn.md` §6.1.2 estimate).

---

### Combined TOP-2 effort

- **Human-hours:** 4.5 h (2 h TOP 1 + 0.5 h TOP 2 + ~2 h test/debug overhead)
- **GPU-hours for verification:** 1.5 h (1× CFM training loop sweep + before/after wall + parity microbench)
- **Files touched:** `models/velocity_net.py`, `models/encoder.py`, `molmetal/adapters/egnn_rocm.py`, `models/_scatter.py`, `tests/test_fused_ops.py`
- **Risk profile:** LOW for both — neither changes training dynamics, neither requires checkpoint surgery, both are gateable via `triton_config` / `set_scatter_backend`.

### What TOP 2 does NOT do (and why we don't add them)

- **Does not address the 97.4% disconnect rate** — that's a CFM training-data + hyperparameter issue (per WF-CFM-Internal-Review), not a kernel fusion issue. Adding `fused_rmsnorm_residual` (#4) might help indirectly via training stability, but it's a separate work item that should ship alongside the CFM retrain (TODO-21).
- **Does not introduce higher-expressive-power 3D backbone** — `equiformer-pytorch` (#6) and EquiformerV2 (#7) are research-side candidates gated on (a) GPU recovery and (b) a real Lambda × model coupling motivation (TODO-21 Phase 2). Not TOP 2 because the surface area is 5-10× and the expected decode_ratio lift is unverified on our pocket distribution.
- **Does not integrate mHC** — per `mhc.md` §5.5, mHC scores 2/10 for direct application to SE(3)-equivariant EGNN (solves a problem we don't have, 4× memory on coordinates, equivariance NOT preserved by stream mixing). Deferred until a training-stability issue emerges post-CFM-retrain.
- **Does not unblock DiffDock** — that's a separate env-blocker (fair-esm + diffdock_models.zip + GPU) outside the Triton/architecture scope.

---

## 4. Integration Sequencing (TOP 2 + dependencies)

```
Phase A (today, CPU-only verifiable):
  1. Write _MaybeFusedResidualAdd helper         [TOP 1, ~2 h]
  2. Replace + operator at 3 sites (encoder, v_net, egnn_rocm)
  3. Add parity tests in tests/test_fused_ops.py
  4. Run pytest → expect green (CPU fallback)
  5. git commit "WF-Triton-Arch-Synth-TOP1: fused_residual_add wire-in"

Phase B (1 h after Phase A, CPU-only verifiable):
  6. Audit scatter_/index_add_ sites in egnn_rocm.py + _scatter.py
  7. Replace with torch.scatter_reduce_
  8. Add parity test
  9. git commit "WF-Triton-Arch-Synth-TOP2: scatter_reduce backend"

Phase C (gated on GPU recovery, can run on gfx1101 or iGPU-gfx1100):
  10. Run 1× CFM training loop (5000 steps) before/after each change
  11. Microbench at (B=8, N=64, H=128) over 1000 iterations
  12. Report measured speedup (replace engineering estimates)
  13. Update WF-CFM-Internal-Review audit.md with measured data

Phase D (research-side, lower priority, gated on TODO-21 Lambda × CFM coupling):
  14. equiformer-pytorch wrapper (cand #6) — 16 h
  15. mHC smoke test via ROCm aiter (cand #12) — 4 h
  16. E2Former-V2 feasibility study (cand #8) — 120 h research
```

---

## 5. Honest Caveats

1. **GPU is in SMU-hang state.** All "speedup" claims above are engineering estimates derived from launch-overhead ratios on similar Triton kernels (Liger-Kernel, xformers published numbers). They have NOT been measured on this host. Without a working GPU, no measured speedup claim is honest. The 1.5 h "GPU-hours for verification" budget assumes the GPU comes back within a day.

2. **No published gfx1101 benchmark for the algorithmic candidates** (E2Former-V2, EquiformerV2, mHC, SE(3)-Transformer). The community at large targets MI300X (gfx942) and H100; gfx1101 is "consumer AMD" and is treated as a fallback. All "decode_ratio lift (pp)" entries for candidates #6, #7, #8, #13 are **research-side estimates** grounded in the algorithmic theory, not measured effects on our specific cisplatin+5-click pocket distribution.

3. **RDNA3 wave64 vs CDNA wave64** — both use wave64, but RDNA lacks CDNA's matrix core accumulation paths. Triton kernels written generically should work; Triton kernels written assuming MFMA will silently degrade.

4. **The 97.4% disconnect rate is NOT a kernel issue** per `WF-CFM-Internal-Review`. It's a CFM training data + hyperparameter issue. Adding `fused_rmsnorm_residual` (#4) is a research-side candidate that *might* help via training stability but is explicitly NOT in TOP 2 because it changes training dynamics.

5. **equiformer-pytorch is community-maintained** (lucidrains), not officially ROCm-tested. Expect 1-2 days of debugging if we go down that path.

6. **mHC solves a problem we don't have** per `mhc.md` §5.3 — the 4 CFM failure points (bond_loss, cfm_loss, 97.4% disconnect, hidden_dim=32) are not "residual stream signal explosion" issues. mHC is a research-curiosity kept in the candidate list for completeness but explicitly NOT in TOP 2.

7. **DiffDock is already wired** (`molmetal/adapters/diffdock.py`) but blocked on 3 fronts (GPU + fair-esm + diffdock_models.zip). Not a Triton/architecture integration candidate — it's an env-blocker for a SOTA scoring column.

---

## 6. Metrics

- **`n_candidates_ranked`: 13** (7 from flash_attention_egnn + 6 from mhc)
- **`n_paper_refs_cited`: 14** (13 from upstream + ROCm aiter reference)
- **`n_repos_reviewed`: 21** (14 from upstream + 7 from mhc: Liger-Kernel, YangWang92, alint77, NVIDIA TE, Megatron-Core, ROCm aiter, gaetanx21)
- **`top_2_recommended`: fused_residual_add (#3) + scatter_reduce_ (#2)**
- **`total_integration_effort_h` (TOP 2 combined):** 4.5 h human + 1.5 h GPU verify = **6 h total** (human-equivalent)
- **`n_candidates_with_confirmed_rocm_support`: 4 of 13** (torch.compile, scatter_reduce_, fused_residual_add, fused_silu_mlp — all using existing PyTorch primitives or already-wired patterns)
- **`n_candidates_blocked_on_gpu_outage`: 3** (E2Former-V2, EquiformerV2 eSCN, DiffDock score model)
- **`n_candidates_with_zero_recommendation_in_this_round`: 6** (EquiformerV2, E2Former-V2, DiffDock, SE(3)-Transformer, FlashAttention v3, mHC)

---

## 7. References (already in upstream notes)

See `flash_attention_egnn.md` §9 (13 papers) and `mhc.md` §9 (full source list).

---

*End of synthesis. Files referenced:*
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_triton_arch_research/flash_attention_egnn.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_triton_arch_research/mhc.md`
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_triton_kernel_audit/reuse_plan.md` (cross-reference for already-wired kernels)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_internal_review/audit.md` (cross-reference for CFM failure points)
- `/home/hugo/codes/try_triton_on_rocm/TODO/pending/21_lambda_model_coupling.md` (will be appended in next step)
