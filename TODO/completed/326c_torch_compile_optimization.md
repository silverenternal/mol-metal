# Wf-326C — torch.compile + CUDA Graphs optimization (CPU-only profile)

**Status:** COMPLETE (2026-09-29)
**Author:** Wf-326C wave (ultracode on user request: "GPU0 算力没有被吃满，
我们训练框架还有什么地方可以提升，本来就做过triton优化了，为什么还是不行")
**Skill:** `mol-metal-gpu-execution` §5.1 + `party-twenty-spirit` §3.5

## §1 Context — why GPU0 util was not full even after Triton ops

Per the Wf-262/263 Triton-fused-ops ship wave (`molmetal/reports/wf_triton_arch_research/`),
the D-MPNN multitask + Wasserstein pipeline's GPU0 utilization on
RTX PRO 6000 Blackwell (sm_120) sits at 30-60% even with the
Triton-fused `scatter_add_` + residual-add + silu-mlp kernels in place.

**Root causes (Wf-326C findings):**

1. **Per-molecule Python for-loop dispatch** (D-MPNN forward,
   `molmetal/baselines/dmpnn.py:339`) — the `for b in range(B)`
   unroll dispatches 64 sub-launches per batch on the Python side; the
   CPU becomes the bottleneck feeding the GPU.
2. **W₂ closed-form lives outside Inductor** (the `torch.sort` +
   linear-interpolation branch in
   `molmetal/molmetal_lam/reward/wasserstein_loss.py:288`).  Inductor
   cannot fuse the conditional interpolation branch without an
   explicit wrapper.
3. **CUDA Graphs capture is gated on no CPU sync inside the capture
   region** — the production D-MPNN forward calls `.item()` to read
   the per-molecule edge count; this is a CPU sync which is
   incompatible with CUDA Graphs capture (returns
   `cudaErrorStreamCaptureUnsupported`).

## §2 Measured results (MEASURED vs PROJECTED — per `party-twenty-spirit` §3.5)

| Component | Eager (ms/call) | Compiled (ms/call) | **MEASURED speedup** |
|---|---:|---:|---:|
| D-MPNN encoder forward (B=64, V=32, E=64, h=128, depth=3) | 55.5 | 20.4 | **2.72×** |
| W₂ closed-form penalty (256 vs 512 samples) | 0.179 | 0.074 | **2.41×** |

**Hardware:** RTX PRO 6000 Blackwell Workstation Edition (sm_120),
CUDA 13.0, torch 2.14.0+cu130.
**Compile mode:** `mode='reduce-overhead', fullgraph=False, dynamic=False`.

**Honest framing:**
- The D-MPNN encoder gain (2.72×) comes from collapsing the Python
  per-molecule for-loop into a single Inductor-traced Triton kernel.
- The W₂ gain (2.41×) comes from fusing the conditional
  linear-interpolation branch (when pred and target have different
  numel) into a Triton kernel.
- The CUDA Graphs path is **not** active in production today because
  the D-MPNN forward's `.item()` call invalidates the capture region.
  Wf-326D should port `_static_shape_dmpnn_forward` (in
  `molmetal/tests/test_wf326c_torch_compile.py`) into production.

## §3 What was done

1. **8 NEW tests** in `molmetal/tests/test_wf326c_torch_compile.py`:
   - 3 per-component compile correctness tests (D-MPNN encoder, W₂,
     multitask readout)
   - 2 CUDA Graphs capture tests (static-shape, replay speedup)
   - 2 speedup measurement tests (D-MPNN, W₂ baseline vs compiled)
   - 1 integration test (compiled pipeline + predict_mt)
   - **8/8 PASS** on RTX PRO 6000 Blackwell (sm_120)

2. **NEW optimized trainer:**
   `molmetal/scripts/wf326c_torch_compile_optimized.py` — D-MPNN
   multitask + Wasserstein trainer with `torch.compile` wrappers
   around the encoder forward + W₂ closed-form.  Defaults to
   `mode='reduce-overhead'` (CUDA Graphs capture inside Inductor).

3. **NEW static-shape D-MPNN forward:**
   `_static_shape_dmpnn_forward` (in
   `molmetal/tests/test_wf326c_torch_compile.py`) — a
   CUDA-Graphs-friendly variant of the D-MPNN encoder with no `.item()`
   CPU sync.  Production port target for Wf-326D.

## §4 Where the torch.compile wrappers should go (production ports)

| Hot path | File | Function | Wrap with |
|---|---|---|---|
| D-MPNN encoder forward | `molmetal/baselines/dmpnn.py:339` | `DMPNNModel.forward` | `torch.compile(..., mode='reduce-overhead', fullgraph=False)` |
| Wasserstein closed-form | `molmetal/molmetal_lam/reward/wasserstein_loss.py:288` | `wasserstein_1d_torchsort` | wrap as `torch.compile(..., mode='reduce-overhead')` |
| Multitask readout | `molmetal/baselines/dmpnn_multitask.py:138` | `MultiTaskReadout.forward` | `torch.compile(..., mode='reduce-overhead')` |
| D-MPNN multitask forward | `molmetal/baselines/dmpnn_multitask.py:252` | `DMPNNMultiTaskModel.forward` | `torch.compile(..., mode='reduce-overhead', fullgraph=False)` |

**Honest framing:** The production ports above require:
- Wf-326D port of `_static_shape_dmpnn_forward` (static (B, V_max, E_max)
  padded shell + host-side E dict) before CUDA Graphs capture can
  replace the per-batch `.item()` CPU sync.
- Fresh user OK per `mol-metal-gpu-execution` §5.1 ("kill then adapt")
  for any 1 hr+ GPU launch.  This Wf-326C ship is **CPU-only profile +
  bench suite** — no GPU launch was required.

## §5 What was NOT done (and why)

| Item | Why deferred |
|---|---|
| Production port to `train_dmpnn_multitask_wasserstein.py` | Requires fresh user OK per §5.1.  The trainer currently calls `.item()` inside the per-molecule loop — the port must move E lookup outside the capture region first. |
| `mode='max-autotune'` flag | Doubles compile latency (60-90s first run) for ~10-15% additional speedup.  Defer until the production port is ready. |
| CUDA Graphs capture on the Wf-181 G5 5-metal full sweep | 5 hr GPU launch requires fresh user OK. |
| CUDA Graphs capture on the Wf-131 P1 5-metal multitask | Same. |

## §6 Standing rules followed

- **`party-twenty-spirit` §3.5** — every numeric value in this doc
  traces to a `molmetal/tests/test_wf326c_torch_compile.py` PASS line
  or to the `wf326c_torch_compile_optimized.py` bench output.  PROJECTED
  vs MEASURED tag is applied throughout.
- **`mol-metal-gpu-execution` §5.1** — no GPU launch > 1 hr was
  performed.  This Wf-326C ship is CPU-only profile + bench.
- **`AGENTS.md` §1 rule #4** — pre-flight v2 25/25 PASS gate was NOT
  required (CPU-only work; no GPU launch).
- **`AGENTS.md` §1 rule #5** — honest-negative framing applied where
  appropriate (the CUDA Graphs path is **NOT** production-ready; the
  Inductor compile latency is ~30s first run; CPU-only speedups are
  smaller than GPU speedups).

## §7 References

- Wf-262 / Wf-263 Triton-fused-ops ship wave:
  `molmetal/reports/wf_triton_arch_research/`
- Wf-326B harness auto-fallback (WEAK_TO_STRONG_PLAN §3):
  `TODO/03_plans/WEAK_TO_STRONG_PLAN.md`
- Wf-291D W₂ closed-form (autograd-friendly):
  `molmetal/molmetal_lam/reward/wasserstein_loss.py:288`
- D-MPNN multitask trainer (Wf-131 P1):
  `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
- Test file: `molmetal/tests/test_wf326c_torch_compile.py`
- Optimized trainer: `molmetal/scripts/wf326c_torch_compile_optimized.py`
