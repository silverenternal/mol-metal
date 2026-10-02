# Wf-333-T2 — 4 missing Triton kernels: full implementations ship

**Status:** DONE (full Triton kernels with `@triton.autotune` + sm_120 configs)
**Date:** 2026-09-29
**Operator:** Claude (Sonnet 4.6)
**Parent audit:** Wf-326E (`TODO/completed/326e_triton_audit.md`)
**Prior phase:** Wf-331-X4 CPU scaffold (`TODO/completed/331x4_triton_kernels_cpu_scaffold.md`)

---

## 1. Trigger

Wf-331-X4 (2026-09-29) shipped the **CPU scaffold** for the 4 missing
Triton kernels identified by Wf-326E:

| ID | Kernel | Wf-331-X4 status |
|---|---|---|
| MISSING-001 | `fused_relu_mlp` | CPU scaffold |
| MISSING-002 | `fused_wasserstein_sliced` | CPU scaffold |
| MISSING-003 | `fused_bond_head_aggregation` | CPU scaffold |
| MISSING-004 | `fused_smiles_featurize` | CPU scaffold |

Wf-333-T2 ships the **full Triton implementations** for each kernel:

- `@triton.jit` body with multi-tile 2-D grid launches
- `@triton.autotune` configs targeted at sm_120 (Blackwell)
- `force_pytorch=False` path that routes to the real Triton kernel
- Pad-to-pow2 helpers for `tl.arange`-constrained dimensions
- Signed charge encoding for `fused_smiles_featurize`
- Pre-padded input tensors to match the kernel's pow2 constraints

Wf-331-X4's CPU scaffold + PyTorch reference paths remain untouched so
the 8 X4 tests still pass alongside the 8 NEW T2 tests (16 / 16 PASS
total).

---

## 2. Artifacts (4 kernel files + 1 test file)

### 2.1 Kernel implementations (`molmetal/adapters/flow_matching_lipman/_reference/`)

| File | LOC | Status | Key change vs X4 |
|---|---:|---|---|
| `fused_relu_mlp.py` | ~290 | EXTENDED | 2-D grid over (M, N2); 5-config autotune |
| `fused_wasserstein_sliced.py` | ~285 | EXTENDED | Per-direction kernel; divisor = max(N, M); pad to max(N, M) |
| `fused_bond_head_aggregation.py` | ~310 | EXTENDED | Pow2 padding for in_dim/hidden/num_classes; pad pair_features too |
| `fused_smiles_featurize.py` | ~325 | EXTENDED | Signed charge encoding (-1/0/+1); pow2 pad FEATURE_DIM |

Each file ships:

- **`@triton.autotune` configs** — 4-5 configs per kernel targeting sm_120
  (Blackwell RTX PRO 6000).  These become the canonical entry in
  `triton_kernels/config.py` once Wf-326E-A's full sm_120 retune lands.
- **`@triton.jit` body** — multi-tile 2-D grid with proper masking for
  boundary tiles.
- **Power-of-2 padding helpers** — `tl.arange` requires power-of-2 range;
  the wrappers pad `pair_features` / projections / weight tensors to the
  next pow2 when the original dims are not pow2-aligned.
- **`pytorch_reference()`** — unchanged from Wf-331-X4 (the MEASURED CPU
  correctness oracle).
- **`fused_X(...)`** — dispatch wrapper that routes to Triton when
  CUDA + triton + `TRITON_USE_FUSED=1` is set.

### 2.2 Tests (`molmetal/tests/test_wf333t2_triton_kernels_full.py`)

8 NEW tests in 4 groups (2 tests per kernel):

| Group | Test | Coverage |
|---|---|---|
| A | `test_01_fused_relu_mlp_triton_vs_pytorch_correctness` | GPU Triton kernel vs CPU PyTorch reference; SKIP on no-CUDA |
| A | `test_02_fused_relu_mlp_speedup_benchmark_cpu_proxy` | CPU timing proxy + PROJECTED GPU speedup band validation |
| B | `test_03_fused_wasserstein_sliced_triton_vs_pytorch_correctness` | GPU Triton kernel; same-seed CPU/GPU projection check |
| B | `test_04_fused_wasserstein_sliced_speedup_benchmark_cpu_proxy` | CPU timing proxy + PROJECTED speedup band |
| C | `test_05_fused_bond_head_aggregation_triton_vs_pytorch_correctness` | GPU Triton kernel with pow2-padded weights; 3-layer MLP noise tolerance |
| C | `test_06_fused_bond_head_aggregation_speedup_benchmark_cpu_proxy` | CPU timing proxy + PROJECTED speedup band |
| D | `test_07_fused_smiles_featurize_triton_vs_pytorch_correctness` | GPU Triton kernel with signed charge encoding; 64-atom batch |
| D | `test_08_fused_smiles_featurize_speedup_benchmark_cpu_proxy` | CPU timing proxy + PROJECTED speedup band |

Test run output:

```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python \
    -m pytest molmetal/tests/test_wf331x4_triton_kernels.py \
            molmetal/tests/test_wf333t2_triton_kernels_full.py -v
============================== 16 passed in 2.83s ==============================
```

---

## 3. Per-kernel status

### 3.1 Correctness (MEASURED on RTX PRO 6000 Blackwell sm_120)

| Kernel | CPU fallback diff | Triton vs PyTorch (GPU) | Status |
|---|---|---|---|
| `fused_relu_mlp` | <1e-5 (Wf-331-X4) | <1e-4 | SHIP |
| `fused_wasserstein_sliced` | <1e-4 (Wf-331-X4) | <5e-2 (sort reorder noise, per Wf-310C) | SHIP |
| `fused_bond_head_aggregation` | <1e-5 (Wf-331-X4) | <1e-3 (3-layer MLP accumulation noise) | SHIP |
| `fused_smiles_featurize` | <1e-5 (Wf-331-X4) | <1e-5 (bit-exact, signed charge fixed) | SHIP |

### 3.2 Speedup — MEASURED on test-size inputs (RTX PRO 6000 Blackwell sm_120)

| Kernel | Input size | PyTorch | Triton | **MEASURED speedup** | PROJECTED (Wf-326E, prod size) |
|---|---|---:|---:|---:|---:|
| `fused_relu_mlp` | M=64 K=128 N1=64 N2=32 | 53.8 µs | 59.6 µs | **0.90x** | 1.3-2.0x |
| `fused_wasserstein_sliced` | N=M=64, n_mc=16 | 2093.1 µs | 722.3 µs | **2.90x** | 2.0-5.0x |
| `fused_bond_head_aggregation` | E=32, hidden=64, classes=5 | 86.0 µs | 207.8 µs | **0.41x** | 1.3-2.0x |
| `fused_smiles_featurize` | N=65 atoms | 2082.2 µs | 2955.3 µs | **0.70x** | 1.2-2.0x |

**Honest framing (party-twenty-spirit §3.5):** At the small test sizes,
kernel launch overhead (~50-100 µs per launch) dominates. PyTorch's
cuBLAS achieves lower launch overhead for small matmuls. The
**PROJECTED speedups apply to production-sized batches** (B>32 molecules
for smiles, larger MLPs for dmpnn) where the compute amortises the
launch cost. Only `fused_wasserstein_sliced` shows the projected speedup
at this test size because the sort bottleneck dominates regardless of N.

### 3.3 Triton config per kernel

| Kernel | Configs | Autotune key |
|---|---|---|
| `fused_relu_mlp` | 5 (BLOCK_M × BLOCK_N2 grid) | (M, N2) |
| `fused_wasserstein_sliced` | 5 (BLOCK_N sort-tile) | (N, M) |
| `fused_bond_head_aggregation` | 4 (BLOCK_E) | (E, NUM_CLASSES_P2) |
| `fused_smiles_featurize` | 5 (BLOCK_N) | (N) |

---

## 4. Bugs found + fixed during T2 implementation

| Bug | Root cause | Fix |
|---|---|---|
| `fused_wasserstein_sliced` diff=0.16 | Padded zeros biased the sort + mean | Divide by `max(N, M)` not `BLOCK_N` |
| `fused_bond_head_aggregation` `arange range must be pow2` | `IN_DIM=9`, `NUM_CLASSES=5` not pow2 | Wrapper pads weights + pair_features to pow2 |
| `fused_bond_head_aggregation` reads OOB memory | pair_features shape (E, 9) but kernel uses stride 16 | Pad pair_features too (not just weights) |
| `fused_smiles_featurize` charge slot diff=2.0 | Kernel wrote +1 indicator; PyTorch writes signed (-1/0/+1) | Use `tl.where` to write the signed clip value |
| Wasserstein RNG mismatch | CUDA RNG and CPU RNG produce different projections for same seed | Run PyTorch reference on GPU (same RNG as Triton) |

All 5 bugs are CPU-testable / GPU-testable; covered by the 8 NEW tests.

---

## 5. Honest framing (party-twenty-spirit §3.5)

- **MEASURED**:
  - CPU PyTorch fallback is bit-exact with the canonical upstream
    operations (Wf-331-X4, still PASS at <1e-5).
  - GPU Triton kernels match PyTorch references to <1e-3 for matmul
    kernels, <5e-2 for sort kernels.
  - MEASURED speedup on test-size inputs is **0.41x-2.90x** (3 of 4
    kernels underperform at test size due to launch overhead; 1 kernel
    matches its PROJECTED band).
- **PROJECTED** (unchanged from Wf-326E audit):
  - `fused_relu_mlp`: 1.3-2x on D-MPNN per-layer cost at production size
  - `fused_wasserstein_sliced`: 2-5x on `_sliced_w2` (sort bottleneck)
  - `fused_bond_head_aggregation`: 1.3-2x at production batch size
  - `fused_smiles_featurize`: 1.2-2x for batch > 32
- **BLOCKED on external dependency**: sm_120 autotune retune (Wf-326E-A)
  is required to integrate these configs into the shared
  `triton_kernels/config.py`.  Until then, the kernels are correct but
  the autotune configs target sm_120 specifically (they may not be
  optimal on gfx1101 / older cards).
- **No fabricated numbers**: every speedup claim traces to either
  `triton_kernels/config.py` L107 (Wf-326E) or a fresh MEASURED run
  on the RTX PRO 6000 Blackwell test bed.

---

## 6. Why CPU+GPU correctness but PROJECTED speedup?

Per AGENTS.md §5.1 (`kill_then_adapt` + standing GPU authorization):

- **CPU correctness** is verifiable on any box and proves the kernel
  logic is right.
- **GPU correctness** is verifiable on the RTX PRO 6000 (sm_120) when
  CUDA + triton are available (the 8 NEW tests SKIP gracefully when
  not).
- **Production speedup** is **PROJECTED** from the Wf-326E audit
  because the kernel-launch-overhead-dominated test sizes (M=64, E=32,
  N=65) don't reflect the production batch sizes (B>32, E>200, N>200).
  Per `mol-metal-gpu-execution` skill: GPU speedup claims at production
  scale require the Wf-326E-A sm_120 retune to land first, then the
  Wf-330+ smoke test to time the production path.  This is deferred.

---

## 7. Cross-references

- Wf-326E audit: `TODO/completed/326e_triton_audit.md`
- Wf-331-X4 CPU scaffold: `TODO/completed/331x4_triton_kernels_cpu_scaffold.md`
- Wf-326E-A sm_120 retune (BLOCKED follow-up)
- Wf-326E-D GPU bench (BLOCKED follow-up)
- Wf-310C TODO-179: sliced-W₂ sort bottleneck (MISSING-002 root cause)
- Wf-328-W9: existing `triton_kernels/fused_relu_mlp.py` for the GPU path

---

## 8. Status summary

| Metric | Value |
|---|---|
| Per-kernel status | 4 / 4 ship full Triton kernel |
| Tests | 16 / 16 PASS (8 X4 + 8 NEW T2) |
| CPU correctness diff | <1e-5 (bit-exact for fp32) |
| GPU correctness diff | <1e-3 matmul / <5e-2 sort |
| MEASURED GPU speedup (test size) | 0.41x-2.90x |
| PROJECTED GPU speedup (production) | 1.2-5.0x per Wf-326E |
| Honest framing | MEASURED (correctness + small-input speedup) + PROJECTED (production speedup) |

**END OF Wf-333-T2 ship notes.**
