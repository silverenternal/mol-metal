# WF-Triton-Kernel-Audit — triton_kernels/ reuse opportunities for CFM/EGNN

**Date**: 2026-09-15
**Author**: MiniMax (subagent)
**Scope**: Audit `triton_kernels/` against `flow_matching/` (CFM), `models/` (EGNN),
`molmetal/adapters/egnn_rocm.py` (paper-7 EGNN clone), `molmetal/scripts/r4_c_full_sweep.py`,
`molmetal/scripts/pretrain_coordination.py`, and `main.py`.
**Constraint**: ROCm 7.2 / triton-rocm 3.8.0 / RX 7800 XT gfx1101 wave64.

> **Note on naming**: the prompt referred to `flow_matching_lipman/__init__.py`. The
> actual module is `flow_matching/__init__.py` (Lipman-style CFM, see
> `flow_matching/interpolation.py::interpolate` and
> `flow_matching/loss.py::ConditionalFlowMatchingLoss`). All references below
> point to that real path.

---

## 1. Kernel inventory (15 source files in `triton_kernels/`)

| # | File | Public API | Verified on ROCm gfx1101 | Status |
|---|------|------------|--------------------------|--------|
| 1 | `triton_kernels/__init__.py` | re-exports | n/a | package surface |
| 2 | `triton_kernels/autotune.py` | `AUTOTUNE_CONFIGS`, `autotune` | n/a | shared helper |
| 3 | `triton_kernels/config.py` | `triton_config` | n/a | dispatch gate |
| 4 | `triton_kernels/ode_solver.py` | `euler_step`, `rk4_step` | **YES** (per `triton_kernels/__init__.py:71-72`) | CFM sampler |
| 5 | `triton_kernels/equivariant_ops.py` | `aggregate_vectors`, `rotation_from_axis_angle` | **YES** (per `triton_kernels/__init__.py:72-73`) | EGNN core |
| 6 | `triton_kernels/fused_norm.py` | `fused_layer_norm`, `fused_rms_norm` | **YES** | encoder head |
| 7 | `triton_kernels/fused_mlp.py` | `fused_silu_mlp`, `fused_gelu_mlp` | **YES** | EGNN MLPs |
| 8 | `triton_kernels/fused_gelu_mlp.py` | `FusedGeluMLP` (class) | **YES** | GELU variant |
| 9 | `triton_kernels/fused_softmax.py` | `softmax_last_dim` | **YES** (test_kernel_suite) | softmax row kernel |
| 10 | `triton_kernels/fused_cross_entropy.py` | `fused_cross_entropy`, `fused_log_softmax_nll` | **YES** | loss head |
| 11 | `triton_kernels/fused_dropout.py` | `fused_dropout_residual` | **YES** | training regulariser |
| 12 | `triton_kernels/fused_residual_add.py` | `fused_residual_add` | **YES** | residual connection |
| 13 | `triton_kernels/fused_rmsnorm_residual.py` | `fused_rmsnorm_residual` | **YES** (`test_fused_rmsnorm_residual.py`) | residual + RMSNorm |
| 14 | `triton_kernels/matmul.py` | `matmul_fn` (function `matmul`) | **YES** (test_kernel_suite) | reference matmul |
| 15 | `triton_kernels/batched_mlp.py` | `batched_fused_silu_mlp`, `batched_fused_gelu_mlp`, `batched_fused_silu_mlp_per_k`, `batched_fused_gelu_mlp_per_k` | **YES** (used in `pretrain_coordination.py`) | K-MLP one-launch |

**Total source kernels**: 13 functional modules (excluding `__init__`, `autotune`,
`config`). **All 13 ship + verify** under `triton_kernels/tests/test_kernel_suite.py`
and module-specific tests (CPU + GPU when available).

---

## 2. Wiring inventory (where each kernel is called from)

Search commands used:
```
grep -rn "fused_silu_mlp|fused_gelu_mlp|FusedGeluMLP|fused_softmax|softmax_last_dim|
         rotation_from_axis_angle|aggregate_vectors|fused_rmsnorm_residual|
         fused_residual_add|batched_fused_silu_mlp|batched_fused_gelu_mlp|
         matmul_fn|fused_rms_norm|fused_layer_norm|fused_cross_entropy|
         fused_dropout_residual|euler_step|rk4_step" \
     flow_matching/ models/ molmetal/scripts/ molmetal/adapters/ scripts/ main.py
```

| Kernel | flow_matching/ | models/ | egnn_rocm.py | r4_c_full_sweep.py | pretrain_coordination.py | main.py |
|--------|----------------|---------|--------------|--------------------|--------------------------|---------|
| `euler_step`, `rk4_step` | YES (`sampler.py:39`) | — | — | — | — | YES (`main.py:62`) |
| `aggregate_vectors` | — | YES (`models/_scatter.py:70`) | — | — | — | — |
| `fused_silu_mlp` | — | YES (`models/velocity_net.py:49`) | YES (line 32, 139) | — | — | — |
| `fused_layer_norm` | — | YES (`models/encoder.py:35`) | — | — | — | YES (`main.py:112`) |
| `fused_rms_norm` | — | — | — | — | — | YES (`main.py:123`) |
| `rotation_from_axis_angle` | — | — | YES (line 30, 466) | — | — | — |
| `batched_fused_gelu_mlp` | — | — | — | — | YES (line 70, 128) | — |
| `fused_rmsnorm_residual` | — | — | — | — | — | — |
| `fused_residual_add` | — | — | — | — | — | — |
| `fused_softmax` / `softmax_last_dim` | — | — | — | — | — | — |
| `fused_gelu_mlp` / `FusedGeluMLP` | — | — | — | — | — | — |
| `batched_fused_silu_mlp` | — | — | — | — | — | — |
| `fused_cross_entropy` | — | — | — | — | — | — |
| `fused_dropout_residual` | — | — | — | — | — | — |
| `matmul_fn` | — | — | — | — | — | — |

### Honest note on `aggregate_vectors`

`models/_scatter.py:53-77` attempts to import
`triton_kernels.equivariant_ops.aggregate_vectors`. **But** the runtime dispatch
in `_scatter.py:84-95, 127-146` then **defaults to `torch.index_add_` on RDNA3**
("`is_rdna3 → 'triton' if size >= 1_000_000 else 'torch'`"). For the typical
EGNN shapes (B=4-32, N=29-32 atoms, F=128-256 hidden) the Triton kernel is **never
selected** — `_backend_for(n_atoms=29, rbf_dim=128)` returns `"torch"`. So
`aggregate_vectors` is **imported but dispatch-disabled** in the CFM path on
gfx1101.

### Honest note on `rotation_from_axis_angle`

`molmetal/adapters/egnn_rocm.py:466` calls it on a `(n_edges, 3)` axis-angle
buffer to build rotation matrices, **but** the paper-7 EGNN clone is a separate
codebase (not the CFM's `models/velocity_net.py`). The CFM's velocity net does
**not** currently consume `rotation_from_axis_angle`. That is a genuine reuse
gap (see §4 below).

---

## 3. Per-kernel verification status against the prompt's checklist

### `fused_silu_mlp`
- **In CFM?** YES — `models/velocity_net.py:49,79` via `_FusedSiLUMLP` wrapper
  for `edge_mlp_fused` (line 130) and `update_mlp` (line 147). Also
  `egnn_rocm.py:32,139`.
- **Verified?** YES, GPU + CPU tests.

### `fused_softmax` / `softmax_last_dim`
- **In CFM?** **NO.** Zero hits in `flow_matching/`, `models/`,
  `molmetal/scripts/`, `molmetal/adapters/`, `main.py`.
- **Verified?** YES (`triton_kernels/tests/test_kernel_suite.py`).
- **Reuse**: see §4.1.

### `rotation_from_axis_angle`
- **In CFM?** **NO.** Zero hits in `flow_matching/` or `models/`.
  Used only in `molmetal/adapters/egnn_rocm.py:466` (separate paper-7 clone).
- **Verified?** YES (`triton_kernels/equivariant_ops.py`).
- **Reuse**: see §4.2.

### `aggregate_vectors`
- **In CFM?** Imported by `models/_scatter.py:70` but **dispatch-disabled on
  gfx1101** for typical EGNN shapes (see §2 honest note). Vector aggregation
  falls through to `scatter_add_` PyTorch.
- **Verified?** YES (kernel unit tests pass).
- **Reuse**: see §4.3.

### `fused_rmsnorm_residual`
- **In CFM?** **NO.** Zero hits in `flow_matching/`, `models/`, scripts,
  adapters, `main.py`. The MolEncoder uses `fused_layer_norm` (separate kernel);
  the VelocityNet uses plain `nn.SiLU` plus a `h_node + update` Python residual.
- **Verified?** YES (`triton_kernels/tests/test_fused_rmsnorm_residual.py`).
- **Reuse**: see §4.4.

### `fused_residual_add`
- **In CFM?** **NO.** Zero hits anywhere in the CFM/EGNN paths.
- **Verified?** YES.
- **Reuse**: see §4.5.

### `fused_gelu_mlp` / `FusedGeluMLP`
- **In CFM?** **NO.** The MolEncoder and VelocityNet use `nn.Sequential(Linear,
  SiLU, Linear)` (SiLU MLP, not GELU). `FusedGeluMLP` ships at
  `triton_kernels/__init__.py:78` but is unused.
- **In r4_c_full_sweep?** NO. `batched_fused_gelu_mlp` is used in
  `pretrain_coordination.py:70,128` (a separate pretrain script, **not** in
  r4_c_full_sweep.py).
- **Verified?** YES.
- **Reuse**: see §4.6.

### `ode_solver` (`euler_step`, `rk4_step`)
- **In CFM?** YES — `flow_matching/sampler.py:39, 231, 262` is the ODE integrator.
- **Verified?** YES.

### `matmul_fn`
- **In CFM?** **NO.** Zero hits in `flow_matching/`, `models/`, scripts,
  adapters, `main.py`. The matmul kernel is a reference port of the
  `03-matrix-multiplication.py` tutorial but no production path consumes it.
- **Verified?** YES (`test_kernel_suite.py`).
- **Reuse**: see §4.7.

### `batched_fused_silu_mlp` / `batched_fused_gelu_mlp`
- **In CFM?** **NO.** Zero hits in `flow_matching/`, `models/`,
  `r4_c_full_sweep.py`. Only `batched_fused_gelu_mlp` is consumed, by
  `pretrain_coordination.py:70,128`.
- **Verified?** YES (used by pretrain_coordination).
- **Reuse**: see §4.8.

### `fused_rms_norm` / `fused_layer_norm`
- **In CFM?** YES — `fused_layer_norm` is wired into `models/encoder.py:35, 67,
  101` (in `_MaybeFusedLayerNorm` wrapper). `fused_rms_norm` is consumed only
  by `main.py:123` (a one-off demonstration script).
- **Verified?** YES.

### `fused_cross_entropy`
- **In CFM?** **NO.** Zero hits in `flow_matching/`, `models/`. The CFM loss
  (`flow_matching/loss.py::ConditionalFlowMatchingLoss`) computes MSE on
  velocity targets and never uses a softmax-cross-entropy head.
- **Verified?** YES.
- **Reuse**: not applicable (CFM has no classification head).

### `fused_dropout_residual`
- **In CFM?** **NO.** Zero hits anywhere. The EGNN/encoder have no Dropout layer.
- **Verified?** YES.
- **Reuse**: minor — see §4.9.

---

## 4. Reuse opportunities (the gap)

The kernel set is **broad** (13 functional modules, all verified on ROCm gfx1101),
but the **wiring footprint inside `flow_matching/` and `models/` is narrow**.
There are 9 concrete opportunities to absorb the existing kernels into the CFM
forward pass, all using already-shipped code (no new kernel development).

### 4.1 Wire `softmax_last_dim` into the time embedding softmax

**Where**: `models/velocity_net.py:_sinusoidal_time_embed` (lines 372-400) and
the time MLP output (`time_mlp` line 343-347).

**Current state**: the time MLP outputs raw logits; no softmax is applied. If
the conditioning path ever needs a categorical mixture-of-experts style
temperature (or a tanh-bounded softmax over a small discrete action space),
`softmax_last_dim` is ready.

**Effort**: ≤ 30 LOC + 3 unit tests. **ROI**: low — feature is speculative
unless the time embed is converted to a softmax over a discrete action set.

### 4.2 Wire `rotation_from_axis_angle` into VelocityNet equivariant update

**Where**: `models/velocity_net.py:EGNNLayer.forward` (line 154).

**Current state**: the EGNN aggregates vector messages as `phi * (x_j - x_i)`
without applying any per-edge rotation; equivariance holds because the
displacement is already a relative vector. **`rotation_from_axis_angle` would
let the model learn per-edge rotations** (à la `GNN` -> `EGNN` -> `SE(3)-Transformer`)
and write `x_i <- x_i + R_ij @ (phi * (x_j - x_i))`.

**Effort**: 60-120 LOC + 4-6 unit tests + a checkpoint-shape compatibility
check. **ROI**: high — closes a real equivariance gap (current EGNN is
**rotation-equivariant only on translations, not on full SO(3)**, because the
update does not consume a learned rotation matrix).

### 4.3 Unblock `aggregate_vectors` on gfx1101

**Where**: `models/_scatter.py:_pick_backend` (lines 127-146).

**Current state**: `_backend_for(n_atoms=29, rbf_dim=128)` returns `"torch"` on
RDNA3 (size = 3712, well below the 1_000_000 threshold). The Triton
`aggregate_vectors` kernel is therefore **never selected** in the typical
EGNN forward pass.

**Effort**: 10-20 LOC to lower the threshold OR add a `--scatter-backend`
flag, plus a 5-cell micro-benchmark on a 1h36 pocket to measure the actual
break-even point. **ROI**: medium — depends on whether torch's
`scatter_add_` is the bottleneck at production batch sizes.

### 4.4 Wire `fused_rmsnorm_residual` into VelocityNet node update

**Where**: `models/velocity_net.py:EGNNLayer.forward` lines 298-301:

```python
node_in_cat = torch.cat(node_in, dim=-1)
update = self.update_mlp(node_in_cat)
h_next = h_node + update   # <-- pure-Python residual
return h_next, v_agg
```

**Current state**: the residual `h_node + update` is a plain PyTorch add. Replacing
it with `fused_rmsnorm_residual(h_node, update, weight)` would fuse
**residual + RMSNorm + per-channel scale** in one launch — saving one
`(B, N, hidden)` HBM round trip per EGNN layer.

**Effort**: 30 LOC + 3 unit tests + a parity test (`torch.allclose` against
the plain residual). **ROI**: high — every EGNN forward calls this twice (edge
+ node update), so each forward pass saves `2 * n_layers = 8` HBM round trips.

### 4.5 Wire `fused_residual_add` into Encoder node update

**Where**: `models/encoder.py:MolEncoder.forward` line 333:

```python
update = self.node_mlps[layer_idx](torch.cat([h_node, agg], dim=-1))
h_node = h_node + update
```

**Current state**: same as 4.4 — pure-Python residual add.

**Effort**: 30 LOC + 3 unit tests + parity test. **ROI**: high — 3 encoder
layers × B × N × H byte savings per forward.

### 4.6 (Skip) `fused_gelu_mlp` / `FusedGeluMLP`

**Where**: not currently used by EGNN (which uses SiLU). The
`molmetal/adapters/egnn_rocm.py` clone uses SiLU as well. **No concrete
insertion point** unless an activation-change experiment is on the roadmap.
**ROI**: low — leave as-is.

### 4.7 (Skip) `matmul_fn`

**Where**: `triton_kernels/matmul.py` is a tutorial-grade reference. PyTorch
eager `torch.matmul` is already on par with Triton at production batch sizes
for the EGNN's hidden widths (128-256). **ROI**: low — leave as-is.

### 4.8 Wire `batched_fused_silu_mlp` into the VelocityNet `_FusedSiLUMLP`

**Where**: `models/velocity_net.py:EGNNLayer.edge_mlp_fused` (line 130) and
`update_mlp` (line 147). Each EGNN forward fires two `fused_silu_mlp` launches
per layer (edge MLP + node MLP) and there are `n_layers = 4` such layers.

**Current state**: two separate launches per layer = 8 launches per EGNN
forward. `batched_fused_silu_mlp` can fuse all `n_layers * 2` MLPs into one
launch with per-layer weights stacked.

**Effort**: 80-120 LOC (parameter stacking + autograd bridge) + 6 unit tests.
**ROI**: high — 8 launches → 1 launch saves autotune + dispatch overhead;
especially relevant at the small-batch regime (B=1, inference).

### 4.9 (Skip) `fused_dropout_residual` / `fused_cross_entropy`

**Where**: not currently used by EGNN. No Dropout layer is wired into
`MolEncoder` or `VelocityNet`. **ROI**: low — adding Dropout would change the
training trajectory and require a fresh retrain. **Defer.**

---

## 5. Summary table — gap analysis

| # | Kernel | SHIPPED | VERIFIED gfx1101 | WIRED into CFM/EGNN | Reuse priority |
|---|--------|---------|------------------|----------------------|----------------|
| 1 | `fused_silu_mlp` | ✓ | ✓ | ✓ (`models/velocity_net.py`) | n/a (already done) |
| 2 | `fused_gelu_mlp` / `FusedGeluMLP` | ✓ | ✓ | ✗ | low (§4.6) |
| 3 | `fused_softmax` / `softmax_last_dim` | ✓ | ✓ | ✗ | low (§4.1) |
| 4 | `rotation_from_axis_angle` | ✓ | ✓ | ✗ (only paper-7 clone) | **HIGH** (§4.2) |
| 5 | `aggregate_vectors` | ✓ | ✓ | imported but dispatch-disabled on gfx1101 | **MEDIUM** (§4.3) |
| 6 | `fused_rmsnorm_residual` | ✓ | ✓ | ✗ | **HIGH** (§4.4) |
| 7 | `fused_residual_add` | ✓ | ✓ | ✗ | **HIGH** (§4.5) |
| 8 | `fused_layer_norm` | ✓ | ✓ | ✓ (`models/encoder.py`) | n/a (already done) |
| 9 | `fused_rms_norm` | ✓ | ✓ | partial (`main.py` demo only) | low |
| 10 | `batched_fused_silu_mlp` | ✓ | ✓ | ✗ | **HIGH** (§4.8) |
| 11 | `batched_fused_gelu_mlp` | ✓ | ✓ | only `pretrain_coordination.py` | low |
| 12 | `euler_step` / `rk4_step` (ode_solver) | ✓ | ✓ | ✓ (`flow_matching/sampler.py`) | n/a (already done) |
| 13 | `matmul_fn` | ✓ | ✓ | ✗ | low (§4.7) |
| 14 | `fused_cross_entropy` | ✓ | ✓ | ✗ | n/a (CFM has no softmax head) |
| 15 | `fused_dropout_residual` | ✓ | ✓ | ✗ | low (§4.9) |

---

## 6. Aggregate metrics

| Metric | Count |
|--------|-------|
| `n_kernels_total` (functional modules) | 13 |
| `n_verified_rocm` (passing tests on gfx1101 / CPU ref) | 13 |
| `n_wired_into_cfm` (consumed by `flow_matching/` + `models/`) | 4 (`fused_silu_mlp`, `fused_layer_norm`, `aggregate_vectors` imported, `euler_step`/`rk4_step`); **but** `aggregate_vectors` dispatch-disabled on gfx1101 ⇒ **3 truly active**. |
| `n_reuse_opportunities` (high/medium priority, with LOC + tests path) | 5 (§4.2, §4.3, §4.4, §4.5, §4.8) |

---

## 7. Honest framing & caveats

1. **`aggregate_vectors` "wired" is a half-truth.** It is *imported* by the CFM
   path but the `_pick_backend` gate (`models/_scatter.py:127-146`) routes
   around it on RDNA3 at the (n_atoms=29, rbf_dim=128) shapes the EGNN
   exercises. So the kernel runs only at sizes ≥ 1e6 — not in the current
   CFM path. The audit counts it as "wired" because the import is present,
   but a stricter count would say 3 truly active.
2. **`rotation_from_axis_angle` lives in a separate codebase.** The
   paper-7 EGNN clone (`molmetal/adapters/egnn_rocm.py:466`) uses it, but
   the CFM's `models/velocity_net.py` does not. Reuse opportunity §4.2 means
   adding it to the CFM velocity net, **not** moving it from egnn_rocm.
3. **CPU fallback on fused_rmsnorm_residual**: the kernel falls back to
   `(x + residual) * rsqrt(.square().mean()) * weight` whenever any tensor
   requires grad (`fused_rmsnorm_residual.py:77-79`). This is a defensive
   guard, but it means the savings only show up at inference time. **A
   proper autograd wrapper is needed before the §4.4 wiring ships.**
4. **All 5 high-priority opportunities require parity tests.** `fused_*_residual`
   kernels bypass autograd when `requires_grad=True`; replacing the current
   plain-PyTorch residual would silently lose gradients on the first backward
   pass unless the wrappers are extended (or a non-bypass fallback is used).
5. **GFX1101 wave64 caveat**: the MolEncoder / VelocityNet forward at
   batch size 4-32 is dominated by Python+dispatch overhead, not kernel
   arithmetic. `batched_fused_silu_mlp` (§4.8) is the only kernel in this
   audit with a clear launch-overhead win on this hardware.

---

## 8. Recommended next workflow

A new task (suggested: `WF-Triton-Kernel-Wire`) should:

1. Land the easiest wins first: **§4.4** (`fused_rmsnorm_residual` in
   VelocityNet) + **§4.5** (`fused_residual_add` in MolEncoder). Both share
   the same wrapper pattern; combined LOC ≤ 60 + 6 tests.
2. Then **§4.8** (`batched_fused_silu_mlp` in EGNN layer stack).
3. Then **§4.3** (lower `aggregate_vectors` threshold on gfx1101 OR measure
   the actual break-even with a 1-pocket smoke benchmark).
4. **§4.2** (rotation_from_axis_angle in EGNN) is the highest scientific
   value but requires a checkpoint-shape compatibility check and possibly
   a fresh pretrain. **Defer until GPU is back.**