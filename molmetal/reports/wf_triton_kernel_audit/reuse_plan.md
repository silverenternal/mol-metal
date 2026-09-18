# WF-Triton-Kernel-Audit Reuse Plan

**Date:** 2026-09-15
**Author:** audit (WF-Triton-Kernel-Audit)
**Status:** HONEST-FRAME — kernels identified, sites located, expected gains are engineering estimates from autotune-model ratios (1.3–1.6x for MLP, 1.2–1.4x for norm, 1.1–1.25x for residual), NOT measured on this host. Verification requires GPU; current host has HSA init failure (cuda_available=False per WF-GPU-Auto-Recover 2026-09-15).

---

## 1. Audit scope & current state

`/home/hugo/codes/try_triton_on_rocm/triton_kernels/` ships 13 modules (see `__init__.py`):

| Module | Public kernel | GPU-verified site | Currently wired into CFM/EGNN? |
| --- | --- | --- | --- |
| `fused_mlp.py` | `fused_silu_mlp`, `fused_gelu_mlp` | triton_kernels/tests | **YES** — `models/velocity_net.py:49,79` (edge_mlp, update_mlp) AND `molmetal/adapters/egnn_rocm.py:32,139` (via `_MaybeFusedSiLUMLP`) |
| `fused_norm.py` | `fused_layer_norm`, `fused_rms_norm` | triton_kernels/tests | **PARTIAL** — `models/encoder.py:35` uses `fused_layer_norm` only (no RMSNorm); `velocity_net.py` and `conditioner.py` use NO fused norm |
| `fused_softmax.py` | `softmax_last_dim` | triton_kernels/tests | **NO** — no `softmax` call in `models/` or `flow_matching/` (grep returns 0) |
| `fused_cross_entropy.py` | `fused_cross_entropy`, `fused_log_softmax_nll` | triton_kernels/tests | **NO** — no `nn.CrossEntropyLoss` or `F.cross_entropy` in models/flow_matching |
| `fused_dropout.py` | `fused_dropout_residual` | triton_kernels/tests | **NO** — `models/_scatter.py` and `egnn_rocm.py` use pure PyTorch dropout / residual |
| `fused_residual_add.py` | `fused_residual_add` | triton_kernels/tests | **NO** — `velocity_net.py:301` (h_next = h_node + update), `encoder.py:333` (h_node = h_node + update), `egnn_rocm.py` (multiple sites) all use plain `+` |
| `fused_rmsnorm_residual.py` | `fused_rmsnorm_residual` | triton_kernels/tests | **NO** |
| `batched_mlp.py` | `batched_fused_silu_mlp`, `batched_fused_gelu_mlp{,_per_k}` | triton_kernels/tests (single) | **NO** |
| `equivariant_ops.py` | `aggregate_vectors`, `rotation_from_axis_angle` | triton_kernels/tests | **PARTIAL** — `models/_scatter.py:70` imports `aggregate_vectors`; `egnn_rocm.py:30` imports `rotation_from_axis_angle` (used). EGNN coord update at `molmetal/adapters/egnn_rocm.py:264` (`F.silu(self.mlp(mlp_in))`) is PyTorch |
| `matmul.py` | `matmul_fn` | triton_kernels/tests | **NO** — training uses `torch.matmul` everywhere |
| `ode_solver.py` | `euler_step`, `rk4_step` | triton_kernels/tests | **YES** — `flow_matching/sampler.py:39` imports both |
| `autotune.py` | `AUTOTUNE_CONFIGS`, `autotune_fn` | n/a (config) | **YES** (transitively, via every kernel) |
| `config.py` | `triton_config` | n/a (config) | **YES** (singleton, dispatch gate) |

**Coverage verdict (honest):**
- Already wired (1): `fused_silu_mlp` × 2 sites + `fused_layer_norm` × 1 site + ODE steppers + `aggregate_vectors` + `rotation_from_axis_angle` + `triton_config`.
- Partial (1): `fused_layer_norm` only; `velocity_net.py` has NO norm between layers.
- Unused (8 modules, 14 kernels): `fused_rms_norm`, `softmax_last_dim`, `fused_cross_entropy`, `fused_dropout_residual`, `fused_residual_add`, `fused_rmsnorm_residual`, `batched_fused_silu_mlp` + `batched_fused_gelu_mlp` (+ per_k variants), `matmul_fn`, plus the GELU sibling `fused_gelu_mlp`.

The user's framing "most are wired into r4_c_full_sweep, not into flow_matching_lipman" is **only half correct**: r4_c_full_sweep.py uses the same `models/velocity_net.py` and `models/encoder.py` (which DO have partial wirings — `_FusedSiLUMLP` + `_MaybeFusedLayerNorm`). The actual gap is **inside the CFM training loop** (encoder MLP, time_mlp, cond_proj, residual adds), not r4_c_full_sweep specifically.

---

## 2. Prioritisation — top 3 kernels to wire immediately

Criteria: **(a) call-site frequency** (how often the kernel would fire per training step), **(b) bandwidth savings** (HBM round trips avoided), **(c) parameter-shape compatibility** (no checkpoint surgery required), **(d) test coverage** (already has a passing triton_kernels/test).

| Rank | Kernel | Sites to wire | Why now |
| --- | --- | --- | --- |
| **1** | `fused_residual_add` | `velocity_net.py:301` (`h_next = h_node + update`), `encoder.py:333` (`h_node = h_node + update`), `egnn_rocm.py` (>=1 site) | **Fires 4 + 3 = 7 times per training step.** Each `+` writes one full (B,N,H) intermediate to HBM. Bandwidth-bound, no shape change, autograd is a plain `add`. |
| **2** | `fused_rmsnorm_residual` (or `fused_rms_norm` + residual) | new `_MaybeFusedRMSNorm` in `models/encoder.py` + new RMSNorm layer in `models/velocity_net.py` between EGNN layers | **Pre-norm transformer pattern absent from current EGNN.** Adding fused RMSNorm between EGNN layers would (a) stabilise training and (b) save another HBM round trip per layer. Currently NO norm between EGNN layers in `velocity_net.py`. |
| **3** | `fused_silu_mlp` for `time_mlp` and `cond_proj` | `velocity_net.py:343-356` (currently `nn.Sequential(Linear, SiLU, Linear)` × 2) | **Trivial wiring — drop in `_FusedSiLUMLP` from `velocity_net.py:52`** (already used at line 130 / 147 for EGNN MLPs). Saves 1 HBM round trip × 2 MLPs × 1 forward per step. Zero risk; identical parameter layout; checkpoint-compatible. |

**Not in top 3 (and why):**
- `fused_gelu_mlp` — no GELU activations in the CFM stack (all SiLU). Replacing SiLU with GELU changes semantics; not worth a behaviour swap.
- `softmax_last_dim` / `fused_cross_entropy` — no softmax / CE in the CFM forward pass. Would only matter for the bond-head classifier (BondOrderHead); audit needed there before recommending.
- `fused_dropout_residual` — no dropout currently in CFM (intentional, all dropout happens in encoder aggregation). Wiring it would CHANGE the training dynamics.
- `batched_fused_silu_mlp` — useful if we ever run K independent EGNNs in parallel; we don't. Defer.
- `matmul_fn` — `torch.matmul` on gfx1101 already routes through rocBLAS; Triton matmul is unlikely to beat it on these shapes. Defer until a benchmark shows otherwise.
- `rotation_from_axis_angle` — already wired (egnn_rocm.py:30); used at the coord-update site. Done.

---

## 3. Integration plan (file:line + changes + tests + expected speedup)

### Kernel #1 — `fused_residual_add` into CFM training step

**Target sites (all GPU-resident, training-time, no eval gate changes needed):**

- `models/velocity_net.py:301` — `h_next = h_node + update` (inside `EGNNLayer.forward`); runs `n_layers` times per forward.
- `models/encoder.py:333` — `h_node = h_node + update` (inside `MolEncoder.forward` message-passing loop); runs `n_layers` times per forward.
- `molmetal/adapters/egnn_rocm.py` — at least one coord-update residual site in `EquivariantGraphConv.forward` (around line 264-275; needs `grep -n 'h.*=.*h.*+\|self.h.*+' egnn_rocm.py` to confirm exact line).

**Changes (single helper, three call-sites):**

1. Add `_MaybeFusedResidualAdd` to `models/velocity_net.py` (next to existing `_FusedSiLUMLP` at line 52). Reuse `triton_config.use_fused_residual_add(n_elem)` (already at `config.py:206`). The wrapper signature is `add(h, update) -> h + update` and is a drop-in for the `+` operator.
2. In `EGNNLayer.forward`, replace `h_next = h_node + update` (line 301) with `h_next = _MaybeFusedResidualAdd.apply(h_node, update)` (or a thin `nn.Module` wrapper that gates on `triton_config.use_fused_residual_add`).
3. In `MolEncoder.forward`, replace `h_node = h_node + update` (line 333) with the same helper.
4. In `egnn_rocm.py`, replace any plain `self.h = self.h + ...` with the helper.

**Tests:**
- `tests/test_velocity_net.py` (or new `tests/test_fused_residual_wiring.py`):
  - **Identity test:** `_MaybeFusedResidualAdd(h, 0)` == `h` (numerically exact).
  - **Shape parity:** output of fused path == output of `h + update` to within `atol=1e-6`.
  - **Backward parity:** `torch.autograd.gradcheck` on `h + update` vs fused path with random tensors of shape `(B=2, N=8, H=64)`.
  - **CPU fallback:** with `triton_config.set_enabled(False)` or `force=True`, the path equals `h + update` exactly.
- Existing test files (`test_velocity_net.py`, `test_encoder.py`, `test_egnn_rocm.py`) must continue to pass unchanged — the change is wire-internal.

**Expected speedup (engineering estimate, NOT measured):**
- For each EGNN layer, the residual add is `(B*N*H)` FP32 elements → at `(B=8, N=64, H=128)` = 65 536 elements × 4 B = 256 KiB per residual. RTX 7800 XT has ~624 GB/s HBM; PyTorch `+` launch overhead ~10 µs per call.
- Replacing with a single Triton launch (one-pass elementwise) cuts launch overhead roughly in half AND avoids the intermediate tensor allocation. Combined effect: **~1.1–1.25x speedup on the residual-add portion of each layer** (5–10% wall-clock on the full training step, since MLP + scatter dominate).
- Across 7 call-sites × 4 EGNN layers, **expected end-to-end training step speedup: ~5–10%** (engineering estimate; will be < measurement noise until GPU recovers).

### Kernel #2 — `fused_rmsnorm_residual` (or plain `fused_rms_norm`) between EGNN layers

**Target site:** `models/velocity_net.py` — insert a `_MaybeFusedRMSNorm(hidden_dim)` after each `EGNNLayer.forward` (before next layer), or equivalently wrap `EGNNLayer.forward` to do `(h_next + residual, RMSNorm(h_next))` as a pre-norm transformer block.

**Changes:**
1. Add `_MaybeFusedRMSNorm(hidden_dim, eps=1e-5)` to `models/encoder.py` (next to existing `_MaybeFusedLayerNorm` at line 98), mirroring the same `triton_config.should_use_fused(x, op="rms_norm")` gate.
2. In `VelocityNet.__init__` (line 360-362), build `self.layer_norms = nn.ModuleList([_MaybeFusedRMSNorm(hidden_dim) for _ in range(n_layers)])`.
3. In `VelocityNet.forward` (line 477-484), apply `h = self.layer_norms[i](h)` between layers. (Equivariance of the coord update is preserved because we normalise the scalar h, not the positions.)
4. Optional Phase-2: also wrap each `EGNNLayer.forward` to use `fused_rmsnorm_residual` directly (saves one extra HBM round trip). Defer until #1 + #3 land.

**Tests:**
- `tests/test_velocity_net.py`:
  - **Mean-1 variance-check** on `RMSNorm(h)` over random `(B=4, N=16, H=64)` input: `out.mean(-1) ≈ 0`, `out.var(-1) ≈ 1`.
  - **Equivariance preservation:** `RMSNorm(rotate(h)) == rotate(RMSNorm(h))` for scalar features (h is invariant → RMSNorm commutes trivially; the test is a regression guard).
  - **Backward parity** (autograd) vs `F.rms_norm` reference.
  - **Eval-mode off** (per `config.py:177` default): RMSNorm falls back to PyTorch unless `TRITON_USE_FUSED_EVAL=1`.

**Expected speedup (engineering estimate, NOT measured):**
- The current `velocity_net.py` has **NO normalisation between EGNN layers** — adding fused RMSNorm is mostly a *training-quality* change (helps convergence / loss-stability, per EGNN ablation literature), not just a speedup.
- Bandwidth-wise: a single RMSNorm over `(B, N, H)` writes HBM twice (read + write) in PyTorch; Triton fuses mean+var+normalize+affine into one pass → **~1.2–1.4x speedup on the norm step** (engineering estimate). Per-layer speedup ~5–8%; full-step ~5–8% since 4 layers × 1 norm each.

### Kernel #3 — `fused_silu_mlp` for `time_mlp` and `cond_proj`

**Target sites:** `models/velocity_net.py:343-358` — `self.time_mlp` and `self.cond_proj` are currently plain `nn.Sequential(Linear, SiLU, Linear)`.

**Changes:**
1. Replace `nn.Sequential(Linear, SiLU, Linear)` with `_FusedSiLUMLP` (already defined at `models/velocity_net.py:52`).
2. Parameter names must remain `time_mlp.0.weight` / `time_mlp.0.bias` / `time_mlp.2.weight` / `time_mlp.2.bias` — **IMPORTANT**: `_FusedSiLUMLP` exposes `.linear1.*` / `.linear2.*`, NOT `.0.*` / `.2.*`. **This is a checkpoint-breaking change** unless we either (a) re-save existing checkpoints with the new param names, or (b) add a `state_dict` rename hook.
3. **Recommendation:** use the `_MaybeFusedSiLUMLP` gating pattern from `egnn_rocm.py:101` instead — it preserves `linear1`/`linear2` names AND falls back to PyTorch when `triton_config.use_fused_mlp(...)` is False. Add `_MaybeFusedSiLUMLP` to `models/velocity_net.py` next to `_FusedSiLUMLP`, and use it for both `time_mlp` and `cond_proj`.

**Tests:**
- `tests/test_velocity_net.py`:
  - **State-dict round-trip:** `_MaybeFusedSiLUMLP(in=64, hidden=128, out=64).load_state_dict(_FusedSiLUMLP(...).state_dict())` succeeds (or vice versa).
  - **Output parity:** `_MaybeFusedSiLUMLP(x)` == `nn.Sequential(Linear, SiLU, Linear)(x)` to `atol=1e-6` when fused is disabled.
  - **Forward in training mode** dispatches to Triton; in eval mode with `TRITON_USE_FUSED_EVAL=0` falls back to PyTorch.

**Expected speedup (engineering estimate, NOT measured):**
- These two MLPs are tiny: `(B, 64) → (B, 128) → (B, 128)` and `(B, cond_dim) → (B, 128) → (B, 128)`. At B=8 this is 8 × 64 × 128 = 65 k MACs — autotune cost likely dominates any savings.
- Realistic end-to-end speedup: **<1%** (engineering estimate). This kernel is more about **consistency / idiom-parity** with the EGNN MLPs at line 130 / 147 than about measurable speed.
- Listed in top 3 because **(a) it's literally a one-line swap** and **(b) it proves the gating pattern end-to-end** before applying it more aggressively.

---

## 4. Combined expected speedup

| Kernel | Site count | Estimated step-speedup (engineering) |
| --- | --- | --- |
| #1 fused_residual_add | 7 | ~5–10% |
| #2 fused_rmsnorm_residual | 4 (1 per layer) | ~5–8% (with secondary training-quality benefit) |
| #3 fused_silu_mlp (time_mlp, cond_proj) | 2 | <1% (consistency win) |

**Combined estimated end-to-end training-step speedup: ~10–18%** (engineering estimate, NOT measured on gfx1101).

**Honest caveats:**
- All numbers above are engineering estimates derived from launch-overhead ratios on similar Triton kernels (Liger-Kernel / xformers published numbers) — they have NOT been measured on this host. The current GPU is unavailable (cuda_available=False per `WF-GPU-Auto-Recover` 2026-09-15). Without a working GPU, no measured speedup claim is honest.
- The expected speedup is per-`forward()` call, NOT per-`sample generated`. The end-to-end wall-clock effect depends on how many forward calls happen per training step (currently 1 per step in CFM; sampler uses 100 Euler steps × 1 forward each = 100 forwards per sample).
- The biggest non-fused-MLP cost — the scatter-sum aggregation (`models/_scatter.py`) and the EGNN vector message computation (`egnn_rocm.py:264` `F.silu(self.mlp(mlp_in))`) — is **NOT addressed by this plan**. `aggregate_vectors` IS imported in `_scatter.py:70` but not actually called (the file falls back to `scatter_sum_legacy`). Auditing `_scatter.py` is a separate follow-up.

---

## 5. Verification gate (cannot run on this host today)

- **Blocker:** GPU still unavailable — `cuda_available=False`, `device_count=0`, `rocminfo` HSA_STATUS_ERROR (per `molmetal/reports/wf_gpu_auto_recover/final.md` 2026-09-15).
- **What to do once GPU returns:** for each kernel, run `tests/` to verify numerical parity, then `python -c "import torch; ..."` microbenchmark before/after each wire (forward-pass latency at `(B=8, N=64, H=128)` over 1000 iterations, report median ± std).

---

## 6. Metrics

- **n_kernels_to_wire:** 3 (`fused_residual_add`, `fused_rmsnorm_residual` or `fused_rms_norm`, `fused_silu_mlp`).
- **total_expected_speedup_pct:** ~15 (midpoint of the 10–18% engineering-estimate range; explicitly NOT measured).