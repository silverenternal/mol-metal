# WF-VRAM-Fix / 02_checkpoint — per-EGNN-layer grad checkpoint verdict

**Date**: 2026-09-18
**Workstream**: VRAM TOP #1 (BF16 + grad checkpoint)
**Sub-deliverable**: Per-layer gradient checkpointing wrapper around each EGNN message-passing block
**Status**: SHIPPED. 5/5 tests pass CPU-only (2.93 s). E2E EGNN forward + backward verified finite.

---

## What shipped

| File | Change |
|---|---|
| `molmetal/adapters/egnn_rocm_checkpoint.py` | NEW: 200 LOC. `CheckpointedEGNNLayer`, `egnn_checkpoint_default_enabled`, `set_egnn_checkpoint_default_enabled`. |
| `molmetal/adapters/egnn_rocm.py:36-46` | MODIFIED: added `from .egnn_rocm_checkpoint import (CheckpointedEGNNLayer, egnn_checkpoint_default_enabled)` with mandatory-use-reentrant-False docstring. |
| `molmetal/adapters/egnn_rocm.py:548-572` | MODIFIED: `self.layers = nn.ModuleList([CheckpointedEGNNLayer(block=EquivariantGraphConv(hidden_dim, hidden_dim), enabled=egnn_checkpoint_default_enabled()) for _ in range(n_layers)])`. Forward signature is UNCHANGED — `CheckpointedEGNNLayer.forward(*args, **kwargs)` is fully transparent. |
| `molmetal/molmetal_lam/tests/test_egnn_checkpoint.py` | NEW: 5 tests (basic_wrap + disabled_passthrough + preserve_rng + use_reentrant_false + env_var_disabled). All pass in 2.93 s. |

## Pre-flight citations

- `molmetal/adapters/egnn_rocm.py:175-518` — `EquivariantGraphConv` is the block being wrapped; it accepts `(h, x, edge_index, dative_bond_edge_attr=None, edge_types=None)` — that signature is the contract the wrapper must respect, hence `forward(*args, **kwargs)`.
- `molmetal/adapters/egnn_rocm.py:542-545` (legacy) — original `nn.ModuleList([EquivariantGraphConv(...) for _ in range(n_layers)])` was the construction site; same shape, just wrapped per layer.
- `molmetal/adapters/egnn_rocm.py:586-589` — `EGNN.forward` loop is the caller; `layer(h, x, edge_index, dative_bond_edge_attr=..., edge_types=...)` flows through unchanged.
- `molmetal/models/_scatter.py:67-84` — confirmed `scatter_sum_legacy` only requires `dim=0` + 2-D `values`, so checkpointing one layer at a time is safe (the scattered tensors are recomputed deterministically inside the recompute).
- `molmetal/adapters/egnn_rocm.py:37-65` — the `_maybe_fused_residual_add` pattern is the env-gated dispatch model the wrapper's env-driven `enabled` flag mirrors (process-wide default + per-instance override).
- `molmetal/reports/wf_vram_fix/01_amp.md` — sister verdict; the AMP wrapper ships BF16 autocast + env-var hygiene; this wrapper ships the *other half* of the VRAM TOP #1 fix.
- `TODO/pending/24_cfm_architecture_redo_plan.md` — the broader CFM audit context; the checkpoint wrapper is Phase-0 (infra, not retrain).

## Design decisions

1. **Per-layer, not per-EGNN.** Wrapping the whole `EGNN` would re-evaluate `input_proj` / `output_proj` for free on backward, but those projections are tiny Linear layers. The VRAM hot spot is the per-layer activation: each `EquivariantGraphConv` materialises `[n_edges, hidden_dim]` scalar-message and `[n_edges, 3]` coord-message tensors that PyTorch holds for backward. Per-layer wrap gives `O(n_layers)` activation relief where it matters; per-EGNN wrap is wasteful. Documented at `egnn_rocm_checkpoint.py:20-32`.

2. **`use_reentrant=False` is mandatory, full stop.** PyTorch 2.14 + ROCm 7.2 + gfx1101 raises `Expected tensor metadata` on the reentrant path during backward on HIP builds. The wrapper does NOT expose this as a parameter — it is always `False`. Even if a caller passes `enabled=True`, the flag cannot be flipped to `True`. Captured in `egnn_rocm_checkpoint.py:88-94` and re-stated at the call site in `egnn_rocm_checkpoint.py:184`.

3. **`preserve_rng_state=True`.** EGNN message passing has no dropout today, but the wrapper is generic. Setting the flag ensures future stochastic regularisation inside `self.block` (dropout / feature noise) is bit-equivalent between the first forward and the backward recompute. The `test_checkpoint_preserve_rng` test pins the invariant: two checkpointed forwards with the same RNG seed must produce bit-equal output (and they must equal a direct forward from the same seed).

4. **Env-var gate, default ON.** `MOLMETAL_EGNN_CHECKPOINT` defaults to `"1"` so the wrapper is the zero-config VRAM win (same posture as `MOLMETAL_CFM_AMP=1` in the AMP sister). Setting `MOLMETAL_EGNN_CHECKPOINT=0` reverts to the legacy per-layer direct forward. `egnn_checkpoint_default_enabled()` reads the env var once at import; `set_egnn_checkpoint_default_enabled(value)` lets tests and benchmarks override without touching the env. Mirrors the `set_use_fused_coord_update` pattern at `molmetal/adapters/egnn_rocm.py:118-126`.

5. **Zero-overhead disabled path.** When `enabled=False`, `forward` is `self.block(*args, **kwargs)` — no checkpoint context manager, no RNG-state capture/restore, no autograd graph rewriting. `test_checkpoint_disabled_passthrough` asserts bit-exact equality vs the direct block call.

6. **No new pip deps.** Only `torch`, `torch.nn`, `torch.utils.checkpoint`, stdlib `os`. Listed at `egnn_rocm_checkpoint.py:55-60`.

## Test results

```
$ uv run pytest -q --no-header --tb=short molmetal/molmetal_lam/tests/test_egnn_checkpoint.py
.....                                                                    [100%]
5 passed, 1 warning in 2.93s
```

| Test | Validates |
|---|---|
| `test_checkpoint_wrap_basic` | Forward shape matches; backward populates `x.grad` AND every `block.*.grad`. |
| `test_checkpoint_disabled_passthrough` | `enabled=False` is a true identity forward (`torch.equal(direct, via_wrap)`). |
| `test_checkpoint_preserve_rng` | Two checkpointed forwards with the same RNG seed produce bit-equal output AND equal the direct forward from the same seed. |
| `test_checkpoint_use_reentrant_false` | Monkey-patch + reload stub records kwargs; `use_reentrant=False` and `preserve_rng_state=True` both confirmed at the call site. |
| `test_env_var_disabled` | `MOLMETAL_EGNN_CHECKPOINT=0` flips the helper AND causes `EGNN(n_layers=2).layers[*].enabled == False`. |

## Regression check (existing EGNN tests)

```
$ uv run pytest -q --no-header --tb=short \
    molmetal/molmetal_lam/tests/test_egnn_checkpoint.py \
    molmetal/tests/test_egnn_coord_scatter_wired.py \
    molmetal/tests/test_egnn_velocity_cfg.py \
    molmetal/tests/test_egnn_predictor_device.py
.........................                                                [100%]
25 passed, 1 warning in 3.18s
```

All 20 pre-existing EGNN tests continue to pass after the wrap — confirms the transparent-forward contract is bit-clean for the Linear + coord path.

## End-to-end smoke (no GPU)

```python
from molmetal.adapters.egnn_rocm import EGNN
net = EGNN(in_node_dim=4, hidden_dim=4, n_layers=2)
# net.layers[0] is CheckpointedEGNNLayer(enabled=True, block=EquivariantGraphConv(...))
net.train()
h = torch.randn(3, 4, requires_grad=True)
x = torch.randn(3, 3)
ei = torch.tensor([[0,1,2],[1,2,0]])
h2, x2 = net(h, x, ei)
h2.sum().backward()
# h.grad finite: True
```

Backward completes in CPU eval and all gradients are finite — no surprise interaction between the checkpoint wrapper and the existing `_MaybeFusedSiLUMLP` / `_scatter_sum` paths.

## Honest framing

- **NOT MEASURED**: actual VRAM reduction on RX 7800 XT — GPU is blocked per `wf_gpu_recovery_now/final.md` (cuda_available=True but dGPU SMU hang history). Expected reduction is `~50%` of per-layer activations: each of `n_layers` `EquivariantGraphConv` activations drops from `[n_edges, hidden_dim]` + `[n_edges, 3]` + assorted intermediates to a single tuple-of-tensors handle held by the autograd graph. Backward will recompute these (one extra forward per layer, ~30% wall-clock cost).
- **NOT MEASURED**: backward wall-clock cost on the CFM training loop. The Phase-2 fused kernels (`rotation_from_axis_angle`, `aggregate_vectors`) are also called during recompute — if the fused kernel autotune cache is process-cold, the first recompute will trigger compilation. Mitigated by the `torch.cuda.graph` warm-up in the existing training scripts.
- **NOT MEASURED**: interaction with `EquivariantGraphConv._project_to_hidden` (line 511-518) lazy-allocated `_h_proj`. The lazy allocation only fires on the first call where `h.size(-1) != hidden_dim`, which is the input projection layer's responsibility — by the time `EGNN.forward` enters the per-layer loop `h` is already `hidden_dim`-sized. No risk of re-allocation under recompute.

## Follow-ups

1. Run the CFM 5000-step diagnostic on a recovered GPU with both `MOLMETAL_CFM_AMP=1` AND `MOLMETAL_EGNN_CHECKPOINT=1` to measure VRAM and wall-clock deltas vs the BF16-only baseline.
2. Add an `--egnn-checkpoint` CLI flag in `r4_lambda_only_run.py` for finer-grained benchmarking at single-layer granularity (turn off layers 0..N-1 only).
3. Update `wf_triton_kernel_audit` (2026-09-15) recommendation #1 "fused_residual_add wire-in 7 sites" to defer-checkpoint confirmation: the recompute path will benefit MORE from fused_residual_add (it runs twice per backward) — bumped to TOP #1 + #2 combined priority.