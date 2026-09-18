# WF-VRAM-Fix / 01_amp — CFM AMP wrapper verdict

**Date**: 2026-09-18
**Workstream**: VRAM TOP #1 (BF16 + grad checkpoint)
**Sub-deliverable**: AMP wrapper (BF16 autocast + ROCm env-var hygiene)
**Status**: SHIPPED. 10/10 tests pass CPU-only.

---

## What shipped

| File | Change |
|---|---|
| `molmetal/adapters/flow_matching_lipman/amp.py` | NEW: 240 LOC. `CFMAMPContext`, `aggregate_loss`, `check_rocm_env_vars`, `recommended_amp_kwargs`, `ROCM_GFX1101_ENV_VARS` constant. |
| `molmetal/adapters/flow_matching_lipman/__init__.py:52-95` | MODIFIED: import + `maybe_enable_amp()` reading `MOLMETAL_CFM_AMP` and `MOLMETAL_CFM_AMP_DTYPE` env vars. |
| `molmetal/molmetal_lam/tests/test_amp.py` | NEW: 10 tests (6 required + 4 supplementary for weights/empty-dict/env-var-checks). All pass in 2.27 s. |

## Pre-flight citations

- `molmetal/adapters/flow_matching_lipman/__init__.py:52-58` — TritonConfig-gated fused MLP wiring (the import-area pattern I mirrored for the AMP import block).
- `molmetal/adapters/flow_matching_lipman/_reference/__init__.py:1-8` — public entry point that `_reference_loader.py` configures at import time (kept untouched).
- `models/_scatter.py:78-94` — RDNA3 (gfx1100/1101/1102) capability-detect via `torch.cuda.get_device_capability(0)`; `_IS_RDNA3` boolean at line 97. The RDNA3 fallback pattern at `_pick_backend` (lines 127-146) is what I mimicked in `check_rocm_env_vars` (lazy, never raises).
- `models/_scatter.py:84-90` — `if not torch.cuda.is_available(): return False` early-out is the lazy-CI guard I reused.
- `models/velocity_net.py:54-86` — `_maybe_fused_residual_add` gated wrapper; the structure (constructor + `__call__` + diagnostic) is the model I followed for `CFMAMPContext.__init__` / `__enter__` / `env_summary`.
- `molmetal/scripts/metallo_drug_smoke_retrain.py:1-80` — explains why AMP is needed (5000-step probe → 14 GB HBM at batch_size=4; BF16 reclaims ~30-40 %).
- `TODO/pending/24_cfm_architecture_redo_plan.md` — referenced from the module docstring as the broader audit context.

## Design decisions

1. **BF16 default, no GradScaler.** BF16 has the same exponent range as FP32 (8 exponent bits) so the loss-doesn't-overflow reason for `GradScaler` doesn't apply. The `CFMAMPContext` class deliberately omits a GradScaler attribute — the docstring at `amp.py:67-71` says so explicitly. If you switch to FP16 via `MOLMETAL_CFM_AMP_DTYPE=float16` you must wrap the optimizer step yourself (canonical PyTorch AMP examples recipe).

2. **`aggregate_loss` always forces FP32.** This is the single correctness rule the PyTorch AMP docs (https://pytorch.org/docs/stable/amp.html, "Autocasting" section) require, and the project has already hit a related class of issue: `wf_cfm_p0_fixes` shipped F3 (vocab_mask BEFORE F.cross_entropy) to recover ~88 % of the wasted gradient. Defensive FP32 promotion at the loss-aggregation point costs nothing and removes an entire category of "loss plateau at a spurious floor" bugs.

3. **Env vars warned, not required.** `check_rocm_env_vars` mirrors the RDNA3 lazy-detect pattern at `models/_scatter.py:84-94`: emit a `RuntimeWarning` if either var is missing, never raise. This keeps CPU-only CI (where `torch.cuda.is_available()` is False) green.

4. **Env-var knobs (`MOLMETAL_CFM_AMP`, `MOLMETAL_CFM_AMP_DTYPE`).** Both default to "1" / "bfloat16" so the wrapper is a zero-config VRAM win for the metallodrug smoke retrain. Setting `MOLMETAL_CFM_AMP=0` short-circuits to `_NullAutocast` (no `torch.autocast` call) so legacy callers pay nothing.

5. **No new pip deps.** Only `torch` and stdlib (`os`, `warnings`). Listed in `amp.py:23-30` module docstring.

## Test results

```
$ uv run pytest molmetal/molmetal_lam/tests/test_amp.py -v
collected 10 items
molmetal/molmetal_lam/tests/test_amp.py::test_bf16_autocast_enabled                       PASSED
molmetal/molmetal_lam/tests/test_amp.py::test_fp16_autocast_via_env_var                   PASSED
molmetal/molmetal_lam/tests/test_amp.py::test_aggregate_loss_forces_float32                PASSED
molmetal/molmetal_lam/tests/test_amp.py::test_aggregate_loss_with_weights                 PASSED
molmetal/molmetal_lam/tests/test_amp.py::test_aggregate_loss_empty_dict_raises            PASSED
molmetal/molmetal_lam/tests/test_amp.py::test_disabled_passthrough                        PASSED
molmetal/molmetal_lam/tests/test_amp.py::test_recommended_kwargs_gfx1101_defaults         PASSED
molmetal/molmetal_lam/tests/test_amp.py::test_rocm_env_vars_set_constant                   PASSED
molmetal/molmetal_lam/tests/test_amp.py::test_check_rocm_env_vars_warns_when_missing       PASSED
molmetal/molmetal_lam/tests/test_amp.py::test_check_rocm_env_vars_quiet_when_set           PASSED
======================== 10 passed, 1 warning in 2.27s =========================
```

The single warning is unrelated (`hypothesis` plugin note about `.hypothesis/` dir).

## Runtime smoke

```
$ uv run python -c "from molmetal.adapters.flow_matching_lipman import maybe_enable_amp; ..."
default:  True  torch.bfloat16
disabled: False
```

Both `MOLMETAL_CFM_AMP=1` and `MOLMETAL_CFM_AMP=0` produce the right `CFMAMPContext` shape, and `MOLMETAL_CFM_AMP_DTYPE=float16` would dispatch to `torch.float16` (verified in the unit test).

## Honest framing

- **NO GPU run.** Per the task spec, this PR is CPU-only; the autocast wrapper is exercised only at the dtype-checking layer. The actual VRAM reclaim (~30-40 % claim from the AMP literature) is a PROJECTION, not a measurement on gfx1101. The follow-up workflow (wf_vram_fix/02_grad_checkpoint or equivalent) needs to actually run `metallo_drug_smoke_retrain.py` under the AMP wrapper to confirm the bytes-saved number on the RX 7800 XT.
- **No GradScaler path tested.** The `float16` env-var branch is wired but the FP16-with-GradScaler integration is left as a follow-up; BF16 is the canonical path on gfx1101.
- **`maybe_enable_amp` only logs at INFO.** If the smoke script doesn't configure `logging.basicConfig` you won't see the decision line in stdout. Recommend wiring `basicConfig(level=INFO)` at the top of `metallo_drug_smoke_retrain.py:80-90` before the AMP entry-point is called.

## Next steps

1. Wire `maybe_enable_amp()` into `metallo_drug_smoke_retrain.py:80-90` and re-run the 1000-step smoke. Verify the HBM delta (use `torch.cuda.max_memory_allocated()` before/after the wrapper).
2. Sub-deliverable 02: per-EGNN-layer gradient checkpointing (the second half of the VRAM TOP #1 fix).
3. After both ship: full 10 000-step retrain (per `TODO/pending/24_cfm_architecture_redo_plan.md` path-a).

## Files touched (final)

```
NEW   molmetal/adapters/flow_matching_lipman/amp.py                240 LOC
MOD   molmetal/adapters/flow_matching_lipman/__init__.py           +44 LOC
NEW   molmetal/molmetal_lam/tests/test_amp.py                      190 LOC
```

All other files untouched (per task constraint).
