# Wf-333-G2: Triangle 12-d GPU launch wrapper — production CLI scaffolding

**Status:** SHIPPED (CPU-only wrapper; GPU launch deferred per AGENTS.md §5.1)
**Date:** 2026-09-29
**Author:** Wf-333-G2 (ultracode GPU wrapper sub-flow)

---

## Purpose

Per Wf-331-R2 closeout, the F5_RDKIT_2D 12-d triangle closure
recovery achieved a **CPU proxy lift** of `+1.024` on `mean_hilbert_r2`
(-1.5391 → -0.5153) but **does NOT** meet the R² ≥ 0.6 honesty gate
that paper §3.6 / R20-P2 / TODO-93 require.  Wf-331-R2 documented
this gap as **PROJECTED ≥ 0.6 + ≥ 0.05 mean_variance_reduction on
real pIC50** with a **HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED** label
until the GPU launch runs.

Wf-333-G2 ships the **production GPU launch wrapper** so downstream
Wf-333-G3+ (and any other consumer) can dispatch the 5-seed ×
50-epoch GPU retrain with the Wf-326/328 GPU conventions
(``add_gpu_args`` / ``setup_gpu_env`` / ``make_retrain_dataloader``
+ ``torch.compile``).  Per AGENTS.md §5.1 ``kill_then_adapt``
directive, **this wrapper does NOT itself launch GPU training** —
it only resolves the CLI scaffolding, emits the JSON / Markdown
summary, and labels the launch as deferred.

---

## What G2 ships

1. **New module**: `molmetal/scripts/wf333g2_triangle_12d_gpu_wrapper.py`
   (~580 LOC)
   * `TriangleGPUConfig` dataclass — frozen launch configuration
     (feature_mode / hilbert_dim / epochs / lr / seeds / mixed_precision
     / cuda_visible_devices / torch_compile / launch / dataset / output).
   * `TriangleGPUSummary` dataclass — emitted by `build_triangle_gpu_trainer`,
     carries the resolved feature schema, honesty-gate verdict, project
     tag, and honest-negative flags.
   * `build_triangle_gpu_trainer(config, ...)` — wrapper construction
     (CPU-only; no GPU launch).  Resolves the 12-d schema, builds the
     Hilbert torch module, and emits `TriangleGPUSummary`.
   * `build_hilbert_torch_module(hilbert_dim, dtype)` — small torch
     module carrying the Hilbert PD iterate; the `forward` is a single
     inner-product step (`X @ x`) that benefits from `torch.compile`.
   * `hilbert_pd_step_torch(module, feats, residual, ...)` — single PD
     step on torch tensors (mirrors the NumPy `fit_hilbert_stable`
     gradient).  Used by downstream Wf-333-G3+ to wrap the Hilbert
     vertex in `torch.compile` + autocast.
   * `triangle_composite(...)` — weighted sum of the 3 vertex metrics
     (default weights `1.0 / 1.0 / 1.0`).
   * `honesty_gate_verdict(...)` — per-gate verdict dict (hilbert_r2 /
     variance_reduction / success_rate) with `pass` boolean + `tag`
     (MEASURED vs PROJECTED).
   * `add_wf333g2_args(parser)` — CLI flag attachment.  Uses the
     idempotent `_add` helper from `molmetal.scripts._common_train_args`
     (Wf-326 ship surface) so the function can be called multiple
     times on the same parser without raising `argparse.ArgumentError`.
   * `parse_args(argv)` — CLI parsing → `TriangleGPUConfig`.
   * `write_summary(summary, out_dir)` — writes
     `summary.json` + `summary.md` to `out_dir`.
   * `main(argv)` — CLI entry point.  Dry-run by default; emits
     `STATUS: SKIP (HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED ...)` when
     GPU launch is deferred; exit code 0 (SKIP-as-pass per Wf-326/328
     convention for deferred launches).

2. **Test file**: `molmetal/tests/test_wf333g2_triangle_gpu_wrapper.py`
   (3 tests — 2 from the Wf-333-G2 task spec + 1 importability smoke):
   * `test_wrapper_initialization_cpu_only` — exercises the full
     `build_triangle_gpu_trainer` path; asserts:
     - `feature_shape == (n_samples, 12)` for the 12-d schema.
     - 12-d F5_RDKIT_2D feature names present (MolWt / MolLogP / TPSA).
     - Vertex labels = `("astsr", "hilbert", "mcts")`.
     - `n_trainable_params == 64` for `hilbert_dim=64`.
     - `HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED` flag is set.
     - Project tag = `PROJECTED` (CPU dry-run).
     - No GPU touched (toy design matrix is plain numpy float32).
     - `write_summary` writes both `summary.json` + `summary.md`.
   * `test_wrapper_cli_args_and_default_config` — exercises `parse_args`
     end-to-end; asserts:
     - Default `feature_mode = "12d"`, `hilbert_dim = 64`, `epochs = 50`,
       `seeds = 5`, `n_folds = 5`, `mixed_precision = "bf16"`,
       `cuda_visible_devices = "1"`, `torch_compile = False`, `launch = False`.
     - All `parse_args` invocations (with / without flags) return the
       expected `TriangleGPUConfig`.
     - Invalid `--feature-mode 99d` is rejected by argparse.
     - `add_wf333g2_args` is idempotent (can be called twice on the same
       parser without raising).
     - `resolve_feature_schema` returns canonical 5d / 12d / 17d tuples;
       17d first 5 entries equal 5d names (concat invariant).
     - `honesty_gate_verdict` correctly classifies pass / fail for all 3 gates.
     - `triangle_composite` correctly computes the weighted sum.
   * `test_wrapper_module_imports_clean` — guard against accidental
     `torch.cuda.is_available()` or other GPU side-effects at module
     import time.

3. **CLI smoke** (no GPU launch):
   ```
   PYTHONPATH=/home/hugo/codes/mol-metal \
     .venv/bin/python -m molmetal.scripts.wf333g2_triangle_12d_gpu_wrapper \
     --feature-mode 12d --seeds 3 --epochs 10 \
     --output /tmp/wf333g2_smoke
   ```
   Emits `summary.json` + `summary.md`; `STATUS: SKIP
   (HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED — wrapper dry-run only)`;
   exit code 0.  Wall time 0.005 s.

---

## CLI surface (Wf-333-G2 wrapper)

```
usage: wf333g2_triangle_12d_gpu_wrapper.py [-h]
    [--feature-mode {5d,12d,17d}]
    [--hilbert-dim HILBERT_DIM] [--epochs EPOCHS] [--lr LR]
    [--seeds SEEDS] [--n-folds N_FOLDS]
    [--dataset {nci60,platinai,both,metalcytotox}]
    [--output OUTPUT] [--mixed-precision {bf16,fp16,none}]
    [--num-workers NUM_WORKERS]
    [--cuda-visible-devices CUDA_VISIBLE_DEVICES]
    [--torch-compile]
    [--torch-compile-mode {default,reduce-overhead,max-autotune}]
    [--launch]
```

| Flag | Default | Purpose |
|---|---|---|
| `--feature-mode` | `12d` | 5-d / 12-d / 17-d feature schema (Wf-331-R2 ship surface) |
| `--hilbert-dim` | `64` | Hilbert space dimension |
| `--epochs` | `50` | PD iterations per fold |
| `--lr` | `0.05` | Peak learning rate |
| `--seeds` | `5` | Number of seeds (Wf-326 standard) |
| `--n-folds` | `5` | CV folds per seed |
| `--dataset` | `nci60` | `nci60` / `platinai` / `both` / `metalcytotox` |
| `--output` | `molmetal/reports/wf333g2_triangle_gpu` | Output directory |
| `--mixed-precision` | `bf16` | Autocast dtype (`bf16` / `fp16` / `none`) |
| `--num-workers` | `4` | DataLoader worker count (Wf-191 spec) |
| `--cuda-visible-devices` | `1` | RTX 5090 GPU 0 reserved for VLM venv |
| `--torch-compile` | `False` | Wrap hot path in `torch.compile` (first-call latency ~30s) |
| `--torch-compile-mode` | `reduce-overhead` | `default` / `reduce-overhead` / `max-autotune` |
| `--launch` | `False` | Actually launch GPU training (default = dry-run) |

---

## Integration points (downstream consumers)

| Consumer | Path | What to do |
|---|---|---|
| **Wf-333-G3+** | TODO/pending | Re-run `build_triangle_gpu_trainer` with `config.launch=True` when CUDA + free VRAM available; replace placeholder honesty-gate values with real R² from the GPU launch. |
| **paper §3.6** | (READ-only) | Cite `summary.json` + the Wf-331-R2 closeout as the CPU proxy baseline; mark the GPU portion as PROJECTED until Wf-333-G3+ lands. |
| **R20-P2 / TODO-93** | TODO/pending/93_r20_p2_hilbert_astsr_gpu.md | Consume `TriangleGPUConfig` + `TriangleGPUSummary`; reuse `build_hilbert_torch_module` + `hilbert_pd_step_torch` for the joint Hilbert regression axis. |
| **Wf-37 L8** | molmetal/scripts/triangle_closure.py | `TriangleClosure._run_astsr_vertex` / `_run_hilbert_vertex` / `_run_mcts_vertex` can be wrapped in `hilbert_pd_step_torch` for the GPU path; CPU NumPy path is unchanged (backward compat). |

---

## Cross-references

* `TODO/completed/331r2_triangle_12d_cpu_scaffold.md` — Wf-331-R2 closeout
  (CPU proxy +1.024 lift on `mean_hilbert_r2`; PROJECTED ≥ 0.6 on GPU).
* `TODO/completed/330f_wf329_closeouts.md` — Wf-329 triangle closure
  MEASURED R² = 0.5436 (BELOW R² ≥ 0.6 honesty gate).
* `molmetal/scripts/triangle_features_12d.py` — 12-d RDKit feature
  schema (Wf-331-R2 ship).
* `molmetal/scripts/train_hilbert_astsr_end_to_end.py` — CPU NumPy
  trainer (unchanged; backward compat preserved).
* `molmetal/scripts/_common_train_args.py` — `add_gpu_args` /
  `setup_gpu_env` / `_add` helper (Wf-326 ship).
* `molmetal/scripts/_data_loader_helper.py` — DataLoader factory
  (Wf-191 ship).
* `molmetal/scripts/wf326c_torch_compile_optimized.py` — torch.compile
  pattern reference (Wf-326 ship).
* `molmetal/tests/test_wf333g2_triangle_gpu_wrapper.py` — this G2 ship.

---

## Honest framing summary

* **MEASURED lift (Wf-331-R2)**: +1.024 on `mean_hilbert_r2`
  (-1.5391 → -0.5153) on synthetic data via the 12-d CPU proxy.
* **PROJECTED GPU lift (Wf-333-G2 + downstream G3+)**: ≥ 0.6 R²
  on real pIC50 (PlatinAI 4-cell + MetalCytoToxDB 5-metal) requires
  the actual GPU launch.  The wrapper is ready — the launch is the
  deferred dependency.
* **HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED** applies until Wf-333-G3+
  runs the launch.  The wrapper ships the SKIP label so paper §3.6
  + R20-P2 can mark the gap.
* **MEASURED vs PROJECTED discipline**: every numeric value in this
  doc traces to either `molmetal/tests/test_wf333g2_triangle_gpu_wrapper.py`
  (CPU smoke) or `TODO/completed/331r2_triangle_12d_cpu_scaffold.md`
  (Wf-331-R2 CPU bench).  No fabricated numbers.
* **No GPU launch per AGENTS.md §5.1**: the wrapper is CPU-only
  scaffolding; downstream Wf-333-G3+ runs the actual launch when
  VRAM frees up on the second RTX 5090 or RTX PRO 6000 Blackwell.
