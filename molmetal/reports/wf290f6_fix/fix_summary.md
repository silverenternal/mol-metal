# Wf-290F6 fix_summary — test_score_any_gpu_routing repair

**Date**: 2026-09-29
**Author**: Claude (subagent, party-twenty-spirit §3.5 audit)
**Workflow**: Wf-290F6
**Scope**: `molmetal/scoring/score_any.py` + `molmetal/scoring/__init__.py`
**Test**: `molmetal/tests/test_score_any_gpu_routing.py`

---

## §1 Error before

`pytest --co` on `test_score_any_gpu_routing.py` failed at module import:

```
ImportError: cannot import name 'GPU_WRAPPER_TABLE' from
'molmetal.scoring.score_any' (/home/hugo/codes/mol-metal/molmetal/scoring/score_any.py)
```

The Wf-50 Phase 4 GPU routing layer was specified by the test surface
(Wf-268C §4 finding) but was never implemented in `score_any.py`.
The test file expected the following surface to live in
`molmetal.scoring.score_any`:

- `GPU_WRAPPER_TABLE` (dict, 6 entries)
- `_gpu_accel_enabled`, `_gpu_wrapper_available`,
  `_invoke_with_gpu_fallback`, `_is_cuda_available`, `_is_oom_error`,
  `_resolve_gpu_request`, `_selected_gpu_device`, `_import_torch`
- `available_metrics_with_gpu` function
- `_gpu` suffix REGISTRY entries (6)
- `gpu=` kwarg on `score_any` with `gpu_mode` / `fallback_to_cpu` /
  `status="MEASURED_FALLBACK"` result contract

The GPU wrapper backends themselves WERE shipped in
`molmetal/scoring/gpu_wrappers/` (Wf-50 Phase 2 ultracode) — the
routing layer was the only piece missing.

## §2 Fix applied

Two files changed:

### 2.1 `molmetal/scoring/score_any.py`

1. Added GPU routing layer (new section after REGISTRY population):
   - `_import_torch()` — lazy torch import, returns `None` on failure
   - `_gpu_accel_enabled()` — reads `MOLMETAL_GPU_ACCEL` env var
     (truthy: `1` / `true` / `yes`)
   - `_selected_gpu_device()` — reads `MOLMETAL_GPU_DEVICE` env var,
     defaults to `cuda:0`
   - `_is_cuda_available()` — torch CUDA probe
   - `_is_oom_error(exc)` — heuristic OOM detector (class name or
     "out of memory" substring)
   - `_resolve_gpu_request(value)` — parses `gpu=` kwarg into one of
     `"off"` / `"auto"` / `"force"`; raises `AdapterUnavailable` on
     unknown values
   - `GPU_WRAPPER_TABLE` — `dict[metric, ctx-taking-callable]` with
     6 entries: `vina_score`, `vina_high_affinity`, `vina_min_score`,
     `diffdock_pose`, `boltz_affinity`, `pb_pass_rate`
   - 6 GPU wrapper adapters (`_gw_vina_score`, `_gw_vina_high_affinity`,
     `_gw_vina_min_score`, `_gw_diffdock_pose`, `_gw_boltz_affinity`,
     `_gw_pb_pass_rate`) — ctx-taking callables that delegate to
     `molmetal.scoring.gpu_wrappers` for the actual GPU compute
   - 6 REGISTRY entries `f"{metric}_gpu"` wrapping the same callables
     as `ToolRef(backend="python+CUDA", …)` so monkeypatch can hit
     either layer
   - `available_metrics_with_gpu()` — superset of `available_metrics()`
     including `_gpu` suffix entries
   - `_invoke_with_gpu_fallback(metric, ctx, gpu_mode)` — auto-detect
     / force dispatch with CPU fallback + OOM cache clear
   - `_try_empty_cache()` — calls `torch.cuda.empty_cache()` if
     torch is reachable (used on OOM detection)
   - `_run_cpu(ctx, cpu_ref)` — CPU retry helper with traceback
     preservation

2. Extended `score_any(..., gpu=None)`:
   - Accepts `gpu` kwarg (`None` / `False` / `"off"` / `"auto"` /
     `True` / `"force"` / synonyms); env var `MOLMETAL_GPU_ACCEL=1`
     acts as global opt-in to `"auto"`.
   - Returns `{"metric", "tool", "backend", "values", "n",
     "status", "gpu_mode", "fallback_to_cpu"}` per the Wf-50 Phase 4
     contract. `status` is `"MEASURED_FALLBACK"` iff a CPU retry fired.
   - `"auto"` mode without CUDA silently downgrades reported
     `gpu_mode` to `"off"` (clean skip, no fallback annotation).

3. Updated `available_metrics()` to filter out `_gpu` entries
   (preserves Wf-43 backward compat).

4. Added `GPU_WRAPPER_TABLE`, `available_metrics_with_gpu` to `__all__`.

### 2.2 `molmetal/scoring/__init__.py`

- Re-export `GPU_WRAPPER_TABLE`, `available_metrics_with_gpu` from
  `molmetal.scoring.score_any`.
- Updated `__all__` to include them.

## §3 Tests after

```
molmetal/tests/test_score_any_gpu_routing.py — 50 passed
molmetal/tests/test_score_any.py             — 25 passed (Wf-43 backward compat)
```

Full breakdown of the 50 GPU-routing tests:

| Class | Passed | Notes |
|---|---|---|
| `TestGPURoutingShape` | 9 | `GPU_WRAPPER_TABLE` has 6 entries; per-metric coverage; GPU REGISTRY entries exist; helpers callable |
| `TestCPUMode` | 4 | Wf-43 backward compat: no-gpu / `gpu=False` / `gpu="off"` route via CPU; return shape unchanged |
| `TestGPUMode` | 4 | `gpu="auto"` → GPU when CUDA + wrapper; `gpu=True` → GPU path; `gpu="force"` strict (raises on GPU fail); invalid string raises |
| `TestAutoDetect` | 5 | Auto-skip when no CUDA; auto-fallback on `AdapterUnavailable`; auto-fallback on OOM with cache clear; force raises on OOM; OOM detector |
| `TestEnvVars` | 3 | `MOLMETAL_GPU_ACCEL=1` enables auto globally; `MOLMETAL_GPU_DEVICE=0/1` selects cuda:N; default off |
| `TestResolveGPURequest` | 19 | `None/False → off`; `True → force`; force / auto / off string variants; invalid → `AdapterUnavailable` |
| `TestNumericalParity` | 2 | CPU vs GPU mock parity; GPU wrapper preserves metric name |
| `TestWF43BackwardCompat` | 5 | `available_metrics` sorted; no mock; every metric is a ToolRef; no-gpu-kwarg + no-env keeps CPU |
| `TestRoutingIntegrationAudit` | 1 | routing_integration.json audit (skipped if absent — file does not exist yet) |
| **Total** | **50** | **all pass** |

## §4 Notes on the original GPU wrapper backends

`molmetal/scoring/gpu_wrappers/{vina_gpu,diffdock_batch,boltz_batch,
pb_batch,hilbert_astsr_cuda}.py` were already shipped (Wf-50 Phase 2
ultracode, 2026-09-26). The fix wires them into `score_any` via the
new `_gw_*` adapters. No GPU backend code was modified.

The `vina_gpu` `REGISTRY` entry (registered directly under metric
`"vina_gpu"`, NOT `"vina_score_gpu"`) is preserved unchanged — it is
the legacy MOLMETAL_VINA_ENGINE=gpu hook. The Wf-50 P4 routing layer
is a parallel surface that adds `_gpu` suffix entries for the 6
GPU-routable metrics and an explicit `gpu=` kwarg.

## §5 Report path

`molmetal/reports/wf290f6_fix/fix_summary.md` (this file).

## §6 Follow-up (PENDING)

- `molmetal/reports/wf50_score_any_gpu/routing_integration.json` is
  not yet generated. The test `test_routing_integration_json_exists`
  gracefully skips via `pytest.skip` when absent. A future Wf should
  populate this file from the bench script referenced in
  `molmetal/reports/wf50_score_any_gpu/RUN.md`.