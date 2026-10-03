# `molmetal.scoring` — GPU Routing Integration (Wf-50 Phase 4)

> **Wf-50 Phase 4 ship**: opt-in GPU wrappers for the 5 highest-impact CPU-bound scoring tools wired into `score_any()` as a routing layer.

## 1. Scope

Wf-50 Phase 1 surfaced the CPU-vs-GPU speed-up audit (`molmetal/reports/wf50_gpu_audit/critical_path.md`).
Wf-50 Phase 2 shipped the 5 GPU batch wrappers (`molmetal/scoring/gpu_wrappers/`).
Wf-50 Phase 3 measured the speed-ups (`molmetal/reports/wf50_gpu_bench/cross_wrapper_comparison.md`).
**Wf-50 Phase 4 (this document)** wires the wrappers into `score_any()` so call-sites can opt in
without code changes — by setting `gpu=True` on the call, or by exporting
`MOLMETAL_GPU_ACCEL=1` to turn on GPU routing globally.

Backward compatibility: omitting the `gpu` kwarg (the Wf-43 default) yields
exactly the Wf-43 CPU behaviour. No existing call-site breaks.

## 2. Opt-in modes

| `gpu=` value | Effect |
|---|---|
| `False` / `None` / `"off"` (default) | **CPU path only** — Wf-43 unchanged. |
| `True` / `"force"` / `"1"` | **Require GPU wrapper**; raises `AdapterUnavailable` on GPU failure (no CPU fallback). |
| `"auto"` | Use GPU wrapper iff (a) CUDA is available AND (b) the metric has a GPU wrapper; otherwise CPU path. GPU failure → fall back to CPU with `status="MEASURED_FALLBACK"` + WARNING log. |

**Env vars**:

* `MOLMETAL_GPU_ACCEL=1` — global `gpu="auto"` for every `score_any()` call. unset / `0` → unchanged.
* `MOLMETAL_GPU_DEVICE=0` (or `1`, `2`, …) — pick GPU-0 / GPU-1 / …. Forwarded to GPU wrappers as the `device` kwarg (`cuda:0`, `cuda:1`, …). unset → `cuda:0`.

## 3. Per-metric GPU wrappers

| Metric | GPU wrapper | VRAM ceiling | Fallback to CPU on |
|---|---|---|---|
| `vina_score` | `vina_gpu.vina_score_gpu` | 1.2 GB | no CUDA / no Vina-GPU binary / OOM |
| `vina_high_affinity` | `vina_gpu.vina_high_affinity_gpu` | 1.2 GB | as above (boolean on top of `vina_score_gpu`) |
| `vina_min_score` | `vina_gpu.vina_min_score_gpu` | 1.2 GB | as above (aggregate of `vina_score_gpu`) |
| `diffdock_pose` | `diffdock_batch.diffdock_pose_batch` | 8 GB | no CUDA / no DiffDock backend / OOM |
| `boltz_affinity` | `boltz_batch.boltz_affinity_batch` | 2 GB | no CUDA / no pIC50 checkpoint / OOM |
| `pb_pass_rate` | `pb_batch.pb_pass_rate_batch` | 0 MB (RDKit-only) | PoseBusters not installed |

Notes:

* `pb_pass_rate`'s "GPU" wrapper is a **thread-pool dispatcher** (PoseBusters is RDKit-bound,
  no CUDA MMFF); the speed-up comes from `ThreadPoolExecutor` overlap, not GPU compute.
  The wrapper is registered under the same `gpu=` knob for API consistency.
* `hilbert_astsr_optimize_batch` is exposed in `molmetal/scoring/gpu_wrappers/` but is NOT
  routed through `score_any()` — it is used internally by `reward/learned_shaping.py`.

## 4. Usage examples

### 4.1 Per-call opt-in

```python
from molmetal.scoring import score_any

# CPU path (Wf-43 unchanged).
cpu_result = score_any("boltz_affinity", molecules=["CCO", "CC(=O)Oc1ccccc1C(=O)O"])

# GPU path (requires CUDA + pIC50 checkpoint).
gpu_result = score_any("boltz_affinity", molecules=["CCO", "CC(=O)Oc1ccccc1C(=O)O"],
                       gpu="auto")
print(gpu_result["backend"])  # "python+CUDA"
print(gpu_result["status"])   # "MEASURED" (or "MEASURED_FALLBACK" on failure)
```

### 4.2 Force GPU

```python
gpu_forced = score_any("vina_score",
                       molecules=["CCO"],
                       pocket=my_pocket,
                       gpu="force")
# Raises AdapterUnavailable when the Vina-GPU binary is missing.
```

### 4.3 Global opt-in via env

```bash
export MOLMETAL_GPU_ACCEL=1
export MOLMETAL_GPU_DEVICE=0
python my_screening_script.py   # every score_any() call now routes via GPU when possible
```

### 4.4 Numerical parity check

The GPU wrappers expose the same call signature as the CPU wrappers and produce
bit-for-bit comparable outputs (within fp32 tolerance on the GPU batch path).
For `boltz_affinity` the GPU batched path calls `predict_pic50_batch` with the
same model weights, so values match within `1e-4`. For `vina_score` the GPU
backend runs the same Monte-Carlo search, so values match within the
exhaustiveness tolerance.

## 5. Failure mode + fallback contract

When `gpu="auto"` (or env opt-in) and the GPU wrapper raises
`AdapterUnavailable` (no binary, no checkpoint, no CUDA) OR a torch CUDA
OOM error:

1. `torch.cuda.empty_cache()` is invoked to free VRAM (best-effort).
2. A WARNING log line is emitted: `score_any(metric, gpu='auto'): GPU wrapper <tool> unavailable/OOM; falling back to CPU <cpu_tool>`.
3. The CPU wrapper is invoked **once**.
4. The returned dict carries `status="MEASURED_FALLBACK"`, `gpu_mode="auto"`, and `fallback_to_cpu=True`.

When `gpu="force"` (or `gpu=True` without "auto" intent) the GPU failure is
re-raised as `AdapterUnavailable` — no silent CPU substitution. The caller
must inspect the exception to see whether to retry with `gpu="auto"` or `gpu=False`.

When `gpu=False` (default) the GPU layer is bypassed entirely. The Wf-43
return shape is preserved (no `gpu_mode` / `fallback_to_cpu` keys).

## 6. Audit + bench

* `molmetal/reports/wf50_gpu_audit/gpu_readiness_matrix.json` — Phase 1 audit (5 surfaces GPU-ready today).
* `molmetal/reports/wf50_gpu_wrappers/wrapper_bench.json` — Phase 3 wrapper speedup data.
* `molmetal/reports/wf50_score_any_gpu/routing_integration.json` — Phase 4 integration audit (this ship).
* `molmetal/tests/test_score_any_gpu_routing.py` — ~20 unit tests covering the routing layer (CPU mode / GPU mode / auto-detect / OOM fallback / env-var opt-in / device selector / numerical parity).

## 7. Test gate

```bash
pytest molmetal/tests/test_score_any_gpu_routing.py -v
pytest molmetal/tests/test_score_any.py -v            # Wf-43 backward compat
pytest molmetal/tests/test_gpu_wrappers.py -v        # Wf-50 Phase 2 wrapper tests
```

All three suites must pass.
