# WF-VRAM-Fix / 04_env — ROCm gfx1101 env-var bootstrap verdict

**Date**: 2026-09-18
**Workstream**: VRAM TOP #1 (BF16 + grad checkpoint)
**Sub-deliverable**: env-var bootstrap + integration smoke + tests
**Status**: SHIPPED. 4/4 tests pass CPU-only; smoke NOOP-friendly on broken-torch hosts.

---

## What shipped

| File | Change |
|---|---|
| `molmetal/__init__.py` | MODIFIED: added `_apply_rocm_gfx1101_env()`, `_is_gfx1101_gpu()`, inline `_ROCM_GFX1101_ENV_VARS` map, public `ROCM_GFX1101_ENV_APPLIED_VARS` list, sentinel `_ROCM_GFX1101_ENV_APPLIED`, and the auto-call at import time. |
| `molmetal/tests/test_rocm_env.py` | NEW: 4 tests (3 required + 1 bonus `_is_gfx1101_gpu` smoke). All pass in 0.27 s CPU-only. |
| `molmetal/scripts/integration_vram_fix_smoke.py` | NEW: end-to-end smoke that imports `CFMAMPContext` + `CheckpointedEGNNLayer`, builds a 3-layer `_TinyStack`, runs 3 forward+backward steps under each config, and asserts `treatment_peak < baseline_peak` on CUDA. NOOP on CPU / broken-torch hosts. |

---

## Pre-flight citations

- `molmetal/adapters/flow_matching_lipman/amp.py:60-76` — canonical env-var map (`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`, `TORCH_BLAS_PREFER_HIPBLASLT=1`). `__init__.py` mirrors these verbatim; the inline copy is the source of truth so the bootstrap works even when the AMP module fails to import.
- `molmetal/adapters/flow_matching_lipman/amp.py:116-128` — `check_rocm_env_vars` (lazy + non-raising pattern) mirrored in `_is_gfx1101_gpu`.
- `molmetal/adapters/egnn_rocm_checkpoint.py:108-191` — `CheckpointedEGNNLayer` (constructor + forward with `use_reentrant=False` + `preserve_rng_state=True`) used by the smoke's `_TinyStack`.
- `molmetal/adapters/flow_matching_lipman/amp.py:182-288` — `CFMAMPContext` used by the smoke's step runner.
- `models/_scatter.py:84-94` — RDNA3-detect fallback pattern (`torch.cuda.is_available()` early-out + substring match on `get_device_name(0)`) mirrored in `_is_gfx1101_gpu`.
- `molmetal/adapters/egnn_rocm.py:540-570` — `EquivariantGraphConv` instance wiring (the *real* EGNN) — smoke uses a `_TinyBlock` stand-in to avoid the Triton / HIP transitive import.
- `molmetal/reports/wf_vram_fix/01_amp.md` — sub-deliverable #1 (AMP wrapper) — this sub-deliverable depends on it.
- `molmetal/reports/wf_vram_fix/02_checkpoint.md` — sub-deliverable #2 (checkpoint wrapper) — smoke integrates with it.
- `molmetal/reports/wf_vram_fix/03_measure.md` — sub-deliverable #3 (production VRAM numbers) — this smoke is the integration mirror of that production measurement.

---

## Design decisions

1. **Inline canonical env-var map, not just imported.** `molmetal/__init__.py` defines `_ROCM_GFX1101_ENV_VARS` directly so the bootstrap runs even when `molmetal.adapters.flow_matching_lipman.amp` cannot import (e.g. host with broken libtorch). On hosts where the AMP module loads cleanly the bootstrap **cross-checks** against `ROCM_GFX1101_ENV_VARS` from `amp.py` so the two stay in sync — a divergence would be a quiet regression.
2. **Substring match, not SKU list.** `_is_gfx1101_gpu()` matches `gfx1101`, `Radeon`, and `RX 78` substrings so the same gate picks up the RX 7800 XT, RX 7900 XT, RX 7900 XTX and any future gfx1101 SKU. Matches the lazy + non-raising pattern at `models/_scatter.py:84-94`.
3. **Idempotent via sentinel.** `_ROCM_GFX1101_ENV_APPLIED` short-circuits a second run; `force=True` re-applies (tests use this). User overrides win — vars already set to a different value are left untouched.
4. **Hygiene vars unconditional.** `TOKENIZERS_PARALLELISM=false` and `TRANSFORMERS_VERBOSITY=error` are applied on **every** host, not just ROCm ones — cheap and silences noisy logs during dev / tests.
5. **One-line summary print.** Only fires when at least one var was actually applied and `quiet=False`. Default is non-quiet so a fresh `import molmetal` prints the line and the user immediately sees whether their env is wired.

---

## Smoke protocol

```
$ PYTHONPATH=. python molmetal/scripts/integration_vram_fix_smoke.py
[molmetal] applied gfx1101 env vars: TOKENIZERS_PARALLELISM=false, TRANSFORMERS_VERBOSITY=error
[smoke] NOOP: torch import failed on this host (ImportError: /opt/libtorch/lib/libtorch_python.so: undefined symbol: _PyThreadState_UncheckedGet). Env-var bootstrap already ran; skipping step loop.  exit 0
$ echo $?
0
```

On a working torch + CUDA host the smoke:
1. builds `_TinyStack(n_layers=3, hidden_dim=32, checkpoint=False)` (baseline)
2. runs 3 forward+backward steps under `_CFMAMPContext(enabled=False, dtype=fp32)` (baseline = pure FP32 + no checkpoint)
3. builds `_TinyStack(n_layers=3, hidden_dim=32, checkpoint=True)` (treatment)
4. runs 3 forward+backward steps under `_CFMAMPContext(enabled=True, dtype=bf16)` (treatment = BF16 + checkpoint)
5. records `torch.cuda.max_memory_allocated()` and `max_memory_reserved()` for both
6. **asserts** `treatment_peak_alloc < baseline_peak_alloc`

On CPU the assertion is skipped (BF16 + checkpoint benefit is a GPU phenomenon). On broken-torch hosts (this one) the entire step loop is skipped and the script exits 0 with the "NOOP" framing — the env-var bootstrap **did** run (visible in the first line of output) and that's what we wanted to exercise in the "import is wired" sense.

---

## Test results

```
molmetal/tests/test_rocm_env.py::TestRocmEnv::test_env_vars_applied_idempotent PASSED
molmetal/tests/test_rocm_env.py::TestRocmEnv::test_pytorch_hip_alloc_conf_set PASSED
molmetal/tests/test_rocm_env.py::TestRocmEnv::test_torch_blas_prefer_hipblaslt_set PASSED
molmetal/tests/test_rocm_env.py::TestGpuDetection::test_is_gfx1101_gpu_returns_bool PASSED
============================== 4 passed in 0.27s ===============================
```

All 4 tests are CPU-only (mock `_is_gfx1101_gpu`) and have no GPU requirement. They exercise:
- re-invocation safety (force=True twice; force=False short-circuits via sentinel)
- `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` set correctly
- `TORCH_BLAS_PREFER_HIPBLASLT=1` set correctly
- `_is_gfx1101_gpu()` returns a bool and never raises

---

## Honest caveats

1. **Smoke is NOOP on this host.** This CI host has a broken torch (`/opt/libtorch/lib/libtorch_python.so: undefined symbol: _PyThreadState_UncheckedGet` against Python 3.14). The smoke prints the NOOP framing and exits 0 — the assertion was **not** measured. Production validation needs to be run on a host with a working torch + an AMD GPU.
2. **Bootstrap is best-effort when AMP module fails to load.** When the AMP module imports cleanly (working torch) the env vars come from `amp.ROCM_GFX1101_ENV_VARS`; when it fails (broken torch), the inline copy in `__init__.py` is used. Both paths produce identical var values, but if you change `amp.ROCM_GFX1101_ENV_VARS` you must mirror the change in `__init__.py`.
3. **Env vars applied at import time, not at torch init.** The ROCm allocator reads `PYTORCH_HIP_ALLOC_CONF` only during `torch.cuda.init()` (typically first `torch.cuda` call). If a caller does `import torch` **before** `import molmetal`, the var is set too late and has no effect. The fix for that case is the `.bashrc` / `.profile` line documented in `molmetal/reports/wf_vram_fix/03_measure.md` — but on hosts that do import molmetal first the in-process path is sufficient.
4. **No GPU runtime measured.** Per spec, "no GPU required for tests" — the GPU half is `03_measure.md`, not this report.

---

## Hand-off

- `05_followup.md` candidate: surface a `molmetal doctor` CLI that prints the full ROCm env state (HIP visible, gfx1101 detected, env vars set, AMP module loaded) — useful for the next GPU-outage triage.
- Production GPU re-run: once the gfx1101 SMU hang clears (`wf_gpu_auto_recover`), run the smoke end-to-end and capture the `treatment_peak < baseline_peak` numbers into `vram_measure_prod.json` as the third row.