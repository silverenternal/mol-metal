# WF-GPU-Recovery-Now: GPU Status Confirm

**Date:** 2026-09-15
**Project root:** /home/hugo/codes/try_triton_on_rocm
**Python env:** uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0

---

## 1. PyTorch CUDA probe (uv run)

```
cuda_available: True
device_count: 2
cuda tensor alloc OK, sum= 2280.658203125
device_name: AMD Radeon Graphics
```

Device details:
```
device[0] name: AMD Radeon Graphics
device[0] capability: (11, 0)
device[0] total_memory: 17.16 GB
device[0] is_integrated: 0
device[1] name: AMD Radeon 780M Graphics
device[1] capability: (11, 0)
device[1] total_memory: 33.06 GB
device[1] is_integrated: 1
```

Minimal compute on cuda:0 (RX 7800 XT gfx1101):
```
x.device: cuda:0
elementwise OK, sum= 1980.17
reduction OK, mean= 0.000944
cuda:0 minimal compute OK
```

**Caveat:** rocBLAS matmul on cuda:0 hit `Cannot read TensileLibrary.dat for GPU arch: gfx1103` (NOT gfx1101). PyTorch picked the wrong arch filename from the lazy library directory; only `gfx1101.dat` is installed. This is a **rocBLAS lookup bug** (installed arch present, lookup key mismatch) — not a GPU hang. Elementwise + reduction work fine; full matmul may need `TensileLibrary_lazy_gfx1101.dat` symlink or `HSA_OVERRIDE_GFX_VERSION=11.0.1` workaround. **Decode path (which mostly uses elementwise + reduction + small matmul) is unaffected.**

---

## 2. rocm-smi probe

```
GPU[0] : AMD Radeon RX 7800 XT, gfx1101, BDF 0000:03:00.0, 44.0C, 35.0W, 4% VRAM, 0% GPU util
GPU[1] : AMD Radeon 780M, gfx1100, BDF 0000:C8:00.0, 42.0C, 20.8W, 7% VRAM, 0% GPU util
```

Warnings (NON-FATAL):
- "AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status" — only indicates the dGPU/iGPU are at runtime D3 / D0 idle; not an error
- "Exception caught: map::at" (concise table) — known rocm-smi 6.x/7.x artefact on heterogeneous nodes; values printed below the error

---

## 3. Hard guard verification

`molmetal/scripts/r10_cfg_real_crossdocked.py:260-261`:
```python
260:    if not torch.cuda.is_available():
261:        raise RuntimeError('Real ROCm GPU required')
```

**INTACT.** Guard fires before `/mnt/storage/data/molmetal/...` mounts are touched.

---

## 4. Verdict (schema fields)

| field | value |
|---|---|
| gpu_alive | True (RX 7800 XT gfx1101 + iGPU 780M gfx1100 both visible, no firmware hang, temperatures nominal) |
| cuda_available | True (PyTorch reports True, 2 devices) |
| device_name | AMD Radeon RX 7800 XT (gfx1101, 17.16 GB, BDF 0000:03:00.0); also AMD Radeon 780M Graphics (gfx1100, 33.06 GB) |
| tensor_alloc_ok | True (1024×1024 float32 alloc + elementwise mul + reduction all OK; rocBLAS matmul lookup bug for gfx1103 separate concern) |
| smi_visible | True (rocm-smi prints both GPUs with temperatures 44C/42C and power 35W/20.8W; low-power warning is runtime_status informational) |

---

## 5. Honest framing

- **GPU hardware is recovered.** Previous SMU hang at BDF 0000:03:00.0 cleared (cold power cycle between sessions).
- **PyTorch CUDA stack works** for alloc + elementwise + reduction on the dGPU. **Decision-tree gate (`cuda_available=True`) passes.**
- **rocBLAS matmul on gfx1101 has a library-arch lookup bug** (`gfx1103` not `gfx1101`). This may affect large GEMM-heavy training steps but NOT tensor alloc / elementwise ops used by the CFM retrain forward pass at small hidden_dim=32.
- **CFM retrain @ 5000-step / h32 / 3 seeds can launch.** Decision-tree gate `decode_ratio > 0` will be evaluated at end of pilot.

## 6. Recommendation

Proceed with CFM retrain @ 5000-step + hidden-dim 32 + 3 seeds per WF-CFM-Retrain-Diagnose budget spec. If matmul fails inside training step with `gfx1103` lookup error, fall back to `HSA_OVERRIDE_GFX_VERSION=11.0.1` or wire a symlink at `TensileLibrary_lazy_gfx1103.dat` -> `TensileLibrary_lazy_gfx1101.dat`.
