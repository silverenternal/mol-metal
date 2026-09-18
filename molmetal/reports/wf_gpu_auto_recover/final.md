# WF-GPU-Auto-Recover — Final Report

**Date:** 2026-09-15
**Workflow:** WF-GPU-Auto-Recover (single invocation)
**GPU recovered this cycle:** NO

## 1. GPU Probe (step 1 of WF)

| Probe | Result |
|---|---|
| `torch.cuda.is_available()` | **False** |
| `torch.cuda.device_count()` | **0** |
| `torch.__version__` | 2.14.0+rocm7.2 |
| `/dev/kfd` | present |
| `/dev/dri/renderD128` | present |
| `amdgpu` kernel module | loaded |
| `rocminfo` (HSA userland) | **HSA_STATUS_ERROR** (generic failure) |

JSON artifact: `molmetal/reports/wf_gpu_auto_recover/cfm_probe/gpu_probe.json`

Conclusion: kernel driver is loaded but the ROCm userland runtime (`rocminfo`, `libhsa-runtime`) cannot enumerate the GPU. This matches the prior iGPU-probe failure mode (driver OK, userland broken). PyTorch's `torch.cuda.is_available()` returns False as a consequence.

## 2. CFM Retrain Probe (step 2 of WF)

**NOT executed.** Per WF spec: "if cuda_available=True run CFM retrain @ 5000-step". Since `cuda_available=False`, the CFM probe command was skipped — running the script on a CUDA-less box would silently fall back to CPU and produce non-comparable numbers, violating honest-framing.

The intended command was:

```
uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
  --seeds 42 0 1234 --train-steps 5000 --n-train 32 --n-samples 16 \
  --hidden-dim 32 --n-layers 2 --lr 0.0001 --vocab-mask --bond-head learned \
  --output-dir molmetal/reports/wf_gpu_auto_recover/cfm_probe/ \
  --gpu-binary scripts/_fake_vina.sh
```

Both required files exist:
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py`
- `/home/hugo/codes/try_triton_on_rocm/scripts/_fake_vina.sh`

so the probe is ready to launch the moment `cuda_available=True`.

## 3. Comparison to baseline

Cannot compare — the probe did not run. Recording expected slots for the next successful run:

| metric | 2000-step baseline | 5000-step iGPU baseline | this run (5000-step, hd=32) |
|---|---|---|---|
| `n_decoded` | 0 | NOT_MEASURED | NOT_MEASURED |
| `decode_ratio` | 0/384 | NOT_MEASURED | NOT_MEASURED |
| `bond_loss_final` | (recorded in r10 logs) | NOT_MEASURED | NOT_MEASURED |
| wall-clock | (recorded in r10 logs) | NOT_MEASURED | NOT_MEASURED |

If the next probe recovers the GPU, the table above will be populated with measured numbers in this same `final.md` (append, do not overwrite history).

## 4. TODO/pending/21 update

**Not updated.** Per spec: "If decode_ratio > 0.5: update TODO/pending/21_lambda_model_coupling.md + paper §4 with MEASURED number." Since decode_ratio was not measured, the gate condition was not met.

## 5. Next action (for the user / next WF-GPU-Auto-Recover invocation)

1. Reboot OR reload `amdgpu` and restart `rocm-smi` / `hsamgr` so the userland runtime re-enumerates the device.
2. Re-run this workflow. As soon as `rocminfo` returns a non-error HSA status AND `torch.cuda.is_available()` flips to True, the CFM probe will execute automatically.

Honest-framing note: this report records a NO-OP cycle. No fabricated decode ratios, no assumed GPU times, no fabricated lift numbers. The probe is staged and waiting on the runtime.
