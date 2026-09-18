# WF-GPU-Diag-Fix — Final Verification Report

**Date:** 2026-09-14
**Workflow:** WF-GPU-Diag-Fix-Verify (Phase 3 of GPU-outage investigation)
**Project root:** `/home/hugo/codes/try_triton_on_rocm`
**Environment:** Arch Linux (7.2.4-zen2-1-zen), uv-managed Python 3.12, ROCm 7.2 / triton-rocm 3.8.0 / torch 2.14.0+rocm7.2, RX 7800 XT (gfx1101, wave64)
**Sandbox:** no sudo, no `modprobe`, no kernel-module reload

---

## 1. TL;DR

| Phase | torch CUDA | rocm-smi GPU[0] | Status |
|-------|------------|------------------|--------|
| **Pre-fix** | `False` (0 devices) | low-power / `error` | SMU firmware hang on RX 7800 XT |
| **Env patch applied** (`~/.bashrc` + `~/.profile` ROCm exports, see `env_fix.md`) | `False` (0 devices) | low-power / `error` | unchanged — env vars cannot reset a hung SMU |
| **Probe attempted** | n/a — hard guard `RuntimeError('Real ROCm GPU required')` raised at `r10_cfg_real_crossdocked.py:261` | n/a | probe **did not run** |

**Honest framing (mandatory):**
- **The env-var patch is correct and necessary** (it is the right hygiene), but it **does NOT bring the GPU back**. The SMU firmware on the discrete RX 7800 XT is unresponsive; userspace env-var tuning cannot recover a kernel-level firmware hang.
- **CFM retrain probe was BLOCKED** before the first training step ran. The script enforces `torch.cuda.is_available() == True` (line 260-261) by design — it cannot be coerced into a `--gpu-binary fake` fallback without code modification, and that modification is out of scope for WF-GPU-Diag-Fix.
- **No fix is possible without sudo + a power-cycle / module reload** (or cold boot). Documented in §5.

---

## 2. Pre-fix vs Post-fix GPU Status

### 2.1 Pre-fix (before any user action, baseline)

```
$ source ~/.bashrc && uv run python -c "import torch; ..."
cuda_available: False
device_count: 0
device_name: N/A

$ rocm-smi
WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status
GPU[0] : Navi 32 [RX 7700 XT / 7800 XT], runtime_status=error, VRAM 1% busy, GPU 0%
GPU[1] : HawkPoint1 (iGPU, gfx1103), 65 °C, 49.6 W, 2800 MHz, VRAM 90%, GPU 14%
```

### 2.2 Env patch applied

Per `env_fix.md`, 10 env vars were added to `~/.bashrc` (lines 16-32) and `~/.profile` (lines 5-20): `ROCM_PATH`, `HIP_PATH`, `HIP_PLATFORM`, `LD_LIBRARY_PATH` (augmented), `PATH` (augmented), `HIP_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `PYTORCH_ROCM_ARCH=gfx1101`, `HSA_ENABLE_INTERRUPT`, `HSA_FORCE_FINE_GRAIN_PCIE`, `PYTORCH_ROCM_DEVICE_REGULAR_BLOCK`.

### 2.3 Post-fix (current state, after env patch)

```
$ source ~/.bashrc && uv run python -c "import torch; ..."
cuda_available: False
device_count: 0
device_name: N/A
torch: 2.14.0+rocm7.2
hip: 7.2.53211

$ rocm-smi --showuse --showpower --showtemp
WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status
GPU[1] : 66 °C, 64.3 W, GPU 4%
GPU[0] : GPU 0%   (still in error state)
```

**State change:** NONE. The env vars are present (verified in the live shell); the GPU is still unreachable.

### 2.4 KFD topology snapshot (post-fix)

```
Node 1 (RX 7800 XT): simd_count=120, wave_front_size=32, gfx_target_version=110001
Node 2 (iGPU)      : simd_count=24,  wave_front_size=32, gfx_target_version=110003
```

KFD *sees* the GPU (driver-level enumeration works) but the GPU cannot come out of low-power D-state because the SMU firmware is unresponsive. The KFD driver happily creates a queue via ioctl (verified — `KFD_IOC_CREATE_QUEUE` returned `rc=0, queue_id=0x1000000000000`), but `hsa_init` returns `HSA_STATUS_ERROR (0x1000)` because the runtime cannot reach a usable compute agent.

### 2.5 Kernel log (definitive evidence — `journalctl -b`)

```
amdgpu 0000:03:00.0: ring jpeg_dec uses VM inv eng 4 on hub 8
amdgpu 0000:03:00.0: ring mes_kiq_3.1.0 uses VM inv eng 14 on hub 0
amdgpu 0000:03:00.0: [drm] Cannot find any crtc or sizes
amdgpu 0000:03:00.0: SMU: No response msg_reg: 3b resp_reg: 0
amdgpu 0000:03:00.0: [SetDfCstate] failed!
amdgpu 0000:03:00.0: Failed to disallow df cstate
amdgpu 0000:03:00.0: SMU: No response msg_reg: 3b resp_reg: 0
amdgpu 0000:03:00.0: Failed to retrieve enabled ppfeatures!
amdgpu 0000:03:00.0: SMU: No response msg_reg: 3b resp_reg: 0
amdgpu 0000:03:00.0: SMC failed to set mp1 state 2, -62
amdgpu 0000:03:00.0: SMU is resuming...
amdgpu 0000:03:00.0: SMU: No response msg_reg: e resp_reg: 0
amdgpu 0000:03:00.0: Failed to SetDriverDramAddr!
amdgpu 0000:03:00.0: Failed to setup smc hw!
amdgpu 0000:03:00.0: resume of IP block <smu> failed -62
amdgpu 0000:03:00.0: amdgpu_device_ip_resume_phase2 failed during unwind: -62
amdgpu 0000:03:00.0: can't suspend (amdgpu_pmops_runtime_suspend [amdgpu] returned -62)
amdgpu 0000:03:00.0: VM memory stats for proc (0) task (0) is non-zero when fini
```

This is a **firmware-level fault** on the SMU IP block. It requires:
- cold power-cycle of the GPU (PCIe hot-reset, ideally full system reboot), or
- `sudo rmmod amdgpu && sudo modprobe amdgpu` (not available in sandbox), or
- `sudo echo 1 > /sys/bus/pci/devices/0000:03:00.0/remove` followed by PCI rescan (also not available).

---

## 3. CFM Retrain Probe — NOT EXECUTED

### 3.1 Why

The harness `molmetal/scripts/r10_cfg_real_crossdocked.py` enforces a hard guard:

```python
# line 260-261
if not torch.cuda.is_available():
    raise RuntimeError('Real ROCm GPU required')
```

The user instruction explicitly passed `--gpu-binary scripts/_fake_vina.sh`, which is the *docking* fallback (Vina vs. a fake docking script). That flag does NOT bypass the GPU requirement. It only swaps out the docking backend; the CFM training step still requires `torch.cuda`.

### 3.2 What happened when invoked

```
$ uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --seeds 42 --train-steps 100 --n-train 8 --n-samples 4 \
    --hidden-dim 32 --n-layers 2 --lr 0.0001 --vocab-mask \
    --bond-head learned --output-dir molmetal/reports/wf_gpu_diag/probe/ \
    --gpu-binary scripts/_fake_vina.sh

Traceback (most recent call last):
  File ".../molmetal/scripts/r10_cfg_real_crossdocked.py", line 404, in <module>
    raise SystemExit(main())
  File ".../molmetal/scripts/r10_cfg_real_crossdocked.py", line 261, in main
    raise RuntimeError('Real ROCm GPU required')
RuntimeError: Real ROCm GPU required
```

Wall-clock for the failed start: ~2 s (import overhead + the guard).

### 3.3 Probe metrics — n/a (probe did not run)

| Metric | Value |
|--------|-------|
| `n_decoded` | n/a — `train_step` not entered |
| `bond_loss` (initial / final) | n/a — no gradient step |
| `n_request` | n/a |
| `wall_seconds` | ~2 s (import + guard) |
| `n_decoded / n_request` | 0 / 0 |

---

## 4. Root-cause Summary (one-screen)

| Layer | Symptom | Root cause | Fixable from userspace? |
|-------|---------|------------|--------------------------|
| Kernel module | `amdgpu` loaded, `/dev/kfd` open, KFD topology populated | driver loaded fine, GPU probe OK at driver level | n/a |
| GPU firmware | `SMU: No response`, `Failed to SetDriverDramAddr!`, `resume of IP block <smu> failed -62` | **SMU IP block stuck in low-power state** (RDNA3 power-management bug, likely triggered by runtime-PM auto-suspend) | NO — needs kernel module unload or hardware reset |
| HSA runtime | `hsa_init → HSA_STATUS_ERROR (0x1000)`; `hsa_iterate_agents → 4107 NOT_INITIALIZED`, 0 agents | driver can't expose a working compute agent because SMU is down | NO |
| KFD ioctl | `KFD_IOC_GET_VERSION → 1.23` (works); `KFD_IOC_CREATE_QUEUE → rc=0` (works) | driver queues are decoupled from SMU availability | n/a |
| torch HIP | `hipInit → 101 hipErrorInvalidDevice`; `torch.cuda.is_available() → False` | HIP can't reach a usable device because of HSA init failure | NO |
| env vars | `ROCM_PATH`, `HIP_PLATFORM`, `HSA_OVERRIDE_GFX_VERSION` etc. set | env hygiene correct, but irrelevant when the kernel can't reach the GPU | already patched |

**Verdict:** the outage is in layer 2 (firmware). Layers 3-6 are correct cascade failures of layer 2. Env vars cannot fix this.

---

## 5. Manual Recovery Steps (USER ACTION REQUIRED, sandbox cannot)

The user must run these with `sudo`. None can be done inside the Claude sandbox.

### 5.1 Option A — cold power-cycle (preferred)

1. Save all work.
2. `sudo shutdown -h now` (full power-off; not reboot, which on some boards leaves PCIe aux power on).
3. Wait 30 s.
4. Power on, boot Arch.
5. Re-source `~/.bashrc`, re-run the verification command — torch should now see the GPU.

### 5.2 Option B — runtime module reload (if cold cycle is unavailable)

```bash
sudo -i
# kill zombie processes that AMD-SMI shows attached to GPU 0
pkill -9 -f 'python3.12.*molmetal' 2>/dev/null || true
sleep 2
# unload and reload amdgpu
modprobe -r amdgpu
sleep 5
modprobe amdgpu ppfeaturemask=0xfff7bfff cwsr_enable=1
# verify
rocm-smi
```

### 5.3 Option C — PCIe hot-reset (last resort)

```bash
sudo -i
echo 1 > /sys/bus/pci/devices/0000:03:00.0/remove
sleep 2
echo 1 > /sys/bus/pci/rescan
sleep 5
rocm-smi
```

### 5.4 Disable runtime-PM to prevent recurrence

```bash
echo 'options amdgpu runpm=0' | sudo tee /etc/modprobe.d/amdgpu-no-runtime-pm.conf
sudo update-initramfs -u   # or mkinitcpio -P on Arch
```

Once the GPU is back, the existing env patch (`env_fix.md`) is sufficient — no further tuning needed.

---

## 6. Probe Output Artefacts

The output directory `molmetal/reports/wf_gpu_diag/probe/` was created but contains no JSON, since the probe never reached the training loop. (The mkdir succeeded; the script aborted before any `save_metrics` call.)

---

## 7. Recommendation

**Do NOT spend more time on env-var tuning.** The diagnosis in `diagnosis.md` and this verification are conclusive. **The blocker is physical**, not configuration. Until the user performs one of §5.1-§5.3, **WF-3 (Round-12 CFM retrain pilot, N=10×3) and any GPU-required workflows remain blocked.**

**Action items:**

1. **User:** perform §5.1 (cold power-cycle) — most reliable. Then re-run §1 verification: `source ~/.bashrc && uv run python -c "import torch; print(torch.cuda.is_available())"`. Expected output: `True`.
2. **If §5.1 fixes it:** invoke `molmetal/scripts/r10_cfg_real_crossdocked.py` with the probe flags from §1 of the parent task and capture real metrics.
3. **If §5.1 does NOT fix it:** escalate to §5.2 then §5.3. If all three fail, the GPU has a hardware fault (replace or RMA).

**For paper-writing in the meantime:** §4 Table 1 should continue to use the **cite-only SOTA column** (`WF-3-CiteOnly-SOTA`) and the **Lambda-only GPU-free mini-pilot** (`WF-Lambda-Only-MiniPilot`), per the post-R10/R11 plan (`TODO/pending/20_post_r10_r11_action_plan.md`). Do not promise GPU numbers until §5 has been executed.

---

## 8. Cross-references

- Diagnosis: `molmetal/reports/wf_gpu_diag/diagnosis.md` (Phase 1, root-cause SMU firmware hang)
- Env patch: `molmetal/reports/wf_gpu_diag/env_fix.md` (Phase 2, 10 vars added to `~/.bashrc` + `~/.profile`)
- GPU-free fallback pilot: TODO task #482 `WF-Lambda-Only-MiniPilot`
- Cite-only SOTA column: TODO task #457 `WF-3-CiteOnly-SOTA`
- Post-R10/R11 plan: `TODO/pending/20_post_r10_r11_action_plan.md`
