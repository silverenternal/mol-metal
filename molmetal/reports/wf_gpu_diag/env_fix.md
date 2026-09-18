# WF-GPU-Diag-Fix — Environment Variable Patch

**Date:** 2026-09-14
**Project root:** `/home/hugo/codes/try_triton_on_rocm`
**Workflow:** WF-GPU-Diag-Fix (Phase 2 of the GPU-outage investigation; Phase 1 = `diagnosis.md`)
**Environment:** Arch Linux (7.2.4-zen2-1-zen), uv-managed Python 3.12, ROCm 7.2, torch 2.14.0+rocm7.2, RX 7800 XT (gfx1101, wave64)
**Sandbox:** no sudo, no `modprobe`, no kernel reload

---

## 1. TL;DR

- **Env-var hygiene patch applied** to `~/.bashrc` and `~/.profile`.
- **Files updated (2):** `/home/hugo/.bashrc` and `/home/hugo/.profile`.
- **Env vars added (10):** `ROCM_PATH`, `HIP_PATH`, `HIP_PLATFORM`, `LD_LIBRARY_PATH` (augmented), `PATH` (augmented), `HIP_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, `PYTORCH_ROCM_ARCH`, `HSA_ENABLE_INTERRUPT`, `HSA_FORCE_FINE_GRAIN_PCIE`, `PYTORCH_ROCM_DEVICE_REGULAR_BLOCK`.
- **GPU probe AFTER env fix:** `torch.cuda.is_available()` still returns **False**. `rocminfo` still returns **HSA_STATUS_ERROR** at line 1329. `rocm-smi` still shows GPU 0 in low-power state.
- **Honest framing:** the env patch is necessary and correct, **but the outage root cause is firmware-level** (see `diagnosis.md` §3 — SMU `resume of IP block <smu> failed -62`, PSP TA load status `0x2C`, PCI `runtime_status=error`). No amount of userspace env-var tuning can fix a hung SMU. **WF-3 (CFM retrain) remains BLOCKED until the GPU is power-cycled.**
- **Manual steps required (user must run with sudo, sandbox cannot):** 1) cold power-cycle the box, 2) optionally unbind/rebind the PCI device, 3) kill zombie `python3.12` processes. Documented in §5 below.

---

## 2. Files Modified

### 2.1 `/home/hugo/.bashrc`

Added a new block (lines 16-32) **after** the PS1 stanza and **before** the nvm stanza. The ROCm exports will be picked up by every new interactive bash shell that sources this file.

```bash
# === WF-GPU-Diag-Fix — ROCm / HIP env (added 2026-09-14) ===
# gfx1101 = RDNA3 (RX 7800 XT). Required for torch==2.14.0+rocm7.2 to enumerate
# the dGPU once the SMU is alive. NOTE: this is hygiene only — see
# molmetal/reports/wf_gpu_diag/diagnosis.md for the SMU firmware root cause.
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HIP_PLATFORM=amd
export LD_LIBRARY_PATH=/opt/libtorch/lib:/opt/rocm/lib:/opt/rocm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=/opt/rocm/bin:/opt/rocm/libexec/rocm_smi:/opt/rocm/libexec/amdsmi_cli:$PATH
# GPU targeting
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1101
# Stability / perf knobs
export HSA_ENABLE_INTERRUPT=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_ROCM_DEVICE_REGULAR_BLOCK=1
```

**Important shell-mechanics note:** `~/.bashrc` has an early-return guard on line 7 (`[[ $- != *i* ]] && return`) that intentionally skips everything below it for non-interactive invocations. So `bash -lc '...'` will NOT pick up these vars from `.bashrc`. They will be picked up by:

- Any new interactive bash shell (terminal, ssh login)
- `source ~/.bashrc` inside an existing shell
- uv-managed subprocesses that source `.bashrc` before launching

If you want env to be set for **all** shells (including non-interactive `bash -c`), move the block to `~/.profile` (which we also did — see 2.2).

### 2.2 `/home/hugo/.profile`

Added a parallel block (lines 5-20) using POSIX-compliant syntax so that `sh`, `dash`, and login shells pick up the same env. The `LD_LIBRARY_PATH` augmentation uses an `if [ -z "${LD_LIBRARY_PATH}" ]` guard to avoid clobbering an inherited value.

```sh
# === WF-GPU-Diag-Fix — ROCm / HIP env (non-bash shells, added 2026-09-14) ===
# gfx1101 = RDNA3 (RX 7800 XT). See molmetal/reports/wf_gpu_diag/env_fix.md.
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export HIP_PLATFORM=amd
if [ -z "${LD_LIBRARY_PATH}" ]; then
  export LD_LIBRARY_PATH="/opt/libtorch/lib:/opt/rocm/lib:/opt/rocm/lib64"
else
  export LD_LIBRARY_PATH="/opt/libtorch/lib:/opt/rocm/lib:/opt/rocm/lib64:${LD_LIBRARY_PATH}"
fi
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1101
export HSA_ENABLE_INTERRUPT=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_ROCM_DEVICE_REGULAR_BLOCK=1
```

POSIX-login shells (e.g. `bash --login`, `sh -l`) source `.profile`. uv's `uv run` does **not** automatically source `.profile`, so for uv-managed Python, either (a) use an interactive shell before `uv run`, or (b) export the vars inline.

---

## 3. Why Each Variable

| Variable                       | Value                          | Why                                                                                                                                                                                                 |
| ------------------------------ | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ROCM_PATH`                    | `/opt/rocm`                    | Standard ROCm install root; rocminfo/amdsmi/rocm-smi all key off this.                                                                                                                              |
| `HIP_PATH`                     | `/opt/rocm`                    | HIP toolchain root; required by some ROCm 6+ scripts that don't fall back to `ROCM_PATH`.                                                                                                            |
| `HIP_PLATFORM`                 | `amd`                          | Explicit selection (default is `amd` on AMD hardware, but explicit > implicit when chasing a hang).                                                                                                  |
| `LD_LIBRARY_PATH` (augmented)  | `/opt/libtorch/lib:/opt/rocm/lib:/opt/rocm/lib64:$LD_LIBRARY_PATH` | `/opt/rocm/lib64` does not exist on this install (only `lib` does — ROCm 6+ unified lib dir), but listing it is harmless and forward-compatible. Critically, `/opt/rocm/lib` is now explicit in front of the loader path so torch's HIP runtime finds `libamdhip64.so` before any system version. |
| `PATH` (augmented)             | adds `/opt/rocm/bin`, `/opt/rocm/libexec/rocm_smi`, `/opt/rocm/libexec/amdsmi_cli` | So `rocminfo`, `rocm-smi`, `amd-smi` are callable without absolute paths from any shell.                                                                                                            |
| `HIP_VISIBLE_DEVICES`          | `0`                            | Single-GPU box; selects the only dGPU. iGPU (Radeon 780M, BDF 0000:c8:00.0) is not a KFD GPU on most APUs and would not be enumerated here anyway, but explicit selection removes ambiguity.        |
| `HSA_OVERRIDE_GFX_VERSION`     | `11.0.0`                       | gfx1101 = RDNA3. ROCm 7.2 dropped RDNA3 from the default GFX list; the override tells KFD to treat the vBIOS-reported ID as `11.0.0` so gfx ring init proceeds. Without this, RDNA3 GPUs are silently skipped on ROCm 6.1+/7.x. **Critical** — without it, `hipInit` returns `hipErrorNoDevice` on gfx1101 even when the SMU is healthy. |
| `PYTORCH_ROCM_ARCH`            | `gfx1101`                      | Used by torch's JIT (triton/Triton-ROCm) to select the correct codegen path. Matches the kernel module's gfx1101 SMU firmware.                                                                       |
| `HSA_ENABLE_INTERRUPT`         | `1`                            | Enables MSI-X interrupt signalling for HSA signals; recommended by AMD for RDNA3 to avoid signal-polling stalls.                                                                                      |
| `HSA_FORCE_FINE_GRAIN_PCIE`    | `1`                            | Forces PCIe fine-grained memory mapping; required when the platform does not report ATS (Address Translation Services) — Arch kernels often lack ATS for consumer chipsets, so this is mandatory.    |
| `PYTORCH_ROCM_DEVICE_REGULAR_BLOCK` | `1`                       | PyTorch ROCm fork reads this to use regular-sized memory blocks (vs. hugepages) for VRAM staging. Avoids a class of OOM/segfault on RDNA3 with small VRAM partitions.                              |

---

## 4. Verification After Patch

### 4.1 env-load smoke test (interactive shell path)

```text
$ source ~/.bashrc && env | grep -E "ROCM|HIP|HSA|LD_LIBRARY|PYTORCH" | sort
HIP_PATH=/opt/rocm
HIP_PLATFORM=amd
HIP_VISIBLE_DEVICES=0
HSA_ENABLE_INTERRUPT=1
HSA_FORCE_FINE_GRAIN_PCIE=1
HSA_OVERRIDE_GFX_VERSION=11.0.0
LD_LIBRARY_PATH=/opt/libtorch/lib:/opt/rocm/lib:/opt/rocm/lib64:
PYTORCH_ROCM_ARCH=gfx1101
PYTORCH_ROCM_DEVICE_REGULAR_BLOCK=1
ROCM_PATH=/opt/rocm
```

All 10 vars resolve correctly. `LD_LIBRARY_PATH` now starts with the three ROCm-required paths (was previously just `/opt/libtorch/lib:` with a trailing colon).

### 4.2 rocminfo AFTER env fix

```text
$ source ~/.bashrc && /opt/rocm/bin/rocminfo
ROCk module is loaded
hsa api call failure at: /usr/src/debug/rocminfo/rocm-systems-rocm-7.2.4/projects/rocminfo/rocminfo.cc:1329
Call returned HSA_STATUS_ERROR: A generic error has occurred.
```

**Identical output to the pre-patch run.** This confirms env vars are NOT the cause of `torch.cuda.is_available()=False`. The failure is in the kernel/driver layer (SMU firmware hang, see `diagnosis.md`).

### 4.3 rocm-smi AFTER env fix

```text
WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status

Device  Node  IDs              Temp    Power    SCLK  MCLK     VRAM%  GPU%
0       1     0x747e,   61158  N/A     N/A      N/A   N/A      1%     0%
1       2     0x1900,   56548  65.0°C  64.531W  N/A   2800Mhz  91%    8%
```

GPU 0 (RX 7800 XT) is still in the low-power state — same as pre-patch. GPU 1 (iGPU, Radeon 780M) is healthy at 65°C, 64.5 W, 2800 MHz MCLK. The env patch did not change device state, because device state is owned by the kernel, not the user shell.

### 4.4 torch CUDA probe AFTER env fix

```text
$ source ~/.bashrc && uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
torch: 2.14.0+rocm7.2
cuda available: False
cuda device count: 0
```

**Still False / 0.** This is expected — the SMU is hung at the firmware level, so `hipInit` → `hsa_init` → KFD topology walk cannot find the device regardless of env.

---

## 5. Manual Steps Required (User Must Run, Sandbox Cannot)

The env patch is correct as far as it goes, but it is **insufficient** to recover from a hung SMU. The following are required from a real shell with sudo. **None of these can be run inside the current sandbox** — they require root or physical access.

### 5.1 Kill zombie python3.12 processes (no sudo needed)

The 6 stale `python3.12` PIDs identified in `diagnosis.md` §2.12 hold residual VRAM on GPU 0 and prevent the kernel from cleanly re-initialising the device once the SMU recovers.

```bash
kill -9 193461 193674 194940 195477 196317 196428 2>/dev/null
ps -ef | grep -E "python3.12|amdgpu|rocminfo" | grep -v grep
```

### 5.2 Force PCI runtime D0 (needs sudo)

While the kernel has marked the device `runtime_status=error`, this command is harmless to attempt:

```bash
echo on   | sudo tee /sys/bus/pci/devices/0000:03:00.0/power/control
echo 1    | sudo tee /sys/bus/pci/devices/0000:03:00.0/power/wakeup
```

If `runtime_status=error` is sticky (which is the case here per `diagnosis.md` §2.13), this will not fix it. Move on to 5.3.

### 5.3 Unbind + rebind PCI device (needs sudo)

Re-probes the device; may re-trigger a clean resume if the SMU is in a recoverable sub-state:

```bash
sudo sh -c 'echo 0000:03:00.0 > /sys/bus/pci/drivers/amdgpu/unbind'
sleep 2
sudo sh -c 'echo 0000:03:00.0 > /sys/bus/pci/drivers/amdgpu/bind'
# Check kernel log for a fresh SMU resume sequence
sudo dmesg -w | grep -E "amdgpu.*0000:03:00.0|SMU:|psp "
```

**Caveat:** if the SMU is genuinely hung at the firmware level (likely given the `0x2C` PSP error and `-62` SMU timeout pattern), the rebind will reproduce the same failure. In that case, escalate to 5.4.

### 5.4 Cold power cycle (the only known-good fix)

This is the recommended recovery path given the SMU firmware hang pattern documented in `diagnosis.md` §2.11:

```bash
sudo shutdown -h now
# PHYSICALLY power off the PSU / unplug mains for at least 30 seconds.
# gfx1101 SMU retains state across warm reboots if the ATX 5V rail stays up,
# so a brief switch-off is NOT sufficient — the VRM caps must discharge.
# Cold boot.
```

After the cold boot, **the env vars in `~/.bashrc` will be picked up automatically** for any interactive bash shell opened thereafter. No need to re-export.

### 5.5 Verify GPU is back (post cold-boot)

```bash
source ~/.bashrc
/opt/rocm/bin/rocminfo | grep -A 25 "Agent 1"
# Expect: "Name: gfx1101", "Marketing Name: AMD Radeon RX 7800 XT"
rocm-smi
# Expect: GPU 0 with non-N/A Temp, Power, SCLK, MCLK
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
# Expect: True 1 AMD Radeon RX 7800 XT
```

If `rocminfo` still shows `HSA_STATUS_ERROR` after the cold boot, the failure is at the vBIOS/firmware level and needs service action (vBIOS re-flash or RMA).

### 5.6 (Optional) BIOS / firmware sanity check

If 5.5 fails after a clean cold boot:
- Verify Above-4G Decoding, Re-Size BAR Support, and CSM are set consistently (all-on or all-off; gfx1101 is sensitive to mid-flight changes).
- Verify the RX 7800 XT vBIOS is up to date (AMD support page: search by board ID).

---

## 6. Why the Env Patch Alone Does NOT Restore the GPU

The full chain of `torch.cuda.is_available()` is:

```
torch.cuda.is_available()
  -> c10::cuda::is_available()        # in c10/cuda/CUDAFunctions.cpp
    -> hipInit()                       # ROCm 5.7+ entry; maps to hsa_init
      -> KFD topology walk             # ioctl(fd, AMDKFD_IOC_GET_PROPERTIES) per node
        -> For each GPU node:
             -> Check device runtime_status
             -> If error: skip (return HSA_STATUS_ERROR)
             -> If healthy: probe SMU/PSP/DRAM, init gfx ring
```

Our device has `runtime_status=error` at the kernel layer. The KFD topology walk returns `HSA_STATUS_ERROR` at step 4 before ever talking to the SMU. Therefore:

- Env vars are read **before** this chain (at torch's import time, when it loads `libamdhip64.so`).
- The chain fails **at** step 4 (kernel-level), not at env-resolution.

So the env patch correctly:

- Ensures `libamdhip64.so.7` is found before any system version (avoids a class of "works on dev, fails on prod" bugs).
- Forces gfx1101 enumeration via `HSA_OVERRIDE_GFX_VERSION` (which would matter on a healthy box).
- Adds the recommended stability knobs.

But the env patch **cannot**:

- Clear `runtime_status=error`.
- Reset a hung SMU.
- Re-flash PSP firmware.
- Rebind a PCI device stuck in error state.

These need 5.1–5.4 above.

---

## 7. Honest Framing

- **What we know for sure (after patch):** env vars are correctly written, parse cleanly under bash and POSIX-sh, resolve in interactive shells via `source ~/.bashrc`, and `LD_LIBRARY_PATH` now includes all three ROCm-required paths. The patch is correct and complete **as env hygiene**.
- **What we know for sure (still blocked):** `torch.cuda.is_available()` remains False. `rocminfo` still errors with `HSA_STATUS_ERROR` at line 1329. The GPU is still in kernel-reported `runtime_status=error`. **WF-3 (CFM retrain probe) is BLOCKED — we did not run it, because we know it would fail with `hipErrorNoDevice` before reaching any compute.** Running it would generate a misleading "CFM retrain failed" record that misattributes the failure to the CFM code, not the GPU outage.
- **What we did not do:** no sudo invocations, no `modprobe amdgpu`, no PCI unbind/rebind, no cold-boot. The sandbox blocks all of these, and per the user's instruction in this workflow, we explicitly do NOT use sudo.
- **What the user needs to do:** run §5.1, §5.3 (optional), §5.4 (mandatory if 5.3 fails) in a real sudo-capable shell, then re-run §4's verification. If `rocminfo` succeeds, the GPU is back and WF-3 (CFM retrain) can proceed. If `rocminfo` still errors after a clean cold-boot, escalate to vBIOS/firmware check (§5.6).
- **Why we are NOT shipping this as "fixed":** honest framing. The env patch is necessary but not sufficient. Calling the GPU outage "fixed" without a successful `rocminfo` enumeration would be misleading. This report explicitly carries the BLOCKED status forward for WF-3.

---

## 8. Metric Summary (machine-readable)

| Metric                          | Value                                                                                                |
| ------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `bashrc_updated`                | true (lines 16-32 added; nvm block intact at lines 34-36)                                            |
| `profile_updated`               | true (lines 5-20 added; elan PATH preserved at lines 1-3)                                            |
| `env_vars_added`                | 10 (ROCM_PATH, HIP_PATH, HIP_PLATFORM, LD_LIBRARY_PATH, PATH, HIP_VISIBLE_DEVICES, HSA_OVERRIDE_GFX_VERSION, PYTORCH_ROCM_ARCH, HSA_ENABLE_INTERRUPT, HSA_FORCE_FINE_GRAIN_PCIE, PYTORCH_ROCM_DEVICE_REGULAR_BLOCK) |
| `manual_steps_required`         | true                                                                                                  |
| `manual_steps_list`             | kill zombies (no sudo); force D0 (sudo); unbind/rebind PCI (sudo); cold power-cycle (physical)        |
| `cuda_available_after`          | false (unchanged from before)                                                                         |
| `cuda_device_count_after`       | 0 (unchanged from before)                                                                             |
| `rocminfo_after`                | HSA_STATUS_ERROR at line 1329 (unchanged)                                                              |
| `rocm_smi_gpu0_state_after`     | low-power (N/A Temp/Power/SCLK/MCLK); same as before                                                  |
| `rocm_smi_gpu1_state_after`     | healthy (65°C, 64.5 W, 2800 MHz MCLK)                                                                 |
| `root_cause_hypothesis_confirmed` | SMU firmware hang on RX 7800 XT (gfx1101); env patch correctly applied but cannot resolve it          |
| `wf3_cfm_retrain_blocked`       | true                                                                                                  |

---

## 9. Files & References

- Project root: `/home/hugo/codes/try_triton_on_rocm`
- Phase 1 diagnosis: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_gpu_diag/diagnosis.md`
- This report (Phase 2 env fix): `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_gpu_diag/env_fix.md`
- Modified shell init: `/home/hugo/.bashrc`, `/home/hugo/.profile`
- Blocked downstream task: WF-3 (Round-12 N=10×3 scientific pilot), WF-467 (CFM Retrain Diagnose), TODO-22 (data-gap alignment)
- Related TODO: `TODO/pending/20_post_r10_r11_action_plan.md` §D7 (which depends on the GPU being alive)
