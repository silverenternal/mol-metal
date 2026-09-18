# WF-GPU-Diag-Fix — GPU Outage Diagnosis

**Date:** 2026-09-14
**Project root:** /home/hugo/codes/try_triton_on_rocm
**Environment:** Arch Linux (7.2.4-zen2-1-zen), uv-managed Python 3.12, ROCm 7.2, torch 2.14.0+rocm7.2, triton-rocm 3.8.0, RX 7800 XT (gfx1101, wave64)

---

## 1. Executive Summary

`torch.cuda.is_available()` returns **False** despite `/dev/kfd`, `amdgpu` kernel module, libhsa-runtime64.so, libamdhip64.so, and torch 2.14.0+rocm7.2 all being present on the system. **The root cause is NOT environment variables** — it is a **frozen-gpu firmware / SMU-failure state on the discrete AMD RX 7800 XT (BDF 0000:03:00.0)**. The integrated Radeon 780M iGPU (BDF 0000:c8:00.0) is healthy.

HSA initialization (`rocminfo`) fails with `HSA_STATUS_ERROR: A generic error has occurred` at `rocminfo.cc:1329` because the KFD driver cannot bring the GPU out of low-power D-state — the SMU firmware is unresponsive (`SMU: No response`, `Failed to setup smc hw!`, `Failed to SetDriverDramAddr!`, `resume of IP block <smu> failed -62`). PCI runtime_status reports "error" and `/sys/bus/pci/devices/0000:03:00.0/power/control` is "auto", which lets the runtime suspend a GPU that has already failed initialization.

The stale `python3.12` processes (PIDs 193461, 193674, 194940, 195477, 196317, 196428) that AMD-SMI shows attached to GPU 0 are zombies from prior tasks and are not the cause, but they compound the problem by holding VRAM allocations across a firmware-failed device.

---

## 2. Diagnostic Data

### 2.1 /dev/kfd
- **Exists:** yes (`crw-rw-rw- 237,0 root /dev/kfd`, mode 666, owned by root, world-readable/writable).
- **Implication:** KFD topology layer exists but does NOT guarantee the GPU is initialised.

### 2.2 lsmod | grep amdgpu
- **Module loaded:** yes.
- **Users count:** `amdgpu 19169280 32` (32 dependent users).
- **Supporting modules also present:** `drm_buddy`, `amdxcp`, `i2c_algo_bit`, `drm_ttm_helper`, `ttm`, `drm_exec`, `drm_panel_backlight_quirks`, `gpu_sched`, `drm_suballoc_helper`, `video`, `drm_display_helper`, `cec`.
- **Caveat:** a module can load and a PCI device can probe while still failing to fully resume an IP block. This is exactly what we observe.

### 2.3 rocminfo (full output)
```
ROCk module is loaded
hsa api call failure at: /usr/src/debug/rocminfo/rocm-systems-rocm-7.2.4/projects/rocminfo/rocminfo.cc:1329
Call returned HSA_STATUS_ERROR: A generic error has occurred.
```
- **HSA_STATUS_ERROR line:** line 1329 of `rocminfo.cc` (per the debug path `/usr/src/debug/rocminfo/rocm-systems-rocm-7.2.4/...`).
- That call site is the standard HSA topology walk; failure here means KFD failed to enumerate the GPUs.
- Setting `LD_LIBRARY_PATH=/opt/libtorch/lib:/opt/rocm/lib:/opt/rocm/lib64` did NOT change the output — confirming this is a kernel/driver-level failure, not a userspace library resolution issue.

### 2.4 env | grep -E "HIP|ROCM|HSA|LD_LIBRARY"
```
ROCM_PATH=/opt/rocm
LD_LIBRARY_PATH=/opt/libtorch/lib:
```
- **Missing env vars (the "fix-env-vars" hypothesis is WRONG):**
  - `HIP_PATH` (unset)
  - `HIP_PLATFORM` (unset, defaults to `amd`)
  - `HSA_ENABLE_INTERRUPT=1` (unset)
  - `HSA_FORCE_FINE_GRAIN_PCIE=1` (unset)
  - `HSA_XNACK=1` (unset, depends on kernel config)
  - `GPU_TARGETS` / `AMDGPU_TARGETS` (unset, build-only)
  - `ROCM_HOME` (unset; `ROCM_PATH` is set)
  - `PYTORCH_ROCM_DEVICE_REGULAR_BLOCK=1` (unset, optional perf knob)
  - `LD_LIBRARY_PATH` is **broken**: missing `/opt/rocm/lib` and `/opt/rocm/lib64`. The trailing colon and absence of ROCm paths means a torch build linked against ROCm 7.2 cannot resolve `libhsa-runtime64.so.1` and `libamdhip64.so.7` at runtime if they are not found via ldconfig first.
- **Why ldconfig currently "works":** `/etc/ld.so.conf.d/rocm.conf` contains `/opt/rocm/lib`, and the libs are present (see §2.6 / §2.7), so they ARE in the cache. This is why adding them to `LD_LIBRARY_PATH` does not fix anything: the loader was already finding them.
- The env vars above are *recommended best-practice* but **NOT** the reason `torch.cuda.is_available()` is False.

### 2.5 torch CUDA probe
```
torch.__version__: 2.14.0+rocm7.2
torch.version.hip: 7.2.53211
cuda available: False
cuda device count: 0
```
- torch+HIP versions match the ROCm install (7.2.53211 == `/opt/rocm/lib/libamdhip64.so.7.2.53211`).
- `torch.cuda.is_available()` calls `hipInit` (or the ROCm 5.7+ equivalent `hipInit` indirectly via `c10::hip`). On this system that call returns `hipErrorNoDevice`.

### 2.6 ROCm libs on disk
```
/opt/rocm/lib/libamdhip64.so        -> libamdhip64.so.7
/opt/rocm/lib/libamdhip64.so.7      -> libamdhip64.so.7.2.53211
/opt/rocm/lib/libamdhip64.so.7.2.53211
/opt/rocm/lib/libhsa-runtime64.so   -> libhsa-runtime64.so.1
/opt/rocm/lib/libhsa-runtime64.so.1 -> libhsa-runtime64.so.1.18.0
/opt/rocm/lib/libhsa-runtime64.so.1.18.0
/opt/rocm/lib/libhsa-amd-aqlprofile64.so* (AQL profile lib)
/opt/rocm/lib/libhsakmt.a           (KFD static thunk)
```
- **No `/opt/rocm/lib64`** directory exists — only `/opt/rocm/lib` and `/opt/rocm/libexec`. This is normal for ROCm 6+ (unified lib dir).
- **libhsa-runtime64.so.1.18.0** is installed (HSA runtime 1.18). Matches ROCm 7.2.
- **libamdhip64.so.7.2.53211** matches torch's `torch.version.hip = 7.2.53211`. **The ABI matches.** This rules out a torch/ROCm ABI mismatch.

### 2.7 ldconfig cache
```
libhsa-runtime64.so.1 (libc6,x86-64) => /opt/rocm/lib/libhsa-runtime64.so.1
libhsa-runtime64.so  (libc6,x86-64) => /opt/rocm/lib/libhsa-runtime64.so
libamdhip64.so.7     (libc6,x86-64) => /opt/rocm/lib/libamdhip64.so.7
libamdhip64.so       (libc6,x86-64) => /opt/rocm/lib/libamdhip64.so
```
- All four entries are present. `/etc/ld.so.conf.d/rocm.conf` correctly points to `/opt/rocm/lib`. **Libraries are resolvable.**

### 2.8 echo $LD_LIBRARY_PATH
```
/opt/libtorch/lib:
```
- One entry (`/opt/libtorch/lib`) and a trailing colon (interpreted as current directory). ROCm paths missing here, **but** ldconfig covers the gap. Adding the missing paths is a good hygiene fix but does not address the real failure.

### 2.9 /sys/class/drm/card*/device/power/control
```
auto (×16) on (×1)
```
- 17 DRM connectors/paths listed. All in `auto` (runtime PM enabled) except one in `on` (forced on). The on/auto split reflects AMD's runtime D-state policy.
- GPU 0 (`/sys/class/drm/card1`, BDF 0000:03:00.0) is in `auto`. **This is the wrong policy for a gfx1101 dGPU that is failing to resume its SMU IP block.** When `auto`, the kernel may runtime-suspend the device after probe failures, leaving it permanently stuck in D3.

### 2.10 /dev/dri
```
crw-rw----@ card1       (226,1, root)
crw-rw----@ card2       (226,2, root)
crw-rw-rw-  renderD128  (226,128, root, mode 666)
crw-rw-rw-  renderD129  (226,129, root, mode 666)
```
- `renderD128` corresponds to card1 (RX 7800 XT, BDF 0000:03:00.0).
- `renderD129` corresponds to card2 (Radeon 780M, BDF 0000:c8:00.0).
- Both render nodes are mode 666 — accessible to non-root.
- User `hugo` (uid 1000) is in `video` (983) and `render` (987) groups — has access.
- **The render node is present, but the GPU underneath it is stuck.** The node can be opened but any GPU command buffer submission hangs / fails because the gfx ring has not been brought up.

### 2.11 dmesg | grep -E "amdgpu|HSA|kfd" (kernel log, key lines)
```
Sep 14 11:50:18 amdgpu 0000:03:00.0: psp gfx command LOAD_TA(0x1) failed and response status is (0x2C)
Sep 14 11:50:18 amdgpu 0000:03:00.0: psp gfx command LOAD_TA(0x1) failed and response status is (0x2C)
Sep 14 11:50:18 amdgpu 0000:03:00.0: RAP: optional rap ta ucode is not available
Sep 14 11:50:18 amdgpu 0000:03:00.0: SECUREDISPLAY: optional securedisplay ta ucode is not available
Sep 14 11:50:18 amdgpu 0000:03:00.0: SMU is resuming...
Sep 14 11:50:18 amdgpu 0000:03:00.0: SMU is resumed successfully!
Sep 14 11:50:18 amdgpu 0000:03:00.0: [drm] DMUB hardware initialized: version=0x07003300
Sep 14 11:50:18 amdgpu 0000:03:00.0: [drm] Cannot find any crtc or sizes
Sep 14 11:50:18 amdgpu 0000:03:00.0: ring gfx_0.0.0 uses VM inv eng 0 on hub 0
Sep 14 11:50:18 amdgpu 0000:03:00.0: ring comp_1.0.0 uses VM inv eng 1 on hub 0
... (other rings configured normally)
Sep 14 11:51:56 amdgpu 0000:03:00.0: SMU: No response msg_reg: 3b resp_reg: 0
Sep 14 11:51:56 amdgpu 0000:03:00.0: [SetDfCstate] failed!
Sep 14 11:51:56 amdgpu 0000:03:00.0: Failed to disallow df cstate
Sep 14 11:52:01 amdgpu 0000:03:00.0: SMU: No response msg_reg: 3b resp_reg: 0
Sep 14 11:52:01 amdgpu 0000:03:00.0: Failed to retrieve enabled ppfeatures!
Sep 14 11:52:06 amdgpu 0000:03:00.0: SMU: No response msg_reg: 3b resp_reg: 0
Sep 14 11:52:06 amdgpu 0000:03:00.0: SMC failed to set mp1 state 2, -62
Sep 14 11:52:11 amdgpu 0000:03:00.0: SMU: No response msg_reg: e resp_reg: 0
Sep 14 11:52:11 amdgpu 0000:03:00.0: Failed to SetDriverDramAddr!
Sep 14 11:52:11 amdgpu 0000:03:00.0: Failed to setup smc hw!
Sep 14 11:52:11 amdgpu 0000:03:00.0: resume of IP block <smu> failed -62
Sep 14 11:52:11 amdgpu 0000:03:00.0: amdgpu_device_ip_resume_phase2 failed during unwind: -62
Sep 14 11:52:23 amdgpu 0000:03:00.0: [drm] Cannot find any crtc or sizes
Sep 14 11:52:23 amdgpu 0000:03:00.0: can't suspend (amdgpu_pmops_runtime_suspend [amdgpu] returned -62)
Sep 14 11:52:23 amdgpu 0000:03:00.0: VM memory stats for proc (0) task (0) is non-zero when fini
```

### 2.12 rocm-smi & amd-smi (live device state)
```
rocm-smi WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status
GPU[0] BDF 0000:03:00.0 (0x747e, RX 7800 XT): Temp=N/A, Power=N/A, SCLK=N/A, MCLK=N/A, PwrCap=N/A, GPU%=0%, VRAM%=1%
GPU[1] BDF 0000:c8:00.0 (0x1900, Radeon 780M): Temp=63.0°C, Power=61.279W, MCLK=2800MHz, Perf=auto, VRAM%=93%, GPU%=4%

amd-smi:
GPU 0 (0000:03:00.0): N/A, Mem-Uti N/A, Temp N/A, UEC 0, Power N/A, Mem 174/16368 MB
GPU 1 (0000:c8:00.0): Radeon 780M Graphics, N/A, 955/1024 MB
```

- **GPU 0 (RX 7800 XT) is in low-power state.** All telemetry N/A, no clock, no power. The 174/16368 MB usage is residual VRAM held by 6 zombie `python3.12` processes.
- **GPU 1 (iGPU) is fully alive.** 63°C, 61W, 2800 MHz MCLK, 93% VRAM used by the display stack.

### 2.13 PCI power state for the failed GPU
```
/sys/bus/pci/devices/0000:03:00.0/power/control    = auto
/sys/bus/pci/devices/0000:03:00.0/power/runtime_status = error
```
- **PCIe runtime_status = `error`** — the kernel has marked the device as failed. This is downstream of the SMU resume failure.
- `control = auto` allows runtime D-state transitions, which is **the worst setting** for a device that already failed resume.

---

## 3. Root-Cause Hypothesis

The GPU outage is a **hardware/firmware hang on the discrete RX 7800 XT's SMU** — not a software misconfiguration. The chain of events:

1. **PCIe device probed successfully** — `amdgpu` module loaded, BDF 0000:03:00.0 enumerated, ring/VM engines configured, DRM/KFD topology nodes created.
2. **PSP trusted-application load failed** at probe — `psp gfx command LOAD_TA(0x1) failed and response status is (0x2C)`. Response code `0x2C` is `PSP_STATUS_INVALID_COMMAND_STATE` (or similar — AMD internal), meaning a previous TA was not in the expected state. This is the first hard signal.
3. **SMU appeared to resume** (`SMU is resumed successfully!`) but the SMU is **non-responsive to subsequent commands**. Sequence:
   - `SMU: No response msg_reg: 3b resp_reg: 0` — firmware does not acknowledge MMIO message.
   - `Failed to disallow df cstate`, `Failed to retrieve enabled ppfeatures!` — first hints of hang.
   - `SMC failed to set mp1 state 2, -62` (`-62 = -ETIME`) — SMU command timeout.
   - `Failed to SetDriverDramAddr!`, `Failed to setup smc hw!` — DRAM training cannot proceed without SMU.
   - `resume of IP block <smu> failed -62` — the entire device resume aborted during unwind.
4. **KFD/HSA initialization cannot enumerate GPU 0** because the device is left in a half-initialised state where the gfx ring exists but SMU/DRAM are not configured. `rocminfo` reports `HSA_STATUS_ERROR` at the topology walk.
5. **PCI runtime_status = error**, with `control = auto`, the kernel has left the device in a permanent low-power state — telemetry is N/A, all commands fail.

### Why it is NOT an env-var problem
- LD_LIBRARY_PATH only has `/opt/libtorch/lib:` — missing `/opt/rocm/lib`, but **ldconfig covers this** (rocm.conf is installed and all libs are cached).
- All ROCm shared libraries are present at the expected version and ABI-matches torch.
- Setting the "correct" env vars and re-running rocminfo did NOT change its output.
- `/dev/kfd` is mode 666 and the user is in video+render groups — no permission issue.

### Compounding factors
- **Stale `python3.12` processes (PIDs 193461, 193674, 194940, 195477, 196317, 196428)** hold GTT/VRAM on GPU 0. They cannot be killed cleanly while the GPU is in `error` runtime_status, and they cannot release VRAM while the SMU is hung.
- `LD_LIBRARY_PATH` is missing ROCm paths and `/opt/rocm/lib64` — fix is hygiene, not the outage cause.

### Hypothesised upstream cause
- The most common trigger for this exact PSP/SMU hang pattern on gfx1101 is **one of**:
  - A previous GPU-intensive workload was killed (kill -9 or OOM) and left the SMU in a bad state across a suspend/resume cycle.
  - A BIOS/firmware "Above 4G decoding" / "Resize BAR" / "Re-Size BAR Support" change without a power-cycle (gfx1101 SMU is sensitive to this).
  - An `amdgpu` driver version mismatch with the firmware version embedded in the kernel module (the dmesg driver line is `7.2.4-zen2-1-zen` — Zen-tuned custom build, may carry SMU firmware that doesn't match the vBIOS in the RX 7800 XT).
- The error code `0x2C` is consistent with a **firmware state that has not been cleared since the last boot** — i.e. the GPU entered S5 with SMU dirty and a fresh `modprobe amdgpu` cannot recover.

---

## 4. Recommended Fix Path (in order of cheapest → most invasive)

### Fix A — environment variable hygiene (cheap, do regardless)
Add to `~/.bashrc` or project shell wrapper:
```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export LD_LIBRARY_PATH=/opt/libtorch/lib:/opt/rocm/lib:/opt/rocm/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=/opt/rocm/bin:/opt/rocm/libexec/rocm_smi:/opt/rocm/libexec/amdsmi_cli:$PATH
export HSA_ENABLE_INTERRUPT=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_ROCM_DEVICE_REGULAR_BLOCK=1
```
**Expected outcome:** no change to `torch.cuda.is_available()` until the GPU firmware is fixed. This is hygiene only.

### Fix B — runtime power policy (no sudo needed for the sysfs writes)
```bash
# Force the GPU to stay in D0
echo on | sudo tee /sys/bus/pci/devices/0000:03:00.0/power/control
# Try to wake it up
echo 1 | sudo tee /sys/bus/pci/devices/0000:03:00.0/power/wakeup
```
**Expected outcome:** usually no help once runtime_status=`error`; the kernel has already given up on the device. Worth trying as a 30-second experiment.

### Fix C — unbind + rebind PCI device (no full reboot)
```bash
sudo sh -c 'echo 0000:03:00.0 > /sys/bus/pci/drivers/amdgpu/bind'
# If already bound but hung:
sudo sh -c 'echo 0000:03:00.0 > /sys/bus/pci/drivers/amdgpu/unbind; sleep 2; echo 0000:03:00.0 > /sys/bus/pci/drivers/amdgpu/bind'
```
**Expected outcome:** may re-trigger probe + resume cleanly. **Caveat:** if the SMU is truly hung at the firmware level, the rebind will just reproduce the same `resume of IP block <smu> failed -62` and the device will return to runtime_status=error.

### Fix D — kill zombie python3.12 processes + retry
```bash
kill -9 193461 193674 194940 195477 196317 196428 2>/dev/null
```
**Expected outcome:** frees GTT/VRAM accounting, may allow a clean rebind. Low cost, recommended as a precursor to Fix C.

### Fix E — full power cycle (the only known-good fix for a stuck SMU)
1. `shutdown -h now`
2. Wait 30 seconds (allow VRM capacitors to discharge — gfx1101 SMU retains state across warm reboots if the ATX 5V rail stays up).
3. Cold boot.
4. Verify with `/opt/rocm/bin/rocminfo` — should now report the RX 7800 XT as an HSA agent.
**Expected outcome:** SMU resets cleanly, gfx ring initializes, KFD enumerates the GPU, `torch.cuda.is_available()` returns True.

### Fix F — BIOS/firmware check (only if Fix E fails)
- Verify Above-4G Decoding, Resize BAR, and CSM are set consistently (either all on or all off — gfx1101 is sensitive to mid-flight changes).
- Verify the RX 7800 XT vBIOS is up to date (vBIOS-issued SMU firmware can override the kernel module's payload).

### Recommendation
**Fix D → Fix E.** Kill the zombie processes (Fix D), try rebind (Fix C in passing), and if `rocminfo` still errors, escalate to a cold power cycle (Fix E). The outage is firmware-level, not env-level.

---

## 5. Verification Checklist (after Fix E)

```bash
# 1. rocminfo should now list the RX 7800 XT as agent
/opt/rocm/bin/rocminfo | grep -A 20 "Agent 1"
# Expect: "Name: gfx1101", "Marketing Name: AMD Radeon RX 7800 XT", "Device Type: GPU"

# 2. rocm-smi should report full telemetry
rocm-smi
# Expect: GPU[0] with non-N/A Temp, Power, SCLK, MCLK

# 3. amd-smi should show GPU 0 with Mem/GFX usage
/opt/rocm/bin/amd-smi

# 4. torch CUDA probe
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
# Expect: True 1 AMD Radeon RX 7800 XT

# 5. minimal CFM retrain probe (placeholder for full WF-3)
/opt/rocm/bin/rocminfo | grep -E "Marketing Name|Device Type"
```

If `rocminfo` after Fix E still shows `HSA_STATUS_ERROR`, then the failure is at the vBIOS/firmware level (Fix F territory) and the GPU needs a service action.

---

## 6. Honest Framing

- **What we know for sure:** `/dev/kfd`, `amdgpu`, `libhsa-runtime64.so`, `libamdhip64.so`, `torch==2.14.0+rocm7.2`, user groups, library cache are all correctly configured. The `LD_LIBRARY_PATH` is missing ROCm paths but ldconfig covers it; this is hygiene, not the cause.
- **What we know for sure:** GPU 0 (RX 7800 XT) is in a kernel-reported low-power / runtime_status=error state, the SMU IP block failed to resume with errno -62, and the PSP TA load returned 0x2C. KFD cannot enumerate the GPU and HSA fails at topology walk.
- **What we don't know without hardware intervention:** whether the SMU will recover on a cold power cycle. The pattern is consistent with a cold-boot-recoverable hang, but not guaranteed.
- **What we did NOT do:** we did not run any retrain probe, because the GPU is not functional. Any "minimal CFM retrain probe" would have to run *after* a successful cold boot and `rocminfo` enumeration. **WF-3 (CFM retrain) remains BLOCKED** until then.
- **What we did do:** we ruled out the env-var hypothesis the user proposed. The user's instruction "fix env vars" is correct as hygiene but **does not address the root cause** of `torch.cuda.is_available()=False`.

---

## 7. Metric Summary (machine-readable)

| Metric                          | Value                                                                                   |
|---------------------------------|-----------------------------------------------------------------------------------------|
| `kfd_exists`                    | true                                                                                    |
| `amdgpu_loaded`                 | true (32 dependent users)                                                               |
| `hsa_status_error_line`         | 1329                                                                                    |
| `ld_library_path`               | `/opt/libtorch/lib:` (missing ROCm paths; trailing colon)                               |
| `hip_version`                   | 7.2.53211 (matches libamdhip64.so.7.2.53211)                                            |
| `cuda_available_before`         | false                                                                                   |
| `missing_env_vars_list`         | HIP_PATH, HSA_ENABLE_INTERRUPT, HSA_FORCE_FINE_GRAIN_PCIE, HSA_XNACK, ROCM_HOME, PYTORCH_ROCM_DEVICE_REGULAR_BLOCK, /opt/rocm/lib and /opt/rocm/lib64 in LD_LIBRARY_PATH |
| `pci_runtime_status_gpu0`       | `error`                                                                                 |
| `pci_power_control_gpu0`        | `auto`                                                                                  |
| `amdgpu_resume_failure`         | `resume of IP block <smu> failed -62`                                                   |
| `psp_ta_load_failure`           | `psp gfx command LOAD_TA(0x1) failed and response status is (0x2C)`                     |
| `gpu0_temp`                     | N/A (low-power state)                                                                   |
| `gpu0_power`                    | N/A (low-power state)                                                                   |
| `gpu1_temp`                     | 63.0°C (healthy)                                                                        |
| `gpu1_power`                    | 61.279W (healthy)                                                                       |
| `zombie_python3.12_pids`        | 193461, 193674, 194940, 195477, 196317, 196428                                          |
| `root_cause_hypothesis`         | SMU firmware hang on RX 7800 XT (gfx1101) following failed PSP TA load + bad suspend; not env-var related |

---

## 8. Files & References

- Project root: `/home/hugo/codes/try_triton_on_rocm`
- This report: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_gpu_diag/diagnosis.md`
- Related TODO: TODO-22 (`22_data_gap_alignment_plan.md`) — also blocked by GPU outage.
- Prior blocked task: WF-467 (`WF-CFM-Retrain-Diagnose: BLOCKED by GPU outage (HSA init fails)`).
