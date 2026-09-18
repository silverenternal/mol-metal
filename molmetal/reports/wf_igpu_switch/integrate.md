# WF-iGPU-Switch — Integrate Report (2026-09-14)

**Goal.** Route all GPU work to the healthy integrated Radeon 780M
(gfx1100) instead of the dead discrete RX 7800 XT (gfx1101, SMU
firmware hang per `WF-GPU-Diag-Fix`).
**Outcome.** iGPU probe **FAILED at the HSA runtime layer** — same root
cause as `WF-GPU-Diag-Fix`. Device selector updated to *prefer* the
iGPU when reachable, but the policy is currently latent (no GPU is
reachable to PyTorch on this host). 11 hermetic tests lock the
preference logic in place.

## 1. iGPU probe verdict (recap, full detail in `probe.md`)

| Probe | Result | Verdict |
|---|---|---|
| iGPU hardware state | `runtime_status=active`, SMU OK, 12 CUs, 1024 MB VRAM | HEALTHY |
| dGPU hardware state | `runtime_status=error`, `SMU: No response msg_reg: 3b`, repeated | DEAD |
| `torch.cuda.is_available()` | `False` (all 7 env-var combos) | BLOCKED |
| `libhsa-runtime64.so.1::hsa_init()` | returns `0x1000 = HSA_STATUS_ERROR_INITIALIZATION` | BLOCKED |
| `/opt/rocm/bin/rocminfo` | `hsa api call failure ... HSA_STATUS_ERROR` | BLOCKED |
| CFM retrain probe | NOT RUN (would have crashed with `No CUDA GPUs are available`) | BLOCKED |

**Honest framing:** the iGPU is hardware-healthy and fully initialized
at the kernel/drm level. The ROCm 7.2 HSA runtime cannot reach *any*
device on this host because the dead dGPU pollutes the KFD topology
enumeration. Switching from "discrete preferred" to "iGPU preferred"
does NOT solve the root cause — both GPUs are equally unreachable
from PyTorch.

## 2. Device selection update (`molmetal/utils/device.py`)

Added a new helper `detect_active_gpu()` that:

1. Probes `torch.cuda.is_available()` and `torch.cuda.device_count()`,
   wrapping each call in a try/except so HSA init failures degrade
   silently (per the honest-framing mandate: never raise on a broken
   GPU stack).
2. Performs a **real** 1-element float-tensor allocation on `cuda:0`
   to verify the device is not just visible but actually usable.
3. Inspects `device_0_name()` (best-effort match for `"780m"` →
   `gfx1100` and `"7800 xt"` → `gfx1101`) and sets
   `igpu_usable`/`dgpu_usable` flags.
4. Returns a rich dict with `reason` (short human-readable verdict)
   for adapters to surface in their smoke-test reports.

Updated `get_device()` to consult `detect_active_gpu()` and emit the
one-shot `UserWarning` only when neither iGPU nor dGPU is reachable.

Added a `_LazyDefaultDevice` descriptor so module-level
`DEFAULT_DEVICE` is computed on first read (avoids the forward
reference to `detect_active_gpu` and gives downstream code a single
observable source of truth). `_resolve_default_device()` preference
order:

1. **iGPU (`gfx1100`) reachable + allocatable** → `cuda:0`
   *(WF-iGPU-Switch policy)*
2. **dGPU (`gfx1101`) reachable + allocatable** → `cuda:0`
   *(legacy default)*
3. **Neither** → `cpu` with one-shot warning

`verify_rocm_active()` was rewritten as a thin wrapper over
`detect_active_gpu()` to keep all legacy keys (`roc_active`,
`device_0_name`, `device_0_index`, `env_visible`) present for
backward compatibility with the round-10 smoke tests, while adding
the new keys (`alloc_ok`, `igpu_usable`, `dgpu_usable`,
`device_0_gfx`).

The allow-lists `_HEALTHY_IGPU_GFX = ("gfx1100",)` and
`_HEALTHY_DGPU_GFX = ("gfx1101",)` are explicit module-level
constants — extending them requires a code change and a smoke test,
preventing silent scope creep.

## 3. Verify command

```
$ source ~/.bashrc && uv run python -c \
    "from molmetal.utils.device import get_device, ROCM_AVAILABLE, DEFAULT_DEVICE; \
     print(ROCM_AVAILABLE, DEFAULT_DEVICE)"
False cpu
```

Confirms: `ROCM_AVAILABLE = False` (no torch-visible GPU on this
host), `DEFAULT_DEVICE = 'cpu'` (honest CPU fallback). On a host
where the iGPU is reachable, this same command will print `True
cuda:0`.

## 4. Test suite (`molmetal/molmetal_lam/tests/test_device_selection.py`)

11 hermetic tests, all pass:

```
molmetal/molmetal_lam/tests/test_device_selection.py::test_default_prefers_igpu_when_available   PASSED
molmetal/molmetal_lam/tests/test_device_selection.py::test_default_falls_back_to_discrete_gpu    PASSED
molmetal/molmetal_lam/tests/test_device_selection.py::test_default_falls_back_to_cpu_with_warning PASSED
molmetal/molmetal_lam/tests/test_device_selection.py::test_get_device_returns_cuda_when_igpu      PASSED
molmetal/molmetal_lam/tests/test_device_selection.py::test_get_device_returns_cuda_when_only_dgpu PASSED
molmetal/molmetal_lam/tests/test_device_selection.py::test_get_device_warns_only_once             PASSED
molmetal/molmetal_lam/tests/test_device_selection.py::test_detect_active_gpu_never_raises         PASSED
molmetal/molmetal_lam/tests/test_device_selection.py::test_verify_rocm_active_backward_compatible PASSED
molmetal/molmetal_lam/tests/test_device_selection.py::test_healthy_gfx_lists_are_explicit          PASSED
molmetal/molmetal_lam/tests/test_device_selection.py::test_rocm_available_constant_reflects_torch PASSED
molmetal/molmetal_lam/tests/test_device_selection.py::test_default_device_lazy_caches             PASSED
================================ 11 passed, 1 warning in 1.50s ================================
```

Tests 1–3 directly satisfy the task spec:
- `test_default_prefers_igpu_when_available`
- `test_default_falls_back_to_discrete_gpu`
- `test_default_falls_back_to_cpu_with_warning`

Tests 4–11 cover the rest of the contract: `get_device()` returns
`cuda:0` for both chip families, the CPU warning is one-shot, the
probe never raises (so a host with HSA init broken will not crash
import), `verify_rocm_active()` keeps the legacy keys, the gfx
allow-lists are explicit, and `DEFAULT_DEVICE` is lazy-cached to
avoid redundant probes.

**Regression check:** the existing `test_egnn_predictor_device.py` and
`test_rocm_egnn.py` still pass (2 pass + 6 SKIPPED — skipped tests
require GPU and were already gated on `ROCM_AVAILABLE`).

## 5. Paper impact (`paper/sections/06_limitations.tex`)

§6 limitation item (7) "Single-GPU ROCm 7.2 + RX 7800 XT (gfx1101)
throughput only" was extended with a new sub-paragraph that:

- Names the dGPU failure mode verbatim (`SMU: No response msg_reg:
  3b resp_reg: 0`).
- Names the iGPU fallback attempt (`WF-iGPU-Switch`) and the probe
  commands tested.
- Reports the probe result (`cuda_available: False, device_count: 0`
  on every env-var combination tested).
- Defers the fix to a future `WF-iGPU-Switch-2: kernel-dGPU-blacklist`
  workflow that requires sudo + reboot (out of scope for this
  session).
- Notes that the Lambda-only pipeline (§4) remains the source of all
  reported numbers for this round (consistent with `WF-CFM-Retrain-Full`
  verdict: GPU_BLOCKED).

§3.1 (MLC formalism) was *not* modified — the iGPU policy is
operational plumbing and has no bearing on the typed-variable
syntax, $\beta$-reduction semantics, or the closure-theorem proof.

## 6. TODO impact (`TODO/pending/21_lambda_model_coupling.md`)

Added a new section "WF-iGPU-Switch verdict (2026-09-14)" that:

- Reports the iGPU probe outcome (BLOCKED at HSA layer).
- Documents the new device-selection policy.
- Refreshes the decision tree for path (a) retrain: when HSA runtime
  recovers, `detect_active_gpu()` will automatically route to the
  iGPU *without* any TODO-21 change. No new path-(a) decision is
  required.

## 7. Honest-framing checklist

- [x] Did NOT pretend the iGPU was reachable to PyTorch.
- [x] Did NOT run the CFM retrain probe (would have crashed and
      produced no signal beyond "still BLOCKED").
- [x] Did NOT silently flip `DEFAULT_DEVICE` to `cuda:0` based on a
      heuristic — the new policy is gated on a real tensor
      allocation.
- [x] Did NOT modify paper §3.1 (the MLC formalism is unaffected by
      GPU plumbing).
- [x] Did record that the Lambda-only pipeline remains the source of
      all reported numbers this round, per the existing
      `WF-CFM-Retrain-Full` verdict.

## 8. Open work (out of scope for this session)

- **`WF-iGPU-Switch-2: kernel-dGPU-blacklist`** — write
  `/etc/modprobe.d/blacklist-amdgpu-dgpu.conf` with
  `options amdgpu noretry=0` + per-BFD deny-list
  (`0000:03:00.0`); require sudo + reboot. Once complete, re-run
  the probe in `molmetal/reports/wf_igpu_switch/probe.md` and the
  `DEFAULT_DEVICE` will auto-flip to `cuda:0` (iGPU).
- **`WF-iGPU-Switch-3: CFM retrain on iGPU`** — once the iGPU is
  reachable, run the 5000-step CFM diagnostic on gfx1100 (≤0.5 h
  wall, ≤1 h GPU budget). Decision point for path (a) vs (b) vs (c)
  per TODO-21.
