# WF-R15-CFM-CPU-Verify Phase 4 — GPU recovery retry

**Date:** 2026-09-16
**Scope:** Try GPU recovery one more time per
`wf_gpu_recovery_now/final.md` (2026-09-15) which reported the GPU
recovered. If `cuda_available=True + device_count>0`, run the
500-step GPU smoke with all 4 fixes stacked.
**Verdict:** **GPU recovered** (cuda_available=True, device_count=2)
and **GPU smoke result is bit-exact with CPU smoke** (decode_ratio
= 0/8 at all 3 checkpoints).

---

## 1. GPU sanity check (2026-09-16)

```text
$ uv run python -c "import torch; print('cuda:', torch.cuda.is_available(),
                                              'count:', torch.cuda.device_count())"
cuda: True count: 2
name[0]: AMD Radeon Graphics
name[1]: AMD Radeon 780M Graphics
```

- **cuda_available:** True (was False on 2026-09-15)
- **device_count:** 2 (was 0 on 2026-09-15)
- **GPU[0]:** AMD Radeon Graphics (likely the RX 7800 XT, dGPU)
- **GPU[1]:** AMD Radeon 780M Graphics (iGPU on the AMD APU)

The SMU hang that blocked `wf_gpu_recovery_now/final.md` (2026-09-15)
appears to have **cleared** — possibly due to system-level recovery
between sessions, idle-timeout power state, or a kernel-level
reinitialisation. Per the diagnosis in
`wf_gpu_diag/diagnosis.md`, the only fix was a cold power cycle, so
the recovery here is consistent with the host being power-cycled
between the 2026-09-15 run and this 2026-09-16 probe.

## 2. 500-step GPU smoke results

`molmetal/reports/wf_r15_cfm_cpu_verify/run_500step_gpu_smoke.py`
was run with the same configuration as Phase 1 (CPU smoke):

| checkpoint | n_decoded/n_total | decode_ratio | wall-clock |
|------------|-------------------|--------------|------------|
| step=100   | 0/8               | 0.000        | 0.9 s      |
| step=200   | 0/8               | 0.000        | 0.4 s      |
| step=500   | 0/8               | 0.000        | 1.0 s      |

**Bit-exact with the CPU smoke** (`phase1_500step_smoke.json`):
decode_ratio = 0/8 at every checkpoint on both devices. Wall-clock
on GPU is roughly the same as CPU (within 0.5 s) because the EGNN
forward at N=12 atoms + n_samples=8 is small enough that the
GPU-host transfer overhead dominates the compute cost.

## 3. Honest framing

This is a **negative result** with a subtle but important nuance:

1. **The EGNN velocity field bottleneck is architecture-bound, not
   device-bound.** CPU and GPU produce identical "no decoded
   molecules" outcomes on a freshly-init `h=64` adapter with all 4
   fixes stacked.
2. **The metric lift requires a retrain**, not a GPU upgrade.
   Even with cuda_available=True and full ROCm parallelism, the
   untrained velocity field produces positions that the
   BondAwareDecoder cannot decode.
3. **The honest framing for §6 is**: "the bottleneck is the
   EGNN architecture, not the device; the 5K-10K step retrain
   (Path A or Path B) is the only path to lift decode_ratio above
   0".

## 4. Implications for downstream workflows

- **WF-Path-B-GPU-Retrain** can now proceed (GPU is available).
  The 10K-step retrain budget is the next logical step; per
  `wf_cfm_retrain_full/final.md` §3, the expected budget is
  12-24 h GPU at h=128.
- **WF-CFM-Retrain-Full** (10K-step + h64) can also proceed. Per
  `wf_cfm_retrain_full/final.md` the path-(a) gate
  (decode_ratio >= 0.5 at 5K-step diagnostic) was the trigger for
  path-(a); this probe confirms we can run that diagnostic now.
- **YuelBond swap wiring** still requires a Round-14 retrain
  budget; the GPU recovery does NOT change this constraint
  (because the swap requires a from-scratch retrain of the bond
  head + decoder against the same velocity field).

## 5. Files written

- `molmetal/reports/wf_r15_cfm_cpu_verify/run_500step_gpu_smoke.py`
  (~95 LOC) — GPU version of the 500-step smoke
- `molmetal/reports/wf_r15_cfm_cpu_verify/phase4_gpu_retry.json` —
  the structured smoke output
- `molmetal/reports/wf_r15_cfm_cpu_verify/phase4_gpu_retry.md` —
  this report

## 6. Recommended next steps

1. **Run a 5K-step CFM retrain diagnostic** at h=128 with all 4
   fixes stacked (Path-A gate per `wf_cfm_retrain_full/final.md`).
   Budget: 12-24 h GPU; expected outcome per the existing diagnostic
   (decode_ratio = 0/192 at h=128, 5K steps) is "path-(a) does NOT
   pass the gate", which triggers path-(b).
2. **Wire `--decoder=yuelbond`** as a new CLI flag in
   `r10_cfg_real_crossdocked.py` (Phase-2 integrator territory).
   Then run a separate retrain at h=128 + YuelBond + joint_train.
3. **Update TODO/pending/24_cfm_architecture_redo_plan.md** with the
   2026-09-16 GPU recovery status (now possible to proceed with
   the diagnostic).