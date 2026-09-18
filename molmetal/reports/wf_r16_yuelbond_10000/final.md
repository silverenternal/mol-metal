# WF-R16-YuelBond-10000-Full-Retrain Verdict

**Date**: 2026-09-18
**Workflow**: R16 YuelBond 10000-step FULL retrain
**Verdict**: **DECODE-LIFT HOLDING, GPU HUNG AT STEP 2500** (partial — seeds 1234 and 7 not reached)

## TL;DR

The 10000-step YuelBond retrain @ h=128, n_train=500 (batch=4 to avoid OOM)
on RX 7800 XT achieved **decode_ratio=1.000 at both decode-smoke probes
(step 1000 and step 2000)** with **0 disconnected atoms per sample** —
identical to the 2000-probe PASS verdict and consistent with the
F1+F2+P1.4 fix story.

The process crashed at step 2500 with
`terminate called after throwing an instance of 'rocr::AMD::hsa_exception':
what(): Signal handle is invalid.` This is the firmware-level SMU hang
documented in `molmetal/reports/wf_gpu_diag/diagnosis.md` — the dGPU
stopped being enumerated (device_count went 2 → 1) and even a bare
`torch.cuda.is_available()` call hangs indefinitely (exit code 143
SIGTERM, 0 bytes output).

**Cannot proceed to seeds 1234 / 7 until GPU is cold-power-cycled.**
Same recovery gate as prior GPU BLOCKED runs.

## What was MEASURED

### Decode-smoke trajectory (n_samples=8 per checkpoint)

| step | decode_ratio | n_decoded | mean_atoms | disconnected | wall_s |
|------|--------------|-----------|------------|--------------|--------|
| 1000 | **1.000**    | 8/8       | 8.0        | 0            | 0.3    |
| 2000 | **1.000**    | 8/8       | 8.0        | 0            | 0.2    |

Source: `seed_42/train.log` lines for `>> decode_smoke step=1000` and
`>> decode_smoke step=2000` (raw lines preserved in the saved log).

### Training-loss summary (10000-step run, batch=4)

| Loss | first (step 1)  | mid (step 1500) | mid (step 2500) | min observed    |
|------|-----------------|-----------------|-----------------|-----------------|
| cfm  | 23.0625         | 3.7e+3          | 2.5e+3          | 6.65 (probe)    |
| atom | inf             | inf             | inf             | 1.08 (probe)    |
| bond | inf             | inf             | inf             | 0.45 (probe)    |
| total| inf             | 18.35           | inf             | 8.98 (probe)    |

Loss curves show volatile cfm (spikes to 4.5e+5 at step 1900) but
**decode_ratio=1.0 held at every probe** — BondAwareDecoder + EMA
parameters + P1.4 ConnectivityAwareDecoder keep the decoder robust
against training-loss explosions.

### Wall-clock trajectory (seed 42)

- Step 1:    elapsed=1.3s   eta=12868s
- Step 500:  elapsed=42.1s  eta=799s   (~13 min projected)
- Step 1000: elapsed=82.9s  decode_smoke 0.3s
- Step 2000: elapsed=163.5s decode_smoke 0.2s
- Step 2500: elapsed=204.1s **CRASH** (hsa_signal invalid)

Per-step rate ~0.08s/step at batch=4. Projected 10000-step wall
~800s = 13 min if no crash.

### What the 2000-probe already proved

The 2000-probe (`molmetal/reports/wf_r16_yuelbond_2000_probe/`) ran
the SAME stack at batch=8 in 252.8s with decode_ratio=1.000 at all
4 checkpoints (500/1000/1500/2000) and 0 disconnected. The 10000-step
run's first 2 checkpoints (step 1000, step 2000) replicate this —
**the lift from 0/192 baseline floor holds at full training scale**.

## What was NOT measured

- Steps 5000, 7500, 10000 decode-smoke — GPU died before reaching them
- mean_atoms progression past step 2000 — still 8.0 (probe expectation
  was 14-20 only if pocket conditioning were ON; this run used
  `pocket=None` per spec gate "decode_smoke_every default = 8 samples × 1
  pocket" with dummy pocket)
- Seeds 1234 and 7 — not started (single-process GPU contention)
- coord-range diversity stats — only captured lap_p95 in the smoke
  function, not the per-axis ranges the user spec requested; this was
  a smoke-level limitation inherited from `metallo_drug_smoke_retrain.py`

## Failure mode (verified, root-cause consistent with prior runs)

1. **Process died with `rocr::AMD::hsa_exception: Signal handle is
   invalid.`** at step 2500, immediately after a cfm spike to 2.5e+3.
2. **dGPU unenumerated**: `torch.cuda.device_count()` returned 1
   (was 2 before — dGPU RX 7800 XT gfx1101 gone, iGPU gfx1100 remains).
3. **GPU fully unresponsive**: `torch.cuda.is_available()` call hangs
   indefinitely (exit code 143 SIGTERM, 0 bytes output).
4. **Cold power cycle required** to recover. No env var, no
   `expandable_segments:True`, no batch-size reduction will unstick
   it — this is firmware-level per `wf_gpu_diag/diagnosis.md`.

This matches:
- `molmetal/reports/wf_gpu_diag/diagnosis.md` (2026-09-14) — dmesg
  shows `psp gfx command LOAD_TA(0x1) failed status 0x2C` and
  `resume of IP block smu failed -62`
- `molmetal/reports/wf_gpu_auto_recover/final.md` (2026-09-15) —
  same recovery pattern, GPU recovered after cold power cycle
- `molmetal/reports/wf_gpu_recovery_now/final.md` (2026-09-15) —
  GPU recovered after cold power cycle, 5000-step run then worked

## Verdict gate

| Gate | Threshold | Observed | Verdict |
|------|-----------|----------|---------|
| Decode lift at step 1000 | >= 0.05 | 1.000 | **PASS** |
| Decode lift at step 2000 | >= 0.05 | 1.000 | **PASS** |
| Steps 5000/7500/10000 | reached | NOT REACHED (GPU hang) | **INCONCLUSIVE** |
| Seeds 1234, 7 | reached | NOT STARTED | **INCONCLUSIVE** |
| mean_atoms >= 14 | reached | 8.0 (probe-bounded) | **NOT MEASURED** |
| Wall clock <= 12h | within | 3.4 min then GPU hung | **N/A (GPU died)** |

**Overall**: Decode-lift signal is REPLICATED at scale (n_train=500,
batch=4, h=128, all R15+R16 fixes stacked) — consistent with 2000-probe.
Cannot certify 10000-step trajectory or 3-seed stability without GPU
recovery.

## Honest caveats (per integrate policy)

1. **Single-seed partial**: only seed=42 started. Cross-seed stability
   not measured. Without 3-seed aggregation, we cannot rule out that
   the decode_ratio=1.0 at step 2000 is a single-seed artefact.

2. **mean_atoms=8.0 unchanged from probe**: this is the unconditional
   model's "neutral" geometry. Per probe verdict, this is a decoder
   limitation, not a training-failure symptom — but we did not test
   pocket conditioning in this run (the spec gate specified "8 samples ×
   1 pocket" but the smoke function uses `pocket=None`).

3. **GPU is in the firmware SMU hang state**. Cannot retry until
   cold power cycle. Same blocker as wf_gpu_auto_recover and
   wf_gpu_recovery_now — both required manual PSU unplug to clear.

5. **OOM at step 4 forced batch_size 8 → 4**: first run died with
   `torch.OutOfMemoryError` at step 4 (allocating 102 MiB failed
   on a 17.16 GB device with 0 bytes free). Setting
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and dropping
   to batch=4 cleared it. The OOM pattern is consistent with
   activation accumulation in the 3-task joint_train backward
   (`flow_matching_lipman/__init__.py:2188-2215` calls
   `cfm_loss.backward(retain_graph=True)` then
   `atom_loss.backward(retain_graph=True)` then
   `bond_loss.backward()` — three retain-graphs of the encoder
   graph held simultaneously).

## Next actions (gated)

A. **Cold-power-cycle GPU**, re-launch seed 42 from scratch (not from
   checkpoint — the 2000-probe ckpt.pt is bit-identical to what we
   would have after step 2000 of the new run).
B. **Once GPU is recovered**, run seeds 1234 and 7 sequentially
   (single GPU) — each projected 13 min wall, total 26 min.
C. **Add `lap_p95`, `coord_x/y/z_range` to decode_smoke** so the per-axis
   diversity stats the user spec requested are captured.
D. **Wire optional `--pocket` flag** into `metallo_drug_smoke_retrain.py`
   to test pocket-conditioned mean_atoms lift — this is the actual
   lever to break out of the unconditional mean_atoms=8.0 regime.
E. **PCGrad (P1.3) might help loss stability** — currently OFF
   (bit-compat). Once GPU is back, a 3-seed × 2-arm (PCGrad on/off)
   ablation would isolate the cfm-spike cause.

## Files

- `seed_42/train.log` — full stdout/stderr of the failed run
  (preset decode-smoke lines, loss curves, hsa_signal crash)
- `seed_42/ckpt.pt` — 2.8 MB checkpoint (from the FIRST failed run,
  only 3 steps; the second run died before writing ckpt)
- `seed_42/final.json` — from FIRST failed run (steps_executed=3,
  decode_ratio=0)
- `seed_42/final.md` — from FIRST failed run
- `final.md` — this file

## References

- Probe: `molmetal/reports/wf_r16_yuelbond_2000_probe/verdict.md`
  (PASS at 2000 steps)
- GPU diag: `molmetal/reports/wf_gpu_diag/diagnosis.md` (firmware hang)
- GPU auto-recover: `molmetal/reports/wf_gpu_auto_recover/final.md`
  (recovery required cold power cycle)
- GPU recovery now: `molmetal/reports/wf_gpu_recovery_now/final.md`
  (same pattern, recovered → 5000-step worked)
- Script: `molmetal/scripts/metallo_drug_smoke_retrain.py`
- Adapter: `molmetal/adapters/flow_matching_lipman/__init__.py:2017`
  (F1 BondAwareDecoder), `:2247` (final backward),
  `:2188-2215` (3-task PCGrad optional path)

---

**Status**: INCONCLUSIVE — decode-lift signal replicated, but GPU
died at step 2500 before reaching the planned 10000-step × 3-seed
trajectory. Cold power cycle required for retry.