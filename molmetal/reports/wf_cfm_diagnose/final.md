# WF-CFM-Retrain-Diagnose — final report

**Date:** 2026-09-14 (UTC)
**Status:** **NOT RUN — ROCm/HSA runtime unavailable in this session.**
**Author:** WF-CFM-Retrain-Diagnose

---

## 1. Goal (verbatim from spec)

> WF-CFM-Retrain-Diagnose goal: run a SHORT CFM retrain diagnostic
> (5000-step + hidden-dim 32 vs original 2000-step + hidden-dim 32) to
> test whether retraining lifts decode_ratio from 0/384 to nonzero.
> Budget: ≤30 min wall-clock.
> If decode_ratio > 0.5: SUCCESS.
> If decode_ratio in [0, 0.5]: PARTIAL.
> If decode_ratio = 0: FAILURE.

## 2. CLI (verbatim from spec)

```bash
uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --seeds 42 0 1234 7 2024 31415 \
    --train-steps 5000 --n-train 32 --ode-steps 64 \
    --n-samples 16 --hidden-dim 32 --n-layers 2 --lr 0.0001 \
    --vocab-mask --bond-head learned \
    --output-dir molmetal/reports/wf_cfm_diagnose/ \
    --gpu-binary scripts/_fake_vina.sh
```

Script confirmed (molmetal/scripts/r10_cfg_real_crossdocked.py:190–251):
* All flags match the spec.
* Hard guard at line 261: `if not torch.cuda.is_available(): raise RuntimeError('Real ROCm GPU required')`.

## 3. Result table — MEASURED (failed to start)

| metric                  | value         | source            |
|-------------------------|--------------:|-------------------|
| `torch.cuda.is_available()` | **False** | `uv run python -c "import torch; print(torch.cuda.is_available())"` |
| `torch.cuda.device_count()` | 0         | same              |
| `torch.cuda.init()`     | raises `No CUDA GPUs are available` | manual call |
| `rocminfo`              | `HSA_STATUS_ERROR` (ROCk module loaded but HSA init fails) | `/opt/rocm/bin/rocminfo` |
| `run status`            | **NOT RUN** | script bails at line 261 |
| `decode_ratio_after_5000_steps` | **NOT MEASURED** | n/a (no GPU) |
| `bond_loss_initial`     | **NOT MEASURED** | n/a (no GPU) |
| `bond_loss_final`       | **NOT MEASURED** | n/a (no GPU) |
| `n_decoded`             | **0** (trivially — harness never produced samples) | not measured |
| `n_finite`              | **0** | not measured |
| `n_requested`           | 0 | not measured |
| `improvement_vs_baseline` | **n/a** | baseline = 0/384, this run = 0/0 |

## 4. Hardware snapshot (MEASURED)

| Device | PCI | DID     | Name                       | Power state |
|-------:|-----|---------|----------------------------|-------------|
| 0      | 0000:03:00.0 | 0x747E | AMD Radeon Graphics (iGPU) | low-power / N/A |
| 1      | 0000:c8:00.0 | 0x1900 | AMD Radeon 780M Graphics   | perf=auto, active, 57°C, 51.7W |

Both AMD; gfx1101 (RDNA3); integrated APU not discrete. The spec
mentions "RX 7800 XT gfx1101 wave64" but this hardware is actually
`AMD Radeon 780M Graphics` (mobile RDNA3 iGPU) — `gfx1101` is the
shared GPU family, but the chip is the APU.

## 5. Env vars tried (none helped)

* `HIP_VISIBLE_DEVICES=0` / `1` / `2` — no effect
* `ROCR_VISIBLE_DEVICES=0` / `1` / `2` — no effect
* `HSA_OVERRIDE_GFX_VERSION=10.3.0` / `11.0.1` — no effect
* `torch.cuda.init()` direct call — same `No CUDA GPUs are available`

User lacks sudo (`/sys/class/drm/card*/device/power/control` is root-owned,
`sudo: a password is required` even with `NOPASSWD`-style invocations).
Therefore unable to: bump perf level on iGPU, set
`/sys/class/drm/card0/device/power/control = on`, or otherwise
reset the ROCk module.

## 6. Previous successful baseline (project memory)

`molmetal/reports/wf2_cfg_e2e_a5/report.json`:
* `date_utc = 2026-09-14T03:22:11.691642+00:00`
* `gpu = AMD Radeon Graphics`, `hip = 7.2.53211`
* exit 0; `decode_ratio = 0/384`
* `n_decoded / n_finite = 0.000000`

So the **same hardware** ran successfully 4 hours earlier in the
project session — the iGPU works *intermittently*. Today it does
not. Uptime `4:03` at the time of this run; the wf2 baseline was
written before the most recent system reboot.

## 7. Honest framing — MEASURED vs PROJECTED

### MEASURED (2026-09-14, this run)

* GPU side is the blocker. The harness script and CLI flags are
  exactly as specified; the script aborts before training begins.
* `decode_ratio` cannot be measured because no CFM checkpoint was
  trained. The previous 2000-step baseline (0/384) is the only
  MEASURED datapoint.

### PROJECTED (NOT measured today)

* If the GPU comes back (e.g., next session, sudo reboot), the
  exact same command (`uv run python molmetal/scripts/...`) should
  reproduce the wf2 baseline at 2000 steps; doubling to 5000 steps
  is the cheapest experiment to test the hypothesis "retrain at
  higher step count lifts decode_ratio".
* Memory says "increase `--train-steps` from 2000 → 10000+ could
  lift the CFM field enough" (wf2_final.md §3). The 5000-step
  diagnostic is half that escalation; if it lands ≥ 0.5 we still
  need a 10000-step run, if it lands 0 we know the architectural
  issue is structural, not training-time.

## 8. Verdict — no escalation evidence possible today

**Cannot decide between Round-12 paths (full retrain vs. CFM
rework vs. lambda-only fallback).** The diagnostic is blocked by
GPU/runtime availability, not by the CFM itself.

| Round-12 path                                         | Today (no GPU) | When GPU is back |
|-------------------------------------------------------|----------------|------------------|
| (a) full retrain @ 10000-step + hidden-dim 64          | **BLOCKED**    | re-run diagnostic first; if `decode_ratio_after_5000_steps > 0.5`, schedule (a) |
| (b) CFM architectural rework                          | **BLOCKED**    | if `decode_ratio_after_5000_steps` ∈ [0, 0.5] |
| (c) lambda-only column for Round-12                   | **STILL OPEN** | lambda path does not need GPU; can ship today without diagnostic |

**Honest framing:** the CFM retrain diagnostic is *not* a small
artifact — it's the only direct test of whether the geometric
generator learns pocket-conditioned placement. The current GPU
outage forces us to keep the lambda-only path as the *default*
for Round-12, and revisit the CFM decision once the GPU is back.

## 9. Files written

* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_diagnose/final.md` — this report.
* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_diagnose/preflight.log` — GPU/HSA runtime probe.
* `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_cfm_diagnose/run.log` — empty (script never ran past CUDA check).

## 10. Recommended next step (user-facing)

Either:
1. **Re-run this diagnostic when GPU is back online** (preferred —
   the same `uv run` command should work; this is the smallest
   experiment that can answer the CFM question).
2. **Proceed with Round-12 lambda-only column today** (no GPU
   needed; the lambda path is the WF-Lambda-1/2/3/4 verified
   generator and is the recommended fallback in wf2_final.md).
