# TODO-87 — Wf-29 Phase 3 CPU-Only Fallback Plan (defensive, contingent)

**Date**: 2026-09-26
**Status**: 🛡️ **Defensive contingency — likely SKIP** (Phase 2 GPU-1 already SHIPPED 2026-09-26, see TODO-86)
**Priority**: LOW (only if Phase 2 had failed; not blocking R18 close-out)
**Owner**: Hugo Lee + ultracode agent
**Parent**: `Wf-29` (workflow) — Phase 3 of multi-phase rollout

## Why this plan exists

Wf-29 Phase 2 (TODO-86) **already shipped on GPU 1** (RTX 5090, 32 GB,
fully idle, peak ~1.3 GB used). Total wall = **1154.1 s (~19 min)** for
all 4 train scripts (se3sb / transport / multiph + R17 baseline).
**No OOM, no contention, no decorator exit-code issue that blocked
artifact production.** All 38 tests in
`molmetal/tests/test_wf29_smoke_bench.py` pass.

This Phase 3 plan is **defensive only**: a fallback that would be
executed **iff** Phase 2 had failed (venv/import/OOM/CUDA mismatch) to
verify the train scripts are usable on CPU-only hardware.

**Decision**: with Phase 2 already SHIPPED, this Phase 3 is **kept as a
contingency document** and **not executed now**. If a future block hits
GPU resource contention (Wf-23 Blackwell queue, CI budget, etc.),
re-open this file and execute the 4 steps below.

## Pre-flight: did Phase 2 actually fail? (skip-gate)

Before running this Phase 3, verify that Phase 2 *did* fail:

```bash
# Skip-gate — if all 3 conditions below are TRUE, Phase 2 succeeded.
# In that case, do NOT run Phase 3. Mark this TODO as RESOLVED-NO-OP.
test -f /home/hugo/codes/mol-metal/molmetal/reports/wf29_smoke/se3sb/metrics.json
test -f /home/hugo/codes/mol-metal/molmetal/reports/wf29_smoke/transport/metrics.json
test -f /home/hugo/codes/mol-metal/molmetal/reports/wf29_smoke/multiph/metrics.json
pytest /home/hugo/codes/mol-metal/molmetal/tests/test_wf29_smoke_bench.py -q
```

**Current state (2026-09-26)**: all 3 metrics.json present + 38 tests
pass → **skip-gate triggers, Phase 3 = NO-OP**. This file is kept as
archival contingency per the workflow-authoring skill ("keep alternatives
auditable").

## What Phase 3 would do (if executed)

### Goal

Re-run the 3 R18 train scripts on **CPU-only** to verify they remain
runnable when no GPU is available. Same data, smaller sample cap
(`--train-max-per-metal 50` vs GPU's 100), same eval shape, smaller
wall budget tolerated.

### Hardware assumption

- venv: `molmetal/.venvs/mol-metal-venv/bin/python` (torch 2.14.0+cu130,
  numpy 2.5.3 — **same as Phase 2**, no env switch needed)
- CPU only: `CUDA_VISIBLE_DEVICES=""` (force-disable CUDA)
- Data: `.data/molmetal/metalcytotox_5metal_multitask.csv` (exists)
- Pre-existing reports dir: `molmetal/reports/wf29_smoke_cpu/` (NEW)

### Scope (smaller than Phase 2)

3 scripts, **not** 4. We deliberately drop `train_dmpnn_multitask.py`
because:
- R17 baseline already shipped on CPU in R17 close-out (TODO-50)
- D-MPNN is the **fastest** of the 4 on CPU; not a stress test
- Wall budget cut from "4 hours" → "3 hours" (1 hr per script)

| # | Script | Args | Expected wall |
|---|---|---|---|
| 1 | `train_se3sb.py` | `--data ... --seeds 0 --epochs 1 --train-max-per-metal 50` | 30-90 min |
| 2 | `train_transport.py` | same shape | 30-90 min |
| 3 | `train_multiph.py` | same shape | 30-90 min |

Note: `--epochs 1` (down from Phase 2's 10). The point is to verify
"the script trains without OOM/import errors" — not to produce a
competitive AUC. AUC will be near random for `--epochs 1` and that's
fine; we report it honestly.

### Quick & smart decisions

1. **CUDA disable via env var, NOT re-install**: `CUDA_VISIBLE_DEVICES=""`
   forces CPU. No need to install `torch-cpu`; the existing `torch-cu130`
   wheel falls back to CPU when no GPU is visible.

2. **Run sequentially, NOT in parallel**: same as Phase 2. CPU cores
   can saturate on even 1 epoch of EGNN. Parallel = thrash.

3. **Use `--train-max-per-metal 50` (half of Phase 2)**: cuts wall ~2×,
   still validates the dataloader + model fwd/bwd paths.

4. **Skip the comparison.md + ablation_lift.json aggregation**: Phase 3
   is a "scripts work?" smoke, not a benchmark comparison. A single
   `cpu_only_summary.md` is sufficient.

5. **Tests: 5 tests, NOT 38**: `test_cpu_smoke_<script>.py` × 3 +
   `test_cpu_metrics_json_present` × 1 + `test_cpu_wallclock_logged` × 1.

6. **Honest framing template** (from `party-twenty-spirit` skill §3.5):
   report AUC near 0.5 (1-epoch from random init) AS-IS, no fudging.

### Outputs

```
molmetal/reports/wf29_smoke_cpu/
├── se3sb/
│   ├── train.log
│   └── metrics.json
├── transport/
│   ├── train.log
│   └── metrics.json
├── multiph/
│   ├── train.log
│   └── metrics.json
└── cpu_only_summary.md
```

No ckpt retention (CPU ckpts are useless; Phase 2 GPU ckpts are the
canonical ones).

### Test file (NEW)

`molmetal/tests/test_wf29_cpu_fallback.py` (5 tests):
- `test_se3sb_metrics_json_present`
- `test_transport_metrics_json_present`
- `test_multiph_metrics_json_present`
- `test_wallclock_logged_in_each_log` (each log has a `total wall=` line)
- `test_cpu_only_summary_md_present_and_has_honest_framing`

### Honest framing (must include in cpu_only_summary.md)

> **CPU vs GPU**: CPU is 10-100× slower than RTX 5090 for these
> 3-layer EGNN scripts. Wall-time per script: **30-90 min** vs Phase 2's
> **245-275 s**. AUC: near random (1 epoch × random init). This bench
> **does NOT replace** the Phase 2 GPU numbers (TODO-86) — it only
> confirms the scripts are CPU-runnable for environments without GPU.

## Execution (if skip-gate does NOT trigger — i.e., Phase 2 actually failed)

```bash
cd /home/hugo/codes/mol-metal
mkdir -p molmetal/reports/wf29_smoke_cpu/{se3sb,transport,multiph}

CUDA_VISIBLE_DEVICES="" molmetal/.venvs/mol-metal-venv/bin/python \
  molmetal/scripts/train_se3sb.py \
  --data .data/molmetal/metalcytotox_5metal_multitask.csv \
  --seeds 0 --epochs 1 --train-max-per-metal 50 \
  --out-dir molmetal/reports/wf29_smoke_cpu/se3sb \
  2>&1 | tee molmetal/reports/wf29_smoke_cpu/se3sb/train.log

CUDA_VISIBLE_DEVICES="" molmetal/.venvs/mol-metal-venv/bin/python \
  molmetal/scripts/train_transport.py \
  --data .data/molmetal/metalcytotox_5metal_multitask.csv \
  --seeds 0 --epochs 1 --train-max-per-metal 50 \
  --out-dir molmetal/reports/wf29_smoke_cpu/transport \
  2>&1 | tee molmetal/reports/wf29_smoke_cpu/transport/train.log

CUDA_VISIBLE_DEVICES="" molmetal/.venvs/mol-metal-venv/bin/python \
  molmetal/scripts/train_multiph.py \
  --data .data/molmetal/metalcytotox_5metal_multitask.csv \
  --seeds 0 --epochs 1 --train-max-per-metal 50 \
  --out-dir molmetal/reports/wf29_smoke_cpu/multiph \
  2>&1 | tee molmetal/reports/wf29_smoke_cpu/multiph/train.log

# Aggregate
$EDITOR molmetal/reports/wf29_smoke_cpu/cpu_only_summary.md
pytest molmetal/tests/test_wf29_cpu_fallback.py -v
```

**Verify**: `ls molmetal/reports/wf29_smoke_cpu/*/metrics.json` — all 3
files present.

## Risk register

| Risk | Mitigation |
|---|---|
| 1-epoch wall > 90 min | Stop and report (honest negative); not worth waiting |
| Multiprocessing pool leaks CPU cores | Use `--num-workers 0` if exposed; check CLI |
| Random-init AUC near 0.5 misread as "broken" | Document explicitly in `cpu_only_summary.md` |
| Torch CUDA fallback is buggy on this kernel | Use `torch.set_default_device("cpu")` if needed |

## Status

- **2026-09-26**: Phase 2 GPU-1 SHIPPED (TODO-86). Skip-gate triggers.
  Phase 3 = **NO-OP**, archived as defensive contingency.

## Next

If future Wf-29-block hits GPU unavailability, re-open this file and
execute the 4 commands in the `Execution` section above. Otherwise,
treat as DONE.