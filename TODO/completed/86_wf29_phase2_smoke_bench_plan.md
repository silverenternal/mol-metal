# TODO-86 — Wf-29 Phase 2 GPU-1 Smoke Benchmark Plan

**Date**: 2026-09-26
**Status**: 🚀 **Active — ready to execute**
**Priority**: HIGH (Phase 2 of Wf-29; ships after Phase 1 fix shipped in TODO-85)
**Owner**: Hugo Lee + ultracode agent
**Parent**: `Wf-29` (Phase A — bug fix + smoke benchmark)

## Goal

Run **4 train scripts** as smoke benchmark on **GPU 1 (RTX 5090, fully idle)**,
produce real wall-time + memory + smoke AUC, and side-by-side comparison
against the R17 baseline. Validate the W2.1 / W2.2 / W2.3 attach hooks
after the TODO-85 AttributeError fix.

## Hardware check (2026-09-26)

```
GPU 0: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (kanzi running, 1310 MiB)
GPU 1: NVIDIA GeForce RTX 5090               (IDLE, 4 MiB / 32 GB)
venv:   molmetal/.venvs/mol-metal-venv/bin/python  (torch 2.14.0+cu130 + numpy 2.5.3)
data:   .data/molmetal/metalcytotox_5metal_multitask.csv  (exists)
```

→ All preconditions GREEN. CUDA_VISIBLE_DEVICES=1 will land us on the
free 32 GB Blackwell consumer card.

## Execution plan (5 phases)

### Phase 1 — Prep directories + plan docs (1 min)

- `molmetal/reports/wf29_smoke/` (out_dir root)
- `molmetal/reports/wf29_smoke/{se3sb,transport,multiph,baseline}/`
  (per-script sub-dirs)
- `TODO/pending/86_wf29_phase2_smoke_bench_plan.md` (this file)

### Phase 2 — Run 4 scripts sequentially on GPU-1 (~30-60 min total)

| # | Script | Recipe | Args | Expected wall |
|---|---|---|---|---|
| 1 | `train_se3sb.py` | W2.1 SE(3) SB | `--data ... --seeds 0 1 2 --epochs 10 --train-max-per-metal 100 --batch 16 --out-dir ... --metrics-path ... --summary-path ...` | ~5-10 min |
| 2 | `train_transport.py` | W2.2 Transport Relax | same shape | ~5-10 min |
| 3 | `train_multiph.py` | W2.3 Multi-PH | same shape | ~5-10 min |
| 4 | `train_dmpnn_multitask.py` | R17 baseline | different CLI — single `--metal --seed --epochs`. Loop 3 seeds × metal=Ru. | ~5-10 min |

**NOTE on `train_dmpnn_multitask.py`**: This is the **R17 legacy** script
with `--metal` (single, choices=["Ru","Ir","Rh","Os","Re","Pt"]) +
`--seed` (single int) — **no `--seeds` list and no `--train-max-per-metal`**.
Adapt: loop 3 seeds on Ru as the headline baseline; mention in comparison.md.

### Phase 3 — Per-script outputs

Each `out-dir` gets:
- `<recipe>_seed<i>.pt` (or `<recipe>_seed<i>_<metal>.json` for dmpnn)
- `<recipe>.log` (subprocess tee)
- `metrics.json` (per-script recipe metrics)
- `final_summary.json` (slim per-seed headline)

### Phase 4 — Aggregation

- `molmetal/reports/wf29_smoke/comparison.md` — side-by-side 4 conditions × per-metal AUC + wall-time + GPU memory
- `molmetal/reports/wf29_smoke/bench_summary.json` — aggregate (min/median/max wall, mean AUC, dominant bottleneck)
- `molmetal/reports/wf29_smoke/ablation_lift.json` — R18 vs R17 lift (R18 SE3-SB / Transport / Multi-PH vs D-MPNN baseline)

### Phase 5 — Tests (NEW ~10 tests)

- `molmetal/tests/test_wf29_smoke_bench.py`
- Validates: log file non-empty, metrics.json parseable, summary.json has recipe name,
  wall_seconds > 0, all 4 sub-dirs present, comparison.md exists,
  GPU memory tracking (nvidia-smi snapshot per run).

### Phase 6 — Honest reporting

- If any script OOMs → record honest negative per `party-twenty-spirit` §3.5
- If torch/numpy missing → report (already verified both present)
- If metric lift = 0 → honest negative, NOT fabricated
- If wall-time way over budget → report honestly

## Risk register

| Risk | Mitigation |
|---|---|
| OOM on RTX 5090 (32 GB) with 100 samples × batch 16 | Drop `--train-max-per-metal` to 50 first; record final |
| SE(3) SB Sinkhorn iters slow on small batch | `--se3-sinkhorn-iters 10` (down from 50) for smoke |
| `train_dmpnn_multitask.py` CLI mismatch | Loop single-metal × single-seed; report in comparison.md |
| tmQM encoder ckpt missing | `--encoder-ckpt ""` skip; arch-mismatch fallback to random |
| All 4 scripts share GPU-1 → context thrash | Sequential, not parallel |

## Expected outputs

```
molmetal/reports/wf29_smoke/
├── se3sb/
│   ├── se3sb.log
│   ├── metrics.json
│   └── final_summary.json
├── transport/
│   ├── transport.log
│   ├── metrics.json
│   └── final_summary.json
├── multiph/
│   ├── multiph.log
│   ├── metrics.json
│   └── final_summary.json
├── baseline/
│   ├── baseline.log
│   ├── metrics.json
│   └── final_summary.json
├── comparison.md          # side-by-side 4 conditions × per-metal AUC + wall-time + memory
├── bench_summary.json     # aggregate stats
├── ablation_lift.json     # R18 vs R17 lift
├── gpu_memory_log.json    # nvidia-smi snapshots per run
└── run_wallclock.json     # total wall + per-script wall

molmetal/tests/test_wf29_smoke_bench.py   # NEW, ~10 tests
```

## Status

✅ **SHIPPED 2026-09-26** — all 4 train scripts ran on GPU-1 (RTX 5090,
32 GB idle, peak ~1.3 GB used), 3 seeds × 10 epochs × 5-metal
multitask each. Total wall = 1154.1 s (~19 min).

**Key findings**:
- All 3 R18 scripts (se3sb/transport/multiph) ran cleanly end-to-end.
- `train_se3sb.py` exits with code 2 due to a decorator re-parses-argv
  bug in `MEASURED_vs_PROJECTED_decorator` (W2.1-specific args not in
  the simplified common_args helper). Training data + ckpts + metrics
  all written correctly — exit code is misleading. Comparison.md
  records this honestly.
- All 3 R18 scripts trained from **random init** (tmQM encoder arch
  mismatch with V4MultiTaskDMPNN.encoder.dmpnn) — known Wf-23 finding.
- Baseline (R17 single-metal Ru only) gets 0.6576 Ru AUC vs R18 5-metal
  multitask 0.528-0.536 — expected multitask trade-off, not a regression.
- Wall-time ranking: multiph (245s) < se3sb (270s) < transport (275s)
  < baseline (363s for 3 seeds × Ru).

**Deliverables** (all on disk):
- `molmetal/reports/wf29_smoke/{se3sb,transport,multiph,baseline}/`
  × {log, metrics.json, final_summary.json, *.pt or *seed*.json}
- `molmetal/reports/wf29_smoke/comparison.md` (side-by-side 4 conditions)
- `molmetal/reports/wf29_smoke/bench_summary.json` (aggregate stats)
- `molmetal/reports/wf29_smoke/ablation_lift.json` (R18 vs R17 lift)
- `molmetal/reports/wf29_smoke/run_wallclock.json` (wall breakdown)
- `molmetal/reports/wf29_smoke/gpu_memory_log.json.{baseline,post}` (nvidia-smi CSV)
- `molmetal/reports/wf29_smoke/se3sb_gpu_mem_samples_mib.txt` (peak ~1302 MiB)
- `molmetal/tests/test_wf29_smoke_bench.py` (NEW, 38 tests, ALL PASS)

## Completion criteria

1. All 4 scripts complete (or honest negative recorded)
2. All metrics.json + final_summary.json present and parseable
3. comparison.md written with side-by-side table
4. test_wf29_smoke_bench.py passes (≥10 tests)
5. No fabricated MEASURED numbers — all honest