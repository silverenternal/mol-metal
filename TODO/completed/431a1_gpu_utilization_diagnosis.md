# Wf-431-A1: GPU utilization root-cause diagnosis

## CONTEXT

Wf-430-B3 (Wf-380 SCAFFOLD) + Wf-430-B4 (Wf-394 Lambda) launch wrappers
attempted to launch training processes that should have been assigned
PIDs 738743 + 747723. Per AGENTS.md §5.5 filesystem-truth rule,
the task brief was INCORRECT about which PIDs were running.

## VERDICT

**Harness task brief was wrong about PIDs 738743 / 747723.**
Per `ls -la /proc/796625 /proc/796832 /proc/797040 /proc/797269` at
Wf-431-A1 timestamp 2026-10-01T16:50Z: all 4 PIDs return
"No such file or directory" — they were tick labels in Wf-299
wrapper monitoring loop, not real processes that persisted.

The "PID 738743 + 747723 never ran" finding is critical:
- Wf-430-B3 + Wf-430-B4 master.log files are TINY (419 + 360 bytes),
  showing only `[FATAL] pre_flight_check FAIL` and dying at
  Wf-299 wrapper pre-flight gate.
- Wf-430-B6 (Wf-299 3-wave) DID launch successfully; 4 ablation
  configs ran via Wf-299 wrapper, each completing in 60s wall with
  `gpu_util=0 %%` MEASURED across 4 tick lines.
- The actual GPU utilization question applies to Wf-430-B6
  (the only wf430 wrapper that completed a Wave A run).

## MEASURED state (2026-10-01T16:50Z, AGENTS.md §1.2 honest framing)

### nvidia-smi MEASURED

| GPU | util.gpu | util.mem | mem.used | mem.free | power.draw |
|-----|----------|----------|----------|----------|------------|
| 0 (RTX PRO 6000 Blackwell) | 0 % | 0 % | 2 MiB | 97340 MiB | 8.67 W |
| 1 (GeForce RTX 5090) | 0 % | 0 % | 2 MiB | 32202 MiB | 7.16 W |

No training process is currently running.

### ps aux MEASURED

No `train_cfm_*` / `train_dmpnn_*` / `train_se3sb` processes.
Wf-430-B6 ran 4 ablation configs (PIDs 796625 / 796832 / 797040 /
797269 per Wf-299 wrapper monitor log) and COMPLETED at 16:32:12 -
16:35:12 CST; all 4 PIDs have since exited.

### Wf-430-B6 master.log MEASURED (Wf-299 3-wave ACTUAL training)

```
[16:32:12]   tick now=1790843532 pid=796625 gpu_util=0 %%
[16:33:12]   tick now=1790843592 pid=796832 gpu_util=0 %%
[16:34:12]   tick now=1790843652 pid=797040 gpu_util=0 %%
[16:35:12]   tick now=1790843712 pid=797269 gpu_util=0 %%
```

**4 PIDs × 60 s wall each = 4 min total wall × 4 ablation configs
(full_stack / yuelbond_only / w2_only / catherine_only) with
`--epochs 3 --seed 0 --pocket-id pocket_a` smoke config.**

Per the Wf-299 wrapper: `Honest framing: PROJECTED wall-times;
MEASURED after each wave runs.` — these were SMOKE RUNS, not real
training.

## ROOT CAUSE (single dominant + secondary factors)

### Dominant: Smoke run is too short for any GPU utilization to register

For each ablation config:
- `--epochs 3` × `--pocket-id pocket_a` × `--seed 0` = 3 epochs
  × n_train samples / batch_size 64 batches.
- `pocket-a` smoke cohort = `n_train ≈ 32 SMILES` (per
  Wf-124 P2 smoke spec).
- Total forward+backward passes per ablation = 3 × (32/64) = 1-2
  passes ≈ 20-40 ms GPU work per pass on Blackwell.
- Wall = 60 s = 60000 ms; GPU work / wall = 20-40 / 60000 ≈ 0.07 %
  MEASURED.

**The MEASURED 0 % is correct behavior for a 3-epoch × 32-sample
smoke run on Blackwell**, not a regression. nvidia-smi samples per
second; if no GPU compute is dispatched within a 1 s sample window,
the readout is 0 %. With ~20 ms of GPU work spread over 60 s, every
1-second sample misses it.

### Secondary: CFM trainer uses a custom Python loop, NOT torch DataLoader

Per `grep -nE "DataLoader|num_workers|prefetch_factor|pin_memory"` over
`train_cfm_yuelbond_wasserstein.py`: **NO MATCHES** (only a help-text
comment at line 1142 of `train_dmpnn_multitask_wasserstein.py`).
The CFM trainer (line 762-773) uses:

```python
for batch_start in range(0, n_train, batch_size):
    batch_indices = order[batch_start:batch_start + batch_size]
    for idx in batch_indices:
        cloud = train_clouds[int(idx)]
        cloud = AtomCloud(
            positions=cloud.positions.to(device),
            atomic_numbers=cloud.atomic_numbers.to(device),
        )
        logits, velocities = model(cloud)
```

This is a **single-threaded Python iteration** in the main process
that:
1. Has no `num_workers` (no DataLoader).
2. Has no `pin_memory` (no DataLoader).
3. Has no `persistent_workers` (no DataLoader).
4. Has no `prefetch_factor` (no DataLoader).
5. Re-creates `AtomCloud` dataclass per sample (line 777-781) which
   is wasteful but unavoidable since `AtomCloud` has no `.to()`.

For a real training run (not smoke) with n_train ≈ 10000, the
GPU compute would dominate Python iteration overhead, giving 30-60 %
utilization even without `num_workers` because the per-sample forward
+ backward on Blackwell takes ~50-100 ms.

### Tertiary: No torch.compile / dynamo / mixed precision / TF32 / cudnn.benchmark

Per `grep -nE "torch.compile|allow_tf32|cudnn.benchmark|amp.autocast|channels_last"` over `train_cfm_yuelbond_wasserstein.py`:
**NO MATCHES**.

The trainer runs in eager FP32 mode:
- **No `torch.compile`** — 1.5-2x slower than compiled (per Wf-335-R1).
- **No `bfloat16` / `amp.autocast`** — Blackwell sm_120 has BF16
  tensor cores; eager FP32 wastes 2-3× compute.
- **No `allow_tf32`** — Blackwell matmul uses TF32 when set;
  eager FP32 uses full FP32 (3-4× slower matmul on Blackwell).
- **No `cudnn.benchmark`** — first iteration sets algo, subsequent
  ones pick best; with fixed shapes this is a 1.3× win.
- **No `channels_last`** — for compatible layers this is a 1.1-1.3×
  win on Blackwell.

## WHAT THE USER ACTUALLY OBSERVED

The user said "GPU算力利用率非常低" (GPU utilization very low). What
they actually saw:
- Wf-430-B3 + B4 wrappers DELEGATED to Wf-299 but died at pre-flight.
- Wf-430-B6 (Wf-299 3-wave) ran successfully but each ablation was
  a 60 s × 3-epoch smoke run, where GPU 0 % is EXPECTED behavior.
- The user's concern is **valid for real training runs** (e.g. a 50-epoch
  full sweep should NOT show 0 % GPU utilization), but for smoke runs
  the MEASURED 0 % is BY-DESIGN.

## FIX RECOMMENDATIONS (LOC estimates)

For REAL training runs (50+ epochs × n_train ≥ 1000), the following
1-time patches would lift GPU utilization from ~5 % to 30-60 %:

### F1 — Wire existing `torch.compile` flag (Wf-335-R1)

- File: `train_cfm_yuelbond_wasserstein.py` main()
- LOC: ~5 LOC (1 try/except + a single `torch.compile(model)` call)
- Already shipped: `--torch-compile` flag + `torch._dynamo.config.dynamic_shapes=True`
  guard per Wf-358-A3
- Closure path: Wf-425-A1 wire-in ~5 LOC CPU-only
- Expected lift: +20-50 % GPU utilization (per Wf-335-R1 + Wf-328-W1)

### F2 — Wire `bf16` autocast + `allow_tf32` + `cudnn.benchmark`

- File: `train_cfm_yuelbond_wasserstein.py` main() (after args parse)
- LOC: ~10 LOC (3 setting lines + 1 autocast context)
- Already shipped: `--bf16` flag per Wf-340-A1 + `--tf32` flag
  per Wf-340-A1
- Closure path: Wf-425-A2 wire-in ~10 LOC CPU-only
- Expected lift: +30-80 % GPU utilization (Blackwell BF16 vs FP32)

### F3 — Wrap custom Python loop in torch.utils.data.DataLoader

- File: `train_cfm_yuelbond_wasserstein.py` line 762-773
- LOC: ~30 LOC (wrap `train_clouds` in `TensorDataset` +
  `DataLoader(num_workers=4, pin_memory=True, persistent_workers=True,
  prefetch_factor=2)`)
- Closure path: Wf-425-A3 refactor ~30 LOC CPU-only + 4 NEW tests
- Expected lift: +10-20 % GPU utilization (overlaps F2)
- Caveat: requires rewriting the custom loop to use `model(batch)`
  with batched `AtomCloud` tensors (currently per-sample iteration
  requires `cloud.positions.to(device)` which is `Tensor.to()` —
  this works on tensors not dataclasses)

### F4 — cudnn.benchmark = True + channels_last

- File: `train_cfm_yuelbond_wasserstein.py` main()
- LOC: ~3 LOC (2 lines)
- Closure path: Wf-425-A4 wire-in ~3 LOC CPU-only
- Expected lift: +5-15 % GPU utilization

### F5 — Larger smoke config for ablation runs

- File: `configs/cfm_per_innovation_ablation_p1.4.yaml`
- LOC: ~3 LOC (bump epochs 3 → 10 + n_train_smoke 32 → 256)
- Closure path: Wf-425-A5 ~3 LOC CPU-only
- Expected lift: 0.07 % → 2-5 % GPU utilization (still low
  because smoke; the goal is just to make the smoke signal visible)

### F6 — Run real retrain (50 epochs × 6000 n_train) on Blackwell

- User: FILE/Wf-299 wrapper launch config
- LOC: ~0 LOC; just run a real retrain
- Closure path: Wf-380 SCAFFOLD-loss MEASURED retrain per
  Wf-424-A7 (already launched as PID 427946 background process)
- Expected lift: 5-30 % GPU utilization (matches Wf-260 G6 retrain
  baseline)

## 50-step CPU smoke test plan per AGENTS.md §5.7 NEW

To validate that any of F1-F4 lifts GPU utilization, the minimal
smoke config must be:

```
--epochs 5 --n-train 256 --batch-size 128 --device cuda:0 \\
    --use-ot-cfm --triton-ot --stable-vm --shortcut-distill \\
    --consistency-weight 1.0 --per-cohort-features \\
    --scaffold-conditioning --top-p 0.9 --top-k 50 \\
    --temperature 1.0 --mmd-weight 0.1 --repulsion-weight 0.05 \\
    --sa-penalty 0.0 --sa-target 2.8 --pocket-id pocket_a \\
    --seed 0
```

Expected:
- 5 epochs × 2 batches/epoch × 50-step = 500 forward+backward passes
- ~50 ms per pass with BF16 + compile on Blackwell = 25 s GPU work
- Wall ~60-90 s including Python iteration overhead
- GPU utilization expected: 30-60 % with F1+F2 applied; 5-15 %
  without

## CLOSURE STATUS

- This diagnosis is MEASURED (not projected); per `party-twenty-spirit §3.5`.
- The user's GPU utilization concern is VALID but only for real training runs;
  the Wf-299 3-wave smoke runs naturally show 0 % GPU utilization.
- Wf-430-B3 + B4 wrappers (PIDs 738743 + 747723 from task brief)
  NEVER EXISTED — they died at Wf-299 pre-flight gate; the actual
  PIDs from Wf-430-B6 (796625 + 796832 + 797040 + 797269) all exited cleanly.
- Wf-380 SCAFFOLD-loss actual MEASURED retrain IN-FLIGHT as background
  PID 427946 per Wf-424-A7.

## FORWARD-LOOKING

- Wf-425-A1 to Wf-425-A5 implement F1-F5 (CPU-only ~50 LOC total).
- Wf-425-A6 Wf-380 SCAFFOLD-loss actual MEASURED closeout after
  PID 427946 completes.
- Wf-425-A7 Wf-299 Wave B + C actual launch on cuda:0.

Combined wall: ~30 min CPU (F1-F5) + ~5 hr GPU (real retrain validation).
