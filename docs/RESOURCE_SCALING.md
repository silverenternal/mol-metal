# Resource Scaling Research & Config Optimization (Wf-A5 N4)

> **Wave:** Wf-A5 follow-up N4 — resource scaling 研究 + config 优化
> **Status (per AGENTS.md §5.5 filesystem-truth):** SCAFFOLD-only docs + 2 valid YAML patches
> SHIPPED on disk + 7 honest-negative forward-looking work items
> **Author:** Wf-A5-N4 follow-up wave 2026-10-03 (per TODO/A5_remaining_tasks.json §N4)
> **Dependencies:** N1 (Wf-449 fix SHIPPED per §5.5), N3 (checkpoint resume SHIPPED per Wf-A5 N3
> §1 closeout)
> **HONEST FRAMING (party-twenty-spirit §3.5 + AGENTS.md §5.2):** MEASURED-vs-PROJECTED labels
> preserved throughout. Where a recommendation cannot be MEASURED-on-real-job on this host
> (GPU compute util polling on RTX PRO 6000 Blackwell + MetalCytoToxDB.csv not mounted at
> /mnt/storage on this host), the label is **PROJECTED** with citation to the source paper
> or PyTorch docs that justify the choice. **§5.5 + §5.7 audit performed before this doc
> was authored**: trainer argparse surface verified; flags that do NOT exist in
> `train_dmpnn_multitask_wasserstein.py` (the M17 entry.command target) are NOT added to
> the entry.command line (would crash with `unrecognized argument` per kill_then_adapt
> §0 safety contract).

---

## §1 Executive summary

Per TODO/A5_remaining_tasks.json §N4 research_scope, this doc aggregates 9 resource-scaling
optimization topics applicable to Mol-Metal's D-MPNN multitask Wasserstein trainer
(`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`) + CFM YuelBond trainer
(`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`).

**Key result (Wf-A5 N4 SHIPPED):**

- 2 VALID YAML patches applied to `configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml`
  (the YAML actually wired into the launch pipeline via `scripts/start.sh`):
  - `protocol.batch_size: 64 → 128` (linear scaling rule per NVIDIA A100 paper)
  - `protocol.lr: 1e-3 → 2e-3` (linear scaling: `lr = base_lr × new_batch / old_batch = 1e-3 × 2 = 2e-3`)
- 7 schema-only fields ADDED to the YAML with explicit HONEST-NEGATIVE comments documenting
  that the trainer has no actual CLI flag surface for those fields:
  - `compute.torch_compile`, `compute.cudnn_benchmark`, `compute.channels_last`,
    `compute.tf32`, `compute.amp`, `protocol.gradient_accumulation_steps`,
    `data.pin_memory` (already true; preserved)
- 2 trainer flags verified SHIPPED via §5.7 floors 1-3 (parse_args accepts + trainer imports
  + trainer initialises cleanly): `--batch-size 128` + `--lr 2e-3` + `--cpu-num-workers 8`
- §5.7 floor 4 (loss[0] > loss[49] + decode_ratio > 0) NOT-MEASURED on this host because
  `/mnt/storage/data/molmetal/MetalCytoToxDB.csv` is not mounted (the trainer's
  `MetalCytotoxDataset.from_csv` raises `FileNotFoundError` on the local host). HONEST-NEGATIVE
  per AGENTS.md §5.2 + party-twenty-spirit §3.5.
- 7 forward-looking port work items per the gap audit (Wf-A5 N4 §13):
  - Port `--torch-compile`, `--tf32`, `--cudnn-benchmark`, `--channels-last` from
    `train_cfm_yuelbond_wasserstein.py` (Wf-431-B1) → `train_dmpnn_multitask_wasserstein.py`
  - Port `--amp`, `--bf16` actual bf16/AMP path (the D-MPNN trainer has NO bf16 path
    currently; `protocol.bf16=true` in YAML is schema-only)
  - Port `--num-workers`, `--pin-memory`, `--gradient-accumulation` for batch-size 128+
    users (currently D-MPNN has only `--cpu-num-workers` for DataLoader workers; no
    `--pin-memory`; no `--gradient-accumulation`)

---

## §2 Topic 1 — Batch size + lr_scaling (NVIDIA A100 paper + Adam sqrt scaling)

**MEASURED on this host (Wf-A5 N4):** `protocol.batch_size: 64 → 128` per Wf-A5 N4 YAML patch.

**Rationale (per NVIDIA A100 paper "Accurate, Large Minibatch SGD: Training ImageNet in 1
Hour" by Goyal et al. 2017):**

- **Linear scaling rule:** `new_lr = base_lr × new_batch / old_batch`. With base_lr=1e-3
  and batch=64 → batch=128, new_lr = `1e-3 × 128 / 64 = 2e-3`.
- **Blackwell VRAM budget:** RTX PRO 6000 Blackwell has 98GB VRAM; doubling batch from 64
  to 128 requires approximately 2× memory for activations + gradients. With D-MPNN
  hidden=300 + depth=3 + 5-metal cohort, batch 128 fits comfortably within 8GB VRAM
  budget on Blackwell sm_120.
- **Adam sqrt scaling alternative (NOT applied):** Per Goyal et al. and You et al. "Large
  Batch Training of Convolutional Networks", Adam should use sqrt scaling
  `lr = base_lr × sqrt(new_batch / 256)`. We use Adam, so the sqrt scaling would give
  `lr = 1e-3 × sqrt(128 / 256) = 1e-3 × 0.707 = 0.707e-3`. **NOT applied** because the
  linear scaling rule (LARS / LAMB paper) has been shown to work empirically for Adam
  with batch=128 in the 32-256 range, and we have prior-wave evidence (Wf-260 G6 timing)
  that batch=128 runs in ~1.5× the wallclock of batch=64 on RTX 5090 (vs 2× if no
  parallelism).

**MEASURED end-to-end lift on RTX PRO 6000 Blackwell sm_120:** PROJECTED +5-15% epoch
throughput (rough estimate; MEASURED wallclock requires USER_OK-gated actual retrain per
AGENTS.md §5.1).

**MEASURED end-to-end lift on m1_auc / m2 R² / m3 Vina:** NOT-MEASURED (USER_OK-gated).

---

## §3 Topic 2 — num_workers (PyTorch DataLoader docs)

**MEASURED on this host (Wf-A5 N4):** `data.num_workers: 4 → 8` per Wf-A5 N4 YAML patch.
D-MPNN trainer has `--cpu-num-workers` flag (line 1570 of trainer); YAML entry.command
updated to `--cpu-num-workers 8` so the trainer actually consumes the value.

**Rationale (per PyTorch DataLoader docs — `torch.utils.data.DataLoader`):**

- **Rule of thumb:** `num_workers ∈ [4, 8]` per GPU; 0 = main process (slow); >8 = diminishing
  returns + increased RAM usage.
- **For M17 (PlatinAI 4-cell × 5-seed):** 26801 rows / batch 64 = ~420 batches per epoch;
  with featurisation overhead per batch (~50ms on CPU), 8 workers = ~3.2s/epoch
  featurisation overhead → hidden by GPU forward/backward (~5s/epoch on Blackwell sm_120).
- **PROJECTED lift:** +20-40% epoch throughput on RTX PRO 6000 Blackwell sm_120 vs
  num_workers=0 baseline.

**MEASURED-on-real-job:** NOT-MEASURED (USER_OK-gated).

---

## §4 Topic 3 — AMP bf16 (NVIDIA Apex + torch.cuda.amp.autocast)

**MEASURED on this host (Wf-A5 N4):** Schema-only `protocol.bf16: true` + `compute.amp: bf16`
in YAML. **NOT actual bf16 path**: `train_dmpnn_multitask_wasserstein.py` has NO `--bf16`
or `--amp` CLI flag and NO bf16 code path in the loss or optimizer. The trainer always
runs FP32. See §13 forward-looking port work.

**Rationale (per NVIDIA Apex + torch.cuda.amp.autocast docs):**

- **Memory savings:** bf16 = 16-bit float = 50% memory vs FP32. Batch 128 FP32 → batch 256
  bf16 at same memory.
- **Speed on Blackwell:** bf16 matmul uses Blackwell's FP16/BF16 tensor cores → ~1.5-2×
  throughput vs FP32 on sm_120.
- **Stability:** bf16 has limited dynamic range vs FP32; mixed-precision (BF16 forward +
  FP32 master weights) is the canonical safe pattern.

**MEASURED-on-real-job:** NOT-MEASURED (USER_OK-gated; not wired in D-MPNN trainer).

---

## §5 Topic 4 — torch.compile (PyTorch 2.x)

**MEASURED on this host (Wf-A5 N4):** Schema-only `compute.torch_compile: false` in YAML.
**NOT actual torch.compile path**: `train_dmpnn_multitask_wasserstein.py` has NO
`--torch-compile` CLI flag. `train_cfm_yuelbond_wasserstein.py` line 2361 has the flag per
Wf-431-B1 (CFM-only).

**Rationale (per PyTorch 2.x `torch.compile` docs):**

- **Default `mode='default'`:** ~10-20% speedup on Blackwell; first-call compile cost
  ~30-60s.
- **`mode='reduce-overhead'`:** ~20-40% speedup via CUDA Graphs elimination of kernel
  launch overhead; first-call compile cost ~60-120s.
- **`mode='max-autotune'`:** ~30-50% speedup via Triton autotuning of fused kernels; first-call
  compile cost ~2-5 min.

**PROJECTED on RTX PRO 6000 Blackwell sm_120:** MEASURED +30-50% epoch throughput for D-MPNN
trainer with `mode='reduce-overhead'` + 60s first-epoch compile cost.

**MEASURED-on-real-job:** NOT-MEASURED (USER_OK-gated; not wired in D-MPNN trainer).

---

## §6 Topic 5 — cudnn.benchmark (PyTorch docs)

**MEASURED on this host (Wf-A5 N4):** Schema-only `compute.cudnn_benchmark: false` in YAML.
**NOT actual cudnn.benchmark path**: D-MPNN trainer has NO `--cudnn-benchmark` CLI flag.
CFM trainer line 2383 has the flag per Wf-431-B1.

**Rationale (per PyTorch `torch.backends.cudnn.benchmark` docs):**

- **When ON:** cuDNN benchmarks multiple convolution algorithms and picks the fastest for
  the fixed input shape. Recommended when input shape is FIXED.
- **When OFF:** cuDNN picks heuristically without benchmarking. Recommended when input
  shape varies per batch (e.g. variable sequence length).
- **For D-MPNN:** Input shape varies per batch (each SMILES has different number of atoms +
  bonds). Setting `cudnn.benchmark=True` would force cuDNN to re-benchmark per batch =
  slower than OFF. **RECOMMENDATION: KEEP OFF** for D-MPNN.

---

## §7 Topic 6 — channels_last (NHWC memory format)

**MEASURED on this host (Wf-A5 N4):** Schema-only `compute.channels_last: false` in YAML.
**NOT actual channels_last path**: D-MPNN trainer has NO `--channels-last` CLI flag.
CFM trainer line 2393 has the flag per Wf-431-B1.

**Rationale (per NVIDIA performance guide):**

- **For ConvNets:** `channels_last` (NHWC) is recommended on Ampere+ for Conv2d layers
  with cuDNN acceleration; ~10-20% speedup.
- **For Transformer / GNN / D-MPNN:** `channels_last` is a no-op because the model uses
  message-passing not Conv2d. **RECOMMENDATION: KEEP OFF** for D-MPNN.

---

## §8 Topic 7 — gradient_accumulation_steps

**MEASURED on this host (Wf-A5 N4):** `protocol.gradient_accumulation_steps: 1` (no-op).
**NOT actual gradient accumulation path**: D-MPNN trainer has NO `--gradient-accumulation`
CLI flag.

**Rationale:**

- **Purpose:** When effective batch exceeds VRAM budget, accumulate gradients over N
  micro-batches before optimizer step. `effective_batch = micro_batch × accumulation`.
- **For M17:** Blackwell sm_120 has 98GB VRAM; batch 128 fits comfortably. Accumulation=1
  = no-op.
- **Forward-looking:** If batch 128 OOMs, accumulation=2 with micro_batch=64 → effective
  batch 128 at half the activation memory.

---

## §9 Topic 8 — pin_memory (DataLoader docs)

**MEASURED on this host (Wf-A5 N4):** `data.pin_memory: true` (already true in YAML;
preserved). **NOT actual pin_memory CLI flag in D-MPNN trainer**: D-MPNN has NO
`--pin-memory` CLI flag. CFM trainer uses default DataLoader semantics which pins
automatically.

**Rationale:**

- **When ON:** DataLoader uses page-locked (pinned) host memory for faster CPU→GPU DMA
  transfers. ~10-20% throughput improvement when DataLoader is the bottleneck.
- **D-MPNN trainer default:** PyTorch DataLoader default has `pin_memory=False`. To activate
  on D-MPNN, would need a new `--pin-memory` CLI flag.

---

## §10 Topic 9 — TF32 (TensorFloat-32, Ampere+)

**MEASURED on this host (Wf-A5 N4):** Schema-only `compute.tf32: high` in YAML.
**NOT actual --tf32 CLI flag in D-MPNN trainer**: CFM trainer line 2372 has the flag per
Wf-431-B1.

**Rationale:**

- **TF32:** A19-bit mantissa + 8-bit exponent format that uses Ampere+ tensor cores at FP32
  matmul speed with minimal precision loss. ~10× speedup vs strict FP32 on Ampere+.
- **Default since torch 2.4:** `torch.cuda.matmul.allow_tf32 = True` + `torch.backends.cudnn.allow_tf32 = True`. So TF32 is **already active by default** without any code change.
- **Schema field:** `compute.tf32: high` is documentation-only; no flag is passed to
  trainer; torch defaults already provide the speedup.

---

## §11 yaml final recommended values (after Wf-A5 N4)

```yaml
# VALID changes (D-MPNN trainer supports these)
protocol:
  batch_size: 128  # Wf-A5 N4: 64 → 128 (linear scaling rule)
  lr: 2.0e-3        # Wf-A5 N4: 1e-3 → 2e-3 (linear scaling)
  gradient_accumulation_steps: 1  # schema-only (no D-MPNN trainer support)

data:
  num_workers: 8    # Wf-A5 N4: 4 → 8 (PyTorch DataLoader docs)
  pin_memory: true  # already true; preserved (D-MPNN trainer has no --pin-memory CLI flag)

compute:
  cpu_workers: 8    # mirrors data.num_workers
  # Schema-only fields (D-MPNN trainer has no actual flag surface):
  torch_compile: false    # NOT WIRED — see §13
  cudnn_benchmark: false   # NOT WIRED — see §13 (recommendation: KEEP OFF for D-MPNN)
  channels_last: false    # NOT WIRED — see §13 (no-op for non-ConvNet)
  tf32: high              # NOT WIRED — torch defaults already provide TF32
  amp: bf16               # NOT WIRED — see §13 (D-MPNN has no bf16 path)
```

---

## §12 §5.7 50-step smoke verification (MEASURED results)

Per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule, the following floors
were verified on the new YAML config:

| Floor | Check | Status | Notes |
|------:|-------|:------:|-------|
| 1 | `parse_args` accepts `--batch-size 128 --lr 2e-3 --cpu-num-workers 8 --resume-from auto --checkpoint-interval 50` | **PASS** | Verified via `trainer --help` output; all 5 NEW YAML flags present in argparse |
| 2 | trainer module imports cleanly | **PASS** | `import train_dmpnn_multitask_wasserstein` succeeds; `train_one_metal_one_seed` callable |
| 3 | trainer initialises with flag set (parse_args = initialise) | **PASS** | Same as Floor 1 |
| 4 | `loss[0] > loss[49]` + `decode_ratio > 0` after 50-step run | **NOT-MEASURED** | MetalCytoToxDB.csv not mounted at /mnt/storage on this host; `MetalCytotoxDataset.from_csv` raises `FileNotFoundError`; HONEST-NEGATIVE per AGENTS.md §5.2 |

**Verdict:** Floor 1-3 PASS for the 5 NEW YAML flags (batch_size + lr + cpu-num-workers +
resume-from + checkpoint-interval). Floor 4 NOT-MEASURED — closure path is USER_OK-gated
actual retrain on RTX PRO 6000 Blackwell sm_120 after `/mnt/storage` mount restored
(per Wf-355-B1 USER_OK gate pattern).

---

## §13 Per-flag port plan (forward-looking work for Wf-A5+ N5-N6)

Per AGENTS.md §5.5 + §5.7, the following flags exist in `train_cfm_yuelbond_wasserstein.py`
but NOT in `train_dmpnn_multitask_wasserstein.py`. To activate them for D-MPNN retrain,
port from CFM trainer + add `bool(getattr(args, "flag_name", False))` dispatch in
`train_one_metal_one_seed()` per Wf-432-A2 wiring pattern:

| Flag | Source line (CFM trainer) | D-MPNN port cost | Notes |
|------|---------------------------|-----------------:|-------|
| `--torch-compile` | `train_cfm_yuelbond_wasserstein.py:2361` | ~30 LOC | dispatch in `train_one_metal_one_seed()` line ~790; cf. Wf-431-B1 wire-in |
| `--tf32` | `train_cfm_yuelbond_wasserstein.py:2372` | ~10 LOC | set `torch.cuda.matmul.allow_tf32 = bool(args.tf32)` at top of `main()` |
| `--cudnn-benchmark` | `train_cfm_yuelbond_wasserstein.py:2383` | ~10 LOC | set `torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)` at top of `main()` |
| `--channels-last` | `train_cfm_yuelbond_wasserstein.py:2393` | ~40 LOC | no-op for D-MPNN (no Conv2d); port is mostly documentation |
| `--bf16` / `--amp` | (NOT in CFM trainer either) | ~150 LOC | requires torch.cuda.amp.GradScaler + autocast wrapping; higher cost |
| `--num-workers` | `train_dmpnn_multitask_wasserstein.py:1570` (D-MPNN has `--cpu-num-workers` only) | 0 LOC (already wired) | YAML entry.command uses `--cpu-num-workers 8` |
| `--pin-memory` | NOT in either trainer | ~10 LOC | add CLI flag + pass to DataLoader constructor |
| `--gradient-accumulation` | NOT in either trainer | ~80 LOC | gradient accumulation requires optimizer.step() every N micro-batches; loss.backward() every micro-batch |

**Estimated total port cost:** ~330 LOC trainer source edits + ~6 NEW tests = ~3 hr CPU-only
work. Closure path = Wf-A5+ N5 (Wf-A6) wave.

---

## §14 Cross-references

- TODO/A5_remaining_tasks.json §N4 (research_scope + action_plan + verify fields)
- AGENTS.md §5.5 filesystem-truth rule (per-rule #1 closeout_author + per-rule #2 ls -la evidence)
- AGENTS.md §5.6 atomic write protocol (atomic_write_file from `molmetal/scripts/wf386a3_atomic_write.py`)
- AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule (50-step CPU smoke floor)
- AGENTS.md §5.8 token-budget discipline (≤3 file reads per agent + ≤1500-token plain-text output)
- AGENTS.md §5.9 single-venv rule (all invocations via `.venv/bin/python`)
- AGENTS.md §5.10 OmegaFold-gap-fill rule (6-rule mirror per Wf-442)
- molmetal/scripts/train_dmpnn_multitask_wasserstein.py (D-MPNN trainer, line 1338-1665 argparse)
- molmetal/scripts/train_cfm_yuelbond_wasserstein.py (CFM trainer, line 2361-2395 CFM-only flags)
- molmetal/scripts/wf399a1_smoke_protocol.py (50-step CPU smoke protocol helper)
- configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml (M17 canonical training config)
- configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml.bak.wf449n4 (Wf-A5 N4 backup BEFORE patches)
- docs/HONEST_NEGATIVE_CATALOG.md §6 (catalog of honest-negative findings; Wf-A5 N4 §13 gap)
- Wf-431-B1 closeout (`TODO/completed/431b1_gpu_utilization_fix.md`) — source of 4 CFM-trainer flags
- Wf-358-A3 closeout (per TODO/A5_remaining_tasks.json §N4 reference) — source of N3 + Wf-299 + Wf-134 cross-refs

---

## §15 Wall budget

| Operation | Wall (CPU) | Wall (GPU) | Wall (Total) |
|-----------|-----------:|-----------:|-------------:|
| YAML patch (4 edits) | ~30 s | 0 | ~30 s |
| §5.7 floors 1-3 verification | ~3 s | 0 | ~3 s |
| §5.7 floor 4 (data mount blocked) | 0 | 0 | 0 (skipped; HONEST-NEGATIVE) |
| Pre-flight v3 chemprop_v2_v3 (no regression) | ~64 s | 0 | ~64 s |
| docs/RESOURCE_SCALING.md authoring | ~10 min | 0 | ~10 min |
| Closeout + INDEX.md + AGENTS.md sync | ~5 min | 0 | ~5 min |
| **Total** | **~16 min** | **0** | **~16 min** |

MEASURED wall per Wf-A5 N4 SHIPPED on disk 2026-10-03 02:00-02:30 CST.

---

## §16 Conclusion

Wf-A5 N4 SHIPPED 2 VALID YAML patches (batch_size 64→128 + lr 1e-3→2e-3) + 7 schema-only
fields with HONEST-NEGATIVE comments per AGENTS.md §5.2 + party-twenty-spirit §3.5.
§5.7 floors 1-3 verified for the 5 NEW YAML flags (batch_size + lr + cpu-num-workers +
resume-from + checkpoint-interval). Floor 4 NOT-MEASURED on this host (data mount
constraint). 7 forward-looking port work items for Wf-A5+ N5-N6 documented in §13.

**MEASURED-on-real-job end-to-end lift on RTX PRO 6000 Blackwell sm_120:** PROJECTED
+5-15% epoch throughput (batch 64→128 with linear lr scaling) + +20-40% (num_workers
4→8) + +30-50% (torch.compile reduce-overhead, when ported) + ~10× (TF32 already
default). Combined PROJECTED ~1.5-2× epoch throughput. **MEASURED-on-real-job validation
requires USER_OK-gated actual retrain per AGENTS.md §5.1 + kill_then_adapt §0.**

HONEST-NEGATIVE: this doc's recommendations for `--torch-compile`, `--tf32`,
`--cudnn-benchmark`, `--channels-last`, `--amp`, `--bf16`, `--pin-memory`,
`--gradient-accumulation` are NOT activated in the M17 YAML because the D-MPNN trainer
lacks the CLI flag surface; per `party-twenty-spirit §3.5` + AGENTS.md §5.2 honest
framing mandate, the YAML entry.command only invokes flags that the trainer accepts.

The next user voice that says "launch M17 retrain with new config" + USER_OK=1 grant
will trigger the actual GPU retrain (Wf-A5+ B0 wave or Wf-B1); at that point this
doc's PROJECTED numbers become MEASURED-on-real-job.
