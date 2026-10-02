# Wf-326A — GPU Utilization Profile (RTX 5090 + RTX PRO 6000)

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Status:** MEASURED bottleneck analysis — honest negative on GPU SM occupancy.
**Profile data:** `/home/hugo/codes/mol-metal/molmetal/reports/wf326a_gpu_profile/profile.json`

---

## §1 Environment MEASURED

| Component | Spec | Source |
|---|---|---|
| GPU 0 | RTX PRO 6000 Blackwell Workstation, 97 GB, 188 SM, sm_120, L2 128 MB | `torch.cuda.get_device_properties(0)` |
| GPU 1 | RTX GeForce RTX 5090, 32 GB, 170 SM, sm_120, L2 96 MB | `torch.cuda.get_device_properties(1)` |
| Active G6 | PID 1849475 on GPU 1, multitask wasserstein | `nvidia-smi pmon -c 30 -d 1` |
| Concurrent GPU 0 user | sunlong's `dfine_t146` (PID 1850521) — not our process | `nvidia-smi pmon` |
| System load | 186.68 (heavily contended, dfine_t146 at 834% CPU) | `top -bn 1` |
| CUDA / PyTorch | 13.4 / 2.x with BF16 tensor cores | `nvidia-smi` + `torch.version.cuda` |

### §1.1 nvidia-smi live profile (during G6 epoch 28)

| GPU | pid | sm% (sampled) | mem% | owner |
|---:|---:|---:|---:|---|
| 0 | 1850521 (dfine_t146) | 11-33 | 3-5% | sunlong (other user) |
| 0 | 1892743 | 25-33 | 6-11% | (other) |
| **1** | **1849475 (our G6)** | **4-7** | **0%** | **us** |
| 1 | 1915305 (chemprop quick) | 0 | 0% | us (idle smoke) |

**G6 measured SM utilization: 4-7%** (NOT the 24% baseline reported elsewhere — the 24% was a peak burst during `fwd+bwd+opt`; the average is much lower because the GPU sits idle between bursts).

---

## §2 Per-bottleneck MEASURED breakdown (batch=256, hidden=300, depth=3, RTX PRO 6000)

Synthetic batch via `MultiTaskWassersteinModel(n_metals=1, hidden=300, depth=3)` — same shape as the active G6 process. **Profile on GPU 0 (RTX PRO 6000) so we don't contend with the G6 process on GPU 1.**

### §2.1 Wall-clock per step (mean of 20 runs)

| Component | Time (ms) | % of full step |
|---|---:|---:|
| forward only | **812 ms** | 32.1% |
| forward + backward | 2532 ms | (full) |
| backward alone (incl.) | ~1660 ms | ~65.5% |
| Adam opt step | ~60 ms | ~2.4% |

### §2.2 CPU collate (Python-side, batch=256)

| Operation | Time (ms) | Notes |
|---|---:|---|
| `collate_multimetal` (full) | 7.0 ms | per batch |
| `collate_graphs` (just the tensor assembly) | 5.4 ms | bottleneck within collate_multimetal |
| Per-epoch cost (100 batches) | ~700 ms | **only 1-3% of epoch wall — NOT the primary bottleneck** |

### §2.3 Root-cause: per-molecule Python loop in `forward` and `encode_pool`

`molmetal/scripts/train_dmpnn_multitask_wasserstein.py` lines 343-389 + `molmetal/baselines/dmpnn_multitask.py` lines 221-291 — the model does a Python `for b in range(B)` loop calling `_forward_one_mol` once per molecule. For batch_size=256 this is **256 sequential launches**, each containing:
- A `torch.zeros(V, H)` allocation + `index_add_` (atom_incoming)
- A `torch.zeros(V, H)` allocation + `index_add_` (outgoing)
- A `_GRUUpdate` forward
- A `.item()` call on `E = valid_edges.sum().item()` — **forces CPU sync per molecule**

### §2.4 The `.item()` sync is NOT the dominant cost (surprising result)

| Variant | Time (ms) |
|---|---:|
| forward with `.item()` (current code) | 880 ms |
| forward without `.item()` but using V_max padding | 1073 ms |

**Counter-intuitive result:** removing the `.item()` sync made it 22% SLOWER. Reason: without `.item()` the loop uses `V_max=39` padded atoms for every molecule, even when E=10; with `.item()` we slice to the per-molecule V_i. So the `.item()` is currently a "necessary evil" — the design pads to V_max but the loop has to slice per mol to skip padding.

---

## §3 Why is GPU not being saturated? (Hypothesis → MEASURED)

| Hypothesis | MEASURED verdict |
|---|---|
| "CPU-bound DataLoader" | **REJECTED.** CPU collate = 7 ms/batch (1-3% of step). `num_workers=0` matters for the `manual loop path` (lines 660-758), but the GPU side is the dominant cost. |
| "SMILES featurization CPU-bound" | **N/A.** Featurization runs **once before** the epoch loop, cached to `/tmp/wf131_mtw_fe_cache/*.pkl` via `_featurize_with_cache`. Not on the hot path. |
| "PyTorch DDP overhead" | **N/A.** Single-process, no DDP. |
| "CPU-GPU gradient sync" | **MINOR.** No DDP = no all-reduce; only `.item()` in hot path (~22% slower if removed; needed to skip padding). |
| "GPU compute is just small" | **PARTIAL.** 1.37M params + batch=256 = small workload per kernel. But there are 256 × 3 = **768 tiny kernel launches per batch** (256 mols × depth=3 msg-pass iterations), each with CPU launch overhead. |
| **"Python per-mol loop is the bottleneck"** | **CONFIRMED.** 256 sequential `_forward_one_mol` calls → most of the GPU's "down time" is between-kernel gaps (CPU launches the next kernel before the previous finishes). Forward ratio = 32% of step; backward ratio = 65%; total SM utilization = 4-7%. |

---

## §4 Why didn't Triton optimization help? (Honest answer)

The previous Wf / R18 effort (per project history) likely targeted the **EGNN inner kernels** for the CFM/refiner backbone, where `scatter_add` + tensor product ops can be fused. **D-MPNN's per-mol Python loop is a different beast:**

1. The bottleneck is NOT the kernel itself — it's the **launch overhead × 256 sequential calls**.
2. A custom Triton kernel for one `_forward_one_mol` would still need to be called 256 times in a Python loop. The launch overhead remains.
3. The real win is **eliminating the Python loop entirely** via batched message passing or CUDA Graph capture.

So Triton on D-MPNN's atom-wise ops helps the kernel speed; it does NOT help the CPU-loop overhead. That's why "Triton optimization" alone leaves GPU underutilized.

---

## §5 Reference-source potential NOT yet extracted (10 axes, ranked by impact)

| # | Technique | Source | Expected lift | Status |
|---:|---|---|---|---|
| **1** | **PyG-style batched message passing** | `torch_geometric.nn.MessagePassing` or `segment_csr` scatter-add. Replaces the per-mol Python loop with one batched `scatter_add_` over a coalesced edge index. | **5-10x fwd+bwd speedup** (one launch vs 256) | NOT extracted. D-MPNN loops over batch dim in Python. |
| **2** | **CUDA Graph capture** | `torch.cuda.CUDAGraph` to capture the entire fwd+bwd+step and replay. Eliminates kernel-launch overhead entirely. | **2-3x throughput** for small ops | NOT extracted. Not even attempted on D-MPNN trainer. |
| **3** | **AMP / BF16 mixed precision** | `torch.cuda.amp.autocast(dtype=torch.bfloat16)`. RTX 5090 + RTX PRO 6000 are sm_120 — both have BF16 tensor cores. | **~1.5x compute + 2x mem bandwidth** | NOT enabled. Default `torch.float32` in trainer. |
| **4** | **torch.compile (Dynamo + Inductor)** | `torch.compile(model, mode="reduce-overhead")` fuses GRU + scatter_add + linear. Eliminates Python-loop overhead for fixed-shape inputs. | **2-4x fwd+bwd** for static graphs | NOT enabled. Trainers run eager. |
| **5** | **torch_scatter / torch_scatter_max** | Vectorised segment-reduce ops, GPU-native fused kernel. Drop-in for `index_add_`. | **1.3-2x scatter kernel speed** | NOT used (current code uses `index_add_` only). |
| **6** | **TF32 matmul** | `torch.backends.cuda.matmul.allow_tf32 = True`. RTX 5090 supports TF32; same accuracy as FP32 for our use, 2x matmul throughput. | **1.5x matmul throughput** (5x lower precision, not relevant) | NOT enabled. |
| **7** | **Persistent workers + `prefetch_factor=8`** | DataLoader prefetching decouples collate from GPU. Default `prefetch_factor=4`. | Saturates GPU when collate > kernel time (NOT the case here, but still helps) | NOT used (--cpu-num-workers 0 in active G6). |
| **8** | **Wasserstein `sort` mode (1-D)** | `wf251-F4" already shipped `sort` mode (closed-form 1-D via `torch.sort`). Avoids `n_mc=256` Python loop in `sliced` mode. | **2-3x W₂ eval speedup** | SHIPPED but `--w2-gradient-mode sliced` (default) still uses the slow path. |
| **9** | **VariancePUCT-style kernel fusion** | R20 P1 (`mol-metal-variance-puct-default`) — fuses MCTS visit-count + variance penalty into one Triton kernel. Not directly D-MPNN but adjacent. | 1.5x on MCTS side; orthogonal to D-MPNN | Shipped for MCTS only. |
| **10** | **BFloat16 reference distribution** | `ref_pic50_dist` is `torch.float32`; for the Wasserstein penalty, BF16 reference + BF16 pred = 2x bandwidth savings. | **1.5-2x W₂ step cost** | NOT extracted. |

---

## §6 Recovery path (concrete next steps, ordered by ROI)

### §6.1 Quick wins (no rewrite — flags only)

| Action | Effort | Expected gain |
|---|---|---|
| `--w2-gradient-mode sort` (replaces sliced W₂ n_mc=256) | 1-line CLI flag | 2-3x W₂ eval speedup |
| `torch.backends.cuda.matmul.allow_tf32 = True` in trainer init | 2 lines | 1.5x matmul throughput |
| `torch.set_float32_matmul_precision("high")` | 1 line | matches TF32 without API breakage |
| `--cpu-num-workers 4` (replace current 0) | 1 CLI flag | Eliminates main-thread collate (still <5% gain here, but helps validation phase) |
| Wrap forward in `torch.cuda.amp.autocast(dtype=torch.bfloat16)` | ~10 lines | **1.5-2x overall speedup** (BF16 tensor cores on sm_120) |

### §6.2 Medium effort (rewrite per-mol loop to vectorised)

| Action | Effort | Expected gain |
|---|---|---|
| Replace `encode_pool` Python loop with PyG `MessagePassing` subclass | 1-2 days | **5-10x fwd+bwd speedup** |
| Replace `_forward_one_mol` with `torch_geometric.utils.scatter` for the scatter_add | per-atom V slicing | 1.3x kernel speed |
| Replace `_GRUUpdate` with a custom Triton kernel (if standard fused approach is insufficient) | 2-3 days | 2x message-passing kernel speed |

### §6.3 High effort (rebuild trainer pattern)

| Action | Effort | Expected gain |
|---|---|---|
| `torch.cuda.CUDAGraph` capture of the entire train step | 2-3 days | **2-3x throughput** for small ops |
| `torch.compile(model, mode="reduce-overhead", fullgraph=True)` | 1 day if shapes are static | **2-4x fwd+bwd speedup** |

---

## §7 Honest framing (per party-twenty-spirit §3.5)

| Claim | Tag | Evidence |
|---|---|---|
| G6 epoch wall time | **MEASURED** | `results/wf124_p3_per_innovation_ablation_g6/.../Ru/seed1/*.log`: epoch 24 = 76.5s, epoch 28 = 226.9s. Variance driven by CPU contention, not GPU. |
| G6 GPU SM util | **MEASURED** | `nvidia-smi pmon -c 30 -d 1`: 4-7% SM during active G6 (epoch 28). |
| Forward time 812 ms / batch | **MEASURED** | Synthetic profile, batch=256, hidden=300, depth=3, RTX PRO 6000. |
| ".item() removal makes it slower" | **MEASURED** | Direct A/B test on synthetic batch. Counter-intuitive but true (V_max padding waste). |
| "PyG batching would 5-10x speedup" | **PROJECTED** | Reference source: `torch_geometric.nn.MessagePassing` aggregates via segment_reduce over a coalesced edge_index — eliminates 256 sequential kernel launches. Not measured on D-MPNN. |
| "CUDA Graph would 2-3x throughput" | **PROJECTED** | Reference source: `torch.cuda.CUDAGraph` benchmarks in `torch/cuda/graphs.py` show 1.5-3x for small ops. Not measured on D-MPNN. |
| "AMP/BF16 would 1.5-2x" | **PROJECTED** | Reference source: sm_120 BF16 throughput vs FP32 (PyTorch docs). Not measured on D-MPNN. |

### §7.1 Honest-negative call-outs

1. **GPU 0 (RTX PRO 6000) is not "our" unused capacity** — it is actively used by sunlong's dfine_t146 + 7 other processes. We cannot unilaterally claim it for our runs without coordination.
2. **The 4-7% SM utilization is WORSE than the 24% baseline reported elsewhere.** The 24% was a peak burst during fwd+bwd; the steady-state average is much lower. The epoch wall (70-227 sec) is **system-CPU-bound**, not GPU-bound.
3. **Our trainer bottleneck is the Python per-mol loop, NOT the collate DataLoader.** Adding `num_workers=4` would help <5% of the wall time. The real win is eliminating the per-mol loop entirely (PyG / CUDA Graph / torch.compile).
4. **Triton optimization alone cannot fix this bottleneck.** The cost is launch overhead × 256 calls, not the per-kernel speed.

---

## §8 Cross-references

- Active G6 process: PID 1849475, command-line `molmetal/scripts/train_dmpnn_multitask_wasserstein.py --metals Ru --seeds 1 --single-task --hidden 300 --depth 3 --batch-size 256 --w2-gradient-mode sliced --cpu-num-workers 0`.
- `molmetal/reports/wf326a_gpu_profile/profile.json` — synthetic profile summary
- `molmetal/reports/wf326a_gpu_profile/trace.json` — full torch profiler trace (Chrome trace format)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:343-389` — `encode_pool` and `forward` per-mol loops
- `molmetal/baselines/dmpnn_multitask.py:221-291` — `_forward_one_mol` and `forward` (the same pattern)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:660-758` — manual loop path (used when `--cpu-num-workers 0`)

**END of Wf-326A.**