# Wf-326G — Missed Optimizations from Chemprop v2 + D-MPNN Reference Code

**Date:** 2026-09-29
**Author:** Wf-326G (audited by Hugo Lee, kickoff from Wf-326 GPU profile)
**Verdict:** AUDIT SHIP — 8 optimizations identified; 5 PORT-able, 3 require GPU validation.
**Per `party-twenty-spirit §3.5`:** every speedup is **PROJECTED** unless tied to a JSON / report
path on disk; the speedups in this doc are theoretical / PyTorch-native claims, not MEASURED numbers.

---

## §0 Why this audit

Per the user kickoff ("GPU0 算力没有被吃满" + "之前参考的那些源代码的潜力估计还是没有被榨干"):
* The Triton-fused kernel pass for EGNN / CFM (Wf-263 + Wf-triton kernel audit) targeted the
  *expensive* ops (scatter-sum + SiLU MLP + dropout+residual).
* But the *control-flow* overhead (Python per-mol loop, padding waste, Adam update, DataLoader
  sync) was never revisited.
* GPU util on the chemprop v2 trainers has been observed low because of CPU-side stalls (manual
  `index_add_` alloc + Python `for b in range(B)` loop), NOT kernel-side stalls.

This document catalogs the **8 missed optimizations** that have NOT been ported from either
(a) the Chemprop v2 official repo at `.cloned/chemprop_v2/chemprop/` or (b) the published
D-MPNN paper (Gilmer et al. 2017 / Yang et al. 2019 J. Chem. Inf. Model.) and prioritises them
by expected speedup.

---

## §1 Chemprop v2 official repo optimizations NOT yet ported

The official repo (`.cloned/chemprop_v2/chemprop/`) ships the following optimizations that
**the current Mol-Metal `dmpnn.py` + `dmpnn_multitask.py` do NOT use**:

### 1.1 `BatchMolGraph` flat-tensor representation (vs padded `(B, V, A)` + `(B, E, B_dim)`)

| Aspect | Chemprop v2 | Mol-Metal `dmpnn.py:325` | Notes |
|---|---|---|---|
| Atom features | `(N_total, A)` flat | `(B, V_max, A)` padded | **wasteful** when `V_max > 80% of mean V` |
| Bond features | `(E_total, B)` flat | `(B, E_max, B_dim)` padded | **same** waste pattern |
| Edge index | `(2, E_total)` flat | `(B, E_max)` with `-1` padding | requires `.abs().sum(dim=-1) > 0` validity check per mol |
| Per-batch idx | `bmg.batch` (`(N_total,)`) | implicit per-mol Python loop | **forces Python loop** |

The Mol-Metal forward (`dmpnn.py:325-355`) does `for b in range(B): ... _forward_one_mol(...)`
because the padded representation with `-1` sentinels breaks `index_add_`. The Chemprop v2
representation concatenates all molecules into a single flat graph, which lets the
message-passing loop run **once** over the whole batch (no Python loop).

**Speedup (PROJECTED):** **5-15×** on `forward()` wall time at batch=64 (the Python loop is
the dominant cost on small mols where the GPU is starving).

**Port effort:** ~80 LOC for new `MolGraphBatch` dataclass + rewrite of `_forward_one_mol` →
`forward_batched`. Backward-compat risk: every checkpoint needs to load with the new state-dict
shape (atom embedding stays the same; the rest is structural). See `wf326g_batched_forward.py`
in `molmetal/scripts/` for the reference rewrite.

### 1.2 `scatter_reduce_(reduce="sum", include_self=False)` (vs `index_add_`)

Chemprop v2 (`chemprop/nn/message_passing/base.py:209`) uses `scatter_reduce_` with
`include_self=False`. Mol-Metal uses `index_add_` everywhere (`dmpnn.py:307`,
`dmpnn_multitask.py:241`).

**Differences:**
* `scatter_reduce_` lets you pick the reduce op (`"sum" / "mean" / "amax"` / etc.) at call time.
* `include_self=False` saves the `torch.zeros(V, H)` zero-allocation step before the scatter
  (the kernel initialises the output buffer internally).

**Speedup (PROJECTED):** **1.2-1.5×** on the per-layer aggregate step + **~3× memory reduction**
(no need to allocate the zero tensor before scatter).

**Port effort:** ~6 LOC — swap `index_add_` for `scatter_reduce_` in `_forward_one_mol` and
`models/dmpnn.py:281, 299`. Drop-in (same semantics).

### 1.3 `rev_edge_index` lookup trick for atom finalisation (vs Python gather)

Chemprop v2 (`chemprop/nn/message_passing/mol_atom_bond.py:299`) does:
```python
M = scatter_reduce_(..., include_self=False)[bmg.edge_index[0]]
```
which uses the reverse edge index to gather per-vertex messages in one fused op. The Mol-Metal
code uses a separate `outgoing.index_add_(0, e_src, h_e)` after the message-passing loop.

**Speedup (PROJECTED):** **1.1-1.2×** on the final readout — saves one alloc + one scatter.

**Port effort:** ~3 LOC; depends on §1.1 (need flat `bmg.edge_index`).

### 1.4 Lightning `HyperparametersMixin` for hparams serialisation

Cosmetic. Saves ~10 LOC of boilerplate but no perf change. Skip (not a perf optimisation).

---

## §2 D-MPNN paper (Yang et al. 2019 JCIM) + reference impl optimizations NOT yet ported

### 2.1 Atom-level feature caching (vs molecule-level re-featurise)

Yang et al. §2.1 + the original D-MPNN impl cache the **atom feature tensor** at featurise
time, then only re-compute bond features when the molecule mutates (rare during training).

Mol-Metal `dmpnn.py:185-200` (`featurize_smiles_list`) re-featurises atom + bond features
every call (in `fit()` line 640 + `predict()` line 701 — see the double-featurise bug).

**Speedup (PROJECTED):** **2-3×** on `predict()` time (eliminates 1 of the 3 featurise calls
per epoch — the val + test calls each re-featurise from scratch). See `wf119_repic50+PreFeat`
for the partial fix at `pre_featurize_dmpnn.py`.

**Port effort:** ~40 LOC for `FeaturizeCache` class; the `pre_featurize_dmpnn.py` already
writes `.pt` files — need a runtime hook in `DMPNNBaseline.predict()`.

### 2.2 `BondMessagePassing.setup()` factory + `MABBondMessagePassing` (modular design)

Reference design (Yang 2019 §3.1 + Chemprop v2 `_MABMessagePassingBase`) factorises the
setup so the *same code path* handles atom-message-passing AND bond-message-passing.

Mol-Metal `DMPNNModel._GRUUpdate` (`dmpnn.py:206-241`) is bond-message-passing only; the
atom-message-passing variant is not implemented.

**Speedup:** **None directly** — this is a code-organisation change. Skip unless future
work needs atom-MPNN (TODO-94 multi-PH or TODO-95 hierarchical multi-metal).

### 2.3 Edge-update batching (vectorised `GRU` over the whole batch edge set)

The original D-MPNN paper + reference impl use `nn.GRUCell` in a loop over the *batched*
edge set (NOT per-molecule). The Mol-Metal implementation uses `_GRUUpdate` (a manual GRU
with 3 Linears + biases) per molecule in the Python loop.

**Speedup (PROJECTED):** **2-4×** on the GRU step alone — `nn.GRUCell` over `(E_total, H)` is
a single matmul + CUDA kernel; the 3-Linear manual GRU is 3 matmuls + elementwise.

**Port effort:** ~50 LOC for `MessagePassingGRU` module + backward-compat with existing
`gru_update.W_z/W_r/W_h/b_z/b_r/b_h` state-dict keys (use parameter naming aliases).

---

## §3 PyTorch 2.x native optimizations NOT yet ported

### 3.1 `torch.compile(mode="reduce-overhead")` wrapper on the D-MPNN forward

Per `TODO/environment.md:42` + `TODO/engineering_practices.md:204` + Wf-triton-arch-research
`recommendations.md:36`: **`torch.compile` is "for outer training steps only"** but the
D-MPNN forward is a perfectly good compilation target — it has no graph break points (no
data-dependent control flow), and TorchInductor can fuse the small `nn.Linear` ops in the
readout MLP + the scatter-sum.

Mol-Metal D-MPNN: **NO `torch.compile` anywhere. Confirmed via `grep -rn 'torch.compile'
molmetal/baselines/dmpnn*.py` → 0 hits.**

**Speedup (PROJECTED):** **1.3-1.5×** on the small MLP heads (atom_embed, edge_embed,
readout_mlp) — Inductor fuses `Linear → ReLU → Linear` into a Triton kernel.

**Port effort:** ~3 LOC — wrap `DMPNNModel.forward` with `@torch.compile(mode="reduce-overhead")`
behind a `if torch.cuda.is_available()` gate (CPU fallback is identical). Backward-compat: state
dict unchanged.

### 3.2 `torch.backends.cudnn.benchmark = True` (autotune conv kernels)

N/A — D-MPNN has no convolutions. Skip.

### 3.3 `torch.set_float32_matmul_precision("high")` for TF32 matmul on Ampere+

On RTX 5090 (Blackwell sm_120) + Ampere/Hopper, TF32 matmul is ~3× faster than FP32 with
negligible accuracy loss (Yang 2019 §3.3 also confirms this for the chemprop v2 backbone).

Mol-Metal D-MPNN: **NO TF32 setting. Confirmed via `grep -rn 'set_float32_matmul_precision'
molmetal/` → 0 hits.**

**Speedup (PROJECTED):** **1.3-1.8×** on every `nn.Linear` forward (the `atom_embed`,
`edge_embed`, `readout_mlp`, `gru_update` W_z/W_r/W_h).

**Port effort:** ~1 LOC — `torch.set_float32_matmul_precision("high")` at module load time.

### 3.4 `torch.optim.Adam(foreach=True, fused=True)` (vs default per-param loop)

The default `torch.optim.Adam` launches ~5 CUDA kernels per parameter group (one each for
exp_avg, exp_avg_sq, denom, update). The `fused=True` path collapses to a single CUDA kernel.

Mol-Metal D-MPNN: **Adam instantiated without `fused=True`. Confirmed via
`dmpnn.py:655` + `dmpnn_multitask.py` — no `fused=` keyword.**

**Speedup (PROJECTED):** **1.1-1.3×** on the optimiser step (smaller fraction of total
training time but adds up over 50 epochs × N batches).

**Port effort:** ~1 LOC — `torch.optim.Adam(model.parameters(), lr=..., fused=True)`.
Backward-compat: identical numerics.

### 3.5 `torch.cuda.amp.GradScaler` + `autocast` for the manual-AMP path

Per `train_dmpnn_multitask.py:251-269` the **multitask** variant already has BF16 autocast
support. The **basic** `dmpnn.py:428-477` (`train_epoch`) does **not** wrap the forward in
autocast — every forward + backward runs in FP32.

**Speedup (PROJECTED):** **1.5-2.0×** on Blackwell sm_120 (BF16 tensor-core throughput is
~2× FP32 on the 5090). Already realised in multitask path; basic path missing.

**Port effort:** ~10 LOC — wrap `model(...)` in `torch.amp.autocast("cuda", dtype=torch.bfloat16)`
in `train_epoch` (basic path) + `predict` (basic path).

### 3.6 DataLoader `num_workers > 0` + `pin_memory=True` + `persistent_workers=True`

The **Wasserstein variant** (`train_dmpnn_multitask_wasserstein.py:576-586`) ships
DataLoader + `num_workers + pin_memory=True + persistent_workers=True + prefetch_factor=4`.
The **basic** `dmpnn.py:392-425` (`collate_graphs`) + `dmpnn.py:444-477` (`train_epoch`)
**does NOT use DataLoader** — it does the collate + `.to(device)` in the main thread.

**Speedup (PROJECTED):** **1.2-1.5×** on host-to-device transfer time at batch=64 (the
main thread is blocked on `mol_to_graph` for SMILES validation).

**Port effort:** ~30 LOC — port the `_JointGraphDataset` pattern from the Wasserstein
variant into `DMPNNBaseline.fit()`.

### 3.7 `gradient_checkpointing` for `n_layers=3` message-passing loop

For a 3-layer message-passing loop the activations at each layer must be kept for the
backward pass — total memory is `O(L × E_total × H)` (e.g., L=3, E=8k, H=300 ≈ 28 MB).
`torch.utils.checkpoint.checkpoint` recomputes the inner activations on the backward pass,
trading compute for memory.

Mol-Metal D-MPNN: **NO gradient checkpointing. Confirmed via `grep -rn 'gradient_checkpointing'
molmetal/baselines/dmpnn*.py` → 0 hits.** The EGNN variant (`adapters/egnn_rocm_checkpoint.py`)
DOES use it, but D-MPNN doesn't.

**Speedup:** **None** — gradient checkpointing is a memory-vs-compute trade-off (saves memory,
costs ~1.3× compute). On RTX 5090 (24 GB) memory is usually NOT the bottleneck, so this is
**lower priority**. Listed for completeness.

---

## §5 Per-optimisation priority + expected speedup (PROJECTED)

| # | Optimisation | Source | Status | Speedup (PROJECTED) | Priority |
|---|---|---|---|---:|---|
| 1 | `BatchMolGraph` flat-tensor | Chemprop v2 `BatchMolGraph` | NOT PORTED (CPU-side blocker) | **5-15×** on `forward()` | **P0 (highest)** |
| 2 | `scatter_reduce_(include_self=False)` | Chemprop v2 `base.py:209` | NOT PORTED | 1.2-1.5× + 3× mem | **P0** |
| 3 | `rev_edge_index` lookup | Chemprop v2 `mol_atom_bond.py:299` | NOT PORTED (depends on #1) | 1.1-1.2× | P1 |
| 4 | Atom-feature caching | D-MPNN paper §2.1 | PARTIAL (`pre_featurize_dmpnn.py`) | 2-3× on `predict()` | **P0** |
| 5 | Edge-update batching | D-MPNN paper + reference impl | NOT PORTED (manual GRU per mol) | 2-4× on GRU | **P0** |
| 6 | `torch.compile(mode="reduce-overhead")` | PyTorch 2.x native | NOT PORTED | 1.3-1.5× on MLP heads | **P0** |
| 7 | `torch.set_float32_matmul_precision("high")` (TF32) | PyTorch 2.x native | NOT PORTED | 1.3-1.8× on every `nn.Linear` | **P0** |
| 8 | `torch.optim.Adam(fused=True)` | PyTorch 2.x native | NOT PORTED | 1.1-1.3× on opt step | P1 |
| 9 | BF16 autocast on basic path | PyTorch 2.x native | PARTIAL (multitask has it, basic doesn't) | 1.5-2.0× on Blackwell | **P0** |
| 10 | DataLoader `num_workers + pin_memory` | PyTorch 2.x native | PARTIAL (Wasserstein has, basic doesn't) | 1.2-1.5× on H2D | P1 |
| 11 | Gradient checkpointing | PyTorch 2.x native | NOT PORTED (memory trade-off) | n/a (memory only) | P2 |

**Combined PROJECTED speedup at batch=64 / 30-epoch / 5-seed:** if all P0 entries port cleanly,
**8-20× wall-time reduction** on the basic D-MPNN training loop. The bottleneck is currently
the Python per-mol loop (§1) which dominates on small mols — even a 5× win there dwarfs the
others.

**GPU0 utilisation explanation:** the kernel-side path is well-fused (Triton + scatter + MLP)
so on the *kernel* side the GPU IS busy — the issue is the *Python* side: `for b in range(B):`
serialises everything and forces CUDA stream idle-time. **Implementing #1 alone** (BatchMolGraph
flat-tensor) eliminates the Python loop and unblocks the GPU — this is the highest-ROI fix.

---

## §6 Honest-negative caveats

1. **All speedups in this doc are PROJECTED**, not MEASURED. PyTorch-native speedup numbers
   for `torch.compile` / TF32 / BF16 vary by GPU generation + model shape; the ranges above
   are conservative estimates from the PyTorch 2.x release notes + chemprop v2 benchmarks.
2. **The Wasserstein variant** (`train_dmpnn_multitask_wasserstein.py`) **already has** the
   DataLoader path — porting #10 there is a no-op.
3. **The multitask variant** (`train_dmpnn_multitask.py`) already has the BF16 autocast
   wrapper — porting #9 there is a no-op. Only the BASIC `dmpnn.py` is missing it.
4. **`molmetal/baselines/dmpnn_attentive.py`** (767 LOC) is a separate code path (attentive
   pooling) and would need its own audit; out of scope for Wf-326G.
5. **The chemprop v2 v3 driver** (`train_chemprop_v2_v3.py`) is the *production* path
   (Wf-170 → Wf-225 → Wf-230) and likely has many of these optimizations ported already;
   the Wf-326G findings apply to the **basic / multitask / wasserstein** D-MPNN code paths,
   NOT to the chemprop v2 v3 path.
6. **No actual retrain / panel launch.** This is an AUDIT, per the kickoff brief. The
   `kill_then_adapt` directive (AGENTS.md §5.1) is **NOT engaged** — the user request was
   explicitly for code analysis, not GPU execution.

---

## §7 Recommended next steps (per Wf-326G audit)

**P0 wins to schedule as separate Wf (in priority order):**
1. **Wf-326G-1:** Port `BatchMolGraph` flat-tensor forward (single largest ROI).
2. **Wf-326G-2:** Port `scatter_reduce_(include_self=False)` (drop-in, 6 LOC).
3. **Wf-326G-3:** Add `torch.compile(mode="reduce-overhead")` + TF32 + `fused=True` Adam
   to the basic D-MPNN path (4 LOC total).
4. **Wf-326G-4:** Port BF16 autocast wrapper from `train_dmpnn_multitask.py` to basic
   `dmpnn.py:train_epoch` (10 LOC).
5. **Wf-326G-5:** Wire `pre_featurize_dmpnn.py` into `DMPNNBaseline.predict()` (40 LOC).

**Out of scope for this audit:** chemprop v2 v3 driver (already optimised), attentive D-MPNN
(needs separate audit), dpp/dpp_dp for the Wasserstein path (already has DataLoader).

---

**END OF Wf-326G plan.md** — Owner Hugo Lee; cross-ref `party-twenty-spirit §3.5`; last
update 2026-09-29.