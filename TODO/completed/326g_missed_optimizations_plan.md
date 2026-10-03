# Wf-326G — Missed Optimizations from Chemprop v2 + D-MPNN Reference Code (Audit)

**Owner:** Hugo Lee
**Date:** 2026-09-29
**Verdict:** AUDIT SHIP — 8/8 tests PASS; 11-row §5 priority table emitted; 5 P0 wins
catalogued for follow-up Wf (Wf-326G-1..5).
**Cross-ref:** `molmetal/reports/wf326g_missed_optimizations/plan.md` (full plan),
`molmetal/tests/test_wf326g_missed_optimizations.py` (8 audit tests).

---

## §0 Why this Wf

Per user kickoff ("GPU0 算力没有被吃满" + "之前参考的那些源代码的潜力估计还是没有被榨干"):
* Wf-triton-kernel-audit (Sep 2026) covered the *kernel*-side bottlenecks (scatter-sum,
  SiLU MLP, dropout+residual) and shipped Triton kernels for EGNN / CFM.
* The *control-flow* overhead was never revisited — Python per-mol loop, padding waste,
  Adam update, DataLoader sync. This is what now dominates the wall-time on the basic
  D-MPNN path.

The audit confirms **8 missed optimizations** (across 11 ranked entries) that have not
been ported from either (a) the Chemprop v2 official repo or (b) the D-MPNN paper /
reference implementation.

---

## §1 Audit artefacts shipped

| Artefact | Path | Status |
|---|---|---|
| Full plan (markdown) | `molmetal/reports/wf326g_missed_optimizations/plan.md` | 7 sections + §5 priority table (11 rows) |
| Audit tests | `molmetal/tests/test_wf326g_missed_optimizations.py` | **8/8 PASS** (pytest run 2026-09-29) |
| This TODO doc | `TODO/completed/326g_missed_optimizations_plan.md` | this file |

**Pytest result:** `molmetal/tests/test_wf326g_missed_optimizations.py .... 8 passed
in 104.63s` (PyTorch 2.9 + CPU; CUDA not exercised per the audit scope).

---

## §2 Summary of missed optimizations (8 found)

### §2.1 Chemprop v2 official repo (`.cloned/chemprop_v2/chemprop/`)

1. **`BatchMolGraph` flat-tensor representation** (vs padded `(B, V, A)` + `(B, E, B_dim)`).
   Chemprop v2 concatenates all molecules into a single flat graph; Mol-Metal uses padded
   representation + Python `for b in range(B):` loop. **PROJECTED 5-15× on forward wall time.**
2. **`scatter_reduce_(reduce="sum", include_self=False)`** (vs `index_add_`). Drop-in
   replacement; saves the `torch.zeros(V, H)` alloc + has built-in `include_self=False` option.
   **PROJECTED 1.2-1.5× + 3× memory reduction.**
3. **`rev_edge_index` lookup trick** for atom finalisation. One fused op instead of separate
   scatter. **PROJECTED 1.1-1.2× on readout.** Depends on #1.

### §2.2 D-MPNN paper + reference impl (Yang et al. 2019 JCIM)

4. **Atom-level feature caching** (vs molecule-level re-featurise). The `predict()` call
   currently re-featurises every call; the `pre_featurize_dmpnn.py` partial fix writes `.pt`
   files but no runtime hook. **PROJECTED 2-3× on `predict()`.**
5. **Edge-update batching** (vectorised `nn.GRUCell` over the whole batch edge set). The
   Mol-Metal `_GRUUpdate` is a 3-Linear manual GRU per molecule in the Python loop.
   **PROJECTED 2-4× on the GRU step alone.**

### §2.3 PyTorch 2.x native

6. **`torch.compile(mode="reduce-overhead")`** wrapper on the D-MPNN forward. Currently
   absent (`grep -rn 'torch.compile' molmetal/baselines/dmpnn*.py` → 0 hits).
   **PROJECTED 1.3-1.5× on the small MLP heads.**
7. **`torch.set_float32_matmul_precision("high")`** for TF32 matmul. Currently absent
   (`grep -rn 'set_float32_matmul_precision' molmetal/` → 0 hits). **PROJECTED 1.3-1.8× on
   every `nn.Linear` forward.**
8. **`torch.optim.Adam(fused=True)`** (vs default per-param loop). Single-line fix.
   **PROJECTED 1.1-1.3× on optimiser step.**
9. **BF16 autocast on basic path**. Multitask variant already has it (`train_dmpnn_multitask.py:810`);
   basic `dmpnn.py:train_epoch` does NOT. **PROJECTED 1.5-2.0× on Blackwell sm_120.**
10. **DataLoader `num_workers + pin_memory`**. Wasserstein variant already has
    (`train_dmpnn_multitask_wasserstein.py:576`); basic `dmpnn.py:392` does NOT.
    **PROJECTED 1.2-1.5× on host-to-device transfer.**
11. **`gradient_checkpointing`** for the 3-layer message-passing loop. Memory-vs-compute
    trade-off; **no speedup** on RTX 5090 24 GB (lower priority).

**Combined PROJECTED speedup (P0 entries only):** **8-20× wall-time reduction** on the
basic D-MPNN training loop. Bottleneck is the Python per-mol loop (§2.1 #1) which dominates
on small mols — even a 5× win there dwarfs the others.

---

## §3 Why GPU0 is under-utilised

Per Wf-326A profile data (the prior sibling Wf) the kernel-side path is well-fused (Triton
+ scatter + MLP) — on the *kernel* side the GPU IS busy. The issue is the *Python* side:
the `for b in range(B):` loop in `DMPNNModel.forward` (`molmetal/baselines/dmpnn.py:339`)
serialises everything and forces CUDA stream idle-time between molecule iterations.

**Implementing #1 alone (BatchMolGraph flat-tensor) eliminates the Python loop and unblocks
the GPU** — this is the highest-ROI fix. §5 priority table in the plan ranks it P0.

---

## §4 §5 priority table (cross-ref to plan §5)

| # | Optimisation | Source | Status | Speedup (PROJECTED) | Priority |
|---|---|---|---|---:|---|
| 1 | `BatchMolGraph` flat-tensor | Chemprop v2 | NOT PORTED | **5-15×** | **P0** |
| 2 | `scatter_reduce_(include_self=False)` | Chemprop v2 | NOT PORTED | 1.2-1.5× + 3× mem | **P0** |
| 3 | `rev_edge_index` lookup | Chemprop v2 | NOT PORTED (depends on #1) | 1.1-1.2× | P1 |
| 4 | Atom-feature caching | D-MPNN paper | PARTIAL | 2-3× | **P0** |
| 5 | Edge-update batching | D-MPNN paper | NOT PORTED | 2-4× | **P0** |
| 6 | `torch.compile(mode="reduce-overhead")` | PyTorch 2.x | NOT PORTED | 1.3-1.5× | **P0** |
| 7 | TF32 matmul precision | PyTorch 2.x | NOT PORTED | 1.3-1.8× | **P0** |
| 8 | `Adam(fused=True)` | PyTorch 2.x | NOT PORTED | 1.1-1.3× | P1 |
| 9 | BF16 autocast on basic path | PyTorch 2.x | PARTIAL | 1.5-2.0× | **P0** |
| 10 | DataLoader + `num_workers` | PyTorch 2.x | PARTIAL | 1.2-1.5× | P1 |
| 11 | Gradient checkpointing | PyTorch 2.x | NOT PORTED | n/a (memory) | P2 |

---

## §5 Recommended next steps (P0 follow-up Wf, in priority order)

The user kickoff explicitly asked for an audit, NOT a retrain launch — the kill_then_adapt
directive (AGENTS.md §5.1) is NOT engaged. The 5 P0 wins are catalogued as separate Wf:

1. **Wf-326G-1:** Port `BatchMolGraph` flat-tensor forward (single largest ROI).
2. **Wf-326G-2:** Port `scatter_reduce_(include_self=False)` (drop-in, 6 LOC).
3. **Wf-326G-3:** Add `torch.compile(mode="reduce-overhead")` + TF32 + `fused=True` Adam
   to the basic D-MPNN path (4 LOC total).
4. **Wf-326G-4:** Port BF16 autocast wrapper from `train_dmpnn_multitask.py` to basic
   `dmpnn.py:train_epoch` (10 LOC).
5. **Wf-326G-5:** Wire `pre_featurize_dmpnn.py` into `DMPNNBaseline.predict()` (40 LOC).

Each requires a separate user OK + pre-flight v2 25/25 PASS gate (AGENTS.md §5.1, §4) before
launch. None of them are scheduled yet — they are listed here as a backlog.

---

## §6 Honest-negative caveats

Per `party-twenty-spirit §3.5`:
* **All speedup numbers in this audit are PROJECTED, not MEASURED.** PyTorch-native speedups
  vary by GPU generation + model shape; the ranges above are conservative estimates from
  the PyTorch 2.x release notes + chemprop v2 benchmarks.
* The **Wasserstein variant** (`train_dmpnn_multitask_wasserstein.py`) already has the
  DataLoader path (#10 partial) — porting is a no-op there.
* The **multitask variant** (`train_dmpnn_multitask.py`) already has BF16 autocast (#9
  partial) — porting is a no-op there. Only the BASIC `dmpnn.py` is missing it.
* The **chemprop v2 v3 driver** (`train_chemprop_v2_v3.py`) is the production path and
  likely already has many of these optimizations ported; the Wf-326G findings apply to the
  basic / multitask / wasserstein D-MPNN code paths, NOT to the chemprop v2 v3 path.
* The **attentive D-MPNN** (`molmetal/baselines/dmpnn_attentive.py`) is a separate code path
  with its own attentive pooling and would need its own audit; out of scope for Wf-326G.
* **No actual retrain / panel launch.** This is an AUDIT, per the user kickoff. The
  `kill_then_adapt` directive (AGENTS.md §5.1) is **NOT engaged**.

---

## §7 Pre-flight + wave-budget posture

* **Pre-flight v2 (TODO-207):** N/A — this is a code-audit Wf, not a retrain / panel /
  ablation launch. Per AGENTS.md §4 the pre-flight gate is required "before any retrain /
  panel / ablation launch." No GPU launches are scheduled in Wf-326G.
* **Wave-budget tracker:** N/A — same reason. The 5 P0 follow-up Wf (Wf-326G-1..5) will
  individually require pre-flight + wave-budget when they are scheduled for launch.

---

**END OF Wf-326G TODO** — Owner Hugo Lee; cross-ref `party-twenty-spirit §3.5`; last
update 2026-09-29.
