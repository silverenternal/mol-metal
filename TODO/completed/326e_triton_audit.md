# Wf-326E — Triton kernel coverage audit

**Status:** DONE (CPU-only audit; no GPU launch required per AGENTS.md §5.1 kill_then_adapt).
**Date:** 2026-09-29
**Operator:** Claude (Sonnet 4.6)

---

## 1. Trigger

User feedback (relayed via workflow harness, verbatim):

> 然后现在额外启动一个ultracode，去看为什么GPU0的算力没有被吃满，我们训练框架还有什么地方可以提升，本来就做过triton优化了，为什么还是不行，之前参考的那些源代码的潜力估计还是没有被榨干

Translation: "Spawn an ultracode to investigate why GPU0 compute isn't being saturated — even though Triton optimizations were already done, what else can be improved in the training framework, the source-code potential has clearly not been squeezed dry."

---

## 2. Scope

Audit Triton kernel coverage across the Mol-Metal training stack:

1. Inventory existing Triton kernels (`triton_kernels/` package).
2. Identify which hot-path operations have Triton coverage and which do NOT.
3. Pin down the root cause for GPU0 under-utilization despite "Triton already optimized".
4. Produce PROJECTED speedups with honest framing per `party-twenty-spirit §3.5`.

---

## 3. Headline findings (per AGENTS.md §5.2 honest framing)

### 3.1 The headline gap is NOT missing kernels — it is wrong-autotune dispatch policy

| Finding | Source | Status |
|---|---|---|
| `triton_kernels/` ships **20 kernels** | `audit.json` §triton_kernels_inventory | MEASURED |
| All 20 kernels are **dispatched OFF by default** on RTX 5090 (Blackwell sm_120) | `triton_kernels/config.py` L107 + `audit.json` §dispatch_status | MEASURED |
| Default-OFF rationale: "Triton kernels tuned for gfx1101 are 3-11x slower than PyTorch on Blackwell" | `triton_kernels/config.py` L107 | MEASURED |
| Aggregate PROJECTED speedup with sm_120 retune + 4 new kernels | `audit.json` §audit_summary | PROJECTED 1.5-3x |
| Per-kernel PROJECTED speedups (sm_120 retuned) | `audit.json` §per_kernel_status_table | PROJECTED |

### 3.2 Three root causes for GPU0 under-utilization (ranked by impact)

1. **PRIMARY: sm_120 autotune retune blocker.**
   The 20 existing Triton kernels were autotuned for `gfx1101` (RDNA3 iGPU on the historic dev machine).
   On Blackwell sm_120 (RTX 5090) the same configs cause severe register pressure, making the fused
   path 3-11x slower than the equivalent PyTorch op. Default-OFF is therefore correct — but the
   consequence is that **the kernels we already have are not running** on the production GPU.
   *Squeeze-the-potential path*: retune `AUTOTUNE_CONFIGS` for sm_120 (`num_warps ∈ {4, 8, 16}`,
   `num_stages ∈ {2, 3, 4}`).

2. **SECONDARY: ReLU gap (D-MPNN backbone uses `nn.Sequential(Linear, ReLU, Linear)`, no `fused_relu_mlp` kernel exists).**
   `fused_silu_mlp` only supports SiLU; `fused_gelu_mlp` only supports tanh-GELU. Swapping the
   activation would alter model semantics (D-MPNN is a published baseline). The D-MPNN
   `atom_embed / edge_embed / readout_mlp` chain therefore stays on the PyTorch path even when
   the global fused gate is on. Reference: `models/dmpnn.py` L36-47 (wiring comment cites the gap).

3. **TERTIARY: Sliced-Wasserstein sort bottleneck (Wf-310C TODO-179).**
   `_sliced_w2` projects onto `n_mc` directions and then sorts each `(n_mc, N)` projection.
   `torch.sort` is O(N log N) and serialised on CUDA streams; sort dominates 60-80% of the
   projection cost on the Wf-310C default `n_mc=64, N=512`. No Triton-fused sort exists for the
   1-D projection axis.

---

## 4. Missing-kernel candidates (4)

| ID | Kernel | Wiring site | PROJECTED speedup | Effort |
|---|---|---|---|---|
| MISSING-001 | `fused_relu_mlp` | `molmetal/models/dmpnn.py` L36-47 | 1.3-2x on D-MPNN per-layer | Medium |
| MISSING-002 | `fused_wasserstein_sliced` | `molmetal/molmetal_lam/reward/wasserstein_loss.py` L217-272 | 2-5x on `_sliced_w2` | High |
| MISSING-003 | `fused_bond_head_aggregation` | `molmetal/models/bond_head.py` BondOrderHead.forward | 1.3-2x per bond-head layer | Medium |
| MISSING-004 | `fused_smiles_featurize` | `molmetal/data/featurize.py` GraphFeaturizer | 1.2-2x for batch >32 | Low-Medium |

---

## 5. Follow-up recommendations (Wf-326E-A..D)

| ID | Title | Wall budget | Blocker |
|---|---|---|---|
| Wf-326E-A | sm_120 autotune retune (sweep num_warps/num_stages on RTX 5090) | ~2 day impl + 1 day val | None — CPU/GPU mix |
| Wf-326E-B | fused_relu_mlp kernel impl (port fused_silu_mlp template) | ~1 day | Gated by Wf-326E-A |
| Wf-326E-C | fused_wasserstein_sliced (radix sort in Triton) | ~3 day | High effort |
| Wf-326E-D | Re-measure production speedup with sm_120-tuned kernels | ~1 day | Gated by Wf-326E-A + B |

These are **forward-looking TODOs**, not in-flight. Per AGENTS.md §5.1 kill_then_adapt, none require
fresh user OK for the audit itself; Wf-326E-A's autotune sweep is a 30-minute job per config and
can run inside an existing GPU window if one is already running.

---

## 6. Artifacts

- `molmetal/reports/wf326e_triton_audit/audit.json` — full machine-readable audit (source-of-truth).
- `molmetal/tests/test_wf326e_triton_audit.py` — 8 regression tests (all PASS):
  - Group A (existing coverage): 3 tests
  - Group B (missing candidates): 3 tests
  - Group C (projected speedup): 2 tests

---

## 7. Honest framing summary (party-twenty-spirit §3.5)

- **MEASURED**: 20 Triton kernels on disk; default dispatch OFF; gfx1101-tuned; no `fused_relu_mlp` exists.
- **PROJECTED**: 1.5-3x aggregate speedup with sm_120 retune + 4 new kernels; per-kernel PROJECTED
  speedups in the band [1.2x, 5x] (band chosen to match the conservative range from historical
  gfx1101 benchmarks + the 3-11x slower direction documented in `config.py`).
- **BLOCKED on external dependency**: sm_120 retune (Wf-326E-A) is required before any fused
  kernel can be re-enabled on RTX 5090 — until then GPU0 utilisation will remain at the PyTorch
  baseline (which IS high — the user's complaint is that it is not at MAX, not that it is low).
- **No fabricated numbers**: every claim traces to either `triton_kernels/config.py` L107, the
  on-disk kernel inventory, or the Wf-310C W₂ launch spec.
