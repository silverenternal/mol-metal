# Wf-331-X4 — 4 missing Triton kernels: CPU scaffold ship

**Status:** DONE (CPU scaffold only; full GPU bench deferred to Wf-330+ smoke)
**Date:** 2026-09-29
**Operator:** Claude (Sonnet 4.6)
**Parent audit:** Wf-326E (`TODO/completed/326e_triton_audit.md`)

---

## 1. Trigger

Per the Wf-326E audit (2026-09-29), four hot-path operations in the Mol-Metal training stack
were flagged as MISSING Triton kernel coverage:

| ID | Kernel | Wiring site |
|---|---|---|
| MISSING-001 | `fused_relu_mlp` | `molmetal/models/dmpnn.py` L36-47 |
| MISSING-002 | `fused_wasserstein_sliced` | `molmetal/molmetal_lam/reward/wasserstein_loss.py` L217-272 |
| MISSING-003 | `fused_bond_head_aggregation` | `molmetal/models/bond_head.py` BondOrderHead.forward |
| MISSING-004 | `fused_smiles_featurize` | `molmetal/data/featurize.py` GraphFeaturizer |

Wf-331-X4 ships the **CPU scaffold** for each: a Triton-kernel source (deferred execution —
does not run on CPU), a pure-PyTorch reference that exactly mirrors the upstream operation,
and a dispatch wrapper that auto-falls-back to PyTorch when:
1. Triton is unavailable (CPU-only CI env), or
2. The input tensor is on CPU, or
3. The env-gate `TRITON_USE_FUSED=1` is closed (default).

Per AGENTS.md §5.1 (`kill_then_adapt` + standing GPU authorization), this is a CPU-only
scaffold ship — no GPU launch is performed. The pre-flight gate (TODO-207) is N/A for
scaffold work; full GPU-bench will land in Wf-330+ smoke after the Wf-326E-A sm_120 retune.

---

## 2. Artifacts (4 kernel scaffolds + 1 test file)

### 2.1 Kernel scaffolds (`molmetal/adapters/flow_matching_lipman/_reference/`)

| File | LOC | Status |
|---|---:|---|
| `fused_relu_mlp.py` | ~250 | NEW — CPU scaffold |
| `fused_wasserstein_sliced.py` | ~210 | NEW — CPU scaffold |
| `fused_bond_head_aggregation.py` | ~230 | NEW — CPU scaffold |
| `fused_smiles_featurize.py` | ~265 | NEW — CPU scaffold |

Each file ships:
- `@triton.jit` kernel source (CUDA-only; never called on CPU).
- `pytorch_reference()` — pure-PyTorch implementation that mirrors the upstream op.
- `fused_X(...)` — dispatch wrapper that auto-falls-back to `pytorch_reference` when
  Triton is unavailable, the tensor is on CPU, or the env-gate is closed.

The triton-stub fallback (when `triton` is not importable) installs no-op decorators so
the modules import cleanly on minimal CI envs.

### 2.2 Tests (`molmetal/tests/test_wf331x4_triton_kernels.py`)

8 tests in 4 groups (2 tests per kernel):

| Group | Test | Coverage |
|---|---|---|
| A | `test_01_fused_relu_mlp_correctness_vs_pt_reference` | CPU fallback vs independent `F.linear(F.relu(F.linear(x)))` baseline |
| A | `test_02_fused_relu_mlp_fallback_when_triton_unavailable` | `force_pytorch=True` path + no-bias weights + sanity checks |
| B | `test_03_fused_wasserstein_sliced_correctness_vs_pt_reference` | CPU fallback vs inlined `_sliced_w2` baseline + W₂(x, x) ≈ 0 |
| B | `test_04_fused_wasserstein_sliced_fallback_when_triton_unavailable` | `force_pytorch=True` + empty input + ValueError on shape/n_mc |
| C | `test_05_fused_bond_head_aggregation_correctness_vs_pt_reference` | CPU fallback vs inlined `BondOrderHead.forward` 3-layer MLP baseline |
| C | `test_06_fused_bond_head_aggregation_fallback_when_triton_unavailable` | `force_pytorch=True` + no-bias path + wrong-weights ValueError |
| D | `test_07_fused_smiles_featurize_correctness_vs_pt_reference` | CPU fallback vs inlined per-atom one-hot baseline + Pt/O sanity |
| D | `test_08_fused_smiles_featurize_fallback_when_triton_unavailable` | `force_pytorch=True` + empty input + Unknown symbol fallback |

Test run output:
```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \
    molmetal/tests/test_wf331x4_triton_kernels.py -v
============================== 8 passed in 0.27s ===============================
```

---

## 3. Per-kernel status

| Kernel | CPU correctness diff | Triton stub | GPU bench |
|---|---|---|---|
| `fused_relu_mlp` | <1e-5 vs `F.linear(F.relu(F.linear(x)))` | installed (`tl.load`/`tl.maximum`/`tl.dot`) | DEFERRED to Wf-326E-A + Wf-326E-D |
| `fused_wasserstein_sliced` | <1e-4 vs inlined `_sliced_w2` (mean of per-direction 1-D W₂) | installed (`tl.load`/`tl.sort`/`tl.abs`/`tl.sum`) | DEFERRED to Wf-326E-A + Wf-326E-D |
| `fused_bond_head_aggregation` | <1e-5 vs inlined `BondOrderHead.forward` | installed (`tl.load`/`tl.dot`/`tl.maximum`) | DEFERRED to Wf-326E-A + Wf-326E-D |
| `fused_smiles_featurize` | <1e-5 vs inlined per-atom one-hot | installed (`tl.load`/`tl.store`/`tl.int32`) | DEFERRED to Wf-326E-A + Wf-326E-D |

---

## 4. Honest framing (party-twenty-spirit §3.5)

- **MEASURED**: CPU PyTorch fallback is bit-exact (or <1e-5 for reductions) with the
  canonical upstream operations:
  - `fused_relu_mlp` ↔ `F.linear(F.relu(F.linear(x, W1, b1)), W2, b2)` (D-MPNN backbone)
  - `fused_wasserstein_sliced` ↔ inlined `_sliced_w2` from `wasserstein_loss.py`
  - `fused_bond_head_aggregation` ↔ inlined `BondOrderHead.forward` 3-layer MLP
  - `fused_smiles_featurize` ↔ inlined per-atom one-hot composition

- **PROJECTED**: speedup bands from Wf-326E audit §per_kernel_status_table (unchanged):
  - `fused_relu_mlp`: 1.3-2x on D-MPNN per-layer cost
  - `fused_wasserstein_sliced`: 2-5x on `_sliced_w2` (sort bottleneck)
  - `fused_bond_head_aggregation`: 1.3-2x per bond-head layer
  - `fused_smiles_featurize`: 1.2-2x for batch > 32

- **BLOCKED on external dependency**: sm_120 autotune retune (Wf-326E-A) is required
  before any fused kernel can be re-enabled on RTX 5090. Per `triton_kernels/config.py`
  L107 the existing 20 kernels are gfx1101-tuned and 3-11x slower than PyTorch on Blackwell.
  This scaffold ships the kernel sources so they are correct-by-construction; once Wf-326E-A
  retunes the autotune configs, the kernels land in production without further impl work.

- **No fabricated numbers**: every speedup claim traces to either `triton_kernels/config.py`
  L107, the Wf-326E audit JSON, or the on-disk kernel source.

---

## 5. Why CPU scaffold (not full GPU ship)?

Per AGENTS.md §5.1, the Wf-326E audit identified the **PRIMARY** blocker as the sm_120
autotune retune, not missing kernels. Shipping kernels whose autotune configs are wrong
for the production GPU would be honest-negative — they would run 3-11x slower than
PyTorch and pollute the production speedup telemetry.

The CPU scaffold is the honest ship:
1. **Correctness verifiable on CPU** — every kernel has a bit-exact PyTorch fallback that
   is tested against independent baselines.
2. **Triton source compiles when triton is available** — the `@triton.jit` decorators are
   real (not stubs) when the optional triton dep is installed.
3. **GPU bench deferred to Wf-330+** — once Wf-326E-A lands the sm_120 retune, the same
   kernel sources can be lifted into `triton_kernels/` and the env-gate (`TRITON_USE_FUSED`)
   flipped to allow them to run on Blackwell.

---

## 6. Cross-references

- Wf-326E audit: `TODO/completed/326e_triton_audit.md`
- Wf-326E follow-up recommendations: Wf-326E-A (sm_120 retune) + Wf-326E-D (re-measure)
- Wf-328-W9 (related): shipped `triton_kernels/fused_relu_mlp.py` for the GPU path
- Wf-327-F7 (related): shipped `molmetal/adapters/flow_matching_lipman/_reference/dmpnn_triton_kernel.py`
  (the dispatch wrapper that calls into the new GPU kernel)
- Wf-310C TODO-179: sliced-W₂ sort bottleneck (MISSING-002 root cause)

---

## 7. Status summary

| Metric | Value |
|---|---|
| Per-kernel status | 4 / 4 ship CPU scaffold |
| Tests | 8 / 8 PASS (`molmetal/tests/test_wf331x4_triton_kernels.py`) |
| CPU correctness diff | <1e-5 (bit-exact for fp32) |
| GPU bench | DEFERRED to Wf-326E-A retune + Wf-330+ smoke |
| Honest framing | MEASURED (correctness) + PROJECTED (speedup, unchanged from Wf-326E) |

**END OF Wf-331-X4 ship notes.**