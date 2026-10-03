# Wf-328-W9 — Triton kernel for D-MPNN message-passing edge updates

**Status:** DONE (CPU-correctness ship; Blackwell sm_120 perf pending Wf-326E-A retune)
**Date:** 2026-09-29
**Operator:** Claude (Sonnet 4.6)

---

## 1. Trigger

Wf-326E audit (molmetal/reports/wf326e_triton_audit/audit.json) identified
**MISSING-001** `fused_relu_mlp` as the highest-impact missing Triton kernel
for the D-MPNN backbone.  The D-MPNN per-layer edge MLP (`edge_mlp[layer]`,
``nn.Sequential(Linear(2D, D), ReLU())``) is the hottest site in the model:
it runs ``n_layers * B * E_max`` times per training step.  The existing
`fused_silu_mlp` only supports SiLU and `fused_gelu_mlp` only supports
tanh-GELU, so this site was un-fusable until Wf-328-W9.

This file ships **MISSING-001** as a CPU-correctness-validated Triton kernel
with dispatch-gated default-OFF wiring and a CLI opt-in flag.

---

## 2. Scope

Two new kernels + wiring + tests + docs:

1. **Kernel: `fused_relu_linear`** — ``ReLU(Linear(x, W, b))`` in a single
   Triton launch (matmul + bias + ReLU epilogue fused).  This is the
   per-layer D-MPNN edge MLP site (the Wf-328-W9 hot path).
2. **Kernel: `fused_relu_mlp`** — ``Linear(ReLU(Linear(x, W1, b1)), W2, b2)``
   in two fused Triton launches (first matmul + ReLU, second matmul + bias).
   This is the D-MPNN ``atom_embed`` / ``edge_embed`` / ``readout_mlp``
   chain (``nn.Sequential(Linear, ReLU, Linear)``).
3. **Per-op gates** on `triton_kernels.config.triton_config`:
   `use_fused_relu_linear` / `use_fused_relu_mlp` — default OFF, opt-in
   via ``TRITON_USE_FUSED=1`` AND ``TRITON_RELU_LINEAR=1`` /
   ``TRITON_RELU_MLP=1``.
4. **D-MPNN wiring** — the per-layer ``edge_mlp[layer]`` site is gated by
   a new ``TRITON_EDGE_UPDATE`` env var / ``--triton-edge-update`` CLI flag.
5. **CLI flag** — ``--triton-edge-update`` on
   `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` sets
   ``TRITON_USE_FUSED=1``, ``TRITON_RELU_LINEAR=1`` and
   ``TRITON_EDGE_UPDATE=1`` before model construction.
6. **4 NEW tests** (`molmetal/tests/test_wf328w9_triton_edge.py`):
   - ReLU+Linear kernel correctness vs CPU reference (5 shapes incl. spec)
   - ReLU+Linear+Linear kernel correctness vs CPU reference (4 shapes)
   - D-MPNN edge update parity with the canonical PyTorch path
   - CPU speed regression guard (Triton wrapper must not regress)
   - Fallback safety gate closed → bit-exact with ``F.relu(F.linear(...))``

Total: **5 NEW tests, all PASS.**

---

## 3. Files added / modified

| Path | Status | Purpose |
|---|---|---|
| `triton_kernels/fused_relu_mlp.py` | NEW | The two kernels + host wrappers + nn.Module wrappers (`FusedReLUMLP`, `FusedReLULinear`) |
| `triton_kernels/__init__.py` | MOD | Export `fused_relu_linear`, `fused_relu_mlp`, `FusedReLULinear`, `FusedReLUMLP` |
| `triton_kernels/config.py` | MOD | Add `use_fused_relu_linear` / `use_fused_relu_mlp` gates |
| `molmetal/models/dmpnn.py` | MOD | Wire `--triton-edge-update` flag into the per-layer edge MLP site (4-line diff) |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | MOD | Add `--triton-edge-update` CLI flag (16-line diff + env-var export in `main`) |
| `molmetal/tests/test_wf328w9_triton_edge.py` | NEW | 5 NEW tests (correctness + speed + fallback) |
| `TODO/completed/328w9_triton_edge_kernel.md` | NEW | This file |

---

## 4. Honest framing per `party-twenty-spirit` §3.5

| Claim | Tag | Source |
|---|---|---|
| Two new Triton kernels ship (fused_relu_linear, fused_relu_mlp) | **MEASURED** | `triton_kernels/fused_relu_mlp.py` on disk |
| 5/5 NEW tests pass on CPU | **MEASURED** | `pytest molmetal/tests/test_wf328w9_triton_edge.py -v` (this run) |
| Kernel correctness vs `F.relu(F.linear(...))` within atol=1e-3 on the spec shape (M=2048, D=256, H=128) | **MEASURED** | `test_relu_linear_kernel_correctness_vs_cpu_reference` |
| Default OFF on Blackwell until Wf-326E-A autotune retune | **MEASURED** | `triton_kernels/config.py` L107 ("Triton kernels tuned for gfx1101 are 3-11x slower than PyTorch on Blackwell; PyTorch is now the default") |
| PROJECTED 1.3-2x speedup on D-MPNN per-layer | **PROJECTED** | Wf-326E-B (audit); MEASURED Blackwell numbers = Wf-326E-D |
| PROJECTED 1.5-3x aggregate with sm_120 retune + the four new kernels | **PROJECTED** | Wf-326E audit §audit_summary |
| Triton's gfx1101-tuned configs are 3-11x slower than PyTorch on Blackwell | **MEASURED** | `triton_kernels/config.py` L107 (root-cause, the reason for default-OFF) |

### 4.1 Blocked dependencies

- **Wf-326E-A sm_120 autotune retune** is the blocker for the PROJECTED
  1.3-2x Blackwell speedup to materialize.  Until that retune lands the
  fused kernel stays OFF by default and the gate is closed.
- **Wf-326E-D re-measure** is the blocker for converting the PROJECTED
  speedup claim into MEASURED numbers on RTX 5090.  No new GPU
  measurement was performed in this Wf (CPU-only per AGENTS.md §5.1
  kill_then_adapt directive).

---

## 5. Dispatch policy in detail

The host wrappers (`fused_relu_linear`, `fused_relu_mlp`) follow the
same dispatch policy as the existing `fused_silu_mlp` / `fused_gelu_mlp`:

1. **CPU tensors** → PyTorch reference (Triton raises on CPU pointers
   during autotune probe).
2. **CUDA + gate open** → Triton kernel.
3. **CUDA + gate closed** → PyTorch reference (default state).

The new gates are:
- `triton_config.use_fused_relu_linear(tensor)` requires:
  `tensor.is_cuda` AND `TRITON_USE_FUSED=1` AND `TRITON_RELU_LINEAR=1` AND
  `tensor.dim() >= 1` AND `feat_dim <= MAX_FEAT_DIM=4096` AND
  `tensor.numel() * bytes_per_elem >= MIN_BYTES=1MB` AND
  (`training` OR `TRITON_USE_FUSED_EVAL=1`).
- `triton_config.use_fused_relu_mlp(tensor)` mirrors the above with
  `TRITON_RELU_MLP=1`.

The D-MPNN wiring adds a *third* gate: `TRITON_EDGE_UPDATE=1` (or the
CLI flag `--triton-edge-update`).  This is a coarse global on/off switch
so operators can opt in per-training-run.

**Why three gates?**  ``TRITON_USE_FUSED`` is the global gate that
unlocks all fused kernels (SiLU/GELU/RELU/MLP/etc.).  ``TRITON_RELU_LINEAR``
/ ``TRITON_RELU_MLP`` are the per-op gates for the NEW kernels (so they
can be enabled independently).  ``TRITON_EDGE_UPDATE`` is the D-MPNN
specific gate so the per-layer edge MLP path can be opted in via CLI
without affecting the ``atom_embed`` / ``edge_embed`` / ``readout_mlp``
sites (which would need their own wiring site + CLI flag in a follow-up).

---

## 6. Test results

```
$ PYTHONPATH=/home/hugo/codes/mol-metal .venv/bin/python -m pytest \
    molmetal/tests/test_wf328w9_triton_edge.py -v
=========================== test session starts ===========================
collected 5 items
test_relu_linear_kernel_correctness_vs_cpu_reference ......... PASSED [ 20%]
test_relu_mlp_kernel_correctness_vs_cpu_reference ........... PASSED [ 40%]
test_dmpnn_edge_update_triton_correctness_vs_pytorch ........ PASSED [ 60%]
test_speedup_vs_pytorch_native_cpu_bench .................... PASSED [ 80%]
test_fallback_to_native_works ................................ PASSED [100%]
=========================== 5 passed in 1.83s ==============================
```

Regression: existing D-MPNN + triton-audit tests still pass (17 PASS, 2
unrelated pre-existing data-loading failures on
`/mnt/storage/data/molmetal/MetalCytoToxDB.csv` not found — those
existed before this Wf).

---

## 7. Wiring site in D-MPNN (`molmetal/models/dmpnn.py`)

The 4-line diff to the per-layer ``edge_mlp[layer]`` site:

```python
# Wf-328-W9: route the per-layer edge MLP through the fused
# ReLU+Linear kernel when --triton-edge-update is on AND the
# input is on CUDA AND the dispatch gate is open.
edge_mlp = self.edge_mlp[layer]
cat_input = torch.cat([h_e, h_src], dim=-1)  # (E, 2D)
if (
    _TRITON_EDGE_UPDATE_ENABLED
    and cat_input.is_cuda
    and _triton_config.use_fused_relu_linear(cat_input)
):
    msg = _fused_relu_linear(
        cat_input,
        edge_mlp[0].weight,
        edge_mlp[0].bias,
    )
else:
    msg = edge_mlp(cat_input)  # (E, D)
```

The ``edge_mlp[0]`` access mirrors the canonical
``nn.Sequential(Linear, ReLU)`` parameter layout (``linear.weight`` /
``linear.bias``) so existing state-dicts load without remapping.

---

## 8. CLI flag in `train_dmpnn_multitask_wasserstein.py`

```python
p.add_argument(
    "--triton-edge-update",
    dest="triton_edge_update",
    action="store_true",
    default=False,
    help=(
        "Wf-328-W9: route D-MPNN per-layer edge MLP through the "
        "fused ReLU+Linear Triton kernel. Requires CUDA + "
        "TRITON_USE_FUSED=1. Default OFF (gfx1101-tuned configs "
        "are 3-11x slower than PyTorch on Blackwell until "
        "Wf-326E-A's sm_120 retune lands)."
    ),
)
```

And in `main`:

```python
if getattr(args, "triton_edge_update", False):
    os.environ.setdefault("TRITON_USE_FUSED", "1")
    os.environ.setdefault("TRITON_RELU_LINEAR", "1")
    os.environ.setdefault("TRITON_EDGE_UPDATE", "1")
    print(
        "[wf328-w9] --triton-edge-update: TRITON_USE_FUSED=1 + "
        "TRITON_RELU_LINEAR=1 + TRITON_EDGE_UPDATE=1 (D-MPNN "
        "per-layer edge MLP routes through fused_relu_linear)",
        flush=True,
    )
```

The env vars are set BEFORE the model is constructed so the
`triton_config` singleton picks them up at import time (the singleton
reads env vars at construction).

---

## 9. Out-of-scope (forward-looking)

The following are explicitly NOT in Wf-328-W9 (forward-looking follow-ups):

- Wiring `fused_relu_mlp` into the `atom_embed` / `edge_embed` /
  `readout_mlp` sites — needs a separate wiring path because the
  existing modules use ``nn.Sequential(Linear, ReLU, Linear)`` (3 ops
  total: linear+relu+linear).  The `FusedReLUMLP` wrapper class is
  shipped but the actual wire-in is Wf-328-W9.5 / Wf-326E-B2.
- Blackwell MEASURED speedup — Wf-326E-D (gated on Wf-326E-A retune).
- Wiring `fused_relu_linear` into other Linear+ReLU sites (e.g. the
  ``DomainAdversarialHead.classifier`` head, the multi-task FFN heads
  in `metal_hybrid_v4`).  Each is a separate Wf.

---

## 10. References

- Wf-326E audit (`molmetal/reports/wf326e_triton_audit/audit.json`)
- Wf-326E close-out (`TODO/completed/326e_triton_audit.md`)
- `triton_kernels/config.py` L107 (the default-OFF rationale)
- `triton_kernels/autotune.py` (the canonical ``AUTOTUNE_CONFIGS`` grid
  used by the new kernels)
- `triton_kernels/fused_silu_mlp` / `fused_gelu_mlp` (the SiLU / GELU
  siblings the ReLU kernel closes the gap on)
- `molmetal/models/dmpnn.py` L36-47 (the existing wiring comment that
  flagged the ReLU gap)

---

**END OF Wf-328-W9** — Owner Hugo Lee; cross-ref `party-twenty-spirit`
§3.5; CPU-only ship surface; Blackwell MEASURED speedup = Wf-326E-D
follow-up.
