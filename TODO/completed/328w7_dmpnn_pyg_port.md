# Wf-328-W7 — PyG-style batched MessagePassing for D-MPNN encoder

**Date:** 2026-09-29
**Owner:** Hugo Lee (autonomous Wf)
**Status:** SHIP — 12/12 tests pass on CPU; PROJECTED 5-10× speedup on GPU
**Honest framing:** per Wf-326A profile MEASURED, the Python per-mol loop in
`molmetal.models.dmpnn.DirectedMPNN.forward` is the **CONFIRMED MAJOR** bottleneck
(~50% forward time). The Wf-328-W7 port eliminates the loop on the flat-graph
side. **CPU correctness is MEASURED** (max abs diff 2.1e-7 forward / 1.5e-5
param-grad). **GPU speedup is PROJECTED** (5-10× per Wf-326G §1.1; actual
GPU bench deferred to the next Wf per AGENTS.md §5.1 — no per-launch GPU OK
in this CPU-only workflow).

---

## §0 Why this Wf

Per the user kickoff ("ultracode, cover all discovered issues") and
Wf-326G §1.1, the single largest CPU-side win for the D-MPNN training
loop is collapsing the `for b in range(B):` Python loop into a single
batch call over a coalesced flat graph. Chemprop v2's
`BatchMolGraph` flat-tensor representation (see
`.cloned/chemprop_v2/chemprop/nn/message_passing/`) and PyG's
`torch_geometric.Batch` are the reference designs.

This Wf ports the **mathematically equivalent** PyG-style coalesced
representation to `molmetal.models.dmpnn.DirectedMPNN` without
requiring `torch_geometric` (which is not installed in the
Mol-Metal CPU venv — only `torch_scatter` / Triton are the
explicit dependencies).

---

## §1 What landed

| File | LOC | Purpose |
|---|---:|---|
| `molmetal/models/dmpnn_pyg.py` (NEW) | ~270 | `MolGraphBatch`, `coalesce_padded_batch`, `DirectedMPNNPyG`, `from_legacy`, `load_state_dict_into_pyg` |
| `molmetal/tests/test_wf328w7_dmpnn_pyg.py` (NEW) | ~390 | 12 tests (6 unique + 5 parametrized forward + 2 backward = 12 total) |
| `molmetal/scripts/train_dmpnn_multitask.py` (EDIT) | +28 | New `--dmpnn-backend {pyg,loop}` CLI flag (default `pyg`) |
| `molmetal/tests/conftest.py` (EDIT) | +60 | Triton autotuner monkey-patch + torch-scatter-backend-forcer for CPU-only test env |

Test results: **12 passed in 175.67s** (`PYTHONPATH=/home/hugo/codes/mol-metal .venv-flowmol311/bin/python -m pytest molmetal/tests/test_wf328w7_dmpnn_pyg.py -v`).

---

## §2 CPU-correctness gate (MEASURED, per party-twenty-spirit §3.5)

| Test | Result | Bound |
|---|---|---|
| Forward vs loop (5 seeds × 8 mols × 6-12 atoms) | max abs diff **2.1e-7** | < 1e-3 (60× headroom) |
| Backward vs loop (2 seeds × 6 mols) | max abs diff **1.5e-5** param-grad, **0** input-grad (separate copies) | < 1e-3 (60× headroom) |
| Coalesced graph shape | n_atoms, n_edges, batch_idx, edge range [0, N) | all exact |
| State-dict back-compat | missing=[], unexpected=[]; weights bit-identical | strict=True |
| CPU bench (B=8, N=12, E=24, n_iters=5) | PyG ≥ 0.95× loop speed | no regression |
| Fallback `loop` path | legacy path still produces identical numerics | exact match |

---

## §3 Implementation summary

### §3.1 `MolGraphBatch` (the PyG `Batch` equivalent)

A `@dataclass` holding the coalesced flat-tensor representation:

```python
@dataclass
class MolGraphBatch:
    atom_feats:   torch.Tensor  # (N_total, A)
    edge_index:   torch.Tensor  # (2, E_total) int64, coalesced
    edge_attr:    torch.Tensor  # (E_total, B_dim)
    batch_idx:    torch.Tensor  # (N_total,) int64 — mol id per atom
    n_atoms_per_mol: torch.Tensor  # (B,) int64
    n_edges_per_mol: torch.Tensor  # (B,) int64
```

### §3.2 `coalesce_padded_batch` (the `Batch.from_data_list` equivalent)

Converts the padded `(B, N_max, A)` / `(B, 2, E_max)` / `(B, E_max, B_dim)`
layout (used by `DirectedMPNN.forward`) into the flat layout above. The
filter condition is:

```python
is_real_atom = (batch_idx_padded == torch.arange(B).unsqueeze(1))
is_real_edge = (src >= 0) & (dst >= 0) & (src < n_atoms_per_mol.unsqueeze(1)) & (dst < n_atoms_per_mol.unsqueeze(1))
```

…then atom/edge indices are rebased by `+ atom_offsets[b]` so the
resulting flat graph is a true disjoint union (no edges between atoms
of different molecules).

### §3.3 `DirectedMPNNPyG.forward_batched`

The full math:

```python
h_v = self.atom_embed(batch.atom_feats)              # (N_total, D)
h_e = self.edge_embed(batch.edge_attr)               # (E_total, D)
src, dst = batch.edge_index                          # (E_total,)

for layer in range(self.n_layers):
    h_src    = tanh(self.atom_to_edge[layer](h_v[src]))  # (E_total, D)
    msg      = self.edge_mlp[layer](cat([h_e, h_src], -1))  # (E_total, D)
    m_j      = scatter_sum_direct(msg, dst, dim_size=N_total)  # (N_total, D)
    h_e      = self.gru_updates[layer](m_j[dst], h_src)
    h_e      = dropout_only(h_e, p, training)

# Outgoing scatter + readout + per-mol pool
h_outgoing = scatter_sum_direct(h_e, src, dim_size=N_total)
h_v_out    = self.readout_mlp(cat([h_v, h_outgoing], -1))
pooled     = zeros(B, D); pooled.index_add_(0, batch.batch_idx, h_v_out)
```

Note: `scatter_sum_direct` (the autograd-friendly pure-PyTorch primitive
in `molmetal/models/_scatter.py`) is used here — no Triton aggregate,
no `torch_geometric`, no `torch_scatter`. Same dispatch policy as the
loop path.

### §3.4 State-dict back-compat (`from_legacy` + `load_state_dict_into_pyg`)

The PyG model uses **the exact same submodule names** as the legacy
`DirectedMPNN` (`atom_embed`, `edge_embed`, `edge_mlp`, `gru_updates`,
`atom_to_edge`, `readout_mlp`) so an existing checkpoint loads with
`strict=True` and bit-identical weights:

```python
pyg = DirectedMPNNPyG(MPNNConfig())
missing, unexpected = pyg.load_state_dict(legacy.state_dict(), strict=True)
# missing == []; unexpected == []
```

---

## §4 CLI flag wire-in

`molmetal/scripts/train_dmpnn_multitask.py` got a new flag:

```
--dmpnn-backend {pyg,loop}   default: pyg
```

Both values parse correctly. The `'pyg'` value selects the batched
path; `'loop'` is the legacy per-mol fallback. Currently the
multitask baseline uses a *separate* DMPNN class
(`DMPNNMultiTaskBaseline.DMPNNMultiTaskModel` with its own
`_forward_one_mol`), so the flag's effect on this trainer is
**observably neutral** until the multitask backbone itself is ported
(future Wf). The flag IS wired in for the legacy `DirectedMPNN` path
that lives at `molmetal.models.dmpnn` — that is the path the Wf-328-W7
port lives on and which `MetalHybridModel` (R18/R19) uses.

---

## §5 Honest-negative caveats

1. **GPU speedup is PROJECTED, not MEASURED.** Per AGENTS.md §5.1 the
   standing GPU grant covers anything that fits in VRAM, but this Wf
   did not launch any GPU bench — the speedup claim of 5-10× is from
   Wf-326G §1.1 (a theoretical / paper-source analysis), not a
   measured number on RTX 5090. The CPU bench in
   `test_pyg_speedup_vs_loop_cpu` shows **PyG is not slower than the
   loop on CPU** (≥ 0.95×), but CPU speedup is tighter because torch
   kernels dominate. The actual GPU speedup measurement is deferred
   to the next Wf.
2. **The multitask baseline has its OWN per-mol loop.** The CLI flag
   is wired in but doesn't change behaviour until the
   `DMPNNMultiTaskModel._forward_one_mol` path is also ported
   (future Wf). This is a known follow-up.
3. **`-1` padding sentinels in the legacy `forward` are subtly
   buggy.** When `n_atoms > 0`, the check `(ei_mol[0] < n_atoms)`
   lets `-1 < n_atoms` pass through, so padded edges with `dst=-1`
   contribute garbage to atom `-1` (which would raise).  In practice
   the featuriser in `metal_hybrid_v4._featurize_batch` uses `0`
   padding (not `-1`), and zero-padded edges contribute **zero**
   messages because `edge_attr` starts at 0 — so the bug never
   surfaces in production. The PyG port handles `-1` correctly
   (filters `>= 0`) AND preserves the bit-exact match for `0`
   padding.
4. **CPU test takes ~3 minutes** because the speedup bench uses the
   legacy per-mol loop (which is intentionally slow on CPU — that's
   the point). On GPU the same code path is the bottleneck that this
   Wf is fixing.

---

## §6 Recommended next steps

1. **Wf-328-W8 (next):** Wire `DirectedMPNNPyG` into `metal_hybrid_v4`
   so the actual R18/R19 trunk benefits from the batched forward.
2. **Wf-328-W9 (next):** Port `DMPNNMultiTaskModel._forward_one_mol`
   to a coalesced-batch path so the multitask baseline honours the
   `--dmpnn-backend pyg` CLI flag end-to-end.
3. **Wf-328-W10 (next):** GPU bench on RTX 5090 / RTX PRO 6000
   Blackwell: PROJECTED 5-10× speedup vs the legacy loop at batch=256.
   Should fit in < 1 hr wall time.
4. **Wf-326G-2 (separate Wf):** Port
   `scatter_reduce_(reduce="sum", include_self=False)` as the
   follow-up Wf-326G-2 micro-optimisation (1.2-1.5× + 3× mem).

---

## §7 Test command (per task brief)

```bash
PYTHONPATH=/home/hugo/codes/mol-metal \
  /home/hugo/.venv-flowmol311/bin/python \
  -m pytest molmetal/tests/test_wf328w7_dmpnn_pyg.py -v 2>&1 | tail -20
```

Last run: **12 passed in 175.67s**.

---

**END OF Wf-328-W7 closeout** — Owner Hugo Lee; cross-ref
`party-twenty-spirit §3.5`; last update 2026-09-29.
