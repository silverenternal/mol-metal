# Wf-358-B2 — SO(3)-equivariant D-MPNN message passing scaffold (SHIPPED)

**Workflow**: Wf-358 ultracode residual wave / Wf-357-C2 architecture improvement #2
**Date**: 2026-09-29
**Status**: SHIPPED (CPU-only scaffold; GPU retrain deferred per AGENTS.md §5.1)
**Honest framing**: SCAFFOLD only — PROJECTED +10-20% m3 Vina lift (NOT measured on Mol-Metal yet)

## Summary

Wf-358-B2 implements an **SO(3)-equivariant message passing scaffold** for
the D-MPNN backbone. Per Wf-357-C2 §1 criterion (c): the canonical
:class:`molmetal.models.dmpnn.DirectedMPNN` is **E(3)-invariant only**
— it treats the molecule as a 2D graph and discards 3D coordinates.
For Vina-style tasks where pocket geometry matters, the model cannot
exploit bond-length / bond-angle / pocket-distance information.

This scaffold augments the D-MPNN message passing with:

1. **SchNet-style radial basis function (RBF) embedding** of pairwise
   distances ``‖x_dst − x_src‖₂``.  Distances are SO(3)-invariant
   scalars, so the augmented edge features remain E(3)-invariant and
   the *pooled graph embedding* is still rotation-equivariant up to
   the trivial action (per Wf-357-C2 §1 criterion (c): "outputs
   transform as a scalar, not as a vector").
2. **EGNN-style equivariant coordinate update** in the spirit of
   Satorras 2021 + SchNet 2017 — the per-atom ``coord_scale`` MLP
   emits a scalar weight that scales the relative direction vector
   ``(x_dst − x_src)``, so the resulting coordinate update
   ``Δx = Σ scalar_ij · (x_dst − x_src)`` is **equivariant** under
   rotation of all input coordinates.

The scaffold is wired as an **opt-in flag** (``--equivariant-mp``,
default OFF, see
:data:`molmetal.scripts.train_dmpnn_multitask_wasserstein.parse_args`)
— without the flag the model behaves bit-identically to the legacy
:class:`DirectedMPNN`.  Default OFF preserves the published baseline
numerical signature for Wf-124 P3 / Wf-181 P1 / Wf-131 P1
reproducibility audits.

## Files shipped

| Path | LOC | Purpose |
|---|---:|---|
| `molmetal/models/equivariant_dmpnn.py` | 347 (≈250 code) | SO(3)-equivariant D-MPNN module |
| `molmetal/tests/test_wf358b2_equivariant_mp.py` | 369 | 4 NEW + 1 bonus tests |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | +50 | `--equivariant-mp` CLI flag + `DEFAULT_EQUIVARIANT_MP=False` |

## Architecture

### Module structure (`molmetal/models/equivariant_dmpnn.py`)

```
EquivariantDMPNN (nn.Module)
├── base: DirectedMPNN              # legacy D-MPNN (state-dict loadable)
├── edge_rbf_proj: Linear(D+n_rbf → D)   # RBF fusion into edge_attr
├── coord_mlps: ModuleList          # per-layer scalar coord MLP (EGNN gate)
└── forward(h_atom, edge_index, edge_attr, coords) → (N, D)
    1. Embed atoms (legacy path)
    2. Compute pairwise distances → RBF embedding (SchNet §3.1)
    3. Fuse RBF into edge_attr via edge_rbf_proj
    4. For each layer:
       a. D-MPNN message passing (legacy edge_mlp / GRU / atom_to_edge)
       b. EGNN coord update: scalar_ij = tanh(MLP([h_e, rbf]))
                              Δx_i = Σ_j scalar_ij · (x_i − x_j)
    5. Readout (legacy path: sum outgoing h_e + concat h_v)
```

### Key design constraints (per Wf-357-C2 §1)

* **Outputs are scalars** (per-atom `(N, D)` features) — they transform
  as scalars under SO(3), not as vectors.  This means the *pooled
  graph embedding* is trivially equivariant under rotation.
* **Coord trajectory is a side channel** — the updated `x` is
  discarded (this is a feature extractor, not a generative model per
  Wf-357-C2 §1 criterion (d)).
* **Default OFF** — the legacy `DirectedMPNN` numerical signature is
  preserved exactly when `--equivariant-mp` is not passed.

## CLI flag

```bash
python molmetal/scripts/train_dmpnn_multitask_wasserstein.py \
    --metals Pt --seeds 0 --epochs 30 \
    --equivariant-mp                     # NEW: enable SO(3)-equivariant MP
```

The flag is wired into the existing
:func:`parse_args` with banner observability (`equivariant_mp={True,False}`)
and follows the Wf-328-W9 / Wf-334-U1 style (env-var opt-in
`EQUIVARIANT_MP=1` + CLI flag).

## Tests (4 NEW + 1 bonus = 5 tests)

| Test | Verdict |
|---|---|
| `test_rbf_expansion_invariant_under_so3` | PASS — RBF(dist) is SO(3)-invariant scalar (max abs diff < 1e-5 over 4 random SO(3) rotations) |
| `test_forward_and_gradient_flow` | PASS — forward returns `(N, D)`; legacy D-MPNN params receive gradient through pooled-output loss; coord_mlps params receive gradient through coord-aware loss (proves EGNN coord update is differentiable end-to-end) |
| `test_so3_rotation_equivariance_probe` | PASS — per-atom / pooled outputs are SO(3)-invariant under 4 random SO(3) rotations (max abs diff < 1e-3) |
| `test_cli_flag_equivariant_mp_wired` | PASS — `parse_args` accepts `--equivariant-mp`; default is OFF |
| `test_backward_compat_with_dmpnn` | PASS — legacy `DirectedMPNN` instantiates + forward + state-dict round-trip unchanged; `EquivariantDMPNN` wraps the legacy backbone with strict state-dict loadability |

## Wall budget

* Implementation: ~10 min CPU
* Tests: ~5 min CPU (5 tests + import verification)
* **Total**: ~15 min CPU (well under the PROJECTED 45-75 min CPU-only budget)

## Honest framing per `party-twenty-spirit` §3.5

### SCAFFOLD limitations

* The `coord_mlps` parameters feed only into the EGNN coord-update
  side channel; they only receive gradient when the loss involves the
  *final coord trajectory* ``x`` (the per-atom features ``out`` are
  computed from the legacy D-MPNN path which only sees the original
  coords via the RBF augmentation).  This is a SCAFFOLD-level
  limitation — production wiring requires adding a coord-regression
  loss head (TODO per Wf-358-B2 §3 follow-up).
* No chemprop / BondHead / DMPNN-Cache integration yet.  The scaffold
  sits alongside the legacy D-MPNN path; integration into the
  Wasserstein trainer's :class:`WassersteinDMPNN` is a separate Wf.
* GPU retrain / Vina cohort lift NOT measured.  PROJECTED +10-20%
  m3 Vina lift is *projected* not measured.

### What IS measured

* **5/5 NEW tests PASS** (CPU-only correctness).
* **98/103 backward-compat tests PASS** (the 5 failures are
  pre-existing data-access failures — `MetalCytoToxDB.csv` not at
  `/mnt/storage/data/molmetal/MetalCytoToxDB.csv`; verified
  pre-existing via `git stash`).
* **SO(3) invariance probe**: pooled / per-atom outputs agree to
  within max abs diff < 1e-3 under 4 random SO(3) rotations
  (verifies the EGNN invariance contract).
* **Backward compat**: legacy `DirectedMPNN` state-dict loads into
  `EquivariantDMPNN.base` strictly (no missing / unexpected keys).

## Acceptance gate (Wf-358-B2)

- [x] `molmetal/models/equivariant_dmpnn.py` created (~120 LOC code).
- [x] SO(3) rotation-aware message passing implemented.
- [x] CLI flag `--equivariant-mp` (default OFF) wired.
- [x] 4 NEW tests + 1 bonus = 5 tests, all PASS.
- [x] Backward compat verified (legacy `DirectedMPNN` path unchanged).
- [x] CPU-only correctness gate passed.

## Follow-up (NOT in scope of Wf-358-B2)

* **G5 / G6 retrain** with `--equivariant-mp` to MEASURE the
  +10-20% m3 Vina lift on real data.  BLOCKED on user OK per
  AGENTS.md §5.1.
* **Coord-regression loss head** to train the EGNN coord update path
  end-to-end (the current scaffold leaves `coord_mlps` params under-
  trained when the loss is on `out`).
* **chemprop v2 / v3 + BondHead integration** — the scaffold currently
  drops in as a drop-in replacement for `DirectedMPNN`; integration
  into the multi-task Wasserstein trainer requires additional wiring.
* **GPU autograd correctness** — the current CPU-only tests confirm
  autograd semantics; GPU correctness (BF16 / AMP / CUDA Graphs) is
  deferred to a GPU-validation wave per AGENTS.md §5.1.

## Reference

* Wf-357-C2 architecture improvement #2 spec (Catherine).
* Cao et al. 2025 (NVIDIA + MIT), "SO(3)-Averaged Flow Matching with
  Guidance Distillation and Reflow", ICML 2025, arXiv:2507.09785 —
  the SO(3)-averaged objective is a related but distinct technique
  (averages the *loss* over SO(3) rather than parameterizing the
  *message passing* over SO(3)).
* Satorras et al. 2021, "E(n) Equivariant Graph Neural Networks",
  ICML 2021 — the canonical EGNN formulation that this scaffold
  adapts for D-MPNN message passing.
* Schütt et al. 2017, "SchNet: A continuous-filter convolutional neural
  network for modeling quantum interactions" — the RBF expansion
  formulation used in `_rbf_expansion`.