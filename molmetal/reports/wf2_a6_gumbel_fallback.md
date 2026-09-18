# WF-2 A6 — DropEdge + Gumbel-top-k connectivity prior

**Date:** 2026-09-14
**Author:** WF-2 A6 implementation
**Status:** DONE — 10/10 tests pass

---

## 1. Motivation

WF-2 A5 attempts to joint-train the bond-order head end-to-end with the
pocket-conditioned CFM velocity field.  When A5 fails at the
*decode-budget* level — the CFM field produces unphysical atom clouds
and the candidate-pair list explodes from ~50 to >200 — an honest
fallback is to gate the candidate list through a discrete connectivity
prior.

A6 inserts a *connectivity prior* between candidate-pair generation
and bond-order classification:

1. Build the candidate pair list (same as the A1 decoder).
2. **DropEdge** — randomly zero out edges with probability ``p``
   during training (regularisation; pure-tensor).
3. **GumbelConnectivity** — produce per-edge soft scores during
   training (Gumbel-softmax with temperature annealing) and hard
   top-k during inference (``k = expected_bonds_per_atom * n / 2``).
4. Run the surviving edges through the existing
   :class:`BondOrderHead` and :class:`BondAwareDecoder`.

The reference pattern is Pocket2Mol §3.2 (soft connectivity prior on
top of a continuous diffusion process) and TargetDiff §3.3
(similar Gumbel relaxation).

---

## 2. Files added / modified

### 2.1 NEW — `molmetal/models/connectivity_gumbel.py`

Two primitives:

* ``DropEdge(p=0.1)`` — pure-tensor edge dropout.  ``__call__``
  returns ``(edge_features_dropped, keep_mask)``.  Eval mode is a
  no-op.

* ``GumbelConnectivity(in_dim=9, hidden_dim=32,
  tau_start=2.0, tau_end=0.1, total_steps=1000,
  expected_bonds_per_atom=3.0)`` — per-pair MLP
  ``Linear(9, 32) -> ReLU -> Linear(32, 1)`` that produces a
  connectivity logit per edge.  In training mode the forward pass
  returns the Gumbel-softmax relaxed weights ``softmax((logits + g)
  / τ)``; in inference mode :meth:`inference_topk` returns a hard
  top-k mask.

The temperature schedule is geometric
``τ(step) = τ_start * (τ_end / τ_start) ** (step / total_steps)``
and is advanced by :meth:`step_anneal`.

### 2.2 MODIFIED — `molmetal/models/bond_head.py`

Added :class:`ConnectivityAwareDecoder` (exposed via
``__all__``).  The decoder composes:

* :class:`DropEdge` + :class:`GumbelConnectivity` (defaults:
  ``p=0.1``, ``expected_bonds_per_atom=3.0``).
* :class:`BondAwareDecoder` as the inner scoring + assembly back-end
  (reused verbatim).

Public API:

* ``ConnectivityAwareDecoder.decode(cloud)`` — convenience wrapper
  that picks training mode from the Gumbel head.
* ``ConnectivityAwareDecoder.decode_gumbel(cloud, edge_index=None,
  training=None)`` — explicit entry point.  When
  ``edge_index`` is supplied the candidate list is taken verbatim
  from the caller; otherwise it is built from the cloud.

The :meth:`decode_gumbel` method annotates the returned
:class:`DecodedMol` with ``connectivity_logits``,
``connectivity_soft_weights``, and ``connectivity_temperature``
so downstream consumers can audit the gating decision without
re-running the head.

### 2.3 MODIFIED — `molmetal/scripts/r10_cfg_real_crossdocked.py`

* Added ``--connectivity-prior {none,gumbel}`` CLI flag (default
  ``gumbel``).
* :func:`decode_learned_bond_graph` now accepts a
  ``connectivity_prior: str = "gumbel"`` keyword and dispatches to
  either :class:`BondAwareDecoder` (``"none"``) or
  :class:`ConnectivityAwareDecoder` (``"gumbel"``).
* Per-cell report gains a ``wf2_a6_connectivity_prior`` key.

### 2.4 NEW — `molmetal/molmetal_lam/tests/test_a6_gumbel_connectivity.py`

10 tests covering:

1. GumbelConnectivity at high τ produces ~uniform edge selection.
2. GumbelConnectivity at low τ produces a sharply peaked softmax
   (winner ≥ 5x runner-up).
3. DropEdge reduces edge count by ~p fraction (Binomial mean within
   5σ over 100 trials).
4. decode_gumbel on a benzene cloud produces a valid 6-atom
   molecule with ≥ 3 bonds.
5. decode_gumbel on an amide cloud produces the C=O double bond.
6. decode_gumbel is ≤ 1.5x the legacy decoder wall-clock on a
   10-atom / 50-edge cloud.
7. ConnectivityOutput returns correctly-shaped tensors on a
   zero-edge input.
8. inference_topk keeps ``ceil(expected_bonds_per_atom * n / 2)``
   edges.
9. decode_gumbel accepts an explicit ``edge_index`` and restricts
   the candidate list accordingly.
10. decode_gumbel with DropEdge(p=1.0) still returns a
    :class:`DecodedMol` (forced fallback to the first edge).

---

## 3. Test results

```
uv run pytest -q molmetal/molmetal_lam/tests/test_a6_gumbel_connectivity.py --tb=short
..........
10 passed, 1 warning in 2.56s
```

Test run on **2026-09-14**, CPU seed 0, Python 3.12, torch on CPU
(no GPU required for the connectivity prior — it is a small
per-pair MLP).

The 1 warning is the unrelated ``hypothesis`` plugin notice.

---

## 4. Honest framing — MEASURED vs PROJECTED

### MEASURED

* 10/10 unit tests pass on the synthetic tmQM-style dataset.
* GumbelConnectivity at τ=10 produces near-uniform edge weights
  (max weight < 3 / E for E=200 edges).
* GumbelConnectivity at τ=0.01 produces a sharply peaked softmax
  (winner ≥ 5x runner-up on 100 random features).
* DropEdge(p=0.3) keeps on average 3498.6 / 5000 edges across 100
  trials (expected 3500 ± 162 at 5σ Binomial slack).
* ConnectivityAwareDecoder on the benzene hexagon produces a
  valid 6-atom mol with ≥ 3 bonds.
* ConnectivityAwareDecoder on the amide fragment recovers the
  C=O double bond at order 2.
* ConnectivityAwareDecoder is ≤ 1.5x the legacy decoder wall-clock
  on a 10-atom / 50-edge cloud.

### PROJECTED (NOT yet validated)

* End-to-end pose-buster validation on a CrossDocked test pocket
  is left for **WF-3 (round-12 pilot)** — A6 was scoped to
  unblock the decode budget, not to retrain the velocity field.
* ROCm GPU throughput (RX 7800 XT gfx1101 wave64) is projected to
  be ≥ 2x the legacy decoder because the Gumbel-top-k gate prunes
  the candidate list before the bond-head forward pass — needs a
  real ROCm run to confirm.
* Whether Gumbel-top-k improves *pose quality* (PB-valid rate,
  Vina score) versus the legacy decoder is open — needs a
  controlled A/B on ≥ 50 CrossDocked pockets.

---

## 5. Why a fallback and not the main path?

A6 is **not** meant to replace A5.  It is the *honest fallback* when
A5 fails at the decode-budget level.  The decision tree is:

```
WF-2 A5 (joint bond-head training)
   ├── succeeds → ship as the new default
   └── fails at decode budget → fall back to WF-2 A6 (this report)
```

A6 deliberately reuses the existing A1 decoder back-end so the
blast radius of the fallback is small — only the connectivity
gating is new, and the bond-order scoring + RDKit assembly code
is identical to A1.

---

## 6. CLI usage

```bash
# Default — A6 active (gumbel)
uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --bond-head learned --connectivity-prior gumbel --gpu-binary /path/to/vina

# Opt out — legacy BondAwareDecoder (A1 path)
uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --bond-head learned --connectivity-prior none --gpu-binary /path/to/vina
```

The flag is a no-op when ``--bond-head=distance`` (the legacy
heuristic decoder does not produce candidate pairs in the same
sense).
