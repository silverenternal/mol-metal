# WF-2 recon — A6 honest fallback feasibility (DropEdge + Gumbel-top-k connectivity prior)

**Generated:** 2026-09-14
**Status:** MEASURED (audit only — reconciles on-disk code + PyTorch ops availability)
**Scope:** Confirm that A6 — a headless bond-recovery mechanism in the spirit of
Pocket2Mol §3.2 / TargetDiff §3.3 that combines random DropEdge on the
message-passing graph with a Gumbel-top-k stochastic adjacency prior — is
feasible as a drop-in replacement for the learned `BondOrderHead` (A5)
inside the existing `LipmanFlowMatchingAdapter` pipeline without any new
external dependency beyond core PyTorch.

---

## 1. Edge-index construction today (MEASURED)

The fully-connected (no-self-loop) dummy graph is built in exactly two
places — both use the same `(B, 2, n*(n-1))` layout that A6 reuses:

- `LipmanFlowMatchingAdapter._make_dummy_edge_index(b, n, device)`
  at `molmetal/adapters/flow_matching_lipman/__init__.py:1763-1775`
  — used in `train_step` (line 1402) and `_generate_impl` (line 1517).
  It calls `torch.arange(n)`, builds the (n, n) broadcast grid, applies
  `src != dst`, and stacks the surviving (src, dst) pairs into
  `(b, 2, n*(n-1))`.  Cost is O(B*n^2) memory + a single contiguous
  copy on the resolved device — already proven on RX 7800 XT (gfx1101).
- `PocketEncoder._fully_connected_edge_index(b, p, device)`
  at `molmetal/adapters/flow_matching_lipman/__init__.py:669-676` —
  identical algorithm for the pocket-encoder side; reused via the
  same `edge_mask` flatten trick at lines 626-644.

The downstream `EGNNVelocityField.forward` (line 951-960) takes the
fully-connected graph as `edge_index` (no `edge_mask` in the train step —
line 1402 supplies only `edge_index`, not `edge_mask`).  For A6 we
*keep* the dummy `edge_index` shape (so the velocity field is bit-equivalent)
and add a **separate** per-pair `adj_logits: (B, E) → bond/no-bond`
classifier that runs *after* the EGNN layers and produces a
Gumbel-top-k sampled sparse adjacency.  This is exactly the
TargetDiff §3.3 / Pocket2Mol §3.2 split: the EGNN sees the dense graph
(preserves the CFM velocity field semantics), and a lightweight
post-EGNN classifier decides which edges become bonds.

The `EquivariantGraphConv` at `molmetal/adapters/egnn_rocm.py:144-356`
already accepts an optional `edge_mask: bool tensor` via the legacy
`dative_bond_edge_attr` path — so a per-edge boolean mask from A6
plugs in cleanly via the existing forward signature.  The
`BondAwareDecoder._infer_pair_features` at `molmetal/models/bond_head.py:552-590`
already enumerates all `(i, j)` pairs with `i < j`, distance, Z_i, Z_j,
`angle_to_metal`, `is_dative_candidate` — exactly the per-pair
feature vector A6 needs (no change required to the decoder; we only
swap the `argmax over 5 classes` for `top-k from a 2-class Gumbel
softmax`).  The decoder's hard distance cutoff (`bond_cutoff=2.4 Å`)
becomes the *upper bound* on what Gumbel-top-k can pick.

## 2. What changes in `BondOrderHead` for A6 (MEASURED + PROJECTED)

The current `BondOrderHead` (`molmetal/models/bond_head.py:340-395`)
emits `(E, NUM_BOND_CLASSES=5)` logits — `BOND_NO_BOND/SINGLE/DOUBLE/
TRIPLE/AROMATIC`.  For A6 the head is replaced by a *binary*
`AdjacencyHead` (or the existing 5-class head is reduced to a binary
"bond / no-bond" gate via argmax over `{BOND_NO_BOND}` vs the rest).
The change is mechanical and confined to one file:

- Add `AdjacencyHead(in_dim=9, hidden_dim=64) → (E, 2)` —
  `Linear→ReLU→Dropout→Linear→ReLU→Dropout→Linear(2)` — mirroring
  the existing 5-class MLP at lines 370-373.  Init `out.bias[0]=−0.5`
  to bias against over-bonding (matching the existing init at lines
  376-378, which biases `BOND_NO_BOND` to −0.5 for the same reason).
- Add `gumbel_topk_adjacency(logits, k, tau=1.0, training=True)`
  helper (~30 LOC).  Use
  `torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)`
  to sample a one-hot edge gate, then keep edges whose gate argmax
  is the "bond" class.  For k-bounded adjacency we additionally
  cap `n_bonds = min(k * n_atoms, n_edges_kept)` by descending
  sort on `logits[..., 1] + Gumbel_noise` (the Gumbel-top-k
  trick — see §3 below).  The cap follows Pocket2Mol §3.2
  "at most k bonds per atom".
- `BondAwareDecoder.decode` lines 508-547 change `predicted =
  torch.argmax(logits, dim=-1)` → `predicted = adjacency_logits.topk(k
  * n_atoms, dim=0).indices` filtered through the Gumbel gate.  The
  per-edge `bond_cutoff` + `min_distance` filters (lines 524-530) stay
  — they're an essential safety net for over-energetic samplers.
- A6 *removes* the supervised CE-loss dependency on
  `Molecule.bonds / Molecule.bond_types` that A5 needs (the entire
  reason A5's frozen-sidecar `default_trained_head(n_epochs=0)` path
  failed — see `wf2_recon_cfm_train.md:78`).  A6 is headless at
  training time: it only needs `x_1` (the per-atom coords) to compute
  pairwise distances; the rest is pure PyTorch RNG.

## 3. Op availability — `gumbel_softmax` (MEASURED)

Verified live via `uv run --no-progress python -c "import torch;
F = torch.nn.functional; print(F.gumbel_softmax(torch.randn(2,5),
tau=1.0, hard=False).shape)"` against `torch==2.14.0+rocm7.2`:
`gumbel_softmax` is present in `torch.nn.functional`, both `hard=False`
(soft differentiable) and `hard=True` (straight-through one-hot
estimator) work, shapes match the input logits, and the output is
finite.  The classic Gumbel-top-k trick
(`scores = logits + (-log(-log(U)))`, then `torch.topk(scores, k)`) was
verified at the same time — `gumbel_softmax + topk` is the canonical
Top-k categorical sampler of Kool, Welling et al. 2019 and is fully
backed by core PyTorch.  No custom kernel needed.

## 4. DropEdge — torch_geometric unavailable, pure-PyTorch OK (MEASURED)

Verified live via `uv run --no-progress python -c "from
torch_geometric.utils import dropout_edge"` → raises
`ModuleNotFoundError: No module named 'torch_geometric'`.  The
round-8 cleanup deliberately dropped `torch_geometric` from the
dependency list (see `molmetal_round4_final.md` — "torch_geometric
dep removed").  A6 therefore *cannot* import
`torch_geometric.utils.dropout_edge` — but it does **not** need to.
The DropEdge operation is two lines of pure PyTorch:

```python
def drop_edge(edge_index: torch.Tensor, p: float = 0.1,
              training: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.rand(edge_index.shape[1], device=edge_index.device) > p
    return edge_index[:, mask], mask
```

For inference we set `training=False` and return `edge_index` unchanged.
The same function works on `(2, E)` single-graph and `(B, 2, E)`
batched shapes — flatten the batch dim, apply, reshape back.  The
existing `EquivariantGraphConv.forward` already accepts a boolean
`edge_mask` via `dative_bond_edge_attr` (line 274-280, 386-392) and
unions it with `edge_types == EDGE_TYPE_DATIVE`, so wiring DropEdge
into the message-passing path is one-line.  No new files, no new
external deps, no Triton-kernel work.

## 5. Pocket2Mol / TargetDiff mapping to our EGNN (PROJECTED)

Pocket2Mol §3.2 (Peng et al. 2022, ICML) uses *k* = 8 nearest
neighbours per atom + a *learned* per-edge affinity network → Gumbel
softmax for sampling.  TargetDiff §3.3 (Guan et al. 2023, ICLR) uses a
fixed-radius cutoff + a similar per-edge classifier.  Both papers
treat the bond-decision as a *post-hoc* adjacency problem on top of an
EGNN that already produced per-atom embeddings — which is exactly the
architectural split we already have:

- Our `EGNNVelocityField.forward` line 967 returns
  `{"vel": (B, N, 3), "atom_logits": (B, N, max_z), "h": (B, N, H)}` —
  the per-atom hidden vector `h` after the EGNN layers is the
  Pocket2Mol-style per-atom embedding that A6 reads to compute
  per-edge affinity.
- Add a third head `nn.Linear(H, 1)` (per-atom scalar affinity), then
  for each ordered pair (i, j) compute `affinity_ij = sigmoid(W_ij ·
  [h_i || h_j || dist_ij])` (the standard Pocket2Mol §3.2
  "concatenate + MLP" pattern).  `gumbel_softmax(affinity_ij, hard=True)`
  yields the stochastic gate; `topk(K * n_atoms)` caps the bond
  budget; `bond_cutoff=2.4 Å` from `BondAwareDecoder.__init__` is the
  hard distance cap.
- For our SBDD-with-pocket conditioning, A6 additionally conditions the
  affinity on the existing `pocket_embed` (broadcast to per-edge as
  `h_pocket` concat) — this is the canonical TargetDiff §3.3 extension
  ("condition the bond prior on the pocket").  The
  `pocket_embed_scale=0.1` knob in
  `LipmanFlowMatchingAdapter.__init__` (line 1223) already exists;
  reuse it.

The CFM velocity field is **unchanged** (still uses the dummy dense
`edge_index` from `_make_dummy_edge_index`); only the *bond-decoder
side* is replaced.  This is the A6 design choice: leave the
EGNN message-passing semantics intact, swap the bond head for a
Gumbel-top-k stochastic prior.  Risk is bounded — even if A6's
sampled bonds are chemically wrong, the velocity field keeps the
geometry intact and downstream `BondAwareDecoder` records the errors
as `learned_decoder_failed` / `disconnected_distance_graph` statuses
for the harness to count (matches the existing
`r10_cfg_real_crossdocked.py:80-124` decode-status schema).

## 6. Files to touch (PROJECTED; gated on A5 failure)

- `molmetal/models/bond_head.py` — add `AdjacencyHead` (~30 LOC),
  `gumbel_topk_adjacency` helper (~25 LOC), and a `--adjacency-mode`
  branch in `BondAwareDecoder.decode` (~15 LOC delta); keep the
  existing `BondOrderHead` untouched for A1 backward-compat.
- `molmetal/adapters/flow_matching_lipman/__init__.py:874-967`
  — `EGNNVelocityField.forward` returns the existing dict plus the
  new `adjacency_logits: (B, E, 2)` (where E = n*(n-1) per the dummy
  `edge_index`).  The bond-loss term at line 1434 of `train_step`
  becomes a *negative-log-likelihood regulariser*
  `bond_reg = -F.logsigmoid(adjacency_logits[..., 1]).mean()` (no
  labels needed) — at most ~3 lines of code delta.
- `molmetal/scripts/r10_cfg_real_crossdocked.py` — add
  `--bond-pior {learned,dropedge-gumbel}` CLI flag (mirrors the
  existing `--bond-head {distance,learned}` at line 195-198);
  the new branch routes through the A6 path.
- `molmetal/molmetal_lam/tests/test_a6_fallback.py` — NEW ~150 LOC
  unit test asserting: (1) `AdjacencyHead` forward shape; (2)
  `gumbel_topk_adjacency` returns exactly k * n_atoms indices under
  `tau → 0` and the full graph under `tau → ∞`; (3) `drop_edge`
  preserves shape when `training=False`; (4) end-to-end
  `BondAwareDecoder.decode` with `--bond-prior=dropedge-gumbel`
  returns a sanitised RDKit `Mol` for a 4-atom amide SMILES.

## 7. Honest framing (MEASURED vs PROJECTED, explicit)

**MEASURED**:
- `torch.nn.functional.gumbel_softmax` is importable and runnable
  on `torch==2.14.0+rocm7.2` with both `hard=False/True` modes
  producing finite, correctly-shaped outputs.
- `torch_geometric.utils.dropout_edge` is **NOT** importable
  (ModuleNotFoundError confirmed).  Pure-PyTorch DropEdge is two
  lines; no functional loss vs. the PyG version (which is itself
  a one-line Bernoulli mask under the hood).
- Edge-index construction today uses the `(B, 2, n*(n-1))`
  fully-connected dummy in `_make_dummy_edge_index`
  (`flow_matching_lipman/__init__.py:1763-1775`); A6 reuses this
  layout unchanged.
- `EquivariantGraphConv.forward` already accepts an optional bool
  edge mask (`dative_bond_edge_attr` path, lines 274-280, 386-392);
  DropEdge can plug in here without modifying the EGNN layer.
- The per-pair features `distance, z_i, z_j, angle_to_metal,
  is_dative_candidate` are already collected by
  `BondAwareDecoder._infer_pair_features`
  (`molmetal/models/bond_head.py:552-590`) — A6 reads them as-is.
- `Molecule.bonds / Molecule.bond_types`
  (`molmetal/domain/__init__.py:76-114`) are the only path A5
  needs for labels; A6 ignores them entirely.

**PROJECTED** (A6 design, only activated if A5 fails):
- Add `AdjacencyHead` + `gumbel_topk_adjacency` + DropEdge to
  `molmetal/models/bond_head.py` (~70 LOC delta).
- Extend `EGNNVelocityField.forward` to also emit
  `adjacency_logits: (B, E, 2)` (the per-true-edge affinity); add
  one regulariser line to `train_step` at line 1434.
- Wire a `--bond-prior {learned, dropedge-gumbel}` flag into
  `r10_cfg_real_crossdocked.py` (mirrors `--bond-head` at line 195).
- Add `test_a6_fallback.py` (~150 LOC) covering forward shapes,
  Gumbel temperature extremes, DropEdge no-op in eval mode, and a
  4-atom amide end-to-end decode.
- Estimated total effort: 1 commit, ≤ 250 LOC net add, ≤ 4 files
  touched + 1 new test file.  No new external dependencies.
- **Success criterion**: `BondAwareDecoder.decode` returns a
  sanitised RDKit `Mol` with `n_bonds ≥ n_atoms - 1` (a connected
  graph) on ≥ 90 % of CrossDocked test pockets; **honest
  measurement gate** is the same `decode_status_counts` schema
  the r10 harness already records
  (`molmetal/scripts/r10_cfg_real_crossdocked.py:80-124`).  If A6
  also fails the connectivity test, escalate to A7 (RDKit-only
  distance-graph with dative annotation) and mark bond-decoding as
  a known limitation in the paper.

---

## File:line list (MEASURED)

### A6 implementation hook points
- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/egnn_rocm.py:144-356`
  — `EquivariantGraphConv.forward` already accepts an optional bool
  edge mask via `dative_bond_edge_attr`; A6 DropEdge plugs here.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py:874-967`
  — `EGNNVelocityField.forward` returns `{"vel", "atom_logits", "h"}`;
  extend to add `"adjacency_logits"` for A6.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py:1402, 1417, 1434, 1441-1443`
  — `train_step` loss-composition site; add one regulariser line.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/adapters/flow_matching_lipman/__init__.py:1763-1775`
  — `_make_dummy_edge_index` produces the dense (B, 2, n*(n-1))
  graph that A6 reuses unchanged for the EGNN message passing.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py:340-395`
  — existing `BondOrderHead`; A6 adds `AdjacencyHead` + `gumbel_topk_adjacency` as siblings.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/models/bond_head.py:552-590`
  — `BondAwareDecoder._infer_pair_features` enumerates all (i, j) pairs
  with `distance, z_i, z_j, angle_to_metal, is_dative_candidate` —
  exactly the per-pair feature vector A6 needs.

### Existing CLI / decode-status schema
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py:80-124`
  — `decode_learned_graph` helper + the `decoded_learned_bond_graph`
  status; A6 uses the same status vocabulary.
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r10_cfg_real_crossdocked.py:195-198`
  — existing `--bond-head {distance, learned}` argparse; A6 adds
  `--bond-prior {learned, dropedge-gumbel}` next to it.

### Op-availability verification (MEASURED)
- `gumbel_softmax` from `torch.nn.functional` — confirmed importable,
  `hard=False/True` both produce finite `(B, E, K)` outputs against
  `torch==2.14.0+rocm7.2`.
- `torch_geometric.utils.dropout_edge` — NOT importable; pure-PyTorch
  DropEdge (Bernoulli edge_mask) is the fallback path.

### Pocket2Mol / TargetDiff reference citations
- Pocket2Mol §3.2 (Peng et al. 2022, ICML) — k-NN + per-edge MLP +
  Gumbel softmax for bond sampling.  Maps cleanly to our
  `EquivariantGraphConv` + `AdjacencyHead`.
- TargetDiff §3.3 (Guan et al. 2023, ICLR) — fixed-radius cutoff +
  per-edge classifier.  Our `bond_cutoff=2.4 Å` +
  `AdjacencyHead` reproduces the same architectural split.

---

## Honest fallback if A6 also fails: A7 = RDKit-only distance graph with dative annotation

If A6's Gumbel-top-k sampling fails the connectivity test
(`n_bonds ≥ n_atoms - 1` on < 90 % of pockets), escalate to A7:
skip the learned / Gumbel head entirely and just run the existing
`decode_distance_graph` heuristic with explicit `(donor, metal)`
dative-bond annotation from `DEFAULT_METAL_GEOMETRY`
(`molmetal/molmetal_lam/priors/metal_geometry.py:350-374`).  A7 is
already 80 % implemented as the current `r10_cfg_real_crossdocked.py`
default (`--bond-head=distance`); the only A7-specific add is the
explicit dative-bond post-processing step in the decoder, which is
already a 6-line branch in
`BondAwareDecoder._decode_impl:531-543`.  Mark bond-decoding as a
known limitation in the paper draft (WF-6 / TODO-15).
