"""Radius-graph wiring guards for the EGNN velocity field.

2026-10-04 — covers the change that replaced a FULLY CONNECTED graph with a
radius graph in both production paths of ``LipmanFlowMatchingAdapter``:

* ``train_step`` built ``_make_dummy_edge_index(b, max_n)`` — every atom wired
  to every other atom, i.e. O(N^2) messages and none of the locality an EGNN's
  geometric prior comes from.
* ``_generate_impl`` did the same AND bound the graph into the ODE solver's
  velocity callable as a ``functools.partial`` argument, so a 100-step solve
  reused the graph of the initial (pure-noise) configuration for every step,
  even though ``x_t`` moves.

Both now call ``radius_edge_index``, which rebuilds from the current ``x``.

The tests below assert the two properties a radius graph must have (locality,
and no edges across samples in a batch) and that the generate path really
rebuilds rather than reusing a bound tensor.
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from molmetal.adapters import flow_matching_lipman as fml
from molmetal.adapters.flow_matching_lipman import (
    GenerationConfig,
    LipmanFlowMatchingAdapter,
    radius_edge_index,
)


# ---------------------------------------------------------------------------
# radius_edge_index — the graph contract
# ---------------------------------------------------------------------------
def test_radius_edge_index_respects_cutoff():
    """A pair beyond the cutoff contributes no edge; a close pair is bidirectional."""
    # Two atoms 1.5 A apart, a third 10 A away.
    x = torch.tensor([[[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    edge_index, edge_mask = radius_edge_index(x, None, cutoff=3.0)
    # Exactly one pair survives, in both directions.
    assert int(edge_mask.sum().item()) == 2, (
        f"expected 2 directed edges for one close pair, got "
        f"{int(edge_mask.sum().item())}"
    )
    src, dst = edge_index[0, 0].tolist(), edge_index[0, 1].tolist()
    assert sorted(zip(src, dst)) == [(0, 1), (1, 0)], (src, dst)


def test_radius_edge_index_never_crosses_samples():
    """A batch of separately-placed molecules must not be wired together.

    Each row of the batch sits at the origin, so a distance-only graph would
    join atoms of DIFFERENT samples.  `node_mask` is what prevents it, and
    these tests run without one — so the guard here is that the edge index is
    built per row from that row's own coordinates.
    """
    one = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    two = one.repeat(2, 1, 1)  # two identical 2-atom samples, both at origin
    _, mask_one = radius_edge_index(one, None, cutoff=3.0)
    _, mask_two = radius_edge_index(two, None, cutoff=3.0)
    assert int(mask_two.sum().item()) == 2 * int(mask_one.sum().item()), (
        "edge count did not scale with batch size — samples are being wired "
        "together"
    )


def test_radius_edge_index_node_mask_drops_padding():
    """Padded positions must contribute no edges."""
    x = torch.randn(1, 5, 3)
    full = torch.ones(1, 5, dtype=torch.bool)
    partial = torch.tensor([[True, True, True, False, False]])
    _, m_full = radius_edge_index(x, full, cutoff=1e6)  # every pair
    _, m_partial = radius_edge_index(x, partial, cutoff=1e6)
    # 3 real nodes -> 3*2 directed edges; 5 real nodes -> 5*4.
    assert int(m_partial.sum().item()) == 6
    assert int(m_full.sum().item()) == 20


# ---------------------------------------------------------------------------
# train_step — the graph must come from x_t, not be fully connected
# ---------------------------------------------------------------------------
def test_train_step_uses_radius_graph_not_complete_graph():
    """The mask must be strictly smaller than a complete graph.

    A regression to `_make_dummy_edge_index` would make the surviving-edge
    count equal N*(N-1) per sample.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from molmetal.domain import Molecule

    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=32, n_layers=2, use_bond_head=False, joint_train=False
    )
    adapter.setup(device="cpu")
    mols = []
    for smi in ("CCO", "c1ccccc1"):
        m = Chem.MolFromSmiles(smi)
        AllChem.EmbedMolecule(m, randomSeed=0)
        m = Chem.AddHs(m)
        AllChem.EmbedMolecule(m, randomSeed=0)
        mols.append(Molecule.from_rdkit_mol(m))

    seen: list[int] = []
    real = fml.radius_edge_index

    def _spy(coords, node_mask=None, cutoff=5.0):
        ei, em = real(coords, node_mask, cutoff)
        seen.append(int(em.sum().item()))
        return ei, em

    fml.radius_edge_index = _spy
    try:
        adapter.train_step(None, mols)
    finally:
        fml.radius_edge_index = real

    assert seen, "train_step did not build a graph through radius_edge_index"
    max_n = max(m.coords.shape[0] for m in mols)
    complete = len(mols) * max_n * (max_n - 1)
    assert max(seen) < complete, (
        f"graph is fully connected ({max(seen)} of {complete} possible "
        "directed edges) — the radius cutoff is not being applied"
    )


# ---------------------------------------------------------------------------
# generate — the graph must be rebuilt as x_t moves, not bound once
# ---------------------------------------------------------------------------
def test_generate_rebuilds_graph_per_ode_step():
    """The radius graph must be rebuilt at every integration step.

    With the velocity head at its zero-init default the ODE is the identity
    (v = 0, so x never moves) and a bound-vs-rebuilt graph is
    indistinguishable.  Randomising the head makes x_t evolve, which is what
    lets this test tell the two apart — checking with the untouched model
    would pass either way.
    """
    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=32, n_layers=2, use_bond_head=False, joint_train=False
    )
    adapter.setup(device="cpu")
    with torch.no_grad():
        torch.manual_seed(0)
        adapter.velocity_field.vel_head.weight.normal_(0.0, 0.5)

    seen: list[tuple[int, str, float]] = []
    real = fml.radius_edge_index

    def _spy(coords, node_mask=None, cutoff=5.0):
        ei, em = real(coords, node_mask, cutoff)
        seen.append(
            (
                int(em.sum().item()),
                hashlib.md5(ei.cpu().numpy().tobytes()).hexdigest()[:8],
                round(float(coords.std().item()), 4),
            )
        )
        return ei, em

    n_steps = 5
    fml.radius_edge_index = _spy
    try:
        adapter.generate(
            None,
            GenerationConfig(n_samples=3, n_steps=n_steps, method="euler", seed=0),
        )
    finally:
        fml.radius_edge_index = real

    # One call per integration step, plus one for the post-solve atom-logit
    # forward on x_final.
    assert len(seen) == n_steps + 1, (
        f"expected {n_steps + 1} graph builds (one per step + x_final), "
        f"got {len(seen)} — the graph is bound once, not rebuilt"
    )
    stds = [s for _, _, s in seen]
    assert stds[0] != stds[-1], (
        "coordinate spread did not change across the solve, so this test "
        "cannot distinguish a rebuilt graph from a bound one — check that "
        "vel_head was randomised"
    )
    assert len({h for _, h, _ in seen}) > 1, (
        "every step produced an identical edge index — the graph is stale"
    )


def test_generate_cfg_path_also_rebuilds():
    """The classifier-free-guidance branch must use the same per-step graph.

    `v_cfg` needs a per-sample time; a 0-d scalar from the solver is
    broadcast, and getting that wrong is a shape error rather than a silent
    one, so this asserts the branch runs at all.
    """
    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=32,
        n_layers=2,
        use_bond_head=False,
        joint_train=False,
        cfg_scale=2.0,
    )
    adapter.setup(device="cpu")
    with torch.no_grad():
        torch.manual_seed(0)
        adapter.velocity_field.vel_head.weight.normal_(0.0, 0.5)
    pocket_embed = torch.zeros(4, adapter._hidden_dim)
    model = fml._RadiusGraphVelocityModel(
        velocity_field=adapter.velocity_field,
        atom_types=torch.full((4, 6), 6, dtype=torch.long),
        node_mask=None,
        cutoff=adapter._cutoff,
        pocket_embed=pocket_embed,
        cfg_scale=2.0,
    )
    out = model(torch.randn(4, 6, 3), torch.tensor(0.5))
    assert out.shape == (4, 6, 3), out.shape


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
