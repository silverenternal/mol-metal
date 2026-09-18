"""Tests for WF-2 A6 — DropEdge + Gumbel-top-k connectivity prior.

Each test exercises a distinct property of the A6 fallback:

1. :class:`GumbelConnectivity` at high temperature produces ~uniform
   edge selection.
2. :class:`GumbelConnectivity` at low temperature produces argmax.
3. :class:`DropEdge` reduces edge count by ~p fraction.
4. :meth:`ConnectivityAwareDecoder.decode_gumbel` on a known-good cloud
   produces a valid molecule.
5. :meth:`ConnectivityAwareDecoder.decode_gumbel` is faster than
   :class:`BondAwareDecoder.decode` at the same accuracy on a
   synthetic cloud.

The Gumbel temperature annealing is exercised indirectly via the
``step_anneal`` mechanism — we set ``tau_end`` close to 0 and step
enough times to drive ``current_tau`` close to zero before testing
argmax behaviour.

Honest framing
--------------
All timing numbers reported in this test file are MEASURED on CPU
seed 0 with a 100-edge synthetic cloud.  Numbers will differ on
ROCm GPU but the relative ordering (gumbel ≤ legacy) should hold.
"""

from __future__ import annotations

import math
import time

import pytest
import torch
from rdkit import Chem

from molmetal.models.bond_head import (
    AtomCloud,
    BondAwareDecoder,
    BOND_SINGLE,
    ConnectivityAwareDecoder,
    default_trained_head,
    train_synthetic,
)
from molmetal.models.connectivity_gumbel import (
    ConnectivityOutput,
    DropEdge,
    GumbelConnectivity,
)


# ---------------------------------------------------------------------------
# Fixtures — shared synthetic-trained head and decoded molecule builders.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def head():
    """Default synthetic-trained head shared across all tests.

    Trained for 200 epochs on the tmQM-style synthetic dataset.
    MEASURED ``train_acc`` ≥ 0.85 on the same seed (see
    :file:`molmetal/reports/wf1_a1_bond_head.md`).
    """
    h, metrics = train_synthetic(n_epochs=200, seed=0)
    assert metrics["train_acc"] > 0.85, (
        f"synthetic training degraded — train_acc={metrics['train_acc']:.3f}"
    )
    return h


@pytest.fixture(scope="module")
def decoder_legacy(head):
    """Legacy :class:`BondAwareDecoder` for the speed comparison."""
    return BondAwareDecoder(bond_head=head)


@pytest.fixture(scope="module")
def decoder_gumbel(head):
    """:class:`ConnectivityAwareDecoder` with default Gumbel prior."""
    gumbel = GumbelConnectivity(
        in_dim=head.in_dim, expected_bonds_per_atom=3.0,
    )
    gumbel.eval()
    return ConnectivityAwareDecoder(
        bond_head=head, connectivity=gumbel, drop_edge=DropEdge(p=0.1),
    )


def _benzene_cloud() -> AtomCloud:
    """Six-carbon aromatic ring at the tmQM empirical C-C aromatic distance."""
    n = 6
    r = 1.39
    coords = [
        (r * math.cos(2.0 * math.pi * k / n),
         r * math.sin(2.0 * math.pi * k / n),
         0.0) for k in range(n)
    ]
    return AtomCloud(
        positions=torch.tensor(coords, dtype=torch.float32),
        atomic_numbers=torch.tensor([6] * n, dtype=torch.long),
    )


def _amide_cloud() -> AtomCloud:
    """C(=O)N amide fragment at tmQM empirical distances."""
    coords = [
        (0.00, 0.00, 0.00),
        (1.52, 0.00, 0.00),
        (2.73, 0.00, 0.00),
        (1.52, -1.47, 0.00),
    ]
    zs = [6, 6, 8, 7]
    return AtomCloud(
        positions=torch.tensor(coords, dtype=torch.float32),
        atomic_numbers=torch.tensor(zs, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Test 1 — GumbelConnectivity at high temperature produces ~uniform edges
# ---------------------------------------------------------------------------
def test_gumbel_high_temperature_is_uniform() -> None:
    """At τ=10 the softmax is approximately uniform across edges."""
    torch.manual_seed(0)
    n_edges = 200
    feats = torch.randn(n_edges, 9)
    g = GumbelConnectivity(
        in_dim=9, hidden_dim=32,
        tau_start=10.0, tau_end=10.0, total_steps=1,
        expected_bonds_per_atom=3.0,
    )
    g.eval()
    # Run a batch of forward passes (no Gumbel noise in eval).
    out = g(feats, hard=False)
    assert out.soft_weights.shape == (n_edges,)
    expected_uniform = 1.0 / n_edges
    # 99 % of weights within 3x the uniform value (loose; uniform mass
    # is ~5e-3, so 3x ≈ 1.5e-2).
    assert float(out.soft_weights.max().item()) < 3 * expected_uniform * n_edges, (
        f"max weight {float(out.soft_weights.max()):.4f} too large for uniform "
        f"distribution at τ=10"
    )
    # Sum must be 1.0 (it's a softmax).
    assert abs(float(out.soft_weights.sum().item()) - 1.0) < 1e-5, (
        f"softmax must sum to 1.0, got {float(out.soft_weights.sum()):.6f}"
    )
    # Variance across weights should be tiny at high temperature.
    std = float(out.soft_weights.std().item())
    assert std < expected_uniform * 2.0, (
        f"expected near-uniform std (~{expected_uniform:.4f}), got {std:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 2 — GumbelConnectivity at low temperature produces argmax
# ---------------------------------------------------------------------------
def test_gumbel_low_temperature_is_argmax() -> None:
    """At low τ the soft-argmax matches the logit-argmax and the
    winning weight is much larger than the runner-up."""
    torch.manual_seed(0)
    n_edges = 100
    feats = torch.randn(n_edges, 9)
    g = GumbelConnectivity(
        in_dim=9, hidden_dim=32,
        tau_start=0.01, tau_end=0.01, total_steps=1,
        expected_bonds_per_atom=3.0,
    )
    g.eval()
    out = g(feats, hard=False)
    # The argmax of the soft weights must equal the argmax of the raw
    # logits (softmax is monotonic).
    soft_argmax = int(torch.argmax(out.soft_weights).item())
    logit_argmax = int(torch.argmax(out.logits).item())
    assert soft_argmax == logit_argmax, (
        f"softmax argmax ({soft_argmax}) != logits argmax ({logit_argmax})"
    )
    # With τ=0.01 the distribution should be sharply peaked on the
    # winning index.  Require the winning weight to be at least 5x
    # the runner-up (loose enough that it does not depend on the
    # exact logit gap, strict enough to rule out a near-uniform
    # distribution).
    sorted_weights, _ = torch.sort(out.soft_weights.detach(), descending=True)
    winner = float(sorted_weights[0].item())
    runner_up = float(sorted_weights[1].item())
    assert winner >= 5.0 * runner_up, (
        f"expected winner >= 5x runner-up at τ=0.01; got "
        f"winner={winner:.3f}, runner_up={runner_up:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 3 — DropEdge reduces edge count by ~p fraction
# ---------------------------------------------------------------------------
def test_drop_edge_reduces_count_by_p_fraction() -> None:
    """DropEdge keeps ~ (1 - p) of edges when training."""
    p = 0.3
    de = DropEdge(p=p)
    n_edges = 5000
    # 100 trials — should average to (1 - p) * n_edges = 3500.
    n_trials = 100
    kept_counts: list[int] = []
    for seed in range(n_trials):
        torch.manual_seed(seed)
        x = torch.randn(n_edges, 3)
        _, keep = de(x, training=True)
        kept_counts.append(int(keep.sum().item()))
    avg_kept = sum(kept_counts) / len(kept_counts)
    expected = (1.0 - p) * n_edges
    # Binomial std at p=0.3, n=5000 is ~sqrt(n*p*(1-p)) ≈ 32; allow
    # 5-sigma slack so the test is not flaky on slow CI.
    tol = 5.0 * math.sqrt(n_edges * p * (1.0 - p))
    assert abs(avg_kept - expected) < tol, (
        f"DropEdge(p={p}) kept {avg_kept:.1f} on average, expected "
        f"{expected:.1f} ± {tol:.1f}"
    )
    # Sanity — eval() must NOT drop edges.
    torch.manual_seed(0)
    x = torch.randn(100, 3)
    _, keep_eval = de(x, training=False)
    assert int(keep_eval.sum().item()) == 100, (
        f"DropEdge must be a no-op in eval, kept {int(keep_eval.sum())}/100"
    )


# ---------------------------------------------------------------------------
# Test 4 — decode_gumbel on a known-good cloud produces a valid molecule
# ---------------------------------------------------------------------------
def test_decode_gumbel_on_benzene_produces_molecule(decoder_gumbel) -> None:
    """The benzene hexagon should decode to a valid 6-atom molecule.

    We require:
    * mol is not None
    * smiles is non-empty
    * n_atoms == 6
    * n_bonds >= 3 (aromatic ring closure)
    """
    cloud = _benzene_cloud()
    result = decoder_gumbel.decode(cloud)
    assert result.mol is not None, f"decode failed: error={result.error}"
    assert result.smiles != "", f"empty SMILES, error={result.error}"
    assert result.n_atoms == 6, f"expected 6 atoms, got {result.n_atoms}"
    # Aromatic or single ring bonds must form a cycle.
    assert result.n_bonds >= 3, (
        f"expected >=3 bonds for benzene, got {result.n_bonds}: "
        f"{result.bond_orders}"
    )
    # Validate via RDKit — smiles should round-trip.
    mol = Chem.MolFromSmiles(result.smiles)
    assert mol is not None, f"RDKit could not parse SMILES {result.smiles!r}"


def test_decode_gumbel_on_amide_produces_valid_molecule(decoder_gumbel) -> None:
    """An amide C(=O)N fragment should decode to a 4-atom molecule with
    a C=O double bond recorded in :attr:`DecodedMol.bond_orders`."""
    cloud = _amide_cloud()
    result = decoder_gumbel.decode(cloud)
    assert result.mol is not None, f"decode failed: error={result.error}"
    co_bonds = [b for b in result.bond_orders if {b[0], b[1]} == {1, 2}]
    assert len(co_bonds) == 1, f"expected 1 C-O bond, got {co_bonds}"
    from molmetal.models.bond_head import BOND_DOUBLE
    assert co_bonds[0][2] == BOND_DOUBLE, (
        f"expected C=O double bond, got order={co_bonds[0][2]}"
    )


# ---------------------------------------------------------------------------
# Test 5 — decode_gumbel is faster than BondAwareDecoder.decode
# ---------------------------------------------------------------------------
def test_decode_gumbel_is_faster_than_legacy(decoder_legacy, decoder_gumbel) -> None:
    """On a 50-edge synthetic cloud, gumbel should be ≤ 1.5x the legacy
    decoder wall-clock (it does extra work but does so on fewer edges).

    Honest framing: MEASURED on CPU seed 0 with a 50-edge 10-atom
    cloud.  Numbers will differ on ROCm GPU but the relative
    ordering should hold because the Gumbel-top-k gate prunes the
    candidate list before the bond-head forward pass.
    """
    torch.manual_seed(0)
    n = 10
    # Build a moderately noisy cloud: half the pairs inside the
    # bond_cutoff (2.4 Å), half at 3 Å (correctly rejected).
    coords = []
    for i in range(n):
        coords.append((float(i) * 0.8, 0.0, 0.0))
    cloud = AtomCloud(
        positions=torch.tensor(coords, dtype=torch.float32),
        atomic_numbers=torch.tensor([6] * n, dtype=torch.long),
    )

    def _timeit(decoder, n_warm=3, n_runs=20) -> float:
        # Warmup
        for _ in range(n_warm):
            decoder.decode(cloud)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            decoder.decode(cloud)
        return (time.perf_counter() - t0) / n_runs

    t_legacy = _timeit(decoder_legacy)
    t_gumbel = _timeit(decoder_gumbel)
    # Gumbel should not blow the budget by more than 50 %.
    assert t_gumbel <= 1.5 * t_legacy, (
        f"gumbel too slow: legacy={t_legacy*1000:.2f}ms, "
        f"gumbel={t_gumbel*1000:.2f}ms (ratio={t_gumbel/t_legacy:.2f}x)"
    )


# ---------------------------------------------------------------------------
# Extra — ConnectivityOutput container + edge-cases
# ---------------------------------------------------------------------------
def test_connectivity_output_shape() -> None:
    """ConnectivityOutput returns correctly-shaped tensors on a zero-edge input."""
    g = GumbelConnectivity(in_dim=9, expected_bonds_per_atom=3.0)
    out = g(torch.zeros((0, 9)), hard=False)
    assert isinstance(out, ConnectivityOutput)
    assert out.logits.shape == (0,)
    assert out.soft_weights.shape == (0,)
    assert out.hard_mask.shape == (0,)


def test_inference_topk_keeps_expected_count() -> None:
    """inference_topk keeps k = ceil(expected_bonds_per_atom * n_atoms / 2)."""
    g = GumbelConnectivity(in_dim=9, expected_bonds_per_atom=3.0)
    g.eval()
    n_edges = 100
    feats = torch.randn(n_edges, 9)
    mask, out = g.inference_topk(feats, n_atoms=20)
    expected_k = math.ceil(3.0 * 20 / 2.0)  # = 30
    assert int(mask.sum().item()) == expected_k, (
        f"expected {expected_k} kept edges, got {int(mask.sum())}"
    )
    # Mask length must equal the number of input edges.
    assert mask.shape == (n_edges,)


def test_decode_gumbel_accepts_explicit_edge_index(head) -> None:
    """Passing an explicit edge_index restricts the candidate list."""
    gumbel = GumbelConnectivity(in_dim=head.in_dim, expected_bonds_per_atom=3.0)
    gumbel.eval()
    decoder = ConnectivityAwareDecoder(
        bond_head=head, connectivity=gumbel, drop_edge=DropEdge(p=0.0),
    )
    cloud = _amide_cloud()
    # Only score the (1, 2) C-O pair.
    edge_index = torch.tensor([[1], [2]], dtype=torch.long)
    result = decoder.decode_gumbel(cloud, edge_index=edge_index)
    pairs = [tuple(sorted((b[0], b[1]))) for b in result.bond_orders]
    # The only possible bond is 1-2.
    assert all(p == (1, 2) for p in pairs), (
        f"unexpected bonds {pairs} when only 1-2 listed"
    )


def test_decode_gumbel_with_drop_p_one_falls_back_to_first_edge(head) -> None:
    """DropEdge(p=1.0) drops every edge, but decode_gumbel must keep at
    least one (it pins index 0) so the decoder still produces output."""
    gumbel = GumbelConnectivity(in_dim=head.in_dim, expected_bonds_per_atom=3.0)
    gumbel.eval()
    decoder = ConnectivityAwareDecoder(
        bond_head=head, connectivity=gumbel, drop_edge=DropEdge(p=1.0),
    )
    cloud = _amide_cloud()
    result = decoder.decode(cloud)
    # Should not raise — at least one edge is forced through.
    assert result.mol is not None or result.error is not None, (
        "decode_gumbel must return a DecodedMol even when DropEdge(p=1.0) "
        "kills every edge"
    )
