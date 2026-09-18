"""Pytest: WF-CFM-Rescue Phase 1 — YuelBond decoder unit tests.

Five tests covering the lit-anchored decoder swap:

1. ``test_yuelbond_head_forward_returns_correct_shape`` — forward
   pass returns ``(E, 5)`` logits on a simple cisplatin-like cloud.
2. ``test_yuelbond_decoder_featurise_pairs_match_edge_count`` — the
   ``featurise`` wrapper emits ``PairFeature`` whose ``edge_index``
   has 2 columns (i, j) and matches the pair count.
3. ``test_yuelbond_decoder_handles_empty_cloud`` — degenerate inputs
   (``N=0`` or ``N=1``) return ``decode_succeeded=False`` without
   crashing.
4. ``test_yuelbond_decoder_distance_cutoff_filters_long_pairs`` — pairs
   beyond the cutoff (3.5 Å) are NOT included in the edge list.
5. ``test_yuelbond_decoder_dative_flag_for_N_Pt_pair`` — the dative
   candidate flag is True when (Z_i=7, Z_j=78).

All tests run on CPU in <2 s — no GPU or external evaluators required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    YuelBondDecoder,
    YuelBondDecoderHead,
)
from molmetal.models.bond_head import (
    NUM_BOND_CLASSES,
    AtomCloud,
    BOND_NO_BOND,
    BOND_SINGLE,
)


def _cisplatin_like_cloud() -> AtomCloud:
    """Build a tiny Pt(II) cloud with 4 ligands at typical bond distances.

    Layout: Pt at origin, 4 donors at ~2.05 Å in a square-planar motif.
    """
    positions = torch.tensor([
        [0.0, 0.0, 0.0],            # Pt
        [2.05, 0.0, 0.0],            # Cl  (right)
        [-2.05, 0.0, 0.0],           # Cl  (left)
        [0.0, 2.05, 0.0],            # NH3 (top)
        [0.0, -2.05, 0.0],           # NH3 (bottom) — using N for dative
    ], dtype=torch.float32)
    atomic_numbers = torch.tensor([78, 17, 17, 7, 7], dtype=torch.long)
    return AtomCloud(positions=positions, atomic_numbers=atomic_numbers)


def _distant_cloud() -> AtomCloud:
    """Build a 4-atom cloud with all pairs > 5 Å — no bonds expected."""
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
    ], dtype=torch.float32)
    atomic_numbers = torch.tensor([6, 6, 6, 6], dtype=torch.long)
    return AtomCloud(positions=positions, atomic_numbers=atomic_numbers)


# ---------------------------------------------------------------------------
# Test 1: forward pass returns (E, 5) logits
# ---------------------------------------------------------------------------
def test_yuelbond_head_forward_returns_correct_shape():
    """YuelBondDecoderHead forward on a 5-atom cloud returns (E, 5) logits.

    E is the number of upper-triangular pairs within the cutoff
    (default 3.5 Å).  For cisplatin-like with 4 donors at 2.05 Å and
    Cl-Cl pair at ~4.1 Å (which is > cutoff), we expect:
      - 4 Pt-donor pairs (within cutoff)
      - 4 donor-donor pairs (within cutoff if distance ≤ 3.5 Å;
        the perpendicular Cl-Cl is 4.1 > cutoff so excluded)
      - Total E in [4, 10]
    """
    torch.manual_seed(0)
    head = YuelBondDecoderHead(hidden_dim=32, n_layers=2)
    cloud = _cisplatin_like_cloud()
    out = head(cloud)
    assert isinstance(out, tuple), (
        "YuelBondDecoderHead.forward must return (logits, (i_idx, j_idx))"
    )
    logits, (i_idx, j_idx) = out
    assert logits.dim() == 2, f"logits should be (E, 5), got shape {logits.shape}"
    assert logits.shape[-1] == NUM_BOND_CLASSES, (
        f"logits last dim should be {NUM_BOND_CLASSES}, got {logits.shape[-1]}"
    )
    assert i_idx.shape == j_idx.shape, "i_idx and j_idx must have same shape"
    assert i_idx.numel() == logits.shape[0], (
        f"logits.shape[0]={logits.shape[0]} must match edge count {i_idx.numel()}"
    )
    # Should be at least 4 Pt-donor pairs
    assert i_idx.numel() >= 4, (
        f"Expected >= 4 edges (Pt-donor pairs), got {i_idx.numel()}"
    )
    # No bond class bias should be slightly negative (init check)
    assert logits[:, BOND_NO_BOND].mean() < 0.0, (
        "BOND_NO_BOND class should be biased negative at init"
    )


# ---------------------------------------------------------------------------
# Test 2: featurise wrapper emits PairFeature with matching edge count
# ---------------------------------------------------------------------------
def test_yuelbond_decoder_featurise_pairs_match_edge_count():
    """YuelBondDecoder.featurise emits PairFeature with consistent edge count.

    The returned :class:`PairFeature` has ``edge_index`` of shape ``(2, E)``
    and the other arrays of length ``E``.  ``decode_succeeded=True`` for a
    non-empty cloud.
    """
    torch.manual_seed(0)
    decoder = YuelBondDecoder(hidden_dim=32, n_layers=2)
    cloud = _cisplatin_like_cloud()
    result = decoder.featurise(cloud)
    assert result.decode_succeeded, f"decode should succeed, error={result.error}"
    assert result.pair_features is not None
    assert result.logits.shape[0] == result.edge_index.shape[1], (
        f"logits rows ({result.logits.shape[0]}) must match edge_index cols "
        f"({result.edge_index.shape[1]})"
    )
    # PairFeature arrays must agree on E
    e = result.pair_features.edge_index.shape[1]
    assert result.pair_features.distance.shape[0] == e
    assert result.pair_features.z_i.shape[0] == e
    assert result.pair_features.z_j.shape[0] == e
    # Edge_index contains valid atom indices
    n = cloud.atomic_numbers.shape[0]
    assert (result.edge_index >= 0).all()
    assert (result.edge_index < n).all()


# ---------------------------------------------------------------------------
# Test 3: degenerate inputs handled without crashing
# ---------------------------------------------------------------------------
def test_yuelbond_decoder_handles_empty_cloud():
    """Empty (N=0) and single-atom (N=1) clouds return decode_succeeded=False.

    Both cases have no candidate pairs.  The decoder must NOT crash with
    an index error or shape mismatch.
    """
    decoder = YuelBondDecoder()
    # N=0
    cloud_empty = AtomCloud(
        positions=torch.zeros(0, 3),
        atomic_numbers=torch.zeros(0, dtype=torch.long),
    )
    result = decoder.featurise(cloud_empty)
    assert not result.decode_succeeded, "N=0 cloud should not 'succeed'"
    assert result.error is not None
    # N=1
    cloud_one = AtomCloud(
        positions=torch.zeros(1, 3),
        atomic_numbers=torch.tensor([6], dtype=torch.long),
    )
    result_one = decoder.featurise(cloud_one)
    # N=1 should not crash; may return decode_succeeded=False with empty edges
    assert result_one.edge_index.shape[1] == 0, (
        f"N=1 should have 0 edges, got {result_one.edge_index.shape[1]}"
    )


# ---------------------------------------------------------------------------
# Test 4: distance cutoff filters long pairs
# ---------------------------------------------------------------------------
def test_yuelbond_decoder_distance_cutoff_filters_long_pairs():
    """Pairs beyond the cutoff are NOT included in the edge list.

    With the default cutoff of 3.5 Å and atoms at 10 Å apart, NO pairs
    should be returned.
    """
    decoder = YuelBondDecoder()
    cloud = _distant_cloud()
    result = decoder.featurise(cloud)
    # All pairs are 10*sqrt(2) ≈ 14.1 Å (diagonal) or 10 Å (axis);
    # all far beyond the 3.5 Å cutoff.
    assert result.edge_index.shape[1] == 0, (
        f"All 4 atoms are >10 Å apart; cutoff=3.5 Å should yield 0 edges, "
        f"got {result.edge_index.shape[1]}"
    )
    # The error should mention empty_edge_list OR the head should return
    # (logits shape (0, 5), empty indices)
    if not result.decode_succeeded:
        assert result.error in ("empty_edge_list", None) or "empty" in (result.error or "")


# ---------------------------------------------------------------------------
# Test 5: dative candidate flag for (N -> Pt) pair
# ---------------------------------------------------------------------------
def test_yuelbond_decoder_dative_flag_for_N_Pt_pair():
    """Dative flag is True when Z_i in {7,8,16} and Z_j = 78 (Pt).

    On the cisplatin cloud, the 2 Pt-NH3 pairs should have
    ``is_dative_candidate=True``; the 2 Pt-Cl pairs should have
    ``is_dative_candidate=False`` (Cl is not in {7,8,16}).
    """
    decoder = YuelBondDecoder()
    cloud = _cisplatin_like_cloud()
    result = decoder.featurise(cloud)
    assert result.decode_succeeded
    pf = result.pair_features
    e = pf.edge_index.shape[1]
    # Walk each edge and check the dative flag
    n_dative = int(pf.is_dative_candidate.sum().item())
    # We have 2 NH3 -> Pt pairs, expect n_dative >= 2 (other pairs in
    # the cluster might also be dative if both atoms are in the
    # donor set AND one is Pt, but our cloud only has 1 Pt).
    assert n_dative >= 2, (
        f"Expected at least 2 dative flags (NH3 -> Pt × 2), got {n_dative}"
    )
    # The 2 Pt-N edges (i=Pt, j=N) and 2 N-Pt edges (i=N, j=Pt)
    # depending on ordering.  Verify that every dative flag has Z_i in
    # {7,8,16} AND Z_j == 78 OR Z_i == 78 AND Z_j in {7,8,16}.
    for k in range(e):
        if pf.is_dative_candidate[k]:
            zi = int(pf.z_i[k])
            zj = int(pf.z_j[k])
            ok = (
                (zi in (7, 8, 16) and zj == 78)
                or (zj in (7, 8, 16) and zi == 78)
            )
            assert ok, (
                f"Dative pair (Z_i={zi}, Z_j={zj}) must have one donor "
                f"and one Pt; got (Z_i={zi}, Z_j={zj})"
            )
