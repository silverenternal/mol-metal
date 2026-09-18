"""Smoke tests for the SBDD environment (REINVENT4 wrapper + voxelization)."""
from __future__ import annotations

import numpy as np
import pytest

from molmetal_lam.sbdd_env import (
    DEFAULT_SCORER_WEIGHTS,
    REINVENT4Scorer,
    SpatialTile,
    batch_score,
    pocket_to_spatial_tiles,
    voxel_grid_shape,
)


# ---------------------------------------------------------------------------
# REINVENT4Scorer
# ---------------------------------------------------------------------------
def test_reinvent_scorer_score() -> None:
    """Aspirin should produce a finite, non-negative score within a sane range."""
    scorer = REINVENT4Scorer()
    aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
    s = scorer.score(aspirin)
    assert isinstance(s, float)
    assert 0.0 <= s <= 3.0, f"score out of range: {s}"
    breakdown = scorer.component_breakdown(aspirin)
    # QED for aspirin should be well above zero.
    assert 0.0 <= breakdown["qed"] <= 1.0


def test_reinvent_batch_score() -> None:
    """batch_score returns 10 scores, all in [0, 3]."""
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",       # aspirin
        "CCO",                            # ethanol
        "c1ccccc1",                       # benzene
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",    # ibuprofen
        "CN1CCC[C@H]1c2cccnc2",          # nicotine
        "CC(=O)Nc1ccc(O)cc1",            # paracetamol
        "C1=CC=C2C=CC=CC2=C1",            # naphthalene
        "CCOCC",                          # diethyl ether
        "CC(=O)C",                        # acetone
        "C1CCCCC1",                       # cyclohexane
    ]
    scores = batch_score(smiles_list)
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (10,)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 3.0)
    # And the per-call object should agree.
    scorer = REINVENT4Scorer()
    manual = np.array([scorer.score(s) for s in smiles_list], dtype=np.float32)
    np.testing.assert_allclose(scores, manual, rtol=1e-5)


# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------
def _make_synthetic_pocket():
    """Build a small Pocket object with mixed atom types (C/N/O)."""
    import torch
    from molmetal.domain import Pocket
    centre = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    # 4 atoms inside the box: C, N, O, C  (so we exercise all channels).
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [1.0, -1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    atom_types = torch.tensor([6, 7, 8, 6], dtype=torch.long)
    return Pocket(
        pdb_id="syn",
        coords=coords,
        atom_types=atom_types,
        residue_ids=torch.zeros(4, dtype=torch.long),
        chain_ids=torch.zeros(4, dtype=torch.long),
        mask=torch.ones(4, dtype=torch.bool),
        center=centre,
        radius=4.0,
    )


def test_voxelization_shape() -> None:
    """A 32³ voxel grid should produce exactly 32 768 spatial tiles."""
    pocket = _make_synthetic_pocket()
    tiles = pocket_to_spatial_tiles(pocket, grid_size=32, resolution=0.5)
    assert isinstance(tiles, list)
    assert len(tiles) == 32 ** 3, f"expected 32768, got {len(tiles)}"
    assert all(isinstance(t, SpatialTile) for t in tiles)
    shape = voxel_grid_shape(32)
    assert shape == (32, 32, 32)
    # Each tile has a 3-D coord.
    for t in tiles[:5]:
        assert len(t.coords) == 3


def test_spatial_tile_hbond_detection() -> None:
    """A pocket with N and O atoms should have hbond_donor and hbond_acceptor voxels flagged.

    Note: in our coarse-grained channel mapping both N and O can act as
    hydrogen-bond donors AND acceptors (e.g. amide-N donor, pyridine-N
    acceptor).  The test therefore asserts:
      * 4 occupied voxels
      * 1 O atom → contributes ≥1 acceptor voxel
      * 1 N atom → contributes ≥1 donor voxel
      * 2 C atoms → 2 hydrophobic voxels
    """
    pocket = _make_synthetic_pocket()
    tiles = pocket_to_spatial_tiles(pocket, grid_size=16, resolution=0.5)
    n_occupied = sum(t.occupied for t in tiles)
    n_hb_donor = sum(t.hbond_donor for t in tiles)
    n_hb_acceptor = sum(t.hbond_acceptor for t in tiles)
    n_hydrophobic = sum(t.hydrophobic for t in tiles)

    # We placed 4 atoms → 4 occupied voxels.
    assert n_occupied == 4, f"expected 4 occupied voxels, got {n_occupied}"
    # The single N atom gives at least 1 donor voxel (it can also be an acceptor).
    assert n_hb_donor >= 1, f"expected ≥1 hbond donor, got {n_hb_donor}"
    # The single O atom gives at least 1 acceptor voxel (it can also be a donor).
    assert n_hb_acceptor >= 1, f"expected ≥1 hbond acceptor, got {n_hb_acceptor}"
    # Two C → two hydrophobic voxels.
    assert n_hydrophobic == 2, f"expected 2 hydrophobic, got {n_hydrophobic}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))