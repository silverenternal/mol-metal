# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T57/T58 — the corpus geometry the model is trained on.

T57: RDKit embeds each disconnected fragment at the origin, so multi-component
SMILES put atoms on top of each other (8,782 coincident pairs over 327 of 399
training clouds).  ``separate_fragments`` moves them apart deterministically.

T58: ``_place_metal_at_donor`` tested clearance against heavy atoms only, so
the metal landed on C-H hydrogens (104 of 399 clouds closer than the M-H clash
threshold).  It now tests every atom and tilts at the same M-L distance rather
than pushing out of the dative cutoff.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]


def _load_trainer():
    if "_t58_trainer" in sys.modules:
        return sys.modules["_t58_trainer"]
    sys.path.insert(0, str(_REPO / "molmetal" / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "_t58_trainer", str(_REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t58_trainer"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_coincident_fragments_are_pulled_apart():
    """Six waters embedded at the origin must not stay on one point."""
    from molmetal.data.fragment_layout import min_pair_distance, separate_fragments

    # Three water oxygens all at the origin, hydrogens offset identically —
    # exactly what EmbedMolecule produces for [OH2].[OH2].[OH2].
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.76, 0.59, 0.0],
            [-0.76, 0.59, 0.0],
            [0.0, 0.0, 0.0],
            [0.76, 0.59, 0.0],
            [-0.76, 0.59, 0.0],
            [0.0, 0.0, 0.0],
            [0.76, 0.59, 0.0],
            [-0.76, 0.59, 0.0],
        ],
        dtype=torch.float64,
    )
    frags = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    assert min_pair_distance(coords) == 0.0

    out = separate_fragments(coords, frags, clearance=2.0)

    assert float(torch.cdist(out[:3], out[3:6]).min()) >= 2.0
    assert float(torch.cdist(out[:3], out[6:9]).min()) >= 2.0
    assert float(torch.cdist(out[3:6], out[6:9]).min()) >= 2.0
    # Intra-fragment geometry is untouched: only whole fragments move.
    assert torch.allclose(out[:3] - out[0], coords[:3] - coords[0], atol=1e-9)


def test_separation_is_deterministic():
    from molmetal.data.fragment_layout import separate_fragments

    torch.manual_seed(0)
    coords = torch.randn(12, 3, dtype=torch.float64)
    coords[3:6] = coords[0:3]
    frags = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    a = separate_fragments(coords, frags)
    b = separate_fragments(coords, frags)
    assert torch.equal(a, b)


def test_a_single_fragment_is_returned_unchanged():
    from molmetal.data.fragment_layout import separate_fragments

    coords = torch.randn(5, 3, dtype=torch.float64)
    out = separate_fragments(coords, [list(range(5))])
    assert torch.equal(out, coords)


def test_the_keep_atom_fragment_does_not_move():
    """The metal's fragment is placed afterwards and must not be displaced."""
    from molmetal.data.fragment_layout import separate_fragments

    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # metal
            [0.0, 0.0, 0.0],  # water 1 oxygen (on top of the metal)
            [0.0, 0.0, 0.0],  # water 2 oxygen
        ],
        dtype=torch.float64,
    )
    out = separate_fragments(coords, [[0], [1], [2]], keep_atom=0, clearance=2.0)
    assert torch.equal(out[0], coords[0])
    assert float(torch.cdist(out[1:], out[:1]).min()) >= 2.0


def test_min_pair_distance_of_a_single_atom_is_inf():
    from molmetal.data.fragment_layout import min_pair_distance

    assert min_pair_distance(torch.zeros(1, 3)) == float("inf")


def test_the_metal_clears_hydrogens_and_keeps_its_bond_length():
    """T58 — an H sits exactly where 'straight out from the donor' lands.

    Layout: donor N at the origin, one C at (-3, 0, 0) so the outward
    direction is +x, and an H at (1.0, 0, 0).  The old code placed the metal
    at (2.05, 0, 0), i.e. 1.05 A from that H, because it only tested heavy
    atoms.  The fix must end up at least 1.2 A from EVERY atom while keeping
    the M-L distance at 2.05 A.
    """
    tr = _load_trainer()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    z = torch.tensor([44, 7, 6, 1], dtype=torch.long)  # Ru, N, C, H

    out, ok = tr._place_metal_at_donor(pos, z, 0, 1)

    assert ok
    d = (out - out[0]).norm(dim=-1)
    d[0] = 9.0
    assert float(d.min()) >= 1.2 - 1e-4
    assert float((out[0] - out[1]).norm()) == pytest.approx(2.05, abs=1e-4)
    # The H is the binding constraint here, so it is the one that moved out.
    assert float((out[0] - out[3]).norm()) >= 1.2 - 1e-4


def test_the_metal_does_not_walk_out_of_the_dative_cutoff():
    """Tilting keeps |M-L| at the bond length; pushing out would not."""
    tr = _load_trainer()
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    z = torch.tensor([44, 7, 6, 1], dtype=torch.long)
    out, _ = tr._place_metal_at_donor(pos, z, 0, 1)
    assert float((out[0] - out[1]).norm()) <= tr.METAL_DONOR_BOND_LENGTH + 1e-4


def test_the_recorded_cohort_measurement_still_holds():
    """The artifact's after-state invariants, so a regression cannot hide."""
    path = _REPO / "metrics" / "by_metric" / "wfa113_t57_fragment_layout.json"
    if not path.exists():
        pytest.skip("probe artifact not present")
    d = json.loads(path.read_text())
    before, after = d["before"], d["after"]

    assert before["coincident_pairs"] > 0, "the defect must be visible in the before state"
    assert after["coincident_pairs"] == 0
    assert after["clouds_with_coincident_atoms"] == 0
    assert before["metal_h_below_1p17A"] > 0
    assert after["metal_h_below_1p17A"] == 0
    assert after["nonbonded_below_1A"] == 0
    assert after["clouds_with_zero_dative"] == 0
