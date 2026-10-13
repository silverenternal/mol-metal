# SPDX-License-Identifier: Apache-2.0
"""SanitizeMol branch in DecoderRework.decode (bond_head=None fallback)
must populate DecodedMol.error.

WHY THIS EXISTS
---------------
Companion test to ``test_wf_round7_bond_head_silent_failure.py``.
Two other live DecodedMol construction sites still swallowed SanitizeMol
and returned ``error=None`` before the BLK-C-L3 fix to
``cfm_decoder_yuelbond.py:765``:

  1. ``molmetal/models/bond_head.py:1043-1070`` --
     ``BondAwareDecoder._assemble_mol`` -- pinned in
     test_wf_round7_bond_head_silent_failure.py.
  2. ``molmetal/molmetal_lam/lam_chem/decoder_rework.py:904-919`` --
     ``DecoderRework.decode`` fallback when ``bond_head is None``.
     Pinned here.

Same shape: ``Chem.SanitizeMol`` caught, ``sanitized`` set, DecodedMol
returned without ``error=``. The BiasSweep / L3-guidance probe
aggregates ``error_stage`` from ``DecodedMol.error`` to distinguish
"head rejection" from "RDKit valence / kekulize refusal". Without
``error``, the instrument distinguishes nothing.

RED CHECK
---------
To verify the test goes red for the intended reason (and not an
incidental one), revert the populate on the SanitizeMol branch
(``error=None`` always) and re-run -- the assertion fails because
``decoded.error`` is ``None`` instead of a populated string. Then
restore the fix and confirm GREEN.

MEASURED
--------
The reproducer is a 3-atom carbon triangle at 1.4 A with the
bond_head=None fallback in ``DecoderRework.decode``. The fallback
path itself produces SINGLE bonds (cyclopropane is fine) and so does
NOT exercise SanitizeMol on this geometry. To probe the SanitizeMol
branch on this code path, we have to land a bond-graph that RDKit
will reject.

Strategy: craft an oversaturated 4-atom cloud (carbons at short
distance with all-SINGLE bonds, then call ``dec.decode`` with
``bond_head=None``). The bond_head=None fallback in
``DecoderRework.decode`` rejects dative bonds and emits SINGLE bonds
on every candidate pair -- which on a 4-atom close-packed cloud
produces an over-connected molecule that RDKit refuses via
``Explicit valence for atom # X C, N, is greater than permitted``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.models.bond_head import (
    AtomCloud,
)
from molmetal.molmetal_lam.lam_chem.decoder_rework import DecoderRework


def _oxygen_oversaturated() -> AtomCloud:
    """A central oxygen at ~1.5 A from each of four Cs.

    Carbon has allowed valence 4 so 4 SINGLE bonds leaves it
    saturated. OXYGEN has allowed valence 2 -- so any bond graph
    with 3+ SINGLE bonds on an O (which the bond_head=None fallback
    will emit on a 4-neighbour layout) oversaturates O, and RDKit
    refuses with ``Explicit valence for atom # X O, 2, is greater
    than permitted``.

    This is a deterministic, CPU-only reproducer of the SanitizeMol
    failure path in ``DecoderRework.decode(cloud, bond_head=None)``.
    """
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],   # O at origin
            [1.5, 0.0, 0.0],   # C neighbours
            [0.0, 1.5, 0.0],
            [0.0, 0.0, 1.5],
            [1.5, 1.5, 1.5],   # diagonal C
        ],
        dtype=torch.float32,
    )
    atomic_numbers = torch.tensor([8, 6, 6, 6, 6], dtype=torch.long)
    return AtomCloud(positions=positions, atomic_numbers=atomic_numbers)


def test_decoder_rework_bond_head_none_sanitize_branch_populates_decoded_error():
    """When RDKit SanitizeMol refuses, ``DecodedMol.error`` MUST be set.

    Branch under test:
      ``molmetal/molmetal_lam/lam_chem/decoder_rework.py:904-922``
      ``DecoderRework.decode`` fallback when ``bond_head is None``.

    Before the fix this branch returned ``DecodedMol(..., error=None)``
    on SanitizeMol failure. After the fix it returns
    ``DecodedMol(..., error=<str>)`` so the rejection reason is recorded.
    """
    dec = DecoderRework()
    cloud = _oxygen_oversaturated()
    out = dec.decode(cloud, bond_head=None)

    assert out.n_atoms == 5
    assert out.sanitized is False, (
        f"sanitized must be False for the SanitizeMol branch to be the "
        f"one exercised, got sanitized={out.sanitized}; if True the "
        f"bond graph is not oversaturated on a 5-atom close-packed "
        f"cloud and we are not probing SanitizeMol"
    )
    # THE FIX: the rejection reason must be populated.
    assert out.error is not None and str(out.error).strip() != "", (
        f"DecodedMol.error must be populated when Chem.SanitizeMol raises "
        f"(the reproduction is the explicit-valence refusal on O with "
        f"4 carbon neighbours at SINGLE-bond distance), got "
        f"error={out.error!r}; pre-fix the DecoderRework SanitizeMol "
        f"branch swallowed the exception and returned error=None"
    )


def test_decoder_rework_bond_head_none_success_branch_error_none():
    """The success path on the bond_head=None fallback must keep error=None.

    On a geometry that DOES sanitise, ``error`` must remain None so
    consumers gating on ``error is None`` keep working bit-exactly.
    """
    dec = DecoderRework()
    # Use a sparse 3-atom cloud with realistic spacing so the bond
    # graph is exactly one or two SINGLES which RDKit accepts.
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.5, 0.0],
        ],
        dtype=torch.float32,
    )
    cloud = AtomCloud(
        positions=positions,
        atomic_numbers=torch.tensor([6, 6, 6], dtype=torch.long),
    )
    out = dec.decode(cloud, bond_head=None)
    # If sanitized, error must be None. If not, this cloud is also
    # oversaturated and the test geometry needs revising.
    if out.sanitized:
        assert out.error is None, (
            f"DecodedMol.error must be None on the SanitizeMol success "
            f"path, got error={out.error!r}; the fix must not invent "
            f"error strings where SanitizeMol did not raise"
        )
