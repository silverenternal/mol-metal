# SPDX-License-Identifier: Apache-2.0
"""SanitizeMol branch in BondAwareDecoder._assemble_mol must populate DecodedMol.error.

WHY THIS EXISTS
---------------
The BLK-C-L3 fix at
``molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py:765`` populated
``DecodedMol.error`` on the SanitizeMol exception path. The fix closed the
gap on the YuelBond decoder but TWO OTHER LIVE DecodedMol construction
sites still swallow SanitizeMol and return ``error=None``:

  1. ``molmetal/models/bond_head.py:1043-1070`` --
     ``BondAwareDecoder._assemble_mol``.
     ``try: Chem.SanitizeMol(mol); sanitized = True / except Exception:
     sanitized = False`` -- the ``except`` branch captures no exception
     text. The DecodedMol at :1062 omits the ``error=`` kwarg.

  2. ``molmetal/molmetal_lam/lam_chem/decoder_rework.py:904-919`` --
     ``DecoderRework.decode`` fallback when ``bond_head is None``.
     Same shape: ``Chem.SanitizeMol`` caught, ``sanitized`` set,
     DecodedMol returns without ``error=``.

The BiasSweep / L3-guidance probe aggregates ``error_stage`` from
``DecodedMol.error`` to distinguish "head rejection" from "RDKit
valence/kekulize refusal". A bond graph the head predicted correctly
but RDKit refuses to sanitise is the OPPOSITE diagnosis from one the
head got wrong -- they tell you whether the head or the bond graph
is the bottleneck. Without ``error``, the instrument distinguishes
nothing, and the 70.7% aromatic-collapse finding at BO 1.30-1.50 is
uninterpretable on the legacy decoder paths.

This test pins the fix on the BondAwareDecoder path (site 1 above).
Site 2 is structurally identical and is covered by an analogous test
in test_wf_round7_decoder_rework_silent_failure.py.

RED CHECK
---------
To verify the test goes red for the intended reason (and not an
incidental one), revert the populate on the SanitizeMol branch
(``error=None`` always) and re-run -- the assertion fails because
``decoded.error`` is ``None`` instead of a populated string. Then
restore the fix and confirm GREEN.

MEASURED
--------
The reproducer is a 3-atom carbon triangle at aromatic bond distance
(1.4 A) with an externally-forced TRIPLE bond on every pair, which
forces the BondAwareDecoder's bond-pattern mask / valence cap aside
and triggers ``Chem.SanitizeMol``'s ``Explicit valence for atom # 0
C, 6, is greater than permitted`` rejection. This is a deterministic,
CPU-only reproducer -- no tmQM shards, no GPU, no trained checkpoint.

Note: the YuelBondBondDecoder test in
``test_wf_blkc_l3_sanitize_error_propagation.py`` exercises the
YuelBond path with forced aromatic bonds (Kekulize of an aromatic
3-ring); this test exercises BondAwareDecoder with forced TRIPLE
bonds (valence overflow on a 3-ring) -- different rejection reason,
same bug shape.
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
    BondOrderHead,
    BondAwareDecoder,
)


def _three_carbon_oversaturated_triangle() -> AtomCloud:
    """Three carbons at 1.4 A -- BondAwareDecoder builds a candidate
    pair list for every (i, j) i<j, and we will force the bond selector
    to land on the TRIPLE class for every pair. Each C atom ends up
    with 6 bonds of order 3 (sum 18, max allowed 4), which Chem.SanitizeMol
    refuses with ``Explicit valence for atom # X C, 6, is greater than permitted``.

    This deterministically exercises the SanitizeMol exception branch
    in BondAwareDecoder._assemble_mol -- the very branch whose
    ``DecodedMol.error`` field is the subject of this test.
    """
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [0.7, 1.21, 0.0],
        ],
        dtype=torch.float32,
    )
    atomic_numbers = torch.tensor([6, 6, 6], dtype=torch.long)
    return AtomCloud(positions=positions, atomic_numbers=atomic_numbers)


def test_bond_assemble_mol_sanitize_branch_populates_decoded_error():
    """When RDKit SanitizeMol refuses, ``DecodedMol.error`` MUST be set.

    Branch under test:
      ``molmetal/models/bond_head.py:1043-1070``
      ``BondAwareDecoder._assemble_mol`` -- the ``except Exception``
      block around ``Chem.SanitizeMol``.

    Before the fix this branch returned ``DecodedMol(..., error=None)``
    and downstream ``error_stage`` read ``"unknown_no_error_field"``.
    After the fix the same branch returns ``DecodedMol(..., error=<str>)``
    so the rejection reason is recorded.

    Note on the reproducer: ``BondAwareDecoder._assemble_mol`` is
    private API. We call it directly with crafted ``bond_orders`` that
    are KNOWN to fail ``Chem.SanitizeMol`` -- a 3-ring with three
    TRIPLE bonds has 6 bond-orders on every atom (allowed valence = 4),
    which RDKit refuses with ``Explicit valence for atom # X C, 6,
    is greater than permitted``. The bond-pattern mask / valence cap
    in the public ``dec.decode(cloud)`` path would block these, which
    is why we hit ``_assemble_mol`` directly -- this test probes the
    branch the public path can also land on via the bond_head=None
    fallback in ``decoder_rework.decode`` and via any caller that
    passes pre-filtered bond orders in.
    """
    head = BondOrderHead(in_dim=9, hidden_dim=64, dropout=0.1)
    dec = BondAwareDecoder(bond_head=head)
    cloud = _three_carbon_oversaturated_triangle()
    # Craft an oversaturated bond graph: 3 TRIPLE bonds in a 3-ring
    # gives each C a bond-order sum of 9, well above the allowed 4.
    oversaturated_bonds = [(0, 1, 3), (0, 2, 3), (1, 2, 3)]  # 3 = BOND_TRIPLE
    d = dec._assemble_mol(
        cloud,
        bond_orders=oversaturated_bonds,
        dative_bonds=[],
    )

    # Sanitization is the failure point.
    assert d.sanitized is False, (
        f"sanitized must be False for the SanitizeMol branch to be the one "
        f"exercised, got sanitized={d.sanitized}; if True the bonds are "
        f"not actually oversaturated and we are not probing SanitizeMol"
    )
    assert d.smiles == "", (
        f"smiles must be empty when SanitizeMol failed, got smiles={d.smiles!r}"
    )
    assert d.n_bonds == 3, (
        f"expected 3 bonds in the assembled bond graph, got n_bonds={d.n_bonds}; "
        f"if 0, RWMol.AddBond rejected the oversaturated graph upstream and "
        f"we are not probing SanitizeMol"
    )
    # THE FIX: the rejection reason must be populated. Pre-fix this is
    # None on every such decode, exactly the gap that BondAwareDecoder
    # _assemble_mol inherits from the legacy code path.
    assert d.error is not None and str(d.error).strip() != "", (
        f"DecodedMol.error must be populated when Chem.SanitizeMol raises "
        f"(the reproduction is the explicit-valence refusal on a 3-ring "
        f"with three TRIPLE bonds), got error={d.error!r}; pre-fix the "
        f"SanitizeMol branch swallowed the exception and returned "
        f"error=None, which is the silent-failure gap that this test "
        f"pins down on the BondAwareDecoder path (BLK-C-L3 fix)"
    )
    # The rejection must come from RDKit's exception class -- confirms
    # the exception was captured, not a generic placeholder string.
    assert ("Valence" in d.error) or ("Explicit" in d.error) or ("Sanitize" in d.error) or ("Kekulize" in d.error), (
        f"DecodedMol.error must contain the RDKit exception class / keyword "
        f"(Valence / Explicit / Sanitize / Kekulize), got error={d.error!r}; "
        f"this is what distinguishes RDKit's valence refusal from a "
        f"head-rejection log on the downstream sweep"
    )


def test_bond_assemble_mol_no_numeric_drift_on_failure_path():
    """The fix must not change ``smiles``, ``bond_orders``, ``n_atoms``,
    ``n_bonds``, or ``sanitized`` on the failure path.

    Pre-fix these are populated identically; the fix only adds the
    ``error`` field. This test pins that the new information does not
    shift any existing numeric value.
    """
    head = BondOrderHead(in_dim=9, hidden_dim=64, dropout=0.1)
    dec = BondAwareDecoder(bond_head=head)
    cloud = _three_carbon_oversaturated_triangle()
    oversaturated_bonds = [(0, 1, 3), (0, 2, 3), (1, 2, 3)]
    d = dec._assemble_mol(
        cloud,
        bond_orders=oversaturated_bonds,
        dative_bonds=[],
    )

    assert d.sanitized is False
    assert d.smiles == ""
    assert d.n_atoms == 3
    assert d.n_bonds == 3
    assert isinstance(d.n_atoms, int)
    assert isinstance(d.n_bonds, int)
    assert d.dative_bonds == []
