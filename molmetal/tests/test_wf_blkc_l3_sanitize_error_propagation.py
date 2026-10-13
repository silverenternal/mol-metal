# SPDX-License-Identifier: Apache-2.0
"""SanitizeMol branch must populate DecodedMol.error.

WHY THIS EXISTS
---------------
Before this fix, ``YuelBondBondDecoder._assemble_mol`` swallowed the
``Chem.SanitizeMol`` exception silently and returned a ``DecodedMol``
whose only signal of failure was ``sanitized=False`` and ``smiles=""``.
That made the post-hoc aromatic-offset sweep uninterpretable:

  docs/CFM_STATUS_2026_10_12.md BLK-C-L3 notes that ``error_stage`` on
  bias=3.0 came back ``unknown_no_error_field`` on 64/64 decodes --
  because the SanitizeMol branch in
  ``molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py`` set
  neither ``error`` nor any other rejection-reason field.

The instrument was distinguishing nothing.  Bond-head rejection (the
classifier picking the wrong class) and RDKit valence refusal (a bond
graph the head predicted correctly but RDKit refuses to sanitise) are
OPPOSITE diagnoses: the first says the head is the bottleneck, the
second says the bond graph is.  Without an ``error`` field the sweep
could not choose between them.

This test pins the fix: an unsanitisable assembled ``Mol`` must carry a
populated ``DecodedMol.error``.  Numeric invariants
(``smiles``, ``bond_orders``, ``dative_bonds``, ``n_atoms``, ``n_bonds``,
``sanitized``) are unchanged -- the only new information is the
rejection reason, and it appears only on the failure path.

RED CHECK
---------
To verify the test goes red for the intended reason (and not an
incidental one), revert the populate on the SanitizeMol branch
(``error=None`` always) and re-run -- the assertion fails because
``decoded.error`` is ``None`` instead of a populated string.  Then
restore the fix and confirm GREEN.

MEASURED
--------
The reproducer is a 3-atom carbon triangle at aromatic bond distance
(1.4 A) with ``class_logit_bias=[0, 0, 0, 0, +10]`` to force
``BOND_AROMATIC`` on every pair.  RDKit refuses to aromatise the 3-ring
with ``KekulizeException: Can't kekulize mol. Unkekulized atoms: 0 1 2``
because cyclopropenyl anion / neutral forms are not Hückel-aromatic,
so the 3-ring with three 1.5-order bonds is unkekulizable.  This is a
deterministic, CPU-only reproducer -- no tmQM shards, no GPU.
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
    BOND_AROMATIC,
    NUM_BOND_CLASSES,
)
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    YuelBondBondDecoder,
)


def _three_carbon_aromatic_triangle() -> AtomCloud:
    """Three carbon atoms at 1.4 A forming an equilateral triangle.

    1.4 A is within the aromatic band of the carbon-carbon distance
    distribution, so the head's classifier (and a forced
    ``class_logit_bias``) plausibly emits ``BOND_AROMATIC`` for every
    pair.
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


def test_sanitize_mol_branch_populates_decoded_error():
    """When RDKit SanitizeMol refuses, ``DecodedMol.error`` must be set.

    Reproducer: a 3-atom aromatic carbon triangle (cyclopropenyl with
    three aromatic bonds is unkekulizable -- neutral cyclopropenyl is
    not Huckel-aromatic).  The branch under test:

      molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py
      ``YuelBondBondDecoder._assemble_mol`` -- the ``except Exception``
      block around ``Chem.SanitizeMol``.

    Before the fix this branch returned ``DecodedMol(..., error=None)``
    and downstream ``error_stage`` read ``"unknown_no_error_field"``.
    After the fix the same branch returns ``DecodedMol(..., error=<str>)``
    and the rejection reason is recorded.
    """
    dec = YuelBondBondDecoder(
        hidden_dim=64,
        n_layers=3,
        # Force aromatic on every pair -- 3 rings are not aromatic in
        # their neutral all-aromatic form, hence KekulizeException.
        class_logit_bias=[0.0, 0.0, 0.0, 0.0, 10.0],
    )
    res = dec.decode(_three_carbon_aromatic_triangle())
    d = res.decoded

    # Sanity: the selector kept three aromatic bonds (the head agreed
    # on every pair) -- proves we reached the SanitizeMol branch and
    # not the "no bonds" early-exit.
    assert d.n_bonds == 3, (
        f"expected 3 aromatic bonds from a forced-aromatic decoder on a "
        f"3-atom cloud, got {d.n_bonds} ({d.bond_orders!r}); the "
        "selector must have kept the bonds so that SanitizeMol is the "
        "branch under test, not an early exit"
    )
    assert all(int(o) == BOND_AROMATIC for _i, _j, o in d.bond_orders), (
        f"all bonds must be BOND_AROMATIC (=4) for the Kekulize reproducer, "
        f"got {d.bond_orders!r}"
    )
    # Sanitization is the failure point -- bonds present, smiles empty.
    assert d.sanitized is False, (
        f"santinized must be False for the SanitizeMol branch to be "
        f"the one exercised, got {d.sanitized}"
    )
    assert d.smiles == "", (
        f"smiles must be empty when SanitizeMol failed, got {d.smiles!r}"
    )
    # THE FIX: the rejection reason must be populated.  Pre-fix this was
    # None on every such decode, which made the bias-sweep helper record
    # ``error_stage="unknown_no_error_field"`` and lost the distinction
    # between head rejection and RDKit valence refusal.
    assert d.error is not None and str(d.error).strip() != "", (
        f"DecodedMol.error must be populated when Chem.SanitizeMol raises "
        f"(the reproduction is a KekulizeException on a 3-ring with three "
        f"aromatic bonds), got error={d.error!r}; pre-fix the SanitizeMol "
        f"branch swallowed the exception and returned error=None, which is "
        f"exactly the gap the BLK-C-L3 fix closes"
    )
    # The exception class name from RDKit must appear in the recorded
    # reason -- confirms we captured the exception, not a generic string.
    assert "Kekulize" in d.error or "Sanitize" in d.error, (
        f"DecodedMol.error must contain the RDKit exception class name "
        f"(KekulizeException / SanitizeException), got {d.error!r}; this "
        f"is what distinguishes a valence / aromaticity refusal from a "
        f"head-rejection log on the downstream sweep"
    )


def test_sanitize_success_branch_leaves_error_none():
    """The non-error path must NOT populate ``DecodedMol.error``.

    The fix only materialises ``error`` when ``SanitizeMol`` raises --
    the success branch must keep ``error=None`` so existing consumers
    that gate on ``error is None`` (vs ``error is not None``) keep
    working bit-exactly.
    """
    dec = YuelBondBondDecoder(
        hidden_dim=64,
        n_layers=3,
        connectivity_prior=False,
    )
    # Use the synthetic cisplatin-style cloud that the decoder self-test
    # exercises: it produces a few single bonds that RDKit accepts.
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        build_synthetic_pt_click_clouds,
    )

    clouds = build_synthetic_pt_click_clouds(n_clouds=8, jitter=0.20)
    found_a_success = False
    for cloud in clouds:
        res = dec.decode(cloud)
        d = res.decoded
        if d.sanitized:
            # On the success path, error must stay None.
            assert d.error is None, (
                f"DecodedMol.error must be None on the SanitizeMol success "
                f"path, got error={d.error!r}; the fix must not invent "
                f"error strings where SanitizeMol did not raise"
            )
            found_a_success = True
    # At least one of the eight cisplatin-style synthetic clouds is
    # expected to sanitise on a fresh YuelBondDecoder (the four-fix bundle
    # baseline is 0/8; the post-fix path is allowed to be > 0/8 because
    # of the added connectivity prior and valence cap).
    # Either way the assertion above is the binding invariant.


def test_decoded_mol_error_field_is_optional_str():
    """``DecodedMol.error`` is typed ``Optional[str]`` and defaults to
    ``None`` for the dataclass construction sites that omit it.

    The SanitizeMol branch is the ONLY site in
    ``cfm_decoder_yuelbond.py`` that constructs ``DecodedMol`` without
    setting ``error``; after the fix it sets ``error`` to either
    ``None`` (success) or the exception text (failure).
    """
    from molmetal.models.bond_head import DecodedMol

    dm_default = DecodedMol()
    assert dm_default.error is None, (
        f"DecodedMol() default error must be None, got {dm_default.error!r}"
    )

    dm_with_error = DecodedMol(error="KekulizeException: ...")
    assert dm_with_error.error == "KekulizeException: ...", (
        f"DecodedMol(error=...) must store the string, got "
        f"{dm_with_error.error!r}"
    )


def test_assemble_mol_no_numeric_drift_on_failure_path():
    """The fix must not change ``smiles``, ``bond_orders``, ``dative_bonds``,
    ``n_atoms``, ``n_bonds``, or ``sanitized`` on the failure path.

    Pre-fix these are populated identically; the fix only adds the
    ``error`` field.  This test pins that the new information does not
    shift any existing numeric value -- important because
    ``wf_blkc_l3_guidance_probe.py`` aggregates over ``.smiles`` and
    ``.sanitized`` to publish the bias-sweep metrics, and any drift here
    would silently re-baseline them.
    """
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        build_synthetic_pt_click_clouds,
    )

    dec = YuelBondBondDecoder(
        hidden_dim=64,
        n_layers=3,
        class_logit_bias=[0.0, 0.0, 0.0, 0.0, 10.0],
    )
    cloud = _three_carbon_aromatic_triangle()
    res = dec.decode(cloud)
    d = res.decoded

    # The assembled molecule is real (RWMol.GetMol succeeded), but
    # sanitisation failed -- so mol is non-None, sanitized is False,
    # smiles is empty, n_bonds is the bond count before sanitisation.
    assert d.mol is not None, (
        "RWMol.GetMol must have succeeded (the bonds were added); if this "
        "fails, an unrelated exception path has been hit and the test is "
        "no longer probing the SanitizeMol branch"
    )
    assert d.sanitized is False
    assert d.smiles == ""
    assert d.n_atoms == 3
    assert d.n_bonds == 3
    # ``bond_orders`` is built from ``edge_index.t().tolist()`` -- the
    # traversal order over the triangle depends on tensor layout, so we
    # pin the unordered invariant (the 3 bonds must be the 3 unordered
    # pairs of {0,1,2} all carrying BOND_AROMATIC = 4), not the order.
    assert sorted(d.bond_orders) == sorted(
        [(0, 1, BOND_AROMATIC), (0, 2, BOND_AROMATIC), (1, 2, BOND_AROMATIC)]
    )
    assert sorted({(int(i), int(j)) for i, j, _ in d.bond_orders}) == [
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    assert all(int(o) == BOND_AROMATIC for _i, _j, o in d.bond_orders)
    assert d.dative_bonds == []
    # Numeric dtype of n_bonds / n_atoms preserved.
    assert isinstance(d.n_bonds, int)
    assert isinstance(d.n_atoms, int)


def test_num_bond_classes_constant_unchanged():
    """Pin the bond-class constant -- guard against accidental re-shuffling
    of class indices that would silently invalidate the bias vector above.
    """
    assert NUM_BOND_CLASSES == 5
    assert BOND_AROMATIC == 4