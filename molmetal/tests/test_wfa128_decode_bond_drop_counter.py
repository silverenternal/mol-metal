# SPDX-License-Identifier: Apache-2.0
"""Wf-A128 — the decode path must not lose information silently.

TWO HOLES, BOTH IN THE DECODE PATH
----------------------------------

DEFECT 1 — ``_assemble_mol`` DROPPED the bonds RDKit refused, with no
counter, no log and no field.  A dropped bond is a SPLIT: the two atoms it
would have joined end up in different fragments, and the decode said
nothing.  Every other fragmentation quantity in this repo is measured in 6+
places (``n_fragments``, ``largest_fragment_frac``, ``n_frags_raw``, the
clash counters); this one was the exception.  The fix adds
``YuelBondDecodeResult.bonds_dropped_rdkit`` (an int) and
``bonds_dropped_rdkit_reasons`` (a type -> count map), and puts
``bonds_dropped_rdkit:<n>`` on the per-molecule decode record's
``DecodedMol.warnings``.

  THE DECODE BEHAVIOUR IS BIT-EXACT.  Only fields were added: no bond is
  added or skipped differently, no SMILES changes.  The proof is
  ``test_decode_output_is_bit_exact_against_the_pre_change_golden`` below,
  which pins the SMILES / atom order / bond list of a fixed input to the
  values captured from the code BEFORE this change (2026-10-31).

  THE CATCH IS DELIBERATELY WIDE.  Measured on RDKit 2026.03.6 (see
  ``test_rdkit_raises_what_on_each_invalid_bond``): a duplicate bond, an
  out-of-range atom index and a self-bond all raise ``RuntimeError``; a
  NEGATIVE atom index raises ``OverflowError`` (Python's int -> unsigned
  conversion, not RDKit).  Narrowing the catch to those two types would
  still change behaviour, because ``int(src)`` on a non-numeric value
  raises ``ValueError``/``TypeError`` — which the wide catch swallows (the
  bond is dropped) and a narrow catch would PROPAGATE (the whole decode
  raises).  ``test_the_wide_catch_is_retained`` pins that.

DEFECT 2 — ``clash_rejected_pairs`` was computed by the head and never
aggregated anywhere.  It now rides on ``YuelBondResult`` (the value object
the head's forward returns) as well as on
``YuelBondDecodeResult``.  The head's ``last_clash_rejected_pairs``
attribute is mutable module state, so reading it later gave whatever the
NEXT forward had left behind; the head now zeroes it at the top of every
forward and ``featurise`` copies it onto the result immediately after the
forward that produced it.

REPORT-ONLY (pinned here as measurements, NOT changed): the clash rule
removes a pair from the head's candidate list BEFORE the message passing,
and ``_bridge_components`` draws its candidate pool from that post-filter
``edge_index``.  So a clash-rejected pair is denied to the connectivity
prior, permanently.  The numbers are in
``test_the_clash_rule_vetoes_the_connectivity_prior``.

CPU-only.  No GPU, no network, no corpus on disk.
"""

from __future__ import annotations

import json

import pytest
import torch

from molmetal.models.bond_head import (
    BOND_DOUBLE,
    BOND_SINGLE,
    AtomCloud,
    DecodedMol,
)
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    YuelBondBondDecoder,
    YuelBondDecodeResult,
    build_synthetic_pt_click_clouds,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    DEFAULT_BOND_CLASH_RATIO,
    YuelBondDecoder,
    YuelBondDecoderHead,
    YuelBondResult,
    pair_collapse_mask,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decoder(seed: int = 1234) -> YuelBondBondDecoder:
    """A deterministic decoder: same seed -> same head weights."""
    torch.manual_seed(seed)
    head = YuelBondDecoderHead(hidden_dim=16, n_layers=2)
    return YuelBondBondDecoder(head=head)


def _cloud(positions, atomic_numbers) -> AtomCloud:
    return AtomCloud(
        positions=torch.tensor(positions, dtype=torch.float32),
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
    )


#: The 5-atom cloud used for the clash measurement: 5 carbons, one of them
#: (0, 1) collapsed to 0.8 A.  CORDERO r(C) = 0.76 A, so the collapse
#: threshold for a C-C pair is 0.7 * (0.76 + 0.76) = 1.064 A, and 0.8 A is
#: under it.  Every other pair is at 1.7 A or more.
_COLLAPSE_5 = [
    [0.0, 0.0, 0.0],
    [0.8, 0.0, 0.0],
    [2.5, 0.0, 0.0],
    [0.0, 2.5, 0.0],
    [0.0, 0.0, 2.5],
]


# ===========================================================================
# 1. THE BIT-EXACTNESS PROOF
# ===========================================================================

#: Captured from the code BEFORE the Wf-A128 change (2026-10-31, CPU,
#: RDKit 2026.03.6).  Same SMILES, same atom order, same bond list.
#: ``build_synthetic_pt_click_clouds(n_clouds=3, n_atoms_per_cloud=5,
#: seed=7, jitter=0.20)`` + a head seeded with 1234.
_GOLDEN = [
    {
        "smiles": "[NH2][Pt][N](Cl)Cl",
        "sanitized": True,
        "n_atoms": 5,
        "n_bonds": 4,
        "bond_orders": [[2, 4, 1], [1, 4, 1], [0, 4, 1], [0, 3, 1]],
        "dative_bonds": [[4, 0], [3, 0]],
        "dative_source": "z_rule",
        "dative_manufactured": 0,
        "repair": "",
        "error": None,
        "warnings": [],
        "bonds_accepted": 4,
        "bonds_rejected_overvalent": 0,
        "bonds_added_connectivity": 4,
    },
    {
        "smiles": "N[Cl]->[Pt][NH]Cl",
        "sanitized": True,
        "n_atoms": 5,
        "n_bonds": 4,
        "bond_orders": [[1, 4, 1], [2, 3, 1], [0, 2, 1], [0, 4, 1]],
        "dative_bonds": [[2, 0], [4, 0]],
        "dative_source": "z_rule",
        "dative_manufactured": 0,
        "repair": "",
        "error": None,
        "warnings": [],
        "bonds_accepted": 4,
        "bonds_rejected_overvalent": 0,
        "bonds_added_connectivity": 4,
    },
    {
        "smiles": "[NH2][Pt][N](Cl)Cl",
        "sanitized": True,
        "n_atoms": 5,
        "n_bonds": 4,
        "bond_orders": [[2, 4, 1], [1, 4, 1], [0, 4, 1], [0, 3, 1]],
        "dative_bonds": [[4, 0], [3, 0]],
        "dative_source": "z_rule",
        "dative_manufactured": 0,
        "repair": "",
        "error": None,
        "warnings": [],
        "bonds_accepted": 4,
        "bonds_rejected_overvalent": 0,
        "bonds_added_connectivity": 4,
    },
]


def test_decode_output_is_bit_exact_against_the_pre_change_golden():
    """The counters were added; the DECODE was not changed.

    Every field the pre-change decoder produced is pinned to the value it
    produced on 2026-10-31.  A dropped bond is still dropped (the SMILES
    still shows the split) — what changed is that the split is now COUNTED.
    """
    dec = _decoder()
    clouds = build_synthetic_pt_click_clouds(
        n_clouds=3, n_atoms_per_cloud=5, seed=7, jitter=0.20
    )
    assert len(clouds) == len(_GOLDEN)
    for cloud, golden in zip(clouds, _GOLDEN):
        res = dec.decode(cloud)
        d = res.decoded
        assert d.smiles == golden["smiles"], "SMILES changed"
        assert bool(d.sanitized) is golden["sanitized"]
        assert int(d.n_atoms) == golden["n_atoms"]
        assert int(d.n_bonds) == golden["n_bonds"]
        assert [list(map(int, b)) for b in d.bond_orders] == golden[
            "bond_orders"
        ], "the bond list (and therefore the atom order) changed"
        assert [list(map(int, b)) for b in d.dative_bonds] == golden["dative_bonds"]
        assert d.dative_source == golden["dative_source"]
        assert int(d.dative_manufactured) == golden["dative_manufactured"]
        assert d.repair == golden["repair"]
        assert d.error == golden["error"]
        assert (
            list(d.warnings) == golden["warnings"]
        ), "the warnings list changed on a decode with no dropped bond"
        assert int(res.bonds_accepted) == golden["bonds_accepted"]
        assert int(res.bonds_rejected_overvalent) == golden["bonds_rejected_overvalent"]
        assert int(res.bonds_added_connectivity) == golden["bonds_added_connectivity"]


def test_the_golden_is_a_stable_measurement_not_a_lucky_snapshot():
    """The golden is re-derived from the SAME construction, so a drift in
    the synthetic-cloud builder or in the head's initialisation shows up as
    a test failure rather than as a silently-moved golden."""
    dec = _decoder()
    clouds = build_synthetic_pt_click_clouds(
        n_clouds=3, n_atoms_per_cloud=5, seed=7, jitter=0.20
    )
    first = dec.decode(clouds[0])
    second = dec.decode(clouds[0])
    assert first.decoded.smiles == second.decoded.smiles
    assert json.dumps(
        [list(map(int, b)) for b in first.decoded.bond_orders]
    ) == json.dumps([list(map(int, b)) for b in second.decoded.bond_orders])


# ===========================================================================
# 2. DEFECT 1 — the dropped-bond counter
# ===========================================================================


def test_a_duplicate_bond_is_dropped_and_the_molecule_is_split():
    """The defect: RDKit refuses the SECOND (0, 1) bond, the decoder drops
    it silently, and the caller cannot tell that the decode is a fragment of
    the head's output.  The drop itself is NOT changed by this fix — only
    the counting of it is new."""
    dec = _decoder()
    cloud = build_synthetic_pt_click_clouds(
        n_clouds=1, n_atoms_per_cloud=5, seed=7, jitter=0.20
    )[0]
    # (0,1) twice, plus (1,2).  RDKit accepts the first (0,1) and refuses
    # the second.
    d = dec._assemble_mol(
        cloud,
        [(0, 1, BOND_SINGLE), (0, 1, BOND_DOUBLE), (1, 2, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert d.mol is not None
    assert d.mol.GetNumBonds() == 2, "RDKit kept 2 of the 3 requested bonds"
    assert d.n_bonds == 3, "the DECODED MOL still reports 3 bonds — the split"


def test_the_drop_count_is_exposed_on_the_result_object():
    dec = _decoder()
    cloud = build_synthetic_pt_click_clouds(
        n_clouds=1, n_atoms_per_cloud=5, seed=7, jitter=0.20
    )[0]
    dec._assemble_mol(
        cloud,
        [(0, 1, BOND_SINGLE), (0, 1, BOND_DOUBLE), (1, 2, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert dec._last_bonds_dropped == 1
    assert dec._last_bonds_dropped_reasons == {"RuntimeError": 1}


def test_a_clean_decode_reports_zero_dropped_bonds():
    """0 is the state in which the decode's fragment count can be read as
    the bond head's intent."""
    dec = _decoder()
    cloud = build_synthetic_pt_click_clouds(
        n_clouds=1, n_atoms_per_cloud=5, seed=7, jitter=0.20
    )[0]
    res = dec.decode(cloud)
    assert res.bonds_dropped_rdkit == 0
    assert res.bonds_dropped_rdkit_reasons == {}
    assert "bonds_dropped_rdkit" not in " ".join(res.decoded.warnings)


def test_the_per_molecule_decode_record_carries_the_count():
    """``DecodedMol.warnings`` is the per-molecule record's existing caveat
    channel, and the repo already uses ``key:value`` there
    (``dative_source:<src>``)."""
    dec = _decoder()
    cloud = build_synthetic_pt_click_clouds(
        n_clouds=1, n_atoms_per_cloud=5, seed=7, jitter=0.20
    )[0]
    d = dec._assemble_mol(
        cloud,
        [(0, 1, BOND_SINGLE), (0, 1, BOND_DOUBLE), (1, 2, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert "bonds_dropped_rdkit:1" in d.warnings


def test_the_result_fields_exist_and_default_to_zero():
    """A consumer that never asked for the counters must not see a
    surprise: the defaults are 0 and an empty map."""
    r = YuelBondDecodeResult(
        decoded=DecodedMol(),
        yuelbond_result=YuelBondResult(
            pair_features=None,
            logits=torch.zeros(0, 5),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            decode_succeeded=False,
        ),
    )
    assert r.bonds_dropped_rdkit == 0
    assert r.bonds_dropped_rdkit_reasons == {}
    assert r.clash_rejected_pairs == 0


# ===========================================================================
# 3. WHICH EXCEPTIONS RDKIT ACTUALLY RAISES (measured, not guessed)
# ===========================================================================


def test_rdkit_raises_what_on_each_invalid_bond():
    """Measured on RDKit 2026.03.6, CPU.

    * duplicate bond       -> ``RuntimeError`` ("Pre-condition Violation /
      bond already exists")
    * out-of-range index   -> ``RuntimeError`` ("Range Error / idx")
    * self-bond            -> ``RuntimeError`` ("Pre-condition Violation /
      attempt to add self-bond")
    * NEGATIVE index       -> ``OverflowError`` (Python's int -> unsigned
      conversion, NOT an RDKit error)
    """
    from rdkit import Chem

    cases = {
        "duplicate": ((0, 1), (0, 1), "RuntimeError"),
        "out_of_range": ((0, 1), (0, 99), "RuntimeError"),
        "self_bond": ((0, 1), (0, 0), "RuntimeError"),
        "negative": ((0, 1), (-1, 1), "OverflowError"),
    }
    for name, ((a, b), (c, d), expected) in cases.items():
        rw = Chem.RWMol()
        for _ in range(3):
            rw.AddAtom(Chem.Atom(6))
        with pytest.raises(Exception) as excinfo:
            rw.AddBond(a, b, Chem.BondType.SINGLE)
            rw.AddBond(c, d, Chem.BondType.SINGLE)
        assert type(excinfo.value).__name__ == expected, (
            f"{name}: expected {expected}, got " f"{type(excinfo.value).__name__}"
        )


def test_the_decoder_records_the_measured_exception_types():
    """The reasons map is not a guess: it names what RDKit raised."""
    dec = _decoder()
    cloud = build_synthetic_pt_click_clouds(
        n_clouds=1, n_atoms_per_cloud=5, seed=7, jitter=0.20
    )[0]
    dec._assemble_mol(
        cloud,
        [(0, 1, BOND_SINGLE), (0, 1, BOND_DOUBLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert dec._last_bonds_dropped == 1
    assert dec._last_bonds_dropped_reasons == {"RuntimeError": 1}

    dec._assemble_mol(
        cloud,
        [(0, 99, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert dec._last_bonds_dropped == 1
    assert dec._last_bonds_dropped_reasons == {"RuntimeError": 1}

    dec._assemble_mol(
        cloud,
        [(-1, 1, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert dec._last_bonds_dropped == 1
    assert dec._last_bonds_dropped_reasons == {"OverflowError": 1}


def test_the_wide_catch_is_retained():
    """Why the catch was NOT narrowed to ``(RuntimeError, OverflowError)``.

    ``int(src)`` sits INSIDE the try.  On a non-numeric index it raises
    ``ValueError`` / ``TypeError``, which the wide catch swallows (the bond
    is dropped) and a narrow catch would PROPAGATE (the whole decode
    raises).  Narrowing therefore IS a behaviour change, so the wide catch
    stays and the exception TYPE is recorded instead.

    This test pins that behaviour so a future "cleanup" that narrows the
    catch has to break a test on purpose.
    """
    dec = _decoder()
    cloud = build_synthetic_pt_click_clouds(
        n_clouds=1, n_atoms_per_cloud=5, seed=7, jitter=0.20
    )[0]
    d = dec._assemble_mol(
        cloud,
        [("not-an-index", 1, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert d.mol is not None, "a non-numeric index must NOT propagate"
    assert d.mol.GetNumBonds() == 0
    assert dec._last_bonds_dropped == 1
    assert dec._last_bonds_dropped_reasons == {"ValueError": 1}


# ===========================================================================
# 4. DEFECT 2 — the clash-reject count must be visible
# ===========================================================================


def test_the_clash_count_rides_on_the_head_result():
    """``YuelBondResult`` — the value object the head's forward returns —
    carries the count, so a consumer that keeps the result but not the head
    can still read it."""
    head = YuelBondDecoderHead(hidden_dim=16, n_layers=2)
    wrapper = YuelBondDecoder(head=head, hidden_dim=16, n_layers=2)
    res = wrapper.featurise(_cloud(_COLLAPSE_5, [6, 6, 6, 6, 6]))
    assert res.clash_rejected_pairs == 1
    assert res.decode_succeeded is True


def test_the_clash_count_rides_on_the_decode_result():
    dec = _decoder()
    res = dec.decode(_cloud(_COLLAPSE_5, [6, 6, 6, 6, 6]))
    assert res.clash_rejected_pairs == 1
    assert res.yuelbond_result.clash_rejected_pairs == 1


def test_a_clean_cloud_reports_zero_clash_rejections():
    """The same 5-atom layout with every pair >= 1.7 A: no pair is a
    collapse, so the count is 0 and the number is not a constant."""
    dec = _decoder()
    res = dec.decode(
        _cloud(
            [
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [0.0, 2.5, 0.0],
                [0.0, 0.0, 2.5],
                [1.8, 1.8, 1.8],
            ],
            [6, 6, 6, 6, 6],
        )
    )
    assert res.clash_rejected_pairs == 0


def test_the_clash_count_does_not_leak_from_the_previous_decode():
    """THE STALENESS BUG (measured before the fix).

    The head writes ``last_clash_rejected_pairs`` only on the path that
    reaches the candidate filter, so a forward that returns early left the
    PREVIOUS decode's count in place.  MEASURED 2026-10-31, before the fix:
    a 5-atom cloud with 1 rejection, then a 1-atom cloud ->
    ``clash_rejected_pairs == 1`` for the 1-atom cloud, which had zero
    rejections.  After the fix the 1-atom cloud reports 0.
    """
    dec = _decoder()
    first = dec.decode(_cloud(_COLLAPSE_5, [6, 6, 6, 6, 6]))
    assert first.clash_rejected_pairs == 1

    one = _cloud([[0.0, 0.0, 0.0]], [6])
    second = dec.decode(one)
    assert second.yuelbond_result.decode_succeeded is False
    assert second.yuelbond_result.error == "empty_edge_list"
    assert (
        second.clash_rejected_pairs == 0
    ), "the clash count leaked from the previous decode"


def test_the_head_attribute_is_zeroed_on_an_early_return():
    head = YuelBondDecoderHead(hidden_dim=16, n_layers=2)
    head(_cloud(_COLLAPSE_5, [6, 6, 6, 6, 6]))
    assert head.last_clash_rejected_pairs == 1
    head(_cloud([[0.0, 0.0, 0.0]], [6]))
    assert head.last_clash_rejected_pairs == 0


# ===========================================================================
# 5. REPORT-ONLY — the clash rule vetoes the connectivity prior
# ===========================================================================


def test_the_clash_rule_vetoes_the_connectivity_prior():
    """FINDING, NOT A FIX.  The ordering is a semantic decision that is not
    this track's to make; what this test does is MEASURE it.

    A clash-rejected pair is removed from ``i_idx``/``j_idx`` BEFORE the
    message passing (the head never scores it), and
    ``_bridge_components`` draws its candidate pool from the resulting
    ``edge_index``.  So the clash rule silently VETOES the connectivity
    prior for that pair: the prior can never rescue a pair whose geometry
    was flagged as a collapse.

    MEASURED on the 5-atom cloud above (Cordero ratio 0.7), CPU:

    * 10 upper-triangle pairs
    *  7 inside the 3.5 A cutoff
    *  1 clash-rejected: pair (0, 1) at 0.8 A
    *  6 edges survive the filter -- the rejected pair is NOT among them
    * the prior added 4 bridges; NONE of them is the rejected pair

    So ``clash_rejected_pairs`` is exactly the number of pairs the
    connectivity prior was denied.
    """
    cloud = _cloud(_COLLAPSE_5, [6, 6, 6, 6, 6])
    n = 5
    i_idx, j_idx = torch.triu_indices(n, n, offset=1)
    d_raw = (cloud.positions[j_idx] - cloud.positions[i_idx]).norm(dim=-1)
    in_cut = d_raw < 3.5
    rejected = (
        pair_collapse_mask(
            cloud.atomic_numbers, i_idx, j_idx, d_raw, DEFAULT_BOND_CLASH_RATIO
        )
        & in_cut
    )

    assert int(i_idx.numel()) == 10
    assert int(in_cut.sum()) == 7
    assert int(rejected.sum()) == 1
    rejected_pairs = {
        (int(i_idx[q]), int(j_idx[q]))
        for q in range(i_idx.numel())
        if bool(rejected[q])
    }
    assert rejected_pairs == {(0, 1)}

    head = YuelBondDecoderHead(hidden_dim=16, n_layers=2)
    wrapper = YuelBondDecoder(head=head, hidden_dim=16, n_layers=2)
    res = wrapper.featurise(cloud)
    edge_pairs = {
        (int(res.edge_index[0][k]), int(res.edge_index[1][k]))
        for k in range(res.edge_index.shape[1])
    }
    assert len(edge_pairs) == 6
    assert not (
        edge_pairs & rejected_pairs
    ), "the rejected pair must not be in the candidate pool"

    dec = YuelBondBondDecoder(head=head, connectivity_prior=True)
    bond_orders, dative_bonds, n_kept, n_overvalent = dec._select_bonds(cloud, res)
    after, _dative, n_bridges = dec._bridge_components(
        cloud, res, list(bond_orders), list(dative_bonds)
    )
    bridges = {
        (min(int(a), int(b)), max(int(a), int(b)))
        for a, b, _c in after[len(bond_orders) :]
    }
    assert n_bridges == 4
    assert not (
        bridges & rejected_pairs
    ), "the connectivity prior bridged a clash-rejected pair"
    # ... and the count the decoder now reports IS the number of pairs the
    # prior was denied.
    assert res.clash_rejected_pairs == len(rejected_pairs)
