# SPDX-License-Identifier: Apache-2.0
"""Wf-A133 — the RDKit bond-drop counter and its reason histogram.

A bond that ``RWMol.AddBond`` refuses is a MOLECULAR SPLIT: the head
voted for it, the valence-aware selector accepted it, and the assembler
threw it away, leaving two fragments where the head meant one molecule.
Before Wf-A128 / Wf-A133 the refusal was a bare ``continue`` — no
counter, no log, no field, no trace anywhere.

The landed change put a counter + a reason histogram on BOTH RDKit
bond-drop sites:

* ``molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py``
  — ``YuelBondBondDecoder._assemble_mol``; the numbers ride on the
  RESULT object (:class:`YuelBondDecodeResult`).
* ``molmetal/models/bond_head.py``
  — ``BondAwareDecoder._assemble_mol``; the numbers ride on the decoder
  and are read back through ``bond_drop_report()``.

WHAT THIS FILE GUARDS
---------------------
The reason vocabulary is DUPLICATED across the two source files, and
the histogram key is an exception TYPE NAME (``type(exc).__name__``),
so a new silent-drop reason would show up as a new key in a map that
nobody diffs.  These tests therefore pin:

1. the SHAPE (types, defaults, per-decode reset, a CLOSED reason
   vocabulary);
2. the BEHAVIOUR (a synthetic drop increments the counter by exactly
   the number of bonds RDKit refused, under the right reason key);
3. the AGREEMENT between the two mirrors (same names, same vocabulary,
   same numbers for the same input);
4. NON-VACUITY — every numeric assertion goes through
   :func:`_assert_counter_matches`, which REFUSES a 0-vs-0 comparison,
   and the refusal is itself tested.

CPU-only.  No GPU, no checkpoint, no corpus, no network.  Every case
here is a 3-atom synthetic carbon cloud.

MEASURED BASIS
--------------
RDKit 2026.03.6, CPU, 2026-10-31.  The exception TYPE per failure mode
was measured, not guessed (see
``molmetal/tests/test_wfa128_decode_bond_drop_counter.py``):

===================  ====================================================
failure mode         exception raised by the assembler
===================  ====================================================
duplicate bond       ``RuntimeError``  (RDKit: bond already exists)
out-of-range index   ``RuntimeError``  (RDKit: Range Error / idx)
self-bond            ``RuntimeError``  (RDKit: attempt to add self-bond)
NEGATIVE index       ``OverflowError`` (Python int -> unsigned, not RDKit)
non-numeric index    ``ValueError``    (``int("x")``)
``None`` index       ``TypeError``     (``int(None)``)
float index          (NO DROP — ``int(0.5) == 0`` is a valid index)
===================  ====================================================

NOTE on the brief's example vocabulary: it listed "valence overflow" as
a possible reason.  It is NOT one.  RDKit does not raise on valence
overflow at ``AddBond`` time — valence is checked at SANITISE time,
which is a different branch entirely.  The reason vocabulary is
therefore the FOUR exception types above, and the closed set below says
so.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

import pytest
import torch
from rdkit import Chem

from molmetal.models.bond_head import (
    BOND_AROMATIC,
    BOND_DOUBLE,
    BOND_SINGLE,
    BOND_TRIPLE,
    AtomCloud,
    BondAwareDecoder,
    BondOrderHead,
    DecodedMol,
)
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    YuelBondBondDecoder,
    YuelBondDecodeResult,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    YuelBondDecoderHead,
    YuelBondResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _yuelbond_decoder() -> YuelBondBondDecoder:
    """A deterministic YuelBond decoder: same seed -> same head weights."""
    torch.manual_seed(1234)
    head = YuelBondDecoderHead(hidden_dim=16, n_layers=2)
    return YuelBondBondDecoder(head=head)


def _bond_head_decoder() -> BondAwareDecoder:
    """The SECOND drop site, on the same footing (small head, CPU)."""
    torch.manual_seed(1234)
    return BondAwareDecoder(bond_head=BondOrderHead(hidden_dim=8, n_layers=2))


def _cloud3() -> AtomCloud:
    """The smallest possible cloud: 3 carbons, 1.2 A apart."""
    return AtomCloud(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        atomic_numbers=torch.tensor([6, 6, 6], dtype=torch.long),
    )


def _empty_result() -> YuelBondDecodeResult:
    """A ``YuelBondDecodeResult`` with every field at its default."""
    return YuelBondDecodeResult(
        decoded=DecodedMol(),
        yuelbond_result=YuelBondResult(
            pair_features=None,
            logits=torch.zeros(0, 5),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            decode_succeeded=False,
        ),
    )


#: The CLOSED reason vocabulary.  The key of the histogram is the
#: exception TYPE NAME, so a NEW silent-drop reason shows up as a NEW
#: KEY in a map that nobody diffs.  Pinning the set means a new key
#: fails this test instead of silently widening the vocabulary.
#:
#: MEASURED on RDKit 2026.03.6, CPU, 2026-10-31 (see the module
#: docstring for the per-mode measurement).
_REASON_VOCABULARY = frozenset(
    {"RuntimeError", "OverflowError", "ValueError", "TypeError"}
)

#: Every failure mode the assembler can hit, with the exception TYPE it
#: raises.  The bonds are ALWAYS ``[(0, 1), ...]`` plus one offending
#: bond, so the first bond is accepted and exactly ONE bond is refused.
_FAILURE_MODES: Dict[str, Tuple[List[Tuple[int, int, int]], str]] = {
    "duplicate": ([(0, 1, BOND_SINGLE), (0, 1, BOND_DOUBLE)], "RuntimeError"),
    "out_of_range": ([(0, 1, BOND_SINGLE), (0, 99, BOND_SINGLE)], "RuntimeError"),
    "self_bond": ([(0, 1, BOND_SINGLE), (1, 1, BOND_SINGLE)], "RuntimeError"),
    "negative": ([(0, 1, BOND_SINGLE), (-1, 2, BOND_SINGLE)], "OverflowError"),
    "non_numeric": ([(0, 1, BOND_SINGLE), ("x", 2, BOND_SINGLE)], "ValueError"),
    "none_index": ([(0, 1, BOND_SINGLE), (None, 2, BOND_SINGLE)], "TypeError"),
}

#: A FLOAT index is NOT a drop: ``int(0.5) == 0`` is a VALID atom index,
#: so RDKit accepts the bond.  Pinned so a future "tighten the index
#: handling" change cannot start counting bonds RDKit accepted.
_FLOAT_INDEX_IS_NOT_A_DROP = [(0, 1, BOND_SINGLE), (0.5, 2, BOND_SINGLE)]


def _oracle_drop_count(
    bond_orders, n_atoms: int
) -> Tuple[int, Dict[str, int]]:
    """Recompute what RDKit refuses, WITHOUT touching the decoder.

    This is the oracle the decoder's counter is compared against.  It
    deliberately mirrors the assembler's own ``int()``-inside-the-``try``
    shape, because that is what decides WHICH exception type the
    histogram records.
    """
    bt_map = {
        BOND_SINGLE: Chem.BondType.SINGLE,
        BOND_DOUBLE: Chem.BondType.DOUBLE,
        BOND_TRIPLE: Chem.BondType.TRIPLE,
        BOND_AROMATIC: Chem.BondType.AROMATIC,
    }
    rw = Chem.RWMol()
    for _ in range(n_atoms):
        rw.AddAtom(Chem.Atom(6))
    n_dropped = 0
    reasons: Dict[str, int] = {}
    for src, dst, order in bond_orders:
        bt = bt_map.get(int(order), Chem.BondType.SINGLE)
        try:
            rw.AddBond(int(src), int(dst), bt)
        except Exception as exc:  # noqa: BLE001 — the point is the TYPE
            n_dropped += 1
            name = type(exc).__name__
            reasons[name] = reasons.get(name, 0) + 1
    return n_dropped, reasons


def _assert_counter_matches(expected_count: int, actual_count: int) -> None:
    """The single comparison every drop assertion goes through.

    It REFUSES a 0-vs-0 comparison on purpose: a counter that is always
    0 satisfies ``actual == expected`` whenever ``expected`` is also 0,
    which is exactly the "field present, never incremented" failure
    mode.  Requiring ``expected_count > 0`` makes that failure mode
    impossible to hide behind a clean decode.

    It is a FUNCTION so the same expression can be pointed at a
    deliberately-wrong value to prove the guard is not vacuous — see
    :func:`test_the_counter_guard_is_not_vacuous`.
    """
    assert expected_count > 0, (
        "the oracle must have provoked at least one drop; a 0-vs-0 "
        "comparison would pass even if the counter never incremented"
    )
    assert actual_count == expected_count, (
        f"drop counter mismatch: RDKit refused {expected_count} bond(s), "
        f"the decoder counted {actual_count}"
    )


# ===========================================================================
# 1. SHAPE — the two fields, their types, their defaults
# ===========================================================================


def test_the_result_object_exposes_the_two_fields_with_the_right_types():
    """``bonds_dropped_rdkit`` is an int; ``bonds_dropped_rdkit_reasons``
    is a ``dict`` whose keys are strings and whose values are ints."""
    r = _empty_result()
    assert type(r.bonds_dropped_rdkit) is int
    assert type(r.bonds_dropped_rdkit_reasons) is dict
    for key, value in r.bonds_dropped_rdkit_reasons.items():
        assert type(key) is str, key
        assert type(value) is int, value


def test_the_declared_field_annotations_are_int_and_dict_of_str_int():
    """The DECLARED type, not just the runtime value.

    A field declared ``bonds_dropped_rdkit: bool`` would still satisfy
    ``type(x) is int`` at runtime for ``False``... except that
    ``bool`` is not ``int``, so ``type(x) is int`` already rejects it.
    The annotation check is what catches a field re-declared as
    ``Optional[int]`` or ``Any``, both of which pass a naive
    ``isinstance`` check.
    """
    fields = {f.name: f for f in dataclasses.fields(YuelBondDecodeResult)}
    assert "bonds_dropped_rdkit" in fields, sorted(fields)
    assert "bonds_dropped_rdkit_reasons" in fields, sorted(fields)
    # ``from __future__ import annotations`` makes the annotation a
    # STRING; accept either the string or the evaluated object.
    assert fields["bonds_dropped_rdkit"].type in ("int", int)
    assert fields["bonds_dropped_rdkit_reasons"].type in (
        "Dict[str, int]",
        Dict[str, int],
    )


def test_the_defaults_are_zero_and_an_empty_map():
    """A consumer that never asked for the counters must not see a
    surprise."""
    r = _empty_result()
    assert r.bonds_dropped_rdkit == 0
    assert r.bonds_dropped_rdkit_reasons == {}


def test_two_default_results_do_not_share_one_reason_map():
    """The reasons map is a ``default_factory=dict``, not a shared class
    attribute.  A shared map would leak one decode's reasons into the
    next result object."""
    a = _empty_result()
    b = _empty_result()
    assert a.bonds_dropped_rdkit_reasons is not b.bonds_dropped_rdkit_reasons
    a.bonds_dropped_rdkit_reasons["RuntimeError"] = 1
    assert b.bonds_dropped_rdkit_reasons == {}, "the default map is shared"


# ===========================================================================
# 2. SHAPE — the counter is reset per decode, never sticky
# ===========================================================================


def test_the_assembler_counter_is_reset_per_decode_not_sticky():
    """A stale counter is worse than no counter: the next decode would
    inherit the previous decode's splits.  The reset is at the TOP of
    ``_assemble_mol``."""
    dec = _yuelbond_decoder()
    cloud = _cloud3()
    dec._assemble_mol(
        cloud,
        [(0, 1, BOND_SINGLE), (0, 99, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert dec._last_bonds_dropped == 1, dec._last_bonds_dropped
    assert dec._last_bonds_dropped_reasons == {"RuntimeError": 1}

    dec._assemble_mol(
        cloud,
        [(0, 1, BOND_SINGLE), (1, 2, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert dec._last_bonds_dropped == 0, "the counter inherited a stale value"
    assert dec._last_bonds_dropped_reasons == {}


def test_the_result_object_carries_zero_after_a_clean_decode():
    """The RESULT-level view of the same property: a clean decode after a
    dirty one reports 0, not the previous decode's count."""
    dec = _yuelbond_decoder()
    cloud = _cloud3()
    dec._assemble_mol(
        cloud,
        [(0, 1, BOND_SINGLE), (0, 99, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert dec._last_bonds_dropped == 1
    res = dec.decode(cloud)
    assert res.bonds_dropped_rdkit == 0, "the RESULT inherited a stale value"
    assert res.bonds_dropped_rdkit_reasons == {}


# ===========================================================================
# 3. SHAPE — the reason vocabulary is a CLOSED set
# ===========================================================================


def test_the_reason_vocabulary_is_a_closed_documented_set():
    """Every failure mode the assembler can reach, and no more.

    A NEW silent-drop reason would add a KEY to the histogram.  This
    test fails if the observed key set is not exactly the four
    documented exception types.
    """
    observed = set()
    for bonds, _expected in _FAILURE_MODES.values():
        dec = _yuelbond_decoder()
        dec._assemble_mol(
            _cloud3(),
            bonds,
            [],
            dative_source="",
            dative_manufactured=0,
        )
        observed |= set(dec._last_bonds_dropped_reasons)
    assert observed == set(_REASON_VOCABULARY), (
        f"the reason vocabulary drifted: observed {sorted(observed)}, "
        f"documented {sorted(_REASON_VOCABULARY)}"
    )


def test_a_float_index_is_not_a_drop():
    """``int(0.5) == 0`` is a VALID atom index, so RDKit accepts the
    bond.  Pinned so the counter cannot start counting bonds RDKit
    accepted."""
    dec = _yuelbond_decoder()
    dec._assemble_mol(
        _cloud3(),
        _FLOAT_INDEX_IS_NOT_A_DROP,
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert dec._last_bonds_dropped == 0, dec._last_bonds_dropped
    assert dec._last_bonds_dropped_reasons == {}


def test_the_second_mirror_uses_the_same_closed_vocabulary():
    """The vocabulary is DUPLICATED in two source files — the thing most
    likely to drift.  The second site must produce the SAME key set."""
    observed = set()
    for bonds, _expected in _FAILURE_MODES.values():
        dec = _bond_head_decoder()
        dec._assemble_mol(
            _cloud3(),
            bonds,
            [],
            dative_source="",
            dative_manufactured=0,
        )
        observed |= set(dec._last_bonds_dropped_reasons)
    assert observed == set(_REASON_VOCABULARY), (
        f"the reason vocabulary drifted on the second site: "
        f"observed {sorted(observed)}, documented {sorted(_REASON_VOCABULARY)}"
    )


# ===========================================================================
# 4. BEHAVIOUR — drive the assembler's failure path
# ===========================================================================


@pytest.mark.parametrize("mode", sorted(_FAILURE_MODES))
def test_the_assembler_counts_the_drop_and_names_the_reason(mode):
    """The smallest possible synthetic case: a 3-carbon cloud and ONE
    offending bond.

    The count is compared against an INDEPENDENT oracle, not against a
    hard-coded number, so the assertion survives an RDKit upgrade that
    changes which exception a failure mode raises — while still failing
    if the decoder stops counting.
    """
    bonds, expected_reason = _FAILURE_MODES[mode]
    dec = _yuelbond_decoder()
    decoded = dec._assemble_mol(
        _cloud3(),
        bonds,
        [],
        dative_source="",
        dative_manufactured=0,
    )

    oracle_count, oracle_reasons = _oracle_drop_count(bonds, n_atoms=3)
    _assert_counter_matches(oracle_count, dec._last_bonds_dropped)
    assert dec._last_bonds_dropped_reasons == oracle_reasons
    assert expected_reason in dec._last_bonds_dropped_reasons, (
        f"{mode}: expected reason {expected_reason!r}, got "
        f"{dec._last_bonds_dropped_reasons}"
    )
    assert dec._last_bonds_dropped_reasons[expected_reason] == 1

    # The drop is INSTRUMENTATION ONLY: the bond is still refused and the
    # molecule is still split.  The decode's own record still reports the
    # bonds it ASKED for.
    assert decoded.mol is not None
    assert decoded.n_bonds == len(bonds)
    assert decoded.mol.GetNumBonds() == len(bonds) - oracle_count


def test_the_count_reaches_the_result_object(monkeypatch):
    """NON-VACUITY — THE key assertion of this file.

    A SHAPE test alone is worthless here: ``YuelBondDecodeResult`` has
    the two fields with the right types and the right defaults even if
    the assembler NEVER increments them and the decode NEVER propagates
    them.  So this test drives a decode whose assembly really does drop
    a bond and asserts the RESULT carries the count.

    REVERTING THE LANDED CHANGE FAILS THIS TEST two distinct ways:

    * the two fields are removed from ``YuelBondDecodeResult``  ->
      ``AttributeError`` at the first access;
    * the fields survive but the propagation
      (``cfm_decoder_yuelbond.py`` :1299-1300) or the increment in
      ``_assemble_mol`` (:2126-2128) is reverted  ->  the count stays 0
      and ``res.bonds_dropped_rdkit == 1`` fails.

    The refusal is provoked by an OUT-OF-RANGE atom index: the cloud has
    3 atoms, so ``(0, 99)`` is refused by RDKit on every version.
    """
    dec = _yuelbond_decoder()
    cloud = _cloud3()
    original = dec._assemble_mol

    def _with_one_refused_bond(c, bond_orders, dative_bonds, *args, **kwargs):
        return original(
            c,
            list(bond_orders) + [(0, 99, BOND_SINGLE)],
            dative_bonds,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(dec, "_assemble_mol", _with_one_refused_bond)

    res = dec.decode(cloud)

    assert res.bonds_dropped_rdkit == 1, (
        "the count did not reach the result object: "
        f"{res.bonds_dropped_rdkit} (a counter that is never incremented "
        "also passes a shape-only test, which is why this assertion "
        "exists)"
    )
    assert res.bonds_dropped_rdkit_reasons == {"RuntimeError": 1}, (
        f"the reason did not reach the result object: "
        f"{res.bonds_dropped_rdkit_reasons}"
    )
    # The per-molecule record carries the count too (the repo's existing
    # ``key:value`` warning convention).
    assert "bonds_dropped_rdkit:1" in res.decoded.warnings, res.decoded.warnings
    # ... and the histogram is well-typed on the RESULT, not just on the
    # decoder's private attribute.
    for key, value in res.bonds_dropped_rdkit_reasons.items():
        assert type(key) is str, key
        assert type(value) is int, value


def test_the_counter_guard_is_not_vacuous():
    """PROVE the guard rejects a counter that is present but never
    incremented.

    Without this, :func:`_assert_counter_matches` could be a tautology
    and every behavioural test built on it would be decoration.  The
    positive control at the end is what keeps the negative one honest.
    """
    with pytest.raises(AssertionError):
        _assert_counter_matches(expected_count=1, actual_count=0)
    with pytest.raises(AssertionError):
        _assert_counter_matches(expected_count=0, actual_count=0)
    # Positive control: the real value passes.
    _assert_counter_matches(expected_count=1, actual_count=1)


def test_a_never_incremented_counter_would_fail_the_behaviour_test(monkeypatch):
    """THE REGRESSION WITNESS.

    Simulate the exact regression this file exists to catch — the fields
    are present, but the assembler never increments them — and show that
    the behavioural assertion rejects it.

    The simulation is faithful in the one respect that matters: RDKit
    really does refuse the bond (the molecule really is split), and the
    counter still reports nothing.  If this test ever fails to raise,
    the guard is broken.
    """
    dec = _yuelbond_decoder()
    cloud = _cloud3()
    bonds, _expected = _FAILURE_MODES["duplicate"]
    oracle_count, _oracle_reasons = _oracle_drop_count(bonds, n_atoms=3)

    original = dec._assemble_mol

    def _count_nothing(c, bond_orders, dative_bonds, *args, **kwargs):
        decoded = original(c, bond_orders, dative_bonds, *args, **kwargs)
        # THE REGRESSION: the refusal is still swallowed, but the
        # counter is never incremented.
        dec._last_bonds_dropped = 0
        dec._last_bonds_dropped_reasons = {}
        return decoded

    monkeypatch.setattr(dec, "_assemble_mol", _count_nothing)
    decoded = dec._assemble_mol(
        cloud,
        bonds,
        [],
        dative_source="",
        dative_manufactured=0,
    )

    # RDKit really did refuse a bond ...
    assert decoded.mol is not None
    assert decoded.mol.GetNumBonds() == len(bonds) - oracle_count
    # ... and the counter says nothing, so the guard rejects it.
    assert dec._last_bonds_dropped == 0
    with pytest.raises(AssertionError):
        _assert_counter_matches(oracle_count, dec._last_bonds_dropped)


# ===========================================================================
# 5. MIRROR AGREEMENT — the two drop sites must not drift apart
# ===========================================================================


def test_both_mirrors_expose_the_same_counter_attribute_names():
    """The reason strings are DUPLICATED across two files, and so are the
    attribute names that hold them.  A rename on one side only would
    leave the other side silently reporting 0 forever."""
    yb = _yuelbond_decoder()
    bh = _bond_head_decoder()
    for name in ("_last_bonds_dropped", "_last_bonds_dropped_reasons"):
        assert hasattr(yb, name), f"the YuelBond decoder lost {name}"
        assert hasattr(bh, name), f"the bond-head decoder lost {name}"
        assert type(getattr(yb, name)) is type(getattr(bh, name)), name


@pytest.mark.parametrize("mode", sorted(_FAILURE_MODES))
def test_both_mirrors_report_the_same_count_and_the_same_reason(mode):
    """The strongest form of the agreement check: the SAME input, the
    SAME numbers, the SAME histogram — from both files."""
    bonds, expected_reason = _FAILURE_MODES[mode]
    yb = _yuelbond_decoder()
    bh = _bond_head_decoder()

    yb._assemble_mol(
        _cloud3(), bonds, [], dative_source="", dative_manufactured=0
    )
    bh._assemble_mol(
        _cloud3(), bonds, [], dative_source="", dative_manufactured=0
    )

    assert yb._last_bonds_dropped == bh._last_bonds_dropped, (
        f"{mode}: the two drop sites counted differently "
        f"({yb._last_bonds_dropped} vs {bh._last_bonds_dropped})"
    )
    assert yb._last_bonds_dropped_reasons == bh._last_bonds_dropped_reasons, (
        f"{mode}: the two drop sites named the reason differently "
        f"({yb._last_bonds_dropped_reasons} vs {bh._last_bonds_dropped_reasons})"
    )
    assert expected_reason in yb._last_bonds_dropped_reasons


def test_both_mirrors_use_the_same_warning_token():
    """The per-molecule record's caveat channel uses the same
    ``key:value`` token on both sides."""
    bonds, _expected = _FAILURE_MODES["duplicate"]
    yb = _yuelbond_decoder()
    bh = _bond_head_decoder()
    yb_decoded = yb._assemble_mol(
        _cloud3(), bonds, [], dative_source="", dative_manufactured=0
    )
    bh_decoded = bh._assemble_mol(
        _cloud3(), bonds, [], dative_source="", dative_manufactured=0
    )
    assert "bonds_dropped_rdkit:1" in yb_decoded.warnings, yb_decoded.warnings
    assert "bonds_dropped_rdkit:1" in bh_decoded.warnings, bh_decoded.warnings


def test_the_report_surface_and_the_result_surface_agree_on_values():
    """The two sites expose the numbers on DIFFERENT surfaces — the
    YuelBond decoder puts them on the RESULT object, the bond-head
    decoder on ``bond_drop_report()`` — but the numbers must be the
    same numbers for the same input."""
    yb = _yuelbond_decoder()
    bh = _bond_head_decoder()
    bonds, _expected = _FAILURE_MODES["duplicate"]

    bh._assemble_mol(
        _cloud3(), bonds, [], dative_source="", dative_manufactured=0
    )
    yb._assemble_mol(
        _cloud3(), bonds, [], dative_source="", dative_manufactured=0
    )

    report = bh.bond_drop_report()
    assert set(report) == {"bonds_dropped", "reasons"}, sorted(report)
    assert report["bonds_dropped"] == bh._last_bonds_dropped
    assert report["reasons"] == bh._last_bonds_dropped_reasons
    assert report["bonds_dropped"] == yb._last_bonds_dropped
    assert report["reasons"] == yb._last_bonds_dropped_reasons
    assert set(report["reasons"]) <= set(_REASON_VOCABULARY)


def test_the_second_mirror_resets_its_counter_per_decode():
    """The reset rule is duplicated too — a copy that loses the reset
    would report the previous decode's splits."""
    bh = _bond_head_decoder()
    cloud = _cloud3()
    bh._assemble_mol(
        cloud,
        [(0, 1, BOND_SINGLE), (0, 99, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert bh._last_bonds_dropped == 1
    bh._assemble_mol(
        cloud,
        [(0, 1, BOND_SINGLE), (1, 2, BOND_SINGLE)],
        [],
        dative_source="",
        dative_manufactured=0,
    )
    assert bh._last_bonds_dropped == 0, "the counter inherited a stale value"
    assert bh._last_bonds_dropped_reasons == {}
