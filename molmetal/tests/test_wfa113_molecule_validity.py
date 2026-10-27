"""Wf-A113 — a generation metric that CAN fail.

``gen_validity`` counts RDKit-strict parses, and ``C.C.C.C`` parses.  MEASURED
on ``molmetal/reports/cfm_c8_fullgen_2026_10_24`` it read 1.000 at all three
epochs while the same pool averaged 2.5 fragments per molecule.  The product
``combined = gen_validity x bond_half`` therefore could not rank epochs, which
is the defect ``todo.json`` records as SA-7 / D29.

These contracts pin the replacement:

1. the old predicate ACCEPTS a loose-atom string (so the test cannot pass by
   accident if someone "fixes" it upstream);
2. ``molecule_is_valid`` rejects the same string;
3. it accepts a single-fragment metal complex;
4. it rejects a metal-free molecule and a one-heavy-atom fragment;
5. ``molecule_validity`` is a selectable early-stop criterion and DEGRADES
   when the epoch did not measure it, rather than gating on a fabricated 0.0.

CPU-only, no checkpoint, no sampler.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def cfm():
    try:
        from molmetal.scripts import train_cfm_yuelbond_wasserstein as _m
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"CFM trainer unavailable: {exc}")
    return _m


#: Four heavy atoms, four fragments, sanitises cleanly.  This is the shape the
#: c8 pool produced: the decode is "valid" by parse and useless as a molecule.
LOOSE = "[Pt].[C].[C].[C]"
#: One connected Pt(II) centre with two N and two Cl donors.
COMPLEX = "[Pt](N)(N)(Cl)Cl"
#: Connected, but no metal at all.
ORGANIC = "CCO"
#: A lone metal atom: one heavy atom, one fragment.
BARE = "[Pt]"


def test_old_predicate_accepts_loose_atoms(cfm):
    """The blind spot is real, not assumed: RDKit-strict says OK here."""
    assert cfm._rdkit_strict_reason(LOOSE, True, "") is None


def test_molecule_is_valid_rejects_loose_atoms(cfm):
    row = cfm.decode_chemistry(LOOSE)
    assert row is not None, "the loose-atom string must still parse"
    assert row["n_fragments"] == 4
    assert row["has_metal"] is True
    assert cfm.molecule_is_valid(row, True) is False


def test_molecule_is_valid_accepts_single_fragment_complex(cfm):
    row = cfm.decode_chemistry(COMPLEX)
    assert row is not None
    assert row["n_fragments"] == 1
    assert row["has_metal"] is True
    assert cfm.molecule_is_valid(row, True) is True


def test_molecule_is_valid_rejects_metal_free_and_bare_metal(cfm):
    organic = cfm.decode_chemistry(ORGANIC)
    assert organic is not None and organic["has_metal"] is False
    assert cfm.molecule_is_valid(organic, True) is False

    bare = cfm.decode_chemistry(BARE)
    assert bare is not None and bare["n_heavy"] == 1
    assert cfm.molecule_is_valid(bare, True) is False


def test_molecule_is_valid_is_a_strict_refinement(cfm):
    """Every accepted sample was already accepted by the parse predicate."""
    for smi in (LOOSE, COMPLEX, ORGANIC, BARE):
        row = cfm.decode_chemistry(smi)
        strict_ok = cfm._rdkit_strict_reason(smi, True, "") is None
        if cfm.molecule_is_valid(row, strict_ok):
            assert strict_ok, smi


def test_unknown_rows_are_never_valid(cfm):
    assert cfm.molecule_is_valid(None, True) is False
    assert cfm.molecule_is_valid({}, True) is False
    # A row exists but the decode did not pass RDKit-strict.
    assert cfm.molecule_is_valid(cfm.decode_chemistry(COMPLEX), False) is False


def test_molecule_validity_is_selectable_and_degrades(cfm):
    assert "molecule_validity" in cfm.EARLY_STOP_METRICS

    # Measured: used as asked.
    name, value, note = cfm.resolve_early_stop_criterion(
        "molecule_validity",
        decode_ratio=0.9,
        gen_validity=1.0,
        gen_n=64,
        bond_exact_type_rate=0.4,
        bond_n_scored=100.0,
        molecule_validity=0.25,
    )
    assert (name, value) == ("molecule_validity", 0.25)
    assert note == ""

    # Saturated at 0.0 is recorded, not silently treated as signal.
    name, value, note = cfm.resolve_early_stop_criterion(
        "molecule_validity",
        decode_ratio=0.9,
        gen_validity=1.0,
        gen_n=64,
        bond_exact_type_rate=0.4,
        bond_n_scored=100.0,
        molecule_validity=0.0,
    )
    assert name == "molecule_validity" and value == 0.0
    assert "saturated" in note

    # Not measured this epoch (gen_eval_n=0): must degrade, never report 0.0
    # as if it were a measurement.
    name, _value, note = cfm.resolve_early_stop_criterion(
        "molecule_validity",
        decode_ratio=0.9,
        gen_validity=0.0,
        gen_n=0,
        bond_exact_type_rate=0.4,
        bond_n_scored=100.0,
        molecule_validity=None,
    )
    assert name != "molecule_validity"
    assert note, "a degraded criterion must say what gated instead"


def test_gen_validity_definition_is_published(cfm):
    """The name change is data, not prose: the artifact must carry it."""
    src = cfm.__file__
    text = open(src, encoding="utf-8").read()
    assert "rdkit_parse_rate: Chem.MolFromSmiles + SanitizeMol accept" in text
    # Both the pooled and the per-call site write it.
    assert text.count('"gen_validity_definition"') >= 2
