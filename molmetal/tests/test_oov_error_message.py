"""Tests for IMPROVEMENT_BACKLOG §B2 — improved OOV atom error message.

The legacy behaviour was a bare ``AssertionError`` from
:func:`Molecule.from_smiles` with no indication of *which* element
caused the failure.  The :class:`LipmanFlowMatchingAdapter` now exposes
:meth:`_validate_smiles_against_vocab` that parses the SMILES with
RDKit and raises a :class:`ValueError` whose message:

* names the offending SMILES (repr-quoted),
* lists the exact OOV atomic numbers (sorted, deduplicated),
* shows the current vocab (sorted) so the user can see what is and
  is not allowed, and
* points at the source file / line where the vocab lives so the fix
  (extending ``_atom_vocab``) is a one-line edit.

These tests pin both the positive path (no error for in-vocab atoms)
and the negative path (the message carries the OOV element + the
current vocab).  Element 99 (Einsteinium, ``[Es]``) is used as the
canonical "guaranteed OOV" sentinel — it is well outside the default
20-element metallo-drug vocab
``(1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 21, 22, 33, 34, 35, 39, 40, 53, 72, 78)``.
"""

from __future__ import annotations

import re

import pytest

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter


# ---------------------------------------------------------------------------
# Helper: build the adapter without going through setup() (we don't need
# the velocity field for vocab validation).
# ---------------------------------------------------------------------------
def _make_adapter() -> LipmanFlowMatchingAdapter:
    """Construct an adapter with vocab_mask enabled (matches the
    production default).  We deliberately skip :meth:`setup` to avoid
    building the EGNN — vocab validation is a pure-Python + RDKit
    check that does not need a velocity field.
    """
    return LipmanFlowMatchingAdapter(
        hidden_dim=32,  # toy scale — setup()-time warning only
        n_layers=2,
        vocab_mask=True,
    )


# ---------------------------------------------------------------------------
# 1. Negative path — Einsteinium (Z=99) is OOV and must trigger the
#    improved error message that names the OOV element AND the current
#    vocab.
# ---------------------------------------------------------------------------
def test_oov_einsteinium_raises_with_improved_message() -> None:
    adapter = _make_adapter()
    # [Es] = Einsteinium, Z=99 — guaranteed to be outside the default
    # 20-element metallo-drug vocab.
    smiles_with_es = "[Es]"

    with pytest.raises(ValueError) as excinfo:
        adapter._validate_smiles_against_vocab(smiles_with_es)

    msg = str(excinfo.value)

    # 1. The message must mention the SMILES (repr-quoted, so [Es] is
    #    shown verbatim rather than as 'Es' without brackets).
    assert (
        "'[Es]'" in msg
    ), f"Error message must contain the repr-quoted SMILES; got:\n{msg}"

    # 2. The message must name the OOV element (Z=99) so the user
    #    knows exactly which atom to add to the vocab.
    assert (
        "99" in msg
    ), f"Error message must mention the OOV atomic number 99; got:\n{msg}"

    # 3. The message must show the current vocab — every element of
    #    the 20-element default vocab should appear in the message.
    for z in adapter._atom_vocab:
        assert str(z) in msg, (
            f"Error message must mention every vocab element "
            f"(missing {z}); got:\n{msg}"
        )

    # 4. The message should hint at the source file/line of the vocab.
    assert (
        "molmetal/adapters/flow_matching_lipman/__init__.py" in msg
    ), f"Error message must point at the vocab source file; got:\n{msg}"
    assert ":1825" in msg, (
        f"Error message must point at line 1825 of the vocab source; " f"got:\n{msg}"
    )


# ---------------------------------------------------------------------------
# 2. The negative-path message must use the documented phrasing from
#    IMPROVEMENT_BACKLOG §B2 — pinning the exact contract.
# ---------------------------------------------------------------------------
def test_oov_error_message_contract() -> None:
    adapter = _make_adapter()
    with pytest.raises(ValueError) as excinfo:
        adapter._validate_smiles_against_vocab("[Es]")
    msg = str(excinfo.value)

    # The contract from §B2 says the message contains three parts:
    #   (a) "OOV atoms not in vocab (size=N): [...]: ..."
    #   (b) a Hint line pointing at _atom_vocab:1825
    #   (c) "Current vocab: [...]"
    assert re.search(r"OOV atoms not in vocab", msg), (
        f"Message must contain the 'OOV atoms not in vocab' phrase; " f"got:\n{msg}"
    )
    assert re.search(r"Current vocab:", msg), (
        f"Message must contain the 'Current vocab:' phrase; " f"got:\n{msg}"
    )
    assert re.search(
        r"Hint:", msg, re.IGNORECASE
    ), f"Message must contain a 'Hint:' line; got:\n{msg}"


# ---------------------------------------------------------------------------
# 3. Multi-element OOV — the message must list every distinct OOV Z,
#    sorted ascending, and report the correct count.
# ---------------------------------------------------------------------------
def test_oov_multiple_atoms_lists_all_sorted() -> None:
    adapter = _make_adapter()
    # Z=99 (Es) + Z=100 (Fm) — both well outside the default vocab.
    # RDKit accepts [Es].[Fm] as a multi-component SMILES.
    smiles_multi = "[Es].[Fm]"

    with pytest.raises(ValueError) as excinfo:
        adapter._validate_smiles_against_vocab(smiles_multi)

    msg = str(excinfo.value)
    assert (
        "99" in msg and "100" in msg
    ), f"Both OOV atomic numbers must appear in the message; got:\n{msg}"
    # The list must be sorted ascending — 99 must come before 100.
    assert msg.index("99") < msg.index(
        "100"
    ), f"OOV atomic numbers must be sorted ascending; got:\n{msg}"
    # Count line: "has 2 OOV atoms" (not "has 1").
    assert (
        "has 2 OOV atoms" in msg
    ), f"Message must report 'has 2 OOV atoms'; got:\n{msg}"


# ---------------------------------------------------------------------------
# 4. Positive path — in-vocab SMILES must NOT raise.
# ---------------------------------------------------------------------------
def test_in_vocab_smiles_passes() -> None:
    adapter = _make_adapter()
    # Aspirin: C9H8O4 — only C (6), H (1 implicit), O (8).  All three
    # are in the default vocab.
    smiles_ok = "CC(=O)OC1=CC=CC=C1C(=O)O"
    # Should not raise.
    adapter._validate_smiles_against_vocab(smiles_ok)


# ---------------------------------------------------------------------------
# 5. Malformed SMILES — must raise a clear ValueError (not a silent
#    None-attribute-deref from RDKit).
# ---------------------------------------------------------------------------
def test_malformed_smiles_raises_value_error() -> None:
    adapter = _make_adapter()
    with pytest.raises(ValueError) as excinfo:
        adapter._validate_smiles_against_vocab("not_a_smiles@@@")
    msg = str(excinfo.value)
    assert (
        "could not be parsed by RDKit" in msg
    ), f"Malformed-SMILES error must be a clear ValueError; got:\n{msg}"
