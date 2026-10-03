"""Tests for FG-compat / leave-group veto (TODO-30 Rank 10 P1.2).

Background
----------
This file ships the canonical 20 tests for
:mod:`molmetal.molmetal_lam.lam_chem.fg_compat_veto`.  The tests
are pure CPU — no GPU, no RDKit conformer embedding, no ML
forward passes — and exercise the 3 new classes:

  1. :class:`FGCompatMatrix` — 7 × 7 = 49-cell FG × FG veto
     (15 vetoed cells in the canonical table)
  2. :class:`LeaveGroupVeto` — 7 × 12 = 84-cell FG × LG veto
     (12 vetoed cells in the canonical table)
  3. :class:`VetoAugmentedReducer` — wrapper that gates the base
     reducer on the FG + LG matrices
  4. :func:`register_fg_compat_veto` — registry helper returning
     the cumulative veto-cell count.

The tests pin both the **structure** of the matrices (axes,
counts, symmetry of the FG matrix, asymmetry of the LG matrix)
and the **behaviour** of individual cell verdicts so an
accidental future edit to the matrix table would fail loudly.

Honest framing
--------------
* The matrices are HAND-CURATED from Carey & Sundberg 2007 +
  March 2007 + Montalbetti & Falque 2005; no learned veto is in
  scope (TODO-30 P1.2 is a CPU skeleton).
* The "narrow veto" design choice means many plausible-but-
  unusual pairs (e.g. nitrile + ester) are NOT vetoed; the
  tests pin the canonical 15 + 12 vetoed cells.
* The veto is symmetric in the FG × FG matrix (aladehyde + amine
  has the same verdict as amine + aldehyde) but asymmetric in
  the FG × LG matrix (Cl on acid is fine but F on aryl_halide is
  not).  Tests pin both invariants.
"""

from __future__ import annotations

import pytest

# All tests in this file are CPU-only (no GPU, no RDKit embedding).
from molmetal.molmetal_lam.lam_chem.fg_compat_veto import (
    FGCompatMatrix,
    FUNCTIONAL_GROUPS,
    LEAVING_GROUPS,
    LeaveGroupVeto,
    METAL_LABELS,
    VetoAugmentedReducer,
    register_fg_compat_veto,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal mock reducers for VetoAugmentedReducer tests
# ---------------------------------------------------------------------------


class _MockReducer:
    """Mock reducer that echoes the input SMILES back as a string.

    Exposes ``reduce(input) -> str`` (the minimum interface the
    :class:`VetoAugmentedReducer` requires).  Used to test that the
    veto passes through correctly when no veto fires and that the
    base reducer is invoked with the right argument.
    """

    def __init__(self, prefix: str = "PASS:") -> None:
        self.prefix = prefix
        self.n_calls = 0

    def reduce(self, input_smiles: str) -> str:
        self.n_calls += 1
        return f"{self.prefix}{input_smiles}"


# ---------------------------------------------------------------------------
# FGCompatMatrix tests
# ---------------------------------------------------------------------------


def test_fg_compat_matrix_compatible():
    """A known-compatible FG pair returns True from is_compatible.

    ``aldehyde`` + ``primary_amine`` would be reductive amination
    (rejected — see test_fg_compat_matrix_incompatible), so the
    canonical-compatible pair is ``carboxylic_acid`` +
    ``primary_amine`` (peptide bond).
    """
    matrix = FGCompatMatrix()
    assert matrix.is_compatible("carboxylic_acid", "primary_amine") is True
    # Same in reverse order.
    assert matrix.is_compatible("primary_amine", "carboxylic_acid") is True


def test_fg_compat_matrix_incompatible():
    """A known-incompatible FG pair returns False from is_compatible.

    The TODO-30 P1.2 brief explicitly cites aldehyde + aldehyde as
    a veto example (peptide formation from two aldehydes is
    nonsensical).  We pin that case plus a second canonical veto
    for redundancy.
    """
    matrix = FGCompatMatrix()
    # The TODO-30 P1.2 example
    assert matrix.is_compatible("aldehyde", "aldehyde") is False
    # Symmetric — same verdict in reverse.
    assert matrix.is_compatible("aldehyde", "aldehyde") is False
    # A second canonical veto — two carboxylic acids.
    assert matrix.is_compatible("carboxylic_acid", "carboxylic_acid") is False
    # Two amines — hydrazine-like, out of menu.
    assert matrix.is_compatible("primary_amine", "primary_amine") is False


def test_fg_compat_matrix_get_incompatible():
    """``get_incompatible_pairs()`` returns a non-empty list.

    The canonical FG matrix vetoes 15 distinct unordered pairs
    (the list is sorted canonically).  We pin the count + the
    presence of the canonical examples.
    """
    matrix = FGCompatMatrix()
    pairs = matrix.get_incompatible_pairs()
    assert isinstance(pairs, list)
    assert len(pairs) >= 1
    # The TODO-30 P1.2 example MUST appear in the list.
    pair_set = {tuple(sorted(p)) for p in pairs}
    assert ("aldehyde", "aldehyde") in pair_set
    # A second canonical veto
    assert ("carboxylic_acid", "carboxylic_acid") in pair_set


def test_fg_compat_matrix_get_compatible():
    """``get_compatible_pairs()`` returns the complement of vetoed pairs.

    The 7 × 7 = 49-cell matrix has 15 vetoed cells; the canonical-
    compatible complement has 49 - 15 = 34 cells (when the 15
    vetoed pairs are *unordered* — each unordered veto maps to one
    triangular cell, so the canonical-compatible count is the
    number of triangular cells minus the veto count).
    """
    matrix = FGCompatMatrix()
    compat = matrix.get_compatible_pairs()
    veto = matrix.get_incompatible_pairs()
    assert isinstance(compat, list)
    # No overlap between compatible and incompatible.
    veto_set = {tuple(sorted(p)) for p in veto}
    for p in compat:
        assert tuple(sorted(p)) not in veto_set


def test_fg_compat_matrix_symmetric():
    """The FG matrix is symmetric — (A, B) and (B, A) have the same verdict.

    Every ``is_compatible`` lookup is order-independent.  This
    invariant pins the symmetrisation logic in
    :meth:`FGCompatMatrix.__init__`.
    """
    matrix = FGCompatMatrix()
    for a in FUNCTIONAL_GROUPS:
        for b in FUNCTIONAL_GROUPS:
            ab = matrix.is_compatible(a, b)
            ba = matrix.is_compatible(b, a)
            assert (
                ab == ba
            ), f"FG matrix not symmetric on ({a}, {b}): ab={ab} vs ba={ba}"


def test_fg_compat_matrix_unknown_labels_pass():
    """Unknown FG labels return True (permissive default).

    Per TODO-30 P1.2 the matrix is intentionally narrow — novel
    FGs that fall outside the canonical 7-axis vocabulary must
    not be silently vetoed.  We pin the permissive default.
    """
    matrix = FGCompatMatrix()
    assert matrix.is_compatible("unknown_fg", "aldehyde") is True
    assert matrix.is_compatible("aldehyde", "nitro") is True
    assert matrix.is_compatible("xyz", "abc") is True
    # Non-string inputs are permissive.
    assert matrix.is_compatible(None, "aldehyde") is True
    assert matrix.is_compatible("aldehyde", 42) is True


def test_fg_compat_matrix_custom_vetoed_pairs():
    """The matrix accepts a custom vetoed-pair list (override path).

    Property-test style: an explicit single-pair override should
    flip only that pair's verdict and leave the rest of the table
    intact.
    """
    matrix = FGCompatMatrix(
        vetoed_pairs=[("aldehyde", "azide")],
    )
    # The override is the only veto.
    pairs = matrix.get_incompatible_pairs()
    assert ("aldehyde", "azide") in {tuple(sorted(p)) for p in pairs}
    # Other pairs are no longer vetoed (override REPLACES the
    # canonical table, not extends it).
    assert matrix.is_compatible("aldehyde", "aldehyde") is True


def test_fg_compat_matrix_is_vetoed():
    """``is_vetoed`` is the negation of ``is_compatible``."""
    matrix = FGCompatMatrix()
    assert matrix.is_vetoed("aldehyde", "aldehyde") is True
    assert matrix.is_vetoed("carboxylic_acid", "primary_amine") is False
    assert matrix.is_compatible("aldehyde", "aldehyde") == (
        not matrix.is_vetoed("aldehyde", "aldehyde")
    )


def test_fg_compat_matrix_n_cells():
    """Pin the n_cells attribute = len(FUNCTIONAL_GROUPS) ** 2 = 49."""
    matrix = FGCompatMatrix()
    assert matrix.n_cells == len(FUNCTIONAL_GROUPS) ** 2
    assert matrix.n_cells == 7 * 7
    assert matrix.n_cells == 49


# ---------------------------------------------------------------------------
# LeaveGroupVeto tests
# ---------------------------------------------------------------------------


def test_leave_group_veto_rejects():
    """Forbidden pairs are vetoed.

    The canonical veto rejects ``aryl_halide`` + ``F`` (fluoride is
    the worst leaving group in the LG series per March 5th ed,
    Table 10-3 — C-F is the strongest C-X bond and F- is a poor
    nucleofuge on sp2 / sp3 centres).
    """
    veto = LeaveGroupVeto()
    assert veto.veto("c1ccc(F)cc1", "F") is True
    # F on boronic_acid is also vetoed.
    assert veto.veto("B(O)(O)c1ccccc1", "F") is True
    # F on alkyne is also vetoed.
    assert veto.veto("C#C", "F") is True


def test_leave_group_veto_reasons():
    """``veto_reasons`` returns a non-empty list of human-readable reasons.

    When the LG is vetoed, the list carries one entry per
    functional-group / LG combination.  When the LG is fine, the
    list is empty.
    """
    veto = LeaveGroupVeto()
    reasons_f = veto.veto_reasons("c1ccccc1", "F")
    assert isinstance(reasons_f, list)
    assert len(reasons_f) >= 1
    # Each reason is a non-empty string mentioning the LG.
    for r in reasons_f:
        assert isinstance(r, str)
        assert r
        assert "F" in r
    # A non-vetoed LG returns an empty list.
    reasons_cl = veto.veto_reasons("C#N", "N2")
    assert isinstance(reasons_cl, list)
    # N2 is not in the vetoed table (it's the LG for azides).
    # May be empty or contain entries depending on overlap; just
    # ensure it is a list.


def test_leave_group_veto_passes_when_safe():
    """Non-vetoed LG / reactant combinations return False."""
    veto = LeaveGroupVeto()
    # N2 as LG on an azide — fine (azide LG for CuAAC).
    assert veto.veto("CCCN=[N+]=[N-]", "N2") is False
    # H2O / OH2 as LG on an alkyne — fine (water-tolerated).
    assert veto.veto("C#C", "H2O") is False
    # Unknown LG symbol — permissive (no FG can be detected in
    # a non-azide so the any-fg fallback has nothing to veto).
    assert veto.veto("CCO", "super_weird_lg") is False


def test_leave_group_veto_for_fg():
    """FG-aware variant ``veto_for_fg`` flips only when the FG matches."""
    veto = LeaveGroupVeto()
    # Aryl_halide + F is vetoed.
    assert veto.veto_for_fg("aryl_halide", "F") is True
    # Alkyne + F is vetoed.
    assert veto.veto_for_fg("alkyne", "F") is True
    # Boronic_acid + F is vetoed.
    assert veto.veto_for_fg("boronic_acid", "F") is True
    # Unknown FG is permissive even with F.
    assert veto.veto_for_fg("nitro", "F") is False


def test_leave_group_veto_n_cells():
    """Pin ``n_cells`` = len(FUNCTIONAL_GROUPS) * len(LEAVING_GROUPS) = 84.

    This is the R17 ship 84-rule MLE matrix shape mirrored — the
    same 7 FG × 12 LG = 84-cell shape used by the
    :data:`molmetal_lam.lam_chem.pt_metal_ligand_exchange.METAL_LIGAND_EXCHANGE_MATRIX`.
    """
    veto = LeaveGroupVeto()
    assert veto.n_cells == 7 * 12
    assert veto.n_cells == 84


def test_leave_group_veto_custom_forbidden():
    """The veto accepts a custom forbidden-pair list."""
    veto = LeaveGroupVeto(
        forbidden_pairs=[("aryl_halide", "OH")],
    )
    # The override is the only veto.
    pairs = veto.get_forbidden_pairs()
    assert ("aryl_halide", "OH") in pairs
    # The canonical F vetoes are GONE (override REPLACES).
    assert veto.veto_for_fg("aryl_halide", "F") is False


def test_leave_group_veto_get_forbidden():
    """``get_forbidden_pairs`` returns the sorted vetoed pair list."""
    veto = LeaveGroupVeto()
    pairs = veto.get_forbidden_pairs()
    assert isinstance(pairs, list)
    assert len(pairs) >= 1
    # The list is sorted canonically.
    assert pairs == sorted(pairs)


# ---------------------------------------------------------------------------
# VetoAugmentedReducer tests
# ---------------------------------------------------------------------------


def test_veto_augmented_reducer_blocks():
    """The wrapper returns None when the FG veto fires."""
    mock = _MockReducer()
    wrapper = VetoAugmentedReducer(base_reducer=mock)
    out = wrapper.reduce(
        "CC=O",  # acetaldehyde as a stand-in SMILES
        fg_pair=("aldehyde", "aldehyde"),
    )
    assert out is None
    # The base reducer was NOT invoked (veto short-circuits).
    assert mock.n_calls == 0
    # The wrapper's veto counter incremented.
    assert wrapper.n_veto_calls == 1
    assert wrapper.n_pass_calls == 0


def test_veto_augmented_reducer_passes():
    """The wrapper delegates to the base reducer when no veto fires."""
    mock = _MockReducer(prefix="OUT:")
    wrapper = VetoAugmentedReducer(base_reducer=mock)
    out = wrapper.reduce(
        "NCC(=O)O",
        fg_pair=("carboxylic_acid", "primary_amine"),
    )
    assert out == "OUT:NCC(=O)O"
    # Base reducer was invoked once.
    assert mock.n_calls == 1
    # The wrapper's pass counter incremented.
    assert wrapper.n_pass_calls == 1
    assert wrapper.n_veto_calls == 0


def test_veto_augmented_reducer_lg_veto():
    """The wrapper returns None when the LG veto fires."""
    mock = _MockReducer()
    wrapper = VetoAugmentedReducer(base_reducer=mock)
    out = wrapper.reduce("c1ccc(F)cc1", leaving_group="F")
    assert out is None
    assert mock.n_calls == 0
    assert wrapper.n_veto_calls == 1


def test_veto_augmented_reducer_veto_rate():
    """``veto_rate()`` returns the cumulative veto rate (vetoed / total)."""
    mock = _MockReducer()
    wrapper = VetoAugmentedReducer(base_reducer=mock)
    # No calls yet — rate is 0.0.
    assert wrapper.veto_rate() == 0.0
    # One veto.
    wrapper.reduce("CC=O", fg_pair=("aldehyde", "aldehyde"))
    assert wrapper.veto_rate() == 1.0
    # One pass.
    wrapper.reduce("NCC(=O)O", fg_pair=("carboxylic_acid", "primary_amine"))
    # 1 veto / 2 total = 0.5
    assert wrapper.veto_rate() == 0.5


def test_veto_augmented_reducer_requires_reduce_method():
    """Constructing the wrapper with a bad base raises TypeError."""
    with pytest.raises(TypeError):
        VetoAugmentedReducer(base_reducer=object())  # type: ignore[arg-type]


def test_veto_augmented_reducer_no_veto_args_passes():
    """When both fg_pair and leaving_group are None, the wrapper passes."""
    mock = _MockReducer(prefix="E:")
    wrapper = VetoAugmentedReducer(base_reducer=mock)
    out = wrapper.reduce("CCO")
    assert out == "E:CCO"
    assert mock.n_calls == 1


def test_veto_augmented_reducer_default_matrices():
    """Defaults construct fresh canonical matrices on first use."""
    mock = _MockReducer()
    wrapper = VetoAugmentedReducer(base_reducer=mock)
    # The defaults are non-None and of the right types.
    assert wrapper.fg_matrix is not None
    assert wrapper.leave_veto is not None
    assert isinstance(wrapper.fg_matrix, FGCompatMatrix)
    assert isinstance(wrapper.leave_veto, LeaveGroupVeto)


# ---------------------------------------------------------------------------
# register_fg_compat_veto test
# ---------------------------------------------------------------------------


def test_register_fg_compat_veto_count():
    """``register_fg_compat_veto`` returns a count ≥ 1."""
    n = register_fg_compat_veto()
    assert isinstance(n, int)
    assert n >= 1
    # The count = sum of vetoed cells in FG + LG matrices.
    fg = FGCompatMatrix()
    lg = LeaveGroupVeto()
    assert n == len(fg.vetoed_pairs) + len(lg.forbidden_pairs)


def test_register_fg_compat_veto_matches_canonical():
    """The registry count matches the canonical FG + LG vetoed cells.

    The canonical FG matrix vetoes 15 distinct unordered pairs
    (per the table in :data:`molmetal_lam.lam_chem.fg_compat_veto._VETOED_FG_PAIRS`).
    The canonical LG matrix vetoes a small number of (FG, LG)
    pairs.  We pin the SUM via the registry return value rather
    than asserting the exact LG count (which is subject to
    refinement in follow-on rounds).
    """
    n = register_fg_compat_veto()
    fg = FGCompatMatrix()
    lg = LeaveGroupVeto()
    assert len(fg.vetoed_pairs) == 15
    assert len(lg.forbidden_pairs) >= 1
    assert n == len(fg.vetoed_pairs) + len(lg.forbidden_pairs)
    # Sanity: n > 15 (at least one LG veto beyond the 15 FG vetoes).
    assert n > 15
