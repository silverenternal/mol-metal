"""Property-based well-formedness checks for ``Tile`` and tile libraries.

This module defines the invariants that every :class:`~molmetal_lam.tile_lib.tile.Tile`
in a library must satisfy.  The invariants are designed to be cheap
(no RDKit embedding or conformer generation) so they can run at every
library build and at CI time without blowing the test budget.

The two public functions are:

* :func:`assert_tile_well_formed` — check a single :class:`Tile`.
* :func:`assert_library_well_formed` — check a whole library list.

Both functions use a *soft* failure policy:  they **log a warning**
and return ``False`` for the failing tile rather than raising.  This
is intentional — the Phase-1 fragment library is curated from real
molecules and is allowed to contain edge cases (degenerate rings,
unusual tautomers) that some pipelines may want to keep while others
filter out.  A property-based test that raises on every edge case
would block iterative curation.

The CI test suite (see ``molmetal_lam/tests/test_tile_properties.py``)
asserts stricter invariants on top of these helpers (MW <= 500,
uniqueness, library size in range).

Allowed atoms
-------------
The :data:`ALLOWED_ATOMS` set is the periodic-table subset that the
Phase-1 fragment library uses by design.  Metals are intentionally
**not** in the set — the tile library is the small-molecule organic
side of the pipeline; metal-coordination chemistry lives one layer up
in the :mod:`molmetal_lam.atoms` combinator system.

Adding a new element to a hardcoded SMILES pool requires either
extending :data:`ALLOWED_ATOMS` or curating the SMILES down to the
existing set.  Property tests will fail loudly if either is missed.
"""

from __future__ import annotations

import logging
from typing import List

from molmetal_lam.tile_lib.tile import Tile

__all__ = [
    "ALLOWED_ATOMS",
    "assert_tile_well_formed",
    "assert_library_well_formed",
]


# Module-level logger — callers can configure this to silence warnings
# during exploratory MCTS runs.
_LOG = logging.getLogger(__name__)


# Periodic-table subset admitted by the Phase-1 SMARTS-diverse pool.
# These are the elements that appear in ChEMBL reactive-handle fragments
# and the ZINC click-chemistry subset (Sterling & Irwin, 2015).  Metals
# (Pt, Ru, Zn, Ir, Cu, Au, ...) are excluded by design — they live one
# layer up in the metal-coordination combinators, not the constant pool.
# R10 axis A — ``B`` (boron) is admitted so the Suzuki coupling handles
# (aryl-boronic acids R-B(OH)2) pass the well-formedness check.
ALLOWED_ATOMS: frozenset[str] = frozenset(
    {
        "C",   # carbon
        "N",   # nitrogen
        "O",   # oxygen
        "F",   # fluorine
        "S",   # sulfur
        "Cl",  # chlorine
        "Br",  # bromine
        "I",   # iodine
        "P",   # phosphorus
        "B",   # boron (R10 axis A — Suzuki coupling handles)
    }
)


# Hard upper bound for the drug-likeness MW filter (Lipinski's Rule of 5
# caps at 500 Da; we use the same ceiling for tiles, even though tiles
# are typically much smaller — this catches accidental junk in
# hardcoded SMILES pools).
MAX_TILE_MW: float = 500.0


def _tile_atom_set(smiles: str) -> set[str]:
    """Return the set of element symbols present in ``smiles``.

    Uses RDKit's atom iterator (lazy-imported) so we do not pay the
    RDKit import cost unless a property test is actually run.
    """
    from rdkit import Chem  # type: ignore[import-not-found]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    return {atom.GetSymbol() for atom in mol.GetAtoms()}


def assert_tile_well_formed(tile: Tile) -> bool:
    """Return True if ``tile`` satisfies all well-formedness invariants.

    The invariants are intentionally *permissive* (no embedding
    required, no conformer check) so this can run in tight CI loops
    and at every :func:`FRAGMENT_LIBRARY_200_TILES` call.

    Parameters
    ----------
    tile : Tile
        The tile to validate.  Must be a :class:`Tile` instance.

    Returns
    -------
    bool
        ``True`` if every check passes, ``False`` otherwise.  A
        warning is logged with the failing reason on every miss so
        CI logs are self-explanatory.

    Notes
    -----
    Failures are *logged* and not raised so the helper is safe to call
    inside a long build pipeline that may encounter edge cases.  The
    stricter property tests in
    ``molmetal_lam/tests/test_tile_properties.py`` are what gate
    CI quality.
    """
    # 1. SMILES parses via RDKit.
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - environment guard
        _LOG.warning(
            "Tile %s: RDKit unavailable (%s); skipping SMILES parse check",
            tile.tile_id,
            e,
        )
        return False

    mol = Chem.MolFromSmiles(tile.smiles)
    if mol is None:
        _LOG.warning(
            "Tile %s: SMILES failed RDKit parse: %r",
            tile.tile_id,
            tile.smiles,
        )
        return False

    # 2. MW <= 500 (drug-likeness ceiling).
    if not (0.0 <= tile.mw <= MAX_TILE_MW):
        _LOG.warning(
            "Tile %s: MW %.2f outside [0, %.0f] (smiles=%r)",
            tile.tile_id,
            tile.mw,
            MAX_TILE_MW,
            tile.smiles,
        )
        return False

    # 3. Atom whitelist — only allowed organic elements.
    atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
    forbidden = atoms - ALLOWED_ATOMS
    if forbidden:
        _LOG.warning(
            "Tile %s: contains forbidden atoms %s (smiles=%r); allowed=%s",
            tile.tile_id,
            sorted(forbidden),
            tile.smiles,
            sorted(ALLOWED_ATOMS),
        )
        return False

    return True


def assert_library_well_formed(tiles: List[Tile]) -> List[Tile]:
    """Apply :func:`assert_tile_well_formed` to every tile in ``tiles``.

    Returns the **list of well-formed tiles** (i.e. the input minus
    any failing tiles).  Failing tiles are not raised — they are
    logged at WARNING level and silently filtered out of the return
    value, so callers can keep the validated subset without aborting
    the build.

    Parameters
    ----------
    tiles : List[Tile]
        Input library (may contain failing tiles).

    Returns
    -------
    List[Tile]
        The well-formed subset of ``tiles`` in the original order.
        May be shorter than the input; an empty result is possible
        but unusual.

    Examples
    --------
    >>> from molmetal_lam.tile_lib.library import FRAGMENT_LIBRARY_200_TILES
    >>> tiles = FRAGMENT_LIBRARY_200_TILES()
    >>> validated = assert_library_well_formed(tiles)
    >>> assert len(validated) >= 150
    """
    if not tiles:
        return []

    passed: List[Tile] = []
    n_failed = 0
    for tile in tiles:
        if assert_tile_well_formed(tile):
            passed.append(tile)
        else:
            n_failed += 1

    if n_failed:
        _LOG.warning(
            "assert_library_well_formed: %d / %d tiles failed (returning %d)",
            n_failed,
            len(tiles),
            len(passed),
        )
    return passed
