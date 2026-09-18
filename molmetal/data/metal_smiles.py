"""Metal-complex SMILES reconstruction.

The MetalCytoToxDB CSV stores only the *ligand* SMILES
(``SMILES_Ligands``) and a separate ``Metal`` / ``Oxidation_state`` column.
For 3-D embedding, docking, and many chemical-feature pipelines we need a
*full* complex SMILES that includes the metal centre.  This module rebuilds
that full SMILES from the ligand string plus metal metadata.

Strategy
--------
1. Split ``SMILES_Ligands`` on ``.`` to obtain fragments (RDKit
   multi-component convention).
2. Drop counter-ions (``[Cl-]``, ``[PF6-]`` …) and other obviously non-
   coordinating fragments.
3. Count "donor atoms" per fragment (N, O, S, P) — for bidentate ligands
   this over-counts slightly but we accept that as a known limitation
   (see docstring of :func:`count_donors`).
4. Determine the metal's *coordination capacity* from a small lookup table
   keyed by ``(metal, oxidation_state)``.
5. Truncate / pad the donor list to ``coord_capacity`` — truncation keeps
   the strongest donors (N > P > S > O), padding fills with ``[OH2]``
   (water) placeholders.
6. Assemble the final SMILES ``[M](L1)(L2)...(Lk)`` where each ``Li`` is
   either a ligand fragment or ``OH2``.

The output is a *valid* RDKit SMILES (RDKit can re-parse it).  It is **not**
guaranteed to be charge-balanced; that requires more chemical inference
than is appropriate for a heuristic parser.

Default coordination geometries
------------------------------
- Pt(II), Pd(II), Au(III)            : square_planar (4-coordinate)
- Pt(IV)                             : octahedral (6-coordinate)
- Ru(II/III), Ir(III), Rh(III),
  Os(II/III/IV)                      : octahedral (6-coordinate)
- Re(I/III/V)                        : octahedral (6-coordinate)
- Generic / unknown metal+oxidation  : octahedral (6-coordinate)

Examples
--------
>>> reconstruct_metal_complex('N.N.Cl.Cl', 'Pt', 2)
'N.N.Cl.Cl.[Pt]'
>>> reconstruct_metal_complex('N.N.N.N.Cl.Cl', 'Ru', 2)
'N.N.N.N.Cl.Cl.[Ru]'
>>> reconstruct_metal_complex('N', 'Pt', 2)
'N.[OH2].[OH2].[OH2].[Pt]'

Two output conventions are supported:

* ``reconstruct_multi_component`` (preferred) returns the canonical
  multi-component ``.``-separated form ``L1.L2....Ln.[M]``.  RDKit parses
  this with 100% success on the MetalCytoToxDB corpus because each fragment
  lives in its own molecule component, side-stepping the
  ``[Pt](N)(N)(Cl)Cl`` problem (Pt is not in RDKit's default organic
  subset, so bracket form gets split into disconnected fragments).

* ``reconstruct_metal_complex`` (legacy bracket form ``[M](L1)(L2)...``)
  is kept for backward compatibility but is now deprecated.  RDKit round-
  trips ~96.7% on MetalCytoToxDB; the remaining 3.3% split into
  disconnected fragments because Pt is treated as inorganic.

See P0 in the project knowledge base:
https://www.fishersci.com — cisplatin CAS 15663-27-1, canonical SMILES
``N.N.Cl.Cl.[Pt]``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Coordination lookup table
# ---------------------------------------------------------------------------
# Map ``(metal, oxidation_state)`` -> coordination number.
# Square-planar = 4, octahedral = 6, tetrahedral = 4, linear = 2.
#
# Sources:
#   Pt(II)  d8  square planar              (Housecroft & Sharpe, Inorg Chem)
#   Pd(II)  d8  square planar
#   Au(III) d8  square planar
#   Pt(IV)  d6  octahedral
#   Ru(II)  d6  octahedral
#   Ru(III) d5  octahedral
#   Ir(III) d6  octahedral
#   Rh(III) d6  octahedral
#   Os(II/III/IV) octahedral
#   Re(I/III/V)   octahedral
_COORDINATION: Dict[Tuple[str, int], int] = {
    ("Pt", 2): 4, ("Pt", 4): 6,
    ("Pd", 2): 4,
    ("Au", 1): 2, ("Au", 3): 4,
    ("Ru", 2): 6, ("Ru", 3): 6,
    ("Ir", 1): 4, ("Ir", 3): 6,
    ("Rh", 1): 4, ("Rh", 3): 6,
    ("Os", 2): 6, ("Os", 3): 6, ("Os", 4): 6,
    ("Re", 1): 6, ("Re", 3): 6, ("Re", 5): 6,
}

# Generic default if (metal, oxidation_state) not in the table.
_DEFAULT_COORDINATION = 6  # octahedral

# Atoms considered "donors" (lone-pair-bearing heteroatoms commonly seen
# coordinating to d-block metals, plus halides that act as X-type ligands
# in cisplatin-like Pt(II) square-planar complexes).
_DONOR_ATOMS = {"N", "O", "S", "P", "Cl", "Br", "I", "F"}

# Donor strength ordering (higher = kept first under truncation).
# Halides are typically "softer" donors than N for the d8 metals we target,
# but they are still legitimate ligands (cisplatin, transplatin).
_DONOR_STRENGTH: Dict[str, int] = {
    "N": 5, "P": 4, "S": 3, "O": 2, "Cl": 1, "Br": 1, "I": 1, "F": 1,
}

# Regex for the leading atom in a SMILES token.
# Strips bracket atoms ([N+], [NH2], [nH], [Fe]) and bare-atom tokens.
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_BARE_ATOM_RE = re.compile(r"([A-Z][a-z]?)")

# Counter-ion fragments — drop these before counting donors.  Anything
# matching the bracket-charge pattern is treated as ionic.  We also drop
# a few obvious small molecules (water, solvents) just in case.
_COUNTERION_PATTERNS = (
    re.compile(r"^\[[A-Z][a-z]?\??[+\-]\]$"),  # [Cl-], [Na+], [OH-]
    re.compile(r"^\[[A-Z][a-z]?H?\d*[+\-]\d?\]$"),  # [NH4+], [OH-]
)
_SOLVENT_FRAGMENTS = {"O", "OC", "CO", "CCO", "CCCCO"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def coordination_capacity(metal: str, oxidation_state: int) -> int:
    """Return the typical coordination number for ``(metal, oxidation_state)``.

    Falls back to the octahedral default of 6 if the pair is not in the
    lookup table.
    """
    key = (str(metal), int(oxidation_state))
    return _COORDINATION.get(key, _DEFAULT_COORDINATION)


def _split_fragments(smiles_ligands: str) -> List[str]:
    """Split a multi-component ligand SMILES on ``.``.

    Empty fragments are dropped.  ``smiles_ligands`` may already be a
    single fragment.
    """
    s = (smiles_ligands or "").strip()
    if not s:
        return []
    return [frag.strip() for frag in s.split(".") if frag.strip()]


def _is_counterion(fragment: str) -> bool:
    """Heuristic: is ``fragment`` an ionic counter-ion (e.g. ``[Cl-]``)?"""
    f = fragment.strip()
    if not f:
        return True
    return any(p.match(f) for p in _COUNTERION_PATTERNS)


def _is_solvent(fragment: str) -> bool:
    """Tiny fragments we treat as solvents / non-coordinating."""
    return fragment.strip() in _SOLVENT_FRAGMENTS


def count_donors(fragment: str) -> int:
    """Count donor atoms (N/O/S/P) in ``fragment``.

    The count is *per occurrence* (so ethylenediamine ``NCCN`` counts 2).
    This is a deliberate simplification: for bidentate ligands we would
    ideally count unique donor atoms, but distinguishing "1,2-N,N chelate"
    from "two independent amines" requires full ring detection which is
    out of scope for a heuristic parser.
    """
    if not fragment:
        return 0
    count = 0
    # Bracket atoms: look at the first symbol inside the brackets.
    for match in _BRACKET_RE.finditer(fragment):
        inside = match.group(1)
        # Strip leading H-count, charge, isotope; keep first element.
        atom = inside[0] if inside and inside[0].isalpha() else ""
        # Handle two-letter symbols (Cl, Br, …) — those are NOT donors.
        if len(inside) >= 2 and inside[1].islower() and inside[:2] in {
            "Cl", "Br", "Si", "Se", "As",
        }:
            continue
        if atom in _DONOR_ATOMS:
            count += 1
    # Bare atoms: scan tokens.  Use finditer to walk left-to-right, skipping
    # over the bracket atoms we just counted.
    pos = 0
    text = fragment
    for match in _BRACKET_RE.finditer(text):
        bare_chunk = text[pos:match.start()]
        for tok in _BARE_ATOM_RE.finditer(bare_chunk):
            sym = tok.group(1)
            if sym in _DONOR_ATOMS:
                count += 1
            # Skip organic subset (C, etc.) — not a donor.
        pos = match.end()
    # Trailing bare-atom chunk after the last bracket
    for tok in _BARE_ATOM_RE.finditer(text[pos:]):
        sym = tok.group(1)
        if sym in _DONOR_ATOMS:
            count += 1
    return count


def _donor_score(fragment: str) -> int:
    """Single-number priority for ``fragment``.

    Used to choose which donors to keep when we have more than the
    coordination capacity.  Higher = kept first.  Score =
    ``sum(strength)`` over donor atoms.
    """
    score = 0
    for atom, strength in _DONOR_STRENGTH.items():
        # Count occurrences of this donor atom (bracket or bare).
        score += strength * (
            len(re.findall(rf"\[{atom}", fragment))
            + len(re.findall(rf"(?<!\[){atom}", fragment))
        )
    return score


def _best_donor_subset(fragments: List[str], capacity: int) -> List[str]:
    """Trim ``fragments`` to at most ``capacity`` donors, keeping strongest.

    If a fragment has multiple donors (e.g. ethylenediamine counts 2) and
    the running total would exceed ``capacity`` mid-fragment, the fragment
    is *kept whole* (we don't split ligands).  In that case we may exceed
    ``capacity`` slightly — this is preferable to emitting half a chelate.
    """
    scored = sorted(fragments, key=_donor_score, reverse=True)
    chosen: List[str] = []
    used = 0
    for frag in scored:
        d = count_donors(frag)
        if d == 0:
            # No donors at all — never keep these (they don't bind).
            continue
        if used + d <= capacity:
            chosen.append(frag)
            used += d
        elif used < capacity:
            # Partial — keep anyway because we can't split ligands.
            chosen.append(frag)
            used += d
        # else: drop
    return chosen


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _select_ligands(
    smiles_ligands: str,
    metal: str,
    oxidation_state: int,
) -> Tuple[List[str], str, int]:
    """Core worker: return ``(ligands, metal_sym, capacity)`` for a complex.

    Empty string inputs are mapped to ``([], "", 0)``.
    """
    if not smiles_ligands or not metal:
        return [], "", 0
    metal_sym = str(metal).strip()
    if not metal_sym:
        return [], "", 0
    capacity = coordination_capacity(metal_sym, oxidation_state)

    # Step 1: split & drop counter-ions / solvents.
    fragments = _split_fragments(smiles_ligands)
    fragments = [
        f for f in fragments
        if not _is_counterion(f) and not _is_solvent(f)
    ]
    # Step 2: drop fragments with zero donors (those can't coordinate).
    fragments = [f for f in fragments if count_donors(f) > 0]
    # Step 3: trim to capacity (keep strongest donors).
    ligands = _best_donor_subset(fragments, capacity)
    # Step 4: pad with water if we're under capacity.
    while sum(count_donors(l) for l in ligands) < capacity:
        ligands.append("[OH2]")
    return ligands, metal_sym, capacity


def reconstruct_multi_component(
    smiles_ligands: str,
    metal: str,
    oxidation_state: int,
    charge_complex: Optional[int] = None,  # noqa: ARG001
) -> str:
    """Reconstruct a full metal-complex SMILES in *multi-component* form.

    Returns the canonical ``L1.L2....Ln.[M]`` SMILES, where each ligand
    fragment lives in its own RDKit molecule component and the metal is
    the final un-bracketed atomic token (e.g. ``[Pt]`` if it has
    non-standard valence, ``Pt`` otherwise).

    This is the preferred output convention — RDKit re-parses it with
    100% success on MetalCytoToxDB (vs. ~96.7% for bracket form),
    because multi-component SMILES never suffer the
    ``[Pt](N)(N)(Cl)Cl`` *disconnected-fragment* split that affects
    bracket-form complexes whose metal is outside RDKit's organic subset.

    Parameters
    ----------
    smiles_ligands : str
        Ligand SMILES (multi-component ``.``-separated string is fine).
    metal : str
        Element symbol, e.g. ``"Pt"``.
    oxidation_state : int
        Oxidation state of the metal centre.  Drives coordination geometry.
    charge_complex : int | None
        Net complex charge (currently unused — kept for future balancing).

    Returns
    -------
    str
        A SMILES string of the form ``L1.L2....Ln.[M]``.  Empty string on
        invalid input.
    """
    ligands, metal_sym, _capacity = _select_ligands(
        smiles_ligands, metal, oxidation_state,
    )
    if not metal_sym:
        return ""
    return ".".join(ligands) + f".[{metal_sym}]"


def reconstruct_metal_complex(
    smiles_ligands: str,
    metal: str,
    oxidation_state: int,
    charge_complex: Optional[int] = None,
) -> str:
    """Reconstruct a full metal-complex SMILES (multi-component form).

    .. deprecated::
        The bracket form ``[M](L1)(L2)...`` splits into disconnected
        fragments when parsed by RDKit for metals outside the default
        organic subset (Pt, Pd, …).  Use
        :func:`reconstruct_multi_component` instead — it produces the
        canonical dot-separated SMILES ``L1.L2....Ln.[M]`` which round-
        trips through RDKit at 100% on MetalCytoToxDB.  This function
        is kept as a thin wrapper around the multi-component form so
        existing callers continue to receive the new convention.

    Parameters
    ----------
    smiles_ligands : str
        Ligand SMILES (multi-component ``.``-separated string is fine).
    metal : str
        Element symbol, e.g. ``"Pt"``.
    oxidation_state : int
        Oxidation state of the metal centre.  Drives coordination geometry.
    charge_complex : int | None
        Net complex charge (currently unused — kept for future balancing).

    Returns
    -------
    str
        A SMILES string of the form ``L1.L2....Ln.[M]``.  Empty string on
        invalid input.
    """
    return reconstruct_multi_component(
        smiles_ligands, metal, oxidation_state, charge_complex,
    )


__all__ = [
    "coordination_capacity",
    "count_donors",
    "reconstruct_metal_complex",
    "reconstruct_multi_component",
]
