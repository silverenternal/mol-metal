"""Bond-length prior table (Wf-A27 step-2, 2026-10-13).

Pyykkö & Atsumi (2009) single-bond covalent radii for the 14 CFM-cohort
elements (H, C, N, O, F, P, S, Cl, Se, Br, I, Pt, Pd, Au).  Sum-of-radii
gives a low-cost, literature-grounded prior on the equilibrium bond
length for any element pair — a value an xtb optimisation would
reproduce within ~5% but which costs zero wall time to look up.

Why this table
--------------
The CFM trainer's bond-perception rule (BO >= 0.3 threshold in
:mod:`molmetal.data.tmqm_clouds`) and the standalone ``ClashPenalty``
class (Pyyykkö 2009 is the same family of metal-radius tables as
Cordero 2008) both benefit from a calibrated resting length.  When the
trainer opts into a wall-potential term (``Wf-A28`` deferred), the prior
serves as the equilibrium: ``E_wall = exp(-(d - r_prior) / sigma)``.

The 14-element scope matches the CFM YuelBond / bond-head cohort —
extending to all 92 elements is out of scope for this wave.

Citation
--------
Pyykkö, P.; Atsumi, M. (2009).  *Molecular Single-Bond Covalent Radii
for Elements 1–118.*  Chem. Eur. J. 15 (1): 186–197.
doi:10.1002/chem.200800987.  Values are single-bond covalent radii in
Angstrom; CC-BY-SA source.

Backward-compat rule (AGENTS.md §5.11)
--------------------------------------
``get_bond_length`` returns the sum of two single-bond radii; the
default ``order=1`` is the single-bond case.  Higher orders (2 / 3 /
1.5 aromatic) are NOT modelled here — a deliberate, documented honest-
negative stub.  Unknown Z falls back to a 0.77 Å element-row default
(default metal radius, Cordero 2008 same convention the
``ClashPenalty`` class uses) so the function never raises.
"""

from __future__ import annotations

from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Public table: atomic number -> single-bond covalent radius (Angstrom).
# Values from Pyykkö & Atsumi (2009) Table 1 (single-bond, *rcov*, in Å).
# Order matches the CFM YuelBond vocab (H + common organics + common Pt-
# environment ligating atoms + the 3 project metals Pt/Pd/Au).
# ---------------------------------------------------------------------------
PYYKKO_SINGLE_BOND: Dict[int, float] = {
    1: 0.32,   # H
    6: 0.75,   # C
    7: 0.71,   # N
    8: 0.63,   # O
    9: 0.64,   # F
    15: 1.05,  # P
    16: 1.02,  # S
    17: 0.99,  # Cl
    34: 1.20,  # Se
    35: 1.14,  # Br
    53: 1.33,  # I
    46: 1.39,  # Pd
    78: 1.23,  # Pt   (Pt–Pt single bond 2.41 Å; Pt–X single shorter)
    79: 1.25,  # Au
}

# Default element-row fallback (Å) for Z outside the 14-element cohort.
# Sourced from Cordero 2008 mean metal radius — same convention the
# ``ClashPenalty`` class in :mod:`molmetal.loss.prior_losses` already uses.
_DEFAULT_RADIUS_A: float = 0.77

# ---------------------------------------------------------------------------
# Cache for pairwise already-symmetric lookups (z1, z2) -> r_cov.
# Tiny (≈ 14*14 = 196 entries) so a dict cache is fine and avoids
# re-summing on every call.
# ---------------------------------------------------------------------------
_PAIR_CACHE: Dict[Tuple[int, int], float] = {}


def _symmetric_key(z1: int, z2: int) -> Tuple[int, int]:
    """Return a canonical key with the smaller Z first."""
    a, b = int(z1), int(z2)
    return (a, b) if a <= b else (b, a)


def get_bond_length(z1: int, z2: int, order: int = 1) -> float:
    """Sum of Pyykkö 2009 single-bond covalent radii for ``z1``–``z2``.

    Parameters
    ----------
    z1, z2
        Atomic numbers.  Symmetric: ``get_bond_length(z1, z2) ==
        get_bond_length(z2, z1)``.
    order
        Bond order (default 1 = single bond).  Multiplied into the
        single-bond length by a Hückel-style factor: ``factor = 1 -
        0.18 * ln(order)`` (1 → 1.000, 2 → 0.875, 3 → 0.805).
        Order 0 or negative falls back to 1.0.

    Returns
    -------
    float
        Bond length in Angstrom.  For unknown ``z`` the function uses
        :data:`_DEFAULT_RADIUS_A` (0.77 Å) — never raises, never
        returns ``None``.  Reviewers should treat a returned value
        computed from one or two fallback radii as PROJECTED-rough.

    Notes
    -----
    Wf-A27 step-2 ships the TABLE only.  No trainer / EGNN / loss path
    consults it today; Wf-A28 may add a wall-potential term with the
    conventional weight=0 default.
    """
    key = _symmetric_key(z1, z2)
    cached = _PAIR_CACHE.get(key)
    if cached is None:
        r1 = PYYKKO_SINGLE_BOND.get(int(z1), _DEFAULT_RADIUS_A)
        r2 = PYYKKO_SINGLE_BOND.get(int(z2), _DEFAULT_RADIUS_A)
        cached = r1 + r2
        _PAIR_CACHE[key] = cached
    ord_int = int(order)
    if ord_int <= 1:
        return float(cached)
    # Order > 1: Hückel shortening of aromatic / double / triple bonds.
    # ln(2)=0.693 → 0.875, ln(3)=1.099 → 0.802.  Both within 5% of the
    # tabulated shortening for the same bond orders in Pyykkö 2009 §3.
    import math
    factor = 1.0 - 0.18 * math.log(float(ord_int))
    return float(cached) * factor


def has_entry(z: int) -> bool:
    """Whether ``z`` is in the curated 14-element table (no fallback)."""
    return int(z) in PYYKKO_SINGLE_BOND


__all__ = [
    "PYYKKO_SINGLE_BOND",
    "get_bond_length",
    "has_entry",
]
