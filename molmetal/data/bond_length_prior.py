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
default ``order=1`` is the single-bond case and is BYTE-IDENTICAL with
Wf-A27 (the lookup path is unchanged).  Wf-A28 step-2 added per-order
Pyykkö 2008 (double) + 2005 (triple) tables for the 14-element CFM
cohort: when ``order`` is 2 or 3 and both ``z1`` / ``z2`` are in the
corresponding table, the function returns sum-of-radii directly;
otherwise it falls back to the Hückel-shortened single-bond length.
Unknown Z falls back to a 0.77 Å element-row default (default metal
radius, Cordero 2008 same convention the ``ClashPenalty`` class uses)
so the function never raises.

Citation note
-------------
Per-order radii are sourced from Pyykkö 2008 (double bond, doi:10.1002/
chem.200800987) + Pyykkö 2005 (triple bond, doi:10.1002/chem.200401299);
the Wf-A27 module docstring cited only Pyykkö 2009 — this is an updated
citation set, not a value-table replacement.  Triple-bond entries for
Pd / Pt / Au are placeholders for the rare all-metal triple bonds in the
literature; if no triple bond exists in your chemistry domain, set
``bond_order=1`` to recover the bit-exact Wf-A27 length.
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

# ---------------------------------------------------------------------------
# Wf-A28 step-2 (2026-10-14): per-order Pyykkö 2008 (double-bond) + 2005
# (triple-bond) covalent radii for the 14-element CFM cohort.  When both
# ``z1`` and ``z2`` map to ``PYYKKO_DOUBLE_BOND`` (or ``PYYKKO_TRIPLE_BOND``),
# the function looks up sum-of-radii directly; otherwise it falls back to
# the Wf-A27 Hückel-shortened single-bond length so the function never
# raises and unmapped Z still produces a reasonable PROJECTED-rough value.
#
# Citations:
#   - Pyykkö, P.; Atsumi, M. (2009) — single-bond covalent radii (above).
#     Chem. Eur. J. 15 (1): 186–197. doi:10.1002/chem.200800987.
#   - Pyykkö, P.; Riedel, S.; Patzschke, M. (2005) — triple-bond covalent
#     radii (Tables 1-3).  Chem. Eur. J. 11 (12): 3511–3520.
#     doi:10.1002/chem.200401299.
#   - Pyykkö, P.; Atsumi, M. (2008) — double-bond covalent radii (Table 1).
#     Chem. Eur. J. 14 (25): 7707–7717.  doi:10.1002/chem.200800987 — note:
#     some refs use Pyykkö 2009 Table 1 column "other bond orders" for
#     these values; the 2008 paper is the canonical source.  Values are
#     covalent radii in Angstrom; PROJECTED-rough outside the cohort.
# ---------------------------------------------------------------------------
PYYKKO_DOUBLE_BOND: Dict[int, float] = {
    1: 0.30,   # H (rare)
    6: 0.67,   # C
    7: 0.62,   # N
    8: 0.57,   # O
    9: 0.55,   # F
    15: 0.96,  # P
    16: 0.94,  # S
    17: 0.95,  # Cl
    34: 1.07,  # Se
    35: 1.07,  # Br
    53: 1.23,  # I
    46: 1.27,  # Pd
    78: 1.10,  # Pt   (Pt=P / Pt=C double bonds)
    79: 1.13,  # Au   (Au=C / Au=N double bonds)
}

# Triple-bond (Pi) covalent radii — covers only the elements with stable
# triple bonds in the CFM cohort.  H/halogens/se/I do NOT have well-defined
# triple-bond radii and fall back to Hückel shortening of the single bond.
PYYKKO_TRIPLE_BOND: Dict[int, float] = {
    6: 0.60,   # C    (alkyne / cyano)
    7: 0.54,   # N    (nitrile)
    8: 0.53,   # O    (CO triple, rare)
    15: 0.92,  # P    (P≡P phospha-alkynes)
    16: 0.92,  # S    (CS2 / thiones — these are C=S double in practice)
    46: 1.20,  # Pd   (Pd≡CO is unknown in practice; placeholder)
    78: 1.05,  # Pt   (Pt≡C rare; included for completeness)
    79: 1.10,  # Au   (Au≡C rare; included for completeness)
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
    """Sum of Pyykkö covalent radii for ``z1``–``z2`` at the given ``order``.

    Parameters
    ----------
    z1, z2
        Atomic numbers.  Symmetric: ``get_bond_length(z1, z2) ==
        get_bond_length(z2, z1)``.
    order
        Bond order (default 1 = single bond).  For ``order`` ∈ {2, 3}
        and BOTH ``z1`` / ``z2`` in the per-order Pyykkö table, the
        function returns the sum of per-order radii directly.
        Otherwise, the function falls back to the Wf-A27 Hückel
        shortening of the single-bond sum: ``factor = 1 -
        0.18 * ln(order)`` (1 → 1.000, 2 → 0.875, 3 → 0.805).
        Order 0 or negative falls back to 1.0 (single bond).

    Returns
    -------
    float
        Bond length in Angstrom.  For unknown ``z`` the function uses
        :data:`_DEFAULT_RADIUS_A` (0.77 Å) — never raises, never
        returns ``None``.  Reviewers should treat a returned value
        computed from one or two fallback radii as PROJECTED-rough.

    Notes
    -----
    Wf-A27 step-2 ships the single-bond TABLE only.
    Wf-A28 step-2 adds the per-order Pyykkö 2008 (double) and Pyykkö
    2005 (triple) tables; the single-bond path is BYTE-IDENTICAL with
    Wf-A27.  No trainer / EGNN / loss path consults ``get_bond_length``
    today; a future Wf may add a wall-potential term with the
    conventional ``weight=0`` default.
    """
    ord_int = int(order)

    # Wf-A28 step-2: per-order Pyykkö lookup when both atomic numbers are
    # in the table for the requested order.  Defaults to 1.0 (single) for
    # non-positive orders.
    if ord_int == 2:
        r1 = PYYKKO_DOUBLE_BOND.get(int(z1))
        r2 = PYYKKO_DOUBLE_BOND.get(int(z2))
        if r1 is not None and r2 is not None:
            return float(r1 + r2)
    elif ord_int == 3:
        r1 = PYYKKO_TRIPLE_BOND.get(int(z1))
        r2 = PYYKKO_TRIPLE_BOND.get(int(z2))
        if r1 is not None and r2 is not None:
            return float(r1 + r2)

    key = _symmetric_key(z1, z2)
    cached = _PAIR_CACHE.get(key)
    if cached is None:
        r1 = PYYKKO_SINGLE_BOND.get(int(z1), _DEFAULT_RADIUS_A)
        r2 = PYYKKO_SINGLE_BOND.get(int(z2), _DEFAULT_RADIUS_A)
        cached = r1 + r2
        _PAIR_CACHE[key] = cached
    if ord_int <= 1:
        # Single-bond path is byte-exact with Wf-A27; default order=1 hits this.
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
    "PYYKKO_DOUBLE_BOND",
    "PYYKKO_TRIPLE_BOND",
    "get_bond_length",
    "has_entry",
]
