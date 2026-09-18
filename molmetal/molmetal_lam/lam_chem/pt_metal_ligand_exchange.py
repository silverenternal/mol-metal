"""Pt(II) metal-ligand exchange SMARTS library (WF-Lambda-Boost Phase 2 + 3).

Background
----------
WF-Lambda-Fix-FullPath-v2 (2026-09-15) shipped the F2(a) base rules:

    * :class:`MetalLigandExchange` — single SMARTS pattern
      ``[Pt:1]([Cl:2])[*:3].[NH3:4]>>[Pt:1]([NH3:4])[*:3].[Cl:2]``
      (cisplatin activation step: Pt-Cl → Pt-NH3).

    * :class:`AquaExchange` — single SMARTS pattern
      ``[Pt:1]([Cl:2])[*:3].[OH2:4]>>[Pt:1]([OH:4])[*:3].[Cl:2]``
      (Reedijk 1987 aquation kinetics).

Both rules had ONE pattern each.  In production the MCTS can only fire
the rule on a *single* substrate shape — Pt-Cl + NH3 (or H2O).  This
module ships **5 + 2 = 7 SMARTS patterns** that cover the canonical
Pt(II) coordination chemistry, so the MCTS has multiple redex shapes
to explore.

Honest framing
--------------
* The SMARTS patterns are hand-curated from standard Pt pharmacology
  (Lippard 1995 + Reedijk 1987 + Comba-Hambley 2009).  No ML fitting.
* Some patterns (Pd / Au) extend the chemistry beyond Pt — these are
  the d8 analogues that share the strict square-planar coordination
  geometry per the WF-Lambda-Fix-FullPath-v2 F2(a) module note.
* The AquaExchange ``pKa1 ~ 6.5`` context is propagated via the
  ``AquaContext`` dataclass (carries the protonation state of the
  leaving group).  The ``pka1`` field is documentation only — the
  search layer reads it for logging, not for branch decisions.
* RDKit cannot sanitise most Pt-containing products (Pt_II not in
  default valence table).  The patterns still register as fired via
  the ``_L4_COUNTERS`` instrumentation so the search layer can
  record the chemistry happened.

Lit anchors
-----------
* Lippard 1995 — Pt(II) coordination chemistry textbook (d8
  square-planar, associative substitution pathway, 5-coordinate
  trigonal-bipyramidal transition state).
* Reedijk 1987 — Pt(II) aquation kinetics (Pt-Cl + H2O → Pt-OH2 +
  Cl, pKa1 ~ 6.5 for the first aquation of cisplatin; t_1/2 ~ 2 h
  at 37 °C in 4 mM Cl-).
* Comba-Hambley 2009 — d-block coordination modeling (extension of
  the Pt(II) framework to Pd(II) and Au(III)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# MetalLigandExchange SMARTS library — 5 patterns.
# ---------------------------------------------------------------------------
# Each entry is a (pattern_name, smarts, description) tuple.
# The SMARTS use RDKit atom-maps (``:N``) so the product keeps the
# Pt-N bond and releases the leaving group (Cl-, Br-, H2O) as a
# separate term.
# ---------------------------------------------------------------------------
METAL_LIGAND_EXCHANGE_SMARTS: List[Tuple[str, str, str]] = [
    # Pattern 1 — cisplatin canonical (already in beta_reductions.py
    # line 1184-1185).  Pt-Cl + NH3 → Pt-NH3 + Cl.  This is the
    # historical default.
    (
        "Pt_Cl_NH3",
        "[Pt:1]([Cl:2])[*:3].[NH3:4]>>[Pt:1]([NH3:4])[*:3].[Cl:2]",
        "cisplatin canonical — Pt-Cl + NH3 (Lippard 1995 §4.2)",
    ),
    # Pattern 2 — Pt-Cl + primary amine (R-NH2).  Same reaction
    # mechanism but with a substituted amine donor.  This matches
    # aminated ligands (e.g. ethylenediamine, amino acids).
    (
        "Pt_Cl_RNH2",
        "[Pt:1]([Cl:2])[*:3].[NH2:4]>>[Pt:1]([NH2:4])[*:3].[Cl:2]",
        "Pt-Cl + primary amine R-NH2 → Pt-NHR + Cl (Lippard 1995 §4.3)",
    ),
    # Pattern 3 — Pt-Br + NH3 (the bromide-leaving analogue; some
    # trans-Pt candidates use Pt-Br in place of Pt-Cl).
    (
        "Pt_Br_NH3",
        "[Pt:1]([Br:2])[*:3].[NH3:4]>>[Pt:1]([NH3:4])[*:3].[Br:2]",
        "Pt-Br + NH3 → Pt-NH3 + Br (trans-Pt pharmacology, Reedijk 1987)",
    ),
    # Pattern 4 — Pd(II)-Cl + NH3.  Pd is d8 like Pt but ~10^3 more
    # labile.  Same square-planar coordination.  Comba-Hambley 2009
    # treat Pd-Cl substitution as the same SMARTS shape with a
    # different metal atom symbol.
    (
        "Pd_Cl_NH3",
        "[Pd:1]([Cl:2])[*:3].[NH3:4]>>[Pd:1]([NH3:4])[*:3].[Cl:2]",
        "Pd-Cl + NH3 → Pd-NH3 + Cl (Comba-Hambley 2009 §3.4, d8 analogue)",
    ),
    # Pattern 5 — Au(III)-Cl + NH3.  Au_III is d8 (5d^8), square-
    # planar coordination.  Same SMARTS shape.  The Au_III arity
    # bug was caught by the closure-theorem property test (WF-Lambda-4).
    (
        "Au_Cl_NH3",
        "[Au:1]([Cl:2])[*:3].[NH3:4]>>[Au:1]([NH3:4])[*:3].[Cl:2]",
        "Au_III-Cl + NH3 → Au_III-NH3 + Cl (d8 5d analogue, Lippard 1995)",
    ),
]


# ---------------------------------------------------------------------------
# AquaExchange SMARTS library — 2 patterns.
# ---------------------------------------------------------------------------
# The historical default in beta_reductions.py is the first pattern.
# Pattern 2 covers the second aquation step (Reedijk 1987 aquation
# kinetics: Pt-Cl2 → Pt-Cl(OH2) → Pt-(OH2)2).
# ---------------------------------------------------------------------------
AQUA_EXCHANGE_SMARTS: List[Tuple[str, str, str]] = [
    # Pattern 1 — first aquation (cisplatin canonical).  Pt-Cl +
    # H2O → Pt-OH + HCl.  The water has lost one proton (now an
    # ``[OH]`` hydroxyl, charge 0 because Pt_II is divalent — the
    # positive charge that forms on the Pt-OH2 complex is reported
    # in the product term's Atom.charge field, not the SMILES).
    (
        "Pt_Cl_H2O_first",
        "[Pt:1]([Cl:2])[*:3].[OH2:4]>>[Pt:1]([OH:4])[*:3].[Cl:2]",
        "first aquation — Pt-Cl + H2O → Pt-OH + HCl "
        "(Reedijk 1987, pKa1 ~ 6.5, t_1/2 ~ 2 h at 37 °C in 4 mM Cl-)",
    ),
    # Pattern 2 — second aquation (the diaqua complex).  After the
    # first Cl has left, the second Cl is even more labile (trans
    # effect from the H2O).  This is the species that binds DNA-N7-
    # guanine (the actual cytotoxic event).  SMARTS shape: any Pt
    # bearing a hydroxyl + a Cl leaving group + H2O donor.
    (
        "Pt_OHCl_H2O_second",
        "[Pt:1]([OH:2])([Cl:3])[*:4].[OH2:5]>>"
        "[Pt:1]([OH:2])([OH:5])[*:4].[Cl:3]",
        "second aquation — Pt-OH(Cl) + H2O → Pt-(OH)2 + Cl "
        "(Reedijk 1987, the diaqua complex that binds DNA-N7-guanine)",
    ),
]


# ---------------------------------------------------------------------------
# AquaContext — pKa context propagation.
# ---------------------------------------------------------------------------
# AquaExchange has a *physico-chemical context* (pKa, ionic strength,
# temperature) that the search layer can read for logging / metrics.
# This is the AquaExchange analogue of the ``AquaContext`` docstring
# field on the rule.  The dataclass is *documentation-only* — the
# search layer does not branch on it.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AquaContext:
    """Aquation kinetic context for AquaExchange SMARTS.

    Attributes
    ----------
    pka1 : float
        First aquation pKa.  For cisplatin this is ~6.5 (Reedijk
        1987).  For transplatin it's ~4.5 (faster aquation).
    ionic_strength_M : float
        Ionic strength of the medium (NaCl concentration
        approximation).  Higher Cl- shifts the equilibrium back
        toward Pt-Cl (the *reverse* of aquation).
    temperature_K : float
        Reaction temperature.  310 K (37 °C) is the physiological
        default.
    notes : str
        Free-form notes (e.g. "first aquation", "second aquation",
        "diaqua complex").

    Honest framing
    --------------
    This is a documentation dataclass.  The search layer reads it
    for logging only — it does NOT branch on these values.  The
    aquation reaction SMARTS are unconditional: when the rule
    fires, the products form regardless of pKa / Cl- concentration.
    Real aquation kinetics (pKa, rate constants) are the
    :class:`molmetal_lam.reactions.rate_predictor`'s job, not this
    module's.
    """

    pka1: float = 6.5
    ionic_strength_M: float = 0.10
    temperature_K: float = 310.0
    notes: str = "first aquation (Reedijk 1987 cisplatin canonical)"


# Canonical AquaContext instances for the 2 patterns.
AQUA_CONTEXTS: Dict[str, AquaContext] = {
    "Pt_Cl_H2O_first": AquaContext(
        pka1=6.5, ionic_strength_M=0.10, temperature_K=310.0,
        notes="first aquation (Reedijk 1987 cisplatin canonical)",
    ),
    "Pt_OHCl_H2O_second": AquaContext(
        pka1=7.2, ionic_strength_M=0.10, temperature_K=310.0,
        notes="second aquation — diaqua complex that binds DNA-N7-guanine",
    ),
}


# ---------------------------------------------------------------------------
# Convenience accessors.
# ---------------------------------------------------------------------------
def get_metal_ligand_exchange_patterns() -> List[Tuple[str, str, str]]:
    """Return the canonical 5 MetalLigandExchange SMARTS patterns."""
    return list(METAL_LIGAND_EXCHANGE_SMARTS)


def get_aqua_exchange_patterns() -> List[Tuple[str, str, str]]:
    """Return the canonical 2 AquaExchange SMARTS patterns."""
    return list(AQUA_EXCHANGE_SMARTS)


def get_aqua_context(pattern_name: str) -> AquaContext:
    """Return the AquaContext for a pattern name.  Falls back to the
    ``first aquation`` default when the pattern is unknown."""
    return AQUA_CONTEXTS.get(pattern_name, AQUA_CONTEXTS["Pt_Cl_H2O_first"])


# ---------------------------------------------------------------------------
# Pretty-printer for the audit report.
# ---------------------------------------------------------------------------
def render_metal_ligand_exchange_table() -> str:
    """Render the 5 MetalLigandExchange SMARTS as a markdown table."""
    rows = [
        "| pattern_name | description | SMARTS |",
        "| --- | --- | --- |",
    ]
    for name, smarts, desc in METAL_LIGAND_EXCHANGE_SMARTS:
        rows.append(f"| `{name}` | {desc} | `{smarts}` |")
    return "\n".join(rows)


def render_aqua_exchange_table() -> str:
    """Render the 2 AquaExchange SMARTS as a markdown table."""
    rows = [
        "| pattern_name | description | pKa1 | SMARTS |",
        "| --- | --- | --- | --- |",
    ]
    for name, smarts, desc in AQUA_EXCHANGE_SMARTS:
        ctx = AQUA_CONTEXTS.get(name)
        pka_str = f"{ctx.pka1:.1f}" if ctx else "—"
        rows.append(f"| `{name}` | {desc} | {pka_str} | `{smarts}` |")
    return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover
    print(render_metal_ligand_exchange_table())
    print()
    print(render_aqua_exchange_table())
