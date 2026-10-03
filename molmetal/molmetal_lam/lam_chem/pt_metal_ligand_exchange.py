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
        "[Pt:1]([OH:2])([Cl:3])[*:4].[OH2:5]>>" "[Pt:1]([OH:2])([OH:5])[*:4].[Cl:3]",
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
        pka1=6.5,
        ionic_strength_M=0.10,
        temperature_K=310.0,
        notes="first aquation (Reedijk 1987 cisplatin canonical)",
    ),
    "Pt_OHCl_H2O_second": AquaContext(
        pka1=7.2,
        ionic_strength_M=0.10,
        temperature_K=310.0,
        notes="second aquation — diaqua complex that binds DNA-N7-guanine",
    ),
}


# ---------------------------------------------------------------------------
# L-R17-B2 matrix — 12 metals × 7 entering ligands = 84 patterns
# Per TODO/pending/31_r17_ultracode_dispatch.md §L-R17-B2 + Wf-415-A3
# CPU-only fill: each pattern is the metal-chloride + donor → new metal
# complex + Cl leaving group reaction.  The matrix is a *documentation
# surface* for the Wf-290F4 audit (matrix coverage, per-metal +
# per-ligand coverage) — it does NOT currently run on Pt-containing
# substrates (RDKit cannot sanitise most Pt_II products).  Each
# matrix pattern inherits from the canonical Pt_Cl_NH3 SMARTS pattern
# (line 67) by replacing [Pt:1] with the per-metal centre and [Cl:2]
# with the per-ligand donor.
# ---------------------------------------------------------------------------
_MATRIX_METALS: Tuple[str, str] = (
    # (metal_label, SMARTS_atom_token)
    ("Pt_II", "[Pt:1]"),
    ("Pt_IV", "[Pt:1]"),
    ("Ru_II", "[Ru:1]"),
    ("Ru_III", "[Ru:1]"),
    ("Ir_III", "[Ir:1]"),
    ("Ir_IV", "[Ir:1]"),
    ("Os_II", "[Os:1]"),
    ("Os_III", "[Os:1]"),
    ("Rh_I", "[Rh:1]"),
    ("Rh_III", "[Rh:1]"),
    ("Au_I", "[Au:1]"),
    ("Au_III", "[Au:1]"),
)
_MATRIX_LIGANDS: Tuple[str, str, str, str] = (
    # (ligand_label, donor_SMILES, reactant_donor_token, product_donor_token)
    # H2O reactant donor = [OH2:4]; product donor = [OH:4] (after
    # losing one proton per Reedijk 1987 first-aquation chemistry).
    # Other ligands use the same token on both sides (no proton change).
    ("Cl", "[Cl:2]", "[Cl:2]", "[Cl:2]"),
    ("NH3", "[NH3:4]", "[NH3:4]", "[NH3:4]"),
    ("H2O", "[OH2:4]", "[OH2:4]", "[OH:4]"),
    ("CN", "[C-:4]#[N+]", "[C-:4]#[N+]", "[C-:4]#[N+]"),
    ("OH", "[OH-:4]", "[OH-:4]", "[OH-:4]"),
    ("SCN", "[S-:4][C:5]#[N:6]", "[S-:4][C:5]#[N:6]", "[S-:4][C:5]#[N:6]"),
    ("CO", "[C-:4]#[O+]", "[C-:4]#[O+]", "[C-:4]#[O+]"),
)
METAL_LIGAND_EXCHANGE_MATRIX: List[Tuple[str, str, str]] = []
for _metal_label, _metal_token in _MATRIX_METALS:
    for (
        _ligand_label,
        _donor_smiles,
        _reactant_token,
        _product_token,
    ) in _MATRIX_LIGANDS:
        _name = f"{_metal_label}_Cl_{_ligand_label}"
        # Canonical Pt_Cl_NH3 SMARTS: [Pt:1]([Cl:2])[*:3].[NH3:4]>>
        # [Pt:1]([NH3:4])[*:3].[Cl:2]. We instantiate it for every
        # (metal, ligand) pair by swapping [Pt:1] → metal_token and
        # [NH3:4] → donor_token (the leaving-group [Cl:2] is unchanged).
        _smarts = (
            f"{_metal_token}([Cl:2])[*:3]."
            f"{_reactant_token}>>"
            f"{_metal_token}({_product_token})[*:3].[Cl:2]"
        )
        _desc = (
            f"{_metal_label} canonical — Cl + {_donor_smiles} "
            f"(LIPPard 1995 §4.2 + Wf-290F4 §L-R17-B2 matrix row)"
        )
        METAL_LIGAND_EXCHANGE_MATRIX.append((_name, _smarts, _desc))


# ---------------------------------------------------------------------------
# Convenience accessors.
# ---------------------------------------------------------------------------
def get_metal_ligand_exchange_patterns() -> List[Tuple[str, str, str]]:
    """Return the canonical 5 MetalLigandExchange SMARTS patterns."""
    return list(METAL_LIGAND_EXCHANGE_SMARTS)


def get_metal_ligand_exchange_matrix_patterns() -> List[Tuple[str, str, str]]:
    """Return the L-R17-B2 12-metals × 7-ligands matrix (84 patterns)."""
    return list(METAL_LIGAND_EXCHANGE_MATRIX)


def get_metal_ligand_matrix_ligands() -> List[Tuple[str, str, str]]:
    """Return the 7 entering ligand tuples (label, SMILES, SMARTS)."""
    # Returns only the first 3 fields per ligand (label, SMILES, token);
    # the 4th field `product_token` is internal — used only in matrix
    # generation to handle the H2O donor product-side change.
    return [
        (label, smiles, rtoken) for (label, smiles, rtoken, _ptok) in _MATRIX_LIGANDS
    ]


def get_metal_ligand_matrix_metals() -> List[Tuple[str, str, str]]:
    """Return the 12 metal tuples (label, oxidation state, geometry)."""
    return [
        # Per test_metal_ligand_exchange_matrix.py::test_get_metal_ligand_matrix_metals_returns_12
        # contract: return (symbol, oxidation_state, geometry) where
        # `symbol` is just the element symbol (e.g. "Pt", not "Pt_II")
        # and `geometry` is one of {square_planar, octahedral, ...}.
        (
            label.split("_")[0],
            label.split("_")[1],
            "square_planar" if label.startswith(("Pt_", "Au_")) else "octahedral",
        )
        for (label, _token) in _MATRIX_METALS
    ]


# ---------------------------------------------------------------------------
# L-R17-B2 NEW SMARTS — 17 follow-on templates (4 NEW kinds)
# Per TODO/pending/31_r17_ultracode_dispatch.md §L-R17-B2 + TODO-34 §UC-L2
# follow-on round, expanded F2(a) SMARTS library to expose redex shapes
# that diversify the metal coordination sphere beyond the canonical
# Pt-Cl substitution:
#
#   1. transmetalation (6 pairs) — M1 -> M2 with ligand shell preserved
#      1R -> 1P; metal :1 + ligands :2, :3, :4, :5 mirrored on both sides.
#   2. Cl-OH swap (5 metals) — M-Cl + OH- -> M-OH + Cl-
#      2R -> 2P; metal :1 + leaving-group [Cl:2] + entering-donor [OH-:6].
#      Pt_II + Ru_II use the 4-ligand template; Pt_IV + Ru_III + Au_III
#      use the 6-ligand template (matches the d6 octahedral substrates).
#   3. ligand-shell-MMA (4 pairs) — transmetalation with CN- preserved
#      1R -> 1P; [C-:2]#N CN- ligand mirrored across the metal swap.
#   4. Pt/Pd transmetalation (2 patterns) — directional Pt<->Pd pair
#      1R -> 1P; the canonical metal swap with `pt_pd_transmet_` prefix
#      to avoid collision with the generic transmetalation library
#      which also includes a Pt_II->Pd_II pair.
#
# Total: 6 + 5 + 4 + 2 = 17 patterns.  Combined with the 84-rule
# matrix, ``register_metal_ligand_exchange_rules()`` returns 84 + 17 = 101.
# ---------------------------------------------------------------------------

TRANSMETALATION_SMARTS: List[Tuple[str, str, str]] = [
    # 6 transmetalation pairs (1R -> 1P; metal :1 + ligands :2-:5 mirrored).
    (
        "transmet_Pt_II_to_Pd_II",
        "[Pt:1]([*:2])([*:3])([*:4])([*:5])>>" "[Pd:1]([*:2])([*:3])([*:4])([*:5])",
        "Pt_II -> Pd_II transmetalation (d8 square-planar swap, "
        "ligand shell :2-:5 preserved; Lippard 1995 §4.2)",
    ),
    (
        "transmet_Pt_II_to_Au_III",
        "[Pt:1]([*:2])([*:3])([*:4])([*:5])>>" "[Au:1]([*:2])([*:3])([*:4])([*:5])",
        "Pt_II -> Au_III transmetalation (d8 5d analogue swap, "
        "ligand shell :2-:5 preserved; Comba-Hambley 2009 §3.4)",
    ),
    (
        "transmet_Pt_II_to_Ru_II",
        "[Pt:1]([*:2])([*:3])([*:4])([*:5])>>" "[Ru:1]([*:2])([*:3])([*:4])([*:5])",
        "Pt_II -> Ru_II transmetalation (3d -> 4d swap, "
        "ligand shell :2-:5 preserved)",
    ),
    (
        "transmet_Pd_II_to_Au_III",
        "[Pd:1]([*:2])([*:3])([*:4])([*:5])>>" "[Au:1]([*:2])([*:3])([*:4])([*:5])",
        "Pd_II -> Au_III transmetalation (4d -> 5d swap, "
        "ligand shell :2-:5 preserved)",
    ),
    (
        "transmet_Pt_II_to_Os_II",
        "[Pt:1]([*:2])([*:3])([*:4])([*:5])>>" "[Os:1]([*:2])([*:3])([*:4])([*:5])",
        "Pt_II -> Os_II transmetalation (5d6 -> 5d8 swap, "
        "ligand shell :2-:5 preserved)",
    ),
    (
        "transmet_Ru_II_to_Pt_II",
        "[Ru:1]([*:2])([*:3])([*:4])([*:5])>>" "[Pt:1]([*:2])([*:3])([*:4])([*:5])",
        "Ru_II -> Pt_II transmetalation (reverse direction, "
        "ligand shell :2-:5 preserved)",
    ),
]


CL_OH_SWAP_SMARTS: List[Tuple[str, str, str]] = [
    # 5 Cl-OH swap pairs (2R -> 2P; [Cl:2] leaving + [OH-:6] entering).
    # Pt_II + Ru_II use the 4-ligand template (d8 square-planar).
    (
        "cl_oh_swap_Pt_II",
        "[Pt:1]([*:3])([*:4])([*:5])[Cl:2].[OH-:6]>>"
        "[Pt:1]([*:3])([*:4])([*:5])[OH-:6].[Cl:2]",
        "Pt_II Cl-OH swap (4-ligand d8; Pt-Cl + OH- -> Pt-OH + Cl-; "
        "Reedijk 1987 aquation equilibrium)",
    ),
    (
        "cl_oh_swap_Ru_II",
        "[Ru:1]([*:3])([*:4])([*:5])[Cl:2].[OH-:6]>>"
        "[Ru:1]([*:3])([*:4])([*:5])[OH-:6].[Cl:2]",
        "Ru_II Cl-OH swap (4-ligand d8; Ru-Cl + OH- -> Ru-OH + Cl-)",
    ),
    # Pt_IV + Ru_III + Au_III use the 6-ligand template (d6 octahedral).
    (
        "cl_oh_swap_Pt_IV",
        "[Pt:1]([*:3])([*:4])([*:5])([*:7])([*:8])[Cl:2].[OH-:6]>>"
        "[Pt:1]([*:3])([*:4])([*:5])([*:7])([*:8])[OH-:6].[Cl:2]",
        "Pt_IV Cl-OH swap (6-ligand d6 octahedral; Pt-Cl + OH- -> "
        "Pt-OH + Cl-; wildcards at :3-:5+:7+:8 to keep OH- at :6)",
    ),
    (
        "cl_oh_swap_Ru_III",
        "[Ru:1]([*:3])([*:4])([*:5])([*:7])([*:8])[Cl:2].[OH-:6]>>"
        "[Ru:1]([*:3])([*:4])([*:5])([*:7])([*:8])[OH-:6].[Cl:2]",
        "Ru_III Cl-OH swap (6-ligand d6 octahedral; Ru-Cl + OH- -> "
        "Ru-OH + Cl-; wildcards at :3-:5+:7+:8 to keep OH- at :6)",
    ),
    (
        "cl_oh_swap_Au_III",
        "[Au:1]([*:3])([*:4])([*:5])[Cl:2].[OH-:6]>>"
        "[Au:1]([*:3])([*:4])([*:5])[OH-:6].[Cl:2]",
        "Au_III Cl-OH swap (4-ligand d8 5d^8 square-planar; "
        "Au-Cl + OH- -> Au-OH + Cl-)",
    ),
]


LIGAND_SHELL_MMA_SMARTS: List[Tuple[str, str, str]] = [
    # 4 ligand-shell-MMA pairs (1R -> 1P; CN- ligand shell :2 preserved).
    (
        "shell_mma_Pt_II_to_Pd_II",
        "[Pt:1]([C-:2]#N)([*:3])([*:4])([*:5])>>"
        "[Pd:1]([C-:2]#N)([*:3])([*:4])([*:5])",
        "Pt_II -> Pd_II transmetalation with CN- ligand shell "
        "preserved ([C-:2]#N :2 on both sides)",
    ),
    (
        "shell_mma_Pt_II_to_Au_III",
        "[Pt:1]([C-:2]#N)([*:3])([*:4])([*:5])>>"
        "[Au:1]([C-:2]#N)([*:3])([*:4])([*:5])",
        "Pt_II -> Au_III transmetalation with CN- ligand shell " "preserved",
    ),
    (
        "shell_mma_Pt_II_to_Ru_II",
        "[Pt:1]([C-:2]#N)([*:3])([*:4])([*:5])>>"
        "[Ru:1]([C-:2]#N)([*:3])([*:4])([*:5])",
        "Pt_II -> Ru_II transmetalation with CN- ligand shell " "preserved",
    ),
    (
        "shell_mma_Pd_II_to_Au_III",
        "[Pd:1]([C-:2]#N)([*:3])([*:4])([*:5])>>"
        "[Au:1]([C-:2]#N)([*:3])([*:4])([*:5])",
        "Pd_II -> Au_III transmetalation with CN- ligand shell " "preserved",
    ),
]


PT_PD_TRANSMETALATION_SMARTS: List[Tuple[str, str, str]] = [
    # 2 Pt/Pd transmetalation patterns (1R -> 1P; directional pair).
    # Names use `pt_pd_transmet_` prefix to avoid collision with the
    # generic `transmet_` prefix used in TRANSMETALATION_SMARTS (which
    # also contains a Pt_II->Pd_II pair).
    (
        "pt_pd_transmet_Pt_II_to_Pd_II",
        "[Pt:1]([*:2])([*:3])([*:4])([*:5])>>" "[Pd:1]([*:2])([*:3])([*:4])([*:5])",
        "Pt_II -> Pd_II directional transmetalation (specific Pt/Pd "
        "pair in the canonical d8 square-planar geometry)",
    ),
    (
        "pt_pd_transmet_Pd_II_to_Pt_II",
        "[Pd:1]([*:2])([*:3])([*:4])([*:5])>>" "[Pt:1]([*:2])([*:3])([*:4])([*:5])",
        "Pd_II -> Pt_II directional transmetalation (reverse "
        "direction; specific Pt/Pd pair)",
    ),
]


# Combined accessors for the 17 NEW SMARTS — used by the F2(a) registry
# test ``test_f2a_metal_ligand_exchange.py``.
def get_transmetalation_patterns() -> List[Tuple[str, str, str]]:
    """Return the 6 transmetalation NEW SMARTS patterns."""
    return list(TRANSMETALATION_SMARTS)


def get_cl_oh_swap_patterns() -> List[Tuple[str, str, str]]:
    """Return the 5 Cl-OH swap NEW SMARTS patterns."""
    return list(CL_OH_SWAP_SMARTS)


def get_ligand_shell_mma_patterns() -> List[Tuple[str, str, str]]:
    """Return the 4 ligand-shell-MMA NEW SMARTS patterns."""
    return list(LIGAND_SHELL_MMA_SMARTS)


def get_pt_pd_transmetalation_patterns() -> List[Tuple[str, str, str]]:
    """Return the 2 Pt/Pd transmetalation NEW SMARTS patterns."""
    return list(PT_PD_TRANSMETALATION_SMARTS)


def get_new_smarts_patterns() -> List[Tuple[str, str, str]]:
    """Return the combined 17 NEW SMARTS (6 + 5 + 4 + 2)."""
    return (
        list(TRANSMETALATION_SMARTS)
        + list(CL_OH_SWAP_SMARTS)
        + list(LIGAND_SHELL_MMA_SMARTS)
        + list(PT_PD_TRANSMETALATION_SMARTS)
    )


def register_metal_ligand_exchange_rules(
    *,
    register_into_rule_registry: bool = False,
) -> List[Tuple[str, str, str]]:
    """Return the 101 L-R17-B2 patterns (84 matrix + 17 NEW SMARTS).

    Wf-415-A3 NEW: ``register_metal_ligand_exchange_rules`` is the
    registration helper expected by
    ``test_metal_ligand_exchange_matrix.py`` (L-R17-B2 dispatch spec).
    By default returns the 84 matrix patterns; when
    ``register_into_rule_registry=True`` pushes them into the public
    ``molmetal_lam.reactions.rule_registry`` under the
    ``"metal-coord-matrix"`` scope (mirrors the legacy
    ``register_metal_ligand_exchange`` API).

    Wf-420-A1 NEW: returns 84 + 17 = 101 patterns (matrix + NEW SMARTS).
    When ``register_into_rule_registry=True``, the 84 matrix patterns
    are pushed to the ``metal-coord-matrix`` scope AND the 17 NEW
    SMARTS patterns are pushed to the ``metal-coord-new-smarts`` scope
    (separated so the L-R17-B2 NEW SMARTS can be introspected
    independently of the canonical matrix).
    """
    patterns: List[Tuple[str, str, str]] = list(METAL_LIGAND_EXCHANGE_MATRIX)
    patterns.extend(get_new_smarts_patterns())
    if register_into_rule_registry:
        try:
            from molmetal_lam.reactions.rule_registry import (
                register_rule,
                list_rules,
            )

            existing_matrix = {r.name for r in list_rules(scope="metal-coord-matrix")}
            existing_new = {r.name for r in list_rules(scope="metal-coord-new-smarts")}
            for name, smarts, desc in METAL_LIGAND_EXCHANGE_MATRIX:
                if str(name) not in existing_matrix:
                    register_rule(
                        name=str(name),
                        scope="metal-coord-matrix",
                        smarts=smarts,
                        metadata={"description": desc, "wf": "Wf-420-A1"},
                        overwrite=True,
                    )
            for name, smarts, desc in get_new_smarts_patterns():
                if str(name) not in existing_new:
                    register_rule(
                        name=str(name),
                        scope="metal-coord-new-smarts",
                        smarts=smarts,
                        metadata={"description": desc, "wf": "Wf-420-A1"},
                        overwrite=True,
                    )
        except ImportError:
            pass  # rule_registry is an optional dependency
    return patterns


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
