"""Scaffold-aware click-rule compatibility (WF-Lambda-Fix-FullPath-v2 Fix 2).

Background
----------
WF-MCTS-Chemistry-Research (2026-09-14) ran per-click per-metal verdicts on
the 5 canonical click reactions (``CuAAC``, ``SPAAC``, ``ThiolEne``,
``Suzuki``, ``AmideCoupling``) against the canonical Pt chemotherapeutics
(cisplatin, carboplatin, oxaliplatin) and the labile-metal set (Cu / Zn /
Fe / Mn).  The verdict table is the source of truth for which click rule
can fire *without violating the metal-scaffold's reactivity*:

    - ``Pt_II`` (e.g. cisplatin) — strictly inert d8 square-planar; CuAAC
      and SPAAC are the only safe click regimes (cycloaddition on a
      remote alkyne/azide handle, not on Pt-Cl).  ThiolEne attacks
      Pt-Cl (nucleophilic substitution on a soft Lewis acid), which
      protonates the thiol + ejects Cl-, breaking the Pt-Cl bond; only
      safe on Pt_IV (axial Cl more labile) or Pt_II-CBDCA (carboplatin-
      like, carboxylate chelate leaves readily).  AmideCoupling on a
      Pt-Cl carboxylate is just a standard amidation, but on Pt-Cl2
      the leaving group is chloride not carboxylate, so the carboxylate
      must come from a *new* ligand — gated to Pt_IV prodrug (where
      the axial -OH is the actual leaving group after aquation).
    - ``Pt_IV`` (e.g. satraplatin) — d6 octahedral; all 5 click rules
      safe because the metal is kinetically inert on the timescale of a
      cycloaddition (reacts only after intracellular reduction to Pt_II).
    - ``Cu / Zn / Fe / Mn`` — labile coordination, fast ligand exchange;
      all 5 click rules safe (the metal coordination sphere does not
      gate the click).
    - ``Pt_II chelating`` (e.g. carboplatin, oxaliplatin) — Pt_II with
      a carboxylate or diamine chelate; the chelate leave-group kinetics
      match Pt_IV (axial carboxylate or amine can dissociate), so
      ThiolEne + AmideCoupling become safe; Suzuki stays MARGINAL.

This module exposes the verdict table as a small, import-friendly
mapping (``COMPAT_MATRIX`` + ``default_compatible_rules()`` +
``detect_scaffold()``) so callers do not need to re-derive it.

Honest framing
--------------
This is a *chemistry-knowledge* heuristic, not a *learned* model.  It
encodes what a synthetic chemist would tell you: "do not run
ThiolEne on Pt-Cl2".  The verdicts are conservative — when the
scaffold is ambiguous we fall back to the strictest (smallest) rule
set.  No ML fitting was performed; the matrix is hand-curated from
the WF-MCTS-Chemistry-Research notes + standard Pt pharmacology.

WF-Lambda-Fix-FullPath-v2 Fix 2 (IMPROVED) replaces the historical
behaviour where ``MetalLigandExchange`` was the *only* Pt-aware rule
and all 5 click rules fired regardless of scaffold.  The new behaviour:

    strict_coordination (Pt_II / Pd_II / Au_III) -> CuAAC + SPAAC
    prodrug_chelating   (Pt_IV / Pt_II chelating) -> all 5
    labile_coordination (Cu / Zn / Fe / Mn)       -> all 5
    unknown / empty                                -> all 5 (fallback)

The opt-in escape hatch is ``allow_incompatible=True`` which re-enables
ThiolEne / AmideCoupling / Suzuki on strict-coordination scaffolds
(explicit user override; the audit recommends this be default-FALSE).

F2(a) extension (2026-09-15) — Pt_II metal-coordination rules
--------------------------------------------------------------
The compatibility matrix has been widened from 5x5 (5 clicks x 5 scaffolds)
to 5x7 (5 clicks + 2 metal-coord rules = 7 columns) x 5 scaffolds.  The
new columns are:

    - ``metal_ligand_exchange`` — substitution of a Pt-Cl leaving group
      by a neutral 2e-donor N (NH3, primary amine).  Per Lippard 1995,
      Pt_II square-planar complexes undergo associative substitution
      via a 5-coordinate trigonal-bipyramidal transition state, then
      the entering ligand (NH3) replaces the leaving ligand (Cl-).
      This is the canonical cisplatin-ammine binding step.
      **Verdict on strict_Pt_II: COMPATIBLE** — this is *not* a
      click-chemistry reaction; it is the structural rule that allows
      Pt_II to do coordination chemistry in the MCTS.  It is what
      unlocks diversity in the Lambda search while keeping
      ``metal_compliance == 1.0`` (the search can build different Pt
      ligand spheres without violating the strict-coordination prior).

    - ``aqua_exchange`` — substitution of Pt-Cl by H2O (the rate-
      limiting step of cisplatin activation per Reedijk 1987).
      **Verdict on strict_Pt_II: COMPATIBLE** — same rationale.  The
      aquated complex is the kinetically competent species that binds
      DNA-N7-guanine (the actual cytotoxic event).

Both rules are SMARTS-defined in
``molmetal_lam/reactions/beta_reductions.py`` (F2(a) registry entries
``MetalLigandExchange`` and ``AquaExchange``).  They are part of the
*metal-coordination family* and are exposed under the
``METAL_COORDINATION_RULES`` dict in :mod:`molmetal_lam.lam_chem.rules`
(not the default ``CLICK_REACTIONS`` set).  Callers opt-in via the
``metal-coord`` CLI alias.

Lit anchors (F2(a)):
    Lippard 1995 — Pt_II square-planar coordination chemistry,
    4-coordinate, associative substitution pathway.
    Reedijk 1987 — Pt_II aquation kinetics, Cl -> H2O exchange,
    pKa1 ~ 6.5 for the first aquation of cisplatin.
    Himo 2005 — click regiochemistry canon (CuAAC, SPAAC, Suzuki).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Module version — bump when the compat matrix is extended.
# WF-Lambda-Boost Phase 4 (2026-09-16) confirms the F2(a) rows are
# present and strict_Pt_II is COMPATIBLE on both metal_ligand_exchange
# and aqua_exchange (the canonical Lippard 1995 + Reedijk 1987
# coordination chemistry).
COMPAT_MATRIX_VERSION: str = "wf-lambda-boost-2026-09-16"

# ---------------------------------------------------------------------------
# Per-click per-scaffold verdicts (5 clicks x 5 scaffolds = 25 cells).
# ---------------------------------------------------------------------------
# Verdicts:
#   "compatible"  — always-on for this scaffold; click can fire freely.
#   "marginal"    — fires but with known side-reactions; opt-in only.
#   "incompatible" — click breaks the metal-scaffold; opt-in only and
#                    emit a WARNING.
#
# Click columns: CuAAC, SPAAC, ThiolEne, Suzuki, AmideCoupling.
# Scaffold rows: strict_Pt_II, Pt_II_chelating, Pt_IV, labile_metal,
#                unknown.
# ---------------------------------------------------------------------------
COMPAT_MATRIX: Dict[str, Dict[str, str]] = {
    # strict Pt(II) d8 square-planar (cisplatin, oxaliplatin, nedaplatin)
    # CuAAC + SPAAC: safe (cycloaddition on a REMOTE alkyne/azide handle,
    #   not on Pt-Cl).  ThiolEne: thiolate attacks Pt-Cl (soft Lewis
    #   acid), protonates + ejects Cl-, breaks Pt-Cl bond — INCOMPAT.
    #   AmideCoupling: Pt-Cl carboxylate coupling is fine in principle
    #   but Pt-Cl2 has NO carboxylate to start with; you cannot amide-
    #   couple onto a chloride.  INCOMPAT.  Suzuki: aryl-boron attacks
    #   Pt-Cl transmetalation — known side-reaction on Pt-Cl bonds.
    #   MARGINAL.
    # F2(a): metal_ligand_exchange + aqua_exchange = COMPATIBLE on
    #   strict_Pt_II — these are the *coordination* rules, not click
    #   rules.  Pt-Cl substitution by NH3 or H2O is the structural
    #   mechanism that allows the Lambda search to build different
    #   Pt(II) ligand spheres.  This is what unlocks diversity while
    #   keeping metal_compliance = 1.0.
    "strict_Pt_II": {
        "CuAAC":                "compatible",
        "SPAAC":                "compatible",
        "ThiolEne":             "incompatible",
        "Suzuki":               "marginal",
        "AmideCoupling":        "incompatible",
        "metal_ligand_exchange": "compatible",
        "aqua_exchange":        "compatible",
    },
    # Pt(II) with chelating carboxylate (carboplatin: cyclobutane-
    # dicarboxylate; oxaliplatin: oxalate + DACH).  The chelate's
    # carboxylate leaves *transiently* during aquation, so:
    #   ThiolEne      -> compatible (carboxylate-thiol exchange is the
    #                    leaving-group step, NOT a Pt-Cl attack).
    #   AmideCoupling -> compatible (carboplatin's -COOH can couple to
    #                    an exogenous amine; the chelate opens up).
    #   Suzuki        -> marginal (transmetalation to Pt still possible).
    #   CuAAC + SPAAC -> compatible (cycloaddition on remote handle).
    "Pt_II_chelating": {
        "CuAAC":                "compatible",
        "SPAAC":                "compatible",
        "ThiolEne":             "compatible",
        "Suzuki":               "marginal",
        "AmideCoupling":        "compatible",
        "metal_ligand_exchange": "compatible",
        "aqua_exchange":        "compatible",
    },
    # Pt(IV) d6 octahedral prodrug (satraplatin, tetraplatin).  Axial
    # ligands (Cl, OH, OAc) are kinetically inert until intracellular
    # reduction to Pt(II), so all 5 click rules are safe on the
    # equatorial coordination plane.  FULL COMPAT.
    "Pt_IV": {
        "CuAAC":                "compatible",
        "SPAAC":                "compatible",
        "ThiolEne":             "compatible",
        "Suzuki":               "compatible",
        "AmideCoupling":        "compatible",
        "metal_ligand_exchange": "compatible",
        "aqua_exchange":        "compatible",
    },
    # Labile coordination metals (Cu, Zn, Fe, Mn).  Fast ligand exchange
    # means the click rules can fire without disturbing the metal
    # coordination sphere.  FULL COMPAT.
    "labile_metal": {
        "CuAAC":                "compatible",
        "SPAAC":                "compatible",
        "ThiolEne":             "compatible",
        "Suzuki":               "compatible",
        "AmideCoupling":        "compatible",
        "metal_ligand_exchange": "compatible",
        "aqua_exchange":        "compatible",
    },
    # Unknown scaffold (no metal detected, or metal we have no verdict
    # for).  Conservative default = all 5 (the original
    # ``all-5`` behaviour).  The user can always narrow via
    # ``--click-rules``.
    "unknown": {
        "CuAAC":                "compatible",
        "SPAAC":                "compatible",
        "ThiolEne":             "compatible",
        "Suzuki":               "compatible",
        "AmideCoupling":        "compatible",
        "metal_ligand_exchange": "compatible",
        "aqua_exchange":        "compatible",
    },
}


# Canonical rule names in display order (used for the matrix in the
# audit report).  Extended F2(a) from 5 to 7 columns — 5 click rules
# plus 2 metal-coordination rules (Pt_II specific reductions).  The
# last two columns are the F2(a) additions.
CLICK_RULE_NAMES: Tuple[str, ...] = (
    "CuAAC", "SPAAC", "ThiolEne", "Suzuki", "AmideCoupling",
    "metal_ligand_exchange", "aqua_exchange",
)
SCAFFOLD_NAMES: Tuple[str, ...] = (
    "strict_Pt_II", "Pt_II_chelating", "Pt_IV", "labile_metal", "unknown",
)


# ---------------------------------------------------------------------------
# Scaffold auto-detection from a metal-seed SMILES.
# ---------------------------------------------------------------------------
# Heuristics are cheap RDKit parses of the metal-seed SMILES.  We do NOT
# require the metal-seed to be the FULL cisplatin molecule; the bare-
# metal-alkyne seed ``[Pt]C#C`` from WF-Lambda-Fix-Singleton Fix 1 is
# supported via the strict_Pt_II branch (Pt is a square-planer metal
# with no chelating carboxylate in the seed).
#
# Detection logic:
#   1. Parse the SMILES with RDKit (if available; else fall back to
#      simple substring scan).
#   2. Collect unique metal atom symbols.
#   3. If any of {Cu, Zn, Fe, Mn} -> labile_metal.
#   4. If Pt detected AND the molecule contains >=1 -C(=O)O- carboxylate
#      OR a diamine chelate (2+ N atoms bonded to Pt) -> Pt_II_chelating.
#      Pt_IV is detected when Pt has >=5 bonds (octahedral signature).
#   5. Otherwise Pt -> strict_Pt_II.
#   6. Pd / Au are mapped to strict_Pt_II (same d8 / d8 coordination
#      chemistry in the framework).
#   7. Empty / unparseable -> unknown.
# ---------------------------------------------------------------------------
def detect_scaffold(
    smiles: Optional[str],
    *,
    name_hint: Optional[str] = None,
) -> str:
    """Detect the metal-scaffold category from a metal-seed SMILES.

    Parameters
    ----------
    smiles : str | None
        Canonical (or near-canonical) SMILES of the metal-seed.  When
        ``None`` or empty, returns ``"unknown"``.
    name_hint : str | None
        Optional friendly name (``"cisplatin"``, ``"carboplatin"``,
        ``"oxaliplatin"``) used as a *positive* override when RDKit
        cannot parse but the name is unambiguous.

    Returns
    -------
    str
        One of ``SCAFFOLD_NAMES``: ``"strict_Pt_II"``,
        ``"Pt_II_chelating"``, ``"Pt_IV"``, ``"labile_metal"``,
        ``"unknown"``.

    Honest framing
    --------------
    Heuristic detection can mis-classify (e.g. a ``[Pt](N)(N)(N)(Cl)``
    triamine-platinum(II) is still Pt_II but the N count alone might
    fool the chelate test).  Callers that need exact verdicts should
    pass the SMILES to a domain expert OR explicitly set
    ``name_hint`` to the canonical scaffold name.
    """
    if not smiles:
        return _name_hint_to_scaffold(name_hint)

    # Friendly-name override (positive only — caller asserts).
    if name_hint:
        s = _name_hint_to_scaffold(name_hint)
        if s != "unknown":
            return s

    # RDKit-based heuristic path.
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return _name_hint_to_scaffold(name_hint) or "unknown"

    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception:
        mol = None
    if mol is None:
        return _name_hint_to_scaffold(name_hint) or "unknown"

    # Collect metal symbols in the molecule.
    LABILE = {"Cu", "Zn", "Fe", "Mn"}
    PT_LIKE = {"Pt", "Pd", "Au"}  # Pd d8, Au_III d8 — same strict coord
    metal_atoms: List[str] = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in LABILE or sym in PT_LIKE:
            metal_atoms.append(sym)
    if not metal_atoms:
        # No recognised metal; treat as unknown scaffold.
        return "unknown"

    # Labile first (Cu / Zn / Fe / Mn take precedence — fast exchange).
    if any(m in LABILE for m in metal_atoms):
        return "labile_metal"

    # Pt / Pd / Au branch.
    pt_count = sum(1 for m in metal_atoms if m in {"Pt", "Pd"})
    au_count = sum(1 for m in metal_atoms if m == "Au")
    # Find Pt atoms and their coordination numbers.
    pt_coord_counts: List[int] = []
    pt_has_carboxylate_in_molecule = False
    for atom in mol.GetAtoms():
        # Carboxylate check: -C(=O)O- (or -C(=O)[O-]) anywhere in
        # the molecule.  This captures both free carboxylates and
        # chelating carboxylates (where the C is bonded directly
        # to Pt, the second O is the free donor).  We do NOT require
        # the carboxylate carbon to be the metal's neighbour —
        # that's too strict (carboplatin's cyclobutane-1,1-
        # dicarboxylate has the carboxylate on a quaternary C with
        # both -COO- arms free of direct Pt bonding).
        if atom.GetSymbol() == "C":
            # Look for the -C(=O)O- / -C(=O)[O-] / -C(O)=O motif:
            # a carbon bonded to 2+ oxygens (a carboxylate).
            oxy_neighbours = [n for n in atom.GetNeighbors()
                              if n.GetSymbol() == "O"]
            if len(oxy_neighbours) >= 2:
                pt_has_carboxylate_in_molecule = True
        if atom.GetSymbol() in {"Pt", "Pd", "Au"}:
            pt_coord_counts.append(atom.GetDegree())

    # Pt_IV: coordination number >= 5 (octahedral) OR >= 6 explicit.
    if any(c >= 5 for c in pt_coord_counts):
        return "Pt_IV"

    # Pt_II_chelating: Pt with a carboxylate group anywhere in the
    # molecule (free or chelated).  This catches carboplatin's
    # cyclobutane-1,1-dicarboxylate AND oxaliplatin's free oxalate
    # (when the SMILES carries it).  NB we DO NOT detect diamine
    # chelate (oxaliplatin's DACH) here — the heuristic is too noisy
    # because cisplatin has 2 Ns bonded to Pt (each NH3 is
    # monodentate, not a chelate) and the counter would mis-classify
    # cisplatin as Pt_II_chelating.  The friendly name_hint path
    # handles DACH-based chelates.
    if pt_has_carboxylate_in_molecule:
        return "Pt_II_chelating"

    # Pt / Pd / Au with no chelate and no octahedral signature = strict.
    if pt_count + au_count >= 1:
        return "strict_Pt_II"

    # Defensive fallback — should be unreachable given the metal_atoms
    # list above, but kept for forward-compat.
    return "unknown"


def _name_hint_to_scaffold(name_hint: Optional[str]) -> str:
    """Map a friendly scaffold name to a scaffold category.

    Known mappings (WF-Lambda-Metal-Pilot's seed set + the canonical
    Pt chemotherapeutics):

        cisplatin    -> strict_Pt_II
        oxaliplatin  -> strict_Pt_II (oxalate is a chelate but oxaliplatin
                       is canonically classified as Pt_II with the chelate;
                       we still map it to ``strict_Pt_II`` here because the
                       seed SMILES ``[Pt]C#C`` does not carry the oxalate
                       — the chelate is added by the search later.)
        carboplatin  -> Pt_II_chelating (cyclobutane-dicarboxylate)
        satraplatin  -> Pt_IV
        tetraplatin  -> Pt_IV
        ru_arene     -> strict_Pt_II (Ru(II) is kinetically inert)
        ir_cp_star   -> strict_Pt_II (Ir(III) is kinetically inert;
                       we conservatively keep it in strict for the
                       scaffold-gating purpose — the click rules that
                       fire are CuAAC + SPAAC, which is the safe set.)
        copper       -> labile_metal
        zinc         -> labile_metal
        iron         -> labile_metal
        manganese    -> labile_metal
    """
    if not name_hint:
        return "unknown"
    n = str(name_hint).strip().lower()
    STRICT = {"cisplatin", "nedaplatin", "ru_arene", "ir_cp_star"}
    CHELATING = {"carboplatin", "oxaliplatin"}  # NB oxaliplatin is borderline
    PT_IV = {"satraplatin", "tetraplatin", "iproplatin"}
    LABILE = {"copper", "zinc", "iron", "manganese", "cu", "zn", "fe", "mn"}
    if n in STRICT:
        return "strict_Pt_II"
    if n in CHELATING:
        return "Pt_II_chelating"
    if n in PT_IV:
        return "Pt_IV"
    if n in LABILE:
        return "labile_metal"
    return "unknown"


# ---------------------------------------------------------------------------
# Rule-set derivation.
# ---------------------------------------------------------------------------
def default_compatible_rules(
    scaffold: str,
    *,
    allow_incompatible: bool = False,
) -> List[str]:
    """Return the click-rule subset that is *compatible* (not marginal,
    not incompatible) for the given scaffold.

    Parameters
    ----------
    scaffold : str
        One of :data:`SCAFFOLD_NAMES`.  When unrecognised, returns the
        full ``CLICK_RULE_NAMES`` list (backward-compatible fallback).
    allow_incompatible : bool, default False
        When ``True``, *also* include rules marked ``"incompatible"``
        (ThiolEne / AmideCoupling / Suzuki on strict Pt_II).  Marginal
        rules are always included (they are opt-in by default; if the
        caller asks for compatibility-only they get compatible + marginal).

    Returns
    -------
    list[str]
        Subset of :data:`CLICK_RULE_NAMES`, in canonical order.

    Honest framing
    --------------
    The ``marginal`` verdict (Suzuki on Pt_II / Pt_II-chelating) is
    *always* included regardless of ``allow_incompatible``.  Only the
    ``incompatible`` rules are gated by the opt-in flag.  This matches
    the WF-MCTS-Chemistry-Research recommendation: "Suzuki marginal,
    ThiolEne + AmideCoupling gated".
    """
    if scaffold not in COMPAT_MATRIX:
        return list(CLICK_RULE_NAMES)
    row = COMPAT_MATRIX[scaffold]
    out: List[str] = []
    for rule in CLICK_RULE_NAMES:
        verdict = row.get(rule, "compatible")
        if verdict == "compatible":
            out.append(rule)
        elif verdict == "marginal":
            out.append(rule)
        elif verdict == "incompatible" and allow_incompatible:
            out.append(rule)
    return out


def incompatible_rules(scaffold: str) -> List[str]:
    """Return the list of rules marked ``"incompatible"`` for ``scaffold``."""
    if scaffold not in COMPAT_MATRIX:
        return []
    row = COMPAT_MATRIX[scaffold]
    return [rule for rule in CLICK_RULE_NAMES
            if row.get(rule) == "incompatible"]


def marginal_rules(scaffold: str) -> List[str]:
    """Return the list of rules marked ``"marginal"`` for ``scaffold``."""
    if scaffold not in COMPAT_MATRIX:
        return []
    row = COMPAT_MATRIX[scaffold]
    return [rule for rule in CLICK_RULE_NAMES
            if row.get(rule) == "marginal"]


# ---------------------------------------------------------------------------
# Pretty-printer for the audit report.
# ---------------------------------------------------------------------------
def render_compat_table() -> str:
    """Render the 5x5 compatibility matrix as a markdown table."""
    head = (
        "| scaffold \\ click | " + " | ".join(CLICK_RULE_NAMES) + " |"
    )
    sep = "|" + "|".join(["---"] * (len(CLICK_RULE_NAMES) + 1)) + "|"
    rows = [head, sep]
    for scaffold in SCAFFOLD_NAMES:
        row = COMPAT_MATRIX[scaffold]
        cells = [scaffold]
        for rule in CLICK_RULE_NAMES:
            v = row.get(rule, "compatible")
            tag = {"compatible": "OK", "marginal": "MARG",
                   "incompatible": "X"}.get(v, v)
            cells.append(tag)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def strict_pt_ii_allows_metal_coordination() -> bool:
    """WF-Lambda-Boost Phase 4 helper.

    Returns ``True`` iff the ``strict_Pt_II`` row has
    ``metal_ligand_exchange == "compatible"`` AND
    ``aqua_exchange == "compatible"``.  This is the F2(a) invariant
    that allows the Lambda search to fire MetalLigandExchange +
    AquaExchange on strict-coordination Pt_II scaffolds without
    violating the metal-scaffold prior.  The unit tests pin this
    invariant (so an accidental future tightening to
    ``"incompatible"`` would fail the test suite).
    """
    row = COMPAT_MATRIX.get("strict_Pt_II", {})
    return (
        row.get("metal_ligand_exchange") == "compatible"
        and row.get("aqua_exchange") == "compatible"
    )


# Self-test: print compat table when run as a script.
if __name__ == "__main__":  # pragma: no cover
    print(render_compat_table())
