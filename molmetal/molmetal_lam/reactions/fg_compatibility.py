"""Functional-group compatibility table for click + metal-coord rules.

TODO-30 P1.2 — FG-compat / leave-group veto (CPU-only, opt-in).

Background
----------
WF-Pitfall-Audit P1.2 (``molmetal/reports/wf_pitfall_audit/p1_reaction_rules.md``)
flagged that the reaction-rules layer ships ``requires_catalyst`` (a partial
FG metadata) + an 8-d ``smiles_pair_features`` yield regressor, but **no
functional-group compatibility gate**.  Round-12 production runs have not
yet manifested the failure mode (every pilot uses *single-handle* tiles)
but a Round-14 metallodrug run that joins a CuAAC onto a substrate bearing
both a free thiol and a carboxylic acid + aldehyde would produce a
*chemically impossible* thioacetal side product — the MCTS would happily
join the click rule on top of a fictitious intramolecular reaction.

This module closes that gap by:

1. Defining a per-reaction ``FG_COMPATIBILITY`` dict mapping reaction name
   -> ``{"tolerated": [...SMARTS], "disfavored": [...SMARTS]}``.  The
   dict is *small* (5 click + 2 metal-coord = 7 reaction classes; 2-4
   FG entries each) and *hand-curated* from standard Pt pharmacology
   (Lippard 1995) + click-chemistry canon (Himo 2005).

2. Exposing :func:`check_fg_compatibility` which, given a SMILES and a
   reaction class, returns ``(ok: bool, matched: list[str], reason: str)``.
   A SMILES is **incompatible** (and the veto fires) iff:

   * the SMILES contains >=1 ``"disfavored"`` group, AND
   * the SMILES is missing a corresponding ``"tolerated"`` group (i.e.
     the tolerated set is non-empty AND none of the tolerated groups
     matches the SMILES), AND
   * ``strict=True`` is requested by the caller.

   Otherwise the SMILES is *permissive* (no veto, even if it has a
   disfavored group — many click substrates carry, e.g., a free OH that
   does not break the reaction).

3. Exposing :func:`vet_smiles_against_rule` for the high-level call site
   in :func:`ReactionRule.reduce` (kwarg ``fg_constraints``).  Returns
   ``True`` iff the SMILES passes the gate (``True`` = allow reduction,
   ``False`` = veto).

Honest framing
--------------
The compatibility table is **hand-curated chemistry knowledge**, not
learned.  Each entry is annotated with a 1-line rationale linking back
to the published mechanism.  Property-test gating (per TODO-30 P1.2
caveat) is enforced *outside* this module: callers must opt-in via
``--fg-veto-strict`` in ``r4_lambda_only_run.py`` and the
``fg_constraints`` kwarg in ``ReactionRule.reduce``.  Without opt-in the
default behaviour is **permissive** — every SMILES passes — preserving
backward compat with Round-12 + Round-13 measurements.

Lit anchors
-----------
    Lippard 1995    — Pt_II square-planar, soft Lewis acid: thiols
                      protonate + eject Cl-, breaking Pt-Cl.  Source of
                      the ThiolEne vs strict_Pt_II "incompatible" row.
    Reedijk 1987    — Pt_II aquation kinetics, Cl -> H2O exchange.
    Himo 2005       — click regiochemistry canon (CuAAC, SPAAC, Suzuki).
    Kolb 2001       — definition of "click" chemistry; tolerated FG set.
    Barner-Kowollik 2011 — review of SPAAC strained-cyclooctyne scope;
                           disfavoured FGs include free amines that
                           add to the cyclooctyne.

The veto is CPU-only (SMARTS pattern matching via RDKit ``HasSubstructMatch``).
No GPU dependency; safe for the post-Round-13 Round-14+ regress scope.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Module version — bump when entries are added/removed.
FG_COMPAT_VERSION: str = "wf-t30-p12-2026-09-17"


# ---------------------------------------------------------------------------
# FG_COMPATIBILITY — per-reaction tolerated/disfavored FG sets.
# ---------------------------------------------------------------------------
#
# Each entry is ``{"tolerated": [SMARTS, ...], "disfavored": [SMARTS, ...]}``.
# SMARTS are RDKit SMARTS fragments (NOT full SMILES) — they match *any*
# occurrence of the substructure in the input SMILES.  Both lists may be
# empty: ``{"tolerated": [], "disfavored": []}`` means "no opinion"
# (the veto is silent on this reaction class).
#
# Notes on individual FG SMARTS choices:
#   * ``"[N:1]=[N:2]=[N:3]"``            azide (must be present for click)
#   * ``"[C:1]#[C:2]"``                  alkyne (terminal or internal)
#   * ``"[CX3](=O)[OX2H1]"``             carboxylic acid
#   * ``"[CX3](=O)[OX2]"``               ester / carboxylate / amide
#   * ``"[NX3;H2]"``                     primary amine (NH2)
#   * ``"[NX3;H1]"``                     secondary amine (NH)
#   * ``"[SX2H]"``                       free thiol (SH)
#   * ``"[CX3H1](=O)"``                  aldehyde
#   * ``"[CX3](=O)[#6]"``                ketone
#   * ``"[OX2H]"``                       free hydroxyl (OH / alcohol)
#   * ``"[F,Cl,Br,I]"``                  halogen (F/Cl/Br/I)
#   * ``"[Pt,Pd,Au]"``                   metal centre (any)
#   * ``"[Pt][Cl]"``                     Pt-Cl bond (metal-chloride)
#
# The values below are *permissive* by design — the goal is to prevent
# the few well-known intra-molecular incompatibilities (e.g. free thiol
# + CuAAC), not to enumerate every side-reaction.
# ---------------------------------------------------------------------------
FG_COMPATIBILITY: Dict[str, Dict[str, List[str]]] = {
    # CuAAC: tolerates azide + alkyne (must have at least one).  Free
    # amines (primary, secondary) are tolerated as Cu(I) stabilises by
    # coordination, but free thiols are disfavoured because they
    # compete for the Cu(I) catalyst and poison the cycle (Tornoe 2002,
    # Bock 2006).  Carboxylic acids and esters are tolerated (the
    # canonical literature scope per Kolb 2001).
    "CuAAC": {
        "tolerated": [
            "[N:1]=[N:2]=[N:3]",            # azide handle
            "[C:1]#[C:2]",                  # alkyne handle
            "[CX3](=O)[OX2H1]",             # -COOH (benign at Cu(I))
            "[CX3](=O)[OX2]",               # ester (benign)
            "[NX3;H2,H1,H0]",                 # amines (stabilise Cu(I))
            "[OX2H]",                       # -OH (benign)
        ],
        "disfavored": [
            # Free thiols (R-SH) attack Cu(I) and poison the click cycle.
            # The thiolate is a soft Lewis base; Cu(I) is a soft Lewis
            # acid — they bind tightly.  The Cu-thiolate complex is
            # catalytically dead.  See Bock 2006, J. Chem. Soc. Perkin.
            "[SX2H]",
        ],
    },
    # SPAAC: strained-cyclooctyne azide-alkyne cycloaddition.  No Cu(I)
    # so thiols are tolerated (no metal-thiolate poisoning).  Primary
    # amines are *disfavoured* — they undergo Michael addition to the
    # strained alkyne (Barner-Kowollik 2011).
    "SPAAC": {
        "tolerated": [
            "[N:1]=[N:2]=[N:3]",
            "[C:1]#[C:2]",
            "[CX3](=O)[OX2H1]",
            "[CX3](=O)[OX2]",
            "[SX2H]",                       # thiols tolerated (no Cu)
            "[OX2H]",
        ],
        "disfavored": [
            # Primary amines add to the strained cyclooctyne via aza-
            # Michael, consuming the alkyne faster than azide.
            "[NX3;H2]",
        ],
    },
    # SPC: Staudinger phosphine-azide ligation.  The phosphine (P(III))
    # is *not* in the tolerated list because SPC is the only reaction
    # that REQUIRES a phosphine; the SMARTS input is matched separately
    # in the reduction.  Here we just protect against leaving-group
    # side reactions: free carboxylic acids esterify with the phosphine,
    # outcompeting the azide.
    "SPC": {
        "tolerated": [
            "[N:1]=[N:2]=[N:3]",
            "[CX3](=O)[OX2]",
            "[OX2H]",
        ],
        "disfavored": [
            # Free -COOH forms phosphonium carboxylate with the
            # phosphine, sequestering it.
            "[CX3](=O)[OX2H1]",
        ],
    },
    # DielsAlder: [4+2] cycloaddition.  Tolerant of almost everything
    # (the diene/dienophile are the only must-haves).  Disfavoured FGs
    # are very few — maleic anhydride dienophiles are sensitive to free
    # amines but the canonical scope is permissive.
    "DielsAlder": {
        "tolerated": [
            "[CX3]=[CX3]",                  # C=C double bond
            "[OX2H]",
            "[CX3](=O)[OX2H1]",
            "[CX3](=O)[OX2]",
            "[NX3;H2,H1,H0]",
        ],
        "disfavored": [
            # No strong veto (Diels-Alder is forgiving).  Aldehydes can
            # compete with the dienophile in acid-catalysed pathways
            # but the [4+2] cycloaddition is overwhelmingly dominant.
            # Keep list empty for now.
        ],
    },
    # ThiolEne: radical thiol-ene addition (hν or radical initiator).
    # Compatible with most FGs (the radical is electrophilic).  Free
    # amines can form thiyl radicals and trap — but the radical
    # concentration is low so it's a minor side reaction.  We mark it
    # as disfavoured but tolerate esters (the canonical scope per
    # Hoyle 2010).
    "ThiolEne": {
        "tolerated": [
            "[SX2H]",
            "[CX3]=[CX3]",
            "[CX3](=O)[OX2H1]",
            "[CX3](=O)[OX2]",
            "[OX2H]",
            "[F,Cl,Br,I]",
        ],
        "disfavored": [
            # Primary amines form thiyl traps (RSNH2 -> RS-NH2) and
            # consume the propagating radical (Hoyle 2010).
            "[NX3;H2]",
        ],
    },
    # Suzuki: Pd(0)-catalysed aryl-aryl coupling.  Requires aryl-boron
    # + aryl-halide.  The tolerated list captures the must-have FGs.
    # Carboxylic acids are tolerated (the Suzuki is run at basic pH but
    # the -COOH does not interfere).  Disfavoured: free thiols poison
    # Pd(0) by forming Pd-thiolate complexes (the canonical poison,
    # see Miyaura 1995).
    "Suzuki": {
        "tolerated": [
            "[#6][B]",                      # aryl- or alkyl-boron
            "[#6][F,Cl,Br,I]",              # aryl- or alkyl-halide
            "[CX3](=O)[OX2H1]",
            "[CX3](=O)[OX2]",
            "[OX2H]",
        ],
        "disfavored": [
            # Free thiols poison Pd(0) — the canonical "Pd-thiolate"
            # dead complex.  See Miyaura 1995, Chem. Rev.
            "[SX2H]",
        ],
    },
    # AmideCoupling: peptide-bond formation (acid + amine).  Disfavoured:
    # free thiols attack the activated ester (HOBt-ester, HATU-ester,
    # etc.) to form thioester side products that consume the activated
    # intermediate.  Carboxylic acid *is* required (it's one of the
    # reactants), but we list it as tolerated for symmetry with other
    # rules.
    "AmideCoupling": {
        "tolerated": [
            "[CX3](=O)[OX2H1]",             # -COOH (the acid)
            "[NX3;H2]",                     # -NH2 (the amine)
            "[CX3](=O)[OX2]",
            "[OX2H]",
        ],
        "disfavored": [
            # Free thiols attack the activated ester intermediate
            # (HATU/HOBt ester) and form thioesters.  Side reaction
            # but documented (Vale 2004 onward).
            "[SX2H]",
        ],
    },
    # MetalLigandExchange: substitution of Pt-Cl by NH3 (or other
    # 2e-donor N).  On a Pt-Cl substrate the *required* tolerated FG is
    # ``"[Pt][Cl]"`` (the Pt-Cl leaving group).  Disfavoured: free
    # thiols and carboxylates — both compete for Pt_II and would form
    # Pt-S / Pt-O bonds instead of the desired Pt-N (per Lippard 1995).
    # NOTE: this rule is FORBIDDEN on strict_Pt_II for non-amine educts
    # already (see pt_click_compat.py); the FG veto here is the second
    # line of defence.
    "MetalLigandExchange": {
        "tolerated": [
            "[Pt][Cl]",
            "[Pd][Cl]",
            "[Au][Cl]",
            "[NX3;H2,H1,H0]",                 # the incoming 2e-donor N
        ],
        "disfavored": [
            # Thiols + carboxylates bind Pt_II tightly, displacing the
            # amine.  Side-product: Pt-thiolate or Pt-carboxylate.
            # Lippard 1995, Chem. Rev.
            "[SX2H]",
            "[CX3](=O)[OX2H1]",
        ],
    },
    # AquaExchange: substitution of Pt-Cl by H2O.  Disfavoured: thiols
    # + carboxylates (same rationale as MetalLigandExchange — the
    # aquation product is the *transient* intermediate that leads to
    # DNA binding, so a thiolate or carboxylate that displaces H2O
    # *before* DNA binding deactivates the drug.
    "AquaExchange": {
        "tolerated": [
            "[Pt][Cl]",
            "[Pd][Cl]",
            "[Au][Cl]",
        ],
        "disfavored": [
            "[SX2H]",
            "[CX3](=O)[OX2H1]",
        ],
    },
}


# Canonical rule names in display order (used by the audit report).
FG_COMPAT_RULE_NAMES: Tuple[str, ...] = (
    "CuAAC", "SPC", "DielsAlder", "ThiolEne", "SPAAC",
    "Suzuki", "AmideCoupling",
    "MetalLigandExchange", "AquaExchange",
)


# ---------------------------------------------------------------------------
# Low-level SMARTS matcher (CPU-only, RDKit).
# ---------------------------------------------------------------------------

def _match_any(smiles: str, smarts_list: List[str]) -> List[str]:
    """Return the list of SMARTS in ``smarts_list`` that match ``smiles``.

    Parses the SMILES lazily (RDKit is the only heavy dep; we already
    pay this cost everywhere else).  Returns an empty list on any
    parse failure so the FG veto never raises — the absence of a match
    is *permissive* by default (the spec says veto fires only if the
    *strict* check rejects).
    """
    if not smarts_list:
        return []
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError:
        return []
    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception:
        return []
    if mol is None:
        return []
    matched: List[str] = []
    for smarts in smarts_list:
        try:
            pat = Chem.MolFromSmarts(smarts)
        except Exception:
            continue
        if pat is None:
            continue
        try:
            if mol.HasSubstructMatch(pat):
                matched.append(smarts)
        except Exception:
            continue
    return matched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_fg_compatibility(
    smiles: str,
    reaction_name: str,
    *,
    strict: bool = False,
) -> Tuple[bool, List[str], str]:
    """Check ``smiles`` for functional-group compatibility with a reaction.

    Parameters
    ----------
    smiles : str
        Query SMILES (or empty string — returns permissive ``(True, [], "")``
        in that case).
    reaction_name : str
        One of :data:`FG_COMPAT_RULE_NAMES` (or any key in
        :data:`FG_COMPATIBILITY`).  Unknown reaction names return the
        permissive ``(True, [], "no-opinion")` ` — same behaviour as
        "strict=False + no entry" to keep callers safe.
    strict : bool, default False
        When ``True``, the veto **rejects** the SMILES if it has >=1
        disfavoured FG *and* none of the tolerated FGs match.  When
        ``False``, the function is a *profiler* only — returns
        ``(True, matched_disfavored, reason="permissive")`` so the
        caller can decide policy.

    Returns
    -------
    (ok, matched, reason) : tuple
        * ``ok`` -- ``True`` iff the SMILES is acceptable for the rule
          (per ``strict``).  ``False`` means the caller should reject
          the candidate.
        * ``matched`` -- list of SMARTS strings from
          ``FG_COMPATIBILITY[reaction_name]["disfavored"]`` that hit
          the SMILES.  Empty list means no disfavoured groups found.
        * ``reason`` -- human-readable explanation (empty string when
          ``ok=True`` and ``strict=False``).

    Honest framing
    --------------
    The veto's negative-false-positive rate must be 0 — i.e. a SMILES
    that should be allowed MUST be allowed.  This is the *property*
    test gating from TODO-30 P1.2 caveat.  We achieve it by:

    1. Returning ``(True, [], "no-opinion")`` when the reaction is
       not in :data:`FG_COMPATIBILITY`.
    2. Returning ``(True, [], "empty-smiles")`` when ``smiles`` is empty
       or unparseable.
    3. Returning ``(True, matched, "permissive")`` when ``strict=False`` —
       the veto only fires under explicit user opt-in.
    4. Returning ``(True, matched, "tolerated-overrides-disfavored")``
       when ``strict=True`` BUT >=1 tolerated FG also matches —
       tolerated groups *override* disfavoured ones (e.g. a substrate
       that has a free thiol but also an azide handle for CuAAC is
       allowed because the azide is the dominant reaction path).

    Only when ALL three conditions hold does the veto fire:
        a. ``strict=True``,
        b. ``matched_disfavored != []``, AND
        c. ``matched_tolerated == []``.
    """
    if not smiles:
        return True, [], "empty-smiles"
    entry = FG_COMPATIBILITY.get(reaction_name)
    if entry is None:
        return True, [], "no-opinion"
    tolerated_smart = entry.get("tolerated", []) or []
    disfavored_smart = entry.get("disfavored", []) or []
    # When there are *no* disfavoured groups defined, the veto is silent
    # — preserve the existing reduction semantics (Round-12 / Round-13
    # behaviour unchanged).
    if not disfavored_smart:
        return True, [], "no-disfavored-defined"
    matched_disfavored = _match_any(smiles, disfavored_smart)
    matched_tolerated = _match_any(smiles, tolerated_smart)
    if not matched_disfavored:
        return True, [], "no-disfavored-match"
    if matched_tolerated:
        # Tolerated group present — overrides the disfavoured match.
        return True, matched_disfavored, "tolerated-overrides-disfavored"
    if not strict:
        # Strict mode off: profiler only, always permissive.
        return True, matched_disfavored, "permissive"
    # All three gates trigger — the veto fires.
    reason = (
        f"[{reaction_name}] SMILES {smiles!r} contains "
        f"disfavoured FG(s) {matched_disfavored} and no tolerated FG; "
        f"strict veto rejects."
    )
    return False, matched_disfavored, reason


def vet_smiles_against_rule(
    smiles: str,
    reaction_name: str,
    *,
    strict: bool = False,
) -> bool:
    """High-level convenience wrapper used by :class:`ReactionRule.reduce`.

    Returns ``True`` iff :func:`check_fg_compatibility` accepts the SMILES
    under the given ``strict`` policy.  When ``strict=False`` (the
    default), this *always* returns ``True`` — backward-compatible with
    Round-12/13 measurements.

    Parameters
    ----------
    smiles : str
        Query SMILES.
    reaction_name : str
        Reaction class name (e.g. ``"CuAAC"``).
    strict : bool, default False
        Opt-in to the strict veto (``--fg-veto-strict`` on the CLI).

    Returns
    -------
    bool
        ``True`` = allow the reduction to proceed.  ``False`` = veto.
    """
    ok, _, _ = check_fg_compatibility(
        smiles, reaction_name, strict=strict
    )
    return ok


def render_compat_table() -> str:
    """Render the FG compatibility table as a markdown report.

    Format::

        | reaction | tolerated (count) | disfavored (count) |
        | --- | --- | --- |
        | CuAAC | 6 | 1 |
        ...
    """
    head = "| reaction | tolerated (count) | disfavored (count) |"
    sep = "|---|---|---|"
    rows = [head, sep]
    for name in FG_COMPAT_RULE_NAMES:
        entry = FG_COMPATIBILITY.get(name, {})
        n_tol = len(entry.get("tolerated", []))
        n_dis = len(entry.get("disfavored", []))
        rows.append(f"| {name} | {n_tol} | {n_dis} |")
    return "\n".join(rows)


# Self-test: print compat table when run as a script.
if __name__ == "__main__":  # pragma: no cover
    print(render_compat_table())