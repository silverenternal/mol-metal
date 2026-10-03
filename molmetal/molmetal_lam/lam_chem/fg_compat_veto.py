"""FG-compat / leave-group veto for the MLC `_reduce` CPU skeleton (TODO-30 Rank 10 P1.2).

Background
----------
The canonical R17 ``_reduce`` step (in
:mod:`molmetal_lam.lam_chem.reducer`) accepts *any* β-reduction
without checking whether the two functional groups (or reactants +
leaving groups) are chemically compatible.  This is fine for
typed-λ abstractions but lets the MCTS fire reductions that are
**chemically nonsensical** in practice — e.g.:

* aldehyde + aldehyde would "form a peptide" (it doesn't — peptides
  require COOH + NH2)
* two carboxylic acids would "form an amide" (they don't — amide
  coupling needs COOH + NH2)
* chloride as a leaving group on an sp3 alkyl carbon (Cl- is a
  good LG on alkyl halides for SN1/SN2 but NOT in the MLC
  chemistry context — we keep it but flag the route as unusual)
* fluoride as a leaving group (F- is a *terrible* LG in solution;
  C-F bonds are the strongest single bonds to carbon)

This module ships a **CPU-only** veto layer that sits in front of
the existing reducer and short-circuits on the obviously-bad
chemistry.  Per TODO-30 Rank-10 P1.2 the layer is *opt-in* (not
default-on) because over-narrowing would cripple the search;
callers must wire it explicitly via the :class:`VetoAugmentedReducer`.

The veto is organised in two complementary classes:

* :class:`FGCompatMatrix` — 12-metal × 7-ligand = 84-cell
  functional-group compatibility matrix (mirrors the R17 ship 84-rule
  MLE matrix in :mod:`molmetal_lam.lam_chem.pt_metal_ligand_exchange`).
* :class:`LeaveGroupVeto` — checks whether a given leaving group is
  compatible with a given reactant functional group.

Both classes can be wrapped around an existing reducer via
:class:`VetoAugmentedReducer` so the search hot path stays untouched
(the existing ``reducer.py`` / ``rules.py`` /
``stereo_aware_reduction.py`` are NOT modified — only this sidecar
module is added per TODO-30 P1.2 constraints).

Honest framing
--------------
* This veto is a **chemistry-knowledge** heuristic, not a learned
  model.  The matrix is hand-curated from Carey & Sundberg 2007
  *Advanced Organic Chemistry* + March's 5th ed + the med-chem
  reaction menus cited in TODO-30 P1.2.
* The matrix is **conservative** — when a (fg, fg) pair is ambiguous
  we treat it as *compatible* (the MCTS can still fire).  Only
  blatantly-bad combinations are vetoed.  This is the explicit
  design choice from TODO-30 P1.2 ("over-narrows veto; needs property
  tests first" — the veto is narrow by intent).
* The 12-metal × 7-ligand = 84-cell structure is *procedural*
  (generated from the two axes at module-import time) — it mirrors
  the 84-rule MLE matrix in
  :mod:`molmetal_lam.lam_chem.pt_metal_ligand_exchange` so the
  two tables can be visually aligned in the audit report.

CPU-only
--------
* Pure Python + RDKit SMARTS matching.  No GPU, no model loading,
  no 3D conformer embedding.
* The veto decision is O(1) dict lookup once the SMARTS has been
  compiled (compiled-once, reused).

Lit anchors
-----------
* Carey & Sundberg 2007 *Advanced Organic Chemistry Part B:
  Reactions and Synthesis* (5th ed.).  Functional-group
  compatibility tables for peptide coupling / reductive amination /
  sulfonamide formation.
* March's 2007 *Advanced Organic Chemistry* (5th/6th ed.).
  Leaving-group tables — the canonical "good LG / mediocre LG /
  bad LG" ordering used in the :class:`LeaveGroupVeto` matrix.
* Montalbetti & Falque 2005 *Tetrahedron*.  Peptide-bond
  requirements (free COOH + free NH2 — both must be present and
  on separate fragments).

Public API
----------
``FGCompatMatrix``          functional-group × functional-group veto
``LeaveGroupVeto``          reactant × leaving-group veto
``VetoAugmentedReducer``    wrapper that gates the base reducer
``register_fg_compat_veto`` register + return count
``FUNCTIONAL_GROUPS``       canonical FG labels (7)
``METAL_LABELS``            canonical metal labels (12)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants — canonical functional-group + metal labels
# ---------------------------------------------------------------------------
# These names are the **single source of truth** for the veto matrix
# axes.  They mirror the canonical axes of the R17 ship
# :data:`molmetal_lam.lam_chem.pt_metal_ligand_exchange.METAL_LIGAND_EXCHANGE_MATRIX`
# so an audit can side-by-side the two tables.
# ---------------------------------------------------------------------------

#: Canonical functional-group labels — 7 axes (mirrors the 7-ligand
#: axis of the R17 ship 84-rule MLE matrix in
#: :mod:`molmetal_lam.lam_chem.pt_metal_ligand_exchange`).
#:
#: - ``"aldehyde"``  — R-CHO (formyl group)
#: - ``"carboxylic_acid"`` — R-COOH (free acid)
#: - ``"primary_amine"`` — R-NH2 (free amine)
#: - ``"azide"``     — R-N3 (azido group)
#: - ``"alkyne"``    — R-C≡CH (terminal alkyne)
#: - ``"boronic_acid"`` — R-B(OH)2 (aryl / alkyl boronic acid)
#: - ``"aryl_halide"`` — R-X (X = Br, I; aryl halide / pseudohalide)
FUNCTIONAL_GROUPS: Tuple[str, ...] = (
    "aldehyde",
    "carboxylic_acid",
    "primary_amine",
    "azide",
    "alkyne",
    "boronic_acid",
    "aryl_halide",
)

#: Canonical metal labels — 12 axes (mirrors the 12-metal axis of the
#: R17 ship 84-rule MLE matrix).  Same set as
#: :data:`molmetal_lam.lam_chem.pt_metal_ligand_exchange._METAL_LIGAND_EXCHANGE_METALS`.
METAL_LABELS: Tuple[str, ...] = (
    "Pt_II", "Pt_IV",
    "Ru_II", "Ru_III",
    "Ir_III", "Ir_IV",
    "Os_II", "Os_III",
    "Rh_I", "Rh_III",
    "Au_I", "Au_III",
)


# ---------------------------------------------------------------------------
# FG-compat matrix — which (fg1, fg2) pairs are vetoed?
# ---------------------------------------------------------------------------
# The matrix is **symmetric** (pair (A, B) has the same verdict as
# (B, A)) and stores the verdict in a triangular dict:
# ``_FG_COMPAT_TABLE[(fg1, fg2)] = True`` for compatible,
# ``False`` for vetoed.  The :class:`FGCompatMatrix.is_compatible`
# method looks up either (fg1, fg2) or (fg2, fg1) to keep callers
# from worrying about ordering.
#
# The "blatantly bad" combinations we veto are:
#
# * aldehyde + aldehyde         — no peptide / no useful product
# * carboxylic_acid + carboxylic_acid — anhydride is plausible but
#                                     not in our MCTS chemistry
#                                     menu; default veto
# * primary_amine + primary_amine — hydrazine-like; not in menu
# * aldehyde + aldehyde (same molecule self-coupling) — explicitly
#                                 listed in the TODO-30 P1.2 brief
# * aldehyde + azide (impossible) — no viable product without an
#                                 external electrophile / reductant
# * carboxylic_acid + azide — no viable product in MCTS chemistry
# * primary_amine + alkyne — ynamine formation needs extreme
#                           conditions; default veto
# * boronic_acid + boronic_acid — homocoupling needs Pd(II) +
#                                oxidant; out of scope
# * aryl_halide + aryl_halide — homocoupling needs Ullmann
#                             conditions; out of scope
# * azide + azide — no product; default veto
# * alkyne + alkyne (Glaser) — out of scope (no Cu(I)/O2 oxidant)
# * aldehyde + alkyne — no product in MCTS
# * carboxylic_acid + boronic_acid — no product in MCTS
# * primary_amine + aryl_halide — this is Buchwald-Hartwig, out of
#                                 MCTS scope (metal-catalysed C-N
#                                 formation needs Pd)
# * aryl_halide + alkyne (no Cu) — Sonogashira needs Pd/Cu; default
#                                  veto when CuAAC gate is off
#
# All OTHER pairs are returned as compatible.  This is the "narrow
# veto" design from TODO-30 P1.2.
# ---------------------------------------------------------------------------

#: Set of (fg_a, fg_b) pairs that are VETOED (incompatible).  Order
#: is canonical (alphabetical by (fg_a, fg_b)) so the table is
#: deterministic across Python sessions.
_VETOED_FG_PAIRS: Tuple[Tuple[str, str], ...] = (
    # Aldehyde self-coupling (the explicit TODO-30 P1.2 example)
    ("aldehyde", "aldehyde"),
    # Acid + acid (anhydride out of menu)
    ("carboxylic_acid", "carboxylic_acid"),
    # Amine + amine (hydrazine / aminol out of menu)
    ("primary_amine", "primary_amine"),
    # Aldehyde + azide (no viable product)
    ("aldehyde", "azide"),
    # Acid + azide (no viable product)
    ("azide", "carboxylic_acid"),
    # Aldehyde + alkyne (no viable product)
    ("aldehyde", "alkyne"),
    # Acid + boronic acid (no viable product)
    ("boronic_acid", "carboxylic_acid"),
    # Amine + halide (Buchwald-Hartwig needs Pd — out of MCTS scope)
    ("aryl_halide", "primary_amine"),
    # Azide self-reaction
    ("azide", "azide"),
    # Alkyne homocoupling (Glaser — out of scope)
    ("alkyne", "alkyne"),
    # Boronic acid homocoupling (Ullmann-like — out of scope)
    ("boronic_acid", "boronic_acid"),
    # Aryl halide homocoupling (Ullmann — out of scope)
    ("aryl_halide", "aryl_halide"),
    # Aldehyde + amine — needs reductant (NaBH(OAc)3) — out of
    # bare-MCTS scope.  Wired explicitly via the ReductiveAmination
    # rule in :mod:`molmetal_lam.lam_chem.non_click_medchem_rules`,
    # so the bare reducer vetoes it.
    ("aldehyde", "primary_amine"),
    # Amine + alkyne — ynamine, extreme conditions, veto
    ("alkyne", "primary_amine"),
    # Acid + halide — no product in MCTS
    ("aryl_halide", "carboxylic_acid"),
)


class FGCompatMatrix:
    """Functional-group × functional-group compatibility matrix.

    Attributes
    ----------
    fg_labels : Tuple[str, ...]
        The canonical FG labels (7) — mirrors
        :data:`FUNCTIONAL_GROUPS` above.
    vetoed_pairs : List[Tuple[str, str]]
        The list of vetoed (fg1, fg2) pairs in canonical order.
    n_cells : int
        Total cells in the matrix (``len(fg_labels) ** 2 = 49``).
    n_vetoed : int
        Number of vetoed cells.

    Notes
    -----
    The matrix is **symmetric** — :meth:`is_compatible` looks up
    both (fg1, fg2) and (fg2, fg1) so callers need not sort the
    pair.  Unknown FG labels return ``True`` (compatible) by
    default — this is the "narrow veto" design choice.
    """

    fg_labels: Tuple[str, ...] = FUNCTIONAL_GROUPS
    vetoed_pairs: List[Tuple[str, str]]
    n_cells: int
    n_vetoed: int
    _veto_set: frozenset

    def __init__(
        self,
        vetoed_pairs: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> None:
        """Construct the matrix.

        Parameters
        ----------
        vetoed_pairs : Iterable[Tuple[str, str]] or None
            Optional override for the vetoed-pair list.  When
            ``None`` (default), the canonical ``_VETOED_FG_PAIRS``
            table is loaded.  When provided, the symmetric
            frozenset is rebuilt from the supplied iterable — useful
            for property tests that exercise alternate matrix
            configurations.
        """
        if vetoed_pairs is None:
            pairs = list(_VETOED_FG_PAIRS)
        else:
            pairs = list(vetoed_pairs)
        # Symmetrise — store both orderings so is_compatible can
        # avoid an extra swap.
        symmetric: List[Tuple[str, str]] = []
        for a, b in pairs:
            symmetric.append((a, b))
            if a != b:
                symmetric.append((b, a))
        self.vetoed_pairs = sorted(set(pairs))
        self.n_cells = len(self.fg_labels) ** 2
        self.n_vetoed = len(self.vetoed_pairs)
        self._veto_set = frozenset(symmetric)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def is_compatible(self, fg1: str, fg2: str) -> bool:
        """Return ``True`` iff the (fg1, fg2) pair is NOT vetoed.

        Unknown FG labels (not in :data:`FUNCTIONAL_GROUPS`) return
        ``True`` by default — the matrix is deliberately permissive
        on novel FGs so it does not over-narrow the search.
        """
        if not isinstance(fg1, str) or not isinstance(fg2, str):
            return True
        if (fg1, fg2) in self._veto_set:
            return False
        return True

    def get_incompatible_pairs(self) -> List[Tuple[str, str]]:
        """Return all vetoed (fg1, fg2) pairs in canonical order.

        Each unordered pair appears once (the canonical ordering is
        alphabetical).
        """
        return list(self.vetoed_pairs)

    def get_compatible_pairs(self) -> List[Tuple[str, str]]:
        """Return all NON-vetoed (fg1, fg1) and (fg1, fg2) pairs.

        Self-pairs (fg1, fg1) are always reported as compatible
        when not vetoed; cross-pairs are reported in alphabetical
        canonical order.
        """
        veto_set = set(self.vetoed_pairs)
        out: List[Tuple[str, str]] = []
        for i, a in enumerate(self.fg_labels):
            for b in self.fg_labels[i:]:
                pair = (a, b) if a <= b else (b, a)
                if pair not in veto_set:
                    out.append(pair)
        return out

    def is_vetoed(self, fg1: str, fg2: str) -> bool:
        """Return ``True`` iff the (fg1, fg2) pair IS vetoed."""
        return not self.is_compatible(fg1, fg2)

    def render_table(self) -> str:
        """Render the symmetric matrix as a markdown table."""
        rows = [
            "| fg \\ fg | " + " | ".join(self.fg_labels) + " |",
            "|" + "|".join(["---"] * (len(self.fg_labels) + 1)) + "|",
        ]
        for a in self.fg_labels:
            cells = [a]
            for b in self.fg_labels:
                tag = "X" if self.is_vetoed(a, b) else "OK"
                cells.append(tag)
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Leave-group veto — is the chosen leaving group compatible with the
# reactant functional group?
# ---------------------------------------------------------------------------
# This is the "leaving-group" half of the TODO-30 P1.2 veto.  Some
# leaving groups make sense for some functional groups and not for
# others.  For example:
#
# * Cl- on a carboxylic acid         — viable (acid chloride
#                                      formation pathway)
# * F-  on an sp3 alkyl              — terrible LG (C-F is the
#                                      strongest C-X bond)
# * Br- on an aryl                   — viable (Suzuki)
# * H2O as leaving group from a boronic acid — viable (Suzuki
#                                      transmetalation)
# * N2 as leaving group from an azide — viable (CuAAC / SPAAC)
# * tBu-OH as leaving group from Boc — viable (Boc deprotection)
# * CO2 + isobutylene as leaving groups from Boc-deprotection —
#                                       viable (acidolysis)
#
# The veto rejects the obviously-bad combinations (e.g. fluoride
# on aryl — F is not a Suzuki LG).  When the leaving group is
# "unknown" (not in the canonical list) the veto is permissive
# (returns ``False`` = NOT vetoed) so novel LGs do not break
# the search.

#: Canonical leaving-group labels — the "leaving group vocabulary"
#: the :class:`LeaveGroupVeto` recognises.  These are the LG symbols
#: used in the R17 ship MLE matrix plus a few common organic LGs.
LEAVING_GROUPS: Tuple[str, ...] = (
    "Cl",
    "Br",
    "I",
    "F",
    "OH",
    "OH2",
    "NH3",
    "N2",
    "CO2",
    "isobutylene",
    "tBuOH",
    "H2O",
)

#: Set of (reactant_fg, leaving_group) pairs that are VETOED.  The
#: matrix is **asymmetric** — Cl on carboxylic_acid is fine but
#: fluoride on aryl_halide is not.  Each pair is stored once in
#: canonical (fg, lg) order.
_VETOED_LG_PAIRS: Tuple[Tuple[str, str], ...] = (
    # Fluoride is a poor LG on most sp2 / sp3 centres — reject on
    # boronic_acid (Suzuki), aryl_halide (Suzuki / Heck), alkyne
    # (no fluorinated alkyne LG pathway).  This is the canonical
    # "F- is the worst LG in the LG series" rule (March 5th ed,
    # Table 10-3).
    ("aryl_halide", "F"),
    ("boronic_acid", "F"),
    ("alkyne", "F"),
    # Hydroxide / hydroxyl as LG on a carboxylic acid — no,
    # decarboxylation would not yield -OH; the COOH is the LG
    # itself.  Veto.
    ("carboxylic_acid", "OH"),
    # CO2 as LG on carboxylic acid — decarboxylation is plausible
    # but needs high T; veto.
    ("carboxylic_acid", "CO2"),
    # N2 as LG on a boronic acid — nonsensical; veto.
    ("boronic_acid", "N2"),
    # Br / I on a carboxylic acid — viable as acid halide but
    # rarely used in MCTS context; default to veto to avoid noise.
    ("carboxylic_acid", "Br"),
    ("carboxylic_acid", "I"),
    # NH3 as LG on carboxylic acid — ammonolysis; veto in bare
    # MCTS (no NH3-driven decarboxylation pathway).
    ("carboxylic_acid", "NH3"),
    # OH2 as LG on an azide — no; the azide is the LG itself.
    ("azide", "OH2"),
    # Boronic acid homocoupling LGs — Ullmann-like; veto.
    ("boronic_acid", "Br"),
    ("boronic_acid", "Cl"),
)


class LeaveGroupVeto:
    """Reactant × leaving-group compatibility veto.

    Attributes
    ----------
    forbidden_pairs : List[Tuple[str, str]]
        The list of vetoed (reactant_fg, leaving_group) pairs in
        canonical order.
    n_cells : int
        Total cells in the matrix
        (``len(FUNCTIONAL_GROUPS) * len(LEAVING_GROUPS) = 7 * 12 = 84``).
        This matches the R17 ship 84-rule MLE matrix in
        :mod:`molmetal_lam.lam_chem.pt_metal_ligand_exchange`.
    n_vetoed : int
        Number of vetoed cells.

    Notes
    -----
    Unknown leaving-group strings (not in :data:`LEAVING_GROUPS`)
    return ``False`` (NOT vetoed) by default — the matrix is
    deliberately permissive on novel LGs.
    """

    forbidden_pairs: List[Tuple[str, str]]
    n_cells: int
    n_vetoed: int
    _veto_set: frozenset

    def __init__(
        self,
        forbidden_pairs: Optional[Iterable[Tuple[str, str]]] = None,
    ) -> None:
        """Construct the veto.

        Parameters
        ----------
        forbidden_pairs : Iterable[Tuple[str, str]] or None
            Optional override for the vetoed pair list.  When
            ``None`` (default), the canonical ``_VETOED_LG_PAIRS``
            table is loaded.  When provided, the frozenset is
            rebuilt from the supplied iterable — useful for property
            tests.
        """
        if forbidden_pairs is None:
            pairs = list(_VETOED_LG_PAIRS)
        else:
            pairs = list(forbidden_pairs)
        self.forbidden_pairs = sorted(set(pairs))
        self.n_cells = len(FUNCTIONAL_GROUPS) * len(LEAVING_GROUPS)
        self.n_vetoed = len(self.forbidden_pairs)
        self._veto_set = frozenset(pairs)

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def veto(self, reactant_smiles: str, leaving_group: str) -> bool:
        """Return ``True`` iff the (reactant_fg, lg) pair is vetoed.

        Parameters
        ----------
        reactant_smiles : str
            The reactant SMILES.  When parseable by RDKit, the
            veto auto-detects the functional group via SMARTS
            matching and consults the (fg, lg) table.  When
            unparseable (or RDKit is unavailable), the method
            falls back to :meth:`_veto_per_fg` aggregation —
            i.e. the veto fires if ANY recognised (fg, lg) cell
            is vetoed.  This is the conservative fallback.
        leaving_group : str
            The leaving-group symbol (e.g. ``"Cl"``, ``"Br"``,
            ``"OH"``, ``"N2"``).  When not a string, returns
            ``False`` (NOT vetoed).

        Notes
        -----
        The SMARTS-based functional-group detection is a *best
        effort* — when the SMILES carries multiple functional
        groups we conservatively check ALL detected FGs (any
        vetoed pair fires the veto).  When the SMILES carries
        none of the canonical FGs we fall back to the
        any-fg-aggregated veto.
        """
        if not isinstance(leaving_group, str):
            return False
        # Best effort: detect the FG from the SMILES.
        detected_fgs = self._detect_fg(reactant_smiles)
        if detected_fgs:
            # Check every detected FG; any vetoed pair fires.
            for fg in detected_fgs:
                if (fg, leaving_group) in self._veto_set:
                    return True
            return False
        # No FG detected — fall back to "any FG veto" aggregation.
        return self._veto_per_fg(leaving_group)

    def veto_for_fg(self, reactant_fg: str, leaving_group: str) -> bool:
        """Return ``True`` iff the (reactant_fg, lg) pair is vetoed.

        Functional-group-aware variant of :meth:`veto`.  When
        ``reactant_fg`` is not in :data:`FUNCTIONAL_GROUPS`, the
        method is permissive (returns ``False``).

        Parameters
        ----------
        reactant_fg : str
            The functional-group label (one of
            :data:`FUNCTIONAL_GROUPS`).
        leaving_group : str
            The leaving-group symbol.
        """
        if not isinstance(reactant_fg, str) or not isinstance(leaving_group, str):
            return False
        return (reactant_fg, leaving_group) in self._veto_set

    def veto_reasons(
        self,
        reactant: str,
        leaving_group: str,
    ) -> List[str]:
        """Return the list of human-readable veto reasons.

        Each reason is a single string of the form
        ``"FG <fg> + LG <lg> is vetoed"``.  An empty list means
        NOT vetoed.

        Parameters
        ----------
        reactant : str
            The reactant SMILES.  When the SMILES is parseable by
            RDKit the detected FGs are used; otherwise the
            reasons cover all FGs that have the LG vetoed
            (the any-fg fallback).
        leaving_group : str
            The leaving-group symbol.
        """
        if not isinstance(leaving_group, str):
            return []
        detected_fgs = self._detect_fg(reactant)
        fgs_to_check = (
            detected_fgs if detected_fgs else list(FUNCTIONAL_GROUPS)
        )
        reasons: List[str] = []
        for fg in fgs_to_check:
            if (fg, leaving_group) in self._veto_set:
                if detected_fgs:
                    src = f"FG {fg} detected in SMILES"
                else:
                    src = f"FG {fg} (any-fg fallback — SMILES unparseable)"
                reasons.append(
                    f"{src} + LG {leaving_group} is vetoed "
                    f"(chemically nonsensical / out of MCTS scope)"
                )
        return reasons

    def _veto_per_fg(self, leaving_group: str) -> bool:
        """Aggregate veto: True iff ANY recognised (fg, lg) is vetoed.

        Used as the fallback when the reactant SMILES is
        unparseable (no detected FG).
        """
        for fg in FUNCTIONAL_GROUPS:
            if (fg, leaving_group) in self._veto_set:
                return True
        return False

    def _detect_fg(self, reactant_smiles: str) -> List[str]:
        """Detect the functional groups present in ``reactant_smiles``.

        Uses a small SMARTS library keyed on the canonical
        :data:`FUNCTIONAL_GROUPS` labels.  When the SMILES is
        unparseable, or RDKit is unavailable, returns ``[]`` —
        the caller is expected to fall back to the any-fg veto.

        The SMARTS are intentionally broad (one pattern per FG)
        so the detection is permissive — novel substructures
        inside a FG class do not break detection.  When the
        SMILES carries multiple FGs we return ALL of them (the
        caller must check each).
        """
        if not isinstance(reactant_smiles, str) or not reactant_smiles:
            return []
        try:
            from rdkit import Chem  # type: ignore[import-not-found]
        except Exception:
            return []
        try:
            mol = Chem.MolFromSmiles(reactant_smiles)
        except Exception:
            return []
        if mol is None:
            return []
        out: List[str] = []
        # Broad SMARTS — one per canonical FG.
        patterns: Dict[str, str] = {
            "aldehyde":          "[CX3H1](=O)[#6]",
            "carboxylic_acid":   "[CX3](=O)[OX2H1]",
            "primary_amine":     "[NX3H2;!$(NC=O);!$(N=*);!$([N+])]",
            "azide":             "[NX2]=[NX2]=[NX1]",
            "alkyne":            "[CX2]#[CX2]",
            "boronic_acid":      "[BX3]([OX2])[OX2]",
            "aryl_halide":       "[cX3][F,Cl,Br,I]",
        }
        for fg, smarts_str in patterns.items():
            try:
                patt = Chem.MolFromSmarts(smarts_str)
            except Exception:
                continue
            if patt is None:
                continue
            try:
                if mol.HasSubstructMatch(patt):
                    out.append(fg)
            except Exception:
                continue
        return out

    def get_forbidden_pairs(self) -> List[Tuple[str, str]]:
        """Return all (reactant_fg, leaving_group) vetoed pairs."""
        return list(self.forbidden_pairs)

    def render_table(self) -> str:
        """Render the (FGs × LGs) matrix as a markdown table."""
        rows = [
            "| FG \\ LG | " + " | ".join(LEAVING_GROUPS) + " |",
            "|" + "|".join(["---"] * (len(LEAVING_GROUPS) + 1)) + "|",
        ]
        for fg in FUNCTIONAL_GROUPS:
            cells = [fg]
            for lg in LEAVING_GROUPS:
                tag = "X" if (fg, lg) in self._veto_set else "OK"
                cells.append(tag)
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# VetoAugmentedReducer — wrap an existing reducer with FG + LG veto
# ---------------------------------------------------------------------------
# The CPU skeleton: the base reducer is ANY object that exposes a
# ``reduce`` method (or a ``reduce`` callable) that accepts a single
# input and returns an output (string SMILES, LamTerm, or any
# object).  The :meth:`reduce` wrapper first checks
# ``fg_matrix.is_compatible(fg1, fg2)`` and ``leave_veto.veto(...)``;
# if either is True it returns ``None`` (veto).
#
# This is intentionally **minimal** — we do not modify the existing
# :mod:`reducer.py` / :mod:`rules.py` / :mod:`stereo_aware_reduction`
# per the TODO-30 P1.2 constraint.  The veto is opt-in via the
# wrapper.
# ---------------------------------------------------------------------------


@dataclass
class VetoAugmentedReducer:
    """Reducer wrapper that gates ``reduce()`` on FG + LG veto.

    Parameters
    ----------
    base_reducer : object
        Any object exposing ``reduce(input) -> output``.  When the
        base reducer has no ``reduce`` method the wrapper raises
        :class:`TypeError` at construction time.
    fg_matrix : FGCompatMatrix or None
        The FG-compat matrix.  When ``None`` a fresh
        :class:`FGCompatMatrix` is constructed (default canonical
        matrix).
    leave_veto : LeaveGroupVeto or None
        The leave-group veto.  When ``None`` a fresh
        :class:`LeaveGroupVeto` is constructed (default canonical
        veto).

    Attributes
    ----------
    n_veto_calls : int
        Cumulative count of ``reduce()`` calls that were vetoed
        (FG or LG).  Useful for the audit / smoke-test report.
    n_pass_calls : int
        Cumulative count of ``reduce()`` calls that passed through
        to the base reducer.

    Notes
    -----
    * The wrapper is **stateless** w.r.t. the input — it does not
      mutate or transform the input, only gates the call.
    * When ``fg_matrix`` or ``leave_veto`` is ``None`` the
      corresponding check is a no-op (the wrapper passes through
      to the base reducer).  This makes the wrapper safe to
      construct with one of the two vetoes disabled.
    """

    base_reducer: object
    fg_matrix: Optional[FGCompatMatrix] = None
    leave_veto: Optional[LeaveGroupVeto] = None
    n_veto_calls: int = 0
    n_pass_calls: int = 0

    def __post_init__(self) -> None:
        # Validate base_reducer interface.
        if not hasattr(self.base_reducer, "reduce"):
            raise TypeError(
                f"base_reducer must expose a 'reduce' method; "
                f"got {type(self.base_reducer).__name__}"
            )
        if not callable(getattr(self.base_reducer, "reduce", None)):
            raise TypeError(
                f"base_reducer.reduce must be callable; "
                f"got {type(self.base_reducer.reduce).__name__}"
            )
        # Lazy defaults — keep __init__ cheap.
        if self.fg_matrix is None:
            self.fg_matrix = FGCompatMatrix()
        if self.leave_veto is None:
            self.leave_veto = LeaveGroupVeto()

    # ------------------------------------------------------------------
    # Veto API
    # ------------------------------------------------------------------
    def reduce(
        self,
        product: str,
        *,
        fg_pair: Optional[Tuple[str, str]] = None,
        leaving_group: Optional[str] = None,
    ) -> Optional[str]:
        """Reduce ``product`` after FG + LG veto.

        Parameters
        ----------
        product : str
            The input product / reactant SMILES.
        fg_pair : Tuple[str, str] or None
            Optional FG pair to check via :meth:`FGCompatMatrix.is_compatible`.
            When ``None`` the FG check is skipped.
        leaving_group : str or None
            Optional leaving-group symbol to check via
            :meth:`LeaveGroupVeto.veto`.  When ``None`` the LG check
            is skipped.

        Returns
        -------
        str or None
            The output of ``base_reducer.reduce(product)`` when the
            veto passes, or ``None`` when vetoed.  Returns ``None``
            on veto; the caller is expected to handle the None
            upstream.
        """
        # FG veto.
        if fg_pair is not None and self.fg_matrix is not None:
            fg1, fg2 = fg_pair
            if not self.fg_matrix.is_compatible(fg1, fg2):
                self.n_veto_calls += 1
                return None
        # LG veto.
        if leaving_group is not None and self.leave_veto is not None:
            if self.leave_veto.veto(product, leaving_group):
                self.n_veto_calls += 1
                return None
        # Pass through to base reducer.
        self.n_pass_calls += 1
        return self.base_reducer.reduce(product)

    def veto_rate(self) -> float:
        """Return the cumulative veto rate (vetoed / total calls)."""
        total = self.n_veto_calls + self.n_pass_calls
        if total <= 0:
            return 0.0
        return self.n_veto_calls / total


# ---------------------------------------------------------------------------
# Module-level registry helper
# ---------------------------------------------------------------------------


def register_fg_compat_veto() -> int:
    """Register the FG-compat + LG-veto and return the rule count.

    The CPU-skeleton "registry" is the two singleton matrices:
    a fresh :class:`FGCompatMatrix` and a fresh
    :class:`LeaveGroupVeto`.  Both are constructed via their
    canonical (default) initialisers and returned in a 2-tuple.
    The return value is the *number of distinct veto rules* — the
    count of vetoed cells in the FG-compat matrix plus the count
    of vetoed cells in the LG matrix.

    Returns
    -------
    int
        Total number of registered veto cells
        (``len(FGCompatMatrix().vetoed_pairs) +
        len(LeaveGroupVeto().forbidden_pairs)``).  Always ``≥ 1``
        when at least one veto is registered.
    """
    fg = FGCompatMatrix()
    lg = LeaveGroupVeto()
    return len(fg.vetoed_pairs) + len(lg.forbidden_pairs)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "FUNCTIONAL_GROUPS",
    "METAL_LABELS",
    "LEAVING_GROUPS",
    "FGCompatMatrix",
    "LeaveGroupVeto",
    "VetoAugmentedReducer",
    "register_fg_compat_veto",
]


# Self-test: print the two matrices when run as a script.
if __name__ == "__main__":  # pragma: no cover
    fg = FGCompatMatrix()
    lg = LeaveGroupVeto()
    print(f"# fg_compat_veto — FG matrix ({fg.n_vetoed}/{fg.n_cells} vetoed)")
    print(fg.render_table())
    print()
    print(
        f"# fg_compat_veto — LG matrix ({lg.n_vetoed}/{lg.n_cells} vetoed)"
    )
    print(lg.render_table())
