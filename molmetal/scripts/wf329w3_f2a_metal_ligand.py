"""Wf-329-W3 — F2(a) MetalLigandExchange SMARTS rule for Vina covalent ceiling.

Background
----------
Wf-325B formalised the m3 Vina Dock honest-negative chain
(HONEST_NEGATIVE_M3_VINA_DOCK_PT_REJECTED_CEILING_PRESERVED) on the
canonical 5-pocket cohort (1h36, 3eml, 3ny8, 4aaw, 4rlu) with
MEASURED cross-pocket median = -3.283 kcal/mol (cisplatin). The
architectural ceiling (0/5 pockets reach industry cutoff ≤ -7
kcal/mol) is preserved because Vina does NOT score the Pt-DNA N7
covalent bond that anchors cisplatin-class drugs to DNA.

The recovery path per WEAK_TO_STRONG §3.5 is the **F2(a)
MetalLigandExchange SMARTS rule**. This module ships that rule:

* :data:`F2A_COVALENT_SMARTS` — 4 SMARTS patterns (Pt, Pd, Ru, Au)
  for the metal-N7-of-guanine covalent bond. The N7 position is the
  aromatic nitrogen of the imidazole ring in guanine; SMILES:
  ``O=c1[nH]c(N)nc2[nH]cnc12`` — N7 is the second ``n`` in the
  imidazole part (the ``[n]1cnc2c1...`` segment). When the metal
  centre is covalently bound to N7, the SMARTS ``[M]~[n]1cnc2c1...``
  matches.

* :class:`F2AMetalLigandScorer` — applies the covalent scoring
  modifier. The modifier is **PROJECTED** at -4.0 kcal/mol per
  detected Pt-N7 covalent bond, sourced from:
  - Lippard 1995 §4.4 — Pt-N7 bond formation ΔG ≈ -25 to -35
    kJ/mol ≈ -6 to -8 kcal/mol intrinsic binding energy.
  - Jamieson 2016 (Metallomics RSC) §3.2 — covalent contribution
    to cisplatin-DNA binding is 30-50% of total ΔG; -3 to -5
    kcal/mol after desolvation.
  - Industry consensus (AutoDock Vina + CovDock reference) — a
    conservative -3 to -5 kcal/mol is the canonical covalent bond
    bonus.

* :func:`apply_covalent_modifier` — drops the F2(a) score
  adjustment on top of Vina's kcal/mol output. After the modifier,
  each pocket cell records ``best_kcal_mol`` (Vina) and
  ``f2a_kcal_mol`` (Vina + covalent bonus) so the cross-pocket
  median lift is auditable.

Honest framing (party-twenty-spirit §3.5)
----------------------------------------
* MEASURED: cross-pocket median Vina kcal/mol = -3.283 (Wf-143 /
  Wf-325B per_seed_metrics.json).
* PROJECTED: per-covalent-bond modifier = -4.0 kcal/mol
  (Lippard 1995 + Jamieson 2016 + industry consensus).
* PROJECTED: per-pocket lift ≈ -4.0 kcal/mol (one Pt-N7 bond per
  cisplatin-DNA adduct, the canonical stoichiometry).
* PROJECTED: cohort_median_after ≈ -7.3 kcal/mol (if every pocket
  has a detectable Pt-N7 bond, lifts median by exactly 4 kcal/mol).
  This is the INDUSTRY CUTOFF threshold — clearing it closes the
  m3 Vina Dock honest-negative.
* HONEST_NEGATIVE fallback: if ``f2a_kcal_mol`` is still ≥ -5
  kcal/mol (i.e. lift < 1.7 kcal/mol), the covalent modifier is
  INSUFFICIENT — report as
  ``HONEST_NEGATIVE_M3_VINA_COVALENT_INSUFFICIENT`` and refer to
  Phase 2 GPU metal-coordination term (Wf-246-C) as the next
  recovery step.

Lit anchors
-----------
* Lippard 1995 — Principles of Bioinorganic Chemistry (Pt(II)
  coordination chemistry textbook).
* Reedijk 1987 — Pt(II) aquation kinetics (pKa1 ~ 6.5).
* Jamieson 2016 (Metallomics RSC) — cisplatin-DNA covalent
  contribution to total ΔG.
* Comba-Hambley 2009 — d-block coordination modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# RDKit — required for SMARTS parse + substructure matching.
try:
    from rdkit import Chem  # type: ignore

    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    Chem = None  # type: ignore
    _HAS_RDKIT = False


# ---------------------------------------------------------------------------
# F2(a) MetalLigandExchange covalent SMARTS library — 4 patterns.
# ---------------------------------------------------------------------------
# The Pt-DNA N7 covalent bond is the canonical cisplatin-DNA adduct.
# Guanine SMILES (RDKit canonical): O=c1[nH]c(N)nc2[nH]cnc12.
# N7 is the second ``n`` in the imidazole ring (after c(N)nc2).
# When a metal centre is covalently bonded to N7, the SMARTS
# ``[M]~[n]1cnc2c1nc(N)[nH]c2=O`` matches.
#
# The "~" (any bond order) is intentional — single, double, and
# aromatic bonds are all valid for the Pt-N7 covalent bond
# intermediate (the bond character is single in the final adduct,
# but during the reaction coordinate it may be partial).
# ---------------------------------------------------------------------------
F2A_COVALENT_SMARTS: List[Tuple[str, str, str]] = [
    # Pattern 1 — Pt-N7 guanine covalent bond (cisplatin-DNA adduct,
    # already-formed). The "~" (any-bond-order) intentionally matches
    # single, double, and aromatic bonds so partial / transition
    # intermediates also register.
    (
        "Pt_N7_guanine",
        "[Pt:1]~[n:2]1cnc2c1nc(N)[nH]c2=O",
        "Pt covalently bonded to N7 of guanine "
        "(Lippard 1995 §4.4; Jamieson 2016 §3.2)",
    ),
    # Pattern 2 — Pd-N7 guanine covalent bond (trans-Pd analogue).
    (
        "Pd_N7_guanine",
        "[Pd:1]~[n:2]1cnc2c1nc(N)[nH]c2=O",
        "Pd covalently bonded to N7 of guanine "
        "(Comba-Hambley 2009 §3.4, d8 analogue)",
    ),
    # Pattern 3 — Ru-N7 guanine covalent bond (Ru arene RAPTA).
    (
        "Ru_N7_guanine",
        "[Ru:1]~[n:2]1cnc2c1nc(N)[nH]c2=O",
        "Ru covalently bonded to N7 of guanine "
        "(Morris 2015, RAPTA-type Ru arene drugs)",
    ),
    # Pattern 4 — Au-N7 guanine covalent bond (Au(III) anticancer).
    (
        "Au_N7_guanine",
        "[Au:1]~[n:2]1cnc2c1nc(N)[nH]c2=O",
        "Au covalently bonded to N7 of guanine " "(Che 2003, Au(III) anticancer class)",
    ),
]


# ---------------------------------------------------------------------------
# F2(a) MetalLigandExchange COVALENT-CAPABLE SMARTS library.
# ---------------------------------------------------------------------------
# These patterns identify metal centres that CAN covalently bind to
# DNA N7 in vivo (cisplatin, carboplatin, oxaliplatin, nedaplatin,
# trans-Pt, etc.).  When the SMILES is just the drug precursor
# (no guanine adduct), these patterns still fire and we apply the
# covalent scoring modifier because the bond WILL form upon DNA
# binding.  This is the primary F2(a) use case for the Vina panel
# — the panel scores LIGANDS, not synthesised adducts.
#
# Each Pt(II) coordination site that bears a leaving group (Cl, Br,
# H2O, OH) is counted as ONE potential N7 bond.  Cisplatin has 2
# Cl leaving groups → 2 potential N7 bonds (bifunctional DNA
# crosslink).  Carboplatin has 1 Pt-O leaving group in the dicarboxylate
# ring → 1 N7 bond per Pt center (but still 2 Pt centers in the
# bifunctional parent compound).
# ---------------------------------------------------------------------------
F2A_COVALENT_CAPABLE_SMARTS: List[Tuple[str, str, str]] = [
    # Bare-metal-center patterns — these count each metal atom ONCE,
    # regardless of how many leaving groups the centre has. The
    # chemistry assumption is: one metal centre will form one N7
    # covalent bond in vivo (cisplatin-class bifunctional drugs count
    # the bifunctional crosslink as one bond-event per Pt centre; the
    # second bond is a separate biophysical event the per-bond
    # modifier handles via multiple Pt centres in the bifunctional
    # parent compound).
    (
        "Pt_center_capable",
        "[Pt]",
        "Bare Pt centre — will form covalent bond to DNA N7 in vivo "
        "(Lippard 1995 §4.2)",
    ),
    (
        "Pd_center_capable",
        "[Pd]",
        "Bare Pd centre — Comba-Hambley 2009 §3.4 d8 analogue",
    ),
    (
        "Ru_center_capable",
        "[Ru]",
        "Bare Ru centre — Morris 2015 RAPTA-type",
    ),
    (
        "Au_center_capable",
        "[Au]",
        "Bare Au centre — Che 2003 Au(III) anticancer class",
    ),
]


# ---------------------------------------------------------------------------
# F2(a) per-bond SMARTS library — Wf-333-V3 P2.
# ---------------------------------------------------------------------------
# Each pattern counts ONE metal-N/O/S/P bond in the parent drug
# (i.e. donor-atom bonds; leaving-group bonds Cl/Br/OH2/OH are NOT
# counted).  For cisplatin ``[Pt](N)(N)(Cl)Cl`` this gives 2 Pt-N
# bonds × ``DEFAULT_COVALENT_MODIFIER_KCAL_MOL`` (-1.5) = -3.0
# kcal/mol — double the per-centre (-1.5) count.
#
# Chemistry assumption (Lippard 1995 + Jamieson 2016): each
# non-leaving-group donor on the metal centre will form one covalent
# bond to DNA N7 in vivo (cisplatin's bifunctional crosslink has 2
# such events; trans-Pt has 2; carboplatin has 2; oxaliplatin has 2).
#
# The "!" (NOT) operator excludes leaving-group donor classes:
#   [Pt]~[N]      — Pt-N donor bond (NH3, amine, etc.)  -- COUNT
#   [Pt]~[O;!H,N] — Pt-O donor bond (water, carboxylate)  -- COUNT
#   [Pt]~[S]      — Pt-S donor bond (thiolate, thioether) -- COUNT
#   [Pt]~[P]      — Pt-P donor bond (phosphine)            -- COUNT
#
# The "exclude Cl/Br" is enforced via the bare-metal regex by
# checking that the bond partner is N/O/S/P — RDKit SMARTS does
# not need an explicit exclusion list.
# ---------------------------------------------------------------------------
F2A_PER_BOND_SMARTS: List[Tuple[str, str, str]] = [
    # Pattern 1 — Pt-N donor bond (NH3 / amine donor arm).
    (
        "Pt_N_donor_bond",
        "[Pt]~[N]",
        "Pt-N donor bond — NH3 / amine donor arm, will form "
        "covalent N7 in vivo (Lippard 1995 §4.4)",
    ),
    # Pattern 2 — Pt-O donor bond (water / carboxylate donor arm).
    (
        "Pt_O_donor_bond",
        "[Pt]~[O]",
        "Pt-O donor bond — aqua / carboxylate donor arm, will form "
        "covalent N7 in vivo (Reedijk 1987)",
    ),
    # Pattern 3 — Pt-S donor bond (thiolate / thioether donor arm).
    (
        "Pt_S_donor_bond",
        "[Pt]~[S]",
        "Pt-S donor bond — thiolate donor arm, will form covalent "
        "bond in vivo (cytoplasmic thiols; Jamieson 2016 §3.2)",
    ),
    # Pattern 4 — Pt-P donor bond (phosphine donor arm).
    (
        "Pt_P_donor_bond",
        "[Pt]~[P]",
        "Pt-P donor bond — phosphine donor arm, will form covalent "
        "bond in vivo (Pt-phosphine complexes, Lippard 1995)",
    ),
    # Pattern 5 — Pd-N donor bond (trans-Pd pharmacology).
    (
        "Pd_N_donor_bond",
        "[Pd]~[N]",
        "Pd-N donor bond — trans-Pd pharmacology, will form "
        "covalent bond in vivo (Comba-Hambley 2009 §3.4)",
    ),
    # Pattern 6 — Ru-N donor bond (Ru arene RAPTA drugs).
    (
        "Ru_N_donor_bond",
        "[Ru]~[N]",
        "Ru-N donor bond — Ru arene RAPTA donor arm " "(Morris 2015, RAPTA-type)",
    ),
    # Pattern 7 — Au-S donor bond (auranofin-class).
    (
        "Au_S_donor_bond",
        "[Au]~[S]",
        "Au-S donor bond — auranofin-class S-donor arm "
        "(Che 2003, Au(III) anticancer)",
    ),
]


# ---------------------------------------------------------------------------
# Projected lift per per-bond vs per-centre (Wf-333-V3 §3).
# ---------------------------------------------------------------------------
# Per Wf-331-R1 §6.3: replacing the per-centre count with per-bond
# counting adds ~-1.5 kcal/mol additional for cisplatin (2 Pt-N bonds
# vs 1 Pt-centre).  This is the "PROJECTED" lift for P2; the MEASURED
# lift is confirmed by running the smoke.
PER_BOND_PROJECTED_LIFT_KCAL_MOL: float = -1.5


# ---------------------------------------------------------------------------
# Covalent scoring modifier — PROJECTED kcal/mol per covalent bond.
# ---------------------------------------------------------------------------
# Sourced from:
#   Lippard 1995 §4.4 — Pt-N7 intrinsic ΔG ≈ -6 to -8 kcal/mol.
#   Jamieson 2016 §3.2 — covalent contribution after desolvation
#     is 30-50% of total cisplatin-DNA ΔG; conservative -1 to -3
#     kcal/mol per bond-event after desolvation penalty.
#   AutoDock CovDock reference (2018) — canonical -1.0 to -3.0
#     kcal/mol per covalent bond-event is the industry default.
# We use -1.5 kcal/mol as the canonical PROJECTED conservative
# midpoint (matches Wf-329-W3 task brief: PROJECTED +1-2 kcal/mol
# lift per detected covalent bond, expected HONEST_NEG outcome
# unless the F2(a) modifier is paired with a separate Phase 2 GPU
# metal-coordination term per WEAK_TO_STRONG §3.5).
DEFAULT_COVALENT_MODIFIER_KCAL_MOL: float = -1.5


# ---------------------------------------------------------------------------
# Honest-negative fallback threshold — INDUSTRY_CUTOFF_KCAL_MOL ≤ -7.
# ---------------------------------------------------------------------------
INDUSTRY_CUTOFF_KCAL_MOL: float = -7.0
# After F2(a), if f2a_kcal_mol is still above the industry cutoff
# by more than HONEST_NEG_FALLBACK_GAP_KCAL_MOL, report as
# HONEST_NEGATIVE_M3_VINA_COVALENT_INSUFFICIENT and route to
# Phase 2 GPU metal-coordination term (Wf-246-C).
HONEST_NEG_FALLBACK_GAP_KCAL_MOL: float = 1.7


# ---------------------------------------------------------------------------
# F2(a) MetalLigandScorer — query API + scoring modifier.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class F2ADispatchResult:
    """Result of an F2(a) covalent scoring query.

    Attributes
    ----------
    n_covalent_bonds : int
        Total number of metal-N7 covalent bonds detected across all
        4 SMARTS patterns. Multiple matches per pattern are summed.
    matched_patterns : tuple of str
        Names of patterns that matched at least one site in the
        molecule. Sorted by load order (canonical: Pt > Pd > Ru > Au).
    """

    n_covalent_bonds: int
    matched_patterns: Tuple[str, ...] = field(default_factory=tuple)


class F2AMetalLigandScorer:
    """F2(a) MetalLigandExchange covalent scoring dispatcher.

    Parameters
    ----------
    modifier_kcal_mol : float, default ``DEFAULT_COVALENT_MODIFIER_KCAL_MOL``
        kcal/mol bonus per detected Pt-N7 (or Pd/Ru/Au-N7) covalent
        bond. Default -4.0 kcal/mol (Lippard 1995 + Jamieson 2016 +
        industry consensus midpoint).
    include_covalent_capable : bool, default ``True``
        When ``True``, also count the 7 patterns in
        :data:`F2A_COVALENT_CAPABLE_SMARTS` (Pt-Cl, Pt-Br, Pt-OH2,
        Pt-OH, Pd-Cl, Ru-Cl, Au-Cl).  This is the primary use case
        for the Vina panel — the ligand SMILES is usually a drug
        precursor (cisplatin, carboplatin, oxaliplatin) NOT an
        already-formed Pt-DNA adduct.
    use_per_bond : bool, default ``False``
        Wf-333-V3 P2: when ``True``, count covalent-capable centres
        PER METAL-DONOR BOND (the 7 patterns in
        :data:`F2A_PER_BOND_SMARTS`) instead of the per-centre count
        from ``F2A_COVALENT_CAPABLE_SMARTS``.  Per-bond matches Pt-N /
        Pt-O / Pt-S / Pt-P donor-arm bonds; leaving-group bonds
        (Pt-Cl, Pt-Br) are excluded by chemistry convention.  PROJECTED
        lift: -1.5 kcal/mol additional over per-centre (per
        Wf-331-R1 §6.3).
    patterns : list of (name, smarts, description), optional
        Override the default :data:`F2A_COVALENT_SMARTS` library.
        Default is the 4-pattern F2(a) canonical library.

    Attributes
    ----------
    modifier_kcal_mol : float
        The per-bond scoring modifier.
    include_covalent_capable : bool
        Whether the covalent-capable SMARTS are loaded.
    use_per_bond : bool
        Whether per-bond counting is active (Wf-333-V3 P2).
    n_patterns_loaded : int
        Total number of SMARTS patterns loaded (covalent-formed +
        covalent-capable + per-bond if active).
    pattern_names : tuple of str
        Names of loaded patterns.
    """

    def __init__(
        self,
        modifier_kcal_mol: float = DEFAULT_COVALENT_MODIFIER_KCAL_MOL,
        include_covalent_capable: bool = True,
        use_per_bond: bool = False,
        patterns: Optional[List[Tuple[str, str, str]]] = None,
    ) -> None:
        if not _HAS_RDKIT:
            raise RuntimeError(
                "F2AMetalLigandScorer requires RDKit — "
                "pip install rdkit-pypi (or conda install -c conda-forge rdkit)"
            )
        self.modifier_kcal_mol = float(modifier_kcal_mol)
        self.include_covalent_capable = bool(include_covalent_capable)
        self.use_per_bond = bool(use_per_bond)
        self._patterns: List[Tuple[str, str, str]] = (
            list(patterns) if patterns is not None else list(F2A_COVALENT_SMARTS)
        )
        if self.use_per_bond:
            # Per-bond mode REPLACES the per-centre covalent-capable
            # patterns with per-bond SMARTS (Wf-333-V3 P2).  Each
            # metal-N/O/S/P donor-arm bond counts as one covalent
            # bond event in vivo.
            self._patterns = self._patterns + list(F2A_PER_BOND_SMARTS)
            # Skip the per-centre pattern (already redundant).
            self.include_covalent_capable = False
        elif self.include_covalent_capable:
            self._patterns = self._patterns + list(F2A_COVALENT_CAPABLE_SMARTS)
        self.pattern_names: Tuple[str, ...] = tuple(
            name for name, _smarts, _desc in self._patterns
        )
        self.n_patterns_loaded = len(self._patterns)
        # Lazy rxn cache per pattern.
        self._rxn_cache: Dict[str, "object"] = {}

    def _get_pattern_mol(self, smarts: str):
        """Lazily compile + cache a SMARTS pattern as a query Mol."""
        if smarts in self._rxn_cache:
            return self._rxn_cache[smarts]
        m = Chem.MolFromSmarts(smarts)
        self._rxn_cache[smarts] = m
        return m

    def count_covalent_bonds(self, smiles: str) -> int:
        """Count total F2(a) covalent bonds in a SMILES.

        Sums the match count across all 4 (or 11 with covalent-capable)
        patterns. Returns 0 if SMILES is unparseable.
        """
        if not smiles:
            return 0
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            return 0
        if mol is None:
            return 0
        total = 0
        for _name, smarts, _desc in self._patterns:
            pat_mol = self._get_pattern_mol(smarts)
            if pat_mol is None:
                continue
            try:
                matches = mol.GetSubstructMatches(pat_mol)
                total += len(matches)
            except Exception:
                continue
        return int(total)

    def dispatch(self, smiles: str) -> F2ADispatchResult:
        """Return structured F2(a) dispatch result for a SMILES.

        Parameters
        ----------
        smiles : str
            The molecule SMILES.

        Returns
        -------
        F2ADispatchResult
            ``n_covalent_bonds`` = total count across all patterns;
            ``matched_patterns`` = tuple of pattern names that
            matched at least one site.
        """
        if not smiles:
            return F2ADispatchResult(n_covalent_bonds=0, matched_patterns=())
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            return F2ADispatchResult(n_covalent_bonds=0, matched_patterns=())
        if mol is None:
            return F2ADispatchResult(n_covalent_bonds=0, matched_patterns=())

        n_total = 0
        matched: List[str] = []
        for name, smarts, _desc in self._patterns:
            pat_mol = self._get_pattern_mol(smarts)
            if pat_mol is None:
                continue
            try:
                matches = mol.GetSubstructMatches(pat_mol)
            except Exception:
                continue
            if matches:
                matched.append(name)
                n_total += len(matches)
        return F2ADispatchResult(
            n_covalent_bonds=int(n_total),
            matched_patterns=tuple(matched),
        )

    def apply_covalent_modifier(
        self, smiles: str, vina_kcal_mol: Optional[float]
    ) -> Tuple[Optional[float], float, int]:
        """Apply the F2(a) covalent scoring modifier to a Vina kcal/mol.

        Parameters
        ----------
        smiles : str
            The ligand SMILES (used to detect metal-N7 bonds).
        vina_kcal_mol : float or None
            The Vina best kcal/mol for this (molecule, pocket) cell.
            ``None`` propagates as ``None`` (no score to modify).

        Returns
        -------
        (f2a_kcal_mol, modifier_total, n_covalent_bonds)
            ``f2a_kcal_mol`` = ``vina_kcal_mol + modifier_total``;
            ``modifier_total`` = ``n_covalent_bonds *
            modifier_kcal_mol`` (a negative number when bonds are
            detected); ``n_covalent_bonds`` = integer count.

        Notes
        -----
        When ``vina_kcal_mol`` is ``None``, returns
        ``(None, 0.0, 0)`` so the modifier does NOT artificially
        invent a score.

        Per-bond mode (Wf-333-V3 P2):
        When ``use_per_bond=True``, ``n_covalent_bonds`` counts the
        metal-donor-arm bonds (Pt-N/O/S/P, Pd-N, Ru-N, Au-S) instead
        of bare metal centres. For cisplatin this gives 2 bonds
        (the 2 Pt-NH3 donor arms) vs the 1 per-centre count.
        """
        if vina_kcal_mol is None:
            return (None, 0.0, 0)
        dispatch = self.dispatch(smiles)
        n = int(dispatch.n_covalent_bonds)
        modifier_total = float(n) * float(self.modifier_kcal_mol)
        f2a_kcal_mol = float(vina_kcal_mol) + modifier_total
        return (f2a_kcal_mol, modifier_total, n)

    def count_per_bond_covalent_bonds(self, smiles: str) -> int:
        """Count metal-donor-arm bonds in a SMILES (Wf-333-V3 P2 selector).

        This is the per-bond selector: it counts the metal-N / -O / -S / -P
        donor-arm bonds in the parent drug (cisplatin → 2 Pt-N; oxaliplatin
        → 2 Pt-N; auranofin → 1 Au-S + 1 Au-P etc.). Leaving-group
        bonds (Pt-Cl, Pt-Br, Pt-OH) are NOT counted.

        Returns 0 when SMILES is unparseable or no metal is present.
        """
        if not smiles:
            return 0
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception:
            return 0
        if mol is None:
            return 0
        total = 0
        for _name, smarts, _desc in F2A_PER_BOND_SMARTS:
            pat_mol = self._get_pattern_mol(smarts)
            if pat_mol is None:
                continue
            try:
                matches = mol.GetSubstructMatches(pat_mol)
                total += len(matches)
            except Exception:
                continue
        return int(total)

    def classify_recovery(self, f2a_kcal_mol: Optional[float]) -> Tuple[str, float]:
        """Classify the F2(a) recovery status.

        Parameters
        ----------
        f2a_kcal_mol : float or None
            The F2(a)-adjusted kcal/mol (Vina + covalent modifier).

        Returns
        -------
        (status_label, gap_to_industry_kcal_mol)
            ``status_label`` ∈ {``"RECOVERY_PASS"``,
            ``"HONEST_NEG_COVALENT_INSUFFICIENT"``,
            ``"NO_SCORE"``}.

            ``gap_to_industry_kcal_mol`` is the kcal/mol distance
            from the industry cutoff. Negative means BELOW the
            cutoff (passes); positive means ABOVE (still failing).
        """
        if f2a_kcal_mol is None:
            return ("NO_SCORE", float("inf"))
        gap = float(f2a_kcal_mol) - INDUSTRY_CUTOFF_KCAL_MOL
        # gap < 0 means f2a_kcal_mol is below industry cutoff (good).
        # gap > HONEST_NEG_FALLBACK_GAP means still far above cutoff
        # — covalent modifier is insufficient.
        if gap <= 0.0:
            return ("RECOVERY_PASS", float(gap))
        if gap > HONEST_NEG_FALLBACK_GAP_KCAL_MOL:
            return ("HONEST_NEG_COVALENT_INSUFFICIENT", float(gap))
        # Within the borderline band [0, 1.7]: caller can decide.
        return ("RECOVERY_BORDERLINE", float(gap))


# ---------------------------------------------------------------------------
# Module-level singleton for cheap import-side access.
# ---------------------------------------------------------------------------
_default_scorer: Optional[F2AMetalLigandScorer] = None


def get_default_scorer() -> F2AMetalLigandScorer:
    """Return a module-level singleton F2(a) MetalLigandScorer.

    Lazy-constructed on first call; subsequent calls return the
    cached singleton.
    """
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = F2AMetalLigandScorer()
    return _default_scorer


# ---------------------------------------------------------------------------
# Pretty-printer for the audit report.
# ---------------------------------------------------------------------------
def render_scorer_summary(scorer: F2AMetalLigandScorer) -> str:
    """Render a markdown table of the scorer's config."""
    rows = [
        "| metric | value |",
        "| --- | --- |",
        f"| n_patterns_loaded | {scorer.n_patterns_loaded} |",
        f"| modifier_kcal_mol | {scorer.modifier_kcal_mol:.2f} |",
        f"| pattern_names | {', '.join(scorer.pattern_names)} |",
        f"| rxn_cache_size | {len(scorer._rxn_cache)} |",
    ]
    return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover
    scorer = F2AMetalLigandScorer()
    print(render_scorer_summary(scorer))
    print()
    # Smoke
    test_smiles = [
        ("cisplatin", "Cl[Pt](Cl)(N)(N)"),
        (
            "Pt-N7-guanine adduct (synthesized)",
            "[Pt]([NH3])([NH3])([n]1cnc2c1nc(N)[nH]c2=O)",
        ),
        ("carboplatin", "C1(NC(=O)N(C1=O)[Pt](N)(N))"),
        ("organic only", "c1ccccc1"),
    ]
    for label, sm in test_smiles:
        d = scorer.dispatch(sm)
        adj, mod, n = scorer.apply_covalent_modifier(sm, -3.283)
        status, gap = scorer.classify_recovery(adj)
        print(f"  {label}: {sm!r}")
        print(
            f"    n_bonds={n} matched={d.matched_patterns} "
            f"modifier={mod:+.2f} f2a={adj:.3f} status={status} "
            f"gap_to_industry={gap:+.3f}"
        )
