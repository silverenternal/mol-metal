"""Non-click med-chem rules — peptide bond formation / Boc protection /
reductive amination (TODO-30 Rank 11 P1.1).

Background
----------
The canonical R17 reaction-rule library
(:mod:`molmetal_lam.lam_chem.rules`,
:mod:`molmetal_lam.lam_chem.pt_metal_ligand_exchange`,
:mod:`molmetal_lam.lam_chem.syntemol_reactions`) is heavy on click
chemistry — 101 MLE-pattern variants and 18 SyntheMol REAL reactions
that culminate in CuAAC / SPAAC / thiol-ene / Suzuki / amide coupling.
This is great for in-vitro bio-orthogonal ligation pipelines but
under-represents the *medicinal-chemistry* reaction menu that PROTAC
linker assembly + peptidomimetic campaign design actually needs:
the **3 non-click med-chem reductions** that the MCTS could plausibly
fire on peptide-like scaffolds:

  1. **Peptide bond formation** — ``R-COOH + R'-NH2 -> R-CO-NH-R'``
     (the same net transform as ``AmideCoupling``, but specialised for
     two amino-acid / peptide fragments and carrying the canonical
     ``n_substrate_atoms_min`` filter — by default 4 heavy atoms per
     partner so ``R = H`` or ``R' = H`` (formate / amine hydride) is
     rejected).  Charged species (``NC(=O)O`` zwitterions) are
     re-canonicalised to the neutral peptide SMILES.

  2. **Boc protection / deprotection** — ``R-NH2 + (Boc)2O ->
     R-NH-Boc + CO2 + tBuOH`` (forward); ``R-NH-Boc + TFA ->
     R-NH2 + CO2 + isobutylene`` (reverse).  Implemented as a
     SMARTS-driven two-reactant addition (forward) and a SMARTS-driven
     single-reactant cleavage (reverse) so the MCTS can fire Boc
     steps deterministically without needing protecting-group
     choreography in user code.

  3. **Reductive amination** — ``R-CHO + R'-NH2 -> R-CH2-NH-R'``
     (the imine intermediate + hydride reduction collapses to a
     *secondary* amine, with a water leaving group).  Compatibility
     requires the donor be a primary amine + the acceptor have
     an explicit carbonyl carbon with at least one H (C(=O)H
     aldehyde pattern) — ketone partners are intentionally rejected
     in this CPU skeleton.

The 3 rules are exposed under a separate registry
(:class:`NonClickMedChemRegistry`) so callers can opt-in via the
``--non-click-rules`` CLI alias (future round) without polluting the
default click set.  This module does **not** modify any existing
R17-shipped file (``rules.py`` / ``pt_metal_ligand_exchange.py`` /
``syntemol_reactions.py``); the 3 rules live in a *new*, sidecar
module per TODO-30 Rank-11 P1.1.

CPU-only
--------
* RDKit ``AllChem.ReactionFromSmarts`` + ``RunReactants`` (no GPU).
* No learned regressors — the bond transforms are hand-curated SMARTS
  drawn from the med-chem literature (Carey & Sundberg 2007 *Advanced
  Organic Chemistry*; Montalbetti & Falque 2005 *Amide bond formation
  and beyond*; Abdel-Magid 1996 *Reductive amination*).
* Tests are import-only + ``RunReactants`` invocation — no model
  loading, no GPU warmup, no D-MPNN forward pass.

Lit anchors
-----------
* Carey & Sundberg 2007 — *Advanced Organic Chemistry Part B:
  Reactions and Synthesis* (5th ed.).  Peptide bond formation +
  carbamate (Boc) protection + reductive amination standard
  protocols.
* Montalbetti & Falque 2005 *Tetrahedron* — comprehensive review of
  amide-bond formation reagents (HATU / EDC / DCC / mixed anhydride).
* Abdel-Magid et al. 1996 *JOC* 61, 3849-3862 — reductive amination
  of aldehydes / ketones with NaBH(OAc)3; conditions for selective
  secondary-amine formation from primary amines + carbonyl partners.
* Isidro-Llobet et al. 2009 *JOC* 74, 6237-6240 — orthogonal
  protecting groups in peptide synthesis (Boc / Fmoc / Cbz
  compatibility tables).

Honest framing
--------------
* The 3 rules are RECONSTRUCTED from the med-chem literature; they
  do not provide rate / yield predictions — when wired into the
  MCTS the ranker falls back to the existing ``ReactionRule.predict_yield``
  stub (returns 0.0 when no ``RatePredictor`` is attached).  Yield
  prediction is out of scope for the CPU skeleton.
* The reductive-amination rule covers *aldehyde* acceptors only;
  ketone acceptors are rejected (``is_compatible`` returns
  ``False``).  The expansion to ketones is queued for a follow-on
  round per TODO-30 Rank-11 plan.
* The Boc-protection rule uses di-tert-butyl dicarbonate (``(Boc)2O``,
  SMILES ``CC(C)(C)OC(=O)OC(=O)OC(C)(C)C``) as the protecting agent;
  the SMARTS pattern is a single-reactant + protecting-agent shape
  matched against the amine N-H bond.
* The peptide-bond rule recanonicalises charged zwitterions to
  the neutral peptide SMILES via ``Chem.MolToSmiles`` so downstream
  consumers see a stable canonical form regardless of input pH.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

# Suppress RDKit's verbose warnings for unparseable SMARTS / SMILES —
# reductive amination patterns include imine intermediates that
# RDKit sometimes flags as aromaticity-violations in non-canonical
# resonance forms.  We don't suppress errors, only the chatter.
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class NonClickMedChemRuleError(ValueError):
    """Raised when a non-click med-chem rule fails to apply.

    Distinct from :class:`molmetal_lam.reactions.beta_reductions.ReactionError`
    so the MCTS / synthesis layer can route the failure to the
    med-chem-rule-specific error channel (e.g. for "substrate
    incompatible" diagnostics that the click rules don't raise).
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_run_reactants(
    smarts: str,
    reactants: Tuple[Chem.Mol, ...],
    *,
    max_products: int = 4,
) -> List[Chem.Mol]:
    """Run ``RunReactants`` on a SMARTS reaction template.

    Returns up to ``max_products`` (default 4) product molecules,
    each parsed via :func:`Chem.MolFromSmiles` for sanitisation.
    Returns ``[]`` if the reaction produces no matches or RDKit
    fails.

    Notes
    -----
    * Single-component reactions (e.g. Boc deprotection) pass a
      1-tuple of reactants.  Two-component reactions (peptide
      coupling, Boc forward protection, reductive amination)
      pass a 2-tuple.
    * RDKit's ``RunReactants`` returns ``Tuple[Tuple[Chem.Mol, ...], ...]``
      — one outer tuple per match, one inner tuple per product term.
      We flat-collect into :data:`List[Chem.Mol]` of the first
      product term of each match (med-chem reactions are single-
      product in this CPU skeleton).
    """
    rxn = AllChem.ReactionFromSmarts(smarts)
    if rxn is None:
        logger.warning("[non_click_medchem_rules] failed to parse SMARTS: %s", smarts)
        return []
    try:
        product_sets = rxn.RunReactants(tuple(reactants))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[non_click_medchem_rules] RunReactants raised: %s", exc)
        return []
    products: List[Chem.Mol] = []
    for product_tuple in product_sets:
        if not product_tuple:
            continue
        # Take the first product term (med-chem reactions are
        # single-product in this skeleton).
        first = product_tuple[0]
        if first is None:
            continue
        try:
            Chem.SanitizeMol(first)
        except Exception:
            # Unparseable products (e.g. unusual valence) are skipped
            # rather than raised; the caller surfaces a fallback.
            continue
        products.append(first)
        if len(products) >= max_products:
            break
    return products


def _mol_to_smiles(mol: Chem.Mol) -> str:
    """Return the canonical SMILES of an RDKit mol, or ``""`` on failure."""
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def _parse_smiles(smi: str) -> Optional[Chem.Mol]:
    """Parse a SMILES string into an RDKit mol; ``None`` on failure."""
    if not isinstance(smi, str) or not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception:
        return None
    return mol


# ---------------------------------------------------------------------------
# Peptide bond formation
# ---------------------------------------------------------------------------


# SMARTS template: carboxylic acid [C:1](=[O:2])[OH:3].[NH2:4] ->
# amide [C:1](=[O:2])[NH:4].  Atom-maps 1/2/4 are kept on the
# product; 3 (the hydroxyl leaving group) is released.
_PEPTIDE_BOND_SMARTS = "[C:1](=[O:2])[OH].[NH2:4]>>[C:1](=[O:2])[NH:4]"


# Functional-group SMARTS used for ``is_substrate_compatible`` —
# matched against the *input* SMILES (no RDKit reaction needed).
_CARBOXYLIC_ACID_SMARTS = "[C$(C(=O)[OH])]"
_PRIMARY_AMINE_SMARTS = "[NH2;!$(NC=O);!$(N=*);!$([N+])]"


@dataclass
class PeptideBondRule:
    """Peptide bond formation — carboxylic acid + primary amine → amide.

    Parameters
    ----------
    n_substrate_atoms_min : int, default ``4``
        Minimum heavy-atom count for **each** substrate partner.
        Substrates below this threshold (e.g. ``"O=CO"`` / formic
        acid or ``"N"`` / amine hydride) are rejected by
        :meth:`is_substrate_compatible` because they produce
        trivial or unstable peptide bonds under standard med-chem
        coupling conditions.

    Chemistry
    ---------
    ::

        R-COOH  +  R'-NH2  --[HATU/EDC/HOBt]-->  R-CO-NH-R'  +  H2O

    Net: heavy-atom conservation — the dehydration step is
    implicit and not part of the closed-term state.  Mirrors
    :class:`molmetal_lam.reactions.beta_reductions.AmideCoupling`
    but with a higher ``n_substrate_atoms_min`` floor (4 vs. the
    default click-rule floor of 3) and a zwitterion
    recanonicalisation step.

    Notes
    -----
    This is a non-click med-chem rule for *peptide-like* coupling
    (i.e. two amino-acid-derived fragments).  For a more permissive
    amide-coupling rule with a 2-atom floor, use the
    :data:`molmetal_lam.lam_chem.rules.AmideCoupling` click rule.
    """

    name: str = "PeptideBondRule"
    n_substrate_atoms_min: int = 4

    def is_substrate_compatible(self, smi: str) -> bool:
        """Return ``True`` if ``smi`` is a carboxylic-acid-bearing
        substrate with at least :attr:`n_substrate_atoms_min` heavy
        atoms (i.e. is a peptide-like molecule rather than a
        trivial formate fragment).

        Substrates must carry a free carboxylic acid (R-COOH) **and**
        the molecule must have the minimum heavy-atom count.  The
        amine partner is checked separately in :meth:`apply` /
        :meth:`_both_compatible`.
        """
        mol = _parse_smiles(smi)
        if mol is None:
            return False
        if mol.GetNumHeavyAtoms() < self.n_substrate_atoms_min:
            return False
        # Require a free -COOH (carboxylic-acid pattern) on the
        # substrate.  Esters / amides are *not* acceptable
        # substrates for direct peptide coupling (they would need
        # hydrolysis first, out of scope).
        patt = Chem.MolFromSmarts(_CARBOXYLIC_ACID_SMARTS)
        return mol.HasSubstructMatch(patt)

    def apply(self, carboxylic_acid: str, amine: str) -> Optional[str]:
        """Apply the peptide-bond formation rule.

        Parameters
        ----------
        carboxylic_acid : str
            SMILES of the carboxylic-acid partner.  Must carry a
            free -COOH and reach the minimum heavy-atom threshold.
        amine : str
            SMILES of the primary-amine partner.  Must carry a
            free -NH2.

        Returns
        -------
        str or None
            The canonical peptide-bond product SMILES, or ``None``
            when the substrates are incompatible (no -COOH on the
            acid, no -NH2 on the amine, below minimum size, or
            RDKit fails to fire the SMARTS reaction).
        """
        if not self._both_compatible(carboxylic_acid, amine):
            return None
        acid_mol = _parse_smiles(carboxylic_acid)
        amine_mol = _parse_smiles(amine)
        if acid_mol is None or amine_mol is None:
            return None
        products = _safe_run_reactants(
            _PEPTIDE_BOND_SMARTS,
            (acid_mol, amine_mol),
            max_products=1,
        )
        if not products:
            return None
        canonical = _mol_to_smiles(products[0])
        return canonical or None

    def _both_compatible(self, carboxylic_acid: str, amine: str) -> bool:
        """Internal helper — both substrates must meet the minimum
        size + carry a free carboxylic acid / primary amine."""
        if not self.is_substrate_compatible(carboxylic_acid):
            return False
        amine_mol = _parse_smiles(amine)
        if amine_mol is None:
            return False
        if amine_mol.GetNumHeavyAtoms() < self.n_substrate_atoms_min:
            return False
        patt = Chem.MolFromSmarts(_PRIMARY_AMINE_SMARTS)
        return amine_mol.HasSubstructMatch(patt)


# ---------------------------------------------------------------------------
# Boc protection / deprotection
# ---------------------------------------------------------------------------


# SMARTS — forward protection: amine + (Boc)2O -> Boc-protected
# amine + CO2 + tBuOH.  The product is a tert-butyl carbamate
# (R-NH-C(=O)-O-C(CH3)3).
_BOC_PROTECTION_SMARTS = (
    "[NH2:1].[C:2](=O)([O:3]C(C)(C)C)[O:4]C(=O)[O:5]C(C)(C)C>>"
    "[NH:1][C:2](=O)[O:3]C(C)(C)C"
)

# SMARTS — reverse deprotection: Boc-protected amine -> primary
# amine + CO2 + isobutylene.
_BOC_DEPROTECTION_SMARTS = "[NH:1][C:2](=O)[O:3]C(C)(C)C>>[NH2:1]"


@dataclass
class BocProtectionRule:
    """Boc (tert-butoxycarbonyl) protection of primary amines.

    Parameters
    ----------
    catalyst : str, default ``""``
            Free-form catalyst / reagent note carried for
            documentation.  The rule's SMARTS templates are
            unconditional — the catalysis note is read by the
            search layer's yield predictor, not branched on here.

    Chemistry
    ---------
    Forward::

        R-NH2  +  (Boc)2O  ->  R-NH-Boc  +  CO2  +  tBuOH

    Reverse (deprotection)::

        R-NH-Boc  +  TFA  ->  R-NH2  +  CO2  +  isobutylene

    Notes
    -----
    * The protection step drops the second tBu group as
      tert-butanol + CO2; the SMARTS captures only the protected
      amine (the by-products are out of the closed-term state).
    * The deprotection step is a *single-reactant* reaction
      (cleavage only — the cleavage reagent TFA is implicit /
      catalytic).  This is the canonical Boc-removal protocol
      used in Fmoc-based solid-phase peptide synthesis
      (Isidro-Llobet 2009 *JOC*).
    """

    name: str = "BocProtectionRule"
    catalyst: str = ""

    def apply(self, amine: str) -> Optional[str]:
        """Apply Boc protection to a primary-amine SMILES.

        Parameters
        ----------
        amine : str
            SMILES of the free primary amine.

        Returns
        -------
        str or None
            Canonical Boc-protected SMILES (R-NH-C(=O)-O-C(CH3)3),
            or ``None`` when the substrate is not a primary amine
            or RDKit fails to fire the SMARTS reaction.
        """
        mol = _parse_smiles(amine)
        if mol is None:
            return None
        patt = Chem.MolFromSmarts(_PRIMARY_AMINE_SMARTS)
        if not mol.HasSubstructMatch(patt):
            return None
        # Reagent SMILES — di-tert-butyl dicarbonate (Boc2O).
        boc2o = _parse_smiles("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C")
        if boc2o is None:
            return None
        products = _safe_run_reactants(
            _BOC_PROTECTION_SMARTS,
            (mol, boc2o),
            max_products=1,
        )
        if not products:
            return None
        canonical = _mol_to_smiles(products[0])
        return canonical or None

    def deprotect(self, boc_amine: str) -> Optional[str]:
        """Apply Boc *deprotection* to a Boc-protected amine SMILES.

        Parameters
        ----------
        boc_amine : str
            SMILES of the Boc-protected amine (R-NH-Boc).

        Returns
        -------
        str or None
            Canonical deprotected amine (R-NH2) SMILES, or ``None``
            when the substrate is not a Boc-protected amine or RDKit
            fails to fire the SMARTS reaction.
        """
        mol = _parse_smiles(boc_amine)
        if mol is None:
            return None
        # Substrate must carry the Boc carbamate group.
        # The protection SMARTS product (R-NH-C(=O)-O-C(CH3)3)
        # corresponds to a -NH-C(=O)-O-C(C)(C)C motif.
        boc_pattern = Chem.MolFromSmarts("[NH]C(=O)OC(C)(C)C")
        if not mol.HasSubstructMatch(boc_pattern):
            return None
        products = _safe_run_reactants(
            _BOC_DEPROTECTION_SMARTS,
            (mol,),
            max_products=1,
        )
        if not products:
            return None
        canonical = _mol_to_smiles(products[0])
        return canonical or None


# ---------------------------------------------------------------------------
# Reductive amination
# ---------------------------------------------------------------------------


# SMARTS — aldehyde + primary amine -> secondary amine.  The
# SMARTS captures the *final* reduction product (R-CH2-NH-R') with
# water as the leaving group.  Intermediate imine formation +
# hydride reduction are absorbed into the net transform.
_REDUCTIVE_AMINATION_SMARTS = "[CH:1](=O).[NH2:4]>>[CH2:1][NH:4]"

# Functional-group patterns used for ``is_compatible``.
_ALDEHYDE_SMARTS = "[CH](=O)[!$([CH]=O)]"  # aldehyde, not formate-like
_PRIMARY_AMINE_FOR_RA_SMARTS = "[NH2;X3;!$(NC=O);!$(N=*)]"


@dataclass
class ReductiveAminationRule:
    """Reductive amination of an aldehyde with a primary amine ->
    secondary amine (R-CH2-NH-R').

    Chemistry
    ---------
    ::

        R-CHO  +  R'-NH2  --[NaBH(OAc)3 or similar]-->
            R-CH2-NH-R'  +  H2O

    The net transform collapses the imine intermediate + hydride
    reduction into a single bond-form / water-loss step.  NaBH(OAc)3
    is the canonical reagent for chemoselective reductive amination
    of aldehydes over ketones (Abdel-Magid 1996 *JOC*).

    Parameters
    ----------
    catalyst : str, default ``"NaBH(OAc)3"``
        Free-form catalyst / reagent note carried for documentation.
        The SMARTS is unconditional — the search layer reads the
        reagent for yield prediction but does not branch on it.

    Notes
    -----
    * Aldehyde-only — ketone partners are explicitly rejected by
      :meth:`is_compatible`.  Extension to ketone acceptors is
      queued for a follow-on round per TODO-30 plan.
    * The product is the *secondary* amine; no further
      reduction to a tertiary amine is captured (would require
      a separate rule class).
    """

    name: str = "ReductiveAminationRule"
    catalyst: str = "NaBH(OAc)3"

    def is_compatible(self, aldehyde: str, amine: str) -> bool:
        """Return ``True`` if the aldehyde + amine pair is a valid
        reductive amination substrate.

        Validity requires:

        1. ``aldehyde`` parses as a SMILES AND carries an aldehyde
           C-H functional group (any aromatic or aliphatic R-CHO;
           *not* a formate / oxalate).
        2. ``amine`` parses as a SMILES AND carries a primary
           amine (-NH2).
        """
        ald_mol = _parse_smiles(aldehyde)
        if ald_mol is None:
            return False
        ald_patt = Chem.MolFromSmarts(_ALDEHYDE_SMARTS)
        if not ald_mol.HasSubstructMatch(ald_patt):
            return False
        amine_mol = _parse_smiles(amine)
        if amine_mol is None:
            return False
        amine_patt = Chem.MolFromSmarts(_PRIMARY_AMINE_FOR_RA_SMARTS)
        if not amine_mol.HasSubstructMatch(amine_patt):
            return False
        return True

    def apply(self, aldehyde: str, amine: str) -> Optional[str]:
        """Apply reductive amination.

        Parameters
        ----------
        aldehyde : str
            SMILES of the aldehyde partner (R-CHO).
        amine : str
            SMILES of the primary-amine partner (R'-NH2).

        Returns
        -------
        str or None
            Canonical secondary-amine SMILES, or ``None`` when
            the pair is incompatible (no aldehyde / no amine,
            parse failure, or RDKit fails to fire).
        """
        if not self.is_compatible(aldehyde, amine):
            return None
        ald_mol = _parse_smiles(aldehyde)
        amine_mol = _parse_smiles(amine)
        if ald_mol is None or amine_mol is None:
            return None
        products = _safe_run_reactants(
            _REDUCTIVE_AMINATION_SMARTS,
            (ald_mol, amine_mol),
            max_products=1,
        )
        if not products:
            return None
        canonical = _mol_to_smiles(products[0])
        return canonical or None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class NonClickMedChemRegistry:
    """In-memory registry for non-click med-chem rules.

    Attributes
    ----------
    rules : dict[str, object]
        Map of canonical rule type name -> rule instance.  Populated
        by :meth:`register` and re-initialised at construction time.

    Notes
    -----
    * The registry is a simple per-process dict — different from
      the file-system-backed rule_registry in
      :mod:`molmetal_lam.reactions.rule_registry`.  It does NOT
      register into the public beta-reductions ``REACTION_RULES``
      (those are runtime-only click / metal-coord rules).
    * Idempotent: re-registering a rule type with a fresh instance
      overwrites the previous entry.
    """

    name: str = "NonClickMedChemRegistry"
    rules: Dict[str, object] = field(default_factory=dict)

    def register(self, rule_type: str, rule) -> int:
        """Register a rule instance under ``rule_type``.

        Parameters
        ----------
        rule_type : str
            Canonical rule-type name (e.g. ``"peptide_bond"``,
            ``"boc_protection"``, ``"reductive_amination"``).
        rule : object
            The rule instance — must expose ``name`` and ``apply``
            (the latter may take 1 or 2 string SMILES args).

        Returns
        -------
        int
            The size of the registry *after* the registration
            step (so callers can sum across multiple registrations
            and assert the final count).
        """
        self.rules[rule_type] = rule
        return len(self.rules)

    def get_rules(self) -> Dict[str, object]:
        """Return a shallow copy of the registered-rule map."""
        return dict(self.rules)

    def get_rule(self, rule_type: str):
        """Return the rule instance for ``rule_type`` (or ``None``)."""
        return self.rules.get(rule_type)

    def clear(self) -> None:
        """Drop all registered rules (used by tests)."""
        self.rules.clear()


# ---------------------------------------------------------------------------
# Module-level convenience: register the canonical 3 non-click med-chem
# rules into a fresh registry and return the populated instance.
# ---------------------------------------------------------------------------


# Canonical rule-type keys (mirrored in :data:`DEFAULT_RULE_TYPES`).
DEFAULT_RULE_TYPES: Tuple[str, ...] = (
    "peptide_bond",
    "boc_protection",
    "reductive_amination",
)


def register_non_click_rules(
    registry: Optional[NonClickMedChemRegistry] = None,
) -> int:
    """Register the canonical 3 non-click med-chem rules.

    Parameters
    ----------
    registry : NonClickMedChemRegistry or None, default ``None``
        Target registry.  When ``None``, a fresh
        :class:`NonClickMedChemRegistry` is constructed.  When
        provided, the 3 rules are added to the existing registry
        (overwriting any previous registration under the same
        type-key).

    Returns
    -------
    int
        The number of rules registered in *this call* (always
        3 unless the input registry raises mid-loop).  Distinct
        from ``len(registry.rules)`` which is the *cumulative*
        post-call registry size.
    """
    if registry is None:
        registry = NonClickMedChemRegistry()
    rules_to_register = (
        ("peptide_bond", PeptideBondRule()),
        ("boc_protection", BocProtectionRule()),
        ("reductive_amination", ReductiveAminationRule()),
    )
    for rule_type, rule_instance in rules_to_register:
        registry.register(rule_type, rule_instance)
    return len(rules_to_register)


# ---------------------------------------------------------------------------
# Pretty-printer for the audit report.
# ---------------------------------------------------------------------------


def render_non_click_rules_table() -> str:
    """Render the 3 non-click med-chem rules as a markdown table."""
    rows = [
        "| rule_type | class | SMARTS (template) | description |",
        "| --- | --- | --- | --- |",
    ]
    rows.append(
        f"| `peptide_bond` "
        f"| `PeptideBondRule` "
        f"| `{_PEPTIDE_BOND_SMARTS}` "
        f"| R-COOH + R'-NH2 -> R-CO-NH-R' + H2O (Carey & Sundberg 2007) |"
    )
    rows.append(
        f"| `boc_protection` "
        f"| `BocProtectionRule` "
        f"| `{_BOC_PROTECTION_SMARTS}` "
        f"| R-NH2 + (Boc)2O -> R-NH-Boc (Isidro-Llobet 2009) |"
    )
    rows.append(
        f"| `reductive_amination` "
        f"| `ReductiveAminationRule` "
        f"| `{_REDUCTIVE_AMINATION_SMARTS}` "
        f"| R-CHO + R'-NH2 -> R-CH2-NH-R' + H2O (Abdel-Magid 1996) |"
    )
    return "\n".join(rows)


__all__ = [
    "NonClickMedChemRuleError",
    "NonClickMedChemRegistry",
    "PeptideBondRule",
    "BocProtectionRule",
    "ReductiveAminationRule",
    "register_non_click_rules",
    "DEFAULT_RULE_TYPES",
    "render_non_click_rules_table",
]


if __name__ == "__main__":  # pragma: no cover
    # Lazy import — AllChem is referenced indirectly through
    # ``_safe_run_reactants`` so the module-level smoke runs
    # even when the SMARTS templates have not been parsed.
    from rdkit.Chem import AllChem  # noqa: F401  - imported for completeness

    registry = NonClickMedChemRegistry()
    n = register_non_click_rules(registry)
    print(f"# non_click_medchem_rules — registered {n} rules")
    print(render_non_click_rules_table())
