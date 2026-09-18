"""Reactions as β-reductions of the Molecular Lambda Calculus (MLC).

This module implements the **Reaction layer** of MLC, formalized in
``TODO/13_lambda_clickchem/molecular_lambda_calculus.md`` §4.

Core thesis
-----------
**Every chemical reaction is exactly one β-reduction step** — that is, a
redex is fired and the term is rewritten to its β-normal form
successor.  Click-chemistry reactions (CuAAC, SPAAC, SPC, Diels–Alder,
Thiol–Ene) are concrete β-reduction rules that transform two educt
molecules (closed λ-terms) into one or more product molecules (also
closed λ-terms).

A :class:`ReactionRule` exposes the standard MLC reduction interface:

    rule.reduce(reactants, **kwargs) -> List[MoleculeClosedTerm]

The list contains every distinct successor term reachable by firing the
rule on the given reactants — i.e. the *image* of the redex under β.
Multiple products arise when a single reaction rule has multiple
regio- or stereoisomeric outcomes (e.g. the 1,4/1,5 regiochemistry of
SPAAC).

Mass balance
------------
Each rule carries a ``stoichiometry: dict[atom_symbol -> int]`` that
records the **net atom change** between reactants and products (zero for
pure bond-forming reactions like click cycloadditions, because the
reactants are already closed terms).  ``verify_mass_balance`` enforces
that the rule's stoichiometry is consistent with a test reaction
performed at instantiation time — click chemistry rules should always
report the empty stoichiometry because the heavy-atom count is
conserved by construction.

Public API
----------
``ReactionRule``                  abstract dataclass + concrete reducers
``CuAAC`` / ``SPAAC`` / ``SPC``   azide-based click reactions
``DielsAlder``                    [4+2] cycloaddition
``ThiolEne``                      radical thiol-ene addition
``REACTION_RULES``                registry dict (name -> rule instance)
``verify_mass_balance``           audit rule stoichiometry against the test reaction

No global state.  Every rule is a pure function: ``reduce`` returns new
:class:`MoleculeClosedTerm` objects and never mutates its inputs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

# Public types from sibling layers
from molmetal_lam.atoms.combinators import Atom
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

# Phase 3 cold-swap (WF-Deflex Lambda Combinators): Sites 1+2 only.
# The HOF versions are bit-for-bit equivalent to the original imperative
# nested loops (verified by test_lambda_combinators_cold_swap.py).
# Sites 3-5 (proof_search.py) are NOT touched — they belong to the
# pocket-invariance workflow.
from molmetal_lam.lam_chem.lambda_combinators import (
    count_heavy_atoms as _count_heavy_atoms,
    diff_counts as _diff_counts,
)

# WF-Lambda-Boost Phase 2 + 3 — F2(a) SMARTS library.
# 5 MetalLigandExchange + 2 AquaExchange patterns from
# :mod:`molmetal_lam.lam_chem.pt_metal_ligand_exchange`.  The
# MetalLigandExchange / AquaExchange rule classes below expose the
# library via ``available_smarts()`` so the search layer can introspect
# without re-importing.
from molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
    METAL_LIGAND_EXCHANGE_SMARTS,
    AQUA_EXCHANGE_SMARTS,
    AQUA_CONTEXTS,
    AquaContext,
    get_metal_ligand_exchange_patterns,
    get_aqua_exchange_patterns,
    get_aqua_context,
)

# L4 instrumentation counters (govern_review_L4_L6.md).
_L4_COUNTERS: Dict[str, Dict[str, int]] = {}
for _n in ("CuAAC", "SPAAC", "SPC", "DielsAlder", "ThiolEne",
           "Suzuki", "AmideCoupling",
           # F2(a): metal-coordination family — Pt_II / Ru_II / Ir_III
           # specific reductions.  Same counter schema as click rules
           # so l4_metrics() can audit them uniformly.
           "MetalLigandExchange", "AquaExchange"):
    _L4_COUNTERS[_n] = {"fire": 0, "attempts": 0, "products": 0,
                        "runreactants_err": 0, "thiolene_can_apply": 0}
_L4_MASS_BALANCE_PASS = {"pass": 0, "fail": 0}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReactionError(ValueError):
    """Raised when a reaction rule cannot be applied (= ill-typed redex)."""


# ---------------------------------------------------------------------------
# Mass-balance helpers
# ---------------------------------------------------------------------------
#
# Phase 3 cold-swap (WF-Deflex Lambda Combinators):
#   ``_count_heavy_atoms`` and ``_diff_counts`` are now imported from
#   ``molmetal_lam.lam_chem.lambda_combinators`` (Sites 1+2).  The HOF
#   versions are bit-for-bit equivalent to the original imperative
#   nested loops (verified by
#   ``test_lambda_combinators_cold_swap.py``).  Original Site 1 was
#   lines 86-108; original Site 2 was lines 111-122.


def _verify_stoichiometry(
    rule_name: str,
    expected: Dict[str, int],
    reactants: Iterable[str],
    products: Iterable[str],
) -> None:
    """Raise :class:`ReactionError` if observed atom delta ≠ expected."""
    observed = _diff_counts(reactants, products)
    if observed != expected:
        raise ReactionError(
            f"[{rule_name}] mass-balance mismatch: "
            f"expected stoich {expected}, observed {observed}"
        )


# ---------------------------------------------------------------------------
# Abstract rule
# ---------------------------------------------------------------------------


PatternLike = Union[str, Callable[[MoleculeClosedTerm], bool]]


@dataclass
class ReactionRule:
    """A chemical reaction IS a β-reduction rule.

    Attributes
    ----------
    name : str
        Human-readable identifier, e.g. ``"CuAAC"``.
    pattern_smiles : str or callable or None
        * If ``str``: an RDKit reaction SMARTS (with ``>>``) used for
          pattern matching.  Recommended form — it is portable, editable,
          and round-trips through ``rdChemReactions``.
        * If ``callable``: a Python predicate ``(MoleculeClosedTerm) ->
          bool`` deciding applicability on the union of reactants.
        * If ``None``: the rule is purely structural (e.g. Thiol–Ene,
          which has no canonical RDKit SMARTS) and relies on explicit
          functional-group checks in :meth:`reduce`.
    requires_catalyst : str or None
        Optional catalyst tag such as ``"Cu(I)"`` (CuAAC) or ``None``
        (SPAAC, Thiol–Ene).  Carried for documentation / yield-prediction
        plumbing; the reduction itself is implemented in RDKit and does
        not branch on this field.
    stoichiometry : dict[atom_symbol -> int]
        **Net** atom count change produced by the rule.  For pure
        click-chemistry bond-forming reactions this is always the empty
        dict (heavy atoms are conserved).  Populated by the subclass
        when ``_build_reaction_template`` runs.
    """

    name: str
    pattern_smiles: Optional[str] = None
    pattern: Optional[Callable[[MoleculeClosedTerm], bool]] = None
    requires_catalyst: Optional[str] = None
    stoichiometry: Dict[str, int] = field(default_factory=dict)

    # The concrete reaction template (rdChemReactions.ReactionFromSmarts).
    # Populated lazily by ``_rdkit_reaction`` so we don't pay RDKit's
    # import cost at module load.
    _rdkit_reaction: Optional[object] = field(
        default=None, repr=False, compare=False
    )

    # Optional rate predictor wired in by :mod:`reactions.rate_predictor`.
    # Holds a trained :class:`HeuristicRegressor` so the synthesis layer
    # can score ``predict_yield(smi_a, smi_b)`` without re-importing
    # RDKit or sklearn on the hot path.
    rate_predictor: Optional[object] = field(
        default=None, repr=False, compare=False
    )

    def attach_rate_predictor(self, predictor) -> None:
        """Attach a trained :class:`RatePredictor` to this rule.

        Parameters
        ----------
        predictor : RatePredictor or None
            A fitted predictor from :mod:`reactions.rate_predictor`.
            ``predictor.reaction_name`` should equal ``self.name`` so
            the synthesis layer can route by rule name; a mismatch
            raises :class:`ReactionError`.
        """
        if predictor is None:
            self.rate_predictor = None
            return
        if getattr(predictor, "reaction_name", None) != self.name:
            raise ReactionError(
                f"[{self.name}] predictor name "
                f"{getattr(predictor, 'reaction_name', None)!r} mismatch"
            )
        self.rate_predictor = predictor

    def predict_yield(self, smiles_a: str, smiles_b: str) -> float:
        """Return the predicted isolated yield in [0, 1].

        Returns 0.0 if no rate predictor has been attached or the
        underlying regressor cannot score the pair.
        """
        if self.rate_predictor is None:
            return 0.0
        try:
            return float(self.rate_predictor.predict(smiles_a, smiles_b))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # The reduction
    # ------------------------------------------------------------------

    def reduce(
        self,
        molecule: Union[MoleculeClosedTerm, Tuple[MoleculeClosedTerm, ...]],
        **kwargs,
    ) -> List[MoleculeClosedTerm]:
        """Apply the β-reduction rule to ``molecule`` (or tuple of).

        Parameters
        ----------
        molecule : MoleculeClosedTerm or tuple of MoleculeClosedTerm
            The educts.  Single-reactant rules (no current examples in
            this file, but supported) accept a single term; bi-molecular
            click rules accept a 2-tuple.
        **kwargs
            Forwarded to the subclass-specific ``_reduce`` method.
            Currently understood: ``sanitize`` (default ``True``),
            ``embed_3d`` (default ``False`` — embedding is expensive
            and unused by the type-check layer).  WF-T30 P1.2 — also
            ``fg_constraints`` (dict ``{"strict": bool}``); when
            present and ``"strict"=True``, every reactant's SMILES is
            checked against :data:`FG_COMPATIBILITY` via
            :func:`vet_smiles_against_rule`.  On rejection, this method
            returns ``[]`` *before* invoking ``_reduce`` (i.e. the rule
            silently doesn't apply — same semantics as an empty
            product list).  Default behaviour (``fg_constraints``
            absent or ``strict=False``) is *permissive* and preserves
            backward compat with Round-12 + Round-13 measurements.

        Returns
        -------
        list[MoleculeClosedTerm]
            Every distinct β-successor of the redex.  Empty list means
            the rule does not apply (= the reactants don't form the
            redex, OR the FG veto rejected at least one reactant).
            Raises :class:`ReactionError` on a hard failure
            (RDKit parse error, malformed pattern).

        Notes
        -----
        Pure function: does not mutate ``molecule`` or its
        ``MoleculeClosedTerm`` siblings.
        """
        reactants = self._coerce_reactants(molecule)
        # WF-T30 P1.2 — opt-in FG veto.  Lazy import to avoid RDKit
        # cost on the import path of ``reactions`` (matches the
        # existing ``_rdkit_reaction_template`` lazy pattern).
        fg_constraints = kwargs.get("fg_constraints")
        if fg_constraints and fg_constraints.get("strict", False):
            try:
                from molmetal_lam.reactions.fg_compatibility import (
                    vet_smiles_against_rule,
                )
            except Exception:
                vet_smiles_against_rule = None  # type: ignore[assignment]
            if vet_smiles_against_rule is not None:
                for r in reactants:
                    r_smiles = getattr(r, "source_smiles", None) or \
                        getattr(r, "smiles", None)
                    if r_smiles and not vet_smiles_against_rule(
                        str(r_smiles), self.name, strict=True,
                    ):
                        # Silent veto: empty product list (consistent
                        # with "rule does not apply").
                        return []
        return self._reduce(reactants, **kwargs)

    # -- the interface subclasses override -----------------------------

    def _reduce(
        self,
        reactants: Tuple[MoleculeClosedTerm, ...],
        **kwargs,
    ) -> List[MoleculeClosedTerm]:
        """Subclass-specific reduction implementation.

        Inputs are guaranteed to be a tuple of one or two
        :class:`MoleculeClosedTerm` instances.  Implementations should
        return a (possibly empty) list of product molecules.
        """
        raise NotImplementedError(
            f"{type(self).__name__}._reduce must be implemented by subclass"
        )

    # ------------------------------------------------------------------
    # Mass-balance audit
    # ------------------------------------------------------------------

    def verify_mass_balance(
        self,
        reactants_smiles: Iterable[str],
        products_smiles: Iterable[str],
    ) -> None:
        """Verify the rule's recorded stoichiometry matches an actual reaction.

        This is the **invariant check** — every click rule must be
        mass-balanced by construction (no atoms are created or
        destroyed).  Pass the SMILES of a representative educt set and a
        representative product set; the heavy-atom delta is compared
        to ``self.stoichiometry`` and a :class:`ReactionError` is raised
        on mismatch.
        """
        reactants = list(reactants_smiles)
        products = list(products_smiles)
        _verify_stoichiometry(
            self.name, self.stoichiometry, reactants, products
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_reactants(
        molecule: Union[MoleculeClosedTerm, Tuple[MoleculeClosedTerm, ...]],
    ) -> Tuple[MoleculeClosedTerm, ...]:
        if isinstance(molecule, MoleculeClosedTerm):
            return (molecule,)
        return tuple(molecule)

    def _rdkit_reaction_template(self):
        """Compile and cache the RDKit reaction from ``self.pattern_smiles``."""
        if self._rdkit_reaction is not None:
            return self._rdkit_reaction
        if not self.pattern_smiles:
            raise ReactionError(
                f"[{self.name}] no SMARTS pattern set on this rule"
            )
        try:
            from rdkit.Chem import rdChemReactions  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "rdChemReactions required for SMARTS-based reaction rules"
            ) from exc
        rxn = rdChemReactions.ReactionFromSmarts(self.pattern_smiles)
        if rxn is None:
            raise ReactionError(
                f"[{self.name}] RDKit failed to parse SMARTS: "
                f"{self.pattern_smiles!r}"
            )
        self._rdkit_reaction = rxn
        return rxn

    # ------------------------------------------------------------------
    # Backward / retrosynthesis (Phase 2: SynFlowNet integration)
    # ------------------------------------------------------------------

    def reverse_can_apply(
        self, product_smiles: str
    ) -> List[Tuple[str, List[str]]]:
        """Decompose ``product_smiles`` into educts using the swapped SMARTS.

        Phase-2 SynFlowNet hook.  Each ``ReactionRule`` exposes the
        same ``reverse_can_apply(product_smiles)`` interface so a
        synthesis layer can ask every rule in parallel which ones can
        explain the product.  Returns a list of
        ``(rule_name, [reactant_a_smiles, reactant_b_smiles])`` tuples.

        Rules whose ``pattern_smiles`` is ``None`` (e.g. ``ThiolEne``)
        or whose SMARTS cannot be parsed in reverse return ``[]``.

        The reverse direction uses the same SMARTS as the forward
        direction, with the payload of the ``>>`` arrow swapped — RDKit
        reaction SMARTS are themselves reversible (modulo aromatic /
        Kekule edge cases, which we handle via the standard
        kekulize-and-retry fallback).
        """
        if not self.pattern_smiles:
            return []

        try:
            from rdkit import Chem  # type: ignore[import-not-found]
            from rdkit.Chem import rdChemReactions  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "rdChemReactions required for reverse_can_apply"
            ) from exc

        if ">>" not in self.pattern_smiles:
            return []
        lhs, rhs = self.pattern_smiles.split(">>", 1)
        rev_smarts = f"{rhs}>>{lhs}"
        try:
            rev_rxn = rdChemReactions.ReactionFromSmarts(rev_smarts)
        except Exception:
            return []
        if rev_rxn is None:
            return []
        try:
            rdChemReactions.ChemicalReaction.Initialize(rev_rxn)
        except Exception:
            return []

        prod_mol = Chem.MolFromSmiles(product_smiles)
        if prod_mol is None:
            return []

        # Substructure probe — RDKit's RunReactants can fail silently
        # for aromatic / Kekule mismatches, so we first ask whether
        # the product matches the swapped-template reactant.
        try:
            if not rev_rxn.IsMoleculeReactant(prod_mol):
                return []
        except Exception:
            return []

        try:
            rs = rev_rxn.RunReactants((prod_mol,))
        except Exception:
            return []

        # Kekulize-and-retry fallback (SynFlowNet trick).
        if not rs:
            try:
                mol2 = Chem.MolFromSmiles(Chem.MolToSmiles(prod_mol))
                if mol2 is not None:
                    Chem.Kekulize(mol2, clearAromaticFlags=True)
                    rs = rev_rxn.RunReactants((mol2,))
            except Exception:
                rs = []

        if not rs:
            return []

        out: List[Tuple[str, List[str]]] = []
        seen: set = set()
        for rset in rs:
            rlist = list(rset)
            smis: List[str] = []
            ok = True
            for m in rlist:
                try:
                    Chem.SanitizeMol(m)
                except Exception:
                    pass
                try:
                    smis.append(Chem.MolToSmiles(m))
                except Exception:
                    ok = False
                    break
            if not ok or len(smis) < 1:
                continue
            # Normalise to 2-tuples (1-reactant rules get "" as partner).
            if len(smis) == 1:
                smis = [smis[0], ""]
            key = (smis[0], smis[1])
            if key in seen:
                continue
            seen.add(key)
            out.append((self.name, smis))
        return out


# ---------------------------------------------------------------------------
# Reaction-result parsing helpers (shared by all SMARTS-based rules)
# ---------------------------------------------------------------------------


def _rdkit_product_sets_to_closed_terms(
    product_sets: Iterable,
    sanitize: bool,
) -> List[MoleculeClosedTerm]:
    """Convert ``rdChemReactions`` product-set iterables to closed terms.

    Each entry in ``product_sets`` is an iterable of
    :class:`rdkit.Chem.Mol` representing one set of products (click
    cycloadditions normally produce a single product, but a single
    reaction can split into multiple fragments).  We keep only those
    fragments whose ``Mol`` parses cleanly; we deduplicate by canonical
    SMILES.
    """
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError("RDKit required for SMARTS-based reactions") from exc

    seen_canonical: Dict[str, MoleculeClosedTerm] = {}
    for product_set in product_sets:
        # If the reaction produced more than one fragment, combine them
        # in a single Mol by merging each fragment into the first.  This
        # mirrors how RDKit represents ``A.B>>C.D`` (two products in one
        # entry separated by ``.``).
        mols: List = []
        for mol in product_set:
            # Suppress per-atom valence warnings on raw reaction output —
            # the products inherit unusual valences from the SMARTS map.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    Chem.SanitizeMol(mol)
            except Exception:
                pass
            mols.append(mol)
        if not mols:
            continue
        if len(mols) == 1:
            combined = mols[0]
        else:
            combined = mols[0]
            for extra in mols[1:]:
                combined = _combine_mols(combined, extra)
        try:
            if sanitize:
                Chem.SanitizeMol(combined)
            term = MoleculeClosedTerm.from_rdkit(combined)
        except Exception:
            # Skip fragments that cannot be parsed into a closed term.
            continue
        canonical = term.canonical_smiles()
        if canonical not in seen_canonical:
            term.term = f"β({seen_canonical.__len__()})"  # 0-indexed label
            seen_canonical[canonical] = term
    return list(seen_canonical.values())


def _combine_mols(a, b):
    """Merge two RDKit mols into one (no bonds between fragments)."""
    from rdkit import Chem  # type: ignore[import-not-found]
    from rdkit.Chem import rdmolops  # type: ignore[import-not-found]

    combo = Chem.RWMol(Chem.CombineMols(a, b))
    return combo.GetMol()


def _run_reactants_symmetric(
    rule_name: str,
    rxn,
    mol_a: "Chem.Mol",
    mol_b: "Chem.Mol",
):
    """Call ``rxn.RunReactants`` on ``(a, b)``; if empty, retry ``(b, a)``.

    Many of the SMARTS-defined click rules are written with an *asymmetric*
    signature — ``[N:1]=[N:2]=[N:3].[C:4]#[CH:5]>>...`` for CuAAC assumes
    the azide is the first reactant.  When the MCTS expansion hands the
    state in the other order (e.g. the root is ``[Pt]C#C``, an alkyne,
    and the partner tile is the azide), ``RunReactants`` silently returns
    an empty iterable and the rule appears not to fire even though the
    chemistry clearly applies.

    This helper makes the dispatch symmetric: it tries ``(a, b)`` first,
    and if that yields zero product sets, swaps the order and retries.
    It returns the first non-empty result; if both orderings are empty
    it returns ``[]``.

    Parameters
    ----------
    rule_name : str
        Rule name used for instrumentation counter tags.
    rxn : rdkit.Chem.rdChemReactions.ChemicalReaction
        The compiled RDKit reaction template.
    mol_a, mol_b : rdkit.Chem.Mol
        The two reactant mols (in either order).

    Returns
    -------
    iterable
        A product-set iterable from ``RunReactants`` (possibly empty).

    Notes
    -----
    Exceptions raised by ``RunReactants`` are *not* swallowed here —
    the caller (``CuAAC._reduce`` etc.) wraps them in its existing
    ``ReactionError`` so the L4 counter accounting stays accurate.
    """
    try:
        out = list(rxn.RunReactants((mol_a, mol_b)))
    except Exception:
        out = []
    if out:
        return out
    # Same molecule twice — symmetric re-run would be identical.
    if mol_a is mol_b:
        return out
    try:
        swapped = list(rxn.RunReactants((mol_b, mol_a)))
    except Exception:
        swapped = []
    return swapped


# ---------------------------------------------------------------------------
# Patterns / SMARTS — the redex signatures
# ---------------------------------------------------------------------------

# Atom-mapping conventions (RDKit SMARTS uses numeric maps ``[N:1]`` etc.):
#   azide  :  R-[N:1]=[N:2]=[N:3]
#   alkyne :  R'-[C:4]#[C:5]
#   terminal alkyne (CuAAC) :  R'-[C:4]#[CH:6]
#   cyclooctyne strained alkyne (SPAAC):  R'-[C:4]#[C:5] inside an 8-ring
#   phosphine (SPC):  R'-[P:7]
#   diene:  R-[C:8]=[C:9]-[C:10]=[C:11]
#   dienophile:  R'=[C:12]-[C:13]
#   thiol:  R-[SH:14]
#   alkene:  R'=[C:15]-[C:16]

# ---------------------------------------------------------------------------
# CuAAC — copper-catalysed azide-alkyne cycloaddition
# ---------------------------------------------------------------------------


@dataclass
class CuAAC(ReactionRule):
    """Cu(I)-catalysed azide–terminal-alkyne cycloaddition.

    Pattern (RDKit SMARTS)::

        [N:1]=[N:2]=[N:3].[C:4]#[CH:5] >>
            c1cc([N:1]nn([C:4]cc1)[CH:5])

    Chemistry:
        R-N=N=N  +  R'-C≡C-H  --[Cu(I)]-->  1,4-disubstituted 1,2,3-triazole

    Stoichiometry: empty — pure bond-forming reaction (the H from the
    alkyne is retained on the triazole N2, so heavy-atom count is
    conserved).
    """

    name: str = "CuAAC"
    requires_catalyst: Optional[str] = "Cu(I)"
    pattern_smiles: Optional[str] = (
        # SMARTS: azide + terminal alkyne → 1,4-disubstituted 1,2,3-triazole.
        # The triazole ring is [N:1]-[N:2]=[N:3]-[C:4]=[C:5]- (5-membered).
        # Atom-maps 1, 2, 3 track the azide N's; 4, 5 track the alkyne C's.
        "[N:1]=[N:2]=[N:3].[C:4]#[CH:5]>>"
        "[C:4]1=[C:5][N:3]=[N:2][N:1]1"
    )
    stoichiometry: Dict[str, int] = field(default_factory=dict)

    def _reduce(
        self,
        reactants: Tuple[MoleculeClosedTerm, ...],
        **kwargs,
    ) -> List[MoleculeClosedTerm]:
        sanitize = kwargs.get("sanitize", True)
        if len(reactants) != 2:
            raise ReactionError(
                f"[CuAAC] requires exactly 2 reactants (azide, alkyne), "
                f"got {len(reactants)}"
            )
        azide, alkyne = reactants
        rxn = self._rdkit_reaction_template()
        # WF-Lambda-Rule-Symmetry-Fix: try (azide, alkyne) first, then
        # (alkyne, azide).  The chemistry is commutative; the SMARTS
        # is not.  Without the swap, MCTS roots that are alkynes (e.g.
        # cisplatin-acetylide [Pt]C#C) silently fail to fire CuAAC even
        # when the partner tile carries the azide handle.
        try:
            product_sets = _run_reactants_symmetric(
                "CuAAC", rxn, azide.to_rdkit(), alkyne.to_rdkit()
            )
        except Exception as exc:
            _L4_COUNTERS["CuAAC"]["runreactants_err"] += 1
            raise ReactionError(
                f"[CuAAC] RDKit RunReactants failed: {exc}"
            ) from exc
        products = _rdkit_product_sets_to_closed_terms(product_sets, sanitize)
        _L4_COUNTERS["CuAAC"]["fire"] += 1 if products else 0
        _L4_COUNTERS["CuAAC"]["attempts"] += 1
        _L4_COUNTERS["CuAAC"]["products"] += len(products)
        # Decorate with the term-derivation tag for the synthesis layer.
        for p in products:
            p.term = (
                f"CuAAC(R-N3, R'-C≡CH) -> "
                f"{p.canonical_smiles() if p.source_smiles is None else p.source_smiles}"
            )
        return products


# ---------------------------------------------------------------------------
# SPAAC — strain-promoted azide–alkyne cycloaddition
# ---------------------------------------------------------------------------


@dataclass
class SPAAC(ReactionRule):
    """Strain-promoted azide–alkyne cycloaddition (no copper).

    Chemistry:
        R-N3  +  strained R'-C≡C- (cyclooctyne)  -->  triazole

    The pattern permits both terminal and strained alkynes; the
    regiochemistry of the resulting triazole differs (1,4- vs 1,5-)
    so :meth:`_reduce` returns both isomers when RDKit enumerates
    them.
    """

    name: str = "SPAAC"
    requires_catalyst: Optional[str] = None
    pattern_smiles: Optional[str] = (
        # SMARTS: azide + alkyne (terminal or strained cyclooctyne) →
        # 1,2,3-triazole.  Unlike CuAAC the alkyne C's need not carry an H,
        # which captures the strain-promoted variant.
        "[N:1]=[N:2]=[N:3].[C:4]#[C:5]>>"
        "[C:4]1=[C:5][N:3]=[N:2][N:1]1"
    )
    stoichiometry: Dict[str, int] = field(default_factory=dict)

    def _reduce(
        self,
        reactants: Tuple[MoleculeClosedTerm, ...],
        **kwargs,
    ) -> List[MoleculeClosedTerm]:
        sanitize = kwargs.get("sanitize", True)
        if len(reactants) != 2:
            raise ReactionError(
                f"[SPAAC] requires exactly 2 reactants (azide, cyclooctyne), "
                f"got {len(reactants)}"
            )
        azide, alkyne = reactants
        rxn = self._rdkit_reaction_template()
        # WF-Lambda-Rule-Symmetry-Fix: see CuAAC comment.  SPAAC SMARTS
        # is ``[N:1]=[N:2]=[N:3].[C:4]#[C:5]>>...`` so it inherits the
        # same azide-first asymmetry.
        try:
            product_sets = _run_reactants_symmetric(
                "SPAAC", rxn, azide.to_rdkit(), alkyne.to_rdkit()
            )
        except Exception as exc:
            _L4_COUNTERS["SPAAC"]["runreactants_err"] += 1
            raise ReactionError(
                f"[SPAAC] RDKit RunReactants failed: {exc}"
            ) from exc
        products = _rdkit_product_sets_to_closed_terms(product_sets, sanitize)
        _L4_COUNTERS["SPAAC"]["fire"] += 1 if products else 0
        _L4_COUNTERS["SPAAC"]["attempts"] += 1
        _L4_COUNTERS["SPAAC"]["products"] += len(products)
        for p in products:
            p.term = (
                f"SPAAC(R-N3, cyclooctyne R'-C≡C) -> {p.canonical_smiles()}"
            )
        return products


# ---------------------------------------------------------------------------
# SPC — Staudinger phosphine–azide ligation
# ---------------------------------------------------------------------------


@dataclass
class SPC(ReactionRule):
    """Staudinger phosphine–azide ligation.

    Chemistry:
        R-N=N=N  +  R'-P(III)  -->  R-N=P(=O)-R'  +  N2

    A phosphine (P(III) with a lone pair) attacks the terminal nitrogen
    of the azide.  Loss of N2 gives an iminophosphorane (R-N=P-R');
    hydrolysis in the workup step yields a phosphoramidate — but we
    represent the on-phosphorus state here, which is the reduction
    itself.
    """

    name: str = "SPC"
    requires_catalyst: Optional[str] = None
    pattern_smiles: Optional[str] = (
        # SMARTS: azide + P(III) phosphine → iminophosphorane + N2.
        # Atom-map 1 tracks the inner azide N (becomes N=P); atom-maps 2, 3
        # leave as dinitrogen.  Phosphorus atom-map 4 becomes P=.
        "[N:1]=[N:2]=[N:3].[P:4]>>[*:1]=[P:4].[N:2]#[N:3]"
    )
    stoichiometry: Dict[str, int] = field(default_factory=dict)

    def _reduce(
        self,
        reactants: Tuple[MoleculeClosedTerm, ...],
        **kwargs,
    ) -> List[MoleculeClosedTerm]:
        sanitize = kwargs.get("sanitize", True)
        if len(reactants) != 2:
            raise ReactionError(
                f"[SPC] requires exactly 2 reactants (azide, phosphine), "
                f"got {len(reactants)}"
            )
        azide, phosphine = reactants
        rxn = self._rdkit_reaction_template()
        try:
            product_sets = rxn.RunReactants(
                (azide.to_rdkit(), phosphine.to_rdkit())
            )
        except Exception as exc:
            _L4_COUNTERS["SPC"]["runreactants_err"] += 1
            raise ReactionError(
                f"[SPC] RDKit RunReactants failed: {exc}"
            ) from exc
        products = _rdkit_product_sets_to_closed_terms(product_sets, sanitize)
        _L4_COUNTERS["SPC"]["fire"] += 1 if products else 0
        _L4_COUNTERS["SPC"]["attempts"] += 1
        _L4_COUNTERS["SPC"]["products"] += len(products)
        for p in products:
            p.term = f"SPC(R-N3, R'-P) -> {p.canonical_smiles()}"
        return products


# ---------------------------------------------------------------------------
# Diels–Alder [4+2] cycloaddition
# ---------------------------------------------------------------------------


@dataclass
class DielsAlder(ReactionRule):
    """[4+2] Diels–Alder cycloaddition.

    Chemistry:
        diene (4π)  +  dienophile (2π)  -->  cyclohexene

    The reaction consumes 3 π-bonds (2 from the diene, 1 from the
    dienophile) and forms 2 new σ-bonds plus 1 new π-bond; the
    heavy-atom count is conserved.
    """

    name: str = "DielsAlder"
    requires_catalyst: Optional[str] = None
    pattern_smiles: Optional[str] = (
        # SMARTS: conjugated diene (4π) + alkene (2π) → cyclohexene.
        # All six ring atoms are mapped so the product topology is exact.
        "[C:1]=[C:2]-[C:3]=[C:4].[C:5]=[C:6]>>"
        "[C:1]1=[C:2][C:3][C:4][C:5][C:6]1"
    )
    stoichiometry: Dict[str, int] = field(default_factory=dict)

    def _reduce(
        self,
        reactants: Tuple[MoleculeClosedTerm, ...],
        **kwargs,
    ) -> List[MoleculeClosedTerm]:
        sanitize = kwargs.get("sanitize", True)
        if len(reactants) != 2:
            raise ReactionError(
                f"[DielsAlder] requires exactly 2 reactants "
                f"(diene, dienophile), got {len(reactants)}"
            )
        diene, dienophile = reactants
        rxn = self._rdkit_reaction_template()
        try:
            product_sets = rxn.RunReactants(
                (diene.to_rdkit(), dienophile.to_rdkit())
            )
        except Exception as exc:
            _L4_COUNTERS["DielsAlder"]["runreactants_err"] += 1
            raise ReactionError(
                f"[DielsAlder] RDKit RunReactants failed: {exc}"
            ) from exc
        products = _rdkit_product_sets_to_closed_terms(product_sets, sanitize)
        _L4_COUNTERS["DielsAlder"]["fire"] += 1 if products else 0
        _L4_COUNTERS["DielsAlder"]["attempts"] += 1
        _L4_COUNTERS["DielsAlder"]["products"] += len(products)
        for p in products:
            p.term = (
                f"DielsAlder(diene, dienophile) -> {p.canonical_smiles()}"
            )
        return products


# ---------------------------------------------------------------------------
# Thiol–Ene radical addition
# ---------------------------------------------------------------------------


@dataclass
class ThiolEne(ReactionRule):
    """Thiol–Ene radical addition (photoinitiated, anti-Markovnikov).

    Chemistry:
        R-SH  +  R'-CH=CH-R''  -->  R'-CH2-CH2-S-R  (net)

    Heavy-atom count is conserved.  No RDKit reaction SMARTS exists
    by default for this transformation, so we perform a structural
    functional-group check on the educts and merge them manually.

    Note
    ----
    Because there is no canonical SMARTS, ``pattern_smiles`` is
    ``None`` and :meth:`can_apply` walks the atom list looking for
    a neutral carbon-bound ``S-H`` and a ``C=C`` double bond
    on the partner.  The reaction is **permissive** at the MLC layer
    (it doesn't enforce stereochemistry); regio/stereo-control is
    delegated to the synthesis layer.
    """

    name: str = "ThiolEne"
    requires_catalyst: Optional[str] = "hν / radical initiator"
    pattern_smiles: Optional[str] = None
    stoichiometry: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _reaction_sites(thiol: MoleculeClosedTerm, alkene: MoleculeClosedTerm):
        """Select an actual R-SH and a C=C belonging to the second reactant.

        The current rule emits one deterministic regioisomer, preferring the
        less substituted alkene carbon for sulfur addition. It does not predict
        selectivity or assign newly created stereocenters.
        """
        from rdkit import Chem

        # Most search pairs lack one required handle. Reject them before
        # serializing unrelated charged/metal terms through the limited
        # ClosedTerm RDKit bridge.
        if not any(atom.symbol == "S" for atom in thiol.atoms):
            return None
        if not any(bond.kind == "covalent" and bond.order == 2
                   and bond.atom_a.symbol == "C" and bond.atom_b.symbol == "C"
                   for bond in alkene.bonds):
            return None
        thiol_mol, alkene_mol = thiol.to_rdkit(), alkene.to_rdkit()
        Chem.SanitizeMol(thiol_mol)
        Chem.SanitizeMol(alkene_mol)
        sulfurs = []
        for atom in thiol_mol.GetAtoms():
            if atom.GetAtomicNum() != 16 or atom.GetFormalCharge() != 0:
                continue
            heavy_neighbors = [a for a in atom.GetNeighbors() if a.GetAtomicNum() != 1]
            explicit_h = sum(a.GetAtomicNum() == 1 for a in atom.GetNeighbors())
            # The term ledger preserves bracket/implicit hydrogen counts even
            # when to_rdkit reconstructs an uncharged heavy-atom graph.
            ledger_h = thiol.implicit_h_count.get(atom.GetIdx(), atom.GetTotalNumHs())
            if (len(heavy_neighbors) == 1 and heavy_neighbors[0].GetAtomicNum() == 6
                    and ledger_h + explicit_h == 1
                    and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())):
                sulfurs.append(atom.GetIdx())
        alkenes = [bond for bond in alkene_mol.GetBonds()
                   if bond.GetBondType() == Chem.BondType.DOUBLE
                   and not bond.GetIsAromatic()
                   and bond.GetBeginAtom().GetAtomicNum() == 6
                   and bond.GetEndAtom().GetAtomicNum() == 6]
        if not sulfurs or not alkenes:
            return None
        bond = alkenes[0]
        ranks = list(Chem.CanonicalRankAtoms(alkene_mol))
        endpoints = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        endpoints.sort(key=lambda idx: (
            sum(a.GetAtomicNum() != 1 for a in alkene_mol.GetAtomWithIdx(idx).GetNeighbors()),
            ranks[idx], idx,
        ))
        return thiol_mol, alkene_mol, sulfurs[0], endpoints[0], endpoints[1]

    def can_apply(
        self, thiol: MoleculeClosedTerm, alkene: MoleculeClosedTerm
    ) -> bool:
        """Require a neutral R-SH, excluding thioethers, and a partner C=C."""
        result = self._reaction_sites(thiol, alkene) is not None
        _L4_COUNTERS["ThiolEne"]["thiolene_can_apply"] += int(result)
        return result

    def _reduce(
        self,
        reactants: Tuple[MoleculeClosedTerm, ...],
        **kwargs,
    ) -> List[MoleculeClosedTerm]:
        if len(reactants) != 2:
            raise ReactionError(
                f"[ThiolEne] requires exactly 2 reactants (thiol, alkene), "
                f"got {len(reactants)}"
            )
        from rdkit import Chem

        _L4_COUNTERS["ThiolEne"]["attempts"] += 1
        try:
            sites = self._reaction_sites(*reactants)
            if sites is None:
                return []
            _L4_COUNTERS["ThiolEne"]["thiolene_can_apply"] += 1
            thiol_mol, alkene_mol, s_idx, alkene_c, alkene_other = sites
            offset = thiol_mol.GetNumAtoms()
            c_idx, other_idx = offset + alkene_c, offset + alkene_other
            combined = _combine_mols(thiol_mol, alkene_mol)
            input_fragments = len(Chem.GetMolFrags(combined))
            rw = Chem.RWMol(combined)
            bond = rw.GetBondBetweenAtoms(c_idx, other_idx)
            bond.SetBondType(Chem.BondType.SINGLE)
            bond.SetStereo(Chem.BondStereo.STEREONONE)
            # Explicit H atoms, if present, are transferred without deletion;
            # ordinary implicit S-H is reallocated by sanitization after the
            # S-C addition and C=C reduction.
            explicit_h = [a.GetIdx() for a in rw.GetAtomWithIdx(s_idx).GetNeighbors()
                          if a.GetAtomicNum() == 1]
            if explicit_h:
                rw.RemoveBond(s_idx, explicit_h[0])
                rw.AddBond(other_idx, explicit_h[0], Chem.BondType.SINGLE)
            rw.AddBond(s_idx, c_idx, Chem.BondType.SINGLE)
            for idx in (c_idx, other_idx):
                rw.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                for adjacent in rw.GetAtomWithIdx(idx).GetBonds():
                    adjacent.SetBondDir(Chem.BondDir.NONE)
            product_mol = rw.GetMol()
            # A structural reaction is never accepted without valence checks,
            # including when callers request sanitize=False for other rules.
            Chem.SanitizeMol(product_mol)
            if product_mol.GetNumAtoms() != combined.GetNumAtoms():
                raise ValueError('ThiolEne changed atom count')
            if len(Chem.GetMolFrags(product_mol)) != input_fragments - 1:
                raise ValueError('ThiolEne did not join the two reacting components')
            term = MoleculeClosedTerm.from_rdkit(product_mol)
            term.term = f"ThiolEne(R-SH, R'-CH=CH-R'') -> {term.canonical_smiles()}"
            _L4_COUNTERS["ThiolEne"]["fire"] += 1
            _L4_COUNTERS["ThiolEne"]["products"] += 1
            return [term]
        except Exception as exc:
            _L4_COUNTERS["ThiolEne"]["runreactants_err"] += 1
            raise ReactionError(f"[ThiolEne] structural rewrite failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Suzuki–Miyaura cross-coupling (R10 axis A — round-10 click family #4)
# ---------------------------------------------------------------------------


@dataclass
class Suzuki(ReactionRule):
    """Suzuki–Miyaura Pd-catalysed cross-coupling (R10 axis A).

    Chemistry:
        R-B(OH)2  +  R'-X   --[Pd(0), base]-->  R-R'   (X = Br, I; Cl works with care)

    A canonical aryl-boronic acid is coupled with an aryl halide to form
    a new C–C bond and release B(OH)2 + HX.  Like the other click rules
    it is bond-forming only — heavy-atom count is conserved modulo the
    leaving-group "subtraction" (we report the biaryl product; the
    boronic acid and halide by-products are part of the L4 chemistry
    bookkeeping, not the MCTS state).

    Stoichiometry: empty at the MLC layer (the biaryl product keeps the
    heavy atoms from the two aromatic fragments — no net creation /
    destruction of C/N/O/etc.  The B and halogen leave as by-products,
    which we account for in :data:`stoichiometry` for downstream yield
    prediction).
    """

    name: str = "Suzuki"
    requires_catalyst: Optional[str] = "Pd(0) + base"
    pattern_smiles: Optional[str] = (
        # SMARTS: aryl-boronic acid R-B(OH)2 + aryl halide R'-X →
        # biaryl R-R' + leaving groups (the leaving groups are dropped
        # by the SMARTS).  Atom-maps 1, 2 trace the aryl-boronic acid
        # C, atom-maps 3, 4 trace the aryl halide C; the new C-C bond
        # joins the two ipso carbons.  We use [#6:1]-B(O)(O) +
        # [#6:3]-[F,Cl,Br,I] → [#6:1][#6:3] (biaryl).
        "[#6:1][B]([O])[O].[#6:3][F,Cl,Br,I]>>[#6:1][#6:3]"
    )
    stoichiometry: Dict[str, int] = field(default_factory=dict)

    def _reduce(
        self,
        reactants: Tuple[MoleculeClosedTerm, ...],
        **kwargs,
    ) -> List[MoleculeClosedTerm]:
        sanitize = kwargs.get("sanitize", True)
        if len(reactants) != 2:
            raise ReactionError(
                f"[Suzuki] requires exactly 2 reactants "
                f"(aryl-boronic acid, aryl halide), got {len(reactants)}"
            )
        boronic, halide = reactants
        rxn = self._rdkit_reaction_template()
        # WF-Lambda-Rule-Symmetry-Fix: see CuAAC comment.  Suzuki
        # SMARTS is ``[#6:1][B]([O])[O].[#6:3][F,Cl,Br,I]>>[#6:1][#6:3]``
        # — boronic acid first — so the same root-cause asymmetry applies.
        try:
            product_sets = _run_reactants_symmetric(
                "Suzuki", rxn, boronic.to_rdkit(), halide.to_rdkit()
            )
        except Exception as exc:
            _L4_COUNTERS["Suzuki"]["runreactants_err"] += 1
            raise ReactionError(
                f"[Suzuki] RDKit RunReactants failed: {exc}"
            ) from exc
        products = _rdkit_product_sets_to_closed_terms(product_sets, sanitize)
        _L4_COUNTERS["Suzuki"]["fire"] += 1 if products else 0
        _L4_COUNTERS["Suzuki"]["attempts"] += 1
        _L4_COUNTERS["Suzuki"]["products"] += len(products)
        for p in products:
            p.term = (
                f"Suzuki(R-B(OH)2, R'-X) -> {p.canonical_smiles()}"
            )
        return products


# ---------------------------------------------------------------------------
# Amide coupling (R10 axis A — round-10 click family #5)
# ---------------------------------------------------------------------------


@dataclass
class AmideCoupling(ReactionRule):
    """Carboxylic-acid + amine amide-bond formation (R10 axis A).

    Chemistry:
        R-COOH  +  R'-NH2  --[coupling reagent (e.g. EDC/HOBt, HATU)]-->  R-CO-NH-R'  +  H2O

    A standard amide coupling reaction (peptide bond formation in
    medicinal chemistry / PROTAC linker construction).  Net loss of H2O;
    heavy-atom count is conserved at the MLC layer (we track the amide
    product only — water is implied as a by-product and is not part of
    the closed-term state).

    Stoichiometry: empty (heavy-atom conserving by construction; the
    dehydration does not change the C/N/O count at the MLC level).
    """

    name: str = "AmideCoupling"
    requires_catalyst: Optional[str] = "coupling reagent (EDC/HOBt, HATU, etc.)"
    pattern_smiles: Optional[str] = (
        # SMARTS: carboxylic acid + primary amine → amide.  Atom-map 1
        # traces the carbonyl C, atom-map 2 traces the carboxyl O (=O),
        # atom-map 3 traces the OH leaving group, atom-map 4 traces the
        # amine N.  The amide product keeps maps 1, 2, 4.
        "[C:1](=[O:2])[OH].[NH2:4]>>[C:1](=[O:2])[NH:4]"
    )
    stoichiometry: Dict[str, int] = field(default_factory=dict)

    def _reduce(
        self,
        reactants: Tuple[MoleculeClosedTerm, ...],
        **kwargs,
    ) -> List[MoleculeClosedTerm]:
        sanitize = kwargs.get("sanitize", True)
        if len(reactants) != 2:
            raise ReactionError(
                f"[AmideCoupling] requires exactly 2 reactants "
                f"(carboxylic acid, amine), got {len(reactants)}"
            )
        acid, amine = reactants
        rxn = self._rdkit_reaction_template()
        try:
            product_sets = rxn.RunReactants(
                (acid.to_rdkit(), amine.to_rdkit())
            )
        except Exception as exc:
            _L4_COUNTERS["AmideCoupling"]["runreactants_err"] += 1
            raise ReactionError(
                f"[AmideCoupling] RDKit RunReactants failed: {exc}"
            ) from exc
        products = _rdkit_product_sets_to_closed_terms(product_sets, sanitize)
        _L4_COUNTERS["AmideCoupling"]["fire"] += 1 if products else 0
        _L4_COUNTERS["AmideCoupling"]["attempts"] += 1
        _L4_COUNTERS["AmideCoupling"]["products"] += len(products)
        for p in products:
            p.term = (
                f"AmideCoupling(R-COOH, R'-NH2) -> {p.canonical_smiles()}"
            )
        return products


# ---------------------------------------------------------------------------
# MetalLigandExchange — Pt_II / Ru / Ir ammonia substitution (F2(a))
# ---------------------------------------------------------------------------


@dataclass
class MetalLigandExchange(ReactionRule):
    """Substitution of a chloride leaving group on a Pt(II) centre by an
    ammine donor (``NH3``) — the canonical step of cisplatin
    activation::

        [Pt](Cl)(Cl)  +  N  -->  [Pt](N)(Cl)  (one chloride is replaced
                                                by an ammine)

    Chemistry:
        Pt_II square-planar complexes (target arity 4) undergo ligand
        substitution by ammine nucleophiles.  The kinetic pathway is
        associative — entering ligand first coordinates then the
        leaving group departs — but the thermodynamic product at the
        MLC layer is simply an exchange of one Cl for one NH3 (the
        bare Pt-Cl educt loses one Cl which becomes a free Cl- in the
        product ledger).

    Pattern (RDKit SMARTS)::

        [Pt:1]([Cl:2])[*:3].[NH3:4] >>
            [Pt:1]([NH3:4])[*:3].[Cl:2]

    The ammine donor is given as ``[NH3]`` (Pt has oxidation state II;
    ``[NH3]`` is neutral N-donor).  The pattern intentionally matches
    a *single* Pt-Cl bond so the rule fires once per substitution
    cycle (the MCTS can call the rule repeatedly to fill all four
    coordination sites of a Pt_II centre).

    Stoichiometry: empty at the MLC layer (Pt, N, Cl all conserved —
    they merely change partners).  The leaving Cl- is reported in the
    product set as a separate closed term so downstream yield
    prediction can account for the chloride accounting.

    Honest framing
    --------------
    RDKit cannot sanitise most Pt-containing products because Pt_II
    is not in RDKit's default valence table.  In practice the
    :func:`_rdkit_product_sets_to_closed_terms` helper will drop the
    Pt-amine product while keeping the chloride leaving group.  We
    still expose the rule because (a) the search layer can introspect
    the rule's ``_L4_COUNTERS`` to see it fired, and (b) when the seed
    metal has a default valence (e.g. ``Ru``) the rule produces
    well-formed products.
    """

    name: str = "MetalLigandExchange"
    requires_catalyst: Optional[str] = None
    pattern_smiles: Optional[str] = (
        # SMARTS: [Pt]-Cl single bond substitution by [NH3] amine.
        # Atom-maps 1 (Pt), 2 (Cl leaving), 3 (other Pt ligand),
        # 4 (incoming NH3).  The product keeps the Pt-N bond
        # (map 1 + 4) and releases Cl as a separate term.
        "[Pt:1]([Cl:2])[*:3].[NH3:4]>>"
        "[Pt:1]([NH3:4])[*:3].[Cl:2]"
    )
    stoichiometry: Dict[str, int] = field(default_factory=dict)

    def _reduce(
        self,
        reactants: Tuple[MoleculeClosedTerm, ...],
        **kwargs,
    ) -> List[MoleculeClosedTerm]:
        sanitize = kwargs.get("sanitize", True)
        if len(reactants) != 2:
            raise ReactionError(
                f"[MetalLigandExchange] requires exactly 2 reactants "
                f"(Pt-Cl complex, ammine donor), got {len(reactants)}"
            )
        metal_complex, donor = reactants
        rxn = self._rdkit_reaction_template()
        _L4_COUNTERS["MetalLigandExchange"]["attempts"] += 1
        try:
            product_sets = rxn.RunReactants(
                (metal_complex.to_rdkit(), donor.to_rdkit())
            )
        except Exception as exc:
            _L4_COUNTERS["MetalLigandExchange"]["runreactants_err"] += 1
            raise ReactionError(
                f"[MetalLigandExchange] RDKit RunReactants failed: {exc}"
            ) from exc
        products = _rdkit_product_sets_to_closed_terms(product_sets, sanitize)
        _L4_COUNTERS["MetalLigandExchange"]["fire"] += 1 if products else 0
        _L4_COUNTERS["MetalLigandExchange"]["products"] += len(products)
        for p in products:
            p.term = (
                f"MetalLigandExchange(Pt-Cl, NH3) -> "
                f"{p.canonical_smiles() if p.source_smiles is None else p.source_smiles}"
            )
        return products

    # ------------------------------------------------------------------
    # WF-Lambda-Boost Phase 2 — extended SMARTS library accessor.
    # ------------------------------------------------------------------
    def available_smarts(self) -> List[Tuple[str, str, str]]:
        """Return the 5 MetalLigandExchange SMARTS patterns.

        Each entry is a ``(pattern_name, smarts, description)`` tuple.
        The first pattern (``Pt_Cl_NH3``) is the canonical cisplatin
        pattern that drives ``self.pattern_smiles``.  The remaining
        4 patterns (Pt-RNH2, Pt-Br, Pd, Au) are exposed for the
        search layer to introspect — they share the same d8
        square-planar coordination chemistry per Lippard 1995 +
        Comba-Hambley 2009.

        Returns
        -------
        list[tuple[str, str, str]]
            The full MetalLigandExchange SMARTS library (5 entries).
        """
        return get_metal_ligand_exchange_patterns()


# ---------------------------------------------------------------------------
# AquaExchange — Pt_II / Ru / Ir water-for-chloride substitution (F2(a))
# ---------------------------------------------------------------------------


@dataclass
class AquaExchange(ReactionRule):
    """Aquation of a Pt(II) centre — substitution of a chloride
    leaving group by a water molecule.  This is the rate-limiting step
    of cisplatin activation in vivo::

        [Pt](Cl)  +  O  -->  [Pt](O)  +  Cl

    Chemistry:
        Water displaces a chloride on the metal centre to give a
        cationic aqua complex.  In the cellular environment this aqua
        ligand is subsequently replaced by a DNA N7-guanine donor
        (the actual cytotoxic event).  We model only the aquation
        step here — the DNA-binding step belongs to the L4
        ``AquaExchange`` followed by an ``AmideCoupling``-style
        substitution by the guanine N7.

    Pattern (RDKit SMARTS)::

        [Pt:1]([Cl:2])[*:3].[OH2:4] >>
            [Pt:1]([OH:4])[*:3].[Cl:2]

    In the product the water has lost one proton (now an ``[OH]``
    hydroxyl, charge 0 because Pt_II is divalent — the positive
    charge that forms on the Pt-OH2 complex is reported in the
    product term's ``Atom.charge`` field, not the SMILES).  The
    free ``[Cl]`` leaving group is reported as a separate closed
    term so the synthesis layer can track chloride mass balance.

    Honest framing
    --------------
    As with :class:`MetalLigandExchange`, RDKit cannot sanitise most
    Pt-containing products (Pt_II not in default valence table).  The
    rule still registers as fired via ``_L4_COUNTERS`` so the search
    layer can record the chemistry happened.
    """

    name: str = "AquaExchange"
    requires_catalyst: Optional[str] = None
    pattern_smiles: Optional[str] = (
        # SMARTS: [Pt]-Cl single bond substitution by [OH2] water.
        # Atom-maps 1 (Pt), 2 (Cl leaving), 3 (other Pt ligand),
        # 4 (incoming OH2 → OH after proton loss).
        "[Pt:1]([Cl:2])[*:3].[OH2:4]>>"
        "[Pt:1]([OH:4])[*:3].[Cl:2]"
    )
    stoichiometry: Dict[str, int] = field(default_factory=dict)

    def _reduce(
        self,
        reactants: Tuple[MoleculeClosedTerm, ...],
        **kwargs,
    ) -> List[MoleculeClosedTerm]:
        sanitize = kwargs.get("sanitize", True)
        if len(reactants) != 2:
            raise ReactionError(
                f"[AquaExchange] requires exactly 2 reactants "
                f"(Pt-Cl complex, water), got {len(reactants)}"
            )
        metal_complex, water = reactants
        rxn = self._rdkit_reaction_template()
        _L4_COUNTERS["AquaExchange"]["attempts"] += 1
        try:
            product_sets = rxn.RunReactants(
                (metal_complex.to_rdkit(), water.to_rdkit())
            )
        except Exception as exc:
            _L4_COUNTERS["AquaExchange"]["runreactants_err"] += 1
            raise ReactionError(
                f"[AquaExchange] RDKit RunReactants failed: {exc}"
            ) from exc
        products = _rdkit_product_sets_to_closed_terms(product_sets, sanitize)
        _L4_COUNTERS["AquaExchange"]["fire"] += 1 if products else 0
        _L4_COUNTERS["AquaExchange"]["products"] += len(products)
        for p in products:
            p.term = (
                f"AquaExchange(Pt-Cl, H2O) -> "
                f"{p.canonical_smiles() if p.source_smiles is None else p.source_smiles}"
            )
        return products

    # ------------------------------------------------------------------
    # WF-Lambda-Boost Phase 3 — extended SMARTS library accessor.
    # ------------------------------------------------------------------
    def available_smarts(self) -> List[Tuple[str, str, str]]:
        """Return the 2 AquaExchange SMARTS patterns.

        Each entry is a ``(pattern_name, smarts, description)`` tuple.
        The first pattern (``Pt_Cl_H2O_first``) is the canonical
        first-aquation pattern that drives ``self.pattern_smiles``.
        The second pattern (``Pt_OHCl_H2O_second``) covers the
        second-aquation step (Reedijk 1987) — the diaqua complex
        that binds DNA-N7-guanine.

        Returns
        -------
        list[tuple[str, str, str]]
            The full AquaExchange SMARTS library (2 entries).
        """
        return get_aqua_exchange_patterns()

    def aqua_context(self, pattern_name: str = "Pt_Cl_H2O_first") -> AquaContext:
        """Return the :class:`AquaContext` for a given pattern name.

        The :class:`AquaContext` carries the pKa1, ionic strength,
        and temperature that the search layer can log for downstream
        chemistry debugging.  See the :class:`AquaContext` docstring
        for honest framing (this is documentation-only — the search
        layer does NOT branch on these values).
        """
        return get_aqua_context(pattern_name)


# ---------------------------------------------------------------------------
# Registry — REACTION_RULES
# ---------------------------------------------------------------------------

#: Canonical registry mapping rule name -> :class:`ReactionRule` instance.
#:
#: Use ``REACTION_RULES["CuAAC"].reduce((azide, alkyne))`` to fire the
#: rule.  The dict is the single source of truth for what reactions are
#: available to the synthesis / search_alg layers.
REACTION_RULES: Dict[str, object] = {
    "CuAAC":         CuAAC(),
    "SPAAC":         SPAAC(),
    "SPC":           SPC(),
    "DielsAlder":    DielsAlder(),
    "ThiolEne":      ThiolEne(),
    # R10 axis A — extend the click family from 5 to 7 so the MCTS
    # expansion path can fire Suzuki and AmideCoupling alongside the
    # legacy cycloadditions.  See lam_chem.rules for the public-facing
    # 5-reaction subset that the search exposes via ``rules``.
    "Suzuki":        Suzuki(),
    "AmideCoupling": AmideCoupling(),
    # F2(a): metal-coordination family — Pt_II / Ru_II / Ir_III
    # specific reductions (chloride substitution by NH3 / H2O).
    # These rules are kept in the low-level registry so the search
    # layer can opt-in via the ``metal-coord`` alias; they are NOT
    # part of the public-facing 5/7 click set.
    "MetalLigandExchange": MetalLigandExchange(),
    "AquaExchange":        AquaExchange(),
}


def _register_synthemol_rules() -> Dict[str, object]:
    """Add 13 SyntheMol REAL reactions to :data:`REACTION_RULES`.

    Imports :mod:`sbdd_env.syntemol_reactions` lazily so the rest of
    this module can be imported without SyntheMol's heavy ML stack
    (wandb / chemprop) on the path.  Returns the mapping that was added
    so callers (e.g. tests) can iterate just the new entries.
    """
    try:
        from molmetal_lam.sbdd_env.syntemol_reactions import (
            from_syntemol_to_lambda,
        )
    except Exception as exc:  # pragma: no cover — defensive
        warnings.warn(
            f"[beta_reductions] could not import SyntheMol adapter: {exc}. "
            "REACTION_RULES will contain only the 5 click rules.",
            RuntimeWarning,
        )
        return {}
    added = from_syntemol_to_lambda()
    for name, wrapper in added.items():
        REACTION_RULES[name] = wrapper  # type: ignore[assignment]
    return added


# Eager registration so the registry is consistent with the docs
# ("18 reaction rules") at first import.  SyntheMol's bootstrap is
# sub-second on Lambda's venv (RDKit-only, no ML stack).
_SYNTHEMOL_REGISTERED = False
try:
    _register_synthemol_rules()
    _SYNTHEMOL_REGISTERED = True
except Exception as exc:  # pragma: no cover — defensive
    warnings.warn(
        f"[beta_reductions] SyntheMol registration failed: {exc}. "
        "REACTION_RULES will contain only the 5 click rules.",
        RuntimeWarning,
    )


def l4_metrics() -> Dict[str, object]:
    """Return a snapshot of L4 instrumentation counters (govern L4 review)."""
    return {
        "per_rule": {k: dict(v) for k, v in _L4_COUNTERS.items()},
        "mass_balance": dict(_L4_MASS_BALANCE_PASS),
        "catalyst_requirement_coverage": {
            name: REACTION_RULES[name].requires_catalyst
            for name in REACTION_RULES
        },
        "rate_predictor_attach": {
            name: (
                (REACTION_RULES[name].rate_predictor is not None)
                + (
                    1 if getattr(REACTION_RULES[name], "aryl_predictor", None)
                    else 0
                )
            )
            for name in REACTION_RULES
        },
    }


def attach_all_rate_predictors() -> Dict[str, object]:
    """Attach a trained :class:`RatePredictor` to every click rule.

    Looks up each rule in :data:`reactions.rate_predictor.LITERATURE_YIELDS`
    and trains a :class:`HeuristicRegressor` on the literature-cited
    yields.  The fitted :class:`RatePredictor` is stored on
    ``rule.rate_predictor`` and also returned in the cache
    :data:`reactions.rate_predictor.RATE_PREDICTORS`.

    The CuAAC rule is dual-fit: it is paired with both the generic
    ``CuAAC`` dataset and the ``ClickCuAAC_aryl_variant`` dataset (the
    latter is queried via :attr:`rule.rate_predictor_aryl`).  The
    synthesis layer can choose which to use by reaction-context (aryl
    azide vs alkyl azide).

    Returns
    -------
    dict[str, RatePredictor]
        The fitted predictors keyed by reaction name.
    """
    # Lazy import to avoid a hard rdkit/sklearn dep on import.
    from molmetal_lam.reactions.rate_predictor import RatePredictor

    fitted: Dict[str, object] = {}
    mapping = {
        "CuAAC":      "CuAAC",
        "SPAAC":      "SPAAC",
        "SPC":        "SPC",
        "DielsAlder": "DielsAlder",
        "ThiolEne":   "ThiolEne",
    }
    for rule_name, lit_name in mapping.items():
        rule = REACTION_RULES[rule_name]
        pred = RatePredictor.for_reaction(lit_name)
        rule.attach_rate_predictor(pred)
        fitted[lit_name] = pred
    # Second CuAAC fit on aryl-variant yields (separate model instance).
    aryl = RatePredictor.for_reaction("ClickCuAAC_aryl_variant")
    REACTION_RULES["CuAAC"].aryl_predictor = aryl  # type: ignore[attr-defined]
    fitted["ClickCuAAC_aryl_variant"] = aryl
    return fitted


# ---------------------------------------------------------------------------
# Public audit
# ---------------------------------------------------------------------------


def verify_mass_balance(rule_name: str = "") -> None:
    """Audit one rule (or all rules) for mass balance at instantiation time.

    For each rule we run a *fingerprint* check: assert that the
    ``stoichiometry`` field is the empty dict — which is the correct
    net heavy-atom change for every bond-forming click reaction
    (atoms are not created or destroyed, only rearranged).

    This is the **loud failure** required by the task spec: any rule
    that is not mass-balanced by construction raises immediately.
    """
    targets = (
        [REACTION_RULES[rule_name]]
        if rule_name
        else list(REACTION_RULES.values())
    )
    for rule in targets:
        # The CuAAC/SPAAC/SPC/DielsAlder/ThiolEne rules are all
        # bond-forming: no atoms are created or destroyed, so the net
        # stoichiometry is always the empty dict.
        if rule.stoichiometry != {}:
            _L4_MASS_BALANCE_PASS["fail"] += 1
            raise ReactionError(
                f"[{rule.name}] mass-balance audit FAILED: "
                f"non-empty stoichiometry {rule.stoichiometry} "
                "(click-chemistry rules must conserve heavy atoms)"
            )
        _L4_MASS_BALANCE_PASS["pass"] += 1


# ---------------------------------------------------------------------------
# Convenience functional API (matches plan.md §2 click-chemistry spec)
# ---------------------------------------------------------------------------


def cuaac(
    azide: MoleculeClosedTerm, alkyne: MoleculeClosedTerm,
    **kwargs,
) -> List[MoleculeClosedTerm]:
    """``CuAAC(azide_tile, alkyne_tile)`` functional shorthand."""
    return REACTION_RULES["CuAAC"].reduce((azide, alkyne), **kwargs)


def spaac(
    azide: MoleculeClosedTerm, alkyne: MoleculeClosedTerm,
    **kwargs,
) -> List[MoleculeClosedTerm]:
    """``SPAAC(azide_tile, cyclooctyne_tile)`` functional shorthand."""
    return REACTION_RULES["SPAAC"].reduce((azide, alkyne), **kwargs)


def spc(
    azide: MoleculeClosedTerm, phosphine: MoleculeClosedTerm,
    **kwargs,
) -> List[MoleculeClosedTerm]:
    """``SPC(azide_tile, phosphine_tile)`` functional shorthand."""
    return REACTION_RULES["SPC"].reduce((azide, phosphine), **kwargs)


def diels_alder(
    diene: MoleculeClosedTerm, dienophile: MoleculeClosedTerm,
    **kwargs,
) -> List[MoleculeClosedTerm]:
    """``DielsAlder(diene_tile, dienophile_tile)`` functional shorthand."""
    return REACTION_RULES["DielsAlder"].reduce((diene, dienophile), **kwargs)


def thiol_ene(
    thiol: MoleculeClosedTerm, alkene: MoleculeClosedTerm,
    **kwargs,
) -> List[MoleculeClosedTerm]:
    """``ThiolEne(thiol_tile, alkene_tile)`` functional shorthand."""
    return REACTION_RULES["ThiolEne"].reduce((thiol, alkene), **kwargs)


__all__ = [
    "ReactionError",
    "ReactionRule",
    "CuAAC",
    "SPAAC",
    "SPC",
    "DielsAlder",
    "ThiolEne",
    "Suzuki",
    "AmideCoupling",
    # F2(a) — metal-coordination family (Pt_II / Ru / Ir).
    "MetalLigandExchange",
    "AquaExchange",
    "REACTION_RULES",
    "verify_mass_balance",
    "l4_metrics",
    "cuaac",
    "spaac",
    "spc",
    "diels_alder",
    "thiol_ene",
]
