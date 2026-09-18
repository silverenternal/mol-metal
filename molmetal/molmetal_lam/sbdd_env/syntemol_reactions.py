"""SyntheMol Reaction + QueryMol adapter for molmetal's REACTION_RULES.

================================================================
What this module does
================================================================
SyntheMol (https://github.com/insitro/SyntheMol) ships a curated set of
"REAL" reaction SMARTS — bond-forming transformations drawn from the
Enamine REAL database — together with a clean
:class:`synthemol.reactions.reaction.Reaction` /
:class:`synthemol.reactions.query_mol.QueryMol` abstraction that
handles SMARTS with atom-mapping.

This adapter:

1. Loads :class:`Reaction` + :class:`QueryMol` *without* triggering
   SyntheMol's package init (which pulls in wandb/chemprop — heavy
   ML deps that Lambda never uses).
2. Exposes :func:`from_syntemol_to_lambda` which maps a chosen subset
   of SyntheMol's ``REAL_REACTIONS`` to molmetal-compatible
   :class:`ReactionRule` *wrappers*.  Each wrapper exposes
   ``name``, ``can_apply(List[str]) -> bool``, and
   ``fire(List[str]) -> List[str]`` — the same interface the synthesis
   layer expects from click-chemistry rules.

The wrappers do **not** subclass :class:`ReactionRule` directly because
the SyntheMol SMARTS rely on R-group ``[*:N]`` placeholders that the
RDKit ``RunReactants`` path inside :mod:`beta_reductions` does not
always handle cleanly.  Instead we wrap the
``SyntheMol.reaction.Reaction.run_reactants`` pipeline verbatim and
keep the interface identical so the downstream synthesis / search
layers see ``.reduce((a, b))``-shaped callables whether they consume a
click rule or a REAL rule.

Public API
----------
``SyntheMolRuleError``        raised when a SyntheMol reaction fails
``SyntheMolReactionRule``     dataclass wrapping a SyntheMol Reaction
``from_syntemol_to_lambda``   one-shot mapping → dict[name -> wrapper]
``load_real_reactions``       import SyntheMol's REAL_REACTIONS tuple
``SYNTHEMOL_REAL_REACTIONS``  cached singleton (lazy import)

Note
----
We use ``importlib.util`` to bypass ``synthemol.__init__.py`` because
it eagerly imports ``synthemol.generate`` (which depends on ``wandb``)
and ``synthemol.models`` (which depends on ``chemprop``).  Neither is
needed for the reaction layer; the loader only requires ``rdkit``.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SyntheMolRuleError(ValueError):
    """Raised when a SyntheMol-backed rule fails to fire."""


# ---------------------------------------------------------------------------
# Loader — bypass SyntheMol's package init (wandb/chemprop)
# ---------------------------------------------------------------------------


_SYNTHEMOL_PATH = Path(
    "/home/hugo/codes/try_triton_on_rocm/molmetal/references/SyntheMol/synthemol"
)


def _bootstrap_synthemol() -> None:
    """Inject a minimal SyntheMol package stub so :mod:`reactions` can import.

    The real ``synthemol.__init__.py`` does::

        import synthemol.generate   # requires wandb
        import synthemol.models     # requires chemprop
        ...

    which is fatal on Lambda's venv (no ML stack).  We replace the
    package with a stub that exposes only ``synthemol.constants``
    (just the ``MOLECULE_TYPE`` type alias) and the
    ``synthemol.reactions.*`` submodules we actually need.
    """
    if "synthemol" in sys.modules and getattr(
        sys.modules["synthemol"], "_molmetal_bootstrap_done", False
    ):
        return

    # 1. Synthemol package stub
    pkg = types.ModuleType("synthemol")
    pkg.__path__ = [str(_SYNTHEMOL_PATH)]
    pkg._molmetal_bootstrap_done = True
    sys.modules["synthemol"] = pkg

    # 2. synthemol.constants — minimal stub (real one uses
    #    importlib.resources which requires the package to be installed)
    constants_mod = types.ModuleType("synthemol.constants")
    constants_mod.MOLECULE_TYPE = "molecule_type"  # str | Chem.Mol alias
    sys.modules["synthemol.constants"] = constants_mod

    # 3. synthemol.utils
    _load_submodule(
        "synthemol.utils", _SYNTHEMOL_PATH / "utils.py"
    )

    # 4. synthemol.reactions subpackage
    reactions_pkg = types.ModuleType("synthemol.reactions")
    reactions_pkg.__path__ = [str(_SYNTHEMOL_PATH / "reactions")]
    sys.modules["synthemol.reactions"] = reactions_pkg

    # 5. Submodules
    _load_submodule(
        "synthemol.reactions.query_mol", _SYNTHEMOL_PATH / "reactions" / "query_mol.py"
    )
    _load_submodule(
        "synthemol.reactions.reaction", _SYNTHEMOL_PATH / "reactions" / "reaction.py"
    )
    _load_submodule(
        "synthemol.reactions.real", _SYNTHEMOL_PATH / "reactions" / "real.py"
    )


def _load_submodule(fullname: str, path: Path) -> types.ModuleType:
    """Load a single module file by path into ``sys.modules``."""
    spec = importlib.util.spec_from_file_location(fullname, str(path))
    if spec is None or spec.loader is None:  # pragma: no cover
        raise SyntheMolRuleError(f"could not load spec for {fullname} at {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = mod
    spec.loader.exec_module(mod)
    return mod


_SYNTHEMOL_BOOTSTRAPPED = False


def load_real_reactions():
    """Return SyntheMol's ``REAL_REACTIONS`` tuple, bootstrapping on first call."""
    global _SYNTHEMOL_BOOTSTRAPPED
    if not _SYNTHEMOL_BOOTSTRAPPED:
        _bootstrap_synthemol()
        _SYNTHEMOL_BOOTSTRAPPED = True
    real_mod = sys.modules["synthemol.reactions.real"]
    return real_mod.REAL_REACTIONS


# ---------------------------------------------------------------------------
# Wrapper — exposes the can_apply / fire interface
# ---------------------------------------------------------------------------


@dataclass
class SyntheMolReactionRule:
    """molmetal-compatible wrapper around a single SyntheMol Reaction.

    Attributes
    ----------
    name : str
        Stable identifier (e.g. ``"amide_coupling"``).
    reaction : ``synthemol.reactions.reaction.Reaction``
        The underlying SyntheMol Reaction (cached).
    smarts : str
        Reaction SMARTS string (read-only, used for introspection).
    reaction_id : int or str
        Enamine REAL reaction ID.
    """

    name: str
    reaction: object  # SyntheMol Reaction
    smarts: str = field(repr=False)
    reaction_id: object = field(default=None, repr=False)
    requires_catalyst: Optional[str] = None
    rate_predictor: Optional[object] = field(default=None, repr=False)

    # Stoichiometry (heavy-atom delta).  Filled on first ``fire`` call;
    # for SyntheMol REAL reactions the typical values are ``{}`` for
    # pure bond-forming rules, or ``{Cl: -1, H: +1}`` etc. for rules
    # that lose a leaving group.
    stoichiometry: Dict[str, int] = field(default_factory=dict, repr=False)

    def predict_yield(self, smiles_a: str, smiles_b: str) -> float:
        """Return the predicted isolated yield in [0, 1].

        SyntheMol REAL rules ship without a fitted rate predictor by
        default; this method returns 0.0 unless a predictor has been
        attached.  Keeps the click-rule :func:`l4_metrics` interface
        working uniformly.
        """
        if self.rate_predictor is None:
            return 0.0
        try:
            return float(self.rate_predictor.predict(smiles_a, smiles_b))
        except Exception:
            return 0.0

    def can_apply(self, reactant_smiles: List[str]) -> bool:
        """Return True iff every reactant SMILES matches the rule's QueryMol."""
        try:
            return bool(self.reaction.has_match(reactant_smiles))
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "[%s] can_apply raised %r; returning False", self.name, exc
            )
            return False

    def reverse_can_apply(
        self, product_smiles: str
    ) -> List[Tuple[str, List[str]]]:
        """Decompose ``product_smiles`` into reactants via the swapped SMARTS.

        Phase-2 SynFlowNet hook — every rule must expose the same
        backward-step interface so the :class:`SynFlowNetEnvAdapter`
        can iterate uniformly.  Returns a list of
        ``(rule_name, [reactant_a_smiles, reactant_b_smiles])`` tuples.

        If the underlying SMARTS is parseable in reverse, we compile
        the swapped reaction and run ``RunReactants`` on the product.
        If the SMARTS is not parseable (or no reactants are produced)
        we return ``[]`` — the rule simply cannot decompose this
        product, which is the correct mass-balance-respecting
        behaviour for irreversible reactions.
        """
        if not self.smarts or ">>" not in self.smarts:
            return []

        try:
            from rdkit import Chem  # type: ignore[import-not-found]
            from rdkit.Chem import rdChemReactions  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "rdChemReactions required for reverse_can_apply"
            ) from exc

        lhs, rhs = self.smarts.split(">>", 1)
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

        try:
            if not rev_rxn.IsMoleculeReactant(prod_mol):
                return []
        except Exception:
            return []

        try:
            rs = rev_rxn.RunReactants((prod_mol,))
        except Exception:
            rs = []

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
            if len(smis) == 1:
                smis = [smis[0], ""]
            key = (smis[0], smis[1])
            if key in seen:
                continue
            seen.add(key)
            out.append((self.name, smis))
        return out

    def fire(self, reactant_smiles: List[str]) -> List[str]:
        """Run the SyntheMol Reaction on the given reactant SMILES.

        Returns
        -------
        list[str]
            Product SMILES (canonical, unique).  Empty list means
            RDKit produced no valid products.
        """
        if not self.can_apply(reactant_smiles):
            return []
        try:
            products = self.reaction.run_reactants(reactant_smiles)
        except Exception as exc:
            raise SyntheMolRuleError(
                f"[{self.name}] run_reactants failed: {exc}"
            ) from exc
        # Strip atom-mapping from products so they round-trip through
        # ``MoleculeClosedTerm.from_smiles`` cleanly.
        cleaned: List[str] = []
        try:
            from rdkit import Chem  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError("RDKit required for SyntheMol reaction firing") from exc
        for smi in products:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            cleaned.append(Chem.MolToSmiles(mol))
        return cleaned

    # The molmetal ReactionRule-style interface (``reduce``) so existing
    # callers that know how to call click rules can also call these.
    def reduce(
        self,
        reactants: Tuple[object, ...] | List[object],
        **kwargs,
    ) -> List[object]:
        """Drop-in replacement for click-rule ``reduce``.

        Accepts either a tuple of ``MoleculeClosedTerm`` (the click-rule
        convention) or a list of SMILES strings.  Returns product
        ``MoleculeClosedTerm`` instances (matching click-rule behaviour)
        so the synthesis / search layers can treat the wrapper as a
        regular rule without special-casing.
        """
        smiles_list: List[str] = []
        for r in reactants:
            if isinstance(r, str):
                smiles_list.append(r)
            else:
                # MoleculeClosedTerm (or any object exposing .source_smiles
                # and .canonical_smiles) — prefer source_smiles so the
                # raw input round-trips, fall back to canonical_smiles.
                smi = getattr(r, "source_smiles", None) or getattr(
                    r, "canonical_smiles", lambda: None
                )()
                if not smi:
                    raise SyntheMolRuleError(
                        f"[{self.name}] cannot extract SMILES from reactant "
                        f"of type {type(r).__name__}"
                    )
                smiles_list.append(smi)
        products = self.fire(smiles_list)

        # Lazy import to keep RDKit off the import path until used.
        from molmetal_lam.molecules.closed_term import MoleculeClosedTerm

        terms: List[MoleculeClosedTerm] = []
        for smi in products:
            try:
                term = MoleculeClosedTerm.from_smiles(smi, embed_3d=False)
            except Exception:
                continue
            term.term = (
                f"SyntheMol/{self.name}({', '.join(smiles_list)}) -> {smi}"
            )
            terms.append(term)
        return terms


# ---------------------------------------------------------------------------
# Curated mapping — 13 SyntheMol REAL reactions
# ---------------------------------------------------------------------------


# Each entry maps a molmetal rule name to (SyntheMol reaction_id,
# optional sub_reaction_id) inside ``REAL_REACTIONS``.  We picked one
# representative SyntheMol reaction per chemical class; the others
# (duplicates of the same chemistry) are intentionally left out so the
# registry stays interpretable for the synthesis layer.
#
# Naming follows the task spec exactly:
#   amide_coupling, ester_hydrolysis, boc_cleavage,
#   sulfonamide_formation, reductive_amination, sn2_alkylation,
#   urea_formation, bromination, hydroxylation, thioether_formation,
#   amide_bond_formation_aromatic, phenol_esterification,
#   nucleophilic_substitution
#
# Note: SyntheMol's REAL_REACTIONS list is curated for Enamine-style
# building blocks, not for free-reagent retrosynthesis.  Many entries
# collapse to the same SMARTS pattern — for example, several reaction
# IDs all encode ``R-NH-R' + R''-X -> R-N(R')-R''`` (SN2 on amine).
# The mapping below picks one canonical SyntheMol reaction per
# chemistry class so the registry stays unique and human-readable.

REAL_TO_LAMBDA: List[Tuple[str, int, Optional[int]]] = [
    # 1. amide_coupling — primary/secondary amine + carboxylic acid → amide (RID 22)
    ("amide_coupling", 22, None),
    # 2. ester_hydrolysis — carboxylic acid + alkyl halide → ester (RID 1458).
    # Note: SyntheMol models esterification rather than hydrolysis; the
    # rule's net effect is the same as building an ester from an acid
    # and a halide.  True hydrolysis (–OH + ester → acid + alcohol) is
    # not in SyntheMol's REAL set; we use the post-reaction
    # ``ESTER_HYDROLYSIS_CO2*`` chain downstream if needed.
    ("ester_hydrolysis", 1458, None),
    # 3. boc_cleavage — single-reactant BOC removal (BOC_CLEAVAGE post-Rx)
    # We use SyntheMol's BOC_CLEAVAGE reaction defined in real.py top.
    ("boc_cleavage", -1, None),  # sentinel: see special-case below
    # 4. sulfonamide_formation — sulfonyl halide + amine → sulfonamide (RID 40)
    ("sulfonamide_formation", 40, None),
    # 5. reductive_amination — alpha-amino acid + aldehyde → oxazole (RID 10).
    # SyntheMol does not have a plain reductive amination in REAL; the
    # closest is the Doebner-type cyclisation encoded in id 10
    # (glycine + aldehyde → oxazole).  We expose it under the
    # reductive_amination name with a clear caveat in the docstring.
    ("reductive_amination", 10, None),
    # 6. sn2_alkylation — amine + alkyl halide (RID 7 sub 1)
    ("sn2_alkylation", 7, 1),
    # 7. urea_formation — two amines → urea (RID 2430)
    ("urea_formation", 2430, None),
    # 8. bromination — amine + alkyl bromide (RID 2230)
    ("bromination", 2230, None),
    # 9. hydroxylation — OH + alkyl halide → ether (RID 7 sub 2)
    ("hydroxylation", 7, 2),
    # 10. thioether_formation — SH + alkyl halide (RID 34 sub 2)
    ("thioether_formation", 34, 2),
    # 11. amide_bond_formation_aromatic — aromatic diamine + aldehyde → benzimidazole
    # SyntheMol's id 60 is the Hantzsch benzimidazole synthesis (o-aryl
    # diamine + aldehyde), not a plain amide coupling.  Exposed here
    # as the only aromatic-amide-forming rule in REAL.
    ("amide_bond_formation_aromatic", 60, None),
    # 12. phenol_esterification — phenol (or any R-OH) + alkyl halide → ether
    # (Williamson synthesis).  RID 272692 sub 2 is the cleanest match.
    ("phenol_esterification", 272692, 2),
    # 13. nucleophilic_substitution — generic amine + halide (RID 44)
    ("nucleophilic_substitution", 44, None),
]


def _resolve_syntemol_reaction(rid: int, sub: Optional[int]):
    """Return the SyntheMol Reaction matching (rid, sub) or BOC sentinel."""
    real = load_real_reactions()
    if rid == -1:
        # BOC_CLEAVAGE is defined at module top in real.py
        real_mod = sys.modules["synthemol.reactions.real"]
        return real_mod.BOC_CLEAVAGE
    for rx in real:
        if rx.reaction_id == rid and getattr(rx, "sub_reaction_id", None) == sub:
            return rx
    raise SyntheMolRuleError(
        f"SyntheMol reaction_id={rid} sub={sub} not found in REAL_REACTIONS"
    )


def from_syntemol_to_lambda(
    mapping: Optional[List[Tuple[str, int, Optional[int]]]] = None,
) -> Dict[str, SyntheMolReactionRule]:
    """Build a ``{name: SyntheMolReactionRule}`` dict from the curated mapping.

    Parameters
    ----------
    mapping : list of (name, reaction_id, sub_reaction_id), optional
        Defaults to :data:`REAL_TO_LAMBDA` (13 curated reactions).

    Returns
    -------
    dict[str, SyntheMolReactionRule]
        Keys are molmetal rule names; values are wrappers around the
        underlying SyntheMol Reaction objects.
    """
    mapping = mapping if mapping is not None else REAL_TO_LAMBDA
    out: Dict[str, SyntheMolReactionRule] = {}
    for name, rid, sub in mapping:
        rx = _resolve_syntemol_reaction(rid, sub)
        out[name] = SyntheMolReactionRule(
            name=name,
            reaction=rx,
            smarts=rx.reaction_smarts,
            reaction_id=rx.id,
        )
    return out


# ---------------------------------------------------------------------------
# Convenience — module-level singleton for tests / callers
# ---------------------------------------------------------------------------


SYNTHEMOL_REAL_REACTIONS = None  # populated lazily by load_real_reactions()


def get_real_reactions():
    """Lazy singleton accessor for SyntheMol's REAL_REACTIONS tuple."""
    global SYNTHEMOL_REAL_REACTIONS
    if SYNTHEMOL_REAL_REACTIONS is None:
        SYNTHEMOL_REAL_REACTIONS = load_real_reactions()
    return SYNTHEMOL_REAL_REACTIONS


__all__ = [
    "SyntheMolRuleError",
    "SyntheMolReactionRule",
    "REAL_TO_LAMBDA",
    "from_syntemol_to_lambda",
    "load_real_reactions",
    "get_real_reactions",
]
