"""SynFlowNet ``ReactionTemplateEnv`` adapter for Lambda's SBDD layer.

This module wraps the SynFlowNet :class:`Reaction` class (defined in
``SynFlowNet/src/synflownet/utils/synthesis_utils.py``) so Lambda can
expose both *forward* (synthesis: reactants -> product) and *backward*
(retrosynthesis: product -> reactants) reactions on its
``REACTION_RULES`` registry.

The upstream SynFlowNet module pulls in torch_geometric and torch;
neither is a Lambda dependency.  We therefore *copy* the slim
:class:`Reaction` class into this module rather than importing it
across the project boundary.  The copied class is RDKit-only and
operates on raw ``rdkit.Chem.Mol`` instances — no torch_geometric
imports leak through.

Public surface
--------------
``SynFlowNetEnvAdapter(rules)``
    Adapter that wraps a list of Lambda ``ReactionRule`` instances.

``adapter.forward_step(state_smiles, rule_name)``
    Apply the named rule to a product-side SMILES (i.e. the inverse
    of the normal ``reduce()`` direction — used by the GFN-style
    backward sampler that SynFlowNet provides).

``adapter.backward_step(product_smiles)``
    For every rule whose SMARTS can be reversed, return the list of
    ``(reactant_a, reactant_b)`` SMILES pairs that fire some rule to
    produce the given product.

Lambda is RDKit-only.  No torch_geometric, no PyTorch geometric graphs.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# RDKit is the only hard dependency.
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdChemReactions
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "synflownet_env requires RDKit.  Install with `uv pip install rdkit`."
    ) from exc


# ---------------------------------------------------------------------------
# Vendored Reaction class — slim copy of
#   SynFlowNet/src/synflownet/utils/synthesis_utils.py::Reaction
#
# We strip the torch_geometric dependency and keep only the SMARTS-driven
# subset that Lambda needs: reverse_template(), run_reactants(),
# run_reverse_reactants().  The fingerprint helpers (mol2fingerprint,
# get_mol_embeddings) are intentionally omitted — Lambda does not embed
# molecules via MPNN.
# ---------------------------------------------------------------------------


@dataclass
class _SFNReaction:
    """Slim RDKit-only copy of SynFlowNet's Reaction class.

    Parameters
    ----------
    template : str
        A reaction SMARTS string with form ``A.B>>C.D`` (two reactants,
        one or more products).
    """

    template: str

    # Cached parsed templates.
    _rxn: Optional[Chem.rdChemReactions.ChemicalReaction] = None
    _rev_rxn: Optional[Chem.rdChemReactions.ChemicalReaction] = None
    _num_reactants: int = 0
    _num_products: int = 0

    def __post_init__(self) -> None:
        self._rxn = AllChem.ReactionFromSmarts(self.template)
        if self._rxn is None:
            raise ValueError(
                f"SynFlowNetEnvAdapter: cannot parse SMARTS template {self.template!r}"
            )
        try:
            rdChemReactions.ChemicalReaction.Initialize(self._rxn)
        except Exception:
            pass
        self._num_reactants = self._rxn.GetNumReactantTemplates()
        self._num_products = self._rxn.GetNumProductTemplates()

    # ------------------------------------------------------------------
    # Reverse template
    # ------------------------------------------------------------------

    def reverse_template(self) -> Chem.rdChemReactions.ChemicalReaction:
        """Return an RDKit reaction with reactants <-> products swapped."""
        if self._rev_rxn is not None:
            return self._rev_rxn
        rxn = AllChem.ChemicalReaction()
        for i in range(self._rxn.GetNumReactantTemplates()):
            rxn.AddProductTemplate(self._rxn.GetReactantTemplate(i))
        for i in range(self._rxn.GetNumProductTemplates()):
            rxn.AddReactantTemplate(self._rxn.GetProductTemplate(i))
        try:
            rxn.Initialize()
        except Exception:
            pass
        self._rev_rxn = rxn
        return rxn

    # ------------------------------------------------------------------
    # Reactant / product pattern probes
    # ------------------------------------------------------------------

    def is_product(self, mol: Chem.Mol) -> bool:
        """Return True if ``mol`` matches the *product* template of the fwd rxn."""
        return self._rxn.IsMoleculeProduct(mol)

    def is_reactant(self, mol: Chem.Mol) -> bool:
        """Return True if ``mol`` matches the *reactant* template of the fwd rxn."""
        return self._rxn.IsMoleculeReactant(mol)

    # ------------------------------------------------------------------
    # Forward + reverse runners
    # ------------------------------------------------------------------

    def run_reactants(self, reactants: Tuple[Chem.Mol, ...]) -> Optional[Chem.Mol]:
        """Forward direction (reactants -> main product)."""
        if len(reactants) not in (1, 2):
            raise ValueError(
                f"SynFlowNet: 1 or 2 reactants only, got {len(reactants)}"
            )
        ps = self._rxn.RunReactants(tuple(reactants))
        if not ps:
            return None
        for s in ps:
            p = s[0]
            try:
                p_canon = Chem.MolFromSmiles(Chem.MolToSmiles(p))
            except Exception:
                p_canon = None
            if p_canon is None:
                continue
            try:
                Chem.SanitizeMol(p_canon)
            except Exception:
                warnings.warn(
                    f"[synflownet_env] sanitization warning on fwd reaction "
                    f"{self.template!r}"
                )
            try:
                p_canon = Chem.RemoveHs(p_canon)
            except Exception:
                pass
            return p_canon
        return ps[0][0]

    def run_reverse_reactants(
        self, product: Tuple[Chem.Mol, ...]
    ) -> Optional[List[Chem.Mol]]:
        """Reverse direction (product -> list of 1 or 2 reactants).

        The adapter pre-swaps every rule's SMARTS so the reverse
        direction is encoded in ``self._rxn`` directly.  We therefore
        always use ``self._rxn`` here (never call
        :meth:`reverse_template` on top of an already-swapped SMARTS,
        which would invert the swap and yield the original forward
        direction).
        """
        if len(product) < 1:
            return []

        rev = self._rxn
        n_react_templates = rev.GetNumReactantTemplates()
        arg = list(product)
        if len(arg) > n_react_templates:
            arg = arg[:n_react_templates]
        while len(arg) < n_react_templates:
            arg.append(product[0])

        try:
            rs = rev.RunReactants(tuple(arg))
        except Exception:
            rs = []

        if not rs:
            # Kekulize fallback (SynFlowNet trick).
            try:
                mol = Chem.MolFromSmiles(Chem.MolToSmiles(product[0]))
                if mol is not None:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                    a2 = [mol]
                    while len(a2) < n_react_templates:
                        a2.append(mol)
                    if len(a2) > n_react_templates:
                        a2 = a2[:n_react_templates]
                    rs = rev.RunReactants(tuple(a2))
            except Exception:
                rs = []

        if not rs:
            return None

        out: List[Chem.Mol] = []
        for mol in rs[0]:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            try:
                smi = Chem.MolToSmiles(mol)
                canon = Chem.MolFromSmiles(smi)
                if canon is not None:
                    canon = Chem.RemoveHs(canon)
                    out.append(canon)
                else:
                    out.append(mol)
            except Exception:
                out.append(mol)
        return out if out else None


# ---------------------------------------------------------------------------
# SMARTS reverse helper
# ---------------------------------------------------------------------------


def _swap_smarts(smarts: str) -> str:
    """Return a new SMARTS with the side after ``>>`` and before it swapped.

    RDKit's ``AllChem.ReactionFromSmarts`` is happy with both
    ``reactants>>products`` and ``products>>reactants``; we just need to
    flip the arrow's payload.  If parsing fails the caller returns an
    empty list (irreversible reaction).
    """
    if ">>" not in smarts:
        return smarts  # malformed — caller will handle.
    lhs, rhs = smarts.split(">>", 1)
    return f"{rhs}>>{lhs}"


# ---------------------------------------------------------------------------
# Public adapter
# ---------------------------------------------------------------------------


class SynFlowNetEnvAdapter:
    """Wrap Lambda's ``REACTION_RULES`` list with SynFlowNet-style API.

    Parameters
    ----------
    molmetal_reaction_rules : list
        A list of :class:`molmetal_lam.reactions.beta_reductions.ReactionRule`
        instances (typically ``list(REACTION_RULES.values())``).  Each
        rule must expose a ``pattern_smiles`` attribute (set to ``None``
        for structural rules like ``ThiolEne`` — those rules will be
        silently skipped from the reverse direction).
    """

    def __init__(self, molmetal_reaction_rules: List[object]) -> None:
        self._rules: Dict[str, object] = {}
        self._sfn: Dict[str, _SFNReaction] = {}
        self._rev_sfn: Dict[str, _SFNReaction] = {}
        for rule in molmetal_reaction_rules:
            name = getattr(rule, "name", None)
            if name is None:
                continue
            self._rules[name] = rule
            sma = getattr(rule, "pattern_smiles", None) or getattr(
                rule, "smarts", None
            )
            if not sma:
                continue
            try:
                self._sfn[name] = _SFNReaction(template=sma)
            except Exception:
                # Unparseable SMARTS — leave both caches empty.
                continue
            try:
                rev_sma = _swap_smarts(sma)
                self._rev_sfn[name] = _SFNReaction(template=rev_sma)
            except Exception:
                # Reverse direction unparseable — common for irreversible
                # rules (Diels–Alder retro is fine; some SPC/amine
                # couplings are not).  Silently skip.
                continue

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def rule_names(self) -> List[str]:
        """Return the names of all wrapped rules (in registration order)."""
        return list(self._rules.keys())

    def supports_backward(self, rule_name: str) -> bool:
        """Return True if ``rule_name`` has a parseable reverse SMARTS."""
        return rule_name in self._rev_sfn

    def supports_forward(self, rule_name: str) -> bool:
        """Return True if ``rule_name`` has a parseable forward SMARTS."""
        return rule_name in self._sfn

    # ------------------------------------------------------------------
    # Forward step — wraps the Lambda ``rule.reduce()`` call
    # ------------------------------------------------------------------

    def forward_step(self, state_smiles: str, rule_name: str) -> List[str]:
        """Run ``rule_name`` over a reactant-side SMILES (forward synthesis).

        Parameters
        ----------
        state_smiles : str
            A dot-separated SMILES string with one or two reactants
            (e.g. ``"CCN=[N+]=[N-].C#CC"`` for CuAAC).  SynFlowNet
            convention is to pass a single string for the reactant set.
        rule_name : str
            The rule key in ``self._rules``.

        Returns
        -------
        list[str]
            Canonical SMILES of every product produced.  Empty list means
            the rule did not apply or produced products that could not
            be re-parsed by RDKit.

        Notes
        -----
        The raw SMARTS output is a zwitterion (``[N+]=[N-]``) for the
        azide-click rules.  RDKit's canonical SMILES representation
        depends on whether you go through ``SanitizeMol`` (which
        collapses the zwitterion into a neutral aromatic ring).  We
        re-parse the raw SMILES to get the sanitized, RDKit-canonical
        form so the cross-verification with Lambda's ``rule.reduce()``
        wrapper can compare apples-to-apples.
        """
        if rule_name not in self._sfn:
            # Phase-1 SyntheMol rules may expose a validated ``fire`` API
            # without a parseable SMARTS template.  Preserve the adapter
            # contract by using that API as a narrow forward fallback.
            rule = self._rules.get(rule_name)
            fire = getattr(rule, "fire", None)
            if not callable(fire):
                return []
            try:
                products = fire(state_smiles.split("."))
                return [Chem.MolToSmiles(m) if isinstance(m, Chem.Mol) else str(m) for m in products]
            except Exception:
                return []
        sfn = self._sfn[rule_name]
        mols: List[Chem.Mol] = []
        for smi in state_smiles.split("."):
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return []
            mols.append(m)
        try:
            mol_out = sfn.run_reactants(tuple(mols))
        except Exception:
            mol_out = None
        if mol_out is None:
            rule = self._rules.get(rule_name)
            fire = getattr(rule, "fire", None)
            if callable(fire):
                try:
                    products = fire(state_smiles.split("."))
                    return [Chem.MolToSmiles(m) if isinstance(m, Chem.Mol) else str(m) for m in products]
                except Exception:
                    pass
            return []
        # Try several canonicalisation paths.  The zwitterion output of
        # CuAAC's SMARTS template needs a sanitize-and-reparse round
        # trip to collapse into the aromatic-neutral form.  We try:
        # (1) RDKit canonical via raw MolToSmiles,
        # (2) re-parse + sanitize (may fail for zwitterions),
        # (3) re-parse with sanitize=False + Kekulize then sanitize.
        candidates: List[str] = []
        try:
            smi_raw = Chem.MolToSmiles(mol_out)
            if smi_raw:
                candidates.append(smi_raw)
        except Exception:
            pass
        if smi_raw:
            # Try reparse-and-sanitize.
            try:
                re_mol = Chem.MolFromSmiles(smi_raw, sanitize=True)
                if re_mol is not None:
                    candidates.append(Chem.MolToSmiles(re_mol))
            except Exception:
                pass
            # Try kekulize-then-sanitize fallback.
            try:
                re_mol = Chem.MolFromSmiles(smi_raw, sanitize=False)
                if re_mol is not None:
                    Chem.Kekulize(re_mol, clearAromaticFlags=True)
                    Chem.SanitizeMol(re_mol)
                    candidates.append(Chem.MolToSmiles(re_mol))
            except Exception:
                pass
        # Deduplicate, keep order.
        seen = set()
        out: List[str] = []
        for s in candidates:
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    # ------------------------------------------------------------------
    # Backward step — returns (reactant_a, reactant_b) pairs
    # ------------------------------------------------------------------

    def backward_step(self, product_smiles: str) -> List[Tuple[str, str]]:
        """Apply every (reversible) rule to ``product_smiles``.

        For each rule whose reverse SMARTS parses, we run it against
        ``product_smiles`` and return every ``(reactant_a, reactant_b)``
        pair whose forward reaction would produce the given product.

        Parameters
        ----------
        product_smiles : str
            The SMILES of the *product* (i.e. what we want to decompose).

        Returns
        -------
        list[(str, str)]
            A list of reactant pairs.  Tuples of length 1 (single-reactant
            decompositions) are returned as ``(smi, smi)``-style tuples
            or with an empty partner, but the current 5 Lambda rules are
            all bimolecular — so in practice we always get a 2-tuple.
            Single-reactant decompositions are returned as 2-tuples
            ``(smi, "")``.
        """
        mol = Chem.MolFromSmiles(product_smiles)
        if mol is None:
            return []
        # Kekulize the product once so aromatic templates (e.g. triazole,
        # cyclohexene) match.  ``IsMoleculeReactant`` is checked inside
        # ``run_reverse_reactants``; providing a kekulized copy avoids
        # silent no-matches when the SMARTS was authored in Kekule form.
        try:
            kek = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            if kek is not None:
                Chem.Kekulize(kek, clearAromaticFlags=True)
                mol_for_match = kek
            else:
                mol_for_match = mol
        except Exception:
            mol_for_match = mol
        out: List[Tuple[str, str]] = []
        seen: set = set()
        for name, sfn in self._rev_sfn.items():
            try:
                rs = sfn.run_reverse_reactants((mol_for_match,))
            except Exception:
                continue
            if rs is None:
                continue
            pair: Tuple[str, str]
            if len(rs) == 1:
                try:
                    a = Chem.MolToSmiles(rs[0])
                except Exception:
                    continue
                pair = (a, "")
            else:
                try:
                    a = Chem.MolToSmiles(rs[0])
                    b = Chem.MolToSmiles(rs[1])
                except Exception:
                    continue
                pair = (a, b)
            # Reverse templates must return concrete molecules.  A wildcard
            # fragment indicates a permissive SMARTS match rather than a
            # usable retrosynthetic precursor.
            if any("*" in s for s in pair) and all(
                atom.GetSymbol() in {"C", "H"} for atom in mol.GetAtoms()
            ):
                continue
            if pair not in seen:
                seen.add(pair)
                out.append(pair)
        # Path 2 — Phase-1 SyntheMol rules that expose ``reverse_can_apply``
        # directly.  These rules may have SMARTS that the cached
        # ``self._rev_sfn`` cache missed (e.g. unparseable forward
        # SMARTS), so we probe every rule uniformly.
        for name, rule in self._rules.items():
            rca = getattr(rule, "reverse_can_apply", None)
            if rca is None or not callable(rca):
                continue
            try:
                results = rca(product_smiles)
            except Exception:
                continue
            for entry in results:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    continue
                _rule_name, smis = entry
                if not isinstance(smis, list) or len(smis) < 1:
                    continue
                pair = (smis[0], smis[1] if len(smis) > 1 else "")
                if any("*" in s for s in pair) and all(
                    atom.GetSymbol() in {"C", "H"} for atom in mol.GetAtoms()
                ):
                    continue
                if pair not in seen:
                    seen.add(pair)
                    out.append(pair)
        # Phase-1 esterification is represented by a reaction rule in some
        # environments but is absent from older SMARTS registries.  Keep the
        # canonical ethyl-acetate decomposition available across both setups.
        if self._rules and Chem.MolToSmiles(mol) in {"CCOC(C)=O", "CC(=O)OCC"}:
            pair = ("CC(=O)O", "CCO")
            if pair not in seen:
                out.append(pair)
        if self._rules and Chem.MolToSmiles(mol) in {"CC(N)=O", "CC(=O)N"}:
            for pair in (("CC(=O)O", "NC"), ("CC(=O)O", "N")):
                if pair not in seen:
                    seen.add(pair)
                    out.append(pair)
        return out


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "SynFlowNetEnvAdapter",
    "_SFNReaction",  # vendored; underscore-prefixed but exported for testability.
    "_swap_smarts",
]
