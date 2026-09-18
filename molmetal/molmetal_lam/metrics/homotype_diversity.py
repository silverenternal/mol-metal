"""Homotype diversity metric for Lambda-native chemical-diversity measurement.

This module implements **WF-Lambda-2**: the third first-class algorithmic
asset Lambda brings to the paper, alongside ``alpha_equivalence``
(deduplication) and ``beta_NF`` (synthesizability).

Two mols that are very different *chemically* (e.g. benzene vs decalin)
are also very different *under* ``homotype_diversity`` because they have
different typed-variable hit histograms, different reduction depths, and
different click-rule firings.  Crucially, two **constitutional isomers**
that collapse to the same Morgan fingerprint (Tanimoto = 1.0) but
differ in their typed-variable hit histograms (e.g. branched vs linear
chain) are pulled apart by ``homotype_diversity`` — which is exactly
the contribution this metric makes to the paper: it is **independent
of SE(3) distance and atom-level Morgan Tanimoto**.

Spec
----
* ``HomotypeSignature``:
    * ``typed_variable_counts: dict[str, int]`` — per-constructor-symbol
      multiset, e.g. ``{"C": 12, "N": 3, "Pt": 1, "O": 5}``
    * ``beta_reduction_depth: int`` — number of β-reduction steps taken
      to reach this term (>=0)
    * ``click_rule_fires: dict[str, int]`` — per-click-rule fire counts
      on the path to this term, e.g. ``{"CuAAC": 2, "SPAAC": 0, "ThiolEne": 1}``
    * ``from_term(term, reduction_history)`` — build from a Lambda term +
      its reduction sequence (caller is responsible for the history list).
    * ``from_mol(mol, prior=None)`` — build from an RDKit mol using
      atomic numbers as typed-variable proxies (per the spec).

* ``homotype_distance(sig_a, sig_b) -> float in [0, 1]``:
    * ``0.5 * cosine_distance(typed_variable_counts)``
    * ``0.3 * normalised |Δβ-depth| / max_depth``
    * ``0.2 * Jaccard_distance(click_rule_fires)``
    * Sum is bounded in [0, 1] when each component is bounded in [0, 1].

* ``homotype_diversity(mol_set) -> float in [0, 1]``:
    * Mean pairwise ``homotype_distance`` over the set.  Equivalent to
      ``1 - mean_pairwise_similarity`` but Lambda-native.

Honest framing
--------------
**MEASURED**: typed-variable symbols on ``MoleculeClosedTerm.atoms``
(``Atom.symbol`` at ``atoms/combinators.py:66-95``); the
``_value_signature`` multiset view (closed_term.py:477-498).  RDKit
``Chem.Mol`` per-atom symbol counts are MEASURED.

**PROJECTED**: the *fuse* into a single pairwise Lambda-native
diversity score — there is no existing helper that does this; the
multiset-Jaccard / cosine / normalised-depth combination is the new
glue introduced here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

try:
    from rdkit import Chem  # type: ignore
except Exception:  # pragma: no cover
    Chem = None  # type: ignore


__all__ = [
    "HomotypeSignature",
    "homotype_distance",
    "homotype_diversity",
]


# ---------------------------------------------------------------------------
# Reference click-rule vocabulary
# ---------------------------------------------------------------------------
# The default click-rule vocabulary covers the three reactions exposed
# by ``molmetal_lam/reactions/click_reactions.py`` (CuAAC, SPAAC,
# ThiolEne).  When callers pass a different vocabulary via
# ``HomotypeSignature.from_mol(prior=...)`` they can override the rule
# keys, but the default union of the three is the canonical Lambda
# click set we report distances against.
DEFAULT_CLICK_RULE_KEYS = ("CuAAC", "SPAAC", "ThiolEne")


# ---------------------------------------------------------------------------
# Helper: cosine distance over count dicts
# ---------------------------------------------------------------------------
def _cosine_distance(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> float:
    """Cosine distance ``1 - cos(theta)`` between two sparse count vectors.

    Returns 0.0 when both vectors are zero (identical empty signatures),
    otherwise 1 - <a,b> / (||a|| * ||b||).
    """
    if not counts_a and not counts_b:
        return 0.0
    keys = set(counts_a) | set(counts_b)
    dot = 0.0
    na2 = 0.0
    nb2 = 0.0
    for k in keys:
        va = float(counts_a.get(k, 0))
        vb = float(counts_b.get(k, 0))
        dot += va * vb
        na2 += va * va
        nb2 += vb * vb
    denom = math.sqrt(na2) * math.sqrt(nb2)
    if denom == 0.0:
        return 0.0
    sim = dot / denom
    # Clip to guard against tiny FP overshoot.
    sim = max(-1.0, min(1.0, sim))
    return float(1.0 - sim)


# ---------------------------------------------------------------------------
# Helper: Jaccard distance over (rule -> count) keys
# ---------------------------------------------------------------------------
def _jaccard_distance(counts_a: Dict[str, int], counts_b: Dict[str, int]) -> float:
    """Jaccard distance ``1 - |A ∩ B| / |A ∪ B|`` over the *key sets*.

    Note: this is the set-version of Jaccard (uses the keys with non-zero
    count).  The spec says "Jaccard distance over click_rule_fires",
    which in our context is best interpreted as the asymmetric set
    overlap of rules-with-fires-vs-no-fires — a zero-vs-nonzero pattern
    is what separates "CuAAC path" from "SPAAC path" in the search
    tree.
    """
    set_a = {k for k, v in counts_a.items() if v}
    set_b = {k for k, v in counts_b.items() if v}
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return float(1.0 - inter / union)


# ---------------------------------------------------------------------------
# Helper: extended-vocab enrichment (WF-Lambda-2.E)
# ---------------------------------------------------------------------------
def _bump(counts: Dict[str, int], key: str, n: int = 1) -> None:
    """In-place ``counts[key] += n`` helper."""
    counts[key] = counts.get(key, 0) + n


def _enrich_extended_vocab(mol, counts: Dict[str, int]) -> None:
    """Augment ``counts`` with hybridisation, ring, and H-count features.

    For each atom we emit:
      * ``C_sp3`` / ``C_sp2`` / ``C_sp`` / ``C_ar`` for carbon atoms
        (depending on ``Chem.HybridizationType`` and aromaticity).
      * ``N_sp3`` / ``N_sp2`` / ``N_sp`` / ``N_ar`` for nitrogen atoms
        (analogous — common in organometallics e.g. Pt-amines).
      * ``O_sp3`` / ``O_sp2`` for oxygen atoms.
      * ``ring_<size>`` for each atom that participates in a ring of
        that size (a fused atom can contribute to multiple ring_*
        tokens, one per ring it belongs to).
      * ``aromatic_ring_<size>`` for each atom that participates in
        an aromatic ring of that size.
      * ``H<n>`` for each atom's implicit hydrogen count (RDKit's
        ``GetNumImplicitHs``).  ``n >= 4`` is bucketed to ``H4``.
    """
    # Pre-compute per-atom ring memberships so we can distinguish
    # aromatic_ring_<size> from generic ring_<size>.
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()  # tuple of tuples of atom indices
    aromatic_atom_rings: List[tuple] = []
    for ring in atom_rings:
        if not ring:
            continue
        is_aromatic = True
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            if not atom.GetIsAromatic():
                is_aromatic = False
                break
        if is_aromatic:
            aromatic_atom_rings.append(ring)

    # Map: atom_idx -> list of ring sizes it belongs to
    ring_sizes_by_atom: Dict[int, List[int]] = {}
    for ring in atom_rings:
        size = len(ring)
        for idx in ring:
            ring_sizes_by_atom.setdefault(idx, []).append(size)
    aromatic_ring_sizes_by_atom: Dict[int, List[int]] = {}
    for ring in aromatic_atom_rings:
        size = len(ring)
        for idx in ring:
            aromatic_ring_sizes_by_atom.setdefault(idx, []).append(size)

    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        idx = atom.GetIdx()
        hyb = atom.GetHybridization()
        is_aromatic = bool(atom.GetIsAromatic())

        # Hybridisation class (carbon + heteroatoms).
        if sym == "C":
            if is_aromatic:
                _bump(counts, "C_ar")
            elif hyb == Chem.HybridizationType.SP3:
                _bump(counts, "C_sp3")
            elif hyb == Chem.HybridizationType.SP2:
                _bump(counts, "C_sp2")
            elif hyb == Chem.HybridizationType.SP:
                _bump(counts, "C_sp")
            else:
                # Fall back to sp3 (saturated) when RDKit can't classify.
                _bump(counts, "C_sp3")
        elif sym == "N":
            if is_aromatic:
                _bump(counts, "N_ar")
            elif hyb == Chem.HybridizationType.SP3:
                _bump(counts, "N_sp3")
            elif hyb == Chem.HybridizationType.SP2:
                _bump(counts, "N_sp2")
            elif hyb == Chem.HybridizationType.SP:
                _bump(counts, "N_sp")
            else:
                _bump(counts, "N_sp3")
        elif sym == "O":
            if hyb == Chem.HybridizationType.SP2:
                _bump(counts, "O_sp2")
            else:
                _bump(counts, "O_sp3")

        # Ring-class tokens (one emission per ring membership).
        for size in ring_sizes_by_atom.get(idx, []):
            _bump(counts, f"ring_{size}")
        for size in aromatic_ring_sizes_by_atom.get(idx, []):
            _bump(counts, f"aromatic_ring_{size}")

        # Implicit H count.
        try:
            n_h = int(atom.GetTotalNumHs())
        except Exception:
            n_h = int(atom.GetNumImplicitHs())
        n_h = max(0, min(n_h, 4))
        _bump(counts, f"H{n_h}")


# ---------------------------------------------------------------------------
# HomotypeSignature
# ---------------------------------------------------------------------------
@dataclass
class HomotypeSignature:
    """A Lambda-native structural fingerprint of a molecule / term.

    The three fields are jointly the "homotype" of the term: its
    typed-variable hits, the depth of its reduction history, and the
    click rules that fired on the way there.

    Attributes
    ----------
    typed_variable_counts : dict[str, int]
        e.g. ``{"C": 12, "N": 3, "Pt": 1, "O": 5}``.
    beta_reduction_depth : int
        Non-negative integer count of β-reduction steps that produced
        this term.  Zero for a term that was never reduced (e.g. a
        freshly parsed RDKit mol that was already in BNF).
    click_rule_fires : dict[str, int]
        Per-rule fire counts along the reduction history.  Defaults to
        all-zero for terms built from an RDKit mol (no click history
        is available outside the MCTS run that produced the term).
    """

    typed_variable_counts: Dict[str, int] = field(default_factory=dict)
    beta_reduction_depth: int = 0
    click_rule_fires: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_term(
        cls,
        term,
        reduction_history: Optional[Sequence] = None,
        rule_name_extractor=None,
    ) -> "HomotypeSignature":
        """Build a signature from a ``MoleculeClosedTerm`` (or duck-typed).

        Parameters
        ----------
        term : MoleculeClosedTerm-like
            Object exposing ``atoms`` (list of ``Atom``-like with
            ``.symbol``).
        reduction_history : sequence, optional
            Iterable of intermediate terms produced along the β-reduction
            path.  Length-1 = no reductions; length-N = N-1 reductions.
            When ``None``, depth is 0.
        rule_name_extractor : callable, optional
            ``f(reduction_step) -> Optional[str]`` returning the click
            rule that fired at each step.  When ``None``, no rule
            accounting is done and ``click_rule_fires`` is the
            all-zero default.
        """
        counts: Dict[str, int] = {}
        try:
            atoms = list(getattr(term, "atoms", []))
        except Exception:
            atoms = []
        for atom in atoms:
            sym = getattr(atom, "symbol", None) or "?"
            counts[sym] = counts.get(sym, 0) + 1

        depth = 0
        rule_counts: Dict[str, int] = {k: 0 for k in DEFAULT_CLICK_RULE_KEYS}
        if reduction_history is not None:
            try:
                depth = max(0, len(list(reduction_history)) - 1)
            except Exception:
                depth = 0
            if rule_name_extractor is not None:
                for step in reduction_history:
                    name = rule_name_extractor(step)
                    if name is None:
                        continue
                    rule_counts[name] = rule_counts.get(name, 0) + 1

        return cls(
            typed_variable_counts=counts,
            beta_reduction_depth=int(depth),
            click_rule_fires=rule_counts,
        )

    @classmethod
    def from_mol(
        cls,
        mol,
        prior: Optional[Dict[str, int]] = None,
        use_extended_vocab: bool = True,
    ) -> "HomotypeSignature":
        """Build a signature from an ``rdkit.Chem.Mol``.

        Per the spec, atomic symbols are used as the typed-variable
        proxy.  ``prior`` is an optional click-rule-fires dict that
        carries over from a synthesis pathway when the caller has it
        (defaults to all-zero when absent).  ``beta_reduction_depth``
        is set to 0 for an RDKit-parsed mol — the molecule has no
        reduction history unless the caller attaches one.

        When ``use_extended_vocab`` is True (default), the typed-variable
        histogram is augmented with per-atom hybridisation classes
        (``C_sp3`` / ``C_sp2`` / ``C_sp`` / ``C_ar``), per-atom ring
        membership (e.g. ``ring_5`` / ``ring_6``), aromatic ring
        membership (``aromatic_ring_<size>``), and per-atom implicit
        hydrogen counts (``H0`` ... ``H4+``).  This enriches the metric
        so it can distinguish constitutional isomers like cyclohexane
        vs hex-1-ene that collapse to identical raw symbol multisets.
        """
        if Chem is None or mol is None:
            return cls(
                typed_variable_counts={},
                beta_reduction_depth=0,
                click_rule_fires=dict(prior) if prior else {
                    k: 0 for k in DEFAULT_CLICK_RULE_KEYS
                },
            )

        counts: Dict[str, int] = {}
        try:
            for atom in mol.GetAtoms():  # type: ignore[attr-defined]
                sym = atom.GetSymbol()
                counts[sym] = counts.get(sym, 0) + 1
        except Exception:
            counts = {}

        # Extended vocabulary enrichment (WF-Lambda-2.E).
        if use_extended_vocab:
            try:
                _enrich_extended_vocab(mol, counts)
            except Exception:
                # Never let enrichment failures corrupt the basic
                # signature — degrade gracefully to symbol-only counts.
                pass

        # Honour explicit ``prior`` if given, else default all-zero rule
        # counts for the canonical click-rule vocabulary.
        if prior is None:
            rule_counts = {k: 0 for k in DEFAULT_CLICK_RULE_KEYS}
        else:
            rule_counts = dict(prior)

        return cls(
            typed_variable_counts=counts,
            beta_reduction_depth=0,
            click_rule_fires=rule_counts,
        )


# ---------------------------------------------------------------------------
# Distance function
# ---------------------------------------------------------------------------
# Component weights from the spec.
_W_COS = 0.5
_W_DEPTH = 0.3
_W_JACCARD = 0.2


def homotype_distance(sig_a: HomotypeSignature, sig_b: HomotypeSignature) -> float:
    """Pairwise homotype distance in [0, 1].

    Components (per spec):
      * 0.5 * cosine distance over typed_variable_counts
      * 0.3 * normalised |Δβ-depth| / max_depth  (0.0 when both depths = 0)
      * 0.2 * Jaccard distance over click_rule_fires

    Returns 0.0 when both signatures are identical on all three
    components.  Returns a value close to 1.0 when the two signatures
    disagree on every component.
    """
    # 1. Cosine on typed variables.
    cos_d = _cosine_distance(
        sig_a.typed_variable_counts, sig_b.typed_variable_counts
    )

    # 2. Normalised depth difference.
    da = int(sig_a.beta_reduction_depth)
    db = int(sig_b.beta_reduction_depth)
    max_depth = max(da, db, 1)
    depth_d = float(abs(da - db)) / float(max_depth)

    # 3. Jaccard on click-rule fires.
    jac_d = _jaccard_distance(sig_a.click_rule_fires, sig_b.click_rule_fires)

    total = _W_COS * cos_d + _W_DEPTH * depth_d + _W_JACCARD * jac_d
    # Clip to [0, 1] defensively against floating-point drift.
    return float(max(0.0, min(1.0, total)))


# ---------------------------------------------------------------------------
# Set-level aggregator
# ---------------------------------------------------------------------------
def homotype_diversity(mol_set: Sequence) -> float:
    """Mean pairwise ``homotype_distance`` over ``mol_set``.

    Returns 0.0 for a set of size 0 or 1 (no pairwise comparisons).

    Notes
    -----
    The spec says ``homotype_diversity(mol_set: list[RDKit mol])``.  We
    accept either a list of ``rdkit.Chem.Mol`` objects, a list of
    pre-built ``HomotypeSignature`` objects, or a mixed sequence; for
    any non-``HomotypeSignature`` element we build one via
    ``HomotypeSignature.from_mol``.
    """
    if mol_set is None:
        return 0.0
    seq = list(mol_set)
    n = len(seq)
    if n < 2:
        return 0.0

    sigs: List[HomotypeSignature] = []
    for item in seq:
        if isinstance(item, HomotypeSignature):
            sigs.append(item)
        else:
            sigs.append(HomotypeSignature.from_mol(item))

    total = 0.0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += homotype_distance(sigs[i], sigs[j])
            n_pairs += 1

    if n_pairs == 0:
        return 0.0
    return float(total / n_pairs)
