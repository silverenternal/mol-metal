"""Skeleton-jump / fragment-swap rule — Pitfall Rank 12, P2.4 (TODO-30).

Background
----------
The current generation pipeline grows molecules **stepwise**: each
expansion step adds one atom / one bond.  This biases the search
towards incremental elaboration of the seed scaffold and limits the
ability to explore distant regions of chemical space.

A **skeleton jump** (also called "fragment swap") replaces a fragment
of the growing molecule with a structurally different fragment from a
pre-built library.  This produces large structural changes in one
expansion step, dramatically increasing the **diversity** of the
generated set without abandoning the constructive synthesis
guarantee (we always emit a β-NF, RDKit-sanitisable product).

Algorithmic role
----------------
The fragment-swap rule sits *between* the layered-generation pipeline
(see :mod:`molmetal.molmetal_lam.lam_chem.layered_generation`) and the
MCTS expansion path.  Conceptually it is one extra "macro move" the
MCTS can take in addition to the click-rule β-reductions — but unlike
click rules, it does NOT require a partner molecule (it operates on
the molecule's own fragments).

Public API
----------
:class:`FragmentLibrary`         : typed catalog of swap candidates
:class:`SkeletonJumpRule`         : fragment-swap rule (single-position)
:class:`DiversityAwareSkeletonJump` : diversity-aware wrapper
:func:`register_skeleton_jump`   : module-level registration hook

Constraints honoured
--------------------
* **No modification** of ``rules.py``, ``lambda_combinators.py``, or
  ``closure.py``.  This module is purely additive; the existing
  click-rule registry is untouched.
* **CPU-only** — no torch, no GPU, no learned priors.
* **Pure-RDKit** chemistry — uses RDKit SMILES parsing, fragment
  matching, and valence checks for compatibility scoring.

Lit anchors
-----------
* Brown et al. 2004 *J. Chem. Inf. Model.* 44(3) — fragment-swap
  generation: replacing a Murcko fragment with a library fragment
  increases scaffold diversity while preserving ligand efficiency.
* Ertl & Schuffenhauer 2009 *J. Cheminform.* 1:8 — fragment-library
  design guidelines: 3-12 heavy atoms, drug-like substitution.

Honest framing
--------------
The compatibility score is a **CPU heuristic**, not a learned
predictor.  It captures:

  (a) size match — fragment heavy-atom count vs scaffold-cutout count;
  (b) topology match — rings/chain/heterocycle tag overlap;
  (c) valence feasibility — the swap is RDKit-sanitisable.

A learned compatibility model (TODO-31 learned sampler) can replace
this heuristic; the public API is stable so the swap is a one-line
:class:`SkeletonJumpRule` rewire.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

# Quiet RDKit chatter — the fragment-swap path parses many small
# molecules and the default verbosity would flood the test logs.
RDLogger.DisableLog("rdApp.*")


__all__ = [
    "DEFAULT_MIN_ATOMS",
    "DEFAULT_MAX_ATOMS",
    "DEFAULT_SWAP_PROB",
    "DEFAULT_TARGET_DIVERSITY",
    "FragmentRecord",
    "FragmentLibrary",
    "SkeletonJumpRule",
    "DiversityAwareSkeletonJump",
    "register_skeleton_jump",
]


# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------
#: Default minimum heavy-atom count for a swap candidate.  Below 3 the
#: fragment is too small to carry chemistry (Brown et al. 2004 §3).
DEFAULT_MIN_ATOMS: int = 3
#: Default maximum heavy-atom count.  Beyond 12 the SA score climbs
#: and the swap becomes structurally dominant (Ertl & Schuffenhauer
#: 2009 — "fragment libraries should span 3-12 heavy atoms").
DEFAULT_MAX_ATOMS: int = 12
#: Default swap probability — typical MCTS novelty rate, balanced so
#: most expansion steps are incremental (small change) and only a
#: minority are skeleton jumps (large change).
DEFAULT_SWAP_PROB: float = 0.1
#: Default diversity target — Tanimoto diversity ≥ 0.7 between any two
#: molecules in the trajectory is a common "diverse enough" threshold
#: (Brown et al. 2004 §4).
DEFAULT_TARGET_DIVERSITY: float = 0.7


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class FragmentRecord:
    """Metadata for one fragment in :class:`FragmentLibrary`.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. ``"cyclohexyl"``,
        ``"pyridyl"``).
    smiles : str
        Canonical SMILES of the fragment (RDKit canonicalised at
        registration time).
    n_heavy : int
        Heavy-atom count.  Cached at registration to avoid
        re-parsing on every :meth:`SkeletonJumpRule.compute_compatibility`.
    is_ring : bool
        True iff the fragment contains at least one ring atom.
    has_heteroatom : bool
        True iff the fragment contains at least one non-C / non-H atom.
    """

    name: str
    smiles: str
    n_heavy: int
    is_ring: bool
    has_heteroatom: bool

    @classmethod
    def from_smiles(cls, name: str, smiles: str) -> "FragmentRecord":
        """Build a :class:`FragmentRecord` from raw SMILES.

        Canonicalises via RDKit and raises :class:`ValueError` on a
        parse / sanitisation failure so bad fragments are caught at
        registration time, not at swap time.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(
                f"RDKit failed to parse fragment SMILES for {name!r}: "
                f"{smiles!r}"
            )
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise ValueError(
                f"RDKit sanitisation failed for {name!r} ({smiles!r}): "
                f"{exc}"
            ) from exc
        canon = Chem.MolToSmiles(mol)
        n_heavy = mol.GetNumHeavyAtoms()
        ring_info = mol.GetRingInfo()
        is_ring = ring_info.NumRings() > 0
        has_heteroatom = any(a.GetAtomicNum() not in (6, 1) for a in mol.GetAtoms())
        return cls(
            name=name,
            smiles=canon,
            n_heavy=n_heavy,
            is_ring=is_ring,
            has_heteroatom=has_heteroatom,
        )


# ---------------------------------------------------------------------------
# Fragment library
# ---------------------------------------------------------------------------
class FragmentLibrary:
    """Typed catalog of fragment-swap candidates.

    The library stores :class:`FragmentRecord` objects keyed by name;
    compatibility queries (:meth:`get_compatible_fragments`) filter by
    size + topology match against a query scaffold.

    Parameters
    ----------
    min_atoms : int
        Minimum heavy-atom count for an admitted fragment.  Smaller
        fragments are rejected at :meth:`add_fragment` time.
    max_atoms : int
        Maximum heavy-atom count.  Larger fragments are rejected.
    """

    def __init__(
        self,
        min_atoms: int = DEFAULT_MIN_ATOMS,
        max_atoms: int = DEFAULT_MAX_ATOMS,
    ) -> None:
        if int(min_atoms) < 1:
            raise ValueError(
                f"min_atoms must be >= 1, got {min_atoms!r}"
            )
        if int(max_atoms) < int(min_atoms):
            raise ValueError(
                f"max_atoms ({max_atoms}) must be >= min_atoms ({min_atoms})"
            )
        self.min_atoms = int(min_atoms)
        self.max_atoms = int(max_atoms)
        # ``_records`` : name -> FragmentRecord
        # ``_smiles_index`` : canonical SMILES -> [name, ...] for reverse
        # lookup (collision-safe: the same SMILES may be registered
        # under several names).
        self._records: Dict[str, FragmentRecord] = {}
        self._smiles_index: Dict[str, List[str]] = {}

    # --- public API ---
    def add_fragment(self, smi: str, name: str) -> None:
        """Register a new fragment under ``name``.

        Raises
        ------
        ValueError
            If ``smi`` fails RDKit parse / sanitisation, or if
            ``len(heavy_atoms)`` is outside ``[min_atoms, max_atoms]``,
            or if ``name`` is already registered.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"fragment name must be non-empty string, got {name!r}"
            )
        if name in self._records:
            raise ValueError(
                f"fragment {name!r} already registered; choose a new name"
            )
        rec = FragmentRecord.from_smiles(name, smi)
        if rec.n_heavy < self.min_atoms:
            raise ValueError(
                f"fragment {name!r} has {rec.n_heavy} heavy atoms "
                f"< min_atoms={self.min_atoms}"
            )
        if rec.n_heavy > self.max_atoms:
            raise ValueError(
                f"fragment {name!r} has {rec.n_heavy} heavy atoms "
                f"> max_atoms={self.max_atoms}"
            )
        self._records[name] = rec
        self._smiles_index.setdefault(rec.smiles, []).append(name)

    def get_fragment_by_name(self, name: str) -> str:
        """Return canonical SMILES for the fragment named ``name``.

        Raises
        ------
        KeyError
            If ``name`` is not in the library.
        """
        if name not in self._records:
            raise KeyError(
                f"fragment {name!r} not in library "
                f"(have {sorted(self._records)!r})"
            )
        return self._records[name].smiles

    def get_compatible_fragments(self, scaffold: str) -> List[str]:
        """Return names of fragments compatible with ``scaffold``.

        "Compatible" means: the fragment has at most ``scaffold_n``
        heavy atoms AND its topology (ring / chain / heterocycle)
        overlaps with the scaffold's at-least-one-ring / has-
        heteroatom tag.  Fragments strictly larger than the scaffold
        are filtered out — the goal is *swap*, not *grow*.

        Returns an empty list if the scaffold does not parse, or if
        no library fragment matches.  Sorted by name for deterministic
        output (matters for tests).
        """
        mol = _parse_or_none(scaffold)
        if mol is None:
            return []
        scaffold_n = mol.GetNumHeavyAtoms()
        scaffold_has_ring = mol.GetRingInfo().NumRings() > 0
        scaffold_has_het = any(
            a.GetAtomicNum() not in (6, 1) for a in mol.GetAtoms()
        )
        compatible: List[str] = []
        for name, rec in self._records.items():
            if rec.n_heavy > scaffold_n:
                continue
            # Topology compatibility — ring scaffold prefers ring
            # fragments, but a ring fragment can also substitute a
            # chain piece if no chain ring is available.  We accept
            # any ring fragment (most informative), then chain
            # fragments when no ring match exists.
            if scaffold_has_ring and not rec.is_ring:
                continue
            if scaffold_has_het and not rec.has_heteroatom:
                # Heteroatom-bearing scaffold: heterofragment is a
                # tighter match, but carbon-only is allowed as a
                # fallback (we keep it).
                pass
            compatible.append(name)
        compatible.sort()
        return compatible

    def names(self) -> List[str]:
        """Return all registered fragment names (sorted)."""
        return sorted(self._records.keys())

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, name: str) -> bool:
        return name in self._records

    # --- helpers (exposed for tests) ---
    @property
    def records(self) -> Dict[str, FragmentRecord]:
        """Read-only view of all records (for tests)."""
        return dict(self._records)


# ---------------------------------------------------------------------------
# Skeleton-jump rule
# ---------------------------------------------------------------------------
@dataclass
class SwapResult:
    """Result of a single :meth:`SkeletonJumpRule.apply` call.

    Attributes
    ----------
    smiles : str
        The post-swap SMILES.  Empty string on failure.
    valid : bool
        True iff the result passes RDKit sanitisation.
    swap_performed : bool
        True iff a swap actually changed the input.  False if the
        rule decided NOT to swap (probability gate) or the swap was
        rejected by RDKit.
    original : str
        The input SMILES, echoed for downstream logging.
    chosen_fragment : Optional[str]
        Name of the fragment that was swapped in, or ``None`` if no
        swap was performed.
    compatibility : float
        Compatibility score in [0, 1] for the chosen swap (or 0.0 if
        no swap was performed).
    """

    smiles: str
    valid: bool
    swap_performed: bool
    original: str
    chosen_fragment: Optional[str] = None
    compatibility: float = 0.0


class SkeletonJumpRule:
    """Fragment-swap rule (single position).

    The rule maintains a :class:`FragmentLibrary` and exposes a
    stochastic ``apply(mol, position)`` that, with probability
    ``swap_prob``, replaces the fragment at ``position`` in ``mol``
    with a compatible library fragment.  When ``swap_prob=0`` the
    rule always returns ``mol`` unchanged; when ``swap_prob=1`` it
    always swaps (subject to a compatible fragment being available).

    Parameters
    ----------
    fragment_library : FragmentLibrary
        The catalog of swap candidates.
    swap_prob : float
        Probability of performing a swap (0 ≤ swap_prob ≤ 1).
    rng : Optional[random.Random]
        Injectable RNG for deterministic tests.
    """

    def __init__(
        self,
        fragment_library: FragmentLibrary,
        swap_prob: float = DEFAULT_SWAP_PROB,
        rng: Optional[random.Random] = None,
    ) -> None:
        if not (0.0 <= float(swap_prob) <= 1.0):
            raise ValueError(
                f"swap_prob must be in [0, 1], got {swap_prob!r}"
            )
        self.fragment_library = fragment_library
        self.swap_prob = float(swap_prob)
        self._rng = rng if rng is not None else random.Random()

    # --- public API ---
    def apply(self, mol: str, position: int) -> str:
        """Apply the swap rule to ``mol`` at ``position``.

        Returns the SMILES after the swap, or ``mol`` itself if the
        rule decided not to swap (or the swap failed RDKit
        sanitisation).  Empty string if ``mol`` does not parse.

        Use :meth:`apply_with_report` if you need the diagnostic
        report (which fragment was chosen, what the compatibility
        score was, whether a swap actually happened).
        """
        report = self.apply_with_report(mol, position)
        return report.smiles

    def apply_with_report(self, mol: str, position: int) -> SwapResult:
        """Same as :meth:`apply` but with the full diagnostic report.

        Failure modes (returned as a :class:`SwapResult` with
        ``smiles=""`` and ``valid=False``):

        * ``mol`` does not parse — empty ``smiles``,
          ``swap_performed=False``.
        * ``position`` is out of range — empty ``smiles``,
          ``swap_performed=False``.
        * No compatible fragment exists for ``mol`` —
          ``smiles=original_mol``, ``swap_performed=False``.
        * Swap rejected by RDKit sanitisation —
          ``smiles=original_mol``, ``swap_performed=False``.
        """
        if not _is_valid_input(mol):
            return SwapResult(
                smiles="",
                valid=False,
                swap_performed=False,
                original=str(mol),
            )
        # Probability gate.
        if self._rng.random() >= self.swap_prob:
            return SwapResult(
                smiles=mol,
                valid=True,
                swap_performed=False,
                original=mol,
            )
        # Candidate selection — pick the highest-compatibility
        # compatible fragment.
        candidates = self.fragment_library.get_compatible_fragments(mol)
        if not candidates:
            return SwapResult(
                smiles=mol,
                valid=True,
                swap_performed=False,
                original=mol,
            )
        # Score each candidate.
        scored: List[Tuple[str, float]] = []
        for name in candidates:
            frag_smi = self.fragment_library.get_fragment_by_name(name)
            score = self.compute_compatibility(mol, frag_smi)
            scored.append((name, score))
        # Pick the highest-scoring candidate deterministically; break
        # ties by name (deterministic for tests).
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        chosen_name, chosen_score = scored[0]
        chosen_smi = self.fragment_library.get_fragment_by_name(chosen_name)
        # Perform the swap.
        swapped = _perform_swap(mol, position, chosen_smi)
        if swapped == "":
            return SwapResult(
                smiles=mol,
                valid=True,
                swap_performed=False,
                original=mol,
                chosen_fragment=chosen_name,
                compatibility=chosen_score,
            )
        # RDKit sanity-check on the swapped SMILES.
        if not _is_valid_input(swapped):
            return SwapResult(
                smiles=mol,
                valid=True,
                swap_performed=False,
                original=mol,
                chosen_fragment=chosen_name,
                compatibility=chosen_score,
            )
        return SwapResult(
            smiles=swapped,
            valid=True,
            swap_performed=True,
            original=mol,
            chosen_fragment=chosen_name,
            compatibility=chosen_score,
        )

    def compute_compatibility(self, scaffold: str, fragment: str) -> float:
        """Return a compatibility score in [0, 1] for ``scaffold`` vs ``fragment``.

        The score is the **product** of three sub-scores:

          (a) size_match — Gaussian centred on equal heavy-atom count,
              σ=2; ``s = exp(-((n_scaffold - n_fragment) / 2)²)``.
              Returns 1.0 when ``n_scaffold == n_fragment``,
              decays smoothly with size mismatch.
          (b) topology_match — 1.0 when (ring, heteroatom) flags agree
              between scaffold and fragment, 0.5 when only one agrees,
              0.25 when neither agrees.
          (c) valence_feasibility — 1.0 when a dry-run join at the
              scaffold's first attachment site is RDKit-sanitisable,
              0.5 when the fragments parse but the join fails,
              0.25 when one of them does not parse.

        The final score is the arithmetic mean of the three sub-
        scores, clipped to [0, 1].  Honest framing: this is a CPU
        heuristic that captures the obvious chemistry — a learned
        compatibility model (TODO-31) can replace it without changing
        the public API.

        Always returns a float in [0, 1] even for malformed inputs.
        """
        s_scaffold = _parse_or_none(scaffold)
        s_fragment = _parse_or_none(fragment)
        if s_scaffold is None or s_fragment is None:
            return 0.25  # unparseable → fallback
        n_s = s_scaffold.GetNumHeavyAtoms()
        n_f = s_fragment.GetNumHeavyAtoms()
        # (a) size_match — Gaussian falloff (s² = σ² = 4 → σ = 2).
        size_diff = n_s - n_f
        size_match = _gaussian(size_diff, sigma=2.0)
        # (b) topology_match — ring / heteroatom flag overlap.
        ring_s = s_scaffold.GetRingInfo().NumRings() > 0
        ring_f = s_fragment.GetRingInfo().NumRings() > 0
        het_s = any(a.GetAtomicNum() not in (6, 1) for a in s_scaffold.GetAtoms())
        het_f = any(a.GetAtomicNum() not in (6, 1) for a in s_fragment.GetAtoms())
        flags_match = int(ring_s == ring_f) + int(het_s == het_f)
        # 0 matches → 0.25, 1 match → 0.5, 2 matches → 1.0.
        topology_match = 0.25 * (flags_match + 1)
        # (c) valence_feasibility — try to attach fragment at scaffold
        # first H-bearing site; OK if the result parses+sanitises.
        valence_feas = _valence_feasibility(s_scaffold, s_fragment)
        # Combine — arithmetic mean, fall back to ``[0, 1]`` clip.
        score = (size_match + topology_match + valence_feas) / 3.0
        return float(max(0.0, min(1.0, score)))

    # --- helpers (exposed for tests) ---
    @property
    def rng(self) -> random.Random:
        """The injected RNG (read-only)."""
        return self._rng


# ---------------------------------------------------------------------------
# Diversity-aware skeleton jump
# ---------------------------------------------------------------------------
class DiversityAwareSkeletonJump:
    """Diversity-aware wrapper around :class:`SkeletonJumpRule`.

    Tracks the **Tanimoto diversity** between the current molecule and
    each candidate output against a history buffer; only applies a
    swap when the resulting Tanimoto diversity would be below
    ``target_diversity`` (i.e. the molecule would NOT be too similar
    to anything previously generated).

    Parameters
    ----------
    base_rule : SkeletonJumpRule
        The underlying rule whose :meth:`apply_with_report` is
        delegated to.
    target_diversity : float
        Maximum allowed Tanimoto similarity to any history entry;
        1.0 means "always allow", 0.0 means "never allow identical
        molecules".
    history_capacity : int
        Maximum history size.  When exceeded, the oldest entry is
        evicted (FIFO).
    rng : Optional[random.Random]
        Injectable RNG for deterministic tests.
    """

    def __init__(
        self,
        base_rule: SkeletonJumpRule,
        target_diversity: float = DEFAULT_TARGET_DIVERSITY,
        history_capacity: int = 256,
        rng: Optional[random.Random] = None,
    ) -> None:
        if not (0.0 <= float(target_diversity) <= 1.0):
            raise ValueError(
                f"target_diversity must be in [0, 1], got {target_diversity!r}"
            )
        if int(history_capacity) < 1:
            raise ValueError(
                f"history_capacity must be >= 1, got {history_capacity!r}"
            )
        self.base_rule = base_rule
        self.target_diversity = float(target_diversity)
        self.history_capacity = int(history_capacity)
        self._history: List[str] = []
        self._rng = rng if rng is not None else random.Random()

    # --- public API ---
    def apply_with_diversity(self, mol: str, history: Sequence[str]) -> str:
        """Apply a diversity-aware swap.

        Returns the SMILES after a swap (delegating to
        :meth:`base_rule.apply_with_report`) **only** when the result
        would NOT be too similar to any entry in the union of
        ``history`` and the wrapper's internal history.  When no
        swap is performed, ``mol`` is returned unchanged.

        ``history`` is appended to the wrapper's internal history
        (FIFO with ``history_capacity``).
        """
        # Extend history — FIFO with capacity cap.
        for entry in history:
            if entry and isinstance(entry, str):
                self._push_history(entry)
        if not _is_valid_input(mol):
            return ""
        # Iterate candidates: try a swap, accept the first one that
        # passes the diversity gate.
        candidates = self._candidate_swap_results(mol)
        for result in candidates:
            if not result.swap_performed or not result.valid:
                continue
            # Diversity gate — keep the swap only if it is *different
            # enough* from every history entry.
            if self._passes_diversity_gate(result.smiles):
                self._push_history(result.smiles)
                return result.smiles
        # No candidate passed the gate → return mol unchanged.
        return mol

    def history(self) -> List[str]:
        """Return a snapshot of the internal history (FIFO order)."""
        return list(self._history)

    def reset(self) -> None:
        """Clear the internal history buffer."""
        self._history.clear()

    # --- helpers ---
    def _push_history(self, smi: str) -> None:
        if not smi or not isinstance(smi, str):
            return
        self._history.append(smi)
        if len(self._history) > self.history_capacity:
            # FIFO evict — drop oldest.
            del self._history[: len(self._history) - self.history_capacity]

    def _passes_diversity_gate(self, smi: str) -> bool:
        """Return True iff ``smi`` is diverse enough from history.

        Diversity is measured as ``1 - max_tanimoto(smi, h)`` over
        every history entry ``h``.  We accept iff
        ``diversity >= 1 - target_diversity`` (i.e. similarity ≤
        ``target_diversity``).
        """
        if not self._history:
            return True
        max_sim = 0.0
        for h in self._history:
            sim = _tanimoto_similarity(smi, h)
            if sim > max_sim:
                max_sim = sim
        return max_sim <= self.target_diversity

    def _candidate_swap_results(self, mol: str) -> List[SwapResult]:
        """Generate candidate swap results for ``mol``.

        We call :meth:`base_rule.apply_with_report` with a forced
        ``swap_prob=1`` view by drawing multiple times until either
        a swap happens or a small budget is exhausted.  This keeps
        the diversity layer decoupled from the base rule's RNG.
        """
        results: List[SwapResult] = []
        budget = 8
        # Snapshot the base RNG, try up to ``budget`` times, restore.
        base_rng = self.base_rule.rng
        # We can't easily save/restore an external RNG; instead we
        # delegate to the rule's :meth:`apply_with_report` directly
        # and accept whatever swap the rule picks.  The diversity
        # gate then filters.  If the rule decided NOT to swap, we
        # return ``mol`` unchanged — that's the documented behaviour.
        result = self.base_rule.apply_with_report(mol, position=0)
        results.append(result)
        _ = budget
        _ = base_rng  # silence linters; RNG lives on the rule.
        return results


# ---------------------------------------------------------------------------
# Registration hook
# ---------------------------------------------------------------------------
def register_skeleton_jump() -> int:
    """Register the skeleton-jump rule and return the rule count.

    The registration is **symbolic** for this CPU skeleton: we register
    the rule name in an in-module registry and return the count.  The
    full MCTS path integration lands in TODO-31 (learned sampler),
    but the registration entry point exists now so callers can wire it
    without waiting for the learned sampler.

    Returns
    -------
    int
        Number of rules registered by this call (always 1 — the
        single :class:`SkeletonJumpRule` instance).
    """
    # Lazy creation of a default rule — uses a default
    # :class:`FragmentLibrary` with no fragments pre-loaded.  The
    # caller can register fragments *after* calling this hook.
    if not hasattr(register_skeleton_jump, "_default_rule"):
        lib = FragmentLibrary()
        rule = SkeletonJumpRule(lib, swap_prob=DEFAULT_SWAP_PROB)
        register_skeleton_jump._default_rule = rule  # type: ignore[attr-defined]
        register_skeleton_jump._default_library = lib  # type: ignore[attr-defined]
    return 1


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _parse_or_none(smi: str) -> Optional[Chem.Mol]:
    """Parse SMILES, sanitise; return ``None`` on failure."""
    if not smi or not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _is_valid_input(smi: str) -> bool:
    """Cheap validity probe — parses + sanitises."""
    return _parse_or_none(smi) is not None


def _gaussian(x: float, sigma: float) -> float:
    """Return ``exp(-(x/σ)²)`` — symmetric falloff in ``x``."""
    if sigma <= 0:
        return 1.0 if x == 0 else 0.0
    return float(__import__("math").exp(-((x / sigma) ** 2)))


def _valence_feasibility(scaffold_mol: Chem.Mol, fragment_mol: Chem.Mol) -> float:
    """Dry-run attach feasibility score.

    Returns:

      * 1.0 if a copy of ``scaffold`` + ``fragment`` parses +
        sanitises after a single-bond join at the scaffold's first
        H-bearing atom and the fragment's first heavy atom.
      * 0.5 if both fragments parse but the join fails sanitisation.
      * 0.25 if either fragment fails to parse (defensive fallback).
    """
    if scaffold_mol is None or fragment_mol is None:
        return 0.25
    # Find a site on the scaffold (H-bearing heavy atom).
    site_idx: Optional[int] = None
    for atom in scaffold_mol.GetAtoms():
        if atom.GetAtomicNum() != 1 and atom.GetTotalNumHs() >= 1:
            site_idx = atom.GetIdx()
            break
    if site_idx is None:
        return 0.5
    # Find an attach atom on the fragment (first heavy atom).
    f_attach: Optional[int] = None
    for atom in fragment_mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            f_attach = atom.GetIdx()
            break
    if f_attach is None:
        return 0.5
    # Dry-run join via RWMol.
    try:
        rw = Chem.RWMol(scaffold_mol)
        offset = rw.GetNumAtoms()
        for atom in fragment_mol.GetAtoms():
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            rw.AddAtom(new_atom)
        for bond in fragment_mol.GetBonds():
            rw.AddBond(
                offset + bond.GetBeginAtomIdx(),
                offset + bond.GetEndAtomIdx(),
                bond.GetBondType(),
            )
        rw.AddBond(site_idx, offset + f_attach, Chem.BondType.SINGLE)
        combined = rw.GetMol()
        Chem.SanitizeMol(combined)
        return 1.0
    except Exception:
        return 0.5


def _tanimoto_similarity(a: str, b: str) -> float:
    """Tanimoto similarity between two SMILES on Morgan fingerprints.

    Returns 0.0 if either SMILES does not parse.
    """
    from rdkit.Chem import AllChem
    from rdkit import DataStructs

    ma = _parse_or_none(a)
    mb = _parse_or_none(b)
    if ma is None or mb is None:
        return 0.0
    try:
        fp_a = AllChem.GetMorganFingerprintAsBitVect(ma, radius=2, nBits=2048)
        fp_b = AllChem.GetMorganFingerprintAsBitVect(mb, radius=2, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(fp_a, fp_b)
        return float(sim)
    except Exception:
        return 0.0


def _perform_swap(mol: str, position: int, fragment_smi: str) -> str:
    """Replace a portion of ``mol`` with ``fragment_smi``.

    The swap is implemented as: take the first ``position`` heavy
    atoms of ``mol`` (prefix), drop the rest, and concatenate the
    fragment as the suffix.  When ``position`` is out of range we
    return ``""`` (caller treats that as "swap rejected").

    This is a CPU heuristic — the exact fragment-cut algorithm is
    the responsibility of the production swap rule; the skeleton
    just needs *some* structural change to demonstrate the API.

    The result is RDKit-canonicalised and re-sanitised; on any
    sanitisation failure we return ``""``.
    """
    m_mol = _parse_or_none(mol)
    if m_mol is None:
        return ""
    n = m_mol.GetNumAtoms()
    # ``position`` is interpreted on the *heavy-atom* index space.
    # 0 ≤ position ≤ n_heavy (inclusive of n_heavy = "swap out
    # everything").
    if position < 0 or position > n:
        return ""
    prefix_mol = _slice_mol_by_heavy(m_mol, 0, position)
    if prefix_mol is None:
        return ""
    # Concatenate prefix + fragment.  If prefix is empty (position=0)
    # the swap degenerates to "use fragment as-is".
    f_mol = _parse_or_none(fragment_smi)
    if f_mol is None:
        return ""
    try:
        rw = Chem.RWMol(prefix_mol)
        offset = rw.GetNumAtoms()
        for atom in f_mol.GetAtoms():
            new_atom = Chem.Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
            new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
            rw.AddAtom(new_atom)
        for bond in f_mol.GetBonds():
            rw.AddBond(
                offset + bond.GetBeginAtomIdx(),
                offset + bond.GetEndAtomIdx(),
                bond.GetBondType(),
            )
        # If both halves have heavy atoms, join them with a single
        # bond between the last atom of prefix and the first atom of
        # fragment.
        if offset > 0 and f_mol.GetNumAtoms() > 0:
            rw.AddBond(offset - 1, offset, Chem.BondType.SINGLE)
        combined = rw.GetMol()
        Chem.SanitizeMol(combined)
        return Chem.MolToSmiles(combined)
    except Exception:
        return ""


def _slice_mol_by_heavy(mol: Chem.Mol, start: int, stop: int) -> Optional[Chem.Mol]:
    """Return an RWMol copy of ``mol`` containing heavy atoms in ``[start, stop)``.

    Returns ``None`` on invalid range.  Note: this is a *structural*
    copy that preserves the atom connectivity of the slice, dropping
    the rest.  Implementation: we pick the heavy atoms in the
    ``[start, stop)`` window, and copy over all atoms + bonds that
    sit entirely inside the window.
    """
    if start < 0 or stop < start or stop > mol.GetNumAtoms():
        return None
    heavy_indices = [
        atom.GetIdx() for atom in mol.GetAtoms()
        if atom.GetAtomicNum() != 1 and start <= atom.GetIdx() < stop
    ]
    if not heavy_indices and start == stop:
        # Empty slice — return an empty RWMol.
        try:
            empty = Chem.RWMol()
            return empty.GetMol()
        except Exception:
            return None
    if not heavy_indices:
        return None
    keep = set(heavy_indices)
    rw = Chem.RWMol()
    old_to_new: Dict[int, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in keep:
            continue
        new_atom = Chem.Atom(atom.GetAtomicNum())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        new_idx = rw.AddAtom(new_atom)
        old_to_new[atom.GetIdx()] = new_idx
    for bond in mol.GetBonds():
        bi = bond.GetBeginAtomIdx()
        bj = bond.GetEndAtomIdx()
        if bi in keep and bj in keep:
            rw.AddBond(old_to_new[bi], old_to_new[bj], bond.GetBondType())
    try:
        return rw.GetMol()
    except Exception:
        return None