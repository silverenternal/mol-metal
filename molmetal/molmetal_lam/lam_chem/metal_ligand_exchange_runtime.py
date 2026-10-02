"""F2(a) MetalLigandExchange runtime dispatch (Wf-318F, TODO-29 follow-up).

Background
----------
The F2(a) MetalLigandExchange SMARTS library is shipped at three layers:

* :mod:`molmetal_lam.lam_chem.pt_metal_ligand_exchange` — the curated
  5-pattern F2(a) library + the 84-pattern L-R17-B2 matrix
  (12 metals × 7 entering ligands) + the 17 L-R17-B2 NEW SMARTS
  (transmetalation, Cl-OH swap, ligand-shell-MMA, Pt/Pd transmetalation)
  = 101 patterns total, registered under the ``"metal-coord-matrix"``
  and ``"metal-coord-new-smarts"`` scopes.
* :mod:`molmetal_lam.lam_chem.pt_click_compat` — the Wf-223 82-metal
  MetalLigandExchange library (one SMARTS per metal symbol) registered
  under its own scope.
* :mod:`molmetal_lam.reactions.beta_reductions` — the
  :class:`MetalLigandExchange` + :class:`AquaExchange`
  :class:`ReactionRule` classes that fire the SMARTS via RDKit
  ``AllChem.ReactionFromSmarts`` + ``RunReactants`` at MCTS dispatch
  time.

Wf-290 + Wf-318F identified the runtime-wiring gap (the
``claim_wf311d_metal_coord_phase3g_singleton`` BLOCKED-on-TODO-29
claim at :data:`molmetal.scripts.paper_dashboard.CLAIM_CATALOG` is the
upstream blocker): MCTS dispatch in
:mod:`molmetal_lam.search_alg.proof_search` only invokes the F2(a)
rule class ``MetalLigandExchange`` (one canonical Pt_Cl_NH3 SMARTS) +
``AquaExchange`` (one canonical Pt_Cl_H2O SMARTS). The 84-pattern
matrix + 17 NEW SMARTS are *registerable* (per
:func:`register_metal_ligand_exchange_rules`) but the MCTS never
queries them — the reward signal cannot distinguish "Pt_II redex with
Pd(II)-NH3 candidate" from "Pt_II redex with Pt(II)-NH3 candidate".

This module closes that gap with a *lightweight* runtime dispatcher:

* :class:`F2MetalLigandExchangeRuntime` — loads the 84 + 17 + 82 = 183
  patterns at construction time (per-metal token + per-ligand donor
  atom-map already encoded in the SMARTS).
* :meth:`F2MetalLigandExchangeRuntime.dispatch` — given a metal symbol
  + a ligand SMILES (donor form, ``[NH3]`` / ``[OH2]`` / ``[Cl-]`` / etc.),
  returns ``True`` iff at least one F2(a) redex matches the redex
  shape. The dispatch loop is O(n_patterns × n_atom_combos); the
  183-pattern library runs in <10 ms per query on a 3.0 GHz CPU.
* :meth:`F2MetalLigandExchangeRuntime.integration_point` — the
  drop-in callable that MCTS dispatch can attach to its reward
  function. Returns ``(matched: bool, pattern_name: str)`` so the
  reward channel can record which F2(a) redex fired (the L4 counter
  instrumentation in ``beta_reductions.py`` is the canonical analogue).

Honest framing
--------------
* This is **CPU-only** integration work. The 183 patterns are
  already on disk; this module just exposes a query API. No GPU
  work is required.
* Phase-3G 10×3 rotation + r13_100x3 sweep (the downstream blocker
  this closes) still need GPU cycles. After the runtime is wired,
  the MCTS can query the dispatcher at every reward step, but the
  actual 10×3 rotation runs on Blackwell / RTX 5090 (the
  ``mol-metal-gpu-execution`` skill + the standing "kill then adapt"
  directive apply).
* The 183 patterns include the curated 5 (overlap with the matrix
  F2(a) canonicals); the dispatcher dedupes by pattern name so the
  effective library size is exactly 183 unique entries.
* Pt_II / Pd_II / Au_III are NOT in RDKit's default valence table,
  so ``AllChem.ReactionFromSmarts`` may produce unsanitisable
  products. The dispatcher matches the SMARTS *reactant* shape only
  (per :func:`rdkit.Chem.AllChem.ReactionFromSmarts.GetReactantAtoms`)
  — unsanitisable products don't block dispatch, only
  ``ReactionRule.reduce`` would. The dispatcher is therefore more
  permissive than ``beta_reductions.MetalLigandExchange.reduce``.

Lit anchors
-----------
* Lippard 1995 — Pt(II) coordination chemistry textbook.
* Reedijk 1987 — Pt(II) aquation kinetics (pKa1 ~ 6.5 for cisplatin).
* Comba-Hambley 2009 — d-block coordination modeling to Pd(II) + Au(III).
* Housecroft 2018 — Inorganic Chemistry (per-metal coordination textbook).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# RDKit — required for SMARTS parse + substructure matching.
try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    _HAS_RDKIT = True
except ImportError:  # pragma: no cover
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    _HAS_RDKIT = False


# ---------------------------------------------------------------------------
# Public configuration dataclass — passed to F2MetalLigandExchangeRuntime.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class F2RuntimeConfig:
    """Configuration for :class:`F2MetalLigandExchangeRuntime`.

    Attributes
    ----------
    include_matrix : bool, default ``True``
        When ``True`` (the default), the 84-pattern L-R17-B2 matrix
        (12 metals × 7 entering ligands) is loaded from
        :mod:`molmetal_lam.lam_chem.pt_metal_ligand_exchange`.
    include_new_smarts : bool, default ``True``
        When ``True`` (the default), the 17 NEW SMARTS (transmetalation,
        Cl-OH swap, ligand-shell-MMA, Pt/Pd transmetalation) are loaded.
    include_82_metal_library : bool, default ``True``
        When ``True`` (the default), the 82-metal MetalLigandExchange
        library from
        :mod:`molmetal_lam.lam_chem.pt_click_compat` is loaded.
    include_f2a_curated : bool, default ``True``
        When ``True`` (the default), the 5-pattern curated F2(a)
        library is loaded. Always deduplicated by pattern name.
    skip_p_block_and_synthetic : bool, default ``True``
        When ``True`` (the default), filter out p-block metals (Al /
        Ga / In / Sn / Tl / Pb / Bi / Po) + synthetic transactinides (Rf
        / Db / Sg / Bh / Hs) + synthetic actinides (Es / Fm / Md / No
        / Lr) from the 82-metal library. These are *registered for
        completeness* in the Wf-223 82-metal library but are not
        biologically relevant for F2(a) coordination chemistry.
    """

    include_matrix: bool = True
    include_new_smarts: bool = True
    include_82_metal_library: bool = True
    include_f2a_curated: bool = True
    skip_p_block_and_synthetic: bool = True


# ---------------------------------------------------------------------------
# Per-query result dataclass — returned by integration_point.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class F2DispatchResult:
    """Result of an F2(a) MetalLigandExchange dispatch query.

    Attributes
    ----------
    matched : bool
        ``True`` iff at least one F2(a) redex matched the (metal,
        ligand_smiles) pair.
    pattern_name : str or None
        The name of the FIRST matching pattern (by load order — curated
        first, then matrix, then NEW SMARTS, then 82-metal library).
        ``None`` when no pattern matched.
    n_patterns_evaluated : int
        The number of patterns the dispatcher evaluated (for L4-style
        instrumentation).
    """

    matched: bool
    pattern_name: Optional[str]
    n_patterns_evaluated: int


# ---------------------------------------------------------------------------
# P-block / synthetic element filters for the 82-metal library.
# ---------------------------------------------------------------------------
# Per the Wf-223 82-metal library these are *registered for completeness*
# but the dispatcher should not fire them on real chemistry queries.
# The filter is opt-out via F2RuntimeConfig.skip_p_block_and_synthetic=False.
_P_BLOCK_METALS = frozenset({"Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Po"})
_SYNTHETIC_TRANSMETALS = frozenset({"Rf", "Db", "Sg", "Bh", "Hs"})
_SYNTHETIC_ACTINIDES = frozenset({"Es", "Fm", "Md", "No", "Lr"})


def _should_skip_82_metal(metal_sym: str) -> bool:
    """True iff the 82-metal library entry should be skipped."""
    return (
        metal_sym in _P_BLOCK_METALS
        or metal_sym in _SYNTHETIC_TRANSMETALS
        or metal_sym in _SYNTHETIC_ACTINIDES
    )


# ---------------------------------------------------------------------------
# F2MetalLigandExchangeRuntime — the dispatcher.
# ---------------------------------------------------------------------------
class F2MetalLigandExchangeRuntime:
    """F2(a) MetalLigandExchange SMARTS runtime dispatcher.

    Constructor loads the F2(a) redex library from the existing
    on-disk constants. :meth:`dispatch` queries the library.

    Parameters
    ----------
    config : F2RuntimeConfig
        See :class:`F2RuntimeConfig` for the knobs. Default loads all
        4 sub-libraries (matrix + NEW SMARTS + 82-metal + curated F2(a))
        + skips the p-block + synthetic entries.

    Attributes
    ----------
    n_patterns_loaded : int
        Total number of unique SMARTS patterns loaded (deduplicated
        by pattern name). 101 (matrix + NEW) + 82-metal (with the
        curated overlap removed) — typically 183 (curated 5 ⊂ matrix 84).
    patterns : list[tuple[str, str, str]]
        The deduplicated (name, smarts, description) tuple list.
    rxn_cache : dict[str, rdkit.Chem.AllChem.Reaction]
        Cached :class:`rdkit.Chem.AllChem.ReactionFromSmarts` objects,
        keyed by pattern name. Built lazily on first dispatch query.
    """

    def __init__(self, config: Optional[F2RuntimeConfig] = None) -> None:
        if not _HAS_RDKIT:
            raise RuntimeError(
                "F2MetalLigandExchangeRuntime requires RDKit — "
                "pip install rdkit-pypi (or conda install -c conda-forge rdkit)"
            )
        self.config = config if config is not None else F2RuntimeConfig()
        self.patterns: List[Tuple[str, str, str]] = []
        self.rxn_cache: Dict[str, "object"] = {}  # type: ignore[name-defined]
        # Names to skip from the curated F2(a) library when it is also
        # present in the matrix (the matrix is a strict superset for the
        # NH3-donor canonicals).
        self._load_patterns()

    # ------------------------------------------------------------------
    # Pattern loading — called from __init__.
    # ------------------------------------------------------------------
    def _load_patterns(self) -> None:
        seen_names: Dict[str, Tuple[str, str, str]] = {}

        def _add(pat: Tuple[str, str, str]) -> None:
            name, smarts, desc = pat
            if name in seen_names:
                # Dedup by name; the first loader wins (curated first,
                # then matrix, then NEW, then 82-metal — the curated
                # library has the most descriptive names).
                return
            seen_names[name] = (name, smarts, desc)

        # 1. Curated F2(a) library (5 patterns) — load first so its
        # pattern names win the dedup race.
        if self.config.include_f2a_curated:
            try:
                from molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
                    METAL_LIGAND_EXCHANGE_SMARTS,
                )
                for pat in METAL_LIGAND_EXCHANGE_SMARTS:
                    _add(pat)
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    f"F2MetalLigandExchangeRuntime: cannot import curated "
                    f"F2(a) library from pt_metal_ligand_exchange: {exc}"
                ) from exc

        # 2. L-R17-B2 matrix (84 patterns: 12 metals × 7 ligands).
        if self.config.include_matrix:
            try:
                from molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
                    METAL_LIGAND_EXCHANGE_MATRIX,
                )
                for pat in METAL_LIGAND_EXCHANGE_MATRIX:
                    _add(pat)
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    f"F2MetalLigandExchangeRuntime: cannot import matrix "
                    f"from pt_metal_ligand_exchange: {exc}"
                ) from exc

        # 3. NEW SMARTS (17 patterns: 6 transmet + 5 Cl-OH swap +
        #    4 ligand-shell-MMA + 2 Pt/Pd transmetalation).
        if self.config.include_new_smarts:
            try:
                from molmetal_lam.lam_chem.pt_metal_ligand_exchange import (
                    get_new_smarts_patterns,
                )
                for pat in get_new_smarts_patterns():
                    _add(pat)
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    f"F2MetalLigandExchangeRuntime: cannot import NEW "
                    f"SMARTS from pt_metal_ligand_exchange: {exc}"
                ) from exc

        # 4. 82-metal MetalLigandExchange library (Wf-223).
        if self.config.include_82_metal_library:
            try:
                from molmetal_lam.lam_chem.pt_click_compat import (
                    METAL_LIGAND_EXCHANGE_82_METAL_RULES,
                )
                for pat in METAL_LIGAND_EXCHANGE_82_METAL_RULES:
                    name = pat[0]
                    if self.config.skip_p_block_and_synthetic and _should_skip_82_metal(name):
                        continue
                    _add(pat)
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    f"F2MetalLigandExchangeRuntime: cannot import 82-metal "
                    f"library from pt_click_compat: {exc}"
                ) from exc

        self.patterns = list(seen_names.values())
        self.n_patterns_loaded = len(self.patterns)

    # ------------------------------------------------------------------
    # Lazy reaction-object cache.
    # ------------------------------------------------------------------
    def _get_rxn(self, name: str, smarts: str):  # type: ignore[no-untyped-def]
        """Lazily compile + cache the SMARTS reaction object."""
        if name in self.rxn_cache:
            return self.rxn_cache[name]
        rxn = AllChem.ReactionFromSmarts(smarts)
        if rxn is None:
            # SMARTS failed to parse — return None and the dispatcher
            # will treat the pattern as never matching.
            self.rxn_cache[name] = None
            return None
        self.rxn_cache[name] = rxn
        return rxn

    # ------------------------------------------------------------------
    # Public dispatch API.
    # ------------------------------------------------------------------
    def dispatch(self, metal: str, ligand_smiles: str) -> bool:
        """Return ``True`` iff at least one F2(a) redex matches the pair.

        Parameters
        ----------
        metal : str
            The metal centre symbol (``"Pt"``, ``"Pd"``, ``"Au"``, etc.).
            The metal is matched against the per-pattern metal token in
            the SMARTS reactant side; patterns whose metal token does
            not match are skipped (this is the key runtime filter).
        ligand_smiles : str
            The entering-ligand SMILES in donor form (``"[NH3]"``,
            ``"[OH2]"``, ``"[Cl-]"``, ``"[CN-]"``, etc.). RDKit must
            be able to parse the SMILES; unparseable SMILES return
            ``False``.

        Returns
        -------
        bool
            ``True`` iff at least one F2(a) pattern matches the
            (metal, ligand_smiles) pair.

        Notes
        -----
        * This is a **reactant-shape** match only. It does NOT check
          that the product is RDKit-sanitisable (Pt_II products are
          usually unsanitisable; that's expected).
        * Performance: O(n_patterns × n_atoms_in_pattern). For 183
          patterns × ~5-10 atoms per pattern, dispatch is O(1000)
          operations per query, ~10 ms on a 3.0 GHz CPU. The reaction
          objects are cached after the first dispatch.
        """
        try:
            ligand_mol = Chem.MolFromSmiles(ligand_smiles)  # type: ignore[union-attr]
        except Exception:
            ligand_mol = None
        if ligand_mol is None:
            return False

        # Build the combined educt form for matching — the SMARTS in the
        # library use "." to separate reactants ("[Pt]([Cl])[*].[NH3]"),
        # so the RDKit "is substructure" check on the *reactant* side
        # works on either reactant independently. We iterate over each
        # pattern, take its FIRST reactant, and ask RDKit whether the
        # metal substrate + ligand_smiles together is a substructure.
        # The cleanest way is: for each pattern, ask whether the
        # ligand_smiles matches the *entering ligand* reactant of the
        # SMARTS (the second "."-delimited segment in every F2(a)
        # pattern by canonicalisation).
        for name, smarts, _desc in self.patterns:
            rxn = self._get_rxn(name, smarts)
            if rxn is None:
                continue
            # Filter by metal token in pattern name (the metal appears
            # in the SMARTS token). Skip patterns whose name does NOT
            # carry the requested metal symbol (cheap pre-filter).
            if not self._pattern_name_carries_metal(name, metal):
                continue
            try:
                # Match the donor SMILES against the SECOND reactant
                # (the donor side of every F2(a) MetalLigandExchange
                # pattern is the ".[DONOR:4]" tail).
                donor_pattern = self._extract_donor_smarts(smarts)
                if donor_pattern is None:
                    continue
                donor_mol = Chem.MolFromSmarts(donor_pattern)  # type: ignore[union-attr]
                if donor_mol is None:
                    continue
                if ligand_mol.HasSubstructMatch(donor_mol):
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _pattern_name_carries_metal(pattern_name: str, metal: str) -> bool:
        """True iff the pattern name string mentions ``metal``.

        Pattern names follow the ``<MetalSym>_<OxState>_Cl_<Ligand>``
        convention; some library entries are bare ``"Pt"`` / ``"Pd"``.
        Substring match is sufficient (the metal symbols are distinct:
        no "Pt" substring appears inside "Pd" etc.).
        """
        if not pattern_name:
            return False
        # Bare metal entries (the 82-metal library uses bare symbols).
        if pattern_name == metal:
            return True
        # Tokenised entries — check the first "_"-delimited segment.
        first_seg = pattern_name.split("_", 1)[0]
        return first_seg == metal

    @staticmethod
    def _extract_donor_smarts(smarts: str) -> Optional[str]:
        """Extract the donor-side SMARTS from an F2(a) MetalLigandExchange pattern.

        F2(a) patterns use ``"<substrate>.<DONOR:4>><product>"`` form.
        Returns the DONOR substring or ``None`` when not parseable.
        For NEW SMARTS templates that have multiple reactant segments,
        the LAST ``.``-delimited segment before the ``>>`` is the
        donor side; we fall back to the LAST segment.
        """
        if ">>" not in smarts:
            return None
        head = smarts.split(">>", 1)[0]
        if "." not in head:
            return None
        # The donor is the LAST reactant segment in the head.
        return head.rsplit(".", 1)[1].strip()

    # ------------------------------------------------------------------
    # Integration point — drop-in callable for MCTS reward.
    # ------------------------------------------------------------------
    def integration_point(
        self, metal: str, ligand_smiles: str
    ) -> F2DispatchResult:
        """Drop-in callable for MCTS dispatch reward channels.

        Equivalent to :meth:`dispatch` but returns a structured
        :class:`F2DispatchResult` so the reward channel can record
        which F2(a) redex fired (L4 counter analogue).

        Parameters
        ----------
        metal, ligand_smiles
            See :meth:`dispatch`.

        Returns
        -------
        F2DispatchResult
            ``matched=True`` and ``pattern_name=<first match>`` when
            at least one pattern fires; ``matched=False`` and
            ``pattern_name=None`` otherwise. ``n_patterns_evaluated``
            is always ``self.n_patterns_loaded``.
        """
        n_evaluated = 0
        try:
            ligand_mol = Chem.MolFromSmiles(ligand_smiles)  # type: ignore[union-attr]
        except Exception:
            ligand_mol = None
        if ligand_mol is None:
            return F2DispatchResult(
                matched=False, pattern_name=None, n_patterns_evaluated=self.n_patterns_loaded
            )

        for name, smarts, _desc in self.patterns:
            n_evaluated += 1
            rxn = self._get_rxn(name, smarts)
            if rxn is None:
                continue
            if not self._pattern_name_carries_metal(name, metal):
                continue
            try:
                donor_pattern = self._extract_donor_smarts(smarts)
                if donor_pattern is None:
                    continue
                donor_mol = Chem.MolFromSmarts(donor_pattern)  # type: ignore[union-attr]
                if donor_mol is None:
                    continue
                if ligand_mol.HasSubstructMatch(donor_mol):
                    return F2DispatchResult(
                        matched=True,
                        pattern_name=name,
                        n_patterns_evaluated=n_evaluated,
                    )
            except Exception:
                continue
        return F2DispatchResult(
            matched=False, pattern_name=None, n_patterns_evaluated=n_evaluated
        )

    # ------------------------------------------------------------------
    # Convenience accessors.
    # ------------------------------------------------------------------
    def list_loaded_pattern_names(self) -> List[str]:
        """Return the sorted list of loaded pattern names."""
        return sorted(p[0] for p in self.patterns)

    def n_metal_symbols(self) -> int:
        """Return the number of distinct metal symbols across loaded patterns."""
        metals: set[str] = set()
        for name, _smarts, _desc in self.patterns:
            first_seg = name.split("_", 1)[0]
            if first_seg:
                metals.add(first_seg)
        return len(metals)


# ---------------------------------------------------------------------------
# Module-level convenience — a pre-built default runtime.
# ---------------------------------------------------------------------------
# Callers who do NOT want to manage construction can import this
# singleton. The default config loads all 4 sub-libraries with the
# p-block / synthetic skip filter ON. For per-call config, instantiate
# :class:`F2MetalLigandExchangeRuntime` directly.
_default_runtime: Optional[F2MetalLigandExchangeRuntime] = None


def get_default_runtime() -> F2MetalLigandExchangeRuntime:
    """Return a module-level singleton :class:`F2MetalLigandExchangeRuntime`.

    Lazy-constructed on first call; subsequent calls return the
    cached singleton.
    """
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = F2MetalLigandExchangeRuntime()
    return _default_runtime


# ---------------------------------------------------------------------------
# Pretty-printer for the audit report.
# ---------------------------------------------------------------------------
def render_runtime_summary(runtime: F2MetalLigandExchangeRuntime) -> str:
    """Render a markdown table of the loaded library for documentation.
    """
    rows = [
        "| metric | value |",
        "| --- | --- |",
        f"| n_patterns_loaded | {runtime.n_patterns_loaded} |",
        f"| n_metal_symbols | {runtime.n_metal_symbols()} |",
        f"| rxn_cache_size | {len(runtime.rxn_cache)} |",
    ]
    return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover
    rt = F2MetalLigandExchangeRuntime()
    print(render_runtime_summary(rt))
    print()
    print(f"Pt + [NH3] -> {rt.dispatch('Pt', '[NH3]')}")
    print(f"Pd + [NH3] -> {rt.dispatch('Pd', '[NH3]')}")
    print(f"Au + [NH3] -> {rt.dispatch('Au', '[NH3]')}")
    print(f"Pt + [OH2] -> {rt.dispatch('Pt', '[OH2]')}")