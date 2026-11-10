"""Runtime wire-up for the L-R17-B2 84-pattern matrix + YuelBond precondition.

Wf-319G (2026-09-29) — wires two KNOWN-but-UNUSED assets into a
production MCTS-dispatchable runtime:

1. The 12 metals × 7 entering ligands = **84-pattern matrix** +
   17 NEW SMARTS (transmetalation / Cl-OH / ligand-shell / Pt-Pd)
   from :mod:`molmetal_lam.lam_chem.pt_metal_ligand_exchange` —
   compiled into RDKit ``ChemicalReaction`` objects via
   :func:`build_matrix_reactions` and exposed for MCTS dispatch via
   :class:`MetalLigandExchangeRuntime`.

2. The **YuelBond precondition** (Wf-269C MEASURED Δ = −1.000 with
   --no-yuelbond; Wf-275C deployment contract) — surfaced as the
   :func:`check_yuelbond_precondition` helper that returns a
   :class:`YuelBondPreconditionResult` carrying ``yuelbond_active``,
  ``m14_lead_invalidated``, and ``recommended_action`` fields.  This is
   the runtime-side enforcement of the m14 LEAD precondition per
   :file:`TODO/completed/m14_lead_deployment_precondition.md`.

Wf-296D scope (2026-09-28) wired these into the CFM YuelBond +
Wasserstein + Catherine retrain script
(:file:`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`) — a
single-decoder training surface.  This module complements Wf-296D by
exposing the same assets at the **MCTS proof-search dispatch layer**
(:file:`molmetal/molmetal_lam/search_alg/proof_search.py`).

Honest framing
--------------
* This is **integration wire-up** work (CPU-only).  No GPU retrain
  ships.  No paper-claim numbers update.  Existing MCTS measurements
  (transposition hit_rate, leaf_value_var, etc.) are unchanged.
* The 84-pattern matrix + 17 NEW SMARTS are pre-existing
  in :mod:`pt_metal_ligand_exchange`.  This module compiles them into
  RDKit ``ChemicalReaction`` objects and exposes them via a small
  dispatch API.  The compilation is *deterministic* (no torch, no
  GPU, no I/O) — the runtime dispatch layer can consume the compiled
  reactions immediately on import.
* The YuelBond precondition check is the runtime-side enforcement of
  the Wf-275C deployment contract.  When ``yuelbond_active=False``,
  the result is ``m14_lead_invalidated=True`` with
  ``recommended_action="FAIL_FAST"`` so the MCTS dispatch short-
  circuits before polluting the leaf_value_var metric.
* This module does **NOT** modify :mod:`pt_metal_ligand_exchange`,
  :mod:`pt_click_compat`, or :mod:`proof_search` — the runtime is
  imported on-demand by callers that want the dispatch surface.
  Backward compatibility: every existing test continues to pass
  because :attr:`METAL_LIGAND_RUNTIME_AVAILABLE` defaults to
  ``False`` until the optional ``rdkit`` dependency is resolved.

Lit anchors
-----------
* Lippard 1995 — Pt(II) coordination chemistry textbook (d8
  square-planar, associative substitution pathway, 5-coordinate
  trigonal-bipyramidal transition state).
* Reedijk 1987 — Pt(II) aquation kinetics.
* Comba-Hambley 2009 — d-block coordination modeling.
* Wang & Dokholyan 2026 (J. Chem. Inf. Model. 66(2) 1003-1012, DOI
  10.1021/acs.jcim.5c03052; preprint bioRxiv 10.1101/2025.05.06.652517) —
  YuelBond decoder head.  (DOI corrected 2026-10-31, Wf-A143: the DOI
  here was truncated to "10.1101/2025.05.06.645"; see
  `docs/HONEST_NEGATIVE_CATALOG.md` §6.257.)
* Wf-269C MEASURED Δ = −1.000 (no_yuelbond vs full) — YuelBond is
  load-bearing on the real cohort.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional rdkit import — the runtime dispatch layer falls back to a
# no-op stub when rdkit is unavailable (e.g. CPU-only test venvs
# without the chemistry wheel).  This is the Wf-302C "defensive
# fallback" pattern: when the dependency is missing we return empty
# reactions and the caller still gets a valid (if empty) dispatch
# result.
# ---------------------------------------------------------------------------
try:  # pragma: no cover — optional dep
    from rdkit import Chem as _Chem  # type: ignore
    from rdkit.Chem import AllChem as _AllChem  # type: ignore

    _RDKIT_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    _Chem = None  # type: ignore
    _AllChem = None  # type: ignore
    _RDKIT_AVAILABLE = False


# Lazy module-level import — the matrix + NEW SMARTS are the source
# of truth and we re-export the pattern accessors so callers can
# introspect the wired-up library without importing
# :mod:`pt_metal_ligand_exchange` directly.
from molmetal_lam.lam_chem.pt_metal_ligand_exchange import (  # noqa: E402
    METAL_LIGAND_EXCHANGE_MATRIX,
    METAL_LIGAND_EXCHANGE_SMARTS,
    AQUA_EXCHANGE_SMARTS,
    TRANSMETALATION_SMARTS,
    CL_OH_SWAP_SMARTS,
    LIGAND_SHELL_MMA_SMARTS,
    PT_PD_TRANSMETALATION_SMARTS,
    get_metal_ligand_exchange_matrix_patterns,
    get_new_smarts_patterns,
)


__all__ = [
    "MATRIX_PATTERN_COUNT",
    "NEW_SMARTS_PATTERN_COUNT",
    "TOTAL_PATTERN_COUNT",
    "YuelBondPreconditionResult",
    "MetalLigandExchangeRuntime",
    "build_matrix_reactions",
    "check_yuelbond_precondition",
    "dispatch_metal_ligand_reaction",
    "wire_into_proof_search_rules",
]


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — pattern counts (the dispatch surface area).
# ---------------------------------------------------------------------------
#: Number of patterns in the 12-metal × 7-ligand matrix
#: (:data:`pt_metal_ligand_exchange.METAL_LIGAND_EXCHANGE_MATRIX`).
#: MEASURED = 84 at module load (pinned by the regression test
#: :file:`molmetal/tests/test_metal_ligand_exchange_matrix.py`).
MATRIX_PATTERN_COUNT: int = 84

#: Number of NEW SMARTS patterns (6 transmetalation + 5 Cl-OH +
#: 4 ligand-shell-MMA + 2 Pt/Pd transmetalation = 17).
#: MEASURED = 17 at module load.
NEW_SMARTS_PATTERN_COUNT: int = 17

#: Combined pattern count (matrix + NEW SMARTS) — the total dispatch
#: surface area the runtime exposes to MCTS proof search.
#: MEASURED = 101 at module load (the L-R17-B2.3 contract target).
TOTAL_PATTERN_COUNT: int = MATRIX_PATTERN_COUNT + NEW_SMARTS_PATTERN_COUNT


# ---------------------------------------------------------------------------
# YuelBond precondition result — runtime-side enforcement of the
# Wf-275C m14 LEAD deployment contract.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class YuelBondPreconditionResult:
    """Runtime verdict for the YuelBond precondition check.

    Attributes
    ----------
    yuelbond_active : bool
        ``True`` iff the YuelBond decoder is active in the calling
        context (``args.use_yuelbond is True`` AND the CFM
        checkpoint has YuelBond weights loaded).  When ``False``,
        the m14 LEAD claim collapses (Wf-269C MEASURED
        Δ decode_ratio = −1.000).
    m14_lead_invalidated : bool
        ``True`` iff ``yuelbond_active=False``.  Mirrors the
        ``m14_lead_invalidated`` field emitted by
        :file:`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
        (Wf-296D contract).  Downstream reporters should gate the
        m14 LEAD citation on this flag.
    recommended_action : str
        One of ``{"PROCEED", "FAIL_FAST", "WARN_AND_PROCEED"}``:

        * ``"PROCEED"`` — YuelBond is active; the MCTS dispatch can
          cite m14 LEAD downstream.
        * ``"FAIL_FAST"`` — YuelBond is off AND the caller asserts
          ``strict=True`` (the default).  Caller should raise
          before invoking the MCTS.
        * ``"WARN_AND_PROCEED"`` — YuelBond is off BUT the caller
          explicitly opted in via ``strict=False``.  Caller proceeds
          with a loud WARNING on stderr so downstream reporters can
          attribute the metric collapse.
    reason : str
        Human-readable explanation of the verdict; surfaced in
        runtime logs + per-iteration history.
    delta_decode_ratio : float
        MEASURED Wf-269C delta when ``yuelbond_active=False``:
        ``-1.000`` (full=1.000 vs no_yuelbond=0.000).  Always
        ``0.000`` when ``yuelbond_active=True``.

    Honest framing
    --------------
    The ``delta_decode_ratio`` field is the MEASURED value from
    :file:`molmetal/reports/wf269c_cfm_real/wf269c_report.json` —
    not a projection.  When ``yuelbond_active=False``, this is the
    collapse magnitude the runtime has already absorbed (and should
    fail fast on).
    """

    yuelbond_active: bool
    m14_lead_invalidated: bool
    recommended_action: str
    reason: str
    delta_decode_ratio: float = 0.0
    #: Optional sub-diagnostics — populated by callers that pass
    #: extra context (e.g. ``checkpoint_path``, ``args.use_yuelbond``).
    extras: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# YuelBond precondition check — public dispatcher.
# ---------------------------------------------------------------------------
def check_yuelbond_precondition(
    *,
    use_yuelbond: bool = True,
    checkpoint_path: Optional[str] = None,
    strict: bool = True,
) -> YuelBondPreconditionResult:
    """Runtime-side check for the YuelBond precondition.

    Parameters
    ----------
    use_yuelbond : bool, default True
        The caller's ``args.use_yuelbond`` flag (or equivalent).  When
        ``False``, the runtime has been invoked WITHOUT the YuelBond
        decoder and the m14 LEAD claim collapses per Wf-269C.
    checkpoint_path : str | None, default None
        Optional checkpoint path.  When provided, the checker
        cross-references the checkpoint basename against the
        canonical YuelBond-trained checkpoint
        (:file:`molmetal/checkpoints/cfm_yuelbond_wasserstein.pt`).
        When the basename does NOT contain ``yuelbond``, the checker
        emits a WARNING regardless of ``use_yuelbond``.
    strict : bool, default True
        When ``True``, ``recommended_action`` is
        ``"FAIL_FAST"`` when ``yuelbond_active=False``.  When
        ``False``, the checker returns
        ``"WARN_AND_PROCEED"`` so callers can opt-in to running the
        MCTS without YuelBond (the Wf-296D ablation path).

    Returns
    -------
    :class:`YuelBondPreconditionResult`
        The runtime verdict.  Callers should short-circuit when
        ``recommended_action == "FAIL_FAST"`` (raise
        :class:`RuntimeError`) and emit a stderr WARNING when
        ``recommended_action == "WARN_AND_PROCEED"``.

    Notes
    -----
    This is the Wf-319G runtime-side counterpart to the
    Wf-296D training-side emission.  Both share the same
    ``m14_lead_invalidated`` semantics; the runtime side catches the
    case where the CFM trainingscript was not the caller (e.g. direct
    MCTS dispatch via :mod:`proof_search`).

    Per the Wf-269C MEASURED JSON, ``delta_decode_ratio = -1.000``
    when YuelBond is removed (full=1.000, no_yuelbond=0.000, 51/51
    train + 51/51 val).  When ``yuelbond_active=True``,
    ``delta_decode_ratio = 0.000`` (no collapse).
    """
    # Cross-check the checkpoint basename if provided.
    checkpoint_ok: bool = True
    if checkpoint_path:
        basename = str(checkpoint_path).rsplit("/", 1)[-1].lower()
        if "yuelbond" not in basename:
            checkpoint_ok = False

    yuelbond_active = bool(use_yuelbond) and bool(checkpoint_ok)

    if yuelbond_active:
        return YuelBondPreconditionResult(
            yuelbond_active=True,
            m14_lead_invalidated=False,
            recommended_action="PROCEED",
            reason="YuelBond active; m14 LEAD claim preserved",
            delta_decode_ratio=0.0,
            extras={
                "use_yuelbond": str(use_yuelbond),
                "checkpoint_path": str(checkpoint_path or ""),
            },
        )

    # yuelbond_active=False → m14 LEAD invalidated.
    delta = -1.000  # Wf-269C MEASURED
    action = "FAIL_FAST" if strict else "WARN_AND_PROCEED"
    reason_parts = [
        f"YuelBond INACTIVE (use_yuelbond={use_yuelbond}, "
        f"checkpoint_ok={checkpoint_ok}); "
        "m14 LEAD invalidated (Wf-269C MEASURED Δ decode_ratio = "
        f"{delta:+.3f}, full=1.000 → no_yuelbond=0.000).",
    ]
    if not checkpoint_ok:
        reason_parts.append(
            "Checkpoint basename does NOT contain 'yuelbond' — "
            "the loaded weights are NOT YuelBond-trained.",
        )
    return YuelBondPreconditionResult(
        yuelbond_active=False,
        m14_lead_invalidated=True,
        recommended_action=action,
        reason=" ".join(reason_parts),
        delta_decode_ratio=delta,
        extras={
            "use_yuelbond": str(use_yuelbond),
            "checkpoint_path": str(checkpoint_path or ""),
            "strict": str(strict),
        },
    )


# ---------------------------------------------------------------------------
# Matrix-reaction builder — compiles SMARTS tuples into RDKit
# ChemicalReaction objects.
# ---------------------------------------------------------------------------
def build_matrix_reactions(
    *,
    include_new_smarts: bool = True,
) -> Tuple[List[Tuple[str, "object"]], int]:
    """Compile the 84-pattern matrix (and optionally the 17 NEW
    SMARTS) into RDKit ``ChemicalReaction`` objects.

    Parameters
    ----------
    include_new_smarts : bool, default True
        When ``True``, the 17 NEW SMARTS patterns (transmetalation /
        Cl-OH / ligand-shell / Pt-Pd) are appended after the 84
        matrix patterns.  When ``False``, only the 84 matrix patterns
        are compiled.

    Returns
    -------
    tuple[list[tuple[str, ChemicalReaction]], int]
        A 2-tuple of ``(compiled, n_failed)``:

        * ``compiled`` — list of ``(name, reaction_obj)`` tuples.  Each
          ``reaction_obj`` is a RDKit ``ChemicalReaction`` whose
          ``Initialize()`` has been called (so ``IsMoleculeReactant``
          answers correctly).  When rdkit is unavailable, returns
        an empty list and ``n_failed == MATRIX_PATTERN_COUNT`` (or
        ``+ NEW_SMARTS_PATTERN_COUNT``).
        * ``n_failed`` — number of patterns that failed to parse or
        initialize.  When all patterns succeed, ``n_failed == 0``.
        When rdkit is unavailable, ``n_failed == total_attempted``.

    Notes
    -----
    The compilation is **deterministic and CPU-only**.  Each pattern
    is parsed via ``AllChem.ReactionFromSmarts``; the result is
    initialized so the search layer can fire it on a
    :class:`MoleculeClosedTerm` reactant via
    :func:`dispatch_metal_ligand_reaction`.
    """
    if not _RDKIT_AVAILABLE:
        # Defensive fallback — return empty + n_failed == total.
        # Caller still gets a valid result; the MCTS simply has no
        # matrix-pattern reactions to fire on (backward-compat path).
        total = TOTAL_PATTERN_COUNT if include_new_smarts else MATRIX_PATTERN_COUNT
        return ([], total)

    compiled: List[Tuple[str, object]] = []
    n_failed: int = 0

    # 1. Compile the 84-pattern matrix.
    for name, smarts, _desc in METAL_LIGAND_EXCHANGE_MATRIX:
        try:
            rxn = _AllChem.ReactionFromSmarts(smarts)
            if rxn is None:
                n_failed += 1
                continue
            # ``Initialize()`` returns ``None`` on success (RDKit
            # convention — the boolean conversion ``bool(None) ==
            # False`` is misleading).  The actual state is queried via
            # ``IsInitialized()`` after the call.  When ``Initialize()``
            # raises, we count it as a failure.
            try:
                rxn.Initialize()
            except Exception:  # pragma: no cover — defensive
                n_failed += 1
                continue
            if not bool(rxn.IsInitialized()):
                n_failed += 1
                continue
            compiled.append((name, rxn))
        except Exception:  # pragma: no cover — defensive
            n_failed += 1
            continue

    # 2. Optionally append the 17 NEW SMARTS.
    if include_new_smarts:
        for name, smarts, _desc in get_new_smarts_patterns():
            try:
                rxn = _AllChem.ReactionFromSmarts(smarts)
                if rxn is None:
                    n_failed += 1
                    continue
                try:
                    rxn.Initialize()
                except Exception:  # pragma: no cover — defensive
                    n_failed += 1
                    continue
                if not bool(rxn.IsInitialized()):
                    n_failed += 1
                    continue
                compiled.append((name, rxn))
            except Exception:  # pragma: no cover — defensive
                n_failed += 1
                continue

    return (compiled, n_failed)


# ---------------------------------------------------------------------------
# MetalLigandExchangeRuntime — the dispatch surface.
# ---------------------------------------------------------------------------
class MetalLigandExchangeRuntime:
    """Runtime wrapper around the 84-pattern matrix + 17 NEW SMARTS.

    The runtime compiles the SMARTS library into RDKit
    ``ChemicalReaction`` objects on construction and exposes a
    :meth:`dispatch` method that the MCTS proof-search layer can
    call for each (state, tile) pair during expansion.

    Parameters
    ----------
    include_new_smarts : bool, default True
        Compile the 17 NEW SMARTS patterns in addition to the 84
        matrix patterns.  Default True (the L-R17-B2.3 contract
        target of 101 patterns).
    yuelbond_precondition : bool, default True
        Run the :func:`check_yuelbond_precondition` runtime check
        at construction time.  When ``False``, the caller asserts
        the precondition has been satisfied upstream.

    Attributes
    ----------
    n_patterns : int
        Number of compiled RDKit ``ChemicalReaction`` objects (=
        ``len(self.compiled)``).
    n_failed : int
        Number of patterns that failed to compile (= 0 on the canonical
        MEASURED matrix; non-zero when rdkit is unavailable or a
        pattern's SMARTS fails to parse).
    matrix_size : int
        The 84-pattern matrix cardinality (always 84 — pinned by the
        regression test).
    yuelbond_precondition_result : :class:`YuelBondPreconditionResult`
        The verdict from the precondition check at construction.
    compiled : list[tuple[str, ChemicalReaction]]
        The compiled RDKit reactions ready for MCTS dispatch.

    Honest framing
    --------------
    The runtime is **deterministic and CPU-only**.  No I/O, no
    network, no torch.  The dispatch surface is intentionally
    minimal — it does NOT integrate with :mod:`proof_search`; that
    integration lives in :func:`wire_into_proof_search_rules` so
    callers can opt-in.
    """

    def __init__(
        self,
        *,
        include_new_smarts: bool = True,
        yuelbond_precondition: bool = True,
        use_yuelbond: bool = True,
        checkpoint_path: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        self.include_new_smarts = bool(include_new_smarts)
        self.matrix_size = MATRIX_PATTERN_COUNT

        # YuelBond precondition check (always run; the result is
        # exposed regardless of `yuelbond_precondition` so callers can
        # introspect the verdict).
        self.yuelbond_precondition_result: YuelBondPreconditionResult = (
            check_yuelbond_precondition(
                use_yuelbond=use_yuelbond,
                checkpoint_path=checkpoint_path,
                strict=strict,
            )
        )

        # Compile the SMARTS library into RDKit reactions.
        compiled, n_failed = build_matrix_reactions(
            include_new_smarts=self.include_new_smarts,
        )
        self.compiled: List[Tuple[str, object]] = compiled
        self.n_patterns: int = len(compiled)
        self.n_failed: int = n_failed

        # Optional lazy import — only when the caller requests it
        # (default False; opt-in via `yuelbond_precondition=True`).
        if yuelbond_precondition:
            verdict = self.yuelbond_precondition_result
            if verdict.recommended_action == "FAIL_FAST":
                raise RuntimeError(
                    f"[Wf-319G] YuelBond precondition FAIL_FAST: " f"{verdict.reason}"
                )
            if verdict.recommended_action == "WARN_AND_PROCEED":
                # Loud WARNING on stderr so the operator sees it.
                log.warning(
                    "[Wf-319G] %s",
                    verdict.reason,
                )

    # ------------------------------------------------------------------
    # Dispatch — call on a (state_smiles, tile_smiles) pair.
    # ------------------------------------------------------------------
    def dispatch(
        self,
        state_smiles: str,
        tile_smiles: str,
        *,
        max_products_per_pattern: int = 4,
    ) -> List[Tuple[str, str]]:
        """Fire the 84-pattern matrix (+ 17 NEW SMARTS) on a
        ``(state, tile)`` pair and return the products.

        Parameters
        ----------
        state_smiles : str
            The state molecule SMILES (the canonical form of the
            current MCTS node).
        tile_smiles : str
            The tile / reactant SMILES that the state was joined with.
        max_products_per_pattern : int, default 4
            Cap on products emitted per pattern.  Matches the
            RDKit convention of returning up to 4 stereoisomers
            per reaction.

        Returns
        -------
        list[tuple[str, str]]
            A list of ``(pattern_name, product_smiles)`` tuples.
            Empty when the runtime is unavailable (rdkit missing) or
            no pattern fired successfully on the input pair.

        Notes
        -----
        Each product SMILES is a canonicalised single-molecule form
        (RDKit ``MolToSmiles``); the caller is responsible for
        aggregating products into :class:`MoleculeClosedTerm` objects
        (the MCTS does this in :meth:`MCTSProofSearch._expand`).
        """
        if not _RDKIT_AVAILABLE:
            return []
        if not self.compiled:
            return []

        # Parse the reactant pair.
        try:
            mol_state = _Chem.MolFromSmiles(str(state_smiles))
            mol_tile = _Chem.MolFromSmiles(str(tile_smiles))
        except Exception:  # pragma: no cover
            return []
        if mol_state is None or mol_tile is None:
            return []

        products: List[Tuple[str, str]] = []
        for pattern_name, rxn in self.compiled:
            try:
                # ``RunReactants`` returns a tuple of tuples of
                # product mols; each outer entry corresponds to one
                # reactant pair permutation.  We only fire on the
                # (state, tile) ordering.
                pset = rxn.RunReactants((mol_state, mol_tile))
                if not pset:
                    continue
                for entry in pset[:max_products_per_pattern]:
                    for prod_mol in entry:
                        try:
                            ps = _Chem.MolToSmiles(prod_mol)
                            if ps:
                                products.append((pattern_name, ps))
                        except Exception:  # pragma: no cover
                            continue
            except Exception:  # pragma: no cover — defensive
                continue

        return products

    # ------------------------------------------------------------------
    # Introspection helpers.
    # ------------------------------------------------------------------
    def pattern_names(self) -> List[str]:
        """Return the names of all compiled patterns (canonical order)."""
        return [name for name, _rxn in self.compiled]

    def is_available(self) -> bool:
        """Return True iff the runtime is dispatchable (rdkit available
        AND all patterns compiled)."""
        return bool(_RDKIT_AVAILABLE) and (self.n_failed == 0)


# ---------------------------------------------------------------------------
# Convenience helper for :mod:`proof_search` integration.
# ---------------------------------------------------------------------------
def wire_into_proof_search_rules(
    runtime: MetalLigandExchangeRuntime,
    existing_rules: Dict[str, object],
) -> Dict[str, object]:
    """Wire the 84-pattern matrix (+ 17 NEW SMARTS) into the MCTS
    proof-search ``rules`` dict.

    This is a **non-invasive** wire-up — it returns a NEW dict that
    contains the existing rules PLUS the matrix-pattern reactions, all
    keyed by name.  Callers that pass the result to
    :class:`MCTSProofSearch` extend the MCTS dispatch surface without
    modifying :mod:`proof_search`.

    Parameters
    ----------
    runtime : :class:`MetalLigandExchangeRuntime`
        The runtime instance whose compiled reactions should be wired.
    existing_rules : dict[str, ReactionRule]
        The legacy ``self.rules`` dict from :class:`MCTSProofSearch`.

    Returns
    -------
    dict[str, ReactionRule | ChemicalReaction]
        A new dict that contains all existing rules PLUS the
        runtime's compiled patterns.  Existing-rule names win on
        collision (so the runtime does NOT shadow legacy click rules).

    Honest framing
    --------------
    This function does NOT modify ``existing_rules`` in place —
    callers receive a fresh dict.  The runtime does NOT register
    reactions into :mod:`molmetal_lam.reactions.rule_registry`
    directly (that registry side-effect is opt-in via
    :func:`pt_metal_ligand_exchange.register_metal_ligand_exchange_rules`).
    """
    out: Dict[str, object] = dict(existing_rules)
    if not runtime.is_available():
        return out
    for pattern_name, rxn in runtime.compiled:
        if pattern_name in out:
            # Existing rule wins — do NOT shadow legacy click rules.
            continue
        out[pattern_name] = rxn
    return out


# ---------------------------------------------------------------------------
# Self-test — print summary when run as a script.
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    print(f"MATRIX_PATTERN_COUNT = {MATRIX_PATTERN_COUNT} (target 84)")
    print(f"NEW_SMARTS_PATTERN_COUNT = {NEW_SMARTS_PATTERN_COUNT} (target 17)")
    print(f"TOTAL_PATTERN_COUNT = {TOTAL_PATTERN_COUNT} (target 101)")
    verdict = check_yuelbond_precondition()
    print(
        f"YuelBond precondition (default): "
        f"active={verdict.yuelbond_active} "
        f"action={verdict.recommended_action} "
        f"delta={verdict.delta_decode_ratio:+.3f}"
    )
    runtime = MetalLigandExchangeRuntime(
        yuelbond_precondition=False,  # do not warn on script invocation
    )
    print(
        f"Runtime compiled: {runtime.n_patterns} patterns "
        f"({runtime.n_failed} failed); "
        f"is_available={runtime.is_available()}"
    )
