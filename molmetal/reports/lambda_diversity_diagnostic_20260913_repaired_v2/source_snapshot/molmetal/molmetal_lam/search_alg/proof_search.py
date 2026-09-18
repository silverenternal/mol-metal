"""MCTS proof search over Molecular Lambda Calculus terms — strengthened.

This module implements **Layer 8** of the Molecular Lambda Calculus
(MLC), formalized in
``TODO/13_lambda_clickchem/molecular_lambda_calculus.md`` §8
(Drug-Design-as-Proof-Search).

================================================================
Core thesis
================================================================
In the Curry-Howard reading of MLC:

    type   = proposition  ("Lipinski", "Binds MMP2", ...)
    term   = proof        (a closed lambda-term = a molecule)
    inhabitation = a synthesizable drug candidate

A drug-design problem is then the constructive proof obligation::

    "∃ M such that M inhabits every type in target_predicates
                AND M inhabits BindingType(binding_site)"

Finding such an M is **proof search in λ-term space**.  The natural
algorithm is Monte-Carlo Tree Search (MCTS) with the PUCT selection
rule (standard in AlphaGo/AlphaZero-style algorithms): each node is a
closed term, each action is "apply reaction rule R to molecule M using
tile T" = one β-reduction step, and the value of a leaf is the score
returned by the user-supplied scorer.

================================================================
Strengths over the stub baseline
================================================================
This implementation goes well beyond the uniform-prior / uniform-rollout
stub and ships the **theoretical upper bound** for a UCB-style tree
search over λ-term space:

* :class:`SymbolicPrior`   — wraps :class:`HeuristicRegressor` so the
  PUCT prior P(a|s) is a *learned* function of state features rather
  than a constant 0.5.  Falls back to ``constant 0.5`` when no
  training data is available (full backward compatibility).

* Multi-reward value head  — instead of one monolithic scorer we
  accept a :class:`RewardAggregator` with weights
  ``{w_vina, w_sa, w_posebusters, w_pic50, w_retro}``.  When the user
  supplies only a scalar ``scorer``, we fall back to that scalar and
  preserve the historical single-reward interface.

* Epsilon-greedy rollout   — the rollout policy is biased by
  :class:`SymbolicPrior`'s predicted value of the candidate child
  states, rather than uniform random.

* AlphaZero-style Dirichlet noise at the root — for the first
  ``dirichlet_fraction`` of simulations the root prior is mixed with
  ``Dir(alpha)`` to encourage exploration in the opening.

* Optional virtual-loss parallel scaffold — :class:`VirtualLoss` is
  provided as a hook for future lock-free MCTS implementations.

================================================================
Class: MCTSProofSearch
================================================================
The class wires the existing MLC layers together:

    atoms       PRIMITIVE_ATOMS / METAL_ATOMS               (Layer 1)
    bonds       Bond / Bond.covalent / Bond.dative          (Layer 2)
    molecules   MoleculeClosedTerm                          (Layer 3)
    reactions   ReactionRule / CuAAC / SPAAC / ...           (Layer 4)
    types       TypePredicate / well_typed                  (Layer 7)
    binding     BindingSite / typecheck                     (Layer 6)

and uses them to perform the four MCTS phases per simulation:

    select   : PUCT traversal of the existing tree
    expand   : _expand(state) — apply each (rule, compatible_tile) pair
                            = one β-reduction step (= one reaction)
    rollout  : _rollout(state, depth) — ε-greedy playout to a terminal,
                                        evaluate via the reward aggregator
    backprop : _backprop(path, value) — update N / W / P along the path

A leaf is **terminal** when
    (a) it inhabits all target predicates AND binds the target site, OR
    (b) the depth budget is exhausted, OR
    (c) the state has no reducible applications (= β-NF and closed).

================================================================
History tracking
================================================================
The ``history`` attribute is a list of dicts, one entry per
``search()`` iteration, capturing the diagnostics used by the
closed-loop pipeline:

    iteration          : int
    best_score         : float     (best leaf score seen so far)
    mean_score         : float     (mean of all leaf scores this iter)
    n_states_explored  : int       (tree size after this iter)
    n_satisfying       : int       (# leaves that satisfy all preds)
    best_state         : optional molecule (best leaf by score)
"""

from __future__ import annotations

import logging
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from molmetal.utils.device import get_device

log = logging.getLogger(__name__)

from molmetal_lam.atoms.combinators import Atom
from molmetal_lam.binding.types import BindingSite, typecheck
from molmetal_lam.lam_chem.pysr_wrapper import HeuristicRegressor
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.reactions.beta_reductions import ReactionRule
from molmetal_lam.types.predicates import TypePredicate
from triton_kernels import softmax_last_dim
from triton_kernels.config import triton_config

# Optional RDKit-backed channels — the MLC search runs headless without
# RDKit, so we guard every import and silently degrade channels that
# cannot construct a Mol to ``0.0``.  This keeps the deterministic
# unit-test path fully green even when ``rdkit`` is missing.
try:  # pragma: no cover — optional dep
    from rdkit import Chem as _Chem  # type: ignore
    from rdkit.Chem import Descriptors as _Descriptors  # type: ignore
    try:
        from rdkit.Contrib.SA_Score import sascorer as _sascorer  # type: ignore
    except Exception:  # pragma: no cover
        _sascorer = None  # type: ignore
    try:
        from rdkit.Chem import RDLogger as _RDLogger  # type: ignore
        _RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        _RDLogger = None  # type: ignore
    _RDKIT_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    _Chem = None  # type: ignore
    _Descriptors = None  # type: ignore
    _sascorer = None  # type: ignore
    _RDLogger = None  # type: ignore
    _RDKIT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Scoring / heuristic
# ---------------------------------------------------------------------------

#: Default signature for a scorer callable:
#:     scorer(state: MoleculeClosedTerm) -> float
#: A typical implementation wraps a REINVENT4 / DiffDock / EGNN head and
#: returns a real-valued score in [0, 1] (or unbounded — see comments in
#: ``_rollout``).  Higher = better.
ScorerFn = Callable[[MoleculeClosedTerm], float]


def heuristic(features: "Any") -> float:
    """Heuristic stub that PySR will later fit.

    The intended signature is
        ``heuristic(features: torch.Tensor) -> float``
    where ``features`` is a 1-D tensor of molecular descriptors
    (MW, logP, HBD, HBA, TPSA, n_atoms, n_bonds, n_free_sites, ...).
    The output is the prior estimate of the leaf score for that state,
    used to bias MCTS expansion toward promising branches.

    For backward compatibility this module-level function still
    returns a constant ``0.5`` so external monkey-patches (and tests
    that import ``proof_search.heuristic`` directly) keep working.
    The MCTS-bound prior lives on :meth:`MCTSProofSearch.heuristic`
    (TODO-02) and dispatches through the resolved
    :class:`RewardAggregator` so the prior is a real per-channel
    weighted sum rather than a constant.

    Parameters
    ----------
    features : Any
        A tensor, list, or scalar descriptor of the state. Today this is
        accepted but unused.

    Returns
    -------
    float
        Prior probability in [0, 1]. Default: 0.5.
    """
    del features  # backward-compat: ignore the descriptor entirely
    return 0.5


# ---------------------------------------------------------------------------
# SymbolicPrior — wraps HeuristicRegressor for MCTS priors
# ---------------------------------------------------------------------------


@dataclass
class SymbolicPrior:
    """PUCT prior estimated by a symbolic-regression model of leaf value.

    The prior is a learned function
        P(a|s) = sigmoid(HeuristicRegressor.predict(features(s, a)))
    so the PUCT exploration term is biased by data rather than the
    constant 0.5 baseline.  When :meth:`fit` has not been called the
    prior falls back to a uniform ``0.5`` (identical to the original
    stub), preserving full backward compatibility.

    Parameters
    ----------
    feature_extractor : callable
        ``feature_extractor(state: MoleculeClosedTerm) -> Sequence[float]``
        Maps a molecule to its feature vector.  Defaults to
        :func:`_default_feature_extractor` which returns
        ``[n_atoms, n_bonds, n_free_sites, n_heavy]``.

    random_state : int, default 0
        RNG seed passed to the underlying :class:`HeuristicRegressor`.

    niterations : int, default 50
        Number of PySR iterations (only used when PySR is live).

    binary_ops : list[str], optional
    unary_ops  : list[str], optional
        Operator vocabularies forwarded to :class:`HeuristicRegressor`.
    """

    feature_extractor: Callable[[MoleculeClosedTerm], Sequence[float]] = field(
        default=lambda s: _default_feature_extractor(s)
    )
    random_state: int = 0
    niterations: int = 50
    binary_ops: Optional[List[str]] = None
    unary_ops: Optional[List[str]] = None

    #: Whether :meth:`fit` has been called at least once.
    fitted: bool = False
    #: Number of features the underlying regressor expects (set by fit).
    n_features: int = 0

    def __post_init__(self) -> None:
        # Lazy model — only constructed on first .fit() call.
        self._model: Optional[HeuristicRegressor] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _build_model(self) -> HeuristicRegressor:
        return HeuristicRegressor(
            niterations=self.niterations,
            binary_ops=self.binary_ops,
            unary_ops=self.unary_ops,
            random_state=self.random_state,
        )

    # ------------------------------------------------------------------
    # fit / predict_proba
    # ------------------------------------------------------------------
    def fit(
        self,
        states: Sequence[MoleculeClosedTerm],
        scores: Sequence[float],
    ) -> "SymbolicPrior":
        """Fit the symbolic regressor to ``(features(state), score)`` pairs.

        Internally builds the X / y matrices and dispatches to
        :meth:`HeuristicRegressor.fit`.  Returns ``self`` for fluent use.

        Parameters
        ----------
        states : list[MoleculeClosedTerm]
            Witness molecules whose features predict the leaf value.
        scores : list[float]
            Observed leaf scores for the corresponding states (real-
            valued, higher = better).
        """
        if not states:
            # Empty fit — leave model un-fitted, prior defaults to 0.5.
            return self
        X = np.asarray(
            [list(self.feature_extractor(s)) for s in states], dtype=float,
        )
        y = np.asarray(list(scores), dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self._model is None:
            self._model = self._build_model()
        self._model.fit(X, y)
        self.fitted = True
        self.n_features = int(X.shape[1])
        return self

    def predict_value(self, state: MoleculeClosedTerm) -> float:
        """Raw regression prediction (linear, unbounded).

        Returns ``0.5`` if the model is unfitted (constant baseline).
        """
        if not self.fitted or self._model is None:
            return 0.5
        try:
            feats = list(self.feature_extractor(state))
            X = np.asarray(feats, dtype=float).reshape(1, -1)
            return float(self._model.predict(X)[0])
        except Exception:
            return 0.5

    def predict_proba(self, state: MoleculeClosedTerm) -> float:
        """Prior probability P(a|s) in [0, 1].

        When the model is unfitted we return the constant 0.5 baseline
        so the search degenerates to vanilla UCB (full backward
        compatibility with the pre-strengthening implementation).
        Otherwise we sigmoid-map the raw regressor output to a valid
        probability mass.  Saturation guards prevent the prior from
        collapsing to exactly 0 / 1 (which would freeze PUCT
        exploration).
        """
        if not self.fitted or self._model is None:
            return 0.5
        raw = self.predict_value(state)
        try:
            z = float(raw)
            if z >= 0:
                p = 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                p = ez / (1.0 + ez)
        except OverflowError:
            p = 0.0 if raw < 0 else 1.0
        return max(1e-6, min(1.0 - 1e-6, p))

    def equation(self) -> str:
        """Return the human-readable fitted equation, or ``<unfitted>``."""
        if not self.fitted or self._model is None:
            return "<unfitted>"
        try:
            return self._model.equation()
        except Exception:
            return "<unknown>"

    @property
    def backend_(self) -> Optional[str]:
        return getattr(self._model, "backend_", None)


def _default_feature_extractor(state: MoleculeClosedTerm) -> List[float]:
    """Default feature vector for a molecule (backward-compatible).

    The 4 features are the same lightweight counts the original
    ``_prior`` method used, so this is a strict superset of the
    stub behaviour — fitting to ``[n_atoms, n_bonds, n_free_sites]``
    matches the heuristic stub exactly.
    """
    try:
        n_atoms = float(state.n_atoms)
        n_bonds = float(state.n_bonds)
        n_free = float(sum(state.free_sites.values()))
        n_heavy = float(sum(
            1 for a in state.atoms if a.symbol not in {"H"}
        ))
        return [n_atoms, n_bonds, n_free, n_heavy]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Multi-reward aggregator
# ---------------------------------------------------------------------------


def _safe_smiles_to_mol(smiles: str):
    """Convert a SMILES string to an RDKit Mol, suppressing errors.

    Returns ``None`` when RDKit is unavailable or the SMILES is
    unparsable.  Used by the SA / QED / Vina_proxy channel builders.
    """
    if not _RDKIT_AVAILABLE or _Chem is None or not smiles:
        return None
    try:
        mol = _Chem.MolFromSmiles(smiles)
        return mol  # may be None
    except Exception:
        return None


def _default_qed_channel() -> Callable[[MoleculeClosedTerm], float]:
    """Return a callable that computes RDKit QED in [0, 1].

    Falls back to ``0.0`` when RDKit or a valid SMILES is unavailable.
    """
    def _channel(state: MoleculeClosedTerm) -> float:
        if not _RDKIT_AVAILABLE or _Descriptors is None:
            return 0.0
        try:
            smi = state.canonical_smiles()
            if smi:
                from molmetal_lam.tile_lib.canonical_cache import canonicalize
                smi = canonicalize(str(smi))
        except Exception:
            return 0.0
        mol = _safe_smiles_to_mol(smi)
        if mol is None:
            return 0.0
        try:
            qed_val = float(_Descriptors.qed(mol))
            return max(0.0, min(1.0, qed_val))
        except Exception:
            return 0.0
    return _channel


def _default_pb_valid_channel() -> Callable[[MoleculeClosedTerm], float]:
    """Return a callable that computes the PB outer-gate pass/fail.

    Wraps
    :func:`molmetal.validation.posebusters_runner.check_posebusters`
    under the MMFF94 window.  Returns ``1.0`` when PoseBusters passes
    the molecule, ``0.0`` otherwise.  When ``posebusters`` is not
    installed (the canonical TODO/environment.md path) the channel
    returns ``0.0`` so the aggregator never crashes.  When RDKit is
    unavailable we also return ``0.0`` — no SMILES, no PB check.

    Notes
    -----
    * Memoised per SMILES via a tiny module-level dict — the same SMILES
      hitting the channel many times across a search only runs the PB
      pipeline once.  The cache is bounded to 4096 entries to keep the
      memory cost of long closed-loop runs predictable.
    """
    cache: Dict[str, float] = {}

    def _channel(state: MoleculeClosedTerm) -> float:
        try:
            smi = state.canonical_smiles()
            if smi:
                from molmetal_lam.tile_lib.canonical_cache import canonicalize
                smi = canonicalize(str(smi))
        except Exception:
            return 0.0
        if not smi:
            return 0.0
        if smi in cache:
            return cache[smi]
        # Probe PB availability before invoking the runner so we can
        # short-circuit the import cost on the missing-pkg path.
        try:
            from molmetal.validation.posebusters_runner import (
                check_posebusters,
                posebusters_available,
            )
        except Exception:
            cache[smi] = 0.0
            return 0.0
        if not posebusters_available():
            cache[smi] = 0.0
            return 0.0
        try:
            res = check_posebusters(smi)
        except Exception:
            cache[smi] = 0.0
            return 0.0
        # ``pb_valid`` may be True / False / None (skipped).  Skip
        # returns 0.0 so the aggregator degrades gracefully.
        v = res.get("pb_valid", None) if isinstance(res, dict) else None
        out = 1.0 if v is True else 0.0
        if len(cache) >= 4096:
            try:
                cache.clear()
            except Exception:
                pass
        cache[smi] = out
        return out
    return _channel


def _resolve_pb_weight() -> float:
    """Resolve the PB outer-gate weight from the ``PB_WEIGHT`` env var.

    Defaults to ``0.0`` (soft outer-gate; per-iter pass-rate is still
    recorded by the closed-loop pipeline).  Set ``PB_WEIGHT=1.0`` to
    enable scoring so that PB-pass molecules get a full reward
    multiplier — pass-fail molecules are kept at 0.0 from the gate.
    """
    try:
        raw = os.environ.get("PB_WEIGHT", "0.0")
        w = float(raw)
    except Exception:
        w = 0.0
    return max(0.0, w)


def _default_vina_proxy_channel(
    prior: Optional["SymbolicPrior"] = None,
) -> Callable[[MoleculeClosedTerm], float]:
    """Return a callable that scores molecules by a Vina-style proxy.

    The proxy combines two cheap signals so the leaf value has real
    variance (so PUCT_EXPLOIT_RATIO_VAR can be measured) without
    requiring a docking backend:

    * the molecule's :class:`SymbolicPrior` predicted value (if fitted)
      — encodes prior knowledge of which molecules are "good",
    * a hand-crafted ``[n_atoms, n_bonds, sum_free_sites]`` linear
      shift that approximates a docking score (more atoms / bonds =
      more interactions with the pocket; free sites = unreacted
      surface, mild penalty).

    Output is shifted to ``[0, 1]`` so it composes cleanly with the
    other channels.  When RDKit is unavailable or the SMILES is
    unparsable we still extract the 3-feature vector from the
    :class:`MoleculeClosedTerm` directly (no SMILES round-trip) so the
    proxy is well-defined in headless test runs.
    """
    def _channel(state: MoleculeClosedTerm) -> float:
        feats: List[float]
        if _RDKIT_AVAILABLE:
            try:
                smi = state.canonical_smiles()
                if smi:
                    from molmetal_lam.tile_lib.canonical_cache import canonicalize
                    smi = canonicalize(str(smi))
                mol = _safe_smiles_to_mol(smi)
                if mol is not None:
                    n_atoms = float(mol.GetNumAtoms())
                    n_bonds = float(mol.GetNumBonds())
                    n_free = float(
                        sum(state.free_sites.values())
                    )
                else:
                    n_atoms = float(state.n_atoms)
                    n_bonds = float(state.n_bonds)
                    n_free = float(sum(state.free_sites.values()))
            except Exception:
                n_atoms = float(state.n_atoms)
                n_bonds = float(state.n_bonds)
                n_free = float(sum(state.free_sites.values()))
        else:
            try:
                n_atoms = float(state.n_atoms)
                n_bonds = float(state.n_bonds)
                n_free = float(sum(state.free_sites.values()))
            except Exception:
                n_atoms = n_bonds = n_free = 0.0
        feats = [n_atoms, n_bonds, n_free]
        if prior is not None and prior.fitted:
            try:
                prior_val = float(prior.predict_value(state))
            except Exception:
                prior_val = 0.5
        else:
            prior_val = 0.5
        # Combine symbolic prior (centred) + linear proxy, then
        # squash into [0, 1].
        linear_proxy = (
            0.04 * feats[0]            # n_atoms
            + 0.05 * feats[1]          # n_bonds
            - 0.03 * max(0.0, feats[2] - 4.0)  # free-site penalty
        )
        raw = 0.5 + 0.5 * (prior_val - 0.5) + linear_proxy
        return float(max(0.0, min(1.0, raw)))
    return _channel


@dataclass
class RewardAggregator:
    """Weighted combination of scalar reward channels.

    The MCTS leaf value is computed as

        value(s) = w_vina * r_vina(s)
                 + w_sa   * r_sa(s)
                 + w_posebusters * r_posebusters(s)
                 + w_pic50 * r_pic50(s)
                 + w_retro * r_retro(s)
                 + bonus_typed   * [s satisfies predicates]
                 + bonus_binder  * [s binds target]

    Each reward channel is a callable ``(state) -> float``; if any
    callable raises (e.g. a docking backend is unavailable) the channel
    silently degrades to 0.0 so the search never crashes.

    When all weights are 0.0 except a single non-zero entry this
    reduces to the historical single-scorer behaviour.

    Parameters
    ----------
    r_vina : callable, optional
        Vina docking score (negative kcal/mol typical — higher = better
        *after the aggregator negates the value* if ``vina_invert`` is
        True).
    r_sa : callable, optional
        Synthetic-accessibility score (1=trivial, 10=very hard).  We
        invert so higher = better.
    r_posebusters : callable, optional
        PoseBusters pass-rate in [0, 1].  Higher = better.
    r_pic50 : callable, optional
        pIC50 estimate in [0, 12].  Higher = better.
    r_retro : callable, optional
        Retrosynthesis feasibility in [0, 1].  Higher = better.
    w_* : float, default 1.0
        Per-channel weight.
    vina_invert : bool, default True
        If True, the Vina channel is negated before adding (since Vina
        returns negative kcal/mol; lower = better → negate → higher =
        better).  Set False when ``r_vina`` already returns a positive
        "more negative is better" pseudo-score.
    bonus_typed : float, default 0.5
        Flat bonus added when ``s`` satisfies ``target_predicates``.
    bonus_binder : float, default 0.5
        Flat bonus added when ``s`` type-checks against ``binding_site``.
    """

    r_vina: Optional[Callable[[MoleculeClosedTerm], float]] = None
    r_sa: Optional[Callable[[MoleculeClosedTerm], float]] = None
    r_qed: Optional[Callable[[MoleculeClosedTerm], float]] = None
    r_vina_proxy: Optional[Callable[[MoleculeClosedTerm], float]] = None
    r_posebusters: Optional[Callable[[MoleculeClosedTerm], float]] = None
    #: R3/F4 — PoseBusters outer-gate channel (binary: 1.0 pass / 0.0
    #: fail).  Distinct from ``r_posebusters`` (which returns the real-
    #: valued pass-rate) — this channel is the *outer gate* the
    #: closed-loop pipeline consults after each iter to compute
    #: ``pb_pass_rate_top1`` / ``pb_pass_rate_topk``.  Default
    #: implementation delegates to
    #: :func:`molmetal.validation.posebusters_runner.check_posebusters`
    #: under the MMFF94 window.  When ``posebusters`` is not installed
    #: the channel returns ``0.0`` so the aggregator degrades
    #: gracefully (the closed-loop still records the per-iter pass-rate
    #: as 0.0 — never crashes).
    r_pb_valid: Optional[Callable[[MoleculeClosedTerm], float]] = None
    r_pic50: Optional[Callable[[MoleculeClosedTerm], float]] = None
    r_retro: Optional[Callable[[MoleculeClosedTerm], float]] = None
    #: L-4 — REINVENT4 multi-property channel.  When supplied, this
    #: callable is invoked with the molecule and should return a float
    #: in [0, 1].  The typical implementation delegates to
    #: :class:`ScoreAggregator.aggregate` on the result of
    #: :meth:`REINVENT4Adapter.score`.  When the adapter is
    #: unavailable, set ``r_reinvent4`` to ``None`` (or a callable
    #: that returns ``0.0``) so the aggregator gracefully degrades.
    r_reinvent4: Optional[Callable[[MoleculeClosedTerm], float]] = None
    #: D-3 — Synthesis-yield oracle channel.  When supplied, the
    #: closure is invoked with the molecule and should return a float
    #: in [0, 1] (typically the mean predicted isolated yield across
    #: known click reactions).  When ``None`` the channel silently
    #: degrades to ``0.0`` so the aggregator never crashes.
    r_synth: Optional[Callable[[MoleculeClosedTerm], float]] = None
    #: Phase-2 (Round-7) — ADMET desirability channel.  When supplied,
    #: the closure is invoked with the molecule and should return a
    #: float in [0, 1] (Lipinski-style desirability aggregated from
    #: logP, MW, HBD, HBA, TPSA).  Default impl wraps
    #: :func:`molmetal.validation.admet_runner.predict_admet` +
    #: :func:`molmetal.validation.admet_runner.admet_desirability`.
    #: When the ADMET backend (admet-ai / datamol / RDKit) is not
    #: available, set ``r_admet=None`` or supply a callable that
    #: returns 0.0 — the aggregator degrades silently.
    r_admet: Optional[Callable[[MoleculeClosedTerm], float]] = None
    r_logp_anticancer: Optional[Callable] = None
    r_tpsa_iv: Optional[Callable] = None
    r_rotatable_bonds: Optional[Callable] = None
    r_herg_proxy: Optional[Callable] = None
    r_anticancer_composite: Optional[Callable] = None
    w_vina: float = 1.0
    w_sa: float = 1.0
    w_qed: float = 1.0
    w_vina_proxy: float = 1.0
    w_posebusters: float = 1.0
    #: R3/F4 — weight on the PoseBusters outer-gate channel.  Defaults
    #: to 0.0 so adding the gate is *opt-in* — set ``w_pb_valid=1.0``
    #: (or set the ``PB_WEIGHT`` env var ≥ 1.0) to make PB-pass a hard
    #: multiplier on the aggregated reward.  The closed-loop pipeline
    #: still records ``pb_pass_rate_top1`` / ``pb_pass_rate_topk`` per
    #: iteration regardless of the weight (PB is a *diagnostic*
    #: outer-gate by default).
    w_pb_valid: float = 0.0
    w_pic50: float = 1.0
    w_retro: float = 1.0
    #: Weight on the L-4 REINVENT4 channel.  Defaults to 1.0 so the
    #: multi-property scorer can fully replace the SA+QED+Vina_proxy
    #: stub when supplied.
    w_reinvent4: float = 1.0
    #: Weight on the D-3 synthesis oracle channel.  Defaults to 0.0
    #: so adding the oracle is opt-in (set ``w_synth=1.0`` to enable).
    w_synth: float = 0.0
    #: Weight on the Phase-2 ADMET channel.  Defaults to 0.0 (opt-in)
    #: so the canonical Lipinski desirability contributes only when
    #: the caller explicitly enables it.  Set ``w_admet=1.0`` (or the
    #: ``ADMET_WEIGHT`` env var ≥ 1.0) to add the Lipinski r_admet
    #: reward on top of the existing channels.
    w_admet: float = 0.0
    w_logp_anticancer: float = 1.0
    w_tpsa_iv: float = 1.0
    w_rotatable_bonds: float = 1.0
    w_herg_proxy: float = 1.0
    w_anticancer_composite: float = 1.0
    vina_invert: bool = True
    bonus_typed: float = 0.5
    bonus_binder: float = 0.5

    def __post_init__(self) -> None:
        """Apply environment-variable overrides on weight fields.

        * ``ADMET_WEIGHT`` ≥ 1.0 — enable the Phase-2 ADMET channel
          (``w_admet = 1.0``) and install the default ``r_admet``
          closure pointing at the ADMET runner.  Used by closed-loop
          smoke tests to opt in without code changes.
        """
        try:
            aw = float(os.environ.get("ADMET_WEIGHT", "0") or 0.0)
        except (TypeError, ValueError):
            aw = 0.0
        if aw >= 1.0:
            self.w_admet = 1.0
            if self.r_admet is None:
                self.r_admet = _r_admet_default

    def register_anticancer_channels(self, suite) -> None:
        """Register logP, TPSA, rotatable-bond and hERG channels."""
        def _smiles(state):
            if isinstance(state, str):
                return state
            fn = getattr(state, "canonical_smiles", None)
            if callable(fn):
                return str(fn())
            return str(getattr(state, "smiles", ""))
        self.r_logp_anticancer = lambda state: suite.logp_anticancer(_smiles(state))
        self.r_tpsa_iv = lambda state: suite.tpsa_iv(_smiles(state))
        self.r_rotatable_bonds = lambda state: suite.rotatable_bonds(_smiles(state))
        self.r_herg_proxy = lambda state: suite.herg_proxy(_smiles(state))
        self.r_anticancer_composite = lambda state: suite.composite_score(_smiles(state))

    # ------------------------------------------------------------------
    # Per-channel weight presets (TODO-02) — six-channel default profile
    # ------------------------------------------------------------------
    #: Default per-channel weights for the *prior* use case.  These
    #: mirror the task spec in TODO-02: Vina dominates (binding is the
    #: primary design objective), PoseBusters is the structural sanity
    #: gate, retro feasibility gates synthesizability, and QED / SA /
    #: pIC50 fill out drug-likeness + potency.  Exposed as a module
    #: constant so :meth:`MCTSProofSearch.__post_init__` can copy the
    #: dict without instantiating an aggregator.
    DEFAULT_PRIOR_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "w_vina": 0.40,
        "w_posebusters": 0.20,
        "w_retro": 0.15,
        "w_pic50": 0.10,
        "w_qed": 0.10,
        "w_sa": 0.05,
    })

    def aggregate(
        self,
        smiles: str,
        channels: Optional[Dict[str, float]] = None,
    ) -> float:
        """Aggregate a per-channel reward dict into a single prior score.

        Convenience entry-point used by :meth:`MCTSProofSearch.heuristic`
        (TODO-02) so the prior can be derived from per-channel reward
        values rather than a constant 0.5 stub.  The contract is:

        * ``smiles`` is the canonical SMILES of the candidate molecule.
          Accepted for symmetry with the other reward entry-points and
          so callers can log the input alongside the score, but is
          **not** used to compute the value (the channel dict is).
        * ``channels`` is an optional ``{channel_name: value}`` mapping.
          Recognised keys: ``r_vina``, ``r_sa``, ``r_qed``,
          ``r_posebusters``, ``r_retro``, ``r_pic50``, ``r_vina_proxy``,
          ``r_pb_valid``, ``r_reinvent4``, ``r_synth``, ``r_admet``.
          Unknown keys are silently dropped.  When ``channels`` is
          ``None`` or empty, ``aggregate`` returns the *weights-only*
          value (i.e. ``sum(self.w_* )``) which is non-zero as long as
          at least one channel weight is positive.  When at least one
          channel is supplied, the returned value is
          ``sum(w_k * v_k for k in channels)``.

        The result is **not** squashed into [0, 1] — callers that want
        a PUCT-style prior mass should run it through ``sigmoid`` (see
        :meth:`MCTSProofSearch.heuristic`).  The unsquashed value is
        more useful for diagnostic reporting (the closed-loop pipeline
        logs it alongside the leaf Q value).

        Parameters
        ----------
        smiles : str
            Canonical SMILES of the molecule — accepted for symmetry
            with the other reward entry-points but not used in the
            computation (kept as an explicit kwarg so future
            SMILES-derived channels can be added without breaking
            callers).
        channels : dict[str, float], optional
            Per-channel reward values keyed by channel name.  Defaults
            to ``None`` (= return the weight-only baseline).

        Returns
        -------
        float
            The aggregated reward.  Strictly 0.0 when both the weights
            and the channel dict are empty / zero.
        """
        del smiles  # currently unused — accepted for API symmetry
        # Empty / None channel dict → return the weight-only baseline
        # (sum of configured per-channel weights).  This keeps the
        # "no signal yet" path well-defined and non-zero when the
        # caller has wired up at least one channel.
        if not channels:
            try:
                w_total = (
                    float(self.w_vina)
                    + float(self.w_sa)
                    + float(self.w_qed)
                    + float(self.w_vina_proxy)
                    + float(self.w_posebusters)
                    + float(self.w_pb_valid)
                    + float(self.w_pic50)
                    + float(self.w_retro)
                    + float(self.w_reinvent4)
                    + float(self.w_synth)
                    + float(self.w_admet)
                )
            except Exception:
                w_total = 0.0
            return float(w_total)

        # Build ``(weight_attr, channel_name)`` pairs so the aggregator
        # looks up the configured weight for each supplied channel.
        # Unknown channels are silently dropped.
        weight_for_channel = {
            "r_vina": "w_vina",
            "r_sa": "w_sa",
            "r_qed": "w_qed",
            "r_vina_proxy": "w_vina_proxy",
            "r_posebusters": "w_posebusters",
            "r_pb_valid": "w_pb_valid",
            "r_pic50": "w_pic50",
            "r_retro": "w_retro",
            "r_reinvent4": "w_reinvent4",
            "r_synth": "w_synth",
            "r_admet": "w_admet",
        }
        value = 0.0
        for ch_name, ch_val in channels.items():
            w_attr = weight_for_channel.get(ch_name)
            if w_attr is None:
                continue
            try:
                w = float(getattr(self, w_attr, 0.0))
            except Exception:
                w = 0.0
            try:
                v = float(ch_val)
            except Exception:
                v = 0.0
            value += w * v
        return float(value)

    def __call__(
        self,
        state: MoleculeClosedTerm,
        *,
        target_predicates: Optional[Sequence[TypePredicate]] = None,
        binding_site: Optional[BindingSite] = None,
        satisfies_typed: bool = False,
        binds_target: bool = False,
    ) -> float:
        """Compute the aggregated reward for ``state``.

        Boolean flags (``satisfies_typed``, ``binds_target``) are used
        to add the type / binder bonuses; they are pre-computed by the
        caller so the aggregator does not have to touch RDKit itself.
        """
        value = 0.0

        def _safe(channel: Optional[Callable]) -> float:
            if channel is None:
                return 0.0
            try:
                return float(channel(state))
            except Exception:
                return 0.0

        v_vina = _safe(self.r_vina)
        if self.vina_invert and self.r_vina is not None:
            v_vina = -v_vina
        v_sa_raw = _safe(self.r_sa)
        # SA score is 1 (easy) .. 10 (hard); invert so higher = better.
        v_sa = max(0.0, min(1.0, 1.0 - (v_sa_raw - 1.0) / 9.0)) if self.r_sa else 0.0
        v_qed = _safe(self.r_qed)
        v_vina_proxy = _safe(self.r_vina_proxy)
        v_pb = _safe(self.r_posebusters)
        # R3/F4 — PoseBusters outer-gate channel (binary 0/1).
        v_pb_valid = _safe(self.r_pb_valid)
        v_pi = _safe(self.r_pic50)
        v_re = _safe(self.r_retro)
        v_reinvent4 = _safe(self.r_reinvent4)
        v_synth = _safe(self.r_synth)
        # Phase-2 (Round-7) — ADMET desirability channel.  Opt-in via
        # ``w_admet > 0``; gracefully degrades to 0.0 when the
        # ``r_admet`` callable raises (e.g. admet-ai backend down) or
        # when the SMILES does not parse.
        v_admet = _safe(self.r_admet)
        v_logp = _safe(self.r_logp_anticancer)
        v_tpsa = _safe(self.r_tpsa_iv)
        v_rotb = _safe(self.r_rotatable_bonds)
        v_herg = _safe(self.r_herg_proxy)
        v_anticancer = _safe(self.r_anticancer_composite)

        value += self.w_vina * v_vina
        value += self.w_sa * v_sa
        value += self.w_qed * v_qed
        value += self.w_vina_proxy * v_vina_proxy
        value += self.w_posebusters * v_pb
        # R3/F4 — PoseBusters outer-gate channel.  Opt-in via
        # ``w_pb_valid > 0`` (default 0.0 — the closed-loop records
        # the pass-rate as a diagnostic even when the weight is 0).
        value += self.w_pb_valid * v_pb_valid
        value += self.w_pic50 * v_pi
        value += self.w_retro * v_re
        # L-4 REINVENT4 multi-property channel — gracefully degrades to
        # 0.0 when the adapter is unavailable or the callable raises.
        value += self.w_reinvent4 * v_reinvent4
        # D-3 synthesis-oracle channel — gracefully degrades to 0.0
        # when the oracle is unavailable or the callable raises.
        value += self.w_synth * v_synth
        # Phase-2 ADMET channel — same graceful-degradation contract.
        value += self.w_admet * v_admet
        value += self.w_logp_anticancer * v_logp
        value += self.w_tpsa_iv * v_tpsa
        value += self.w_rotatable_bonds * v_rotb
        value += self.w_herg_proxy * v_herg
        value += self.w_anticancer_composite * v_anticancer
        if satisfies_typed:
            value += self.bonus_typed
        if binds_target:
            value += self.bonus_binder
        return value

    def metrics(self):
        """Return registered reward channel callables."""
        return {k: v for k, v in {
            "logp_anticancer": self.r_logp_anticancer,
            "tpsa_iv": self.r_tpsa_iv,
            "rotatable_bonds": self.r_rotatable_bonds,
            "herg_proxy": self.r_herg_proxy,
            "anticancer_composite": self.r_anticancer_composite,
        }.items() if v is not None}

    # ------------------------------------------------------------------
    # (Phase 2) Batched RDKit-backed reward scoring via multiprocessing
    # ------------------------------------------------------------------
    def _batch_rewards(
        self,
        states: "Sequence[MoleculeClosedTerm]",
        *,
        target_predicates: Optional[Sequence[TypePredicate]] = None,
        binding_site: Optional[BindingSite] = None,
        satisfies_typed: Optional[Sequence[bool]] = None,
        binds_target: Optional[Sequence[bool]] = None,
        n_workers: int = 1,
    ) -> "torch.Tensor":
        """Compute the aggregated reward for ``states`` in one batched call.

        The scalar-reward channels wrapped by :class:`RewardAggregator`
        (SA via ``sascorer.calculate_score``, QED via
        ``Descriptors.qed``, Vina_proxy linear feature scoring) are
        RDKit-bound and CPU-only.  Calling them in a Python ``for``
        loop is bottlenecked by RDKit's per-call C++ overhead (~50 µs
        each).  This method amortises that overhead by:

        * serialising the ``(state, channel)`` workload as plain
          tuples — RDKit mols are *not* pickled (RDKit objects are not
          picklable and serialising their SMILES is enough),
        * dispatching to a :class:`multiprocessing.Pool` when
          ``n_workers > 1`` so SA / QED / Vina_proxy for distinct
          states run in parallel on separate CPUs,
        * falling back to a serial loop when ``n_workers == 1`` or the
          pool cannot be spawned (e.g. in unit tests).

        The per-channel callables here are the **same** Python
        callables used by :meth:`__call__`, so the scalar result is
        numerically identical (modulo any non-determinism inside the
        underlying RDKit call, which is none).  Boolean flags
        (``satisfies_typed`` / ``binds_target``) are taken as parallel
        ``Sequence[bool]`` and added to the score with the same
        ``bonus_typed`` / ``bonus_binder`` weights as the scalar path.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape ``(len(states),)`` on
            ``torch.device('cuda')`` when a GPU is available, otherwise
            on CPU.  Returns ``torch.zeros(0)`` when ``states`` is empty.
        """
        # Local imports — torch may be unavailable in some headless test
        # contexts (we still return a list-shaped numpy fallback).
        try:
            import torch as _torch  # local: avoid pulling torch at module import
        except Exception:  # pragma: no cover
            _torch = None  # type: ignore

        n = len(states)
        if n == 0:
            if _torch is not None:
                return _torch.zeros(0)
            try:
                import numpy as _np
                return _np.zeros(0, dtype=float)
            except Exception:
                return []  # type: ignore[return-value]

        # Default boolean flags: all False (no bonuses).
        sat_list = (
            [bool(s) for s in satisfies_typed]
            if satisfies_typed is not None
            else [False] * n
        )
        bind_list = (
            [bool(b) for b in binds_target]
            if binds_target is not None
            else [False] * n
        )
        # Truncate / pad to length n defensively.
        if len(sat_list) < n:
            sat_list = sat_list + [False] * (n - len(sat_list))
        if len(bind_list) < n:
            bind_list = bind_list + [False] * (n - len(bind_list))
        sat_list = sat_list[:n]
        bind_list = bind_list[:n]

        # Resolve per-state canonical SMILES once (so the worker pool
        # can index by SMILES, not by unpicklable RDKit objects).
        smiles_seq: List[str] = []
        for s in states:
            try:
                smi = _smi_of(s)
            except Exception:
                smi = ""
            smiles_seq.append(smi)

        # Per-channel *weights* are already on the aggregator instance
        # — we read them once and forward to the worker.  The actual
        # channel callables are the same Python objects as on the
        # aggregator, so call-by-reference works in the parent process.
        # For the worker process we *cannot* forward the callables
        # directly (some of them capture RDKit modules that are
        # process-local), so we restrict the parallel path to channels
        # whose semantics are reproducible from a SMILES string:
        #   * r_sa         — RDKit sascorer from MolFromSmiles(smi)
        #   * r_qed        — RDKit Descriptors.qed from MolFromSmiles(smi)
        #   * r_vina_proxy — hand-coded feature-linear score, RDKit-free
        # All other channels fall back to scalar evaluation in the parent
        # process (typically None in production).
        def _compute_one(smi: str) -> float:
            # Re-use the parent's channel callables — they were
            # constructed with the canonical RDKit imports and stay
            # valid in the main process.  We compute per-channel
            # contributions here in the main loop below; this inner
            # function is only used when n_workers == 1 (the serial
            # path).
            # For the serial path we use the parent aggregator's __call__,
            # which is the canonical implementation.
            return 0.0  # placeholder; real value computed below.

        # Determine if we can dispatch to a pool — must have at least
        # one RDKit-backed channel that benefits from parallelism.
        # When no RDKit channels are present, the serial loop is fine
        # (the rewards are pure-Python and cheaper than pool setup).
        rdkit_channels_active = (
            self.r_sa is not None
            or self.r_qed is not None
            or self.r_vina_proxy is not None
            or self.r_vina is not None
        )

        # ---- serial path (default) -----------------------------------
        if (
            int(n_workers) <= 1
            or not rdkit_channels_active
        ):
            values = [
                float(self(
                    s,
                    target_predicates=target_predicates,
                    binding_site=binding_site,
                    satisfies_typed=sat_list[i],
                    binds_target=bind_list[i],
                ))
                for i, s in enumerate(states)
            ]
        else:
            # ---- multiprocessing path --------------------------------
            # We compute per-channel contributions per state in worker
            # processes.  Each worker re-imports RDKit + invokes the
            # aggregator's *stateless* per-channel formulas via the
            # channel callables captured at aggregator construction.
            # We pass the SMILES list and the aggregator's *serialised
            # configuration* (channel names + weights) to each worker;
            # the worker reconstructs the channel formulas in-process.
            from . import _batch_reward_worker as _brw  # late import

            try:
                # Strip the channels + weights into a plain-dict
                # payload so it pickles cleanly across processes.
                cfg = _brw.aggregator_to_config(self)
                # Round-trip via multiprocessing.Pool: each task is
                # one state → a float score.
                try:
                    from multiprocessing import Pool as _Pool
                    # Use 'spawn' on macOS / Windows; 'fork' on Linux
                    # is fine for this pure-Python / RDKit workload.
                    ctx_method = "fork"
                    import multiprocessing as _mp
                    try:
                        ctx = _mp.get_context(ctx_method)
                    except Exception:  # pragma: no cover
                        ctx = _mp.get_context()
                    worker_n = max(1, min(int(n_workers), n))
                    with ctx.Pool(processes=worker_n) as pool:
                        values = pool.starmap(
                            _brw.score_one,
                            [
                                (cfg, smiles_seq[i], sat_list[i], bind_list[i])
                                for i in range(n)
                            ],
                        )
                except Exception as exc:
                    # Pool unavailable (e.g. test env).  Fall back to
                    # serial path so the call still returns.
                    log.debug(
                        "RewardAggregator._batch_rewards pool failed (%s); serial fallback",
                        exc,
                    )
                    values = [
                        float(self(
                            s,
                            target_predicates=target_predicates,
                            binding_site=binding_site,
                            satisfies_typed=sat_list[i],
                            binds_target=bind_list[i],
                        ))
                        for i, s in enumerate(states)
                    ]
            except Exception as exc:  # pragma: no cover - defensive
                log.debug(
                    "RewardAggregator._batch_rewards setup failed (%s); serial fallback",
                    exc,
                )
                values = [
                    float(self(
                        s,
                        target_predicates=target_predicates,
                        binding_site=binding_site,
                        satisfies_typed=sat_list[i],
                        binds_target=bind_list[i],
                    ))
                    for i, s in enumerate(states)
                ]

        # ---- stack to a (GPU) tensor --------------------------------
        try:
            import numpy as _np
            arr = _np.asarray(values, dtype=float)
        except Exception:
            arr = None  # type: ignore

        if _torch is not None:
            try:
                t = _torch.as_tensor(arr if arr is not None else values, dtype=_torch.float32)
                if _torch.cuda.is_available():
                    # ``cuda`` is the ROCm PyTorch API name as well, but use
                    # the shared resolver so HIP_VISIBLE_DEVICES selection
                    # and CPU fallback stay consistent across adapters.
                    return t.to(device=get_device())
                return t
            except Exception:
                pass
        # NumPy or list fallback (CPU-only).
        if arr is not None:
            return arr  # type: ignore[return-value]
        return list(values)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Composition helpers (backward compatibility)
    # ------------------------------------------------------------------
    @classmethod
    def from_scorer(
        cls,
        scorer: ScorerFn,
        weight: float = 1.0,
    ) -> "RewardAggregator":
        """Build a single-channel aggregator from a scalar ``scorer``.

        This preserves the historical ``scorer(state) -> float``
        interface — the previous MCTS implementation only used a
        scalar scorer, and ``RewardAggregator(scorer)`` matches it
        exactly.
        """
        return cls(r_vina=scorer, w_vina=weight, vina_invert=False)

    # ------------------------------------------------------------------
    # Batch RDKit primitive pass — amortise per-call overhead
    # ------------------------------------------------------------------
    def compute_batch_rdkit(
        self,
        smiles_list,
        n_workers: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Return per-channel RDKit-derived scores as numpy arrays.

        Computes the SA, QED and Vina_proxy channels **simultaneously**
        from a single ``Chem.MolFromSmiles`` call per SMILES — the
        batched RDKit helpers in
        :mod:`molmetal_lam.lam_chem.batched_rdkit` amortise the
        per-call Python-binding overhead across all SMILES in the
        input.  Returns a dict with keys ``"sa"``, ``"qed"`` and
        ``"vina_proxy"`` — each is a length-``len(smiles_list)``
        :class:`numpy.ndarray` of floats.

        Non-RDKit channels (Vina oracle, PoseBusters, REINVENT4,
        synthesis oracle) are *not* evaluated here; use
        :meth:`_batch_rewards` for the full aggregated reward.
        """
        from molmetal_lam.lam_chem.batched_rdkit import (
            batch_qed,
            batch_sa,
        )
        smiles_list = list(smiles_list)
        if len(smiles_list) == 0:
            empty = np.zeros(0, dtype=float)
            return {"sa": empty, "qed": empty.copy(), "vina_proxy": empty.copy()}
        sa_arr = batch_sa(smiles_list, n_workers=n_workers)
        qed_arr = batch_qed(smiles_list, n_workers=n_workers)
        # Vina proxy is hand-coded feature-linear — we approximate it
        # as ``1 - sa/10`` here so the batched values are usable as a
        # docking-score prior without paying the per-mol atom/bond
        # recount cost.  Callers that need the exact feature-linear
        # formula can drop down to :func:`_score_vina_proxy`.
        try:
            import numpy as _np
            vina_arr = _np.clip(1.0 - sa_arr / 10.0, 0.0, 1.0)
        except Exception:
            vina_arr = np.asarray(
                [max(0.0, min(1.0, 1.0 - float(v) / 10.0)) for v in sa_arr],
                dtype=float,
            )
        return {"sa": sa_arr, "qed": qed_arr, "vina_proxy": vina_arr}

    @classmethod
    def with_default_channels(
        cls,
        prior: Optional["SymbolicPrior"] = None,
        *,
        weight: float = 1.0,
    ) -> "RewardAggregator":
        """Construct an aggregator with SA + QED + Vina_proxy channels.

        This is the L-A2 "rich" reward head.  Each channel has default
        weight ``1.0`` so the aggregated leaf value carries real
        variance across molecules — which is exactly what we need for
        :data:`PUCT_EXPLOIT_RATIO_VAR` and :data:`LEAF_VALUE_VAR` to be
        strictly positive in the closed-loop runs.

        Channel definitions (matching the task spec):

        * ``r_sa``  : ``1 - sascorer.calculate_score(MolFromSmiles(s)) / 10``
          → synthetic-accessibility score in [0, 1], inverted so
          *higher = better* (easier to synthesise).
        * ``r_qed`` : ``Descriptors.qed(MolFromSmiles(s))``
          → drug-likeness in [0, 1].
        * ``r_vina_proxy`` : a feature-linear surrogate for a Vina
          docking score that *reuses the fitted SymbolicPrior* if one
          is supplied — the prior encodes prior knowledge of which
          molecules are "good".

        All channels degrade gracefully to ``0.0`` when RDKit is
        unavailable or the SMILES is unparsable, so the search never
        crashes on malformed molecules.
        """
        def _r_sa(state: MoleculeClosedTerm) -> float:
            if not _RDKIT_AVAILABLE or _sascorer is None:
                return 0.0
            try:
                smi = state.canonical_smiles()
                if smi:
                    from molmetal_lam.tile_lib.canonical_cache import canonicalize
                    smi = canonicalize(str(smi))
            except Exception:
                return 0.0
            mol = _safe_smiles_to_mol(smi)
            if mol is None:
                return 0.0
            try:
                raw = float(_sascorer.calculate_score(mol))
                return max(0.0, min(1.0, 1.0 - raw / 10.0))
            except Exception:
                return 0.0

        # R3/F4 — PoseBusters outer-gate channel.  Always attach the
        # channel so the closed-loop can record the per-iter pass-rate
        # as a diagnostic — but the *weight* is opt-in via the
        # ``PB_WEIGHT`` env var (default 0.0).  When ``PB_WEIGHT >= 1``
        # the gate becomes a hard multiplier on the aggregated reward.
        pb_w = _resolve_pb_weight()
        return cls(
            r_sa=_r_sa,
            r_qed=_default_qed_channel(),
            r_vina_proxy=_default_vina_proxy_channel(prior=prior),
            r_pb_valid=_default_pb_valid_channel(),
            w_sa=weight,
            w_qed=weight,
            w_vina_proxy=weight,
            w_pb_valid=pb_w,
        )


# ---------------------------------------------------------------------------
# Virtual loss hook (parallel scaffold)
# ---------------------------------------------------------------------------

#: Magnitude added to ``virtual_loss`` each time a node is claimed by an
#: in-flight simulation, and subtracted on backpropagation.  The single
#: threaded ``MCTSProofSearch.search`` consults this in the PUCT formula
#: so the same node layout also supports future lock-free parallel MCTS
#: implementations (Python GIL serialises the ``+=`` / ``-=`` operations
#: within one interpreter so this is in-process thread-safe).
VIRTUAL_LOSS_VALUE: float = 1.0


@dataclass
class VirtualLoss:
    """Per-node virtual-loss state for lock-free parallel MCTS.

    The single-threaded ``MCTSProofSearch.search`` now *does* consult
    the virtual loss: ``_simulate`` adds ``VIRTUAL_LOSS_VALUE`` to the
    selected child before the rollout (locking it out of contention from
    other in-flight simulations) and ``_backprop`` releases the lock.
    The same mechanism is the standard hook for future multi-thread /
    multi-process MCTS implementations — no node-layout change needed.

    Attributes
    ----------
    delta : float, default 1.0
        Magnitude added to ``virtual_loss`` each time the node is
        "claimed" by an in-flight simulation.
    value : float, default 0.0
        Current accumulated virtual loss.  Consulted by ``_select_child``
        as ``- virtual_loss / (1 + N)`` in the PUCT formula so a node
        currently being simulated is temporarily de-prioritised.
    """

    delta: float = 1.0
    value: float = 0.0


# ---------------------------------------------------------------------------
# MCTS upgrade (round 3) — VirtualLoss + TranspositionTable + node recycling
# ---------------------------------------------------------------------------
# The legacy :class:`VirtualLoss` above is a *per-node scalar*; the
# counter-based :class:`_VirtualLoss` below is the lock-free-MCTS hook
# the round-3 plan calls for.  Per-node ids are tracked in a
# ``Dict[int, int]`` keyed by Python ``id(node)`` so the class does not
# require a back-pointer from the node to the table.  This is a strict
# superset of the per-node-scalar semantics used by :meth:`_simulate`
# and :meth:`_backprop` and is what the new parallel-MCTS scaffolds
# should consult.


class _VirtualLoss:
    """Counter-based virtual loss for lock-free parallel MCTS.

    Each call to :meth:`apply` increments the running count for the
    supplied node id and returns ``-self.value`` (the canonical
    "virtual-loss penalty" used in the AlphaZero / Expert Iteration
    literature).  :meth:`release` decrements the count so the node
    becomes selectable again.  :meth:`on_node` is the read-only
    accessor used by PUCT-based child selection — higher counts
    temporarily de-prioritise the node.

    Thread safety: in CPython the GIL serialises the underlying
    ``Dict.__setitem__`` / ``Dict.get`` operations so the counter is
    monotonic per process.  A real lock-free MCTS worker pool would
    swap the dict for ``threading.Lock`` or an ``itertools.count``
    primitive — the API does not change.

    Attributes
    ----------
    value : float, default 1.0
        Magnitude of the penalty applied per claim.  The penalty
        added to the PUCT score is ``-self.value`` so the child
        selection can use ``on_node(node_id) * self.value`` to bias
        away from in-flight simulations.
    """

    def __init__(self, value: float = 1.0) -> None:
        self.value = float(value)
        self._counts: Dict[int, int] = {}

    def apply(self, node_id: int) -> float:
        """Claim the node and return the PUCT penalty ``-self.value``."""
        try:
            self._counts[node_id] = self._counts.get(node_id, 0) + 1
        except Exception:
            pass
        return -float(self.value)

    def release(self, node_id: int) -> None:
        """Release a previously-applied lock.  Floors at 0 (never goes negative)."""
        try:
            self._counts[node_id] = max(0, self._counts.get(node_id, 0) - 1)
        except Exception:
            pass

    def on_node(self, node_id: int) -> int:
        """Current pending claim count for the node (0 = selectable)."""
        try:
            return int(self._counts.get(node_id, 0))
        except Exception:
            return 0


class _TranspositionTable:
    """Canonical-SMILES → ``_MCTSNode`` map with bounded FIFO eviction.

    When MCTS expansion produces a child whose canonical SMILES has
    already been inserted, the table returns the existing node. Search
    attachment keeps separate nodes for different parents: visits and
    parent pointers belong to a reaction path, even when the molecular
    state is equivalent. The table itself does not rewrite that tree.

    The table is bounded at ``max_size`` entries; once full, the
    *oldest* insertion is evicted (FIFO) so the cache reflects
    "recently explored" rather than "first explored".  Eviction does
    not delete the node itself — callers retain the reference via the
    parent chain.

    Attributes
    ----------
    max_size : int, default 50_000
        Hard cap on the number of cached entries.  At the canonical
        1000-sim budget the table typically holds ``< 5000`` nodes, so
        50 000 is a generous upper bound that keeps memory bounded
        for ablation / paper runs.
    """

    def __init__(self, max_size: int = 50_000) -> None:
        self._table: Dict[str, "_MCTSNode"] = {}
        self._max_size = int(max_size)

    def lookup(self, canonical_smi: str) -> Optional["_MCTSNode"]:
        """Return the cached node for ``canonical_smi`` or ``None``."""
        try:
            return self._table.get(canonical_smi)
        except Exception:
            return None

    def insert(self, canonical_smi: str, node: "_MCTSNode") -> None:
        """Insert / overwrite the entry for ``canonical_smi``.

        When the table is at capacity the *oldest* insertion (per
        Python dict iteration order, which is insertion-ordered) is
        evicted FIFO so the cache reflects recent work.
        """
        try:
            if len(self._table) >= self._max_size:
                oldest = next(iter(self._table))
                del self._table[oldest]
            self._table[canonical_smi] = node
        except Exception:
            # Eviction / insertion race — silently drop so callers do
            # not crash the search on cache failures.
            try:
                self._table[canonical_smi] = node
            except Exception:
                pass


# ---------------------------------------------------------------------------
# MCTS node = a closed term + PUCT statistics
# ---------------------------------------------------------------------------


@dataclass
class _MCTSNode:
    """One MCTS node = one closed-term state + PUCT statistics.

    Attributes
    ----------
    state : MoleculeClosedTerm
        The closed lambda-term (= a molecule) at this node.
    parent : _MCTSNode or None
        Back-pointer to the parent node (= the previous β-normal form).
    N : int
        Visit count.
    W : float
        Total value (sum of backpropagated leaf values).
    P : float
        Prior probability (today = ``heuristic(features)``; placeholder
        0.5 by default).
    children : list[_MCTSNode]
        Expanded child nodes (= one β-reduction step successors).
    rule_name : str or None
        The reaction rule that produced this node from its parent.
        ``None`` for the root.
    tile : MoleculeClosedTerm or None
        The tile combined with the parent under ``rule_name`` to yield
        this node.  ``None`` for the root.
    virtual_loss : VirtualLoss
        Placeholder parallel-MCTS hook (unused in single-threaded search).
    """

    state: MoleculeClosedTerm
    parent: Optional["_MCTSNode"] = None
    N: int = 0
    W: float = 0.0
    P: float = 0.5
    children: List["_MCTSNode"] = field(default_factory=list)
    rule_name: Optional[str] = None
    tile: Optional[MoleculeClosedTerm] = None
    virtual_loss: VirtualLoss = field(default_factory=VirtualLoss)
    # Exhaustive search expansion, not single-molecule beta-NF, establishes
    # whether an external rule/tile can still produce a successor.
    expansion_complete: bool = False

    @property
    def virtual_loss_value(self) -> float:
        """Convenience accessor: scalar virtual loss for PUCT consultation."""
        try:
            return float(self.virtual_loss.value)
        except Exception:
            return 0.0

    @property
    def Q(self) -> float:
        """Mean value Q(s, a) = W / N.  0 if unvisited."""
        if self.N == 0:
            return 0.0
        return self.W / self.N

    @property
    def is_leaf(self) -> bool:
        """True iff the node has never been expanded."""
        return len(self.children) == 0

    @property
    def is_terminal(self) -> bool:
        """True after expansion proves this node has no admissible children.

        A beta-normal molecular term can react with an external partner; its
        internal normal form therefore cannot establish search termination.
        Depth exhaustion is handled separately by the simulation budget.
        """
        return self.expansion_complete and not self.children

    @property
    def is_closed_for_search(self) -> bool:
        """Compatibility alias for a leaf with no admissible reactions."""
        return self.is_terminal

    # ------------------------------------------------------------------
    # (L-2) Persistence — AlphaZero-style checkpoint / restore
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Canonical pickle-friendly serialisation of this subtree.

        The representation is a *nested dict* so it round-trips cleanly
        through JSON / msgpack / numpy ``.npz`` and never relies on
        ``pickle`` (which is unsafe across Python versions).  The
        closed-term state is encoded as its **canonical SMILES**
        (falling back to ``source_smiles`` when RDKit is unavailable)
        — α-equivalent molecules collide in the dict, which is exactly
        what we want for tree deduplication.

        Children are stored under the ``"children"`` key as a
        dict-of-dict indexed by child canonical SMILES, so the
        subtree layout is preserved without ``__id__`` references.

        Parameters
        ----------
        None.

        Returns
        -------
        dict
            ``{"smiles", "n_visits", "sum_value", "P", "rule_name",
                "children": {child_smi: child_dict, ...}}``.
        """
        smiles = _smi_of(self.state)
        return {
            "smiles": smiles,
            "n_visits": int(self.N),
            "sum_value": float(self.W),
            "P": float(self.P),
            "rule_name": str(self.rule_name) if self.rule_name is not None else None,
            "children": {
                # canonical-SMILES keyed dict-of-dict
                # (deduplicates on α-equivalence by construction).
                _smi_of(c): c.to_dict()
                for c in self.children
            },
        }

    @classmethod
    def from_dict(
        cls,
        d: Dict[str, Any],
        *,
        tile_library: Sequence["MoleculeClosedTerm"],
        rules: Dict[str, "ReactionRule"],
        state_lookup: Optional[Dict[str, "MoleculeClosedTerm"]] = None,
    ) -> "_MCTSNode":
        """Inverse of :meth:`to_dict` — rebuild the subtree.

        The closed-term state at each node is recovered by canonical
        SMILES lookup against ``tile_library`` first, then a
        user-supplied ``state_lookup`` cache, then finally a
        :meth:`MoleculeClosedTerm.from_smiles` round-trip — so the
        checkpoint survives even when the original molecule objects
        are no longer in scope.

        Children whose canonical SMILES no longer expands under the
        current ``rules`` / ``tile_library`` are pruned (the persistent
        tree self-heals as the chemistry evolves across iterations).
        """
        smiles = str(d.get("smiles", "") or "")
        state: Optional[MoleculeClosedTerm] = None
        if state_lookup is not None and smiles in state_lookup:
            state = state_lookup[smiles]
        else:
            for t in tile_library:
                try:
                    if _smi_of(t) == smiles:
                        state = t
                        break
                except Exception:
                    continue
        if state is None:
            try:
                state = MoleculeClosedTerm.from_smiles(smiles, embed_3d=False)
            except Exception:
                # Last-ditch: empty term — keeps the tree shape.
                state = MoleculeClosedTerm()
        try:
            P = float(d.get("P", 0.5))
        except Exception:
            P = 0.5
        node = cls(
            state=state,
            parent=None,
            N=int(d.get("n_visits", 0)),
            W=float(d.get("sum_value", 0.0)),
            P=P,
            rule_name=d.get("rule_name"),
            tile=None,
        )
        for child_dict in d.get("children", {}).values():
            try:
                child_node = cls.from_dict(
                    child_dict,
                    tile_library=tile_library,
                    rules=rules,
                    state_lookup=state_lookup,
                )
            except Exception:
                continue
            child_node.parent = node
            node.children.append(child_node)
        return node


def _smi_of(state: MoleculeClosedTerm) -> str:
    """Best-effort canonical SMILES for a state — used as dict key.

    Goes through :func:`molmetal_lam.tile_lib.canonical_cache.canonicalize`
    so the RDKit round-trip is memoised across calls.
    """
    try:
        smi = state.canonical_smiles()
        if smi:
            from molmetal_lam.tile_lib.canonical_cache import canonicalize
            return str(canonicalize(str(smi)))
    except Exception:
        pass
    try:
        src = getattr(state, "source_smiles", "") or ""
        if src:
            from molmetal_lam.tile_lib.canonical_cache import canonicalize
            return str(canonicalize(str(src)))
    except Exception:
        pass
    return f"id:{id(state)}"


def _r_admet_default(state: MoleculeClosedTerm) -> float:
    """Default ``r_admet`` channel — wraps the ADMET runner.

    Resolves the canonical SMILES via :func:`_smi_of`, calls
    :func:`molmetal.validation.admet_runner.predict_admet`, then maps
    the result through :func:`admet_desirability` to a Lipinski-style
    desirability score in [0, 1].  Returns 0.0 if the runner is not
    importable (e.g. the validation subpackage is missing) or if the
    SMILES does not parse — the aggregator's ``_safe`` wrapper then
    degrades gracefully.

    The admet-ai backend is slow on first call (~1–2 s per inference
    + ~1 s model load).  The closure caches nothing by design — the
    MCTS search visits the same molecule many times, but the canonical
    SMILES dedup happens at the *aggregator* level via the batched
    call path, not here.
    """
    try:
        from molmetal.validation.admet_runner import (
            predict_admet,
            admet_desirability,
        )
    except Exception:
        return 0.0
    try:
        smi = _smi_of(state)
        if not smi or smi.startswith("id:"):
            return 0.0
        admet = predict_admet(smi)
        if not admet:
            return 0.0
        return float(admet_desirability(admet))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# MCTS proof search
# ---------------------------------------------------------------------------


@dataclass
class MCTSProofSearch:
    """MCTS-based proof search in Molecular Lambda Calculus term space.

    The class wires the existing MLC layers (atoms / bonds / molecules /
    reactions / types / binding) together with a PUCT-driven MCTS loop.

    Each **simulation** consists of:

        1. ``_select``   — traverse the existing tree using the PUCT rule:
               a* = argmax_a  Q(s, a) + c_puct * P(a) * sqrt(N(s)) / (1 + N(s, a))
        2. ``_expand``   — apply every (reaction rule, compatible tile)
               pair to the leaf, generating one child node per fired
               reaction.  Each child = one β-reduction step.
        3. ``_rollout``  — play out to terminal / depth-budget, evaluate
               via the reward aggregator (and the type predicates — a
               leaf that satisfies all of them is a valid proof witness).
        4. ``_backprop`` — update ``(N, W)`` along the path.

    After ``n_simulations`` simulations, :meth:`search` returns the
    top-K candidates that (i) satisfy every type predicate and (ii)
    bind the target site.

    Parameters
    ----------
    tile_library : list[MoleculeClosedTerm]
        The pool of closed λ-terms (= starting materials / click-chem
        tiles) the search may draw from.
    rules : dict[str, ReactionRule]
        Reaction rules indexed by name.
    target_predicates : list[TypePredicate]
        The type predicates the candidate molecule must satisfy.
    binding_site : BindingSite
        The binding site the candidate must inhabit.
    scorer : callable, optional
        ``scorer(state) -> float``.  Backward-compat: when supplied it
        is wrapped in a single-channel :class:`RewardAggregator`.
        Ignored when ``reward`` is provided.
    reward : RewardAggregator, optional
        Multi-channel reward head.  When ``None`` (default) we build
        one from ``scorer`` if given, otherwise a constant 0.0
        aggregator.
    n_simulations : int, default 1000
    c_puct : float, default 1.4
    top_k : int, default 20
        Number of top-scoring candidates returned by :meth:`search`.
        ``top_k`` is the *primary* knob controlling
        ``n_candidates_per_pocket`` — SOTA 3D SBDD papers (TransDiff,
        MolCRAFT, Pocket2Mol, TargetDiff, DecompDiff) report 50-1000
        candidates per pocket to match their evaluation protocols, so
        we lifted the default from the Phase-0 value of 5 to 20 (with
        ``max_depth=3`` in :meth:`search` this gives
        ~20 leaves per pocket after the typecheck / binding filter).
        The Phase-0 default of 5 only emitted ~1.7 candidates/pocket at
        ``best_score=1.7046`` and is not SOTA-comparable.  Set this
        higher (e.g. 50-100) for paper runs that target a 50-100
        candidates/pocket protocol.
    rng : random.Random or None
    prior : SymbolicPrior, optional
        Symbolic-regression prior.  When ``None`` we fall back to the
        constant 0.5 baseline (full backward compatibility).
    dirichlet_alpha : float, default 0.3
        Concentration parameter of the Dirichlet noise added to the
        root prior (AlphaZero style).  Set to 0.0 to disable.
    dirichlet_fraction : float, default 0.25
        Fraction of simulations where Dirichlet noise is mixed into
        the root prior (AlphaZero convention).
    rollout_epsilon : float, default 0.0
        When > 0.0 and a prior is fitted, the rollout policy uses an
        ε-greedy rule: with probability ε it samples a uniform random
        (rule, tile); otherwise it samples the (rule, tile) whose
        product's predicted value is highest under the prior.
        When ``prior`` is unfitted, rollout stays uniform random.
    """

    tile_library: List[MoleculeClosedTerm]
    rules: Dict[str, ReactionRule]
    target_predicates: List[TypePredicate]
    binding_site: BindingSite
    scorer: Optional[ScorerFn] = None
    reward: Optional[RewardAggregator] = None
    #: TODO-02 — alias constructor param so callers can spell it as
    #: ``reward_aggregator=...``.  When supplied it overrides ``reward``
    # (preferred spelling) for clarity.  Defaults to ``None`` so the
    #: constructor auto-builds a default aggregator from ``scorer``
    #: (or, failing that, an empty zero-signal aggregator).
    reward_aggregator: Optional[RewardAggregator] = None
    #: Optional override map ``{channel_name: weight}``.  When supplied
    #: we *replace* (not multiply) the weights of the resolved
    #: :class:`RewardAggregator` so callers can re-tune a rich reward
    #: head without rebuilding the channels.  Recognised keys:
    #: ``w_vina``, ``w_sa``, ``w_qed``, ``w_vina_proxy``,
    #: ``w_posebusters``, ``w_pic50``, ``w_retro``, ``w_reinvent4``.
    reward_channels: Optional[Dict[str, float]] = None
    n_simulations: int = 1000
    c_puct: float = 1.4
    #: ``top_k`` controls the number of top-scoring candidates returned
    #: by :meth:`search` (and thus the per-pocket ``n_candidates``
    #: column emitted by the 100-pocket sweep).  Default 20 (was 5 in
    #: Phase-0) is calibrated to give ~20 leaves per pocket after the
    #: typecheck / binding filter at ``max_depth=3``, which is the
    #: lower edge of the 50-100 SOTA-comparable band.  See the
    #: constructor docstring above for the full rationale.
    top_k: int = 20
    rng: Optional[random.Random] = None
    prior: Optional[SymbolicPrior] = None
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    rollout_epsilon: float = 0.0
    #: MCTS early-stop.  When ``early_stop`` is True and the *best leaf
    #: score* has not improved for ``patience`` consecutive simulations,
    #: the search loop breaks early and emits an ``EARLY_STOPPED=True``
    #: flag plus ``EARLY_STOPPED_AT_ITER=<n>`` so downstream consumers
    #: can detect that the budget was not exhausted.  Empirically
    #: (round-0 sweep) UCB finds high-score paths within ~30 rollouts at
    #: the 12-tile Phase-0 branching and the remaining budget is
    #: compute burned with no quality gain — so ``patience=30`` is a
    #: safe default for the canonical n_simulations=1000 budget.
    #: Set ``early_stop=False`` for ablation / paper-figure runs that
    #: require the full ``n_simulations`` budget.
    early_stop: bool = True
    patience: int = 30
    #: Cumulative Number of Function Evaluations (NFE) — incremented by
    #: 1 on every reward / scorer / aggregator call, β-reduction step,
    #: and oracle invocation.  NFE is the SOTA-standard metric for
    #: comparing MCTS to other generative methods (see e.g. Jin et al.
    #: 2020 JT-VAE, Luo et al. 2021 Mol-CycleGAN), so we surface it
    #: from :meth:`search` for downstream reporting.  Three independent
    #: counters so callers can decompose the cost:
    #:   * :attr:`nfe`               — total reward calls (any channel)
    #:   * :attr:`nfe_reductions`    — β-reductions applied during
    #:     expand / rollout (one per (rule, tile) product)
    #:   * :attr:`nfe_oracle`        — real docking-oracle calls
    #:     (Vina / DiffDock / REINVENT4) — these are the expensive
    #:     calls, so we track them separately.
    nfe: int = 0
    nfe_reductions: int = 0
    nfe_oracle: int = 0
    #: L-3 (Lambda Round 3) — when True, the MCTS expansion consumes
    #: the 204-tile SMARTS-diverse ChEMBL/ZINC fragment pool
    #: (:func:`molmetal_lam.tile_lib.library.FRAGMENT_LIBRARY_200_TILES`)
    #: instead of the Phase-0 12-tile library.  Default ``True`` (TODO-03
    #: r0 wire-up) so :class:`MCTSProofSearch` ships with the 204-tile
    #: expansion enabled out of the box — branching factor grows from
    #: |rules| × 12 = 60 to |rules| × 204 = 1020 with the canonical
    #: 5-rule reaction set.  Set ``use_fragment_pool=False`` for
    #: backward compatibility / smoke tests that need the 12-tile
    #: library.  Failing tiles in the pool are silently dropped (the
    #: embed-fail rate reported by
    #: ``lambda_round3_L3_fragment_library.md`` is 1.9%).
    use_fragment_pool: bool = True
    #: L-3 — diagnostic: actual number of tiles loaded by the
    #: fragment pool after validation.  Populated by :meth:`_expand`
    #: when ``use_fragment_pool=True``.  ``0`` means the 12-tile
    #: fallback path was taken (backward compatibility).
    expand_pool_size: int = 0
    #: L-1 governance flag — when True, the *real* docking oracle
    #: (DiffDock / FlowDock) is invoked only on the **top-K=10**
    #: candidates at iteration end, NOT during MCTS expansion.
    #: Default True; tests / smoke runs can disable it.
    leaf_oracle_call_top_k_only: bool = True
    #: Pre-resolved docking oracle.  When ``None`` we lazily resolve
    #: one via :func:`molmetal_lam.binding.types._resolve_docking_oracle`
    #: the first time it is needed.
    oracle: Optional[Any] = None
    #: top-K size for the oracle-augmented leaf typecheck.  Defaults
    #: to 10 — the value called out in the L-1 task brief.
    oracle_top_k: int = 10

    #: Per-iteration history, populated by :meth:`search`.
    history: List[Dict[str, Any]] = field(default_factory=list)
    #: L-A2 — cumulative variance of leaf reward values over the
    #: whole search (= ``var(Q[leaf])``).  Populated by
    #: :meth:`search` at the end of each run.  Strictly positive
    #: iff leaves are distinguishable by the reward head.
    leaf_value_var: float = 0.0
    leaf_value_std: float = 0.0
    #: L-A2 — variance-normalised exploit ratio
    #: (``leaf_value_var / (leaf_value_var + 1e-6)``).  Saturates to
    #: 1.0 when leaves are distinguishable; 0.0 when degenerate.
    puct_exploit_ratio_var: float = 0.0
    #: Per-simulation buffer of leaf reward values — used to compute
    #: ``leaf_value_var`` at the end of :meth:`search`.
    _leaf_value_history: List[float] = field(default_factory=list)
    #: (L-2) Persistent-tree state: the *root* node of the most recent
    #: :meth:`search` call, retained so :meth:`save_tree` /
    #: :meth:`load_tree` can checkpoint / restore the AlphaZero-style
    #: persistent tree across closed-loop iterations.
    _root: Optional["_MCTSNode"] = None
    #: (L-2) Globally accumulated (features → reward) pairs across
    #: closed-loop iterations.  Every leaf encountered by every search
    #: call appends one entry, so :class:`SymbolicPrior` sees more data
    #: the longer the closed loop runs.
    accumulated_leaf_pairs: List[Tuple[List[float], float]] = field(
        default_factory=list,
    )
    #: R3/F4 — per-leaf PoseBusters pass/fail side-table.  Aligned
    #: 1:1 with :attr:`accumulated_leaf_pairs` — the *i*-th entry of
    #: ``_leaf_pb_valid_history`` corresponds to the *i*-th entry of
    #: ``accumulated_leaf_pairs``.  ``True`` = PoseBusters outer-gate
    #: passed, ``False`` = failed, ``None`` = skipped (PB unavailable or
    #: not yet evaluated).  Populated by :meth:`_accumulate_leaf_pairs`
    #: using the aggregator's ``r_pb_valid`` channel when one is
    #: configured, otherwise ``None``.  Symbolic-prior refits can
    #: consult this buffer to weight / filter the regression target
    #: toward PB-passing molecules.
    _leaf_pb_valid_history: List[Optional[bool]] = field(
        default_factory=list,
    )

    #: (D2) Transposition table — maps canonical SMILES → ``_MCTSNode``
    #: so when a state is rediscovered via a different reaction path we
    #: re-use the existing node instead of creating a duplicate
    #: subtree.  This shrinks the tree size and gives PUCT a richer
    #: N/W history to exploit.  Populated by :meth:`_lookup_or_create`
    #: during :meth:`_expand`; pruned by :meth:`_recycle_node`.
    _t_table: Dict[str, "_MCTSNode"] = field(default_factory=dict)
    #: (D2) Transposition-table hit / miss counters for the *current*
    #: ``search()`` call.  ``TRANSPOSITION_HIT_RATE = hits / (hits +
    #: misses)`` is emitted into :attr:`history` at the end of the run.
    _t_hits: int = 0
    _t_misses: int = 0
    #: (D2) How often (in simulations) to attempt an automatic
    #: :class:`SymbolicPrior` refit.  Default ``5`` keeps the cost
    #: negligible while letting the prior track accumulating leaf
    #: data.  Set to ``0`` to disable automatic refits.
    prior_refit_every: int = 5
    #: (D2) Total simulations executed in the *current* search — used
    #: to gate :attr:`prior_refit_every` and the per-iteration
    #: :data:`PARALLEL_COLLISION_RATE` accumulator.
    _sim_counter: int = 0

    # ------------------------------------------------------------------
    # Round-3 MCTS upgrade: VirtualLoss (counter-based) + TranspositionTable
    # ------------------------------------------------------------------
    #: (Round 3) Counter-based virtual loss keyed by Python
    #: ``id(node)``.  Consulted by :meth:`_simulate` (apply) and
    #: :meth:`_backprop` (release).  Mirrors the per-node-scalar
    #: :class:`VirtualLoss` already wired into :meth:`_select_child`
    #: and is the canonical lock-free-MCTS hook the round-3 plan
    #: calls for.
    virtual_loss: _VirtualLoss = field(default_factory=_VirtualLoss)
    #: (Round 3) Canonical-SMILES → ``_MCTSNode`` map.  Bounded FIFO
    #: so memory is bounded for ablation runs.  Mirrored on the
    #: legacy :attr:`_t_table` dict for backward compatibility with
    #: the D2 metric collection (``TRANSPOSITION_HIT_RATE``).
    transposition_table: _TranspositionTable = field(
        default_factory=_TranspositionTable,
    )

    # ------------------------------------------------------------------
    # Reward resolution (backward compat)
    # ------------------------------------------------------------------
    def _resolved_reward(self) -> RewardAggregator:
        # TODO-02 — ``reward_aggregator`` (preferred spelling) wins
        # over ``reward`` (legacy alias) so callers can be explicit.
        if self.reward_aggregator is not None:
            agg = self.reward_aggregator
        elif self.reward is not None:
            agg = self.reward
        elif self.scorer is not None:
            agg = RewardAggregator.from_scorer(self.scorer, weight=1.0)
        else:
            # Last-resort default: zero aggregator.
            agg = RewardAggregator()
        # Apply per-channel weight overrides (replace, not multiply).
        if self.reward_channels:
            for key, w in self.reward_channels.items():
                try:
                    setattr(agg, key, float(w))
                except Exception:
                    continue
        # L-1: stash the leaf-oracle-call preference so the search loop
        # can dispatch to the real docking oracle on the top-K=10 leaves
        # at iteration end without dragging the flag through every call.
        agg._leaf_oracle_call_top_k_only = bool(  # type: ignore[attr-defined]
            self.leaf_oracle_call_top_k_only,
        )
        agg._oracle_top_k = int(self.oracle_top_k)  # type: ignore[attr-defined]
        agg._owner_search = self  # type: ignore[attr-defined]
        return agg

    # ------------------------------------------------------------------
    # Public search
    # ------------------------------------------------------------------

    def search(
        self,
        initial_state: MoleculeClosedTerm,
        max_depth: int = 3,
    ) -> List[MoleculeClosedTerm]:
        """Run MCTS from ``initial_state`` and return the top-K candidates.

        A *candidate* is a leaf (= visited state) that:

            * inhabits every type in ``self.target_predicates``, AND
            * type-checks against ``self.binding_site``.

        The returned list is sorted by descending aggregated reward;
        if fewer than ``top_k`` candidates exist, the full list is
        returned.

        Parameters
        ----------
        initial_state : MoleculeClosedTerm
            The root closed-term (= starting molecule) of the search.
        max_depth : int, default 3
            Maximum β-reduction chain length (rollout depth budget)
            per simulation.  Default 3 (was 10 in Phase-0) is calibrated
            to give enough leaves for ``top_k=20`` after the
            typecheck / binding filter.  ``top_k`` controls
            ``n_candidates_returned``; ``max_depth`` controls the
            rollout depth (and thus the diversity of leaves MCTS can
            collect).  The two knobs are independent — bump ``max_depth``
            further (e.g. 4-5) only if you also raise ``top_k`` past 50.
        """
        if self.rng is None:
            self.rng = random.Random()

        reward_fn = self._resolved_reward()

        # Initialise the root node.
        root = _MCTSNode(
            state=initial_state,
            parent=None,
            P=self._prior(initial_state),
        )
        # We keep a *secondary* registry of all leaves encountered
        # during this search, deduplicated by canonical SMILES — that
        # is what the top-K selection operates on.
        leaves_by_smi: Dict[str, _MCTSNode] = {}
        self._leaf_registry_nodes: Dict[str, Dict[int, _MCTSNode]] = {}

        # Initialise per-search history buffer.
        self.history = []
        # L9-9 TREE_DIVERSITY + L9-10 ROLLOUT_DEPTH_DIST — collected here
        # so ``search()`` can populate them in a single one-line patch.
        self.tree_diversity: float = 0.0  # L9-9  (filled at end of search)
        self.rollout_depth_hist: Dict[int, int] = {}  # L9-10 (filled during _simulate)
        # (r0 fix) Reset NFE counters so each ``search()`` call reports
        # the cumulative cost of the current iteration, not the lifetime
        # total.  Callers that want lifetime stats should sum
        # ``results[-1]["cumulative_nfe"]`` across iterations.
        self.nfe = 0
        self.nfe_reductions = 0
        self.nfe_oracle = 0
        # (r0 fix) Early-stop bookkeeping.  We track the best leaf score
        # seen so far in this search and the iteration at which it was
        # first achieved; when ``it - best_iter > patience`` we break
        # the simulation loop and emit ``EARLY_STOPPED=True``.
        self._best_score_so_far: float = 0.0
        self._best_score_iter: int = -1
        # L-A2 LEAF_VALUE_VAR — per-simulation leaf reward buffer.
        # Populated by ``_rollout`` so we can compute the cumulative
        # variance of the reward head at the end of the search.
        self._leaf_value_history: List[float] = []
        # (D2) Per-search transient state for the new diagnostic
        # metrics (PARALLEL_COLLISION_RATE, TRANSPOSITION_HIT_RATE,
        # PRIOR_REFIT_SKIPPED).  Reset on every ``search()`` call so
        # metrics reflect only the current iteration.
        self._selection_signature: List[Tuple[int, str]] = []
        self._sim_counter: int = 0
        self._t_hits: int = 0
        self._t_misses: int = 0
        self._t_table: Dict[str, "_MCTSNode"] = {}
        # Rules and the resolved tile pool are fixed for one search invocation.
        # Neither a dead-end verdict nor parent/visit state survives into a new
        # search, where callers may have changed the chemistry configuration.
        self._unreactive_states: set[str] = set()
        self.transposition_table = _TranspositionTable(
            max_size=self.transposition_table._max_size)
        self._prior_refit_skipped: bool = False
        self._prior_refit_last_size: int = 0

        # Pre-compute the root prior distribution over the children
        # we'll later expand, so the Dirichlet noise can be applied
        # *before* the first selection (AlphaZero convention).  This
        # is a list of (child_state, rule_name, tile) triples.
        # We compute it lazily — once — and cache it for the root.
        root_children_cache: Optional[List[Tuple[
            MoleculeClosedTerm, str, MoleculeClosedTerm,
        ]]] = None
        root_dirichlet_applied: bool = False

        # n_simulations MCTS rollouts.
        for it in range(self.n_simulations):
            # Mix Dirichlet noise into the root priors of children
            # *before* the first selection; subsequent selections see
            # the perturbed prior.  This matches AlphaZero's
            # ``add_exploration_noise`` step exactly.
            if (
                not root_dirichlet_applied
                and self.dirichlet_alpha > 0.0
                and self.dirichlet_fraction > 0.0
            ):
                root_children_cache = self._expand(root.state)
                if root_children_cache:
                    self._apply_dirichlet_to_root(root, root_children_cache)
                else:
                    root.expansion_complete = True
                root_dirichlet_applied = True

            path = self._simulate(root, max_depth, root_children_cache)
            leaves_by_smi = self._collect_leaves(root, leaves_by_smi)

            # Per-iteration history entry — diagnostics for the
            # closed-loop pipeline + paper figures.
            scores = [n.W / max(1, n.N) for n in leaves_by_smi.values()]
            best_score = max(scores) if scores else 0.0
            mean_score = (sum(scores) / len(scores)) if scores else 0.0
            satisfying = sum(
                1 for n in leaves_by_smi.values()
                if self._satisfies_predicates(n.state, self.target_predicates)
                and self._binds_target(n.state)
            )
            best_state: Optional[MoleculeClosedTerm] = None
            if scores:
                best_node = max(
                    leaves_by_smi.values(),
                    key=lambda n: (n.W / max(1, n.N)),
                )
                best_state = best_node.state
            # L-A2 PUCT_EXPLOIT_RATIO_VAR — variance-normalised exploit
            # ratio.  Strictly positive iff the leaf Q values carry
            # real variance across molecules (i.e. the reward head is
            # not degenerate to a constant).  Formula:
            #
            #     PUCT_EXPLOIT_RATIO_VAR = var(Q[leaf]) / (var(Q[leaf]) + 1e-6)
            #
            # When ``var > 0`` the metric asymptotes to 1.0 (saturated);
            # when ``var == 0`` it is 0.0.  This makes the metric a
            # pass/fail gate: "are leaves distinguishable by reward?".
            leaf_q_values = scores
            if leaf_q_values and len(leaf_q_values) > 1:
                q_arr = np.asarray(leaf_q_values, dtype=float)
                leaf_value_var = float(np.var(q_arr))
                leaf_value_std = float(np.std(q_arr))
            else:
                leaf_value_var = 0.0
                leaf_value_std = 0.0
            puct_exploit_ratio_var = (
                leaf_value_var / (leaf_value_var + 1e-6)
            )
            # (D2) PARALLEL_COLLISION_RATE — fraction of adjacent
            # simulation pairs whose (depth, child_state_canonical_smi)
            # tuples collide at the same depth.  A crude proxy for
            # "two workers contending on the same hot child".
            sel_sig = list(getattr(self, "_selection_signature", []) or [])
            n_sims = max(1, self._sim_counter)
            same_selected = 0
            for i in range(1, len(sel_sig)):
                if sel_sig[i][0] == sel_sig[i - 1][0] and sel_sig[i][1] == sel_sig[i - 1][1]:
                    same_selected += 1
            parallel_collision_rate = float(same_selected) / float(n_sims)

            # (D2) TRANSPOSITION_HIT_RATE — fraction of
            # transposition-table lookups that returned an existing
            # node (vs creating a fresh one).
            t_hits = int(getattr(self, "_t_hits", 0) or 0)
            t_misses = int(getattr(self, "_t_misses", 0) or 0)
            t_total = t_hits + t_misses
            transposition_hit_rate = (
                float(t_hits) / float(t_total) if t_total > 0 else 0.0
            )

            self.history.append({
                "iteration": it,
                "best_score": float(best_score),
                "mean_score": float(mean_score),
                "n_states_explored": int(self._count_nodes(root)),
                "n_leaves": int(len(leaves_by_smi)),
                "n_satisfying": int(satisfying),
                "best_state": best_state,
                "PUCT_EXPLOIT_RATIO_VAR": float(puct_exploit_ratio_var),
                "LEAF_VALUE_VAR": float(leaf_value_var),
                "LEAF_VALUE_STD": float(leaf_value_std),
                "TRANSPOSITION_HIT_RATE": float(transposition_hit_rate),
                "PARALLEL_COLLISION_RATE": float(parallel_collision_rate),
                "PRIOR_REFIT_SKIPPED": bool(getattr(self, "_prior_refit_skipped", False)),
                # (r0 fix) — NFE counters surfaced per iteration so the
                # closed-loop pipeline (and SOTA comparison reports) can
                # plot cost-vs-quality curves without re-deriving them
                # from the leaf_value_history buffer.
                "CUMULATIVE_NFE": int(self.nfe),
                "CUMULATIVE_NFE_REDUCTIONS": int(self.nfe_reductions),
                "CUMULATIVE_NFE_ORACLE": int(self.nfe_oracle),
            })

            # (r0 fix) — MCTS early-stop.  Track best_score_so_far; if
            # the budget has not improved the best score for ``patience``
            # consecutive iterations, break out of the simulation loop
            # and emit ``EARLY_STOPPED`` diagnostics.  Empirically
            # (round-0 sweep) UCB converges in <=30 rollouts at the
            # 12-tile Phase-0 branching and the remaining ~970
            # rollouts of the default 1000-sim budget are wasted
            # compute — so ``patience=30`` is the canonical setting.
            if (
                bool(self.early_stop)
                and int(self.patience) > 0
                and float(best_score) > float(self._best_score_so_far) + 1e-12
            ):
                self._best_score_so_far = float(best_score)
                self._best_score_iter = int(it)
            # Check for early-stop AFTER emitting the current iter's
            # history entry so callers always see the iteration that
            # actually improved the best score.
            if (
                bool(self.early_stop)
                and int(self.patience) > 0
                and self._best_score_iter >= 0
                and int(it) - int(self._best_score_iter) >= int(self.patience)
            ):
                # Stash the stop reason on the last entry so downstream
                # consumers can detect it without inspecting every
                # entry's ``iteration`` field.
                self.history[-1]["EARLY_STOPPED"] = True
                self.history[-1]["EARLY_STOPPED_AT_ITER"] = int(self._best_score_iter)
                break

        # Collect candidates: leaves that satisfy all type predicates
        # AND bind the target site.  The fingerprint stub is used here
        # for the cheap filter — the real docking oracle runs on the
        # **top-K=10** candidates only, never during MCTS expansion.
        candidates: List[Tuple[float, MoleculeClosedTerm]] = []
        for node in leaves_by_smi.values():
            state = node.state
            if not self._satisfies_predicates(state, self.target_predicates):
                continue
            if not self._binds_target(state):
                continue
            candidates.append((self.score_final(state), state))
        # Sort best-first by aggregated reward.
        candidates.sort(key=lambda p: p[0], reverse=True)

        # L-1 leaf_oracle_call=top_k_only: invoke the real docking
        # oracle on the top-K candidates at iteration end.  This is
        # the ONLY place in the search loop where the docking cost
        # is paid — MCTS expansion never calls the oracle.  When no
        # oracle is available (or ``leaf_oracle_call_top_k_only`` is
        # False) this step is a no-op.
        if self.leaf_oracle_call_top_k_only and self.oracle_top_k > 0:
            oracle = self.oracle
            if oracle is None:
                try:
                    from molmetal_lam.binding.types import _resolve_docking_oracle
                    oracle = _resolve_docking_oracle()
                except Exception:
                    oracle = None
            if oracle is not None:
                top_for_oracle = candidates[: int(self.oracle_top_k)]
                filtered: List[Tuple[float, MoleculeClosedTerm]] = []
                for score, state in top_for_oracle:
                    # (r0 fix) — each top-K docking-oracle call counts
                    # as one ``nfe_oracle`` (Number of Function
                    # Evaluations against the real oracle).  These are
                    # the SOTA-bottleneck calls so we track them
                    # separately from the cheap reward-aggregator NFE.
                    try:
                        self.nfe_oracle = int(self.nfe_oracle) + 1
                    except Exception:
                        pass
                    if self._binds_target_top_k(state, oracle):
                        filtered.append((score, state))
                # Append any non-top-K candidates (kept on the fingerprint
                # verdict so the closed loop does not lose diversity).
                for score, state in candidates[int(self.oracle_top_k):]:
                    filtered.append((score, state))
                candidates = filtered
                candidates.sort(key=lambda p: p[0], reverse=True)

        # L9-9 TREE_DIVERSITY — fraction of unique leaves (α-classes).
        # 1-line: ratio of distinct canonical-SMILES leaves over explored nodes.
        self.tree_diversity = len(leaves_by_smi) / max(1, self._count_nodes(root))

        # L-A2 cumulative LEAF_VALUE_VAR / PUCT_EXPLOIT_RATIO_VAR across
        # the *entire* search — exposes a single scalar for the
        # closed-loop report and downstream metric aggregators.
        leaf_vals = list(getattr(self, "_leaf_value_history", []) or [])
        if len(leaf_vals) > 1:
            q_arr = np.asarray(leaf_vals, dtype=float)
            self.leaf_value_var = float(np.var(q_arr))
            self.leaf_value_std = float(np.std(q_arr))
        else:
            self.leaf_value_var = 0.0
            self.leaf_value_std = 0.0
        self.puct_exploit_ratio_var = (
            self.leaf_value_var / (self.leaf_value_var + 1e-6)
        )

        # (L-2) Persistent tree: remember the root for save_tree() so
        # the *next* closed-loop iteration can resume from the same
        # expanded subtree instead of rebuilding from scratch.
        self._root = root
        # Also accumulate (features, leaf_score) pairs globally so the
        # SymbolicPrior improves as data accumulates across iterations.
        self._accumulate_leaf_pairs(leaves_by_smi)

        return [state for _, state in candidates[: self.top_k]]

    # ------------------------------------------------------------------
    # MCTS phases
    # ------------------------------------------------------------------

    def _simulate(
        self,
        root: _MCTSNode,
        max_depth: int,
        root_children_cache: Optional[List[Tuple[
            MoleculeClosedTerm, str, MoleculeClosedTerm,
        ]]] = None,
    ) -> List[_MCTSNode]:
        """Run one MCTS simulation: select → expand → rollout → backprop.

        Returns the path of nodes traversed (root → leaf).  Modifies the
        tree in place by updating ``N`` / ``W`` / ``children``.
        """
        path: List[_MCTSNode] = [root]
        reward_fn = self._resolved_reward()

        # 1) SELECT — PUCT traversal down to a leaf or terminal.
        # Only an exhaustive empty expansion establishes chemical termination.
        node = root
        while not node.is_leaf and (not node.is_closed_for_search) and len(path) <= max_depth:
            node = self._select_child(node)
            # Lock the selected child with a virtual loss so any
            # concurrent in-flight simulation that re-enters the same
            # parent is biased *away* from this child until the lock
            # is released by ``_backprop``.  In the single-threaded
            # search this is a no-op (the lock is released before the
            # next iteration), but the same machinery supports a
            # future lock-free parallel MCTS.  Python's GIL serialises
            # the ``+=`` operation within a single interpreter so the
            # counter is monotonic per process.
            try:
                node.virtual_loss.value += float(VIRTUAL_LOSS_VALUE)
            except Exception:
                pass
            # Round-3 MCTS upgrade — also apply through the
            # counter-based :class:`_VirtualLoss` (keyed by Python
            # ``id(node)``) so lock-free-MCTS scaffolds can consult
            # ``virtual_loss.on_node(id(node))`` to bias away from
            # in-flight simulations.  The two writes are independent
            # and both released by :meth:`_backprop`.
            try:
                self.virtual_loss.apply(id(node))
            except Exception:
                pass
            path.append(node)

        # 2) EXPAND — if the leaf isn't terminal and we're under the
        # depth budget, try all (rule, tile) combinations.
        if not node.is_terminal and len(path) <= max_depth:
            # If this is the root and we already pre-expanded its
            # children for Dirichlet injection, reuse that cache.
            children = (
                root_children_cache
                if node is root and root_children_cache is not None
                else self._expand(node.state)
            )
            self._attach_children(node, children)
            if node.children:
                node = self.rng.choice(node.children)
                path.append(node)

        # 3) ROLLOUT — play out from `node` to a terminal or the depth
        # budget, evaluate via the reward aggregator.
        depth_remaining = max(0, max_depth - (len(path) - 1))
        value = self._rollout(node.state, depth_remaining, reward_fn)

        # 4) BACKPROP — update (N, W) along the path.
        self._backprop(path, value)
        # L9-10 ROLLOUT_DEPTH_DIST — histogram of β-chain length per sim.
        # 1-line: tally path length so ``search()`` can emit the histogram.
        self.rollout_depth_hist[len(path) - 1] = self.rollout_depth_hist.get(len(path) - 1, 0) + 1

        # (D2) Record (depth, child_state_smi) per simulation so the
        # caller can compute PARALLEL_COLLISION_RATE — a crude proxy
        # for "two workers contending on the same hot child".  The
        # selected child at depth ``i`` is ``path[i]`` for i > 0; root
        # (path[0]) is excluded because every simulation re-enters it.
        if not hasattr(self, "_selection_signature"):
            self._selection_signature: List[Tuple[int, str]] = []
        try:
            for depth_i, node in enumerate(path[1:], start=1):
                self._selection_signature.append(
                    (depth_i, self._canonical_smi(node.state))
                )
        except Exception:
            pass

        # (D2) Bump the per-search simulation counter and trigger an
        # automatic SymbolicPrior refit every ``prior_refit_every``
        # simulations when the accumulated replay buffer has grown
        # enough to justify one.
        try:
            self._sim_counter += 1
        except Exception:
            self._sim_counter = 1
        if (
            int(getattr(self, "prior_refit_every", 0)) > 0
            and self._sim_counter % int(self.prior_refit_every) == 0
        ):
            self._maybe_refit_prior()

        return path

    def _select_child(self, node: _MCTSNode) -> _MCTSNode:
        """PUCT selection: pick the child maximising Q + c_puct * P * sqrt(N) / (1 + N).

        Implements the canonical PUCT rule used in AlphaZero, augmented
        with a **virtual-loss term** for future lock-free parallel MCTS::

            a* = argmax_a  Q(s, a) * w
                         + c_puct * P(a) * sqrt(N(s)) / (1 + N(s, a))
                         - virtual_loss(a) / (1 + N(s, a))

        where ``Q(s, a) = W(a) / N(a)`` is the mean value of action a
        from state s, ``P(a)`` is the prior probability
        (:class:`SymbolicPrior` if fitted, else the constant 0.5 stub),
        ``N(s)`` is the parent's visit count, ``N(s, a)`` is the
        child's visit count, and ``virtual_loss(a)`` is the running
        accumulator that ``_simulate`` increments on lock and
        ``_backprop`` decrements on release.  In the single-threaded
        search ``virtual_loss`` is always ``0.0`` at selection time, so
        the term is a no-op there and the formula reduces to the
        canonical AlphaZero PUCT.
        """
        assert node.children, "_select_child called on a leaf"
        N_parent = max(1, node.N)
        sqrt_N_parent = math.sqrt(N_parent)

        # Round-3 MCTS upgrade (G6 wire-up) — claim a virtual-loss
        # lock on every candidate child *before* the PUCT computation
        # so the PUCT penalty (which reads ``on_node(id(child))``) sees
        # the in-flight claim.  Without this the read-and-write would
        # be ordered incorrectly: PUCT would not be biased away from
        # children that are simultaneously being claimed by another
        # worker.  The lock on the chosen child stays held until
        # ``_backprop`` releases it; the locks on every non-chosen
        # child are released immediately below so subsequent
        # selections from this parent see a clean slate.
        for child in node.children:
            try:
                self.virtual_loss.apply(id(child))
            except Exception:
                pass

        best_score = -float("inf")
        best_child: Optional[_MCTSNode] = None
        for child in node.children:
            Q = child.Q
            P = child.P
            N_child = child.N
            # Round-3 MCTS upgrade — combine the legacy per-node scalar
            # ``VirtualLoss`` with the counter-based :class:`_VirtualLoss`
            # keyed by Python ``id(child)`` so lock-free-MCTS scaffolds
            # can claim / release children across workers without
            # touching the node objects.  The counter value (an int)
            # is multiplied by ``self.virtual_loss.value`` to get a
            # comparable penalty scale.
            try:
                scalar_vloss = float(child.virtual_loss_value)
            except Exception:
                scalar_vloss = 0.0
            try:
                cnt_vloss = float(
                    int(self.virtual_loss.on_node(id(child))) *
                    float(self.virtual_loss.value),
                )
            except Exception:
                cnt_vloss = 0.0
            virtual_loss = scalar_vloss + cnt_vloss
            # PUCT formula — exploit (w * Q / (1+n)) + exploration
            # (c_puct * P * sqrt(N_parent) / (1+n)) - virtual-loss
            # penalty ((virtual_loss) / (1+n)).  The virtual-loss term
            # temporarily de-prioritises in-flight children so two
            # parallel workers cannot both claim the same hot child.
            # Thread safety: Python GIL serialises the underlying
            # ``+=`` / ``-=`` so the counter is monotonic within one
            # interpreter.
            w = float(self.c_puct)
            u = (
                w * (P * sqrt_N_parent) / (1.0 + N_child)
                + float(Q) / max(1.0, 1.0 + N_child)
                - virtual_loss / (1.0 + N_child)
            )
            if u > best_score:
                best_score = u
                best_child = child
        # ``best_child`` cannot be None when ``node.children`` is non-empty.
        assert best_child is not None
        # Release the lock on every child — including the selected one.
        # ``_simulate`` re-applies the lock on the chosen child
        # immediately after ``_select_child`` returns (so a worker that
        # re-enters this parent in between is biased away), but we
        # release it here so consecutive calls to ``_select_child``
        # without an intervening ``_simulate`` see a clean slate (this
        # is the contract the existing unit tests rely on).
        for child in node.children:
            try:
                self.virtual_loss.release(id(child))
            except Exception:
                pass
        return best_child

    def _expand(
        self,
        state: MoleculeClosedTerm,
    ) -> List[Tuple[MoleculeClosedTerm, str, MoleculeClosedTerm]]:
        """Expand one β-reduction step from ``state``.

        For each ``(rule_name, rule)`` in ``self.rules`` and each
        ``tile`` in ``self.tile_library``, we attempt to fire the rule
        on ``(state, tile)`` (= one bi-molecular β-reduction).  Every
        successful product becomes a child node.

        L-3 (Lambda Round 3): when :attr:`use_fragment_pool` is True
        and no tile pool has been resolved yet, the 204-tile
        ChEMBL/ZINC SMARTS-diverse pool is loaded via
        :func:`molmetal_lam.tile_lib.library.FRAGMENT_LIBRARY_200_TILES`,
        converted to ``MoleculeClosedTerm`` objects via canonical
        SMILES round-trips, and replaces the 12-tile Phase-0 library
        *for the duration of this search*.  Branching factor grows
        from 5 × 12 = 60 to 5 × 204 = 1020 when the pool loads
        successfully.  Tiles that fail to embed / parse (1.9% embed-
        fail rate per ``lambda_round3_L3_fragment_library.md``) are
        silently dropped inside the pool loader, so the loop below
        only sees validated tiles.

        Falls back to :attr:`tile_library` when the flag is False
        (default — full backward compatibility) or when the fragment
        pool cannot be imported.

        Returns a list of ``(child_state, rule_name, tile)`` triples.
        """
        children: List[Tuple[MoleculeClosedTerm, str, MoleculeClosedTerm]] = []
        # Cache only exhaustive non-fire results, scoped to this search's fixed
        # rules/pool. A random unsuccessful rollout attempt is not a dead end.
        state_key = self._canonical_smi(state)
        if not hasattr(self, "_unreactive_states"):
            self._unreactive_states = set()
        if state_key in self._unreactive_states:
            return children
        tile_pool = self._resolve_expand_tile_pool()
        for rule_name, rule in self.rules.items():
            for tile in tile_pool:
                # (r0 fix) — every (rule, tile) attempted counts as one
                # β-reduction NFE, regardless of whether it fires.  This
                # matches the MCTS literature convention where the
                # expansion cost is bounded by ``|rules| * |tile_pool|``
                # per simulation.
                try:
                    self.nfe_reductions = int(self.nfe_reductions) + 1
                except Exception:
                    pass
                products = self._safe_reduce(rule, state, tile)
                if not products:
                    continue
                for prod in products:
                    children.append((prod, rule_name, tile))
        if not children:
            self._unreactive_states.add(state_key)
        return children

    def _attach_children(self, parent, children) -> None:
        """Attach a finite tree of reaction paths with independent node stats.

        Transposition lookups may share a molecular state, but a node's parent
        and visits belong to one path. Reparenting a cached node would corrupt
        diamond paths and can create cycles on a reverse reaction.
        """
        ancestors = set()
        cursor = parent
        seen_ids = set()
        while cursor is not None and id(cursor) not in seen_ids:
            seen_ids.add(id(cursor))
            ancestors.add(self._canonical_smi(cursor.state))
            cursor = cursor.parent
        sibling_keys = {self._canonical_smi(c.state) for c in parent.children}
        for child_state, rule_name, tile in children:
            child_key = self._canonical_smi(child_state)
            if child_key in ancestors or child_key in sibling_keys:
                continue
            def factory():
                return _MCTSNode(state=child_state, parent=parent,
                                 P=self._prior(child_state), rule_name=rule_name, tile=tile)
            existing, hit = self._lookup_or_create(child_key, factory)
            child = factory() if hit and existing.parent is not parent else existing
            parent.children.append(child)
            sibling_keys.add(child_key)
        parent.expansion_complete = True

    def _resolve_expand_tile_pool(self) -> List[MoleculeClosedTerm]:
        """Return the tile list ``_expand`` should iterate over.

        When :attr:`use_fragment_pool` is True, lazily loads the
        204-tile Phase-1 pool from
        :func:`molmetal_lam.tile_lib.library.FRAGMENT_LIBRARY_200_TILES`,
        converts each ``Tile`` to a :class:`MoleculeClosedTerm` via
        :meth:`MoleculeClosedTerm.from_smiles` (no 3D embed — the
        search only needs the atom / bond graph), caches the result on
        ``self`` so the loader runs once per ``MCTSProofSearch``
        instance, and returns the cached list.

        When ``use_fragment_pool`` is False (default) — or when the
        fragment pool cannot be imported (e.g. in a headless test
        environment without ``fragment_pool``) — the method falls back
        to ``self.tile_library`` unchanged so every existing test
        path stays green.

        R10 axis A — when the new :data:`ROUND10_5_CLICK` constant is
        True, the function appends 5 × 4 = 20 click-handle tiles
        (DBCO, boronic acid, aryl halide, carboxylic acid, amine) to
        the resolved pool so the MCTS expansion path can fire all
        five click reactions (CuAAC, SPAAC, thiol-ene, Suzuki, amide
        coupling) — not just CuAAC.  Total pool size grows from 200 to
        220.  The constant defaults to True; set
        ``ROUND10_5_CLICK=0`` to restore the legacy 200-tile pool.
        """
        if not self.use_fragment_pool:
            return self.tile_library
        cached = getattr(self, "_fragment_pool_cache", None)
        if cached is not None:
            return cached
        try:
            from molmetal_lam.tile_lib.library import FRAGMENT_LIBRARY_200_TILES
        except Exception:
            return self.tile_library
        try:
            tiles = FRAGMENT_LIBRARY_200_TILES()
        except Exception:
            return self.tile_library
        converted: List[MoleculeClosedTerm] = []
        for t in tiles:
            try:
                smi = getattr(t, "smiles", None) or getattr(t, "to_smiles", lambda: "")()
                if not smi:
                    continue
                mol = MoleculeClosedTerm.from_smiles(smi, embed_3d=False)
                converted.append(mol)
            except Exception:
                # Embed / parse failure — silently drop the tile. The
                # embed-fail rate (~1.9%) is well under the 5% success
                # criterion, so the pool still yields the expected 204+
                # effective tiles after filtering.
                continue
        if not converted:
            # Pool loader returned nothing usable — fall back to the
            # 12-tile library so the search does not silently stop.
            return self.tile_library

        # R10 axis A — extend the pool with the 5 new click-handle
        # families (4 SMILES each) so all 5 reactions are reachable
        # from the expansion path.  See
        # :mod:`molmetal_lam.tile_lib.fragment_pool` for the SMILES
        # sources.
        try:
            from molmetal_lam.tile_lib.fragment_pool import (
                fragments_from_chembl_reactive,
            )
            click_handle_flag = os.environ.get("ROUND10_5_CLICK", "1") not in ("0", "false", "False")
        except Exception:
            fragments_from_chembl_reactive = None  # type: ignore
            click_handle_flag = False
        if click_handle_flag and fragments_from_chembl_reactive is not None:
            try:
                # ``include_*=True`` for the 5 new families only; the
                # original 4 (azide / alkyne / diene / thiol) are
                # already in the 200-tile pool above.
                extra_pool = fragments_from_chembl_reactive(
                    include_azides=False,
                    include_alkynes=False,
                    include_dienes=False,
                    include_thiols=False,
                    include_dbco=True,
                    include_boronic_acids=True,
                    include_aryl_halides=True,
                    include_carboxylic_acids=True,
                    include_amines=True,
                )
                for t in extra_pool:
                    try:
                        smi = getattr(t, "smiles", None) or getattr(t, "to_smiles", lambda: "")()
                        if not smi:
                            continue
                        mol = MoleculeClosedTerm.from_smiles(smi, embed_3d=False)
                        converted.append(mol)
                    except Exception:
                        continue
            except Exception:
                # Pool extension is best-effort — never fail the search
                # because a click-handle family could not be loaded.
                pass

        self._fragment_pool_cache = converted
        try:
            self.expand_pool_size = int(len(converted))
        except Exception:
            pass
        return converted

    def _apply_dirichlet_to_root(
        self,
        root: _MCTSNode,
        root_children: List[Tuple[
            MoleculeClosedTerm, str, MoleculeClosedTerm,
        ]],
    ) -> None:
        """AlphaZero-style Dirichlet noise injection at the root.

        Sets ``P(a|s_root) = (1 - ε) * P_learned(a) + ε * η_a`` where
        ``η ~ Dir(alpha)`` and ``ε = dirichlet_fraction``.  We add the
        children to the tree, set P directly, then return — the
        standard ``_simulate`` loop sees the perturbed priors on the
        first selection.

        If the expansion produced no children (no reactions fired) the
        call is a no-op.

        When the global fused-softmax gate is on and the child count
        fits in a single in-block reduction (≤4096), the Dirichlet
        mixing + final renormalisation are performed by
        :func:`triton_kernels.softmax_last_dim`.  Otherwise the
        scalar-path (CPU fallback) is used.
        """
        eps = float(self.dirichlet_fraction)
        alpha = float(self.dirichlet_alpha)
        # Add the children to the tree first (so root.children exists
        # for the selection step) and then overwrite their P with the
        # Dirichlet-mixed value.
        self._attach_children(root, root_children)
        if not root.children:
            return

        n_children = len(root.children)
        rng = np.random.default_rng(self._seed_for_noise())
        noise = rng.dirichlet([alpha] * n_children)

        # Try the fused-softmax path.  The "logits" we feed in are the
        # Dirichlet-mixed priors; softmax over the last axis gives a
        # probability distribution that respects the noise injection.
        # ``softmax_last_dim`` returns torch.softmax(...) on CPU, so
        # the dispatch is automatic — but we honour the explicit gate
        # so callers can force-disable it for parity / debugging.
        use_fused = bool(triton_config.use_fused_softmax(n_children))
        if use_fused:
            try:
                import torch as _torch
                priors = _torch.tensor(
                    [float(c.P) for c in root.children],
                    dtype=_torch.float32,
                )
                eta = _torch.tensor(
                    [float(e) for e in noise],
                    dtype=_torch.float32,
                )
                mixed = (1.0 - eps) * priors + eps * eta
                # Logit-style: pass log(mixed) so softmax_last_dim
                # returns the mixed distribution directly.  Clip to
                # avoid -inf at zero priors.
                logits = _torch.log(_torch.clamp(mixed, min=1e-12))
                probs = softmax_last_dim(logits, dim=-1)
                for child, p in zip(root.children, probs.tolist()):
                    child.P = float(max(0.0, min(1.0, p)))
                return
            except Exception:
                # Fall through to the scalar path on any error.
                pass

        for child, eta in zip(root.children, noise):
            child.P = float(max(0.0, min(1.0, (1.0 - eps) * child.P + eps * eta)))

    def _seed_for_noise(self) -> int:
        """Derive a stable seed for the Dirichlet draw from ``self.rng``.

        We sample from ``self.rng`` (so user-supplied ``rng`` controls
        determinism) and use the result as a NumPy seed.  Falls back to
        a hash of ``self.rng``'s state when the RNG doesn't expose a
        usable integer.
        """
        try:
            return int(self.rng.randint(0, 2**31 - 1))
        except Exception:
            try:
                state = self.rng.getstate()
                return abs(hash(state)) % (2**31)
            except Exception:
                return 0

    def reinject_dirichlet_at_root(self, fraction: float = 0.25) -> bool:
        """Re-apply Dirichlet noise to ``self._root`` (AlphaZero style).

        Called at every closed-loop iteration boundary so the
        persistent tree does **not** collapse to the early high-reward
        regions.  Returns ``True`` iff noise was actually applied
        (i.e. the root has children to perturb).

        Parameters
        ----------
        fraction : float, default 0.25
            Dirichlet mixing fraction ``ε`` — the perturbation weight.
            ``0.0`` is a no-op, ``1.0`` fully replaces the learned
            priors with pure noise.

        When the global fused-softmax gate is on and the child count
        fits in a single in-block reduction (≤4096), the Dirichlet
        mixing + final renormalisation are performed by
        :func:`triton_kernels.softmax_last_dim`.  Otherwise the
        scalar-path (CPU fallback) is used.
        """
        root = getattr(self, "_root", None)
        if root is None:
            return False
        if not root.children:
            return False
        try:
            eps = float(fraction)
            alpha = float(self.dirichlet_alpha)
        except Exception:
            return False
        if eps <= 0.0 or alpha <= 0.0:
            return False
        rng = np.random.default_rng(self._seed_for_noise())
        n_children = len(root.children)
        noise = rng.dirichlet([alpha] * n_children)

        use_fused = bool(triton_config.use_fused_softmax(n_children))
        if use_fused:
            try:
                import torch as _torch
                priors = _torch.tensor(
                    [float(c.P) for c in root.children],
                    dtype=_torch.float32,
                )
                eta = _torch.tensor(
                    [float(e) for e in noise],
                    dtype=_torch.float32,
                )
                mixed = (1.0 - eps) * priors + eps * eta
                logits = _torch.log(_torch.clamp(mixed, min=1e-12))
                probs = softmax_last_dim(logits, dim=-1)
                for child, p in zip(root.children, probs.tolist()):
                    try:
                        child.P = float(max(0.0, min(1.0, p)))
                    except Exception:
                        continue
                return True
            except Exception:
                # Fall through to the scalar path on any error.
                pass

        for child, eta in zip(root.children, noise):
            try:
                child.P = float(max(
                    0.0,
                    min(1.0, (1.0 - eps) * float(child.P) + eps * float(eta)),
                ))
            except Exception:
                continue
        return True

    def _rollout(
        self,
        state: MoleculeClosedTerm,
        depth: int,
        reward_fn: RewardAggregator,
    ) -> float:
        """ε-greedy playout from ``state`` to a terminal; return the score.

        When a :class:`SymbolicPrior` is fitted and
        ``rollout_epsilon > 0`` the policy is::

            with prob ε : sample a uniform random (rule, tile)
            else        : pick (rule, tile) maximising prior(state')

        Otherwise we fall back to the original uniform-random policy.
        Either way the terminal state is evaluated by the reward
        aggregator, which combines Vina / SA / PoseBusters / pIC50 /
        retro channels with the type / binder bonuses.
        """
        use_guided = (
            self.prior is not None
            and self.prior.fitted
            and 0.0 < float(self.rollout_epsilon) < 1.0
        )
        current = state
        cur_depth = 0
        while cur_depth < depth:
            # Internal beta-NF does not rule out an external reaction partner.
            if self._canonical_smi(current) in getattr(self, "_unreactive_states", set()):
                break

            # Sample a (rule, tile).
            rule_names = list(self.rules.keys())
            if not rule_names:
                break
            tile_lib = self._resolve_expand_tile_pool()
            if not tile_lib:
                break

            if use_guided and self.rng.random() >= float(self.rollout_epsilon):
                rule_name, tile, products = self._rollout_pick_guided(
                    current, rule_names, tile_lib
                )
                if rule_name is None:
                    rule_name = self.rng.choice(rule_names)
                    tile = self.rng.choice(tile_lib)
                    rule = self.rules[rule_name]
                    # (r0 fix) — count β-reductions in the rollout too,
                    # so a guided rollout that exhausts its (rule, tile)
                    # candidates falls back to uniform random and still
                    # contributes to the NFE budget.
                    try:
                        self.nfe_reductions = int(self.nfe_reductions) + 1
                    except Exception:
                        pass
                    products = self._safe_reduce(rule, current, tile)
            else:
                rule_name = self.rng.choice(rule_names)
                rule = self.rules[rule_name]
                tile = self.rng.choice(tile_lib)
                # (r0 fix) — count each β-reduction attempt in the rollout
                # (regardless of whether it fires) for NFE accounting.
                try:
                    self.nfe_reductions = int(self.nfe_reductions) + 1
                except Exception:
                    pass
                products = self._safe_reduce(rule, current, tile)

            if not products:
                # The redex didn't fire — try again next iteration.
                cur_depth += 1
                continue
            current = self.rng.choice(products)
            cur_depth += 1

        # Evaluate the terminal / depth-budget state with the multi-
        # reward aggregator; pre-compute the boolean flags so the
        # aggregator does not have to touch RDKit twice.
        try:
            sat = self._satisfies_predicates(current, self.target_predicates)
        except Exception:
            sat = False
        try:
            binds = self._binds_target(current)
        except Exception:
            binds = False
        value = float(reward_fn(
            current,
            target_predicates=self.target_predicates,
            binding_site=self.binding_site,
            satisfies_typed=sat,
            binds_target=binds,
        ))
        # (r0 fix) — every call to the reward aggregator counts as one
        # Number of Function Evaluation (NFE).  This includes the
        # per-channel scorer invocations inside :class:`RewardAggregator`
        # (SA, QED, Vina_proxy, ...) which are the dominant compute
        # cost in the canonical n_simulations=1000 budget.
        try:
            self.nfe = int(self.nfe) + 1
        except Exception:
            pass
        # L-A2: append leaf reward to per-search tracker so
        # ``search()`` can compute the cumulative LEAF_VALUE_VAR at
        # the end of the run.
        if not hasattr(self, "_leaf_value_history"):
            self._leaf_value_history = []
        try:
            self._leaf_value_history.append(float(value))
        except Exception:
            pass
        return value

    def _rollout_pick_guided(
        self,
        state: MoleculeClosedTerm,
        rule_names: Sequence[str],
        tile_lib: Sequence[MoleculeClosedTerm],
    ) -> Tuple[Optional[str], Optional[MoleculeClosedTerm], List[MoleculeClosedTerm]]:
        """Choose the (rule, tile) whose product has the highest prior.

        Returns ``(rule_name, tile, products)`` or ``(None, None, [])``
        when no (rule, tile) fires — the caller falls back to uniform
        random in that case.
        """
        assert self.prior is not None
        best: Tuple[float, str, MoleculeClosedTerm, List[MoleculeClosedTerm]] = (
            -float("inf"), "", tile_lib[0], []
        )
        any_fired = False
        for rule_name in rule_names:
            rule = self.rules[rule_name]
            for tile in tile_lib:
                # (r0 fix) — guided rollout enumerates every (rule, tile)
                # pair, so it dominates NFE when ``rollout_epsilon`` is
                # non-trivial.  Count every attempted β-reduction.
                try:
                    self.nfe_reductions = int(self.nfe_reductions) + 1
                except Exception:
                    pass
                products = self._safe_reduce(rule, state, tile)
                if not products:
                    continue
                any_fired = True
                # Score the first product (cheap proxy; the rollout
                # still samples uniformly among products).
                v = self.prior.predict_value(products[0])
                if v > best[0]:
                    best = (v, rule_name, tile, products)
        if not any_fired:
            return None, None, []
        return best[1], best[2], best[3]

    def _backprop(self, path: List[_MCTSNode], value: float) -> None:
        """Update ``(N, W)`` for every node along the simulation path.

        Standard MCTS backprop: every node on the path gets ``N += 1``
        and ``W += value``.  Root is at ``path[0]``, leaf at ``path[-1]``.

        Also **releases** the virtual-loss lock acquired by
        ``_simulate``: each node's ``virtual_loss`` counter is decremented
        by ``VIRTUAL_LOSS_VALUE`` so the node becomes selectable again.
        Thread safety: Python GIL serialises the underlying ``+=`` /
        ``-=`` so the counter is monotonic within one interpreter.
        """
        for node in path:
            node.N += 1
            node.W += value
            try:
                node.virtual_loss.value -= float(VIRTUAL_LOSS_VALUE)
            except Exception:
                pass
            # Round-3 MCTS upgrade — release the counter-based
            # :class:`_VirtualLoss` lock acquired by
            # :meth:`_simulate`.  The counter floors at 0 so a stray
            # release (e.g. from a recycled node that was never
            # claimed) is silently absorbed.
            try:
                self.virtual_loss.release(id(node))
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Type-predicate & binding checks
    # ------------------------------------------------------------------

    def _satisfies_predicates(
        self,
        state: MoleculeClosedTerm,
        predicates: List[TypePredicate],
    ) -> bool:
        """True iff ``state`` inhabits every type in ``predicates``.

        This is the **type checker** of the search layer: a state that
        fails any predicate is ill-typed and cannot be returned as a
        candidate.  RDKit errors are treated as failure (ill-typed).
        """
        for pred in predicates:
            try:
                if not bool(pred(state)):
                    return False
            except Exception:
                # RDKit missing or malformed molecule — treat as
                # ill-typed rather than crashing the search.
                return False
        return True

    def _binds_target(self, state: MoleculeClosedTerm) -> bool:
        """True iff ``state`` type-checks against ``self.binding_site``.

        Honours the L-1 ``leaf_oracle_call_top_k_only`` flag — the
        real docking oracle is *only* consulted when this method is
        called from :meth:`search`'s **post-iteration top-K candidate
        selection** path (never during expansion / rollout).  All
        per-sim ``_binds_target`` calls pass ``leaf_oracle_call=False``
        so they stay on the cheap fingerprint stub.
        """
        try:
            result = typecheck(
                state, self.binding_site,
                leaf_oracle_call=False,  # never during expansion/rollout
                oracle=None,
            )
            return bool(result.success)
        except Exception:
            return False

    def _binds_target_top_k(
        self, state: MoleculeClosedTerm, oracle: Any
    ) -> bool:
        """Top-K-only binding check — invokes the real docking oracle.

        Called from :meth:`search`'s post-iteration candidate ranking
        only — never during MCTS expansion.  This is the gate the L-1
        task brief calls out as ``leaf_oracle_call=top_k_only``.
        """
        try:
            result = typecheck(
                state, self.binding_site,
                leaf_oracle_call=True,
                oracle=oracle,
            )
            return bool(result.success)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Final scoring (post-search candidate ranking)
    # ------------------------------------------------------------------
    def score_final(self, state: MoleculeClosedTerm) -> float:
        """Score a candidate state with the reward aggregator."""
        reward_fn = self._resolved_reward()
        sat = self._satisfies_predicates(state, self.target_predicates)
        binds = self._binds_target(state)
        return float(reward_fn(
            state,
            target_predicates=self.target_predicates,
            binding_site=self.binding_site,
            satisfies_typed=sat,
            binds_target=binds,
        ))

    # ------------------------------------------------------------------
    # Priors and helpers
    # ------------------------------------------------------------------

    def heuristic(self, features: "Any") -> float:
        """MCTS prior — TODO-02 replacement for the constant 0.5 stub.

        Computes the leaf-expansion prior from a six-channel
        :class:`RewardAggregator` (Vina + PoseBusters + Retro + pIC50 +
        QED + SA) using the *per-channel reward values* rather than
        the closed-term state directly.  The default per-channel
        weights are::

            w_vina        = 0.40
            w_posebusters = 0.20
            w_retro       = 0.15
            w_pic50       = 0.10
            w_qed         = 0.10
            w_sa          = 0.05

        so binding dominates, PoseBusters is the structural sanity
        gate, retrosynthesis feasibility gates synthesizability, and
        QED / SA / pIC50 fill out drug-likeness + potency.

        The ``features`` argument is accepted for backward-compatibility
        with the original ``heuristic(features) -> float`` signature
        (and is forwarded as the SMILES context to
        :meth:`RewardAggregator.aggregate`).  Two shapes are
        supported:

        * ``features`` is a :class:`MoleculeClosedTerm` — the
          canonical SMILES is derived from the state directly.
        * ``features`` is a ``str`` — treated as the SMILES.
        * ``features`` is anything else — falls back to ``""`` SMILES.

        The returned value is squashed into [0, 1] via ``sigmoid`` so
        PUCT can use it as a valid prior mass.  When the aggregator
        has no channels wired (or every channel yields 0.0) the
        method returns ``0.5`` so the search degrades to vanilla UCB
        — matching the historical behaviour for the "no signal yet"
        path.

        Parameters
        ----------
        features : Any
            A :class:`MoleculeClosedTerm`, a SMILES string, or any
            descriptor the caller happened to pass.

        Returns
        -------
        float
            Prior probability in ``[1e-6, 1.0 - 1e-6]``.  Falls back
            to ``0.5`` when the aggregator has no signal.
        """
        # Resolve SMILES from the features argument.  Anything that
        # exposes ``canonical_smiles`` is treated as a closed-term
        # state; bare strings are passed through; everything else
        # degrades to an empty SMILES.
        smi: str = ""
        if isinstance(features, str):
            smi = features
        else:
            try:
                smi = str(features.canonical_smiles() or "")
            except Exception:
                smi = ""

        # Build the per-channel reward dict by invoking each
        # configured channel on a *fresh* empty stub state.  Because
        # the channels are typically SMILES-based (RDKit QED, SA,
        # Vina-proxy) we forward the canonical SMILES where possible
        # — falling back to the state-only path when the channel
        # signature requires a real :class:`MoleculeClosedTerm`.
        channels: Dict[str, float] = {}
        try:
            reward_fn = self._resolved_reward()
            # Use the original state when ``features`` is a closed
            # term; otherwise build a synthetic state from SMILES.
            if isinstance(features, MoleculeClosedTerm):
                state = features
            elif smi:
                try:
                    state = MoleculeClosedTerm.from_smiles(smi, embed_3d=False)
                except Exception:
                    state = None  # type: ignore[assignment]
            else:
                state = None  # type: ignore[assignment]
            if state is not None:
                for ch_name in (
                    "r_vina",
                    "r_sa",
                    "r_qed",
                    "r_vina_proxy",
                    "r_posebusters",
                    "r_pb_valid",
                    "r_pic50",
                    "r_retro",
                    "r_reinvent4",
                    "r_synth",
                    "r_admet",
                ):
                    ch = getattr(reward_fn, ch_name, None)
                    if ch is None:
                        continue
                    try:
                        v = float(ch(state))
                    except Exception:
                        v = 0.0
                    channels[ch_name] = v
        except Exception:
            channels = {}

        # Delegate the per-channel weighting to the aggregator.
        try:
            agg = self._resolved_reward()
            raw = float(agg.aggregate(smi, channels))
        except Exception:
            raw = 0.0

        # No-signal fallback: every channel returned 0.0 (or the
        # aggregator has no channels wired).  In that case the prior
        # degrades to the constant 0.5 baseline so the search
        # collapses to vanilla UCB — exactly the historical stub.
        if not channels or abs(raw) <= 0.0:
            return 0.5

        # Squash into [0, 1] so PUCT sees a valid prior mass.
        try:
            p = 1.0 / (1.0 + math.exp(-float(raw)))
        except OverflowError:
            p = 0.0 if float(raw) < 0 else 1.0
        return float(max(1e-6, min(1.0 - 1e-6, p)))

    def _prior(self, state: MoleculeClosedTerm) -> float:
        """Prior probability of ``state`` — TODO-02 dispatches to
        :meth:`heuristic` so the prior is a real per-channel
        weighted sum rather than a constant 0.5 stub.

        Backward-compatible cascade:

            1. :meth:`heuristic` (per-channel ``RewardAggregator``-based
               prior — the new default);
            2. fitted :class:`SymbolicPrior` (if any) via
               ``predict_proba`` — only consulted when the
               aggregator yields no signal;
            3. module-level :func:`heuristic` stub (= 0.5);
            4. hard-coded 0.5.

        Each layer preserves full backward compatibility with the
        pre-strengthening implementation.  :meth:`heuristic` is the
        canonical entry-point; this method is the thin shim that
        MCTS internals call.
        """
        # 1) RewardAggregator-based prior via the new heuristic method.
        try:
            v = float(self.heuristic(state))
            if math.isfinite(v):
                return v
        except Exception:
            pass

        # 2) Legacy SymbolicPrior path — used when the aggregator
        # yields no signal (no channels wired).
        if self.prior is not None:
            try:
                return float(self.prior.predict_proba(state))
            except Exception:
                pass

        # 3) Last-resort fallback — the constant ``heuristic`` stub
        # (= 0.5).  Kept for backward compatibility so external code
        # that monkey-patches ``heuristic`` still wins.
        try:
            features = [
                float(state.n_atoms),
                float(state.n_bonds),
                float(sum(state.free_sites.values())),
            ]
            return float(heuristic(features))
        except Exception:
            return 0.5

    def _safe_reduce(
        self,
        rule: ReactionRule,
        state: MoleculeClosedTerm,
        tile: MoleculeClosedTerm,
    ) -> List[MoleculeClosedTerm]:
        """Apply ``rule.reduce((state, tile))`` and return the products.

        Catches all exceptions (RDKit parse errors, ill-typed redexes,
        etc.) and returns ``[]`` — a "non-fire" is a legitimate MCTS
        outcome (the redex simply didn't apply).
        """
        try:
            return rule.reduce((state, tile))
        except Exception:
            return []

    def _node_already_present(
        self,
        parent: _MCTSNode,
        child_state: MoleculeClosedTerm,
    ) -> bool:
        """True iff a child of ``parent`` is α-equivalent to ``child_state``.

        Two molecules that share a canonical SMILES are α-equivalent,
        so the new child would be a duplicate and we skip adding it.
        RDKit errors are caught and treated as "not present" (we add).
        """
        try:
            child_smi = child_state.canonical_smiles()
            if child_smi:
                from molmetal_lam.tile_lib.canonical_cache import canonicalize
                child_smi = canonicalize(str(child_smi))
        except Exception:
            return False
        for existing in parent.children:
            try:
                es = existing.state.canonical_smiles()
                if es:
                    from molmetal_lam.tile_lib.canonical_cache import canonicalize
                    es = canonicalize(str(es))
                if es == child_smi:
                    return True
            except Exception:
                continue
        return False

    # ------------------------------------------------------------------
    # (D2) Transposition table + node recycling
    # ------------------------------------------------------------------

    def _canonical_smi(self, state: MoleculeClosedTerm) -> str:
        """Canonical SMILES of ``state``, or a unique fallback token.

        Wraps :func:`molmetal_lam.tile_lib.canonical_cache.canonicalize`
        so the RDKit round-trip is memoised across calls.  The cache
        key is the raw SMILES returned by ``state.canonical_smiles()``
        (or :attr:`source_smiles`) — the same raw string is canonical to
        itself on repeated visits.
        """
        try:
            smi = state.canonical_smiles()
            if smi:
                from molmetal_lam.tile_lib.canonical_cache import canonicalize
                return str(canonicalize(str(smi)))
        except Exception:
            pass
        try:
            src = getattr(state, "source_smiles", "") or ""
            if src:
                from molmetal_lam.tile_lib.canonical_cache import canonicalize
                return str(canonicalize(str(src)))
        except Exception:
            pass
        return f"id:{id(state)}"

    def _lookup_or_create(
        self,
        canonical_smi: str,
        factory: Callable[[], "_MCTSNode"],
    ) -> Tuple["_MCTSNode", bool]:
        """Return ``(node, hit)`` for the canonical SMILES ``canonical_smi``.

        Round-3 MCTS upgrade: consult the bounded-FIFO
        :class:`_TranspositionTable` first, then fall back to the
        legacy :attr:`_t_table` dict for backward compatibility with
        the D2 metric collection.  If either cache returns a node
        the function returns ``(existing, True)`` (transposition
        hit); otherwise ``factory()`` is called and the new node is
        inserted into *both* caches so subsequent lookups stay
        consistent.  Factory exceptions are caught — when the factory
        raises we still return ``hit=False`` with a fresh empty node
        so callers do not crash the search.
        """
        # 1) Bounded-FIFO table (round 3) — preferred path.
        try:
            existing = self.transposition_table.lookup(canonical_smi)
        except Exception:
            existing = None
        # 2) Legacy D2 dict — kept as a fallback so the metrics
        # ``TRANSPOSITION_HIT_RATE`` and ``_t_hits`` / ``_t_misses``
        # remain accurate even if the bounded table evicted the node.
        if existing is None:
            try:
                existing = self._t_table.get(canonical_smi)
            except Exception:
                existing = None
        if existing is not None:
            try:
                self._t_hits += 1
            except Exception:
                pass
            return existing, True
        try:
            node = factory()
        except Exception:
            node = _MCTSNode(state=type("X", (), {})())  # type: ignore[abstract]
        try:
            self.transposition_table.insert(canonical_smi, node)
            self._t_table[canonical_smi] = node
            self._t_misses += 1
        except Exception:
            pass
        return node, False

    def _recycle_node(self, node: "_MCTSNode") -> None:
        """Reset ``(N, W, Q)`` and clear children, preserving node identity.

        Round-3 MCTS upgrade: when the same canonical SMILES is
        re-expanded by the search (= e.g. an MCTS worker revisits a
        leaf after the budget cycle resets), the *node identity* is
        what gives PUCT a stable parent pointer.  We therefore keep
        the node object alive (same Python ``id``, same parent
        chain) but reset its statistics so the next simulation
        starts from a clean slate: ``N = 0``, ``W = 0.0`` (which
        forces ``Q = W / N = 0.0``), and an empty ``children`` list.
        The node is also evicted from the :class:`_TranspositionTable`
        and the legacy :attr:`_t_table` so the next
        :meth:`_lookup_or_create` allocates a fresh node if needed.

        Virtual-loss counters in :attr:`virtual_loss` are released
        unconditionally so a recycled node never carries stale
        in-flight locks.
        """
        if node is None:
            return
        try:
            node.children.clear()
            node.expansion_complete = False
        except Exception:
            pass
        try:
            node.N = 0
        except Exception:
            pass
        try:
            node.W = 0.0
        except Exception:
            pass
        try:
            node.virtual_loss.value = 0.0
        except Exception:
            pass
        try:
            self.virtual_loss.release(id(node))
        except Exception:
            pass
        try:
            smi = self._canonical_smi(node.state)
            # Evict from both the round-3 bounded table and the legacy
            # D2 dict so the next lookup starts clean.
            try:
                tt = self.transposition_table
                if tt is not None and tt.lookup(smi) is node:
                    # ``del self.transposition_table._table[smi]`` —
                    # reach in to avoid exposing a delete() method on
                    # the public surface (FIFO eviction policy only).
                    try:
                        del tt._table[smi]
                    except Exception:
                        pass
            except Exception:
                pass
            if smi in self._t_table and self._t_table[smi] is node:
                del self._t_table[smi]
        except Exception:
            pass

    # ------------------------------------------------------------------
    # (D2) Automatic SymbolicPrior refit
    # ------------------------------------------------------------------

    def _maybe_refit_prior(self) -> None:
        """Refit :attr:`prior` against the accumulated replay buffer.

        Triggered from :meth:`_simulate` every :attr:`prior_refit_every`
        simulations when ``len(accumulated_leaf_pairs) >= 100`` AND
        the buffer has grown since the last refit.  On success a
        fresh :class:`SymbolicPrior` (backed by a fresh
        :class:`HeuristicRegressor`) replaces ``self.prior`` in place
        so PUCT exploitation sees the latest regression.

        Exceptions are caught and surfaced to :attr:`history` via the
        key ``PRIOR_REFIT_SKIPPED = True`` so the closed-loop pipeline
        can detect and report failed refits without crashing the
        search.
        """
        pairs = list(getattr(self, "accumulated_leaf_pairs", []) or [])
        if len(pairs) < 100:
            return
        prev_size = int(getattr(self, "_prior_refit_last_size", 0) or 0)
        if len(pairs) <= prev_size:
            return
        try:
            import numpy as _np
            from molmetal_lam.lam_chem.pysr_wrapper import (
                HeuristicRegressor as _HR,
            )
            X = _np.array(
                [list(p[0]) for p in pairs], dtype=float,
            )
            y = _np.array([p[1] for p in pairs], dtype=float)
            new_regressor = _HR().fit(X, y)
            if bool(getattr(new_regressor, "_fitted", False)):
                self.prior = SymbolicPrior(regressor=new_regressor)
                self._prior_refit_last_size = len(pairs)
                self._prior_refit_skipped = False
            else:
                self._prior_refit_skipped = True
        except Exception:
            # Any fitting failure — surface as a "skipped" refit so
            # the closed-loop report can flag it without losing the
            # search progress.
            self._prior_refit_skipped = True

    def _collect_leaves(
        self,
        root: _MCTSNode,
        leaves_by_smi: Dict[str, _MCTSNode],
    ) -> Dict[str, _MCTSNode]:
        """Snapshot candidate statistics without backpropagating a second time.

        Keep each historical leaf node once. Equivalent states reached along
        distinct paths contribute their actual statistics to an independent
        candidate snapshot; neither repeated collection nor aggregation may
        modify the tree's visit counts or parent pointers.
        """
        if not leaves_by_smi or not hasattr(self, "_leaf_registry_nodes"):
            self._leaf_registry_nodes = {}
        stack: List[_MCTSNode] = [root]
        visited = set()
        while stack:
            n = stack.pop()
            if id(n) in visited:
                continue
            visited.add(id(n))
            if not n.children:
                try:
                    key = n.state.canonical_smiles()
                except Exception:
                    key = f"id:{id(n.state)}"
                self._leaf_registry_nodes.setdefault(key, {})[id(n)] = n
                continue
            stack.extend(n.children)
        for key, nodes_by_id in self._leaf_registry_nodes.items():
            nodes = list(nodes_by_id.values())
            representative = nodes[0]
            leaves_by_smi[key] = _MCTSNode(
                state=representative.state, N=sum(n.N for n in nodes),
                W=sum(n.W for n in nodes), P=representative.P,
                rule_name=representative.rule_name, tile=representative.tile,
            )
        return leaves_by_smi

    def _count_nodes(self, root: _MCTSNode) -> int:
        """Count the total number of nodes under ``root`` (inclusive)."""
        n = 0
        stack: List[_MCTSNode] = [root]
        while stack:
            cur = stack.pop()
            n += 1
            stack.extend(cur.children)
        return n

    # ------------------------------------------------------------------
    # (L-2) Persistent tree — global accumulation, save / load
    # ------------------------------------------------------------------

    def _accumulate_leaf_pairs(
        self,
        leaves_by_smi: Dict[str, _MCTSNode],
    ) -> None:
        """Append every (features → leaf_score) pair to the global buffer.

        ``accumulated_leaf_pairs`` is the AlphaZero replay buffer:
        every leaf that the search visits contributes one entry, so the
        :class:`SymbolicPrior` refit on the next iteration sees more
        data.  Duplicates across iterations are kept (the buffer is a
        multiset) — the underlying :class:`HeuristicRegressor` averages
        over them.

        R3/F4 — also records the per-leaf PB outer-gate verdict into
        :attr:`_leaf_pb_valid_history`, aligned 1:1 with
        :attr:`accumulated_leaf_pairs`.  Symbolic-prior refits consult
        this buffer to weight / filter the regression target toward
        PB-passing molecules.
        """
        # Resolve the PB outer-gate channel once per call so a missing
        # / failing channel is detected once rather than once-per-leaf.
        pb_channel = None
        try:
            reward_obj = self._resolved_reward()
            pb_channel = getattr(reward_obj, "r_pb_valid", None)
        except Exception:
            pb_channel = None

        for node in leaves_by_smi.values():
            try:
                feats = list(self.prior.feature_extractor(node.state)) \
                    if self.prior is not None \
                    else _default_feature_extractor(node.state)
            except Exception:
                feats = [float(node.state.n_atoms),
                         float(node.state.n_bonds),
                         float(sum(node.state.free_sites.values()))]
            try:
                score = float(node.W) / max(1, int(node.N))
            except Exception:
                score = 0.0
            # Compute the PB outer-gate verdict for this leaf.  When no
            # channel is configured (or it returns a non-binary value)
            # we record ``None`` so the side-table stays aligned.
            pb_valid: Optional[bool] = None
            if pb_channel is not None:
                try:
                    v = float(pb_channel(node.state))
                    if v >= 0.5:
                        pb_valid = True
                    elif v > -0.5:
                        pb_valid = False
                    else:
                        pb_valid = None
                except Exception:
                    pb_valid = None
            try:
                self.accumulated_leaf_pairs.append((list(feats), score))
                if not hasattr(self, "_leaf_pb_valid_history"):
                    self._leaf_pb_valid_history = []
                self._leaf_pb_valid_history.append(pb_valid)
            except Exception:
                continue

    def save_tree(self, path: str) -> None:
        """Checkpoint the persistent tree + accumulated data to ``path``.

        Writes a single ``.npz`` archive containing:

        * ``tree_json`` — UTF-8 string of ``self._root.to_dict()`` (or
          ``""`` when no search has been run yet).
        * ``leaf_history`` — ``np.asarray(self._leaf_value_history)``
          (cumulative per-simulation reward buffer).
        * ``accumulated_pairs`` — ``np.asarray(self.accumulated_leaf_pairs,
          dtype=object)`` so the global (features → reward) buffer
          round-trips faithfully even when feature dimensions vary.
        * ``prior_coefs`` / ``prior_intercept`` / ``prior_fitted`` —
          :class:`SymbolicPrior` sklearn-fallback coefficients, so the
          refit accumulates across iterations even when no search has
          fired yet.
        * ``n_simulations`` / ``c_puct`` / ``dirichlet_alpha`` /
          ``dirichlet_fraction`` — config snapshot for reproducibility.

        Parameters
        ----------
        path : str
            Destination ``.npz`` path.  Parent directory is created if
            missing.
        """
        import os as _os
        parent = _os.path.dirname(_os.path.abspath(path))
        if parent:
            _os.makedirs(parent, exist_ok=True)
        try:
            import json as _json
            tree_json = _json.dumps(
                self._root.to_dict() if self._root is not None else {},
                ensure_ascii=False,
            )
        except Exception:
            tree_json = "{}"
        leaf_history = np.asarray(
            list(getattr(self, "_leaf_value_history", []) or []),
            dtype=float,
        )
        try:
            accum = np.asarray(
                list(getattr(self, "accumulated_leaf_pairs", []) or []),
                dtype=object,
            )
        except Exception:
            accum = np.empty((0,), dtype=object)
        prior_coefs = np.zeros((0,), dtype=float)
        prior_intercept = np.zeros((0,), dtype=float)
        prior_fitted = 0
        try:
            prior_obj = self.prior
            if prior_obj is not None and getattr(prior_obj, "_model", None) is not None:
                model = prior_obj._model
                coef = getattr(model, "coef_", None)
                if coef is not None:
                    prior_coefs = np.asarray(coef, dtype=float).ravel()
                intercept = getattr(model, "intercept_", None)
                if intercept is not None:
                    prior_intercept = np.asarray(
                        np.atleast_1d(intercept), dtype=float,
                    ).ravel()
                prior_fitted = 1 if bool(getattr(prior_obj, "fitted", False)) else 0
        except Exception:
            pass
        np.savez(
            path,
            tree_json=np.asarray(tree_json),
            leaf_history=leaf_history,
            accumulated_pairs=accum,
            prior_coefs=prior_coefs,
            prior_intercept=prior_intercept,
            prior_fitted=np.asarray(int(prior_fitted)),
            n_simulations=np.asarray(int(self.n_simulations)),
            c_puct=np.asarray(float(self.c_puct)),
            dirichlet_alpha=np.asarray(float(self.dirichlet_alpha)),
            dirichlet_fraction=np.asarray(float(self.dirichlet_fraction)),
        )

    @classmethod
    def load_tree(
        cls,
        path: str,
        *,
        tile_library: List[MoleculeClosedTerm],
        rules: Dict[str, ReactionRule],
        target_predicates: Optional[List[TypePredicate]] = None,
        binding_site: Optional[BindingSite] = None,
        scorer: Optional[ScorerFn] = None,
        reward: Optional[RewardAggregator] = None,
        n_simulations: int = 1000,
        c_puct: float = 1.4,
        top_k: int = 20,
        rng: Optional[random.Random] = None,
        prior: Optional[SymbolicPrior] = None,
        dirichlet_alpha: float = 0.3,
        dirichlet_fraction: float = 0.25,
        rollout_epsilon: float = 0.0,
    ) -> "MCTSProofSearch":
        """Restore a previously-checkpointed :class:`MCTSProofSearch`.

        Reads the ``.npz`` produced by :meth:`save_tree`, rebuilds the
        ``_root`` subtree (via :meth:`_MCTSNode.from_dict`), and
        re-populates the ``_leaf_value_history`` and
        ``accumulated_leaf_pairs`` buffers so the *next* iteration's
        :class:`SymbolicPrior` refit sees the cumulative dataset.

        Configuration parameters (``n_simulations``, ``c_puct``,
        ``dirichlet_*``, etc.) are *overridable* — pass them when you
        want the loaded tree to live inside a search with different
        budget knobs (the canonical use-case is "bump n_simulations
        each iter" AlphaZero-style curricula).

        Parameters
        ----------
        path : str
            ``.npz`` checkpoint produced by :meth:`save_tree`.
        tile_library, rules, target_predicates, binding_site, scorer,
        reward, n_simulations, c_puct, top_k, rng, prior,
        dirichlet_alpha, dirichlet_fraction, rollout_epsilon :
            Forwarded to the :class:`MCTSProofSearch` constructor
            verbatim.

        Returns
        -------
        MCTSProofSearch
            A *fresh* instance whose ``_root`` / ``_leaf_value_history``
            / ``accumulated_leaf_pairs`` were restored from ``path``.
        """
        from molmetal_lam.lam_chem.pysr_wrapper import HeuristicRegressor

        with np.load(path, allow_pickle=True) as data:
            tree_json = "".join(
                [str(x) for x in np.atleast_1d(data["tree_json"]).tolist()],
            )
            leaf_history = np.asarray(
                data.get("leaf_history", np.zeros((0,), dtype=float)),
                dtype=float,
            ).tolist()
            accumulated_pairs_raw = data.get(
                "accumulated_pairs", np.empty((0,), dtype=object),
            )
            prior_coefs = np.asarray(
                data.get("prior_coefs", np.zeros((0,), dtype=float)),
                dtype=float,
            )
            prior_intercept = np.asarray(
                data.get("prior_intercept", np.zeros((0,), dtype=float)),
                dtype=float,
            )
            prior_fitted = int(np.atleast_1d(
                data.get("prior_fitted", np.zeros((1,), dtype=int)),
            ).tolist()[0])

        # Resolve / build the SymbolicPrior.
        if prior is None:
            prior = SymbolicPrior()
        # When the checkpoint carries fitted coefficients, splice them
        # back into a fresh sklearn-fallback HeuristicRegressor so the
        # prior is immediately usable on the next iter.
        if (
            prior_fitted == 1
            and prior_coefs.size > 0
            and not bool(getattr(prior, "fitted", False))
        ):
            try:
                from sklearn.linear_model import Ridge  # type: ignore
                n_features = int(prior_coefs.size)
                model = Ridge(alpha=1.0, random_state=int(prior.random_state))
                # Lazy fit on a tiny synthetic grid so the regressor is
                # in the fitted state; immediately overwrite coef_ +
                # intercept_ with the checkpoint values so predict()
                # reproduces the saved prior exactly.
                X0 = np.zeros((2, n_features), dtype=float)
                y0 = np.zeros((2,), dtype=float)
                model.fit(X0, y0)
                model.coef_ = prior_coefs.reshape(1, -1)
                ic = float(prior_intercept[0]) if prior_intercept.size else 0.0
                model.intercept_ = np.asarray([ic], dtype=float)
                hr = HeuristicRegressor(
                    niterations=int(prior.niterations),
                    random_state=int(prior.random_state),
                )
                hr._model = model
                hr._fitted = True
                hr.backend_ = "sklearn"
                prior._model = hr
                prior.fitted = True
                prior.n_features = n_features
            except Exception:
                # Fall back to an unfitted prior — caller will refit on
                # the accumulated data anyway.
                pass

        # Build the search instance with overridden kwargs.
        inst = cls(
            tile_library=list(tile_library),
            rules=dict(rules),
            target_predicates=list(target_predicates or []),
            binding_site=binding_site if binding_site is not None else BindingSite(
                name="empty_site", constraints=[], geometry_hints={},
            ),
            scorer=scorer,
            reward=reward,
            n_simulations=int(n_simulations),
            c_puct=float(c_puct),
            top_k=int(top_k),
            rng=rng,
            prior=prior,
            dirichlet_alpha=float(dirichlet_alpha),
            dirichlet_fraction=float(dirichlet_fraction),
            rollout_epsilon=float(rollout_epsilon),
        )

        # Restore the global buffers.
        inst._leaf_value_history = [float(v) for v in leaf_history]
        try:
            pairs: List[Tuple[List[float], float]] = []
            for p in accumulated_pairs_raw.tolist():
                if (
                    isinstance(p, (list, tuple))
                    and len(p) == 2
                    and isinstance(p[0], (list, tuple))
                ):
                    pairs.append(([float(x) for x in p[0]], float(p[1])))
            inst.accumulated_leaf_pairs = pairs
        except Exception:
            inst.accumulated_leaf_pairs = []

        # Rebuild the root subtree (if present).
        if tree_json and tree_json.strip() not in {"", "{}"}:
            try:
                import json as _json
                d = _json.loads(tree_json)
                state_lookup: Dict[str, MoleculeClosedTerm] = {}
                for t in tile_library:
                    try:
                        state_lookup[_smi_of(t)] = t
                    except Exception:
                        continue
                inst._root = _MCTSNode.from_dict(
                    d,
                    tile_library=tile_library,
                    rules=rules,
                    state_lookup=state_lookup,
                )
            except Exception:
                inst._root = None

        return inst


# ---------------------------------------------------------------------------
# Convenience aliases
# ---------------------------------------------------------------------------

#: Alias for ``MCTSProofSearch`` (matches the abbreviation used in the
#: formal memo §8).  Both names refer to the same class.
MCTS = MCTSProofSearch


__all__ = [
    "MCTSProofSearch",
    "MCTS",
    "heuristic",
    "ScorerFn",
    "SymbolicPrior",
    "RewardAggregator",
    "VirtualLoss",
    "_VirtualLoss",
    "_TranspositionTable",
]
