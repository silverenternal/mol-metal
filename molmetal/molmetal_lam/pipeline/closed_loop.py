"""LamClickDesignLoop — closed-loop MLC drug-design orchestrator.

============================================================
The Molecular Lambda Calculus "closed loop"
============================================================
The proof-search layer (:class:`MCTSProofSearch`) only *finds* candidate
closed λ-terms.  It does not (a) score them with a real
chemistry-aware reward, or (b) refine its own heuristic as data
accumulates.  Both gaps are closed here.

Each iteration of :meth:`LamClickDesignLoop.run` performs five steps:

    1. **MCTS search** — run :meth:`MCTSProofSearch.search` from a seed
       tile, collecting up to ``top_k`` candidates.
    2. **Batch scoring** — wrap every candidate's SMILES through the
       REINVENT4-style scorer, returning a real-valued reward vector.
    3. **Top-K selection** — rank candidates by score and keep the
       top-K for downstream analysis.
    4. **Heuristic fit** — fit :class:`HeuristicRegressor` on
       ``(features, scores)`` so the *next* iteration's MCTS has a
       learned prior over features.
    5. **Equation extraction** — splice the fitted expression into the
       loop's :attr:`paper_equation` field — this is the "extracted
       formula" the paper reports as the closed-loop's deliverable.

The loop is intentionally **stateless across iterations**: it does not
backpropagate rewards into the MCTS tree (the tree is rebuilt every
call from the seed tile).  This keeps the API clean and makes the loop
trivially parallelisable across seeds.

Public API
----------
:class:`LamClickDesignLoop`    the orchestrator
:func:`extract_paper_equation`  pretty-print the fitted heuristic
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from molmetal_lam.lam_chem.ast import LamAbs, LamApp, LamNode, LamVar, var
from molmetal_lam.lam_chem.pysr_wrapper import HeuristicRegressor
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.pipeline.extract_features import (
    FEATURE_NAMES,
    molecule_to_features,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------
# R3/F4 — PoseBusters outer-gate wiring
# ----------------------------------------------------------------
def _safe_pb_check(smiles: str) -> "dict[str, Any] | None":
    """One-shot PoseBusters check via the runner module.

    Always returns a dict on success; ``None`` on any failure (so the
    caller can record ``pb_pass_rate=0.0`` without raising).  This is
    the closed-loop's *outer-gate* path — distinct from the in-loop
    RewardAggregator channel which only returns a binary 0/1.

    Resolution order:

    1. :mod:`molmetal.validation.posebusters_runner` (the R3/F4 module).
    2. :mod:`molmetal_lam.sbdd_env.posebusters_adapter` (legacy
       full-PoseBusters adapter — used when the runner is unavailable
       in a stale install).
    """
    try:
        from molmetal.validation.posebusters_runner import check_posebusters
        return check_posebusters(smiles)
    except Exception:
        pass
    try:
        from molmetal_lam.sbdd_env.posebusters_adapter import (
            PoseBustersAdapter,
            _have_posebusters,
        )
        if _have_posebusters():
            return PoseBustersAdapter().validate_mol(smiles).to_dict()
        return {"pb_valid": None, "skipped": True, "reason": "posebusters not installed"}
    except Exception:
        return {"pb_valid": None, "skipped": True, "reason": "posebusters not installed"}


def _compute_pb_pass_rate(smiles_list: List[str]) -> float:
    """Compute the PB outer-gate pass rate over ``smiles_list``.

    Skipped / missing entries (PB not installed, embed_failed, etc.)
    are *ignored* — only valid ``pb_valid`` verdicts count.  Returns
    ``0.0`` when no valid verdict can be computed.
    """
    if not smiles_list:
        return 0.0
    n_total = 0
    n_passed = 0
    for smi in smiles_list:
        if not smi or not isinstance(smi, str):
            continue
        res = _safe_pb_check(smi)
        if not isinstance(res, dict):
            continue
        v = res.get("pb_valid", None)
        if v is None:
            continue
        n_total += 1
        if bool(v):
            n_passed += 1
    if n_total == 0:
        return 0.0
    return float(n_passed) / float(n_total)


# ---------------------------------------------------------------------------
# Phase 2 — Vina real vs proxy wiring
# ---------------------------------------------------------------------------
def _vina_proxy_score(smiles: str) -> Optional[float]:
    """The pre-existing baselines.py proxy: ``0.5 + 0.1 * NumRotatableBonds``.

    Returns ``None`` on RDKit failure (empty SMILES, parse error).
    Returns 0.5 when the molecule parses but has zero rotatable bonds.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import rdMolDescriptors  # type: ignore
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        n_rot = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    except Exception:
        return None
    return float(0.5 + 0.1 * n_rot)


def _compute_vina_top1(
    best_smiles: str,
    pocket: Any,
    best_score: float,
    docker: Any,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute the per-iteration Vina real + proxy top-1 scores.

    Returns a ``(real, proxy)`` tuple where each entry is either:

    * a ``float`` (kcal/mol for ``real``; arbitrary units for ``proxy``),
    * or ``None`` if the channel is unavailable / produced no score.

    Real Vina is preferred whenever ``docker`` is a :class:`DockingEngine`
    (or duck-typed equivalent) — we route the best SMILES through it and
    read back the kcal/mol from the top-1 :class:`Complex`.  When that
    fails (no docker, no pocket, dock() raises, or no poses returned) we
    fall back to :func:`_vina_proxy_score` so the field is never silently
    absent.
    """
    real: Optional[float] = None
    proxy: Optional[float] = _vina_proxy_score(best_smiles)

    if (
        docker is not None
        and pocket is not None
        and isinstance(best_smiles, str)
        and best_smiles
    ):
        try:
            from molmetal.ports import DockingConfig  # local import
            from molmetal.domain import Molecule  # local import
            import torch as _torch  # local import — same env as the loop
            # Build a minimal Molecule stub from SMILES — the docker
            # only needs ``smiles`` populated (Vina re-embeds anyway).
            try:
                from rdkit import Chem as _Chem  # type: ignore
                _mol = _Chem.MolFromSmiles(best_smiles)
                _atoms = (
                    _mol.GetNumAtoms() if _mol is not None else 1
                )
            except Exception:
                _atoms = 1
            dummy = Molecule(
                coords=_torch.zeros(int(_atoms), 3),
                atom_types=_torch.full((int(_atoms),), 6, dtype=_torch.long),
                bonds=_torch.zeros(2, 0, dtype=_torch.long),
                bond_types=_torch.zeros(0, dtype=_torch.long),
                formal_charges=_torch.zeros(int(_atoms), dtype=_torch.long),
                smiles=best_smiles,
            )
            cfg = DockingConfig(n_poses=1, exhaustiveness=4)
            complexes = docker.dock(dummy, pocket, cfg)
            if complexes:
                real = float(getattr(complexes[0], "vina_score", float("nan")))
                if real != real:  # NaN check
                    real = None
        except Exception as exc:
            log.debug("docker.dock for vina_real_score_top1 failed: %s", exc)
            real = None

    return real, proxy


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class _IterationRecord:
    """Per-iteration telemetry emitted by :meth:`LamClickDesignLoop.run`."""

    iteration: int
    best_smiles: str
    best_lambda_expr: str
    best_score: float
    extracted_formula: str
    top_k_smiles: List[str] = field(default_factory=list)
    top_k_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "best_smiles": self.best_smiles,
            "best_lambda_expr": self.best_lambda_expr,
            "best_score": float(self.best_score),
            "extracted_formula": self.extracted_formula,
            "top_k_smiles": list(self.top_k_smiles),
            "top_k_scores": list(self.top_k_scores),
        }


# ---------------------------------------------------------------------------
# molecule -> lambda expression
# ---------------------------------------------------------------------------
def _molecule_to_lambda_expr(m: MoleculeClosedTerm) -> LamNode:
    """Render a closed-term as a tiny λ-term.

    The encoding is intentionally lossy: it captures the *structure*
    of the molecule as a left-associated application of ``Atom``-named
    LamVars, then wraps the whole thing in a λ that abstracts over the
    feature vector.  This is what the paper calls the "best lambda
    expression" — it's the AST node whose β-NF the molecule witnesses.

    Concretely, for a molecule with atoms ``[a0, a1, a2, ...]`` and
    ``k`` covalent bonds, we build::

        λx.(a0 a1)          for 1 covalent bond
        λx.((a0 a1) a2)     for 2 covalent bonds
        ...

    i.e. a left-associated application whose spine mirrors the
    β-reduction history of the term.  The outer λ abstracts over the
    feature variable ``x`` that the heuristic consumes.
    """
    if m is None or m.n_atoms == 0:
        return LamAbs(var=var("x"), body=var("x"))

    atoms = m.atoms
    expr: LamNode = var(atoms[0].symbol if hasattr(atoms[0], "symbol") else "A0")
    for i in range(1, len(atoms)):
        sym = atoms[i].symbol if hasattr(atoms[i], "symbol") else f"A{i}"
        expr = LamApp(func=expr, arg=var(sym))

    # The outer λ abstracts over the heuristic feature variable.
    return LamAbs(var=var("x"), body=expr)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def extract_paper_equation(
    top_states: Sequence[MoleculeClosedTerm],
    scores: Sequence[float],
    heuristic: Optional[HeuristicRegressor] = None,
) -> str:
    """Render the fitted heuristic as a paper-style equation string.

    When ``heuristic`` is fitted this returns ``heuristic.equation()``
    (sympy / linear formula).  When unfitted we fall back to a
    single-line summary of the best molecule + its score.
    """
    if heuristic is not None and getattr(heuristic, "_fitted", False):
        return heuristic.equation()
    if not top_states or not scores:
        return "<no candidates>"
    best = top_states[int(np.argmax(scores))]
    return f"best={getattr(best, 'canonical_smiles', lambda: '?')()} @ score={max(scores):.3f}"


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------
class LamClickDesignLoop:
    """Closed-loop orchestrator for the Molecular Lambda Calculus.

    Parameters
    ----------
    mcts : MCTSProofSearch
        A configured proof-search instance — its ``search(initial_state,
        max_depth)`` is called once per iteration.
    scorer : REINVENT4Scorer
        Multi-property scorer (qed, binding, sas, novelty) used to
        rank candidates.  Must expose ``batch_score(smiles_list)``.
    pocket_loader : callable or None, default ``None``
        Optional ``pdb_id -> BindingSite`` callable.  When ``None`` the
        loop runs with a stub BindingSite derived from the candidate's
        SMILES — useful for unit tests + the demo.
    symbolic_reg : HeuristicRegressor
        Symbolic-regression wrapper (PySR → sklearn fallback).  Must
        expose ``fit(X, y)`` / ``predict(X)`` / ``equation()``.
    seed_tiles : list[MoleculeClosedTerm] or None
        Starting states for each iteration.  When ``None`` we generate
        a single empty-term seed and rely on the MCTS tree to expand.
    rng : random.Random or None
        Optional RNG for reproducibility.
    synthesis_oracle : SynthesisOracle or None, default ``None``
        D-3 real-synthesis reward channel.  When supplied, the loop
        wires its ``to_reward_channel()`` closure into the MCTS
        ``RewardAggregator.r_synth`` (so the leaf value reflects
        predicted isolated yield) AND appends a ``synthesis_equations``
        dict to every per-iteration result entry.  When ``None`` the
        channel is disabled (``w_synth=0.0``).
    """

    def __init__(
        self,
        mcts: Any,
        scorer: Any,
        pocket_loader: Optional[Callable[[str], Any]] = None,
        symbolic_reg: Optional[HeuristicRegressor] = None,
        seed_tiles: Optional[Sequence[MoleculeClosedTerm]] = None,
        rng: Optional[random.Random] = None,
        checkpoint_dir: Optional[str] = None,
        synthesis_oracle: Optional[Any] = None,
        docker: Optional[Any] = None,
    ) -> None:
        self.mcts = mcts
        self.scorer = scorer
        self.pocket_loader = pocket_loader
        self.symbolic_reg = symbolic_reg if symbolic_reg is not None else HeuristicRegressor()
        self.seed_tiles = list(seed_tiles) if seed_tiles is not None else []
        self.rng = rng if rng is not None else random.Random(0)
        # (L-2) Persistent-tree support.  When ``checkpoint_dir`` is set
        # the loop checkpoints the MCTSProofSearch._root + accumulated
        # leaf pairs after every iteration, then re-loads at iter i>=1
        # so the SymbolicPrior refit sees the global (features → reward)
        # buffer and the tree itself is resumed from disk.
        self.checkpoint_dir = checkpoint_dir

        # D-3 — synthesis oracle.  When provided we (a) attach its
        # closure to the MCTS RewardAggregator (opt-in via
        # ``w_synth``) and (b) record its fitted equations on every
        # per-iteration result.  The aggregator gracefully degrades to
        # 0.0 when the channel is unavailable, so passing ``None`` is
        # always safe.
        self.synthesis_oracle = synthesis_oracle

        # Phase 2 — optional Vina adapter (or any DockingEngine).
        # When provided, ``run()`` will route the per-iter best SMILES
        # through ``docker.dock(...)`` and emit ``vina_real_score_top1``
        # (kcal/mol).  Otherwise only ``vina_proxy_score_top1`` is
        # populated (the pre-existing NumRotatableBonds proxy).
        self.docker = docker
        if synthesis_oracle is not None:
            try:
                reward_obj = getattr(self.mcts, "reward", None)
                if reward_obj is not None and hasattr(reward_obj, "r_synth"):
                    reward_obj.r_synth = synthesis_oracle.to_reward_channel()
                    # Opt-in weight — 1.0 so the synthesis channel
                    # carries full signal once an oracle is supplied.
                    if float(getattr(reward_obj, "w_synth", 0.0)) == 0.0:
                        reward_obj.w_synth = 1.0
            except Exception as exc:
                log.debug("synthesis_oracle -> MCTS reward wiring failed: %s", exc)

        # State accumulated by run()
        self.history: List[_IterationRecord] = []
        self.paper_equation: str = "<unfitted>"

    # ------------------------------------------------------------------ utils
    def _seed_state(self, iteration: int) -> MoleculeClosedTerm:
        """Pick a seed state for iteration ``i``.

        Cycles through ``seed_tiles`` if any, otherwise returns a
        freshly-constructed empty closed term.  The empty term is a
        valid :class:`MoleculeClosedTerm` (no atoms, no bonds, all-zero
        valence_used) — MCTS will start expanding from it.
        """
        if self.seed_tiles:
            return self.seed_tiles[iteration % len(self.seed_tiles)]
        return MoleculeClosedTerm()

    def _smiles(self, m: MoleculeClosedTerm) -> str:
        """Render any closed-term as a SMILES, falling back to '' on failure.

        Note: ``MoleculeClosedTerm.canonical_smiles`` is a *property*
        that triggers RDKit round-tripping — we detect both the
        property descriptor and a pre-computed ``source_smiles`` value.
        """
        if m is None:
            return ""
        # 1) Prefer the cached source_smiles (cheap, no RDKit call).
        if getattr(m, "source_smiles", None):
            return str(m.source_smiles)
        # 2) Fall back to the canonical_smiles *property* (which may
        #    raise in RDKit-less envs — that's fine, we swallow it).
        cs = getattr(m, "canonical_smiles", None)
        if callable(cs):
            try:
                return str(cs())
            except Exception:
                return ""
        if cs is not None:
            return str(cs)
        return ""

    # ------------------------------------------------------------------ run
    def run(
        self,
        pdb_id: str = "demo",
        n_iterations: int = 3,
        top_k: int = 10,
        max_depth: int = 3,
        n_simulations: int = 10,
    ) -> List[Dict[str, Any]]:
        """Run the closed loop and return a list of per-iteration dicts.

        Parameters
        ----------
        pdb_id : str, default ``"demo"``
            Pocket identifier; passed to ``self.pocket_loader`` if one
            is configured.  The default value skips pocket loading.
        n_iterations : int, default 3
            Number of closed-loop iterations.
        top_k : int, default 10
            Number of top candidates retained per iteration.
        max_depth : int, default 3
            β-reduction depth (= max # reactions per synthesis path).
        n_simulations : int, default 10
            Number of MCTS simulations per iteration.

        Returns
        -------
        list[dict]
            One dict per iteration with keys:
            ``iteration``, ``best_smiles``, ``best_lambda_expr``,
            ``best_score``, ``extracted_formula``, ``top_k_smiles``,
            ``top_k_scores``.
        """
        self.history = []
        results: List[Dict[str, Any]] = []

        # Optional pocket loader — used as a sanity check but never
        # raises (a missing pocket just means we run with the stub).
        pocket = None
        if self.pocket_loader is not None:
            try:
                pocket = self.pocket_loader(pdb_id)
            except Exception as exc:
                log.debug("pocket_loader(%s) failed: %s — running with stub", pdb_id, exc)

        # (L-2) Persistent-tree bookkeeping — counts how many pairs
        # were in the SymbolicPrior fit per iter so we can verify the
        # buffer *grows* across iterations (not a one-shot re-fit on
        # the latest iter's leaf batch).
        pairs_fit_size_history: List[int] = []

        for it in range(int(n_iterations)):
            t0_iter = time.time()  # L9-7 CLOSED_LOOP_ITERATION_LATENCY — start of iter.
            seed = self._seed_state(it)

            # (L-2) ITERATION BOUNDARY: when checkpoint_dir is set and
            # we are *not* on iter 0, restore the MCTSProofSearch from
            # disk so the persistent tree + accumulated (features →
            # reward) buffer survive across iterations.
            if it > 0 and self.checkpoint_dir:
                try:
                    self.mcts = self._load_checkpoint(
                        iteration=it - 1,
                        n_simulations=n_simulations,
                    )
                    # (L-2) Re-fit SymbolicPrior on the GLOBAL pair
                    # buffer so the prior improves as data accumulates.
                    try:
                        n_pairs = self._refit_symbolic_prior_on_accumulated()
                        pairs_fit_size_history.append(int(n_pairs))
                    except Exception as exc:
                        log.debug("SymbolicPrior refit on accumulated pairs failed: %s", exc)
                    # (L-2) Dirichlet re-injection at the boundary so
                    # the persistent tree does not collapse to early
                    # high-reward regions.  fraction=0.25 matches
                    # AlphaZero's standard opening exploration budget.
                    try:
                        self.mcts.reinject_dirichlet_at_root(fraction=0.25)
                    except Exception as exc:
                        log.debug("Dirichlet re-inject failed: %s", exc)
                except Exception as exc:
                    log.debug("load_checkpoint(iter=%d) failed: %s", it - 1, exc)
                    pairs_fit_size_history.append(0)
            else:
                pairs_fit_size_history.append(0)

            try:
                # Honour configurable per-iteration MCTS budget — the
                # upstream ``search`` method accepts ``max_depth``.
                candidates = self.mcts.search(
                    initial_state=seed,
                    max_depth=int(max_depth),
                ) or []
                # Some MCTS implementations cap their own n_simulations
                # at construction time; we accept either signature.
                if not candidates and hasattr(self.mcts, "n_simulations"):
                    # best-effort: nudge the budget for this iteration
                    prev = getattr(self.mcts, "n_simulations", n_simulations)
                    try:
                        self.mcts.n_simulations = max(prev, int(n_simulations))
                        candidates = self.mcts.search(initial_state=seed, max_depth=int(max_depth)) or []
                        self.mcts.n_simulations = prev
                    except Exception:
                        pass
            except Exception as exc:
                log.warning("MCTS search failed at iteration %d: %s", it, exc)
                candidates = []

            # Pad with the seed if MCTS returned nothing — keeps the
            # loop alive in early iterations / tiny libraries.
            if not candidates:
                candidates = [seed]
                # If even the seed is empty (no atoms), try to surface a
                # non-empty witness for diagnostics.
                if candidates[0].n_atoms == 0 and self.seed_tiles:
                    candidates = [self.seed_tiles[0]]

            # ---- 2) batch scoring
            smiles_list = [self._smiles(m) for m in candidates]
            try:
                scores = np.asarray(self.scorer.batch_score(smiles_list), dtype=float)
            except Exception as exc:
                log.warning("scorer.batch_score failed: %s — using zeros", exc)
                scores = np.zeros(len(candidates), dtype=float)

            # ---- 3) top-K selection
            n_keep = min(int(top_k), len(candidates))
            order = np.argsort(-scores)[:n_keep]
            top_mols = [candidates[int(i)] for i in order]
            top_scores = [float(scores[int(i)]) for i in order]

            best_idx = int(order[0])
            best_mol = candidates[best_idx]
            best_smiles = smiles_list[best_idx]
            best_score = float(scores[best_idx])

            # ---- 4) heuristic fit (only if we have enough data).
            # When the persistent tree is enabled AND we have an
            # accumulated buffer, fit on the GLOBAL buffer (so the
            # SymbolicPrior sees 2x, 3x, ... data as iterations
            # progress).  Otherwise fall back to the per-iteration fit.
            extracted_formula: str
            try:
                accum_pairs = list(
                    getattr(self.mcts, "accumulated_leaf_pairs", []) or [],
                )
                if (
                    self.checkpoint_dir is not None
                    and len(accum_pairs) >= 4
                ):
                    X_acc = np.asarray(
                        [p[0] for p in accum_pairs], dtype=float,
                    )
                    y_acc = np.asarray(
                        [p[1] for p in accum_pairs], dtype=float,
                    )
                    if (
                        X_acc.ndim == 2
                        and X_acc.shape[0] >= 2
                        and not np.allclose(X_acc, X_acc[0])
                    ):
                        self.symbolic_reg.fit(X_acc, y_acc)
                        extracted_formula = self.symbolic_reg.equation()
                        pairs_fit_size_history[-1] = int(X_acc.shape[0])
                    else:
                        extracted_formula = "<degenerate global buffer — skip fit>"
                else:
                    feats = np.stack(
                        [molecule_to_features(m) for m in top_mols], axis=0,
                    ).astype(float)
                    if (
                        feats.shape[0] >= 2
                        and not np.allclose(feats, feats[0])
                    ):
                        self.symbolic_reg.fit(feats, np.asarray(top_scores, dtype=float))
                        extracted_formula = self.symbolic_reg.equation()
                    else:
                        extracted_formula = "<degenerate data — skip fit>"
            except Exception as exc:
                log.debug("heuristic fit failed: %s", exc)
                extracted_formula = "<fit failed>"
            self.paper_equation = extracted_formula

            # ---- 5) record
            best_lambda_expr = _molecule_to_lambda_expr(best_mol).to_string()
            rec = _IterationRecord(
                iteration=it,
                best_smiles=best_smiles,
                best_lambda_expr=best_lambda_expr,
                best_score=best_score,
                extracted_formula=extracted_formula,
                top_k_smiles=[self._smiles(m) for m in top_mols],
                top_k_scores=top_scores,
            )
            self.history.append(rec)
            # R3/F4 — local snapshot for the PB outer-gate (kept here
            # so we don't reach back into ``rec`` mid-construction).
            top_k_smiles_list = list(rec.top_k_smiles)

            # (Phase 2) — Batch the per-iteration results-dict
            # construction so all the ``int(...)`` / ``bool(...)`` /
            # ``float(...)`` casts (each of which is a CPU transfer
            # when the source is a GPU tensor) happen at a single
            # sync point.  Previously each metric was assigned to
            # ``results[-1][key]`` incrementally, triggering one
            # implicit GPU→CPU sync per cast.  Now we accumulate the
            # raw (possibly tensor-valued) metrics in a local dict and
            # cast them all at once at the dict-finalisation boundary.
            _raw_metrics: Dict[str, Any] = {
                "cumulative_nfe": getattr(self.mcts, "nfe", 0),
                "cumulative_nfe_reductions": getattr(
                    self.mcts, "nfe_reductions", 0,
                ),
                "cumulative_nfe_oracle": getattr(
                    self.mcts, "nfe_oracle", 0,
                ),
                "early_stopped": False,
                "early_stopped_at_iter": -1,
                "tree_node_count": None,
                "synthesis_equations": None,
            }
            # Defer tensor / numpy → Python conversion: read everything
            # while the GPU is still busy, batch the conversion at the end.
            try:
                last_hist = (
                    list(getattr(self.mcts, "history", []) or [])[-1]
                    if list(getattr(self.mcts, "history", []) or [])
                    else {}
                )
                _raw_metrics["early_stopped"] = last_hist.get(
                    "EARLY_STOPPED", False,
                )
                _raw_metrics["early_stopped_at_iter"] = last_hist.get(
                    "EARLY_STOPPED_AT_ITER", -1,
                )
            except Exception:
                pass
            if hasattr(self.mcts, "_root") and self.mcts._root is not None:
                try:
                    _raw_metrics["tree_node_count"] = int(
                        self.mcts._count_nodes(self.mcts._root),
                    )
                except Exception:
                    _raw_metrics["tree_node_count"] = None
            if self.synthesis_oracle is not None:
                try:
                    rxn_names = list(self.synthesis_oracle.reactions())
                    _raw_metrics["synthesis_equations"] = {
                        rxn_name: str(self.synthesis_oracle.equation(rxn_name))
                        for rxn_name in rxn_names
                    }
                except Exception as exc:
                    log.debug("synthesis_equations record failed: %s", exc)

            # Single sync point — convert every metric to its native
            # Python type.  When the underlying source is a GPU tensor
            # (e.g. ``self.mcts.nfe`` is ever exposed as a tensor)
            # this is the ONLY place the sync happens.
            try:
                _final_metrics: Dict[str, Any] = {
                    "cumulative_nfe": int(
                        _raw_metrics["cumulative_nfe"] or 0,
                    ),
                    "cumulative_nfe_reductions": int(
                        _raw_metrics["cumulative_nfe_reductions"] or 0,
                    ),
                    "cumulative_nfe_oracle": int(
                        _raw_metrics["cumulative_nfe_oracle"] or 0,
                    ),
                    "early_stopped": bool(_raw_metrics["early_stopped"]),
                    "early_stopped_at_iter": int(
                        _raw_metrics["early_stopped_at_iter"],
                    ),
                }
            except Exception:
                _final_metrics = {
                    "cumulative_nfe": 0,
                    "cumulative_nfe_reductions": 0,
                    "cumulative_nfe_oracle": 0,
                    "early_stopped": False,
                    "early_stopped_at_iter": -1,
                }
            if _raw_metrics["tree_node_count"] is not None:
                _final_metrics["tree_node_count"] = int(
                    _raw_metrics["tree_node_count"],
                )
            if _raw_metrics["synthesis_equations"] is not None:
                _final_metrics["synthesis_equations"] = (
                    _raw_metrics["synthesis_equations"]
                )

            # Single dict-construction step — no incremental mutation,
            # no per-key CPU transfer.
            base_dict = rec.to_dict()
            # R3/F4 — PoseBusters outer-gate pass rates.  ``top1`` is
            # the single best SMILES, ``topk`` is the mean of every
            # top-K SMILES emitted this iteration.  Both default to
            # ``0.0`` when PB is unavailable / skipped / failed.
            try:
                pb_top1 = _compute_pb_pass_rate([best_smiles])
            except Exception:
                pb_top1 = 0.0
            try:
                pb_topk = _compute_pb_pass_rate(top_k_smiles_list)
            except Exception:
                pb_topk = 0.0
            # Phase 2 — Vina top-1 binding affinity.  Emit
            # ``vina_real_score_top1`` (kcal/mol) when a real Vina
            # adapter is configured; otherwise emit the
            # ``vina_proxy_score_top1`` placeholder (the
            # ``0.5 + 0.1 * NumRotatableBonds`` proxy from the
            # baselines.py fallback).  Both are ``None`` when neither
            # channel can produce a score (e.g. empty SMILES).
            vina_real_top1, vina_proxy_top1 = _compute_vina_top1(
                best_smiles=best_smiles,
                pocket=pocket,
                best_score=best_score,
                docker=getattr(self, "docker", None),
            )
            results.append({
                **base_dict,
                "synthesis_equations": _final_metrics.get(
                    "synthesis_equations",
                ),
                "closed_loop_iteration_latency_s": float(
                    time.time() - t0_iter,
                ),
                "n_pairs_seen_by_prior": int(pairs_fit_size_history[-1]),
                "dirichlet_reinjected": bool(
                    it > 0 and self.checkpoint_dir is not None,
                ),
                "cumulative_nfe": _final_metrics["cumulative_nfe"],
                "cumulative_nfe_reductions": _final_metrics[
                    "cumulative_nfe_reductions"
                ],
                "cumulative_nfe_oracle": _final_metrics[
                    "cumulative_nfe_oracle"
                ],
                "early_stopped": _final_metrics["early_stopped"],
                "early_stopped_at_iter": _final_metrics[
                    "early_stopped_at_iter"
                ],
                # R3/F4 — PoseBusters outer-gate pass rates.
                "pb_pass_rate_top1": float(pb_top1),
                "pb_pass_rate_topk": float(pb_topk),
                # Phase 2 — Vina binding affinity (real or proxy).
                "vina_real_score_top1": vina_real_top1,
                "vina_proxy_score_top1": vina_proxy_top1,
                **(
                    {"tree_node_count": _final_metrics["tree_node_count"]}
                    if "tree_node_count" in _final_metrics
                    else {}
                ),
            })

            # (L-2) PERSISTENT-TREE CHECKPOINT — save _root +
            # accumulated buffer + SymbolicPrior coefs at end of iter.
            if self.checkpoint_dir:
                try:
                    self._save_checkpoint(iteration=it)
                except Exception as exc:
                    log.debug("save_checkpoint(iter=%d) failed: %s", it, exc)

        # (L-2) Append the per-iteration fit-size trajectory so callers
        # can verify the buffer *grows* across iterations.
        self._pairs_fit_size_history = pairs_fit_size_history
        return results

    # ------------------------------------------------------------------
    # (L-2) Persistent-tree helpers
    # ------------------------------------------------------------------
    def _checkpoint_path(self, iteration: int) -> str:
        """Compute the on-disk checkpoint path for iteration ``i``."""
        import os as _os
        return _os.path.join(
            str(self.checkpoint_dir),
            f"mcts_checkpoint_iter_{int(iteration):04d}.npz",
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """Persist the MCTS state to disk (AlphaZero-style checkpoint)."""
        path = self._checkpoint_path(iteration)
        try:
            self.mcts.save_tree(path)
        except Exception as exc:
            log.debug("save_tree(%s) failed: %s", path, exc)

    def _load_checkpoint(
        self,
        iteration: int,
        *,
        n_simulations: int,
    ) -> Any:
        """Restore the MCTS state from the previous-iteration checkpoint."""
        path = self._checkpoint_path(iteration)
        import os as _os
        if not _os.path.exists(path):
            log.debug("checkpoint %s missing — falling back to current mcts", path)
            return self.mcts
        # Try the official load_tree path; if that fails (e.g. mock
        # MCTS without save/load hooks) fall back to the live object.
        try:
            mcts_cls = type(self.mcts)
            load_fn = getattr(mcts_cls, "load_tree", None)
            if load_fn is None:
                return self.mcts
            loaded = load_fn(
                path,
                tile_library=list(getattr(self.mcts, "tile_library", []) or []),
                rules=dict(getattr(self.mcts, "rules", {}) or {}),
                target_predicates=list(
                    getattr(self.mcts, "target_predicates", []) or [],
                ),
                binding_site=getattr(self.mcts, "binding_site", None),
                scorer=getattr(self.mcts, "scorer", None),
                reward=getattr(self.mcts, "reward", None),
                n_simulations=int(n_simulations),
                c_puct=float(getattr(self.mcts, "c_puct", 1.4)),
                top_k=int(getattr(self.mcts, "top_k", 5)),
                rng=self.rng,
                prior=getattr(self.mcts, "prior", None),
                dirichlet_alpha=float(
                    getattr(self.mcts, "dirichlet_alpha", 0.3),
                ),
                dirichlet_fraction=float(
                    getattr(self.mcts, "dirichlet_fraction", 0.25),
                ),
                rollout_epsilon=float(
                    getattr(self.mcts, "rollout_epsilon", 0.0),
                ),
            )
            return loaded
        except Exception as exc:
            log.debug("load_tree(%s) failed (%s) — falling back", path, exc)
            return self.mcts

    def _refit_symbolic_prior_on_accumulated(self) -> int:
        """Refit :attr:`self.symbolic_reg` on the global pair buffer.

        Returns the number of (features → reward) pairs seen by the
        fit.  Returns 0 when no pairs are available or the fit
        degenerate-skipped.  Also refits ``self.mcts.prior`` when one
        is attached, so the PUCT prior improves alongside the
        SymbolicPrior used by the closed-loop paper-equation.
        """
        accum_pairs = list(
            getattr(self.mcts, "accumulated_leaf_pairs", []) or [],
        )
        if len(accum_pairs) < 2:
            return 0
        X = np.asarray([p[0] for p in accum_pairs], dtype=float)
        y = np.asarray([p[1] for p in accum_pairs], dtype=float)
        if X.ndim != 2 or X.shape[0] < 2 or np.allclose(X, X[0]):
            return 0
        try:
            self.symbolic_reg.fit(X, y)
        except Exception:
            return 0
        # Also refit the MCTS-attached prior so PUCT selection in the
        # next iteration sees the accumulated evidence.
        prior_obj = getattr(self.mcts, "prior", None)
        if prior_obj is not None and hasattr(prior_obj, "fit"):
            try:
                # Reconstruct states via the feature_extractor — we
                # don't have the original molecules, but we can call
                # fit() with empty states list since we already have
                # the X / y matrices; SymbolicPrior.fit internally
                # re-uses its feature_extractor only when states are
                # provided.  We work around by passing placeholder
                # states whose feature_extractor yields the right
                # vector — but that requires real states.  Instead,
                # bypass by injecting the regressor directly.
                from molmetal_lam.lam_chem.pysr_wrapper import (
                    HeuristicRegressor,
                )
                hr = HeuristicRegressor(
                    niterations=int(getattr(prior_obj, "niterations", 1)),
                    random_state=int(getattr(prior_obj, "random_state", 0)),
                )
                hr.fit(X, y)
                prior_obj._model = hr  # type: ignore[attr-defined]
                prior_obj.fitted = True  # type: ignore[attr-defined]
                prior_obj.n_features = int(X.shape[1])  # type: ignore[attr-defined]
            except Exception:
                pass
        return int(X.shape[0])


__all__ = ["LamClickDesignLoop", "extract_paper_equation"]
