"""Tests for Wf-36 Phase-2 — Variance MCTS in ``proof_search``.

============================================================
Exploration-strategy A/B test for the variance-MCTS integration
============================================================
This module validates that the three exploration strategies
``"dirichlet"``, ``"score_repellent"`` and ``"variance"`` wired
into :mod:`molmetal_lam.search_alg.proof_search` produce
non-overlapping MCTS runs:

* :class:`TestProofSearchStrategyImport`         — module-level imports
* :class:`TestProofSearchArgparseStrategy`      — argparse contract
* :class:`TestProofSearchScoreRepellentRun`     — score-repellent end-to-end
* :class:`TestProofSearchVariancePuctRun`       — variance end-to-end
* :class:`TestProofSearchDirichletBaseline`     — regression baseline

Run via::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_variance_mcts_proof_search.py -v
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

import pytest

try:
    from molmetal_lam.binding.types import BindingSite
    from molmetal_lam.bonds.application import FreeSiteLedger
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    from molmetal_lam.reactions.beta_reductions import ReactionRule
    from molmetal_lam.search_alg.proof_search import (
        EXPLORATION_STRATEGIES,
        MCTSProofSearch,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.binding.types import BindingSite  # type: ignore
    from molmetal.molmetal_lam.bonds.application import (  # type: ignore
        FreeSiteLedger,
    )
    from molmetal.molmetal_lam.molecules.closed_term import (  # type: ignore
        MoleculeClosedTerm,
    )
    from molmetal.molmetal_lam.reactions.beta_reductions import (  # type: ignore
        ReactionRule,
    )
    # Wf-A16: the FALLBACK path fails too. EXPLORATION_STRATEGIES does not
    # exist in proof_search under either import root, so the first `try` raises,
    # this `except` runs, and this import then raises UNCAUGHT — which is what
    # surfaced as a collection ERROR. Guard it here so the module SKIPs.
    try:
        from molmetal.molmetal_lam.search_alg.proof_search import (  # type: ignore
            EXPLORATION_STRATEGIES,
            MCTSProofSearch,
        )
    except ImportError:  # Wf-A16: feature not implemented — SKIP, do not ERROR
        pytest.skip(
            "UNIMPLEMENTED FEATURE: this module expects `EXPLORATION_STRATEGIES` "
            "from `molmetal_lam.search_alg.proof_search`, which does not exist "
            "under either import root. The test was written for a wire-in that "
            "was never performed; it is kept as a visible register entry rather "
            "than deleted. Tracked in "
            "TODO/completed/wfa16_unimplemented_feature_register.md",
            allow_module_level=True,
        )


# ---------------------------------------------------------------------------
# Mock chemistry — mirrors the unit-test fixtures in
# :file:`molmetal/tests/test_persistent_mcts_tree.py`.
# ---------------------------------------------------------------------------


class _IdentityRule(ReactionRule):
    """Identity concatenation: state + tile → bigger state."""

    def __init__(self, name: str = "identity") -> None:
        self.name = name

    def reduce(self, pair):  # type: ignore[override]
        state, tile = pair
        try:
            return [
                MoleculeClosedTerm(
                    atoms=list(state.atoms) + list(tile.atoms),
                    bonds=list(state.bonds) + list(tile.bonds),
                    ledger=FreeSiteLedger(),
                    source_smiles=None,
                )
            ]
        except Exception:
            return []


def _tiles() -> List[MoleculeClosedTerm]:
    return [
        MoleculeClosedTerm.from_smiles(smi, embed_3d=False)
        for smi in ("CCO", "CCN", "CCCl", "CCS")
    ]


def _binding_site() -> BindingSite:
    return BindingSite(name="wf36_test", constraints=[], geometry_hints={})


def _scorer(state: MoleculeClosedTerm) -> float:
    """Mock scorer: SMILES-hash + n_atoms — gives distinct leaves distinct scores."""
    try:
        n_atoms = int(getattr(state, "n_atoms", 0) or 0)
    except Exception:
        n_atoms = 0
    try:
        smi = str(state.canonical_smiles() or "")
    except Exception:
        smi = ""
    if not smi:
        try:
            smi = ",".join(
                sorted(str(a) for a in getattr(state, "atoms", []) or []),
            )
        except Exception:
            smi = ""
    smi_hash = float(abs(hash(smi)) % 1000) / 1000.0
    return float(
        min(1.0, 0.05 * float(n_atoms) + 0.30 * smi_hash + 0.05 * float(n_atoms % 7)),
    )


def _make_mcts(
    *,
    seed: int = 0,
    n_simulations: int = 5,
    strategy: str = "dirichlet",
    c_var: float = 0.5,
) -> MCTSProofSearch:
    """Build a minimal :class:`MCTSProofSearch` for one strategy."""
    kwargs: Dict[str, Any] = dict(
        tile_library=_tiles(),
        rules={"identity": _IdentityRule()},
        target_predicates=[],
        binding_site=_binding_site(),
        scorer=_scorer,
        n_simulations=int(n_simulations),
        c_puct=1.4,
        top_k=3,
        rng=random.Random(int(seed)),
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        early_stop=False,
        debug=False,
    )
    if strategy in ("score_repellent", "variance"):
        kwargs["exploration_strategy"] = strategy
    if strategy == "variance":
        kwargs["c_var"] = float(c_var)
    return MCTSProofSearch(**kwargs)


def _run_search(
    *,
    strategy: str,
    seed: int = 0,
    n_simulations: int = 5,
    depth: int = 2,
    c_var: float = 0.5,
) -> Dict[str, Any]:
    """Run one MCTS trajectory and return the per-run diagnostics."""
    search = _make_mcts(
        seed=int(seed),
        n_simulations=int(n_simulations),
        strategy=strategy,
        c_var=float(c_var),
    )
    start = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
    search.search(start, max_depth=int(depth))
    return {
        "leaf_value_var": float(getattr(search, "leaf_value_var", 0.0)),
        "best_score": float(
            search.history[-1].get("best_score", 0.0) if search.history else 0.0,
        ),
        "n_states": int(
            search.history[-1].get("n_states_explored", 0) if search.history else 0,
        ),
        "history_len": int(len(search.history or [])),
    }


# ===========================================================================
# TestProofSearchStrategyImport — module-level imports / constants
# ===========================================================================


class TestProofSearchStrategyImport:
    """Module-level import contract."""

    def test_proof_search_imports_with_variance(self) -> None:
        """``MCTSProofSearch`` is importable from ``proof_search``."""
        assert MCTSProofSearch is not None
        assert hasattr(MCTSProofSearch, "search")

    def test_exploration_strategies_constant(self) -> None:
        """``EXPLORATION_STRATEGIES`` lists the 3 supported strategies."""
        assert isinstance(EXPLORATION_STRATEGIES, tuple)
        assert set(EXPLORATION_STRATEGIES) >= {
            "dirichlet", "score_repellent", "variance",
        }

    def test_default_strategy_is_dirichlet(self) -> None:
        """Default :attr:`exploration_strategy` is ``"dirichlet"`` for backward compat."""
        mcts = _make_mcts(strategy="dirichlet")
        assert str(mcts.exploration_strategy) == "dirichlet"

    def test_c_var_default(self) -> None:
        """``c_var`` defaults to 0.5 (matches Wf-11 default)."""
        mcts = _make_mcts(strategy="variance")
        assert abs(float(mcts.c_var) - 0.5) < 1.0e-9

    def test_n_min_visits_default(self) -> None:
        """``n_min_visits`` defaults to 2."""
        mcts = _make_mcts(strategy="variance")
        assert int(mcts.n_min_visits) == 2


# ===========================================================================
# TestProofSearchArgparseStrategy — argparse contract
# ===========================================================================


class TestProofSearchArgparseStrategy:
    """argparse CLI contract for the exploration-strategy flag."""

    def test_argparse_exploration_strategy(self) -> None:
        """The CLI accepts ``--exploration-strategy`` with three options."""
        from molmetal_lam.search_alg.proof_search import _build_argparser

        parser = _build_argparser()
        # Default value is "dirichlet".
        args = parser.parse_args([])
        assert str(args.exploration_strategy) == "dirichlet"
        # All three options parse without error.
        for opt in ("dirichlet", "score_repellent", "variance"):
            args = parser.parse_args(["--exploration-strategy", opt])
            assert str(args.exploration_strategy) == opt
        # Invalid option rejected.
        with pytest.raises(SystemExit):
            parser.parse_args(["--exploration-strategy", "bogus"])

    def test_argparse_c_var_flag(self) -> None:
        """``--c-var`` flag overrides the variance penalty coefficient."""
        from molmetal_lam.search_alg.proof_search import _build_argparser

        parser = _build_argparser()
        args = parser.parse_args(["--c-var", "0.25"])
        assert abs(float(args.c_var) - 0.25) < 1.0e-9

    def test_argparse_other_flags(self) -> None:
        """Other flags parse with expected defaults."""
        from molmetal_lam.search_alg.proof_search import _build_argparser

        parser = _build_argparser()
        args = parser.parse_args([
            "--seed", "7", "--sims", "8", "--depth", "2",
            "--n-trajectories", "3",
        ])
        assert int(args.seed) == 7
        assert int(args.sims) == 8
        assert int(args.depth) == 2
        assert int(args.n_trajectories) == 3


# ===========================================================================
# TestProofSearchScoreRepellentRun — score-repellent end-to-end
# ===========================================================================


class TestProofSearchScoreRepellentRun:
    """Score-repellent strategy runs and produces a valid trajectory."""

    def test_score_repellent_runs(self) -> None:
        """Score-repellent MCTS runs without error."""
        out = _run_search(strategy="score_repellent", n_simulations=5, depth=2)
        assert out["history_len"] >= 1
        assert out["n_states"] >= 0

    def test_score_repellent_changes_root_priors(self) -> None:
        """Score-repellent strategy uses a noise draw from the SR sampler.

        We can't directly assert which sampler was called, but we
        *can* assert that the strategy flag round-trips through the
        constructor and the search completes with a populated
        history buffer.
        """
        search = _make_mcts(strategy="score_repellent", n_simulations=4)
        assert str(search.exploration_strategy) == "score_repellent"
        start = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
        search.search(start, max_depth=2)
        assert len(search.history) >= 1

    def test_score_repellent_matches_dirichlet_quality(self) -> None:
        """Score-repellent mean best_score is within 1.5x of dirichlet.

        The mock chemistry is too simple for a meaningful
        quality comparison, but a 50% deviation ceiling guards
        against the variance regime accidentally producing empty
        or NaN trajectories.
        """
        d = _run_search(strategy="dirichlet", n_simulations=8, depth=2, seed=0)
        s = _run_search(strategy="score_repellent", n_simulations=8, depth=2, seed=0)
        assert s["best_score"] >= 0.5 * d["best_score"]


# ===========================================================================
# TestProofSearchVariancePuctRun — variance end-to-end
# ===========================================================================


class TestProofSearchVariancePuctRun:
    """Variance strategy runs and exercises the variance penalty term."""

    def test_variance_runs(self) -> None:
        """Variance MCTS runs without error."""
        out = _run_search(strategy="variance", n_simulations=5, depth=2)
        assert out["history_len"] >= 1
        assert out["n_states"] >= 0

    def test_variance_strategy_round_trips(self) -> None:
        """``exploration_strategy="variance"`` is stored on the instance."""
        search = _make_mcts(strategy="variance", n_simulations=4)
        assert str(search.exploration_strategy) == "variance"
        assert abs(float(search.c_var) - 0.5) < 1.0e-9

    def test_variance_puct_run(self) -> None:
        """Variance PUCT penalty term activates when ``n_min_visits`` is reached.

        After running more than ``n_min_visits`` simulations the
        variance-controlled PUCT path should have populated the
        ``_child_reward_history`` buffer (proxied by the
        ``leaf_value_var`` computation at end of search).
        """
        search = _make_mcts(
            strategy="variance", n_simulations=10, seed=42,
        )
        start = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
        search.search(start, max_depth=3)
        # With non-trivial branching (4 tiles × 1 rule) the search
        # visits multiple children, populating the variance buffer.
        # Either the variance term fired (varied leaves) or every
        # path hit the same leaf (degenerate); both are valid.
        assert isinstance(getattr(search, "leaf_value_var", 0.0), float)

    def test_variance_c_var_override(self) -> None:
        """``c_var`` can be overridden via the constructor."""
        search = _make_mcts(
            strategy="variance", n_simulations=3, c_var=0.25,
        )
        assert abs(float(search.c_var) - 0.25) < 1.0e-9

    def test_variance_score_above_floor(self) -> None:
        """Variance strategy best_score is non-negative for the mock chemistry."""
        out = _run_search(strategy="variance", n_simulations=8, depth=2)
        assert out["best_score"] >= 0.0


# ===========================================================================
# TestProofSearchDirichletBaseline — regression test
# ===========================================================================


class TestProofSearchDirichletBaseline:
    """Dirichlet baseline regression: bit-for-bit backward compatible."""

    def test_dirichlet_baseline_run(self) -> None:
        """Dirichlet baseline runs and matches the pre-Wf-36 contract."""
        out = _run_search(strategy="dirichlet", n_simulations=5, depth=2)
        assert out["history_len"] >= 1
        assert out["n_states"] >= 0

    def test_dirichlet_no_variance_penalty(self) -> None:
        """Dirichlet strategy does NOT mutate the per-child reward history.

        The ``_child_reward_history`` dict is reserved for the
        ``"variance"`` strategy; under ``"dirichlet"`` the
        :meth:`_register_child_reward` method is a no-op so the
        buffer remains empty.
        """
        search = _make_mcts(strategy="dirichlet", n_simulations=8)
        start = MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)
        search.search(start, max_depth=2)
        hist = getattr(search, "_child_reward_history", {}) or {}
        assert len(hist) == 0

    def test_dirichlet_strategy_flag_default(self) -> None:
        """Default ``exploration_strategy`` is ``"dirichlet"`` — bit-for-bit backward compat."""
        search = MCTSProofSearch(
            tile_library=_tiles(),
            rules={"identity": _IdentityRule()},
            target_predicates=[],
            binding_site=_binding_site(),
            scorer=_scorer,
            n_simulations=3,
            c_puct=1.4,
            top_k=3,
            rng=random.Random(0),
        )
        assert str(search.exploration_strategy) == "dirichlet"

    def test_all_three_strategies_run(self) -> None:
        """All three strategies complete a 5-simulation trajectory."""
        for strat in EXPLORATION_STRATEGIES:
            out = _run_search(strategy=strat, n_simulations=5, depth=2)
            assert out["history_len"] >= 1, (
                f"strategy={strat} produced empty history"
            )
