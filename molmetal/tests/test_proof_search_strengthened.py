"""Tests for the strengthened MCTS proof search.

Covers the four new pieces added on top of the original stub:

* :class:`SymbolicPrior`     — fit / predict / sigmoid-mapping
* :class:`RewardAggregator`  — multi-reward merging, SA inversion,
                                Vina negation, type / binder bonuses
* AlphaZero-style Dirichlet noise at the root — deterministic seed
  ⇒ same best child as the baseline (uniform-prior) run
* End-to-end MCTS with the new pieces wired in (smoke test)

These tests are designed to be run on a fresh checkout without
requiring Vina / SA / PoseBusters backends — every reward channel
is mocked with a simple scalar function.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import pytest

from molmetal_lam.binding.types import BindingSite, KINASE_ATP, typecheck
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.reactions.beta_reductions import ReactionRule
from molmetal_lam.search_alg.proof_search import (
    MCTS,
    MCTSProofSearch,
    RewardAggregator,
    SymbolicPrior,
    VirtualLoss,
    heuristic,
)
from molmetal_lam.types.predicates import LIPINSKI, TypePredicate


# ---------------------------------------------------------------------------
# Test fixtures — minimal mock reaction rule + binding site
# ---------------------------------------------------------------------------


class _IdentityRule(ReactionRule):
    """Trivial reaction rule used in tests: concatenates atoms.

    Every (state, tile) pair produces a single product whose atom list
    is ``state.atoms + tile.atoms``.  This gives the MCTS enough
    branching to exercise the selection / expansion code without
    pulling in real chemistry.
    """

    def __init__(self, name: str = "identity") -> None:
        self.name = name

    def reduce(self, pair):  # type: ignore[override]
        state, tile = pair
        try:
            from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
            from molmetal_lam.bonds.application import FreeSiteLedger
            new_atoms = list(state.atoms) + list(tile.atoms)
            return [
                MoleculeClosedTerm(
                    atoms=new_atoms,
                    bonds=list(state.bonds) + list(tile.bonds),
                    ledger=FreeSiteLedger(),
                    source_smiles=None,
                )
            ]
        except Exception:
            return []

    def __repr__(self) -> str:  # pragma: no cover
        return f"_IdentityRule({self.name!r})"


def _tiles() -> List[MoleculeClosedTerm]:
    """Tiny tile library: 4 distinct small molecules."""
    return [
        MoleculeClosedTerm.from_smiles(smi)
        for smi in ("CCO", "CCN", "CCCl", "CCS")
    ]


def _binding_site() -> BindingSite:
    """Empty BindingSite — type-check will fall back to property preds."""
    return BindingSite(name="empty_site", constraints=[], geometry_hints={})


def _rules() -> dict:
    return {"identity": _IdentityRule()}


# ---------------------------------------------------------------------------
# (i) SymbolicPrior: fit + predict
# ---------------------------------------------------------------------------


class TestSymbolicPrior:
    """``SymbolicPrior`` must round-trip fit / predict."""

    def test_unfitted_returns_half(self) -> None:
        p = SymbolicPrior()
        s = MoleculeClosedTerm.from_smiles("CCO")
        assert p.fitted is False
        assert 0.0 < p.predict_proba(s) < 1.0
        assert abs(p.predict_proba(s) - 0.5) < 1e-6
        assert p.equation() == "<unfitted>"

    def test_fit_then_predict_proba_in_unit_interval(self) -> None:
        # Synthetic dataset: score = n_atoms (the first feature).
        tiles = _tiles()
        X_states = []
        y = []
        rng = np.random.default_rng(0)
        # Build 8 pseudo-molecules by hand so the feature vector
        # actually varies.
        for k in range(8):
            atoms = []
            for _ in range(2 + k):
                from molmetal_lam.atoms.combinators import PRIMITIVE_ATOMS
                from molmetal_lam.bonds.application import distinct
                atoms.append(distinct(PRIMITIVE_ATOMS["C"]))
            from molmetal_lam.bonds.application import FreeSiteLedger
            from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
            s = MoleculeClosedTerm(
                atoms=atoms, bonds=[], ledger=FreeSiteLedger(),
            )
            X_states.append(s)
            y.append(float(len(atoms)))
        prior = SymbolicPrior(random_state=0, niterations=1)
        prior.fit(X_states, y)
        assert prior.fitted is True
        assert prior.n_features == 4
        for s in X_states:
            prob = prior.predict_proba(s)
            assert 0.0 < prob < 1.0, f"proba out of range: {prob}"

    def test_fit_with_empty_states_is_noop(self) -> None:
        p = SymbolicPrior()
        p.fit([], [])
        assert p.fitted is False
        s = MoleculeClosedTerm.from_smiles("CCO")
        assert abs(p.predict_proba(s) - 0.5) < 1e-6

    def test_equation_after_fit(self) -> None:
        s = MoleculeClosedTerm.from_smiles("CCO")
        prior = SymbolicPrior(random_state=0, niterations=1)
        prior.fit([s] * 5, [0.0, 0.1, 0.2, 0.3, 0.4])
        eq = prior.equation()
        assert eq != "<unfitted>"
        # sklearn fallback will yield either a linear formula or an
        # ``RF(top3: ...)`` description — both are non-trivial.
        assert isinstance(eq, str)

    def test_predict_value_is_finite_after_fit(self) -> None:
        s = MoleculeClosedTerm.from_smiles("CCO")
        prior = SymbolicPrior(random_state=0, niterations=1)
        prior.fit([s] * 5, [0.0, 0.1, 0.2, 0.3, 0.4])
        v = prior.predict_value(s)
        assert np.isfinite(v)


# ---------------------------------------------------------------------------
# (ii) RewardAggregator: multi-reward merge is correct
# ---------------------------------------------------------------------------


class TestRewardAggregator:
    """The weighted reward combination must obey the documented contract."""

    def test_from_scorer_matches_scalar(self) -> None:
        def scorer(state):
            return 0.7
        agg = RewardAggregator.from_scorer(scorer, weight=2.0)
        s = MoleculeClosedTerm.from_smiles("CCO")
        # Single channel with weight 2.0 ⇒ raw 0.7 * 2.0 = 1.4
        v = agg(s, satisfies_typed=False, binds_target=False)
        assert abs(v - 1.4) < 1e-6

    def test_vina_invert_default(self) -> None:
        # Vina returns -9.0 kcal/mol (good binder); without invert ⇒
        # -9.0, with invert ⇒ +9.0.
        agg_in = RewardAggregator(r_vina=lambda s: -9.0, w_vina=1.0)
        agg_out = RewardAggregator(
            r_vina=lambda s: -9.0, w_vina=1.0, vina_invert=False,
        )
        s = MoleculeClosedTerm.from_smiles("CCO")
        assert agg_in(s, satisfies_typed=False, binds_target=False) > 0
        assert agg_out(s, satisfies_typed=False, binds_target=False) < 0

    def test_sa_inverted_to_unit_interval(self) -> None:
        # SA=1  (trivial)  → 1.0  ; SA=10 (impossible) → 0.0
        agg = RewardAggregator(r_sa=lambda s: 1.0, w_sa=1.0)
        s = MoleculeClosedTerm.from_smiles("CCO")
        v1 = agg(s, satisfies_typed=False, binds_target=False)
        agg2 = RewardAggregator(r_sa=lambda s: 10.0, w_sa=1.0)
        v10 = agg2(s, satisfies_typed=False, binds_target=False)
        assert abs(v1 - 1.0) < 1e-6
        assert abs(v10 - 0.0) < 1e-6

    def test_bonuses_applied(self) -> None:
        agg = RewardAggregator(bonus_typed=0.5, bonus_binder=0.3)
        s = MoleculeClosedTerm.from_smiles("CCO")
        v_neither = agg(s, satisfies_typed=False, binds_target=False)
        v_typed = agg(s, satisfies_typed=True, binds_target=False)
        v_both = agg(s, satisfies_typed=True, binds_target=True)
        assert abs(v_typed - v_neither - 0.5) < 1e-6
        assert abs(v_both - v_typed - 0.3) < 1e-6

    def test_failing_channel_returns_zero(self) -> None:
        # A channel that raises must not crash the aggregator.
        def bad(state):
            raise RuntimeError("backend down")
        agg = RewardAggregator(r_vina=bad, w_vina=1.0)
        s = MoleculeClosedTerm.from_smiles("CCO")
        v = agg(s, satisfies_typed=False, binds_target=False)
        assert v == 0.0

    def test_weighted_sum(self) -> None:
        agg = RewardAggregator(
            r_vina=lambda s: 5.0, w_vina=2.0, vina_invert=False,
            r_sa=lambda s: 4.0, w_sa=0.5,
            r_posebusters=lambda s: 0.8, w_posebusters=1.0,
            r_pic50=lambda s: 7.5, w_pic50=0.4,
            r_retro=lambda s: 0.9, w_retro=1.0,
            bonus_typed=0.0, bonus_binder=0.0,
        )
        s = MoleculeClosedTerm.from_smiles("CCO")
        # vina_invert=False so we keep 5.0
        # sa=4 → 1 - (4-1)/9 = 0.6667
        # posebusters=0.8
        # pic50=7.5
        # retro=0.9
        expected = (
            2.0 * 5.0
            + 0.5 * (1.0 - 3.0 / 9.0)
            + 1.0 * 0.8
            + 0.4 * 7.5
            + 1.0 * 0.9
        )
        v = agg(s, satisfies_typed=False, binds_target=False)
        assert abs(v - expected) < 1e-6


# ---------------------------------------------------------------------------
# (iii) Dirichlet noise: deterministic ⇒ same best child as baseline
# ---------------------------------------------------------------------------


class TestDirichletDeterminism:
    """With a deterministic seed the Dirichlet root prior must reproduce
    the same argmax child as the unperturbed baseline.
    """

    def _run(self, *, dirichlet_alpha: float, dirichlet_frac: float, seed: int,
             state_scorer: bool = False):
        rng = random.Random(seed)
        if state_scorer:
            # A scorer that depends on the state, so the rollout value
            # differs across children — this makes the Dirichlet
            # perturbation visible in the per-iteration diagnostics.
            scorer = lambda s: float(s.n_atoms) / 10.0
        else:
            scorer = lambda s: 0.5
        mcts = MCTSProofSearch(
            tile_library=_tiles(),
            rules=_rules(),
            target_predicates=[],
            binding_site=_binding_site(),
            scorer=scorer,
            n_simulations=8,
            c_puct=1.4,
            top_k=3,
            rng=rng,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_fraction=dirichlet_frac,
        )
        initial = MoleculeClosedTerm.from_smiles("CCO")
        mcts.search(initial, max_depth=3)
        return mcts

    def test_zero_noise_matches_baseline(self) -> None:
        """alpha=0 ⇒ no noise ⇒ root priors are the constant 0.5 stub.

        With ``dirichlet_alpha=0`` the Dirichlet draw is a degenerate
        point mass at the all-ones direction → the noise step is a
        no-op, so the tree topology and history of best/mean scores
        must coincide exactly with the unperturbed baseline.
        """
        a = self._run(dirichlet_alpha=0.0, dirichlet_frac=0.25, seed=0)
        b = self._run(dirichlet_alpha=0.0, dirichlet_frac=0.0, seed=0)
        assert len(a.history) == len(b.history)
        for ha, hb in zip(a.history, b.history):
            assert ha["best_score"] == hb["best_score"]
            assert ha["mean_score"] == hb["mean_score"]
            assert ha["n_states_explored"] == hb["n_states_explored"]

    def test_noise_with_same_seed_is_reproducible(self) -> None:
        """Two runs with the same seed must produce identical histories."""
        a = self._run(dirichlet_alpha=0.3, dirichlet_frac=0.25, seed=42)
        b = self._run(dirichlet_alpha=0.3, dirichlet_frac=0.25, seed=42)
        assert len(a.history) == len(b.history)
        for ha, hb in zip(a.history, b.history):
            assert ha["best_score"] == hb["best_score"]
            assert ha["mean_score"] == hb["mean_score"]
            assert ha["n_states_explored"] == hb["n_states_explored"]

    def test_noise_with_different_seed_differs(self) -> None:
        """Sanity-check that Dirichlet draws depend on the RNG seed.

        Two distinct seeds must yield at least one distinct Dirichlet
        draw — verified by inspecting the root's children priors after
        the first iteration of an artificial single-iteration run.

        Why inspect P directly?  Because the test rig above uses
        alpha-equivalent children (all of which collapse to the same
        canonical SMILES after the first dedup) plus a noisy rollout,
        so the *per-iteration diagnostics* may coincide even when the
        Dirichlet draw is genuinely different.  Inspecting the priors
        is a sharper probe.
        """
        # Build a single-simulation MCTS so we can read the root
        # children's priors directly (the tree is tiny).
        def build(seed: int):
            rng = random.Random(seed)
            mcts = MCTSProofSearch(
                tile_library=_tiles(),
                rules=_rules(),
                target_predicates=[],
                binding_site=_binding_site(),
                scorer=lambda s: float(s.n_atoms) / 10.0,
                n_simulations=2,
                c_puct=1.4,
                top_k=3,
                rng=rng,
                dirichlet_alpha=0.3,
                dirichlet_fraction=0.5,  # large noise so the effect is visible
            )
            initial = MoleculeClosedTerm.from_smiles("CCO")
            mcts.search(initial, max_depth=2)
            return mcts

        a = build(1)
        b = build(2)
        # Read the root's children's priors after one simulation.
        # We don't expose root via search(); rebuild the same root
        # object to inspect P directly by re-running _apply_dirichlet.
        from molmetal_lam.search_alg.proof_search import _MCTSNode
        # Simpler: just re-run the Dirichlet step on a fresh root.
        def root_priors(mcts) -> list:
            r = _MCTSNode(state=MoleculeClosedTerm.from_smiles("CCO"))
            children = mcts._expand(r.state)
            mcts._apply_dirichlet_to_root(r, children)
            return [c.P for c in r.children]
        pa = root_priors(a)
        pb = root_priors(b)
        # The priors must (i) include values different from the
        # constant 0.5 baseline and (ii) differ between seeds.
        assert any(abs(p - 0.5) > 1e-3 for p in pa), (
            f"Dirichlet did not perturb root priors: {pa}"
        )
        assert pa != pb, (
            "Dirichlet draws identical for two different seeds — "
            "RNG-draw bug."
        )


# ---------------------------------------------------------------------------
# (iv) End-to-end smoke test with all pieces wired in
# ---------------------------------------------------------------------------


class TestEndToEndWiring:
    """Verify the new pieces integrate without crashing the search loop."""

    def test_run_with_fitted_prior_and_multi_reward(self) -> None:
        rng = random.Random(7)
        prior = SymbolicPrior(random_state=0, niterations=1)
        s_tile = MoleculeClosedTerm.from_smiles("CCO")
        prior.fit([s_tile] * 5, [0.1, 0.2, 0.3, 0.4, 0.5])
        reward = RewardAggregator(
            r_vina=lambda s: -7.0,
            r_sa=lambda s: 3.0,
            r_posebusters=lambda s: 0.9,
            r_pic50=lambda s: 7.5,
            r_retro=lambda s: 0.85,
            w_vina=0.5, w_sa=0.2, w_posebusters=0.1, w_pic50=0.1, w_retro=0.1,
            bonus_typed=0.5, bonus_binder=0.5,
        )
        mcts = MCTSProofSearch(
            tile_library=_tiles(),
            rules=_rules(),
            target_predicates=[],
            binding_site=_binding_site(),
            reward=reward,
            prior=prior,
            n_simulations=5,
            c_puct=1.4,
            top_k=2,
            rng=rng,
            dirichlet_alpha=0.3,
            dirichlet_fraction=0.25,
            rollout_epsilon=0.1,
        )
        initial = MoleculeClosedTerm.from_smiles("CCO")
        cands = mcts.search(initial, max_depth=2)
        # We expect at most ``top_k`` candidates and a non-empty history.
        assert len(cands) <= 2
        assert len(mcts.history) == 5
        for h in mcts.history:
            assert "best_score" in h
            assert "n_states_explored" in h

    def test_backward_compat_with_only_scorer(self) -> None:
        """Passing only ``scorer`` (the historical interface) must work."""
        rng = random.Random(0)
        mcts = MCTSProofSearch(
            tile_library=_tiles(),
            rules=_rules(),
            target_predicates=[],
            binding_site=_binding_site(),
            scorer=lambda s: 0.42,
            n_simulations=3,
            c_puct=1.4,
            top_k=2,
            rng=rng,
        )
        initial = MoleculeClosedTerm.from_smiles("CCO")
        cands = mcts.search(initial, max_depth=2)
        assert isinstance(cands, list)
        assert len(mcts.history) == 3

    def test_alias_MCTS_resolves(self) -> None:
        assert MCTS is MCTSProofSearch

    def test_heuristic_stub_returns_half(self) -> None:
        assert heuristic([1, 2, 3]) == 0.5
        assert heuristic("any input") == 0.5

    def test_virtual_loss_dataclass(self) -> None:
        vl = VirtualLoss(delta=2.0)
        assert vl.delta == 2.0
        vl2 = VirtualLoss()
        assert vl2.delta == 1.0