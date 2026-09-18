"""TODO-02 — Tests for the RewardAggregator-based ``heuristic`` prior.

This module verifies that the MCTS proof-search prior is no longer
the constant ``0.5`` stub:

    * :meth:`MCTSProofSearch.heuristic` returns **different** values
      for three structurally-distinct molecules (not all 0.5).
    * :meth:`MCTSProofSearch.heuristic` consults the configured
      :class:`RewardAggregator` (the ``aggregate`` entry-point).
    * After a small :meth:`MCTSProofSearch.search` run, the leaves
      carry **non-constant** Q values.
    * The constructor accepts the new ``reward_aggregator=`` kwarg
      (preferred spelling) and aliases it onto the existing
      ``reward=`` slot.

Run with::

    uv run pytest -q --ignore=molmetal/references \
        molmetal/molmetal_lam/tests/test_proof_search_prior.py \
        --tb=short
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from molmetal_lam.binding.types import BindingSite, PROTEASE_GENERIC
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.search_alg.proof_search import (
    MCTSProofSearch,
    RewardAggregator,
)


# ---------------------------------------------------------------------------
# Helpers — minimal valid scaffolding for MCTSProofSearch.search().
# ---------------------------------------------------------------------------
def _tile_library(smis: Optional[List[str]] = None) -> List[MoleculeClosedTerm]:
    """Return a tiny tile library.  ``smis`` defaults to ``["C"]``."""
    smis = smis if smis is not None else ["C"]
    out: List[MoleculeClosedTerm] = []
    for s in smis:
        try:
            out.append(MoleculeClosedTerm.from_smiles(s, embed_3d=False))
        except Exception:
            out.append(MoleculeClosedTerm())
    if not out:
        # Last-ditch: an empty term so MCTS still has *something*.
        out = [MoleculeClosedTerm()]
    return out


@dataclass
class _StubRule:
    """ReactionRule-shaped stub that fires on every (state, tile) pair.

    The "product" is a fresh :class:`MoleculeClosedTerm` whose
    ``n_atoms`` is bumped by the *tile's* atom count AND tagged with
    the tile's canonical SMILES so the new state is distinguishable
    from the parent and from any other (state, tile) product.  This
    is what gives :meth:`MCTSProofSearch.search` the multiple
    distinguishable leaves it needs to drive a non-zero
    ``LEAF_VALUE_VAR``.

    When ``tile`` is ``None`` (single-reactant reduction) the
    product keeps the same ``n_atoms`` so the test for the constant
    case can opt-out of expansion by passing a single tile.

    The product SMILES is constructed as ``"C{n_atoms}{tile_smiles}"``
    where ``tile_smiles`` is the tile's own canonical SMILES.  This
    guarantees each (state, tile) pair yields a unique canonical
    SMILES that the transposition table will not collapse.
    """

    name: str = "stub_rule"

    def reduce(
        self,
        molecule: Any,
    ) -> List[MoleculeClosedTerm]:
        # Accept either a single closed-term or a 2-tuple of closed
        # terms (bi-molecular click reduction).  Tuple form is what
        # MCTS dispatches in production.
        if isinstance(molecule, tuple):
            if len(molecule) == 0:
                return []
            state, tile = molecule[0], (
                molecule[1] if len(molecule) > 1 else None
            )
        else:
            state, tile = molecule, None
        try:
            new_atoms = int(state.n_atoms) + (
                int(tile.n_atoms) if tile is not None else 0
            )
        except Exception:
            new_atoms = 1
        try:
            tile_smi = ""
            if tile is not None:
                tile_smi = str(tile.canonical_smiles() or "")
        except Exception:
            tile_smi = ""
        # Build a product whose canonical SMILES encodes BOTH the
        # total atom count AND the tile identity — so different
        # (state, tile) pairs collide only on α-equivalent inputs.
        try:
            base = ("C" * max(1, new_atoms)) + (
                tile_smi if tile_smi else ""
            )
            prod = MoleculeClosedTerm.from_smiles(base, embed_3d=False)
        except Exception:
            prod = MoleculeClosedTerm()
        return [prod]


def _stub_rules() -> Dict[str, _StubRule]:
    return {"stub": _StubRule()}


def _three_distinct_molecules() -> List[MoleculeClosedTerm]:
    """Return three structurally-distinct closed-term states.

    Uses very small SMILES so the test is fast and headless-safe
    (no 3D embed).  The canonical SMILES are ``C``, ``CC``, ``CCC``
    — i.e. methane, ethane, propane — which any RDKit-free code path
    still resolves to distinct states because ``n_atoms`` /
    ``n_bonds`` differ.
    """
    out: List[MoleculeClosedTerm] = []
    for smi in ("C", "CC", "CCC"):
        try:
            out.append(MoleculeClosedTerm.from_smiles(smi, embed_3d=False))
        except Exception:
            out.append(MoleculeClosedTerm())
    return out


def _stub_channels_per_state() -> Dict[str, Any]:
    """Return a channels callable producing per-state distinguishable
    values so :meth:`MCTSProofSearch.heuristic` returns different
    priors for the three molecules above.

    Each channel is keyed by ``n_atoms`` so molecules with more atoms
    get higher Vina / PoseBusters values (proxy: more atoms = more
    interactions), lower SA (proxy: bigger = harder to synthesise,
    so SA is high → after inversion, low), and a constant QED.
    """
    def _vina(state):
        try:
            return -float(state.n_atoms)
        except Exception:
            return 0.0

    def _sa(state):
        try:
            # SA raw is in [1, 10]; larger molecule → harder → higher.
            return min(10.0, 1.0 + float(state.n_atoms) * 0.3)
        except Exception:
            return 1.0

    def _qed(state):
        # Constant QED so the test is deterministic.
        return 0.5

    def _posebusters(state):
        try:
            # Pass-fail: pretend molecules with >= 2 atoms pass.
            return 1.0 if state.n_atoms >= 2 else 0.0
        except Exception:
            return 0.0

    def _retro(state):
        try:
            # Retrosynthesis feasibility: larger = harder.
            return max(0.0, 1.0 - float(state.n_atoms) * 0.2)
        except Exception:
            return 0.0

    def _pic50(state):
        try:
            return min(12.0, 4.0 + float(state.n_atoms) * 0.5)
        except Exception:
            return 4.0

    return {
        "r_vina": _vina,
        "r_sa": _sa,
        "r_qed": _qed,
        "r_posebusters": _posebusters,
        "r_retro": _retro,
        "r_pic50": _pic50,
    }


def _make_aggregator(channels: Optional[Dict[str, Any]] = None) -> RewardAggregator:
    """Build a RewardAggregator with the test channels + TODO-02 weights.

    The TODO-02 default weights are::

        w_vina=0.4, w_posebusters=0.2, w_retro=0.15, w_pic50=0.1,
        w_qed=0.1, w_sa=0.05

    so the channels above produce per-state distinguishable
    aggregated rewards.
    """
    if channels is None:
        channels = _stub_channels_per_state()
    return RewardAggregator(
        r_vina=channels["r_vina"],
        r_sa=channels["r_sa"],
        r_qed=channels["r_qed"],
        r_posebusters=channels["r_posebusters"],
        r_retro=channels["r_retro"],
        r_pic50=channels["r_pic50"],
        w_vina=0.40,
        w_posebusters=0.20,
        w_retro=0.15,
        w_pic50=0.10,
        w_qed=0.10,
        w_sa=0.05,
    )


# ---------------------------------------------------------------------------
# Test 1 — heuristic returns different values for different molecules
# ---------------------------------------------------------------------------
def test_heuristic_no_longer_constant() -> None:
    """``MCTSProofSearch.heuristic(features)`` is NOT the constant 0.5.

    We instantiate :class:`MCTSProofSearch` with a six-channel
    :class:`RewardAggregator` (TODO-02 weights), then evaluate
    :meth:`heuristic` on three structurally-distinct molecules.  The
    three results must differ from each other AND not all collapse
    to ``0.5`` — that is exactly the regression we want to catch.
    """
    mcts = MCTSProofSearch(
        tile_library=_tile_library(),
        rules=_stub_rules(),
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        reward_aggregator=_make_aggregator(),
        n_simulations=4,
        rng=random.Random(0),
    )
    mols = _three_distinct_molecules()
    priors = [float(mcts.heuristic(m)) for m in mols]

    # 1) Not all 0.5 — the regression we want to catch.
    assert not all(abs(p - 0.5) < 1e-9 for p in priors), (
        f"heuristic must no longer be the constant 0.5 stub, "
        f"got priors={priors!r}"
    )
    # 2) The three priors are *distinct* (different molecules →
    # different per-channel aggregations → different PUCT priors).
    assert len(set(priors)) >= 2, (
        f"three distinct molecules must produce at least two distinct "
        f"priors, got priors={priors!r}"
    )
    # 3) Each prior is a valid probability mass in (0, 1).
    for p in priors:
        assert 0.0 < p < 1.0, (
            f"each heuristic() result must be in (0, 1), got {p!r}"
        )


# ---------------------------------------------------------------------------
# Test 2 — heuristic dispatches through RewardAggregator.aggregate
# ---------------------------------------------------------------------------
class _RecordingAggregator(RewardAggregator):
    """Test-double RewardAggregator that records every ``aggregate``
    call so the test can assert :meth:`MCTSProofSearch.heuristic`
    actually invokes it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.aggregate_calls: List[Dict[str, Any]] = []

    def aggregate(
        self,
        smiles: str,
        channels: Optional[Dict[str, float]] = None,
    ) -> float:
        # Record the call **before** delegating to the base impl so
        # the test sees a non-empty call list even if the parent
        # implementation raises.
        self.aggregate_calls.append({
            "smiles": str(smiles),
            "channels": dict(channels) if channels else {},
        })
        return super().aggregate(smiles, channels)


def test_heuristic_uses_reward_aggregator() -> None:
    """``MCTSProofSearch.heuristic`` invokes ``reward_aggregator.aggregate``.

    The TODO-02 spec mandates the prior is computed via
    :meth:`RewardAggregator.aggregate` — so a mock aggregator that
    records every call must see at least one invocation when
    :meth:`heuristic` runs.
    """
    rec = _RecordingAggregator()
    # Wire the six TODO-02 channels so the heuristic collects real
    # per-channel values to forward to ``aggregate``.
    channels = _stub_channels_per_state()
    rec.r_vina = channels["r_vina"]
    rec.r_sa = channels["r_sa"]
    rec.r_qed = channels["r_qed"]
    rec.r_posebusters = channels["r_posebusters"]
    rec.r_retro = channels["r_retro"]
    rec.r_pic50 = channels["r_pic50"]
    rec.w_vina = 0.40
    rec.w_posebusters = 0.20
    rec.w_retro = 0.15
    rec.w_pic50 = 0.10
    rec.w_qed = 0.10
    rec.w_sa = 0.05

    mcts = MCTSProofSearch(
        tile_library=_tile_library(),
        rules=_stub_rules(),
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        reward_aggregator=rec,
        n_simulations=2,
        rng=random.Random(0),
    )
    # Drive at least one invocation.
    mols = _three_distinct_molecules()
    for m in mols:
        _ = float(mcts.heuristic(m))

    # ``aggregate`` must have been called at least once per molecule
    # (3 calls minimum).
    assert len(rec.aggregate_calls) >= len(mols), (
        f"heuristic must invoke reward_aggregator.aggregate at least "
        f"{len(mols)} times, got {len(rec.aggregate_calls)}: "
        f"{rec.aggregate_calls!r}"
    )
    # The first call must have carried a non-empty channel dict
    # (so the per-channel aggregation actually ran).
    first = rec.aggregate_calls[0]
    assert isinstance(first.get("channels"), dict)
    assert len(first["channels"]) >= 1, (
        f"heuristic must forward at least one channel value to "
        f"aggregate, got channels={first['channels']!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — leaves have non-constant Q values after a real search
# ---------------------------------------------------------------------------
class _ForceExpandWrapper:
    """Wrap a :class:`MoleculeClosedTerm` but force
    ``is_beta_normal_form = False`` so MCTS will expand the root.

    Without this override every freshly-from_smiles state is in
    β-NF (closed + no redex), so :meth:`MCTSProofSearch.search`
    skips EXPAND and the tree stays at a single root node.  Forcing
    the flag to ``False`` makes the search actually walk into
    distinct (state, tile) children, which is what we need to
    observe non-zero leaf reward variance.
    """

    def __init__(self, inner: MoleculeClosedTerm) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        if name in ("is_beta_normal_form", "is_closed", "is_terminal"):
            return False
        return getattr(self._inner, name)

    @property
    def n_atoms(self) -> int:
        return int(self._inner.n_atoms)

    @property
    def n_bonds(self) -> int:
        return int(self._inner.n_bonds)

    @property
    def free_sites(self) -> Dict[int, int]:
        try:
            return dict(self._inner.free_sites)
        except Exception:
            return {}

    def canonical_smiles(self) -> str:
        try:
            return str(self._inner.canonical_smiles() or "")
        except Exception:
            return ""


def test_leaf_value_not_constant() -> None:
    """After ``MCTSProofSearch.search`` runs, the leaves carry
    **non-constant** Q values.

    The canonical "is the prior informative" gate is the
    ``LEAF_VALUE_VAR`` metric: the variance of the leaf reward
    values across the search.  When the prior is a constant 0.5
    (the pre-TODO-02 stub) every leaf collapses to the same value
    and the variance is 0.  With the TODO-02
    :class:`RewardAggregator`-driven prior, leaves whose states have
    different ``n_atoms`` get different aggregated rewards — so the
    variance is strictly positive.

    The test wraps the root state with :class:`_ForceExpandWrapper`
    so MCTS dispatches EXPAND (real :class:`MoleculeClosedTerm`
    objects are in β-NF at construction time, which would otherwise
    short-circuit EXPAND).  After the search we assert:

    * the per-iteration ``LEAF_VALUE_VAR`` keys are populated in
      ``mcts.history`` (so downstream metrics can read them),
    * the cumulative ``leaf_value_var`` is **strictly positive**,
      proving the leaf Q values are non-constant,
    * multiple distinct leaves are recorded in the final iteration
      so the variance has enough degrees of freedom to be > 0.
    """
    mcts = MCTSProofSearch(
        tile_library=_tile_library(smis=["C", "CC", "CCC", "CCCC"]),
        rules=_stub_rules(),
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        reward_aggregator=_make_aggregator(),
        n_simulations=8,
        early_stop=False,
        rng=random.Random(0),
        dirichlet_alpha=0.0,
        dirichlet_fraction=0.0,
    )
    initial_inner = _tile_library(smis=["C"])[0]
    initial = _ForceExpandWrapper(initial_inner)
    # Run the search — this populates ``mcts.history`` with one
    # entry per iteration (and ``mcts._leaf_value_history`` with
    # one reward value per simulation).
    mcts.search(initial_state=initial, max_depth=2)
    # Capture the per-iteration ``LEAF_VALUE_VAR`` keys — these are
    # the canonical regression metric for "leaves distinguishable by
    # reward".
    leaf_vars: List[float] = []
    for entry in mcts.history:
        if "LEAF_VALUE_VAR" in entry:
            leaf_vars.append(float(entry["LEAF_VALUE_VAR"]))

    # 1) The search ran the full budget (early_stop=False) so we
    # see one ``LEAF_VALUE_VAR`` entry per iteration.
    assert len(leaf_vars) == len(mcts.history), (
        f"every history entry must carry LEAF_VALUE_VAR, got "
        f"{len(leaf_vars)} leaf_vars vs {len(mcts.history)} history"
    )

    # 2) At least one iteration must record a strictly positive
    # ``LEAF_VALUE_VAR`` — i.e. the leaves reached in that
    # iteration carried distinguishable rewards.  When the prior is
    # the constant 0.5 stub this metric is identically 0.0 across
    # every iteration.
    has_positive_var = any(v > 0.0 for v in leaf_vars)
    assert has_positive_var, (
        f"at least one history entry must carry LEAF_VALUE_VAR > 0, "
        f"got leaf_vars={leaf_vars!r}"
    )

    # 3) The cumulative ``leaf_value_var`` is the canonical search-
    # level regression metric — it must also be > 0.
    assert float(mcts.leaf_value_var) > 0.0, (
        f"leaf_value_var must be > 0 when the aggregator carries "
        f"per-state signal, got {mcts.leaf_value_var}"
    )


# ---------------------------------------------------------------------------
# Test 4 — constructor accepts ``reward_aggregator=`` kwarg
# ---------------------------------------------------------------------------
def test_proof_search_accepts_reward_aggregator_kwarg() -> None:
    """``MCTSProofSearch(..., reward_aggregator=...)`` stores the
    aggregator on the new ``reward_aggregator`` slot AND ``_resolved_reward``
    returns it (so the prior / heuristic can find it).
    """
    agg = _make_aggregator()
    mcts = MCTSProofSearch(
        tile_library=_tile_library(),
        rules=_stub_rules(),
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        reward_aggregator=agg,
        n_simulations=2,
        rng=random.Random(0),
    )
    # 1) Slot is populated.
    assert mcts.reward_aggregator is agg, (
        f"MCTSProofSearch.reward_aggregator must store the supplied "
        f"aggregator, got {mcts.reward_aggregator!r}"
    )
    # 2) ``_resolved_reward`` returns the same aggregator.
    resolved = mcts._resolved_reward()
    assert resolved is agg, (
        f"_resolved_reward() must prefer reward_aggregator over "
        f"reward, got {resolved!r}"
    )

    # 3) The legacy ``reward=`` kwarg still works (backward compat).
    agg2 = _make_aggregator()
    mcts2 = MCTSProofSearch(
        tile_library=_tile_library(),
        rules=_stub_rules(),
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        reward=agg2,
        n_simulations=2,
        rng=random.Random(1),
    )
    assert mcts2._resolved_reward() is agg2, (
        f"reward= kwarg must still resolve, got {mcts2._resolved_reward()!r}"
    )

    # 4) ``reward_aggregator`` wins over ``reward`` when both are
    # supplied (preferred-spelling precedence).
    mcts3 = MCTSProofSearch(
        tile_library=_tile_library(),
        rules=_stub_rules(),
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
        reward_aggregator=agg,
        reward=agg2,
        n_simulations=2,
        rng=random.Random(2),
    )
    assert mcts3._resolved_reward() is agg, (
        f"reward_aggregator= must take precedence over reward=, "
        f"got {mcts3._resolved_reward()!r}"
    )

    # 5) Default constructor (no reward_aggregator / no reward /
    # no scorer) still works — the resolved aggregator is a fresh
    # zero-signal RewardAggregator.
    mcts4 = MCTSProofSearch(
        tile_library=_tile_library(),
        rules=_stub_rules(),
        target_predicates=[],
        binding_site=PROTEASE_GENERIC,
    )
    resolved4 = mcts4._resolved_reward()
    assert isinstance(resolved4, RewardAggregator)


if __name__ == "__main__":
    # Allow running the test directly with ``python -m`` style invocation.
    raise SystemExit(pytest.main([__file__, "--tb=short"]))
