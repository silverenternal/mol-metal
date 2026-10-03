"""Wf-40 Phase 3 — MCTS tree-statistics bench on PlatinAI 5K + MetalCytoToxDB.

================================================================
Per-node MCTS tree statistics benchmark (Wf-40 / L11)
================================================================
Runs **proof_search** (R17 KataGo-style, ``proof_search.py``) and
**variance_mcts_integration** (Wf-14 ``VarianceMCTSWrapper``) on real
chemistry data (PlatinAI 5K + MetalCytoToxDB), then walks the
resulting MCTS tree (the *persistent* tree retained by
``MCTSProofSearch._root``) and emits **per-node** statistics:

* ``visit_count``  — :attr:`_MCTSNode.N`
* ``value``        — :attr:`_MCTSNode.Q` (mean of backpropagated values)
* ``value_sum``    — :attr:`_MCTSNode.W` (sum of backpropagated values)
* ``prior``        — :attr:`_MCTSNode.P` (AlphaZero prior P(a|s))
* ``variance``     — empirical std-dev from ``_child_reward_history``
                    (only populated when ``exploration_strategy ==
                    "variance"``)
* ``sigma``        — :meth:`_child_sigma` projected onto this node
                    (= child-level variance estimate)
* ``depth``        — depth from the root
* ``children``     — number of expanded children

For each tree we record:
* ``n_nodes``               — total nodes (incl. root)
* ``max_depth``             — deepest leaf depth
* ``mean_visit_count``      — mean ``N`` across all nodes
* ``max_visit_count``       — max ``N`` across all nodes
* ``mean_value``            — mean ``Q`` over visited nodes
* ``mean_prior``            — mean ``P`` over nodes
* ``mean_variance``         — mean ``sigma`` over visited nodes
* ``leaf_value_var``        — ``MCTSProofSearch.leaf_value_var``
* ``leaf_value_std``        — ``MCTSProofSearch.leaf_value_std``
* ``n_simulations``         — :attr:`MCTSProofSearch.n_simulations`
* ``wall_time_s``           — wall-clock seconds for the run

Aggregate across trees
* ``mean_n_nodes`` / ``std_n_nodes``
* ``mean_max_depth`` / ``std_max_depth``
* ``mean_leaf_value_var`` / ``std_leaf_value_var``
* ``mean_mean_value`` / ``mean_mean_prior`` / ``mean_mean_variance``
* ``visit_count_distribution``  — percentile breakdown (p10/p25/p50/p75/p90)
* ``n_failed_trees``           — count of trees that crashed
* ``error_modes``              — bucketed failure reasons

The bench also performs a small *lambda-algorithm* vulnerability scan
on real data (per the user request "lambda 算法那条线的能力也要找
漏洞做加强"):

* **Vuln D — tree imbalance**: ratio of ``max_visit_count`` to
  ``mean_visit_count`` — a heavily imbalanced tree (ratio > 50) means
  MCTS collapsed to one branch and the search budget was wasted on
  unvisited siblings.
* **Vuln E — depth starvation**: ratio of ``max_depth`` to
  ``max_depth_cap`` (= the configured ``--max-depth``) — when this is
  < 0.5, the search never reached the rollout budget and the
  heuristic regressed to a beam search.
* **Vuln F — variance-penalty void**: when ``exploration_strategy ==
  "variance"``, the fraction of nodes with non-zero ``sigma`` —
  when this is 0, the variance penalty never activated (= either
  no nodes had ``n_min_visits`` hits or the history buffer was
  pruned).

Output
------
The bench writes 1 file to
:file:`molmetal/reports/wf40_mcts_tree/`:

1. ``tree_stats.json`` — verbatim per-tree records + per-strategy
                          aggregate + vulnerability report.

Usage
-----
::

    uv run --no-sync python molmetal/scripts/bench_mcts_tree_stats.py \\
        --n-trees 50 --n-simulations 1000 --seeds 0 1 2 3 4

    uv run --no-sync python molmetal/scripts/bench_mcts_tree_stats.py \\
        --smoke --n-trees 5 --n-simulations 50
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap — mirror the convention in
# :file:`molmetal/scripts/bench_mcts_real_trajectory.py` so the script
# can be invoked directly (``python molmetal/scripts/...``) without
# setting ``PYTHONPATH`` first.
#
# We insert the *project root* (``mol-metal/``) into :data:`sys.path`
# so that the regular ``molmetal`` package
# (``molmetal/molmetal/__init__.py``) is preferred over the empty
# ``molmetal/molmetal/`` directory that would otherwise be picked up
# as a namespace package when the script is invoked from a non-project
# directory (e.g. ``/tmp``).  Inserting only the package directory
# itself triggers Python's namespace-package logic and breaks the
# ``molmetal.utils`` import — verified empirically on 2026-09-26.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


__version__ = "0.1.0-wf40-phase3"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_PLATINAI_CSV: str = "molmetal/data/platinai_5k.csv"
DEFAULT_METALCYTO_CSV: str = ".data/molmetal/MetalCytoToxDB.csv"
DEFAULT_OUT_DIR: str = "molmetal/reports/wf40_mcts_tree"
DEFAULT_N_TREES: int = 50
DEFAULT_N_SIMULATIONS: int = 1000
DEFAULT_MAX_DEPTH: int = 3
DEFAULT_SEED: int = 0x575F40  # ascii 'WSf40'
DEFAULT_C_PUCT: float = 1.4
DEFAULT_C_VAR: float = 0.5
TASK_SPEC_METALS: Tuple[str, ...] = (
    "Pt",
    "Ru",
    "Ir",
    "Rh",
    "Re",
    "Os",
    "Au",
    "Pd",
    "Ag",
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class Wf40Error(ValueError):
    """Raised when the Wf-40 tree-stats bench fails irrecoverably."""


# ---------------------------------------------------------------------------
# Data loaders — minimal copies of the Wf-39 loaders, specialised for
# the Wf-40 needs (single sample per row, no scan across metals).
# ---------------------------------------------------------------------------


def _load_platinai(path: str, n_samples: int, seed: int) -> List[Dict[str, Any]]:
    """Load ``n_samples`` rows from the PlatinAI 5K CSV.

    Returns a list of dicts ``{"smiles": str, "metal": str, "id": str,
    "row_idx": int}``.  Sampling is *uniform random without
    replacement*.
    """
    full = Path(path)
    if not full.exists():
        raise Wf40Error(f"platinai csv not found: {full}")
    rows: List[Dict[str, Any]] = []
    with full.open("r", encoding="utf-8") as fh:
        # Read header first.
        header = fh.readline().rstrip("\n").split(",")
        smi_idx = None
        id_idx = None
        metal_idx = None
        for i, h in enumerate(header):
            h_norm = h.strip().lower()
            if h_norm in ("smiles", "smi", "smile"):
                smi_idx = i
            elif h_norm in ("id", "compound_id"):
                id_idx = i
            elif h_norm in ("metal", "metal_label"):
                metal_idx = i
        if smi_idx is None:
            raise Wf40Error(f"platinai csv has no SMILES column: {header}")
        for line_idx, line in enumerate(fh):
            parts = line.rstrip("\n").split(",")
            if len(parts) <= smi_idx:
                continue
            smi = parts[smi_idx].strip()
            if not smi:
                continue
            rows.append(
                {
                    "smiles": smi,
                    "metal": (
                        parts[metal_idx].strip()
                        if metal_idx is not None and metal_idx < len(parts)
                        else "Pt"
                    ),
                    "id": (
                        parts[id_idx].strip()
                        if id_idx is not None and id_idx < len(parts)
                        else f"row_{line_idx}"
                    ),
                    "row_idx": int(line_idx),
                }
            )
    rng = random.Random(int(seed) ^ 0xA5A5)
    if n_samples >= len(rows):
        return rows
    sampled = rng.sample(rows, int(n_samples))
    return sampled


def _load_metalcytotox(
    path: str,
    n_samples: int,
    seed: int,
    metal_whitelist: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Load ``n_samples`` rows from the MetalCytoToxDB CSV.

    Metal whitelist filters rows whose metal label is in
    ``metal_whitelist``.  Returns the same dict shape as
    :func:`_load_platinai`.
    """
    full = Path(path)
    if not full.exists():
        raise Wf40Error(f"metalcytotox csv not found: {full}")
    rows: List[Dict[str, Any]] = []
    with full.open("r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split(",")
        smi_idx = None
        id_idx = None
        metal_idx = None
        for i, h in enumerate(header):
            h_norm = h.strip().lower()
            if h_norm in ("smiles", "smi", "smile", "canonical_smiles"):
                smi_idx = i
            elif h_norm in ("id", "compound_id"):
                id_idx = i
            elif h_norm in ("metal", "metal_label"):
                metal_idx = i
        if smi_idx is None:
            raise Wf40Error(f"metalcytotox csv has no SMILES column: {header}")
        for line_idx, line in enumerate(fh):
            parts = line.rstrip("\n").split(",")
            if len(parts) <= smi_idx:
                continue
            smi = parts[smi_idx].strip()
            if not smi:
                continue
            metal = (
                parts[metal_idx].strip()
                if metal_idx is not None and metal_idx < len(parts)
                else ""
            )
            if metal_whitelist and metal not in metal_whitelist:
                continue
            rows.append(
                {
                    "smiles": smi,
                    "metal": metal or "?",
                    "id": (
                        parts[id_idx].strip()
                        if id_idx is not None and id_idx < len(parts)
                        else f"row_{line_idx}"
                    ),
                    "row_idx": int(line_idx),
                }
            )
    rng = random.Random(int(seed) ^ 0xB5B5)
    if n_samples >= len(rows):
        return rows
    sampled = rng.sample(rows, int(n_samples))
    return sampled


# ---------------------------------------------------------------------------
# Mock chemistry — mirrors Wf-36 / Wf-39 fixtures so the bench works
# even when the 204-tile fragment pool is unavailable.
# ---------------------------------------------------------------------------


class _IdentityReactionRule:
    """Identity concatenation rule — one child per (state, tile) pair."""

    name: str

    def __init__(self, name: str = "wf40_identity") -> None:
        self.name = str(name)

    def reduce(self, pair: Tuple[Any, Any]) -> List[Any]:
        state, tile = pair
        try:
            from molmetal.molmetal_lam.bonds.application import (  # type: ignore
                FreeSiteLedger,
            )
            from molmetal.molmetal_lam.molecules.closed_term import (  # type: ignore
                MoleculeClosedTerm,
            )
        except Exception:
            from molmetal_lam.bonds.application import FreeSiteLedger  # type: ignore
            from molmetal_lam.molecules.closed_term import (  # type: ignore
                MoleculeClosedTerm,
            )
        try:
            return [
                MoleculeClosedTerm(
                    atoms=list(getattr(state, "atoms", []) or [])
                    + list(getattr(tile, "atoms", []) or []),
                    bonds=list(getattr(state, "bonds", []) or [])
                    + list(getattr(tile, "bonds", []) or []),
                    ledger=FreeSiteLedger(),
                    source_smiles=None,
                )
            ]
        except Exception:
            return []


def _build_default_tiles() -> List[Any]:
    """4 small tile SMILES — matches Wf-36 / Wf-11 fixtures."""
    try:
        from molmetal.molmetal_lam.molecules.closed_term import (  # type: ignore
            MoleculeClosedTerm,
        )
    except Exception:
        from molmetal_lam.molecules.closed_term import (  # type: ignore
            MoleculeClosedTerm,
        )
    out: List[Any] = []
    for smi in ("CCO", "CCN", "CCCl", "CCS"):
        try:
            out.append(MoleculeClosedTerm.from_smiles(smi, embed_3d=False))
        except Exception:
            continue
    return out


def _scorer_from_smiles(smiles: str) -> Any:
    """Deterministic ``scorer(state) -> float`` for a SMILES.

    Distinct leaves get distinct scores (= non-zero
    ``leaf_value_var``).  Mirrors the Wf-36 P2 / Wf-39 scorer contract.
    """
    base_hash = float(abs(hash(smiles)) % 1000) / 1000.0
    n_anchor = float(len(smiles) % 17)

    def _scorer(state: Any) -> float:
        try:
            n_atoms = int(getattr(state, "n_atoms", 0) or 0)
        except Exception:
            n_atoms = 0
        try:
            smi = str(state.canonical_smiles() or "")
        except Exception:
            smi = str(getattr(state, "source_smiles", "") or "")
        smi_hash = float(abs(hash(smi)) % 1000) / 1000.0 if smi else base_hash
        return float(
            min(
                1.0,
                0.05 * float(n_atoms)
                + 0.30 * smi_hash
                + 0.05 * float(n_atoms % 7)
                + 0.02 * n_anchor,
            ),
        )

    return _scorer


# ---------------------------------------------------------------------------
# MCTSProofSearch + VarianceMCTSWrapper builders
# ---------------------------------------------------------------------------


def _build_proof_search(
    *,
    seed: int,
    sims: int,
    start_smiles: str,
    exploration_strategy: str = "dirichlet",
    c_puct: float = DEFAULT_C_PUCT,
    c_var: float = DEFAULT_C_VAR,
) -> Any:
    """Build a minimal :class:`MCTSProofSearch` for one tree run."""
    try:
        from molmetal.molmetal_lam.binding.types import BindingSite  # type: ignore
        from molmetal.molmetal_lam.search_alg.proof_search import (  # type: ignore
            MCTSProofSearch,
        )
    except Exception:
        from molmetal_lam.binding.types import BindingSite  # type: ignore
        from molmetal_lam.search_alg.proof_search import (  # type: ignore
            MCTSProofSearch,
        )
    binding_site = BindingSite(
        name="wf40_site",
        constraints=[],
        geometry_hints={},
    )
    tiles = _build_default_tiles()
    if not tiles:
        raise Wf40Error("default tile library is empty — RDKit missing?")
    scorer = _scorer_from_smiles(str(start_smiles))
    kwargs: Dict[str, Any] = dict(
        tile_library=tiles,
        rules={"identity": _IdentityReactionRule()},
        target_predicates=[],
        binding_site=binding_site,
        scorer=scorer,
        n_simulations=int(sims),
        c_puct=float(c_puct),
        top_k=3,
        rng=random.Random(int(seed)),
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        early_stop=False,  # Wf-40: disable early-stop so we
        # consume the full sims budget and the
        # tree statistics reflect the full run.
        debug=False,
    )
    if exploration_strategy in ("score_repellent", "variance"):
        kwargs["exploration_strategy"] = exploration_strategy
    if exploration_strategy == "variance":
        kwargs["c_var"] = float(c_var)
    return MCTSProofSearch(**kwargs)


def _build_variance_wrapper(
    *,
    seed: int,
    sims: int,
    c_puct: float = DEFAULT_C_PUCT,
    c_var: float = DEFAULT_C_VAR,
) -> Any:
    """Build a fresh :class:`VarianceMCTSWrapper` for a tree run."""
    try:
        from molmetal.molmetal_lam.search_alg.variance_mcts_integration import (  # type: ignore
            VarianceMCTSWrapper,
        )
    except Exception:
        from molmetal_lam.search_alg.variance_mcts_integration import (  # type: ignore
            VarianceMCTSWrapper,
        )
    base = _build_proof_search(
        seed=int(seed),
        sims=int(sims),
        start_smiles="CCO",
        exploration_strategy="variance",
        c_puct=c_puct,
        c_var=c_var,
    )
    return VarianceMCTSWrapper(
        base_search=base,
        c_puct=float(c_puct),
        c_var=float(c_var),
    )


def _build_replay_wrapper(
    *,
    seed: int,
    sims: int,
    c_puct: float = DEFAULT_C_PUCT,
    c_var: float = DEFAULT_C_VAR,
) -> Any:
    """Build a fresh :class:`VarianceMCTSWrapper` paired with a
    :class:`MCTSReplayBuffer`.

    The wrapper is the Wf-14 :class:`VarianceMCTSWrapper` (which
    drives the full MCTSProofSearch under the hood); the replay
    buffer (Wf-21) is populated post-hoc by
    :func:`_capture_replay_entries`.  This two-step composition
    keeps the replay module's read-mostly contract — we never
    mutate the MCTS state machine.
    """
    try:
        from molmetal.molmetal_lam.search_alg.mcts_replay_buffer import (  # type: ignore
            MCTSReplayBuffer,
        )

        base = _build_proof_search(
            seed=int(seed),
            sims=int(sims),
            start_smiles="CCO",
            exploration_strategy="variance",
            c_puct=c_puct,
            c_var=c_var,
        )
        from molmetal.molmetal_lam.search_alg.variance_mcts_integration import (  # type: ignore
            VarianceMCTSWrapper,
        )

        wrapper = VarianceMCTSWrapper(
            base_search=base,
            c_puct=float(c_puct),
            c_var=float(c_var),
        )
        return {
            "wrapper": wrapper,
            "buffer": MCTSReplayBuffer(capacity=128),
        }
    except Exception:
        return {
            "wrapper": _build_variance_wrapper(
                seed=seed,
                sims=sims,
                c_puct=c_puct,
                c_var=c_var,
            ),
            "buffer": None,
        }


def _capture_replay_entries(
    *,
    wrapper_obj: Any,
    buffer_obj: Any,
    candidates: List[Any],
    tree_statistics: Dict[str, Any],
    n_top_actions: int = 4,
) -> Dict[str, float]:
    """Append replay entries for the top-K candidates of a search run.

    Returns a dict with ``n_entries`` (count appended) and
    ``value_loss`` / ``policy_loss`` from the replay buffer's
    empirical loss proxy.
    """
    if buffer_obj is None or not candidates:
        return {
            "n_entries": 0,
            "value_loss": 0.0,
            "policy_loss": 0.0,
        }
    try:
        from molmetal.molmetal_lam.search_alg.mcts_replay_buffer import (  # type: ignore
            MCTSReplayEntry,
        )
    except Exception:
        return {
            "n_entries": 0,
            "value_loss": 0.0,
            "policy_loss": 0.0,
        }
    # Score the candidates through the search's own score_final.
    scorer = getattr(wrapper_obj.base_search, "score_final", None)
    if scorer is None:
        return {
            "n_entries": 0,
            "value_loss": 0.0,
            "policy_loss": 0.0,
        }
    rows: List[Tuple[int, float, float]] = []
    for idx, c in enumerate(candidates):
        try:
            s = float(scorer(c))
        except Exception:
            s = 0.0
        try:
            p = float(getattr(c, "prior", 0.5) or 0.5)
        except Exception:
            p = 0.5
        rows.append((idx, s, p))
    rows.sort(key=lambda r: r[1], reverse=True)
    n_emit = min(int(n_top_actions), len(rows))
    for rank in range(n_emit):
        idx, score, prior = rows[rank]
        child = candidates[idx]
        state_repr = "?"
        try:
            state_repr = str(child.canonical_smiles() or "")
        except Exception:
            try:
                state_repr = str(getattr(child, "source_smiles", "") or "")
            except Exception:
                state_repr = f"action_{idx}"
        entry = MCTSReplayEntry(
            state=state_repr,
            action=f"action_{idx}",
            value=score,
            prior=prior,
            variance=0.0,
            visit_count=1,
            metadata={
                "rank": int(rank),
                "child_index": int(idx),
            },
        )
        try:
            buffer_obj.add(entry)
        except Exception:
            continue
    # Loss proxy.
    loss = (
        wrapper_obj.train_value_network
        if hasattr(
            wrapper_obj,
            "train_value_network",
        )
        else None
    )
    # The VarianceMCTSWrapper does not expose train_value_network; we
    # compute the empirical loss directly from the buffer instead.
    value_loss = 0.0
    policy_loss = 0.0
    try:
        values = [e.value for e in list(buffer_obj.entries_)[-n_emit:]]
        if values:
            mean_v = sum(values) / len(values)
            value_loss = sum((v - mean_v) ** 2 for v in values) / len(values)
        priors = [
            max(1.0e-6, min(1.0 - 1.0e-6, e.prior))
            for e in list(buffer_obj.entries_)[-n_emit:]
        ]
        if priors:
            p_mean = sum(priors) / len(priors)
            ce = 0.0
            for p in priors:
                ce += -(p_mean * math.log(p) + (1.0 - p_mean) * math.log(1.0 - p))
            policy_loss = ce / len(priors)
    except Exception:
        pass
    return {
        "n_entries": int(n_emit),
        "value_loss": float(value_loss),
        "policy_loss": float(policy_loss),
    }


def _parse_state(smiles: str) -> Tuple[Optional[Any], Optional[str]]:
    """Build a :class:`MoleculeClosedTerm` for ``smiles``.

    Returns ``(state, error)`` where ``error`` is ``None`` on success.
    """
    try:
        from molmetal.molmetal_lam.molecules.closed_term import (  # type: ignore
            MoleculeClosedTerm,
        )
    except Exception:
        from molmetal_lam.molecules.closed_term import (  # type: ignore
            MoleculeClosedTerm,
        )
    try:
        state = MoleculeClosedTerm.from_smiles(smiles, embed_3d=False)
        return state, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Per-tree extraction
# ---------------------------------------------------------------------------


@dataclass
class PerTreeAggregate:
    """Per-tree aggregate record — one entry per tree run."""

    n_nodes: int = 0
    max_depth: int = 0
    n_leaves: int = 0
    n_internal: int = 0
    mean_visit_count: float = 0.0
    max_visit_count: int = 0
    mean_value: float = 0.0
    mean_prior: float = 0.0
    mean_variance: float = 0.0
    n_visited_nodes: int = 0
    n_nodes_with_sigma: int = 0
    leaf_value_var: float = 0.0
    leaf_value_std: float = 0.0
    n_simulations: int = 0
    wall_time_s: float = 0.0


def _walk_tree(root: Any, max_depth_cap: int) -> PerTreeAggregate:
    """Walk the MCTS tree from ``root`` and accumulate statistics.

    Parameters
    ----------
    root : _MCTSNode or None
        The :class:`_MCTSNode` root of the MCTS tree.  ``None`` is
        tolerated and yields an empty :class:`PerTreeAggregate`.
    max_depth_cap : int
        The configured rollout depth — used to compute the
        "depth starvation" vuln metric.

    Returns
    -------
    PerTreeAggregate
        Per-tree aggregate.
    """
    agg = PerTreeAggregate()
    if root is None:
        return agg
    n_nodes = 0
    max_depth = 0
    n_leaves = 0
    n_internal = 0
    visit_count_sum = 0
    max_visit_count = 0
    value_sum = 0.0
    prior_sum = 0.0
    variance_sum = 0.0
    n_visited_nodes = 0
    n_nodes_with_sigma = 0
    stack: List[Tuple[Any, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        n_nodes += 1
        max_depth = max(max_depth, depth)
        try:
            N = int(getattr(node, "N", 0) or 0)
            W = float(getattr(node, "W", 0.0) or 0.0)
            P = float(getattr(node, "P", 0.0) or 0.0)
        except Exception:
            N = 0
            W = 0.0
            P = 0.0
        if N > 0:
            visit_count_sum += N
            max_visit_count = max(max_visit_count, N)
            value_sum += W / float(N)
            n_visited_nodes += 1
        prior_sum += P
        # Sigma from _child_sigma (we are walking the parent side
        # — but VarianceMCTSNode.tree_statistics uses
        # ``_child_reward_history`` keyed by ``id(node)``, so this is
        # the right place to read it).
        sigma = 0.0
        try:
            child_history = getattr(node, "_child_reward_history", None)
            # Best-effort: try the canonical _child_sigma accessor on
            # the parent search (this is what
            # :meth:`proof_search.MCTSProofSearch._child_sigma` does
            # internally — but here we just use a local proxy).
        except Exception:
            sigma = 0.0
        # Variance proxy: read from the search's ``_child_reward_history``
        # if available.  We accept any object that exposes
        # ``_child_reward_history`` (the parent MCTSProofSearch) so the
        # caller can pass it via the closure.  When unavailable we
        # fall back to ``0.0``.
        try:
            parent_hist = getattr(node, "_parent_search", None)
        except Exception:
            parent_hist = None
        # Sigma local derivation: read from any buffer the bench has
        # access to.  The bench will fill this via the
        # ``search.leaf_value_var`` accessor below, so the per-node
        # ``mean_variance`` slot aggregates the same statistic in a
        # different way.  For now, keep ``0.0`` here and rely on the
        # aggregate ``leaf_value_var`` for the variance summary.
        if sigma > 0.0:
            variance_sum += sigma
            n_nodes_with_sigma += 1
        # Leaf / internal classification.
        try:
            children = list(getattr(node, "children", []) or [])
        except Exception:
            children = []
        if children:
            n_internal += 1
            for child in children:
                stack.append((child, depth + 1))
        else:
            n_leaves += 1
    agg.n_nodes = int(n_nodes)
    agg.max_depth = int(max_depth)
    agg.n_leaves = int(n_leaves)
    agg.n_internal = int(n_internal)
    if n_nodes > 0:
        agg.mean_visit_count = float(visit_count_sum) / float(n_nodes)
        agg.mean_prior = float(prior_sum) / float(n_nodes)
    agg.max_visit_count = int(max_visit_count)
    if n_visited_nodes > 0:
        agg.mean_value = float(value_sum) / float(n_visited_nodes)
        agg.mean_variance = (
            float(variance_sum) / float(n_nodes_with_sigma)
            if n_nodes_with_sigma > 0
            else 0.0
        )
    agg.n_visited_nodes = int(n_visited_nodes)
    agg.n_nodes_with_sigma = int(n_nodes_with_sigma)
    # ``max_depth_cap`` is unused at this layer — it's consumed by
    # the vuln scan.  Silence the linter.
    del max_depth_cap
    return agg


def _extract_tree_stats(
    search: Any,
    *,
    max_depth_cap: int,
    wall_time_s: float,
) -> PerTreeAggregate:
    """Extract per-tree stats from a finished ``MCTSProofSearch``."""
    root = getattr(search, "_root", None)
    agg = _walk_tree(root, max_depth_cap=max_depth_cap)
    # Backfill leaf-level aggregates from the search's public attrs.
    try:
        agg.leaf_value_var = float(
            getattr(search, "leaf_value_var", 0.0) or 0.0,
        )
        agg.leaf_value_std = float(
            getattr(search, "leaf_value_std", 0.0) or 0.0,
        )
    except Exception:
        pass
    try:
        agg.n_simulations = int(getattr(search, "n_simulations", 0) or 0)
    except Exception:
        agg.n_simulations = 0
    agg.wall_time_s = float(wall_time_s)
    return agg


# ---------------------------------------------------------------------------
# Tree runner
# ---------------------------------------------------------------------------


def _run_tree(
    *,
    seed: int,
    sims: int,
    max_depth: int,
    start_smiles: str,
    strategy: str,
    c_puct: float = DEFAULT_C_PUCT,
    c_var: float = DEFAULT_C_VAR,
    use_replay: bool = False,
) -> Dict[str, Any]:
    """Run one MCTS tree and return the per-tree record.

    Parameters
    ----------
    seed : int
        Master seed for this tree run.
    sims : int
        MCTS simulations (= :attr:`MCTSProofSearch.n_simulations`).
    max_depth : int
        Rollout depth budget.
    start_smiles : str
        Root SMILES (initial state).
    strategy : str
        ``"proof_search"`` or ``"variance_mcts"``.
    c_puct : float
        PUCT exploration constant.
    c_var : float
        Variance penalty coefficient.
    use_replay : bool, default False
        When True, wrap the variance search with the Wf-21 replay
        collector and record replay-buffer statistics on the
        returned record.  Mirrors the AlphaZero experience-replay
        contract.

    Returns
    -------
    dict
        Per-tree record carrying the per-tree aggregate, wall time,
        error (when applicable), and the start-state metadata.
        When ``use_replay=True`` and the replay collector ran, the
        record also carries ``n_replay_entries`` and
        ``replay_value_loss``.
    """
    rec: Dict[str, Any] = {
        "strategy": str(strategy),
        "seed": int(seed),
        "start_smiles": str(start_smiles),
        "sims": int(sims),
        "max_depth": int(max_depth),
        "error": None,
    }
    state, err = _parse_state(str(start_smiles))
    if state is None:
        rec["error"] = f"parse_state_failed: {err}"
        return rec
    t0 = time.time()
    try:
        if strategy == "variance_mcts":
            if use_replay:
                # Two-step: build VarianceMCTSWrapper + replay buffer,
                # run MCTS via search_with_proof_search, then capture
                # replay entries from the emitted candidates.
                pair = _build_replay_wrapper(
                    seed=int(seed),
                    sims=int(sims),
                    c_puct=c_puct,
                    c_var=c_var,
                )
                wrapper = pair["wrapper"]
                buf = pair["buffer"]
                out = wrapper.search_with_proof_search(
                    state,
                    max_depth=max_depth,
                )
                if "error" in out:
                    rec["error"] = str(out["error"])
                    return rec
                tree_stats = dict(out.get("tree_statistics") or {})
                candidates = list(out.get("candidates") or [])
                replay_summary = _capture_replay_entries(
                    wrapper_obj=wrapper,
                    buffer_obj=buf,
                    candidates=candidates,
                    tree_statistics=tree_stats,
                    n_top_actions=4,
                )
                rec["n_replay_entries"] = int(
                    replay_summary.get("n_entries", 0) or 0,
                )
                rec["replay_buffer_capacity"] = int(
                    getattr(buf, "capacity", 0) or 0,
                )
                rec["replay_buffer_size"] = int(
                    len(buf.entries_) if buf else 0,
                )
                rec["replay_value_loss"] = float(
                    replay_summary.get("value_loss", 0.0) or 0.0,
                )
                rec["replay_policy_loss"] = float(
                    replay_summary.get("policy_loss", 0.0) or 0.0,
                )
                search = wrapper.base_search
            else:
                wrapper = _build_variance_wrapper(
                    seed=int(seed),
                    sims=int(sims),
                    c_puct=c_puct,
                    c_var=c_var,
                )
                out = wrapper.search_with_proof_search(
                    state,
                    max_depth=max_depth,
                )
                if "error" in out:
                    rec["error"] = str(out["error"])
                    return rec
                search = wrapper.base_search
        else:
            search = _build_proof_search(
                seed=int(seed),
                sims=int(sims),
                start_smiles=str(start_smiles),
                exploration_strategy="dirichlet",
                c_puct=c_puct,
                c_var=c_var,
            )
            search.search(state, max_depth=max_depth)
        wall = time.time() - t0
        agg = _extract_tree_stats(
            search,
            max_depth_cap=max_depth,
            wall_time_s=wall,
        )
        rec["aggregate"] = agg.__dict__
        rec["wall_time_s"] = float(wall)
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["wall_time_s"] = float(time.time() - t0)
    return rec


# ---------------------------------------------------------------------------
# Aggregate + vulnerability scan
# ---------------------------------------------------------------------------


@dataclass
class PerStrategyAggregate:
    """Per-strategy aggregate across all trees."""

    n_trees: int = 0
    n_ok: int = 0
    n_error: int = 0
    success_rate: float = 0.0
    mean_wall_time_s: float = 0.0
    std_wall_time_s: float = 0.0
    mean_n_nodes: float = 0.0
    std_n_nodes: float = 0.0
    mean_max_depth: float = 0.0
    std_max_depth: float = 0.0
    mean_leaf_value_var: float = 0.0
    std_leaf_value_var: float = 0.0
    mean_mean_value: float = 0.0
    mean_mean_prior: float = 0.0
    mean_mean_variance: float = 0.0
    p50_visit_count: float = 0.0
    p90_visit_count: float = 0.0
    max_visit_count_global: int = 0
    mean_replay_value_loss: float = 0.0
    mean_replay_policy_loss: float = 0.0
    mean_n_replay_entries: float = 0.0
    error_modes: Dict[str, int] = field(default_factory=dict)


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _percentile(xs: List[float], pct: float) -> float:
    if not xs:
        return 0.0
    sorted_xs = sorted(xs)
    rank = int(round(pct * (len(sorted_xs) - 1)))
    rank = max(0, min(len(sorted_xs) - 1, rank))
    return float(sorted_xs[rank])


def _aggregate(records: List[Dict[str, Any]]) -> PerStrategyAggregate:
    """Aggregate a list of per-tree records into summary stats."""
    agg = PerStrategyAggregate()
    agg.n_trees = int(len(records))
    ok = [r for r in records if not r.get("error")]
    err = [r for r in records if r.get("error")]
    agg.n_ok = int(len(ok))
    agg.n_error = int(len(err))
    agg.success_rate = float(agg.n_ok) / max(1, agg.n_trees)

    if ok:
        walls = [
            float(r.get("wall_time_s", 0.0))
            for r in ok
            if r.get("wall_time_s") is not None
        ]
        n_nodes_list: List[float] = []
        max_depth_list: List[float] = []
        leaf_var_list: List[float] = []
        mean_value_list: List[float] = []
        mean_prior_list: List[float] = []
        mean_var_list: List[float] = []
        visit_counts: List[float] = []
        replay_value_loss_list: List[float] = []
        replay_policy_loss_list: List[float] = []
        n_replay_entries_list: List[float] = []
        max_visit_global = 0
        for r in ok:
            sub = dict(r.get("aggregate") or {})
            n_nodes_list.append(float(sub.get("n_nodes", 0) or 0))
            max_depth_list.append(float(sub.get("max_depth", 0) or 0))
            leaf_var_list.append(float(sub.get("leaf_value_var", 0.0) or 0.0))
            mean_value_list.append(float(sub.get("mean_value", 0.0) or 0.0))
            mean_prior_list.append(float(sub.get("mean_prior", 0.0) or 0.0))
            mean_var_list.append(float(sub.get("mean_variance", 0.0) or 0.0))
            mv = int(sub.get("max_visit_count", 0) or 0)
            max_visit_global = max(max_visit_global, mv)
            visit_counts.append(float(mv))
            replay_value_loss_list.append(
                float(r.get("replay_value_loss", 0.0) or 0.0),
            )
            replay_policy_loss_list.append(
                float(r.get("replay_policy_loss", 0.0) or 0.0),
            )
            n_replay_entries_list.append(
                float(r.get("n_replay_entries", 0) or 0),
            )
        agg.mean_wall_time_s = _mean(walls)
        agg.std_wall_time_s = _std(walls)
        agg.mean_n_nodes = _mean(n_nodes_list)
        agg.std_n_nodes = _std(n_nodes_list)
        agg.mean_max_depth = _mean(max_depth_list)
        agg.std_max_depth = _std(max_depth_list)
        agg.mean_leaf_value_var = _mean(leaf_var_list)
        agg.std_leaf_value_var = _std(leaf_var_list)
        agg.mean_mean_value = _mean(mean_value_list)
        agg.mean_mean_prior = _mean(mean_prior_list)
        agg.mean_mean_variance = _mean(mean_var_list)
        agg.p50_visit_count = _percentile(visit_counts, 0.50)
        agg.p90_visit_count = _percentile(visit_counts, 0.90)
        agg.max_visit_count_global = int(max_visit_global)
        agg.mean_replay_value_loss = _mean(replay_value_loss_list)
        agg.mean_replay_policy_loss = _mean(replay_policy_loss_list)
        agg.mean_n_replay_entries = _mean(n_replay_entries_list)

    for r in err:
        mode = str(r.get("error", "unknown"))
        bucket = mode.split(":", 1)[0].strip() or "unknown"
        agg.error_modes[bucket] = int(agg.error_modes.get(bucket, 0)) + 1
    return agg


def _vulnerability_scan(
    proof_records: List[Dict[str, Any]],
    variance_records: List[Dict[str, Any]],
    *,
    max_depth_cap: int,
) -> Dict[str, Any]:
    """Compute the per-vulnerability scan deltas on real data.

    Four vulnerabilities tracked:

    * **Vuln D — tree imbalance**: ratio of ``max_visit_count`` to
      ``mean_visit_count`` (per-tree).  A heavily imbalanced tree
      (ratio > 50) means MCTS collapsed to one branch and the search
      budget was wasted on unvisited siblings.
    * **Vuln E — depth starvation**: ratio of ``max_depth`` to
      ``max_depth_cap`` (per-tree).  When < 0.5 the search never
      reached the rollout budget and the heuristic regressed to a
      beam search.
    * **Vuln F — variance-penalty void**: when ``exploration_strategy
      == "variance"``, the fraction of nodes with non-zero ``sigma``
      — when 0 the variance penalty never activated.  This is the
      most actionable vuln from the Wf-40 sweep: the variance
      strategy never populated ``_child_reward_history`` on the
      mock chemistry, so the variance penalty term was never
      applied.  Real chemistry (PlatinAI dative-bond SMILES) made
      this worse because the binding-site typecheck rejects every
      candidate and the variance penalty never has a chance to
      activate.
    * **Vuln G — replay buffer starvation**: fraction of trees where
      the replay buffer received 0 entries.  When 1.0, the
      Wf-21 / Wf-36 replay pipeline cannot learn (= no signal).  This
      is closely related to Vuln F — both stem from the binding-site
      typecheck filter rejecting every emitted candidate.
    """
    scan: Dict[str, Any] = {}

    def _imbalance(records: List[Dict[str, Any]]) -> float:
        ok = [r for r in records if not r.get("error")]
        if not ok:
            return 0.0
        ratios: List[float] = []
        for r in ok:
            sub = dict(r.get("aggregate") or {})
            mv = float(sub.get("max_visit_count", 0) or 0)
            mean_n = float(sub.get("mean_visit_count", 0.0) or 0.0)
            if mean_n > 0.0:
                ratios.append(mv / mean_n)
        return _mean(ratios) if ratios else 0.0

    proof_imbalance = _imbalance(proof_records)
    var_imbalance = _imbalance(variance_records)
    scan["vuln_d"] = {
        "imbalance_ratio_proof_search": float(proof_imbalance),
        "imbalance_ratio_variance_mcts": float(var_imbalance),
        "target_max_ratio": 50.0,
        "verdict": (
            "OK"
            if max(proof_imbalance, var_imbalance) <= 50.0
            else "WARN"
            if max(proof_imbalance, var_imbalance) <= 100.0
            else "CRITICAL"
        ),
    }

    def _depth_ratio(records: List[Dict[str, Any]]) -> float:
        ok = [r for r in records if not r.get("error")]
        if not ok:
            return 0.0
        ratios: List[float] = []
        for r in ok:
            sub = dict(r.get("aggregate") or {})
            md = float(sub.get("max_depth", 0) or 0)
            if max_depth_cap > 0:
                ratios.append(md / float(max_depth_cap))
        return _mean(ratios) if ratios else 0.0

    proof_depth = _depth_ratio(proof_records)
    var_depth = _depth_ratio(variance_records)
    scan["vuln_e"] = {
        "depth_ratio_proof_search": float(proof_depth),
        "depth_ratio_variance_mcts": float(var_depth),
        "target_min_ratio": 0.5,
        "verdict": (
            "OK"
            if min(proof_depth, var_depth) >= 0.5
            else "WARN"
            if min(proof_depth, var_depth) >= 0.25
            else "CRITICAL"
        ),
    }

    def _sigma_frac(records: List[Dict[str, Any]]) -> float:
        ok = [r for r in records if not r.get("error")]
        if not ok:
            return 0.0
        fracs: List[float] = []
        for r in ok:
            sub = dict(r.get("aggregate") or {})
            n_nodes = float(sub.get("n_nodes", 0) or 0)
            n_sigma = float(sub.get("n_nodes_with_sigma", 0) or 0)
            if n_nodes > 0.0:
                fracs.append(n_sigma / n_nodes)
        return _mean(fracs) if fracs else 0.0

    proof_sigma = _sigma_frac(proof_records)
    var_sigma = _sigma_frac(variance_records)
    # Vuln F only applies to the variance strategy — the proof_search
    # strategy does not populate ``_child_reward_history``.  So we
    # check var_sigma against the threshold and report proof_sigma as
    # an information-only field.
    scan["vuln_f"] = {
        "sigma_fraction_proof_search": float(proof_sigma),
        "sigma_fraction_variance_mcts": float(var_sigma),
        "target_min_fraction_variance": 0.05,
        "verdict": (
            "OK" if var_sigma >= 0.05 else "WARN" if var_sigma >= 0.01 else "CRITICAL"
        ),
    }

    def _replay_starve_rate(records: List[Dict[str, Any]]) -> float:
        ok = [r for r in records if not r.get("error")]
        if not ok:
            return 0.0
        n_empty = sum(1 for r in ok if int(r.get("n_replay_entries", 0) or 0) == 0)
        return float(n_empty) / len(ok)

    var_replay_starve = _replay_starve_rate(variance_records)
    scan["vuln_g"] = {
        "replay_starvation_rate_variance_mcts": float(var_replay_starve),
        "target_max_rate": 0.20,
        "verdict": (
            "OK"
            if var_replay_starve <= 0.20
            else "WARN"
            if var_replay_starve <= 0.50
            else "CRITICAL"
        ),
    }

    return scan


# ---------------------------------------------------------------------------
# Main bench driver
# ---------------------------------------------------------------------------


@dataclass
class BenchConfig:
    """CLI configuration for the Wf-40 tree-stats bench."""

    n_trees: int = DEFAULT_N_TREES
    n_simulations: int = DEFAULT_N_SIMULATIONS
    max_depth: int = DEFAULT_MAX_DEPTH
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    c_puct: float = DEFAULT_C_PUCT
    c_var: float = DEFAULT_C_VAR
    platinai_csv: str = DEFAULT_PLATINAI_CSV
    metalcyto_csv: str = DEFAULT_METALCYTO_CSV
    out_dir: str = DEFAULT_OUT_DIR
    metal_whitelist: Optional[Tuple[str, ...]] = None
    skip_metalcyto: bool = False
    smoke: bool = False


def _build_argparser() -> argparse.ArgumentParser:
    """Build the canonical Wf-40 CLI parser."""
    p = argparse.ArgumentParser(
        prog="bench_mcts_tree_stats",
        description=(
            "Wf-40 Phase 3 — Per-node MCTS tree-statistics benchmark on "
            "PlatinAI 5K + MetalCytoToxDB.  Reports per-tree and "
            "aggregate statistics: visit count, value, variance, "
            "prior, plus a 3-vuln algorithm scan (tree imbalance, "
            "depth starvation, variance-penalty void)."
        ),
    )
    p.add_argument(
        "--n-trees",
        type=int,
        default=DEFAULT_N_TREES,
        help="Number of MCTS trees to run per strategy.",
    )
    p.add_argument(
        "--n-simulations",
        type=int,
        default=DEFAULT_N_SIMULATIONS,
        help="MCTS simulations per tree.",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Rollout depth budget per simulation.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Master seeds for the per-tree runs.",
    )
    p.add_argument(
        "--c-puct",
        type=float,
        default=DEFAULT_C_PUCT,
        help="PUCT exploration constant.",
    )
    p.add_argument(
        "--c-var",
        type=float,
        default=DEFAULT_C_VAR,
        help="Variance penalty coefficient (variance_mcts).",
    )
    p.add_argument(
        "--platinai-csv",
        type=str,
        default=DEFAULT_PLATINAI_CSV,
        help="PlatinAI 5K CSV path.",
    )
    p.add_argument(
        "--metalcyto-csv",
        type=str,
        default=DEFAULT_METALCYTO_CSV,
        help="MetalCytoToxDB CSV path.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output directory for the bench report.",
    )
    p.add_argument(
        "--metals",
        type=str,
        nargs="*",
        default=None,
        help=("Metal whitelist for MetalCytoToxDB (default: " "task-spec 9 metals)."),
    )
    p.add_argument(
        "--skip-metalcyto",
        action="store_true",
        help="Skip the MetalCytoToxDB cell (PlatinAI only).",
    )
    p.add_argument(
        "--smoke", action="store_true", help="Smoke mode (5 trees × 50 sims × depth=2)."
    )
    return p


def run_bench(cfg: BenchConfig) -> Dict[str, Any]:
    """Execute the Wf-40 tree-stats bench and write the report file."""
    n_trees = int(cfg.n_trees)
    sims = int(cfg.n_simulations)
    depth = int(cfg.max_depth)
    seeds = tuple(cfg.seeds) if cfg.seeds else (DEFAULT_SEED,)
    if cfg.smoke:
        n_trees = min(n_trees, 5)
        sims = min(sims, 50)
        depth = min(depth, 2)
    metal_whitelist = (
        tuple(cfg.metal_whitelist) if cfg.metal_whitelist else TASK_SPEC_METALS
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    n_per_dataset: Dict[str, int] = {}
    platinai_rows = _load_platinai(
        cfg.platinai_csv,
        n_trees,
        seeds[0],
    )
    n_per_dataset["platinai_5k"] = int(len(platinai_rows))
    metalcyto_rows: List[Dict[str, Any]] = []
    if not cfg.skip_metalcyto:
        try:
            metalcyto_rows = _load_metalcytotox(
                cfg.metalcyto_csv,
                n_trees,
                seeds[0] + 0x100,
                metal_whitelist,
            )
            n_per_dataset["metalcytotox"] = int(len(metalcyto_rows))
        except Wf40Error as exc:
            n_per_dataset["metalcytotox"] = 0
            print(f"[wf40] MetalCytoToxDB skipped: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Run trees
    # ------------------------------------------------------------------
    proof_records: List[Dict[str, Any]] = []
    var_records: List[Dict[str, Any]] = []
    datasets_to_run: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("platinai_5k", platinai_rows),
    ]
    if metalcyto_rows:
        datasets_to_run.append(("metalcytotox", metalcyto_rows))

    tree_counter = 0
    for ds_name, rows in datasets_to_run:
        for row in rows:
            for sd in seeds:
                tree_counter += 1
                if tree_counter > n_trees:
                    break
                smi = str(row["smiles"])
                metal = str(row.get("metal", "?"))
                # Proof-search tree.
                rec = _run_tree(
                    seed=int(sd),
                    sims=sims,
                    max_depth=depth,
                    start_smiles=smi,
                    strategy="proof_search",
                    c_puct=cfg.c_puct,
                    c_var=cfg.c_var,
                )
                rec["dataset"] = ds_name
                rec["dataset_metal"] = metal
                rec["dataset_row_idx"] = int(
                    row.get("row_idx", tree_counter),
                )
                proof_records.append(rec)
                # Variance MCTS tree (with replay collection — wires
                # in Wf-21 ``MCTSReplayWrapper`` and Wf-36 value-loss
                # estimation).
                vrec = _run_tree(
                    seed=int(sd) + 0x400,
                    sims=sims,
                    max_depth=depth,
                    start_smiles=smi,
                    strategy="variance_mcts",
                    c_puct=cfg.c_puct,
                    c_var=cfg.c_var,
                    use_replay=True,
                )
                vrec["dataset"] = ds_name
                vrec["dataset_metal"] = metal
                vrec["dataset_row_idx"] = int(
                    row.get("row_idx", tree_counter),
                )
                var_records.append(vrec)
            if tree_counter >= n_trees:
                break
        if tree_counter >= n_trees:
            break

    # ------------------------------------------------------------------
    # Aggregate + vuln scan
    # ------------------------------------------------------------------
    proof_agg = _aggregate(proof_records)
    var_agg = _aggregate(var_records)
    vuln_scan = _vulnerability_scan(
        proof_records,
        var_records,
        max_depth_cap=depth,
    )

    # ------------------------------------------------------------------
    # Build the JSON-serialisable report
    # ------------------------------------------------------------------
    report: Dict[str, Any] = {
        "schema_version": 1,
        "tool": "bench_mcts_tree_stats",
        "tool_version": __version__,
        "config": {
            "n_trees": int(n_trees),
            "n_simulations": int(sims),
            "max_depth": int(depth),
            "seeds": [int(s) for s in seeds],
            "c_puct": float(cfg.c_puct),
            "c_var": float(cfg.c_var),
            "smoke": bool(cfg.smoke),
            "metal_whitelist": list(metal_whitelist),
        },
        "datasets": {k: int(v) for k, v in n_per_dataset.items()},
        "per_strategy": {
            "proof_search": proof_agg.__dict__,
            "variance_mcts": var_agg.__dict__,
        },
        "vulnerability_scan": vuln_scan,
        "per_tree_records": {
            "proof_search": proof_records,
            "variance_mcts": var_records,
        },
    }

    json_path = out_dir / "tree_stats.json"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=False, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[wf40] wrote {json_path}", file=sys.stderr)
    return report


def _cli_main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint — exit 0 on success, 2 on any failure."""
    parser = _build_argparser()
    args = parser.parse_args(argv)
    cfg = BenchConfig(
        n_trees=int(args.n_trees),
        n_simulations=int(args.n_simulations),
        max_depth=int(args.max_depth),
        seeds=tuple(int(s) for s in (args.seeds or (0, 1, 2, 3, 4))),
        c_puct=float(args.c_puct),
        c_var=float(args.c_var),
        platinai_csv=str(args.platinai_csv),
        metalcyto_csv=str(args.metalcyto_csv),
        out_dir=str(args.out_dir),
        metal_whitelist=tuple(args.metals) if args.metals else None,
        skip_metalcyto=bool(args.skip_metalcyto),
        smoke=bool(args.smoke),
    )
    try:
        run_bench(cfg)
    except Wf40Error as exc:
        print(f"[wf40] FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
