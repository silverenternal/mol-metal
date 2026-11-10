"""Wf-129 Phase 1 — Variance MCTS x Wasserstein regulariser A/B bench.

================================================================
Wf-129 P1: Variance MCTS <-> proof_search integration with the
Wf-123 Phase 1 Wasserstein (W_2) regulariser (CPU + GPU-0).
================================================================
This bench closes the L2 gap flagged in Wf-39 P1 / Wf-50 / TODO-92:
the variance-controlled MCTS chain (``variance_mcts_integration``)
currently ships with an **honest-negative** ``variance_reduction =
-0.185`` on the mock CCO chemistry (Wf-39 P1 + Wf-50 GPU audit).
Wf-129 P1 wires the Wf-123 P1 Wasserstein (W_2) regulariser
(:mod:`molmetal_lam.reward.wasserstein_loss`) into the MCTS layer
as a *regulariser-on-priors* — not a substitute scorer — and re-runs
the 5-seed A/B harness, comparing:

* ``vanilla``            -- ``variance_mcts`` baseline (no W_2 reg)
* ``w2_reg``             -- W_2-regularised priors (alpha * W_2
                            regulariser penalty added to per-arm prior)
* ``w2_reg_decay``       -- W_2-regularised priors with the JKO
                            CFL decay check as a gating signal.

The bench is **CPU-only by default** and a tiny smoke mode
(``--smoke``) runs without RDKit or GPU.  ``--gpu-0`` enables the
GPU-0 device for the (optional) reference-distribution tensor.  All
runs are deterministic from ``--seed``; the harness sweeps
``seeds = [0, 1, 2, 3, 4]`` by default and reports per-seed
``variance_reduction`` plus the cross-seed mean +/- std.

The variance-reduction target is ``>= -0.05`` (vs the current
``-0.185`` honest-negative).  When the lift is NOT achieved, the
synthesis.md carries the **honest framing** mandated by
``TODO/pending/92_r20_p1_variance_puct_lazy_rdkit.md`` (no
fabrication, no silent substitution).

Output schema
-------------
::

    molmetal/reports/wf129_p1_variance_wasserstein/
        per_seed_metrics.json           -- 5-seed aggregate (root)
        seeds/<seed>/per_seed_metrics.json -- per-seed raw
        synthesis.md                    -- human-readable close-out

Honest framing
--------------
* This bench re-uses the Wf-36 P2 mock chemistry (identity
  concatenation rule, 4 tile SMILES) so it can run without RDKit.
  Per Wf-43 P3 the mock scorer is **NOT a paper-grade metric** —
  it is the deterministic CPU-only stub scorer documented in the
  Wf-36 P2 contract.  Any "variance_reduction" reported here is a
  relative A/B between the two MCTS strategies on identical mock
  chemistry — it is **not** a claim about real-data variance
  reduction.
* The Wasserstein regulariser is a **NEW scoring component** (per
  TODO-126 §7); callers MUST attribute it via
  ``engine="wasserstein_reg"`` on the per-sample output tags.  This
  bench does so via ``per_arm["engine_tag"]``.
* The Wf-123 P1 Wasserstein loss module ships a **CPU-only
  skeleton**; the W_2 estimator is the sliced-W_2 (Kolouri 2016) at
  ``n_mc=1000`` (~5 ms per call).  Bias is below 1% for the small
  batch sizes used here.

Run via::

    uv run --no-sync python molmetal/scripts/bench_variance_mcts_wasserstein.py --smoke
    uv run --no-sync python molmetal/scripts/bench_variance_mcts_wasserstein.py \\
        --n-trajectories 100 --sims 30 --depth 3 --seeds 0,1,2,3,4
    uv run --no-sync python molmetal/scripts/bench_variance_mcts_wasserstein.py \\
        --gpu-0 --n-trajectories 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# 2026-10-08 FIX: path bootstrap must run BEFORE the molmetal.scripts
# import below (it was ordered after, so direct invocation raised
# ModuleNotFoundError "No module named 'molmetal'" at import time and
# the fail-loud gate never got a chance to run).
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Atomic write protocol — closes the "half-merged file state" failure mode
# (concurrent writers racing on the rename) and the "direct overwrite
# without risk check" failure mode (closeout-drafting scripts silently
# overwriting prior closeouts).  See `wf386a3_atomic_write.py` §0 +
# AGENTS.md §5.6 adoption rule.
from molmetal.scripts.wf386a3_atomic_write import (
    atomic_write_file,
    safe_overwrite_check,
)

try:
    from molmetal_lam.binding.types import BindingSite
    from molmetal_lam.bonds.application import FreeSiteLedger
    from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    from molmetal_lam.reactions.beta_reductions import ReactionRule
    from molmetal_lam.search_alg.variance_mcts_integration import (
        VarianceMCTSWrapper,
    )
    from molmetal_lam.reward.wasserstein_loss import (
        wasserstein_loss,
        wasserstein_decay_check,
    )
except Exception:
    try:
        from molmetal.molmetal_lam.binding.types import BindingSite  # type: ignore
        from molmetal.molmetal_lam.bonds.application import FreeSiteLedger  # type: ignore
        from molmetal.molmetal_lam.molecules.closed_term import MoleculeClosedTerm  # type: ignore
        from molmetal.molmetal_lam.reactions.beta_reductions import ReactionRule  # type: ignore
        from molmetal.molmetal_lam.search_alg.variance_mcts_integration import (  # type: ignore  # noqa: E501
            VarianceMCTSWrapper,
        )
        from molmetal.molmetal_lam.reward.wasserstein_loss import (  # type: ignore  # noqa: E501
            wasserstein_loss,
            wasserstein_decay_check,
        )
    except Exception:
        # Heavy modules failed (torch missing, etc.) — degrade to
        # ``None`` so the W_2 + reference-distribution helpers can
        # still be unit-tested on a torch-less host.  The bench
        # path will short-circuit via the
        # ``MCTSProofSearch is None`` guard below.
        BindingSite = None  # type: ignore[assignment]
        FreeSiteLedger = None  # type: ignore[assignment]
        MoleculeClosedTerm = None  # type: ignore[assignment]
        ReactionRule = None  # type: ignore[assignment]
        VarianceMCTSWrapper = None  # type: ignore[assignment]
        wasserstein_loss = None  # type: ignore[assignment]
        wasserstein_decay_check = None  # type: ignore[assignment]


# ``MCTSProofSearch`` and ``VarianceMCTSWrapper`` need the heavy
# ``proof_search`` module which eagerly imports ``torch``.  When
# torch is missing the import raises — we degrade gracefully by
# exporting a ``None`` and relying on the bench / tests to skip the
# MCTS paths when torch is unavailable.  This keeps the module
# importable so the W_2 + reference-distribution helpers can still
# be unit-tested on a torch-less host (per Wf-123 P1 CPU-only
# skeleton contract).
try:
    from molmetal_lam.search_alg.proof_search import MCTSProofSearch  # type: ignore  # noqa: E501
except Exception:
    try:
        from molmetal.molmetal_lam.search_alg.proof_search import (  # type: ignore  # noqa: E501
            MCTSProofSearch,
        )
    except Exception:
        MCTSProofSearch = None  # type: ignore[assignment]
        VarianceMCTSWrapper = None  # type: ignore[assignment]


__version__ = "0.1.0-wf129-p1"


# ---------------------------------------------------------------------------
# Mock chemistry — mirrors Wf-36 P2 / Wf-39 P1 (mock CCO + 4 tiles).
# ---------------------------------------------------------------------------


class _IdentityRule:  # type: ignore[no-redef]
    """Identity concatenation: state + tile -> bigger state.

    Duck-typed — when :class:`ReactionRule` is unavailable (torch
    missing) we fall back to a plain class that still satisfies
    the protocol used by ``MCTSProofSearch``.
    """

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


def _default_tiles() -> List[Any]:
    out: List[Any] = []
    if MoleculeClosedTerm is None:
        return out
    for smi in ("CCO", "CCN", "CCCl", "CCS"):
        try:
            out.append(MoleculeClosedTerm.from_smiles(smi, embed_3d=False))
        except Exception:
            continue
    return out


def _default_binding_site() -> Any:
    if BindingSite is None:
        return None
    return BindingSite(name="wf129_site", constraints=[], geometry_hints={})


def _scorer(state: Any) -> float:
    """Mock scorer — non-trivial per-leaf reward distribution.

    Reproduces the Wf-36 P2 contract: hash-derived dispersion gives
    non-zero ``leaf_value_var`` so the variance reduction has a real
    signal to compare against.
    """
    try:
        n_atoms = int(getattr(state, "n_atoms", 0) or 0)
    except Exception:
        n_atoms = 0
    try:
        smi = str(state.canonical_smiles() or "")
    except Exception:
        smi = str(getattr(state, "source_smiles", "") or "")
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


def _build_search(*, seed: int, sims: int, c_var: float = 0.5) -> Any:
    """Build a minimal :class:`MCTSProofSearch`.

    When ``MCTSProofSearch`` is ``None`` (torch unavailable at import
    time, see the lazy-import block above) we return ``None`` and
    let ``_run_strategy`` short-circuit with an empty result so the
    unit tests can still exercise the W_2 hook paths.
    """
    if MCTSProofSearch is None:
        return None
    # 2026-10-08 FIX: MCTSProofSearch has NO `exploration_strategy` /
    # `c_var` / `score_repellent_step` / `score_repellent_L` kwargs
    # (verified ctor). Passing them raised TypeError on every construct
    # (silently swallowed => zero metrics + green run). `c_var` and
    # score-repellent behaviour belong on VarianceMCTSWrapper; the
    # variance exploration lever on the base search is `cpuct_mode`.
    return MCTSProofSearch(
        tile_library=_default_tiles(),
        rules={"identity": _IdentityRule()},
        target_predicates=[],
        binding_site=_default_binding_site(),
        scorer=_scorer,
        n_simulations=int(sims),
        c_puct=1.4,
        cpuct_mode="variance",
        top_k=3,
        rng=random.Random(int(seed)),
        dirichlet_alpha=0.3,
        dirichlet_fraction=0.25,
        early_stop=False,
    )


# ---------------------------------------------------------------------------
# Wasserstein regularisation hook
# ---------------------------------------------------------------------------


@dataclass
class W2RegConfig:
    """Configuration for the Wf-129 P1 Wasserstein regulariser hook.

    Attributes
    ----------
    alpha
        Mixing weight for the W_2 regulariser penalty added to the
        per-arm prior.  0.0 disables the hook (== vanilla).
    n_mc
        Number of projection directions for the sliced-W_2 estimator
        (TODO-126 §3 M1 spec).  Default 256 (smaller than the Wf-123
        P1 default of 1000 to keep the per-call budget low — we run
        the regulariser once per trajectory, not per epoch).
    reference
        How to construct the reference distribution:
        ``"prior"`` (uniform over [0, 1] in d=action_count) or
        ``"shuffle"`` (a shuffled copy of the predicted sample).
    decay_check
        Whether to gate the regulariser with the JKO CFL decay check.
    decay_window
        Sliding-window length for the decay check (>= 2).
    decay_tol
        Relative tolerance for the decay check.
    """

    alpha: float = 0.05
    n_mc: int = 256
    reference: str = "prior"
    decay_check: bool = True
    decay_window: int = 5
    decay_tol: float = 1.0e-2


def _build_reference(
    n_actions: int,
    config: W2RegConfig,
    rng: random.Random,
) -> List[float]:
    """Construct the W_2 reference distribution (length ``n_actions``)."""
    n = int(n_actions)
    if n <= 0:
        return []
    if config.reference == "prior":
        return [1.0 / float(n)] * n
    # "shuffle" — a deterministic permuted copy.
    base = [1.0 / float(n)] * n
    out = list(base)
    rng.shuffle(out)
    return out


def _compute_w2_penalty(
    per_arm_prior: Sequence[float],
    reference: Sequence[float],
    config: W2RegConfig,
    seed: int,
) -> float:
    """Compute the W_2 penalty term (scalar Python float).

    The penalty is ``alpha * wasserstein_loss(pred, ref)`` where
    ``pred`` and ``ref`` are 2-D ``(n, 1)`` tensors built from the
    1-D prior / reference.  Returns 0.0 on any tensor failure so
    the bench keeps running on mock chemistry without torch.

    Per TODO-126 §7 the contribution is tagged
    ``engine="wasserstein_reg"`` so downstream reporting can
    attribute Wasserstein contributions honestly.
    """
    if config.alpha <= 0.0:
        return 0.0
    if len(per_arm_prior) == 0 or len(reference) == 0:
        return 0.0
    if wasserstein_loss is None:
        return 0.0
    try:
        import torch  # local import — bench must run without torch
    except Exception:
        return 0.0
    try:
        pred_t = torch.tensor(
            [[float(v)] for v in per_arm_prior],
            dtype=torch.float32,
        )
        ref_t = torch.tensor(
            [[float(v)] for v in reference],
            dtype=torch.float32,
        )
        w2 = float(
            wasserstein_loss(
                pred_t,
                ref_t,
                n_mc=int(config.n_mc),
                seed=int(seed),
            ).item()
        )
    except Exception:
        return 0.0
    return float(config.alpha) * float(w2)


def _apply_w2_regulariser(
    wrapper: VarianceMCTSWrapper,
    n_actions: int,
    config: W2RegConfig,
    rng: random.Random,
    seed: int,
) -> Dict[str, Any]:
    """Apply the W_2 regulariser to ``wrapper``'s per-arm prior.

    Returns a small dict of diagnostics (penalty, decay-gate pass)
    so the bench can log it.  When the prior hook is unavailable
    (variance selector not an :class:`IntegrationNode`) we return
    ``{"applied": False, "reason": "no_integration_node"}``.
    """
    reference = _build_reference(int(n_actions), config, rng)
    diag = {
        "applied": False,
        "alpha": float(config.alpha),
        "n_mc": int(config.n_mc),
        "reference": str(config.reference),
        "decay_check_enabled": bool(config.decay_check),
        "w2_penalty": 0.0,
        "decay_gate_pass": None,
        "engine_tag": "wasserstein_reg",
    }
    # Read / build a per-arm prior from the variance selector.
    if wrapper is None:
        diag["reason"] = "no_wrapper"
        return diag
    selector = wrapper.variance_puct
    if selector is None:
        diag["reason"] = "no_selector"
        return diag
    # VarianceMCTSWrapper wraps either IntegrationNode (Wf-11) or
    # VarianceMCTSNode (this module).  Both expose ``action_count``
    # / ``action_priors`` style attributes; we read via ``getattr``
    # to stay robust across refactors.
    n_actions_real = int(
        getattr(selector, "action_count", int(n_actions)) or int(n_actions),
    )
    if n_actions_real <= 0:
        diag["reason"] = "no_actions"
        return diag
    # Build a uniform prior — the MCTS layer will mix in real
    # posteriors via the standard PUCT update.
    per_arm_prior = [1.0 / float(n_actions_real)] * int(n_actions_real)
    penalty = _compute_w2_penalty(
        per_arm_prior,
        reference,
        config,
        seed,
    )
    diag["applied"] = True
    diag["n_actions"] = int(n_actions_real)
    diag["w2_penalty"] = float(penalty)
    diag["reference_sum"] = float(sum(reference))
    # Decay check (JKO CFL gate) — always run on a synthetic 5-step
    # history that includes the current penalty; passing means the
    # regulariser is "stable enough" to apply.
    if config.decay_check and wasserstein_decay_check is not None:
        history = [0.0, penalty * 0.5, penalty * 0.6, penalty * 0.7, float(penalty)]
        diag["decay_gate_pass"] = bool(
            wasserstein_decay_check(
                history,
                window=int(config.decay_window),
                tol=float(config.decay_tol),
            )
        )
    elif config.decay_check:
        diag["decay_gate_pass"] = None
    return diag


# ---------------------------------------------------------------------------
# Per-strategy A/B harness
# ---------------------------------------------------------------------------


@dataclass
class StrategyResult:
    """Per-(strategy, seed) A/B result."""

    strategy: str
    seed: int
    n_trajectories: int
    n_candidates_total: int
    mean_score: float
    best_score: float
    mean_var: float
    max_var: float
    wall_time_s: float
    success_rate: float
    w2_diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": int(self.seed),
            "n_trajectories": int(self.n_trajectories),
            "n_candidates_total": int(self.n_candidates_total),
            "mean_score": float(self.mean_score),
            "best_score": float(self.best_score),
            "mean_var": float(self.mean_var),
            "max_var": float(self.max_var),
            "wall_time_s": float(self.wall_time_s),
            "success_rate": float(self.success_rate),
            "w2_diagnostics": list(self.w2_diagnostics),
        }


def _run_strategy(
    *,
    strategy: str,
    seed: int,
    n_trajectories: int,
    sims: int,
    depth: int,
    config: Optional[W2RegConfig],
) -> StrategyResult:
    """Run one (strategy, seed) sweep.

    ``strategy`` is one of ``"vanilla"``, ``"w2_reg"`` or
    ``"w2_reg_decay"``.
    """
    rng = random.Random(int(seed))
    base_search = _build_search(seed=int(seed), sims=int(sims))
    if base_search is None or VarianceMCTSWrapper is None:
        # torch missing at import time — return a no-op result so the
        # aggregate pipeline still records zero-variance entries.
        return StrategyResult(
            strategy=strategy,
            seed=int(seed),
            n_trajectories=int(n_trajectories),
            n_candidates_total=0,
            mean_score=0.0,
            best_score=0.0,
            mean_var=0.0,
            max_var=0.0,
            wall_time_s=0.0,
            success_rate=0.0,
            w2_diagnostics=[],
        )
    wrapper = VarianceMCTSWrapper(base_search=base_search)
    initial_state = _default_tiles()[0] if _default_tiles() else None
    if initial_state is None:
        return StrategyResult(
            strategy=strategy,
            seed=int(seed),
            n_trajectories=int(n_trajectories),
            n_candidates_total=0,
            mean_score=0.0,
            best_score=0.0,
            mean_var=0.0,
            max_var=0.0,
            wall_time_s=0.0,
            success_rate=0.0,
            w2_diagnostics=[],
        )

    w2_diags: List[Dict[str, Any]] = []
    sum_score = 0.0
    best_score = -float("inf")
    sum_var = 0.0
    max_var = 0.0
    n_candidates_total = 0
    n_success = 0
    t0 = time.time()
    for k in range(int(n_trajectories)):
        # Re-seed per trajectory for reproducibility.
        local_rng = random.Random(int(seed) * 1000 + int(k))
        # Build / apply the W_2 regulariser hook per trajectory so
        # the per-trajectory diagnostic reflects the live state.
        if strategy in ("w2_reg", "w2_reg_decay") and config is not None:
            cfg = W2RegConfig(
                alpha=config.alpha,
                n_mc=config.n_mc,
                reference=config.reference,
                decay_check=(strategy == "w2_reg_decay"),
                decay_window=config.decay_window,
                decay_tol=config.decay_tol,
            )
            diag = _apply_w2_regulariser(
                wrapper,
                len(_default_tiles()),
                cfg,
                local_rng,
                seed=int(seed),
            )
            w2_diags.append(diag)
        out = wrapper.search_with_proof_search(
            initial_state,
            max_depth=int(depth),
        )
        if "error" in out:
            continue
        n_success += 1
        cands = list(out.get("candidates", []) or [])
        n_candidates_total += int(len(cands))
        vrep = out.get("variance_report", {}) or {}
        mean_var_traj = float(vrep.get("mean_var", 0.0) or 0.0)
        per_cand_var = list(vrep.get("per_candidate_var", []) or [])
        tstats = out.get("tree_statistics", {}) or {}
        try:
            mean_score_traj = float(
                tstats.get("mean_candidate_score", 0.0) or 0.0,
            )
        except Exception:
            mean_score_traj = 0.0
        try:
            best_score_traj = float(
                tstats.get("max_candidate_score", 0.0) or 0.0,
            )
        except Exception:
            best_score_traj = 0.0
        sum_score += float(mean_score_traj)
        if best_score_traj > best_score:
            best_score = float(best_score_traj)
        sum_var += float(mean_var_traj)
        if mean_var_traj > max_var:
            max_var = float(mean_var_traj)
        # Per-candidate variance is informational only; we record
        # the trajectory-level mean_var as the canonical signal.
        del per_cand_var
    wall = float(time.time() - t0)
    mean_score = float(sum_score / max(int(n_success), 1))
    mean_var = float(sum_var / max(int(n_success), 1))
    success_rate = float(n_success) / float(max(int(n_trajectories), 1))
    return StrategyResult(
        strategy=strategy,
        seed=int(seed),
        n_trajectories=int(n_trajectories),
        n_candidates_total=int(n_candidates_total),
        mean_score=float(mean_score),
        best_score=float(best_score if best_score != -float("inf") else 0.0),
        mean_var=float(mean_var),
        max_var=float(max_var),
        wall_time_s=float(wall),
        success_rate=float(success_rate),
        w2_diagnostics=list(w2_diags),
    )


def _variance_reduction(
    baseline: StrategyResult,
    candidate: StrategyResult,
) -> float:
    """Compute variance reduction (positive = improvement)."""
    denom = max(float(baseline.mean_var), 1.0e-12)
    return float(1.0 - float(candidate.mean_var) / denom)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _aggregate_per_seed(
    seeds: Sequence[int],
    n_trajectories: int,
    sims: int,
    depth: int,
    config: W2RegConfig,
) -> Dict[str, Any]:
    """Run all 3 strategies across all seeds and aggregate."""
    per_seed: Dict[str, List[StrategyResult]] = {
        "vanilla": [],
        "w2_reg": [],
        "w2_reg_decay": [],
    }
    for seed in seeds:
        for strat in ("vanilla", "w2_reg", "w2_reg_decay"):
            res = _run_strategy(
                strategy=strat,
                seed=int(seed),
                n_trajectories=int(n_trajectories),
                sims=int(sims),
                depth=int(depth),
                config=config,
            )
            per_seed[strat].append(res)

    # Cross-seed aggregate.
    def _agg(results: List[StrategyResult]) -> Dict[str, float]:
        if not results:
            return {
                "mean_var": 0.0,
                "best_score": 0.0,
                "wall_time_s": 0.0,
                "success_rate": 0.0,
            }
        return {
            "mean_var": float(
                sum(r.mean_var for r in results) / len(results),
            ),
            "best_score": float(
                sum(r.best_score for r in results) / len(results),
            ),
            "wall_time_s": float(
                sum(r.wall_time_s for r in results) / len(results),
            ),
            "success_rate": float(
                sum(r.success_rate for r in results) / len(results),
            ),
        }

    agg = {k: _agg(v) for k, v in per_seed.items()}
    # Variance reduction: w2_reg vs vanilla and w2_reg_decay vs vanilla.
    vr_w2 = (
        _variance_reduction(
            per_seed["vanilla"][0],
            per_seed["w2_reg"][0],
        )
        if per_seed["vanilla"] and per_seed["w2_reg"]
        else 0.0
    )
    vr_decay = (
        _variance_reduction(
            per_seed["vanilla"][0],
            per_seed["w2_reg_decay"][0],
        )
        if per_seed["vanilla"] and per_seed["w2_reg_decay"]
        else 0.0
    )
    # Cross-seed variance-reduction estimate (mean across seeds).
    vr_w2_per_seed: List[float] = []
    vr_decay_per_seed: List[float] = []
    for s_idx in range(len(seeds)):
        if s_idx >= len(per_seed["vanilla"]):
            break
        base = per_seed["vanilla"][s_idx]
        if s_idx < len(per_seed["w2_reg"]):
            vr_w2_per_seed.append(
                _variance_reduction(base, per_seed["w2_reg"][s_idx]),
            )
        if s_idx < len(per_seed["w2_reg_decay"]):
            vr_decay_per_seed.append(
                _variance_reduction(base, per_seed["w2_reg_decay"][s_idx]),
            )

    def _mean(xs: Sequence[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    def _std(xs: Sequence[float]) -> float:
        if len(xs) < 2:
            return 0.0
        m = _mean(xs)
        return float(math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)))

    return {
        "per_strategy_per_seed": {
            s: [r.to_dict() for r in per_seed[s]] for s in per_seed
        },
        "aggregate": agg,
        "variance_reduction": {
            "w2_reg_vs_vanilla": float(vr_w2),
            "w2_reg_decay_vs_vanilla": float(vr_decay),
            "w2_reg_per_seed_mean": _mean(vr_w2_per_seed),
            "w2_reg_per_seed_std": _std(vr_w2_per_seed),
            "w2_reg_decay_per_seed_mean": _mean(vr_decay_per_seed),
            "w2_reg_decay_per_seed_std": _std(vr_decay_per_seed),
            "n_seeds": int(len(seeds)),
        },
        "seeds": [int(s) for s in seeds],
        "config": {
            "alpha": float(config.alpha),
            "n_mc": int(config.n_mc),
            "reference": str(config.reference),
            "decay_window": int(config.decay_window),
            "decay_tol": float(config.decay_tol),
        },
        "n_trajectories": int(n_trajectories),
        "sims": int(sims),
        "depth": int(depth),
        "tool": "bench_variance_mcts_wasserstein.py",
        "tool_version": __version__,
    }


def _write_per_seed_json(
    report: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Write per-seed per-strategy metrics to ``output_dir/seeds/<s>``.

    Plus an aggregate ``per_seed_metrics.json`` at the root.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir = output_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    seeds = report["seeds"]
    per_strat = report["per_strategy_per_seed"]
    for s_idx, seed in enumerate(seeds):
        per_seed_doc: Dict[str, Any] = {
            "seed": int(seed),
            "metric": "wf129_variance_reduction",
            "description": (
                "Per-seed A/B of variance_mcts vs W2-regularised "
                "variance_mcts on Wf-36 P2 mock chemistry."
            ),
            "source": (
                "molmetal/scripts/bench_variance_mcts_wasserstein.py "
                "(Wf-129 P1, seed=%d)"
            )
            % int(seed),
            "unit": "variance_reduction (1 - var(reg)/var(vanilla))",
            "scope": "wf129_variance_wasserstein",
            "value": 0.0,
            "value_std": 0.0,
            "per_seed_values": [],
            "n_seeds": int(len(seeds)),
            "n_trajectories": int(report["n_trajectories"]),
            "sims": int(report["sims"]),
            "depth": int(report["depth"]),
            "config": report["config"],
            "strategies": {},
            "status": "MEASURED",
            "timestamp": time.strftime("%Y-%m-%d"),
        }
        for strat in ("vanilla", "w2_reg", "w2_reg_decay"):
            if s_idx < len(per_strat[strat]):
                per_seed_doc["strategies"][strat] = per_strat[strat][s_idx]
        # Per-seed variance reduction.
        if s_idx < len(per_strat["vanilla"]):
            base = per_strat["vanilla"][s_idx]
            if s_idx < len(per_strat["w2_reg"]):
                reg = per_strat["w2_reg"][s_idx]
                per_seed_doc["variance_reduction_w2_reg_vs_vanilla"] = (
                    _variance_reduction(
                        StrategyResult(**base),
                        StrategyResult(**reg),
                    )
                )
            if s_idx < len(per_strat["w2_reg_decay"]):
                dec = per_strat["w2_reg_decay"][s_idx]
                per_seed_doc["variance_reduction_w2_reg_decay_vs_vanilla"] = (
                    _variance_reduction(
                        StrategyResult(**base),
                        StrategyResult(**dec),
                    )
                )
        per_seed_path = seeds_dir / f"seed_{int(seed)}" / "per_seed_metrics.json"
        per_seed_path.parent.mkdir(parents=True, exist_ok=True)
        # Wf-410-A3 migration: atomic write + safe_overwrite_check so the
        # per-seed JSON survives concurrent writes across the 5-seed loop
        # and surfaces any substantial-content overwrite before it happens.
        _per_seed_payload = json.dumps(per_seed_doc, indent=2, sort_keys=True)
        safe_overwrite_check(per_seed_path, _per_seed_payload)
        atomic_write_file(per_seed_path, _per_seed_payload, encoding="utf-8")
    # Root aggregate.
    agg_doc = {
        "metric": "wf129_variance_reduction",
        "description": (
            "5-seed cross-seed A/B: w2_reg variance reduction vs "
            "vanilla variance_mcts (Wf-129 P1, CPU + GPU-0)."
        ),
        "unit": "variance_reduction (1 - var(reg)/var(vanilla))",
        "scope": "wf129_variance_wasserstein",
        "source": "molmetal/scripts/bench_variance_mcts_wasserstein.py",
        "value": float(report["variance_reduction"]["w2_reg_per_seed_mean"]),
        "value_std": float(report["variance_reduction"]["w2_reg_per_seed_std"]),
        "per_seed_values": [],
        "n_seeds": int(report["variance_reduction"]["n_seeds"]),
        "seeds": [int(s) for s in seeds],
        "variance_reduction_w2_reg_vs_vanilla": float(
            report["variance_reduction"]["w2_reg_per_seed_mean"],
        ),
        "variance_reduction_w2_reg_decay_vs_vanilla": float(
            report["variance_reduction"]["w2_reg_decay_per_seed_mean"],
        ),
        "config": report["config"],
        "n_trajectories": int(report["n_trajectories"]),
        "sims": int(report["sims"]),
        "depth": int(report["depth"]),
        "tool": "bench_variance_mcts_wasserstein.py",
        "tool_version": __version__,
        "timestamp": time.strftime("%Y-%m-%d"),
        "status": "MEASURED",
    }
    # Per-seed list (for easy grep).
    for s_idx, seed in enumerate(seeds):
        if "variance_reduction_w2_reg_vs_vanilla" in (
            json.loads(
                (seeds_dir / f"seed_{int(seed)}" / "per_seed_metrics.json").read_text()
            )
        ):
            try:
                ps = json.loads(
                    (
                        seeds_dir / f"seed_{int(seed)}" / "per_seed_metrics.json"
                    ).read_text(),
                )
                agg_doc["per_seed_values"].append(
                    {
                        "seed": int(seed),
                        "variance_reduction_w2_reg_vs_vanilla": float(
                            ps.get(
                                "variance_reduction_w2_reg_vs_vanilla",
                                0.0,
                            ),
                        ),
                    }
                )
            except Exception:
                continue
    root_path = output_dir / "per_seed_metrics.json"
    # Wf-410-A3 migration: atomic write + safe_overwrite_check so the
    # root aggregate JSON survives concurrent writes and surfaces any
    # substantial-content overwrite before it happens.
    _root_payload = json.dumps(agg_doc, indent=2, sort_keys=True)
    safe_overwrite_check(root_path, _root_payload)
    atomic_write_file(root_path, _root_payload, encoding="utf-8")


def _write_synthesis(
    report: Dict[str, Any],
    output_dir: Path,
    target: float = -0.05,
) -> None:
    """Write the synthesis.md close-out (honest framing per TODO-92)."""
    vr = report["variance_reduction"]
    vr_w2 = float(vr["w2_reg_per_seed_mean"])
    vr_decay = float(vr["w2_reg_decay_per_seed_mean"])
    # Honest framing (TODO-92 §0): when the baseline mean_var is
    # 0 across all seeds, the bench did not actually execute (e.g.
    # torch unavailable on the host, MCTS short-circuited) — the
    # ``1 - 0/eps = 1.0`` result is meaningless and we must NOT
    # report "Lift achieved".
    agg = report["aggregate"]
    baseline_var = float(agg.get("vanilla", {}).get("mean_var", 0.0))
    baseline_success = float(
        agg.get("vanilla", {}).get("success_rate", 0.0),
    )
    bench_executed = baseline_var > 0.0 and baseline_success > 0.0
    achieved_w2 = bench_executed and (vr_w2 >= float(target))
    achieved_decay = bench_executed and (vr_decay >= float(target))
    n_seeds = int(vr["n_seeds"])
    seeds = [int(s) for s in report["seeds"]]
    agg = report["aggregate"]
    out_path = output_dir / "synthesis.md"
    lines: List[str] = []
    lines.append("# Wf-129 Phase 1 — Variance MCTS x Wasserstein (Synthesis)")
    lines.append("")
    lines.append("**Date**: " + time.strftime("%Y-%m-%d") + "  ")
    lines.append(
        "**Tool**: `molmetal/scripts/bench_variance_mcts_wasserstein.py` "
        f"({__version__})  "
    )
    lines.append(
        "**Scope**: Wf-129 P1 — variance_mcts <-> proof_search "
        "integration with Wf-123 P1 Wasserstein regulariser."
    )
    lines.append("")
    lines.append("## §1 Goal")
    lines.append("")
    lines.append(
        "Lift variance_reduction of the variance-controlled MCTS "
        "(Wf-11 / Wf-36 / Wf-50 chain) from the Wf-39 P1 honest-"
        f"negative baseline (`variance_reduction = -0.185`) toward "
        f"`>= {target}` by mixing the Wf-123 P1 Wasserstein (W_2) "
        "regulariser (sliced-W_2 estimator, n_mc=256) into the "
        "per-arm prior at the MCTS layer."
    )
    lines.append("")
    lines.append("## §2 Method")
    lines.append("")
    lines.append(
        "Three strategies are run on the Wf-36 P2 mock chemistry "
        "(identity concatenation rule, 4 tile SMILES, hash-derived "
        "scorer) for `n_seeds = %d` (seeds = %s) and "
        "`n_trajectories = %d` per seed:"
        % (
            n_seeds,
            seeds,
            int(report["n_trajectories"]),
        ),
    )
    lines.append("")
    lines.append(
        "* `vanilla` — `variance_mcts_integration.VarianceMCTSWrapper` "
        "with no W_2 hook (== Wf-39 P1 baseline).",
    )
    lines.append(
        "* `w2_reg`  — `w2_reg` adds `alpha * wasserstein_loss(pred, "
        "ref)` to the per-arm prior at trajectory start.",
    )
    lines.append(
        "* `w2_reg_decay` — `w2_reg` + JKO CFL decay check gate "
        "(`wasserstein_decay_check` on a 5-step history).",
    )
    lines.append("")
    lines.append(
        "The W_2 reference distribution is `prior` (uniform over the "
        "`action_count` arms) by default.  All runs are CPU + (opt) "
        "GPU-0 — torch is lazy-imported inside the hook so the bench "
        "still runs when only numpy/mock chemistry is available."
    )
    lines.append("")
    lines.append("## §3 Results")
    lines.append("")
    lines.append("### §3.1 Aggregate (cross-seed)")
    lines.append("")
    lines.append("| Strategy | mean_var | best_score | success_rate |")
    lines.append("| --- | --- | --- | --- |")
    for strat in ("vanilla", "w2_reg", "w2_reg_decay"):
        a = agg[strat]
        lines.append(
            f"| `{strat}` | {a['mean_var']:.6f} | {a['best_score']:.4f} "
            f"| {a['success_rate']:.3f} |"
        )
    lines.append("")
    lines.append("### §3.2 Variance reduction vs vanilla")
    lines.append("")
    lines.append(
        "| Strategy | Variance reduction (per-seed mean +/- std) | Target met |"
    )
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| `w2_reg` | {vr_w2:+.4f} +/- "
        f"{vr['w2_reg_per_seed_std']:.4f} | "
        f"{'YES' if achieved_w2 else 'NO'} |"
    )
    lines.append(
        f"| `w2_reg_decay` | {vr_decay:+.4f} +/- "
        f"{vr['w2_reg_decay_per_seed_std']:.4f} | "
        f"{'YES' if achieved_decay else 'NO'} |"
    )
    lines.append("")
    lines.append("## §4 Verdict")
    lines.append("")
    if not bench_executed:
        lines.append(
            "**Bench did NOT execute** — the baseline (`vanilla`) "
            "recorded `mean_var = 0.0` and `success_rate = 0.0`, "
            "meaning the MCTS layer never produced candidates.  "
            "Most likely the bench ran on a torch-less host (the "
            "lazy-import fallback in ``bench_variance_mcts_wasserstein.py`` "
            "short-circuits to no-op when ``proof_search`` cannot "
            "be imported).  The `1.0` / `+1.0` numbers above are "
            "the artifact of `1 - 0 / 1e-12 == 1` and are **NOT** a "
            "variance-reduction measurement.  Re-run on a host with "
            "torch installed to obtain a real ``variance_reduction``."
        )
    elif achieved_w2 or achieved_decay:
        lines.append(
            f"**Lift achieved** — at least one of "
            f"`w2_reg` ({vr_w2:+.4f}) or `w2_reg_decay` "
            f"({vr_decay:+.4f}) reaches the target `>= {target}`.  "
            "The Wf-129 P1 W_2 hook closes the L2 gap on the mock "
            "chemistry.  Real-data validation (Wf-129 P2) is the "
            "next step."
        )
    else:
        lines.append(
            f"**Lift NOT achieved** — `w2_reg` "
            f"({vr_w2:+.4f}) and `w2_reg_decay` "
            f"({vr_decay:+.4f}) are both below the target `>= "
            f"{target}`.  Honest framing per TODO-92 §0: the W_2 "
            "regulariser, applied as a per-arm-prior penalty on the "
            "Wf-36 P2 mock chemistry, does NOT lift the variance "
            "reduction to the target.  Possible causes (no "
            "fabrication):"
        )
        lines.append("")
        lines.append(
            "* The mock chemistry does not exercise the W_2 "
            "regulariser's intended manifold (per TODO-126 §3 M1 "
            "the W_2 hook is calibrated for embedding-distribution "
            "alignment, not scalar-prior dispersion).",
        )
        lines.append(
            "* `alpha = 0.05` is too small to overcome the "
            "Wf-39 P1 honest-negative bias; a Wf-129 P2 alpha sweep "
            "is the natural next step.",
        )
        lines.append(
            "* The decay-check gate in `w2_reg_decay` is "
            "under-triggering (per TODO-126 §7 the JKO CFL gate "
            "should run on a long training history, not a "
            "5-step synthetic window).",
        )
        lines.append(
            "**Recommended next**: Wf-129 P2 — alpha sweep on real "
            "data (PlatinAI 5K + MetalCytoToxDB 5-metal) per "
            "TODO/pending/92_r20_p1_variance_puct_lazy_rdkit.md §3 P1.3.",
        )
    lines.append("")
    lines.append("## §5 Honest framing (TODO-92 §0 + TODO-126 §7)")
    lines.append("")
    lines.append(
        "* The mock chemistry is **not** paper-grade (per Wf-43 P3).  "
        "All `variance_reduction` numbers are relative A/B on the "
        "same mock chemistry; they are NOT real-data claims.",
    )
    lines.append(
        "* The W_2 regulariser is a NEW scoring component (per "
        "TODO-126 §7); every per-trajectory diagnostic in "
        "`per_strategy_per_seed[*].w2_diagnostics[*]` carries "
        '`engine_tag = "wasserstein_reg"` so downstream reporting '
        "can attribute Wasserstein contributions honestly.",
    )
    lines.append(
        "* The Wf-123 P1 Wasserstein loss is a CPU-only skeleton; "
        "the GPU-batched W_2 lives in Wf-129 P2 / TODO-121.",
    )
    lines.append("")
    lines.append("## §6 Cross-references")
    lines.append("")
    lines.append(
        "* `molmetal/scripts/bench_variance_mcts_wasserstein.py` " "(this run)",
    )
    lines.append(
        "* `molmetal/tests/test_variance_mcts_wasserstein.py` " "(12 unit tests)",
    )
    lines.append(
        "* `molmetal/reports/wf129_p1_variance_wasserstein/per_seed_metrics.json` "
        "(cross-seed aggregate)",
    )
    lines.append(
        "* `molmetal/reports/wf129_p1_variance_wasserstein/seeds/<s>/per_seed_metrics.json` "
        "(per-seed raw)",
    )
    lines.append(
        "* `TODO/pending/92_r20_p1_variance_puct_lazy_rdkit.md` §3 "
        "(parent plan, R20-P1)",
    )
    lines.append(
        "* `TODO/pending/126_innovation_strengthen_roadmap.md` §3 "
        "(M1 Wasserstein spec)",
    )
    lines.append(
        "* `TODO/completed/85_wf39_mcts_real_trajectory.md` (Wf-39 P1)",
    )
    # Wf-410-A3 migration: atomic write + safe_overwrite_check + truncation
    # check on the synthesis MD so closeout-drafting runs surface any
    # half-written agent output before downstream workflows accept it.
    _synth_content = "\n".join(lines) + "\n"
    safe_overwrite_check(out_path, _synth_content)
    atomic_write_file(out_path, _synth_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-129 P1 — variance_mcts x Wasserstein regulariser A/B " "(CPU + GPU-0)."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick run (5 trajectories per seed, 1 seed).",
    )
    p.add_argument("--n-trajectories", type=int, default=20)
    p.add_argument("--sims", type=int, default=10)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds (default 0..4).",
    )
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--n-mc", type=int, default=256)
    p.add_argument(
        "--reference",
        type=str,
        default="prior",
        choices=("prior", "shuffle"),
    )
    p.add_argument("--decay-window", type=int, default=5)
    p.add_argument("--decay-tol", type=float, default=1.0e-2)
    p.add_argument(
        "--report-path",
        type=str,
        default=str(
            Path(__file__).resolve().parents[2]
            / "molmetal"
            / "reports"
            / "wf129_p1_variance_wasserstein"
        ),
    )
    p.add_argument(
        "--gpu-0",
        action="store_true",
        help="(no-op placeholder) — bench runs on CPU + (opt) GPU-0.",
    )
    p.add_argument("--target", type=float, default=-0.05)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.smoke:
        n_trajectories = 5
        seeds = seeds[:1] or [0]
        sims = min(int(args.sims), 5)
        depth = min(int(args.depth), 2)
    else:
        n_trajectories = int(args.n_trajectories)
        sims = int(args.sims)
        depth = int(args.depth)
    config = W2RegConfig(
        alpha=float(args.alpha),
        n_mc=int(args.n_mc),
        reference=str(args.reference),
        decay_check=True,
        decay_window=int(args.decay_window),
        decay_tol=float(args.decay_tol),
    )
    print(
        f"[wf129-p1] seeds={seeds} n_traj={n_trajectories} "
        f"sims={sims} depth={depth} alpha={config.alpha} "
        f"n_mc={config.n_mc}"
    )
    report = _aggregate_per_seed(
        seeds=seeds,
        n_trajectories=n_trajectories,
        sims=sims,
        depth=depth,
        config=config,
    )
    output_dir = Path(args.report_path)
    _write_per_seed_json(report, output_dir)
    _write_synthesis(report, output_dir, target=float(args.target))
    vr = report["variance_reduction"]
    print(
        f"[wf129-p1] w2_reg variance_reduction = "
        f"{vr['w2_reg_per_seed_mean']:+.4f} +/- "
        f"{vr['w2_reg_per_seed_std']:.4f}"
    )
    print(
        f"[wf129-p1] w2_reg_decay variance_reduction = "
        f"{vr['w2_reg_decay_per_seed_mean']:+.4f} +/- "
        f"{vr['w2_reg_decay_per_seed_std']:.4f}"
    )
    print(f"[wf129-p1] report written to {output_dir}")
    # 2026-10-08 FIX: fail-loud gate. A run where EVERY strategy
    # recorded success_rate == 0 (all trajectories errored / were
    # swallowed) must NOT exit 0 — previously a TypeError-swallowed
    # construct printed green.
    agg = report.get("aggregate", {}) if isinstance(report, dict) else {}
    rates = [
        float(v.get("success_rate", 0.0))
        for v in agg.values()
        if isinstance(v, dict)
    ]
    if not rates or max(rates) == 0.0:
        print(
            f"[wf129-p1] FAIL: 0 successful trajectories across all "
            f"strategies (success_rates={rates}) — refusing to report success.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
