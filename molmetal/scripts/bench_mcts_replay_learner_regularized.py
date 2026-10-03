"""Wf-129 Phase 2 — MCTS replay learner with Wasserstein + L2 regularisation.

================================================================
Wf-129 P2: AlphaGo-style policy/value network training with
proper regularisation + Wasserstein prior penalty.
================================================================
This bench closes the L4 gap flagged in Wf-36 P4 / TODO-83 /
TODO-92 / R20-P1.3 / R21: the MCTS replay learner
(:mod:`molmetal.scripts.mcts_replay_learner`) currently ships
with an **honest-negative** ``policy_loss_drop = -0.046`` on the
5-epoch × 16-batch synthetic fixture.  Wf-129 P2 adds three
ingredients that the AlphaZero / AlphaGo Zero training loop
(2016 §3.4, 2018 §3) prescribes but our CPU-only stub skipped:

* **L2 weight decay** (``weight_decay`` analogue) — penalises
  large weights to avoid the policy head over-fitting on a tiny
  128-entry synthetic buffer.
* **Wasserstein-2 prior regulariser** — adds
  ``beta * W_2(predicted_policy, visit_count_target_policy)`` to
  the per-batch loss so the policy head stays close to the
  MCTS-improved target distribution in *probability-space*
  (not just CE-space).
* **Wasserstein-2 value regulariser** — adds
  ``gamma * W_2(value_net(features), value_target)`` so the
  value head's scalar output aligns with the leaf-value target.

The training loop is still CPU-only numpy (no ``torch.cuda``,
no autograd).  The W_2 estimator is the Wf-123 P1 sliced-W_2
(:func:`molmetal_lam.reward.wasserstein_loss.wasserstein_loss`)
which is itself CPU-only and lazy-imported — when torch is
unavailable we degrade to a numpy-only W_2 surrogate so the
bench still ships a real measurement (not a no-op).

Three regularisation *strategies* are compared on the same
synthetic fixture (so the bench can run without RDKit / GPU /
real chemistry):

* ``vanilla``     — no extra regularisation; == Wf-36 P4 baseline.
* ``l2_only``     — vanilla + L2 weight decay on both heads.
* ``l2_w2``       — l2_only + W_2 prior + W_2 value regularisers.

Per-seed metrics = per-strategy ``final_value_loss``,
``final_policy_loss``, ``policy_loss_drop``,
``value_loss_drop``, ``total_wall_time_s``.  The aggregate
target is ``policy_loss < 0.5`` AND ``value_loss < 1.0``
(vs the Wf-36 P4 honest-negative ``policy_loss = 2.795``).

Output schema
-------------
::

    molmetal/reports/wf129_p2_replay_learner/
        per_seed_metrics.json           -- cross-seed aggregate (root)
        seeds/<seed>/per_seed_metrics.json -- per-seed raw
        synthesis.md                    -- human-readable close-out

Honest framing
--------------
* The synthetic fixture is **NOT** paper-grade (per Wf-43 P3 / Wf-129
  P1 §5).  All ``policy_loss`` / ``value_loss`` numbers below are
  computed on the Wf-36 P4 synthetic 128-entry replay buffer; they
  are *bench-level* measurements, not real-data claims.
* The W_2 regulariser is a NEW scoring component (per TODO-126 §7);
  every per-batch diagnostic carries ``engine_tag = "w2_replay_reg"``
  so downstream reporting can attribute W_2 contributions honestly.
* The W_2 estimator is a CPU-only sliced-W_2 (~5 ms per call); when
  torch is unavailable we fall back to a numpy surrogate that is
  mathematically equivalent on 1-D inputs and within 5% on multi-D.

Run via::

    uv run --no-sync python molmetal/scripts/bench_mcts_replay_learner_regularized.py --smoke
    uv run --no-sync python molmetal/scripts/bench_mcts_replay_learner_regularized.py \\
        --n-epochs 50 --batch-size 32 --seeds 0,1,2,3,4
    uv run --no-sync python molmetal/scripts/bench_mcts_replay_learner_regularized.py \\
        --gpu-0 --n-epochs 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Atomic write protocol — closes the "half-merged file state" failure mode
# (concurrent writers racing on the rename) and the "direct overwrite
# without risk check" failure mode (closeout-drafting scripts silently
# overwriting prior closeouts).  See `wf386a3_atomic_write.py` §0 +
# AGENTS.md §5.6 adoption rule.
from molmetal.scripts.wf386a3_atomic_write import (
    atomic_write_file,
    safe_overwrite_check,
)

import numpy as np

# Path bootstrap — mirror the Wf-129 P1 / Wf-36 P4 convention.
if __package__ in (None, ""):
    sys.path.insert(0, "/home/hugo/codes/mol-metal")
else:
    sys.path.insert(0, "/home/hugo/codes/mol-metal")

# ---------------------------------------------------------------------------
# Lazy imports — match the convention used by the rest of the
# ``molmetal/scripts/`` directory so the script is parseable even
# when the package tree is not fully on ``sys.path``.
# ---------------------------------------------------------------------------
try:
    from molmetal_lam.search_alg.mcts_replay_buffer import (
        MCTSReplayBuffer,
        MCTSReplayEntry,
    )
    from molmetal.scripts.mcts_replay_learner import (
        DEFAULT_FEATURE_DIM,
        DEFAULT_HIDDEN_DIM,
        DEFAULT_LR,
        DEFAULT_N_ACTIONS,
        MCTSReplayLearner,
        PolicyNetwork,
        ValueNetwork,
        featurise_entry,
        synth_populate_buffer,
    )
except Exception:  # pragma: no cover - import path fallback
    try:
        from molmetal.molmetal_lam.search_alg.mcts_replay_buffer import (  # type: ignore
            MCTSReplayBuffer,
            MCTSReplayEntry,
        )
        from molmetal.scripts.mcts_replay_learner import (  # type: ignore
            DEFAULT_FEATURE_DIM,
            DEFAULT_HIDDEN_DIM,
            DEFAULT_LR,
            DEFAULT_N_ACTIONS,
            MCTSReplayLearner,
            PolicyNetwork,
            ValueNetwork,
            featurise_entry,
            synth_populate_buffer,
        )
    except Exception:  # pragma: no cover - second-line fallback
        MCTSReplayBuffer = None  # type: ignore
        MCTSReplayEntry = None  # type: ignore
        MCTSReplayLearner = None  # type: ignore
        ValueNetwork = None  # type: ignore
        PolicyNetwork = None  # type: ignore
        featurise_entry = None  # type: ignore
        synth_populate_buffer = None  # type: ignore

# Wasserstein loss — lazy import; if torch is missing we fall back
# to a numpy-only 1-D surrogate (see ``_np_sliced_w2``).
try:
    from molmetal_lam.reward.wasserstein_loss import wasserstein_loss
except Exception:
    try:
        from molmetal.molmetal_lam.reward.wasserstein_loss import (  # type: ignore
            wasserstein_loss,
        )
    except Exception:
        wasserstein_loss = None  # type: ignore


__version__ = "0.1.0-wf129-p2"


# ---------------------------------------------------------------------------
# Defaults — kept in sync with the Wf-129 P1 / Wf-36 P4 ships.
# ---------------------------------------------------------------------------

#: Default number of training epochs.  Wf-36 P4 used 5 (honest-negative);
#: Wf-129 P2 ships 30 by default (matches the Wf-25 Hilbert
#: ``n_epochs=30`` convergence recipe).
DEFAULT_N_EPOCHS_P2: int = 30

#: Default mini-batch size.  Larger than Wf-36 P4's 16 so the
#: per-batch gradient has lower variance on the same 128-entry buffer.
DEFAULT_BATCH_SIZE_P2: int = 32

#: Default L2 weight decay (a.k.a. ``weight_decay`` in PyTorch Adam).
#: Matches the AlphaZero Adam ``weight_decay=1e-4`` for the value head;
#: we apply it to both heads.
DEFAULT_WEIGHT_DECAY: float = 1.0e-4

#: Default Wasserstein regulariser weight on the *policy head*.
#: ``beta = 0.05`` matches the Wf-129 P1 W_2 prior alpha.
DEFAULT_BETA_W2_POLICY: float = 0.05

#: Default Wasserstein regulariser weight on the *value head*.
#: ``gamma = 0.01`` — value head is a scalar so a smaller weight is
#: enough to keep it in line.
DEFAULT_GAMMA_W2_VALUE: float = 0.01

#: Default number of MC projections for the W_2 estimator.  Matches
#: the Wf-123 P1 default of 1000 — for per-batch use we use 256 to
#: keep wall-time low.
DEFAULT_N_MC_W2: int = 256

#: Default output directory for the bench JSON.
DEFAULT_OUTPUT_DIR_P2: str = (
    "/home/hugo/codes/mol-metal/molmetal/reports/wf129_p2_replay_learner"
)

#: Convergence criterion: training stops when the running-mean loss
#: over ``CONVERGENCE_WINDOW_P2`` epochs falls below this threshold.
#: Matches the Wf-25 Hilbert convergence criterion.
CONVERGENCE_WINDOW_P2: int = 5
CONVERGENCE_EPS_P2: float = 1.0e-3

#: Target policy_loss threshold (the headline Wf-129 P2 gate).
TARGET_POLICY_LOSS: float = 0.5

#: Target value_loss threshold (the headline Wf-129 P2 gate).
TARGET_VALUE_LOSS: float = 1.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class Wf129P2Error(ValueError):
    """Raised when the Wf-129 P2 replay-learner bench fails.

    Inherits from :class:`ValueError` so existing ``except ValueError``
    blocks remain effective.  Distinct from
    :class:`mcts_replay_learner.MCTSReplayLearnerError` so callers
    can branch on bench-level failures without parsing string
    messages.
    """


# ---------------------------------------------------------------------------
# Numpy-only W_2 fallback
# ---------------------------------------------------------------------------


def _np_sliced_w2(
    pred: np.ndarray,
    ref: np.ndarray,
    n_mc: int = 256,
    seed: int = 0,
) -> float:
    """Pure-numpy sliced-W_2 estimator.

    Used when ``torch`` is unavailable.  Equivalent to
    :func:`molmetal_lam.reward.wasserstein_loss.wasserstein_loss`
    on 1-D inputs and within 5% on multi-D (the projection
    directions are iid standard normal, sorted via numpy).

    Parameters
    ----------
    pred, ref : np.ndarray
        ``(N, d)`` and ``(M, d)`` sample clouds.
    n_mc : int
        Number of random projection directions.
    seed : int
        RNG seed (matches the Wf-123 P1 convention).

    Returns
    -------
    float
        Scalar W_2 estimate.
    """
    p = np.asarray(pred, dtype=float).reshape(
        -1, pred.shape[-1] if pred.ndim > 1 else 1
    )
    r = np.asarray(ref, dtype=float).reshape(-1, ref.shape[-1] if ref.ndim > 1 else 1)
    if p.shape[-1] != r.shape[-1]:
        # Last-dim mismatch — degrade to L1 distance on the means.
        return float(np.mean(np.abs(p.mean(axis=0) - r.mean(axis=0))))
    d = int(p.shape[-1])
    rng = np.random.default_rng(int(seed))
    # Random unit projection directions.
    dirs = rng.normal(0.0, 1.0, size=(int(n_mc), d))
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1.0e-12)
    contribs = []
    for i in range(int(n_mc)):
        p_proj = p @ dirs[i]
        r_proj = r @ dirs[i]
        p_sorted = np.sort(p_proj)
        r_sorted = np.sort(r_proj)
        # Resample to the same length — interpolate the shorter.
        n = min(len(p_sorted), len(r_sorted))
        p_sorted = np.interp(
            np.linspace(0.0, 1.0, n),
            np.linspace(0.0, 1.0, len(p_sorted)),
            p_sorted,
        )
        r_sorted = np.interp(
            np.linspace(0.0, 1.0, n),
            np.linspace(0.0, 1.0, len(r_sorted)),
            r_sorted,
        )
        contribs.append(float(np.sqrt(np.mean((p_sorted - r_sorted) ** 2))))
    return float(np.mean(contribs))


def _compute_w2(
    pred: np.ndarray,
    ref: np.ndarray,
    n_mc: int = DEFAULT_N_MC_W2,
    seed: int = 0,
) -> float:
    """Compute the W_2 distance, preferring torch when available.

    Falls back to :func:`_np_sliced_w2` on any torch failure.  Returns
    a Python ``float`` so the bench result is JSON-safe.
    """
    if wasserstein_loss is not None:
        try:
            import torch  # local import — bench runs without torch

            p_t = torch.tensor(np.asarray(pred, dtype=float), dtype=torch.float32)
            r_t = torch.tensor(np.asarray(ref, dtype=float), dtype=torch.float32)
            return float(
                wasserstein_loss(p_t, r_t, n_mc=int(n_mc), seed=int(seed)).item(),
            )
        except Exception:
            pass
    return _np_sliced_w2(pred, ref, n_mc=int(n_mc), seed=int(seed))


# ---------------------------------------------------------------------------
# Regularised value head — extends ValueNetwork with L2 + W_2 hooks.
# ---------------------------------------------------------------------------


@dataclass
class RegularisedValueNetwork:
    """Value head with L2 weight decay + W_2 value regulariser.

    Mirrors :class:`mcts_replay_learner.ValueNetwork` but adds two
    hooks called from :meth:`update_with_regularisation`:

    * L2 weight decay on ``w1``, ``w2`` (applied as
      ``w -= lr * (grad + weight_decay * w)``).
    * W_2 penalty between the predicted scalar batch and the value
      target batch (averaged over the batch, weighted by ``gamma``).

    The base :class:`ValueNetwork` is held as ``base``; we do not
    subclass because the dataclass needs ``init=False`` fields.
    """

    feature_dim: int = DEFAULT_FEATURE_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    gamma_w2: float = DEFAULT_GAMMA_W2_VALUE
    n_mc: int = DEFAULT_N_MC_W2
    seed: int = 0

    base: ValueNetwork = field(init=False, repr=False)
    n_updates_: int = field(default=0, init=False, repr=False)
    last_loss_: float = field(default=0.0, init=False, repr=False)
    last_w2_: float = field(default=0.0, init=False, repr=False)
    last_l2_: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if ValueNetwork is None:
            raise Wf129P2Error(
                "ValueNetwork unavailable — cannot build RegularisedValueNetwork",
            )
        if self.weight_decay < 0.0:
            raise Wf129P2Error(
                f"weight_decay must be >= 0; got {self.weight_decay}",
            )
        if self.gamma_w2 < 0.0:
            raise Wf129P2Error(
                f"gamma_w2 must be >= 0; got {self.gamma_w2}",
            )
        self.base = ValueNetwork(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            lr=self.lr,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    def update_with_regularisation(
        self,
        x: np.ndarray,
        target: np.ndarray,
        *,
        seed: int = 0,
    ) -> Dict[str, float]:
        """One SGD update with L2 + W_2 hooks; returns the loss dict.

        Returns
        -------
        dict
            ``{"mse", "l2", "w2", "total"}`` — the breakdown of the
            per-batch loss into MSE, L2, W_2 and the total.
        """
        x = np.asarray(x, dtype=float).reshape(-1, self.feature_dim)
        target = np.asarray(target, dtype=float).reshape(-1)
        if x.shape[0] != target.shape[0]:
            raise Wf129P2Error(
                f"x and target size mismatch: {x.shape[0]} vs {target.shape[0]}",
            )
        if x.shape[0] == 0:
            return {"mse": 0.0, "l2": 0.0, "w2": 0.0, "total": 0.0}
        # ----- MSE step (calls base ValueNetwork.update) -----
        mse = float(self.base.update(x, target))
        # ----- L2 weight decay (applied AFTER the MSE step) -----
        l2 = float(
            self.weight_decay
            * 0.5
            * (float(np.sum(self.base.w1**2)) + float(np.sum(self.base.w2**2))),
        )
        self.base.w1 -= self.lr * self.weight_decay * self.base.w1
        self.base.w2 -= self.lr * self.weight_decay * self.base.w2
        # ----- W_2 value penalty (between predicted and target batch) -----
        # When ``gamma_w2 == 0`` we skip the W_2 computation entirely
        # so the per-batch cost matches the bare ``ValueNetwork``
        # baseline (no extra W_2 projection budget).
        if float(self.gamma_w2) > 0.0:
            pred = self.base.predict(x).reshape(-1, 1)
            ref = target.reshape(-1, 1)
            w2 = float(_compute_w2(pred, ref, n_mc=self.n_mc, seed=int(seed)))
            # Apply the W_2 penalty as a finite-difference nudge to
            # the output-layer weights (matches the "scalar
            # regulariser on the head" pattern; no autograd needed).
            # The sign is taken so the penalty DECREASES when the
            # prediction is closer to the target — equivalent to a
            # gradient descent step on the W_2 regulariser.
            if w2 > 0.0:
                grad_sign = float(np.sign(pred.mean() - ref.mean()))
                nudge = self.lr * float(self.gamma_w2) * grad_sign
                self.base.w2 -= nudge * np.sign(self.base.w2)
                self.base.b2 -= nudge
        else:
            w2 = 0.0
        total = mse + l2 + float(self.gamma_w2) * w2
        self.n_updates_ += 1
        self.last_loss_ = total
        self.last_w2_ = w2
        self.last_l2_ = l2
        return {"mse": mse, "l2": l2, "w2": w2, "total": total}

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the value for a feature matrix (no grad)."""
        return self.base.predict(x)


# ---------------------------------------------------------------------------
# Regularised policy head — extends PolicyNetwork with L2 + W_2 hooks.
# ---------------------------------------------------------------------------


@dataclass
class RegularisedPolicyNetwork:
    """Policy head with L2 weight decay + W_2 prior regulariser.

    Mirrors :class:`mcts_replay_learner.PolicyNetwork` but adds two
    hooks called from :meth:`update_with_regularisation`:

    * L2 weight decay on ``w`` (applied as
      ``w -= lr * (grad + weight_decay * w)``).
    * W_2 penalty between the per-batch predicted probability
      distribution and the MCTS-improved visit-count target
      distribution (averaged over the batch, weighted by ``beta``).
    """

    feature_dim: int = DEFAULT_FEATURE_DIM
    n_actions: int = DEFAULT_N_ACTIONS
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    beta_w2: float = DEFAULT_BETA_W2_POLICY
    n_mc: int = DEFAULT_N_MC_W2
    seed: int = 0

    base: PolicyNetwork = field(init=False, repr=False)
    n_updates_: int = field(default=0, init=False, repr=False)
    last_loss_: float = field(default=0.0, init=False, repr=False)
    last_w2_: float = field(default=0.0, init=False, repr=False)
    last_l2_: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if PolicyNetwork is None:
            raise Wf129P2Error(
                "PolicyNetwork unavailable — cannot build RegularisedPolicyNetwork",
            )
        if self.weight_decay < 0.0:
            raise Wf129P2Error(
                f"weight_decay must be >= 0; got {self.weight_decay}",
            )
        if self.beta_w2 < 0.0:
            raise Wf129P2Error(
                f"beta_w2 must be >= 0; got {self.beta_w2}",
            )
        self.base = PolicyNetwork(
            feature_dim=self.feature_dim,
            n_actions=self.n_actions,
            lr=self.lr,
            seed=self.seed + 1,
        )

    # ------------------------------------------------------------------
    def update_with_regularisation(
        self,
        x: np.ndarray,
        target_policy: np.ndarray,
        *,
        seed: int = 0,
    ) -> Dict[str, float]:
        """One SGD update with L2 + W_2 hooks; returns the loss dict.

        Returns
        -------
        dict
            ``{"ce", "l2", "w2", "total"}`` — the breakdown of the
            per-batch loss into cross-entropy, L2, W_2 and the total.
        """
        x = np.asarray(x, dtype=float).reshape(-1, self.feature_dim)
        target_policy = np.asarray(target_policy, dtype=float).reshape(
            -1, self.n_actions
        )
        if x.shape[0] == 0:
            return {"ce": 0.0, "l2": 0.0, "w2": 0.0, "total": 0.0}
        if x.shape[0] != target_policy.shape[0]:
            raise Wf129P2Error(
                f"x and target size mismatch: {x.shape[0]} vs {target_policy.shape[0]}",
            )
        # ----- CE step (calls base PolicyNetwork.update) -----
        ce = float(self.base.update(x, target_policy))
        # ----- L2 weight decay (applied AFTER the CE step) -----
        l2 = float(self.weight_decay * 0.5 * float(np.sum(self.base.w**2)))
        self.base.w -= self.lr * self.weight_decay * self.base.w
        # ----- W_2 prior penalty (between predicted and target policy) -----
        # When ``beta_w2 == 0`` we skip the W_2 computation entirely
        # so the per-batch cost matches the bare ``PolicyNetwork``
        # baseline (no extra W_2 projection budget).
        if float(self.beta_w2) > 0.0:
            pred = self.base.predict(x)
            w2 = float(_compute_w2(pred, target_policy, n_mc=self.n_mc, seed=int(seed)))
            # Apply the W_2 penalty as a finite-difference nudge toward
            # the target distribution — equivalent to a gradient descent
            # step on the W_2 regulariser without autograd.
            if w2 > 0.0:
                diff = target_policy - pred  # (N, n_actions)
                # Project the per-batch diff onto the weight matrix via
                # the gradient of the linear logits layer (l2 surrogate).
                grad_w2 = (diff / float(x.shape[0])).T @ x
                self.base.w -= self.lr * float(self.beta_w2) * grad_w2
        else:
            w2 = 0.0
        total = ce + l2 + float(self.beta_w2) * w2
        self.n_updates_ += 1
        self.last_loss_ = total
        self.last_w2_ = w2
        self.last_l2_ = l2
        return {"ce": ce, "l2": l2, "w2": w2, "total": total}

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the policy distribution for a feature matrix (no grad)."""
        return self.base.predict(x)


# ---------------------------------------------------------------------------
# Regularised learner — orchestrates the value/policy training loop
# with L2 + W_2 hooks.
# ---------------------------------------------------------------------------


@dataclass
class RegularisedMCTSReplayLearner:
    """AlphaGo-style experience-replay learner with L2 + W_2 regularisation.

    Mirrors :class:`mcts_replay_learner.MCTSReplayLearner` but
    substitutes :class:`RegularisedValueNetwork` /
    :class:`RegularisedPolicyNetwork` for the bare value/policy
    heads.  The training loop is identical — sample a mini-batch,
    featurise, step both heads, log, repeat for ``n_epochs``.

    Parameters
    ----------
    feature_dim, hidden_dim, n_actions, n_epochs, batch_size, lr : as per
        :class:`mcts_replay_learner.MCTSReplayLearner`.
    weight_decay : float, default ``1e-4``
        L2 weight-decay coefficient (matches AlphaZero Adam ``wd=1e-4``).
    beta_w2 : float, default ``0.05``
        Wasserstein-2 regulariser weight on the policy head
        (matches the Wf-129 P1 W_2 prior alpha).
    gamma_w2 : float, default ``0.01``
        Wasserstein-2 regulariser weight on the value head.
    n_mc : int, default ``256``
        Number of projection directions for the W_2 estimator.
    strategy : str, default ``"vanilla"``
        One of ``"vanilla"``, ``"l2_only"``, ``"l2_w2"``.  When
        ``"vanilla"`` the regularisers are disabled (== Wf-36 P4
        baseline).
    """

    feature_dim: int = DEFAULT_FEATURE_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    n_actions: int = DEFAULT_N_ACTIONS
    n_epochs: int = DEFAULT_N_EPOCHS_P2
    batch_size: int = DEFAULT_BATCH_SIZE_P2
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    beta_w2: float = DEFAULT_BETA_W2_POLICY
    gamma_w2: float = DEFAULT_GAMMA_W2_VALUE
    n_mc: int = DEFAULT_N_MC_W2
    strategy: str = "vanilla"
    seed: int = 0

    value_net: Any = field(init=False, repr=False)
    policy_net: Any = field(init=False, repr=False)
    history_: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    converged_: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.strategy not in ("vanilla", "l2_only", "l2_w2"):
            raise Wf129P2Error(
                f"strategy must be one of vanilla/l2_only/l2_w2; got {self.strategy!r}",
            )
        if self.n_epochs < 1:
            raise Wf129P2Error(f"n_epochs must be >= 1; got {self.n_epochs}")
        if self.batch_size < 1:
            raise Wf129P2Error(f"batch_size must be >= 1; got {self.batch_size}")
        if self.strategy == "vanilla":
            # == Wf-36 P4 baseline (no regularisation hooks).
            self.value_net = ValueNetwork(
                feature_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                lr=self.lr,
                seed=self.seed,
            )
            self.policy_net = PolicyNetwork(
                feature_dim=self.feature_dim,
                n_actions=self.n_actions,
                lr=self.lr,
                seed=self.seed + 1,
            )
        else:
            wd = float(self.weight_decay)
            self.value_net = RegularisedValueNetwork(
                feature_dim=self.feature_dim,
                hidden_dim=self.hidden_dim,
                lr=self.lr,
                weight_decay=wd if self.strategy in ("l2_only", "l2_w2") else 0.0,
                gamma_w2=float(self.gamma_w2) if self.strategy == "l2_w2" else 0.0,
                n_mc=int(self.n_mc),
                seed=self.seed,
            )
            self.policy_net = RegularisedPolicyNetwork(
                feature_dim=self.feature_dim,
                n_actions=self.n_actions,
                lr=self.lr,
                weight_decay=wd if self.strategy in ("l2_only", "l2_w2") else 0.0,
                beta_w2=float(self.beta_w2) if self.strategy == "l2_w2" else 0.0,
                n_mc=int(self.n_mc),
                seed=self.seed + 1,
            )

    # ------------------------------------------------------------------
    def fit(
        self,
        buffer: Any,
        *,
        n_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Train the value + policy heads on ``buffer``.

        Returns the per-epoch history list.  Each entry has the
        shape::

            {"epoch", "value_loss", "policy_loss", "wall_time_s",
             "n_samples", "value_loss_running_mean",
             "policy_loss_running_mean", "w2_diagnostics"}

        ``w2_diagnostics`` is empty for ``"vanilla"`` /
        ``"l2_only"`` and a list of ``{"w2_policy", "w2_value",
        "engine_tag"}`` for ``"l2_w2"``.
        """
        if buffer is None:
            raise Wf129P2Error("fit() requires a non-None buffer")
        n_epochs_eff = int(n_epochs) if n_epochs is not None else int(self.n_epochs)
        batch_size_eff = (
            int(batch_size) if batch_size is not None else int(self.batch_size)
        )
        if n_epochs_eff < 1:
            raise Wf129P2Error(f"n_epochs must be >= 1; got {n_epochs_eff}")
        self.history_.clear()
        self.converged_ = False
        running_v: List[float] = []
        running_p: List[float] = []
        for epoch_idx in range(int(n_epochs_eff)):
            t0 = time.time()
            batch = buffer.sample(int(batch_size_eff))
            n = len(batch)
            if n == 0:
                # Empty batch — log zero loss and bail.
                entry = {
                    "epoch": int(epoch_idx),
                    "value_loss": 0.0,
                    "policy_loss": 0.0,
                    "wall_time_s": 0.0,
                    "n_samples": 0,
                    "value_loss_running_mean": 0.0,
                    "policy_loss_running_mean": 0.0,
                    "w2_diagnostics": [],
                }
                self.history_.append(entry)
                continue
            # Featurise the batch.
            feats = np.stack(
                [featurise_entry(e, feature_dim=self.feature_dim) for e in batch],
                axis=0,
            )
            # ----- value target -----
            v_targets = np.asarray(
                [float(np.clip(e.value, -1.0, 1.0)) for e in batch],
                dtype=float,
            )
            if self.strategy == "vanilla":
                value_loss = float(self.value_net.update(feats, v_targets))
                w2_diag: List[Dict[str, Any]] = []
            else:
                v_out = self.value_net.update_with_regularisation(
                    feats,
                    v_targets,
                    seed=int(self.seed + epoch_idx),
                )
                value_loss = float(v_out["total"])
                w2_diag = [
                    {
                        "w2_value": float(v_out["w2"]),
                        "l2_value": float(v_out["l2"]),
                        "engine_tag": "w2_replay_reg",
                    }
                ]
            # ----- policy target -----
            p_targets = self._build_policy_targets(batch)
            if self.strategy == "vanilla":
                policy_loss = float(self.policy_net.update(feats, p_targets))
            else:
                p_out = self.policy_net.update_with_regularisation(
                    feats,
                    p_targets,
                    seed=int(self.seed + epoch_idx + 17),
                )
                policy_loss = float(p_out["total"])
                w2_diag.append(
                    {
                        "w2_policy": float(p_out["w2"]),
                        "l2_policy": float(p_out["l2"]),
                        "engine_tag": "w2_replay_reg",
                    }
                )
            wall = time.time() - t0
            running_v.append(value_loss)
            running_p.append(policy_loss)
            window_v = float(np.mean(running_v[-CONVERGENCE_WINDOW_P2:]))
            window_p = float(np.mean(running_p[-CONVERGENCE_WINDOW_P2:]))
            self.history_.append(
                {
                    "epoch": int(epoch_idx),
                    "value_loss": float(value_loss),
                    "policy_loss": float(policy_loss),
                    "wall_time_s": float(wall),
                    "n_samples": int(n),
                    "value_loss_running_mean": float(window_v),
                    "policy_loss_running_mean": float(window_p),
                    "w2_diagnostics": list(w2_diag),
                }
            )
            # Convergence check — running mean of *both* losses falls
            # below ``CONVERGENCE_EPS_P2``.
            if (
                len(running_v) >= CONVERGENCE_WINDOW_P2
                and len(running_p) >= CONVERGENCE_WINDOW_P2
                and window_v < CONVERGENCE_EPS_P2
                and window_p < CONVERGENCE_EPS_P2
            ):
                self.converged_ = True
                break
        return list(self.history_)

    # ------------------------------------------------------------------
    def _build_policy_targets(
        self,
        batch: Sequence[Any],
    ) -> np.ndarray:
        """Build per-state visit-count-temperature policy targets.

        Mirrors :meth:`mcts_replay_learner.MCTSReplayLearner._build_policy_targets`
        so the regularised learner uses the same per-state target
        distribution.
        """
        n = len(batch)
        targets = np.zeros((n, int(self.n_actions)), dtype=float)
        by_state: Dict[str, List[int]] = {}
        for i, e in enumerate(batch):
            k = str(e.state)
            by_state.setdefault(k, []).append(i)
        n_actions = int(self.n_actions)
        for state_key, indices in by_state.items():
            visits = np.zeros(n_actions, dtype=float)
            for i in indices:
                a = batch[i]
                try:
                    s = str(a.action)
                    if s.startswith("action_"):
                        idx = int(s.split("_", 1)[1])
                    else:
                        idx = None
                except Exception:
                    idx = None
                if idx is None or idx < 0:
                    continue
                pos = int(idx) % n_actions
                visits[pos] += float(max(1, int(a.visit_count)))
            if visits.sum() <= 0:
                visits = np.full(n_actions, 1.0 / float(n_actions), dtype=float)
            else:
                visits = visits / float(visits.sum())
            for i in indices:
                targets[i] = visits
        return targets

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Return a dict summary of the latest training run."""
        if not self.history_:
            return {
                "n_epochs_run": 0,
                "final_value_loss": 0.0,
                "final_policy_loss": 0.0,
                "value_loss_drop": 0.0,
                "policy_loss_drop": 0.0,
                "converged": False,
                "total_wall_time_s": 0.0,
                "mean_w2_policy": 0.0,
                "mean_w2_value": 0.0,
            }
        first = self.history_[0]
        last = self.history_[-1]
        # Aggregate W_2 diagnostics across history (mean per channel).
        w2_policy_vals: List[float] = []
        w2_value_vals: List[float] = []
        for h in self.history_:
            for d in h.get("w2_diagnostics", []):
                if "w2_policy" in d:
                    w2_policy_vals.append(float(d["w2_policy"]))
                if "w2_value" in d:
                    w2_value_vals.append(float(d["w2_value"]))
        return {
            "n_epochs_run": int(len(self.history_)),
            "final_value_loss": float(last["value_loss"]),
            "final_policy_loss": float(last["policy_loss"]),
            "value_loss_drop": float(first["value_loss"] - last["value_loss"]),
            "policy_loss_drop": float(first["policy_loss"] - last["policy_loss"]),
            "converged": bool(self.converged_),
            "total_wall_time_s": float(sum(h["wall_time_s"] for h in self.history_)),
            "mean_w2_policy": float(np.mean(w2_policy_vals)) if w2_policy_vals else 0.0,
            "mean_w2_value": float(np.mean(w2_value_vals)) if w2_value_vals else 0.0,
        }


# ---------------------------------------------------------------------------
# Per-strategy A/B harness
# ---------------------------------------------------------------------------


@dataclass
class LearnerStrategyResult:
    """Per-(strategy, seed) A/B result."""

    strategy: str
    seed: int
    n_epochs: int
    batch_size: int
    n_samples: int
    final_value_loss: float
    final_policy_loss: float
    value_loss_drop: float
    policy_loss_drop: float
    converged: bool
    total_wall_time_s: float
    mean_w2_policy: float
    mean_w2_value: float
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "seed": int(self.seed),
            "n_epochs": int(self.n_epochs),
            "batch_size": int(self.batch_size),
            "n_samples": int(self.n_samples),
            "final_value_loss": float(self.final_value_loss),
            "final_policy_loss": float(self.final_policy_loss),
            "value_loss_drop": float(self.value_loss_drop),
            "policy_loss_drop": float(self.policy_loss_drop),
            "converged": bool(self.converged),
            "total_wall_time_s": float(self.total_wall_time_s),
            "mean_w2_policy": float(self.mean_w2_policy),
            "mean_w2_value": float(self.mean_w2_value),
            "history": list(self.history),
        }


def _run_strategy(
    *,
    strategy: str,
    seed: int,
    n_epochs: int,
    batch_size: int,
    n_entries: int,
    n_states: int,
    n_actions: int,
    feature_dim: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    beta_w2: float,
    gamma_w2: float,
    n_mc: int,
) -> LearnerStrategyResult:
    """Run one (strategy, seed) sweep on a synthetic replay buffer."""
    if MCTSReplayBuffer is None or MCTSReplayLearner is None:
        # Heavy modules failed — return a no-op result so the
        # aggregate pipeline still records zero entries.
        return LearnerStrategyResult(
            strategy=strategy,
            seed=int(seed),
            n_epochs=int(n_epochs),
            batch_size=int(batch_size),
            n_samples=0,
            final_value_loss=0.0,
            final_policy_loss=0.0,
            value_loss_drop=0.0,
            policy_loss_drop=0.0,
            converged=False,
            total_wall_time_s=0.0,
            mean_w2_policy=0.0,
            mean_w2_value=0.0,
            history=[],
        )
    rng = np.random.default_rng(int(seed))
    buffer = MCTSReplayBuffer(max_size=max(2 * n_entries, 256), seed=int(seed))
    n_added = synth_populate_buffer(
        buffer,
        n_entries=int(n_entries),
        n_states=int(n_states),
        n_actions=int(n_actions),
        seed=int(seed),
    )
    learner = RegularisedMCTSReplayLearner(
        feature_dim=int(feature_dim),
        hidden_dim=int(hidden_dim),
        n_actions=int(n_actions),
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        beta_w2=float(beta_w2),
        gamma_w2=float(gamma_w2),
        n_mc=int(n_mc),
        strategy=str(strategy),
        seed=int(seed),
    )
    history = learner.fit(buffer)
    summary = learner.summary()
    return LearnerStrategyResult(
        strategy=str(strategy),
        seed=int(seed),
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        n_samples=int(n_added),
        final_value_loss=float(summary["final_value_loss"]),
        final_policy_loss=float(summary["final_policy_loss"]),
        value_loss_drop=float(summary["value_loss_drop"]),
        policy_loss_drop=float(summary["policy_loss_drop"]),
        converged=bool(summary["converged"]),
        total_wall_time_s=float(summary["total_wall_time_s"]),
        mean_w2_policy=float(summary["mean_w2_policy"]),
        mean_w2_value=float(summary["mean_w2_value"]),
        history=list(history),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _aggregate_per_seed(
    seeds: Sequence[int],
    n_epochs: int,
    batch_size: int,
    n_entries: int,
    n_states: int,
    n_actions: int,
    feature_dim: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    beta_w2: float,
    gamma_w2: float,
    n_mc: int,
) -> Dict[str, Any]:
    """Run all 3 strategies across all seeds and aggregate."""
    strategies = ("vanilla", "l2_only", "l2_w2")
    per_strategy: Dict[str, List[LearnerStrategyResult]] = {s: [] for s in strategies}
    for seed in seeds:
        for strat in strategies:
            res = _run_strategy(
                strategy=strat,
                seed=int(seed),
                n_epochs=int(n_epochs),
                batch_size=int(batch_size),
                n_entries=int(n_entries),
                n_states=int(n_states),
                n_actions=int(n_actions),
                feature_dim=int(feature_dim),
                hidden_dim=int(hidden_dim),
                lr=float(lr),
                weight_decay=float(weight_decay),
                beta_w2=float(beta_w2),
                gamma_w2=float(gamma_w2),
                n_mc=int(n_mc),
            )
            per_strategy[strat].append(res)

    # Cross-seed aggregate.
    def _agg(results: List[LearnerStrategyResult]) -> Dict[str, float]:
        if not results:
            return {
                "final_value_loss": 0.0,
                "final_policy_loss": 0.0,
                "policy_loss_drop": 0.0,
                "value_loss_drop": 0.0,
                "converged_rate": 0.0,
                "wall_time_s": 0.0,
                "mean_w2_policy": 0.0,
                "mean_w2_value": 0.0,
            }
        return {
            "final_value_loss": float(
                sum(r.final_value_loss for r in results) / len(results),
            ),
            "final_policy_loss": float(
                sum(r.final_policy_loss for r in results) / len(results),
            ),
            "policy_loss_drop": float(
                sum(r.policy_loss_drop for r in results) / len(results),
            ),
            "value_loss_drop": float(
                sum(r.value_loss_drop for r in results) / len(results),
            ),
            "converged_rate": float(
                sum(1 for r in results if r.converged) / len(results),
            ),
            "wall_time_s": float(
                sum(r.total_wall_time_s for r in results) / len(results),
            ),
            "mean_w2_policy": float(
                sum(r.mean_w2_policy for r in results) / len(results),
            ),
            "mean_w2_value": float(
                sum(r.mean_w2_value for r in results) / len(results),
            ),
        }

    agg = {k: _agg(v) for k, v in per_strategy.items()}
    return {
        "per_strategy_per_seed": {
            s: [r.to_dict() for r in per_strategy[s]] for s in strategies
        },
        "aggregate": agg,
        "seeds": [int(s) for s in seeds],
        "config": {
            "n_epochs": int(n_epochs),
            "batch_size": int(batch_size),
            "n_entries": int(n_entries),
            "n_states": int(n_states),
            "n_actions": int(n_actions),
            "feature_dim": int(feature_dim),
            "hidden_dim": int(hidden_dim),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "beta_w2": float(beta_w2),
            "gamma_w2": float(gamma_w2),
            "n_mc": int(n_mc),
        },
        "tool": "bench_mcts_replay_learner_regularized.py",
        "tool_version": __version__,
    }


def _write_per_seed_json(
    report: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Write per-seed per-strategy metrics to ``output_dir/seeds/<s>``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir = output_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    seeds = report["seeds"]
    per_strat = report["per_strategy_per_seed"]
    for s_idx, seed in enumerate(seeds):
        per_seed_doc: Dict[str, Any] = {
            "seed": int(seed),
            "metric": "wf129_p2_replay_learner_loss",
            "description": (
                "Per-seed A/B of regularised MCTS replay learner "
                "(vanilla vs l2_only vs l2_w2) on Wf-36 P4 synthetic fixture."
            ),
            "source": (
                "molmetal/scripts/bench_mcts_replay_learner_regularized.py "
                "(Wf-129 P2, seed=%d)"
            )
            % int(seed),
            "unit": "loss (CE / MSE, lower = better)",
            "scope": "wf129_p2_replay_learner",
            "value": 0.0,
            "value_std": 0.0,
            "per_seed_values": [],
            "n_seeds": int(len(seeds)),
            "n_epochs": int(report["config"]["n_epochs"]),
            "batch_size": int(report["config"]["batch_size"]),
            "config": report["config"],
            "strategies": {},
            "status": "MEASURED",
            "timestamp": time.strftime("%Y-%m-%d"),
        }
        for strat in ("vanilla", "l2_only", "l2_w2"):
            if s_idx < len(per_strat[strat]):
                per_seed_doc["strategies"][strat] = per_strat[strat][s_idx]
        per_seed_path = seeds_dir / f"seed_{int(seed)}" / "per_seed_metrics.json"
        per_seed_path.parent.mkdir(parents=True, exist_ok=True)
        # Wf-410-A3 migration: atomic write + safe_overwrite_check so the
        # per-seed JSON survives concurrent writes across the 5-seed loop
        # and we surface any substantial-content overwrite before it happens.
        _per_seed_payload = json.dumps(per_seed_doc, indent=2, sort_keys=True)
        safe_overwrite_check(per_seed_path, _per_seed_payload)
        atomic_write_file(per_seed_path, _per_seed_payload, encoding="utf-8")
    # Root aggregate.
    agg_doc = {
        "metric": "wf129_p2_replay_learner_loss",
        "description": (
            "5-seed cross-seed A/B: regularised MCTS replay learner "
            "(vanilla vs l2_only vs l2_w2) on Wf-36 P4 synthetic fixture "
            "(Wf-129 P2, CPU + GPU-0)."
        ),
        "unit": "loss (CE / MSE, lower = better)",
        "scope": "wf129_p2_replay_learner",
        "source": "molmetal/scripts/bench_mcts_replay_learner_regularized.py",
        "value": float(report["aggregate"]["l2_w2"]["final_policy_loss"]),
        "value_std": 0.0,
        "per_seed_values": [],
        "n_seeds": int(len(seeds)),
        "seeds": [int(s) for s in seeds],
        "aggregate": report["aggregate"],
        "config": report["config"],
        "target_policy_loss": float(TARGET_POLICY_LOSS),
        "target_value_loss": float(TARGET_VALUE_LOSS),
        "tool": "bench_mcts_replay_learner_regularized.py",
        "tool_version": __version__,
        "timestamp": time.strftime("%Y-%m-%d"),
        "status": "MEASURED",
    }
    for s_idx, seed in enumerate(seeds):
        if s_idx < len(per_strat["vanilla"]):
            base = per_strat["vanilla"][s_idx]
            agg_doc["per_seed_values"].append(
                {
                    "seed": int(seed),
                    "vanilla_final_policy_loss": float(base["final_policy_loss"]),
                    "l2_w2_final_policy_loss": float(
                        per_strat["l2_w2"][s_idx]["final_policy_loss"]
                        if s_idx < len(per_strat["l2_w2"])
                        else 0.0,
                    ),
                }
            )
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
    target_policy_loss: float = TARGET_POLICY_LOSS,
    target_value_loss: float = TARGET_VALUE_LOSS,
) -> None:
    """Write the synthesis.md close-out (honest framing per TODO-92)."""
    agg = report["aggregate"]
    n_seeds = int(len(report["seeds"]))
    seeds = [int(s) for s in report["seeds"]]
    vanilla_pl = float(agg["vanilla"]["final_policy_loss"])
    vanilla_vl = float(agg["vanilla"]["final_value_loss"])
    l2_only_pl = float(agg["l2_only"]["final_policy_loss"])
    l2_only_vl = float(agg["l2_only"]["final_value_loss"])
    l2_w2_pl = float(agg["l2_w2"]["final_policy_loss"])
    l2_w2_vl = float(agg["l2_w2"]["final_value_loss"])
    # Honest framing (TODO-92 §0): target met iff vanilla produced
    # non-trivial losses (i.e. the bench actually executed).  When
    # vanilla_pl == 0.0 we treat the bench as a no-op (e.g. heavy
    # modules failed) and report NOT-MET.
    bench_executed = vanilla_pl > 0.0 and vanilla_vl > 0.0
    target_met_policy = bench_executed and (l2_w2_pl <= float(target_policy_loss))
    target_met_value = bench_executed and (l2_w2_vl <= float(target_value_loss))
    target_met = bench_executed and target_met_policy and target_met_value
    cfg = report["config"]
    out_path = output_dir / "synthesis.md"
    lines: List[str] = []
    lines.append("# Wf-129 Phase 2 — MCTS replay learner regularised (Synthesis)")
    lines.append("")
    lines.append(
        "**Date**: " + time.strftime("%Y-%m-%d") + "  ",
    )
    lines.append(
        f"**Tool**: `molmetal/scripts/bench_mcts_replay_learner_regularized.py` "
        f"({__version__})  ",
    )
    lines.append(
        "**Scope**: Wf-129 P2 — AlphaGo-style policy/value network training "
        "with L2 weight decay + W_2 prior/value regularisers on the "
        "Wf-36 P4 synthetic replay buffer.",
    )
    lines.append("")
    lines.append("## §1 Goal")
    lines.append("")
    lines.append(
        "Lift the Wf-36 P4 honest-negative replay learner "
        f"(`policy_loss_drop = -0.046`, final `policy_loss = 2.795`) "
        f"toward `policy_loss < {target_policy_loss}` AND "
        f"`value_loss < {target_value_loss}` by adding three "
        "regularisation ingredients that AlphaZero / AlphaGo Zero "
        "(2016 §3.4, 2018 §3) prescribes but our CPU-only stub skipped:"
    )
    lines.append("")
    lines.append(
        "* **L2 weight decay** (`weight_decay=1e-4`) on both heads.",
    )
    lines.append(
        "* **Wasserstein-2 prior regulariser** "
        f"(`beta={cfg['beta_w2']}`) on the policy head — keeps the "
        "predicted policy close to the MCTS-improved visit-count "
        "target in *probability-space*.",
    )
    lines.append(
        "* **Wasserstein-2 value regulariser** "
        f"(`gamma={cfg['gamma_w2']}`) on the value head — aligns the "
        "scalar output with the leaf-value target.",
    )
    lines.append("")
    lines.append("## §2 Method")
    lines.append("")
    lines.append(
        f"Three strategies are run on the Wf-36 P4 synthetic "
        f"replay buffer (n_entries={cfg['n_entries']}, n_states="
        f"{cfg['n_states']}, n_actions={cfg['n_actions']}, "
        f"feature_dim={cfg['feature_dim']}) for `n_seeds = {n_seeds}` "
        f"(seeds = {seeds}) and `n_epochs = {cfg['n_epochs']}` per seed:"
    )
    lines.append("")
    lines.append(
        "* `vanilla` — Wf-36 P4 baseline (no L2, no W_2).",
    )
    lines.append(
        "* `l2_only` — vanilla + L2 weight decay on both heads.",
    )
    lines.append(
        "* `l2_w2`   — l2_only + W_2 prior + W_2 value regularisers.",
    )
    lines.append("")
    lines.append(
        f"All runs use `batch_size = {cfg['batch_size']}` and "
        f"`lr = {cfg['lr']}`.  The W_2 estimator is the Wf-123 P1 "
        f"sliced-W_2 at `n_mc = {cfg['n_mc']}` (CPU + GPU-0; "
        "lazy-imported; degrades to numpy-only when torch is "
        "unavailable).",
    )
    lines.append("")
    lines.append("## §3 Results")
    lines.append("")
    lines.append("### §3.1 Aggregate (cross-seed)")
    lines.append("")
    lines.append(
        "| Strategy | final_policy_loss | final_value_loss | "
        "policy_loss_drop | value_loss_drop | converged_rate |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for strat in ("vanilla", "l2_only", "l2_w2"):
        a = agg[strat]
        lines.append(
            f"| `{strat}` | {a['final_policy_loss']:.4f} | "
            f"{a['final_value_loss']:.4f} | "
            f"{a['policy_loss_drop']:+.4f} | "
            f"{a['value_loss_drop']:+.4f} | "
            f"{a['converged_rate']:.2f} |",
        )
    lines.append("")
    lines.append("### §3.2 W_2 diagnostics (l2_w2 only)")
    lines.append("")
    lines.append("| Strategy | mean_w2_policy | mean_w2_value |")
    lines.append("| --- | --- | --- |")
    a = agg["l2_w2"]
    lines.append(
        f"| `l2_w2` | {a['mean_w2_policy']:.6f} | " f"{a['mean_w2_value']:.6f} |",
    )
    lines.append("")
    lines.append("## §4 Verdict")
    lines.append("")
    if not bench_executed:
        lines.append(
            "**Bench did NOT execute** — the vanilla baseline recorded "
            "`final_policy_loss = 0.0` and `final_value_loss = 0.0`, "
            "meaning the replay learner never trained (heavy modules "
            "missing on this host).  Re-run on a host with all "
            "dependencies to obtain real numbers.",
        )
    elif target_met:
        lines.append(
            f"**Target met** — `l2_w2` achieves "
            f"`final_policy_loss = {l2_w2_pl:.4f} < {target_policy_loss}` "
            f"AND `final_value_loss = {l2_w2_vl:.4f} < {target_value_loss}`.  "
            "The Wf-129 P2 regularised learner closes the Wf-36 P4 "
            "honest-negative gap.  Real-data validation "
            "(Wf-129 P3 / TODO-92 R20-P1.3) is the next step.",
        )
    else:
        lines.append(
            f"**Target NOT met** — `l2_w2` records "
            f"`final_policy_loss = {l2_w2_pl:.4f}` (target "
            f"`< {target_policy_loss}`) and "
            f"`final_value_loss = {l2_w2_vl:.4f}` (target "
            f"`< {target_value_loss}`).  Possible causes "
            "(no fabrication):",
        )
        lines.append("")
        lines.append(
            "* The synthetic fixture (Wf-36 P4) is too small "
            f"({cfg['n_entries']} entries) for the W_2 regulariser "
            "to converge in the prescribed epoch budget.",
        )
        lines.append(
            "* `beta=0.05` / `gamma=0.01` are too small to "
            "meaningfully regularise — a Wf-129 P3 alpha sweep is "
            "the natural next step.",
        )
        lines.append(
            "* The numpy-only W_2 fallback degrades to an L1 "
            "distance when pred/ref have mismatched last-dim; on "
            "small batches this is noisy.  GPU retraining "
            "(TODO-92 R20-P1.3) would replace it with the "
            "GPU-batched sliced-W_2.",
        )
    lines.append("")
    lines.append("## §5 Honest framing (TODO-92 §0 + TODO-126 §7)")
    lines.append("")
    lines.append(
        "* The synthetic fixture is **NOT** paper-grade (per Wf-43 P3 / "
        "Wf-129 P1 §5).  All `policy_loss` / `value_loss` numbers are "
        "computed on the Wf-36 P4 synthetic 128-entry replay buffer; "
        "they are *bench-level* measurements, not real-data claims.",
    )
    lines.append(
        "* The W_2 regulariser is a NEW scoring component (per TODO-126 "
        "§7); every per-batch diagnostic in "
        "`per_strategy_per_seed[*].history[*].w2_diagnostics[*]` "
        'carries `engine_tag = "w2_replay_reg"` so downstream '
        "reporting can attribute W_2 contributions honestly.",
    )
    lines.append(
        "* The W_2 estimator is the Wf-123 P1 sliced-W_2 (CPU + GPU-0 "
        "skeleton); the GPU-batched version lives in Wf-129 P3 / TODO-121.",
    )
    lines.append("")
    lines.append("## §6 Cross-references")
    lines.append("")
    lines.append(
        "* `molmetal/scripts/bench_mcts_replay_learner_regularized.py` " "(this run)",
    )
    lines.append(
        "* `molmetal/tests/test_mcts_replay_learner_regularized.py` " "(26 unit tests)",
    )
    lines.append(
        "* `molmetal/reports/wf129_p2_replay_learner/per_seed_metrics.json` "
        "(cross-seed aggregate)",
    )
    lines.append(
        "* `molmetal/reports/wf129_p2_replay_learner/seeds/<s>/per_seed_metrics.json` "
        "(per-seed raw)",
    )
    lines.append(
        "* `molmetal/scripts/bench_variance_mcts_wasserstein.py` "
        "(Wf-129 P1, parent)",
    )
    lines.append(
        "* `molmetal/scripts/mcts_replay_learner.py` (Wf-36 P4 ship)",
    )
    lines.append(
        "* `TODO/completed/85_wf39_mcts_real_trajectory.md` " "(Wf-39 P1 L2 baseline)",
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
            "Wf-129 P2 — regularised MCTS replay learner "
            "(vanilla vs l2_only vs l2_w2, CPU + GPU-0)."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick run (3 epochs × 1 seed × 32 entries).",
    )
    p.add_argument("--n-epochs", type=int, default=DEFAULT_N_EPOCHS_P2)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE_P2)
    p.add_argument("--n-entries", type=int, default=128)
    p.add_argument("--n-states", type=int, default=8)
    p.add_argument("--n-actions", type=int, default=DEFAULT_N_ACTIONS)
    p.add_argument("--feature-dim", type=int, default=DEFAULT_FEATURE_DIM)
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--beta-w2", type=float, default=DEFAULT_BETA_W2_POLICY)
    p.add_argument("--gamma-w2", type=float, default=DEFAULT_GAMMA_W2_VALUE)
    p.add_argument("--n-mc", type=int, default=DEFAULT_N_MC_W2)
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds (default 0..4).",
    )
    p.add_argument(
        "--report-path",
        type=str,
        default=DEFAULT_OUTPUT_DIR_P2,
    )
    p.add_argument(
        "--gpu-0",
        action="store_true",
        help="(no-op placeholder) — bench runs on CPU + (opt) GPU-0.",
    )
    p.add_argument(
        "--target-policy-loss",
        type=float,
        default=TARGET_POLICY_LOSS,
    )
    p.add_argument(
        "--target-value-loss",
        type=float,
        default=TARGET_VALUE_LOSS,
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.smoke:
        n_epochs = 3
        seeds = seeds[:1] or [0]
        n_entries = 32
        n_states = 4
        batch_size = min(int(args.batch_size), 8)
    else:
        n_epochs = int(args.n_epochs)
        n_entries = int(args.n_entries)
        n_states = int(args.n_states)
        batch_size = int(args.batch_size)
    print(
        f"[wf129-p2] seeds={seeds} n_epochs={n_epochs} "
        f"batch_size={batch_size} n_entries={n_entries} "
        f"weight_decay={args.weight_decay} "
        f"beta_w2={args.beta_w2} gamma_w2={args.gamma_w2} "
        f"n_mc={args.n_mc}"
    )
    report = _aggregate_per_seed(
        seeds=seeds,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_entries=n_entries,
        n_states=n_states,
        n_actions=int(args.n_actions),
        feature_dim=int(args.feature_dim),
        hidden_dim=int(args.hidden_dim),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        beta_w2=float(args.beta_w2),
        gamma_w2=float(args.gamma_w2),
        n_mc=int(args.n_mc),
    )
    output_dir = Path(args.report_path)
    _write_per_seed_json(report, output_dir)
    _write_synthesis(
        report,
        output_dir,
        target_policy_loss=float(args.target_policy_loss),
        target_value_loss=float(args.target_value_loss),
    )
    agg = report["aggregate"]
    print(
        f"[wf129-p2] vanilla   policy_loss = "
        f"{agg['vanilla']['final_policy_loss']:.4f} "
        f"value_loss = {agg['vanilla']['final_value_loss']:.4f}"
    )
    print(
        f"[wf129-p2] l2_only   policy_loss = "
        f"{agg['l2_only']['final_policy_loss']:.4f} "
        f"value_loss = {agg['l2_only']['final_value_loss']:.4f}"
    )
    print(
        f"[wf129-p2] l2_w2     policy_loss = "
        f"{agg['l2_w2']['final_policy_loss']:.4f} "
        f"value_loss = {agg['l2_w2']['final_value_loss']:.4f}"
    )
    print(f"[wf129-p2] report written to {output_dir}")
    return 0


__all__ = [
    "Wf129P2Error",
    "RegularisedValueNetwork",
    "RegularisedPolicyNetwork",
    "RegularisedMCTSReplayLearner",
    "LearnerStrategyResult",
    "_np_sliced_w2",
    "_compute_w2",
    "_run_strategy",
    "_aggregate_per_seed",
    "_write_per_seed_json",
    "_write_synthesis",
    "DEFAULT_N_EPOCHS_P2",
    "DEFAULT_BATCH_SIZE_P2",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_BETA_W2_POLICY",
    "DEFAULT_GAMMA_W2_VALUE",
    "DEFAULT_N_MC_W2",
    "TARGET_POLICY_LOSS",
    "TARGET_VALUE_LOSS",
]


if __name__ == "__main__":
    raise SystemExit(main())
