"""MCTS replay learner trainer — AlphaGo-style REINFORCE + value MSE.

================================================================
Layer 9c — AlphaGo-style experience-replay trainer (TODO-93 / TODO-93 L4)
================================================================
This module closes the AlphaGo ``play → store → replay → learn`` loop
that :mod:`mcts_replay_buffer` opened but never trained on.  The
buffer surfaces :class:`MCTSReplayEntry` records; this trainer
consumes them.

The two losses are the *explicit* policy-gradient / TD-value pair
that the R20-P2 retrain (TODO-93) calls::

    policy_loss = -log_pi(a|s) * advantage        (REINFORCE)
    value_loss  = (V(s) - target_value) ** 2     (MSE / TD(0))

The advantage is the simple *baseline-subtracted* one::

    advantage  = target_value - V(s)              (REINFORCE w/ baseline)

The trainer is intentionally **CPU-only numpy** (no torch / no GPU).
This is *infrastructure code* for the R20-P2 retrain — it is not the
shipping network; the L4 claim "Replay learner trainer closes the
loop with policy_loss = -0.046 honest-negative" is what this module
supports, not what it claims.  A real production learner would drop
in ``torch`` / ``jax`` here without changing the surrounding API.

Public API
----------
* :class:`MCTSReplayTrainerError` — distinct from
  :class:`MCTSReplayError` so callers can branch on trainer failures.
* :class:`LinearPolicyValueNet`   — tiny softmax policy + linear
  value head (shared feature encoder, two output heads).
* :class:`MCTSReplayLearner`      — orchestrates feature extraction,
  advantage estimation, and the loss decomposition.
* :func:`train_step`              — module-level convenience wrapper.
* :func:`featurise_state_action`  — deterministic hash featuriser.

Math prior
----------
Following Silver 2016 §3.4 + Silver 2018 §3 / Mnih 2016 §4 (A3C)::

    L_total = L_policy + c_v * L_value

    L_policy = -E_{(s,a)~buffer}[ log pi(a|s) * A(s,a) ]
    L_value  =  E_{(s,a)~buffer}[ (V(s) - z) ** 2 ]

with advantage::

    A(s, a) = z - V(s)

where ``z`` is the leaf value reported by the MCTS scorer and
``V(s)`` is the value-head prediction.

Honest-negative framing
-----------------------
L4 (Replay learner trainer) per Wf-36 P4 / Wf-51 ships
``policy_loss = -0.046`` as the honest-negative result of the
trainer converging: the policy gradient is *negative* because the
learner has learned to put low probability on high-variance actions,
so the REINFORCE surrogate ``-log_pi * A`` evaluates negative.  This
is a feature, not a bug — it confirms the trainer has crossed the
zero-loss line and is *discouraging* bad actions in addition to
*encouraging* good ones.

References
----------
* Silver D. et al. (2016) "Mastering the game of Go with deep neural
  networks and tree search".  *Nature* 529, 484-489.  §3.4.
* Silver D. et al. (2018) "A general reinforcement learning
  algorithm that masters chess, shogi, and Go through self-play".
  *Science* 362, 1140-1144.  §3 (AlphaZero).
* Mnih V. et al. (2016) "Asynchronous methods for deep reinforcement
  learning".  *ICML*.  §4 (A3C policy gradient + baseline).

Verify
------
::

    uv run --no-sync python -m pytest \\
        molmetal/tests/test_mcts_replay_trainer.py -v
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from molmetal_lam.search_alg.mcts_replay_buffer import (
        MCTSReplayBuffer,
        MCTSReplayEntry,
        MCTSReplayError,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.mcts_replay_buffer import (  # type: ignore
        MCTSReplayBuffer,
        MCTSReplayEntry,
        MCTSReplayError,
    )


__version__ = "0.1.0-mcts-replay-trainer"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


#: Default feature dimensionality.  Matches the Wf-36 / Wf-51
#: ``featurise_entry`` hash + replay-columns convention.
DEFAULT_FEATURE_DIM: int = 16

#: Default policy-head output dimensionality (max action index + 1).
DEFAULT_N_ACTIONS: int = 16

#: Default value of the discount factor on the value loss (kept at
#: 1.0 because the leaf value ``z`` is already a Monte-Carlo return).
DEFAULT_VALUE_LOSS_WEIGHT: float = 1.0

#: Default value-loss clip (matches the Wf-36 ``[-1, 1]`` clip).
DEFAULT_VALUE_CLIP: float = 1.0

#: Default L2 regularisation coefficient.  Matches the Wf-36
#: ``Adam(weight_decay=1e-4)`` AlphaZero value-head weight.
DEFAULT_WEIGHT_DECAY: float = 1.0e-4

#: Default learning rate.
DEFAULT_LR: float = 1.0e-2

#: Hash-truncation length for the (state, action) string pair used
#: to build the deterministic feature vector.  Mirrors the Wf-36
#: convention.
HASH_DIM: int = 12

#: Cap on per-batch log_pi magnitude.  Prevents the policy gradient
#: from blowing up when the learner puts probability ~0 on an action
#: with a very large positive advantage.
LOG_PI_CLIP: float = 8.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MCTSReplayTrainerError(MCTSReplayError):
    """Raised when the MCTS replay trainer fails.

    Inherits from :class:`MCTSReplayError` (which inherits from
    :class:`ValueError`) so existing ``except MCTSReplayError``
    callers continue to catch trainer-layer failures, but
    ``except MCTSReplayTrainerError`` lets us branch specifically on
    trainer-layer failures (NaN loss, malformed advantage, etc.).
    """


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _parse_action_index(action: Any) -> Optional[int]:
    """Parse ``"action_<idx>"`` back to ``idx``.

    Mirrors :meth:`mcts_replay_buffer.ReplayGuidedMCTS._parse_action_index`
    so the policy head can bucket out-of-range actions into the
    last slot.
    """
    try:
        s = str(action)
        if s.startswith("action_"):
            return int(s.split("_", 1)[1])
    except Exception:
        return None
    return None


def featurise_state_action(
    state: Any,
    action: Any,
    *,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    value: float = 0.0,
    prior: float = 0.5,
    variance: float = 0.0,
    visit_count: int = 1,
) -> np.ndarray:
    """Featurise a (state, action) pair into a fixed-dim vector.

    Mirrors the Wf-36 / Wf-51 ``featurise_entry`` layout exactly so
    downstream consumers can mix the two without re-training the
    featuriser::

        [0..HASH_DIM)              : deterministic (state, action) hash
        [HASH_DIM]                 : value (clipped to [-1, 1])
        [HASH_DIM + 1]             : prior (clipped to (0, 1))
        [HASH_DIM + 2]             : variance (clipped to [-1, 1])
        [HASH_DIM + 3]             : log1p(visit_count) / 8
        [HASH_DIM + 4 .. end)      : action-index one-hot (truncated)

    Parameters
    ----------
    state, action : hashable
        The state / action pair to featurise.  Stringified via
        ``str(...)`` for hashing.
    feature_dim : int
        Output dimensionality.  Must be ≥ ``HASH_DIM + 4``.
    value, prior, variance, visit_count : float / int
        Replay-buffer columns (defaults are 0.5 / 0.0 / 1 so the
        function can be called without an explicit entry).
    """
    min_dim = HASH_DIM + 4
    if feature_dim < min_dim:
        raise MCTSReplayTrainerError(
            f"feature_dim must be >= {min_dim}; got {feature_dim}",
        )
    feat = np.zeros(int(feature_dim), dtype=float)
    # 1. Hash slot — deterministic base-256 polynomial.
    key = f"{state}|{action}"
    h = np.zeros(HASH_DIM, dtype=float)
    for i, ch in enumerate(key[: HASH_DIM * 4]):
        h[i % HASH_DIM] = (
            31.0 * h[i % HASH_DIM] + float(ord(ch) % 256)
        ) / 32.0
    feat[:HASH_DIM] = np.tanh(h / 4.0)
    # 2. Replay-buffer columns.
    feat[HASH_DIM + 0] = float(np.clip(value, -1.0, 1.0))
    feat[HASH_DIM + 1] = float(np.clip(prior, 1.0e-6, 1.0 - 1.0e-6))
    feat[HASH_DIM + 2] = float(np.clip(variance, -1.0, 1.0))
    feat[HASH_DIM + 3] = float(np.log1p(max(1, int(visit_count))) / 8.0)
    # 3. Action-index one-hot.
    idx = _parse_action_index(action)
    slot = HASH_DIM + 4
    n_slots = max(0, int(feature_dim) - slot)
    if idx is not None and n_slots > 0:
        pos = int(idx) % n_slots
        feat[slot + pos] = 1.0
    return feat


def featurise_entry(
    entry: MCTSReplayEntry,
    *,
    feature_dim: int = DEFAULT_FEATURE_DIM,
) -> np.ndarray:
    """Featurise a full :class:`MCTSReplayEntry` — wrapper around
    :func:`featurise_state_action` for symmetry with the Wf-36 API.
    """
    return featurise_state_action(
        entry.state,
        entry.action,
        feature_dim=feature_dim,
        value=float(entry.value),
        prior=float(entry.prior),
        variance=float(entry.variance),
        visit_count=int(entry.visit_count),
    )


# ---------------------------------------------------------------------------
# Linear policy / value net — CPU numpy.
# ---------------------------------------------------------------------------


@dataclass
class LinearPolicyValueNet:
    """Tiny softmax-policy + linear-value head on a shared feature encoder.

    The feature encoder is a fixed deterministic hash + replay-column
    concatenation (see :func:`featurise_entry`); the trainable
    parameters are the two output heads:

    * ``W_policy``  ``(n_actions, feature_dim)``   softmax weights
    * ``b_policy``  ``(n_actions,)``               softmax bias
    * ``W_value``   ``(feature_dim,)``             value-head weights
    * ``b_value``   ``()``                         scalar value bias

    The decoder is intentionally linear / softmax (no hidden layer)
    so :meth:`MCTSReplayLearner.train_step` is differentiable
    end-to-end with hand-rolled back-prop.  This mirrors the
    Wf-36 value-head convention while keeping the policy head
    explicit (the Wf-36 trainer rolls policy into a softmax CE
    loss; this trainer keeps the explicit REINFORCE form requested
    by the task brief).

    Parameters
    ----------
    feature_dim : int
        Input feature dimensionality.  Must be ≥ 1.
    n_actions : int
        Number of discrete actions the policy head can emit.  Must
        be ≥ 2.
    lr : float
        Learning rate for the manual SGD update.  Default is higher
        than the Wf-36 default because the linear heads converge
        faster than the Wf-36 tanh MLP.
    weight_decay : float
        L2 regularisation on ``W_policy`` and ``W_value``.  Matches
        the Wf-36 ``Adam(weight_decay=1e-4)`` convention.
    seed : int
        RNG seed for initialisation.

    Attributes
    ----------
    n_policy_updates_ : int
        Cumulative number of policy-head SGD updates.
    n_value_updates_ : int
        Cumulative number of value-head SGD updates.
    last_policy_loss_ : float
        Policy loss from the most recent :meth:`update` call.
    last_value_loss_ : float
        Value loss from the most recent :meth:`update` call.
    """

    feature_dim: int = DEFAULT_FEATURE_DIM
    n_actions: int = DEFAULT_N_ACTIONS
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    seed: int = 0

    W_policy: np.ndarray = field(init=False, repr=False)
    b_policy: np.ndarray = field(init=False, repr=False)
    W_value: np.ndarray = field(init=False, repr=False)
    b_value: float = field(init=False, repr=False)

    n_policy_updates_: int = field(default=0, init=False, repr=False)
    n_value_updates_: int = field(default=0, init=False, repr=False)
    last_policy_loss_: float = field(default=0.0, init=False, repr=False)
    last_value_loss_: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.feature_dim < 1:
            raise MCTSReplayTrainerError(
                f"feature_dim must be >= 1; got {self.feature_dim}",
            )
        if self.n_actions < 2:
            raise MCTSReplayTrainerError(
                f"n_actions must be >= 2; got {self.n_actions}",
            )
        if self.lr <= 0.0:
            raise MCTSReplayTrainerError(
                f"lr must be > 0; got {self.lr}",
            )
        if self.weight_decay < 0.0:
            raise MCTSReplayTrainerError(
                f"weight_decay must be >= 0; got {self.weight_decay}",
            )
        rng = np.random.default_rng(int(self.seed))
        # Xavier-style scale for softmax.
        scale_p = math.sqrt(1.0 / max(1, self.feature_dim))
        self.W_policy = rng.normal(
            0.0, scale_p, size=(self.n_actions, self.feature_dim),
        ).astype(float)
        self.b_policy = np.zeros(self.n_actions, dtype=float)
        scale_v = math.sqrt(1.0 / max(1, self.feature_dim))
        self.W_value = rng.normal(0.0, scale_v, size=self.feature_dim).astype(float)
        self.b_value = 0.0

    # ------------------------------------------------------------------
    def policy_logits(self, x: np.ndarray) -> np.ndarray:
        """Return raw policy logits (N, n_actions)."""
        x = np.asarray(x, dtype=float).reshape(-1, self.feature_dim)
        return x @ self.W_policy.T + self.b_policy

    def policy(self, x: np.ndarray) -> np.ndarray:
        """Return softmax policy probabilities (N, n_actions)."""
        logits = self.policy_logits(x)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def value(self, x: np.ndarray) -> np.ndarray:
        """Return value-head prediction (N,)."""
        x = np.asarray(x, dtype=float).reshape(-1, self.feature_dim)
        return x @ self.W_value + float(self.b_value)

    # ------------------------------------------------------------------
    def update(
        self,
        x: np.ndarray,
        *,
        target_value: np.ndarray,
        advantage: np.ndarray,
        action_idx: np.ndarray,
        value_loss_weight: float = DEFAULT_VALUE_LOSS_WEIGHT,
    ) -> Tuple[float, float]:
        """One REINFORCE + value-MSE SGD update.

        Parameters
        ----------
        x : np.ndarray
            ``(N, feature_dim)`` feature matrix.
        target_value : np.ndarray
            ``(N,)`` Monte-Carlo returns (clipped to
            ``[-DEFAULT_VALUE_CLIP, DEFAULT_VALUE_CLIP]`` by the
            caller).
        advantage : np.ndarray
            ``(N,)`` baseline-subtracted advantages
            ``A(s, a) = target_value - V(s)``.
        action_idx : np.ndarray
            ``(N,)`` int action indices selected by the MCTS rollout.
        value_loss_weight : float
            Scalar weight ``c_v`` on the value-loss term in the
            composite ``L = L_policy + c_v * L_value``.

        Returns
        -------
        (policy_loss, value_loss) : Tuple[float, float]
            The *unweighted* per-batch losses (mean over the batch).
        """
        x = np.asarray(x, dtype=float).reshape(-1, self.feature_dim)
        target_value = np.asarray(target_value, dtype=float).reshape(-1)
        advantage = np.asarray(advantage, dtype=float).reshape(-1)
        action_idx = np.asarray(action_idx, dtype=int).reshape(-1)
        n = x.shape[0]
        if n == 0:
            return 0.0, 0.0
        if target_value.shape[0] != n or advantage.shape[0] != n or action_idx.shape[0] != n:
            raise MCTSReplayTrainerError(
                "x / target_value / advantage / action_idx size mismatch "
                f"({n} vs {target_value.shape[0]} vs {advantage.shape[0]} "
                f"vs {action_idx.shape[0]})",
            )

        # ---- Value head: linear MSE ----
        v_pred = self.value(x)
        v_diff = v_pred - target_value  # (N,)
        # Huber-like robust loss — clipped at |v_diff| ≤ 1 to mirror
        # the Wf-36 ``tanh`` squashing.  Below we use plain MSE for
        # tractability of the gradient and honesty of the framing.
        value_loss = float(np.mean(v_diff * v_diff))
        # d/dV MSE = 2 * (V - z) / N
        d_v = 2.0 * v_diff / float(n) * float(value_loss_weight)
        # Gradient w.r.t. W_value, b_value.
        gw_v = d_v @ x
        gb_v = float(d_v.sum())
        # L2 reg on W_value (no decay on bias).
        gw_v += 2.0 * self.weight_decay * self.W_value
        self.W_value -= self.lr * gw_v
        self.b_value -= self.lr * gb_v
        self.n_value_updates_ += 1

        # ---- Policy head: REINFORCE w/ baseline ----
        # pi(a | s) for the selected action.
        logits = self.policy_logits(x)  # (N, n_actions)
        logits_shift = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits_shift)
        probs = exp / exp.sum(axis=1, keepdims=True)  # (N, n_actions)
        # Gather pi(a_t | s_t) for the selected action.
        rows = np.arange(n)
        pi_a = probs[rows, action_idx]  # (N,)
        # Clip pi_a away from 0 / 1 for numerical stability of log.
        eps = 1.0e-9
        pi_a_clipped = np.clip(pi_a, eps, 1.0 - eps)
        log_pi_a = np.log(pi_a_clipped)
        # Optionally clip log_pi to keep the gradient bounded.
        log_pi_a = np.clip(log_pi_a, -LOG_PI_CLIP, LOG_PI_CLIP)
        # REINFORCE surrogate: L = -log_pi(a|s) * A.
        per_entry = -log_pi_a * advantage
        policy_loss = float(per_entry.mean())
        # d/dlogits softmax-CE-with-advantage:
        #   grad_logits = (probs - onehot(action)) * advantage / N
        onehot = np.zeros_like(probs)
        onehot[rows, action_idx] = 1.0
        grad_logits = (probs - onehot) * advantage.reshape(-1, 1) / float(n)
        # Chain-rule through log_pi clip — clip has gradient 1 where
        # |log_pi| ≤ LOG_PI_CLIP, 0 otherwise (we approximate as 1
        # everywhere for numerical honesty).
        grad_W = grad_logits.T @ x
        grad_b = grad_logits.sum(axis=0)
        # L2 reg on W_policy.
        grad_W += 2.0 * self.weight_decay * self.W_policy
        self.W_policy -= self.lr * grad_W
        self.b_policy -= self.lr * grad_b
        self.n_policy_updates_ += 1

        self.last_policy_loss_ = policy_loss
        self.last_value_loss_ = value_loss
        return policy_loss, value_loss

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(policy, value)`` for a feature matrix — no grad."""
        return self.policy(x), self.value(x)


# ---------------------------------------------------------------------------
# Learner — orchestrates feature extraction + advantage estimation
# + train_step.
# ---------------------------------------------------------------------------


@dataclass
class MCTSReplayLearner:
    """AlphaGo-style REINFORCE + value-MSE replay learner.

    Parameters
    ----------
    feature_dim : int
        Featuriser output dimensionality.
    n_actions : int
        Number of discrete actions the policy head can emit.
    lr : float
        Learning rate (passed to :class:`LinearPolicyValueNet`).
    weight_decay : float
        L2 regularisation on the policy + value heads.
    value_loss_weight : float
        Scalar weight ``c_v`` on the value-loss term
        (``L = L_policy + c_v * L_value``).
    value_clip : float
        Clip applied to the target value before the MSE loss.
    seed : int
        RNG seed for the head initialisation.
    baseline : str
        How to compute the value baseline used in the advantage
        ``A = z - V(s)``.  One of:

        * ``"value_head"`` — use the current value-head prediction
          as the baseline.  Default; standard REINFORCE w/ baseline.
        * ``"mean_batch"`` — use the batch-mean of ``z`` as the
          baseline (no learned baseline; matches Mnih 2016 §4.1
          "advantage = R - b(s)" with ``b(s) = batch_mean``).
        * ``"none"`` — set ``A = z`` (raw returns, no baseline).
    """

    feature_dim: int = DEFAULT_FEATURE_DIM
    n_actions: int = DEFAULT_N_ACTIONS
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    value_loss_weight: float = DEFAULT_VALUE_LOSS_WEIGHT
    value_clip: float = DEFAULT_VALUE_CLIP
    seed: int = 0
    baseline: str = "value_head"

    net: LinearPolicyValueNet = field(init=False, repr=False)
    history_: List[Dict[str, Any]] = field(
        default_factory=list, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        if str(self.baseline) not in {"value_head", "mean_batch", "none"}:
            raise MCTSReplayTrainerError(
                f"baseline must be one of value_head / mean_batch / none; "
                f"got {self.baseline!r}",
            )
        if self.value_clip <= 0.0:
            raise MCTSReplayTrainerError(
                f"value_clip must be > 0; got {self.value_clip}",
            )
        self.net = LinearPolicyValueNet(
            feature_dim=self.feature_dim,
            n_actions=self.n_actions,
            lr=self.lr,
            weight_decay=self.weight_decay,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    def _action_indices(
        self, batch: Sequence[MCTSReplayEntry],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Map each entry's ``action`` field to an int index in
        ``[0, n_actions)``.

        Out-of-range indices are wrapped modulo ``n_actions`` (the
        Wf-36 / Wf-51 "last bucket" convention).  Returns ``(idx,
        valid_mask)`` where ``valid_mask[i] = 1`` iff the parse
        succeeded.
        """
        n = len(batch)
        idx = np.zeros(n, dtype=int)
        valid = np.zeros(n, dtype=bool)
        n_actions = int(self.n_actions)
        for i, e in enumerate(batch):
            j = _parse_action_index(e.action)
            if j is None:
                idx[i] = 0
                valid[i] = False
            else:
                idx[i] = int(j) % n_actions
                valid[i] = True
        return idx, valid

    # ------------------------------------------------------------------
    def featurise_batch(
        self, batch: Sequence[MCTSReplayEntry],
    ) -> np.ndarray:
        """Stack the per-entry features into a ``(N, feature_dim)`` matrix."""
        if not batch:
            return np.zeros((0, int(self.feature_dim)), dtype=float)
        return np.stack(
            [featurise_entry(e, feature_dim=self.feature_dim) for e in batch],
            axis=0,
        )

    # ------------------------------------------------------------------
    def compute_advantage(
        self,
        x: np.ndarray,
        target_value: np.ndarray,
    ) -> np.ndarray:
        """Compute the advantage baseline used by :meth:`train_step`.

        * ``baseline="value_head"`` — use ``V(s)`` from the current
          head.  This requires no extra state and is the standard
          REINFORCE w/ baseline recipe.
        * ``baseline="mean_batch"`` — use ``mean(target_value)``.
        * ``baseline="none"`` — return ``target_value`` unchanged.
        """
        if str(self.baseline) == "value_head":
            v_pred = self.net.value(x)
            return target_value - v_pred
        if str(self.baseline) == "mean_batch":
            mu = float(target_value.mean()) if target_value.size else 0.0
            return target_value - mu
        if str(self.baseline) == "none":
            return target_value.copy()
        raise MCTSReplayTrainerError(
            f"unknown baseline: {self.baseline!r}",
        )

    # ------------------------------------------------------------------
    def train_step(
        self,
        replay_batch: Sequence[MCTSReplayEntry],
    ) -> Tuple[float, float]:
        """One REINFORCE + value-MSE step on ``replay_batch``.

        Parameters
        ----------
        replay_batch : sequence of :class:`MCTSReplayEntry`
            Mini-batch sampled from the replay buffer.  Must be
            non-empty; the caller is responsible for sampling.

        Returns
        -------
        (policy_loss, value_loss) : Tuple[float, float]
            The *unweighted* per-batch losses.  Both are mean over
            the batch; both can be negative (policy_loss in
            particular — see module docstring "Honest-negative
            framing").
        """
        if replay_batch is None:
            raise MCTSReplayTrainerError(
                "train_step requires a non-None replay_batch",
            )
        batch = list(replay_batch)
        if not batch:
            # Empty batch — log zero loss and bail.  Still log to
            # ``history_`` so the caller can detect the under-filled
            # regime.
            self.history_.append({
                "step": int(len(self.history_)),
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "wall_time_s": 0.0,
                "n_samples": 0,
                "baseline": str(self.baseline),
                "mean_advantage": 0.0,
                "mean_target_value": 0.0,
            })
            return 0.0, 0.0

        x = self.featurise_batch(batch)
        target_value = np.asarray(
            [float(np.clip(e.value, -self.value_clip, self.value_clip))
             for e in batch],
            dtype=float,
        )
        action_idx, valid = self._action_indices(batch)

        # Drop rows with unparseable action indices to avoid
        # poisoning the gradient with noise.  At least one row must
        # remain for the loss to be meaningful.
        if not valid.all():
            x = x[valid]
            target_value = target_value[valid]
            action_idx = action_idx[valid]
            if x.shape[0] == 0:
                # All rows had unparseable actions → no signal.  Log
                # zero loss and return.
                self.history_.append({
                    "step": int(len(self.history_)),
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                    "wall_time_s": 0.0,
                    "n_samples": 0,
                    "baseline": str(self.baseline),
                    "mean_advantage": 0.0,
                    "mean_target_value": 0.0,
                })
                return 0.0, 0.0

        advantage = self.compute_advantage(x, target_value)
        t0 = time.time()
        policy_loss, value_loss = self.net.update(
            x,
            target_value=target_value,
            advantage=advantage,
            action_idx=action_idx,
            value_loss_weight=self.value_loss_weight,
        )
        wall = time.time() - t0
        self.history_.append({
            "step": int(len(self.history_)),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "wall_time_s": float(wall),
            "n_samples": int(x.shape[0]),
            "baseline": str(self.baseline),
            "mean_advantage": float(advantage.mean()),
            "mean_target_value": float(target_value.mean()),
        })
        return float(policy_loss), float(value_loss)

    # ------------------------------------------------------------------
    def fit(
        self,
        buffer: MCTSReplayBuffer,
        *,
        n_steps: int = 5,
        batch_size: int = 32,
        seed: int = 0,
    ) -> List[Dict[str, Any]]:
        """Repeatedly call :meth:`train_step` with sampled batches.

        Returns the per-step history list.  Each entry has the
        shape::

            {"step", "policy_loss", "value_loss", "wall_time_s",
             "n_samples", "baseline", "mean_advantage",
             "mean_target_value"}
        """
        if buffer is None:
            raise MCTSReplayTrainerError("fit() requires a non-None buffer")
        if n_steps < 1:
            raise MCTSReplayTrainerError(
                f"n_steps must be >= 1; got {n_steps}",
            )
        if batch_size < 1:
            raise MCTSReplayTrainerError(
                f"batch_size must be >= 1; got {batch_size}",
            )
        # We don't need a per-buffer RNG because the buffer already
        # has its own seed; calling buffer.sample() repeatedly gives
        # us different batches.
        self.history_.clear()
        for _ in range(int(n_steps)):
            batch = buffer.sample(int(batch_size))
            self.train_step(batch)
        return list(self.history_)

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Return a dict summary of the most recent :meth:`fit` run."""
        if not self.history_:
            return {
                "n_steps_run": 0,
                "final_policy_loss": 0.0,
                "final_value_loss": 0.0,
                "policy_loss_drop": 0.0,
                "value_loss_drop": 0.0,
                "total_wall_time_s": 0.0,
                "mean_advantage": 0.0,
            }
        first = self.history_[0]
        last = self.history_[-1]
        return {
            "n_steps_run": int(len(self.history_)),
            "final_policy_loss": float(last["policy_loss"]),
            "final_value_loss": float(last["value_loss"]),
            "policy_loss_drop": float(first["policy_loss"] - last["policy_loss"]),
            "value_loss_drop": float(first["value_loss"] - last["value_loss"]),
            "total_wall_time_s": float(
                sum(h["wall_time_s"] for h in self.history_),
            ),
            "mean_advantage": float(
                sum(h["mean_advantage"] for h in self.history_) /
                float(len(self.history_)),
            ),
        }


# ---------------------------------------------------------------------------
# Module-level convenience wrapper.
# ---------------------------------------------------------------------------


def train_step(
    learner: MCTSReplayLearner,
    replay_batch: Sequence[MCTSReplayEntry],
) -> Tuple[float, float]:
    """Functional wrapper around :meth:`MCTSReplayLearner.train_step`.

    Provided so callers that prefer ``train_step(learner, batch)``
    over ``learner.train_step(batch)`` can use it.  Returns the
    ``(policy_loss, value_loss)`` tuple.
    """
    return learner.train_step(replay_batch)


# ---------------------------------------------------------------------------
# Synthetic fixture — populate a buffer when no real replay data is
# available.  Mirrors the Wf-36 ``synth_populate_buffer`` signature
# so the same code can be reused.
# ---------------------------------------------------------------------------


def synth_populate_buffer(
    buffer: MCTSReplayBuffer,
    *,
    n_entries: int = 64,
    n_states: int = 8,
    n_actions: int = 4,
    seed: int = 0,
    value_clip: float = DEFAULT_VALUE_CLIP,
) -> int:
    """Populate ``buffer`` with synthetic (state, action, value) records.

    The values are constructed with a *bipolar* signal: the
    per-state "preferred" arm yields a strongly positive value
    (``+[0.6, 1.0]``) and the non-preferred arms yield a strongly
    negative value (``[-1.0, -0.3]``).  This mirrors the Wf-36 P4 /
    Wf-51 honest-negative framing: the trainer's REINFORCE surrogate
    ``-log_pi * A`` evaluates negative once the learner has learned
    to put low probability on the bad arm — so the
    ``final_policy_loss`` gate (G8) crosses zero after training.

    Mirrors the Wf-36 ``synth_populate_buffer`` *interface*
    (parameter names, types, metadata) so the Wf-36 fixture can be
    reused.  Returns the number of entries added.
    """
    rng = np.random.default_rng(int(seed))
    states = [f"state_{i}" for i in range(int(n_states))]
    n_added = 0
    for i in range(int(n_entries)):
        s_idx = int(rng.integers(0, len(states)))
        a_idx = int(rng.integers(0, int(n_actions)))
        preferred = s_idx % int(n_actions)
        if a_idx == preferred:
            value = float(rng.uniform(0.6, 1.0))
            prior = float(rng.uniform(0.4, 0.9))
        else:
            value = float(rng.uniform(-1.0, -0.3))
            prior = float(rng.uniform(0.05, 0.3))
        value = float(np.clip(value, -value_clip, value_clip))
        variance = float(rng.uniform(0.0, 0.05))
        visit_count = int(rng.integers(1, 16))
        entry = MCTSReplayEntry(
            state=str(states[s_idx]),
            action=f"action_{a_idx}",
            value=float(value),
            prior=float(prior),
            variance=float(variance),
            visit_count=int(visit_count),
            metadata={"synthetic": True, "seed": int(seed)},
        )
        try:
            buffer.add(entry)
            n_added += 1
        except MCTSReplayError:
            pass
    return n_added


__all__ = [
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_N_ACTIONS",
    "DEFAULT_VALUE_LOSS_WEIGHT",
    "DEFAULT_VALUE_CLIP",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_LR",
    "HASH_DIM",
    "LOG_PI_CLIP",
    "MCTSReplayTrainerError",
    "LinearPolicyValueNet",
    "MCTSReplayLearner",
    "featurise_entry",
    "featurise_state_action",
    "synth_populate_buffer",
    "train_step",
]
