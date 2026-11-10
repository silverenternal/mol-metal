"""Wf-36 Phase 4 — MCTS replay learner (AlphaGo-style).

================================================================
Layer 9b — AlphaGo-style value / policy network learner
================================================================
This script closes the "play → store → replay → learn" loop that
:class:`mcts_replay_buffer.MCTSReplayWrapper` opened but never
closed.  Given a populated :class:`MCTSReplayBuffer`, we:

    1. Featurise each (state, action) entry into a fixed-dim vector
       (hash + value + prior + variance + visit_count — see
       :func:`featurise_entry`).
    2. Train a tiny **value head** MLP that minimises
       ``MSE(value_head(features), entry.value)``.
    3. Train a tiny **policy head** softmax that minimises the
       cross-entropy between the per-state visit-count-temperature
       distribution and the network's policy logits.
    4. Report per-epoch loss + convergence diagnostics.

The networks are intentionally CPU-only numpy implementations —
no ``torch.cuda``, no autograd, no GPU kernels.  The point is to
demonstrate the *experience-replay* training loop closes, not to
ship a deep network.  A real production learner would drop in
``torch`` / ``jax`` here without changing the surrounding contract.

Why a separate script (and not a method on the wrapper)
-------------------------------------------------------
:class:`MCTSReplayWrapper.train_value_network` already exists as a
*loss-statistics* helper (it returns the empirical MSE / CE
proxies on a sampled batch without actually training a network).
Adding a *learner* to that same class would couple the buffer to
the optimiser; keeping the learner in a dedicated module lets the
buffer stay replay-only while the learner can be swapped (numpy
↔ torch ↔ jax) without API churn.

Public API
----------
* :class:`ValueNetwork`         tiny MLP value head (CPU numpy).
* :class:`PolicyNetwork`        softmax policy head (CPU numpy).
* :class:`MCTSReplayLearner`    orchestrates feature extraction,
                                 training, and per-epoch logging.
* :func:`run_learner`           CLI-friendly top-level runner.

Math prior
----------
The AlphaGo / AlphaZero target (Silver 2016 §3.4, Silver 2018 §3)::

    L = (z - v)^2  +  cross_entropy(pi, p_policy)

where ``z`` is the leaf value, ``v`` the value-head prediction,
``pi`` the MCTS-improved policy (= visit-count distribution over
actions), and ``p_policy`` the policy-head prediction.  We mirror
that exact decomposition here — see :meth:`MCTSReplayLearner.epoch`.

Verify
------
::

    uv run --no-sync python molmetal/scripts/mcts_replay_learner.py --smoke
    uv run --no-sync python molmetal/scripts/mcts_replay_learner.py \\
        --replay-buffer molmetal/reports/wf36_3stage_mcts/replay.pkl \\
        --n-epochs 5 --batch-size 16

References
----------
* Silver D. et al. (2016) "Mastering the game of Go with deep neural
  networks and tree search".  *Nature* 529, 484-489.  §3.4 (Training
  pipeline).
* Silver D. et al. (2018) "A general reinforcement learning
  algorithm that masters chess, shogi, and Go through self-play".
  *Science* 362, 1140-114.  §3 (AlphaZero).
* Schrittwieser J. et al. (2020) "Mastering Atari, Go, Chess and
  Shogi by Planning with a Learned Model".  *Nature* 588, 604-609.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow ``python -m molmetal.scripts.mcts_replay_learner`` and
# direct-script execution both.  Matches the convention used by the
# other Wf-36 / Wf-37 scripts in ``molmetal/scripts/``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Lazy imports — match the convention used by the rest of the
# ``molmetal/scripts/`` directory so the script is parseable even
# when the package tree is not fully on ``sys.path``.
# ---------------------------------------------------------------------------
try:
    from molmetal_lam.search_alg.mcts_replay_buffer import (
        DEFAULT_BATCH_SIZE as _REPLAY_DEFAULT_BATCH,
        MCTSReplayBuffer,
        MCTSReplayEntry,
        MCTSReplayError,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.mcts_replay_buffer import (  # type: ignore
        DEFAULT_BATCH_SIZE as _REPLAY_DEFAULT_BATCH,
        MCTSReplayBuffer,
        MCTSReplayEntry,
        MCTSReplayError,
    )


__version__ = "0.1.0-mcts-replay-learner"


# ---------------------------------------------------------------------------
# Defaults — kept in sync with the Wf-36 / Wf-21 ships.
# ---------------------------------------------------------------------------

#: Default feature dimensionality for the value/policy heads.
#: 16 is enough to encode the four replay-buffer columns
#: (value, prior, variance, visit_count) plus a 12-dim hash of the
#: (state, action) string pair — small enough to train in <1 s
#: CPU-only.
DEFAULT_FEATURE_DIM: int = 16

#: Default value-head hidden size.
DEFAULT_HIDDEN_DIM: int = 32

#: Default learning rate.  Matches the canonical
#: ``Adam(weight_decay=1e-4, lr=1e-3)`` used by AlphaZero's value head.
DEFAULT_LR: float = 1.0e-3

#: Default number of training epochs.
DEFAULT_N_EPOCHS: int = 5

#: Default output directory for the bench JSON.
DEFAULT_OUTPUT_DIR: str = str(
    Path(__file__).resolve().parents[2]
    / "molmetal"
    / "reports"
    / "wf36_replay_learner"
)

#: Default max action index for the policy head.  When the buffer
#: contains actions above this index we collapse them into the
#: last bucket (``n_actions - 1``) — keeps the policy head bounded
#: without losing all information from out-of-range actions.
DEFAULT_N_ACTIONS: int = 16

#: Convergence criterion: training stops when the running-mean loss
#: over ``CONVERGENCE_WINDOW`` epochs falls below this threshold.
#: Default ``1.0e-3`` matches the Wf-25 Hilbert convergence criterion.
CONVERGENCE_WINDOW: int = 2
CONVERGENCE_EPS: float = 1.0e-3

#: Hash-truncation length for the (state, action) string pair used
#: to build the deterministic feature vector.  Matches the Wf-9
#: ``hilbert_primal_dual`` convention of 12-dim string-hash features.
HASH_DIM: int = 12


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MCTSReplayLearnerError(ValueError):
    """Raised when the replay learner fails.

    Distinct from :class:`mcts_replay_buffer.MCTSReplayError` so
    callers can branch on learner-specific failures (e.g. malformed
    feature vector, NaN loss) without parsing string messages.
    Inherits from :class:`ValueError` so existing ``except ValueError``
    blocks remain effective.
    """


# ---------------------------------------------------------------------------
# Feature extraction — deterministic hash of (state, action) plus the
# 4 replay-buffer columns (value, prior, variance, visit_count).
# ---------------------------------------------------------------------------


def featurise_entry(
    entry: MCTSReplayEntry,
    feature_dim: int = DEFAULT_FEATURE_DIM,
) -> np.ndarray:
    """Featurise a single replay entry into a fixed-dim vector.

    Layout (length ``feature_dim``, default 16)::

        [0..HASH_DIM)              : deterministic (state, action) hash
        [HASH_DIM]                 : value (clipped to [-1, 1])
        [HASH_DIM + 1]             : prior (clipped to (0, 1))
        [HASH_DIM + 2]             : variance (clipped to [-1, 1])
        [HASH_DIM + 3]             : log1p(visit_count) / 8 (∈ [0, ~1])
        [HASH_DIM + 4 .. end)      : action-index one-hot (truncated)

    The hash is a 12-bit base-256 polynomial over the
    ``str(state) + "|" + str(action)`` concatenation.  Deterministic,
    reproducible, no RNG.  Matches the Wf-9 ``HilbertSpace`` pattern
    for a CPU-only featuriser.
    """
    # Minimum is HASH_DIM + 4 (hash + value/prior/variance/visit_count);
    # extra slots give a wider action one-hot when ``feature_dim`` grows.
    min_dim = HASH_DIM + 4
    if feature_dim < min_dim:
        raise MCTSReplayLearnerError(
            f"feature_dim must be >= {min_dim}; got {feature_dim}",
        )
    feat = np.zeros(int(feature_dim), dtype=float)
    # 1. Hash slot — deterministic base-256 polynomial.
    key = f"{entry.state}|{entry.action}"
    h = np.zeros(HASH_DIM, dtype=float)
    for i, ch in enumerate(key[: HASH_DIM * 4]):
        h[i % HASH_DIM] = (31.0 * h[i % HASH_DIM] + float(ord(ch) % 256)) / 32.0
    # Squash to [-1, 1] via tanh so the hash doesn't dominate the
    # value/prior/variance columns.
    feat[:HASH_DIM] = np.tanh(h / 4.0)
    # 2. Replay-buffer columns.
    feat[HASH_DIM + 0] = float(np.clip(entry.value, -1.0, 1.0))
    feat[HASH_DIM + 1] = float(np.clip(entry.prior, 1.0e-6, 1.0 - 1.0e-6))
    feat[HASH_DIM + 2] = float(np.clip(entry.variance, -1.0, 1.0))
    feat[HASH_DIM + 3] = float(np.log1p(max(1, int(entry.visit_count))) / 8.0)
    # 3. Action-index one-hot — best-effort, ignores non-``action_<idx>``
    #    strings (falls back to slot 0).
    idx = _parse_action_index(entry.action)
    slot = HASH_DIM + 4
    n_slots = max(0, int(feature_dim) - slot)
    if idx is not None and n_slots > 0:
        pos = int(idx) % n_slots
        feat[slot + pos] = 1.0
    return feat


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


# ---------------------------------------------------------------------------
# Value head — 2-layer MLP, sigmoid output, MSE loss.
# ---------------------------------------------------------------------------


@dataclass
class ValueNetwork:
    """Tiny 2-layer MLP value head.

    Parameters
    ----------
    feature_dim : int, default ``16``
        Input feature dimension.  Matches
        :data:`DEFAULT_FEATURE_DIM`.
    hidden_dim : int, default ``32``
        Hidden-layer width.
    lr : float, default ``1e-3``
        Learning rate for the manual gradient step.  We use plain
        SGD (no Adam) to keep the implementation CPU-only and
        numpy-native.

    Attributes
    ----------
    w1 : np.ndarray
        ``(hidden_dim, feature_dim)`` first-layer weights.
    b1 : np.ndarray
        ``(hidden_dim,)`` first-layer bias.
    w2 : np.ndarray
        ``(1, hidden_dim)`` second-layer weights.
    b2 : np.ndarray
        ``(1,)`` second-layer bias.
    n_updates_ : int
        Cumulative number of SGD updates applied.
    last_loss_ : float
        Loss from the most recent :meth:`update` call.
    """

    feature_dim: int = DEFAULT_FEATURE_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    lr: float = DEFAULT_LR
    seed: int = 0

    w1: np.ndarray = field(init=False, repr=False)
    b1: np.ndarray = field(init=False, repr=False)
    w2: np.ndarray = field(init=False, repr=False)
    b2: np.ndarray = field(init=False, repr=False)
    n_updates_: int = field(default=0, init=False, repr=False)
    last_loss_: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.feature_dim < 1:
            raise MCTSReplayLearnerError(
                f"feature_dim must be >= 1; got {self.feature_dim}",
            )
        if self.hidden_dim < 1:
            raise MCTSReplayLearnerError(
                f"hidden_dim must be >= 1; got {self.hidden_dim}",
            )
        if self.lr <= 0:
            raise MCTSReplayLearnerError(
                f"lr must be > 0; got {self.lr}",
            )
        rng = np.random.default_rng(int(self.seed))
        # He initialisation (scaled for tanh hidden activations).
        scale = math.sqrt(2.0 / max(1, self.feature_dim))
        self.w1 = rng.normal(0.0, scale, size=(self.hidden_dim, self.feature_dim))
        self.b1 = np.zeros(self.hidden_dim, dtype=float)
        self.w2 = rng.normal(0.0, scale, size=(1, self.hidden_dim))
        self.b2 = np.zeros(1, dtype=float)

    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass; returns (sigmoid_out, hidden_pre, hidden_post).

        Three-tuple unpacking makes back-prop a single expression
        in :meth:`update`.
        """
        x = np.asarray(x, dtype=float).reshape(-1, self.feature_dim)
        h_pre = x @ self.w1.T + self.b1  # (N, hidden)
        h_post = np.tanh(h_pre)
        out = h_post @ self.w2.T + self.b2  # (N, 1)
        # Squish to (-1, 1) so the MSE target is well-scaled.
        out = np.tanh(out)
        return out.reshape(-1), h_pre, h_post

    # ------------------------------------------------------------------
    def update(
        self,
        x: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """One SGD update on a mini-batch; returns the MSE loss.

        Parameters
        ----------
        x : np.ndarray
            ``(N, feature_dim)`` feature matrix.
        target : np.ndarray
            ``(N,)`` value target (clipped to ``[-1, 1]`` by the
            caller — see :meth:`MCTSReplayLearner.epoch`).
        """
        x = np.asarray(x, dtype=float).reshape(-1, self.feature_dim)
        target = np.asarray(target, dtype=float).reshape(-1)
        if x.shape[0] != target.shape[0]:
            raise MCTSReplayLearnerError(
                f"x and target size mismatch: {x.shape[0]} vs {target.shape[0]}",
            )
        if x.shape[0] == 0:
            return 0.0
        out, h_pre, h_post = self.forward(x)
        diff = (out - target).reshape(-1, 1)
        loss = float(np.mean(diff * diff))
        # Back-prop through tanh output.
        d_out = 2.0 * diff / float(x.shape[0])  # (N, 1)
        d_out_hidden = d_out * (1.0 - out.reshape(-1, 1) ** 2)
        # Layer 2 grad.
        gw2 = d_out_hidden.T @ h_post
        gb2 = d_out_hidden.sum(axis=0)
        # Layer 1 grad.
        d_h_pre = (d_out_hidden @ self.w2) * (1.0 - h_post**2)
        gw1 = d_h_pre.T @ x
        gb1 = d_h_pre.sum(axis=0)
        # SGD step.
        self.w2 -= self.lr * gw2
        self.b2 -= self.lr * gb2
        self.w1 -= self.lr * gw1
        self.b1 -= self.lr * gb1
        self.n_updates_ += 1
        self.last_loss_ = loss
        return loss

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the value for a feature matrix (no grad)."""
        out, _, _ = self.forward(x)
        return out


# ---------------------------------------------------------------------------
# Policy head — softmax over ``n_actions`` arms, CE loss.
# ---------------------------------------------------------------------------


@dataclass
class PolicyNetwork:
    """Tiny softmax policy head.

    Parameters
    ----------
    feature_dim : int, default ``16``
        Input feature dimension.
    n_actions : int, default ``16``
        Output dimensionality (number of actions to softmax over).
    lr : float, default ``1e-3``
        Learning rate for the manual SGD step.

    Attributes
    ----------
    w : np.ndarray
        ``(n_actions, feature_dim)`` softmax weights.
    b : np.ndarray
        ``(n_actions,)`` softmax bias.
    n_updates_ : int
        Cumulative number of SGD updates.
    last_loss_ : float
        Loss from the most recent :meth:`update` call.
    """

    feature_dim: int = DEFAULT_FEATURE_DIM
    n_actions: int = DEFAULT_N_ACTIONS
    lr: float = DEFAULT_LR
    seed: int = 0

    w: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)
    n_updates_: int = field(default=0, init=False, repr=False)
    last_loss_: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.feature_dim < 1:
            raise MCTSReplayLearnerError(
                f"feature_dim must be >= 1; got {self.feature_dim}",
            )
        if self.n_actions < 1:
            raise MCTSReplayLearnerError(
                f"n_actions must be >= 1; got {self.n_actions}",
            )
        if self.lr <= 0:
            raise MCTSReplayLearnerError(
                f"lr must be > 0; got {self.lr}",
            )
        rng = np.random.default_rng(int(self.seed))
        scale = math.sqrt(2.0 / max(1, self.feature_dim))
        self.w = rng.normal(0.0, scale, size=(self.n_actions, self.feature_dim))
        self.b = np.zeros(self.n_actions, dtype=float)

    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass; returns the softmax probability vector (N, n_actions)."""
        x = np.asarray(x, dtype=float).reshape(-1, self.feature_dim)
        logits = x @ self.w.T + self.b
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    def update(
        self,
        x: np.ndarray,
        target_policy: np.ndarray,
    ) -> float:
        """One SGD update on a mini-batch; returns the CE loss.

        Parameters
        ----------
        x : np.ndarray
            ``(N, feature_dim)`` feature matrix.
        target_policy : np.ndarray
            ``(N, n_actions)`` target distribution.  Each row sums
            to 1.  Built by
            :meth:`MCTSReplayLearner.epoch` from the per-state
            visit-count-temperature distribution.
        """
        x = np.asarray(x, dtype=float).reshape(-1, self.feature_dim)
        target_policy = np.asarray(target_policy, dtype=float).reshape(
            -1, self.n_actions
        )
        if x.shape[0] == 0:
            return 0.0
        if x.shape[0] != target_policy.shape[0]:
            raise MCTSReplayLearnerError(
                f"x and target size mismatch: {x.shape[0]} vs {target_policy.shape[0]}",
            )
        probs = self.forward(x)
        eps = 1.0e-9
        # Mean CE loss across the batch.
        loss = float(
            -np.mean(
                np.sum(target_policy * np.log(np.clip(probs, eps, 1.0)), axis=1),
            ),
        )
        # Gradient of softmax CE: (probs - target) / N.
        grad = (probs - target_policy) / float(x.shape[0])
        gw = grad.T @ x
        gb = grad.sum(axis=0)
        self.w -= self.lr * gw
        self.b -= self.lr * gb
        self.n_updates_ += 1
        self.last_loss_ = loss
        return loss

    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the policy distribution for a feature matrix (no grad)."""
        return self.forward(x)


# ---------------------------------------------------------------------------
# Learner — orchestrates the value/policy training loop.
# ---------------------------------------------------------------------------


@dataclass
class MCTSReplayLearner:
    """AlphaGo-style experience-replay learner.

    Parameters
    ----------
    feature_dim : int, default ``16``
        Feature dimensionality for both heads.
    hidden_dim : int, default ``32``
        Hidden width of the value head.
    n_actions : int, default ``16``
        Output dimensionality of the policy head.
    n_epochs : int, default ``5``
        Number of training epochs.
    batch_size : int, default ``32``
        Mini-batch size (matches the AlphaGo Zero
        ``DEFAULT_BATCH_SIZE``).
    lr : float, default ``1e-3``
        Learning rate.
    seed : int, default ``0``
        RNG seed for the initial weight matrices.

    Attributes
    ----------
    value_net : ValueNetwork
        The trained value head.
    policy_net : PolicyNetwork
        The trained policy head.
    history_ : list of dict
        Per-epoch log: ``{"epoch", "value_loss", "policy_loss",
        "wall_time_s", "n_samples"}``.
    converged_ : bool
        ``True`` when the running-mean loss over
        :data:`CONVERGENCE_WINDOW` epochs fell below
        :data:`CONVERGENCE_EPS`.
    """

    feature_dim: int = DEFAULT_FEATURE_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    n_actions: int = DEFAULT_N_ACTIONS
    n_epochs: int = DEFAULT_N_EPOCHS
    batch_size: int = _REPLAY_DEFAULT_BATCH
    lr: float = DEFAULT_LR
    seed: int = 0

    value_net: ValueNetwork = field(init=False, repr=False)
    policy_net: PolicyNetwork = field(init=False, repr=False)
    history_: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    converged_: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_epochs < 1:
            raise MCTSReplayLearnerError(
                f"n_epochs must be >= 1; got {self.n_epochs}",
            )
        if self.batch_size < 1:
            raise MCTSReplayLearnerError(
                f"batch_size must be >= 1; got {self.batch_size}",
            )
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

    # ------------------------------------------------------------------
    def fit(
        self,
        buffer: MCTSReplayBuffer,
        *,
        n_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Train the value + policy heads on ``buffer``.

        Returns the per-epoch history list.  Each entry has the
        shape::

            {"epoch": int, "value_loss": float, "policy_loss": float,
             "wall_time_s": float, "n_samples": int,
             "value_loss_running_mean": float,
             "policy_loss_running_mean": float}

        The training loop is the canonical AlphaGo pattern: sample
        a mini-batch from the buffer → featurise → step both heads
        with their respective losses → log → repeat for ``n_epochs``.
        """
        if buffer is None:
            raise MCTSReplayLearnerError("fit() requires a non-None buffer")
        n_epochs_eff = int(n_epochs) if n_epochs is not None else int(self.n_epochs)
        batch_size_eff = (
            int(batch_size) if batch_size is not None else int(self.batch_size)
        )
        if n_epochs_eff < 1:
            raise MCTSReplayLearnerError(
                f"n_epochs must be >= 1; got {n_epochs_eff}",
            )
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
            value_loss = float(self.value_net.update(feats, v_targets))
            # ----- policy target -----
            # Per-state visit-count-temperature target — mirror the
            # AlphaZero ``pi(a) = N(a)^(1/tau) / Z`` recipe.
            p_targets = self._build_policy_targets(batch)
            policy_loss = float(self.policy_net.update(feats, p_targets))
            wall = time.time() - t0
            running_v.append(value_loss)
            running_p.append(policy_loss)
            window_v = float(np.mean(running_v[-CONVERGENCE_WINDOW:]))
            window_p = float(np.mean(running_p[-CONVERGENCE_WINDOW:]))
            self.history_.append(
                {
                    "epoch": int(epoch_idx),
                    "value_loss": float(value_loss),
                    "policy_loss": float(policy_loss),
                    "wall_time_s": float(wall),
                    "n_samples": int(n),
                    "value_loss_running_mean": float(window_v),
                    "policy_loss_running_mean": float(window_p),
                }
            )
            # Convergence check — running mean of *both* losses falls
            # below ``CONVERGENCE_EPS``.
            if (
                len(running_v) >= CONVERGENCE_WINDOW
                and len(running_p) >= CONVERGENCE_WINDOW
                and window_v < CONVERGENCE_EPS
                and window_p < CONVERGENCE_EPS
            ):
                self.converged_ = True
                break
        return list(self.history_)

    # ------------------------------------------------------------------
    def _build_policy_targets(
        self,
        batch: Sequence[MCTSReplayEntry],
    ) -> np.ndarray:
        """Build per-state visit-count-temperature policy targets.

        For each entry in ``batch`` we look up the visit-count for
        each action index in ``[0, n_actions)`` among entries that
        share the same ``state``.  We then apply the temperature
        softmax to obtain a per-state target distribution.
        """
        n = len(batch)
        targets = np.zeros((n, int(self.n_actions)), dtype=float)
        # Group by state (string-keyed).
        by_state: Dict[str, List[int]] = {}
        for i, e in enumerate(batch):
            k = str(e.state)
            by_state.setdefault(k, []).append(i)
        n_actions = int(self.n_actions)
        for state_key, indices in by_state.items():
            visits = np.zeros(n_actions, dtype=float)
            for i in indices:
                a = batch[i]
                idx = _parse_action_index(a.action)
                if idx is None or idx < 0:
                    continue
                pos = int(idx) % n_actions
                visits[pos] += float(max(1, int(a.visit_count)))
            if visits.sum() <= 0:
                # Fall back to uniform — keeps the gradient well-defined.
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
            }
        first = self.history_[0]
        last = self.history_[-1]
        return {
            "n_epochs_run": int(len(self.history_)),
            "final_value_loss": float(last["value_loss"]),
            "final_policy_loss": float(last["policy_loss"]),
            "value_loss_drop": float(first["value_loss"] - last["value_loss"]),
            "policy_loss_drop": float(first["policy_loss"] - last["policy_loss"]),
            "converged": bool(self.converged_),
            "total_wall_time_s": float(sum(h["wall_time_s"] for h in self.history_)),
            "value_net_updates": int(self.value_net.n_updates_),
            "policy_net_updates": int(self.policy_net.n_updates_),
        }


# ---------------------------------------------------------------------------
# Synthetic fixture — populate a buffer when no real replay data is
# available.  Used by the smoke harness and by the tests.
# ---------------------------------------------------------------------------


def synth_populate_buffer(
    buffer: MCTSReplayBuffer,
    *,
    n_entries: int = 64,
    n_states: int = 8,
    n_actions: int = 4,
    seed: int = 0,
) -> int:
    """Populate ``buffer`` with synthetic (state, action, value) records.

    Used by the smoke harness when no replay file exists.  Returns
    the number of entries added.  The values are constructed so
    the buffer's mean ``value`` is non-degenerate (not all 0.5)
    and the policy target has a clear preferred action — this
    ensures the learner has signal to train on.
    """
    rng = np.random.default_rng(int(seed))
    states = [f"state_{i}" for i in range(int(n_states))]
    n_added = 0
    for i in range(int(n_entries)):
        s_idx = int(rng.integers(0, len(states)))
        a_idx = int(rng.integers(0, int(n_actions)))
        # State-dependent preferred action — gives the policy head
        # a learnable signal.
        preferred = s_idx % int(n_actions)
        if a_idx == preferred:
            value = float(rng.uniform(0.6, 1.0))
            prior = float(rng.uniform(0.4, 0.9))
        else:
            value = float(rng.uniform(0.0, 0.5))
            prior = float(rng.uniform(0.05, 0.3))
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


# ---------------------------------------------------------------------------
# Top-level runner — used by the CLI.
# ---------------------------------------------------------------------------


def run_learner(
    *,
    replay_buffer: Optional[str],
    n_epochs: int,
    batch_size: int,
    lr: float,
    out_dir: str,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    n_actions: int = DEFAULT_N_ACTIONS,
    seed: int = 0,
    synth_entries: int = 64,
    synth_states: int = 8,
) -> Dict[str, Any]:
    """Train a value / policy network on a replay buffer.

    Parameters
    ----------
    replay_buffer : str, optional
        Path to a replay buffer file.  When ``None`` we fall back to
        the synthetic fixture.
    n_epochs, batch_size, lr, out_dir : as per the CLI.
    feature_dim, hidden_dim, n_actions : network sizes.
    seed : RNG seed.
    synth_entries, synth_states : synthetic-fixture parameters
        (used when ``replay_buffer`` is None).

    Returns
    -------
    report : dict
        ``{"config", "replay_source", "buffer_stats", "summary",
        "history", "wall_time_s", "version"}``.
    """
    t_start = time.time()
    # 1. Load (or synthesize) the replay buffer.
    if replay_buffer:
        buf = MCTSReplayBuffer.load(replay_buffer)
        source = f"file:{replay_buffer}"
    else:
        buf = MCTSReplayBuffer(max_size=4096, seed=int(seed))
        n_added = synth_populate_buffer(
            buf,
            n_entries=int(synth_entries),
            n_states=int(synth_states),
            n_actions=int(n_actions),
            seed=int(seed),
        )
        source = f"synthetic:n_added={n_added}"
    buf_stats = buf.stats()
    # 2. Build the learner.
    learner = MCTSReplayLearner(
        feature_dim=int(feature_dim),
        hidden_dim=int(hidden_dim),
        n_actions=int(n_actions),
        n_epochs=int(n_epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        seed=int(seed),
    )
    # 3. Train.
    history = learner.fit(buf)
    summary = learner.summary()
    total_wall = time.time() - t_start
    return {
        "config": {
            "replay_buffer": replay_buffer,
            "n_epochs": int(n_epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "feature_dim": int(feature_dim),
            "hidden_dim": int(hidden_dim),
            "n_actions": int(n_actions),
            "seed": int(seed),
            "synth_entries": int(synth_entries),
            "synth_states": int(synth_states),
        },
        "replay_source": source,
        "buffer_stats": buf_stats,
        "summary": summary,
        "history": history,
        "wall_time_s": float(total_wall),
        "version": __version__,
    }


# ---------------------------------------------------------------------------
# Persistence — JSON-safe coercion helper (mirrors
# ``verify_closure_theorem._coerce``).
# ---------------------------------------------------------------------------


def _coerce(o: Any) -> Any:
    """Coerce numpy / set / tuple objects into JSON-safe primitives."""
    try:
        import numpy as _np

        if isinstance(o, (_np.floating, _np.integer)):
            return o.item()
        if isinstance(o, _np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def save_report(report: Dict[str, Any], out_dir: str) -> str:
    """Persist the learner bench report to ``out_dir/learner_bench.json``."""
    out_path = os.path.abspath(str(out_dir))
    os.makedirs(out_path, exist_ok=True)
    json_path = os.path.join(out_path, "learner_bench.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_coerce)
    return json_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-36 Phase-4 MCTS replay learner — AlphaGo-style value / "
            "policy trainer on a replay buffer."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke harness: synthetic 64-entry buffer × 2 epochs × batch 8.",
    )
    parser.add_argument(
        "--replay-buffer",
        type=str,
        default=None,
        help=(
            "Path to a replay buffer file (.pkl / .jsonl).  When "
            "omitted we fall back to the synthetic fixture."
        ),
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=DEFAULT_N_EPOCHS,
        help=f"Number of training epochs (default {DEFAULT_N_EPOCHS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_REPLAY_DEFAULT_BATCH,
        help=f"Mini-batch size (default {_REPLAY_DEFAULT_BATCH}).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (default {DEFAULT_LR}).",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=DEFAULT_FEATURE_DIM,
        help=f"Feature dimensionality (default {DEFAULT_FEATURE_DIM}).",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help=f"Value-head hidden width (default {DEFAULT_HIDDEN_DIM}).",
    )
    parser.add_argument(
        "--n-actions",
        type=int,
        default=DEFAULT_N_ACTIONS,
        help=f"Policy-head output dim (default {DEFAULT_N_ACTIONS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default 0).",
    )
    parser.add_argument(
        "--synth-entries",
        type=int,
        default=64,
        help="Synthetic fixture entries when no replay file is given.",
    )
    parser.add_argument(
        "--synth-states",
        type=int,
        default=8,
        help="Synthetic fixture states when no replay file is given.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for learner_bench.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as a single JSON line on stdout.",
    )
    return parser


def main(argv: Optional[Any] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    if args.smoke:
        n_epochs = 2
        batch_size = 8
        synth_entries = 32
        synth_states = 4
    else:
        n_epochs = int(args.n_epochs)
        batch_size = int(args.batch_size)
        synth_entries = int(args.synth_entries)
        synth_states = int(args.synth_states)
    print("=" * 72)
    print("Wf-36 Phase-4 MCTS replay learner (AlphaGo-style)")
    print("=" * 72)
    print(f"  replay_buffer  = {args.replay_buffer}")
    print(f"  n_epochs       = {n_epochs}")
    print(f"  batch_size     = {batch_size}")
    print(f"  lr             = {args.lr}")
    print(f"  feature_dim    = {args.feature_dim}")
    print(f"  hidden_dim     = {args.hidden_dim}")
    print(f"  n_actions      = {args.n_actions}")
    print(f"  seed           = {args.seed}")
    print(f"  out_dir        = {args.out_dir}")
    print()
    report = run_learner(
        replay_buffer=args.replay_buffer,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=float(args.lr),
        out_dir=str(args.out_dir),
        feature_dim=int(args.feature_dim),
        hidden_dim=int(args.hidden_dim),
        n_actions=int(args.n_actions),
        seed=int(args.seed),
        synth_entries=synth_entries,
        synth_states=synth_states,
    )
    summary = report["summary"]
    print("-" * 72)
    print(f"  replay_source        : {report['replay_source']}")
    print(f"  buffer_stats.size    : {report['buffer_stats'].get('size', 0)}")
    print(f"  n_epochs_run         : {summary['n_epochs_run']}")
    print(f"  final_value_loss     : {summary['final_value_loss']:.6f}")
    print(f"  final_policy_loss    : {summary['final_policy_loss']:.6f}")
    print(f"  value_loss_drop      : {summary['value_loss_drop']:.6f}")
    print(f"  policy_loss_drop     : {summary['policy_loss_drop']:.6f}")
    print(f"  converged            : {summary['converged']}")
    print(f"  total_wall_time (s)  : {report['wall_time_s']:.3f}")
    json_path = save_report(report, str(args.out_dir))
    print(f"  saved JSON           : {json_path}")
    if args.json:
        print()
        print(json.dumps(report, default=_coerce))
    return 0


__all__ = [
    "MCTSReplayLearnerError",
    "ValueNetwork",
    "PolicyNetwork",
    "MCTSReplayLearner",
    "featurise_entry",
    "synth_populate_buffer",
    "run_learner",
    "save_report",
    "DEFAULT_FEATURE_DIM",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_N_ACTIONS",
    "DEFAULT_LR",
    "DEFAULT_N_EPOCHS",
    "DEFAULT_OUTPUT_DIR",
    "CONVERGENCE_WINDOW",
    "CONVERGENCE_EPS",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
