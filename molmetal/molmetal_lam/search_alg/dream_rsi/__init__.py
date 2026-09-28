"""pi-Dream-RSI Python port — replay-safe dream-rsi search layer.

This subpackage is a pure-Python port of the TypeScript reference
implementation at
``extensions/pi-dream-rsi/engine/tree.ts`` (and friends), with the same
contract:

* :class:`DiscoveryTree` exposes the recorded tree's structure and the
  observation/leaf/legality helpers the policy reads.
* :class:`PolicyHost` is the *prefix-only* observed-tree owner (live +
  replay modes share one implementation).
* :func:`make_replay_probe` deterministically reveals the frozen world —
  it never generates candidates.
* :class:`ReplayEpisode` runs a policy against the frozen world under
  the Eq. 1 objective.
* :class:`DreamLoop` evaluates every policy candidate on every recorded
  world and selects ``argmax V`` with the ``V* >= V_0`` invariant.

The TypeScript invariants listed in the dream-rsi README are preserved
verbatim:

*   ``pi_0`` is always among the candidates and ``V* >= V_0`` is
    *asserted* — if it breaks, ``run_dream_phase`` raises and the
    iteration is reported as failed.
*   Non-deterministic replays disqualify a candidate: ``v_mean = -inf``.
*   Failed episodes are not averaged as zero — they contribute
    ``-inf`` and are excluded from the scored-set average.
*   Replay never generates candidates — it reveals recorded children.

See ``tests/test_dream_rsi_port.py`` for the contract tests.
"""

from .discovery_tree import (
    ROOT_ID,
    CellMeta,
    DiscoveryTree,
    Observation,
    TreeJSON,
    TreeNode,
    cell_id,
    is_opened,
    is_success,
)
from .replay_probe import ProbeFn, ProbeResult, RevealedNode, make_replay_probe
from .policy_host import (
    Decision,
    HostOptions,
    PolicyHost,
)
from .replay_episode import (
    EpisodeMetrics,
    ReplayEpisode,
    ReplayEpisodeOptions,
    ReplayEpisodeResult,
    RewardConfig,
    replay_value,
    trace_metrics,
)
from .dream import (
    DreamLoop,
    DreamOptions,
    DreamResult,
    PolicyCandidate,
    VersionEvaluation,
    fallback_grid,
)

__all__ = [
    "ROOT_ID",
    "CellMeta",
    "DiscoveryTree",
    "Observation",
    "TreeJSON",
    "TreeNode",
    "cell_id",
    "is_opened",
    "is_success",
    "ProbeFn",
    "ProbeResult",
    "RevealedNode",
    "make_replay_probe",
    "Decision",
    "HostOptions",
    "PolicyHost",
    "EpisodeMetrics",
    "ReplayEpisode",
    "ReplayEpisodeOptions",
    "ReplayEpisodeResult",
    "RewardConfig",
    "replay_value",
    "trace_metrics",
    "DreamLoop",
    "DreamOptions",
    "DreamResult",
    "PolicyCandidate",
    "VersionEvaluation",
    "fallback_grid",
]
