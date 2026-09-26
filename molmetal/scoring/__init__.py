"""molmetal.scoring — unified real-tool scoring router.

Wf-43 Phase 1 + Phase 2. Replaces every mock-based metric with a
forced route to the real, locally-deployed scoring tool. NO mock
fallback is permitted: ``score_any(metric, ...)`` either calls the
genuine evaluator or raises.

Public surface:
    * :func:`score_any`         — main router
    * :func:`available_metrics` — list metric names registered
    * :func:`tool_for_metric`   — which real tool backs each metric
    * :class:`AdapterUnavailable` — raised when a metric has no live tool
    * :class:`PolicyEntry`      — per-metric routing-policy dataclass (Wf-43 Phase 2)
    * :data:`REGISTRY`          — metric → adapter mapping dict
    * :data:`ROUTING_POLICY`    — metric → policy-entry mapping dict (Wf-43 Phase 2)

All real adapters live in :mod:`molmetal.molmetal_lam.sbdd_env`. This
module is a thin, declarative router — it does NOT import heavy ML
backends at import time; adapters are imported lazily on first call so
a CI environment without all backends still works for tests that hit
only a subset of metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molmetal.scoring.score_any import (
    REGISTRY,
    ROUTING_POLICY,
    AdapterUnavailable,
    PolicyEntry,
    available_metrics,
    score_any,
    tool_for_metric,
)

__all__ = [
    "REGISTRY",
    "ROUTING_POLICY",
    "AdapterUnavailable",
    "PolicyEntry",
    "available_metrics",
    "score_any",
    "tool_for_metric",
]

if TYPE_CHECKING:  # pragma: no cover - import-only type hints
    from typing import Dict, List, Optional
