# Copyright 2026 The Mol-Metal Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Variance-scaled ``c_puct`` (R20 FIX 2) — SCAFFOLD module.

Motivation
----------
``proof_search.py`` currently holds ``c_puct`` FIXED at ``1.4``
(:data:`c_puct`, default declared at the ``ProofSearcher`` dataclass) and uses
it only to scale the PUCT **exploration** term
(``w = float(self.c_puct)`` in the selection loop).  The per-node leaf-value
variance buffer (``leaf_value_var`` / ``LEAF_VALUE_VAR``) is computed and
reported (``PUCT_EXPLOIT_RATIO_VAR``) but never feeds back into the search
parameter, so high-uncertainty subtrees are not explored more aggressively
than low-uncertainty ones.

Reference pattern
-----------------
KataGo scales its exploration constant by the running standard deviation of
the root value estimate, i.e. it multiplies by a term proportional to
``sigma / (sigma + c)`` -- shrinkage toward the prior rather than a raw
variance blow-up.  KataGo's methods write-up projects roughly **+75 Elo** for
variance-/uncertainty-scaled exploration over a fixed constant.

Reference: https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md
(KataGo methods document, "Exploration" section; KataGo v1.x).

This module is a **SCAFFOLD**: it provides the pure function and the
node-adapter helper.  It is NOT yet wired into ``ProofSearcher`` -- doing so is
the follow-up wave (wire-in must satisfy AGENTS.md §5.7 smoke gating).

Formula
-------
``effective_cpuct = base * (1 + alpha * sqrt(variance))``

with ``variance`` the per-node ``LEAF_VALUE_VAR`` and ``alpha`` the shrinkage
toward uniform exploration (KataGo-style default ``0.8``).
"""

from __future__ import annotations

import math

__all__ = ["effective_cpuct", "cpuct_from_node", "MAX_VARIANCE"]

#: Safety clamp -- variances above this are treated as the clamp threshold.
MAX_VARIANCE = 10.0


def effective_cpuct(base: float, variance: float, alpha: float = 0.8) -> float:
    """Variance-scaled c_puct: KataGo-style sqrt(sigma^2) scaling with shrinkage.

    formula: ``effective_cpuct = base * (1 + alpha * sqrt(variance))``

    where ``variance`` is the per-node ``LEAF_VALUE_VAR`` (already buffered in
    ``proof_search.py``) and ``alpha`` is the shrinkage toward uniform
    exploration (KataGo default ``0.8``).

    Edge cases (all returning a safe, monotone value):

    * ``variance < 0``  (numerical noise / bad buffer) -> ``base``
    * ``variance == 0`` (single visit, no estimate yet) -> ``base``
    * ``variance > MAX_VARIANCE`` (10) -> clamped at ``2 * base``

    The clamp keeps the exploration term bounded at 2x the fixed constant, so
    a pathological reward head cannot dominate the PUCT selection.
    """
    base = float(base)
    variance = float(variance)
    if not math.isfinite(variance) or variance <= 0.0:
        return base
    if variance > MAX_VARIANCE:
        return 2.0 * base
    return base * (1.0 + float(alpha) * math.sqrt(variance))


def cpuct_from_node(node, base: float = 1.4, alpha: float = 0.8) -> float:
    """Variance-scaled c_puct for a ``proof_search`` node.

    Reads the node's ``leaf_value_var`` attribute when present, falling back to
    ``0.0`` (i.e. the fixed ``base``) for nodes that carry no variance buffer
    yet.  Uses ``getattr`` deliberately -- older node dataclasses and test
    doubles do not all define the field.
    """
    variance = getattr(node, "leaf_value_var", 0.0)
    try:
        variance = float(variance)
    except (TypeError, ValueError):
        variance = 0.0
    return effective_cpuct(base, variance, alpha)
