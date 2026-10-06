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
"""Unit tests for the variance-scaled c_puct SCAFFOLD (Wf-R21-B1)."""

from __future__ import annotations

import math

from molmetal.molmetal_lam.search_alg.variance_cpuct import (
    cpuct_from_node,
    effective_cpuct,
)


def test_zero_variance_returns_base():
    assert effective_cpuct(1.4, 0.0) == 1.4


def test_small_variance_scales_up():
    # 1.4 * (1 + 0.8 * sqrt(0.04)) = 1.4 * (1 + 0.8 * 0.2) = 1.624
    got = effective_cpuct(1.4, 0.04)
    assert math.isclose(got, 1.624, rel_tol=1e-9)


def test_negative_variance_returns_base():
    assert effective_cpuct(1.4, -1.0) == 1.4


def test_huge_variance_clamps_at_2x_base():
    assert effective_cpuct(1.4, 100.0) == 2.8


def test_cpuct_from_node_reads_attribute():
    class _Node:
        leaf_value_var = 0.04

    class _NoVarNode:
        pass

    assert math.isclose(cpuct_from_node(_Node()), 1.624, rel_tol=1e-9)
    assert cpuct_from_node(_NoVarNode()) == 1.4
