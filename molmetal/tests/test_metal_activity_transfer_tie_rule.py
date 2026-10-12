"""Regression: the duplicate-collapse tie rule is a real, selectable choice.

``collapse_duplicates`` (used by the Paper-A transfer probe) aggregates one label
per unique ligand by majority.  An exact tie (``sum * 2 == n``) is the load-
bearing case: the default ``tie_rule="active"`` resolves it to ACTIVE
(``sum * 2 >= n``) -- the ANTI-conservative direction for an activity claim --
while ``tie_rule="inactive"`` resolves it to INACTIVE (``sum * 2 > n``).

This matters because 80 ligand groups tie on MetalCytoToxDB and the direction of
the "Ru beats Krasnov" statement flips with the rule (see
``metrics/by_metric/metal_activity_transfer_tie_sensitivity.json``).  These tests
pin the predicate on synthetic ties so the two rules can never silently become
the same function again.  CPU-only; no data files.
"""

from __future__ import annotations

import numpy as np
import pytest

from molmetal.scripts.metal_activity_transfer_probe import collapse_duplicates


def _labels(smiles, y, tie_rule):
    uniq, labels, n_raw = collapse_duplicates(
        np.asarray(smiles, dtype=object), np.asarray(y, dtype=int), tie_rule=tie_rule
    )
    return dict(zip(uniq, labels)), n_raw


def test_exact_tie_resolves_active_under_default_rule():
    """n=2, sum=1 -> active (1*2 >= 2). The default is the anti-conservative one."""
    labels, n_raw = _labels(["A", "A"], [1, 0], "active")
    assert labels == {"A": 1}
    assert n_raw == 2


def test_exact_tie_resolves_inactive_under_conservative_rule():
    """The same tie must flip to 0 when the rule is the conservative one."""
    labels, _ = _labels(["A", "A"], [1, 0], "inactive")
    assert labels == {"A": 0}


def test_even_split_of_four_is_a_tie_under_both_directions():
    """n=4, sum=2 -> active; the conservative rule keeps it 0. No rounding slip."""
    assert _labels(["A"] * 4, [1, 1, 0, 0], "active")[0] == {"A": 1}
    assert _labels(["A"] * 4, [1, 1, 0, 0], "inactive")[0] == {"A": 0}


def test_non_tie_majorities_are_rule_independent():
    """A real majority must not move: active-majority stays 1, inactive-majority 0."""
    for rule in ("active", "inactive"):
        assert _labels(["B"] * 3, [1, 1, 0], rule)[0] == {"B": 1}
        assert _labels(["C"] * 3, [1, 0, 0], rule)[0] == {"C": 0}


def test_unknown_tie_rule_raises():
    with pytest.raises(ValueError, match="tie_rule"):
        collapse_duplicates(np.asarray(["A"], dtype=object), np.asarray([1]), tie_rule="nope")
