# SPDX-License-Identifier: Apache-2.0
"""Same --seed must mean the same batch order."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")


def test_permutation_is_reproducible_for_the_same_seed_and_epoch():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _epoch_permutation,
    )

    a = _epoch_permutation(64, 7, 3)
    b = _epoch_permutation(64, 7, 3)
    assert a == b
    assert sorted(a) == list(range(64))


def test_permutation_changes_with_epoch_and_seed():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _epoch_permutation,
    )

    base = _epoch_permutation(64, 7, 0)
    assert _epoch_permutation(64, 7, 1) != base
    assert _epoch_permutation(64, 8, 0) != base


def test_permutation_handles_small_and_empty_cohorts():
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
        _epoch_permutation,
    )

    assert _epoch_permutation(0, 1, 0) == []
    assert _epoch_permutation(1, 1, 0) == [0]
