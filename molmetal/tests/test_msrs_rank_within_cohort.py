#!/usr/bin/env python3
"""Tests for the within-cohort ranker (C3).

The label construction is the whole experiment: if relevance is not the
within-cohort rank, ``lambdarank`` optimises the wrong thing and the arm looks
like a level model with extra steps.  Group sizes must also agree with the
sorted key order, or LightGBM silently mis-slices the queries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molmetal.scripts.msrs_rank_within_cohort import group_sizes, relevance


def _frame(keys, time_h=None):
    time_h = [10.0] * len(keys) if time_h is None else list(time_h)
    return pd.DataFrame(
        {
            "ligand_canonical": [k.split("|")[0] for k in keys],
            "cell_line_key": ["A"] * len(keys),
            "doi": [k.split("|")[1] for k in keys],
            "time_h": time_h,
        }
    )


def test_relevance_is_the_within_cohort_rank():
    frame = _frame(["L1|D1", "L1|D1", "L2|D1", "L2|D1"])
    y = np.array([1.0, 3.0, 2.0, 0.5])
    rel = relevance(frame, y)
    # cohort L1: 3.0 ranks highest -> 1; cohort L2: 2.0 -> 1
    assert rel.tolist() == [0, 1, 1, 0]


def test_relevance_is_zero_for_singleton_cohorts():
    frame = _frame(["L1|D1", "L2|D1"])
    y = np.array([1.0, 2.0])
    assert relevance(frame, y).tolist() == [0, 0]


def test_relevance_handles_three_metal_cohorts():
    frame = _frame(["L1|D1"] * 3)
    y = np.array([2.0, 1.0, 3.0])
    assert relevance(frame, y).tolist() == [1, 0, 2]


def test_relevance_separates_cohorts_by_doi():
    frame = _frame(["L1|D1", "L1|D1", "L1|D2", "L1|D2"])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    rel = relevance(frame, y)
    assert rel.tolist() == [0, 1, 0, 1]


def test_group_sizes_sum_to_rows_and_match_sorted_keys():
    keys = np.array(["b", "a", "a", "c", "c", "c"])
    sizes = group_sizes(keys)
    assert sizes.sum() == len(keys)
    assert sizes.tolist() == [2, 1, 3]  # sorted key order a, b, c


def test_group_sizes_of_one_cohort_is_the_whole_frame():
    assert group_sizes(np.array(["x"] * 5)).tolist() == [5]


def test_relevance_rejects_nothing_but_leaves_nan_ordering_stable():
    frame = _frame(["L1|D1", "L1|D1"])
    y = np.array([np.nan, 1.0])
    rel = relevance(frame, y)
    # NaN sorts last under argsort, so the finite value takes rank 0
    assert sorted(rel.tolist()) == [0, 1]
