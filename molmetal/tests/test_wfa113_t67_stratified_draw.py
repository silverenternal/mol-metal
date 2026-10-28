# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T67 — the tmQM cohort draw must be metal-proportional.

THE DEFECT
----------
``load_tmqm_clouds`` streamed the XYZ shards in file order and reservoir-sampled
the first ``_CAP = 5 * n_clouds`` candidates.  The shards are METAL-BLOCKED:
MEASURED on the shipped corpus, the first 512 filtered candidates are 100% Re,
the first 4096 contain no Pt at all, and Pt first appears around candidate
4,200.  ``DEFAULT_TRAIN_CLOUDS`` is 256, so every tmQM run that did not declare
``n_train`` trained on a 256-cloud cohort that was 100% Re, and the 400-cloud
smoke cohort was Ru/Re/Os with zero Pt — while Pt is the headline metal of the
project and 24.5% of the pool.

These tests pin the properties of the replacement draw, on synthetic metadata,
so they run in milliseconds and do not need the 33k-complex corpus.
"""

from __future__ import annotations

import random

import pytest

from molmetal.data.tmqm_clouds import _stratified_code_draw

METALS = ["Pt", "Ru", "Ir", "Rh", "Os", "Re"]
# Pool shares roughly matching the shipped corpus (29307 six-metal complexes).
SHARES = {"Ru": 0.29, "Pt": 0.255, "Ir": 0.157, "Rh": 0.133, "Re": 0.125, "Os": 0.039}


def _meta_and_codes(per_metal: int = 4000):
    meta = {}
    codes = []
    for m, share in SHARES.items():
        k = max(1, int(round(per_metal * share)))
        for i in range(k):
            code = f"{m}{i:05d}"
            meta[code] = {"metal": m}
            codes.append(code)
    return meta, codes


def test_quota_sums_to_the_request():
    meta, codes = _meta_and_codes()
    for n in (12, 64, 192, 256, 400, 1000):
        drawn, quota, pool = _stratified_code_draw(meta, codes, n, seed=0)
        assert sum(quota.values()) == n, (n, quota)
        # Oversampling is bounded: slack is 25% plus a small floor.
        assert len(drawn) >= n
        assert len(drawn) <= n + 4 * len(METALS) + int(0.25 * n) + len(METALS)


def test_every_metal_is_present_at_small_n():
    """The zero-Pt finding, stated as an invariant."""
    meta, codes = _meta_and_codes()
    for n in (6, 12, 64, 256, 400):
        drawn, quota, _pool = _stratified_code_draw(meta, codes, n, seed=0)
        assert set(quota) == set(SHARES)
        assert all(v >= 1 for v in quota.values()), (n, quota)
        metals_drawn = {meta[c]["metal"] for c in drawn}
        assert metals_drawn == set(SHARES), (n, metals_drawn)


def test_quota_tracks_the_pool_share():
    meta, codes = _meta_and_codes()
    _drawn, quota, _pool = _stratified_code_draw(meta, codes, 1000, seed=0)
    for m, share in SHARES.items():
        assert abs(quota[m] / 1000.0 - share) <= 0.01, (m, quota[m], share)


def test_draw_is_deterministic_for_a_seed_and_changes_across_seeds():
    meta, codes = _meta_and_codes()
    a, _qa, _pa = _stratified_code_draw(meta, codes, 256, seed=7)
    b, _qb, _pb = _stratified_code_draw(meta, codes, 256, seed=7)
    c, _qc, _pc = _stratified_code_draw(meta, codes, 256, seed=8)
    assert a == b, "same seed must produce the same draw"
    assert a != c, "a different seed must produce a different draw"


def test_draw_does_not_depend_on_input_order():
    """``load_tmqm_clouds`` hands in a set; the draw sorts internally."""
    meta, codes = _meta_and_codes()
    shuffled = list(codes)
    random.Random(3).shuffle(shuffled)
    a, _qa, _pa = _stratified_code_draw(meta, codes, 128, seed=11)
    b, _qb, _pb = _stratified_code_draw(meta, shuffled, 128, seed=11)
    assert a == b


def test_no_code_is_drawn_twice():
    meta, codes = _meta_and_codes()
    drawn, _quota, _pool = _stratified_code_draw(meta, codes, 400, seed=0)
    assert len(drawn) == len(set(drawn))


def test_min_per_metal_is_a_floor():
    meta, codes = _meta_and_codes()
    _drawn, quota, _pool = _stratified_code_draw(
        meta, codes, 60, seed=0, min_per_metal=4
    )
    for m in SHARES:
        assert quota[m] >= 4, (m, quota)


def test_pool_smaller_than_request_is_not_over_drawn():
    meta, codes = _meta_and_codes(per_metal=4)
    drawn, quota, pool = _stratified_code_draw(meta, codes, 10_000, seed=0)
    for m in SHARES:
        assert quota[m] <= pool[m], (m, quota[m], pool[m])
    assert len(drawn) <= len(codes)


def test_a_metal_absent_from_the_pool_is_simply_not_quotaed():
    meta, codes = _meta_and_codes()
    codes = [c for c in codes if meta[c]["metal"] != "Pt"]
    meta = {c: v for c, v in meta.items() if v["metal"] != "Pt"}
    _drawn, quota, pool = _stratified_code_draw(meta, codes, 120, seed=0)
    assert "Pt" not in quota
    assert "Pt" not in pool
    assert sum(quota.values()) == 120


def test_empty_pool_returns_empty():
    assert _stratified_code_draw({}, [], 10, seed=0) == ([], {}, {})
