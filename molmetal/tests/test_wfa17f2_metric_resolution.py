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
"""Wf-A17 F2 — LOMO metric-resolution guard.

The Rh fold has 108 real rows and ``--rh-loso-validation`` halves them, so
n_test=54 with 17 positives.  A *random* scorer scores 0.500 +/- 0.085
there, which is wider than every ablation delta ever measured against Rh.
These tests pin that resolution so a future ablation cannot silently be read
as a ranking.
"""

import math

import pytest

from molmetal.scripts.train_metal_lomo_wasserstein import (
    _aggregate,
    auc_noise_floor,
)


def test_noise_floor_is_the_spread_around_one_half():
    """auc_noise_floor returns the SD of a random scorer, which centres at 0.5.

    On a balanced 54-row fold that SD is ~0.079 — the spread is what a claimed
    delta has to beat, not the 0.5 centre.
    """
    assert auc_noise_floor(54, 27) == pytest.approx(0.0785, abs=0.005)


def test_rh_loso_fold_cannot_resolve_ablation_deltas():
    """n_test=54 gives sd ~0.085; the classic 'fix' deltas are ~0.01-0.05."""
    noise = auc_noise_floor(54, 17)
    assert noise > 0.08
    # Every delta F1/F2 ablations reported as a win is far inside this band.
    assert max(0.0016, 0.05, 0.15) < 2 * noise


def test_large_folds_are_resolvable():
    """Ru/Ir are large enough that a real effect is visible."""
    assert auc_noise_floor(3849, 1097) < 0.015
    assert auc_noise_floor(1250, 568) < 0.02


def test_noise_floor_nan_on_degenerate_fold():
    assert math.isnan(auc_noise_floor(10, 0))
    assert math.isnan(auc_noise_floor(10, 10))
    assert math.isnan(auc_noise_floor(1, 1))


def test_aggregate_reports_resolution_per_fold():
    per_run = [
        {"fold": "Rh", "test_auc": 0.34, "n_test": 54, "pos_rate_test": 0.3148},
        {"fold": "Ru", "test_auc": 0.58, "n_test": 3849, "pos_rate_test": 0.285},
    ]
    agg = _aggregate(per_run, n_seeds=3)
    res = agg["per_fold_resolution"]
    assert res["Rh"]["n_test"] == 54
    assert res["Rh"]["n_pos"] == 17
    assert res["Ru"]["auc_resolution"] < res["Rh"]["auc_resolution"]
    # worst fold gates how big a claim has to be to mean anything
    assert agg["worst_fold_resolution"] == pytest.approx(res["Rh"]["auc_resolution"])
    # SE of a 3-seed mean shrinks the per-run band by sqrt(n_seeds)
    assert res["Rh"]["auc_resolution_se"] == pytest.approx(
        res["Rh"]["auc_resolution"] / math.sqrt(3), rel=1e-6
    )


def test_aggregate_is_backward_compatible():
    """Default n_seeds keeps the original call signature working."""
    agg = _aggregate([{"fold": "Rh", "test_auc": 0.5}])
    assert agg["test_auc_mean"] == 0.5
    assert "per_fold_resolution" in agg