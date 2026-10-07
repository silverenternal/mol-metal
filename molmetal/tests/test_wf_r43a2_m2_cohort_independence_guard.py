# Copyright 2026 Mol-Metal contributors.
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

"""Wf-R43 m2 — R2 cohort-independence guard tests.

Reproduces the duplication the guard exists to catch, then checks that the
guard FAILs on it and PASSes once the record discloses alias status.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molmetal.scripts.wf_r43a2_m2_cohort_independence_guard import (
    KNOWN_ALIASES,
    audit_m2_cohort_independence,
)

# The exact pair of bit-identical r2_mean values recorded in
# metrics/by_metric/m2_r2_per_cohort.json before the Wf-R43-A2 fix.
DUPLICATE_R2 = -0.14401547578768395


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "m2.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _duplicated_payload() -> dict:
    """The pre-fix record: 3 rows, hilbert bit-identical to platinai."""
    return {
        "metric": "m2_r2_per_cohort",
        "per_cohort": [
            {
                "cohort": "platinai_4cell",
                "r2_mean": DUPLICATE_R2,
                "baseline_r2_mechanical": -0.12505399802714393,
                "lift_vs_baseline": -0.018961477760540024,
                "target_name": "pIC50_mean_mcf7_a549_a2780_a2780cis",
            },
            {
                "cohort": "hilbert",
                "r2_mean": DUPLICATE_R2,
                "baseline_r2_mechanical": -0.12505399802714393,
                "lift_vs_baseline": -0.018961477760540024,
                "target_name": "pIC50_mean_mcf7_a549_a2780_a2780cis",
            },
            {
                "cohort": "metalcytotox_combined",
                "r2_mean": 0.19808189300924267,
                "baseline_r2_mechanical": 0.22268113216745766,
                "lift_vs_baseline": -0.02459923915821499,
                "target_name": "activity",
            },
        ],
        "aggregate": {
            "mean_r2": -0.029983019522041743,
            "n_cohorts_tested": 3,
            "aggregate_status": "MEASURED_PARTIAL_LIFT",
        },
    }


def _fixed_payload() -> dict:
    """Post-fix record: hilbert re-split, alias disclosed, 2 independent."""
    return {
        "metric": "m2_r2_per_cohort",
        "per_cohort": [
            {
                "cohort": "platinai_4cell",
                "r2_mean": -0.14401547578768395,
                "baseline_r2_mechanical": -0.12505399802714393,
                "lift_vs_baseline": -0.018961477760540024,
                "target_name": "pIC50_mean_mcf7_a549_a2780_a2780cis",
                "cohort_is_alias_of": None,
                "independent_cohort": True,
                "cv_seed_offset": 0,
            },
            {
                "cohort": "hilbert",
                "r2_mean": -0.1470103206,
                "baseline_r2_mechanical": -0.1268,
                "lift_vs_baseline": -0.0202,
                "target_name": "pIC50_mean_mcf7_a549_a2780_a2780cis",
                "cohort_is_alias_of": "platinai_4cell",
                "independent_cohort": False,
                "cv_seed_offset": 1,
            },
            {
                "cohort": "metalcytotox_combined",
                "r2_mean": 0.19808189300924267,
                "baseline_r2_mechanical": 0.22268113216745766,
                "lift_vs_baseline": -0.02459923915821499,
                "target_name": "activity",
                "cohort_is_alias_of": None,
                "independent_cohort": True,
                "cv_seed_offset": 0,
            },
        ],
        "aggregate": {
            "mean_r2": -0.03098130111985416,
            "mean_r2_independent": 0.02703320861077936,
            "n_cohorts_tested": 3,
            "n_cohorts_independent": 2,
            "n_cohorts_aliased": 1,
            "aggregate_status": "MEASURED_PARTIAL_LIFT",
        },
    }


def test_absent_metric_is_skip(tmp_path: Path) -> None:
    r = audit_m2_cohort_independence(tmp_path / "nope.json")
    assert r["status"] == "SKIP"


def test_malformed_json_fails_not_crashes(tmp_path: Path) -> None:
    p = tmp_path / "m2.json"
    p.write_text("{not json", encoding="utf-8")
    assert audit_m2_cohort_independence(p)["status"] == "FAIL"


def test_duplicate_r2_under_distinct_names_fails(tmp_path: Path) -> None:
    r = audit_m2_cohort_independence(_write(tmp_path, _duplicated_payload()))
    assert r["status"] == "FAIL"
    assert ["platinai_4cell", "hilbert"] in r["duplicate_r2_groups"]


def test_undisclosed_alias_fails(tmp_path: Path) -> None:
    """Same duplicates, but even without the r2 clash the alias is undisclosed."""
    payload = _duplicated_payload()
    # Break the duplicate so check 2 is clean; check 1 must still FAIL.
    payload["per_cohort"][1]["r2_mean"] = -0.1470103206
    r = audit_m2_cohort_independence(_write(tmp_path, payload))
    assert r["status"] == "FAIL"
    assert r["undisclosed_aliases"] == ["hilbert"]


def test_missing_independent_count_fails(tmp_path: Path) -> None:
    payload = _fixed_payload()
    del payload["aggregate"]["n_cohorts_independent"]
    r = audit_m2_cohort_independence(_write(tmp_path, payload))
    assert r["status"] == "FAIL"
    assert "n_cohorts_independent" in r["evidence"]


def test_fixed_record_passes(tmp_path: Path) -> None:
    r = audit_m2_cohort_independence(_write(tmp_path, _fixed_payload()))
    assert r["status"] == "PASS", r["evidence"]
    assert r["n_independent_declared"] == 2


def test_empty_per_cohort_fails(tmp_path: Path) -> None:
    r = audit_m2_cohort_independence(_write(tmp_path, {"metric": "m2_r2_per_cohort"}))
    assert r["status"] == "FAIL"


def test_loader_hilbert_is_an_alias_of_platinai() -> None:
    """The root cause: load_cohort('hilbert') returns the platinai rows.

    This is the defect the whole guard exists for. If someone later gives
    hilbert a real dataset this test fails, and the alias entry in
    KNOWN_ALIASES / COHORT_ALIASES should be revisited.
    """
    np = pytest.importorskip("numpy")
    from molmetal.scripts.wf331r5_r2_w2_chemprop import load_cohort

    Xp, yp, sp, tp = load_cohort("platinai_4cell")
    Xh, yh, sh, th = load_cohort("hilbert")
    assert np.array_equal(Xp, Xh)
    assert np.array_equal(yp, yh)
    assert sp == sh
    assert tp == th
    assert KNOWN_ALIASES["hilbert"] == "platinai_4cell"


def test_seed_offset_makes_the_two_rows_differ() -> None:
    """The Wf-R43-A2 fix: hilbert must not replay platinai's folds."""
    from molmetal.scripts.wf388a1_m2_per_cohort_real import (
        COHORT_REGISTRY,
        baseline_anchor_for,
        cohort_seed_offset,
        train_eval_per_cohort,
    )

    assert cohort_seed_offset("hilbert") == 1
    assert cohort_seed_offset("platinai_4cell") == 0

    rows = {
        c: train_eval_per_cohort(cohort_data=COHORT_REGISTRY[c](), cohort=c)
        for c in ("platinai_4cell", "hilbert", "metalcytotox_combined")
    }
    p, h = rows["platinai_4cell"], rows["hilbert"]
    assert p["r2_mean"] != h["r2_mean"]
    # The old target_name dispatch made hilbert's anchor unreachable.
    assert h["baseline_r2_mechanical"] != p["baseline_r2_mechanical"]
    assert baseline_anchor_for("hilbert") == "hilbert"
    # Alias disclosure rides along on each row.
    assert p["independent_cohort"] is True
    assert h["independent_cohort"] is False
    assert h["cohort_is_alias_of"] == "platinai_4cell"


def test_aggregate_counts_independent_cohorts() -> None:
    from molmetal.scripts.wf388a1_m2_per_cohort_real import (
        COHORT_REGISTRY,
        aggregate_per_cohort_r2,
        train_eval_per_cohort,
    )

    rows = []
    for c in ("platinai_4cell", "hilbert", "metalcytotox_combined"):
        r = train_eval_per_cohort(cohort_data=COHORT_REGISTRY[c](), cohort=c)
        r["cohort"] = c
        rows.append(r)
    agg = aggregate_per_cohort_r2(rows)
    assert agg["n_cohorts_tested"] == 3
    assert agg["n_cohorts_independent"] == 2
    assert agg["n_cohorts_aliased"] == 1
    # The aliased negative row was double-counted before the fix, dragging
    # the mean negative; on independent cohorts alone it is positive.
    assert agg["mean_r2"] < 0
    assert agg["mean_r2_independent"] > 0


def test_on_disk_metric_is_flagged() -> None:
    """The shipped record still carries the pre-fix duplication."""
    root = Path(__file__).resolve().parents[2]
    path = root / "metrics" / "by_metric" / "m2_r2_per_cohort.json"
    if not path.is_file():
        pytest.skip("m2 metric record not on disk")
    r = audit_m2_cohort_independence(path)
    assert r["status"] == "FAIL"
    assert r["undisclosed_aliases"] == ["hilbert"]