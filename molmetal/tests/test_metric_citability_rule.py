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

"""Tests for the paper-citability rule (docs/METRIC_CITABILITY_RULE.md).

CPU-only, no I/O. Each test is a minimal payload that fails exactly one clause,
so a failure names the clause rather than "the rule broke".

The rule decides two things:
  quotable       -- is this a measurement of this metric at all?
  headline_grade -- is the protocol disclosed well enough for a results table?

Coverage:
  1. A fully declared multi-run payload is table-grade.
  2. Each non-quotable condition fails quotable (and therefore headline_grade).
  3. Each of the five disclosure facts fails headline_grade alone.
  4. A std attached to a single run is flagged; a per-sample std is not.
  5. The file's own "not a claim" statuses stay quotable but never table-grade.
  6. headline_caveat is carried through untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in str(sys.path):
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.registry import assess_citability  # noqa: E402


def _good(**over):
    """A payload that passes every clause; override one key per test."""
    payload = {
        "metric": "demo",
        "status": "MEASURED",
        "value": 0.5,
        "n_seeds": 5,
        "dataset": "tmQM",
        "split": "5-fold CV",
    }
    payload.update(over)
    return payload


def _assess(payload, tracked=None, source_rel=None):
    return assess_citability(payload, payload.get("value"), source_rel, tracked)


def test_fully_declared_payload_is_table_grade():
    r = _assess(_good())
    assert r["quotable"] is True
    assert r["headline_grade"] is True
    assert r["not_quotable"] == []
    assert r["not_headline"] == []


# --- quotable -------------------------------------------------------------


def test_missing_status_is_not_quotable():
    r = _assess(_good(status=None))
    assert r["quotable"] is False
    assert any("no status" in m for m in r["not_quotable"])


def test_projected_status_is_not_quotable():
    r = _assess(_good(status="PROJECTED_ARITHMETIC"))
    assert r["quotable"] is False
    assert any("PROJECTED_ARITHMETIC" in m for m in r["not_quotable"])


def test_missing_value_is_not_quotable():
    r = _assess(_good(value=None))
    assert r["quotable"] is False
    assert "no scalar headline" in r["not_quotable"]


def test_declared_no_headline_reason_replaces_generic_gap():
    r = _assess(_good(value=None, no_headline_reason="probe scale, no scalar by design"))
    assert r["quotable"] is False
    assert r["not_quotable"] == [
        "no scalar headline by design: probe scale, no scalar by design"
    ]


def test_positive_control_is_not_quotable():
    r = _assess(_good(cisplatin_reference_mean_is_control=True))
    assert r["quotable"] is False
    assert any("positive control" in m for m in r["not_quotable"])


def test_half_synthetic_value_is_not_quotable():
    r = _assess(_good(value_is_half_synthetic=True))
    assert r["quotable"] is False
    assert any("mixes real and synthetic" in m for m in r["not_quotable"])


def test_untracked_source_is_not_quotable():
    r = _assess(_good(), source_rel="molmetal/reports/x/y.json", tracked=set())
    assert r["quotable"] is False
    assert any("not tracked by git" in m for m in r["not_quotable"])


def test_tracked_source_passes():
    r = _assess(
        _good(),
        source_rel="metrics/by_metric/demo.json",
        tracked={"metrics/by_metric/demo.json"},
    )
    assert r["quotable"] is True


# --- headline_grade -------------------------------------------------------


def test_undeclared_run_count_blocks_headline():
    r = _assess(_good(n_seeds=None))
    assert r["quotable"] is True
    assert r["headline_grade"] is False
    assert any("run count not declared" in m for m in r["not_headline"])


def test_single_run_blocks_headline():
    r = _assess(_good(n_seeds=1))
    assert r["quotable"] is True
    assert r["headline_grade"] is False
    assert any("single run" in m for m in r["not_headline"])


def test_seeds_list_counts_as_run_count():
    r = _assess(_good(n_seeds=None, seeds=[0, 1, 2]))
    assert r["headline_grade"] is True


def test_aggregate_run_count_is_read():
    r = _assess(_good(n_seeds=None, aggregate={"n_runs_measured": 25}))
    assert r["headline_grade"] is True


def test_undeclared_dataset_blocks_headline():
    r = _assess(_good(dataset=None))
    assert r["quotable"] is True
    assert r["headline_grade"] is False
    assert "evaluation dataset not declared" in r["not_headline"]


def test_data_source_counts_as_dataset():
    r = _assess(_good(dataset=None, data_source="MetalCytoToxDB.csv"))
    assert r["headline_grade"] is True


def test_undeclared_split_blocks_headline():
    r = _assess(_good(split=None))
    assert r["quotable"] is True
    assert r["headline_grade"] is False
    assert "split not declared" in r["not_headline"]


def test_protocol_mentioning_folds_counts_as_split():
    r = _assess(_good(split=None, protocol="5-fold CV, seed 0"))
    assert r["headline_grade"] is True


def test_protocol_without_fold_or_split_does_not_count():
    r = _assess(_good(split=None, protocol="deterministic pass, seeds=1"))
    assert r["headline_grade"] is False


# --- std semantics --------------------------------------------------------


def test_run_level_std_on_a_single_run_is_flagged():
    r = _assess(_good(n_seeds=1, value_std=0.0))
    assert r["headline_grade"] is False
    assert any("std reported for a single run" in m for m in r["not_headline"])


def test_per_sample_std_is_not_an_error_bar():
    """m6_kappa: std across 449 molecules, not across runs. Must not be flagged
    as a std-for-one-run, or the rule reports a defect that is not there."""
    r = _assess(_good(n_seeds=1, std=0.1588, n_samples=449))
    assert not any("std reported" in m for m in r["not_headline"])
    # Still not table-grade: there really is only one pass.
    assert r["headline_grade"] is False


def test_seed_std_name_is_run_level():
    r = _assess(_good(n_seeds=1, per_seed_std=0.01))
    assert any("std reported" in m for m in r["not_headline"])


# --- statuses that decline the claim --------------------------------------


def test_not_a_claim_status_is_quotable_but_not_headline():
    r = _assess(_good(status="MEASURED_NOT_A_CLAIM"))
    assert r["quotable"] is True
    assert r["headline_grade"] is False
    assert any("declines the claim" in m for m in r["not_headline"])


def test_not_a_quality_metric_status_is_quotable_but_not_headline():
    r = _assess(_good(status="MEASURED_BUT_NOT_A_QUALITY_METRIC"))
    assert r["quotable"] is True
    assert r["headline_grade"] is False


# --- caveat passthrough ---------------------------------------------------


def test_headline_caveat_is_carried_through():
    r = _assess(_good(headline_caveat="cohort aliased"))
    assert r["headline_caveat"] == "cohort aliased"


def test_absent_caveat_is_none():
    assert _assess(_good())["headline_caveat"] is None


def test_blank_caveat_is_none():
    assert _assess(_good(headline_caveat="   "))["headline_caveat"] is None
