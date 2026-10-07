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

"""Wf-R43 m1 — tests for the Ru-AUC paper-cohort comparability guard.

CPU-only; no trainer is invoked and nothing is trained.  9 tests:

  1. the on-disk record really is cross-cohort (the m1 premise)
  2. auditing the real record FAILs and reports the cohort delta
  3. the FAIL evidence forbids the like-for-like reading
  4. the top-23 remediation cohort is CPU-constructible from the release CSV
  5. a synthetic 23-cell-line / 10-fold record PASSes
  6. 23 cell lines but an incomplete fold sweep still FAILs
  7. a non-finite (NaN) fold AUC is NOT counted as a measurement
  8. a record with no paper_protocol_measurement FAILs rather than inheriting
  9. malformed JSON FAILs loudly; an absent metric SKIPs
 10. the gate is registered on "all" (count 115) and SKIPs by default
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.wf_r43a1_m1_paper_cohort_guard import (  # noqa: E402
    EXPECTED_FOLDS,
    GATE_NAME,
    METRIC_KEY,
    OPT_IN_ENV,
    PAPER_CELL_LINES,
    PAPER_MEAN_AUC,
    audit_m1_paper_cohort,
)

REAL_METRIC = REPO_ROOT / "metrics" / "by_metric" / "metalcytotox_ru_auc.json"
RELEASE_CSV = REPO_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv"


def _write(tmp_path: Path, payload: dict, name: str = "m.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _comparable_record(cell_lines: int = PAPER_CELL_LINES, folds: int = 10):
    """A record that SHOULD pass the guard."""
    per_fold = [0.79 + 0.001 * i for i in range(folds)]
    return {
        "metric": "metalcytotox_ru_auc",
        "value": 0.7613038529802696,
        "status": "MEASURED_PARTIAL_LIFT",
        "paper_protocol_measurement": {
            "folds_run": folds,
            "per_fold_auc": per_fold,
            "mean_auc": sum(per_fold) / len(per_fold),
            "std_auc": 0.03,
            "cohort": {
                "rows": 11517,
                "complexes": 4502,
                "cell_lines": cell_lines,
            },
        },
    }


# ---------------------------------------------------------------------------
# 1-3  the real on-disk record is cross-cohort, and the guard says so
# ---------------------------------------------------------------------------


def test_m1_premise_record_is_cross_cohort_on_disk():
    """m1 is only a live item if the recorded CV is genuinely cross-cohort."""
    assert REAL_METRIC.is_file(), "m1 metric file absent — re-verify premise"
    payload = json.loads(REAL_METRIC.read_text(encoding="utf-8"))
    cohort = payload["paper_protocol_measurement"]["cohort"]
    assert cohort["cell_lines"] != PAPER_CELL_LINES, (
        f"the recorded cohort is already {cohort['cell_lines']} cell lines — "
        "the cross-cohort gap this guard closes has been closed; re-author"
    )
    assert cohort["cell_lines"] > PAPER_CELL_LINES


def test_audit_real_record_fails_and_reports_the_delta():
    result = audit_m1_paper_cohort(REAL_METRIC, repo_root=REPO_ROOT)
    assert result["status"] == "FAIL"
    assert result["cross_cohort"] is True
    assert result["observed_cell_lines"] != PAPER_CELL_LINES
    assert result["folds_run"] == EXPECTED_FOLDS
    assert METRIC_KEY in result["evidence"]


def test_fail_evidence_forbids_the_like_for_like_reading():
    """The whole point: the -0.038 vs the paper may not be quoted as-is."""
    result = audit_m1_paper_cohort(REAL_METRIC, repo_root=REPO_ROOT)
    ev = result["evidence"]
    assert "CROSS-COHORT" in ev
    assert "NOT a like-for-like comparison" in ev
    assert str(PAPER_MEAN_AUC) in ev
    assert "Remediation" in ev, (
        "a FAIL must say what would close it, not just that it is open"
    )


# ---------------------------------------------------------------------------
# 4  the remediation cohort is CPU-constructible today (no data blocker)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not RELEASE_CSV.is_file(), reason="release CSV not on disk")
def test_top23_remediation_cohort_is_constructible_on_cpu():
    """MEASURED on the release CSV: top-23 ~ the paper's 23-line funnel."""
    result = audit_m1_paper_cohort(REAL_METRIC, repo_root=REPO_ROOT)
    top = result["top_n_cohort"]
    assert top is not None, (
        "the top-23 cohort could not be measured on CPU — the remediation "
        "would be data-blocked, which changes the honest framing"
    )
    assert top["n_cell_lines"] == PAPER_CELL_LINES
    assert top["n_rows"] == 11517
    assert top["n_complexes"] == 4502


# ---------------------------------------------------------------------------
# 5-7  positive + partial-sweep behaviour
# ---------------------------------------------------------------------------


def test_comparable_record_passes(tmp_path):
    result = audit_m1_paper_cohort(
        _write(tmp_path, _comparable_record()), recompute_cohort=False
    )
    assert result["status"] == "PASS"
    assert result["cross_cohort"] is False
    assert result["per_fold_auc_finite"] == EXPECTED_FOLDS


def test_partial_sweep_on_the_right_cohort_still_fails(tmp_path):
    """Matching cell-line scope does not excuse an incomplete CV."""
    result = audit_m1_paper_cohort(
        _write(tmp_path, _comparable_record(folds=4)), recompute_cohort=False
    )
    assert result["status"] == "FAIL"
    assert "incomplete" in result["evidence"]
    assert f"4/{EXPECTED_FOLDS}" in result["evidence"]


def test_non_finite_fold_auc_is_not_counted():
    """NaN is exactly what a crashed fold leaves behind — it is not data."""
    from molmetal.scripts.wf_r43a1_m1_paper_cohort_guard import _is_finite_number

    assert _is_finite_number(0.81) is True
    assert _is_finite_number(0) is True
    assert _is_finite_number(float("nan")) is False
    assert _is_finite_number(float("inf")) is False
    assert _is_finite_number(None) is False
    assert _is_finite_number("0.81") is False
    assert math.isnan(float("nan"))


def test_nan_fold_in_a_full_sweep_still_fails(tmp_path):
    payload = _comparable_record()
    payload["paper_protocol_measurement"]["per_fold_auc"][3] = float("nan")
    result = audit_m1_paper_cohort(
        _write(tmp_path, payload), recompute_cohort=False
    )
    assert result["status"] == "FAIL"
    assert result["per_fold_auc_finite"] == EXPECTED_FOLDS - 1


# ---------------------------------------------------------------------------
# 8-9  degenerate records
# ---------------------------------------------------------------------------


def test_record_without_paper_protocol_block_fails_loudly(tmp_path):
    """Absent comparable evidence must FAIL, never silently inherit the value."""
    payload = {
        "metric": "metalcytotox_ru_auc",
        "value": 0.7613038529802696,
        "status": "MEASURED_PARTIAL_LIFT",
    }
    result = audit_m1_paper_cohort(_write(tmp_path, payload), recompute_cohort=False)
    assert result["status"] == "FAIL"
    assert "no paper_protocol_measurement" in result["evidence"]
    assert result["mean_auc"] == 0.7613038529802696


def test_untraceable_cohort_scope_fails_loudly(tmp_path):
    payload = _comparable_record()
    payload["paper_protocol_measurement"]["cohort"] = {
        "rows": 15799,
        "complexes": 4793,
    }
    result = audit_m1_paper_cohort(_write(tmp_path, payload), recompute_cohort=False)
    assert result["status"] == "FAIL"
    assert "untraceable" in result["evidence"]


def test_malformed_json_fails_and_absent_metric_skips(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    result = audit_m1_paper_cohort(bad, recompute_cohort=False)
    assert result["status"] == "FAIL"
    assert "unreadable" in result["evidence"]

    result = audit_m1_paper_cohort(tmp_path / "nope.json", recompute_cohort=False)
    assert result["status"] == "SKIP"


# ---------------------------------------------------------------------------
# 10  gate registration
# ---------------------------------------------------------------------------


def test_gate_registered_on_all_and_skips_by_default(monkeypatch):
    monkeypatch.delenv(OPT_IN_ENV, raising=False)
    pre = pytest.importorskip("molmetal.scripts.pre_flight_check")
    gates = dict(pre.build_gates("all"))
    assert GATE_NAME in gates, "m1 gate is not registered on the 'all' pipeline"
    assert len(gates) == pre.EXPECTED_GATE_COUNTS["all"], (
        f"gate-count drift: build_gates('all')={len(gates)} vs "
        f"EXPECTED_GATE_COUNTS['all']={pre.EXPECTED_GATE_COUNTS['all']}"
    )
    result = gates[GATE_NAME]()
    assert result.status == "SKIP"
    assert OPT_IN_ENV in result.evidence


def test_gate_enforces_when_opted_in(monkeypatch):
    monkeypatch.setenv(OPT_IN_ENV, "1")
    pre = pytest.importorskip("molmetal.scripts.pre_flight_check")
    gates = dict(pre.build_gates("all"))
    result = gates[GATE_NAME]()
    assert result.status == "FAIL", (
        "with WF43_M1_PAPER_COHORT=1 the real m1 record must FAIL — the "
        "recorded CV is still cross-cohort"
    )
    assert "CROSS-COHORT" in result.evidence
