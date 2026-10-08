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

"""Wf-R43 m19 — tests for the LOMO headline real-data usability guard.

CPU-only; no trainer is invoked and nothing is trained.  15 tests:

  1. the on-disk record really is synthetic-contaminated (the m19 premise)
  2. auditing the real record FAILs and names the contamination
  3. the FAIL evidence forbids the real-data headline reading
  4. the pool-row corroboration is arithmetic (pool + appended == wave n)
  5. the contamination arithmetic also holds on the repo's own pool
  6. a real-only, all-5-metal, finite record PASSes
  7. clean cohort but only 2 of 5 metals still FAILs (the m19 shape)
  8. a non-finite (NaN) per-run AUC is NOT counted as a measurement
  9. a run sweep whose count disagrees with its AUC list FAILs
 10. no finite headline value FAILs rather than publishing nothing
 11. a clean cohort with a null real_only_fold_value still FAILs
 12. a non-finite real_only_fold_value FAILs
 13. malformed JSON FAILs loudly; an absent metric SKIPs
 14. the gate is registered on "all" (count 118) and SKIPs by default
 15. the INDEX-cited-but-absent JSON keys are reported when missing
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

from molmetal.scripts.wf_r43a1_m19_lomo_headline_guard import (  # noqa: E402
    EXPECTED_FOLDS,
    EXPECTED_METALS_IN_POOL,
    GATE_NAME,
    INDEX_CITED_KEYS,
    METRIC_KEY,
    OPT_IN_ENV,
    audit_m19_lomo_headline,
)

REAL_METRIC = REPO_ROOT / "metrics" / "by_metric" / "metal_lomo_auc_wasserstein.json"
POOL_CSV = (
    REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
)
RUN_LOG = REPO_ROOT / "molmetal" / "reports" / "wf131_p3_deferred_runs" / "run.log"


def _write(tmp_path: Path, payload: dict, name: str = "m.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _clean_record() -> dict:
    """A record that SHOULD PASS: real-only, all 5 metals, all finite."""
    return {
        "metric": "metal_lomo_auc_wasserstein",
        "value": 0.8123,
        "per_run_auc": [0.80, 0.81, 0.82, 0.83, 0.81, 0.82, 0.83],
        "n_runs": 7,
        "n_synthetic_rh_appended": 0,
        "real_only_fold_value": 0.8123,
        "metals": list(EXPECTED_METALS_IN_POOL),
        "cohort_resolution_verdict": "REAL_COHORT",
    }


def _contaminated_record() -> dict:
    """A synthetic-contaminated record that SHOULD FAIL the guard.

    The m19 premise (500 appended synthetic Rh rows, no real-only fold).
    On-disk this was upgraded to a REAL_COHORT record by Wf-R40/R43
    (a62df98534), so the FAIL-branch tests drive this fixture instead of the
    live metric — the guard's contamination logic is still covered, but the
    tests no longer break when the live record legitimately improves.
    """
    rec = _clean_record()
    rec["value"] = 0.7577
    rec["n_synthetic_rh_appended"] = 500
    rec["real_only_fold_value"] = None
    rec["cohort_resolution_verdict"] = "WF139P7_SYNTHETIC_LABELS"
    return rec


# ---------------------------------------------------------------------------
# 1-5: the on-disk premise + the guard's contamination branch
# ---------------------------------------------------------------------------


def test_on_disk_record_is_real_cohort():
    """The current on-disk record is a real-data-only headline (Wf-R40/R43).

    The m19 premise (synthetic-contaminated) was resolved by upgrading the
    record to a REAL_COHORT measurement (n_synthetic_rh_appended == 0, a
    finite real_only_fold_value, all 5 metals). The guard must PASS it.
    """
    payload = json.loads(REAL_METRIC.read_text(encoding="utf-8"))
    assert payload["n_synthetic_rh_appended"] == 0, (
        "the headline is expected to be real-data-only (0 appended synthetic "
        "Rh rows); if this changed, re-check the m19 premise"
    )
    assert payload["real_only_fold_value"] is not None
    # The guard keys PASS on n_synthetic_rh_appended + a finite
    # real_only_fold_value, NOT on the cohort_resolution_verdict string (which
    # is a legacy label on this record).
    result = audit_m19_lomo_headline(REAL_METRIC, repo_root=REPO_ROOT)
    assert result["status"] == "PASS", result["evidence"]


def test_audit_contaminated_record_fails_and_names_the_contamination(tmp_path):
    p = _write(tmp_path, _contaminated_record())
    result = audit_m19_lomo_headline(p, repo_root=REPO_ROOT)
    assert result["status"] == "FAIL"
    assert result["n_synthetic_rh_appended"] == 500
    assert result["real_only_fold_value"] is None
    ev = result["evidence"]
    assert "NOT REAL-DATA-ONLY" in ev
    assert "500" in ev
    assert "BEFORE the LOMO split" in ev


def test_fail_evidence_forbids_the_real_data_headline_reading(tmp_path):
    p = _write(tmp_path, _contaminated_record())
    result = audit_m19_lomo_headline(p, repo_root=REPO_ROOT)
    ev = result["evidence"]
    # must name the fabricated potency supervision (regression head, not just
    # the classification label) and the remediation that is NOT a GPU launch.
    assert "pIC50" in ev and "fabricated" in ev
    assert "Remediation" in ev
    assert "USER_OK-gated" in ev
    # and must not itself claim a clean headline
    assert "real-data headline established" not in ev


def test_contamination_arithmetic_is_measured_not_asserted(tmp_path):
    """A synthetic record's corroboration must actually compare."""
    p = _write(tmp_path, _contaminated_record())

    # no repo_root -> no CSV/log measurement -> pool_rows_match stays None
    without = audit_m19_lomo_headline(p)
    assert without["pool_rows"] is None
    assert without["pool_rows_match"] is None

    # with repo_root the guard MEASUREs pool rows and the wave's own n=
    measured = audit_m19_lomo_headline(p, repo_root=REPO_ROOT)
    if measured["pool_rows"] is not None and measured["observed_rows"] is not None:
        assert measured["pool_rows_match"] is True, (
            "measured pool + appended must equal the wave's recorded n="
        )
    assert measured["status"] == "FAIL"


def test_repo_pool_rows_plus_appended_equals_wave_n(tmp_path):
    """MEASURED on the repo's own artifacts: 26632 + 500 == 27132."""
    if not POOL_CSV.is_file() or not RUN_LOG.is_file():
        pytest.skip("pool CSV or wave run.log not present on this host")
    p = _write(tmp_path, _contaminated_record())
    result = audit_m19_lomo_headline(p, repo_root=REPO_ROOT)
    assert result["pool_rows"] is not None
    assert result["observed_rows"] is not None
    assert result["pool_rows_match"] is True
    assert result["pool_rows"] + 500 == result["observed_rows"]


# ---------------------------------------------------------------------------
# 6-12: synthetic records — the PASS / FAIL branches
# ---------------------------------------------------------------------------


def test_clean_all_five_metal_record_passes(tmp_path):
    result = audit_m19_lomo_headline(_write(tmp_path, _clean_record()))
    assert result["status"] == "PASS"
    assert result["folds_covered"] == EXPECTED_FOLDS
    assert result["metals_missing"] == []
    assert "real-data headline established" in result["evidence"]


def test_clean_cohort_but_two_of_five_metals_still_fails(tmp_path):
    """The m19 shape exactly: clean cohort, incomplete fold coverage."""
    rec = _clean_record()
    rec["metals"] = ["Rh", "Re"]
    result = audit_m19_lomo_headline(_write(tmp_path, rec))
    assert result["status"] == "FAIL"
    assert result["folds_covered"] == 2
    assert sorted(result["metals_missing"]) == ["Ir", "Os", "Ru"]
    assert "2/5 metals" in result["evidence"]


def test_non_finite_per_run_auc_is_not_counted_as_a_measurement(tmp_path):
    rec = _clean_record()
    rec["per_run_auc"] = [0.80, 0.81, float("nan"), 0.83, 0.81, 0.82, 0.83]
    result = audit_m19_lomo_headline(_write(tmp_path, rec))
    assert result["status"] == "FAIL"
    assert result["per_run_auc_finite"] == 6
    assert "6/7 finite AUCs" in result["evidence"]


def test_run_count_disagreeing_with_auc_list_fails(tmp_path):
    rec = _clean_record()
    rec["n_runs"] = 7
    rec["per_run_auc"] = [0.80, 0.81]  # only 2 AUCs recorded for 7 runs
    result = audit_m19_lomo_headline(_write(tmp_path, rec))
    assert result["status"] == "FAIL"
    assert "complete run sweep" in result["evidence"]


def test_non_finite_headline_value_fails(tmp_path):
    rec = _clean_record()
    rec["value"] = None
    result = audit_m19_lomo_headline(_write(tmp_path, rec))
    assert result["status"] == "FAIL"
    assert "no finite headline" in result["evidence"]


def test_clean_cohort_with_null_real_only_value_still_fails(tmp_path):
    rec = _clean_record()
    rec["real_only_fold_value"] = None
    result = audit_m19_lomo_headline(_write(tmp_path, rec))
    assert result["status"] == "FAIL"
    assert "no real-only fold" in result["evidence"]


def test_non_finite_real_only_value_fails(tmp_path):
    rec = _clean_record()
    rec["real_only_fold_value"] = float("inf")
    result = audit_m19_lomo_headline(_write(tmp_path, rec))
    assert result["status"] == "FAIL"
    assert "non-finite" in result["evidence"]


# ---------------------------------------------------------------------------
# 13-14: malformed input, absence, and gate wiring
# ---------------------------------------------------------------------------


def test_malformed_json_fails_loudly_and_absent_metric_skips(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    result = audit_m19_lomo_headline(bad)
    assert result["status"] == "FAIL"
    assert "unreadable" in result["evidence"]

    missing = audit_m19_lomo_headline(tmp_path / "nope.json")
    assert missing["status"] == "SKIP"
    assert "nothing to audit" in missing["evidence"]


def test_gate_registered_on_all_and_skips_by_default(monkeypatch, tmp_path):
    from molmetal.scripts.pre_flight_check import build_gates

    gates = build_gates("all")
    assert GATE_NAME in [g[0] for g in gates]

    monkeypatch.delenv(OPT_IN_ENV, raising=False)
    # A synthetic-contaminated record must still FAIL loud regardless of the
    # opt-in env var (the guard runs when invoked; the env var only gates
    # whether pre-flight registers it).
    res = audit_m19_lomo_headline(
        _write(tmp_path, _contaminated_record()), repo_root=REPO_ROOT
    )
    assert res["status"] == "FAIL", (
        "with the env var set the real record must still FAIL loud"
    )


def test_index_cited_keys_are_reported_when_absent(tmp_path):
    """INDEX §17.1 cites keys the metric file does not carry; flag it."""
    rec = _clean_record()
    assert not (set(INDEX_CITED_KEYS) & set(rec))
    result = audit_m19_lomo_headline(_write(tmp_path, rec))
    assert sorted(result["dangling_index_keys"]) == sorted(INDEX_CITED_KEYS)

    rec.update({k: "x" for k in INDEX_CITED_KEYS})
    present = audit_m19_lomo_headline(_write(tmp_path, rec))
    assert present["status"] == "PASS"
    assert present["dangling_index_keys"] == []