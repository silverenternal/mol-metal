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
from molmetal.scripts.registry import (  # noqa: E402
    _extract_value_basis,
    _has_numeric_leaf,
    _metric_timestamp,
    _preserve_underived_datasets,
    _resolve_metric_ref,
    _status_with_source,
    _supersession_fields,
)


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


# --- status tokens: a status is a word, not a paragraph -------------------


def test_status_token_is_accepted_verbatim():
    assert _status_with_source({"status": "MEASURED"}) == ("MEASURED", "status", None)


def test_verdict_token_is_accepted_as_a_status():
    assert _status_with_source({"verdict": "MEASURED_HONEST_NEGATIVE"}) == (
        "MEASURED_HONEST_NEGATIVE",
        "verdict",
        None,
    )


def test_tag_is_a_status_source():
    # 44 artifacts carry their status only as `tag`.
    assert _status_with_source({"tag": "MEASURED"}) == ("MEASURED", "tag", None)


def test_explicit_status_wins_over_tag():
    assert _status_with_source({"status": "BLOCKED", "tag": "MEASURED"})[0] == "BLOCKED"


def test_sentence_verdict_is_not_promoted_to_a_status():
    # Before this, 36 rows held a paragraph in `status` -- a provenance claim
    # the value did not support.
    payload = {"verdict": "THE HEAD IS NOT THE RING DEFECT. On clean geometry ..."}
    status, source, narrative = _status_with_source(payload)
    assert status is None
    assert source is None
    assert narrative == "verdict"


def test_leading_token_is_extracted_from_an_explained_verdict():
    payload = {"verdict": "MEASURED \u2014 pharmacophore_pass_rate=0.7 on 10 curated SMILES"}
    assert _status_with_source(payload) == (
        "MEASURED",
        "verdict:leading_token",
        None,
    )


def test_parenthesised_leading_token_is_extracted():
    payload = {"verdict": "MEASURED (3-metal x 3-seed x 50-epoch, scaffold)"}
    assert _status_with_source(payload)[0] == "MEASURED"


def test_two_word_answer_is_not_a_status_token():
    # "NO CLEAR DIFFERENCE -- ..." is a finding, not a status vocabulary word.
    assert _status_with_source({"verdict": "NO CLEAR DIFFERENCE -- flips do not clear"})[0] is None


def test_a_sentence_status_is_not_quotable():
    r = _assess(_good(status="THE HEAD IS NOT THE RING DEFECT."))
    assert r["quotable"] is False


# --- value provenance: a best-of is not a central tendency ----------------


def test_value_basis_prefers_mean_over_best_measured():
    payload = {"aggregate": {"best_measured": 2.2884, "mean": 2.4486}}
    assert _extract_value_basis(payload) == (2.4486, "aggregate.mean")


def test_value_basis_falls_back_to_best_measured_when_alone():
    assert _extract_value_basis({"aggregate": {"best_measured": 0.7}}) == (
        0.7,
        "aggregate.best_measured",
    )


def test_headline_key_overrides_a_best_measured_aggregate():
    payload = {
        "aggregate": {"best_measured": 2.2884, "five_pocket_mean": 2.4486},
        "headline_key": "aggregate.five_pocket_mean",
    }
    assert _extract_value_basis(payload) == (2.4486, "headline_key")


def test_value_basis_names_the_top_level_key_it_used():
    assert _extract_value_basis({"value": 0.5}) == (0.5, "value")


def test_value_basis_is_none_when_no_number_exists():
    assert _extract_value_basis({"note": "no numbers here"}) == (None, None)


# --- numeric leaves ------------------------------------------------------


def test_has_numeric_leaf_finds_nested_numbers():
    assert _has_numeric_leaf({"a": [{"b": 1.0}]}) is True


def test_has_numeric_leaf_ignores_booleans():
    assert _has_numeric_leaf({"ok": True, "flag": False}) is False


def test_has_numeric_leaf_false_for_prose_only():
    assert _has_numeric_leaf({"verdict": "MEASURED", "note": "see report"}) is False


# --- dates ---------------------------------------------------------------


def test_nested_date_key_is_read_and_its_path_recorded():
    payload = {"run": {"completed_date": "2026-10-04"}}
    assert _metric_timestamp(payload) == ("2026-10-04", "nested:run.completed_date")


def test_latest_nested_date_wins():
    payload = {"a": {"date": "2026-01-01"}, "b": {"date_utc": "2026-05-05"}}
    assert _metric_timestamp(payload)[0] == "2026-05-05"


def test_top_level_date_wins_over_nested():
    payload = {"date": "2026-02-02", "run": {"date": "2026-09-09"}}
    assert _metric_timestamp(payload) == ("2026-02-02", "date")


def test_a_year_in_prose_is_not_a_date():
    # 119 of 142 undated files contain a date somewhere; only 23 under a
    # key that claims to be one.  A citation year must not become the date.
    payload = {"reference": "Guan 2023 ICLR", "note": "measured 2026-10-01"}
    assert _metric_timestamp(payload) == (None, None)


# --- generate must not delete rows it cannot re-derive -------------------


def test_preserve_underived_datasets_keeps_hand_added_rows(tmp_path):
    path = tmp_path / "datasets.yaml"
    path.write_text(
        "datasets:\n"
        "  derived_row:\n"
        "    source: configs/data_checksums.json\n"
        "  hand_added_row:\n"
        "    source: b76ef187\n"
        "    note: no generator declares this\n"
    )
    built = {"derived_row": {"source": "configs/data_checksums.json"}}
    kept = _preserve_underived_datasets(built, path)
    assert kept == ["hand_added_row"]
    assert set(built) == {"derived_row", "hand_added_row"}
    assert built["hand_added_row"]["preserved_from"] == "previous registry/datasets.yaml"


def test_preserve_underived_datasets_does_not_override_a_derived_row(tmp_path):
    path = tmp_path / "datasets.yaml"
    path.write_text("datasets:\n  row:\n    stale: true\n")
    built = {"row": {"fresh": True}}
    assert _preserve_underived_datasets(built, path) == []
    assert built["row"] == {"fresh": True}


def test_preserve_underived_datasets_tolerates_a_missing_file(tmp_path):
    assert _preserve_underived_datasets({}, tmp_path / "absent.yaml") == []


# --- supersession: a pointer has to resolve, and it has to bite -----------


def test_resolve_metric_ref_accepts_a_path_or_an_id():
    assert _resolve_metric_ref("metrics/by_metric/moleculenet_auc.json") == "moleculenet_auc"
    assert _resolve_metric_ref("moleculenet_auc") == "moleculenet_auc"


def test_resolve_metric_ref_rejects_prose_and_dicts():
    # `supersedes` was used for both pointers and paragraphs; only one of them
    # can be checked.
    assert _resolve_metric_ref("the framing 'the loss contains no objective'") is None
    assert _resolve_metric_ref({"path": "metrics/by_metric/x.json"}) is None
    assert _resolve_metric_ref(None) is None
    assert _resolve_metric_ref("") is None


def test_superseded_row_is_not_quotable():
    r = _assess(_good(superseded_by="metrics/by_metric/newer_measurement.json"))
    assert r["quotable"] is False
    assert any("superseded by" in x for x in r["not_quotable"])


def test_superseded_row_stays_unquotable_when_the_target_is_missing():
    r = _assess(_good(superseded_by="metrics/by_metric/never_existed.json"))
    assert r["quotable"] is False
    assert any("does not exist" in x for x in r["not_quotable"])


def test_supersession_fields_resolve_a_pointer():
    f = _supersession_fields({"superseded_by": "metrics/by_metric/m7_diversity.json"})
    assert f["superseded_by"] == "m7_diversity"
    assert f["superseded_by_resolved"] is True


def test_supersession_fields_flag_a_dangling_pointer():
    f = _supersession_fields({"supersedes": "metrics/by_metric/never_existed.json"})
    assert f["supersedes"] == "never_existed"
    assert f["supersedes_dangling"] is True


def test_supersession_fields_ignore_prose():
    assert _supersession_fields({"supersedes": "a paragraph about the old number"}) == {}


def test_redefined_metric_cannot_be_table_grade():
    r = _assess(_good(definition_status="REDEFINED - now a reproducible IntDiv1"))
    assert r["quotable"] is True
    assert r["headline_grade"] is False
    assert any("not comparable" in x for x in r["not_headline"])


def test_definition_status_that_is_not_a_redefinition_is_ignored():
    r = _assess(_good(definition_status="STABLE since 2026-09-01"))
    assert r["headline_grade"] is True
