# SPDX-License-Identifier: Apache-2.0
"""Wf-A155 — PoseBusters validity column: regression guard.

WHY THIS EXISTS
---------------
Wf-A155 branch POSE added the repo's FIRST pool-scale PoseBusters
measurement (``metrics/by_metric/wfa155_posebusters_validity.json``) over
the published CFM decode pool
``molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv``.

Three things are easy to get wrong with this metric and nothing in the tree
would have caught them:

1. **It is not a restatement of the 8.51 % headline.**  The 8.51 % is an
   END-TO-END single-molecule rate over 9 RUNS
   (``cfm_fragmentation_audit.json`` ->
   ``/summary/generation_side/single_molecule_rate_end_to_end``).  The
   Wf-A155 number is a SINGLE-POOL GEOMETRY rate over ONE decode.  If a
   later refactor makes the two agree, that is a coincidence to notice, not
   a validity argument.
2. **The denominators must stay separated.**  A row that cannot be parsed
   and a row that cannot be embedded are DIFFERENT failures, and the strict
   and scored rates are computed over different populations.  Collapsing
   them silently changes the headline.
3. **PoseBusters has no metal-coordination-sphere check.**  A % PB-Valid for
   a metallodrug pool says nothing about whether the metal is coordinated
   correctly.  The artifact must keep saying so.

WHAT THIS GUARD ASSERTS
-----------------------
* The aggregation maths on a synthetic record list (rates, denominators,
  per-check breakdown).
* The unparseable-row path: a row RDKit cannot parse is SKIPPED (never
  handed to PoseBusters) and COUNTED, and it does not pollute the
  per-check breakdown.
* The artifact's JSON schema: required keys present, counts internally
  consistent, the recorded pool sha256 matches the pool on disk.
* The artifact is NOT a restatement of the 8.51 % or the 25.05 % numbers,
  and it carries the four comparability breaks + the metal-coordination
  honest negative.
* The script is importable, never imports ``torch``, and never touches CUDA.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest

from molmetal.scripts.wf_a155_posebusters_validity import (
    DEFAULT_EMBED_METHOD,
    DEFAULT_N_CONFORMERS,
    DEFAULT_OPTIMIZE_METHOD,
    LONGEST_POOL_SMILES_CALL_SECONDS,
    NARRATIVE_PATHS,
    STATUS_EMBED_FAILED,
    STATUS_PB_INVALID,
    STATUS_PB_VALID,
    STATUS_SKIPPED,
    STATUS_TIMEOUT,
    STATUS_UNPARSEABLE,
    _end_to_end_headline,
    _leaf_map,
    _mask_identifier_tokens,
    _narrative_self_consistency,
    _prior_posebusters_artifacts,
    _published_cross_check,
    _resolve,
    _standalone_in_any,
    _NUM_RE,
    aggregate,
    build_comparability_breaks,
    build_pool_profile,
    classify_result,
    longest_pool_measurement_binding,
    pool_composition,
    read_pool,
    read_pool_epoch_labels,
    read_pool_fragment_counts,
    select_rows,
    sha256_file,
    strict_rate_via_repo_helper,
    summarize_conformer_probe,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "wf_a155_posebusters_validity.py"
ARTIFACT = REPO_ROOT / "metrics" / "by_metric" / "wfa155_posebusters_validity.json"
POOL = (
    REPO_ROOT
    / "molmetal"
    / "reports"
    / "cfm_c8_fullgen_2026_10_24"
    / "generated_smiles.csv"
)

#: The published END-TO-END audit artifact.  The "8.51 %" headline (the
#: median over its 9 generation-side runs) lives at
#: /summary/generation_side/single_molecule_rate_end_to_end inside it.
FRAGMENTATION_AUDIT = (
    REPO_ROOT / "metrics" / "by_metric" / "cfm_fragmentation_audit.json"
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def _record(
    status: str,
    *,
    checks: Dict[str, Any] | None = None,
    n_conformers: int = 5,
    n_pb_rows: int = 1,
    elapsed: float = 1.0,
) -> Dict[str, Any]:
    """Build a synthetic per-row record with the shape ``aggregate`` consumes."""
    return {
        "row_index": 0,
        "smiles": "CCO",
        "status": status,
        "pb_valid": True if status == STATUS_PB_VALID else False,
        "n_conformers": n_conformers,
        "n_pb_rows": n_pb_rows,
        "checks": checks or {},
        "failures": [],
        "missing_checks": [],
        "elapsed_s": elapsed,
        "error": None,
    }


def _published_end_to_end_median() -> float:
    """The published END-TO-END median headline, READ from the audit artefact.

    The value, its run count and its jsonpath all come from
    ``metrics/by_metric/cfm_fragmentation_audit.json`` on disk.  Nothing
    about this number is typed into the test.
    """
    doc = json.loads(FRAGMENTATION_AUDIT.read_text(encoding="utf-8"))
    node = doc["summary"]["generation_side"]["single_molecule_rate_end_to_end"]
    return float(node["median"])


def _published_pool_profile() -> Dict[str, Any]:
    """Build the emitter's pool profile for the published pool.

    Wf-A164 replaced every pool-specific literal with a COMPUTED value, so
    the test builds the profile exactly the way the emitter does -- from the
    pool on disk -- instead of typing any of its numbers.
    """
    smiles, columns = read_pool(POOL)
    return build_pool_profile(
        pool_path=POOL,
        pool_sha256=sha256_file(POOL),
        n_data_rows=len(smiles),
        columns=columns,
        epoch_labels=read_pool_epoch_labels(POOL),
        fragment_counts=read_pool_fragment_counts(POOL),
        composition=pool_composition(smiles),
    )


def _independent_pool_rows() -> List[Dict[str, str]]:
    """Read the pool CSV with the stdlib, independently of the emitter.

    The emitter's readers are cross-checked against this independent read,
    so a reader that is wrong cannot pass by agreeing with itself.
    """
    import csv

    with POOL.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ----------------------------------------------------------------
# 1. Aggregation maths on a synthetic list
# ----------------------------------------------------------------
def test_aggregate_maths_on_synthetic_records() -> None:
    records = [
        _record(STATUS_PB_VALID, checks={"a": True, "b": True}),
        _record(STATUS_PB_VALID, checks={"a": True, "b": True}),
        _record(STATUS_PB_VALID, checks={"a": True, "b": True}),
        _record(STATUS_PB_INVALID, checks={"a": True, "b": False}),
        _record(STATUS_PB_INVALID, checks={"a": True, "b": False}),
        _record(STATUS_EMBED_FAILED, n_conformers=0),
        _record(STATUS_EMBED_FAILED, n_conformers=0),
        _record(STATUS_UNPARSEABLE, n_conformers=0),
        _record(STATUS_UNPARSEABLE, n_conformers=0),
        _record(STATUS_SKIPPED, n_conformers=0),
    ]
    counts = aggregate(records)

    assert counts["n_rows_in_scope"] == 10
    assert counts["n_pb_valid"] == 3
    assert counts["n_pb_invalid"] == 2
    assert counts["n_embed_failed"] == 2
    assert counts["n_unparseable"] == 2
    assert counts["n_skipped_posebusters_missing"] == 1
    assert counts["n_scored"] == 5

    # STRICT denominator = every row in scope.
    assert counts["pb_valid_rate_strict"] == pytest.approx(3 / 10)
    # SCORED denominator = rows that produced a PoseBusters verdict.
    assert counts["pb_valid_rate_scored"] == pytest.approx(3 / 5)
    # Third, non-headline variant: embed failure counts as a failure.
    assert counts["pb_valid_rate_scored_plus_embed_failed"] == pytest.approx(3 / 7)

    assert counts["denominators"]["strict"]["n"] == 10
    assert counts["denominators"]["scored"]["n"] == 5
    assert counts["denominators"]["scored_plus_embed_failed"]["n"] == 7
    assert counts["denominators"]["scored_plus_embed_failed"]["is_headline"] is False


def test_aggregate_on_empty_input_does_not_divide_by_zero() -> None:
    counts = aggregate([])
    assert counts["n_rows_in_scope"] == 0
    assert counts["n_scored"] == 0
    assert counts["pb_valid_rate_strict"] == 0.0
    assert counts["pb_valid_rate_scored"] == 0.0
    assert counts["pb_valid_rate_scored_plus_embed_failed"] == 0.0


def test_aggregate_per_check_breakdown_separates_missing_from_false() -> None:
    records = [
        _record(STATUS_PB_VALID, checks={"ok": True, "bad": False, "gone": None}),
        _record(STATUS_PB_INVALID, checks={"ok": True, "bad": False, "gone": None}),
        # An UNPARSEABLE row has no check verdicts at all -- it must not be
        # counted as failing every check.
        _record(STATUS_UNPARSEABLE, checks={"ok": True, "bad": True, "gone": True}),
    ]
    counts = aggregate(records)
    assert counts["per_check_not_true_counts"] == {"bad": 2, "gone": 2}
    assert counts["per_check_explicit_false_counts"] == {"bad": 2}
    assert counts["per_check_missing_counts"] == {"gone": 2}
    # A check that is True is never in the breakdown.
    assert "ok" not in counts["per_check_not_true_counts"]


def test_aggregate_records_partial_embed_rows() -> None:
    records = [
        _record(STATUS_PB_VALID, n_conformers=5),
        _record(STATUS_PB_INVALID, n_conformers=2),
        _record(STATUS_PB_INVALID, n_conformers=0),
    ]
    counts = aggregate(records)
    # Only rows that embedded at least one but fewer than 5 conformers count.
    assert counts["n_rows_with_partial_embed"] == 1
    assert counts["n_conformers_embedded_histogram"] == {"0": 1, "2": 1, "5": 1}


def test_pool_composition_counts_metals_and_explicit_hydrogens() -> None:
    """The pool's metal content and explicit-H content must be MEASURED.

    This is the measured backbone of the artifact's central honest-negative:
    PoseBusters has no metal-coordination-sphere check, and MMFF94 cannot
    parameterise most metal centres -- so a 100 %-metal pool makes the PB
    verdict substantially a force-field verdict.
    """
    composition = pool_composition(
        [
            "CCO",  # no metal, no explicit H
            "C[Pt](C)(C)C",  # one metal centre
            "CC[Ir]Cl",  # another metal centre
            "[H]C([H])O",  # explicit hydrogens, no metal
            "not_a_smiles",  # unparseable -> skipped, not counted
        ]
    )
    assert composition["n_rows_parsed_by_rdkit"] == 4
    assert composition["n_rows_with_at_least_one_metal_atom"] == 2
    assert composition["fraction_rows_with_a_metal_atom"] == pytest.approx(2 / 5)
    assert composition["total_metal_atoms"] == 2
    assert composition["metal_element_counts"] == {"Ir": 1, "Pt": 1}
    assert composition["n_rows_with_an_explicit_hydrogen_atom_in_the_smiles"] == 1


def test_pool_composition_of_the_published_pool_is_all_metal() -> None:
    """Every row of the published pool carries a metal centre."""
    smiles, _columns = read_pool(POOL)
    composition = pool_composition(smiles)
    n_rows = len(smiles)
    assert composition["n_rows_parsed_by_rdkit"] == n_rows
    assert composition["n_rows_with_at_least_one_metal_atom"] == n_rows
    assert composition["fraction_rows_with_a_metal_atom"] == pytest.approx(1.0)
    assert (
        composition["n_rows_with_an_explicit_hydrogen_atom_in_the_smiles"] == 0
    )
    assert sum(composition["metal_element_counts"].values()) == n_rows


def test_strict_rate_cross_check_uses_the_repo_helper() -> None:
    """The strict rate must equal the repo's own ``pb_pass_rate`` number.

    The artifact must not re-derive the number and hope it agrees; it must
    call ``molmetal.validation.posebusters_runner.pb_pass_rate`` -- the
    existing aggregation surface -- and show the two agree.
    """
    records = [
        _record(STATUS_PB_VALID, checks={"a": True}),
        _record(STATUS_PB_VALID, checks={"a": True}),
        _record(STATUS_PB_INVALID, checks={"a": False}),
        _record(STATUS_EMBED_FAILED, n_conformers=0),
        _record(STATUS_UNPARSEABLE, n_conformers=0),
    ]
    counts = aggregate(records)
    assert counts["pb_valid_rate_strict"] == pytest.approx(2 / 5)
    assert counts["pb_valid_rate_strict_via_repo_helper"] == pytest.approx(2 / 5)
    assert strict_rate_via_repo_helper(records) == pytest.approx(2 / 5)
    # A skipped row stays in the denominator of the repo helper too.
    with_skip = records + [_record(STATUS_SKIPPED, n_conformers=0)]
    assert strict_rate_via_repo_helper(with_skip) == pytest.approx(2 / 6)
    # Empty input keeps the helper's historical 0.0.
    assert strict_rate_via_repo_helper([]) == 0.0


def test_aggregate_splits_on_the_pool_fragment_count() -> None:
    """The rate must be reportable CONDITIONAL on the pool's fragment count.

    PoseBusters' ``all_atoms_connected`` check is exactly the check a
    multi-fragment row cannot pass, so the pooled rate mixes a geometry
    question with a fragmentation question.  The single-fragment conditional
    rate is the number that speaks to 3D geometry alone.
    """
    single_valid = _record(STATUS_PB_VALID, checks={"all_atoms_connected": True})
    single_valid["n_frags_raw"] = 1
    single_invalid = _record(STATUS_PB_INVALID, checks={"all_atoms_connected": True})
    single_invalid["n_frags_raw"] = 1
    multi_invalid = _record(STATUS_PB_INVALID, checks={"all_atoms_connected": False})
    multi_invalid["n_frags_raw"] = 3
    multi_embed_failed = _record(STATUS_EMBED_FAILED, n_conformers=0)
    multi_embed_failed["n_frags_raw"] = 2

    counts = aggregate([single_valid, single_invalid, multi_invalid, multi_embed_failed])
    groups = counts["by_pool_fragment_count"]
    single = groups["single_fragment"]
    multi = groups["multi_fragment"]

    assert single["n_rows"] == 2
    assert single["n_pb_valid"] == 1
    assert single["n_scored"] == 2
    assert single["pb_valid_rate_scored"] == pytest.approx(1 / 2)
    assert single["pb_valid_rate_strict"] == pytest.approx(1 / 2)

    assert multi["n_rows"] == 2
    assert multi["n_pb_valid"] == 0
    assert multi["n_embed_failed"] == 1
    assert multi["pb_valid_rate_scored"] == pytest.approx(0.0)
    assert multi["pb_valid_rate_strict"] == pytest.approx(0.0)

    # Rows with no fragment count land in their own bucket, never silently
    # folded into either group.
    unknown = _record(STATUS_PB_VALID, checks={"a": True})
    assert unknown.get("n_frags_raw") is None
    groups2 = aggregate([unknown])["by_pool_fragment_count"]
    assert groups2["fragment_count_unknown"]["n_rows"] == 1
    assert groups2["single_fragment"]["n_rows"] == 0
    assert groups2["multi_fragment"]["n_rows"] == 0


def test_read_pool_fragment_counts_matches_the_pool() -> None:
    """The fragment histogram must agree with an INDEPENDENT read of the pool.

    The expected histogram is computed from the pool CSV with the stdlib
    ``csv`` module rather than typed into the test, so the emitter's reader
    is checked against a second implementation instead of against itself.
    """
    counts = read_pool_fragment_counts(POOL)
    rows = _independent_pool_rows()
    assert len(counts) == len(rows)

    histogram = Counter(counts)
    independent = Counter(int(row["n_frags_raw"]) for row in rows)
    assert histogram == independent
    assert sum(histogram.values()) == len(rows)

    # The published artifact's single-fragment group must be the pool's own
    # single-fragment row count.
    artifact = _load_artifact()
    single = artifact["counts"]["by_pool_fragment_count"]["single_fragment"]
    assert single["n_rows"] == independent.get(1, 0)


def test_read_pool_epoch_labels_matches_the_pool() -> None:
    """The pool is THREE epoch-wise decodes of ONE checkpoint, not one decode.

    The artifact's provenance must be MEASURED from the pool's own columns
    rather than asserted, so the epoch label list is read from disk.
    """
    labels = read_pool_epoch_labels(POOL)
    rows = _independent_pool_rows()
    assert len(labels) == len(rows)

    histogram = Counter(labels)
    independent = Counter(row["epoch"] for row in rows)
    assert histogram == independent
    # The pool is epoch-WISE, not a single decode: more than one epoch
    # label, and every epoch carries the same number of rows.
    assert len(histogram) > 1
    assert len(set(histogram.values())) == 1


def test_aggregate_splits_on_the_pool_epoch() -> None:
    records = []
    for epoch, status in (("0", STATUS_PB_VALID), ("1", STATUS_PB_INVALID)):
        record = _record(status, checks={"a": True})
        record["epoch"] = epoch
        records.append(record)
    groups = aggregate(records)["by_pool_epoch"]
    assert groups["0"]["n_rows"] == 1
    assert groups["0"]["n_pb_valid"] == 1
    assert groups["1"]["n_rows"] == 1
    assert groups["1"]["n_pb_valid"] == 0
    assert groups["1"]["pb_valid_rate_scored"] == pytest.approx(0.0)


def test_aggregate_reports_elapsed_percentiles() -> None:
    """The per-row wall-clock distribution must be visible in the artifact."""
    records = [_record(STATUS_PB_VALID, elapsed=float(value)) for value in (1, 2, 3, 4, 5)]
    counts = aggregate(records)
    percentiles = counts["row_elapsed_s_percentiles"]
    assert percentiles["p50"] == pytest.approx(3.0)
    assert percentiles["max"] == pytest.approx(5.0)
    assert percentiles["p90"] >= percentiles["p50"]
    assert percentiles["p99"] >= percentiles["p90"]
    assert counts["sum_row_elapsed_s"] == pytest.approx(15.0)


def test_aggregate_elapsed_percentiles_on_empty_input_are_zero() -> None:
    counts = aggregate([])
    assert counts["row_elapsed_s_percentiles"] == {
        "p50": 0.0,
        "p90": 0.0,
        "p99": 0.0,
        "max": 0.0,
    }
    assert counts["sum_row_elapsed_s"] == 0.0


def test_aggregate_reports_proximity_to_the_timeout_cap() -> None:
    """Rows that came within 10 % of the cap are the near-miss timeouts."""
    records = [
        _record(STATUS_PB_VALID, elapsed=1.0),
        _record(STATUS_PB_INVALID, elapsed=40.0),
        _record(STATUS_TIMEOUT, elapsed=45.0),
    ]
    counts = aggregate(records, row_timeout_s=45.0)
    assert counts["row_timeout_s"] == 45.0
    # 45.0 >= 0.9 * 45 and 40.0 < 40.5 -> exactly one row at the cap.
    assert counts["n_rows_at_or_above_90pct_of_timeout_cap"] == 1
    # Without a cap the proximity counter is simply absent (None).
    counts_no_cap = aggregate(records)
    assert counts_no_cap["n_rows_at_or_above_90pct_of_timeout_cap"] is None


# ----------------------------------------------------------------
# 2. Unparseable-row handling
# ----------------------------------------------------------------
@pytest.mark.parametrize(
    "result,parseable,expected",
    [
        ({"pb_valid": True, "failures": []}, True, STATUS_PB_VALID),
        ({"pb_valid": False, "failures": ["bond_lengths"]}, True, STATUS_PB_INVALID),
        ({"pb_valid": False, "failures": ["embed_failed"]}, True, STATUS_EMBED_FAILED),
        ({"pb_valid": False, "failures": ["empty_smiles"]}, True, STATUS_PB_INVALID),
        ({"pb_valid": None, "skipped": True, "failures": []}, True, STATUS_SKIPPED),
        ({"pb_valid": True, "failures": []}, False, STATUS_UNPARSEABLE),
        (None, True, STATUS_EMBED_FAILED),
        (None, False, STATUS_UNPARSEABLE),
    ],
)
def test_classify_result_covers_every_status(
    result: Dict[str, Any] | None,
    parseable: bool,
    expected: str,
) -> None:
    assert classify_result(result, parseable=parseable) == expected


def test_classify_result_fails_closed_on_a_missing_verdict() -> None:
    # A result with pb_valid missing entirely is NOT pb_valid.
    assert classify_result({"failures": ["x"]}, parseable=True) == STATUS_PB_INVALID
    # pb_valid=None but not skipped is also not a pass.
    assert (
        classify_result({"pb_valid": None, "skipped": False}, parseable=True)
        == STATUS_PB_INVALID
    )


def test_unparseable_row_is_skipped_and_never_reaches_posebusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row RDKit cannot parse must NOT be handed to PoseBusters."""
    import molmetal.scripts.wf_a155_posebusters_validity as mod

    calls: List[str] = []

    def _boom(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        calls.append("called")
        raise AssertionError("check_posebusters must not be called for an unparseable row")

    monkeypatch.setattr(
        "molmetal.validation.posebusters_runner.check_posebusters", _boom
    )
    record = mod._run_one((0, "this is not a smiles [[[", 5, "ETKDGv3", "MMFF94"))
    assert record["status"] == STATUS_UNPARSEABLE
    assert record["n_conformers"] == 0
    assert record["checks"] == {}
    assert calls == []


def test_parseable_row_does_reach_the_runner() -> None:
    """A well-formed row runs through the real PoseBusters surface.

    The assertion is deliberately about the PIPELINE (the row is parseable
    and produces a verdict) and not about the chemistry verdict, which would
    pin a PoseBusters version into the test.
    """
    import molmetal.scripts.wf_a155_posebusters_validity as mod

    record = mod._run_one((0, "CCO", 5, "ETKDGv3", "MMFF94"))
    assert record["status"] in (STATUS_PB_VALID, STATUS_PB_INVALID)
    assert record["n_conformers"] >= 1
    assert record["checks"]


# ----------------------------------------------------------------
# 3. Pool reading + selection
# ----------------------------------------------------------------
def test_read_pool_returns_the_published_pool() -> None:
    smiles, columns = read_pool(POOL)
    assert len(smiles) == len(_independent_pool_rows())
    assert columns == ["epoch", "smiles", "n_frags_raw", "largest_frag_frac"]
    assert all(isinstance(value, str) for value in smiles)


def test_select_rows_full_pool_and_deterministic_subset() -> None:
    smiles, _ = read_pool(POOL)

    full_indices, full_scope = select_rows(smiles, sample_size=None, sample_seed=0)
    assert len(full_indices) == len(smiles)
    assert full_scope["is_subset"] is False
    assert full_scope["kind"] == "full_pool"

    first, scope_a = select_rows(smiles, sample_size=50, sample_seed=7)
    second, scope_b = select_rows(smiles, sample_size=50, sample_seed=7)
    assert first == second
    assert len(first) == 50
    assert scope_a["is_subset"] is True
    assert scope_a["n_selected"] == 50
    assert scope_a["n_pool"] == len(smiles)
    assert scope_a["sampling"]["seed"] == 7
    assert "SUBSET" in scope_a["label"]
    assert scope_a["kind"] == scope_b["kind"]


# ----------------------------------------------------------------
# 4. The artifact on disk
# ----------------------------------------------------------------
def _load_artifact() -> Dict[str, Any]:
    assert ARTIFACT.is_file(), (
        f"the Wf-A155 artifact is missing: {ARTIFACT}. The script MUST be "
        "runnable end-to-end and MUST produce it."
    )
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_artifact_has_the_required_top_level_keys() -> None:
    artifact = _load_artifact()
    for key in (
        "metric",
        "tag",
        "date",
        "wave",
        "purpose",
        "generated_at_utc",
        "source_pool",
        "protocol",
        "scope",
        "tool_versions",
        "counts",
        "headline",
        "not_a_restatement_of",
        "comparability_vs_frontier_band",
        "limitations",
        "conformer_sensitivity_probe",
        "timeout_recheck",
    ):
        assert key in artifact, f"missing top-level key: {key}"
    assert artifact["metric"] == "wfa155_posebusters_validity"
    assert artifact["tag"] == "MEASURED"


def test_artifact_counts_are_internally_consistent() -> None:
    artifact = _load_artifact()
    counts = artifact["counts"]

    assert counts["n_scored"] == counts["n_pb_valid"] + counts["n_pb_invalid"]
    assert (
        counts["n_rows_in_scope"]
        == counts["n_scored"]
        + counts["n_embed_failed"]
        + counts["n_unparseable"]
        + counts["n_skipped_posebusters_missing"]
        + counts["n_timeout"]
    )
    assert counts["pb_valid_rate_strict"] == pytest.approx(
        counts["n_pb_valid"] / counts["n_rows_in_scope"]
    )
    # The repo's own pb_pass_rate() helper must produce the same number.
    assert counts["pb_valid_rate_strict_via_repo_helper"] == pytest.approx(
        counts["pb_valid_rate_strict"]
    )
    if counts["n_scored"]:
        assert counts["pb_valid_rate_scored"] == pytest.approx(
            counts["n_pb_valid"] / counts["n_scored"]
        )
    assert set(counts["status_counts"]) <= {
        STATUS_PB_VALID,
        STATUS_PB_INVALID,
        STATUS_EMBED_FAILED,
        STATUS_UNPARSEABLE,
        STATUS_SKIPPED,
        STATUS_TIMEOUT,
    }
    assert sum(counts["status_counts"].values()) == counts["n_rows_in_scope"]


def test_artifact_provenance_matches_the_pool_on_disk() -> None:
    artifact = _load_artifact()
    source = artifact["source_pool"]
    smiles, _columns = read_pool(POOL)
    assert source["sha256"] == sha256_file(POOL)
    assert source["n_data_rows"] == len(smiles)
    assert source["path"].endswith(
        "molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv"
    )
    protocol = artifact["protocol"]
    assert protocol["n_conformers"] == DEFAULT_N_CONFORMERS
    assert protocol["embed_method"] == DEFAULT_EMBED_METHOD
    assert protocol["optimize_method"] == DEFAULT_OPTIMIZE_METHOD
    assert protocol["posebusters_config"] == "mol"
    assert artifact["tool_versions"]["posebusters"] is not None
    assert artifact["tool_versions"]["rdkit"] is not None


def test_artifact_is_not_a_restatement_of_the_851_or_2505_numbers() -> None:
    artifact = _load_artifact()
    strict = artifact["counts"]["pb_valid_rate_strict"]
    scored = artifact["counts"]["pb_valid_rate_scored"]

    # Both published numbers are READ from their own artefacts -- the
    # end-to-end headline from the fragmentation audit, the single-pool
    # fragment fraction from the pool's own n_frags_raw column.
    end_to_end_median = _published_end_to_end_median()
    fragment_counts = read_pool_fragment_counts(POOL)
    single_fragment_fraction = (
        sum(1 for value in fragment_counts if value == 1) / len(fragment_counts)
    )

    # The new metric must be DISTINCT from both published numbers.
    assert not math.isclose(strict, end_to_end_median, abs_tol=1e-6)
    assert not math.isclose(scored, end_to_end_median, abs_tol=1e-6)
    assert not math.isclose(strict, single_fragment_fraction, abs_tol=1e-6)
    assert not math.isclose(scored, single_fragment_fraction, abs_tol=1e-6)

    # The artifact must SAY what it is not a restatement of.  The block has
    # exactly three entries -- the end-to-end headline, the single-pool
    # fragment count, the prior PoseBusters artifacts.  The pre-Wf-A164
    # emitter and the current one name them differently, so the test checks
    # the TOPICS by their content, not by the key spelling.
    restatements = artifact["not_a_restatement_of"]
    assert len(restatements) == 3
    for value in restatements.values():
        assert isinstance(value, str) and value.strip()
    text = " ".join(restatements.values())
    assert "cfm_fragmentation_audit.json" in text
    assert "single_molecule_rate_end_to_end" in text
    profile = _published_pool_profile()
    assert str(profile["n_single_fragment_rows"]) in text
    assert str(profile["n_data_rows"]) in text

    # ... and the numbers it quotes must be the FULL POOL's: a subset
    # artifact is explicitly "NOT the pool rate" and must not sit here.
    scope = artifact["scope"]
    assert scope["is_subset"] is False
    assert scope["n_selected"] == scope["n_pool"] == profile["n_data_rows"]


def test_artifact_carries_the_comparability_statement_and_honest_negative() -> None:
    artifact = _load_artifact()
    comparability = artifact["comparability_vs_frontier_band"]
    profile = _published_pool_profile()

    # The breaks are NOT a module-level constant.  Wf-A164 replaced the
    # COMPARABILITY_BREAKS tuple with build_comparability_breaks(profile),
    # because the breaks embed the pool's OWN row count, epoch structure
    # and path -- a constant would be FALSE for every pool but one.  The
    # test therefore asks the emitter's own builder what the breaks for
    # THIS pool are and checks the artifact against that answer.
    live_breaks = build_comparability_breaks(profile)
    assert live_breaks, "the emitter must produce at least one comparability break"
    assert len(comparability["breaks"]) == len(live_breaks)
    assert comparability["n_independent_comparability_breaks"] == len(live_breaks)

    # Same ids, same axes, same per-break shape.
    #
    # Wf-A182: the drift is GONE.  The published artifact used to be
    # produced by the pre-Wf-A164 emitter, so its break TEXT still
    # differed from the live builder's -- the pool phrase was hand-written
    # rather than computed, and break 3 carried the Wf-A176/Wf-A177
    # correction of the POST-PROCESSING claim.  Wf-A182 republished the
    # break block FROM the emitter's own builder (the old renderings are
    # preserved verbatim in
    # `comparability_vs_frontier_band.superseded_reading_2026_11_01_wfa182`),
    # so the artifact and the producer now agree TEXT-FOR-TEXT and the
    # structural assertions below are joined by an exact equality.
    assert comparability["breaks"] == [dict(entry) for entry in live_breaks]
    assert [entry["id"] for entry in comparability["breaks"]] == [
        entry["id"] for entry in live_breaks
    ]
    assert {entry["axis"] for entry in comparability["breaks"]} == {
        entry["axis"] for entry in live_breaks
    }
    for entry in comparability["breaks"]:
        assert set(entry) == {"id", "axis", "frontier", "ours", "consequence"}
        for key in ("axis", "frontier", "ours", "consequence"):
            assert isinstance(entry[key], str) and entry[key].strip(), (
                f"comparability break {entry.get('axis')} has an empty {key}"
            )

    # The breaks describe THIS pool: the DENOMINATOR break names the pool's
    # own path and row count, both read from the pool on disk.
    denominator = next(
        entry for entry in comparability["breaks"] if entry["axis"] == "DENOMINATOR"
    )
    assert profile["path"] in denominator["ours"]
    assert str(profile["n_data_rows"]) in denominator["ours"]

    # The artifact's own statement must name the same axes as the breaks.
    statement = comparability["statement"].lower()
    for entry in comparability["breaks"]:
        assert entry["axis"].lower() in statement

    limitations = " ".join(artifact["limitations"]).lower()
    # The metal-coordination honest negative MUST be present.
    assert "metal-coordination-sphere" in limitations
    # The hydrogens break MUST be present.
    assert "hydrogens" in limitations
    # Both denominators must be documented.
    assert "denominator" in limitations


def test_comparability_breaks_are_a_function_of_the_pool_profile() -> None:
    """The builder must PRODUCE the breaks; it cannot return a constant.

    This is the regression guard for the Wf-A164 defect class: a
    pool-specific claim baked into a ``tag: MEASURED`` artefact.  If the
    breaks were a constant again, a run against a different pool would
    publish that pool's rows under the default pool's numbers -- so the
    builder must change its output when the profile changes.
    """
    profile = _published_pool_profile()
    breaks = build_comparability_breaks(profile)

    # A different profile MUST give different text: the pool's path and row
    # count are embedded in the breaks.
    other = dict(profile)
    other["path"] = "some/other/pool.csv"
    other["n_data_rows"] = profile["n_data_rows"] + 1
    other_breaks = build_comparability_breaks(other)
    assert [entry["id"] for entry in other_breaks] == [
        entry["id"] for entry in breaks
    ]
    denominator = next(
        entry for entry in breaks if entry["axis"] == "DENOMINATOR"
    )
    other_denominator = next(
        entry for entry in other_breaks if entry["axis"] == "DENOMINATOR"
    )
    assert denominator["ours"] != other_denominator["ours"]
    assert other["path"] in other_denominator["ours"]
    assert profile["path"] not in other_denominator["ours"]


def test_artifact_reports_the_metal_forcefield_shortfall() -> None:
    """MMFF94 cannot parameterise most metal centres -- that must be visible.

    The PoseBusters ``internal_energy`` check comes back missing (None) for
    metal-containing molecules, and the runner fails closed, so those rows
    land in ``per_check_missing_counts``.
    """
    artifact = _load_artifact()
    counts = artifact["counts"]
    missing = counts["per_check_missing_counts"]
    assert "internal_energy" in missing
    assert missing["internal_energy"] > 0
    # And the missing count can never exceed the scored population.
    assert missing["internal_energy"] <= counts["n_scored"]


def test_artifact_conformer_probe_is_well_formed() -> None:
    artifact = _load_artifact()
    probe = artifact["conformer_sensitivity_probe"]
    assert probe["n_rows_probed"] > 0
    assert probe["n_comparable"] <= probe["n_rows_probed"]
    assert probe["n_agree"] + probe["n_disagree"] == probe["n_comparable"]
    assert 0.0 <= probe["agreement_rate"] <= 1.0


def test_artifact_censoring_is_measured_not_only_asserted() -> None:
    """The per-row wall-clock cap CENSORS rows; the artifact must quantify it.

    A timeout is a censoring event, not a chemistry verdict.  The artifact
    therefore carries a ``timeout_recheck`` block that re-runs the censored
    rows under a longer cap, so the direction and size of the timeout bias
    is MEASURED rather than described in prose.
    """
    artifact = _load_artifact()
    counts = artifact["counts"]
    recheck = artifact["timeout_recheck"]

    assert counts["n_timeout"] >= 0
    # The protocol must document the cap that produced the censoring.
    assert artifact["protocol"]["row_timeout_s"] > 0

    if counts["n_timeout"] == 0:
        # No censoring happened -- the block is then legitimately empty.
        assert recheck is None or recheck["n_rechecked"] == 0
        return

    assert recheck is not None, (
        "the run produced timeouts, so the censoring MUST have been measured"
    )
    assert recheck["n_timeouts_in_main_pass"] == counts["n_timeout"]
    assert recheck["n_rechecked"] > 0
    assert recheck["n_rechecked"] <= recheck["recheck_max_rows"]
    assert recheck["recheck_cap_s"] > artifact["protocol"]["row_timeout_s"]
    # Every re-checked row must carry a verdict we can reason about.
    assert len(recheck["rows"]) == recheck["n_rechecked"]
    for row in recheck["rows"]:
        assert row["status"] in {
            STATUS_PB_VALID,
            STATUS_PB_INVALID,
            STATUS_EMBED_FAILED,
            STATUS_TIMEOUT,
        }
        assert "smiles" in row and "elapsed_s" in row
    counted = (
        recheck["n_that_would_be_pb_valid"]
        + recheck["n_that_would_be_pb_invalid"]
        + recheck["n_still_embed_failed"]
        + recheck["n_still_timed_out"]
    )
    assert counted <= recheck["n_rechecked"]
    # Every re-checked row is accounted for by the status histogram.
    assert sum(recheck["recheck_status_counts"].values()) == recheck["n_rechecked"]


def test_artifact_pool_provenance_is_measured_not_asserted() -> None:
    """The pool's structure must be measured from its own columns.

    MEASURED 2026-11-01: the pool is THREE epoch-wise decodes of ONE
    checkpoint (epoch 0/1/2, 704 rows each), not a single decode. An
    artifact that called it "one decode" would be wrong, and the epoch
    breakdown makes the structure visible in the data.
    """
    artifact = _load_artifact()
    source = artifact["source_pool"]
    profile = _published_pool_profile()

    # The epoch structure is the pool's OWN column, not a literal.
    assert source["n_distinct_epochs"] == profile["n_distinct_epochs"]
    assert (
        source["epoch_column_distribution"]
        == profile["epoch_column_distribution"]
    )
    # The structure note must carry the pool's own measured numbers.
    note = source["pool_structure_note"]
    assert str(profile["n_data_rows"]) in note
    assert str(profile["n_rows_per_epoch_max"]) in note
    counts = artifact["counts"]
    assert set(counts["by_pool_epoch"]) >= set(profile["epoch_column_distribution"])
    assert sum(
        group["n_rows"] for key, group in counts["by_pool_epoch"].items()
        if not key.startswith("_")
    ) == counts["n_rows_in_scope"]


def test_artifact_timeout_cap_is_not_silent() -> None:
    """The artifact must say out loud that the cap censors rows."""
    artifact = _load_artifact()
    limitations = " ".join(artifact["limitations"]).lower()
    assert "censor" in limitations
    assert "row_timeout_s" in limitations or "wall-clock cap" in limitations
    protocol = artifact["protocol"]
    assert "row_timeout_policy" in protocol
    assert protocol["device"] == "CPU"


def test_conformer_probe_summary_maths() -> None:
    probe_rows = [
        {"row_index": 0, "smiles": "CCO", "status_n": STATUS_PB_VALID, "status_1": STATUS_PB_VALID},
        {"row_index": 1, "smiles": "CC", "status_n": STATUS_PB_INVALID, "status_1": STATUS_PB_INVALID},
        {"row_index": 2, "smiles": "CN", "status_n": STATUS_PB_VALID, "status_1": STATUS_PB_INVALID},
        # Not comparable: one arm could not embed.
        {"row_index": 3, "smiles": "C[Pt]", "status_n": STATUS_EMBED_FAILED, "status_1": STATUS_PB_VALID},
    ]
    summary = summarize_conformer_probe(probe_rows)
    assert summary["n_rows_probed"] == 4
    assert summary["n_comparable"] == 3
    assert summary["n_agree"] == 2
    assert summary["n_disagree"] == 1
    assert summary["agreement_rate"] == pytest.approx(2 / 3)
    assert summary["n_embed_failed_on_either_arm"] == 1
    assert summary["disagreements"][0]["row_index"] == 2


# ----------------------------------------------------------------
# 5. The script: importable, CPU-only, no forbidden paths
# ----------------------------------------------------------------
def test_script_is_importable_and_exposes_its_entry_points() -> None:
    import molmetal.scripts.wf_a155_posebusters_validity as mod

    assert callable(mod.main)
    assert callable(mod.run)
    assert callable(mod.aggregate)
    assert callable(mod.classify_result)


def test_script_never_imports_torch_and_never_touches_cuda() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    imported: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
    for name in imported:
        root = name.split(".")[0]
        assert root != "torch", f"the script must not import torch: {name}"

    # No CUDA entry points, by text (the docstring may mention the ban).
    assert "torch.cuda" not in source
    assert "cuda.is_available" not in source
    assert "device=\"cuda\"" not in source


def test_importing_the_script_does_not_pull_in_torch() -> None:
    before = set(sys.modules)
    import importlib

    importlib.reload(
        importlib.import_module("molmetal.scripts.wf_a155_posebusters_validity")
    )
    new_torch_modules = {
        name for name in set(sys.modules) - before if name.split(".")[0] == "torch"
    }
    assert not new_torch_modules, (
        f"importing the script pulled in torch modules: {sorted(new_torch_modules)}"
    )


def test_script_does_not_reference_forbidden_paths() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    # paper/ is off-limits (AGENTS.md 5.3 / TODO-84).
    assert "paper/" not in source
    # Checkpoints and results/ must never be touched.
    assert "molmetal/checkpoints" not in source
    assert "results/" not in source


def test_default_output_path_is_under_metrics_by_metric() -> None:
    import molmetal.scripts.wf_a155_posebusters_validity as mod

    out = Path(mod.DEFAULT_OUT)
    assert out.name == "wfa155_posebusters_validity.json"
    assert out.parent.name == "by_metric"
    assert out.parent.parent.name == "metrics"
    assert out == ARTIFACT


def test_pool_default_path_is_the_published_pool() -> None:
    import molmetal.scripts.wf_a155_posebusters_validity as mod

    assert Path(mod.DEFAULT_POOL) == POOL


# ----------------------------------------------------------------
# 6. Wf-A182 -- the residual-defect repair (D1..D5 + C1)
# ----------------------------------------------------------------
# The adversarial verify of wave Wf-A181 found five residual defects in
# this artifact and its producer, plus one claim the artifact could not
# support.  Two of them were MATERIAL:
#
#   D1 -- the three longest-pool-SMILES call seconds were a module
#         CONSTANT, so the artifact certified a measurement for whatever
#         pool it named (the Wf-A164 defect class);
#   D2 -- `published_cross_check` was published as `null` while the
#         producer computes a populated block, and the `null` was what
#         made the narrative paths vacuously satisfied.
#
# The rest (D3 a derived number with no label, D4 an unscoped field
# name, D5 an undisclosed value, C1 a blanket RAW claim) are MINOR, but
# each is a claim a `tag: MEASURED` artifact must not make without the
# evidence for it.
#
# Every assertion below is written so that it can FAIL: the negative
# controls strip the binding / the backing out of a COPY of the artifact
# and show that the corresponding assertion then fails.  A test that
# cannot fail is not a test.


def _delete_leaf(node: Dict[str, Any], leaf_path: str) -> None:
    """Delete the leaf at a dotted/indexed path such as ``a.b[0].c``."""
    tokens = re.findall(r"[^\.\[\]]+|\[\d+\]", leaf_path)
    current: Any = node
    for token in tokens[:-1]:
        if token.startswith("["):
            current = current[int(token[1:-1])]
        else:
            current = current[token]
    last = tokens[-1]
    if last.startswith("["):
        del current[int(last[1:-1])]
    else:
        del current[last]


def _d1_binding_is_sound(artifact: Dict[str, Any]) -> None:
    """The D1 contract, as assertions (so a negative control can break it).

    The three longest-pool-SMILES call seconds are a MEASUREMENT of ONE
    pool.  Publishing them without naming that pool certifies a number
    for every other pool, which is the Wf-A164 defect class.  The binding
    must therefore name the pool (by path AND by content hash), the
    measurement record the numbers came from, and that record must exist
    on disk with the SAME rationale the artifact renders.
    """
    protocol = artifact["protocol"]
    seconds = protocol["longest_pool_smiles_call_seconds"]
    binding = protocol.get("longest_pool_smiles_call_seconds_provenance")

    assert isinstance(binding, dict) and binding, (
        "the longest-pool call seconds must carry a provenance binding"
    )
    assert binding["applicable_to_this_pool"] is True, (
        "the binding must certify THIS pool"
    )
    assert binding["measured_on_pool_path"] == artifact["source_pool"]["path"], (
        "the binding must name the pool the artifact measures"
    )
    assert binding["measured_on_pool_sha256"] == artifact["source_pool"]["sha256"], (
        "the binding must pin the pool by CONTENT, not by path alone"
    )
    assert binding["seconds"] == seconds, (
        "the binding must carry the numbers it certifies"
    )
    assert binding["measurement_date"], "the binding must carry a date"
    assert binding["measurement_tool"], "the binding must carry the tool"

    record_path = REPO_ROOT / binding["measurement_record_path"]
    assert record_path.is_file(), (
        f"the measurement record named by the binding does not exist: "
        f"{record_path}"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["source_pool"]["path"] == artifact["source_pool"]["path"], (
        "the measurement record must be about THIS pool"
    )
    assert record["source_pool"]["sha256"] == artifact["source_pool"]["sha256"]
    # The rationale the artifact renders must be the record's own text --
    # verbatim, not a paraphrase.  The record field is a dotted path.
    record_node: Any = record
    for part in binding["measurement_record_field"].split("."):
        record_node = record_node[part]
    assert record_node == protocol["row_timeout_rationale"], (
        "the rendered rationale must be the measurement record's own field"
    )


def test_d1_longest_pool_seconds_are_bound_to_the_pool_they_were_measured_on() -> None:
    artifact = _load_artifact()
    _d1_binding_is_sound(artifact)
    assert artifact["protocol"]["longest_pool_smiles_call_seconds"] == list(
        LONGEST_POOL_SMILES_CALL_SECONDS
    )


def test_d1_negative_control__the_binding_is_required() -> None:
    """The D1 test MUST FAIL when the pool binding is removed.

    This is the proof that the D1 assertion is not vacuous: it is the
    SAME assertion body the positive test runs, against an artifact whose
    binding has been deleted.
    """
    stripped = copy.deepcopy(_load_artifact())
    stripped["protocol"].pop("longest_pool_smiles_call_seconds_provenance")
    with pytest.raises(AssertionError):
        _d1_binding_is_sound(stripped)

    # And the producer refuses to certify the numbers for a FOREIGN pool.
    profile = _published_pool_profile()
    foreign = longest_pool_measurement_binding(
        pool_path="molmetal/reports/some_other_pool/generated_smiles.csv",
        pool_sha256=profile["sha256"],
    )
    assert foreign["applicable_to_this_pool"] is False
    assert "seconds" not in foreign, (
        "the numbers must not be emitted for a pool they were not measured on"
    )


def test_d2_published_cross_check_is_the_producer_computed_block() -> None:
    artifact = _load_artifact()
    profile = _published_pool_profile()
    block = artifact["published_cross_check"]

    assert isinstance(block, dict) and block, (
        "published_cross_check must not be null: the producer computes a block"
    )
    assert block == _published_cross_check(
        profile,
        profile["single_fragment_fraction"],
        profile["n_single_fragment_rows"],
    ), "the artifact must carry the producer's COMPUTED block"

    assert block["n_matches"] == len(block["matches"]) >= 1
    match = block["matches"][0]
    assert match["published_rate"] == match["this_run_rate"], (
        "the published rate for THIS pool must be reproduced bit-for-bit"
    )
    assert match["published_n_single"] == match["this_run_n_single"]
    assert match["rate_identical"] is True

    # The shape matches the sibling artifact's block (the same eight keys).
    sibling = json.loads(
        (
            REPO_ROOT / "metrics" / "by_metric" / "wfa155_ligand_connected_rate.json"
        ).read_text(encoding="utf-8")
    )
    assert set(block) == set(sibling["published_cross_check"])

    # Wf-A164's defect class: a candidate that measures ANOTHER pool must
    # not carry that pool's numbers.
    for entry in block["published_artifacts_considered"]:
        if not entry.get("names_this_pool"):
            assert entry["published_rate"] is None
            assert entry["published_n_single"] is None
            assert entry["source_pool"] is None

    # The `null` that was published before is preserved verbatim.
    superseded = artifact["published_cross_check_superseded_2026_11_01"]
    assert superseded["superseded"]["published_cross_check"] is None
    assert superseded["wave"] == "Wf-A182"


def test_d3_end_to_end_median_pct_is_labelled_as_derived() -> None:
    artifact = _load_artifact()
    headline = artifact["end_to_end_headline"]
    derivation = headline.get("median_pct_derivation")
    assert isinstance(derivation, str) and derivation.strip(), (
        "median_pct is a unit conversion, not a value read from disk"
    )
    assert "DERIVED" in derivation
    assert "`median`" in derivation
    assert headline["median_pct"] == headline["median"] * 100
    assert "end_to_end_headline.median_pct_derivation" in NARRATIVE_PATHS


def test_d4_prior_posebusters_sample_size_field_carries_its_scope() -> None:
    artifact = _load_artifact()
    block = artifact["prior_posebusters_artifacts_measured"]

    assert "largest_n_samples" not in block, (
        "the unscoped name must be gone -- it read as a repo-wide maximum"
    )
    scoped = block["largest_n_samples_among_posebusters_candidates"]
    assert isinstance(scoped, int)
    assert "repo-wide" in block["largest_n_samples_scope"]

    # The old name and its value survive verbatim in the dated record.
    superseded = block["superseded_field_name_2026_11_01"]
    assert superseded["field"] == "largest_n_samples"
    assert superseded["value"] == scoped

    # And the scope claim is REAL: another metrics/ artifact reports a
    # LARGER sample size, so the unscoped name was an overstatement.
    other = json.loads(
        (
            REPO_ROOT / "metrics" / "by_metric" / "metalcytotox_combined_r2.json"
        ).read_text(encoding="utf-8")
    )
    repo_wide = other["per_method"]["wf77_p2_astr_baseline/metalcytotox_5metal"][
        "n_samples"
    ]
    assert repo_wide > scoped


def test_d5_wfa181_record_discloses_that_the_cross_check_was_null() -> None:
    artifact = _load_artifact()
    record = artifact["wfa181_narrative_backing_repair_2026_11_01"]

    # APPEND-ONLY: the original list still says what Wf-A181 added.
    assert "published_cross_check" in record["what_was_added"]

    disclosure = record["disclosure_2026_11_01_wfa182"]
    assert disclosure["what"].count("null") >= 1, (
        "the disclosure must say the published value was null"
    )
    assert "published_cross_check" in disclosure["what"]
    assert disclosure["why"].strip()
    assert disclosure["wave"] == "Wf-A182"


def test_c1_post_processing_break_does_not_assert_the_band_is_raw() -> None:
    artifact = _load_artifact()
    band = artifact["comparability_vs_frontier_band"]
    post = next(
        entry for entry in band["breaks"] if entry["axis"] == "POST-PROCESSING"
    )
    frontier = post["frontier"]
    consequence = post["consequence"]

    # The blanket claim is GONE -- it was FALSE under the second of the
    # two provenance readings the record itself holds.
    assert "the published band's two endpoints are RAW" not in frontier
    assert "both arms are RAW" not in consequence

    # The claim is now SCOPED, and the open question is stated.
    assert "band_provenance" in frontier
    assert "NOT settled on disk" in frontier

    # The invariants that hold under BOTH readings survive.
    lowered = consequence.lower()
    assert "external" in lowered
    assert "homogeneous" in lowered
    assert "not on our scale" in lowered

    # The superseded rendering is preserved VERBATIM (AGENTS.md 5.5 #6).
    superseded = band["superseded_reading_2026_11_01_wfa182"]
    texts = {
        (entry["id"], entry["field"]): entry["text"]
        for entry in superseded["superseded"]
    }
    assert "the published band's two endpoints are RAW" in texts[(3, "frontier")]
    assert "both arms are RAW" in texts[(3, "consequence")]
    assert superseded["wave"] == "Wf-A182"
    assert "§5.5" in superseded["rule"]


def test_artifact_agrees_with_the_emitter_on_every_touched_block() -> None:
    """Producer <-> artefact parity, compared programmatically."""
    artifact = _load_artifact()
    profile = _published_pool_profile()

    assert artifact["protocol"]["longest_pool_smiles_call_seconds"] == list(
        LONGEST_POOL_SMILES_CALL_SECONDS
    )
    assert artifact["protocol"]["longest_pool_smiles_call_seconds_provenance"] == (
        longest_pool_measurement_binding(
            pool_path=profile["path"], pool_sha256=profile["sha256"]
        )
    )
    assert artifact["published_cross_check"] == _published_cross_check(
        profile,
        profile["single_fragment_fraction"],
        profile["n_single_fragment_rows"],
    )
    assert artifact["end_to_end_headline"] == _end_to_end_headline()
    assert artifact["prior_posebusters_artifacts_measured"] == (
        _prior_posebusters_artifacts()
    )
    assert artifact["comparability_vs_frontier_band"]["breaks"] == [
        dict(entry) for entry in build_comparability_breaks(profile)
    ]


def test_stored_guard_block_is_current_and_passes() -> None:
    artifact = _load_artifact()
    stored = artifact["narrative_self_consistency"]

    assert stored["pass"] is True
    assert stored["unbacked_numbers"] == {}
    assert stored["stale_paths"] == {}
    assert stored["unresolved_narrative_paths"] == []
    assert stored["unclassified_leaves"] == {}

    # The stored block IS what the producer computes for this artifact.
    current = {
        key: value
        for key, value in stored.items()
        if key != "superseded_2026_11_01_wfa182"
    }
    assert current == _narrative_self_consistency(artifact), (
        "the stored guard block is stale -- recompute it"
    )

    # The pre-Wf-A182 block is preserved verbatim.
    old = stored["superseded_2026_11_01_wfa182"]
    assert old["pass"] is True
    assert old["n_machine_leaves"] < stored["n_machine_leaves"]


def test_guard_still_bites__negative_control() -> None:
    """The guard must still FAIL when a backing is taken away.

    Two independent negative controls:
      * the PATH guard -- a designated narrative path that stops
        resolving is a hard failure (the D3 label cannot be dropped);
      * the NUMBER guard -- a prose number that loses its only machine
        backing is reported as unbacked.
    """
    # (a) the PATH guard.
    broken = copy.deepcopy(_load_artifact())
    broken["end_to_end_headline"].pop("median_pct_derivation")
    verdict = _narrative_self_consistency(broken)
    assert verdict["pass"] is False
    assert "end_to_end_headline.median_pct_derivation" in (
        verdict["unresolved_narrative_paths"]
    )

    # (b) the NUMBER guard.  Find a prose number whose ONLY backing is a
    # single machine leaf, delete that leaf, and the number becomes
    # unbacked.  The token is DISCOVERED, never typed in.
    artifact = _load_artifact()
    leaves = _leaf_map(artifact)
    narrative_leaves: set = set()
    for path in NARRATIVE_PATHS:
        narrative_leaves.update(_resolve(artifact, path))
    machine_strings = {
        path: value
        for path, value in leaves.items()
        if path not in narrative_leaves and isinstance(value, str)
    }
    machine_numbers = {
        path: float(value)
        for path, value in leaves.items()
        if path not in narrative_leaves
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    uniquely_backed = None
    for path in sorted(narrative_leaves):
        text = leaves.get(path)
        if not isinstance(text, str):
            continue
        for token in set(_NUM_RE.findall(_mask_identifier_tokens(text))):
            norm = token.replace(",", "")
            try:
                value = float(norm)
            except ValueError:
                continue
            decimals = len(norm.split(".")[1]) if "." in norm else 0
            tolerance = 0.5 * (10.0 ** (-decimals)) if decimals else 0.0
            backers = {
                candidate
                for candidate, text_value in machine_strings.items()
                if _standalone_in_any(token, [text_value])
            }
            backers |= {
                candidate
                for candidate, number in machine_numbers.items()
                if abs(number - value) <= tolerance + 1e-12
            }
            if len(backers) == 1:
                uniquely_backed = (token, next(iter(backers)))
                break
        if uniquely_backed:
            break

    assert uniquely_backed is not None, (
        "no prose number has a unique machine backing -- the negative "
        "control cannot be built"
    )
    token, leaf_path = uniquely_backed
    broken = copy.deepcopy(artifact)
    _delete_leaf(broken, leaf_path)
    verdict = _narrative_self_consistency(broken)
    assert verdict["pass"] is False, (
        f"removing the only backing of {token!r} must fail the guard"
    )
    assert token in verdict["unbacked_numbers"]
